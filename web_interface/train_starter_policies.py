"""Train small real Robosuite policies from demonstrations and DAgger labels.

Run with the isolated robotics runtime, from web_interface:
    .venv/bin/python -E train_starter_policies.py --tasks Lift Stack PickPlaceCan

Only the best development checkpoint for each task is exported. Final evaluation
seeds are never used for fitting or checkpoint selection. No images or external
models are downloaded. Exported TorchScript contains only the learned network,
normalization, a resettable episode clock, and the fixed OSC action layout.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time


APP = Path(__file__).resolve().parent
DEFAULT_OUTPUT = APP / "runs/robotics/starter-policies"
TASKS = ("Lift", "Stack", "PickPlaceCan")


def validate_seed_splits(training, development, final):
    parts = [set(training), set(development), set(final)]
    if not all(parts) or any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Training, development, and final seeds must be nonempty and disjoint.")


def build_policy(mean, std, horizon):
    """Create a time-conditioned MLP; numeric imports follow runtime bootstrap."""
    import torch

    class TimeConditionedPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clamp_min(.01))
            self.register_buffer("frequencies", torch.tensor([1., 2., 4., 8.]) * torch.pi)
            self.horizon = float(horizon)
            self.step = 0
            self.network = torch.nn.Sequential(
                torch.nn.Linear(len(mean) + 9, 128), torch.nn.Tanh(),
                torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 4),
            )

        def logits(self, state, steps):
            normalized = ((state - self.mean) / self.std).clamp(-10., 10.)
            clock = steps.reshape(-1, 1) / self.horizon
            phase = clock * self.frequencies
            features = torch.cat((normalized, clock, torch.sin(phase), torch.cos(phase)), dim=1)
            return self.network(features)

        def forward(self, state):
            steps = torch.full((state.size(0),), float(self.step), dtype=state.dtype, device=state.device)
            values = self.logits(state, steps)
            self.step += 1
            translation = torch.tanh(values[:, :3])
            rotation = torch.zeros_like(translation)
            gripper = torch.where(values[:, 3:] >= 0, torch.ones_like(values[:, 3:]), -torch.ones_like(values[:, 3:]))
            return torch.cat((translation, rotation, gripper), dim=1)

        @torch.jit.export
        def reset(self):
            self.step = 0

    return TimeConditionedPolicy()


def export_policy(policy, path, metadata):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy.eval()
    policy.reset()
    serialized = json.dumps(metadata, indent=2, allow_nan=False)
    torch.jit.save(torch.jit.script(policy), str(path), _extra_files={"metadata.json": serialized})
    Path(str(path) + ".metadata.json").write_text(serialized + "\n")


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _log(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


@contextmanager
def staged_task(directory):
    """Keep the previously evaluated task intact until its replacement is ready.

    The catalog reader verifies the checkpoint digest, so the brief interval
    between completed-file replacement and atomic catalog publication cannot
    advertise the previous checkpoint's metrics for the new weights.
    """
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".pending-{directory.name}-", dir=directory.parent) as temporary:
        pending = Path(temporary)
        yield pending
        directory.mkdir(parents=True, exist_ok=True)
        for source in pending.iterdir():
            if source.is_file():
                source.replace(directory / source.name)


class LiftTeacher:
    horizon = 180
    state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "cube_pos", "cube_quat"]

    def reset(self, observation):
        pass

    def action(self, observation, step):
        import numpy as np
        target = np.array(observation["cube_pos"], copy=True)
        phase = sum(step >= boundary for boundary in (25, 50, 65))
        target[2] += (.10, .008, .008, .22)[phase]
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip((target - observation["robot0_eef_pos"]) * 12., -1., 1.)
        action[-1] = -1. if phase < 2 else 1.
        return action


def _teacher(task, env):
    if task == "Lift":
        return LiftTeacher()
    from starter_teachers import make_teacher
    return make_teacher(task, env)


def _environment(task, horizon):
    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
    controller = refactor_composite_controller_config(
        suite.load_part_controller_config(default_controller="OSC_POSE"), "Panda", ["right"])
    return suite.make(task, robots=["Panda"], controller_configs=controller, control_freq=20,
                      horizon=horizon, hard_reset=True, ignore_done=False, reward_shaping=True,
                      has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                      use_object_obs=True, renderer="mujoco")


def _state(observation, keys):
    import numpy as np
    return np.concatenate([np.asarray(observation[key]).reshape(-1) for key in keys]).astype(np.float32)


def rollout(env, task, seed, *, policy=None, collect=False, teacher_probability=0., noise=0., variation=None):
    """Run physics to native success or the horizon, optionally collect labels."""
    import numpy as np
    import torch
    from robotics_env import apply_variation, delayed_action, perturb_observation, seed_episode
    teacher = _teacher(task, env)
    seed_episode(seed, env)
    observation = env.reset()
    variation = variation or {}
    if variation:
        apply_variation(env, variation)
        env._obs_cache = {}
        observation = env._get_observations(force_update=True)
    teacher.reset(observation)
    if policy is not None:
        policy.reset()
    rng = np.random.default_rng(seed + 90000)
    queue, states, labels, times = [], [], [], []
    success, total_reward = False, 0.
    started = time.monotonic()
    for step in range(teacher.horizon):
        observed = perturb_observation(observation, variation.get("observation_noise", 0), rng)
        state = _state(observed, teacher.state_keys)
        label = teacher.action(observation, step) if collect or policy is None else None
        if collect:
            states.append(state)
            labels.append(label)
            times.append(step)
        if policy is None:
            action = label.copy()
        else:
            if hasattr(policy, "act"):
                learned = policy.act(observed)
            else:
                with torch.inference_mode():
                    learned = policy(torch.from_numpy(state).unsqueeze(0)).numpy()[0]
            action = label.copy() if collect and rng.random() < teacher_probability else learned.copy()
        if noise:
            action[:3] += rng.normal(0, noise, 3)
            action[:3] = np.clip(action[:3], -1, 1)
        action = delayed_action(action, int(variation.get("action_delay", 0)), queue)
        observation, reward, done, _ = env.step(action)
        total_reward += float(reward)
        success = bool(env._check_success())
        if success or done:
            break
    record = {"seed": seed, "success": success, "steps": step + 1,
              "return": total_reward, "seconds": round(time.monotonic() - started, 4)}
    data = (states, labels, times) if collect else None
    return record, data


def _summary(records):
    successes = sum(row["success"] for row in records)
    return {"successes": successes, "episodes": len(records), "success_rate": successes / len(records),
            "mean_steps": sum(row["steps"] for row in records) / len(records),
            "mean_return": sum(row["return"] for row in records) / len(records), "records": records}


def evaluate(env, task, policy, seeds, variation=None):
    return _summary([rollout(env, task, seed, policy=policy, variation=variation)[0] for seed in seeds])


def fit_policy(policy, dataset, updates, seed):
    import numpy as np
    import torch
    torch.manual_seed(seed)
    states = torch.as_tensor(np.asarray(dataset[0]), dtype=torch.float32)
    actions = torch.as_tensor(np.asarray(dataset[1]), dtype=torch.float32)
    steps = torch.as_tensor(dataset[2], dtype=torch.float32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    history = []
    policy.train()
    for update in range(updates):
        indices = torch.randint(0, len(states), (min(512, len(states)),))
        values = policy.logits(states[indices], steps[indices])
        movement = torch.nn.functional.mse_loss(torch.tanh(values[:, :3]), actions[indices, :3])
        grip = torch.nn.functional.binary_cross_entropy_with_logits(values[:, 3], (actions[indices, -1] + 1) / 2)
        loss = movement + .15 * grip
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update % 500 == 0 or update == updates - 1:
            history.append({"update": update + 1, "loss": float(loss.detach()),
                            "translation_mse": float(movement.detach()), "gripper_bce": float(grip.detach())})
    policy.eval()
    return history


def train_task(task, args):
    with staged_task(args.output / task.lower()) as output:
        return _train_task(task, args, output)


def _train_task(task, args, output):
    import numpy as np
    import torch
    import robosuite
    output.mkdir(parents=True, exist_ok=True)
    development_seeds = list(range(1000, 1000 + args.development_episodes))
    final_seeds = list(range(2000, 2000 + args.final_episodes))
    training_seeds = list(range(args.demonstrations + args.dagger_rounds * args.dagger_episodes))
    validate_seed_splits(training_seeds, development_seeds, final_seeds)
    horizon = {"Lift": 180, "Stack": 220, "PickPlaceCan": 260}[task]
    env = _environment(task, horizon)
    dataset = ([], [], [])
    collection, candidates = [], []
    started = time.monotonic()
    try:
        for seed in range(args.demonstrations):
            record, data = rollout(env, task, seed, collect=True, noise=.015)
            collection.append({**record, "round": 0, "controller": "teacher"})
            for destination, source in zip(dataset, data):
                destination.extend(source)
            if (seed + 1) % 16 == 0 or seed + 1 == args.demonstrations:
                _log(task=task, phase="demonstrations", episodes=seed + 1,
                     successes=sum(item["success"] for item in collection), transitions=len(dataset[0]))
        states = np.asarray(dataset[0])
        torch.manual_seed(args.training_seed)
        policy = build_policy(states.mean(0), states.std(0), horizon)
        best_rank, best_weights, best_index = None, None, None
        for round_index in range(args.dagger_rounds + 1):
            if round_index:
                first_seed = args.demonstrations + (round_index - 1) * args.dagger_episodes
                for seed in range(first_seed, first_seed + args.dagger_episodes):
                    record, data = rollout(env, task, seed, policy=policy, collect=True,
                                           teacher_probability=.25 if round_index == 1 else 0., noise=.01)
                    collection.append({**record, "round": round_index, "controller": "dagger_mixture"})
                    for destination, source in zip(dataset, data):
                        destination.extend(source)
                _log(task=task, phase="dagger_collected", round=round_index, transitions=len(dataset[0]))
            history = fit_policy(policy, dataset, args.updates, args.training_seed + round_index)
            development = evaluate(env, task, policy, development_seeds)
            candidate = {"round": round_index, "transitions": len(dataset[0]), "updates": args.updates,
                         "loss_history": history, "development": development}
            candidates.append(candidate)
            rank = (development["successes"], development["mean_return"], -development["mean_steps"])
            if best_rank is None or rank > best_rank:
                best_rank, best_weights, best_index = rank, copy.deepcopy(policy.state_dict()), round_index
            _log(task=task, phase="development", round=round_index, successes=development["successes"],
                 episodes=development["episodes"], loss=history[-1]["loss"], best_round=best_index)
            _json(output / "training-progress.json", {"candidates": candidates, "collection": collection})
            if development["successes"] == len(development_seeds):
                break
        policy.load_state_dict(best_weights)
        teacher = _teacher(task, env)
        environment = {"task": task, "robots": ["Panda"], "controller": "OSC_POSE", "control_freq": 20,
                       "horizon": horizon, "env_version": robosuite.__version__}
        metadata = {"type": "time_conditioned_behavior_cloning_mlp", "state_keys": teacher.state_keys,
                    "input_features": {"observation.state": {"type": "STATE", "shape": [states.shape[1]]}},
                    "output_features": {"action": {"type": "ACTION", "shape": [7]}},
                    "environment": environment, "torchscript_input": "state",
                    "normalization": {"embedded": True, "method": "training_state_mean_std", "std_floor": .01},
                    "notes": ["Learned translation and binary gripper, zero OSC rotation delta.",
                              "Episode clock is reset by the exported reset() method.",
                              "No scripted demonstrator or fallback is embedded in the policy."]}
        checkpoint = output / "best.pt"
        export_policy(policy, checkpoint, metadata)
        from robotics_policies import load_policy
        adapter = load_policy({"format": "torchscript", "source": "local", "name": str(checkpoint)}, env)
        try:
            nominal = evaluate(env, task, adapter, final_seeds)
            _log(task=task, phase="final_nominal", successes=nominal["successes"], episodes=nominal["episodes"])
            variation_seeds = final_seeds[:args.variation_episodes]
            conditions = {"mass_1_5": {"mass_scale": 1.5}, "friction_0_6": {"friction_scale": .6},
                          "object_x_plus_0_03": {"object_x_offset": .03}, "action_delay_1": {"action_delay": 1}}
            variations = {}
            if variation_seeds:
                for name, values in conditions.items():
                    variations[name] = {"values": values, **evaluate(env, task, adapter, variation_seeds, values)}
                    _log(task=task, phase="final_variation", condition=name,
                         successes=variations[name]["successes"], episodes=variations[name]["episodes"])
        finally:
            adapter.close()
        training = {"method": "behavior cloning" if best_index == 0 else "behavior cloning with DAgger",
                    "architecture": "128x128 tanh MLP with sinusoidal episode time",
                    "training_seed": args.training_seed, "training_seeds": training_seeds,
                    "used_training_seeds": [record["seed"] for record in collection],
                    "selected_training_seeds": [record["seed"] for record in collection if record["round"] <= best_index],
                    "development_seeds": development_seeds, "final_seeds": final_seeds,
                    "selection": "Maximum development rollout successes; ties use mean shaped return then fewer steps.",
                    "selected_round": best_index, "selected_transitions": candidates[best_index]["transitions"],
                    "demonstrations": args.demonstrations, "max_dagger_rounds": args.dagger_rounds,
                    "dagger_episodes_per_round": args.dagger_episodes, "updates_per_round": args.updates,
                    "physics": "Native Robosuite hard resets, default task geometry and robot initialization; no cameras during fitting/evaluation.",
                    "action_contract": "Panda OSC_POSE, 20 Hz, delta translation outputs [-1,1], zero rotation deltas, gripper -1/open +1/close.",
                    "limitations": "Compact state policies trained from a scripted demonstrator; no images, language conditioning, VLA pretraining, or broad robustness guarantee.",
                    "seconds": round(time.monotonic() - started, 2)}
        evaluation = {**nominal, "split": "untouched final held-out seeds", "variations": variations}
        metrics = {"task": task, "training": training, "candidates": candidates, "collection": collection,
                   "evaluation": evaluation, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        _json(output / "metrics.json", metrics)
        _json(output / "training-config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        (output / "training-progress.json").unlink(missing_ok=True)
        entry = {"id": "starter-" + task.lower(), "label": f"{task} · Panda · learned state policy",
                 "description": f"Locally trained behavior-cloning policy; {nominal['successes']}/{nominal['episodes']} held-out nominal successes.",
                 "policy": {"format": "torchscript", "source": "local", "name": f"{task.lower()}/best.pt"},
                 "checkpoint_sha256": metrics["checkpoint_sha256"],
                 "environment": {key: value for key, value in environment.items() if key != "env_version"},
                 "mapping": {"state_keys": teacher.state_keys},
                 "instruction": {"Lift": "Lift the cube.", "Stack": "Stack the red cube on the green cube.",
                                 "PickPlaceCan": "Place the can into its target bin."}[task],
                 "evaluation": evaluation, "training": training,
                 "metrics_path": f"{task.lower()}/metrics.json"}
        return entry
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demonstrations", type=int, default=64)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-episodes", type=int, default=32)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--development-episodes", type=int, default=10)
    parser.add_argument("--final-episodes", type=int, default=20)
    parser.add_argument("--variation-episodes", type=int, default=10)
    parser.add_argument("--training-seed", type=int, default=314159)
    args = parser.parse_args(argv)
    if min(args.demonstrations, args.updates, args.development_episodes, args.final_episodes) < 1 or min(args.dagger_rounds, args.dagger_episodes, args.variation_episodes) < 0:
        parser.error("Episode/update counts must be positive; optional DAgger/variation counts cannot be negative.")
    args.output = args.output.resolve()
    from setup_robotics import bootstrap_runtime
    bootstrap_runtime()
    import torch
    torch.set_num_threads(2)
    catalog_path = args.output / "catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {"schema_version": 1, "policies": []}
    for task in args.tasks:
        entry = train_task(task, args)
        catalog["policies"] = [item for item in catalog["policies"] if item["id"] != entry["id"]] + [entry]
        _json(catalog_path, catalog)
        _log(task=task, phase="saved", checkpoint=entry["policy"]["name"])


if __name__ == "__main__":
    main()
