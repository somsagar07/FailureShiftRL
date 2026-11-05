"""Native action/feature equivalence; run with the isolated robotics runtime."""
import copy
import hashlib
import random
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Dict
from unittest import mock

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from setup_robotics import bootstrap_runtime
bootstrap_runtime()

import numpy as np
import torch
from robotics_policies import LeRobotAdapter, RobomimicAdapter, SB3Adapter, TorchScriptAdapter, load_policy


class Environment:
    action_spec = (np.full(2, -1.), np.full(2, 1.))


class StatefulScript(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.step = 0

    def forward(self, state):
        self.step += 1
        return torch.tanh(state + self.step * .01)


class FeatureScript(torch.nn.Module):
    def __init__(self, invalid=False, queued=False):
        super().__init__()
        self.step = 0
        self.invalid = invalid
        self.queued = queued

    def forward(self, state):
        self.step += 1
        features = torch.cat((state, state * 2), dim=-1)
        if self.invalid and self.step == 2:
            features = features[:, :2]
        if self.invalid and self.step == 3:
            features = features * float("nan")
        if self.queued and self.step % 2 == 0:
            features = torch.empty(0, dtype=state.dtype, device=state.device)
        return {"action": torch.tanh(state + torch.rand_like(state) * .01), "features": features}


class TimeConditionedPolicy(torch.nn.Module):
    """A misleading name/architecture must not enable the known starter path."""
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(2))
        self.register_buffer("std", torch.ones(2))
        self.register_buffer("frequencies", torch.tensor([1., 2., 4., 8.]) * torch.pi)
        self.step = 0
        self.horizon = 180.
        self.network = torch.nn.Sequential(torch.nn.Linear(11, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 4))

    def forward(self, state):
        self.step += 1
        return torch.tanh(state)


class ImageScript(torch.nn.Module):
    def forward(self, observation: Dict[str, torch.Tensor]):
        return torch.tanh(observation["state"])


class ObservationPolicy(torch.nn.Module):
    """An unsupported native architecture can still execute its normal API."""
    def reset(self):
        pass

    def select_action(self, observation):
        return torch.tanh(observation["observation.state"])


def rng_state():
    return (torch.random.get_rng_state().clone(),
            [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            np.random.get_state(), random.getstate())


def restore_rng(state):
    torch.random.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])
    np.random.set_state(state[2])
    random.setstate(state[3])


class FeatureContracts(unittest.TestCase):
    def assert_rng_equal(self, expected):
        actual = rng_state()
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual[1], expected[1])))
        self.assertEqual(actual[2][0], expected[2][0])
        np.testing.assert_array_equal(actual[2][1], expected[2][1])
        self.assertEqual(actual[2][2:], expected[2][2:])
        self.assertEqual(actual[3], expected[3])

    def require_interface(self, adapter):
        self.assertTrue(callable(getattr(adapter, "enable_features", None)),
                        "adapters need an opt-in feature interface")
        adapter.enable_features()

    def script_adapter(self, script, directory, metadata=None):
        path = Path(directory) / "policy.pt"
        torch.jit.save(torch.jit.script(script), str(path))
        return TorchScriptAdapter(path, Environment(), metadata or {"format": "torchscript"},
                                  {"state_keys": ["state"]})

    def test_script_fallback_consumes_once_and_reload_clears_pending_features(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.script_adapter(StatefulScript(), directory)
            self.require_interface(adapter)
            for _ in range(2):
                adapter.reset()
                self.assertIsNone(adapter.take_features())
                np.testing.assert_allclose(adapter.act({"state": [.1, .2]}), np.tanh([.11, .21]), rtol=1e-6)
                np.testing.assert_array_equal(adapter.take_features(), np.array([.1, .2], np.float32))
                self.assertIsNone(adapter.take_features())
                self.assertEqual(adapter.policy.step, 1)
            self.assertEqual(adapter.feature_info()["kind"], "observation")
            self.assertEqual(adapter.feature_info()["dimension"], 2)
            adapter.close()
            self.assertIsNone(adapter.take_features())

    def test_explicit_script_features_share_the_only_action_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.script_adapter(FeatureScript(), directory)
            self.require_interface(adapter)
            before = rng_state()
            adapter.act({"state": [.1, .2]})
            after = rng_state()
            np.testing.assert_array_equal(adapter.take_features(), np.array([.1, .2, .2, .4], np.float32))
            self.assertEqual(adapter.policy.step, 1)
            self.assertEqual(adapter.feature_info()["kind"], "policy_hidden")
            self.assertIn("features", adapter.feature_info()["source_layer"])
            adapter.reset()
            restore_rng(before)
            adapter.policy(torch.tensor([[.1, .2]]))
            self.assert_rng_equal(after)
            adapter.close()

    def test_invalid_features_never_change_actions_or_the_established_feature_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.script_adapter(FeatureScript(invalid=True), directory)
            self.require_interface(adapter)
            adapter.act({"state": [.1, .2]})
            self.assertEqual(adapter.take_features().shape, (4,))
            for _ in range(2):
                self.assertTrue(np.isfinite(adapter.act({"state": [.1, .2]})).all())
                self.assertIsNone(adapter.take_features())
                self.assertEqual(adapter.feature_info()["dimension"], 4)
                self.assertTrue(adapter.feature_info().get("warning"))
            adapter.close()

    def test_explicit_script_can_report_no_fresh_features_for_a_queued_action(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.script_adapter(FeatureScript(queued=True), directory)
            self.require_interface(adapter)
            adapter.act({"state": [.1, .2]})
            self.assertEqual(adapter.take_features().shape, (4,))
            adapter.act({"state": [.2, .3]})
            self.assertIsNone(adapter.take_features())
            self.assertNotIn("warning", adapter.feature_info())
            self.assertEqual(adapter.policy.step, 2)
            adapter.close()

    def test_metadata_and_architecture_alone_cannot_claim_starter_hidden_features(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.script_adapter(TimeConditionedPolicy(), directory,
                {"format": "torchscript", "type": "time_conditioned_behavior_cloning_mlp"})
            self.require_interface(adapter)
            adapter.act({"state": [.1, .2]})
            self.assertEqual(adapter.feature_info()["kind"], "observation")
            self.assertEqual(adapter.take_features().shape, (2,))
            adapter.close()

    def test_image_observation_fallback_is_bounded_and_never_labeled_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pt"
            torch.jit.save(torch.jit.script(ImageScript()), str(path))
            adapter = TorchScriptAdapter(path, Environment(), {"format": "torchscript",
                "torchscript_input": "dict", "input_features": {
                    "camera": {"shape": [3, 128, 128], "type": "VISUAL"},
                    "state": {"shape": [2], "type": "STATE"}}})
            self.require_interface(adapter)
            observation = {"camera": np.full((128, 128, 3), 128, np.uint8), "state": [.1, .2]}
            adapter.act(observation)
            vector = adapter.take_features()
            self.assertLessEqual(vector.size, 2048)
            np.testing.assert_array_equal(vector[-2:], np.array([.1, .2], np.float32))
            np.testing.assert_allclose(vector[:-2], 128 / 255)
            self.assertEqual(adapter.feature_info()["kind"], "observation")
            adapter.close()

    def test_unsupported_native_architecture_keeps_action_api_and_labels_observations(self):
        adapter = LeRobotAdapter(ObservationPolicy(), lambda value: value, lambda value: value,
            Environment(), metadata={"format": "lerobot", "type": "act", "input_features": {
                "observation.state": {"shape": [2], "type": "STATE"}}},
            mapping={"state_keys": ["state"]})
        self.require_interface(adapter)
        np.testing.assert_allclose(adapter.act({"state": [.1, .2]}), np.tanh([.1, .2]), rtol=1e-6)
        np.testing.assert_array_equal(adapter.take_features(), np.array([.1, .2], np.float32))
        self.assertEqual(adapter.feature_info()["kind"], "observation")
        adapter.close()

    def test_unavailable_native_hook_falls_back_without_claiming_hidden_features(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        training_env = gym.make("Pendulum-v1")
        model = PPO("MlpPolicy", training_env, n_steps=2, batch_size=2,
                    policy_kwargs={"net_arch": [8]}, device="cpu", seed=1)
        env = Environment()
        env.action_spec = (np.array([-2.]), np.array([2.]))
        adapter = SB3Adapter(model, env, {"format": "sb3", "type": "PPO"},
                             {"state_keys": ["state"]}, "cpu", model.observation_space)
        with mock.patch.object(model.policy.action_net, "register_forward_pre_hook",
                               side_effect=RuntimeError("Hooks unavailable")):
            self.require_interface(adapter)
        observation = np.array([1., 0., 0.], np.float32)
        expected = model.predict(observation, deterministic=True)[0]
        np.testing.assert_array_equal(adapter.act({"state": observation}), expected)
        self.assertEqual(adapter.feature_info()["kind"], "observation")
        np.testing.assert_array_equal(adapter.take_features(), observation)
        adapter.close()
        training_env.close()

    def test_sb3_flatten_only_input_is_an_observation_not_a_learned_embedding(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        training_env = gym.make("Pendulum-v1")
        model = PPO("MlpPolicy", training_env, n_steps=2, batch_size=2,
                    policy_kwargs={"net_arch": []}, device="cpu", seed=1)
        env = Environment()
        env.action_spec = (np.array([-2.]), np.array([2.]))
        adapter = SB3Adapter(model, env, {"format": "sb3", "type": "PPO"},
                             {"state_keys": ["state"]}, "cpu", model.observation_space)
        self.require_interface(adapter)
        adapter.act({"state": np.array([1., 0., 0.], np.float32)})
        self.assertEqual(adapter.feature_info()["kind"], "observation")
        self.assertEqual(adapter.take_features().shape, (3,))
        adapter.close()
        training_env.close()

    def test_all_existing_starters_keep_actions_clock_weights_and_rng_on_cpu_and_cuda(self):
        from train_starter_policies import build_policy
        paths = sorted((APP / "runs/robotics/starter-policies").glob("*/best.pt"))
        if not paths:
            self.skipTest("bundled starter checkpoints are not installed")
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for device in devices:
                with self.subTest(task=path.parent.name, device=device):
                    env = Environment()
                    env.name = {"lift": "Lift", "stack": "Stack", "pickplacecan": "PickPlaceCan"}[path.parent.name]
                    env.robot_names = ["Panda"]
                    env.action_spec = (np.full(7, -1.), np.full(7, 1.))
                    adapter = load_policy({"format": "torchscript", "name": str(path)}, env,
                                          device=device, mapping={"state_keys": ["state"]})
                    original = torch.jit.load(str(path), map_location=device).eval()
                    eager = build_policy(original.mean.cpu().numpy(), original.std.cpu().numpy(), original.horizon)
                    eager.load_state_dict(original.state_dict())
                    eager.to(device).eval()
                    expected_features = []
                    handle = eager.network[-1].register_forward_pre_hook(
                        lambda module, args: expected_features.append(args[0].detach().clone()))
                    self.require_interface(adapter)
                    self.assertEqual(adapter.feature_info()["kind"], "policy_hidden")
                    for step, offset in ((0, -.5), (1, .1), (int(original.horizon) - 1, 1.)):
                        state = original.mean + original.std * offset
                        adapter.policy.step = original.step = eager.step = step
                        before = rng_state()
                        with torch.inference_mode():
                            expected = original(state[None]).cpu().numpy()[0]
                            eager(state[None])
                        self.assert_rng_equal(before)
                        actual = adapter.act({"state": state.cpu().numpy()})
                        self.assert_rng_equal(before)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_allclose(adapter.take_features(),
                            expected_features[-1].cpu().numpy()[0], rtol=1e-5, atol=1e-6)
                        self.assertEqual(adapter.policy.step, step + 1)
                        self.assertEqual(adapter.feature_info()["dimension"], 128)
                    adapter.reset()
                    self.assertEqual(adapter.policy.step, 0)
                    self.assertIsNone(adapter.take_features())
                    handle.remove()
                    adapter.close()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_sb3_actor_hooks_preserve_native_actions_and_cleanup(self):
        import gymnasium as gym
        from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
        for cls in (PPO, A2C, SAC, TD3, DDPG):
            with self.subTest(algorithm=cls.__name__):
                training_env = gym.make("Pendulum-v1")
                kwargs = {"policy_kwargs": {"net_arch": [8]}, "device": "cpu", "seed": 1}
                if cls is PPO:
                    kwargs.update(n_steps=2, batch_size=2)
                model = cls("MlpPolicy", training_env, **kwargs)
                env = Environment()
                env.action_spec = (np.array([-2.]), np.array([2.]))
                adapter = SB3Adapter(model, env, {"format": "sb3", "type": cls.__name__},
                                     {"state_keys": ["state"]}, "cpu", model.observation_space)
                obs = {"state": np.array([1., 0., 0.], np.float32)}
                adapter.reset()
                before = rng_state()
                expected = [adapter.act(obs) for _ in range(3)]
                after = rng_state()
                self.require_interface(adapter)
                adapter.enable_features()
                adapter.reset()
                restore_rng(before)
                actual = []
                for _ in range(3):
                    actual.append(adapter.act(obs))
                    self.assertEqual(adapter.take_features().shape, (8,))
                    self.assertIsNone(adapter.take_features())
                self.assertEqual(adapter.feature_info()["kind"], "policy_hidden")
                np.testing.assert_array_equal(actual, expected)
                self.assert_rng_equal(after)
                adapter.act(obs)
                adapter.reset()
                self.assertIsNone(adapter.take_features())
                adapter.close()
                self.assertFalse(any(m._forward_pre_hooks for m in model.policy.modules()))
                training_env.close()

    def test_robomimic_recurrent_state_is_advanced_only_by_native_inference(self):
        from robomimic.config import config_factory
        from robomimic.algo import algo_factory, RolloutPolicy
        import robomimic.utils.obs_utils as obs_utils
        config = config_factory("bc")
        with config.unlocked():
            config.observation.modalities.obs.low_dim = ["object"]
            config.observation.modalities.obs.rgb = []
            config.algo.actor_layer_dims = [8]
            config.algo.rnn.enabled = True
            config.algo.rnn.hidden_dim = 8
            config.algo.rnn.num_layers = 1
            config.algo.gmm.enabled = False
        obs_utils.initialize_obs_utils_with_config(config)
        native = algo_factory("bc", config, {"object": [3]}, 2, torch.device("cpu"))
        adapter = RobomimicAdapter(RolloutPolicy(native), Environment(),
            {"format": "robomimic", "type": "bc", "input_features": {"object": {"shape": [3], "type": "STATE"}}},
            {}, "cpu", obs_utils, preprocess_images=True)
        obs = {"object-state": np.array([.1, .2, .3])}
        adapter.reset()
        before = rng_state()
        expected = [adapter.act(obs) for _ in range(3)]
        hidden = copy.deepcopy(native._rnn_hidden_state)
        after = rng_state()
        self.require_interface(adapter)
        adapter.reset()
        restore_rng(before)
        actual = []
        for _ in range(3):
            actual.append(adapter.act(obs))
            self.assertEqual(adapter.take_features().shape, (8,))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(native._rnn_counter, 3)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(hidden, native._rnn_hidden_state)))
        self.assert_rng_equal(after)
        adapter.close()

    def test_lerobot_chunk_consumption_never_reuses_old_features_or_resamples(self):
        from lerobot.configs import FeatureType, PolicyFeature, NormalizationMode
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors
        common = dict(device="cpu", pretrained_backbone_weights=None,
            input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,))},
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            normalization_mapping={key: NormalizationMode.IDENTITY for key in ("STATE", "ENV", "ACTION", "VISUAL")})
        for kind in ("act", "diffusion"):
            with self.subTest(policy=kind):
                if kind == "act":
                    config = ACTConfig(**common, chunk_size=2, n_action_steps=2, dim_model=16,
                        n_heads=2, dim_feedforward=32, n_encoder_layers=1, n_decoder_layers=1,
                        use_vae=False, dropout=0.)
                    policy = ACTPolicy(config)
                    target, axis = policy.model.action_head, 1
                else:
                    config = DiffusionConfig(**common, horizon=8, n_obs_steps=2, n_action_steps=2,
                        down_dims=(8, 16), kernel_size=3, n_groups=4, diffusion_step_embed_dim=8,
                        num_train_timesteps=4, num_inference_steps=2)
                    policy = DiffusionPolicy(config).eval()
                    target, axis = policy.diffusion.unet.final_conv[-1], 2
                pre, post = make_pre_post_processors(config)
                metadata = {"format": "lerobot", "type": kind, "input_features": {
                    "observation.state": {"shape": [2], "type": "STATE"},
                    "observation.environment_state": {"shape": [3], "type": "ENV"}}}
                adapter = LeRobotAdapter(policy, pre, post, Environment(), metadata=metadata,
                    mapping={"state_keys": ["arm"], "observation_map": {
                        "observation.environment_state": "object-state"}, "action_scale": .1})
                obs = {"arm": np.array([.1, .2]), "object-state": np.array([.3, .4, .5])}
                adapter.reset()
                before = rng_state()
                expected = [adapter.act(obs) for _ in range(4)]
                after = rng_state()
                self.require_interface(adapter)
                adapter.reset()
                restore_rng(before)
                captured = []
                handle = target.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0].detach().clone()))
                actual = []
                for step in range(4):
                    actual.append(adapter.act(obs))
                    vector = adapter.take_features()
                    if step % 2:
                        self.assertIsNone(vector)
                    else:
                        np.testing.assert_allclose(vector, captured[-1].float().mean(axis).cpu().numpy()[0])
                np.testing.assert_array_equal(actual, expected)
                self.assert_rng_equal(after)
                handle.remove()
                adapter.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
