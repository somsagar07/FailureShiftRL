"""Adapter contracts: mappings and native inference semantics, not learned skill."""
import json
import importlib.util
import pickle
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


class Environment:
    action_spec = (np.full(2, -1.0), np.full(2, 1.0))


class MappingTests(unittest.TestCase):
    def test_state_order_selection_and_normalization_are_explicit(self):
        from robotics_policies import ObservationMapper
        mapper = ObservationMapper({"state_keys": ["gripper", "arm"],
                                    "state_indices": [2, 0], "state_mean": [1, 2], "state_std": [2, 4]})
        np.testing.assert_array_equal(mapper.state({"arm": [5, 9], "gripper": [6]}), [4, 1])

    def test_state_mapping_rejects_missing_keys_nonfinite_and_inconsistent_dimensions(self):
        from robotics_policies import ObservationMapper, PolicyError
        cases = [({}, {"arm": [1]}), ({"state_keys": ["missing"]}, {"arm": [1]}),
                 ({"state_keys": ["arm"]}, {"arm": [np.nan]}),
                 ({"state_keys": ["arm"], "state_indices": [2]}, {"arm": [1]}),
                 ({"state_keys": ["arm"], "state_std": [0]}, {"arm": [1]}),
                 ({"state_keys": ["arm"], "state_mean": [0, 1]}, {"arm": [1]})]
        for mapping, observation in cases:
            with self.subTest(mapping=mapping), self.assertRaises(PolicyError):
                ObservationMapper(mapping).state(observation)

    def test_mapping_typos_do_not_get_ignored(self):
        from robotics_policies import ObservationMapper, PolicyError
        with self.assertRaisesRegex(PolicyError, "state_key"):
            ObservationMapper({"state_key": ["arm"]})

    def test_camera_mapping_requires_actual_camera_and_exact_shape(self):
        from robotics_policies import ObservationMapper, PolicyError
        mapper = ObservationMapper({"observation_map": {"observation.images.front": "agentview_image"}})
        feature = {"shape": [3, 2, 4], "type": "VISUAL"}
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        self.assertEqual(mapper.feature("observation.images.front", feature,
                                       {"agentview_image": image}).shape, (2, 4, 3))
        for obs in ({}, {"agentview_image": np.zeros((3, 4, 3))}):
            with self.assertRaises(PolicyError):
                mapper.feature("observation.images.front", feature, obs)

    def test_missing_named_feature_is_not_replaced_by_unrelated_state(self):
        from robotics_policies import ObservationMapper, PolicyError
        with self.assertRaises(PolicyError):
            ObservationMapper({"state_keys": ["arm"]}).feature("gripper", {"shape": [2], "type": "STATE"},
                                                                    {"arm": np.array([.1, .2])})

    def test_action_mapping_runs_after_native_postprocessing_and_never_clips_implicitly(self):
        from robotics_policies import ActionMapper, PolicyError
        mapper = ActionMapper(Environment(), {"action_indices": [2, 0],
                                             "action_scale": [2, .5], "action_offset": [.1, -.1]})
        np.testing.assert_allclose(mapper([.8, 100, .2]), [.5, .3])
        for action in ([2, 0], [np.inf, 0], [0], [[0, 0], [0, 0]]):
            with self.subTest(action=action), self.assertRaises(PolicyError):
                ActionMapper(Environment(), {})(action)
        np.testing.assert_array_equal(ActionMapper(Environment(), {"clip_actions": True})([2, -2]), [1, -1])

    def test_action_indices_cannot_refer_to_missing_dimension(self):
        from robotics_policies import ActionMapper, PolicyError
        with self.assertRaises(PolicyError):
            ActionMapper(Environment(), {"action_indices": [0, 3]})([0, 0])

    def test_bfloat16_vla_actions_are_checked_and_converted(self):
        from robotics_policies import ActionMapper
        action = torch.tensor([.25, -.5], dtype=torch.bfloat16)
        np.testing.assert_array_equal(ActionMapper(Environment())(action), [.25, -.5])


class ScriptedPolicy(torch.nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, :2] * .1


class StatefulScriptedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(1))

    @torch.jit.export
    def reset(self):
        self.count.zero_()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        self.count.add_(1)
        return state[:, :2] * 0 + self.count * .1


class CameraScriptedPolicy(torch.nn.Module):
    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        image = observation["front"]
        return torch.stack((image[:, 0, 0, 0], image[:, 0, -1, 0]), dim=1)


class NoResetScriptedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        self.count.add_(1)
        return state[:, :2] * 0 + self.count * .1


class TorchScriptTests(unittest.TestCase):
    def test_module_without_exported_reset_restores_saved_mutable_attributes(self):
        from robotics_policies import load_policy
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutable.pt"
            torch.jit.script(NoResetScriptedPolicy()).save(str(path))
            adapter = load_policy({"format": "torchscript", "name": str(path)}, Environment(),
                                  device="cpu", mapping={"state_keys": ["arm"]})
            np.testing.assert_allclose(adapter.act({"arm": [0, 0]}), [.1, .1])
            np.testing.assert_allclose(adapter.act({"arm": [0, 0]}), [.2, .2])
            adapter.reset()
            np.testing.assert_allclose(adapter.act({"arm": [0, 0]}), [.1, .1])

    def test_dict_camera_uses_declared_channel_shape_and_upright_pixels(self):
        from robotics_policies import load_policy
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.pt"
            torch.jit.script(CameraScriptedPolicy()).save(str(path), _extra_files={"metadata.json": json.dumps({
                "torchscript_input": "dict", "input_features": {"front": {"shape": [3, 2, 4], "type": "VISUAL"}}})})
            adapter = load_policy({"format": "torchscript", "name": str(path)}, Environment(), device="cpu",
                                 mapping={"observation_map": {"front": "agentview_image"}})
            camera = np.zeros((2, 4, 3), np.uint8)
            camera[1] = 255
            adapter.reset()
            np.testing.assert_allclose(adapter.act({"agentview_image": camera}), [1, 0])

    def test_real_checkpoint_state_mapping_metadata_and_episode_reset(self):
        from robotics_policies import load_policy, read_policy_metadata
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            torch.jit.script(StatefulScriptedPolicy()).save(str(path), _extra_files={
                "metadata.json": json.dumps({"environment": {"task": "Lift", "robots": ["Panda"]},
                    "input_features": {"observation.state": {"shape": [2], "type": "STATE"}}})})
            selection = {"format": "torchscript", "source": "local", "name": str(path)}
            metadata = read_policy_metadata(selection)
            self.assertEqual(metadata["environment"]["task"], "Lift")
            # Real Robosuite environments are also checked against metadata; this fixture
            # intentionally supplies the matching identifiers.
            env = Environment()
            env.name = "Lift"
            env.robot_names = ["Panda"]
            adapter = load_policy(selection, env, device="cpu", mapping={"state_keys": ["arm"]})
            adapter.reset()
            np.testing.assert_allclose(adapter.act({"arm": [2, 3]}), [.1, .1])
            np.testing.assert_allclose(adapter.act({"arm": [2, 3]}), [.2, .2])
            adapter.reset()
            np.testing.assert_allclose(adapter.act({"arm": [2, 3]}), [.1, .1])
            adapter.close()

    def test_checkpoint_task_mismatch_fails_before_rollout(self):
        from robotics_policies import load_policy, PolicyError
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            torch.jit.script(ScriptedPolicy()).save(str(path))
            Path(str(path) + ".metadata.json").write_text(json.dumps({"environment": {"task": "Door"}}))
            env = Environment()
            env.name = "Lift"
            with self.assertRaisesRegex(PolicyError, "Door.*Lift|Lift.*Door"):
                load_policy({"format": "torchscript", "name": str(path)}, env,
                            device="cpu", mapping={"state_keys": ["arm"]})


class ChunkPolicy:
    """Native select_action owns its queue, as ACT and VLA policies do."""
    def __init__(self):
        self.queue = deque()

    def reset(self):
        self.queue.clear()

    def select_action(self, observation):
        if not self.queue:
            # This makes camera, task and preprocessor ordering observable in action.
            assert observation["task"] == ["lift the block"]
            assert observation["observation.images.front"].shape == (1, 3, 2, 4)
            state = observation["observation.state"]
            self.queue.extend([state, state + .1])
        return self.queue.popleft()


class Processor:
    def __init__(self, before):
        self.before = before
        self.calls_since_reset = 0

    def reset(self):
        self.calls_since_reset = 0

    def __call__(self, value):
        self.calls_since_reset += 1
        if not self.before:
            return value * 2
        value = dict(value)
        value["observation.state"] = value["observation.state"].unsqueeze(0) + .1
        value["observation.images.front"] = value["observation.images.front"].unsqueeze(0)
        value["task"] = [value["task"]]
        return value


class LeRobotContractTests(unittest.TestCase):
    def test_hub_policy_files_are_materialized_from_one_immutable_revision(self):
        from robotics_policies import read_policy_metadata, _materialize_lerobot_checkpoint
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snapshots" / revision
            root.mkdir(parents=True)
            (root / "config.json").write_text(json.dumps({"type": "pi05"}))
            (root / "policy_preprocessor.json").write_text(json.dumps({"steps": [
                {"registry_name": "normalizer_processor", "state_file": "normalizer.safetensors"}]}))
            (root / "policy_postprocessor.json").write_text(json.dumps({"steps": []}))
            (root / "model.safetensors").touch()
            (root / "normalizer.safetensors").touch()
            revisions = []
            def download(repo_id, filename, revision=None):
                revisions.append(revision)
                return str(root / filename)
            selection = {"format": "lerobot", "source": "huggingface", "name": "example/policy", "revision": "main"}
            with mock.patch("huggingface_hub.hf_hub_download", side_effect=download):
                metadata = read_policy_metadata(selection)
                self.assertEqual(metadata["resolved_revision"], revision)
                selection["revision"] = metadata["resolved_revision"]
                local, _ = _materialize_lerobot_checkpoint(selection)
            self.assertEqual(Path(local), root)
            self.assertEqual(revisions[0], "main")
            self.assertTrue(all(value == revision for value in revisions[1:]))

    def test_pi_weights_use_native_remapping_but_never_silent_random_fallback(self):
        from safetensors.torch import save_file
        from robotics_policies import _load_lerobot_model, PolicyError
        class TinyPi(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = torch.nn.Linear(1, 2, bias=False)
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise AssertionError("The upstream catch-and-return loader must not be used")
            def _fix_pytorch_state_dict_keys(self, state, config):
                return {key.removeprefix("old_"): value for key, value in state.items()}
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(type="pi05", device="cpu")
            path = Path(directory) / "model.safetensors"
            save_file({"old_weight": torch.tensor([[.2], [.4]])}, path)
            loaded = _load_lerobot_model(TinyPi, config, directory)
            np.testing.assert_allclose(loaded.model(torch.ones(1, 1)).detach().numpy(), [[.2, .4]])
            save_file({"old_weight": torch.zeros(9, 9)}, path)
            with self.assertRaisesRegex(PolicyError, "weights|initialized|load"):
                _load_lerobot_model(TinyPi, config, directory)
            path.unlink()
            with self.assertRaises(PolicyError):
                _load_lerobot_model(TinyPi, config, directory)

    def test_native_processors_instruction_camera_and_chunk_reset(self):
        from robotics_policies import LeRobotAdapter
        preprocessor, postprocessor = Processor(True), Processor(False)
        adapter = LeRobotAdapter(ChunkPolicy(), preprocessor, postprocessor, Environment(),
            metadata={"format": "lerobot", "type": "smolvla", "input_features": {
                "observation.state": {"shape": [2], "type": "STATE"},
                "observation.images.front": {"shape": [3, 2, 4], "type": "VISUAL"}}},
            mapping={"state_keys": ["arm"], "observation_map": {
                "observation.images.front": "agentview_image"}, "action_scale": .5},
            instruction="lift the block", device="cpu")
        observation = {"arm": np.array([.2, .3]), "agentview_image": np.zeros((2, 4, 3), np.uint8)}
        adapter.reset()
        np.testing.assert_allclose(adapter.act(observation), [.3, .4])
        np.testing.assert_allclose(adapter.act(observation), [.4, .5])
        adapter.reset()
        self.assertEqual(preprocessor.calls_since_reset, 0)
        self.assertEqual(postprocessor.calls_since_reset, 0)
        np.testing.assert_allclose(adapter.act(observation), [.3, .4])

    def test_metadata_read_does_not_load_models_or_execute_directory_python(self):
        from robotics_policies import read_policy_metadata
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"type": "pi05", "input_features": {
                "observation.images.front": {"type": "VISUAL", "shape": [3, 224, 224]}},
                "output_features": {"action": {"type": "ACTION", "shape": [7]}}}))
            (path / "model.py").write_text("raise RuntimeError('uploaded modules must never execute')")
            result = read_policy_metadata({"format": "lerobot", "name": str(path)})
            self.assertEqual(result["type"], "pi05")
            self.assertEqual(result["input_features"]["observation.images.front"]["shape"], [3, 224, 224])

    def test_uploaded_processor_class_is_rejected_before_import(self):
        from robotics_policies import _validate_processor_config, PolicyError
        for name in ("uploaded_model.RunCommand", "lerobot.envs.factory.make_env"):
            with self.subTest(name=name), self.assertRaisesRegex(PolicyError, "processor|registered"):
                _validate_processor_config({"steps": [{"class": name, "config": {"trust_remote_code": True}}]})

    def test_inference_rejects_native_training_reinitialization(self):
        from safetensors.torch import save_file
        from robotics_policies import _load_lerobot_model, PolicyError
        class ReinitializingPolicy:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            save_file({"weight": torch.ones(1, 1)}, Path(directory) / "model.safetensors")
            config = SimpleNamespace(type="vla_jepa", device="cpu", reinit_modules=["weight"])
            with self.assertRaisesRegex(PolicyError, "reinit|initializ"):
                _load_lerobot_model(ReinitializingPolicy, config, directory)

    def test_native_loader_rejects_checkpoint_python_assets(self):
        from robotics_policies import _validate_lerobot_assets, PolicyError
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "modeling_uploaded.py").write_text("raise RuntimeError('must not execute')")
            with self.assertRaisesRegex(PolicyError, "Python"):
                _validate_lerobot_assets({"source": "local", "name": str(path)}, {})

    def test_transformers_cannot_execute_dynamic_checkpoint_modules(self):
        from robotics_policies import _no_dynamic_checkpoint_code, PolicyError
        import transformers.dynamic_module_utils as dynamic
        original = dynamic.get_class_from_dynamic_module
        with _no_dynamic_checkpoint_code(), self.assertRaisesRegex(PolicyError, "Python"):
            dynamic.get_class_from_dynamic_module("uploaded.Policy", "/nonexistent/checkpoint")
        self.assertIs(dynamic.get_class_from_dynamic_module, original)


class PredictPolicy:
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if episode_start[0]:
            state = 0
        return np.array([state * .1, observation[0] * .1]), state + 1


class SB3ContractTests(unittest.TestCase):
    def test_recurrent_predict_state_and_episode_start_reset(self):
        from robotics_policies import SB3Adapter
        space = type("Space", (), {"shape": (2,)})()
        adapter = SB3Adapter(PredictPolicy(), Environment(),
            {"format": "sb3", "type": "PPO"}, {"state_keys": ["arm"]}, "cpu", space)
        adapter.reset()
        np.testing.assert_allclose(adapter.act({"arm": [2, 3]}), [0, .2])
        np.testing.assert_allclose(adapter.act({"arm": [2, 3]}), [.1, .2])
        adapter.reset()
        np.testing.assert_allclose(adapter.act({"arm": [2, 3]}), [0, .2])

    def test_checkpoint_pickle_cannot_execute_uploaded_code(self):
        import io
        from robotics_policies import _SafeSB3Unpickler, PolicyError
        class Executable:
            def __reduce__(self):
                return eval, ("40 + 2",)
        with self.assertRaisesRegex(PolicyError, "Python|custom|unsafe|allowed"):
            _SafeSB3Unpickler(io.BytesIO(pickle.dumps(Executable()))).load()

    def test_saved_normalizer_runs_before_native_prediction(self):
        from robotics_policies import SB3Adapter
        class Normalizer:
            def normalize_obs(self, observation):
                return np.clip((observation - 2) / 2, -1, 1)
        adapter = SB3Adapter(PredictPolicy(), Environment(), {"format": "sb3", "type": "PPO"},
            {"state_keys": ["arm"]}, "cpu", type("Space", (), {"shape": (2,)})(),
            normalizer=Normalizer())
        adapter.reset()
        np.testing.assert_allclose(adapter.act({"arm": [10, 20]}), [0, .1])


class RobomimicContractTests(unittest.TestCase):
    def test_frame_history_repeats_initial_frame_and_clears_on_reset(self):
        from robotics_policies import RobomimicAdapter
        class NativePolicy:
            def start_episode(self):
                pass
            def __call__(self, ob):
                return [ob["object"][0, 0] * .1, ob["object"][-1, 0] * .1]
        adapter = RobomimicAdapter(NativePolicy(), Environment(), {
            "format": "robomimic", "type": "bc", "frame_stack": 2,
            "input_features": {"object": {"shape": [1], "type": "STATE"}}},
            {}, "cpu", None, preprocess_images=True)
        adapter.reset()
        np.testing.assert_allclose(adapter.act({"object-state": [1]}), [.1, .1])
        np.testing.assert_allclose(adapter.act({"object-state": [3]}), [.1, .3])
        np.testing.assert_allclose(adapter.act({"object-state": [5]}), [.3, .5])
        adapter.reset()
        np.testing.assert_allclose(adapter.act({"object-state": [2]}), [.2, .2])

    def test_native_rgb_processing_and_object_alias_preserve_raw_observation(self):
        from robotics_policies import RobomimicAdapter
        # Robomimic 0.3 expects RGB preprocessing before RolloutPolicy, while
        # 0.5 moved that operation into RolloutPolicy itself.
        class NativePolicy:
            def start_episode(self):
                pass
            def __call__(self, ob):
                return [ob["object"][0] * .1, ob["agentview_image"][0, 0, 0]]
        class NativeObsUtils:
            @staticmethod
            def process_obs(obs, obs_key):
                return obs.transpose(2, 0, 1) / 255.0
        adapter = RobomimicAdapter(NativePolicy(), Environment(), {
            "format": "robomimic", "type": "bc", "input_features": {
                "object": {"shape": [1], "type": "STATE"},
                "agentview_image": {"shape": [3, 2, 4], "type": "VISUAL"}}},
            {}, "cpu", NativeObsUtils(), preprocess_images=True)
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        image[1] = 255
        adapter.reset()
        np.testing.assert_allclose(adapter.act({"object-state": [3], "agentview_image": image}), [.3, 1])
        self.assertEqual(int(image[0, 0, 0]), 0)


class NativeCheckpointSmokeTests(unittest.TestCase):
    """Random tiny checkpoints validate loading/inference, not task success."""
    @unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "isolated SB3 runtime not active")
    def test_real_sb3_single_camera_cnn_preserves_native_pixels(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        from robotics_policies import load_policy, read_policy_metadata
        class CameraEnv(gym.Env):
            observation_space = gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
            action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
            def reset(self, *, seed=None, options=None):
                return self.observation_space.sample(), {}
            def step(self, action):
                return self.observation_space.sample(), 0., False, False, {}
        with tempfile.TemporaryDirectory() as directory:
            model = PPO("CnnPolicy", CameraEnv(), n_steps=2, batch_size=2,
                policy_kwargs={"net_arch": [8], "features_extractor_kwargs": {"features_dim": 8}},
                device="cpu", seed=1)
            path = Path(directory) / "cnn.zip"
            model.save(path)
            selection = {"format": "sb3", "name": str(path), "algorithm": "PPO"}
            metadata = read_policy_metadata(selection)
            self.assertEqual(metadata["input_features"]["observation.image"]["type"], "VISUAL")
            adapter = load_policy(selection, Environment(), device="cpu",
                mapping={"observation_map": {"observation.image": "agentview_image"}})
            camera = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
            adapter.reset()
            expected, _ = model.predict(camera.transpose(2, 0, 1), deterministic=True)
            np.testing.assert_allclose(adapter.act({"agentview_image": camera}), expected, atol=1e-7)
            adapter.close()

    @unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "isolated SB3 runtime not active")
    def test_real_sb3_other_continuous_algorithms(self):
        import gymnasium as gym
        from stable_baselines3 import A2C, SAC, TD3, DDPG
        from robotics_policies import load_policy
        with tempfile.TemporaryDirectory() as directory:
            training_env = gym.make("Pendulum-v1")
            env = Environment()
            env.action_spec = (np.array([-2.]), np.array([2.]))
            for algorithm in (A2C, SAC, TD3, DDPG):
                with self.subTest(algorithm=algorithm.__name__):
                    model = algorithm("MlpPolicy", training_env, policy_kwargs={"net_arch": [8]},
                                      device="cpu", seed=1)
                    path = Path(directory) / f"{algorithm.__name__}.zip"
                    model.save(path)
                    adapter = load_policy({"format": "sb3", "name": str(path), "algorithm": algorithm.__name__},
                        env, device="cpu", mapping={"state_keys": ["state"]})
                    adapter.reset()
                    observation = np.array([1., 0., 0.], np.float32)
                    expected, _ = model.predict(observation, deterministic=True)
                    np.testing.assert_allclose(adapter.act({"state": observation}), expected)
                    adapter.close()
            training_env.close()

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "isolated LeRobot runtime not active")
    def test_native_smolvla_saved_tokenizer_processors_preserve_instruction_and_units(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
        from robotics_policies import LeRobotAdapter
        class EchoVLA:
            def reset(self):
                pass
            def select_action(self, observation):
                pixel = observation["observation.images.front"][:, 0, 0, 0].unsqueeze(1)
                token = observation[OBS_LANGUAGE_TOKENS][:, 0].unsqueeze(1)
                return observation["observation.state"] * .1 + pixel * .1 + token * .01
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "lift": 2, "block": 3, "lower": 4},
                                                    unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(path / "tokenizer")
            features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 4))}
            config = SmolVLAConfig(device="cpu", vlm_model_name=str(path / "tokenizer"),
                tokenizer_max_length=4, pad_language_to="max_length", input_features=features,
                output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))})
            stats = {"observation.state": {"mean": [1., 1.], "std": [2., 4.]},
                     "action": {"mean": [.1, -.1], "std": [2., 3.]}}
            pre, post = make_pre_post_processors(config, dataset_stats=stats)
            pre.save_pretrained(path / "processors")
            post.save_pretrained(path / "processors")
            pre, post = make_pre_post_processors(config, pretrained_path=str(path / "processors"))
            adapter = LeRobotAdapter(EchoVLA(), pre, post, Environment(), metadata={"format": "lerobot",
                "type": "smolvla", "input_features": {key: {"type": feature.type.value, "shape": list(feature.shape)}
                                                          for key, feature in features.items()}},
                mapping={"state_keys": ["arm"], "observation_map": {"observation.images.front": "agentview_image"},
                         "action_scale": .5}, instruction="lift block")
            camera = np.zeros((2, 4, 3), np.uint8)
            camera[1] = 255
            obs = {"arm": [3., 5.], "agentview_image": camera}
            adapter.reset()
            np.testing.assert_allclose(adapter.act(obs), [.27, .28], atol=1e-6)
            adapter.instruction = "lower block"
            np.testing.assert_allclose(adapter.act(obs), [.29, .31], atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "isolated SB3 runtime not active")
    def test_real_sb3_ppo_archive(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from robotics_policies import load_policy
        with tempfile.TemporaryDirectory() as directory:
            training_env = VecNormalize(DummyVecEnv([lambda: gym.make("Pendulum-v1")]), clip_obs=1.)
            training_env.obs_rms.mean[:] = [.5, 1, 2]
            training_env.obs_rms.var[:] = [4, 9, 16]
            training_env.training = False
            model = PPO("MlpPolicy", training_env, n_steps=2, batch_size=2,
                        policy_kwargs={"net_arch": [8]}, device="cpu", seed=1)
            path = Path(directory) / "ppo.zip"
            model.save(path)
            training_env.save(Path(directory) / "ppo.vecnormalize.pkl")
            env = Environment()
            env.action_spec = (np.array([-2.]), np.array([2.]))
            adapter = load_policy({"format": "sb3", "name": str(path), "algorithm": "PPO"}, env,
                                  device="cpu", mapping={"state_keys": ["state"]})
            adapter.reset()
            observation = np.array([1., 0., 0.], np.float32)
            expected, _ = model.predict(training_env.normalize_obs(observation), deterministic=True)
            np.testing.assert_allclose(adapter.act({"state": observation}), expected)
            adapter.close()
            training_env.close()

    @unittest.skipUnless(importlib.util.find_spec("robomimic"), "isolated Robomimic runtime not active")
    def test_real_robomimic_recurrent_bc_checkpoint(self):
        from robomimic.config import config_factory
        from robomimic.algo import algo_factory
        import robomimic.utils.obs_utils as obs_utils
        from robotics_policies import load_policy, read_policy_metadata
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
        checkpoint = {"algo_name": "bc", "config": config.dump(), "model": native.serialize(),
                      "shape_metadata": {"all_shapes": {"object": [3]}, "ac_dim": 2},
                      "env_metadata": {"env_name": "Lift", "env_version": "1.4.1",
                                       "env_kwargs": {"robots": ["Panda"]}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bc.pth"
            torch.save(checkpoint, path)
            env = Environment()
            env.name, env.robot_names = "Lift", ["Panda"]
            selection = {"format": "robomimic", "name": str(path)}
            self.assertEqual(read_policy_metadata(selection)["environment"]["task"], "Lift")
            self.assertEqual(read_policy_metadata(selection)["environment"]["env_version"], "1.4.1")
            adapter = load_policy(selection, env, device="cpu")
            observation = {"object-state": np.array([.1, .2, .3])}
            adapter.reset()
            first = adapter.act(observation)
            second = adapter.act(observation)
            self.assertFalse(np.array_equal(first, second), "RNN should advance its hidden state")
            adapter.reset()
            np.testing.assert_allclose(adapter.act(observation), first)
            adapter.close()

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "isolated LeRobot runtime not active")
    def test_real_lerobot_act_and_saved_processors(self):
        from lerobot.configs import FeatureType, PolicyFeature, NormalizationMode
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from robotics_policies import load_policy
        config = ACTConfig(device="cpu", chunk_size=2, n_action_steps=2, dim_model=16,
            n_heads=2, dim_feedforward=32, n_encoder_layers=1, n_decoder_layers=1,
            use_vae=False, dropout=0., pretrained_backbone_weights=None,
            input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,))},
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            normalization_mapping={"STATE": NormalizationMode.IDENTITY, "ENV": NormalizationMode.IDENTITY,
                                   "ACTION": NormalizationMode.IDENTITY, "VISUAL": NormalizationMode.IDENTITY})
        policy = ACTPolicy(config)
        preprocessor, postprocessor = make_pre_post_processors(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrained_model"
            policy.save_pretrained(path)
            preprocessor.save_pretrained(path)
            postprocessor.save_pretrained(path)
            adapter = load_policy({"format": "lerobot", "name": str(path)}, Environment(), device="cpu",
                mapping={"state_keys": ["arm"], "observation_map": {
                    "observation.environment_state": "object-state"}, "action_scale": .1})
            obs = {"arm": np.array([.1, .2]), "object-state": np.array([.3, .4, .5])}
            adapter.reset()
            first, second = adapter.act(obs), adapter.act(obs)
            self.assertFalse(np.array_equal(first, second), "ACT should consume distinct action chunks")
            adapter.reset()
            np.testing.assert_allclose(adapter.act(obs), first)
            adapter.close()


if __name__ == "__main__":
    unittest.main()
