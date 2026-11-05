"""Native policy inference adapters for the isolated robotics worker.

Importing this module does not import Torch, NumPy, a simulator, or a policy
framework. Call the public functions after setup_robotics.bootstrap_runtime().
Checkpoint directories are data: they are never added to sys.path.
"""
from __future__ import annotations

import base64
import copy
import importlib
import io
import json
import pickle
import re
import zipfile
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class PolicyError(ValueError):
    """An actionable checkpoint, embodiment, or inference contract mismatch."""


_MAPPING_KEYS = frozenset({"state_keys", "state_indices", "observation_map", "action_indices",
                           "action_scale", "action_offset", "state_mean", "state_std",
                           "torchscript_input", "clip_actions"})
_FORMATS = frozenset({"robomimic", "torchscript", "sb3", "lerobot"})
_JSON_LIMIT = 16 * 1024 * 1024


def _mapping(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyError("Policy mapping must be a JSON object.")
    unknown = set(value) - _MAPPING_KEYS
    if unknown:
        raise PolicyError(f"Unknown mapping fields: {', '.join(sorted(unknown))}.")
    result = copy.deepcopy(value)
    for key in ("state_keys", "state_indices", "action_indices"):
        if key in result:
            items = result[key]
            if not isinstance(items, list) or not items:
                raise PolicyError(f"mapping.{key} must be a nonempty list.")
            valid = (all(isinstance(x, str) and x for x in items) if key == "state_keys" else
                     all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in items))
            if not valid or len(set(items)) != len(items):
                raise PolicyError(f"mapping.{key} needs distinct {'names' if key == 'state_keys' else 'nonnegative indices'}.")
    obs_map = result.get("observation_map", {})
    if not isinstance(obs_map, dict) or not all(isinstance(k, str) and isinstance(v, str) and k and v
                                               for k, v in obs_map.items()):
        raise PolicyError("observation_map must map expected policy feature names to raw environment keys.")
    if "clip_actions" in result and not isinstance(result["clip_actions"], bool):
        raise PolicyError("clip_actions must be true or false; clipping is disabled by default.")
    if result.get("torchscript_input", "state") not in ("state", "dict"):
        raise PolicyError("torchscript_input must be 'state' or 'dict'.")
    return result


def _array(value, label, dtype=None):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy has no bfloat16 dtype; VLA inference commonly produces it.
        if value.is_floating_point():
            value = value.float()
        value = value.numpy()
    try:
        result = np.asarray(value, dtype=dtype)
        if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
            raise ValueError("nonfinite or nonnumeric")
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"{label} must contain only finite numeric values.") from exc
    return result


def _parameter(value, dimension, label, *, positive=False):
    import numpy as np
    result = _array(value, label, np.float32)
    if result.ndim > 1 or (result.ndim == 1 and result.shape != (dimension,)):
        raise PolicyError(f"{label} must be a scalar or have exactly {dimension} entries.")
    if positive and np.any(result <= 0):
        raise PolicyError(f"{label} must be strictly positive.")
    return result


class ObservationMapper:
    """Explicit observation ordering, shape checks, and optional state scaling."""
    def __init__(self, mapping=None):
        self.mapping = _mapping(mapping)

    def state(self, observation, *, key="observation.state"):
        import numpy as np
        keys = self.mapping.get("state_keys")
        if keys is None:
            raw_key = self.mapping.get("observation_map", {}).get(key, key)
            if raw_key not in observation:
                raise PolicyError("Set mapping.state_keys in training order (or map the state feature in "
                                  "observation_map); the adapter cannot infer state ordering or units.")
            keys = [raw_key]
        parts = []
        for name in keys:
            if name not in observation:
                raise PolicyError(f"Missing state observation '{name}'. Available: {', '.join(observation)}.")
            part = _array(observation[name], f"State '{name}'", np.float32)
            if part.ndim > 1:
                raise PolicyError(f"State '{name}' has shape {part.shape}; state_keys must name vectors, not cameras.")
            parts.append(part.reshape(-1))
        state = np.concatenate(parts)
        indices = self.mapping.get("state_indices")
        if indices is not None:
            if max(indices) >= state.size:
                raise PolicyError(f"state_indices refer past the {state.size} concatenated state dimensions.")
            state = state[indices]
        mean = _parameter(self.mapping.get("state_mean", 0), state.size, "state_mean")
        std = _parameter(self.mapping.get("state_std", 1), state.size, "state_std", positive=True)
        return _array((state - mean) / std, "Mapped state", np.float32)

    def feature(self, key, feature, observation):
        kind = str(feature.get("type", "STATE")).upper()
        raw_key = self.mapping.get("observation_map", {}).get(key, key)
        if kind == "STATE" and key == "observation.state":
            value = self.state(observation, key=key)
        else:
            if raw_key not in observation:
                raise PolicyError(f"Missing {'camera' if kind in ('VISUAL', 'RGB', 'DEPTH') else 'observation'} "
                                  f"'{raw_key}' required by '{key}'. Set observation_map and enable that camera/key.")
            value = _array(observation[raw_key], f"Observation '{raw_key}'")
        shape = tuple(feature.get("shape", ()))
        expected = shape
        if kind in ("VISUAL", "RGB", "DEPTH") and len(shape) == 3:
            expected = (shape[1], shape[2], shape[0])
        if shape and value.shape != expected:
            raise PolicyError(f"Observation '{key}' has shape {value.shape}; expected {expected}. "
                              "Check state order/indices and camera resolution.")
        return value


class ActionMapper:
    """Map native postprocessed actions to the controller, then check limits."""
    def __init__(self, env, mapping=None):
        import numpy as np
        self.mapping = _mapping(mapping)
        try:
            low, high = env.action_spec
        except (AttributeError, TypeError, ValueError) as exc:
            raise PolicyError("The environment must expose its raw Robosuite action_spec.") from exc
        self.low, self.high = _array(low, "Controller lower bounds"), _array(high, "Controller upper bounds")
        if self.low.ndim != 1 or self.high.shape != self.low.shape or np.any(self.low > self.high):
            raise PolicyError("The controller exposes invalid action bounds.")
        self.dimension = self.low.size
        self.scale = _parameter(self.mapping.get("action_scale", 1), self.dimension, "action_scale")
        self.offset = _parameter(self.mapping.get("action_offset", 0), self.dimension, "action_offset")
        if "action_indices" in self.mapping and len(self.mapping["action_indices"]) != self.dimension:
            raise PolicyError(f"action_indices must select exactly {self.dimension} controller dimensions.")

    def __call__(self, action):
        import numpy as np
        action = _array(action, "Policy action", np.float32)
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        if action.ndim != 1:
            raise PolicyError(f"Policy action has shape {action.shape}; expected one action vector. "
                              "Use the framework's select_action method for queued action chunks.")
        indices = self.mapping.get("action_indices")
        if indices is not None:
            if max(indices) >= action.size:
                raise PolicyError(f"action_indices refer past the policy's {action.size} outputs.")
            action = action[indices]
        if action.shape != self.low.shape:
            raise PolicyError(f"Policy outputs {action.size} action dimensions; controller requires {self.dimension}. "
                              "Choose a matching embodiment/controller or supply explicit action_indices and units.")
        action = _array(action * self.scale + self.offset, "Mapped action", np.float32)
        if self.mapping.get("clip_actions", False):
            return np.clip(action, self.low, self.high)
        outside = np.flatnonzero((action < self.low) | (action > self.high))
        if outside.size:
            index = int(outside[0])
            raise PolicyError(f"Action[{index}]={float(action[index]):.6g} exceeds controller bounds "
                              f"[{float(self.low[index]):.6g}, {float(self.high[index]):.6g}]. "
                              "Check action units, scale/offset and controller; clipping requires clip_actions=true.")
        return action


def _selection(selection):
    if not isinstance(selection, dict) or selection.get("format") not in _FORMATS:
        raise PolicyError("Choose policy format robomimic, torchscript, sb3, or lerobot.")
    result = dict(selection)
    source = result.setdefault("source", "local")
    if source not in ("local", "huggingface"):
        raise PolicyError("Resolve an uploaded checkpoint to a local path before loading it.")
    name = result.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PolicyError("Choose a checkpoint file/folder or Hugging Face model ID.")
    if source == "huggingface" and result["format"] != "lerobot":
        raise PolicyError("Hugging Face model IDs are supported for LeRobot; provide a local native checkpoint for this format.")
    if source == "local":
        path = Path(name).expanduser().resolve()
        if not path.exists():
            raise PolicyError(f"Checkpoint does not exist: {path}.")
        if result["format"] == "lerobot" and path.name == "config.json":
            path = path.parent
        if result["format"] != "lerobot" and not path.is_file():
            raise PolicyError("This format requires its native checkpoint file.")
        result["name"] = str(path)
    return result


def _json_bytes(data, label):
    if len(data) > _JSON_LIMIT:
        raise PolicyError(f"{label} exceeds the 16 MiB metadata limit.")
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise PolicyError(f"Invalid JSON in {label}: {exc}") from exc
    if not isinstance(result, dict):
        raise PolicyError(f"{label} must contain a JSON object.")
    return result


def _json_file(path):
    if path.stat().st_size > _JSON_LIMIT:
        raise PolicyError(f"Metadata file {path.name} is too large.")
    return _json_bytes(path.read_bytes(), path.name)


def _zip_json(archive, name):
    if archive.getinfo(name).file_size > _JSON_LIMIT:
        raise PolicyError(f"Checkpoint metadata {name} is too large.")
    return _json_bytes(archive.read(name), name)


def _sidecar(path):
    candidates = ([path / "robotics_policy.json"] if path.is_dir() else
                  [Path(str(path) + ".metadata.json"), path.with_suffix(".metadata.json")])
    for candidate in candidates:
        if candidate.is_file():
            return _json_file(candidate)
    return {}


def _features(raw):
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise PolicyError("Checkpoint feature schema must be an object.")
    result = {}
    for key, feature in raw.items():
        if not isinstance(key, str) or not isinstance(feature, dict):
            raise PolicyError("Checkpoint features need names, shape, and type.")
        shape = feature.get("shape", [])
        if not isinstance(shape, (list, tuple)) or not all(isinstance(x, int) and x > 0 for x in shape):
            raise PolicyError(f"Invalid feature shape for '{key}'.")
        result[key] = {"shape": list(shape), "type": str(feature.get("type", "STATE")).upper()}
    return result


def _metadata_base(selection, raw):
    result = {"format": selection["format"], "source": selection["source"],
              "type": raw.get("type", selection["format"]),
              "input_features": _features(raw.get("input_features")),
              "output_features": _features(raw.get("output_features")),
              "environment": copy.deepcopy(raw.get("environment", {})),
              "warnings": list(raw.get("warnings", []))}
    if not isinstance(result["environment"], dict):
        raise PolicyError("Checkpoint environment metadata must be an object.")
    for key in ("state_keys", "torchscript_input", "normalization", "robot_type"):
        if key in raw:
            result[key] = copy.deepcopy(raw[key])
    result["cameras"] = []
    for key, feature in result["input_features"].items():
        shape = feature["shape"]
        if feature["type"] in ("VISUAL", "RGB", "DEPTH") and len(shape) == 3:
            # LeRobot names do not define raw Robosuite cameras; the runner applies
            # observation_map. Robomimic names do follow <camera>_image.
            name = key[:-6] if key.endswith("_image") else (key[:-6] if key.endswith("_depth") else None)
            result["cameras"].append({"key": key, "name": name, "height": shape[1], "width": shape[2],
                                      "depth": feature["type"] == "DEPTH"})
    return result


def _robomimic_checkpoint(path):
    import torch
    import numpy as np
    # Older native checkpoints may contain NumPy arrays in normalization data.
    # Only these known NumPy constructors are allowlisted; never fall back to
    # weights_only=False or execute arbitrary checkpoint Python.
    allowed = [np.ndarray, np.dtype, np.core.multiarray._reconstruct, np.core.multiarray.scalar]
    if hasattr(np, "dtypes"):
        allowed.extend(getattr(np.dtypes, name) for name in dir(np.dtypes)
                       if name.endswith("DType") and isinstance(getattr(np.dtypes, name), type))
    try:
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PolicyError("Cannot safely load this Robomimic checkpoint. Export a native state-dictionary "
                          "checkpoint with JSON/config and tensor or NumPy statistics; uploaded Python objects "
                          f"are not executed. Loader detail: {exc}") from exc
    if not isinstance(checkpoint, dict) or not all(k in checkpoint for k in ("config", "model", "shape_metadata")):
        raise PolicyError("Not a native Robomimic checkpoint (expected config, model, and shape_metadata).")
    return checkpoint


def _robomimic_metadata(selection, checkpoint):
    config = checkpoint["config"]
    if isinstance(config, str):
        config = _json_bytes(config, "Robomimic config")
    shape = checkpoint["shape_metadata"]
    modalities = config.get("observation", {}).get("modalities", {}).get("obs", {})
    kinds = {key: kind for kind, keys in modalities.items() for key in keys}
    features = {key: {"shape": list(value), "type": {"rgb": "VISUAL", "depth": "DEPTH"}.get(kinds.get(key), "STATE")}
                for key, value in shape.get("all_shapes", {}).items()}
    env_meta = checkpoint.get("env_metadata", {})
    if isinstance(env_meta, list):
        if len(env_meta) != 1:
            raise PolicyError("This Robomimic checkpoint has multiple environments; export the intended task metadata.")
        env_meta = env_meta[0]
    kwargs = env_meta.get("env_kwargs", {})
    environment = {"task": env_meta.get("env_name"), "robots": kwargs.get("robots"),
                   "env_version": env_meta.get("env_version"),
                   "controller_configs": kwargs.get("controller_configs"), "env_kwargs": copy.deepcopy(kwargs)}
    environment = {k: v for k, v in environment.items() if v is not None}
    result = _metadata_base(selection, {"type": checkpoint.get("algo_name", config.get("algo_name", "robomimic")),
        "environment": environment, "input_features": features,
        "output_features": {"action": {"shape": [int(shape["ac_dim"])], "type": "ACTION"}}})
    result["normalization"] = {"observation": checkpoint.get("obs_normalization_stats") is not None,
                               "action": checkpoint.get("action_normalization_stats") is not None}
    result["frame_stack"] = int(config.get("train", {}).get("frame_stack", 1))
    if not 1 <= result["frame_stack"] <= 128:
        raise PolicyError("Robomimic frame_stack must be between 1 and 128 observations.")
    return result


def _lerobot_config_path(selection, filename="config.json"):
    if selection["source"] == "local":
        path = Path(selection["name"]) / filename
        if not path.is_file():
            raise PolicyError(f"LeRobot folder is missing {filename}. Upload the complete pretrained_model folder, "
                              "including saved processors and their normalization tensors.")
        return path
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=selection["name"], filename=filename,
                                revision=selection.get("revision")))


def read_policy_metadata(selection) -> dict:
    """Read native schemas without creating an environment or a policy model."""
    selection = _selection(selection)
    fmt = selection["format"]
    path = Path(selection["name"])
    if fmt == "robomimic":
        return _robomimic_metadata(selection, _robomimic_checkpoint(path))
    if fmt == "lerobot":
        config_path = _lerobot_config_path(selection)
        config = _json_file(config_path)
        raw = dict(config)
        if selection["source"] == "local":
            sidecar = _sidecar(path)
            raw.update({k: v for k, v in sidecar.items() if k not in ("type", "input_features", "output_features")})
        if not config.get("type"):
            raise PolicyError("LeRobot config.json is missing its registered policy 'type'.")
        metadata = _metadata_base(selection, raw)
        if selection["source"] == "huggingface":
            parts = config_path.parts
            for index, part in enumerate(parts[:-1]):
                if part == "snapshots" and re.fullmatch(r"[a-f0-9]{40}", parts[index + 1]):
                    metadata["resolved_revision"] = parts[index + 1]
                    break
            if not metadata.get("resolved_revision"):
                raise PolicyError("Could not resolve the Hugging Face checkpoint to an immutable snapshot revision.")
        return metadata
    if fmt == "torchscript":
        raw = {}
        try:
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith(("/extra/metadata.json", "/extra/robotics_metadata.json")):
                        raw.update(_zip_json(archive, name))
        except zipfile.BadZipFile as exc:
            raise PolicyError("TorchScript requires a torch.jit.save archive, not a Python model or state_dict.") from exc
        raw.update(_sidecar(path))
        return _metadata_base(selection, raw)
    # SB3's space information is pickled. Metadata reading is restricted to its
    # JSON declarations; loading below uses a restrictive unpickler for spaces.
    try:
        with zipfile.ZipFile(path) as archive:
            data = _zip_json(archive, "data")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise PolicyError("SB3 requires a native model.save() .zip checkpoint.") from exc
    raw = _sidecar(path)
    raw.setdefault("type", selection.get("algorithm") or "sb3")
    metadata = _metadata_base(selection, raw)
    normalizer_path = _sb3_normalizer_path(path)
    metadata["normalization"] = {"vecnormalize_sidecar": normalizer_path.name if normalizer_path else None}
    if normalizer_path is None:
        metadata["warnings"].append("SB3 VecNormalize statistics are saved separately. If training used it, include "
            "<checkpoint stem>.vecnormalize.pkl (or sibling vecnormalize.pkl), or export a TorchScript policy with "
            "normalization included. A .zip alone cannot detect missing statistics. state_mean/state_std reproduce "
            "only affine normalization, not VecNormalize observation clipping.")
    for source_key, destination in (("observation_space", "input_features"), ("action_space", "output_features")):
        space = data.get(source_key, {})
        if isinstance(space, dict) and ":serialized:" in space:
            metadata[destination] = _space_features(_decode_sb3(space, source_key), action=source_key == "action_space")
    # Recompute derived camera declarations after decoding Dict observation space.
    metadata["cameras"] = _metadata_base(selection, metadata)["cameras"]
    return metadata


def _validate_environment(metadata, env):
    expected = metadata.get("environment", {})
    actual_task = getattr(env, "name", None) or getattr(env, "_env_name", None) or type(env).__name__
    if expected.get("task") and expected["task"] != actual_task:
        raise PolicyError(f"Checkpoint task is '{expected['task']}', but selected environment is '{actual_task}'.")
    actual_robots = getattr(env, "robot_names", None)
    if actual_robots is None and hasattr(env, "robots"):
        actual_robots = [type(robot.robot_model).__name__ for robot in env.robots]
    expected_robots = expected.get("robots")
    if expected_robots and actual_robots:
        left = [expected_robots] if isinstance(expected_robots, str) else list(expected_robots)
        right = [actual_robots] if isinstance(actual_robots, str) else list(actual_robots)
        if left != right:
            raise PolicyError(f"Checkpoint robots {left} do not match environment robots {right}.")


def _device(device):
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?|mps", device):
        raise PolicyError("Policy device must be auto, cpu, cuda, cuda:N, or mps.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise PolicyError("CUDA was requested, but this worker cannot use CUDA. Choose CPU or repair the runtime.")
    return device


class _Adapter:
    def __init__(self, env, metadata, mapping=None, device="cpu"):
        self.env = env
        self.metadata = metadata
        self.mapping = _mapping(mapping)
        self.observations = ObservationMapper(self.mapping)
        self.actions = ActionMapper(env, self.mapping)
        self.device = device
        self.closed = False
        self._features = None

    def describe(self):
        return {**copy.deepcopy(self.metadata), "device": self.device, "mapping": copy.deepcopy(self.mapping),
                "controller_action_dimension": self.actions.dimension,
                "compatibility": "Dimensions and limits are validated; action units and robot embodiment must match training."}

    def _open(self):
        if self.closed:
            raise PolicyError("The policy adapter is closed.")

    def reset(self):
        self._open()
        if self._features is not None:
            self._features.reset()

    def enable_features(self):
        """Opt into representations observed alongside ordinary action calls."""
        self._open()
        if self._features is None:
            from robotics_features import FeatureCapture
            self._features = FeatureCapture(self)

    def take_features(self):
        """Consume the latest fresh decision's feature vector, if available."""
        return self._features.take() if self._features is not None else None

    def feature_info(self):
        return self._features.describe() if self._features is not None else {"enabled": False}

    def _begin_features(self):
        if self._features is not None:
            self._features.begin()

    def _feature_input(self, value):
        if self._features is not None:
            self._features.input(value)

    def _finish_features(self, action):
        if self._features is not None:
            self._features.commit()
        return action

    def close(self):
        if self._features is not None:
            self._features.close()
        self.closed = True
        self.env = None
        if hasattr(self, "policy"):
            self.policy = None


class TorchScriptAdapter(_Adapter):
    def __init__(self, path, env, metadata, mapping=None, device="cpu"):
        super().__init__(env, metadata, mapping, device)
        self.path = str(path)
        self._advanced = False
        self._load()

    def _load(self):
        import torch
        self.policy = torch.jit.load(self.path, map_location=self.device).eval()
        if self._features is not None:
            self._features.bind(self)

    def reset(self):
        super().reset()
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        elif self._advanced:
            # A ScriptModule can mutate attributes even without an exported reset.
            # Reload restores every saved attribute, not only its state_dict.
            self._load()
        self._advanced = False

    def act(self, observation):
        import torch
        import numpy as np
        self._open()
        self._begin_features()
        schema = self.metadata.get("input_features", {})
        if self.mapping.get("torchscript_input", self.metadata.get("torchscript_input", "state")) == "dict":
            if not schema:
                raise PolicyError("TorchScript dict input needs input_features in metadata.json to validate feature shapes.")
            value = {}
            for key, feature in schema.items():
                array = self.observations.feature(key, feature, observation)
                if feature["type"] in ("VISUAL", "RGB"):
                    array = _upright_image(array).transpose(2, 0, 1).copy()
                    if array.dtype == np.uint8:
                        array = array.astype(np.float32) / 255.0
                    elif array.size and (array.min() < 0 or array.max() > 1):
                        raise PolicyError(f"TorchScript camera '{key}' must be uint8 [0,255] or float [0,1].")
                value[key] = torch.as_tensor(np.array(array, copy=True), dtype=torch.float32,
                                            device=self.device).unsqueeze(0)
        else:
            state = self.observations.state(observation)
            if schema:
                feature = schema.get("observation.state") or next(iter(schema.values()))
                if tuple(feature.get("shape", ())) != state.shape:
                    raise PolicyError(f"TorchScript state shape {state.shape} does not match {feature.get('shape')}.")
            value = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            method = getattr(self.policy, "select_action", None) or self.policy
            self._feature_input(value)
            self._advanced = True
            output = method(value)
        if isinstance(output, dict) and set(output) in ({"action", "features"}, {"actions", "features"}):
            if self._features is not None:
                self._features.script_output(output["features"])
            output = output["action"] if "action" in output else output["actions"]
        if isinstance(output, dict) and set(output) in ({"action"}, {"actions"}):
            output = next(iter(output.values()))
        if not isinstance(output, torch.Tensor):
            raise PolicyError("TorchScript must return one action tensor (or {'action': tensor} with optional "
                              "'features': tensor from the same forward); "
                              "export recurrent state/chunk handling inside select_action and reset.")
        return self._finish_features(self.actions(output))


def _upright_image(value):
    # Robosuite's default image convention is OpenGL (bottom row first). Respect
    # installations explicitly configured for OpenCV without importing a simulator.
    import sys
    macros = sys.modules.get("robosuite.macros")
    convention = getattr(macros, "IMAGE_CONVENTION", "opengl")
    return value[::-1].copy() if convention == "opengl" else value.copy()


class LeRobotAdapter(_Adapter):
    def __init__(self, policy, preprocessor, postprocessor, env, *, metadata,
                 mapping=None, instruction="", device="cpu"):
        super().__init__(env, metadata, mapping, device)
        self.policy = policy
        self.preprocessor, self.postprocessor = preprocessor, postprocessor
        self.instruction = instruction

    def reset(self):
        super().reset()
        self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            if callable(getattr(processor, "reset", None)):
                processor.reset()

    def act(self, observation):
        import torch
        import numpy as np
        self._open()
        self._begin_features()
        batch = {}
        for key, feature in self.metadata["input_features"].items():
            value = self.observations.feature(key, feature, observation)
            if feature["type"] in ("VISUAL", "RGB"):
                value = _upright_image(value)
                tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1).contiguous()
                if tensor.dtype == torch.uint8:
                    tensor = tensor.float() / 255.0
                else:
                    tensor = tensor.float()
                    if tensor.numel() and (tensor.min() < 0 or tensor.max() > 1):
                        raise PolicyError(f"LeRobot camera '{key}' must be uint8 [0,255] or float [0,1].")
            else:
                tensor = torch.as_tensor(np.array(value, copy=True), dtype=torch.float32)
            # Native AddBatchDimension, device, tokenizer and normalizer steps own
            # the rest of input processing. Do not pre-batch or pre-normalize twice.
            batch[key] = tensor
        batch["task"] = self.instruction
        if self.metadata.get("robot_type"):
            batch["robot_type"] = self.metadata["robot_type"]
        with torch.inference_mode():
            prepared = self.preprocessor(batch)
            self._feature_input(prepared)
            output = self.policy.select_action(prepared)
            output = self.postprocessor(output)
        return self._finish_features(self.actions(output))

    def close(self):
        super().close()
        self.preprocessor = self.postprocessor = None


def _validate_processor_config(config):
    steps = config.get("steps", [])
    if not isinstance(steps, list):
        raise PolicyError("LeRobot processor steps must be a list.")
    for step in steps:
        if not isinstance(step, dict):
            raise PolicyError("Invalid LeRobot processor step.")
        # Even an installed lerobot.* callable can be a factory for remote code.
        # Native saved processor pipelines use registered ProcessorStep classes.
        if not isinstance(step.get("registry_name"), str) or not step["registry_name"]:
            raise PolicyError("LeRobot processor steps must identify an installed registered processor with "
                              "registry_name. Arbitrary class/callable imports are not allowed.")
        state_file = step.get("state_file")
        if state_file and (Path(state_file).is_absolute() or ".." in Path(state_file).parts):
            raise PolicyError("LeRobot processor state files must stay within the checkpoint folder.")


def _validate_lerobot_assets(selection, config):
    if selection["source"] == "local":
        path = Path(selection["name"])
        if any(path.rglob("*.py")) or any(path.rglob("*.pyc")):
            raise PolicyError("LeRobot checkpoint folders must contain model/configuration data, not Python modules. "
                              "Export the native pretrained_model folder without uploaded Python code.")
    if config.get("auto_map") or config.get("trust_remote_code"):
        raise PolicyError("This checkpoint requests custom Python code. Use an installed native LeRobot policy "
                          "and processor, or export TorchScript.")


@contextmanager
def _no_dynamic_checkpoint_code():
    """Block Transformers' code-from-checkpoint path in this isolated worker.

    Some native VLA constructors unconditionally pass trust_remote_code=True.
    Installed model classes work normally, but that flag cannot authorize code
    supplied in a checkpoint. Also guard aliases imported before this context.
    """
    import sys
    import transformers.dynamic_module_utils as dynamic
    original = dynamic.get_class_from_dynamic_module

    def reject(*args, **kwargs):
        raise PolicyError("This policy/tokenizer requires Python modules supplied by its checkpoint. "
                          "Install a native implementation or use an export; checkpoint Python is not executed.")

    def replace(source, target):
        for name, module in list(sys.modules.items()):
            if name.startswith("transformers.") and module is not None:
                # vars avoids triggering Transformers' lazy import __getattr__.
                if vars(module).get("get_class_from_dynamic_module") is source:
                    setattr(module, "get_class_from_dynamic_module", target)

    replace(original, reject)
    try:
        yield
    finally:
        replace(reject, original)


def _materialize_lerobot_checkpoint(selection):
    """Give native loaders local files from a single already-resolved snapshot.

    Some LeRobot 0.6.1 overrides accept revision but fail to forward it. Local
    materialization prevents those overrides from fetching a moving Hub branch.
    """
    root = _lerobot_config_path(selection).parent
    processor_configs = {}
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = _lerobot_config_path(selection, filename)
        if path.parent != root:
            raise PolicyError("LeRobot processor configuration did not resolve to the same checkpoint snapshot.")
        config = _json_file(path)
        _validate_processor_config(config)
        processor_configs[filename] = config
        for step in config.get("steps", []):
            if step.get("state_file"):
                state_path = _lerobot_config_path(selection, step["state_file"])
                if not state_path.is_relative_to(root):
                    raise PolicyError("LeRobot processor statistics resolved outside the checkpoint snapshot.")
    weights = _lerobot_config_path(selection, "model.safetensors")
    if weights.parent != root:
        raise PolicyError("LeRobot weights did not resolve to the same checkpoint snapshot as its configuration.")
    return str(root), processor_configs


def _load_lerobot_model(policy_class, config, directory):
    from safetensors import safe_open
    from safetensors.torch import load_file
    weights = Path(directory) / "model.safetensors"
    if getattr(config, "reinit_modules", None):
        raise PolicyError("This LeRobot config requests reinit_modules during weight loading. "
                          "Disable training-time reinitialization and export a complete matching checkpoint for inference.")
    try:
        with safe_open(weights, framework="pt", device="cpu") as tensors:
            if not list(tensors.keys()):
                raise PolicyError("LeRobot model.safetensors contains no weights.")
        if config.type in ("pi0", "pi05", "pi0_fast"):
            # These native 0.6.1 from_pretrained overrides catch *all* weight-load
            # failures and return a fresh or partially loaded model. Use their
            # own key-remapping method, then require strict loading to succeed.
            policy = policy_class(config)
            state = policy._fix_pytorch_state_dict_keys(load_file(weights), config)
            state = {key if key.startswith("model.") else f"model.{key}": value for key, value in state.items()}
            policy.load_state_dict(state, strict=True)
        else:
            policy = policy_class.from_pretrained(directory, config=config, strict=True)
        return policy.to(config.device).eval()
    except PolicyError:
        raise
    except Exception as exc:
        raise PolicyError(f"LeRobot weights could not be loaded: {exc}. "
                          "Refusing to run a freshly initialized or partially loaded policy.") from exc


def _load_lerobot(selection, env, metadata, mapping, instruction, device):
    # Read and vet processor configs before the native factory can import any
    # dynamic processor classes. A missing processor never falls back to fresh
    # normalization statistics or a randomly initialized processor.
    _validate_lerobot_assets(selection, _json_file(_lerobot_config_path(selection)))
    name, processor_configs = _materialize_lerobot_checkpoint(selection)
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    revision = selection.get("revision")
    config = PreTrainedConfig.from_pretrained(name, revision=revision)
    config.device = device
    config.pretrained_path = name
    config.pretrained_revision = revision
    overrides = {}
    for step in processor_configs["policy_preprocessor.json"].get("steps", []):
        identity = step.get("registry_name") or step.get("class", "").rsplit(".", 1)[-1]
        if identity in ("device_processor", "DeviceProcessorStep"):
            overrides[identity] = {"device": device}
    with _no_dynamic_checkpoint_code():
        policy_class = get_policy_class(config.type)
        from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
        for processor_config in processor_configs.values():
            for step in processor_config.get("steps", []):
                try:
                    step_class = ProcessorStepRegistry.get(step["registry_name"])
                except KeyError as exc:
                    raise PolicyError(f"LeRobot processor '{step['registry_name']}' is not installed/registered.") from exc
                if not isinstance(step_class, type) or not issubclass(step_class, ProcessorStep):
                    raise PolicyError(f"Registered processor '{step['registry_name']}' is not a native ProcessorStep.")
        policy = _load_lerobot_model(policy_class, config, name)
        preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=name,
            pretrained_revision=revision, preprocessor_overrides=overrides)
    return LeRobotAdapter(policy, preprocessor, postprocessor, env, metadata=metadata,
                          mapping=mapping, instruction=instruction, device=device)


class RobomimicAdapter(_Adapter):
    def __init__(self, policy, env, metadata, mapping, device, obs_utils, *, preprocess_images):
        super().__init__(env, metadata, mapping, device)
        self.policy, self.obs_utils = policy, obs_utils
        self.preprocess_images = preprocess_images
        self.frame_stack = metadata.get("frame_stack", 1)
        self.history = deque(maxlen=self.frame_stack)

    def reset(self):
        super().reset()
        self.policy.start_episode()
        self.history.clear()

    def act(self, observation):
        import torch
        import numpy as np
        self._open()
        self._begin_features()
        raw = dict(observation)
        # This alias is defined by Robomimic's EnvRobosuite wrapper itself.
        if "object" not in raw and "object-state" in raw:
            raw["object"] = raw["object-state"]
        mapped = {}
        for key, feature in self.metadata["input_features"].items():
            value = self.observations.feature(key, feature, raw)
            if feature["type"] in ("VISUAL", "RGB", "DEPTH"):
                value = _upright_image(value)
                if feature["type"] == "DEPTH":
                    # MuJoCo stores a normalized depth buffer; native Robomimic
                    # depth observations are metric distances.
                    model = self.env.sim.model
                    near = model.vis.map.znear * model.stat.extent
                    far = model.vis.map.zfar * model.stat.extent
                    value = near / (1.0 - value * (1.0 - near / far))
                if self.preprocess_images:
                    value = self.obs_utils.process_obs(obs=value, obs_key=key)
            mapped[key] = value
        if self.frame_stack > 1:
            # Native FrameStackWrapper pads the first observation by repeating it.
            # Keep that ordering without taking ownership of environment resets.
            current = {key: np.array(value, copy=True) for key, value in mapped.items()}
            if not self.history:
                self.history.extend([current] * self.frame_stack)
            else:
                self.history.append(current)
            mapped = {key: np.stack([frame[key] for frame in self.history]) for key in current}
        with torch.inference_mode():
            self._feature_input(mapped)
            return self._finish_features(self.actions(self.policy(ob=mapped)))

    def close(self):
        super().close()
        self.history.clear()


def _load_robomimic(checkpoint, env, metadata, mapping, device):
    import inspect
    import torch
    import robomimic
    from robomimic.utils import file_utils, obs_utils
    from robomimic.algo import RolloutPolicy
    # 0.5 introduced native action denormalization and rotated action conversion.
    # Failing explicitly prevents older runtimes from treating those as commands.
    supports_action_stats = "action_normalization_stats" in inspect.signature(RolloutPolicy).parameters
    if checkpoint.get("action_normalization_stats") is not None and not supports_action_stats:
        raise PolicyError(f"Robomimic {robomimic.__version__} cannot interpret this checkpoint's action normalization. "
                          "Use its matching runtime or export a TorchScript policy including native preprocessing "
                          "and action postprocessing; do not discard these statistics.")
    config = checkpoint["config"]
    config = json.loads(config) if isinstance(config, str) else config
    goals = config.get("observation", {}).get("modalities", {}).get("goal", {})
    if any(goals.values()):
        raise PolicyError("This Robomimic policy requires goal observations. Export an inference policy with an "
                          "explicit goal contract; current raw Robosuite observations do not contain its goals.")
    try:
        policy, _ = file_utils.policy_from_checkpoint(ckpt_dict=checkpoint, device=torch.device(device))
    except Exception as exc:
        raise PolicyError(f"Robomimic {robomimic.__version__} could not restore '{metadata['type']}': {exc}. "
                          "This runtime supports its native architectures; checkpoints from newer releases or "
                          "custom training code need a compatible runtime or a TorchScript export.") from exc
    image_processing_in_policy = "postprocess_visual_obs" in inspect.signature(policy._prepare_observation).parameters
    return RobomimicAdapter(policy, env, metadata, mapping, device, obs_utils,
                           preprocess_images=not image_processing_in_policy)


_SAFE_PICKLE_GLOBALS = {
    "builtins": {"set", "frozenset", "slice", "complex"},
    "collections": {"OrderedDict", "deque"},
    "numpy": {"ndarray", "dtype"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "numpy.core.numeric": {"_frombuffer"},
    "numpy._core.numeric": {"_frombuffer"},
    "numpy.random._pickle": {"__generator_ctor", "__bit_generator_ctor", "__randomstate_ctor"},
    "numpy.random.bit_generator": {"__pyx_unpickle_SeedSequence", "SeedSequence"},
    "numpy.random._pcg64": {"PCG64", "PCG64DXSM"},
    "numpy.random._mt19937": {"MT19937"},
    "numpy.random._philox": {"Philox"},
    "numpy.random._sfc64": {"SFC64"},
    "gymnasium.spaces.box": {"Box"},
    "gymnasium.spaces.dict": {"Dict"},
    "gymnasium.spaces.discrete": {"Discrete"},
    "gymnasium.spaces.multi_discrete": {"MultiDiscrete"},
    "gymnasium.spaces.multi_binary": {"MultiBinary"},
    "gymnasium.spaces.tuple": {"Tuple"},
    "torch.nn.modules.activation": {"Tanh", "ReLU", "ELU", "LeakyReLU", "GELU", "SiLU", "Sigmoid", "Softplus"},
    "torch.nn.modules.linear": {"Identity"},
    "torch.nn.modules.rnn": {"LSTM", "GRU"},
    "torch.optim.rmsprop": {"RMSprop"},
    "torch.optim.adam": {"Adam"},
    "torch.optim.adamw": {"AdamW"},
    "torch.optim.sgd": {"SGD"},
    "stable_baselines3.common.policies": {"ActorCriticPolicy", "ActorCriticCnnPolicy", "MultiInputActorCriticPolicy"},
    "stable_baselines3.common.type_aliases": {"TrainFreq", "TrainFrequencyUnit"},
    "stable_baselines3.common.torch_layers": {"FlattenExtractor", "NatureCNN", "CombinedExtractor"},
    "stable_baselines3.common.buffers": {"RolloutBuffer", "DictRolloutBuffer", "ReplayBuffer", "DictReplayBuffer"},
    "stable_baselines3.her.her_replay_buffer": {"HerReplayBuffer"},
    "stable_baselines3.sac.policies": {"SACPolicy", "CnnPolicy", "MultiInputPolicy"},
    "stable_baselines3.td3.policies": {"TD3Policy", "CnnPolicy", "MultiInputPolicy"},
    "stable_baselines3.dqn.policies": {"DQNPolicy", "CnnPolicy", "MultiInputPolicy"},
    "stable_baselines3.common.noise": {"NormalActionNoise", "OrnsteinUhlenbeckActionNoise", "VectorizedActionNoise"},
    "stable_baselines3.common.vec_env.vec_normalize": {"VecNormalize"},
    "stable_baselines3.common.running_mean_std": {"RunningMeanStd"},
    "sb3_contrib.common.recurrent.policies": {"RecurrentActorCriticPolicy", "RecurrentActorCriticCnnPolicy",
                                              "RecurrentMultiInputActorCriticPolicy"},
    "sb3_contrib.common.recurrent.buffers": {"RecurrentRolloutBuffer", "RecurrentDictRolloutBuffer"},
    "sb3_contrib.tqc.policies": {"TQCPolicy", "CnnPolicy", "MultiInputPolicy"},
}


class _SafeSB3Unpickler(pickle.Unpickler):
    """Decode native space/architecture data without cloudpickle code execution."""
    def find_class(self, module, name):
        if name not in _SAFE_PICKLE_GLOBALS.get(module, set()):
            raise PolicyError(f"SB3 checkpoint references unsupported Python object {module}.{name}. "
                              "Custom Python policies/functions are not executed; export TorchScript instead.")
        return getattr(importlib.import_module(module), name)

    def persistent_load(self, pid):
        raise PolicyError("SB3 metadata may not contain persistent Python references.")


def _decode_sb3(value, key):
    if not isinstance(value, dict) or ":serialized:" not in value:
        return value
    try:
        data = base64.b64decode(value[":serialized:"], validate=True)
        return _SafeSB3Unpickler(io.BytesIO(data)).load()
    except PolicyError:
        raise
    except Exception as exc:
        raise PolicyError(f"Cannot safely decode SB3 {key}: {exc}. Export a native policy or TorchScript.") from exc


def _space_features(space, *, action=False):
    if hasattr(space, "spaces"):
        if not isinstance(space.spaces, dict):
            raise PolicyError("SB3 tuple observations are not supported; export a named Dict observation policy.")
        features = {}
        for key, child in space.spaces.items():
            shape = list(child.shape)
            kind = "VISUAL" if len(shape) == 3 and shape[0] in (1, 3, 4) else "STATE"
            features[key] = {"shape": shape, "type": kind}
        return features
    shape = getattr(space, "shape", None)
    if shape is None:
        raise PolicyError("SB3 observations/actions must use continuous Box spaces or named Dict observations.")
    if not action and len(shape) == 3 and shape[0] in (1, 3, 4):
        return {"observation.image": {"shape": list(shape), "type": "VISUAL"}}
    return {"action" if action else "observation.state":
            {"shape": list(shape), "type": "ACTION" if action else "STATE"}}


class SB3Adapter(_Adapter):
    def __init__(self, policy, env, metadata, mapping, device, observation_space, normalizer=None):
        super().__init__(env, metadata, mapping, device)
        self.policy, self.observation_space = policy, observation_space
        self.state = None
        self.episode_start = True
        self.normalizer = normalizer

    def reset(self):
        super().reset()
        self.state = None
        self.episode_start = True

    def act(self, observation):
        import numpy as np
        self._open()
        self._begin_features()
        if hasattr(self.observation_space, "spaces"):
            value = {}
            for key, feature in _space_features(self.observation_space).items():
                item = self.observations.feature(key, feature, observation)
                if feature["type"] == "VISUAL":
                    item = item.transpose(2, 0, 1).copy()
                value[key] = item
        else:
            feature = next(iter(_space_features(self.observation_space).values()))
            if feature["type"] == "VISUAL":
                value = self.observations.feature("observation.image", feature, observation).transpose(2, 0, 1).copy()
            else:
                value = self.observations.state(observation)
            expected = tuple(self.observation_space.shape)
            if value.shape != expected:
                raise PolicyError(f"SB3 observation shape is {value.shape}; checkpoint requires {expected}. "
                                  "Set state_keys and state_indices in training order.")
        if self.normalizer is not None:
            value = self.normalizer.normalize_obs(value)
            for key, item in (value.items() if isinstance(value, dict) else [("state", value)]):
                _array(item, f"SB3 normalized {key}")
        self._feature_input(value)
        action, state = self.policy.predict(value, state=self.state,
            episode_start=np.array([self.episode_start]), deterministic=True)
        action = self.actions(action)
        self.state, self.episode_start = state, False
        return self._finish_features(action)

    def close(self):
        super().close()
        self.normalizer = self.state = None


def _sb3_normalizer_path(checkpoint):
    for path in (checkpoint.with_suffix(".vecnormalize.pkl"), checkpoint.parent / "vecnormalize.pkl"):
        if path.is_file():
            return path
    return None


def _load_sb3_normalizer(checkpoint, observation_space, mapping):
    from stable_baselines3.common.vec_env import VecNormalize
    path = _sb3_normalizer_path(Path(checkpoint))
    if path is None:
        return None
    if any(k in mapping for k in ("state_mean", "state_std")):
        raise PolicyError("The checkpoint has saved VecNormalize statistics. Remove manual state_mean/state_std "
                          "to avoid normalizing the same observation twice.")
    if path.stat().st_size > _JSON_LIMIT:
        raise PolicyError("VecNormalize sidecar is larger than the 16 MiB statistics limit.")
    with path.open("rb") as stream:
        normalizer = _SafeSB3Unpickler(stream).load()
    if not isinstance(normalizer, VecNormalize):
        raise PolicyError("The normalization sidecar must come from VecNormalize.save().")
    if normalizer.observation_space != observation_space:
        raise PolicyError("VecNormalize sidecar observation space does not match the SB3 checkpoint.")
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def _load_sb3(selection, env, metadata, mapping, device):
    import numpy as np
    from gymnasium.spaces import Box
    with zipfile.ZipFile(selection["name"]) as archive:
        data = _zip_json(archive, "data")
    custom_objects = {}
    for key, value in data.items():
        if isinstance(value, dict) and ":serialized:" in value:
            # Training schedules do not affect deterministic inference. Replacing
            # them avoids executing lambdas serialized by cloudpickle.
            if key in ("lr_schedule", "clip_range", "exploration_schedule"):
                custom_objects[key] = lambda _: 0.0
            elif key == "clip_range_vf":
                custom_objects[key] = None
            elif key == "learning_rate":
                custom_objects[key] = 0.0
            else:
                custom_objects[key] = _decode_sb3(value, key)
    policy_class = custom_objects.get("policy_class", data.get("policy_class"))
    module = getattr(policy_class, "__module__", "")
    algorithm = selection.get("algorithm") or metadata.get("type")
    if algorithm in (None, "sb3"):
        algorithm = ("SAC" if module.startswith("stable_baselines3.sac.") else
                     "RecurrentPPO" if module.startswith("sb3_contrib.common.recurrent.") else None)
    supported = {"ppo": "PPO", "a2c": "A2C", "sac": "SAC", "td3": "TD3", "ddpg": "DDPG",
                 "recurrentppo": "RecurrentPPO", "trpo": "TRPO", "tqc": "TQC"}
    if not isinstance(algorithm, str) or algorithm.lower() not in supported:
        raise PolicyError("Set selection.algorithm to the training algorithm (PPO, A2C, SAC, TD3, DDPG; "
                          "RecurrentPPO/TRPO/TQC require sb3-contrib). SB3 archives do not reliably identify it.")
    algorithm = supported[algorithm.lower()]
    package = "sb3_contrib" if algorithm in ("RecurrentPPO", "TRPO", "TQC") else "stable_baselines3"
    cls = getattr(importlib.import_module(package), algorithm)
    action_space = custom_objects.get("action_space", data.get("action_space"))
    if not isinstance(action_space, Box) or len(action_space.shape) != 1:
        raise PolicyError("Robosuite requires a continuous vector action space; this SB3 policy uses another space.")
    # Always invoke the native loader: it restores architecture, preprocessing,
    # action unscaling and recurrent policy behavior. Its tensors use weights_only.
    policy = cls.load(selection["name"], device=device, custom_objects=custom_objects)
    metadata = copy.deepcopy(metadata)
    metadata["type"] = algorithm
    metadata["input_features"] = _space_features(policy.observation_space)
    metadata["output_features"] = _space_features(policy.action_space, action=True)
    if not mapping.get("action_indices") and not any(k in mapping for k in ("action_scale", "action_offset")):
        low, high = env.action_spec
        if action_space.shape == np.shape(low) and (np.any(action_space.low < low) or np.any(action_space.high > high)):
            raise PolicyError("SB3 training action bounds exceed the selected controller bounds. "
                              "Specify the matching controller or an explicit action scale/offset.")
    normalizer = _load_sb3_normalizer(selection["name"], policy.observation_space, mapping)
    return SB3Adapter(policy, env, metadata, mapping, device, policy.observation_space, normalizer)


def load_policy(selection, env, *, device="auto", instruction="", mapping=None, progress=None):
    """Load a native policy in the worker; do not step or reset the environment."""
    selection = _selection(selection)
    mapping = _mapping(mapping)
    device = _device(device)
    if progress:
        progress(f"Reading {selection['format']} checkpoint metadata")
    checkpoint = None
    if selection["format"] == "robomimic":
        checkpoint = _robomimic_checkpoint(selection["name"])
        metadata = _robomimic_metadata(selection, checkpoint)
    else:
        metadata = read_policy_metadata(selection)
    if metadata.get("resolved_revision"):
        selection["revision"] = metadata["resolved_revision"]
    _validate_environment(metadata, env)
    if "state_keys" not in mapping and metadata.get("state_keys"):
        mapping["state_keys"] = metadata["state_keys"]
    if progress:
        progress(f"Loading native {metadata['type']} policy on {device}")
    try:
        if selection["format"] == "torchscript":
            return TorchScriptAdapter(selection["name"], env, metadata, mapping, device)
        if selection["format"] == "lerobot":
            return _load_lerobot(selection, env, metadata, mapping, instruction, device)
        if selection["format"] == "robomimic":
            return _load_robomimic(checkpoint, env, metadata, mapping, device)
        return _load_sb3(selection, env, metadata, mapping, device)
    except PolicyError:
        raise
    except (ImportError, ModuleNotFoundError) as exc:
        raise PolicyError(f"The isolated runtime is missing a dependency for {metadata['type']}: {exc}. "
                          "Run setup_robotics.py and install this policy type's optional dependencies in that runtime.") from exc
    except Exception as exc:
        raise PolicyError(f"Could not load {selection['format']} policy '{metadata['type']}': {exc}. "
                          "Check native checkpoint version, architecture, and saved processor files.") from exc
