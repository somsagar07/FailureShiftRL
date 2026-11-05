"""Isolate Torch from the app process while exercising an exported policy."""
import json
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from setup_robotics import bootstrap_runtime
bootstrap_runtime()
import numpy as np
import torch
from robotics_policies import load_policy, read_policy_metadata
from train_starter_policies import build_policy, export_policy


class ControllerContract:
    action_spec = (np.full(7, -1.), np.full(7, 1.))


torch.manual_seed(17)
policy = build_policy(np.array([2., -5., 8.]), np.array([.5, 4., 2.]), 180)
path = Path(sys.argv[1]) / "policy.pt"
metadata = {
    "type": "starter_behavior_cloning",
    "state_keys": ["pose", "gripper"],
    "input_features": {"observation.state": {"shape": [3], "type": "STATE"}},
    "output_features": {"action": {"shape": [7], "type": "ACTION"}},
}
export_policy(policy, path, metadata)
selection = {"format": "torchscript", "source": "local", "name": str(path)}
parsed = read_policy_metadata(selection)
assert parsed["state_keys"] == ["pose", "gripper"]
adapter = load_policy(selection, ControllerContract())
observation = {"pose": np.array([2., -5.]), "gripper": np.array([8.])}
state = torch.tensor([[2., -5., 8.]])
with torch.inference_mode():
    policy.reset()
    eager = policy(state).numpy()[0]
first = adapter.act(observation)
np.testing.assert_allclose(first, eager, atol=1e-6)
later = None
for _ in range(30):
    later = adapter.act(observation)
assert not np.allclose(first, later), "The time-conditioned policy must advance within an episode"
adapter.reset()
np.testing.assert_allclose(adapter.act(observation), first, atol=1e-7)
extreme = adapter.act({"pose": np.array([1e6, -1e6]), "gripper": np.array([1e6])})
assert np.isfinite(extreme).all() and np.max(np.abs(extreme)) <= 1
assert first.shape == (7,)
adapter.close()
print(json.dumps({"shape": [7], "reset": True, "bounded": True, "metadata": True}))
