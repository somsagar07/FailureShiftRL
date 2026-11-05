# Robotics policy checkpoints

The robotics worker loads existing policies for evaluation. Training the local
starter policies is a separate command, documented in the
[workspace guide](README.md#locally-trained-starter-policies).
Matching action dimensions alone does not establish compatibility: the task,
robot, observation order, controller, action units, and normalization must also
match training. A VLA trained to command a physical robot's joints is not
automatically compatible with Robosuite's Cartesian OSC controller.

The **Starter policy** selector fills the exact environment and observation
mapping for a completed local checkpoint. The included Lift, Stack, and
PickPlaceCan trainers export TorchScript with internal state normalization and an
episode clock reset by `reset()`. Keep the selected 20 Hz OSC_POSE control contract
when testing these policies. Their learned outputs control translation and the
gripper; rotation deltas are fixed at zero. The displayed success counts come from
held-out rollouts recorded in each task's `metrics.json`, not training loss.

Analyze captures features without changing native policy decisions. Supported
native action heads expose their hidden input during the existing forward pass;
queued actions do not duplicate the previous hidden vector. The bundled scripted
starters expose the verified pure 128-dimensional MLP prefix at the current episode
clock. Other TorchScript exports can return `{"action": tensor, "features": tensor}`
from the same forward/selection call. Unrecognized or unavailable hidden sources
fall back to bounded, explicitly labeled mapped-observation features. Large VLA
checkpoints still require their compatible assets; recognizing a feature hook does
not establish that a particular checkpoint has been evaluated here.

The worker dependency lock uses an immutable official LeRobot source snapshot
(`9e5ee9859480e425a97a6836bb3fee479539beb5`) with its SHA-256 checksum. The published
0.6.1 release caps Pillow and Transformers below their security fixes. This snapshot
accepts Pillow 13 and Transformers 5.22. Setup verifies the archive identity,
dependency constraints, and native policy imports before activation; `--render`
also checks EGL simulation. Torch and CUDA remain shared with the app environment.

Episode means are L2-normalized. PCA is fitted from a bounded reservoir of baseline
decision features and frozen for subsequent variations. DQN uses the full episode
vectors, not their three displayed coordinates. The DQN checkpoint is separate from
the tested robot policy and never updates its weights.

| Format | Required artifact | Inference behavior |
| --- | --- | --- |
| Robomimic | Native `.pth` checkpoint with config, weights, and shape metadata | Native `RolloutPolicy`, observation statistics, recurrent reset, configured frame history, and version-appropriate image preprocessing |
| TorchScript | `torch.jit.save` archive | Batched state tensor or declared observation dictionary; exported `select_action`/`reset` methods when present |
| Stable-Baselines3 | Native `model.save()` `.zip`; select the training algorithm | Native deterministic `predict`, including observation preprocessing and training-space action unscaling; recurrent state retained where its extension is installed |
| LeRobot, including VLA | Complete `pretrained_model` directory or Hugging Face model ID | Installed registered policy class, saved native preprocessor, `select_action`, saved native postprocessor; instruction and mapped cameras/state are supplied |

The isolated runtime currently contains Robomimic 0.3.0. Newer architectures or
checkpoints with 0.5 action normalization need their matching runtime or a
TorchScript export that includes the required conversions. Missing optional
dependencies for a LeRobot type produce an explicit loading error. LeRobot
checkpoints must include `config.json`, `model.safetensors`,
`policy_preprocessor.json`, `policy_postprocessor.json`, and any referenced
processor `.safetensors` files. Legacy checkpoints without saved processors need
native LeRobot migration; fresh normalization statistics are never substituted.

Hugging Face configurations are resolved to an immutable snapshot revision.
Configuration, processor files, and weights then load from that revision.
Native loaders receive the local snapshot, so a loader cannot accidentally ignore
the requested revision. The π0 family uses its native weight-key conversion with
strict weight loading; missing or incompatible weights fail instead of returning
a freshly initialized or partially loaded model.
Configurations with training-time `reinit_modules` are rejected for inference.
Some models also reference separately hosted backbone/tokenizer assets; these
must exist and be compatible with the saved policy. Installing a registered
policy type does not establish compatibility with every embodiment.

Checkpoint directories are never imported as Python packages. Robomimic tensors
load with `weights_only=True`; native NumPy statistics use a narrow constructor
allowlist. SB3 metadata uses a restricted reader for known library classes.
Custom Python policies require an inference export. Transformers dynamic
checkpoint modules are also blocked, including when an upstream VLA constructor
sets `trust_remote_code=True`. Thus VLA variants requiring custom checkpoint
Python, such as some FAST tokenizer configurations, need an installed native
implementation or a suitable export. Native SmolVLA, π0, and π0.5 dispatch is
available when their assets and controller contract match.
Saved LeRobot processor steps must use installed registry names and resolve to
`ProcessorStep` subclasses. Arbitrary processor class/callable imports are rejected.

Mappings are JSON objects. `observation_map` maps **expected policy key → raw
Robosuite key**. `state_keys` gives the exact concatenation order for
`observation.state` (and the vector input of TorchScript/SB3 policies):

```json
{
  "state_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
  "observation_map": {
    "observation.images.front": "agentview_image",
    "observation.images.wrist": "robot0_eye_in_hand_image"
  }
}
```

This is an example of naming/order syntax, not a declaration that those features
match a particular checkpoint. `state_indices` selects coordinates after
concatenation. Optional `state_mean`/`state_std` apply `(state - mean) / std` before
the native preprocessor. Do not repeat normalization already stored in a native
policy or processor. Named Robomimic observations use `observation_map`; the
native `object` key is mapped from Robosuite's `object-state` automatically.

`action_indices` selects policy outputs in controller order. `action_scale` and
`action_offset` then apply `selected_action * scale + offset` **after** native
postprocessing. Their units and signs must come from the trained controller
contract. No coordinate conversion, joint ordering, action units, or gripper
convention is inferred. Missing observations, mismatched camera resolutions,
wrong output dimensions, and nonfinite values fail explicitly. Commands outside
`env.action_spec` fail unless `clip_actions: true` is requested. Native framework
action postprocessing remains intact.

SB3 `VecNormalize` statistics are a separate artifact from the model archive.
Save them with `VecNormalize.save()` as `<checkpoint stem>.vecnormalize.pkl`
(for example, `policy.vecnormalize.pkl` beside `policy.zip`), or as sibling
`vecnormalize.pkl`. Upload this companion with the checkpoint. The adapter uses
the native saved normalization, including variance epsilon and observation
clipping, with updates disabled. Manual `state_mean`/`state_std` together with a
saved normalizer are rejected. If no sidecar exists, a visible warning explains
that the archive cannot tell whether statistics are missing. Affine mean/std
mapping alone cannot reproduce VecNormalize clipping. Custom observation wrappers
that are not represented by the declared mapping require an inference export.
For an SB3 policy with a single image Box observation, map `observation.image` to
the raw Robosuite camera key. Named Dict image observations use their saved names.
SB3 receives the raw camera pixel values and orientation, transposed to the native
channel layout; its own policy performs image scaling.

TorchScript defaults to a float32 `[1, state_dimension]` input and a single action
tensor of shape `[action_dimension]` or `[1, action_dimension]`. Export recurrent
or chunked behavior inside the module and provide an exported `reset()` method.
Without `reset()`, the module reloads from its saved state between episodes.
An optional `select_action()` method is used instead of `forward()` when present.
Tuple outputs containing external hidden state and multi-action chunks are
rejected; the module must return the next action.

Supply TorchScript metadata in an embedded `metadata.json` extra file, or in
`<checkpoint>.metadata.json` (for example, `policy.pt.metadata.json`). The metadata
can declare the environment and state ordering:

```json
{
  "environment": {"task": "Lift", "robots": ["Panda"]},
  "state_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
  "input_features": {
    "observation.state": {"type": "STATE", "shape": [9]}
  },
  "output_features": {"action": {"type": "ACTION", "shape": [7]}}
}
```

For a TorchScript dictionary input, set `torchscript_input` to `dict` in metadata
or mapping and declare every input feature. State features are float32 batched
tensors. Visual features declare `[channels, height, width]` and receive upright
float32 `[1, channels, height, width]` images in `[0,1]`. Raw Robosuite uint8 HWC
images are converted accordingly; additional training-specific transformations
belong in the export. LeRobot receives the same unbatched image convention before
its native saved processor performs batching, tokenization, and normalization.

Native LeRobot environment metadata can be supplied in a separate
`robotics_policy.json` inside its folder, using the same `environment` structure.
The adapter does not insert non-native fields into LeRobot `config.json`.

Run the adapter checks from `web_interface` in the activated isolated runtime:

```bash
.venv/bin/python -E - <<'PY'
from setup_robotics import bootstrap_runtime
bootstrap_runtime()
import unittest
suite = unittest.defaultTestLoader.discover('tests', pattern='test_robotics_policies.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

The tests create real small TorchScript, recurrent Robomimic BC, SB3 PPO/A2C/SAC/TD3/DDPG, and
LeRobot ACT checkpoints. A separate native SmolVLA processor fixture checks saved
normalization, camera orientation, instruction tokenization, and action units with
a tiny local tokenizer and a deterministic model stand-in. These fixtures validate
interfaces and state handling; they are not learned-policy performance validation
and do not download large VLA weights.

Native API references: [Robomimic 0.3 rollout implementation](https://github.com/ARISE-Initiative/robomimic/blob/v0.3.0/robomimic/algo/algo.py),
[SB3 save format](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html),
[LeRobot policy and processor factory](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/src/lerobot/policies/factory.py),
and [SmolVLA native processors](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/src/lerobot/policies/smolvla/processor_smolvla.py).
