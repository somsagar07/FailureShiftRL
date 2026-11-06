# FailureShift

A local workspace for image classification, summarization, image generation,
and Robosuite policy evaluation. In the first three domains,
**Analyze** displays the selected model's embeddings in 3D. **Find failure** runs a
DQN search, evaluates model outputs, and colors measured regions
of the graph. **Adapt** selects action combinations across training inputs and
fine-tunes a separate model checkpoint with explicit targets.
Robosuite evaluates an existing policy in simulation with controlled environment
variations, paired starting states, and recorded rollouts.

## Run

From the repository root:

```bash
cd web_interface
bash run.sh
```

The launcher creates its own `.venv` and installs missing requirements before
starting the server. First setup needs internet access to PyPI and the official
PyTorch wheel index. Setup detects NVIDIA hardware and selects a compatible CUDA
build; computers without it use CPU. It checks the required imports, then opens
a browser on this computer. Keep the terminal open. Press Ctrl+C to stop.
After code changes, stop and restart the server, then refresh the
browser. If port 8000 is occupied, use `bash run.sh --port 8001`.
If the browser does not open automatically, visit `http://127.0.0.1:8000`.

The server uses the Python standard library. Image transformations use Pillow;
embeddings and DQN use PyTorch, Transformers and NumPy; diffusion and LoRA use
Diffusers and PEFT; downloads use Hugging Face Hub and datasets.
To install/check the environment separately from starting the server:

```bash
bash setup.sh
.venv/bin/python -I check_environment.py
```

The launcher always uses `.venv/bin/python`, so packages from other projects or
your Conda base environment do not affect this app. Setup also runs `pip check`
for dependency conflicts. To repair an existing CPU-only installation on an NVIDIA
machine, run `bash setup.sh --cuda`, then restart the server. Verify it with
`.venv/bin/python -I check_environment.py --require-cuda`; this performs a real GPU
tensor operation. `bash setup.sh --cpu` explicitly installs CPU wheels;
`bash setup.sh --dry-run` previews the installation decision. No GPU is required.
The app's Transformers is constrained to the compatible 4.x API.
Diffusers and PEFT are pinned to the versions tested together in this application.
Image generation and fine-tuning can take substantially longer than analysis on CPU.

Vev, the default generation evaluator, runs in a separate process with pinned
Transformers 5.17 dependencies under `.vev-runtime/`; these pins require Python
3.12 or newer. Setup installs app requirements
first, then this overlay with dependency resolution disabled. It shares the app's
Torch/CUDA, Pillow, and Accelerate, reuses matching package versions, and checks the
worker's dependency graph and imports before activating a replacement. It does not
upgrade the app's Transformers, Diffusers, or PEFT, or install another Torch build.
The launcher checks both environments. To repair or inspect just Vev:

```bash
.venv/bin/python -E setup_vev.py
.venv/bin/python -E setup_vev.py --check
```

These commands install/check dependencies without downloading model weights.
Restart the server after repairing dependencies. The runtime is excluded from Git;
keep `requirements-vev.txt` separate from the app's `requirements.txt`.

### Over SSH

From the repository root on the server:

```bash
cd web_interface
.venv/bin/python -E server.py --port 8000
```

On your laptop, keep a second terminal open. Replace `USER@SERVER` with your SSH login:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@SERVER
```

Open `http://127.0.0.1:8000` on your laptop. Match the local and remote port if
changing it; the server validates the browser Host header. The service listens
only on the server's loopback interface.

## Model choices for discovery

The model dropdown groups **general-purpose** models separately from
**dataset-trained classifiers** and **task-trained summarizers**. All use existing
pretrained weights; no project-specific training is required to Analyze or Find
failure. Hugging Face selections download the required files on first use.

- Classification: CLIP Base/Large can classify against candidate class names.
  Names come from dataset label metadata automatically, or from **Candidate
  classes** (comma/newline separated). Include the whole relevant class set;
  scores change when that set changes. CLIP features are its projected image
  features. MobileNet, ResNet, ViT, Beans and CIFAR classifiers use their fixed
  pretrained label vocabularies; ImageNet training does not make their heads
  compatible with arbitrary dataset IDs. Local folders alone do not supply
  reviewed labels; provide labels separately.
- Summarization: Qwen3 0.6B/1.7B, Qwen2.5 Instruct, SmolLM2 Instruct and FLAN-T5
  are general instruction choices. T5 Small and DistilBART are task-trained
  alternatives. Causal instruction models use a summarization prompt, reserve
  room for generated output and return only completion tokens. Analyze uses
  the selected model's final source-token states. Dataset references and the
  existing ROUGE/NLI review criteria still apply.
- Generation: Tiny/Small Stable Diffusion, SD 1.5 and SD Turbo use the existing
  SD1/2 pipeline adapter. Turbo selects four steps at 512 pixels in the UI and
  disables classifier-free guidance. Only CLIP conditioning files are loaded
  for Analyze; Find failure loads the generator and visual evaluator. SDXL,
  SD3, FLUX and custom remote-code pipelines remain unsupported.

New causal-LM and zero-shot CLIP support covers discovery. Their Adapt path is
not enabled yet. General capability does not guarantee correctness: the existing
failure criteria are automated review signals. LAYA's primary release handles
text decisions, and the separate community LAYA-Vision extension needs a custom
adapter; neither is advertised as a supported image classifier here.

## Object detection workspace

Choose the DETR preset or a compatible local or Hugging Face detector, then
load a local image folder or an image dataset. The dataset presets include
COCO, Oxford Pets, and CPPE-5; they default to 20 images. Analyze projects
pooled detector features into the 3D search space. Select an image to expand
darkness and saturation combinations, then select a combination to inspect
the predicted boxes and labels. Find failures runs the existing terminal DQN
against the enabled combinations. Fine-tuning is not available here yet.

The loader resizes images to 224 × 224 pixels. Detection scores compare the
transformed image with the detector's original predictions; dataset bounding
boxes are not loaded or used as ground truth. A changed prediction is a review
candidate, not a confirmed detection error, and newly added detections are
reported separately from the baseline-referenced severity score.

## Robosuite workspace

The optional robotics runtime is installed separately from the app and Vev. It
shares Torch/CUDA without changing their packages. On a new machine, run from
`web_interface` (Python 3.12 or newer):

```bash
.venv/bin/python -E setup_robotics.py --render
.venv/bin/python -E server.py --port 8000
```

`setup_robotics.py --check --render` checks installed dependencies and runs a real
headless Lift/Panda episode fragment. No policy weights are downloaded during
setup. EGL rendering still requires a working graphics driver, even when policy
inference uses CPU. The runtime and temporary installation directories are ignored
by Git. The setup pins are in `requirements-robotics.txt`.

1. Choose **Robosuite** in Workspace, then the task, robot, and controller.
   **Load environment** resets the simulator and shows its scene, cameras,
   observation shapes, and action dimensions. No policy is needed for this preview.
2. Choose a locally trained **Starter policy** to fill its task, controller,
   checkpoint, and observation mapping. Its measured held-out success count is
   shown beside the selection. **Refresh** picks up newly finished training runs.
   Alternatively, select a policy format and its server file/folder, or upload from your browser.
   LeRobot also accepts a Hugging Face repository and revision. Folder uploads
   send relative files in 8 MiB chunks; large files are not loaded into server RAM.
3. For a policy without native Robosuite observation metadata, set the observation
   mapping under **Observation and action mapping**, and enter the task instruction. State ordering,
   camera names, action units, and controller semantics must match training.
4. Select environment variations. The defaults are contact friction, object mass,
   and object X position. Add/remove groups and select their values; ten groups
   are available.
5. **Analyze** runs baseline episodes and saves their initial simulation states.
   Each graph point represents an episode's pooled policy features, projected into
   a fixed PCA basis fitted from baseline decisions. Select an episode to expand
   its variations. Untested children have dashed boundaries and a pending layout;
   evaluated children use their measured features in the same PCA basis.
   The saved reset includes the robot controller's initial references as well as
   physics and task state. Policy and processor state reset before each episode. Changing settings
   requires another Analyze run.
6. **Find failure** opens DQN controls. Set **RL steps**, **New rollout budget**, and
   **Reward criterion**, then press **Start discovery**. Additional options expose
   exploration, learning rate, batch size, replay capacity, and search seed.
   A separate terminal DQN learns which variations fail from each baseline episode's
   full feature vector. The tested robot policy stays fixed. New pairs run from the
   saved initial states; repeated pairs and neutral conditions reuse measured results.
   After the rollout budget is used, remaining RL steps train from valid cached pairs.
   Live counters distinguish steps, new simulator calls, cache hits, and optimizer updates.
7. Select graph nodes to compare outcomes, paired new failures/recoveries, and
   baseline/variation replays. Reports open in a new tab. Cancel preserves partial
   results. Inference and simulation run on the server, including over SSH.
   Loading another environment preview preserves the last evaluation;
   **Show last evaluation** returns to its graph. Edited settings require a fresh
   Analyze before testing variations.

| Format | Input | Compatibility |
| --- | --- | --- |
| Robomimic | Native `.pth` checkpoint | Native observation preprocessing and normalization; task/robot/controller metadata checked. The installed adapter targets stable Robomimic 0.3 checkpoints; unsupported newer architectures or action encodings produce an error. |
| TorchScript | Exported `.pt` policy | One action tensor per call; explicit state order or dictionary schema. Recurrent/chunk state must be handled in exported `select_action` / `reset` methods. |
| Stable-Baselines3 | Native `.zip` checkpoint | Select the matching PPO/A2C/SAC/TD3/DDPG algorithm. Normalization statistics must accompany policies that used VecNormalize, or be supplied explicitly. |
| LeRobot / VLA | Pretrained folder or HF repository | Uses the installed native policy, saved preprocessor/postprocessor, instruction tokenizer, normalization, and action queue. ACT, Diffusion, SmolVLA, PI0, and PI05 imports are checked. A compatible trained embodiment and action mapping are required. |

An example LeRobot observation mapping is:

```json
{
  "observation_map": {"observation.images.front": "agentview_image"},
  "state_keys": ["robot0_joint_pos", "robot0_gripper_qpos"]
}
```

This is an example of mapping syntax, **not a universal state definition**. Use
the exact inputs and order the policy was trained with. A VLA that predicts joint
positions cannot be used as Cartesian delta commands merely because its action
vector has the same length. Choose the trained controller and supply any explicit
index, scale, and offset mapping. Saved processors handle native normalization.
Missing cameras, incompatible shapes, nonfinite values, and out-of-range commands
raise errors before being scored as policy failures. Use checkpoints you trust;
uploaded custom Python modules are not executed.

Task success comes from Robosuite's actual task criterion. The default DQN reward
is 1 for task failure and 0 for success; **New failure after baseline success** only
rewards failures whose matched baseline succeeded. Errors are excluded from reward
learning and measured rates. Observed colors leave untested pairs gray. The optional
**DQN expected reward** view estimates every action after at least one optimizer
update. The graph switches to this view automatically when learned estimates become
available; an explicit choice of observed colors is preserved during polling.
Original episode colors average Q values across all actions, and expanded children
use their individual Q values, including untested actions. These estimates are not
measured outcomes or calibrated probabilities. Inspect shows the selected Q value;
pending positions and missing replays still require actual simulation.
Children are larger when DQN action preference varies less across baseline episodes;
a purple dashed boundary means that variation cannot be measured. Robotics adaptation
is not implemented. Legacy API discovery requests without `search` still run the
paired sweep. Lighting/camera changes require a camera-based policy,
and object changes require applicable task objects. The current variation presets
are not guaranteed valid for every custom task.

Known native policy adapters capture hidden features during existing inference,
without extra decisions or advancing action queues. The bundled TorchScript starters
use their verified 128-dimensional hidden representation. Unsupported architectures
use explicitly labeled observation features. No graph coordinate is claimed to be
a unique state or an invertible representation. Existing analyses created before
episode features were added need a fresh Analyze run for DQN discovery.

Runs live under `runs/robotics/`: compact snapshots, bounded feature samples, reports,
one final frame per measured episode, and at most the first episode plus first
observed failure video per condition. DQN runs retain a checkpoint, step history,
and a float32 prediction matrix; polling fetches one selected episode's estimates.
Uploaded policies live
in its `policies/` directory. A server path avoids copying existing checkpoints.
Deleting a completed run folder removes its saved results; keep the Analyze run
while you still need its matched-state discovery or referenced baseline replays.

### Locally trained starter policies

The starter trainer supports **Lift**, **Stack**, and **PickPlaceCan**, using Panda
and OSC_POSE at 20 Hz. To reproduce training on this machine:

```bash
.venv/bin/python -E train_starter_policies.py --tasks Lift Stack PickPlaceCan
```

It collects state/action demonstrations in the simulator, fits a small neural
policy, and optionally adds DAgger labels from its own rollouts. Exported policies
use learned translation/gripper commands, embedded normalization, and a resettable
episode clock. They do not call the demonstrator during inference. These are state
policies for trying the workspace; they are not image or language-conditioned VLAs.

Only the best development checkpoint for each task is saved as
`runs/robotics/starter-policies/<task>/best.pt`. Training, development, and final
evaluation use disjoint seeds. Each task also has `metrics.json` with full trial
results and `training-config.json` with reproduction settings. The catalog lists
only completed evaluations. Checkpoint selection uses development rollouts, not
final test results. A high success rate on these small tests does not establish
robustness to arbitrary variations.

Only compact weights and metrics are retained; demonstrations stay in memory.
No image datasets or external policy weights are downloaded. Re-running a task
replaces its saved starter after final evaluation; a checkpoint digest ties the
catalog's scores to the evaluated weights. Use `--output /another/directory` to
keep a separate experiment. The workspace reads its default catalog under
`runs/robotics/starter-policies/`.

Local training on 2026-11-01 retained these checkpoints:

| Task | Held-out nominal successes | Variation successes | Checkpoint |
| --- | --- | --- | --- |
| Lift | 20 / 20 | 40 / 40 | `lift/best.pt` |
| Stack | 20 / 20 | 40 / 40 | `stack/best.pt` |
| PickPlaceCan | 20 / 20 | 40 / 40 | `pickplacecan/best.pt` |

Paths are relative to `runs/robotics/starter-policies/`. Each variation test used
10 episodes at mass ×1.5, friction ×0.6, object X offset +0.03 m, and one control
step of action delay. Friction ×0.6 is an extra scripted probe; the UI presets are
×0.5, ×1, and ×1.5. Nominal final seeds were 2000–2019, and variation seeds were
2000–2009. Development used 1000–1009; these seeds were excluded from training.
Stack's three candidates each succeeded on 9/10 development trials; the predefined
return tiebreaker retained its initial behavior-cloning policy. The other two
tasks stopped after 10/10 development successes. All three saved policies are
learned networks, totaling about 279 KiB. These are small simulator evaluations,
not estimates of general robot reliability.

For local engineering verification:

```bash
.venv/bin/python -E -m unittest discover -s tests -p 'test_robotics*.py'
.venv/bin/python -E -m unittest discover -s tests -p 'test_starter_training.py'
.venv/bin/python -E tests/robotics_smoke.py
.venv/bin/python -E tests/robotics_reset_case.py
```

The simulation smoke checks real state restoration, interventions, subprocess
rollouts, and JPEG/MP4 artifacts with a small untrained TorchScript policy. It
tests integration, not task performance or VLA quality. A supplied trained policy
still needs task-specific evaluation.

See [Robotics policy checkpoints](ROBOTICS_POLICIES.md) for exact mapping,
normalization, processor, and checkpoint compatibility requirements.

## Use

1. Select a domain in **Workspace**.
2. Choose **Local model** and enter a saved model directory, or choose **Hugging Face**
   and select a preset model. **Other…** reveals a custom model ID field.
   All three domains require a compatible model to compute embeddings.
3. For classification, select **Local folder** or **Hugging Face dataset**, then load
   the input data. The selected image appears immediately; use **Dataset image**
   to browse samples before Analyze. Summarization accepts a **Hugging Face dataset**
   with source text and reference summaries, or **Paste text** for one article.
   Image generation accepts a prompt dataset or one freely written prompt.
4. Active actions appear as chips, such as **Darkness (5)**. Click **×** to remove
   one, or use **Add action** to bring it back or add an optional transformation. Click a chip's name to choose all
   fixed values or one value; removing and re-adding it keeps that choice.
   Summarization has a chip for each text edit. Generation groups values under
   **Style (5)**, **Lighting (5)**, and **Detail (5)**, with 125 combinations per prompt.
   **Restore defaults** resets the original selections. Optional actions stay off until added.
5. Choose an **Analyze device**: **Automatic** uses CUDA when available, and
   **CPU** or **CUDA GPU** selects it explicitly. Click **Analyze**. For classification,
   the **Search space** graph shows the model embeddings. Once the original
   embeddings finish, click a point to dim the other images and
   compute that image's action combinations.
   Click an action point to inspect the transformed image.
6. Text datasets show one point per article or prompt. Select a point to expand
   its action variants. For a pasted article or one written prompt, the single
   input and its variants are analyzed together and expand automatically. Click a point or use
   **Preview** to inspect the edited source text or styled prompt and its similarity.
7. Click **Find failure** at the bottom right of the graph, set the episode budget
   and evaluation criteria, and click **Start discovery**. Episode, reward, loss,
   exploration rate, measured pair count and selected-model calls update live.
8. Select a measured input/action to inspect the actual model output and score
   components. After a discovery run has finished with measured results, click
   **Adapt** beside **Find failure**. Review the selected actions, supply training
   references if needed, and press **Start fine-tuning**.

For a short classification demo, choose **Hugging Face → MobileNetV2** and
**Hugging Face dataset → Cats and dogs**, enter `20` in **Rows**, click **Load input data**, then
**Analyze**. The overview has one point per image. Selecting one image expands
its 125 action nodes with the default actions enabled.
Leaving **Rows** blank loads the full selected splits.
Presets download model weights on first use. Full dataset loads use the Hugging Face
cache; row-limited loads stream and retain only the selected rows for this server session.

Model presets follow the selected domain:

| Domain | Presets |
| --- | --- |
| Image classification | [MobileNetV2](https://huggingface.co/google/mobilenet_v2_1.0_224), [ResNet18](https://huggingface.co/microsoft/resnet-18), [ViT · Beans](https://huggingface.co/nateraw/vit-base-beans), [ViT · CIFAR-10](https://huggingface.co/aaraki/vit-base-patch16-224-in21k-finetuned-cifar10) |
| Summarization | [T5 Small summarizer](https://huggingface.co/Falconsai/text_summarization), [DistilBART CNN](https://huggingface.co/sshleifer/distilbart-cnn-12-6) |
| Image generation | [Tiny Stable Diffusion](https://huggingface.co/segmind/tiny-sd), [Small Stable Diffusion](https://huggingface.co/segmind/small-sd) |

### Input datasets

- **Local folder:** enter a path on the computer running the server and click
  **Load input data**, or click **Choose folder** to use the browser's folder picker.
  Images in subfolders are included; relative paths distinguish duplicate filenames.
  The folder picker copies files into `runs/datasets/`. Entering a path reads the
  original files without copying them.
- **Hugging Face dataset:** select a dataset or use **Other…** for a custom ID.
  The app fetches the available splits and columns from the repository metadata.
  **Splits** supports multiple selections and defaults to `train` when available,
  otherwise the first available split. **Dataset options** lets you select a
  row range, configuration, image column, or label column; columns default to automatic detection.
  A row count or range uses streaming; leaving Rows blank downloads the full
  selection through the Hugging Face cache. Private/gated datasets require your normal
  Hugging Face authentication.

| Dataset preset | Default split | Configuration | Image column |
| --- | --- | --- | --- |
| [Cats and dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs) | `train` | `default` | `image` |
| [Beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) | `train` | `default` | `image` |
| [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10) | `train` | `plain_text` | `img` |
| [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) | `train` | `default` | `image` |

Summarization presets are [SAMSum](https://huggingface.co/datasets/knkarthick/samsum)
(`default`: `dialogue` → `summary`) and
[CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail)
(`3.0.0`: `article` → `highlights`). Both default to the full `train` split.
Use **Dataset options** to change the configuration, source/reference columns, or
row selection. Automatic column detection requires an unambiguous match; custom
datasets can select their columns explicitly. The selected article and reference
can be inspected before Analyze. Reference coverage is displayed as a count;
discovery requires a nonempty reference for every selected row. Invalid source
rows produce a row-specific error rather than silently disappearing.
Full article datasets stay in the Hugging Face dataset cache. Limited selections
are retained in memory; the app does not precompute every article/action pair.

ImageNet-1k matches both MobileNetV2 and ResNet18. Accept access on its Hugging Face
page and authenticate on the server (`.venv/bin/hf auth login`) before loading.
Train and validation provide labels; test images have no public target labels.
The complete class vocabulary is checked before resolving the two crane classes
and MobileNet's extra background output. A custom or reordered vocabulary still
requires review where class names are ambiguous. For a small ImageNet run, enter
`200` in Rows: the loader streams the first 200 examples and retains their labels
and image bytes without preparing the complete dataset.

These are the full datasets. **Rows** is optional: blank selects all rows, `20`
selects the first 20, and `[20:30]` selects indices 20 through 29. Ranges are
zero-based with an exclusive end. If multiple splits are selected, their rows
are joined in the order shown in the picker before applying the range. No rows
are randomly sampled. Out-of-range or malformed selections report an error.
The image picker retains source names such as `train[20]` or `validation[0]`.
JSON label keys still use zero-based indices within the loaded selection.
For Hugging Face row limits, only the selected rows are materialized in memory,
with bounded Arrow writer batches and no additional dataset cache on disk.
Published split counts allow irrelevant splits to be skipped. A range starting
inside a split may still scan preceding rows. Network reads can include metadata,
Parquet row groups, or archive chunks beyond the exact selected images.
Streaming failures report an error; they never fall back to a full download.
Parquet subsets use synchronous batch readers and close them when the selection
ends, avoiding outstanding scanner work after stopping early. Legacy API split
expressions are resolved from published split sizes before applying Rows.

Supported image formats are PNG, JPEG, WebP, BMP, GIF, and TIFF. Browser folder
uploads allow up to 10,000 files of at most 20 MB each; entering a local path avoids
the upload limit. Image previews accept up to 20 megapixels. Symlinked local files
and directories are skipped. Non-image files are ignored; unreadable images report
an error when viewed.

The Hugging Face adapter uses an
[Image feature column](https://huggingface.co/docs/datasets/image_load) containing
embedded bytes or cached image files. Limited image-folder streams also fetch the
selected remote image files and retain their bytes for later random access.
Plain URL text columns are not image columns.
The image selector lists 100 images at a time with previous/next controls.

For classification, **Analyze** downloads the selected model's needed files,
loads its image processor and weights, and extracts features on the selected device.
Progress appears above the viewer. Existing Hugging Face files are reused; gated/private
models need your normal Hugging Face authentication.

The image adapter supports compatible Transformers vision `AutoModel` checkpoints,
including the classification presets. A local model must be a `save_pretrained` directory with
`config.json`, `preprocessor_config.json`, and weights; arbitrary `.pt` files are not
supported. Custom repository code is disabled. Unsupported models report an error.
Discovery additionally requires a pretrained single-label classification head with
at least two classes. Regression and multilabel heads are explicitly unsupported.

Summarization accepts compatible Transformers encoder-decoder checkpoints, such as
T5 and BART. A local directory needs `config.json`, tokenizer files, and pretrained
weights. Analyze runs only the encoder; source text uses the model's configured
summarization prefix (or `summarize: ` for T5). Dataset articles use the model's
input token limit, including the prefix and special tokens, consistently in
Analyze, discovery, and Adapt. References stay complete, so loss of information
from a long article can affect its measured score. Reports record this input
policy. Pasted articles exceeding the limit produce a clear error.

Image generation accepts **Stable Diffusion 1/2 pipelines** in Diffusers directory
format with `model_index.json`, `text_encoder/`, and `tokenizer/`. The two presets
use this format. Only the selected pipeline's CLIP text encoder and tokenizer are
loaded/downloaded during Analyze. SDXL, FLUX, single-file checkpoints, and custom-code
pipelines are unsupported. Discovery loads the UNet and VAE and accepts standard
safetensors or `.bin` component weights, preferring safetensors when both exist.
`.bin` files use weights-only loading.
Prompt length is checked against the encoder's token limit.

Generation defaults to [GenEval 2](https://github.com/facebookresearch/GenEval2):
800 prompts with visual questions and expected answers. Its official JSONL is
loaded from a pinned revision and cached through Datasets; it publishes one `test`
split. [PartiPrompts](https://huggingface.co/datasets/nateraw/parti-prompts) and
**Other Hugging Face dataset** are also available. Dataset options expose split
selection, Rows, the prompt column, and an optional checks column. Checks accept
`[["How many cats?", "two"]]` or
`[{"question":"How many cats?","answer":"two","skill":"count"}]`.
The full prompt and checks can be inspected before Analyze. Missing dataset checks
remain visible. Discovery uses a visual content rubric for those prompts; it does
not invent reference answers or silently switch to CLIP scoring.

The generation action groups are:

| Group | Fixed values |
| --- | --- |
| Style | Photograph, watercolor, pencil drawing, digital illustration, 3D render |
| Lighting | Even, soft, directional, high-key, low-key |
| Detail | Minimal, simple, moderate, detailed, intricate |

Every combination uses one value from each enabled group. Remove Lighting to get
25 Style × Detail combinations, or remove both additional groups for five styles.
An omitted group adds no instruction; restoring it keeps its prior value selection.
All 125 combinations are enabled by default. Each action preserves the entire
original prompt, including literal braces. Prompts that already request conflicting
presentation settings should be reviewed; the interface does not rewrite their scene.
Long prompts or variants fail clearly instead of silently discarding tokens.

The running server retains one active model family and its evaluation scorers.
Repeated discovery runs reuse loaded weights while creating fresh per-run scoring
settings, seeds, output paths, and measurements. Summarization and generation
reuse the encoder already loaded by Analyze when the devices match. Classification
loads its pretrained head on the first discovery and shares the base model afterward.
Switching model families evicts the previous evaluation resources, and fine-tuning
clears the inference cache before loading trainable weights. Local checkpoint
changes invalidate reuse. Keeping the server running avoids repeated weight loads;
the first download/load and actual inference still take time. Restarting clears
the memory cache but preserves downloaded Hugging Face files on disk.

### Loading and Analyze performance

The Analyze sidebar's **Additional options** contains **Analyze device**, with
Automatic (CUDA when available) selected by default.

Analyze prepares only the model's required configuration, processor/tokenizer,
and one weight format, preferring safetensors. For indexed checkpoints it downloads
the named shards. Hugging Face revisions are pinned before loading, and prompt
analysis continues to download only the diffusion text-conditioning components.
Four download workers and four Transformers checkpoint-loading workers are enabled
by default. Parallel checkpoint loading benefits models split into multiple files;
a single-file model has no shards to load concurrently.

During image generation discovery, UNet and VAE use memory-efficient sequential
loading by default, which was faster for the tested preset. Set
`FAILURESHIFT_COMPONENT_WORKERS=2` to load standard unsharded components concurrently.
That path uses more host RAM to avoid unsafe global initialization state in the
pinned Diffusers version. Sharded components always retain sequential loading.
Set `FAILURESHIFT_COMPONENT_WORKERS=1` to explicitly select the default.
To compare Transformers loading modes, set
`HF_ENABLE_PARALLEL_LOADING=false`; set `HF_PARALLEL_LOADING_WORKERS` to adjust
the shard-worker count. Environment overrides must be present when starting the server.

Image Analyze prepares inputs with four reader threads and keeps one batch ready
while the model processes the current batch. CUDA batches contain up to 64 images;
CPU batches contain up to 8. Images are decoded directly from their source; model
inputs and transformations stay in memory instead of being saved as preview PNGs.
Every selected input is
still embedded, in dataset order, using float32 inference. A batch that exceeds
available GPU memory is retried in smaller pieces, and the smaller working size is
remembered for the rest of that model/device session. A single input that cannot fit
reports an error so you can choose CPU. CPU and CUDA features have separate cache identities.
Discovery initially selects the same device as Analyze. Moving between devices
can produce small numerical differences; these changes do not introduce quantization.

In a 256-image CIFAR-10 comparison with the CIFAR-10 ViT preset on the RTX 4090,
these image preparation and batching changes reduced the embedding stage from
3.16 s to 0.86 s (total Analyze: 5.02 s to 2.87 s). Both runs used fresh model
objects and downloaded weights. The largest normalized-feature difference was
less than 0.0000005. This is a short workload measurement, not a full-dataset estimate.

Open the **Analyze** timing summary below the graph to see the actual device and
wall-clock seconds for **Files / download**, **Weight loading**, **Embeddings**,
and **PCA**. File preparation includes Hub metadata, local cache lookup and any
downloads; it is not a network-only measurement. Total includes remaining input
and result preparation; time waiting behind another model job is shown separately.
Reused stages are marked and take zero new compute time. Timings depend on the
selected model, inputs, and hardware.

Measured on the RTX 4090 with PyTorch 2.11.0 (CUDA 12.8), using fresh model objects
and already-downloaded weights (median of three runs per device):

| Analyze workload | CPU embeddings | CUDA embeddings | CPU total | CUDA total |
| --- | ---: | ---: | ---: | ---: |
| MobileNetV2, 20 repository images | 0.269 s | 0.132 s | 0.312 s | 0.185 s |
| DistilBART, one article and 14 edits | 0.411 s | 0.012 s | 0.998 s | 0.737 s |

These measurements exclude first downloads and CUDA context initialization; they
are not cold-disk benchmarks. Weight loading itself took slightly longer on CUDA
because weights must move to the GPU. With the downloaded Tiny Stable Diffusion
preset and a shared text encoder, component/pipeline loading took 0.243 s sequentially
versus 1.218 s with two workers (one warmup and three alternating runs per mode).
The concurrent path's initialization overhead outweighed its parallelism, so the
sequential path remains the default. These component measurements exclude GPU
transfer and generation.
Use the per-stage timings to compare your actual model and input workload.

## Explore the graph

The interaction idea comes from [Graphify](https://github.com/Graphify-Labs/graphify).
The viewers run locally without a graph package, external JavaScript, or CDN.

- **Overview:** each blue point is an original input. Drag to rotate in 3D,
  Shift-drag (or right-drag) to pan, and scroll or use **+ / −** to zoom.
- **Expand:** click a point or choose a **Dataset image**. Other originals dim while
  warm-colored action points appear, connected to the selected original. Other
  dimmed images remain selectable. Expansion preserves your zoom, pan, and rotation;
  use **Fit** explicitly if you want to frame the new points.
- **Inspect:** click an action point or choose it from **Preview**. The inspector
  shows the original and altered input, action values, and cosine similarity to
  the original. The dropdown also reaches coincident or occluded points.
- **Reset:** **All inputs** collapses the expansion; **Fit** frames all current
  points. With the canvas focused, arrows rotate, +/− zoom, 0 fits, Escape shows
  all originals. Camera movement never changes the measured point positions.

During discovery, **Predicted** is the default graph view. After the first DQN
learning update, every original input receives its mean predicted reward across
all actions. Expanded action nodes use that input's individual `Q(s, a)` values,
including untested actions. Green is low predicted reward and red is high; the
fixed color range is 0–1, matching the task rewards. Values outside that range use
the nearest end color, while the inspector retains their actual values. These
are expected reward estimates, not calibrated failure probabilities. Before any
learning update, the legend shows **Awaiting learning**; a run shorter than the
16-example replay warmup does not produce a trained landscape.

Choose **Measured** to color actual measured severity: green is low, red is high,
and gray is unevaluated. Original points average their measured action pairs;
expanded nodes show the score for that particular input/action. A pulse identifies
the current input/action. Colors never move the points or change the camera.
Selecting an original expands its full action graph during search as well as
afterward. Image embeddings appear in batches, using the analysis model and the
same fitted PCA; text/prompt positions are already available from Analyze.
Interactive expansion takes priority between discovery episodes. Model operations
remain serialized, and discovery resumes when the expansion finishes. In Measured
view, newly plotted actions stay gray until their outputs are evaluated.
Progress updates preserve your selection and camera, including clicks that span a
status refresh. Graph clicks preview inputs; predictions and detailed scores open
only when you choose an entry from **Inspect**. Measured pairs are listed by severity,
highest first, with unevaluated selections last. Live updates preserve your selected
pair. Clear that selector or select
another graph node to close the measured details. The compact legend explains
the active color scale, current and selected nodes, and action sizes.

In Predicted view a thin dashed gray ring marks an estimate without model evidence;
a solid gray ring marks a tested pair (or at least one tested action for an original).
These evidence rings are separate from the purple missing-SD boundary. The browser
fetches predictions only for the focused input and reuses them until the DQN
checkpoint changes. Prediction queries never increase measured coverage or run
the tested model. The final full prediction matrix is saved with the report as a
NumPy array; its rows follow original input indices and its columns follow
`predicted_landscape.action_ids`. Completed snapshots can be read after a server
restart. This does not restore Analyze sessions or resume training.

Action sphere size reflects **variation in DQN action preference across original
inputs**: lower variation means a larger sphere. The inspector gives the action's
mean preference, sample standard deviation, original-input count, and the number of
inputs actually tested with that action. This follows the original experiment's
`softmax(Q)` summary, with the size ordering reversed. The mean is a relative policy
preference, not a failure probability; the spread is not calibrated model confidence.
Equal spreads have equal sizes. Unknown spread uses a neutral-size sphere with a
**purple dashed boundary**, so it cannot be mistaken for measured high or low
variation. A measured zero standard deviation is known, not missing. Selecting an
action adds a separate dark outer ring, preserving its boundary and size during
live updates. Input parents keep their own visual sizes and use the selected color view.

Statistics use one current DQN checkpoint over the original input embeddings and
refresh periodically during search and at completion. Cached repeats do not add
statistical samples. Before the first optimizer update, there are no learned policy
statistics. With a single original image, article, or prompt, the spread is unknown.
The purple boundary remains in those cases, including after training finishes.
This describes input-to-input variation using one DQN; low variation alone does
not establish a reliable failure, and high variation can reveal input-specific
failures. Judge candidates using their measured outputs and test them on fresh
inputs before drawing conclusions about generalization.

## Discovery and rewards

The search uses the [paper's macroscopic DQN structure](https://arxiv.org/abs/2406.07145),
with **new task-specific rewards**. Each episode samples an original input, chooses
one action combination, evaluates it and trains a 64–64 MLP from replay. The state
is the full normalized model embedding, not the 3D PCA coordinate. Epsilon decays
from 1.0 to 0.6 across the budget. Each transition terminates after one action, so
the target is its measured reward, without a bootstrapped future value. A batch
of 16 transitions is required before loss appears; short runs may only collect data.
This implements macroscopic search, not a sequence of incremental corruptions.

Baselines and variants use the same model revision and inference settings.
Deterministic input/action outcomes are cached within each run; repeated episodes
still update the DQN. The interface distinguishes episodes, unique measured pairs
and actual selected-model prediction calls. Graph embedding and NLI/CLIP/visual-judge
calls are additional work.
Each run gets its own seed; generation uses the same noise seed for an input's
baseline and all its variants so changes can be compared. To explore another noise
realization, start a new run with a different seed.

| Domain | Evaluation criteria | Interpretation |
| --- | --- | --- |
| Classification | Target class error, confidence assigned to a wrong class, and loss of confidence in the target | Errors against matched dataset labels or manually supplied targets. |
| Summarization | Reference ROUGE-L overlap, repetition, and optionally NLI source support | Automated quality signals; not a proof of factual correctness. |
| Generation | Maximum measured content, instruction-following, visual-defect, and optional matched-suite disparity severity | One combined search reward; automated candidates with retained evidence. |

Rewards and plotted severities lie in `[0,1]`. Their scales differ across tasks
and do not represent calibrated failure probabilities.

**Classification.** Hugging Face datasets automatically select an unambiguous
`ClassLabel` or conventional scalar label column. **Dataset options → Label column**
lets you choose explicitly when multiple taxonomies or unfamiliar names are present.
Class IDs with `ClassLabel` metadata are decoded
to names. Analyze matches these names to the model's declared classes; it ignores
case, whitespace, underscores and hyphens. The built-in ImageNet mapping also
accepts `cat` (five domestic cat classes) and `dog` (118 domestic dog classes).
Every member must match the model's declared names unambiguously; IDs come from
that model, including checkpoints with a background-class offset. The source and
member names are recorded in `imagenet_pet_labels.json`. Wild cats, wolves,
dingoes, African hunting dogs and unrelated classes are excluded. This mapping
is available offline and does not infer labels from model predictions.
Unnamed numeric labels need an explicit mapping, and missing values need labels.
Existing dataset labels can also be incompatible with the selected model: an
ImageNet classifier cannot predict Beans disease classes or all CIFAR-10 classes.
For those presets, **Use ViT · Beans** or **Use ViT · CIFAR-10** selects a classifier
with matching classes while retaining the loaded images. Analyze then resolves
their labels automatically. The app never changes the selected model silently.

The status distinguishes **dataset labels loaded** from **labels ready** for the
model. Known but incompatible names show **model mismatches**, with examples and
advice to select a matching model. Only unresolved or absent labels need review.
**Upload JSON** (up to 2 MB) merges labels by input index into the current map,
for example `{"0": 281, "1": "goldfish, Carassius auratus"}`. **Edit labels** lets you
type a model class name or #ID (such as #281) beside each thumbnail, or paste/edit
the full JSON. Numeric names remain names; JSON numbers are model IDs.
**View current labels** shows one label, its source (Dataset or Manual) and status
in pages of five. Matched labels are ready immediately and require no approval.
**Restore dataset labels**, inside the editor, discards overrides.
Edits persist across reanalysis of the same dataset/model and reset when either changes.

All inputs need valid targets before discovery starts. Missing labels are never
replaced by the tested model's predictions. Dataset class IDs are never assumed
to equal model class IDs. Let `e` indicate a target mismatch, `w` be the largest probability
of a non-target class, and `d` be the positive relative drop in target probability.
Reward is `0.5e + 0.25ew + 0.25d`. Severity is 1 for a mismatch, otherwise `d`.
Metrics distinguish existing baseline errors from newly induced errors. Baseline
mistakes retain a search signal but are not described as caused by the corruption.
For a category target, the actual top-1 model class must belong to its group.
Target confidence sums that group's probabilities; switching between correct
breeds is not a category error. Reports record the group members. Adapt uses the
same targets: a category loss increases the sum of its allowed class probabilities
without inventing a breed label.

**Summarization.** Datasets use each article's own reference summary. For a pasted
article, enter an explicit reference summary. References are stored with measured
baselines in the report; one global reference cannot replace a dataset's targets.
The fast mode needs only the
selected summarizer. The grounded mode also downloads
[cross-encoder/nli-MiniLM2-L6-H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768)
to estimate support for each generated sentence in overlapping windows of the
actually evaluated input. It does not silently fall back when that scorer is
unavailable. The inspector retains each assessed claim, its strongest source window,
and a diagnostic comparison with whitespace-collapsed source text. A support change
above 0.20, changed lexical window boundaries, or incomplete coverage marks the NLI
assessment unstable or incomplete. Stability is not factuality: wrong-speaker claims
can still receive high, stable NLI support.

Reference quality is `ROUGE-L F1 × distinct-bigram ratio`; reliable grounded
quality is the geometric mean of reference quality and source support. If either
the original or edited assessment fails these consistency/coverage checks, both
sides of the reward use reference quality and the result is marked inconclusive.
The raw NLI measurements remain available for inspection. Empty
summaries score zero. Reward is `0.5(1−quality) + 0.5 max(0, baseline_quality−quality)`;
severity is `1−quality`. Quality below 0.5 or a drop above the chosen threshold is
flagged for review. Reference overlap penalizes valid paraphrases, NLI can miss
unsupported claims, and edits that change facts can legitimately change a summary.
Inspect the source, reference and generated output before treating a flag as a
failure or selecting it for training.

Discovery reports include an evidence status for each evaluated pair:

- **Failure:** a provided-label mismatch under the tested label-preservation
  assumption, or an observed empty summary. Existing baseline errors are distinguished
  from errors introduced by an action.
- **Candidate:** an automatic judge, reference-overlap metric, prediction change,
  or literal-number check raises a review signal.
- **No flag:** the evaluated checks raised no flag; this does not establish correctness.
- **Inconclusive:** input validity, truncation, or unstable/incomplete grounding
  prevents a clear interpretation.

Action guidance identifies edits that can change meaning or hide labelled content.
Text checks also identify changed numeric literals or negation markers—for example,
punctuation removal turning `5-8` into `58`. These are review guards, not proofs
that all remaining actions preserve semantics. If the summarizer truncates an article,
source support uses the visible input and the incomplete-reference comparison is flagged.

The summary counts distinct measured input/action pairs and baseline inputs. Repeated
DQN visits reuse the same measurement and seed; they do not add independent evidence.
Search-selected counts do not estimate dataset-wide prevalence. Predicted graph colors
remain expected rewards, while measured colors are evaluator severity. Vev rating
distributions are retained in the report and inspector; they are not calibrated
failure probabilities. Unconfigured bias comparisons remain explicitly unevaluated.

**Generation.** Find failure runs one combined reward with the independent
[Vev-4B](https://huggingface.co/CountingSheep/vev-4b) visual evaluator by default.
Its weights are released for noncommercial use under
[CC BY-NC 4.0](https://huggingface.co/CountingSheep/vev-4b/blob/793c449dfb5579655c979f94552176dd41dc597e/README.md);
the bundled inference source has its separate Apache 2.0 license and notices.
The judge downloads on first use and runs on the selected device.
CUDA uses NF4 weights with BF16 computation; allow about 3–4 GB for the compact
model cache and about 9 GB of temporary staging space for the original checkpoint.
On Linux, sufficient shared memory can provide the temporary staging space.
CPU execution needs an unquantized merged checkpoint.
No failure category needs to be selected before searching.

`reward = severity = max(content, instructions, visual defects, evaluated bias comparison)`

Each component is in `[0,1]`. Missing and inapplicable components are `null`,
with an explicit reason; they are never recorded as successful zero scores.
Original severity and the positive severity increase remain separate diagnostics,
so an already-failing baseline does not reduce the action's reward. Combined
quality is `1−severity`. The default candidate threshold is **0.5**; change it in
**Additional options**. A zero score never flags a candidate, even at threshold 0.
The same scalar drives DQN updates, measured graph severity, estimated reward
landscapes, and Adapt action selection.

- **Content:** when dataset checks exist, severity is the fraction failed.
  For expected yes/no answers, Vev evaluates the original question without the
  target and selects yes when `P(yes) >= 0.5`; the result is matched to the expected
  answer. For other targets, it verifies whether the quoted expected answer is
  supported by the image, using the same yes threshold. Reports distinguish this
  target verification from an independently generated answer and retain its
  probabilities. Without checks, a visual rubric assesses the prompt's objects,
  counts, attributes, relationships, and requested readable text. Supplied checks
  define that prompt's content coverage.
- **Instructions:** a rubric assesses the enabled Style, Lighting, and Detail
  settings. Removed groups add no requirement; original inputs have no added
  presentation requirements. The most severe instruction violation is requested.
- **Visual defects:** a rubric assesses unintended anatomy, geometry, boundaries,
  textures, and rendering artifacts relative to the requested scene and style.
  Intentional abstraction, surrealism, sparse detail, or lighting are not defects.
- **Bias comparison:** an optional, explicitly configured matched suite measures
  differences in prompt-fulfillment errors across named groups. Its scope is the
  action across that suite, not a demographic judgment about the selected image.

Rubric anchors are 0 none, 1 minor, 2 moderate, 3 major, and 4 severe. Vev returns a
distribution over all five levels; rubric severity is its expected level divided
by 4. The probabilities and expected severity remain in the evidence. Malformed
or nonfinite judge results stop evaluation without adding an observation. Images
withheld by the pipeline safety checker are not graded as black-image failures.
The inspector retains component status, questions, answers, check results, and
score evidence after an input/action is selected.

These are **uncalibrated severity and disparity signals**, not failure
probabilities, a universal detector, or the official GenEval 2 score. Different
component scales require validation against reviewed examples. The search can
exploit evaluator mistakes or concentrate on one easy defect; inspect candidates
and rerun with fresh seeds before claiming a persistent failure or improvement.
Explicit `checks` and `clip` API modes and old reports remain supported for
reproducing previous runs; the normal UI always uses `combined`. BLIP remains
available only for legacy question-check API runs. Explicit Qwen judge selections
remain supported for comparing previous combined evaluations.

Generation discovery uses three inference optimizations:

- **CUDA precision:** **Additional options → Generation precision → Automatic**
  renders with float16 on CUDA and float32 on CPU. **Full precision (FP32)** keeps
  float32 rendering for comparisons or models that need it. The report records
  requested and effective precision. The rendering pipeline gets its own text
  encoder when lowering precision, so Analyze retains its float32 embeddings.
  Adapt continues to load separate training weights. Float16 can change pixels
  and judged scores compared with float32, even with the same seed. Nonfinite
  conditioning or rendered pixels stop evaluation instead of becoming failures.
- **Batched visual review:** independent content and rubric questions for an
  image are submitted together. Vev reuses the image prefix across question
  branches, in batches of up to four. An out-of-memory batch is retried smaller
  without skipping questions; the question wording, answer choices, and native
  readout stay the same. Cancelling Vev stops its worker and releases its model
  resources. BLIP only batches questions of equal token length to preserve its
  decoder behavior.
- **Conditioning reuse:** full prompt conditioning and the empty negative prompt
  are reused in a bounded **8 MiB CPU memory cache**. This is separate from the
  pooled embeddings used in the graph. Repeated seeds can reuse conditioning;
  each new image still goes through all selected diffusion steps. The cache
  writes no files and is released with inference resources before training.

These follow the supported [Diffusers precision options](https://huggingface.co/docs/diffusers/v0.30.3/en/optimization/fp16)
and [Transformers batched generation behavior](https://huggingface.co/docs/transformers/main/en/llm_tutorial#padding-side).
The selected model, scheduler, step count, image resolution, checks, and reward
formula remain explicit. Actual speed depends on model size, questions per image,
GPU memory, and other processes using the GPU.

### Optional matched bias comparison

In **Find failure → Bias comparison**, supply a JSON suite and the number of
repeats per case (default 3, allowed 2–16). For example:

```json
{
  "name": "Scientist scene quality",
  "reference": "equal_error_rates",
  "cases": [{
    "name": "Laboratory objects",
    "groups": [
      {"name": "Woman prompt", "prompt": "A woman scientist beside a blue flask.",
       "checks": [{"question": "Is there a blue flask?", "answer": "yes"}]},
      {"name": "Man prompt", "prompt": "A man scientist beside a blue flask.",
       "checks": [{"question": "Is there a blue flask?", "answer": "yes"}]}
    ]
  }]
}
```

Review prompts so only the intended group description changes and the checks
represent comparable requirements. Every case needs the same 2–8 unique group
names and independent nonempty checks; suites accept 1–32 named cases.
`equal_error_rates` states the comparison standard explicitly. Group names come
from the suite; identities are not inferred from generated images.

Every case/repeat uses the same noise seed across groups, with the selected action
applied to each prompt. The score is the largest group mean content error minus
the smallest, equally weighting cases and repeats. Reports include per-group
counts/rates, prompts, seeds, and observed check answers. This is an observed
quality disparity in the supplied suite; it does not establish causal bias,
representational fairness, or statistical significance. Small suites are noisy.

A newly evaluated action adds `cases × groups × repeats` generation calls.
The original-input suite is evaluated once too. Completed comparison results are
cached once per action within the run, with fresh measurements for a new run or
seed. These attempts are included in model-call totals and separately recorded as
`comparison_model_calls`; visual-judge calls are additional work. Comparison images
are transient and are not saved to disk. Cancellation is checked between samples
and judge batches. Without a suite, bias is explicitly **not evaluated**.

## Adapt

**Adapt** is enabled after discovery finishes with at least one measured result
for the current analysis. It initially selects up to three actions with observed
failures, ranked by failure rate, tested-input coverage, severity, and reward.
Each input/action pair counts once, regardless of repeated RL visits. These are
observed rates on tested inputs, not calibrated failure probabilities or DQN values.
If no failures were observed, the initial selection is empty.

**Find failure** and **Adapt** switch between panels while preserving results and
FT selections. In Adapt, the graph stays visible as you scroll the training controls.
Red outer rings mark the selected **action combinations** wherever they appear in
the graph. Click an action node for **Add to FT actions** or **Remove from FT actions**
beside the selected node; the change applies to all training inputs. The popup
follows the node as you move the graph. **Choose from
the action list** also includes unvisited actions, shown as unevaluated. **View**
in the selection table focuses an example without changing the camera. Removing
an action remains in effect when closing and reopening Adapt. New analysis or
discovery clears the selection.

- **Classification:** each selected combination is applied to every analyzed image,
  retaining the labels used for discovery. **Include original inputs** adds one clean
  example per image (enabled by default). For 50 images and two selected actions,
  that makes 150 training examples. Category labels such as `cat` and `dog` use
  `-log(sum(probabilities of allowed classes))`; individual labels use the usual
  cross-entropy loss. Only the pretrained classification head is updated.
- **Summarization:** use all articles and their paired references from the analyzed
  dataset, the pasted article and its reference, or select
  **Training dataset (JSON)** and upload/paste an array of `{text, summary}` records.
  Each selected action is applied to each article. Check that the reference remains
  correct for its edited source. The dataset plan reads inputs lazily rather than
  storing the Cartesian product. `negate_sentence` requires a corrected reference
  for every record; in JSON, use `action_summaries`, for example
  `{"text":"The filter is working.","summary":"The filter works.","action_summaries":{"3":"The filter does not work."}}`.
  Use the action ID from the selection, not necessarily `3`. Clean originals are
  included by default. The encoder-decoder trains on these supervised pairs.
- **Generation datasets:** upload/paste desired-image JSON such as
  `{"input_index":0,"action_id":1,"image_path":"/data/desired-watercolor.png"}`.
  Indices refer to the loaded selection, starting at zero. Each record trains
  exactly that prompt/action pair with its approved target image; the caption is
  taken from the selected prompt and combination. Use the action ID displayed in
  the inspector; IDs stay stable when values are removed. A subset of pairs is allowed, but
  every selected action needs at least one target. Duplicate pairs and dataset
  prompt overrides are rejected. GenEval 2 supplies checks, not training images.
  For a single written prompt, image paths can instead be entered per action,
  one absolute path **on the server** per line; JSON may optionally include a
  reviewed `prompt` override. Removing an action excludes its records
  from training while preserving the uploaded JSON for later selections.
  Images must be at most 20 MB and
  20 megapixels. Failed generated outputs are never automatically used as targets.
  Training honors the photo's EXIF display orientation and preserves its full scene
  by default. Dimensions are rounded to multiples of eight, so the original aspect
  ratio is approximate. Mixed portrait and landscape targets can share an effective
  batch without cropping or padding. Check that each reference depicts both the
  caption and the selected style, lighting and detail settings; an ordinary photo
  is not automatically a valid target for a different requested style.
  **Adapt → Additional options → Training resolution** defaults to the model's
  native resolution (512 for Tiny Stable Diffusion), independently of discovery's
  preview size. An explicit size overrides this default. **Target framing → Center
  crop** restores square cropping when intentionally requested; review its edges.
  Training updates UNet LoRA, saves the adapter, and saves a fused standalone pipeline.

The panel shows examples per epoch and total example exposure before starting.
There is no fixed training-example count cap: selected combinations are applied
on demand, and only the current chunks are decoded and prepared. Every selected
example is used once per epoch. Training does not write a PNG for every transformed
image. Training JSON and API request bodies remain limited to 20 MB; this transport
limit does not restrict the expanded image/action dataset.

Set epochs and learning rate, then press **Start fine-tuning**. Opening Adapt never
starts training. A missing target or invalid setting is explained beside the button.
Default learning rates are `0.001` for classification, `0.00002` for summarization,
and `0.0001` for generation, consistently in the UI and backend. Explicit values
override these defaults; they are starting points, not guarantees of improvement.
Expand **Additional options** for device, effective batch size, training seed,
weight decay, gradient clipping, and working-data cleanup. Training has its own
seed, independent of discovery. Weight decay defaults to `0.01`; the maximum
gradient norm defaults to `1`. Set the gradient norm to `0` to disable clipping.
Both optimizer controls apply to all three workspaces and are recorded in the report.
The **Training device** defaults to **Automatic**, which trains on CUDA when available
and otherwise on CPU. **CPU** and **CUDA GPU** select a device explicitly; an
unavailable explicit CUDA selection reports an error. All three tasks use the
selected device for model updates. The original model directory is unchanged.

**Batch size** defaults to **Automatic**, with a bounded starting size chosen for
the task and available device memory. A manual size specifies examples per optimizer
update. Smaller chunks accumulate gradients when the batch cannot fit in GPU memory;
this keeps the effective batch size and uses all examples. The last update in an
epoch may contain fewer examples. Progress shows the effective batch, chunk size,
memory adjustments, epoch, examples processed, and optimizer updates. CPU training
also uses bounded chunks rather than preparing the entire dataset in RAM.

If a forward/backward pass runs out of CUDA memory, training retries that update
with smaller chunks. If even one example cannot fit, the job reports an error;
it does not silently reduce the dataset or switch devices. An out-of-memory error
during an optimizer update stops the run because that update may have partially
changed parameters. A smaller model or a GPU with more free memory is needed when
model weights and optimizer state cannot fit.

Progress and loss describe training examples. **Open adaptation report** opens a
new tab with the selected action definitions, observed evidence, input and example
counts, targets, settings, and checkpoint path under `adaptation_selection`.
Generation reports also record the effective target resolution and framing policy.
Job status and saved reports include the actual training device and its name;
`settings.device` records the requested `auto`, `cpu`, or `cuda` selection.
`settings.batch_size` records the requested automatic or manual effective batch;
`effective_batch_size`, `micro_batch_size`, and `oom_retries` record runtime behavior.
Adapt reports store the base targets and a compact training plan, instead of
duplicating one record per transformed example.
Use **Select checkpoint for Analyze** and evaluate on separate held-out inputs to
check whether failures have been reduced.
When choosing epochs or learning rates, use a validation set distinct from the
final test inputs. Compare the same original/action pairs and generation seeds
against the base model, including clean inputs and actions outside the training
selection. Lower training loss alone does not establish reduced failure severity;
retain the original model if the candidate does not meet your validation criteria.

For image generation, optional **Validation prompts** run a paired comparison as
part of Adapt. Enter unseen original scene prompts, one per line (1–64); use at
least 8 for an improvement verdict. The backend rejects normalized duplicates and
overlap with training scenes or effective captions. It evaluates clean prompts and
the selected actions with matching generation settings, seeds, and scorer revisions,
first with the base model and then with the separately loaded saved checkpoint.
Validation uses discovery's rendering resolution and inference steps, while training
uses its own resolution. The API also accepts optional per-prompt `checks` in
`validation_data` and 1–4 `validation_repeats`; the requirement-check metric needs
checks for every validation prompt. Each panel is limited to 4,096 generated images.

The report shows selected-action and clean severity changes, component scores,
failure flags, and paired 95% bootstrap intervals over distinct prompts. `improved`
requires at least a .01 reduction in selected mean severity, an interval wholly
below zero, no increase in selected failure rate, and no more than .01 deterioration
in clean or overall mean severity. A statistically supported deterioration of more
than .01 produces `regressed`; other outcomes are `inconclusive`. These are automated
review signals, not calibrated quality probabilities. Repeated seeds and actions
are not independent prompts; intervals also hold any shared bias probe suite fixed.
Use separate untouched data for final testing and inspect actions outside the
training selection separately.

Validation previews are available from the adaptation report. Each before/after
panel retains at most 64 previews within 32 MiB, in generation order; every score
is retained, and discarded previews have no image link. Selection of previews is
independent of scores. Cancelled or failed
validation retains an already saved candidate checkpoint and its partial report.
No candidate activates automatically. Completed inconclusive or regressed candidates
can still be explicitly selected for inspection; omitted validation reports
`not_requested` and makes no claim of output improvement.

**Clear temporary data after training** is enabled by default. After the checkpoint
and report are saved successfully, it releases the associated feature storage and
preview RAM. Data currently in use by another operation is retained, with a status
message. Failed or cancelled training keeps its working data. Source datasets, base
model downloads, saved checkpoints, discovery evidence, and reports are preserved.
Existing graph coordinates and cached expansions remain available; a new expansion
that needs released features asks you to Analyze again. You can repeat Adapt using
the saved selection and source inputs without rebuilding embeddings first.

The Workspace toggle collapses the sidebar to a compact domain icon rail and
remembers the choice in this browser. Toggling it preserves graph selections and
camera position.

## Jobs and saved outputs

**Cancel** stops between model operations; an in-flight model load or forward pass
must finish first. Graph inspection and progress polling remain available. Analysis,
discovery and fine-tuning share a model execution lock to avoid simultaneous heavy
jobs. If inputs change during a job, its results stay associated with the old analysis
and cancellation remains available; they are never overlaid on the new model.

Each run saves under `runs/discoveries/<job-id>/`: a JSON report, episode JSONL,
policy weights, and generated images when applicable. Report links open in a new
tab, preserving the workspace; binary artifacts may download directly depending
on the browser. **Open analysis report** is also available immediately after
Analyze and shows the current analysis as JSON, including model, coordinates and
imported label coverage. Save the JSON from that tab to keep a copy.
Fine-tuning saves a separate job directory containing `checkpoint/` and training
metadata. Cancelled/error jobs preserve measured results. After a server restart,
saved reports remain available through `GET /api/discoveries`; interrupted jobs are
marked interrupted, not resumed. Analysis is session-local, so analyze again to
start a new search or fine-tune after restarting.

## Embedding geometry

Classification features come from the selected model's pooled output, or the mean
of its final spatial/token features. The presets expose pooled features as documented for
[MobileNetV2](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/mobilenet_v2)
and [ResNet](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/resnet).
An absent learned pooler is bypassed in favor of pretrained hidden features;
other missing base weights cause a clear error.

Summarization uses the attention-mask mean of the final encoder hidden states;
padding tokens do not contribute. These are input representations, not decoder
logits or summary-quality scores. See the [T5 encoder outputs](https://huggingface.co/docs/transformers/v4.47.0/en/model_doc/t5).

Generation uses the attention-mask mean of the pipeline's final CLIP text-encoder
states. It summarizes the [prompt-conditioning representation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img),
not the denoised image latent. Models sharing the same text encoder can produce
the same prompt geometry even when their image generators differ. The original
prompt remains unchanged and is shown with its requirements in the inspector.

All features are L2-normalized. For datasets in all three workspaces, PCA
is fitted **once on original inputs only**; every transformation uses the same
mean and basis. For a pasted article or a single written prompt, PCA is fitted **once on
the input and all selected action variants together**. Changing the action space
in those workspaces refits the view. The footer identifies the fitting population
and retained variance. Three dimensions only approximate the full embedding space;
cosine similarities use the full features. Movement alone is not a failure score.
Small or identical datasets can have fewer than three meaningful axes; unused
axes are zero and some points can overlap.

Original features and completed expansions are cached for the server session.
Normalized features are stored once in float32 and shared through read-only views.
Discovery reads these full-dimensional original features as DQN states;
episodes do not recompute them. First-time action evaluations still need model
predictions to measure failure, and repeated input/action pairs reuse their
outcomes within the discovery run. Full live policy scans are paced by elapsed time
and their measured cost, instead of repeating after every ten episodes. The final
scan still covers every original and action. Large live prediction matrices are
avoided by retaining a frozen small policy and computing inspected rows on demand;
final predictions are written in batches to the discovery artifact.

Working features use a shared 256 MiB RAM budget first, then at most 1 GiB of owned
temporary disk space. Preview images use a shared 32 MiB RAM cache. Configure these
before starting the server with `FAILURESHIFT_FEATURE_RAM_MB`,
`FAILURESHIFT_FEATURE_DISK_MB`, and `FAILURESHIFT_PREVIEW_RAM_MB`; `0` disables that
storage tier. `FAILURESHIFT_PREDICTION_DISK_MB` sets the per-run final prediction
artifact budget (default 1024 MiB). Exhausted budgets produce an explicit error;
they never drop dataset rows. Completed working data is released after successful
fine-tuning by default. Historical run directories are not removed automatically.
Analyze must run again after restarting the server.
Reload input data after changing image files. Model work is serialized and stale
action jobs stop between batches. Small inputs use dense SVD for PCA. Larger image
datasets retain full vectors in the selected storage tier and fit the covariance and project positions
in chunks, retaining every original. Both paths use a 256 MB estimated working-array
budget; very wide feature models can still exceed it and report an error.
No dataset rows are silently sampled. Covariance PCA can lose precision on nearly
degenerate axes; numerically unresolved axes are suppressed.

## Default and optional actions

| Domain | Default combinations | What Analyze previews |
| --- | --- | --- |
| Image classification | 125 per image: 5 rotations × 5 darkening values × 5 saturation factors | An altered image |
| Summarization | 14 enabled named text edits | Edited source text; discovery adds actual summaries |
| Image generation | 125 per prompt: 5 styles × 5 lighting settings × 5 detail levels | Transformed prompts; discovery adds generated images and visual-check results |

Classification values match the repository's experiment configuration. Generation
uses presentation actions; explicitly saved style-only and template spaces remain readable.
The defaults and their action IDs are preserved. **Add action** exposes 19 image
controls, 28 text edits, and 9 generation groups in total:

- Images add brightness, contrast, sharpness, blur, deterministic noise, JPEG
  compression, pixelation, hue shifts, grayscale, inversion, posterization,
  solarization, translation, zoom, and horizontal flipping.
- Text adds case and whitespace variations, contraction expansion/reduction,
  quote styles, sentence formatting, sentence reordering/duplication/removal,
  and nonbreaking spaces.
- Generation adds composition, viewpoint, framing, color palette, focus, and texture.
  These instructions can conflict with a prompt's existing constraints; inspect the
  enabled settings and judge evidence.

The interface supports removing and restoring fixed values. New axes multiply
the Cartesian space; text edits remain separate actions. The browser shows the count
and blocks more than 5,000 combinations per input before analysis. The backend applies
the same guard without materializing an oversized product. Reduce enabled groups or
select individual values to fit the budget; nothing is silently sampled.
Turning a classification transformation off leaves that property unchanged.
Every combination starts from the original
input. Images are resized to 224 × 224, matching the experiment, and transformations
are applied in the order rotation → saturation → darkening.
Optional image transformations then follow the catalog's fixed order. Random image
noise is deterministic for the same input and action, including previews and discovery.

Text edits are bounded adaptations of the original named functions. NLTK/Pattern
tokenization is not reproduced exactly. Noun inflection and the unfinished
verb-tense action are excluded and identified in the UI. Each selected text action
is applied separately with a fixed seed of 42; an unchanged result is labeled.

For classification, every dataset image is paired with every selected action
combination. For example, 100 images with the default actions represent 12,500
combinations. The full space is indexed immediately. The overview embeds all
originals; transformed embeddings are computed only when an image is selected.
PNG previews are encoded only when viewed and kept in the bounded RAM cache.

Preview configuration is saved under `runs/previews/`; uploaded input
images are under `runs/datasets/`, and temporary feature overflow uses owned
directories under `runs/embeddings/`. Local and Hugging Face source images are read
in place, with identity checks that reject changed inputs until they are reloaded.
All are excluded from Git. After a server restart, load your input and click Analyze again.

## Checks

```bash
.venv/bin/python -E -m unittest discover -s tests -v
node --test tests/*.cjs
```

The Python suite covers data loading, transformations, model adapters, CPU/CUDA
execution, PCA, DQN learning, cancellation, saved reports, and fine-tuning. Model
tests use small local checkpoints. JavaScript tests cover form controls, graph
interaction, live updates, and responses arriving after the user changes inputs.

Chrome checks on the RTX 4090 completed Analyze and discovery with pretrained
MobileNetV2, DistilBART, and Tiny Stable Diffusion with an independent CLIP scorer.
The default Falconsai summarizer also completed CUDA analysis and discovery with
the production grounded NLI scorer. First-use model downloads and offline cache
reuse were checked.
Adapt was checked in Chrome with all three pretrained models: 100 classification
examples from the 50-image cats-and-dogs dataset, four summarization examples from
a JSON corpus, and a desired-image LoRA training step. Each saved checkpoint was
reloaded for Analyze and a new discovery run. Action chips, keyboard controls,
sample previews, graph selection, and narrow-screen layouts were also checked.
These checks verify the training and checkpoint workflow, not held-out model improvement.
