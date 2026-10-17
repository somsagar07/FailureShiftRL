# failureshiftRL

A local workspace for image classification, summarization, and image generation.
**Analyze** displays the selected model's embeddings in 3D. **Find failure** runs a
DQN search, evaluates model outputs, and colors measured regions
of the graph. Review examples and supply desired targets to fine-tune a separate
model checkpoint.

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
Transformers is constrained to the compatible 4.x API.
Diffusers and PEFT are pinned to the versions tested together in this application.
Image generation and fine-tuning can take substantially longer than analysis on CPU.

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

## Use

1. Select a domain in **Workspace**.
2. Choose **Local model** and enter a saved model directory, or choose **Hugging Face**
   and select a preset model. **Other…** reveals a custom model ID field.
   All three domains require a compatible model to compute embeddings.
3. For classification, select **Local folder** or **Hugging Face dataset**, then load
   the input data. For the other domains, enter text or a prompt template.
4. **Rotation**, **Darkness**, and **Saturation** start checked. Uncheck an action to
   remove it; its values dropdown becomes disabled. Each selected action can use
   all fixed values or one value. Other domains offer their own fixed action choices.
5. Choose an **Analyze device**: **Automatic** uses CUDA when available, and
   **CPU** or **CUDA GPU** selects it explicitly. Click **Analyze**. For classification,
   once the original embeddings finish, click a point to dim the other images and
   compute that image's action combinations.
   Click an action point to inspect the transformed image.
6. For summarization and generation, the single input and its action variants are
   analyzed together. Its action graph expands automatically. Click a point or use
   **Preview** to inspect the edited source text or filled prompt and its similarity.
7. Click **Find failure** at the bottom right of the graph, set the episode budget
   and evaluation criteria, and click **Start discovery**. Episode, reward, loss,
   exploration rate, measured pair count and selected-model calls update live.
8. Select a measured input/action to inspect the actual model output and score
   components. After discovery completes or is cancelled, open **Fine-tune reviewed
   examples**, provide desired targets, and start training when ready.

For a short classification demo, choose **Hugging Face → MobileNetV2** and
**Hugging Face dataset → Cats and dogs**, click **Load input data**, then
**Analyze**. The overview has one point per image. Selecting one image expands
its 125 action nodes when all actions are enabled.
Presets download weights and datasets on first use. Later runs reuse the Hugging Face cache.

Model presets follow the selected domain:

| Domain | Presets |
| --- | --- |
| Image classification | [MobileNetV2](https://huggingface.co/google/mobilenet_v2_1.0_224), [ResNet18](https://huggingface.co/microsoft/resnet-18) |
| Summarization | [T5 Small summarizer](https://huggingface.co/Falconsai/text_summarization), [DistilBART CNN](https://huggingface.co/sshleifer/distilbart-cnn-12-6) |
| Image generation | [Tiny Stable Diffusion](https://huggingface.co/segmind/tiny-sd), [Small Stable Diffusion](https://huggingface.co/segmind/small-sd) |

### Input datasets

- **Local folder:** enter a path on the computer running the server and click
  **Load input data**, or click **Choose folder** to use the browser's folder picker.
  Images in subfolders are included; relative paths distinguish duplicate filenames.
  The folder picker copies files into `runs/datasets/`. Entering a path reads the
  original files without copying them.
- **Hugging Face dataset:** select a preset to fill in its ID, split, configuration,
  and image column. **Other…** accepts a custom dataset ID and settings.
  Split and optional fields remain editable for presets too.
  A single image column is detected automatically. Loading downloads the selected
  dataset through the Hugging Face cache. Private/gated datasets require your normal
  Hugging Face authentication.

| Dataset preset | Default split | Configuration | Image column |
| --- | --- | --- | --- |
| [Cats and dogs: 50 demo images](https://huggingface.co/datasets/hf-internal-testing/cats_vs_dogs_sample) | `train` | Automatic | `image` |
| [Beans: 20 test images](https://huggingface.co/datasets/AI-Lab-Makerere/beans) | `test[:20]` | `default` | `image` |
| [CIFAR-10: 20 test images](https://huggingface.co/datasets/uoft-cs/cifar10) | `test[:20]` | `plain_text` | `img` |

Supported image formats are PNG, JPEG, WebP, BMP, GIF, and TIFF. Browser folder
uploads allow up to 10,000 files of at most 20 MB each; entering a local path avoids
the upload limit. Image previews accept up to 20 megapixels. Symlinked local files
and directories are skipped. Non-image files are ignored; unreadable images report
an error when viewed.

The Hugging Face adapter uses an
[Image feature column](https://huggingface.co/docs/datasets/image_load) containing
embedded bytes or cached image files. URL-only image datasets are not supported.
[Split syntax](https://huggingface.co/docs/datasets/loading) such as `train[:100]`
can limit the selected rows, but may still download the underlying dataset files.
The image selector lists 100 images at a time with previous/next controls.

For classification, **Analyze** downloads the selected model's needed files,
loads its image processor and weights, and extracts features on the selected device.
Progress appears above the viewer. Existing Hugging Face files are reused; gated/private
models need your normal Hugging Face authentication.

The image adapter supports compatible Transformers vision `AutoModel` checkpoints,
including the two presets. A local model must be a `save_pretrained` directory with
`config.json`, `preprocessor_config.json`, and weights; arbitrary `.pt` files are not
supported. Custom repository code is disabled. Unsupported models report an error.
Discovery additionally requires a pretrained single-label classification head with
at least two classes. Regression and multilabel heads are explicitly unsupported.

Summarization accepts compatible Transformers encoder-decoder checkpoints, such as
T5 and BART. A local directory needs `config.json`, tokenizer files, and pretrained
weights. Analyze runs only the encoder; source text uses the model's configured
summarization prefix (or `summarize: ` for T5). Inputs beyond the encoder's supported
token limit produce a clear error instead of being silently truncated.

Image generation accepts **Stable Diffusion 1/2 pipelines** in Diffusers directory
format with `model_index.json`, `text_encoder/`, and `tokenizer/`. The two presets
use this format. Only the selected pipeline's CLIP text encoder and tokenizer are
loaded/downloaded during Analyze. SDXL, FLUX, single-file checkpoints, and custom-code
pipelines are unsupported. Discovery loads the UNet and VAE and accepts standard
safetensors or `.bin` component weights, preferring safetensors when both exist.
`.bin` files use weights-only loading.
Prompt length is checked against the encoder's token limit.

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

CUDA Analyze uses batches and float32 weights. A batch that exceeds available GPU
memory is retried in smaller pieces; a single input that cannot fit reports an
error so you can choose CPU. CPU and CUDA features have separate cache identities.
Discovery initially selects the same device as Analyze. Moving between devices
can produce small numerical differences; these changes do not introduce quantization.

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

During discovery, the colors represent measured severity: green is low, red is
high, and gray is unevaluated. Original points average their measured action pairs;
expanded nodes show the score for that particular input/action. A pulse identifies
the current input/action. Colors never move the points or change the camera.
Use the measured-pair selector to inspect visited examples without requesting
all action embeddings. During search, selecting an original shows its measured
action nodes as they become available. Each new image variant is embedded once
using the analysis model and the same fitted PCA; text/prompt action positions are
already available from Analyze. Previously expanded nodes also remain selectable.
Progress updates preserve your selection and camera, including clicks that span a
status refresh. Expanding every remaining image action can be requested after search.

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
live updates. Input parents keep their own visual sizes and measured-severity colors.

Statistics use one current DQN checkpoint over the original input embeddings and
refresh periodically during search and at completion. Cached repeats do not add
statistical samples. Before the first optimizer update, there are no learned policy
statistics. With a single original image, article, or prompt, the spread is unknown;
the current summarization and generation workspaces each accept one original input.
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
and actual selected-model prediction calls. Graph embedding and NLI/CLIP scorer
calls are additional work.
Each run gets its own seed; generation uses the same noise seed for an input's
baseline and all its variants so changes can be compared. To explore another noise
realization, start a new run with a different seed.

| Domain | Evaluation criteria | Interpretation |
| --- | --- | --- |
| Classification | Target class error, confidence assigned to a wrong class, and loss of confidence in the target | With supplied labels: errors. Without labels: prediction changes requiring review. |
| Summarization | Reference ROUGE-L overlap, repetition, and optionally NLI source support | Automated quality signals; not a proof of factual correctness. |
| Generation | Independent CLIP similarity between the actual generated image and its prompt, plus degradation from the baseline | Prompt-alignment candidates; not demographic bias or aesthetic quality. |

Rewards and plotted severities lie in `[0,1]`. Their scales differ across tasks
and do not represent calibrated failure probabilities.

**Classification.** Supply optional JSON labels indexed by input position, for
example `{"0": 281, "1": "goldfish, Carassius auratus"}`. IDs and exact names must
belong to the selected model's label space; dataset IDs are never silently assumed
to match ImageNet IDs. Without a label, the original prediction becomes the
consistency target. Let `e` indicate a target mismatch, `w` be the largest probability
of a non-target class, and `d` be the positive relative drop in target probability.
Reward is `0.5e + 0.25ew + 0.25d`. Severity is 1 for a mismatch, otherwise `d`.
Metrics distinguish existing baseline errors from newly induced errors. Baseline
mistakes retain a search signal but are not described as caused by the corruption.

**Summarization.** Enter a human reference summary. The fast mode needs only the
selected summarizer. The grounded mode also downloads
[cross-encoder/nli-MiniLM2-L6-H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768)
to estimate support for each generated sentence in overlapping windows of the
actually evaluated input. It does not silently fall back when that scorer is
unavailable. Reference quality is `ROUGE-L F1 × distinct-bigram ratio`; grounded
quality is the geometric mean of reference quality and source support. Empty
summaries score zero. Reward is `0.5(1−quality) + 0.5 max(0, baseline_quality−quality)`;
severity is `1−quality`. Quality below 0.5 or a drop above the chosen threshold is
flagged for review. Reference overlap penalizes valid paraphrases, NLI can miss
unsupported claims, and edits that change facts can legitimately change a summary.
Inspect the source, reference and generated output before treating a flag as a
failure or selecting it for training.

**Generation.** Find failure loads the selected pipeline's UNet/VAE and an independent
[CLIP ViT-B/32 scorer](https://huggingface.co/openai/clip-vit-base-patch32), in addition
to the components loaded during Analyze. This can require substantial downloads.
For cosine similarity `c`, quality is `(c+1)/2`; reward and severity use the same
baseline-aware formula as summarization. A cosine below 0.25 or a normalized quality
drop above the chosen threshold is a review candidate. The 0.25 cutoff is a heuristic,
not a calibrated failure probability. Individual generated images and noise seeds
are retained for inspection; safety-filter results are recorded separately.

## Fine-tune reviewed examples

Select a measured input/action pair and provide its explicit desired target:

- Classification: a selected-model class ID or exact class label.
- Summarization: a corrected reference summary for the edited input.
- Generation: an absolute path **on the server** to a desired image; the selected
  action's prompt supplies its caption. Failed generated images are not automatically
  used as positive training examples.

Add or remove reviewed examples, set epochs and learning rate, and press **Start
fine-tuning**. Classification updates only the existing classifier head. Summarization
trains the encoder-decoder on reviewed text/summary pairs. Generation trains UNet
LoRA, saves the adapter, and saves a fused standalone pipeline that can be selected
as a local model. The original model directory is unchanged. Fine-tuning currently
runs on CPU; use a small generation dataset and epoch budget for an initial test.

Progress and loss describe the training examples. Use **Select checkpoint for Analyze**
and evaluate on separate inputs to check whether failures have been reduced.

## Jobs and saved outputs

**Cancel** stops between model operations; an in-flight model load or forward pass
must finish first. Graph inspection and progress polling remain available. Analysis,
discovery and fine-tuning share a model execution lock to avoid simultaneous heavy
jobs. If inputs change during a job, its results stay associated with the old analysis
and cancellation remains available; they are never overlaid on the new model.

Each run saves under `runs/discoveries/<job-id>/`: a JSON report, episode JSONL,
policy weights, and generated images when applicable. Report links appear in the UI.
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
the same prompt geometry even when their image generators differ. The reference
prompt fills the template with `unique`, `scientist`, and `corporate office`,
regardless of the selected action values; it is shown in the inspector.

All features are L2-normalized. For classification, PCA is fitted **once on original
dataset images only**; every transformed image uses the same mean and basis. For
the single-input summarization and generation workspaces, PCA is fitted **once on
the input and all selected action variants together**. Changing the action space
in those workspaces refits the view. The footer identifies the fitting population
and retained variance. Three dimensions only approximate the full embedding space;
cosine similarities use the full features. Movement alone is not a failure score.
Small or identical datasets can have fewer than three meaningful axes; unused
axes are zero and some points can overlap.

Original features and completed expansions are cached for the server session.
Reload input data after changing image files. Model work is serialized and stale
action jobs stop between batches. Exact PCA has a 256 MB estimated working-array
limit and reports an error when exceeded; it never silently samples the dataset.
Use a smaller dataset split or a smaller feature model if needed.

## Fixed action space

| Domain | Default combinations | What Analyze previews |
| --- | --- | --- |
| Image classification | 125 per image: 5 rotations × 5 darkening values × 5 saturation factors | An altered image |
| Summarization | 14 available named text edits | Edited source text; discovery adds actual summaries |
| Image generation | 27: 3 descriptions × 3 professions × 3 settings | A filled prompt; discovery adds generated images |

Classification values and generation choices match the repository's experiment
configurations. The interface supports removing and restoring fixed values.
Turning a classification transformation off leaves that property unchanged.
Every combination starts from the original
input. Images are resized to 224 × 224, matching the experiment, and transformations
are applied in the order rotation → saturation → darkening.

Text edits are bounded adaptations of the original named functions. NLTK/Pattern
tokenization is not reproduced exactly. Noun inflection and the unfinished
verb-tense action are excluded and identified in the UI. Each selected text action
is applied separately with a fixed seed of 42; an unchanged result is labeled.

For classification, every dataset image is paired with every selected action
combination. For example, 100 images with the default actions represent 12,500
combinations. The full space is indexed immediately. The overview embeds all
originals; transformed embeddings are computed only when an image is selected.
PNG previews are rendered and cached when viewed or embedded.

Preview images and configuration are saved under `runs/previews/`; uploaded input
images are under `runs/datasets/`, and embedding arrays are under `runs/embeddings/`.
All are excluded from Git. After a server restart, load your input and click Analyze again.

## Checks

```bash
.venv/bin/python -E -m unittest discover -s tests -v
node tests/test_ui.cjs
node tests/test_embedding_view.cjs
node tests/test_embedding_controller.cjs
node tests/test_discovery_controller.cjs
node tests/test_embedding_live.cjs
```

The Python suite covers data loading, transformations, model adapters, CPU/CUDA
execution, PCA, DQN learning, cancellation, saved reports, and fine-tuning. Model
tests use small local checkpoints. JavaScript tests cover form controls, graph
interaction, live updates, and responses arriving after the user changes inputs.

Chrome checks on the RTX 4090 completed Analyze and discovery with pretrained
MobileNetV2, DistilBART, and Tiny Stable Diffusion with an independent CLIP scorer.
First-use model downloads and offline cache reuse were also checked. The production
NLI scorer download remains untested; its adapter is covered with a local test model.
