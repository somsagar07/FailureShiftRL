# failureshiftRL

A local viewer for model embeddings and input combinations across image
classification, summarization, and image generation. **Analyze** runs the selected
model's encoder locally and displays a 3D projection. Nodes preview images, edited
source text, or filled prompts. This version does **not** generate summaries or
images, compute failure scores, or run an RL loop.

## Run

```bash
cd ~/Desktop/ml_failure/FailureShiftRL/web_interface
bash run.sh
```

The launcher creates its own `.venv` and installs missing requirements before
starting the server. First setup needs internet access to PyPI and the PyTorch CPU
wheel index. It checks the required imports, then opens a browser on this computer.
Keep the terminal open. Press
Ctrl+C to stop. After code changes, stop and restart the server, then refresh the
browser. If port 8000 is occupied, use `bash run.sh --port 8001`.
If the browser does not open automatically, visit `http://127.0.0.1:8000`.

The server uses the Python standard library. Image transformations use Pillow;
embeddings use PyTorch, Transformers and NumPy; downloads use Hugging Face Hub and
datasets. To install/check the environment separately from starting the server:

```bash
bash setup.sh
.venv/bin/python -I check_environment.py
```

The launcher always uses `.venv/bin/python`, so packages from other projects or
your Conda base environment do not affect this app. Setup also runs `pip check`
for dependency conflicts. Setup installs CPU-only PyTorch if it is missing;
no GPU is required. Transformers is constrained to the compatible 4.x API.

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
5. Click **Analyze**. For classification, once the original embeddings finish,
   click a point to dim the other images and compute that image's action combinations.
   Click an action point to inspect the transformed image.
6. For summarization and generation, the single input and its action variants are
   analyzed together. Its action graph expands automatically. Click a point or use
   **Preview** to inspect the edited source text or filled prompt and its similarity.

For a short classification demo, choose **Hugging Face → MobileNetV2** and
**Hugging Face dataset → Cats and dogs**, click **Load input data**, then
**Analyze**. The overview has one point per image. Selecting one image expands
its 125 action nodes when all actions are enabled.
Presets are saved choices, not bundled weights or datasets: first use needs a
download, and later runs reuse the Hugging Face cache.

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
loads its image processor and weights, and extracts features on CPU. Progress
appears above the viewer. Existing Hugging Face files are reused; gated/private
models need your normal Hugging Face authentication.

The image adapter supports compatible Transformers vision `AutoModel` checkpoints,
including the two presets. A local model must be a `save_pretrained` directory with
`config.json`, `preprocessor_config.json`, and weights; arbitrary `.pt` files are not
supported. Custom repository code is disabled. Unsupported models report an error;
the app never substitutes another model or generates placeholder embeddings.

Summarization accepts compatible Transformers encoder-decoder checkpoints, such as
T5 and BART. A local directory needs `config.json`, tokenizer files, and pretrained
weights. The adapter runs only the encoder; source text uses the model's configured
summarization prefix (or `summarize: ` for T5). Inputs beyond the encoder's supported
token limit produce a clear error instead of being silently truncated.

Image generation accepts **Stable Diffusion 1/2 pipelines** in Diffusers directory
format with `model_index.json`, `text_encoder/`, and `tokenizer/`. The two presets
use this format. Only the selected pipeline's CLIP text encoder and tokenizer are
loaded/downloaded; its UNet and VAE are unnecessary for this prompt view. SDXL,
FLUX, single-file checkpoints, and custom-code pipelines are not supported by this
adapter. An incompatible selection reports an error without substituting another
model. Prompt length is checked against the encoder's token limit.

## Explore the graph

The interaction idea comes from [Graphify](https://github.com/Graphify-Labs/graphify).
The viewers run locally without a graph package, external JavaScript, or CDN.

- **Overview:** each blue point is an original input. Drag to rotate in 3D,
  Shift-drag (or right-drag) to pan, and scroll or use **+ / −** to zoom.
- **Expand:** click a point or choose a **Dataset image**. Other originals dim while
  warm-colored action points appear, connected to the selected original. Other
  dimmed images remain selectable.
- **Inspect:** click an action point or choose it from **Preview**. The inspector
  shows the original and altered input, action values, and cosine similarity to
  the original. The dropdown also reaches coincident or occluded points.
- **Reset:** **All inputs** collapses the expansion; **Fit** frames all current
  points. With the canvas focused, arrows rotate, +/− zoom, 0 fits, Escape shows
  all originals. Camera movement never changes the measured point positions.

**Find failure** at the right end of the graph toolbar is a disabled placeholder
for the future failure-discovery step.

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

| Domain | Default combinations | What a node shows |
| --- | --- | --- |
| Image classification | 125 per image: 5 rotations × 5 darkening values × 5 saturation factors | An altered image |
| Summarization | 14 available named text edits | Edited source text, not a model-generated summary |
| Image generation | 27: 3 descriptions × 3 professions × 3 settings | A filled prompt, not a generated image |

Classification values and generation choices match the repository's experiment
configurations. The current interface only removes or restores these values; adding
new values/types is intentionally disabled. Turning a classification transformation
off leaves that property unchanged. Every combination starts from the original
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
All are excluded from Git. After a server restart,
load your input and click Analyze again. The original research scripts
and source images are untouched.

## Checks

```bash
.venv/bin/python -E -m unittest discover -s tests -v
node tests/test_ui.cjs
node tests/test_embedding_view.cjs
node tests/test_embedding_controller.cjs
```

The Python tests check combination counts, real image transformations, folder
loading/uploads, dataset/model boundaries, and HTTP handlers without a listening
socket. JavaScript tests check model/dataset selection, action controls, the
embedding workflow and image/text inspector, and legacy SVG graph interactions.

Embedding tests use real image transformations and NumPy with a deterministic
encoder at the model boundary, checking a shared PCA basis, full-space cosine,
caching and cancellation. Renderer/controller tests exercise camera behavior,
dimming, picking, expansion ordering and stale-response handling.

Text-domain tests cover the shared baseline/variant PCA, full-space similarity,
unchanged edits, task/model cache identities, and the HTTP flow. Adapter tests use
small locally saved Transformers models to check the actual encoder interfaces,
padding, prefixes, input limits, and incompatible checkpoints without downloads.

CPU smoke tests with cached pretrained DistilBART and T5-base completed all 14
text edits, producing 1024D and 768D features respectively. A cached pretrained
MobileNetV2 smoke test covered four images and all 125 variants of one image.
The generation adapter has been checked with tiny local CLIP test models; a
fresh pretrained generation-preset download has not been exercised here.

Native browser rendering and fresh Hugging Face network downloads remain
unverified in the development sandbox. Run the environment check above before a
demo; first-use downloads require network access from the machine running the app.
