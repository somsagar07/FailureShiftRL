"""Vev image decisions with dependency-free validation for the application.

VevBackend is constructed only in the isolated worker. No framework, model
weights, or downloaded Python are loaded on import.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


DEFAULT_MODEL = "CountingSheep/vev-4b"
DEFAULT_REVISION = "793c449dfb5579655c979f94552176dd41dc597e"
CODE_REVISION = "2d8c403b1c8459d34be214d76c19e0b158a1979c"
ROOT = Path(__file__).resolve().parent
MAX_QUESTIONS = 256
MAX_ROW_CHARS = 32768
MAX_REQUEST_CHARS = 2 * 1024 * 1024
MAX_PIXELS = 1048576
RAM_STAGE_ROOT = Path("/dev/shm")
COMPACT_MARKER = ".complete.json"
NF4_EXCLUDED_MODULE_PATTERNS = [
    r"(?:^|.*\.)visual(?:\.|$)",
    r"(?:^|.*\.)lm_head(?:\.|$)",
]


def normalize_selection(selection):
    if not isinstance(selection, dict):
        raise ValueError("The Vev model selection must be an object.")
    source, name = selection.get("source"), selection.get("name")
    if source not in ("local", "huggingface") or not isinstance(name, str) or not name.strip():
        raise ValueError("Choose a local Vev checkpoint or a Hugging Face model name.")
    if set(selection) - {"source", "name", "revision"}:
        raise ValueError("Unsupported Vev model-selection fields.")
    revision = selection.get("revision")
    if "revision" in selection and (
        not isinstance(revision, str) or not revision.strip()
        or len(revision) > 200 or any(ord(c) < 32 for c in revision)
        or revision.startswith("/") or any(p in (".", "..") for p in revision.split("/"))
    ):
        raise ValueError("The model revision must be a nonempty Hugging Face revision.")
    if source == "local":
        path = Path(name).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"The local Vev checkpoint directory does not exist: {path}")
        name = str(path)
    else:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("The Hugging Face Vev model name must be owner/model.")
        if name == DEFAULT_MODEL and revision is None:
            revision = DEFAULT_REVISION
    return {"source": source, "name": name, **({"revision": revision} if revision is not None else {})}


def validate_rows(rows):
    """Reject target labels, unsupported scales, and oversized work before encoding."""
    if not isinstance(rows, list) or not rows or len(rows) > MAX_QUESTIONS:
        raise ValueError(f"Vev needs between 1 and {MAX_QUESTIONS} questions per request.")
    result, identifiers, characters = [], set(), 0
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"id", "type", "question", "criteria"}:
            raise ValueError("Vev rows allow only id, type, question, and score criteria; target answers are prohibited.")
        identifier = row.get("id")
        if type(identifier) not in (str, int) or identifier in identifiers:
            raise ValueError("Each Vev question needs a unique string or integer id.")
        identifiers.add(identifier)
        if row.get("type") not in ("noul", "score"):
            raise ValueError("Vev question type must be noul or score.")
        texts = [row.get("question")]
        if row["type"] == "score":
            criteria = row.get("criteria")
            if not isinstance(criteria, list) or len(criteria) != 5:
                raise ValueError("Vev score questions require five ordered criteria, for levels 0 through 4.")
            texts += criteria
        elif "criteria" in row:
            raise ValueError("Only Vev score questions accept criteria.")
        if any(not isinstance(text, str) or not text.strip()
               or any(ord(c) < 32 and c not in "\n\r\t" for c in text) for text in texts):
            raise ValueError("Vev questions and criteria must be nonempty text without control characters.")
        size = sum(map(len, texts))
        characters += size
        if size > MAX_ROW_CHARS or characters > MAX_REQUEST_CHARS:
            raise ValueError("Vev question text exceeds the request character limit.")
        result.append({**row, **({"criteria": list(row["criteria"])} if row["type"] == "score" else {})})
    return result


def answer_from_logits(row, logits):
    """Native FP32 softmax, with only Vev's Yes/No order changed for callers."""
    import torch
    size = 2 if row["type"] == "noul" else 5
    if not torch.is_tensor(logits) or logits.ndim != 1 or logits.shape[0] != size:
        raise ValueError("Vev returned an invalid answer-logit shape.")
    logits = logits.float()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Vev returned nonfinite answer logits.")
    probabilities = torch.softmax(logits, dim=0)
    if row["type"] == "noul":
        probabilities = probabilities.flip(0)
    values = probabilities.cpu().tolist()
    total = sum(values)
    if any(not math.isfinite(x) or x < 0 or x > 1 for x in values) or abs(total - 1) > 1e-5:
        raise ValueError("Vev returned an invalid answer probability distribution.")
    # Remove floating-point summation error only. There is no smoothing or
    # fitted temperature; this is the released model's answer distribution.
    values = [x / total for x in values]
    return {
        "id": row["id"], "type": row["type"], "probabilities": values,
        "labels": ["No", "Yes"] if row["type"] == "noul" else ["0", "1", "2", "3", "4"],
        "value": values[1] if row["type"] == "noul" else sum(i * p for i, p in enumerate(values)),
    }


def _read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Vev checkpoint metadata must be an object: {path.name}")
    return value


def validate_checkpoint(path):
    """Only complete, merged Qwen3.5 safetensors checkpoints are supported."""
    path = Path(path)
    config = _read_object(path / "config.json")
    metadata = _read_object(path / "vev.json")
    if metadata.get("base") != "." or (path / "adapter").exists():
        raise ValueError("Vev requires a merged checkpoint with vev.json base='.'; merge the adapter first.")
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Vev requires a compatible Qwen3.5 visual checkpoint.")
    required = {"config.json", "vev.json", "tokenizer.json", "tokenizer_config.json"}
    if not any((path / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        raise ValueError("Vev checkpoint is missing its image processor configuration.")
    if (path / "model.safetensors").is_file():
        required.add("model.safetensors")
    elif (path / "model.safetensors.index.json").is_file():
        required.add("model.safetensors.index.json")
        mapping = _read_object(path / "model.safetensors.index.json").get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Vev checkpoint has an invalid safetensors index.")
        for name in mapping.values():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", name):
                raise ValueError("Vev checkpoint index references an unsupported weight file.")
            required.add(name)
    else:
        raise ValueError("Vev checkpoint has no complete safetensors weights.")
    if any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in required):
        raise ValueError("Vev checkpoint is incomplete; a required model or processor file is missing.")
    return config


def _is_nf4_config(config):
    quantization = config.get("quantization_config", {})
    return (quantization.get("quant_method") == "bitsandbytes"
            and quantization.get("load_in_4bit", quantization.get("_load_in_4bit")) is True
            and quantization.get("bnb_4bit_quant_type") == "nf4"
            and quantization.get("bnb_4bit_use_double_quant") is True
            and quantization.get("bnb_4bit_compute_dtype") == "bfloat16")


def compact_cache_complete(path, selection):
    try:
        path = Path(path)
        marker = _read_object(path / COMPACT_MARKER)
        if (marker.get("schema") != 1 or marker.get("name") != selection["name"]
                or marker.get("revision") != selection["revision"]
                or marker.get("code_revision") != CODE_REVISION):
            return False
        sizes = marker.get("files")
        if not isinstance(sizes, dict) or not sizes:
            return False
        for name, size in sizes.items():
            if not isinstance(name, str) or Path(name).name != name or type(size) is not int:
                return False
            if not (path / name).is_file() or (path / name).stat().st_size != size:
                return False
        return _is_nf4_config(validate_checkpoint(path))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def save_compact_checkpoint(model, processor, source, destination, selection):
    """Publish a complete compact cache atomically; never leave failed partial saves."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = int(model.get_memory_footprint() * 1.1) + 256 * 1024**2
    if shutil.disk_usage(destination.parent).free < required:
        raise OSError("Not enough free disk space to save Vev's compact NF4 cache; free space or set XDG_CACHE_HOME.")
    temporary_parent = Path(os.environ.get("VEV_COMPACT_STAGING_DIR", str(destination.parent)))
    if not temporary_parent.is_dir() or temporary_parent.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError("VEV_COMPACT_STAGING_DIR must exist on the same filesystem as Vev's compact cache.")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=temporary_parent))
    try:
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size="1GB")
        processor.save_pretrained(temporary)
        for original in source.iterdir():
            if original.is_file() and (original.name == "vev.json"
                                       or original.name.startswith(("LICENSE", "NOTICE", "README"))):
                shutil.copy2(original, temporary / original.name)
        config = validate_checkpoint(temporary)
        if not _is_nf4_config(config):
            raise ValueError("The saved Vev checkpoint does not contain the requested NF4 configuration.")
        marker = {
            "schema": 1, "name": selection["name"], "revision": selection["revision"],
            "code_revision": CODE_REVISION,
            "files": {p.name: p.stat().st_size for p in temporary.iterdir() if p.is_file()},
        }
        (temporary / COMPACT_MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
        if not compact_cache_complete(temporary, selection):
            raise ValueError("Vev compact cache failed its completeness check.")
        if destination.exists():
            if compact_cache_complete(destination, selection):
                return
            raise ValueError(f"An incomplete Vev compact cache exists at {destination}; remove it before retrying.")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _snapshot_assets(info):
    names, size = [], 0
    for entry in info.siblings or []:
        name = entry.rfilename
        if ("/" in name or "\\" in name or name.startswith(".")
                or not (name.endswith((".json", ".safetensors", ".jinja", ".txt", ".model"))
                        or name.startswith(("README", "LICENSE", "NOTICE")))):
            continue
        if type(entry.size) is not int or entry.size < 0:
            raise ValueError("Cannot safely stage Vev weights without Hugging Face file sizes.")
        names.append(name)
        size += entry.size
    if "vev.json" not in names or "config.json" not in names or not any(n.endswith(".safetensors") for n in names):
        raise ValueError("The selected Hugging Face repository is not a merged Vev checkpoint.")
    return names, size


@contextmanager
def stage_snapshot(name, revision, info, cache_root, snapshot_download):
    names, size = _snapshot_assets(info)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    # RAM staging avoids persisting both the 9 GB release and its compact cache.
    # Disk staging is allowed only when both copies and working headroom fit.
    ram_required = int(size * 1.1) + 128 * 1024**2
    disk_required = int(size * 1.7) + 512 * 1024**2
    supplied_parent = os.environ.get("VEV_STAGING_DIR")
    if supplied_parent:
        stage_parent = Path(supplied_parent)
        if not stage_parent.is_dir() or shutil.disk_usage(stage_parent).free < ram_required:
            disk_parent = Path(os.environ.get("VEV_COMPACT_STAGING_DIR", str(cache_root)))
            if not disk_parent.is_dir() or shutil.disk_usage(disk_parent).free < disk_required:
                raise OSError("The worker's staging directories do not have enough space for the full Vev checkpoint.")
            stage_parent = disk_parent
    elif RAM_STAGE_ROOT.is_dir() and os.access(RAM_STAGE_ROOT, os.W_OK) \
            and shutil.disk_usage(RAM_STAGE_ROOT).free >= ram_required:
        stage_parent = RAM_STAGE_ROOT
    elif shutil.disk_usage(cache_root).free >= disk_required:
        stage_parent = cache_root
    else:
        raise OSError("Vev's full checkpoint needs temporary space before NF4 conversion. "
                      "Free RAM-backed /dev/shm space or disk space, or choose an existing local checkpoint.")
    stage = Path(tempfile.mkdtemp(prefix="failureshift-vev-", dir=stage_parent))
    try:
        snapshot_download(name, revision=revision, local_dir=str(stage), allow_patterns=names, max_workers=2)
        validate_checkpoint(stage)
        yield stage
    finally:
        shutil.rmtree(stage)


def cache_directory():
    root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
    return root / "failureshiftrl" / "vev"


def _compact_path(root, selection):
    key = json.dumps([selection["name"], selection["revision"], CODE_REVISION, "nf4-bf16-v1"])
    return root / ("nf4-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24])


@contextmanager
def checkpoint_source(selection, device):
    """Resolve immutable assets and retain an owned full snapshot only while loading."""
    if selection["source"] == "local":
        path = Path(selection["name"])
        validate_checkpoint(path)
        yield path, selection.get("revision"), None
        return
    from huggingface_hub import HfApi, snapshot_download
    revision, info = selection.get("revision"), None
    if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        info = HfApi().model_info(selection["name"], revision=revision, files_metadata=True)
        revision = info.sha
    if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("Could not resolve an immutable Vev model revision.")
    resolved = {**selection, "revision": revision}
    root = cache_directory()
    root.mkdir(parents=True, exist_ok=True)
    compact = _compact_path(root, resolved)
    # A cross-process lock avoids duplicate quantization when two requests start
    # workers concurrently. The OS releases it on cancellation or process death.
    import fcntl
    with (root / (compact.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if compact_cache_complete(compact, resolved):
            if device == "cpu":
                raise ValueError("The cached Vev checkpoint uses CUDA NF4. Select CUDA, or provide an "
                                 "unquantized merged local Vev checkpoint for CPU execution.")
            yield compact, revision, None
            return
        if compact.exists():
            raise ValueError(f"The Vev compact cache is incomplete: {compact}. Remove that directory and retry.")
        # A Hub snapshot may exist but contain only configs/tokenizers. Never
        # treat snapshot_download(local_files_only=True) alone as completeness.
        cached = None
        try:
            possible = Path(snapshot_download(selection["name"], revision=revision, local_files_only=True))
            validate_checkpoint(possible)
            cached = possible
        except (OSError, ValueError):
            pass
        if cached is not None:
            yield cached, revision, compact if device.startswith("cuda") else None
            return
        info = info or HfApi().model_info(selection["name"], revision=revision, files_metadata=True)
        with stage_snapshot(selection["name"], revision, info, root, snapshot_download) as path:
            yield path, revision, compact if device.startswith("cuda") else None


def _vendor_modules():
    source = ROOT / "vendor" / "vev"
    manifest = _read_object(source / "SOURCE.json")
    if manifest.get("revision") != CODE_REVISION:
        raise RuntimeError("The bundled Vev inference source revision is not the reviewed revision.")
    expected = {"LICENSE", "NOTICE", "vev/__init__.py", "vev/model.py",
                "vev/readout.py", "vev/state.py", "vev/tokens.py"}
    if set(manifest.get("files", {})) != expected:
        raise RuntimeError("The bundled Vev inference source manifest is incomplete.")
    for name, metadata in manifest["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != metadata["sha256"]:
            raise RuntimeError(f"The bundled Vev inference source failed its integrity check: {name}")
    for name, module in tuple(sys.modules.items()):
        if name == "vev" or name.startswith("vev."):
            location = getattr(module, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(source):
                raise RuntimeError("Vev was already imported from outside the reviewed application vendor directory.")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from vev.model import CheckpointEngine, VevModel, encode_rows, render_row
    from vev.readout import LabelTokens
    from vev.state import serialize_state
    return CheckpointEngine, VevModel, encode_rows, render_row, LabelTokens, serialize_state


class VevBackend:
    """Native Vev readout and shared image prefix with bounded question branches."""

    def __init__(self, selection, device, batch_size=4):
        selection = normalize_selection(selection)
        if type(batch_size) is not int or not 1 <= batch_size <= 4:
            raise ValueError("Vev batch_size must be an integer from 1 through 4.")
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
            raise ValueError("Vev device must be cpu, cuda, or cuda:<index>.")
        import torch
        import transformers
        if transformers.__version__ != "5.17.0":
            raise RuntimeError("Vev requires the isolated Transformers 5.17 runtime. "
                               "Run .venv/bin/python -E setup_vev.py, then restart the Vev worker.")
        self.torch, self.device = torch, torch.device(device)
        self.batch_size = batch_size
        quantized = self.device.type == "cuda"
        if quantized and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Vev but is unavailable. Choose CPU explicitly "
                               "with an unquantized local checkpoint, or repair the CUDA runtime.")
        if quantized:
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("Vev's NF4 backend requires a CUDA device with BF16 support.")
        self.dtype = torch.bfloat16 if quantized else torch.float32
        Engine, Model, self._encode_rows, self._render_row, Labels, self._serialize_state = _vendor_modules()
        with checkpoint_source(selection, device) as (path, revision, compact):
            config = validate_checkpoint(path)
            already_quantized = bool(config.get("quantization_config"))
            if already_quantized and (not quantized or not _is_nf4_config(config)):
                raise ValueError("This Vev checkpoint is quantized. CUDA NF4 with BF16 computation is required; "
                                 "choose an unquantized merged checkpoint for CPU execution.")
            options = dict(local_files_only=True, trust_remote_code=False, weights_only=True,
                           use_safetensors=True, dtype=self.dtype, attn_implementation="sdpa",
                           output_loading_info=True)
            if quantized:
                options["device_map"] = {"": device}
                if not already_quantized:
                    options["quantization_config"] = transformers.BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                        llm_int8_skip_modules=list(NF4_EXCLUDED_MODULE_PATTERNS),
                    )
            full, loading = transformers.AutoModelForImageTextToText.from_pretrained(str(path), **options)
            if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
                raise ValueError("The Vev checkpoint is missing compatible pretrained model weights.")
            if not quantized:
                full = full.to(self.device, dtype=self.dtype)
            full.eval()
            actual_quantized = [name for name, module in full.named_modules()
                                if type(module).__name__ == "Linear4bit"]
            excluded = [name for name, module in full.named_modules()
                        if isinstance(module, torch.nn.Linear)
                        and ("visual" in name.split(".") or "lm_head" in name.split("."))]
            if quantized and not actual_quantized:
                raise RuntimeError("Vev requested NF4, but no quantized model layers were loaded.")
            if any("visual" in name.split(".") or "lm_head" in name.split(".") for name in actual_quantized):
                raise RuntimeError("Vev's vision tower and LM head must remain unquantized.")
            self.processor = transformers.AutoProcessor.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False)
            self.labels = Labels(self.processor.tokenizer)
            if compact is not None:
                save_compact_checkpoint(full, self.processor, path, compact, {**selection, "revision": revision})
            self.model = Model.__new__(Model)
            torch.nn.Module.__init__(self.model)
            self.model.backbone = full.model
            self.model.lm_head_weight = full.lm_head.weight
            self.model.lora_r = 0
            self.model.eval()
            self.model.requires_grad_(False)
            del full
        owner = self

        class CappedEngine(Engine):
            def _chunks(self, names, length, fits):
                return Engine._chunks(names, length, lambda batch, tokens: batch <= owner.batch_size and fits(batch, tokens))

        self.engine = CappedEngine.__new__(CappedEngine)
        self.engine.device, self.engine.model = self.device, self.model
        self.engine.processor, self.engine.tok = self.processor, self.processor.tokenizer
        self.engine.merge, self.engine.labels = self.processor.image_processor.merge_size, self.labels
        self.engine.autocast_dtype = self.dtype if quantized else None
        self.engine.prefix_min_tokens, self.engine.share_min_saved_tokens = 4096, 2048
        self.engine.max_batch_tokens, self.engine.cache_memory_fraction = 32768, 0.5
        self.engine.image_pad_id = self.engine.tok.convert_tokens_to_ids("<|image_pad|>")
        text_config = self.model.backbone.config.text_config
        self.context_limit = min(32768, int(text_config.max_position_embeddings))
        self.identity = {
            "name": selection["name"], "source": selection["source"], "revision": revision,
            "model_revision": revision, "architecture": "vev", "code_revision": CODE_REVISION,
            "dtype": "bfloat16" if quantized else "float32", "device": device,
            "quantized": quantized, "attention": "sdpa", "max_pixels": MAX_PIXELS,
            "max_question_batch": 4, "quantized_modules": actual_quantized,
            "excluded_linear_modules": excluded,
            "quantization": {"method": "bitsandbytes", "bits": 4, "type": "nf4",
                             "compute_dtype": "bfloat16", "double_quant": True,
                             "excluded_module_patterns": list(NF4_EXCLUDED_MODULE_PATTERNS)} if quantized else None,
            "readout": "native_fp32_answer_tokens", "temperature": 1.0,
            "cache_lifetime": "one decide call; image prefix and branch caches released before returning",
        }

    def _clear_image_state(self):
        # Native shared caches are local variables. Qwen also keeps position
        # deltas on the backbone; clear those across images and failed calls.
        if hasattr(self.model.backbone, "rope_deltas"):
            self.model.backbone.rope_deltas = None

    def _logits(self, segments, encoded):
        if len(encoded) == 1:
            return {name: self.engine._logits(row) for name, row in encoded.items()}
        return self.engine._logits_shared_batched(segments, encoded)

    @contextmanager
    def _cpu_reference_kernels(self):
        """Transformers 5.17 selects installed FLA kernels even for CPU inputs.

        Use its own undecorated reference functions on explicitly requested CPU
        execution. The worker is single-threaded and owns one device; restore
        every function even when inference fails.
        """
        if self.device.type != "cpu":
            yield
            return
        import inspect
        from transformers.models.qwen3_5 import modeling_qwen3_5
        names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update")
        original = {name: getattr(modeling_qwen3_5, name) for name in names}
        try:
            for name, function in original.items():
                setattr(modeling_qwen3_5, name, inspect.unwrap(function))
            yield
        finally:
            for name, function in original.items():
                setattr(modeling_qwen3_5, name, function)

    def decide(self, image, rows):
        from PIL import Image
        import gc
        rows = validate_rows(rows)
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("Vev needs a decoded, nonempty PIL image.")
        image = image.convert("RGB")
        if image.width * image.height > MAX_PIXELS:
            scale = math.sqrt(MAX_PIXELS / (image.width * image.height))
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                                 Image.Resampling.BICUBIC)
        self._clear_image_state()
        encoded, logits = {}, None
        try:
            segments, images = self._serialize_state({"image": {"path": "input-image"}}, image_loader=lambda _: image)
            questions = [
                {"type": row["type"], "instructions": row["question"],
                 **({"criteria": row["criteria"]} if row["type"] == "score" else {})}
                for row in rows
            ]
            prepared = self._encode_rows(
                self.processor, [self._render_row(self.processor, segments, q, self.labels) for q in questions], images)
            names = [f"q{i:06d}" for i in range(len(rows))]
            for name, row, question in zip(names, prepared, questions):
                row["answer_ids"] = self.labels.answer_ids(question)
                if int(row["input_ids"].shape[0]) > self.context_limit:
                    raise ValueError("A Vev question exceeds the model context limit; shorten its question or criteria.")
                encoded[name] = row
            with self.torch.inference_mode(), self._cpu_reference_kernels():
                while True:
                    oom = None
                    try:
                        logits = self._logits(segments, encoded)
                    except self.torch.cuda.OutOfMemoryError as error:
                        # Native cache/branch tensors otherwise remain reachable
                        # through the exception's inference-frame traceback.
                        oom = error.with_traceback(None)
                        oom.__context__ = oom.__cause__ = None
                    if oom is None:
                        break
                    logits = None
                    self._clear_image_state()
                    gc.collect()
                    self.torch.cuda.empty_cache()
                    if self.batch_size == 1 or len(rows) == 1:
                        raise oom
                    self.batch_size = max(1, self.batch_size // 2)
                if not isinstance(logits, dict) or set(logits) != set(names):
                    raise ValueError("Vev did not return exactly one answer for every question.")
                return [answer_from_logits(row, logits[name]) for row, name in zip(rows, names)]
        finally:
            encoded.clear()
            logits = None
            self._clear_image_state()
