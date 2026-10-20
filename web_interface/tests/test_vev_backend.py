"""Vev's application boundary preserves native probabilities and isolates requests."""
import base64
import gc
import importlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

from PIL import Image


def backend_module(test):
    try:
        return importlib.import_module("vev_backend")
    except ModuleNotFoundError as error:
        if error.name == "vev_backend":
            test.fail("The application Vev backend has not been implemented.")
        raise


class VevRequestTests(unittest.TestCase):
    def test_expected_answers_and_malformed_scales_never_enter_native_prompts(self):
        module = backend_module(self)
        valid = {"id": "a", "type": "noul", "question": "Is there a cat?"}
        self.assertEqual(module.validate_rows([valid]), [valid])
        for row in (
            {**valid, "answer": "Yes"},
            {**valid, "expected": "Yes"},
            {**valid, "question": ""},
            {**valid, "type": "score", "criteria": ["low", "high"]},
            {**valid, "type": "score", "criteria": ["", "a", "b", "c", "d"]},
            {**valid, "question": "x" * 32769},
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                module.validate_rows([row])
        with self.assertRaises(ValueError):
            module.validate_rows([{**valid, "id": i} for i in range(257)])

    def test_native_yes_no_order_and_unsmoothed_ordinal_expectation(self):
        module = backend_module(self)
        import torch
        binary = module.answer_from_logits(
            {"id": "yes", "type": "noul", "question": "Cat?"},
            torch.tensor([math.log(3), 0.]),
        )
        self.assertEqual(binary["labels"], ["No", "Yes"])
        self.assertAlmostEqual(binary["probabilities"][0], .25, places=6)
        self.assertAlmostEqual(binary["value"], .75, places=6)
        score = module.answer_from_logits(
            {"id": "rubric", "type": "score", "question": "Quality?",
             "criteria": ["none", "poor", "some", "good", "excellent"]},
            torch.tensor([0., 0., 0., 0., math.log(6)]),
        )
        self.assertEqual(score["labels"], ["0", "1", "2", "3", "4"])
        for actual, expected in zip(score["probabilities"], [.1, .1, .1, .1, .6]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertAlmostEqual(score["value"], 3., places=6)

    def test_nonfinite_or_wrong_shape_logits_are_errors_not_scores(self):
        module = backend_module(self)
        import torch
        row = {"id": 0, "type": "noul", "question": "Cat?"}
        for logits in ([0.], [0., 1., 2.], [float("nan"), 1.],
                       [float("inf"), 1.], [[0., 1.]]):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                module.answer_from_logits(row, torch.tensor(logits))

    def test_selection_pins_default_and_rejects_ambiguous_or_executable_sources(self):
        module = backend_module(self)
        selected = module.normalize_selection({"source": "huggingface", "name": "CountingSheep/vev-4b"})
        self.assertEqual(selected["revision"], "793c449dfb5579655c979f94552176dd41dc597e")
        for selection in (
            {"source": "url", "name": "https://example.test/model"},
            {"source": "huggingface", "name": "../model"},
            {"source": "huggingface", "name": "owner/model", "revision": ""},
            {"source": "local", "name": "/nonexistent/vev-checkpoint"},
        ):
            with self.subTest(selection=selection), self.assertRaises((ValueError, FileNotFoundError)):
                module.normalize_selection(selection)


def checkpoint_fixture(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (path / "vev.json").write_text(json.dumps({"base": ".", "base_model": "test/tiny"}))
    (path / "model.safetensors").write_bytes(b"test checkpoint")
    (path / "processor_config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")


class VevCacheTests(unittest.TestCase):
    def test_incomplete_and_adapter_only_checkpoints_fail_before_model_loading(self):
        module = backend_module(self)
        self.assertTrue(hasattr(module, "validate_checkpoint"), "Checkpoint validation is missing.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_fixture(root)
            self.assertEqual(module.validate_checkpoint(root)["model_type"], "qwen3_5")
            (root / "vev.json").write_text('{"base": "Qwen/base"}')
            with self.assertRaisesRegex(ValueError, "merged"):
                module.validate_checkpoint(root)
            (root / "vev.json").write_text('{"base": "."}')
            (root / "model.safetensors").unlink()
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map": {"tensor": "../outside.safetensors"}}')
            with self.assertRaises(ValueError):
                module.validate_checkpoint(root)

    def test_cache_is_published_only_when_complete_and_failed_save_is_cleaned(self):
        module = backend_module(self)
        self.assertTrue(hasattr(module, "save_compact_checkpoint"), "Atomic compact cache persistence is missing.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            checkpoint_fixture(source)
            destination = root / "compact"

            class SavedModel:
                def get_memory_footprint(self):
                    return 10

                def save_pretrained(self, path, **kwargs):
                    checkpoint_fixture(Path(path))
                    config = {"model_type": "qwen3_5", "quantization_config": {
                        "quant_method": "bitsandbytes", "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_compute_dtype": "bfloat16",
                    }}
                    (Path(path) / "config.json").write_text(json.dumps(config))

            class SavedProcessor:
                def save_pretrained(self, path):
                    pass

            selection = {"name": "owner/model", "revision": "a" * 40}
            module.save_compact_checkpoint(SavedModel(), SavedProcessor(), source, destination, selection)
            self.assertTrue(module.compact_cache_complete(destination, selection))
            (destination / "model.safetensors").write_bytes(b"truncated")
            self.assertFalse(module.compact_cache_complete(destination, selection))
            failure = root / "failed"

            class BrokenModel(SavedModel):
                def save_pretrained(self, path, **kwargs):
                    super().save_pretrained(path, **kwargs)
                    raise OSError("disk full during save")

            with self.assertRaisesRegex(OSError, "disk full"):
                module.save_compact_checkpoint(BrokenModel(), SavedProcessor(), source, failure, selection)
            self.assertFalse(failure.exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["compact", "source"])

    def test_full_checkpoint_staging_prefers_ram_and_removes_only_its_own_directory(self):
        module = backend_module(self)
        self.assertTrue(hasattr(module, "stage_snapshot"), "Bounded full-checkpoint staging is missing.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ram, disk = root / "ram", root / "disk"
            ram.mkdir()
            disk.mkdir()
            unrelated = ram / "other-work"
            unrelated.mkdir()
            seen = []
            def snapshot_download(repo, *, revision, local_dir, allow_patterns, **kwargs):
                seen.append(Path(local_dir))
                self.assertEqual(revision, "a" * 40)
                self.assertNotIn("custom_model.py", allow_patterns)
                checkpoint_fixture(Path(local_dir))
                return str(local_dir)
            siblings = [SimpleNamespace(rfilename=name, size=20) for name in (
                "config.json", "vev.json", "model.safetensors", "processor_config.json",
                "tokenizer.json", "tokenizer_config.json", "custom_model.py")]
            info = SimpleNamespace(siblings=siblings)
            with patch.object(module, "RAM_STAGE_ROOT", ram):
                with module.stage_snapshot("owner/model", "a" * 40, info, disk, snapshot_download) as path:
                    self.assertTrue(path.is_relative_to(ram))
                    self.assertTrue((path / "model.safetensors").exists())
            self.assertFalse(seen[0].exists())
            self.assertTrue(unrelated.exists())

    def test_complete_hub_snapshot_is_reused_without_network_or_temporary_full_copy(self):
        module = backend_module(self)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "existing"
            checkpoint_fixture(snapshot)
            selection = {"source": "huggingface", "name": "owner/model", "revision": "a" * 40}
            def download(name, *, revision, local_files_only):
                self.assertEqual((name, revision, local_files_only), ("owner/model", "a" * 40, True))
                return str(snapshot)
            def network():
                self.fail("A complete pinned local Hub snapshot must not require a network lookup.")
            hub = SimpleNamespace(snapshot_download=download, HfApi=network)
            with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                 patch.object(module, "cache_directory", return_value=root / "compact"):
                with module.checkpoint_source(selection, "cuda") as (path, revision, compact):
                    self.assertEqual(path, snapshot)
                    self.assertEqual(revision, "a" * 40)
                    self.assertTrue(compact.is_relative_to(root / "compact"))
            self.assertTrue((snapshot / "model.safetensors").exists())

    def test_parent_owned_staging_is_used_and_preserved_for_parent_cleanup(self):
        module = backend_module(self)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owned = root / "owned"
            owned.mkdir()
            disk = root / "disk"
            disk.mkdir()
            info = SimpleNamespace(siblings=[
                SimpleNamespace(rfilename=name, size=20) for name in (
                    "config.json", "vev.json", "model.safetensors", "processor_config.json",
                    "tokenizer.json", "tokenizer_config.json",
                )
            ])
            def download(*args, local_dir, **kwargs):
                checkpoint_fixture(Path(local_dir))
            with patch.dict(os.environ, {"VEV_STAGING_DIR": str(owned),
                                         "VEV_COMPACT_STAGING_DIR": str(disk)}):
                with module.stage_snapshot("owner/model", "a" * 40, info, root, download) as path:
                    self.assertEqual(path.parent, owned)
                    self.assertTrue(path.exists())
            self.assertTrue(owned.exists())
            self.assertEqual(list(owned.iterdir()), [])


class VevNativeTests(unittest.TestCase):
    def test_native_cpu_checkpoint_batch_parity_oom_cleanup_and_worker_protocol(self):
        module = backend_module(self)
        self.assertTrue(hasattr(module, "VevBackend"), "Native Vev loading and inference are missing.")
        root = Path(__file__).resolve().parents[1]
        if not (root / ".vev-runtime" / "transformers").is_dir():
            self.skipTest("Run setup_vev.py to enable the isolated native Vev integration checks.")
        result = subprocess.run(
            [sys.executable, "-E", str(root / "tests" / "vev_runtime_case.py")],
            cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


class VevWorkerTests(unittest.TestCase):
    def test_decoded_images_and_answers_are_released_before_waiting_for_next_request(self):
        import vev_worker
        references = []
        image = Image.new("RGB", (8, 8), "red")
        raw = io.BytesIO()
        image.save(raw, format="PNG")
        payload = base64.b64encode(raw.getvalue()).decode("ascii")

        class Answers(list):
            pass

        class Backend:
            identity = {"name": "test"}
            def decide(self, image, questions):
                answers = Answers([{"id": "a", "type": "noul", "probabilities": [.25, .75], "value": .75}])
                references.extend([weakref.ref(image), weakref.ref(answers)])
                return answers

        lines = iter([
            '{"command":"load","selection":{},"device":"cpu"}\n',
            json.dumps({"command": "decide", "image": payload,
                        "questions": [{"id": "a", "type": "noul", "question": "Cat?"}]}) + "\n",
        ])
        test = self
        class Input:
            def readline(self, limit):
                try:
                    return next(lines)
                except StopIteration:
                    gc.collect()
                    test.assertEqual(len(references), 2)
                    test.assertTrue(all(ref() is None for ref in references),
                                    "The worker retained a previous image or answer across requests.")
                    return ""
        output = io.StringIO()
        vev_worker.serve(Input(), output, lambda *_: Backend())
        self.assertEqual(len(output.getvalue().splitlines()), 2)

    def test_non_png_invalid_and_oversized_images_are_rejected(self):
        import vev_worker
        for format in ("PNG", "JPEG"):
            output = io.BytesIO()
            Image.new("RGB", (16, 16)).save(output, format=format)
            payload = base64.b64encode(output.getvalue()).decode("ascii")
            if format == "JPEG":
                with self.assertRaises(ValueError):
                    vev_worker.decode_image(payload)
            else:
                with patch.object(vev_worker, "MAX_IMAGE_PIXELS", 255), self.assertRaises(ValueError):
                    vev_worker.decode_image(payload)
                with patch.object(vev_worker, "MAX_IMAGE_BYTES", 1), self.assertRaises(ValueError):
                    vev_worker.decode_image(payload)
        with self.assertRaises(ValueError):
            vev_worker.decode_image("%%%")


if __name__ == "__main__":
    unittest.main()
