"""Tiny native Qwen3.5 checks in the worker's isolated dependency environment."""
import base64
import gc
import io
import json
import os
from pathlib import Path
import string
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".vev-runtime"), str(ROOT)]
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from PIL import Image
from vev_backend import VevBackend


def save_fixture(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import (
        PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor,
        Qwen3VLVideoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration,
    )
    words = ["<|pad|>", "<|unk|>", "<|endoftext|>", "<|image_pad|>", "<|video_pad|>",
             "<|vision_start|>", "<|vision_end|>", "Yes", "No"]
    words += list(string.ascii_uppercase)
    words += [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    words += [str(i) for i in range(10)]
    words += ["Is", "there", "a", "cat?", "red", "blue", "Quality?", "none", "poor",
              "some", "good", "excellent", "Question:", "Answer", "Scale,", "State:",
              "with", "the", "level", "number", "only.", "Yes", "or", "No.",
              "lowest", "(0)", "to", "highest", "(4):"]
    words = list(dict.fromkeys(words))
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<|unk|>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<|pad|>", unk_token="<|unk|>",
        eos_token="<|endoftext|>", model_max_length=1024,
        additional_special_tokens=words[3:7],
    )
    template = ("{% for message in messages %}{% for content in message.content %}"
                "{% if content.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|> "
                "{% else %}{{content.text}} {% endif %}{% endfor %}{% endfor %}")
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(min_pixels=256, max_pixels=256, patch_size=8,
                                             temporal_patch_size=2, merge_size=2),
        tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(), chat_template=template,
    )
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=len(words), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
            max_position_embeddings=1024, layer_types=["linear_attention", "full_attention"],
            linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
            linear_num_value_heads=2, partial_rotary_factor=1.,
            rope_parameters={"rope_type": "default", "rope_theta": 10000,
                             "mrope_section": [2, 3, 3], "partial_rotary_factor": 1.},
            pad_token_id=0, eos_token_id=2,
        ),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           patch_size=8, temporal_patch_size=2, spatial_merge_size=2,
                           out_hidden_size=32, num_position_embeddings=16),
        image_token_id=3, video_token_id=4, vision_start_token_id=5, vision_end_token_id=6,
        tie_word_embeddings=False,
    )
    torch.manual_seed(194)
    model = Qwen3_5ForConditionalGeneration(config)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    (path / "vev.json").write_text('{"base": ".", "base_model": "fixture/tiny-vev"}')


class NativeVevTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.path = Path(cls.temp.name) / "tiny"
        cls.path.mkdir()
        save_fixture(cls.path)
        cls.selection = {"source": "local", "name": str(cls.path)}

    def setUp(self):
        self.backend = VevBackend(self.selection, "cpu")
        self.image = Image.new("RGB", (16, 16), "red")
        self.rows = [
            {"id": "n1", "type": "noul", "question": "Is there a cat?"},
            {"id": "s", "type": "score", "question": "Quality?",
             "criteria": ["none", "poor", "some", "good", "excellent"]},
            {"id": "n2", "type": "noul", "question": "Is there a red cat?"},
            {"id": "n3", "type": "noul", "question": "Is there a blue cat?"},
            {"id": "n4", "type": "noul", "question": "Is there a cat?"},
        ]

    def test_shared_image_prefix_matches_single_rows_and_does_not_survive_next_image(self):
        expected = [self.backend.decide(self.image, [row])[0] for row in self.rows]
        observed_batches, cache_refs = [], []
        forward = self.backend.model.backbone.forward
        def observe(*args, **kwargs):
            observed_batches.append((kwargs["input_ids"].shape[0], "pixel_values" in kwargs))
            cache = kwargs.get("past_key_values")
            if cache is not None:
                cache_refs.append(weakref.ref(cache))
            return forward(*args, **kwargs)
        with patch.object(self.backend.model.backbone, "forward", new=observe):
            result = self.backend.decide(self.image, self.rows)
        self.assertEqual([row["id"] for row in result], ["n1", "s", "n2", "n3", "n4"])
        for actual, wanted in zip(result, expected):
            torch.testing.assert_close(torch.tensor(actual["probabilities"]), torch.tensor(wanted["probabilities"]),
                                       atol=2e-6, rtol=2e-6)
        self.assertEqual(sum(image for _, image in observed_batches), 1)
        self.assertTrue(all(count <= 4 for count, _ in observed_batches))
        gc.collect()
        self.assertTrue(all(ref() is None for ref in cache_refs))
        self.assertIsNone(self.backend.model.backbone.rope_deltas)
        self.backend.decide(Image.new("RGB", (16, 16), "blue"), self.rows)
        self.assertEqual(self.backend.decide(self.image, self.rows), result)
        self.assertEqual(self.backend.identity["dtype"], "float32")
        self.assertFalse(self.backend.identity["quantized"])

    def test_oom_halves_batch_and_releases_failed_native_cache_before_retry(self):
        expected = self.backend.decide(self.image, self.rows)
        forward = self.backend.model.backbone.forward
        calls, failed_refs, releases = [], [], []
        def limited(*args, **kwargs):
            output = forward(*args, **kwargs)
            count = kwargs["input_ids"].shape[0]
            if "pixel_values" not in kwargs:
                calls.append(count)
            if count > 2:
                failed_refs.extend([weakref.ref(output.last_hidden_state),
                                    weakref.ref(kwargs["past_key_values"])])
                raise torch.cuda.OutOfMemoryError("injected allocation pressure after native inference")
            return output
        def empty_cache():
            gc.collect()
            self.assertIsNone(sys.exc_info()[0])
            self.assertTrue(all(ref() is None for ref in failed_refs))
            releases.append(True)
        with patch.object(self.backend.model.backbone, "forward", new=limited), \
             patch.object(torch.cuda, "empty_cache", new=empty_cache):
            actual = self.backend.decide(self.image, self.rows)
        self.assertEqual(calls, [4, 2, 2, 1])
        self.assertEqual(releases, [True])
        self.assertEqual(self.backend.batch_size, 2)
        for answer, wanted in zip(actual, expected):
            self.assertAlmostEqual(answer["value"], wanted["value"], places=6)

    def test_oom_at_one_propagates_and_later_request_is_clean(self):
        forward = self.backend.model.backbone.forward
        def fail(*args, **kwargs):
            forward(*args, **kwargs)
            raise torch.cuda.OutOfMemoryError("persistent pressure")
        with patch.object(self.backend.model.backbone, "forward", new=fail):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                self.backend.decide(self.image, self.rows[:1])
        self.assertEqual(len(self.backend.decide(self.image, self.rows)), 5)

    def test_worker_serves_multiple_requests_with_protocol_only_stdout_and_exits_on_eof(self):
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        requests = [
            {"command": "decide", "image": encoded, "questions": self.rows},
            {"command": "load", "selection": self.selection, "device": "cpu", "batch_size": 4},
            {"command": "decide", "image": "not base64", "questions": self.rows},
            {"command": "decide", "image": encoded, "questions": self.rows},
            {"command": "decide", "image": encoded, "questions": [self.rows[1]]},
        ]
        completed = subprocess.run(
            [sys.executable, "-E", str(ROOT / "vev_worker.py")],
            input="".join(json.dumps(row) + "\n" for row in requests),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT, timeout=90,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        outputs = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(outputs), 5, completed.stdout)
        self.assertEqual([row["ok"] for row in outputs], [False, True, False, True, True])
        self.assertEqual(outputs[0]["error_type"], "RuntimeError")
        self.assertEqual([row["id"] for row in outputs[3]["answers"]], ["n1", "s", "n2", "n3", "n4"])
        self.assertEqual([row["id"] for row in outputs[4]["answers"]], ["s"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
