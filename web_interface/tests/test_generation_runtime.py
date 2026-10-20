"""Real tiny diffusion checks for conditioning reuse and inference precision."""
import copy
import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import discovery_models as dm
import test_generation_models as fixtures


class GenerationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.GenerationTests.setUpClass()
        cls.addClassCleanup(fixtures.GenerationTests.doClassCleanups)

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.model = {'source': 'local', 'name': str(fixtures.GenerationTests.pipeline),
                      'task': 'generation', 'device': 'cpu'}
        self.config = {'task': 'generation', 'device': 'cpu', 'generation_metric': 'clip',
                       'generation_steps': 2, 'generation_size': 32,
                       '_clip_model': {'source': 'local', 'name': str(fixtures.GenerationTests.clip)}}

    def evaluator(self, **options):
        evaluator = dm.create_evaluator(self.model, {**self.config, **options}, self.scratch.name)
        # CPU tests must not touch a busy CUDA device through Evaluator.close().
        evaluator._shared_resources = True
        self.addCleanup(evaluator.close)
        return evaluator

    def direct_image(self, pipeline, prompt, seed):
        with torch.inference_mode():
            return np.asarray(pipeline(prompt, num_inference_steps=2, height=32, width=32,
                generator=torch.Generator(device='cpu').manual_seed(seed)).images[0])

    def test_cached_conditioning_preserves_seeded_pixels_and_skips_repeated_encoder_work(self):
        evaluator = self.evaluator()
        expected = {seed: self.direct_image(evaluator.pipeline, 'a cat', seed) for seed in (7, 8)}
        with patch.object(evaluator.pipeline.text_encoder, 'forward',
                          wraps=evaluator.pipeline.text_encoder.forward) as encode:
            images = [np.asarray(evaluator._render('a cat', seed)[0]) for seed in (7, 8, 7)]
        self.assertEqual(encode.call_count, 2)
        for seed, actual in zip((7, 8, 7), images):
            np.testing.assert_array_equal(actual, expected[seed])
        self.assertFalse(np.array_equal(images[0], images[1]))

    def test_distinct_prompts_reuse_only_the_empty_negative_conditioning(self):
        evaluator = self.evaluator()
        expected = self.direct_image(evaluator.pipeline, 'a dog', 7)
        with patch.object(evaluator.pipeline.text_encoder, 'forward',
                          wraps=evaluator.pipeline.text_encoder.forward) as encode:
            evaluator._render('a cat', 7)
            actual = np.asarray(evaluator._render('a dog', 7)[0])
        self.assertEqual(encode.call_count, 3)
        np.testing.assert_array_equal(actual, expected)

    def test_cpu_precision_is_reported_without_changing_analyze_weights(self):
        evaluator = self.evaluator()
        metadata = evaluator.describe()
        self.assertEqual(metadata.get('generation_precision'), 'float32')
        self.assertEqual(metadata.get('generation_precision_requested'), 'auto')
        self.assertIs(evaluator.pipeline.text_encoder, evaluator.encoder.model)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in
                            evaluator.encoder.model.parameters()))

    def test_precision_selection_rejects_invalid_values_before_loading_weights(self):
        from generation_runtime import generation_precision
        for requested, device, expected in (('auto', 'cpu', 'float32'), ('auto', 'cuda', 'float16'),
                                             ('float32', 'cpu', 'float32'), ('float32', 'cuda', 'float32')):
            with self.subTest(requested=requested, device=device):
                self.assertEqual(generation_precision(requested, device), expected)
        for value in ('float16', 'bfloat16', '', None, True, 16):
            with self.subTest(value=value), patch.object(dm, 'load_diffusion',
                    side_effect=AssertionError('Invalid precision reached model loading')):
                with self.assertRaisesRegex(ValueError, 'generation_precision'):
                    self.evaluator(generation_precision=value)

    def test_half_rendering_detaches_text_encoder_and_preserves_analyze_features(self):
        from generation_runtime import GenerationRuntime
        encoder = dm.DiffusionPromptEncoder(self.model)
        pipeline, _ = dm.load_diffusion(self.model, encoder=encoder)
        pipeline.set_progress_bar_config(disable=True)
        features = encoder.encode(['a cat', 'a dog'])
        weights = {name: value.clone() for name, value in encoder.model.state_dict().items()}
        # The real tiny modules can exercise conversion on CPU; public CPU
        # precision selection always remains float32 and no CUDA is touched.
        runtime = GenerationRuntime(pipeline, encoder, 'cpu', 'float16')
        self.assertIsNot(pipeline.text_encoder, encoder.model)
        self.assertEqual(pipeline.text_encoder.dtype, torch.float16)
        for component in (pipeline.unet, pipeline.vae):
            self.assertEqual(component.dtype, torch.float16)
        with torch.inference_mode():
            images = [runtime.render('a cat', steps=2, size=32,
                generator=torch.Generator(device='cpu').manual_seed(7)).images[0] for _ in range(2)]
        np.testing.assert_array_equal(np.asarray(images[0]), np.asarray(images[1]))
        np.testing.assert_array_equal(encoder.encode(['a cat', 'a dog']), features)
        self.assertEqual(encoder.model.dtype, torch.float32)
        for name, value in encoder.model.state_dict().items():
            self.assertTrue(torch.equal(value, weights[name]), name)
        rendering_encoder = pipeline.text_encoder
        GenerationRuntime(pipeline, encoder, 'cpu', 'float16')
        self.assertIs(pipeline.text_encoder, rendering_encoder)

    def test_shared_evaluator_reuses_pipeline_and_conditioning_with_new_run_settings(self):
        first = self.evaluator(generation_precision='float32')
        first._render('a cat', 7)
        with patch.object(dm, 'load_diffusion', side_effect=AssertionError('Reloaded shared model')):
            second = dm.create_evaluator(self.model, {**self.config, 'generation_steps': 1},
                self.scratch.name, shared=first)
        second._shared_resources = True
        self.addCleanup(second.close)
        self.assertIs(first._generation_runtime, second._generation_runtime)
        self.assertIs(first.pipeline.text_encoder, second.pipeline.text_encoder)
        self.assertEqual(second.steps, 1)
        with patch.object(second.pipeline.text_encoder, 'forward',
                          wraps=second.pipeline.text_encoder.forward) as encode:
            second._render('a cat', 8)
        self.assertEqual(encode.call_count, 0)
        first.close()
        self.assertEqual(second.describe()['generation_precision_requested'], 'auto')
        self.assertEqual(np.asarray(second._render('a dog', 7)[0]).shape, (32, 32, 3))

    def test_cache_stays_within_cpu_budget_and_evicts_least_recent_prompt(self):
        from generation_runtime import GenerationRuntime
        evaluator = self.evaluator()
        # One tiny full CLIP state is 12 tokens * 12 channels * 4 bytes.
        runtime = GenerationRuntime(evaluator.pipeline, evaluator.encoder, 'cpu', 'float32',
                                    cache_bytes=3 * 12 * 12 * 4)
        with torch.inference_mode(), patch.object(evaluator.pipeline.text_encoder, 'forward',
                wraps=evaluator.pipeline.text_encoder.forward) as encode:
            for prompt in ('a cat', 'a dog', 'a cat', 'a bird', 'a cat', 'a dog'):
                runtime.conditioning(prompt)
                self.assertLessEqual(runtime.cache_bytes, 1728)
            self.assertEqual(encode.call_count, 5)
        retained = [*runtime._prompts.values(), runtime._negative]
        self.assertTrue(all(value.device.type == 'cpu' and not value.requires_grad for value in retained))
        self.assertEqual(runtime.cache_bytes, sum(value.numel() * value.element_size() for value in retained))
        oversized = GenerationRuntime(evaluator.pipeline, evaluator.encoder, 'cpu', 'float32', cache_bytes=1)
        with torch.inference_mode():
            oversized.conditioning('a cat')
        self.assertEqual(oversized.cache_bytes, 0)

    def test_cache_scope_changes_with_pipeline_or_dtype(self):
        from generation_runtime import GenerationRuntime
        evaluator = self.evaluator()
        runtime = GenerationRuntime(evaluator.pipeline, evaluator.encoder, 'cpu', 'float16')
        with torch.inference_mode():
            runtime.conditioning('a cat')
        runtime.pipeline.to(dtype=torch.float32)
        with torch.inference_mode(), patch.object(runtime.pipeline.text_encoder, 'forward',
                wraps=runtime.pipeline.text_encoder.forward) as encode:
            positive, negative = runtime.conditioning('a cat')
            self.assertEqual(encode.call_count, 2)
        self.assertEqual(positive.dtype, torch.float32)
        self.assertEqual(negative.dtype, torch.float32)
        runtime.pipeline = copy.deepcopy(runtime.pipeline)
        with torch.inference_mode(), patch.object(runtime.pipeline.text_encoder, 'forward',
                wraps=runtime.pipeline.text_encoder.forward) as encode:
            runtime.conditioning('a cat')
            self.assertEqual(encode.call_count, 2)

    def test_nonfinite_decoder_output_cannot_be_silently_saved_as_black_pixels(self):
        evaluator = self.evaluator()
        decode = evaluator.pipeline.vae.decode
        def invalid_decode(*args, **kwargs):
            pixels, = decode(*args, **kwargs)
            return (torch.full_like(pixels, float('nan')),)
        with patch.object(evaluator.pipeline.vae, 'decode', side_effect=invalid_decode):
            with self.assertRaisesRegex(ValueError, 'nonfinite image pixels.*float32'):
                evaluator.predict('a cat', seed=7, artifact_name='invalid')
        self.assertFalse((Path(self.scratch.name) / 'invalid.png').exists())

    def test_raw_decoder_nonfinite_values_fail_before_scoring_and_leave_no_hooks(self):
        evaluator = self.evaluator()
        decoder = evaluator.pipeline.vae.decoder
        decode = decoder.forward
        original_hooks = dict(decoder._forward_hooks)
        expected = self.direct_image(evaluator.pipeline, 'a cat', 7)
        for index, invalid in enumerate((float('nan'), float('inf'), -float('inf'))):
            with self.subTest(invalid=invalid):
                def invalid_decode(*args, **kwargs):
                    return torch.full_like(decode(*args, **kwargs), invalid)
                artifact_name = f'invalid_raw_{index}'
                with patch.object(decoder, 'forward', side_effect=invalid_decode), \
                        patch.object(evaluator.scorer, 'forward', wraps=evaluator.scorer.forward) as score:
                    with self.assertRaisesRegex(ValueError, 'nonfinite image pixels.*float32'):
                        evaluator.predict('a cat', seed=7, artifact_name=artifact_name)
                    self.assertEqual(score.call_count, 0)
                self.assertFalse((Path(self.scratch.name) / f'{artifact_name}.png').exists())
                self.assertEqual(dict(decoder._forward_hooks), original_hooks)
                np.testing.assert_array_equal(np.asarray(evaluator._render('a cat', 7)[0]), expected)
                self.assertEqual(dict(decoder._forward_hooks), original_hooks)

    def test_decoder_exception_removes_guard_and_allows_next_render(self):
        evaluator = self.evaluator()
        decoder = evaluator.pipeline.vae.decoder
        original_hooks = dict(decoder._forward_hooks)
        with patch.object(decoder, 'forward', side_effect=RuntimeError('decoder interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'decoder interrupted'):
                evaluator._render('a cat', 7)
        self.assertEqual(dict(decoder._forward_hooks), original_hooks)
        self.assertEqual(np.asarray(evaluator._render('a cat', 7)[0]).shape, (32, 32, 3))
        self.assertEqual(dict(decoder._forward_hooks), original_hooks)

    def test_nonfinite_prompt_conditioning_is_rejected_without_poisoning_cache(self):
        evaluator = self.evaluator()
        encode = evaluator.pipeline.text_encoder.forward
        def invalid_encode(*args, **kwargs):
            result = encode(*args, **kwargs)
            result.last_hidden_state = torch.full_like(result.last_hidden_state, float('nan'))
            return result
        with patch.object(evaluator.pipeline.text_encoder, 'forward', side_effect=invalid_encode):
            with self.assertRaisesRegex(ValueError, 'nonfinite prompt conditioning.*float32'):
                evaluator._render('a cat', 7)
        self.assertEqual(evaluator._generation_runtime.cache_bytes, 0)
        self.assertEqual(np.asarray(evaluator._render('a cat', 7)[0]).shape, (32, 32, 3))

    def test_close_releases_runtime_pipeline_and_cached_tensors(self):
        evaluator = self.evaluator()
        evaluator._render('a cat', 7)
        references = [weakref.ref(evaluator._generation_runtime), weakref.ref(evaluator.pipeline),
                      weakref.ref(evaluator._generation_runtime._negative)]
        evaluator.close()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))


if __name__ == '__main__':
    unittest.main()
