"""Device choices, cache isolation, and measured stages for Analyze."""
import math
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from embeddings import EmbeddingManager, _model_key
import test_embeddings as embedding_fixtures

PixelEncoder = embedding_fixtures.PixelEncoder


class AnalyzeDeviceTests(unittest.TestCase):
    setUp = embedding_fixtures.EmbeddingTests.setUp
    wait = embedding_fixtures.EmbeddingTests.wait
    preview = embedding_fixtures.EmbeddingTests.preview
    expansion = embedding_fixtures.EmbeddingTests.expansion

    def analyze(self, preview, device='cpu', manager=None):
        manager = manager or self.manager
        job = manager.start(preview['id'], device=device)
        result = self.wait(lambda: manager.get(job['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        return result

    def assert_timings(self, job):
        self.assertEqual(set(job['timings_seconds']),
                         {'preparation', 'model_loading', 'embedding', 'pca', 'total'})
        for value in job['timings_seconds'].values():
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0)
        stages = sum(value for name, value in job['timings_seconds'].items() if name != 'total')
        self.assertGreaterEqual(job['timings_seconds']['total'] + 1e-9, stages)
        self.assertGreaterEqual(job['queue_wait_seconds'], 0)

    def test_auto_falls_back_to_cpu_and_reports_actual_device_and_stages(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            job = self.analyze(self.preview(), device='auto')
        self.assertEqual((job['requested_device'], job['device'], job['device_name']), ('auto', 'cpu', 'CPU'))
        self.assertEqual(self.encoders[0].model['device'], 'cpu')
        self.assertFalse(job['model_reused'])
        self.assertFalse(job['features_reused'])
        self.assert_timings(job)

    def test_invalid_or_unavailable_devices_do_not_queue_work(self):
        preview = self.preview()
        for device in ('mps', 'CUDA', 'cuda:0', '', None, True, 1, {}):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, 'device|CPU|CUDA|cpu|cuda'):
                self.manager.start(preview['id'], device=device)
        with patch.object(torch.cuda, 'is_available', return_value=False), \
                self.assertRaisesRegex(ValueError, 'CUDA.*available|CUDA.*unavailable'):
            self.manager.start(preview['id'], device='cuda')
        self.assertFalse(self.manager.jobs)
        self.assertFalse(self.encoders)

    def test_device_is_not_part_of_checkpoint_identity(self):
        selection = {'source': 'huggingface', 'name': 'fixture/vision', 'task': 'classification', 'revision': 'a'*40}
        key = _model_key(selection)
        self.assertEqual(key, _model_key({**selection, 'device': 'cuda', 'requested_device': 'auto',
                                          'device_name': 'fixture GPU', 'timings_seconds': {'total': 1}}))
        self.assertNotEqual(key, _model_key({**selection, 'revision': 'b'*40}))

    def test_cpu_and_cuda_jobs_use_independent_features_and_original_device_for_expansions(self):
        preview = self.preview()
        cpu = self.analyze(preview)
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_name', return_value='Fixture CUDA device'):
            gpu = self.analyze(preview, device='cuda')
            automatic = self.manager.start(preview['id'], device='auto')
        self.assertNotEqual(cpu['id'], gpu['id'])
        self.assertEqual(automatic['id'], gpu['id'])
        self.assertEqual(gpu['device_name'], 'Fixture CUDA device')
        self.assertEqual([encoder.model['device'] for encoder in self.encoders], ['cpu', 'cuda'])
        self.assertFalse(gpu['features_reused'])
        np.testing.assert_array_equal(self.manager.discovery_context(cpu['id'])['states'],
                                      self.manager.discovery_context(gpu['id'])['states'])
        self.assertNotEqual(self.manager._job_bases[cpu['id']]['path'], self.manager._job_bases[gpu['id']]['path'])
        self.expansion(cpu, 0)
        self.assertEqual(self.encoders[-1].model['device'], 'cpu')
        self.assert_timings(gpu)

    def test_cached_dataset_features_skip_loading_embedding_and_pca(self):
        preview = self.preview()
        first = self.analyze(preview)
        second_preview = self.preview(dataset_id=preview['config']['dataset_id'])
        second = self.analyze(second_preview)
        self.assertTrue(second['features_reused'])
        self.assertTrue(second['model_reused'])
        self.assertEqual(len(self.encoders), 1)
        for stage in ('preparation', 'model_loading', 'embedding', 'pca'):
            self.assertEqual(second['timings_seconds'][stage], 0)
        self.assertEqual([point['position'] for point in first['points']],
                         [point['position'] for point in second['points']])
        self.assert_timings(second)

    def test_model_preparation_is_separate_pins_revision_and_runs_only_for_new_weights(self):
        phases = []
        commit = 'c'*40
        prepared = []
        manager = None

        def prepare(selection):
            phases.append(next(iter(manager.jobs.values()))['phase'])
            prepared.append(dict(selection))
            time.sleep(.01)
            return dict(selection), {'revision': commit, 'preparation_seconds': .01, 'files_reused': True}

        def factory(selection):
            phases.append(next(iter(manager.jobs.values()))['phase'])
            time.sleep(.01)
            return PixelEncoder(selection)

        manager = EmbeddingManager(self.previews, self.root/'prepared', encoder_factory=factory, model_preparer=prepare)
        first = self.analyze(self.preview(), manager=manager)
        second = self.analyze(self.preview(), manager=manager)
        self.assertEqual(phases, ['preparing model files', 'loading model weights'])
        self.assertEqual(len(prepared), 1)
        self.assertEqual(manager.discovery_context(first['id'])['model']['revision'], commit)
        self.assertTrue(first['files_reused'])
        self.assertTrue(second['model_reused'])
        self.assertGreaterEqual(first['timings_seconds']['preparation'], .009)
        self.assertGreaterEqual(first['timings_seconds']['model_loading'], .009)
        self.assertEqual(second['timings_seconds']['preparation'], 0)
        self.assertEqual(second['timings_seconds']['model_loading'], 0)
        self.assert_timings(first)
        self.assert_timings(second)

    def test_waiting_for_the_shared_model_lock_is_reported_outside_active_timings(self):
        preview = self.preview()
        with self.manager.model_lock:
            job = self.manager.start(preview['id'], device='cpu')
            time.sleep(.04)
        result = self.wait(lambda: self.manager.get(job['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertGreaterEqual(result['queue_wait_seconds'], .035)
        self.assert_timings(result)

    def test_cuda_memory_pressure_splits_batches_without_changing_model_device_or_features(self):
        class LimitedEncoder(PixelEncoder):
            torch = torch

            def __init__(self, selection):
                super().__init__(selection)
                self.device = selection['device']

            def encode(self, images):
                if len(images) > 2:
                    raise torch.cuda.OutOfMemoryError('fixture batch memory limit')
                return super().encode(images)

        manager = EmbeddingManager(self.previews, self.root/'split', encoder_factory=LimitedEncoder)
        preview = self.preview(colors=['red', 'lime', 'blue'] * 4)
        cpu = self.analyze(preview)
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_name', return_value='Fixture CUDA device'), \
                patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'empty_cache'):
            gpu = self.analyze(preview, 'cuda', manager=manager)
        self.assertEqual(gpu['device'], 'cuda')
        self.assertEqual(len(manager._encoder.seen), 12)
        self.assertLessEqual(max(manager._encoder.calls), 2)
        np.testing.assert_array_equal(self.manager.discovery_context(cpu['id'])['states'],
                                      manager.discovery_context(gpu['id'])['states'])
        np.testing.assert_array_equal([point['position'] for point in cpu['points']],
                                      [point['position'] for point in gpu['points']])

    def test_single_input_cuda_oom_is_actionable_and_never_falls_back(self):
        class NoRoomEncoder(PixelEncoder):
            torch = torch

            def __init__(self, selection):
                super().__init__(selection)
                self.device = selection['device']

            def encode(self, images):
                raise ValueError('wrapped adapter error') from torch.cuda.OutOfMemoryError('fixture no memory')

        manager = EmbeddingManager(self.previews, self.root/'oom', encoder_factory=NoRoomEncoder)
        preview = self.preview(colors=['red'])
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_name', return_value='Fixture CUDA device'), \
                patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'empty_cache'):
            job = manager.start(preview['id'], device='cuda')
            job = self.wait(lambda: manager.get(job['id']))
        self.assertEqual(job['status'], 'error')
        self.assertRegex(job['error'], 'CPU|cpu')
        self.assertEqual(job['device'], 'cuda')
        self.assertFalse(job['points'])
        self.assert_timings(job)

    def test_cuda_model_loading_oom_is_actionable_and_does_not_retry_on_cpu(self):
        devices = []

        def oversized(selection):
            devices.append(selection['device'])
            raise ValueError('checkpoint transfer failed') from torch.cuda.OutOfMemoryError('fixture weight memory limit')

        manager = EmbeddingManager(self.previews, self.root/'load-oom', encoder_factory=oversized)
        preview = self.preview()
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_name', return_value='Fixture CUDA device'):
            job = manager.start(preview['id'], device='cuda')
            job = self.wait(lambda: manager.get(job['id']))
        self.assertEqual(job['status'], 'error')
        self.assertRegex(job['error'], 'CPU|cpu')
        self.assertRegex(job['error'], 'smaller model')
        self.assertEqual(devices, ['cuda'])
        self.assertEqual(job['device'], 'cuda')
        self.assertFalse(job['points'])
        self.assertIsNone(manager._encoder)
        self.assert_timings(job)


if __name__ == '__main__':
    unittest.main()
