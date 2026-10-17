"""The RL state must be the exact analyzed features, never the plotted PCA."""
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from embeddings import EmbeddingManager
from server import PreviewManager, validate_config


class FeatureEncoder:
    revision = 'a' * 40
    feature_method = 'fixture features'

    def __init__(self, model):
        self.model = model

    def encode(self, values):
        return np.array([[1, len(value), 2, 3] if isinstance(value, str)
                         else [1, *np.asarray(value).mean(axis=(0, 1))]
                         for value in values], dtype=float)


class DiscoveryContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.previews = PreviewManager(self.root / 'previews')
        self.manager = EmbeddingManager(self.previews, self.root / 'embeddings', FeatureEncoder)

    def wait(self, getter):
        until = time.monotonic() + 5
        while time.monotonic() < until:
            job = getter()
            if job['status'] not in ('loading', 'queued', 'running'):
                self.assertNotEqual(job['status'], 'error', job.get('error'))
                return job
            time.sleep(.005)
        self.fail('Work did not finish')

    def analyze(self, task, model=None):
        config = dict(task=task, model=model or {'source': 'huggingface', 'name': 'test/model'})
        if task == 'classification':
            images = self.root / 'images'
            images.mkdir(exist_ok=True)
            Image.new('RGB', (5, 6), 'red').save(images / 'red.png')
            Image.new('RGB', (5, 6), 'blue').save(images / 'blue.png')
            dataset = self.previews.datasets.start({'source': 'folder', 'path': str(images)})
            self.wait(lambda: self.previews.datasets.get(dataset['id']))
            config['dataset_id'] = dataset['id']
        preview = self.previews.start(validate_config(config))
        self.wait(lambda: self.previews.get(preview['id']))
        job = self.manager.start(preview['id'])
        return self.wait(lambda: self.manager.get(job['id']))

    def test_all_domains_supply_original_full_features_and_pinned_model(self):
        self.assertTrue(callable(getattr(self.manager, 'discovery_context', None)),
                        'Completed analysis needs a public discovery context')
        for task in ('classification', 'summarization', 'generation'):
            with self.subTest(task=task):
                job = self.analyze(task)
                context = self.manager.discovery_context(job['id'])
                self.assertEqual(context['states'].shape, (len(job['points']), 4))
                np.testing.assert_allclose(np.linalg.norm(context['states'], axis=1), 1)
                self.assertEqual(context['model']['revision'], 'a' * 40)
                self.assertEqual(context['preview']['id'], job['preview_id'])
                original = context['states'].copy()
                context['states'][:] = 0
                context['job']['points'].clear()
                np.testing.assert_array_equal(self.manager.discovery_context(job['id'])['states'], original)
                self.assertTrue(self.manager.get(job['id'])['points'])

    def test_context_refuses_a_replaced_local_checkpoint(self):
        self.assertTrue(callable(getattr(self.manager, 'discovery_context', None)))
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'preprocessor_config.json').write_text('{}')
        job = self.analyze('classification', {'source': 'local', 'name': str(model)})
        (model / 'new_weights.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.manager.discovery_context(job['id'])

    def test_model_lock_excludes_loading_and_release_preserves_cached_graph(self):
        self.assertTrue(hasattr(self.manager, 'model_lock'), 'Model execution needs a shared resource lock')
        self.assertTrue(callable(getattr(self.manager, 'release_encoder', None)))
        job = self.analyze('summarization')
        context = self.manager.discovery_context(job['id'])
        with self.manager.model_lock:
            self.manager.release_encoder()
            self.assertIsNone(self.manager._encoder)
            # Cached variants must stay viewable without loading another model.
            expansion = self.manager.expand(job['id'], 0)
            self.assertEqual(expansion['status'], 'completed')
        np.testing.assert_array_equal(self.manager.discovery_context(job['id'])['states'], context['states'])
        entered = threading.Event()
        factory = self.manager.encoder_factory

        def observe(model):
            entered.set()
            return factory(model)

        self.manager.encoder_factory = observe
        preview = self.previews.start(validate_config({'task': 'generation',
                         'model': {'source': 'huggingface', 'name': 'test/second'}}))
        self.wait(lambda: self.previews.get(preview['id']))
        with self.manager.model_lock:
            next_job = self.manager.start(preview['id'])
            self.assertFalse(entered.wait(.1), 'Embedding worker must wait for discovery/training')
        self.wait(lambda: self.manager.get(next_job['id']))
        self.assertTrue(entered.is_set())


if __name__ == '__main__':
    unittest.main()
