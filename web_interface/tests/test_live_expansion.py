"""Interactive graph work must complete while discovery is still running."""
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from discovery import DiscoveryManager
from embeddings import EmbeddingManager
from server import PreviewManager, validate_config


class LiveExpansionTests(unittest.TestCase):
    def test_expansion_during_discovery_uses_exclusive_model_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images'
            images.mkdir()
            for index, color in enumerate(('red', 'blue', 'green')):
                Image.new('RGB', (12, 12), color).save(images / f'{index}.png')
            operation_lock = threading.Lock()
            entered = threading.Event()
            release = threading.Event()
            overlaps = []

            @contextmanager
            def model_operation():
                acquired = operation_lock.acquire(blocking=False)
                if not acquired:
                    overlaps.append('Concurrent model operations')
                try:
                    yield
                finally:
                    if acquired:
                        operation_lock.release()

            class Encoder:
                class_labels = {0: 'fixture'}
                def __init__(self, model):
                    pass

                def encode(self, values):
                    with model_operation():
                        return np.array([[1, *np.asarray(value).mean(axis=(0, 1))]
                                         for value in values], dtype=float)

            class Evaluator:
                def resolve_label(self, value):
                    return int(value)

                def predict(self, value, **kwargs):
                    with model_operation():
                        entered.set()
                        if not release.wait(5):
                            raise RuntimeError('Test did not release inference')
                        time.sleep(.005)
                        return {'class_id': 0, 'label': 'fixture', 'confidence': .8}

                def score(self, output, baseline, target=None):
                    with model_operation():
                        return dict(reward=.1, severity=.1, failure=False,
                                    criterion='fixture score', metrics={})

                def describe(self):
                    return {}

                def close(self):
                    pass

            def wait(getter, predicate, timeout=5):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    result = getter()
                    if predicate(result):
                        return result
                    time.sleep(.005)
                self.fail(f'Background work did not reach the expected state: {result}')

            done = lambda job: job['status'] not in ('queued', 'running', 'loading')
            previews = PreviewManager(root / 'previews')
            embeddings = EmbeddingManager(previews, root / 'embeddings', Encoder)
            discovery = DiscoveryManager(previews, embeddings, root / 'discoveries',
                                         evaluator_factory=lambda *args, **kwargs: Evaluator())
            dataset = previews.datasets.start({'source': 'folder', 'path': str(images)})
            wait(lambda: previews.datasets.get(dataset['id']), done)
            preview = previews.start(validate_config(dict(dataset_id=dataset['id'],
                                     model={'source': 'huggingface', 'name': 'test/model'})))
            wait(lambda: previews.get(preview['id']), done)
            analysis = embeddings.start(preview['id'], device='cpu')
            wait(lambda: embeddings.get(analysis['id']), done)
            search = discovery.start(dict(embedding_id=analysis['id'], episodes=10000, device='cpu',
                                          label_map={'0': 0, '1': 0, '2': 0}))
            try:
                self.assertTrue(entered.wait(5), 'Discovery must enter real model execution')
                expansion = embeddings.expand(analysis['id'], 0)
                self.assertNotEqual(expansion['status'], 'completed')
                release.set()
                expanded = wait(lambda: embeddings.get_expansion(analysis['id'], expansion['id']), done)
                self.assertEqual(expanded['status'], 'completed', expanded.get('error'))
                self.assertEqual(len(expanded['nodes']), 125)
                self.assertTrue(all(len(node['position']) == 3 for node in expanded['nodes']))
                current = discovery.get(search['id'])
                self.assertEqual(current['status'], 'running', current.get('error'))
                resumed = wait(lambda: discovery.get(search['id']),
                               lambda job: job['episode'] > current['episode'] or done(job))
                self.assertGreater(resumed['episode'], current['episode'], resumed.get('error'))
                self.assertEqual(overlaps, [], 'Embedding and scoring must share the GPU exclusively')
            finally:
                release.set()
                discovery.cancel(search['id'])
                wait(lambda: discovery.get(search['id']), done)
                if embeddings._worker is not None:
                    embeddings._worker.join(5)


if __name__ == '__main__':
    unittest.main()
