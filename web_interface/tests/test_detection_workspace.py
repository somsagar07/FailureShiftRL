import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from PIL import Image


class Dataset:
    def ready(self, key):
        if key != 'a' * 32:
            raise KeyError(key)
        return {'count': 2, 'task': 'classification'}

    def page(self, key, offset=0, limit=100):
        return {'items': [{'name': f'image-{i}'} for i in range(offset, min(offset + limit, 2))]}

    def image(self, key, index):
        return Image.new('RGB', (32, 32), (180 + index, 120, 100))


class FakeDetector:
    def __init__(self, name, device='auto'):
        pass

    def embeddings(self, image):
        return np.array([[image.getpixel((0, 0))[0], 1.]], dtype='float32')

    def predict(self, image):
        score = 0.9 if image.getpixel((0, 0))[0] > 130 else 0.2
        return {'detections': [{'label': 'cat', 'score': score, 'box': [2., 2., 20., 20.]}]}


class DetectionWorkspaceTests(unittest.TestCase):
    def wait(self, manager, job):
        for _ in range(300):
            job = manager.get(job['id'])
            if job['status'] not in ('running', 'queued'):
                return job
            time.sleep(.02)
        self.fail('Job did not complete')

    def test_analysis_and_dqn_discovery_on_actual_pixels(self):
        from detection_workspace import DetectionManager
        with tempfile.TemporaryDirectory() as path:
            manager = DetectionManager(Dataset(), Path(path), detector_factory=FakeDetector)
            analysis = self.wait(manager, manager.analyze({'dataset_id': 'a' * 32,
                'model': {'source': 'huggingface', 'name': 'org/detector'}}))
            self.assertEqual(analysis['status'], 'completed', analysis.get('error'))
            self.assertEqual(len(analysis['points']), 2)
            job = self.wait(manager, manager.discover({'analysis_id': analysis['id'],
                'episodes': 20, 'seed': 7, 'actions': ['darkness']}))
            self.assertEqual(job['status'], 'completed', job.get('error'))
            self.assertEqual(job['completed'], 20)
            self.assertTrue(job['records'])
            self.assertTrue(all('severity' in record for record in job['records']))
            self.assertTrue(manager.image(job['id'], 0, 'original').startswith(b'\x89PNG'))
            self.assertGreater(job['policy_updates'], 0)

    def test_rejects_invalid_settings_and_unsupported_training(self):
        from detection_workspace import DetectionManager
        with tempfile.TemporaryDirectory() as path:
            manager = DetectionManager(Dataset(), Path(path), detector_factory=FakeDetector)
            with self.assertRaises(ValueError):
                manager.analyze({'dataset_id': 'a' * 32, 'model': {'source': 'unsafe', 'name': 'bad'}})
            with self.assertRaises((ValueError, KeyError)):
                manager.discover({'analysis_id': 'missing', 'episodes': 0, 'actions': []})

    def test_expansion_and_inspection_use_model_features_and_predictions(self):
        from detection_workspace import DetectionManager
        with tempfile.TemporaryDirectory() as path:
            manager = DetectionManager(Dataset(), Path(path), detector_factory=FakeDetector)
            analysis = self.wait(manager, manager.analyze({'dataset_id': 'a' * 32,
                'model': {'source': 'huggingface', 'name': 'org/detector'}}))
            expansion = self.wait(manager, manager.expand({'analysis_id': analysis['id'],
                'input_index': 1, 'actions': ['darkness']}))
            self.assertEqual(expansion['status'], 'completed', expansion.get('error'))
            self.assertEqual(len(expansion['nodes']), 5)
            self.assertEqual(len(expansion['nodes'][0]['position']), 3)
            inspect = self.wait(manager, manager.inspect({'expansion_id': expansion['id'], 'action_id': 4}))
            self.assertEqual(inspect['status'], 'completed', inspect.get('error'))
            self.assertEqual(inspect['baseline']['detections'][0]['label'], 'cat')
            self.assertLess(inspect['output']['detections'][0]['score'], inspect['baseline']['detections'][0]['score'])
            self.assertTrue(manager.image(inspect['id'], 1, 'original').startswith(b'\x89PNG'))
            self.assertTrue(manager.image(inspect['id'], 1, 'action-4').startswith(b'\x89PNG'))
