"""Exercise discovery, cancellation and reviewed training through real HTTP routing."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from server import create_server
import test_http
from test_http import MemoryServer


class Encoder:
    feature_method = 'fixture RGB'

    def __init__(self, model):
        pass

    def encode(self, images):
        return np.array([[1, *np.asarray(image).mean(axis=(0, 1))] for image in images])


class Evaluator:
    def __init__(self, model, config, directory, progress=None):
        pass

    def predict(self, value, *, seed=0, artifact_name='sample'):
        prob = float(np.asarray(value).mean() / 255)
        cls = int(prob >= .25)
        return dict(class_id=cls, label=str(cls), confidence=max(prob, 1-prob),
                    probabilities=[1-prob, prob])

    def score(self, output, baseline, *, target=None):
        target = baseline['class_id'] if target is None else int(target)
        failure = output['class_id'] != target
        severity = 1-output['probabilities'][target]
        return dict(reward=severity, severity=severity, failure=failure,
                    criterion='fixture prediction consistency', metrics={'target': target})

    def describe(self):
        return {'criterion': 'fixture prediction consistency'}

    def close(self):
        pass


class DiscoveryHTTPTests(unittest.TestCase):
    request = test_http.HTTPTests.request

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with patch('server.ThreadingHTTPServer', MemoryServer):
            self.server = create_server(8000, Path(self.tmp.name) / 'runs' / 'previews')

    def wait(self, path, job):
        deadline = time.monotonic() + 15
        while job['status'] in ('loading', 'queued', 'running') and time.monotonic() < deadline:
            time.sleep(.01)
            code, body = self.request(path)
            self.assertEqual(code, 200, body)
            job = json.loads(body)
        self.assertEqual(job['status'], 'completed', job.get('error', job))
        return job

    def test_routes_validate_missing_analysis_and_untrusted_requests(self):
        code, body = self.request('/api/discoveries')
        self.assertEqual(code, 200, 'Discovery history endpoint is missing')
        for path in ('/api/discoveries', '/api/finetunes'):
            for data in ({}, [], None):
                with self.subTest(path=path, data=data):
                    self.assertEqual(self.request(path, 'POST', body=data)[0], 400)
            self.assertEqual(self.request(path, 'POST', body={}, origin='https://elsewhere.test')[0], 403)
        for path in ('/api/discoveries/' + 'f'*32, '/api/finetunes/' + 'f'*32,
                     '/discoveries/' + 'f'*32 + '/artifacts/report.json',
                     '/discoveries/' + 'f'*32 + '/artifacts/../../server.py'):
            self.assertEqual(self.request(path)[0], 404)

    def test_analysis_search_report_and_reviewed_finetune_roundtrip(self):
        code, _ = self.request('/api/discoveries')
        self.assertEqual(code, 200, 'Discovery endpoints must exist before running the workflow')

        def train(model, task, examples, directory, settings, progress, cancelled):
            self.assertEqual(task, 'classification')
            self.assertEqual(len(examples), 1)
            self.assertIsInstance(examples[0]['input'], Image.Image)
            self.assertEqual(examples[0]['target'], '1')
            progress(completed=1, total=1, loss=.2)
            checkpoint = Path(directory) / 'checkpoint'
            checkpoint.mkdir()
            (checkpoint / 'config.json').write_text('{}')
            return dict(checkpoint=str(checkpoint), steps=1, training_loss=[.2])

        with patch('server.ThreadingHTTPServer', MemoryServer), patch('server.HuggingFaceEncoder', Encoder):
            self.server = create_server(8000, Path(self.tmp.name) / 'isolated' / 'previews',
                                        evaluator_factory=Evaluator, finetune_fn=train,
                                        model_preparer=lambda model: (model, {}))
        images = Path(self.tmp.name) / 'images'
        images.mkdir()
        Image.new('RGB', (8, 8), 'red').save(images / 'red.png')
        code, body = self.request('/api/datasets', 'POST', body={'source': 'folder', 'path': str(images)})
        self.assertEqual(code, 202)
        dataset = json.loads(body)
        # Dataset has a distinct terminal status ('ready').
        until = time.monotonic() + 5
        while dataset['status'] == 'loading' and time.monotonic() < until:
            time.sleep(.01)
            dataset = json.loads(self.request('/api/datasets/' + dataset['id'])[1])
        self.assertEqual(dataset['status'], 'ready', dataset)
        code, body = self.request('/api/previews', 'POST', body={
            'dataset_id': dataset['id'], 'model': {'source': 'huggingface', 'name': 'test/model'},
            'action_space': {'rotation': [0], 'darkness': [.1, .5], 'saturation': []}})
        preview = json.loads(body)
        self.wait('/api/previews/' + preview['id'], preview)
        code, body = self.request('/api/embeddings', 'POST', body={'preview_id': preview['id']})
        analysis = json.loads(body)
        self.wait('/api/embeddings/' + analysis['id'], analysis)
        code, body = self.request('/api/discoveries', 'POST', body={
            'embedding_id': analysis['id'], 'episodes': 4, 'seed': 7})
        self.assertEqual(code, 202, body)
        job = json.loads(body)
        job = self.wait('/api/discoveries/' + job['id'], job)
        self.assertEqual(job['episode'], 4)
        self.assertTrue(job['records'])
        self.assertTrue(job['history'])
        self.assertNotIn('probabilities', json.dumps(job))
        code, report = self.request(job['artifacts']['report'])
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(report)['id'], job['id'])
        record = job['records'][0]
        code, body = self.request('/api/finetunes', 'POST', body={
            'discovery_id': job['id'], 'examples': [{'input_index': record['input_index'],
            'action_id': record['action_id'], 'target': '1'}], 'epochs': 1, 'learning_rate': .001})
        self.assertEqual(code, 202, body)
        training = json.loads(body)
        training = self.wait('/api/finetunes/' + training['id'], training)
        self.assertTrue(Path(training['checkpoint']).is_dir())
        self.assertEqual(training['completed'], 1)


if __name__ == '__main__':
    unittest.main()
