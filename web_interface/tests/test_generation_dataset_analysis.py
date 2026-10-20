"""Prompt datasets retain full scenes and fit only original input features."""
import copy
import gc
import json
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from embeddings import EmbeddingManager
from server import PreviewManager, create_server, validate_config


class Prompts:
    """Immutable row/page dataset boundary without remote downloads."""
    id = 'b' * 32

    def __init__(self):
        self.rows = [
            dict(index=0, name='train[0]', text='A red dog beside two blue trees.',
                 checks=[dict(question='How many trees?', answer='two', skill='counting')],
                 source_split='train', source_index=0),
            dict(index=1, name='test[2]', text='A green cat above a yellow bowl.', checks=[],
                 source_split='test', source_index=2),
        ]
        self.metadata = dict(id=self.id, task='generation', source='huggingface', name='fixture/prompts',
                             count=2, status='ready', prompt_column='scene', checks_column='questions',
                             checked_count=1, missing_checks_count=1)
        self.reads = []

    def ready(self, dataset_id):
        if dataset_id != self.id:
            raise KeyError(dataset_id)
        return copy.deepcopy(self.metadata)

    get = ready

    def text(self, dataset_id, index):
        self.ready(dataset_id)
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise ValueError('Input index is outside the dataset.')
        self.reads.append(index)
        return copy.deepcopy(self.rows[index])

    def page(self, dataset_id, offset=0, limit=100):
        self.ready(dataset_id)
        return dict(items=[dict(index=row['index'], name=row['name'], text_preview=row['text'][:200],
                                check_count=len(row['checks']),
                                source_split=row['source_split'], source_index=row['source_index'])
                           for row in self.rows[offset:offset+limit]], count=len(self.rows), offset=offset)


class PromptEncoder:
    feature_method = 'fixture prompt features'
    max_tokens = 64

    def __init__(self, model):
        self.seen = []

    def encode(self, texts, *, truncate=False):
        if truncate:
            raise ValueError('Generation must preserve the full scene.')
        for index, text in enumerate(texts):
            count = len(text.split()) + 2
            if count > self.max_tokens:
                raise ValueError(f'Diffusion prompt feature extraction failed: Text input {index+1} has {count} tokens, '
                                 f'exceeding this model\'s supported input limit of {self.max_tokens} tokens. '
                                 'Shorten the input; truncation is disabled.')
        self.seen.extend(texts)
        return np.array([[1, len(text), text.count('a'), text.count('b'), sum(map(ord, text)) % 31]
                         for text in texts], dtype=np.float32)


class GenerationDatasetAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.previews = PreviewManager(self.root/'previews')
        self.prompts = Prompts()
        self.previews.datasets = self.prompts
        self.encoders = []

        def factory(model):
            encoder = PromptEncoder(model)
            self.encoders.append(encoder)
            return encoder

        self.embeddings = EmbeddingManager(self.previews, self.root/'embeddings', factory,
                                           feature_memory_bytes=0, feature_disk_bytes=4096)

    def wait(self, getter, status='completed'):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = getter()
            if result['status'] not in ('queued', 'running'):
                self.assertEqual(result['status'], status, result.get('error'))
                return result
            time.sleep(.005)
        self.fail('Background job did not finish')

    def preview(self, **changes):
        selection = dict(task='generation', dataset_id=self.prompts.id,
            model={'source': 'huggingface', 'name': 'fixture/diffusion'},
            action_space={'styles': ['watercolor', '3D render']})
        started = self.previews.start(validate_config({**selection, **changes}))
        return self.wait(lambda: copy.deepcopy(self.previews.jobs[started['id']]))

    def analyze(self, preview=None, status='completed'):
        preview = preview or self.preview()
        started = self.embeddings.start(preview['id'], device='cpu')
        return self.wait(lambda: self.embeddings.get(started['id']), status)

    def expand(self, job, index=1, status='completed'):
        started = self.embeddings.expand(job['id'], index)
        return self.wait(lambda: self.embeddings.get_expansion(job['id'], started['id']), status)

    def test_free_text_prompt_and_style_keep_original_content_verbatim(self):
        text = '  Two cubes labeled {A} and {B}, on a blue table.\n'
        preview = self.preview(dataset_id='', text=text, action_space={'styles': ['watercolor']})
        self.assertEqual(preview['original']['text'], text)
        self.assertEqual(preview['nodes'][0]['text'], 'Watercolor style. Scene: ' + text)
        self.assertEqual(preview['nodes'][0]['action'], {'id': 1, 'style': 'watercolor'})
        job = self.analyze(preview)
        self.assertEqual(job['projection']['fit_count'], 2)
        self.assertEqual(job['points'][0]['text'], text)

    def test_preview_loads_only_selected_full_prompt_with_associated_checks(self):
        self.prompts.rows[0]['text'] *= 1000
        preview = self.preview(text=None)
        self.assertEqual((preview['input_count'], preview['total']), (2, 4))
        self.assertEqual(preview['config']['dataset_id'], self.prompts.id)
        self.assertEqual(self.prompts.reads, [])
        self.assertEqual(preview['nodes'], [])
        first = self.previews.get(preview['id'], 0)
        self.assertEqual(first['original']['text'], self.prompts.rows[0]['text'])
        self.assertEqual(first['checks'], [dict(question='How many trees?', answer='two', skill='counting')])
        self.assertEqual(first['original']['checks'], first['checks'])
        self.assertEqual(first['original']['source_split'], 'train')
        self.assertEqual(first['nodes'][0]['text'], 'Watercolor style. Scene: ' + first['original']['text'])
        second = self.previews.get(preview['id'], 1)
        self.assertEqual(second['checks'], [])
        self.assertEqual(second['input_name'], 'test[2]')
        self.assertEqual(self.previews.jobs[preview['id']]['nodes'], [])
        manifest = json.loads((self.previews.directory/preview['id']/'manifest.json').read_text())
        self.assertEqual(manifest['nodes'], [])

    def test_generation_rejects_invalid_dataset_ids_and_task_mismatch(self):
        for invalid in ('bad', None, 12):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_config(dict(task='generation', dataset_id=invalid))
        self.prompts.metadata['task'] = 'summarization'
        with self.assertRaisesRegex(ValueError, 'summarization|generation|task'):
            self.preview()

    def test_analyze_fits_only_original_prompts_and_exposes_requirement_coverage(self):
        job = self.analyze()
        self.assertEqual((job['total'], job['completed'], len(job['points'])), (2, 2, 2))
        self.assertEqual(job['projection']['fit_count'], 2)
        self.assertIn('original', job['projection']['method'])
        self.assertEqual(self.encoders[0].seen, [row['text'] for row in self.prompts.rows])
        self.assertEqual(job['requirements'], dict(dataset_id=self.prompts.id, total=2, count=1, missing=1,
                                                  prompt_column='scene', checks_column='questions'))
        self.assertFalse(job['input_policy']['truncate'])
        self.assertEqual(job['input_policy']['max_input_tokens'], 64)
        self.assertEqual([point['has_checks'] for point in job['points']], [True, False])
        context = self.embeddings.discovery_context(job['id'])
        states = context['states']
        self.assertEqual((states.shape, states.dtype), ((2, 5), np.dtype('float32')))
        self.assertFalse(states.flags.writeable)
        np.testing.assert_allclose(np.linalg.norm(states, axis=1), [1, 1], atol=1e-7)
        self.assertFalse(np.allclose(states[0], states[1]))
        self.assertEqual(self.embeddings.storage.usage()['memory_bytes'], 0)
        self.assertGreater(self.embeddings.storage.usage()['disk_bytes'], 0)

    def test_expansion_retains_checks_and_projects_styles_in_the_original_pca(self):
        job = self.analyze()
        expansion = self.expand(job, 0)
        original = self.prompts.rows[0]['text']
        self.assertEqual(expansion['original']['text'], original)
        self.assertFalse(expansion['original']['text_is_snippet'])
        self.assertEqual(expansion['original']['checks'], self.prompts.rows[0]['checks'])
        self.assertEqual(expansion['original']['source_index'], 0)
        self.assertEqual([node['text'] for node in expansion['nodes']],
                         ['Watercolor style. Scene: ' + original, '3D render style. Scene: ' + original])
        self.assertEqual([node['id'] for node in expansion['nodes']], [1, 4])
        self.assertEqual(len(self.encoders[0].seen), 4)
        self.assertEqual(len(expansion['neighbors']), 1)
        self.assertTrue(all(len(node['position']) == 3 and -1 <= node['similarity'] <= 1
                            for node in expansion['nodes']))
        self.assertEqual(self.embeddings.get(job['id'])['points'], job['points'])
        gc.collect()
        self.assertEqual(self.embeddings.release_analysis(job['id'])['status'], 'released')
        self.assertEqual(self.expand(job, 0)['nodes'], expansion['nodes'])
        self.assertEqual(self.embeddings.discovery_node(job['id'], 0, 1), expansion['nodes'][0])
        with self.assertRaisesRegex(ValueError, '[Aa]nalyze.*again'):
            self.embeddings.expand(job['id'], 1)

    def test_grouped_actions_expand_lazily_and_preserve_ids_for_discovery(self):
        from core import default_action_space

        preview = self.preview(action_space=default_action_space('generation'))
        self.assertEqual((preview['input_count'], preview['total']), (2, 250))
        job = self.analyze(preview)
        self.assertEqual(len(self.encoders[0].seen), 2)
        self.assertEqual(job['projection']['fit_count'], 2)
        expansion = self.expand(job, 1)
        self.assertEqual(len(expansion['nodes']), 125)
        last = expansion['nodes'][-1]
        self.assertEqual(last['id'], 215)
        self.assertEqual(last['text'], '3D render style. Low-key lighting. Detail level: intricate. Scene: '
                         + self.prompts.rows[1]['text'])
        self.assertEqual(self.embeddings.discovery_node(job['id'], 1, 215), last)
        self.assertEqual(self.embeddings.get(job['id'])['points'], job['points'])
        reduced = self.analyze(self.preview(action_space=dict(style=[], lighting=['soft'], detail=[])))
        self.assertTrue(reduced['features_reused'])
        node = self.expand(reduced, 1)['nodes'][0]
        self.assertEqual(node['id'], 12)
        self.assertEqual(node['text'], 'Soft lighting. Scene: ' + self.prompts.rows[1]['text'])

    def test_measured_styles_are_lazy_and_original_features_reuse_across_action_spaces(self):
        first = self.analyze()
        measured = self.embeddings.discovery_node(first['id'], 1, 4)
        self.assertEqual(measured['text'], '3D render style. Scene: A green cat above a yellow bowl.')
        self.assertEqual(len(self.encoders[0].seen), 3)
        self.assertEqual(self.embeddings.discovery_node(first['id'], 1, 4), measured)
        self.assertEqual(len(self.encoders[0].seen), 3)
        second = self.analyze(self.preview(action_space={'styles': ['photograph']}))
        self.assertTrue(second['features_reused'])
        self.assertEqual(second['points'], first['points'])
        self.assertEqual(len(self.encoders[0].seen), 3)
        self.assertEqual(self.expand(first)['nodes'][1], measured)

    def test_overlength_original_fails_with_global_row_and_no_partial_features(self):
        self.embeddings.BATCH_SIZE = 1
        self.prompts.rows[1]['text'] = 'cat ' * 63
        result = self.analyze(status='error')
        self.assertIn('Input 2', result['error'])
        self.assertIn('test[2]', result['error'])
        self.assertIn('truncation is disabled', result['error'])
        self.assertEqual(result['points'], [])
        self.assertEqual(self.embeddings.storage.usage()['disk_bytes'], 0)

    def test_overlength_style_reports_input_and_action_without_truncation(self):
        self.prompts.rows[1]['text'] = 'cat ' * 61
        job = self.analyze()
        result = self.expand(job, status='error')
        self.assertIn('Input 2', result['error'])
        self.assertIn('test[2]', result['error'])
        self.assertIn('watercolor', result['error'])
        self.assertIn('truncation is disabled', result['error'])
        self.assertEqual(result['nodes'], [])
        with self.assertRaisesRegex(ValueError, r'Input 2.*3D render.*truncation is disabled'):
            self.embeddings.discovery_node(job['id'], 1, 4)

    def test_real_arrow_prompt_rows_preserve_checks_and_split_provenance(self):
        from datasets import Dataset
        from data_sources import DatasetManager
        self.previews.datasets = DatasetManager(self.root/'real-datasets')
        rows = Dataset.from_dict({'scene': ['One blue tree.', 'Two yellow birds above a red house.'],
                                  'questions': [[{'question': 'What color is the tree?', 'answer': 'blue'}], []]})
        with patch('datasets.load_dataset', return_value=rows):
            started = self.previews.datasets.start(dict(task='generation', source='huggingface',
                name='fixture/prompts', split='validation', prompt_column='scene', checks_column='questions'))
            deadline = time.monotonic() + 5
            while self.previews.datasets.get(started['id'])['status'] == 'loading' and time.monotonic() < deadline:
                time.sleep(.005)
            dataset = self.previews.datasets.get(started['id'])
            self.assertEqual(dataset['status'], 'ready', dataset.get('error'))
        preview = self.preview(dataset_id=dataset['id'])
        job = self.analyze(preview)
        self.assertEqual(job['requirements']['count'], 1)
        self.assertEqual(self.encoders[0].seen, ['One blue tree.', 'Two yellow birds above a red house.'])
        expansion = self.expand(job, 0)
        self.assertEqual(expansion['original']['checks'], [dict(question='What color is the tree?', answer='blue')])
        self.assertEqual(expansion['original']['source_split'], 'validation')
        self.assertEqual(expansion['original']['source_index'], 0)
        self.assertEqual(self.expand(job, 1)['original']['checks'], [])

    def test_selected_prompt_expansion_resumes_live_discovery_with_exclusive_model_access(self):
        from discovery import DiscoveryManager
        operation_lock = threading.Lock()
        entered, release = threading.Event(), threading.Event()
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

        class Encoder(PromptEncoder):
            def encode(self, texts, *, truncate=False):
                with model_operation():
                    return super().encode(texts, truncate=truncate)

        class Evaluator:
            def predict(self, value, **kwargs):
                with model_operation():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('Test did not release inference')
                    return {'prompt': value, 'checks': kwargs['checks']}

            def score(self, output, baseline, *, target=None):
                with model_operation():
                    return dict(reward=.1, severity=.1, failure=False, criterion='fixture score', metrics={})

            def describe(self):
                return {}

            def close(self):
                pass

        def wait_for(getter, predicate):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                result = getter()
                self.assertNotEqual(result['status'], 'error', result.get('error'))
                if predicate(result):
                    return result
                time.sleep(.005)
            self.fail(f'Background work did not reach the expected state: {result}')

        self.prompts.rows[1]['checks'] = [dict(question='What color is the cat?', answer='green')]
        self.prompts.metadata.update(checked_count=2, missing_checks_count=0)
        self.embeddings.encoder_factory = Encoder
        job = self.analyze()
        discovery = DiscoveryManager(self.previews, self.embeddings, self.root/'discoveries',
                                     evaluator_factory=lambda *args, **kwargs: Evaluator())
        search = discovery.start(dict(embedding_id=job['id'], episodes=10000, device='cpu'))
        done = lambda value: value['status'] not in ('queued', 'running')
        try:
            self.assertTrue(entered.wait(5), 'Discovery must enter inference before selecting a prompt')
            expansion = self.embeddings.expand(job['id'], 1)
            self.assertNotEqual(expansion['status'], 'completed')
            release.set()
            expanded = self.wait(lambda: self.embeddings.get_expansion(job['id'], expansion['id']))
            self.assertEqual(expanded['original']['checks'], self.prompts.rows[1]['checks'])
            self.assertEqual(len(expanded['nodes']), 2)
            current = discovery.get(search['id'])
            self.assertEqual(current['status'], 'running', current.get('error'))
            resumed = wait_for(lambda: discovery.get(search['id']),
                               lambda result: result['episode'] > current['episode'] or done(result))
            self.assertGreater(resumed['episode'], current['episode'])
            self.assertEqual(overlaps, [])
        finally:
            release.set()
            discovery.cancel(search['id'])
            wait_for(lambda: discovery.get(search['id']), done)
            if self.embeddings._worker is not None:
                self.embeddings._worker.join(5)


class GenerationPromptHTTPTests(unittest.TestCase):
    def test_full_prompt_endpoint_preserves_checks_and_rejects_invalid_row(self):
        from test_dataset_http import MemoryConnection, MemoryServer
        with tempfile.TemporaryDirectory() as temporary, patch('server.ThreadingHTTPServer', MemoryServer):
            server = create_server(8000, temporary, model_preparer=lambda model: (model, {}))
            server.previews.datasets = Prompts()
            for index, expected in ((0, 200), (2, 400)):
                request = f'GET /api/datasets/{Prompts.id}/text/{index} HTTP/1.1\r\nHost: 127.0.0.1:8000\r\n\r\n'
                connection = MemoryConnection(request.encode())
                with patch('server.BaseHTTPRequestHandler.log_message'):
                    server.handler(connection, ('127.0.0.1', 1), server)
                headers, body = connection.output.split(b'\r\n\r\n', 1)
                self.assertEqual(int(headers.split()[1]), expected, body)
                if expected == 200:
                    value = json.loads(body)
                    self.assertEqual(value['text'], 'A red dog beside two blue trees.')
                    self.assertEqual(value['checks'], [dict(question='How many trees?', answer='two', skill='counting')])
                    self.assertEqual(value['source_split'], 'train')


class GenerationScorerReuseTests(unittest.TestCase):
    def test_judge_and_metric_changes_do_not_reuse_an_incompatible_scorer(self):
        class Evaluator:
            device = 'cpu'
            revision = None

            def __init__(self, shared):
                self.scorer_weights = shared.scorer_weights if shared is not None else object()

            def close(self):
                pass

        def load(selection, config, directory, progress, *, encoder, shared):
            return Evaluator(shared)

        with tempfile.TemporaryDirectory() as temporary, \
                patch('discovery_models._dependencies', return_value=(None, None)), \
                patch('discovery_models._device', return_value='cpu'), \
                patch('discovery_models.create_evaluator', side_effect=load):
            root = Path(temporary)
            manager = EmbeddingManager(None, root/'embeddings')
            model = dict(source='huggingface', name='fixture/diffusion', task='generation')
            config = dict(task='generation', generation_metric='checks', judge_model='Qwen/Qwen3-VL-2B-Instruct')
            first = manager.evaluation_model(model, config, root/'first')
            same = manager.evaluation_model(model, {**config, 'seed': 3}, root/'same')
            self.assertIs(first.scorer_weights, same.scorer_weights)
            other_judge = manager.evaluation_model(model,
                {**config, 'judge_model': 'Salesforce/blip-vqa-base'}, root/'judge')
            self.assertIsNot(first.scorer_weights, other_judge.scorer_weights)
            other_metric = manager.evaluation_model(model,
                {**config, 'judge_model': 'Salesforce/blip-vqa-base', 'generation_metric': 'clip'}, root/'metric')
            self.assertIsNot(other_judge.scorer_weights, other_metric.scorer_weights)
            local_judge = root/'judge-model'
            local_judge.mkdir()
            (local_judge/'config.json').write_text('{}')
            with_local = {**config, '_judge_model': {'source': 'local', 'name': str(local_judge)}}
            local = manager.evaluation_model(model, with_local, root/'local')
            (local_judge/'config.json').write_text('{"model_type":"blip"}')
            replaced = manager.evaluation_model(model, with_local, root/'replaced')
            self.assertIsNot(local.scorer_weights, replaced.scorer_weights)


if __name__ == '__main__':
    unittest.main()
