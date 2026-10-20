"""Dataset analysis retains original states and expands only the selected article."""
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
from rl_policy import TerminalDQN
from server import PreviewManager, create_server, validate_config


class Articles:
    """The immutable dataset boundary, with the same row/page contract as Arrow."""
    id = 'a' * 32

    def __init__(self):
        self.rows = [
            dict(index=0, name='train[0]', text='ALPHA Report: ' * 50, summary='Alpha summary.',
                 source_split='train', source_index=0),
            dict(index=1, name='test[2]', text='BETA Article! ' * 80, summary=None,
                 source_split='test', source_index=2),
        ]
        self.metadata = dict(id=self.id, task='summarization', source='huggingface', name='fixture/articles',
                             count=2, status='ready', text_column='article', summary_column='highlights',
                             reference_count=1, missing_reference_count=1)
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
                                has_reference=row['summary'] is not None, source_split=row['source_split'],
                                source_index=row['source_index'])
                           for row in self.rows[offset:offset+limit]], count=len(self.rows), offset=offset)


class ArticleEncoder:
    feature_method = 'fixture text features'
    max_tokens = 64

    def __init__(self, model):
        self.seen = []

    def encode(self, texts, *, truncate=False):
        if not truncate:
            raise ValueError('Dataset sources must use bounded model inputs.')
        self.seen.extend(texts)
        return np.array([[1, len(text), text.count('A'), text.count('!'), sum(map(ord, text)) % 31]
                         for text in texts], dtype=np.float32)


class SummaryDatasetAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.previews = PreviewManager(self.root/'previews')
        self.articles = Articles()
        self.previews.datasets = self.articles
        self.encoders = []

        def factory(model):
            encoder = ArticleEncoder(model)
            self.encoders.append(encoder)
            return encoder

        self.embeddings = EmbeddingManager(self.previews, self.root/'embeddings', factory,
                                           feature_memory_bytes=0, feature_disk_bytes=4096)

    def wait(self, getter):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = getter()
            if result['status'] not in ('queued', 'running'):
                self.assertEqual(result['status'], 'completed', result.get('error'))
                return result
            time.sleep(.005)
        self.fail('Background job did not finish')

    def preview(self, **changes):
        selection = dict(task='summarization', dataset_id=self.articles.id,
            model={'source': 'huggingface', 'name': 'fixture/summary'},
            action_space={'actions': ['to_lowercase', 'remove_punctuation']})
        config = validate_config({**selection, **changes})
        started = self.previews.start(config)
        return self.wait(lambda: copy.deepcopy(self.previews.jobs[started['id']]))

    def analyze(self, preview=None):
        preview = preview or self.preview()
        started = self.embeddings.start(preview['id'], device='cpu')
        return self.wait(lambda: self.embeddings.get(started['id']))

    def expand(self, job, index=1):
        started = self.embeddings.expand(job['id'], index)
        return self.wait(lambda: self.embeddings.get_expansion(job['id'], started['id']))

    def test_preview_keeps_dataset_and_reads_full_article_and_reference_only_when_selected(self):
        preview = self.preview(text='')
        self.assertEqual(preview['config']['dataset_id'], self.articles.id)
        self.assertEqual((preview['input_count'], preview['total']), (2, 4))
        self.assertEqual(preview['dataset']['text_column'], 'article')
        self.assertEqual(self.articles.reads, [])
        self.assertEqual(preview['nodes'], [])
        selected = self.previews.get(preview['id'], 1)
        self.assertEqual(selected['original']['text'], 'BETA Article! ' * 80)
        self.assertIsNone(selected['reference_summary'])
        self.assertEqual(selected['nodes'][0]['text'], ('BETA Article! ' * 80).lower())
        self.assertEqual(self.previews.get(preview['id'], 0)['reference_summary'], 'Alpha summary.')
        self.assertEqual(self.previews.jobs[preview['id']]['nodes'], [])

    def test_dataset_task_mismatch_and_invalid_dataset_ids_are_rejected(self):
        for invalid in ('bad', 12, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_config(dict(task='summarization', dataset_id=invalid))
        with self.assertRaisesRegex(ValueError, 'summarization|classification|task'):
            self.previews.start(validate_config(dict(task='classification', dataset_id=self.articles.id)))
        self.articles.metadata['task'] = 'classification'
        with self.assertRaisesRegex(ValueError, 'summarization|classification|task'):
            self.preview()

    def test_random_edits_are_distinct_per_row_and_preserve_legacy_seed(self):
        text = 'zero one two three four five six seven eight nine'
        for row in self.articles.rows:
            row['text'] = text
        config = validate_config(dict(task='summarization', dataset_id=self.articles.id,
                                      action_space={'actions': ['delete_random_word']}))
        preview = self.previews.start(config)
        self.wait(lambda: self.previews.jobs[preview['id']])
        first = self.previews.get(preview['id'], 0)['nodes'][0]['text']
        second = self.previews.get(preview['id'], 1)['nodes'][0]['text']
        self.assertEqual(first, 'zero one two three four five six seven eight')
        self.assertEqual(second, 'one two three four five six seven eight nine')
        self.assertEqual(self.previews.get(preview['id'], 1)['nodes'][0]['text'], second)
        single = self.previews.start(validate_config(dict(task='summarization', text=text,
                                    action_space={'actions': ['delete_random_word']})))
        self.assertEqual(self.wait(lambda: self.previews.get(single['id']))['nodes'][0]['text'], first)

    def test_analyze_embeds_only_originals_into_distinct_float32_dqn_states(self):
        job = self.analyze()
        self.assertEqual(job['dataset_id'], self.articles.id)
        self.assertEqual((job['total'], job['completed'], len(job['points'])), (2, 2, 2))
        self.assertEqual(job['projection']['fit_count'], 2)
        self.assertIn('original', job['projection']['method'])
        self.assertEqual(self.encoders[0].seen, [row['text'] for row in self.articles.rows])
        self.assertTrue(all(point['text_is_snippet'] and len(point['text']) <= 200 for point in job['points']))
        self.assertEqual(job['references'], dict(dataset_id=self.articles.id, total=2, count=1, missing=1,
                                                text_column='article', summary_column='highlights'))
        self.assertTrue(job['input_policy']['truncate'])
        self.assertEqual(job['input_policy']['max_input_tokens'], 64)
        context = self.embeddings.discovery_context(job['id'])
        states = context['states']
        self.assertEqual((states.shape, states.dtype), ((2, 5), np.dtype('float32')))
        self.assertFalse(states.flags.writeable)
        self.assertFalse(np.allclose(states[0], states[1]))
        self.assertEqual(self.embeddings.storage.usage()['memory_bytes'], 0)
        self.assertGreater(self.embeddings.storage.usage()['disk_bytes'], 0)
        policy = TerminalDQN(states.shape[1], 2, batch_size=2)
        policy.observe(states[0], 0, .1)
        self.assertIsInstance(policy.observe(states[1], 1, .9), float)
        self.assertEqual(policy.action_statistics(states)['input_count'], 2)

    def test_selected_article_expands_lazily_in_original_pca_and_survives_cleanup(self):
        job = self.analyze()
        expanded = self.expand(job)
        original = self.articles.rows[1]['text']
        self.assertEqual(expanded['original']['text'], original)
        self.assertFalse(expanded['original']['text_is_snippet'])
        self.assertIsNone(expanded['original']['reference_summary'])
        self.assertEqual([node['text'] for node in expanded['nodes']], [original.lower(), original.replace('!', '')])
        self.assertEqual(self.encoders[0].seen, [row['text'] for row in self.articles.rows] +
                         [original.lower(), original.replace('!', '')])
        self.assertEqual(len(expanded['neighbors']), 1)
        self.assertTrue(all(len(node['position']) == 3 and -1 <= node['similarity'] <= 1
                            for node in expanded['nodes']))
        gc.collect()
        self.assertEqual(self.embeddings.release_analysis(job['id'])['status'], 'released')
        self.assertEqual(self.expand(job)['nodes'], expanded['nodes'])
        self.assertEqual(self.embeddings.discovery_node(job['id'], 1, 2), expanded['nodes'][0])
        with self.assertRaisesRegex(ValueError, '[Aa]nalyze.*again'):
            self.embeddings.expand(job['id'], 0)

    def test_discovery_projects_one_action_and_reuses_dataset_originals_across_action_spaces(self):
        first = self.analyze()
        node = self.embeddings.discovery_node(first['id'], 1, 2)
        self.assertEqual(node['text'], self.articles.rows[1]['text'].lower())
        self.assertEqual(len(self.encoders[0].seen), 3)
        self.assertEqual(self.embeddings.discovery_node(first['id'], 1, 2), node)
        self.assertEqual(len(self.encoders[0].seen), 3)
        second = self.analyze(self.preview(action_space={'actions': ['remove_punctuation']}))
        self.assertTrue(second['features_reused'])
        self.assertEqual(len(self.encoders[0].seen), 3)
        self.assertEqual(second['points'], first['points'])
        self.assertEqual(self.expand(first)['nodes'][0], node)

    def test_real_arrow_rows_keep_long_articles_and_reference_associations_through_analyze(self):
        from datasets import Dataset
        from data_sources import DatasetManager
        self.previews.datasets = DatasetManager(self.root/'real-datasets')
        long_article = 'LONG Article. ' * 5000
        rows = Dataset.from_dict({'body': ['First article.', long_article], 'gold': ['First summary.', 'Second summary.']})
        with patch('datasets.load_dataset', return_value=rows):
            dataset = self.previews.datasets.start(dict(source='huggingface', task='summarization',
                name='fixture/articles', text_column='body', summary_column='gold'))
            deadline = time.monotonic() + 5
            while self.previews.datasets.get(dataset['id'])['status'] == 'loading' and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(self.previews.datasets.get(dataset['id'])['status'], 'ready')
        preview = self.preview(dataset_id=dataset['id'])
        job = self.analyze(preview)
        self.assertEqual(job['references']['count'], 2)
        self.assertEqual(self.encoders[0].seen, ['First article.', long_article])
        self.assertEqual(self.previews.get(preview['id'], 0)['reference_summary'], 'First summary.')
        expansion = self.expand(job)
        self.assertEqual(expansion['original']['text'], long_article)
        self.assertEqual(expansion['original']['reference_summary'], 'Second summary.')
        self.assertEqual(expansion['nodes'][0]['text'], long_article.lower())

    def test_summary_graph_expands_during_rl_and_resumes_with_exclusive_model_access(self):
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

        class Encoder(ArticleEncoder):
            def encode(self, texts, *, truncate=False):
                with model_operation():
                    return super().encode(texts, truncate=truncate)

        class Evaluator:
            def predict(self, value, **kwargs):
                with model_operation():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('Test did not release inference')
                    return {'summary': value, 'reference': kwargs['reference_summary']}

            def score(self, output, baseline, *, target=None):
                with model_operation():
                    return dict(reward=.1, severity=.1, failure=False,
                                criterion='fixture score', metrics={})

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

        self.articles.rows[1]['summary'] = 'Beta summary.'
        self.articles.metadata.update(reference_count=2, missing_reference_count=0)
        self.embeddings.encoder_factory = Encoder
        job = self.analyze()
        discovery = DiscoveryManager(self.previews, self.embeddings, self.root/'discoveries',
                                     evaluator_factory=lambda *args, **kwargs: Evaluator())
        search = discovery.start(dict(embedding_id=job['id'], episodes=10000, device='cpu'))
        done = lambda value: value['status'] not in ('queued', 'running')
        try:
            self.assertTrue(entered.wait(5), 'Discovery must enter real inference before selecting an input')
            expansion = self.embeddings.expand(job['id'], 1)
            self.assertNotEqual(expansion['status'], 'completed')
            release.set()
            expanded = self.wait(lambda: self.embeddings.get_expansion(job['id'], expansion['id']))
            self.assertEqual(expanded['original']['text'], self.articles.rows[1]['text'])
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


class SummaryTextHTTPTests(unittest.TestCase):
    def test_full_article_endpoint_keeps_reference_and_rejects_invalid_index(self):
        from test_dataset_http import MemoryConnection, MemoryServer
        with tempfile.TemporaryDirectory() as temporary, patch('server.ThreadingHTTPServer', MemoryServer):
            server = create_server(8000, temporary, model_preparer=lambda model: (model, {}))
            server.previews.datasets = Articles()
            for index, expected in ((0, 200), (2, 400)):
                request = f'GET /api/datasets/{Articles.id}/text/{index} HTTP/1.1\r\nHost: 127.0.0.1:8000\r\n\r\n'
                connection = MemoryConnection(request.encode())
                with patch('server.BaseHTTPRequestHandler.log_message'):
                    server.handler(connection, ('127.0.0.1', 1), server)
                headers, body = connection.output.split(b'\r\n\r\n', 1)
                self.assertEqual(int(headers.split()[1]), expected, body)
                if expected == 200:
                    value = json.loads(body)
                    self.assertEqual(value['text'], 'ALPHA Report: ' * 50)
                    self.assertEqual(value['summary'], 'Alpha summary.')


if __name__ == '__main__':
    unittest.main()
