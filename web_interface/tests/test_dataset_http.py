"""Dataset metadata and selection through the actual HTTP handler, without Hub downloads."""
import io
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server import create_server


class MemoryServer:
    def __init__(self, address, handler):
        self.server_port = address[1]
        self.handler = handler


class MemoryConnection:
    def __init__(self, payload):
        self.payload = payload
        self.output = b''

    def makefile(self, *args):
        return io.BytesIO(self.payload)

    def sendall(self, data):
        self.output += data


class DatasetHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with patch('server.ThreadingHTTPServer', MemoryServer):
            self.server = create_server(8000, self.tmp.name, model_preparer=lambda model: (model, {}))

    def tearDown(self):
        self.tmp.cleanup()

    def request(self, path, method='GET', body=None):
        payload = b'' if body is None else json.dumps(body).encode()
        headers = f'{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:8000\r\n'
        if method == 'POST':
            headers += f'Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n'
        conn = MemoryConnection(headers.encode() + b'\r\n' + payload)
        with patch('server.BaseHTTPRequestHandler.log_message'):
            self.server.handler(conn, ('127.0.0.1', 1), self.server)
        headers, body = conn.output.split(b'\r\n\r\n', 1)
        return int(headers.split()[1]), json.loads(body)

    def test_options_expose_real_configs_splits_and_columns_without_loading_rows(self):
        from datasets import Audio, ClassLabel, Features, Image, Value
        info = SimpleNamespace(splits={'validation': SimpleNamespace(num_examples=13),
                                       'test': SimpleNamespace(num_examples=5)},
                               features=Features({'photo': Image(), 'label': ClassLabel(names=['cat', 'dog']),
                                                  'audio': Audio(), 'description': Value('string')}))
        with patch('datasets.get_dataset_config_names', return_value=['large', 'default']), \
             patch('datasets.get_dataset_config_info', return_value=info) as metadata, \
             patch('datasets.load_dataset', side_effect=AssertionError('Metadata must not download image rows')):
            status, body = self.request('/api/datasets/options?name=org/images')
        self.assertEqual(status, 200, body)
        self.assertEqual(body, dict(name='org/images', configs=['large', 'default'], config='default',
                                   splits=[dict(name='validation', count=13), dict(name='test', count=5)],
                                   image_columns=['photo'], label_columns=['label'],
                                   label_column_options=['label', 'description'], text_columns=['description'],
                                   suggested_text_column=None, suggested_summary_column=None,
                                   prompt_columns=['description'], suggested_prompt_column='description',
                                   checks_columns=[], suggested_checks_column=None))
        metadata.assert_called_once_with('org/images', config_name='default')

    def test_geneval2_options_expose_the_pinned_benchmark_without_huggingface_lookup(self):
        with patch('datasets.get_dataset_config_names', side_effect=AssertionError('GenEval2 is an official JSONL benchmark')), \
                patch('datasets.load_dataset', side_effect=AssertionError('Options must not load benchmark rows')):
            status, body = self.request('/api/datasets/options?name=geneval2')
        self.assertEqual(status, 200, body)
        self.assertEqual(body['source'], 'benchmark')
        self.assertEqual(body['splits'], [{'name': 'test', 'count': 800}])
        self.assertEqual(body['suggested_prompt_column'], 'prompt')
        self.assertEqual(body['suggested_checks_column'], 'vqa_list')
        self.assertEqual(body['prompt_columns'], ['prompt'])
        status, body = self.request('/api/datasets/options?name=geneval2&config=missing')
        self.assertEqual(status, 400, body)

    def test_metadata_offers_arbitrary_scalar_labels_without_guessing_their_meaning(self):
        from datasets import Audio, Features, Image, Value
        info = SimpleNamespace(splits={'test': {'num_examples': 2}},
            features=Features({'image': Image(), 'species': Value('string'), 'row_number': Value('int64'),
                               'score': Value('float64'), 'audio': Audio()}))
        with patch('datasets.get_dataset_config_names', return_value=['default']), \
             patch('datasets.get_dataset_config_info', return_value=info):
            status, body = self.request('/api/datasets/options?name=org/images')
        self.assertEqual(status, 200, body)
        self.assertEqual(body['label_columns'], [])
        self.assertEqual(body.get('label_column_options'), ['species', 'row_number'])

    def test_options_use_requested_config_or_first_real_config_and_unknown_counts(self):
        from datasets import Features, Image
        info = SimpleNamespace(splits={'evaluation': {'name': 'evaluation'}}, features=Features({'image': Image()}))
        for query, chosen in [('', 'small'), ('&config=large', 'large')]:
            with self.subTest(query=query), patch('datasets.get_dataset_config_names', return_value=['small', 'large']), \
                 patch('datasets.get_dataset_config_info', return_value=info) as metadata:
                status, body = self.request('/api/datasets/options?name=org/images' + query)
                self.assertEqual(status, 200, body)
                self.assertEqual(body['config'], chosen)
                self.assertEqual(body['splits'], [{'name': 'evaluation', 'count': None}])
                self.assertEqual(body['label_columns'], [])
                metadata.assert_called_once_with('org/images', config_name=chosen)

    def test_metadata_failures_are_actionable_and_do_not_invent_train(self):
        from datasets import Features, Image
        fixtures = [([], SimpleNamespace(splits={}, features=None), 'configuration'),
                    (['default'], SimpleNamespace(splits={}, features=Features({'image': Image()})), 'split'),
                    (['default'], SimpleNamespace(splits={'test': {}}, features=None), 'column')]
        for configs, info, phrase in fixtures:
            with self.subTest(phrase=phrase), patch('datasets.get_dataset_config_names', return_value=configs), \
                 patch('datasets.get_dataset_config_info', return_value=info):
                status, body = self.request('/api/datasets/options?name=org/images')
                self.assertEqual(status, 400, body)
                self.assertIn(phrase, body['error'].lower())
                self.assertNotIn('splits', body)
        with patch('datasets.get_dataset_config_names', side_effect=RuntimeError('Repository access denied')):
            status, body = self.request('/api/datasets/options?name=org/private')
            self.assertEqual(status, 400, body)
            self.assertIn('org/private', body['error'])
            self.assertIn('access', body['error'].lower())

    def test_options_validate_query_and_config(self):
        from datasets import Features, Image
        info = SimpleNamespace(splits={'test': {'num_examples': 3}}, features=Features({'image': Image()}))
        for query in ['', '?name=', '?name=../invalid', '?name=org/images&name=org/other',
                      '?name=org/images&config=one&config=two', '?name=&name=org/images',
                      '?name=org/images&config=&config=default']:
            with self.subTest(query=query), patch('datasets.get_dataset_config_names', return_value=['default']), \
                 patch('datasets.get_dataset_config_info', return_value=info):
                status, body = self.request('/api/datasets/options' + query)
                self.assertEqual(status, 400, body)
        with patch('datasets.get_dataset_config_names', return_value=['small', 'large']):
            status, body = self.request('/api/datasets/options?name=org/images&config=missing')
            self.assertEqual(status, 400, body)
            self.assertIn('small', body['error'])
            self.assertIn('large', body['error'])

    def test_post_rejects_malformed_selection_before_starting_background_work(self):
        for fields in [{'splits': []}, {'splits': ['train', 'train']}, {'splits': 'train'},
                       {'subset': 20}, {'subset': '0'}, {'subset': '[3:1]'}, {'label_column': []}]:
            with self.subTest(fields=fields), patch('data_sources.threading.Thread'):
                status, body = self.request('/api/datasets', 'POST', dict(source='huggingface', name='org/images', **fields))
                self.assertEqual(status, 400, body)
                self.assertIn('error', body)

    def test_post_selection_keeps_subset_labels_aligned_through_items_endpoint(self):
        from datasets import Dataset, Features, Image, Value
        features = Features({'image': Image(), 'species': Value('string')})
        datasets = {'train': Dataset.from_dict({'image': [None]*3, 'species': ['cat', 'dog', 'fox']}, features=features),
                    'test': Dataset.from_dict({'image': [None]*2, 'species': ['owl', 'elk']}, features=features)}
        info = SimpleNamespace(splits={'train': {'num_examples': 3}, 'test': {'num_examples': 2}}, features=features)
        with patch('datasets.get_dataset_config_names', return_value=['default']), \
             patch('datasets.get_dataset_config_info', return_value=info), \
             patch('datasets.load_dataset', side_effect=lambda path, **kw: datasets[kw['split']]):
            status, body = self.request('/api/datasets', 'POST',
                dict(source='huggingface', name='org/images', splits=['test', 'train'], subset='[1:4]',
                     config='', column='', label_column='species'))
            self.assertEqual(status, 202, body)
            path = '/api/datasets/' + body['id']
            deadline = time.monotonic() + 5
            while body['status'] == 'loading' and time.monotonic() < deadline:
                time.sleep(.01)
                status, body = self.request(path)
            self.assertEqual(body['status'], 'ready', body)
        self.assertEqual(body['count'], 3)
        self.assertEqual(body.get('source_count'), 5)
        self.assertEqual(body['label_column'], 'species')
        self.assertEqual(body['splits'], ['test', 'train'])
        status, items = self.request(path + '/items')
        self.assertEqual(status, 200)
        self.assertEqual([item['label'] for item in items['items']], ['elk', 'cat', 'dog'])
        self.assertEqual([item['name'] for item in items['items']], ['test[1]', 'train[0]', 'train[1]'])
        self.assertEqual([item['index'] for item in items['items']], [0, 1, 2])
        self.assertEqual([item['source_index'] for item in items['items']], [1, 0, 1])


if __name__ == '__main__':
    unittest.main()
