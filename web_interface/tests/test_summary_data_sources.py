import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, Features, Image, Value

from data_sources import DatasetManager, hf_dataset_options


def text_rows(texts, summaries, text_column='article', summary_column='summary'):
    return Dataset.from_dict(
        {text_column: texts, summary_column: summaries,
         'id': [f'example-{index}' for index in range(len(texts))],
         'unrelated_image': [{'path': 'must-not-decode.png', 'bytes': None}] * len(texts)},
        features=Features({text_column: Value('string'), summary_column: Value('string'),
                           'id': Value('string'), 'unrelated_image': Image()}))


class SummaryDatasetTests(unittest.TestCase):
    def wait(self, manager, dataset_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            info = manager.get(dataset_id)
            if info['status'] != 'loading':
                return info
            time.sleep(.01)
        self.fail('Dataset did not finish loading')

    def options(self, splits):
        return dict(name='org/texts', configs=['default'], config='default',
                    splits=[dict(name=name, count=len(rows)) for name, rows in splits.items()],
                    image_columns=[], label_columns=[], text_columns=['article', 'summary', 'id'])

    def test_metadata_suggests_only_unique_semantic_string_columns(self):
        examples = [
            ({'id': Value('string'), 'article': Value('string'), 'summary': Value('string'),
              'score': Value('float64'), 'photo': Image()}, 'article', 'summary'),
            ({'dialogue': Value('large_string'), 'highlights': Value('string')}, 'dialogue', 'highlights'),
            ({'document': Value('string'), 'abstract': Value('string')}, 'document', 'abstract'),
            ({'article': Value('string'), 'text': Value('string'), 'summary': Value('string'),
              'highlights': Value('string')}, None, None),
            ({'id': Value('string'), 'document_id': Value('string'), 'summary_id': Value('string')}, None, None),
            ({'article': [Value('string')], 'summary': Value('int64')}, None, None),
        ]
        for features, source, reference in examples:
            with self.subTest(features=features), \
                 patch('datasets.get_dataset_config_names', return_value=['default']), \
                 patch('datasets.get_dataset_config_info', return_value=SimpleNamespace(
                     features=Features(features), splits={'train': {'num_examples': 12}})):
                options = hf_dataset_options('org/texts')
            expected = [key for key, value in features.items()
                        if isinstance(value, Value) and value.dtype in ('string', 'large_string')]
            self.assertEqual(options.get('text_columns'), expected)
            self.assertEqual(options.get('suggested_text_column'), source)
            self.assertEqual(options.get('suggested_summary_column'), reference)
            self.assertEqual(options['image_columns'], ['photo'] if 'photo' in features else [])

    def test_selected_splits_keep_missing_references_and_apply_combined_range(self):
        train = text_rows(['train zero', 'train one', 'train two'], [None, 'ref one', ''])
        test = text_rows(['test zero', 'test one'], ['ref test', ' \n'])
        splits = {'train': train, 'test': test}
        with tempfile.TemporaryDirectory() as tmp, \
             patch('data_sources.hf_dataset_options', return_value=self.options(splits)), \
             patch('datasets.load_dataset', side_effect=lambda path, **kw: splits[kw['split']]), \
             patch('data_sources.extract_dataset_labels', side_effect=AssertionError('Text rows are not class labels')):
            manager = DatasetManager(tmp)
            job = manager.start(dict(source='huggingface', task='summarization', name='org/texts',
                                     splits=['test', 'train'], subset='[1:5]'))
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(info['task'], 'summarization')
            self.assertEqual(info['count'], 4)
            self.assertEqual(info['source_count'], 5)
            self.assertEqual(info['subset_start'], 1)
            self.assertEqual(info['split_counts'], [{'name': 'test', 'count': 2}, {'name': 'train', 'count': 3}])
            self.assertEqual(info['text_column'], 'article')
            self.assertEqual(info['summary_column'], 'summary')
            self.assertEqual(info['reference_count'], 1)
            self.assertEqual(info['missing_reference_count'], 3)
            self.assertEqual(info['invalid_text_count'], 0)
            self.assertIsInstance(manager.records[job['id']], Dataset)
            self.assertEqual(manager.records[job['id']].column_names, ['article', 'summary'])
            self.assertEqual(list(Path(tmp).iterdir()), [], 'Dataset contents must remain in Hugging Face storage')
            self.assertEqual(manager.text(job['id'], 2), dict(index=2, name='train[1]', text='train one',
                summary='ref one', source_split='train', source_index=1))
            self.assertIsNone(manager.text(job['id'], 0)['summary'])
            self.assertEqual(manager.page(job['id'])['items'], [
                dict(index=0, name='test[1]', text_preview='test one', has_reference=False, source_split='test', source_index=1),
                dict(index=1, name='train[0]', text_preview='train zero', has_reference=False, source_split='train', source_index=0),
                dict(index=2, name='train[1]', text_preview='train one', has_reference=True, source_split='train', source_index=1),
                dict(index=3, name='train[2]', text_preview='train two', has_reference=False, source_split='train', source_index=2),
            ])

    def test_previews_are_short_and_full_text_and_reference_remain_independent(self):
        source = 'Source paragraph. ' * 200
        reference = 'Independent gold summary.'
        dataset = text_rows([source], [reference], 'body', 'gold')
        with tempfile.TemporaryDirectory() as tmp, patch('datasets.load_dataset', return_value=dataset):
            manager = DatasetManager(tmp)
            job = manager.start(dict(source='huggingface', task='summarization', name='org/texts',
                                     text_column='body', summary_column='gold'))
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            item = manager.page(job['id'])['items'][0]
            self.assertEqual(item.get('text_preview'), source[:200])
            self.assertNotIn('text', item)
            self.assertNotIn('summary', item)
            self.assertTrue(item['has_reference'])
            full = manager.text(job['id'], 0)
            self.assertEqual(full['text'], source)
            self.assertEqual(full['summary'], reference)
            for index in [-1, 1, True, '0']:
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'index'):
                    manager.text(job['id'], index)
            with self.assertRaisesRegex(ValueError, 'image|classification'):
                manager.image(job['id'], 0)

    def test_default_and_count_subsets_keep_all_selected_text_rows(self):
        dataset = text_rows([f'Source {index}' for index in range(1031)], [None] * 1031)
        for subset, expected in [('', 1031), ('20', 20), ('[20:30]', 10)]:
            with self.subTest(subset=subset), tempfile.TemporaryDirectory() as tmp, \
                 patch('datasets.load_dataset', return_value=dataset):
                manager = DatasetManager(tmp)
                job = manager.start(dict(source='huggingface', task='summarization', name='org/texts', subset=subset))
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'ready', info.get('error'))
                self.assertEqual(info['count'], expected)
                self.assertEqual(info['source_count'], 1031)
                self.assertEqual(info.get('reference_count'), 0)
                self.assertEqual(info['missing_reference_count'], expected)
                self.assertEqual(manager.text(job['id'], 0)['text'], f'Source {20 if subset.startswith("[") else 0}')

    def test_missing_ambiguous_and_incompatible_columns_fail_with_selection_errors(self):
        cases = [
            ({'id': ['1'], 'summary': ['Ref']}, {}, 'source text column'),
            ({'article': ['A'], 'text': ['B'], 'summary': ['Ref']}, {}, 'source text column'),
            ({'article': ['A'], 'summary': ['Ref'], 'abstract': ['Ref 2']}, {}, 'reference summary column'),
            ({'article': ['A'], 'id': ['1']}, {}, 'reference summary column'),
            ({'article': ['A'], 'summary': ['Ref']}, {'text_column': 'missing'}, 'source text column'),
            ({'article': ['A'], 'summary': ['Ref']}, {'summary_column': 'missing'}, 'reference summary column'),
            ({'article': ['A'], 'summary': ['Ref']}, {'summary_column': 'article'}, 'different'),
            ({'article': ['A'], 'score': [1]}, {'summary_column': 'score'}, 'reference summary column'),
        ]
        for rows, fields, message in cases:
            with self.subTest(rows=rows, fields=fields), tempfile.TemporaryDirectory() as tmp, \
                 patch('datasets.load_dataset', return_value=Dataset.from_dict(rows)):
                manager = DatasetManager(tmp)
                job = manager.start(dict(source='huggingface', task='summarization', name='org/texts', **fields))
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'error')
                self.assertIn(message, info['error'])

    def test_explicit_columns_resolve_ambiguous_aliases(self):
        dataset = Dataset.from_dict({'article': ['correct source'], 'text': ['wrong source'],
                                     'summary': ['correct reference'], 'abstract': ['wrong reference']})
        with tempfile.TemporaryDirectory() as tmp, patch('datasets.load_dataset', return_value=dataset):
            manager = DatasetManager(tmp)
            job = manager.start(dict(source='huggingface', task='summarization', name='org/texts',
                                     text_column='article', summary_column='summary'))
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(manager.text(job['id'], 0)['text'], 'correct source')
            self.assertEqual(manager.text(job['id'], 0)['summary'], 'correct reference')

    def test_columns_cannot_shift_between_selected_splits(self):
        train = text_rows(['train source'], ['train reference'])
        for text_column, summary_column, message in [('document', 'summary', 'source text column'),
                                                     ('article', 'highlights', 'reference summary column')]:
            test = text_rows(['test source'], ['test reference'], text_column, summary_column)
            splits = {'train': train, 'test': test}
            with self.subTest(text_column=text_column, summary_column=summary_column), \
                 tempfile.TemporaryDirectory() as tmp, \
                 patch('data_sources.hf_dataset_options', return_value=self.options(splits)), \
                 patch('datasets.load_dataset', side_effect=lambda path, **kw: splits[kw['split']]):
                manager = DatasetManager(tmp)
                job = manager.start(dict(source='huggingface', task='summarization', name='org/texts',
                                         splits=['train', 'test']))
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'error')
                self.assertIn(message, info['error'])
                with self.assertRaises(ValueError):
                    manager.ready(job['id'])

    def test_invalid_source_rows_fail_without_silently_changing_counts(self):
        dataset = text_rows(['valid source', None, ' \n ', 'x' * 200001], ['ref', None, '', 'other ref'])
        splits = {'train': dataset}
        with tempfile.TemporaryDirectory() as tmp, \
             patch('data_sources.hf_dataset_options', return_value=self.options(splits)), \
             patch('datasets.load_dataset', return_value=dataset):
            manager = DatasetManager(tmp)
            job = manager.start(dict(source='huggingface', task='summarization', name='org/texts', splits=['train']))
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'error')
            self.assertEqual(info['count'], 4)
            self.assertEqual(info['invalid_text_count'], 3)
            self.assertEqual(info['reference_count'], 2)
            self.assertEqual(info['missing_reference_count'], 2)
            self.assertIn('row 2', info['error'])
            self.assertIn('train[1]', info['error'])
            self.assertIn('empty', info['error'])

    def test_oversized_text_and_reference_rows_report_limits(self):
        for source, reference, limit in [('x' * 200001, 'ref', '200000'), ('source', 'r' * 50001, '50000')]:
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as tmp, \
                 patch('datasets.load_dataset', return_value=text_rows([source], [reference])):
                manager = DatasetManager(tmp)
                job = manager.start(dict(source='huggingface', task='summarization', name='org/texts'))
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'error')
                self.assertEqual(info['count'], 1)
                self.assertIn(limit, info['error'])

    def test_range_overflow_and_unknown_splits_are_rejected(self):
        dataset = text_rows(['A', 'B'], [None, 'Ref'])
        for fields, error in [({'subset': '3'}, 'exceeds'), ({'subset': '[1:3]'}, 'exceeds'),
                               ({'splits': ['test']}, 'train')]:
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as tmp, \
                 patch('data_sources.hf_dataset_options', return_value=self.options({'train': dataset})), \
                 patch('datasets.load_dataset', return_value=dataset):
                manager = DatasetManager(tmp)
                job = manager.start(dict(source='huggingface', task='summarization', name='org/texts', **fields))
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'error')
                self.assertIn(error, info['error'])

    def test_rejects_incompatible_task_source_and_invalid_column_types(self):
        invalid = [dict(task='segmentation'), dict(task=None), dict(task=[]),
                   dict(task='summarization', source='folder'), dict(task='summarization', source='upload'),
                   dict(task='summarization', text_column=[]), dict(task='summarization', summary_column=None)]
        with tempfile.TemporaryDirectory() as tmp, patch('data_sources.threading.Thread'):
            manager = DatasetManager(tmp)
            for fields in invalid:
                payload = dict(source='huggingface', name='org/texts')
                payload.update(fields)
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    manager.start(payload)

    def test_classification_remains_default_and_cannot_supply_text(self):
        dataset = Dataset.from_dict({'image': [None], 'label': ['cat']},
                                    features=Features({'image': Image(), 'label': Value('string')}))
        with tempfile.TemporaryDirectory() as tmp, patch('datasets.load_dataset', return_value=dataset):
            manager = DatasetManager(tmp)
            job = manager.start(dict(source='huggingface', name='org/images'))
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(info.get('task'), 'classification')
            self.assertEqual(manager.page(job['id'])['items'], [dict(index=0, name='Image 1', label='cat')])
            with self.assertRaisesRegex(ValueError, 'summarization|text dataset'):
                manager.text(job['id'], 0)


if __name__ == '__main__':
    unittest.main()
