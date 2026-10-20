import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, Features, Image, Value

import data_sources
from data_sources import DatasetManager, hf_dataset_options


def prompt_rows(prompts, checks=None, column='prompt', checks_column='checks'):
    values = {column: prompts,
              'unrelated_image': [{'path': 'must-not-decode.png', 'bytes': None}] * len(prompts),
              'summary': ['Not a generation reference'] * len(prompts)}
    features = {column: Value('string'), 'unrelated_image': Image(), 'summary': Value('string')}
    if checks is not None:
        values[checks_column] = checks
        features[checks_column] = [[Value('string')]]
    return Dataset.from_dict(values, features=Features(features))


class GenerationDatasetTests(unittest.TestCase):
    def start(self, manager, **fields):
        try:
            return manager.start(dict(source='huggingface', task='generation', name='org/prompts', **fields))
        except ValueError as exc:
            self.fail(f'Valid generation dataset was rejected: {exc}')

    def wait(self, manager, dataset_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            info = manager.get(dataset_id)
            if info['status'] != 'loading':
                return info
            time.sleep(.01)
        self.fail('Dataset did not finish loading')

    def options(self, splits):
        return dict(name='org/prompts', configs=['default'], config='default',
                    splits=[dict(name=name, count=len(rows)) for name, rows in splits.items()])

    def test_metadata_discovers_prompt_and_supported_check_columns(self):
        features = Features({'Prompt': Value('string'), 'Category': Value('string'),
                             'vqa_list': [[Value('string')]],
                             'checks': [{'question': Value('string'), 'answer': Value('string'), 'skill': Value('string')}],
                             'tags': [Value('string')], 'image': Image()})
        with patch('datasets.get_dataset_config_names', return_value=['default']), \
             patch('datasets.get_dataset_config_info', return_value=SimpleNamespace(
                 features=features, splits={'train': {'num_examples': 1632}})):
            result = hf_dataset_options('nateraw/parti-prompts')
        self.assertEqual(result.get('prompt_columns'), ['Prompt', 'Category'])
        self.assertEqual(result.get('suggested_prompt_column'), 'Prompt')
        self.assertEqual(result.get('checks_columns'), ['vqa_list', 'checks'])
        self.assertIsNone(result.get('suggested_checks_column'))

    def test_multisplit_subset_preserves_all_prompts_checks_and_provenance(self):
        splits = {'train': prompt_rows(['train zero', 'train one', 'train two'],
                                      [None, [['How many cups?', 'two']], []]),
                  'test': prompt_rows(['test zero', 'test one'], [[['Is it red?', 'Yes']], []])}
        with tempfile.TemporaryDirectory() as tmp, \
             patch('data_sources.hf_dataset_options', return_value=self.options(splits)), \
             patch('datasets.load_dataset', side_effect=lambda path, **kw: splits[kw['split']]), \
             patch('data_sources.extract_dataset_labels', side_effect=AssertionError('Prompts have no class labels')):
            manager = DatasetManager(tmp)
            job = self.start(manager, splits=['test', 'train'], subset='[1:5]', checks_column='checks')
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(info['count'], 4)
            self.assertEqual(info['source_count'], 5)
            self.assertEqual(info['checked_count'], 1)
            self.assertEqual(info['missing_checks_count'], 3)
            self.assertEqual(info['prompt_column'], 'prompt')
            self.assertEqual(info['checks_column'], 'checks')
            self.assertEqual(info['split_counts'], [{'name': 'test', 'count': 2}, {'name': 'train', 'count': 3}])
            self.assertIsInstance(manager.records[job['id']], Dataset)
            self.assertEqual(manager.records[job['id']].column_names, ['prompt', 'checks'])
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertEqual(manager.text(job['id'], 2), dict(index=2, name='train[1]', text='train one',
                checks=[{'question': 'How many cups?', 'answer': 'two'}], source_split='train', source_index=1))
            self.assertEqual(manager.page(job['id'])['items'], [
                dict(index=0, name='test[1]', text_preview='test one', check_count=0, source_split='test', source_index=1),
                dict(index=1, name='train[0]', text_preview='train zero', check_count=0, source_split='train', source_index=0),
                dict(index=2, name='train[1]', text_preview='train one', check_count=1, source_split='train', source_index=1),
                dict(index=3, name='train[2]', text_preview='train two', check_count=0, source_split='train', source_index=2)])
            self.assertNotIn('reference_count', info)
            self.assertNotIn('summary', manager.text(job['id'], 2))
            with self.assertRaisesRegex(ValueError, 'image|classification'):
                manager.image(job['id'], 0)

    def test_all_count_and_range_keep_prompt_only_rows_and_single_split_identity(self):
        dataset = prompt_rows([f'Prompt {i}' for i in range(1031)], column='Prompt')
        for subset, count, first in [('', 1031, 0), ('all', 1031, 0), ('20', 20, 0), ('[20:30]', 10, 20)]:
            with self.subTest(subset=subset), tempfile.TemporaryDirectory() as tmp, \
                 patch('datasets.load_dataset', return_value=dataset):
                manager = DatasetManager(tmp)
                job = self.start(manager, prompt_column='Prompt', subset=subset)
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'ready', info.get('error'))
                self.assertEqual(info['count'], count)
                self.assertEqual(info['checked_count'], 0)
                self.assertEqual(info['missing_checks_count'], count)
                self.assertEqual(manager.text(job['id'], 0), dict(index=0, name=f'train[{first}]',
                    text=f'Prompt {first}', checks=[], source_split='train', source_index=first))

    def test_preview_is_bounded_without_truncating_full_prompt(self):
        prompt = 'Two blue cups beside a red bowl. ' * 100
        with tempfile.TemporaryDirectory() as tmp, patch('datasets.load_dataset', return_value=prompt_rows([prompt])):
            manager = DatasetManager(tmp)
            job = self.start(manager)
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(manager.page(job['id'])['items'][0]['text_preview'], prompt[:200])
            self.assertEqual(manager.text(job['id'], 0)['text'], prompt)
            for index in [-1, 1, True, '0']:
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'index'):
                    manager.text(job['id'], index)

    def test_invalid_prompts_and_checks_fail_with_row_identity_without_drops(self):
        cases = [(prompt_rows(['good', None, ' ']), {}, 'Prompt at row 2', 'train[1]'),
                 (prompt_rows(['good', 'x' * 10001]), {}, '10000', 'train[1]'),
                 (prompt_rows(['good', 'bad'], [[['q', 'a']], [['q', '']]]),
                  {'checks_column': 'checks'}, 'checks at row 2', 'train[1]')]
        for dataset, fields, message, identity in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp, \
                 patch('datasets.load_dataset', return_value=dataset):
                manager = DatasetManager(tmp)
                job = self.start(manager, **fields)
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'error')
                self.assertEqual(info['count'], len(dataset))
                self.assertIn(message, info['error'])
                self.assertIn(identity, info['error'])
                with self.assertRaises(ValueError):
                    manager.ready(job['id'])

    def test_ambiguous_missing_or_invalid_columns_report_selection_error(self):
        cases = [({'caption': ['one'], 'prompt': ['two']}, {}, 'prompt column'),
                 ({'body': ['one']}, {}, 'prompt column'),
                 ({'prompt': ['one']}, {'prompt_column': 'missing'}, 'prompt column'),
                 ({'prompt': ['one'], 'checks': ['not a check list']}, {'checks_column': 'checks'}, 'checks column'),
                 ({'prompt': ['one']}, {'checks_column': 'missing'}, 'checks column')]
        for rows, fields, message in cases:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as tmp, \
                 patch('datasets.load_dataset', return_value=Dataset.from_dict(rows)):
                manager = DatasetManager(tmp)
                job = self.start(manager, **fields)
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'error')
                self.assertIn(message, info['error'])

    def test_published_benchmark_has_one_test_split_and_uses_pinned_arrow_json(self):
        helper = getattr(data_sources, 'benchmark_dataset_options', None)
        self.assertTrue(callable(helper), 'Benchmark metadata must be available without loading rows')
        options = helper('geneval2')
        self.assertEqual(options['splits'], [{'name': 'test', 'count': 800}])
        self.assertEqual(options['suggested_prompt_column'], 'prompt')
        self.assertEqual(options['suggested_checks_column'], 'vqa_list')
        dataset = Dataset.from_dict({'prompt': ['two blue cups'], 'vqa_list': [[['How many cups?', 'two']]],
                                     'skills': [['count']], 'atom_count': [3]})
        def load(path, **kwargs):
            self.assertEqual(path, 'json')
            self.assertEqual(kwargs.get('split'), 'test')
            self.assertEqual(kwargs.get('data_files'), {'test':
                'https://raw.githubusercontent.com/facebookresearch/GenEval2/a6e82d2289e8d418f27f0adee77908b07060eea3/geneval2_data.jsonl'})
            return dataset
        with tempfile.TemporaryDirectory() as tmp, patch('datasets.load_dataset', side_effect=load):
            manager = DatasetManager(tmp)
            try:
                job = manager.start(dict(source='benchmark', task='generation', name='geneval2'))
            except ValueError as exc:
                self.fail(f'Built-in benchmark was rejected: {exc}')
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(info['splits'], ['test'])
            self.assertEqual(info['checked_count'], 1)
            self.assertEqual(info['source_revision'], 'a6e82d2289e8d418f27f0adee77908b07060eea3')
            self.assertEqual(manager.text(job['id'], 0)['checks'],
                             [{'question': 'How many cups?', 'answer': 'two', 'skill': 'count'}])
            self.assertEqual(set(manager.records[job['id']].column_names), {'prompt', 'vqa_list', 'skills', 'atom_count'})

    def test_benchmark_rejects_invented_splits_and_invalid_tasks(self):
        with tempfile.TemporaryDirectory() as tmp, patch('datasets.load_dataset', side_effect=AssertionError('No download')):
            manager = DatasetManager(tmp)
            for fields in [{'splits': ['train']}, {'splits': ['test', 'train']}, {'split': 'test[:20]'},
                           {'task': 'classification'}, {'name': 'unknown'}]:
                payload = dict(source='benchmark', task='generation', name='geneval2')
                payload.update(fields)
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    manager.start(payload)

    def test_check_skill_metadata_cannot_silently_truncate_published_checks(self):
        for skills in [['count'], ['count', ''], ['count', None]]:
            dataset = Dataset.from_dict({'prompt': ['two blue cups'],
                'vqa_list': [[['How many cups?', 'two'], ['Are the cups blue?', 'Yes']]],
                'skills': [skills], 'atom_count': [3]})
            with self.subTest(skills=skills), tempfile.TemporaryDirectory() as tmp, \
                 patch('datasets.load_dataset', return_value=dataset):
                manager = DatasetManager(tmp)
                job = manager.start(dict(source='benchmark', task='generation', name='geneval2'))
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'error')
                self.assertEqual(info['count'], 1)
                self.assertEqual(info['invalid_checks_count'], 1)
                self.assertIn('test[0]', info['error'])
                self.assertIn('skill', info['error'])


if __name__ == '__main__':
    unittest.main()
