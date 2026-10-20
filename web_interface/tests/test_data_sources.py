import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from data_sources import DatasetManager


def image_rows(indices, with_labels=True):
    from datasets import Audio, Dataset, Features, Image as HFImage, Value
    data = {'photo': [{'path': f'row-{index}.png', 'bytes': None} for index in indices],
            'unrelated_audio': [{'path': 'must-not-decode.wav', 'bytes': None} for _ in indices]}
    features = {'photo': HFImage(), 'unrelated_audio': {'path': Value('string'), 'bytes': Value('binary')}}
    if with_labels:
        data['label'] = [f'class-{index}' for index in indices]
        features['label'] = Value('string')
    return Dataset.from_dict(data, features=Features(features)).cast_column('unrelated_audio', Audio())


class DatasetTests(unittest.TestCase):
    def test_selected_splits_are_combined_in_order_before_subsetting(self):
        datasets = {'train': image_rows(range(30)), 'test': image_rows(range(100, 110))}
        options = dict(name='org/images', configs=['default'], config='default',
                       splits=[dict(name='train', count=30), dict(name='test', count=10)],
                       image_columns=['photo'], label_columns=['label'])
        # An ignored subset or reordered split list changes both these image paths and labels.
        with tempfile.TemporaryDirectory() as tmp, \
             patch('data_sources.hf_dataset_options', return_value=options), \
             patch('datasets.load_dataset', side_effect=lambda path, **kw: datasets[kw['split']]):
            manager = DatasetManager(tmp)
            job = manager.start({'source': 'huggingface', 'name': 'org/images',
                                 'splits': ['test', 'train'], 'subset': '[8:13]'})
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(info['count'], 5)
            self.assertEqual(info['source_count'], 40)
            self.assertEqual(info['splits'], ['test', 'train'])
            self.assertEqual(info['subset'], '[8:13]')
            self.assertEqual(info['config'], 'default')
            self.assertEqual(info.get('split_counts'), [{'name': 'test', 'count': 10}, {'name': 'train', 'count': 30}])
            self.assertEqual(info.get('subset_start'), 8)
            self.assertEqual([row['photo']['path'] for row in manager.records[job['id']]],
                             ['row-108.png', 'row-109.png', 'row-0.png', 'row-1.png', 'row-2.png'])
            self.assertEqual(manager.labels(job['id'])['values'],
                             {'0': 'class-108', '1': 'class-109', '2': 'class-0', '3': 'class-1', '4': 'class-2'})
            self.assertEqual(set(manager.records[job['id']].column_names), {'photo', 'label'})
            self.assertFalse(manager.records[job['id']].features['photo'].decode)
            self.assertEqual(manager.page(job['id'], 1, 3)['items'], [
                {'index': 1, 'name': 'test[9]', 'source_split': 'test', 'source_index': 9, 'label': 'class-109'},
                {'index': 2, 'name': 'train[0]', 'source_split': 'train', 'source_index': 0, 'label': 'class-0'},
                {'index': 3, 'name': 'train[1]', 'source_split': 'train', 'source_index': 1, 'label': 'class-1'},
            ])

    def test_blank_subset_keeps_full_split_and_count_subset_keeps_first_rows(self):
        dataset = image_rows(range(1031))
        for subset, expected_count in [('', 1031), ('20', 20), ('[20:30]', 10)]:
            with self.subTest(subset=subset), tempfile.TemporaryDirectory() as tmp, \
                 patch('datasets.load_dataset', return_value=dataset) as load:
                manager = DatasetManager(tmp)
                job = manager.start({'source': 'huggingface', 'name': 'org/images', 'subset': subset})
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'ready', info.get('error'))
                self.assertEqual(info['count'], expected_count)
                self.assertEqual(info.get('source_count'), 1031)
                self.assertEqual(info.get('splits'), ['train'])
                load.assert_called_once_with('org/images', name=None, split='train',
                                             **({'streaming': True} if subset else {}))
                start = 20 if subset == '[20:30]' else 0
                self.assertEqual(manager.labels(job['id'])['values']['0'], f'class-{start}')

    def test_subset_bounds_are_not_silently_clamped(self):
        dataset = image_rows(range(30))
        for subset in ['31', '[20:31]', '[30:31]']:
            with self.subTest(subset=subset), tempfile.TemporaryDirectory() as tmp, \
                 patch('datasets.load_dataset', return_value=dataset):
                manager = DatasetManager(tmp)
                job = manager.start({'source': 'huggingface', 'name': 'org/images', 'subset': subset})
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'error')
                self.assertIn('30', info['error'])

    def test_multiple_splits_keep_images_even_when_one_split_has_no_labels(self):
        datasets = {'train': image_rows([0, 1]), 'test': image_rows([2, 3], with_labels=False)}
        options = dict(name='org/images', configs=['default'], config='default',
                       splits=[dict(name='train', count=2), dict(name='test', count=2)],
                       image_columns=['photo'], label_columns=['label'])
        with tempfile.TemporaryDirectory() as tmp, \
             patch('data_sources.hf_dataset_options', return_value=options), \
             patch('datasets.load_dataset', side_effect=lambda path, **kw: datasets[kw['split']]):
            manager = DatasetManager(tmp)
            job = manager.start({'source': 'huggingface', 'name': 'org/images', 'splits': ['train', 'test']})
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(info['count'], 4)
            self.assertEqual(manager.labels(job['id'])['values'], {'0': 'class-0', '1': 'class-1'})

    def test_unknown_split_reports_available_splits_without_loading_images(self):
        options = dict(name='org/images', configs=['default'], config='default',
                       splits=[dict(name='validation', count=8)], image_columns=['photo'], label_columns=[])
        with tempfile.TemporaryDirectory() as tmp, \
             patch('data_sources.hf_dataset_options', return_value=options), \
             patch('datasets.load_dataset', side_effect=AssertionError('Must validate the split before loading')):
            manager = DatasetManager(tmp)
            job = manager.start({'source': 'huggingface', 'name': 'org/images', 'splits': ['train']})
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'error')
            self.assertIn('validation', info['error'])

    def test_explicit_label_column_selects_the_requested_taxonomy(self):
        from datasets import Dataset, Features, ClassLabel, Image as HFImage, Value
        dataset = Dataset.from_dict({'image': [None, None], 'fine': [0, 1], 'coarse': [0, 0],
                                     'caption_class': ['first', 'second']},
            features=Features({'image': HFImage(), 'fine': ClassLabel(names=['cat', 'dog']),
                               'coarse': ClassLabel(names=['animal']), 'caption_class': Value('string')}))
        for column, expected in [('coarse', {'0': 'animal', '1': 'animal'}),
                                 ('caption_class', {'0': 'first', '1': 'second'})]:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as tmp, \
                 patch('datasets.load_dataset', return_value=dataset):
                manager = DatasetManager(tmp)
                job = manager.start({'source': 'huggingface', 'name': 'org/images', 'label_column': column})
                info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'ready', info.get('error'))
                self.assertEqual(info['label_column'], column)
                self.assertEqual(manager.labels(job['id'])['values'], expected)

    def test_rejects_invalid_split_subset_and_label_selection_types(self):
        invalid = [{'splits': []}, {'splits': 'train'}, {'splits': None}, {'splits': ['train', 'train']},
                   {'splits': ['train', ' train ']}, {'splits': ['']}, {'splits': ['train', 2]},
                   {'splits': ['train'], 'split': False}, {'splits': ['train'], 'split': 'test'},
                   {'split': None}, {'subset': None}, {'subset': 20}, {'subset': '0'}, {'subset': '-1'},
                   {'subset': '[:20]'}, {'subset': '[20:]'}, {'subset': '[20:20]'}, {'subset': '[30:20]'},
                   {'subset': '[1:3:2]'}, {'subset': '20%'}, {'label_column': []}, {'label_column': None}]
        with tempfile.TemporaryDirectory() as tmp, patch('data_sources.threading.Thread'):
            manager = DatasetManager(tmp)
            for fields in invalid:
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    manager.start(dict(source='huggingface', name='org/images', **fields))

    def test_hf_labels_are_retained_and_decoded_without_reading_unrelated_media(self):
        from datasets import Dataset, Features, ClassLabel, Image as HFImage, Value
        dataset = Dataset.from_dict(
            {'photo': [None]*4, 'labels': [1, 0, -1, None], 'unrelated': ['ignore']*4},
            features=Features({'photo': HFImage(), 'labels': ClassLabel(names=['cat', 'dog']),
                               'unrelated': Value('string')}))
        with tempfile.TemporaryDirectory() as tmp, patch('datasets.load_dataset', return_value=dataset):
            manager = DatasetManager(Path(tmp))
            job = manager.start({'source': 'huggingface', 'name': 'org/labelled', 'split': 'test'})
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(info.get('label_count'), 2)
            self.assertEqual(manager.labels(job['id'])['values'], {'0': 'dog', '1': 'cat'})
            self.assertEqual(set(manager.records[job['id']].column_names), {'photo', 'labels'})
            self.assertEqual(manager.page(job['id'])['items'][0]['label'], 'dog')

    def test_string_labels_are_retained_but_numeric_labels_without_names_need_review(self):
        from datasets import Dataset, Features, Image as HFImage, Value
        for dtype, values, expected in [('string', ['cat', None, ' '], {'0': 'cat'}),
                                         ('int64', [0, 1, None], {})]:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                dataset = Dataset.from_dict({'image': [None]*3, 'label': values},
                    features=Features({'image': HFImage(), 'label': Value(dtype)}))
                with patch('datasets.load_dataset', return_value=dataset):
                    manager = DatasetManager(Path(tmp))
                    job = manager.start({'source': 'huggingface', 'name': 'org/labels'})
                    info = self.wait(manager, job['id'])
                self.assertEqual(info['status'], 'ready', info.get('error'))
                self.assertEqual(manager.labels(job['id'])['values'], expected)

    def test_ambiguous_label_columns_do_not_choose_a_taxonomy_silently(self):
        from datasets import Dataset, Features, ClassLabel, Image as HFImage
        dataset = Dataset.from_dict({'image': [None], 'fine': [0], 'coarse': [0]},
            features=Features({'image': HFImage(), 'fine': ClassLabel(names=['cat']),
                               'coarse': ClassLabel(names=['animal'])}))
        with tempfile.TemporaryDirectory() as tmp, patch('datasets.load_dataset', return_value=dataset):
            manager = DatasetManager(Path(tmp))
            job = manager.start({'source': 'huggingface', 'name': 'org/ambiguous'})
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(info.get('label_count'), 0)
            self.assertIn('Multiple', info['label_message'])

    def wait(self, manager, dataset_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            info = manager.get(dataset_id)
            if info['status'] != 'loading':
                return info
            time.sleep(.01)
        self.fail('Dataset did not finish loading')

    def test_folder_recurses_and_keeps_images_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'images'
            (root/'cats').mkdir(parents=True)
            (root/'dogs').mkdir()
            Image.new('RGB', (9, 7), 'red').save(root/'cats'/'photo.png')
            Image.new('RGB', (9, 7), 'blue').save(root/'dogs'/'photo.png')
            (root/'notes.txt').write_text('not an image')
            manager = DatasetManager(Path(tmp)/'cache')
            job = manager.start({'source':'folder', 'path':str(root)})
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(info['count'], 2)
            page = manager.page(job['id'], 0, 1)
            self.assertEqual(page['items'], [{'index':0, 'name':'cats/photo.png'}])
            self.assertEqual(manager.image(job['id'], 0).getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(manager.image(job['id'], 1).getpixel((0, 0)), (0, 0, 255))
            with self.assertRaises(ValueError):
                manager.image(job['id'], -1)

    def test_hf_dataset_id_split_and_column_reach_loader(self):
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), 'green').save(buffer, format='PNG')
        rows = [{'photo': {'bytes':buffer.getvalue(), 'path':None}}]
        with tempfile.TemporaryDirectory() as tmp, patch('data_sources.load_hf_images', return_value=(rows, 'photo')) as load:
            manager = DatasetManager(Path(tmp))
            job = manager.start({'source':'huggingface', 'name':'org/data', 'split':'test[:10]', 'config':'small', 'column':'photo'})
            info = self.wait(manager, job['id'])
            self.assertEqual(info['status'], 'ready', info.get('error'))
            self.assertEqual(info['count'], 1)
            load.assert_called_once_with('org/data', 'test[:10]', 'small', 'photo')
            self.assertEqual(manager.page(job['id'])['items'], [{'index': 0, 'name': 'Image 1'}])
            self.assertEqual(manager.image(job['id'], 0).size, (224, 224))

    def test_browser_folder_upload_requires_all_files_and_sanitizes_paths(self):
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), 'red').save(buffer, format='PNG')
        payload = buffer.getvalue()
        with tempfile.TemporaryDirectory() as tmp:
            manager = DatasetManager(Path(tmp))
            with self.assertRaises(ValueError):
                manager.start({'source':'upload', 'name':'folder', 'files':[{'name':'../bad.png', 'size':len(payload)}]})
            job = manager.start({'source':'upload', 'name':'folder', 'files':[{'name':'folder/photo.png', 'size':len(payload)}]})
            with self.assertRaises(ValueError):
                manager.finish_upload(job['id'])
            manager.upload(job['id'], 0, payload)
            self.assertEqual(manager.finish_upload(job['id'])['status'], 'ready')
            self.assertEqual(manager.image(job['id'], 0).getpixel((0, 0)), (255, 0, 0))

    def test_empty_folder_produces_readable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = DatasetManager(Path(tmp)/'cache')
            job = manager.start({'source':'folder', 'path':tmp})
            self.assertIn('No supported images', self.wait(manager, job['id'])['error'])


if __name__ == '__main__':
    unittest.main()
