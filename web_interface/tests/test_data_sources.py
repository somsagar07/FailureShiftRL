import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from data_sources import DatasetManager


class DatasetTests(unittest.TestCase):
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
