import tempfile
import io
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from server import PreviewManager, validate_config
from models import ModelManager


class PreviewTests(unittest.TestCase):
    def config(self, manager, values=None, count=1):
        root = manager.directory/'test-inputs'
        root.mkdir(exist_ok=True)
        for i in range(count):
            Image.new('RGB', (8, 8), ('red', 'blue')[i%2]).save(root/f'{i}.png')
        dataset = manager.datasets.start({'source':'folder', 'path':str(root)})
        deadline = time.monotonic() + 5
        while manager.datasets.get(dataset['id'])['status'] == 'loading' and time.monotonic() < deadline:
            time.sleep(.01)
        return validate_config({'dataset_id':dataset['id'], **(values or {})})

    def wait(self, manager, preview_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = manager.get(preview_id)
            if result['status'] not in ('queued', 'running'):
                return result
            time.sleep(.01)
        self.fail('Preview did not finish')

    def test_entire_dataset_action_space_is_available_with_lazy_image_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PreviewManager(Path(tmp))
            preview = manager.start(self.config(manager, count=2))
            result = self.wait(manager, preview['id'])
            self.assertEqual(result['status'], 'completed', result.get('error'))
            self.assertEqual(len(result['nodes']), 125)
            self.assertEqual(result['input_count'], 2)
            self.assertEqual(result['total'], 250)
            self.assertTrue(all('prediction' not in node for node in result['nodes']))
            self.assertFalse((Path(tmp)/preview['id']/'input-1').exists())
            second = manager.get(preview['id'], 1)
            self.assertIn('/input-1/', second['nodes'][124]['image'])
            pixels = manager.image(preview['id'], 1, 'action-124')
            self.assertEqual(Image.open(io.BytesIO(pixels)).size, (224, 224))
            original = Image.open(io.BytesIO(manager.image(preview['id'], 1, 'original')))
            self.assertEqual(original.getpixel((100, 100)), (0, 0, 255))
            with self.assertRaises(ValueError):
                manager.get(preview['id'], 2)

    def test_removed_dimensions_and_values_are_not_generated(self):
        space = dict(rotation=[-5, 5], darkness=[.2], saturation=[])
        with tempfile.TemporaryDirectory() as tmp:
            manager = PreviewManager(Path(tmp))
            preview = manager.start(self.config(manager, {'action_space': space}))
            result = self.wait(manager, preview['id'])
            self.assertEqual(len(result['nodes']), 2)
            self.assertEqual(result['config']['action_space'], space)
            self.assertEqual(result['nodes'][1]['action'], dict(id=1, rotation=5, darkness=.2, saturation=1))

    def test_generation_previews_prompts_without_generating_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PreviewManager(Path(tmp))
            preview = manager.start(validate_config({'task':'generation'}))
            result = self.wait(manager, preview['id'])
            self.assertEqual(len(result['nodes']), 27)
            self.assertIn('scientist', result['nodes'][0]['text'])
            self.assertNotIn('{', result['nodes'][0]['text'])
            self.assertNotIn('image', result['nodes'][0])

    def test_summary_previews_selected_text_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PreviewManager(Path(tmp))
            preview = manager.start(validate_config({'task':'summarization','text':'HELLO World!', 'action_space':{'actions':['to_lowercase']}}))
            result = self.wait(manager, preview['id'])
            self.assertEqual(result['nodes'][0]['text'], 'hello world!')

    def test_additions_outside_fixed_action_space_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({'action_space':{'rotation':[20], 'darkness':[], 'saturation':[]}})

    def test_no_actions_and_invalid_input_are_rejected(self):
        for config in ({'action_space':{'rotation':[], 'darkness':[], 'saturation':[]}},
                       {'task':{}}, {'task':'generation','text':'missing slots'}, {'dataset_id':''}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)


class ModelTests(unittest.TestCase):
    def test_local_model_is_selected_without_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'model.pth'
            checkpoint.write_bytes(b'not a pickle')
            model = ModelManager().select({'source':'local','name':str(checkpoint)})
            self.assertEqual(model['status'], 'ready')
            self.assertEqual(model['path'], str(checkpoint))

    def test_huggingface_model_downloads_without_an_allow_download_flag(self):
        manager = ModelManager()
        with patch('models.download_snapshot', return_value='/tmp/cached-model') as download:
            job = manager.select({'source':'huggingface','name':'org/model'})
            deadline = time.monotonic() + 5
            while manager.get(job['id'])['status'] == 'downloading' and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(manager.get(job['id'])['status'], 'ready')
            download.assert_called_once_with('org/model')


if __name__ == '__main__':
    unittest.main()
