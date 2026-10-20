import tempfile
import io
import time
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from server import PreviewManager, validate_config
from models import ModelManager


class PreviewTests(unittest.TestCase):
    def test_different_images_can_prepare_concurrently_and_keep_exact_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PreviewManager(Path(tmp))
            preview = manager.start(self.config(manager, count=2))
            self.wait(manager, preview['id'])
            read_image = manager.datasets.image
            arrived = threading.Barrier(2)

            def parallel_read(dataset_id, index):
                # Serializing unrelated inputs here would keep the GPU idle.
                arrived.wait(timeout=2)
                return read_image(dataset_id, index)

            with patch.object(manager.datasets, 'image', side_effect=parallel_read), ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(manager.image, preview['id'], index, 'original') for index in range(2)]
                images = [Image.open(io.BytesIO(future.result())).convert('RGB') for future in futures]
            self.assertEqual([image.getpixel((80, 80)) for image in images], [(255, 0, 0), (0, 0, 255)])

    def test_concurrent_consumers_share_one_frozen_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PreviewManager(Path(tmp))
            preview = manager.start(self.config(manager))
            self.wait(manager, preview['id'])
            read_image = manager.datasets.image

            def consume(index):
                if index % 2:
                    return manager.original_image(preview['id'], 0).tobytes()
                data = manager.image(preview['id'], 0, 'original')
                with Image.open(io.BytesIO(data)) as image:
                    return image.convert('RGB').tobytes()

            with patch.object(manager.datasets, 'image', wraps=read_image) as read, ThreadPoolExecutor(4) as pool:
                images = list(pool.map(consume, range(8)))
            self.assertEqual(read.call_count, 1)
            self.assertTrue(all(image == images[0] for image in images))

    def test_analysis_and_http_previews_do_not_write_duplicate_image_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PreviewManager(Path(tmp))
            preview = manager.start(self.config(manager))
            self.wait(manager, preview['id'])
            with patch.object(Image.Image, 'save', side_effect=AssertionError('Analysis must not encode PNG')):
                image = manager.original_image(preview['id'], 0)
            self.assertEqual(image.getpixel((80, 80)), (255, 0, 0))
            pixels = manager.image(preview['id'], 0, 'action-0')
            self.assertEqual(Image.open(io.BytesIO(pixels)).size, (224, 224))
            self.assertEqual(list((manager.directory/preview['id']).rglob('*.png')), [])

    def test_evicted_or_released_preview_rechecks_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PreviewManager(Path(tmp), cache_bytes=224*224*3)
            preview = manager.start(self.config(manager, count=2))
            self.wait(manager, preview['id'])
            original = manager.original_image(preview['id'], 0).tobytes()
            manager.original_image(preview['id'], 1)
            self.assertLessEqual(manager.cache.usage_bytes, 224*224*3)
            self.assertEqual(manager.original_image(preview['id'], 0).tobytes(), original)
            with manager.pin_preview(preview['id']):
                self.assertEqual(manager.release_preview(preview['id'])['status'], 'busy')
            self.assertEqual(manager.release_preview(preview['id'])['status'], 'released')
            path = manager.datasets.records[preview['config']['dataset_id']][0]['path']
            Image.new('RGB', (8, 8), 'green').save(path)
            with self.assertRaisesRegex(ValueError, 'changed.*[Aa]nalyze'):
                manager.original_image(preview['id'], 0)
            self.assertTrue(path.is_file())
            self.assertEqual(manager.get(preview['id'])['status'], 'completed')

    def test_source_replacement_during_decode_does_not_publish_mixed_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PreviewManager(Path(tmp))
            preview = manager.start(self.config(manager))
            self.wait(manager, preview['id'])
            source = manager.datasets.records[preview['config']['dataset_id']][0]['path']
            read = manager.datasets.image

            def replace_after_read(dataset_id, index):
                image = read(dataset_id, index)
                Image.new('RGB', (8, 8), 'blue').save(source)
                return image

            with patch.object(manager.datasets, 'image', side_effect=replace_after_read):
                with self.assertRaisesRegex(ValueError, 'changed.*[Aa]nalyze'):
                    manager.original_image(preview['id'], 0)
            self.assertEqual(manager.cache.usage_bytes, 0)
            fresh = manager.original_image(preview['id'], 0)
            self.assertEqual(fresh.getpixel((0, 0)), (0, 0, 255))
            fresh.putpixel((0, 0), (255, 255, 255))
            self.assertEqual(manager.original_image(preview['id'], 0).getpixel((0, 0)), (0, 0, 255))

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
            self.assertEqual(len(result['nodes']), 125)
            self.assertIn('scientist', result['nodes'][0]['text'])
            self.assertTrue(result['nodes'][0]['text'].startswith('Photograph style. Even lighting. Detail level: minimal. Scene: '))
            self.assertNotIn('{', result['nodes'][0]['text'])
            self.assertNotIn('image', result['nodes'][0])

    def test_generation_grouped_actions_do_not_substitute_literal_scene_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PreviewManager(Path(tmp))
            text = 'A sign reads {profession}; two green cups beside it.'
            started = manager.start(validate_config({'task':'generation','text':text,
                'action_space':{'style':[], 'lighting':['soft'], 'detail':['detailed']}}))
            result = self.wait(manager,started['id'])
            self.assertEqual(result['original']['text'],text)
            self.assertEqual(result['input_count'],1)
            self.assertEqual(len(result['nodes']),1)
            self.assertEqual(result['nodes'][0]['text'],'Soft lighting. Detail level: detailed. Scene: '+text)

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
                       {'task':{}}, {'task':'generation','text':''}, {'dataset_id':''}):
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
