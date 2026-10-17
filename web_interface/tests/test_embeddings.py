"""Embedding semantics with real Pillow inputs; model inference is injected."""
import hashlib
import io
import math
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from embeddings import EmbeddingManager, HuggingFaceEncoder
from server import PreviewManager, validate_config


class PixelEncoder:
    """A deterministic model boundary, never a production fallback."""
    feature_method = 'fixture RGB means plus constant'

    def __init__(self, model):
        self.model = model
        self.seen = []
        self.calls = []
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def encode(self, images):
        self.calls.append(len(images))
        if self.block:
            self.block = False
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError('Test did not release inference')
        values = []
        for image in images:
            self.seen.append(hashlib.sha256(image.tobytes()).hexdigest())
            means = np.asarray(image, dtype=np.float64).mean(axis=(0, 1)) / 255
            constant = 2 if self.model['name'] == 'fixture/second' else 1
            values.append([*means, constant])
        return np.asarray(values)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.previews = PreviewManager(self.root/'previews')
        self.encoders = []

        def factory(model):
            encoder = PixelEncoder(model)
            self.encoders.append(encoder)
            return encoder

        self.manager = EmbeddingManager(self.previews, self.root/'embeddings', encoder_factory=factory)

    def wait(self, getter):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            result = getter()
            if result['status'] not in ('queued', 'running', 'loading'):
                return result
            time.sleep(.005)
        self.fail('Background work did not finish')

    def preview(self, colors=('red', 'lime'), space=None, model='fixture/vision', dataset_id=None):
        if dataset_id is None:
            directory = self.root/f'images-{len(self.previews.datasets.jobs)}'
            directory.mkdir()
            for index, color in enumerate(colors):
                Image.new('RGB', (13, 9), color).save(directory/f'{index:03}.png')
            dataset = self.previews.datasets.start({'source': 'folder', 'path': str(directory)})
            dataset = self.wait(lambda: self.previews.datasets.get(dataset['id']))
            self.assertEqual(dataset['status'], 'ready', dataset.get('error'))
            dataset_id = dataset['id']
        config = validate_config(dict(
            dataset_id=dataset_id, model={'source': 'huggingface', 'name': model},
            action_space=space or {'rotation': [0], 'darkness': [.1, .5], 'saturation': []}))
        job = self.previews.start(config)
        job = self.wait(lambda: self.previews.get(job['id']))
        self.assertEqual(job['status'], 'completed', job.get('error'))
        return job

    def overview(self, preview):
        job = self.manager.start(preview['id'])
        result = self.wait(lambda: self.manager.get(job['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        return result

    def expansion(self, job, index):
        expansion = self.manager.expand(job['id'], index)
        result = self.wait(lambda: self.manager.get_expansion(job['id'], expansion['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        return result

    def test_overview_embeds_every_dataset_image_without_action_work(self):
        preview = self.preview(colors=['red', 'lime', 'blue'] * 7)
        result = self.overview(preview)
        self.assertEqual(result['completed'], 21)
        self.assertEqual(result['total'], 21)
        self.assertEqual(len(result['points']), 21)
        self.assertEqual(len(self.encoders[0].seen), 21)
        self.assertLess(max(self.encoders[0].calls), 21)
        self.assertEqual(result['feature_dim'], 4)
        self.assertEqual(result['feature_method'], PixelEncoder.feature_method)
        self.assertEqual(result['model'], {**preview['config']['model'], 'task': 'classification'})
        for index, point in enumerate(result['points']):
            self.assertEqual(point['index'], index)
            self.assertEqual(point['name'], f'{index:03}.png')
            self.assertEqual(point['image'], self.previews.get(preview['id'], index)['original']['image'])
            self.assertEqual(len(point['position']), 3)
        self.assertTrue(np.isfinite([point['position'] for point in result['points']]).all())

    def test_action_images_match_previews_and_use_original_pca_and_full_cosines(self):
        preview = self.preview()
        overview = self.overview(preview)
        np.testing.assert_allclose([point['position'] for point in overview['points']],
                                   [[.5, 0, 0], [-.5, 0, 0]], atol=1e-12)
        self.assertEqual(overview['projection']['rank'], 1)
        self.assertAlmostEqual(overview['projection']['variance'], 1)
        expansion = self.expansion(overview, 0)
        self.assertEqual(expansion['original'], overview['points'][0])
        self.assertEqual(expansion['completed'], 2)
        self.assertEqual(len(expansion['nodes']), 2)
        for node, red in zip(expansion['nodes'], [229 / 255, 127 / 255]):
            # Original PCA axis is (R-G)/sqrt(2); blue/constant directions are not refit.
            expected_x = red / math.sqrt(2 * (red * red + 1))
            np.testing.assert_allclose(node['position'], [expected_x, 0, 0], atol=1e-12)
            expected_cosine = (red + 1) / math.sqrt(2 * (red * red + 1))
            self.assertAlmostEqual(node['similarity'], expected_cosine)
            self.assertEqual(node['neighbors'][0]['index'], 0)
            self.assertAlmostEqual(node['neighbors'][0]['similarity'], expected_cosine)
            self.assertAlmostEqual(node['neighbors'][1]['similarity'], 1 / math.sqrt(2 * (red * red + 1)))
            rendered = Image.open(io.BytesIO(self.previews.image(preview['id'], 0, f'action-{node["id"]}')))
            digest = hashlib.sha256(rendered.convert('RGB').tobytes()).hexdigest()
            self.assertIn(digest, self.encoders[0].seen[2:])
            self.assertEqual(node['action'], preview['nodes'][node['id']]['action'])
            self.assertEqual(node['image'], preview['nodes'][node['id']]['image'])
        self.assertEqual(expansion['neighbors'][0]['index'], 1)
        self.assertAlmostEqual(expansion['neighbors'][0]['similarity'], .5)
        self.assertEqual(self.manager.get(overview['id'])['points'], overview['points'])

    def test_singleton_and_duplicate_inputs_have_no_invented_axes(self):
        for colors in [('red',), ('blue', 'blue', 'blue')]:
            with self.subTest(colors=colors):
                result = self.overview(self.preview(colors=colors))
                self.assertEqual(result['projection']['rank'], 0)
                self.assertEqual(result['projection']['variance'], 0.0)
                self.assertEqual(result['projection']['components'], [0.0, 0.0, 0.0])
                self.assertTrue(all(point['position'] == [0.0, 0.0, 0.0] for point in result['points']))
                expansion = self.expansion(result, 0)
                self.assertTrue(all(node['position'] == [0.0, 0.0, 0.0] for node in expansion['nodes']))
                self.assertLess(expansion['nodes'][-1]['similarity'], 1)

    def test_truncated_projection_reports_variance_without_using_plot_distance_as_similarity(self):
        result = self.overview(self.preview(colors=('red', 'lime', 'blue', 'black', 'white')))
        self.assertEqual(result['projection']['rank'], 4)
        self.assertGreater(result['projection']['variance'], 0)
        self.assertLess(result['projection']['variance'], 1)
        action = self.expansion(result, 0)['nodes'][-1]
        expected = 382 / math.sqrt(2 * (127 ** 2 + 255 ** 2))
        self.assertAlmostEqual(action['similarity'], expected)
        displayed_distance = np.linalg.norm(np.asarray(action['position']) - result['points'][0]['position'])
        self.assertNotAlmostEqual(1 - displayed_distance ** 2 / 2, expected, places=6)

    def test_source_changes_do_not_change_already_visible_preview_pixels(self):
        preview = self.preview()
        original = self.previews.image(preview['id'], 0, 'original')
        # The existing visible preview is a frozen PNG even if the source is replaced.
        path = self.previews.datasets.records[preview['config']['dataset_id']][0]['path']
        Image.new('RGB', (13, 9), 'blue').save(path)
        overview = self.overview(preview)
        digest = hashlib.sha256(Image.open(io.BytesIO(original)).convert('RGB').tobytes()).hexdigest()
        self.assertEqual(self.encoders[0].seen[0], digest)
        action = self.expansion(overview, 0)['nodes'][0]
        expected_pixels = Image.open(io.BytesIO(self.previews.image(preview['id'], 0, f'action-{action["id"]}')))
        action_digest = hashlib.sha256(expected_pixels.convert('RGB').tobytes()).hexdigest()
        self.assertEqual(self.encoders[0].seen[2], action_digest)

    def test_changed_pixels_cannot_reuse_another_previews_dataset_features(self):
        preview = self.preview()
        self.overview(preview)
        path = self.previews.datasets.records[preview['config']['dataset_id']][0]['path']
        Image.new('RGB', (13, 9), 'blue').save(path)
        changed = self.preview(dataset_id=preview['config']['dataset_id'])
        job = self.manager.start(changed['id'])
        result = self.wait(lambda: self.manager.get(job['id']))
        self.assertEqual(result['status'], 'error')
        self.assertIn('reload', result['error'].lower())
        self.assertFalse(result.get('points'))

    def test_cache_preserves_dataset_model_preview_and_action_identities(self):
        first_preview = self.preview()
        first = self.overview(first_preview)
        expansion = self.expansion(first, 0)
        count = sum(len(encoder.seen) for encoder in self.encoders)
        self.assertEqual(self.manager.start(first_preview['id'])['id'], first['id'])
        cached = self.expansion(first, 0)
        self.assertEqual(cached['nodes'], expansion['nodes'])
        self.assertEqual(sum(len(encoder.seen) for encoder in self.encoders), count)
        different_actions = self.preview(dataset_id=first_preview['config']['dataset_id'],
                                         space={'rotation': [0], 'darkness': [.2], 'saturation': []})
        second = self.overview(different_actions)
        self.assertNotEqual(first['id'], second['id'])
        self.assertIn(different_actions['id'], second['points'][0]['image'])
        self.assertEqual(sum(len(encoder.seen) for encoder in self.encoders), count)
        action_result = self.expansion(second, 0)
        self.assertEqual(action_result['nodes'][0]['action']['darkness'], .2)
        self.assertNotEqual(action_result['nodes'][0]['position'], expansion['nodes'][0]['position'])
        other_model = self.overview(self.preview(dataset_id=first_preview['config']['dataset_id'], model='fixture/second'))
        self.assertNotEqual(other_model['points'][0]['position'], first['points'][0]['position'])
        reloaded = self.overview(self.preview(colors=('blue', 'lime')))
        self.assertNotEqual(reloaded['id'], first['id'])
        self.assertEqual(reloaded['model']['name'], 'fixture/vision')
        self.assertGreater(sum(len(encoder.seen) for encoder in self.encoders), count + 2)

    def test_discovery_node_embeds_only_the_visited_action_in_the_existing_pca(self):
        preview = self.preview(colors=('red', 'lime', 'blue'))
        job = self.overview(preview)
        action_id = preview['nodes'][0]['id']
        count = len(self.encoders[0].seen)
        node = self.manager.discovery_node(job['id'], 1, action_id)
        self.assertEqual(len(self.encoders[0].seen), count + 1)
        self.assertEqual(node, self.manager.discovery_node(job['id'], 1, action_id))
        self.assertEqual(len(self.encoders[0].seen), count + 1)
        ordinary = next(entry for entry in self.expansion(job, 1)['nodes'] if entry['id'] == action_id)
        # BLAS uses different reductions for a single vector and a batch.
        np.testing.assert_allclose(node['position'], ordinary['position'], rtol=0, atol=1e-14)
        self.assertEqual({key: value for key, value in node.items() if key != 'position'},
                         {key: value for key, value in ordinary.items() if key != 'position'})
        with self.assertRaises(ValueError):
            self.manager.discovery_node(job['id'], 1, 'missing-action')

    def test_invalid_embeddings_fail_without_fabricated_points(self):
        for bad in (np.array([[0., 0.], [0., 0.]]),
                    np.array([[1., np.nan], [1., 2.]]),
                    np.array([[1., np.inf], [1., 2.]]), np.array([1., 2.])):
            with self.subTest(bad=bad):
                class InvalidEncoder:
                    feature_method = 'invalid fixture'
                    def encode(self, images):
                        return bad
                manager = EmbeddingManager(self.previews, self.root/'invalid', encoder_factory=lambda model: InvalidEncoder())
                preview = self.preview()
                result = manager.start(preview['id'])
                result = self.wait(lambda: manager.get(result['id']))
                self.assertEqual(result['status'], 'error')
                self.assertTrue(result['error'])
                self.assertFalse(result.get('points'))

    def test_encoder_failure_retains_identity_and_never_falls_back(self):
        def broken(model):
            raise ValueError('This checkpoint has no compatible image encoder')
        manager = EmbeddingManager(self.previews, self.root/'broken', encoder_factory=broken)
        preview = self.preview(model='fixture/incompatible')
        job = manager.start(preview['id'])
        result = self.wait(lambda: manager.get(job['id']))
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['model']['name'], 'fixture/incompatible')
        self.assertEqual(result['preview_id'], preview['id'])
        self.assertIn('no compatible image encoder', result['error'])
        self.assertFalse(result.get('points'))
        with self.assertRaises(ValueError):
            manager.expand(job['id'], 0)

    def test_non_image_missing_model_and_invalid_image_indexes_are_rejected(self):
        text_preview = self.previews.start(validate_config({'task': 'summarization'}))
        self.wait(lambda: self.previews.get(text_preview['id']))
        with self.assertRaises(ValueError):
            self.manager.start(text_preview['id'])
        no_model = self.preview(model='')
        with self.assertRaises(ValueError):
            self.manager.start(no_model['id'])
        result = self.overview(self.preview())
        for index in (-1, 2, True, '0'):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.manager.expand(result['id'], index)

    def test_superseded_expansion_stops_between_batches_and_new_selection_finishes(self):
        preview = self.preview(space={'rotation': [-10, -5, 0, 5, 10],
                                     'darkness': [.1, .2, .3, .4, .5], 'saturation': []})
        result = self.overview(preview)
        encoder = self.encoders[0]
        encoder.block = True
        stale = self.manager.expand(result['id'], 0)
        self.assertTrue(encoder.entered.wait(3), 'Expansion never began inference')
        current = self.manager.expand(result['id'], 1)
        self.assertNotEqual(current['id'], stale['id'])
        encoder.release.set()
        self.addCleanup(encoder.release.set)
        current = self.wait(lambda: self.manager.get_expansion(result['id'], current['id']))
        stale = self.wait(lambda: self.manager.get_expansion(result['id'], stale['id']))
        self.assertEqual(current['status'], 'completed', current.get('error'))
        self.assertEqual(current['input_index'], 1)
        self.assertEqual(len(current['nodes']), 25)
        self.assertEqual(stale['status'], 'cancelled')
        self.assertLess(stale['completed'], stale['total'])
        self.assertLess(len(encoder.seen), 2 + 25 + 25)
        self.assertTrue(all('/input-1/' in node['image'] for node in current['nodes']))

    def test_queued_models_do_not_allocate_or_infer_concurrently(self):
        entered, release = threading.Event(), threading.Event()
        state = {'busy': False, 'overlap': False, 'loads': 0}

        class SerialEncoder(PixelEncoder):
            def __init__(self, model):
                super().__init__(model)
                state['overlap'] |= state['busy']
                state['loads'] += 1
            def encode(self, images):
                state['overlap'] |= state['busy']
                state['busy'] = True
                if self.model['name'] == 'fixture/first':
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('Test did not release first model')
                values = super().encode(images)
                state['busy'] = False
                return values

        manager = EmbeddingManager(self.previews, self.root/'serial', encoder_factory=SerialEncoder)
        first = manager.start(self.preview(model='fixture/first')['id'])
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(3))
        second = manager.start(self.preview(model='fixture/second')['id'])
        self.assertEqual(manager.get(second['id'])['status'], 'queued')
        release.set()
        for job in (first, second):
            result = self.wait(lambda: manager.get(job['id']))
            self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertFalse(state['overlap'])
        self.assertEqual(state['loads'], 2)

    def test_invalid_action_vectors_do_not_publish_partial_expansions(self):
        result = self.overview(self.preview())
        self.encoders[0].encode = lambda images: np.zeros((len(images), 4))
        expansion = self.manager.expand(result['id'], 0)
        expansion = self.wait(lambda: self.manager.get_expansion(result['id'], expansion['id']))
        self.assertEqual(expansion['status'], 'error')
        self.assertFalse(expansion['nodes'])
        self.assertEqual(self.manager.get(result['id'])['status'], 'completed')

    def test_memory_limit_fails_clearly_without_sampling_the_dataset(self):
        self.manager.MAX_PCA_BYTES = 1
        job = self.manager.start(self.preview()['id'])
        result = self.wait(lambda: self.manager.get(job['id']))
        self.assertEqual(result['status'], 'error')
        self.assertIn('PCA', result['error'])
        self.assertFalse(result['points'])
        self.assertFalse(list(self.manager.directory.glob('*.npy')))


class TensorFixture:
    """NumPy-backed tensor protocol for testing the model extraction boundary."""
    def __init__(self, data):
        self.data = np.asarray(data)
        self.ndim = self.data.ndim
    def mean(self, dim):
        return TensorFixture(self.data.mean(axis=dim))
    def to(self, *args, **kwargs):
        return self
    def detach(self):
        return self
    def numpy(self):
        return self.data


class ModelValidationTests(unittest.TestCase):
    def test_local_file_and_incomplete_directory_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root/'model.pth'
            checkpoint.write_bytes(b'not an executable pickle')
            for path in (checkpoint, root):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'directory|config|processor'):
                    HuggingFaceEncoder({'source': 'local', 'name': str(path)})
            (root/'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'processor'):
                HuggingFaceEncoder({'source': 'local', 'name': str(root)})

    def test_loader_disables_custom_code_and_unsafe_weights_and_pooling_keeps_features(self):
        state = {'inference': False, 'evaluation': False, 'cpu': False, 'model_options': None,
                 'processor_options': None, 'missing_keys': []}

        @contextmanager
        def inference_mode():
            state['inference'] = True
            try:
                yield
            finally:
                state['inference'] = False

        class Model:
            config = SimpleNamespace(_commit_hash='fixed-model-revision')
            def to(self, device):
                state['cpu'] = device == 'cpu'
                return self
            def eval(self):
                state['evaluation'] = True
                return self
            def __call__(self, **inputs):
                if not state['inference'] or not state['evaluation'] or not state['cpu']:
                    raise AssertionError('The model ran outside CPU evaluation/inference mode')
                return state['output']

        def load_model(name, **options):
            state['model_options'] = options
            if options.get('output_loading_info'):
                return Model(), {'missing_keys': state['missing_keys'], 'mismatched_keys': [], 'unexpected_keys': ['classifier.weight']}
            return Model()

        def load_processor(name, **options):
            state['processor_options'] = options
            return lambda images, return_tensors: {'pixel_values': TensorFixture(np.ones((len(images), 3, 2, 2)))}

        torch = SimpleNamespace(__version__='2.6.0', set_num_threads=lambda count: None,
                                inference_mode=inference_mode, is_tensor=lambda value: isinstance(value, TensorFixture),
                                float64='float64')
        transformers = SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load_model),
                                       AutoImageProcessor=SimpleNamespace(from_pretrained=load_processor))
        cases = [
            (dict(pooler_output=TensorFixture([[1, 2], [3, 4]])), [[1, 2], [3, 4]], 'pooler_output'),
            (dict(pooler_output=TensorFixture(np.arange(8).reshape(2, 2, 2, 1))), [[.5, 2.5], [4.5, 6.5]], 'spatial'),
            (dict(last_hidden_state=TensorFixture(np.arange(12).reshape(2, 2, 3))), [[1.5, 2.5, 3.5], [7.5, 8.5, 9.5]], 'token'),
            (dict(last_hidden_state=TensorFixture(np.arange(16).reshape(2, 2, 2, 2))), [[1.5, 5.5], [9.5, 13.5]], 'spatial'),
        ]
        with patch.dict('sys.modules', {'torch': torch, 'transformers': transformers}):
            for output, expected, method in cases:
                with self.subTest(method=method, output=output):
                    encoder = HuggingFaceEncoder({'source': 'huggingface', 'name': 'fixture/vision'})
                    state['output'] = SimpleNamespace(**output)
                    np.testing.assert_allclose(encoder.encode([Image.new('RGB', (2, 2))] * 2), expected)
                    self.assertIn(method, encoder.feature_method)
                    self.assertIs(state['processor_options']['trust_remote_code'], False)
                    self.assertIs(state['processor_options']['use_fast'], False)
                    self.assertIs(state['model_options']['trust_remote_code'], False)
                    self.assertIs(state['model_options']['weights_only'], True)
            encoder = HuggingFaceEncoder({'source': 'huggingface', 'name': 'fixture/vision'})
            state['output'] = SimpleNamespace(logits=TensorFixture([[1, 2]]))
            with self.assertRaisesRegex(ValueError, 'vision AutoModel'):
                encoder.encode([Image.new('RGB', (2, 2))])
            # Classifier checkpoints can omit a learned base-model pooler. Its newly
            # initialized random values must never become the selected model's features.
            state['missing_keys'] = ['vit.pooler.dense.weight', 'vit.pooler.dense.bias']
            state['output'] = SimpleNamespace(pooler_output=TensorFixture([[900, 800]]),
                                             last_hidden_state=TensorFixture([[[1, 2], [3, 4]]]))
            encoder = HuggingFaceEncoder({'source': 'huggingface', 'name': 'fixture/vision'})
            np.testing.assert_allclose(encoder.encode([Image.new('RGB', (2, 2))]), [[2, 3]])
            self.assertIn('token', encoder.feature_method)
            state['missing_keys'] = ['vit.embeddings.position_embeddings']
            with self.assertRaisesRegex(ValueError, 'missing|pretrained'):
                HuggingFaceEncoder({'source': 'huggingface', 'name': 'fixture/vision'})
            torch.__version__ = '2.5.1'
            with self.assertRaisesRegex(ValueError, '2.6'):
                HuggingFaceEncoder({'source': 'huggingface', 'name': 'fixture/vision'})


if __name__ == '__main__':
    unittest.main()
