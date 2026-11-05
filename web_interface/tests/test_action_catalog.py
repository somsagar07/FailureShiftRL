"""Optional catalogs preserve saved actions and execute bounded real transforms."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core


class CatalogTests(unittest.TestCase):
    def available(self, task):
        self.assertTrue(callable(getattr(core, 'available_action_space', None)),
                        'A separate optional catalog must be available without changing defaults')
        return core.available_action_space(task)

    def test_defaults_keep_original_counts_ids_and_dictionary_shapes(self):
        self.assertEqual(len(core.actions_for('classification')), 125)
        self.assertEqual(core.actions_for('classification')[0],
                         {'id': 0, 'rotation': -10, 'darkness': .1, 'saturation': .5})
        text = core.actions_for('summarization')
        self.assertEqual(len(text), 14)
        self.assertEqual(text[-1], {'id': 13, 'name': 'repeat_random_word'})
        generated = core.actions_for('generation')
        self.assertEqual(len(generated), 125)
        self.assertEqual(generated[0], {'id': 43, 'style': 'photograph', 'lighting': 'even', 'detail': 'minimal'})
        self.assertEqual(generated[-1], {'id': 215, 'style': '3D render', 'lighting': 'low-key', 'detail': 'intricate'})

    def test_full_catalog_is_opt_in_and_returns_independent_values(self):
        for task, minimum in [('classification', 15), ('generation', 8)]:
            catalog = self.available(task)
            self.assertGreaterEqual(len(catalog), minimum)
            for key, values in core.default_action_space(task).items():
                self.assertEqual(catalog[key], values)
            catalog[next(iter(catalog))].clear()
            self.assertTrue(self.available(task)[next(iter(catalog))])
        self.assertGreaterEqual(len(self.available('summarization')['actions']), 26)
        self.assertEqual(len(core.default_action_space('summarization')['actions']), 14)

    def test_optional_text_actions_keep_appended_ids_when_filtered(self):
        catalog = self.available('summarization')
        actions = core.actions_for('summarization', catalog)
        self.assertGreaterEqual(len(actions), 26)
        self.assertEqual(actions[14], {'id': 14, 'name': 'to_uppercase'})
        self.assertEqual(core.actions_for('summarization', {'actions': ['to_uppercase', 'to_lowercase']}),
                         [{'id': 14, 'name': 'to_uppercase'}, {'id': 2, 'name': 'to_lowercase'}])

    def test_optional_image_dimensions_do_not_enter_old_actions(self):
        space = {'rotation': [0], 'darkness': [], 'saturation': [], 'gaussian_blur': [1, 2]}
        actions = core.actions_for('classification', space)
        self.assertEqual(actions, [dict(id=0, rotation=0, darkness=0, saturation=1, gaussian_blur=1),
                                   dict(id=1, rotation=0, darkness=0, saturation=1, gaussian_blur=2)])
        space['gaussian_blur'] = []
        self.assertEqual(core.actions_for('classification', space),
                         [dict(id=0, rotation=0, darkness=0, saturation=1)])

    def test_optional_generation_ids_survive_filtering_and_empty_groups(self):
        base = {'style': ['watercolor'], 'lighting': [], 'detail': []}
        original = core.actions_for('generation', base)[0]
        self.assertEqual(original, {'id': 72, 'style': 'watercolor', 'lighting': None, 'detail': None})
        expanded = core.actions_for('generation', {**base, 'framing': ['wide', 'close-up']})
        self.assertEqual(len({original['id'], *(action['id'] for action in expanded)}), 3)
        self.assertTrue(all(action['id'] > 215 for action in expanded))
        self.assertEqual(core.actions_for('generation', {**base, 'framing': ['close-up'], 'focus': []}),
                         [expanded[1]])
        self.assertEqual(core.actions_for('generation', {**base, 'framing': []}), [original])

    def test_all_optional_generation_combinations_have_distinct_browser_safe_ids(self):
        space = self.available('generation')
        for key in ('style', 'lighting', 'detail'):
            space[key] = []
        actions = core.actions_for('generation', space)
        self.assertEqual(len(actions), 3072)
        self.assertEqual(len({action['id'] for action in actions}), 3072)
        self.assertLess(max(action['id'] for action in actions), 2**53)
        chosen = {key: [values[-1]] if values else [] for key, values in space.items()}
        self.assertEqual(core.actions_for('generation', chosen), [actions[-1]])

    def test_generation_groups_preserve_scene_and_enter_judge_rubric(self):
        from generation_failure import _presentation_settings
        space = {'style': [], 'lighting': [], 'detail': [], 'viewpoint': ['overhead'], 'color_palette': ['muted']}
        action = core.actions_for('generation', space)[0]
        scene = '  Two red cups left of a green bowl.\nA sign says {home}.  '
        self.assertEqual(core.apply_prompt(scene, action),
                         'Viewpoint: overhead. Color palette: muted. Scene: ' + scene)
        self.assertEqual(_presentation_settings(action), {'viewpoint': 'overhead', 'color_palette': 'muted'})

    def test_bounded_large_spaces_are_allowed_and_oversized_spaces_never_enumerate(self):
        safe = {'rotation': list(range(20)), 'darkness': [i/20 for i in range(20)],
                'saturation': [i/10 for i in range(10)]}
        self.assertEqual(len(core.actions_for('classification', safe)), 4000)
        oversized = {**safe, 'gaussian_blur': [0, 1]}
        with patch('core.itertools.product', side_effect=AssertionError('Enumerated an oversized space')):
            with self.assertRaisesRegex(ValueError, r'8000 combinations.*5000'):
                core.actions_for('classification', oversized)

    def test_numeric_bounds_integer_parameters_and_unknown_names_are_rejected(self):
        base = {'rotation': [0], 'darkness': [], 'saturation': []}
        for key, value in [('gaussian_noise', float('inf')), ('brightness', -1), ('brightness', 10**1000),
                           ('posterize_bits', 2.5), ('jpeg_quality', True), ('horizontal_flip', .5),
                           ('solarize_threshold', 257), ('invented_transform', 1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                core.actions_for('classification', {**base, key: [value]})
        with self.assertRaises(ValueError):
            core.actions_for('generation', {'style': [], 'lighting': [], 'detail': [], 'framing': ['invent a cat']})

    def test_metadata_marks_risks_without_making_identity_actions_risky(self):
        self.assertTrue(callable(getattr(core, 'action_metadata', None)))
        self.assertTrue(callable(getattr(core, 'action_semantic_risks', None)))
        metadata = core.action_metadata('classification')
        self.assertEqual(metadata['gaussian_blur']['identity'], 0)
        self.assertEqual(metadata['gaussian_blur']['semantic_risk'], 'visibility')
        self.assertEqual(core.action_semantic_risks('classification', dict(rotation=0, darkness=0, saturation=1)), [])
        risks = core.action_semantic_risks('classification', dict(rotation=0, gaussian_blur=2))
        self.assertEqual([row['name'] for row in risks], ['gaussian_blur'])
        self.assertTrue(risks[0]['reason'])
        self.assertEqual(core.action_semantic_risks('summarization', {'name': 'negate_sentence'})[0]['semantic_risk'], 'meaning')
        self.assertTrue(all(not core.action_semantic_risks('classification', action)
                            for action in core.actions_for('classification')))

    def test_http_catalog_keeps_defaults_separate_and_accepts_optional_configuration(self):
        from server import create_server, validate_config
        from test_http import MemoryServer, MemoryConnection
        with tempfile.TemporaryDirectory() as directory, patch('server.ThreadingHTTPServer', MemoryServer):
            server = create_server(8000, directory, model_preparer=lambda model: (model, {}))
            connection = MemoryConnection(b'GET /api/catalog HTTP/1.1\r\nHost: 127.0.0.1:8000\r\n\r\n')
            with patch('server.BaseHTTPRequestHandler.log_message'):
                server.handler(connection, ('127.0.0.1', 1), server)
            payload = json.loads(connection.output.split(b'\r\n\r\n', 1)[1])
        self.assertIn('action_catalog', payload)
        self.assertIn('action_metadata', payload)
        self.assertEqual(payload['limits']['max_actions'], 5000)
        self.assertEqual(len(payload['tasks']['classification']), 3)
        self.assertEqual(len(payload['tasks']['summarization']['actions']), 14)
        self.assertEqual(len(payload['tasks']['generation']), 3)
        for task, addition in [('classification', {'gaussian_blur': [1]}),
                               ('generation', {'viewpoint': ['overhead']})]:
            space = {key: [] for key in payload['tasks'][task]}
            space.update(addition)
            self.assertEqual(validate_config({'task': task, 'dataset_id': 'a'*32,
                                              'action_space': space})['action_space'], space)


class ImageTransformTests(unittest.TestCase):
    def fixture(self, mode='RGB'):
        array = np.random.default_rng(7).integers(0, 256, (33, 47, 3), dtype=np.uint8)
        image = Image.fromarray(array).convert(mode)
        if mode in ('RGBA', 'LA'):
            image.putalpha(Image.fromarray(np.arange(33*47, dtype=np.uint8).reshape(33, 47)))
        return image

    def test_original_pipeline_pixels_remain_identical(self):
        image = self.fixture()
        action = {'id': 0, 'rotation': -10, 'darkness': .1, 'saturation': .5}
        expected = ImageEnhance.Brightness(ImageEnhance.Color(image.rotate(-10)).enhance(.5)).enhance(.9)
        self.assertEqual(core.apply_image(image, action).tobytes(), expected.tobytes())

    def test_every_optional_image_value_is_deterministic_bounded_and_mode_preserving(self):
        self.assertTrue(callable(getattr(core, 'available_action_space', None)))
        catalog = core.available_action_space('classification')
        for mode in ('RGB', 'RGBA', 'L', 'LA'):
            image = self.fixture(mode)
            before = image.tobytes()
            for name, values in catalog.items():
                if name in ('rotation', 'darkness', 'saturation'):
                    continue
                for value in values:
                    with self.subTest(mode=mode, name=name, value=value):
                        action = dict(id=25, rotation=0, darkness=0, saturation=1, **{name: value})
                        result = core.apply_image(image, action)
                        self.assertEqual((result.size, result.mode), (image.size, mode))
                        self.assertEqual(result.tobytes(), core.apply_image(image, action).tobytes())
                        self.assertEqual(image.tobytes(), before)
                        pixels = np.asarray(result)
                        self.assertTrue(np.isfinite(pixels).all())
                        self.assertGreaterEqual(pixels.min(), 0)
                        self.assertLessEqual(pixels.max(), 255)

    def test_optional_identity_values_preserve_exact_source_pixels(self):
        self.assertTrue(callable(getattr(core, 'action_metadata', None)))
        action = {'id': 1, **{name: row['identity'] for name, row in core.action_metadata('classification').items()}}
        for mode in ('RGB', 'RGBA', 'L', 'LA', 'P', '1'):
            image = self.fixture(mode)
            self.assertEqual(core.apply_image(image, action).tobytes(), image.tobytes())

    def test_transform_order_is_independent_of_dictionary_order(self):
        image = self.fixture()
        first = dict(id=4, rotation=5, darkness=.1, saturation=.7, gaussian_noise=.1, contrast=1.5, translate_x=.1)
        second = dict(reversed(list(first.items())))
        self.assertEqual(core.apply_image(image, first).tobytes(), core.apply_image(image, second).tobytes())

    def test_noise_uses_local_randomness_and_ignores_catalog_position(self):
        image = self.fixture()
        action = dict(id=1, rotation=0, darkness=0, saturation=1, gaussian_noise=.1)
        state = np.random.get_state()
        first = core.apply_image(image, action)
        after = np.random.get_state()
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(state, after)))
        self.assertNotEqual(first.tobytes(), image.tobytes())
        self.assertEqual(first.tobytes(), core.apply_image(image, {**action, 'id': 123}).tobytes())

    def test_photometric_edits_preserve_transparency(self):
        for mode in ('RGBA', 'LA'):
            image = self.fixture(mode)
            for name, value in [('brightness', .5), ('contrast', .5), ('sharpness', 2),
                                ('gaussian_blur', 1), ('gaussian_noise', .1), ('jpeg_quality', 25),
                                ('hue_shift', .1), ('grayscale', 1), ('invert', 1),
                                ('posterize_bits', 3), ('solarize_threshold', 128)]:
                with self.subTest(mode=mode, name=name):
                    result = core.apply_image(image, {name: value})
                    self.assertEqual(result.getchannel('A').tobytes(), image.getchannel('A').tobytes())

    def test_optional_transforms_change_a_nonuniform_image(self):
        image = self.fixture()
        for name, value in [('brightness', .5), ('contrast', .5), ('sharpness', 2), ('gaussian_blur', 1),
                            ('jpeg_quality', 25), ('pixelation', 4), ('hue_shift', .1), ('grayscale', 1),
                            ('invert', 1), ('posterize_bits', 3), ('solarize_threshold', 128),
                            ('translate_x', .1), ('translate_y', .1), ('zoom', 1.25), ('horizontal_flip', 1)]:
            with self.subTest(name=name):
                result = core.apply_image(image, dict(rotation=0, darkness=0, saturation=1, **{name: value}))
                self.assertTrue(result.tobytes() != image.tobytes(), f'{name} left pixels unchanged')

    def test_unknown_and_invalid_direct_image_actions_are_rejected(self):
        for extra in ({'invented_transform': 1}, {'gaussian_noise': float('nan')}, {'pixelation': 0}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                core.apply_image(self.fixture(), dict(rotation=0, darkness=0, saturation=1, **extra))


class TextTransformTests(unittest.TestCase):
    def test_new_text_actions_transform_real_content(self):
        examples = [
            ('to_uppercase', 'A cat.', 'A CAT.'),
            ('to_titlecase', 'a BLUE cat.', 'A Blue Cat.'),
            ('normalize_whitespace', ' A\n\tcat.  ', 'A cat.'),
            ('expand_contractions', "We can't go. They won't stay.", 'We cannot go. They will not stay.'),
            ('contract_phrases', 'We cannot go. They will not stay.', "We can't go. They won't stay."),
            ('unicode_quotes', '"A cat" isn\'t here.', '“A cat” isn’t here.'),
            ('ascii_quotes', '“A cat” isn’t here.', '"A cat" isn\'t here.'),
            ('add_line_breaks', 'A cat. A dog!', 'A cat.\nA dog!'),
            ('add_sentence_numbers', 'A cat. A dog!', '1. A cat.\n2. A dog!'),
            ('bullet_sentences', 'A cat. A dog!', '- A cat.\n- A dog!'),
            ('reverse_sentences', 'A cat. A dog!', 'A dog! A cat.'),
            ('duplicate_sentence', 'A cat.', 'A cat. A cat.'),
            ('truncate_last_sentence', 'A cat. A dog!', 'A cat.'),
            ('replace_spaces_with_nbsp', 'A cat.', 'A\u00a0cat.'),
        ]
        for name, source, expected in examples:
            with self.subTest(name=name):
                self.assertEqual(core.apply_text(source, {'name': name}, random.Random(42)), expected)

    def test_new_text_actions_are_reproducible_and_bounded_on_short_inputs(self):
        self.assertTrue(callable(getattr(core, 'available_action_space', None)))
        for name in core.available_action_space('summarization')['actions'][14:]:
            for text in ('', ' ', 'a', 'é.', 'First sentence! Second sentence? Third sentence.'):
                with self.subTest(name=name, text=text):
                    first = core.apply_text(text, {'name': name}, random.Random(17))
                    self.assertEqual(first, core.apply_text(text, {'name': name}, random.Random(17)))
                    self.assertLessEqual(len(first), 3*len(text)+10)

    def test_unknown_text_actions_are_rejected_even_for_empty_inputs(self):
        for text in ('', 'one word'):
            with self.assertRaisesRegex(ValueError, 'Unknown text action'):
                core.apply_text(text, {'name': 'invented_transform'}, random.Random(42))

    def test_unknown_generation_action_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            core.apply_prompt('a cat', {'style': 'watercolor', 'invented_transform': 'add a dog'})


if __name__ == '__main__':
    unittest.main()
