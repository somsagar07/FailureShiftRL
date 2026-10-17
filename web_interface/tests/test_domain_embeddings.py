"""Full non-image preview → embedding → expansion flow without remote downloads."""
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from embeddings import EmbeddingManager, _normalize, _selection
from server import PreviewManager, validate_config


class TextFeatures:
    """Deterministic injected model boundary; never used by the application."""
    feature_method = 'test text statistics'

    def __init__(self, model):
        self.model = model
        self.seen = []

    def encode(self, texts):
        self.seen.extend(texts)
        return np.array([[1, len(text), sum(c.isupper() for c in text),
                          sum(c in '.,!?' for c in text), sum(ord(c) for c in text) % 31]
                         for text in texts], dtype=float)


class DomainEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.previews = PreviewManager(self.root/'previews')
        self.encoders = []

        def factory(model):
            encoder = TextFeatures(model)
            self.encoders.append(encoder)
            return encoder

        self.manager = EmbeddingManager(self.previews, self.root/'embeddings', encoder_factory=factory)

    def wait(self, getter):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = getter()
            if result['status'] not in ('queued', 'running'):
                self.assertEqual(result['status'], 'completed', result.get('error'))
                return result
            time.sleep(.005)
        self.fail('Background job did not finish')

    def preview(self, task='summarization', **extra):
        config = {'task': task, 'model': {'source': 'huggingface', 'name': 'test/model'}}
        if task == 'summarization':
            config.update(text='HELLO, World!', action_space={'actions': ['to_lowercase', 'remove_punctuation']})
        config.update(extra)
        job = self.previews.start(validate_config(config))
        return self.wait(lambda: self.previews.get(job['id']))

    def analyze(self, preview):
        job = self.manager.start(preview['id'])
        return self.wait(lambda: self.manager.get(job['id']))

    def expand(self, job):
        expansion = self.manager.expand(job['id'], 0)
        return self.wait(lambda: self.manager.get_expansion(job['id'], expansion['id']))

    def test_summary_fits_original_and_variants_in_one_space_with_exact_cosine(self):
        preview = self.preview()
        job = self.analyze(preview)
        self.assertEqual(job['task'], 'summarization')
        self.assertEqual(job['model']['task'], 'summarization')
        self.assertEqual(job['projection']['fit_count'], 3)
        self.assertIn('variant', job['projection']['method'])
        self.assertGreater(job['projection']['rank'], 0)
        original = job['points'][0]
        self.assertEqual(original['text'], preview['original']['text'])
        self.assertNotIn('image', original)
        self.assertEqual(self.encoders[0].seen, [preview['original']['text']]+[n['text'] for n in preview['nodes']])
        expansion = self.expand(job)
        vectors = _normalize(TextFeatures({}).encode(self.encoders[0].seen), 3)
        for index, node in enumerate(expansion['nodes']):
            self.assertEqual(node['text'], preview['nodes'][index]['text'])
            self.assertEqual(node['action'], preview['nodes'][index]['action'])
            self.assertAlmostEqual(node['similarity'], float(vectors[0] @ vectors[index+1]))
            # All three vectors fit exactly into <=2 PCA axes: distance preserves cosine here.
            distance = np.linalg.norm(np.array(node['position']) - original['position'])
            self.assertAlmostEqual(1-distance**2/2, node['similarity'])
        self.assertEqual(len(self.encoders[0].seen), 3, 'Expansion must reuse the analyzed variants')
        self.assertEqual(self.expand(job)['nodes'], expansion['nodes'])
        self.assertEqual(self.manager.start(preview['id'])['id'], job['id'])

    def test_generation_uses_a_concrete_stable_baseline_and_all_prompt_variants(self):
        preview = self.preview('generation')
        baseline = 'A unique scientist working on a project in a corporate office.'
        self.assertEqual(preview['original']['text'], baseline)
        job = self.analyze(preview)
        self.assertEqual(job['projection']['fit_count'], 28)
        expansion = self.expand(job)
        self.assertEqual(len(expansion['nodes']), 27)
        self.assertEqual(expansion['nodes'][0]['text'], baseline)
        self.assertAlmostEqual(expansion['nodes'][0]['similarity'], 1)
        np.testing.assert_allclose(expansion['nodes'][0]['position'], job['points'][0]['position'])
        self.assertTrue(all('{' not in text for text in self.encoders[0].seen))
        chosen = self.preview('generation', action_space={
            'description': ['cool'], 'profession': ['artist'], 'setting': ['classroom']})
        self.assertEqual(chosen['original']['text'], baseline)
        chosen_job = self.analyze(chosen)
        self.assertNotEqual(chosen_job['id'], job['id'])
        self.assertEqual(chosen_job['projection']['fit_count'], 2)
        self.assertEqual(len(self.expand(chosen_job)['nodes']), 1)

    def test_discovery_text_node_reuses_exact_analyzed_variant(self):
        job = self.analyze(self.preview())
        expected = self.expand(job)['nodes'][0]
        count = len(self.encoders[0].seen)
        self.assertEqual(self.manager.discovery_node(job['id'], 0, expected['id']), expected)
        self.assertEqual(len(self.encoders[0].seen), count)

    def test_task_input_and_action_changes_do_not_share_cached_features(self):
        first = self.analyze(self.preview())
        second = self.analyze(self.preview(text='A different input.'))
        third = self.analyze(self.preview('generation'))
        self.assertNotEqual(first['points'][0]['text'], second['points'][0]['text'])
        self.assertEqual(len(self.encoders), 2, 'Tasks sharing a repository ID need distinct encoders')
        self.assertEqual([encoder.model['task'] for encoder in self.encoders], ['summarization', 'generation'])
        self.assertNotEqual(first['id'], third['id'])
        for job in (first, second, third):
            for invalid in (-1, 1, True, '0'):
                with self.assertRaises(ValueError):
                    self.manager.expand(job['id'], invalid)

    def test_unchanged_text_actions_keep_zero_rank_and_do_not_invent_spread(self):
        preview = self.preview(text='lowercase', action_space={'actions': ['to_lowercase']})
        job = self.analyze(preview)
        self.assertEqual(job['projection']['rank'], 0)
        self.assertEqual(job['points'][0]['position'], [0., 0., 0.])
        node = self.expand(job)['nodes'][0]
        self.assertTrue(node['unchanged'])
        self.assertEqual(node['position'], [0., 0., 0.])
        self.assertAlmostEqual(node['similarity'], 1)

    def test_failed_analysis_can_be_retried_without_regenerating_preview(self):
        preview = self.preview()
        original_factory = self.manager.encoder_factory
        self.manager.encoder_factory = lambda model: (_ for _ in ()).throw(ValueError('temporary loading problem'))
        failed = self.manager.start(preview['id'])
        deadline = time.monotonic() + 5
        while self.manager.get(failed['id'])['status'] != 'error' and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(self.manager.get(failed['id'])['status'], 'error')
        self.manager.encoder_factory = original_factory
        retry = self.analyze(preview)
        self.assertNotEqual(retry['id'], failed['id'])

    def test_local_model_validation_depends_on_workspace(self):
        summary = self.root/'summary'
        summary.mkdir()
        (summary/'config.json').write_text('{}')
        pipeline = self.root/'diffusion'
        pipeline.mkdir()
        (pipeline/'model_index.json').write_text('{}')
        for task, path in (('summarization', summary), ('generation', pipeline)):
            result = _selection({'source': 'local', 'name': str(path)}, task)
            self.assertEqual(result['task'], task)
        with self.assertRaisesRegex(ValueError, 'preprocessor'):
            _selection({'source': 'local', 'name': str(summary)}, 'classification')


if __name__ == '__main__':
    unittest.main()
