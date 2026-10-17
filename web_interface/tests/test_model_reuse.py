"""Loaded model reuse without stale checkpoints or per-run settings."""
import shutil
import os
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors.torch import load_file, save_file

from embeddings import EmbeddingManager, _model_key
import test_discovery_models as evaluator_fixtures
import test_generation_models as generation_fixtures


class ModelReuseTests(unittest.TestCase):
    setUpClass = classmethod(evaluator_fixtures.EvaluatorTests.setUpClass.__func__)
    model = evaluator_fixtures.EvaluatorTests.model

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.manager = EmbeddingManager(None, Path(self.scratch.name)/'embeddings')

    def evaluation(self, task='classification', model=None, **config):
        evaluator = self.manager.evaluation_model(
            model or self.model(task), {'task': task, **config}, Path(self.scratch.name)/'run')
        self.addCleanup(evaluator.close)
        return evaluator

    def test_summary_reuses_analyze_weights_with_fresh_reference_and_decoding_settings(self):
        selection = self.model('summarization')
        encoder = self.manager._load_encoder(selection, _model_key(selection))
        before = encoder.encode(['a cat'])
        with patch('transformers.AutoModelForSeq2SeqLM.from_pretrained',
                   side_effect=AssertionError('Analyze weights were loaded again')):
            first = self.evaluation('summarization', reference_summary='a cat', max_new_tokens=4)
            self.assertIs(first.model, encoder.model)
            prediction = first.predict('a cat')
            first.close()
            second = self.evaluation('summarization', reference_summary='a cat', max_new_tokens=4)
            self.assertEqual(second.predict('a cat'), prediction)
            third = self.evaluation('summarization', reference_summary='a dog', max_new_tokens=2)
            self.assertEqual(third.reference, 'a dog')
            self.assertEqual(third.max_new_tokens, 2)
            self.assertEqual(third.measure_summary('a dog')['quality'], 1)
            self.assertIs(third.model, encoder.model)
        np.testing.assert_array_equal(before, encoder.encode(['a cat']))

    def test_classifier_reuses_weights_and_preserves_exact_features_and_probabilities(self):
        selection = self.model()
        encoder = self.manager._load_encoder(selection, _model_key(selection))
        image = Image.new('RGB', (20, 20), 'red')
        before = encoder.encode([image])
        first = self.evaluation()
        probabilities = first.predict(image)
        self.assertIs(encoder.model, first.model.base_model)
        first.close()
        with patch('transformers.AutoModelForImageClassification.from_pretrained',
                   side_effect=AssertionError('Classifier weights were loaded again')):
            second = self.evaluation(failure_threshold=.4)
            self.assertEqual(second.predict(image), probabilities)
            self.assertEqual(second.threshold, .4)
        np.testing.assert_array_equal(before, encoder.encode([image]))

    def test_changed_checkpoint_replaces_cached_weights_and_release_drops_the_model(self):
        directory = Path(shutil.copytree(self.vision, Path(self.scratch.name)/'checkpoint'))
        selection = self.model(path=directory)
        image = Image.new('RGB', (20, 20), 'red')
        first = self.evaluation(model=selection)
        before = first.predict(image)
        stamp = (directory/'model.safetensors').stat()
        weights = load_file(directory/'model.safetensors')
        weights['classifier.weight'].zero_()
        weights['classifier.bias'] = torch.tensor([-10., -10., 10.])
        save_file(weights, directory/'model.safetensors', metadata={'format': 'pt'})
        os.utime(directory/'model.safetensors', ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        second = self.evaluation(model=selection)
        self.assertIsNot(first.model, second.model)
        self.assertNotEqual(before, second.predict(image))
        self.assertEqual(second.predict(image)['class_id'], 2)
        self.manager.release_encoder()
        third = self.evaluation(model=selection)
        self.assertIsNot(second.model, third.model)
        self.assertEqual(second.predict(image), third.predict(image))

    def test_auto_device_reuses_cpu_weights_when_cuda_is_unavailable(self):
        first = self.evaluation(device='cpu')
        with patch.object(torch.cuda, 'is_available', return_value=False), \
                patch('transformers.AutoModelForImageClassification.from_pretrained',
                      side_effect=AssertionError('auto resolved to CPU but reloaded its weights')):
            second = self.evaluation(device='auto')
        self.assertIs(first.model, second.model)
        self.assertEqual(second.device, 'cpu')

    def test_auto_cpu_summary_reuses_the_analysis_model(self):
        selection = self.model('summarization')
        encoder = self.manager._load_encoder(selection, _model_key(selection))
        with patch.object(torch.cuda, 'is_available', return_value=False), \
                patch('transformers.AutoModelForSeq2SeqLM.from_pretrained',
                      side_effect=AssertionError('auto resolved to CPU but reloaded Analyze weights')):
            evaluator = self.evaluation('summarization', device='auto', reference_summary='a cat')
        self.assertIs(evaluator.model, encoder.model)

    def test_remote_revisions_are_isolated_and_switching_evicts_old_weights(self):
        load_config = transformers.AutoConfig.from_pretrained
        load_model = transformers.AutoModelForImageClassification.from_pretrained
        load_processor = transformers.AutoImageProcessor.from_pretrained

        def config_at_revision(name, **options):
            config = load_config(self.vision, local_files_only=True)
            config._commit_hash = options['revision']
            return config

        def local_model(name, **options):
            options.pop('revision', None)
            options['local_files_only'] = True
            return load_model(self.vision, **options)

        def local_processor(name, **options):
            options.pop('revision', None)
            options['local_files_only'] = True
            return load_processor(self.vision, **options)

        selection = {'source': 'huggingface', 'name': 'fixture/vision', 'task': 'classification'}
        with patch.object(transformers.AutoConfig, 'from_pretrained', side_effect=config_at_revision), \
                patch.object(transformers.AutoModelForImageClassification, 'from_pretrained', side_effect=local_model), \
                patch.object(transformers.AutoImageProcessor, 'from_pretrained', side_effect=local_processor):
            first = self.evaluation(model={**selection, 'revision': 'a'*40})
            old_weights = weakref.ref(first.model)
            first.close()
            second = self.evaluation(model={**selection, 'revision': 'b'*40})
            self.assertEqual(second.revision, 'b'*40)
            self.assertIsNone(old_weights(), 'The previous model must be released when its revision changes')
            second.close()
            third = self.evaluation(model={**selection, 'revision': 'a'*40})
            self.assertEqual(third.revision, 'a'*40)

    def test_resolved_revision_stays_pinned_after_explicit_weight_release(self):
        load_config = transformers.AutoConfig.from_pretrained
        load_model = transformers.AutoModelForImageClassification.from_pretrained
        load_processor = transformers.AutoImageProcessor.from_pretrained
        commits = iter(['a'*40, 'b'*40])

        def config_at_revision(name, **options):
            config = load_config(self.vision, local_files_only=True)
            revision = options.get('revision')
            config._commit_hash = next(commits) if revision == 'main' else revision
            return config

        def local_model(name, **options):
            options.pop('revision', None)
            options['local_files_only'] = True
            return load_model(self.vision, **options)

        def local_processor(name, **options):
            options.pop('revision', None)
            options['local_files_only'] = True
            return load_processor(self.vision, **options)

        selection = {'source': 'huggingface', 'name': 'fixture/vision', 'task': 'classification', 'revision': 'main'}
        with patch.object(transformers.AutoConfig, 'from_pretrained', side_effect=config_at_revision), \
                patch.object(transformers.AutoModelForImageClassification, 'from_pretrained', side_effect=local_model), \
                patch.object(transformers.AutoImageProcessor, 'from_pretrained', side_effect=local_processor):
            first = self.evaluation(model=selection)
            self.assertEqual(first.revision, 'a'*40)
            first.close()
            self.manager.release_encoder()
            second = self.evaluation(model=selection)
            self.assertEqual(second.revision, 'a'*40)


class GenerationReuseTests(unittest.TestCase):
    setUpClass = classmethod(generation_fixtures.GenerationTests.setUpClass.__func__)
    model = generation_fixtures.GenerationTests.model

    def test_pipeline_reuses_prompt_encoder_and_keeps_seeded_images_and_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = EmbeddingManager(None, root/'embeddings')
            selection = self.model()
            encoder = manager._load_encoder(selection, _model_key(selection))
            features = encoder.encode(['a cat'])
            config = {'task': 'generation', 'generation_steps': 2, 'generation_size': 32,
                      '_clip_model': {'source': 'local', 'name': str(self.clip)}}
            with patch('transformers.CLIPTextModel.from_pretrained',
                       side_effect=AssertionError('Prompt encoder weights were loaded again')):
                first = manager.evaluation_model(selection, config, root/'first')
            self.assertIs(first.pipeline.text_encoder, encoder.model)
            before = first.predict('a cat', seed=3)
            original = np.array(Image.open(root/'first'/before['image']))
            first.close()
            with patch('discovery_models.load_diffusion',
                       side_effect=AssertionError('Diffusion pipeline was loaded again')):
                second = manager.evaluation_model(selection, config, root/'second')
                after = second.predict('a cat', seed=3)
            self.assertEqual(before, after)
            np.testing.assert_array_equal(original, np.array(Image.open(root/'second'/after['image'])))
            np.testing.assert_array_equal(features, encoder.encode(['a cat']))
            second.close()
            manager.release_encoder()


if __name__ == '__main__':
    unittest.main()
