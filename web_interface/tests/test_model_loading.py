"""Selective, immutable model preparation without network or GPU dependencies."""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec('model_loading') is not None:
    import model_loading as ml
else:
    ml = None


COMMIT = 'a' * 40


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ml, 'Analyze requires the selective model preparation helper.')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.snapshot = Path(self.temporary.name)/'models--team--checkpoint'/'snapshots'/COMMIT
        self.snapshot.mkdir(parents=True)

    def hub(self, files, *, offline=False, commit=COMMIT):
        """Replace external Hub I/O while retaining real selection and file checks."""
        hub = types.ModuleType('huggingface_hub')
        events = []

        def model_info(repo_id, revision=None):
            events.append(('info', repo_id, revision))
            if offline:
                raise ConnectionError('offline')
            return types.SimpleNamespace(sha=commit, siblings=[types.SimpleNamespace(rfilename=name) for name in files])

        def download(repo_id, filename, revision=None, **kwargs):
            events.append(('file', repo_id, filename, revision, kwargs))
            self.assertEqual(revision, COMMIT)
            path = self.snapshot/filename
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(files[filename], bytes):
                path.write_bytes(files[filename])
            else:
                path.write_text(files[filename] if isinstance(files[filename], str) else json.dumps(files[filename]))
            return str(path)

        def snapshot_download(repo_id, *, revision=None, **kwargs):
            events.append(('snapshot', repo_id, revision, kwargs))
            self.assertEqual(repo_id, 'team/checkpoint')
            self.assertLessEqual(kwargs.get('max_workers', 4), 4)
            if not kwargs.get('local_files_only'):
                self.assertEqual(revision, COMMIT)
                for name in kwargs['allow_patterns']:
                    self.assertIn(name, files)
                    download(repo_id, name, revision)
            return str(self.snapshot)

        hub.HfApi = lambda: types.SimpleNamespace(model_info=model_info)
        hub.hf_hub_download = download
        hub.snapshot_download = snapshot_download
        replacement = patch.dict(sys.modules, {'huggingface_hub': hub})
        replacement.start()
        self.addCleanup(replacement.stop)
        self.hub_replacement = replacement
        return events

    def selection(self, task='summarization'):
        return {'source': 'huggingface', 'name': 'team/checkpoint', 'task': task, 'revision': 'main', 'device': 'cpu'}

    def test_parallel_shard_defaults_respect_explicit_operator_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            ml.configure_parallel_loading()
            self.assertEqual(os.environ['HF_ENABLE_PARALLEL_LOADING'], 'true')
            self.assertEqual(os.environ['HF_PARALLEL_LOADING_WORKERS'], '4')
        with patch.dict(os.environ, {'HF_ENABLE_PARALLEL_LOADING': 'false', 'HF_PARALLEL_LOADING_WORKERS': '1'}, clear=True):
            ml.configure_parallel_loading()
            self.assertEqual(os.environ['HF_ENABLE_PARALLEL_LOADING'], 'false')
            self.assertEqual(os.environ['HF_PARALLEL_LOADING_WORKERS'], '1')

    def test_local_preparation_keeps_selection_and_reports_reuse(self):
        (self.snapshot/'config.json').write_text('{}')
        selection = {'source': 'local', 'name': str(self.snapshot), 'task': 'summarization', 'device': 'cpu', 'revision': COMMIT}
        with patch.object(ml.time, 'perf_counter', side_effect=[10., 18.]):
            prepared, metadata = ml.prepare_analysis_model(selection)
        self.assertEqual(prepared, selection)
        self.assertEqual(metadata, {'revision': COMMIT, 'preparation_seconds': 8., 'files_reused': True})

    def test_hub_preparation_pins_revision_and_downloads_only_one_checkpoint_format(self):
        files = {'config.json': {}, 'tokenizer_config.json': {}, 'tokenizer.json': {}, 'spiece.model': '',
                 'model.safetensors': '', 'pytorch_model.bin': '', 'tf_model.h5': '', 'onnx/model.onnx': '',
                 'README.md': '', 'optimizer.pt': '', 'model.fp16.safetensors': ''}
        events = self.hub(files)
        prepared, metadata = ml.prepare_analysis_model(self.selection())
        self.assertEqual(prepared, {**self.selection(), 'source': 'local', 'name': str(self.snapshot), 'revision': COMMIT})
        self.assertEqual(metadata['revision'], COMMIT)
        self.assertGreaterEqual(metadata['preparation_seconds'], 0.)
        selected = [item[3]['allow_patterns'] for item in events if item[0] == 'snapshot']
        self.assertEqual(len(selected), 1)
        self.assertEqual(set(selected[0]), {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'spiece.model', 'model.safetensors'})

    def test_safetensors_weight_map_selects_only_referenced_shards(self):
        files = {'config.json': {}, 'tokenizer.json': {},
                 'model.safetensors.index.json': {'weight_map': {'one': 'model-00001-of-00002.safetensors', 'two': 'model-00002-of-00002.safetensors'}},
                 'model-00001-of-00002.safetensors': '', 'model-00002-of-00002.safetensors': '',
                 'unreferenced.safetensors': '', 'pytorch_model.bin': '', 'pytorch_model.bin.index.json': {}}
        events = self.hub(files)
        ml.prepare_analysis_model(self.selection())
        selected = next(item[3]['allow_patterns'] for item in events if item[0] == 'snapshot')
        self.assertEqual(set(selected), {'config.json', 'tokenizer.json', 'model.safetensors.index.json',
                                        'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'})
        index_request = next(item for item in events if item[0] == 'file')
        self.assertEqual(index_request[3], COMMIT)

    def test_generation_prepares_prompt_encoder_without_diffusion_or_scorer_weights(self):
        files = {'model_index.json': {}, 'text_encoder/config.json': {}, 'text_encoder/model.safetensors': '',
                 'text_encoder/pytorch_model.bin': '', 'tokenizer/tokenizer_config.json': {},
                 'tokenizer/vocab.json': {}, 'tokenizer/merges.txt': '', 'tokenizer/README.md': '',
                 'unet/config.json': {}, 'unet/diffusion_pytorch_model.safetensors': '',
                 'vae/diffusion_pytorch_model.safetensors': '', 'safety_checker/model.safetensors': '',
                 'text_encoder_2/model.safetensors': '', 'scheduler/scheduler_config.json': {}}
        events = self.hub(files)
        ml.prepare_analysis_model(self.selection('generation'))
        selected = next(item[3]['allow_patterns'] for item in events if item[0] == 'snapshot')
        self.assertEqual(set(selected), {'model_index.json', 'text_encoder/config.json', 'text_encoder/model.safetensors',
                                        'tokenizer/tokenizer_config.json', 'tokenizer/vocab.json', 'tokenizer/merges.txt'})

    def test_generation_includes_referenced_metadata_within_each_prompt_component(self):
        files = {'model_index.json': {}, 'text_encoder/model.safetensors': '',
                 'text_encoder/config.json': {'configuration_files': ['config.4.0.0.json']},
                 'text_encoder/config.4.0.0.json': {},
                 'tokenizer/tokenizer_config.json': {'fast_tokenizer_files': ['assets/tokenizer.4.0.0.json']},
                 'tokenizer/assets/tokenizer.4.0.0.json': {}, 'unet/config.4.0.0.json': {}}
        events = self.hub(files)
        ml.prepare_analysis_model(self.selection('generation'))
        selected = next(item[3]['allow_patterns'] for item in events if item[0] == 'snapshot')
        self.assertEqual(set(selected), set(files) - {'unet/config.4.0.0.json'})

    def test_metadata_references_cannot_escape_their_component_or_select_code(self):
        for prefix, filename, field in [('text_encoder/', 'config.json', 'configuration_files'),
                                        ('tokenizer/', 'tokenizer_config.json', 'fast_tokenizer_files')]:
            for reference in ['../unet/config.json', '/tmp/config.json', 'other/code.py',
                              'config.py', 'nested/../../config.json', 'missing.json', 'config*.json']:
                with self.subTest(prefix=prefix, reference=reference):
                    files = {'model_index.json': {}, 'text_encoder/config.json': {},
                             'text_encoder/model.safetensors': '', prefix + filename: {field: [reference]},
                             'unet/config.json': {}, prefix + 'config.py': '', prefix + 'subdir/config.json': {}}
                    self.hub(files)
                    with self.assertRaisesRegex(ValueError, 'metadata|reference'):
                        ml.prepare_analysis_model(self.selection('generation'))

    def test_classification_includes_processor_and_can_use_binary_only_checkpoint(self):
        events = self.hub({'config.json': {}, 'preprocessor_config.json': {}, 'pytorch_model.bin': '', 'tf_model.h5': ''})
        ml.prepare_analysis_model(self.selection('classification'))
        selected = next(item[3]['allow_patterns'] for item in events if item[0] == 'snapshot')
        self.assertEqual(set(selected), {'config.json', 'preprocessor_config.json', 'pytorch_model.bin'})

    def test_checkpoint_index_cannot_escape_component_or_select_other_weight_formats(self):
        for shard in ['../unet/model.safetensors', '/tmp/model.safetensors', 'another.bin', 'missing.safetensors']:
            with self.subTest(shard=shard):
                self.hub({'config.json': {}, 'model.safetensors.index.json': {'weight_map': {'weight': shard}}})
                with self.assertRaisesRegex(ValueError, 'shard|checkpoint'):
                    ml.prepare_analysis_model(self.selection())

    def test_missing_weights_and_mutable_revision_metadata_are_rejected(self):
        self.hub({'config.json': {}, 'tokenizer.json': {}})
        with self.assertRaisesRegex(ValueError, 'checkpoint|weights'):
            ml.prepare_analysis_model(self.selection())
        self.hub({'config.json': {}, 'model.safetensors': ''}, commit='main')
        with self.assertRaisesRegex(ValueError, 'immutable|revision'):
            ml.prepare_analysis_model(self.selection())

    def test_offline_snapshot_resolves_commit_and_validates_referenced_files(self):
        (self.snapshot/'config.json').write_text('{}')
        (self.snapshot/'model.safetensors').write_text('weights')
        self.hub({}, offline=True)
        prepared, metadata = ml.prepare_analysis_model(self.selection())
        self.assertEqual(prepared['name'], str(self.snapshot))
        self.assertEqual(metadata['revision'], COMMIT)
        self.assertTrue(metadata['files_reused'])
        (self.snapshot/'model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'checkpoint|weights'):
            ml.prepare_analysis_model(self.selection())

    def test_incomplete_download_keeps_the_previous_branch_reference(self):
        self.hub({'config.json': {}, 'model.safetensors': ''})
        reference = self.snapshot.parent.parent/'refs'/'main'
        reference.parent.mkdir()
        reference.write_text('b' * 40)
        with patch.object(sys.modules['huggingface_hub'], 'snapshot_download', return_value=str(self.snapshot)):
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                ml.prepare_analysis_model(self.selection())
        self.assertEqual(reference.read_text(), 'b' * 40)

    def test_ref_update_rejects_cache_symlinks_outside_the_refs_directory(self):
        self.hub({'config.json': {}, 'model.safetensors': ''})
        outside = Path(self.temporary.name)/'outside'
        outside.mkdir()
        (self.snapshot.parent.parent/'refs').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'refs directory'):
            ml.prepare_analysis_model(self.selection())
        self.assertEqual(list(outside.iterdir()), [])

    def test_read_only_ref_update_keeps_valid_prepared_weights_and_removes_temporary_file(self):
        self.hub({'config.json': {}, 'model.safetensors': ''})
        with patch.object(ml.os, 'replace', side_effect=PermissionError('read-only cache')), \
                self.assertLogs('model_loading', level='WARNING'):
            prepared, _ = ml.prepare_analysis_model(self.selection())
        self.assertTrue((Path(prepared['name'])/'model.safetensors').is_file())
        self.assertEqual(list((self.snapshot.parent.parent/'refs').iterdir()), [])

    def test_invalid_selections_fail_before_hub_access(self):
        events = self.hub({'config.json': {}, 'model.safetensors': ''})
        for update in [{'source': 'other'}, {'name': '../escape'}, {'task': 'audio'}, {'name': ''}, {'revision': 2},
                       {'revision': '../escape'}, {'revision': '/absolute'}, {'revision': 'refs//main'}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                ml.prepare_analysis_model({**self.selection(), **update})
        self.assertEqual(events, [])


class PreparedSnapshotTests(unittest.TestCase):
    setUp = ModelLoadingTests.setUp
    hub = ModelLoadingTests.hub
    selection = ModelLoadingTests.selection

    def _versioned_round_trip(self, metadata_file, field, asset, versioned):
        import numpy as np
        import torch
        from transformers import T5Config, T5ForConditionalGeneration
        from test_text_encoders import save_tokenizer, VOCAB
        from text_encoders import SummarizationEncoder
        full = Path(self.temporary.name)/'full'
        torch.manual_seed(33)
        T5ForConditionalGeneration(T5Config(vocab_size=len(VOCAB), d_model=12, d_ff=24,
            num_layers=1, num_decoder_layers=1, num_heads=3, d_kv=4, dropout_rate=0.,
            decoder_start_token_id=0, eos_token_id=3, pad_token_id=0)).save_pretrained(full)
        save_tokenizer(full)
        (full/versioned).write_bytes((full/asset).read_bytes())
        if asset != metadata_file:
            (full/asset).unlink()
        metadata = json.loads((full/metadata_file).read_text())
        metadata[field] = [versioned]
        (full/metadata_file).write_text(json.dumps(metadata))
        expected = SummarizationEncoder({'source': 'local', 'name': str(full), 'task': 'summarization'}).encode(['a cat'])
        files = {path.name: path.read_bytes() for path in full.iterdir() if path.is_file()}
        self.hub(files)
        prepared, timing = ml.prepare_analysis_model(self.selection())
        self.hub_replacement.stop()
        self.assertEqual(timing['revision'], COMMIT)
        self.assertTrue((Path(prepared['name'])/versioned).is_file())
        actual = SummarizationEncoder(prepared).encode(['a cat'])
        np.testing.assert_array_equal(actual, expected)

    def test_versioned_configuration_survives_selective_preparation_and_real_local_loading(self):
        self._versioned_round_trip('config.json', 'configuration_files', 'config.json', 'config.4.0.0.json')

    def test_versioned_tokenizer_survives_selective_preparation_and_real_local_loading(self):
        self._versioned_round_trip('tokenizer_config.json', 'fast_tokenizer_files', 'tokenizer.json', 'tokenizer.4.0.0.json')


class HubCacheTests(unittest.TestCase):
    def test_pinned_preparation_retains_branch_refs_for_real_hub_offline_resolution(self):
        import huggingface_hub as hub
        files = {'config.json': '{}', 'model.safetensors': 'weights'}
        info = hub.ModelInfo(id='team/checkpoint', sha=COMMIT, siblings=[{'rfilename': name} for name in files])
        for requested in (None, 'release/v1'):
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as directory:
                snapshot = Path(directory)/'models--team--checkpoint'/'snapshots'/COMMIT
                def materialize(repo_id, filename, revision=None, **kwargs):
                    self.assertEqual(repo_id, 'team/checkpoint')
                    self.assertEqual(revision, COMMIT)
                    path = snapshot/filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(files[filename])
                    return str(path)
                selection = {'source': 'huggingface', 'name': 'team/checkpoint', 'task': 'summarization'}
                if requested is not None:
                    selection['revision'] = requested
                # Keep snapshot_download and cache lookup real; replace only
                # external metadata and file transfers into this isolated cache.
                with patch.object(hub.constants, 'HF_HUB_CACHE', directory), \
                        patch.object(hub.HfApi, 'model_info', return_value=info), \
                        patch.object(hub.HfApi, 'repo_info', return_value=info), \
                        patch.object(hub, 'hf_hub_download', side_effect=materialize), \
                        patch('huggingface_hub._snapshot_download.hf_hub_download', side_effect=materialize):
                    prepared, _ = ml.prepare_analysis_model(selection)
                    cached_file = hub.try_to_load_from_cache('team/checkpoint', 'config.json',
                        cache_dir=directory, revision=requested)
                    self.assertEqual(cached_file, str(snapshot/'config.json'))
                    with patch.object(hub.HfApi, 'model_info', side_effect=ConnectionError('offline')):
                        offline, timing = ml.prepare_analysis_model(selection)
                    self.assertEqual(offline, prepared)
                    self.assertTrue(timing['files_reused'])


class ShardLoadingTests(unittest.TestCase):
    def test_real_sharded_checkpoint_has_identical_outputs_with_parallel_loading(self):
        import torch
        from transformers import T5Config, T5ForConditionalGeneration
        torch.set_num_threads(1)
        torch.manual_seed(31)
        reference = T5ForConditionalGeneration(T5Config(vocab_size=37, d_model=12, d_ff=24,
            num_layers=1, num_decoder_layers=1, num_heads=3, d_kv=4, dropout_rate=0.,
            decoder_start_token_id=0, eos_token_id=1, pad_token_id=0)).eval()
        inputs = {'input_ids': torch.tensor([[2, 3, 1]]), 'decoder_input_ids': torch.tensor([[0, 4]])}
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory, max_shard_size='5KB', safe_serialization=True)
            self.assertTrue((Path(directory)/'model.safetensors.index.json').is_file())
            outputs = []
            for setting in ('false', 'true'):
                with patch.dict(os.environ, {'HF_ENABLE_PARALLEL_LOADING': setting, 'HF_PARALLEL_LOADING_WORKERS': '4'}):
                    ml.configure_parallel_loading()
                    loaded = T5ForConditionalGeneration.from_pretrained(directory, local_files_only=True, weights_only=True)
                self.assertTrue(all(torch.equal(value, loaded.state_dict()[name]) for name, value in reference.state_dict().items()))
                with torch.inference_mode():
                    outputs.append(loaded(**inputs).logits)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
