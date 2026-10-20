"""Offline adapter checks with real, locally saved tiny Transformers models."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import (
    AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, CLIPTextConfig,
    CLIPTextModel, CLIPTokenizerFast, GPT2Config, PreTrainedTokenizerFast,
    T5Config, T5ForConditionalGeneration,
)

import text_encoders


VOCAB = {word: index for index, word in enumerate([
    '[PAD]', '[UNK]', '[BOS]', '[EOS]', 'summarize:', 'brief:', 'task:',
    'a', 'cat', 'in', 'the', 'garden', 'red', 'blue', 'runs', 'quietly',
])}


def clip_vocabulary():
    # A tiny CLIP-compatible BPE vocabulary usable by both slow and fast loaders.
    vocabulary = {word if index < 7 else word+'</w>': index for word, index in VOCAB.items()}
    merges = []
    for word in list(VOCAB)[7:]:
        pieces = [*word[:-1], word[-1]+'</w>']
        for char in pieces:
            vocabulary.setdefault(char, len(vocabulary))
        prefix = pieces[0]
        for char in pieces[1:]:
            pair = (prefix, char)
            if pair not in merges:
                merges.append(pair)
            prefix += char
            vocabulary.setdefault(prefix, len(vocabulary))
    return vocabulary, merges


def save_tokenizer(path, clip=False, limit=12):
    vocabulary, merges = clip_vocabulary()
    backend = Tokenizer(BPE(vocabulary, merges, unk_token='[UNK]', end_of_word_suffix='</w>') if clip
                        else WordLevel(VOCAB, unk_token='[UNK]'))
    backend.pre_tokenizer = Sequence([WhitespaceSplit()]) if clip else WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single='[BOS] $A [EOS]' if clip else '$A [EOS]',
        special_tokens=[('[BOS]', 2), ('[EOS]', 3)],
    )
    cls = CLIPTokenizerFast if clip else PreTrainedTokenizerFast
    tokenizer = cls(tokenizer_object=backend, bos_token='[BOS]', eos_token='[EOS]',
                    unk_token='[UNK]', pad_token='[PAD]', model_max_length=limit)
    tokenizer.save_pretrained(path)


class TextEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.summary = cls.root/'summary'
        cls.pipeline = cls.root/'pipeline'
        torch.manual_seed(734)
        cls.summary_reference = T5ForConditionalGeneration(T5Config(
            vocab_size=len(VOCAB), d_model=12, d_kv=4, d_ff=24,
            num_layers=2, num_decoder_layers=1, num_heads=3,
            dropout_rate=.4, pad_token_id=0, eos_token_id=3,
            decoder_start_token_id=0,
        )).eval()
        cls.summary_reference.save_pretrained(cls.summary)
        save_tokenizer(cls.summary)
        cls.clip_reference = CLIPTextModel(CLIPTextConfig(
            vocab_size=len(clip_vocabulary()[0]), hidden_size=12, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=3, max_position_embeddings=12,
            bos_token_id=2, eos_token_id=3, pad_token_id=0, attention_dropout=.4,
        )).eval()
        with torch.no_grad():
            cls.clip_reference.text_model.final_layer_norm.weight.copy_(torch.linspace(.4, 1.6, 12))
            cls.clip_reference.text_model.final_layer_norm.bias.copy_(torch.linspace(-.2, .3, 12))
        cls.clip_reference.save_pretrained(cls.pipeline/'text_encoder')
        save_tokenizer(cls.pipeline/'tokenizer', clip=True)
        (cls.pipeline/'model_index.json').write_text(json.dumps({
            '_class_name': 'StableDiffusionPipeline',
            'text_encoder': ['transformers', 'CLIPTextModel'],
            'tokenizer': ['transformers', 'CLIPTokenizerFast'],
            'unet': ['diffusers', 'UNet2DConditionModel'],
            'vae': ['diffusers', 'AutoencoderKL'],
        }))

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)

    def selection(self, path=None, task='summarization'):
        return {'source': 'local', 'name': str(path or self.summary), 'task': task}

    def copied(self, source):
        return Path(shutil.copytree(source, Path(self.scratch.name)/source.name))

    def summary_expected(self, ids):
        with torch.inference_mode():
            output = self.summary_reference.get_encoder()(
                input_ids=torch.tensor([ids]), attention_mask=torch.ones((1, len(ids))),
                return_dict=True,
            )
        return output.last_hidden_state[0].mean(dim=0).numpy()

    def test_summarization_pools_final_encoder_states_with_t5_prefix(self):
        encoder = text_encoders.SummarizationEncoder(self.selection())
        expected = self.summary_expected([4, 7, 8, 3])
        actual = encoder.encode(['a cat'])
        self.assertEqual(actual.shape, (1, 12))
        np.testing.assert_allclose(actual[0], expected, rtol=2e-6, atol=2e-6)
        self.assertIn('last_hidden_state', encoder.feature_method)
        self.assertIsNone(encoder.revision)

    def test_direct_adapters_default_to_cpu_and_expose_device_switching(self):
        for path, task, adapter in [(self.summary, 'summarization', text_encoders.SummarizationEncoder),
                                    (self.pipeline, 'generation', text_encoders.DiffusionPromptEncoder)]:
            with self.subTest(task=task):
                encoder = adapter(self.selection(path, task))
                before = encoder.encode(['a cat'])
                self.assertEqual(encoder.device, 'cpu')
                encoder.set_device('cpu')
                np.testing.assert_array_equal(before, encoder.encode(['a cat']))
                with self.assertRaisesRegex(ValueError, 'device|cpu|cuda'):
                    encoder.set_device('invalid')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware is unavailable')
    def test_cuda_text_adapters_keep_weights_and_inputs_together_and_preserve_features(self):
        for path, task, adapter in [(self.summary, 'summarization', text_encoders.SummarizationEncoder),
                                    (self.pipeline, 'generation', text_encoders.DiffusionPromptEncoder)]:
            with self.subTest(task=task):
                selection = self.selection(path, task)
                cpu = adapter({**selection, 'device': 'cpu'})
                gpu = adapter({**selection, 'device': 'cuda'})
                observed = []

                def observe(module, args, kwargs):
                    observed.append((next(module.parameters()).device.type,
                                     next(module.parameters()).dtype,
                                     {value.device.type for value in kwargs.values() if torch.is_tensor(value)},
                                     torch.is_inference_mode_enabled()))

                network = gpu.encoder if task == 'summarization' else gpu.model
                hook = network.register_forward_pre_hook(observe, with_kwargs=True)
                try:
                    texts = ['a cat', 'a cat in the garden runs quietly']
                    expected = cpu.encode(texts)
                    actual = gpu.encode(texts)
                finally:
                    hook.remove()
                self.assertEqual(gpu.device, 'cuda')
                self.assertEqual(observed, [('cuda', torch.float32, {'cuda'}, True)])
                self.assertEqual(actual.dtype, np.float64)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
                gpu.set_device('cpu')
                self.assertEqual(gpu.device, 'cpu')
                np.testing.assert_allclose(gpu.encode(texts), expected, rtol=2e-6, atol=2e-6)

    def test_summarization_is_padding_invariant_and_runs_encoder_only_in_inference_mode(self):
        encoder = text_encoders.SummarizationEncoder(self.selection())
        observations = []
        handle = encoder.model.get_encoder().register_forward_pre_hook(
            lambda module, args: observations.append((module.training, torch.is_inference_mode_enabled()))
        )
        self.addCleanup(handle.remove)
        with patch.object(encoder.model.get_decoder(), 'forward', side_effect=AssertionError('decoder invoked')):
            alone = encoder.encode(['a cat'])
            together = encoder.encode(['a cat', 'a cat in the garden runs quietly'])
        np.testing.assert_allclose(alone[0], together[0], rtol=2e-6, atol=2e-6)
        self.assertEqual(observations, [(False, True), (False, True)])
        self.assertEqual(next(encoder.model.parameters()).device.type, 'cpu')

    def test_summarization_prefers_task_prefix_then_config_prefix_and_respects_empty_prefix(self):
        directory = self.copied(self.summary)
        config = json.loads((directory/'config.json').read_text())
        for task_params, prefix, ids in [
            ({'summarization': {'prefix': 'task: '}}, 'brief: ', [6, 7, 8, 3]),
            ({}, 'brief: ', [5, 7, 8, 3]),
            ({'summarization': {'prefix': ''}}, 'brief: ', [7, 8, 3]),
        ]:
            with self.subTest(task_params=task_params):
                config.update(task_specific_params=task_params, prefix=prefix)
                (directory/'config.json').write_text(json.dumps(config))
                encoder = text_encoders.SummarizationEncoder(self.selection(directory))
                np.testing.assert_allclose(encoder.encode(['a cat'])[0], self.summary_expected(ids),
                                           rtol=2e-6, atol=2e-6)

    def test_summary_token_limit_counts_prefix_and_special_tokens_without_truncation(self):
        encoder = text_encoders.SummarizationEncoder(self.selection())
        self.assertEqual(encoder.encode([' '.join(['cat']*10)]).shape, (1, 12))
        with self.assertRaisesRegex(ValueError, r'13.*12|12.*13'):
            encoder.encode(['a cat', ' '.join(['cat']*11)])

    def test_summary_dataset_truncation_bounds_padded_model_inputs_and_preserves_prefix(self):
        encoder = text_encoders.SummarizationEncoder(self.selection())
        observed = []

        def observe(module, args, kwargs):
            observed.append(tuple(kwargs['input_ids'].shape))

        handle = encoder.encoder.register_forward_pre_hook(observe, with_kwargs=True)
        self.addCleanup(handle.remove)
        actual = encoder.encode(['a cat', ' '.join(['cat'] * 100000)], truncate=True)
        self.assertEqual(observed, [(2, 12)])
        np.testing.assert_allclose(actual[1], self.summary_expected([4] + [8]*10 + [3]),
                                   rtol=2e-6, atol=2e-6)
        prepared = encoder._prepare([' '.join(['cat']*100)], encoder.prefix, truncate=True)
        self.assertEqual(tuple(prepared['input_ids'].shape), (1, 12))
        with self.assertRaisesRegex(ValueError, 'truncation is disabled'):
            encoder.encode([' '.join(['cat']*11)])

    def test_summary_respects_model_input_limit_and_finite_fallbacks(self):
        directory = self.copied(self.summary)
        config = json.loads((directory/'config.json').read_text())
        for tokenizer_limit, model_limit, count, limit in [(12, 6, 5, 6), (10**30, None, 511, 512),
                                                          (4096, None, 2047, 2048)]:
            with self.subTest(limit=limit):
                config.pop('n_positions', None)
                if model_limit is not None:
                    config['n_positions'] = model_limit
                (directory/'config.json').write_text(json.dumps(config))
                save_tokenizer(directory, limit=tokenizer_limit)
                encoder = text_encoders.SummarizationEncoder(self.selection(directory))
                with self.assertRaisesRegex(ValueError, rf'\b{limit}\b'):
                    encoder.encode([' '.join(['cat']*count)])

    def test_summary_rejects_non_seq2seq_architecture(self):
        directory = Path(self.scratch.name)/'causal'
        GPT2Config(n_embd=12, n_layer=1, n_head=3).save_pretrained(directory)
        with self.assertRaisesRegex(ValueError, 'encoder.decoder|sequence.to.sequence|seq2seq'):
            text_encoders.SummarizationEncoder(self.selection(directory))

    def test_diffusion_uses_final_clip_layer_norm_and_non_padding_mean(self):
        with patch.dict('sys.modules', {'diffusers': None}):
            encoder = text_encoders.DiffusionPromptEncoder(self.selection(self.pipeline, 'generation'))
            actual = encoder.encode(['a cat', 'a cat in the garden'])
        with torch.inference_mode():
            output = self.clip_reference(input_ids=torch.tensor([[2, 7, 8, 3]]),
                                         output_hidden_states=True, return_dict=True)
        expected = output.last_hidden_state[0].mean(dim=0).numpy()
        np.testing.assert_allclose(actual[0], expected, rtol=2e-6, atol=2e-6)
        self.assertFalse(np.allclose(actual[0], output.pooler_output[0].numpy()))
        self.assertFalse(np.allclose(actual[0], output.hidden_states[-1][0].mean(dim=0).numpy()))
        self.assertEqual(actual.shape, (2, 12))
        self.assertIn('CLIP', encoder.feature_method)
        self.assertIn('conditioning', encoder.feature_method)

    def test_diffusion_padding_and_attention_mask_follow_pipeline_configuration(self):
        directory = self.copied(self.pipeline)
        config_path = directory/'text_encoder'/'config.json'
        config = json.loads(config_path.read_text())
        for use_mask in (False, True):
            with self.subTest(use_attention_mask=use_mask):
                config['use_attention_mask'] = use_mask
                config_path.write_text(json.dumps(config))
                encoder = text_encoders.DiffusionPromptEncoder(self.selection(directory, 'generation'))
                observations = []

                def observe(module, args, kwargs):
                    observations.append((module.training, torch.is_inference_mode_enabled(), kwargs.get('attention_mask')))

                handle = encoder.model.register_forward_pre_hook(observe, with_kwargs=True)
                self.addCleanup(handle.remove)
                alone = encoder.encode(['a cat'])
                together = encoder.encode(['a cat', 'a cat in the garden runs quietly'])
                np.testing.assert_allclose(alone[0], together[0], rtol=2e-6, atol=2e-6)
                self.assertEqual([item[:2] for item in observations], [(False, True), (False, True)])
                self.assertEqual(observations[1][2] is not None, use_mask)
                if use_mask:
                    self.assertEqual(observations[1][2][0].tolist(), [1, 1, 1, 1, 0, 0, 0, 0, 0])

    def test_diffusion_rejects_overlength_prompts_without_truncation(self):
        encoder = text_encoders.DiffusionPromptEncoder(self.selection(self.pipeline, 'generation'))
        self.assertEqual(encoder.encode([' '.join(['cat']*10)]).shape, (1, 12))
        with self.assertRaisesRegex(ValueError, r'13.*12|12.*13'):
            encoder.encode([' '.join(['cat']*11)])

    def test_diffusion_loads_declared_slow_clip_tokenizer_used_by_sd_presets(self):
        directory = self.copied(self.pipeline)
        index_path = directory/'model_index.json'
        index = json.loads(index_path.read_text())
        index['tokenizer'] = ['transformers', 'CLIPTokenizer']
        index_path.write_text(json.dumps(index))
        encoder = text_encoders.DiffusionPromptEncoder(self.selection(directory, 'generation'))
        actual = encoder.encode(['a cat', 'a cat in the garden'])
        with torch.inference_mode():
            expected = self.clip_reference(input_ids=torch.tensor([[2, 7, 8, 3]]), return_dict=True)
        np.testing.assert_allclose(actual[0], expected.last_hidden_state[0].mean(dim=0).numpy(),
                                   rtol=2e-6, atol=2e-6)

    def test_missing_clip_normalizer_cannot_silently_change_prompt_tokens(self):
        directory = self.copied(self.pipeline)
        index_path = directory/'model_index.json'
        index = json.loads(index_path.read_text())
        index['tokenizer'] = ['transformers', 'CLIPTokenizer']
        index_path.write_text(json.dumps(index))
        with patch.dict('sys.modules', {'ftfy': None}):
            with self.assertRaisesRegex(ValueError, 'ftfy'):
                text_encoders.DiffusionPromptEncoder(self.selection(directory, 'generation'))

    def test_diffusion_rejects_unsupported_pipeline_or_component_declarations(self):
        directory = self.copied(self.pipeline)
        index_path = directory/'model_index.json'
        original = json.loads(index_path.read_text())
        changes = [
            {'_class_name': 'StableDiffusionXLPipeline'}, {'_class_name': 'FluxPipeline'},
            {'text_encoder': ['transformers', 'T5EncoderModel']},
            {'text_encoder': ['custom_package', 'CLIPTextModel']},
            {'tokenizer': ['transformers', 'T5Tokenizer']}, {'tokenizer': None},
            {'text_encoder_2': ['transformers', 'CLIPTextModelWithProjection']},
        ]
        for change in changes:
            with self.subTest(change=change):
                index_path.write_text(json.dumps({**original, **change}))
                with self.assertRaisesRegex(ValueError, 'Stable Diffusion|CLIP|unsupported|Unsupported'):
                    text_encoders.DiffusionPromptEncoder(self.selection(directory, 'generation'))

    def test_both_adapters_reject_missing_partial_and_mismatched_pretrained_weights(self):
        for source, task, adapter in [(self.summary, 'summarization', text_encoders.SummarizationEncoder),
                                      (self.pipeline, 'generation', text_encoders.DiffusionPromptEncoder)]:
            directory = self.copied(source)
            weight_path = directory/('text_encoder/model.safetensors' if task == 'generation' else 'model.safetensors')
            original = load_file(weight_path)
            key = next(key for key in original if key.endswith('self_attn.k_proj.weight') or
                       (key.startswith('encoder.') and key.endswith('SelfAttention.k.weight')))
            for failure in ('missing file', 'missing key', 'mismatch'):
                with self.subTest(task=task, failure=failure):
                    weight_path.unlink(missing_ok=True)
                    state = dict(original)
                    if failure == 'missing key':
                        del state[key]
                    elif failure == 'mismatch':
                        state[key] = state[key][:-1]
                    if failure != 'missing file':
                        save_file(state, weight_path, metadata={'format': 'pt'})
                    with self.assertRaisesRegex(ValueError, 'weights|checkpoint|size mismatch|shape'):
                        adapter(self.selection(directory, task))

    def test_adapters_reject_nonfinite_model_features(self):
        for directory, task, cls in [(self.summary, 'summarization', text_encoders.SummarizationEncoder),
                                     (self.pipeline, 'generation', text_encoders.DiffusionPromptEncoder)]:
            with self.subTest(task=task):
                encoder = cls(self.selection(directory, task))
                with torch.no_grad():
                    encoder.model.get_input_embeddings().weight.fill_(float('nan'))
                with self.assertRaisesRegex(ValueError, 'finite'):
                    encoder.encode(['a cat'])

    def test_bad_selections_and_batches_are_clear_errors(self):
        for cls, task in [(text_encoders.SummarizationEncoder, 'summarization'),
                          (text_encoders.DiffusionPromptEncoder, 'generation')]:
            for selection in [None, {'source': 'other', 'name': 'a'},
                              {'source': 'huggingface', 'name': '../bad'},
                              self.selection(self.summary/'config.json', task)]:
                with self.subTest(selection=selection, task=task), self.assertRaises(ValueError):
                    cls(selection)
        encoder = text_encoders.SummarizationEncoder(self.selection())
        for batch in [[], 'a cat', [None], [123]]:
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, 'list|text|input'):
                encoder.encode(batch)

    def test_safe_weights_loading_requires_supported_pytorch(self):
        with patch.object(torch, '__version__', '2.5.1'), self.assertRaisesRegex(ValueError, '2.6'):
            text_encoders.SummarizationEncoder(self.selection())

    def test_summary_pins_resolved_revision_for_safe_model_and_tokenizer_loads(self):
        commit = 'a'*40
        original_config = AutoConfig.from_pretrained
        original_model = AutoModelForSeq2SeqLM.from_pretrained
        original_tokenizer = AutoTokenizer.from_pretrained

        def config_download(name, **options):
            self.assertEqual(name, 'fixture/summary')
            self.assertEqual(options.get('revision'), 'release')
            self.assertIs(options.get('trust_remote_code'), False)
            result = original_config(self.summary, local_files_only=True)
            result._commit_hash = commit
            return result

        def model_download(name, **options):
            self.assertEqual(name, 'fixture/summary')
            self.assertEqual(options.get('revision'), commit)
            self.assertIs(options.get('weights_only'), True)
            self.assertIs(options.get('trust_remote_code'), False)
            return original_model(self.summary, **{**options, 'local_files_only': True})

        def tokenizer_download(name, **options):
            self.assertEqual(name, 'fixture/summary')
            self.assertEqual(options.get('revision'), commit)
            self.assertIs(options.get('trust_remote_code'), False)
            return original_tokenizer(self.summary, **{**options, 'local_files_only': True})

        with patch.object(AutoConfig, 'from_pretrained', side_effect=config_download), \
             patch.object(AutoModelForSeq2SeqLM, 'from_pretrained', side_effect=model_download), \
             patch.object(AutoTokenizer, 'from_pretrained', side_effect=tokenizer_download):
            encoder = text_encoders.SummarizationEncoder({
                'source': 'huggingface', 'name': 'fixture/summary', 'task': 'summarization', 'revision': 'release',
            })
            self.assertEqual(encoder.revision, commit)
            np.testing.assert_allclose(encoder.encode(['a cat'])[0], self.summary_expected([4, 7, 8, 3]),
                                       rtol=2e-6, atol=2e-6)

    def test_diffusion_pins_index_revision_and_loads_only_declared_conditioning_folders(self):
        commit = 'b'*40
        snapshot = Path(self.scratch.name)/'snapshots'/commit
        shutil.copytree(self.pipeline, snapshot)
        original_config = AutoConfig.from_pretrained
        original_model = CLIPTextModel.from_pretrained
        original_tokenizer = CLIPTokenizerFast.from_pretrained

        def index_download(repo_id, filename, **options):
            self.assertEqual((repo_id, filename), ('fixture/diffusion', 'model_index.json'))
            self.assertEqual(options.get('revision'), 'release')
            return str(snapshot/filename)

        def component_download(loader, folder, name, options):
            self.assertEqual(name, 'fixture/diffusion')
            self.assertEqual(options.get('revision'), commit)
            self.assertEqual(options.get('subfolder'), folder)
            self.assertIs(options.get('trust_remote_code'), False)
            return loader(snapshot, **{**options, 'local_files_only': True})

        def model_download(name, **options):
            self.assertIs(options.get('weights_only'), True)
            return component_download(original_model, 'text_encoder', name, options)

        with patch('huggingface_hub.hf_hub_download', side_effect=index_download), \
             patch('huggingface_hub.snapshot_download', side_effect=AssertionError('full pipeline download')), \
             patch.object(AutoConfig, 'from_pretrained', side_effect=lambda name, **options:
                          component_download(original_config, 'text_encoder', name, options)), \
             patch.object(CLIPTextModel, 'from_pretrained', side_effect=model_download), \
             patch.object(CLIPTokenizerFast, 'from_pretrained', side_effect=lambda name, **options:
                          component_download(original_tokenizer, 'tokenizer', name, options)):
            encoder = text_encoders.DiffusionPromptEncoder({
                'source': 'huggingface', 'name': 'fixture/diffusion', 'task': 'generation', 'revision': 'release',
            })
            self.assertEqual(encoder.revision, commit)
            actual = encoder.encode(['a cat'])
        with torch.inference_mode():
            output = self.clip_reference(input_ids=torch.tensor([[2, 7, 8, 3]]), return_dict=True)
        np.testing.assert_allclose(actual[0], output.last_hidden_state[0].mean(dim=0).numpy(),
                                   rtol=2e-6, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
