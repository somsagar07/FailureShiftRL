"""Requirement scoring uses image answers, never the expected answer as model input."""
import tempfile
import unittest
import gc
import sys
import weakref
from pathlib import Path
from unittest.mock import patch

from generation_judge import QuestionJudge, check_answers, normalize_answer


def save_blip_fixture(directory):
    import torch
    from transformers import BertTokenizerFast, BlipConfig, BlipForQuestionAnswering, BlipImageProcessor, BlipProcessor
    directory.mkdir(parents=True, exist_ok=True)
    words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'yes', 'no', 'one', 'two', 'cat', 'is', 'there', '?']
    (directory/'vocab.txt').write_text('\n'.join(words))
    tokenizer = BertTokenizerFast(vocab_file=str(directory/'vocab.txt'), model_max_length=128)
    config = BlipConfig(text_config=dict(vocab_size=len(words), hidden_size=12, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=3, max_position_embeddings=128,
        encoder_hidden_size=12, bos_token_id=2, eos_token_id=3, sep_token_id=3, pad_token_id=0),
        vision_config=dict(hidden_size=12, intermediate_size=24, num_hidden_layers=1,
                           num_attention_heads=3, image_size=16, patch_size=8))
    torch.manual_seed(181)
    model = BlipForQuestionAnswering(config)
    model.generation_config.suppress_tokens = list(range(5))
    model.save_pretrained(directory)
    BlipProcessor(BlipImageProcessor(size={'height':16, 'width':16}), tokenizer).save_pretrained(directory)


def save_qwen_fixture(directory, *, rating=None):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen3VLVideoProcessor,
                              Qwen3VLProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration)
    words = ['<|pad|>', '<|unk|>', '<|endoftext|>', '<|image_pad|>', '<|video_pad|>', '<|vision_start|>',
             '<|vision_end|>', 'yes', 'no', 'one', 'two', 'cat', 'is', 'there', '?']
    if rating is not None:
        if type(rating) is not int or not 0 <= rating <= 4:
            raise ValueError('Fixture rating must be an integer from 0 to 4.')
        words.append(str(rating))
    vocab = {word:index for index,word in enumerate(words)}
    backend = Tokenizer(WordLevel(vocab, unk_token='<|unk|>'))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, pad_token='<|pad|>', unk_token='<|unk|>',
        eos_token='<|endoftext|>', model_max_length=256,
        additional_special_tokens=words[3:7])
    template = ('{% for message in messages %}{% for content in message.content %}'
        '{% if content.type == "image" %}<|vision_start|><|image_pad|><|vision_end|> '
        '{% else %}{{ content.text }}{% endif %}{% endfor %}{% endfor %}')
    processor = Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(min_pixels=256, max_pixels=256,
        patch_size=8, temporal_patch_size=2, merge_size=2), tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(), chat_template=template)
    config = Qwen3VLConfig(text_config=dict(vocab_size=len(words), hidden_size=24, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=3, num_key_value_heads=3, head_dim=8,
        max_position_embeddings=256, rope_scaling={'rope_type':'default','mrope_section':[2,1,1]},
        pad_token_id=0, eos_token_id=2), vision_config=dict(depth=1, hidden_size=24,
        intermediate_size=48, num_heads=3, patch_size=8, temporal_patch_size=2, spatial_merge_size=2,
        out_hidden_size=24, num_position_embeddings=16, deepstack_visual_indexes=[]),
        image_token_id=3, video_token_id=4, vision_start_token_id=5, vision_end_token_id=6,
        pad_token_id=0, eos_token_id=2)
    torch.manual_seed(182)
    model = Qwen3VLForConditionalGeneration(config)
    model.generation_config.suppress_tokens = list(range(7))
    if rating is not None:
        # Constrain this tiny fixture to a numeric EOS so real multimodal
        # inference exercises the rubric protocol without pretending the random
        # weights can judge image quality. Production models are unconstrained.
        model.generation_config.suppress_tokens = list(range(len(words)-1))
        model.generation_config.eos_token_id = len(words)-1
    model.save_pretrained(directory)
    processor.save_pretrained(directory)


class QuestionScoringTests(unittest.TestCase):
    def test_case_punctuation_and_number_aliases_match_without_substring_matching(self):
        for actual, expected in [(' YES. ', 'Yes'), ('seven', '7'), ('Two.', 'two'), ('blue', 'Blue')]:
            self.assertEqual(normalize_answer(actual), normalize_answer(expected))
        for actual, expected in [('not blue', 'blue'), ('one or two', 'one'), ('yesterday', 'yes')]:
            self.assertNotEqual(normalize_answer(actual), normalize_answer(expected))

    def test_wrong_count_and_no_answer_are_independent_failed_checks(self):
        requirements = [{'question': 'How many cats?', 'answer': 'two', 'skill': 'count'},
                        {'question': 'Are the cats red?', 'answer': 'Yes', 'skill': 'attribute'}]
        result = check_answers(requirements, ['three', 'No'])
        self.assertEqual(result['quality'], 0.)
        self.assertEqual(result['checks_passed'], 0)
        self.assertEqual(result['checks_total'], 2)
        self.assertEqual(result['checks'][0]['expected'], 'two')
        self.assertEqual(result['checks'][0]['answer'], 'three')
        self.assertEqual(result['checks'][0]['skill'], 'count')

    def test_quality_is_passed_fraction_not_a_calibrated_probability(self):
        checks = [{'question': 'Is there a cat?', 'answer': 'Yes'},
                  {'question': 'How many cats?', 'answer': 'two'}]
        result = check_answers(checks, ['Yes', 'one'])
        self.assertEqual(result['quality'], .5)
        self.assertEqual([row['passed'] for row in result['checks']], [True, False])

    def test_missing_or_empty_judge_output_is_an_error_not_an_observed_failure(self):
        checks = [{'question': 'Any cat?', 'answer': 'Yes'}]
        for answers in ([], [''], [None], ['Yes', 'No']):
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                check_answers(checks, answers)
        with self.assertRaises(ValueError):
            check_answers([], [])


class RealQuestionJudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        save_blip_fixture(cls.root/'blip')
        save_qwen_fixture(cls.root/'qwen')

    def test_both_architectures_load_real_weights_and_answer_the_image_without_targets(self):
        from PIL import Image
        image = Image.new('RGB', (16,16), 'red')
        checks = [{'question':'is there cat ?', 'answer':'one two'},
                  {'question':'is there cat ?', 'answer':'no two'}]
        for name in ('blip','qwen'):
            with self.subTest(name=name):
                judge = QuestionJudge({'source':'local','name':str(self.root/name)}, 'cpu')
                # Observe the inputs to real inference, so this fails if a
                # target leaks into either architecture's question prompt.
                tokens = []
                generate = judge.model.generate
                def observe(**kwargs):
                    tokens.extend(row[mask.bool()].tolist() for row, mask in
                                  zip(kwargs['input_ids'], kwargs['attention_mask']))
                    return generate(**kwargs)
                with patch.object(judge.model, 'generate', new=observe):
                    result = judge.evaluate(image,checks)
                self.assertEqual(len(tokens), 2)
                self.assertEqual(tokens[0], tokens[1])
                target_ids = judge.processor.tokenizer.convert_tokens_to_ids(['one', 'two', 'no'])
                self.assertFalse(set(tokens[0]) & set(target_ids))
                self.assertEqual(result['checks_total'],2)
                self.assertEqual(result['checks_passed'],0)
                self.assertEqual(result['checks'][0]['answer'],result['checks'][1]['answer'])
                self.assertTrue(result['checks'][0]['answer'])
                self.assertIn(judge.kind, ('blip','qwen3_vl'))

    def test_variable_length_batches_match_independent_answers_in_order(self):
        from PIL import Image
        image = Image.new('RGB', (16,16), 'red')
        questions = ['cat ?', 'one ?', 'is there one cat ?', 'is there cat ?', 'is there cat ?']
        for name in ('blip', 'qwen'):
            with self.subTest(name=name):
                judge = QuestionJudge({'source':'local','name':str(self.root/name)}, 'cpu', batch_size=3)
                expected = [judge.answer(image, question) for question in questions]
                sizes, masks = [], []
                generate = judge.model.generate
                def observe(**kwargs):
                    sizes.append(kwargs['input_ids'].shape[0])
                    masks.append(kwargs['attention_mask'].tolist())
                    self.assertFalse(kwargs['generation_config'].do_sample)
                    self.assertEqual(kwargs['generation_config'].num_beams, 1)
                    self.assertEqual(kwargs['generation_config'].max_new_tokens, 16)
                    return generate(**kwargs)
                with patch.object(judge.model, 'generate', new=observe):
                    answers = judge.answer_many(image, questions)
                self.assertEqual(answers, expected)
                if name == 'qwen':
                    self.assertNotEqual(expected[0], expected[2])
                    self.assertEqual(sizes, [3, 2])
                    self.assertEqual(masks[0][0][0], 0)
                    self.assertTrue(all(row[-1] == 1 for mask in masks for row in mask))
                else:
                    # BLIP's answer decoder does not mask padded question
                    # embeddings: only adjacent equal lengths may share a batch.
                    self.assertEqual(sizes, [2, 1, 2])
                    self.assertTrue(all(all(row) for mask in masks for row in mask))

    def test_cpu_default_remains_one_question_per_inference(self):
        from PIL import Image
        judge = QuestionJudge({'source':'local','name':str(self.root/'qwen')}, 'cpu')
        sizes = []
        generate = judge.model.generate
        def observe(**kwargs):
            sizes.append(kwargs['input_ids'].shape[0])
            return generate(**kwargs)
        with patch.object(judge.model, 'generate', new=observe):
            answers = judge.answer_many(Image.new('RGB',(16,16)), ['cat ?', 'is there one cat ?'])
        self.assertEqual(len(answers), 2)
        self.assertEqual(sizes, [1, 1])

    def test_long_questions_fail_before_inference_instead_of_truncating(self):
        from PIL import Image
        for name in ('blip', 'qwen'):
            with self.subTest(name=name):
                judge = QuestionJudge({'source':'local','name':str(self.root/name)}, 'cpu', batch_size=3)
                with patch.object(judge.model,'generate',wraps=judge.model.generate) as generate:
                    with self.assertRaisesRegex(ValueError,'token limit'):
                        judge.answer_many(Image.new('RGB',(16,16)), ['cat ?', 'cat '*300])
                generate.assert_not_called()

    def test_total_token_budget_bounds_real_batches(self):
        from PIL import Image
        judge = QuestionJudge({'source':'local','name':str(self.root/'qwen')}, 'cpu',
                              batch_size=4, batch_token_limit=100)
        sizes = []
        generate = judge.model.generate
        def observe(**kwargs):
            sizes.append(kwargs['input_ids'].shape[0])
            self.assertLessEqual(kwargs['input_ids'].numel() + 16 * sizes[-1], 100)
            return generate(**kwargs)
        with patch.object(judge.model, 'generate', new=observe):
            answers = judge.answer_many(Image.new('RGB',(16,16)), ['cat ?'] * 4)
        self.assertEqual(len(answers), 4)
        self.assertEqual(sizes, [2, 2])

    def test_free_cuda_memory_limits_prepared_batches(self):
        from PIL import Image
        judge = QuestionJudge({'source':'local','name':str(self.root/'qwen')}, 'cpu', batch_size=4)
        prepared = judge._prepare(Image.new('RGB',(16,16)), ['cat ?'] * 4)
        # The memory query is the external boundary. Inputs and model config
        # are real CPU fixtures; this test never initializes or touches CUDA.
        judge.device = 'cuda'
        for free_mib, expected in ((4096, 4), (512, 1), (50, 1)):
            with self.subTest(free_mib=free_mib), patch.object(judge.torch.cuda, 'mem_get_info',
                    return_value=(free_mib * 1024**2, 16 * 1024**3)):
                self.assertEqual(judge._batch_count(prepared), expected)

    def test_cancellation_is_checked_before_and_after_every_chunk(self):
        from PIL import Image
        image = Image.new('RGB',(16,16))
        for question_count in (2, 5):
            with self.subTest(question_count=question_count):
                judge = QuestionJudge({'source':'local','name':str(self.root/'qwen')}, 'cpu', batch_size=2)
                cancelled = [True]
                calls = []
                generate = judge.model.generate
                def cancel_after_inference(**kwargs):
                    calls.append(kwargs['input_ids'].shape[0])
                    output = generate(**kwargs)
                    cancelled[0] = True
                    return output
                with patch.object(judge.model, 'generate', new=cancel_after_inference):
                    with self.assertRaises(InterruptedError):
                        judge.answer_many(image, ['cat ?'] * question_count, cancelled=lambda: cancelled[0])
                    self.assertEqual(calls, [])
                    cancelled[0] = False
                    with self.assertRaises(InterruptedError):
                        judge.answer_many(image, ['cat ?'] * question_count, cancelled=lambda: cancelled[0])
                self.assertEqual(calls, [2])

    def test_invalid_cancellation_context_fails_before_inference(self):
        from PIL import Image
        judge = QuestionJudge({'source':'local','name':str(self.root/'qwen')}, 'cpu', batch_size=2)
        for cancelled in (True, False, lambda: None, lambda: 1, lambda: 'no'):
            with self.subTest(cancelled=cancelled), self.assertRaises(ValueError):
                judge.answer_many(Image.new('RGB',(16,16)), ['cat ?'], cancelled=cancelled)

    def test_cuda_oom_retries_pending_questions_and_remembers_smaller_batches(self):
        from PIL import Image
        image = Image.new('RGB',(16,16), 'red')
        judge = QuestionJudge({'source':'local','name':str(self.root/'qwen')}, 'cpu', batch_size=4)
        questions = ['cat ?', 'is there one cat ?', 'one ?', 'is there cat ?', 'is there one cat ?']
        expected = [judge.answer(image, question) for question in questions]
        calls, failed_tensors, cache_releases = [], [], []
        generate = judge.model.generate
        def limited_memory(**kwargs):
            count = kwargs['input_ids'].shape[0]
            calls.append(count)
            output = generate(**kwargs)
            if count > 2:
                failed_tensors.extend([weakref.ref(kwargs['input_ids']), weakref.ref(output)])
                raise judge.torch.cuda.OutOfMemoryError('injected CUDA memory pressure after real inference')
            return output
        def release_cache():
            gc.collect()
            self.assertIsNone(sys.exc_info()[0])
            self.assertTrue(all(reference() is None for reference in failed_tensors))
            cache_releases.append(True)
        with patch.object(judge.model, 'generate', new=limited_memory), \
             patch.object(judge.torch.cuda, 'empty_cache', new=release_cache):
            self.assertEqual(judge.answer_many(image, questions), expected)
            self.assertEqual(calls, [4, 2, 2, 1])
            self.assertEqual(judge.batch_size, 2)
            self.assertEqual(cache_releases, [True])
            calls.clear()
            self.assertEqual(judge.answer_many(image, questions), expected)
            self.assertEqual(calls, [2, 2, 1])

    def test_cancellation_after_oom_prevents_retry(self):
        from PIL import Image
        judge = QuestionJudge({'source':'local','name':str(self.root/'qwen')}, 'cpu', batch_size=4)
        cancelled, calls = [False], []
        generate = judge.model.generate
        def fail_and_cancel(**kwargs):
            calls.append(kwargs['input_ids'].shape[0])
            generate(**kwargs)
            cancelled[0] = True
            raise judge.torch.cuda.OutOfMemoryError('injected CUDA memory pressure')
        with patch.object(judge.model, 'generate', new=fail_and_cancel):
            with self.assertRaises(InterruptedError):
                judge.answer_many(Image.new('RGB',(16,16)), ['cat ?'] * 4, cancelled=lambda: cancelled[0])
        self.assertEqual(calls, [4])

    def test_cuda_oom_at_one_question_is_propagated(self):
        from PIL import Image
        judge = QuestionJudge({'source':'local','name':str(self.root/'qwen')}, 'cpu', batch_size=2)
        calls = []
        generate = judge.model.generate
        def fail(**kwargs):
            calls.append(kwargs['input_ids'].shape[0])
            generate(**kwargs)
            raise judge.torch.cuda.OutOfMemoryError('injected persistent CUDA memory pressure')
        with patch.object(judge.model, 'generate', new=fail):
            with self.assertRaises(judge.torch.cuda.OutOfMemoryError):
                judge.answer_many(Image.new('RGB',(16,16)), ['cat ?'] * 2)
        self.assertEqual(calls, [2, 1])
        self.assertEqual(judge.batch_size, 1)

    def test_invalid_generation_count_or_empty_answer_is_never_scored(self):
        from PIL import Image
        for mode in ('missing', 'extra', 'empty_count', 'empty_answer'):
            with self.subTest(mode=mode):
                judge = QuestionJudge({'source':'local','name':str(self.root/'qwen')}, 'cpu', batch_size=2)
                generate = judge.model.generate
                changed = [False]
                def corrupt(**kwargs):
                    output = generate(**kwargs)
                    if changed[0]:
                        return output
                    changed[0] = True
                    if mode == 'missing':
                        return output[:-1]
                    if mode == 'extra':
                        return judge.torch.cat([output, output[:1]], dim=0)
                    if mode == 'empty_count':
                        return output[:0]
                    output[:, kwargs['input_ids'].shape[-1]:] = judge.processor.tokenizer.eos_token_id
                    return output
                with patch.object(judge.model, 'generate', new=corrupt), self.assertRaises(ValueError):
                    judge.answer_many(Image.new('RGB',(16,16)), ['cat ?', 'is there one cat ?'])


if __name__ == '__main__':
    unittest.main()
