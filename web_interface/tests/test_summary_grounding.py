"""Deterministic checks of NLI evidence, sensitivity, and incomplete coverage."""
import importlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import BatchEncoding, BertConfig, BertForSequenceClassification

from test_text_encoders import VOCAB, save_tokenizer


class CharacterTokenizer:
    """Preserve whitespace so the test can reproduce format-sensitive NLI."""

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(char) + 3 for char in text]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return ''.join(chr(int(value) - 3) for value in ids if value > 3)

    def __call__(self, premises, claims, *, return_tensors='pt', padding=False,
                 truncation=False, **kwargs):
        if truncation:
            raise AssertionError('Grounding must not silently truncate pairs.')
        if isinstance(premises, str):
            premises, claims = [premises], [claims]
        rows = [[1] + self.encode(premise) + [2] + self.encode(claim) + [2]
                for premise, claim in zip(premises, claims)]
        width = max(map(len, rows))
        return BatchEncoding({
            'input_ids': torch.tensor([row + [0] * (width - len(row)) for row in rows]),
            'attention_mask': torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
        })


class DeterministicNLI:
    """Only replaces expensive model inference; the scorer runs unchanged."""

    def __init__(self, probability):
        self.probability = probability
        self.batch_sizes = []

    def __call__(self, input_ids, attention_mask):
        self.batch_sizes.append(len(input_ids))
        logits = []
        for row in input_ids.tolist():
            if 2 not in row:
                raise AssertionError('A valid pair lost its content before model inference.')
            split = row.index(2)
            premise = ''.join(chr(value - 3) for value in row[1:split])
            claim = ''.join(chr(value - 3) for value in row[split + 1:] if value > 3)
            support = self.probability(premise, claim)
            logits.append([math.log((1 - support) / 2), math.log(support),
                           math.log((1 - support) / 2)])
        return SimpleNamespace(logits=torch.tensor(logits))


class GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def scorer(self, probability=lambda premise, claim: .8, limit=128):
        # A missing new module is an explicit red test, not a collection error.
        try:
            cls = importlib.import_module('summary_grounding').GroundedScorer
        except ModuleNotFoundError:
            self.fail('GroundedScorer assessment implementation is missing.')
        scorer = cls.__new__(cls)
        scorer.torch = torch
        scorer.tokenizer = CharacterTokenizer()
        scorer.model = DeterministicNLI(probability)
        scorer.entailment_id = 1
        scorer.limit = limit
        scorer.device = 'cpu'
        scorer.identity = {'name': 'deterministic-test', 'revision': 'test'}
        return scorer

    def codes(self, assessment):
        return {warning['code'] for warning in assessment['warnings']}

    def test_stable_support_keeps_strongest_raw_evidence_for_each_claim(self):
        scorer = self.scorer(lambda premise, claim: .9 if claim[:-1] in premise else .1, limit=64)
        result = scorer.assess('A cat rests. ' + 'x ' * 20 + 'A dog runs.',
                               'A cat rests. A dog runs.')
        self.assertTrue(result['reliable'])
        self.assertAlmostEqual(result['source_support'], .9, places=6)
        self.assertEqual(result['format_delta'], 0.)
        self.assertEqual(len(result['claims']), 2)
        for claim in result['claims']:
            self.assertIn(claim['text'][:-1], claim['evidence'])
            self.assertAlmostEqual(claim['support'], .9, places=6)
        json.dumps(result, allow_nan=False)

    def test_whitespace_instability_preserves_raw_support_and_marks_uncertainty(self):
        scorer = self.scorer(lambda premise, claim: .08 if '\n' in premise else .93, limit=256)
        result = scorer.assess('Riley: Chloe is on TV.\nJames: Which channel?', 'James asks.')
        self.assertFalse(result['reliable'])
        self.assertAlmostEqual(result['source_support'], .08, places=6)
        self.assertAlmostEqual(result['collapsed_source_support'], .93, places=6)
        self.assertAlmostEqual(result['format_delta'], .85, places=6)
        self.assertIn('format_sensitive', self.codes(result))
        self.assertIn('\n', result['claims'][0]['evidence'])
        self.assertAlmostEqual(scorer.score('Riley: Chloe is on TV.\nJames: Which channel?',
                                            'James asks.'), .08, places=6)

    def test_opposing_claim_deltas_cannot_cancel_reliability_failure(self):
        def probability(premise, claim):
            return (.1 if '\n' in premise else .9) if claim == 'Cat.' else (.9 if '\n' in premise else .1)
        result = self.scorer(probability).assess('Cat.\nDog.', 'Cat. Dog.')
        self.assertAlmostEqual(result['format_delta'], 0., places=6)
        self.assertAlmostEqual(result['max_claim_format_delta'], .8, places=6)
        self.assertFalse(result['reliable'])

    def test_small_format_change_retains_raw_score(self):
        result = self.scorer(lambda premise, claim: .4 if '\n' in premise else .5).assess(
            'Cat.\nDog.', 'Cat.')
        self.assertTrue(result['reliable'])
        self.assertAlmostEqual(result['source_support'], .4, places=6)
        self.assertAlmostEqual(result['format_delta'], .1, places=6)

    def test_batched_pair_inference_is_bounded_and_matches_evidence(self):
        scorer = self.scorer(lambda premise, claim: .7, limit=48)
        result = scorer.assess('cat dog ' * 14, 'Cat. Dog. Bird.')
        self.assertAlmostEqual(result['source_support'], .7, places=6)
        self.assertGreater(max(scorer.model.batch_sizes), 1)
        self.assertLessEqual(max(scorer.model.batch_sizes), 16)
        self.assertLess(len(scorer.model.batch_sizes), result['coverage']['pairs_scored'])

    def test_changed_window_content_is_reported_even_when_support_stays_stable(self):
        result = self.scorer(limit=48).assess('cat\n\n' * 15, 'Cat.')
        self.assertFalse(result['reliable'])
        self.assertIn('format_windowing_changed', self.codes(result))
        self.assertTrue(result['coverage']['format_windowing_changed'])

    def test_long_claim_is_retained_unscored_and_does_not_inflate_mean(self):
        scorer = self.scorer(limit=48)
        result = scorer.assess('Cat.', 'Cat. ' + 'x' * 30 + '.')
        self.assertFalse(result['reliable'])
        self.assertIsNone(result['source_support'])
        self.assertEqual(len(result['claims']), 2)
        self.assertAlmostEqual(result['claims'][0]['support'], .8, places=6)
        self.assertIsNone(result['claims'][1]['support'])
        self.assertIn('claim_too_long', self.codes(result))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            scorer.score('Cat.', 'Cat. ' + 'x' * 30 + '.')

    def test_claim_budget_retains_all_claims_with_unscored_tail(self):
        result = self.scorer().assess('Cat.', ' '.join('Cat.' for _ in range(35)))
        self.assertEqual(len(result['claims']), 35)
        self.assertIsNone(result['claims'][-1]['support'])
        self.assertIsNone(result['source_support'])
        self.assertIn('claim_budget_exceeded', self.codes(result))

    def test_source_window_budget_marks_partial_evidence_and_no_aggregate(self):
        result = self.scorer(limit=48).assess('cat ' * 200, 'Cat.')
        self.assertFalse(result['reliable'])
        self.assertIsNone(result['source_support'])
        self.assertTrue(result['coverage']['source_truncated'])
        self.assertLessEqual(result['coverage']['raw_windows_scored'], 32)
        self.assertIn('source_window_budget_exceeded', self.codes(result))
        self.assertEqual(result['claims'][0]['status'], 'partial')

    def test_empty_source_and_summary_are_explicit(self):
        scorer = self.scorer()
        empty_source = scorer.assess('  ', 'Cat.')
        self.assertFalse(empty_source['reliable'])
        self.assertIsNone(empty_source['source_support'])
        self.assertIn('empty_source', self.codes(empty_source))
        empty_summary = scorer.assess('Cat.', ' \n ')
        self.assertEqual(empty_summary['source_support'], 0.)
        self.assertEqual(empty_summary['claims'], [])
        self.assertEqual(scorer.score('Cat.', ''), 0.)
        self.assertEqual(scorer.model.batch_sizes, [])

    def test_oversized_pair_is_unscored_without_losing_left_padded_valid_pair(self):
        class OverflowTokenizer(CharacterTokenizer):
            padding_side = 'left'

            def __call__(self, premises, claims, **kwargs):
                expanded = [claim + ('x' * 100 if claim == 'Long.' else '') for claim in claims]
                inputs = super().__call__(premises, expanded, **kwargs)
                for index, mask in enumerate(inputs['attention_mask']):
                    length = int(mask.sum())
                    row = inputs['input_ids'][index, :length].clone()
                    inputs['input_ids'][index].zero_()
                    inputs['input_ids'][index, -length:] = row
                    inputs['attention_mask'][index].zero_()
                    inputs['attention_mask'][index, -length:] = 1
                return inputs

        scorer = self.scorer(limit=48)
        scorer.tokenizer = OverflowTokenizer()
        result = scorer.assess('Cat.', 'Cat. Long.')
        self.assertIsNone(result['source_support'])
        self.assertFalse(result['reliable'])
        self.assertAlmostEqual(result['claims'][0]['support'], .8, places=6)
        self.assertIsNone(result['claims'][1]['support'])
        self.assertIn('pair_input_limit', self.codes(result))
        self.assertEqual(result['coverage']['pairs_unscored'], 1)

    def test_invalid_logits_are_rejected_instead_of_becoming_high_support(self):
        scorer = self.scorer()
        for logits in (torch.tensor([[0., float('nan'), 0.]]),
                       torch.tensor([[0., float('inf'), 0.]]),
                       torch.tensor([[0., 1.]])):
            with self.subTest(logits=logits):
                scorer.model = lambda **kwargs: SimpleNamespace(logits=logits)
                with self.assertRaisesRegex(ValueError, 'invalid|nonfinite'):
                    scorer.assess('Cat.', 'Cat.')

    def test_cuda_oom_retries_same_pairs_at_smaller_batches_without_losing_coverage(self):
        probability = lambda premise, claim: {'Cat.': .3, 'Dog.': .6, 'Bird.': .9}[claim]
        source = ' '.join(f'word{index}' for index in range(28))
        summary = 'Cat. Dog. Bird.'
        expected = self.scorer(probability, limit=48).assess(source, summary)

        class OneOOM(DeterministicNLI):
            def __init__(self, fail_on):
                super().__init__(probability)
                self.attempts = []
                self.fail_on = fail_on

            def __call__(self, input_ids, attention_mask):
                self.attempts.append(len(input_ids))
                if len(self.attempts) == self.fail_on:
                    raise torch.cuda.OutOfMemoryError('simulated CUDA OOM')
                return super().__call__(input_ids, attention_mask)

        for fail_on in (1, 2):
            with self.subTest(fail_on=fail_on):
                scorer = self.scorer(probability, limit=48)
                scorer.model = OneOOM(fail_on)
                try:
                    actual = scorer.assess(source, summary)
                except torch.cuda.OutOfMemoryError:
                    self.fail('A batched CUDA OOM must retry without dropping any pairs.')
                self.assertEqual(actual, expected)
                self.assertEqual(scorer.model.attempts[fail_on - 1:fail_on + 1], [16, 8])

    def test_cuda_oom_at_one_pair_fails_with_clear_diagnostic(self):
        scorer = self.scorer()

        def out_of_memory(**kwargs):
            raise torch.cuda.OutOfMemoryError('simulated CUDA OOM')

        scorer.model = out_of_memory
        with self.assertRaisesRegex(torch.cuda.OutOfMemoryError, 'single pair'):
            scorer.assess('Cat.', 'Cat.')

    def test_unrelated_runtime_errors_are_not_retried_as_cuda_oom(self):
        scorer = self.scorer()
        attempts = []

        def unrelated_error(**kwargs):
            attempts.append(True)
            raise RuntimeError('CUDA out of memory text in an unrelated RuntimeError')

        scorer.model = unrelated_error
        with self.assertRaisesRegex(RuntimeError, 'unrelated RuntimeError'):
            scorer.assess('Cat. Dog.', 'Cat. Dog.')
        self.assertEqual(len(attempts), 1)

    def test_constructor_records_policy_and_loads_local_checkpoint(self):
        cls = type(self.scorer())
        with tempfile.TemporaryDirectory() as directory:
            config = BertConfig(vocab_size=len(VOCAB), hidden_size=12, intermediate_size=24,
                                num_hidden_layers=1, num_attention_heads=3,
                                max_position_embeddings=128, num_labels=3,
                                id2label={0: 'contradiction', 1: 'entailment', 2: 'neutral'})
            model = BertForSequenceClassification(config)
            with torch.no_grad():
                model.classifier.weight.zero_()
                model.classifier.bias.copy_(torch.tensor([-4., 4., -4.]))
            model.save_pretrained(directory)
            save_tokenizer(Path(directory), limit=128)
            scorer = cls({'source': 'local', 'name': directory}, 'cpu')
            result = scorer.assess('a cat in the garden', 'a cat')
            self.assertGreater(result['source_support'], .99)
            self.assertTrue(result['reliable'])
            self.assertTrue(result['identity']['scorer_revision'])
            self.assertTrue(result['identity']['policy_revision'])


if __name__ == '__main__':
    unittest.main()
