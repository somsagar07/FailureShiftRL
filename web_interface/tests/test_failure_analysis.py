"""Failure evidence must distinguish labels, heuristics, and unmeasured claims."""
import copy
import json
import tempfile
import unittest
from types import SimpleNamespace

from discovery import DiscoveryManager, _public
from failure_analysis import input_analysis


class FailureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manager = DiscoveryManager(None, None, self.directory.name)

    def assess(self, task, *, failure=True, metrics=None, output=None, baseline=None, target=None):
        score = dict(reward=.7, severity=.8, failure=failure,
                     criterion='fixture review signal', metrics=metrics or {})
        evaluator = SimpleNamespace(config={'task': task}, score=lambda *a, **k: copy.deepcopy(score))
        result = self.manager._score(evaluator, output or {}, baseline or {}, target)
        self.assertIn('analysis', result, 'Each measurement needs an explicit evidence assessment')
        return result['analysis']

    def test_existing_label_error_is_not_called_a_new_action_failure(self):
        analysis = self.assess('classification', target='cat',
                               metrics={'error': True, 'baseline_error': True, 'target_label': 'cat'})
        self.assertEqual(analysis['status'], 'failure')
        self.assertIn('baseline_error', [item['code'] for item in analysis['reasons']])
        self.assertNotIn('new_label_error', [item['code'] for item in analysis['reasons']])

    def test_unlabelled_change_is_only_a_candidate(self):
        analysis = self.assess('classification', metrics={'error': True, 'baseline_error': False})
        self.assertEqual(analysis['status'], 'candidate')
        self.assertEqual(analysis['basis'], 'prediction_consistency')

    def test_changed_text_without_action_metadata_requires_target_review(self):
        for action in (None, {}, {'id': 9}):
            with self.subTest(action=action):
                analysis = input_analysis('summarization', 'Alice paid Bob.',
                                          'Bob paid Alice.', action)
                self.assertEqual(analysis['status'], 'review')
                self.assertIn('action_metadata_unavailable',
                              [item['code'] for item in analysis['warnings']])
        self.assertEqual(input_analysis('summarization', 'Alice paid Bob.',
                                       'Alice paid Bob.', None)['status'], 'unchanged')

    def test_visibility_risk_prevents_asserting_labelled_failure(self):
        output = {'input_analysis': {'status': 'review', 'warnings': [
            {'code': 'action_visibility', 'message': 'The crop may remove the labelled subject.'}]}}
        analysis = self.assess('classification', target='cat', output=output,
                               metrics={'error': True, 'baseline_error': False})
        self.assertEqual(analysis['status'], 'inconclusive')
        self.assertIn('action_visibility', [item['code'] for item in analysis['warnings']])

    def test_low_reference_overlap_is_not_a_verified_factual_error(self):
        analysis = self.assess('summarization', output={'summary': 'A valid paraphrase.'},
                               metrics={'rouge_l': .1, 'nonrepetition': 1., 'quality': .1})
        self.assertEqual(analysis['status'], 'candidate')
        self.assertIn('reference_overlap', [item['code'] for item in analysis['reasons']])

    def test_unstable_nli_is_inconclusive_even_when_current_flag_is_false(self):
        analysis = self.assess('summarization', failure=False, output={
            'summary': 'Mary is at the train station.',
            'grounding': {'reliable': False, 'format_delta': .85, 'warnings': [
                {'code': 'format_sensitive', 'message': 'Line breaks change the NLI score.'}]}})
        self.assertEqual(analysis['status'], 'inconclusive')
        self.assertIn('format_sensitive', [item['code'] for item in analysis['warnings']])

    def test_empty_summary_is_an_observed_output_failure(self):
        analysis = self.assess('summarization', output={'summary': '   '})
        self.assertEqual(analysis['status'], 'failure')
        self.assertIn('empty_output', [item['code'] for item in analysis['reasons']])

    def test_a_truncated_original_prevents_claiming_a_fair_quality_comparison(self):
        analysis = self.assess('summarization', failure=False,
            output={'summary': 'A short summary.', 'input_coverage': {'truncated': False}},
            baseline={'input_coverage': {'truncated': True}})
        self.assertEqual(analysis['status'], 'inconclusive')
        self.assertIn('baseline_input_truncated', [item['code'] for item in analysis['warnings']])

    def test_visual_judge_flags_are_candidates_and_unchecked_bias_is_visible(self):
        analysis = self.assess('generation', output={
            'seed': 42, 'failure_scores': {'content': .8, 'visual': .2, 'instructions': None, 'bias': None},
            'failure_evidence': {'content': {'status': 'evaluated'}, 'visual': {'status': 'evaluated'},
                                 'bias': {'status': 'not_evaluated'}}})
        self.assertEqual(analysis['status'], 'candidate')
        self.assertIn('content', [item['code'] for item in analysis['reasons']])
        self.assertIn('bias_not_evaluated', [item['code'] for item in analysis['warnings']])
        self.assertIn('single_seed', [item['code'] for item in analysis['warnings']])

    def test_public_reports_keep_judge_distributions_but_strip_classifier_vectors(self):
        report = {'output': {'class_id': 0, 'probabilities': [.9, .1]},
                  'failure_evidence': {'visual': {'readout': 'vev_expected_ordinal',
                                                 'probabilities': [.1, .2, .3, .2, .2]}},
                  'check_evidence': [{'method': 'vev_target_verification', 'probabilities': [.4, .6]}]}
        public = _public(report)
        self.assertNotIn('probabilities', public['output'])
        self.assertEqual(public['failure_evidence']['visual'].get('probabilities'), [.1, .2, .3, .2, .2])
        self.assertEqual(public['check_evidence'][0].get('probabilities'), [.4, .6])
        json.dumps(public, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
