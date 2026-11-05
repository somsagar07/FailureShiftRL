"""Held-out generation evaluation uses CPU-only inference boundary fixtures."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generation_failure import combined_score


class PanelEvaluator:
    """Replace GPU/download boundaries; keep the real combined scoring formula."""
    def __init__(self, directory, *, fail=False, cancel=None, severity=.5, scorer='fixed'):
        self.directory = Path(directory)
        self.closed = False
        self.fail, self.cancel, self.severity = fail, cancel, severity
        self.scorer = scorer
        self.ready = False

    def set_runtime_callbacks(self, **callbacks):
        self.callbacks = callbacks
        self.ready = True

    def describe(self):
        return {'task': 'generation', 'revision': 'diffusion-revision',
                'criterion': 'combined generation failure review signal',
                'failure_threshold': .5, 'generation_precision': 'float32',
                'scorer': {'name': self.scorer, 'revision': 'pinned' if self.ready else None}}

    def predict(self, value, *, seed, artifact_name, checks, reference_prompt, action):
        if self.fail:
            raise ValueError('Scorer failed')
        if self.cancel is not None:
            self.cancel['value'] = True
        filename = artifact_name + '.png'
        (self.directory / filename).write_bytes(b'CPU fixture image')
        return {'prompt': value, 'seed': seed, 'image': filename, 'quality': 1-self.severity,
                'failure_scores': {'content': self.severity, 'instructions': None if action is None else .1,
                                   'visual': .2, 'bias': None}}

    def score(self, output, baseline):
        return combined_score(output, baseline)

    def close(self):
        self.closed = True


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.evaluators = []
        self.actions = [{'id': 4, 'style': 'photograph'}]
        self.records = [{'prompt': 'Two red birds', 'checks': [['How many birds?', 'two']]},
                        {'prompt': 'A blue dog'}]
        self.config = {'task': 'generation', 'seed': 4294967295, 'device': 'cpu',
                       'generation_steps': 5, 'generation_size': 64,
                       'generation_precision': 'float32', 'validation_phase': 'before'}

    def factory(self, model, config, directory, progress=None):
        evaluator = PanelEvaluator(directory, **getattr(self, 'options', {}))
        self.evaluators.append(evaluator)
        return evaluator

    def evaluate(self, **kwargs):
        import generation_validation
        self.assertTrue(hasattr(generation_validation, 'evaluate_generation'),
                        'Generation panels are not implemented')
        arguments = dict(model={'source': 'local', 'name': '/base'}, config=self.config,
                         records=self.records, actions=self.actions, directory=self.directory,
                         progress=lambda **fields: None, cancelled=lambda: False,
                         evaluator_factory=self.factory)
        arguments.update(kwargs)
        return generation_validation.evaluate_generation(**arguments)

    def test_evaluates_originals_and_actions_with_identical_per_scene_seeds(self):
        panel = self.evaluate(expected_scorer={'name': 'fixed', 'revision': 'pinned'})
        pairs = panel['pairs']
        self.assertEqual([(row['input_index'], row['action_id'], row['seed']) for row in pairs],
                         [(0, None, 4294967295), (0, 4, 4294967295), (1, None, 0), (1, 4, 0)])
        self.assertEqual(pairs[1]['prompt'], 'Photograph style. Scene: Two red birds')
        self.assertEqual(pairs[1]['reference_prompt'], 'Two red birds')
        self.assertEqual(pairs[1]['checks'], [{'question': 'How many birds?', 'answer': 'two'}])
        self.assertEqual(pairs[1]['score']['severity'], .5)
        self.assertEqual(panel['settings']['generation_steps'], 5)
        self.assertEqual(panel['scorer'], {'name': 'fixed', 'revision': 'pinned'})
        self.assertTrue(self.evaluators[0].closed)
        self.assertEqual(len({row['output']['image'] for row in pairs}), 4)
        self.assertTrue(all((self.directory / row['output']['image']).is_file() for row in pairs))
        json.dumps(panel, allow_nan=False)

    def test_repeats_have_independent_but_matched_original_action_seeds(self):
        panel = self.evaluate(config={**self.config, 'validation_repeats': 2})
        self.assertEqual([row['seed'] for row in panel['pairs']],
                         [4294967295, 4294967295, 1, 1, 0, 0, 2, 2])
        self.assertEqual(len({row['output']['image'] for row in panel['pairs']}), 8)

    def test_runtime_failure_or_cancellation_closes_the_evaluator(self):
        self.options = {'fail': True}
        with self.assertRaisesRegex(ValueError, 'Scorer failed'):
            self.evaluate()
        self.assertTrue(self.evaluators[-1].closed)
        state = {'value': False}
        self.options = {'cancel': state}
        with self.assertRaises(InterruptedError):
            self.evaluate(cancelled=lambda: state['value'])
        self.assertTrue(self.evaluators[-1].closed)
        with self.assertRaises(InterruptedError):
            self.evaluate(cancelled=lambda: True)
        self.assertEqual(len(self.evaluators), 2, 'pre-cancelled panels never load weights')

    def test_rejects_changed_scorer_after_readiness_and_closes_resources(self):
        with self.assertRaisesRegex(ValueError, 'scorer'):
            self.evaluate(expected_scorer={'name': 'other', 'revision': 'pinned'})
        self.assertTrue(self.evaluators[-1].closed)
        self.assertFalse(list(self.directory.glob('*.png')))

    def test_invalid_bounds_fail_before_loading_models(self):
        for config in ({'validation_repeats': 0}, {'validation_repeats': 5},
                       {'seed': True}, {'validation_phase': '../unsafe'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.evaluate(config={**self.config, **config})
        self.assertEqual(self.evaluators, [])

    def test_panel_size_is_bounded_before_loading_weights(self):
        records = [{'prompt': f'Scene {index}'} for index in range(64)]
        actions = [{'id': index, 'style': 'photograph'} for index in range(64)]
        with self.assertRaisesRegex(ValueError, '4096'):
            self.evaluate(records=records, actions=actions)
        self.assertEqual(self.evaluators, [])

    def test_cancellation_from_progress_stops_without_loading_next_image(self):
        state = {'cancelled': False}
        def progress(**fields):
            if fields.get('validation_completed') == 1:
                state['cancelled'] = True
        with self.assertRaises(InterruptedError):
            self.evaluate(progress=progress, cancelled=lambda: state['cancelled'])
        self.assertEqual(len(list(self.directory.glob('*.png'))), 1)
        self.assertTrue(self.evaluators[-1].closed)

    def test_before_after_artifact_names_remain_distinct_in_one_job_directory(self):
        before = self.evaluate()
        after = self.evaluate(config={**self.config, 'validation_phase': 'after'})
        self.assertEqual(before['settings'], after['settings'])
        self.assertEqual(len({row['output']['image'] for panel in (before, after) for row in panel['pairs']}), 8)

    def test_retained_previews_are_bounded_without_dropping_any_measurement(self):
        records = [{'prompt': f'A distinct scene {index}'} for index in range(40)]
        panel = self.evaluate(records=records)
        self.assertEqual(len(panel['pairs']), 80)
        self.assertEqual(len(list(self.directory.glob('*.png'))), 64)
        self.assertEqual(panel['artifacts']['retained_images'], 64)
        self.assertEqual(panel['artifacts']['evaluated_images'], 80)
        self.assertTrue(all(row['output']['image_retained'] for row in panel['pairs'][:64]))
        self.assertTrue(all(not row['output']['image_retained'] and 'image' not in row['output']
                            for row in panel['pairs'][64:]))
        self.assertTrue(all('severity' in row['score'] for row in panel['pairs']))

    def test_preview_byte_budget_keeps_all_scores_and_does_not_leave_broken_image_links(self):
        with patch('generation_validation.MAX_PREVIEW_BYTES', len(b'CPU fixture image')):
            panel = self.evaluate()
        self.assertEqual(len(panel['pairs']), 4)
        self.assertEqual(panel['artifacts']['retained_images'], 1)
        self.assertEqual(panel['artifacts']['retained_bytes'], len(b'CPU fixture image'))
        self.assertEqual(len(list(self.directory.glob('*.png'))), 1)
        self.assertTrue(all('image' not in row['output'] for row in panel['pairs'][1:]))


class ValidationDataTests(unittest.TestCase):
    def validator(self):
        self.assertIsNotNone(importlib.util.find_spec('generation_validation'),
                             'The held-out validation module is missing')
        from generation_validation import validate_validation_data
        return validate_validation_data

    def test_preserves_scene_text_and_normalizes_checks_without_mutating_input(self):
        records = [{'prompt': '  Two red birds.\n', 'checks': [['How many birds?', 'two']]}]
        original = copy.deepcopy(records)
        result = self.validator()(records, ['A blue dog'])
        self.assertEqual(result, [{'prompt': '  Two red birds.\n',
                                  'checks': [{'question': 'How many birds?', 'answer': 'two'}]}])
        self.assertEqual(records, original)

    def test_rejects_normalized_training_overlap_and_duplicate_holdouts(self):
        validate = self.validator()
        for prompt in ('Ａ RED bird', 'a  red\n BIRD', ' A red bird '):
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, 'overlap'):
                validate([{'prompt': prompt}], ['a red bird'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate([{'prompt': 'A cat'}, {'prompt': 'a\tcat'}], [])

    def test_rejects_empty_unbounded_or_malformed_records(self):
        validate = self.validator()
        for records in (None, [], 'text', [{'prompt': ''}], [{'prompt': ' \n'}],
                        [{'prompt': 'A cat', 'image_path': '/target.png'}],
                        [{'prompt': 'A cat', 'checks': [{'question': 'What?'}]}],
                        [{'prompt': f'Holdout {index}'} for index in range(65)]):
            with self.subTest(records=repr(records)[:60]), self.assertRaises(ValueError):
                validate(records, [])
        self.assertEqual(len(validate([{'prompt': f'Scene {i}'} for i in range(64)], [])), 64)


def fixture_panel(count=8, clean=.2, selected=.8, repeats=1):
    panel = {'status': 'completed', 'records': [{'prompt': f'Scene {i}', 'checks': []} for i in range(count)],
             'actions': [{'id': 4, 'style': 'photograph'}],
             'settings': {'seed': 42, 'validation_repeats': repeats, 'generation_steps': 5,
                          'generation_size': 64, 'generation_precision': 'float32'},
             'scorer': {'name': 'fixed', 'revision': 'pinned'}, 'pairs': []}
    for index in range(count):
        for repeat in range(repeats):
            for action_id, value in ((None, clean), (4, selected)):
                severity = value[index] if isinstance(value, list) else value
                panel['pairs'].append({'input_index': index, 'action_id': action_id, 'repeat': repeat,
                    'seed': 42+index+repeat*count, 'reference_prompt': f'Scene {index}', 'checks': [],
                    'prompt': f'Scene {index}' if action_id is None else f'Photograph style. Scene: Scene {index}',
                    'output': {'quality': 1-severity, 'failure_scores': {
                        'content': severity, 'instructions': None if action_id is None else severity,
                        'visual': severity, 'bias': None}},
                    'score': {'severity': severity, 'failure': severity >= .5}})
    return panel


class ComparisonTests(unittest.TestCase):
    def compare(self, before, after):
        import generation_validation
        self.assertTrue(hasattr(generation_validation, 'compare_generation'),
                        'Matched generation comparison is not implemented')
        return generation_validation.compare_generation(before, after)

    def test_reports_actual_paired_severity_components_flags_and_intervals(self):
        result = self.compare(fixture_panel(), fixture_panel(selected=.6))
        self.assertEqual(result['verdict'], 'improved')
        self.assertEqual(result['prompt_count'], 8)
        selected = result['selected']
        self.assertAlmostEqual(selected['before_mean_severity'], .8)
        self.assertAlmostEqual(selected['after_mean_severity'], .6)
        self.assertAlmostEqual(selected['mean_change'], -.2)
        for bound in selected['ci95']:
            self.assertAlmostEqual(bound, -.2)
        self.assertEqual(selected['before_failure_rate'], 1)
        self.assertEqual(selected['after_failure_rate'], 1)
        self.assertAlmostEqual(selected['components']['content']['mean_change'], -.2)
        self.assertIsNone(result['clean']['components']['instructions']['mean_change'])
        self.assertEqual(len(result['per_pair']), 16)
        self.assertEqual(len(result['per_prompt']), 8)
        json.dumps(result, allow_nan=False)

    def test_few_prompts_stay_inconclusive_even_with_many_seeds_or_actions(self):
        result = self.compare(fixture_panel(count=2, repeats=4),
                              fixture_panel(count=2, selected=.1, repeats=4))
        self.assertEqual(result['verdict'], 'inconclusive')
        self.assertEqual(result['prompt_count'], 2)
        self.assertIn('8 distinct', result['reason'])

    def test_clean_regression_blocks_selection_and_reports_a_measured_regression(self):
        result = self.compare(fixture_panel(), fixture_panel(clean=.4, selected=.6))
        self.assertEqual(result['verdict'], 'regressed')
        self.assertAlmostEqual(result['clean']['mean_change'], .2)
        self.assertGreater(result['clean']['ci95'][0], 0)

    def test_decimal_guardrail_boundaries_tolerate_float_roundoff(self):
        result = self.compare(fixture_panel(clean=.3, selected=.57),
                              fixture_panel(clean=.31, selected=.56))
        self.assertEqual(result['verdict'], 'improved')

    def test_flag_increase_blocks_improvement_even_when_mean_severity_improves(self):
        before = fixture_panel(selected=[.95, .49, .49, .49, .49, .49, .49, .49])
        after = fixture_panel(selected=[0, .5, .5, .5, .5, .5, .5, .5])
        self.assertEqual(self.compare(before, after)['verdict'], 'inconclusive')

    def test_bootstrap_resamples_prompts_and_is_deterministic(self):
        before = fixture_panel(selected=.5, repeats=4)
        after = fixture_panel(selected=[.1, .1, .1, .1, .9, .9, .9, .9], repeats=4)
        result = self.compare(before, after)
        self.assertEqual(result, self.compare(before, after))
        self.assertAlmostEqual(result['selected']['mean_change'], 0)
        self.assertLess(result['selected']['ci95'][0], -.15)
        self.assertGreater(result['selected']['ci95'][1], .15)
        self.assertEqual(result['verdict'], 'inconclusive')

    def test_rejects_mismatched_settings_scorers_pairs_and_incomplete_panels(self):
        before = fixture_panel()
        for change in ('settings', 'scorer', 'seed', 'prompt', 'checks', 'action_id', 'missing', 'duplicate'):
            after = copy.deepcopy(before)
            if change == 'settings':
                after['settings']['generation_steps'] = 6
            elif change == 'scorer':
                after['scorer']['revision'] = 'changed'
            elif change == 'missing':
                after['pairs'].pop()
            elif change == 'duplicate':
                after['pairs'][1] = copy.deepcopy(after['pairs'][0])
            else:
                after['pairs'][0][change] = 'changed'
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compare(before, after)

    def test_rejects_nonfinite_or_boolean_scores_and_changed_component_coverage(self):
        before = fixture_panel()
        for value in (float('nan'), float('inf'), True, -.1, 1.1):
            after = copy.deepcopy(before)
            after['pairs'][0]['score']['severity'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.compare(before, after)
        after = copy.deepcopy(before)
        after['pairs'][1]['output']['failure_scores']['instructions'] = None
        with self.assertRaises(ValueError):
            self.compare(before, after)


if __name__ == '__main__':
    unittest.main()
