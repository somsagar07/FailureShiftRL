import copy
import importlib
import importlib.util
import unittest


def matched_suite():
    return {
        'name': 'Matched drawing quality',
        'reference': 'equal_error_rates',
        'cases': [
            {'name': 'cup', 'groups': [
                {'name': 'group A', 'prompt': 'A person in a blue coat holding a cup.',
                 'checks': [['Is a person holding a cup?', 'Yes']]},
                {'name': 'group B', 'prompt': 'A person in a red coat holding a cup.',
                 'checks': [{'question': 'Is a person holding a cup?', 'answer': 'Yes',
                             'skill': 'relation'}]},
            ]},
            {'name': 'book', 'groups': [
                {'name': 'group B', 'prompt': 'A person in a red coat reading a book.',
                 'checks': [['Is a person reading a book?', 'Yes']]},
                {'name': 'group A', 'prompt': 'A person in a blue coat reading a book.',
                 'checks': [['Is a person reading a book?', 'Yes']]},
            ]},
        ],
    }


def structured_output(checks, probability=.8):
    """Replace model inference while using the production decision evidence."""
    from generation_failure import evaluate_requirements

    class Judge:
        supports_decisions = True

        def decide(self, image, questions, *, cancelled=None):
            return [dict(id=row['id'], type='noul',
                         probabilities=[1 - probability, probability], value=probability)
                    for row in questions]

    result = evaluate_requirements(Judge(), object(), checks)
    return dict(content_failure=1 - result['quality'], checks=result['checks'],
                check_evidence=result['check_evidence'])


class BiasSuiteValidationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('generation_bias'),
                             'Matched-suite validation must be implemented')
        self.normalize = importlib.import_module('generation_bias').normalize_bias_suite

    def test_missing_suite_stays_absent(self):
        self.assertIsNone(self.normalize(None))

    def test_normalization_preserves_explicit_names_prompts_and_independent_checks(self):
        source = matched_suite()
        source['name'] = ' Matched drawing quality '
        source['cases'][0]['groups'][0]['name'] = ' group A '
        source['cases'][0]['groups'][0]['prompt'] = '  A blue coat and {literal}.\n'
        normalized = self.normalize(source)
        self.assertEqual(normalized['name'], 'Matched drawing quality')
        self.assertEqual(normalized['cases'][0]['groups'][0], {
            'name': 'group A', 'prompt': '  A blue coat and {literal}.\n',
            'checks': [{'question': 'Is a person holding a cup?', 'answer': 'Yes'}],
        })
        self.assertEqual(normalized['cases'][0]['groups'][1]['checks'][0]['skill'], 'relation')
        self.assertEqual([g['name'] for g in normalized['cases'][1]['groups']],
                         ['group B', 'group A'])
        source['cases'][0]['groups'][0]['checks'][0][1] = 'No'
        self.assertEqual(normalized['cases'][0]['groups'][0]['checks'][0]['answer'], 'Yes')

    def test_schema_requires_all_fields_and_rejects_unknown_fields(self):
        for level in ('suite', 'case', 'group'):
            base = matched_suite()
            target = (base if level == 'suite' else base['cases'][0] if level == 'case'
                      else base['cases'][0]['groups'][0])
            for key in list(target):
                candidate = copy.deepcopy(base)
                changed = (candidate if level == 'suite' else candidate['cases'][0]
                           if level == 'case' else candidate['cases'][0]['groups'][0])
                del changed[key]
                with self.subTest(level=level, missing=key), self.assertRaises(ValueError):
                    self.normalize(candidate)
            target['extra'] = 'must not silently disappear'
            with self.subTest(level=level, unknown=True), self.assertRaises(ValueError):
                self.normalize(base)

    def test_reference_and_container_types_are_strict(self):
        for value in ('', 'null', False, 0, float('nan'), [], {'name': 'x'}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.normalize(value)
        for reference in ('', 'population_share', ' equal_error_rates ', None, True, float('inf')):
            source = matched_suite()
            source['reference'] = reference
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self.normalize(source)
        for path in ('cases', 'case', 'groups', 'group'):
            for value in (None, {}, '', True, float('nan')):
                source = matched_suite()
                if path == 'cases':
                    source['cases'] = value
                elif path == 'case':
                    source['cases'][0] = value
                elif path == 'groups':
                    source['cases'][0]['groups'] = value
                else:
                    source['cases'][0]['groups'][0] = value
                with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                    self.normalize(source)

    def test_empty_duplicate_or_inconsistent_groups_and_cases_are_rejected(self):
        examples = []
        source = matched_suite()
        source['cases'] = []
        examples.append(source)
        source = matched_suite()
        source['cases'][0]['groups'] = []
        examples.append(source)
        source = matched_suite()
        source['cases'][0]['groups'].pop()
        examples.append(source)
        source = matched_suite()
        source['cases'][0]['groups'][1]['name'] = ' group A '
        examples.append(source)
        source = matched_suite()
        source['cases'][1]['groups'][0]['name'] = 'unmatched C'
        examples.append(source)
        source = matched_suite()
        source['cases'][1]['name'] = ' cup '
        examples.append(source)
        for source in examples:
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.normalize(source)

    def test_name_prompt_and_check_validation_rejects_invalid_evidence(self):
        for level in ('suite', 'case', 'group'):
            for value in ('', '   ', 'x' * 81, 'bad\x00name', 'bad\nname', None,
                          7, True, float('nan')):
                source = matched_suite()
                target = (source if level == 'suite' else source['cases'][0] if level == 'case'
                          else source['cases'][0]['groups'][0])
                target['name'] = value
                with self.subTest(level=level, value=value), self.assertRaises(ValueError):
                    self.normalize(source)
        for field, values in (
            ('prompt', ['', ' ', None, True, float('inf'), 'x' * 10001, 'bad\x00prompt']),
            ('checks', [None, [], {}, '', [['q']], [['q', '']], [['q', float('nan')]],
                        [{'question': 'q', 'answer': 'a', 'extra': 'x'}], [['q', 'a']] * 129]),
        ):
            for value in values:
                source = matched_suite()
                source['cases'][0]['groups'][0][field] = value
                with self.subTest(field=field, value=repr(value)[:80]), self.assertRaises(ValueError):
                    self.normalize(source)

    def test_case_and_group_bounds_accept_limits_and_reject_excess(self):
        source = matched_suite()
        source['cases'] = [copy.deepcopy(source['cases'][0]) for _ in range(32)]
        for index, case in enumerate(source['cases']):
            case['name'] = f'case {index}'
            case['groups'] = [copy.deepcopy(case['groups'][0]) for _ in range(8)]
            for group_index, group in enumerate(case['groups']):
                group['name'] = f'group {group_index}'
        self.assertEqual(len(self.normalize(source)['cases']), 32)
        source['cases'].append({**copy.deepcopy(source['cases'][0]), 'name': 'extra'})
        with self.assertRaises(ValueError):
            self.normalize(source)
        source['cases'] = source['cases'][:1]
        source['cases'][0]['groups'].append({
            **copy.deepcopy(source['cases'][0]['groups'][0]), 'name': 'extra'})
        with self.assertRaises(ValueError):
            self.normalize(source)


class BiasProbeTests(unittest.TestCase):
    def setUp(self):
        self.Probe = getattr(importlib.import_module('generation_bias'), 'BiasProbe', None)
        self.assertTrue(callable(self.Probe), 'Repeated matched comparisons must be implemented')

    @staticmethod
    def constant_predict(value, *, checks, reference_prompt, seed, artifact_name):
        return {'content_failure': 0.25}

    def test_missing_suite_is_not_evaluated_and_does_not_call_predictor(self):
        def unexpected(*args, **kwargs):
            self.fail('An absent suite must not run inference')

        result = self.Probe(None, repeats=4, seed=8).evaluate(None, unexpected)
        self.assertIsNone(result['score'])
        self.assertEqual(result['status'], 'not_evaluated')
        self.assertTrue(result['reason'])
        self.assertEqual(result['scope'], 'matched_suite_action')
        self.assertEqual(result['repeats'], 4)
        self.assertEqual(result['seed'], 8)
        self.assertEqual(result['group_rates'], {})
        self.assertEqual(result['group_sample_counts'], {})
        self.assertEqual(result['cases'], [])
        self.assertEqual(result['sample_count'], 0)

    def test_repeats_and_seed_are_strictly_bounded_integers_even_without_suite(self):
        for repeats in (None, True, False, 1, 17, 2.0, '3', float('nan'), float('inf')):
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                self.Probe(None, repeats=repeats)
        for seed in (None, True, False, -1, 2**32, 42.0, '42', float('nan'), float('inf')):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.Probe(None, seed=seed)
        for repeats, seed in ((2, 0), (16, 2**32 - 1)):
            result = self.Probe(None, repeats=repeats, seed=seed).evaluate(None, self.constant_predict)
            self.assertEqual((result['repeats'], result['seed']), (repeats, seed))

    def test_hand_computed_mean_gap_retains_all_observations_and_counts(self):
        outcomes = iter([0, .75, .5, .25, 1, .25, .5, .25])

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            return {'content_failure': next(outcomes), 'image': object()}

        result = self.Probe(matched_suite(), repeats=2, seed=19).evaluate(None, predict)
        self.assertEqual(result['score'], .375)
        self.assertEqual(result['group_rates'], {'group A': .25, 'group B': .625})
        self.assertEqual(result['group_sample_counts'], {'group A': 4, 'group B': 4})
        self.assertEqual((result['case_count'], result['group_count'], result['sample_count']), (2, 2, 8))
        self.assertEqual(result['status'], 'evaluated')
        self.assertEqual(result['name'], 'quality_disparity')
        self.assertEqual(result['method'], 'max_group_mean_content_failure_gap')
        self.assertEqual(result['reference'], 'equal_error_rates')
        self.assertEqual(result['scope'], 'matched_suite_action')
        self.assertEqual(result['suite_name'], 'Matched drawing quality')
        self.assertEqual(result['cases'][0]['group_rates'], {'group A': .25, 'group B': .5})
        self.assertEqual(result['cases'][1]['group_rates'], {'group A': .25, 'group B': .75})
        self.assertEqual(result['cases'][0]['samples'][0], {
            'repeat': 0, 'seed': 19, 'groups': [
                {'name': 'group A', 'prompt': 'A person in a blue coat holding a cup.',
                 'transformed_prompt': 'A person in a blue coat holding a cup.',
                 'checks': [{'question': 'Is a person holding a cup?', 'answer': 'Yes'}],
                 'content_failure': 0.0},
                {'name': 'group B', 'prompt': 'A person in a red coat holding a cup.',
                 'transformed_prompt': 'A person in a red coat holding a cup.',
                 'checks': [{'question': 'Is a person holding a cup?', 'answer': 'Yes',
                             'skill': 'relation'}],
                 'content_failure': .75},
            ],
        })
        self.assertNotIn('image', result['cases'][0]['samples'][0]['groups'][0])

    def test_group_means_have_equal_case_and_repeat_weights_instead_of_mean_case_gaps(self):
        source = matched_suite()
        source['cases'][0]['groups'][0]['checks'] *= 3
        outcomes = iter([0, 1, 0, 1, 0, 1, 0, 1])

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            return {'content_failure': next(outcomes)}

        result = self.Probe(source, repeats=2).evaluate(None, predict)
        self.assertEqual(result['group_rates'], {'group A': .5, 'group B': .5})
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['cases'][0]['group_rates'], {'group A': 0, 'group B': 1})
        self.assertEqual(result['cases'][1]['group_rates'], {'group A': 1, 'group B': 0})

    def test_all_groups_share_case_repeat_seeds_with_unsigned_wraparound(self):
        seen = []

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            seen.append((seed, artifact_name))
            return {'content_failure': 0}

        result = self.Probe(matched_suite(), repeats=2, seed=2**32 - 1).evaluate(None, predict)
        self.assertEqual([seed for seed, _ in seen],
                         [4294967295, 4294967295, 0, 0, 1000002, 1000002, 1000003, 1000003])
        self.assertEqual([[sample['seed'] for sample in case['samples']] for case in result['cases']],
                         [[4294967295, 0], [1000002, 1000003]])
        self.assertEqual(len({name for _, name in seen}), 8)
        self.assertTrue(all(name.startswith('bias-') for _, name in seen))

    def test_complete_action_transforms_each_prompt_and_preserves_reference_checks(self):
        received = []

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            received.append((value, reference_prompt, copy.deepcopy(checks)))
            return {'content_failure': 0}

        action = {'id': 91, 'style': 'watercolor', 'lighting': 'soft', 'detail': 'intricate'}
        self.Probe(matched_suite(), repeats=2).evaluate(action, predict)
        self.assertEqual(received[0], (
            'Watercolor style. Soft lighting. Detail level: intricate. Scene: '
            'A person in a blue coat holding a cup.',
            'A person in a blue coat holding a cup.',
            [{'question': 'Is a person holding a cup?', 'answer': 'Yes'}],
        ))
        self.assertEqual(received[1][0], 'Watercolor style. Soft lighting. Detail level: intricate. '
                         'Scene: A person in a red coat holding a cup.')
        self.assertEqual(received[4][1], 'A person in a red coat reading a book.')

    def test_cache_uses_complete_canonical_action_and_keeps_results_independent(self):
        calls = []

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            calls.append(seed)
            return {'content_failure': 0 if 'blue' in value else .5}

        probe = self.Probe(matched_suite(), repeats=2)
        action = {'id': 4, 'style': 'watercolor', 'lighting': None}
        first = probe.evaluate(action, predict)
        self.assertEqual(len(calls), 8)
        first['score'] = 1
        first['cases'][0]['samples'][0]['groups'][0]['checks'][0]['answer'] = 'corrupted'
        cached = probe.evaluate({'lighting': None, 'style': 'watercolor', 'id': 4}, predict)
        self.assertEqual(len(calls), 8)
        self.assertEqual(cached['score'], .5)
        self.assertEqual(cached['cases'][0]['samples'][0]['groups'][0]['checks'][0]['answer'], 'Yes')
        probe.evaluate({**action, 'lighting': 'soft'}, predict)
        self.assertEqual(len(calls), 16, 'Same action ID with different settings must be measured')
        probe.evaluate({**action, 'detail': None}, predict)
        self.assertEqual(len(calls), 24, 'The full action, including disabled fields, is the cache key')
        probe.evaluate(None, predict)
        probe.evaluate(None, predict)
        self.assertEqual(len(calls), 32)

    def test_new_probe_uses_its_own_suite_repeats_seed_and_cache(self):
        calls = []

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            calls.append((seed, reference_prompt))
            return {'content_failure': 0}

        self.Probe(matched_suite(), repeats=2, seed=10).evaluate(None, predict)
        other = matched_suite()
        other['cases'] = other['cases'][:1]
        other['cases'][0]['groups'][0]['prompt'] = 'A different explicitly chosen prompt.'
        result = self.Probe(other, repeats=3, seed=11).evaluate(None, predict)
        self.assertEqual(len(calls), 14)
        self.assertEqual(calls[8], (11, 'A different explicitly chosen prompt.'))
        self.assertEqual(result['sample_count'], 6)

    def test_source_and_callback_mutation_cannot_change_config_or_evidence(self):
        source = matched_suite()
        probe = self.Probe(source, repeats=2)
        source['cases'][0]['groups'][0]['prompt'] = 'caller changed prompt'
        source['cases'][0]['groups'][0]['checks'][0][1] = 'No'
        action = {'style': 'watercolor'}

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            self.assertTrue(value.startswith('Watercolor style.'))
            self.assertEqual(checks[0]['answer'], 'Yes')
            checks[0]['answer'] = 'callback changed answer'
            action['style'] = 'photograph'
            return {'content_failure': 0}

        result = probe.evaluate(action, predict)
        observation = result['cases'][0]['samples'][0]['groups'][0]
        self.assertEqual(observation['prompt'], 'A person in a blue coat holding a cup.')
        self.assertEqual(observation['checks'][0]['answer'], 'Yes')

    def test_actual_check_answers_survive_callback_output_and_cache_mutations(self):
        returned_traces = []

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            blue = 'blue' in reference_prompt
            trace = [{'question': check['question'], 'expected': check['answer'],
                      'answer': '  YES.  ' if blue else 'No.', 'passed': blue,
                      **({'skill': check['skill']} if 'skill' in check else {})}
                     for check in checks]
            returned_traces.append(trace)
            return {'content_failure': .25 if blue else .75, 'checks': trace}

        probe = self.Probe(matched_suite(), repeats=2)
        result = probe.evaluate(None, predict)
        observed = result['cases'][0]['samples'][0]['groups']
        expected = [{'question': 'Is a person holding a cup?', 'expected': 'Yes',
                     'answer': '  YES.  ', 'passed': True}]
        self.assertEqual(observed[0].get('check_results'), expected)
        self.assertEqual(observed[1]['check_results'], [
            {'question': 'Is a person holding a cup?', 'expected': 'Yes',
             'answer': 'No.', 'passed': False, 'skill': 'relation'}])
        self.assertEqual(result['score'], .5, 'Evidence must not change the supplied content score')
        returned_traces[0][0]['answer'] = 'mutated callback output'
        self.assertEqual(observed[0]['check_results'], expected)
        observed[0]['check_results'][0]['answer'] = 'mutated result'
        cached = probe.evaluate(None, predict)
        self.assertEqual(len(returned_traces), 8)
        self.assertEqual(cached['cases'][0]['samples'][0]['groups'][0]['check_results'], expected)
        cached['cases'][0]['samples'][0]['groups'][0]['check_results'][0]['answer'] = 'mutated cache copy'
        again = probe.evaluate(None, predict)
        self.assertEqual(again['cases'][0]['samples'][0]['groups'][0]['check_results'], expected)

    def test_open_answer_support_evidence_survives_matched_comparisons(self):
        source = matched_suite()
        source['cases'] = source['cases'][:1]
        for group in source['cases'][0]['groups']:
            group['checks'] = [{'question': 'How many cups?', 'answer': 'two'}]

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            return structured_output(checks, .8 if 'blue' in reference_prompt else .2)

        result = self.Probe(source, repeats=2).evaluate(None, predict)
        groups = result['cases'][0]['samples'][0]['groups']
        self.assertEqual(result['score'], 1.)
        self.assertEqual([row['check_results'][0]['answer'] for row in groups], ['yes', 'no'])
        self.assertEqual([row['check_results'][0]['expected'] for row in groups], ['two', 'two'])
        self.assertEqual([row['check_results'][0]['passed'] for row in groups], [True, False])
        for index, group in enumerate(groups):
            self.assertIn('check_evidence', group)
            evidence = group['check_evidence'][0]
            self.assertEqual(evidence['method'], 'vev_target_verification')
            self.assertEqual(evidence['readout'], 'yes' if index == 0 else 'no')
            self.assertAlmostEqual(evidence['probabilities'][1], .8 if index == 0 else .2)
            self.assertIn('UNTRUSTED', evidence['question'])
            self.assertIn('"two"', evidence['question'])

    def test_companion_evidence_is_independent_of_callback_results_and_cached_copies(self):
        returned = []

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            output = structured_output(checks)
            returned.append(output)
            return output

        probe = self.Probe(matched_suite(), repeats=2)
        result = probe.evaluate(None, predict)
        group = result['cases'][0]['samples'][0]['groups'][0]
        self.assertIn('check_evidence', group)
        expected = copy.deepcopy(returned[0]['check_evidence'])
        returned[0]['check_evidence'][0]['probabilities'][0] = 999
        returned[0]['check_evidence'][0]['readout'] = 'mutated callback'
        self.assertEqual(group['check_evidence'], expected)
        group['check_evidence'][0]['probabilities'][0] = 999
        cached = probe.evaluate(None, predict)
        cached_group = cached['cases'][0]['samples'][0]['groups'][0]
        self.assertEqual(cached_group['check_evidence'], expected)
        self.assertEqual(len(returned), 8)
        cached_group['check_evidence'][0]['probability_labels'][0] = 'mutated cache copy'
        again = probe.evaluate(None, predict)
        self.assertEqual(again['cases'][0]['samples'][0]['groups'][0]['check_evidence'], expected)

    def test_invalid_companion_evidence_never_enters_reports_or_cache(self):
        checks = [{'question': 'Is a person holding a cup?', 'answer': 'Yes'}]
        valid = structured_output(checks)
        row = valid['check_evidence'][0]
        invalid = [None, [], {}, '', [None], [row, row], (row,)]
        for field in row:
            invalid.append([{key: value for key, value in row.items() if key != field}])
        for field, values in (
            ('id', ['check:1', 0, None]),
            ('type', ['score', None]),
            ('method', ['vev_target_verification', 'other', None]),
            ('status', ['not_evaluated', None]),
            ('question', ['An unrelated question?', None]),
            ('readout', ['no', 'two', None]),
            ('probability_labels', [['yes', 'no'], ['no'], None]),
            ('threshold', [.6, None, True]),
            ('passed', [False, 1, None]),
            ('value', [.1, float('nan'), float('inf'), True, None]),
            ('probabilities', [[.1, .1], [0., 0., 1.], [float('nan'), .8],
                               [float('inf'), .8], [-.1, 1.1], [True, False], None]),
        ):
            invalid.extend([{**row, field: value}] for value in values)
        invalid.append([{**row, 'image': object()}])
        malformed_outputs = [{**valid, 'check_evidence': evidence} for evidence in invalid]
        malformed_outputs.append({key: value for key, value in valid.items() if key != 'checks'})
        malformed_outputs.append({**valid, 'checks': [{**valid['checks'][0], 'passed': False}]})
        malformed_outputs.append({**valid, 'checks': [{**valid['checks'][0], 'answer': 'no'}]})
        for malformed in malformed_outputs:
            with self.subTest(malformed=repr(malformed)[:150]):
                probe = self.Probe(matched_suite(), repeats=2)
                attempts = []

                def predict(value, *, checks, reference_prompt, seed, artifact_name):
                    attempts.append(seed)
                    return malformed if len(attempts) == 1 else {'content_failure': 0}

                with self.assertRaises(ValueError):
                    probe.evaluate(None, predict)
                self.assertEqual(len(attempts), 1)
                retried = probe.evaluate(None, predict)
                self.assertEqual(len(attempts), 9)
                self.assertNotIn('check_evidence', retried['cases'][0]['samples'][0]['groups'][0])

    def test_companion_evidence_ids_cannot_be_duplicated_or_reordered(self):
        source = matched_suite()
        checks = [{'question': 'Is a person holding a cup?', 'answer': 'Yes'},
                  {'question': 'Is a dog visible?', 'answer': 'No'}]
        for case in source['cases']:
            for group in case['groups']:
                group['checks'] = checks
        valid = structured_output(checks)
        for evidence in [valid['check_evidence'][::-1],
                         [valid['check_evidence'][0], valid['check_evidence'][0]]]:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                self.Probe(source, repeats=2).evaluate(None,
                    lambda *args, **kwargs: {**valid, 'check_evidence': evidence})

    def test_invalid_or_incomplete_check_trace_is_rejected_and_never_cached(self):
        row = {'question': 'Is a person holding a cup?', 'expected': 'Yes',
               'answer': 'No', 'passed': False}
        invalid = [None, [], {}, '', True, float('nan'), [None], [row, row], (row,)]
        for field in row:
            invalid.append([{key: value for key, value in row.items() if key != field}])
        for field, values in (
            ('question', ['An unrelated question?', float('inf')]),
            ('expected', ['a different expected answer', float('nan')]),
            ('answer', ['', '   ', None, True, [], object(), float('nan'), float('inf'),
                        'x' * 1001, 'bad\x00answer']),
            ('passed', [None, 0, 1, 'false', float('nan')]),
            ('skill', ['', None, float('nan'), 'x' * 201]),
        ):
            invalid.extend([{**row, field: value}] for value in values)
        invalid.append([{**row, 'image': object()}])
        for trace in invalid:
            with self.subTest(trace=repr(trace)[:100]):
                probe = self.Probe(matched_suite(), repeats=2)
                attempts = []

                def predict(value, *, checks, reference_prompt, seed, artifact_name):
                    attempts.append(seed)
                    if len(attempts) == 1:
                        return {'content_failure': 0, 'checks': trace}
                    return {'content_failure': 0}

                with self.assertRaises(ValueError):
                    probe.evaluate(None, predict)
                self.assertEqual(len(attempts), 1)
                retried = probe.evaluate(None, predict)
                self.assertEqual(len(attempts), 9)
                self.assertNotIn('check_results', retried['cases'][0]['samples'][0]['groups'][0])

    def test_invalid_content_scores_raise_without_caching_partial_results(self):
        for invalid in (None, {}, {'content_failure': None}, {'content_failure': True},
                        {'content_failure': '0.2'}, {'content_failure': float('nan')},
                        {'content_failure': float('inf')}, {'content_failure': -.01},
                        {'content_failure': 1.01}):
            with self.subTest(invalid=invalid):
                probe = self.Probe(matched_suite(), repeats=2)
                attempts = []

                def predict(value, *, checks, reference_prompt, seed, artifact_name):
                    attempts.append(seed)
                    return invalid if len(attempts) == 2 else {'content_failure': 0}

                with self.assertRaises(ValueError):
                    probe.evaluate(None, predict)
                self.assertEqual(len(attempts), 2)
                self.assertEqual(probe.evaluate(None, predict)['sample_count'], 8)
                self.assertEqual(len(attempts), 10)

    def test_predictor_errors_propagate_and_full_suite_is_retried(self):
        probe = self.Probe(matched_suite(), repeats=2)
        calls = []

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            calls.append(seed)
            if len(calls) == 3:
                raise RuntimeError('model unavailable')
            return {'content_failure': .1}

        with self.assertRaisesRegex(RuntimeError, 'model unavailable'):
            probe.evaluate(None, predict)
        self.assertEqual(probe.evaluate(None, predict)['sample_count'], 8)
        self.assertEqual(len(calls), 11)

    def test_cancellation_stops_between_inferences_and_does_not_cache_partial_result(self):
        probe = self.Probe(matched_suite(), repeats=2)
        calls = []

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            calls.append(seed)
            return {'content_failure': 0}

        with self.assertRaises(InterruptedError):
            probe.evaluate(None, predict, cancelled=lambda: len(calls) >= 1)
        self.assertEqual(len(calls), 1)
        result = probe.evaluate(None, predict)
        self.assertEqual(len(calls), 9)
        self.assertEqual(result['sample_count'], 8)

    def test_cancellation_before_start_and_after_final_inference_does_not_cache(self):
        probe = self.Probe(matched_suite(), repeats=2)
        calls = []

        def predict(value, *, checks, reference_prompt, seed, artifact_name):
            calls.append(seed)
            return {'content_failure': 0}

        with self.assertRaises(InterruptedError):
            probe.evaluate(None, predict, cancelled=lambda: True)
        self.assertEqual(calls, [])
        with self.assertRaises(InterruptedError):
            probe.evaluate(None, predict, cancelled=lambda: len(calls) == 8)
        self.assertEqual(len(calls), 8)
        probe.evaluate(None, predict)
        self.assertEqual(len(calls), 16)
        with self.assertRaises(InterruptedError):
            probe.evaluate(None, predict, cancelled=lambda: True)
        self.assertEqual(len(calls), 16)

    def test_progress_reports_phase_and_completed_sample_counts(self):
        probe = self.Probe(matched_suite(), repeats=2)
        events = []

        def progress(**fields):
            events.append(fields)

        result = probe.evaluate(None, self.constant_predict, progress=progress)
        self.assertTrue(events)
        self.assertTrue(all(isinstance(event['phase'], str) and event['phase'] for event in events))
        self.assertEqual(events[-1]['completed'], 8)
        self.assertEqual(events[-1]['total'], result['sample_count'])

    def test_invalid_or_nonfinite_action_is_rejected_before_inference(self):
        def unexpected(*args, **kwargs):
            self.fail('Invalid actions must not run inference')

        for action in (False, [], 'watercolor', {}, {'style': 'unknown'},
                       {'style': 'watercolor', 'id': float('nan')},
                       {'style': 'watercolor', 'extra': object()},
                       {'style': 'watercolor', 3: 'not a text key'}):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.Probe(matched_suite()).evaluate(action, unexpected)


if __name__ == '__main__':
    unittest.main()
