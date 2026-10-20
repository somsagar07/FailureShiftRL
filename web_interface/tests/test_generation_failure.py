"""Combined generation rewards preserve independent evidence and missing signals."""
import copy
import json
import unittest

from generation_failure import assess_generation, combined_score, evaluate_requirements


def measured(content=0., instructions=None, visual=0., bias=None):
    return {'failure_scores': dict(content=content, instructions=instructions, visual=visual, bias=bias)}


class CombinedScoreTests(unittest.TestCase):
    def test_severe_visual_defect_dominates_good_content_without_baseline_discount(self):
        output = measured(content=.0, instructions=.25, visual=1.)
        baseline = measured(content=.0, visual=.75)
        result = combined_score(output, baseline)
        self.assertEqual(result['reward'], 1.)
        self.assertEqual(result['severity'], 1.)
        self.assertTrue(result['failure'])
        self.assertEqual(result['metrics']['baseline_severity'], .75)
        self.assertEqual(result['metrics']['severity_increase'], .25)
        self.assertEqual(result['metrics']['quality'], 0.)
        self.assertEqual(result['metrics']['baseline_quality'], .25)

    def test_already_failing_baseline_does_not_dilute_reward(self):
        result = combined_score(measured(content=.5), measured(visual=1.))
        self.assertEqual(result['reward'], .5)
        self.assertEqual(result['metrics']['severity_increase'], 0.)

    def test_optional_absent_scores_remain_null_and_bias_can_dominate(self):
        result = combined_score(measured(content=.25), measured())
        self.assertIsNone(result['metrics']['instructions_failure'])
        self.assertIsNone(result['metrics']['bias_failure'])
        result = combined_score(measured(content=.25, bias=.75), measured())
        self.assertEqual(result['severity'], .75)
        self.assertEqual(result['metrics']['bias_failure'], .75)
        json.dumps(result, allow_nan=False)

    def test_threshold_is_inclusive_but_zero_severity_never_flags_failure(self):
        for severity, threshold, expected in [(0., 0., False), (.25, .5, False),
                                               (.5, .5, True), (1., 1., True)]:
            with self.subTest(severity=severity, threshold=threshold):
                result = combined_score(measured(content=severity), measured(), threshold)
                self.assertEqual(result['failure'], expected)
                self.assertEqual(result['reward'], severity)

    def test_invalid_component_scores_and_thresholds_cannot_be_rewards(self):
        for value in (True, False, float('nan'), float('inf'), -.01, 1.01, '0.5', [], {}):
            for field in ('content', 'instructions', 'visual', 'bias'):
                with self.subTest(value=value, field=field):
                    output = measured()
                    output['failure_scores'][field] = value
                    with self.assertRaises(ValueError):
                        combined_score(output, measured())
                    with self.assertRaises(ValueError):
                        combined_score(measured(), output)
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                combined_score(measured(), measured(), value)

    def test_content_and_visual_are_required_and_unknown_components_are_rejected(self):
        for field in ('content', 'visual'):
            for missing in (True, False):
                with self.subTest(field=field, missing=missing):
                    output = measured()
                    if missing:
                        del output['failure_scores'][field]
                    else:
                        output['failure_scores'][field] = None
                    with self.assertRaises(ValueError):
                        combined_score(output, measured())
        output = measured()
        output['failure_scores']['confidence'] = 1.
        with self.assertRaises(ValueError):
            combined_score(output, measured())
        for output in (None, {}, {'failure_scores': []}):
            with self.subTest(output=output), self.assertRaises(ValueError):
                combined_score(output, measured())


class ScriptedJudge:
    """Replace only expensive inference; production still performs answer scoring."""
    def __init__(self, answers):
        self.answers = iter(answers)
        self.questions = []

    def answer(self, image, question):
        self.questions.append(question)
        return next(self.answers)


class BatchJudge:
    """Replace inference while keeping production question construction and scoring."""
    def __init__(self, answers):
        self.answers = answers
        self.batches = []

    def answer_many(self, image, questions, *, cancelled=None):
        self.batches.append(list(questions))
        return self.answers

    def answer(self, image, question):
        raise AssertionError('Assessment should submit independent questions together.')


class BatchedAssessmentTests(unittest.TestCase):
    def test_checks_and_enabled_rubrics_are_batched_without_changing_scores_or_leaking_targets(self):
        judge = BatchJudge(['two', 'blue', '3', '1'])
        checks = [{'question': 'How many cats?', 'answer': 'two'},
                  {'question': 'What color is the lake?', 'answer': 'secret-target'}]
        result = assess_generation(judge, object(), 'Cats by a lake',
                                   action={'style': 'watercolor'}, checks=checks)
        self.assertEqual(result['failure_scores'],
                         dict(content=.5, instructions=.75, visual=.25, bias=None))
        self.assertEqual(result['checks_passed'], 1)
        self.assertEqual(result['failure_evidence']['instructions']['answer'], '3')
        self.assertEqual(result['failure_evidence']['visual']['answer'], '1')
        self.assertEqual(len(judge.batches), 1)
        self.assertEqual(judge.batches[0][:2], ['How many cats?', 'What color is the lake?'])
        self.assertEqual(len(judge.batches[0]), 4)
        self.assertNotIn('secret-target', '\n'.join(judge.batches[0]))

    def test_content_fallback_and_baseline_exclude_inapplicable_instruction_question(self):
        judge = BatchJudge(['1', '4'])
        result = assess_generation(judge, object(), 'A floating cat')
        self.assertEqual(result['failure_scores'],
                         dict(content=.25, instructions=None, visual=1., bias=None))
        self.assertEqual(len(judge.batches), 1)
        self.assertEqual(len(judge.batches[0]), 2)

    def test_comparison_checks_use_batching_and_still_reject_abstentions(self):
        checks = [{'question': 'How many cats?', 'answer': 'two'},
                  {'question': 'Is the lake blue?', 'answer': 'yes'}]
        judge = BatchJudge(['two', 'no'])
        result = evaluate_requirements(judge, object(), checks)
        self.assertEqual(result['quality'], .5)
        self.assertEqual(judge.batches, [['How many cats?', 'Is the lake blue?']])
        with self.assertRaisesRegex(ValueError, 'unknown|abstain'):
            evaluate_requirements(BatchJudge(['two', 'unknown: cannot determine']), object(), checks)

    def test_missing_or_extra_batch_answers_never_produce_partial_scores(self):
        for answers in (None, '00', [], ['0'], ['0', '0', '0']):
            with self.subTest(answers=answers), self.assertRaisesRegex(ValueError, 'one answer|every|count'):
                assess_generation(BatchJudge(answers), object(), 'A cat')

    def test_cancelled_batch_is_not_returned_as_completed_evidence(self):
        judge = BatchJudge(['0', '0'])
        with self.assertRaises(InterruptedError):
            assess_generation(judge, object(), 'A cat', cancelled=lambda: bool(judge.batches))
        self.assertEqual(len(judge.batches), 1)

class GenerationAssessmentTests(unittest.TestCase):
    image = object()

    def test_fallback_rubrics_preserve_complete_prompt_and_raw_evidence(self):
        prompt = 'Two {red} cats floating over a turquoise lake. END-OF-SCENE'
        judge = ScriptedJudge(['1', ' 4 '])
        result = assess_generation(judge, self.image, prompt)
        self.assertEqual(result['failure_scores'],
                         dict(content=.25, instructions=None, visual=1., bias=None))
        self.assertEqual(result['quality'], 0.)
        self.assertEqual(result['failure_evidence']['instructions']['status'], 'not_applicable')
        self.assertEqual(result['failure_evidence']['bias']['status'], 'not_evaluated')
        content = result['failure_evidence']['content']
        self.assertEqual(content['method'], 'prompt_fulfillment_rubric')
        self.assertEqual(content['answer'], '1')
        self.assertEqual(content['score'], .25)
        visual = result['failure_evidence']['visual']
        self.assertEqual(visual['answer'], ' 4 ')
        self.assertEqual(visual['score'], 1.)
        for evidence in (content, visual):
            self.assertIn(prompt, evidence['question'])
            self.assertEqual(evidence['status'], 'evaluated')
        self.assertNotIn('checks', result)
        json.dumps(result, allow_nan=False)

    def test_dataset_checks_define_content_without_putting_expected_answers_in_questions(self):
        checks = [{'question': 'How many cats?', 'answer': 'two', 'skill': 'count'},
                  {'question': 'What color is the lake?', 'answer': 'target-secret-93'}]
        judge = ScriptedJudge(['two', 'turquoise', '0'])
        original = copy.deepcopy(checks)
        result = assess_generation(judge, self.image, 'Cats by a lake', checks=checks)
        self.assertEqual(result['failure_scores']['content'], .5)
        self.assertEqual(result['quality'], .5)
        self.assertEqual(result['checks_total'], 2)
        self.assertEqual(result['checks_passed'], 1)
        self.assertEqual(result['checks'][1]['expected'], 'target-secret-93')
        self.assertEqual(result['checks'][1]['answer'], 'turquoise')
        self.assertEqual(result['failure_evidence']['content']['method'], 'dataset_checks')
        self.assertEqual(checks, original)
        self.assertEqual(judge.questions[:2], ['How many cats?', 'What color is the lake?'])
        self.assertNotIn('target-secret-93', '\n'.join(judge.questions))

    def test_only_enabled_presentation_settings_enter_the_instruction_rubric(self):
        action = {'id': 77, 'style': 'watercolor', 'lighting': None, 'detail': 'minimal'}
        judge = ScriptedJudge(['0', '3', '1'])
        result = assess_generation(judge, self.image, 'A floating cat', action=action)
        evidence = result['failure_evidence']['instructions']
        self.assertEqual(result['failure_scores']['instructions'], .75)
        self.assertEqual(result['quality'], .25)
        self.assertEqual(evidence['method'], 'presentation_rubric')
        self.assertEqual(evidence['answer'], '3')
        self.assertIn('"style": "watercolor"', evidence['question'])
        self.assertIn('"detail": "minimal"', evidence['question'])
        self.assertNotIn('lighting', evidence['question'].lower())
        self.assertNotIn('77', evidence['question'])
        self.assertEqual(action['lighting'], None)

    def test_style_only_actions_and_all_disabled_axes_have_correct_applicability(self):
        result = assess_generation(ScriptedJudge(['0', '2', '0']), self.image, 'A cat',
                                   action={'id': 1, 'style': 'watercolor'})
        self.assertEqual(result['failure_scores']['instructions'], .5)
        result = assess_generation(ScriptedJudge(['0', '0']), self.image, 'A cat',
                                   action={'style': None, 'lighting': None, 'detail': None})
        self.assertIsNone(result['failure_scores']['instructions'])
        self.assertEqual(result['failure_evidence']['instructions']['status'], 'not_applicable')

    def test_visual_rubric_uses_requested_style_and_surreal_scene_as_context(self):
        result = assess_generation(ScriptedJudge(['0', '0', '0']), self.image,
                                   'An intentionally surreal floating cat',
                                   action={'style': 'pencil drawing'})
        question = result['failure_evidence']['visual']['question']
        self.assertIn('An intentionally surreal floating cat', question)
        self.assertIn('pencil drawing', question)
        self.assertIn('intentional', question.lower())
        self.assertIn('surreal', question.lower())

    def test_invalid_rubric_answers_stop_assessment_instead_of_becoming_failure_scores(self):
        for answer in ('', 'unknown', 'I cannot tell', '1 or 4', '4 severe', '0.5', '5', '-1',
                       'NaN', '2.', 'two', None, True, 4):
            for location in (0, 1, 2):
                answers = ['0', '0', '0']
                answers[location] = answer
                with self.subTest(answer=answer, location=location), self.assertRaisesRegex(ValueError, 'rating|rubric'):
                    assess_generation(ScriptedJudge(answers), self.image, 'A cat', action={'style': 'photograph'})

    def test_invalid_context_is_rejected_before_any_judge_question(self):
        for prompt in (None, '', ' ', 'a' * 10001, 'cat\x00'):
            judge = ScriptedJudge([])
            with self.subTest(prompt=repr(prompt)[:30]), self.assertRaises(ValueError):
                assess_generation(judge, self.image, prompt)
            self.assertEqual(judge.questions, [])
        for action in ('watercolor', [], {}, {'id': 4}, {'style': 'unknown'},
                       {'style': True}, {'style': 'watercolor', 'lighting': 'blue'},
                       {'style': 'watercolor', 'id': True}, {'style': 'watercolor', 'id': -1},
                       {'style': 'watercolor', 'arbitrary': 'value'},
                       {'description': 'cool', 'profession': 'artist', 'setting': 'classroom'}):
            judge = ScriptedJudge([])
            with self.subTest(action=action), self.assertRaises(ValueError):
                assess_generation(judge, self.image, 'A cat', action=action)
            self.assertEqual(judge.questions, [])
        judge = ScriptedJudge([])
        with self.assertRaises(ValueError):
            assess_generation(judge, self.image, 'A cat', checks=[{'question': 'Any cat?'}])
        self.assertEqual(judge.questions, [])


class RequirementEvaluationTests(unittest.TestCase):
    image = object()

    def test_requirements_share_complete_answer_matching_without_rubric_inferences(self):
        judge = ScriptedJudge(['two', 'not blue'])
        checks = [{'question': 'How many cats?', 'answer': '2', 'skill': 'count'},
                  {'question': 'Which color?', 'answer': 'blue'}]
        result = evaluate_requirements(judge, self.image, checks)
        self.assertEqual(result['quality'], .5)
        self.assertEqual(result['checks_passed'], 1)
        self.assertEqual(result['checks_total'], 2)
        self.assertEqual([row['passed'] for row in result['checks']], [True, False])
        self.assertEqual(judge.questions, ['How many cats?', 'Which color?'])

    def test_abstention_answers_cannot_raise_content_failure(self):
        checks = [{'question': 'Is there a cat?', 'answer': 'yes'}]
        for answer in ('unknown', 'Unclear.', "I don't know", 'cannot determine',
                       'I cannot tell from the image.', 'insufficient information', 'n/a'):
            for assess in (False, True):
                judge = ScriptedJudge([answer, '0'])
                with self.subTest(answer=answer, assess=assess), self.assertRaisesRegex(ValueError, 'unknown|abstain'):
                    if assess:
                        assess_generation(judge, self.image, 'A cat', checks=checks)
                    else:
                        evaluate_requirements(judge, self.image, checks)

    def test_literal_unknown_answer_is_valid_when_it_is_the_expected_scene_text(self):
        result = evaluate_requirements(ScriptedJudge(['Unknown.']), self.image,
            [{'question': 'What word is written on the sign?', 'answer': 'unknown'}])
        self.assertEqual(result['quality'], 1.)

    def test_explained_abstentions_do_not_become_content_failure_or_bias_measurements(self):
        checks = [{'question': 'Is there a cat?', 'answer': 'yes'}]
        answers = ('unknown: cannot determine', 'uncertain due to image quality', 'not clear',
                   'I am unable to tell', 'Unclear because the image is blurry.',
                   "I'm unsure whether a cat is present.", 'I’m not sure from this image.',
                   'It is not clear from the image.', "It's unknown from this view.",
                   'There is insufficient information to answer.', 'Not enough information to tell.',
                   "Sorry, I can't determine that.", 'I cannot confidently determine the answer.',
                   'It cannot be determined from the image.', 'I do not have enough information.',
                   'N/A: no reliable answer is available.')
        for answer in answers:
            for assess in (False, True):
                judge = ScriptedJudge([answer, '0'])
                with self.subTest(answer=answer, assess=assess):
                    with self.assertRaisesRegex(ValueError, 'unknown|abstain'):
                        if assess:
                            assess_generation(judge, self.image, 'A cat', checks=checks)
                        else:
                            evaluate_requirements(judge, self.image, checks)
                    self.assertEqual(judge.questions, ['Is there a cat?'])

    def test_abstention_phrases_can_still_be_exact_expected_scene_text(self):
        for text in ('unknown: cannot determine', 'uncertain due to image quality',
                     'not clear', 'I am unable to tell'):
            with self.subTest(text=text):
                result = evaluate_requirements(ScriptedJudge([text]), self.image,
                    [{'question': 'What text is printed on the sign?', 'answer': text}])
                self.assertEqual(result['quality'], 1.)
                self.assertTrue(result['checks'][0]['passed'])

    def test_negative_semantic_answers_and_embedded_uncertainty_words_are_not_abstentions(self):
        for answer, expected in [('not blue', 'blue'), ('not two', 'two'), ('no', 'yes'),
                                 ('the sign says unknown', 'blue'), ('uncertainty', 'blue')]:
            with self.subTest(answer=answer):
                result = evaluate_requirements(ScriptedJudge([answer]), self.image,
                    [{'question': 'Which answer describes the image?', 'answer': expected}])
                self.assertEqual(result['quality'], 0.)
                self.assertFalse(result['checks'][0]['passed'])

    def test_missing_checks_and_empty_answers_are_errors(self):
        for checks in (None, [], [{'answer': 'yes'}]):
            judge = ScriptedJudge([])
            with self.subTest(checks=checks), self.assertRaises(ValueError):
                evaluate_requirements(judge, self.image, checks)
            self.assertEqual(judge.questions, [])
        for answer in ('', ' ', None, True):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                evaluate_requirements(ScriptedJudge([answer]), self.image,
                                      [{'question': 'Is there a cat?', 'answer': 'yes'}])

    def test_cancelled_evaluation_stops_before_the_first_question(self):
        for assess in (False, True):
            judge = ScriptedJudge([])
            with self.subTest(assess=assess), self.assertRaises(InterruptedError):
                if assess:
                    assess_generation(judge, self.image, 'A cat', cancelled=lambda: True)
                else:
                    evaluate_requirements(judge, self.image,
                        [{'question': 'Is there a cat?', 'answer': 'yes'}], cancelled=lambda: True)
            self.assertEqual(judge.questions, [])

    def test_cancellation_after_an_answer_stops_following_questions(self):
        for assess in (False, True):
            cancelled = [False]

            class CancellingJudge(ScriptedJudge):
                def answer(self, image, question):
                    answer = super().answer(image, question)
                    cancelled[0] = True
                    return answer

            judge = CancellingJudge(['yes', '2', '0'])
            checks = [{'question': 'Is there a cat?', 'answer': 'yes'},
                      {'question': 'How many cats?', 'answer': '2'}]
            with self.subTest(assess=assess), self.assertRaises(InterruptedError):
                if assess:
                    assess_generation(judge, self.image, 'A cat', checks=checks, cancelled=lambda: cancelled[0])
                else:
                    evaluate_requirements(judge, self.image, checks, cancelled=lambda: cancelled[0])
            self.assertEqual(judge.questions, ['Is there a cat?'])

    def test_cancellation_after_last_rubric_does_not_return_completed_assessment(self):
        judge = ScriptedJudge(['0', '0'])
        with self.assertRaises(InterruptedError):
            assess_generation(judge, self.image, 'A cat', cancelled=lambda: len(judge.questions) == 2)
        self.assertEqual(len(judge.questions), 2)

    def test_cancellation_context_requires_a_callable_with_boolean_result(self):
        for cancelled in (False, True, lambda: None, lambda: 'no', lambda: 1):
            judge = ScriptedJudge(['0', '0'])
            with self.subTest(cancelled=cancelled), self.assertRaises(ValueError):
                assess_generation(judge, self.image, 'A cat', cancelled=cancelled)
            self.assertEqual(judge.questions, [])


if __name__ == '__main__':
    unittest.main()
