"""Prompt requirements and desired images stay attached to their own dataset rows."""
import copy
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from adaptation import TrainingExamples, prepare, summarize
from core import actions_for, apply_prompt
from discovery import DiscoveryManager
from test_discovery import Analyses, Evaluator, EMBEDDING_ID


DATASET_ID = 'e' * 32
ROWS = [dict(text='two green cats', checks=[{'question': 'How many cats?', 'answer': 'two'}]),
        dict(text='a red bird', checks=[{'question': 'Is the bird red?', 'answer': 'Yes'}])]


class DatasetRows:
    def __init__(self):
        self.rows = copy.deepcopy(ROWS)

    def ready(self, identity):
        assert identity == DATASET_ID
        count = sum(bool(row['checks']) for row in self.rows)
        return dict(id=identity, task='generation', count=len(self.rows), name='fixture/prompts',
                    source='huggingface', prompt_column='prompt', checks_column='vqa_list',
                    checked_count=count, missing_checks_count=len(self.rows)-count)

    def text(self, identity, index):
        assert identity == DATASET_ID
        return dict(index=index, name=f'test[{index}]', **copy.deepcopy(self.rows[index]))


class Previews:
    def __init__(self, snapshot):
        self.snapshot, self.datasets = snapshot, DatasetRows()

    def get(self, identity, input_index=0):
        result = copy.deepcopy(self.snapshot)
        row = self.datasets.text(DATASET_ID, input_index)
        result.update(original={'text': row['text'], 'checks': row['checks']}, checks=row['checks'],
                      input_index=input_index, dataset=self.datasets.ready(DATASET_ID))
        result['nodes'] = [dict(id=action['id'], action=action, text=apply_prompt(row['text'], action))
                           for action in result['actions']]
        return result


class GenerationDatasetDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.analyses = Analyses()
        self.analyses.context['states'] = np.array([[.6, .8], [.8, .6]], dtype=np.float32)
        self.analyses.context['model']['task'] = 'generation'
        preview = self.analyses.context['preview']
        preview.update(input_count=2, actions=[dict(id=0, style='photograph'), dict(id=1, style='watercolor')])
        preview['config'].update(task='generation', dataset_id=DATASET_ID, text='')
        self.previews = Previews(preview)
        self.analyses.context['preview'] = self.previews.get(preview['id'])
        self.evaluator = Evaluator()
        self.manager = DiscoveryManager(self.previews, self.analyses, self.root/'discoveries',
                                       evaluator_factory=lambda *args, **kwargs: self.evaluator)

    def discover(self, **options):
        job = self.manager.start(dict(embedding_id=EMBEDDING_ID, episodes=40, seed=2, **options))
        deadline = time.monotonic()+10
        while time.monotonic()<deadline:
            report = self.manager.get(job['id'])
            if report['status'] in ('completed', 'error', 'cancelled'):
                self.assertEqual(report['status'], 'completed', report.get('error'))
                return report
            time.sleep(.01)
        self.fail('Discovery did not finish.')

    def test_all_baselines_and_actions_receive_their_own_checks_and_matched_seeds(self):
        report = self.discover()
        self.assertEqual(report['config']['generation_metric'], 'combined')
        self.assertEqual(report['config']['judge_model'], 'CountingSheep/vev-4b')
        self.assertEqual(report['config']['failure_threshold'], .5)
        self.assertEqual({row['input_index'] for row in report['baseline']}, {0, 1})
        for value, options in self.evaluator.calls:
            index = 0 if 'cats' in value else 1
            self.assertEqual(options['checks'], ROWS[index]['checks'])
            self.assertEqual(options['reference_prompt'], ROWS[index]['text'])
            self.assertEqual(options['seed'], 2+index)
            if value == ROWS[index]['text']:
                self.assertIsNone(options['action'])
            else:
                self.assertIn(options['action'], self.previews.snapshot['actions'])
        self.assertEqual(len(report['predicted_landscape']['input_mean_reward']), 2)
        self.assertEqual(report['config']['dataset_id'], DATASET_ID)

    def test_partial_requirements_never_silently_switch_metric(self):
        self.previews.datasets.rows[1]['checks'] = []
        with self.assertRaisesRegex(ValueError, '1 prompts.*requirements'):
            self.manager.start(dict(embedding_id=EMBEDDING_ID, generation_metric='checks'))
        self.assertFalse(self.manager.jobs)
        report = self.discover(generation_metric='clip')
        self.assertEqual(report['config']['generation_metric'], 'clip')
        self.assertTrue(all('checks' not in options for value, options in self.evaluator.calls))

    def test_combined_discovery_keeps_partial_requirements_and_uses_visual_fallback(self):
        self.previews.datasets.rows[1]['checks'] = []
        report = self.discover()
        self.assertEqual(report['config']['generation_metric'], 'combined')
        self.assertIsNone(report['config']['generation_bias_suite'])
        self.assertEqual(report['config']['generation_bias_repeats'], 3)
        for value, options in self.evaluator.calls:
            if 'bird' in value:
                self.assertEqual(options['checks'], [])
            else:
                self.assertEqual(options['checks'], ROWS[0]['checks'])

    def test_combined_rejects_unsupported_judge_and_incomplete_bias_suite_before_launch(self):
        for options in (dict(judge_model='Salesforce/blip-vqa-base'),
                        dict(generation_bias_repeats=True), dict(generation_bias_repeats=1),
                        dict(generation_bias_suite={'name':'incomplete'})):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.manager.start(dict(embedding_id=EMBEDDING_ID, **options))
        self.assertFalse(self.manager.jobs)

    def test_qwen_remains_an_explicit_combined_judge(self):
        report = self.discover(judge_model='Qwen/Qwen3-VL-2B-Instruct')
        self.assertEqual(report['config']['judge_model'], 'Qwen/Qwen3-VL-2B-Instruct')

    def test_comparison_inference_attempts_are_included_in_live_model_calls(self):
        class CountingEvaluator(Evaluator):
            def set_runtime_callbacks(self, **callbacks):
                self.callbacks = callbacks

            def predict(self, value, **options):
                self.callbacks['on_model_call']()
                self.callbacks['on_model_call']()
                return super().predict(value, **options)

        self.evaluator = CountingEvaluator()
        report = self.discover()
        main_calls = len(self.evaluator.calls)
        self.assertEqual(report['comparison_model_calls'], 2 * main_calls)
        self.assertEqual(report['model_calls'], 3 * main_calls)
        self.assertLess(main_calls, report['episodes'], 'cached RL visits perform no more model calls')

    def test_generation_precision_is_validated_and_retained_for_reproduction(self):
        for value in ('half', 'float64', True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.manager.start(dict(embedding_id=EMBEDDING_ID, generation_precision=value))
        self.assertFalse(self.manager.jobs)
        report = self.discover(generation_precision='float32')
        self.assertEqual(report['config']['generation_precision'], 'float32')
        report = self.discover()
        self.assertEqual(report['config']['generation_precision'], 'auto')

    def test_adapt_maps_each_approved_image_to_one_prompt_style_pair(self):
        report = self.discover()
        context = self.manager._context(EMBEDDING_ID, include_states=False)
        desired = []
        for index, color in enumerate(('green', 'red')):
            image = self.root/f'target-{index}.png'
            Image.new('RGB', (8, 8), color).save(image)
            desired.append(str(image))
        data = [dict(input_index=1, action_id=0, image_path=desired[1]),
                dict(input_index=0, action_id=1, image_path=desired[0])]
        spec = prepare(report, context, {'action_ids': [0, 1], 'training_data': data})
        examples = TrainingExamples({**spec, 'settings': {'seed': 9}}, context, self.manager._training_input)
        self.assertEqual(list(examples), [dict(input='Photograph style. Scene: a red bird', target=desired[1]),
                                          dict(input='Watercolor style. Scene: two green cats', target=desired[0])])
        self.assertEqual(spec['adaptation_selection']['example_count'], 2)
        self.assertEqual(spec['adaptation_selection']['dataset']['id'], DATASET_ID)
        self.assertTrue(summarize(report, context)['defaults']['uses_dataset'])

    def test_grouped_action_ids_reach_policy_and_exact_adaptation_prompts(self):
        space = dict(style=['watercolor'], lighting=['soft'], detail=['minimal', 'intricate'])
        preview = self.previews.snapshot
        preview['actions'] = actions_for('generation', space)
        preview['config']['action_space'] = space
        self.analyses.context['preview'] = self.previews.get(preview['id'])
        report = self.discover()
        self.assertEqual(report['predicted_landscape']['action_ids'], [85, 89])
        self.assertEqual({record['action_id'] for record in report['records']}, {85, 89})
        context = self.manager._context(EMBEDDING_ID, include_states=False)
        target = self.root / 'approved.png'
        Image.new('RGB', (8, 8), 'green').save(target)
        spec = prepare(report, context, dict(action_ids=[89], training_data=[
            dict(input_index=0, action_id=89, image_path=str(target))]))
        examples = TrainingExamples({**spec, 'settings': {'seed': 9}}, context, self.manager._training_input)
        self.assertEqual(list(examples), [dict(input='Watercolor style. Soft lighting. '
            'Detail level: intricate. Scene: two green cats', target=str(target))])

    def test_adapt_rejects_missing_index_duplicate_pair_and_unrelated_prompt_override(self):
        report = self.discover()
        context = self.manager._context(EMBEDDING_ID, include_states=False)
        image = self.root/'target.png'
        Image.new('RGB', (8, 8), 'green').save(image)
        record = dict(input_index=0, action_id=0, image_path=str(image))
        bad = [dict(action_id=0, image_path=str(image)), dict(record, input_index=2),
               dict(record, input_index=True), dict(record, prompt='unrelated scene')]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                prepare(report, context, {'action_ids': [0], 'training_data': [row]})
        with self.assertRaisesRegex(ValueError, 'once|duplicate'):
            prepare(report, context, {'action_ids': [0], 'training_data': [record, record]})


if __name__ == '__main__':
    unittest.main()
