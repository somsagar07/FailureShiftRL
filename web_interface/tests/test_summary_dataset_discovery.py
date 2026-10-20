"""Dataset rows retain their own source and supervision across discovery and Adapt."""
import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adaptation import TrainingExamples, prepare, summarize
from core import apply_text
from discovery import DiscoveryManager
from test_discovery import Analyses, Evaluator, EMBEDDING_ID


DATASET_ID = 'd' * 32
ROWS = [dict(text='A RED cat runs in the garden.', summary='A red cat runs.'),
        dict(text='A BLUE cat runs quietly.', summary='A blue cat runs.')]


class DatasetRows:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)

    def ready(self, identity):
        assert identity == DATASET_ID
        count = sum(bool(row['summary'].strip()) for row in self.rows)
        return dict(id=identity, task='summarization', count=len(self.rows), name='fixture/articles',
                    source='huggingface', text_column='article', summary_column='highlights',
                    reference_count=count, missing_reference_count=len(self.rows)-count)

    def text(self, identity, index):
        assert identity == DATASET_ID
        return dict(index=index, name=f'train[{index}]', **self.rows[index])


class Previews:
    def __init__(self, snapshot, rows):
        self.snapshot, self.datasets = snapshot, DatasetRows(rows)

    def get(self, identity, input_index=0):
        import random
        result = copy.deepcopy(self.snapshot)
        row = self.datasets.text(DATASET_ID, input_index)
        result.update(original={'text': row['text']}, reference_summary=row['summary'], input_index=input_index,
                      dataset=self.datasets.ready(DATASET_ID))
        result['nodes'] = [dict(id=action['id'], action=action,
                               text=apply_text(row['text'], action, random.Random(42+input_index*1000003+action['id'])))
                           for action in result['actions']]
        return result


class SummaryDatasetDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.analyses = Analyses()
        self.analyses.context['states'] = np.array([[.6, .8], [.8, .6]], dtype=np.float32)
        preview = self.analyses.context['preview']
        preview.update(input_count=2, actions=[dict(id=0, name='to_lowercase'), dict(id=1, name='negate_sentence')])
        preview['config'].update(dataset_id=DATASET_ID, text='')
        self.previews = Previews(preview, ROWS)
        self.analyses.context['preview'] = self.previews.get(preview['id'])
        self.evaluator = Evaluator()
        self.manager = DiscoveryManager(self.previews, self.analyses, Path(self.temp.name)/'discoveries',
            evaluator_factory=lambda *args, **kwargs: self.evaluator)

    def wait(self, getter):
        deadline = time.monotonic()+10
        while time.monotonic()<deadline:
            result = getter()
            if result['status'] in ('completed', 'cancelled', 'error'):
                return result
            time.sleep(.01)
        self.fail('Worker did not finish')

    def discover(self):
        job = self.manager.start(dict(embedding_id=EMBEDDING_ID, episodes=40, summary_metric='reference', seed=2))
        result = self.wait(lambda: self.manager.get(job['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        return result

    def test_each_article_uses_its_own_reference_in_baseline_and_action_calls(self):
        report = self.discover()
        self.assertEqual({row['input_index'] for row in report['baseline']}, {0, 1})
        for value, options in self.evaluator.calls:
            index = 0 if 'RED' in value or 'red' in value else 1
            self.assertEqual(options['reference_summary'], ROWS[index]['summary'])
        for baseline in report['baseline']:
            self.assertEqual(baseline['reference_summary'], ROWS[baseline['input_index']]['summary'])
        self.assertEqual(report['config']['dataset_id'], DATASET_ID)
        self.assertEqual(len(report['predicted_landscape']['input_mean_reward']), 2)

    def test_missing_reference_blocks_before_model_or_worker_starts(self):
        self.previews.datasets.rows[1]['summary'] = ''
        with self.assertRaisesRegex(ValueError, '1.*reference'):
            self.manager.start(dict(embedding_id=EMBEDDING_ID))
        self.assertFalse(self.evaluator.calls)
        self.assertFalse(self.manager.jobs)

    def test_global_reference_cannot_override_a_dataset(self):
        with self.assertRaisesRegex(ValueError, 'dataset|per.article'):
            self.manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='One wrong target for every row.'))
        self.assertFalse(self.manager.jobs)

    def test_adapt_streams_all_articles_and_keeps_targets_without_copying_corpus(self):
        report = self.discover()
        context = self.manager._context(EMBEDDING_ID, include_states=False)
        summary = summarize(report, context)
        self.assertTrue(summary['defaults']['uses_dataset'])
        self.assertEqual(summary['defaults']['dataset']['reference_count'], 2)
        specification = prepare(report, context, {'action_ids': [0]})
        self.assertNotIn('training_data', specification)
        self.assertEqual(specification['training_plan']['dataset_id'], DATASET_ID)
        examples = TrainingExamples({**specification, 'settings': {'seed': 17}}, context, self.manager._training_input)
        self.assertTrue(examples.truncate_inputs)
        self.assertEqual(len(examples), 4)
        self.assertEqual([row['target'] for row in examples], [ROWS[0]['summary']]*2+[ROWS[1]['summary']]*2)
        self.assertEqual([row['input'] for row in examples],
                         [ROWS[0]['text'], ROWS[0]['text'].lower(), ROWS[1]['text'], ROWS[1]['text'].lower()])

    def test_fact_changing_action_needs_corrected_targets_for_each_article(self):
        report = self.discover()
        context = self.manager._context(EMBEDDING_ID, include_states=False)
        with self.assertRaisesRegex(ValueError, 'negate_sentence.*corrected'):
            prepare(report, context, {'action_ids': [1]})
        reviewed = [dict(row, action_summaries={'1': f'Corrected target {index}.'}) for index, row in enumerate(ROWS)]
        specification = prepare(report, context, {'action_ids': [1], 'training_data': reviewed, 'include_originals': False})
        examples = TrainingExamples({**specification, 'settings': {'seed': 17}}, context, self.manager._training_input)
        self.assertEqual([row['target'] for row in examples], ['Corrected target 0.', 'Corrected target 1.'])


class SummaryDatasetModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import T5Config, T5ForConditionalGeneration
        from test_text_encoders import VOCAB, save_tokenizer
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.model_path = cls.directory/'model'
        torch.manual_seed(11)
        T5ForConditionalGeneration(T5Config(vocab_size=len(VOCAB), d_model=12, d_kv=4, d_ff=24,
            num_layers=1, num_decoder_layers=1, num_heads=3, pad_token_id=0, eos_token_id=3,
            decoder_start_token_id=0)).save_pretrained(cls.model_path)
        save_tokenizer(cls.model_path, limit=12)
        cls.model = dict(source='local', name=str(cls.model_path), task='summarization')

    def test_real_inference_accepts_long_articles_but_requires_each_reference(self):
        from discovery_models import SummarizationEvaluator
        evaluator = SummarizationEvaluator(self.model, dict(task='summarization', dataset_id=DATASET_ID,
            summary_metric='reference', max_new_tokens=4, device='cpu'), self.directory/'evaluation')
        self.addCleanup(evaluator.close)
        with self.assertRaisesRegex(ValueError, 'article.*reference'):
            evaluator.predict('a cat')
        with patch.object(evaluator.encoder, '_prepare', wraps=evaluator.encoder._prepare) as prepare_inputs:
            result = evaluator.predict('a cat in the garden '*100, reference_summary='a cat in the garden')
        self.assertIsInstance(result['summary'], str)
        self.assertTrue(prepare_inputs.call_args.kwargs['truncate'])
        first = evaluator.measure_summary('a cat', reference='a cat')
        other = evaluator.measure_summary('a cat', reference='red garden')
        again = evaluator.measure_summary('a cat', reference='a cat')
        self.assertEqual(first['quality'], 1.)
        self.assertEqual(other['quality'], 0.)
        self.assertEqual(first, again, 'Switching articles must not leak reference state')
        self.assertTrue(evaluator.describe()['input_policy']['truncate'])

    def test_real_dataset_training_uses_all_row_targets_and_saves_reloadable_weights(self):
        import torch
        from fine_tuning import fine_tune
        from text_encoders import SummarizationEncoder
        from transformers import AutoModelForSeq2SeqLM
        rows = [dict(text='a cat in the garden '*20, summary='a cat'),
                dict(text='red blue garden '*20, summary='red garden')]
        context = {'preview': {'config': {'task': 'summarization', 'dataset_id': DATASET_ID}},
                   'summary_row': lambda index: rows[index]}
        actions = [{'action_id': 0, 'action': {'id': 0, 'name': 'to_lowercase'}}]
        job = {'training_plan': {'kind': 'summarization', 'dataset_id': DATASET_ID, 'input_count': 2,
                               'example_count': 4, 'action_ids': [0], 'include_originals': True},
               'adaptation_selection': {'actions': actions}, 'settings': {'seed': 17}}
        examples = TrainingExamples(job, context, lambda context, index, action: rows[index]['text'])
        before = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).state_dict()
        with tempfile.TemporaryDirectory(dir=self.directory) as output:
            report = fine_tune(self.model, 'summarization', examples, output,
                dict(device='cpu', epochs=1, learning_rate=.001, batch_size=2, seed=17,
                     weight_decay=.01, max_grad_norm=1., generation_size=128),
                lambda **fields: None, lambda: False)
            self.assertEqual(report['processed_examples'], 4)
            self.assertTrue(report['input_policy']['truncate'])
            trained = SummarizationEncoder(dict(self.model, name=report['checkpoint']))
            self.assertTrue(any(not torch.equal(value, trained.model.state_dict()[key]) for key, value in before.items()))
            self.assertEqual(trained.encode(['a cat']).shape[0], 1)


if __name__ == '__main__':
    unittest.main()
