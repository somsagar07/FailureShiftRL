"""Action selection expands real preview inputs with explicit training targets."""
import copy
import json
import tempfile
import threading
import time
import tracemalloc
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from discovery import DiscoveryManager
from server import PreviewManager, create_server
import test_http
from test_http import MemoryServer


EMBEDDING_ID = 'a' * 32
DISCOVERY_ID = 'c' * 32


class AnalysisSnapshot:
    """Replace model embedding extraction with an immutable analyzed snapshot."""
    def __init__(self, preview):
        self.model_lock = threading.RLock()
        self.released = 0
        self.context = dict(
            job={'id': EMBEDDING_ID, 'status': 'completed', 'preview_id': preview['id']},
            model={'source': 'huggingface', 'name': 'fixture/model', 'task': preview['config']['task']},
            model_key='same-model', states=np.tile([[.6, .8]], (preview['input_count'], 1)),
            preview=preview)

    def discovery_context(self, identity, *, include_states=True):
        if identity != EMBEDDING_ID:
            raise KeyError(identity)
        result = copy.deepcopy({key: value for key, value in self.context.items() if key != 'states'})
        if include_states:
            result['states'] = self.context['states']
        return result

    def release_encoder(self):
        self.released += 1


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.previews = PreviewManager(self.directory / 'previews')
        self.captured = []

    def wait(self, getter):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = getter()
            if result['status'] in ('ready', 'completed', 'cancelled', 'error'):
                return result
            time.sleep(.01)
        self.fail('Worker did not finish')

    def train(self, model, task, examples, directory, settings, progress, cancelled):
        self.captured.append((task, list(examples), settings))
        checkpoint = directory / 'checkpoint'
        checkpoint.mkdir()
        return dict(checkpoint=str(checkpoint), steps=1, training_loss=.2)

    def manager(self, task='classification', records=None, input_count=3):
        if task == 'classification':
            images = self.directory / 'images'
            images.mkdir()
            for index in range(input_count):
                color = ('red', 'green', 'blue')[index % 3]
                Image.new('RGB', (8, 8), color).save(images / f'{index}.png')
            dataset = self.previews.datasets.start({'source': 'folder', 'path': str(images)})
            dataset = self.wait(lambda: self.previews.datasets.get(dataset['id']))
            space = {'rotation': [0], 'darkness': [.1, .2, .3, .4, .5], 'saturation': []}
            config = {'task': task, 'dataset_id': dataset['id'], 'text': '', 'action_space': space}
        elif task == 'summarization':
            config = {'task': task, 'dataset_id': '', 'text': 'The FILTER is Working.',
                      'action_space': {'actions': ['to_lowercase', 'negate_sentence', 'delete_random_character']}}
        else:
            config = {'task': task, 'dataset_id': '',
                      'text': 'A {description} {profession} in a {setting}.',
                      'action_space': {'description': ['unique', 'cool'], 'profession': ['scientist'], 'setting': ['classroom']}}
        config['model'] = {'source': 'huggingface', 'name': 'fixture/model'}
        preview = self.previews.start(config)
        preview = self.wait(lambda: self.previews.get(preview['id']))
        self.assertEqual(preview['status'], 'completed', preview.get('error'))
        self.analyses = AnalysisSnapshot(preview)
        manager = DiscoveryManager(self.previews, self.analyses, self.directory / 'discoveries', finetune_fn=self.train)
        if records is None:
            records = [self.record(0, preview['actions'][0]['id'], True, .8)]
        manager.jobs[DISCOVERY_ID] = dict(
            id=DISCOVERY_ID, embedding_id=EMBEDDING_ID, preview_id=preview['id'], task=task,
            status='completed', phase='completed', records=records,
            config={'seed': 17, 'generation_size': 256, 'reference_summary': 'The filter works.',
                    'label_map': ({'0': 'cat', '1': 1, '2': 'dog',
                                   **{str(index): index % 3 for index in range(3, input_count)}}
                                  if task == 'classification' else {})},
            policy_statistics={'actions': [{'action_id': action['id'], 'mean': .2, 'std': .1}
                                           for action in preview['actions']]})
        (manager.directory / DISCOVERY_ID).mkdir()
        manager._persist(DISCOVERY_ID)
        return manager

    @staticmethod
    def record(index, action, failure, severity, visits=1):
        return dict(input_index=index, action_id=action, failure=failure,
                    failure_rate=float(failure), severity=severity, reward=severity,
                    visits=visits, output={'class_id': 999, 'summary': 'failed output'})

    def start(self, manager, **config):
        job = manager.start_finetune({'discovery_id': DISCOVERY_ID, **config})
        result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        return result

    def test_adaptation_training_defaults_to_auto_and_accepts_explicit_device(self):
        manager=self.manager()
        for device in (None,'cpu','cuda'):
            with self.subTest(device=device):
                settings={} if device is None else {'device':device}
                result=self.start(manager,action_ids=[0],**settings)
                self.assertEqual(result['settings'].get('device'),device or 'auto')
                self.assertEqual(self.captured[-1][2].get('device'),device or 'auto')

    def test_adaptation_rejects_invalid_devices_before_launch(self):
        manager=self.manager()
        for device in ('gpu',None,True,['cuda']):
            with self.subTest(device=device),self.assertRaisesRegex(ValueError,'device'):
                manager.start_finetune({'discovery_id':DISCOVERY_ID,'action_ids':[0],'device':device})
        self.assertFalse(manager.finetunes)

    def test_summary_counts_unique_pairs_and_ranks_observed_failures(self):
        records = [self.record(0, 0, True, .8, 1000), self.record(1, 0, False, .2),
                   self.record(0, 1, True, .7), self.record(1, 1, True, .9), self.record(2, 1, False, .1),
                   self.record(2, 2, True, .6), self.record(0, 4, False, .1)]
        records.append(dict(records[0], visits=1001))
        manager = self.manager(records=records)
        self.assertTrue(callable(getattr(manager, 'adaptation', None)), 'Discovery must expose an action adaptation summary')
        summary = manager.adaptation(DISCOVERY_ID)
        self.assertEqual(summary['suggested_action_ids'], [2, 1, 0])
        self.assertEqual([row['action_id'] for row in summary['actions']], [2, 1, 0, 4, 3])
        row = next(row for row in summary['actions'] if row['action_id'] == 0)
        self.assertEqual((row['evaluated_inputs'], row['failure_count'], row['failure_rate']), (2, 1, .5))
        self.assertAlmostEqual(row['mean_severity'], .5)
        self.assertAlmostEqual(row['mean_reward'], .5)
        self.assertEqual((row['preference_mean'], row['preference_std']), (.2, .1))
        self.assertEqual(row['example_input_index'], 0)
        unvisited = summary['actions'][-1]
        self.assertEqual(unvisited['action']['darkness'], .4)
        self.assertEqual((unvisited['evaluated_inputs'], unvisited['failure_count']), (0, 0))
        for name in ('failure_rate', 'mean_severity', 'mean_reward', 'example_input_index'):
            self.assertIsNone(unvisited[name])
        self.assertEqual(summary['input_count'], 3)
        self.assertIsNone(summary['max_examples'])
        self.assertTrue(summary['defaults']['include_originals'])

    def test_summary_requires_finished_measurements_and_attached_analysis(self):
        manager = self.manager()
        self.assertTrue(callable(getattr(manager, 'adaptation', None)), 'Adaptation summary is required')
        for status, records in [('running', manager.jobs[DISCOVERY_ID]['records']), ('completed', [])]:
            manager.jobs[DISCOVERY_ID].update(status=status, records=records)
            with self.assertRaises(ValueError):
                manager.adaptation(DISCOVERY_ID)
        manager.jobs[DISCOVERY_ID].update(status='cancelled', records=[self.record(0, 0, True, .8)])
        self.assertEqual(manager.adaptation(DISCOVERY_ID)['suggested_action_ids'], [0])
        with patch.object(self.analyses, 'discovery_context', side_effect=ValueError('Analyze again: model changed.')):
            with self.assertRaisesRegex(ValueError, 'Analyze again'):
                manager.adaptation(DISCOVERY_ID)

    def test_classification_expands_all_inputs_keeps_ground_truth_and_clean_originals(self):
        manager = self.manager()
        result = self.start(manager, action_ids=[1, 4])
        task, examples, settings = self.captured[0]
        self.assertEqual(task, 'classification')
        self.assertEqual(len(examples), 9)
        self.assertEqual([row['target'] for row in examples], ['cat'] * 3 + [1] * 3 + ['dog'] * 3)
        red_examples = [row['input'].getpixel((0, 0)) for row in examples[:3]]
        self.assertCountEqual(red_examples, [(255, 0, 0), (204, 0, 0), (127, 0, 0)])
        self.assertEqual(settings['seed'], 17)
        self.assertEqual(self.analyses.released, 1)
        report = json.loads((manager.directory / result['id'] / 'report.json').read_text())
        selection = report['adaptation_selection']
        self.assertEqual(selection['action_ids'], [1, 4])
        self.assertEqual((selection['input_count'], selection['example_count']), (3, 9))
        self.assertTrue(selection['include_originals'])
        self.assertEqual([row['action']['id'] for row in selection['actions']], [1, 4])
        self.assertIsNone(selection['actions'][0]['failure_rate'])
        self.assertNotIn('examples', report)
        self.assertEqual(report['training_plan']['example_count'], 9)
        self.assertEqual(report['base_targets'], ['cat', 1, 'dog'])
        self.assertNotIn('recommended', json.dumps(report).lower())

    def test_disabling_clean_examples_keeps_every_input(self):
        manager = self.manager()
        result = self.start(manager, action_ids=[4], include_originals=False)
        self.assertEqual([row['target'] for row in self.captured[0][1]], ['cat', 1, 'dog'])
        self.assertEqual(result['adaptation_selection']['example_count'], 3)

    def test_invalid_action_selection_never_launches(self):
        manager = self.manager()
        for config in ({'action_ids': []}, {'action_ids': [0, 0]}, {'action_ids': [True]},
                       {'action_ids': [99]}, {'action_ids': ['0']}, {'action_ids': [0], 'examples': []},
                       {'action_ids': [0], 'include_originals': 1}, {'action_ids': [0], 'training_data': []}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                manager.start_finetune({'discovery_id': DISCOVERY_ID, **config})
        self.assertFalse(manager.finetunes)
        manager.jobs[DISCOVERY_ID]['config']['label_map'].pop('1')
        with self.assertRaisesRegex(ValueError, 'label|target'):
            manager.start_finetune({'discovery_id': DISCOVERY_ID, 'action_ids': [0]})

    def test_summary_defaults_and_optional_corpus_apply_each_selected_action(self):
        manager = self.manager('summarization')
        self.start(manager, action_ids=[2])
        self.assertEqual(self.captured[0][1], [
            {'input': 'The FILTER is Working.', 'target': 'The filter works.'},
            {'input': 'the filter is working.', 'target': 'The filter works.'}])
        result = self.start(manager, action_ids=[2, 3], training_data=[
            {'text': 'A FILTER is Working.', 'summary': 'It works.', 'action_summaries': {'3': 'It does not work.'}},
            {'text': 'WATER is Clear.', 'summary': 'It is clear.', 'action_summaries': {'3': 'It is not clear.'}}])
        self.assertEqual(self.captured[1][1], [
            {'input': 'A FILTER is Working.', 'target': 'It works.'},
            {'input': 'a filter is working.', 'target': 'It works.'},
            {'input': 'A FILTER is not Working.', 'target': 'It does not work.'},
            {'input': 'WATER is Clear.', 'target': 'It is clear.'},
            {'input': 'water is clear.', 'target': 'It is clear.'},
            {'input': 'WATER is not Clear.', 'target': 'It is not clear.'}])
        self.assertEqual((result['adaptation_selection']['input_count'], result['adaptation_selection']['example_count']), (2, 6))

    def test_text_randomness_is_reproducible_and_seed_is_recorded(self):
        manager = self.manager('summarization')
        self.start(manager, action_ids=[4], include_originals=False, seed=21)
        self.start(manager, action_ids=[4], include_originals=False, seed=21)
        self.assertEqual(self.captured[0][1], self.captured[1][1])
        self.assertNotEqual(self.captured[0][1][0]['input'], 'The FILTER is Working.')

    def test_summary_requires_corrected_negation_targets_and_valid_corpus(self):
        manager = self.manager('summarization')
        invalid = [({'action_ids': [3]}, 'action_summaries'),
                   ({'action_ids': [2], 'training_data': []}, 'training'),
                   ({'action_ids': [2], 'training_data': [{'text': 'source'}]}, 'summary'),
                   ({'action_ids': [2], 'training_data': [{'text': 'x' * 20001, 'summary': 'target'}]}, '20,000'),
                   ({'action_ids': [2], 'training_data': [{'text': 'source', 'summary': 'x' * 50001}]}, '50,000'),
                   ({'action_ids': [2], 'training_data': [{'text': 'source', 'summary': 'target', 'action_summaries': {'99': 'other'}}]}, 'action')]
        for config, message in invalid:
            with self.subTest(config={key: type(value).__name__ for key, value in config.items()}), self.assertRaisesRegex(ValueError, message):
                manager.start_finetune({'discovery_id': DISCOVERY_ID, **config})
        self.assertFalse(manager.finetunes)

    def test_millions_of_examples_use_compact_plan_and_indexed_targets(self):
        from adaptation import prepare
        actions = [dict(id=index, rotation=0, saturation=1, darkness=.1) for index in range(125)]
        context = {'preview': {'input_count': 50000, 'actions': actions,
                               'config': {'task': 'classification'}}}
        discovery = dict(id=DISCOVERY_ID, embedding_id=EMBEDDING_ID, task='classification',
                         records=[self.record(0, 0, True, .8)],
                         config={'label_map': {str(index): index for index in range(50000)}})
        tracemalloc.start()
        try:
            specification = prepare(discovery, context, {'action_ids': list(range(125))})
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 20_000_000, 'Planning must not allocate the 6.3 million input/action recipes')
        self.assertEqual(specification['adaptation_selection']['example_count'], 6300000)
        self.assertLess(len(json.dumps(specification)), 1_000_000)
        self.assertNotIn('examples', specification)

        from adaptation import TrainingExamples
        loaded = []
        def load(context, input_index, action_id):
            loaded.append((input_index, action_id))
            return input_index, action_id
        examples = TrainingExamples(dict(task='classification', settings={'seed': 17}, **specification),
                                    context, load)
        self.assertIsInstance(examples, Sequence)
        self.assertEqual(len(examples), 6300000)
        self.assertEqual([examples.target(index) for index in (0, 125, 126, -1)], [0, 0, 1, 49999])
        self.assertEqual(loaded, [], 'Target validation must not load training inputs')
        for index, expected in [(0, (0, None)), (1, (0, 0)), (125, (0, 124)),
                                (126, (1, None)), (-1, (49999, 124))]:
            self.assertEqual(examples[index]['input'], expected)
        self.assertEqual(len(loaded), 5)
        for index in (6300000, -6300001):
            with self.assertRaises(IndexError):
                examples[index]

    def test_large_corpus_trains_every_example_with_exact_counts_and_compact_report(self):
        manager = self.manager('summarization')
        observed = []
        def train(model, task, examples, directory, settings, progress, cancelled):
            self.assertIsInstance(examples, Sequence)
            self.assertNotIsInstance(examples, list)
            self.assertEqual(len(examples), 5000)
            self.assertEqual(examples.target(4999), 'Reference 2499')
            self.assertEqual(examples[4999], {'input': 'source 2499', 'target': 'Reference 2499'})
            for epoch in range(2):
                rows = [(row['input'], row['target']) for row in examples]
                self.assertEqual(len(set(rows)), 5000)
                observed.append(len(rows))
            progress(completed=40, total=40, effective_batch_size=256, micro_batch_size=64,
                     oom_retries=2, epoch=2, epochs=2, processed_examples=10000,
                     examples_per_epoch=5000, total_examples=10000, loss=.2)
            checkpoint = directory / 'checkpoint'
            checkpoint.mkdir()
            return dict(checkpoint=str(checkpoint), steps=40, training_loss=.2)
        manager.finetune_fn = train
        corpus = [{'text': f'SOURCE {index}', 'summary': f'Reference {index}'} for index in range(2500)]
        with self.analyses.model_lock:
            job = manager.start_finetune(dict(discovery_id=DISCOVERY_ID, action_ids=[2],
                                             training_data=corpus, epochs=2))
            self.assertIsNone(job['total'])
            self.assertEqual(job['settings']['batch_size'], 'auto')
            self.assertEqual((job['example_count'], job['examples_per_epoch'], job['total_examples']),
                             (5000, 5000, 10000))
        result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(observed, [5000, 5000])
        report = json.loads((manager.directory / job['id'] / 'report.json').read_text())
        self.assertNotIn('examples', report)
        self.assertEqual(len(report['training_data']), 2500)
        for key, expected in dict(completed=40, total=40, effective_batch_size=256, micro_batch_size=64,
                                  oom_retries=2, epoch=2, epochs=2, processed_examples=10000,
                                  examples_per_epoch=5000, total_examples=10000).items():
            self.assertEqual(result[key], expected, key)
            self.assertEqual(report[key], expected, key)

    def test_training_transforms_validated_originals_without_populating_action_png_cache(self):
        manager = self.manager()
        preview = self.analyses.context['preview']
        self.previews.original_image(preview['id'], 0)
        with patch.object(self.previews, 'image', side_effect=AssertionError('Training must not render cached previews')):
            self.start(manager, action_ids=[1, 4])
        rows = self.captured[0][1]
        self.assertEqual([row['input'].getpixel((0, 0)) for row in rows[:3]],
                         [(255, 0, 0), (204, 0, 0), (127, 0, 0)])
        self.assertEqual(list((self.previews.directory / preview['id']).glob('*/action-*.png')), [])

    def test_changed_source_image_requires_reload_before_training(self):
        manager = self.manager()
        preview = self.analyses.context['preview']
        self.previews.original_image(preview['id'], 0)
        Image.new('RGB', (8, 8), 'white').save(self.directory / 'images' / '0.png')
        job = manager.start_finetune(dict(discovery_id=DISCOVERY_ID, action_ids=[1], include_originals=True))
        result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'error')
        self.assertRegex(result['error'], 'source changed.*Analyze again')
        self.assertEqual(self.captured, [])
        self.assertIsNone(result['checkpoint'])

    def test_thousand_image_dataset_trains_all_four_thousand_examples_without_action_pngs(self):
        manager = self.manager(input_count=1000)
        reads = []
        original = self.previews.original_image
        def load(identity, index):
            reads.append(index)
            return original(identity, index)
        def train(model, task, examples, directory, settings, progress, cancelled):
            self.assertEqual(len(examples), 4000)
            self.assertEqual(examples.target(3999), 0)
            self.assertEqual(reads, [])
            for index, row in enumerate(examples):
                self.assertIsInstance(row['input'], Image.Image)
                if index < 12:
                    self.assertEqual(row['target'], ('cat', 1, 'dog')[index // 4])
                else:
                    self.assertEqual(row['target'], (index // 4) % 3)
            self.assertEqual(reads, [index for index in range(1000) for _ in range(4)])
            progress(completed=63, total=63, effective_batch_size=64, micro_batch_size=8,
                     oom_retries=0, epoch=1, epochs=1, processed_examples=4000,
                     examples_per_epoch=4000, total_examples=4000)
            checkpoint = directory / 'checkpoint'
            checkpoint.mkdir()
            return dict(checkpoint=str(checkpoint), steps=63, training_loss=.2)
        manager.finetune_fn = train
        with patch.object(self.previews, 'original_image', side_effect=load):
            result = self.start(manager, action_ids=[0, 1, 2], batch_size=64)
        self.assertEqual((result['example_count'], result['processed_examples'], result['total']), (4000, 4000, 63))
        self.assertEqual(list(self.previews.directory.glob('*/*/action-*.png')), [])
        report = json.loads((manager.directory / result['id'] / 'report.json').read_text())
        self.assertEqual(len(report['base_targets']), 1000)
        self.assertNotIn('examples', report)

    def test_auto_and_large_manual_batches_are_accepted_and_invalid_values_rejected(self):
        manager = self.manager('summarization')
        for batch_size in ('auto', 1, 33, 1024):
            result = self.start(manager, action_ids=[2], batch_size=batch_size)
            self.assertEqual(result['settings']['batch_size'], batch_size)
        for batch_size in (0, 1025, None, True, 1.5, '32', [], {}):
            with self.subTest(batch_size=batch_size), self.assertRaisesRegex(ValueError, '[Bb]atch'):
                manager.start_finetune(dict(discovery_id=DISCOVERY_ID, action_ids=[2], batch_size=batch_size))

    def test_corpus_payload_limit_remains_separate_from_example_count(self):
        manager = self.manager('summarization')
        with patch('adaptation.MAX_UPLOAD_BYTES', 256), self.assertRaisesRegex(ValueError, '20 MB'):
            manager.start_finetune(dict(discovery_id=DISCOVERY_ID, action_ids=[2], training_data=[
                {'text': 'source' * 100, 'summary': 'reference'}]))
        self.assertFalse(manager.finetunes)

    def test_corpus_payload_size_matches_compact_utf8_browser_json(self):
        from adaptation import MAX_UPLOAD_BYTES, prepare
        manager = self.manager('summarization')
        corpus = [{'text': '猫' * 18000, 'summary': 'Reference'} for _ in range(200)]
        wire_size = len(json.dumps(corpus, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        self.assertLess(wire_size, MAX_UPLOAD_BYTES)
        self.assertGreater(len(json.dumps(corpus).encode('utf-8')), MAX_UPLOAD_BYTES)
        specification = prepare(manager.jobs[DISCOVERY_ID], self.analyses.context,
                                {'action_ids': [2], 'training_data': corpus})
        self.assertEqual(specification['adaptation_selection']['example_count'], 400)
        self.assertEqual(specification['training_data'][-1], corpus[-1])

        # A payload exactly at the byte boundary must not be rejected for the
        # whitespace that Python's default serializer inserts between fields.
        ascii_corpus = [{'text': 'Source', 'summary': 'Reference'}]
        compact_size = len(b'[{"text":"Source","summary":"Reference"}]')
        with patch('adaptation.MAX_UPLOAD_BYTES', compact_size):
            specification = prepare(manager.jobs[DISCOVERY_ID], self.analyses.context,
                                    {'action_ids': [2], 'training_data': ascii_corpus})
        self.assertEqual(specification['adaptation_selection']['example_count'], 2)

    def test_cancellation_during_lazy_image_read_stops_before_returning_example(self):
        manager = self.manager()
        reads = []
        original = self.previews.original_image
        def load(identity, index):
            reads.append(index)
            result = original(identity, index)
            manager.cancel_finetune(manager.active)
            return result
        def train(model, task, examples, directory, settings, progress, cancelled):
            examples[0]
            self.fail('Cancelled image reads must not return a training example')
        manager.finetune_fn = train
        with patch.object(self.previews, 'original_image', side_effect=load):
            job = manager.start_finetune(dict(discovery_id=DISCOVERY_ID, action_ids=[0]))
            result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'cancelled', result.get('error'))
        self.assertEqual(reads, [0])
        self.assertIsNone(result['checkpoint'])

    def test_generation_uses_only_desired_references_and_transformed_or_explicit_captions(self):
        manager = self.manager('generation')
        desired = self.directory / 'desired.png'
        Image.new('RGB', (8, 8), 'white').save(desired)
        result = self.start(manager, action_ids=[0, 1], training_data=[
            {'action_id': 0, 'image_path': str(desired)},
            {'action_id': 1, 'image_path': str(desired), 'prompt': 'My reviewed reference caption.'}])
        self.assertEqual(self.captured[0][1], [
            {'input': 'A unique scientist in a classroom.', 'target': str(desired)},
            {'input': 'My reviewed reference caption.', 'target': str(desired)}])
        self.assertFalse(result['adaptation_selection']['include_originals'])
        self.assertEqual(result['adaptation_selection']['example_count'], 2)

    def test_generation_rejects_missing_mismatched_or_nonimage_references(self):
        manager = self.manager('generation')
        desired = self.directory / 'desired.png'
        Image.new('RGB', (8, 8), 'white').save(desired)
        bad_image = self.directory / 'not-image.png'
        bad_image.write_text('not an image')
        for data in (None, [], [{'action_id': 0, 'image_path': str(desired)}],
                     [{'action_id': 2, 'image_path': str(desired)}],
                     [{'action_id': 0, 'image_path': 'relative.png'}],
                     [{'action_id': 0, 'image_path': str(bad_image)}]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                manager.start_finetune({'discovery_id': DISCOVERY_ID, 'action_ids': [0, 1], 'training_data': data})
        with self.assertRaisesRegex(ValueError, 'original'):
            manager.start_finetune({'discovery_id': DISCOVERY_ID, 'action_ids': [0], 'include_originals': True,
                                   'training_data': [{'action_id': 0, 'image_path': str(desired)}]})
        self.assertFalse(manager.finetunes)

    def test_generation_rejects_corrupt_or_oversized_images_before_launch(self):
        manager = self.manager('generation')
        corrupted = self.directory / 'corrupted.png'
        Image.new('RGB', (8, 8), 'white').save(corrupted)
        data = bytearray(corrupted.read_bytes())
        data[data.index(b'IDAT') + 4] ^= 1
        corrupted.write_bytes(data)
        oversized = self.directory / 'oversized.png'
        Image.new('L', (5000, 4001)).save(oversized)
        for path in (corrupted, oversized):
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                manager.start_finetune({'discovery_id': DISCOVERY_ID, 'action_ids': [0],
                                       'training_data': [{'action_id': 0, 'image_path': str(path)}]})
        self.assertFalse(manager.finetunes)

    def test_queued_adaptation_checks_model_identity_and_supports_cancellation(self):
        manager = self.manager()
        with self.analyses.model_lock:
            job = manager.start_finetune({'discovery_id': DISCOVERY_ID, 'action_ids': [0]})
            self.analyses.context['model_key'] = 'changed'
        result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'error')
        self.assertIn('model changed', result['error'])
        self.assertFalse(self.captured)
        with self.analyses.model_lock:
            job = manager.start_finetune({'discovery_id': DISCOVERY_ID, 'action_ids': [0]})
            manager.cancel_finetune(job['id'])
            result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'cancelled')
        self.assertFalse(self.captured)

    def test_http_summary_and_action_training_use_real_routes(self):
        manager = self.manager('summarization')
        with patch('server.ThreadingHTTPServer', MemoryServer):
            self.server = create_server(8000, self.directory / 'http' / 'previews')
        # The handler retains this manager; install its analyzed fixture and trainer.
        http_manager = self.server.discoveries
        http_manager.embeddings = self.analyses
        http_manager.previews = self.previews
        def train(model, task, examples, directory, settings, progress, cancelled):
            progress(device='cuda', device_name='Test GPU')
            return self.train(model, task, examples, directory, settings, progress, cancelled)
        http_manager.finetune_fn = train
        http_manager.jobs = manager.jobs
        code, body = test_http.HTTPTests.request(self, '/api/discoveries/' + DISCOVERY_ID + '/adaptation')
        self.assertEqual(code, 200, body)
        summary = json.loads(body)
        self.assertEqual(summary['suggested_action_ids'], [2])
        self.assertEqual(summary['defaults']['source_text'], 'The FILTER is Working.')
        code, body = test_http.HTTPTests.request(self, '/api/finetunes', 'POST', body={
            'discovery_id': DISCOVERY_ID, 'action_ids': [2], 'include_originals': False, 'device': 'cuda'})
        self.assertEqual(code, 202, body)
        job = json.loads(body)
        result = self.wait(lambda: http_manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(result.get('device'), 'cuda')
        self.assertEqual(result.get('device_name'), 'Test GPU')
        self.assertEqual(self.captured[-1][2]['device'], 'cuda')
        self.assertEqual(result['adaptation_selection']['example_count'], 1)
        self.assertIn('report', job.get('artifacts', {}), 'Accepted training jobs must publish their report link')
        report_url = result['artifacts']['report']
        code, body = test_http.HTTPTests.request(self, report_url)
        self.assertEqual(code, 200, body)
        report = json.loads(body)
        self.assertEqual(report['id'], result['id'])
        self.assertEqual(report['settings']['device'], 'cuda')
        self.assertEqual(report['device'], 'cuda')
        self.assertEqual(report['device_name'], 'Test GPU')
        self.assertEqual(report['adaptation_selection']['action_ids'], [2])
        self.assertEqual(report['adaptation_selection']['example_count'], 1)
        self.assertEqual(report['adaptation_selection']['actions'][0]['failure_count'], 1)
        for filename in ('policy.pt', 'episodes.jsonl', 'reference.png', 'model.safetensors'):
            (http_manager.directory / result['id'] / filename).write_bytes(b'private training artifact')
            code, _ = test_http.HTTPTests.request(self, report_url.replace('report.json', filename))
            self.assertIn(code, (400, 404), filename)
        self.assertEqual(test_http.HTTPTests.request(self, '/discoveries/' + 'f' * 32 + '/artifacts/report.json')[0], 404)
        report_path = http_manager.directory / result['id'] / 'report.json'
        report_path.unlink()
        private = self.directory / 'private.json'
        private.write_text('{"private": true}')
        report_path.symlink_to(private)
        self.assertEqual(test_http.HTTPTests.request(self, report_url)[0], 404)
        self.assertEqual(test_http.HTTPTests.request(self, '/api/discoveries/' + 'f' * 32 + '/adaptation')[0], 404)


if __name__ == '__main__':
    unittest.main()
