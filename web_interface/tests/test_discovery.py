"""Offline model boundary exercises live workers, persistence and reviewed training."""
import copy
import json
import os
import tempfile
import threading
import time
import unittest
from collections.abc import Sequence
from unittest.mock import patch
from pathlib import Path

import numpy as np

from discovery import DiscoveryManager


EMBEDDING_ID = 'a'*32
PREVIEW_ID = 'b'*32


class Analyses:
    def __init__(self):
        self.model_lock = threading.RLock()
        self.released = 0
        self.node_requests = []
        self.context = dict(
            job={'id': EMBEDDING_ID, 'status': 'completed', 'preview_id': PREVIEW_ID},
            model={'source': 'huggingface', 'name': 'fixture/model', 'task': 'summarization'},
            model_key='unchanged', states=np.array([[.6, .8]]),
            preview={'id': PREVIEW_ID, 'status': 'completed', 'input_count': 1,
                     'config': {'task': 'summarization', 'text': 'original'},
                     'original': {'text': 'original'},
                     'actions': [{'id': 4}, {'id': 9}],
                     'nodes': [{'id': 4, 'text': 'good'}, {'id': 9, 'text': 'bad'}]})

    def discovery_context(self, identity, *, include_states=True):
        if identity != EMBEDDING_ID:
            raise KeyError(identity)
        result = copy.deepcopy({key: value for key, value in self.context.items() if key != 'states'})
        if include_states:
            result['states'] = self.context['states']
        return result

    def release_encoder(self):
        self.released += 1

    def wait_for_expansion(self, identity, cancelled):
        pass

    def discovery_node(self, identity, index, action, evaluator=None):
        self.node_requests.append((identity, index, action))
        return {'id': action, 'action': {'name': 'fixture'}, 'text': 'variant',
                'position': [action / 10, 0, 0], 'similarity': .9, 'neighbors': []}


class Evaluator:
    def __init__(self, gate=None, fail_at=None):
        self.gate, self.fail_at = gate, fail_at
        self.calls = []
        self.closed = False
        self.entered = threading.Event()

    def predict(self, value, **kwargs):
        self.calls.append((value, kwargs))
        self.entered.set()
        if self.gate and not self.gate.wait(5):
            raise RuntimeError('test inference release timeout')
        if len(self.calls) == self.fail_at:
            raise ValueError('fixture inference failed')
        return {'summary': value, 'probabilities': [.2, .8]}

    def score(self, output, baseline, *, target=None):
        bad = output['summary'] == 'bad'
        return dict(reward=float(bad), severity=float(bad), failure=bad,
                    criterion='review signal', metrics={'quality': float(not bad)})

    def describe(self):
        return {'metric': 'fixture actual calls'}

    def close(self):
        self.closed = True


class DiscoveryTests(unittest.TestCase):
    def test_context_shares_float32_states_and_metadata_only_context_needs_no_vectors(self):
        original = np.array([[.6, .8]], dtype=np.float32)
        self.embeddings.context['states'] = original
        manager = self.manager()
        context = manager._context(EMBEDDING_ID)
        self.assertTrue(np.shares_memory(context['states'], original))
        self.assertFalse(context['states'].flags.writeable)
        with self.assertRaises(ValueError):
            context['states'][0, 0] = 100
        self.assertTrue(original.flags.writeable)
        del self.embeddings.context['states']
        metadata = manager._context(EMBEDDING_ID, include_states=False)
        self.assertNotIn('states', metadata)
        self.assertEqual(metadata['preview']['id'], PREVIEW_ID)

    def test_discovery_transforms_directly_without_writing_preview_images(self):
        from PIL import Image
        from unittest.mock import Mock
        from core import apply_image
        action = {'id': 4, 'rotation': 20, 'saturation': .5, 'darkness': .3}
        original = Image.new('RGB', (16, 16), 'red')
        manager = self.manager()
        manager.previews = Mock()
        manager.previews.original_image.return_value = original
        manager.previews.image.side_effect = AssertionError('Encoded preview used for model input')
        context = {'preview': {'id': PREVIEW_ID, 'config': {'task': 'classification'}, 'actions': [action]}}
        np.testing.assert_array_equal(manager._input(context, 0, 4), apply_image(original, action))
        self.assertIs(manager._input(context, 0), original)
        manager.previews.image.assert_not_called()

    def test_live_policy_scans_are_throttled_and_final_predictions_include_every_input(self):
        manager = self.manager()
        with patch.object(manager, '_policy_statistics', wraps=manager._policy_statistics) as scans:
            result = self.run_job(manager)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertLess(scans.call_count, 5, 'A fast RL loop must not rescan all inputs every ten episodes')
        self.assertTrue(scans.call_args.kwargs['persist'])
        self.assertEqual(result['predicted_landscape']['policy_updates'], 25)

    def test_large_live_landscape_uses_frozen_policy_without_full_matrix_and_final_file_matches(self):
        from rl_policy import TerminalDQN
        self.embeddings.context['states'] = np.tile(np.array([[.6, .8]], dtype=np.float32), (300, 1))
        self.embeddings.context['preview']['input_count'] = 300
        manager = self.manager()
        context = manager._context(EMBEDDING_ID)
        identity = 'e' * 32
        (self.directory / identity).mkdir()
        manager.jobs[identity] = {'id': identity, 'predicted_landscape': None}
        policy = TerminalDQN(2, 2, batch_size=2)
        policy.observe([.6, .8], 0, 1.)
        policy.observe([.6, .8], 0, 1.)
        with patch.object(manager, 'PREDICTION_RAM_BYTES', 1):
            manager._policy_statistics(identity, policy, context, [])
        self.assertNotIn(identity, manager._prediction_values)
        before = manager.predictions(identity, 299)
        policy.observe([.6, .8], 1, 0.)
        self.assertEqual(manager.predictions(identity, 299), before, 'Live predictions must use one frozen version')
        manager._policy_statistics(identity, policy, context, [], persist=True)
        final = manager.predictions(identity, 299)
        values = np.load(manager.artifact(identity, 'predictions-2.npy'))
        np.testing.assert_allclose(values[299], [action['q_value'] for action in final['actions']])
        self.assertNotIn(identity, manager._prediction_policies)
        self.assertFalse(list((self.directory / identity).glob('*.tmp')))

    def test_training_options_validation_and_cleanup_follow_saved_success_only(self):
        from unittest.mock import Mock
        def train(model, task, examples, directory, settings, progress, cancelled):
            self.assertEqual(settings['weight_decay'], .03)
            self.assertEqual(settings['max_grad_norm'], 0.)
            self.assertEqual(settings['seed'], 103)
            checkpoint = directory / 'checkpoint'
            checkpoint.mkdir()
            (checkpoint / 'model.bin').write_bytes(b'trained weights')
            return {'checkpoint': str(checkpoint), 'steps': 1, 'training_loss': .2}
        manager = self.manager(finetune=train)
        discovery = self.run_job(manager)
        action = discovery['records'][0]['action_id']
        request = dict(discovery_id=discovery['id'], examples=[{'input_index': 0, 'action_id': action, 'target': 'reviewed'}],
                       weight_decay=.03, max_grad_norm=0., seed=103)
        for fields in ({'weight_decay': -1}, {'weight_decay': True}, {'max_grad_norm': float('nan')},
                       {'max_grad_norm': 101}, {'cleanup_after_finetune': 'true'}):
            with self.assertRaises(ValueError):
                manager.start_finetune(request | fields)
        cleanup_calls = []
        def release(identity):
            job = next(reversed(manager.finetunes.values()))
            report = json.loads((manager.directory / job['id'] / 'report.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertTrue((Path(report['checkpoint']) / 'model.bin').exists())
            cleanup_calls.append(identity)
            return {'status': 'released', 'released_memory_bytes': 8, 'released_disk_bytes': 16}
        self.embeddings.release_analysis = release
        manager.previews = Mock()
        manager.previews.pin_preview = None
        manager.previews.release_preview.return_value = {'status': 'released', 'released_memory_bytes': 20, 'released_disk_bytes': 0}
        training = manager.start_finetune(request)
        result = self.wait(lambda: manager.get_finetune(training['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(cleanup_calls, [EMBEDDING_ID])
        self.assertEqual(result['cleanup']['status'], 'completed')
        self.assertEqual(result['cleanup']['released_memory_bytes'], 28)
        self.assertTrue((Path(result['checkpoint']) / 'model.bin').exists())
        self.assertTrue(manager.artifact(discovery['id'], 'policy.pt').exists())
        second = manager.start_finetune(request | {'cleanup_after_finetune': False})
        self.assertEqual(self.wait(lambda: manager.get_finetune(second['id']))['cleanup']['status'], 'disabled')
        self.assertEqual(cleanup_calls, [EMBEDDING_ID])
        manager.finetune_fn = lambda *args: (_ for _ in ()).throw(ValueError('training failed'))
        failed = manager.start_finetune(request)
        self.assertEqual(self.wait(lambda: manager.get_finetune(failed['id']))['status'], 'error')
        self.assertEqual(cleanup_calls, [EMBEDDING_ID])

    def test_prediction_quota_failure_keeps_learned_policy_and_measured_report(self):
        manager = self.manager()
        with patch.dict(os.environ, {'FAILURESHIFT_PREDICTION_DISK_MB': '0'}):
            result = self.run_job(manager)
        self.assertEqual(result['status'], 'error')
        self.assertIn('budget', result['error'])
        self.assertEqual(result['episode'], 40)
        self.assertTrue(manager.artifact(result['id'], 'policy.pt').is_file())
        report = json.loads(manager.artifact(result['id'], 'report.json').read_text())
        self.assertEqual(report['records'], result['records'])
        self.assertFalse(list((self.directory / result['id']).glob('predictions-*')))

    def test_cancelled_training_and_failed_report_save_never_release_working_data(self):
        from unittest.mock import Mock
        manager = self.manager()
        discovery = self.run_job(manager)
        action = discovery['records'][0]['action_id']
        request = dict(discovery_id=discovery['id'], examples=[{'input_index': 0, 'action_id': action, 'target': 'reviewed'}])
        self.embeddings.release_analysis = Mock()
        def train(model, task, examples, directory, settings, progress, cancelled):
            checkpoint = directory / 'checkpoint'
            checkpoint.mkdir()
            return {'checkpoint': str(checkpoint), 'steps': 1, 'training_loss': .2}
        def cancel_train(*args):
            result = train(*args)
            manager.cancel_finetune(manager.active)
            return result
        manager.finetune_fn = cancel_train
        job = manager.start_finetune(request)
        self.assertEqual(self.wait(lambda: manager.get_finetune(job['id']))['status'], 'cancelled')
        manager.finetune_fn = train
        persist = manager._persist
        def fail_final_report(identity):
            if identity in manager.finetunes and manager.finetunes[identity]['status'] == 'completed':
                raise OSError('report storage unavailable')
            persist(identity)
        with patch.object(manager, '_persist', side_effect=fail_final_report):
            job = manager.start_finetune(request)
            result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'error')
        self.assertIn('report storage unavailable', result['error'])
        self.embeddings.release_analysis.assert_not_called()

    def test_predictions_cover_unvisited_pairs_without_inference_and_survive_restart(self):
        context = self.embeddings.context
        context['states'] = np.tile([[.6, .8]], (50, 1))
        context['preview']['input_count'] = 50
        manager = self.manager()
        result = self.run_job(manager)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertIn('predicted_landscape', result)
        landscape = result['predicted_landscape']
        self.assertEqual(len(landscape['input_mean_reward']), 50)
        self.assertEqual(landscape['action_ids'], [4, 9])
        tested = {record['input_index'] for record in result['records']}
        untested = next(index for index in range(50) if index not in tested)
        calls = len(self.evaluators[0].calls)
        prediction = manager.predictions(result['id'], untested)
        self.assertEqual(prediction['input_index'], untested)
        self.assertEqual(prediction['policy_updates'], landscape['policy_updates'])
        self.assertEqual([item['action_id'] for item in prediction['actions']], [4, 9])
        self.assertAlmostEqual(sum(item['q_value'] for item in prediction['actions']) / 2,
                               landscape['input_mean_reward'][untested], places=6)
        self.assertAlmostEqual(sum(item['preference'] for item in prediction['actions']), 1., places=6)
        self.assertEqual(len(self.evaluators[0].calls), calls)
        self.assertEqual(manager.get(result['id'])['unique_evaluations'], result['unique_evaluations'])
        restored = DiscoveryManager(None, None, manager.directory)
        self.assertEqual(restored.predictions(result['id'], untested), prediction)
        for index in (-1, 50, True, '0'):
            with self.assertRaises(ValueError):
                manager.predictions(result['id'], index)

    def test_short_discovery_has_no_trained_landscape(self):
        manager = self.manager()
        job = manager.start(dict(embedding_id=EMBEDDING_ID, episodes=2, reference_summary='reference'))
        result = self.wait(lambda: manager.get(job['id']))
        self.assertIn('predicted_landscape', result)
        self.assertIsNone(result['predicted_landscape'])
        with self.assertRaisesRegex(ValueError, 'trained'):
            manager.predictions(result['id'], 0)

    def test_full_dataset_classification_resolves_unique_typed_labels_once(self):
        from classification_labels import resolve_target
        context = self.embeddings.context
        context['preview']['config']['task'] = 'classification'
        context['states'] = np.tile([[.6, .8]], (50000, 1))
        values = [0, '0', 'cat', 'dog']
        context['job']['labels'] = {
            'label_map': {str(index): values[index % 4] for index in range(50000)},
            'model_labels': {'0': '1', '1': '0', '2': 'cat', '3': 'dog'}}
        with patch('discovery.resolve_target', wraps=resolve_target) as resolve:
            settings = self.manager()._config({}, context)
        self.assertEqual(resolve.call_count, 4)
        self.assertEqual(len(settings['label_map']), 50000)
        self.assertEqual({key: settings['label_map'][key] for key in ('0', '1', '2', '3')},
                         {'0': 0, '1': 1, '2': 2, '3': 3})

    def test_classification_resolution_cache_preserves_invalid_and_ambiguous_label_errors(self):
        context = self.embeddings.context
        context['preview']['config']['task'] = 'classification'
        context['states'] = np.array([[.6, .8], [.6, .8]])
        context['job']['labels'] = {'label_map': {'0': 0, '1': 1},
                                  'model_labels': {'0': 'Cat', '1': 'cat'}}
        manager = self.manager()
        for value, message in [('cat', 'Ambiguous'), ('unknown', 'match this model'),
                               (9, 'match this model'), (True, 'explicit model class IDs')]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                manager._config({'label_map': {'1': value}}, context)

    def test_classification_accepts_automatically_mapped_pet_categories_without_manual_labels(self):
        from classification_labels import match_dataset_labels
        from test_classification_labels import imagenet_labels
        context = self.embeddings.context
        context['preview']['config']['task'] = 'classification'
        context['job']['labels'] = match_dataset_labels({'values': {'0': 'cat'}}, imagenet_labels(), 1)
        settings = self.manager()._config({}, context)
        self.assertEqual(settings['label_map'], {'0': 'cat'})

    def test_classification_requires_reviewed_labels_and_uses_mapped_dataset_targets(self):
        context = self.embeddings.context
        context['preview']['config']['task'] = 'classification'
        context['job']['labels'] = {'label_map': {}, 'model_labels': {'0': 'dog', '4': 'cat'}}
        manager = self.manager()
        with self.assertRaisesRegex(ValueError, '1.*review'):
            manager.start({'embedding_id': EMBEDDING_ID})
        context['job']['labels']['label_map'] = {'0': 4}
        settings = manager._config({}, context)
        self.assertEqual(settings['label_map'], {'0': 4})
        self.assertEqual(manager._config({'label_map': {'0': 'dog'}}, context)['label_map'], {'0': 0})
        with self.assertRaisesRegex(ValueError, 'match this model'):
            manager._config({'label_map': {'0': 99}}, context)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.embeddings = Analyses()
        self.evaluators = []

    def manager(self, evaluator=None, finetune=None):
        def factory(model, config, directory, progress=None):
            actual = evaluator or Evaluator()
            self.evaluators.append(actual)
            return actual
        return DiscoveryManager(None, self.embeddings, self.directory,
                                evaluator_factory=factory, finetune_fn=finetune)

    def wait(self, getter):
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            result = getter()
            if result['status'] not in ('queued', 'running'):
                return result
            time.sleep(.01)
        self.fail('worker failed to reach a terminal status')

    def run_job(self, manager, **options):
        job = manager.start(dict(embedding_id=EMBEDDING_ID, episodes=40, seed=7,
                                 summary_metric='reference', reference_summary='reference', **options))
        return self.wait(lambda: manager.get(job['id']))

    def test_measured_pairs_caching_replay_and_persisted_reports(self):
        manager = self.manager()
        result = self.run_job(manager)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(result['episode'], 40)
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual(result['unique_evaluations'], 2)
        self.assertEqual({record['action_id'] for record in result['records']}, {4, 9})
        self.assertEqual(sum(record['visits'] for record in result['records']), 40)
        self.assertEqual(result['epsilon'], .6)
        self.assertIsNotNone(result['loss'])
        self.assertEqual(len(result['baseline']), 1)
        self.assertTrue(any(entry['cached'] for entry in result['history']))
        self.assertNotIn('probabilities', json.dumps(result))
        self.assertTrue(self.evaluators[0].closed)
        self.assertEqual(self.embeddings.released, 0,
                         'Read-only discovery must retain reusable Analyze weights')
        self.assertEqual(len(self.embeddings.node_requests), result['unique_evaluations'])
        for record in result['records']:
            self.assertEqual(record['node']['id'], record['action_id'])
            self.assertEqual(record['node']['position'], [record['action_id'] / 10, 0, 0])
        statistics = result['policy_statistics']
        self.assertEqual(statistics['input_count'], 1)
        self.assertEqual(statistics['policy_updates'], 25)
        self.assertEqual({action['action_id'] for action in statistics['actions']}, {4, 9})
        self.assertTrue(all(action['std'] is None for action in statistics['actions']))
        self.assertTrue(all(action['evaluated_inputs'] == 1 for action in statistics['actions']))
        report = json.loads(manager.artifact(result['id'], 'report.json').read_text())
        self.assertEqual(report['records'], result['records'])
        self.assertEqual(report['policy_statistics'], statistics)
        self.assertEqual(len(manager.artifact(result['id'], 'episodes.jsonl').read_text().splitlines()), 40)
        self.assertTrue(manager.artifact(result['id'], 'policy.pt').is_file())
        self.assertEqual(manager.list()[0]['id'], result['id'])
        with self.assertRaises((ValueError, FileNotFoundError)):
            manager.artifact(result['id'], '../report.json')

    def test_cancel_during_baseline_serializes_jobs_and_allows_retry(self):
        gate = threading.Event()
        evaluator = Evaluator(gate)
        manager = self.manager(evaluator)
        job = manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='reference', episodes=2))
        self.assertTrue(evaluator.entered.wait(5))
        with self.assertRaises(RuntimeError):
            manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='reference'))
        manager.cancel(job['id'])
        gate.set()
        result = self.wait(lambda: manager.get(job['id']))
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['episode'], 0)
        self.assertEqual(result['model_calls'], 1)
        self.assertTrue(evaluator.closed)
        self.assertTrue(manager.artifact(job['id'], 'report.json').is_file())
        self.assertEqual(self.run_job(manager)['status'], 'completed')

    def test_error_retains_partial_results_and_can_retry(self):
        evaluator = Evaluator(fail_at=3)
        manager = self.manager(evaluator)
        result = self.run_job(manager)
        self.assertEqual(result['status'], 'error')
        self.assertIn('fixture inference failed', result['error'])
        self.assertGreater(result['episode'], 0)
        self.assertEqual(result['model_calls'], 3)
        self.assertGreater(len(result['records']), 0)
        self.assertTrue(manager.artifact(result['id'], 'policy.pt').exists())
        evaluator.fail_at = None
        self.assertEqual(self.run_job(manager)['status'], 'completed')

    def test_projection_error_retains_the_completed_measurement(self):
        manager = self.manager()
        with patch.object(self.embeddings, 'discovery_node', side_effect=ValueError('projection failed')):
            result = self.run_job(manager)
        self.assertEqual(result['status'], 'error')
        self.assertIn('projection failed', result['error'])
        self.assertEqual(result['model_calls'], 2)
        self.assertEqual(result['unique_evaluations'], 1)
        self.assertEqual(result['episode'], 1)
        self.assertEqual(len(result['records']), 1)
        self.assertEqual(result['records'][0]['visits'], 1)
        self.assertNotIn('node', result['records'][0])
        report = json.loads(manager.artifact(result['id'], 'report.json').read_text())
        self.assertEqual(report['records'], result['records'])
        self.assertEqual(report['unique_evaluations'], 1)

    def test_cancel_during_final_cleanup_is_reflected_in_the_report(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        class PausingCleanup(Evaluator):
            def close(self):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('test cleanup release timeout')
                super().close()

        manager = self.manager(PausingCleanup())
        job = manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='reference', episodes=1))
        try:
            self.assertTrue(entered.wait(5))
            requested = manager.cancel(job['id'])
            self.assertEqual(requested['status'], 'running')
        finally:
            release.set()
        result = self.wait(lambda: manager.get(job['id']))
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['episode'], 1)
        self.assertEqual(len(result['records']), 1)
        report = json.loads(manager.artifact(result['id'], 'report.json').read_text())
        self.assertEqual(report['status'], 'cancelled')
        self.assertIsNone(manager.active)

    def test_explicit_cuda_setting_reaches_evaluation(self):
        devices = []
        def factory(model, config, directory, progress=None):
            devices.append(config['device'])
            return Evaluator()
        manager = DiscoveryManager(None, self.embeddings, self.directory, evaluator_factory=factory)
        result = self.run_job(manager, device='cuda')
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(devices, ['cuda'])

    def test_rejects_bad_config_before_loading_models(self):
        manager = self.manager()
        for bad in ({'episodes': True}, {'episodes': 0}, {'episodes': 10001}, {'seed': -1},
                    {'failure_threshold': float('nan')}, {'label_map': {'2': 1}},
                    {'generation_size': 33}, {'generation_steps': 101}, {'max_new_tokens': 513},
                    {'failure_threshold': 10**500},
                    {'device': 'cuda:99'}, {'unknown': 1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='reference', **bad))
        self.assertEqual(self.evaluators, [])

    def test_reviewed_targets_only_and_finetune_progress_checkpoint(self):
        captured = []
        def train(model, task, examples, directory, settings, progress, cancelled):
            captured.extend(examples)
            self.assertEqual(settings.get('device'), 'auto')
            self.assertFalse(cancelled())
            progress(completed=1, total=1, loss=.2, device='cuda', device_name='Test GPU')
            checkpoint = directory/'checkpoint'
            checkpoint.mkdir()
            return {'checkpoint': str(checkpoint), 'steps': 1, 'training_loss': .2}
        manager = self.manager(finetune=train)
        discovery = self.run_job(manager)
        action = discovery['records'][0]['action_id']
        for examples in ([{'input_index': 0, 'action_id': action}],
                         [{'input_index': 0, 'action_id': 500, 'target': 'reviewed'}],
                         [{'input_index': 0, 'action_id': action, 'target': ''}]):
            with self.assertRaises(ValueError):
                manager.start_finetune({'discovery_id': discovery['id'], 'examples': examples})
        valid_examples = [{'input_index': 0, 'action_id': action, 'target': 'desired'}]
        for settings in ({'batch_size': 1025}, {'learning_rate': .2}, {'examples': valid_examples*257},
                         {'device': 'gpu'}, {'device': None}, {'device': True}):
            with self.assertRaises(ValueError):
                manager.start_finetune(dict(discovery_id=discovery['id'], examples=valid_examples) | settings)
        job = manager.start_finetune(dict(discovery_id=discovery['id'], examples=[
            {'input_index': 0, 'action_id': action, 'target': 'reviewed desired output'}]))
        result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(captured[0]['target'], 'reviewed desired output')
        self.assertIn(captured[0]['input'], ('good', 'bad'))
        self.assertEqual(result['loss'], .2)
        self.assertEqual(result.get('device'), 'cuda')
        self.assertEqual(result.get('device_name'), 'Test GPU')
        self.assertEqual(result['settings'].get('device'), 'auto')
        self.assertTrue(Path(result['checkpoint']).is_dir())
        self.assertEqual(self.embeddings.released, 1,
                         'Fine-tuning must invalidate retained inference weights')

    def test_reviewed_training_total_counts_optimizer_batches(self):
        def train(model, task, examples, directory, settings, progress, cancelled):
            checkpoint = directory/'checkpoint'
            checkpoint.mkdir()
            return {'checkpoint': str(checkpoint), 'steps': 3, 'training_loss': .2}

        manager = self.manager(finetune=train)
        discovery = self.run_job(manager)
        examples = [dict(input_index=record['input_index'], action_id=record['action_id'], target='desired')
                    for record in discovery['records']]
        self.assertEqual(len(examples), 2)
        with self.embeddings.model_lock:
            job = manager.start_finetune(dict(discovery_id=discovery['id'], examples=examples,
                                              batch_size=2, epochs=3))
        result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(job['total'], 3)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(result['completed'], result['total'])

    def test_large_legacy_reviewed_requests_materialize_only_requested_examples(self):
        reads = []
        def train(model, task, examples, directory, settings, progress, cancelled):
            self.assertIsInstance(examples, Sequence)
            self.assertNotIsInstance(examples, list)
            self.assertEqual(len(examples), 3000)
            self.assertEqual(examples.target(2999), 'Reference 2999')
            self.assertEqual(reads, [], 'Neither job launch nor target validation should load inputs')
            self.assertEqual(examples[2999], {'input': 'bad', 'target': 'Reference 2999'})
            self.assertEqual(reads, [(1499, 9)])
            checkpoint = directory / 'checkpoint'
            checkpoint.mkdir()
            return {'checkpoint': str(checkpoint), 'steps': 3, 'training_loss': .2,
                    'effective_batch_size': 1024, 'micro_batch_size': 8, 'oom_retries': 1,
                    'epoch': 1, 'epochs': 1, 'processed_examples': 3000,
                    'examples_per_epoch': 3000, 'total_examples': 3000, 'total': 3}
        manager = self.manager(finetune=train)
        discovery = self.run_job(manager)
        self.embeddings.context['states'] = np.tile([[.6, .8]], (1500, 1))
        self.embeddings.context['preview']['input_count'] = 1500
        records = [dict(input_index=index, action_id=action)
                   for index in range(1500) for action in (4, 9)]
        manager.jobs[discovery['id']]['records'] = records
        reviewed = [dict(record, target=f'Reference {index}') for index, record in enumerate(records)]
        original = manager._input
        def load(context, index, action_id=None):
            reads.append((index, action_id))
            return original(context, index, action_id)
        with patch.object(manager, '_input', side_effect=load):
            job = manager.start_finetune(dict(discovery_id=discovery['id'], examples=reviewed, batch_size=1024))
            result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(result['total'], 3)
        self.assertEqual(result['example_count'], 3000)
        self.assertEqual(result['processed_examples'], 3000)
        self.assertEqual(result['micro_batch_size'], 8)
        self.assertEqual(result['oom_retries'], 1)
        report = json.loads((self.directory / job['id'] / 'report.json').read_text())
        self.assertEqual(len(report['examples']), 3000)
        self.assertEqual(report['examples'][-1], reviewed[-1])

    def test_reviewed_request_payload_size_is_validated_before_launch(self):
        manager = self.manager()
        discovery = self.run_job(manager)
        action = discovery['records'][0]['action_id']
        with patch('adaptation.MAX_UPLOAD_BYTES', 128), self.assertRaisesRegex(ValueError, '20 MB'):
            manager.start_finetune(dict(discovery_id=discovery['id'], examples=[
                dict(input_index=0, action_id=action, target='reference' * 20)]))
        self.assertFalse(manager.finetunes)

    def test_measured_graph_node_is_published_before_discovery_completes(self):
        entered, release = threading.Event(), threading.Event()

        class PausingEvaluator(Evaluator):
            def predict(self, value, **kwargs):
                if len(self.calls) == 2:
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('test inference release timeout')
                return super().predict(value, **kwargs)

        manager = self.manager(PausingEvaluator())
        job = manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='reference', episodes=40, seed=7))
        try:
            self.assertTrue(entered.wait(5))
            live = manager.get(job['id'])
            self.assertEqual(live['status'], 'running')
            self.assertTrue(live['records'])
            self.assertEqual(live['records'][0]['node']['id'], live['records'][0]['action_id'])
        finally:
            release.set()
            self.wait(lambda: manager.get(job['id']))

    def test_queued_model_change_is_rejected_before_loading(self):
        manager = self.manager()
        with self.embeddings.model_lock:
            job = manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='reference', episodes=2))
            self.embeddings.context['model_key'] = 'modified while queued'
        result = self.wait(lambda: manager.get(job['id']))
        self.assertEqual(result['status'], 'error')
        self.assertIn('model changed', result['error'])
        self.assertEqual(self.evaluators, [])

    def test_cancel_waiting_for_inference_lock_loads_no_models(self):
        manager = self.manager()
        with self.embeddings.model_lock:
            job = manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='reference'))
            manager.cancel(job['id'])
            result = self.wait(lambda: manager.get(job['id']))
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(self.evaluators, [])

    def test_report_creation_failure_releases_active_slot(self):
        manager = self.manager()
        with patch.object(manager, '_persist', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='reference'))
        self.assertIsNone(manager.active)
        self.assertEqual(self.run_job(manager)['status'], 'completed')

    def test_cancelled_finetune_retains_actual_progress_without_checkpoint(self):
        entered, release = threading.Event(), threading.Event()
        def train(model, task, examples, directory, settings, progress, cancelled):
            progress(completed=1, total=2, loss=.4)
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test training release timeout')
            if cancelled():
                raise InterruptedError('cancelled')
            self.fail('Cancellation callback was not signalled')
        manager = self.manager(finetune=train)
        discovery = self.run_job(manager)
        record = discovery['records'][0]
        job = manager.start_finetune(dict(discovery_id=discovery['id'], examples=[
            {'input_index': record['input_index'], 'action_id': record['action_id'], 'target': 'desired'}]))
        self.assertTrue(entered.wait(5))
        with self.assertRaises(RuntimeError):
            manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='reference'))
        manager.cancel_finetune(job['id'])
        release.set()
        result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['loss'], .4)
        self.assertIsNone(result['checkpoint'])
        self.assertEqual(json.loads((self.directory/job['id']/'report.json').read_text())['status'], 'cancelled')
        self.assertEqual(self.run_job(manager)['status'], 'completed')

    def test_failed_report_save_still_releases_slot_with_measured_results(self):
        manager = self.manager()
        with self.embeddings.model_lock:
            job = manager.start(dict(embedding_id=EMBEDDING_ID, reference_summary='reference', episodes=1))
            original = manager._persist
            manager._persist = lambda identity: (_ for _ in ()).throw(OSError('disk full'))
        result = self.wait(lambda: manager.get(job['id']))
        self.assertEqual(result['status'], 'error')
        self.assertIn('disk full', result['error'])
        self.assertEqual(result['episode'], 1)
        self.assertEqual(len(result['records']), 1)
        self.assertIsNone(manager.active)
        manager._persist = original
        self.assertEqual(self.run_job(manager)['status'], 'completed')

    def test_all_summary_metrics_require_reference_before_loading(self):
        manager = self.manager()
        for metric in ('grounded', 'reference'):
            with self.assertRaisesRegex(ValueError, 'reference'):
                manager.start(dict(embedding_id=EMBEDDING_ID, summary_metric=metric))
        self.assertEqual(self.evaluators, [])

    def test_all_class_labels_are_resolved_before_any_inference(self):
        class LabelEvaluator(Evaluator):
            def resolve_label(self, target):
                raise ValueError('model has no such class')
        self.embeddings.context['preview']['config']['task'] = 'classification'
        evaluator = LabelEvaluator()
        manager = self.manager(evaluator)
        job = manager.start(dict(embedding_id=EMBEDDING_ID, label_map={'0': 99}))
        result = self.wait(lambda: manager.get(job['id']))
        self.assertEqual(result['status'], 'error')
        self.assertIn('no such class', result['error'])
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(evaluator.calls, [])
        self.assertTrue(evaluator.closed)

    def test_training_loss_history_is_exposed_as_final_scalar(self):
        def train(model, task, examples, directory, settings, progress, cancelled):
            checkpoint = directory/'checkpoint'
            checkpoint.mkdir()
            return {'checkpoint': str(checkpoint), 'steps': 2, 'training_loss': [.4, .2]}
        manager = self.manager(finetune=train)
        discovery = self.run_job(manager)
        record = discovery['records'][0]
        job = manager.start_finetune(dict(discovery_id=discovery['id'], examples=[
            {'input_index': 0, 'action_id': record['action_id'], 'target': 'desired'}]))
        result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['loss'], .2)

    def test_artifact_rejects_symlinked_job_directory(self):
        manager = self.manager()
        result = self.run_job(manager)
        folder = self.directory/result['id']
        saved = self.directory/'saved'
        folder.rename(saved)
        external = self.directory/'external'
        external.mkdir()
        (external/'report.json').write_text('private unrelated file')
        folder.symlink_to(external, target_is_directory=True)
        with self.assertRaises((ValueError, FileNotFoundError)):
            manager.artifact(result['id'], 'report.json')

    def test_restart_marks_interrupted_report_and_keeps_completed_measurements(self):
        manager = self.manager()
        result = self.run_job(manager)
        path = manager.artifact(result['id'], 'report.json')
        report = json.loads(path.read_text())
        report['status'] = 'running'
        path.write_text(json.dumps(report))
        restored = self.manager()
        job = restored.get(result['id'])
        self.assertEqual(job['status'], 'error')
        self.assertEqual(job['records'], result['records'])
        self.assertIsNone(restored.active)
        self.assertEqual(json.loads(restored.artifact(result['id'], 'report.json').read_text())['status'], 'error')


if __name__ == '__main__':
    unittest.main()
