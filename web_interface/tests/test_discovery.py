"""Offline model boundary exercises live workers, persistence and reviewed training."""
import copy
import json
import tempfile
import threading
import time
import unittest
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
        self.context = dict(
            job={'id': EMBEDDING_ID, 'status': 'completed', 'preview_id': PREVIEW_ID},
            model={'source': 'huggingface', 'name': 'fixture/model', 'task': 'summarization'},
            model_key='unchanged', states=np.array([[.6, .8]]),
            preview={'id': PREVIEW_ID, 'status': 'completed', 'input_count': 1,
                     'config': {'task': 'summarization', 'text': 'original'},
                     'original': {'text': 'original'},
                     'actions': [{'id': 4}, {'id': 9}],
                     'nodes': [{'id': 4, 'text': 'good'}, {'id': 9, 'text': 'bad'}]})

    def discovery_context(self, identity):
        if identity != EMBEDDING_ID:
            raise KeyError(identity)
        return copy.deepcopy(self.context)

    def release_encoder(self):
        self.released += 1


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
        self.assertEqual(self.embeddings.released, 1)
        report = json.loads(manager.artifact(result['id'], 'report.json').read_text())
        self.assertEqual(report['records'], result['records'])
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
            self.assertFalse(cancelled())
            progress(completed=1, total=1, loss=.2)
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
        for settings in ({'batch_size': 33}, {'learning_rate': .2}, {'examples': valid_examples*257}):
            with self.assertRaises(ValueError):
                manager.start_finetune(dict(discovery_id=discovery['id'], examples=valid_examples) | settings)
        job = manager.start_finetune(dict(discovery_id=discovery['id'], examples=[
            {'input_index': 0, 'action_id': action, 'target': 'reviewed desired output'}]))
        result = self.wait(lambda: manager.get_finetune(job['id']))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(captured[0]['target'], 'reviewed desired output')
        self.assertIn(captured[0]['input'], ('good', 'bad'))
        self.assertEqual(result['loss'], .2)
        self.assertTrue(Path(result['checkpoint']).is_dir())

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
