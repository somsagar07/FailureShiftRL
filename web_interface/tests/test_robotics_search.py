"""Real terminal-DQN learning with controlled, already-emitted rollout records."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


class SearchSettingsTests(unittest.TestCase):
    def normalize(self, data):
        self.assertIsNotNone(importlib.util.find_spec('robotics_search'),
                             'The standalone robotics search module must exist.')
        return importlib.import_module('robotics_search').normalize_search(data)

    def test_default_settings_and_inclusive_boundaries(self):
        self.assertEqual(self.normalize({}), dict(episodes=200, rollout_budget=36,
            epsilon_start=1., epsilon_end=.1, learning_rate=.002, batch_size=16,
            replay_capacity=4096, seed=42, criterion='task_failure'))
        settings = dict(episodes=10000, rollout_budget=10000, epsilon_start=0,
            epsilon_end=0, learning_rate=.1, batch_size=1024, replay_capacity=100000,
            seed=2**31-1001, criterion='new_failure')
        self.assertEqual(self.normalize(settings), settings)
        self.assertEqual(self.normalize(dict(episodes=1, rollout_budget=1, batch_size=1,
            replay_capacity=1, learning_rate=.000001, seed=0))['replay_capacity'], 1)

    def test_unknown_fields_and_non_json_or_nonfinite_values_are_rejected(self):
        cases = [None, [], '', {'steps': 10}, {'episodes': True}, {'episodes': 1.0},
            {'episodes': 0}, {'episodes': 10001}, {'rollout_budget': 0},
            {'rollout_budget': 10001}, {'epsilon_start': True}, {'epsilon_end': '0.1'},
            {'epsilon_end': -1}, {'epsilon_start': 1.1}, {'epsilon_start': .05},
            {'learning_rate': float('nan')}, {'learning_rate': float('inf')},
            {'epsilon_start': 10**400}, {'learning_rate': 10**400},
            {'learning_rate': 0}, {'learning_rate': .100001}, {'batch_size': 0},
            {'batch_size': 1025}, {'replay_capacity': 15}, {'replay_capacity': 100001},
            {'seed': -1}, {'seed': 2**31-1000}, {'criterion': 'success'},
            {'criterion': ['task_failure']}]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.normalize(value)

    def test_normalization_is_independent_and_stdlib_import_safe(self):
        source = {'episodes': 7}
        normalized = self.normalize(source)
        normalized['episodes'] = 8
        self.assertEqual(source, {'episodes': 7})
        self.assertEqual(self.normalize({})['episodes'], 200)
        command = (
            'import json, sys; import robotics_search; '
            'assert not ({"numpy", "torch", "robotics", "rl_policy"} & set(sys.modules)); '
            'print(json.dumps(robotics_search.normalize_search({})))'
        )
        result = subprocess.run([sys.executable, '-S', '-c', command], cwd=ROOT,
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['episodes'], 200)


class RoboticsSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def run_search(self, directory, *, states=None, seeds=None, actions=None,
                   baselines=None, settings=None, outcome=None):
        import robotics_search
        search = getattr(robotics_search, 'run_search', None)
        self.assertTrue(callable(search), 'The rollout-backed search entry point must exist.')
        states = np.eye(2, dtype=np.float32) if states is None else states
        seeds = [41, 42] if seeds is None else seeds
        actions = [dict(id=11, values={'mass_scale': .5}),
                   dict(id=29, values={'mass_scale': 1.5})] if actions is None else actions
        baselines = [dict(action_id='baseline', seed=seed, success=True, steps=2)
                     for seed in seeds] if baselines is None else baselines
        events, calls = [], []

        def evaluate(index, action):
            calls.append((index, action['id']))
            result = outcome(index, action) if outcome else dict(success=index != actions.index(action))
            row = dict(action_id=action['id'], input_index=index, seed=seeds[index],
                       success=None, steps=3, seconds=.01, **{'return': 1.})
            row.update(result)
            events.append(dict(type='record', record=row))
            return row

        result = search(states, seeds, actions, baselines, settings or {}, evaluate,
                        events.append, directory)
        history = [json.loads(line) for line in (Path(directory)/'dqn-history.jsonl').read_text().splitlines()]
        return result, calls, events, history

    def test_real_replay_learns_state_specific_failures_and_saves_matching_estimates(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls, events, history = self.run_search(directory, settings=dict(
                episodes=360, rollout_budget=4, batch_size=8, learning_rate=.01, seed=7))
            metrics = result['search']
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(set(calls)), 4)
            self.assertEqual(metrics['model_calls'], 4)
            self.assertEqual(metrics['cache_hits'], 356)
            self.assertEqual(metrics['updates'], 353)
            self.assertEqual(metrics['observed_failures'], 2)
            self.assertEqual(metrics['valid_visits'], 360)
            self.assertEqual(len(history), 360)
            self.assertEqual([history[0]['epsilon'], history[-1]['epsilon']], [1., .1])
            self.assertEqual(sum(row['source'] == 'rollout' for row in history), 4)
            self.assertEqual({row['reward'] for row in history}, {0., 1.})
            self.assertTrue(all(row['input_index'] in (0, 1) and row['action_id'] in (11, 29)
                                for row in history))
            estimates = np.load(Path(directory)/'dqn-predictions.npy', allow_pickle=False)
            self.assertEqual(estimates.shape, (2, 2))
            self.assertEqual(estimates.dtype, np.float32)
            self.assertEqual(estimates.argmax(axis=1).tolist(), [0, 1])
            self.assertGreater(estimates[0, 0]-estimates[0, 1], .4)
            self.assertGreater(estimates[1, 1]-estimates[1, 0], .4)
            checkpoint = torch.load(Path(directory)/'dqn-policy.pt', weights_only=True)
            self.assertEqual(checkpoint['updates'], 353)
            self.assertEqual(checkpoint['batch_size'], 8)
            self.assertEqual(checkpoint['seed'], 7)
            self.assertEqual(checkpoint['optimizer_state']['param_groups'][0]['lr'], .01)
            landscape = result['predicted_landscape']
            self.assertEqual(landscape['seeds'], [41, 42])
            self.assertEqual(landscape['action_ids'], [11, 29])
            self.assertEqual(landscape['policy_updates'], 353)
            self.assertTrue(landscape['available'])
            np.testing.assert_allclose(landscape['input_mean_reward'], estimates.mean(axis=1))
            self.assertEqual([row['action_id'] for row in result['policy_statistics']['actions']], [11, 29])
            self.assertEqual(len([event for event in events if event['type'] == 'record']), 4)
            self.assertEqual(set(result['artifacts'].values()),
                             {'dqn-policy.pt', 'dqn-predictions.npy', 'dqn-history.jsonl'})
            self.assertFalse(list(Path(directory).glob('*.tmp')))
            json.dumps(events, allow_nan=False)

    def test_budget_exhaustion_trains_only_measured_pairs_without_new_records(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls, events, history = self.run_search(directory, settings=dict(
                episodes=25, rollout_budget=1, batch_size=4, replay_capacity=8))
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['search']['model_calls'], 1)
            self.assertEqual(result['search']['cache_hits'], 24)
            self.assertEqual(result['search']['updates'], 22)
            self.assertEqual(result['search']['replay_size'], 8)
            self.assertEqual({(row['input_index'], row['action_id']) for row in history}, set(calls))
            self.assertEqual(len([event for event in events if event['type'] == 'record']), 1)

    def test_neutral_visits_reuse_exact_seed_baseline_once_and_preserve_zero_states(self):
        baselines = [dict(action_id='baseline', seed=42, success=False, image='seed-42.jpg',
                          embedding=dict(position=[1, 2, 3], similarity=1.)),
                     dict(action_id='baseline', seed=41, success=True, image='seed-41.jpg',
                          embedding=dict(position=[3, 2, 1], similarity=1.))]
        untouched = json.loads(json.dumps(baselines))
        with tempfile.TemporaryDirectory() as directory:
            result, calls, events, history = self.run_search(directory,
                states=np.zeros((2, 2), dtype=np.float32), baselines=baselines,
                actions=[dict(id=97, values={'mass_scale': 1, 'action_delay': 0})],
                settings=dict(episodes=30, rollout_budget=1, batch_size=4))
            self.assertEqual(calls, [])
            self.assertEqual(result['search']['model_calls'], 0)
            self.assertEqual(result['search']['baseline_reuses'], 2)
            self.assertEqual(result['search']['cache_hits'], 28)
            self.assertEqual(result['search']['observed_failures'], 0)
            self.assertEqual(result['search']['updates'], 27)
            reused = [event['record'] for event in events if event['type'] == 'record']
            self.assertEqual(len(reused), 2)
            self.assertEqual({(row['input_index'], row['seed']) for row in reused}, {(0, 41), (1, 42)})
            self.assertTrue(all(row['reused_baseline'] and row['action_id'] == 97 for row in reused))
            self.assertTrue(all(row['image'] == f"seed-{row['seed']}.jpg" for row in reused))
            self.assertTrue(all(row['reward'] == float(row['seed'] == 42) for row in history))
            self.assertEqual(baselines, untouched)

    def test_new_failure_rewards_require_a_successful_matched_baseline(self):
        baselines = [dict(seed=41, success=True), dict(seed=42, success=False)]
        with tempfile.TemporaryDirectory() as directory:
            result, calls, _, history = self.run_search(directory, baselines=baselines,
                actions=[dict(id=11, values={'mass_scale': .5})],
                settings=dict(episodes=40, rollout_budget=2, batch_size=4, criterion='new_failure'),
                outcome=lambda index, action: dict(success=False))
            self.assertEqual(len(calls), 2)
            self.assertEqual(result['search']['observed_failures'], 2)
            self.assertEqual(result['search']['observed_new_failures'], 1)
            self.assertTrue(all(row['reward'] == float(row['seed'] == 41) for row in history))

    def test_rollout_errors_do_not_become_rewards_and_cached_training_avoids_them(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls, events, history = self.run_search(directory, states=[[1, 0]], seeds=[41],
                settings=dict(episodes=40, rollout_budget=2, batch_size=4, epsilon_end=1),
                outcome=lambda index, action: dict(error='simulator stopped', success=None)
                    if action['id'] == 11 else dict(success=False))
            self.assertEqual(len(calls), 2)
            metrics = result['search']
            self.assertEqual(metrics['rollout_errors'], 1)
            self.assertEqual(metrics['observed_failures'], 1)
            self.assertEqual(metrics['valid_visits'], 39)
            self.assertEqual(metrics['updates'], 36)
            self.assertEqual(metrics['cache_hits'], 38)
            errors = [row for row in history if row['reward'] is None]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]['action_id'], 11)
            self.assertEqual(len([event for event in events if event['type'] == 'record']), 2)

    def test_all_errors_stop_with_no_invented_training_or_untrained_predictions(self):
        for budget, expected_calls in ((1, 1), (10, 2)):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory:
                result, calls, events, history = self.run_search(directory, states=[[1]], seeds=[41],
                    settings=dict(episodes=100, rollout_budget=budget, batch_size=1),
                    outcome=lambda index, action: dict(error='reset failed', success=False))
                metrics = result['search']
                self.assertEqual(len(calls), expected_calls)
                self.assertEqual(len(history), expected_calls)
                self.assertEqual(metrics['episode'], expected_calls)
                self.assertEqual(metrics['updates'], 0)
                self.assertEqual(metrics['observed_failures'], 0)
                self.assertEqual(metrics['valid_visits'], 0)
                self.assertEqual(metrics['cache_hits'], 0)
                self.assertTrue(metrics['stopped_reason'])
                self.assertTrue(all(row['reward'] is None for row in history))
                self.assertFalse(result['predicted_landscape']['available'])
                self.assertEqual(result['predicted_landscape']['input_mean_reward'], [None])
                self.assertEqual(result['policy_statistics']['actions'], [])
                self.assertTrue(np.isnan(np.load(Path(directory)/'dqn-predictions.npy')).all())
                json.dumps(events, allow_nan=False)

    def test_seed_reproduces_learning_visits_and_saved_predictions(self):
        settings = dict(episodes=45, rollout_budget=4, batch_size=4, learning_rate=.007,
                        epsilon_start=.8, epsilon_end=.2, seed=19)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = self.run_search(first, settings=settings)
            b = self.run_search(second, settings=settings)
            self.assertEqual(a[0]['search'], b[0]['search'])
            self.assertEqual(a[1], b[1])
            self.assertEqual(a[3], b[3])
            np.testing.assert_array_equal(np.load(Path(first)/'dqn-predictions.npy'),
                                          np.load(Path(second)/'dqn-predictions.npy'))

    def test_fast_cached_steps_compute_statistics_once_at_the_final_checkpoint(self):
        import robotics_search
        self.assertTrue(callable(getattr(robotics_search, 'run_search', None)),
                        'The rollout-backed search entry point must exist.')
        from rl_policy import TerminalDQN
        statistics = TerminalDQN.action_statistics
        snapshots = []

        def inspect(policy, *args, **kwargs):
            snapshots.append(policy.updates)
            return statistics(policy, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('robotics_search.time.monotonic', return_value=100.), \
                mock.patch.object(TerminalDQN, 'action_statistics', inspect):
            result, _, events, _ = self.run_search(directory,
                settings=dict(episodes=60, rollout_budget=1, batch_size=4))
            self.assertEqual(snapshots, [57])
            progress = [event for event in events if event['type'] == 'progress']
            self.assertTrue(all('search' in event and 'current' in event for event in progress))
            self.assertEqual(progress[-1]['policy_statistics'], result['policy_statistics'])
            self.assertEqual(progress[-1]['predicted_landscape'], result['predicted_landscape'])
            self.assertNotIn('q_values', json.dumps(progress))

    def test_prediction_artifact_covers_all_inputs_across_inference_batches(self):
        states = np.tile(np.eye(2, dtype=np.float32), (129, 1))
        seeds = list(range(100, 358))
        with tempfile.TemporaryDirectory() as directory:
            result, _, _, _ = self.run_search(directory, states=states, seeds=seeds,
                settings=dict(episodes=4, rollout_budget=1, batch_size=1))
            estimates = np.load(Path(directory)/'dqn-predictions.npy', mmap_mode='r')
            self.assertEqual(estimates.shape, (258, 2))
            self.assertTrue(np.isfinite(estimates).all())
            np.testing.assert_allclose(estimates[0], estimates[256], atol=1e-7)
            np.testing.assert_allclose(estimates[1], estimates[257], atol=1e-7)
            self.assertEqual(result['predicted_landscape']['seeds'], seeds)

    def test_maximum_landscape_is_stored_as_a_matrix_without_a_matrix_in_events(self):
        states = np.tile(np.eye(2, dtype=np.float32), (500, 1))
        seeds = list(range(1000))
        actions = [dict(id=index+10, values={'mass_scale': .5}) for index in range(5000)]
        with tempfile.TemporaryDirectory() as directory:
            result, calls, events, _ = self.run_search(directory, states=states, seeds=seeds,
                actions=actions, settings=dict(episodes=2, rollout_budget=1, batch_size=1))
            values = np.load(Path(directory)/'dqn-predictions.npy', mmap_mode='r', allow_pickle=False)
            self.assertEqual(values.shape, (1000, 5000))
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(result['predicted_landscape']['input_mean_reward']), 1000)
            self.assertEqual(len(result['policy_statistics']['actions']), 5000)
            self.assertLess(len(json.dumps(events)), 2000000)

    def test_live_prediction_files_are_ready_at_publish_and_have_immutable_update_names(self):
        from robotics_search import run_search
        snapshots = []
        clock = iter(range(100, 1000))
        with tempfile.TemporaryDirectory() as directory:
            def publish(event):
                landscape = event.get('predicted_landscape')
                if landscape is None:
                    return
                path = Path(directory)/landscape['values_file']
                values = np.load(path, allow_pickle=False)
                self.assertEqual(values.shape, (1, 1))
                self.assertTrue(np.isfinite(values).all())
                np.testing.assert_allclose(landscape['input_mean_reward'], values.mean(axis=1))
                if path.name != 'dqn-predictions.npy':
                    self.assertEqual(path.name, f"dqn-predictions-{landscape['policy_updates']}.npy")
                snapshots.append((path, landscape['policy_updates']))

            with mock.patch('robotics_search.time.monotonic', side_effect=lambda: next(clock)):
                run_search([[1]], [42], [dict(id=5, values={'mass_scale': .5})],
                    [dict(seed=42, success=True)], dict(episodes=12, rollout_budget=1, batch_size=1),
                    lambda index, action: dict(seed=42, action_id=5, success=False), publish, directory)
            self.assertGreater(len(snapshots), 2)
            self.assertEqual(snapshots[-1][0].name, 'dqn-predictions.npy')
            self.assertEqual(snapshots[-1][1], 12)
            self.assertTrue(all(not path.exists() for path, _ in snapshots[:-1]))
            self.assertFalse(list(Path(directory).glob('*.tmp')))

    def test_bad_shapes_and_misaligned_records_are_rejected_before_rollouts(self):
        cases = [dict(states=[]), dict(states=[1, 0]), dict(states=[[float('nan'), 0], [0, 1]]),
                 dict(states=np.zeros((1001, 2))), dict(seeds=[41]), dict(seeds=[41, 41]),
                 dict(actions=[]), dict(actions=[dict(id=i, values={}) for i in range(5001)]),
                 dict(actions=[dict(id=1, values={}), dict(id=1, values={})]),
                 dict(baselines=[dict(seed=41, success=True), dict(seed=43, success=False)]),
                 dict(baselines=[dict(seed=41, success=True), dict(seed=42, success=None)])]
        for options in cases:
            with self.subTest(keys=list(options)), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.run_search(directory, settings={'episodes': 1}, **options)

    def test_evaluator_cancellation_propagates_without_fabricating_a_record(self):
        def cancelled(index, action):
            raise InterruptedError('cancelled')

        with tempfile.TemporaryDirectory() as directory, self.assertRaises(InterruptedError):
            self.run_search(directory, outcome=cancelled, settings={'episodes': 10})


if __name__ == '__main__':
    unittest.main()
