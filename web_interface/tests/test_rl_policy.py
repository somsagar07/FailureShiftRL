"""Terminal DQN learns observed rewards on actual input feature vectors."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl_policy import TerminalDQN


class TerminalDQNTests(unittest.TestCase):
    def test_chunked_statistics_collect_means_without_full_reward_matrix(self):
        policy = TerminalDQN(2, 3)
        policy.network = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            policy.network.weight.copy_(torch.tensor([[.2, .4], [.7, .1], [.5, .8]]))
        policy.updates = 1
        states = np.tile(np.eye(2, dtype=np.float32), (150, 1))
        states.setflags(write=False)
        means = np.full(300, np.nan, dtype=np.float32)
        policy.action_statistics(states, input_mean_out=means)
        np.testing.assert_allclose(means, np.tile([1.4 / 3, 1.3 / 3], 150), atol=1e-7)
        with self.assertRaises(ValueError):
            policy.action_statistics(states, input_mean_out=np.empty(1))

    def test_landscape_retains_each_inputs_rewards_in_action_order(self):
        policy = TerminalDQN(2, 3)
        policy.network = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            policy.network.weight.copy_(torch.tensor([[.2, .4], [.7, .1], [.5, .8]]))
        policy.updates = 1
        states = np.tile(np.eye(2, dtype=np.float32), (150, 1))
        values = np.full((300, 3), np.nan, dtype=np.float32)
        policy.action_statistics(states, q_values_out=values)
        np.testing.assert_allclose(values, np.tile([[.2, .7, .5], [.4, .1, .8]], (150, 1)), atol=1e-7)
        self.assertEqual(policy.updates, 1)

    def test_action_landscape_uses_original_inputs_at_current_checkpoint(self):
        policy = TerminalDQN(2, 3, seed=9)
        policy.network = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            policy.network.weight.copy_(torch.log(torch.tensor([[.7, .1], [.2, .3], [.1, .6]])))
        policy.updates = 1
        result = policy.action_statistics(np.eye(2, dtype=np.float32))
        self.assertEqual(result['input_count'], 2)
        self.assertEqual(result['policy_updates'], 1)
        self.assertEqual(result['std_ddof'], 1)
        np.testing.assert_allclose([action['mean'] for action in result['actions']], [.4, .25, .35], atol=1e-7)
        np.testing.assert_allclose([action['std'] for action in result['actions']],
                                   [.3 * 2**.5, .05 * 2**.5, .25 * 2**.5], atol=1e-7)
        self.assertEqual(policy.updates, 1, 'Inspecting a policy must not train it')
        # Cross several internal batches, including a final incomplete batch.
        many = policy.action_statistics(np.repeat(np.eye(2, dtype=np.float32), 300, axis=0))
        np.testing.assert_allclose([action['mean'] for action in many['actions']], [.4, .25, .35], atol=1e-7)
        np.testing.assert_allclose([action['std'] for action in many['actions']],
                                   np.array([.3, .05, .25]) * np.sqrt(600/599), atol=1e-7)

    def test_single_input_and_untrained_policy_do_not_claim_certainty(self):
        policy = TerminalDQN(2, 3, seed=3, batch_size=2)
        self.assertEqual(policy.action_statistics([[1, 0]])['actions'], [])
        policy.observe([1, 0], 0, .5)
        policy.observe([1, 0], 0, .5)
        result = policy.action_statistics([[1, 0]])
        self.assertEqual(result['input_count'], 1)
        self.assertAlmostEqual(sum(action['mean'] for action in result['actions']), 1., places=6)
        self.assertTrue(all(action['std'] is None for action in result['actions']))
        for states in ([], [[1]], [[float('nan'), 0]]):
            with self.assertRaises(ValueError):
                policy.action_statistics(states)

    def test_replay_learns_state_specific_terminal_reward_preferences(self):
        policy = TerminalDQN(2, 2, seed=7, batch_size=16, learning_rate=.01)
        states = np.eye(2, dtype=np.float32)
        first_loss = []
        last_loss = []
        for episode in range(160):
            index, action = episode % 2, (episode // 2) % 2
            loss = policy.observe(states[index], action, 1. if index == action else -1.)
            if loss is not None:
                (first_loss if episode < 40 else last_loss).append(loss)
        self.assertEqual([policy.choose(state, epsilon=0) for state in states], [0, 1])
        self.assertLess(np.mean(last_loss[-20:]), np.mean(first_loss))
        self.assertEqual(policy.updates, 145)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.pt'
            policy.save(path)
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload['state_dimension'], 2)
            self.assertEqual(payload['action_count'], 2)
            self.assertEqual(payload['updates'], policy.updates)

    def test_seed_reproduces_actions_and_learning(self):
        policies = [TerminalDQN(2, 3, seed=3, batch_size=2) for _ in range(2)]
        results = []
        for policy in policies:
            results.append([(policy.choose([1, 0], epsilon=.7), policy.observe([1, 0], i % 3, .3)) for i in range(8)])
        self.assertEqual(results[0], results[1])

    def test_invalid_shapes_actions_and_nonfinite_rewards_are_rejected(self):
        policy = TerminalDQN(2, 3)
        for state in ([1], [float('nan'), 0]):
            with self.assertRaises(ValueError):
                policy.choose(state)
        for action, reward in ((-1, 0), (3, 0), (True, 0), (0, float('inf'))):
            with self.assertRaises(ValueError):
                policy.observe([1, 0], action, reward)


if __name__ == '__main__':
    unittest.main()
