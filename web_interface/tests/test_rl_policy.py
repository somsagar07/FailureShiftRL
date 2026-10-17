"""Terminal DQN learns observed rewards on actual input feature vectors."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl_policy import TerminalDQN


class TerminalDQNTests(unittest.TestCase):
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
