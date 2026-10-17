"""Finite, one-step DQN on normalized original model features.

Every transition terminates after one action: the TD target is its measured
reward. There is deliberately no bootstrapped future reward or Q softmax.
"""
import math
import random
from collections import deque
from pathlib import Path

import numpy as np


class TerminalDQN:
    def __init__(self, state_dimension, action_count, *, seed=0, batch_size=16,
                 learning_rate=.002, replay_capacity=4096):
        import torch
        if any(type(value) is not int or value < 1 for value in
               (state_dimension, action_count, batch_size, replay_capacity)):
            raise ValueError('Policy dimensions and replay sizes must be positive integers.')
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError('Policy learning rate must be finite and positive.')
        self.torch = torch
        self.state_dimension, self.action_count = state_dimension, action_count
        self.seed, self.batch_size = seed, batch_size
        self.random = random.Random(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = torch.nn.Sequential(
                torch.nn.Linear(state_dimension, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, action_count))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.replay = deque(maxlen=replay_capacity)
        self.updates = 0

    def _state(self, state):
        array = np.asarray(state, dtype=np.float32)
        if array.shape != (self.state_dimension,) or not np.isfinite(array).all():
            raise ValueError('Policy state must be a finite original embedding vector.')
        return array.copy()

    def choose(self, state, epsilon=1.):
        state = self._state(state)
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError('Epsilon must be between zero and one.')
        if self.random.random() < epsilon:
            return self.random.randrange(self.action_count)
        with self.torch.inference_mode():
            values = self.network(self.torch.from_numpy(state))
            if not self.torch.isfinite(values).all():
                raise ValueError('Policy produced nonfinite action values.')
            return int(values.argmax().item())

    def observe(self, state, action, reward):
        state = self._state(state)
        if type(action) is not int or not 0 <= action < self.action_count:
            raise ValueError('Action index is outside this policy.')
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError('Terminal reward must be finite.')
        self.replay.append((state, action, float(reward)))
        if len(self.replay) < self.batch_size:
            return None
        torch = self.torch
        samples = self.random.sample(list(self.replay), self.batch_size)
        states = torch.from_numpy(np.stack([sample[0] for sample in samples]))
        actions = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
        targets = torch.tensor([sample[2] for sample in samples], dtype=torch.float32)
        estimates = self.network(states).gather(1, actions[:, None]).squeeze(1)
        loss = torch.nn.functional.mse_loss(estimates, targets)
        if not torch.isfinite(loss):
            raise ValueError('Policy training produced a nonfinite loss.')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        return float(loss.detach().item())

    def save(self, path):
        """Weights-only-loadable artifact with enough metadata to inspect/reload."""
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        self.torch.save(dict(format='terminal-dqn-v1', state_dimension=self.state_dimension,
                             action_count=self.action_count, seed=self.seed, updates=self.updates,
                             batch_size=self.batch_size, model_state=self.network.state_dict(),
                             optimizer_state=self.optimizer.state_dict(),
                             target='observed terminal reward; no bootstrap'), temporary)
        temporary.replace(path)
