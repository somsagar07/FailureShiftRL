"""Finite, one-step DQN on normalized original model features.

Every transition terminates after one action: the TD target is its measured
reward. There is deliberately no bootstrapped future reward. A separate
softmax-Q summary follows the original experiment's visualization convention;
it never changes the training target or epsilon-greedy action selection.
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

    def action_statistics(self, states):
        """Describe action preference variation over originals, not failure odds.

        This is a single current checkpoint evaluated on every original state.
        Repeated RL visits supply no additional statistical samples. Variation
        across inputs is undefined for a single article, prompt, or image.
        """
        states = np.asarray(states, dtype=np.float32)
        if (states.ndim != 2 or not len(states) or states.shape[1] != self.state_dimension
                or not np.isfinite(states).all()):
            raise ValueError('Policy statistics require finite original embedding vectors.')
        result = dict(method='softmax_q_across_original_inputs', temperature=1,
                      input_count=len(states), policy_updates=self.updates, std_ddof=1, actions=[])
        if not self.updates:
            return result
        # Merge batch moments instead of retaining an inputs-by-actions matrix.
        count = 0
        means = np.zeros(self.action_count, dtype=np.float64)
        squared_deviations = np.zeros_like(means)
        with self.torch.inference_mode():
            for offset in range(0, len(states), 256):
                values = self.network(self.torch.from_numpy(states[offset:offset+256]))
                if not self.torch.isfinite(values).all():
                    raise ValueError('Policy produced nonfinite action values.')
                preferences = values.softmax(dim=-1).to(dtype=self.torch.float64).cpu().numpy()
                batch_mean = preferences.mean(axis=0)
                batch_count = len(preferences)
                total = count + batch_count
                delta = batch_mean - means
                squared_deviations += ((preferences-batch_mean)**2).sum(axis=0)
                squared_deviations += delta**2 * (count * batch_count / total)
                means += delta * (batch_count / total)
                count = total
        spreads = np.sqrt(squared_deviations / (count-1)) if count > 1 else None
        result['actions'] = [dict(action_index=index, mean=float(mean),
                                  std=float(spreads[index]) if spreads is not None else None)
                             for index, mean in enumerate(means)]
        return result
