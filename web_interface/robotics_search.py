"""Terminal DQN over measured, deterministic robotics episode variations.

Configuration parsing stays standard-library-only so the web server need not
load the isolated simulator's numeric runtime. Numeric imports belong inside
``run_search`` and its runtime helpers.
"""
import copy
import json
import math
from numbers import Integral
from pathlib import Path
import random
import time


_DEFAULTS = dict(episodes=200, rollout_budget=36, epsilon_start=1., epsilon_end=.1,
                 learning_rate=.002, batch_size=16, replay_capacity=4096,
                 seed=42, criterion='task_failure')


def normalize_search(data):
    """Validate the explicit discovery search object without coercing JSON types."""
    if not isinstance(data, dict):
        raise ValueError('Search settings must be a JSON object.')
    unknown = set(data).difference(_DEFAULTS)
    if unknown:
        raise ValueError('Unknown search settings: '+', '.join(sorted(map(str, unknown)))+'.')
    settings = dict(_DEFAULTS, **data)
    for name, low, high in [('episodes', 1, 10000), ('rollout_budget', 1, 10000),
                           ('batch_size', 1, 1024), ('replay_capacity', 1, 100000),
                           ('seed', 0, 2**31-1001)]:
        value = settings[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'Search {name} must be an integer from {low} to {high}.')
    for name, low, high in [('epsilon_start', 0., 1.), ('epsilon_end', 0., 1.),
                           ('learning_rate', .000001, .1)]:
        value = settings[name]
        if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
            raise ValueError(f'Search {name} must be a finite number from {low} to {high}.')
        settings[name] = float(value)
    if settings['epsilon_end'] > settings['epsilon_start']:
        raise ValueError('Search epsilon_end must not exceed epsilon_start.')
    if settings['replay_capacity'] < settings['batch_size']:
        raise ValueError('Search replay_capacity must be at least batch_size.')
    if settings['criterion'] not in ('task_failure', 'new_failure'):
        raise ValueError('Search criterion must be task_failure or new_failure.')
    return settings


def _inputs(states, seeds, actions, baselines):
    import numpy as np
    from robotics import GROUPS

    try:
        states = np.asarray(states, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError('Search states must be finite episode feature vectors.') from None
    if states.ndim != 2 or not 1 <= len(states) <= 1000 or not states.shape[1] or not np.isfinite(states).all():
        raise ValueError('Search requires 1 to 1000 finite episode feature vectors.')
    if not isinstance(seeds, (list, tuple, np.ndarray)) or len(seeds) != len(states) or any(
            isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 for seed in seeds):
        raise ValueError('Search seeds must match the episode feature rows.')
    seeds = list(map(int, seeds))
    if len(set(seeds)) != len(seeds):
        raise ValueError('Search episode seeds must be unique.')
    if not isinstance(actions, (list, tuple)) or not 1 <= len(actions) <= 5000:
        raise ValueError('Search requires 1 to 5000 environment actions.')
    neutral = {group['id']: group['neutral'] for group in GROUPS}
    identifiers, neutral_indexes = set(), set()
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or type(action.get('id')) not in (int, str):
            raise ValueError('Search actions must have unique identifiers and environment values.')
        if action['id'] in identifiers:
            raise ValueError('Search action identifiers must be unique.')
        identifiers.add(action['id'])
        values = action.get('values')
        if not isinstance(values, dict) or any(key not in neutral for key in values) or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
            raise ValueError('Search actions must use finite, supported environment values.')
        if all(value == neutral[key] for key, value in values.items()):
            neutral_indexes.add(index)
    if not isinstance(baselines, (list, tuple)) or any(not isinstance(row, dict) or
            isinstance(row.get('seed'), bool) or not isinstance(row.get('seed'), Integral) or
            row.get('error') or type(row.get('success')) is not bool for row in baselines):
        raise ValueError('Search requires successful or failed baseline records with matching seeds.')
    by_seed = {int(row['seed']): row for row in baselines}
    if len(by_seed) != len(baselines) or set(by_seed) != set(seeds):
        raise ValueError('Search baseline records must match the episode feature seeds exactly.')
    return states, seeds, list(actions), [by_seed[seed] for seed in seeds], neutral_indexes


def _choose_allowed(policy, state, epsilon, allowed):
    """Use epsilon-greedy selection within the currently executable action set."""
    if len(allowed) == policy.action_count:
        return policy.choose(state, epsilon)
    if policy.random.random() < epsilon:
        return policy.random.choice(allowed)
    torch = policy.torch
    with torch.inference_mode():
        values = policy.network(torch.tensor(state, dtype=torch.float32))
        if not torch.isfinite(values).all():
            raise ValueError('Policy produced nonfinite action values.')
        return allowed[int(values[allowed].argmax().item())]


def _predictions(policy, states, seeds, actions, valid_actions, settings, directory, filename):
    """Persist a bounded matrix in batches; publish only the small input means."""
    import numpy as np

    temporary = directory/(filename+'.tmp')
    values = None
    means = np.empty(len(states), dtype=np.float32) if policy.updates else None
    try:
        values = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                          shape=(len(states), len(actions)))
        # No optimizer update means there are no learned predictions to expose.
        if not policy.updates:
            values[:] = np.nan
        statistics = policy.action_statistics(states, q_values_out=values, input_mean_out=means)
        values.flush()
        values._mmap.close()
        values = None
        temporary.replace(directory/filename)
    finally:
        if values is not None:
            values._mmap.close()
        temporary.unlink(missing_ok=True)
    for action in statistics['actions']:
        index = action['action_index']
        action['action_id'] = actions[index]['id']
        action['evaluated_inputs'] = sum(index in known for known in valid_actions)
    landscape = dict(method='terminal_q_reward', policy_updates=policy.updates,
        input_count=len(states), action_count=len(actions), seeds=seeds,
        input_mean_reward=means.tolist() if means is not None else [None]*len(states),
        action_ids=[action['id'] for action in actions], values_file=filename,
        color_range=[0, 1], available=bool(policy.updates), criterion=settings['criterion'],
        semantics='Expected terminal reward estimates; not measured outcomes or calibrated failure probabilities.')
    return statistics, landscape


def run_search(states, seeds, actions, baselines, settings, evaluate, publish, directory):
    """Train a separate DQN without modifying or calling the tested robot policy.

    ``evaluate(input_index, action)`` performs and emits exactly one real rollout
    record. Simulator failures are returned as error records; exceptions such as
    cancellation propagate. This function emits only explicit baseline reuses.

    A step is a terminal learning visit. ``model_calls`` counts distinct simulator
    attempts, including errors; ``cache_hits`` counts repeated visits, and
    ``baseline_reuses`` counts newly visited neutral pairs. Their sum is the
    number of completed steps. Observed failures count only actual rollouts.
    """
    from rl_policy import TerminalDQN

    settings = normalize_search(settings)
    states, seeds, actions, baselines, neutral = _inputs(states, seeds, actions, baselines)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    policy = TerminalDQN(states.shape[1], len(actions), seed=settings['seed'],
        batch_size=settings['batch_size'], learning_rate=settings['learning_rate'],
        replay_capacity=settings['replay_capacity'])
    rng = random.Random(settings['seed'])
    cache = {}
    valid_actions = [set() for _ in states]
    errors = [set() for _ in states]
    all_actions = list(range(len(actions)))
    metrics = dict(episode=0, episodes=settings['episodes'], epsilon=settings['epsilon_start'],
        loss=None, reward=None, updates=0, model_calls=0, cache_hits=0, baseline_reuses=0,
        rollout_budget=settings['rollout_budget'], criterion=settings['criterion'],
        rollout_errors=0, observed_failures=0, observed_new_failures=0, valid_visits=0,
        skipped_error_visits=0, measured_pairs=0, unique_reward_pairs=0, replay_size=0,
        stopped_reason=None)
    current = dict(input_index=None, action_id=None)
    statistics = landscape = None
    last_statistics = last_progress = time.monotonic()
    last_snapshot_updates = 0
    previous_prediction = None

    def progress(*, snapshot=False, message=None):
        event = dict(type='progress', phase='search', search=dict(metrics), current=dict(current),
                     active_action_id=current['action_id'])
        if snapshot:
            event.update(policy_statistics=statistics, predicted_landscape=landscape)
        if message:
            event['message'] = message
        publish(event)

    def snapshot(*, final=False):
        nonlocal statistics, landscape, previous_prediction, last_snapshot_updates, last_statistics
        filename = 'dqn-predictions.npy' if final else f'dqn-predictions-{policy.updates}.npy'
        statistics, landscape = _predictions(policy, states, seeds, actions, valid_actions,
                                             settings, directory, filename)
        last_snapshot_updates = policy.updates
        last_statistics = time.monotonic()
        progress(snapshot=True)
        # Immutable checkpoint filenames prevent a reader from pairing a new
        # matrix with the previous progress event's update count.
        if previous_prediction is not None and previous_prediction.name != filename:
            previous_prediction.unlink(missing_ok=True)
        previous_prediction = directory/filename

    progress(message='Training DQN from measured episode variations.')
    history_path = directory/'dqn-history.jsonl'
    # History is intentionally append-only and line-buffered, retaining completed
    # steps if a worker is cancelled. Model and matrix snapshots are atomic.
    with history_path.open('w', encoding='utf-8', buffering=1) as history:
        for episode in range(1, settings['episodes']+1):
            can_rollout = metrics['model_calls'] < settings['rollout_budget']
            inputs = [index for index in range(len(states)) if
                      (len(errors[index]) < len(actions) if can_rollout else valid_actions[index] or neutral)]
            if not inputs:
                metrics['stopped_reason'] = ('no_valid_pairs' if can_rollout
                                             else 'rollout_budget_exhausted_without_rewards')
                break
            index = rng.choice(inputs)
            allowed = ([action for action in all_actions if action not in errors[index]] if can_rollout
                       else sorted(valid_actions[index] | neutral))
            fraction = (episode-1)/max(1, settings['episodes']-1)
            # Interpolate from the end to preserve the exact configured endpoints.
            epsilon = ((1-fraction)*settings['epsilon_start']+fraction*settings['epsilon_end'])
            action_index = _choose_allowed(policy, states[index], epsilon, allowed)
            action = actions[action_index]
            key = (index, action_index)
            current = dict(input_index=index, action_id=action['id'])
            metrics.update(episode=episode, epsilon=epsilon, reward=None, loss=None)
            if key in cache:
                record, reward = cache[key]
                source = 'cache'
                metrics['cache_hits'] += 1
            else:
                if action_index in neutral:
                    record = copy.deepcopy(baselines[index])
                    record.update(input_index=index, action_id=action['id'], reused_baseline=True)
                    publish(dict(type='record', record=record))
                    source = 'baseline'
                    metrics['baseline_reuses'] += 1
                else:
                    metrics['model_calls'] += 1
                    progress(message=f"Evaluating episode seed {seeds[index]} · action {action['id']}.")
                    record = evaluate(index, action)
                    source = 'rollout'
                    if not isinstance(record, dict) or any(record.get(field, expected) != expected
                            for field, expected in [('input_index', index), ('seed', seeds[index]),
                                                    ('action_id', action['id'])]):
                        raise ValueError('Rollout record does not match the requested episode and action.')
                valid = not record.get('error') and type(record.get('success')) is bool
                reward = (float(not record['success'] and (settings['criterion'] == 'task_failure'
                           or baselines[index]['success'])) if valid else None)
                cache[key] = (record, reward)
                if valid:
                    valid_actions[index].add(action_index)
                    metrics['unique_reward_pairs'] += 1
                    if source == 'rollout':
                        metrics['measured_pairs'] += 1
                        metrics['observed_failures'] += int(not record['success'])
                        metrics['observed_new_failures'] += int(baselines[index]['success'] and not record['success'])
                else:
                    errors[index].add(action_index)
                    metrics['rollout_errors'] += 1
            if reward is None:
                metrics['skipped_error_visits'] += 1
                loss = None
            else:
                metrics['valid_visits'] += 1
                loss = policy.observe(states[index], action_index, reward)
            metrics.update(reward=reward, loss=loss, updates=policy.updates, replay_size=len(policy.replay))
            row = dict(metrics, input_index=index, action_index=action_index, action_id=action['id'],
                       seed=seeds[index], source=source, cached=source == 'cache',
                       reused_baseline=bool(record.get('reused_baseline')))
            history.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
            now = time.monotonic()
            if policy.updates != last_snapshot_updates and now-last_statistics >= 2.:
                snapshot()
                last_progress = now
            elif episode == 1 or now-last_progress >= .25:
                progress()
                last_progress = now
    policy.save(directory/'dqn-policy.pt')
    snapshot(final=True)
    artifacts = dict(dqn_policy='dqn-policy.pt', dqn_predictions='dqn-predictions.npy',
                     dqn_history=history_path.name)
    for name, filename in artifacts.items():
        publish(dict(type='artifact', name=name, filename=filename))
    return dict(search=dict(metrics), current=dict(current), policy_statistics=statistics,
                predicted_landscape=landscape, artifacts=artifacts)
