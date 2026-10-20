"""Serialized background failure discovery and explicit reviewed-example training."""
import copy
import json
import math
import os
import random
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager, ExitStack, nullcontext
from pathlib import Path

import numpy as np

from classification_labels import resolve_target

from rl_policy import TerminalDQN


TERMINAL = frozenset(('completed', 'cancelled', 'error'))
IDENTITY = re.compile(r'[0-9a-f]{32}')
TRAINING_PROGRESS_FIELDS = ('completed', 'total', 'loss', 'phase', 'device', 'device_name',
                            'effective_batch_size', 'micro_batch_size', 'oom_retries', 'epoch', 'epochs',
                            'processed_examples', 'examples_per_epoch', 'total_examples')


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer between {minimum} and {maximum}.')
    return value


def _number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum or not math.isfinite(value):
        raise ValueError(f'{name} must be finite and between {minimum} and {maximum}.')
    return float(value)


def _identity(value):
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise ValueError('Invalid job ID.')
    return value


def _public(value, job_id=None):
    """Copy JSON data, removing large classifier vectors from every public view."""
    if isinstance(value, dict):
        result = {key: _public(item, job_id) for key, item in value.items() if key != 'probabilities'}
        if job_id and isinstance(result.get('image'), str):
            filename = result['image']
            if re.fullmatch(r'[A-Za-z0-9_-]+\.png', filename):
                result['image'] = f'/discoveries/{job_id}/artifacts/{filename}'
        return result
    if isinstance(value, (tuple, list)):
        return [_public(item, job_id) for item in value]
    return value


def _json(value):
    """Reject nonfinite values and non-JSON evaluator outputs at the boundary."""
    return json.loads(json.dumps(value, allow_nan=False))


class DiscoveryManager:
    HISTORY_LIMIT = 500
    PREDICTION_RAM_BYTES = 64 * 1024**2
    POLICY_REFRESH_SECONDS = 2.

    def __init__(self, previews, embeddings, directory, evaluator_factory=None, finetune_fn=None):
        self.previews, self.embeddings = previews, embeddings
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.evaluator_factory = evaluator_factory
        self.finetune_fn = finetune_fn
        self.lock = threading.RLock()
        self.jobs, self.finetunes = {}, {}
        self._contexts, self._events = {}, {}
        self._prediction_values = {}
        self._prediction_policies = {}
        self.active = None
        # Reports remain inspectable after a server restart; interrupted jobs do
        # not silently resume or occupy the active inference slot.
        for path in self.directory.glob('*/report.json'):
            try:
                report = json.loads(path.read_text())
                _identity(report['id'])
                if path.parent.name != report['id'] or path.parent.is_symlink():
                    continue
                interrupted = report['status'] not in TERMINAL
                if interrupted:
                    report.update(status='error', phase='interrupted', error='Server stopped before this job finished.')
                table = self.finetunes if report.get('kind') == 'finetune' else self.jobs
                table[report['id']] = report
                if interrupted:
                    try:
                        self._persist(report['id'])
                    except OSError:
                        pass  # Retain the recovered view even if storage is unavailable.
            except (ValueError, KeyError, OSError):
                continue

    def _context(self, embedding_id, *, include_states=True):
        context = self.embeddings.discovery_context(embedding_id, include_states=include_states)
        preview = context['preview']
        if include_states:
            states = np.asarray(context['states'], dtype=np.float32).view()
            if (states.ndim != 2 or not len(states) or not states.shape[1] or
                    len(states) != preview['input_count']):
                raise ValueError('Analysis must contain finite original input embeddings.')
            for offset in range(0, len(states), 1024):
                batch = states[offset:offset+1024]
                if not np.isfinite(batch).all():
                    raise ValueError('Analysis must contain finite original input embeddings.')
                if not np.allclose(np.linalg.norm(batch, axis=1), 1., rtol=1e-4, atol=1e-5):
                    raise ValueError('Discovery requires normalized original model embeddings.')
            states.setflags(write=False)
            context['states'] = states
        actions = preview['actions']
        ids = [action['id'] for action in actions]
        if not ids or any(type(action_id) is not int or action_id < 0 for action_id in ids) or len(set(ids)) != len(ids):
            raise ValueError('Analysis must contain distinct fixed action IDs.')
        dataset_id = preview['config'].get('dataset_id')
        if preview['config']['task'] == 'summarization' and dataset_id:
            dataset = self.previews.datasets.ready(dataset_id)
            if dataset.get('task') != 'summarization' or dataset['count'] != preview['input_count']:
                raise ValueError('The article dataset does not match this analysis. Analyze again.')
            # Keep the corpus in its dataset storage. Adapt also uses this accessor
            # after analysis features have been released, without copying articles.
            context['summary_dataset'] = dataset
            context['summary_row'] = lambda index: self.previews.datasets.text(dataset_id, index)
        elif preview['config']['task'] == 'generation' and dataset_id:
            dataset = self.previews.datasets.ready(dataset_id)
            if dataset.get('task') != 'generation' or dataset['count'] != preview['input_count']:
                raise ValueError('The prompt dataset does not match this analysis. Analyze again.')
            context['generation_dataset'] = dataset
            context['generation_row'] = lambda index: self.previews.datasets.text(dataset_id, index)
        return context

    def _config(self, data, context):
        allowed = {'embedding_id', 'episodes', 'seed', 'label_map', 'reference_summary', 'summary_metric',
                   'failure_threshold', 'generation_steps', 'generation_size', 'generation_metric', 'judge_model',
                   'generation_bias_suite', 'generation_bias_repeats', 'generation_precision', 'max_new_tokens', 'device'}
        if not isinstance(data, dict) or set(data)-allowed:
            raise ValueError('Unknown discovery settings.')
        task = context['preview']['config']['task']
        combined = task == 'generation' and data.get('generation_metric', 'combined') == 'combined'
        config = dict(task=task, episodes=_integer(data.get('episodes', 200), 'Episodes', 1, 10000),
                      seed=_integer(data.get('seed', 42), 'Seed', 0, 2**32-1),
                      failure_threshold=_number(data.get('failure_threshold', .5 if combined else .15), 'Failure threshold', 0, 1),
                      generation_steps=_integer(data.get('generation_steps', 12), 'Generation steps', 1, 100),
                      generation_size=_integer(data.get('generation_size', 256), 'Generation size', 128, 1024),
                      max_new_tokens=_integer(data.get('max_new_tokens', 96), 'Output length', 1, 512),
                      summary_metric=data.get('summary_metric', 'grounded'), device=data.get('device', 'auto'),
                      reference_summary=data.get('reference_summary', ''), label_map=data.get('label_map', {}))
        if config['generation_size'] % 8:
            raise ValueError('Generation size must be a multiple of eight.')
        if config['device'] not in ('cpu', 'auto', 'cuda') or config['summary_metric'] not in ('grounded', 'reference'):
            raise ValueError('Choose a supported device and summary metric.')
        if not isinstance(config['reference_summary'], str) or len(config['reference_summary']) > 50000:
            raise ValueError('Reference summary must be text of at most 50,000 characters.')
        if task == 'summarization':
            dataset = context.get('summary_dataset')
            if dataset:
                if config['reference_summary'].strip():
                    raise ValueError('This dataset uses per-article references; a single reference cannot override every row.')
                missing = dataset['count'] - dataset['reference_count']
                if missing:
                    raise ValueError(f'{missing} article references are missing. Choose a reference-summary column '
                                     'with a nonempty summary for every selected row.')
                config.update(dataset_id=dataset['id'], dataset=copy.deepcopy(dataset), reference_mode='dataset')
            elif not config['reference_summary'].strip():
                raise ValueError('Summary scoring requires an explicit reference summary.')
        if task == 'generation':
            from generation_judge import DEFAULT_JUDGE, JUDGE_MODELS, COMBINED_JUDGES
            from generation_bias import normalize_bias_suite
            dataset = context.get('generation_dataset')
            metric = data.get('generation_metric', 'combined')
            precision = data.get('generation_precision', 'auto')
            if precision not in ('auto', 'float32'):
                raise ValueError('Choose automatic or full float32 generation precision.')
            if metric not in ('combined', 'checks', 'clip'):
                raise ValueError('Choose combined generation failure scoring or an explicit legacy metric.')
            judge = data.get('judge_model', DEFAULT_JUDGE)
            if judge not in JUDGE_MODELS:
                raise ValueError('Choose a supported visual judge.')
            if combined and judge not in COMBINED_JUDGES:
                raise ValueError('Combined failure scoring requires a Vev or Qwen3-VL judge.')
            if combined:
                from core import LEGACY_GENERATION_SPACE
                if any(set(action) & set(LEGACY_GENERATION_SPACE) for action in context['preview']['actions']):
                    raise ValueError('Combined scoring requires presentation actions. Analyze with Style, Lighting, or Detail.')
            suite = normalize_bias_suite(data.get('generation_bias_suite'))
            repeats = _integer(data.get('generation_bias_repeats', 3), 'Bias comparison repeats', 2, 16)
            if suite is not None and not combined:
                raise ValueError('Bias comparisons require the combined reward.')
            if metric == 'checks':
                missing = dataset['count'] - dataset.get('checked_count', 0) if dataset else 1
                if missing:
                    raise ValueError(f'{missing} prompts have no requirements. Choose a checks column with questions '
                                     'and expected answers for every row, or select CLIP scene alignment.')
            config.update(generation_metric=metric, judge_model=judge, generation_precision=precision,
                          generation_bias_suite=suite, generation_bias_repeats=repeats)
            if dataset:
                config.update(dataset_id=dataset['id'], dataset=copy.deepcopy(dataset), requirements_mode='dataset')
        labels = config['label_map']
        if task != 'generation' and ({'generation_bias_suite', 'generation_bias_repeats'} & set(data)):
            raise ValueError('Bias comparison settings apply to image generation.')
        if task != 'generation' and 'generation_precision' in data:
            raise ValueError('Generation precision applies only to image generation.')
        if not isinstance(labels, dict) or len(labels) > len(context['states']):
            raise ValueError('Labels must map original input indices to model labels.')
        for key, label in labels.items():
            if not isinstance(key, str) or not re.fullmatch(r'0|[1-9][0-9]*', key) or int(key) >= len(context['states']):
                raise ValueError('Label input index is outside this analysis.')
            if not ((type(label) is int and label >= 0) or (isinstance(label, str) and label.strip() and len(label) <= 512)):
                raise ValueError('Use explicit model class IDs or exact model label names.')
        if task != 'classification' and labels:
            raise ValueError('Class labels apply only to classification.')
        if task == 'classification':
            imported = context['job'].get('labels', {})
            labels = {**imported.get('label_map', {}), **labels}
            missing = len(context['states']) - len(labels)
            if missing:
                raise ValueError(f'{missing} input labels need review. Upload JSON or enter the missing labels before discovery.')
            if imported.get('model_labels'):
                resolved = {}
                for key, value in labels.items():
                    target = (type(value), value)
                    if target not in resolved:
                        resolved[target] = resolve_target(value, imported['model_labels'], imported.get('groups'))
                    labels[key] = resolved[target]
            config['label_map'] = labels
        config['source_text'] = '' if context.get('summary_dataset') or context.get('generation_dataset') else context['preview']['config'].get('text', '')
        return _json(config)

    def start(self, config):
        if not isinstance(config, dict):
            raise ValueError('Discovery settings must be a JSON object.')
        embedding_id = _identity(config.get('embedding_id'))
        context = self._context(embedding_id)
        settings = self._config(config, context)
        job_id = uuid.uuid4().hex
        job = dict(id=job_id, embedding_id=embedding_id, preview_id=context['preview']['id'],
                   task=settings['task'], status='queued', phase='queued', episode=0,
                   episodes=settings['episodes'], epsilon=1., loss=None, reward=None,
                   mean_reward=0., model_calls=0, comparison_model_calls=0, unique_evaluations=0, criterion='',
                   config=settings, model=copy.deepcopy(context['model']), current=None,
                   records=[], history=[], baseline=[], artifacts={}, policy_statistics=None,
                   predicted_landscape=None)
        return self._launch(job, context, self._discover)

    def _launch(self, job, context, worker):
        job_id = job['id']
        table = self.finetunes if job.get('kind') == 'finetune' else self.jobs
        with self.lock:
            self._reserve(job_id)
            try:
                (self.directory/job_id).mkdir()
                table[job_id], self._contexts[job_id] = job, context
                self._events[job_id] = threading.Event()
                self._persist(job_id)
                threading.Thread(target=worker, args=(job_id,), daemon=True).start()
            except Exception:
                self.active = None
                table.pop(job_id, None)
                self._contexts.pop(job_id, None)
                self._events.pop(job_id, None)
                raise
            return self.get(job_id) if table is self.jobs else self.get_finetune(job_id)

    def _reserve(self, job_id):
        if self.active is not None:
            raise RuntimeError('A discovery or fine-tuning job is already active. Cancel it or wait for completion.')
        self.active = job_id

    def get(self, job_id):
        with self.lock:
            return _public(self.jobs[_identity(job_id)], job_id)

    def list(self):
        with self.lock:
            return [self.get(job_id) for job_id in reversed(self.jobs)]

    def predictions(self, job_id, input_index):
        """Read one input's current DQN estimates without running the tested model."""
        with self.lock:
            job = self.jobs[_identity(job_id)]
            landscape = job.get('predicted_landscape')
            if not landscape or not landscape['policy_updates']:
                raise ValueError('No trained predictions are available for this run yet.')
            index = _integer(input_index, 'Input index', 0, len(landscape['input_mean_reward'])-1)
            values = self._prediction_values.get(job_id)
            snapshot = self._prediction_policies.get(job_id)
            if values is None and snapshot is not None:
                import torch
                with torch.inference_mode():
                    row = snapshot['network'](torch.tensor(snapshot['states'][index], dtype=torch.float32)).numpy().astype(np.float64)
            else:
                if values is None:
                    filename = landscape.get('values_file')
                    if not filename:
                        raise ValueError('Predictions were interrupted before saving. Run discovery again.')
                    values = np.load(self.artifact(job_id, filename), mmap_mode='r', allow_pickle=False)
                if values.shape != (len(landscape['input_mean_reward']), len(landscape['action_ids'])):
                    raise ValueError('Saved predictions do not match this discovery run.')
                row = np.array(values[index], dtype=np.float64, copy=True)
            if not np.isfinite(row).all():
                raise ValueError('Saved predictions contain nonfinite values.')
            preferences = np.exp(row-row.max())
            preferences /= preferences.sum()
            return dict(discovery_id=job_id, input_index=index, policy_updates=landscape['policy_updates'],
                        actions=[dict(action_id=action_id, q_value=float(value), preference=float(preference))
                                 for action_id, value, preference in zip(landscape['action_ids'], row, preferences)])

    def get_finetune(self, job_id):
        with self.lock:
            return _public(self.finetunes[_identity(job_id)])

    def _adaptation_context(self, job_id):
        discovery = self.get(job_id)
        if discovery['status'] not in TERMINAL or not discovery['records']:
            raise ValueError('Finish or cancel discovery with at least one measured result before adaptation.')
        context = self._context(discovery['embedding_id'], include_states=False)
        if (context['preview']['id'] != discovery['preview_id'] or
                context['preview']['config']['task'] != discovery['task']):
            raise ValueError('Discovery no longer matches its analysis. Analyze again.')
        return discovery, context

    def adaptation(self, job_id):
        from adaptation import summarize
        discovery, context = self._adaptation_context(job_id)
        return _json(summarize(discovery, context))

    def cancel(self, job_id):
        return self._cancel(job_id, self.jobs)

    def cancel_finetune(self, job_id):
        return self._cancel(job_id, self.finetunes)

    def _cancel(self, job_id, table):
        with self.lock:
            job = table[_identity(job_id)]
            if job['status'] not in TERMINAL:
                self._events[job_id].set()
                job['phase'] = 'cancelling after current model operation'
            return self.get(job_id) if table is self.jobs else self.get_finetune(job_id)

    def _check(self, job_id):
        if self._events[job_id].is_set():
            raise InterruptedError('Cancelled by user.')

    @contextmanager
    def _inference_access(self, job_id):
        while not self.embeddings.model_lock.acquire(timeout=.1):
            self._check(job_id)
        try:
            self._check(job_id)
            yield
        finally:
            self.embeddings.model_lock.release()

    @contextmanager
    def _model_access(self, job_id, *, retain_models=False):
        with self._preview_access(job_id), self._inference_access(job_id):
            context = self._contexts[job_id]
            fresh = self._context(context['job']['id'], include_states=False)
            if fresh['model_key'] != context['model_key']:
                raise ValueError('The model changed while this job was waiting. Analyze again.')
            del fresh
            if not retain_models:
                self.embeddings.release_encoder()
            yield

    def _preview_access(self, job_id):
        pin = getattr(self.previews, 'pin_preview', None)
        return pin(self._contexts[job_id]['preview']['id']) if callable(pin) else nullcontext()

    def _close_evaluator(self, evaluator):
        with self.embeddings.model_lock:
            evaluator.close()

    def _update(self, job_id, **fields):
        with self.lock:
            table = self.jobs if job_id in self.jobs else self.finetunes
            table[job_id].update(fields)

    def _progress(self, job_id, **fields):
        fields = _json(fields)
        with self.lock:
            job = self.jobs.get(job_id) or self.finetunes[job_id]
            if self._events[job_id].is_set():
                return
            if job_id in self.finetunes:
                # Training adapters may report either step or completed.
                if 'step' in fields and 'completed' not in fields:
                    fields['completed'] = fields['step']
                if 'training_loss' in fields and 'loss' not in fields:
                    fields['loss'] = fields['training_loss']
                if 'adaptation_selection' in job:
                    if fields.get('phase') == 'validating reviewed examples':
                        fields['phase'] = 'preparing training examples'
                    elif fields.get('phase') == 'training reviewed examples':
                        fields['phase'] = 'training selected actions'
                job.update({key: fields[key] for key in TRAINING_PROGRESS_FIELDS if key in fields})
            elif 'phase' in fields:
                job['phase'] = fields['phase']

    def _input(self, context, index, action_id=None):
        preview = context['preview']
        if preview['config']['task'] == 'classification':
            from core import apply_image
            original = self.previews.original_image(preview['id'], index)
            if action_id is None:
                return original
            action = next(action for action in preview['actions'] if action['id'] == action_id)
            return apply_image(original, action)
        if preview['config']['task'] in ('summarization', 'generation') and preview['config'].get('dataset_id'):
            preview = self.previews.get(preview['id'], index)
        if action_id is None:
            return preview['original']['text']
        return next(node['text'] for node in preview['nodes'] if node['id'] == action_id)

    def _training_input(self, context, index, action_id=None):
        return self._input(context, index, action_id)

    def _predict(self, job_id, evaluator, value, index, artifact_name, *, action=None):
        self._check(job_id)
        with self.lock:
            job = self.jobs[job_id]
            job['model_calls'] += 1
            seed = (job['config']['seed']+index) % 2**32
        # Count actual attempts, including a failed operation. The same seed is
        # deliberately used for each input's baseline and all its variants.
        options = {}
        context = self._contexts[job_id]
        if context.get('summary_dataset'):
            options['reference_summary'] = context['summary_row'](index)['summary']
        if context['preview']['config']['task'] == 'generation':
            options['reference_prompt'] = self._input(context, index)
            if job['config'].get('generation_metric') in ('combined', 'checks'):
                options['checks'] = context['generation_row'](index)['checks'] if context.get('generation_row') else []
            if job['config'].get('generation_metric') == 'combined':
                options['action'] = copy.deepcopy(action)
        return _json(evaluator.predict(value, seed=seed, artifact_name=artifact_name, **options))

    def _comparison_call(self, job_id):
        self._check(job_id)
        with self.lock:
            self.jobs[job_id]['model_calls'] += 1
            self.jobs[job_id]['comparison_model_calls'] += 1

    def _score(self, evaluator, output, baseline, target):
        score = _json(evaluator.score(output, baseline, target=target))
        score['reward'] = _number(score.get('reward'), 'Evaluator reward', -1, 1)
        score['severity'] = _number(score.get('severity'), 'Evaluator severity', 0, 1)
        if type(score.get('failure')) is not bool or not isinstance(score.get('criterion'), str) or not isinstance(score.get('metrics'), dict):
            raise ValueError('Evaluator returned an invalid measured score.')
        return score

    def _policy_statistics(self, job_id, policy, context, records, *, persist=False):
        actions = context['preview']['actions']
        shape = (len(context['states']), len(actions))
        byte_count = math.prod(shape) * np.dtype(np.float32).itemsize
        values, means, snapshot, temporary = None, None, None, None
        filename = None
        try:
            if policy.updates:
                means = np.empty(shape[0], dtype=np.float32)
                if persist:
                    # Write final rewards a batch at a time. This file is part
                    # of the discovery report; no second working copy is kept.
                    try:
                        budget = int(os.environ.get('FAILURESHIFT_PREDICTION_DISK_MB', '1024')) * 1024**2
                    except ValueError as exc:
                        raise ValueError('FAILURESHIFT_PREDICTION_DISK_MB must be a positive integer.') from exc
                    if budget <= 0 or byte_count + 4096 > budget:
                        raise ValueError('Final predictions exceed the disk budget. Set FAILURESHIFT_PREDICTION_DISK_MB '
                                         'to a larger budget or analyze a smaller input range.')
                    if byte_count + 4096 > shutil.disk_usage(self.directory).free:
                        raise ValueError('Not enough disk space to save final discovery predictions.')
                    filename = f'predictions-{policy.updates}.npy'
                    temporary = self.directory / job_id / (filename + '.tmp')
                    values = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=shape)
                    if hasattr(os, 'posix_fallocate'):
                        with temporary.open('r+b') as stream:
                            os.posix_fallocate(stream.fileno(), 0, temporary.stat().st_size)
                elif byte_count <= self.PREDICTION_RAM_BYTES:
                    values = np.empty(shape, dtype=np.float32)
                else:
                    # Keep only the small policy snapshot and shared features.
                    # Inspecting one original evaluates its row on demand.
                    snapshot = dict(network=copy.deepcopy(policy.network).eval(), states=context['states'])
            statistics = policy.action_statistics(context['states'], q_values_out=values, input_mean_out=means)
            if temporary is not None:
                values.flush()
                values._mmap.close()
                values = None
                temporary.replace(self.directory / job_id / filename)
        finally:
            if temporary is not None:
                if isinstance(values, np.memmap):
                    values._mmap.close()
                temporary.unlink(missing_ok=True)
        coverage = {}
        for record in records:
            coverage.setdefault(record['action_id'], set()).add(record['input_index'])
        for action in statistics['actions']:
            action['action_id'] = actions[action.pop('action_index')]['id']
            action['evaluated_inputs'] = len(coverage.get(action['action_id'], ()))
        landscape = None
        if means is not None:
            landscape = dict(method='terminal_q_reward', policy_updates=policy.updates,
                             input_mean_reward=means.tolist(),
                             action_ids=[action['id'] for action in actions], color_range=[0, 1])
            if filename:
                landscape['values_file'] = filename
        with self.lock:
            if values is not None and not persist:
                self._prediction_values[job_id] = values
            else:
                self._prediction_values.pop(job_id, None)
            if snapshot is not None:
                self._prediction_policies[job_id] = snapshot
            else:
                self._prediction_policies.pop(job_id, None)
            self.jobs[job_id].update(policy_statistics=statistics, predicted_landscape=landscape)

    def _discover(self, job_id):
        evaluator, policy = None, None
        status, error = 'completed', None
        context = self._contexts[job_id]
        config = self.jobs[job_id]['config']
        directory = self.directory/job_id
        try:
            self._update(job_id, status='running', phase='waiting for model')
            with ExitStack() as cleanup:
                cleanup.enter_context(self._preview_access(job_id))
                with self._model_access(job_id, retain_models=True):
                    self._update(job_id, phase='loading selected model and scorers')
                    factory = self.evaluator_factory or self.embeddings.evaluation_model
                    evaluator = factory(context['model'], config, directory,
                                        progress=lambda **fields: self._progress(job_id, **fields))
                    cleanup.callback(self._close_evaluator, evaluator)
                    if config['task'] == 'generation' and hasattr(evaluator, 'set_runtime_callbacks'):
                        evaluator.set_runtime_callbacks(
                            cancelled=self._events[job_id].is_set,
                            on_model_call=lambda:self._comparison_call(job_id),
                            progress=lambda **fields:self._progress(job_id, **fields))
                    self._check(job_id)
                    if config['task'] == 'classification' and config['label_map']:
                        resolve = getattr(evaluator, 'resolve_label', None)
                        if resolve is None:
                            raise ValueError('The classification evaluator cannot validate explicit model labels.')
                        for target in config['label_map'].values():
                            resolve(target)
                    self._update(job_id, scoring=_json(evaluator.describe()))
                states = context['states']
                actions = context['preview']['actions']
                policy = TerminalDQN(states.shape[1], len(actions), seed=config['seed'])
                rng = random.Random(config['seed'])
                baseline_cache, pair_cache, aggregates, nodes = {}, {}, {}, {}
                total_reward = 0.
                next_statistics, next_report = 0., 0.
                last_statistics_update = 0
                with (directory/'episodes.jsonl').open('a') as stream:
                    for episode in range(1, config['episodes']+1):
                        self._check(job_id)
                        self.embeddings.wait_for_expansion(context['job']['id'], self._events[job_id])
                        index = rng.randrange(len(states))
                        epsilon = 1.-.4*(episode-1)/max(1, config['episodes']-1)
                        action_index = policy.choose(states[index], epsilon=epsilon)
                        action_id = actions[action_index]['id']
                        pair = (index, action_id)
                        target = config['label_map'].get(str(index))
                        self._update(job_id, phase='evaluating baseline' if index not in baseline_cache else 'evaluating action',
                                     epsilon=epsilon, current={'input_index': index, 'action_id': action_id})
                        # Expansion and evaluation share weights, so serialize
                        # model work while leaving gaps for interactive graphs.
                        projection_error = None
                        with self._inference_access(job_id):
                            if index not in baseline_cache:
                                output = self._predict(job_id, evaluator, self._input(context, index), index, f'input-{index}-baseline')
                                baseline_cache[index] = output
                                score = self._score(evaluator, output, output, target)
                                baseline = dict(input_index=index, output=output, **score)
                                if context.get('summary_dataset'):
                                    baseline['reference_summary'] = context['summary_row'](index)['summary']
                                with self.lock:
                                    self.jobs[job_id]['baseline'].append(baseline)
                            self._check(job_id)
                            cached = pair in pair_cache
                            if not cached:
                                self._update(job_id, phase='evaluating action')
                                output = self._predict(job_id, evaluator, self._input(context, index, action_id), index,
                                                       f'input-{index}-action-{action_id}', action=actions[action_index])
                                score = self._score(evaluator, output, baseline_cache[index], target)
                                pair_cache[pair] = (output, score)
                            if pair not in nodes and not self._events[job_id].is_set():
                                try:
                                    nodes[pair] = self.embeddings.discovery_node(
                                        context['job']['id'], index, action_id, evaluator=evaluator)
                                except Exception as exc:
                                    # Keep the completed model measurement even
                                    # when its graph projection cannot be built.
                                    projection_error = exc
                        output, score = pair_cache[pair]
                        # A completed measurement remains useful even if a cancel
                        # arrives during its model operation.
                        loss = policy.observe(states[index], action_index, score['reward'])
                        total_reward += score['reward']
                        previous = aggregates.get(pair)
                        visits = previous['visits']+1 if previous else 1
                        record = dict(input_index=index, action_id=action_id, visits=visits,
                                      severity=score['severity'], failure_rate=float(score['failure']),
                                      reward=score['reward'], failure=score['failure'], output=output,
                                      metrics=score['metrics'], criterion=score['criterion'])
                        if pair in nodes:
                            record['node'] = nodes[pair]
                        aggregates[pair] = record
                        entry = dict(episode=episode, input_index=index, action_id=action_id,
                                     epsilon=epsilon, loss=loss, reward=score['reward'], severity=score['severity'],
                                     cached=cached, model_calls=self.jobs[job_id]['model_calls'])
                        stream.write(json.dumps(_public(entry, job_id), allow_nan=False)+'\n')
                        stream.flush()
                        if policy.updates > last_statistics_update and time.monotonic() >= next_statistics:
                            started = time.monotonic()
                            self._policy_statistics(job_id, policy, context, aggregates.values())
                            elapsed = time.monotonic() - started
                            last_statistics_update = policy.updates
                            # Full graph scans get at most a tenth of the loop's
                            # wall time. A final full scan always follows below.
                            next_statistics = time.monotonic() + max(self.POLICY_REFRESH_SECONDS, elapsed * 9)
                        with self.lock:
                            job = self.jobs[job_id]
                            job['history'] = (job['history']+[entry])[-self.HISTORY_LIMIT:]
                            job.update(episode=episode, reward=score['reward'], mean_reward=total_reward/episode,
                                       loss=loss, records=list(aggregates.values()), unique_evaluations=len(pair_cache),
                                       criterion=score['criterion'], current=record, phase='learning from terminal reward')
                            if time.monotonic() >= next_report:
                                started = time.monotonic()
                                self._persist(job_id)
                                next_report = time.monotonic() + max(2., (time.monotonic()-started) * 9)
                        if projection_error is not None:
                            raise projection_error
                self._check(job_id)
        except InterruptedError:
            status = 'cancelled'
        except Exception as exc:
            status, error = 'error', str(exc)
        finally:
            if policy is not None:
                try:
                    policy.save(directory/'policy.pt')
                    self._policy_statistics(job_id, policy, context, self.jobs[job_id]['records'], persist=True)
                except Exception as exc:
                    status, error = 'error', f'Could not save discovery results: {exc}'
            self._finish(job_id, status, error)

    def _persist(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id) or self.finetunes[job_id]
            directory = self.directory/job_id
            if job_id in self.jobs:
                job['artifacts'] = {'report': f'/discoveries/{job_id}/artifacts/report.json',
                                    'episodes': f'/discoveries/{job_id}/artifacts/episodes.jsonl',
                                    'policy': f'/discoveries/{job_id}/artifacts/policy.pt' if (directory/'policy.pt').exists() else None}
                filename = (job.get('predicted_landscape') or {}).get('values_file')
                if filename:
                    job['artifacts']['predictions'] = f'/discoveries/{job_id}/artifacts/{filename}'
            else:
                job['artifacts'] = {'report': f'/discoveries/{job_id}/artifacts/report.json'}
            report = _public(job, job_id if job_id in self.jobs else None)
            temporary = directory/'report.json.tmp'
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
            temporary.replace(directory/'report.json')

    def _finish(self, job_id, status, error):
        with self.lock:
            job = self.jobs.get(job_id) or self.finetunes[job_id]
            if status == 'completed' and self._events[job_id].is_set():
                status = 'cancelled'
            job.update(status=status, phase=status)
            if error:
                job['error'] = error
            try:
                self._persist(job_id)
            except Exception as exc:
                job.update(status='error', phase='error', error=f'Could not save job report: {exc}')
            else:
                if job_id in self.finetunes and job['status'] == 'completed':
                    self._cleanup_training(job)
                    try:
                        self._persist(job_id)
                    except Exception as exc:
                        job['cleanup'] = dict(status='error', message=f'Checkpoint saved; could not update cleanup report: {exc}')
            finally:
                if self.active == job_id:
                    self.active = None
                self._contexts.pop(job_id, None)
                if job_id in self.jobs:
                    self._prediction_values.pop(job_id, None)
                    self._prediction_policies.pop(job_id, None)

    def _cleanup_training(self, job):
        """Release this run's disposable data only after its result is saved."""
        if not job['settings']['cleanup_after_finetune']:
            job['cleanup'] = dict(status='disabled', message='Working data retained.')
            return
        discovery = self.jobs[job['discovery_id']]
        released_memory, released_disk, skipped = 0, 0, False
        try:
            operations = ((self.embeddings, 'release_analysis', discovery['embedding_id']),
                          (self.previews, 'release_preview', discovery['preview_id']))
            for manager, method, identity in operations:
                release = getattr(manager, method, None)
                if not callable(release):
                    skipped = True
                    continue
                result = release(identity)
                released_memory += result.get('released_memory_bytes', 0)
                released_disk += result.get('released_disk_bytes', 0)
                skipped = skipped or result['status'] not in ('released', 'already_released', 'completed')
            message = ('Some working data is still in use and was retained.' if skipped else
                       'Working data cleared. Checkpoint, reports and source data retained.')
            job['cleanup'] = dict(status='skipped' if skipped else 'completed', message=message,
                                  released_memory_bytes=released_memory, released_disk_bytes=released_disk)
        except Exception as exc:
            job['cleanup'] = dict(status='error', message=f'Checkpoint saved; working data cleanup failed: {exc}',
                                  released_memory_bytes=released_memory, released_disk_bytes=released_disk)

    def artifact(self, job_id, filename):
        _identity(job_id)
        with self.lock:
            if job_id not in self.jobs:
                self.finetunes[job_id]
                if filename != 'report.json':
                    raise ValueError('Only the training report is available for download.')
        if not isinstance(filename, str) or not re.fullmatch(r'(?:report\.json|episodes\.jsonl|policy\.pt|predictions-\d+\.npy|[A-Za-z0-9_-]+\.png)', filename):
            raise ValueError('Unknown discovery artifact.')
        folder = self.directory/job_id
        directory = folder.resolve()
        if folder.is_symlink() or directory.parent != self.directory.resolve():
            raise FileNotFoundError(filename)
        path = directory/filename
        if path.is_symlink() or path.resolve().parent != directory or not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def start_finetune(self, config):
        from adaptation import _records, _image_path
        allowed = {'discovery_id', 'examples', 'epochs', 'learning_rate', 'batch_size', 'seed',
                   'action_ids', 'include_originals', 'training_data', 'device',
                   'weight_decay', 'max_grad_norm', 'cleanup_after_finetune'}
        if not isinstance(config, dict) or set(config)-allowed:
            raise ValueError('Unknown fine-tuning settings.')
        if 'action_ids' in config:
            if 'examples' in config:
                raise ValueError('Choose action_ids or reviewed examples, not both.')
            return self._start_adaptation(config)
        if {'include_originals', 'training_data'} & set(config):
            raise ValueError('include_originals and training_data require an action_ids selection.')
        discovery_id = _identity(config.get('discovery_id'))
        discovery = self.get(discovery_id)
        if discovery['status'] not in TERMINAL:
            raise ValueError('Finish or cancel discovery before reviewing examples for fine-tuning.')
        examples = _records(config.get('examples'), 'examples')
        pairs = {(record['input_index'], record['action_id']) for record in discovery['records']}
        reviewed, seen = [], set()
        for example in examples:
            if not isinstance(example, dict) or set(example) != {'input_index', 'action_id', 'target'}:
                raise ValueError('Each reviewed example needs input_index, action_id and an explicit desired target.')
            index, action, target = example['input_index'], example['action_id'], example['target']
            if type(index) is not int or type(action) is not int or (index, action) not in pairs:
                raise ValueError('Fine-tuning accepts only measured input/action pairs from this discovery.')
            if (index, action) in seen:
                raise ValueError('Each reviewed input/action pair must appear only once.')
            task = discovery['task']
            if task == 'classification':
                if not ((type(target) is int and target >= 0) or (isinstance(target, str) and target.strip() and len(target) <= 512)):
                    raise ValueError('Supply an explicit model class ID or exact class label.')
            elif not isinstance(target, str) or not target.strip() or len(target) > 50000:
                raise ValueError('Supply an explicit reviewed target summary or desired image path.')
            if task == 'generation':
                target = _image_path(target)
            seen.add((index, action))
            reviewed.append(dict(input_index=index, action_id=action, target=target))
        settings = self._finetune_settings(config, discovery)
        context = self._context(discovery['embedding_id'], include_states=False)
        job_id = uuid.uuid4().hex
        job = dict(id=job_id, kind='finetune', discovery_id=discovery_id, task=discovery['task'],
                   status='queued', phase='queued', loss=None, checkpoint=None, examples=reviewed, settings=settings,
                   **self._training_counts(len(reviewed), settings))
        return self._launch(job, context, self._finetune)

    def _finetune_settings(self, config, discovery):
        device = config.get('device', 'auto')
        if not isinstance(device, str) or device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('Choose a fine-tuning device: auto, cpu, or cuda.')
        batch_size = config.get('batch_size', 'auto')
        if batch_size != 'auto':
            batch_size = _integer(batch_size, 'Batch size', 1, 1024)
        cleanup = config.get('cleanup_after_finetune', True)
        if type(cleanup) is not bool:
            raise ValueError('Cleanup after fine-tuning must be true or false.')
        return dict(device=device, epochs=_integer(config.get('epochs', 1), 'Epochs', 1, 100),
                    learning_rate=_number(config.get('learning_rate', .00005), 'Learning rate', 1e-8, .1),
                    batch_size=batch_size,
                    weight_decay=_number(config.get('weight_decay', .01), 'Weight decay', 0, 1),
                    max_grad_norm=_number(config.get('max_grad_norm', 1.), 'Maximum gradient norm', 0, 100),
                    cleanup_after_finetune=cleanup,
                    seed=_integer(config.get('seed', discovery['config']['seed']), 'Seed', 0, 2**32-1),
                    generation_size=discovery['config']['generation_size'])

    @staticmethod
    def _training_counts(example_count, settings):
        batch_size = settings['batch_size']
        automatic = batch_size == 'auto'
        return dict(completed=0, total=None if automatic else math.ceil(example_count / batch_size) * settings['epochs'],
                    example_count=example_count, examples_per_epoch=example_count,
                    total_examples=example_count * settings['epochs'], processed_examples=0,
                    epoch=0, epochs=settings['epochs'], effective_batch_size=None if automatic else batch_size,
                    micro_batch_size=None, oom_retries=0)

    def _start_adaptation(self, config):
        from adaptation import prepare
        discovery, context = self._adaptation_context(_identity(config.get('discovery_id')))
        settings = self._finetune_settings(config, discovery)
        specification = _json(prepare(discovery, context, config))
        job = dict(id=uuid.uuid4().hex, kind='finetune', discovery_id=discovery['id'], task=discovery['task'],
                   status='queued', phase='queued', loss=None, checkpoint=None, settings=settings, **specification,
                   **self._training_counts(specification['adaptation_selection']['example_count'], settings))
        return self._launch(job, context, self._finetune)

    def _finetune(self, job_id):
        status, error = 'completed', None
        job = self.finetunes[job_id]
        context = self._contexts[job_id]
        directory = self.directory/job_id
        try:
            self._update(job_id, status='running', phase='waiting for model')
            with self._model_access(job_id):
                adapting = 'adaptation_selection' in job
                self._update(job_id, phase='preparing training examples' if adapting else 'preparing reviewed training examples')
                from adaptation import TrainingExamples
                examples = TrainingExamples(job, context, self._training_input, lambda: self._check(job_id))
                self._check(job_id)
                train = self.finetune_fn
                if train is None:
                    from fine_tuning import fine_tune
                    train = fine_tune
                self._update(job_id, phase='training selected actions' if adapting else 'training reviewed examples')
                result = _json(train(context['model'], job['task'], examples, directory, job['settings'],
                                     lambda **fields: self._progress(job_id, **fields),
                                     lambda: self._events[job_id].is_set()))
                self._check(job_id)
                checkpoint = Path(result['checkpoint']).resolve()
                if not checkpoint.is_dir() or directory.resolve() not in checkpoint.parents:
                    raise ValueError('Training did not return a separate saved checkpoint inside its job directory.')
                loss = result.get('training_loss', result.get('loss', job['loss']))
                if isinstance(loss, list):
                    loss = loss[-1] if loss else job['loss']
                if loss is not None:
                    loss = _number(loss, 'Final training loss', 0, 1e30)
                fields = {key: result[key] for key in TRAINING_PROGRESS_FIELDS if key in result}
                fields.update(result=result, checkpoint=str(checkpoint), loss=loss,
                              completed=result.get('steps', job['completed']),
                              total=result.get('total', job['total'] if job['total'] is not None else result.get('steps')))
                self._update(job_id, **fields)
        except InterruptedError:
            status = 'cancelled'
        except Exception as exc:
            status, error = 'error', str(exc)
        finally:
            self._finish(job_id, status, error)
