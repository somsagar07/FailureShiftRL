"""Serialized background failure discovery and explicit reviewed-example training."""
import copy
import io
import json
import math
import random
import re
import threading
import uuid
from contextlib import contextmanager, ExitStack
from pathlib import Path

import numpy as np

from rl_policy import TerminalDQN


TERMINAL = frozenset(('completed', 'cancelled', 'error'))
IDENTITY = re.compile(r'[0-9a-f]{32}')


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

    def __init__(self, previews, embeddings, directory, evaluator_factory=None, finetune_fn=None):
        self.previews, self.embeddings = previews, embeddings
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.evaluator_factory = evaluator_factory
        self.finetune_fn = finetune_fn
        self.lock = threading.RLock()
        self.jobs, self.finetunes = {}, {}
        self._contexts, self._events = {}, {}
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

    def _context(self, embedding_id):
        context = self.embeddings.discovery_context(embedding_id)
        states = np.asarray(context['states'], dtype=np.float32)
        preview = context['preview']
        if (states.ndim != 2 or not len(states) or not states.shape[1] or
                len(states) != preview['input_count'] or not np.isfinite(states).all()):
            raise ValueError('Analysis must contain finite original input embeddings.')
        if not np.allclose(np.linalg.norm(states, axis=1), 1., rtol=1e-4, atol=1e-5):
            raise ValueError('Discovery requires normalized original model embeddings.')
        actions = preview['actions']
        ids = [action['id'] for action in actions]
        if not ids or any(type(action_id) is not int or action_id < 0 for action_id in ids) or len(set(ids)) != len(ids):
            raise ValueError('Analysis must contain distinct fixed action IDs.')
        context['states'] = states.copy()
        return context

    def _config(self, data, context):
        allowed = {'embedding_id', 'episodes', 'seed', 'label_map', 'reference_summary', 'summary_metric',
                   'failure_threshold', 'generation_steps', 'generation_size', 'max_new_tokens', 'device'}
        if not isinstance(data, dict) or set(data)-allowed:
            raise ValueError('Unknown discovery settings.')
        task = context['preview']['config']['task']
        config = dict(task=task, episodes=_integer(data.get('episodes', 200), 'Episodes', 1, 10000),
                      seed=_integer(data.get('seed', 42), 'Seed', 0, 2**32-1),
                      failure_threshold=_number(data.get('failure_threshold', .15), 'Failure threshold', 0, 1),
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
        if task == 'summarization' and not config['reference_summary'].strip():
            raise ValueError('Summary scoring requires an explicit reference summary.')
        labels = config['label_map']
        if not isinstance(labels, dict) or len(labels) > len(context['states']):
            raise ValueError('Labels must map original input indices to model labels.')
        for key, label in labels.items():
            if not isinstance(key, str) or not re.fullmatch(r'0|[1-9][0-9]*', key) or int(key) >= len(context['states']):
                raise ValueError('Label input index is outside this analysis.')
            if not ((type(label) is int and label >= 0) or (isinstance(label, str) and label.strip() and len(label) <= 512)):
                raise ValueError('Use explicit model class IDs or exact model label names.')
        if task != 'classification' and labels:
            raise ValueError('Class labels apply only to classification.')
        config['source_text'] = context['preview']['config'].get('text', '')
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
                   mean_reward=0., model_calls=0, unique_evaluations=0, criterion='',
                   config=settings, model=copy.deepcopy(context['model']), current=None,
                   records=[], history=[], baseline=[], artifacts={}, policy_statistics=None)
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

    def get_finetune(self, job_id):
        with self.lock:
            return _public(self.finetunes[_identity(job_id)])

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
        with self._inference_access(job_id):
            context = self._contexts[job_id]
            fresh = self._context(context['job']['id'])
            if fresh['model_key'] != context['model_key']:
                raise ValueError('The model changed while this job was waiting. Analyze again.')
            del fresh
            if not retain_models:
                self.embeddings.release_encoder()
            yield

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
                job.update({key: fields[key] for key in ('completed', 'total', 'loss', 'phase') if key in fields})
            elif 'phase' in fields:
                job['phase'] = fields['phase']

    def _input(self, context, index, action_id=None):
        preview = context['preview']
        if preview['config']['task'] == 'classification':
            from PIL import Image
            name = 'original' if action_id is None else f'action-{action_id}'
            with Image.open(io.BytesIO(self.previews.image(preview['id'], index, name))) as image:
                return image.convert('RGB').copy()
        if action_id is None:
            return preview['original']['text']
        return next(node['text'] for node in preview['nodes'] if node['id'] == action_id)

    def _predict(self, job_id, evaluator, value, index, artifact_name):
        self._check(job_id)
        with self.lock:
            job = self.jobs[job_id]
            job['model_calls'] += 1
            seed = (job['config']['seed']+index) % 2**32
        # Count actual attempts, including a failed operation. The same seed is
        # deliberately used for each input's baseline and all its variants.
        return _json(evaluator.predict(value, seed=seed, artifact_name=artifact_name))

    def _score(self, evaluator, output, baseline, target):
        score = _json(evaluator.score(output, baseline, target=target))
        score['reward'] = _number(score.get('reward'), 'Evaluator reward', -1, 1)
        score['severity'] = _number(score.get('severity'), 'Evaluator severity', 0, 1)
        if type(score.get('failure')) is not bool or not isinstance(score.get('criterion'), str) or not isinstance(score.get('metrics'), dict):
            raise ValueError('Evaluator returned an invalid measured score.')
        return score

    def _policy_statistics(self, job_id, policy, context, records):
        statistics = policy.action_statistics(context['states'])
        actions = context['preview']['actions']
        coverage = {}
        for record in records:
            coverage.setdefault(record['action_id'], set()).add(record['input_index'])
        for action in statistics['actions']:
            action['action_id'] = actions[action.pop('action_index')]['id']
            action['evaluated_inputs'] = len(coverage.get(action['action_id'], ()))
        self._update(job_id, policy_statistics=statistics)

    def _discover(self, job_id):
        evaluator, policy = None, None
        status, error = 'completed', None
        context = self._contexts[job_id]
        config = self.jobs[job_id]['config']
        directory = self.directory/job_id
        try:
            self._update(job_id, status='running', phase='waiting for model')
            with ExitStack() as cleanup:
                with self._model_access(job_id, retain_models=True):
                    self._update(job_id, phase='loading selected model and scorers')
                    factory = self.evaluator_factory or self.embeddings.evaluation_model
                    evaluator = factory(context['model'], config, directory,
                                        progress=lambda **fields: self._progress(job_id, **fields))
                    cleanup.callback(self._close_evaluator, evaluator)
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
                        with self._inference_access(job_id):
                            if index not in baseline_cache:
                                output = self._predict(job_id, evaluator, self._input(context, index), index, f'input-{index}-baseline')
                                baseline_cache[index] = output
                                score = self._score(evaluator, output, output, target)
                                baseline = dict(input_index=index, output=output, **score)
                                with self.lock:
                                    self.jobs[job_id]['baseline'].append(baseline)
                            self._check(job_id)
                            cached = pair in pair_cache
                            if not cached:
                                self._update(job_id, phase='evaluating action')
                                output = self._predict(job_id, evaluator, self._input(context, index, action_id), index,
                                                       f'input-{index}-action-{action_id}')
                                score = self._score(evaluator, output, baseline_cache[index], target)
                                pair_cache[pair] = (output, score)
                            if pair not in nodes and not self._events[job_id].is_set():
                                nodes[pair] = self.embeddings.discovery_node(
                                    context['job']['id'], index, action_id, evaluator=evaluator)
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
                        if episode % 10 == 0 or policy.updates == 1:
                            self._policy_statistics(job_id, policy, context, aggregates.values())
                        with self.lock:
                            job = self.jobs[job_id]
                            job['history'] = (job['history']+[entry])[-self.HISTORY_LIMIT:]
                            job.update(episode=episode, reward=score['reward'], mean_reward=total_reward/episode,
                                       loss=loss, records=list(aggregates.values()), unique_evaluations=len(pair_cache),
                                       criterion=score['criterion'], current=record, phase='learning from terminal reward')
                            if episode % 10 == 0:
                                self._persist(job_id)
                self._check(job_id)
        except InterruptedError:
            status = 'cancelled'
        except Exception as exc:
            status, error = 'error', str(exc)
        finally:
            if policy is not None:
                try:
                    self._policy_statistics(job_id, policy, context, self.jobs[job_id]['records'])
                    policy.save(directory/'policy.pt')
                except Exception as exc:
                    status, error = 'error', f'Could not save policy: {exc}'
            self._finish(job_id, status, error)

    def _persist(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id) or self.finetunes[job_id]
            directory = self.directory/job_id
            if job_id in self.jobs:
                job['artifacts'] = {'report': f'/discoveries/{job_id}/artifacts/report.json',
                                    'episodes': f'/discoveries/{job_id}/artifacts/episodes.jsonl',
                                    'policy': f'/discoveries/{job_id}/artifacts/policy.pt' if (directory/'policy.pt').exists() else None}
            report = _public(job, job_id if job_id in self.jobs else None)
            temporary = directory/'report.json.tmp'
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
            temporary.replace(directory/'report.json')

    def _finish(self, job_id, status, error):
        with self.lock:
            job = self.jobs.get(job_id) or self.finetunes[job_id]
            job.update(status=status, phase=status)
            if error:
                job['error'] = error
            try:
                self._persist(job_id)
            except Exception as exc:
                job.update(status='error', phase='error', error=f'Could not save job report: {exc}')
            finally:
                if self.active == job_id:
                    self.active = None
                self._contexts.pop(job_id, None)

    def artifact(self, job_id, filename):
        _identity(job_id)
        with self.lock:
            self.jobs[job_id]
        if not isinstance(filename, str) or not re.fullmatch(r'(?:report\.json|episodes\.jsonl|policy\.pt|[A-Za-z0-9_-]+\.png)', filename):
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
        allowed = {'discovery_id', 'examples', 'epochs', 'learning_rate', 'batch_size', 'seed'}
        if not isinstance(config, dict) or set(config)-allowed:
            raise ValueError('Unknown fine-tuning settings.')
        discovery_id = _identity(config.get('discovery_id'))
        discovery = self.get(discovery_id)
        if discovery['status'] not in TERMINAL:
            raise ValueError('Finish or cancel discovery before reviewing examples for fine-tuning.')
        examples = config.get('examples')
        if not isinstance(examples, list) or not 1 <= len(examples) <= 256:
            raise ValueError('Select between one and 256 visited examples with explicit desired targets.')
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
                path = Path(target).expanduser()
                if not path.is_absolute() or not path.is_file():
                    raise ValueError('Supply an existing absolute path to a desired training image.')
                target = str(path.resolve())
            seen.add((index, action))
            reviewed.append(dict(input_index=index, action_id=action, target=target))
        settings = dict(epochs=_integer(config.get('epochs', 1), 'Epochs', 1, 100),
                        learning_rate=_number(config.get('learning_rate', .00005), 'Learning rate', 1e-8, .1),
                        batch_size=_integer(config.get('batch_size', 1), 'Batch size', 1, 32),
                        seed=_integer(config.get('seed', discovery['config']['seed']), 'Seed', 0, 2**32-1),
                        generation_size=discovery['config']['generation_size'])
        context = self._context(discovery['embedding_id'])
        job_id = uuid.uuid4().hex
        job = dict(id=job_id, kind='finetune', discovery_id=discovery_id, task=discovery['task'],
                   status='queued', phase='queued', completed=0, total=len(reviewed)*settings['epochs'],
                   loss=None, checkpoint=None, examples=reviewed, settings=settings)
        return self._launch(job, context, self._finetune)

    def _finetune(self, job_id):
        status, error = 'completed', None
        job = self.finetunes[job_id]
        context = self._contexts[job_id]
        directory = self.directory/job_id
        try:
            self._update(job_id, status='running', phase='waiting for model')
            with self._model_access(job_id):
                self._update(job_id, phase='preparing reviewed training examples')
                examples = []
                for example in job['examples']:
                    self._check(job_id)
                    examples.append(dict(input=self._input(context, example['input_index'], example['action_id']),
                                         target=example['target']))
                train = self.finetune_fn
                if train is None:
                    from fine_tuning import fine_tune
                    train = fine_tune
                self._update(job_id, phase='training reviewed examples')
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
                self._update(job_id, result=result, checkpoint=str(checkpoint),
                             loss=loss,
                             completed=result.get('steps', job['completed']))
        except InterruptedError:
            status = 'cancelled'
        except Exception as exc:
            status, error = 'error', str(exc)
        finally:
            self._finish(job_id, status, error)
