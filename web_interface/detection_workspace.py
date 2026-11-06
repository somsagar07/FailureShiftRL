"""Object-detection analysis and terminal DQN discovery over image variations."""
import copy
import io
import itertools
import math
import random
import re
import threading
import uuid
from pathlib import Path

from core import apply_image


class DetectionManager:
    def __init__(self, datasets, directory, *, detector_factory=None):
        self.datasets = datasets
        self.directory = Path(directory)
        self.detector_factory = detector_factory
        self.jobs = {}
        self.contexts = {}
        self.lock = threading.RLock()
        self.active = None

    def get(self, job_id):
        with self.lock:
            return copy.deepcopy(self.jobs[job_id])

    def _update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def _start(self, job, work):
        with self.lock:
            if self.active:
                raise RuntimeError('An object detection job is already running.')
            self.active = job['id']
            self.jobs[job['id']] = job
            threading.Thread(target=self._run, args=(job['id'], work), daemon=True).start()
            return copy.deepcopy(job)

    def _run(self, job_id, work):
        try:
            self._update(job_id, status='running')
            work(job_id)
            self._update(job_id, status='completed', phase='completed')
        except Exception as exc:
            self._update(job_id, status='failed', phase='failed', error=str(exc))
        finally:
            with self.lock:
                if self.active == job_id:
                    self.active = None

    def _start_auxiliary(self, job, work):
        with self.lock:
            self.jobs[job['id']] = job
            threading.Thread(target=self._run, args=(job['id'], work), daemon=True).start()
            return copy.deepcopy(job)

    @staticmethod
    def _actions(selected):
        if (not isinstance(selected, list) or not selected or
                any(not isinstance(x, str) or x not in ('darkness', 'saturation') for x in selected) or
                len(set(selected)) != len(selected)):
            raise ValueError('Select darkness, saturation, or both.')
        darkness = [.1, .2, .3, .4, .5] if 'darkness' in selected else [0]
        saturation = [.5, .55, .6, .65, .7] if 'saturation' in selected else [1]
        return [dict(id=i, darkness=d, saturation=s) for i, (d, s) in enumerate(itertools.product(darkness, saturation))]

    def analyze(self, data):
        if not isinstance(data, dict):
            raise ValueError('Choose a model and image dataset.')
        dataset_id = data.get('dataset_id')
        dataset = self.datasets.ready(dataset_id)
        if dataset.get('task', 'classification') != 'classification' or not dataset['count']:
            raise ValueError('Load a nonempty image dataset.')
        model = data.get('model')
        if (not isinstance(model, dict) or model.get('source') not in ('huggingface', 'local') or
                not isinstance(model.get('name'), str) or not model['name'].strip() or len(model['name']) > 1024):
            raise ValueError('Choose a Hugging Face detector or local detector directory.')
        name = model['name'].strip()
        if model['source'] == 'local' and not Path(name).expanduser().is_dir():
            raise ValueError('The detector directory does not exist.')
        if model['source'] == 'huggingface' and not re.fullmatch(r'[\w.-]+/[\w.-]+', name):
            raise ValueError('Enter a Hugging Face model ID in owner/model form.')
        device = data.get('device', 'auto')
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('Choose automatic, CPU, or CUDA.')
        job = dict(id=uuid.uuid4().hex, kind='analysis', dataset_id=dataset_id,
                   model=dict(source=model['source'], name=name), device=device, status='queued',
                   phase='queued', completed=0, total=dataset['count'], points=[])
        return self._start(job, self._analyze)

    def _analyze(self, job_id):
        import numpy as np
        from embeddings import _pca
        from object_detection import Detector
        job = self.get(job_id)
        self._update(job_id, phase='loading detector')
        detector = (self.detector_factory or Detector)(job['model']['name'], device=job['device'])
        states = []
        names = []
        for i in range(job['total']):
            image = self.datasets.image(job['dataset_id'], i)
            feature = np.asarray(detector.embeddings(image), dtype=np.float32).reshape(-1)
            if not feature.size or not np.isfinite(feature).all() or not np.linalg.norm(feature):
                raise ValueError('Detector returned invalid image features.')
            states.append(feature / np.linalg.norm(feature))
            names.append(self.datasets.page(job['dataset_id'], i, 1)['items'][0]['name'])
            self._update(job_id, phase='embedding originals', completed=i+1)
        states = np.stack(states)
        mean, components, _ = _pca(states)
        positions = (states - mean) @ components.T
        points = [dict(index=i, name=name, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
                  for i, (name, pos) in enumerate(zip(names, positions))]
        with self.lock:
            self.contexts[job_id] = dict(detector=detector, states=states, mean=mean, components=components,
                                         baselines={}, model_lock=threading.RLock())
        self._update(job_id, points=points, feature_dim=states.shape[1],
                     feature_method='pooled detector backbone features')

    def discover(self, data):
        if not isinstance(data, dict):
            raise ValueError('Choose an analyzed dataset.')
        analysis = self.get(data.get('analysis_id'))
        if analysis['kind'] != 'analysis' or analysis['status'] != 'completed':
            raise ValueError('Analyze images before discovery.')
        episodes, seed = data.get('episodes', 200), data.get('seed', 42)
        if type(episodes) is not int or not 1 <= episodes <= 10000 or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Choose 1–10000 episodes and a nonnegative 32-bit seed.')
        actions = self._actions(data.get('actions'))
        job = dict(id=uuid.uuid4().hex, kind='discovery', analysis_id=analysis['id'],
                   dataset_id=analysis['dataset_id'], status='queued', phase='queued', completed=0,
                   total=episodes, seed=seed, points=analysis['points'], actions=actions, records=[], policy_updates=0)
        return self._start(job, self._discover)

    def expand(self, data):
        if not isinstance(data, dict):
            raise ValueError('Choose an analyzed image.')
        analysis = self.get(data.get('analysis_id'))
        index = data.get('input_index')
        if analysis['kind'] != 'analysis' or analysis['status'] != 'completed':
            raise ValueError('Analyze images first.')
        if type(index) is not int or not 0 <= index < len(analysis['points']):
            raise ValueError('Choose an available image.')
        actions = self._actions(data.get('actions'))
        job = dict(id=uuid.uuid4().hex, kind='expansion', analysis_id=analysis['id'],
                   dataset_id=analysis['dataset_id'], input_index=index, actions=actions,
                   points=analysis['points'], status='queued', phase='queued', completed=0,
                   total=len(actions), nodes=[])
        return self._start_auxiliary(job, self._expand)

    def _expand(self, job_id):
        import numpy as np
        job = self.get(job_id)
        context = self.contexts[job['analysis_id']]
        image = self.datasets.image(job['dataset_id'], job['input_index'])
        nodes = []
        for action in job['actions']:
            with context['model_lock']:
                feature = np.asarray(context['detector'].embeddings(apply_image(image, action)), dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(feature)
            if feature.shape != context['mean'].shape or not np.isfinite(feature).all() or not norm:
                raise ValueError('Detector returned invalid action features.')
            position = ((feature / norm - context['mean']) @ context['components'].T).tolist()
            nodes.append(dict(id=action['id'], action=action, position=position))
            self._update(job_id, phase='embedding perturbations', completed=len(nodes), nodes=list(nodes))

    def inspect(self, data):
        if not isinstance(data, dict):
            raise ValueError('Choose an image.')
        expansion = self.get(data.get('expansion_id'))
        action_id = data.get('action_id')
        if expansion['kind'] != 'expansion' or expansion['status'] not in ('queued', 'running', 'completed'):
            raise ValueError('Expand an analyzed image first.')
        if action_id is not None and (type(action_id) is not int or not 0 <= action_id < len(expansion['actions'])):
            raise ValueError('Choose an available action.')
        job = dict(id=uuid.uuid4().hex, kind='inspection', analysis_id=expansion['analysis_id'],
                   dataset_id=expansion['dataset_id'], input_index=expansion['input_index'],
                   points=expansion['points'], actions=expansion['actions'], action_id=action_id,
                   status='queued', phase='queued', completed=0, total=1)
        return self._start_auxiliary(job, self._inspect)

    def _inspect(self, job_id):
        from object_detection import compare_predictions
        job = self.get(job_id)
        context = self.contexts[job['analysis_id']]
        index = job['input_index']
        image = self.datasets.image(job['dataset_id'], index)
        with context['model_lock']:
            if index not in context['baselines']:
                context['baselines'][index] = context['detector'].predict(image)
            baseline = context['baselines'][index]
            action_id = job['action_id']
            output = context['detector'].predict(apply_image(image, job['actions'][action_id])) if action_id is not None else None
        self._update(job_id, baseline=baseline, output=output,
                     metrics=compare_predictions(baseline, output) if output is not None else None, completed=1)

    def _discover(self, job_id):
        import numpy as np
        from object_detection import compare_predictions
        from rl_policy import TerminalDQN
        job = self.get(job_id)
        context = self.contexts[job['analysis_id']]
        states, detector = context['states'], context['detector']
        policy = TerminalDQN(states.shape[1], len(job['actions']), seed=job['seed'])
        rng = random.Random(job['seed'])
        pairs = {}
        for episode in range(job['total']):
            index = rng.randrange(len(states))
            epsilon = 1 - .4 * episode / max(1, job['total']-1)
            action_id = policy.choose(states[index], epsilon=epsilon)
            key = (index, action_id)
            self._update(job_id, phase='evaluating detector', current=dict(input_index=index, action_id=action_id))
            with context['model_lock']:
                if key not in pairs:
                    image = self.datasets.image(job['dataset_id'], index)
                    if index not in context['baselines']:
                        context['baselines'][index] = detector.predict(image)
                    output = detector.predict(apply_image(image, job['actions'][action_id]))
                    baseline = context['baselines'][index]
                    metrics = compare_predictions(baseline, output)
                    severity = float(metrics['severity'])
                    if not math.isfinite(severity) or not 0 <= severity <= 1:
                        raise ValueError('Detector comparison returned an invalid severity.')
                    pairs[key] = dict(input_index=index, action_id=action_id, severity=severity,
                                      reward=severity, baseline=baseline, output=output, metrics=metrics, visits=0)
            record = pairs[key]
            policy.observe(states[index], action_id, record['reward'])
            record['visits'] += 1
            self._update(job_id, completed=episode+1, records=list(copy.deepcopy(list(pairs.values()))),
                         policy_updates=policy.updates)
        if policy.updates:
            with policy.torch.inference_mode():
                estimates = policy.network(policy.torch.tensor(states, dtype=policy.torch.float32)).detach().cpu().numpy()
            if np.isfinite(estimates).all():
                self._update(job_id, predictions=estimates.tolist(), prediction_kind='DQN predicted reward; uncalibrated')

    def image(self, job_id, index, variant):
        from object_detection import render_detections
        job = self.get(job_id)
        if type(index) is not int or not 0 <= index < len(job['points']):
            raise ValueError('Choose an available image.')
        image = self.datasets.image(job['dataset_id'], index)
        prediction = None
        if job['kind'] == 'inspection':
            if index != job['input_index'] or job['status'] != 'completed':
                raise ValueError('Inspection is not ready.')
            if variant == 'original':
                prediction = job['baseline']
            elif variant == f'action-{job["action_id"]}' and job['action_id'] is not None:
                image = apply_image(image, job['actions'][job['action_id']])
                prediction = job['output']
            else:
                raise ValueError('Choose the inspected action.')
        elif job['kind'] == 'discovery':
            records = [r for r in job['records'] if r['input_index'] == index]
            if variant == 'original':
                prediction = records[0]['baseline'] if records else None
            else:
                match = re.fullmatch(r'action-(\d+)', variant)
                if not match or int(match[1]) >= len(job['actions']):
                    raise ValueError('Choose an available action.')
                action_id = int(match[1])
                image = apply_image(image, job['actions'][action_id])
                record = next((r for r in records if r['action_id'] == action_id), None)
                prediction = record['output'] if record else None
        elif variant != 'original':
            raise ValueError('Analyze has no action output yet.')
        if prediction is not None:
            image = render_detections(image, prediction)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
