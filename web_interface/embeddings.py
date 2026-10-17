"""Selected-model input features and fixed PCA coordinates for each analysis."""
import copy
import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from core import apply_image
from model_loading import configure_parallel_loading, prepare_analysis_model
from text_encoders import _encoder_device


def _selection(model, task='classification'):
    if task not in ('classification', 'summarization', 'generation'):
        raise ValueError('Choose an available workspace domain.')
    if not isinstance(model, dict) or model.get('source') not in ('local', 'huggingface'):
        raise ValueError('Select a local or Hugging Face model.')
    name = model.get('name')
    if not isinstance(name, str) or not name.strip() or len(name) > 1024:
        raise ValueError('Select a model before computing embeddings.')
    name = name.strip()
    if model['source'] == 'local':
        path = Path(name).expanduser().resolve()
        if not path.is_dir():
            raise ValueError('Embeddings require a local saved model directory, not a checkpoint file.')
        required = {'classification': ('config.json', 'preprocessor_config.json'),
                    'summarization': ('config.json',), 'generation': ('model_index.json',)}[task]
        for filename in required:
            if not (path/filename).is_file():
                raise ValueError(f'The local model directory must contain {filename}.')
        name = str(path)
    elif not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?', name) or '..' in name:
        raise ValueError('Use a Hugging Face model ID, for example google/vit-base-patch16-224.')
    selection = {'source': model['source'], 'name': name, 'task': task}
    revision = model.get('revision')
    if revision is not None:
        if not isinstance(revision, str) or not revision.strip() or len(revision) > 256:
            raise ValueError('The model revision must be a branch, tag, or commit name.')
        selection['revision'] = revision
    return selection


def _model_key(model):
    # A checkpoint's identity is independent of the execution device and timings.
    identity = {name: model[name] for name in ('source', 'name', 'revision') if model.get(name) is not None}
    identity['task'] = model.get('task', 'classification')
    if model['source'] == 'local':
        root = Path(model['name'])
        # A replaced checkpoint/processor at the same path must not reuse features.
        identity['files'] = [(path.relative_to(root).as_posix(), path.stat().st_size,
                              path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                             for path in sorted(root.rglob('*')) if path.is_file()]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class HuggingFaceEncoder:
    """Float32 AutoModel encoder. Imports model dependencies only on demand."""
    def __init__(self, model):
        selection = _selection(model)
        configure_parallel_loading()
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ValueError('Image embedding dependencies are missing. Stop the server and run bash run.sh from web_interface to install PyTorch and Transformers.') from exc
        version = re.match(r'^(\d+)\.(\d+)', torch.__version__)
        if not version or tuple(map(int, version.groups())) < (2, 6):
            raise ValueError('Image embeddings require PyTorch 2.6 or newer for safe weights-only model loading.')
        self.torch = torch
        self.device = _encoder_device(torch, model.get('device', 'cpu'))
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        options = {'trust_remote_code': False, 'local_files_only': selection['source'] == 'local'}
        if model.get('revision'):
            options['revision'] = model['revision']
        try:
            self.processor = AutoImageProcessor.from_pretrained(selection['name'], use_fast=False, **options)
            loaded, info = AutoModel.from_pretrained(selection['name'], weights_only=True,
                                                     output_loading_info=True, **options)
            missing = info.get('missing_keys', [])
            pooler_keys = [key for key in missing if 'pooler' in key.split('.')]
            if len(pooler_keys) != len(missing) or info.get('mismatched_keys') or info.get('error_msgs'):
                raise ValueError('The checkpoint is missing compatible pretrained base-model weights; newly initialized features are not supported.')
            self._skip_pooler = bool(pooler_keys)
            self.model = loaded.eval()
            self.set_device(self.device)
        except Exception as exc:
            raise ValueError(f'Could not load image model {selection["name"]}: {exc}') from exc
        self.revision = getattr(self.model.config, '_commit_hash', None)
        self.feature_method = None

    def set_device(self, device):
        device = _encoder_device(self.torch, device)
        self.model.to(device=device, dtype=self.torch.float32)
        self.device = device
        return self

    def share_classifier(self, evaluator):
        """Use the validated classifier's identical base on the same device."""
        if evaluator.device != self.device:
            return
        base = evaluator.model.base_model
        if base is self.model:
            return
        if type(base) is not type(self.model):
            return
        original = self.model.state_dict()
        replacement = base.state_dict()
        retained = {key for key in original if not (self._skip_pooler and 'pooler' in key.split('.'))}
        if retained != set(replacement) or any(original[key].shape != replacement[key].shape for key in retained):
            return
        self.model = base

    def encode(self, images):
        torch = self.torch
        try:
            inputs = self.processor(images=images, return_tensors='pt')
            if 'pixel_values' not in inputs:
                raise ValueError('The selected processor does not produce image pixel_values.')
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model(**inputs, return_dict=True)
                features = None if self._skip_pooler else getattr(output, 'pooler_output', None)
                if features is not None:
                    if not torch.is_tensor(features) or features.ndim < 2:
                        raise ValueError('The model returned invalid pooler_output image features.')
                    if features.ndim > 2:
                        features = features.mean(dim=tuple(range(2, features.ndim)))
                        method = 'pooler_output (spatial mean)'
                    else:
                        method = 'pooler_output'
                else:
                    features = getattr(output, 'last_hidden_state', None)
                    if not torch.is_tensor(features) or features.ndim not in (3, 4):
                        raise ValueError('Use a compatible vision AutoModel with pooler_output or token/spatial last_hidden_state features.')
                    if features.ndim == 3:
                        features = features.mean(dim=1)
                        method = 'last_hidden_state (token mean)'
                    else:
                        features = features.mean(dim=(2, 3))
                        method = 'last_hidden_state (spatial mean)'
            if self.feature_method is not None and self.feature_method != method:
                raise ValueError('The model changed feature extraction methods between image batches.')
            self.feature_method = method
            return features.detach().to(dtype=torch.float64, device='cpu').numpy()
        except Exception as exc:
            raise ValueError(f'Image feature extraction failed for the selected model: {exc}') from exc


def create_encoder(model):
    task = model.get('task', 'classification')
    if task == 'classification':
        return HuggingFaceEncoder(model)
    from text_encoders import SummarizationEncoder, DiffusionPromptEncoder
    if task == 'summarization':
        return SummarizationEncoder(model)
    if task == 'generation':
        return DiffusionPromptEncoder(model)
    raise ValueError('Unsupported embedding workspace.')


def _normalize(raw, count, dimension=None):
    import numpy as np
    if np.iscomplexobj(raw):
        raise ValueError('Model features must contain real numbers.')
    try:
        features = np.asarray(raw, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError('The model returned nonnumeric features.') from exc
    if features.ndim != 2 or features.shape[0] != count or not features.shape[1]:
        raise ValueError('The model must return one nonempty feature vector per input.')
    if dimension is not None and features.shape[1] != dimension:
        raise ValueError('The selected model changed embedding dimensions between batches.')
    if not np.isfinite(features).all():
        raise ValueError('The model returned nonfinite features.')
    scale = np.max(np.abs(features), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError('The model returned a zero feature vector; cosine similarity is undefined.')
    scaled = features / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def _pca(features):
    import numpy as np
    mean = np.mean(features, axis=0)
    centered = features - mean
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    # An absolute floor suppresses rounding residue from repeated identical rows.
    tolerance = np.finfo(np.float64).eps * max(features.shape) * max(float(singular[0]), 1.0)
    rank = int(np.count_nonzero(singular > tolerance))
    components = np.zeros((3, features.shape[1]), dtype=np.float64)
    ratios = np.zeros(3, dtype=np.float64)
    retained = min(3, rank)
    if retained:
        components[:retained] = vectors[:retained]
        for vector in components[:retained]:
            # Resolve PCA's sign ambiguity consistently, including numeric ties.
            pivot = int(np.argmax(np.round(np.abs(vector), decimals=12)))
            if vector[pivot] < 0:
                vector *= -1
        ratios[:retained] = singular[:retained] ** 2 / np.sum(singular ** 2)
    projection = {'method': 'PCA on L2-normalized original image features', 'rank': rank,
                  'variance': float(min(1.0, ratios.sum())), 'components': ratios.tolist()}
    return mean, components, projection


class EmbeddingManager:
    """Serialize inference, retain session identities, and lazily expand selected inputs."""
    BATCH_SIZE = 8
    # Exact SVD needs several arrays; fail clearly instead of silently sampling data.
    MAX_PCA_BYTES = 256 * 1024 * 1024

    def __init__(self, previews, directory, encoder_factory=None, model_preparer=None):
        self.previews = previews
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.encoder_factory = encoder_factory or create_encoder
        # Injected factories may represent offline fixtures or external backends.
        # They control their own preparation unless an explicit preparer is given.
        self.model_preparer = model_preparer if model_preparer is not None else (
            prepare_analysis_model if encoder_factory is None else None)
        self.lock = threading.RLock()
        # Discovery and fine-tuning share this resource boundary with analysis.
        # The job-state lock remains free while model work is running.
        self.model_lock = threading.RLock()
        self.jobs = {}
        self.expansions = {}
        self._preview_jobs = {}
        self._bases = {}
        self._job_bases = {}
        self._expansion_cache = {}
        self._queue = deque()
        self._queued_at = {}
        self._worker = None
        self._active_expansion = None
        self._encoder = None
        self._encoder_key = None
        self._encoder_identity = None
        self._evaluator = None
        self._evaluator_key = None
        self._evaluator_identity = None
        self._discovery_nodes = {}
        self._revisions = {}

    def get(self, job_id):
        with self.lock:
            return copy.deepcopy(self.jobs[job_id])

    def discovery_context(self, job_id):
        """Return immutable analysis identity and independent full-dimensional states."""
        import numpy as np
        with self.lock:
            job = copy.deepcopy(self.jobs[job_id])
            if job['status'] != 'completed':
                raise ValueError('Finish Analyze before finding failures.')
            base = self._job_bases[job_id]
            model = copy.deepcopy(job['model'])
            if model['source'] == 'local' and _model_key(model) != base['model_key']:
                raise ValueError('The local model changed after analysis. Analyze the input again.')
            revision = self._revisions.get(base['model_key'])
            if revision:
                model['revision'] = revision
            if job['task'] == 'classification':
                states = np.array(np.load(base['path'], mmap_mode='r'), copy=True)
            else:
                states = base['states'].copy()
            return dict(job=job, preview=self.previews.get(job['preview_id']),
                        model=model, states=states, model_key=base['model_key'])

    def release_encoder(self):
        """Invalidate all retained inference weights before training or a model switch."""
        with self.model_lock:
            self._encoder = None
            self._encoder_key = None
            self._encoder_identity = None
            self._release_evaluator()

    def _release_evaluator(self):
        previous = self._evaluator
        self._evaluator = None
        self._evaluator_key = None
        self._evaluator_identity = None
        if previous is not None:
            previous.close()

    def _model_identity(self, model):
        selected = _selection(model, model.get('task', 'classification'))
        revision = self._revisions.get(_model_key(selected))
        if revision and selected['source'] != 'local':
            selected['revision'] = revision
        return _model_key(selected)

    def evaluation_model(self, model, config, directory, progress=None):
        """Reuse one model/scorer set; each caller receives independent run settings.

        Callers hold model_lock throughout inference. The returned evaluator may
        be closed normally: the manager retains the shared, read-only weights.
        Training uses uncached adapters after release_encoder().
        """
        from discovery_models import _dependencies, _device, create_evaluator
        with self.model_lock:
            selection = _selection(model, config.get('task', model.get('task', 'classification')))
            requested_key = _model_key(selection)
            revision = self._revisions.get(requested_key)
            if revision and selection['source'] != 'local':
                selection['revision'] = revision
            identity = self._model_identity(selection)
            torch, _ = _dependencies()
            resources = {'device': _device(torch, config),
                         'summary_metric': config.get('summary_metric', 'reference')}
            for name in ('_nli_model', '_clip_model'):
                if config.get(name) is not None:
                    scorer = _selection(config[name], 'summarization' if name == '_nli_model' else 'classification')
                    resources[name] = _model_key(scorer)
            resource_key = json.dumps(resources, sort_keys=True)
            key = (identity, resource_key)
            shared = self._evaluator if self._evaluator_key == key else None
            if shared is None:
                self._release_evaluator()
                if self._encoder is not None and self._encoder_identity != identity:
                    self._encoder = None
                    self._encoder_key = None
                    self._encoder_identity = None
            # Sharing must never move an Analyze encoder to another device:
            # later expansions need the original numerical path and PCA basis.
            encoder = self._encoder if (self._encoder_identity == identity and
                getattr(self._encoder, 'device', 'cpu') == resources['device']) else None
            evaluator = create_evaluator(selection, config, directory, progress, encoder=encoder, shared=shared)
            self._revisions[requested_key] = getattr(evaluator, 'revision', None)
            self._revisions[_model_key(selection)] = getattr(evaluator, 'revision', None)
            identity = self._model_identity(selection)
            if encoder is not None and isinstance(encoder, HuggingFaceEncoder):
                encoder.share_classifier(evaluator)
            self._evaluator = copy.copy(evaluator)
            self._evaluator._shared_resources = False
            self._evaluator_key = (identity, resource_key)
            self._evaluator_identity = identity
            evaluator._shared_resources = True
            return evaluator

    def discovery_node(self, job_id, input_index, action_id, evaluator=None):
        """Project only a measured action, without queuing a full graph expansion."""
        import numpy as np
        with self.model_lock:
            with self.lock:
                job = copy.deepcopy(self.jobs[job_id])
                if job['status'] != 'completed':
                    raise ValueError('Finish Analyze before projecting discoveries.')
                if type(input_index) is not int or not 0 <= input_index < len(job['points']):
                    raise ValueError('Select an available input.')
                base = self._job_bases[job_id]
                if job['model']['source'] == 'local' and _model_key(job['model']) != base['model_key']:
                    raise ValueError('The local model changed after analysis. Analyze the input again.')
                cached = self._expansion_cache.get((job_id, input_index))
                if cached is not None:
                    node = next((node for node in cached['nodes'] if node['id'] == action_id), None)
                    if node is None:
                        raise ValueError('Select an available action.')
                    return copy.deepcopy(node)
                cache_key = (job_id, input_index, action_id)
                if cache_key in self._discovery_nodes:
                    return copy.deepcopy(self._discovery_nodes[cache_key])
            preview = self.previews.get(job['preview_id'], input_index)
            entry = next((node for node in preview['nodes'] if node['id'] == action_id), None)
            if entry is None:
                raise ValueError('Select an available action.')
            if job['task'] != 'classification':
                raise ValueError('The analyzed text variants are unavailable. Analyze again.')
            encoder = self._load_encoder(job['model'], base['model_key'], device=job['device'])
            image = apply_image(self._original_image(job['preview_id'], input_index), entry['action'])
            value = _normalize(self._encode_batch(encoder, [image]), 1, base['feature_dim'])[0]
            features = np.load(base['path'], mmap_mode='r')
            node = dict(id=entry['id'], action=entry['action'], image=entry['image'],
                        position=((value-base['mean']) @ base['components'].T).tolist(),
                        similarity=float(np.clip(value @ features[input_index], -1, 1)),
                        neighbors=self._neighbors(features, value, job['points']))
            with self.lock:
                self._discovery_nodes[cache_key] = copy.deepcopy(node)
            return node

    def get_expansion(self, job_id, expansion_id):
        with self.lock:
            self.jobs[job_id]
            result = self.expansions[expansion_id]
            if result['job_id'] != job_id:
                raise KeyError(expansion_id)
            return copy.deepcopy(result)

    def start(self, preview_id, device='auto'):
        if not isinstance(device, str) or device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('Choose an analysis device: auto, cpu, or cuda.')
        actual_device, device_name = 'cpu', 'CPU'
        if device != 'cpu':
            try:
                import torch
            except ImportError as exc:
                raise ValueError('Analysis dependencies are missing. Run bash setup.sh from web_interface.') from exc
            actual_device = _encoder_device(torch, device)
            if actual_device == 'cuda':
                device_name = torch.cuda.get_device_name()
        preview = self.previews.get(preview_id)
        task = preview['config']['task']
        if preview['status'] != 'completed':
            raise ValueError('Generate the input preview before computing embeddings.')
        model = _selection(preview['config']['model'], task)
        dataset = self.previews.datasets.ready(preview['config']['dataset_id']) if task == 'classification' else None
        model_key = _model_key(model)
        key = (dataset['id'] if dataset else preview_id, model_key, actual_device)
        preview_key = (preview_id, model_key, actual_device)
        with self.lock:
            existing = self._preview_jobs.get(preview_key)
            if existing and self.jobs[existing]['status'] != 'error':
                return self.get(existing)
            job_id = uuid.uuid4().hex
            total = dataset['count'] if dataset else 1+len(preview['nodes'])
            self.jobs[job_id] = dict(id=job_id, task=task, preview_id=preview_id,
                                     dataset_id=dataset['id'] if dataset else '', model=model,
                                     requested_device=device, device=actual_device, device_name=device_name,
                                     timings_seconds=self._empty_timings(), queue_wait_seconds=0.0,
                                     model_reused=False, features_reused=False,
                                     status='queued', completed=0, total=total, phase='queued', points=[])
            self._preview_jobs[preview_key] = job_id
            self._enqueue(('overview', job_id, key))
            return self.get(job_id)

    def expand(self, job_id, input_index):
        with self.lock:
            job = self.jobs[job_id]
            if job['status'] != 'completed':
                raise ValueError('Compute input embeddings before expanding an input.')
            if type(input_index) is not int or not 0 <= input_index < len(job['points']):
                raise ValueError('Input index is outside this embedding dataset.')
            if job['model']['source'] == 'local' and _model_key(job['model']) != self._job_bases[job_id]['model_key']:
                raise ValueError('The local model changed after analysis. Analyze the input again.')
            if self._active_expansion:
                active = self.expansions[self._active_expansion]
                if active['status'] in ('queued', 'running'):
                    if active['job_id'] == job_id and active['input_index'] == input_index:
                        return self.get_expansion(job_id, active['id'])
                    active.update(status='cancelled', phase='superseded', error='A newer input selection superseded this expansion.')
            preview = self.previews.get(job['preview_id'], input_index)
            expansion_id = uuid.uuid4().hex
            result = dict(id=expansion_id, job_id=job_id, preview_id=job['preview_id'], input_index=input_index,
                          model=copy.deepcopy(job['model']), status='queued', completed=0,
                          requested_device=job['requested_device'], device=job['device'], device_name=job['device_name'],
                          timings_seconds=self._empty_timings(), queue_wait_seconds=0.0,
                          model_reused=False, features_reused=False,
                          total=len(preview['nodes']), phase='queued', original=copy.deepcopy(job['points'][input_index]),
                          nodes=[], neighbors=[])
            self.expansions[expansion_id] = result
            self._active_expansion = expansion_id
            cached = self._expansion_cache.get((job_id, input_index))
            if cached:
                result.update(copy.deepcopy(cached), status='completed', phase='completed', features_reused=True)
            else:
                self._enqueue(('expand', expansion_id, None))
            return self.get_expansion(job_id, expansion_id)

    def _enqueue(self, work):
        # Caller holds the lock. One worker exits when its queue is drained.
        self._queued_at[work[1]] = time.perf_counter()
        self._queue.append(work)
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    def _run(self):
        while True:
            with self.lock:
                if not self._queue:
                    self._worker = None
                    return
                kind, identifier, key = self._queue.popleft()
                record = self.jobs[identifier] if kind == 'overview' else self.expansions[identifier]
                queued_at = self._queued_at.pop(identifier)
                if record['status'] == 'cancelled':
                    continue
                record.update(status='running', phase='waiting for model resources')
            with self.model_lock:
                # Queue wait includes time blocked by discovery or fine-tuning,
                # and must not inflate weight-loading or active-work timings.
                started = time.perf_counter()
                with self.lock:
                    record['queue_wait_seconds'] = started - queued_at
                    if record['status'] == 'cancelled':
                        continue
                error = None
                try:
                    if kind == 'overview':
                        if record['task'] == 'classification':
                            self._overview(identifier, key)
                        else:
                            self._text_overview(identifier, key)
                    else:
                        self._expand(identifier)
                except Exception as exc:
                    error = str(exc)
                finally:
                    with self.lock:
                        record['timings_seconds']['total'] = time.perf_counter() - started
                        if record['status'] != 'cancelled':
                            if error is not None:
                                record.update(status='error', phase='error', error=error)
                            else:
                                record.update(status='completed', phase='completed')

    @staticmethod
    def _empty_timings():
        return dict.fromkeys(('preparation', 'model_loading', 'embedding', 'pca', 'total'), 0.0)

    @contextmanager
    def _stage(self, record, name, phase):
        started = time.perf_counter()
        if record is not None:
            with self.lock:
                record['phase'] = phase
        try:
            yield
        finally:
            if record is not None:
                with self.lock:
                    record['timings_seconds'][name] += time.perf_counter() - started

    @staticmethod
    def _synchronize(encoder):
        torch = getattr(encoder, 'torch', None)
        if getattr(encoder, 'device', 'cpu') == 'cuda' and torch is not None:
            torch.cuda.synchronize()

    @staticmethod
    def _is_cuda_oom(error, torch):
        seen = set()
        while error is not None and id(error) not in seen:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                return True
            seen.add(id(error))
            error = error.__cause__ or error.__context__
        return False

    def _encode_batch(self, encoder, inputs):
        try:
            return encoder.encode(inputs)
        except Exception as exc:
            torch = getattr(encoder, 'torch', None)
            if (getattr(encoder, 'device', 'cpu') != 'cuda' or torch is None or
                    not self._is_cuda_oom(exc, torch)):
                raise
            if len(inputs) == 1:
                raise ValueError('CUDA ran out of memory for one input. Choose CPU, shorten text inputs, '
                                 'or use a smaller model before analyzing again.') from exc
        # Drop the failed call's traceback and tensors before retrying. Splits
        # preserve input order and stop at one input, without moving devices.
        import numpy as np
        torch.cuda.empty_cache()
        middle = len(inputs) // 2
        return np.concatenate((self._encode_batch(encoder, inputs[:middle]),
                               self._encode_batch(encoder, inputs[middle:])), axis=0)

    def _load_encoder(self, model, key, device=None, record=None):
        device = device or model.get('device', 'cpu')
        encoder_key = (key, device)
        if self._encoder_key != encoder_key:
            # Release the previous model before allocating its replacement.
            self._encoder = None
            self._encoder_key = None
            self._encoder_identity = None
            identity = self._model_identity(model)
            if self._evaluator_identity != identity:
                self._release_evaluator()
            options = dict(model)
            options['device'] = device
            if self._revisions.get(key):
                options['revision'] = self._revisions[key]
            retained = self._evaluator
            same_device = retained is not None and retained.device == device
            reused = same_device and hasattr(retained, 'encoder')
            if reused:
                encoder = retained.encoder
            else:
                metadata = {}
                if self.model_preparer is not None:
                    with self._stage(record, 'preparation', 'preparing model files'):
                        options, metadata = self.model_preparer(options)
                    if record is not None and 'files_reused' in metadata:
                        with self.lock:
                            record['files_reused'] = metadata['files_reused']
                with self._stage(record, 'model_loading', 'loading model weights'):
                    try:
                        encoder = self.encoder_factory({**options, 'device': device})
                        if metadata.get('revision'):
                            encoder.revision = metadata['revision']
                        if same_device and isinstance(encoder, HuggingFaceEncoder):
                            encoder.share_classifier(retained)
                        self._synchronize(encoder)
                    except Exception as exc:
                        if device == 'cuda':
                            import torch
                            if self._is_cuda_oom(exc, torch):
                                raise ValueError('CUDA ran out of memory while loading model weights. '
                                                 'Choose CPU or use a smaller model before analyzing again.') from exc
                        raise
            self._encoder = encoder
            self._encoder_key = encoder_key
            self._revisions[key] = getattr(encoder, 'revision', None)
            self._encoder_identity = self._model_identity(model)
        else:
            reused = True
        if record is not None:
            with self.lock:
                record['model_reused'] = reused
        return self._encoder

    def _check_pca_size(self, count, dimension):
        estimated_bytes = 8 * (3 * count * dimension + 3 * min(count, dimension) ** 2)
        if estimated_bytes > self.MAX_PCA_BYTES:
            raise ValueError('This input/model exceeds the 256 MB PCA working-array limit. Use fewer inputs/actions or a model with fewer feature dimensions.')

    def _text_overview(self, job_id, key):
        """One source needs its variants in the PCA fit to have meaningful axes."""
        import numpy as np
        job = self.get(job_id)
        if job['model']['source'] == 'local' and _model_key(job['model']) != key[1]:
            raise ValueError('The local model changed while embeddings were queued. Analyze the input again.')
        preview = self.previews.get(job['preview_id'])
        entries = [preview['original'], *preview['nodes']]
        record = self.jobs[job_id]
        encoder = self._load_encoder(job['model'], key[1], device=job['device'], record=record)
        features = None
        with self._stage(record, 'embedding', 'embedding input and action variants'):
            for offset in range(0, len(entries), self.BATCH_SIZE):
                texts = [entry['text'] for entry in entries[offset:offset+self.BATCH_SIZE]]
                values = _normalize(self._encode_batch(encoder, texts), len(texts), None if features is None else features.shape[1])
                if features is None:
                    self._check_pca_size(len(entries), values.shape[1])
                    features = np.empty((len(entries), values.shape[1]), dtype=np.float64)
                features[offset:offset+len(texts)] = values
                with self.lock:
                    record['completed'] = offset+len(texts)
        with self._stage(record, 'pca', 'fitting PCA'):
            mean, components, projection = _pca(features)
            projection.update(method='PCA on L2-normalized input and action variant features', fit_count=len(entries))
            positions = (features-mean) @ components.T
        point = dict(index=0, name=preview['input_name'], text=entries[0]['text'], position=positions[0].tolist())
        nodes = []
        for entry, value, position in zip(preview['nodes'], features[1:], positions[1:]):
            node = copy.deepcopy(entry)
            node.update(position=position.tolist(), similarity=float(np.clip(value @ features[0], -1, 1)),
                        neighbors=self._neighbors(features[:1], value, [point]))
            nodes.append(node)
        with self.lock:
            self._job_bases[job_id] = {'model_key': key[1], 'states': features[:1].copy()}
            self._expansion_cache[(job_id, 0)] = dict(nodes=nodes, completed=len(nodes), neighbors=[])
            self.jobs[job_id].update(completed=len(entries), points=[point],
                                     projection=projection, feature_dim=features.shape[1],
                                     feature_method=getattr(encoder, 'feature_method', 'model input features'))

    def _overview(self, job_id, key):
        import numpy as np
        job = self.get(job_id)
        if job['model']['source'] == 'local' and _model_key(job['model']) != key[1]:
            raise ValueError('The local model changed while embeddings were queued. Reload the preview.')
        base = self._bases.get(key)
        record = self.jobs[job_id]
        if base is None:
            encoder = self._load_encoder(job['model'], key[1], device=job['device'], record=record)
            path = self.directory/(uuid.uuid4().hex+'.npy')
            features = None
            digests = []
            try:
                with self._stage(record, 'embedding', 'embedding originals'):
                    for offset in range(0, job['total'], self.BATCH_SIZE):
                        images = [self._original_image(job['preview_id'], index)
                                  for index in range(offset, min(offset+self.BATCH_SIZE, job['total']))]
                        digests.extend(hashlib.sha256(image.tobytes()).hexdigest() for image in images)
                        values = _normalize(self._encode_batch(encoder, images), len(images), None if features is None else features.shape[1])
                        if features is None:
                            n, d = job['total'], values.shape[1]
                            self._check_pca_size(n, d)
                            features = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(n, d))
                        features[offset:offset+len(images)] = values
                        with self.lock:
                            record['completed'] = offset+len(images)
                    features.flush()
                with self._stage(record, 'pca', 'fitting PCA'):
                    mean, components, projection = _pca(features)
                    positions = (features-mean) @ components.T
                base = dict(path=path, mean=mean, components=components, projection=projection,
                            feature_dim=features.shape[1], feature_method=getattr(encoder, 'feature_method', 'image features'),
                            model_key=key[1], image_digests=digests, positions=positions)
                del features
                self._bases[key] = base
            except Exception:
                if features is not None:
                    del features
                path.unlink(missing_ok=True)
                raise
        else:
            with self.lock:
                record.update(features_reused=True, model_reused=self._encoder_key == (key[1], job['device']))
            for index, digest in enumerate(base['image_digests']):
                image = self._original_image(job['preview_id'], index)
                if hashlib.sha256(image.tobytes()).hexdigest() != digest:
                    raise ValueError('The dataset image pixels changed. Reload the dataset to refresh image embeddings.')
        points = []
        for index, position in enumerate(base['positions']):
            preview = self.previews.get(job['preview_id'], index)
            points.append(dict(index=index, name=preview['input_name'], image=preview['original']['image'], position=position.tolist()))
        with self.lock:
            self._job_bases[job_id] = base
            self.jobs[job_id].update(completed=job['total'], points=points,
                                     projection=copy.deepcopy(base['projection']), feature_dim=base['feature_dim'],
                                     feature_method=base['feature_method'])

    def _original_image(self, preview_id, input_index):
        from PIL import Image
        # PreviewManager freezes DatasetManager.image's preprocessing as PNG.
        # Reading that same PNG keeps model inputs and visible images identical.
        with Image.open(io.BytesIO(self.previews.image(preview_id, input_index, 'original'))) as image:
            return image.convert('RGB')

    @staticmethod
    def _neighbors(features, vector, points, exclude=None):
        import numpy as np
        similarities = np.clip(features @ vector, -1, 1)
        order = np.argsort(-similarities, kind='stable')
        nearest = order[:5] if exclude is None else order[order != exclude][:5]
        return [dict(index=int(index), name=points[index]['name'], similarity=float(similarities[index]))
                for index in nearest]

    def _expand(self, expansion_id):
        import numpy as np
        with self.lock:
            expansion = copy.deepcopy(self.expansions[expansion_id])
            job = copy.deepcopy(self.jobs[expansion['job_id']])
        base = self._job_bases[job['id']]
        if job['model']['source'] == 'local' and _model_key(job['model']) != base['model_key']:
            raise ValueError('The local model changed after the original embeddings. Reload the preview.')
        record = self.expansions[expansion_id]
        encoder = self._load_encoder(job['model'], base['model_key'], device=job['device'], record=record)
        features = np.load(base['path'], mmap_mode='r')
        index = expansion['input_index']
        original = self._original_image(job['preview_id'], index)
        preview = self.previews.get(job['preview_id'], index)
        nodes = []
        for offset in range(0, expansion['total'], self.BATCH_SIZE):
            with self.lock:
                if self.expansions[expansion_id]['status'] == 'cancelled':
                    return
            entries = preview['nodes'][offset:offset+self.BATCH_SIZE]
            with self._stage(record, 'embedding', 'embedding transformations'):
                images = [apply_image(original, node['action']) for node in entries]
                values = _normalize(self._encode_batch(encoder, images), len(images), base['feature_dim'])
            with self._stage(record, 'pca', 'projecting transformations'):
                positions = (values-base['mean']) @ base['components'].T
            for entry, value, position in zip(entries, values, positions):
                nodes.append(dict(id=entry['id'], action=entry['action'], image=entry['image'], position=position.tolist(),
                                  similarity=float(np.clip(value @ features[index], -1, 1)),
                                  neighbors=self._neighbors(features, value, job['points'])))
            with self.lock:
                if self.expansions[expansion_id]['status'] == 'cancelled':
                    return
                self.expansions[expansion_id].update(completed=len(nodes), phase='embedding transformations')
        details = dict(nodes=nodes, completed=len(nodes), neighbors=self._neighbors(features, features[index], job['points'], exclude=index))
        with self.lock:
            if self.expansions[expansion_id]['status'] == 'cancelled':
                return
            self._expansion_cache[(job['id'], index)] = copy.deepcopy(details)
            self.expansions[expansion_id].update(details)
