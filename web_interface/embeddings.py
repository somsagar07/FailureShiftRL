"""Selected-model input features and fixed PCA coordinates for each analysis."""
import copy
import hashlib
import io
import json
import os
import re
import threading
import uuid
from collections import deque
from pathlib import Path

from core import apply_image


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
    return {'source': model['source'], 'name': name, 'task': task}


def _model_key(model):
    identity = dict(model)
    if model['source'] == 'local':
        root = Path(model['name'])
        # A replaced checkpoint/processor at the same path must not reuse features.
        identity['files'] = [(path.relative_to(root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
                             for path in sorted(root.rglob('*')) if path.is_file()]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class HuggingFaceEncoder:
    """CPU-only AutoModel encoder. Imports model dependencies only on demand."""
    def __init__(self, model):
        selection = _selection(model)
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ValueError('Image embedding dependencies are missing. Stop the server and run bash run.sh from web_interface to install PyTorch and Transformers.') from exc
        version = re.match(r'^(\d+)\.(\d+)', torch.__version__)
        if not version or tuple(map(int, version.groups())) < (2, 6):
            raise ValueError('Image embeddings require PyTorch 2.6 or newer for safe weights-only model loading.')
        self.torch = torch
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
            self.model = loaded.to('cpu').eval()
        except Exception as exc:
            raise ValueError(f'Could not load image model {selection["name"]}: {exc}') from exc
        self.revision = getattr(self.model.config, '_commit_hash', None)
        self.feature_method = None

    def encode(self, images):
        torch = self.torch
        try:
            inputs = self.processor(images=images, return_tensors='pt')
            if 'pixel_values' not in inputs:
                raise ValueError('The selected processor does not produce image pixel_values.')
            inputs = {key: value.to('cpu') for key, value in inputs.items()}
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

    def __init__(self, previews, directory, encoder_factory=None):
        self.previews = previews
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.encoder_factory = encoder_factory or create_encoder
        self.lock = threading.RLock()
        self.jobs = {}
        self.expansions = {}
        self._preview_jobs = {}
        self._bases = {}
        self._job_bases = {}
        self._expansion_cache = {}
        self._queue = deque()
        self._worker = None
        self._active_expansion = None
        self._encoder = None
        self._encoder_key = None
        self._revisions = {}

    def get(self, job_id):
        with self.lock:
            return copy.deepcopy(self.jobs[job_id])

    def get_expansion(self, job_id, expansion_id):
        with self.lock:
            self.jobs[job_id]
            result = self.expansions[expansion_id]
            if result['job_id'] != job_id:
                raise KeyError(expansion_id)
            return copy.deepcopy(result)

    def start(self, preview_id):
        preview = self.previews.get(preview_id)
        task = preview['config']['task']
        if preview['status'] != 'completed':
            raise ValueError('Generate the input preview before computing embeddings.')
        model = _selection(preview['config']['model'], task)
        dataset = self.previews.datasets.ready(preview['config']['dataset_id']) if task == 'classification' else None
        model_key = _model_key(model)
        key = (dataset['id'] if dataset else preview_id, model_key)
        with self.lock:
            existing = self._preview_jobs.get((preview_id, model_key))
            if existing and self.jobs[existing]['status'] != 'error':
                return self.get(existing)
            job_id = uuid.uuid4().hex
            total = dataset['count'] if dataset else 1+len(preview['nodes'])
            self.jobs[job_id] = dict(id=job_id, task=task, preview_id=preview_id,
                                     dataset_id=dataset['id'] if dataset else '', model=model,
                                     status='queued', completed=0, total=total, phase='queued', points=[])
            self._preview_jobs[(preview_id, model_key)] = job_id
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
                          total=len(preview['nodes']), phase='queued', original=copy.deepcopy(job['points'][input_index]),
                          nodes=[], neighbors=[])
            self.expansions[expansion_id] = result
            self._active_expansion = expansion_id
            cached = self._expansion_cache.get((job_id, input_index))
            if cached:
                result.update(copy.deepcopy(cached), status='completed', phase='completed')
            else:
                self._enqueue(('expand', expansion_id, None))
            return self.get_expansion(job_id, expansion_id)

    def _enqueue(self, work):
        # Caller holds the lock. One worker exits when its queue is drained.
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
                if record['status'] == 'cancelled':
                    continue
                record.update(status='running', phase='loading model')
            try:
                if kind == 'overview':
                    if record['task'] == 'classification':
                        self._overview(identifier, key)
                    else:
                        self._text_overview(identifier, key)
                else:
                    self._expand(identifier)
            except Exception as exc:
                with self.lock:
                    if record['status'] != 'cancelled':
                        record.update(status='error', phase='error', error=str(exc))

    def _load_encoder(self, model, key):
        if self._encoder_key != key:
            # Release the previous model before allocating its replacement.
            self._encoder = None
            self._encoder_key = None
            options = dict(model)
            if self._revisions.get(key):
                options['revision'] = self._revisions[key]
            encoder = self.encoder_factory(options)
            self._encoder = encoder
            self._encoder_key = key
            self._revisions[key] = getattr(encoder, 'revision', None)
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
        encoder = self._load_encoder(job['model'], key[1])
        features = None
        for offset in range(0, len(entries), self.BATCH_SIZE):
            texts = [entry['text'] for entry in entries[offset:offset+self.BATCH_SIZE]]
            values = _normalize(encoder.encode(texts), len(texts), None if features is None else features.shape[1])
            if features is None:
                self._check_pca_size(len(entries), values.shape[1])
                features = np.empty((len(entries), values.shape[1]), dtype=np.float64)
            features[offset:offset+len(texts)] = values
            with self.lock:
                self.jobs[job_id].update(completed=offset+len(texts), phase='embedding input and action variants')
        with self.lock:
            self.jobs[job_id]['phase'] = 'fitting PCA'
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
            self._job_bases[job_id] = {'model_key': key[1]}
            self._expansion_cache[(job_id, 0)] = dict(nodes=nodes, completed=len(nodes), neighbors=[])
            self.jobs[job_id].update(status='completed', phase='completed', completed=len(entries), points=[point],
                                     projection=projection, feature_dim=features.shape[1],
                                     feature_method=getattr(encoder, 'feature_method', 'model input features'))

    def _overview(self, job_id, key):
        import numpy as np
        job = self.get(job_id)
        if job['model']['source'] == 'local' and _model_key(job['model']) != key[1]:
            raise ValueError('The local model changed while embeddings were queued. Reload the preview.')
        base = self._bases.get(key)
        if base is None:
            encoder = self._load_encoder(job['model'], key[1])
            path = self.directory/(uuid.uuid4().hex+'.npy')
            features = None
            digests = []
            try:
                for offset in range(0, job['total'], self.BATCH_SIZE):
                    images = [self._original_image(job['preview_id'], index)
                              for index in range(offset, min(offset+self.BATCH_SIZE, job['total']))]
                    digests.extend(hashlib.sha256(image.tobytes()).hexdigest() for image in images)
                    values = _normalize(encoder.encode(images), len(images), None if features is None else features.shape[1])
                    if features is None:
                        n, d = job['total'], values.shape[1]
                        self._check_pca_size(n, d)
                        features = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(n, d))
                    features[offset:offset+len(images)] = values
                    with self.lock:
                        self.jobs[job_id].update(completed=offset+len(images), phase='embedding originals')
                features.flush()
                with self.lock:
                    self.jobs[job_id]['phase'] = 'fitting PCA'
                mean, components, projection = _pca(features)
                base = dict(path=path, mean=mean, components=components, projection=projection,
                            feature_dim=features.shape[1], feature_method=getattr(encoder, 'feature_method', 'image features'),
                            model_key=key[1], image_digests=digests)
                del features
                self._bases[key] = base
            except Exception:
                if features is not None:
                    del features
                path.unlink(missing_ok=True)
                raise
        else:
            for index, digest in enumerate(base['image_digests']):
                image = self._original_image(job['preview_id'], index)
                if hashlib.sha256(image.tobytes()).hexdigest() != digest:
                    raise ValueError('The dataset image pixels changed. Reload the dataset to refresh image embeddings.')
        features = np.load(base['path'], mmap_mode='r')
        positions = (features - base['mean']) @ base['components'].T
        points = []
        for index, position in enumerate(positions):
            preview = self.previews.get(job['preview_id'], index)
            points.append(dict(index=index, name=preview['input_name'], image=preview['original']['image'], position=position.tolist()))
        with self.lock:
            self._job_bases[job_id] = base
            self.jobs[job_id].update(status='completed', phase='completed', completed=job['total'], points=points,
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
        encoder = self._load_encoder(job['model'], base['model_key'])
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
            images = [apply_image(original, node['action']) for node in entries]
            values = _normalize(encoder.encode(images), len(images), base['feature_dim'])
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
            self.expansions[expansion_id].update(details, status='completed', phase='completed')
