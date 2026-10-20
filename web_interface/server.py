#!/usr/bin/env python3
"""Local model analysis, live RL failure discovery, and reviewed fine-tuning."""
import argparse
import copy
import errno
import hashlib
import io
import json
import mimetypes
import random
import re
import threading
import uuid
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote, parse_qs

from core import (default_action_space, normalize_action_space, actions_for, apply_image, apply_text,
                  apply_prompt, LEGACY_GENERATION_SPACE)
from models import ModelManager
from data_sources import DatasetManager, MAX_UPLOAD_BYTES, hf_dataset_options, benchmark_dataset_options
from embeddings import EmbeddingManager, HuggingFaceEncoder, create_encoder
from discovery import DiscoveryManager
from model_loading import prepare_analysis_model
from working_storage import BoundedCache, budget_bytes

ROOT = Path(__file__).resolve().parent
DEFAULT_TEXT = ('A university research team tested a new water filter in three villages. '
                'The six-month trial found that the filter removed 95 percent of contaminants. '
                'The team will conduct a larger trial next year before recommending widespread use.')
DEFAULT_PROMPT = 'A unique scientist working on a project in a corporate office.'
LEGACY_PROMPT = 'A {description} {profession} working on a project in a {setting}.'
TASKS = ('classification', 'summarization', 'generation')


def validate_config(data):
    if not isinstance(data, dict):
        raise ValueError('Request must be a JSON object.')
    task = data.get('task', 'classification')
    if not isinstance(task, str) or task not in TASKS:
        raise ValueError('Choose an available workspace domain.')
    dataset_id = data.get('dataset_id', '')
    if task in ('summarization', 'generation') and (not isinstance(dataset_id, str) or
            dataset_id and not re.fullmatch(r'[0-9a-f]{32}', dataset_id)):
        raise ValueError(f'Load a {task} dataset or enter input text.')
    text_dataset = task in ('summarization', 'generation') and bool(dataset_id)
    legacy_generation = (task == 'generation' and isinstance(data.get('action_space'), dict) and
                         set(data['action_space']) == set(LEGACY_GENERATION_SPACE))
    prompt = LEGACY_PROMPT if legacy_generation else DEFAULT_PROMPT
    text = data.get('text', prompt if task == 'generation' else DEFAULT_TEXT)
    if text_dataset:
        # Dataset rows are read lazily; a selected source has no text-field limit.
        text = ''
    elif not isinstance(text, str) or len(text) > 20000 or (task != 'classification' and not text.strip()):
        raise ValueError('Enter input text of at most 20,000 characters.')
    space = normalize_action_space(task, data.get('action_space'))
    defaults = LEGACY_GENERATION_SPACE if legacy_generation else default_action_space(task)
    if task == 'generation' and set(space) == {'styles'}:
        defaults = {'styles': defaults['style']}
    if any(value not in defaults[key] for key, values in space.items() for value in values):
        raise ValueError('Choose from the fixed action values. New values are not enabled in this version.')
    if task == 'classification' and (not isinstance(dataset_id, str) or not re.fullmatch(r'[0-9a-f]{32}', dataset_id)):
        raise ValueError('Load a folder or Hugging Face dataset first.')
    model = data.get('model', {'source': 'local', 'name': ''})
    if not isinstance(model, dict) or model.get('source') not in ('local', 'huggingface') or not isinstance(model.get('name'), str) or len(model['name']) > 1024:
        raise ValueError('Invalid model selection.')
    return dict(task=task, dataset_id=dataset_id, text=text, action_space=space,
                model={'source': model['source'], 'name': model['name']})


class PreviewManager:
    def __init__(self, directory, *, cache_bytes=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.datasets = DatasetManager(self.directory.parent/'datasets')
        # Same-input consumers share a snapshot; unrelated images can prepare
        # in parallel. A bounded set of locks avoids one lock per dataset row.
        self.image_locks = [threading.RLock() for _ in range(64)]
        self.cache = BoundedCache(budget_bytes('FAILURESHIFT_PREVIEW_RAM_MB', 32)
                                  if cache_bytes is None else cache_bytes)
        self._identities = {}
        self._pins = {}

    def get(self, job_id, input_index=0):
        with self.lock:
            job = copy.deepcopy(self.jobs[job_id])
        if type(input_index) is not int or not 0 <= input_index < job['input_count']:
            raise ValueError('Input index is outside this preview.')
        if job['config']['task'] == 'classification':
            base = f'/previews/{job_id}/input-{input_index}'
            job['original'] = dict(image=base+'/original.png')
            job['nodes'] = [dict(id=action['id'], action=action, image=base+f'/action-{action["id"]}.png') for action in job['actions']]
            job['input_name'] = self.datasets.page(job['config']['dataset_id'], input_index, 1)['items'][0]['name']
        elif job['config'].get('dataset_id'):
            source = self.datasets.text(job['config']['dataset_id'], input_index)
            text = source['text']
            original = dict(text=text)
            for name in ('source_split', 'source_index'):
                if name in source:
                    original[name] = source[name]
            job.update(original=original, input_name=source['name'])
            if job['config']['task'] == 'summarization':
                job['reference_summary'] = source.get('summary')
            else:
                job['checks'] = copy.deepcopy(source.get('checks', []))
                original['checks'] = copy.deepcopy(job['checks'])
            job['nodes'] = []
            for action in job['actions']:
                transformed = (apply_text(text, action, random.Random(42 + input_index*1000003 + action['id']))
                               if job['config']['task'] == 'summarization' else apply_prompt(text, action))
                job['nodes'].append(dict(id=action['id'], action=action, text=transformed, unchanged=transformed == text))
        job['input_index'] = input_index
        return job

    def _image_job(self, job_id, input_index):
        with self.lock:
            job = self.jobs[job_id]
        if type(input_index) is not int or not 0 <= input_index < job['input_count']:
            raise ValueError('Image index is outside this preview.')
        if job['config']['task'] != 'classification' or job['status'] != 'completed':
            raise ValueError('This image preview is not ready.')
        return job

    def _original_pixels(self, job, input_index):
        dataset_id = job['config']['dataset_id']
        key = (job['id'], input_index)
        identity = self.datasets.image_identity(dataset_id, input_index)
        frozen = self._identities.get(key)
        changed = 'The dataset image source changed. Reload the dataset and Analyze again.'
        if frozen is not None and identity != frozen[0]:
            raise ValueError(changed)
        image = self.cache.get((*key, 'pixels'))
        if image is None:
            image = self.datasets.image(dataset_id, input_index)
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if (self.datasets.image_identity(dataset_id, input_index) != identity or
                    frozen is not None and digest != frozen[1]):
                raise ValueError(changed)
            self._identities[key] = (identity, digest)
            self.cache.put((*key, 'pixels'), image, image.width*image.height*len(image.getbands()))
        # Callers may mutate Pillow objects. Never expose the cache's own image.
        return image.copy()

    def original_image(self, job_id, input_index):
        """Read verified source pixels; retain only a bounded decoded RAM cache."""
        job = self._image_job(job_id, input_index)
        with self.pin_preview(job_id), self.image_locks[input_index % len(self.image_locks)]:
            return self._original_pixels(job, input_index)

    def image(self, job_id, input_index, name):
        job = self._image_job(job_id, input_index)
        actions = {f'action-{action["id"]}': action for action in job['actions']}
        if name != 'original' and name not in actions:
            raise ValueError('Unknown action in this preview.')
        # Render only images the user views; the full input × action set is indexed.
        with self.pin_preview(job_id), self.image_locks[input_index % len(self.image_locks)]:
            original = self._original_pixels(job, input_index)
            key = (job_id, input_index, name)
            payload = self.cache.get(key)
            if payload is None:
                image = original if name == 'original' else apply_image(original, actions[name])
                output = io.BytesIO()
                image.save(output, format='PNG', compress_level=1)
                payload = output.getvalue()
                self.cache.put(key, payload, len(payload))
            return payload

    @contextmanager
    def pin_preview(self, job_id):
        """Prevent disposable-cache release while a consumer is using a preview."""
        with self.lock:
            self.jobs[job_id]
            self._pins[job_id] = self._pins.get(job_id, 0) + 1
        try:
            yield
        finally:
            with self.lock:
                self._pins[job_id] -= 1
                if not self._pins[job_id]:
                    del self._pins[job_id]

    def release_preview(self, job_id):
        """Release owned image RAM only; keep manifests, identities and sources."""
        with self.lock:
            job = self.jobs[job_id]
            if self._pins.get(job_id) or job['status'] in ('queued', 'running'):
                return dict(status='busy', released_memory_bytes=0, released_disk_bytes=0,
                            reason='Preview images are in use.')
            released = self.cache.release(job_id)
            return dict(status='released', released_memory_bytes=released, released_disk_bytes=0)

    def update(self, job_id, **fields):
        with self.lock:
            self.jobs[job_id].update(fields)

    def start(self, config):
        dataset = self.datasets.ready(config['dataset_id']) if config.get('dataset_id') else None
        if dataset and dataset.get('task', 'classification') != config['task']:
            raise ValueError(f'This dataset belongs to {dataset.get("task", "classification")}; load a {config["task"]} dataset.')
        input_count = dataset['count'] if dataset else 1
        actions = actions_for(config['task'], config['action_space'])
        with self.lock:
            if self.active:
                raise RuntimeError('Input combinations are already being generated.')
            job_id = uuid.uuid4().hex
            (self.directory/job_id).mkdir()
            job = dict(id=job_id, config=config, actions=actions, input_count=input_count, action_count=len(actions),
                       status='queued', completed=0, total=input_count*len(actions),
                       original=None, nodes=[])
            if dataset:
                job['dataset'] = copy.deepcopy(dataset)
            self.jobs[job_id] = job
            self.active = job_id
            threading.Thread(target=self.generate, args=(job_id, config), daemon=True).start()
            return copy.deepcopy(job)

    def generate(self, job_id, config):
        try:
            self.update(job_id, status='running')
            folder = self.directory/job_id
            task = config['task']
            if task in ('summarization', 'generation') and config.get('dataset_id'):
                # Retain only the action manifest. Per-row full texts and edits
                # are built when the user or discovery requests that row.
                with self.lock:
                    manifest = copy.deepcopy(self.jobs[job_id])
                manifest.update(status='completed', completed=manifest['total'])
                (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
                self.update(job_id, status='completed', completed=manifest['total'])
                return
            if task == 'classification':
                manifest = self.get(job_id)
                manifest.update(status='completed', completed=manifest['total'])
                (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
                self.update(job_id, status='completed', completed=manifest['total'])
                return
            original_text = config['text']
            if task == 'generation' and set(config['action_space']) == set(LEGACY_GENERATION_SPACE):
                # Keep a concrete reference prompt even when its values are removed
                # from the action space. Placeholders are never model inputs.
                for key, values in LEGACY_GENERATION_SPACE.items():
                    original_text = original_text.replace('{'+key+'}', values[0])
            self.update(job_id, original=dict(text=original_text),
                        input_name='Baseline prompt' if task == 'generation' else 'Source text')
            nodes = []
            for action in actions_for(task, config['action_space']):
                node = dict(id=action['id'], action=action)
                if task == 'summarization':
                    node['text'] = apply_text(config['text'], action, random.Random(42+action['id']))
                    node['unchanged'] = node['text'] == config['text']
                else:
                    node['text'] = apply_prompt(config['text'], action)
                nodes.append(node)
                self.update(job_id, completed=len(nodes))
            self.update(job_id, nodes=nodes)
            # Save before exposing completion, so the response and artifacts agree.
            manifest = self.get(job_id)
            manifest['status'] = 'completed'
            (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
            self.update(job_id, status='completed')
        except Exception as exc:
            self.update(job_id, status='error', error=str(exc))
        finally:
            with self.lock:
                self.active = None


def create_server(port=8000, directory=None, *, evaluator_factory=None, finetune_fn=None,
                  model_preparer=prepare_analysis_model):
    previews = PreviewManager(directory or ROOT/'runs'/'previews')
    models = ModelManager()
    image_encoder, other_encoder = HuggingFaceEncoder, create_encoder
    def encoder_factory(model):
        return image_encoder(model) if model.get('task') == 'classification' else other_encoder(model)
    embeddings = EmbeddingManager(previews, previews.directory.parent/'embeddings',
                                  encoder_factory=encoder_factory, model_preparer=model_preparer)
    discoveries = DiscoveryManager(previews, embeddings, previews.directory.parent/'discoveries',
                                   evaluator_factory=evaluator_factory, finetune_fn=finetune_fn)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, kind='application/json'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, allow_nan=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def local_request(self):
            allowed = (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')
            if self.headers.get('Host') not in allowed:
                self.send(403, {'error': 'Only local browser requests are accepted.'})
                return False
            origin = self.headers.get('Origin')
            if origin and origin not in tuple('http://'+host for host in allowed):
                self.send(403, {'error': 'Cross-origin requests are not allowed.'})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = unquote(urlsplit(self.path).path)
            query = parse_qs(urlsplit(self.path).query)
            try:
                if path == '/api/catalog':
                    return self.send(200, dict(tasks={task: default_action_space(task) for task in TASKS},
                                               text=DEFAULT_TEXT, prompt=DEFAULT_PROMPT))
                if path == '/api/discoveries':
                    return self.send(200, discoveries.list())
                if path == '/api/datasets/options':
                    query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                    if len(query.get('name', [])) != 1 or len(query.get('config', [])) > 1:
                        raise ValueError('Provide one Hugging Face dataset ID and at most one configuration.')
                    name, config = query['name'][0], query.get('config', [''])[0]
                    options = benchmark_dataset_options if name == 'geneval2' else hf_dataset_options
                    return self.send(200, options(name, config))
                match = re.fullmatch(r'/api/discoveries/([0-9a-f]{32})/adaptation', path)
                if match:
                    return self.send(200, discoveries.adaptation(match[1]))
                match = re.fullmatch(r'/api/discoveries/([0-9a-f]{32})/predictions', path)
                if match:
                    if set(query) != {'input'} or len(query['input']) != 1:
                        raise ValueError('Choose one original input for predictions.')
                    return self.send(200, discoveries.predictions(match[1], int(query['input'][0])))
                match = re.fullmatch(r'/api/(discoveries|finetunes)/([0-9a-f]{32})', path)
                if match:
                    value = discoveries.get(match[2]) if match[1] == 'discoveries' else discoveries.get_finetune(match[2])
                    return self.send(200, value)
                match = re.fullmatch(r'/discoveries/([0-9a-f]{32})/artifacts/([A-Za-z0-9_.-]+)', path)
                if match:
                    file = discoveries.artifact(match[1], match[2])
                    return self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
                match = re.fullmatch(r'/api/datasets/([0-9a-f]{32})(/items)?', path)
                if match:
                    if match[2]:
                        return self.send(200, previews.datasets.page(match[1], int(query.get('offset', ['0'])[0]), int(query.get('limit', ['100'])[0])))
                    return self.send(200, previews.datasets.get(match[1]))
                match = re.fullmatch(r'/api/datasets/([0-9a-f]{32})/text/(\d+)', path)
                if match:
                    return self.send(200, previews.datasets.text(match[1], int(match[2])))
                match = re.fullmatch(r'/datasets/([0-9a-f]{32})/images/(\d+)\.png', path)
                if match:
                    with previews.datasets.image(match[1], int(match[2])) as image:
                        buffer = io.BytesIO()
                        image.save(buffer, format='PNG')
                        return self.send(200, buffer.getvalue(), 'image/png')
                match = re.fullmatch(r'/api/(previews|models)/([0-9a-f]{32})', path)
                if match:
                    value = previews.get(match[2], int(query.get('input', ['0'])[0])) if match[1] == 'previews' else models.get(match[2])
                    return self.send(200, value)
                match = re.fullmatch(r'/api/embeddings/([0-9a-f]{32})(?:/expansions/([0-9a-f]{32}))?', path)
                if match:
                    value = embeddings.get_expansion(match[1], match[2]) if match[2] else embeddings.get(match[1])
                    return self.send(200, value)
                match = re.fullmatch(r'/previews/([0-9a-f]{32})/input-(\d+)/(original|action-\d+)\.png', path)
                if match:
                    return self.send(200, previews.image(match[1], int(match[2]), match[3]), 'image/png')
                static = {'/': 'index.html', '/app.js': 'app.js', '/graph.js': 'graph.js',
                          '/embedding-view.js':'embedding-view.js', '/embedding-controller.js':'embedding-controller.js',
                          '/adaptation-controller.js':'adaptation-controller.js',
                          '/discovery-controller.js':'discovery-controller.js', '/style.css': 'style.css'}
                if path in static:
                    file = ROOT/'static'/static[path]
                    return self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or 'text/plain')
                self.send(404, {'error': 'Not found.'})
            except (KeyError, FileNotFoundError):
                self.send(404, {'error': 'Job, preview, or file not found. Analyze the input again if the server restarted.'})
            except (ValueError, ImportError) as exc:
                self.send(400, {'error': str(exc)})

        def do_POST(self):
            if not self.local_request():
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= MAX_UPLOAD_BYTES:
                    return self.send(413, {'error': 'Request exceeds 20 MB.'})
                upload = re.fullmatch(r'/api/datasets/([0-9a-f]{32})/files/(\d+)', self.path)
                if upload:
                    previews.datasets.upload(upload[1], int(upload[2]), self.rfile.read(length))
                    return self.send(200, {'uploaded': True})
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    return self.send(415, {'error': 'Send application/json.'})
                data = json.loads(self.rfile.read(length))
                if self.path in ('/api/discoveries', '/api/finetunes'):
                    if not isinstance(data, dict):
                        raise ValueError('Request must be a JSON object.')
                    job = discoveries.start(data) if self.path == '/api/discoveries' else discoveries.start_finetune(data)
                    return self.send(202, job)
                cancellation = re.fullmatch(r'/api/(discoveries|finetunes)/([0-9a-f]{32})/cancel', self.path)
                if cancellation:
                    if not isinstance(data, dict):
                        raise ValueError('Request must be a JSON object.')
                    job = discoveries.cancel(cancellation[2]) if cancellation[1] == 'discoveries' else discoveries.cancel_finetune(cancellation[2])
                    return self.send(200, job)
                if self.path == '/api/embeddings':
                    if not isinstance(data, dict) or not isinstance(data.get('preview_id'), str) or not re.fullmatch(r'[0-9a-f]{32}', data['preview_id']):
                        raise ValueError('Generate an image preview before loading its embeddings.')
                    device = data.get('device', 'auto')
                    if not isinstance(device, str) or device not in ('auto', 'cpu', 'cuda'):
                        raise ValueError('Choose Automatic, CPU, or CUDA for Analyze.')
                    return self.send(202, embeddings.start(data['preview_id'], device=device))
                expansion = re.fullmatch(r'/api/embeddings/([0-9a-f]{32})/expand', self.path)
                if expansion:
                    if not isinstance(data, dict) or type(data.get('input_index')) is not int:
                        raise ValueError('Select an image from this dataset.')
                    return self.send(202, embeddings.expand(expansion[1], data['input_index']))
                if self.path == '/api/previews':
                    return self.send(202, previews.start(validate_config(data)))
                if self.path == '/api/models':
                    return self.send(202, models.select(data))
                if self.path == '/api/datasets':
                    return self.send(202, previews.datasets.start(data))
                finish = re.fullmatch(r'/api/datasets/([0-9a-f]{32})/finish', self.path)
                if finish:
                    return self.send(200, previews.datasets.finish_upload(finish[1]))
                self.send(404, {'error': 'Not found.'})
            except (ValueError, TypeError) as exc:
                self.send(400, {'error': str(exc)})
            except RuntimeError as exc:
                self.send(409, {'error': str(exc)})
            except KeyError:
                self.send(404, {'error': 'Job or dataset not found. Analyze the input again if the server restarted.'})

        def log_message(self, fmt, *args):
            if len(args) > 1 and str(args[1]) not in ('200', '202'):
                super().log_message(fmt, *args)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.previews = previews
    server.embeddings = embeddings
    server.discoveries = discoveries
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    try:
        server = create_server(args.port)
    except OSError as exc:
        help_text = 'Try --port 8001.' if exc.errno == errno.EADDRINUSE else 'Run this command in your own terminal; this environment may restrict listening sockets.'
        parser.exit(1, f'Could not start local server: {exc}. {help_text}\n')
    address = f'http://127.0.0.1:{server.server_port}'
    print(f'FailureShift: {address}\nPress Ctrl+C to stop.', flush=True)
    if args.open:
        threading.Timer(.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
