#!/usr/bin/env python3
"""Local model-embedding and input-combination viewer. No failure search or RL."""
import argparse
import copy
import errno
import json
import mimetypes
import random
import re
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote, parse_qs

from core import default_action_space, normalize_action_space, actions_for, apply_image, apply_text
from models import ModelManager
from data_sources import DatasetManager, MAX_UPLOAD_BYTES
from embeddings import EmbeddingManager, HuggingFaceEncoder, create_encoder

ROOT = Path(__file__).resolve().parent
DEFAULT_TEXT = ('A university research team tested a new water filter in three villages. '
                'The six-month trial found that the filter removed 95 percent of contaminants. '
                'The team will conduct a larger trial next year before recommending widespread use.')
DEFAULT_PROMPT = 'A {description} {profession} working on a project in a {setting}.'
TASKS = ('classification', 'summarization', 'generation')


def validate_config(data):
    if not isinstance(data, dict):
        raise ValueError('Request must be a JSON object.')
    task = data.get('task', 'classification')
    if not isinstance(task, str) or task not in TASKS:
        raise ValueError('Choose an available workspace domain.')
    text = data.get('text', DEFAULT_PROMPT if task == 'generation' else DEFAULT_TEXT)
    if not isinstance(text, str) or len(text) > 20000 or (task != 'classification' and not text.strip()):
        raise ValueError('Enter input text of at most 20,000 characters.')
    if task == 'generation' and any(text.count('{'+key+'}') != 1 for key in ('description', 'profession', 'setting')):
        raise ValueError('Keep exactly one {description}, {profession} and {setting} in the prompt.')
    space = normalize_action_space(task, data.get('action_space'))
    defaults = default_action_space(task)
    if any(value not in defaults[key] for key, values in space.items() for value in values):
        raise ValueError('Choose from the fixed action values. New values are not enabled in this version.')
    dataset_id = data.get('dataset_id', '')
    if task == 'classification' and (not isinstance(dataset_id, str) or not re.fullmatch(r'[0-9a-f]{32}', dataset_id)):
        raise ValueError('Load a folder or Hugging Face dataset first.')
    model = data.get('model', {'source': 'local', 'name': ''})
    if not isinstance(model, dict) or model.get('source') not in ('local', 'huggingface') or not isinstance(model.get('name'), str) or len(model['name']) > 1024:
        raise ValueError('Invalid model selection.')
    return dict(task=task, dataset_id=dataset_id if task == 'classification' else '', text=text, action_space=space,
                model={'source': model['source'], 'name': model['name']})


class PreviewManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.datasets = DatasetManager(self.directory.parent/'datasets')
        self.image_lock = threading.Lock()

    def get(self, job_id, input_index=0):
        with self.lock:
            job = copy.deepcopy(self.jobs[job_id])
        if type(input_index) is not int or not 0 <= input_index < job['input_count']:
            raise ValueError('Image index is outside this preview.')
        if job['config']['task'] == 'classification':
            base = f'/previews/{job_id}/input-{input_index}'
            job['original'] = dict(image=base+'/original.png')
            job['nodes'] = [dict(id=action['id'], action=action, image=base+f'/action-{action["id"]}.png') for action in job['actions']]
            job['input_name'] = self.datasets.page(job['config']['dataset_id'], input_index, 1)['items'][0]['name']
        job['input_index'] = input_index
        return job

    def image(self, job_id, input_index, name):
        job = self.get(job_id, input_index)
        if job['config']['task'] != 'classification' or job['status'] != 'completed':
            raise ValueError('This image preview is not ready.')
        actions = {f'action-{action["id"]}': action for action in job['actions']}
        if name != 'original' and name not in actions:
            raise ValueError('Unknown action in this preview.')
        from PIL import Image
        # Render only images the user views; the full input × action set is indexed.
        with self.image_lock:
            folder = self.directory/job_id/f'input-{input_index}'
            folder.mkdir(exist_ok=True)
            original_path = folder/'original.png'
            if not original_path.exists():
                self.datasets.image(job['config']['dataset_id'], input_index).save(original_path)
            output_path = folder/(name+'.png')
            if not output_path.exists():
                with Image.open(original_path) as original:
                    apply_image(original.convert('RGB'), actions[name]).save(output_path)
            return output_path.read_bytes()

    def update(self, job_id, **fields):
        with self.lock:
            self.jobs[job_id].update(fields)

    def start(self, config):
        dataset = self.datasets.ready(config['dataset_id']) if config['task'] == 'classification' else None
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
            self.jobs[job_id] = job
            self.active = job_id
            threading.Thread(target=self.generate, args=(job_id, config), daemon=True).start()
            return copy.deepcopy(job)

    def generate(self, job_id, config):
        try:
            self.update(job_id, status='running')
            folder = self.directory/job_id
            task = config['task']
            if task == 'classification':
                manifest = self.get(job_id)
                manifest.update(status='completed', completed=manifest['total'])
                (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
                self.update(job_id, status='completed', completed=manifest['total'])
                return
            original_text = config['text']
            if task == 'generation':
                # Keep a concrete reference prompt even when its values are removed
                # from the action space. Placeholders are never model inputs.
                for key, values in default_action_space('generation').items():
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
                    prompt = config['text']
                    for key, value in action.items():
                        if key != 'id':
                            prompt = prompt.replace('{'+key+'}', value)
                    node['text'] = prompt
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


def create_server(port=8000, directory=None):
    previews = PreviewManager(directory or ROOT/'runs'/'previews')
    models = ModelManager()
    image_encoder, other_encoder = HuggingFaceEncoder, create_encoder
    def encoder_factory(model):
        return image_encoder(model) if model.get('task') == 'classification' else other_encoder(model)
    embeddings = EmbeddingManager(previews, previews.directory.parent/'embeddings', encoder_factory=encoder_factory)

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
                match = re.fullmatch(r'/api/datasets/([0-9a-f]{32})(/items)?', path)
                if match:
                    if match[2]:
                        return self.send(200, previews.datasets.page(match[1], int(query.get('offset', ['0'])[0]), int(query.get('limit', ['100'])[0])))
                    return self.send(200, previews.datasets.get(match[1]))
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
                          '/embedding-view.js':'embedding-view.js', '/embedding-controller.js':'embedding-controller.js', '/style.css': 'style.css'}
                if path in static:
                    file = ROOT/'static'/static[path]
                    return self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or 'text/plain')
                self.send(404, {'error': 'Not found.'})
            except (KeyError, FileNotFoundError):
                self.send(404, {'error': 'Preview or file not found.'})
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
                if self.path == '/api/embeddings':
                    if not isinstance(data, dict) or not isinstance(data.get('preview_id'), str) or not re.fullmatch(r'[0-9a-f]{32}', data['preview_id']):
                        raise ValueError('Generate an image preview before loading its embeddings.')
                    return self.send(202, embeddings.start(data['preview_id']))
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
                self.send(404, {'error': 'Dataset not found. Load it again.'})

        def log_message(self, fmt, *args):
            if len(args) > 1 and str(args[1]) not in ('200', '202'):
                super().log_message(fmt, *args)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.previews = previews
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
    print(f'failureshiftRL: {address}\nPress Ctrl+C to stop.', flush=True)
    if args.open:
        threading.Timer(.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
