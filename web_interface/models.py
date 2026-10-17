"""Select local model files or download a Hugging Face snapshot. No inference."""
import copy
import re
import threading
import uuid
from pathlib import Path


def download_snapshot(model_id):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError('Model download dependencies are missing. Stop the server and run bash run.sh from web_interface to set up its environment.') from exc
    return snapshot_download(repo_id=model_id, repo_type='model')


class ModelManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None

    def get(self, job_id):
        with self.lock:
            return copy.deepcopy(self.jobs[job_id])

    def select(self, data):
        if not isinstance(data, dict):
            raise ValueError('Enter a model source and name.')
        source, name = data.get('source'), data.get('name')
        if not isinstance(name, str) or not name.strip() or len(name) > 1024:
            raise ValueError('Enter a local path or Hugging Face model ID.')
        name = name.strip()
        if source not in ('local', 'huggingface'):
            raise ValueError('Choose Local model or Hugging Face.')
        if source == 'local':
            path = Path(name).expanduser().resolve()
            if not path.is_file() and not path.is_dir():
                raise ValueError('The local model path does not exist on this computer.')
        elif not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?', name) or '..' in name:
            raise ValueError('Use a Hugging Face model ID, for example google/vit-base-patch16-224.')
        with self.lock:
            if self.active:
                raise RuntimeError('A model download is already in progress.')
            job_id = uuid.uuid4().hex
            job = dict(id=job_id, source=source, name=name, status='ready' if source == 'local' else 'downloading')
            if source == 'local':
                job['path'] = str(path)
            self.jobs[job_id] = job
            if source == 'huggingface':
                self.active = job_id
                threading.Thread(target=self.download, args=(job_id, name), daemon=True).start()
            return copy.deepcopy(job)

    def download(self, job_id, name):
        try:
            path = download_snapshot(name)
            with self.lock:
                self.jobs[job_id].update(status='ready', path=str(path))
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(status='error', error=str(exc))
        finally:
            with self.lock:
                self.active = None
