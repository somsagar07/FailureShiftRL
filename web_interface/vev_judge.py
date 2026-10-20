"""Persistent Vev worker; the main app keeps its existing Transformers runtime."""
from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import threading
import time

from PIL import Image

from vev_backend import MAX_PIXELS, cache_directory, normalize_selection, validate_rows


ROOT = Path(__file__).resolve().parent


def _worker_command():
    if not (ROOT / '.vev-runtime' / 'transformers').is_dir():
        raise RuntimeError('Vev dependencies are missing. Run: .venv/bin/python -E setup_vev.py')
    return [sys.executable, '-E', str(ROOT / 'vev_worker.py')]


def _check_cancelled(cancelled):
    if cancelled is None:
        return
    if not callable(cancelled):
        raise ValueError('Generation cancellation must be a callable returning a boolean.')
    state = cancelled()
    if not isinstance(state, bool):
        raise ValueError('Generation cancellation must return a boolean.')
    if state:
        raise InterruptedError('Cancelled by user.')


class VevJudge:
    """Load once, reuse for successive images, and reap the worker on interruption."""

    kind = 'vev'
    supports_decisions = True

    def __init__(self, selection, device, *, batch_size=4, timeout=1800.):
        self.selection = normalize_selection(selection)
        if device not in ('cpu', 'cuda'):
            raise ValueError('Vev requires a resolved cpu or cuda device.')
        if type(batch_size) is not int or not 1 <= batch_size <= 4:
            raise ValueError('Vev batches must contain between 1 and 4 questions.')
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('Vev timeout must be a positive finite number of seconds.')
        self.device, self.batch_size, self.timeout = device, batch_size, timeout
        self.identity = {**self.selection, 'architecture': 'vev'}
        self._process = None
        self._staging = []
        self._stderr = bytearray()
        self._lock = threading.RLock()

    def _stop(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            else:
                process.wait()
            for pipe in (process.stdin, process.stdout, process.stderr):
                pipe.close()
        # Only these unique directories belong to this child. Completed compact
        # checkpoints have already been moved out atomically by the backend.
        for directory in self._staging:
            directory.cleanup()
        self._staging.clear()

    def close(self):
        with self._lock:
            self._stop()

    def _log(self, data):
        self._stderr.extend(data)
        del self._stderr[:-16384]

    def _crash(self, process):
        # stdout and stderr can become readable in either order at process exit.
        while True:
            try:
                tail = os.read(process.stderr.fileno(), 65536)
            except BlockingIOError:
                break
            if not tail:
                break
            self._log(tail)
        detail = self._stderr.decode('utf-8', errors='replace')[-2400:].strip()
        return RuntimeError('Vev worker stopped before returning a decision.'
                            + (f'\n{detail}' if detail else ''))

    def _request(self, payload, cancelled):
        process = self._process
        encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False,
                              separators=(',', ':')) + '\n').encode('utf-8')
        if len(encoded) > 16 * 1024**2:
            raise ValueError('The Vev image and questions exceed the request size limit.')
        output = bytearray()
        sent = 0
        started = time.monotonic()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, 'out')
                selector.register(process.stderr, selectors.EVENT_READ, 'err')
                selector.register(process.stdin, selectors.EVENT_WRITE, 'in')
                while True:
                    _check_cancelled(cancelled)
                    if time.monotonic() - started > self.timeout:
                        raise TimeoutError('Vev evaluation timed out; its worker has been stopped.')
                    for key, _ in selector.select(timeout=.1):
                        if key.data == 'in':
                            try:
                                sent += os.write(key.fd, encoded[sent:sent + 65536])
                            except BrokenPipeError:
                                raise self._crash(process) from None
                            if sent == len(encoded):
                                selector.unregister(process.stdin)
                            continue
                        data = os.read(key.fd, 65536)
                        if key.data == 'err':
                            if data:
                                self._log(data)
                            else:
                                selector.unregister(process.stderr)
                            continue
                        if not data:
                            raise self._crash(process)
                        output.extend(data)
                        if len(output) > 4 * 1024**2:
                            raise RuntimeError('The Vev response exceeded its size limit.')
                        if b'\n' not in output:
                            continue
                        line, extra = output.split(b'\n', 1)
                        if extra.strip():
                            raise RuntimeError('Unexpected extra data in the Vev response.')
                        try:
                            response = json.loads(line)
                        except (ValueError, UnicodeError) as exc:
                            raise RuntimeError('The Vev worker returned an invalid JSON response.') from exc
                        if not isinstance(response, dict) or type(response.get('ok')) is not bool:
                            raise RuntimeError('The Vev worker returned an invalid response.')
                        _check_cancelled(cancelled)
                        if not response['ok']:
                            message = str(response.get('error', 'Vev evaluation failed.'))
                            error_type = response.get('error_type')
                            if error_type in ('ValueError', 'FileNotFoundError'):
                                raise ValueError(message)
                            raise RuntimeError(message)
                        return response
        except BaseException:
            self._stop()
            raise

    def ensure_ready(self, *, cancelled=None):
        with self._lock:
            _check_cancelled(cancelled)
            if self._process is not None and self._process.poll() is None:
                return
            self._stop()
            self._stderr.clear()
            command = _worker_command()
            root = cache_directory()
            root.mkdir(parents=True, exist_ok=True)
            ram = Path('/dev/shm')
            ram_parent = ram if ram.is_dir() and os.access(ram, os.W_OK) else root
            environment = os.environ.copy()
            try:
                for variable, parent in (('VEV_STAGING_DIR', ram_parent),
                                         ('VEV_COMPACT_STAGING_DIR', root)):
                    directory = tempfile.TemporaryDirectory(prefix='failureshift-vev-worker-', dir=parent)
                    self._staging.append(directory)
                    environment[variable] = directory.name
                environment['HF_XET_CACHE'] = str(Path(environment['VEV_STAGING_DIR']) / 'xet')
                environment['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
                self._process = subprocess.Popen(command, cwd=ROOT, env=environment,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                for pipe in (self._process.stdin, self._process.stdout, self._process.stderr):
                    os.set_blocking(pipe.fileno(), False)
                response = self._request(dict(command='load', selection=self.selection,
                                              device=self.device, batch_size=self.batch_size), cancelled)
            except BaseException:
                self._stop()
                raise
            identity = response.get('identity')
            if not isinstance(identity, dict) or not isinstance(identity.get('name'), str):
                self._stop()
                raise RuntimeError('The Vev worker did not return its model identity.')
            self.identity = dict(identity)

    def decide(self, image, questions, *, cancelled=None):
        _check_cancelled(cancelled)
        questions = validate_rows(questions)
        if (not isinstance(image, Image.Image) or image.width < 1 or image.height < 1
                or image.width * image.height > MAX_PIXELS):
            raise ValueError(f'Vev requires a PIL image with at most {MAX_PIXELS} pixels.')
        with io.BytesIO() as buffer:
            image.convert('RGB').save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        with self._lock:
            self.ensure_ready(cancelled=cancelled)
            response = self._request(dict(command='decide', image=encoded, questions=questions), cancelled)
            answers = response.get('answers')
            if not isinstance(answers, list) or len(answers) != len(questions):
                self._stop()
                raise RuntimeError('The Vev worker must return one answer for every question.')
            # generation_failure validates ids, probability distributions and
            # readouts before converting any decision into a reward.
            return answers

    def evaluate(self, image, checks, *, cancelled=None):
        from generation_failure import evaluate_requirements
        return evaluate_requirements(self, image, checks, cancelled=cancelled)
