"""Install/check the Vev worker overlay without changing the app environment."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


APP = Path(__file__).resolve().parent
RUNTIME = APP / '.vev-runtime'
REQUIREMENTS = APP / 'requirements-vev.txt'
REPAIR = 'Run .venv/bin/python -E setup_vev.py from web_interface (or bash setup.sh).'
SMOKE_IMPORTS = {
    'torch': ('Tensor',),
    'PIL.Image': ('open',),
    'accelerate': ('Accelerator',),
    'transformers': ('AutoProcessor', 'AutoModelForImageTextToText', 'BitsAndBytesConfig',
                     'Qwen3_5ForConditionalGeneration'),
    'peft': ('LoraConfig',),
    'bitsandbytes': ('nn',),
    'scipy.special': ('softmax',),
    'pydantic': ('BaseModel',),
    'fla': ('__version__',),
}


def _distributions(paths=None):
    result = {}
    for item in metadata.distributions(path=sys.path if paths is None else paths):
        result.setdefault(canonicalize_name(item.metadata['Name']), item)
    return result


def _shared_only(name):
    return name in {'torch', 'torchvision', 'torchaudio', 'triton'} or name.startswith('nvidia-')


def _pins(requirements):
    if sys.version_info < (3, 12):
        raise RuntimeError('The pinned Vev runtime requires Python 3.12 or newer; '
                           'create the app .venv with a supported Python before running setup.')
    pins = []
    for line in requirements.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        item = Requirement(line)
        if item.marker and not item.marker.evaluate():
            continue
        name = canonicalize_name(item.name)
        if _shared_only(name):
            raise ValueError(f'{name} must stay shared with the app; Torch/CUDA cannot be installed here.')
        specifiers = list(item.specifier)
        if item.url or item.extras or len(specifiers) != 1 or specifiers[0].operator != '==' \
                or '*' in specifiers[0].version:
            raise ValueError(f'Vev requirements must use exact package==version pins: {line}')
        pins.append(item)
    if not pins:
        raise ValueError('Vev requirements are empty.')
    return pins


def _signature(requirements):
    return {
        'python': list(sys.version_info[:2]),
        'requirements_sha256': hashlib.sha256(requirements.read_bytes()).hexdigest(),
    }


def _validate_dependencies(runtime, pins):
    local = _distributions([str(runtime)])
    for name in local:
        if _shared_only(name):
            raise RuntimeError(f'Unexpected {name} in the Vev overlay; Torch and CUDA must remain shared.')
    available = {**_distributions(), **local}
    pending = [(item, 'Vev requirements', '') for item in pins]
    checked = set()
    while pending:
        item, parent, extra = pending.pop()
        if item.marker and not item.marker.evaluate({'extra': extra}):
            continue
        name = canonicalize_name(item.name)
        installed = available.get(name)
        if installed is None or not item.specifier.contains(installed.version, prereleases=True):
            actual = installed.version if installed else 'not installed'
            raise RuntimeError(f'{parent} requires {item}; found {actual}.')
        for active_extra in ('', *sorted(item.extras)):
            key = (name, active_extra)
            if key in checked:
                continue
            checked.add(key)
            for dependency in installed.requires or ():
                pending.append((Requirement(dependency), f'{name} {installed.version}', active_extra))


def _smoke_imports(runtime):
    # Use the same interpreter and -E bootstrap as the worker. The parent never
    # imports overlay modules; no model weights, network calls or GPU tensors.
    code = """
import importlib, json, sys
sys.path.insert(0, sys.argv[1])
for name, exports in json.loads(sys.argv[2]).items():
    module = importlib.import_module(name)
    for export in exports:
        getattr(module, export)
"""
    result = subprocess.run(
        [sys.executable, '-E', '-c', code, str(runtime), json.dumps(SMOKE_IMPORTS)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
             'CUDA_VISIBLE_DEVICES': ''},
    )
    if result.returncode:
        raise RuntimeError('Vev import check failed:\n' + (result.stderr or result.stdout).strip())


def check_runtime(runtime=RUNTIME, requirements=REQUIREMENTS, *, require_marker=True):
    runtime, requirements = Path(runtime), Path(requirements)
    if not runtime.is_dir():
        raise RuntimeError(f'Vev runtime is missing: {runtime}')
    if require_marker:
        try:
            marker = json.loads((runtime / '.ready.json').read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError('Vev runtime has not completed setup.') from exc
        if marker != _signature(requirements):
            raise RuntimeError('Vev runtime needs setup for this Python or dependency version.')
    _validate_dependencies(runtime, _pins(requirements))
    _smoke_imports(runtime)


def install_runtime(runtime=RUNTIME, requirements=REQUIREMENTS):
    runtime, requirements = Path(runtime), Path(requirements)
    pins = _pins(requirements)
    try:
        check_runtime(runtime, requirements)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        pass
    else:
        print(f'Vev runtime is ready: {runtime}')
        return
    before = {name: item.version for name, item in _distributions().items()}
    needed = [str(item) for item in pins
              if canonicalize_name(item.name) not in before or not item.specifier.contains(
                  before.get(canonicalize_name(item.name), '0'), prereleases=True)]
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.vev-runtime-build-', dir=runtime.parent) as temporary:
        staging = Path(temporary) / 'runtime'
        staging.mkdir()
        if needed:
            print('Installing isolated Vev packages; existing Torch/CUDA are shared.', flush=True)
            subprocess.run(
                [sys.executable, '-I', '-m', 'pip', 'install', '--disable-pip-version-check',
                 '--no-input', '--no-cache-dir', '--retries', '1', '--timeout', '60',
                 '--only-binary=:all:', '--no-deps', '--target', str(staging), *needed],
                check=True,
            )
        check_runtime(staging, requirements, require_marker=False)
        after = {name: item.version for name, item in _distributions().items()}
        if after != before:
            raise RuntimeError('The app environment changed during Vev setup; the new runtime was not activated.')
        (staging / '.ready.json').write_text(json.dumps(_signature(requirements), indent=2) + '\n')
        previous = Path(temporary) / 'previous'
        if runtime.exists():
            runtime.rename(previous)
        try:
            staging.rename(runtime)
        except OSError:
            if previous.exists():
                previous.rename(runtime)
            raise
    print(f'Vev runtime is ready: {runtime}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check without installing or downloading.')
    parser.add_argument('--quiet', action='store_true', help='Suppress the check status.')
    args = parser.parse_args(argv)
    try:
        if args.check:
            check_runtime()
            if not args.quiet:
                print(f'Vev runtime is ready: {RUNTIME}')
        else:
            install_runtime()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if not args.quiet:
            print(f'Vev setup: {exc}\n{REPAIR}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
