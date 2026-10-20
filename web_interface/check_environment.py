"""Check the imports actually needed by the local interface, without downloading data."""
import argparse
import importlib
from importlib.metadata import PackageNotFoundError, requires as package_requirements, version
from pathlib import Path
import re
import subprocess
import sys


def supported_torch(installed):
    match = re.match(r'^(\d+)\.(\d+)', installed)
    return bool(match and (2, 6) <= tuple(map(int, match.groups())) < (3, 0))


def nvidia_drivers():
    """Read the installed driver, without requiring a local CUDA toolkit."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def nvidia_capabilities():
    """Read GPU architectures even when the installed wheel cannot run kernels."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10, check=True,
        )
        capabilities = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if capabilities and all(re.fullmatch(r'\d+\.\d+', value) for value in capabilities):
            return capabilities
    except (OSError, subprocess.SubprocessError):
        pass
    # Older nvidia-smi versions may not expose compute_cap. Reading device
    # properties does not require executable kernels in the installed wheel.
    try:
        torch = importlib.import_module('torch')
        if torch.cuda.is_available():
            return ['.'.join(map(str, torch.cuda.get_device_capability(index)))
                    for index in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def installed_torch():
    try:
        torch = importlib.import_module('torch')
        current = {'version': torch.__version__, 'package_version': version('torch'), 'cuda': torch.version.cuda,
                   'available': False}
        if torch.version.cuda and torch.cuda.is_available():
            try:
                check_torch_runtime(torch, require_cuda=True)
            except Exception as exc:
                current['runtime_error'] = str(exc)
            else:
                current['available'] = True
        return current
    except Exception:
        return None


def plan_torch_install(mode, current, drivers):
    """Choose a tested wheel; a CPU wheel cannot satisfy a CUDA request."""
    if mode not in ('auto', 'cpu', 'cuda'):
        raise ValueError(f'Unknown compute mode: {mode}')
    supported = bool(current and supported_torch(current['version']))
    working_cuda = bool(supported and current['cuda'] and current['available'])
    backend = 'cuda' if mode == 'cuda' or (mode == 'auto' and (working_cuda or drivers)) else 'cpu'
    if supported and ((backend == 'cuda' and working_cuda) or
                      (backend == 'cpu' and (mode == 'auto' or not current['cuda']))):
        return {'backend': backend, 'install': False, 'requirement': '-', 'index_url': '-',
                'reason': f"Keeping PyTorch {current['version']} ({backend.upper()})."}

    wheel, release = 'cpu', '2.11.0'
    if backend == 'cuda':
        if not drivers:
            raise RuntimeError('CUDA requested, but no usable NVIDIA GPU/driver was detected. '
                               'Check nvidia-smi and GPU visibility, or use bash setup.sh --cpu.')
        parsed = [tuple(map(int, match.groups())) for driver in drivers
                  if (match := re.match(r'^(\d+)\.(\d+)', driver))]
        minimum = min(parsed) if parsed else (0, 0)
        if minimum < (520, 61):
            raise RuntimeError(f'NVIDIA driver {", ".join(drivers)} is too old for the supported '
                               'CUDA wheels. Use bash setup.sh --cpu or install a supported driver.')
        capabilities = nvidia_capabilities()
        if not capabilities or not all(re.fullmatch(r'\d+\.\d+', value) for value in capabilities):
            raise RuntimeError('Cannot determine the NVIDIA GPU compute capability. Run '
                               'nvidia-smi --query-gpu=compute_cap --format=csv,noheader. '
                               'Use bash setup.sh --cpu or install a compatible CUDA wheel manually.')
        architectures = [tuple(map(int, value.split('.'))) for value in capabilities]
        if any(major not in (5, 6, 7, 8, 9, 10, 12) for major, _ in architectures):
            raise RuntimeError(f'GPU architecture (compute capability {", ".join(capabilities)}) '
                               'is not supported by the automatic CUDA wheel selection. '
                               'Use bash setup.sh --cpu or install a compatible CUDA wheel manually.')
        legacy = any(architecture < (7, 5) for architecture in architectures)
        blackwell = any(architecture >= (10, 0) for architecture in architectures)
        if legacy and blackwell:
            raise RuntimeError('No single supported PyTorch 2.11 wheel covers both legacy '
                               '(pre-Turing) and Blackwell GPU architectures. Use separate '
                               'GPU environments with compatible wheels, or bash setup.sh --cpu.')
        if blackwell and minimum < (570, 26):
            raise RuntimeError('Blackwell GPU architectures require the CUDA 12.8 wheel and '
                               'NVIDIA driver 570.26 or newer. Update the driver or use bash setup.sh --cpu.')
        # Official wheel indexes and Linux driver requirements:
        # https://pytorch.org/get-started/previous-versions/
        # https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
        # PyTorch 2.11 cu128 dropped Maxwell/Pascal/Volta; cu126 retains them:
        # https://dev-discuss.pytorch.org/t/3290
        if minimum >= (570, 26) and not legacy:
            wheel = 'cu128'
        elif minimum >= (560, 28):
            wheel = 'cu126'
        elif minimum >= (520, 61):
            wheel, release = 'cu118', '2.7.1'
    # macOS CPU wheels omit the +cpu local version. Elsewhere retain the suffix
    # so pip cannot keep an installed CUDA build when --cpu explicitly switches.
    suffix = '' if backend == 'cpu' and sys.platform == 'darwin' else f'+{wheel}'
    requirement = f'torch=={release}{suffix}'
    return {'backend': backend, 'install': True,
            'requirement': requirement,
            'index_url': f'https://download.pytorch.org/whl/{wheel}',
            'reason': f'Installing {backend.upper()} PyTorch from the official {wheel} wheel index.'}


def installed_torchvision():
    """Keep distribution metadata even when a mismatched compiled wheel cannot import."""
    try:
        current = dict(version=version('torchvision'), available=False, torch_version=None)
        for requirement in package_requirements('torchvision') or ():
            match = re.match(r'^torch\s*\(?\s*==\s*([0-9][^;,\s)]+)', requirement)
            if match:
                current['torch_version'] = match[1].split('+', 1)[0]
    except PackageNotFoundError:
        return None
    try:
        loaded = importlib.import_module('torchvision')
        getattr(importlib.import_module('torchvision.transforms.v2'), 'ToImage')
    except Exception:
        pass
    else:
        current['package_version'] = current['version']
        current['version'] = loaded.__version__
        current['available'] = True
    return current


def plan_torchvision_install(current, vision=None):
    """Resolve vision on Torch's own wheel index while pinning the installed Torch."""
    if not current or not supported_torch(current['version']):
        raise RuntimeError('Install supported PyTorch before planning its torchvision wheel.')
    cuda = current.get('cuda')
    if cuda and not re.fullmatch(r'\d+\.\d+', cuda):
        raise RuntimeError('Cannot determine a torchvision wheel index for this CUDA build.')
    wheel = 'cu' + cuda.replace('.', '') if cuda else 'cpu'
    release, _, local = current['version'].partition('+')
    if local and local != wheel:
        raise RuntimeError(f'PyTorch build {current["version"]} needs a matching torchvision wheel; '
                           'install that wheel manually before running setup.')
    suffix = '' if wheel == 'cpu' and sys.platform == 'darwin' else '+' + wheel
    plan = dict(install=True, reinstall=False, requirement='torchvision>=0.21,<1',
                index_url=f'https://download.pytorch.org/whl/{wheel}',
                torch_requirement='torch==' + current.get('package_version', current['version']))
    if vision and vision.get('torch_version') == release:
        vision_release = vision['version'].split('+', 1)[0]
        match = re.fullmatch(r'(\d+)\.(\d+)\.\d+', vision_release)
        if match and (0, 21) <= tuple(map(int, match.groups())) < (1, 0):
            plan['requirement'] = 'torchvision==' + vision_release + suffix
            same_build = vision['version'] == vision_release + suffix
            plan['install'] = not (same_build and vision['available'])
            plan['reinstall'] = same_build and not vision['available']
            if not plan['install']:
                plan['requirement'] = 'torchvision==' + vision.get('package_version', vision['version'])
    action = 'Keeping' if not plan['install'] else 'Repairing' if plan['reinstall'] else 'Installing'
    plan['reason'] = (f'{action} torchvision for PyTorch {current["version"]} from the official {wheel} '
                      'wheel index; the Torch version stays pinned.')
    return plan


def _vision_plan_fields(plan):
    return ('reinstall' if plan['reinstall'] else 'install' if plan['install'] else 'keep',
            plan['requirement'], plan['index_url'], plan['torch_requirement'])


def check_torch_runtime(torch, require_cuda=False):
    """Exercise CUDA kernels and cuBLAS, not just the driver discovery API."""
    available = bool(torch.version.cuda and torch.cuda.is_available())
    if not available:
        if require_cuda:
            reason = 'CPU-only PyTorch is installed' if not torch.version.cuda else 'CUDA cannot initialize'
            raise RuntimeError(f'{reason}. Run bash setup.sh --cuda; check nvidia-smi and '
                               'CUDA_VISIBLE_DEVICES if CUDA is still unavailable.')
        return {'backend': 'cpu', 'cuda': torch.version.cuda}
    with torch.inference_mode():
        values = torch.ones((32, 32), device='cuda')
        result = values @ values
        torch.cuda.synchronize()
        if result[0, 0].item() != 32.0:
            raise RuntimeError('CUDA tensor verification produced an incorrect result.')
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {'backend': 'cuda', 'cuda': torch.version.cuda, 'device': device.name,
            'memory_gib': device.total_memory / (1024 ** 3)}


def check_environment(quiet=False, require_cuda=False):
    required = (
        ('Pillow', 'PIL.Image', ('open', 'new')),
        ('huggingface_hub', 'huggingface_hub', ('snapshot_download', 'hf_hub_download')),
        ('datasets', 'datasets', ('Image', 'Dataset', 'load_dataset')),
        ('numpy', 'numpy', ('linalg', 'array')),
        ('torch', 'torch', ('inference_mode', 'Tensor')),
        ('torchvision', 'torchvision', ('transforms', 'ops')),
        ('transformers', 'transformers', ('AutoImageProcessor', 'AutoModel', 'AutoTokenizer', 'AutoModelForSeq2SeqLM', 'AutoModelForImageClassification', 'AutoModelForSequenceClassification', 'CLIPModel', 'CLIPTextModel', 'CLIPTokenizer', 'Qwen3VLProcessor', 'Qwen3VLForConditionalGeneration', 'BlipProcessor', 'BlipForQuestionAnswering')),
        ('sentencepiece', 'sentencepiece', ('SentencePieceProcessor',)),
        ('protobuf', 'google.protobuf', ('__version__',)),
        ('ftfy', 'ftfy', ('fix_text',)),
        ('diffusers', 'diffusers', ('StableDiffusionPipeline', 'DDPMScheduler')),
        ('peft', 'peft', ('LoraConfig', 'get_peft_model_state_dict')),
        ('accelerate', 'accelerate', ('Accelerator',)),
    )
    failures = []
    torch = None
    if not quiet:
        print(f'Python: {sys.executable}')
    for package, module, exports in required:
        try:
            loaded = importlib.import_module(module)
            for name in exports:
                getattr(loaded, name)
            installed = version(package)
            if package == 'torch':
                if not supported_torch(installed):
                    raise ValueError('PyTorch >=2.6,<3 is required for weights-only loading.')
                torch = loaded
            if package == 'torchvision' and not (0,21) <= tuple(map(int, installed.split('.')[:2])) < (1,0):
                raise ValueError('This adapter requires torchvision >=0.21,<1.')
            if package == 'transformers' and not (4,57) <= tuple(map(int, installed.split('.')[:2])) < (5,0):
                raise ValueError('This adapter requires Transformers >=4.57,<5.')
        except Exception as exc:
            failures.append(package)
            if not quiet:
                print(f'MISSING OR BROKEN: {package}: {exc}')
        else:
            if not quiet:
                print(f'OK: {package} {installed}')
    if torch is not None:
        try:
            runtime = check_torch_runtime(torch, require_cuda=require_cuda)
        except Exception as exc:
            failures.append('CUDA runtime')
            if not quiet:
                print(f'BROKEN: CUDA runtime: {exc}')
        else:
            if not quiet:
                print(f'Torch CUDA build: {runtime["cuda"] or "none (CPU-only)"}')
                if runtime['backend'] == 'cuda':
                    print(f'CUDA available: True; {runtime["device"]} '
                          f'({runtime["memory_gib"]:.1f} GiB); GPU tensor operation passed.')
                else:
                    print('CUDA available: False; using CPU.')
                    if nvidia_drivers():
                        print('NVIDIA hardware detected. Enable GPU execution with: bash setup.sh --cuda')
    if failures and not quiet:
        print('Run bash setup.sh from web_interface to install the app requirements.')
    return 1 if failures else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--require-cuda', action='store_true', help='Fail unless a CUDA tensor operation succeeds.')
    planning = parser.add_mutually_exclusive_group()
    planning.add_argument('--setup-plan', choices=('auto', 'cpu', 'cuda'), help='Print the install plan used by setup.sh.')
    planning.add_argument('--vision-plan', action='store_true', help='Plan torchvision for the installed PyTorch build.')
    planning.add_argument('--installed-constraints', action='store_true', help='Pin installed Torch and torchvision during app setup.')
    parser.add_argument('--fresh', action='store_true', help='Plan for a new virtual environment.')
    args = parser.parse_args()
    if args.setup_plan:
        try:
            current = None if args.fresh else installed_torch()
            plan = plan_torch_install(args.setup_plan, current, nvidia_drivers())
            target = current
            if plan['install']:
                wheel = plan['index_url'].rsplit('/', 1)[1]
                target = dict(version=plan['requirement'].removeprefix('torch=='),
                              cuda=wheel[2:-1] + '.' + wheel[-1] if wheel.startswith('cu') else None)
            vision = plan_torchvision_install(target, None if args.fresh else installed_torchvision())
        except (ValueError, RuntimeError) as exc:
            print(f'Setup cannot continue: {exc}', file=sys.stderr)
            raise SystemExit(1)
        print(plan['reason'], file=sys.stderr)
        print(vision['reason'], file=sys.stderr)
        print('\t'.join((plan['backend'], 'install' if plan['install'] else 'keep',
                         plan['requirement'], plan['index_url'], *_vision_plan_fields(vision))))
        raise SystemExit(0)
    if args.vision_plan or args.installed_constraints:
        try:
            if args.vision_plan:
                plan = plan_torchvision_install(installed_torch(), installed_torchvision())
                print(plan['reason'], file=sys.stderr)
                print('\t'.join(_vision_plan_fields(plan)))
            else:
                for package in ('torch', 'torchvision'):
                    print(f'{package}=={version(package)}')
        except (ValueError, RuntimeError, PackageNotFoundError) as exc:
            print(f'Setup cannot continue: {exc}', file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(0)
    status = check_environment(args.quiet, require_cuda=args.require_cuda)
    if status == 0:
        command = [sys.executable, '-E', str(Path(__file__).with_name('setup_vev.py')), '--check']
        if args.quiet:
            command.append('--quiet')
        try:
            status = subprocess.run(command, timeout=150).returncode
        except (OSError, subprocess.SubprocessError) as exc:
            status = 1
            if not args.quiet:
                print(f'Vev runtime check failed: {exc}. Run bash setup.sh from web_interface.')
    raise SystemExit(status)
