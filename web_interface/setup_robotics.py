"""Install/check robotics worker packages without modifying the app environment.

Run with the app interpreter: .venv/bin/python -E setup_robotics.py --render
Workers must call bootstrap_runtime() before importing NumPy, Torch or models.
The server may call runtime_status(); it imports no simulator/model libraries.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile


APP = Path(__file__).resolve().parent
RUNTIME = APP / ".robotics-runtime"
REQUIREMENTS = APP / "requirements-robotics.txt"
COMMAND = ".venv/bin/python -E setup_robotics.py --render"
REPAIR = f"Run {COMMAND} from web_interface."
SMOKE_IMPORTS = {
    "numpy": ("array",),
    "torch": ("jit",),
    "torchvision": ("models",),
    "mujoco": ("MjModel",),
    "robosuite": ("make",),
    "robomimic.utils.file_utils": ("policy_from_checkpoint",),
    "stable_baselines3": ("PPO", "SAC"),
    "lerobot.policies.factory": ("get_policy_class", "make_pre_post_processors"),
    "lerobot.policies.act.modeling_act": ("ACTPolicy",),
    "lerobot.policies.diffusion.modeling_diffusion": ("DiffusionPolicy",),
    "lerobot.policies.smolvla.modeling_smolvla": ("SmolVLAPolicy",),
    "lerobot.policies.pi0.modeling_pi0": ("PI0Policy",),
    "lerobot.policies.pi05.modeling_pi05": ("PI05Policy",),
    "lerobot.policies.pi0_fast.modeling_pi0_fast": ("PI0FastPolicy",),
}

# LeRobot's published release caps dependencies below current security fixes.
# Accept only an immutable, SHA-256-pinned official source archive until release.
SOURCE_ARCHIVE = re.compile(
    r'https://codeload\.github\.com/huggingface/lerobot/tar\.gz/'
    r'[0-9a-f]{40}#sha256=(?P<sha256>[0-9a-f]{64})')


def _validate_source_identity(requirement, installed):
    if not requirement.url:
        return
    match = SOURCE_ARCHIVE.fullmatch(requirement.url)
    try:
        provenance = json.loads(installed.read_text('direct_url.json') or '{}')
        valid = (match is not None and isinstance(provenance, dict)
                 and provenance.get('url') == requirement.url.split('#', 1)[0]
                 and provenance.get('archive_info', {}).get('hashes', {}).get('sha256') == match['sha256'])
    except (ValueError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise RuntimeError('The installed LeRobot source archive identity does not match its checksum pin.')


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _shared_only(name):
    name = _canonical(name)
    return name in {"torch", "torchvision", "torchaudio", "triton"} or name.startswith(
        ("nvidia-", "cuda-")
    )


def _distributions(paths=None):
    result = {}
    for item in metadata.distributions(path=sys.path if paths is None else paths):
        name = item.metadata.get("Name")
        if name:
            result.setdefault(_canonical(name), item)
    return result


def _shared_versions():
    # Explicit interpreter paths exclude any overlay already inserted by a worker.
    paths = list({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    return {name: item.version for name, item in _distributions(paths).items()
            if _shared_only(name)}


def _pins(requirements):
    from packaging.requirements import Requirement

    if sys.version_info < (3, 12):
        raise RuntimeError("The robotics runtime requires Python 3.12 or newer.")
    pins = []
    for line in Path(requirements).read_text().splitlines():
        # A URL fragment is part of its checksum pin, not an inline comment.
        line = re.sub(r"(^|\s+)#.*$", "", line).strip()
        if not line:
            continue
        item = Requirement(line)
        if item.marker and not item.marker.evaluate():
            continue
        if _shared_only(item.name):
            raise ValueError(f"{item.name} must remain shared; Torch/CUDA cannot be installed here.")
        specifiers = list(item.specifier)
        if item.url:
            if _canonical(item.name) != 'lerobot' or not SOURCE_ARCHIVE.fullmatch(item.url):
                raise ValueError('Source pins require an official LeRobot archive, full commit and SHA-256 checksum.')
        elif len(specifiers) != 1 or specifiers[0].operator != "==" or "*" in specifiers[0].version:
            raise ValueError(f"Robotics requirements need exact package==version pins: {line}")
        pins.append(item)
    if not pins:
        raise ValueError("Robotics requirements are empty.")
    return pins


def _signature(requirements):
    return {
        "schema": 1,
        "python": list(sys.version_info[:2]),
        "requirements_sha256": hashlib.sha256(Path(requirements).read_bytes()).hexdigest(),
        "shared": _shared_versions(),
    }


def _marker(runtime, requirements):
    if not runtime.is_dir():
        raise RuntimeError(f"Robotics runtime is missing: {runtime}")
    try:
        marker = json.loads((runtime / ".ready.json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Robotics runtime has not completed setup.") from exc
    if not isinstance(marker, dict) or marker.get("signature") != _signature(requirements):
        raise RuntimeError("Robotics runtime needs setup for this Python, shared Torch or dependency version.")
    if not marker.get("imports_checked"):
        raise RuntimeError("Robotics runtime has not passed its import checks.")
    installed = {name: item.version for name, item in _distributions([str(runtime)]).items()}
    for name in installed:
        if _shared_only(name):
            raise RuntimeError(f"Unexpected {name} in robotics overlay; Torch/CUDA must remain shared.")
    if marker.get("installed") != installed:
        raise RuntimeError("Robotics runtime packages changed or are missing; run setup again.")
    return marker


def runtime_status(runtime=RUNTIME, requirements=REQUIREMENTS):
    """Cheap server status; no NumPy, Torch, simulator import or child process."""
    runtime, requirements = Path(runtime), Path(requirements)
    status = {"available": False, "path": str(runtime), "command": COMMAND, "repair": REPAIR,
              "message": "", "versions": {}}
    try:
        marker = _marker(runtime, requirements)
    except (OSError, RuntimeError, ValueError) as exc:
        status["message"] = str(exc)
    else:
        status.update(available=True, message="Robotics runtime is installed.",
                      versions=marker.get("versions", {}), render=marker.get("render"))
    return status


runtime_available = runtime_status


def _validate_dependencies(runtime, pins):
    from packaging.requirements import Requirement

    local = _distributions([str(runtime)])
    for name in local:
        if _shared_only(name):
            raise RuntimeError(f"Unexpected {name} in robotics overlay; Torch/CUDA must remain shared.")
    for name in ("torch", "torchvision", "torchaudio", "triton", "nvidia", "cuda"):
        if (runtime / name).exists():
            raise RuntimeError(f"Unexpected {name} package in robotics overlay; Torch/CUDA must remain shared.")
    available = {**_distributions(), **local}
    pending = [(item, "Robotics requirements", "") for item in pins]
    checked = set()
    while pending:
        item, parent, extra = pending.pop()
        if item.marker and not item.marker.evaluate({"extra": extra}):
            continue
        name = _canonical(item.name)
        installed = available.get(name)
        if installed is None or not item.specifier.contains(installed.version, prereleases=True):
            actual = installed.version if installed else "not installed"
            raise RuntimeError(f"{parent} requires {item}; found {actual}.")
        _validate_source_identity(item, installed)
        for active_extra in ("", *sorted(item.extras)):
            key = (name, active_extra)
            if key in checked:
                continue
            checked.add(key)
            for dependency in installed.requires or ():
                pending.append((Requirement(dependency), f"{name} {installed.version}", active_extra))
    return {name: available[name].version for name, _ in checked}


def _child_environment():
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", environment["MUJOCO_GL"])
    return environment


def _smoke_imports(runtime):
    code = """
import importlib, json, sys
sys.path.insert(0, sys.argv[1])
for name, exports in json.loads(sys.argv[2]).items():
    module = importlib.import_module(name)
    for export in exports:
        getattr(module, export)
"""
    result = subprocess.run(
        [sys.executable, "-E", "-c", code, str(runtime), json.dumps(SMOKE_IMPORTS)],
        capture_output=True, text=True, timeout=180,
        env={**_child_environment(), "CUDA_VISIBLE_DEVICES": ""},
    )
    if result.returncode:
        raise RuntimeError("Robotics import check failed:\n" + (result.stderr or result.stdout).strip())


def _smoke_render(runtime):
    """Create, reset, step and render a real environment; no model downloads."""
    code = """
import json, os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import robosuite
env = robosuite.make('Lift', robots='Panda', has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=True,
                     camera_names='agentview', camera_heights=96, camera_widths=96,
                     control_freq=20, horizon=5)
try:
    obs = env.reset()
    for _ in range(3):
        obs, reward, done, info = env.step(np.zeros(env.action_dim))
    frame = obs['agentview_image']
    assert frame.shape == (96, 96, 3), frame.shape
    assert np.isfinite(frame).all() and frame.max() > frame.min(), 'Empty rendered frame'
    print('ROBOTICS_RENDER=' + json.dumps({'backend': os.environ['MUJOCO_GL'],
          'task': 'Lift', 'robot': 'Panda', 'steps': 3, 'action_dim': env.action_dim,
          'frame_shape': list(frame.shape)}))
finally:
    env.close()
"""
    result = subprocess.run([sys.executable, "-E", "-c", code, str(runtime)],
                            capture_output=True, text=True, timeout=180, env=_child_environment())
    if result.returncode:
        raise RuntimeError("Robotics headless render check failed:\n" + (result.stderr or result.stdout).strip())
    for line in result.stdout.splitlines():
        if line.startswith("ROBOTICS_RENDER="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError("Robotics render check did not return a verified frame.")


def check_runtime(runtime=RUNTIME, requirements=REQUIREMENTS, *, require_marker=True,
                  smoke=True, render=False):
    runtime, requirements = Path(runtime), Path(requirements)
    if require_marker:
        _marker(runtime, requirements)
    elif not runtime.is_dir():
        raise RuntimeError(f"Robotics runtime is missing: {runtime}")
    versions = _validate_dependencies(runtime, _pins(requirements))
    if smoke:
        _smoke_imports(runtime)
    return {"versions": versions, "render": _smoke_render(runtime) if render else None}


def bootstrap_runtime(runtime=RUNTIME, requirements=REQUIREMENTS):
    """Activate only in a fresh worker, before importing numerical/model modules."""
    runtime = Path(runtime).resolve()
    for name in ("numpy", "torch", "torchvision", "mujoco", "robosuite", "robomimic",
                 "stable_baselines3", "lerobot", "transformers", "diffusers"):
        if name in sys.modules:
            raise RuntimeError(f"Robotics bootstrap must run before importing {name}.")
    check_runtime(runtime, requirements, smoke=False)
    sys.path.insert(0, str(runtime))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    return runtime


def install_runtime(runtime=RUNTIME, requirements=REQUIREMENTS, *, render=False):
    runtime, requirements = Path(runtime), Path(requirements)
    pins = _pins(requirements)
    try:
        check_runtime(runtime, requirements, render=render)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        pass
    else:
        print(f"Robotics runtime is ready: {runtime}")
        return
    before = {name: item.version for name, item in _distributions().items()}
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".robotics-runtime-build-", dir=runtime.parent) as temporary:
        staging = Path(temporary) / "runtime"
        staging.mkdir()
        print("Installing isolated robotics packages; existing Torch/CUDA are shared.", flush=True)
        command = [sys.executable, "-I", "-m", "pip", "install", "--disable-pip-version-check",
                   "--no-input", "--retries", "1", "--timeout", "60",
                   "--no-deps", "--target", str(staging)]
        # egl-probe builds two native EGL query tools. Install the pinned CMake
        # first, then use its native binary without adding the overlay to Python.
        # Its old CMakeLists needs this documented CMake 4 compatibility setting.
        build_tools = [item for item in pins if _canonical(item.name) == "cmake"]
        if build_tools:
            subprocess.run([*command, *map(str, build_tools)], check=True)
        build_environment = {**os.environ, "CMAKE_POLICY_VERSION_MINIMUM": "3.5"}
        if build_tools:
            build_environment["PATH"] = str(staging / "cmake" / "data" / "bin") + os.pathsep + os.environ.get("PATH", "")
        packages = [item for item in pins if _canonical(item.name) != "cmake"]
        if packages:
            subprocess.run([*command, *map(str, packages)], check=True, env=build_environment)
        result = check_runtime(staging, requirements, require_marker=False, render=render)
        after = {name: item.version for name, item in _distributions().items()}
        if before != after:
            raise RuntimeError("The app environment changed during setup; robotics runtime was not activated.")
        marker = {"signature": _signature(requirements), "versions": result["versions"],
                  "imports_checked": True, "render": result["render"],
                  "installed": {name: item.version
                                for name, item in _distributions([str(staging)]).items()}}
        (staging / ".ready.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        previous = Path(temporary) / "previous"
        if runtime.exists():
            runtime.rename(previous)
        try:
            staging.rename(runtime)
        except OSError:
            if previous.exists():
                previous.rename(runtime)
            raise
    print(f"Robotics runtime is ready: {runtime}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without installing or downloading.")
    parser.add_argument("--render", action="store_true", help="Also step/render a real Lift/Panda environment.")
    parser.add_argument("--quiet", action="store_true", help="Suppress check status.")
    args = parser.parse_args(argv)
    try:
        if args.check:
            result = check_runtime(render=args.render)
            if not args.quiet:
                print(json.dumps({"available": True, "path": str(RUNTIME), **result}, indent=2))
        else:
            install_runtime(render=args.render)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if not args.quiet:
            print(f"Robotics setup: {exc}\n{REPAIR}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
