"""Exercise isolated robotics setup using local wheels, without model downloads."""
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def wheel(directory, version="1.0", dependencies=("torch>=2.7,<2.12", "packaging>=20"),
          source=None):
    path = directory / f"robotics_probe-{version}-py3-none-any.whl"
    info = f"robotics_probe-{version}.dist-info"
    files = {
        "robotics_probe.py": source or f"READY = {version!r}\n",
        f"{info}/METADATA": "Metadata-Version: 2.1\nName: robotics-probe\n"
        f"Version: {version}\n" + "".join(f"Requires-Dist: {item}\n" for item in dependencies),
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: tests\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files[f"{info}/RECORD"] = "".join(f"{name},,\n" for name in files)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


class RoboticsSetupTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "setup_robotics.py").is_file(),
                        "Robotics setup must use its own isolated runtime")
        self.setup = importlib.import_module("setup_robotics")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / ".robotics-runtime"
        self.requirements = self.root / "requirements.txt"
        self.requirements.write_text("robotics-probe==1.0\n")
        self.wheels = self.root / "wheels"
        self.wheels.mkdir()
        wheel(self.wheels)
        for context in (
            patch.dict(os.environ, {"PIP_NO_INDEX": "1", "PIP_FIND_LINKS": str(self.wheels)}),
            patch.object(self.setup, "SMOKE_IMPORTS", {"robotics_probe": ("READY",)}),
        ):
            context.start()
            self.addCleanup(context.stop)

    def install(self):
        return self.setup.install_runtime(self.runtime, self.requirements)

    def status(self):
        return self.setup.runtime_status(self.runtime, self.requirements)

    def test_install_shares_torch_and_does_not_modify_the_app(self):
        before = {item.metadata["Name"]: item.version for item in metadata.distributions()}
        self.install()
        self.assertEqual(before, {item.metadata["Name"]: item.version
                                  for item in metadata.distributions()})
        self.assertFalse((self.runtime / "torch").exists())
        self.assertFalse((self.runtime / "nvidia").exists())
        self.assertTrue(self.status()["available"])
        self.assertNotIn("robotics_probe", sys.modules)
        self.assertEqual(self.status()["versions"]["torch"], before["torch"])

    def test_reuse_needs_no_network_and_no_import_in_server_process(self):
        self.install()
        marker = (self.runtime / ".ready.json").read_bytes()
        shutil.rmtree(self.wheels)
        self.install()
        self.assertEqual(marker, (self.runtime / ".ready.json").read_bytes())
        with patch.object(subprocess, "run", side_effect=AssertionError("status spawned child")):
            self.assertTrue(self.status()["available"])
        self.assertNotIn("robotics_probe", sys.modules)

    def test_dependency_conflict_preserves_the_previous_working_runtime(self):
        self.install()
        marker = (self.runtime / ".ready.json").read_bytes()
        self.requirements.write_text("robotics-probe==2.0\n")
        wheel(self.wheels, "2.0", ("torch<1",))
        with self.assertRaisesRegex(RuntimeError, "torch<1"):
            self.install()
        self.assertEqual(marker, (self.runtime / ".ready.json").read_bytes())
        self.assertEqual((self.runtime / "robotics_probe.py").read_text(), "READY = '1.0'\n")
        self.assertFalse(self.status()["available"])

    def test_pins_cannot_install_shared_torch_cuda_or_unpinned_packages(self):
        for requirement in ("torch==2.11.0", "torchvision==0.26.0", "triton==3.6.0",
                            "nvidia-cublas-cu12==12.8.4.1", "cuda-bindings==12.9.7",
                            "robotics-probe>=1"):
            with self.subTest(requirement=requirement):
                self.requirements.write_text(requirement + "\n")
                with self.assertRaises(ValueError):
                    self.install()
                self.assertFalse(self.runtime.exists())

    def test_only_checksum_pinned_official_lerobot_source_is_allowed(self):
        commit='a'*40;digest='b'*64
        url=f'https://codeload.github.com/huggingface/lerobot/tar.gz/{commit}#sha256={digest}'
        self.requirements.write_text('# Official source\nlerobot[smolvla] @ '+url+' # pinned archive\n')
        pins=self.setup._pins(self.requirements)
        self.assertEqual(pins[0].url,url)
        for requirement in [
            f'lerobot @ https://codeload.github.com/huggingface/lerobot/tar.gz/main#sha256={digest}',
            f'lerobot @ https://codeload.github.com/huggingface/lerobot/tar.gz/{commit}',
            f'lerobot @ https://example.com/lerobot/{commit}#sha256={digest}',
            'lerobot @ git+https://github.com/huggingface/lerobot.git@main',
            'robotics-probe @ '+url]:
            self.requirements.write_text(requirement+'\n')
            with self.assertRaisesRegex(ValueError,'source|pin|version'):
                self.setup._pins(self.requirements)

    def test_source_pin_checks_installed_archive_identity(self):
        from types import SimpleNamespace
        from packaging.requirements import Requirement
        url='https://codeload.github.com/huggingface/lerobot/tar.gz/'+'a'*40
        requirement=Requirement('lerobot @ '+url+'#sha256='+'b'*64)
        good=dict(url=url,archive_info=dict(hashes=dict(sha256='b'*64)))
        installed=SimpleNamespace(read_text=lambda name:json.dumps(good))
        self.setup._validate_source_identity(requirement,installed)
        for record in [None,{},dict(good,url='https://example.com/other'),
                       dict(good,archive_info=dict(hashes=dict(sha256='c'*64)))]:
            installed=SimpleNamespace(read_text=lambda name:json.dumps(record) if record else None)
            with self.assertRaisesRegex(RuntimeError,'source|archive|identity'):
                self.setup._validate_source_identity(requirement,installed)

    def test_dependency_extras_are_validated_before_publishing(self):
        self.requirements.write_text("robotics-probe[camera]==1.0\n")
        wheel(self.wheels, dependencies=("missing-camera-package>=1; extra == 'camera'",))
        with self.assertRaisesRegex(RuntimeError, "missing-camera-package"):
            self.install()
        self.assertFalse(self.runtime.exists())

    def test_pythonpath_poisoning_cannot_enter_worker_smoke_checks(self):
        external = self.root / "external"
        external.mkdir()
        (external / "robotics_probe.py").write_text("raise RuntimeError('external module loaded')\n")
        with patch.dict(os.environ, {"PYTHONPATH": str(external)}):
            self.install()
        self.assertTrue(self.status()["available"])

    def test_smoke_failure_never_publishes_a_ready_marker(self):
        wheel(self.wheels, source="raise RuntimeError('native dependency failed')\n")
        with self.assertRaisesRegex(RuntimeError, "native dependency failed"):
            self.install()
        self.assertFalse(self.runtime.exists())

    def test_changed_shared_torch_version_invalidates_ready_status(self):
        self.install()
        original = self.setup._shared_versions
        with patch.object(self.setup, "_shared_versions", side_effect=lambda: {
            **original(), "torch": "2.12.0"}):
            self.assertFalse(self.status()["available"])

    def test_missing_installed_distribution_is_not_reported_as_ready(self):
        self.install()
        shutil.rmtree(self.runtime / "robotics_probe-1.0.dist-info")
        self.assertFalse(self.status()["available"])

    def test_unsupported_python_fails_before_installation(self):
        with patch.object(self.setup.sys, "version_info", (3, 11, 9)):
            with self.assertRaisesRegex(RuntimeError, "Python 3.12"):
                self.install()
        self.assertFalse(self.runtime.exists())

    def test_check_refuses_a_torch_distribution_inside_overlay(self):
        self.install()
        info = self.runtime / "torch-2.11.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Name: torch\nVersion: 2.11.0\n")
        with self.assertRaisesRegex(RuntimeError, "shared|Torch"):
            self.setup.check_runtime(self.runtime, self.requirements, smoke=False)

    def test_bootstrap_inserts_overlay_before_imports_and_preserves_shared_torch(self):
        self.install()
        code = """
import json, sys
from pathlib import Path
import setup_robotics
setup_robotics.bootstrap_runtime(Path(sys.argv[1]), Path(sys.argv[2]))
import robotics_probe
from importlib import metadata
print(json.dumps({'ready': robotics_probe.READY, 'origin': robotics_probe.__file__,
                  'torch': str(metadata.distribution('torch').locate_file('torch'))}))
"""
        child = subprocess.run([sys.executable, "-E", "-c", code,
                                str(self.runtime), str(self.requirements)],
                               cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)
        result = json.loads(child.stdout)
        self.assertEqual(result["ready"], "1.0")
        self.assertTrue(result["origin"].startswith(str(self.runtime)))
        self.assertFalse(result["torch"].startswith(str(self.runtime)))

    def test_missing_runtime_cli_explains_how_to_install_without_writing(self):
        shutil.copy2(ROOT / "setup_robotics.py", self.root / "setup_robotics.py")
        shutil.copy2(self.requirements, self.root / "requirements-robotics.txt")
        child = subprocess.run([sys.executable, "-E", str(self.root / "setup_robotics.py"),
                                "--check"], capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 1, child.stdout + child.stderr)
        self.assertIn(".venv/bin/python -E setup_robotics.py", child.stdout + child.stderr)
        self.assertFalse(self.runtime.exists())


if __name__ == "__main__":
    unittest.main()
