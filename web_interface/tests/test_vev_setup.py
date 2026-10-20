"""Exercise the installer with local wheels and no network or model downloads."""
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


def wheel(directory, release='1.0', dependencies=('torch>=2.6', 'packaging>=20')):
    """A real wheel lets pip exercise target isolation without external downloads."""
    path = directory / f'vev_probe-{release}-py3-none-any.whl'
    info = f'vev_probe-{release}.dist-info'
    contents = {
        'vev_probe.py': f"READY = {release!r}\n",
        f'{info}/METADATA': 'Metadata-Version: 2.1\nName: vev-probe\n'
                           f'Version: {release}\n' +
                           ''.join(f'Requires-Dist: {item}\n' for item in dependencies),
        f'{info}/WHEEL': 'Wheel-Version: 1.0\nGenerator: tests\n'
                        'Root-Is-Purelib: true\nTag: py3-none-any\n',
    }
    contents[f'{info}/RECORD'] = ''.join(f'{name},,\n' for name in contents)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    return path


class VevSetupTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'setup_vev.py').exists(),
                        'Vev needs an isolated runtime installer')
        self.setup = importlib.import_module('setup_vev')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / '.vev-runtime'
        self.requirements = self.root / 'requirements.txt'
        self.requirements.write_text('vev-probe==1.0\n')
        self.wheels = self.root / 'wheels'
        self.wheels.mkdir()
        wheel(self.wheels)
        self.environment = patch.dict(os.environ, {
            'PIP_NO_INDEX': '1', 'PIP_FIND_LINKS': str(self.wheels),
            'CUDA_VISIBLE_DEVICES': '',
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        imports = patch.object(self.setup, 'SMOKE_IMPORTS', {'vev_probe': ('READY',)})
        imports.start()
        self.addCleanup(imports.stop)

    def install(self):
        return self.setup.install_runtime(self.runtime, self.requirements)

    def test_offline_install_shares_base_torch_without_changing_the_app(self):
        before = {item.metadata['Name']: item.version for item in metadata.distributions()}
        self.install()
        after = {item.metadata['Name']: item.version for item in metadata.distributions()}
        self.assertEqual(after, before)
        self.assertFalse((self.runtime / 'torch').exists())
        self.assertFalse((self.runtime / 'packaging').exists())
        self.assertEqual(metadata.distribution('torch').version, before['torch'])
        installed = list(metadata.distributions(path=[str(self.runtime)]))
        self.assertEqual([(item.metadata['Name'], item.version) for item in installed],
                         [('vev-probe', '1.0')])
        self.assertTrue((self.runtime / '.ready.json').is_file())
        self.assertNotIn('vev_probe', sys.modules)

    def test_ready_runtime_can_be_reused_without_the_wheel_repository(self):
        self.install()
        marker = (self.runtime / '.ready.json').read_bytes()
        shutil.rmtree(self.wheels)
        self.install()
        self.assertEqual((self.runtime / '.ready.json').read_bytes(), marker)

    def test_matching_base_versions_are_shared_with_normalized_package_names(self):
        self.requirements.write_text('vev-probe==1.0\nPackaging==' + metadata.version('packaging') + '\n')
        self.install()
        self.assertFalse((self.runtime / 'packaging').exists())

    def test_failed_dependency_validation_preserves_the_previous_runtime(self):
        self.install()
        old_marker = (self.runtime / '.ready.json').read_bytes()
        self.requirements.write_text('vev-probe==2.0\n')
        wheel(self.wheels, '2.0', ('unavailable-vev-test-dependency>=1',))
        with self.assertRaisesRegex(RuntimeError, 'unavailable-vev-test-dependency'):
            self.install()
        self.assertEqual((self.runtime / '.ready.json').read_bytes(), old_marker)
        self.assertEqual((self.runtime / 'vev_probe.py').read_text(), "READY = '1.0'\n")
        self.assertEqual(list(self.root.glob('.vev-runtime-build-*')), [])

    def test_worker_import_check_ignores_an_external_pythonpath(self):
        poisoned = self.root / 'external'
        poisoned.mkdir()
        (poisoned / 'vev_probe.py').write_text("raise RuntimeError('external project imported')\n")
        with patch.dict(os.environ, {'PYTHONPATH': str(poisoned)}):
            self.install()
        self.assertTrue((self.runtime / '.ready.json').is_file())

    def test_import_failure_does_not_publish_a_ready_runtime(self):
        with patch.object(self.setup, 'SMOKE_IMPORTS', {'vev_probe': ('MISSING',)}):
            with self.assertRaisesRegex(RuntimeError, 'MISSING'):
                self.install()
        self.assertFalse(self.runtime.exists())

    def test_installer_refuses_to_target_shared_torch_or_cuda_packages(self):
        for name in ('torch', 'torchvision', 'triton', 'nvidia-cublas-cu12'):
            with self.subTest(name=name):
                self.requirements.write_text(f'{name}==1.0\n')
                with self.assertRaisesRegex(ValueError, 'shared|Torch|CUDA'):
                    self.install()
                self.assertFalse(self.runtime.exists())

    def test_unsupported_python_stops_before_installing_any_package(self):
        with patch.object(self.setup.sys, 'version_info', (3, 11, 9)):
            with self.assertRaisesRegex(RuntimeError, r'Python 3\.12'):
                self.install()
        self.assertFalse(self.runtime.exists())

    def test_missing_runtime_check_reports_a_repair_command(self):
        shutil.copy2(ROOT / 'setup_vev.py', self.root / 'setup_vev.py')
        shutil.copy2(self.requirements, self.root / 'requirements-vev.txt')
        result = subprocess.run([sys.executable, '-E', str(self.root / 'setup_vev.py'), '--check'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('setup_vev.py', result.stdout + result.stderr)
        self.assertIn('.venv/bin/python', result.stdout + result.stderr)
        self.assertFalse(self.runtime.exists())


class VevSetupIntegrationTests(unittest.TestCase):
    def test_launcher_check_detects_missing_vev_with_a_healthy_app_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('check_environment.py', 'setup_vev.py', 'requirements-vev.txt'):
                self.assertTrue((ROOT / name).exists(), 'Vev setup files must be available')
                shutil.copy2(ROOT / name, root / name)
            result = subprocess.run([sys.executable, '-I', str(root / 'check_environment.py')],
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('Vev runtime is missing', result.stdout + result.stderr)
            self.assertIn('setup_vev.py', result.stdout + result.stderr)

    def test_setup_installs_app_before_the_isolated_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(ROOT / 'setup.sh', root / 'setup.sh')
            (root / '.venv' / 'bin').mkdir(parents=True)
            shim = root / '.venv' / 'bin' / 'python'
            shim.write_text(f'''#!{sys.executable}
import pathlib, sys
root = pathlib.Path(__file__).resolve().parents[2]
args = sys.argv[1:]
if '--setup-plan' in args:
    print('cpu\\tkeep\\t-\\t-\\tkeep\\t-\\t-\\ttorch==2.11.0+cpu')
elif '--vision-plan' in args:
    print('keep\\t-\\t-\\ttorch==2.11.0+cpu')
elif '--installed-constraints' in args:
    print('torch==2.11.0+cpu')
elif 'pip' in args and 'install' in args:
    (root / 'app-installed').write_text('ready')
elif 'setup_vev.py' in args:
    assert (root / 'app-installed').exists(), 'Vev was installed before app requirements'
    (root / '.vev-runtime').mkdir()
elif 'check_environment.py' in args:
    assert (root / '.vev-runtime').is_dir(), 'Final check did not include Vev'
''')
            shim.chmod(0o755)
            result = subprocess.run(['bash', str(root / 'setup.sh')],
                                    env={**os.environ, 'TMPDIR': temporary},
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / '.vev-runtime').is_dir())


if __name__ == '__main__':
    unittest.main()
