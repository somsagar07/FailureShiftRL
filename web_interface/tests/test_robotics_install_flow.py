"""Exercise robotics setup orchestration without touching the real environment."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RoboticsInstallFlowTests(unittest.TestCase):
    def test_setup_installs_robotics_after_app_and_checks_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(ROOT / 'setup.sh', root / 'setup.sh')
            shim = root / '.venv/bin/python'
            shim.parent.mkdir(parents=True)
            shim.write_text(f'''#!{sys.executable}
import pathlib,sys
root=pathlib.Path(__file__).resolve().parents[2]
args=sys.argv[1:]
if '--setup-plan' in args:
 print('cpu\\tkeep\\t-\\t-\\tkeep\\t-\\t-\\ttorch==2.11.0+cpu')
elif '--vision-plan' in args:
 print('keep\\t-\\t-\\ttorch==2.11.0+cpu')
elif '--installed-constraints' in args:
 print('torch==2.11.0+cpu')
elif 'pip' in args and 'install' in args:
 (root/'app-installed').touch()
elif 'setup_vev.py' in args:
 assert (root/'app-installed').exists()
 (root/'vev-installed').touch()
elif 'setup_robotics.py' in args:
 assert (root/'app-installed').exists() and (root/'vev-installed').exists()
 assert '--render' not in args, 'Package installation must not require a graphics driver'
 (root/'robotics-installed').touch()
elif 'check_environment.py' in args:
 assert (root/'robotics-installed').exists(), 'Final readiness check ran before robotics setup'
''')
            shim.chmod(0o755)
            result = subprocess.run(['bash', str(root / 'setup.sh')], capture_output=True, text=True,
                                    env={**os.environ, 'TMPDIR': directory}, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / 'robotics-installed').exists())

    def test_launcher_environment_check_detects_missing_robotics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(ROOT / 'check_environment.py', root / 'check_environment.py')
            (root / 'setup_vev.py').write_text('raise SystemExit(0)\n')
            shutil.copy2(ROOT / 'setup_robotics.py', root / 'setup_robotics.py')
            shutil.copy2(ROOT / 'requirements-robotics.txt', root / 'requirements-robotics.txt')
            result = subprocess.run([sys.executable, '-I', str(root / 'check_environment.py')],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('Robotics runtime is missing', result.stdout + result.stderr)
            self.assertIn('setup_robotics.py', result.stdout + result.stderr)
