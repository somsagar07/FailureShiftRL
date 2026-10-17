"""Setup decisions stay testable without package downloads or a physical GPU."""
import unittest
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from packaging.specifiers import SpecifierSet

import check_environment


class TorchSetupTests(unittest.TestCase):
    def plan(self, mode='auto', current=None, drivers=(), capabilities=('8.9',), platform='linux'):
        planner = getattr(check_environment, 'plan_torch_install', None)
        self.assertTrue(callable(planner), 'setup needs a hardware-aware installation plan')
        with patch.object(check_environment, 'nvidia_capabilities', create=True,
                          return_value=list(capabilities)), \
             patch.object(check_environment.sys, 'platform', platform):
            return planner(mode, current, list(drivers))

    def test_auto_replaces_newer_cpu_build_on_an_nvidia_machine(self):
        plan = self.plan(current={'version': '2.14.1+cpu', 'cuda': None, 'available': False},
                         drivers=['580.178.04'])
        self.assertEqual(plan['backend'], 'cuda')
        self.assertTrue(plan['install'])
        self.assertIn('+cu128', plan['requirement'])
        self.assertEqual(plan['index_url'], 'https://download.pytorch.org/whl/cu128')

    def test_fresh_cpu_machine_uses_cpu_wheels(self):
        plan = self.plan()
        self.assertEqual(plan['backend'], 'cpu')
        self.assertTrue(plan['install'])
        self.assertEqual(plan['requirement'], 'torch==2.11.0+cpu')
        self.assertEqual(plan['index_url'], 'https://download.pytorch.org/whl/cpu')

    def test_macos_cpu_requirement_accepts_the_wheel_without_local_cpu_suffix(self):
        plan = self.plan(platform='darwin')
        self.assertTrue(SpecifierSet(plan['requirement'].removeprefix('torch')).contains('2.11.0'))
        self.assertEqual(plan['index_url'], 'https://download.pytorch.org/whl/cpu')

    def test_explicit_cpu_replaces_cuda_even_with_an_nvidia_gpu(self):
        plan = self.plan('cpu', {'version': '2.11.0+cu128', 'cuda': '12.8', 'available': True},
                         ['580.178.04'])
        self.assertEqual(plan['backend'], 'cpu')
        self.assertTrue(plan['install'])
        self.assertFalse(SpecifierSet(plan['requirement'].removeprefix('torch')).contains('2.11.0+cu128'))

    def test_working_cuda_is_preserved_even_without_nvidia_smi(self):
        for mode in ('auto', 'cuda'):
            with self.subTest(mode=mode):
                plan = self.plan(mode, {'version': '2.8.0+cu126', 'cuda': '12.6', 'available': True})
                self.assertEqual(plan['backend'], 'cuda')
                self.assertFalse(plan['install'])

    def test_supported_cpu_is_preserved_on_cpu_machine(self):
        plan = self.plan(current={'version': '2.14.1+cpu', 'cuda': None, 'available': False})
        self.assertEqual(plan['backend'], 'cpu')
        self.assertFalse(plan['install'])

    def test_older_driver_gets_a_compatible_cuda_index(self):
        for driver, expected in [('560.35.03', 'cu126'), ('535.183.01', 'cu118')]:
            with self.subTest(driver=driver):
                plan = self.plan(drivers=[driver])
                self.assertEqual(plan['backend'], 'cuda')
                self.assertTrue(plan['install'])
                self.assertTrue(plan['index_url'].endswith('/'+expected))

    def test_legacy_gpu_uses_cuda_126_even_with_a_new_driver(self):
        for capability in ('5.0', '5.2', '6.0', '6.1', '7.0'):
            with self.subTest(capability=capability):
                plan = self.plan(drivers=['580.178.04'], capabilities=[capability])
                self.assertEqual(plan['requirement'], 'torch==2.11.0+cu126')

    def test_blackwell_requires_cuda_128_and_a_sufficient_driver(self):
        for capability in ('10.0', '12.0'):
            with self.subTest(capability=capability):
                plan = self.plan(drivers=['580.178.04'], capabilities=[capability])
                self.assertEqual(plan['requirement'], 'torch==2.11.0+cu128')
                with self.assertRaisesRegex(RuntimeError, 'driver'):
                    self.plan(drivers=['560.35.03'], capabilities=[capability])

    def test_no_single_supported_wheel_for_legacy_and_blackwell_gpus(self):
        with self.assertRaisesRegex(RuntimeError, 'single.*wheel'):
            self.plan(drivers=['580.178.04', '580.178.04'], capabilities=['6.1', '12.0'])

    def test_unknown_or_unsupported_architecture_does_not_install_a_guess(self):
        for capabilities in ([], ['N/A'], ['3.7'], ['13.0']):
            with self.subTest(capabilities=capabilities), self.assertRaisesRegex(RuntimeError, 'capabilit|architecture'):
                self.plan(drivers=['580.178.04'], capabilities=capabilities)

    def test_cuda_discovery_without_working_kernels_does_not_keep_the_build(self):
        torch = SimpleNamespace(__version__='2.11.0+cu128', version=SimpleNamespace(cuda='12.8'),
                                cuda=SimpleNamespace(is_available=lambda: True))
        with patch.object(check_environment.importlib, 'import_module', return_value=torch), \
             patch.object(check_environment, 'check_torch_runtime',
                          side_effect=RuntimeError('no kernel image is available for execution on the device')):
            current = check_environment.installed_torch()
        self.assertIsNotNone(current)
        self.assertFalse(current['available'])
        plan = self.plan(current=current, drivers=['580.178.04'], capabilities=['6.1'])
        self.assertTrue(plan['install'])
        self.assertEqual(plan['requirement'], 'torch==2.11.0+cu126')

    def test_capability_probe_falls_back_to_torch_if_nvidia_smi_cannot_query_it(self):
        probe = getattr(check_environment, 'nvidia_capabilities', None)
        self.assertTrue(callable(probe), 'setup needs a compute capability probe')
        torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 2,
            get_device_capability=lambda index: [(6, 1), (8, 9)][index]))
        with patch.object(check_environment.subprocess, 'run', side_effect=OSError('query not supported')), \
             patch.object(check_environment.importlib, 'import_module', return_value=torch):
            self.assertEqual(probe(), ['6.1', '8.9'])

    def test_explicit_cuda_does_not_fall_back_to_cpu(self):
        with self.assertRaisesRegex(RuntimeError, 'NVIDIA'):
            self.plan('cuda')

    def test_detected_but_unsupported_driver_is_not_silently_cpu(self):
        with self.assertRaisesRegex(RuntimeError, 'driver'):
            self.plan(drivers=['470.256.02'])

    def test_unsupported_torch_version_is_replaced(self):
        plan = self.plan(current={'version': '2.5.1+cpu', 'cuda': None, 'available': False})
        self.assertTrue(plan['install'])

    def test_required_cuda_fails_for_cpu_only_torch(self):
        checker = getattr(check_environment, 'check_torch_runtime', None)
        self.assertTrue(callable(checker), 'the environment check must enforce required CUDA')
        torch = SimpleNamespace(version=SimpleNamespace(cuda=None),
                                cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaisesRegex(RuntimeError, 'CPU-only'):
            checker(torch, require_cuda=True)
        self.assertEqual(checker(torch)['backend'], 'cpu')

    def test_setup_rejects_conflicting_backend_options_before_any_install(self):
        script = Path(__file__).resolve().parents[1] / 'setup.sh'
        for options in [('--auto', '--cpu'), ('--cpu', '--cuda'), ('--cuda', '--auto')]:
            with self.subTest(options=options):
                result = subprocess.run(['bash', str(script), *options, '--help'],
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Choose only one', result.stderr)

    def test_fresh_setup_dry_run_detects_hardware_without_creating_a_venv(self):
        source = Path(__file__).resolve().parents[1]
        for driver, capability, options, expected_status, expected in [
            ('', '', [], 0, 'from https://download.pytorch.org/whl/cpu'),
            ('580.178.04', '8.9', [], 0, '+cu128 from https://download.pytorch.org/whl/cu128'),
            ('580.178.04', '6.1', [], 0, '+cu126 from https://download.pytorch.org/whl/cu126'),
            ('580.178.04', '7.0', [], 0, '+cu126 from https://download.pytorch.org/whl/cu126'),
            ('580.178.04', '12.0', [], 0, '+cu128 from https://download.pytorch.org/whl/cu128'),
            ('580.178.04', '3.7', [], 1, 'architecture'),
            ('580.178.04', '8.9', ['--cpu'], 0, 'from https://download.pytorch.org/whl/cpu'),
            ('', '', ['--cuda'], 1, 'no usable NVIDIA GPU/driver'),
        ]:
            with self.subTest(driver=driver, capability=capability, options=options), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in ('setup.sh', 'check_environment.py'):
                    shutil.copy2(source / name, root / name)
                shim = root / 'nvidia-smi'
                shim.write_text('#!/bin/sh\n' + (
                    'case "$1" in\n'
                    f"  --query-gpu=driver_version) printf '%s\\n' '{driver}' ;;\n"
                    f"  --query-gpu=compute_cap) printf '%s\\n' '{capability}' ;;\n"
                    '  *) exit 1 ;;\n'
                    'esac\n' if driver else 'exit 1\n'))
                shim.chmod(0o755)
                environment = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH']}
                result = subprocess.run(['bash', str(root / 'setup.sh'), *options, '--dry-run'],
                                        env=environment, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, expected_status, result.stdout + result.stderr)
                self.assertIn(expected, result.stdout + result.stderr)
                self.assertFalse((root / '.venv').exists())


if __name__ == '__main__':
    unittest.main()
