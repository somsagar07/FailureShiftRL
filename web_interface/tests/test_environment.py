"""Setup decisions stay testable without package downloads or a physical GPU."""
import unittest
import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import sys
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


class TorchvisionSetupTests(unittest.TestCase):
    def plan(self, current, vision=None, platform='linux'):
        planner = getattr(check_environment, 'plan_torchvision_install', None)
        self.assertTrue(callable(planner), 'Torchvision needs a plan tied to the selected Torch build')
        with patch.object(check_environment.sys, 'platform', platform):
            return planner(current, vision)

    def test_missing_vision_uses_actual_torch_index_and_pins_torch_for_resolver(self):
        for release, cuda, index in [('2.11.0+cu128', '12.8', 'cu128'),
                                     ('2.8.0+cu126', '12.6', 'cu126'),
                                     ('2.6.0+cu124', '12.4', 'cu124'),
                                     ('2.7.1+cu118', '11.8', 'cu118'),
                                     ('2.11.0+cpu', None, 'cpu')]:
            with self.subTest(release=release):
                plan = self.plan({'version': release, 'cuda': cuda, 'available': False})
                self.assertTrue(plan['install'])
                self.assertEqual(plan['index_url'], f'https://download.pytorch.org/whl/{index}')
                self.assertEqual(plan['torch_requirement'], f'torch=={release}')
                self.assertEqual(plan['requirement'], 'torchvision>=0.21,<1')

    def test_working_vision_is_kept_only_when_torch_release_and_build_match(self):
        current = {'version': '2.8.0+cu126', 'cuda': '12.6', 'available': True}
        vision = {'version': '0.23.0+cu126', 'torch_version': '2.8.0', 'available': True}
        self.assertFalse(self.plan(current, vision)['install'])
        self.assertTrue(self.plan(current, {**vision, 'torch_version': '2.7.1'})['install'])
        wrong_build = self.plan(current, {**vision, 'version': '0.23.0+cpu'})
        self.assertTrue(wrong_build['install'])
        self.assertEqual(wrong_build['requirement'], 'torchvision==0.23.0+cu126')

    def test_explicit_cpu_vision_target_cannot_retain_a_cuda_wheel(self):
        plan = self.plan({'version': '2.11.0+cpu', 'cuda': None, 'available': False},
                         {'version': '0.26.0+cu128', 'torch_version': '2.11.0', 'available': True})
        self.assertTrue(plan['install'])
        self.assertEqual(plan['requirement'], 'torchvision==0.26.0+cpu')
        self.assertEqual(plan['torch_requirement'], 'torch==2.11.0+cpu')

    def test_broken_matching_vision_is_reinstalled_without_reinstalling_torch(self):
        plan = self.plan({'version': '2.11.0+cu128', 'cuda': '12.8', 'available': True},
                         {'version': '0.26.0+cu128', 'torch_version': '2.11.0', 'available': False})
        self.assertTrue(plan['install'])
        self.assertTrue(plan['reinstall'])
        self.assertEqual(plan['requirement'], 'torchvision==0.26.0+cu128')

    def test_macos_matching_wheels_need_no_local_cpu_suffix(self):
        plan = self.plan({'version': '2.11.0', 'cuda': None, 'available': False},
                         {'version': '0.26.0', 'torch_version': '2.11.0', 'available': True}, platform='darwin')
        self.assertFalse(plan['install'])

    def test_pypi_metadata_preserves_the_installed_cuda_build_and_distribution_pin(self):
        probe = getattr(check_environment, 'installed_torchvision', None)
        self.assertTrue(callable(probe))
        modules = {'torchvision': SimpleNamespace(__version__='0.23.0+cu128'),
                   'torchvision.transforms.v2': SimpleNamespace(ToImage=object())}
        with patch.object(check_environment, 'version', return_value='0.23.0'), \
             patch.object(check_environment, 'package_requirements', return_value=['torch (==2.8.0)']), \
             patch.object(check_environment.importlib, 'import_module', side_effect=modules.__getitem__):
            current = probe()
        plan = self.plan({'version': '2.8.0+cu128', 'package_version': '2.8.0', 'cuda': '12.8'}, current)
        self.assertFalse(plan['install'], 'A compatible PyPI wheel must be retained')
        self.assertEqual(plan['requirement'], 'torchvision==0.23.0')
        self.assertEqual(plan['torch_requirement'], 'torch==2.8.0')

    def test_unknown_non_cuda_build_is_not_silently_replaced_by_cpu_vision(self):
        with self.assertRaisesRegex(RuntimeError, 'build|wheel'):
            self.plan({'version': '2.8.0+rocm6.4', 'cuda': None, 'available': False})

    def test_vision_resolver_and_general_requirements_cannot_replace_working_torch(self):
        source = Path(__file__).resolve().parents[1]
        for reinstall in (False, True):
            with self.subTest(reinstall=reinstall), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                shutil.copy2(source / 'setup.sh', root / 'setup.sh')
                (root / 'requirements.txt').write_text('torch>=2.6,<3\ntorchvision>=0.21,<1\n')
                (root / '.venv' / 'bin').mkdir(parents=True)
                shim = root / '.venv' / 'bin' / 'python'
                action = 'reinstall' if reinstall else 'install'
                requirement = 'torchvision==0.23.0+cu126' if reinstall else 'torchvision>=0.21,<1'
                vision_plan = '\t'.join((action, requirement, 'https://download.pytorch.org/whl/cu126', 'torch==2.8.0+cu126'))
                setup_plan = 'cuda\tkeep\t-\t-\t' + vision_plan
                shim.write_text(f'''#!{sys.executable}
import json, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(__file__).resolve().parents[2]
if '--setup-plan' in args:
    print({setup_plan!r})
elif '--vision-plan' in args:
    print({vision_plan!r})
elif '--installed-constraints' in args:
    print('torch==2.8.0+cu126\\ntorchvision==0.23.0+cu126')
elif 'pip' in args and 'install' in args:
    entry = {{'args': args}}
    if '--constraint' in args:
        entry['constraints'] = pathlib.Path(args[args.index('--constraint') + 1]).read_text()
    with (root / 'pip-calls.jsonl').open('a') as log:
        log.write(json.dumps(entry) + '\\n')
''')
                shim.chmod(0o755)
                result = subprocess.run(['bash', str(root / 'setup.sh')],
                                        env={**os.environ, 'TMPDIR': temporary}, capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = [json.loads(line) for line in (root / 'pip-calls.jsonl').read_text().splitlines()]
                self.assertEqual(len(calls), 2, 'Setup must install vision from its official index before app requirements')
                self.assertIn('https://download.pytorch.org/whl/cu126', calls[0]['args'])
                self.assertIn(requirement, calls[0]['args'])
                if reinstall:
                    self.assertIn('--no-deps', calls[0]['args'])
                    self.assertIn('--force-reinstall', calls[0]['args'])
                    self.assertNotIn('torch==2.8.0+cu126', calls[0]['args'])
                else:
                    self.assertIn('torch==2.8.0+cu126', calls[0]['args'])
                self.assertIn('torch==2.8.0+cu126', calls[1].get('constraints', ''))
                self.assertIn('torchvision==0.23.0+cu126', calls[1].get('constraints', ''))


class GenerationEnvironmentTests(unittest.TestCase):
    def check(self, *, missing_module=None, missing_export=None, transformers_version='4.57.6'):
        class Exports:
            def __getattr__(self, name):
                if name == missing_export:
                    raise ImportError(f'{name} is unavailable')
                return object()
        def load(name):
            if name == missing_module:
                raise ImportError(f'{name} is unavailable')
            return Exports()
        versions = {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128', 'transformers': transformers_version}
        output = io.StringIO()
        with patch.object(check_environment.importlib, 'import_module', side_effect=load), \
             patch.object(check_environment, 'version', side_effect=lambda name: versions.get(name, '1.0.0')), \
             patch.object(check_environment, 'check_torch_runtime', return_value={'backend': 'cpu', 'cuda': None}), \
             patch.object(check_environment, 'nvidia_drivers', return_value=[]), contextlib.redirect_stdout(output):
            result = check_environment.check_environment()
        return result, output.getvalue()

    def test_generation_backends_require_importable_vision_and_processor_exports(self):
        for fields in [{'missing_module': 'torchvision'}, {'missing_export': 'Qwen3VLProcessor'},
                       {'missing_export': 'Qwen3VLForConditionalGeneration'}, {'missing_export': 'BlipProcessor'},
                       {'missing_export': 'BlipForQuestionAnswering'}]:
            with self.subTest(fields=fields):
                result, output = self.check(**fields)
                self.assertEqual(result, 1)
                self.assertIn('MISSING OR BROKEN', output)
                self.assertIn(next(iter(fields.values())), output)

    def test_transformers_456_is_rejected_before_a_generation_run(self):
        result, output = self.check(transformers_version='4.56.2')
        self.assertEqual(result, 1)
        self.assertIn('4.57', output)

    def test_importable_generation_backends_pass(self):
        self.assertEqual(self.check()[0], 0)


if __name__ == '__main__':
    unittest.main()
