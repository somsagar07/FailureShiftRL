"""Exercise the persistent judge boundary with real pipes and child processes."""
import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image


WORKER = r'''
import base64, io, json, os, signal, sys, time
from pathlib import Path
from PIL import Image
for line in sys.stdin:
    row = json.loads(line)
    if row['command'] == 'load':
        staging = [os.environ[name] for name in ('VEV_STAGING_DIR','VEV_COMPACT_STAGING_DIR')]
        for path in staging:
            (Path(path) / 'partial-download').write_text('temporary')
        xet = Path(os.environ['HF_XET_CACHE'])
        xet.mkdir()
        (xet / 'download-chunk').write_text('temporary chunk')
        print(json.dumps({'ok': True, 'identity': {
            'name': row['selection']['name'], 'architecture': 'vev',
            'revision': row['selection']['revision'], 'pid': os.getpid(),
            'staging': staging, 'xet':str(xet)}}), flush=True)
    elif row['command'] == 'decide':
        questions = row['questions']
        image = Image.open(io.BytesIO(base64.b64decode(row['image'])))
        image.load()
        mode = questions[0]['question']
        if mode == 'ignore':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if mode in ('wait', 'ignore'):
            time.sleep(20)
        if mode == 'crash':
            print('fixture worker crash', file=sys.stderr, flush=True)
            sys.exit(7)
        if mode == 'invalid':
            print('unstructured output', flush=True)
            continue
        if mode == 'error':
            print(json.dumps({'ok': False, 'error': 'Question too long', 'error_type': 'ValueError'}), flush=True)
            continue
        if mode == 'logs':
            sys.stderr.write('log ' * 30000)
            sys.stderr.flush()
        # Lossless PNG roundtrip: the red pixel is the positive probability.
        value = image.convert('RGB').getpixel((0, 0))[0] / 255.
        answers = [{'id': q['id'], 'type': 'noul',
                    'probabilities': [1-value, value], 'value': value} for q in questions]
        print(json.dumps({'ok': True, 'answers': answers}), flush=True)
    elif row['command'] == 'close':
        break
'''


class VevJudgeTests(unittest.TestCase):
    def setUp(self):
        from vev_judge import VevJudge
        self.VevJudge = VevJudge
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        worker = Path(self.temp.name) / 'worker.py'
        worker.write_text(WORKER)
        command = patch('vev_judge._worker_command', return_value=[sys.executable, '-E', str(worker)])
        command.start()
        self.addCleanup(command.stop)
        cache = patch('vev_judge.cache_directory', return_value=Path(self.temp.name)/'cache')
        cache.start()
        self.addCleanup(cache.stop)
        self.selection = {'source': 'huggingface', 'name': 'CountingSheep/vev-4b'}
        self.image = Image.new('RGB', (16, 16), 'red')
        self.judges = []
        self.addCleanup(lambda: [judge.close() for judge in self.judges])

    def judge(self, **options):
        judge = self.VevJudge(self.selection, 'cuda', **options)
        self.judges.append(judge)
        return judge

    @staticmethod
    def questions(text='Is it red?', count=1):
        return [dict(id=f'check:{i}', type='noul', question=text) for i in range(count)]

    def test_load_is_lazy_and_one_worker_handles_multiple_images_and_many_checks(self):
        judge = self.judge()
        self.assertNotIn('pid', judge.identity)
        first = judge.decide(self.image, self.questions(count=128))
        pid = judge.identity['pid']
        self.assertEqual(len(first), 128)
        self.assertEqual(first[-1]['id'], 'check:127')
        self.assertEqual(first[0]['value'], 1.)
        second = judge.decide(Image.new('RGB', (16, 16), 'black'), self.questions())
        self.assertEqual(second[0]['value'], 0.)
        self.assertEqual(judge.identity['pid'], pid)
        self.assertEqual(judge.identity['revision'], '793c449dfb5579655c979f94552176dd41dc597e')
        judge.close()
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_cancellation_reaps_busy_worker_and_next_request_starts_fresh(self):
        judge = self.judge()
        judge.ensure_ready()
        old_pid = judge.identity['pid']
        start = time.monotonic()
        with self.assertRaises(InterruptedError):
            judge.decide(self.image, self.questions('wait'),
                         cancelled=lambda: time.monotonic() - start > .15)
        self.assertLess(time.monotonic() - start, 3.)
        with self.assertRaises(ProcessLookupError):
            os.kill(old_pid, 0)
        judge.decide(self.image, self.questions())
        self.assertNotEqual(judge.identity['pid'], old_pid)

    def test_timeout_terminates_worker_instead_of_hanging_request(self):
        judge = self.judge(timeout=.3)
        judge.ensure_ready()
        pid = judge.identity['pid']
        with self.assertRaisesRegex(TimeoutError, 'Vev'):
            judge.decide(self.image, self.questions('wait'))
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_worker_crash_reports_bounded_stderr_and_recovers(self):
        judge = self.judge()
        with self.assertRaisesRegex(RuntimeError, 'fixture worker crash'):
            judge.decide(self.image, self.questions('crash'))
        self.assertEqual(judge.decide(self.image, self.questions())[0]['value'], 1.)
        self.assertEqual(judge.decide(self.image, self.questions('logs'))[0]['value'], 1.)

    def test_invalid_protocol_and_backend_errors_are_not_scores(self):
        judge = self.judge()
        with self.assertRaisesRegex(RuntimeError, 'response|JSON'):
            judge.decide(self.image, self.questions('invalid'))
        with self.assertRaisesRegex(ValueError, 'Question too long'):
            judge.decide(self.image, self.questions('error'))
        self.assertEqual(judge.decide(self.image, self.questions())[0]['value'], 1.)

    def test_cancelled_and_invalid_requests_never_start_a_worker(self):
        judge = self.judge()
        with self.assertRaises(InterruptedError):
            judge.decide(self.image, self.questions(), cancelled=lambda: True)
        with self.assertRaises(ValueError):
            judge.decide(self.image, [])
        with self.assertRaises(ValueError):
            judge.decide(self.image, self.questions(), cancelled=lambda: 'yes')
        self.assertNotIn('pid', judge.identity)
        for device in ('auto', 'gpu', None):
            with self.subTest(device=device), self.assertRaises(ValueError):
                self.VevJudge(self.selection, device)

    def test_forced_kill_removes_only_this_workers_staging(self):
        judge = self.judge(timeout=.2)
        judge.ensure_ready()
        staging = judge.identity['staging']
        xet = Path(judge.identity['xet'])
        neighbour = Path(staging[0]).parent / (Path(staging[0]).name + '-other')
        neighbour.mkdir()
        self.addCleanup(neighbour.rmdir)
        with self.assertRaises(TimeoutError):
            judge.decide(self.image, self.questions('ignore'))
        self.assertTrue(neighbour.is_dir())
        self.assertTrue(all(not Path(path).exists() for path in staging))
        self.assertFalse(xet.exists())

    def test_evaluator_closes_owned_worker_but_retains_shared_judge(self):
        from discovery_models import Evaluator
        judge = self.judge()
        judge.ensure_ready()
        pid = judge.identity['pid']
        borrowed = Evaluator({'task': 'generation'}, self.temp.name)
        borrowed._shared_resources = True
        borrowed.judge = judge
        borrowed.close()
        os.kill(pid, 0)
        owner = Evaluator({'task': 'generation'}, self.temp.name)
        owner.judge = judge
        owner.close()
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_factory_defaults_to_vev_and_keeps_existing_qwen_adapter(self):
        from generation_judge import DEFAULT_JUDGE, create_judge
        self.assertEqual(DEFAULT_JUDGE, self.selection['name'])
        judge = create_judge(self.selection, 'cuda')
        self.judges.append(judge)
        self.assertIsInstance(judge, self.VevJudge)
        self.assertIs(judge.supports_decisions, True)
        with patch('generation_judge.QuestionJudge') as old:
            selection = dict(source='huggingface', name='Qwen/Qwen3-VL-2B-Instruct')
            self.assertIs(create_judge(selection, 'cuda'), old.return_value)
            old.assert_called_once_with(selection, 'cuda')

    def test_factory_routes_a_local_vev_checkpoint_without_importing_transformers(self):
        from generation_judge import create_judge
        checkpoint = Path(self.temp.name) / 'local-vev'
        checkpoint.mkdir()
        (checkpoint / 'vev.json').write_text('{}')
        with patch('generation_judge.QuestionJudge', side_effect=AssertionError('Wrong runtime')):
            judge = create_judge(dict(source='local', name=str(checkpoint)), 'cpu')
        self.judges.append(judge)
        self.assertIsInstance(judge, self.VevJudge)
        self.assertNotIn('pid', judge.identity)

    def test_discovery_start_loads_cancellably_and_reports_resolved_identity(self):
        from discovery_models import GenerationEvaluator
        judge = self.judge()
        evaluator = object.__new__(GenerationEvaluator)
        evaluator.judge = judge
        updates = []
        evaluator.set_runtime_callbacks(cancelled=lambda: False,
                                        progress=lambda **fields: updates.append(fields))
        self.assertIn('pid', evaluator.scorer_identity)
        self.assertEqual(updates[-1]['phase'], 'loading visual evaluator')


if __name__ == '__main__':
    unittest.main()
