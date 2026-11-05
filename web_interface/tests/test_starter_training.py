"""Contracts for learned starter policies, independent of task performance."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))


class StarterTrainingTests(unittest.TestCase):
    def training_module(self):
        spec = importlib.util.find_spec("train_starter_policies")
        self.assertIsNotNone(spec, "The reproducible starter trainer is missing")
        import train_starter_policies
        return train_starter_policies

    def test_selection_and_final_rollouts_cannot_share_seeds_with_training(self):
        training = self.training_module()
        training.validate_seed_splits([0, 1], [1000, 1001], [2000, 2001])
        for partitions in (([0, 1], [1], [2000]), ([0], [1000], [0]), ([0], [1000], [1000])):
            with self.subTest(partitions=partitions), self.assertRaises(ValueError):
                training.validate_seed_splits(*partitions)

    def test_export_preserves_normalization_bounds_and_episode_reset(self):
        self.training_module()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [str(APP / ".venv/bin/python"), "-E", str(APP / "tests/starter_training_case.py"), directory],
                cwd=APP, text=True, capture_output=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(report, {"shape": [7], "reset": True, "bounded": True, "metadata": True})

    def test_failed_retraining_preserves_published_checkpoint_and_metrics(self):
        training = self.training_module()
        self.assertTrue(hasattr(training, "staged_task"), "Retraining needs a staging boundary before evaluation")
        with tempfile.TemporaryDirectory() as directory:
            published = Path(directory) / "lift"
            published.mkdir()
            (published / "best.pt").write_bytes(b"previous evaluated weights")
            (published / "metrics.json").write_text('{"successes":20}')
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                with training.staged_task(published) as pending:
                    (pending / "best.pt").write_bytes(b"new unverified weights")
                    (pending / "metrics.json").write_text('{"successes":0}')
                    self.assertEqual((published / "best.pt").read_bytes(), b"previous evaluated weights")
                    raise RuntimeError("evaluation failed")
            self.assertEqual((published / "best.pt").read_bytes(), b"previous evaluated weights")
            self.assertEqual(json.loads((published / "metrics.json").read_text()), {"successes": 20})
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["lift"])

    def test_successful_retraining_publishes_only_completed_task_files(self):
        training = self.training_module()
        self.assertTrue(hasattr(training, "staged_task"), "Retraining needs a staging boundary before evaluation")
        with tempfile.TemporaryDirectory() as directory:
            published = Path(directory) / "lift"
            with training.staged_task(published) as pending:
                (pending / "best.pt").write_bytes(b"evaluated learned weights")
                (pending / "best.pt.metadata.json").write_text('{"state_keys":["cube_pos"]}')
                (pending / "metrics.json").write_text('{"successes":19,"episodes":20}')
                self.assertFalse(published.exists())
            self.assertEqual((published / "best.pt").read_bytes(), b"evaluated learned weights")
            self.assertEqual(json.loads((published / "metrics.json").read_text()), {"successes": 19, "episodes": 20})
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["lift"])


if __name__ == "__main__":
    unittest.main()
