"""Run native feature contracts without importing the robotics overlay here."""
import subprocess
import sys
import unittest
from pathlib import Path


class RoboticsFeatureTests(unittest.TestCase):
    def test_isolated_native_feature_contracts(self):
        from setup_robotics import runtime_status
        if not runtime_status()["available"]:
            self.skipTest("isolated robotics runtime is not installed")
        app = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-E", str(app / "tests/robotics_features_case.py")],
                                cwd=app, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
