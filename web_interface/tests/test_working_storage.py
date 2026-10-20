"""Working features retain every row within explicit RAM and temporary-disk budgets."""
import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class WorkingStorageTests(unittest.TestCase):
    def test_shared_budget_spills_only_overflow_and_preserves_live_views(self):
        from working_storage import FeatureStorage
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            historical = root / 'existing.npy'
            historical.write_bytes(b'historical analysis')
            storage = FeatureStorage(root, memory_bytes=32, disk_bytes=1024)
            first = storage.allocate((2, 4))
            first.array[:] = np.arange(8).reshape(2, 4)
            first.seal()
            self.assertEqual(first.array.dtype, np.float32)
            self.assertIsNone(first.path)
            second = storage.allocate((2, 4))
            second.array[:] = 2
            second.seal()
            self.assertTrue(second.path.is_file())
            states = second.readonly()
            sliced = np.asarray(states)[1:]
            self.assertTrue(np.shares_memory(states, second.array))
            with self.assertRaises(ValueError):
                states[0, 0] = 100
            del states
            gc.collect()
            self.assertEqual(second.release()['status'], 'busy')
            np.testing.assert_array_equal(sliced, [[2, 2, 2, 2]])
            path = second.path
            del sliced
            gc.collect()
            released = second.release()
            self.assertEqual(released['status'], 'released')
            self.assertGreater(released['released_disk_bytes'], 0)
            self.assertFalse(path.exists())
            self.assertEqual(first.release()['released_memory_bytes'], 32)
            self.assertEqual(storage.usage(), {'memory_bytes': 0, 'disk_bytes': 0})
            self.assertEqual(historical.read_bytes(), b'historical analysis')

    def test_exhausted_quota_fails_before_writing_a_partial_feature_file(self):
        from working_storage import FeatureStorage
        with tempfile.TemporaryDirectory() as temporary:
            storage = FeatureStorage(temporary, memory_bytes=0, disk_bytes=16)
            with self.assertRaisesRegex(ValueError, 'budget.*FAILURESHIFT_FEATURE'):
                storage.allocate((100, 4))
            self.assertEqual(storage.usage(), {'memory_bytes': 0, 'disk_bytes': 0})
            self.assertEqual(list(Path(temporary).rglob('*.npy')), [])

    def test_disk_allocation_failure_removes_only_the_unfinished_owned_file(self):
        from working_storage import FeatureStorage
        with tempfile.TemporaryDirectory() as temporary:
            storage = FeatureStorage(temporary, memory_bytes=0, disk_bytes=1024)
            with patch('working_storage.os.posix_fallocate', side_effect=OSError('No space left')):
                with self.assertRaisesRegex(ValueError, 'temporary disk space'):
                    storage.allocate((2, 4))
            self.assertEqual(storage.usage(), {'memory_bytes': 0, 'disk_bytes': 0})
            self.assertEqual(list(Path(temporary).rglob('*.npy')), [])


if __name__ == '__main__':
    unittest.main()
