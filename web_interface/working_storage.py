"""Bounded, process-owned caches for disposable analysis data."""
import io
import os
import tempfile
import threading
import weakref
from collections import OrderedDict
from pathlib import Path


def budget_bytes(name, default_mb):
    value = os.environ.get(name, str(default_mb))
    try:
        megabytes = int(value)
        if megabytes < 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f'{name} must be a nonnegative whole number of MiB.') from exc
    return megabytes * 1024 * 1024


class BoundedCache:
    """LRU entries share one byte limit; an oversized entry is never retained."""
    def __init__(self, capacity_bytes):
        self.capacity_bytes = max(0, capacity_bytes)
        self.usage_bytes = 0
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key):
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return None
            self._entries.move_to_end(key)
            return item[0]

    def put(self, key, value, size):
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self.usage_bytes -= previous[1]
            if size > self.capacity_bytes:
                return
            while self._entries and self.usage_bytes + size > self.capacity_bytes:
                _, (_, removed) = self._entries.popitem(last=False)
                self.usage_bytes -= removed
            self._entries[key] = (value, size)
            self.usage_bytes += size

    def release(self, owner):
        with self._lock:
            released = 0
            for key in list(self._entries):
                if key[0] == owner:
                    _, size = self._entries.pop(key)
                    released += size
            self.usage_bytes -= released
            return released


class FeatureStorage:
    """RAM first, then quota-limited temporary files owned by this instance.

    No existing file is adopted or removed. A feature's read-only NumPy views
    lease its allocation until all derived zero-copy views have been dropped.
    """
    def __init__(self, directory, *, memory_bytes=None, disk_bytes=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memory_limit = (budget_bytes('FAILURESHIFT_FEATURE_RAM_MB', 256)
                             if memory_bytes is None else max(0, memory_bytes))
        self.disk_limit = (budget_bytes('FAILURESHIFT_FEATURE_DISK_MB', 1024)
                           if disk_bytes is None else max(0, disk_bytes))
        self._memory = self._disk = 0
        self._temporary = None
        self.lock = threading.RLock()

    def usage(self):
        with self.lock:
            return dict(memory_bytes=self._memory, disk_bytes=self._disk)

    def allocate(self, shape):
        import numpy as np
        shape = tuple(shape)
        if len(shape) != 2 or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError('Feature storage requires a nonempty matrix.')
        size = shape[0] * shape[1] * np.dtype(np.float32).itemsize
        with self.lock:
            if self._memory + size <= self.memory_limit:
                try:
                    array = np.empty(shape, dtype=np.float32)
                except MemoryError:
                    # The configured quota is an upper bound, not a promise
                    # that the operating system can provide that much RAM.
                    pass
                else:
                    self._memory += size
                    return FeatureBuffer(self, array, memory_bytes=size)
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header, {
                'descr': np.dtype(np.float32).str, 'fortran_order': False, 'shape': shape})
            disk_size = size + len(header.getvalue())
            if self._disk + disk_size > self.disk_limit:
                raise ValueError('Analysis working-storage budget is full. Release completed analysis resources '
                                 'or increase FAILURESHIFT_FEATURE_RAM_MB / FAILURESHIFT_FEATURE_DISK_MB '
                                 'before analyzing again. No input rows were omitted.')
            path = None
            try:
                if self._temporary is None:
                    self._temporary = Path(tempfile.mkdtemp(prefix='.working-', dir=self.directory))
                descriptor, name = tempfile.mkstemp(suffix='.npy', prefix='features-', dir=self._temporary)
                path = Path(name)
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(header.getvalue())
                    stream.truncate(disk_size)
                    # A sparse mapping can later SIGBUS on a full filesystem.
                    # Reserve its actual space before allowing model writes.
                    if hasattr(os, 'posix_fallocate'):
                        os.posix_fallocate(stream.fileno(), 0, disk_size)
                array = np.memmap(path, dtype=np.float32, mode='r+',
                                  offset=len(header.getvalue()), shape=shape)
            except (OSError, MemoryError) as exc:
                if path is not None:
                    path.unlink(missing_ok=True)
                raise ValueError('Could not allocate analysis working storage. Free temporary disk space or '
                                 'increase FAILURESHIFT_FEATURE_RAM_MB, then Analyze again.') from exc
            self._disk += disk_size
            return FeatureBuffer(self, array, path=path, disk_bytes=disk_size)


class FeatureBuffer:
    def __init__(self, storage, array, *, path=None, memory_bytes=0, disk_bytes=0):
        self.storage = storage
        self.array = array
        self.path = path
        self.memory_bytes = memory_bytes
        self.disk_bytes = disk_bytes
        self._leases = 0
        self._sealed = False

    @property
    def released(self):
        return self.array is None

    def seal(self):
        with self.storage.lock:
            if self.path is not None:
                self.array.flush()
            self.array.setflags(write=False)
            self._sealed = True

    def readonly(self, rows=None):
        import numpy as np
        with self.storage.lock:
            if self.released:
                raise ValueError('Analysis working features were released. Analyze the input again.')
            if not self._sealed:
                raise ValueError('Analysis working features are not ready.')
            # Keep the lease on the root frombuffer array. Slices and np.asarray
            # retain that same base, so dropping the initial view is safe too.
            root = np.frombuffer(memoryview(self.array).toreadonly(), dtype=np.float32)
            self._leases += 1
            weakref.finalize(root, self._unpin)
            view = root.reshape(self.array.shape)
            return view if rows is None else view[:rows]

    def _unpin(self):
        with self.storage.lock:
            self._leases -= 1

    def release(self):
        with self.storage.lock:
            result = dict(status='already_released', released_memory_bytes=0, released_disk_bytes=0)
            if self.released:
                return result
            if self._leases:
                return {**result, 'status': 'busy', 'reason': 'Analysis features are in use.'}
            array = self.array
            if self.path is not None:
                self.path.unlink(missing_ok=True)
                array._mmap.close()
            self.array = None
            self.storage._memory -= self.memory_bytes
            self.storage._disk -= self.disk_bytes
            if self.storage._disk == 0 and self.storage._temporary is not None:
                self.storage._temporary.rmdir()
                self.storage._temporary = None
            return dict(status='released', released_memory_bytes=self.memory_bytes,
                        released_disk_bytes=self.disk_bytes)
