"""Temporary, query-major float32 signal cache for full-population validation selection."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


class ValidationCache:
    def __init__(self, pairs, users, scorer, n_items, chunk=32, keys=None):
        if not pairs or chunk < 1:
            raise ValueError('Validation queries and a positive chunk size are required')
        if users is not None and len(users) != len(pairs):
            raise ValueError('Validation users and queries must align')
        self.temporary = TemporaryDirectory(prefix='bundlefuse_validation_')
        self.arrays, self.keys, self.count, self.chunk = {}, [], len(pairs), chunk
        try:
            for start in range(0, len(pairs), chunk):
                seeds = [s for s, _ in pairs[start:start + chunk]]
                us = users[start:start + chunk] if users is not None else None
                raw, normalized = scorer(seeds, us)
                if not self.arrays:
                    self.keys = list(normalized) if keys is None else [k for k in keys if k in normalized]
                    for i, key in enumerate(self.keys):
                        self.arrays[key] = np.lib.format.open_memmap(
                            Path(self.temporary.name) / f'{i}.npy', mode='w+', dtype=np.float32,
                            shape=(len(pairs), n_items))
                for key, array in self.arrays.items():
                    array[start:start + len(seeds)] = normalized[key].detach().cpu().numpy().T
                del raw, normalized
            for array in self.arrays.values():
                array.flush()
        except BaseException:
            self.close()
            raise

    def blocks(self):
        for start in range(0, self.count, self.chunk):
            yield start, {key: torch.from_numpy(array[start:start + self.chunk].T)
                          for key, array in self.arrays.items()}

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
