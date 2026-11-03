"""Portable dataset fingerprints and JSON run metadata shared by fitting and inference."""
import hashlib
import json
from pathlib import Path
import platform
import sys

from .splits import file_hash


def digest_rows(rows):
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def record_dataset(dataset, root, files, *, protocol, split_seed, mask_frac, val_limit, test_limit,
                   val_total, test_total, val_ids=None, test_ids=None, data_profile='extended'):
    dataset.val_ids = list(range(len(dataset.val_pairs))) if val_ids is None else list(val_ids)
    dataset.test_ids = list(range(len(dataset.test_pairs))) if test_ids is None else list(test_ids)
    dataset.metadata = {
        'dataset': dataset.name, 'n_items': dataset.n_items, 'protocol': protocol,
        'data_profile': data_profile,
        'split_seed': split_seed if protocol == 'resplit' else None,
        'mask_frac': mask_frac if protocol == 'resplit' else None,
        'val_limit': val_limit, 'test_limit': test_limit,
        'train_bundles': len(dataset.train), 'val_queries': len(dataset.val_pairs),
        'test_queries': len(dataset.test_pairs), 'val_total': val_total, 'test_total': test_total,
        'full_validation': len(dataset.val_pairs) == val_total,
        'full_test': len(dataset.test_pairs) == test_total,
        'train_sha256': digest_rows(dataset.train),
        'val_pairs_sha256': digest_rows(dataset.val_pairs), 'test_pairs_sha256': digest_rows(dataset.test_pairs),
        'input_sha256': {name: file_hash(Path(root) / name) for name in files if (Path(root) / name).exists()},
    }
    return dataset


def run_metadata():
    import numpy, scipy, sklearn, torch
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest.update(str(path.relative_to(package)).encode()); digest.update(path.read_bytes())
    return {'package_source_sha256': digest.hexdigest(), 'python': sys.version, 'platform': platform.platform(),
            'entrypoint_sha256': {name: file_hash(package.parent / name)
                                 for name in ('evaluator.py', 'inference.py', 'validate.py')},
            'versions': {m.__name__: m.__version__ for m in (numpy, scipy, sklearn, torch)},
            'torch_threads': torch.get_num_threads()}


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
