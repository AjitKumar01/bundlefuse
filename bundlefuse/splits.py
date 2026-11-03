"""Read supplied completion splits or deterministically mask resplit bundles."""
import hashlib
import json
import random
from pathlib import Path

SPLIT_FILES = ('bi_train.txt', 'bi_valid_input.txt', 'bi_valid_gt.txt',
               'bi_test_input.txt', 'bi_test_gt.txt', 'count.json')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_bundles(path, n_items):
    rows = {}
    for line in Path(path).read_text().splitlines():
        fields = [int(value.strip()) for value in line.split(',')]
        bid, items = fields[0], fields[1:]
        if bid in rows or not items or len(items) != len(set(items)):
            raise ValueError(f'{path}: duplicate bundle ID, repeated item, or empty row: {bid}')
        if min(items) < 0 or max(items) >= n_items:
            raise ValueError(f'{path}: item outside catalogue: {bid}')
        rows[bid] = items
    if not rows:
        raise ValueError(f'{path}: empty split')
    return rows


class ProvidedSplits:
    def __init__(self, directory, n_items):
        self.directory = Path(directory).resolve()
        if json.loads((self.directory / 'count.json').read_text())['#I'] != n_items:
            raise ValueError('Split and feature catalogues have different item counts')
        self.train = read_bundles(self.directory / 'bi_train.txt', n_items)
        self.splits = {}
        used = set(self.train)
        for name in ('valid', 'test'):
            inp = read_bundles(self.directory / f'bi_{name}_input.txt', n_items)
            gt = read_bundles(self.directory / f'bi_{name}_gt.txt', n_items)
            if inp.keys() != gt.keys() or used.intersection(inp):
                raise ValueError(f'{name}: inconsistent query IDs or overlapping train/valid/test IDs')
            if any(set(items).intersection(gt[bid]) for bid, items in inp.items()):
                raise ValueError(f'{name}: seeds overlap ground truth')
            used.update(inp)
            self.splits[name] = (inp, gt)
        self.hashes = {name: file_hash(self.directory / name) for name in SPLIT_FILES}

    def pairs(self, split):
        inp, gt = self.splits[split]
        # Neural loaders enumerate nonempty graph rows by ascending original bundle ID.
        return [(inp[bid], gt[bid]) for bid in sorted(inp)]

    def manifest(self):
        return {'protocol': 'provided', 'directory': str(self.directory), 'sha256': self.hashes,
                'train_bundles': len(self.train),
                'valid_queries': len(self.splits['valid'][0]),
                'test_queries': len(self.splits['test'][0]),
                'deduplicate_bundles': False, 'regenerate_masks': False, 'query_cap': None}


def mask(bundles, seed, mask_frac):
    """Split each bundle into ``(seeds, held_out)`` by hiding ``mask_frac`` of its members.

    A fresh ``random.Random(seed)`` shuffles each bundle, keeps ``round((1-mask_frac)*|b|)`` (at
    least one) items as seeds and hides the rest. If nothing would be hidden (tiny bundle) we hide
    exactly the last item so every query has a non-empty ground truth.
    """
    r = random.Random(seed); out = []
    for b in bundles:
        bb = b[:]; r.shuffle(bb)
        k = max(1, int(round(len(bb) * (1 - mask_frac))))
        s, h = bb[:k], bb[k:]
        if not h:
            s, h = bb[:-1], [bb[-1]]
        out.append((s, h))
    return out
