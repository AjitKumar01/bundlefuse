"""Prespecified sparse-input stress tests, retaining folds and complete bundle membership."""
import hashlib
import random


CASES = ('supplied', 'half_seeds', 'quarter_seeds', 'one_seed')


def sparse_pairs(pairs, query_ids, case, seed=20261030):
    if case not in CASES or len(pairs) != len(query_ids):
        raise ValueError('Unknown case or unaligned query IDs')
    output = []
    for (revealed, truth), bid in zip(pairs, query_ids):
        if not revealed or not truth or set(revealed).intersection(truth):
            raise ValueError('Queries need disjoint, nonempty seeds and targets')
        if case == 'supplied':
            output.append((list(revealed), list(truth)))
            continue
        # One fixed permutation makes smaller seed sets nested; no target is used to choose a seed.
        digest = hashlib.sha256(f'{seed}:{bid}'.encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        items = list(revealed); rng.shuffle(items)
        count = 1 if case == 'one_seed' else max(1, int(len(items) * (.5 if case == 'half_seeds' else .25)))
        output.append((items[:count], list(truth) + items[count:]))
    return output
