"""Split-checked inputs and masked-bundle training samples."""
import numpy as np
import torch
from bundlefuse import datasets
from bundlefuse.provenance import digest_rows


def load_data(name, smoke=False):
    ds = datasets.load(name, data_profile='extended')
    groups = [set(tuple(sorted(set(b))) for b in ds.train)]
    groups += [set(tuple(sorted(set(s) | set(g))) for s, g in pairs)
               for pairs in (ds.val_pairs, ds.test_pairs)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Complete bundles overlap across partitions')
    if smoke:
        ds.train = ds.train[:48]
        ds.val_pairs = ds.val_pairs[:8]
        ds.test_pairs = ds.test_pairs[:8]
    features = np.concatenate([ds.features[k] for k in sorted(ds.features)], axis=1).astype('float32')
    _, cats = np.unique(ds.cat_labels, return_inverse=True)
    freq = np.zeros(ds.n_items, dtype='float32')
    for bundle in ds.train:
        freq[list(set(bundle))] += 1
    fingerprint = dict(metadata=ds.metadata, actual_train_sha256=digest_rows(ds.train),
                       actual_val_sha256=digest_rows(ds.val_pairs), actual_test_sha256=digest_rows(ds.test_pairs),
                       inputs='bundle item IDs, catalogue text features, categories, training frequency',
                       auxiliary_histories_used=False, smoke=smoke,
                       feature_shape=list(features.shape), train_bundles=len(ds.train),
                       val_queries=len(ds.val_pairs), test_queries=len(ds.test_pairs),
                       unseen_test_targets=sum(freq[i] == 0 for _, g in ds.test_pairs for i in set(g)))
    fingerprint['unseen_test_targets'] = int(fingerprint['unseen_test_targets'])
    return ds, features, cats, freq, fingerprint


def pad_seeds(rows, n):
    ids = torch.full((len(rows), max(map(len, rows))), n, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.as_tensor(row)
        valid[i, :len(row)] = True
    return ids, valid


def make_batch(bundles, n, rng, negative_pool=512):
    """Reveal half the items or a random nonempty proper subset with equal probability.

    Keep EVERY hidden positive. Negative candidates exclude the entire source bundle.
    Unobserved candidates are treated as negatives for this completion objective;
    they are not asserted to be intrinsically incompatible items.
    """
    seeds, positives, negatives = [], [], []
    for bundle in bundles:
        b = rng.permutation(sorted(set(bundle)))
        if len(b) < 2:
            raise ValueError('Training bundles need two distinct items')
        reveal = max(1, len(b)//2) if rng.random() < .5 else int(rng.integers(1, len(b)))
        seeds.append(b[:reveal].tolist()); positives.append(b[reveal:])
        pool = np.setdiff1d(np.arange(n), b, assume_unique=True)
        negatives.append(rng.choice(pool, min(negative_pool, len(pool)), replace=False))
    max_pos = max(map(len, positives))
    candidates = torch.full((len(bundles), max_pos + max(map(len, negatives))), n, dtype=torch.long)
    pos_mask = torch.zeros_like(candidates, dtype=torch.bool)
    neg_mask = torch.zeros_like(candidates, dtype=torch.bool)
    for i, (pos, neg) in enumerate(zip(positives, negatives)):
        candidates[i, :len(pos)] = torch.as_tensor(pos)
        candidates[i, max_pos:max_pos+len(neg)] = torch.as_tensor(neg)
        pos_mask[i, :len(pos)] = True
        neg_mask[i, max_pos:max_pos+len(neg)] = True
    ids, valid = pad_seeds(seeds, n)
    return ids, valid, candidates, pos_mask, neg_mask


def completion_loss(scores, positive, negative, *, hard, negatives=256):
    """Average per-positive softmax loss against a common sampled negative set.

    Other known positives are excluded from each positive's denominator. Later epochs
    use half hard and half uniform negatives from the sampled pool, without pretending
    this biased sampled objective equals full-catalogue likelihood.
    """
    selected = torch.zeros_like(negative)
    for i in range(len(scores)):
        eligible = torch.where(negative[i])[0]
        count = min(negatives, len(eligible))
        if count < 1:
            raise ValueError('No negative candidates available')
        random_count = count//2 if hard else count
        chosen = eligible[torch.randperm(len(eligible), device=scores.device)[:random_count]]
        selected[i, chosen] = True
        if hard:
            rest = scores[i].detach().masked_fill(~negative[i] | selected[i], -torch.inf)
            selected[i, rest.topk(count-random_count).indices] = True
    denominator = scores.masked_fill(~selected, -torch.inf).logsumexp(1, keepdim=True)
    losses = torch.nn.functional.softplus(denominator - scores)
    return ((losses * positive).sum(1) / positive.sum(1)).mean()
