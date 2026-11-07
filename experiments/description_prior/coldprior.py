"""Spotify cold-track prior variants. Fitting queries only (first 600)."""
import json
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
from bundlefuse import datasets
from cpm2 import incidence, fit_w, group_lift, cold_recall
from cpm3 import obs_matrix, hat_matrix, combine
import cpm4

name = 'spotify'
pop = json.loads(Path(f'results/sufficiency/{name}/evaluation_population.json').read_text())
part = json.loads(Path(f'results/sufficiency/{name}/partition.json').read_text())
ds = datasets.load(name, data_profile='research')
val = [ds.val_pairs[i] for i in pop['validation_original_indices']]
fit = [val[i] for i in part['fitting']][:600]
B = incidence(ds.train, ds.n_items); f = np.asarray(B.sum(0)).ravel().astype(np.float64); m = B.shape[0]
size = np.asarray(B.sum(1)).ravel(); ft = torch.tensor(f, dtype=torch.float32); n = ds.n_items
n0, n1 = int((f == 0).sum()), int((f == 1).sum())
gt = n1 / max(n0, 1)
print(f'unseen={n0} singletons={n1} Good-Turing per-unseen count={gt:.4f}', flush=True)
cpm4.A[:] = sorted({.1, round(gt, 4), 1, 5, 25, 100})
cpm4.TAU[:] = [.25, .5, .75, 1]
X = np.concatenate([np.ones((n, 1), np.float32), ds.features['audio']], 1)
W = fit_w(X, B, f, 1.); XW = X @ W
audio = hat_matrix(fit, XW, X, m)
obs = obs_matrix(fit, f, m, B, size, (.5, .5, 2, 100))


def group_rates(labels):
    """Per-group pair lift, shrunk to the global lift with one pseudo-expected-count."""
    one = sp.csr_matrix((np.ones(n), (np.arange(n), labels)))
    Y = (B @ one).tocsc()
    obs_g = np.asarray(Y.multiply(Y).sum(0)).ravel() - one.T @ f
    tot = one.T @ f
    exp_g = (tot ** 2 - one.T @ (f ** 2)) / m
    glob = obs_g.sum() / exp_g.sum()
    return (obs_g + glob) / (exp_g + 1.), glob, np.bincount(labels)


ra, ga, na = group_rates(ds.cat_labels); rb, gb, nb = group_rates(ds.alb_labels)
print('global lifts', ga, gb, flush=True)


def build(mode):
    out = torch.zeros_like(obs)
    for j, (s, _) in enumerate(fit):
        s = list(s)
        ca = np.bincount(ds.cat_labels[s], minlength=len(na))[ds.cat_labels] / len(s)
        cb = np.bincount(ds.alb_labels[s], minlength=len(nb))[ds.alb_labels] / len(s)
        if mode == 'constant':
            v = (ga - 1) * ca + (gb - ga) * cb
        elif mode == 'per-group':
            v = np.maximum(ra[ds.cat_labels] - 1, 0) * ca + np.maximum(rb[ds.alb_labels] - ra[ds.cat_labels], 0) * cb
        elif mode == 'size-normalized':
            v = (ga - 1) * ca / np.sqrt(na[ds.cat_labels]) + (gb - ga) * cb / np.sqrt(nb[ds.alb_labels])
        elif mode == 'per-group size-normalized':
            v = np.maximum(ra[ds.cat_labels] - 1, 0) * ca / np.sqrt(na[ds.cat_labels]) + \
                np.maximum(rb[ds.alb_labels] - ra[ds.cat_labels], 0) * cb / np.sqrt(nb[ds.alb_labels])
        out[:, j] = torch.tensor(v, dtype=torch.float32)
    return out


for mode in ('constant', 'per-group', 'size-normalized', 'per-group size-normalized'):
    g = build(mode)
    for combo, hat in (('add', (audio + g).clamp_min(0)), ('mult', (audio.clamp_min(0) * (1 + g)))):
        state, v = cpm4.search(obs, hat, ft, fit)
        S = combine(obs, hat, ft, state['kappa'], state['tau'], state['a']).numpy()
        print(f'{mode:28s} {combo:5s} N {v:.4f} {state} {cold_recall(S, fit, f)}', flush=True)
