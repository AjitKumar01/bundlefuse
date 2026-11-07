"""Diagnosis on the FITTING split only: P(c|S) = sum_b P(b|S) B_bc, with a neighbour kernel for P(b|S).

h_b = (sum_i B_bi f_i^-alpha / |b|^gamma)^p, top K bundles, normalised to unit mass.
Also re-runs the link test on pairs with no training co-occurrence.
"""
import json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
from bundlefuse import datasets
from bundlefuse.metrics import ranking_metrics
from cpm2 import incidence, fit_w, group_lift, FLOOR

KERNELS = [(a, g, p, k) for a in (.5, 1.) for g in (0., .5, 1.) for p, k in ((1, None), (2, None), (2, 100), (3, 100))]


def obs_matrix(pairs, f, m, B, size, kernel):
    alpha, gamma, power, k = kernel
    n = B.shape[1]; Bt = B.T.tocsr()
    out = np.zeros((n, len(pairs)), np.float32)
    fs = np.maximum(f, 1)
    vol = size ** gamma
    for j, (s, _) in enumerate(pairs):
        x = np.zeros(n); x[list(s)] = fs[list(s)] ** -alpha
        h = (B @ x) / vol
        if k is not None and k < len(h):
            drop = np.argpartition(-h, k)[k:]
            h[drop] = 0
        h = h ** power
        mass = h.sum()
        if mass > 0:
            out[:, j] = Bt @ (h / mass)
    return torch.tensor(out * m / np.maximum(f, 1)[:, None].astype(np.float32))


def m20(score, pairs):
    return ranking_metrics(score.numpy(), pairs, Ks=(20,))[20]


def hat_matrix(pairs, XW, X, m, extra=None):
    means = torch.tensor(np.stack([X[list(s)].mean(0) for s, _ in pairs], 1))
    out = m * (torch.tensor(XW) @ means)
    if extra is not None:
        out = out + extra
    return out.clamp_min(0)


def combine(obs, hat, ft, kappa, tau, a):
    f = ft[:, None]
    rho = f / (f + kappa)
    post = rho * obs + (1 - rho) * hat
    return torch.log(FLOOR + post) + tau * torch.log(f + a)


def link(ds, B, f, val, lam, rng):
    """Pairs never co-purchased in training, against a popularity-matched decoy for the same seed."""
    n = ds.n_items
    keys = ['audio'] if ds.music else ['content', 'image', 'desc']
    results = {}
    parts = [np.ones((n, 1), np.float32)] + [ds.features[k].astype(np.float32) for k in keys]
    if not ds.music:
        parts.insert(1, np.eye(ds.cat_labels.max() + 1, dtype=np.float32)[ds.cat_labels])
    X = np.concatenate(parts, 1)
    W = fit_w(X, B, f, lam); XW = X @ W
    order = np.argsort(f); rank = np.empty_like(order); rank[order] = np.arange(n)
    N = (B.T @ B).tocsr()
    rows = []
    pos = [(i, g) for s, gt in val for i in s for g in gt]
    for k in rng.permutation(len(pos))[:200000]:
        i, g = pos[k]
        if N[i, g] == 0:
            d = order[np.clip(rank[g] + rng.integers(-50, 51), 0, n - 1)]
            if N[i, d] == 0 and d != g and d != i:
                rows.append((i, g, d))
        if len(rows) >= 20000:
            break
    rows = np.asarray(rows)
    Xn = X[:, 1:] / (np.linalg.norm(X[:, 1:], axis=1, keepdims=True) + 1e-9)
    Xc = Xn if ds.music else X[:, 1 + ds.cat_labels.max() + 1:]
    Xc = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
    auc = lambda p, q: float((p > q).mean())
    raw = lambda A: auc((A[rows[:, 0]] * A[rows[:, 1]]).sum(1), (A[rows[:, 0]] * A[rows[:, 2]]).sum(1))
    lrn_p = (XW[rows[:, 1]] * X[rows[:, 0]]).sum(1); lrn_n = (XW[rows[:, 2]] * X[rows[:, 0]]).sum(1)
    results.update(pairs=len(rows), auc_raw_all=raw(Xn), auc_raw_content=raw(Xc), auc_learned=auc(lrn_p, lrn_n))
    cold_g = f[rows[:, 1]] == 0
    results['cold_target_pairs'] = int(cold_g.sum())
    if cold_g.sum():
        results['auc_learned_cold_target'] = auc(lrn_p[cold_g], lrn_n[cold_g])
        results['auc_raw_content_cold_target'] = auc((Xc[rows[cold_g, 0]] * Xc[rows[cold_g, 1]]).sum(1),
                                                    (Xc[rows[cold_g, 0]] * Xc[rows[cold_g, 2]]).sum(1))
    # how far is W from a multiple of the identity, on the content block
    d = len(W)
    start = 1 + (0 if ds.music else ds.cat_labels.max() + 1)
    Wc = W[start:, start:]; Wc = (Wc + Wc.T) / 2
    c = np.trace(Wc) / len(Wc)
    results['W_identity_share'] = float(c ** 2 * len(Wc) / (Wc ** 2).sum())
    eig = np.linalg.eigvalsh(Wc)
    results['W_negative_eig_mass'] = float(np.abs(eig[eig < 0]).sum() / np.abs(eig).sum())
    return results


def main(name, outdir, lam_link):
    t0 = time.time()
    ds = datasets.load(name, data_profile='research')
    pop = json.loads(Path(f'results/sufficiency/{name}/evaluation_population.json').read_text())
    part = json.loads(Path(f'results/sufficiency/{name}/partition.json').read_text())
    val = [ds.val_pairs[i] for i in pop['validation_original_indices']]
    fit = [val[i] for i in part['fitting']]
    if ds.music:
        fit = fit[:600]
    B = incidence(ds.train, ds.n_items)
    f = np.asarray(B.sum(0)).ravel().astype(np.float64); m = B.shape[0]
    ft = torch.tensor(f, dtype=torch.float32)
    size = np.asarray(B.sum(1)).ravel()
    res = {'kernels': []}
    best = (-1, None)
    for kernel in KERNELS:
        obs = obs_matrix(fit, f, m, B, size, kernel)
        v = m20(torch.log(FLOOR + obs) + torch.log(ft[:, None] + .1), fit)
        res['kernels'].append(dict(kernel=kernel, fit=list(map(float, v))))
        print(kernel, 'R %.4f N %.4f' % v, round(time.time() - t0), flush=True)
        if v[1] > best[0]:
            best = (v[1], kernel)
    print('best kernel', best, flush=True)
    res['best_kernel'] = best[1]
    res['link'] = link(ds, B, f, val, lam_link, np.random.default_rng(7))
    print('link', res['link'], flush=True)
    Path(outdir, f'{name}_diag.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
