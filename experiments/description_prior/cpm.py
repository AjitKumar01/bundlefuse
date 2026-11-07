"""Pilot: co-purchase prior model. Descriptions predict co-purchase lift; counts update that prediction.

    W  = G^-1 (Y'Y - X'FX) G^-1,   Y = B X,  G = X'FX + lam I,  F = diag(f)
    L  = m X W X'                  description-predicted lift
    post_ic = rho * m N_ic / (f_i f_c) + (1-rho) * L_ic      rho = f_c / (f_c + kappa)
    score_c = tau log(f_c + 1) + log mean_i post_ic

Validation only: fitting split picks scalars, tuning split reports. The test split is never read here.
"""
import json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
from bundlefuse import datasets
from bundlefuse.metrics import ranking_metrics

FLOOR = 1e-3


def incidence(train, n):
    r, c = [], []
    for b, items in enumerate(train):
        for i in dict.fromkeys(items):
            r.append(b); c.append(i)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(train), n))


def load(name):
    ds = datasets.load(name, data_profile='research')
    pop = json.loads(Path(f'results/sufficiency/{name}/evaluation_population.json').read_text())
    part = json.loads(Path(f'results/sufficiency/{name}/partition.json').read_text())
    val = [ds.val_pairs[i] for i in pop['validation_original_indices']]
    return ds, [val[i] for i in part['fitting']], [val[i] for i in part['tuning']]


def design(ds, blocks):
    n = ds.n_items
    parts = [np.ones((n, 1), np.float32)]
    for key in blocks:
        if key == 'cat':
            parts.append(np.eye(ds.cat_labels.max() + 1, dtype=np.float32)[ds.cat_labels])
        else:
            parts.append(ds.features[key].astype(np.float32))
    return np.concatenate(parts, 1)


def fit_w(X, B, f, lam):
    """Inverse-variance weighted two-sided ridge. Cold items have f=0 and drop out by construction."""
    X64 = X.astype(np.float64)
    Y = B @ X64
    G0 = (X64.T * f) @ X64
    S = Y.T @ Y - G0
    G = G0 + lam * np.trace(G0) / len(G0) * np.eye(len(G0))
    return np.linalg.solve(G, np.linalg.solve(G, S).T).astype(np.float32)


class Query:
    """Per-seed lift prediction and observed counts, kept per query to reuse across the scalar grid."""
    def __init__(self, B, f, pairs, chunk=256):
        self.B, self.f, self.pairs = B, f, pairs
        self.m = B.shape[0]
        self.ft = torch.tensor(f, dtype=torch.float32)
        self.seed = [sorted(set(s)) for s, _ in pairs]
        allseeds = sorted({i for s in self.seed for i in s})
        self.pos = {i: k for k, i in enumerate(allseeds)}
        self.allseeds = np.asarray(allseeds)
        N = (B.T @ B[:, allseeds]).toarray().astype(np.float32)
        N[allseeds, np.arange(len(allseeds))] = 0
        self.lift_obs = torch.tensor(N)
        self.lift_obs *= self.m / (self.ft[:, None].clamp_min(1) * self.ft[self.allseeds][None].clamp_min(1))
        self.hat = None

    def set_prior(self, L):
        self.hat = torch.tensor(L).clamp_min(0)

    def score(self, kappa, tau):
        """kappa=None means the prior alone; kappa=0 means the counts alone."""
        f = self.ft[:, None]
        if kappa is None:
            post = self.hat
        else:
            rho = torch.ones_like(f) if kappa == 0 else f / (f + kappa)
            post = rho * self.lift_obs if self.hat is None else rho * self.lift_obs + (1 - rho) * self.hat
        out = torch.zeros(len(f), len(self.seed))
        for j, s in enumerate(self.seed):
            out[:, j] = torch.log(FLOOR + post[:, [self.pos[i] for i in s]].mean(1))
        return out + tau * torch.log1p(self.ft)[:, None]


def metrics(scores, pairs):
    return ranking_metrics(scores.numpy(), pairs, Ks=(20,))[20]


def prior_lift(XW, X, seeds, m):
    return m * (XW @ X[seeds].T)


def search(q, kappas, taus):
    """Coordinate search on fitting NDCG@20."""
    tau = 0.5; kappa = kappas[len(kappas) // 2]; best = -1
    for _ in range(2):
        for k in kappas:
            v = metrics(q.score(k, tau), q.pairs)[1]
            if v > best: best, kappa = v, k
        for t in taus:
            v = metrics(q.score(kappa, t), q.pairs)[1]
            if v > best: best, tau = v, t
    return kappa, tau, best


def main(name):
    t0 = time.time()
    ds, fit, tune = load(name)
    B = incidence(ds.train, ds.n_items)
    f = np.asarray(B.sum(0)).ravel().astype(np.float64)
    m = B.shape[0]
    print(f'{name}: n={ds.n_items} m={m} fit={len(fit)} tune={len(tune)} cold={np.mean(f==0):.3f}', flush=True)
    qf, qt = Query(B, f, fit), Query(B, f, tune)
    print('load', round(time.time() - t0), 's', flush=True)
    kappas = [.01, .1, 1, 5, 25, 100]; taus = [0, .25, .5, .75, 1]
    rows = {}
    # counts alone: no description at all
    qf.hat = qt.hat = None
    k, t, _ = search(qf, [0], taus)
    rows['counts only'] = dict(tau=t, tune=metrics(qt.score(0, t), tune))
    print('counts only', rows['counts only'], flush=True)

    specs = {'category block model': ['cat'], 'title': ['content'], 'all description': ['cat', 'content', 'image', 'desc']}
    for label, blocks in specs.items():
        X = design(ds, blocks)
        bestv = -1
        for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1):
            t1 = time.time()
            W = fit_w(X, B, f, lam)
            XW = X @ W
            secs = time.time() - t1
            qf.set_prior(prior_lift(XW, X, qf.allseeds, m)); qt.set_prior(prior_lift(XW, X, qt.allseeds, m))
            k, t, v = search(qf, kappas, taus)
            print(f'  {label} lam={lam} kappa={k} tau={t} fit={v:.4f} wfit={secs:.1f}s', flush=True)
            if v > bestv:
                bestv = v
                sel = dict(lam=lam, kappa=k, tau=t, tune=metrics(qt.score(k, t), tune),
                           seconds=secs)
                # prior alone at the selected lam, same tau search
                _, pt = max(((metrics(qf.score(None, t2), fit)[1], t2) for t2 in taus))
                sel['prior_only'] = metrics(qt.score(None, pt), tune)
        rows[label] = sel
        print(label, sel, flush=True)

    # unlearned similarity in the same slot: W = identity scaled to lift by least squares
    X = design(ds, ['cat', 'content', 'image', 'desc'])[:, 1:]
    hat_f = X @ X[qf.allseeds].T; hat_t = X @ X[qt.allseeds].T
    A = (B.T @ B[:, qf.allseeds]).toarray().astype(np.float32); A[qf.allseeds, np.arange(len(qf.allseeds))] = 0
    D = f[:, None] * f[qf.allseeds][None] / m
    mask = D > 0
    c = (A[mask] * hat_f[mask]).sum() / ((D[mask] * hat_f[mask] ** 2).sum() + 1e-9)
    qf.set_prior(c * hat_f); qt.set_prior(c * hat_t)
    k, t, _ = search(qf, kappas, taus)
    rows['raw similarity prior (W=cI)'] = dict(c=float(c), kappa=k, tau=t, tune=metrics(qt.score(k, t), tune))
    print('raw', rows['raw similarity prior (W=cI)'], flush=True)

    ref = json.loads(Path(f'results/sufficiency/{name}/frozen_methods.json').read_text())
    print('reference tuning', ref['reference']['tuning_metrics'], ref['reference']['config'])
    out = Path(sys.argv[2]) / f'{name}.json'
    out.write_text(json.dumps(rows, indent=1, default=float))
    print('total', round(time.time() - t0), 's')


if __name__ == '__main__':
    main(sys.argv[1])
