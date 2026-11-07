"""Pilot 2. Per-query vectors, paired comparison against the frozen recipes, link diagnostics. Validation only."""
import json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
from bundlefuse import datasets
from bundlefuse.metrics import ranking_metrics
from experiments.novelty.data import load_inputs
from experiments.sufficiency.engine import Bank, score as ref_score

FLOOR = 1e-3
KAPPA = [.1, 1, 5, 25, 100, 400, 1600, 6400]
TAU = [.5, .75, 1, 1.25, 1.5]
A = [.1, 1, 5, 25]
LAMS = (1e-3, 1e-2, 1e-1, 1, 10)


def incidence(train, n):
    r, c = [], []
    for b, items in enumerate(train):
        for i in dict.fromkeys(items):
            r.append(b); c.append(i)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(train), n))


def fit_w(X, B, f, lam):
    X64 = X.astype(np.float64)
    Y = B @ X64
    G0 = (X64.T * f) @ X64
    S = Y.T @ Y - G0
    G = G0 + lam * np.trace(G0) / len(G0) * np.eye(len(G0))
    return np.linalg.solve(G, np.linalg.solve(G, S).T).astype(np.float32)


def group_lift(B, labels, f, m):
    """Observed lift of pairs sharing a label, excluding self-pairs."""
    one = sp.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)), labels)))
    Y = B @ one
    observed = (Y.multiply(Y)).sum() - f.sum()
    totals = one.T @ f
    expected = ((totals ** 2).sum() - (f ** 2).sum()) / m
    return float(observed / expected)


class Vectors:
    """obs: mean observed lift per seed; hat: mean predicted lift per seed. Items x queries float32."""
    def __init__(self, pairs, f, m, B, amp=None):
        self.pairs, self.f, self.m = pairs, torch.tensor(f, dtype=torch.float32), m
        n = B.shape[1]
        self.seeds = [sorted(set(s)) for s, _ in pairs]
        self.obs = torch.zeros(n, len(pairs))
        Bc = B.tocsc(); Bt = B.T.tocsr()
        fsafe = np.maximum(f, 1)
        for j, s in enumerate(self.seeds):
            x = np.zeros(n); x[s] = 1 / fsafe[s]
            h = B @ x
            if amp:
                power, k = amp
                mass = h.sum()
                order = np.argpartition(-h, min(k, len(h) - 1))[:k]
                keep = np.zeros_like(h); keep[order] = h[order] ** power
                h = keep * mass / max(keep.sum(), 1e-12)
            self.obs[:, j] = torch.tensor(Bt @ h / len(s), dtype=torch.float32)
        self.obs *= m / self.f[:, None].clamp_min(1)
        self.hat = None

    def set_prior(self, XW, X, extra=None):
        out = torch.zeros_like(self.obs)
        XWt = torch.tensor(XW)
        means = torch.tensor(np.stack([X[s].mean(0) for s in self.seeds], 1))
        for a in range(0, means.shape[1], 64):
            out[:, a:a + 64] = self.m * (XWt @ means[:, a:a + 64])
        if extra is not None:
            out += extra
        self.hat = out.clamp_min(0)

    def score(self, kappa, tau, a):
        f = self.f[:, None]
        if kappa is None:
            post = self.hat
        elif self.hat is None:
            post = self.obs
        else:
            rho = f / (f + kappa)
            post = rho * self.obs + (1 - rho) * self.hat
        return torch.log(FLOOR + post) + tau * torch.log(f + a)


def m20(scores, pairs):
    return ranking_metrics(np.asarray(scores), pairs, Ks=(20,))[20]


def per_query(scores, pairs):
    out = np.zeros((len(pairs), 2))
    s = np.asarray(scores)
    for j, p in enumerate(pairs):
        out[j] = ranking_metrics(s[:, [j]], [p], Ks=(20,))[20]
    return out


def search(v, kappas=KAPPA):
    state = dict(kappa=kappas[len(kappas) // 2], tau=1., a=1.)
    best = -1
    for _ in range(3):
        moved = False
        for key, grid in (('kappa', kappas), ('tau', TAU), ('a', A)):
            for value in grid:
                c = dict(state, **{key: value})
                r = m20(v.score(c['kappa'], c['tau'], c['a']), v.pairs)[1]
                if r > best + 1e-6:
                    best, state, moved = r, c, True
        if not moved:
            break
    return state, best


def boot(a, b, n=5000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    samples = d[idx].mean(1, keepdims=False) if d.ndim == 1 else d[idx].mean(1)
    return {'diff': d.mean(0).tolist(), 'lo': np.percentile(samples, 2.5, axis=0).tolist(),
            'hi': np.percentile(samples, 97.5, axis=0).tolist()}


def cold_recall(scores, pairs, f):
    """Recall@20 restricted to never-seen and to warm positives, micro-averaged."""
    s = np.asarray(scores).copy()
    hit = {True: 0, False: 0}; tot = {True: 0, False: 0}
    for j, (seeds, gt) in enumerate(pairs):
        s[list(seeds), j] = -np.inf
        top = set(np.argpartition(-s[:, j], 20)[:20].tolist())
        for g in set(gt):
            hit[f[g] == 0] += g in top; tot[f[g] == 0] += 1
    return {'cold': hit[True] / max(tot[True], 1), 'warm': hit[False] / max(tot[False], 1),
            'cold_share': tot[True] / (tot[True] + tot[False])}


def link_diagnostic(XW, Xn, W_x, val, f, m, cat, rng):
    """Do descriptions predict held-out co-purchase of pairs with no training co-occurrence?

    Positive: seed/held-out pair from validation. Negative: same seed with an item of matching
    popularity rank drawn at random. Compares raw cosine with the learned bilinear form.
    """
    order = np.argsort(f); rank = np.empty_like(order); rank[order] = np.arange(len(f))
    pos = [(i, g) for s, gt in val for i in s for g in gt]
    pos = [pos[k] for k in rng.permutation(len(pos))[:40000]]
    rows = []
    for i, g in pos:
        r = rank[g]
        neg = order[np.clip(r + rng.integers(-50, 51), 0, len(f) - 1)]
        rows.append((i, g, neg))
    rows = np.asarray(rows)
    raw_p = (Xn[rows[:, 0]] * Xn[rows[:, 1]]).sum(1); raw_n = (Xn[rows[:, 0]] * Xn[rows[:, 2]]).sum(1)
    lrn_p = (XW[rows[:, 1]] * W_x[rows[:, 0]]).sum(1); lrn_n = (XW[rows[:, 2]] * W_x[rows[:, 0]]).sum(1)
    out = {'pairs': len(rows), 'auc_raw': float((raw_p > raw_n).mean()), 'auc_learned': float((lrn_p > lrn_n).mean())}
    if cat is not None:
        same = cat[rows[:, 0]] == cat[rows[:, 1]]
        out['same_label_share_positive'] = float(same.mean())
        for label, mask in (('cross', ~same), ('same', same)):
            if mask.sum():
                out[f'auc_raw_{label}'] = float((raw_p[mask] > raw_n[mask]).mean())
                out[f'auc_learned_{label}'] = float((lrn_p[mask] > lrn_n[mask]).mean())
    return out


def main(name, outdir):
    t0 = time.time()
    model, val_all, _test, old, frozen, _id = load_inputs(name)
    pop = json.loads(Path(f'results/sufficiency/{name}/evaluation_population.json').read_text())
    part = json.loads(Path(f'results/sufficiency/{name}/partition.json').read_text())
    vi = pop['validation_original_indices']
    val = [val_all[i] for i in vi]
    fit, tune = [val[i] for i in part['fitting']], [val[i] for i in part['tuning']]
    ds = datasets.load(name, data_profile='research')
    assert ds.val_pairs == val_all
    B = incidence(ds.train, ds.n_items)
    f = np.asarray(B.sum(0)).ravel().astype(np.float64)
    m = B.shape[0]
    # frozen recipes on the same tuning queries
    frozen_m = json.loads(Path(f'results/sufficiency/{name}/frozen_methods.json').read_text())
    arrays = {k: old[k] for k in old}
    extra = Path(f'checkpoints/sufficiency/{name}/extra/validation')
    refs = {}
    for label in ['reference'] + (['E3_history_frequency'] if name.startswith('spotify') else []):
        meth = frozen_m[label]
        local = dict()
        if meth['history'] not in ('canonical',):
            local = {meth['history']: np.load(extra / f"{meth['history']}.npy", mmap_mode='r')}
        rows = []
        for ti in part['tuning']:
            z = {}
            for k in meth['keys']:
                if k == 'hist':
                    if meth['history'] != 'canonical':
                        z[k] = np.asarray(local[meth['history']][ti])
                    else:
                        a = meth['alpha']; o = vi[ti]
                        z[k] = np.asarray(old['pair'][o] if a == 0 else old['simp'][o] if a == 1 else
                                          (1 - a) * old['pair'][o] + a * old['simp'][o])
                else:
                    z[k] = np.asarray(old[k][vi[ti]])
            rows.append(ref_score(z, meth['config'], np.asarray(model.freq, np.float32)))
        R = np.stack(rows, 1)
        refs[label] = dict(per=per_query(R, tune), cold=cold_recall(R, tune, f), recorded=meth['tuning_metrics'])
        print(label, 'recomputed', refs[label]['per'].mean(0), 'recorded', meth['tuning_metrics'], refs[label]['cold'], flush=True)
        del R, rows
    amp = (2, 100) if ds.music else None
    vf, vt = Vectors(fit, f, m, B, amp), Vectors(tune, f, m, B, amp)
    print(f'{name}: n={ds.n_items} m={m} fit={len(fit)} tune={len(tune)} vectors {time.time()-t0:.0f}s', flush=True)
    rows = {}
    def record(label, state, vectors=vt):
        S = vectors.score(state['kappa'], state['tau'], state['a']).numpy()
        per = per_query(S, tune)
        rows[label] = dict(state, tune=per.mean(0).tolist(), cold=cold_recall(S, tune, f),
                           **{f'vs_{r}': boot(per, refs[r]['per']) for r in refs})
        print(label, json.dumps(rows[label], default=float), flush=True)
        return per

    vf.hat = vt.hat = None
    state, _ = search(vf, [1.])
    record('counts only', state)

    blocks = (['cat', 'audio'] if ds.music else ['cat', 'content', 'image', 'desc'])
    n = ds.n_items
    extra_f = extra_t = None
    if ds.music:
        la = group_lift(B, ds.cat_labels, f, m); lb = group_lift(B, ds.alb_labels, f, m)
        print('same-artist lift', la, 'same-album lift', lb, flush=True)
        def shared(v):
            out = torch.zeros_like(v.obs)
            for j, s in enumerate(v.seeds):
                ca = np.bincount(ds.cat_labels[s], minlength=ds.cat_labels.max() + 1)[ds.cat_labels]
                cb = np.bincount(ds.alb_labels[s], minlength=ds.alb_labels.max() + 1)[ds.alb_labels]
                out[:, j] = torch.tensor(((la - 1) * ca + (lb - la) * cb) / len(s), dtype=torch.float32)
            return out
        extra_f, extra_t = shared(vf), shared(vt)
        specs = {'audio': ['audio']}
    else:
        cat1 = np.eye(ds.cat_labels.max() + 1, dtype=np.float32)[ds.cat_labels]
        specs = {'category block model': ['cat'], 'title': ['content'], 'all description': ['cat', 'content', 'image', 'desc']}

    def design(keys):
        parts = [np.ones((n, 1), np.float32)]
        for k in keys:
            parts.append(cat1 if k == 'cat' else ds.features[k].astype(np.float32))
        return np.concatenate(parts, 1)

    final = None
    for label, keys in specs.items():
        X = design(keys); best = -1; chosen = None
        for lam in LAMS:
            t1 = time.time(); W = fit_w(X, B, f, lam); XW = X @ W; secs = time.time() - t1
            vf.set_prior(XW, X, extra_f); vt.set_prior(XW, X, extra_t)
            state, v = search(vf)
            print(f'  {label} lam={lam} {state} fit={v:.4f} w={secs:.1f}s', flush=True)
            if v > best:
                best, chosen = v, (lam, state, secs)
        lam, state, secs = chosen
        W = fit_w(X, B, f, lam); XW = X @ W
        vf.set_prior(XW, X, extra_f); vt.set_prior(XW, X, extra_t)
        record(label, dict(state, lam=lam, wfit=secs))
        ps, _ = search(vf, [None])
        record(label + ' / prior only', dict(ps, kappa=None, lam=lam))
        if label in ('all description', 'audio'):
            final = (X, W, XW, lam, state)

    # the guide's question: same model, W replaced by a calibrated multiple of the identity
    X, W, XW, lam, state = final
    Xr = X[:, 1:]
    seeds = np.asarray(sorted({i for p in fit[:300] for i in p[0]}))
    sims = Xr @ Xr[seeds].T
    A = (B.T @ B[:, seeds]).toarray().astype(np.float32); A[seeds, np.arange(len(seeds))] = 0
    E = f[:, None] * f[seeds][None] / m
    c = float((A * sims).sum() / ((E * sims * sims).sum() + 1e-9))
    del sims, A, E
    XWi = c * Xr
    vf.set_prior(XWi / m, Xr, extra_f); vt.set_prior(XWi / m, Xr, extra_t)
    s, _ = search(vf)
    record('raw similarity in the same slot (W = cI)', dict(s, c=c))

    # additive control: same two components, weighted sum of standardized columns
    vf.set_prior(XW, X, extra_f); vt.set_prior(XW, X, extra_t)
    z = lambda t: (t - t.mean(0, keepdim=True)) / (t.std(0, keepdim=True) + 1e-9)
    Zf = (z(torch.log(FLOOR + vf.obs)), z(torch.log(FLOOR + vf.hat)), z(torch.log1p(vf.f))[:, None].expand_as(vf.obs))
    Zt = (z(torch.log(FLOOR + vt.obs)), z(torch.log(FLOOR + vt.hat)), z(torch.log1p(vt.f))[:, None].expand_as(vt.obs))
    grid = (0, .25, .5, 1, 1.5, 2); bestv, bw = -1, None
    for a in grid:
        for b in grid:
            for d in grid:
                if a + b + d == 0: continue
                r = m20(a * Zf[0] + b * Zf[1] + d * Zf[2], fit)[1]
                if r > bestv: bestv, bw = r, (a, b, d)
    S = bw[0] * Zt[0] + bw[1] * Zt[1] + bw[2] * Zt[2]
    per = per_query(S, tune)
    rows['additive fusion, same inputs'] = dict(w=bw, tune=per.mean(0).tolist(), cold=cold_recall(S, tune, f),
                                                **{f'vs_{r}': boot(per, refs[r]['per']) for r in refs})
    print('additive', json.dumps(rows['additive fusion, same inputs'], default=float), flush=True)

    # link diagnostic
    rng = np.random.default_rng(7)
    Xn = Xr / (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-9)
    link = link_diagnostic(XW, Xn, X, val, f, m, ds.cat_labels, rng)
    print('link', link, flush=True)
    out = dict(rows=rows, refs={k: dict(mean=v['per'].mean(0).tolist(), cold=v['cold'], recorded=v['recorded']) for k, v in refs.items()},
               link=link, seconds=time.time() - t0)
    Path(outdir, f'{name}.json').write_text(json.dumps(out, indent=1, default=float))
    print('total', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
