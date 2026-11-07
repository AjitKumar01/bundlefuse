"""Stage 1, validation only: select and freeze. Writes <name>_frozen.json; never reads test pairs."""
import json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
from bundlefuse import datasets
from bundlefuse.metrics import ranking_metrics
from cpm2 import incidence, fit_w, group_lift, FLOOR, boot, per_query, cold_recall
from cpm3 import obs_matrix, hat_matrix, combine, m20
from experiments.novelty.data import load_inputs
from experiments.sufficiency.engine import score as ref_score

KAPPA = [.1, 1, 5, 25, 100, 400, 1600, 6400]
TAU = [.5, .75, 1, 1.25, 1.5, 2]
A = [.1, 1, 5, 25]
LAMS = (1e-3, 1e-2, 1e-1, 1)
KERNELS = [(.5, .5, 1, None), (.5, .5, 2, 100)]


def search(obs, hat, ft, pairs, taus=TAU):
    best, state = -1, None
    for start in (.1, 25, 1600):
        s = dict(kappa=start, tau=1., a=1.)
        for _ in range(3):
            moved = False
            for key, grid in (('kappa', KAPPA), ('tau', taus), ('a', A)):
                for value in grid:
                    c = dict(s, **{key: value})
                    r = m20(combine(obs, hat, ft, c['kappa'], c['tau'], c['a']), pairs)[1]
                    if r > best + 1e-6:
                        best, state, s, moved = r, c, c, True
            if not moved:
                break
    return state, best


def shared(pairs, ds, la, lb, shape):
    out = torch.zeros(shape)
    for j, (s, _) in enumerate(pairs):
        s = list(s)
        ca = np.bincount(ds.cat_labels[s], minlength=ds.cat_labels.max() + 1)[ds.cat_labels]
        cb = np.bincount(ds.alb_labels[s], minlength=ds.alb_labels.max() + 1)[ds.alb_labels]
        out[:, j] = torch.tensor(((la - 1) * ca + (lb - la) * cb) / len(s), dtype=torch.float32)
    return out


def spectrum_test(ds, B, f, val, X, W, rng):
    """Do the positive and negative parts of W separately predict unseen pairs?"""
    n = ds.n_items
    N = (B.T @ B).tocsr()
    order = np.argsort(f); rank = np.empty_like(order); rank[order] = np.arange(n)
    pos = [(i, g) for s, gt in val for i in s for g in gt]
    rows = []
    for k in rng.permutation(len(pos))[:200000]:
        i, g = pos[k]
        if N[i, g] == 0:
            d = order[np.clip(rank[g] + rng.integers(-50, 51), 0, n - 1)]
            if N[i, d] == 0 and d != g and d != i:
                rows.append((i, g, d))
        if len(rows) >= 20000:
            break
    rows = np.asarray(rows)
    S = (W + W.T) / 2
    e, U = np.linalg.eigh(S)
    out = {'neg_eig_mass': float(np.abs(e[e < 0]).sum() / np.abs(e).sum())}
    for label, mask in (('positive', e > 0), ('negative', e < 0)):
        Z = X @ U[:, mask]; sign = e[mask]
        p = ((Z[rows[:, 0]] * sign) * Z[rows[:, 1]]).sum(1); q = ((Z[rows[:, 0]] * sign) * Z[rows[:, 2]]).sum(1)
        out[f'auc_{label}_part'] = float((p > q).mean())
    # null: same fit with bundle membership shuffled within each item's frequency
    perm = rng.permutation(B.shape[1])
    Bn = B[:, perm]
    Wn = fit_w(X, Bn, np.asarray(Bn.sum(0)).ravel().astype(np.float64), LAM_FOR_NULL[0])
    e0 = np.linalg.eigvalsh((Wn + Wn.T) / 2)
    out['null_neg_eig_mass'] = float(np.abs(e0[e0 < 0]).sum() / np.abs(e0).sum())
    out['W_fro'] = float(np.linalg.norm(S)); out['null_W_fro'] = float(np.linalg.norm(e0))
    return out


LAM_FOR_NULL = [None]


def main(name, outdir):
    t0 = time.time()
    model, val_all, _unused, old, _f, _id = load_inputs(name)
    del _unused
    pop = json.loads(Path(f'results/sufficiency/{name}/evaluation_population.json').read_text())
    part = json.loads(Path(f'results/sufficiency/{name}/partition.json').read_text())
    vi = pop['validation_original_indices']
    val = [val_all[i] for i in vi]
    fit = [val[i] for i in part['fitting']]; tune = [val[i] for i in part['tuning']]
    ds = datasets.load(name, data_profile='research')
    B = incidence(ds.train, ds.n_items); f = np.asarray(B.sum(0)).ravel().astype(np.float64); m = B.shape[0]
    ft = torch.tensor(f, dtype=torch.float32); size = np.asarray(B.sum(1)).ravel()
    fit_s = fit[:600] if ds.music else fit
    frozen_m = json.loads(Path(f'results/sufficiency/{name}/frozen_methods.json').read_text())
    refs = {}
    extra = Path(f'checkpoints/sufficiency/{name}/extra/validation')
    for label in ['reference'] + (['E3_history_frequency'] if ds.music else []):
        meth = frozen_m[label]
        local = {meth['history']: np.load(extra / f"{meth['history']}.npy", mmap_mode='r')} if meth['history'] != 'canonical' else {}
        rows = []
        for ti in part['tuning']:
            z = {}
            for k in meth['keys']:
                if k == 'hist':
                    if local:
                        z[k] = np.asarray(local[meth['history']][ti])
                    else:
                        a = meth['alpha']; o = vi[ti]
                        z[k] = np.asarray(old['pair'][o] if a == 0 else old['simp'][o] if a == 1 else
                                          (1 - a) * old['pair'][o] + a * old['simp'][o])
                else:
                    z[k] = np.asarray(old[k][vi[ti]])
            rows.append(ref_score(z, meth['config'], np.asarray(model.freq, np.float32)))
        R = np.stack(rows, 1)
        refs[label] = dict(per=per_query(R, tune), cold=cold_recall(R, tune, f))
        print(label, refs[label]['per'].mean(0), flush=True)
        del R, rows
    del old
    keys = ['audio'] if ds.music else ['content', 'image', 'desc']
    parts = [np.ones((ds.n_items, 1), np.float32)] + [ds.features[k].astype(np.float32) for k in keys]
    if not ds.music:
        parts.insert(1, np.eye(ds.cat_labels.max() + 1, dtype=np.float32)[ds.cat_labels])
    X = np.concatenate(parts, 1)
    la = lb = None
    extra_f = extra_t = None
    if ds.music:
        la, lb = group_lift(B, ds.cat_labels, f, m), group_lift(B, ds.alb_labels, f, m)
        extra_f = shared(fit_s, ds, la, lb, (ds.n_items, len(fit_s)))
        extra_t = shared(tune, ds, la, lb, (ds.n_items, len(tune)))
    best = (-1, None)
    lams = (1.,) if ds.music else LAMS
    for lam in lams:
        t1 = time.time(); W = fit_w(X, B, f, lam); XW = X @ W; wsec = time.time() - t1
        hat = hat_matrix(fit_s, XW, X, m, extra_f)
        for kernel in KERNELS:
            obs = obs_matrix(fit_s, f, m, B, size, kernel)
            state, v = search(obs, hat, ft, fit_s)
            print(f'lam={lam} kernel={kernel} {state} fit={v:.4f} {time.time()-t0:.0f}s', flush=True)
            if v > best[0]:
                best = (v, dict(state, lam=lam, kernel=kernel, wsec=wsec))
    cfg = best[1]
    W = fit_w(X, B, f, cfg['lam']); XW = X @ W
    obs_f = obs_matrix(fit_s, f, m, B, size, cfg['kernel']); obs_t = obs_matrix(tune, f, m, B, size, cfg['kernel'])
    hat_f = hat_matrix(fit_s, XW, X, m, extra_f); hat_t = hat_matrix(tune, XW, X, m, extra_t)
    rows = {}
    def record(label, S, extra_fields=None):
        per = per_query(S, tune)
        rows[label] = dict(extra_fields or {}, tune=per.mean(0).tolist(), cold=cold_recall(S, tune, f),
                           **{f'vs_{r}': boot(per, refs[r]['per']) for r in refs})
        print(label, json.dumps(rows[label], default=float), flush=True)

    record('model', combine(obs_t, hat_t, ft, cfg['kappa'], cfg['tau'], cfg['a']).numpy(), cfg)
    # tau fixed at the derived value
    s1, _ = search(obs_f, hat_f, ft, fit_s, taus=[1.])
    record('model tau=1', combine(obs_t, hat_t, ft, s1['kappa'], s1['tau'], s1['a']).numpy(), s1)
    # raw similarity in the same slot
    Xr = X[:, 1:]
    seeds = np.asarray(sorted({i for p in fit_s[:200] for i in p[0]}))[:2000]
    sims = Xr @ Xr[seeds].T
    Ac = (B.T @ B[:, seeds]).toarray().astype(np.float32); Ac[seeds, np.arange(len(seeds))] = 0
    Ec = f[:, None] * f[seeds][None] / m
    c = float((Ac * sims).sum() / ((Ec * sims * sims).sum() + 1e-9)); del sims, Ac, Ec
    hr_f = hat_matrix(fit_s, c * Xr / m, Xr, m, extra_f); hr_t = hat_matrix(tune, c * Xr / m, Xr, m, extra_t)
    s2, _ = search(obs_f, hr_f, ft, fit_s)
    record('raw similarity W=cI', combine(obs_t, hr_t, ft, s2['kappa'], s2['tau'], s2['a']).numpy(), s2)
    del hr_f, hr_t
    # additive control
    z = lambda t: (t - t.mean(0, keepdim=True)) / (t.std(0, keepdim=True) + 1e-9)
    Zf = (z(torch.log(FLOOR + obs_f)), z(torch.log(FLOOR + hat_f)), z(torch.log1p(ft))[:, None])
    Zt = (z(torch.log(FLOOR + obs_t)), z(torch.log(FLOOR + hat_t)), z(torch.log1p(ft))[:, None])
    grid = (0, .25, .5, 1, 1.5, 2); bv, bw = -1, None
    for a in grid:
        for b in grid:
            for d in grid:
                if a + b + d == 0: continue
                r = m20(a * Zf[0] + b * Zf[1] + d * Zf[2], fit_s)[1]
                if r > bv: bv, bw = r, (a, b, d)
    record('additive, same inputs', (bw[0] * Zt[0] + bw[1] * Zt[1] + bw[2] * Zt[2]).numpy(), dict(w=bw))
    del Zf, Zt
    # no descriptions / no counts
    s3, _ = search(obs_f, torch.zeros_like(obs_f), ft, fit_s)
    record('counts only', combine(obs_t, torch.zeros_like(obs_t), ft, 0.0001, s3['tau'], s3['a']).numpy(), s3)
    LAM_FOR_NULL[0] = cfg['lam']
    spectrum = spectrum_test(ds, B, f, val, X, W, np.random.default_rng(11))
    print('spectrum', spectrum, flush=True)
    frozen = dict(dataset=name, keys=keys, cfg=cfg, tau1=s1, la=la, lb=lb, frozen_at=time.time(),
                  note='Selected on fitting; tuning reported. Test not read when this file was written.')
    Path(outdir, f'{name}_frozen.json').write_text(json.dumps(frozen, indent=1, default=float))
    Path(outdir, f'{name}_val.json').write_text(json.dumps(dict(rows=rows, spectrum=spectrum, refs={k: dict(mean=v['per'].mean(0).tolist(), cold=v['cold']) for k, v in refs.items()}), indent=1, default=float))
    print('total', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
