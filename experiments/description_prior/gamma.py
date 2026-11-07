"""Linear versus Gamma-Poisson posterior, fitting queries only.

linear: (obs_f + kappa mu)/(f + kappa)    prior shape kappa*mu, rate kappa
gamma:  (obs_f + kappa)/(f + kappa/mu)    prior shape kappa, rate kappa/mu
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from bundlefuse import datasets
from bundlefuse.metrics import ranking_metrics
from cpm2 import incidence, fit_w, group_lift
from cpm3 import obs_matrix, hat_matrix
from cpm4 import shared

EPS = 1e-3
name, lam, kernel = sys.argv[1], float(sys.argv[2]), eval(sys.argv[3])
pop = json.loads(Path(f'results/sufficiency/{name}/evaluation_population.json').read_text())
part = json.loads(Path(f'results/sufficiency/{name}/partition.json').read_text())
ds = datasets.load(name, data_profile='research')
val = [ds.val_pairs[i] for i in pop['validation_original_indices']]
fit = [val[i] for i in part['fitting']]
fit = fit[:600] if ds.music else fit
B = incidence(ds.train, ds.n_items); f = np.asarray(B.sum(0)).ravel().astype(np.float64); m = B.shape[0]
size = np.asarray(B.sum(1)).ravel(); ft = torch.tensor(f, dtype=torch.float32)
keys = ['audio'] if ds.music else ['content', 'image', 'desc']
parts = [np.ones((ds.n_items, 1), np.float32)] + [ds.features[k].astype(np.float32) for k in keys]
if not ds.music:
    parts.insert(1, np.eye(ds.cat_labels.max() + 1, dtype=np.float32)[ds.cat_labels])
X = np.concatenate(parts, 1); W = fit_w(X, B, f, lam); XW = X @ W
extra = None
if ds.music:
    la, lb = group_lift(B, ds.cat_labels, f, m), group_lift(B, ds.alb_labels, f, m)
    extra = shared(fit, ds, la, lb, (ds.n_items, len(fit)))
hat = hat_matrix(fit, XW, X, m, extra)
obs = obs_matrix(fit, f, m, B, size, kernel)
F = ft[:, None]
ND = lambda S: ranking_metrics(S.numpy(), fit, Ks=(20,))[20]
counts = obs * F          # m P(c|S)


def run(label, form):
    best = (-1, None)
    for k in (.01, .1, 1, 5, 25, 100, 400, 1600):
        for t in (0, .25, .5, .75, 1, 1.25, 1.5):
            for a in (.1, 1, 25):
                v = ND(torch.log(EPS + form(k)) + t * torch.log(F + a))
                if v[1] > best[0]:
                    best = (v[1], (k, t, a), v[0])
    print(f'{name} {label:30s} N {best[0]:.4f} R {best[2]:.4f} kappa,tau,a={best[1]}', flush=True)


run('linear', lambda k: (counts + k * hat) / (F + k))
run('gamma', lambda k: (counts + k) / (F + k / (hat + EPS)))
run('gamma, flat mu=1 (no description)', lambda k: (counts + k) / (F + k))
