"""Factorial diagnosis of the Spotify gap. Fitting queries only (first 600)."""
import json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
from bundlefuse import datasets
from bundlefuse.metrics import ranking_metrics
from cpm2 import incidence, fit_w, group_lift, FLOOR
from cpm3 import obs_matrix, hat_matrix, combine
from experiments.novelty.data import load_inputs

name = 'spotify'
model, val_all, _u, old, _f, _i = load_inputs(name); del _u
pop = json.loads(Path(f'results/sufficiency/{name}/evaluation_population.json').read_text())
part = json.loads(Path(f'results/sufficiency/{name}/partition.json').read_text())
vi = pop['validation_original_indices']
val = [val_all[i] for i in vi]
idx = part['fitting'][:600]
pairs = [val[i] for i in idx]
ds = datasets.load(name, data_profile='research')
B = incidence(ds.train, ds.n_items); f = np.asarray(B.sum(0)).ravel().astype(np.float64); m = B.shape[0]
size = np.asarray(B.sum(1)).ravel(); n = ds.n_items
ft = torch.tensor(f, dtype=torch.float32)
extra = Path(f'checkpoints/sufficiency/{name}/extra/validation')
amp = np.load(extra / 'amp_selected.npy', mmap_mode='r')


def col(a, orig=True):
    return np.stack([np.asarray(a[vi[i] if orig else i]) for i in idx], 1)


def z(x):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, ddof=1, keepdims=True) + 1e-9)


def report(label, S):
    r, v = ranking_metrics(np.asarray(S), pairs, Ks=(20,))[20]
    print(f'{label:58s} R {r:.4f} N {v:.4f}', flush=True)
    return v


H = col(amp, False); homo = col(old['homo']); rao = col(old['rao'])
print('A. current recipe columns (already standardized)')
report('amplified history alone', H)
report('homo alone', homo)
report('rao alone', rao)
report('pop alone', np.tile(np.log1p(f)[:, None], (1, len(pairs))))
freq = f[:, None]
rf = freq / (freq + .05); rq = freq / (freq + 5.)
P = (.25 * homo + .25 * rao) * 1.425552263325423
gate = lambda p, h: p + rf * (.25 * homo + .25 * rao - p) + rq * .25 * h
report('current recipe (history-frequency gate)', gate(P, H))
report('current recipe without rao', gate(.25 * homo * 1.4256, H) - rf * .25 * rao)

print('B. new components')
obs = obs_matrix(pairs, f, m, B, size, (.5, .5, 2, 100))
Wt = fit_w(np.concatenate([np.ones((n, 1), np.float32), ds.features['audio']], 1), B, f, 1.)
X = np.concatenate([np.ones((n, 1), np.float32), ds.features['audio']], 1); XW = X @ Wt
audio_hat = hat_matrix(pairs, XW, X, m)
la, lb = group_lift(B, ds.cat_labels, f, m), group_lift(B, ds.alb_labels, f, m)
print('same-artist lift', la, 'same-album lift', lb)
same_a = torch.zeros_like(obs); same_b = torch.zeros_like(obs); art_w = torch.zeros_like(obs); alb_w = torch.zeros_like(obs)
na = np.bincount(ds.cat_labels); nb = np.bincount(ds.alb_labels)
for j, (s, _) in enumerate(pairs):
    s = list(s)
    ca = np.bincount(ds.cat_labels[s], minlength=len(na))[ds.cat_labels] / len(s)
    cb = np.bincount(ds.alb_labels[s], minlength=len(nb))[ds.alb_labels] / len(s)
    same_a[:, j] = torch.tensor(ca, dtype=torch.float32); same_b[:, j] = torch.tensor(cb, dtype=torch.float32)
    art_w[:, j] = torch.tensor(ca / na[ds.cat_labels], dtype=torch.float32)
    alb_w[:, j] = torch.tensor(cb / nb[ds.alb_labels], dtype=torch.float32)
report('new count term, tau=1', torch.log(FLOOR + obs) + torch.log(ft[:, None] + 1))
report('new count term raw vs amp col rank check', obs * ft[:, None])
report('audio prior alone, tau=.5', torch.log(FLOOR + audio_hat) + .5 * torch.log(ft[:, None] + 1))
report('same artist alone', same_a + 1e-6 * torch.log1p(ft)[:, None])
report('same album alone', same_b + 1e-6 * torch.log1p(ft)[:, None])
share = (same_a + 2 * same_b)
report('artist + 2 album, popularity tiebreak', share + 1e-6 * torch.log1p(ft)[:, None])

print('C. swap inputs into the current combiner')
zo = z(torch.log(FLOOR + obs).numpy()); zh = z(torch.log(FLOOR + audio_hat).numpy())
zshare = z((same_a * (la - 1) + same_b * (lb - la)).numpy())
report('current prior + my count term (log z)', gate(P, zo))
report('current prior + my raw count*f (z)', gate(P, z((obs * ft[:, None]).numpy())))
report('homo + rao prior replaced by my hat + share', gate((.25 * zh + .25 * zshare) * 1.4256, H) - rf * (.25 * homo + .25 * rao) + rf * (.25 * zh + .25 * zshare))
report('homo kept, rao replaced by my audio hat', gate((.25 * homo + .25 * zh) * 1.4256, H) - rf * .25 * rao + rf * .25 * zh)

print('D. new combiner, prior variants')
best = {}
for label, hat in (('audio + constant-lift share', (audio_hat + (la - 1) * same_a + (lb - la) * same_b).clamp_min(0)),
                   ('audio only', audio_hat),
                   ('share only', ((la - 1) * same_a + (lb - la) * same_b + 1).clamp_min(0))):
    bv, bc = -1, None
    for k in (.1, 1, 5, 25, 100, 400, 1600, 6400):
        for t in (.25, .5, .75, 1):
            for a in (1, 25):
                v = ranking_metrics(combine(obs, hat, ft, k, t, a).numpy(), pairs, Ks=(20,))[20][1]
                if v > bv: bv, bc = v, (k, t, a)
    print(f'{label:58s} best N {bv:.4f} at kappa,tau,a={bc}', flush=True)

print('E. why the log combiner may fail: linear combination in the same units')
rho = lambda k: ft[:, None] / (ft[:, None] + k)
hat = (audio_hat + (la - 1) * same_a + (lb - la) * same_b).clamp_min(0)
Pc = obs * ft[:, None] / m       # estimated P(c|S)
Ph = hat * (ft[:, None] + 1) / m
for k in (.1, 1, 5, 25, 100):
    report(f'P(c|S) = rho obs + (1-rho) hat, popularity inside, k={k}', torch.log(1e-12 + rho(k) * Pc + (1 - rho(k)) * Ph))
for w in (.01, .03, .1, .3, 1):
    report(f'P(c|S) = obs + w*hat, no gate, w={w}', Pc + w * Ph)
