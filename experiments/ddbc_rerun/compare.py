"""DDBC rerun (released checkpoints, Kaggle) against the frozen DIALS, paired, on DDBC's protocol and its queries.

Inputs:
  DDBC   results/ddbc_rerun/<case>.npz from the Kaggle kernel (experiments/ddbc_rerun, ddbc_dump.py): for every query,
         the CLHE-space features of the items DDBC generated, as DDBC's Evaluator.token2feat gives them, and the hits
         of DDBC's own retrieval on its own random pools (to check the rerun against its published Table 1).
  DIALS  the frozen seed-0, 1, 2 models (../dials/runs), scores averaged over the seeds' set predictions.
Pools: experiments.ddbc_protocol.run.score_sets draws (seeded per query and draw, 5 draws, targets + 99|targets|
random items outside the window). The same pools for both methods.
  DIALS: the top-|targets| pool items by score (as before).
  DDBC: its own retrieval rule (Evaluator.retrive_item): for each generated item in turn, the pool item with the
        highest cosine similarity in CLHE space, removed from the pool once picked.
Full catalogue (no pool; our extension of DDBC's retrieval, not in its paper): the same rules over every item except
the seeds. Per query F1 = hits/|targets|, Jaccard = hits/(2|targets| - hits); paired cluster bootstrap and sign flips
on DIALS - DDBC (clusters: identical complete bundles), Holm over the pool tests of the cases present (2 per case:
F1, Jaccard; 8 when all four cases have run).

PYTHONPATH=.:../dials .venv/bin/python -m experiments.ddbc_rerun.compare
"""
import triad  # pins BLAS threads before numpy loads
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from triad import data
from triad import train as T
from triad.evaluate import score_batches
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
import os
from experiments.fair_prior.no_bonus import wrap
NO_BONUS = os.environ.get('TRIAD_NO_BONUS') == '1'  # TRIAD without the unseen-item bonus; writes comparison_no_bonus.json
from experiments.ddbc_protocol.run import CASES, DRAWS, RATIO, PUBLISHED, queries, score_sets
from experiments.mechanisms.statistics import paired_comparison, holm_adjust

OUT = Path('results/ddbc_rerun');RUNS = Path(DATA_ROOT)/'dials'/'runs';CACHE = Path(DATA_ROOT)/'dials'/'cache'
ARCHIVE = Path(DATA_ROOT)/'ddbc_data'/'download'/'ddbc.bin'


def clhe(name):
    """DDBC's item features (clhe.pt from its released archive), L2-normalized rows."""
    with zipfile.ZipFile(ARCHIVE) as z, z.open(f'datasets/{name}/clhe.pt') as f:
        x = torch.load(f, map_location='cpu', weights_only=True).float().numpy()
    return x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def greedy(generated, candidates, features):
    """DDBC's retrieval: each generated item takes the most cosine-similar remaining candidate."""
    g = generated/np.maximum(np.linalg.norm(generated, axis=1, keepdims=True), 1e-12)
    sims = g@features[candidates].T;taken = np.zeros(len(candidates), bool);picked = []
    for row in sims:
        row = np.where(taken, -np.inf, row);j = int(np.argsort(-row)[0]);taken[j] = True;picked.append(candidates[j])
    return picked


def ddbc_sets(run, qs, n, features, rng_seed=20261109):
    """Per query [pool F1, pool Jaccard, full-catalogue F1, full-catalogue Jaccard] for DDBC's generated items."""
    out = []
    for (b, s, t, _), gen, target in zip(qs, run['gen_feat'], run['target_id']):
        assert sorted(target.tolist()) == t, b  # DDBC's targets are our targets, row by row
        m = len(t);tset = set(t);window = np.asarray(sorted(set(s) | tset))
        h = len(tset & set(greedy(gen, np.setdiff1d(np.arange(n), np.asarray(s)), features)))
        pool_scores = []
        for d in range(DRAWS):
            rng = np.random.default_rng([rng_seed, b, d]);others = np.setdiff1d(np.arange(n), window, assume_unique=True)
            pool = np.concatenate([np.asarray(t), rng.choice(others, (RATIO - 1)*m, replace=False)])
            hp = len(tset & set(greedy(gen, pool, features)));pool_scores.append((hp/m, hp/(2*m - hp)))
        out.append([*np.mean(pool_scores, 0), h/m, h/(2*m - h)])
    return np.asarray(out)


def dials_sets(name, ds, qs):
    """Frozen DIALS, seeds 0, 1, 2: per-query set metrics, averaged over the seeds."""
    per_seed = []
    for seed in (0, 1, 2):
        model = (wrap(T.load) if NO_BONUS else T.load)(ds, RUNS/name/f'seed{seed}', CACHE);rows = [];start = 0
        for block in score_batches(model, [q[1] for q in qs], 32):
            rows.append(score_sets(block, qs[start:start + len(block)], ds.n_items, 20261109));start += len(block)
        per_seed.append(np.concatenate(rows))
    return np.mean(per_seed, 0)


def main():
    torch.set_num_threads(4);report = dict(note=__doc__, cases={});records = []
    for name, folder, k in CASES:
        case = f'{name}_k{k}'
        if not (OUT/f'{case}.npz').exists():print(case, 'not run yet, skipped', flush=True);continue
        run = np.load(OUT/f'{case}.npz');ds = data.load(name, CACHE);qs = queries(name, folder, k, ds)
        assert len(qs) == len(run['gen_feat']), case
        m = run['target_id'].shape[1];own = run['own_hits']
        ddbc = ddbc_sets(run, qs, ds.n_items, clhe(name));dials_ = dials_sets(name, ds, qs)
        clusters = [str(tuple(sorted(set(s) | set(t)))) for _, s, t, _ in qs];clean = np.asarray([not q[3] for q in qs])
        summary = lambda a, i=slice(None): dict(pool_f1=a[i, 0].mean(), pool_jaccard=a[i, 1].mean(), full_f1=a[i, 2].mean(), full_jaccard=a[i, 3].mean())
        report['cases'][case] = dict(queries=len(qs), ddbc_own_pools=dict(f1=float((own/m).mean()), jaccard=float((own/(2*m - own)).mean())),
                                     published=PUBLISHED[(name, k)]['DDBC'], ddbc=summary(ddbc), dials=summary(dials_),
                                     ddbc_clean=summary(ddbc, clean), dials_clean=summary(dials_, clean),
                                     generated_codes_that_are_items=float((run['gen_id'] >= 0).mean()), kaggle_seconds=float(run['seconds']))
        for j, metric in ((0, 'pool_f1'), (1, 'pool_jaccard')):
            records.append(dict(paired_comparison(dials_[:, j], ddbc[:, j], clusters, bootstrap=2000, permutations=10000, seed=20261117),
                                case=case, metric=metric))
        r = report['cases'][case]
        print(f"{case}: DDBC own pools F1 {r['ddbc_own_pools']['f1']:.3f} (published {r['published'][0]}) | our pools DDBC F1 {r['ddbc']['pool_f1']:.3f} "
              f"Jacc {r['ddbc']['pool_jaccard']:.3f}, DIALS F1 {r['dials']['pool_f1']:.3f} Jacc {r['dials']['pool_jaccard']:.3f} | "
              f"full catalogue DDBC {r['ddbc']['full_f1']:.3f} DIALS {r['dials']['full_f1']:.3f}", flush=True)
    holm_adjust(records, len(records));report['tests'] = records;write_json(OUT/('comparison_no_bonus.json' if NO_BONUS else 'comparison.json'), report)
    for t in records:print(t['case'], t['metric'], f"DIALS - DDBC {t['delta']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] p_holm {t['p_holm']:.4f}")


if __name__ == '__main__':
    main()
