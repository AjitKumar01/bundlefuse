"""CLHE on the warm and cold clean test queries of Electronic, Food and Clothing (definitions of cold_sparse_check.py).

CLHE = our reruns (results/clhe_blip, preset chosen on validation, results/clhe_blip/chosen.json), mean of the three
runs per query. Their per-query rows have no bundle id: they are in the order of bi_test_input.txt, which is sorted by
bundle id, as are the dials test pairs; on these datasets every test query is clean. Checked: the overall means equal
Table 2 and the warm/cold counts equal cold_sparse_check.json. POG and Spotify reruns kept only averages, so no rows.

Writes results/small_data/clhe_cold.json.
cd ../bundlefuse && VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=.:../dials .venv/bin/python -m experiments.small_data.clhe_cold
"""
import triad
import json, glob
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.provenance import write_json

ROOT = Path(__file__).resolve().parents[3]


def main():
    chosen = json.load(open('results/clhe_blip/chosen.json'))
    check = json.load(open('results/small_data/cold_sparse_check.json'))['rows']
    out = {}
    for name in ('electronic', 'food', 'clothing'):
        folder = f'{name}_clean_blip';ds = data.load(name, ROOT/'dials'/'cache')
        ids = [int(l.split(',')[0]) for l in open(ROOT/'baselines'/'CLHE'/'datasets'/folder/'bi_test_input.txt') if l.strip()]
        assert ids == sorted(ids) and len(ids) == len(ds.test_pairs) == len(ds.clean_test)
        f = np.bincount(np.concatenate([np.unique(b) for b in ds.train]), minlength=ds.n_items)
        runs = sorted(glob.glob(f'results/clhe_blip/{folder}/{chosen[folder]}_seed*/result.json'))
        per = np.mean([json.load(open(p))['queries'] for p in runs], 0)
        warm = np.array([all(f[x] > 0 for x in set(s) | set(g)) for s, g in ds.test_pairs])
        assert int(warm.sum()) == check[name]['groups']['warm']['n']
        out[name] = dict(runs=len(runs), preset=chosen[folder], all=float(per[:, 1].mean()),
                         warm=dict(n=int(warm.sum()), ndcg20=float(per[warm, 1].mean())),
                         cold=dict(n=int((~warm).sum()), ndcg20=float(per[~warm, 1].mean())))
        print(name, out[name], flush=True)
    write_json(Path('results/small_data/clhe_cold.json'), dict(note=__doc__, datasets=out))


if __name__ == '__main__':
    main()
