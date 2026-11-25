"""Summarize the CLHE baselines (experiments.clhe_baselines.run) on the clean test queries used in the paper.

Per dataset and model: test NDCG@20 and Recall@20 on clean test queries (a query is clean when its complete bundle is in
neither training nor validation; dials triad.data), mean of the three runs per query, then over queries. MultiVAE: the
KL weight (0 or 0.2) with the higher mean validation NDCG@20. Per-query rows are matched by bundle id: CLHE's loader
reports the bundle id, and the dials test pairs are in sorted bundle-id order.

Writes results/clhe_baselines/summary.json.
cd ../bundlefuse && PYTHONPATH=.:../dials .venv/bin/python -m experiments.clhe_baselines.summary
"""
import json
from pathlib import Path
import numpy as np
from triad import data

R = Path('results/clhe_baselines');DIALS = Path(__file__).resolve().parents[3]/'dials'
ROOT = Path(__file__).resolve().parents[3]
FOLDERS = dict(electronic='electronic_clean_blip', food='food_clean_blip', clothing='clothing_clean_blip',
               pog_dense='pog_dense', pog='pog', spotify_sparse='spotify_sparse', spotify='spotify')
MODELS = ('MultiDAE', 'MultiVAE', 'BiLSTM', 'HyperGraph', 'Transformer')


def test_ids(name):
    """Test bundle ids in the order of dials' test pairs (sorted)."""
    if name in ('electronic', 'food', 'clothing'):
        path = ROOT/'baselines'/'CLHE'/'datasets'/FOLDERS[name]/'bi_test_input.txt'
    else:
        path = ROOT/'bundlemllm'/'datasets'/name/'bi_test_input.txt'
    return sorted({int(line.split(',')[0]) for line in open(path) if line.strip()})


def runs(folder, key):
    return [json.loads(p.read_text()) for p in sorted((R/folder).glob(f'{key}_seed*.json'))]


def main():
    out = {}
    for name, folder in FOLDERS.items():
        if not (R/folder).exists():
            continue
        ds = data.load(name, DIALS/'cache');ids = test_ids(name);clean = [ids[i] for i in ds.clean_test]
        row = {}
        for model in MODELS:
            keys = ['MultiVAE_a0', 'MultiVAE_a0.2'] if model == 'MultiVAE' else [model]
            options = [(k, runs(folder, k)) for k in keys];options = [(k, r) for k, r in options if len(r) == 3]
            if not options:
                continue
            key, rs = max(options, key=lambda kr: np.mean([r['validation'][1] for r in kr[1]]))
            per = []
            for r in rs:
                q = {int(b): (rec, nd) for b, rec, nd in r['queries']}
                per.append([q[b] for b in clean])
            per = np.mean(per, 0)
            row[model] = dict(setting=key, recall20=float(per[:, 0].mean()), ndcg20=float(per[:, 1].mean()),
                              queries=len(clean), epochs=[r['epoch'] for r in rs], parameters=rs[0]['parameters'])
        out[name] = row
        print(name, {m: round(v['ndcg20'], 4) for m, v in row.items()}, flush=True)
    (R/'summary.json').write_text(json.dumps(dict(note=__doc__, datasets=out), indent=1))


if __name__ == '__main__':
    main()
