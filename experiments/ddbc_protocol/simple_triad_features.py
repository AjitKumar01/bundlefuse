"""The simple baseline under DDBC's pool protocol, rescored with exactly TRIAD's item features.

experiments.ddbc_protocol.run scored its simple baseline (the best of results/simple_baselines on validation NDCG@20)
with the content features of an older loader (experiments.neural_lowrank), whose 128-d projection differs from TRIAD's
(item-item cosine differs by up to .58 on Spotify). This matters only for the content+ItemKNN blend (chosen on
Spotify); RP3beta (chosen on POG_DENSE) uses no features and must reproduce the earlier numbers exactly (asserted).
Same queries, pools (rng seed 20261109, five draws) and scoring as run.py. Writes
results/ddbc_protocol/simple_triad_features.json.

cd ../bundlefuse && PYTHONPATH=.:../dials nice -n 10 .venv/bin/python -m experiments.ddbc_protocol.simple_triad_features
"""
import triad  # pins BLAS threads before numpy loads
import json
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.provenance import write_json
from experiments.simple_baselines.methods import Baselines
from experiments.ddbc_protocol.run import CASES, queries, score_sets

DIALS = Path(__file__).resolve().parents[3]/'dials'


def main():
    old = json.loads(Path('results/ddbc_protocol/summary.json').read_text())['cases'];report = dict(note=__doc__, cases={})
    for name, folder, k in CASES:
        ds = data.load(name, DIALS/'cache');n = ds.n_items
        simple = json.loads(Path(f'results/simple_baselines/{name}/summary.json').read_text())['test']
        best = max(simple, key=lambda m: simple[m]['validation_ndcg20']);params = simple[best]['params']
        assert best == old[f'{name}_k{k}']['simple']['method']  # the blend weight may differ: chosen on validation with TRIAD's features
        qs = queries(name, folder, k, ds);model = Baselines(ds.train, n, ds.features if best == 'hybrid' else None)
        sc = np.concatenate([model.scores(best, params, [q[1] for q in qs[a:a + 128]]) for a in range(0, len(qs), 128)])
        r = score_sets(sc, qs, n, 20261109);m = r.mean(0)
        row = dict(method=best, params=params, earlier_params=old[f'{name}_k{k}']['simple']['params'], pool_f1=float(m[0]), pool_jaccard=float(m[1]), full_f1=float(m[2]), full_jaccard=float(m[3]),
                   earlier=dict((key, old[f'{name}_k{k}']['simple'][key]) for key in ('pool_f1', 'pool_jaccard')), per_query=r.tolist())
        if best != 'hybrid':assert abs(row['pool_f1'] - row['earlier']['pool_f1']) < 1e-12, row
        report['cases'][f'{name}_k{k}'] = row
        print(f"{name} k={k} {best}: F1 {row['pool_f1']:.4f} Jaccard {row['pool_jaccard']:.4f} (earlier {row['earlier']['pool_f1']:.4f} / {row['earlier']['pool_jaccard']:.4f})", flush=True)
    write_json(Path('results/ddbc_protocol/simple_triad_features.json'), report)


if __name__ == '__main__':
    main()
