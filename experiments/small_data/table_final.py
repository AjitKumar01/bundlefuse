"""Final full-data table: test NDCG@20 on clean test queries (mean of seeds 0-2 per query) of the best simple baseline,
the original TRIAD and the final TRIAD, on all seven datasets, with paired tests (bootstrap + sign flips, one cluster
per query). Holm over the 10 tests of the first five datasets; for each Spotify set, Holm over its own 2 tests.

Sources (features match within each row):
  best simple  results/fair_prior/budget_comparison.json picks the method (chosen on validation, popularity prior
               allowed); its per-query clean-test scores are in results/fair_prior/<dataset>.json
  original     the original TRIAD (rank 64, raw-count start, no per-query scaling), trained with the archived dials
               code (branch archive/original-triad): ../dials/runs_original_triad/<dataset>/seed<s>/test.json. Its
               e-commerce features are the BLIP title features of the same data loader (retraining it with
               experiments.small_data.budget_rerun reproduces these runs exactly).
  final        the final TRIAD (full rank, simple-scorer start, per-query scaling):
               e-commerce: ../dials/runs/<dataset>/seed<s>/test.json (dials main code, learning rates halved);
               POG_DENSE, POG: results/small_data/v2/<dataset>_full.json (experiments.small_data.residual, variant v2);
               Spotify sets: results/small_data/kaggle_v2/<dataset>/seed<s>.json (kaggle_triad_v2/run.py)
Writes results/small_data/table_final.json.

cd ../bundlefuse && PYTHONPATH=.:../dials_original .venv/bin/python -m experiments.small_data.table_final
"""
import json
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust

R = Path('results');DIALS = Path('../dials');ECOM = ('electronic', 'food', 'clothing')
NAMES = ('electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify')
FAMILY = dict(electronic='ecommerce', food='ecommerce', clothing='ecommerce', pog_dense='fashion', pog='fashion',
              spotify_sparse='music', spotify='music')


def seeds_mean(files):return np.mean([[q['ndcg20'] for q in json.loads(Path(f).read_text())['queries']] for f in files], 0)


def simple(name):
    best = json.loads((R/'fair_prior'/'budget_comparison.json').read_text())['families'][FAMILY[name]]['rows'][f'{name}:full']['best_simple']
    m = json.loads((R/'fair_prior'/f'{name}.json').read_text())['methods'][best.replace('+prior', '')]
    return np.array(m['queries' if best.endswith('+prior') else 'queries_without_prior'])[:, 1], best


def original(name, clean):
    root = DIALS/'runs_original_triad'/name
    return seeds_mean([root/f'seed{s}'/'test.json' for s in (0, 1, 2)])[clean]


def final(name, clean):
    if name in ECOM:return seeds_mean([DIALS/'runs'/name/f'seed{s}'/'test.json' for s in (0, 1, 2)])[clean]
    if name in ('pog_dense', 'pog'):return np.array(json.loads((R/'small_data'/'v2'/f'{name}_full.json').read_text())['queries'])[clean, 1]
    runs = [json.loads((R/'small_data'/'kaggle_v2'/name/f'seed{s}.json').read_text()) for s in (0, 1, 2)]
    return np.mean([np.array(r['test'])[:, 1] for r in runs], 0)[clean]


def main():
    rows, tests = {}, []
    for name in NAMES:
        clean = data.load(name, DIALS/'cache').clean_test;s, best = simple(name);o, f = original(name, clean), final(name, clean)
        assert len(s) == len(o) == len(f) == len(clean), (name, len(s), len(o), len(f), len(clean))
        cl = [str(i) for i in range(len(clean))];group = 'spotify_sparse' if name == 'spotify_sparse' else 'spotify' if name == 'spotify' else 'main'
        rows[name] = dict(queries=len(clean), best_simple=float(s.mean()), best_simple_method=best, original=float(o.mean()), final=float(f.mean()))
        for vs, base in (('best_simple', s), ('original', o)):
            tests.append(dict(paired_comparison(f, base, cl, bootstrap=2000, permutations=10000, seed=20261121), dataset=name, versus=vs, group=group))
    for g in ('main', 'spotify_sparse', 'spotify'):
        sub = [t for t in tests if t['group'] == g];holm_adjust(sub, len(sub))
    write_json(R/'small_data'/'table_final.json', dict(note=__doc__, rows=rows, tests=tests))
    by = {(t['dataset'], t['versus']): t for t in tests}
    print(f"{'dataset':15s} {'queries':>7s} {'best simple':>12s} {'original':>9s} {'final':>7s} {'final-simple':>13s} {'final-orig':>11s}  best simple method")
    for n, r in rows.items():
        cell = lambda v: f"{r['final'] - r[v]:+.4f}{'*' if by[(n, v)]['p_holm'] < .05 else ' '}"
        print(f"{n:15s} {r['queries']:7d} {r['best_simple']:12.4f} {r['original']:9.4f} {r['final']:7.4f} {cell('best_simple'):>13s} {cell('original'):>11s}  {r['best_simple_method']}")
    print('* : significant after Holm (10 tests over the first five datasets; 2 per Spotify set)')


if __name__ == '__main__':
    main()
