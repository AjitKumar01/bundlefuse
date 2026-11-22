"""Compare the small-data variants with plain TRIAD and the best simple baseline on Electronic, Food and Clothing.

Per condition (n50 ... full), test NDCG@20 (mean over seeds 0-2, per query) of:
  triad     plain TRIAD (results/bundle_budget_retail/runs; full: ../dials/runs, the frozen runs)
  v2        one model with a principled start (experiments.small_data.triad_v2)
  v2_64     the same with rank-64 content maps (41,368 parameters)
  v2_64a    rank 64, content = fixed full-d similarity + learned correction starting at 0 (41,368 parameters)
  residual  TRIAD on top of the tuned simple scorer (experiments.small_data.residual)
  frozen    TRIAD with frozen content maps
  residual_tp  residual with the anchor's prior from training bundles only
  residual_cv  residual with the anchor tuned on training bundles only (no validation query sets any anchor parameter)
  select    per condition, plain TRIAD or residual_cv, whichever has the higher mean validation NDCG@20 (no test data)
  simple    the paper's best simple baseline for the condition (results/fair_prior/budget_comparison.json: chosen on
            validation, frequency prior allowed)
Paired tests (bootstrap + sign flips, one cluster per query) of each variant minus simple, Holm within each variant over
the 15 conditions. Writes results/small_data/comparison.json.

cd ../bundlefuse && PYTHONPATH=.:../dials .venv/bin/python -m experiments.small_data.compare
"""
import json
from pathlib import Path
import numpy as np
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust

R = Path('results');DIALS = Path(__file__).resolve().parents[3]/'dials'
ECOM = ('electronic', 'food', 'clothing');CONDS = ('n50', 'n100', 'n300', 'n600', 'full')


def triad(name, cond):
    root = DIALS/'runs'/name if cond == 'full' else R/'bundle_budget_retail'/'runs'/name/cond
    t = [json.loads((root/f'seed{s}'/'test.json').read_text())['queries'] for s in (0, 1, 2)]
    v = [json.loads((root/f'seed{s}'/'selected.json').read_text())['validation']['ndcg20'] for s in (0, 1, 2)]
    return np.mean([[q['ndcg20'] for q in run] for run in t], 0), float(np.mean(v))


def variant(kind, name, cond):
    r = json.loads((R/'small_data'/kind/f'{name}_{cond}.json').read_text())
    return np.array(r['queries'])[:, 1], float(np.mean([x['validation'] for x in r['runs']]))


def simple(name, cond):
    best = json.loads((R/'fair_prior'/'budget_comparison.json').read_text())['families']['ecommerce']['rows'][f'{name}:{cond}']['best_simple']
    method, prior = best.replace('+prior', ''), best.endswith('+prior')
    m = json.loads((R/'fair_prior'/(f'{name}.json' if cond == 'full' else f'{name}_{cond}.json')).read_text())['methods'][method]
    return np.array(m['queries' if prior else 'queries_without_prior'])[:, 1], best


def main():
    rows, tests = {}, {k: [] for k in ('triad', 'v2', 'v2_64', 'v2_64a', 'residual', 'residual_tp', 'residual_cv', 'frozen', 'select')}
    for cond in CONDS:
        for name in ECOM:
            s, best = simple(name, cond);t, tv = triad(name, cond);r, rv = variant('residual', name, cond);f, fv = variant('frozen', name, cond)
            rt, rtv = variant('residual_tp', name, cond);rc, rcv = variant('residual_cv', name, cond);v2, v2v = variant('v2', name, cond);v64, v64v = variant('v2_64', name, cond);v64a, v64av = variant('v2_64a', name, cond)
            sel, pick = (rc, 'residual_cv') if rcv > tv else (t, 'triad')
            q = dict(triad=t, v2=v2, v2_64=v64, v2_64a=v64a, residual=r, residual_tp=rt, residual_cv=rc, frozen=f, select=sel);clusters = [str(i) for i in range(len(s))]
            rows[f'{name}:{cond}'] = dict(simple=float(s.mean()), simple_method=best, select_picks=pick,
                                          validation=dict(triad=tv, v2=v2v, v2_64=v64v, v2_64a=v64av, residual=rv, residual_tp=rtv, residual_cv=rcv, frozen=fv), **{k: float(v.mean()) for k, v in q.items()})
            for k, v in q.items():
                tests[k].append(dict(paired_comparison(v, s, clusters, bootstrap=2000, permutations=10000, seed=20261121), condition=f'{name}:{cond}'))
    for k in tests:holm_adjust(tests[k], len(tests[k]))
    write_json(R/'small_data'/'comparison.json', dict(note=__doc__, rows=rows, tests=tests))
    print(f"{'condition':18s} {'simple':>7s} {'triad':>7s} {'v2':>7s} {'resid':>7s} {'res_tp':>7s} {'res_cv':>7s} {'frozen':>7s} {'select':>7s} pick")
    by = {k: {t['condition']: t for t in v} for k, v in tests.items()}
    for c, r in rows.items():
        mark = lambda k: ('*' if by[k][c]['p_holm'] < .05 else ' ')
        print(f"{c:18s} {r['simple']:.4f} " + ' '.join(f"{r[k]:.4f}{mark(k)}" for k in ('triad', 'v2', 'residual', 'residual_tp', 'residual_cv', 'frozen', 'select')) + f" {r['select_picks']}")
    print('* : variant - simple significant after Holm (15 tests per variant)')


if __name__ == '__main__':
    main()
