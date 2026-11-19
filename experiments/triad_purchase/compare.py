"""Electronic, Food, Clothing: TRIAD + purchase log against the best simple baseline given the same log.

Best simple with the log: chosen on validation NDCG@20 over every method and lambda of experiments.triad_purchase.
baselines together with the pool without the log (experiments.dials_blip.compare.simple_q, lambda = 0), so the log is
used only where validation prefers it. Three families, each Holm-corrected over its 6 tests (3 datasets x NDCG@20,
Recall@20), paired cluster bootstrap and sign flips as in the paper (clusters: identical complete bundles; on these
splits every bundle is unique):
  triad_purchase_vs_simple_log   does TRIAD + purchase lead the simple baselines when both have the log?
  triad_vs_simple_log            TRIAD without the log against simple baselines with it
  simple_log_vs_simple           does the log help the simple baselines?
Writes results/triad_purchase/compare_simple.json.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.triad_purchase.compare
"""
import triad  # pins BLAS threads before numpy loads
import json
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust
from experiments.dials_blip.compare import simple_q, CONTENT
from .run import OUT, CACHE, DIALS, SEEDS
from .baselines import LAMBDAS, OUT as BASE


def model_q(root, name):
    rs = [json.loads((Path(root)/name/f'seed{s}'/'test.json').read_text())['queries'] for s in SEEDS]
    return np.mean([[[q['recall20'], q['ndcg20']] for q in r] for r in rs], 0)


def simple_log(name):
    """(best simple without the log: per-query [R, N], method) and the best over that pool plus the log runs."""
    pool = {}  # the lambda = 0 pool, exactly as experiments.dials_blip.compare.simple_q builds it
    for k, v in json.loads(Path(f'results/simple_baselines/{name}/summary.json').read_text())['test'].items():
        if k not in CONTENT:pool[k] = (v['validation_ndcg20'], np.asarray(v['queries']), v['params'])
    for root, keep in (('results/bundle_budget_retail', lambda k: k not in CONTENT), ('results/dials_blip', lambda k: True)):
        for k, v in json.loads((Path(root)/'baselines'/name/'full.json').read_text())['methods'].items():
            if keep(k) and (k not in pool or v['validation_ndcg20'] > pool[k][0]):pool[k] = (v['validation_ndcg20'], np.asarray(v['queries']), v['params'])
    method = max(pool, key=lambda k: pool[k][0]);s = pool[method][1]
    check, _, _ = simple_q(name);assert np.array_equal(check, s)
    best = ((0., method), pool[method])
    for lam in LAMBDAS:
        for k, v in json.loads((BASE/f'lam{lam}'/'baselines'/name/'full.json').read_text())['methods'].items():
            if v['validation_ndcg20'] > best[1][0]:best = ((lam, k), (v['validation_ndcg20'], np.asarray(v['queries']), v['params']))
    return s, method, best


def main():
    report = dict(note=__doc__, datasets={});families = {k: [] for k in ('triad_purchase_vs_simple_log', 'triad_vs_simple_log', 'simple_log_vs_simple')}
    for name in ('electronic', 'food', 'clothing'):
        pairs = data.load(name, CACHE).test_pairs;clusters = [str(tuple(sorted(set(s) | set(g)))) for s, g in pairs]
        tp, t = model_q(OUT/'runs', name), model_q(DIALS/'runs', name)
        s, method, ((lam, lmethod), (lval, sl, lparams)) = simple_log(name)
        assert len({len(x) for x in (tp, t, s, sl, clusters)}) == 1, name
        report['datasets'][name] = dict(triad_purchase=tp.mean(0).tolist(), triad=t.mean(0).tolist(), simple=s.mean(0).tolist(), simple_method=method,
                                        simple_log=sl.mean(0).tolist(), simple_log_method=lmethod, simple_log_lambda=lam, simple_log_params=lparams,
                                        simple_log_validation_ndcg20=lval)
        for j, metric in enumerate(('recall20', 'ndcg20')):
            for family, (a, b) in (('triad_purchase_vs_simple_log', (tp, sl)), ('triad_vs_simple_log', (t, sl)), ('simple_log_vs_simple', (sl, s))):
                families[family].append(dict(paired_comparison(a[:, j], b[:, j], clusters, bootstrap=2000, permutations=10000, seed=20261118),
                                             dataset=name, metric=metric))
    for records in families.values():holm_adjust(records, len(records))
    report['tests'] = families;write_json(Path(OUT)/'compare_simple.json', report)
    for name, r in report['datasets'].items():
        print(name, 'R/N', {k: np.round(v, 4).tolist() for k, v in r.items() if k in ('triad_purchase', 'triad', 'simple', 'simple_log')},
              r['simple_method'], '| with log:', r['simple_log_method'], 'lambda', r['simple_log_lambda'], r['simple_log_params'])
    for family, records in families.items():
        for t in records:print(f"{family:30s} {t['dataset']:10s} {t['metric']:8s} delta {t['delta']:+.4f} ci [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] p_holm {t['p_holm']:.4f}")


if __name__ == '__main__':
    main()
