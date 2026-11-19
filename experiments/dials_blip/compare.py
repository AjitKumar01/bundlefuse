"""BLIP text features on Electronic, Food and Clothing: DIALS, CLHE and RaMen, each against its TF-IDF self and DIALS
against each neural baseline.

Inputs (three seeds each, per-query test Recall@20 and NDCG@20, clean graphs):
  DIALS-BLIP   results/dials_blip/<d>/seed{0,1,2}/test.json          (experiments/dials_blip/run.py)
  DIALS-TF-IDF ../dials/runs/<d>/seed{0,1,2}/test.json               (frozen DIALS)
  CLHE-BLIP    results/clhe_blip/<d>_clean_blip/<preset>_seed{2023,0,1}  (Kaggle; preset chosen on validation)
  CLHE-TF-IDF  results/clhe_retail/<d>_clean/<preset>_seed{2023,0,1}     (preset chosen on validation)
  RaMen-BLIP   results/ramen_paper/<d>_clean_blip/<setting>_seed{2306,0,1} (paper version, setting chosen on validation)
  RaMen-TF-IDF results/ramen_paper/<d>_clean/<setting>_seed{2306,0,1}
  Best simple, BLIP: chosen on validation NDCG@20 from one pool, the content methods (content + ItemKNN blend,
               label + ItemKNN + content) rerun with BLIP (results/dials_blip/baselines, experiments/dials_blip/baselines.py)
               and the feature-free methods as before (results/simple_baselines, results/bundle_budget_retail).
Tests: paired cluster bootstrap and sign flips (clusters: identical complete bundles), Holm within each family of six
(3 datasets x 2 metrics): CLHE-BLIP vs CLHE-TF-IDF; RaMen-BLIP vs RaMen-TF-IDF; DIALS-BLIP vs CLHE-BLIP;
DIALS-BLIP vs RaMen-BLIP; DIALS-BLIP vs the best simple baseline with BLIP.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_blip.compare
"""
import glob
import json
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust

FROZEN = Path(DATA_ROOT)/'dials'/'runs';OUT = Path('results/dials_blip')


def dials_q(root):
    return np.mean([[[q[m] for m in ('recall20', 'ndcg20')] for q in json.loads(Path(f).read_text())['queries']]
                    for f in (root/f'seed{s}'/'test.json' for s in (0, 1, 2))], 0)


def clhe_q(folder):
    first = [json.loads(Path(f).read_text()) for f in glob.glob(f'{folder}/*_seed2023/result.json')]
    preset = max(first, key=lambda r: r['validation'][1])['preset']
    runs = [json.loads(Path(f'{folder}/{preset}_seed{s}/result.json').read_text()) for s in (2023, 0, 1)]
    return np.mean([r['queries'] for r in runs], 0), preset


def ramen_q(folder):
    first = [json.loads(Path(f).read_text()) for f in glob.glob(f'{folder}/*_seed2306/result.json')]
    tag = Path(max(glob.glob(f'{folder}/*_seed2306/result.json'), key=lambda f: json.loads(Path(f).read_text())['validation'][1])).parent.name[:-len('_seed2306')]
    runs = [json.loads(Path(f'{folder}/{tag}_seed{s}/result.json').read_text()) for s in (2306, 0, 1)]
    return np.mean([r['queries'] for r in runs], 0), tag


CONTENT = ('hybrid', 'label_knn_content')


def simple_q(name):
    """Best simple baseline with BLIP content, chosen on validation NDCG@20: (per-query [R, N], method, params)."""
    pool = {}
    for k, v in json.loads(Path(f'results/simple_baselines/{name}/summary.json').read_text())['test'].items():
        if k not in CONTENT:pool[k] = (v['validation_ndcg20'], np.asarray(v['queries']), v['params'])
    for root, keep in (('results/bundle_budget_retail', lambda k: k not in CONTENT), ('results/dials_blip', lambda k: True)):
        for k, v in json.loads((Path(root)/'baselines'/name/'full.json').read_text())['methods'].items():
            if keep(k) and (k not in pool or v['validation_ndcg20'] > pool[k][0]):pool[k] = (v['validation_ndcg20'], np.asarray(v['queries']), v['params'])
    method = max(pool, key=lambda k: pool[k][0]);return pool[method][1], method, pool[method][2]


def main():
    report = dict(note=__doc__, datasets={});families = {k: [] for k in ('clhe_blip_vs_clhe_tfidf', 'ramen_blip_vs_ramen_tfidf', 'dials_blip_vs_clhe_blip', 'dials_blip_vs_ramen_blip', 'dials_blip_vs_simple_blip')}
    for name in ('electronic', 'food', 'clothing'):
        pairs = data.load(name).test_pairs;clusters = [str(tuple(sorted(set(s) | set(g)))) for s, g in pairs]
        d_blip, d_tfidf = dials_q(OUT/name), dials_q(FROZEN/name)
        c_blip, p_blip = clhe_q(f'results/clhe_blip/{name}_clean_blip');c_tfidf, p_tfidf = clhe_q(f'results/clhe_retail/{name}_clean')
        r_blip, t_blip = ramen_q(f'results/ramen_paper/{name}_clean_blip');r_tfidf, t_tfidf = ramen_q(f'results/ramen_paper/{name}_clean')
        s_blip, s_method, s_params = simple_q(name)
        assert len({len(x) for x in (d_blip, d_tfidf, c_blip, c_tfidf, r_blip, r_tfidf, s_blip, clusters)}) == 1, name
        report['datasets'][name] = dict(dials_blip=d_blip.mean(0).tolist(), dials_tfidf=d_tfidf.mean(0).tolist(),
                                        clhe_blip=c_blip.mean(0).tolist(), clhe_tfidf=c_tfidf.mean(0).tolist(),
                                        ramen_blip=r_blip.mean(0).tolist(), ramen_tfidf=r_tfidf.mean(0).tolist(),
                                        simple_blip=s_blip.mean(0).tolist(), simple_method=s_method, simple_params=s_params,
                                        clhe_preset=dict(blip=p_blip, tfidf=p_tfidf), ramen_setting=dict(blip=t_blip, tfidf=t_tfidf))
        for j, metric in enumerate(('recall20', 'ndcg20')):
            for family, (a, b) in (('clhe_blip_vs_clhe_tfidf', (c_blip, c_tfidf)), ('ramen_blip_vs_ramen_tfidf', (r_blip, r_tfidf)),
                                   ('dials_blip_vs_clhe_blip', (d_blip, c_blip)), ('dials_blip_vs_ramen_blip', (d_blip, r_blip)),
                                   ('dials_blip_vs_simple_blip', (d_blip, s_blip))):
                families[family].append(dict(paired_comparison(a[:, j], b[:, j], clusters, bootstrap=2000, permutations=10000,
                                                               seed=20261117), dataset=name, metric=metric))
    for records in families.values():holm_adjust(records, len(records))
    report['tests'] = families;write_json(OUT/'comparison.json', report)
    for name, r in report['datasets'].items():
        print(name, {k: np.round(v, 4).tolist() for k, v in r.items() if k not in ('clhe_preset', 'ramen_setting', 'simple_method', 'simple_params')}, r['simple_method'])
    for family, records in families.items():
        for t in records:print(family, t['dataset'], t['metric'], f"delta {t['delta']:+.4f} ci [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] p_holm {t['p_holm']:.4f}")


if __name__ == '__main__':
    main()
