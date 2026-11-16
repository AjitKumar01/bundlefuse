"""Fixed-count bundle-budget test: clean-test NDCG@20 and Recall@20 for DIALS and the best simple baseline (chosen on
validation per condition), plus the best label-free baseline, as training bundles go from all to 50.
Paired cluster bootstrap and sign flips (clusters: identical complete bundles); Holm over all tests of one family.
DIALS full: the frozen runs. Writes comparison.json (retail) or comparison_<family>.json.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget_retail.compare --family retail
"""
import argparse
import dials  # pins BLAS threads before numpy loads
import json
import numpy as np
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import RUNS,groups
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.bundle_budget.compare import LABEL
from .conditions import dataset,condition,conditions,FAMILIES,OUT
from .train import RUNS as B_RUNS


def dials_runs(name,cond):
    root=RUNS/name if cond=='full' else B_RUNS/name/cond
    files=[root/f'seed{s}'/'test.json' for s in (0,1,2)]
    if not all(f.exists() for f in files):return None
    return np.mean([[[q['recall20'],q['ndcg20']] for q in json.loads(f.read_text())['queries']] for f in files],0)


def main(family):
    report={};recs=[]
    for name in FAMILIES[family]:
        ds=dataset(name);keep=sorted(ds.clean_test);cl=[groups(ds.test_pairs)[i] for i in keep];report[name]={}
        for cond in conditions(name):
            path=OUT/'baselines'/name/f'{cond}.json'
            base=json.loads(path.read_text())['methods'] if path.exists() else None
            row=dict(bundles=len(condition(name,cond)));series={}
            d=dials_runs(name,cond)
            if d is not None:series['DIALS']=d[keep]
            if base:
                best=max(base,key=lambda k:base[k]['validation_ndcg20']);free=max((k for k in base if k not in LABEL),key=lambda k:base[k]['validation_ndcg20'])
                series[f'best simple ({best})']=np.asarray(base[best]['queries']);row['best_simple']=best
                if free!=best:series[f'best label-free ({free})']=np.asarray(base[free]['queries'])
                row['all_simple']={k:v['test'] for k,v in base.items()}
            row['means']={k:v.mean(0).tolist() for k,v in series.items()}
            if 'DIALS' in series:
                for k,v in series.items():
                    if not k.startswith('best simple'):continue
                    for j,mt in enumerate(('recall20','ndcg20')):
                        recs.append(dict(dataset=name,condition=cond,comparison='DIALS vs '+k,metric=mt,**paired_comparison(series['DIALS'][:,j],v[:,j],cl,seed=20261109)))
            report[name][cond]=row
            print(f'{name:10s} {cond:5s} '+'  '.join(f'{k} N {v[1]:.4f} R {v[0]:.4f}' for k,v in row['means'].items()),flush=True)
    holm_adjust(recs)
    for r in recs:print(f"{r['dataset']:10s} {r['condition']:5s} {r['comparison']:40s} {r['metric']:8s} {r['delta']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p_holm {r['p_holm']:.4f}")
    write_json(OUT/('comparison.json' if family=='retail' else f'comparison_{family}.json'),dict(note=__doc__,family=family,datasets=report,tests=recs))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',required=True,choices=FAMILIES)
    main(p.parse_args().family)
