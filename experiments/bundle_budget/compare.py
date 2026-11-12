"""Bundle-budget test: clean-test NDCG@20 and Recall@20 for DIALS, the best simple baseline (chosen on validation per
condition), the best label-free baseline, and CLHE, as training bundles go from all to 10%, 1% and none (user-history
pseudo-bundles). Paired cluster bootstrap and sign flips (clusters: identical complete bundles); Holm over all tests.
DIALS full: the frozen runs; CLHE full: the three-seed Kaggle reruns; CLHE other conditions: kaggle_bundle_budget output.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget.compare
"""
import dials  # pins BLAS threads before numpy loads
import hashlib
import json
import numpy as np
from pathlib import Path
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import RUNS,groups
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from .conditions import dataset,DATASETS,CONDITIONS,OUT
from .train import RUNS as B_RUNS

CLHE_FULL=Path(DATA_ROOT)/'clhe_seeds_out';CLHE_BUDGET=Path(DATA_ROOT)/'kaggle_bundle_budget'/'out'
LABEL=('samelabel','transition','label_knn','label_knn_content')


def dials_runs(name,cond):
    root=RUNS/name if cond=='full' else B_RUNS/name/cond
    files=[root/f'seed{s}'/'test.json' for s in (0,1,2)]
    if not all(f.exists() for f in files):return None
    return np.mean([[[q['recall20'],q['ndcg20']] for q in json.loads(f.read_text())['queries']] for f in files],0)


def clhe_runs(name,cond,ds):
    folder=CLHE_FULL/name.replace('_','-')/'clhe_out'/name if cond=='full' else CLHE_BUDGET/name/cond
    files=sorted(folder.glob('seed*.json'))
    if len(files)!=3:return None
    for f,h in json.loads((folder/'split_sha256.json').read_text()).items():
        local=Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/f
        if ('valid' in f or 'test' in f) and local.exists() and hashlib.sha256((Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/f).read_bytes()).hexdigest()!=h:
            raise RuntimeError(f'{name}: {f} differs from our split')
    lines=[l for l in open(Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/'bi_test_input.txt') if l.strip()]
    order=[int(l.split(',')[0]) for l in lines]
    if any(sorted(set(map(int,l.replace(' ','').strip().split(',')[1:])))!=sorted(set(s)) for l,(s,_) in zip(lines,ds.test_pairs)):
        raise RuntimeError(f'{name}: test order differs from the file')
    per={b:i for i,b in enumerate(order)};out=np.zeros((len(order),2))
    for f in files:
        m=np.full((len(order),2),np.nan)
        for b,rec,nd in json.loads(f.read_text())['queries']:m[per[b]]=(rec,nd)
        if np.isnan(m).any():raise RuntimeError(f'{name} {cond}: CLHE misses queries')
        out+=m/3
    return out


def main():
    report={};recs=[]
    for name in DATASETS:
        ds=dataset(name);keep=sorted(ds.clean_test);cl=[groups(ds.test_pairs)[i] for i in keep];report[name]={}
        for cond in CONDITIONS:
            path=OUT/'baselines'/name/f'{cond}.json'
            base=json.loads(path.read_text())['methods'] if path.exists() else None
            row={};series={}
            d=dials_runs(name,cond)
            if d is not None:series['DIALS']=d[keep]
            c=clhe_runs(name,cond,ds)
            if c is not None:series['CLHE']=c[keep]
            if base:
                best=max(base,key=lambda k:base[k]['validation_ndcg20']);free=max((k for k in base if k not in LABEL),key=lambda k:base[k]['validation_ndcg20'])
                series[f'best simple ({best})']=np.asarray(base[best]['queries']);row['best_simple']=best
                if free!=best:series[f'best label-free ({free})']=np.asarray(base[free]['queries'])
                row['all_simple']={k:v['test'] for k,v in base.items()}
            row['means']={k:v.mean(0).tolist() for k,v in series.items()}
            if 'DIALS' in series:
                for k,v in series.items():
                    if k=='DIALS' or not k.startswith(('best simple','CLHE')):continue
                    for j,mt in enumerate(('recall20','ndcg20')):
                        recs.append(dict(dataset=name,condition=cond,comparison='DIALS vs '+k,metric=mt,**paired_comparison(series['DIALS'][:,j],v[:,j],cl,seed=20261109)))
            report[name][cond]=row
            print(f'{name:10s} {cond:13s} '+'  '.join(f'{k} N {v[1]:.4f} R {v[0]:.4f}' for k,v in row['means'].items()),flush=True)
    holm_adjust(recs)
    for r in recs:print(f"{r['dataset']:10s} {r['condition']:13s} {r['comparison']:45s} {r['metric']:8s} {r['delta']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p_holm {r['p_holm']:.4f}")
    write_json(OUT/'comparison.json',dict(note=__doc__,datasets=report,tests=recs))


if __name__=='__main__':main()
