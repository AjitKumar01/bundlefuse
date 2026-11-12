"""DIALS against both trained neural baselines on Electronic, Food and Clothing: RaMen and CLHE, clean variants.

Clean means every graph and collaborative feature comes from training bundles only, the information DIALS has.
RaMen: setting chosen on validation (results/ramen_retail). CLHE: one of its four shipped presets chosen on
validation NDCG@20 (seed 2023), then seeds 0 and 1 (results/clhe_retail). Primary: NDCG@20 and Recall@20, Holm over
all twelve tests. Published-graph variants (user_item.csv, which holds every test bundle) are reported, not tested.
"""
import glob
import json
from pathlib import Path
import numpy as np
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.neural_lowrank import run as base
from experiments.neural_retail.run import DATASETS,data
from .compare import ramen


def clhe(name,variant):
    runs=[json.loads(Path(f).read_text()) for f in glob.glob(f'results/clhe_retail/{name}_{variant}/*/result.json')]
    best=max([r for r in runs if r['seed']==2023],key=lambda r:r['validation'][1])
    chosen=[r for r in runs if r['preset']==best['preset']]
    if sorted(r['seed'] for r in chosen)!=[0,1,2023]:raise RuntimeError(f'{name} {variant}: missing seeds')
    return dict(preset=best['preset'],epochs=[r['epoch'] for r in chosen],seeds=[r['test'] for r in chosen],parameters=best['parameters'],
                queries=np.mean([r['queries'] for r in chosen],axis=0))


def main():
    report={};records=[]
    for name in DATASETS:
        ds,_,_,_,_=data(name);ids=base.groups(ds.test_pairs)
        runs=[json.loads((Path('results/neural_retail')/name/f'v31_loo_seed{s}'/'test.json').read_text()) for s in (0,1,2)]
        dials=np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in runs],axis=0)
        others={f'{m}_{v}':f(name,v) for m,f in (('ramen',ramen),('clhe',clhe)) for v in ('clean','published')}
        report[name]=dict(dials=dials.mean(0).tolist(),**{k:dict({a:b for a,b in r.items() if a!='queries'},test=r['queries'].mean(0).tolist()) for k,r in others.items()})
        for other in ('ramen_clean','clhe_clean'):
            for j,metric in ((1,'ndcg20'),(0,'recall20')):
                records.append(dict(paired_comparison(dials[:,j],others[other]['queries'][:,j],ids,bootstrap=2000,permutations=10000,seed=20261109),
                                    dataset=name,metric=metric,first='dials',second=other))
    holm_adjust(records,len(records))
    write_json(Path('results/clhe_retail/comparison.json'),dict(note=__doc__,datasets=report,primary=records))
    for name,r in report.items():
        print(name,' '.join(f"{k} {np.round(r[k] if k=='dials' else r[k]['test'],4)}" for k in ('dials','ramen_clean','clhe_clean','ramen_published','clhe_published')))
    for r in records:print(r['dataset'],r['metric'],r['second'],round(r['delta'],4),[round(x,4) for x in r['ci95']],r['p_holm'])


if __name__=='__main__':
    main()
