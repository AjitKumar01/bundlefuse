"""DIALS (v3.1, frozen, three seeds) against RaMen trained on the same Electronic, Food and Clothing split.

RaMen's setting is chosen per dataset and variant on validation NDCG@20 (seed 2306), then run with seeds 0 and 1.
Primary: DIALS vs RaMen-clean (item graph from training bundles only, the information DIALS has), NDCG@20 and
Recall@20, Holm over the six tests. RaMen-published (item graph from user_item.csv, which holds every test bundle)
is reported to show how much of RaMen's published e-commerce accuracy that graph accounts for; it is not a fair
comparison and is not tested.
"""
import glob
import json
from pathlib import Path
import numpy as np
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.neural_lowrank import run as base
from experiments.neural_retail.run import DATASETS,data

RESULTS=Path('results/ramen_retail')


def ramen(name,variant):
    runs=[json.loads(Path(f).read_text()) for f in glob.glob(str(RESULTS/f'{name}_{variant}'/'*'/'result.json'))]
    grid=[r for r in runs if r['seed']==2306];best=max(grid,key=lambda r:r['validation'][1])
    chosen=[r for r in runs if (r['lr'],r['n_layer_gat'],r['gamma'])==(best['lr'],best['n_layer_gat'],best['gamma'])]
    if sorted(r['seed'] for r in chosen)!=[0,1,2306]:raise RuntimeError(f'{name} {variant}: missing seeds')
    return dict(lr=best['lr'],n_layer_gat=best['n_layer_gat'],gamma=best['gamma'],epochs=[r['epoch'] for r in chosen],
                seeds=[r['test'] for r in chosen],queries=np.mean([r['queries'] for r in chosen],axis=0))


def main():
    report={};records=[]
    for name in DATASETS:
        ds,_,_,_,_=data(name);ids=base.groups(ds.test_pairs)
        runs=[json.loads((Path('results/neural_retail')/name/f'v31_loo_seed{s}'/'test.json').read_text()) for s in (0,1,2)]
        dials=np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in runs],axis=0)
        variants={v:ramen(name,v) for v in ('published','clean')}
        report[name]=dict(dials=dials.mean(0).tolist(),**{f'ramen_{v}':dict({k:x for k,x in r.items() if k!='queries'},test=r['queries'].mean(0).tolist())
                                                          for v,r in variants.items()})
        for j,metric in ((1,'ndcg20'),(0,'recall20')):
            records.append(dict(paired_comparison(dials[:,j],variants['clean']['queries'][:,j],ids,bootstrap=2000,permutations=10000,seed=20261109),
                                dataset=name,metric=metric,first='dials',second='ramen_clean'))
    holm_adjust(records,len(records))
    write_json(RESULTS/'summary.json',dict(note=__doc__,datasets=report,primary=records))
    for name,r in report.items():
        print(name,'DIALS',np.round(r['dials'],4),'RaMen clean',np.round(r['ramen_clean']['test'],4),'RaMen published-graph',np.round(r['ramen_published']['test'],4))
    for r in records:print(r['dataset'],r['metric'],round(r['delta'],4),[round(x,4) for x in r['ci95']],r['p_holm'])


if __name__=='__main__':
    main()
