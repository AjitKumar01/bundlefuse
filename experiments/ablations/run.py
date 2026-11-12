"""Retrained ablations of DIALS (frozen v3.1): one component switched off, everything else as frozen, retrained.

The frozen configuration (results/neural_relations/plan.json) with config['disabled']=[part]; same seeds (0, 1, 2),
same training, epoch chosen on validation NDCG@20. Unlike the inspect ablations, the rest of the model can adapt.
Parts: hop2 (second graph hop, the only user of the graph SVD), diagonal, lowrank, gate, prior, content, label0, graph.

Test: three-seed per-query means, full DIALS minus the ablation, NDCG@20 and Recall@20, filtered queries on
POG/Spotify (as in the frozen test), all queries on Electronic/Food/Clothing. Paired cluster bootstrap and sign
flips, Holm over the datasets and metrics of that part.

PYTHONPATH=. .venv/bin/python -m experiments.ablations.run train --part hop2 --dataset pog_dense --seed 0
PYTHONPATH=. .venv/bin/python -m experiments.ablations.run test --part hop2
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from bundlefuse.provenance import write_json
from experiments.neural_lowrank import run as base
from experiments.neural_relations import run as r3  # sets base.construct to the v3.1 model
from experiments.neural_retail import run as retail
from experiments.neural_revision.run import log

RETAIL=retail.DATASETS;OTHERS=r3.DATASETS;DATASETS=RETAIL+OTHERS
PARTS=['hop2','diagonal','lowrank','gate','prior','content','label0','graph']
RESULTS=Path('results/ablations');CHECKPOINTS=Path('checkpoints/ablations')


def data(name):
    return retail.data(name) if name in RETAIL else base.data(name)


def frozen_dir(name):
    return (retail.RESULTS if name in RETAIL else r3.RESULTS)/name


def train(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    ds,x,c,_,_=data(args.dataset);key=f'no_{args.part}_seed{args.seed}'
    base.fit(ds,x,c,'loo',args.seed,dict(r3.FROZEN,disabled=[args.part]),RESULTS/args.dataset/key,CHECKPOINTS/args.dataset/key)
    log(f'{args.dataset} {key} COMPLETE')


def test(args):
    from experiments.mechanisms.statistics import paired_comparison,holm_adjust
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    path=RESULTS/f'{args.part}.json';report=dict(note=__doc__,part=args.part,datasets={});records=[]
    for name in DATASETS:
        keys=[f'no_{args.part}_seed{s}' for s in (0,1,2)]
        if not all((RESULTS/name/k/'selected.json').exists() for k in keys):log(f'{name}: not all seeds trained, skipped');continue
        ds,x,c,_,audit=data(name);keep=audit['clean_test_indices'];rows=[];learned=[]
        for k in keys:
            choice=json.loads((RESULTS/name/k/'selected.json').read_text())
            if choice['config']!=dict(r3.FROZEN,disabled=[args.part]):raise RuntimeError(f'{name} {k}: configuration differs')
            out=RESULTS/name/k/'test.json'
            if out.exists():result=json.loads(out.read_text())
            else:
                model=base.load(ds,x,c,choice);result=base.evaluate(model,ds.test_pairs)
                result['hops']=model.hops.tolist();result['exponents']=model.exponents.tolist();write_json(out,result)
            rows.append(result);learned.append(dict(best_epoch=choice['best_epoch'],validation=choice['validation'],hops=result['hops'],exponents=result['exponents']))
        mean=lambda runs:np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in runs],axis=0)
        ablated=mean(rows);full=mean([json.loads((frozen_dir(name)/f'v31_loo_seed{s}'/'test.json').read_text()) for s in (0,1,2)])
        ids=base.groups(ds.test_pairs);clean=[ids[i] for i in keep]
        report['datasets'][name]=dict(full=full[keep].mean(0).tolist(),ablated=ablated[keep].mean(0).tolist(),seeds=learned,queries=len(keep))
        for j,metric in ((1,'ndcg20'),(0,'recall20')):
            records.append(dict(paired_comparison(full[keep,j],ablated[keep,j],clean,bootstrap=2000,permutations=10000,seed=20261109),
                                dataset=name,metric=metric,first='dials',second=f'no_{args.part}'))
        log(f'{name} full {np.round(full[keep].mean(0),4)} no_{args.part} {np.round(ablated[keep].mean(0),4)}')
    holm_adjust(records,len(records));report['primary']=records;write_json(path,report)
    for r in records:print(r['dataset'],r['metric'],round(r['delta'],4),[round(v,4) for v in r['ci95']],r['p_holm'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(required=True)
    t=sub.add_parser('train');t.set_defaults(run=train);t.add_argument('--part',required=True,choices=PARTS)
    t.add_argument('--dataset',required=True,choices=DATASETS);t.add_argument('--seed',type=int,required=True);t.add_argument('--threads',type=int,default=1)
    t=sub.add_parser('test');t.set_defaults(run=test);t.add_argument('--part',required=True,choices=PARTS)
    args=p.parse_args();args.run(args)
