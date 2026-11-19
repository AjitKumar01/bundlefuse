"""Retrained ablations of TRIAD (../dials, package triad), one part removed at a time.

Each ablation retrains with the frozen configuration and seeds 0, 1, 2, so the only difference from the frozen runs
(../dials/runs) is the removed part:
  loo        training scores bundles without removing their own counts, degrees, frequencies and histories
  gate       all term weights fixed at 1
  content    no content term
  labels     no label term (every level), i.e. no same-label score delta_l
  graph      no graph term (co-occurrence and popularity prior)
untrained: the model at initialization with delta and rho set to zero, evaluated without training: 10x the dot
product of the first 64 feature coordinates plus log(1 + raw co-occurrence count), equal weights. Seed-independent.

Test: clean test queries, three-seed per-query means, frozen TRIAD minus ablation, NDCG@20 and Recall@20, paired
cluster bootstrap and sign flips, Holm over the tests of each ablation (2 metrics x the datasets run).

PYTHONPATH=.:../dials .venv/bin/python -m experiments.triad_ablations.run train --part gate --dataset food --seed 0
PYTHONPATH=.:../dials .venv/bin/python -m experiments.triad_ablations.run untrained --dataset food
PYTHONPATH=.:../dials .venv/bin/python -m experiments.triad_ablations.run test
"""
import triad  # pins BLAS threads before numpy loads
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from triad import data
from triad import train as T
from triad.evaluate import evaluate
from triad.model import TRIAD as DIALS
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison,holm_adjust

PARTS=['loo','gate','content','labels','graph']
OUT=Path('results/triad_ablations');CACHE=Path(DATA_ROOT)/'dials'/'cache';FROZEN=Path(DATA_ROOT)/'dials'/'runs'


class Zero(nn.Module):
    def forward(self,x):return torch.zeros(len(x),1)


class Ablated(DIALS):
    def __init__(self,*args,part,**kwargs):
        super().__init__(*args,**kwargs);self.part=part
        if part=='untrained':self.same_label=nn.ModuleList(Zero() for _ in range(self.levels));self.prior=Zero()

    def forward(self,seeds,bundles=None,cache=None):
        if cache is not None and bundles is not None:raise ValueError('The cache is for evaluation only')
        seeds=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s in seeds]
        if bundles is not None:bundles=[torch.as_tensor(sorted(set(b)),dtype=torch.long) for b in bundles]
        loo=None if self.part=='loo' else bundles
        parts={}
        if self.part!='content':parts[0]=self.content_scores(seeds,loo,cache)
        if self.part!='labels':
            for i in range(self.levels):parts[1+i]=self.label_scores(i,seeds)
        if self.part!='graph':parts[1+self.levels]=self.graph_scores(seeds,loo,cache)
        w=self.gate_weights(seeds,loo)
        if self.part in ('gate','untrained'):w=torch.ones_like(w)
        return sum(w[:,j,None]*v for j,v in parts.items())


def build(ds,cache,config):
    return Ablated(ds.features,ds.levels,ds.train,rank=config['rank'],hidden=config['hidden'],part=config['ablation'])


T.build=build  # triad.train.fit and triad.train.load construct models through this name


def train(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    ds=data.load(args.dataset,CACHE);run=OUT/args.part/args.dataset/f'seed{args.seed}'
    config=dict(T.CONFIG,ablation=args.part)
    try:
        T.fit(ds,args.seed,run,CACHE,config)
    except RuntimeError as error:
        # Without leave-one-out, a training target's own counts are in its score, the loss collapses and finally
        # overflows. Treat that as early stopping: mark the saved state as out of patience, let fit finish from the
        # best validation checkpoint (model.pt), and record where training broke down.
        if 'Nonfinite loss' not in str(error):raise
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        last['state']['stale']=config['patience'];torch.save(last,run/'last.pt')
        T.fit(ds,args.seed,run,CACHE,config)
        selected=json.loads((run/'selected.json').read_text())
        T.write_json(run/'selected.json',dict(selected,diverged_at_epoch=last['state']['epoch']+1))
    T.log(f'{args.part} {args.dataset} seed {args.seed} COMPLETE')


@torch.inference_mode()
def untrained(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True);torch.manual_seed(0)
    ds=data.load(args.dataset,CACHE);model=build(ds,CACHE,dict(T.CONFIG,ablation='untrained')).eval();run=OUT/'untrained'/args.dataset
    write_json(run/'test.json',evaluate(model,ds.test_pairs));val=evaluate(model,ds.val_pairs)
    write_json(run/'selected.json',dict(validation=dict(recall20=val['recall20'],ndcg20=val['ndcg20']),parameters=0))
    T.log(f'untrained {args.dataset} COMPLETE')


def test(_):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    report=dict(note=__doc__,parts={});groups=lambda pairs:[str(tuple(sorted(set(s)|set(g)))) for s,g in pairs]
    mean=lambda runs:np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in runs],axis=0)
    for part in PARTS+['untrained']:
        records=[];report['parts'][part]={}
        for name in data.DATASETS:
            runs=[OUT/part/name] if part=='untrained' else [OUT/part/name/f'seed{s}' for s in (0,1,2)]
            if not all((r/'selected.json').exists() for r in runs):continue
            ds=data.load(name,CACHE);keep=ds.clean_test;clusters=[groups(ds.test_pairs)[i] for i in keep]
            for run in runs:
                if not (run/'test.json').exists():write_json(run/'test.json',evaluate(T.load(ds,run,CACHE),ds.test_pairs))
            ablated=mean([json.loads((r/'test.json').read_text()) for r in runs])
            full=mean([json.loads((FROZEN/name/f'seed{s}'/'test.json').read_text()) for s in (0,1,2)])
            report['parts'][part][name]=dict(full=full[keep].mean(0).tolist(),ablated=ablated[keep].mean(0).tolist())
            for j,metric in ((1,'ndcg20'),(0,'recall20')):
                records.append(dict(paired_comparison(full[keep,j],ablated[keep,j],clusters,bootstrap=2000,permutations=10000,seed=20261109),
                                    part=part,dataset=name,metric=metric))
        if records:holm_adjust(records,len(records))  # 2 metrics x the datasets that were run
        report['parts'][part]['primary']=records
        for r in records:print(part,r['dataset'],r['metric'],round(r['delta'],4),[round(x,4) for x in r['ci95']],round(r['p_holm'],4),flush=True)
    write_json(OUT/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(required=True)
    t=sub.add_parser('train');t.set_defaults(run=train);t.add_argument('--part',required=True,choices=PARTS)
    t.add_argument('--dataset',required=True,choices=data.DATASETS);t.add_argument('--seed',type=int,required=True);t.add_argument('--threads',type=int,default=1)
    t=sub.add_parser('untrained');t.set_defaults(run=untrained);t.add_argument('--dataset',required=True,choices=data.DATASETS);t.add_argument('--threads',type=int,default=4)
    sub.add_parser('test').set_defaults(run=test)
    args=p.parse_args();args.run(args)
