"""Development runs of the group-match model. Validation only; test needs a separate fixed plan.

PYTHONPATH=. .venv/bin/python -m experiments.neural_lowrank2.run train --dataset spotify_sparse --tag v2
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.neural_lowrank import run as base
from experiments.neural_revision.run import log
from .model import GroupMatchRelation

RESULTS=Path('results/neural_lowrank2');CHECKPOINTS=Path('checkpoints/neural_lowrank2')
CONFIG=dict(base.CONFIG,group_lr=.01,group_match=True,popularity=True,use_album=True)


def levels(ds,cats,config):
    out=[cats]
    if config.get('use_album',True) and getattr(ds,'alb_labels',None) is not None:out.append(np.asarray(ds.alb_labels))
    return out


def construct(ds,features,cats,config=CONFIG):
    model=GroupMatchRelation(features,levels(ds,cats,config),ds.train,group_match=config['group_match'],popularity=config['popularity'],
                              rank=config['rank'],group_rank=config['group_rank'],hidden=config['hidden'])
    model.disabled=set(config.get('disabled',()))
    return model


base.construct=construct  # base.fit and base.load build models through this name


def train(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    ds,x,c,_,_=base.data(args.dataset)
    config=dict(CONFIG,**{k:v for k,v in (('group_match',not args.no_match),('popularity',not args.no_popularity)) })
    if args.disable:config['disabled']=sorted(args.disable)
    if args.epochs:config['epochs']=args.epochs
    key=f'{args.tag}_{args.arm}_seed{args.seed}'
    base.fit(ds,x,c,args.arm,args.seed,config,RESULTS/args.dataset/key,CHECKPOINTS/args.dataset/key)
    log(f'{args.dataset} {key} COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(required=True)
    t=sub.add_parser('train');t.set_defaults(run=train)
    t.add_argument('--dataset',required=True,choices=base.POG+base.SPOTIFY);t.add_argument('--arm',default='loo_mask',choices=list(base.ARMS))
    t.add_argument('--tag',required=True);t.add_argument('--seed',type=int,default=0);t.add_argument('--threads',type=int,default=1);t.add_argument('--epochs',type=int,default=0)
    t.add_argument('--no-match',action='store_true');t.add_argument('--no-popularity',action='store_true');t.add_argument('--disable',nargs='*',default=[],choices=['content','history','group','pair'])
    args=p.parse_args();args.run(args)
