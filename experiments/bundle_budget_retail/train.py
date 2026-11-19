"""Train DIALS (frozen configuration and loop) on one retail bundle-budget condition and score test once.

The full condition reuses the frozen runs. Runs go to results/bundle_budget_retail/runs/<dataset>/<condition>/seed<s>/.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget_retail.train --dataset electronic --condition n100 --seed 0
"""
import argparse
import triad  # pins BLAS threads before numpy loads
import torch
from triad import train as dials_train
from triad.evaluate import evaluate
from experiments.dials_frozen.compare import CACHE
from .conditions import dataset,condition,with_train,CONDITIONS,DATASETS,OUT

RUNS=OUT/'runs'


def main(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    if args.condition=='full':raise SystemExit('full reuses the frozen runs')
    ds=with_train(dataset(args.dataset),condition(args.dataset,args.condition));run=RUNS/args.dataset/args.condition/f'seed{args.seed}'
    dials_train.fit(ds,args.seed,run,CACHE)
    if not (run/'test.json').exists():
        dials_train.write_json(run/'test.json',evaluate(dials_train.load(ds,run,CACHE),ds.test_pairs))
    dials_train.log(f'{args.dataset} {args.condition} seed {args.seed} COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=DATASETS);p.add_argument('--condition',required=True,choices=CONDITIONS[1:])
    p.add_argument('--seed',type=int,required=True);p.add_argument('--threads',type=int,default=1)
    main(p.parse_args())
