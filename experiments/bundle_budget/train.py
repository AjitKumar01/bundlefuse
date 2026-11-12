"""Train DIALS (frozen configuration and loop) on one bundle-budget condition and score clean test once.

Statistics (counts, label factors, partner histories) come from the condition's bundles; the training rows are the
condition's rows (see conditions.py). The full condition reuses the frozen runs. Runs go to
results/bundle_budget/runs/<dataset>/<condition>/seed<s>/.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget.train --dataset pog --condition frac10 --seed 0
"""
import argparse
import dials  # pins BLAS threads before numpy loads
import torch
from pathlib import Path
from dials import train as dials_train
from dials.evaluate import evaluate
from dials.model import DIALS
from experiments.dials_frozen.compare import CACHE
from .conditions import dataset,condition,with_train,CONDITIONS,DATASETS,OUT

RUNS=OUT/'runs'


def setup(name,cond):
    """The condition's statistics dataset and training-rows dataset, with build() bound to the statistics."""
    ds=dataset(name);stats,rows=condition(name,cond);full=with_train(ds,stats)
    def build(_ds,cache,config=dials_train.CONFIG):
        return DIALS(full.features,full.levels,full.train,dials_train.spectral(full,cache,config),rank=config['rank'],hidden=config['hidden'])
    dials_train.build=build  # fit() and load() resolve build at call time
    return full,with_train(ds,rows)


def main(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    if args.condition=='full':raise SystemExit('full reuses the frozen runs')
    full,rows=setup(args.dataset,args.condition);run=RUNS/args.dataset/args.condition/f'seed{args.seed}'
    dials_train.fit(rows,args.seed,run,CACHE)
    if not (run/'test.json').exists():
        model=dials_train.load(rows,run,CACHE);dials_train.write_json(run/'test.json',evaluate(model,rows.test_pairs))
    dials_train.log(f'{args.dataset} {args.condition} seed {args.seed} COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=DATASETS);p.add_argument('--condition',required=True,choices=CONDITIONS[1:])
    p.add_argument('--seed',type=int,required=True);p.add_argument('--threads',type=int,default=1)
    main(p.parse_args())
