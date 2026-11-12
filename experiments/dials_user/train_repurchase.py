"""Train personalized DIALS + R on Dunnhumby (one seed) with the frozen training loop and configuration; score test once
with full training histories and once with histories restricted to baskets bought before the query basket.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_user.train_repurchase --seed 0
Runs go to results/dials_user/runs/dunnhumby_repurchase/seed<s>/.
"""
import argparse
import dials  # pins BLAS threads before numpy loads
import torch
from pathlib import Path
from dials import data as dials_data
from dials import train as dials_train
from experiments.dials_frozen.compare import CACHE
from experiments.dials_user.repurchase import DIALSRepurchase,History,evaluate

RUNS=Path('results/dials_user/runs/dunnhumby_repurchase')


def setup(ds):
    """History, and the build/evaluate the frozen loop calls, bound to it."""
    history=History(ds);val=[history.query(u) for u,_ in history.split.val]
    def build(ds,cache,config=dials_train.CONFIG):
        return DIALSRepurchase(ds.features,ds.levels,ds.train,dials_train.spectral(ds,cache,config),history,rank=config['rank'],hidden=config['hidden'])
    def evaluate_val(model,pairs):
        if pairs is not ds.val_pairs:raise RuntimeError('The training loop evaluates validation only')
        return evaluate(model,pairs,val)
    dials_train.build=build;dials_train.evaluate=evaluate_val  # fit() and load() resolve these names at call time
    return history


def main(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    ds=dials_data.load('dunnhumby',CACHE);history=setup(ds)
    run=RUNS/f'seed{args.seed}';dials_train.fit(ds,args.seed,run,CACHE);model=dials_train.load(ds,run,CACHE)
    full=[history.query(u) for u,_ in history.split.test];before=[history.before(u,d) for u,d in history.split.test]
    if not (run/'test.json').exists():dials_train.write_json(run/'test.json',evaluate(model,ds.test_pairs,full))
    if not (run/'test_before.json').exists():dials_train.write_json(run/'test_before.json',evaluate(model,ds.test_pairs,before))
    dials_train.log(f'dunnhumby +R seed {args.seed} COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--threads',type=int,default=1);main(p.parse_args())
