"""Train DIALS + U on POG (one seed) with the frozen DIALS training loop and configuration, then score test once.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_user.train --seed 0
Runs go to results/dials_user/runs/<dataset>/seed<s>/ (model.pt, selected.json, test.json).
"""
import argparse
import dials  # pins BLAS threads before numpy loads
import torch
from pathlib import Path
from dials import data as dials_data
from dials import train as dials_train
from dials.evaluate import evaluate
from experiments.dials_frozen.compare import CACHE
from experiments.dials_user.model import DIALSUser,user_households,history_matrix

RUNS=Path('results/dials_user/runs')


def main(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    ds=dials_data.load(args.dataset,CACHE);H=user_households(args.dataset,ds);M=history_matrix([set(c) for c in H[0]],ds.n_items);H=H if H[1] is not None else None
    def build(ds,cache,config=dials_train.CONFIG):
        return DIALSUser(ds.features,ds.levels,ds.train,dials_train.spectral(ds,cache,config),M,rank=config['rank'],hidden=config['hidden'],households=H)
    dials_train.build=build  # fit() and load() construct the model through this name
    run=RUNS/args.dataset/f'seed{args.seed}';dials_train.fit(ds,args.seed,run,CACHE)
    if not (run/'test.json').exists():dials_train.write_json(run/'test.json',evaluate(dials_train.load(ds,run,CACHE),ds.test_pairs))
    dials_train.log(f'{args.dataset} +U seed {args.seed} COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',default='pog');p.add_argument('--seed',type=int,required=True)
    p.add_argument('--threads',type=int,default=1);main(p.parse_args())
