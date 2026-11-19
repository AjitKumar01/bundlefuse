"""Two diagnostics for the retail bundle-budget result (DIALS behind the simple baselines at 300 bundles and below).

steps: the frozen loop counts patience and the epoch limit in epochs, so 50 bundles give 2 gradient steps per epoch
       against about 38 at full size. Here each condition's bundles are repeated (and shuffled, each copy with its
       own random seed/target split) until an epoch has as many training rows as the full split, so steps per
       epoch, patience and the epoch limit match the full-size runs. Statistics (counts, label factors, partner
       histories) still come from the condition's bundles, each counted once, and leave-one-out removes each row's
       bundle once, exactly as before. Runs go to results/bundle_budget_retail/runs_steps/<dataset>/<condition>/seed<s>/.
init:  DIALS with its parameters at initialization, no training step: Z = X, M = 0, every gate weight 1, alpha = beta
       = 0, theta = 1, content = 10 x the dot product of the first 64 feature coordinates. delta and rho are small
       MLPs with random initial weights, so three seeds are scored. Writes results/bundle_budget_retail/init/<dataset>/<condition>.json.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget_retail.diagnostics steps --dataset food --condition n50 --seed 0
PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget_retail.diagnostics init --dataset food --condition n50
"""
import argparse
import triad  # pins BLAS threads before numpy loads
import math
import torch
from triad import train as dials_train
from triad.evaluate import evaluate
from triad.model import TRIAD as DIALS
from experiments.dials_frozen.compare import CACHE
from .conditions import dataset,condition,with_train,CONDITIONS,DATASETS,OUT


def bind(stats):
    """Make dials.train.build use the condition's bundles for every statistic, whatever the training rows are."""
    def build(_ds,cache,config=dials_train.CONFIG):
        return DIALS(stats.features,stats.levels,stats.train,rank=config["rank"],hidden=config["hidden"])
    dials_train.build=build


def steps(args):
    ds=dataset(args.dataset);bundles=condition(args.dataset,args.condition);stats=with_train(ds,bundles)
    copies=math.ceil(len(ds.train)/len(bundles));rows=with_train(ds,(bundles*copies)[:max(len(ds.train),len(bundles))])
    bind(stats);run=OUT/'runs_steps'/args.dataset/args.condition/f'seed{args.seed}'
    dials_train.log(f'{args.dataset} {args.condition}: {len(bundles)} bundles repeated to {len(rows.train)} rows per epoch')
    dials_train.fit(rows,args.seed,run,CACHE)
    if not (run/'test.json').exists():
        dials_train.write_json(run/'test.json',evaluate(dials_train.load(rows,run,CACHE),ds.test_pairs))
    dials_train.log(f'{args.dataset} {args.condition} seed {args.seed} steps COMPLETE')


def init(args):
    ds=dataset(args.dataset);stats=with_train(ds,condition(args.dataset,args.condition));out=dict(dataset=args.dataset,condition=args.condition,seeds={})
    for seed in (0,1,2):
        torch.manual_seed(seed);model=dials_train.build(stats,CACHE)
        val=evaluate(model,ds.val_pairs);test=evaluate(model,ds.test_pairs)
        out['seeds'][seed]=dict(validation={k:val[k] for k in ('recall20','ndcg20')},test=test)
        dials_train.log(f'{args.dataset} {args.condition} init seed {seed} VAL N20={val["ndcg20"]:.4f} TEST N20={test["ndcg20"]:.4f}')
    dials_train.write_json(OUT/'init'/args.dataset/f'{args.condition}.json',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('steps','init'))
    p.add_argument('--dataset',required=True,choices=DATASETS);p.add_argument('--condition',required=True,choices=CONDITIONS)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True)
    if a.mode=='steps' and a.condition=='full':raise SystemExit('full already has full-size epochs')
    {'steps':steps,'init':init}[a.mode](a)
