"""Training-set size on Electronic, Food and Clothing: TRIAD trained on nested random subsets of 50, 100, 300 and 600
training bundles (the conditions of experiments.bundle_budget_retail: one fixed permutation per dataset, seed
20261113), and on all of them, seeds 0-2. Validation and test queries never change. The features are the BLIP title
features of the dials data loader for both models.

  --model final     the final TRIAD: run with PYTHONPATH=.:../dials (the dials main code; triad.train.config_for, so
                    both learning rates halved on these datasets; epoch 0 can be selected)
  --model original  the original TRIAD: run with PYTHONPATH=.:../dials_original (the archived code, its CONFIG)

Writes results/small_data/budget_rerun/<model>/<dataset>_<condition>/seed<s>/{model.pt, selected.json, history.json,
test.json}; the full condition of the final model is ../dials/runs (already trained).

cd ../bundlefuse && PYTHONPATH=.:../dials .venv/bin/python -m experiments.small_data.budget_rerun --model final
"""
import triad  # pins BLAS threads before numpy loads
import argparse, dataclasses
from pathlib import Path
import numpy as np, torch
from triad import data
from triad import train as T
from triad.evaluate import evaluate

DIALS = Path(__file__).resolve().parents[3]/'dials';CACHE = DIALS/'cache';OUT = Path('results/small_data/budget_rerun')
NAMES = ('electronic', 'food', 'clothing');SIZES = (50, 100, 300, 600);SEED = 20261113


def subset(ds, k):
    """The first k bundles of one fixed permutation of the training bundles, in their original order."""
    order = np.random.default_rng(SEED).permutation(len(ds.train))
    return dataclasses.replace(ds, train=[ds.train[i] for i in sorted(order[:k])])


def main():
    p = argparse.ArgumentParser();p.add_argument('--model', choices=('final', 'original'), required=True);a = p.parse_args()
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    for name in NAMES:
        full = data.load(name, CACHE)
        config = T.config_for(name) if a.model == 'final' else dict(T.CONFIG)
        conds = [(f'n{k}', subset(full, k)) for k in SIZES] + ([('full', full)] if a.model == 'original' else [])
        for cond, ds in conds:
            for seed in (0, 1, 2):
                run = OUT/a.model/f'{name}_{cond}'/f'seed{seed}'
                T.fit(ds, seed, run, CACHE, config=config)
                if not (run/'test.json').exists():T.write_json(run/'test.json', evaluate(T.load(ds, run, CACHE), ds.test_pairs))
            print(a.model, name, cond, 'test N20 %.4f' % np.mean([T.json.loads((OUT/a.model/f'{name}_{cond}'/f'seed{s}'/'test.json').read_text())['ndcg20'] for s in (0, 1, 2)]),
                  'epochs', [T.json.loads((OUT/a.model/f'{name}_{cond}'/f'seed{s}'/'selected.json').read_text())['best_epoch'] for s in (0, 1, 2)], flush=True)


if __name__ == '__main__':
    main()
