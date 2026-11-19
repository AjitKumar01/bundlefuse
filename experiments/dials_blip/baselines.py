"""The simple and label-aware baselines on Electronic, Food and Clothing with BLIP text features.

Runs experiments.bundle_budget_retail.baselines on the full training set (condition 'full') with the dataset loaded
through experiments.dials_blip.run.load, so content enters as the same 128-d BLIP projection DIALS uses. All of its
methods are rerun (popularity, co-occurrence, ItemKNN, RP3beta, EASE, content + ItemKNN blend, same label, transition,
label + ItemKNN, label + ItemKNN + content); only the two content methods can change. Writes
results/dials_blip/baselines/<dataset>/full.json.

PYTHONPATH=.:../dials nice -n 10 .venv/bin/python -m experiments.dials_blip.baselines
"""
import triad  # pins BLAS threads before numpy loads
import argparse
from pathlib import Path
from experiments.bundle_budget_retail import baselines, conditions
from experiments.dials_blip import run as blip


def main():
    loaded = {}
    def dataset(name):
        if name not in loaded:loaded[name] = blip.load(name)
        return loaded[name]
    conditions.dataset = baselines.dataset = dataset  # condition('full') reads conditions.dataset
    baselines.OUT = Path('results/dials_blip')
    for name in ('electronic', 'food', 'clothing'):
        conditions.condition.cache_clear();baselines.main(argparse.Namespace(dataset=name, condition='full'))


if __name__ == '__main__':
    main()
