#!/bin/zsh
# Seed 0 selection runs, one thread each, in parallel.
cd "$(dirname $0)/../.."
for d in pog_dense pog; do for a in loo loo_mask; do
  PYTHONPATH=. .venv/bin/python -u -m experiments.neural_lowrank.run train --dataset $d --arm $a --seed 0 > results/neural_lowrank/${d}_${a}_seed0.log 2>&1 &
done; done
wait
