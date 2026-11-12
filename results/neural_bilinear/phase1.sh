#!/bin/zsh
cd /Users/ajit/Projects/Career_change/projects/bundle-dpo/bundlefuse
export PYTHONPATH=.
for d in pog_dense pog; do
  mkdir -p results/neural_bilinear/$d
  .venv/bin/python -u -m experiments.neural_revision.pog --dataset $d >> results/neural_pog/$d/run.log 2>&1 &
  for a in bilinear category both; do
    .venv/bin/python -u -m experiments.neural_bilinear.run train --dataset $d --arm $a --seed 0 > results/neural_bilinear/$d/${a}_seed0.log 2>&1 &
  done
done
wait
grep -L COMPLETE results/neural_bilinear/*/*_seed0.log
tail -qn1 results/neural_bilinear/*/*_seed0.log | cut -c1-90
grep -h COMPLETE results/neural_pog/*/run.log | cut -c1-70
