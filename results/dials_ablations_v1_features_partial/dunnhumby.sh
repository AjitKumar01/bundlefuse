#!/bin/zsh
# Dunnhumby ablations, three at a time, once the frozen dunnhumby runs exist (they are the reference).
cd "$(dirname $0)/../.."
until [ -f ../dials/runs/dunnhumby/seed0/selected.json -a -f ../dials/runs/dunnhumby/seed1/selected.json -a -f ../dials/runs/dunnhumby/seed2/selected.json ]; do sleep 120; done
for p in diagonal lowrank crossfit loo gate content labels graph; do for s in 0 1 2; do echo $p $s; done; done |
  xargs -P 3 -L 1 sh -c 'PYTHONPATH=.:../dials .venv/bin/python -u -m experiments.dials_ablations.run train --part $0 --dataset dunnhumby --seed $1 >> results/dials_ablations/logs/$0_dunnhumby_seed$1.log 2>&1'
PYTHONPATH=.:../dials .venv/bin/python -u -m experiments.dials_ablations.run untrained --dataset dunnhumby >> results/dials_ablations/logs/untrained_dunnhumby.log 2>&1
echo DONE
