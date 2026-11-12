#!/bin/zsh
# Prior with its own gate output plus kappa inside the graph log: POG and POG_DENSE, seeds 0-2, 3 jobs x 4 threads; then compare.
cd "$(dirname $0)/../../.."
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=4
for d in pog pog_dense; do for s in 0 1 2; do echo $d $s; done; done | xargs -P 3 -L 1 sh -c \
  'PYTHONPATH=.:../dials .venv/bin/python -u -m experiments.dials_v2.run train --variant prior_kappa --dataset $0 --seed $1 --threads 4 > results/dials_v2/prior_kappa/logs/$0_seed$1.log 2>&1'
PYTHONPATH=.:../dials .venv/bin/python -u -m experiments.dials_v2.run compare --variant prior_kappa --datasets pog pog_dense > results/dials_v2/prior_kappa/compare.log 2>&1
echo DONE >> results/dials_v2/prior_kappa/compare.log
