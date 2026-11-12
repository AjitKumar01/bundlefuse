#!/bin/zsh
# Candidate-dependent content weight: POG and POG_DENSE, seeds 0-2, three jobs at a time, 4 threads each; then compare.
cd "$(dirname $0)/../../.."
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=4
for d in pog pog_dense; do for s in 0 1 2; do echo $d $s; done; done | xargs -P 3 -L 1 sh -c \
  'PYTHONPATH=.:../dials .venv/bin/python -u -m experiments.dials_v2.run train --dataset $0 --seed $1 --threads 4 > results/dials_v2/candidate_weight/logs/$0_seed$1.log 2>&1'
PYTHONPATH=.:../dials .venv/bin/python -u -m experiments.dials_v2.run compare --datasets pog pog_dense > results/dials_v2/candidate_weight/compare.log 2>&1
