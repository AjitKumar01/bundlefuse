#!/bin/zsh
# POG timing, one method at a time, 4 threads each.
cd "$(dirname $0)/../.."
THREADS=4
export VECLIB_MAXIMUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
PYTHONPATH=.:../dials .venv/bin/python -u -m experiments.timing.run --threads $THREADS > results/timing/dials_simple.log 2>&1
(cd ../baselines/CLHE && ../RaMen_repo/.venv/bin/python -u ../../bundlefuse/experiments/timing/neural.py --model clhe --threads $THREADS) > results/timing/clhe.log 2>&1
(cd ../baselines/RaMen_repo && .venv/bin/python -u ../../bundlefuse/experiments/timing/neural.py --model ramen --threads $THREADS) > results/timing/ramen.log 2>&1
echo DONE >> results/timing/dials_simple.log
