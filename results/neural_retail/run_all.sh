#!/bin/zsh
# v3.1 on Electronic, Food, Clothing (3 seeds each, one thread) in parallel with the simple baselines on all seven datasets.
cd "$(dirname $0)/../.."
for d in electronic food clothing; do for s in 0 1 2; do
  PYTHONPATH=. .venv/bin/python -u -m experiments.neural_retail.run train --dataset $d --seed $s > results/neural_retail/${d}_seed$s.log 2>&1 & sleep 5
done; done
mkdir -p results/simple_baselines
{ for d in electronic food clothing pog_dense pog spotify_sparse spotify; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 PYTHONPATH=. .venv/bin/python -u -m experiments.simple_baselines.run --dataset $d > results/simple_baselines/$d.log 2>&1
done } &
wait
