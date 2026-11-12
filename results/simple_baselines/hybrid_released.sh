#!/bin/zsh
# Content plus ItemKNN blend rerun on the released features only (2026-11-11); the other methods use no features.
cd "$(dirname $0)/../.."
export VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
for d in pog_dense pog spotify_sparse spotify; do
  PYTHONPATH=.:../dials .venv/bin/python -u -m experiments.simple_baselines.run --dataset $d --extended --only hybrid >> results/simple_baselines/hybrid_released.log 2>&1
done
echo DONE >> results/simple_baselines/hybrid_released.log
