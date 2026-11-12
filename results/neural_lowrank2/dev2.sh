#!/bin/zsh
# Spotify_sparse: v2 without the random mask, and v2 without the history view.
cd "$(dirname $0)/../.."
PYTHONPATH=. .venv/bin/python -u -m experiments.neural_lowrank2.run train --dataset spotify_sparse --tag v2 --arm loo --threads 3 > results/neural_lowrank2/spotify_sparse_v2_loo_seed0.log 2>&1 &
PYTHONPATH=. .venv/bin/python -u -m experiments.neural_lowrank2.run train --dataset spotify_sparse --tag v2nohist --disable history --threads 3 > results/neural_lowrank2/spotify_sparse_v2nohist_seed0.log 2>&1 &
wait
