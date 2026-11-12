#!/bin/zsh
# Spotify_sparse overfitting: (a) slower fast learning rate, (b) stronger weight decay on fast parameters. Evaluate every epoch.
cd "$(dirname $0)/../.."
PYTHONPATH=. .venv/bin/python -u -m experiments.neural_relations.run train --dataset spotify_sparse --tag v3lr003 --arm loo --threads 3 --group_lr .003 --eval_every 1 > results/neural_relations/spotify_sparse_v3lr003_seed0.log 2>&1 &
PYTHONPATH=. .venv/bin/python -u -m experiments.neural_relations.run train --dataset spotify_sparse --tag v3wd --arm loo --threads 3 --fast_weight_decay .1 --eval_every 1 > results/neural_relations/spotify_sparse_v3wd_seed0.log 2>&1 &
wait
