#!/bin/zsh
# v3.1: label relations from cross-fitted label co-purchase SVDs; only a 16x16 mixing matrix is learned. Evaluate every epoch.
cd "$(dirname $0)/../.."
job(){ PYTHONPATH=. .venv/bin/python -u -m experiments.neural_relations.run train --dataset $1 --tag v31 --arm loo --label_mode spectral --eval_every 1 --threads $2 > results/neural_relations/$1_v31_seed0.log 2>&1 }
job spotify_sparse 3 & sleep 30; job pog_dense 1 & sleep 30; job pog 1 &
wait
