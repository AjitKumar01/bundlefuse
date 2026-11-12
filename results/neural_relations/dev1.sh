#!/bin/zsh
# v3 development: seed 0, leave-one-out without the random mask, validation only.
cd "$(dirname $0)/../.."
job(){ PYTHONPATH=. .venv/bin/python -u -m experiments.neural_relations.run train --dataset $1 --tag v3 --arm loo --threads $2 > results/neural_relations/$1_v3_seed0.log 2>&1 }
job spotify_sparse 3 & job pog_dense 1 & job pog 1 &
wait
