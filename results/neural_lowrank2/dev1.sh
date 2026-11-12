#!/bin/zsh
# v2 development: seed 0, validation only.
cd "$(dirname $0)/../.."
job(){ PYTHONPATH=. .venv/bin/python -u -m experiments.neural_lowrank2.run train --dataset $1 --tag v2 --threads $2 > results/neural_lowrank2/$1_v2_seed0.log 2>&1 }
job spotify_sparse 4 & job pog_dense 1 & job pog 1 &
wait
