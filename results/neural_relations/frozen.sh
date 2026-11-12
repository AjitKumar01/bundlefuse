#!/bin/zsh
# Frozen v3.1: seeds 1-2 on POG, POG_DENSE, Spotify_sparse; seeds 0-2 on Spotify. Seed 0 of the first three is the development run.
cd "$(dirname $0)/../.."
job(){ PYTHONPATH=. .venv/bin/python -u -m experiments.neural_relations.run frozen --dataset $1 --seed $2 --threads $3 > results/neural_relations/$1_frozen_seed$2.log 2>&1 }
job spotify 0 3 &
for d in pog_dense pog; do for s in 1 2; do job $d $s 1 & sleep 20; done; done
for s in 1 2; do job spotify_sparse $s 2 & sleep 20; done
until grep -q start_epoch results/neural_relations/spotify_frozen_seed0.log 2>/dev/null; do sleep 30; done
for s in 1 2; do job spotify $s 3 & sleep 20; done
wait
