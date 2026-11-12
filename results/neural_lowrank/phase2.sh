#!/bin/zsh
# Selected arm: POG seeds 1-2 (one thread each) and Spotify seeds 0-2 in two memory-bounded chains.
cd "$(dirname $0)/../.."
A=loo_mask
job(){ PYTHONPATH=. .venv/bin/python -u -m experiments.neural_lowrank.run train --dataset $1 --arm $A --seed $2 --threads $3 > results/neural_lowrank/$1_${A}_seed$2.log 2>&1 }
for d in pog_dense pog; do for s in 1 2; do job $d $s 1 & done; done
{ job spotify_sparse 0 3; job spotify_sparse 2 3; job spotify 1 3 } &
{ job spotify_sparse 1 3; job spotify 0 3; job spotify 2 3 } &
wait
