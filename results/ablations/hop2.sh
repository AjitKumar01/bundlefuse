#!/bin/zsh
# Retrained DIALS without the second graph hop: 3 seeds on all seven datasets, threads as in the frozen runs.
cd "$(dirname $0)/../.."
job(){ PYTHONPATH=. .venv/bin/python -u -m experiments.ablations.run train --part hop2 --dataset $1 --seed $2 --threads $3 > results/ablations/$1_hop2_seed$2.log 2>&1 }
for s in 0 1 2; do job spotify $s 3 & sleep 20; done
for d in pog_dense pog; do for s in 0 1 2; do job $d $s 1 & sleep 10; done; done
for s in 0 1 2; do job spotify_sparse $s 1 & sleep 10; done
for d in electronic food clothing; do for s in 0 1 2; do job $d $s 1; done; done
wait
PYTHONPATH=. .venv/bin/python -u -m experiments.ablations.run test --part hop2 > results/ablations/hop2_test.log 2>&1
