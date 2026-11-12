#!/bin/zsh
set -eu
cd "${0:A:h:h}"
mkdir -p results/neural_pog/pog_dense results/neural_pog/pog
touch results/neural_pog/pog_dense/run.log results/neural_pog/pog/run.log
exec >> results/neural_pog/queue.log 2>&1
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') POG neural queue START (POG_DENSE then POG, three seeds each)"
for dataset in pog_dense pog; do
  print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $dataset START"
  .venv/bin/python -u -m experiments.neural_revision.pog --dataset "$dataset" >> "results/neural_pog/$dataset/run.log" 2>&1
  print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $dataset COMPLETE"
done
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') POG neural queue COMPLETE"
