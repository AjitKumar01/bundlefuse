#!/bin/zsh
set -eu
cd "${0:A:h:h}"
mkdir -p results/neural_small
exec >> results/neural_small/queue.log 2>&1
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') independent neural queue START"
for dataset in electronic food clothing; do
  mkdir -p "results/neural_small/$dataset"
  print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $dataset START"
  .venv/bin/python -u -m experiments.neural.run --dataset "$dataset" >> "results/neural_small/$dataset/run.log" 2>&1
  print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $dataset COMPLETE"
done
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') independent neural queue COMPLETE"
