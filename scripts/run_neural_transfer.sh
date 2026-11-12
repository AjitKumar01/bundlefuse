#!/bin/zsh
set -eu
cd "${0:A:h:h}"
mkdir -p results/neural_transfer/food
exec >> results/neural_transfer/queue.log 2>&1
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') fixed-architecture Food transfer START"
.venv/bin/python -u -m experiments.neural_revision.run --dataset food --phase constrained --fixed-condition graph_pair_no_category --output results/neural_transfer --checkpoints checkpoints/neural_transfer >> results/neural_transfer/food/run.log 2>&1
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') fixed-architecture Food transfer COMPLETE"
