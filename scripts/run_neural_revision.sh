#!/bin/zsh
set -eu
cd "${0:A:h:h}"
mkdir -p results/neural_revision/electronic
exec >> results/neural_revision/queue.log 2>&1
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') revised independent neural Electronic pilot START"
.venv/bin/python -u -m experiments.neural_revision.run --dataset electronic >> results/neural_revision/electronic/run.log 2>&1
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') revised independent neural Electronic pilot COMPLETE"
