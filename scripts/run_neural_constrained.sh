#!/bin/zsh
set -eu
cd "${0:A:h:h}"
mkdir -p results/neural_constrained/electronic
exec >> results/neural_constrained/queue.log 2>&1
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') constrained neural Electronic pilot START"
.venv/bin/python -u -m experiments.neural_revision.run --dataset electronic --phase constrained --output results/neural_constrained --checkpoints checkpoints/neural_constrained >> results/neural_constrained/electronic/run.log 2>&1
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') constrained neural Electronic pilot COMPLETE"
