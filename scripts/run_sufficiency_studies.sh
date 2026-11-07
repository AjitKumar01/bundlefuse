#!/bin/zsh
set -eu
cd "${0:A:h:h}"
mkdir -p results/sufficiency
exec >> results/sufficiency/queue.log 2>&1
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') sufficiency queue START"
for dataset in pog_dense pog spotify; do
  mkdir -p "results/sufficiency/$dataset"
  if .venv/bin/python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.exit(0 if p.exists() and json.loads(p.read_text()).get("status")=="complete" else 1)' "results/sufficiency/$dataset/summary.json"; then
    print "$dataset already complete; retaining results"
    continue
  fi
  print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $dataset START"
  .venv/bin/python -u -m experiments.sufficiency.run --dataset "$dataset" >> "results/sufficiency/$dataset/run.log" 2>&1
  print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $dataset COMPLETE"
done
print "$(date -u '+%Y-%m-%dT%H:%M:%SZ') sufficiency queue COMPLETE"
