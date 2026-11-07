#!/bin/zsh
# Retain every checkpoint and cache; run sequentially to bound memory.
set -u
cd "${0:A:h:h}"
mkdir -p results/amplified_fallback
exec >> results/amplified_fallback/queue.log 2>&1
failed=0
for dataset in electronic food clothing dunnhumby spotify_sparse; do
    mkdir -p "results/amplified_fallback/$dataset"
    if [[ -f "results/amplified_fallback/$dataset/summary.json" ]] && \
       .venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("status")=="complete" and not d.get("smoke") else 1)' \
       "results/amplified_fallback/$dataset/summary.json"; then
        print "$(date -u) SKIP completed $dataset"
        continue
    fi
    print "$(date -u) START $dataset"
    .venv/bin/python -u -m experiments.amplified_fallback.run --dataset "$dataset" \
        >> "results/amplified_fallback/$dataset/run.log" 2>&1
    result=$?
    print "$(date -u) END $dataset exit=$result"
    .venv/bin/python scripts/summarize_amplified_fallback.py
    if [[ "$result" != 0 ]]; then failed=1; fi
done
print "$(date -u) QUEUE FINISHED failed=$failed"
exit "$failed"
