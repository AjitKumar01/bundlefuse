#!/bin/zsh
# Retain independent dataset logs/checkpoints; never delete existing run state.
set -u
cd "${0:A:h:h}"
mkdir -p results/novelty_attribution
exec >> results/novelty_attribution/queue.log 2>&1
failed=0
for dataset in pog_dense spotify pog; do
    mkdir -p "results/novelty_attribution/$dataset"
    print "$(date -u) START $dataset"
    .venv/bin/python -u -m experiments.novelty.run --dataset "$dataset" \
        >> "results/novelty_attribution/$dataset/run.log" 2>&1
    result=$?
    print "$(date -u) END $dataset exit=$result"
    if [[ "$result" != 0 ]]; then failed=1; fi
done
print "$(date -u) QUEUE FINISHED failed=$failed"
exit "$failed"
