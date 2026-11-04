#!/bin/zsh
# Sequential workers bound memory and retain independent per-dataset logs.
set -u
cd "${0:A:h:h}"
mkdir -p results/publication_mechanisms
exec >> results/publication_mechanisms/queue.log 2>&1
failed=0
for dataset in pog_dense pog spotify spotify_sparse; do
    mkdir -p "results/publication_mechanisms/$dataset"
    python3 -u -m experiments.mechanisms.run --dataset "$dataset" \
        > "results/publication_mechanisms/$dataset/run.log" 2>&1
    result=$?
    print "$(date -u) $dataset exit=$result"
    if [[ "$result" != 0 ]]; then failed=1; fi
done
if [[ "$failed" == 0 ]]; then
    python3 -m experiments.mechanisms.aggregate || failed=1
fi
exit "$failed"
