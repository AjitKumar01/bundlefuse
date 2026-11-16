#!/bin/zsh
# Spotify fixed-count queue: DIALS 6 at a time (1 thread each, largest conditions first), baselines 2 at a time.
cd /Users/ajit/Projects/Career_change/projects/bundle-dpo/bundlefuse
export PYTHONPATH=.:../dials
L=results/bundle_budget_retail/logs
{ for c in n3000 n1000 n600 n300 n100 n50; do for d in spotify spotify_sparse; do for s in 0 1 2; do echo "$d $c $s"; done; done; done; } |
  xargs -P 6 -L 1 zsh -c '.venv/bin/python -m experiments.bundle_budget_retail.train --dataset $0 --condition $1 --seed $2 > '$L'/dials_$0_$1_s$2.log 2>&1; echo "$(date -u +%FT%TZ) dials $0 $1 $2 exit $?"' &
{ for d in spotify_sparse spotify; do for c in full n3000 n1000 n600 n300 n100 n50; do echo "$d $c"; done; done; } |
  xargs -P 2 -L 1 zsh -c '.venv/bin/python -m experiments.bundle_budget_retail.baselines --dataset $0 --condition $1 > '$L'/base_$0_$1.log 2>&1; echo "$(date -u +%FT%TZ) baselines $0 $1 exit $?"' &
wait
echo "$(date -u +%FT%TZ) QUEUE DONE"
