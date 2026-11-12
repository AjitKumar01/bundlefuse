#!/bin/zsh
# Bundle-budget queue: DIALS runs 9 at a time (1 thread each), baselines one at a time.
cd /Users/ajit/Projects/Career_change/projects/bundle-dpo/bundlefuse
export PYTHONPATH=.:../dials
L=results/bundle_budget/logs
{
  for d in pog_dense pog; do for c in frac1 frac10 pseudo pseudo_clean; do for s in 0 1 2; do
    [[ $d == pog_dense && $c == pseudo && $s == 0 ]] && continue
    echo "$d $c $s"; done; done; done
} | xargs -P 9 -L 1 zsh -c '.venv/bin/python -m experiments.bundle_budget.train --dataset $0 --condition $1 --seed $2 > '$L'/dials_$0_$1_s$2.log 2>&1; echo "$(date -u +%FT%TZ) dials $0 $1 $2 exit $?"' &
(
  while pgrep -f "bundle_budget.baselines --dataset pog_dense --condition full" >/dev/null; do sleep 30; done
  for d in pog_dense pog; do for c in full frac10 frac1 pseudo pseudo_clean; do
    [[ $d == pog_dense && $c == full ]] && continue
    .venv/bin/python -m experiments.bundle_budget.baselines --dataset $d --condition $c > $L/base_${d}_${c}.log 2>&1
    echo "$(date -u +%FT%TZ) baselines $d $c exit $?"
  done; done
) &
wait
echo "$(date -u +%FT%TZ) QUEUE DONE"
