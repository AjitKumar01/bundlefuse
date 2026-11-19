#!/bin/zsh
# Diagnostics queue: equal-steps DIALS runs (8 at a time) and untrained-DIALS scoring (3 at a time).
cd /Users/ajit/Projects/Career_change/projects/bundle-dpo/bundlefuse
export PYTHONPATH=.:../dials
L=results/bundle_budget_retail/logs
{ for d in electronic food clothing; do for c in n50 n100 n300; do for s in 0 1 2; do echo "$d $c $s"; done; done; done; } |
  xargs -P 3 -L 1 zsh -c '.venv/bin/python -m experiments.bundle_budget_retail.diagnostics steps --dataset $0 --condition $1 --seed $2 > '$L'/steps_$0_$1_s$2.log 2>&1; echo "$(date -u +%FT%TZ) steps $0 $1 $2 exit $?"' &
{ for d in electronic food clothing; do for c in full n600 n300 n100 n50; do echo "$d $c"; done; done; } |
  xargs -P 3 -L 1 zsh -c '.venv/bin/python -m experiments.bundle_budget_retail.diagnostics init --dataset $0 --condition $1 > '$L'/init_$0_$1.log 2>&1; echo "$(date -u +%FT%TZ) init $0 $1 exit $?"' &
wait
echo "$(date -u +%FT%TZ) QUEUE DONE"
