#!/bin/zsh
# One CLHE run, from the CLHE repository: job.sh dataset preset seed
/Users/ajit/Projects/Career_change/projects/bundle-dpo/baselines/RaMen_repo/.venv/bin/python ../../bundlefuse/experiments/ramen_retail/clhe_runner.py \
  --epochs 200 --threads 2 --out ../../bundlefuse/results/clhe_retail --dataset $1 --preset $2 --seed $3 2>/dev/null | grep -v "^Train\|^Val\|^seed"
