#!/bin/zsh
# One RaMen run, from the RaMen repository: job.sh dataset lr n_layer_gat gamma seed
.venv/bin/python ../../bundlefuse/experiments/ramen_retail/ramen_runner.py --epochs 200 --threads 2 \
  --out ../../bundlefuse/results/ramen_retail --dataset $1 --lr $2 --n_layer_gat $3 --gamma $4 --seed $5 2>/dev/null | grep -v "^Train\|^Val\|^seed"
