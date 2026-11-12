#!/bin/zsh
# CLHE on our e-commerce split: its four shipped presets, chosen on validation NDCG@20 (seed 2023), then seeds 0 and 1.
HERE="$(cd "$(dirname $0)" && pwd)"
cd "$HERE/../../../baselines/CLHE"
for d in electronic food clothing; do for v in clean published; do for p in pog_dense pog spotify spotify_sparse; do
  echo ${d}_$v $p 2023
done; done; done | xargs -P 4 -L 1 $HERE/job.sh
for d in electronic food clothing; do for v in clean published; do
  best=$(../../bundlefuse/.venv/bin/python -c "
import json,glob;r=[json.load(open(f)) for f in glob.glob('$HERE/${d}_$v/*_seed2023/result.json')]
print(max(r,key=lambda x:x['validation'][1])['preset'])")
  for s in 0 1; do echo ${d}_$v $best $s; done
done; done | xargs -P 4 -L 1 $HERE/job.sh
echo ALL DONE
