#!/bin/zsh
# RaMen on our e-commerce split. Grid on validation NDCG@20 (seed 2306): lr 1e-3 (paper), N in {2, 4} GAT layers
# (paper: 2 for Electronic, 4 for POG), gamma in {0.1, 0.3, 0.5, 0.7, 0.9}. Then seeds 0 and 1 for the chosen setting.
# Each line passed to job.sh is: dataset lr n_layer_gat gamma seed.
HERE="$(cd "$(dirname $0)" && pwd)"
cd "$HERE/../../../baselines/RaMen_repo"
for d in electronic food clothing; do for v in published clean; do for g in 2 4; do for c in 0.1 0.3 0.5 0.7 0.9; do
  echo ${d}_$v 1e-3 $g $c 2306
done; done; done; done | xargs -P 6 -L 1 $HERE/job.sh
for d in electronic food clothing; do for v in published clean; do
  best=$(../../bundlefuse/.venv/bin/python -c "
import json,glob;r=[json.load(open(f)) for f in glob.glob('$HERE/${d}_$v/*_seed2306/result.json')]
b=max(r,key=lambda x:x['validation'][1]);print(b['lr'],b['n_layer_gat'],b['gamma'])")
  for s in 0 1; do echo ${d}_$v $best $s; done
done; done | xargs -P 6 -L 1 $HERE/job.sh
echo ALL DONE
