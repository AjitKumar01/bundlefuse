#!/bin/zsh
# Second simple-baseline suite (content + ItemKNN blend, iALS, SGFCF), merged into each summary.json.
cd "$(dirname $0)/../.."
for d in food clothing pog_dense pog spotify_sparse spotify; do echo $d; done | \
  xargs -P 3 -I{} zsh -c 'PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python -m experiments.simple_baselines.run --dataset {} --extended > results/simple_baselines/{}/extended.log 2>&1; echo "{} exit $?"'
echo ALL DONE
