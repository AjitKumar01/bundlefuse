#!/bin/zsh
# The unchanged model retrained at 4 threads (reference for experiment 1). POG seed 0 is the finished 4-thread check run.
cd "$(dirname $0)/../.."
S=/private/tmp/claude-501/-Users-ajit-Projects-Career-change/2594a4a1-49ad-4214-9ef7-56481f666b97/scratchpad/threads4
until grep -qE "IDENTICAL|DIFFERENT" $S/pog.log; do sleep 30; done
mkdir -p results/dials_v2/reference_4threads/pog results/dials_v2/reference_4threads/logs
cp -R $S/pog/seed0 results/dials_v2/reference_4threads/pog/seed0; cp $S/pog.log results/dials_v2/reference_4threads/logs/pog_seed0.log
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=4
PYTHONPATH=.:../dials .venv/bin/python -u -c "
import dials,torch
from pathlib import Path
from dials import data
from dials import train as T
from dials.evaluate import evaluate
from bundlefuse.provenance import write_json
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
ds=data.load('pog',Path('../dials/cache'));run=Path('results/dials_v2/reference_4threads/pog/seed0')
write_json(run/'test.json',evaluate(T.load(ds,run,Path('../dials/cache')),ds.test_pairs))"
for x in "pog 1" "pog 2" "pog_dense 0" "pog_dense 1" "pog_dense 2"; do echo $x; done | xargs -P 2 -L 1 sh -c \
  'PYTHONPATH=.:../dials .venv/bin/python -u -m experiments.dials_v2.run train --reference --dataset $0 --seed $1 --threads 4 > results/dials_v2/reference_4threads/logs/$0_seed$1.log 2>&1'
until ! pgrep -f "dials_v2.run train" >/dev/null; do sleep 30; done
PYTHONPATH=.:../dials .venv/bin/python -u -m experiments.dials_v2.run compare --datasets pog pog_dense > results/dials_v2/candidate_weight/compare.log 2>&1
echo DONE >> results/dials_v2/candidate_weight/compare.log
