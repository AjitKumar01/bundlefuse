"""Tune and run the paper version of RaMen on Electronic, Food and Clothing (clean graphs: training bundles only).

1. Random search: the same --trials settings drawn from the paper's ranges (ramen_paper_runner.SPACE) for every
   dataset, seed 2306, chosen on validation NDCG@20.
2. The chosen setting with seeds 0 and 1 (three seeds in all, as for the other RaMen runs).
3. Diagnostic: the chosen setting on the published variant (item graph from the released user-item file, which holds
   the test bundles), three seeds. Not a fair comparison; it shows what that graph adds to the corrected model.

From the bundlefuse repository:
PYTHONPATH=. nohup nice -n 10 .venv/bin/python -m experiments.ramen_retail.ramen_paper_search > results/ramen_paper/search.log 2>&1 &
"""
import argparse
import os
import glob
import itertools
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get('RAMEN_REPO', '../baselines/RaMen_repo'));OUT = Path(os.environ.get('RAMEN_OUT', 'results/ramen_paper')).resolve()
PYTHON = os.environ.get('RAMEN_PYTHON', '.venv/bin/python');DATASETS = ['electronic', 'food', 'clothing']
SPACE = dict(L1=range(1, 6), L2=range(1, 6), N=range(1, 6), Z=range(1, 6), H=(4, 8, 16, 32, 64),
             beta=np.round(np.arange(.1, 1, .1), 1), gamma=np.round(np.arange(.1, 1, .1), 1), lambda1=np.round(np.arange(.1, 1, .1), 1))


def run(job):
    dataset, config, seed = job
    args = [f'--{k}={v}' for k, v in config.items()]
    cmd = [PYTHON, '-u', str(Path(__file__).with_name('ramen_paper_runner.py').resolve()), '--dataset', dataset,
           '--seed', str(seed), '--out', str(OUT), '--threads', '2', *args]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if ' val R20 ' in l]
    print(line[-1] if line else f'FAILED {dataset} {config} {seed}\n{r.stderr[-800:]}', flush=True)


def results(variant):
    return [json.loads(Path(f).read_text()) for f in glob.glob(str(OUT/variant/'*_seed2306'/'result.json'))]


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--trials', type=int, default=24)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--variant', default='clean', help="dataset suffix: clean (TF-IDF features) or clean_blip (BLIP text features)")
    p.add_argument('--no-diagnostic', action='store_true', help='skip the published-graph runs');a = p.parse_args();v = a.variant
    rng = np.random.default_rng(20261116)
    configs = [{k: (int(v) if k in ('L1', 'L2', 'N', 'Z', 'H') else float(v)) for k, v in
                ((k, rng.choice(list(s))) for k, s in SPACE.items())} for _ in range(a.trials)]
    OUT.mkdir(parents=True, exist_ok=True);(OUT/'configs.json').write_text(json.dumps(configs, indent=1))
    with ThreadPoolExecutor(a.parallel) as pool:
        list(pool.map(run, [(f'{d}_{v}', c, 2306) for c, d in itertools.product(configs, DATASETS)]))
    chosen = {d: max(results(f'{d}_{v}'), key=lambda r: r['validation'][1])['config'] for d in DATASETS}
    (OUT/f'chosen_{v}.json').write_text(json.dumps(chosen, indent=1));print('chosen', chosen, flush=True)
    with ThreadPoolExecutor(a.parallel) as pool:
        list(pool.map(run, [(f'{d}_{v}', chosen[d], s) for d in DATASETS for s in (0, 1)] +
                           ([] if a.no_diagnostic else [(f'{d}_published', chosen[d], s) for d in DATASETS for s in (2306, 0, 1)])))
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
