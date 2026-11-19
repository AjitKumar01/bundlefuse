"""Official RaMen code on Electronic, Food and Clothing with BLIP title features (<dataset>_{clean,published}_blip,
written by experiments.dials_blip.export_clhe): the same grid and selection as the TF-IDF runs in results/ramen_retail.

Phase 1: lr 1e-3, GAT layers {2, 4}, gamma {0.1, 0.3, 0.5, 0.7, 0.9}, seed 2306, 200 epochs.
Phase 2: the setting with the best validation NDCG@20 per dataset and variant, seeds 0 and 1.
Runs --parallel jobs at once, 2 threads each; finished runs (result.json present) and runs another driver has started
(lock directory in results/ramen_retail/blip_locks) are skipped, so several drivers can share the queue.

.venv/bin/python -m experiments.ramen_retail.blip_grid --parallel 3
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path('../baselines/RaMen_repo');OUT = Path('results/ramen_retail').resolve();RUNNER = Path('experiments/ramen_retail/ramen_runner.py').resolve()
NAMES = [f'{d}_{v}_blip' for d in ('electronic', 'food', 'clothing') for v in ('clean', 'published')]


def done(name, gat, gamma, seed):
    return (OUT/name/f'lr0.001_gat{gat}_gamma{gamma:g}_seed{seed}'/'result.json').exists()


def job(args):
    name, gat, gamma, seed = args
    lock = OUT/'blip_locks'/f'{name}_gat{gat}_gamma{gamma:g}_seed{seed}'
    if done(name, gat, gamma, seed):return
    try:lock.parent.mkdir(parents=True, exist_ok=True);lock.mkdir()  # atomic: a second driver skips jobs already started
    except FileExistsError:return
    subprocess.run(['nice', '-n', '10', '.venv/bin/python', str(RUNNER), '--epochs', '200', '--threads', '2', '--out', str(OUT), '--dataset', name,
                    '--lr', '1e-3', '--n_layer_gat', str(gat), '--gamma', str(gamma), '--seed', str(seed)], cwd=REPO, check=False,
                   stdout=open(OUT/'blip_grid.log', 'a'), stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--parallel', type=int, default=3);a = p.parse_args()
    grid = [(n, g, c, 2306) for n in NAMES for g in (2, 4) for c in (.1, .3, .5, .7, .9)]
    with ThreadPoolExecutor(a.parallel) as pool:list(pool.map(job, grid))
    second = []
    for n in NAMES:
        runs = [json.loads(f.read_text()) for f in (OUT/n).glob('*_seed2306/result.json')]
        best = max(runs, key=lambda r: r['validation'][1]);second += [(n, best['n_layer_gat'], best['gamma'], s) for s in (0, 1)]
    with ThreadPoolExecutor(a.parallel) as pool:list(pool.map(job, second))
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
