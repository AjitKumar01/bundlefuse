"""DIALS sensitivity sweep (Kaggle CPU): one setting changed at a time around the frozen configuration.

Settings (all else as dials.train.CONFIG; selection on validation NDCG@20 as in the frozen runs). A job is
dataset:setting[:seed]; the retail sets run seeds 0, 1, 2, POG_DENSE and Spotify_sparse seed 0 only.
  ref           the frozen configuration (run on the retail sets as a check that Kaggle reproduces the frozen runs;
                POG_DENSE and Spotify_sparse are compared with the frozen seed-0 run and the low-rank ablation's
                seed-0 run, which have the same configuration and seed)
  label_rank    4, 64, 128   rank k of the label SVD (label_factors caps it at K - 1)
  width         4, 64        hidden width of delta_l, rho and the gate (frozen 16)
  rank          32, 128      content rank r (frozen 64)
  hidden        16, 64       content encoder width (frozen 32)
  fast_lr       3e-3, 3e-2   learning rate of the small relation parameters (frozen 1e-2)
Each run is one process with one thread, up to --parallel at once. Runs resume from last.pt; the driver stops its
children at --hours, so the kernel ends and keeps its output before Kaggle's 12-hour limit.

Driver:  python sweep.py --jobs food:ref,food:width4,... --parallel 4 --hours 11 --data DIR --out DIR
One run: python sweep.py --run food:width4 --data DIR --out DIR
"""
import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

GRID = {'ref': {}, 'label_rank4': dict(label_rank=4), 'label_rank64': dict(label_rank=64), 'label_rank128': dict(label_rank=128),
        'width4': dict(width=4), 'width64': dict(width=64), 'rank32': dict(rank=32), 'rank128': dict(rank=128),
        'hidden16': dict(hidden=16), 'hidden64': dict(hidden=64), 'fastlr3e-3': dict(fast_lr=3e-3), 'fastlr3e-2': dict(fast_lr=3e-2)}


def run_one(job, data_dir, out):
    import dials  # pins BLAS threads before numpy loads
    import torch
    from torch import nn
    from triad import train as T
    from triad.evaluate import evaluate
    from triad.model import TRIAD as DIALS, curve

    class Swept(DIALS):
        """DIALS with the width of delta_l, rho and the gate set by config['width'] (modules rebuilt only if it is not 16)."""
        def __init__(self, *args, width=16, **kwargs):
            super().__init__(*args, **kwargs)
            if width != 16:
                self.diagonal = nn.ModuleList(curve(3, width) for _ in range(self.levels));self.prior = curve(1, width)
                self.gate = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, 2 + self.levels))
                nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias, math.log(math.e - 1))  # weights start at 1

    def build(ds, cache, config):
        return Swept(ds.features, ds.levels, ds.train, T.spectral(ds, cache, config), rank=config['rank'], hidden=config['hidden'],
                     width=config.get('width', 16))

    T.build = build  # dials.train.fit and dials.train.load construct models through this name
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    name, setting, seed = (job.split(':') + ['0'])[:3];run = Path(out)/name/setting/f'seed{seed}';cache = run/'cache'  # per run: no concurrent cache writes
    with open(Path(data_dir)/'datasets'/f'{name}.pkl', 'rb') as f:ds = pickle.load(f)
    config = dict(T.CONFIG, **GRID[setting])
    T.fit(ds, int(seed), run, cache, config)
    if not (run/'test.json').exists():
        model = T.load(ds, run, cache)
        T.write_json(run/'test.json', evaluate(model, ds.test_pairs))
    T.log(f'{job} COMPLETE')


def drive(jobs, parallel, hours, data_dir, out):
    started = time.monotonic();pending = list(jobs);running = {}
    logs = Path(out)/'logs';logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for job, p in list(running.items()):
            if p.poll() is not None:print(job, 'exit', p.returncode, flush=True);running.pop(job)
        if time.monotonic() - started > hours*3600:
            for job, p in running.items():p.terminate();print(job, 'stopped at the time limit', flush=True)
            for p in running.values():p.wait()
            break
        while pending and len(running) < parallel:
            job = pending.pop(0);log = open(logs/f"{job.replace(':', '_')}.log", 'a')
            running[job] = subprocess.Popen([sys.executable, __file__, '--run', job, '--data', data_dir, '--out', out],
                                            stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=str(Path(data_dir)/'code')))
            print(job, 'started', flush=True)
        time.sleep(30)
    summary = {}
    for job in jobs:
        name, setting, seed = (job.split(':') + ['0'])[:3];run = Path(out)/name/setting/f'seed{seed}'
        if (run/'selected.json').exists():
            s = json.loads((run/'selected.json').read_text());t = json.loads((run/'test.json').read_text()) if (run/'test.json').exists() else {}
            summary[job] = dict(validation=s['validation'], test=dict(recall20=t.get('recall20'), ndcg20=t.get('ndcg20')), best_epoch=s['best_epoch'],
                                epochs_run=s['epochs_run'], seconds=s['seconds'], parameters=s['parameters'])
        else:summary[job] = 'not finished'
    (Path(out)/'summary.json').write_text(json.dumps(summary, indent=1));print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs');p.add_argument('--run');p.add_argument('--parallel', type=int, default=4);p.add_argument('--hours', type=float, default=11)
    p.add_argument('--data', required=True);p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.run:sys.path.insert(0, str(Path(a.data)/'code'));run_one(a.run, a.data, a.out)
    else:drive(a.jobs.split(','), a.parallel, a.hours, a.data, a.out)
