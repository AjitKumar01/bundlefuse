"""TRIAD ablation training on Kaggle CPU: the same Ablated model and training as experiments/triad_ablations/run.py,
without the bundlefuse imports, reading pickled triad.data.Dataset objects (experiments.dials_sensitivity.export).

A job is part:dataset:seed. Each job is one process with one thread; up to --parallel at once. Runs resume from
last.pt; the driver stops its children at --hours so the kernel ends (and keeps its output) before the 12-hour limit.
A non-finite loss (the leave-one-out ablation) is treated as early stopping, as in run.py.

Driver:  python kaggle.py --jobs loo:pog:0,gate:pog:0 --parallel 4 --hours 11.3 --data DIR --out DIR
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path


def run_one(job, data_dir, out):
    import triad  # pins BLAS threads before numpy loads
    import torch
    from torch import nn
    from triad import train as T
    from triad.evaluate import evaluate
    from triad.model import TRIAD

    class Ablated(TRIAD):
        def __init__(self, *args, part, **kwargs):
            super().__init__(*args, **kwargs);self.part = part

        def forward(self, seeds, bundles=None, cache=None):
            seeds = [torch.as_tensor(sorted(set(s)), dtype=torch.long) for s in seeds]
            if bundles is not None:bundles = [torch.as_tensor(sorted(set(b)), dtype=torch.long) for b in bundles]
            loo = None if self.part == 'loo' else bundles;parts = {}
            if self.part != 'content':parts[0] = self.content_scores(seeds, loo, cache)
            if self.part != 'labels':
                for i in range(self.levels):parts[1 + i] = self.label_scores(i, seeds)
            if self.part != 'graph':parts[1 + self.levels] = self.graph_scores(seeds, loo, cache)
            w = self.gate_weights(seeds, loo)
            if self.part == 'gate':w = torch.ones_like(w)
            return sum(w[:, j, None]*v for j, v in parts.items())

    T.build = lambda ds, cache, config: Ablated(ds.features, ds.levels, ds.train, rank=config['rank'], hidden=config['hidden'], part=config['ablation'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    part, name, seed = job.split(':');run = Path(out)/part/name/f'seed{seed}';cache = Path(out)/'cache'
    with open(Path(data_dir)/'datasets'/f'{name}.pkl', 'rb') as f:ds = pickle.load(f)
    config = dict(T.CONFIG, ablation=part)
    try:T.fit(ds, int(seed), run, cache, config)
    except RuntimeError as error:
        if 'Nonfinite loss' not in str(error):raise
        last = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        last['state']['stale'] = config['patience'];torch.save(last, run/'last.pt')
        T.fit(ds, int(seed), run, cache, config)
        selected = json.loads((run/'selected.json').read_text())
        T.write_json(run/'selected.json', dict(selected, diverged_at_epoch=last['state']['epoch'] + 1))
    if not (run/'test.json').exists():T.write_json(run/'test.json', evaluate(T.load(ds, run, cache), ds.test_pairs))
    T.log(f'{job} COMPLETE')


def drive(jobs, parallel, hours, data_dir, out):
    started = time.monotonic();pending = list(jobs);running = {};logs = Path(out)/'logs';logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for job, p in list(running.items()):
            if p.poll() is not None:print(job, 'exit', p.returncode, flush=True);running.pop(job)
        if time.monotonic() - started > hours*3600:
            for job, p in running.items():p.terminate();print(job, 'stopped at the time limit', flush=True)
            for p in running.values():p.wait()
            break
        while pending and len(running) < parallel:
            job = pending.pop(0)
            running[job] = subprocess.Popen([sys.executable, __file__, '--run', job, '--data', data_dir, '--out', out],
                                            stdout=open(logs/f"{job.replace(':', '_')}.log", 'a'), stderr=subprocess.STDOUT,
                                            env=dict(os.environ, PYTHONPATH=str(Path(data_dir)/'code')))
            print(job, 'started', flush=True)
        time.sleep(30)
    status = {j: (Path(out)/j.split(':')[0]/j.split(':')[1]/f"seed{j.split(':')[2]}"/'test.json').exists() for j in jobs}
    (Path(out)/'status.json').write_text(json.dumps(status, indent=1));print(json.dumps(status, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs');p.add_argument('--run');p.add_argument('--parallel', type=int, default=4);p.add_argument('--hours', type=float, default=11.3)
    p.add_argument('--data', required=True);p.add_argument('--out', required=True);a = p.parse_args()
    if a.run:sys.path.insert(0, str(Path(a.data)/'code'));run_one(a.run, a.data, a.out)
    else:drive(a.jobs.split(','), a.parallel, a.hours, a.data, a.out)
