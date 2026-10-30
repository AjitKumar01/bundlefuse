"""Sequential, reproducible POG_dense/Spotify compact-model feasibility pilot."""
import argparse
import fcntl
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from .artifacts import write_json
from .baselines.compact_ranker import GRID
from .compact_report import report_compact
from .experiments import code_digest, job_environment
from .paths import DATA_ROOT, PROJECT_ROOT
from .provided_splits import ProvidedSplits, file_hash


DATASETS = ('pog_dense', 'spotify')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--output', default='results/compact_pilot')
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'Another runner is active in {output}') from None
    jobs = []
    for dataset in args.datasets:
        directory = Path(DATA_ROOT) / 'bundlemllm' / 'datasets' / dataset
        n = json.loads((directory / 'count.json').read_text())['#I']
        protocol = ProvidedSplits(directory, n).manifest()
        files = [directory / name for name in protocol['sha256']]
        files += [directory / name for name in ('title_emb.npy', 'content_feature.pt', 'description_feature.pt',
                                                'item_info.json') if (directory / name).exists()]
        files.append(Path(PROJECT_ROOT) / 'results' / 'component_study' / dataset / 'result.json')
        jobs.append({'dataset': dataset, 'directory': str(directory), 'protocol': protocol, 'status': 'pending',
                     'input_sha256': {str(path.resolve()): file_hash(path) for path in files}})
    plan = Path(PROJECT_ROOT) / 'docs' / 'COMPACT_PILOT.md'
    config = json.loads(json.dumps({'code_sha256': code_digest(), 'plan_sha256': file_hash(plan), 'grid': GRID,
                                   'inputs': [{k: v for k, v in job.items() if k != 'status'} for job in jobs]}))
    path = output / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['configuration'] != config:
            raise RuntimeError('Source/configuration/inputs changed; use a new output directory')
    else:
        import numpy, scipy, sklearn, torch
        manifest = {'configuration': config, 'python': sys.version, 'platform': platform.platform(),
                    'versions': {m.__name__: m.__version__ for m in (numpy, scipy, sklearn, torch)},
                    'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    'jobs': jobs}
    (output / 'PLAN.md').write_text(plan.read_text())
    def save():
        write_json(path, manifest); report_compact(manifest, output)
    save()
    for job in manifest['jobs']:
        if job['status'] == 'complete':
            continue
        directory = output / job['dataset']; directory.mkdir(exist_ok=True)
        # Preserve every prior attempt's log when resuming an interrupted worker.
        attempt = 1
        while (directory / ('run.log' if attempt == 1 else f'attempt_{attempt}.log')).exists():
            attempt += 1
        log = directory / ('run.log' if attempt == 1 else f'attempt_{attempt}.log')
        env = job_environment({'mode': 'matrix', 'dataset': job['dataset'], 'splits': 1,
                               'seed': 0, 'objective': 'ndcg20'}, directory)
        env.update(FULLRANK_PROTOCOL='provided', PROVIDED_STUDY='compact',
                   SPLIT_DIR=job['directory'], DATA_DIR=job['directory'], SEED='0', PROVIDED_CHUNK='64')
        command = [sys.executable, '-u', '-m', 'bundlefuse.eval.fullrank']
        job.update(status='running', started=time.time(), command=command, log=str(log.relative_to(output)),
                   env={key: env[key] for key in ('FULLRANK_PROTOCOL', 'PROVIDED_STUDY', 'SPLIT_DIR',
                        'DATA_DIR', 'FORCE_DEVICE', 'BUNDLEFUSE_THREADS', 'BUNDLEFUSE_RESULTS_DIR', 'SEED',
                        'PYTHONHASHSEED', 'BUNDLE_DS', 'PROVIDED_CHUNK')})
        save(); print(f'RUN {job["dataset"]}: {log}', flush=True)
        with log.open('w') as stream:
            completed = subprocess.run(command, env=env, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
        job.update(returncode=completed.returncode, seconds=time.time() - job['started'])
        try:
            if completed.returncode:
                raise RuntimeError(f'Worker exited {completed.returncode}: {log}')
            result = json.loads((directory / 'result.json').read_text())
            names = {'BundleFuse'} | {f'{arm}_s{seed}' for arm in ('Raw', 'Engineered') for seed in (0, 1, 2)}
            if set(result['methods']) != names or set(result['per_query']) != names:
                raise RuntimeError('Missing comparison method')
            if any(len(rows) != len(result['test_query_ids']) for rows in result['per_query'].values()):
                raise RuntimeError('Missing query records')
            if not result['parity']['score_bit_identity'] or code_digest() != config['code_sha256']:
                raise RuntimeError('Score parity failed or executed source changed')
            for filename, digest in job['input_sha256'].items():
                if file_hash(filename) != digest:
                    raise RuntimeError(f'Input changed: {filename}')
            job['status'] = 'complete'
        except Exception as error:
            job.update(status='failed', error=str(error)); save(); raise
        save(); print(f'COMPLETE {job["dataset"]}', flush=True)


if __name__ == '__main__':
    main()
