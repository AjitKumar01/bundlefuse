"""Resumable five-job learned-baseline / generalization study with immutable inputs and live reports."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .artifacts import write_json
from .baselines.metadata_ranker import GRID
from .experiments import code_digest, job_environment
from .generalization_report import report_generalization
from .paths import DATA_ROOT, PROJECT_ROOT
from .provided_splits import ProvidedSplits, file_hash


DATASETS = ('pog', 'pog_dense', 'dunnhumby_chrono', 'spotify', 'spotify_sparse')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--output', default='results/generalization_study')
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset in args.datasets:
        if dataset == 'dunnhumby_chrono':
            directory = Path(DATA_ROOT) / 'dunnhumby'
            files = [directory / name for name in ('product.csv', 'transaction_data.csv')]
            module, protocol = 'temporal', None
        else:
            directory = Path(DATA_ROOT) / 'bundlemllm' / 'datasets' / dataset
            count = json.loads((directory / 'count.json').read_text())['#I']
            protocol = ProvidedSplits(directory, count).manifest()
            files = [directory / name for name in protocol['sha256']]
            files += [directory / name for name in ('title_emb.npy', 'content_feature.pt', 'description_feature.pt', 'item_info.json') if (directory / name).exists()]
            reference = Path(PROJECT_ROOT) / 'results' / 'component_study' / dataset / 'result.json'
            if not reference.exists():
                raise RuntimeError(f'Missing frozen reference for {dataset}')
            files.append(reference); module = 'fullrank'
        jobs.append({'dataset': dataset, 'module': module, 'directory': str(directory), 'protocol': protocol,
                     'input_sha256': {str(path.resolve()): file_hash(path) for path in files}, 'status': 'pending'})
    config = {'code_sha256': code_digest(), 'datasets': args.datasets, 'grid': GRID,
              'inputs': [{k: v for k, v in job.items() if k != 'status'} for job in jobs]}
    # JSON roundtrip ensures tuple/list configuration equality after a resume.
    config = json.loads(json.dumps(config))
    path = output / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['configuration'] != config:
            raise RuntimeError('Source/configuration/inputs changed; choose a new output directory')
    else:
        import numpy, scipy, sklearn, torch
        manifest = {'configuration': config, 'python': sys.version, 'platform': platform.platform(),
                    'versions': {m.__name__: m.__version__ for m in (numpy, scipy, sklearn, torch)},
                    'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'jobs': jobs}
    def save():
        write_json(path, manifest); report_generalization(manifest, output)
    save()
    for job in manifest['jobs']:
        if job['status'] == 'complete':
            continue
        dataset = job['dataset']; directory = output / dataset; directory.mkdir(exist_ok=True)
        if (directory / 'validation_cache').exists():
            raise RuntimeError(f'Stale interrupted cache: {directory / "validation_cache"}')
        env = job_environment({'mode': 'matrix', 'dataset': 'dunnhumby' if dataset == 'dunnhumby_chrono' else dataset,
                               'splits': 1, 'seed': 0, 'objective': 'ndcg20'}, directory)
        env.update(PROVIDED_CHUNK='64', SEED='0')
        if dataset == 'dunnhumby_chrono':
            env['DUNN_DIR'] = job['directory']
        else:
            env.update(FULLRANK_PROTOCOL='provided', PROVIDED_STUDY='generalization',
                       SPLIT_DIR=job['directory'], DATA_DIR=job['directory'])
        command = [sys.executable, '-u', '-m', 'bundlefuse.eval.' + job['module']]
        job.update(status='running', command=command, started=time.time(),
                   env={key: env[key] for key in ('FORCE_DEVICE', 'BUNDLEFUSE_THREADS', 'BUNDLEFUSE_RESULTS_DIR',
                        'PROVIDED_CHUNK', 'SEED', 'BUNDLE_DS', 'FULLRANK_PROTOCOL', 'PROVIDED_STUDY',
                        'SPLIT_DIR', 'DATA_DIR', 'DUNN_DIR') if key in env})
        save(); print(f'RUN {dataset}: {directory / "run.log"}', flush=True)
        with (directory / 'run.log').open('w') as log:
            completed = subprocess.run(command, env=env, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        job.update(returncode=completed.returncode, seconds=time.time() - job['started'])
        try:
            if completed.returncode:
                raise RuntimeError(f'Worker exited {completed.returncode}')
            from .eval.generalization import METHODS
            from .robust_protocol import CASES
            result = json.loads((directory / 'result.json').read_text())
            if set(result['cases']) != set(CASES):
                raise RuntimeError('Missing stress-test case')
            for case in result['cases'].values():
                records = json.loads((directory / case['per_query_file']).read_text())['per_query']
                if set(case['methods']) != set(METHODS) or set(records) != set(METHODS) or any(len(rows) != len(result['test_query_ids']) for rows in records.values()):
                    raise RuntimeError('Missing method or test query')
            if code_digest() != config['code_sha256']:
                raise RuntimeError('Executed source changed during the experiment')
            for filename, digest in job['input_sha256'].items():
                if file_hash(filename) != digest:
                    raise RuntimeError(f'Input changed: {filename}')
            job['status'] = 'complete'
        except Exception as error:
            job.update(status='failed', error=str(error)); save(); raise
        save(); print(f'COMPLETE {dataset}', flush=True)


if __name__ == '__main__':
    main()
