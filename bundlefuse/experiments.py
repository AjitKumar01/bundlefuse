"""Resumable, paired experiment matrix with immutable logs and live README updates.

Run: python -m bundlefuse.experiments --output results/evaluation_refresh
Fullrank uses three split seeds; e-commerce/retail use five internal splits.
Music jobs run alone to respect a 24 GB workstation's memory budget.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import numpy as np
from .artifacts import write_json, summarize_seeds, flat_metrics
from .metrics import TIE_POLICY
from .paths import PROJECT_ROOT, RESULTS_DIR

DATASETS = ('pog', 'pog_dense', 'spotify', 'spotify_sparse', 'electronic', 'food', 'clothing', 'dunnhumby')
FULLRANK = set(DATASETS[:4])
OBJECTIVES = ('ndcg10', 'ndcg20', 'ndcg50', 'recall20', 'mrr', 'ndcgfull', 'nll')
CONTROL_FLAGS = ('BASELINES', 'EASE', 'ABLATE', 'VALGAP', 'WSWEEP', 'CASE_STUDY', 'TIMING',
                 'BACKOFF', 'SELDUMP', 'KFOLD', 'SEL_OBJ', 'TRAIN_FRAC', 'IMG_FEATS', 'DDBC_K',
                 'BASELINES_CAP', 'VAL_N', 'TEST_N', 'SEEDS', 'RAMEN_SEED', 'BUNDLE_DS', 'BRDS', 'NO_AUX')


def code_digest():
    digest = hashlib.sha256()
    for path in sorted((Path(PROJECT_ROOT) / 'bundlefuse').rglob('*.py')):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_jobs(datasets, modes, fullrank_seeds, retail_seeds):
    jobs = []
    for dataset in datasets:
        module = 'fullrank' if dataset in FULLRANK else ('dunnhumby' if dataset == 'dunnhumby' else 'bundlerec')
        for mode in ('matrix', 'baselines'):
            if mode not in modes:
                continue
            for seed in fullrank_seeds if dataset in FULLRANK else [0]:
                jobs.append({'id': f'{mode}_{dataset}_seed{seed}', 'mode': mode, 'dataset': dataset,
                             'module': module, 'seed': seed,
                             'splits': 1 if dataset in FULLRANK else retail_seeds, 'objective': 'ndcg20'})
        if 'selection' in modes and dataset in ('pog', 'electronic', 'dunnhumby'):
            for objective in OBJECTIVES:
                jobs.append({'id': f'selection_{dataset}_{objective}', 'mode': 'selection', 'dataset': dataset,
                             'module': module, 'seed': 0, 'splits': 1, 'objective': objective})
    # Non-music jobs first: two can fit in RAM; the larger music jobs require both slots.
    return sorted(jobs, key=lambda job: job['dataset'].startswith('spotify'))


def job_environment(job, directory):
    env = dict(os.environ)
    for key in CONTROL_FLAGS:
        env.pop(key, None)
    env.update(ENCODER='0', FORCE_DEVICE='cpu', BUNDLEFUSE_THREADS='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONHASHSEED='0', BUNDLEFUSE_RESULTS_DIR=str(directory), MATRIX='1',
               SEEDS=str(job['splits']), RAMEN_SEED=str(job['seed']), SEL_OBJ=job['objective'],
               SPLIT_MODE='random')
    if job['dataset'] in FULLRANK:
        env['BUNDLE_DS'] = job['dataset']
    elif job['dataset'] != 'dunnhumby':
        env['BRDS'] = job['dataset']
    if job['mode'] == 'baselines':
        env['BASELINES'] = '1'
    else:
        env['SELDUMP'] = '1'
    if job['mode'] == 'selection' and job['dataset'] == 'pog':
        env['TEST_N'] = '1500'
    return env


def read_result(job, directory):
    filename = (f"baselines_{job['dataset']}.json" if job['mode'] == 'baselines' else
                f"selcmp_{job['dataset']}_{job['objective']}.json")
    data = json.loads((directory / filename).read_text())
    required = {'Ours', 'SimpleHybrid', 'HardSwitchHybrid'} if job['mode'] != 'baselines' else {
        'ItemKNN', 'P3alpha', 'RP3beta', 'iALS', 'BundleKNN', 'LinearCombiner', 'SimpleHybrid', 'HardSwitchHybrid'}
    if not required <= data['methods'].keys():
        raise ValueError(f"{job['id']}: missing methods {required - data['methods'].keys()}")
    for method, values in data['methods'].items():
        for key, value in values.items():
            if key.startswith(('R@', 'N@')) and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{job['id']}: invalid metric {method} {key}={value}")
    if job["mode"] == "baselines" and data.get("seeds") != job["splits"]:
        raise ValueError(f"{job['id']}: baseline seed count mismatch")
    return data


def matrix_records(job, data):
    if job['dataset'] in FULLRANK:
        return [{'seed': job['seed'], 'methods': data['methods']}]
    if len(data.get('per_seed', [])) != job['splits']:
        raise ValueError(f"{job['id']}: missing per-seed matrix results")
    return data['per_seed']


def summarize(manifest, output):
    results = {'matrix': {}, 'baselines': {}, 'selection': {}}
    for dataset in manifest['datasets']:
        for mode in ('matrix', 'baselines'):
            jobs = [j for j in manifest['jobs'] if j['dataset'] == dataset and j['mode'] == mode]
            if not jobs or any(j['status'] != 'complete' for j in jobs):
                continue
            records, raw = [], []
            for job in jobs:
                data = read_result(job, output / job['id'])
                if mode == 'matrix':
                    records.extend(matrix_records(job, data))
                else:
                    raw.extend(data['per_seed'])
                    for record in data['per_seed']:
                        records.append({'seed': record['seed'], 'methods': {
                            name: flat_metrics({int(k): v for k, v in method['test'].items()})
                            for name, method in record['methods'].items() if 'test' in method}})
            results[mode][dataset] = {'methods': summarize_seeds(records), 'per_seed': records}
            if mode == 'baselines':
                results[mode][dataset]['selected'] = raw
        for job in manifest['jobs']:
            if job['dataset'] == dataset and job['mode'] == 'selection' and job['status'] == 'complete':
                results['selection'].setdefault(dataset, {})[job['objective']] = read_result(job, output / job['id'])['methods']['Ours']
    write_json(output / 'summary.json', results)
    return results


def write_progress(manifest, output):
    summary = summarize(manifest, output)
    done = sum(j['status'] == 'complete' for j in manifest['jobs'])
    failed = sum(j['status'] == 'failed' for j in manifest['jobs'])
    lines = [f"Verified refresh: **{done}/{len(manifest['jobs'])} jobs complete**, {failed} failed.",
             f"Run manifest and full logs: [`{os.path.relpath(output, PROJECT_ROOT)}/manifest.json`]({os.path.relpath(output, PROJECT_ROOT)}/manifest.json).",
             'CPU, one thread per process; deterministic item-ID tie breaks. Fullrank: 3 split seeds; retail/e-commerce: 5 splits.',
             '', '| Dataset | BundleFuse R@20 | SimpleHybrid | HardSwitchHybrid | Best other matched baseline |',
             '|---|---:|---:|---:|---|']
    for dataset in manifest['datasets']:
        matrix = summary['matrix'].get(dataset, {}).get('methods', {})
        baselines = summary['baselines'].get(dataset, {}).get('methods', {})
        def cell(method):
            value = matrix.get(method)
            return f"{value['R@20']:.4f} ± {value['R@20_std']:.4f}" if value else 'pending'
        others = {k: v for k, v in baselines.items() if k not in ('SimpleHybrid', 'HardSwitchHybrid')}
        best = max(others, key=lambda k: others[k]['R@20']) if others else None
        best_text = f"{others[best]['R@20']:.4f} ({best})" if best else 'pending'
        lines.append(f"| {dataset} | {cell('Ours')} | {cell('SimpleHybrid')} | {cell('HardSwitchHybrid')} | {best_text} |")
    lines += ['', 'These are matched split/candidate comparisons. The simple hybrids also receive the available raw',
              'content, taxonomy, and auxiliary interactions. Heavy published models are literature context,',
              'not paired results from this refresh. Historical sensitivity/ablation tables have not been regenerated',
              'unless explicitly listed in this run. Missing DDBC/triple2vec reproduction paths remain open.', '']
    block = '\n'.join(lines)
    (output / 'REPORT.md').write_text('# Evaluation refresh\n\n' + block)
    readme = Path(PROJECT_ROOT) / 'README.md'
    text = readme.read_text()
    start, end = '<!-- BEGIN VERIFIED EXPERIMENTS -->', '<!-- END VERIFIED EXPERIMENTS -->'
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        readme.write_text(before + start + '\n' + block + end + after)
    return summary


def promote_complete_results(manifest, output, summary):
    """Publish canonical CSV/JSON together only after all required seed jobs succeeded."""
    if any(job['status'] != 'complete' for job in manifest['jobs']):
        raise RuntimeError('Incomplete experiment run; canonical artifacts were not replaced')
    if set(manifest["datasets"]) != set(DATASETS):
        return  # subset runs must never replace the complete canonical matrix
    root = Path(RESULTS_DIR)
    if summary['matrix']:
        table = {}
        rows = []
        for dataset, data in summary['matrix'].items():
            table[dataset] = {}
            for method, metrics in data['methods'].items():
                keys = ('R@10', 'N@10', 'R@20', 'N@20')
                table[dataset][method] = [[metrics[k], metrics[k + '_std']] for k in keys]
                rows.append([dataset, method] + [f'{metrics[k]:.8f}' for k in keys] + [f"{metrics['R@20_std']:.8f}"])
        temporary = root / 'matrix_zeroparam.csv.tmp'
        with temporary.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['dataset', 'method', 'R@10', 'N@10', 'R@20', 'N@20', 'R@20_std'])
            writer.writerows(rows)
        write_json(root / 'matrix_zeroparam.json', table)
        os.replace(temporary, root / 'matrix_zeroparam.csv')
        write_json(root / 'matrix_zeroparam.meta.json', {'run': os.path.relpath(output, PROJECT_ROOT),
                   'code_sha256': manifest['code_sha256'], 'tie_policy': TIE_POLICY,
                   'seed_ids': {ds: [r['seed'] for r in d['per_seed']] for ds, d in summary['matrix'].items()}})
    for dataset, data in summary['baselines'].items():
        write_json(root / f'baselines_{dataset}.json', {'schema_version': 2, 'dataset': dataset,
                   'methods': data['methods'], 'per_seed': data['selected'],
                   'seeds': len(data['per_seed']), 'tie_policy': TIE_POLICY,
                   'run': os.path.relpath(output, PROJECT_ROOT), 'code_sha256': manifest['code_sha256']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--modes', nargs='+', choices=('matrix', 'baselines', 'selection'), default=['matrix', 'baselines', 'selection'])
    parser.add_argument('--fullrank-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--retail-seeds', type=int, default=5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', default='results/evaluation_refresh')
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    fingerprint = code_digest()
    jobs = make_jobs(args.datasets, args.modes, args.fullrank_seeds, args.retail_seeds)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['code_sha256'] != fingerprint or [j['id'] for j in manifest['jobs']] != [j['id'] for j in jobs]:
            raise RuntimeError('Run configuration/source changed; use a new output directory')
    else:
        import scipy, sklearn, torch
        manifest = {'schema_version': 1, 'branch': subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                    'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    'code_sha256': fingerprint, 'python': sys.version, 'platform': platform.platform(),
                    'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__},
                    'datasets': args.datasets, 'tie_policy': TIE_POLICY,
                    'jobs': [dict(job, status='pending') for job in jobs]}
    lock = threading.Lock()
    capacity = threading.Condition(); available = [max(1, args.workers)]

    def execute(job):
        if job['status'] == 'complete':
            read_result(job, output / job['id'])
            return
        cost = max(1, args.workers) if job['dataset'].startswith('spotify') else 1
        with capacity:
            capacity.wait_for(lambda: available[0] >= cost)
            available[0] -= cost
        directory = output / job['id']; directory.mkdir(exist_ok=True)
        env = job_environment(job, directory)
        command = [sys.executable, '-u', '-m', 'bundlefuse.eval.' + job['module']]
        with lock:
            job.update(status='running', command=command, started=time.time(),
                       env={key: env[key] for key in env if key not in os.environ or env[key] != os.environ[key]})
            write_json(manifest_path, manifest)
            print('RUN', job['id'], flush=True)
        try:
            with (directory / 'run.log').open('w') as logfile:
                result = subprocess.run(command, env=env, cwd=PROJECT_ROOT, stdout=logfile, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'exit {result.returncode}; see {directory / "run.log"}')
            read_result(job, directory)
            with lock:
                job.update(status='complete', seconds=time.time() - job['started'])
                print('DONE', job['id'], f"{job['seconds']:.0f}s", flush=True)
        except Exception as error:
            with lock:
                job.update(status='failed', error=str(error))
                print('FAILED', job['id'], str(error), flush=True)
        finally:
            with lock:
                write_json(manifest_path, manifest)
                write_progress(manifest, output)
            with capacity:
                available[0] += cost
                capacity.notify_all()
    write_json(manifest_path, manifest)
    write_progress(manifest, output)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for future in as_completed([pool.submit(execute, job) for job in manifest['jobs']]):
            future.result()
    summary = write_progress(manifest, output)
    promote_complete_results(manifest, output, summary)
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
