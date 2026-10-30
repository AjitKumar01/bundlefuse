"""Run only BundleFuse on the supplied POG/Spotify neural benchmark splits.

python -m bundlefuse.provided_experiments --output results/provided_splits
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .artifacts import write_json
from .experiments import code_digest, job_environment
from .paths import PROJECT_ROOT, DATA_ROOT
from .provided_splits import ProvidedSplits, file_hash, SPLIT_FILES


DATASETS = ('pog', 'pog_dense', 'spotify', 'spotify_sparse')
REFERENCES = {
    'pog': {'CLHE': 0.0284, 'RaMen': 0.0375, 'source': 'RaMen Table 2'},
    'spotify': {'CLHE': 0.1081, 'RaMen': 0.1091, 'source': 'RaMen Table 2'},
    'pog_dense': {'CLHE': 0.3811, 'source': 'CLHE Table 2'},
    'spotify_sparse': {'CLHE': 0.0980, 'source': 'CLHE Table 2'},
}
REFERENCE_URLS = {'CLHE': 'https://arxiv.org/html/2310.18770',
                  'RaMen': 'https://arxiv.org/html/2507.14361v1'}
SAVED_COPIES = {'pog': 'kaggle_out_pog/RaMen/datasets/pog',
                'pog_dense': 'clhe_out/pog_dense/CLHE/datasets/pog_dense',
                'spotify': 'kaggle_out_spo/RaMen/datasets/spotify',
                'spotify_sparse': 'clhe_out/spotify_sparse/CLHE/datasets/spotify_sparse'}


def report(manifest, output):
    rows = ['BundleFuse only; supplied training bundles and every supplied validation/test query.',
            'No bundle deduplication, resplitting, remasking, or query sampling.', '',
            '| Dataset | Status | BundleFuse R@20 | CLHE reference | RaMen reference | Reference source |',
            '|---|---|---:|---:|---:|---|']
    for job in manifest['jobs']:
        ref = REFERENCES[job['dataset']]
        score = 'pending'
        if job['status'] == 'complete':
            result = json.loads((output / job['dataset'] / 'result.json').read_text())
            score = f"{result['methods']['BundleFuse']['R@20']:.6f}"
        rows.append(f"| {job['dataset']} | {job['status']} | {score} | {ref.get('CLHE', '—')} | "
                    f"{ref.get('RaMen', '—')} | {ref['source']} |")
    rows += ['', 'Reference scores are taken directly from [RaMen Table 2](https://arxiv.org/html/2507.14361v1)',
             'and [CLHE Table 2](https://arxiv.org/html/2310.18770); neural models are not run here.',
             'Both dense/sparse variants already appear in the CLHE paper; earlier local documentation incorrectly',
             'called them newly constructed variants and substituted local reproduction scores.',
             'The supplied files use 70/10/20, consistent with RaMen. The CLHE preprint describes 80/10/10;',
             'its dense/sparse published scores are therefore reference context until that release/paper discrepancy is resolved.',
             'The manifest records hashes of the supplied split files and equality with saved neural-run copies.',
             'Catalogue candidates, seed exclusion, macro Recall/NDCG, and selected settings are recorded per dataset.',
             'Neural scores retain their original tie handling; BundleFuse resolves exact ties by ascending item ID.',
             'This controls the split/mask mismatch. BundleFuse retains its own metadata/features; this is not a',
             'same-feature architecture ablation. Published-score comparisons do not constitute a local neural reproduction.',
             'Electronic/Food are excluded because their public BundleRec data do not establish the published neural split.', '']
    block = '\n'.join(rows)
    (output / 'REPORT.md').write_text('# Provided-split comparison\n\n' + block)
    readme = Path(PROJECT_ROOT) / 'README.md'
    text = readme.read_text()
    start, end = '<!-- BEGIN PROVIDED SPLITS -->', '<!-- END PROVIDED SPLITS -->'
    if start not in text:
        text += '\n## Supplied neural benchmark splits\n\n' + start + '\n' + end + '\n'
    before, remaining = text.split(start, 1)
    _, after = remaining.split(end, 1)
    link = os.path.relpath(output, PROJECT_ROOT)
    readme.write_text(before + start + '\n' + block +
                      f'Full logs, protocol hashes, and per-query metrics: [{link}/REPORT.md]({link}/REPORT.md).\n\n'
                      '```bash\npython -m bundlefuse.provided_experiments --output results/provided_splits\n```\n' + end + after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--output', default='results/provided_splits')
    parser.add_argument('--chunk', type=int, default=64)
    parser.add_argument('--data-root', default=str(Path(DATA_ROOT) / 'bundlemllm' / 'datasets'))
    parser.add_argument('--split-root', help='Optional root containing <dataset>/bi_*.txt and count.json')
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error('--chunk must be positive')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    fingerprint = code_digest()
    jobs = []
    for dataset in args.datasets:
        data = Path(args.data_root).resolve() / dataset
        split = (Path(args.split_root).resolve() / dataset) if args.split_root else data
        count = json.loads((data / 'count.json').read_text())['#I']
        protocol = ProvidedSplits(split, count).manifest()
        saved = Path(DATA_ROOT) / SAVED_COPIES[dataset]
        equality = ({name: file_hash(saved / name) == protocol['sha256'][name] for name in SPLIT_FILES}
                    if saved.exists() else None)
        if equality is not None and not all(equality.values()):
            raise RuntimeError(f'{dataset}: provided split differs from the saved neural-run copy')
        jobs.append({'dataset': dataset, 'data': str(data), 'protocol': protocol,
                     'feature_sha256': {name: file_hash(data / name) for name in
                                        ('title_emb.npy', 'content_feature.pt', 'description_feature.pt', 'item_info.json')
                                        if (data / name).exists()},
                     'saved_neural_copy': str(saved), 'saved_copy_equal': equality,
                     'reference': REFERENCES[dataset], 'reference_urls': REFERENCE_URLS, 'status': 'pending'})
    config = {'code_sha256': fingerprint, 'datasets': args.datasets, 'chunk': args.chunk,
              'inputs': [{k: v for k, v in job.items() if k != 'status'} for job in jobs]}
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['configuration'] != config:
            raise RuntimeError('Source/configuration/input files changed; choose a new output directory')
    else:
        import numpy, scipy, sklearn, torch
        manifest = {'configuration': config, 'python': sys.version, 'platform': platform.platform(),
                    'versions': {m.__name__: m.__version__ for m in (numpy, scipy, sklearn, torch)},
                    'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    'jobs': jobs}
    def save():
        write_json(path, manifest)
        report(manifest, output)
    save()
    for job in manifest['jobs']:
        if job['status'] == 'complete':
            continue
        directory = output / job['dataset']; directory.mkdir(exist_ok=True)
        if (directory / 'validation_cache').exists():
            raise RuntimeError(f'Stale cache from an interrupted run: {directory / "validation_cache"}')
        env = job_environment({'mode': 'matrix', 'dataset': job['dataset'], 'splits': 1,
                               'seed': 0, 'objective': 'ndcg20'}, directory)
        env.update(FULLRANK_PROTOCOL='provided', SPLIT_DIR=job['protocol']['directory'],
                   DATA_DIR=job['data'], PROVIDED_CHUNK=str(args.chunk), SEED='0')
        command = [sys.executable, '-u', '-m', 'bundlefuse.eval.fullrank']
        job.update(status='running', command=command, env={k: v for k, v in env.items()
                   if k in ('FULLRANK_PROTOCOL', 'SPLIT_DIR', 'DATA_DIR', 'PROVIDED_CHUNK', 'FORCE_DEVICE',
                            'SEED', 'BUNDLE_DS', 'SEL_OBJ', 'BUNDLEFUSE_RESULTS_DIR', 'BUNDLEFUSE_THREADS')},
                   started=time.time())
        save()
        print(f'RUN {job["dataset"]}: {directory / "run.log"}', flush=True)
        with (directory / 'run.log').open('w') as log:
            completed = subprocess.run(command, env=env, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        job.update(seconds=time.time() - job['started'], returncode=completed.returncode)
        try:
            if completed.returncode:
                raise RuntimeError(f'exit status {completed.returncode}')
            result = json.loads((directory / 'result.json').read_text())
            if result['protocol'] != job['protocol'] or len(result['per_query']) != job['protocol']['test_queries']:
                raise RuntimeError('Result protocol/query count differs from provided files')
            if code_digest() != fingerprint:
                raise RuntimeError('Source changed while the experiment was running')
            for name, digest in job['protocol']['sha256'].items():
                if file_hash(Path(job['protocol']['directory']) / name) != digest:
                    raise RuntimeError(f'Input changed during the run: {name}')
            for name, digest in job['feature_sha256'].items():
                if file_hash(Path(job['data']) / name) != digest:
                    raise RuntimeError(f'Feature input changed during the run: {name}')
            job['status'] = 'complete'
        except Exception as error:
            job.update(status='failed', error=str(error)); save()
            raise
        save()
        print(f'COMPLETE {job["dataset"]}: {result["methods"]["BundleFuse"]}', flush=True)


if __name__ == '__main__':
    main()
