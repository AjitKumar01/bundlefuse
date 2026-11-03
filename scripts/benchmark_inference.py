"""Benchmark retained BundleFuse checkpoints without fitting or modifying them.

Each dataset runs in a fresh process. Timed online inference is the public predict(k=20)
call; batched timing includes score_batch plus identical top-20 ranking for every query.
Query loading, checkpoint hashing/loading, warm-up and consistency checks are untimed.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def worker(args):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    from bundlefuse.cli_data import load_data
    from bundlefuse.evaluation import describe_model
    from bundlefuse.metrics import top_items
    from bundlefuse.model import load_model
    from bundlefuse.paths import dataset_dir
    from bundlefuse.provenance import digest_rows, run_metadata
    from bundlefuse.splits import ProvidedSplits, file_hash

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1), torch.inference_mode():
        checkpoint = args.checkpoints / args.worker / 'seed_0/model.pt'
        retention = json.loads(checkpoint.with_name('retention.json').read_text())
        sha = file_hash(checkpoint)
        if sha != retention['sha256'] or not retention['verified']:
            raise RuntimeError('Checkpoint fingerprint or verification mismatch')
        start = time.perf_counter()
        model = load_model(checkpoint)
        load_seconds = time.perf_counter() - start
        spec = model.data_spec
        if spec['dataset'] != args.worker:
            raise RuntimeError('Checkpoint dataset differs from requested dataset')
        start = time.perf_counter()
        if spec['protocol'] == 'provided':
            provided = ProvidedSplits(dataset_dir('bundlemllm', args.worker), model.cfg.n_items)
            for filename, value in provided.hashes.items():
                if spec['input_sha256'][filename] != value:
                    raise RuntimeError(f'Supplied split changed: {filename}')
            pairs = provided.pairs('test')
            ids = sorted(provided.splits['test'][0])
            if spec['test_limit']:
                pairs, ids = pairs[:spec['test_limit']], ids[:spec['test_limit']]
            users = [None] * len(pairs)
        else:
            options = argparse.Namespace(dataset=args.worker, protocol=None, data_profile=None,
                                         split_seed=None, mask_frac=None, test_n=None)
            dataset = load_data(options, model)
            pairs, ids = dataset.test_pairs, dataset.test_ids
            users = dataset.test_users or [None] * len(pairs)
            del dataset
            gc.collect()
        if digest_rows(pairs) != spec['test_pairs_sha256']:
            raise RuntimeError('Test query fingerprint differs from fitted checkpoint')
        query_load_seconds = time.perf_counter() - start
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(pairs), min(args.queries, len(pairs)), replace=False).tolist()
        queries = [(pairs[i][0], users[i]) for i in indices]

        start = time.perf_counter_ns()
        first = model.predict(queries[0][0], k=args.k, user=queries[0][1])
        first_request_ms = (time.perf_counter_ns() - start) / 1e6
        for seeds, user in queries[:args.warmup]:
            model.predict(seeds, k=args.k, user=user)
        print(f'LOADED {args.worker}: {model.cfg.n_items} items, {len(queries)} queries; '
              f'checkpoint {load_seconds:.3f}s; first request {first_request_ms:.3f}ms', flush=True)

        samples = np.zeros((args.repeats, len(queries)), dtype=np.float64)
        expected = {}
        for repeat in range(args.repeats):
            # Different orders reduce repeated-query cache effects while keeping the sample fixed.
            for index in rng.permutation(len(queries)):
                seeds, user = queries[index]
                start = time.perf_counter_ns()
                prediction = model.predict(seeds, k=args.k, user=user)
                samples[repeat, index] = (time.perf_counter_ns() - start) / 1e6
                if repeat == 0:
                    expected[index] = prediction
                elif prediction != expected[index]:
                    raise RuntimeError('Repeated online predictions differ')
            print(f'ONLINE {args.worker} repeat {repeat + 1}/{args.repeats}: '
                  f'mean {samples[repeat].mean():.3f}ms', flush=True)
        if first != expected[0]:
            raise RuntimeError('Cold and warm predictions differ')

        batch_times = []
        for repeat in range(args.repeats):
            for start_index in range(0, len(queries), args.batch_size):
                batch = queries[start_index:start_index + args.batch_size]
                seeds, batch_users = zip(*batch)
                start = time.perf_counter_ns()
                scores = model.score_batch(list(seeds), list(batch_users))
                predictions = [top_items(scores[:, j], s, args.k).tolist() for j, s in enumerate(seeds)]
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                batch_times.append({'repeat': repeat, 'queries': len(batch), 'milliseconds': elapsed_ms})
                if any(len(p) != min(args.k, model.cfg.n_items - len(set(s)))
                       for p, s in zip(predictions, seeds)):
                    raise RuntimeError('Invalid batched prediction length')
        flat = samples.ravel()
        batch_ms = sum(row['milliseconds'] for row in batch_times)
        batch_count = sum(row['queries'] for row in batch_times)
        result = {
            'dataset': args.worker, 'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha,
            'checkpoint_bytes': checkpoint.stat().st_size, 'checkpoint_load_seconds': load_seconds,
            'query_load_seconds_excluded': query_load_seconds, 'first_request_ms': first_request_ms,
            'n_items': model.cfg.n_items, 'model': describe_model(model), 'data': spec,
            'runtime': run_metadata(), 'threadpools': threadpool_info(),
            'unique_test_queries': len(queries), 'test_queries_available': len(pairs),
            'query_ids': [ids[i] for i in indices], 'seed_counts': [len(s) for s, _ in queries],
            'warmup_queries': min(args.warmup, len(queries)), 'repeats': args.repeats,
            'online': {'mean_ms': float(flat.mean()), 'median_ms': float(np.median(flat)),
                       'p95_ms': float(np.percentile(flat, 95)), 'p99_ms': float(np.percentile(flat, 99)),
                       'qps': float(1000 / flat.mean()), 'samples_ms': samples.tolist()},
            'batch': {'size': args.batch_size, 'amortized_ms_per_query': batch_ms / batch_count,
                      'qps': 1000 * batch_count / batch_ms, 'samples': batch_times},
            'repeated_online_predictions_identical': True,
            'checkpoint_unchanged': file_hash(checkpoint) == sha,
        }
        if not result['checkpoint_unchanged']:
            raise RuntimeError('Checkpoint changed during benchmark')
        write_json(args.output / f'{args.worker}.json', result)
        print(f'DONE {args.worker}: median {result["online"]["median_ms"]:.3f}ms; '
              f'p95 {result["online"]["p95_ms"]:.3f}ms; '
              f'batch {result["batch"]["amortized_ms_per_query"]:.3f}ms/query', flush=True)


def report(args, jobs):
    summary = {'status': 'complete' if all(j['status'] == 'complete' for j in jobs) else 'running',
               'hardware': args.hardware, 'cpu_threads': 1, 'top_k': args.k,
               'query_seed': args.seed, 'jobs': jobs}
    write_json(args.output / 'manifest.json', summary)
    rows = []
    for job in jobs:
        if job['status'] != 'complete':
            continue
        r = json.loads((args.output / f'{job["dataset"]}.json').read_text())
        rows.append(f'| {r["dataset"]} | {r["n_items"]:,} | {r["online"]["mean_ms"]:.3f} | '
                    f'{r["online"]["median_ms"]:.3f} | {r["online"]["p95_ms"]:.3f} | '
                    f'{r["batch"]["amortized_ms_per_query"]:.3f} | '
                    f'{r["checkpoint_load_seconds"]:.3f} | {r["first_request_ms"]:.3f} |')
    text = f'''# BundleFuse inference timing

Hardware: {args.hardware}. One CPU thread for PyTorch and native BLAS/OpenMP pools.
Current simplified runtime; retained checkpoints from the original reorganized eight-dataset
run (extended signal/data profile, with the fitted content projection recorded in each JSON).
These timings do not measure the pending corrected research-profile fits.

Online timing covers the public `predict(k={args.k})` call: input checks, selected-signal scoring,
fusion, seed exclusion and top-k ranking over the full catalogue. It excludes checkpoint/data
loading, warm-up and metric evaluation. Each dataset uses {args.queries} uniformly sampled
distinct test queries (or all if fewer), seed {args.seed}, {args.repeats} shuffled repetitions,
and up to {args.warmup} warm-up queries. P95 pools all timed calls; no outliers are removed.
All repetitions return identical online predictions. Each dataset runs in a fresh process,
sequentially. Retained checkpoint SHA-256 values are verified before and after benchmarking.

Batch timing includes `score_batch` and top-k ranking for every query; it is amortized throughput
at batch size {args.batch_size}, not single-request latency. Loading is one unpickle measurement
with OS caching uncontrolled; first request includes any lazy graph reconstruction. Neither
loading nor first request is included in warm latency. Full raw timings/configs are in each
dataset JSON, and worker output is retained in `<dataset>.log`.

| Dataset | Items | Mean ms | Median ms | P95 ms | Batch ms/query | Load s | First request ms |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    (args.output / 'REPORT.md').write_text(text + '\n'.join(rows) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'checkpoints/reorganized_162dd8c')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/inference_timing')
    parser.add_argument('--datasets', nargs='+', default=['electronic', 'food', 'clothing', 'dunnhumby',
                                                       'pog_dense', 'pog', 'spotify', 'spotify_sparse'])
    parser.add_argument('--queries', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--hardware', default='local CPU')
    args = parser.parse_args()
    if min(args.queries, args.repeats, args.warmup, args.batch_size, args.k) < 1:
        parser.error('Query, repetition, warm-up, batch and top-k counts must be positive')
    args.checkpoints, args.output = args.checkpoints.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    jobs = [{'dataset': name, 'status': 'pending'} for name in args.datasets]
    env = dict(os.environ, PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    report(args, jobs)
    for job in jobs:
        name = job['dataset']
        job['status'] = 'running'
        report(args, jobs)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', name,
                   '--checkpoints', str(args.checkpoints), '--output', str(args.output),
                   '--queries', str(args.queries), '--repeats', str(args.repeats),
                   '--warmup', str(args.warmup), '--batch-size', str(args.batch_size),
                   '--k', str(args.k), '--seed', str(args.seed)]
        print(f'START {name}', flush=True)
        with (args.output / f'{name}.log').open('w') as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        job['status'] = 'complete' if process.returncode == 0 else 'failed'
        report(args, jobs)
        print((args.output / f'{name}.log').read_text(), flush=True)
        if process.returncode:
            raise RuntimeError(f'Benchmark failed: {name}; see retained log')
    print('ALL DATASETS COMPLETE', args.output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
