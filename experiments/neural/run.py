"""Run the independent neural pilot on the three smallest bundle datasets."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
import numpy as np
import torch
from bundlefuse.provenance import write_json
from .data import load_data
from .evaluation import evaluate
from .training import fit, load_selected


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def sources():
    paths = sorted(Path('experiments/neural').glob('*.py')) + sorted(Path('bundlefuse').rglob('*.py'))
    paths += [Path('scripts/run_neural_small.sh'), Path('tests/test_neural.py')]
    digest = hashlib.sha256()
    for p in paths:
        digest.update(str(p).encode()); digest.update(p.read_bytes())
    return digest.hexdigest(), paths


def report(output, comparisons, trials, status):
    rows = []
    for name, row in comparisons.items():
        rows.append(f'| {name} | — | {row["recall20"]:.6f} | {row["ndcg20"]:.6f} |')
    for key, row in trials.items():
        rows.append(f'| {key} | {row["selected"]["best_epoch"]} | {row["test"]["recall20"]:.6f} | {row["test"]["ndcg20"]:.6f} |')
    means = {}
    for variant in sorted({r['selected']['variant'] for r in trials.values()}):
        values = np.array([[r['test']['recall20'], r['test']['ndcg20']] for r in trials.values()
                          if r['selected']['variant'] == variant])
        means[variant] = dict(seeds=len(values), recall20_mean=float(values[:,0].mean()),
                             ndcg20_mean=float(values[:,1].mean()),
                             recall20_std=float(values[:,0].std(ddof=1)) if len(values)>1 else None,
                             ndcg20_std=float(values[:,1].std(ddof=1)) if len(values)>1 else None)
    write_json(output/'summary.json', dict(status=status, comparisons=comparisons, trials=trials, neural_means=means))
    text = '# Independent neural pilot\n\nStatus: '+status+'\n\n'
    text += 'All methods use identical full-catalogue test queries. Neural checkpoints are selected on validation NDCG@20.\n\n'
    text += '| Model | Selected epoch | Recall@20 | NDCG@20 |\n|---|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
    text += 'The original BundleFuse row has auxiliary/user histories; the matched row does not. '
    text += 'Neural inputs are item IDs, fixed catalogue text features, category labels and training frequency. '
    text += 'No BundleFuse scores, engineered interaction matrices or distillation are used. '
    text += 'These previously inspected splits support an exploratory pilot, not a fresh confirmatory claim.\n'
    if means:
        text += '\nAcross completed neural seeds (sample standard deviation, not a confidence interval):\n\n'
        for name, m in means.items():
            text += f'- {name}: {m["seeds"]} seeds, mean R20 {m["recall20_mean"]:.6f}, mean N20 {m["ndcg20_mean"]:.6f}.\n'
    (output/'REPORT.md').write_text(text)


def run(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output/args.dataset
    checkpoint_dir = args.checkpoints/args.dataset
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (output/'.runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ds, features, cats, freq, identity = load_data(args.dataset, args.smoke)
        config = dict(dim=args.dim, epochs=args.epochs, lr=args.lr, weight_decay=1e-4,
                      batch_size=16, eval_batch=8, candidate_chunk=256, negative_pool=512,
                      negatives=256, eval_every=2, patience=6)
        source_hash, paths = sources()
        protocol = dict(data=identity, training=config, variants=args.variants, seeds=args.seeds,
                        source_sha256=source_hash, torch_version=torch.__version__,
                        evaluation='full catalogue; observed items masked; item-ID tie breaking',
                        selection='validation NDCG@20, earliest epoch on exact ties',
                        negative_sampling='uniform pool; half current-model hard negatives after epoch 2',
                        comparisons='original BundleFuse, input-matched BundleFuse, six repository defaults')
        for protocol_path in (output/'protocol.json', checkpoint_dir/'protocol.json'):
            if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
                raise RuntimeError('Protocol/source changed: use a new output and checkpoint directory')
        write_json(output/'protocol.json', protocol)
        write_json(checkpoint_dir/'protocol.json', protocol)
        archive = checkpoint_dir/'source.tar.gz'
        if not archive.exists():
            with tarfile.open(archive, 'w:gz') as tar:
                for p in paths:
                    tar.add(p, arcname=str(p))
            write_json(output/'git.json', dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        log(f'{args.dataset}: train={len(ds.train)} val={len(ds.val_pairs)} test={len(ds.test_pairs)} items={ds.n_items}')
        if args.smoke:
            comparisons = {}
        else:
            from .comparisons import run_comparisons
            comparisons = run_comparisons(ds, output, checkpoint_dir/'comparisons', log)
        report(output, comparisons, {}, 'training')
        selected = {}
        # Freeze ALL trial selections before any neural held-out evaluation.
        for seed in args.seeds:
            for variant in args.variants:
                key = f'{variant}_seed{seed}'
                selected[key] = fit(ds, features, cats, freq, variant, seed, config,
                                    checkpoint_dir/key, output/key, log)
        write_json(output/'frozen_selection.json', selected)
        trials = {}
        for key, choice in selected.items():
            result_path = output/key/'test.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                model = load_selected(choice, features, cats, freq)
                result = evaluate(model, ds.test_pairs, config['eval_batch'], config['candidate_chunk'])
                write_json(result_path, result)
            trials[key] = dict(selected=choice, test={k:v for k,v in result.items() if k != 'queries'})
            log(f'{key}: TEST R20={result["recall20"]:.6f} N20={result["ndcg20"]:.6f}')
            report(output, comparisons, trials, 'evaluating')
        report(output, comparisons, trials, 'smoke_complete' if args.smoke else 'complete')
        log(f'{args.dataset}: COMPLETE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['electronic','food','clothing'])
    parser.add_argument('--output', type=Path, default=Path('results/neural_small'))
    parser.add_argument('--checkpoints', type=Path, default=Path('checkpoints/neural_small'))
    parser.add_argument('--variants', nargs='+', choices=['dot','relations','contextual'], default=['dot','relations','contextual'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.dim < 4 or args.dim % 4 or args.lr <= 0:
        parser.error('Need positive epochs/lr and dim divisible by four')
    if args.smoke and (args.output == Path('results/neural_small') or args.checkpoints == Path('checkpoints/neural_small')):
        parser.error('Smoke runs require separate output AND checkpoint directories')
    run(args)


if __name__ == '__main__':
    main()
