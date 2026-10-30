"""Fingerprint evaluator inputs without copying or publishing the datasets.

Run before/after experiments; --check verifies that the original inputs are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.artifacts import write_json
from bundlefuse.paths import DATA_ROOT, dataset_dir


def inputs():
    for dataset in ('pog', 'pog_dense', 'spotify', 'spotify_sparse'):
        root = Path(dataset_dir('bundlemllm', dataset))
        bundles = ['bi_full.txt'] if (root / 'bi_full.txt').exists() else [
            'bi_train.txt', 'bi_valid_input.txt', 'bi_valid_gt.txt', 'bi_test_input.txt', 'bi_test_gt.txt']
        names = ['count.json', 'item_info.json', 'content_feature.pt'] + bundles
        names += [name for name in ('title_emb.npy', 'description_feature.pt') if (root / name).exists()]
        yield from (root / name for name in names)
    for dataset in ('electronic', 'food', 'clothing'):
        root = Path(dataset_dir('bundlerec', dataset))
        yield from (root / name for name in ('bundle_item.csv', 'user_bundle.csv', 'session_bundle.csv',
                    'user_item_pretrain.csv', 'item_categories.csv', 'item_titles.csv'))
    root = Path(dataset_dir('dunnhumby', 'dunnhumby'))
    yield from (root / name for name in ('transaction_data.csv', 'product.csv'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/evaluation_refresh/data_manifest.json')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    records = {}
    for path in inputs():
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
                digest.update(block)
        records[str(path.relative_to(DATA_ROOT))] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    if args.check:
        if records != json.loads(Path(args.output).read_text())['inputs']:
            raise RuntimeError('Evaluator input files changed')
        print(f'All {len(records)} evaluator inputs match their recorded SHA-256 fingerprints.')
    else:
        write_json(args.output, {'inputs': records, 'excluded': 'Unused LightGCN/collaborative feature caches are not consumed or hashed.'})
        print(f'Fingerprinted {len(records)} evaluator input files.')


if __name__ == '__main__':
    main()
