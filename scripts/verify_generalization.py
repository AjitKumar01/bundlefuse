"""Verify all learned-baseline / stress-test outputs and resampling statistics without training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.artifacts import write_json
from bundlefuse.eval.components import query_groups
from bundlefuse.eval.generalization import METHODS
from bundlefuse.experiments import code_digest
from bundlefuse.generalization_report import report_generalization
from bundlefuse.paired_stats import paired_comparison, holm_adjust
from bundlefuse.provided_splits import ProvidedSplits, file_hash
from bundlefuse.robust_protocol import CASES, sparse_pairs
from bundlefuse.temporal_data import load_temporal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/generalization_study')
    args = parser.parse_args()
    root = Path(args.output).resolve(); manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['configuration']['code_sha256'] == code_digest()
    saved_stats = json.loads((root / 'paired.json').read_text())
    recalculated, recorded, checks = [], [], []
    for job in manifest['jobs']:
        assert job['status'] == 'complete' and job['returncode'] == 0
        for filename, digest in job['input_sha256'].items():
            assert file_hash(filename) == digest, filename
        directory = root / job['dataset']; result = json.loads((directory / 'result.json').read_text())
        if job['dataset'] == 'dunnhumby_chrono':
            data = load_temporal(job['directory'])
            original, ids, groups = data['test'], data['test_ids'], data['groups']
            assert result['protocol'] == data['protocol']
        else:
            protocol = job['protocol']; count = json.loads((Path(job['directory']) / 'count.json').read_text())['#I']
            data = ProvidedSplits(job['directory'], count)
            original = data.pairs('test'); ids = sorted(data.splits['test'][0]); groups = query_groups(original)
            assert result['protocol'] == data.manifest() == protocol
        assert result['test_query_ids'] == ids and result['query_groups'] == groups
        assert file_hash(directory / 'selection.json') == result['selection_sha256']
        assert result['selected'] == json.loads((directory / 'selection.json').read_text())
        selected = result['selected']['learned']
        assert selected['selected_grid'] == max(range(len(selected['grid'])), key=lambda i: selected['grid'][i]['val_ndcg20'])
        for name, record in selected['selected'].items():
            assert record['hp'] == selected['grid'][selected['selected_grid']]['hp']
            assert record['seed'] == int(name[-1])
            checkpoint = directory / 'learned' / record['checkpoint']
            if checkpoint.exists():
                assert file_hash(checkpoint) == record['checkpoint_sha256']
            history = json.loads((checkpoint.parent / 'history.json').read_text())
            best, best_epoch = -float('inf'), None
            for epoch in history:
                if 'val_ndcg20' in epoch and epoch['val_ndcg20'] > best + 1e-5:
                    best, best_epoch = epoch['val_ndcg20'], epoch['epoch']
            assert best == record['val_ndcg20'] and best_epoch == record['best_epoch']
        for case in CASES:
            pairs = sparse_pairs(original, ids, case)
            info = result['cases'][case]
            assert hashlib.sha256(json.dumps(pairs).encode()).hexdigest() == info['pairs_sha256']
            assert info['queries_changed'] == sum(set(a[0]) != set(b[0]) for a, b in zip(original, pairs))
            file = json.loads((directory / info['per_query_file']).read_text())
            assert file['methods'] == info['methods']
            assert set(file['per_query']) == set(METHODS)
            for method, rows in file['per_query'].items():
                assert len(rows) == len(ids)
                for metric, value in info['methods'][method].items():
                    values = np.asarray([row[metric] for row in rows])
                    assert np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1 + 1e-12))
                    np.testing.assert_allclose(values.mean(), value, rtol=0, atol=1e-12)
            for method in METHODS[1:]:
                for metric in ('R@20', 'N@20'):
                    fresh = paired_comparison([r[metric] for r in file['per_query']['BundleFuse']],
                                               [r[metric] for r in file['per_query'][method]], groups)
                    saved = saved_stats['datasets'][job['dataset']][case][method][metric]
                    for key, value in fresh.items():
                        np.testing.assert_allclose(value, saved[key], rtol=0, atol=1e-14)
                    recalculated.append(fresh); recorded.append(saved)
        if job['dataset'] != 'dunnhumby_chrono':
            reference_path = next(Path(filename) for filename in job['input_sha256'] if 'component_study' in filename)
            reference = json.loads(reference_path.read_text())
            original_metrics = json.loads((directory / 'case_supplied.json').read_text())['per_query']
            for method in METHODS[:3]:
                for a, b in zip(reference['per_query'][method], original_metrics[method]):
                    for metric in a:
                        assert abs(a[metric] - b[metric]) <= 1e-9
        assert not (directory / 'validation_cache').exists()
        checks.append({'dataset': job['dataset'], 'queries': len(ids), 'methods': len(METHODS), 'cases': len(CASES)})
    family = len(manifest['jobs']) * len(CASES) * (len(METHODS) - 1) * 2
    assert saved_stats['planned_tests'] == len(recalculated) == family
    holm_adjust(recalculated, planned_tests=family)
    for fresh, saved in zip(recalculated, recorded):
        assert fresh['p_holm'] == saved['p_holm'] and fresh['reject_05'] == saved['reject_05']
    report_generalization(manifest, root)
    write_json(root / 'validation.json', {'code_sha256': code_digest(), 'datasets': checks, 'recomputed_contrasts': family})
    (root / 'VALIDATION.md').write_text('# Verification\n\nAll input/source hashes, original query IDs/groups, '
        'sparse-mask membership, frozen selections, checkpoint selection histories, available weight hashes, '
        'metric vectors and means, frozen baseline agreement, temporal day boundaries, and all '
        f'{family} bootstrap/sign-flip contrasts and Holm adjustments passed.\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
