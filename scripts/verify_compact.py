"""Audit completed compact-pilot inputs, checkpoints, metrics, gates and paired statistics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.artifacts import write_json
from bundlefuse.eval.components import query_groups
from bundlefuse.experiments import code_digest
from bundlefuse.paired_stats import paired_comparison, holm_adjust
from bundlefuse.provided_splits import ProvidedSplits, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/compact_pilot')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['configuration']['code_sha256'] == code_digest(), 'Source differs from executed snapshot'
    paired = json.loads((root / 'paired.json').read_text())
    fresh_stats, saved_stats, checks = [], [], []
    for job in manifest['jobs']:
        assert job['status'] == 'complete' and job['returncode'] == 0
        for filename, digest in job['input_sha256'].items():
            assert file_hash(filename) == digest, filename
        directory = root / job['dataset']
        n = json.loads((Path(job['directory']) / 'count.json').read_text())['#I']
        data = ProvidedSplits(job['directory'], n)
        result = json.loads((directory / 'result.json').read_text())
        assert data.manifest() == result['protocol'] == job['protocol']
        ids = sorted(data.splits['test'][0]); groups = query_groups(data.pairs('test'))
        assert result['test_query_ids'] == ids and result['query_groups'] == groups
        parity = result['parity']
        assert parity == json.loads((directory / 'parity.json').read_text())
        assert parity['queries_verified'] == parity['queries_total'] == len(ids)
        assert parity['score_bit_identity'] and parity['historical_metrics_match']
        assert parity['fused_score_sha256_reference'] == parity['fused_score_sha256_optimized']
        assert file_hash(directory / 'selection.json') == result['selection_sha256']
        selected = json.loads((directory / 'selection.json').read_text())
        assert selected == result['selected']
        reference_path = next(Path(name) for name in job['input_sha256'] if 'component_study' in name)
        reference = json.loads(reference_path.read_text())
        assert result['per_query']['BundleFuse'] == reference['per_query']['BundleFuse']
        names = {'BundleFuse'} | {f'{arm}_s{seed}' for arm in ('Raw', 'Engineered') for seed in (0, 1, 2)}
        assert set(result['methods']) == set(result['per_query']) == names
        for name, rows in result['per_query'].items():
            assert len(rows) == len(ids)
            for key, value in result['methods'][name].items():
                values = np.asarray([row[key] for row in rows])
                # A perfect ranking can round to 1 + one float64 ULP in DCG/IDCG.
                assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1 + 1e-12))
                np.testing.assert_allclose(value, values.mean(), rtol=0, atol=1e-12)
        for arm in ('Raw', 'Engineered'):
            grid = json.loads((directory / 'learned' / arm / 'grid.json').read_text())
            best = max(range(len(grid)), key=lambda i: grid[i]['val_ndcg20'])
            assert selected['selected'][f'{arm}_s0']['checkpoint'] == grid[best]['checkpoint']
            for seed in (0, 1, 2):
                record = selected['selected'][f'{arm}_s{seed}']
                assert record['seed'] == seed and record['hp'] == grid[best]['hp']
                checkpoint = directory / 'learned' / record['checkpoint']
                assert file_hash(checkpoint) == record['checkpoint_sha256']
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                assert record['parameter_count'] == sum(p.numel() for p in state.values()) == 58314
                assert 'features' not in state and 'item_embedding.weight' not in state
                history = json.loads((checkpoint.parent / 'history.json').read_text())
                score, epoch = -float('inf'), None
                for entry in history:
                    if entry.get('val_ndcg20', -float('inf')) > score + 1e-5:
                        score, epoch = entry['val_ndcg20'], entry['epoch']
                assert score == record['val_ndcg20'] and epoch == record['best_epoch']
                assert abs(selected['validation'][f'{arm}_s{seed}']['N@20'] - score) < 1e-12
        gate = selected['gate']
        ratios = {key: selected['validation']['Engineered_s0'][key] / reference['validation']['BundleFuse'][key]
                  for key in ('R@20', 'N@20')}
        assert gate['retention_target'] == .975 and gate['validation_retention'] == ratios
        improvement = selected['validation']['Engineered_s0']['N@20'] > selected['validation']['Raw_s0']['N@20']
        assert gate['engineered_beats_raw_validation_ndcg20'] == improvement
        assert gate['pass_this_dataset'] == (improvement and min(ratios.values()) >= .975)
        for first, second in (('Engineered_s0', 'Raw_s0'), ('Engineered_s0', 'BundleFuse'), ('Raw_s0', 'BundleFuse')):
            for metric in ('R@20', 'N@20'):
                fresh = paired_comparison([r[metric] for r in result['per_query'][first]],
                                           [r[metric] for r in result['per_query'][second]], groups)
                fresh.update(first=first, second=second, metric=metric, dataset=job['dataset'])
                fresh_stats.append(fresh)
        saved_stats.extend(paired['comparisons'][job['dataset']])
        checks.append({'dataset': job['dataset'], 'queries': len(ids), 'methods': len(names),
                       'validation_gate': gate['pass_this_dataset']})
    assert paired['planned_tests'] == len(fresh_stats) == 6 * len(manifest['jobs'])
    holm_adjust(fresh_stats, planned_tests=paired['planned_tests'])
    assert fresh_stats == saved_stats
    write_json(root / 'validation.json', {'datasets': checks, 'recomputed_contrasts': len(fresh_stats),
                                        'code_sha256': code_digest()})
    (root / 'VALIDATION.md').write_text('# Verification\n\nVerified all supplied input/reference/source hashes, '
        'original test IDs/groups, exhaustive score-bit parity, historical per-query agreement, '
        'checkpoint hashes/counts/selection, learning curves, validation gates and every paired statistic.\n')
    print(json.dumps({'verified': checks, 'recomputed_contrasts': len(fresh_stats)}, indent=2))


if __name__ == '__main__':
    main()
