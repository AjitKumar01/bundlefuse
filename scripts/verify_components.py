"""Verify completed component studies without running any models."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.artifacts import write_json
from bundlefuse.experiments import code_digest
from bundlefuse.provided_splits import ProvidedSplits, file_hash
from bundlefuse.eval.components import METHODS, query_groups
from bundlefuse.component_report import component_report
from bundlefuse.paired_stats import paired_comparison, holm_adjust


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/component_study')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['configuration']['code_sha256'] != code_digest():
        raise RuntimeError('Executed source fingerprint differs from current source')
    verified = []
    recorded_statistics = json.loads((root / 'paired.json').read_text())
    recomputed_statistics = []
    recorded_adjusted = []
    for job in manifest['jobs']:
        if job['status'] != 'complete' or job['returncode'] != 0:
            raise RuntimeError(f'{job["dataset"]} has not completed successfully')
        directory = root / job['dataset']
        result = json.loads((directory / 'result.json').read_text())
        n_items = json.loads((Path(job['data']) / 'count.json').read_text())['#I']
        protocol = ProvidedSplits(job['protocol']['directory'], n_items)
        assert result['protocol'] == protocol.manifest() == job['protocol']
        assert result['test_query_ids'] == sorted(protocol.splits['test'][0])
        assert set(result['methods']) == set(result['per_query']) == set(METHODS)
        assert result['selected'] == json.loads((directory / 'selection.json').read_text())
        assert len(result['query_groups']) == len(protocol.pairs('test'))
        assert result['query_groups'] == query_groups(protocol.pairs('test'))
        assert not (directory / 'validation_cache').exists()
        for filename, digest in job['feature_sha256'].items():
            assert file_hash(Path(job['data']) / filename) == digest
        for filename, digest in protocol.hashes.items():
            if job['saved_copy_equal'] is not None:
                assert file_hash(Path(job['saved_neural_copy']) / filename) == digest
        for method in METHODS:
            rows = result['per_query'][method]
            assert len(rows) == len(protocol.pairs('test'))
            for metric, value in result['methods'][method].items():
                values = np.asarray([row[metric] for row in rows])
                assert np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1 + 1e-12))
                np.testing.assert_allclose(values.mean(), value, rtol=0, atol=1e-12)
        selected = result['selected']
        assert selected['NoIncidence']['alpha_inc'] == selected['NoHigherOrder']['alpha_inc'] == 0
        assert selected['NoRao']['gamma_rao'] == selected['NoHigherOrder']['gamma_rao'] == 0
        assert selected['NoShrinkage']['kappa'] == 0
        assert 'bridge' not in selected['NoBridge']['weights']
        assert 'bridge' not in selected['NoBridge']['prior_keys']
        agreement = directory / 'reference_agreement.json'
        if agreement.exists():
            reference = json.loads(agreement.read_text())
            original = json.loads(Path(reference['reference']).read_text())
            assert original['protocol']['sha256'] == protocol.hashes
            assert original['test_query_ids'] == result['test_query_ids']
            actual = max(abs(a[metric] - b[metric]) for a, b in
                         zip(original['per_query'], result['per_query']['BundleFuse']) for metric in a)
            assert actual == reference['max_absolute_per_query_difference'] and actual <= 1e-9
        for method in METHODS[1:]:
            for metric in ('R@20', 'N@20'):
                recorded = recorded_statistics['datasets'][job['dataset']][method][metric]
                recomputed = paired_comparison(
                    [row[metric] for row in result['per_query']['BundleFuse']],
                    [row[metric] for row in result['per_query'][method]], result['query_groups'])
                for key in recomputed:
                    np.testing.assert_allclose(recomputed[key], recorded[key], rtol=0, atol=1e-14)
                recomputed_statistics.append(recomputed)
                recorded_adjusted.append(recorded)
        verified.append({'dataset': job['dataset'], 'queries': len(result['test_query_ids']),
                         'methods': len(METHODS), 'hashes_match': True, 'constraints_verified': True})
    expected_tests = len(manifest['jobs']) * (len(METHODS) - 1) * 2
    assert recorded_statistics['planned_tests'] == expected_tests
    holm_adjust(recomputed_statistics, planned_tests=expected_tests)
    for actual, expected in zip(recomputed_statistics, recorded_adjusted):
        assert actual['p_holm'] == expected['p_holm'] and actual['reject_05'] == expected['reject_05']
    component_report(manifest, root)
    write_json(root / 'validation.json', {'code_sha256': code_digest(), 'verified': verified,
               'paired_contrasts_recomputed': len(recomputed_statistics),
               'bootstrap_per_contrast': 2000, 'sign_flips_per_contrast': 10000,
               'holm_family': expected_tests})
    (root / 'VALIDATION.md').write_text('# Component-study verification\n\n'
        + f'{len(verified)} dataset jobs complete; every planned method has aligned per-query metrics.\n\n'
        + 'Verified source/input hashes, exact query populations, method selections, ablation constraints, metric bounds and averages, '
          'cache cleanup, and full-model agreement with the preceding verified run where available.\n\n'
        + f'Recomputed all {expected_tests} paired comparisons from the saved per-query metrics and verified canonical duplicate-bundle clusters, '
          'bootstrap intervals, sign-flip p-values, and complete-family Holm adjustment.\n')
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
