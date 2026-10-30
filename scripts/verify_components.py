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
from bundlefuse.eval.components import METHODS
from bundlefuse.component_report import component_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/component_study')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['configuration']['code_sha256'] != code_digest():
        raise RuntimeError('Executed source fingerprint differs from current source')
    verified = []
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
            assert json.loads(agreement.read_text())['max_absolute_per_query_difference'] <= 1e-9
        verified.append({'dataset': job['dataset'], 'queries': len(result['test_query_ids']),
                         'methods': len(METHODS), 'hashes_match': True, 'constraints_verified': True})
    component_report(manifest, root)
    write_json(root / 'validation.json', {'code_sha256': code_digest(), 'verified': verified})
    (root / 'VALIDATION.md').write_text('# Component-study verification\n\n'
        + f'{len(verified)} dataset jobs complete; every planned method has aligned per-query metrics.\n\n'
        + 'Verified source/input hashes, exact query populations, method selections, ablation constraints, metric bounds and averages, '
          'cache cleanup, and full-model agreement with the preceding verified run where available.\n')
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
