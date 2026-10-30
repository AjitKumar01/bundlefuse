"""Verify complete supplied-split experiments and their recorded per-query metrics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.artifacts import write_json
from bundlefuse.experiments import code_digest
from bundlefuse.provided_splits import ProvidedSplits, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/provided_splits')
    args = parser.parse_args()
    output = Path(args.output)
    manifest = json.loads((output / 'manifest.json').read_text())
    if code_digest() != manifest['configuration']['code_sha256']:
        raise RuntimeError('Current scoring/runner source differs from the experiment fingerprint')
    checks = []
    for job in manifest['jobs']:
        assert job['status'] == 'complete' and job['returncode'] == 0, job['dataset']
        directory = output / job['dataset']
        result = json.loads((directory / 'result.json').read_text())
        count = json.loads((Path(job['data']) / 'count.json').read_text())['#I']
        split = ProvidedSplits(job['protocol']['directory'], count)
        assert result['protocol'] == split.manifest() == job['protocol']
        assert result['test_query_ids'] == sorted(split.splits['test'][0])
        assert len(result['per_query']) == len(split.pairs('test'))
        assert result['selected'] == json.loads((directory / 'selection.json').read_text())
        assert not (directory / 'validation_cache').exists()
        for name, digest in job['feature_sha256'].items():
            assert file_hash(Path(job['data']) / name) == digest, name
        if job['saved_copy_equal'] is not None:
            assert all(job['saved_copy_equal'].values())
            for name, digest in split.hashes.items():
                assert file_hash(Path(job['saved_neural_copy']) / name) == digest
        for metric, mean in result['methods']['BundleFuse'].items():
            values = np.array([q[metric] for q in result['per_query']])
            assert np.all(np.isfinite(values)) and np.all((0 <= values) & (values <= 1 + 1e-12))
            np.testing.assert_allclose(values.mean(), mean, rtol=0, atol=1e-12)
        checks.append({'dataset': job['dataset'], 'train': len(split.train),
                       'validation': len(split.pairs('valid')), 'test': len(result['per_query']),
                       'R@20': result['methods']['BundleFuse']['R@20'],
                       'N@20': result['methods']['BundleFuse']['N@20']})
    write_json(output / 'validation.json', {'code_sha256': code_digest(), 'verified': checks})
    lines = ['# Verification', '', f'{len(checks)} supplied-split BundleFuse jobs completed successfully.', '',
             'Verified: unchanged split/feature hashes; exact original query IDs/counts; saved neural split copy',
             'equality where available; selected settings retained; all per-query metrics finite and bounded;',
             'aggregate metrics equal their per-query means; temporary validation caches removed.', '',
             'The scorer/runner fingerprint matches the executed source. No neural training was performed.', '']
    (output / 'VALIDATION.md').write_text('\n'.join(lines))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
