"""Live reports and paired inference for the fixed, prespecified component study."""
import json
import os
from pathlib import Path

from .artifacts import write_json
from .eval.components import METHODS
from .paired_stats import paired_comparison, holm_adjust
from .paths import PROJECT_ROOT


def component_report(manifest, output):
    output = Path(output)
    statistics = {}
    comparisons = []
    lines = ['# Supplied-split controls and component study', '',
             'Every method uses the same supplied training bundles, all validation queries, and all test queries.',
             'Each ablation is retuned on validation with the removed component disabled. All selections are frozen before test scoring.', '',
             '| Dataset | Status | BundleFuse | SimpleHybrid | HardSwitchHybrid |',
             '|---|---|---:|---:|---:|']
    for job in manifest['jobs']:
        dataset = job['dataset']
        if job['status'] != 'complete':
            lines.append(f'| {dataset} | {job["status"]} | pending | pending | pending |')
            continue
        result = json.loads((output / dataset / 'result.json').read_text())
        values = result['methods']
        lines.append(f'| {dataset} | complete | {values["BundleFuse"]["R@20"]:.6f} | '
                     f'{values["SimpleHybrid"]["R@20"]:.6f} | {values["HardSwitchHybrid"]["R@20"]:.6f} |')
        statistics[dataset] = {}
        # Cache immutable per-dataset statistics; aggregation/README updates need not resample old jobs.
        stat_path = output / dataset / 'paired.json'
        if stat_path.exists():
            dataset_stats = json.loads(stat_path.read_text())
        else:
            dataset_stats = {}
            for name in METHODS[1:]:
                dataset_stats[name] = {}
                for metric in ('R@20', 'N@20'):
                    dataset_stats[name][metric] = paired_comparison(
                        [row[metric] for row in result['per_query']['BundleFuse']],
                        [row[metric] for row in result['per_query'][name]], result['query_groups'])
            write_json(stat_path, dataset_stats)
        statistics[dataset] = dataset_stats
        comparisons.extend(value for by_metric in dataset_stats.values() for value in by_metric.values())
    live_table = '\n'.join(lines[5:])
    family = len(manifest['jobs']) * (len(METHODS) - 1) * 2
    holm_adjust(comparisons, planned_tests=family)
    write_json(output / 'paired.json', {'planned_tests': family, 'datasets': statistics,
               'method': 'Cluster bootstrap CI; two-sided paired cluster sign-flip test; Holm across both metrics/all configured datasets and contrasts',
               'conditional_on': 'Fitted configurations and fixed split; not training-seed or split uncertainty'})
    lines += ['', 'R@20 values above. Ablation deltas below are BundleFuse minus the ablated method (percentage points).', '',
              '| Dataset | No incidence | No Rao | No higher-order | No shrinkage | No bridge |',
              '|---|---:|---:|---:|---:|---:|']
    for dataset, stats in statistics.items():
        names = ('NoIncidence', 'NoRao', 'NoHigherOrder', 'NoShrinkage', 'NoBridge')
        lines.append('| ' + dataset + ' | ' + ' | '.join(f'{100 * stats[name]["R@20"]["delta"]:+.3f}' for name in names) + ' |')
    lines += ['', '## Paired control comparisons', '',
              '| Dataset | Control | R@20 gain (pp) | Pointwise cluster-bootstrap 95% CI (pp) | Holm p |',
              '|---|---|---:|---|---:|']
    for dataset, stats in statistics.items():
        for name in ('SimpleHybrid', 'HardSwitchHybrid'):
            record = stats[name]['R@20']
            lo, hi = record['ci95']
            lines.append(f'| {dataset} | {name} | {100 * record["delta"]:+.3f} | '
                         f'[{100 * lo:+.3f}, {100 * hi:+.3f}] | {record["p_holm"]:.4f} |')
    lines += ['', 'Duplicate complete held-out bundles are resampled/flipped as clusters. Means retain every original query.',
              f'Holm correction covers the prespecified family of {family} contrasts (both R@20 and NDCG@20).',
              'Confidence intervals are pointwise. Inference is conditional on these fitted configurations and splits;',
              'it does not measure training/split uncertainty or include published neural summary scores.',
              'The same-information controls use raw content, category/artist/album evidence, co-occurrence, and popularity.',
              'This study does not add a new trained metadata ranker or change the supplied masking protocol.',
              'Commands, hashes, selection records, full logs, and per-query metrics are in the manifest and dataset directories.', '']
    text = '\n'.join(lines)
    (output / 'REPORT.md').write_text(text)
    readme = Path(PROJECT_ROOT) / 'README.md'
    source = readme.read_text()
    start, end = '<!-- BEGIN COMPONENT STUDY -->', '<!-- END COMPONENT STUDY -->'
    if start not in source:
        source += '\n## Supplied-split controls and ablations\n\n' + start + '\n' + end + '\n'
    before, rest = source.split(start, 1); _, after = rest.split(end, 1)
    relative = os.path.relpath(output, PROJECT_ROOT)
    done = sum(job['status'] == 'complete' for job in manifest['jobs'])
    readme.write_text(before + start + f'\n**{done}/{len(manifest["jobs"])} datasets complete.** '
                      'Full BundleFuse, five retuned ablations, SimpleHybrid, and HardSwitchHybrid use all supplied queries.\n\n'
                      + live_table + '\n\n' +
                      f'[Live results, paired uncertainty estimates, and logs]({relative}/REPORT.md).\n\n'
                      '```bash\npython -m bundlefuse.provided_experiments --study components --output results/component_study\n```\n'
                      + end + after)
