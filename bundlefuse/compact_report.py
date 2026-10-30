"""Live compact-model pilot report; primary paired tests and training-seed dispersion."""
import json
import os
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .paired_stats import paired_comparison, holm_adjust
from .paths import PROJECT_ROOT


def report_compact(manifest, output):
    output = Path(output)
    lines = ['# Compact-model feasibility pilot', '',
             'Original supplied inputs only. Direct training; no teacher or sparse-input adaptation.',
             'Raw/engineered arms share architecture, raw columns, parameter count and tuning budget.', '',
             '| Dataset | Status | BundleFuse R@20 | Raw mean ± SD | Engineered mean ± SD | Validation gate |',
             '|---|---|---:|---:|---:|---|']
    stats, records, results = {}, [], {}
    family = len(manifest['jobs']) * 6
    for job in manifest['jobs']:
        dataset = job['dataset']
        if job['status'] != 'complete':
            lines.append(f'| {dataset} | {job["status"]} | pending | pending | pending | pending |')
            continue
        result = json.loads((output / dataset / 'result.json').read_text())
        results[dataset] = result
        values = result['methods']
        def summary(variant):
            scores = [values[f'{variant}_s{seed}']['R@20'] for seed in (0, 1, 2)]
            return f'{np.mean(scores):.6f} ± {np.std(scores):.6f}'
        lines.append(f'| {dataset} | complete | {values["BundleFuse"]["R@20"]:.6f} | {summary("Raw")} | '
                     f'{summary("Engineered")} | {result["selected"]["gate"]["pass_this_dataset"]} |')
        stats[dataset] = []
        for first, second in (('Engineered_s0', 'Raw_s0'), ('Engineered_s0', 'BundleFuse'), ('Raw_s0', 'BundleFuse')):
            for metric in ('R@20', 'N@20'):
                paired = paired_comparison([r[metric] for r in result['per_query'][first]],
                                           [r[metric] for r in result['per_query'][second]], result['query_groups'])
                paired.update(first=first, second=second, metric=metric, dataset=dataset)
                stats[dataset].append(paired); records.append(paired)
    holm_adjust(records, planned_tests=family)
    write_json(output / 'paired.json', {'planned_tests': family, 'comparisons': stats})
    table = '\n'.join(lines[5:])
    lines += ['', '## Warm-cache scoring costs', '',
              '| Dataset | Reference ms/query | Optimized ms/query | Raw ms/query | Engineered ms/query |',
              '|---|---:|---:|---:|---:|']
    for dataset, result in results.items():
        times = result['timing']['warm_cache']
        numbers = [times[name]['median_scoring_ms_per_query'] for name in
                   ('BundleFuse_reference', 'BundleFuse_optimized', 'Raw_s0', 'Engineered_s0')]
        lines.append(f'| {dataset} | ' + ' | '.join(f'{n:.4f}' for n in numbers) + ' |')
    lines += ['', 'Timings are amortized over the same 32-query CPU batch after warmup; three repeats.',
              'Scoring-plus-metrics, offline preprocessing, encoding, training and cache bytes are in each timing.json.',
              'Per-arm seed-0 validation selects learning rate; seeds 1/2 are independent confirmations, not an ensemble.',
              'The 97.5% validation-retention target is an engineering feasibility gate, not a statistical equivalence test.',
              'Query CIs condition on the fitted model; seed SD describes a separate source of variation.',
              'No later stage starts automatically from test results. This pilot cannot establish that every low-rank model fails.', '',
              '## Logs', '']
    for job in manifest['jobs']:
        path = job.get('log', f'{job["dataset"]}/run.log')
        lines.append(f'- [{job["dataset"]}]({path}) — {job["status"]}')
    lines += ['', 'Protocol and limits: [PLAN.md](PLAN.md).', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    readme = Path(PROJECT_ROOT) / 'README.md'
    source = readme.read_text()
    begin, end = '<!-- BEGIN COMPACT PILOT -->', '<!-- END COMPACT PILOT -->'
    if begin not in source:
        source += '\n## Compact-model feasibility pilot\n\n' + begin + '\n' + end + '\n'
    before, rest = source.split(begin, 1); _, after = rest.split(end, 1)
    count = sum(job['status'] == 'complete' for job in manifest['jobs'])
    relative = os.path.relpath(output, PROJECT_ROOT)
    readme.write_text(before + begin + f'\n**{count}/{len(manifest["jobs"])} dataset jobs complete.** '
                      'Exact selected-signal optimization plus a raw-versus-engineered compact two-tower pilot.\n\n'
                      + table + f'\n\n[Protocol, live results and logs]({relative}/REPORT.md). '
                      'No distillation or sparse-input adaptation in this stage.\n' + end + after)
