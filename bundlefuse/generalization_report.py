"""Live learned-baseline, sparse-input, temporal, and efficiency reports."""
import json
import os
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .eval.generalization import METHODS
from .paired_stats import paired_comparison, holm_adjust
from .paths import PROJECT_ROOT
from .robust_protocol import CASES


def report_generalization(manifest, output):
    output = Path(output)
    family = len(manifest['jobs']) * len(CASES) * (len(METHODS) - 1) * 2
    statistics, comparisons, results = {}, [], {}
    lines = ['# Learned baseline and generalization study', '',
             'Selections use only original validation queries and are frozen before every test case.',
             'The learned ranker consumes raw metadata, categories, and training-only collaborative item IDs.', '',
             '| Dataset | Status | BundleFuse R@20 | Learned seed 0 | Learned mean ± SD (3 seeds) |',
             '|---|---|---:|---:|---:|']
    for job in manifest['jobs']:
        dataset = job['dataset']
        if job['status'] != 'complete':
            lines.append(f'| {dataset} | {job["status"]} | pending | pending | pending |')
            continue
        result = json.loads((output / dataset / 'result.json').read_text()); results[dataset] = result
        values = result['cases']['supplied']['methods']
        seeds = [values[f'MetadataRanker_s{s}']['R@20'] for s in (0, 1, 2)]
        lines.append(f'| {dataset} | complete | {values["BundleFuse"]["R@20"]:.6f} | '
                     f'{seeds[0]:.6f} | {np.mean(seeds):.6f} ± {np.std(seeds):.6f} |')
        path = output / dataset / 'paired.json'
        if path.exists():
            statistics[dataset] = json.loads(path.read_text())
        else:
            stats = {}
            for case in CASES:
                records = json.loads((output / dataset / result['cases'][case]['per_query_file']).read_text())['per_query']
                stats[case] = {}
                for method in METHODS[1:]:
                    stats[case][method] = {}
                    for metric in ('R@20', 'N@20'):
                        stats[case][method][metric] = paired_comparison(
                            [row[metric] for row in records['BundleFuse']], [row[metric] for row in records[method]],
                            result['query_groups'])
            write_json(path, stats); statistics[dataset] = stats
        comparisons.extend(record for case in statistics[dataset].values()
                           for method in case.values() for record in method.values())
    live_table = '\n'.join(lines[5:])
    holm_adjust(comparisons, planned_tests=family)
    write_json(output / 'paired.json', {'planned_tests': family, 'datasets': statistics,
               'conditional_on': 'Fixed fitted models and folds; catalogue assumes static metadata availability'})
    lines += ['', '## Sparse-input stress tests', '',
              '| Dataset | Case | Mean revealed | Changed queries | BundleFuse | HardSwitch | Learned seed 0 | Full − learned gain (pp), 95% CI | Holm p |',
              '|---|---|---:|---:|---:|---:|---:|---|---:|']
    for dataset, result in results.items():
        for case in CASES:
            record = result['cases'][case]; values = record['methods']
            paired = statistics[dataset][case]['MetadataRanker_s0']['R@20']
            lo, hi = paired['ci95']
            lines.append(f'| {dataset} | {case} | {record["mean_revealed"]:.2f} | {record["queries_changed"]} | '
                         f'{values["BundleFuse"]["R@20"]:.6f} | {values["HardSwitchHybrid"]["R@20"]:.6f} | '
                         f'{values["MetadataRanker_s0"]["R@20"]:.6f} | {100 * paired["delta"]:+.3f}, '
                         f'[{100 * lo:+.3f}, {100 * hi:+.3f}] | {paired["p_holm"]:.4f} |')
    lines += ['', '## Measured costs', '',
              '| Dataset | Preprocess (s) | Learned training + validation search (s) | Worker peak RSS (GiB) | Full scoring (ms/query) | Learned warm scoring (ms/query) |',
              '|---|---:|---:|---:|---:|---:|']
    for dataset, result in results.items():
        timing = result['timing']; profile = timing['warm_cache_inference']
        lines.append(f'| {dataset} | {timing["preprocessing_seconds"]:.2f} | '
                     f'{timing["learned_training_and_validation_seconds"]:.2f} | '
                     f'{timing["worker_peak_rss_bytes"] / 2**30:.2f} | '
                     f'{profile["BundleFuse"]["median_scoring_ms_per_query"]:.3f} | '
                     f'{profile["MetadataRanker_s0"]["median_scoring_ms_per_query"]:.3f} |')
    lines += ['', '## Learned-training diagnostics', '',
              '| Dataset | Seed | Best epoch | Epochs run | Stop reason | Best at final check | Validation NDCG@20 |',
              '|---|---:|---:|---:|---|---|---:|']
    for dataset, result in results.items():
        for record in result['selected']['learned']['selected'].values():
            lines.append(f'| {dataset} | {record["seed"]} | {record["best_epoch"]} | {record["epochs_run"]} | '
                         f'{record["stopped"]} | {record["best_at_final_check"]} | {record["val_ndcg20"]:.6f} |')
    lines += ['', 'The metadata ranker uses a shared content/category/item-ID encoder and a nonlinear set-query encoder,',
              'sampled-softmax training, a four-configuration learning-rate/regularization grid, and three training seeds.',
              'Hyperparameters are chosen on seed-0 validation; confirmation seeds use those hyperparameters and their own validation-selected checkpoints.',
              'Learning curves and epoch-budget/early-stopping diagnostics are retained; check those before calling a capped run converged.',
              'Dropped seeds move into hidden targets. All cases retain original folds and complete bundle membership, with at least one revealed item.',
              'Half/quarter cases retain those fractions of the supplied seeds; one-seed queries are unchanged when already minimal.',
              'The temporal retail case uses strictly separated whole-day folds, training-only interactions, and an explicit fixed metadata catalogue.',
              'Its base case uses the original half-hidden mask on the new chronological future fold.',
              'Catalogue introduction dates are unavailable; this is not a historical product-launch simulation.',
              f'The prespecified inference family contains {family} contrasts: all configured datasets/cases, five alternatives, and both R@20/NDCG@20.',
              'Pointwise cluster-bootstrap CIs use 2,000 replicates; 10,000 paired cluster sign flips and Holm adjustment test differences.',
              'Retail clusters are households; POG/Spotify clusters are identical complete bundles. Learned seed 0 is the primary competitor; no best-test-seed selection.',
              'Training-seed dispersion is reported separately and does not make the query-bootstrap CI a training-uncertainty interval.',
              'Warm-cache timings use the same 32 sampled test queries and three repeats. Catalogue encoding, preprocessing, search/training, and full-worker peak RSS are separate.',
              'External embedding extraction/pretraining is excluded; worker RSS is not per-method memory. No latency number includes neural-model training.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    readme = Path(PROJECT_ROOT) / 'README.md'; source = readme.read_text()
    start, end = '<!-- BEGIN GENERALIZATION STUDY -->', '<!-- END GENERALIZATION STUDY -->'
    if start not in source:
        source += '\n## Learned baseline and generalization\n\n' + start + '\n' + end + '\n'
    before, rest = source.split(start, 1); _, after = rest.split(end, 1)
    done = sum(job['status'] == 'complete' for job in manifest['jobs'])
    relative = os.path.relpath(output, PROJECT_ROOT)
    readme.write_text(before + start + f'\n**{done}/{len(manifest["jobs"])} dataset jobs complete.** '
                      'A trained metadata/collaborative ranker (three seeds), sparse-input stress tests, chronological retail evaluation, and measured costs.\n\n'
                      + live_table + '\n\n' + f'[Live results and logs]({relative}/REPORT.md).\n\n'
                      '```bash\npython -m bundlefuse.generalization_experiments --output results/generalization_study\n```\n'
                      + end + after)
