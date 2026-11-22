"""Training-set size on Electronic, Food and Clothing: test NDCG@20 (mean of seeds 0-2 per query; every e-commerce test
query is clean) of the best simple baseline, the untrained final TRIAD, the original TRIAD and the final TRIAD, with
50, 100, 300, 600 and all training bundles. Paired tests (bootstrap + sign flips, one cluster per query) of the final
and of the original TRIAD against the best simple baseline, Holm over the 15 conditions within each model. Writes
results/small_data/budget_table.json and the figure ../dials/figures/design/training_size.png.

Sources: best simple: results/fair_prior/budget_comparison.json and results/fair_prior/<dataset>[_<cond>].json;
untrained: results/small_data/v2_untrained.json (the untrained model's scores do not depend on the seed);
original and final: results/small_data/budget_rerun/<model>/<dataset>_<cond>/ (experiments.small_data.budget_rerun),
with the full condition from ../dials/runs_original_triad and ../dials/runs.

cd ../bundlefuse && PYTHONPATH=.:../dials_original .venv/bin/python -m experiments.small_data.budget_table
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from triad import data
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust

R = Path('results');DIALS = Path('../dials');NAMES = ('electronic', 'food', 'clothing');CONDS = ('n50', 'n100', 'n300', 'n600', 'full')
LABEL = dict(electronic='Electronic', food='Food', clothing='Clothing')


def seeds_mean(root):return np.mean([[q['ndcg20'] for q in json.loads((root/f'seed{s}'/'test.json').read_text())['queries']] for s in (0, 1, 2)], 0)


def runs(model, name, cond):
    if cond == 'full':return seeds_mean(DIALS/('runs' if model == 'final' else 'runs_original_triad')/name)
    return seeds_mean(R/'small_data'/'budget_rerun'/model/f'{name}_{cond}')


def epochs(model, name, cond):
    root = DIALS/('runs' if model == 'final' else 'runs_original_triad')/name if cond == 'full' else R/'small_data'/'budget_rerun'/model/f'{name}_{cond}'
    return [json.loads((root/f'seed{s}'/'selected.json').read_text())['best_epoch'] for s in (0, 1, 2)]


def simple(name, cond):
    best = json.loads((R/'fair_prior'/'budget_comparison.json').read_text())['families']['ecommerce']['rows'][f'{name}:{cond}']['best_simple']
    m = json.loads((R/'fair_prior'/(f'{name}.json' if cond == 'full' else f'{name}_{cond}.json')).read_text())['methods'][best.replace('+prior', '')]
    return np.array(m['queries' if best.endswith('+prior') else 'queries_without_prior'])[:, 1], best


def main():
    untrained = json.loads((R/'small_data'/'v2_untrained.json').read_text())['ndcg20'];rows, tests = {}, dict(final=[], original=[])
    for name in NAMES:
        size = len(data.load(name, DIALS/'cache').train)
        for cond in CONDS:
            s, best = simple(name, cond);f, o = runs('final', name, cond), runs('original', name, cond);key = f'{name}:{cond}'
            rows[key] = dict(bundles=size if cond == 'full' else int(cond[1:]), best_simple=float(s.mean()), best_simple_method=best,
                             untrained=untrained[key], original=float(o.mean()), final=float(f.mean()),
                             final_epochs=epochs('final', name, cond), original_epochs=epochs('original', name, cond))
            cl = [str(i) for i in range(len(s))]
            for k, v in (('final', f), ('original', o)):tests[k].append(dict(paired_comparison(v, s, cl, bootstrap=2000, permutations=10000, seed=20261121), condition=key))
    for k in tests:holm_adjust(tests[k], len(tests[k]))
    write_json(R/'small_data'/'budget_table.json', dict(note=__doc__, rows=rows, tests=tests))
    by = {k: {t['condition']: t for t in v} for k, v in tests.items()}
    print(f"{'condition':17s} {'simple':>7s} {'untrained':>9s} {'original':>9s} {'final':>7s}  final epochs")
    for c, r in rows.items():
        mark = lambda k: ('+' if by[k][c]['delta'] > 0 else '-') if by[k][c]['p_holm'] < .05 else ' '
        print(f"{c:17s} {r['best_simple']:.4f} {r['untrained']:9.4f} {r['original']:8.4f}{mark('original')} {r['final']:6.4f}{mark('final')}  {r['final_epochs']}")
    for k in tests:print(k, 'significantly behind / ahead of best simple:', sum(t['p_holm'] < .05 and t['delta'] < 0 for t in tests[k]), '/', sum(t['p_holm'] < .05 and t['delta'] > 0 for t in tests[k]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, name in zip(axes, NAMES):
        rr = [rows[f'{name}:{c}'] for c in CONDS];x = [r['bundles'] for r in rr]
        for key, colour, style, label in (('best_simple', '#888888', '-', 'best simple baseline'), ('original', '#2A7AB0', '-', 'original TRIAD'),
                                          ('untrained', '#D9541E', '--', 'TRIAD untrained'), ('final', '#D9541E', '-', 'TRIAD trained')):
            ax.plot(x, [r[key] for r in rr], style, color=colour, lw=2, marker='o', ms=4, label=label)
        ax.set_xscale('log');ax.set_xticks(x, [str(v) for v in x], fontsize=8);ax.minorticks_off();ax.set_title(LABEL[name], fontsize=10)
        ax.set_xlabel('training bundles', fontsize=9);ax.tick_params(labelsize=8);ax.spines[['top', 'right']].set_visible(False);ax.grid(axis='y', color='#eee')
    axes[0].set_ylabel('test NDCG@20', fontsize=9);axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout();fig.savefig(DIALS/'figures'/'design'/'training_size.png', dpi=180);print('wrote figure')


if __name__ == '__main__':
    main()
