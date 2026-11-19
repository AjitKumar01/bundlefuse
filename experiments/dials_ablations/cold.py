"""Cold-target check for the label term's parts: does removing the low-rank part (or the diagonal, or the whole label
term) hurt on held-out items with little or no training history, where the graph term has nothing to say?

Inputs: the per-query top-20 lists already written by the ablation (results/dials_ablations/<part>/<d>/seed{0,1,2}/
test.json) and by the frozen model (../dials/runs/<d>/seed{0,1,2}/test.json). No retraining, no rescoring.
Groups of held-out targets by training frequency f (bundles containing the item): never (f = 0), weak (1 <= f <= 5),
seen (f > 5). Per query and group: recall@20 over that query's targets in the group (queries without such a target
are left out of the group). Clean test queries only, as in Table 3. Averaged over the three seeds, then paired cluster
bootstrap and sign flips on full - ablated (clusters: identical complete bundles), Holm within each part over
(dataset x group) cells with at least 30 queries.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_ablations.cold
"""
import triad  # pins BLAS threads before numpy loads
import json
from collections import Counter
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust

OUT = Path('results/dials_ablations');CACHE = Path(DATA_ROOT)/'dials'/'cache';FROZEN = Path(DATA_ROOT)/'dials'/'runs'
PARTS = ('lowrank', 'diagonal', 'labels', 'crossfit');DATASETS = ('electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse')
GROUPS = dict(never=lambda f: f == 0, weak=lambda f: 1 <= f <= 5, seen=lambda f: f > 5);MIN_QUERIES = 30


def per_group(runs, pairs, freq):
    """{group: array over queries (nan where the query has no target in the group)} of recall@20, seed-averaged."""
    tops = [[set(q['top20']) for q in json.loads((r/'test.json').read_text())['queries']] for r in runs]
    out = {}
    for g, rule in GROUPS.items():
        rows = []
        for i, (_, targets) in enumerate(pairs):
            t = [c for c in set(targets) if rule(freq[c])]
            rows.append(np.nan if not t else np.mean([len(set(t) & top[i])/len(t) for top in tops]))
        out[g] = np.asarray(rows)
    return out


def main():
    report = dict(note=__doc__, datasets={});tests = {p: [] for p in PARTS}
    for name in DATASETS:
        ds = data.load(name, CACHE);freq = Counter(c for b in ds.train for c in set(b))
        keep = np.asarray(ds.clean_test);pairs = [ds.test_pairs[i] for i in keep]
        clusters = np.asarray([str(tuple(sorted(set(s) | set(g)))) for s, g in pairs])
        full = per_group([FROZEN/name/f'seed{s}' for s in (0, 1, 2)], ds.test_pairs, freq)
        report['datasets'][name] = {}
        for g in GROUPS:
            a = full[g][keep];m = ~np.isnan(a)
            cell = dict(queries=int(m.sum()), targets=int(sum(sum(GROUPS[g](freq[c]) for c in set(t)) for _, t in pairs)), full=float(a[m].mean()) if m.any() else None)
            for part in PARTS:
                runs = [OUT/part/name/f'seed{s}' for s in (0, 1, 2)]
                if not all((r/'test.json').exists() for r in runs):continue
                b = per_group(runs, ds.test_pairs, freq)[g][keep];cell[part] = float(b[m].mean()) if m.any() else None
                if m.sum() >= MIN_QUERIES:
                    tests[part].append(dict(paired_comparison(a[m], b[m], clusters[m].tolist(), bootstrap=2000, permutations=10000, seed=20261117),
                                            dataset=name, group=g, queries=int(m.sum())))
            report['datasets'][name][g] = cell
            print(name, g, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cell.items()}, flush=True)
    for records in tests.values():holm_adjust(records, len(records))
    report['tests'] = tests;write_json(OUT/'cold_targets.json', report)
    for part, records in tests.items():
        for t in records:print(part, t['dataset'], t['group'], t['queries'], f"full - ablated {t['delta']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] p_holm {t['p_holm']:.4f}")


if __name__ == '__main__':
    main()
