"""TRIAD and the best simple baseline on the same Bundle-MLLM HitRate@1 questions, with paired tests.

The questions are exactly those of hit1.py (POG, POG_DENSE: every (test bundle, held-out item) pair) and
hit1_spotify.py (Spotify, Spotify_sparse: a fixed sample of 8,192 pairs, at most 5 input items drawn per question):
for model seed s in 0-2 and draw d in 0-4 the generator is default_rng(1000 s + d) and is consumed in the same order,
so TRIAD's means reproduce hit1.json / hit1_spotify.json exactly (asserted). The simple baseline answers the same
15 question sets (s, d). Hit: the true item scores strictly above all 9 negatives.

Simple baseline: the best of the five main simple baselines on validation NDCG@20 with the frequency prior allowed
(results/fair_prior/<name>.json; lambda chosen on validation, 0 allowed), the baseline of the main comparison; also the
best without the prior (lambda = 0, the setting chosen on validation before the prior). Per question, hits are averaged
over the 15 sets; TRIAD - simple is tested with a paired cluster bootstrap and sign flips (clusters: identical bundles),
Holm over the four datasets. Writes results/bundlemllm_protocol/hit1_simple.json.

cd ../bundlefuse && PYTHONPATH=.:../dials nice -n 10 .venv/bin/python -m experiments.bundlemllm_protocol.hit1_simple
"""
import triad  # pins BLAS threads before numpy loads
import json, sys
from pathlib import Path
import numpy as np, torch
from triad import data
from triad.train import load
from triad.evaluate import score_batches
from bundlefuse.provenance import write_json
from experiments.fair_prior.baselines import WithPrior, base_rate
from experiments.mechanisms.statistics import paired_comparison, holm_adjust

DIALS = Path(__file__).resolve().parents[3]/'dials';OUT = Path('results/bundlemllm_protocol')
QUESTIONS, TOKENS, CANS = 8192, 5, 10
PUBLISHED = dict(pog=.4131, pog_dense=.6675, spotify=.8242, spotify_sparse=.7959)  # Bundle-MLLM, Table 2


def simple_models(name, ds):
    m = json.loads(Path(f'results/fair_prior/{name}.json').read_text())['methods']
    model = WithPrior(ds.train, ds.n_items, ds.levels, ds.features);model.prior = base_rate(model.f, ds.val_pairs)
    best = max(m, key=lambda k: m[k]['validation_ndcg20'])
    best0 = max(m, key=lambda k: m[k]['validation'][0][0])  # first row: published setting, lambda = 0
    return model, dict(prior=(best, m[best]['params'], m[best]['lam']), no_prior=(best0, m[best0]['published_params'], 0.))


def simple_scores(model, spec, inputs):
    name, p, lam = spec;model.lam = lam
    return np.concatenate([model.scores(name, p, inputs[a:a + 128]) for a in range(0, len(inputs), 128)])


def pog(name, ds, model, specs):
    pairs = ds.test_pairs;n = ds.n_items;qid = [(i, t) for i, (s, g) in enumerate(pairs) for t in g]
    hits = dict(triad=np.zeros((len(qid), 15)), **{k: np.zeros((len(qid), 15)) for k in specs})
    for seed in (0, 1, 2):
        m = load(ds, DIALS/'runs'/name/f'seed{seed}', DIALS/'cache');rngs = [np.random.default_rng(1000*seed + d) for d in range(5)]
        start = 0;row = 0
        for values in score_batches(m, [s for s, _ in pairs]):
            batch = [s for s, _ in pairs[start:start + len(values)]]
            simple = {key: simple_scores(model, spec, batch) for key, spec in specs.items()}
            for k, v in enumerate(values):
                i = start + k;s, g = pairs[i];allowed = np.setdiff1d(np.arange(n), np.fromiter(set(s) | set(g), int))
                for d in range(5):
                    r = row
                    for t in g:
                        neg = allowed[rngs[d].integers(0, len(allowed), 9)];col = 5*seed + d
                        hits['triad'][r, col] = float(v[t] > v[neg].max())
                        for key, sc in simple.items():hits[key][r, col] = float(sc[k, t] > sc[k, neg].max())
                        r += 1
                row += len(g)
            start += len(values)
    ref = np.array(json.loads((OUT/'hit1.json').read_text())[name]['all'])
    assert np.allclose(hits['triad'].mean(0).reshape(3, 5), ref, atol=1e-12), (name, hits['triad'].mean(0), ref)
    return hits, [i for i, _ in qid]


def spotify(name, ds, model, specs):
    pairs = ds.test_pairs;n = ds.n_items
    allq = [(i, t) for i, (s, g) in enumerate(pairs) for t in sorted(set(g))]
    pick = np.random.default_rng(7818).choice(len(allq), min(QUESTIONS, len(allq)), replace=False);qs = [allq[j] for j in sorted(pick)]
    hits = dict(triad=np.zeros((len(qs), 15)), **{k: np.zeros((len(qs), 15)) for k in specs})
    for seed in (0, 1, 2):
        m = load(ds, DIALS/'runs'/name/f'seed{seed}', DIALS/'cache');cache = m.cache()
        for d in range(5):
            rng = np.random.default_rng(1000*seed + d);inputs = [];cands = []
            for i, t in qs:
                s, g = pairs[i];s = sorted(set(s));x = s if len(s) <= TOKENS else sorted(rng.permutation(s)[:TOKENS].tolist())
                inside = set(s) | set(g);neg = rng.integers(0, n, 4*CANS);neg = [int(c) for c in neg if c not in inside][:CANS - 1]
                while len(neg) < CANS - 1:
                    c = int(rng.integers(0, n))
                    if c not in inside:neg.append(c)
                inputs.append(x);cands.append((t, neg))
            col = 5*seed + d
            with torch.no_grad():
                for a in range(0, len(inputs), 64):
                    v = m(inputs[a:a + 64], cache=cache).numpy()
                    for k, (t, neg) in enumerate(cands[a:a + 64]):hits['triad'][a + k, col] = float(v[k, t] > v[k, neg].max())
            for key, spec in specs.items():
                sc = simple_scores(model, spec, inputs)
                for k, (t, neg) in enumerate(cands):hits[key][k, col] = float(sc[k, t] > sc[k, neg].max())
            print(name, 'seed', seed, 'draw', d, {k: round(v[:, col].mean(), 4) for k, v in hits.items()}, flush=True)
    ref = np.array(json.loads((OUT/'hit1_spotify.json').read_text())[name]['all'])
    assert np.allclose(hits['triad'].mean(0).reshape(3, 5), ref, atol=1e-12), (name, hits['triad'].mean(0), ref)
    return hits, [i for i, _ in qs]


def main(names):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    report = dict(note=__doc__, datasets={});tests = []
    for name in names:
        ds = data.load(name, DIALS/'cache');model, specs = simple_models(name, ds)
        hits, bundle = (pog if name.startswith('pog') else spotify)(name, ds, model, specs)
        q = {k: v.mean(1) for k, v in hits.items()}
        clusters = [str(tuple(sorted(set(ds.test_pairs[i][0]) | set(ds.test_pairs[i][1])))) for i in bundle]
        r = paired_comparison(q['triad'], q['prior'], clusters, bootstrap=2000, permutations=10000, seed=20261119)
        tests.append(dict(r, dataset=name, test='triad_minus_best_simple', baseline=specs['prior'][0]))
        report['datasets'][name] = dict(questions=len(bundle), bundle_mllm_published=PUBLISHED[name],
                                        triad=float(q['triad'].mean()), triad_seed_sd=float(hits['triad'].mean(0).reshape(3, 5).mean(1).std(ddof=1)),
                                        best_simple=dict(method=specs['prior'][0], lam=specs['prior'][2], params=specs['prior'][1], hit1=float(q['prior'].mean())),
                                        best_simple_no_prior=dict(method=specs['no_prior'][0], params=specs['no_prior'][1], hit1=float(q['no_prior'].mean())))
        d = report['datasets'][name]
        print(f"{name}: TRIAD {d['triad']:.4f}  best simple {specs['prior'][0]} (lambda {specs['prior'][2]}) {d['best_simple']['hit1']:.4f}  "
              f"without prior {specs['no_prior'][0]} {d['best_simple_no_prior']['hit1']:.4f}  Bundle-MLLM (published) {PUBLISHED[name]}", flush=True)
        holm_adjust(tests, len(names));report['tests'] = tests;write_json(OUT/'hit1_simple.json', report)
    for t in tests:print(f"{t['dataset']:15s} TRIAD - {t['baseline']}: {t['delta']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] p_holm {t['p_holm']:.4f}")


if __name__ == '__main__':
    main(sys.argv[1:] or ['pog_dense', 'pog', 'spotify_sparse', 'spotify'])
