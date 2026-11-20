"""Question sets for rerunning Bundle-MLLM's released checkpoints, with TRIAD's and the simple baseline's answers.

Test: one question per (test bundle, held-out item) on POG and POG_DENSE, and the fixed sample of 8,192 on the Spotify
sets, each with the negatives and (Spotify) the at most 5 input items of draw 0 / model seed 0 of
experiments.bundlemllm_protocol.hit1_simple (generator default_rng(0), consumed in the same order), so these are
exactly questions TRIAD and the simple baseline already answered there. TRIAD is scored with seeds 0-2 on the same
question (hits averaged); seed 0 must reproduce hit1_simple's draw-0 mean (asserted). Candidate order: the true item and
the 9 negatives shuffled with default_rng(7818); answer = position.

Validation: 1,024 questions per dataset drawn the same way from the validation split (default_rng(4242)), used only to
choose how Bundle-MLLM's missing inputs (English titles, image features) are reconstructed.

Writes results/bundlemllm_rerun/questions_<name>.json.
cd ../bundlefuse && PYTHONPATH=.:../dials nice -n 10 .venv/bin/python -m experiments.bundlemllm_rerun.questions
"""
import triad  # pins BLAS threads before numpy loads
import json, sys
from pathlib import Path
import numpy as np, torch
from triad import data
from triad.train import load
from bundlefuse.provenance import write_json
from experiments.bundlemllm_protocol.hit1_simple import simple_models, simple_scores, DIALS, QUESTIONS, TOKENS, CANS

OUT = Path('results/bundlemllm_rerun')


def pog_questions(pairs, n, rng):
    out = []
    for i, (s, g) in enumerate(pairs):
        allowed = np.setdiff1d(np.arange(n), np.fromiter(set(s) | set(g), int))
        for t in g:out.append(dict(q=i, inputs=sorted(set(s)), target=int(t), negatives=[int(x) for x in allowed[rng.integers(0, len(allowed), 9)]]))
    return out


def spotify_questions(pairs, n, rng, qs):
    out = []
    for i, t in qs:
        s, g = pairs[i];s = sorted(set(s));x = s if len(s) <= TOKENS else sorted(rng.permutation(s)[:TOKENS].tolist())
        inside = set(s) | set(g);neg = rng.integers(0, n, 4*CANS);neg = [int(c) for c in neg if c not in inside][:CANS - 1]
        while len(neg) < CANS - 1:
            c = int(rng.integers(0, n))
            if c not in inside:neg.append(c)
        out.append(dict(q=i, inputs=[int(v) for v in x], target=int(t), negatives=neg))
    return out


def sample(pairs, k, seed):
    allq = [(i, t) for i, (s, g) in enumerate(pairs) for t in sorted(set(g))]
    pick = np.random.default_rng(seed).choice(len(allq), min(k, len(allq)), replace=False);return [allq[j] for j in sorted(pick)]


def shuffle(questions):
    rng = np.random.default_rng(7818)
    for q in questions:
        c = [q['target']] + q['negatives'];order = rng.permutation(len(c));q['candidates'] = [c[j] for j in order]
        q['answer'] = int(np.flatnonzero(order == 0)[0])
    return questions


def main(names):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        ds = data.load(name, DIALS/'cache');n = ds.n_items
        test = (pog_questions(ds.test_pairs, n, np.random.default_rng(0)) if name.startswith('pog')
                else spotify_questions(ds.test_pairs, n, np.random.default_rng(0), sample(ds.test_pairs, QUESTIONS, 7818)))
        val = spotify_questions(ds.val_pairs, n, np.random.default_rng(4242), sample(ds.val_pairs, 1024, 4242))
        triad_hits = np.zeros((len(test), 3))
        for seed in (0, 1, 2):
            m = load(ds, DIALS/'runs'/name/f'seed{seed}', DIALS/'cache');cache = m.cache()
            with torch.no_grad():
                for a in range(0, len(test), 64):
                    v = m([q['inputs'] for q in test[a:a + 64]], cache=cache).numpy()
                    for k, q in enumerate(test[a:a + 64]):triad_hits[a + k, seed] = float(v[k, q['target']] > v[k, q['negatives']].max())
        ref = json.loads(Path('results/bundlemllm_protocol/hit1_simple.json').read_text())
        src = 'hit1.json' if name.startswith('pog') else 'hit1_spotify.json'
        draw0 = json.loads(Path('results/bundlemllm_protocol', src).read_text())[name]['all'][0][0]
        assert abs(triad_hits[:, 0].mean() - draw0) < 1e-12, (name, triad_hits[:, 0].mean(), draw0)
        model, specs = simple_models(name, ds);sc = simple_scores(model, specs['prior'], [q['inputs'] for q in test])
        simple_hits = np.array([float(sc[k, q['target']] > sc[k, q['negatives']].max()) for k, q in enumerate(test)])
        for k, q in enumerate(test):
            q['triad_hit'] = float(triad_hits[k].mean());q['simple_hit'] = float(simple_hits[k])
            s, g = ds.test_pairs[q['q']];q['cluster'] = str(tuple(sorted(set(s) | set(g))))
        report = dict(note=__doc__, dataset=name, n_items=n, simple=dict(method=specs['prior'][0], lam=specs['prior'][2]),
                      test=shuffle(test), validation=shuffle(val), triad_mean=float(triad_hits.mean()), triad_seed0=float(triad_hits[:, 0].mean()),
                      simple_mean=float(simple_hits.mean()), bundle_mllm_published=ref['datasets'][name]['bundle_mllm_published'])
        write_json(OUT/f'questions_{name}.json', report)
        print(name, 'test', len(test), 'val', len(val), 'TRIAD', round(report['triad_mean'], 4), 'seed0', round(report['triad_seed0'], 4),
              'simple', round(report['simple_mean'], 4), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or ['pog_dense', 'pog', 'spotify_sparse', 'spotify'])
