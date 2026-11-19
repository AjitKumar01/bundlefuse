"""TRIAD HitRate@1 under the Bundle-MLLM protocol on Spotify and Spotify_sparse (Table 2 of Bundle-MLLM also reports
these). As experiments/bundlemllm_protocol/hit1.py, plus the seed truncation of Bundle-MLLM's BundleTestDataset: a
question shows the model at most num_token = 5 of the test bundle's input items, drawn at random afresh for each
question when there are more (utils.py, commit 7818c05), so TRIAD scores each question from those (at most) 5 seeds.
One question per (test bundle, held-out item); 9 negatives uniform with replacement from items in neither the input
nor the held-out set. Spotify questions are many (one per held-out track), so a fixed random sample of 8,192
questions per dataset is scored (Bundle-MLLM's code scores 2,048), 3 model seeds x 5 draws. Also TRIAD without the
unseen-item bonus (TRIAD_NO_BONUS=1). Writes results/bundlemllm_protocol/hit1_spotify[_no_bonus].json.

cd ../dials && [TRIAD_NO_BONUS=1] PYTHONPATH=.:../bundlefuse ../bundlefuse/.venv/bin/python ../bundlefuse/experiments/bundlemllm_protocol/hit1_spotify.py
"""
import triad, json, os, numpy as np, torch
from pathlib import Path
from triad import data
from triad.train import load
NO_BONUS = os.environ.get('TRIAD_NO_BONUS') == '1'
if NO_BONUS:
    from experiments.fair_prior.no_bonus import wrap;load = wrap(load)
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
QUESTIONS, TOKENS, CANS = 8192, 5, 10;out = {}
for name in ('spotify_sparse', 'spotify'):
    ds = data.load(name, Path('cache'));pairs = ds.test_pairs;n = ds.n_items
    allq = [(i, t) for i, (s, g) in enumerate(pairs) for t in sorted(set(g))]
    pick = np.random.default_rng(7818).choice(len(allq), min(QUESTIONS, len(allq)), replace=False);qs = [allq[j] for j in sorted(pick)]
    res = dict(questions_total=len(allq), questions_scored=len(qs), all=[])
    for seed in (0, 1, 2):
        m = load(ds, Path(f'runs/{name}/seed{seed}'), 'cache');cache = m.cache();hits = []
        for d in range(5):
            rng = np.random.default_rng(1000*seed + d);inputs = [];cands = []
            for i, t in qs:
                s, g = pairs[i];s = sorted(set(s));x = s if len(s) <= TOKENS else sorted(rng.permutation(s)[:TOKENS].tolist())
                inside = set(s) | set(g);neg = rng.integers(0, n, 4*CANS);neg = [int(c) for c in neg if c not in inside][:CANS - 1]
                while len(neg) < CANS - 1:
                    c = int(rng.integers(0, n))
                    if c not in inside:neg.append(c)
                inputs.append(x);cands.append((t, neg))
            h = []
            with torch.no_grad():
                for a in range(0, len(inputs), 64):
                    v = m(inputs[a:a + 64], cache=cache).numpy()
                    for k, (t, neg) in enumerate(cands[a:a + 64]):h.append(float(v[k, t] > v[k, neg].max()))
            hits.append(float(np.mean(h)))
        res['all'].append(hits);print(name, 'seed', seed, 'HitRate@1', np.round(hits, 4), flush=True)
    a = np.array(res['all']);res['mean'] = float(a.mean());res['seed_sd'] = float(a.mean(1).std(ddof=1));out[name] = res
    print(f'{name}: HitRate@1 {a.mean():.4f} (seed sd {res["seed_sd"]:.4f})', flush=True)
json.dump(out, open('../bundlefuse/results/bundlemllm_protocol/' + ('hit1_spotify_no_bonus.json' if NO_BONUS else 'hit1_spotify.json'), 'w'), indent=1)
