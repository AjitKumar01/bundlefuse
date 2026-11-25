"""Fresh random splits of Electronic, Food and Clothing: does the learned prior still rank never-bundled items above
once-bundled ones, and does TRIAD stay ahead of the best simple baseline?

For split seeds 1 and 2 (the published split is seed 0), the 70/10/20 split of triad.data.load_resplit is redrawn;
TRIAD (dials main, config_for) is trained with model seeds 0-2 and the simple baselines are retuned on validation
(triad.baselines.run). Reported per split:
  rho(f)     the learned prior at f = 0, 1, 2 training bundles (mean of the three model seeds)
  b(f)       validation base rate: log(share of validation targets with frequency f / share of catalogue items)
  test       clean-test NDCG@20 of TRIAD (mean of model seeds per query) and of the best simple baseline, with a paired
             test (bootstrap + sign flips), Holm over the 6 dataset-split pairs.
Writes results/small_data/split_check.json; runs in results/small_data/split_check/<dataset>_split<k>/.

cd ../bundlefuse && PYTHONPATH=.:../dials .venv/bin/python -m experiments.small_data.split_check
"""
import triad  # pins BLAS threads before numpy loads
import json, math
from pathlib import Path
import numpy as np
import torch
from triad import data, train as T
from triad.baselines import run as baselines
from triad.evaluate import evaluate
from triad.statistics import paired_comparison, holm_adjust

DIALS = Path(__file__).resolve().parents[3]/'dials'
CACHE = DIALS/'cache';OUT = Path('results/small_data/split_check')
NAMES = ('electronic', 'food', 'clothing');SPLITS = (1, 2);SEEDS = (0, 1, 2)


def base_rate(ds, f):
    targets = np.concatenate([np.asarray(t) for _, t in ds.val_pairs])
    out = {}
    for k in (0, 1, 2):
        share_t = np.mean(f[targets] == k);share_c = np.mean(f == k)
        out[k] = float(math.log(share_t/share_c)) if share_t > 0 and share_c > 0 else None
    return out


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows, tests = {}, []
    for name in NAMES:
        for split in SPLITS:
            ds = data.load_resplit(name, seed=split, cache=CACHE);key = f'{name}_split{split}';root = OUT/key
            f = np.bincount(np.concatenate([np.unique(b) for b in ds.train]), minlength=ds.n_items)
            rho, per = [], []
            for seed in SEEDS:
                run = root/f'seed{seed}'
                T.fit(ds, seed, run, CACHE, config=T.config_for(name))
                model = T.load(ds, run, CACHE)
                if not (run/'test.json').exists():T.write_json(run/'test.json', evaluate(model, ds.test_pairs))
                with torch.no_grad():rho.append(model.prior(torch.log1p(torch.tensor([[0.], [1.], [2.]]))).squeeze(1).tolist())
                per.append([q['ndcg20'] for q in json.loads((run/'test.json').read_text())['queries']])
            path = root/'baselines.json'
            if not path.exists():T.write_json(path, baselines(ds, log=lambda *_: None))
            report = json.loads(path.read_text());method = report['best_simple']['method']
            m = report['methods'][method.replace('+prior', '')]
            simple = np.array((m['prior'] if method.endswith('+prior') else m)['queries'])[:, 1]
            clean = np.array(ds.clean_test);tri = np.mean(per, 0)[clean];simple = simple
            t = paired_comparison(tri, simple, [str(i) for i in range(len(tri))], bootstrap=2000, permutations=10000, seed=20261121)
            tests.append(dict(t, key=key))
            rows[key] = dict(rho=dict(zip(('f0', 'f1', 'f2'), np.mean(rho, 0).round(3).tolist())),
                             rho_by_seed=[[round(v, 3) for v in r] for r in rho], base_rate=base_rate(ds, f),
                             triad=float(tri.mean()), best_simple=float(simple.mean()), best_simple_method=method,
                             epochs=[json.loads((root/f'seed{s}'/'selected.json').read_text())['best_epoch'] for s in SEEDS])
            print(key, rows[key], flush=True)
    holm_adjust(tests, len(tests))
    T.write_json(OUT.parent/'split_check.json', dict(note=__doc__, rows=rows, tests=tests))
    for t in tests:
        r = rows[t['key']]
        print(f"{t['key']:18s} rho f0/f1/f2 {r['rho']['f0']:+.2f}/{r['rho']['f1']:+.2f}/{r['rho']['f2']:+.2f}  "
              f"b f0/f1/f2 {r['base_rate'][0]:+.2f}/{r['base_rate'][1]:+.2f}/{r['base_rate'][2]:+.2f}  "
              f"TRIAD {r['triad']:.4f} simple {r['best_simple']:.4f} ({r['best_simple_method']}) "
              f"delta {t['delta']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] p_holm {t['p_holm']:.4f}")


if __name__ == '__main__':
    main()
