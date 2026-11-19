"""Every baseline sees exactly TRIAD's item features: an audit of all code paths that feed the paper's numbers.

Checks (each prints PASS/FAIL; the run fails if any check fails):
  released   POG / Spotify feature files on disk are byte-identical to the release CLHE and RaMen download
             (xhLiu/BundleConstruction on Hugging Face; SHA-256 against the LFS object IDs). Needs network; --offline skips.
  simple     experiments.simple_baselines.run.features (used by the simple and label-aware baselines, their
             --extended suite and the timing runs) returns TRIAD's matrix (triad.data.load) on every dataset, with the
             same training split.
  blip       experiments.dials_blip.run.load (e-commerce content baselines of the main table: dials_blip and
             fair_prior) returns TRIAD's matrix.
  budget     experiments.bundle_budget(_retail).conditions.dataset (training-set-size baselines) returns TRIAD's matrix.
  neural     the CLHE / RaMen exports for e-commerce (<name>_{clean,published,pretrain}_blip) hold, in both feature
             slots, exactly the 768-d BLIP title vectors that TRIAD projects. On POG and Spotify CLHE and RaMen read the
             released files themselves (checked by 'released').
In addition, experiments.fair_prior.baselines asserts that every published simple-baseline number of the main table and
of the training-set-size study is reproduced to 1e-6 with TRIAD's features (lambda = 0).
Not covered: DDBC, rerun from its released checkpoints with its own released inputs, and Bundle-MLLM, whose numbers are published.
Writes results/feature_audit.json.

PYTHONPATH=.:../dials nice -n 10 .venv/bin/python -m experiments.feature_audit.run [--offline]
"""
import triad  # pins BLAS threads before numpy loads
import hashlib, json, sys, urllib.request
from pathlib import Path
import numpy as np, torch
from triad import data
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json

CACHE = Path(DATA_ROOT)/'dials'/'cache';SUPPLIED = ['pog_dense', 'pog', 'spotify_sparse', 'spotify'];ECOM = ['electronic', 'food', 'clothing']
results = []


def check(name, ok, detail=''):
    results.append(dict(check=name, ok=bool(ok), detail=detail));print('PASS' if ok else 'FAIL', name, detail, flush=True)


def same(a, b):return a.shape == b.shape and float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).max()) == 0.


def main(offline):
    triad_x = {n: data.load(n, CACHE).features for n in ECOM + SUPPLIED + ['dunnhumby']}
    if not offline:
        for n in SUPPLIED:
            tree = json.loads(urllib.request.urlopen(f'https://huggingface.co/api/datasets/xhLiu/BundleConstruction/tree/main/{n}').read())
            remote = {f['path'].split('/')[-1]: f.get('lfs', {}).get('oid') for f in tree}
            for f in ('content_feature.pt', 'description_feature.pt'):
                local = hashlib.sha256((data.DATA_ROOT/'bundlemllm'/'datasets'/n/f).read_bytes()).hexdigest()
                check(f'released {n}/{f}', local == remote.get(f), local[:16])
    from experiments.simple_baselines.run import load, features
    for n in ECOM + SUPPLIED + ['dunnhumby']:
        ds, _ = load(n);check(f'simple {n}', same(features(n, ds), triad_x[n]))
    from experiments.dials_blip import run as blip
    for n in ECOM:check(f'blip {n}', same(blip.load(n).features, triad_x[n]))
    from experiments.bundle_budget_retail import conditions as RC
    from experiments.bundle_budget import conditions as PC
    for n in ECOM + ['spotify_sparse', 'spotify']:check(f'budget {n}', same(RC.dataset(n).features, triad_x[n]))
    for n in ('pog_dense', 'pog'):check(f'budget {n}', same(PC.dataset(n).features, triad_x[n]))
    from experiments.dials_blip.export_clhe import titles
    for n in ECOM:
        t = titles(n);raw = np.load(CACHE/'blip'/f"{hashlib.sha256(chr(0).join(t).encode()).hexdigest()[:16]}.npy")
        for v in ('clean_blip', 'published_blip', 'pretrain_blip'):
            d = data.DATA_ROOT/'bundlemllm'/'datasets'/f'{n}_{v}'
            if not d.exists():check(f'neural {n}_{v}', False, 'export missing');continue
            for f in ('content_feature.pt', 'description_feature.pt'):
                check(f'neural {n}_{v}/{f}', same(torch.load(d/f, weights_only=True).numpy(), raw))
    write_json(Path('results/feature_audit.json'), dict(note=__doc__, checks=results))
    if not all(r['ok'] for r in results):sys.exit(1)


if __name__ == '__main__':
    main('--offline' in sys.argv)
