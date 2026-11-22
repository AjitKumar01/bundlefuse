"""Training and validation loss per epoch of the full-rank TRIAD (experiments.small_data.residual, variant v2), full
training data, one panel per dataset: mean of seeds 0-2 with a min-max band; dashed line: the epoch chosen on validation
NDCG@20 (median over seeds). Training loss: TRIAD's loss on the training bundles of the epoch (leave-one-out, random
cuts); validation loss: the same loss on the validation queries, statistics from all training bundles.

Spotify runs keep only train_loss for epochs >= 1, as the local runs.

cd ../bundlefuse && .venv/bin/python -m experiments.small_data.plot_losses [out.png]
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('results/small_data/v2');LOCAL = Path('results/small_data/local_v2');KAGGLE = Path('results/small_data/kaggle_v2')
NAMES = dict(electronic='Electronic', food='Food', clothing='Clothing', pog_dense='POG_DENSE', pog='POG', spotify_sparse='Spotify_sparse',
             spotify='Spotify')
TRAIN, VAL = '#2A7AB0', '#D9541E'


def curves(runs, key, start):
    longest = max(len(r['history']) for r in runs);out = np.full((len(runs), longest), np.nan)
    for i, r in enumerate(runs):
        for h in r['history']:
            if key in h:out[i, h['epoch']] = h[key]
    x = np.arange(longest);keep = ~np.all(np.isnan(out), 0);keep[:start] = False
    return x[keep], out[:, keep]


def runs_of(name):
    """Seeds 0-2 of one dataset: results/small_data/v2/<name>_full.json (local runs of experiments.small_data.residual)
    or results/small_data/local_v2/<name>/seed<s>.json (the Spotify runs, kaggle_triad_v2/run.py on the CPU)."""
    if (R/f'{name}_full.json').exists():return json.loads((R/f'{name}_full.json').read_text())['runs']
    for d in (LOCAL, KAGGLE):
        files = sorted((d/name).glob('seed*.json'))
        if len(files) == 3:return [json.loads(f.read_text()) for f in files]


def main(path):
    names = [n for n in NAMES if runs_of(n)]
    cols = 4;rows = -(-len(names)//cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3*cols, 2.9*rows), squeeze=False)
    for ax in axes.flat[len(names):]:ax.axis('off')
    for ax, n in zip(axes.flat, names):
        runs = runs_of(n)
        for key, colour, label, start in (('train_loss', TRAIN, 'training loss', 1), ('val_loss', VAL, 'validation loss', 0)):
            x, y = curves(runs, key, start)
            with np.errstate(all='ignore'):m, lo, hi = np.nanmean(y, 0), np.nanmin(y, 0), np.nanmax(y, 0)
            ax.fill_between(x, lo, hi, color=colour, alpha=.18, lw=0);ax.plot(x, m, color=colour, lw=2, label=label)
        best = int(np.median([r['best_epoch'] for r in runs]))
        ax.axvline(best, color='#555', ls='--', lw=1);ax.text(best, ax.get_ylim()[1], f' chosen: {best}', va='top', fontsize=8, color='#555')
        ax.set_title(NAMES[n], fontsize=10);ax.set_xlabel('epoch', fontsize=9);ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False);ax.grid(axis='y', color='#eee')
    axes[0][0].set_ylabel('loss', fontsize=9)
    axes[0][0].legend(fontsize=8, frameon=False, loc='upper right')
    fig.tight_layout();fig.savefig(path, dpi=180);print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '../dials/figures/design/v2_loss_curves.png')
