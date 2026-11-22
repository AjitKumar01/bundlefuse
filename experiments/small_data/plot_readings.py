"""Which learned behaviour each dataset relies on: validation NDCG@20 lost when one behaviour of the trained full-rank
TRIAD is removed (experiments.small_data.readings, knockouts), as a share of the trained model's validation NDCG@20,
mean of seeds 0-2. Darker = larger loss; numbers are the loss in percent (negative = removing it helped).

cd ../bundlefuse && .venv/bin/python -m experiments.small_data.plot_readings [out.png]
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('results/small_data/readings.json')
NAMES = dict(electronic='Electronic', food='Food', clothing='Clothing', pog_dense='POG_DENSE', pog='POG',
             spotify_sparse='Spotify_sparse', spotify='Spotify')
ROWS = [('content_maps_at_start', 'learned content maps'), ('symmetric_content', 'asymmetry of the content map'),
        ('no_partner_history', 'partner histories'), ('no_feature_refinement', 'feature refinement'),
        ('no_label_rule', 'label rule'), ('exponents_at_start', 'learned count exponents'),
        ('shared_exponent', 'separate seed / candidate exponents'), ('no_prior', 'popularity prior'),
        ('fixed_gate', 'per-query gate')]


def main(path):
    d = json.loads(R.read_text())['datasets'];names = [n for n in NAMES if d.get(n)]
    loss = np.array([[np.mean([100*(s['knockouts']['trained'] - s['knockouts'][k])/s['knockouts']['trained'] for s in d[n]])
                      if d[n][0]['knockouts'][k] is not None else np.nan for n in names] for k, _ in ROWS])
    fig, ax = plt.subplots(figsize=(1.25*len(names) + 2.6, 4.2))
    ax.imshow(np.clip(loss, 0, 60), cmap='Blues', vmin=0, vmax=60, aspect='auto')
    for i in range(loss.shape[0]):
        for j in range(loss.shape[1]):
            v = loss[i, j];ax.text(j, i, f'{round(v):d}', ha='center', va='center', fontsize=8.5, color='white' if v > 35 else '#222')
    ax.set_xticks(range(len(names)), [NAMES[n] for n in names], fontsize=8.5)
    ax.set_yticks(range(len(ROWS)), [lab for _, lab in ROWS], fontsize=8.5);ax.tick_params(length=0)
    for s in ax.spines.values():s.set_visible(False)
    ax.set_title('validation NDCG@20 lost when the behaviour is removed (% of trained)', fontsize=9.5)
    fig.tight_layout();fig.savefig(path, dpi=180);print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '../dials/figures/design/knockouts.png')
