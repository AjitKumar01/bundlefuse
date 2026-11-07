"""Freeze this follow-up independently of the completed novelty study."""
SEED = 20261119
DATASETS = ['electronic', 'food', 'clothing', 'dunnhumby', 'spotify_sparse']
CONTRASTS = [
    ('amplified:gate', 'amplified:weighted'),
    ('amplified:gate', 'amplified:constant'),
    ('reference:gate', 'reference:weighted'),
    ('reference:gate', 'reference:constant'),
    ('amplified:weighted', 'reference:weighted'),
    ('amplified:gate', 'reference:gate'),
]
PLAN = dict(datasets=DATASETS, seed=SEED, chunk=8, fit_fraction=.7,
    powers=[1., 2., 3.], neighbors=[100, 500, None], rounds=4,
    kappa=[0., .05, .5, 1., 5., 20.], mixture=[0., .05, .25, .5, .75, 1.],
    bootstrap=2000, permutations=10000, planned_tests=len(DATASETS)*len(CONTRASTS)*2,
    banks=['reference', 'amplified'], fusions=['weighted', 'gate', 'constant'],
    sources='Retain each input checkpoint\'s selected source inventory. Replace only simp\'s '
      'effective standardized pair/incidence column. Other columns, including any diffusion '
      'or concordance column, remain unchanged. Historical coefficient may be selected as zero.',
    selection='Group validation by complete bundle; fit on 70%, tune on 30%. Amplification '
      'and coordinate weights use fit groups only. Select gate and constant mixture on tuning '
      'NDCG@20; no refit. Freeze all recipes before test score generation.',
    scope='Exploratory continuation after inspecting POG_DENSE, Spotify and POG. '
      'Spotify_sparse uses retained research mechanism artifacts and all supplied queries. '
      'Other datasets reuse the verified reorganized extended checkpoints and their original '
      'seed-0 random splits/query limits (800 validation, 2000 test ceilings). '
      'Dunnhumby is not the chronological robustness protocol. Original frozen references '
      'were selected on all original validation queries; comparisons with refits are descriptive.',
    retention='Separate output/cache directories; retain all input checkpoints, intermediate '
      'caches, selected recipes, per-query metrics/predictions and progress on interruption.')
