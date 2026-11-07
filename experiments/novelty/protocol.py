"""Exploratory follow-up plan fixed before new test evaluation."""
SEED = 20261119
DATASETS = ['pog_dense', 'spotify', 'pog']
BANKS = ['pair', 'incidence', 'amplified']
FUSIONS = ['equal', 'weighted', 'rrf', 'frequency_linear', 'lambdamart', 'bundlefuse_gate']
PLAN = dict(datasets=DATASETS, banks=BANKS, fusions=FUSIONS, seed=SEED,
    cutoff=20, chunk=8, fit_fraction=.7, random_negatives=128, hard_per_source=16,
    linear_regularization=[.001, .01, .1], ranker_trees=[50, 150], ranker_depth=[3, 5],
    rrf_offsets=[10., 60., 100.], kappa=[0., .05, .5, 1., 5., 20.],
    amplification_powers=[1., 2., 3.], amplification_neighbors=[100, 500, None],
    bootstrap=2000, permutations=10000,
    source_scope='Frozen canonical full-model source columns; replace only the historical-match column. '
      'Agreement is an ordinary explicit column when selected. Other raw columns are identical across banks.',
    supervision='Validation complete-bundle groups split into 70% fitting / 30% tuning. '
      'All supervised fits and amplified-neighbor choice use fitting groups only. '
      'Final hyperparameter choices use full-catalogue tuning NDCG. No refit after tuning.',
    artifact_policy='Read previous artifacts/caches only; keep new arrays, recipes, models and query results. '
      'Source/protocol/input mismatches reject resume; never silently overwrite completed work.',
    interpretation='Exploratory supplied-split follow-up. Canonical original scorer is a separately labelled '
      'reference selected on all validation queries. Raw-factor tests condition on a previously selected '
      'source inventory. Not a complete HOSLIM reproduction or an untouched-split confirmation.')
CONTRASTS = [
    *[(f'{b}:bundlefuse_gate', f'{b}:{f}') for b in BANKS
      for f in ['weighted', 'rrf', 'frequency_linear', 'lambdamart']],
    *[(f'incidence:{f}', f'pair:{f}') for f in FUSIONS],
    *[(f'incidence:{f}', f'amplified:{f}') for f in FUSIONS],
]
PLANNED_TESTS = len(DATASETS) * len(CONTRASTS) * 2
