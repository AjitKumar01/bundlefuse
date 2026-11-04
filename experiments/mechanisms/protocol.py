"""Fixed follow-up plan; existing test results informed the research question."""
DATASETS = ['pog_dense','pog','spotify','spotify_sparse']
SEED = 20261105
CONTRASTS = [('shared','history'),('combined','history'),('combined','shared'),
             ('frequency_matched','combined'),('full','frequency_matched'),
             ('frequency_unit','constant_unit'),('frequency_matched','constant_matched'),
             ('frequency_unit','combined'),('constant_matched','combined'),
             ('constant_matched','constant_unit'),('frequency_matched','frequency_unit'),
             ('full','combined')]
PLAN = dict(datasets=DATASETS, objective='ndcg20', cutoffs=[20], validation='all supplied queries',
            test='all supplied queries, once recipes frozen', signal_profile='research',
            content_rank=None, rounds=4, chunk=16, seed=SEED,
            history=['simp','diff2','pop','cooc','ppmi','csvd'],
            shared=['cat','softcat','style','imgstyle','bridge','homo','audiostyle'],
            mixture_fractions=[0.,.05,.25,.5,.75,1.], kappa=[0.,.05,.25,.5,1.,5.],
            scales=['unit','validation matched'],
            stability=['canonical','reverse','permutation_20261105',
                       'cluster_bootstrap_20261106','cluster_bootstrap_20261107'],
            contrasts=CONTRASTS, planned_tests=len(DATASETS)*len(CONTRASTS)*2,
            bootstrap=2000, sign_flips=10000,
            interpretation='Follow-up on inspected benchmarks; inference conditional on fitted models/splits.',
            retention='Keep model artifacts, every frozen recipe, validation caches, query metrics and predictions.',
            scope='Four released POG/Spotify splits. No chronological-retail or all-eight claim.',
            budget='Same coordinate grid/round ceiling for weight fits; routing grids each have six candidates. '
                   'Full model adds the original optional-term and shrinkage searches. Counts are recorded.')
