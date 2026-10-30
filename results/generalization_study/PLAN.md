# Prespecified learned-baseline and generalization study

Implementation commit: c833049a26b1f15eb83858cdb377dec33c6b1ce6
Source SHA-256: d7b6b4d3e7ad8a58b934f445699f1448b2f95430f91d179928deaa1da7bd668e

Five sequential CPU jobs: pog, pog_dense, dunnhumby_chrono, spotify, spotify_sparse. Each evaluates BundleFuse, SimpleHybrid, HardSwitchHybrid and learned ranker seeds 0/1/2 on supplied, half_seeds, quarter_seeds, one_seed. This gives 120 method/case evaluations.

Provided-fold BundleFuse/control settings are frozen from the verified component study and must reproduce its per-query metrics. The chronological job tunes those methods on its own past validation fold. The learned baseline uses raw content, categories, collaborative item IDs and training-only retail household histories. Cold item-ID factors/biases are disabled. All test conditions use original-validation-selected settings. Removed seeds become additional hidden targets.

Learned seed-0 grid:
```json
[
  {
    "dim": 64,
    "lr": 0.001,
    "regularization": 1e-05,
    "negatives": 32,
    "batch_size": 128,
    "max_epochs": 100,
    "evaluate_every": 5,
    "patience": 4
  },
  {
    "dim": 64,
    "lr": 0.001,
    "regularization": 0.001,
    "negatives": 32,
    "batch_size": 128,
    "max_epochs": 100,
    "evaluate_every": 5,
    "patience": 4
  },
  {
    "dim": 64,
    "lr": 0.003,
    "regularization": 1e-05,
    "negatives": 32,
    "batch_size": 128,
    "max_epochs": 100,
    "evaluate_every": 5,
    "patience": 4
  },
  {
    "dim": 64,
    "lr": 0.003,
    "regularization": 0.001,
    "negatives": 32,
    "batch_size": 128,
    "max_epochs": 100,
    "evaluate_every": 5,
    "patience": 4
  }
]
```

The selected seed-0 hyperparameters are repeated with seeds 1/2, with validation-selected checkpoints. No best-test seed or ensemble. Optimizers: SparseAdam for sparse embedding gradients, AdamW for dense parameters, plus representation regularization. Training uses fresh half-bundle masks each epoch and sampled softmax with 32 negatives excluding known positives. Retail additionally samples two items from training-only household histories in 20% of examples. Learning curves, early stopping and best-at-final-check flags are retained.

Temporal retail: eligible baskets have 3–20 products. Whole-day 70/10/20 cutoffs precede seeded sampling capped at 21,000/3,000/6,000 baskets. No future-frequency filtering or deduplication. Preflight ranges: train days 1–515, validation 516–580, test 581–711. Fixed metadata catalogue: 92,353 items. Product introduction dates are unavailable; catalogue availability is transductive. Graphs and household histories use training interactions only.

Inference: 200 prespecified paired contrasts (5 datasets × 4 cases × 5 alternatives × 2 metrics), pointwise cluster-bootstrap intervals from 2,000 samples, two-sided paired cluster sign-flip tests from 10,000 samples, seed 0, Holm across the complete family. Clusters are complete bundles in POG/Spotify and households in retail. Learned seed 0 is primary; other seeds report variability.

Costs: separate preprocessing, learned training/validation search, catalogue encoding, warm-cache scoring, scoring plus metrics, and whole-worker peak RSS. Same 32 sampled queries, three timing repeats, CPU one thread. External feature extraction/pretraining excluded. Checkpoints are local ignored .pt files with recorded hashes; metrics/learning curves/logs are retained.

Commands:
```bash
python -m bundlefuse.generalization_experiments --output results/generalization_study
python scripts/verify_generalization.py
```
