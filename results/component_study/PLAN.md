# Prespecified component study

Implementation commit: f25ceaa.
Source SHA-256: e23570b49172af9ffb1fa5a1dc9f9a581afbb0ca16f35eb975d4210ceda915c6

Datasets: POG, POG_dense, Spotify, Spotify_sparse, using every supplied training/validation/test row.

Methods: BundleFuse, NoIncidence, NoRao, NoHigherOrder, NoShrinkage, NoBridge, SimpleHybrid, HardSwitchHybrid.

Every ablation is retuned on validation with its named component disabled. All methods are frozen before test scoring. Existing raw-information SimpleHybrid and HardSwitchHybrid controls retain their selection grids.

Primary descriptive metric: Recall@20; secondary: NDCG@20. Both enter the prespecified family of 56 paired comparisons against full BundleFuse. Pointwise 95% confidence intervals use 2,000 cluster-bootstrap samples; two-sided paired tests use 10,000 cluster sign flips, adjusted with Holm across the complete family. Duplicate complete held-out bundle item sets form clusters; individual query multiplicities remain in the macro mean. Resampling seed: 0.

No test-based selection of variants or statistical contrasts. Uncertainty is conditional on fitted models and supplied folds. A trained metadata ranker, alternate-mask robustness, and temporal retail experiments are outside this first prioritized phase.

Commands:
```bash
python -m bundlefuse.provided_experiments --study components --output results/component_study
python scripts/verify_components.py
```
