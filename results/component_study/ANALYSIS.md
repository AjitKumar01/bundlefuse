# Component-study conclusions

All four dataset jobs and all eight methods completed. The 16,436 supplied test queries produce 131,488 method/query records. All full-model per-query metrics exactly reproduce the preceding verified supplied-split run. The full audit is in [VALIDATION.md](VALIDATION.md); [REPORT.md](REPORT.md) contains the live-generated tables and paired control intervals.

## Simple controls do not explain the observed gains

BundleFuse exceeds both SimpleHybrid and HardSwitchHybrid on both Recall@20 and NDCG@20 in every dataset. All 16 control comparisons reject the paired null after Holm adjustment over the complete prespecified family of 56 contrasts (adjusted p ≈ 0.0056). These results are conditional on the fitted models and supplied splits.

| Dataset | BundleFuse R@20 | HardSwitch R@20 | Gain (pp) | Pointwise cluster-bootstrap 95% CI (pp) |
|---|---:|---:|---:|---|
| pog | 0.054458 | 0.041042 | +1.342 | [+0.950, +1.704] |
| pog_dense | 0.395223 | 0.364741 | +3.048 | [+2.474, +3.592] |
| spotify | 0.155646 | 0.099961 | +5.569 | [+5.212, +5.936] |
| spotify_sparse | 0.154288 | 0.115458 | +3.883 | [+3.214, +4.547] |

## Component benefits depend on the dataset

Ablations are retuned on validation. Their differences measure the result of removing a component and retuning the remaining settings, rather than a fixed-weight intervention.

- **Higher-order terms:** removing both terms changes POG by exactly zero. On Spotify, they improve R@20 by 0.385 pp; on Spotify_sparse, by 0.288 pp. Both music improvements survive Holm correction. Dense POG improves by 0.233 pp in R@20, but that comparison does not pass the corrected 0.05 threshold (p = 0.0576); its NDCG@20 gain of 0.853 pp does. Incidence and Rao each matter on Spotify; Rao supplies the active higher-order effect on Spotify_sparse.
- **Shrinkage:** R@20 improves by 0.300 pp on POG, 0.841 pp on Spotify, and 0.265 pp on Spotify_sparse, with corrected significance. Dense POG has a small positive point estimate (+0.132 pp), with no corrected significant gain in either metric.
- **Bridging:** Spotify improves in both metrics, including +0.094 pp in R@20 (corrected p = 0.0060). Dense POG improves significantly in NDCG@20, while its R@20 contrast is not significant after correction. The retuned NoBridge model improves POG NDCG@20 by 0.257 pp (corrected p = 0.0209); its +0.125 pp Recall@20 difference is not significant. On Spotify_sparse, the selected full model already prunes bridging, so its removal changes nothing.

Removing all higher-order terms retains approximately 92–100% of the full model’s R@20 advantage over HardSwitchHybrid. Most of that observed advantage therefore persists in the remaining pairwise/content/routing system; these experiments do not identify a single remaining component as the cause. There is evidence for useful higher-order refinements on music, rather than a universal explanation for the large baseline gaps.

## Statistical interpretation

Identical complete held-out bundles are treated as clusters. Each contrast uses 2,000 cluster-bootstrap samples for a pointwise 95% interval and 10,000 paired cluster sign flips for a two-sided p-value. Holm adjustment includes all seven contrasts, both metrics, and all four datasets. Repeated adjusted p-values near 0.0056 reflect the Monte Carlo resolution: the smallest raw p-value is 1/10,001 and the family contains 56 tests. Pointwise intervals are not simultaneous intervals, so a positive interval can coexist with a nonsignificant Holm-adjusted test.

These tests condition on fitted configurations and the fixed supplied benchmark. They do not estimate training/split uncertainty, establish robustness to different masking rates or temporal retail splits, or establish paired significance against published neural summary scores.

## What this resolves and what remains

The tested simple hybrid explanations and uncertainty about their observed gaps are addressed on these four supplied splits. The component story is now empirically specific: shrinkage helps three datasets reliably; higher-order effects are clearest on music; bridging is conditional and can reduce POG ranking quality.

A strong newly trained metadata ranker, alternative-mask and temporal robustness studies, and reconciliation of the historical manuscript remain open. This study does not claim superiority over every possible learned hybrid or resolve the CLHE preprint/release split-ratio discrepancy documented in the preceding supplied-split report. The default configuration has not been altered using these test outcomes.

## Verification

42 regression tests passed before execution. The executed scorer/runner source fingerprint matches the current implementation. Split and feature hashes, original query IDs and counts, all ablation constraints, metric bounds and means, canonical duplicate-bundle groups, and all 56 resampled contrasts and Holm adjustments were checked. Temporary validation caches were removed. The separate legacy fingerprint check confirmed all 52 recorded evaluator input files remain unchanged.

```bash
python scripts/verify_components.py
```
