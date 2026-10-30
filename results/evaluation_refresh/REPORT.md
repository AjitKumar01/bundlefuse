# Evaluation refresh

Verified refresh: **53/53 jobs complete**, 0 failed.
Run manifest and full logs: [manifest.json](manifest.json).
CPU, one thread per process; deterministic item-ID tie breaks. Fullrank: 3 split seeds; retail/e-commerce: 5 splits.

| Dataset | BundleFuse R@20 | SimpleHybrid | HardSwitchHybrid | Best other matched baseline |
|---|---:|---:|---:|---|
| pog | 0.0681 ± 0.0026 | 0.0378 ± 0.0010 | 0.0443 ± 0.0095 | 0.0243 (P3alpha) |
| pog_dense | 0.3752 ± 0.0065 | 0.3637 ± 0.0082 | 0.3663 ± 0.0082 | 0.2857 (LinearCombiner) |
| spotify | 0.1567 ± 0.0018 | 0.0988 ± 0.0021 | 0.0991 ± 0.0024 | 0.1305 (LinearCombiner) |
| spotify_sparse | 0.1680 ± 0.0044 | 0.1134 ± 0.0012 | 0.1135 ± 0.0014 | 0.1169 (LinearCombiner) |
| electronic | 0.5163 ± 0.0212 | 0.4745 ± 0.0470 | 0.4878 ± 0.0447 | 0.3343 (iALS) |
| food | 0.5939 ± 0.0281 | 0.5488 ± 0.0186 | 0.5559 ± 0.0324 | 0.3389 (iALS) |
| clothing | 0.5881 ± 0.0326 | 0.5412 ± 0.0315 | 0.5518 ± 0.0441 | 0.2657 (BundleKNN) |
| dunnhumby | 0.1504 ± 0.0044 | 0.1422 ± 0.0034 | 0.1422 ± 0.0034 | 0.1405 (LinearCombiner) |

These are matched split/candidate comparisons. The simple hybrids also receive the available raw
content, taxonomy, and auxiliary interactions. Heavy published models are literature context,
not paired results from this refresh. Historical sensitivity/ablation tables have not been regenerated
unless explicitly listed in this run. Missing DDBC/triple2vec reproduction paths remain open.

## Final checks

See [VALIDATION.md](VALIDATION.md) for the verification record and
[SELECTION.md](SELECTION.md) for the corrected selection-objective study.
