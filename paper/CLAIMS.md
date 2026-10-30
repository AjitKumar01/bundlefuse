# Claims and evidence

This ledger describes the verified implementation. Historical manuscript prose and figures must
be reconciled with it before publication.

| Claim | Current evidence and permitted scope |
|---|---|
| BundleFuse exceeds the listed neural reference scores | Supplied POG/Spotify split runs support reported-score comparisons. The CLHE preprint/release ratio discrepancy limits strict equivalence for dense/sparse variants. No paired significance is inferred from published aggregate scores. |
| Gains exceed a simple combination of the available information | Both raw-information hybrid controls lose in R@20 and NDCG@20 on all four supplied splits under the prespecified paired analysis and Holm correction. |
| Higher-order operators explain most of the advantage | Unsupported. Removing both operators retains about 92–100% of the R@20 advantage over HardSwitchHybrid. Higher-order refinements help music; dense POG has NDCG evidence but its R@20 contrast misses the corrected threshold. |
| Shrinkage is universally necessary | Unsupported. Retuned ablations support gains on three datasets; dense POG does not show a corrected significant gain. |
| Bridging always improves ranking | Unsupported. It helps Spotify, is pruned on Spotify_sparse, and its removal improves POG NDCG. |
| No gradient-trained encoder in the default BundleFuse scorer | Supported. Fusion/routing weights are nevertheless selected on validation; SVD, graph and taxonomy statistics are estimated. External pretrained embeddings have an upstream cost. |
| No fitted parameters / zero total cost | Unsupported wording. Describe the specific closed-form scorer and report preprocessing, selection, inference and upstream exclusions separately. |
| Competitive with the tested trained feature-matched ranker | Supported in original-input conditions on all five datasets, for all three trained seeds in both R@20 and NDCG@20 after the planned Holm correction. This covers one architecture/grid, not every learned model. Two Spotify confirmation seeds hit the epoch budget. |
| Robust to sparse inputs and future retail baskets | Conditional. The original chronological retail condition favors BundleFuse, but the learned model wins sparse-retail R@20 and SimpleHybrid wins one-seed POG_dense. Temporal graphs are trained only on earlier interactions; catalogue metadata is fixed and transductive. Product introduction dates are unavailable. |
| Faster / less memory than trained alternatives | A blanket advantage is unsupported. The trained ranker with cached catalogue vectors is faster in measured warm-cache scoring and scoring-plus-metrics. Whole-worker RSS cannot establish per-method memory superiority. External embedding extraction and earlier supplied-fold baseline search remain excluded from new totals. |
| Selective signal computation speeds up BundleFuse without changing results | Supported on all 9,938 original POG_dense/Spotify test queries: bit-identical fused scores and historical metrics. Warm-batch scoring improves 3.92× / 1.46× on the measured CPU. This does not establish every dataset/backend's speedup. |
| Engineered features improve a compact trained ranker | Supported for the tested 58,314-parameter architecture and feature budget on POG_dense/Spotify. Seed-0 R@20 and NDCG@20 gains over the raw arm survive Holm correction over 12 planned contrasts; three-seed means agree in direction. The joint pack does not identify individual feature effects or establish novelty by itself. |
| The compact student preserves BundleFuse accuracy | Unsupported. Both validation-only 97.5% retention gates fail; engineered mean test Recall retention is 89.68% / 67.68%. Frozen caches remain separate from trainable parameters. This pilot does not prove a universal low-rank impossibility result. |
| Distillation or sparse adaptation solves the compact student's gap | Untested. Neither stage was started after the failed feasibility gate. |
| Verified DDBC / triple2vec superiority | Historical claims remain outside the reproduced experiment suite. |

Evidence: [supplied splits](../results/provided_splits/ANALYSIS.md),
[controls and ablations](../results/component_study/ANALYSIS.md), and
[verified learned/generalization findings](../results/generalization_study/ANALYSIS.md), and
[verified compact-model pilot](../results/compact_pilot/ANALYSIS.md).
