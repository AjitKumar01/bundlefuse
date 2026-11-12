# Independent neural pilot

Status: complete

All methods use identical full-catalogue test queries. Neural checkpoints are selected on validation NDCG@20.

| Model | Selected epoch | Recall@20 | NDCG@20 |
|---|---:|---:|---:|
| bundlefuse_original_auxiliary | — | 0.486802 | 0.343218 |
| bundlefuse_matched | — | 0.460453 | 0.344215 |
| popularity | — | 0.059875 | 0.025772 |
| cooccurrence | — | 0.314644 | 0.225382 |
| itemknn | — | 0.323459 | 0.237884 |
| p3alpha | — | 0.327408 | 0.215574 |
| rp3beta | — | 0.317293 | 0.223772 |
| contentknn | — | 0.195328 | 0.092470 |
| dot_seed0 | 26 | 0.305154 | 0.188054 |
| relations_seed0 | 36 | 0.370761 | 0.262569 |
| contextual_seed0 | 32 | 0.387187 | 0.273728 |
| dot_seed1 | 40 | 0.315655 | 0.200739 |
| relations_seed1 | 52 | 0.363150 | 0.258375 |
| contextual_seed1 | 16 | 0.338295 | 0.232964 |
| dot_seed2 | 32 | 0.329480 | 0.212947 |
| relations_seed2 | 22 | 0.390029 | 0.278828 |
| contextual_seed2 | 28 | 0.362669 | 0.262328 |

The original BundleFuse row has auxiliary/user histories; the matched row does not. Neural inputs are item IDs, fixed catalogue text features, category labels and training frequency. No BundleFuse scores, engineered interaction matrices or distillation are used. These previously inspected splits support an exploratory pilot, not a fresh confirmatory claim.

Across completed neural seeds (sample standard deviation, not a confidence interval):

- contextual: 3 seeds, mean R20 0.362717, mean N20 0.256340.
- dot: 3 seeds, mean R20 0.316763, mean N20 0.200580.
- relations: 3 seeds, mean R20 0.374647, mean N20 0.266590.
