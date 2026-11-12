# Independent neural pilot

Status: training

All methods use identical full-catalogue test queries. Neural checkpoints are selected on validation NDCG@20.

| Model | Selected epoch | Recall@20 | NDCG@20 |
|---|---:|---:|---:|
| bundlefuse_original_auxiliary | — | 0.516936 | 0.404863 |
| bundlefuse_matched | — | 0.474663 | 0.388848 |
| popularity | — | 0.008086 | 0.002770 |
| cooccurrence | — | 0.238095 | 0.188419 |
| itemknn | — | 0.238095 | 0.194355 |
| p3alpha | — | 0.238634 | 0.174447 |
| rp3beta | — | 0.238634 | 0.189870 |
| contentknn | — | 0.282210 | 0.164677 |

The original BundleFuse row has auxiliary/user histories; the matched row does not. Neural inputs are item IDs, fixed catalogue text features, category labels and training frequency. No BundleFuse scores, engineered interaction matrices or distillation are used. These previously inspected splits support an exploratory pilot, not a fresh confirmatory claim.
