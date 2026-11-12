# Independent neural pilot

Status: complete

All methods use identical full-catalogue test queries. Neural checkpoints are selected on validation NDCG@20.

| Model | Selected epoch | Recall@20 | NDCG@20 |
|---|---:|---:|---:|
| bundlefuse_original_auxiliary | — | 0.594854 | 0.462355 |
| bundlefuse_matched | — | 0.575071 | 0.449279 |
| popularity | — | 0.035033 | 0.016077 |
| cooccurrence | — | 0.344004 | 0.259456 |
| itemknn | — | 0.341171 | 0.262536 |
| p3alpha | — | 0.342304 | 0.238215 |
| rp3beta | — | 0.337819 | 0.237721 |
| contentknn | — | 0.340888 | 0.236306 |
| dot_seed0 | 26 | 0.371483 | 0.247411 |
| relations_seed0 | 16 | 0.399953 | 0.301327 |
| contextual_seed0 | 24 | 0.448536 | 0.330082 |
| dot_seed1 | 48 | 0.344948 | 0.240198 |
| relations_seed1 | 28 | 0.442918 | 0.331295 |
| contextual_seed1 | 50 | 0.444287 | 0.334353 |
| dot_seed2 | 32 | 0.364873 | 0.233100 |
| relations_seed2 | 28 | 0.455996 | 0.341183 |
| contextual_seed2 | 40 | 0.451747 | 0.333295 |

The original BundleFuse row has auxiliary/user histories; the matched row does not. Neural inputs are item IDs, fixed catalogue text features, category labels and training frequency. No BundleFuse scores, engineered interaction matrices or distillation are used. These previously inspected splits support an exploratory pilot, not a fresh confirmatory claim.

Across completed neural seeds (sample standard deviation, not a confidence interval):

- contextual: 3 seeds, mean R20 0.448190, mean N20 0.332577.
- dot: 3 seeds, mean R20 0.360434, mean N20 0.240236.
- relations: 3 seeds, mean R20 0.432956, mean N20 0.324602.
