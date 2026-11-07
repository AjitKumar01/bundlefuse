# Novelty attribution results

Exploratory supplied-split results conditional on frozen prior source inventory. All controlled rows use the same fitting/tuning groups; original_reference uses its earlier all-validation recipe.

| Method | Recall@20 | NDCG@20 |
|---|---:|---:|
| pair:rrf | 0.319412 | 0.198652 |
| incidence:equal | 0.389637 | 0.282466 |
| amplified:weighted | 0.388851 | 0.287204 |
| amplified:bundlefuse_gate | 0.390985 | 0.288159 |
| incidence:rrf | 0.323117 | 0.203364 |
| pair:bundlefuse_gate | 0.392107 | 0.281107 |
| original_reference | 0.395223 | 0.287876 |
| pair:lambdamart | 0.282980 | 0.241447 |
| incidence:bundlefuse_gate | 0.391911 | 0.285098 |
| amplified:lambdamart | 0.281380 | 0.244121 |
| incidence:lambdamart | 0.290053 | 0.244969 |
| incidence:weighted | 0.391378 | 0.284989 |
| amplified:rrf | 0.320226 | 0.201747 |
| amplified:frequency_linear | 0.350567 | 0.265478 |
| pair:equal | 0.386522 | 0.274642 |
| incidence:frequency_linear | 0.344785 | 0.258344 |
| pair:frequency_linear | 0.343999 | 0.255371 |
| pair:weighted | 0.388599 | 0.277964 |
| amplified:equal | 0.390002 | 0.285475 |

Paired comparisons and Holm correction are recorded in summary.json. This stage is not an exact reproduction of HOSLIM or Kelen kNN; the amplified operator is explicitly defined.
