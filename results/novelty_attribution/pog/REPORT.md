# Novelty attribution results

Exploratory supplied-split results conditional on frozen prior source inventory. All controlled rows use the same fitting/tuning groups; original_reference uses its earlier all-validation recipe.

| Method | Recall@20 | NDCG@20 |
|---|---:|---:|
| pair:rrf | 0.031583 | 0.021181 |
| incidence:equal | 0.046458 | 0.030736 |
| amplified:weighted | 0.051042 | 0.033425 |
| amplified:bundlefuse_gate | 0.053542 | 0.035597 |
| incidence:rrf | 0.031583 | 0.021172 |
| pair:bundlefuse_gate | 0.054458 | 0.036082 |
| original_reference | 0.054458 | 0.036092 |
| pair:lambdamart | 0.022542 | 0.011735 |
| incidence:bundlefuse_gate | 0.054458 | 0.036043 |
| amplified:lambdamart | 0.022042 | 0.011747 |
| incidence:lambdamart | 0.022167 | 0.011622 |
| incidence:weighted | 0.051458 | 0.033780 |
| amplified:rrf | 0.031583 | 0.021162 |
| amplified:frequency_linear | 0.029625 | 0.016926 |
| pair:equal | 0.046458 | 0.030896 |
| incidence:frequency_linear | 0.029708 | 0.016946 |
| pair:frequency_linear | 0.029708 | 0.016948 |
| pair:weighted | 0.051458 | 0.033940 |
| amplified:equal | 0.046375 | 0.030545 |

Paired comparisons and Holm correction are recorded in summary.json. This stage is not an exact reproduction of HOSLIM or Kelen kNN; the amplified operator is explicitly defined.
