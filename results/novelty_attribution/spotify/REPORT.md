# Novelty attribution results

Exploratory supplied-split results conditional on frozen prior source inventory. All controlled rows use the same fitting/tuning groups; original_reference uses its earlier all-validation recipe.

| Method | Recall@20 | NDCG@20 |
|---|---:|---:|
| pair:rrf | 0.127396 | 0.211043 |
| incidence:equal | 0.153111 | 0.255907 |
| amplified:weighted | 0.160019 | 0.271723 |
| amplified:bundlefuse_gate | 0.163278 | 0.281146 |
| incidence:rrf | 0.129600 | 0.216064 |
| pair:bundlefuse_gate | 0.156566 | 0.264012 |
| original_reference | 0.155646 | 0.257926 |
| pair:lambdamart | 0.015975 | 0.022704 |
| incidence:bundlefuse_gate | 0.156566 | 0.264012 |
| amplified:lambdamart | 0.012781 | 0.018879 |
| incidence:lambdamart | 0.016447 | 0.028540 |
| incidence:weighted | 0.156566 | 0.264012 |
| amplified:rrf | 0.133048 | 0.221933 |
| amplified:frequency_linear | 0.107523 | 0.182548 |
| pair:equal | 0.150048 | 0.250092 |
| incidence:frequency_linear | 0.106897 | 0.183985 |
| pair:frequency_linear | 0.111376 | 0.190923 |
| pair:weighted | 0.156566 | 0.264012 |
| amplified:equal | 0.156333 | 0.262938 |

Paired comparisons and Holm correction are recorded in summary.json. This stage is not an exact reproduction of HOSLIM or Kelen kNN; the amplified operator is explicitly defined.
