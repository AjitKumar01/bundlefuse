# Amplified-neighbor and fallback follow-up

Exploratory results using the recorded input checkpoint and split. Original reference was selected on all earlier validation queries. Controlled banks fit/tune on identical 70/30 complete-bundle groups. Constant-mixture controls use the same fallback sources.

| Method | Recall@20 | NDCG@20 |
|---|---:|---:|
| amplified:gate | 0.472495 | 0.331170 |
| reference:constant | 0.439644 | 0.333767 |
| amplified:weighted | 0.454961 | 0.328351 |
| reference:weighted | 0.457274 | 0.332229 |
| amplified:constant | 0.438198 | 0.329159 |
| original_reference | 0.486802 | 0.343218 |
| reference:gate | 0.485501 | 0.343676 |

Paired statistics reserve 60 tests across the five newly added datasets. These are fused score banks, not isolated raw operators; the historical column may receive zero selected weight. See inputs.json for checkpoint/data profile and query scope.
