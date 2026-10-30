# Provided-split comparison

BundleFuse only; supplied training bundles and every supplied validation/test query.
No bundle deduplication, resplitting, remasking, or query sampling.

| Dataset | Status | BundleFuse R@20 | CLHE reference | RaMen reference | Reference source |
|---|---|---:|---:|---:|---|
| pog | complete | 0.054458 | 0.0284 | 0.0375 | RaMen Table 2 |
| pog_dense | complete | 0.395223 | 0.3811 | — | CLHE Table 2 |
| spotify | complete | 0.155646 | 0.1081 | 0.1091 | RaMen Table 2 |
| spotify_sparse | complete | 0.154288 | 0.098 | — | CLHE Table 2 |

Reference scores are taken directly from [RaMen Table 2](https://arxiv.org/html/2507.14361v1)
and [CLHE Table 2](https://arxiv.org/html/2310.18770); neural models are not run here.
Both dense/sparse variants already appear in the CLHE paper; earlier local documentation incorrectly
called them newly constructed variants and substituted local reproduction scores.
The supplied files use 70/10/20, consistent with RaMen. The CLHE preprint describes 80/10/10;
its dense/sparse published scores are therefore reference context until that release/paper discrepancy is resolved.
The manifest records hashes of the supplied split files and equality with saved neural-run copies.
Catalogue candidates, seed exclusion, macro Recall/NDCG, and selected settings are recorded per dataset.
Neural scores retain their original tie handling; BundleFuse resolves exact ties by ascending item ID.
This controls the split/mask mismatch. BundleFuse retains its own metadata/features; this is not a
same-feature architecture ablation. Published-score comparisons do not constitute a local neural reproduction.
Electronic/Food are excluded because their public BundleRec data do not establish the published neural split.
