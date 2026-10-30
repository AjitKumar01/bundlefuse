# Completed supplied-split comparison

All four BundleFuse runs completed and passed [verification](VALIDATION.md). Each used every supplied training bundle and validation/test query. Only BundleFuse was executed.

## Recall@20

| Dataset | BundleFuse | CLHE | RaMen |
|---|---:|---:|---:|
| pog | 0.054458 | 0.0284 | 0.0375 |
| spotify | 0.155646 | 0.1081 | 0.1091 |
| pog_dense | 0.395223 | 0.3811 | — |
| spotify_sparse | 0.154288 | 0.0980 | — |

## NDCG@20

| Dataset | BundleFuse | CLHE | RaMen |
|---|---:|---:|---:|
| pog | 0.036092 | 0.0193 | 0.0226 |
| spotify | 0.257926 | 0.1806 | 0.1882 |
| pog_dense | 0.287876 | 0.2773 | — |
| spotify_sparse | 0.221912 | 0.1212 | — |

POG Recall@20 is 45.2% higher than the published RaMen score; Spotify is 42.7% higher. These are relative differences between point estimates, not significance tests.

## Interpretation

The earlier BundleFuse resplitting/remasking mismatch is resolved for the released POG/Spotify benchmark files. Split/count files match all 24 saved neural-copy file hashes, and selected configurations use only the supplied validation queries. All 16,436 test queries are retained.

BundleFuse scores exceed the listed published CLHE/RaMen point estimates for both R@20 and N@20 on every available comparison. POG and Spotify use the 70/10/20 protocol described in [RaMen, Table 2](https://arxiv.org/html/2507.14361v1).

The dense/sparse CLHE references come from [CLHE, Table 2](https://arxiv.org/html/2310.18770). That preprint describes 80/10/10, whereas the released files use 70/10/20. Their higher BundleFuse scores are therefore published-score comparisons; exact equivalence to the preprint experiment remains unestablished. Matching the released files is verified, rather than inferring file identity from a matching dataset name or ratio.

BundleFuse retains its metadata/features and deterministic item-ID tie handling. These runs support an end-to-end comparison on released benchmark inputs; they do not isolate architecture effects under identical feature encoders or establish statistical significance. Electronic/Food remain outside this experiment because their public BundleRec data do not establish the published neural benchmark split.

## Reproduction

```bash
python -m bundlefuse.provided_experiments --output results/provided_splits
python scripts/verify_provided.py
```

The [manifest](manifest.json) records commands, versions, source fingerprints, split/feature hashes, selected settings, and job durations. Per-dataset directories contain full logs, selection settings, protocol records, and per-query metrics. [Tests](tests.log): 32 passed. Temporary score caches were removed.
