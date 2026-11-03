# Reorganized pipeline full run

Frozen original defaults from 162dd8c. Full supplied queries for POG/Spotify; seed-0 random splits (800 validation / 2,000 test cap) for other datasets. BundleFuse only. Any repaired run will be recorded separately.

| Dataset | Status | R@20 | NDCG@20 | Earlier R@20 | Comparison |
|---|---|---:|---:|---:|---|
| pog_dense | complete | 0.391939 | 0.286365 | 0.395223 | supplied files |
| pog | complete | 0.053542 | 0.036320 | 0.054458 | supplied files |
| electronic | complete | 0.486802 | 0.343218 | separate protocol | seed 0 only |
| food | complete | 0.594854 | 0.462355 | separate protocol | seed 0 only |
| clothing | complete | 0.516936 | 0.404863 | separate protocol | seed 0 only |
| dunnhumby | complete | 0.207513 | 0.153609 | separate protocol | seed 0 only |
| spotify | complete | 0.154316 | 0.255460 | 0.155646 | supplied files |
| spotify_sparse | complete | 0.162959 | 0.228914 | 0.154288 | supplied files |

Logs: `original/<dataset>/run.log`. Jobs are sequential to bound memory. Metrics are selected only on validation. A historical mean from different splits is not an exact parity target.
