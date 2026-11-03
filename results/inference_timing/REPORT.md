# BundleFuse inference timing

Hardware: Apple M5 Pro, 24 GiB RAM. One CPU thread for PyTorch and native BLAS/OpenMP pools.
Current simplified runtime; retained checkpoints from the original reorganized eight-dataset
run (extended signal/data profile, with the fitted content projection recorded in each JSON).
These timings do not measure the pending corrected research-profile fits.

Online timing covers the public `predict(k=20)` call: input checks, selected-signal scoring,
fusion, seed exclusion and top-k ranking over the full catalogue. It excludes checkpoint/data
loading, warm-up and metric evaluation. Each dataset uses 256 uniformly sampled
distinct test queries (or all if fewer), seed 20261103, 3 shuffled repetitions,
and up to 16 warm-up queries. P95 pools all timed calls; no outliers are removed.
All repetitions return identical online predictions. Each dataset runs in a fresh process,
sequentially. Retained checkpoint SHA-256 values are verified before and after benchmarking.

Batch timing includes `score_batch` and top-k ranking for every query; it is amortized throughput
at batch size 32, not single-request latency. Loading is one unpickle measurement
with OS caching uncontrolled; first request includes any lazy graph reconstruction. Neither
loading nor first request is included in warm latency. Full raw timings/configs are in each
dataset JSON, and worker output is retained in `<dataset>.log`.

| Dataset | Items | Mean ms | Median ms | P95 ms | Batch ms/query | Load s | First request ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| electronic | 3,499 | 5.025 | 4.893 | 5.508 | 0.448 | 0.085 | 6.524 |
| food | 3,767 | 1.554 | 1.312 | 1.416 | 0.310 | 0.020 | 2.170 |
| clothing | 4,487 | 2.065 | 1.685 | 1.911 | 0.462 | 0.025 | 3.218 |
| dunnhumby | 21,744 | 16.914 | 8.075 | 37.390 | 2.285 | 0.023 | 43.670 |
| pog_dense | 31,217 | 11.852 | 5.246 | 34.385 | 2.246 | 0.027 | 6.385 |
| pog | 48,676 | 18.332 | 8.217 | 37.876 | 3.032 | 0.059 | 37.135 |
| spotify | 254,155 | 561.698 | 559.592 | 599.408 | 69.868 | 0.134 | 2619.990 |
| spotify_sparse | 213,325 | 259.083 | 264.746 | 288.311 | 41.221 | 0.107 | 264.976 |

A separate [three-query Spotify profile](spotify_profile.json), on queries with 70, 48 and 5
seeds, took 1.671 seconds in total. Sparse `tocsc` conversions accounted for 0.980 seconds
(58.6%); the concordance routine, which includes conversion, accounted for 1.076 seconds
(64.4%). The compressed checkpoint omits `W_csc`, and the concordance function reconstructs
a local CSC matrix on each call without retaining it. This is an optimization target; no
model/operator changes were made for this benchmark. Profiling was run after all unprofiled
measurements, with no competing dataset worker.
