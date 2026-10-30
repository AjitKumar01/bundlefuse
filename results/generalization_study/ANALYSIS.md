# Verified learned-baseline and generalization findings

All five jobs completed with zero failures. The audit checked 120 method/case evaluations, 538,464 method/query records, and recomputed all 200 prespecified paired comparisons. The implementation passed 51 regression tests. Source/input hashes and frozen supplied-fold baseline results match the recorded references.

## Original input conditions

The table uses the original supplied masks for POG/Spotify and the original half-hidden chronological mask for retail. Learned means and population SDs describe three separately trained models, not an ensemble or a best-test-seed selection.

| Dataset | BundleFuse R@20 | SimpleHybrid | HardSwitch | Learned seed 0 | Learned mean ± SD |
|---|---:|---:|---:|---:|---:|
| pog | 0.054458 | 0.034083 | 0.041042 | 0.023167 | 0.023625 ± 0.000863 |
| pog_dense | 0.395223 | 0.363647 | 0.364741 | 0.358566 | 0.358360 ± 0.001269 |
| dunnhumby_chrono | 0.118867 | 0.113179 | 0.113179 | 0.096630 | 0.096970 ± 0.001112 |
| spotify | 0.155646 | 0.099754 | 0.099961 | 0.120616 | 0.118988 ± 0.001154 |
| spotify_sparse | 0.154288 | 0.115149 | 0.115458 | 0.109306 | 0.111407 ± 0.001769 |

BundleFuse exceeds all five alternatives on both R@20 and NDCG@20 in each original-input setting: all 50 base-condition contrasts survive Holm adjustment over the entire 200-test family (adjusted p = 0.019998). This includes all three learned training seeds, both simple controls, and the chronological retail baseline condition. The conclusion applies to this tested architecture/grid and these fixed folds; it is not a comparison with every trained model family.

## Sparse-input robustness has confirmed limits

All settings were frozen using original validation queries. Dropped seeds become hidden targets, so these conditions change both context size and target-set size. Compare methods within a condition rather than treating a raw cross-condition Recall difference as an isolated context effect.

- **POG_dense original score remains 0.395223.** In the separate one-seed test, BundleFuse scores 0.354494 versus SimpleHybrid 0.361017. The SimpleHybrid advantage is supported in both R@20 and NDCG@20 after correction (p = 0.019998). HardSwitch is also nominally ahead in R@20 (0.358252), but that particular Recall contrast does not survive correction.
- **Chronological retail:** the base condition favors BundleFuse, 0.118867 versus learned seed 0 at 0.096630 and both simple controls at 0.113179. With quarter-seed inputs, learned seed 0 reaches 0.077852 versus BundleFuse 0.068694; with one seed, 0.075765 versus 0.065616. These learned-model Recall advantages survive Holm correction (p = 0.019998). Neither sparse-retail NDCG contrast establishes learned-model superiority; the metrics give different signals.
- **Spotify one-seed:** HardSwitch has higher NDCG@20 after correction; its small Recall lead (0.062468 versus 0.061574) is not significant after correction. BundleFuse still beats every learned seed in Recall in that condition.
- **POG and Spotify_sparse:** BundleFuse stays ahead of the tested controls and learned rankers in all tested seed conditions. For POG, half, quarter and one-seed conditions coincide because the original query has only one or two seeds; they are not three different severity levels on that dataset.

These are genuine failure regimes for frozen settings, not a change to the original benchmark scores. A model retuned for single-seed queries was not tested, and no default was changed based on these test results.

## Chronological protocol

Training, validation and test use strictly separated day ranges 1–515, 516–580 and 581–711. Sampling is capped at 21,000/3,000/6,000 baskets after the temporal boundaries are fixed; eligible populations are 103,496/14,790/29,479. All 92,353 metadata-catalogue products remain candidates. A preflight found 60,753 catalogue items absent from the sampled training baskets and 2,321 test baskets containing a cold target; those targets were retained. Interaction graphs, item-frequency counts and auxiliary household histories use training interactions only.

This is a fixed-catalogue temporal experiment. Product metadata is assumed available transductively because introduction dates are unavailable. It establishes a future-basket result under that assumption and sampling budget, not a product-launch or full-history deployment simulation. Retail uncertainty resamples households as clusters.

## Measured efficiency

The learned ranker is substantially faster in the measured warm-cache pipeline. Hardware context: Apple M5 Pro, 24 GiB RAM, 15 physical cores, with one configured computation thread; see [hardware.json](hardware.json). The following are median **amortized milliseconds per query in a 32-query batch**, including metric calculation, after warmup; three repeats use the same selected query IDs for every method. They are not single-request or tail-latency measurements.

| Dataset | BundleFuse scoring + metrics | Learned seed 0 scoring + metrics |
|---|---:|---:|
| pog | 10.148 | 0.353 |
| pog_dense | 8.773 | 0.232 |
| dunnhumby_chrono | 11.230 | 0.625 |
| spotify | 81.441 | 2.523 |
| spotify_sparse | 58.385 | 1.908 |

Learned catalogue vectors are precomputed; catalogue-encoding costs are recorded separately. BundleFuse uses the existing full signal-computation path, including signals later given zero weight. The measurement describes this implementation, rather than a theoretical lower bound for a selectively optimized scorer.

The default BundleFuse scorer avoids gradient-training an encoder, but its graph/content preprocessing and validation fitting are real costs. Learned training plus validation across the grid and repeat seeds took about 5.8–34.4 minutes per dataset. This is six training runs per dataset, not the training time of just the selected seed-0 model. Supplied-fold BundleFuse/control settings were reused, so their earlier search cost is excluded from the new timing totals; the temporal job includes a new search. External embedding extraction/pretraining remains excluded. Whole-worker peak RSS of roughly 4.0–10.5 GiB includes shared data and multiple models and cannot establish per-method memory superiority.

No blanket latency, memory or zero-cost advantage is supported by these records. Accuracy, offline fitting, catalogue encoding and warm serving cost must be stated separately.

## Training quality and inferential scope

All primary seed-0 models stopped by validation patience. Spotify confirmation seeds 1 and 2 reached the 100-epoch budget with their best checkpoint at epoch 90; their later convergence is unestablished. These are reported budget limits, not evidence that all optimizations converged. Most other selected models met the configured patience rule. Learned weights, checkpoint hashes, learning curves and all grid diagnostics are retained locally; `.pt` files are Git-ignored and can be regenerated from the protocol.

Every base and stress comparison uses the prespecified family of 200 tests. Pointwise 95% cluster-bootstrap intervals use 2,000 replicates; two-sided cluster sign-flip p-values use 10,000 samples and Holm adjustment. Repeated adjusted p-values near 0.0200 reflect the Monte Carlo resolution (minimum raw p = 1/10,001). Inference conditions on fitted models and folds. Three-seed dispersion is separate from the query-bootstrap intervals. Published aggregate neural scores do not enter these paired tests.

## Assessment

The absence of a trained feature-matched competitor and a chronological test has been addressed for this experiment suite. Original-input accuracy holds up well. Sparse-input robustness concerns were confirmed in specific regimes rather than eliminated, and the timing audit rejects an unqualified inference-speed claim. Evidence supports a useful closed-form hybrid with explicit operating limits. Broader model-family coverage, different product-availability assumptions and a validated sparse-context adaptation would be separate studies.

Verification: [VALIDATION.md](VALIDATION.md), [audit log](verification.log), [full report](REPORT.md), and [prespecified plan](PLAN.md).
