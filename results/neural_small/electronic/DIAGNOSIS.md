# Electronic: first independent neural pilot

All nine runs completed. The proposed neural scorer does not match BundleFuse.
The mean over three training seeds is the primary neural summary; no best-test-seed selection.

| Model | Macro Recall@20 | Macro NDCG@20 |
|---|---:|---:|
| Original BundleFuse, additional auxiliary inputs | 0.486802 | 0.343218 |
| Input-matched BundleFuse | 0.460453 | 0.344215 |
| Neural dot-product control | 0.316763 | 0.200580 |
| Neural candidate relations | 0.374647 | 0.266590 |
| Neural contextual relations | 0.362717 | 0.256340 |

The best neural architecture by these descriptive means is the simpler relation scorer:
18.6% lower Recall and 22.6% lower NDCG than input-matched BundleFuse. Adding set attention
does not improve the mean and increases across-seed variation. No significance claim is made.

## Observed weaknesses

There are 1,211 training bundles for 3,499 catalogue items. Of the items, 680 never appear in
training, 1,935 appear once, and 3,272 appear at most twice. Training-frequency counts refer
to bundle membership, not gradient updates or negative-sampling exposure.

Of the 600 hidden test-item occurrences, 245 (40.8%) have zero training frequency. The
post-hoc diagnostic below counts hits per target occurrence; it is MICRO recall and must not
be substituted for the macro query Recall@20 above.

| Target training frequency | Input-matched BundleFuse | Neural relations, mean | Neural contextual, mean |
|---|---:|---:|---:|
| Zero | 0.200000 | 0.102041 | 0.096599 |
| One or two | 0.625000 | 0.558594 | 0.554688 |
| Three or more | 0.535354 | 0.478114 | 0.430976 |

The gap is present in every stratum. Unseen-item predictions account for approximately half
the missing hits relative to input-matched BundleFuse. Exact hit counts and denominators are
in `target_frequency_diagnostic.json`.

Every run's final training loss is lower than its loss at the selected epoch, while validation
NDCG has declined. This is consistent with overfitting; more epochs alone are not supported by
these curves. Candidate-conditioned scoring improves the dot-product control, but the additional
set-attention layers do not earn their complexity in this pilot.

## A concrete training mismatch to investigate

The scorer observes the candidate's full-training frequency in both item encoding and final
decoding. Zero-frequency catalogue items can enter negative pools but never appear as training
positives. Dropping a known item's identity does NOT change its frequency feature, so the current
identity dropout does not simulate a genuinely unseen item. This is a plausible source of cold-item
suppression, not a causally established explanation of the whole gap.

The next controlled change should compare the existing relation scorer with simulated unseen-item
training that hides BOTH identity and frequency information consistently. A frequency-feature removal
control can distinguish this effect from a generic regularization gain. On the small catalogue, a
full-catalogue loss control would separately test the hard-negative sampling objective. Any new
selection must use validation only, keep this pilot's checkpoints, and use separate output roots.
No running experiment source was changed during this diagnosis.

## Published RaMen reference

[RaMen Table 2](https://arxiv.org/html/2507.14361v1) reports Electronic Recall@20 0.8371 and
NDCG@20 0.5373, substantially above this pilot numerically. Its paper uses visual/textual BLIP
features and reports 1,750 bundles. Our local pipeline deduplicates to 1,730 bundles and uses
text TF-IDF/SVD plus categories, with a locally reconstructed split. Thus this is a published
reference, not a verified identical-input/split comparison. The direct shortfall against the
input-matched local BundleFuse control is already established without that comparison.
