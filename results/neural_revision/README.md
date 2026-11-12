# Independent neural revision: diagnostics and retry

**Completed:** the first revision and the constrained follow-up have finished. The final architecture
was also evaluated unchanged on Food and Clothing. See [consolidated results](../neural_retry/README.md).

Branch: `independent-neural-v2`. The previous queue was interrupted during Clothing at the user's
request. Electronic and Food completed; no old checkpoints were removed. The new Electronic-only
pilot runs in detached session `independent_neural_revision`.

```sh
tail -F results/neural_revision/queue.log results/neural_revision/electronic/run.log
```

- [Live Electronic results](electronic/REPORT.md)
- [Architecture, training and comparison protocol](../../experiments/neural_revision/README.md)
- [Old model's source-recovery diagnostic](diagnostics/electronic/SIGNAL_DIAGNOSTIC.md)
- [Machine-readable probe results](diagnostics/electronic/signal_diagnostic.json)
- [Pre-launch verification](verification.json)
- [Old queue stop record](../neural_small/stopped.json)

## What was actually verified

On validation queries, frozen v1 hidden representations poorly support linear recovery of several
BundleFuse source scores. For the relation model: category R2 .029, text .127, propagated text .065,
joint surplus .079; pair/co-occurrence/PPMI .28–.32. This is a descriptive linear-probe result,
not a proof that the neural family cannot express these scores. Popularity is explicitly an input,
so its near-perfect probe recovery is not evidence of a discovered mechanism.

The revised model has separate gradient-trained weights for observed directed item pairs and,
optionally, observed seed-pair/candidate triples. Arbitrary coefficients on those supports can
be represented without a low-rank factorization. Synthetic assays verify indexing, gradients,
set-order invariance and history masking. They do not show that benchmark training learns the
optimal coefficients. There is no guaranteed reproduction of all category, spectral or Rao terms.

Other changes preserve title-feature geometry at initialization, remove frequency inputs and
item-ID embeddings, prevent unseen categories from acquiring negative-only weights, use every
eligible catalogue item in the loss, and hide all history paths of simulated unseen items.

The retry tests five conditions at seed 0, selects one using validation NDCG, and repeats that
condition at seeds 1 and 2. All checkpoint choices are frozen before neural test scoring. Separate
diagnostic probes then examine the learned components; their targets never train the neural model.
Every evaluated epoch, optimizer state, source snapshot and per-query prediction is retained.

The paired no-cold/cold condition tests the history-masking change. Comparisons with v1 change
several architectural/training choices together, so they cannot identify one sole cause of any gain.
The splits have already been inspected and the study remains exploratory.
