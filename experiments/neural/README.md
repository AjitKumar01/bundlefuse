# Independent neural bundle completion

This pilot trains directly on masked training bundles. It never reads BundleFuse scores,
operators, predictions, or teachers. BundleFuse checkpoints are used only in the separate
comparison module. Production scoring code is unchanged.

## Architecture

Each item combines a trainable 64-dimensional identity vector, a learned projection of
its supplied text features, a 16-dimensional category embedding, and log training frequency.
A shared MLP produces a 64-dimensional item representation. Unseen identities use a common
unknown embedding. Training randomly drops 10% of identities, independently of category/text.

The three variants have the same item inputs and training protocol:

- `dot`: two layers of set self-attention, masked mean pooling, learned query/item projections,
  and a dot-product score.
- `relations`: candidate-conditioned MLP messages from each observed item; masked sum and max
  pooling; a final MLP receives the pooled messages, candidate representation, log set size and
  log candidate frequency. No self-attention among observed items.
- `contextual`: the same candidate scorer, with two layers of four-head set attention before
  candidate–item messages. Messages receive candidate, original observed-item representation,
  contextualized observed-item representation, and candidate/context elementwise product.

There are no positional encodings, hence the score is invariant to observed-item ordering.
`relations` duplicates the original/context representation in the shared message interface,
keeping its relation/decoder shape identical to `contextual`. Parameter counts are recorded
separately for item-ID tables and other weights, alongside frozen feature-buffer bytes.
None of these architectural ingredients is claimed to be individually new.

## Initial datasets and comparison scope

| Dataset | Items | Training bundles | Validation queries | Test queries |
|---|---:|---:|---:|---:|
| Electronic | 3,499 | 1,211 | 173 | 346 |
| Food | 3,767 | 1,232 | 176 | 353 |
| Clothing | 4,487 | 1,294 | 184 | 371 |

Use the existing `extended` data construction, resplit seed 0, half-reveal evaluation and
the complete validation/test partitions. These are the smallest datasets by item and training
bundle counts. All complete bundles are disjoint across train/validation/test, checked at load.
The hashes match the retained `reorganized_162dd8c` checkpoints. This random-split task is not
a chronological recommendation benchmark. These splits were inspected in previous research.

Neural models and the newly fitted `bundlefuse_matched` control receive the same training
bundles, text-feature arrays, and category labels. Catalogue text TF-IDF/SVD is the existing
loader's unsupervised attribute preprocessing across the known catalogue; it uses no held-out
bundle labels. This is not a claim that absolutely no SVD exists in input preprocessing:
there are no co-occurrence/PPMI/SVD **interaction scores** in the neural model.

`bundlefuse_original_auxiliary` additionally uses the old auxiliary/user histories; its exact
test scores are replayed and checked against the saved report. It is explicitly an unequal-input
reference. Its checkpoint is read-only and checksummed before/after. The matched control
copies its configuration, removes auxiliary/user inputs and personalization, and refits fusion
on validation. BundleFuse and the neural models have different optimization/search budgets.

Other comparisons are the repository defaults for popularity, co-occurrence, item cosine,
content cosine, p3alpha and rp3beta. The latter names denote local implementations, not new
paper reproductions. No CLHE/RaMen numbers from different datasets are placed in these tables.

## Training, selection and testing

For each training bundle, randomly permute its items. With probability one half, reveal half
the items; otherwise reveal a uniformly selected nonempty proper subset size. Keep every
unrevealed item as a positive. Sample up to 512 distinct candidates outside the complete bundle.
Epochs 1–2 use 256 uniform negatives; subsequent epochs use 128 uniform and 128 highest-scoring
remaining candidates from that pool. The loss averages, over positives then bundles, each
positive's softmax loss against the selected negatives. Other known positives are excluded
from its denominator. This hard-negative sampled loss is not exact full-catalogue likelihood.

AdamW: learning rate .001, weight decay .0001, gradient clipping 5, batch 16. Maximum 80 epochs;
full-catalogue validation every two epochs; stop after six unsuccessful validation checks.
Select the highest validation NDCG@20, retaining the earliest epoch on an exact tie. Recall is
also recorded. Use training seeds 0/1/2 on one fixed data split. No hyperparameter grid is run
in this initial pilot. All three architectures and all seeds are frozen before neural test scoring.

Test ranking covers every catalogue item and masks observed items. Ties use ascending item ID,
and Recall/NDCG@20 use the shared project evaluator. This avoids retrieval-stage truncation.
Per-query scores are not stored, but top-20 lists and query metrics are. Across-seed variability
is descriptive; it is not a statistical significance or non-inferiority claim.

CPU, one Torch/BLAS thread, deterministic Torch operations. Item vectors are cached within each
evaluation call; candidate-conditioned scores still require neural computation for every candidate.
Evaluation seconds include ranking/metrics and are not a controlled serving-latency benchmark.

## Run and resume

```sh
.venv/bin/python -m unittest tests.test_neural -v
.venv/bin/python -u -m experiments.neural.run --dataset electronic --smoke --epochs 2 --seeds 0 --output results/neural_smoke --checkpoints checkpoints/neural_smoke
screen -dmS independent_neural_small zsh scripts/run_neural_small.sh
tail -F results/neural_small/queue.log results/neural_small/electronic/run.log
```

The queue runs Electronic, Food, then Clothing. Each dataset trains all three variants and
three seeds, then evaluates frozen neural checkpoints. Interruptions resume from the latest
complete validation checkpoint, with optimizer state and NumPy/Torch RNG states restored.
Source/protocol mismatches fail closed; changed experiments need new output and checkpoint roots.

Every evaluated epoch is retained under `checkpoints/neural_small/<dataset>/<variant>_seedN/`.
There is no checkpoint cleanup. The dataset directory also contains comparator checkpoints and
a source archive. Results include protocol/input hashes, epoch loss/validation history, selected
checkpoint pointers, per-query predictions, parameter counts, and automatically updated reports.
Logs are appended rather than overwritten. Smoke outputs are separately labelled and must never
be used as benchmark results.
