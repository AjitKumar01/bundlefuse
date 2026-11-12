# Independent neural revision

**Current stage: complete.** The initial revision and constrained follow-up completed on Electronic.
The validation-selected `graph_pair_no_category` architecture also completed on Food and Clothing
without architecture/hyperparameter changes, with three seeds per dataset. See the
[consolidated results](../../results/neural_retry/README.md). The constrained Electronic stage used
`scripts/run_neural_constrained.sh`, with separate roots `results/neural_constrained` and
`checkpoints/neural_constrained`; Food/Clothing use `results/neural_transfer` and
`checkpoints/neural_transfer`. See
[`FINDINGS.md`](../../results/neural_revision/FINDINGS.md) for the evidence and limitations.

The follow-up conditions are graph+pair without the category table, graph+pair+joint without that
table, and graph+pair+joint with a nonnegative category table. Pair/joint weights are projected to
nonnegative values after optimizer steps in all three conditions. The category table is omitted
entirely in the first two, so it contributes no trainable parameters. All retain full-catalogue loss
and the same history masking. Run three conditions at seed 0, then repeat the validation-selected
condition at seeds 1 and 2. Tests/probes run as below. Do not resume the initial revision with changed
source: its exact original source archive is retained alongside its checkpoints.

```sh
tail -F results/neural_constrained/queue.log results/neural_constrained/electronic/run.log
```

The first queue was stopped at the user's request during Clothing. Electronic and Food finished;
their results and all Clothing checkpoints remain under `results/neural_small` and
`checkpoints/neural_small`. This experiment has separate code, outputs, checkpoints and source archives.

## Evidence that motivated the change

On Electronic, 40.8% of hidden test-item occurrences have zero training frequency. The previous
network exposed this frequency to its item encoder and final scorer. Zero-frequency items could
be sampled as negatives but never as positives. Identity dropout did not hide frequency, so it
did not reproduce an unseen item's inputs. Validation performance declined while training loss fell.

Validation-only linear probes of the frozen seed-0 relation network recovered little of the
BundleFuse category (R2 .029), text (.127), bridge (.065), and joint-surplus (.079) targets.
Pair/co-occurrence/PPMI recovery was .28–.32. The probes fit half the validation queries and
evaluate the other half. They use a fixed mixture of source-top and random candidates. These
are limits of LINEAR recovery from a particular learned representation, not proofs of neural
incapacity. Popularity recovery near 1 is unsurprising: log frequency was explicitly an input.
See `results/neural_revision/diagnostics/electronic/` for raw diagnostics and scope.

## Architecture

The new scorer is a learned relation network. It does NOT consume BundleFuse scores, matrices,
weights, inverse-frequency rules, PPMI, fused predictions, or teacher labels. Its graph support
is constructed afresh from training bundles, without counts: each observed link occurs once
in the index, even if seen repeatedly. The number of masked training examples then controls
how often that relationship receives supervision. Link weights can be positive or negative.

1. **Attributes:** use the same normalized 128-dimensional title TF-IDF/SVD features as the
   input-matched BundleFuse control. A 128→32→128 GELU network learns a residual to the text
   vector, followed by L2 normalization. The last layer starts at zero, preserving text geometry.
   A trainable positive temperature scales candidate/mean-seed text inner products.
2. **Category compatibility:** a free ordered category×category weight matrix starts at zero.
   Sum the learned compatibility of each observed category with the candidate category. This
   shares supervised evidence across items. It is not a fixed category-similarity score.
   Categories absent from training have their relationship rows/columns masked to zero, so
   they cannot acquire weights solely from appearing as negative candidates.
3. **Specific pairs:** one free weight per directed observed training item pair, initialized at
   zero. Sum the relevant weights for each candidate. This avoids forcing all observed pair
   coefficients through a low-dimensional item embedding table.
4. **Joint observations:** optionally add one free weight per (unordered observed seed pair,
   candidate) triple present in training bundles. It starts at zero and contributes only when
   both seeds are present. Pair and joint coefficients are trained separately.
5. **Attribute propagation:** optionally average current learned text vectors over observed
   training neighbours and apply a learned linear projection before normalizing item vectors.
   The projection starts at zero. This is one learned graph layer, not BundleFuse's bridge rule.

The final score adds these learned terms. There is no item-ID embedding table, no frequency
feature, and no prescribed fallback gate. The category table and sparse relation tables count
as trainable parameters, not free storage. The title preprocessing remains the existing
unsupervised catalogue TF-IDF/SVD; this is not a rich pretrained description/image encoder.

This is a deliberate architecture change, not just a hyperparameter rerun of v1. Its sparse
linear relationship terms, graph propagation and shared attribute encoder use established
ideas; no originality claim is made for their assembly. In particular, a learned sparse
relationship table is not a new invention. The objective is an independent competitive model.

## Corrected training

- Every hidden item is a positive. Its denominator contains all catalogue items outside the
  complete training bundle. Other known positives do not compete with it. No sampled negatives.
- Half of training masks reveal half a bundle; half choose a random nonempty proper subset size.
- In cold-training conditions, hide the history of a random 25% of catalogue items per batch:
  remove all direct, joint and graph-message paths incident on those IDs. Text/categories remain.
  Hidden-history items can still be positive targets. The network gets no indicator of the mask.
- Full-catalogue validation every two epochs; highest NDCG@20 selects the checkpoint. Maximum
  100 epochs; stop after six unsuccessful validation evaluations. AdamW, batch 32, link/category
  learning rate .03, shared encoder learning rate .001, weight decay .001, gradient clip 5.

Initial Electronic conditions (seed 0): `attributes`, `pair_no_cold`, `pair_cold`, `joint_cold`,
`graph_joint_cold`. Select the highest VALIDATION NDCG architecture, then train that condition at
seeds 1 and 2. Freeze all choices before neural test evaluation. This is a sequential exploratory
pilot; the repeated winner's seed-0 validation selection is disclosed. It is not a full factorial
comparison or an equal-compute comparison with the original BundleFuse fitting procedure.

The initial revision and architecture search were restricted to Electronic. After validation
identified the final architecture, the transfer runs fixed `--phase constrained --fixed-condition
graph_pair_no_category`, rather than repeating the architecture search on Food or Clothing.

## Verifying signals without distillation

`diagnose.py` fits separate Ridge probes of v1 frozen neural hidden representations. The probe
targets are BundleFuse raw standardized signals, but the probes are discarded and NEVER used
in neural training, model initialization, checkpoint selection or inference.

`probes.py` repeats validation-only linear recoverability and component rank-alignment checks
for each revised condition after selection. It also measures the initial attribute model, so
preservation at initialization is not falsely called a learned improvement. V1 probes have 194
inputs and revision probes use component scores; do not treat their R2 difference as a controlled
architecture-capacity test. Shared sparse support can inflate rank agreement. None of these
diagnostics proves that every BundleFuse mechanism has been learned or that the scores are sufficient.

Synthetic assays separately check: arbitrary direct/group coefficients can be represented exactly;
set order does not matter; cold masking removes every history route; initial text geometry is
preserved; gradient training actually learns a small completion task. Learned spectral, category
saturation, Rao and propagated scores are not guaranteed to be reproduced by these tests.

## Run, logs, retention

```sh
.venv/bin/python -m unittest tests.test_neural_revision -v
.venv/bin/python -u -m experiments.neural_revision.diagnose --dataset electronic
.venv/bin/python -u -m experiments.neural_revision.run --dataset electronic --smoke --epochs 2 --output results/neural_revision_smoke --checkpoints checkpoints/neural_revision_smoke
screen -dmS independent_neural_revision zsh scripts/run_neural_revision.sh
tail -F results/neural_revision/queue.log results/neural_revision/electronic/run.log
```

Every evaluated epoch retains model, optimizer and RNG states. Resume checks both result-root and
checkpoint-root source/protocol identities. Existing files are never deleted. Selected checkpoints
are replayed against validation metrics. Source archives, baseline checksums and per-query test
predictions remain on disk. Results show cold/warm target micro recall separately from macro query
Recall/NDCG. Source files must not change while the frozen experiment is running.
