# Compact-model feasibility pilot

This development study tests whether compressed engineered features help a compact ranker
under the original supplied POG_dense and Spotify inputs. These test sets have been inspected
in prior research; this is not a new independent confirmatory holdout. No new published
CLHE/RaMen training, distillation, or sparse-input adaptation is included.

## Stage 1: exact scorer optimization

Preserve the existing full-signal implementation as an independent reference. Compute only
dependencies of a validation-frozen BundleFuse configuration. Reuse the seed indicator and
weighted training-bundle overlaps; skip unused diffusion, incidence, Rao, taxonomy and content
signals. Keep operator arithmetic and catalogue-wide z-normalization unchanged.

Compare float32 fused score bits for **every candidate on every original supplied test query**.
Identical scores imply identical rankings under the shared item-ID tie rule. Also reproduce
every historical per-query R/N@10/20/50 record. Any mismatch fails the worker before training.
Fixtures independently cover fashion/music, one/multiple seeds, incidence mixes, Rao and backoff.
Full-population empirical parity initially covers the two pilot datasets, not every dataset/backend.

## Stage 2: matched compact-model comparison

Both arms use a 256-dimensional frozen input per item, a shared learned linear item projection
256→128, and a query MLP (260→64→128, with input LayerNorm). Pool the mean and variance of
revealed item vectors plus log seed count, mean/min log frequency and cold-seed fraction.
Rank all candidates by a normalized 128-dimensional dot product, learned scale and global
popularity coefficient. There are **58,314 trainable parameters** in each arm, no trainable
item/category-ID lookup tables. Both towers use the same item projection; they are intentionally
capacity-limited and do not claim exact representation of the teacher's nonlinear operators.

The raw and engineered arms have identical first 97 columns: fixed projected raw multimodal
content (64), fixed artist/album or category vectors (32), standardized log frequency (1).
The raw arm fills the other 159 columns with additional raw content (128) and taxonomy (31)
projections, so both arms use the full feature budget. The engineered arm uses those columns for:

- Left/right rank-32 factors of the asymmetric degree-normalized collaborative operator (32+32).
- Residualized style content (32).
- Content bridge (32).
- Projected category-PMI features (31).

Fixed random projections use seeds 10/11/12/13/14 and fixed taxonomy maps use 50+j.
Collaborative/ontology/bridge interaction fitting uses the supplied training fold only.
The existing transductive metadata catalogue and externally pretrained content are retained.
Common raw blocks remain byte-identical; the experiment measures feature representation within
this fixed compression budget, not superiority over every possible raw-feature architecture.
Hashes and byte sizes of the frozen inputs are recorded. Frozen training inputs and cached
128-dimensional serving vectors are different costs and are reported separately.

Train only on supplied training bundles using fresh half-reveal masks and sampled-softmax
supervision (one sampled hidden positive, 32 uniform negatives excluding the whole known bundle).
No teacher labels, context-dropout curriculum or candidate-frequency shrinkage is added.
Both arms use batch 128, learning rate {0.001,0.003}, weight decay/representation penalty 1e-4,
100 epochs maximum, validation every 5 epochs and patience 4 checks. Each arm selects its
learning rate using seed-0 **full original validation NDCG@20**. Confirmation seeds 1 and 2 use
that rate with their own validation-selected checkpoint. Four fits per arm, eight per dataset;
no best-test-seed selection or ensembling. Report capped runs explicitly.

## Go/no-go decision and reporting

Before test evaluation, freeze checkpoints and a validation-only feasibility decision:
engineered seed 0 must exceed raw seed 0 in validation NDCG@20 and retain at least 97.5% of
frozen BundleFuse validation R@20 **and** NDCG@20, on **both** datasets. This is an explicit
engineering tolerance, not statistical equivalence or a claim to match BundleFuse exactly.
A failure pauses the student research direction for reassessment; no later stage starts automatically.
Test results assess the chosen models but do not choose features, thresholds or checkpoints.

Primary test comparisons are engineered s0 vs raw s0, engineered s0 vs BundleFuse, and raw s0
vs BundleFuse, each in R@20 and NDCG@20: 12 contrasts over two datasets. Use 2,000 cluster
bootstrap replicates, 10,000 paired cluster sign flips and Holm correction across all 12.
Group identical complete bundles and preserve duplicate-query multiplicity. Training-seed mean
and population SD are reported separately from conditional query uncertainty.

Measure both reference and optimized BundleFuse alongside both seed-0 students on identical
32-query warm CPU batches (one thread, one warmup, three repeats). Report scoring and
scoring-plus-ranking/metrics separately; offline preprocessing, feature construction, fitting,
catalogue encoding and cached bytes separately. Worker RSS is not per-method memory.
External content pretraining/extraction is excluded throughout.

Frequency shrinkage alone is not an impossibility result for dot products: if Y=<a_c,q_Y>
and prior=<b_c,q_prior>, the posterior is <[(1-u_c)a_c,u_c b_c],[q_Y,q_prior]>.
The substantive feasibility issue is whether a small representation retains the useful full
collaborative and higher-order information. Distillation, if studied later, would primarily
test the value of denser ranking supervision rather than claim previously unavailable inputs.

Run or resume with unchanged source and inputs:

```bash
python3 -u -m bundlefuse.compact_experiments --output results/compact_pilot
```

The runner executes datasets sequentially, fingerprints code/data/reference results, preserves
attempt logs, checks completeness and updates the README after job transitions.
