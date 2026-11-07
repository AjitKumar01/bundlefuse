# Sufficiency experiments: completed primary results

All primary E1–E4 runs completed: POG_DENSE 56 configurations / 5,547 queries;
POG 39 / 3,995; Spotify 11 / 3,995. The queue finished at 21:16:29 UTC on
6 November 2026. All 106 saved metric means and row counts were independently
verified, and every original input checkpoint remains unchanged.

These are exploratory, matched comparisons on filtered supplied splits.
Validation/test complete bundles overlapping training were excluded; test
bundles overlapping validation were excluded too. Exact retained indices are
in each dataset's evaluation_population.json. Scalar settings were refitted
on grouped fitting/tuning validation data. These figures must not replace the
previous unfiltered supplied-split or published neural-baseline comparisons.

| Dataset / matched configuration | Recall@20 | NDCG@20 |
|---|---:|---:|
| POG_DENSE reference | 0.358362 | 0.257181 |
| POG_DENSE separate individual/joint support, refitted | 0.361036 | 0.261006 |
| POG reference | 0.056487 | 0.039958 |
| POG separate individual/joint support | 0.056487 | 0.039958 |
| Spotify canonical bank, refitted | 0.155809 | 0.263150 |
| Spotify amplified top-100 bank with original full-score gate | 0.162488 | 0.280295 |
| Spotify amplified bank with separate frequency gate on history | 0.163634 | 0.282711 |
| Spotify amplified bank with effective-support gate on history | 0.163714 | 0.282558 |
| Spotify explicit extra interaction term, refitted | 0.155809 | 0.263150 |

## Interpretation

- **E1, smaller banks:** the matched dense reference selected zero SVD weight;
  POG selected zero bridge weight. Those columns can be omitted from those
  exact frozen recipes without changing rankings. This does not prove they
  are always unnecessary. The exhaustive subset procedure uses a different
  warm-start path from the standalone reference, so it does not always recover
  the same coefficients, even for the full bank. Do not confuse subset count
  enumeration with global weight-search optimality.
- **E2, individual versus joint history:** dense POG benefits from separate
  standardized pair and joint columns. Refitting improves NDCG by 0.003826
  (about 1.49%) versus its validation-selected coupled control; recorded
  Holm-adjusted p = 0.037996. Recall improves numerically by 0.002674 but is
  not significant under correction. With other weights fixed, the NDCG gain
  is 0.002546 but its adjusted p = 0.130187. Therefore the significant result
  supports the refitted procedure, not an isolated causal attribution to the
  extra history coefficient. POG selects zero joint weight and exactly matches
  its pair/reference model.
- **E3, support gates:** effective support does not beat the separately tuned
  frequency control in any of the three datasets under the recorded correction.
  Dense POG selects zero support-gate strength, and its NDCG is slightly worse
  than the frequency control. Spotify's effective-support gate beats a constant
  historical coefficient on NDCG (adjusted p = 0.048595), but not frequency
  or unweighted contributing-bundle count. Keep frequency as the simpler
  default; no new effective-support superiority claim follows.
- **E4, third observed-item interaction:** fixed and refitted Spotify versions
  both select zero extra-term coefficient. The full-neighbor power-two and
  power-three banks also discard their explicit historical column. The retained
  top-100 amplified bank does use history and performs better. This supports
  that tested neighbor-restricted construction, not a blanket claim that
  higher-order relationships are useless.

## Corrected efficiency measurements

The original inference_timing.json files computed some zero-weight menu
columns. They are retained for provenance but must not support efficiency
claims. Rebenchmarking used only actual nonzero score dependencies and verified
stored top-20 predictions. Corrected files are
`<dataset>/inference_timing_active_dependencies.json`.

| Dense configuration | Active columns | Median batch-1 ms/query | NDCG@20 |
|---|---:|---:|---:|
| Reference: category, bridge, popularity, history | 4 | 6.193 | 0.257181 |
| Compact: category, popularity, history | 3 | 4.866 | 0.254851 |

The compact candidate was selected using saved tuning metrics, not its test
score. It is about 21.4% faster by median batch-1 latency and 34.6% by median
batch-8 latency, with about 0.62% relative Recall loss and 0.91% relative NDCG
loss. Its unadjusted paired NDCG difference interval is [-0.003663, -0.000958].
That crosses the negative 1%-loss margin (-0.002572), so noninferiority within
1% has not been established. Tail latency improves much less than the median.
Compact export size and per-model resident memory have not been measured.

The matched POG reference already has three active columns: category, style
and history. Its corrected median latency is 7.260 ms/query; merely removing
its zero-weight bridge does not provide a meaningful additional speed gain.
No refitted E1 subset met both development 1%-loss thresholds relative to
that reference. This is a limitation of the tested selection procedure, not
proof that all three columns are universally indispensable.

## Population effect versus refitting

For context, evaluating the unchanged earlier full model's saved predictions
on the retained query indices gives:

| Dataset | Unchanged original model on filtered population, R@20 / N@20 |
|---|---|
| POG_DENSE | 0.360014 / 0.258000 |
| POG | 0.053525 / 0.035244 |
| Spotify | 0.154882 / 0.257048 |

This helps distinguish the population change from the newly refitted recipes.
It is descriptive, not a matched fitting comparison. Original full-population
scores remain preserved in their original result directories.

## Evidence and remaining work

Per-dataset REPORT.md, summary.json and frozen_methods.json retain all method
results, validation selections and planned contrasts. review_summary.json and
completion_verification.json record the independent result checks. Reproduce
this review and corrected timing using `PYTHONPATH=. python scripts/review_sufficiency_results.py`.
All original logs, partial results, caches and both protocol/source snapshots
from the pre-evaluation 512-to-256 test-budget amendment remain retained.

The primary conclusion is a compact and dataset-dependent estimator: separate
pair/joint weighting is useful on dense POG, ordinary pair history suffices in
POG's selected recipe, and the tested Spotify gain favors truncated amplified
history with frequency control. The runs do not prove universal score
sufficiency or establish the originality of the complete construction.
Conditional bridge/probe studies and compact artifact/memory measurements
remain unrun. No fresh confirmation dataset or minimality certificate exists.
