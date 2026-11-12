# What the neural diagnostics actually establish

The original queue was stopped at the user's request. The first revised Electronic pilot then
completed all five seed-0 conditions and two repeats of the validation-selected graph/joint model.
No retained checkpoints were deleted, and no BundleFuse scores trained the completion models.

## Recovery of source structure is not sufficient for accurate fusion

The first revision's learned components support substantially better linear recovery of several
BundleFuse source scores than v1's frozen hidden representations. Examples on validation-only,
query-disjoint probe fit/holdout partitions:

| Diagnostic target | V1 relations, hidden representation | Revision pair/no-cold, component scores | Revision joint/cold, component scores |
|---|---:|---:|---:|
| Direct historical pair score | 0.281 | 0.665 | 0.626 |
| Co-occurrence | 0.302 | 0.719 | 0.658 |
| PPMI | 0.319 | 0.733 | 0.640 |
| Category | 0.029 | 0.288 | 0.298 |
| Text | 0.127 | 0.559 | 0.577 |
| Propagated text | 0.065 | 0.235 | 0.242 |
| Joint surplus | 0.079 | 0.267 | 0.576 |

These R2 values concern standardized source targets, NOT end-task accuracy. Probe input dimension
and representation type differ, so this is not a matched capacity comparison. A nonlinear encoding
could contain information that a linear probe misses. Sparse source rank alignment is partly driven
by shared nonzero support. The initial attribute model already achieves .945 text-target probe R2;
that preservation must not be attributed to training. Query-wise source standardization can prevent
exact linear recovery even when raw text similarities are reproduced exactly.

The revision's validation-selected graph/joint model achieved mean test Recall@20 .398812 and
NDCG@20 .269921 across three training seeds. This improves v1's full contextual network means
(.362717/.256340), but remains below input-matched BundleFuse (.460453/.344215). It does not establish
that the network has recovered the full BundleFuse scoring mechanism or that every source is learned.

The pair cold-mask control did not improve on its no-cold counterpart in this pilot. Thus the plausible
frequency/cold-input mismatch in v1 was NOT established as a sufficient explanation of the accuracy gap.
Graph propagation increased unseen-target recall descriptively, but other components changed relative
to v1 and the results are exploratory.

## A directly tested failure: unrestricted category penalties

In the fitted pair/no-cold model, 85.2% of the full category-table entries are negative, with mean -2.58.
Signs alone do not prove a problem: a common additive score offset would preserve rankings. We therefore
tested validation-only interventions on each frozen model:

| Frozen model | Original validation NDCG | Remove category term | Clip category weights at zero |
|---|---:|---:|---:|
| Pair/no-cold | 0.229467 | 0.295077 | 0.281095 |
| Graph/joint/cold | 0.268235 | 0.336880 | 0.324205 |

Removing or clipping the category term changes rankings and improves validation performance; this
establishes harm from this fitted term on those validation queries. It does not prove that categories
are intrinsically unhelpful or that negative compatibility is universally wrong. The no-category
graph/joint intervention reaches validation Recall .426493 / NDCG .336880 versus matched BundleFuse
.425723 / .350196. These are interventions, not fresh test results.

Full details: `electronic/pair_validation_interventions.json` and
`electronic/validation_link_interventions.json`. The interventions never overwrite trained checkpoints.

## Focused follow-up

The follow-up trains from scratch with projected nonnegative pair/joint weights, comparing graph+pair
without categories, graph+pair+joint without categories, and graph+pair+joint with a nonnegative category
table. Projection occurs after every gradient step, allowing gradients to reactivate a zero weight.
This differs from a dead ReLU parameterization. Shared neural text/graph parameters remain unconstrained.

Source/protocol snapshots and results are isolated under `checkpoints/neural_constrained` and
`results/neural_constrained`. Only validation selects the architecture and checkpoints. The selected
condition is repeated at seeds 1 and 2 before test scoring. Original and first-revision results are retained.
