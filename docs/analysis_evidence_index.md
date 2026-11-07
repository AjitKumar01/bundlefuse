# BundleFuse analysis: preserved evidence and current interpretation

The earlier analyses are retained. The new mathematical account is
[scoring_principle.tex](scoring_principle.tex). It explains the exact operators,
their different information content, the final optimization, and their effects
on candidate score differences. It supplements the earlier publication documents;
it does not silently replace their historical claims or experimental records.

## What is established by the new derivation

| Claim | Evidence | What this does not establish |
|---|---|---|
| The group term distinguishes training collections that have identical complete item-pair matrices. | Explicit two-corpus construction, proof, and the actual scoring operator. | A superiority theorem over models that also receive bundle membership. |
| The implemented SVD score compares projected PPMI row profiles; it is not the rank-k reconstruction of PPMI. | `F F.T = R M V_k V_k.T M.T R`, checked against the implementation. | That shared-partner similarity always indicates a correct completion. |
| Category centering is a least-squares removal of category means. | Orthogonal projection before row normalization, checked against the operator. | Independence of category and style, or orthogonality after normalization. |
| The bridge includes descriptor-to-partner and partner-to-partner comparisons. | Four-term expansion of its Gram matrix, checked numerically. | A causal estimate of the accuracy gain from each term. |
| The final score uniquely minimizes a frequency-weighted quadratic disagreement with the observed and fallback scores when kappa > 0. | Analytic derivative, positive Hessian, numerical stationarity using the frozen dense recipe. | Globally optimal ranking, or an empirically verified noise model. |
| Seed-frequency and bundle-size discounts have different roles from candidate-frequency backoff. | Exact conditional-inclusion rewrite and contribution calculations. | Complete removal of popularity or equal influence for all bundles. |

Run the new checks from the repository root:

```sh
python3 scripts/check_scoring_principle.py
```

The output is [scoring_principle.json](../results/algebraic_diagnostics/scoring_principle.json).
It records numerical errors, the two synthetic corpora, their actual scores,
and hashes of the code and dense recipe used. These are mathematical verification
cases, not newly reported recommendation benchmarks. They do not change the
model, fit additional weights, modify checkpoints, or use test outcomes to select
a scoring rule.

## Earlier evidence retained

- [Proposed sufficiency experiments (PDF)](../output/pdf/score_sufficiency_experiments.pdf)
  and [TeX source](score_sufficiency_experiments.tex): prioritized tests with equations,
  matched controls, practical thresholds and explicit confirmation limits. The
  [feasibility audit](../results/algebraic_diagnostics/sufficiency_experiment_feasibility.json)
  records query sizes and storage estimates from retained artifacts. No new model
  fitting or ranking evaluation was performed for this proposal.
- [Score sufficiency and stopping principle](novelty_assessment.md#12-when-are-the-retained-scores-sufficient):
  exact shrinkage objective, metric-specific information criterion and ranking bound,
  conditional correlated-measurement theorem, and a finite-family accuracy/cost guarantee.
  Available as a self-contained [PDF](../output/pdf/score_sufficiency.pdf) and
  [TeX source](score_sufficiency.tex), including dense-score definitions.
  The [verification script](../scripts/check_score_sufficiency.py) and
  [saved checks](../results/algebraic_diagnostics/score_sufficiency.json) also show that
  the current fashion scores can miss four-item relations despite identical pair/triple
  counts. These are mathematical checks, not a minimality certificate or new benchmarks.
- [Algebraic mechanism analysis](algebraic_mechanism_analysis.md): original operator
  analysis, quadratic query expansion, the first ranking counterexample, and the
  centering/backoff interaction.
- [Higher-order verification script](../scripts/check_higher_order_structure.py)
  and [saved verification](../results/algebraic_diagnostics/higher_order_toy.json).
- [Query overlap analysis script](../scripts/analyze_evidence_overlap.py)
  and [saved query counts](../results/algebraic_diagnostics/evidence_overlap.json).
- [Existing equation audit](../scripts/audit_argument_equations.py) and
  [saved audit](../results/argument_audit/equations.json).
- [Method prior-art review](method_prior_art_review.md): the wider literature
  search, close precedents, and qualifications about formula equivalence.
- [Technical publication argument](publication_argument_technical.tex) and
  [literature-led publication argument](publication_argument.tex): earlier
  manuscript versions retained for comparison. Statements about which experiments
  were pending describe their writing time, not current queue status.
- Completed [POG dense mechanism evidence](../results/publication_mechanisms/pog_dense/)
  and [POG mechanism evidence](../results/publication_mechanisms/pog/): frozen
  recipes, summaries, per-query results and reports. The new derivation quotes
  these saved results; it does not rerun or relabel them.

## How to state the argument

Start with the distinction between **direct co-membership, simultaneous
co-membership, and shared association patterns**. These are different statistical
objects, even though all can be derived from the same training bundles. The
two-corpus example proves that collapsing bundles into pair counts discards
information used by the group term. The SVD identity proves that the spectral
channel supplies shared-profile similarity rather than another copy of direct
support. Category and descriptor operations extend evidence to candidates with
few or no observed co-memberships. The final score reconciles the selected
signals according to their catalogue-relative separation and candidate support.

This yields a specific construction to explain and evaluate, not a claim that
every ingredient is new or that every available channel contributes to every
dataset. In particular, the full dense recipe does not select standalone cooc or
PPMI, and the full POG recipe selects neither incidence nor SVD. The exact
optimization interpretation remains valid without asserting a calibrated Bayesian
posterior: the dense variance fit has a negative raw slope and the selected
shrinkage is validation-tuned.

The existing supplied-split comparisons establish empirical performance under
their recorded protocols. The algebra supplies the mechanism argument. Neither
should be substituted for the other.
