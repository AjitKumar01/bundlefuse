# From the implementation to a defensible scientific argument

Analysis of `reorganized` at `660bff2`, 5 November 2026. This extends
[research_story.md](research_story.md) with a mathematical interpretation and the
specific evidence needed to defend it. No benchmark was rerun for this analysis.
Historical results below refer to `4516eab` on `evaluation-reliability-experiments`.

## Assessment

The strongest foundation is **explicit compatibility estimation with evidence shared
across items and frequency-dependent regularization**. The method already implements
this structure, but the inventory of named signals obscures it. The paper should
explain the structure first and distinguish exact mathematical identities, modeling
choices, and experimentally supported benefits.

Good accuracy establishes that the construction is useful on the measured tasks.
It does not establish that each component is necessary or that the proposed reason
for each gain is correct. Conversely, a useful method does not require every constant
to follow from a theorem. Constants and search choices can be stated openly and
supported with sensitivity checks.

The exposition below is a rational reconstruction of the current design, not a claim
about the historical order in which it was invented. It should not be presented as a
preregistered sequence of discoveries.

## Start from an established idea

A candidate can be scored by adding its relationships to the observed items. This
connects the work to item-based recommendation and explicit item-relation models.
[EASE](https://arxiv.org/abs/1905.03375) already shows that a simple model with a
closed-form solution can compete with deep recommendation models. Therefore,
"a simple method can beat a neural model" is motivation, not sufficient novelty.

Bundle construction also has established multimodal approaches:
[CLHE](https://arxiv.org/abs/2310.18770) combines bundle composition, item-level
feedback and multimodal features; [RaMen](https://arxiv.org/abs/2507.14361) combines
direct collaborative relationships, multimodal information and group structure.
They already recognize the value of multiple kinds of evidence.

The question for BundleFuse is more specific: **how much can be achieved by estimating
those relationships explicitly, extending them beyond observed pairs, and controlling
their influence without training a new neural bundle encoder?**

## Step 1: predict from historical relations

Let the catalogue contain n items. Let S be the observed set and c a candidate outside
S. Define x_S as the n-dimensional vector whose entries are one for observed items
and zero otherwise. Let B be the binary training-bundle by item matrix.

The first estimator is a sum of recorded item relationships:

\[
r(c\mid S)=(B^\top Bx_S)_c.
\]

For an eligible candidate, this counts its overlaps with the observed items across
training bundles. It immediately exposes two problems: popular seeds and large
bundles can contribute heavily, and absent pairs receive no direct support.

Let f_i be item i's number of training appearances, d_e the size of training bundle e,
D the diagonal matrix with entries (f_i+5)^(-1/2), and V the diagonal matrix with
entries (d_e+1)^(-1/2). The implementation's basic overlap score is

\[
p(S)=B^\top VBDx_S.
\]

This weights common seeds and large bundles less. The offsets and exponents are
design choices. They are not derived optimal corrections, and D weights the seed
side here; this formula should not be described as symmetric normalization.

There is an important implementation detail: this overlap computation retains the
diagonal. It does not affect eligible candidates directly, but observed-item scores
enter catalogue standardization and subsequent propagation. Replacing it with a
zero-diagonal operator everywhere would change the method.

Raw counts, positive PMI, propagation and collaborative SVD can now be introduced
as alternative summaries of history. They need not be four independent conceptual
contributions. Their redundancy is an empirical question.

## Step 2: share evidence across items

Direct item-pair estimates have limited coverage. A category pools evidence from
many items; descriptive vectors allow relationships even without observed bundle
co-occurrence. These are two concrete ways to share evidence.

Let A be the item-by-category membership matrix, T a category-relation matrix built
from training bundles, and F a fixed matrix of item features. Then representative
raw scores have the forms

\[
r_{\rm category}(S)=ATA^\top x_S,
\qquad
r_{\rm feature}(S)=FF^\top x_S/|S|.
\]

These illustrate common structure, not a replacement specification for every
dataset. Category repetition penalties can be encoded in T's diagonal; some modes
use additional group incidence matrices or soft memberships. A category relation is
not necessarily a positive-semidefinite kernel and need not represent similarity.

This matters for the story: two different product categories can go well together,
while two items of one category can compete for the same role. Music can favor
repetition of an artist or album. The code makes domain-specific choices about these
relations. It does not infer every such choice from one universal objective.

In fashion, subtracting a category's feature centroid removes the mean feature vector
of that category before normalization. This motivates looking beyond category
identity, but does not prove statistical independence from category or a disentangled
representation. Graph-adjusted features mix descriptions with propagated descriptions;
their usefulness is conditional in the ablations.

Most raw channels can thus be understood as explicit item-relation operators acting
on the same observed set. This is an algebraic unification of the implementation,
not a new theorem or a claim that the entire final scorer is linear in x_S.

## Step 3: combine comparable scores

For each raw channel h, let m_h(S) and s_h(S) be its mean and standard deviation across
the catalogue for the current query. Define

\[
z_h(c\mid S)=\frac{r_h(c\mid S)-m_h(S)}{s_h(S)+\varepsilon}.
\]

This changes units so one count-based score does not dominate solely because of its
scale. It makes no assertion of independent signals or calibrated probabilities.
The denominator depends on S, so even additive raw channels do not make the final
scorer a single fixed linear item matrix.

The implementation selects nonnegative weights through a finite, greedy validation
search. It then selects the optional multi-item terms and backoff settings. This is
fitting, despite the absence of neural gradient training. The sequential procedure,
signal order, pruning tolerances and search budget belong in the reproducibility
specification. They are not the solution of one proved global optimization problem.

## Step 4: derive the final mixture without an unsupported noise claim

Let Y(c|S) be the full combined score, P(c|S) the restricted fallback before rescaling,
s_0 its nonnegative scale, and kappa a positive backoff strength. Define mu_0=s_0 P.
The implemented score is exactly the minimizer over real t of

\[
J(t)=f_c(t-Y(c\mid S))^2+\kappa(t-\mu_0(c\mid S))^2.
\]

The first penalty discourages departure from the combined score in proportion to
the candidate's training frequency. The second discourages departure from the
fallback. Differentiating gives

\[
J'(t)=2f_c(t-Y)+2\kappa(t-\mu_0),\qquad
J''(t)=2(f_c+\kappa)>0,
\]

so the unique minimizer is

\[
\theta(c\mid S)=\frac{f_cY(c\mid S)+\kappa\mu_0(c\mid S)}{f_c+\kappa}.
\]

This is an exact optimization interpretation of the scoring formula. It does not
prove that frequency is optimal as a penalty weight, that either input score is
unbiased, or that the chosen fallback is best. When kappa is zero the implementation
explicitly returns Y, including for f_c=0; the objective would otherwise be
unidentified at f_c=kappa=0.

The same mixture has a Gaussian posterior interpretation under additional assumptions.
However, the archived four supplied-split fits all have negative raw slopes in the
regression of mean squared Y-minus-mu_0 disagreement on 1/f. Their slopes are about
-0.273, -2.528, -5.336 and -5.869. The code clips variance estimates and then uses
validation to select kappa. These fitted signs do not support treating that noise
model as established; they are not, by themselves, formal hypothesis tests.
Disagreement between two scores is also not measured prediction error relative to
ground truth. The existing historical manuscript already acknowledges the variance
problem, but should avoid retaining stronger confidence claims elsewhere.

Use the regularization interpretation as an exact explanation and the Gaussian model
as an optional, explicitly conditional analogy. Regularization identities alone do
not establish novelty. Related behavioral/descriptive backoff exists in
[empirical-Bayes product search](https://www.amazon.science/publications/addressing-cold-start-in-product-search-via-empirical-bayes).

## What the mixture actually changes

The fallback shares terms with the full score, including selected popularity and Rao
terms. Define H=Y-P and rho_c=f_c/(f_c+kappa). Algebra gives

\[
\boxed{\theta=\rho_c H+[\rho_c+(1-\rho_c)s_0]P.}
\]

When s_0=1, this becomes P+rho_c H: retain the shared fallback and attenuate the
remainder according to training frequency. With the actual selected scale, the
coefficient of P also varies by candidate. This is more precise than describing
two independent experts or implying that kappa affects only historical channels.

H is defined algebraically in general. For the four archived supplied-split selections,
the positively weighted fallback channels are reused with the same coefficients;
H therefore consists of the other terms. A fallback populated by default weights
requires separate interpretation.

There is a useful prediction to check. When every history-only channel is a constant
across cold candidates after standardization, Y=P+a(S) on those candidates for some
query-dependent constant a(S). With positive s_0, enabling backoff preserves their
ordering among themselves while changing their position relative to warm candidates.
This property requires those assumptions; auxiliary histories or different fallback
weights can invalidate them. It suggests measuring cross-frequency ranking changes,
rather than interpreting all gains as improved discrimination among cold items.

Both mixture identities were checked against `apply_fusion` using synthetic score
arrays and all four archived selected configurations. Maximum absolute discrepancy
was 4.77e-7 from float32 arithmetic. This checks algebra against implementation; it
is not a new accuracy experiment.

## Step 5: add multi-item refinements only after the core is established

The incidence term gives extra weight to a training bundle that contains at least two
observed items. Rao's quadratic term uses pairs of observed items, candidate support
and their specified similarities or dissimilarities. Both have explicit motivations
and vanish for one observed item.

Their scope should follow the ablations. Removing both preserves about 92–100% of the
Recall@20 advantage over HardSwitchHybrid. This makes them useful refinements, with
the strongest corrected evidence on music, rather than the principal explanation
for the large baseline gaps. The term "simplicial" in a class name is not evidence
that a new topological mechanism drives the results.

## The evidence that would make the argument convincing

First reconcile the paper with one identified set of executed configurations and
source/input hashes. The historical manuscript, supplied-split studies and later
reorganized/compressed runs must not be silently combined. Existing EASE/SLIM tables
are present in the historical manuscript; their protocol and provenance need checking
before treating those controls as absent or as fully verified for the new claim.

Then prioritize three studies, with validation-only design decisions:

1. **Find the smallest adequate core.** Compare historical relations alone, shared
   category/descriptive evidence alone, their fixed combination, and the same combined
   construction with backoff. Add the two multi-item refinements last. Give competing
   versions comparable validation budgets. Use the same observed/hidden sets. This
   asks which conceptual decisions account for accuracy, rather than merely whether
   removing one of many correlated channels hurts after retuning.
2. **Identify what frequency dependence buys.** Compare disabled backoff, a
   validation-selected constant mixture, and candidate-frequency backoff using the
   same Y and P. Cross these with unit versus matched fallback scale so scale and
   routing effects are distinguishable. Inspect cold/warm competition and frequency
   buckets; separately inspect query seed count. Report within-bucket ranking and
   full-catalogue retrieval, since the former can hide cross-bucket effects.
3. **Check dependence on selection.** Repeat selection with a prespecified small set
   of signal orders and validation resamples. Compare active families, prediction
   stability and held-out accuracy. Correlated channels may exchange weights without
   changing rankings; do not mistake unstable weights for unstable predictions.

Freeze the choices and analysis before new test inspection. Since existing test
results have already informed this research direction, describe these as follow-up
studies and use independent confirmation where possible. Do not select a minimal
model on the test set and present its score as fresh confirmatory evidence.

These experiments have interpretable failure outcomes. If a constant mixture matches
frequency backoff, narrow the routing claim. If a small core retains the accuracy,
make that core the main method and report optional additions separately. If most
signals are necessary, explain their marginal contribution and disclose the search
cost. If selected recipes are brittle, simplify or regularize selection before
claiming a general principle. No new expensive experiment was launched in this review.

## Proposed contribution and reader's path

The defensible contribution is an explicit, frequency-regularized compatibility
estimator, together with evidence about which relation constructions matter and where
the estimator fails. Its novelty must be assessed relative to item-relation models,
multimodal bundle encoders and existing backoff methods. The present results support
a competitive construction; they do not establish a universal advantage, calibrated
uncertainty, or an accuracy-preserving compact replacement.

Suggested opening argument:

> Historical item associations provide a natural starting point for completing a
> partial bundle, but their coverage is uneven and similarity alone need not imply
> suitability for the same bundle. We study explicit compatibility estimates that
> combine recorded relations with evidence shared through item categories and
> descriptions. Candidate training frequency controls regularization toward a
> restricted fallback. This construction permits direct examination of what is gained
> from shared evidence, candidate-dependent combination and multi-item refinements,
> without training a new neural bundle encoder.

The paper should proceed from that starting estimator to its observable limitations,
then to each modification and the test that could disprove its proposed benefit.
The full menu of formulas belongs after this path is clear. The goal is an explanation
that survives an unfavorable ablation, because its claims have explicit boundaries.

Implementation sources: `bundlefuse/operators.py`, `bundlefuse/selected_signals.py`,
`bundlefuse/basket_signals.py`, `bundlefuse/fusion.py`, and `bundlefuse/model.py`.
Evidence scope: [claims ledger](../paper/CLAIMS.md), plus archived supplied-split
selection files and the component/generalization/compact-pilot analyses at `4516eab`.
