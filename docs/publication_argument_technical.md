# BundleFuse: a corrected technical publication argument

The complete mathematical treatment is now in
[the LaTeX source](publication_argument_technical.tex) and
[the compiled document](../output/pdf/publication_argument_technical.pdf).
This Markdown version records the argument and its qualifications. Revised on
5 November 2026 using the completed POG and POG_dense mechanism studies; the music
mechanism studies are pending.

The earlier note's main algebra was correct, but its central emphasis and empirical
status needed revision. The corrected central claim is:

**Historical relations and evidence shared through item attributes provide
complementary compatibility estimates. BundleFuse combines them explicitly.
Frequency backoff and multi-item terms are conditional refinements.**

## 1. Start with a defined prediction problem

Let the catalogue have n > 1 items. Let S be a nonempty observed set, c a candidate
outside S, and x_S its binary indicator vector. Let B be the binary training-bundle
by item incidence matrix. Define item frequency f_i as the sum of column i and bundle
size d_e as the sum of row e. Scores rank eligible candidates; they are not defined
as probabilities.

The natural historical score is (Bᵀ B x_S)_c, which adds recorded co-memberships
between c and the observed items. This motivates correcting the influence of common
seeds and large bundles, and extending coverage beyond exact observed pairs.

## 2. Construct estimates with explicit roles

Define diagonal matrices D_ii = (f_i + 5)^(-1/2) and V_ee = (d_e + 1)^(-1/2).
The weighted overlap vector is

\[
p(S)=B^\top VBDx_S.
\]

D weights the seed side; this operator is generally asymmetric. It retains its
diagonal. Although the diagonal does not directly affect eligible candidates, it
influences propagation and catalogue standardization through observed-item entries.

For E_ee = (d_e + 1)^(-1), propagation uses L = off(D Bᵀ E B D), where off replaces
the diagonal with zero. L is symmetric, but is not generally row-stochastic.

Let A be item-category membership, T a category-relation matrix with candidate
categories on its rows and observed categories on its columns, and F an item-feature
matrix. Representative attribute-based scores are A T Aᵀ x_S and F Fᵀ x_S / |S|.
The first pools evidence across items; the second uses feature relationships beyond
recorded pairs. Category relations need not be similarities or positive-semidefinite
kernels. Their exact construction remains domain dependent.

Fashion centers features by category; music centers them by feature clusters,
not artist labels. Subtracting the relevant group centroid yields zero group mean
before row normalization; normalization can destroy that property. Neither
operation proves independence from category or group identity. Graph-adjusted features also use historical
relations. Thus the evidence families overlap in their underlying information.

## 3. Define fusion and refinements before interpreting them

For a raw channel r_h, subtract its catalogue mean and divide by its catalogue sample
standard deviation plus 10^-9. The resulting z_h is computed separately for each
observed set. This is the Torch full-ranking convention; the NumPy basket variants
use a different variance divisor. Hidden targets are not inputs to this computation.
This dependency statement is not an audit of upstream embedding training.

With nonnegative selected weights w_h, let Y_0 = sum_h w_h z_h^0 denote the additive
core. The incidence surplus adds support from training bundles containing at least
two observed items. Only the overlap and propagated-overlap channels are modified;
let Y_1 use the separately standardized modified channels. The Rao term R combines
candidate support across pairs of observed items under specified category/descriptor
relations. Both raw refinements vanish for a single observed item.

For selected alpha in [0,1] and gamma >= 0, the full combined score is

\[
Y=(1-\alpha)Y_0+\alpha Y_1+\gamma z_R.
\]

This interpolation happens after standardization. It cannot generally be replaced
by adding alpha times the raw incidence surplus before standardizing.

The fallback P uses eligible selected category/descriptive channels, plus the selected
popularity and Rao terms. If none of its eligible channels survives selection, the
implementation uses its prescribed default fallback weights. The LaTeX document
defines this set and its weights exactly. P overlaps Y and depends partly on history.

## 4. Interpret the final mixture as a pointwise regularizer

Let s_0 >= 0 be the fallback scale, mu_0 = s_0 P, and kappa > 0 the backoff strength.
For fixed scores Y and mu_0, define a scalar objective over real t:

\[
J_c(t)=f_c(t-Y)^2+\kappa(t-\mu_0)^2.
\]

Its second derivative is 2(f_c + kappa) > 0, so its unique minimizer is

\[
\theta=\frac{f_cY+\kappa\mu_0}{f_c+\kappa}.
\]

This exactly represents the implemented combination. It is not the training
objective of the full pipeline, nor a proof that frequency is the optimal penalty
weight. At kappa = 0 the implementation explicitly returns Y; when f_c = kappa = 0,
the quadratic alone would not determine a unique score.

A Gaussian posterior interpretation requires additional assumptions. All four
archived supplied-split variance fits have negative raw slopes in the proposed
inverse-frequency regression. That does not support treating its variance model as
established, although the signs alone are not formal rejection tests. Disagreement
between Y and mu_0 is not measured ground-truth prediction error.

## 5. State consequences with their assumptions

Define rho_c = f_c/(f_c + kappa) and H = Y - P. Then exactly

\[
\theta=\rho_cH+[\rho_c+(1-\rho_c)s_0]P.
\]

At unit scale this is P + rho_c H. Otherwise shared terms also receive a
candidate-dependent coefficient. H is an algebraic remainder; identifying it with
history-only channels requires matching coefficients in Y and P.

If Y(c|S) - P(c|S) is constant across cold candidates, s_0 > 0, and all component
scores are held fixed, enabling backoff preserves the ordering among cold candidates.
The proof compares pairwise score differences: the enabled difference is a positive
multiple of the disabled difference. Cross-frequency ordering can still change.
This is an exact-arithmetic proposition; floating-point evaluation can alter
near-ties, so it does not imply bit-identical numerical rankings. Retuning weights,
auxiliary histories or differing fallback coefficients can violate the premises. This is a conditional statement, not universal cold-start robustness.

A candidate-independent mixture is a reweighting of the same fixed signal dictionary
when both component scores use that dictionary. Its improvement over a coarse greedy
weight fit need not imply additional representational power. The value of candidate
frequency must therefore be evaluated against an appropriate constant-mixture control.

## 6. Connect the argument to completed evidence

| Recall@20 | POG | POG_dense |
|---|---:|---:|
| Historical family, including popularity | 0.030875 | 0.375379 |
| Shared category/descriptive family | 0.044542 | 0.112917 |
| Combined without refinements | 0.051458 | 0.390928 |
| Full model | 0.054458 | 0.395223 |

The combined construction beats each separately fitted family in Recall and NDCG
under the current correction reserving 96 tests. This supports complementarity in
these constructions, not necessity of every signal or density as the sole explanation
for the different dataset behavior.

Frequency backoff beats the tested constant mixture on POG with matched fallback
scale; that superiority is not established with unit scale or on POG_dense.
A nonsignificant difference is not proof of equivalence. Cold-item diagnostics and
selection sensitivity, detailed in the LaTeX document, further limit universal claims.

The scientific contribution must concern the particular score construction and its
measured behavior relative to appropriate controls. Existing item-relation,
multimodal-bundle and Bayesian-backoff work precludes claiming novelty merely from
using multiple information sources or deriving a quadratic minimizer. The LaTeX
version includes the relevant references and identifies experiment provenance.

The scoring implementation and active experiment source were not modified by this
revision. The exact equations are scoped to the supplied full-ranking formulation;
retail/commerce details require their dataset-specific specification.

An independent synthetic-catalogue audit passed 52 equation checks with maximum
absolute discrepancy 4.77e-7. See [the audit record](../results/argument_audit/equations.json).
These checks do not establish statistical optimality or exhaustive correctness.
