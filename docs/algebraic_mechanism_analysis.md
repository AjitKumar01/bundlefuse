# What BundleFuse computes differently

This analysis follows the `research` full-ranking POG and POG_dense recipes and
the completed mechanism-study checkpoints. These datasets use binary, unit-weight
training bundles. Retail profiles have different channels and selection logic;
the formulas and results below should not be transferred to them without checking.
The analysis concerns the supplied split and frozen validation-selected weights,
not a proof of superiority on every dataset.

## 1. Begin with the candidate-ranking question

Let \(B\in\{0,1\}^{n\times m}\) be the training-bundle matrix, \(f_i\) the
number of training bundles containing item \(i\), \(x_S\in\{0,1\}^m\) the
indicator for a query's observed set \(S\), and \(c\notin S\) a candidate. A
baseline based on raw co-occurrence assigns

\[
r_{\rm cooc}(c\mid S)=[B^\top Bx_S]_c.
\]

It answers: how many times did the candidate occur with each seed, summed over
seeds? The implemented item-KNN applies inverse square-root frequency on both
sides of the same matrix, using \(D_{ii}=(f_i+1)^{-1/2}\):
\(r_{\rm knn}=DB^\top BDx_S\). For unobserved candidates, the diagonal in
these baseline products contributes zero. Neither score uses attributes or a
query-specific candidate-frequency mixture. This comparison describes the
implemented baselines, not every model in the wider literature.

## 2. BundleFuse represents different relationships in different operators

Define \(D_{ii}=(f_i+5)^{-1/2}\) and \(V_{bb}=(|b|+1)^{-1/2}\), using the
selected full-ranking defaults. The raw first history score is

\[
r_{\rm pair}(c\mid S)=[B^\top VBDx_S]_c.
\]

Unlike item-KNN it discounts a common **seed** but does not impose its matching
candidate-frequency factor at this stage; it also reduces the contribution of
large historical bundles. Written as a sum, it gives each historical bundle a
weight equal to its size-discounted, frequency-weighted overlap with the query,
and adds that weight to every item in the bundle. This is one possible reason
that it discriminates differently from raw counts or item-KNN. It is not by
itself a proof that the chosen normalization is optimal.

The selected `cat` signal does a different operation. Let \(G_{ia}=1\) when
item \(i\) has category \(a\), and let \(T=BG\) count category occurrences
in each training bundle. The code forms an off-diagonal nonnegative
log-association matrix from \(T^\top T\), normalizing by category occurrence
marginals. Call this matrix \(P_{\rm cat}\). For fashion, it adds a penalty
for repeating a seed's category if that category's empirical within-bundle
repetition rate is below a threshold. Hence, with \(K_{\rm cat}\) denoting the
resulting category-level relation including the penalty, the candidate score is

\[
r_{\rm cat}(c\mid S)=[G K_{\rm cat} G^\top x_S]_c.
\]

The matrix \(G K_{\rm cat}G^\top\) is a structured expansion of category
relations to all items in those categories. It is not the same as item-item
co-occurrence: an unseen item pair can share an observed category-pair score.
Conversely every item in the same category receives the same category score for
a fixed query, so category evidence alone cannot rank within that category.
The penalty does not establish that a category pair is semantically compatible.

For an item-feature matrix \(X\), the style feature matrix \(E\) is formed by
subtracting each item's category mean from its feature vector and normalizing
its row. The resulting score is proportional to

\[
r_{\rm style}(\cdot\mid S)=EE^\top x_S.
\]

The actual code divides by \(|S|\); for a nonempty fixed query that positive
factor leaves the standardized style score unchanged apart from the numerical
stabilizer. Centering changes what feature similarity means: items are compared
using their deviation from their category's average, not their raw descriptor
alone. It does not prove perfect separation of category and style.

The bridge makes the relationship between collaborative and content evidence
more explicit. Let \(L\) be the implemented diagonal-corrected normalized
incidence propagation operator, and let \(N\) be the row-normalized \(LE\).
With \(A\) the diagonal matrix normalizing the rows of \(E+\lambda N\), the
bridge vectors and their candidate score are

\[
F=A(E+\lambda N),\qquad r_{\rm bridge}=FF^\top x_S.
\]

Before expanding the outer row-normalization, the kernel contains

\[
(E+\lambda N)(E+\lambda N)^\top
=EE^\top+\lambda EN^\top+\lambda NE^\top+\lambda^2NN^\top.
\]

These cross terms allow an item's own centered descriptor to be compared with
its neighbor-smoothed counterpart, and also compare two smoothed descriptors.
They are the concrete algebraic sense in which the bridge couples feature and
historical structure. It does not follow that this coupling always helps, or
that a content-free item acquires a feature vector. Different datasets use
other bridge constructions.

## 3. Standardization turns fixed weights into query-dependent comparisons

For each channel \(h\), query \(S\), and item \(c\), let
\(z_h(c,S)=[r_h(c,S)-\mu_h(S)]/[\sigma_h(S)+\epsilon]\), where the mean and
sample standard deviation are calculated across the full item catalogue before
seed masking in the Torch full-ranking pipeline. The fitted linear core is
\(Y_c=\sum_h w_h z_h(c,S)\). For two candidates \(c,d\),

\[
Y_c-Y_d=\sum_h\frac{w_h}{\sigma_h(S)+\epsilon}
                 [r_h(c,S)-r_h(d,S)].
\]

The mean cancels within each candidate comparison, but the per-query standard
deviation does not. Thus even though the fitted \(w_h\) are global, the raw
channel scales have query-dependent effective coefficients. This can prevent a
large-scale channel from dominating merely because of its numerical units. It
can also make a low-variance channel influential; variance is not necessarily
signal quality. The formula proves what normalization does, not that it causes
the measured improvement.

## 4. The fitted backoff changes relative evidence per candidate

The model sets \(\rho_c=f_c/(f_c+\kappa)\) when \(\kappa>0\), and computes
\(\theta_c=\rho_cY_c+(1-\rho_c)sP_c\). Here \(P_c\) is constructed from
selected shared channels and, when selected, the popularity channel. It overlaps
with \(Y_c\): it is not an independent, history-free model.

For **POG**, the frozen full recipe has

\[
P_c=.5z_{\rm cat}+ .25z_{\rm style}+ .25z_{\rm bridge},\qquad
Y_c=P_c+.25z_{\rm pair},
\]

with \(\kappa=.5\), \(s\approx1.0424\), and both optional higher-order
coefficients zero. Thus exactly

\[
\theta_c=[\rho_c+(1-\rho_c)s]P_c + .25\rho_c z_{\rm pair}(c,S).
\]

The weight of the pair channel shrinks as candidate training frequency falls,
while the shared component's coefficient tends to \(s\). For a cold candidate,
\(f_c=0\) and \(\theta_c=sP_c\). This is more precise than saying 'add a
popularity score': it changes the relative weight of evidence for each candidate.
It is not calibrated uncertainty, and `bridge` itself uses the training graph.

For **POG_dense**, the frozen full recipe instead uses

\[
P_c=.5z_{\rm cat}+.25z_{\rm bridge}+.25z_{\rm pop},\quad
Y_c=P_c+.5z_{\rm simp}+.25z_{\rm csvd},
\]

where `simp` includes the incidence correction before its own standardization,
\(\kappa=.5\), and \(s\approx1.46055\). The same algebra gives
\(\theta_c=[\rho_c+(1-\rho_c)s]P_c+\rho_c(.5z_{\rm simp}+.25z_{\rm csvd})\).
Note that the fallback here **includes popularity**; it is not purely content.
The Rao coefficient is zero in both frozen full recipes.

The incidence term's overlap dependence is real but not exclusive to this
method: it expands into weighted seed-pair-to-candidate co-occurrence counts.
See [prior-art review](method_prior_art_review.md) for the derivation and related
itemset-neighborhood and HOSLIM literature. The chosen beta inside `simp` and
the outer validation-selected higher-order coefficient are distinct.

## 5. What the existing query-level results show

The independently validation-selected `history`, `shared`, and `combined`
recipes are defined by the mechanism study; their fitted weights differ. The
`shared` family includes the bridge, which itself uses historical graph
structure, so 'history' and 'shared' are labels for model families, not fully
independent sources. Recall@20 is the average fraction of held-out items found;
'any hit' means at least one held-out item appears in the top 20.

| Supplied test set | History R@20 | Shared R@20 | Combined R@20 | Full R@20 |
|---|---:|---:|---:|---:|
| POG (4,000 queries) | .030875 | .044542 | .051458 | .054458 |
| POG_dense (5,938 queries) | .375379 | .112917 | .390928 | .395223 |

The family models recover different queries. On POG, history gets a hit on 137
queries where shared does not; shared gets a hit on 242 where history does not.
The combined model gets a hit on 36 queries where **neither** standalone model
gets a hit. On POG_dense these counts are 2,022, 48, and 158, respectively.
For recall per query, combined improves versus history on 244 POG queries and
loses on 93; the corresponding dense counts are 231 and 102. Scores are paired
on the same test queries. These descriptive counts are not a causal attribution
to individual operators; a change in selected weights can change a ranking even
when the positive candidate was outside both individual top-20 lists.

The full POG improvement includes its selected frequency backoff. The full
dense improvement also includes the selected incidence correction. The available
study's matched constant versus frequency mixture favors frequency on POG but
not consistently on dense; it does not isolate every component. See the reports
and their multiple-testing-adjusted results. We must not turn these observations
into a claim that every signal is necessary or that one mechanism explains all
wins. The original and reorganized runs also have small discrepancies and must
not be mixed as though identical.

Reproduce the query counts with `python3 scripts/analyze_evidence_overlap.py`.
The script reads the saved test-query metrics; it does not fit models or change
experiments. Output: `results/algebraic_diagnostics/evidence_overlap.json`.

## 6. A specific hypothesis that matches the algebra and evidence

BundleFuse builds a candidate ranking from **overlapping but non-identical
relation operators at different resolutions**: exact item associations;
category-pair associations shared by many items; residual descriptors and
features propagated over the history graph; plus selected frequency and
multi-seed adjustments. Query-wise standardization balances their numerical
scales, and candidate-frequency backoff changes their relative influence. The
observed complementary query hits and combined-model gains are consistent with
this explanation.

That construction is more specific than 'add several features', but similarity
to earlier hybrid neighborhood, itemset and popularity-adaptive methods means
its originality must be stated at the level of the actual construction and
substantiated against close competitors. To identify which algebraic differences
cause the gains, matched comparisons would have to hold the signal menu, query
set, tuning budget, standardization and data fixed while varying one operation
at a time. Existing results cannot prove universal superiority or isolate all
of those choices.

## 7. Deeper analysis: the history correction is a quadratic query operator

The earlier operator overview did not fully explain what the multiplications can
represent. For binary unit-weight training bundles, define
\(d_s=(f_s+5)^{-1/2}\), \(v_b=(|b|+1)^{-1/2}\), and

\[
T_{cst}=\sum_b v_b B_{bc}B_{bs}B_{bt}.
\]

This is the weighted number of historical bundles containing candidate \(c\)
and seeds \(s,t\). Expanding the implemented incidence correction yields

\[
r_{\rm simp}(c\mid x)
 =\sum_s x_s d_s\sum_b v_bB_{bc}B_{bs}
 +\beta\sum_{s<t}x_sx_t(d_s+d_t)T_{cst}.
\]

The first term is linear in the binary query \(x\); the second is quadratic.
This makes precise what the code's multiplication by
\(\max(0,|b\cap S|-1)\) does: it adds pair-of-seed evidence for the candidate
when all three co-occur in a historical bundle. The sparse implementation
computes it through bundle overlaps and does not materialize the cubic tensor.
This is an exact identity, not a claim that the count is a statistical residual
above a pairwise-independence model.

For example, consider five training bundles of equal size:
\(\{a,b,c\},\{a,d,p\},\{a,d,q\},\{b,d,r\},\{b,d,t\}\), with distinct
filler items \(p,q,r,t\). Both seeds occur in three bundles. Using the actual
operator with its default \(\beta=1.5\) gives:

| Observed seeds | Pair score c | Pair score d | Pair + incidence c | Pair + incidence d |
|---|---:|---:|---:|---:|
| {a} | .1768 | .3536 | .1768 | .3536 |
| {b} | .1768 | .3536 | .1768 | .3536 |
| {a,b} | .3536 | .7071 | .8839 | .7071 |

The pair score prefers \(d\) on both seeds separately and together; the
incidence-enhanced score prefers \(c\) when both seeds are observed because
\(a,b,c\) share a bundle. For this candidate score difference the mixed finite
difference is positive (approximately .5303), whereas every fixed affine
item-to-item score \(Kx+b\) has zero mixed finite difference. This proves a
representational difference from a **single fixed affine item-item operator**.
It does not prove that a full multi-channel, query-normalized baseline or a
previous higher-order recommender cannot make the same ranking.

This tiny corpus was evaluated with the repository's own
`SimplicialGraphOperator`: run
`PYTHONPATH=. python3 scripts/check_higher_order_structure.py`.

An important detail is that fusion standardizes `pair` and `simp` separately.
If their standard deviations are \(\sigma_p,\sigma_s\), and the selected
higher-order interpolation is \(\alpha\), then for a fixed query and two
candidates the history contribution (up to its global weight) is

\[
\left[\frac{1-\alpha}{\sigma_p+\epsilon}
       +\frac{\alpha}{\sigma_s+\epsilon}\right](p_c-p_d)
+\frac{\alpha}{\sigma_s+\epsilon}(\Delta_c-\Delta_d).
\]

It is therefore not generally correct to describe the implemented score as a
single standardized \(p+\alpha\Delta\). With POG_dense's frozen
\(\alpha=1\), the standardized `simp` channel is used; with POG's
\(\alpha=0\), the standardized `pair` channel is used. HOSLIM and earlier
higher-order itemset-neighborhood models already represent itemset-to-item
relations; this algebra characterizes BundleFuse's particular count-tied,
factorized coefficients rather than establishing priority.

### Relationship to the observed dense result

The full POG_dense recipe and `frequency_matched` recipe have identical selected
channel weights, prior channels and \(\kappa=.5\). The former selects
\(\alpha=1\), the latter \(\alpha=0\); their independently computed fallback
scales are close but not exactly equal (1.4605516 versus 1.4604501). The full
model's NDCG@20 is .287876 versus .279345, a difference of .008531. The saved
conditional cluster-bootstrap contrast has Holm-adjusted \(p=.0096\); the
Recall@20 contrast is not significant after the study's correction. On 322
queries the full model has larger NDCG and on 113 it has smaller NDCG. The
one-seed queries show no difference in these saved outcomes. This is evidence
consistent with the incidence correction helping, with the small fallback-scale
difference preventing a strictly single-variable attribution.

In POG, the full recipe selects \(\alpha=0\), and the saved full and
`frequency_matched` outcomes are identical. Therefore a universal explanation
that credits the incidence interaction for every improvement is contradicted by
the fitted POG recipe.

## 8. A subtle interaction: centering becomes a frequency-dependent offset

For POG write the unstandardized shared and pair channel contributions as

\[
H_c(S)=\sum_{h\in\{cat,style,bridge\}}
       \frac{w_h r_h(c,S)}{\sigma_h(S)+\epsilon},\quad
U_c(S)=\frac{.25\,r_{pair}(c,S)}{\sigma_{pair}(S)+\epsilon},
\]

and define the query-only constants
\(A(S)=\sum_h w_h\mu_h(S)/(\sigma_h(S)+\epsilon)\) and
\(J(S)=.25\mu_{pair}(S)/(\sigma_{pair}(S)+\epsilon)\). Since
\(P_c=H_c-A\), the previously derived POG formula becomes

\[
\theta_c=q_c H_c+\rho_c U_c-q_c A-\rho_c J,\qquad
q_c=s+(1-s)\rho_c.
\]

Equivalently, apart from the query-only constant \(-sA\), it contains the
candidate-frequency term

\[
-\rho_c\big[(1-s)A+J\big].
\]

Without candidate-dependent mixing, a channel's subtracted catalogue mean
cancels when ranking two candidates. With this mixing, it is multiplied by
candidate-specific \(q_c\) or \(\rho_c\) and **does not generally cancel**.
Thus standardization and backoff together do more than rescale relations: they
induce a query-dependent frequency offset. Its sign and magnitude depend on the
query and all participating channel means; it is not an unconditional boost
for rare or cold items.

A concrete special case illustrates the pair channel's contribution. Suppose a
query has a positive catalogue mean pair score, and both a cold candidate and
a warm candidate have raw pair score zero. Both get the same negative
standardized pair score. Their contributions after POG's backoff are zero for
the cold candidate and negative for the warm candidate, because \(\rho=0\)
for the former and \(\rho>0\) for the latter. Other channels can dominate or
reverse the total ranking. This does not prove that this offset caused a net
metric improvement; it identifies an effect that is invisible when examining
normalization and backoff separately.

The conclusions in Sections 7–8 are algebraic properties of the implementation;
they are not claims that these operations are previously unknown, or that
increased expressiveness guarantees better test rankings.
