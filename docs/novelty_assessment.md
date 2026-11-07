# BundleFuse novelty assessment: raw scores versus fusion

## 1. Assessment

BundleFuse has strong recorded benchmark results and several mathematical properties that explain what its scores retain. These establish empirical value and identifiable mechanisms. They do not yet establish that a standalone raw score is new, or that the fusion is a new method superior to strong alternative combiners receiving the same scores.

The most concrete raw-score candidate is the **count-constrained higher-order incidence estimator**. Its candidate–seed–seed coefficients are computed from observed bundle membership rather than fitted independently. The most concrete fusion question is whether the **candidate-frequency coefficient constraints** improve held-out ranking compared with ordinary score and rank fusion.

These are two different research routes:

| Route | What would support the contribution | What is insufficient |
|---|---|---|
| Raw-score construction | A precise distinction from close predecessors and a benefit when fusion, inputs, and evaluation are held fixed | Different constants, a new name, or a win by the complete system |
| Fusion construction | A precise distinction from existing adaptive mixtures and a benefit with the input score columns held fixed | Beating a single source, or changing raw scores and fusion simultaneously |

BundleFuse does not perform bootstrap aggregation or sequential residual fitting. Nevertheless, its final stage combines scores. Explaining that it is neither bagging nor boosting does not resolve the substantive concern: whether standard fusion of the same scores would do as well.

This report audits existing equations, saved results, and prior work. Sections 9–10 record the completed attribution experiments. Section 12 develops the score-sufficiency argument, including proofs, a counterexample evaluated with the production operators, and the limits of the resulting claims.

**Experiment follow-up:** the POG_DENSE, Spotify and POG attribution stage and the five-dataset amplified-neighbor extension are complete on `novelty-score-fusion-experiments`. See Sections 9–10 and the [run protocol and logs](../experiments/novelty/README.md). HOSLIM-style fitting, separate bridge controls and untouched-split confirmation are not claimed to be covered by these stages.

### Domain and task distinctions in the novelty assessment

The target task is bundle completion: given an observed item set S, rank candidates c outside S by their suitability for the missing portion. This must be distinguished from recommending a pre-existing bundle to a user, retrieving an item similar to one item, and learning word representations. Evidence against novelty of a mathematical operator does not, by itself, establish that the complete bundle-completion estimator has been published.

Each predecessor should therefore be assessed separately for (i) its mathematical operation, (ii) its domain and input information, and (iii) its prediction task and final scoring rule. A method from another domain can establish that an operator is known while leaving a task-specific contribution unresolved. Conversely, transferring a known operator to bundles does not by itself establish a substantial methodological contribution. The relevant question is what the formulation changes to address bundle completion and whether that change explains a measurable benefit.

For the PPMI-SVD source, the evidence extends beyond language modelling. Afchar, Hennequin and Guigue (2023), *Of Spiky SVDs and Music Recommendation*, Section 2.2, construct item co-occurrence matrices, retain top-k interactions, apply PPMI, and use truncated SVD embeddings U Sigma with dimension 128. Their datasets include Spotify Million Playlist, Art of the Mix and Cornell playlists. Their focus is embedding geometry and stability of similar-item retrieval, rather than reproducing BundleFuse's full partial-bundle ranking rule. This is a close music/playlist-domain predecessor for the representation, not an established predecessor for the entire estimator. No exact POG fashion-bundle PPMI-SVD completion predecessor is established by this evidence.

Primary sources: [Afchar et al., author-hosted paper, Section 2.2](https://vguigue.github.io/art/recsys23.pdf), [RaMen: bundle construction](https://arxiv.org/abs/2507.14361), and [Vagliano et al.: item-set completion and co-occurrence semantics](https://link.springer.com/article/10.1007/s10791-022-09408-9).

## 2. Notation and the estimator being assessed

Let the catalogue contain \(n\) items and the training collection contain \(m\) bundles. Define the binary incidence matrix

\[
B\in\{0,1\}^{m\times n},\qquad
B_{bi}=1\iff\text{training bundle }b\text{ contains item }i.
\]

Let \(S\) be the observed query set, \(c\notin S\) a candidate, \(f_i=\sum_bB_{bi}\) the training frequency of item \(i\), and \(d_b=\sum_iB_{bi}\) the size of bundle \(b\). In the examined binary, unit-weight formulation,

\[
a_i=(f_i+5)^{-1/2},\qquad v_b=(d_b+1)^{-1/2}.
\]

A raw source score \(r_h(c\mid S)\), indexed by \(h\), is standardized separately for each query over catalogue items:

\[
z_h(c\mid S)=\frac{r_h(c\mid S)-\mu_h(S)}{\sigma_h(S)+\epsilon},
\]

where \(\mu_h\) and \(\sigma_h\) are its catalogue mean and sample standard deviation in the full-ranking pipeline, and \(\epsilon=10^{-9}\). Different raw sources are standardized separately.

The full and restricted scores are weighted sums of these standardized sources. The final interpolation uses candidate frequency. These definitions describe the POG/Spotify analysis; weighted retail profiles require their own formulas.

## 3. Route one: is a raw score a distinct contribution?

### 3.1 Direct historical score: a close neighbor-method variant

The direct historical score is

\[
p_c(S)=\sum_bB_{bc}v_b\sum_{s\in S}B_{bs}a_s.
\]

It retrieves training bundles using frequency-weighted query overlap, discounts bundle size, and transfers each bundle's match score to its member candidates.

Earlier playlist-neighbor systems already score a candidate by summing similarities to historical playlists containing it. Vall et al. describe an all-playlist cosine-neighbor score; Kelen et al. use inverse item frequency and similarity amplification. Our normalization and weighting differ, but historical-set matching and rarity weighting are established principles.

**Position:** the exact weighting can be evaluated as an estimator choice. The direct score alone does not currently support a strong priority claim.

Sources: [Vall et al., Section 5.3](https://link.springer.com/article/10.1007/s11257-018-9215-8); [Kelen et al.](https://eprints.sztaki.hu/9560/1/Kelen_1_30347064_ny.pdf).

### 3.2 Incidence enhancement: the strongest candidate to investigate

Let \(k_b(S)=\sum_{s\in S}B_{bs}\) count the query items present in training bundle \(b\), and let \([u]_+=\max(u,0)\). The enhanced score is

\[
g_c(S)=\sum_bB_{bc}v_b
\left(\sum_{s\in S}B_{bs}a_s\right)
\left(1+\beta[k_b(S)-1]_+\right),
\]

with \(\beta=1.5\) in the examined implementation. Define the weighted triple count

\[
T_{cst}=\sum_bv_bB_{bc}B_{bs}B_{bt}.
\]

For distinct candidate and seed IDs, expansion gives the exact identity

\[
\boxed{g_c(S)=p_c(S)+\beta\sum_{\{s,t\}\subseteq S}(a_s+a_t)T_{cst}.}
\]

Thus the correction is a seed-pair-to-candidate score. Its coefficients are constrained by observed triple counts and seed frequencies. The incidence computation evaluates it without constructing a catalogue-sized cubic tensor of coefficients.

The existing two-corpus diagnostic constructs different bundle collections with identical complete item-pair matrices but different triple counts and different incidence scores. It proves that the score cannot generally be recovered from item-pair statistics alone. It does **not** prove that triple occurrences are unexplained by every pairwise probabilistic model, or that no earlier higher-order method can represent the score.

HOKNN and HOSLIM already use itemset-to-item relations. HOSLIM learns item–item and itemset–item aggregation coefficients. Overlap amplification also has a long history. Therefore the defensible object of investigation is the **particular count-constrained estimator and its computation**, not the invention of higher-order recommendation.

The required comparison is against properly tuned amplified historical-set neighbors and itemset estimators, using matched inputs and the same fusion. Accuracy, fit time, memory, and inference cost should be measured. Avoid asserting an efficiency advantage before measuring the implementations.

Sources: [HOSLIM institutional record](https://experts.umn.edu/en/publications/hoslim-higher-order-sparse-linear-method-for-top-n-recommender-sy/); [author-uploaded paper](https://www.researchgate.net/publication/271908941_HOSLIM_Higher-Order_Sparse_LInear_Method_for_Top-N_Recommender_Systems).

### 3.3 Other raw scores

| Source | Mathematical role | Novelty position |
|---|---|---|
| PPMI | Positive log association relative to item-pair endpoint marginals | Established association construction |
| SVD source | Cosine similarity of compressed PPMI association profiles | Established representation principle; not simply direct PPMI reconstruction |
| Category score | Pool occurrences by category, then assign category-relation scores to candidates | Pooling is established; exact normalization and repetition penalty are design choices to assess |
| Category-residual descriptions | Remove category means before normalizing description vectors | Mean subtraction is a least-squares residualization operation |
| Bridge | Combine own residual descriptions with normalized historical-partner descriptions before similarity scoring | A concrete coupled kernel; exact priority and individual benefit need matched comparison |
| Agreement score \(z_R\) | Candidate affinities on both sides of a seed-pair kernel, divided by total support plus an offset | Related quadratic forms predate this construction; exact bundle-completion formula priority is unresolved |

For example, with residual description matrix \(E\), normalized aggregated-partner descriptions \(N\), bridge strength \(\eta\), and final row-normalization diagonal \(A\), the bridge kernel is

\[
FF^\top=A\left(EE^\top+\eta EN^\top+\eta NE^\top+\eta^2NN^\top\right)A.
\]

This shows which own/partner comparisons the bridge adds. It does not show that adding those comparisons is new or beneficial on every dataset.

For agreement, let \(q_{cs}\) be candidate–seed affinity, \(A_c=\sum_s q_{cs}\), \(K_S\) the seed-pair kernel, and \(\tau>0\) the support offset. Then

\[
r_R(c\mid S)=\frac{q_c^\top K_Sq_c}{A_c+\tau}.
\]

The numerator is closely related to weighted neighbor-endorsement and Rao-type quadratic constructions. University material explicitly describes a score \(\sum_{s,t}a_{sc}a_{tc}d_{st}\). That source establishes a close mathematical formulation, not its earliest publication or an identical recommendation method. The music configuration rewards seed similarity; the POG configuration discounts it. A single diversity interpretation is not valid for both.

Source: [Franceschet, similarity and heterogeneity](https://users.dimi.uniud.it/~massimo.franceschet/ns/plugandplay/similarity/similarity.html).

### PPMI followed by SVD: an explicit equation-level predecessor

Levy, Goldberg and Dagan (2015), *Improving Distributional Similarity with Lessons Learned from Word Embeddings*, Section 2.2 and Equation 1, explicitly factorize a PPMI matrix and use rows of U_r Sigma_r as dense representations. Their notation section also specifies unit-length normalization before similarity calculations. Therefore the combination PPMI, truncated SVD, retained singular-value weighting, and cosine similarity already has a direct precedent; the connection is more precise than generic LSA on a count matrix.

Using their conventional SVD notation for the PPMI matrix M,

\[
M=U\Sigma V^\top,\qquad Z=MV_r=U_r\Sigma_r,\qquad
\widehat Z_i=Z_i/(\|Z_i\|_2+\epsilon).
\]

BundleFuse scores a candidate against the mean of the observed item vectors:

\[
r_{svd}(c\mid S)=\widehat Z_c^\top\frac1{|S|}\sum_{s\in S}\widehat Z_s
=\frac1{|S|}\sum_{s\in S}\widehat Z_c^\top\widehat Z_s.
\]

Apart from the numerical norm stabilizer, the last expression is an average of cosine similarities. Thus the use of a query set adds an aggregation of established similarities, not a new PPMI-SVD representation rule.

The prior paper uses words and their contexts; BundleFuse uses items and their within-bundle co-occurrence partners. Their counts, tasks, and complete systems are not identical. Nonetheless the mathematical representation and similarity operations have already been described. Applying this operator to bundle completion can be useful without being a standalone algorithmic novelty. The distinction between the profile Gram matrix U_r Sigma_r^2 U_r^T and a direct reconstruction U_r Sigma_r V_r^T is real, but it does not create novelty: the earlier construction explicitly uses U_r Sigma_r rows too.

Primary source: [Levy, Goldberg and Dagan, TACL 2015](https://aclanthology.org/Q15-1016.pdf). Earlier PMI/SVD investigations include [Bullinaria and Levy, 2012](https://johnbullinaria.github.io/PUBS/BRM2.pdf); no earliest-priority claim is needed for the conclusion above.

### 3.4 The L matrix and propagated score: a particularly close diffusion precedent

The POG_DENSE document defines

\[
D=\operatorname{diag}((f_i+5)^{-1/2}),\quad
E=\operatorname{diag}((d_b+1)^{-1}),\quad
L=\operatorname{off}(DB^\top EBD).
\]

For distinct items c and i,

\[
L_{ci}=\frac{1}{\sqrt{f_c+5}\sqrt{f_i+5}}
\sum_b\frac{B_{bc}B_{bi}}{d_b+1},\qquad L_{cc}=0.
\]

Zeng et al. (2013), Equation 1, explicitly state the established hybrid mass-diffusion/heat-conduction operator. Translating their user–item incidence into our bundle–item notation, it is

\[
W^{(\lambda)}_{ci}=\frac{1}{f_c^{1-\lambda}f_i^\lambda}
\sum_b\frac{B_{bc}B_{bi}}{d_b}.
\]

On positive-degree items, setting lambda to one half gives the same unsmoothed symmetric normalization. BundleFuse adds 5 to item frequencies, adds 1 to bundle sizes, and removes the diagonal. Thus a formula-level recommendation predecessor exists, not merely an analogy to graph learning. The 2013 paper attributes that hybrid operator to earlier work; it should not be cited as the origin of the entire diffusion family.

The same structural form occurs in Zhou, Huang and Schölkopf (2006), whose normalized hypergraph operator is Theta = D_v^(-1/2) H W_e D_e^(-1) H^T D_v^(-1/2). Setting item–bundle incidence H = B^T and unit edge weights recovers the unsmoothed form before diagonal removal. Our symbol L denotes a normalized adjacency/propagation operator, not the usual Laplacian I - Theta. Diagonal removal means standard spectral/positive-semidefinite properties cannot simply be imported.

The document's raw propagated sources are

\[
r_{\mathrm{propagated}}^0=Lp,\qquad
r_{\mathrm{propagated}}=L(p+\delta).
\]

Propagating an existing item-score vector is also established: Zeng et al. use repeated diffusion W^n f. Our initial p and incidence correction delta are specific constructions; this is not an assertion that their complete algorithm equals ours. In particular, p = B^T diag((d_b+1)^(-1/2)) B D x_S, so Lp is not literally L^2 x_S with the present normalizations. L is query-independent and uses weighted pair projections; multiplying a pairwise-linear p by L remains linear in x_S. Higher-order query dependence in L delta comes from delta, not from L alone.

The selected POG_DENSE full recipe assigns no standalone weight to the propagated/diff2 source. It does use L indirectly through the selected bridge. Hence standalone propagation cannot explain the reported dense model's gain; its effect within the bridge needs a bridge-specific comparison.

Sources: [Zeng et al., Information filtering in sparse online systems: recommendation via semi-local diffusion, Equations 1 and repeated propagation](https://arxiv.org/pdf/1308.3060); [Zhou, Huang and Schölkopf, Learning with Hypergraphs, Section 5](https://papers.nips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf).

### 3.5 Selection limits which operator can explain a win

| Canonical full model | Incidence enhancement selected? | Agreement selected? |
|---|---|---|
| POG | No | No |
| POG_DENSE | Yes | No |
| Spotify | Yes | Yes |

A score cannot explain a result when its selected coefficient is zero. In particular, incidence and agreement cannot be the explanation for the displayed POG win. A whole-method claim must account for the simpler selected recipes as well as the richer ones.

## 4. Route two: is the fusion a new contribution?

### 4.1 Exact fixed-recipe representation

Collect every relevant standardized score into a vector \(z(c,S)\in\mathbb R^H\), where \(H\) is the number of score columns. Include pairwise, incidence-enhanced, and agreement columns separately when needed. For fixed selected coefficient vectors \(a,b\in\mathbb R^H\), write

\[
Y_c=a^\top z(c,S),\qquad P_c=b^\top z(c,S).
\]

Here \(Y_c\) uses all selected sources and \(P_c\) uses a specified subset with overlap. If agreement is selected, the current implementation includes it in both scores. The final score for positive \(\kappa\) is

\[
\theta_c=\rho_cY_c+(1-\rho_c)s_0P_c,
\qquad \rho_c=\frac{f_c}{f_c+\kappa}.
\]

Rearranging gives

\[
\boxed{\theta_c=s_0b^\top z(c,S)+\rho_c(a-s_0b)^\top z(c,S).}
\]

For fixed \(\kappa,s_0,a,b\), this is linear in

\[
\phi(c,S)=\begin{bmatrix}z(c,S)\\\rho_cz(c,S)\end{bmatrix},
\]

with a prescribed relationship between its coefficients. The candidate-frequency interaction makes it more flexible than a constant-weight sum over the original score columns. It does not create a prediction-function family unavailable to a ranker given these expanded features.

This observation does not rule out a useful algorithm. Coefficient restrictions may improve generalization, reduce the number of fitted choices, or make behavior easier to explain. Those are empirical claims about a particular constraint, not consequences of the algebra alone.

### 4.2 The optimization interpretation is local to the gate

For fixed inputs, the final score uniquely minimizes

\[
\frac{f_c}{2}(t-Y_c)^2+\frac{\kappa}{2}(t-s_0P_c)^2.
\]

This is an exact explanation of the interpolation. It is a standard quadratic compromise between two scores, not a joint derivation of the full pipeline or a proof of optimal ranking.

### 4.3 Close adaptive-fusion precedents

CoHeat mixes history and affiliation scores with a coefficient depending on bundle popularity. Its task is user-to-bundle recommendation; its score definitions, coefficient, learned representations and training differ from BundleFuse. Nevertheless, trusting different sources according to observed frequency is already an explicit recommendation principle.

Hu et al. use item-specific confidence to combine behavioral and content representations. This operates at representation level with a learned gate, unlike our prescribed score interpolation. It is another close precedent for the underlying motivation.

Sources: [CoHeat, WWW 2024](https://jeon185.github.io/data/WWW24%28CoHeat%29.pdf); [Hybrid Item-Item Recommendation via Semi-Parametric Embedding, IJCAI 2019](https://www.ijcai.org/proceedings/2019/0350.pdf).

**Position:** the particular overlapping fallback and coefficient constraint can be investigated as an estimator design. A new general fusion principle has not been established.

## 5. What the completed experiments establish

### 5.1 Recorded NDCG@20

| Dataset | Combined, no fallback | Matched-scale constant mixture | Matched-scale frequency mixture | Full BundleFuse |
|---|---:|---:|---:|---:|
| POG | 0.033940 | 0.034056 | 0.036092 | 0.036092 |
| POG_DENSE | 0.276894 | 0.279729 | 0.279345 | 0.287876 |
| Spotify | 0.243196 | 0.243196 | 0.253632 | 0.257926 |

The constant and frequency controls hold the combined recipe's base weights, pairwise historical form, disabled agreement, and fallback source list fixed. They each select from six candidates. Within each dataset, the two matched-scale controls use the same scale. These controls do not optimize arbitrary per-source weights anew.

### 5.2 Supported conclusions

- On POG and Spotify, the matched-scale frequency control improves both Recall@20 and NDCG@20 over the matched-scale constant control. Their recorded dataset-summary Holm-adjusted p-values are approximately 0.0096, reserving the planned 96-test family.
- On POG_DENSE, that frequency control is numerically below the constant control on both metrics. The frequency mixture does not have a universal win in these studies.
- Full POG equals the frequency control in per-query metrics. The full model supplies no further improvement over that selected mixture there.
- Full POG_DENSE improves NDCG over its frequency control by about 0.008531; the recorded adjusted p-value is approximately 0.0096. Its Recall difference does not pass the adjustment, with p approximately 0.27057.
- Full Spotify improves both metrics over its frequency control, with recorded adjusted p-values approximately 0.0096.

The full/control comparisons on POG_DENSE and Spotify also change incidence/agreement choices and can change scale. They do not isolate those operations. The dense full/control scales are close but not identical, so even that contrast is not a perfectly isolated incidence change.

The dataset summaries reserve the planned family conservatively. The four-dataset queue was stopped before the sparse Spotify study completed; no completed pooled four-dataset report is claimed here.

### 5.3 Conclusions not yet established

The current mechanism study does not compare BundleFuse fusion against RRF, a fully retuned same-score linear combiner, or LambdaMART on exactly the same score columns and candidate-frequency features. Earlier information-matched hybrid and learned-ranker studies are useful evidence but are not interchangeable with that comparison.

Beating published CLHE and RaMen benchmark values establishes a reported performance comparison under its stated scope. It does not determine whether our raw scores, fusion, or their interaction explains the difference.

## 6. Experiments that separate the two contribution routes

### 6.1 Factorial design

| Raw-score construction | Standard fusion | BundleFuse fusion |
|---|---|---|
| Matched close-predecessor scores | A | B |
| BundleFuse scores | C | D |

- **C versus A:** change score construction while holding fusion fixed.
- **D versus C:** change fusion while holding score columns fixed.
- **B versus A:** test whether the fusion helps conventional scores too.
- The difference between these effects tests whether the particular scores and fusion work especially well together.

“Standard fusion” should be evaluated with several concrete alternatives, not represented by an intentionally weak single baseline.

### 6.2 Comparators for fusion

| Comparator | Question answered |
|---|---|
| Equal-weight sum of standardized scores | Does careful weighting beat a basic scale-matched combination? |
| Retuned nonnegative weighted sum | Does frequency adaptation add value beyond flexible global weights? |
| Reciprocal Rank Fusion | Does using score magnitudes and adaptation help beyond established rank aggregation? |
| Linear ranker on \([z;\rho z]\) | Are our coefficient constraints useful compared with fitting these interactions freely? |
| LambdaMART on the same scores and frequency | Can a standard boosted ranker learn a better nonlinear combination? |

RRF uses \(\sum_h1/(k+\operatorname{rank}_h(c))\), where \(k\) is an offset. LambdaMART learns ranking functions from query-grouped feature rows and relevance labels. These comparison methods can have trained parameters while BundleFuse retains its own zero-gradient-trained-scoring claim.

Sources: [RRF original paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf); [XGBoost official learning-to-rank documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html).

The linear expanded-feature comparator should receive the same frequency transform for a controlled test of coefficient tying. If multiple transforms or values of \(\kappa\) are searched, that search must be included in the tuning budget.

### 6.3 Comparators for raw scores

First evaluate pair-only, the proposed incidence estimator, and an appropriately tuned amplified historical-set neighbor while fixing the other score sources and fusion. This tests whether the specific joint-occurrence weighting improves the ranking.

Compare itemset-to-item methods such as HOSLIM/HOKNN at explicit fitting and memory budgets. For the bridge, compare own-description similarity and corresponding graph-aggregated feature kernels under the same fusion. These comparisons must use the same available inputs; extra modalities cannot be credited to a new scoring equation.

A useful outcome is not only a mean gain: identify where multiple simultaneous seed matches exist and whether the estimator's gain occurs there. The incidence correction is zero when no training row contains two observed query items. That gives a mechanism-specific diagnostic rather than an unrestricted explanation after seeing results.

### 6.4 Evaluation controls

1. Keep the candidate catalogue, query masks, metrics, seed exclusion and tie handling identical.
2. Give fusion comparators the same score columns, including optional interactions, and disclose frequency inputs.
3. Match the available fitting labels and record actual tuning budgets; compare training-free and trained combiners explicitly.
4. Use query-level fitting/tuning separation. Existing validation queries can support cross-validation; alternatively construct cross-fitted training queries.
5. Do not generate a training query whose withheld answers remain present in its own row used to construct the count operators. Group related queries from the same complete bundle when splitting folds.
6. Evaluate all selected systems on the full catalogue. If training uses negative sampling, disclose it and keep test ranking complete.
7. Report paired query-level differences with the prespecified correction, alongside parameter count, memory, fit time and latency.
8. Since the existing test results have guided this research, describe supplied-split follow-ups as exploratory and seek an untouched split or domain for final confirmation.

## 7. How the outcomes determine the claim

| Outcome | Contribution supported |
|---|---|
| BundleFuse scores improve multiple fixed fusion methods; its fusion adds little | Score construction and the constrained estimator are the main result |
| BundleFuse fusion improves identical score inputs, including predecessor scores | An effective constrained adaptive-fusion design, subject to prior-art distinction |
| Both effects are positive and repeat on untouched evaluation | A supported interaction between score construction and fusion |
| A standard combiner matches accuracy with the same scores | No demonstrated fusion-accuracy advantage; assess simplicity, supervision and cost separately |
| Close predecessor scores match under the same fusion | Reassess the standalone raw-score claim; differences in constants are insufficient |

There is no requirement to outperform every conceivable fusion algorithm to have a contribution. No finite set of experiments can establish that universal statement. The target is a precise distinction from the closest prior methods, supported by strong relevant comparisons.

## 8. Recommended priority and present claim

Resolve the incidence estimator against amplified neighbors first, and run the same-score fusion comparison alongside it. These are direct tests of the two proposed routes; they avoid another expensive comparison of entire architectures whose differences cannot be attributed.

The claim currently supported is:

> BundleFuse is an explicit bundle-completion estimator combining recorded co-membership, selected joint-occurrence information, association-profile and attribute scores with candidate-frequency interpolation. Its recorded benchmark results and controlled mixture studies establish empirical value. The count-constrained higher-order estimator is the clearest standalone mechanism to investigate; an independently superior fusion method remains to be demonstrated.

The unselected incidence term on POG and unselected agreement term on POG/POG_DENSE are important boundaries. They prevent attributing all benchmark gains to one mechanism simply because its algebra is interesting.

## 9. First attribution results: POG_DENSE complete

POG_DENSE completed the first-stage study on all 5,938 supplied test queries. Spotify and POG subsequently completed too (Section 10).

| Controlled method | Recall@20 | NDCG@20 |
|---|---:|---:|
| Pairwise + retuned weighted sum | 0.388599 | 0.277964 |
| Incidence + retuned weighted sum | 0.391378 | 0.284989 |
| Amplified neighbors + retuned weighted sum | 0.388851 | 0.287204 |
| Incidence + refitted frequency gate | 0.391911 | 0.285098 |
| Amplified neighbors + refitted frequency gate | 0.390985 | 0.288159 |
| Original frozen BundleFuse reference | 0.395223 | 0.287876 |

The original reference has its earlier all-validation selection provenance. Controlled methods fit on the same 70% validation groups and tune on the same remaining 30%. Comparing the original reference with those refits is descriptive rather than a matched claim about fusion.

**Raw-score route:** incidence versus pairwise with the weighted sum improves NDCG by 0.007025 (recorded Holm-adjusted p approximately 0.01440), while its Recall difference is not significant after adjustment. With the gate, incidence improves NDCG over pairwise by 0.003991 (adjusted p approximately 0.04600), with no supported Recall gain. However, incidence is numerically below amplification in NDCG under both weighted fusion and the gate; neither direction passes the planned adjustment in those two incidence/amplification contrasts. The amplified raw operator selected exponent 2 and all historical bundles. These findings support nonlinear overlap as useful relative to the pairwise alternative, but do not establish that the particular incidence estimator improves on amplified neighbors.

**Fusion route:** incidence + gate differs from incidence + weighted sum by only +0.000533 Recall and +0.000109 NDCG; both adjusted p-values are 1. The amplified gate also has no significant advantage over the amplified weighted sum under the planned adjustment. Thus a general frequency-fusion advantage is not established on this dataset in the new study.

The tested linear interaction ranker, LambdaMART and RRF are weaker in this run. The best tested LambdaMART bank reaches approximately 0.290053 Recall and 0.244969 NDCG; the best tested frequency-interaction linear bank reaches approximately 0.350567 and 0.265478. These outcomes concern the recorded sampling scheme, validation partition and limited grids, not every possible training procedure for these algorithm classes.

The amplified gate slightly exceeds the original reference's NDCG (0.288159 versus 0.287876), whereas the original reference has higher Recall (0.395223 versus 0.390985). This descriptive comparison gives no strict dominance claim for either system.

Consequently, this first completed dataset does **not** justify upgrading the novelty claim. A conventional amplified historical-neighbor operator plus simple score fusion is a serious comparator. Further interpretation awaits the remaining datasets and, if pursued, the separate itemset/bridge and untouched-split stages.

Evidence: [first-stage report](../results/novelty_attribution/pog_dense/REPORT.md), [paired statistics](../results/novelty_attribution/pog_dense/summary.json), and [amplified-neighbor selection](../results/novelty_attribution/pog_dense/amplified_selection.json). Original-reference score parity has maximum absolute error approximately \(3.81\times10^{-6}\).

## 10. Spotify and POG attribution results complete

Spotify completed all 19 methods on all 4,000 supplied test queries, exiting successfully at 12:48 UTC on 6 November 2026. POG subsequently completed all 19 methods on all 4,000 supplied test queries at 12:56 UTC. The same frozen source inventory and fit/tuning partition qualifications apply as in Section 9. These are fused score-bank comparisons, not scores from isolated historical operators.

| Controlled method | Recall@20 | NDCG@20 |
|---|---:|---:|
| Pairwise bank + retuned weighted sum | 0.156566 | 0.264012 |
| Incidence bank + retuned weighted sum | 0.156566 | 0.264012 |
| Amplified-neighbor bank + retuned weighted sum | 0.160019 | 0.271723 |
| Incidence bank + refitted frequency gate | 0.156566 | 0.264012 |
| Amplified-neighbor bank + refitted frequency gate | 0.163278 | 0.281146 |
| Original frozen BundleFuse reference | 0.155646 | 0.257926 |

The amplified operator selected exponent 2 and the 100 most similar historical bundles. With its other scores retained, adding the gate and its selected fallback scale improves Recall by 0.003260 and NDCG by 0.009424 versus the amplified weighted sum. Both contrasts have recorded Holm-adjusted p approximately 0.01440. The gated amplified bank also exceeds the gated incidence bank by 0.006713 Recall and 0.017134 NDCG, both adjusted p approximately 0.01440. This supports the tested interpolation procedure on the amplified Spotify bank, not the uniqueness of its rational frequency law relative to every alternative adaptive or scale-only formulation.

The pairwise and incidence weighted models both select only `homo` with weight 0.25 and `rao` with weight 1.0; their explicit historical bank column receives zero weight. Their gate selects kappa zero. Their equality therefore does not show that the raw incidence and pairwise arrays are identical, nor that joint occurrence is intrinsically useless. It shows that this selection procedure found no incremental test benefit from retaining either explicit historical column over those common sources. The retained `rao` source itself includes pairwise historical affinities.

The amplified weighted model selects `homo`, `rao` and the amplified historical score, each with weight 0.25. Its gate selects kappa 0.05 and fallback scale approximately 1.42555. Its improvement over the earlier original reference is approximately 4.90% in Recall and 9.00% in NDCG, descriptively; the reference used a different all-validation selection procedure and this supplied test split has already informed the research. No confirmatory superiority claim follows from that reference comparison alone.

The tested frequency-interaction linear and LambdaMART models are substantially weaker on Spotify under the recorded sampled training and grids. These results should not be generalized to all learned fusion procedures. In particular, the large gap warrants checking training and full-catalogue generalization before using those rows as a central publication argument.

**POG:** the amplified weighted sum gives 0.051042 Recall and 0.033425 NDCG. Its frequency fallback improves these to 0.053542 and 0.035597; both differences have recorded Holm-adjusted p approximately 0.01440. The frozen original reference remains numerically higher at 0.054458 and 0.036092. Incidence with its matched gate gives 0.054458 and 0.036043; its advantage over the amplified gated bank is not significant under the planned correction. Amplification is therefore not a universal upgrade. See the [POG report](../results/novelty_attribution/pog/REPORT.md).

**Current conclusion across the three completed attribution datasets:** nonlinear historical overlap is useful, but a standalone advantage for the specific incidence construction over amplified neighbors is not established. The interpolation procedure significantly improves the amplified Spotify and POG banks, while its gain on the incidence bank is absent on Spotify and unsupported on POG_DENSE. The best new Spotify configuration combines amplified-neighbor history, shared BundleFuse sources and the frequency fallback; it does not consistently dominate the existing construction on all datasets. Untouched confirmation remains outstanding; a universal raw-score or fusion novelty claim is not supported.

**Cross-dataset extension complete:** at the user's request, a separately frozen [five-dataset follow-up](../experiments/amplified_fallback/README.md) tested electronic, food, clothing, dunnhumby and Spotify_sparse; all five completed successfully on 6 November 2026. It adds constant-mixture controls and preserves each input checkpoint's split and source profile. The [eight-dataset table](../results/amplified_fallback/README.md) records results and the selected amplified historical weight; a zero weight prevents attributing a row's improvement to amplification. This follow-up does not overwrite the first study or resume the interrupted sparse mechanism queue.

Among those five new datasets, no amplified gated bank significantly improves on the matched reference-history gated bank under the reserved 60-test correction. The frequency gate improves food NDCG relative to both no fallback and a tuned constant mixture (adjusted p approximately .012). It improves both metrics on clothing relative to both controls (adjusted p approximately .006), but both clothing banks select zero weight for their explicit historical column and produce identical gated results. Spotify_sparse also discards that column, and selects kappa zero; its improved original-reference comparison cannot be attributed to amplified history or frequency interpolation. Electronic and dunnhumby have no significant planned contrasts. The cross-dataset evidence therefore strengthens the case for a useful frequency-dependent fallback on particular datasets, rather than for amplified neighbors as a universal upgrade. These are exploratory fixed-split results, conditional on the prior source inventories and recorded dataset profiles.

Evidence: [Spotify report](../results/novelty_attribution/spotify/REPORT.md), [paired statistics](../results/novelty_attribution/spotify/summary.json), [selected recipes](../results/novelty_attribution/spotify/selection/amplified/selected.json), and [neighbor selection](../results/novelty_attribution/spotify/amplified_selection.json). The original reference was reproduced with maximum absolute score error approximately 7.63e-6.

## 11. Local evidence and reproducibility

- [Full method prior-art review](method_prior_art_review.md).
- [Algebraic analysis](algebraic_mechanism_analysis.md) and [evidence index](analysis_evidence_index.md).
- [POG summary](../results/publication_mechanisms/pog/summary.json).
- [POG_DENSE summary](../results/publication_mechanisms/pog_dense/summary.json).
- [Spotify summary](../results/publication_mechanisms/spotify/summary.json).
- [Mechanism experiment protocol and limitations](../experiments/mechanisms/README.md).
- [Recipe selection code](../experiments/mechanisms/run.py) and [fixed-mixture controls](../experiments/mechanisms/selection.py).
- [Fusion implementation](../bundlefuse/fusion.py) and [raw operators](../bundlefuse/operators.py).

This report preserves the distinction between mathematical identities, prior-art comparisons, completed measurements, and proposed experiments. A proposed experiment is complete only where an accompanying result is explicitly recorded.

## 12. When are the retained scores sufficient?

The [standalone document (PDF)](../output/pdf/score_sufficiency.pdf), with
[TeX source](score_sufficiency.tex), develops this analysis independently and
includes mathematical definitions of all five selected dense score columns.

This section addresses a precise concern: if BundleFuse uses several scores, what prevents an indefinitely growing list? The answer must distinguish a **complete estimator**, which already specifies how to produce a ranking, from **statistical sufficiency**, which says that discarded information cannot improve a decision. BundleFuse is a complete estimator. Its current score list has not been shown statistically sufficient or minimal.

The useful research objective is to identify the smallest implementation that preserves the relevant ranking decisions within a stated accuracy and resource budget. There is no universal integer, such as five or eight scores, that the bundle-completion problem itself dictates. The derivations below establish what can replace that unsupported claim.

### 12.1 The organizing principle: use shared relations, then allow supported departures

Consider two shoes with the same category and similar descriptions. A category or description score can give them similar predictions even if only one has repeatedly appeared with the particular trousers and shirt in a query. Historical item identities can distinguish them. Conversely, a newly introduced shoe can have no recorded co-occurrence with any query item, although its attributes allow comparisons with other shoes. A zero historical count cannot distinguish an unobserved relationship from an unsuitable relationship.

These observations motivate one estimation problem: **start from relationships that can be shared across items, and adjust that prediction using the particular candidate's recorded relationships. Limit the adjustment when its history is scarce.** This describes a decision the method makes; it does not require treating each score as an independently invented mechanism.

Use the existing notation: S is the observed set, c is an eligible candidate, Y_c is the selected full score, P_c is the selected restricted fallback, s is its validation-frozen scale, f_c is the candidate's training frequency, and κ is the selected fallback strength. Define F_c = s P_c. For κ > 0, the implemented score is exactly

```text
θ_c = F_c + ρ_c (Y_c − F_c),       ρ_c = f_c / (f_c + κ).
```

The departure Y_c − F_c is computed from the selected scores; ρ_c determines how much of that departure survives. A candidate with f_c = 0 receives F_c. As f_c increases, the interpolation approaches Y_c, with Y_c and F_c held fixed for this interpretation. Frequency is an available measure of historical exposure, not a measured probability that the candidate-specific prediction is correct. In particular, frequent appearance does not guarantee frequent appearance with this query.

The word “shared” must remain concrete. Category relations pool different item IDs with the same label. Description comparisons use item attributes. SVD compares patterns of co-occurrence partners. The bridge uses descriptions and historical neighbors together. F_c can itself use historical information and popularity; it is neither an independent model nor an exclusively descriptive prior. Also, because s can differ from one, Y_c − F_c need not be a purely historical term.

For the frozen dense recipe, write Q_c for the selected restricted sum:

```text
Q_c = 0.5 z_cat(c) + 0.25 z_bridge(c) + 0.25 z_pop(c)
Y_c = Q_c + 0.5 z_simp(c) + 0.25 z_csvd(c)
F_c = s Q_c
θ_c = [s + (1−s)ρ_c] Q_c
      + ρ_c [0.5 z_simp(c) + 0.25 z_csvd(c)].
```

Here z denotes a score standardized across catalogue items for this query. Thus the actual dense estimator has five selected score columns. Its standalone cooc, ppmi, softcat, diff2 and Rao columns are not selected. PPMI and propagation still occur inside selected representations; zero outer weight does not remove such dependencies. The frozen POG recipe has four selected columns; the frozen Spotify recipe has seven weighted columns plus its explicit Rao term. These are selected implementations, not evidence that four, five or eight is a theoretically necessary count.

### 12.2 What the final rule proves without distributional assumptions

For fixed Y_c, F_c and f_c, and κ > 0, define the scalar objective

```text
J_c(t) = (f_c / 2) (t − Y_c)² + (κ / 2) (t − F_c)².
```

The first term charges for departing from the full score, with strength proportional to the candidate's history. The second charges for departing from the shared fallback. Differentiating gives

```text
J'_c(t) = (f_c + κ)t − f_c Y_c − κ F_c.
```

The second derivative is f_c + κ > 0, so there is a unique minimizer, and it is precisely θ_c above. This is an exact optimization interpretation of the final rule, including cold candidates. When κ = 0, the implementation explicitly returns Y_c; if f_c is also zero, the displayed objective alone would have no unique minimizer, so that software convention must be stated separately.

The rule also has an exact stability property. If the full score changes by e_c while the fallback is fixed, the final score changes by ρ_c e_c. Therefore, for two candidates c and d, the change in their score difference has magnitude at most

```text
ρ_c |e_c| + ρ_d |e_d|.
```

Small-frequency candidates are less sensitive to changes confined to the full score. This does not guarantee an accuracy improvement: suppressing a correct correction can also hurt. If an added source changes the fallback as well, its contribution there must be included. Nor does this objective derive the raw score menu, its normalizations, or its fitted weights. Those remain separate estimator choices.

This optimization is a standard quadratic shrinkage construction. Its value here is a faithful account of BundleFuse's behavior, not a claim that quadratic shrinkage was invented by this project.

### 12.3 The precise quantity a completion ranking needs

Sufficiency must be defined for the decision we evaluate. Let X contain all permitted information at prediction time: the observed set, training data, item attributes, candidate eligibility and any allowed user context. Let G be the nonempty random set of hidden items. There are n eligible candidates and a cutoff K ≤ n. A ranking π assigns candidate π(j) to position j.

For Recall@K, define the relevance contribution of candidate c as

```text
V_c = 1{c belongs to G} / |G|,       d_j = 1,  j = 1,…,K.
```

For binary-relevance NDCG@K, instead define

```text
d_j = 1 / log₂(j+1)
D_K(G) = sum of d_j for j = 1,…,min(K, |G|)
V_c = 1{c belongs to G} / D_K(G).
```

Both metrics now have the same form: the realized utility is the sum of d_j V_π(j) over the first K positions. Define the vector η(X) by η_c(X) = E[V_c | X]. Conditional expected utility is then

```text
U(π | X) = sum over j = 1,…,K of d_j η_π(j)(X).
```

An exchange argument proves optimality of sorting η: if a higher-weight position contains a smaller η than a lower-weight position, swapping the two increases utility by the product of their differences. For Recall only the top-K set matters; for NDCG its order matters too.

Consequently, a score representation is sufficient for this decision if it allows the optimal ordering of η to be recovered. Recovering the entire distribution of G is unnecessary. When the hidden-set size varies, sorting plain membership probabilities is not generally the same as sorting these expected normalized contributions. This distinction follows the established NDCG consistency analysis of [Ravikumar, Tewari and Yang (2011), Lemma 4](https://proceedings.mlr.press/v15/ravikumar11a/ravikumar11a.pdf).

BundleFuse's standardized scores are not calibrated estimates of η. A ranking-equivalent calibration would therefore be needed before interpreting numerical score errors as errors in these relevance contributions. A theorem about η cannot silently be asserted for the current numerical θ values.

### 12.4 An exact criterion for the value of one more score

Let Z retain the entire collection of selected candidate-score vectors, frequencies and eligibility information. Let R be a proposed additional score vector. For either metric's V defined above, set

```text
m₀(Z)    = E[V | Z]
m₁(Z,R)  = E[V | Z,R]
Δ(R | Z) = E[ ||m₁(Z,R) − m₀(Z)||₂² ].
```

The expectation is over the specified population of completion queries; the squared norm sums over eligible candidates. Δ measures the change in the best prediction of relevant items after the additional score is revealed. It is not the correlation between two raw scores and not the new score's standalone accuracy.

**Proposition 1 — additional information.** The decrease in the best possible squared prediction error from adding R is exactly Δ(R | Z):

```text
E[||V − m₀||₂²] − E[||V − m₁||₂²] = Δ(R | Z).
```

**Proof.** Write V − m₀ = (V − m₁) + (m₁ − m₀). Conditional on Z and R, V − m₁ has mean zero. Its inner product with m₁ − m₀ therefore has expectation zero. Expanding the square gives the identity.

If Δ = 0, no rule using R can improve expected Recall or NDCG over the best rule using Z, for the metric whose V was used. The conditional expected relevance vectors are equal, so their optimal rankings can be the same. A positive Δ need not improve ranking: it can change predicted values without changing the top-K ordering. This is a conditional-mean version of sufficiency, related to the broader conditional-independence formulation in [Fukumizu, Bach and Jordan (2004)](https://jmlr.org/papers/v5/fukumizu04a.html). The squared-error identity is an instance of the classical conditional-expectation projection property; see [Banerjee, Guo and Wang](https://www.dam.brown.edu/people/huiwang/research/bgw.pdf).

**Proposition 2 — a ranking bound.** Let d = (d₁,…,d_K). The improvement in optimal expected ranking utility from revealing R is bounded by

```text
improvement ≤ min(1, 2 ||d||₂ sqrt(Δ(R | Z))).
```

**Proof.** Represent a ranking by a length-n vector assigning d_j to its jth candidate and zero below K. Every such vector has norm ||d||₂. Let a₀ maximize its dot product with m₀ and a₁ maximize its dot product with m₁. Then

```text
(a₁−a₀)ᵀ m₁
  = (a₁−a₀)ᵀ m₀ + (a₁−a₀)ᵀ(m₁−m₀)
  ≤ 2 ||d||₂ ||m₁−m₀||₂.
```

The first term is nonpositive by optimality of a₀. Taking expectations and using E[||W||₂] ≤ sqrt(E[||W||₂²]) gives the bound. The cap at one follows because both metrics lie in [0,1].

This yields a precise population-level stopping condition: if every admissible extension R has Δ(R | Z) ≤ ε² / (4||d||₂²), then none improves the best Z-based ranking by more than ε. “Admissible” must be a specified class; it cannot mean an unspecified collection of all future algorithms.

There is a second issue: the current combiner might fail to extract the information already present in Z. For any Z-based prediction h on the V scale, define E₀ = E[||h−m₀||₂²]. The same orthogonality argument gives

```text
E[||h−m₁||₂²] = E₀ + Δ(R | Z).
```

Thus a bound relative to the actual ranking produced by h uses E₀ + Δ, not Δ alone. A new model can improve by extracting existing information more effectively, even if its score contains no additional information. This is why showing that PPMI and SVD are derived from the same matrix does not establish that either score is dispensable for BundleFuse's restricted combiner.

These propositions apply to a joint vector of proposed scores as well as one score. Testing additions individually cannot rule out joint information: for two independent fair bits, their exclusive-or is unpredictable from either bit alone and determined by both together. Conditional sufficiency is substantially stronger than “every single addition had a nonsignificant result.”

### 12.5 A finite sufficient aggregate under an explicit observation model

There is a genuine theorem that connects multiple correlated scores to a single aggregate and the frequency fallback. It is useful as a conditional model, but the current experiments do not validate its assumptions.

For one candidate and query, let t be an unknown real-valued compatibility quantity. Condition on the shared information used to construct a fixed F. Suppose a d-dimensional vector of bias-corrected measurements z obeys

```text
z = a t + e,       e | t,F,f ~ Normal(0, Σ/f),       f > 0,
t | F,f ~ Normal(F, τ²).
```

Here a is a known vector of measurement responses to t, Σ is a known positive-definite error covariance matrix, and τ² > 0 is a prior variance. These quantities are fixed within this conditional model; the covariance is allowed to contain nonzero correlations. Neither a common origin nor a different method name establishes error independence.

Define

```text
I = aᵀ Σ⁻¹ a > 0
y = (aᵀ Σ⁻¹ z) / I
κ = 1 / (τ² I).
```

**Proposition 3 — conditional aggregation.** For known a, Σ and f, y is sufficient for t in this likelihood, and the posterior mean is

```text
E[t | z,F,f] = F + [f/(f+κ)](y−F).
```

**Proof.** Up to terms independent of t, the negative log-likelihood is

```text
(f/2)(z−a t)ᵀ Σ⁻¹(z−a t) = (f I/2)(t−y)² + constant.
```

The data enter its t-dependent part only through y. Adding the normal prior contributes (t−F)²/(2τ²); completing the square gives the displayed posterior mean. Normal conjugacy is standard; see [Murphy, Conjugate Bayesian Analysis of the Gaussian Distribution](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf).

This theorem says that a specified collection of correlated measurements can be summarized by one scalar for this scalar target. It does **not** say that a finite number of observations is enough against all possible new measurements. The distinction becomes explicit when another measurement is introduced.

Suppose its response coefficient is a_new, its error variance before dividing by f is v, and its error cross-covariance with the old vector is k. Assume the expanded errors are jointly normal, independent of t conditional on F and f, with a positive-definite covariance before dividing by f. Define

```text
a_res = a_new − kᵀ Σ⁻¹ a
v_res = v − kᵀ Σ⁻¹ k > 0.
```

Subtracting kᵀΣ⁻¹z from the new measurement removes the part predictable from old measurement errors. The remaining measurement has response a_res to t and independent normal error with variance v_res/f. The aggregate precision factor therefore becomes

```text
I_new = I + a_res² / v_res.
```

The new measurement adds no information precisely when a_res = 0 under these assumptions. A nearly duplicated score can still help if its small remaining component measures t accurately. Exact duplicates have a singular expanded covariance and should be removed before applying the displayed inverse formulas.

This is the defensible meaning of “enough measurements” in this model: all admissible additions have zero, or sufficiently small, remaining information. Merely having one score for each named family does not establish that condition.

**Why this is not a theorem certifying today's BundleFuse.** Its standardized scores have not been shown to follow this measurement model. Its grid-selected weights are not the covariance-adjusted coefficients in y. Its full score and fallback share inputs. Historical sums are not automatically sample means with noise variance proportional to 1/f. The quantities a, Σ and τ² are not known, and their estimation would require additional analysis.

A single global κ would additionally require τ² I to be the same across the candidate/query conditions being modeled. The theorem permits conditional parameters; it does not establish that additional homogeneity assumption.

The existing variance diagnostic is particularly relevant. In the saved full recipes, the fitted slope of mean squared (Y−F) against 1/f is negative:

| Recipe | Raw slope | Selected κ |
|---|---:|---:|
| POG | −0.273238 | 0.5 |
| POG_DENSE | −2.528107 | 0.5 |
| Spotify | −5.335897 | 5.0 |

Sources are the respective [POG](../results/publication_mechanisms/pog/full.json), [dense](../results/publication_mechanisms/pog_dense/full.json) and [Spotify](../results/publication_mechanisms/spotify/full.json) recipes. These estimates do not establish the positive inverse-frequency noise relationship. They are not, by themselves, formal rejection tests either. The implementation truncates inadmissible variance estimates and considers validation-selected alternatives. Consequently, the exact quadratic objective in Section 12.2 is a better unconditional description of the current rule than a claim of a calibrated posterior or sufficient statistic.

### 12.6 A direct counterexample to universal completeness of the score menu

The limitation can be demonstrated using the production operators, rather than just asserted abstractly.

Construct four groups of items A, B, C and D. Each group contains q values, indexed 0,…,q−1. For a fixed shift h, the training corpus contains every bundle

```text
{A_a, B_b, C_c, D_((a+b+c+h) mod q)},       a,b,c = 0,…,q−1.
```

There are q³ bundles, each of size four. Assign each item's group as its category and give every item in that category the same descriptor. Compare the corpus with h = 0 to the corpus with h = 1.

Fixing r values from distinct groups, for r = 1, 2 or 3, leaves exactly q^(3−r) possible bundles, regardless of h. Any subset containing two different values from one group occurs zero times in both corpora. Hence the complete single-item, pair and triple counts are identical. Bundle-size discounts are identical as well because all bundles have size four.

For the query S = {A_0, B_0, C_0}, however, the observed completion is D_0 in the first corpus and D_1 in the second. The four-item count resolves this immediately.

Why the current scores fail here is exact:

- Pair, cooc and PPMI depend on identical frequencies and pair counts.
- SVD uses the same PPMI matrix; candidate rows for all D values are identical.
- The incidence correction depends on candidate–seed–seed triple counts, also identical.
- Propagation and the bridge use the same pair operator. Group-identical descriptors cannot distinguish D values; residual descriptions are zero in this construction.
- Category, soft-group and popularity scores are also identical for those candidates. Rao uses the same candidate–seed affinities and query relations.

Standardization and the frequency fallback cannot recover a distinction absent from all their inputs. Every D value receives the same ideal arithmetic score. An ID-based tie-break does not recover the missing relation. With q = 21 there are more indistinguishable D candidates than positions at cutoff 20. Varying h makes any of them the correct completion, so no fixed top-20 rule based on these scores succeeds for every corpus. For NDCG the ordering limitation already appears with two values.

The [verification script](../scripts/check_score_sufficiency.py) constructs q = 2 and q = 21 cases and evaluates all the listed fashion score columns, including the actual frozen dense fusion. The [saved results](../results/algebraic_diagnostics/score_sufficiency.json) verify equal first-through-third-order counts and score equality to floating-point tolerance. The largest between-corpus score discrepancies are 2.39e−7 and 1.20e−7. The script also verifies the correlated-measurement identity in 100 cases and the ranking target and bound in 200 exhaustive four-candidate checks.

This is a representational counterexample, not an estimate of how often real outfits follow such a pattern. It shows exactly why a theorem covering every possible bundle distribution cannot be correct. It also supplies a focused scientific question: are relationships requiring three jointly observed seeds important enough, on the intended data, to justify their extra estimation and computation?

**Scope in the supplied queries:** the subsequent [read-only feasibility audit](../results/algebraic_diagnostics/sufficiency_experiment_feasibility.json) found that all 4,000 POG queries and 5,900 of 5,938 POG_DENSE queries have at most two seeds. In contrast, 3,975 of 4,000 Spotify queries have at least three. A new term requiring three observed items is inactive on all supplied POG queries and 99.36% of supplied dense queries. Any set-score function restricted to query sizes at most two admits an exact expansion into a constant, individual-seed terms and seed-pair terms; this limits query-interaction order, not the difficulty of estimating its coefficients or the necessary number of score representations. The [proposed experiment document](../output/pdf/score_sufficiency_experiments.pdf) gives the proof and accordingly prioritizes separating individual/joint support for fashion and testing higher-order support on Spotify. The proposal has not been launched.

The raw incidence score has degree two in the query indicators: one term for each seed and one for each seed pair. It is not the general degree-two function: its coefficients are tied to weighted training triple counts. Nor is the entire final pipeline degree two; query standardization and the Rao denominator introduce other nonlinear dependence. A claim that BundleFuse spans all low-order completion rules would therefore also be incorrect.

### 12.7 A stopping rule that can actually be certified

The theory suggests two distinct standards. Population sufficiency would require bounding the unresolved information in Section 12.4 across an admissible class. That is generally hard. A more practical, finite standard is to certify a compact implementation against a declared collection of fitted alternatives under the ranking metric itself.

Before looking at a fresh audit sample, freeze a family H of M fitted systems. Include the proposed compact version, relevant score removals, close replacements and justified additions. Freeze the tuning procedures and their outputs using separate data. Define an acceptable absolute metric loss ε and a cost C(h), such as inference time under a fixed measurement protocol. Memory and fitting cost can be additional hard constraints. They need not be forced into an arbitrary single weighted sum.

Let N independent audit units be drawn from the target population, with each system's utility on a unit in [0,1]. A unit can be a complete original bundle with a predefined average over its masks. Independence must apply between units; repeated masks of the same bundle must not be counted as independent observations. User or temporal dependence may require a different unit or bound. Let U(h) be population utility and Ũ(h) its audit average.

For error probability δ in (0,1), define

```text
b = sqrt(log(2M/δ) / (2N)).
```

Hoeffding's inequality and a union bound imply that, with probability at least 1−δ, every member of H satisfies |Ũ(h)−U(h)| ≤ b simultaneously. This is a direct application of [Hoeffding (1963)](https://doi.org/10.1080/01621459.1963.10500830), not a new concentration theorem.

**Proposition 4 — compact-system guarantee.** Select the least-cost member h_small whose audit utility is within ε of the best audit utility. On the simultaneous event above,

```text
U(h_small) ≥ max over h in H of U(h) − ε − 2b.
```

**Proof.** Its population utility is at least its audit utility minus b. Its audit utility is at least the best audit utility minus ε. That best audit utility is at least the population-best member's utility minus b. Combining these three inequalities proves the result. Simultaneity permits this selection after seeing the audit utilities, provided the fitted systems themselves were frozen beforehand.

Alternatively, a particular baseline is certified within tolerance ε of this family when max_h[Ũ(h)−Ũ(baseline)] + 2b ≤ ε. A low-power “not significant” comparison is not such a certificate. The elementary bound can be loose: with M = 100, N = 4,000 and δ = 0.05, 2b is approximately 0.0644 absolute utility, much larger than several observed differences. A useful narrow certificate would need a justified tighter paired bound or more independent data; a theorem should not hide this limitation.

This defines sufficiency **relative to a declared family, population, metric, tolerance and cost constraint**. It does not cover an algorithm invented after the audit. It does give a principled reason to stop expanding the present system. A future model must identify an unresolved prediction problem and justify its additional resource use; the mere possibility of a tiny accuracy increase does not make the existing method incomplete.

The existing studies are informative exploratory comparisons. They do not implement this complete certification procedure, do not establish the smallest adequate subset, and have already exposed test outcomes to the research process. The new mathematical checks do not change that status. No new long benchmark run is implied by this section.

### 12.8 What the evidence allows us to say about BundleFuse itself

The existing results support useful specific mechanisms, not an indispensable fixed list. On dense POG the controlled incidence bank improves NDCG over the pair bank. On POG, selected history and shared relations work together and the fallback helps. On Spotify, a different historical-overlap estimator improved the controlled result. In the follow-up, clothing and sparse Spotify discarded the explicit historical column. These findings favor a compact, dataset-selected implementation and prevent claiming that every available source is needed everywhere.

A coherent publication statement is:

> BundleFuse estimates how an item completes an observed set from recorded item combinations and relationships shared across items. Its historical calculation preserves individual-seed and joint-seed occurrences through coefficients determined by bundle counts. Its final rule limits departures from a shared prediction according to candidate history. The resulting scorer requires no gradient-trained scoring network; its scalar choices are selected on validation data.

This statement explains the organizing choice. The exact historical estimator, the specified sharing operations, their computational implementation and their controlled empirical effects are the objects to defend. The full dense formula is one concrete instance. The story does not require claiming that every selected representation is mathematically independent or newly invented.

The strongest theoretical claims currently available are:

| Claim | Status |
|---|---|
| The final interpolation uniquely solves the stated quadratic objective for κ > 0 | Exact for the current implementation |
| The incidence correction can distinguish histories with identical pair statistics | Established in the earlier algebraic diagnostics |
| The complete fashion score menu can miss a four-item relation | Established by the counterexample here |
| Zero additional conditional-mean information implies no optimal ranking improvement | Proven, with the stated population and metric definitions |
| Correlated normal measurements have a sufficient aggregate yielding this fallback form | Proven under an explicit model; not validated for the current scores |
| The current score count is minimal or sufficient against all future models | False as a universal statement; not established on the intended population |
| A least-cost version is within a stated tolerance of a frozen family | Certifiable by the proposed procedure; not yet certified by existing runs |

The published CLHE/RaMen comparisons establish the recorded benchmark advantage under their documented comparison scope. They do not establish score sufficiency or the causal reason for the advantage. Similarly, a sufficient-statistic theorem would not establish originality: Gaussian aggregation, shrinkage and conditional-expectation arguments predate BundleFuse. Frequency-dependent mixtures also appear in related bundle recommendation, for example [CoHeat](https://jeon185.github.io/data/WWW24%28CoHeat%29.pdf), whose task and learned representations differ from this completion estimator.

The defensible response to “why not add another model?” is therefore:

> BundleFuse specifies a complete estimator with a limited computational and fitting budget. Its components are motivated by observed versus shared relationships, and retained components must justify their incremental ranking value. An extension is a legitimate comparison when it addresses a missing relationship or improves the accuracy–cost tradeoff. The contribution does not depend on claiming that no future extension can ever help.

A guarantee against every future accuracy improvement would be a guarantee of optimality for an unspecified data-generating process. The counterexample explains why that guarantee is unavailable here. A transparent estimator, exact mechanism analysis and evidence for a compact competitive implementation provide a stronger foundation than an unsupported claim that the score list is closed forever.

## 13. Primary sufficiency experiments completed

All 106 E1–E4 configurations completed. The [results review](../results/sufficiency/README.md)
contains the comparisons, population definitions, verification and corrected timings.
Exact complete-bundle overlaps were excluded from validation/test; all earlier
unfiltered results remain preserved. These filtered, newly refitted results are
not direct replacements for the earlier published-split comparison figures.

The strongest accuracy result is separate pair/joint weighting on POG_DENSE:
NDCG improves from 0.257181 to 0.261006, with recorded adjusted p = 0.037996.
The fully refitted procedure also changes other weights; its fixed-other-weight
control improves numerically but does not survive correction. POG selects zero
joint weight and matches its pairwise reference. Effective support does not beat
the separately tuned frequency control. Spotify selects zero weight for the new
candidate–three-seed term, while the top-100 amplified bank retains useful history.

A three-active-score dense candidate has about 21.4% lower median batch-one
latency and 0.91% lower NDCG than the four-active-score reference. Its uncertainty
does not certify the planned 1% noninferiority margin. The corrected timing excludes
zero-weight dependencies; original timing files are retained but should not support
efficiency claims. No compact memory/export or untouched-data sufficiency
certificate is available. The findings support a compact, dataset-dependent
construction and a better dense history balance, not a universally closed score menu.
