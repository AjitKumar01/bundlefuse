# BundleFuse: method-level prior-art review

Search performed 5 November 2026. Scope: the implemented scoring method, not the
origin of datasets. We searched item-neighbor and implicit-feedback methods,
session and playlist continuation, market-basket rules, category-aware
compatibility, hybrid recommendation, hypergraph propagation, higher-order
interactions, quadratic diversity, score fusion and cold-start backoff. We used
primary papers where available, followed their cited work where useful, and
compared task, input, operator and degree of learning. This is a targeted broad
review, not a systematic review of every database or proof of priority; citations
to a paper do not imply the full manuscript has been audited line by line.

## What changed in the positioning

The nearest conceptual family is **hybrid, item- and set-neighborhood recommendation**.
The method uses specific, explicitly computed relation scores for partial bundles;
it is not the first to combine history with content, use categories, score a
candidate against several observed items, perform graph diffusion, or give
additional weight to historical sets sharing several observations. Even the
question of whether simple non-neural methods compete with neural recommendation
has substantial prior literature. Consequently the strongest defensible framing is
an investigated construction and empirical analysis, not a claim that any of those
ideas originated here. Whether the entire exact construction has a meaningful
novel contribution remains open.

## Component map

### 1. Weighted history and item-item co-occurrence

[1. Sarwar et al., Item-Based Collaborative Filtering Recommendation Algorithms
(2001)](https://www.ra.ethz.ch/CDstore/www10/papers/519/) estimate item
relationships and aggregate evidence from observed items. Their inputs are user
ratings, not necessarily unordered bundle membership. BundleFuse's count,
frequency and bundle-size weights are particular variants of an established
item-neighborhood family.

[2. Karypis, Evaluation of Item-Based Top-N Recommendation Algorithms
(2001)](https://citeseerx.ist.psu.edu/document?doi=0739fad62026ca36f101a36f29d53630207a5748&repid=rep1&type=pdf)
explicitly studies combining item similarities to score a basket and a candidate;
its item neighborhoods and fitting choices differ from our bundle incidence model.

[3. Ning and Karypis, Sparse Linear Methods with Side Information for Top-N
Recommendations (2012)](https://archives.iw3c2.org/www2012/proceedings/companion/p581.pdf)
and [4. Steck, EASE (2019)](https://arxiv.org/abs/1905.03375) show that explicit
item relations can also be obtained through regularized fitted models. EASE is
not the same estimator as a co-occurrence count.

[5. Dai et al., Metric Factorization with Item Cooccurrence (2020)](https://www.mdpi.com/2073-8994/12/4/512)
uses PMI computed from item co-occurrences as input to a learned factorization.
Thus PMI applied to item histories predates our use. Its factorized model and
marginal definitions should not be equated with our PPMI signal.

### 2. Matching whole historical sets: a close prior family

[6. Ludewig and Jannach, Evaluation of Session-based Recommendation Algorithms
(2018)](https://arxiv.org/abs/1803.09587) evaluate nearest-neighbor session
methods. [7. Bonnin and Jannach, A Comparison of Playlist Generation Strategies
(2013)](https://web-ainf.aau.at/pub/jannach/files/Workshop_ITWP13.pdf) describe
playlist-neighbor recommendations based on co-occurrence. These methods differ
in similarity, normalization, neighbor selection, order handling, and task.
Nevertheless they make the essential conceptual move: score a candidate by
aggregating historical sets containing it, weighted by similarity to the query.

The relationship to our implemented incidence term is especially direct. For
binary, unweighted training bundles, let \(m_b=|b\cap S|\),
\(a_b=\sum_{s\in b\cap S}(f_s+5)^{-\alpha}\),
\(v_b=(|b|+1)^{-\gamma}\), and \(\beta\ge0\). Before propagation and
standardization, the pair score plus incidence correction is

\[
\sum_{b:c\in b} v_b a_b\,[1+\beta\max(0,m_b-1)].
\]

Thus it is an **all-historical-bundles weighted-neighbor score with a nonlinear
query-overlap weight**. This is an algebraic interpretation of our formula, not
a claim it equals a cited session-kNN: there is no top-k selection here and the
normalization and overlap function differ. When query-item weights are equal,
the overlap-dependent part grows quadratically in \(m_b\) for \(m_b\ge1\).
For profiles with non-unit bundle weights, the factors using the weighted
incidence matrix must be retained separately; the displayed expression is not a
claim about those profiles.

[8. Mobasher et al., Using Sequential and Non-Sequential Patterns in Predictive
Web Usage Mining Tasks](https://doi.org/10.1109/ICDM.2002.1184025) is a lead
for pattern-based recommendation; verify full text and bibliographic details
before using any detailed comparison. [9. Association Rules for Recommendations
with Multiple Items](https://doi.org/10.1287/ijoc.2013.0575) explicitly uses
multi-item antecedents, although its disjunctive consequents and rule-mining task
differ from our score. Multi-item evidence is therefore not a novelty by itself.

### Further direct overlaps found on a second search

[26. Vall et al., Feature-combination hybrid recommender systems for automated
music playlist continuation (2019)](https://doi.org/10.1007/s11257-018-9215-8),
Section 5.3, gives the candidate score explicitly as a sum of cosine similarities
over **all** historical playlists containing the candidate. This is closer than
standard top-k session neighbors: the absence of a neighbor cutoff is already
present. Their main hybrid models integrate song features with collaborative
information, but their all-playlist neighbor score is described as a baseline;
do not attribute the entire BundleFuse scoring pipeline to that baseline.

For binary bundles, their cosine score is proportional, for a fixed query, to
`sum_{b:c in b} |b ∩ S| / sqrt(|b|)`. BundleFuse's pair term with equal seed
weights and `gamma = 1/2` is `sum_{b:c in b} |b ∩ S| / sqrt(|b|+1)`.
This is a close special-case algebraic connection, **not exact equality**:
BundleFuse adds one to size, generally weights seeds by frequency, and can apply
an overlap correction. The precise evaluations and training data can differ.

[27. Kelen et al., Efficient K-NN for Playlist Continuation (2018)](https://doi.org/10.1145/3267471.3267477)
use frequency-weighted playlist overlap and amplify similarity with a power,
with a nearest-neighbor cutoff and optional positional information. This is a
close antecedent for both rare-item weighting and nonlinear strengthening of
similar historical sets. It does not show identity to our exact product of
weighted overlap and unweighted overlap count. Their paper attributes similarity
amplification to earlier work; following that citation remains necessary.

[28. Teinemaa et al., Automatic Playlist Continuation through a Composition of
Collaborative Filters (2018 preprint)](https://arxiv.org/abs/1808.04288)
combines multiple playlist-related collaborative filters and tunes their
combination. This limits any claim that a modular combination of several
interpretable association scores is distinctive by itself.

[29. Antenucci et al., Artist-driven layering and user's behaviour impact on
recommendations in a playlist continuation scenario (2018)](https://doi.org/10.1145/3267471.3267475)
ensemble content and collaborative approaches and apply additional boosts; the
paper considers playlist length and content availability. These are task-level
antecedents to multi-signal assembly and conditional treatment, not evidence
that the specific BundleFuse backoff equation occurs there.

### 3. Propagation and graph-smoothed content

[10. Zhou, Huang and Schölkopf, Learning with Hypergraphs (2006)](https://papers.nips.cc/paper_files/paper/2006/hash/dff8e9c2ac33381546d96deea9922999-Abstract.html)
uses incidence and degree normalization for spectral learning on hypergraphs.
Our code's incidence-derived operators share that mathematical family but have
specific smoothing, diagonal removal and normalizations; they are not identical
to the paper's operator. A linear incidence projection alone is representable as
an item-item matrix and should not be advertised as inherently preserving all
higher-order information.

[11. LightGCN (2020)](https://arxiv.org/abs/2002.02126) linearly propagates
learned embeddings through a user-item graph and combines layers. BundleFuse
uses fixed graph operators and descriptor propagation; LightGCN still learns
embeddings. [12. RecWalk (2019 preprint)](https://arxiv.org/abs/1909.03579)
uses random walks and item-item structure for recommendation. The general idea
of multi-hop propagation is established; exact walks and inputs differ.

[13. Self-Supervised Hypergraph Convolutional Networks for Session-based
Recommendation (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/16578)
uses learned hypergraph representations for session recommendation. It is a
relevant high-order structural comparator, not the same as our explicit
query-overlap correction.

### 4. Category relations and descriptive evidence

[14. McAuley et al., Inferring Networks of Substitutable and Complementary
Products (2015)](https://arxiv.org/abs/1506.08839) and [15. Vasileva et al.,
Learning Type-Aware Embeddings for Fashion Compatibility (2018)](https://arxiv.org/abs/1803.09196)
show that similarity, compatibility and item type are not interchangeable. The
latter learns type-aware image embeddings; our category co-occurrence and
centered descriptor scores have a different construction. Neither source
licenses equating cross-category co-occurrence with economic complementarity.

[16. Melville et al., Content-Boosted Collaborative Filtering (2002)](https://cdn.aaai.org/AAAI/2002/AAAI02-029.pdf)
uses content predictions to enrich sparse user ratings before collaborative
filtering. [17. FEASE (2025 preprint)](https://arxiv.org/abs/2504.02288) extends
a closed-form recommendation model using side features for cold start. Together
with [3], these exclude claims that metadata plus history, or shallow models with
metadata, are new. How close our group-centered features and graph-smoothed
feature kernels are to existing feature-engineered hybrid recommenders still
requires formula-level comparison.

### 5. Fusion and overlap between mechanisms

[18. Volkovs et al., Two-stage Model for Automatic Playlist Continuation at Scale
(2018)](https://doi.org/10.1145/3267471.3267480) combines item- and
playlist-neighbor scores with other collaborative signals, adjusts for item
popularity and uses a learned second-stage reranker. [19. Ferraro et al.,
Automatic Playlist Continuation Using a Hybrid Recommender (2018)](https://doi.org/10.1145/3267471.3267473)
fuses a co-occurrence model with a factorization model using audio and text.
Both are close to the overall strategy, not merely to isolated ingredients. They
differ from BundleFuse's particular scores, fusion and candidate-frequency
fallback; Ferraro's paper also uses playlist position and ranking fusion.

[20. CLHE](https://arxiv.org/abs/2310.18770) and [21. RaMen](https://arxiv.org/abs/2507.14361)
already combine multiple sources and relationships for bundle construction.
Their learned representations and training objectives differ from explicit
candidate scoring, but describing them as neglecting history, item types or
multi-item structure would be inaccurate.

### 6. Quadratic support and frequency-dependent backoff

[22. Rendle, Factorization Machines (2010)](https://jame-zhang.github.io/assets/algo/Factorization-Machines-Rendle2010.pdf)
and [23. Xue et al., DeepICF (2018 preprint)](https://arxiv.org/abs/1811.04392)
model interactions among features or items. They do not establish equivalence to
the candidate-specific redundancy-corrected quadratic term. That term uses
candidate-to-seed support as weights and seed-to-seed dissimilarity as a kernel;
it is mathematically related to a weighted quadratic diversity functional.
[Rao-based diversity has also been used in recommender evaluation](https://www.tandfonline.com/doi/full/10.1080/1369118X.2018.1444076),
but evaluating recommendation-list diversity is a different use from scoring
candidate support. No priority conclusion about this exact construction is
possible from this observation alone.

[24. Han et al., Addressing Cold Start in Product Search via Empirical Bayes
(2022)](https://www.amazon.science/publications/addressing-cold-start-in-product-search-via-empirical-bayes)
combines behavioral signals with nonbehavioral priors. BundleFuse's fallback
shares channels with its main score and uses candidate frequency as a heuristic
proxy; it is not automatically a calibrated posterior. Neither shrinkage nor
adapting reliance to sparse observations is a new research principle.

## What this means for a paper

A plausible story is: item- and set-neighborhood methods provide direct
association evidence; compatibility requires relation-sensitive evidence beyond
one generic notion of similarity; hybrid methods show how attributes can help;
BundleFuse instantiates a specific collection of explicit scores and combines
them for set completion. Its value must be demonstrated by explaining the
operator choices and measuring the resulting behavior against appropriate
neighbors and hybrids. Simple versus neural is not itself a new discovery:
[25. Ludewig et al. (2019)](https://arxiv.org/abs/1910.12781) already perform
such comparisons for session recommendation.

The clearest unresolved checks are (a) whether earlier session-neighbor models
use the same nonlinear overlap weighting (power amplification is already known), (b) whether the exact candidate-weighted
quadratic concordance appeared previously in recommendation or related retrieval,
(c) whether close score-normalized hybrids already use the same adaptive fallback,
and (d) detailed comparison with full equations and code of the closest
playlist and basket methods. The present search cannot certify absence of such
work, and numerical performance is a separate issue from method priority.

## Further citation tracing and formula comparison

### The incidence correction is a constrained itemset-to-item model

For binary, unit-weight bundle incidence let \(B_{bc}=1\) mean candidate \(c\)
appears in historical bundle \(b\), let \(d_s=(f_s+5)^{-\alpha}\), and let
\(v_b=(|b|+1)^{-\gamma}\). The implementation's incidence correction is

\[
\Delta(c\mid S)=\beta\sum_b B_{bc}v_b
\left(\sum_{s\in S}d_sB_{bs}\right)
\max\left(0,\sum_{t\in S}B_{bt}-1\right).
\]

Whenever the sum of seed indicators is zero the weighted overlap is zero, so
expansion gives the exact identity

\[
\Delta(c\mid S)=\beta\sum_{\substack{s,t\in S\\s\ne t}}
 d_s\underbrace{\left(\sum_b v_b B_{bc}B_{bs}B_{bt}\right)}_{T_{cst}}.
\]

The term \(T_{cst}\) is the weighted count of historical bundles containing
candidate \(c\) and both seeds \(s,t\). Thus this is a weighted sum of
seed-pair-to-candidate associations; it can be computed via incidence without
materializing a cubic tensor. This is an exact rewrite for this binary,
unit-bundle-weight case and applies before further diffusion or channel
standardization. It does not mean the implementation explicitly stores the
coefficients, nor does it assert that each triple is a statistically significant
interaction beyond pairwise expectations.

[30. Deshpande and Karypis, Item-Based Top-N Recommendation Algorithms (2004)](https://doi.org/10.1145/963770.963776)
consider itemset-to-item neighborhoods. [Author-hosted technical-report version](https://conservancy.umn.edu/bitstreams/8605a488-5214-4db9-9245-de171e5e1157/download)
was indexed in search but direct access was denied; the description of their
higher-order method is independently supported by the HOSLIM paper below.

[31. Christakopoulou and Karypis, HOSLIM (2014)](https://www.researchgate.net/publication/271908941_HOSLIM_Higher-Order_Sparse_LInear_Method_for_Top-N_Recommender_Systems)
explicitly adds itemset-to-item terms to item-to-item terms, learning sparse
coefficients with regularization. Its introduction identifies prior higher-order
neighborhood methods. Our triple coefficients instead have the constrained
count-based form above and a common seed-frequency weighting; HOSLIM fits
coefficients and can consider other itemset sizes. This is a closely related
methodological predecessor, not proof of identical predictions or a verified
comparison on the same data. The rationale 'pairwise relations miss interactions
among items' was already explicit in this literature.

### Amplification predates playlist continuation

Following the reference in Kelen et al. leads to [32. Breese, Heckerman and
Kadie, Empirical Analysis of Predictive Algorithms for Collaborative Filtering
(1998)](https://arxiv.org/pdf/1301.7363), Section 2.2.3. It raises neighborhood
weights to a power to emphasize stronger neighbors and also studies inverse
frequency weighting. Kelen et al. apply this lineage to playlist neighbors.
For equal seed weights, BundleFuse's local contribution is proportional to
\(m+\beta m(m-1)\), where \(m\) is overlap size. At \(\beta=1\) this is
\(m^2\); the analogy to power amplification is direct, while normalizations,
weighting, cutoff and other scores remain different. The code's default beta
is not asserted to be one.

### Basket completion and set diversity are established direct precedents

[33. Gartrell, Paquet and Koenigstein, Low-Rank Factorization of Determinantal
Point Processes (AAAI 2017)](https://ojs.aaai.org/index.php/AAAI/article/download/10869/10728)
explicitly predicts an item to add to an observed basket, learning a set model
whose kernel captures item quality and diversity. [34. Warlop, Mary and Gartrell,
Multi-Task Determinantal Point Processes for Recommendation (2018)](https://arxiv.org/abs/1805.09916)
proposes a basket-completion model and discusses categories, pairwise similarity
and distinct roles. [35. Gartrell, Dohmatob and Alberdi, Deep Determinantal
Point Processes (2018 preprint)](https://arxiv.org/abs/1811.07245) incorporate
metadata and nonlinear item interactions in a learned DPP. A determinant-based
probabilistic set model is not the same as BundleFuse's candidate-specific
quadratic concordance. But these papers directly precede our intuition that
set composition and redundancy matter for candidate selection; a vague claim of
introducing that intuition would be wrong.

[36. Sun et al., Revisiting Bundle Recommendation (SIGIR 2022)](https://pure.tudelft.nl/ws/portalfiles/portal/145063702/3477495.3531904.pdf)
explicitly distinguishes bundle detection, completion and ranking and discusses
category-level pattern mining. This is method/task context, not a citation to
the origin of our source datasets. The indexed paper was accessible through
search excerpts but a full direct fetch returned 403; detailed claims about its
formulas require a separately accessible copy.

### Current novelty assessment

The historical-pair part is a close variant of a documented all-neighbors score;
the incidence surplus has an exact itemset-to-item representation, a class
studied by HOKNN and HOSLIM; nonlinear neighbor amplification and inverse
frequency are older; and basket-level quality/diversity modeling predates our
concordance term. The plausible novelty, if any, lies in the **particular
constrained combination**, including how counts, redundancy, category relations,
content and backoff interact. That is a narrower claim than a new paradigm. It
is not established merely because the exact lines of code or constants differ.
The search has not proved that the full formula or its quadratic concordance
is unique; further citation searches, full-text access, and comparison with
uncovered neighbors remain necessary before asserting priority.

### Candidate-dependent fusion: a particularly close predecessor

[37. Hu et al., Hybrid Item-Item Recommendation via Semi-Parametric Embedding
(IJCAI 2019)](https://www.ijcai.org/proceedings/2019/0350.pdf), Equations 4–6,
blend behavioral and content representations using an item-specific confidence
weight learned from item statistics, including activity frequency. This is
representation-level fusion with a learned sigmoid and item-item prediction;
BundleFuse mixes query-dependent scores using a specified rational function.
The core intuition of trusting behavioral evidence less for cold items is
already present.

[38. Jeon et al., CoHeat (WWW 2024)](https://datalab.snu.ac.kr/~ukang/papers/coheatWWW24.pdf),
Section 3.3, combines two view scores as
\(\widehat y_{ub}=\gamma_b h_{ub}+(1-\gamma_b)a_{ub}\), with \(\gamma_b\)
increasing in the bundle's interaction count. It explicitly states and proves
that the sensitivity to the sparse-history view decreases as popularity falls,
while sensitivity to the other view increases. This is the **same monotonic
mixture principle and derivative argument** used in our note. The context is
recommending a complete bundle to a user, not completing a partial bundle;
its two scores and its weight function are not the same as BundleFuse's. It
also trains graph representations and uses curriculum learning. Our
\(f/(f+\kappa)\) coefficient and overlapping fallback cannot be identified
with CoHeat merely from this comparison, but the broad frequency-adaptive
mixture and monotonicity proof must not be claimed as new.
