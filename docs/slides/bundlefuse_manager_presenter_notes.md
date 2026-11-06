# BundleFuse — explanations for presenting the 11-slide deck

Read this as an explanation of the argument, rather than a script to memorize. Each slide section answers what the displayed terms mean, why the operation is used, and what it can and cannot establish. The worked examples are hypothetical; the benchmark results in slides 9–11 are measured results.

## The argument in ordinary language

Suppose a customer has selected a particular shirt and a particular pair of trousers. We want to rank the remaining items in the catalogue as possible additions.

The training data contain earlier outfits. If a shoe occurred in an outfit containing this shirt and these trousers, we have a recorded example of the three items being chosen together. If a different shoe occurred with the shirt in one outfit and with the trousers in another, we have two separate observations. These observations need not receive the same score: the second situation never records the three items together.

However, some catalogue shoes occur in hundreds of training outfits, some occur once, and some occur in none. Counting their appearances with the two query items therefore provides very different amounts of information. In particular, a shoe absent from all training outfits has zero recorded co-occurrences with every query item. Zero recorded occurrences does not tell us whether the shoe would be suitable; it tells us that this table cannot answer the question for that shoe.

We can still compute other quantities. If the shoe has a supplied category and description vector, we can use the observed relations between its category and the query categories, or compare its description with the query descriptions. These calculations can assign a score even when this particular shoe has never appeared in an outfit. Their limitation is equally concrete: many shoes share a category, and similar descriptions may describe interchangeable products rather than products that should be chosen together.

BundleFuse computes both kinds of score. It combines the observed bundle matches with scores from item attributes and recorded item relationships. It then uses the candidate's training frequency to determine the coefficient multiplying the full score and the coefficient multiplying a specified subset of those scores.

**The presentation's question is whether these explicit calculations can rank missing bundle members well without fitting a neural scoring model.** The results answer this empirically on the displayed datasets. They do not prove that every calculation is necessary on every dataset.

### Translate the phrases that caused confusion

| Phrase on or around the slides | What it means here |
|---|---|
| “Coverage is uneven” | Items have different training counts, and candidate–query-item pairs have different recorded counts. An item with no training occurrences has no recorded item-pair or triple occurrences. It does not mean some named percentage of the catalogue has been measured unless we report that percentage. |
| “Historical evidence” | A recorded occurrence, a count, or a weighted sum of occurrences from the training bundles. It is not an unspecified confidence measure. |
| “Exact identities” | The exact catalogue IDs, such as shoe number 127, rather than all shoes in its category. Matching category labels combines observations about several IDs; matching a pair of IDs uses only their recorded co-occurrences. |
| “Transfer evidence” | Compute a score for an item using a relation shared with other items: for example, assign it the category-pair score of its category. The score is evaluated through a shared label or vector instead of requiring this particular ID pair to occur in training. |
| “Shared relations” | Specified tables or similarities: category-pair scores, description-vector inner products, or similarities between vectors of association with other items. These are different calculations and should be named when discussed. |
| “Candidate history” | The training-bundle count of the candidate, denoted by \(f_c\), in the binary formulation. It is not the user's personal purchase history. |
| “Control reliance” | Multiply the full score by \(f_c/(f_c+\kappa)\) and the scaled fallback score by \(\kappa/(f_c+\kappa)\). These are numerical coefficients, not a separate learned confidence predictor. |
| “Restricted fallback” | A specified subset of the score sources used in the full combination. It is recomputed for the same query and candidate. It is not a default item list or a second independent model. |

## Slide 1 — What is the task?

The catalogue \(\mathcal I\) contains the eligible item IDs. The observed set \(S\) contains the items already supplied. A candidate \(c\) is any catalogue item outside \(S\). A seed is simply an observed item in \(S\).

For the outfit illustration, write \(S=\{s,t\}\), where \(s\) is one particular shirt ID and \(t\) is one particular trousers ID. These symbols refer to individual catalogue entries, not the categories “shirt” and “trousers.”

The algorithm assigns a real number \(\theta(c\mid S)\) to every eligible candidate. It sorts those numbers and returns the top \(K\):

\[
\widehat C_K(S)=\operatorname{TopK}_{c\in\mathcal I\setminus S}\theta(c\mid S).
\]

The score is a device for ordering candidates. A score of 0.8 does not mean an 80% probability that the candidate belongs in the outfit.

For evaluation, an existing test bundle is divided into observed and withheld items. For example, if the original bundle is \(\{s,t,a\}\) and the query is \(\{s,t\}\), the withheld answer is \(a\). The algorithm receives \(s,t\), not the answer \(a\). It succeeds on this query if \(a\) appears sufficiently high in its ranking.

This operational definition matters. We evaluate recovery of recorded bundle members. We are not claiming that every other shoe is objectively unsuitable, or that the test bundle is the only valid outfit.

**A sentence to present:** “We observe part of a recorded bundle, score all remaining catalogue items, and measure how highly the withheld members are ranked.”

**Lead into slide 2:** “To compute those scores, we have examples of earlier bundles and attributes of the catalogue items. Each can answer questions the other cannot.”

## Slide 2 — Why are earlier bundles alone insufficient?

The two examples now printed on this slide are hypothetical. On the historical side, a newly listed shoe has no recorded occurrence with the selected shirt and trousers, so those exact candidate–seed counts cannot support it. On the descriptive side, assume the withheld item is a shoe: another pair of trousers can resemble an observed item and still fail to recover that missing shoe. This illustrates the distinction between resemblance and the evaluated completion, rather than a universal rule forbidding multiple trousers in every bundle.

### Follow one small training table

Let each row below be a training bundle. The labels are item IDs; only \(s,t\) have been assigned clothing roles for the illustration.

| Training bundle | Item IDs |
|---|---|
| 1 | \(s,t,a\) |
| 2 | \(s,u,a\) |
| 3 | \(t,v,a\) |
| 4 | \(s,t,b\) |
| 5 | \(u,v,b\) |
| 6 | \(u,v,d\) |

Also include an item \(c\) in the catalogue, with known attributes, but no occurrence in these six rows. The query is \(S=\{s,t\}\).

Item \(a\) appeared with \(s\) twice and with \(t\) twice. It also appeared with both together in row 1. Item \(b\) appeared with each query item once, in row 4. Item \(d\) appeared in training but never with either query item. Item \(c\) never appeared in training.

These are three different limitations of a count-based answer:

1. An item can have many or few observations overall.
2. An item can be observed overall but have no recorded relation to this query.
3. A pairwise table can record relations to both query items without identifying whether they were ever observed together in one bundle.

“Uneven coverage” was shorthand for the first two facts. It is clearer to say them directly.

### Why attributes can produce a score when the ID counts cannot

Suppose \(c\) is labelled “shoe.” The query items are labelled “shirt” and “trousers.” Across training bundles, we can count relations between these categories using occurrences of other shoe IDs. A category source assigns \(c\) a score using the “shoe” row or column of that category table. It does not require \(c\) itself to have occurred with \(s\) or \(t\).

Alternatively, let \(E_i\) be a specified normalized description vector for item \(i\). A descriptor source can compute

\[
r_E(c\mid\{s,t\})=E_c^\top\frac{E_s+E_t}{2}.
\]

This number is defined even when \(c\) has no training-bundle occurrences, provided its vector is available. It compares vector coordinates; it does not invent an occurrence of \(c,s,t\).

The limitation is that category scores can tie all candidates in one category, while similar description vectors may identify alternatives to an observed item. That is why neither operation, by itself, answers the whole completion problem.

**A sentence to present:** “The bundle table may have no observations about a candidate with this query. Categories and descriptions still let us compute a score, but they provide a broader comparison than observing these exact items together.”

CLHE and RaMen also address limited observations using multiple inputs. The research question is how effectively our explicit scoring construction can do so without neural scoring-model training.

## Slide 3 — Research question

The preceding slide shows two concrete failures: a candidate can lack recorded co-occurrences with the query, and a candidate can resemble an observed item without being the missing answer. This slide turns those observations into the question the experiments must answer.

**Present the question as:** “Can we obtain accurate completion rankings from recorded bundle occurrences and available item attributes, without training a neural scoring model?”

There are three requirements behind this question.

First, retain the distinction between joint and separate observations. For query items shirt ID s and trousers ID t, a training row containing s, t, and candidate a is different from one row containing s and a and another containing t and a. A scoring rule that uses only the two pair counts cannot generally distinguish those situations. The incidence calculation is one way to retain the information in the original rows. This is a capability we can test; it is not a claim that every displayed dataset selects that calculation.

Second, calculate a score when the candidate–query ID pairs are absent. A zero count can mean no observation was recorded. Attributes allow a category or vector comparison to be evaluated for that candidate. This requires the relevant attributes to be available and does not guarantee that the resulting score is correct.

Third, establish whether allowing these calculations together improves the selected system. Compare historical-only, shared-only, combined, and full systems on the same queries within each dataset. Then compare the full system's reported metrics with published neural results under the stated benchmark scope. These comparisons supply empirical answers, not a theorem that explicit scores always outperform trained models.

The slide states the question and assessment criteria before introducing the competing architectures or our score formula. A useful transition is: “The neural baselines learn representations for this task. We will compare their approach with a system that calculates these scores directly.”

## Slide 4 — What do the neural architectures learn?

The table summarizes three approaches. Explain the type of fitted mapping, rather than expanding every acronym.

### CLHE: learn what information to emphasize at two levels

Each item has multiple representations, including supplied multimodal descriptors and user-feedback features. Item-level attention computes weights for combining these representations. For a partial bundle, a second attention stage combines its item representations into one bundle vector.

The weights depend on fitted neural matrices. Training changes those matrices and item embeddings so that withheld or target items receive larger scores. The final scoring operation is an inner product:

\[
\theta(c\mid S)=e_S^\top v_c,
\]

where \(e_S\) is the partial-bundle vector and \(v_c\) is the candidate vector. Contrastive objectives additionally encourage representations from related augmented views to agree relative to other examples.

A practical explanation of attention is: “Instead of assigning the same fixed contribution to every available representation, the model computes contributions using weights fitted during training.” This is sufficient for the architecture comparison.

The checked POG-sized configuration has 9,483,200 optimizer-registered parameters:

\[
2(48{,}676)(64)+3{,}252{,}672=9{,}483{,}200.
\]

The first term is two 64-dimensional item tables; the second is the remaining registered weights. This count was derived from the local POG_DENSE checkpoint structure and adjusted to the POG catalogue size. It is not a count quoted from the paper. “Optimizer-registered” means included in the parameter collection passed to the optimizer.

For audit questions: the checked local TransformerEncoder keeps attention layers in an ordinary Python list, so those tensors are outside `model.parameters()` and the saved state dictionary. The displayed count uses the optimizer-registered scope consistently. Frozen upstream extractors and their training are excluded.

### RaMen: learn several representations of the same item and bundle

RaMen combines a representation computed from item characteristics with one computed through item–item graph connections. It also learns item memberships in latent hyperedges: groups through which information can be aggregated without requiring every relation to be represented as a separate pair.

Its explicit and implicit representations are scored as

\[
\theta(c\mid S)=g_S^\top g_c+\phi_S^\top\phi_c.
\]

Here \(g\) denotes the explicit representation and \(\phi\) the implicit representation. The sum combines their two candidate scores. Its alignment objective encourages the two representations of the same item or bundle to agree relative to representations of other items or bundles.

The checked POG checkpoint has 16,058,624 registered trainable parameters. Its configuration uses 64-dimensional embeddings, four hyperedges, and four configured graph layers. Frozen description and image tables are excluded. The count depends on the configuration and catalogue size.

### Bundle-MLLM: adapt an existing language model to choose a completion

Bundle-MLLM converts an item's non-textual features into a vector placed in the language model's input alongside textual tokens. The prompt supplies a partial bundle and candidate choices. The model predicts the answer option.

The 7B entry refers to its Llama2-7B **backbone size**. It does not mean seven billion parameters are all updated for this task. In the first adaptation stage, LoRA trains low-rank updates while the backbone remains frozen. A later stage trains multimodal fusion/projection with the LLM and learned LoRA weights frozen. We do not have an audited total for its trained adapters and projection components.

Its primary reported metric is multiple-choice HitRate@1 over a small candidate set. That is a different experiment from ranking the full catalogue. This table compares architectures; the later results slides compare CLHE and RaMen published scores with our corresponding full-ranking results.

**A sentence to present:** “CLHE learns how to combine views and items, RaMen additionally learns graph and group representations, and Bundle-MLLM adapts a pretrained language model. BundleFuse instead computes its scoring representations from counts and supplied attributes.”

## Slide 5 — What does BundleFuse compute instead of learning a neural scorer?

A neural training step adjusts model parameters in response to a loss. For instance, an update has the form \(W\leftarrow W-\eta\nabla_W L\), where \(W\) is a parameter matrix, \(\eta\) is a learning rate, and \(L\) is a training loss. The BundleFuse scoring core does not fit such a matrix or an item-ID embedding table by gradient updates.

It constructs count operators and item vectors, evaluates candidate scores, and selects scalar settings using validation queries. SVD and clustering, where used, still fit representations numerically. Supplied description vectors may come from pretrained models. “No neural-network training” describes this downstream scoring core; it does not mean that producing all inputs or fitting all representations requires no computation.

### How a training bundle produces a candidate score

Define the binary matrix \(B\in\{0,1\}^{m\times n}\): row \(b\) is a training bundle, column \(i\) is an item, and \(B_{bi}=1\) when the item is present. Define

\[
f_i=\sum_b B_{bi},\qquad d_b=\sum_iB_{bi},\qquad
a_i=(f_i+5)^{-1/2},\qquad v_b=(d_b+1)^{-1/2}.
\]

For the query \(S\), bundle \(b\)'s match is

\[
o_b(S)=v_b\sum_{s\in S}B_{bs}a_s.
\]

Only query items present in that row contribute. A common query item contributes less per match than a rare one. For example, training frequencies 4 and 95 give weights \(1/3\) and \(1/10\). A match inside a larger bundle also contributes less: sizes 3 and 15 give bundle factors \(1/2\) and \(1/4\). The offsets and exponents are specified choices, not consequences of a proven optimality theorem.

The direct candidate score adds the matches of the bundles containing it:

\[
p_c(S)=\sum_bB_{bc}o_b(S).
\]

For the six-row example on slide 2, \(f_s=f_t=3\), every bundle has size 3, and a one-seed match contributes \(1/(2\sqrt8)\approx0.176777\). A two-seed match contributes twice that. Consequently:

| Candidate | Rows contributing a query match | Direct score |
|---|---|---:|
| \(a\) | 1, 2, 3 | 0.707107 |
| \(b\) | 4 | 0.353553 |
| \(d\) | None | 0 |
| \(c\) | None; absent from training | 0 |

These are raw scores before standardization and fusion.

### What “preserve joint evidence” means numerically

The incidence-enhanced source changes each row's contribution to

\[
g_c(S)=\sum_bB_{bc}o_b(S)
\left(1+\beta[k_b(S)-1]_+\right),\qquad
k_b(S)=\sum_{s\in S}B_{bs},
\]

where \([x]_+=\max(x,0)\) and the examined coefficient is \(\beta=1.5\). A row with one query match has multiplier 1. A row with two has multiplier 2.5. The multiplication occurs **before** adding the row's contribution to its member candidates.

Rows 1 and 4 in the example have two query matches. Their contributions each increase from approximately 0.353553 to 0.883883. The resulting scores are \(g_a\approx1.237437\) and \(g_b\approx0.883883\). No credit is assigned for a triple that is absent from the training rows.

The reason to retain this calculation is exact: adding candidate–seed counts cannot in general determine whether the candidate occurred with two seeds in the same bundle. The incidence calculation accesses that additional recorded fact. It does not require the entire query set to have occurred together.

The optional source is selected differently by dataset. The displayed POG_DENSE model uses the incidence enhancement; the displayed POG model uses the direct score without it. Therefore the POG result cannot be explained by an incidence term that its selected recipe does not use.

**A sentence to present:** “We score a candidate by the matches of training bundles containing it, discount common query clues and large bundles, and can give extra weight to rows matching several query items together. Other specified scores let candidates be compared through attributes and item relations.”

## Slide 6 — What does the frequency-controlled final score actually change?

The slide combines two already computed numbers, \(Y_c\) and \(P_c\). The full score \(Y_c\) combines all selected sources. The score \(P_c\) combines the chosen category, descriptor or metadata sources, with selected popularity included. Which of these sources are available and retained depends on the dataset. Calling this a “fallback” describes when it receives greater weight; it does not define its contents by itself.

For the selected POG_DENSE model, its contents are exactly \(P_c=0.5z_{cat}+0.25z_{bridge}+0.25z_{pop}\). The category term scores the candidate category against the query categories; the bridge term compares descriptions augmented with descriptions of historically connected items; the popularity term standardizes the candidate’s log training frequency. This formula is a worked example for the presenter; slide 7 shows the general expression. The fallback still depends partly on training bundles through the category and bridge computations. It is not a default popularity list or a history-independent model.

Let \(f_c\) count the training bundles containing the candidate. For \(\kappa>0\),

\[
\theta(c\mid S)=\rho_cY_c+(1-\rho_c)s_0P_c,
\qquad \rho_c=\frac{f_c}{f_c+\kappa}.
\]

“Control reliance” means changing these two coefficients. No separate neural confidence predictor is involved.

The motivation is that the raw candidate-specific occurrence score for an item with zero training appearances contains no positive occurrence contributions. Giving a score computed through categories or descriptions more weight is a reasonable design choice in that situation. Frequency is only a proxy for the amount of information: it does not measure how many of those occurrences are relevant to this particular query, and does not prove that the full score is accurate.

### A numerical example: the gate changes the ranking

Take \(\kappa=5\) and \(s_0=1\), solely for illustration. Suppose the scores have already been computed as follows:

| Candidate | Training count \(f_c\) | Full score \(Y_c\) | Fallback \(P_c\) | Full-score coefficient \(\rho_c\) | Final score |
|---|---:|---:|---:|---:|---:|
| A | 45 | 0.8 | 0.2 | 0.9 | 0.74 |
| B | 0 | 0.9 | 0.3 | 0 | 0.30 |
| C | 0 | 0.1 | 0.7 | 0 | 0.70 |

Without the gate the order by \(Y\) is B, A, C. After the gate it is A, C, B. The rare candidate B is reduced, while the rare candidate C is increased. The rule does not reward rarity automatically; it moves toward the fallback value.

These invented scores demonstrate the arithmetic; they are separate from the six-row training example and are not measured model outputs.

At \(f_c=\kappa\), the two coefficients equal one half. At \(f_c=0\), only the scaled fallback remains. For very large \(f_c\), the full score dominates. If fallback is disabled, the implementation returns \(Y_c\) directly; it does not evaluate a \(0/0\) fraction at zero frequency.

### Why the word “regularization” is justified here

For fixed \(Y_c,P_c\), the final score uniquely minimizes

\[
\frac{f_c}{2}(t-Y_c)^2+\frac{\kappa}{2}(t-s_0P_c)^2
\]

over the scalar \(t\). The derivative is \(f_c(t-Y_c)+\kappa(t-s_0P_c)\); setting it to zero gives the displayed score. This objective penalizes moving away from either input, with candidate frequency setting the first penalty's weight.

This exact statement concerns the last scalar operation. It does not derive every earlier score from one optimization problem.

**A sentence to present:** “Candidate frequency sets a numerical interpolation coefficient. With few recorded appearances, we move toward a score based on the specified fallback sources; with many, we retain more of the full score.”

## Slide 7 — What are the full score and fallback made of?

Each raw scoring rule is indexed by \(h\). It produces \(r_h(c\mid S)\); standardization on slide 8 produces \(z_h(c\mid S)\). The full and fallback scores are

\[
Y_c=\sum_{h\in\mathcal H}w_hz_h(c\mid S),\qquad
P_c=\sum_{h\in\mathcal P}\widetilde w_hz_h(c\mid S).
\]

\(\mathcal H\) is the available source collection, with zero weights excluding sources. \(\mathcal P\) is the specified fallback subset. The scalar weights are fixed by the recipe or selected using validation queries. They need not sum to one.

### Name the calculation when saying “relation”

A category calculation combines counts for item IDs carrying the same supplied category label. This is why several candidates can inherit one category relation. It is not a fitted claim that all items in that category are equally suitable.

A description calculation takes inner products of specified item vectors. In the clothing formulation, category means are subtracted before residual vectors are normalized. Subtracting a category mean removes the average feature coordinates of that category; it does not remove every category distinction or guarantee that remaining coordinates encode compatibility.

A bridged description adds a normalized weighted sum of the descriptions of historically connected items to the item's own description. Thus it can compare an item's own features with features of another item's recorded partners. “Historically connected” here means items sharing training bundles, with the normalization specified by the graph operator. It does not refer to an unspecified nearest-neighbor list.

A PPMI relation compares a recorded pair count with an independent-endpoint baseline. The SVD source instead compares compressed vectors of such associations: two items can be similar because they have similar partners, even if they never occurred together. Having similar partners can also indicate substitutes; the relation is not automatically evidence of complementarity.

These sources reuse parts of the same observations. Their sum should not be interpreted as adding independent likelihoods.

### An actual selected recipe makes “restricted” unambiguous

For the displayed POG_DENSE result, write the standardized category, bridge, incidence-enhanced, popularity, and association-profile scores as \(z_{cat},z_{bridge},z_g,z_{pop},z_{svd}\). Its fitted recipe is

\[
Y_c=0.5z_{cat}+0.25z_{bridge}+0.5z_g+0.25z_{pop}+0.25z_{svd},
\]

\[
P_c=0.5z_{cat}+0.25z_{bridge}+0.25z_{pop}.
\]

Every term is evaluated for the same candidate and query. The fallback omits the incidence-enhanced and SVD terms. It retains category and bridged descriptions, and the selected popularity score. It therefore still uses some training history, and is not an independent content-only model.

This recipe uses \(\kappa=0.5\) and \(s_0\approx1.46055\). Defining \(\rho_c=f_c/(f_c+\kappa)\), its final score can be written exactly as

\[
\theta_c=[s_0+(1-s_0)\rho_c]P_c+
\rho_c(0.5z_g+0.25z_{svd}).
\]

This equation states precisely what is retained and attenuated. At zero frequency the last parenthesis receives coefficient zero. As frequency increases its coefficient approaches one. The shared \(P_c\) part is retained with the shown scale.

The settings are chosen by validation performance. The decimal weights are not deduced from a universal law. The manuscript explanation should motivate their roles; validation establishes which available choices worked under the selected protocol.

**A sentence to present:** “The fallback uses a specified subset of the same score sources. In the dense POG recipe, it keeps category, bridge, and popularity scores while omitting the direct incidence and association-profile terms.”

## Slide 8 — Why standardize, and what data does that use?

Suppose one raw rule produces counts in the hundreds and another produces inner products between −1 and 1. Giving them equal numerical weights would let the first dominate largely because of its units.

For each query and source, we compute scores over all \(n\) catalogue items, then their mean and sample standard deviation:

\[
\mu_h(S)=\frac1n\sum_i r_h(i\mid S),\qquad
\sigma_h^2(S)=\frac1{n-1}\sum_i(r_h(i\mid S)-\mu_h(S))^2.
\]

The score supplied to fusion is

\[
z_h(c\mid S)=\frac{r_h(c\mid S)-\mu_h(S)}{\sigma_h(S)+10^{-9}}.
\]

For example, raw scores \([0,2,4]\) have mean 2 and sample standard deviation 2. Their standardized values are approximately \([-1,0,1]\). Another source with raw scores \([100,200,300]\) produces the same standardized values. The sources agree on the relative positions of these candidates despite their different numerical units.

Standardization does not say that both sources are equally predictive. The validation-selected weights still determine how much each standardized source contributes. A constant raw source contributes zeros after centering. The resulting values remain ranking scores, not probabilities.

### Why this does not use the answers

To calculate these statistics, the scorer needs the observed query, training-derived operators, and available catalogue attributes. It does not need to know which item is the hidden answer. In the six-row example, calculating the mean over candidates does not require being told whether \(a\), \(b\), or another item was withheld from a test bundle.

This is a per-query computation. It does not pool ground-truth labels across test queries. Observed seeds remain included when calculating these score statistics; they are excluded when returning recommendations. Validation labels choose settings before test evaluation; test labels are used to measure the resulting rankings.

The known catalogue and its supplied attributes are part of this benchmark setup. This explanation does not claim a protocol where attributes of future unseen catalogue items are unavailable. The sample-standard-deviation formula above describes the full-ranking POG/Spotify path; retail/commerce uses a different NumPy convention.

**A sentence to present:** “We express a candidate's score as its position relative to the other catalogue scores for this query. This removes arbitrary score units and does not require the hidden completion labels.”

## How to explain Recall@20 and NDCG@20

Both metrics compare the **ordered list returned for one query** with that query's withheld items. An item is relevant here if it belongs to the withheld ground truth. The already observed items are excluded from the recommendation list before the top 20 is selected.

Let \(G\) be the set of withheld items, and let \(L_{20}\) be the first 20 recommended items after excluding observed items. A **hit** is an item in both sets. For one query,

\[
\operatorname{Recall@20}=\frac{|L_{20}\cap G|}{|G|}.
\]

The denominator is the number of withheld items, not the number 20. If three items are withheld and two appear in the first 20 positions, Recall@20 is \(2/3\). If all three appear there, it is 1, even if they occur at positions 18, 19, and 20. If none appear, it is 0. Thus recall answers: **“How many of the missing items did we recover within the first 20?”** It does not distinguish their positions inside those 20.

NDCG@20 additionally considers position. For rank \(r\in\{1,\ldots,20\}\), define

\[
\operatorname{rel}_r=\begin{cases}
1,&\text{the item at rank }r\text{ is in }G,\\
0,&\text{otherwise}.
\end{cases}
\]

The discounted cumulative gain and its ideal value are

\[
\operatorname{DCG@20}=\sum_{r=1}^{20}\frac{\operatorname{rel}_r}{\log_2(r+1)},\qquad
\operatorname{IDCG@20}=\sum_{r=1}^{\min(|G|,20)}\frac{1}{\log_2(r+1)}.
\]

The ideal value is the DCG obtained if the withheld items occupy the earliest possible positions. The normalized score is

\[
\operatorname{NDCG@20}=\frac{\operatorname{DCG@20}}{\operatorname{IDCG@20}}.
\]

The denominator makes the ideal ranking equal to 1, even when queries have different numbers of withheld items. A relevant item at rank 1 contributes \(1/\log_2(2)=1\), while one at rank 10 contributes \(1/\log_2(11)\approx0.289\). Irrelevant items contribute zero. Therefore NDCG answers: **“Were the missing items placed near the top?”** as well as rewarding their recovery within the cutoff.

### Worked comparison: equal recall, different ordering

Suppose a query has two withheld items, A and B. Both systems place both items in the top 20:

| System | Rank of A | Rank of B | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| First | 1 | 10 | 1 | \(\dfrac{1+1/\log_2(11)}{1+1/\log_2(3)}\approx0.790\) |
| Second | 9 | 10 | 1 | \(\dfrac{1/\log_2(10)+1/\log_2(11)}{1+1/\log_2(3)}\approx0.362\) |

Both recover 2 of 2 withheld items, so their recall is the same. The first receives a higher NDCG because it places A at rank 1 instead of rank 9. The ideal denominator is \(1+1/\log_2(3)\approx1.631\): it corresponds to placing the two relevant items at ranks 1 and 2. If neither withheld item appears in the top 20, both metrics are zero for this query.

### How the slide's single number is produced

The evaluator computes each metric separately for each test query, then takes the **arithmetic mean over queries**. For example, if three queries have Recall@20 values \(1, 1/2, 0\), the reported recall for those queries is \((1+1/2+0)/3=0.5\). This gives each query equal weight; it is not one global ratio of all recovered items to all withheld items. The benchmark values on the following slides are averages over their supplied test queries, not the result of the illustrative query above.

The project evaluator uses binary relevance, one-based ranks, and discount \(1/\log_2(r+1)\). The ideal DCG includes at most 20 relevant positions. When scores tie, it orders tied candidates by ascending catalogue item ID. These details make the calculation reproducible; they are not assumptions about how much more valuable one item is than another.

**A sentence to present:** “Recall@20 tells us what fraction of the withheld items we found in the first 20. NDCG@20 also rewards finding them earlier in that list.”

## Slide 9 — What does the POG result establish?

The test asks the algorithm to recover withheld items from each supplied query. The ranking includes the full eligible catalogue, rather than a small multiple-choice candidate list.

Recall@20 is the fraction of a query's withheld items found among its first 20 recommendations, averaged over queries. If a query has two withheld items and the top 20 contains one, its recall is one half. NDCG@20 additionally rewards placing the recovered items closer to the top.

| Method | Recall@20 | NDCG@20 |
|---|---:|---:|
| CLHE, published | 0.02840 | 0.01930 |
| RaMen, published | 0.03750 | 0.02260 |
| BundleFuse, completed research-profile run | 0.05446 | 0.03609 |

The relative Recall improvement over RaMen uses its score as the denominator:

\[
\frac{0.0544583333-0.0375}{0.0375}\times100\approx45.2\%.
\]

The absolute difference is about 0.01696, or 1.696 percentage points. It is not an increase of 45 percentage points, and it is not 45% of all queries suddenly becoming perfectly solved.

Our value uses all 4,000 supplied test queries after scalar settings were selected on the supplied validation queries. The neural numbers are the published values from RaMen Table 2. They were not regenerated during this presentation task. Aggregate published values cannot supply query-paired significance tests against those methods.

**A sentence to present:** “On this supplied POG test set, our Recall@20 is 0.05446, compared with the published RaMen value of 0.03750; the relative increase is 45.2%.”

This is evidence of benchmark performance. It does not identify which operation caused the cross-model difference or establish performance on every domain.

## Slide 10 — What does the Spotify result add?

Spotify applies the same completion task to playlists, with 254,155 catalogue items and 4,000 supplied test queries. The comparison is

| Method | Recall@20 | NDCG@20 |
|---|---:|---:|
| CLHE, published | 0.10810 | 0.18060 |
| RaMen, published | 0.10910 | 0.18820 |
| BundleFuse, completed research-profile run | 0.15565 | 0.25793 |

\[
\frac{0.1556463970-0.1091}{0.1091}\times100\approx42.7\%.
\]

The absolute Recall difference is about 0.04655, or 4.655 percentage points. NDCG also increases, so the comparison is not limited to the count of recovered items within the cutoff.

The purpose of showing this second domain is to test whether the construction works beyond clothing outfits. It does not imply that the exact same source weights were selected in both domains. Available descriptors and fitted scalar settings differ by dataset.

Raw metric levels should not be used to conclude that one domain is intrinsically easier: playlist sizes and withheld-item counts differ from outfit queries. The statement supported here is that our reported score exceeds the published comparator on each of these two tasks.

**A sentence to present:** “The improvement is also observed for playlist completion, with 0.15565 Recall@20 versus the published 0.10910.”

## Slide 11 — What does the internal comparison explain?

The cross-model results do not tell us which of our calculations matter. This experiment asks a narrower question: how do systems selected from different allowed source collections perform on the same supplied queries?

| Allowed source collection / system | POG NDCG@20 | POG_DENSE NDCG@20 | Spotify NDCG@20 |
|---|---:|---:|---:|
| Historical only | 0.02094 | 0.25945 | 0.21170 |
| Shared only | 0.02929 | 0.07697 | 0.17191 |
| Combined, no fallback | 0.03394 | 0.27689 | 0.24320 |
| Full BundleFuse | 0.03609 | 0.28788 | 0.25793 |

“Historical only” allows the configured occurrence, propagation, association-profile, and popularity sources. “Shared only” allows the configured category, description, bridge, and metadata sources. Some of these also depend on training bundles; the labels describe which source collection was allowed, not disjoint underlying datasets. “Combined” makes both collections available, returns their weighted sum directly, and uses the pairwise form of the historical match. “Full” additionally allows the incidence-enhanced historical match, the agreement score, and the frequency-dependent fallback to be selected on validation. Which of those options is actually selected differs by dataset.

Each restricted system is selected on validation. We then evaluate it on the same test population as the other rows within that dataset. Thus this is a comparison of selected systems under different restrictions, not a claim that changing one coefficient in an otherwise frozen recipe produces the entire displayed difference.

### Equations for the four rows

Fix a query \(S\) and a candidate \(c\). All quantities below are evaluated for that same \((c,S)\), except popularity, which is query-independent before standardization. Let \(z_h\) be the **separately standardized** score of source \(h\), as defined on slide 8. Let \(\mathcal H\) and \(\mathcal D\) denote the source collections labelled historical and shared, respectively. For each row \(j\), \(w_h^{(j)}\) is that row's independently validation-selected weight; a source not selected has weight zero.

The first three rows use the **pairwise** historical source, disable the incidence and agreement options, and return the weighted sum directly:

\[
\begin{aligned}
\theta_{\mathrm{history}}(c\mid S)
  &=\sum_{h\in\mathcal H}w_h^{(H)}z_h,\\
\theta_{\mathrm{shared}}(c\mid S)
  &=\sum_{h\in\mathcal D}w_h^{(D)}z_h,\\
\theta_{\mathrm{combined}}(c\mid S)
  &=\sum_{h\in\mathcal H\cup\mathcal D}w_h^{(C)}z_h.
\end{aligned}
\]

The weights in one row are not assumed to equal those in another. The superscripts \((H),(D),(C)\) label the system; they do not mean powers. A shared source such as bridge can itself use training-bundle information; the source sets describe restrictions on scoring rules, not independent input datasets.

For the **full** row, define \(z_{\mathrm{pair}}\) as the standardized direct historical score; \(z_{\mathrm{inc}}\) as the separately standardized incidence-enhanced score; and \(z_R\) as the standardized optional agreement score. They are standardized separately, so one must not replace a mixture of them with one standardized raw mixture. The full score before fallback is

\[
\begin{aligned}
Y_c={}&\sum_{h\ne\mathrm{simp}}w_h^{(F)}z_h
+w_{\mathrm{simp}}^{(F)}
\bigl[(1-\alpha)z_{\mathrm{pair}}+\alpha z_{\mathrm{inc}}\bigr]
+\gamma z_R.
\end{aligned}
\]

Here \(\alpha\in\{0,0.5,1\}\) selects or mixes the pairwise and incidence-enhanced versions, and \(\gamma\) is the selected agreement coefficient. Any other sources whose implementation has separate pair and incidence versions follow the same interpolation. The three displayed full recipes use `simp` but do not select `diff2`, so the displayed formula covers their selected sources.

Let \(\mathcal P\) be the selected fallback-source list, and let \(z_{\mathrm{pop}}\) be standardized \(\log(1+f_c)\). The actual fallback construction for these recipes is

\[
P_c=\sum_{h\in\mathcal P}w_h^{(F)}z_h
+\mathbf 1[w_{\mathrm{pop}}^{(F)}>0]\,w_{\mathrm{pop}}^{(F)}z_{\mathrm{pop}}
+\gamma z_R.
\]

The extra agreement term also enters the fallback when selected; this matters on Spotify. Popularity is included when it has positive weight, even though it is not in the saved `prior` list. For the selected sources shown here, the fallback uses the same corresponding weights as the full score. The implementation also has a default fallback weight when a listed prior source has no full-score weight, but that case does not arise in these three full recipes.

The returned full score is

\[
\theta_{\mathrm{full}}(c\mid S)=
\frac{f_c}{f_c+\kappa}Y_c+
\frac{\kappa}{f_c+\kappa}s_0P_c.
\]

If the selected \(\kappa\) is zero, the implementation returns \(Y_c\) directly; all three full recipes shown here select positive \(\kappa\). The table below expands the **actual saved weights**, so the difference between systems can be read without interpreting shorthand such as “combined.”

### The actual selected equations: POG

Write \(z_{cat}\) for category relations, \(z_{soft}\) for soft category/cluster relations, \(z_{style}\) for residual description similarity, \(z_{bridge}\) for bridged descriptions, and \(z_{pop}\) for standardized log popularity. Then

\[
\begin{aligned}
\theta_H &=0.25z_{\mathrm{pair}}+0.25z_{pop},\\
\theta_D &=1.0z_{cat}+0.25z_{soft}+0.25z_{style}+0.5z_{bridge},\\
\theta_C &=0.5z_{cat}+0.25z_{style}+0.25z_{bridge}+0.25z_{\mathrm{pair}}.
\end{aligned}
\]

For the full row, \(Y_c=\theta_C\), while

\[
P_c=0.5z_{cat}+0.25z_{style}+0.25z_{bridge},\qquad
\theta_F=\frac{f_cY_c+0.5(1.04240243)P_c}{f_c+0.5}.
\]

The selected full model has \(\alpha=0\) and \(\gamma=0\). Thus the last two rows have the same pre-fallback score; the final interpolation is the change.

### The actual selected equations: POG_DENSE

Let \(z_{ppmi}\) denote the standardized direct positive-PMI association score and \(z_{svd}\) the standardized similarity score computed from truncated PPMI association profiles. Then

\[
\begin{aligned}
\theta_H&=1.5z_{\mathrm{pair}}+0.5z_{pop}+0.5z_{ppmi}+0.25z_{svd},\\
\theta_D&=0.25z_{cat}+0.25z_{soft}+0.25z_{bridge},\\
\theta_C&=0.5z_{cat}+0.25z_{bridge}+0.5z_{\mathrm{pair}}
          +0.25z_{pop}+0.25z_{svd}.
\end{aligned}
\]

The full configuration selects \(\alpha=1\), \(\gamma=0\):

\[
\begin{aligned}
Y_c&=0.5z_{cat}+0.25z_{bridge}+0.5z_{\mathrm{inc}}
          +0.25z_{pop}+0.25z_{svd},\\
P_c&=0.5z_{cat}+0.25z_{bridge}+0.25z_{pop},\\
\theta_F&=\frac{f_cY_c+0.5(1.46055159)P_c}{f_c+0.5}.
\end{aligned}
\]

Relative to \(\theta_C\), full substitutes \(z_{\mathrm{inc}}\) for \(z_{\mathrm{pair}}\) **and** applies the interpolation. Its difference from the combined row therefore cannot be ascribed to the interpolation alone.

### The actual selected equations: Spotify

Let \(z_{homo}\) denote the artist/album relation score and \(z_{audio}\) the audio-style description score. Then

\[
\begin{aligned}
\theta_H&=0.25z_{\mathrm{pair}}+1.0z_{ppmi}+0.25z_{svd},\\
\theta_D&=0.25z_{homo}+1.5z_{bridge},\\
\theta_C&=0.25z_{\mathrm{pair}}+1.0z_{pop}+1.0z_{ppmi}+0.5z_{svd}\\
&\quad+0.25z_{homo}+0.25z_{audio}+0.25z_{bridge}.
\end{aligned}
\]

The full configuration selects \(\alpha=1\), \(\gamma=0.2\):

\[
\begin{aligned}
Y_c&=0.25z_{\mathrm{inc}}+1.0z_{pop}+1.0z_{ppmi}+0.5z_{svd}\\
&\quad+0.25z_{homo}+0.25z_{audio}+0.25z_{bridge}+0.2z_R,\\
P_c&=0.25z_{homo}+0.25z_{audio}+0.25z_{bridge}+1.0z_{pop}+0.2z_R,\\
\theta_F&=\frac{f_cY_c+5(1.82610667)P_c}{f_c+5}.
\end{aligned}
\]

Here full changes the pairwise source to the incidence-enhanced source, adds agreement **to both** \(Y_c\) and \(P_c\), and applies the frequency interpolation. The agreement kernel has dataset-specific settings; it should not be described using the POG dissimilarity formula without checking the music configuration.

All coefficients above come from the frozen configuration JSON files, not from fitting on the test set. The NDCG values in the slide are obtained after these scores rank the eligible candidates; they are not the values of the score equations themselves.

The pattern has a concrete reading:

- On POG, allowing only the shared collection gives 0.02929, above the historical-only 0.02094. Nevertheless, allowing both gives 0.03394, above either restriction.
- On POG_DENSE, the ordering of the restricted collections reverses: historical-only is 0.25945 and shared-only is 0.07697. Allowing both still increases the score to 0.27689.
- On Spotify, allowing both also improves over each restricted collection.

This supports making both collections available and selecting their weights. It does not prove each source is useful individually, or that every proposed mathematical interpretation is uniquely responsible for the result.

### What exactly changes between the last two rows?

**Combined, no fallback** returns the selected weighted sum directly: \(\theta(c\mid S)=Y_c\). For these three saved configurations, it uses the pairwise historical score and does not add the optional agreement score. The frequency-dependent interpolation is disabled.

**Full BundleFuse** selects from the same broad source collections while also allowing the incidence-enhanced historical score, the optional agreement score, and the frequency-dependent interpolation. In the saved configurations below, the scalar weights assigned to the named base sources happen to be the same as in the corresponding combined configuration. The historical form, agreement setting, and fallback are separate settings:

| Dataset | Combined, no fallback | Full BundleFuse |
|---|---|---|
| POG | Pairwise historical score; no agreement; no fallback | Same pairwise score and no agreement, with frequency-dependent fallback (\(\kappa=0.5\)) |
| POG_DENSE | Pairwise historical score; no agreement; no fallback | Incidence-enhanced historical score and fallback (\(\kappa=0.5\)); no agreement |
| Spotify | Pairwise historical score; no agreement; no fallback | Incidence-enhanced historical score, agreement score with coefficient 0.2, and fallback (\(\kappa=5\)) |

The incidence-enhanced score increases a training bundle's contribution when it contains multiple observed query items together. The agreement score uses candidate–seed affinities and relationships between pairs of seeds; it does not require all those items to have occurred in one training bundle.

For POG, the two saved recipes use the same source weights and historical form, so their displayed difference reflects the addition of the selected fallback. For POG_DENSE and Spotify, several settings change simultaneously. **Do not attribute the full difference between those rows to the fallback alone.** Use a matched comparison with the historical and agreement settings held fixed to study the fallback separately. The last row is higher in all three columns, which supports the complete selected procedure under this protocol; it does not establish an isolated effect of each option.

The underlying saved settings can be checked in the [POG combined](../../results/publication_mechanisms/pog/combined.json) and [full](../../results/publication_mechanisms/pog/full.json) configurations, the [POG_DENSE combined](../../results/publication_mechanisms/pog_dense/combined.json) and [full](../../results/publication_mechanisms/pog_dense/full.json) configurations, and the [Spotify combined](../../results/publication_mechanisms/spotify/combined.json) and [full](../../results/publication_mechanisms/spotify/full.json) configurations.

The full POG_DENSE Recall@20 is **0.39522**. It is the research-profile result associated with these studies; the earlier reorganized-default run has different settings and approximately 0.39194 Recall@20.

**A sentence to conclude:** “The two allowed score collections have different strengths across datasets. Making both available gives better selected systems here, and the complete method produces the highest displayed NDCG in each column.”

## Questions that require a concrete answer

### What exactly is the proposed contribution?

An explicit estimator for bundle completion: it computes weighted matches against recorded training bundles, optionally rewards multiple matches in the same row, combines these with specified attribute and item-relation scores, and uses a frequency-based interpolation at the end. The empirical contribution is that this construction performs strongly without gradient-training a downstream scoring model.

The strongest mathematical distinction for the incidence part is that its candidate–seed–seed counts cannot in general be recovered from candidate–seed pair counts alone. This is a distinction from a restricted pair-statistics representation, not a proof of novelty relative to every earlier higher-order model. It also cannot explain a result in a recipe where that part has zero weight.

### Is the claim simply that summing more signals helps?

The experiment compares particular restricted and combined source collections, not an arbitrary increase in signal count. Some allowed sources receive zero weight. A defensible answer points to the calculated quantities and the selected recipes. Adding a source is useful only if validation and evaluation support its contribution; the name of the source is not a justification by itself.

### Why might direct counts beat learned representations?

The count calculation uses recorded occurrences without fitting an approximation to each observed relation in a low-dimensional prediction model. That can be useful when those recorded occurrences are predictive. But we have not established that low-dimensional approximation causes CLHE or RaMen's score difference. State the preserved information and measured ranking results separately from that untested causal explanation.

### Is “no parameters” the same as “nothing is fitted”?

No. We compute matrices, may fit SVD/cluster representations, and select scalar settings on validation. The claim is zero gradient-trained parameters in the downstream scoring core. Stored vectors and matrices have memory costs; supplied pretrained features have upstream training costs.

### Does frequency measure reliability?

It measures appearances in training bundles. Using it to choose a coefficient is a prescribed rule motivated by the amount of recorded history. It does not prove that an item with ten appearances has more query-relevant information than one with two. The gate's exact arithmetic and the measured comparisons support discussing the rule without calling it a calibrated confidence estimate.

### Is the method faster, or better on every dataset?

Neither follows from the parameter count. Graph operations, preprocessing, and validation still take time; earlier project studies include faster cached learned rankers. The displayed neural comparison concerns POG and Spotify. It should not be extended to all datasets, input sparsities, or neural methods.

## Sources and details for follow-up questions

### Papers used for the neural table and reported scores

- [CLHE paper](https://arxiv.org/abs/2310.18770): architecture and objectives.
- [RaMen paper, including Table 2](https://arxiv.org/html/2507.14361v1): architecture and POG/Spotify published values.
- [Bundle-MLLM, Sections 3.3 and A.3](https://arxiv.org/html/2407.11712): LoRA stages and 7B backbone.

The CLHE preprint describes a different split ratio from the released dense/sparse variants used here. The deck accordingly uses POG_DENSE for an internal comparison, not a strict matched comparison with its published neural scores. Published-score comparisons also do not certify equality of every preprocessing choice or utilized input channel.

### Project evidence

- [Parameter audit](bundlefuse_manager_evidence.json): count definitions and checkpoint provenance.
- [POG results](../../results/publication_mechanisms/pog/REPORT.md).
- [POG_DENSE results](../../results/publication_mechanisms/pog_dense/REPORT.md) and [selected recipe](../../results/publication_mechanisms/pog_dense/full.json).
- [Spotify results](../../results/publication_mechanisms/spotify/REPORT.md).
- [Historical operators](../../bundlefuse/operators.py), [selected score computation](../../bundlefuse/selected_signals.py), and [score combination and selection](../../bundlefuse/fusion.py).

No new benchmark runs were performed for this presentation. All worked numerical examples in this guide are illustrative and are not substituted for benchmark measurements.

### Rebuild the existing slides

From the repository root, run twice:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf docs/slides/bundlefuse_manager.tex
```

On the current macOS installation, `/Library/TeX/texbin/pdflatex` is available if `pdflatex` is not on `PATH`.
