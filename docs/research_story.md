# A coherent research story for BundleFuse

This is a proposed scientific framing, grounded in the implementation and completed
experiments. It does not change the method or designate a new simplified model as
experimentally verified.

## The central claim

**Bundle completion can be approached by explicitly estimating compatibility from
past bundles and item attributes, then adjusting the use of those estimates to each
candidate's amount of training history. A task-trained neural encoder is not necessary
to obtain strong accuracy in the evaluated original-input conditions.**

The scientific question is: *How far can explicit compatibility estimates take us,
before learning a new representation becomes necessary?*

This gives the components a common purpose. The paper is about estimating one quantity,
the suitability of a candidate for a partial bundle. It is not a collection of eleven
separate contributions. It is also not a claim that neural representations are generally
unnecessary, that shrinkage alone explains the gains, or that every available component
is required.

Suggested title: **BundleFuse: Explicit Compatibility Estimation for Bundle Completion**.

## Begin with the prediction problem

Take a partially observed outfit containing a shirt and trousers. A second shirt may
be very similar to an observed item, while a pair of shoes may better complete the
outfit. Similarity and compatibility therefore need not coincide. In a playlist,
similarity can itself provide useful compatibility evidence. The task does not dictate
one universal relationship between items.

Past bundles offer direct evidence of what belongs together. But their coverage is
uneven: some candidate items occur frequently, some rarely, and some not at all. Even
two individually frequent items may never have appeared together. A missing recorded
pair consequently need not mean that the items are incompatible. This is a motivation
for transferring evidence, not a theorem that an unobserved pair should rank highly.

The reader should now see why neither recorded overlap alone nor feature similarity
alone is a complete answer. BundleFuse uses both, while keeping their roles explicit.

## Two sources of compatibility evidence, one rule for combining them

### First, preserve direct evidence from past bundles

Ask: *Have this candidate and the observed items belonged together before, or are they
connected through the observed item relationships?*

Raw shared counts, popularity-adjusted association, weighted bundle overlaps,
propagation, and a compressed collaborative view are different summaries of the same
historical evidence. They address different distortions and levels of smoothing. They
are not independent measurements, new data sources, or individually novel mechanisms.
Household co-purchase is an additional historical relation in retail; it does not make
the research retail scorer personalized to the current household.

### Second, share evidence beyond the exact observed item pairs

Ask: *What can be inferred from item groups and descriptions when the exact pair has
little or no recorded support?*

Category relations pool observations across related items. Text, image and audio
features provide descriptions that remain available when an item's bundle history is
absent. Graph-adjusted features combine descriptions with historical associations.
These are ways to extend the coverage of compatibility estimates, rather than a new
motivation for every modality.

The distinction is functional, not a clean separation into independent data sources.
Category relations and graph-adjusted features also depend on training history.
The method does not turn a cold item into an item with observed collaborative edges.

### Third, account for the candidate's amount of history

Let \(S\) be the observed set, \(c\notin S\) a candidate, and \(f_c\) the number of
training bundles containing it. Let \(Y(c\mid S)\) be the combined compatibility
score. Let \(P(c\mid S)\) be a restricted fallback assembled mainly from category
and descriptive evidence. Let \(s_0\geq 0\) rescale that fallback and
\(\kappa\geq 0\) control its strength. These settings are selected using validation
examples. The final ranking score is

\[
\theta(c\mid S)=
\begin{cases}
Y(c\mid S), & \kappa=0,\\[3pt]
\displaystyle
\frac{f_c}{f_c+\kappa}Y(c\mid S)
+\frac{\kappa}{f_c+\kappa}s_0P(c\mid S),&\kappa>0.
\end{cases}
\]

When the fallback is enabled, a candidate with no training appearances receives the
fallback score; its weight decreases as appearances increase. This expresses an
assumption about the usefulness of history. Frequency is a proxy for the amount of
evidence, not a measured or calibrated uncertainty. The score is not an estimated
probability of bundle membership.

Crucially, \(Y\) already contains descriptive evidence, and \(P\) can also contain
selected popularity and set-agreement terms. They overlap. Calling them a pure
collaborative expert and a pure content expert would misdescribe the method. The
frequency rule is candidate dependent; the ordinary source weights are fixed after
validation, rather than being a learned query-dependent gate.

## Explain the additional signals only after this structure is clear

The full method retains several possible summaries of each evidence source. Each raw
score is standardized across the catalogue for the current observed set, so scores
with different units can be combined. This standardization does not use the hidden
answers, make the sources independent, or calibrate their predictive reliability.

| Role in the prediction | Examples in the implementation | Why more than one summary exists |
|---|---|---|
| Historical item relations | Weighted bundle overlap, raw co-occurrence, positive PMI, propagation, collaborative SVD; household co-purchase for retail | Recorded counts, popularity adjustment, indirect support and smoothing are different views of history. |
| Evidence shared through attributes | Category relations, within-group similarity, text/image/audio features, graph-adjusted features | Different available attributes can extend coverage beyond exact observed pairs. |
| Candidate prevalence and backoff | Selected popularity score; the frequency-dependent final mixture | A popularity score ranks candidates by prevalence; the mixture changes how the two estimates are used. These are distinct roles. |
| Optional use of several observed items together | Incidence surplus and Rao agreement | Some candidates have support involving multiple observed items that a sum of separate matches can miss. |

These are explanatory groups, not four independently implemented experts or a newly
tested four-block architecture. The equations and full component inventory still need
to be disclosed. Domain-specific menus and search budgets should be reported explicitly.

The incidence surplus increases support from a past bundle when it contains at least
two observed items. The Rao term evaluates a candidate's support across pairs of
observed items using their specified dissimilarities. Explain these exact operations
before using a label such as "set agreement." Neither is the main explanation for the
large observed gains: removing both preserves approximately 92–100% of the Recall@20
advantage over HardSwitchHybrid in the supplied-split component study.

The selected models are substantially smaller than the complete menu suggests:

| Historical supplied-split model | Nonzero ordinary weights | Incidence surplus selected? | Rao term selected? |
|---|---:|---|---|
| POG | 4 | No | No |
| POG_dense | 5 | Yes | No |
| Spotify | 7 | Yes | Yes |
| Spotify_sparse | 3 | No | Yes |

These counts describe the archived supplied-split models, not every configuration or
later reorganized run. Present the available menu separately from the active model.
Validation pruning helps explain the inventory; it does not by itself establish that
the search is optimal or that every retained signal is necessary.

## Make the evidence follow the same argument

The main experimental narrative should answer these questions in order:

1. **Can explicit compatibility estimates compete?** Present the benchmark accuracy
   comparisons with their actual input/split scope. Separate published reference scores
   from methods evaluated locally with paired query results.
2. **Is this merely the benefit of having both history and attributes?** Present the
   information-matched SimpleHybrid and HardSwitchHybrid controls. BundleFuse exceeds
   both in Recall@20 and NDCG@20 on all four supplied splits after the planned Holm
   correction. This rules out those tested simple explanations; it does not rule out
   every possible hybrid.
3. **Is a trained encoder necessary to exploit the supplied information?** Present the
   tested feature-matched learned ranker. BundleFuse wins in the five original-input
   conditions across all three learned seeds. State the architecture/search limits and
   the two budget-limited Spotify confirmation fits.
4. **What does each refinement buy?** Shrinkage has corrected evidence on three
   supplied datasets, but not POG_dense. Higher-order gains are clearest on music;
   bridging is conditional and can hurt POG NDCG. This supports selective components,
   rather than a claim that all mechanisms are universally beneficial.
5. **Where does the approach stop working as well?** Report the one-seed POG_dense
   control win and the learned ranker's sparse chronological-retail Recall wins.
   Candidate training frequency and the number of observed query items are different
   quantities; handling the former does not guarantee adaptation to the latter.

Do not headline inference speed: the tested cached learned rankers are faster. Explain
the absence of a task-trained neural encoder separately from offline preprocessing,
validation search, upstream feature extraction, and serving cost.

The compact-model pilot is secondary evidence: engineered features help that small
learner, yet its accuracy remains below BundleFuse. It motivates preserving explicit
relations, but does not prove that low rank or representation compression causes the
gap. Keep it as a diagnostic or follow-up study rather than a second main method.

## The honest novelty boundary

The existence of multiple information sources is not new. CLHE already combines
multimodal features, user feedback and bundle composition. RaMen already combines
direct collaborative relations, multimodal information and group structure. The
contrast is how the prediction is constructed, not whether those sources are used:

> Can explicit compatibility scores, combined with candidate-frequency backoff,
> provide a competitive alternative to training a new bundle representation?

Likewise, behavioral estimates backed off toward descriptive priors have prior work
in product search. The shrinkage equation alone is not a defensible novelty claim.

The contribution to argue is the particular explicit scoring construction, its
competitive results against the tested controls, and the evidence identifying when
its refinements help. This framing is coherent; its ultimate novelty still requires a
full related-work comparison. A new vocabulary for a signal ensemble is not itself a
new method.

One useful follow-up, if strengthening the method claim, is a prespecified family-level
study: history only, attribute-sharing only, both with fixed fusion, and both with
candidate-frequency backoff, before adding the set refinements. Use matched validation
budgets, distinguish frozen-weight removals from retuned comparisons, and freeze the
protocol before looking at additional test results. Existing component ablations do
not establish all of those contrasts. No new experiment is claimed here.

Also analyze score differences by candidate training frequency, and set-refinement
effects by observed-set size. These are mechanism checks, not guaranteed outcomes.
If they fail to support the proposed explanations, narrow the causal story accordingly.
Because previous test sets have already been inspected, label new analyses as follow-up
and seek independent confirmation before treating them as confirmatory evidence.

## Proposed opening paragraph

Completing a bundle requires finding items that belong with those already present.
Past bundles provide direct evidence of such compatibility, but this evidence is
uneven: some items appear often, while others have little or no recorded history.
Item descriptions and category relations can extend coverage beyond observed pairs,
although resemblance alone does not always imply suitability for the same bundle.
We study whether these forms of evidence can support accurate bundle completion
without training a new neural encoder. BundleFuse constructs explicit compatibility
scores from historical relations and item attributes, and uses candidate training
frequency to adjust their combination with a restricted fallback. Experiments compare
this approach with published reference scores, information-matched hybrid controls and
a trained feature-matched ranker, and identify both useful refinements and failure
conditions.

## Proposed order of the paper

Prediction problem and uneven evidence → historical compatibility estimates → sharing
evidence through attributes → standardization and combination → candidate-frequency
backoff → optional multi-item refinements → accuracy and matched controls → component
diagnosis → sparse-input limits and computational costs.

One framework figure should follow that same flow. Show training bundles and catalogue
attributes feeding the combined estimate and restricted fallback, with candidate
frequency controlling their final mixture. Show the optional set refinements as additions
to those estimates. Avoid a diagram that incorrectly makes the two estimates disjoint.

## Evidence and reading

The completed-study facts above refer to archived commit
`4516eaba8079e655409027681ecb798d7ad45293`, available in Git through
`evaluation-reliability-experiments`, especially:

- `results/provided_splits/ANALYSIS.md` and the four `selection.json` files;
- `results/component_study/ANALYSIS.md`;
- `results/generalization_study/ANALYSIS.md`;
- `results/compact_pilot/ANALYSIS.md`.

They should not be mixed with later configurations as if all came from one run.
The local [claims ledger](../paper/CLAIMS.md) records the permitted scope and benchmark
qualifications, including the CLHE preprint/release split discrepancy.

Primary related work:

- [CLHE: Leveraging Multimodal Features and Item-level User Feedback for Bundle Construction](https://arxiv.org/abs/2310.18770).
- [RaMen: Multi-Strategy Multi-Modal Learning for Bundle Construction](https://arxiv.org/abs/2507.14361).
- [Addressing Cold Start in Product Search via Empirical Bayes](https://www.amazon.science/publications/addressing-cold-start-in-product-search-via-empirical-bayes).
