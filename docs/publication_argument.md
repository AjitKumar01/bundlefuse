# Publication argument: begin with the literature

For the expanded component-by-component comparison, see the
[method-level prior-art review](method_prior_art_review.md).

Read [the literature-led LaTeX argument](publication_argument.tex)
([compiled PDF](../output/pdf/publication_argument.pdf)) first, then
[the mathematical specification](publication_argument_technical.tex).
The new argument develops the motivation before experimental evidence:

1. **Compatibility depends on the relationship required.** McAuley et al. distinguish
   substitutes and complements; resemblance alone is not a universal completion rule.
2. **Explicit recommendation is an established alternative.** EASE, side-information
   SLIM, and FEASE prevent us from claiming that shallow scoring or metadata fusion
   is itself new.
3. **Modern bundle models already recognize multiple sources and relationships.**
   CLHE and RaMen supply the immediate bundle-construction context. We should give
   them credit for that formulation, rather than claim they ignore these problems.
4. **Our question concerns the predictor's construction.** How far can specified
   historical, category and descriptive relation scores support bundle completion
   without a newly trained neural bundle encoder?
5. **Our contribution must be the particular score construction and its behavior.**
   The proposed combination and optional corrections require a precise account of
   what is inherited and what is added. Shrinkage also has prior work and is not
   a sufficient novelty claim.

The LaTeX document includes method-level primary references, a proposed introduction and
explicit novelty boundaries. This is a focused positioning argument, not a claim
that the entire literature has been exhausted.

## Supporting empirical argument

Revised after the completed POG and POG_dense mechanism studies. Spotify and
Spotify_sparse mechanism results are still pending. The earlier derivations and
proposed studies are retained in [the technical note](publication_argument_technical.md).

**Past bundles tell us which items have gone together. Item attributes help extend
that evidence beyond the exact pairs already observed. BundleFuse turns both into
comparable compatibility scores and combines them to complete a partial bundle.**

Start with a candidate item and an unfinished bundle. Historical associations provide
a natural first score: does this candidate tend to appear alongside the observed
items? But exact item histories have limited coverage, and raw counts are affected
by popularity. Categories and descriptions provide another way to estimate whether
the candidate fits. They can share evidence across items, although similarity alone
does not always imply suitability for the same bundle.

This gives the method two questions to answer:

1. What support comes from recorded item relationships?
2. What support can be shared through item categories and descriptions?

BundleFuse constructs explicit scores answering those questions, standardizes their
scales, and selects their combination on validation examples. These are overlapping
uses of information: category relations and graph-adjusted descriptions also use
training history. They are not independent experts.

The numerous signal names are alternative constructions within this argument.
Counts, weighted overlap and collaborative summaries describe historical relations.
Category relations and descriptive features extend their coverage. The paper should
introduce these roles before the full inventory, while disclosing every component,
domain-specific choice and selection budget.

The new evidence supports this core on both completed datasets:

| Recall@20 | POG | POG_dense |
|---|---:|---:|
| Historical evidence alone | 0.030875 | 0.375379 |
| Category/descriptive evidence alone | 0.044542 | 0.112917 |
| Combined evidence, without backoff or multi-item refinements | 0.051458 | 0.390928 |
| Full BundleFuse | 0.054458 | 0.395223 |

The combination beats each family alone on Recall@20 and NDCG@20 under the current
paired correction reserving the full 96-test family. Shared evidence is stronger on
POG; history is stronger on POG_dense. Their combination improves both. This supports
complementary predictive information in these fitted constructions. It does not
establish that every signal is necessary or identify dataset density as the sole
cause of the reversal.

Frequency-dependent backoff and multi-item terms should follow as refinements.
On POG, frequency backoff with matched fallback scale beats the tested constant
mixture in both metrics after correction. On POG_dense, that advantage is not
established. The full model still improves over the combined core on both datasets.
These findings support conditional refinements, rather than a universal claim that
candidate-frequency regularization is the central source of accuracy.

The publication question is:

> How far can explicit compatibility estimates from historical relations and item
> attributes take bundle completion without training a new neural bundle encoder?

The method contribution must be the particular construction of those estimates and
the evidence that it improves on relevant alternatives. Combining history and
attributes is already an established idea; the new results explain this method's
behavior but do not by themselves establish novelty. Earlier matched hybrid and
learned-ranker comparisons belong alongside these mechanism results, with their
recorded scope and protocol qualifications.

Suggested opening paragraph:

> Completing a bundle requires finding items that belong with those already present.
> Past bundles provide direct evidence of such relationships, but their coverage is
> uneven. Item categories and descriptions allow evidence to be shared beyond exact
> observed pairs. We study how far explicit estimates from these sources can support
> bundle completion without training a new neural encoder. BundleFuse constructs
> comparable compatibility scores, combines them using validation-selected weights,
> and includes frequency-dependent and multi-item refinements. Our experiments
> examine the contribution of each evidence family and the conditions under which
> the refinements improve ranking.

Suggested title: **BundleFuse: Explicit Compatibility Estimation for Bundle Completion**.

Evidence: [POG report](../results/publication_mechanisms/pog/REPORT.md),
[POG_dense report](../results/publication_mechanisms/pog_dense/REPORT.md), and their
adjacent `summary.json` files. Selection sensitivity remains a reported limitation;
these results do not establish uniform cold-start robustness or faster inference.
