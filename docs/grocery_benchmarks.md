# Dunnhumby and Instacart: benchmark landscape for DIALS

As of 2026-11-12.

## Summary

There is no shared leaderboard for within-basket completion on Dunnhumby. Published grocery results come from two task
families with different protocols, so their numbers are not comparable to ours.

- **Next-basket recommendation** predicts a user's whole next basket from past baskets. It is personalized by
  definition and is the dominant benchmark family.
  - The reference protocol is the Reality Check (Li et al., ACM TOIS 2023).
  - TIFU-KNN, UP-CF@r and DNNTSP lead.
  - Personal purchase frequency comes within about 2% of them.
- **Within-basket recommendation (our task)** completes the current basket from the items already in it.
  - On Dunnhumby, the only well-known result is triple2vec (CIKM 2018), scored with AUC over sampled negatives.
  - Instacart is the common benchmark: triple2vec, NPA (2024) and BaskERT (2025) all use it.
- **Both families agree that repeat purchases dominate grocery.** That matches our personalized Dunnhumby result: the
  household's own history adds about 45% to every method.

## Next-basket recommendation

On the [Reality Check](https://arxiv.org/html/2109.14233v2) protocol, the best methods reach N@20 of .285 on Dunnhumby
(UP-CF@r) and .432 on Instacart (TIFU-KNN).

- **Protocol:** users split 72 / 8 / 20 into train, validation and test, repeated 5 times. The target is each user's
  last basket, ranked over the full catalogue.
- **Data:**

  | Dataset | Users | Items | Items per basket | Repeat items |
  |---|---|---|---|---|
  | Dunnhumby | 22,530 | 3,920 | 7.45 | 41% |
  | Instacart | 19,435 | 13,897 | 9.61 | 60% |

| Method | Kind | Dunnhumby R@20 | Dunnhumby N@20 | Instacart R@20 | Instacart N@20 |
|---|---|---|---|---|---|
| G-TopFreq | global popularity | .126 | .119 | .099 | .096 |
| P-TopFreq | the user's own frequency | .303 | .270 | .431 | .394 |
| GP-TopFreq | personal, then global, frequency | .314 | .274 | .435 | .396 |
| UP-CF@r | user kNN with recency | .319 | .285 | .459 | .422 |
| TIFU-KNN | time-decayed frequency + user kNN | .319 | .280 | .471 | .432 |
| DNNTSP | deep model | .324 | .284 | .442 | .400 |
| Sets2Sets | deep model | .255 | .188 | .308 | .240 |
| CLEA | deep model | .187 | .186 | .151 | .159 |
| Beacon | deep model | .135 | .124 | .105 | .101 |
| Dream | deep model | .130 | .121 | .100 | .096 |

- **Repeat purchases carry the leaders.** They account for over 97% of the leading methods' performance on both
  datasets.
- **Deep models add little.** DNNTSP's R@10 gain over GP-TopFreq is about 1.3% on Dunnhumby and 1.9% on Instacart.
- **No settled 2025 leader.** Later papers claim gains over TIFU-KNN, for example
  [CL4NBR](https://assets-eu.researchsquare.com/files/rs-4398733/v1_covered_139b3df7-cd94-4172-ace2-a7f37d0228bc.pdf)
  and a [time-aware TIFU-KNN](https://publications.hse.ru/pubs/share/direct/840141443.pdf). None of them reports on the
  same split.

## Within-basket recommendation (our task)

Three papers report basket completion on grocery data. Only triple2vec includes Dunnhumby, and none uses a protocol we
can match number for number.

| Paper | Datasets | Protocol | Headline result |
|---|---|---|---|
| [triple2vec](https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm18a.pdf) (Wan, McAuley et al., CIKM 2018) | Dunnhumby, Instacart | personalized (user in the model); AUC and NDCG | NDCG triple2vec .145 / .127; with adaLoyal (repurchase prior) .185 / .279 (Dunnhumby / Instacart) |
| [NPA, Neural Pattern Associator](https://arxiv.org/abs/2401.16433v2) (Luo et al., 2024) | Instacart, Spotify, a private grocery dataset | baskets split 60 / 20 / 20; random items as context, the rest held out | Instacart R@10: NPA .129, VAE-CF .095, item-item CF .092, co-purchase .089, BERT4Rec .075 |
| [BaskERT](https://papers.tinbergen.nl/25071.pdf) (van Maasakkers, Donkers, Fok, Tinbergen discussion paper, Dec 2025) | Instacart | one masked item ranked among 100 sampled candidates; products clustered to 9,407 | accuracy 37.0%, against CBOW 31.2%, item-item CF 27.2%, popularity 19.7% |

- **triple2vec's repurchase effect matches ours.** adaLoyal more than doubles Instacart NDCG (.127 to .279), as R does
  for every method on Dunnhumby.
- **NPA's protocol is the closest to ours:** context items drawn at random from the basket, recall and NDCG over the
  catalogue. Its preprocessing is not fully specified, so its numbers are a reference, not a like-for-like target.
- **BaskERT can't be placed next to us.** It ranks among 100 sampled candidates and reports accuracy and mean rank,
  not full-catalogue R@20.

## Our Dunnhumby results

Without personalization, DIALS ties the best simple baselines. With the household repurchase term R it gains 45%, and
the best baseline given the same term ties it again.

**Our protocol:**
- 21,744 items; baskets of 3–20 items, split 70 / 10 / 20 into train, validation and test, with 5,707 test queries.
- Half of each basket is given as seeds, and the rest is ranked over the full catalogue.
- Household histories use training baskets only, and training baskets are scored with leave-one-out.

| Method | Personalized | R@20 | N@20 | N@20, history before the query basket only |
|---|---|---|---|---|
| DIALS + R (trained, 3 seeds) | yes | .2094 | .1482 | .1305 |
| Blend + R (best baseline on validation) | yes | .2057 | .1484 | .1334 |
| Frozen DIALS + R at inference | yes | .2094 | .1475 | .1328 |
| ItemKNN + R | yes | .1966 | .1440 | .1288 |
| EASE + R | yes | .1932 | .1426 | .1276 |
| SGFCF + R | yes | .1972 | .1417 | .1274 |
| RP3β + R | yes | .1973 | .1414 | .1260 |
| Repurchase only (about P-TopFreq) | yes | .1647 | .1212 | .0982 |
| DIALS | no | .1490 | .1023 | |
| Blend (content + ItemKNN) | no | .1441 | .1044 | |

- **DIALS + R vs DIALS:** +.0458 N@20 [.042, .050], p = .0008.
- **DIALS + R vs blend + R:** −.0002 N@20 [−.003, .003], p = 1.0; +.0037 R@20, p = .41.
- **Not comparable to the Reality Check numbers.** That task predicts a whole basket from history alone, on a
  different subset (3,920 items).

## Next steps

1. **Add the next-basket leaders as personalized baselines on Dunnhumby.**
   - Methods: GP-TopFreq, TIFU-KNN and UP-CF@r, each also blended with R as above.
   - Cost: all are CPU-only, about an hour in total.
   - Why: reviewers know these methods, and our "repurchase only" is close to P-TopFreq.
2. **Add Instacart as a dataset.**
   - It is the shared within-basket benchmark of triple2vec, NPA and BaskERT, and is on Kaggle as the Instacart Market
     Basket Analysis release.
   - It has real product names (content) and an aisle → department hierarchy (labels), the two signals DIALS uses
     beyond co-occurrence.
   - Evaluate under an NPA-style protocol, and report NPA's published numbers as a reference, not a like-for-like
     target.

**Wording for the paper:**
- Dunnhumby is the warm control, where DIALS ties the strongest collaborative baselines.
- Personalization is a separate table, where DIALS + R matches the best personalized baseline.
- Don't claim that DIALS wins the personalized task.

## Sources

- [A Next Basket Recommendation Reality Check](https://arxiv.org/html/2109.14233v2) (Li et al.,
  [ACM TOIS 2023](https://dl.acm.org/doi/10.1145/3587153))
- [Representing and Recommending Shopping Baskets with Complementarity, Compatibility, and Loyalty](https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm18a.pdf)
  (triple2vec, CIKM 2018)
- [Within-basket Recommendation via Neural Pattern Associator](https://arxiv.org/abs/2401.16433v2) (NPA, 2024)
- [What Did I Forget? Basket Analysis for Large Assortments Using Transformers](https://papers.tinbergen.nl/25071.pdf)
  (BaskERT, Tinbergen Institute discussion paper TI 2025-071)
- [Contrastive Learning for Next-basket Recommendation](https://assets-eu.researchsquare.com/files/rs-4398733/v1_covered_139b3df7-cd94-4172-ace2-a7f37d0228bc.pdf)
  (CL4NBR; claims only, not checked)
- [Time-Dependent Next-Basket Recommendations](https://publications.hse.ru/pubs/share/direct/840141443.pdf) (claims
  only, not checked)
- Our results: `results/dials_user/comparison_dunnhumby_repurchase.json`,
  `results/dials_user/repurchase_baselines/summary.json`
