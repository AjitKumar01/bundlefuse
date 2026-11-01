# Report 3 — Gate 0: candidate-coverage probe (the retrieve-then-rerank feasibility test)

**Status:** executed experiment. Verdict + the two self-corrections that got us to honest numbers.
**Artifacts:** `results/gate0/REPORT.md`, `results/gate0/coverage.json`, script
`reports/gate0_coverage.py`. Reuses the released provided splits and the real
`SimplicialGraphOperator`, so numbers align with `results/compact_pilot`.

---

## 0. What Gate 0 decides

The SCACS architecture (Report 2) is *retrieve-then-rerank*: shortlist candidates with cheap exact
retrievers, then rerank the shortlist with the small interaction-first network. A reranker can only
rank what the shortlist contains, so the shortlist's **recall ceiling** (fraction of held-out items
present in a size-M shortlist) is a hard upper bound on the whole system. Gate 0 asks: *does that
ceiling clear BundleFuse's actual R@20 at a usable shortlist size?* If no, retrieve-then-rerank is
dead and we keep only the (already banked, 3.9×/1.5×) scorer optimization.

Retrievers (each ranked by the evaluator's exact tie policy, score desc / item-id asc):
`cooc` = `C@x_S`, `wpair` = `W_pair@x_S` (full-rank collaborative), `catpmi` = cross-category
complementarity `Σ_s Pcat[cat_s, cat_c]` (BundleFuse's `cat` signal), `content` = content-embedding
cosine to the seed mean. `union_all` = present in the top-M of any leg (budget M per leg).

---

## 1. Results (recall ceiling vs shortlist size M)

BundleFuse actual R@20: **pog_dense 0.395**, **spotify 0.156** (same provided split).

**pog_dense — original** · collab-reachable 53.3% · hard ceiling (collab∨catPMI) 99.9%

| retriever | M=20 | M=50 | M=100 | M=200 | M=500 | M=1000 | M=2000 |
|---|---|---|---|---|---|---|---|
| cooc | 0.285 | 0.325 | 0.358 | 0.396 | 0.444 | 0.481 | 0.519 |
| wpair | 0.310 | 0.349 | 0.377 | 0.409 | 0.449 | 0.483 | 0.519 |
| catpmi | 0.003 | 0.007 | 0.013 | 0.023 | 0.061 | 0.131 | 0.228 |
| content | 0.000 | 0.001 | 0.002 | 0.004 | 0.010 | 0.018 | 0.036 |
| **union_all** | 0.316 | 0.356 | **0.390** | 0.429 | **0.492** | 0.558 | 0.641 |

**pog_dense — single_seed** · collab-reachable 29.9% · hard ceiling 98.4%

| retriever | M=20 | M=100 | M=500 | M=1000 | M=2000 | M=5000 |
|---|---|---|---|---|---|---|
| cooc | 0.212 | 0.246 | 0.283 | 0.302 | 0.328 | 0.394 |
| **union_all** | 0.217 | 0.256 | 0.314 | 0.365 | 0.443 | 0.601 |

**spotify — original** · collab-reachable 80.4% · hard ceiling 95.3%

| retriever | M=20 | M=50 | M=100 | M=200 | M=500 | M=1000 | M=2000 |
|---|---|---|---|---|---|---|---|
| cooc | 0.079 | 0.140 | 0.205 | 0.284 | 0.407 | 0.503 | 0.593 |
| wpair | 0.090 | 0.158 | 0.228 | 0.312 | 0.439 | 0.535 | 0.622 |
| catpmi | 0.007 | 0.015 | 0.026 | 0.047 | 0.100 | 0.169 | 0.277 |
| content | 0.005 | 0.011 | 0.019 | 0.033 | 0.063 | 0.102 | 0.163 |
| **union_all** | 0.103 | 0.180 | **0.261** | 0.358 | **0.508** | 0.622 | 0.728 |

**spotify — single_seed** · collab-reachable 38.0% · hard ceiling 76.5%

| retriever | M=20 | M=100 | M=200 | M=500 | M=1000 | M=2000 | M=5000 |
|---|---|---|---|---|---|---|---|
| cooc | 0.043 | 0.118 | 0.168 | 0.240 | 0.293 | 0.338 | 0.378 |
| **union_all** | 0.051 | 0.138 | 0.197 | 0.290 | 0.367 | 0.447 | 0.553 |

---

## 2. Verdict

**Overall: PASS, but split by domain.**

- **Spotify (music) — strong PASS.** Even a *single* collaborative leg of size 500 covers 0.41
  (3.3× BF's 0.156); `union_all@500 = 0.508`, `@1000 = 0.622`. Collaborative reachability is 80%
  and the hard ceiling 95%. There is large, unambiguous headroom for a reranker to beat BundleFuse.
  Single-seed still clears the bar (union@200 = 0.197 > 0.156). **This is where SCACS should be
  built and is most likely to win decisively.**

- **pog_dense (fashion) — qualified / marginal PASS.** `union_all` crosses BF's 0.395 only around
  `M=100` (0.390) and reaches ~0.49 at `M=500`. The headroom over BundleFuse is real but **thin**,
  and it comes almost entirely from the **collaborative legs** — the cheap metadata/content legs add
  only ~0.04. So on fashion a cheap retriever does little of the work; the reranker inherits the hard
  part. Single-seed POG is the weakest cell: collaborative reachability drops to 30% and `union_all`
  only overtakes 0.395 near `M≈1000`.

---

## 3. Three findings that matter beyond the gate

1. **Content similarity ≠ bundle complementarity.** The `content` leg is near-useless everywhere
   (≤0.06 at M=500). A shoe is not content-similar to its complementary handbag. This is direct
   evidence *for* BundleFuse's design choice to use graph-propagated `bridge` + fusion rather than
   raw content-embedding retrieval, and it kills the naive "just add a content ANN leg" fix.

2. **Category-PMI finds the right *category*, not the right *item*.** `catpmi` reaches 95–99.9% of
   held items in principle (the complementary category is almost always identifiable) yet covers only
   ~0.06–0.10 at M=500, because every item in the complementary category shares one category-level
   score. The hard, value-adding step is *within-category complement selection* — which is exactly
   what BundleFuse's fused collaborative + bridge signals do, and what any reranker must reproduce.

3. **The conjunction really does vanish at a single seed.** Collaborative reachability roughly halves
   from original to single-seed (POG 53→30%, Spotify 80→38%). This is the empirical fingerprint of
   Report 1's `Δ_inc ≡ 0 at |S|=1`: with one seed there is no joint co-occurrence evidence, so the
   model must fall back on priors/content — the regime the learned context-gate is meant to serve.

---

## 4. Honesty note — two self-corrections during this probe

The first two runs were wrong in ways worth recording, because the paper must not inherit them:

- **Tie-break inflation.** The initial rank used optimistic tie-breaking, so once M exceeded the
  positive-score pool, *all* zero-score held items were counted as "covered." This inflated POG
  `cooc` to 0.92 at M=5000 (vs 0.58 corrected). Fixed by ranking under the evaluator's exact policy
  (score desc, item-id asc).
- **Wrong metadata slate.** The slate first used *same-category* items — the opposite of fashion's
  cross-category complementarity — which made the slate useless on POG and hid the true structure.
  Fixed to category-PMI (`cat`), which revealed finding #2 above.

Both corrections lowered the numbers and sharpened the conclusions. The current figures use the
deterministic evaluator tie policy and the complementarity-correct retrievers.

Residual caveats: (a) `union_all@M` uses budget M per leg, so the realized shortlist is ≤4M items
(collaborative legs overlap heavily, so typically closer to ~2M) — read M as "per-leg depth," not
total shortlist size; single-leg `cooc@M` is an exact size-M shortlist. (b) The single-seed rows are
compared to BundleFuse's *original-mask* R@20; BundleFuse's own single-seed R@20 (not measured here)
is lower, so single-seed headroom is understated, not overstated. (c) This bounds recall only; it
says nothing about whether a reranker can realize the ceiling — that is Gate 1/2.

---

## 5. Recommendation

Proceed to **Gate 1 (the representational test), scoped to Spotify first**, where the ceiling gives
a reranker 3× room over BundleFuse. Build SCACS with a fixed gate and test whether interaction-first
attention over exact sparse features closes the compact two-tower's accuracy gap on the dense case.
Treat **pog_dense as the hard stress case**: the thin, collaboration-dominated headroom means a SCACS
win on fashion is *not* guaranteed and would be the stronger result if achieved. Retrieval for SCACS
should be collaborative-led (`cooc`/`wpair`), with category-PMI supplying the complementary
neighborhood; a content leg is not worth its cost. If Gate 1 on Spotify fails to beat the two-tower,
the interaction-first thesis is falsified and we stop with the scorer optimization + the two negative
results (compact pilot, and Report 1's representational analysis) as the honest contribution.
