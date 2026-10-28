# How BundleFuse Fills a Partial Bundle

This document walks through **exactly** what BundleFuse does, in the order the code does it, using
one small example from start to finish. Every number in the example was produced by running the
real code in [`bundlefuse/operators.py`](../bundlefuse/operators.py) (script:
[`docs/toy_example.py`](toy_example.py)), and the key ones are also worked out by hand.

**Contents**

- [0. The task](#0-the-task)
- [1. The running example](#1-the-running-example)
- [Phase A — Preparation (once, from the training bundles)](#phase-a--preparation-once-from-the-training-bundles)
- [Phase B — Filling one partial bundle](#phase-b--filling-one-partial-bundle)
- [Phase C — How the knobs are chosen (once, on validation)](#phase-c--how-the-knobs-are-chosen-once-on-validation)
- [Differences for music datasets](#differences-for-music-datasets)
- [Parts that exist in the code but are off or near-zero](#parts-that-exist-in-the-code-but-are-off-or-near-zero)

---

## 0. The task

- **Input:** a *partial bundle* — a few items that belong together (the **seeds**, written `S`).
- **Output:** every other item in the catalogue, sorted from "most likely to complete this bundle"
  to "least likely". The top 20 are the recommendation.

BundleFuse has three phases:

| Phase | When | What it produces |
|---|---|---|
| **A. Preparation** | Once, from the training bundles | Count tables, category tables, style vectors |
| **B. Filling** | For every partial bundle | A score for every candidate item, then a ranking |
| **C. Tuning** | Once, on a held-out validation set | The few knobs Phase B needs (weights, κ, a scale) |

Phase C uses Phase B, so this document explains B before C.

---

## 1. The running example

**Catalogue** (10 items). Each item has a category and a 3-number content vector. In real data the
content vector is a text/image embedding with hundreds of numbers; here think of the three numbers
as *how casual / how sporty / how formal* the item looks.

| # | Item | Category | Content [casual, sporty, formal] |
|---|---|---|---|
| 0 | white sneakers | shoes | [0.9, 0.3, 0.2] |
| 1 | blue jeans | bottoms | [0.9, 0.1, 0.3] |
| 2 | grey hoodie | tops | [0.8, 0.5, 0.0] |
| 3 | white t-shirt | tops | [0.9, 0.2, 0.2] |
| 4 | black cap | hats | [0.7, 0.6, 0.0] |
| 5 | denim jacket | outerwear | [0.8, 0.1, 0.4] |
| 6 | running shoes | shoes | [0.2, 0.95, 0.0] |
| 7 | leather belt | accessories | [0.4, 0.0, 0.9] |
| 8 | track pants | bottoms | [0.2, 0.9, 0.1] |
| 9 | canvas sneakers | shoes | [0.9, 0.2, 0.2] — **new item, in no training bundle** |

**Training bundles** (7 past outfits):

| Bundle | Items | Size |
|---|---|---|
| B0 | white sneakers, blue jeans, grey hoodie | 3 |
| B1 | white sneakers, blue jeans, white t-shirt, black cap | 4 |
| B2 | blue jeans, white t-shirt, leather belt | 3 |
| B3 | running shoes, track pants, grey hoodie | 3 |
| B4 | running shoes, track pants, black cap | 3 |
| B5 | white sneakers, denim jacket, white t-shirt | 3 |
| B6 | blue jeans, denim jacket, grey hoodie | 3 |

**The partial bundle to fill:** `S = {blue jeans, grey hoodie}`.

**The hidden truth** (what the real outfit also contained): `{canvas sneakers, black cap}`.
Note that canvas sneakers is a brand-new item — no past outfit contains it.

> How partial bundles are made during evaluation (`_mask` in `fullrank_eval.py`): take a
> validation/test bundle, shuffle it, keep half of it (rounded, at least 1) as the seeds, and hide
> the rest as the answer.

---

## Phase A — Preparation (once, from the training bundles)

### A1. Membership table and item frequencies

Build a table with one row per training bundle and one column per item, with a 1 where the item is
in the bundle (`B` in the code). From it:

- **Item frequency `f`** = how many training bundles contain the item.
- **Bundle size `|b|`** = how many items are in the bundle.

| Item | f |
|---|---|
| blue jeans | 4 |
| white sneakers, grey hoodie, white t-shirt | 3 |
| black cap, denim jacket, running shoes, track pants | 2 |
| leather belt | 1 |
| canvas sneakers | **0** (cold item) |

Code: `SimplicialGraphOperator.__init__`.

### A2. Item-pair tables

1. **Co-occurrence `C`**: for each pair of items, the number of bundles containing both.
   Example: white sneakers & blue jeans are together in B0 and B1, so `C = 2`.
2. **PPMI**: "do these two appear together more than chance would predict?"
   `PPMI[i,j] = max(0, ln( C[i,j] × total / (row_i × row_j) ))`, where `row_i` is the sum of
   item i's row of `C` and `total` is the sum of all of `C`.
3. **SVD vectors (`csvd`)**: the PPMI table compressed into a short vector per item (up to 128
   numbers), so two items with similar PPMI rows get similar vectors.

### A3. Category tables

These use only categories, so they work even for brand-new items.

1. **Category complementarity (`Pcat`)**: count, for each pair of *different* categories, how many
   bundles contain both, and turn that into PMI the same way as above.
   Example — shoes & bottoms: bundles containing both = 4 (B0, B1, B3, B4); category totals are
   shoes 5, bottoms 6, all categories 22. So `ln(4 × 22 / (5 × 6)) = ln(2.93) = 1.076`.
2. **Duplicate rate per category**: how often a bundle has *two or more* items of that category.
   `dup_rate = (bundles with ≥2 of it + 0.5) / (bundles with ≥1 of it + 1)`.
   Example — shoes appear in 5 bundles, never twice: `(0 + 0.5) / (5 + 1) = 0.083`.

   | Category | dup_rate | Below 0.15? |
   |---|---|---|
   | shoes | 0.083 | yes → "one per outfit" category |
   | bottoms | 0.071 | yes |
   | tops | 0.071 | yes |
   | hats | 0.167 | no |
   | outerwear | 0.167 | no |
   | accessories | 0.250 | no |

   (Hats/outerwear/accessories are "no" only because they appear in so few bundles that the +0.5
   smoothing dominates. On real data this needs many bundles to matter.)
3. **Soft sub-types (`softcat`)** — real data only, not in the toy example: the content vectors are
   clustered into 400 groups, each item gets a soft membership in every group, and step 1 is
   repeated at the group level. This separates e.g. dress shoes from running shoes inside "shoes".

Code: `DualSpectrumOntology`.

### A4. Style vectors and bridge vectors

1. **Style vector** = the item's content vector **minus the average content vector of its
   category**, then scaled to length 1. This removes "it's a shoe" and keeps "*what kind* of shoe".

   | Item | Style vector |
   |---|---|
   | white sneakers | [ 0.755, −0.619,  0.217] |
   | canvas sneakers | [ 0.656, −0.732,  0.184] |
   | running shoes | [−0.701,  0.685, −0.199] |
   | blue jeans | [ 0.633, −0.753,  0.180] |
   | grey hoodie | [−0.267,  0.802, −0.535] |
   | black cap, denim jacket, leather belt | [0, 0, 0] — only item in its category, so nothing is left |

2. **Bridge vector** = the item's style vector plus half of the *average style of the items it is
   bundled with* (one step through the bundle table), scaled to length 1.
   A cold item has no bundle neighbours, so its bridge vector equals its style vector.
   The black cap has a zero style vector, so its bridge vector is purely its neighbours' style
   (running shoes, track pants → sporty).

Code: `BilateralContentBridge.disentangle`, `build_dense_bridge`.

---

## Phase B — Filling one partial bundle

Input: `S = {blue jeans, grey hoodie}`.

### B1. Compute every signal for every candidate

Each signal answers one question about candidate `c`. Rows marked *(seed)* are scored too but are
thrown out at the end.

**History signals** (need past bundles; always 0 for a never-seen item):

| Signal | Question it answers | Formula |
|---|---|---|
| `cooc` | How many times did `c` appear with my seeds? | `Σ_s C[c,s]` |
| `pair` | Same, but small bundles and rare seeds count more | `Σ_{bundles b containing c} (|b|+1)^−0.5 × Σ_{seeds s in b} (f_s+5)^−0.5` |
| `diff2` | Friend-of-a-friend: is `c` bundled with items that score high on `pair`? | one step of `pair` through the bundle table |
| `ppmi` | Does `c` appear with my seeds more than chance? | `Σ_s PPMI[c,s]` |
| `csvd` | Is `c`'s compressed PPMI vector similar to the seeds'? | `csvd_c · mean_s(csvd_s)` |
| `pop` | How popular is `c` overall? | `ln(1 + f_c)` |

**Prior signals** (need only category/content; work for new items):

| Signal | Question it answers | Formula |
|---|---|---|
| `cat` | Does `c`'s category go well with the seeds' categories, and are we avoiding a duplicate slot? | `Σ_s Pcat[cat(s), cat(c)]` + 0.5 × slot penalty |
| `softcat` | Same as `cat`, at the 400 soft sub-type level | (real data only) |
| `style` | Does `c`'s style match the seeds' style? | `style_c · mean_s(style_s)` |
| `bridge` | Does `c`'s style match the style of the company the seeds usually keep? | `bridge_c · mean_s(bridge_s)` |

**The slot penalty** (inside `cat`): for each seed whose category has `dup_rate < 0.15`, every item
in that same category gets `−(1 − dup_rate)`. Here jeans (bottoms) and hoodie (tops) trigger it, so
all bottoms and tops get `−0.929` ("you already have trousers and a top").

**Hand check — `pair` for white sneakers.** White sneakers are in B0, B1, B5.

- B0 (size 3) contains both seeds: `(3+1)^−0.5 × [(4+5)^−0.5 + (3+5)^−0.5] = 0.5 × (0.333 + 0.354) = 0.344`
- B1 (size 4) contains jeans: `(4+1)^−0.5 × 0.333 = 0.149`
- B5 contains no seed: 0
- Total = **0.493**

**Hand check — `cat` for canvas sneakers.** `Pcat[bottoms, shoes] + Pcat[tops, shoes] = 1.076 + 1.076 = 2.152`;
shoes is not a seed category, so no penalty. Total = **2.152**.

**Hand check — `style` for canvas sneakers.** Mean of the two seed styles =
`[0.183, 0.025, −0.178]`; dot product with `[0.656, −0.732, 0.184]` = **0.069**.

**All raw values for this query:**

| Item | cooc | pair | diff2 | ppmi | pop | cat | style | bridge |
|---|---|---|---|---|---|---|---|---|
| white sneakers | 3 | 0.493 | 0.113 | 0.555 | 1.386 | 2.152 | 0.084 | 0.213 |
| denim jacket | 2 | 0.343 | 0.087 | 0.981 | 1.099 | 1.905 | 0 | 0.230 |
| white t-shirt | 2 | 0.316 | 0.102 | 0.421 | 1.386 | 0.653 | −0.066 | 0.057 |
| black cap | 1 | 0.149 | 0.059 | 0.065 | 1.099 | 1.905 | 0 | −0.352 |
| running shoes | 1 | 0.177 | 0.047 | 0.693 | 1.099 | 2.152 | −0.076 | −0.200 |
| track pants | 1 | 0.177 | 0.047 | 0.693 | 1.099 | 0.653 | −0.066 | −0.190 |
| leather belt | 1 | 0.167 | 0.046 | 0.981 | 0.693 | 2.599 | 0 | 0.087 |
| **canvas sneakers** | **0** | **0** | **0** | **0** | **0** | 2.152 | 0.069 | 0.216 |
| *blue jeans (seed)* | 2 | 1.003 | 0.114 | 0.575 | 1.609 | 0.653 | 0.066 | 0.232 |
| *grey hoodie (seed)* | 2 | 0.864 | 0.098 | 0.575 | 1.386 | 0.653 | 0.066 | 0.232 |

Notice: every history column is **0** for canvas sneakers, but its prior columns (`cat`, `style`,
`bridge`) are among the best. That contrast is what the rest of the method is built around.

Code: `score_all_signals` in `fullrank_eval.py`.

### B2. Put every signal on the same scale (z-score)

The signals have different units, so each column is converted to a z-score **over all catalogue
items for this query**: subtract the column's average, divide by its standard deviation. After
this, 0 means "average for this query", +1 means "one spread above average".

| Item | z pair | z diff2 | z cat | z style | z bridge |
|---|---|---|---|---|---|
| white sneakers | 0.399 | 1.184 | 0.804 | 1.308 | 0.770 |
| denim jacket | −0.081 | 0.453 | 0.475 | −0.132 | 0.850 |
| leather belt | −0.651 | −0.728 | 1.397 | −0.132 | 0.164 |
| **canvas sneakers** | **−1.187** | **−2.010** | 0.804 | 1.051 | 0.781 |
| black cap | −0.707 | −0.334 | 0.475 | −0.132 | −1.937 |

Important side effect: a raw 0 becomes the **lowest** z-score. A new item is not just "unknown"
on history — it is actively pushed to the bottom.

Code: `zcols` in `operators.py`.

### B3. History score `Y` = weighted sum of all chosen signals

`Y(c) = Σ_k w_k × z_k(c)`, using the weights chosen in Phase C.

For this example we use these weights (on real data, Phase C picks them):
`pair 1.0, diff2 0.5, cat 1.0, style 0.5, bridge 0.5` (all others 0).

Canvas sneakers: `1.0×(−1.187) + 0.5×(−2.010) + 1.0×0.804 + 0.5×1.051 + 0.5×0.781 = −0.472`.

Despite great category and style scores, it lands 4th of the 8 candidates because the history
terms drag it down.

### B4. Prior score `μ0` = weighted sum of the prior signals only

`μ0(c) = scale × Σ_{k in prior signals} w_k × z_k(c)`

- Prior signals are `cat`, `softcat`, `style`, `imgstyle`, `bridge` (and `pop`/`rao` if they were
  given a weight — see Phase C).
- `scale` stretches the prior so its spread matches `Y`'s spread, so the two can be blended fairly.
  It is computed once on validation and then frozen. Here it is `std(Y) / std(prior) = 1.381`.

Canvas sneakers: `(0.804 + 0.5×1.051 + 0.5×0.781) × 1.381 = 2.376`.

### B5. Blend history and prior by how often the item has been seen

```
u(c)     = κ / (f_c + κ)
final(c) = (1 − u(c)) × Y(c) + u(c) × μ0(c)
```

`u` is "how much to trust the prior". With κ = 1:

| Item | f | u | Y | μ0 | final |
|---|---|---|---|---|---|
| white sneakers | 3 | 0.25 | 2.833 | 2.545 | 0.75×2.833 + 0.25×2.545 = **2.761** |
| **canvas sneakers** | 0 | **1.00** | −0.472 | 2.376 | **2.376** |
| leather belt | 1 | 0.50 | 0.398 | 1.951 | **1.175** |
| denim jacket | 2 | 0.33 | 0.979 | 1.153 | **1.037** |
| black cap | 2 | 0.33 | −1.434 | −0.773 | **−1.213** |
| running shoes | 2 | 0.33 | −1.482 | −0.716 | **−1.227** |
| white t-shirt | 3 | 0.25 | −1.539 | −2.493 | **−1.777** |
| track pants | 2 | 0.33 | −3.359 | −3.309 | **−3.342** |

A never-seen item (`f = 0`) gets `u = 1`, so its history score is ignored completely and it is
judged only on category and style.

Code: `EmpiricalBayesRouter.conjugate_posterior` (`operators.py`), `apply_simba_full`
(`fullrank_eval.py`).

### B6. Remove the seeds and rank

The seeds' scores are set to −1,000,000,000 so they can never be recommended, then all items are
sorted by `final`. (A tiny random number, 10⁻⁶, is added first to break exact ties.)

| Rank | With blending (κ = 1) | Without blending (κ = 0) |
|---|---|---|
| 1 | white sneakers | white sneakers |
| 2 | **canvas sneakers** ✔ | denim jacket |
| 3 | leather belt | leather belt |
| 4 | denim jacket | **canvas sneakers** ✔ |
| 5 | **black cap** ✔ | **black cap** ✔ |

Hidden truth = {canvas sneakers, black cap}. In the top 3, blending finds 1 of 2; without it, 0
of 2. The black cap stays at rank 5 either way: it is alone in its category (no style vector), its
usual companions are sporty (low bridge score), and it appeared with only one seed in history.
BundleFuse is not magic — it can only use the evidence it has.

Code: `eval_fused` in `fullrank_eval.py`.

---

## Phase C — How the knobs are chosen (once, on validation)

Phase B needs: the weights `w`, the higher-order mix, the prior `scale`, and `κ`. They are chosen
on validation partial bundles (made the same way as test ones) by maximising NDCG@20 — a measure
of how high the hidden items appear in the ranking. Code: `fit_simba` in `fullrank_eval.py`.

### C1. Signal weights (greedy search)

1. Start with every weight at 0.
2. For each signal in turn, try weights `0, 0.25, 0.5, 1, 1.5`, keeping a change only if NDCG@20
   improves by more than 0.0002.
3. Repeat the full pass up to 4 times, stopping early when a pass changes nothing.
4. **Prune:** for each optional signal (`bridge`, `softcat`, `contrast`, `homo`), try removing it;
   if NDCG@20 drops by less than 0.0003, remove it for good.

### C2. Higher-order mix

Try `alpha_inc ∈ {0, 0.5, 1}` and `g_rao ∈ {0, 0.2, 0.35}`:

```
Y = (1 − alpha_inc) × [weighted sum using plain pair/diff2]
  +      alpha_inc  × [weighted sum using pair+delta_inc / diff2+its diffusion]
  +          g_rao  × z(rao)
```

`delta_inc` adds extra credit when a past bundle contained the candidate **and two or more seeds
together** (in the example, white sneakers and denim jacket get 0.515 because B0 and B6 contain
both jeans and hoodie). `rao` rewards candidates supported by *several different-looking* seeds.
A combination is kept only if it beats the current best by more than 0.00005. In practice these
add almost nothing (see the last section).

### C3. Prior signals and scale

- **Which signals form the prior**: those among `cat, softcat, style, imgstyle, bridge, contrast`
  (music: `cat, homo, audiostyle, bridge, contrast`) that received a weight > 0 in C1. If none did,
  all of them are used with weight 0.5. `g_rao × z(rao)` is always added, and `pop` too if it got
  a weight.
- **Scale**: `std(Y) / std(prior)` over all validation queries. Frozen, so a test query's score
  never depends on which other queries are scored alongside it.

### C4. κ (how quickly to trust history)

Try `κ ∈ {0, 0.05, 0.5, 1, κ_est, 5}` together with scale ∈ {frozen scale, 1.0}, and keep whichever
gives the best NDCG@20 (must beat the previous best by > 0.00005, otherwise κ stays 0 = no
blending). `κ_est` comes from a statistical formula (`fit_variance_components`); its assumptions do
not hold on 7 of 8 datasets, so it is just one more grid candidate. If all seen items have the
same frequency, only κ = 0 is allowed.

Intuition for κ: it is the number of past bundles an item needs before history and prior get equal
trust (`u = 0.5` when `f = κ`).

---

## Differences for music datasets

For playlists (Spotify), "category" is the **artist**, and:

- No slot penalty — a playlist happily contains several songs by the same artist.
- `homo` replaces `softcat`: songs score higher if they share the **artist** with a seed, and even
  more (×2) if they share the **album**.
- `audiostyle` (audio features minus artist-cluster average) replaces `style`.

---

## Parts that exist in the code but are off or near-zero

| Part | Status | Why |
|---|---|---|
| `contrast` (~0.3M-parameter neural set encoder) | Off by default (`ENCODER=1` turns it on) | Ties or loses everywhere and is ~7× slower |
| `delta_inc`, `rao` (higher-order signals) | Computed; C2 usually gives them ~0 weight | Add at most +0.79 pp (≤ +0.26 pp on 7 of 8 datasets) |
| Statistical estimate of κ | Only one grid candidate | Its assumptions fail on 7 of 8 datasets |

---

## Reproducing the example

```bash
python3 docs/toy_example.py
```

The weights (B3) and κ = 1 in the example are set by hand to show the mechanism; everything else
is computed by the real code. With only 10 items and one query, real tuning (Phase C) would not be
meaningful.

The script runs the operators on NumPy arrays. The full evaluators run them on PyTorch tensors,
whose standard deviation divides by (n − 1) instead of n. That rescales every z-score by the same
factor, so the rankings are identical; only the absolute numbers differ slightly.
