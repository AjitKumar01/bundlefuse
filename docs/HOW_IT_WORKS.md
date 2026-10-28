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

Column abbreviations used in the item tables below:
sneak = white sneakers, jeans = blue jeans, hood = grey hoodie, tee = white t-shirt, cap = black
cap, jacket = denim jacket, run = running shoes, belt = leather belt, track = track pants,
canvas = canvas sneakers.

### A1. Membership table `B`, item frequencies, and the two weight lists

**Table `B`**: one row per training bundle, one column per item, 1 if the item is in the bundle.

| | sneak | jeans | hood | tee | cap | jacket | run | belt | track | canvas |
|---|---|---|---|---|---|---|---|---|---|---|
| **B0** | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **B1** | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| **B2** | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| **B3** | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| **B4** | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
| **B5** | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| **B6** | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |

**Item frequency `f`** = column sum of `B` (how many bundles contain the item), and the **item
weight** `(f + 5)^−0.5` used by `pair` and `diff2`. The +5 stops rare items from getting huge
weights.

| Item | f | Item weight (f+5)^−0.5 |
|---|---|---|
| white sneakers | 3 | 0.354 |
| blue jeans | 4 | 0.333 |
| grey hoodie | 3 | 0.354 |
| white t-shirt | 3 | 0.354 |
| black cap | 2 | 0.378 |
| denim jacket | 2 | 0.378 |
| running shoes | 2 | 0.378 |
| leather belt | 1 | 0.408 |
| track pants | 2 | 0.378 |
| canvas sneakers | **0** (cold item) | 0.447 |

**Bundle size `|b|`** = row sum of `B`, and the two **bundle weights**: `(|b|+1)^−0.5` (used by
`pair`) and `1/(|b|+1)` (used by the one-step spread in `diff2` and `bridge`). Big bundles get
smaller weights because "being in the same 20-item bundle" says less than "being in the same
3-item bundle".

| Bundle | Size | (\|b\|+1)^−0.5 | 1/(\|b\|+1) |
|---|---|---|---|
| B0, B2, B3, B4, B5, B6 | 3 | 0.500 | 0.250 |
| B1 | 4 | 0.447 | 0.200 |

Code: `SimplicialGraphOperator.__init__`.

### A2. Item-pair tables

**1. Co-occurrence `C`** = `Bᵀ B` with the diagonal set to 0: for each pair of items, the number
of bundles that contain both. Example: sneak & jeans are together in B0 and B1, so `C = 2`.

| | sneak | jeans | hood | tee | cap | jacket | run | belt | track | canvas | **row sum** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **sneak** | 0 | 2 | 1 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 7 |
| **jeans** | 2 | 0 | 2 | 2 | 1 | 1 | 0 | 1 | 0 | 0 | 9 |
| **hood** | 1 | 2 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 6 |
| **tee** | 2 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 7 |
| **cap** | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 5 |
| **jacket** | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| **run** | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 4 |
| **belt** | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| **track** | 0 | 0 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 4 |
| **canvas** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Sum of the whole table (`total`) = 48.

**2. PPMI** — "do these two appear together *more than chance would predict*?"

`PPMI[i,j] = max(0, ln( C[i,j] × total / (rowsum_i × rowsum_j) ))`

A value of 0 means "no more often than chance (or never)". Example — jeans & belt:
`ln(1 × 48 / (9 × 2)) = ln(2.667) = 0.981`. Jeans & cap: `ln(1 × 48 / (9 × 5)) = ln(1.067) = 0.065`
(barely above chance, because both are common).

| | sneak | jeans | hood | tee | cap | jacket | run | belt | track | canvas |
|---|---|---|---|---|---|---|---|---|---|---|
| **sneak** | 0 | 0.421 | 0.134 | 0.673 | 0.316 | 0.539 | 0 | 0 | 0 | 0 |
| **jeans** | 0.421 | 0 | 0.575 | 0.421 | 0.065 | 0.288 | 0 | 0.981 | 0 | 0 |
| **hood** | 0.134 | 0.575 | 0 | 0 | 0 | 0.693 | 0.693 | 0 | 0.693 | 0 |
| **tee** | 0.673 | 0.421 | 0 | 0 | 0.316 | 0.539 | 0 | 1.232 | 0 | 0 |
| **cap** | 0.316 | 0.065 | 0 | 0.316 | 0 | 0 | 0.875 | 0 | 0.875 | 0 |
| **jacket** | 0.539 | 0.288 | 0.693 | 0.539 | 0 | 0 | 0 | 0 | 0 | 0 |
| **run** | 0 | 0 | 0.693 | 0 | 0.875 | 0 | 0 | 0 | 1.792 | 0 |
| **belt** | 0 | 0.981 | 0 | 1.232 | 0 | 0 | 0 | 0 | 0 | 0 |
| **track** | 0 | 0 | 0.693 | 0 | 0.875 | 0 | 1.792 | 0 | 0 | 0 |
| **canvas** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**3. SVD vectors (`csvd`)** — the PPMI table compressed to a short vector per item with truncated
SVD (128 numbers on real data; here `10 − 2 = 8`, the most the code allows for 10 items), and each
vector scaled to length 1. Two items with similar PPMI rows get similar vectors: running shoes
and track pants have identical PPMI rows except for each other, so their vectors differ only in
`d3`.

| | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 |
|---|---|---|---|---|---|---|---|---|
| **sneak** | 0.493 | 0.605 | 0.000 | −0.413 | 0.014 | 0.401 | 0.146 | −0.195 |
| **jeans** | 0.460 | 0.664 | 0.000 | 0.467 | 0.259 | −0.149 | 0.009 | −0.200 |
| **hood** | 0.751 | 0.019 | 0.000 | −0.130 | −0.602 | 0.033 | 0.235 | 0.001 |
| **tee** | 0.391 | 0.662 | 0.000 | 0.581 | −0.207 | 0.081 | −0.091 | 0.115 |
| **cap** | 0.810 | −0.204 | 0.000 | −0.168 | −0.274 | −0.422 | −0.128 | −0.067 |
| **jacket** | 0.480 | 0.513 | 0.000 | −0.189 | 0.489 | −0.295 | 0.337 | 0.175 |
| **run** | 0.717 | −0.311 | 0.600 | 0.043 | 0.138 | 0.082 | −0.028 | 0.016 |
| **belt** | 0.302 | 0.645 | 0.000 | −0.677 | 0.038 | 0.025 | −0.172 | 0.046 |
| **track** | 0.717 | −0.311 | −0.600 | 0.043 | 0.138 | 0.082 | −0.028 | 0.016 |
| **canvas** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

(The exact numbers in each SVD column depend on the SVD routine; what matters is which items end
up close together.)

Code: `SimplicialGraphOperator.__init__`.

### A3. Category tables

These use only categories, so they work even for brand-new items.

**1. Bundle–category table `T`**: how many items of each category each bundle has.

| | shoes | bottoms | tops | hats | outerwear | accessories |
|---|---|---|---|---|---|---|
| **B0** | 1 | 1 | 1 | 0 | 0 | 0 |
| **B1** | 1 | 1 | 1 | 1 | 0 | 0 |
| **B2** | 0 | 1 | 1 | 0 | 0 | 1 |
| **B3** | 1 | 1 | 1 | 0 | 0 | 0 |
| **B4** | 1 | 1 | 0 | 1 | 0 | 0 |
| **B5** | 1 | 0 | 1 | 0 | 1 | 0 |
| **B6** | 0 | 1 | 1 | 0 | 1 | 0 |
| **bundles with ≥1** | 5 | 6 | 6 | 2 | 2 | 1 |
| **bundles with ≥2** | 0 | 0 | 0 | 0 | 0 | 0 |

Total of the "≥1" row = 22.

**2. Category co-occurrence** = for each pair of *different* categories, the number of bundles
containing both (diagonal set to 0):

| | shoes | bottoms | tops | hats | outerwear | accessories |
|---|---|---|---|---|---|---|
| **shoes** | 0 | 4 | 4 | 2 | 1 | 0 |
| **bottoms** | 4 | 0 | 5 | 2 | 1 | 1 |
| **tops** | 4 | 5 | 0 | 1 | 2 | 1 |
| **hats** | 2 | 2 | 1 | 0 | 0 | 0 |
| **outerwear** | 1 | 1 | 2 | 0 | 0 | 0 |
| **accessories** | 0 | 1 | 1 | 0 | 0 | 0 |

**3. Category complementarity `Pcat`** = PMI of that table, using the "≥1" counts as category
totals: `Pcat[g,h] = max(0, ln( cooc[g,h] × 22 / (count_g × count_h) ))`.
Example — shoes & bottoms: `ln(4 × 22 / (5 × 6)) = ln(2.93) = 1.076`.
Shoes & hats: `ln(2 × 22 / (5 × 2)) = ln(4.4) = 1.482` (hats are rare, so appearing with shoes in
both of their bundles is strong evidence).

| | shoes | bottoms | tops | hats | outerwear | accessories |
|---|---|---|---|---|---|---|
| **shoes** | 0 | 1.076 | 1.076 | 1.482 | 0.788 | 0 |
| **bottoms** | 1.076 | 0 | 1.117 | 1.299 | 0.606 | 1.299 |
| **tops** | 1.076 | 1.117 | 0 | 0.606 | 1.299 | 1.299 |
| **hats** | 1.482 | 1.299 | 0.606 | 0 | 0 | 0 |
| **outerwear** | 0.788 | 0.606 | 1.299 | 0 | 0 | 0 |
| **accessories** | 0 | 1.299 | 1.299 | 0 | 0 | 0 |

**4. Duplicate rate per category**: how often a bundle has *two or more* items of that category.
`dup_rate = (bundles with ≥2 + 0.5) / (bundles with ≥1 + 1)`, read off the bottom rows of `T`.
Categories below 0.15 are treated as "one per bundle" and trigger the slot penalty in Phase B.

| Category | Calculation | dup_rate | Below 0.15? | Penalty `−(1 − dup_rate)` |
|---|---|---|---|---|
| shoes | (0 + 0.5) / (5 + 1) | 0.083 | yes | −0.917 |
| bottoms | (0 + 0.5) / (6 + 1) | 0.071 | yes | −0.929 |
| tops | (0 + 0.5) / (6 + 1) | 0.071 | yes | −0.929 |
| hats | (0 + 0.5) / (2 + 1) | 0.167 | no | — |
| outerwear | (0 + 0.5) / (2 + 1) | 0.167 | no | — |
| accessories | (0 + 0.5) / (1 + 1) | 0.250 | no | — |

(Hats/outerwear/accessories are "no" only because they appear in so few bundles that the +0.5
smoothing dominates. On real data this needs many bundles to matter.)

**5. Soft sub-types (`softcat`)** — real data only, so there is no table for the example: the code
clusters content vectors into 400 groups, which needs far more than 10 items. Each item gets a
soft membership in every group, and steps 2–3 are repeated at the group level. This separates e.g.
dress shoes from running shoes inside "shoes".

Code: `DualSpectrumOntology`.

### A4. Style vectors and bridge vectors

**1. Content vectors scaled to length 1** (the input; each row divided by its length):

| | casual | sporty | formal |
|---|---|---|---|
| **sneak** | 0.928 | 0.309 | 0.206 |
| **jeans** | 0.943 | 0.105 | 0.314 |
| **hood** | 0.848 | 0.530 | 0 |
| **tee** | 0.954 | 0.212 | 0.212 |
| **cap** | 0.759 | 0.651 | 0 |
| **jacket** | 0.889 | 0.111 | 0.444 |
| **run** | 0.206 | 0.979 | 0 |
| **belt** | 0.406 | 0 | 0.914 |
| **track** | 0.216 | 0.970 | 0.108 |
| **canvas** | 0.954 | 0.212 | 0.212 |

**2. Category average** of those vectors (e.g. shoes = average of sneak, run, canvas):

| | casual | sporty | formal |
|---|---|---|---|
| **shoes** | 0.696 | 0.500 | 0.139 |
| **bottoms** | 0.580 | 0.538 | 0.211 |
| **tops** | 0.901 | 0.371 | 0.106 |
| **hats** | 0.759 | 0.651 | 0 |
| **outerwear** | 0.889 | 0.111 | 0.444 |
| **accessories** | 0.406 | 0 | 0.914 |

**3. Content minus category average** — this removes "it's a shoe" and keeps "*what kind* of shoe":

| | casual | sporty | formal |
|---|---|---|---|
| **sneak** | 0.232 | −0.191 | 0.067 |
| **jeans** | 0.364 | −0.433 | 0.103 |
| **hood** | −0.053 | 0.159 | −0.106 |
| **tee** | 0.053 | −0.159 | 0.106 |
| **cap** | 0 | 0 | 0 |
| **jacket** | 0 | 0 | 0 |
| **run** | −0.490 | 0.479 | −0.139 |
| **belt** | 0 | 0 | 0 |
| **track** | −0.364 | 0.433 | −0.103 |
| **canvas** | 0.258 | −0.288 | 0.073 |

Cap, jacket and belt are the only items in their categories, so they equal their category average
and nothing is left.

**4. Style vector** = row 3 scaled to length 1 (rows of zeros stay zero):

| | casual | sporty | formal |
|---|---|---|---|
| **sneak** | 0.755 | −0.619 | 0.217 |
| **jeans** | 0.633 | −0.753 | 0.180 |
| **hood** | −0.267 | 0.802 | −0.535 |
| **tee** | 0.267 | −0.802 | 0.535 |
| **cap** | 0 | 0 | 0 |
| **jacket** | 0 | 0 | 0 |
| **run** | −0.701 | 0.685 | −0.199 |
| **belt** | 0 | 0 | 0 |
| **track** | −0.633 | 0.753 | −0.180 |
| **canvas** | 0.656 | −0.732 | 0.184 |

Read it as "compared with other items of my category": jeans are the *less sporty* bottoms, the
hoodie is the *more sporty* top, canvas sneakers are the *less sporty* shoes.

**5. Neighbour style** = one step of spreading the style vectors through the bundle table, then
scaled to length 1. The one-step spread for item `c` is:

`spread(c) = (f_c+5)^−0.5 × Σ_{bundles b containing c} 1/(|b|+1) × Σ_{items i in b, i ≠ c} (f_i+5)^−0.5 × style_i`

So it is a weighted sum of the styles of the items `c` shares bundles with (using the weights from
A1). Hand check for the cap (in B1 and B4), casual column:
B1: `0.2 × (0.354×0.755 + 0.333×0.633 + 0.354×0.267) = 0.115`;
B4: `0.25 × (0.378×(−0.701) + 0.378×(−0.633)) = −0.126`;
`0.378 × (0.115 − 0.126) = −0.004`.

| | spread: casual | sporty | formal | → scaled to length 1: casual | sporty | formal |
|---|---|---|---|---|---|---|
| **sneak** | 0.040 | −0.060 | 0.023 | 0.531 | −0.792 | 0.302 |
| **jeans** | 0.038 | −0.028 | 0.008 | 0.795 | −0.581 | 0.173 |
| **hood** | 0.016 | −0.016 | 0.005 | 0.705 | −0.679 | 0.204 |
| **tee** | 0.076 | −0.075 | 0.022 | 0.699 | −0.687 | 0.200 |
| **cap** | −0.004 | −0.006 | 0.011 | −0.333 | −0.426 | 0.841 |
| **jacket** | 0.045 | −0.044 | 0.013 | 0.699 | −0.687 | 0.200 |
| **run** | −0.054 | 0.081 | −0.031 | −0.532 | 0.791 | −0.302 |
| **belt** | 0.031 | −0.055 | 0.025 | 0.460 | −0.805 | 0.375 |
| **track** | −0.059 | 0.076 | −0.032 | −0.583 | 0.748 | −0.317 |
| **canvas** | 0 | 0 | 0 | 0 | 0 | 0 |

**6. Bridge vector** = `style + 0.5 × neighbour style`, scaled to length 1.

| | casual | sporty | formal |
|---|---|---|---|
| **sneak** | 0.687 | −0.683 | 0.248 |
| **jeans** | 0.691 | −0.700 | 0.179 |
| **hood** | 0.134 | 0.724 | −0.677 |
| **tee** | 0.426 | −0.791 | 0.438 |
| **cap** | −0.333 | −0.426 | 0.841 |
| **jacket** | 0.699 | −0.687 | 0.200 |
| **run** | −0.648 | 0.724 | −0.235 |
| **belt** | 0.460 | −0.805 | 0.375 |
| **track** | −0.618 | 0.753 | −0.226 |
| **canvas** | 0.656 | −0.732 | 0.184 |

Three things to notice:

- **Canvas sneakers** (never in a bundle) have no neighbours, so their bridge vector equals their
  style vector.
- **Jacket and belt** have no style of their own, so their bridge vector is purely their
  neighbours' style.
- **The cap** is a warning case. Its neighbours pull in opposite directions (casual sneak/jeans/tee
  in B1, sporty run/track in B4), so the spread is almost zero (`[−0.004, −0.006, 0.011]`). Scaling
  that tiny leftover to length 1 blows it up into a strong "formal" direction that no neighbour
  actually has. This is a side effect of scaling near-zero vectors to length 1.

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
| `diff2` | Friend-of-a-friend: is `c` bundled with items that score high on `pair`? | the one-step `spread` from A4, applied to the `pair` scores instead of style vectors |
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
bridge vector points in a misleading "formal" direction (see the warning case at the end of A4), and
it appeared with only one seed in history.
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
