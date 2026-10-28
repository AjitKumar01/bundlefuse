# How BundleFuse Works — Step by Step, in Plain Words

This page explains what BundleFuse does without the maths-heavy language of the paper.
If you want the formulas, see [`paper/method.tex`](../paper/method.tex); if you want the code,
the core pieces are in [`bundlefuse/operators.py`](../bundlefuse/operators.py) and the full
pipeline is in [`bundlefuse/fullrank_eval.py`](../bundlefuse/fullrank_eval.py).

---

## The problem in one sentence

You are given **part of a bundle** (say, a shirt and a pair of jeans from an outfit, or three songs
from a playlist) and you must **guess the missing items** by ranking every item in the catalogue.

- The items you are given are called the **seeds** (written `S`).
- Each item we might recommend is called a **candidate** (written `c`).
- We are scored on whether the real missing items appear in our top 20 (Recall@20 / NDCG@20).

---

## The big idea

BundleFuse builds **two opinions** about every candidate and then decides **how much to trust each
one**:

1. **"What has history shown?"** — evidence from past bundles (which items appeared together).
   This is strong for popular items but says *nothing* about new or rare items.
2. **"What makes sense on paper?"** — a *prior* built from item categories and item content
   (text/image/audio features). This works even for an item nobody has ever bundled.

The final score blends the two, and the blend depends on **how often the candidate has been seen**:

```
final score = (1 - u) × history_score  +  u × prior_score
u           = kappa / (times_seen + kappa)
```

- An item seen many times → `u` is near 0 → trust history.
- An item never seen (`times_seen = 0`) → `u = 1` → trust the prior only.

That's the whole method. Everything below explains how each part is built. There is **no neural
network to train** in the default setup — it's all counting, simple linear algebra, and picking a
few knobs on a validation set.

---

## Step 1 — Count what goes with what

From the training bundles we build a table `B`: one row per bundle, one column per item, and a 1
where the item is in the bundle.

From it we get:

- **How often each item appears** (`freq`, called `f_c`). This number is used again in Step 5.
- **Co-occurrence** `C = Bᵀ B`: for every pair of items, how many bundles contain both.

> Toy example: if "white sneakers" and "blue jeans" appear together in 40 past outfits, their
> co-occurrence is 40.

Code: `SimplicialGraphOperator.__init__` in `operators.py`.

---

## Step 2 — Turn the counts into several "history" scores

For a query (the seeds `S`), each candidate gets several scores, each looking at history slightly
differently. All of them are just "add up how strongly the candidate is linked to each seed":

| Signal | Plain meaning |
|---|---|
| `cooc` | Raw count: how many times has `c` appeared with my seeds? (Kept raw on purpose — **popularity is useful signal** here.) |
| `pair` | Same idea, but down-weights huge bundles and super-common items so they don't drown everything else. |
| `diff2` | "Friend of a friend": items that go with items that go with my seeds (a two-step walk). |
| `ppmi` / `csvd` | Co-occurrence corrected for chance ("do these appear together *more than expected*?"), plus a compressed version of it. |
| `pop` | How popular the candidate is overall. |

**Weakness:** for an item that has never appeared in a training bundle, every one of these scores
is zero. That's what Steps 3 and 4 fix.

Code: `score_all_signals` in `fullrank_eval.py`.

---

## Step 3 — Build a "makes sense" prior from categories

Using only item **categories** (shirt, shoes, rock song, ...), we learn three things from the
training bundles:

1. **Complementary categories** (`cat`). Which categories tend to go together? Shirts go with
   trousers; trousers go with shoes. A candidate gets a boost if its category pairs well with the
   seeds' categories.
2. **"Don't give me a second one" penalty** (slot saturation). We measure how often a bundle
   contains *two* items of the same category. For categories where that almost never happens
   (e.g. two pairs of shoes in one outfit, < 15% of the time), a candidate in the same category as
   a seed gets **penalised**. So if you already have shoes, we stop suggesting more shoes.
3. **Finer sub-types** (`softcat`). Coarse categories hide differences (dress shoes vs. running
   shoes). We cluster items by their features into ~400 fuzzy sub-groups and repeat the
   "which groups go together" idea at that finer level.

For **music** the logic flips: playlists *like* similarity, so instead of a penalty we reward
same-artist / same-album candidates (`homo`).

Code: `DualSpectrumOntology` in `operators.py`.

---

## Step 4 — Add a "style" prior from item content

Each item has a feature vector (text, image, or audio embedding). Comparing these directly mostly
tells you "both are shoes", which Step 3 already knows. So:

1. **Remove the category part.** Subtract the average vector of the item's category. What's left
   is the item's *style* within its category (e.g. "minimalist", "sporty").
2. **Borrow style from neighbours** (`bridge`). For each known item, mix in the style of the items
   it is usually bundled with. Now a brand-new item can be matched against the *typical company*
   a seed keeps, not just the seed itself.
3. Score each candidate by how similar its style is to the seeds' style (`style`, `bridge`).

Code: `BilateralContentBridge` in `operators.py`.

---

## Step 5 — Put every score on the same scale

Each signal has its own units (counts, log-ratios, cosine similarities...). For every query we
**z-normalise** each signal over the catalogue (subtract the mean, divide by the standard
deviation) so they can be added together fairly.

Code: `zcols` in `operators.py`.

---

## Step 6 — Pick the weights on the validation set

We need to decide how much each signal counts. This is done with a simple **greedy search** on the
validation split:

1. Start with all weights at 0.
2. For each signal, try weights `{0, 0.25, 0.5, 1, 1.5}` and keep whichever raises validation
   NDCG@20.
3. Repeat a few rounds until nothing improves.
4. Drop optional signals that don't really help.

This gives:

- **`history_score` (Y)** — the weighted sum of all chosen signals.
- **`prior_score` (μ₀)** — the weighted sum of only the signals that work *without history*
  (category, sub-type, style, bridge). It is rescaled once, on validation, so its spread matches
  `Y`. That scale factor is then **frozen**, so a test query's score never depends on which other
  queries happen to be in the same batch.

We checked that this selection does not overfit (the validation→test gap is small; see
[`RESULTS.md`](../RESULTS.md)).

Code: `fit_simba` in `fullrank_eval.py`.

---

## Step 7 — Blend history and prior by how familiar the item is (the key step)

Now apply the formula from "The big idea":

```
final(c) = (1 - u_c) × Y(c)  +  u_c × μ₀(c)
u_c      = kappa / (f_c + kappa)
```

Worked example with `kappa = 1`:

| Candidate | Times seen `f_c` | `u_c` | Which opinion wins |
|---|---|---|---|
| Popular jeans | 99 | 0.01 | 99% history |
| Niche belt | 1 | 0.50 | half and half |
| Brand-new jacket | 0 | 1.00 | 100% prior |

`kappa` is chosen from a small grid (`0, 0.05, 0.5, 1, 5`, plus one data-driven guess) by whatever
gives the best validation score.

**Why this matters:** without this step, a never-seen item gets a very negative history score and
is buried at the bottom, even if it's a perfect stylistic fit. With it, new items compete on the
prior. This "cold-start shrinkage" is the main reason BundleFuse wins on sparse datasets
(+2–5 points of cold-item Recall@20).

Code: `EmpiricalBayesRouter.conjugate_posterior` in `operators.py`; `apply_simba_full` in
`fullrank_eval.py`.

---

## Step 8 — Rank and return

For each query:

1. Set the seeds' own scores to −∞ (never recommend something already in the bundle).
2. Sort all remaining items by final score.
3. Return the top 20.

Code: `eval_fused` in `fullrank_eval.py`.

---

## The whole pipeline in one picture

```
 training bundles ──► counts (Step 1) ──► history signals (Step 2) ─┐
                                                                    ├─► z-normalise (5)
 item categories  ──► category prior (Step 3) ──────────────────────┤
 item features    ──► style prior    (Step 4) ──────────────────────┘
                                                                    │
                                     pick weights on validation (6) │
                                                                    ▼
                 Y = history score            μ₀ = prior score
                         \                       /
                          \   u = κ/(f + κ)     /
                           ▼                   ▼
                  final = (1-u)·Y + u·μ₀    (Step 7)
                                │
                                ▼
                    drop seeds, take top 20 (Step 8)
```

---

## What was tried and did *not* help (kept only to report honestly)

- **Higher-order ("three-way and up") co-occurrence** (`delta_inc`, `rao`). The idea: reward a
  candidate that appeared together with *two or more* of the seeds in the same past bundle. In
  practice this adds almost nothing (≤ 0.34 points) once the simple pairwise signals are in place.
  The code still computes them, and the weight search usually sets them to ~0.
- **A small neural set-encoder** (`contrast`, ~0.3M parameters). Off by default (`ENCODER=1` to
  enable). Turning it off ties or wins everywhere and is ~7× faster.
- **Estimating `kappa` from a statistical model.** The textbook Empirical-Bayes formula would let
  us compute `kappa` from the data, but its assumptions are violated on 7 of 8 datasets (the
  estimated noise comes out negative). So `kappa` is simply picked on validation, and we say so.

---

## Glossary

| Term | Meaning |
|---|---|
| Bundle | A set of items that belong together (an outfit, a playlist, a shopping basket). |
| Seeds `S` | The part of the bundle you are given. |
| Candidate `c` | Any catalogue item we might recommend. |
| `f_c` | How many training bundles contain the candidate. |
| Cold item | An item with `f_c = 0` — never seen in training. |
| Prior `μ₀` | A score that needs no history (categories + content). |
| `kappa` | How many "sightings" an item needs before we trust history as much as the prior. |
| Recall@20 | Fraction of the missing items that show up in our top 20. |
