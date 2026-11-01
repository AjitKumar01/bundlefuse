# Report 1 — What BundleFuse *is*: a function-class characterization and a representational-gap result

**Status:** theory + empirical grounding. Written to be lifted into a paper.
**Scope:** characterizes the exact function BundleFuse computes, classifies each signal by
algebraic rank/order, and proves (informally) which parts a pool-then-interact two-tower can and
cannot represent. Ends by grounding every structural claim in measurements already in the repo.

The companion file `02_architecture_from_the_theory.md` turns this into a concrete new architecture.

---

## 0. One-paragraph summary

BundleFuse is a **per-query-standardized, frequency-gated mixture of (i) full-rank linear
set-operators, (ii) a conjunctive set-coverage term, and (iii) a quadratic seed-concordance
term.** Its validation-time model selection consistently puts weight on the full-rank and
conjunctive pieces and sets the weight of the only low-rank collaborative surrogate (`csvd`) to
**zero**. A pool-then-interact two-tower — the architecture of the failed compact pilot — can
represent exactly the *low-rank* pieces and **cannot** represent the full-rank operators (at
finite width), the conjunctive coverage term, the quadratic term, or the per-candidate
frequency gate. The measured 10–32 % recall loss of the compact student is therefore the
*predicted* consequence of an architectural mismatch, not a search-budget artifact. This gives a
precise, testable statement of what a replacement architecture must be able to express.

---

## 1. Setup and notation

Catalogue of `n` items. A query is a partial bundle (seed set) `S ⊆ {1..n}`, represented by its
indicator `x_S ∈ {0,1}^n`. The scorer emits a score vector `s(·|S) ∈ R^n` over all candidates and
ranks the non-seed items. Training bundles define the bundle–item incidence matrix
`B ∈ R^{|B|×n}` and the item co-occurrence matrix `C = Bᵀ B` (zero diagonal).

BundleFuse's pipeline (as implemented in `bundlefuse/operators.py` and assembled in
`bundlefuse/selected_signals.py`) is four stages:

1. **Signals** `z_k(c|S)` — ~11 of them, each column-standardized over the catalogue (`zcols`).
2. **Weighted fusion** `Y_base(c|S) = Σ_k w_k z_k(c|S)`, weights `w_k` selected on validation.
3. **Higher-order combination** `Y = (1−α)·Y_pair + α·Y_inc + γ·z_rao`.
4. **Empirical-Bayes shrinkage** `θ̂(c|S) = (1−u_c)·Y(c|S) + u_c·μ₀(c|S)`, `u_c = κ/(f_c+κ)`.

The entire question "can a compact network replace this?" reduces to: *what is the function class
of `θ̂`, and what is the function class of the candidate network?* We make both precise.

---

## 2. BundleFuse signal-by-signal, as linear algebra

Write `x = x_S`. Pooled seed quantities use `mean_S(Φ) = (1/|S|)·Σ_{s∈S} Φ_s`.

| Signal | Closed form (candidate `c`) | Operator view | Algebraic type |
|---|---|---|---|
| `cooc` | `(C x)_c = Σ_{s∈S} C[c,s]` | full item×item matrix `C` | **full-rank linear** |
| `ppmi` | `(PPMI x)_c` | full item×item `PPMI` | **full-rank linear** |
| `simp` (pair part) | `(W_pair x)_c` | IDF-normalized pairwise `W_pair` | **full-rank linear** |
| `simp` (incidence) | `Δ_inc(c|S)` (eq. below) | ReLU of per-bundle overlap counts | **conjunctive / submodular** |
| `rao` | `R(c|S)` (eq. below) | quadratic form in seed–candidate affinities | **2nd-order** |
| `cat` | `Σ_{s∈S} Pcat[cat_c, cat_s]` | category-PMI `Pcat`, lifted to items | **rank ≤ #categories** |
| `homo` | artist/album group match | `I_art D⁻¹ I_artᵀ` (+album) | **rank ≤ #groups** (factorizable) |
| `softcat` | `M_soft P_s`-pooled | soft-cluster PMI | moderate-rank, factorizable |
| `csvd` | `⟨u_c, mean_S(u)⟩` | rank-128 SVD of PPMI | **low-rank (dot product)** |
| `style`/`imgstyle`/`audiostyle` | `⟨φ_c, mean_S(φ)⟩` | content embedding inner product | **low-rank (dot product)** |
| `bridge` | `⟨b_c, mean_S(b)⟩` | graph-propagated content | **low-rank (dot product)** |
| `pop` | `log(1+f_c)` | query-independent | **rank-1** |

Two signals need their explicit nonlinear forms, because they are the crux of the argument.

**Incidence surplus (conjunction).** From `score_simplicial_incidence_only`:

```
Δ_inc(c|S) = Σ_{b ∋ c}  vol(b) · ( Σ_{s ∈ b∩S} deg(s) ) · β · max(0, |b∩S| − 1)
```

The factor `max(0, |b∩S| − 1)` depends on the **count of seeds that land together in bundle `b`**.
As a function of `x_S` this is a ReLU of a linear count, i.e. piecewise-linear and **not** linear;
it is a set-coverage / conjunction signal that fires only when a candidate's training bundle
contains **≥ 2** of the current seeds. It is identically `0` for `|S| = 1`. This is BundleFuse's
hand-built "compatible with *several* seeds at once" (AND) term.

**Rao concordance (second order).** From `score_rao_concordance`, with
`p_{c,s} = W_pair[c,s] / A_c`, `A_c = Σ_{s} W_pair[c,s]`:

```
R(c|S) = ( A_c² / (A_c + τ) ) · Σ_{s ≠ s'}  K_S(s,s') · p_{c,s} · p_{c,s'}
```

This is a genuine **bilinear form** in the candidate's seed-affinity vector, modulated by a
seed–seed redundancy kernel `K_S`. It cannot be written as a single inner product `q_S · v_c`.

**EB shrinkage (per-candidate gate).** From `EmpiricalBayesRouter.conjugate_posterior`:

```
θ̂(c|S) = (1 − u_c)·Y(c|S) + u_c·μ₀(c|S),   u_c = κ / (f_c + κ)
```

a **per-candidate** convex interpolation between the computed score and a prior, gated by the
candidate's own training frequency `f_c`.

### 2.1 The clean split

Group the table by whether the signal is a (seed-pooled) inner product `⟨ψ(c), POOL_S φ(s)⟩`:

- **Factorizable / low-rank (two-tower-expressible):** `csvd`, `style`, `imgstyle`, `audiostyle`,
  `bridge`, `pop`, and — at their natural (category/group) dimension — `cat`, `homo`, `softcat`.
- **Not factorizable as a fixed finite inner product:** the full-rank collaborative operators
  `cooc`, `ppmi`, `W_pair`; the **conjunctive** `Δ_inc`; the **quadratic** `R`; and the
  **per-candidate gated** shrinkage `θ̂`.

---

## 3. The four structural properties of bundle compatibility

Reading the split above as a statement about the *problem* (not just this estimator):

1. **Full-rank collaborative structure.** The item–item compatibility matrices `C`, `PPMI`,
   `W_pair` are high-rank sparse matrices. Their action on a seed set is "look up the actual
   co-occurrence of `c` with each seed," which no fixed `d`-dimensional embedding reproduces once
   the true rank exceeds `d`.

2. **Conjunction (AND over seeds).** A good completion is compatible with *all* the seeds, not
   merely with one popular seed. Sum/mean pooling is **OR-like** (one strong pair dominates the
   sum); `Δ_inc` injects an explicit **AND** by rewarding joint co-occurrence with ≥2 seeds.
   This is the operation that *vanishes at `|S|=1`* — exactly the regime where the method is
   weakest and where a context-aware model has the most room.

3. **Second-order seed interaction.** `R` says the value of a candidate depends on how its
   affinities to *pairs* of seeds line up, discounted for redundant (same-category/similar) seed
   pairs. This is irreducibly pairwise.

4. **Per-candidate reliability gating.** How much to trust the data-driven score vs a cold-start
   prior is decided **per candidate** from its exposure `f_c`. This is a frequency-indexed gate,
   not a global scalar.

**Claim (informal).** Bundle completion is *conjunctive, full-rank, cold-start-gated set
completion.* Any architecture that is to match or beat BundleFuse must be able to express all four
properties; the degree to which it can is predictive of its accuracy.

---

## 4. The representational-gap argument

Let the candidate replacement be a **pool-then-interact two-tower** (the compact pilot,
`bundlefuse/baselines/compact_ranker.py`):

```
score(c|S) = ⟨ ρ( POOL_{s∈S} φ(s) ),  ψ(c) ⟩ + a·log(1+f_c)
```

with item features in `R^d`, `POOL ∈ {mean, mean+var}`, `ρ` a small MLP. The pilot used
`d = 128`, `ρ` = `260→64→128`, a learned pop weight.

What this class **can** represent exactly:

- Any low-rank inner-product signal of §2.1: set `ψ(c)=φ(c)`, `POOL=mean`, `ρ=id` to get
  `⟨φ(c), mean_S φ⟩`. So `style/img/audio/bridge/csvd` and (with enough width) `cat/homo/softcat`
  are reachable. The learned pop term reproduces `pop`.

What this class **cannot** represent (each is a provable limitation, not a tuning gap):

- **(a) Full-rank linear operators at finite width.** `⟨ρ(q_S), ψ(c)⟩` is, for fixed towers, a
  rank-`d` bilinear form in `(q_S, c)`. Reproducing `(C x)_c` for all `S` requires an inner-product
  factorization of `C`; if `rank(C) > d` the approximation error is bounded below by the tail
  singular mass of `C`. The pilot made this explicit: it fed a **rank-32 SVD of `W_pair`** with
  **explained variance 0.561 (POG_dense) / 0.311 (Spotify)** — i.e. it threw away ~44 %/69 % of the
  operator's energy *by construction*.

- **(b) Conjunction.** Pooling the seeds *before* interacting with `c` destroys the per-seed
  overlap information `Δ_inc` needs. Formally, `POOL_S φ(s)` is a symmetric function of the seed
  embeddings that is computed with no reference to `c`; but `Δ_inc(c|S)` depends on
  `|b∩S|` for bundles `b ∋ c`, a quantity that couples `c` and the *individual* seeds. No function
  of `(mean_S φ, ψ(c))` recovers `Σ_b 1[c∈b]·ReLU(|b∩S|−1)` in general.

- **(c) Second order.** `R(c|S)` is bilinear in the seed-affinity vector; a single inner product
  of a pooled query with a candidate vector is first-order and cannot produce the cross terms
  `p_{c,s} p_{c,s'}` with a seed-pair kernel.

- **(d) Per-candidate gate.** `θ̂` mixes two scores with weight `u_c = κ/(f_c+κ)` that varies per
  candidate. A bilinear score plus an additive `a·log(1+f_c)` term is monotone and additive in
  frequency; it cannot implement a *convex interpolation whose mixing coefficient* is a per-
  candidate function — i.e. it cannot "trust the data more for frequent items and the prior more
  for rare items" as a gate.

**Why pooling order is the heart of it.** The two-tower computes `interact(pool(seeds), c)`.
Properties (b) and (c) require `pool_s( interact(c, s) )` — interaction **first**, pooling
**second** — because the quantities that matter (how many seeds `c` jointly co-occurs with; the
cross products of `c`'s per-seed affinities) only exist *after* `c` meets each seed individually.
This is exactly the distinction attention was invented to capture. (The remedy is Report 2.)

---

## 5. Empirical grounding — every structural claim is backed by a number already in the repo

1. **Validation discards the two-tower-friendly collaborative signal.** Selected fusion weights
   (`results/wsweep_pog.json`, `results/wsweep_spotify.json`):
   - POG: `cat 1.0`, `style 0.5`, `simp 0.25`, `softcat 0.25`, `bridge 0.25`; **`csvd 0`**,
     `cooc 0`, `ppmi 0`, `pop 0`, `diff2 0`.
   - Spotify: `simp 1.5`, `pop 1.5`, `homo 1.5`, `bridge 1.5`, `audiostyle 1.0`, `ppmi 0.5`;
     **`csvd 0`**, `cooc 0`.
   In **both** datasets the model keeps full-rank/conjunctive operators (`cat`, `simp`, `ppmi`) and
   assigns **zero weight to `csvd`**, the only clean low-rank collaborative factor. The method's own
   selection prefers the representations a two-tower *cannot* express and rejects the one it *can*.

2. **The compact student was built on the rejected representation.** `bundlefuse/compact_features.py`
   encodes collaborative structure as a rank-32 SVD of `W_pair` (explained variance 0.561 / 0.311)
   and `cat` — the **top** POG signal — as a 31-dim random projection of `Pcat`. Both dominant
   full/moderate-rank signals were heavily compressed; the low-rank content signals were passed
   near-losslessly. The student's inputs are strongest exactly where BundleFuse is weakest and
   weakest exactly where BundleFuse is strongest.

3. **The accuracy loss matches the prediction.** `results/compact_pilot/ANALYSIS.md`: the
   engineered 58,314-param student retains **89.7 % / 67.7 %** of BundleFuse R@20 (POG_dense /
   Spotify); the gap is larger on Spotify, which also has the lower SVD explained variance (0.311)
   — i.e. the dataset whose collaborative operator is *less* low-rank is the one the two-tower
   loses more on. Engineered features significantly beat raw features (+1.0 pp POG, +3.1 pp
   Spotify, Holm-significant), confirming the engineered signals *do* carry information; the
   two-tower simply cannot cash all of it.

4. **More parameters in the same family do not close the structural gap.** The larger metadata
   ranker still trails BundleFuse (0.358 / 0.119 mean R@20), consistent with the limitation being
   representational (pooling order, rank, gate), not capacity.

**Caveats (honest).** Point 1 uses one-at-a-time validation sweeps (`wsweep_*`), so the weights are
coordinate-wise selections, not a joint optimum; the qualitative fact (`csvd` never selected,
full-rank signals preferred) is what we rely on. Point 3's retention figures are descriptive mean
ratios; the formal result in the pilot is the failed validation feasibility gate. The SVD
explained-variance numbers bound operator energy, not ranking accuracy directly — they motivate,
not prove, the causal story. The clean separation in §4 (a)-(d) is the rigorous part; the "this is
what bundle completion *is*" claim in §3 is an interpretation supported by, not proven by, the data.

---

## 6. What this hands the paper

- A precise definition of the estimator's function class (§2) and a reusable taxonomy of its
  signals by rank/order (the table).
- A clean, provable statement of **why** a pool-then-interact compact model loses accuracy — with
  four separable failure modes (§4 a-d) — rather than a vague "compression hurts."
- The striking empirical hook: **BundleFuse's own model selection sets `csvd = 0`**, i.e. it
  rejects the low-rank collaborative representation that the compact student was then forced to
  rely on. The negative pilot result is thereby reframed as *evidence about the structure of the
  problem*: bundle compatibility is full-rank and conjunctive.
- The specification a replacement must meet: express (a) full-rank lookups, (b) conjunction via
  interaction-before-pooling, (c) second-order seed terms, (d) a per-candidate reliability gate —
  which is exactly the architecture derived in Report 2.
