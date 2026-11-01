# Report 2 — An architecture derived from the theory: interaction-first set completion with exact sparse compatibility features

**Status:** architecture proposal + novelty positioning + staged experiment plan with go/no-go
gates. Builds directly on `01_bundlefuse_as_a_function_class.md`.
**Thesis in one sentence:** replace the two-tower's *pool-then-interact* with
*interact-then-pool over exact sparse compatibility lookups, followed by a learned per-candidate
reliability gate* — a class that provably **contains** BundleFuse and has capacity to improve it
precisely in the sparse/cold regime where BundleFuse's fixed rules are weakest.

---

## 1. The design principle

Report 1 showed the compact two-tower fails on four separable properties: (a) full-rank
collaborative lookups, (b) conjunction (AND over seeds), (c) second-order seed interaction,
(d) per-candidate frequency gating. Rather than hope a bigger same-family model recovers them
(it didn't — the metadata ranker also trailed), we design the **minimal architecture whose
function class is a strict superset of BundleFuse's**, so that:

- *(soundness)* it can represent BundleFuse exactly, hence cannot be worse than the teacher in
  principle for any input regime — no accuracy is left on the table by construction; and
- *(headroom)* its free parameters let it deviate from BundleFuse's *fixed* fusion weights and
  *fixed* shrinkage law where those are suboptimal — chiefly single-seed / cold-start, the regime
  BundleFuse's conjunction term `Δ_inc` literally zeroes out and where the generalization study
  already found a learned ranker beating BundleFuse on sparse temporal retail.

The efficiency lever is the inversion of a false assumption. The two-tower tried to **globally
low-rank-approximate** `C`/`PPMI`/`W_pair`. But scoring a query never needs all of `C` — it needs
only the entries `C[c, s]` for `s ∈ S`: a handful of **exact sparse lookups per candidate**. We
keep those exact and learn how to *pool* them. No factorization, no energy thrown away.

---

## 2. The architecture: Seed-Cross-Attention Compatibility Scorer (SCACS)

For a candidate `c` and seed set `S`, form a per-pair exact feature vector by sparse lookup:

```
φ(c,s) = [ C[c,s], PPMI[c,s], W_pair[c,s], W_pair[s,c],           # exact collaborative (sparse)
           Pcat[cat_c, cat_s], 1[cat_c = cat_s],                   # exact taxonomic
           ⟨content_c, content_s⟩,                                 # content affinity
           log(1+deg(s)), log(1+f_c) ]                             # exposure context
```

All entries are exact; the collaborative ones are `O(1)` CSR lookups. Then:

**(A) First-order + conjunction via attention (fixes 4a, 4b).**
```
a(c,s) = softmax_s( MLP_att(φ(c,s)) )           # learned, candidate-conditioned seed weights
h1(c|S) = Σ_s a(c,s) · g(φ(c,s))                # learned pooled compatibility
hcount(c|S) = Σ_s σ(MLP_gate(φ(c,s)))           # soft count of "strongly compatible" seeds
```
`hcount` is the learnable analogue of `Δ_inc`'s `max(0,|b∩S|−1)` conjunction: attention weights
can concentrate (OR) or spread (AND), and `hcount` reads off *how many* seeds `c` matches. Because
`φ` is formed **after** `c` meets each seed, this is interaction-first pooling — the operation the
two-tower could not do.

**(B) Second-order head (fixes 4c).**
```
h2(c|S) = ( Σ_s a(c,s)·p(c,s) )² − Σ_s ( a(c,s)·p(c,s) )²        # Gini-style pair sum, O(|S|)
```
optionally modulated by a seed–seed redundancy kernel `K_S` reusing the content/category features,
i.e. a cheap learnable stand-in for Rao `R(c|S)`.

**(C) Per-candidate learned reliability gate (fixes 4d).**
```
Y(c|S)  = w · [h1, hcount, h2]                   # fused interaction score
μ0(c)   = prior head from content/category only  # cold-start score, no collaborative lookup
u_c     = σ( MLP_u( log(1+f_c), log|S|, hcount, coverage(S) ) )   # LEARNED, context-dependent
score(c|S) = (1 − u_c)·Y(c|S) + u_c·μ0(c)
```
`u_c` generalizes BundleFuse's `κ/(f_c+κ)`: it can still shrink rare candidates toward the prior,
but now the shrinkage **also depends on how much seed evidence exists** — the single knob that
should buy the sparse-input win.

### 2.1 Superset proposition (why it can't lose in principle)

There is a parameter setting of SCACS that reproduces BundleFuse up to the realized signal set:
`a(c,s) ∝ W_pair[c,s]`-reading attention with `g = id` recovers the linear pairwise/`cooc`/`ppmi`
terms exactly (they are *inputs*, not approximations); `hcount` with a step gate recovers the
incidence conjunction; `h2` with `K_S` set to the category/feature kernel recovers Rao; `μ0` set to
the content/category prior and `u_c = κ/(f_c+κ)` recovers EB shrinkage. Hence SCACS ⊇ BundleFuse.
Unlike the two-tower, **no structural property from Report 1 §4 is unreachable.**

### 2.2 Efficiency

- Interaction cost per scored candidate is `O(|S| · d_φ)` with `|S|` small (a handful) and `d_φ`
  ~10 — far cheaper than the two-tower's own dot product was *claimed to be* a problem.
- Candidates are **shortlisted** by exact cheap collaborative retrieval (union of top-N neighbors
  of each seed under `C`/`W_pair`) + a metadata slate. SCACS reranks only the shortlist. Full
  catalogue scoring is never needed.
- The expensive BundleFuse step (dense `C @ x_S` over all items, 2–55 ms/query even optimized) is
  replaced by sparse neighbor lookups on a shortlist. This is where a genuine speed story could
  come from — but it is **gated on one measurement** (§4, Gate 0).

---

## 3. Novelty — what is actually new, stated honestly

The contribution is the **architecture class and its motivation**, not distillation.

- **vs Ranking Distillation (1809.07428) / HetComp (2303.01130):** those compress a large teacher
  into a smaller student *of the same architectural family*. Here the student is a *different
  function class* chosen from a representational-gap analysis of the teacher; distillation is
  optional and secondary. The novel object is "interaction-first attention over **exact sparse
  compatibility statistics** with a learned reliability gate."
- **vs CLHE / RaMen (attention bundle models):** those attend over **learned per-item ID
  embeddings** (millions of parameters; collaborative signal is *learned and low-rank*). SCACS
  attends over **exact engineered pair features** (`C[c,s]`, `PPMI[c,s]`, category-PMI): the
  collaborative signal is *exact, not factorized*, and there are **no ID embedding tables**, so
  the parameter count stays tiny. "Attention over exact compatibility lookups" is the differentiator.
- **vs DeepSets / Set Transformer:** same observation — standard set encoders pool learned item
  embeddings; the novelty is that the *values being attended over are exact combinatorial
  statistics* of the (candidate, seed) pair, making the conjunction and collaborative terms exact
  rather than learned approximations.
- **vs BundleFuse itself:** BundleFuse is the `u_c` and fusion weights *frozen by closed form*;
  SCACS makes exactly those context-dependent while keeping the signals exact. The paper's framing:
  *"BundleFuse is the zero-parameter special case; SCACS learns where the closed form is wrong."*

**One-sentence claim for the paper.** *Bundle completion is conjunctive, full-rank, cold-start
compatibility estimation; we show a pool-then-interact ranker cannot express it, and introduce a
tiny interaction-first attention ranker over exact sparse compatibility features that provably
contains the closed-form estimator and learns a context-dependent reliability gate to improve it
under sparse evidence.*

---

## 4. Staged experiments with go/no-go gates

Start on **POG_dense + Spotify**, both **original masks and single-seed** stress, matching the
pilot so numbers are comparable. Each stage can kill the direction cheaply.

**Gate 0 — candidate coverage (cheapest, do first; ~1 hour).**
Measure: does the union of per-seed exact `C`/`W_pair` top-N neighbors + a metadata slate contain
the held-out items for ≥ X% of test queries, at shortlist size ≤ M? Report coverage@M curves for
original and single-seed. *If coverage is poor even at large M → the retrieve-then-rerank efficiency
story is dead; stop, and keep only the scorer optimization.* This is the one measurement the agent's
ANALYSIS.md also flagged as the gate.

**Gate 1 — representational fix on the dense case.** Train SCACS (full interaction-first, exact
features, fixed `u_c` first) on original masks. Compare to: (i) the pool-then-interact two-tower
(pilot), (ii) BundleFuse. *Prediction from Report 1: SCACS closes most of the two-tower's 10–32 %
gap because it stops factorizing `C`/`cat`.* If SCACS does **not** beat the two-tower on the dense
case, the interaction-first thesis is wrong — stop.

**Gate 2 — the sparse-input win.** Turn on the learned gate `u_c` and single-seed training.
Compare SCACS vs BundleFuse vs two-tower on single-seed and on sparse temporal retail
(dunnhumby_chrono, where the generalization study already saw a learned ranker beat BundleFuse by
~1 pp). *The target result:* SCACS ≥ BundleFuse on dense **and** strictly > BundleFuse on
single-seed/cold. This is the publishable claim.

**Decisive ablations (the paper's evidence that each property matters):**
- interaction-first vs pool-first (same features, same budget) → isolates property (b);
- exact sparse lookups vs rank-32 SVD inputs → isolates property (a);
- learned `u_c` vs fixed `κ/(f_c+κ)` vs no gate → isolates property (d);
- with/without the `h2` second-order head → isolates property (c).
Each ablation maps 1:1 to a Report 1 failure mode, so the story is "we predicted four gaps and
measured the value of closing each."

**What would falsify the thesis:** SCACS fails Gate 1 (interaction-first doesn't help), or passes
Gate 1 but the learned gate gives no single-seed advantage at Gate 2 (then BundleFuse's fixed
shrinkage was already optimal and there is no headroom). Both are clean, reportable outcomes.

---

## 5. Risks and honest caveats

- **Attention over (shortlist × seeds) adds a real constant.** It is tiny per pair but the
  shortlist must be small; everything rests on Gate 0 coverage.
- **Transductive category/content assumption** carries over from BundleFuse (`cat`, content
  features assume the catalogue is known); SCACS inherits, does not fix, this.
- **Dense-case parity is not guaranteed to turn into a dense-case *win*.** The honest target is
  *parity on dense + win on sparse + large speedup on shortlist*. If it only reaches parity
  everywhere, the result is "a tiny learnable model matches a hand-tuned closed form and is faster"
  — still worth reporting, weaker claim.
- **The exact sparse lookups are offline-cheap but not free**: `C`/`PPMI`/`W_pair` must be built
  and stored (already done for BundleFuse); SCACS adds no new heavy preprocessing.
- This is a proposal: no SCACS result exists yet. Everything in §4 is a planned, falsifiable
  experiment, not a measured outcome.

---

## 6. Recommended first move

Implement **Gate 0** only (exact-neighbor + metadata shortlist coverage on POG_dense + Spotify,
original and single-seed). It is ~1 hour, needs no training, and decides whether the whole
retrieve-then-rerank architecture is viable before a single parameter is learned. If coverage is
good, implement SCACS with a **fixed** gate for Gate 1 (the representational test), and only then
add the learned gate for Gate 2 (the sparse-input claim).
