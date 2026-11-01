"""Information-theoretic ceiling for bundle completion (Spotify / POG): how predictable is the task?

Frame completion as a channel  S -> C : the seeds carry some information about a held-out item, and
NO scorer (BundleFuse, SCACS, a transformer) can exceed what that channel allows. The governing
quantity is the conditional entropy H(C|S); the tool is Fano's inequality (list form). If the best
models all cluster just under a low ceiling, the plateau is a property of the TASK, not the model.

We cannot know the true p(c|S), so we estimate it with the strongest tractable oracle -- the
co-occurrence conditional MLE, backed off to the (add-1) marginal so every held item has positive
mass:
      p(c|S) proportional to  (1-lambda) * [ sum_{s in S} C[c,s] ]_normalised  +  lambda * p_marg(c)
      p_marg(c) = (freq_c + 1) / sum (freq + 1)            (seeds removed, then renormalised)

Reported per condition (supplied masks and the exact one_seed pairs from robust_protocol):
  - H(C)            marginal entropy (bits)                     -- a-priori uncertainty over the catalogue
  - Hhat(C|S)       held-out cross-entropy under the oracle      -- estimated conditional uncertainty
  - PPL             perplexity = 2^Hhat(C|S)                     -- "effective # of equally-likely completions"
  - Ihat(S;C)       = H(C) - Hhat(C|S)                           -- bits the seeds actually carry
  - Fano R@20 ceil  max Recall@20 consistent with Hhat(C|S)      -- list-Fano, M = catalogue
  - oracle R@20     the co-occurrence oracle's own Recall@20     -- how much a pure MLE already gets

Honesty: cross-entropy of any oracle >= true H, so these are ESTIMATES, and the Fano number is an
illustrative ceiling, not a certified bound. The certified, assumption-free ceiling is the
combinatorial shortlist recall from Gate 0. Run:  python3 reports/info_ceiling.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT, PROJECT_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.robust_protocol import sparse_pairs
from bundlefuse.metrics import ranking_metrics

LN2 = np.log(2.0)
LAMBDAS = [float(x) for x in os.environ.get("LAMBDAS", "0.01,0.05,0.2").split(",")]
DATASETS = os.environ.get("DATASETS", "spotify,pog_dense").split(",")
TEST_N = int(os.environ.get("TEST_N", "0"))     # 0 = all test queries


def log(*a):
    print(*a, flush=True)


def fano_recall_ceiling(H_bits, M, K):
    """Max Recall@K consistent with conditional entropy H_bits, via list-Fano:
        H <= Hb(Pe) + Pe*log2(M-K) + (1-Pe)*log2(K).
    RHS is monotone increasing in Pe, so find the smallest Pe whose RHS >= H; recall <= 1-Pe."""
    def Hb(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rhs = Hb(mid) + mid * np.log2(M - K) + (1 - mid) * np.log2(K)
        if rhs >= H_bits:
            hi = mid
        else:
            lo = mid
    return 1.0 - hi


def run(dataset):
    d = os.path.join(DATA_ROOT, "bundlemllm", "datasets", dataset)
    n = json.load(open(os.path.join(d, "count.json")))["#I"]
    provided = ProvidedSplits(d, n)
    t0 = time.time()
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    C = graph.C.tocsr()
    freq = graph.freq.astype(np.float64)
    marg = (freq + 1.0)
    marg = marg / marg.sum()                      # add-1 marginal, strictly positive
    H_marg = float(-(marg * np.log2(marg)).sum())
    log(f"{dataset}: n={n} built in {time.time()-t0:.0f}s | H(C)={H_marg:.2f} bits")

    test_ids = sorted(provided.splits["test"][0])
    full = provided.pairs("test")
    conditions = {"supplied": full, "one_seed": sparse_pairs(full, test_ids, "one_seed")}

    out = {"dataset": dataset, "n_items": int(n), "H_marginal_bits": H_marg, "conditions": {}}
    for cond, pairs in conditions.items():
        if TEST_N:
            pairs = pairs[:TEST_N]
        # oracle conditional cross-entropy of held items, per lambda; plus oracle's own Recall@20.
        res_by_lambda = {}
        # Recall@20 of the oracle (lambda-independent ranking up to the tiny marginal term; use mid lambda)
        lam_rank = LAMBDAS[len(LAMBDAS) // 2]
        ce = {lam: 0.0 for lam in LAMBDAS}
        ce_marg = 0.0
        n_held = 0
        rec = {k: np.zeros(2) for k in (10, 20, 50)}      # accumulate (recall, ndcg) per query
        for seeds, held in pairs:
            seeds = list(seeds); held = np.asarray(held, np.int64)
            x = np.zeros(n, np.float64); x[seeds] = 1.0
            q = np.asarray(C @ x).ravel()         # sum_s C[c,s]
            q[seeds] = 0.0
            qs = q.sum()
            qn = q / qs if qs > 0 else np.zeros(n)
            # held-out cross-entropies
            pm = marg.copy(); pm[seeds] = 0.0; pm = pm / pm.sum()
            ce_marg += float(-np.log2(pm[held]).sum())
            for lam in LAMBDAS:
                p = (1 - lam) * qn + lam * pm
                p = p / p.sum()
                ce[lam] += float(-np.log2(np.maximum(p[held], 1e-300)).sum())
            n_held += len(held)
            # oracle ranking for Recall@20 (use lam_rank); per-query to avoid a 254k x Q matrix
            p_rank = (1 - lam_rank) * qn + lam_rank * pm
            mm = ranking_metrics(p_rank[:, None], [(seeds, held.tolist())], Ks=(10, 20, 50))
            for k in rec:
                rec[k] += np.asarray(mm[k])
        m = {k: tuple(rec[k] / len(pairs)) for k in rec}
        Hcond = {lam: ce[lam] / n_held for lam in LAMBDAS}
        H_marg_held = ce_marg / n_held
        for lam in LAMBDAS:
            Hc = Hcond[lam]
            res_by_lambda[str(lam)] = {
                "H_cond_bits": Hc, "perplexity": float(2 ** Hc),
                "MI_bits": H_marg_held - Hc,
                "fano_recall20_ceiling": fano_recall_ceiling(Hc, n, 20)}
        out["conditions"][cond] = {
            "n_queries": len(pairs), "mean_targets": float(np.mean([len(g) for _, g in pairs])),
            "H_marginal_heldout_bits": H_marg_held,
            "oracle_cooc_recall": {"R@20": float(m[20][0]), "N@20": float(m[20][1]),
                                   "R@10": float(m[10][0]), "R@50": float(m[50][0])},
            "by_lambda": res_by_lambda, "lambda_for_ranking": lam_rank}
        mid = res_by_lambda[str(lam_rank)]
        log(f"  [{cond}] targets~{out['conditions'][cond]['mean_targets']:.0f} | "
            f"H(C|S)={mid['H_cond_bits']:.2f}b PPL={mid['perplexity']:.0f} MI={mid['MI_bits']:.2f}b "
            f"| Fano R@20<= {mid['fano_recall20_ceiling']:.3f} | oracle R@20={m[20][0]:.4f}")
    return out


def main():
    t0 = time.time()
    results = []
    for ds in DATASETS:
        results.append(run(ds))
    out_dir = os.path.join(PROJECT_ROOT, "results", "info")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "ceiling.json")
    json.dump(results, open(outp, "w"), indent=2)
    log(f"\nwrote {outp} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
