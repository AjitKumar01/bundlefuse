"""Gate 0 - candidate-coverage probe for the SCACS retrieve-then-rerank architecture.

Question: if we shortlist candidates with cheap retrievers, what fraction of the held-out bundle
items survive in a shortlist of size M? That fraction is the *ceiling* on any reranker's recall -
you cannot rerank what you did not retrieve. If the ceiling clears BundleFuse's actual R@20 at a
small M, retrieve-then-rerank is viable; if not, it is dead and we keep only the scorer optimization.

Four cheap, exact retriever legs (each ranked by the evaluator's tie policy, score desc / id asc):
  cooc     : C @ x_S          (raw co-occurrence; full-rank collaborative)
  wpair    : W_pair @ x_S      (IDF-normalized pairwise; full-rank collaborative)
  catpmi   : category-PMI complementarity -- sum_s Pcat[cat_s, cat_c], lifted to items
             (this is BundleFuse's `cat` signal; captures cross-category complements like shoe->bag)
  content  : <content_c, mean_S content>  (low-rank content ANN; catches zero-co-occurrence complements)

No training. Reuses the released provided splits and the real graph operator so numbers line up
with results/compact_pilot. Run from the repo root:  python3 reports/gate0_coverage.py
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT, PROJECT_ROOT
from bundlefuse.provided_splits import ProvidedSplits

# BundleFuse actual R@20 on the same provided split (results/compact_pilot/ANALYSIS.md), original masks.
BUNDLEFUSE_R20 = {"pog_dense": 0.395223, "spotify": 0.155646}
MGRID = [20, 50, 100, 200, 500, 1000, 2000, 5000]
LEGS = ["cooc", "wpair", "catpmi", "content", "union_all"]
ITEM_IDS = None  # np.arange(n), set per dataset


def det_rank(score, seeds, targets):
    """Deterministic 0-based rank of each target under the EVALUATOR tie policy
    (score desc, item-id asc; bundlefuse/metrics.py). Seeds set to -inf so they sort last and never
    occupy a shortlist slot. rank < M  <=>  target is genuinely in a size-M shortlist."""
    s = score.astype(np.float64).copy()
    s[seeds] = -np.inf
    order = np.lexsort((ITEM_IDS, -s))
    pos = np.empty(len(s), dtype=np.int64)
    pos[order] = np.arange(len(s))
    return pos[targets]


def build_category_pmi(train, cat_lbl, n_cat):
    """Replicates DualSpectrumOntology.Pcat: category x category positive PMI, zero diagonal."""
    rows, cols = [], []
    for bi, items in enumerate(train.values()):
        for it in dict.fromkeys(items):
            rows.append(bi)
            cols.append(cat_lbl[it])
    Tc = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(train), n_cat))
    Ctc = (Tc.T @ Tc).tocsr()
    Ctc.setdiag(0)
    Ctc.eliminate_zeros()
    fcc = np.asarray(Tc.sum(0)).ravel() + 1e-9
    tot = float(fcc.sum())
    coo = Ctc.tocoo()
    pv = np.maximum(0.0, np.log((coo.data * tot) / (fcc[coo.row] * fcc[coo.col]) + 1e-12)).astype(np.float32)
    return sp.csr_matrix((pv, (coo.row, coo.col)), shape=(n_cat, n_cat))


def load_content(directory, music, n):
    """L2-normalized content embeddings, matching the scorer's content inputs."""
    import torch
    def pt(name):
        x = torch.load(os.path.join(directory, name), map_location="cpu", weights_only=False).detach().float().numpy()
        return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    parts = []
    if music:
        if os.path.exists(os.path.join(directory, "content_feature.pt")):
            parts.append(pt("content_feature.pt"))
    else:
        t = np.load(os.path.join(directory, "title_emb.npy")).astype(np.float32)
        parts.append(t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8))
        for name in ("content_feature.pt", "description_feature.pt"):
            if os.path.exists(os.path.join(directory, name)):
                parts.append(pt(name))
    cf = np.concatenate(parts, axis=1).astype(np.float32)
    return cf / (np.linalg.norm(cf, axis=1, keepdims=True) + 1e-8)


def load_categories(directory, n, music):
    info = json.load(open(os.path.join(directory, "item_info.json")))
    cat = np.zeros(n, int)
    if not music:
        cmap = {}
        for k, v in info.items():
            cval = v.get("cate") if v.get("cate") is not None else v.get("cate_id")
            cat[int(k)] = cmap.setdefault(cval, len(cmap))
    else:
        amap = {}
        for k, v in info.items():
            cat[int(k)] = amap.setdefault(v.get("artist_uri", "?"), len(amap))
    return cat


def run_dataset(dataset, single_seed):
    global ITEM_IDS
    directory = os.path.join(DATA_ROOT, "bundlemllm", "datasets", dataset)
    n = json.load(open(os.path.join(directory, "count.json")))["#I"]
    music = not os.path.exists(os.path.join(directory, "title_emb.npy"))
    provided = ProvidedSplits(directory, n)
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    C, W, freq = graph.C.tocsr(), graph.W_pair.tocsr(), graph.freq
    cat = load_categories(directory, n, music)
    n_cat = int(cat.max()) + 1
    Pcat = build_category_pmi(provided.train, cat, n_cat)
    CF = load_content(directory, music, n)
    ITEM_IDS = np.arange(n)

    pairs = provided.pairs("test")
    n_held = 0
    n_reach_collab = 0   # held with positive collaborative evidence
    n_reach_catpmi = 0   # held with positive category-PMI evidence
    n_reach_disc = 0     # held reachable by collab OR catpmi (discrete hard ceiling, content excluded)
    cov = {M: {leg: 0 for leg in LEGS} for M in MGRID}

    for qi, (seeds_full, held) in enumerate(pairs):
        seeds = [seeds_full[0]] if single_seed else list(seeds_full)
        held = np.asarray(held)
        x = np.zeros(n, np.float32)
        x[seeds] = 1.0
        s_cooc = np.asarray(C @ x).ravel()
        s_wpair = np.asarray(W @ x).ravel()
        seed_cats = [int(cat[s]) for s in seeds]
        cat_per_cat = np.asarray(Pcat[seed_cats].sum(0)).ravel()  # length n_cat
        s_catpmi = cat_per_cat[cat]                               # lift to items
        s_content = CF @ CF[seeds].mean(0)

        r = {"cooc": det_rank(s_cooc, seeds, held), "wpair": det_rank(s_wpair, seeds, held),
             "catpmi": det_rank(s_catpmi, seeds, held), "content": det_rank(s_content, seeds, held)}

        n_held += len(held)
        reach_collab = (s_cooc[held] > 0) | (s_wpair[held] > 0)
        reach_cat = s_catpmi[held] > 0
        n_reach_collab += int(reach_collab.sum())
        n_reach_catpmi += int(reach_cat.sum())
        n_reach_disc += int((reach_collab | reach_cat).sum())
        for M in MGRID:
            hit = {leg: r[leg] < M for leg in ("cooc", "wpair", "catpmi", "content")}
            for leg in ("cooc", "wpair", "catpmi", "content"):
                cov[M][leg] += int(hit[leg].sum())
            cov[M]["union_all"] += int((hit["cooc"] | hit["wpair"] | hit["catpmi"] | hit["content"]).sum())
        if (qi + 1) % 1000 == 0:
            print(f"  {dataset} {'single' if single_seed else 'orig'}: {qi+1}/{len(pairs)}", flush=True)

    curves = {leg: {M: cov[M][leg] / n_held for M in MGRID} for leg in LEGS}
    return {
        "dataset": dataset, "condition": "single_seed" if single_seed else "original",
        "n_items": int(n), "n_test_queries": len(pairs), "n_held_items": n_held,
        "pct_reachable_collab": round(100.0 * n_reach_collab / n_held, 2),
        "pct_reachable_catpmi": round(100.0 * n_reach_catpmi / n_held, 2),
        "pct_hard_ceiling_discrete": round(100.0 * n_reach_disc / n_held, 2),
        "bundlefuse_r20": BUNDLEFUSE_R20.get(dataset), "recall_ceiling": curves,
    }


def fmt_row(label, curve):
    return f"| {label:<10} | " + " | ".join(f"{curve[M]:.3f}" for M in MGRID) + " |"


def main():
    out_dir = os.path.join(PROJECT_ROOT, "results", "gate0")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    lines = ["# Gate 0 - candidate-coverage probe (v3: category-PMI slate + content leg, evaluator tie policy)",
             "",
             "Recall ceiling = fraction of held-out items kept within a size-M shortlist (rank < M under",
             "score desc / item-id asc). `union_all` = in top-M of any leg (budget M each; shortlist <= 4M).",
             "`catpmi` is cross-category complementarity (BundleFuse's `cat`); `content` is a content-embedding",
             "ANN. The discrete hard ceiling counts held items with positive collaborative OR category-PMI",
             "evidence (content excluded, since cosine reaches everything at large M). BundleFuse R@20 is the bar.",
             ""]
    for dataset in ("pog_dense", "spotify"):
        for single in (False, True):
            t0 = time.perf_counter()
            r = run_dataset(dataset, single)
            r["seconds"] = round(time.perf_counter() - t0, 1)
            results.append(r)
            print(f"done {dataset} {r['condition']:<11} ({r['seconds']}s): "
                  f"union@500={r['recall_ceiling']['union_all'][500]:.3f} "
                  f"hard_ceiling={r['pct_hard_ceiling_discrete']}% vs BF R@20={r['bundlefuse_r20']}", flush=True)
            lines += [
                f"## {dataset} - {r['condition']}", "",
                f"- catalogue {r['n_items']} items, {r['n_test_queries']} test queries, "
                f"{r['n_held_items']} held-out items",
                f"- held reachable by collaboration (positive C/W_pair): **{r['pct_reachable_collab']}%**",
                f"- held reachable by category-PMI complementarity: **{r['pct_reachable_catpmi']}%**",
                f"- **discrete hard ceiling** (collab OR category-PMI): **{r['pct_hard_ceiling_discrete']}%** "
                f"(content ANN can retrieve beyond this at finite M)",
                f"- **BundleFuse actual R@20 = {r['bundlefuse_r20']}** (bar to clear)", "",
                "| retriever | M=" + " | M=".join(str(M) for M in MGRID) + " |",
                "|" + "---|" * (len(MGRID) + 1)] + \
                [fmt_row(leg, r["recall_ceiling"][leg]) for leg in LEGS] + [""]
    json.dump(results, open(os.path.join(out_dir, "coverage.json"), "w"), indent=1)
    open(os.path.join(out_dir, "REPORT.md"), "w").write("\n".join(lines))
    print(f"\nwrote {out_dir}/coverage.json and REPORT.md", flush=True)


if __name__ == "__main__":
    main()
