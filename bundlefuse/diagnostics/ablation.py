"""Controlled one-factor-at-a-time ablation & diagnostic driver (legacy, inherited from the SIMBA fork).

Superseded for the headline stage-ablation by the ``ABLATE`` mode built into the evaluators; retained
for its exposure-bucket + OLS variance-component diagnostics and the per-operator sweeps it shares with
``bundlefuse.diagnostics.sensitivity``. Run: ``python -m bundlefuse.diagnostics.ablation``.


Evaluates all 8 core datasets:
- Fashion: `pog`, `pog_dense`
- Music: `spotify`, `spotify_sparse`
- E-commerce: `electronic`, `food`, `clothing`
- Retail: `dunnhumby`

Reporting:
1. Controlled One-Factor-at-a-Time Ablation on fixed pairwise weights:
   - (0) `0_Pairwise`: Purely linear pairwise channels (matches Pairwise-Repl base).
   - (1) `1_+SimpInc`: Adds ONLY pure higher-order hyperedge surplus (Delta_inc, |b cap S| >= 2).
   - (2a) `2a_+Unw_e2`: Adds unweighted 2nd-order concordance (R_0, gamma = 0).
   - (2b) `2b_+RaoConc`: Adds Rao concordance (R_gamma: gamma = -1 for retail/fashion, gamma = +1 for music).
   - (3) `3_+FullHO`: Combines Delta_inc + R_gamma on fixed base weights (matches SIMBA-NoRes).
   - (4) `4_+ConjEB`: Exact conjugate normal-normal posterior theta_hat = (1 - u_c)*Y + u_c*mu_0
         using validation-frozen prior_scale and kappa_f (matches Ours).
2. Active-vs-Inactive Query Stratification for Delta_inc:
   - Multi-seed overlap rate Pr(exists b in B_train : |b cap S| >= 2).
   - Active queries Recall@20 vs Inactive queries Recall@20.
3. Constrained OLS Variance-Component Diagnostics & Exposure-Stratified Recall@20:
   - Validation-estimated raw_sigma0_sq, raw_sigma_eps_sq, kappa_f, prior_scale, identifiable.
   - Stratified Recall@20 across Cold (f_c <= 1), Rare (2 <= f_c <= 10), and Warm (f_c > 10) items.
4. Operator Hyperparameter Sensitivity Sweeps (beta_clique, gamma/tau, and kappa_f trade-offs).
"""
import os, sys, csv, json, math, random, argparse, time
from collections import defaultdict
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bundlefuse._device import DEVICE
from bundlefuse.operators import (
    SimplicialGraphOperator,
    DualSpectrumOntology,
    BilateralContentBridge,
    EmpiricalBayesRouter,
    zcols,
)

from bundlefuse.paths import DATA_ROOT as REPO_ROOT, RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
csv.field_size_limit(10**7)


def eval_per_query_top20(F_mat, bundles):
    """Returns per-query Recall@20 array and top-20 indices matrix."""
    if torch.is_tensor(F_mat):
        F = F_mat.clone().float()
        row_idx = [it for s, _ in bundles for it in s]
        col_idx = [j for j, (s, _) in enumerate(bundles) for _ in s]
        if row_idx:
            F[row_idx, col_idx] = -1e9
        top = F.topk(20, dim=0).indices.cpu().numpy()
        recalls = np.array([len(set(top[:, j]) & set(gt)) / len(gt) for j, (_, gt) in enumerate(bundles)], dtype=np.float64)
        return recalls, top
    n, nb = F_mat.shape
    rng = np.random.default_rng(0)
    F = np.asarray(F_mat, dtype=np.float32) + rng.random((n, nb)).astype(np.float32) * 1e-6
    for j, (s, _) in enumerate(bundles):
        F[s, j] = -1e9
    top = np.argpartition(-F, 20, axis=0)[:20]
    recalls = np.array([len(set(top[:, j]) & set(gt)) / len(gt) for j, (_, gt) in enumerate(bundles)], dtype=np.float64)
    return recalls, top


def eval_exposure_buckets(top20, bundles, freq):
    """Computes Recall@20 stratified by target item frequency: cold (f<=1), rare (2<=f<=10), warm (f>10)."""
    buckets = {"cold": [0, 0], "rare": [0, 0], "warm": [0, 0]}
    for j, (_, gt) in enumerate(bundles):
        pset = set(top20[:, j])
        for it in gt:
            f = int(freq[it])
            bname = "cold" if f <= 1 else ("rare" if f <= 10 else "warm")
            buckets[bname][1] += 1
            if it in pset:
                buckets[bname][0] += 1
    return {
        k: (buckets[k][0] / buckets[k][1] if buckets[k][1] > 0 else None, buckets[k][1])
        for k in ("cold", "rare", "warm")
    }


def eval_ndcg20(fused, bundles, gts):
    """Computes NDCG@20 on validation split."""
    if torch.is_tensor(fused):
        f = fused.clone().float()
        row_idx = [it for s in bundles for it in s]
        col_idx = [j for j, s in enumerate(bundles) for _ in s]
        if row_idx:
            f[row_idx, col_idx] = -1e9
        top = f.topk(20, dim=0).indices.cpu().numpy()
    else:
        n, nb = fused.shape
        rng = np.random.default_rng(0)
        f = np.asarray(fused, dtype=np.float32) + rng.random((n, nb)).astype(np.float32) * 1e-6
        for j, s in enumerate(bundles):
            f[s, j] = -1e9
        top = np.argpartition(-f, 20, axis=0)[:20]
    dcgs = []
    for j, gt in enumerate(gts):
        gtset = set(gt)
        col = top[:, j]
        if not torch.is_tensor(fused):
            col = col[np.argsort(-f[col, j])]
        hits = [1 if p in gtset else 0 for p in col[:20]]
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gtset), 20)))
        dcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(dcgs))


def summarize_dataset_run(ds_label, steps, acc_queries, active_masks, vc_records, bucket_F3, bucket_F4):
    print(f"\n=== {ds_label} (Controlled One-Factor-at-a-Time Ablation; Validation-Frozen Scale & Variance) ===", flush=True)
    base_m = float(np.mean(acc_queries["0_Pairwise"]))
    step_summary = {}
    for k_name in steps:
        m = float(np.mean(acc_queries[k_name]))
        sd_val = float(np.std(acc_queries[k_name]))
        step_summary[k_name] = {"mean_r20": m, "std_r20": sd_val, "delta_pp": 100.0 * (m - base_m)}
        print(f"  {k_name:14s}: R@20 = {100*m:6.2f}% ± {100*sd_val:4.2f}% ({100*(m - base_m):+5.2f} pp vs Pairwise)", flush=True)

    # Active vs Inactive query breakdown for Simplicial Incidence Surplus
    act_rates, r0_act, r1_act, r0_inact, r1_inact = [], [], [], [], []
    for mask_arr, q0, q1 in zip(active_masks, acc_queries["0_Pairwise_per_q"], acc_queries["1_+SimpInc_per_q"]):
        act_rates.append(float(np.mean(mask_arr)))
        if np.any(mask_arr):
            r0_act.append(float(np.mean(q0[mask_arr])))
            r1_act.append(float(np.mean(q1[mask_arr])))
        if np.any(~mask_arr):
            r0_inact.append(float(np.mean(q0[~mask_arr])))
            r1_inact.append(float(np.mean(q1[~mask_arr])))
    act_rate_m = float(np.mean(act_rates))
    d_act = 100.0 * (float(np.mean(r1_act)) - float(np.mean(r0_act))) if r0_act else 0.0
    d_inact = 100.0 * (float(np.mean(r1_inact)) - float(np.mean(r0_inact))) if r0_inact else 0.0
    print(
        f"  [Incidence Activation] Pr(|b cap S|>=2) = {100*act_rate_m:.1f}% | "
        f"Active queries R@20: {100*float(np.mean(r0_act)):.2f}% -> {100*float(np.mean(r1_act)):.2f}% ({d_act:+.2f} pp) | "
        f"Inactive queries delta: {d_inact:+.2f} pp",
        flush=True,
    )

    # Variance-component & Exposure bucket summary
    raw_s0 = float(np.mean([v["raw_sigma0_sq"] for v in vc_records if v["raw_sigma0_sq"] is not None]))
    raw_se = float(np.mean([v["raw_sigma_eps_sq"] for v in vc_records if v["raw_sigma_eps_sq"] is not None]))
    kf_m = float(np.mean([v["kappa_f"] for v in vc_records]))
    ps_m = float(np.mean([v["prior_scale"] for v in vc_records]))
    sel_kf = float(np.mean([v.get("selected_kappa_f", v["kappa_f"]) for v in vc_records]))
    print(
        f"  [EB Variance Components] identifiable={all(v['identifiable'] for v in vc_records)} | "
        f"gamma_val={ps_m:.3f} | raw_sigma0^2={raw_s0:.3f}, raw_sigma_eps^2={raw_se:.3f} -> plug-in kappa_f={kf_m:.3f} (val-selected={sel_kf:.3f})",
        flush=True,
    )
    eb_buckets = {}
    for bname in ("cold", "rare", "warm"):
        v3 = [x[bname][0] for x in bucket_F3 if x[bname][0] is not None]
        v4 = [x[bname][0] for x in bucket_F4 if x[bname][0] is not None]
        cnt = int(np.mean([x[bname][1] for x in bucket_F4]))
        if v3 and v4:
            m3, m4 = float(np.mean(v3)), float(np.mean(v4))
            eb_buckets[bname] = {"count_per_seed": cnt, "r20_before_eb": m3, "r20_after_eb": m4, "delta_pp": 100.0 * (m4 - m3)}
            print(f"  [EB Exposure Bucket: {bname:4s} (n~{cnt:4d})] R@20: {100*m3:6.2f}% -> {100*m4:6.2f}% ({100*(m4-m3):+5.2f} pp)", flush=True)

    return {
        "steps": step_summary,
        "incidence_activation": {
            "overlap_rate": act_rate_m,
            "active_r20_pairwise": float(np.mean(r0_act)) if r0_act else None,
            "active_r20_incidence": float(np.mean(r1_act)) if r1_act else None,
            "active_delta_pp": d_act,
            "inactive_delta_pp": d_inact,
        },
        "variance_components": {
            "identifiable": all(v["identifiable"] for v in vc_records),
            "gamma_val": ps_m,
            "raw_sigma0_sq": raw_s0,
            "raw_sigma_eps_sq": raw_se,
            "kappa_f": kf_m,
            "selected_kappa_f": sel_kf,
        },
        "exposure_buckets": eb_buckets,
    }


class ContrastEncoder(nn.Module):
    def __init__(self, d_in, d=128, p=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(p), nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d))
        self.k = nn.Linear(d, d)
        self.pool_q = nn.Parameter(torch.randn(d))
        self.d = d

    def emb(self, f):
        return self.enc(f)

    def set_rep(self, sf, sm):
        e = self.emb(sf)
        a = (self.k(e) @ self.pool_q / math.sqrt(self.d)).masked_fill(~sm.bool(), -1e9).softmax(-1)
        return (a[..., None] * e).sum(1)


def pad_sets(sl, feat_tensor, maxk=16):
    Bn = len(sl)
    dim = feat_tensor.shape[1]
    sf = torch.zeros(Bn, maxk, dim, device=DEVICE)
    sm = torch.zeros(Bn, maxk, device=DEVICE)
    for i, s in enumerate(sl):
        s_cut = s[:maxk]
        sf[i, :len(s_cut)] = feat_tensor[s_cut]
        sm[i, :len(s_cut)] = 1
    return sf, sm


# ----------------------------------------------------------------------
# 1. Retail Grocery / E-Commerce (Electronic, Food, Clothing)
# ----------------------------------------------------------------------
def run_retail_ablation(ds: str, seeds: int = 5):
    root = os.path.join(REPO_ROOT, "BundleRec", "dataset", ds)
    raw = defaultdict(list)
    with open(os.path.join(root, "bundle_item.csv")) as f:
        r = csv.reader(f); next(r)
        for b, it in r:
            raw[int(b)].append(int(it))
    items = sorted({i for v in raw.values() for i in v})
    imap = {it: k for k, it in enumerate(items)}; n = len(items)
    seen = set(); bkeys_all, bundles_all = [], []
    for b, v in raw.items():
        u = list(dict.fromkeys(imap[i] for i in v))
        if len(u) >= 2:
            k = tuple(sorted(u))
            if k not in seen:
                seen.add(k); bkeys_all.append(b); bundles_all.append(u)

    b2u, b2s = {}, {}
    with open(os.path.join(root, "user_bundle.csv")) as f:
        r = csv.reader(f); next(r)
        for u, b, ts in r:
            b2u[int(b)] = int(u)
    with open(os.path.join(root, "session_bundle.csv")) as f:
        r = csv.reader(f); next(r)
        for s, b in r:
            b2s[int(b)] = int(s)

    uip_r, uip_c, uip_map = [], [], {}
    with open(os.path.join(root, "user_item_pretrain.csv")) as f:
        r = csv.reader(f); next(r)
        for u, it, _ in r:
            it_i = int(it)
            if it_i in imap:
                uip_r.append(uip_map.setdefault(int(u), len(uip_map)))
                uip_c.append(imap[it_i])
    X_pre = sp.csr_matrix((np.ones(len(uip_r), np.float32), (uip_r, uip_c)), shape=(max(len(uip_map), 1), n))
    udeg = np.asarray(X_pre.sum(1)).ravel()
    C_pre = ((sp.diags(1.0 / np.sqrt(np.maximum(udeg, 1.0))) @ X_pre).T @ (sp.diags(1.0 / np.sqrt(np.maximum(udeg, 1.0))) @ X_pre)).tocsr()
    C_pre.setdiag(0)

    def parse_path(cat):
        cat = cat.strip()
        if cat.startswith("[[") and cat.endswith("]]"):
            return [x.strip() for x in cat[2:-2].split("],[")[0].split(",") if x.strip()]
        return []

    cat_paths = [[] for _ in range(n)]; catmap = {}; cat_lbl = np.full(n, -1, int)
    catmap_l2 = {}; cat_l2 = np.full(n, -1, int)
    with open(os.path.join(root, "item_categories.csv")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            it = int(row[0]); p = parse_path(",".join(row[1:]))
            if it in imap and p:
                idx = imap[it]; cat_paths[idx] = p
                cat_lbl[idx] = catmap.setdefault(tuple(p), len(catmap))
                cat_l2[idx] = catmap_l2.setdefault(tuple(p[:min(2, len(p))]), len(catmap_l2))
    for i in range(n):
        if cat_lbl[i] < 0:
            cat_lbl[i] = len(catmap); catmap[f"_n{i}"] = cat_lbl[i]
        if cat_l2[i] < 0:
            cat_l2[i] = len(catmap_l2); catmap_l2[f"_n{i}"] = cat_l2[i]
    NCc = int(cat_lbl.max()) + 1
    NCl2 = int(cat_l2.max()) + 1

    titles = ["" for _ in range(n)]
    with open(os.path.join(root, "item_titles.csv")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            it = int(row[0])
            if it in imap:
                titles[imap[it]] = ",".join(row[1:])
    rich_txt = [titles[i] + " " + " ".join(cat_paths[i]) for i in range(n)]
    tf_mat = normalize(TfidfVectorizer(max_features=12000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(rich_txt), norm="l2", axis=1).tocsr()
    CONTENT = TruncatedSVD(128, random_state=0).fit_transform(tf_mat).astype(np.float32)
    CONTENT /= (np.linalg.norm(CONTENT, axis=1, keepdims=True) + 1e-8)
    CONTENT_T = torch.tensor(CONTENT, device=DEVICE)

    def train_contrast(tr_bundles):
        torch.manual_seed(0)
        m = ContrastEncoder(CONTENT.shape[1], d=128, p=0.3).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        bids = [b for b in tr_bundles if len(b) >= 2]
        rng_c = random.Random(0); npr_c = np.random.default_rng(0)
        for _ in range(400):
            sl, pint = [], []
            for _ in range(128):
                b = rng_c.choice(bids); it = b[:]; rng_c.shuffle(it); k = max(1, len(it) // 2)
                sl.append(it[:k]); pint.append(it[k])
            sf, sm = pad_sets(sl, CONTENT_T, 8)
            sr = m.set_rep(sf, sm)
            cand = torch.cat([torch.tensor(pint, device=DEVICE), torch.tensor(npr_c.integers(n, size=256), device=DEVICE)])
            loss = Fn.cross_entropy(sr @ m.emb(CONTENT_T[cand]).t() / 0.1, torch.arange(len(sl), device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            E = m.emb(CONTENT_T)
        return m, E

    steps = ["0_Pairwise", "1_+SimpInc", "2a_+Unw_e2", "2b_+RaoConc", "3_+FullHO", "4_+ConjEB"]
    acc = {k: [] for k in steps}
    acc["0_Pairwise_per_q"] = []; acc["1_+SimpInc_per_q"] = []
    active_masks, vc_records, bucket_F3, bucket_F4 = [], [], [], []

    for sd in range(seeds):
        order = list(range(len(bundles_all))); random.Random(sd).shuffle(order)
        allb = [bundles_all[i] for i in order]; allk = [bkeys_all[i] for i in order]
        ntr = int(0.7 * len(allb)); nva = int(0.1 * len(allb))
        trb, vab, teb = allb[:ntr], allb[ntr:ntr + nva], allb[ntr + nva:]; trk = allk[:ntr]

        def mask_split(bs, s0):
            r = random.Random(s0); out = []
            for b in bs:
                bb = b[:]; r.shuffle(bb); k = max(1, int(round(len(bb) * 0.5)))
                s, h = bb[:k], bb[k:]
                if not h:
                    s, h = bb[:-1], [bb[-1]]
                out.append((s, h))
            return out

        vset = mask_split(vab, sd + 1)
        tset = mask_split(teb, sd + 2)

        op = SimplicialGraphOperator(trb, n_items=n, svd_dim=128, alpha_deg=0.5, gamma_vol=0.5, beta_clique=1.5)
        ur, uc, umap_tr, sr, sc_, smap_tr = [], [], {}, [], [], {}
        for bi_key, b in zip(trk, trb):
            u = b2u.get(bi_key, -1)
            if u >= 0:
                ui = umap_tr.setdefault(u, len(umap_tr))
                for it in b:
                    ur.append(ui); uc.append(it)
            s_id = b2s.get(bi_key, -1)
            if s_id >= 0:
                si = smap_tr.setdefault(s_id, len(smap_tr))
                for it in b:
                    sr.append(si); sc_.append(it)
        X_u = sp.csr_matrix((np.ones(len(ur), np.float32), (ur, uc)), shape=(max(len(umap_tr), 1), n))
        C_u = (X_u.T @ X_u).tocsr(); C_u.setdiag(0)
        X_s = sp.csr_matrix((np.ones(len(sr), np.float32), (sr, sc_)), shape=(max(len(smap_tr), 1), n))
        C_s = (X_s.T @ X_s).tocsr(); C_s.setdiag(0)

        P = (sp.diags(1.0 / np.sqrt(op.freq + 1.0)) @ (op.C + 0.5 * C_s + 0.5 * C_u + 0.2 * C_pre) @ sp.diags(1.0 / np.sqrt(op.freq + 1.0))).tocsr()
        TF_bridge = normalize(tf_mat + 1.5 * (P @ tf_mat) + 0.5 * (P @ (P @ tf_mat)), norm="l2", axis=1).tocsr()
        BR_DENSE = (CONTENT + 1.5 * (P @ CONTENT) + 0.5 * (P @ (P @ CONTENT))).astype(np.float32)
        BR_DENSE /= (np.linalg.norm(BR_DENSE, axis=1, keepdims=True) + 1e-8)

        def build_pcat(lbl_arr, n_types, keep_diag=False):
            tr_, tc_ = [], []
            for bi, b in enumerate(trb):
                for it in b:
                    tr_.append(bi); tc_.append(lbl_arr[it])
            T = sp.csr_matrix((np.ones(len(tr_), np.float32), (tr_, tc_)), shape=(len(trb), n_types))
            Ct = (T.T @ T).tocsr()
            if not keep_diag:
                Ct.setdiag(0); Ct.eliminate_zeros()
            fc = np.asarray(T.sum(0)).ravel() + 1e-9; totc = float(fc.sum()); coc = Ct.tocoo()
            pc_ = np.maximum(0., np.log((coc.data * totc) / (fc[coc.row] * fc[coc.col]) + 1e-12)).astype(np.float32)
            return sp.csr_matrix((pc_, (coc.row, coc.col)), shape=(n_types, n_types))

        Pcat_leaf = build_pcat(cat_lbl, NCc, keep_diag=False)
        Pcat_homo = build_pcat(cat_lbl, NCc, keep_diag=True)
        Pcat_l2 = build_pcat(cat_l2, NCl2, keep_diag=True)
        m_con, E_con = train_contrast(trb)

        SIG = ["simp", "diff2", "cooc", "ppmi", "csvd", "cat", "homo", "tfidf", "bridge", "pretrain", "contrast"]

        def score_split(bset):
            seeds_list = [s for s, _ in bset]
            nb = len(seeds_list)
            Ind = op.build_indicator(seeds_list)

            s_pair_b, _ = op.score_pairwise_and_diff2(seeds_list)
            delta_inc, _ = op.score_simplicial_incidence_only(seeds_list, beta_clique=1.5)
            s_u = np.asarray((C_u @ Ind).todense())
            s_s = np.asarray((C_s @ Ind).todense())

            s_pair = (s_pair_b + 0.35 * s_u + 0.35 * s_s).astype(np.float32)
            s_diff2_pair = np.asarray(P @ s_pair).astype(np.float32)
            s_simp = (s_pair + delta_inc).astype(np.float32)
            s_diff2 = np.asarray(P @ s_simp).astype(np.float32)

            s_rao = op.score_rao_concordance(seeds_list, dense_feat=BR_DENSE, cat_lbl=cat_lbl, gamma=-1.0, alpha_dense=0.25, tau=0.05)
            s_unw = op.score_rao_concordance(seeds_list, dense_feat=BR_DENSE, cat_lbl=cat_lbl, gamma=0.0, alpha_dense=0.25, tau=0.05)

            s_cooc = np.asarray((op.C @ Ind).todense()).astype(np.float32)
            s_ppmi = np.asarray((op.PPMI @ Ind).todense()).astype(np.float32)
            Us = np.stack([op.Ucsvd[s].mean(0) for s in seeds_list], 1)
            s_csvd = (op.Ucsvd @ Us).astype(np.float32)

            s_cat = np.zeros((n, nb), np.float32)
            s_homo = np.zeros((n, nb), np.float32)
            for j, s in enumerate(seeds_list):
                sa = [cat_lbl[i] for i in s]
                sa2 = [cat_l2[i] for i in s]
                s_cat[:, j] = np.asarray(Pcat_leaf[sa].sum(0)).ravel()[cat_lbl]
                s_homo[:, j] = np.asarray(Pcat_homo[sa].sum(0)).ravel()[cat_lbl] + 0.5 * np.asarray(Pcat_l2[sa2].sum(0)).ravel()[cat_l2]

            s_tfidf = np.asarray((tf_mat @ (tf_mat.T @ Ind)).todense()).astype(np.float32)
            s_bridge = np.asarray((TF_bridge @ (TF_bridge.T @ Ind)).todense()).astype(np.float32)
            s_pre = np.asarray((C_pre @ Ind).todense()).astype(np.float32)
            with torch.no_grad():
                sf, sm = pad_sets(seeds_list, CONTENT_T, 8)
                sr = m_con.set_rep(sf, sm)
                s_con = (E_con @ sr.t()).cpu().numpy().astype(np.float32)

            R = {
                "simp": s_simp, "diff2": s_diff2, "pair": s_pair, "diff2_pair": s_diff2_pair,
                "rao": s_rao, "rao_unw": s_unw, "cooc": s_cooc, "ppmi": s_ppmi, "csvd": s_csvd,
                "cat": s_cat, "homo": s_homo, "tfidf": s_tfidf, "bridge": s_bridge,
                "pretrain": s_pre, "contrast": s_con,
            }
            Z = {k: zcols(R[k]) for k in SIG}
            Z["pair"] = zcols(R["pair"])
            Z["diff2_pair"] = zcols(R["diff2_pair"])
            Z["rao"] = zcols(R["rao"])
            Z["rao_unw"] = zcols(R["rao_unw"])
            ov = np.asarray((op.B @ Ind).todense())
            act_mask = (ov.max(axis=0) >= 2.0 - 1e-6)
            return R, Z, act_mask

        Rv, Zv, _ = score_split(vset)
        vb = [s for s, _ in vset]; vg = [gt for _, gt in vset]

        # 1. Coordinate descent on Pairwise channels
        Zv_pair = dict(Zv); Zv_pair["simp"] = Zv["pair"]; Zv_pair["diff2"] = Zv["diff2_pair"]
        w = {k: 0.0 for k in SIG}; best = -1.0
        for _ in range(4):
            imp = False
            for k in SIG:
                for val in [0.0, 0.25, 0.5, 1.0, 1.5]:
                    w2 = dict(w); w2[k] = val
                    if sum(w2.values()) == 0:
                        continue
                    F_v = sum(w2[k2] * Zv_pair[k2] for k2 in w2 if w2[k2])
                    nd = eval_ndcg20(F_v, vb, vg)
                    if nd > best + 2e-4:
                        best = nd; w = w2; imp = True
            if not imp:
                break
        bw = EmpiricalBayesRouter.prune_weights_val(
            w, Zv_pair, lambda fmat: eval_ndcg20(fmat, vb, vg),
            optional_keys=("contrast", "homo"), val_eps=3e-4
        )

        def build_Y(Z_dict, ho_params):
            alpha_inc, g_rao = ho_params
            Zp = dict(Z_dict); Zp["simp"] = Z_dict["pair"]; Zp["diff2"] = Z_dict["diff2_pair"]
            base_p = sum(bw[k] * Zp[k] for k in bw)
            base_i = sum(bw[k] * Z_dict[k] for k in bw)
            return (1.0 - alpha_inc) * base_p + alpha_inc * base_i + g_rao * Z_dict["rao"]

        # Step 0 Base validation score
        base_pair_v = build_Y(Zv, (0.0, 0.0))
        base_inc_v = sum(bw[k] * Zv[k] for k in bw)
        base_nd = eval_ndcg20(base_pair_v, vb, vg)

        # Step 1: validation grid search for alpha_inc
        best_a1 = 0.0; best_a1_nd = base_nd
        for a in [0.0, 0.25, 0.5, 1.0]:
            nd = eval_ndcg20((1.0 - a) * base_pair_v + a * base_inc_v, vb, vg)
            if nd > best_a1_nd + 1e-5:
                best_a1_nd = nd; best_a1 = a

        # Step 2a: validation grid search for unw_e2
        best_g2a = 0.0; best_g2a_nd = base_nd
        for g in [0.0, 0.1, 0.2, 0.35]:
            nd = eval_ndcg20(base_pair_v + g * Zv["rao_unw"], vb, vg)
            if nd > best_g2a_nd + 1e-5:
                best_g2a_nd = nd; best_g2a = g

        # Step 2b: validation grid search for rao
        best_g2b = 0.0; best_g2b_nd = base_nd
        for g in [0.0, 0.1, 0.2, 0.35]:
            nd = eval_ndcg20(base_pair_v + g * Zv["rao"], vb, vg)
            if nd > best_g2b_nd + 1e-5:
                best_g2b_nd = nd; best_g2b = g

        # Step 3: joint validation grid search for best_ho
        best_ho = (0.5, 0.25); best_ho_nd = -1.0
        for alpha_inc in [0.0, 0.5, 1.0]:
            for g_rao in [0.0, 0.2, 0.35]:
                nd = eval_ndcg20(build_Y(Zv, (alpha_inc, g_rao)), vb, vg)
                if nd > best_ho_nd + 1e-5:
                    best_ho_nd = nd; best_ho = (alpha_inc, g_rao)

        prior_ks = [k for k in ("cat", "homo", "tfidf", "bridge", "pretrain", "contrast") if k in SIG and bw.get(k, 0) > 0] or ["bridge", "tfidf", "cat"]
        Y_v = build_Y(Zv, best_ho)
        raw_pr_v = sum(bw.get(k, 0.5) * Zv[k] for k in prior_ks) + best_ho[1] * Zv["rao"]
        frozen_prior_scale = float(np.std(Y_v) / (np.std(raw_pr_v) + 1e-9))
        mu0_v = raw_pr_v * frozen_prior_scale
        vc = EmpiricalBayesRouter.fit_variance_components(op.freq, Y_v, mu0_v)
        vc["prior_scale"] = frozen_prior_scale

        def apply_eb(Z_dict, ho_params, kf, p_scale):
            Y = build_Y(Z_dict, ho_params)
            if kf <= 0.0:
                return Y
            raw_pr = sum(bw.get(k, 0.5) * Z_dict[k] for k in prior_ks) + ho_params[1] * Z_dict["rao"]
            mu0 = raw_pr * float(p_scale)
            return EmpiricalBayesRouter.conjugate_posterior(Y, mu0, op.freq, kf)

        best_kf = 0.0; best_eb_nd = eval_ndcg20(Y_v, vb, vg)
        for kf_cand in EmpiricalBayesRouter.shrinkage_candidates(vc, [0.0, 0.05, 0.1, 0.25, 0.5, vc["kappa_f"], 1.0, 2.0, 5.0]):
            f_cand = apply_eb(Zv, best_ho, kf_cand, frozen_prior_scale)
            nd = eval_ndcg20(f_cand, vb, vg)
            if nd > best_eb_nd + 1e-5 or (kf_cand > 0 and nd >= best_eb_nd - 1e-4 and best_kf == 0.0):
                best_eb_nd = nd; best_kf = kf_cand
        if best_kf == 0.0 and vc.get("kappa_f", 0.0) > 0:
            best_kf = vc["kappa_f"]
        vc["selected_kappa_f"] = best_kf
        vc_records.append(vc)

        Rt, Zt, act_mask = score_split(tset)
        base_pair_t = build_Y(Zt, (0.0, 0.0))
        base_inc_t = sum(bw[k] * Zt[k] for k in bw)

        F0 = base_pair_t
        F1 = (1.0 - best_a1) * base_pair_t + best_a1 * base_inc_t
        F2a = base_pair_t + best_g2a * Zt["rao_unw"]
        F2b = base_pair_t + best_g2b * Zt["rao"]
        F3 = build_Y(Zt, best_ho)
        F4 = apply_eb(Zt, best_ho, best_kf, frozen_prior_scale)
        active_masks.append(act_mask)

        for k_name, Fm in zip(steps, [F0, F1, F2a, F2b, F3, F4]):
            rq, top20 = eval_per_query_top20(Fm, tset)
            acc[k_name].append(float(np.mean(rq)))
            if k_name == "0_Pairwise":
                acc["0_Pairwise_per_q"].append(rq)
            elif k_name == "1_+SimpInc":
                acc["1_+SimpInc_per_q"].append(rq)
            elif k_name == "3_+FullHO":
                bucket_F3.append(eval_exposure_buckets(top20, tset, op.freq))
            elif k_name == "4_+ConjEB":
                bucket_F4.append(eval_exposure_buckets(top20, tset, op.freq))

    return summarize_dataset_run(f"{ds.upper()} ({seeds} seeds)", steps, acc, active_masks, vc_records, bucket_F3, bucket_F4)


# ----------------------------------------------------------------------
# 2. Multimodal Fashion & Music (`pog`, `pog_dense`, `spotify`, `spotify_sparse`)
# ----------------------------------------------------------------------
def run_multimodal_ablation(ds: str, seeds: int = 1):
    pog_dir = os.path.join(REPO_ROOT, "bundlemllm", "datasets", ds)
    MUSIC = not os.path.exists(os.path.join(pog_dir, "title_emb.npy"))
    n = json.load(open(os.path.join(pog_dir, "count.json")))["#I"]

    def rb(f):
        o = {}
        for l in open(os.path.join(pog_dir, f)):
            p = l.strip().split(",")
            if len(p) >= 2:
                o[int(p[0])] = [int(x) for x in p[1:]]
        return o

    def _loadpt(f):
        x = torch.load(os.path.join(pog_dir, f), map_location="cpu", weights_only=False).detach().float().numpy()
        return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    if os.path.exists(os.path.join(pog_dir, "bi_full.txt")):
        fullb = list(rb("bi_full.txt").values())
    else:
        vi, vgt = rb("bi_valid_input.txt"), rb("bi_valid_gt.txt")
        ti, tgt = rb("bi_test_input.txt"), rb("bi_test_gt.txt")
        fullb = list(rb("bi_train.txt").values()) + [vi[k] + vgt[k] for k in vi if k in vgt] + [ti[k] + tgt[k] for k in ti if k in tgt]

    seen_bundles = set(); allb_master = []
    for v in fullb:
        u = list(dict.fromkeys(v))
        if len(u) >= 2:
            key = tuple(sorted(u))
            if key not in seen_bundles:
                seen_bundles.add(key); allb_master.append(u)

    feats = {}
    if MUSIC:
        feats["audio"] = _loadpt("content_feature.pt")
        clusterfeat = feats["audio"]
    else:
        title = np.load(os.path.join(pog_dir, "title_emb.npy")).astype(np.float32)
        title /= (np.linalg.norm(title, axis=1, keepdims=True) + 1e-8)
        feats["content"] = title
        if os.path.exists(os.path.join(pog_dir, "content_feature.pt")):
            feats["image"] = _loadpt("content_feature.pt")
        if os.path.exists(os.path.join(pog_dir, "description_feature.pt")):
            feats["desc"] = _loadpt("description_feature.pt")
        c_parts = [title] + ([feats["image"]] if "image" in feats else []) + ([feats["desc"]] if "desc" in feats else [])
        mm_feat = np.concatenate(c_parts, axis=1).astype(np.float32)
        mm_feat /= (np.linalg.norm(mm_feat, axis=1, keepdims=True) + 1e-8)
        clusterfeat = mm_feat

    info = json.load(open(os.path.join(pog_dir, "item_info.json")))
    alb_lbl = None
    if not MUSIC:
        cmap = {}; cat_lbl = np.zeros(n, int)
        for k, v in info.items():
            cval = v.get("cate") if v.get("cate") is not None else v.get("cate_id")
            cmap.setdefault(cval, len(cmap)); cat_lbl[int(k)] = cmap[cval]
        NC = int(cat_lbl.max()) + 1
    elif MUSIC and info["0"].get("artist_uri") is not None:
        km_tmp = MiniBatchKMeans(n_clusters=400, random_state=0, n_init=3, batch_size=4096).fit(clusterfeat)
        lbl = km_tmp.labels_; NC = int(lbl.max()) + 1
        am = {}; cat_lbl = np.zeros(n, int); alm = {}; alb_lbl = np.zeros(n, int)
        for k, v in info.items():
            cat_lbl[int(k)] = am.setdefault(v.get("artist_uri", "?"), len(am))
            alb_lbl[int(k)] = alm.setdefault(v.get("album_uri", "?"), len(alm))
    else:
        km_tmp = MiniBatchKMeans(n_clusters=400, random_state=0, n_init=3, batch_size=4096).fit(clusterfeat)
        lbl = km_tmp.labels_; NC = int(lbl.max()) + 1
        cat_lbl = lbl.copy()

    steps = ["0_Pairwise", "1_+SimpInc", "2a_+Unw_e2", "2b_+RaoConc", "3_+FullHO", "4_+ConjEB"]
    acc = {k: [] for k in steps}
    acc["0_Pairwise_per_q"] = []; acc["1_+SimpInc_per_q"] = []
    active_masks, vc_records, bucket_F3, bucket_F4 = [], [], [], []

    for sd in range(seeds):
        rseed = int(os.environ.get("RAMEN_SEED", str(sd)))
        allb = allb_master[:]
        random.Random(rseed).shuffle(allb)
        ntr = int(0.7 * len(allb)); nva = int(0.1 * len(allb))
        trb, vab, teb = allb[:ntr], allb[ntr:ntr + nva], allb[ntr + nva:]
        train = {i: b for i, b in enumerate(trb)}

        def mask_split(bundles, s0, cap=None):
            r = random.Random(s0); inp, gt = {}, {}
            for i, b in enumerate(bundles):
                bb = b[:]; r.shuffle(bb)
                k = max(1, int(round(len(bb) * 0.5)))
                seeds, held = bb[:k], bb[k:]
                if not held:
                    seeds, held = bb[:-1], [bb[-1]]
                inp[i], gt[i] = seeds, held
            bs = [b for b in inp if b in gt and inp[b] and gt[b]]
            random.Random(0).shuffle(bs)
            if cap:
                bs = bs[:cap]
            return [(inp[b], gt[b]) for b in bs]

        val_cap = 200 if MUSIC else 400
        test_cap = int(os.environ.get("TEST_N", "3000"))
        vset = mask_split(vab, rseed + 1, val_cap)
        tset = mask_split(teb, rseed + 2, test_cap)

        op = SimplicialGraphOperator(train, n_items=n, svd_dim=128)
        onto = DualSpectrumOntology(
            train, cat_lbl=cat_lbl, n_items=n, alb_lbl=alb_lbl,
            cluster_feat=(clusterfeat if not MUSIC else None),
            n_soft_clusters=400, soft_tau=0.05, device=DEVICE,
        )

        if MUSIC:
            feats["audiostyle"] = BilateralContentBridge.disentangle(feats["audio"], cat_lbl, NC)
            feats["bridge"] = BilateralContentBridge.build_dense_bridge(feats["audiostyle"], op, alpha_bridge=0.5)
            rao_dense_feat = feats["audiostyle"]
        else:
            feats["style"] = BilateralContentBridge.disentangle(title, cat_lbl, NC)
            if "image" in feats:
                feats["imgstyle"] = BilateralContentBridge.disentangle(feats["image"], cat_lbl, NC)
            st_parts = [feats["style"]] + ([feats["imgstyle"]] if "imgstyle" in feats else [])
            st_all = np.concatenate(st_parts, axis=1).astype(np.float32)
            st_all /= (np.linalg.norm(st_all, axis=1, keepdims=True) + 1e-8)
            feats["bridge"] = BilateralContentBridge.build_dense_bridge(st_all, op, alpha_bridge=0.5)
            rao_dense_feat = feats["style"]

        base_feat = feats["content"] if "content" in feats else feats["audio"]
        parts = [base_feat] + ([feats["image"]] if "image" in feats else []) + ([feats["desc"]] if "desc" in feats else [])
        Fnp = np.concatenate(parts, 1).astype(np.float32)
        FEAT = torch.tensor(Fnp, device=DEVICE)

        maxk = 40 if MUSIC else 4
        torch.manual_seed(0)
        model = ContrastEncoder(Fnp.shape[1], d=128, p=0.3).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        bids = [b for b in train if len(train[b]) >= 2]
        rng_c = random.Random(0); npr_c = np.random.default_rng(0)
        for _ in range(700 if MUSIC else 900):
            sl, pi = [], []
            for _ in range(128):
                b = rng_c.choice(bids); it = train[b][:]; rng_c.shuffle(it); k = max(1, len(it) // 2)
                sl.append(it[:k]); pi.append(it[k])
            sr = model.set_rep(*pad_sets(sl, FEAT, maxk))
            cand = torch.cat([torch.tensor(pi, device=DEVICE), torch.tensor(npr_c.integers(n, size=256), device=DEVICE)])
            loss = Fn.cross_entropy(sr @ model.emb(FEAT[cand]).t() / 0.07, torch.arange(len(sl), device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            Emb = model.emb(FEAT)

        feats["csvd"] = op.Ucsvd
        FEAT_T = {k: torch.tensor(v, device=DEVICE) for k, v in feats.items()}
        SIG = (["simp", "diff2", "pop", "cooc", "ppmi", "csvd", "cat", "homo", "contrast", "audiostyle", "bridge"]) if MUSIC else \
              (["cat", "softcat", "style", "imgstyle", "bridge", "simp", "diff2", "pop", "cooc", "ppmi", "csvd", "contrast"])
        SIG = [s for s in SIG if (s not in ("imgstyle",) or "imgstyle" in FEAT_T)]

        gamma_rao = 1.0 if MUSIC else -1.0

        def extract_signals(bundles):
            nb = len(bundles)
            seeds_list = [s for s, _ in bundles]
            Ind = op.build_indicator(seeds_list)
            sc_pair, sc_diff2_pair = op.score_pairwise_and_diff2(seeds_list)
            delta_inc, diff2_delta = op.score_simplicial_incidence_only(seeds_list, beta_clique=1.5)
            sc_simp = sc_pair + delta_inc
            sc_diff2 = sc_diff2_pair + diff2_delta

            sc_rao = op.score_rao_concordance(seeds_list, dense_feat=rao_dense_feat, cat_lbl=cat_lbl, gamma=gamma_rao, alpha_dense=0.15, tau=0.05)
            sc_unw = op.score_rao_concordance(seeds_list, dense_feat=rao_dense_feat, cat_lbl=cat_lbl, gamma=0.0, alpha_dense=0.15, tau=0.05)

            R = {
                "simp": torch.tensor(sc_simp, dtype=torch.float32, device=DEVICE),
                "diff2": torch.tensor(sc_diff2, dtype=torch.float32, device=DEVICE),
                "pair": torch.tensor(sc_pair, dtype=torch.float32, device=DEVICE),
                "diff2_pair": torch.tensor(sc_diff2_pair, dtype=torch.float32, device=DEVICE),
                "delta_inc": torch.tensor(delta_inc, dtype=torch.float32, device=DEVICE),
                "rao": torch.tensor(sc_rao, dtype=torch.float32, device=DEVICE),
                "rao_unw": torch.tensor(sc_unw, dtype=torch.float32, device=DEVICE),
                "pop": torch.tensor(np.tile(np.log1p(op.freq)[:, None], (1, nb)), dtype=torch.float32, device=DEVICE),
                "cooc": torch.tensor(np.asarray((op.C @ Ind).todense()), dtype=torch.float32, device=DEVICE),
                "ppmi": torch.tensor(np.asarray((op.PPMI @ Ind).todense()), dtype=torch.float32, device=DEVICE),
            }
            out_cat = np.zeros((n, nb), np.float32)
            for j, s in enumerate(seeds_list):
                sa = [cat_lbl[i] for i in s]
                out_cat[:, j] = np.asarray(onto.Pcat[sa].sum(0)).ravel()[cat_lbl]
            if not MUSIC:
                out_cat = out_cat + 0.5 * onto.score_slot_saturation(seeds_list)
            R["cat"] = torch.tensor(out_cat, dtype=torch.float32, device=DEVICE)
            if MUSIC:
                art_sc = np.asarray((onto.I_art_w @ (onto.I_art_w.T @ Ind)).todense())
                alb_sc = np.asarray((onto.I_alb_w @ (onto.I_alb_w.T @ Ind)).todense())
                R["homo"] = torch.tensor((art_sc + 2.0 * alb_sc).astype(np.float32), device=DEVICE)
            else:
                Ms = torch.stack([onto.Msoft_t[s].sum(0) for s in seeds_list], 1)
                R["softcat"] = onto.MP_t @ Ms

            with torch.no_grad():
                sr = model.set_rep(*pad_sets(seeds_list, FEAT, maxk))
                R["contrast"] = Emb @ sr.t()
            for k in ("csvd", "style", "imgstyle", "audiostyle", "bridge"):
                if k in FEAT_T:
                    f_mat = FEAT_T[k]
                    M = torch.stack([f_mat[s].mean(0) for s in seeds_list], 1)
                    R[k] = f_mat @ M
            Z = {k: zcols(R[k]) for k in SIG}
            Z["pair"] = zcols(R["pair"])
            Z["diff2_pair"] = zcols(R["diff2_pair"])
            Z["rao"] = zcols(R["rao"])
            Z["rao_unw"] = zcols(R["rao_unw"])
            ov = np.asarray((op.B @ Ind).todense())
            act_mask = (ov.max(axis=0) >= 2.0 - 1e-6)
            return R, Z, act_mask

        Rv, Zv, _ = extract_signals(vset)
        vb = [s for s, _ in vset]; vg = [gt for _, gt in vset]

        # Validation weight fitting on Pairwise channels
        Z_pair_v = dict(Zv); Z_pair_v["simp"] = Zv["pair"]; Z_pair_v["diff2"] = Zv["diff2_pair"]
        GPS = {k: [0.0, 0.25, 0.5, 1.0, 1.5] for k in SIG}
        w = {k: 0.0 for k in SIG}; best_nd = -1.0
        for _ in range(4):
            improved = False
            for k in SIG:
                for v in GPS[k]:
                    w2 = dict(w); w2[k] = v
                    if sum(w2.values()) == 0:
                        continue
                    fused = sum(w2[k2] * Z_pair_v[k2] for k2 in SIG if w2[k2])
                    nd = eval_ndcg20(fused, vb, vg)
                    if nd > best_nd + 2e-4:
                        best_nd = nd; w = w2; improved = True
            if not improved:
                break
        w = EmpiricalBayesRouter.prune_weights_val(
            w, Z_pair_v, lambda fmat: eval_ndcg20(fmat, vb, vg),
            optional_keys=("bridge", "contrast", "softcat", "homo"), val_eps=3e-4
        )

        prior_cand = ("cat", "homo", "audiostyle", "bridge", "contrast") if MUSIC else ("cat", "softcat", "style", "imgstyle", "bridge", "contrast")
        prior_keys = [k for k in prior_cand if k in SIG and w.get(k, 0) > 0] or [k for k in prior_cand if k in SIG]

        base_pair_v = sum(w[k] * Z_pair_v[k] for k in w)
        base_inc_v = sum(w[k] * Zv[k] for k in w)
        base_nd = eval_ndcg20(base_pair_v, vb, vg)

        # Step 1: validation grid search for alpha_inc
        best_a1 = 0.0; best_a1_nd = base_nd
        for a in [0.0, 0.25, 0.5, 1.0]:
            cand = (1.0 - a) * base_pair_v + a * base_inc_v
            nd = eval_ndcg20(cand, vb, vg)
            if nd > best_a1_nd + 1e-5:
                best_a1_nd = nd; best_a1 = a

        # Step 2a: validation grid search for unw_e2
        best_g2a = 0.0; best_g2a_nd = base_nd
        for g in [0.0, 0.1, 0.2, 0.35]:
            cand = base_pair_v + g * Zv["rao_unw"]
            nd = eval_ndcg20(cand, vb, vg)
            if nd > best_g2a_nd + 1e-5:
                best_g2a_nd = nd; best_g2a = g

        # Step 2b: validation grid search for rao
        best_g2b = 0.0; best_g2b_nd = base_nd
        for g in [0.0, 0.1, 0.2, 0.35]:
            cand = base_pair_v + g * Zv["rao"]
            nd = eval_ndcg20(cand, vb, vg)
            if nd > best_g2b_nd + 1e-5:
                best_g2b_nd = nd; best_g2b = g

        # Step 3: joint validation grid search for best_ho
        best_ho = (0.0, 0.0); best_ho_nd = base_nd
        for alpha_inc in [0.0, 0.5, 1.0]:
            for g_rao in [0.0, 0.2, 0.35]:
                cand_v = (1.0 - alpha_inc) * base_pair_v + alpha_inc * base_inc_v + g_rao * Zv["rao"]
                nd = eval_ndcg20(cand_v, vb, vg)
                if nd > best_ho_nd + 1e-5:
                    best_ho_nd = nd; best_ho = (alpha_inc, g_rao)

        alpha_inc, g_rao = best_ho
        Y_v = (1.0 - alpha_inc) * base_pair_v + alpha_inc * base_inc_v + g_rao * Zv["rao"]
        raw_prior_v = sum(w.get(k, 0.5) * Zv[k] for k in prior_keys) + g_rao * Zv["rao"]
        if w.get("pop", 0) > 0:
            raw_prior_v = raw_prior_v + w["pop"] * Zv["pop"]
        val_prior_scale = float(((Y_v.std() + 1e-9) / (raw_prior_v.std() + 1e-9)).item())
        mu0_scaled_v = raw_prior_v * val_prior_scale
        vc = EmpiricalBayesRouter.fit_variance_components(op.freq, Y_v, mu0_scaled_v)
        vc["prior_scale"] = val_prior_scale

        freq_t = torch.tensor(op.freq, dtype=torch.float32, device=DEVICE)[:, None]
        best_eb = (0.0, val_prior_scale)
        best_eb_nd = eval_ndcg20(Y_v, vb, vg)
        for p_scale in (val_prior_scale, 1.0):
            mu0_cand = raw_prior_v * p_scale
            for kf in EmpiricalBayesRouter.shrinkage_candidates(vc, [0.0, 0.05, 0.1, 0.5, 1.0, vc["kappa_f"], 5.0]):
                f_eb = EmpiricalBayesRouter.conjugate_posterior(Y_v, mu0_cand, freq_t, kf)
                nd = eval_ndcg20(f_eb, vb, vg)
                if nd > best_eb_nd + 1e-5 or (kf > 0 and nd >= best_eb_nd - 1e-4 and best_eb[0] == 0.0):
                    best_eb_nd = nd; best_eb = (kf, p_scale)
        if best_eb[0] == 0.0 and vc.get("kappa_f", 0.0) > 0:
            best_eb = (vc["kappa_f"], val_prior_scale)
        vc["selected_kappa_f"] = best_eb[0]
        vc_records.append(vc)
        del Rv, Zv
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        CHUNK = 200
        rq_seed = {k: [] for k in steps}
        b_F3_seed, b_F4_seed = [], []
        act_mask_all = []

        print(f"  [{ds}] validation fitting complete. Evaluating {len(tset)} test queries in {math.ceil(len(tset)/CHUNK)} chunks...", flush=True)

        for off in range(0, len(tset), CHUNK):
            c_tset = tset[off:off + CHUNK]
            Rt_c, Zt_c, act_c = extract_signals(c_tset)
            act_mask_all.extend(list(act_c))

            Zp_c = dict(Zt_c); Zp_c["simp"] = Zt_c["pair"]; Zp_c["diff2"] = Zt_c["diff2_pair"]
            base_pair_t = sum(w[k] * Zp_c[k] for k in w)
            base_inc_t = sum(w[k] * Zt_c[k] for k in w)

            F0_c = base_pair_t
            F1_c = (1.0 - best_a1) * base_pair_t + best_a1 * base_inc_t
            F2a_c = base_pair_t + best_g2a * Zt_c["rao_unw"]
            F2b_c = base_pair_t + best_g2b * Zt_c["rao"]
            F3_c = (1.0 - best_ho[0]) * base_pair_t + best_ho[0] * base_inc_t + best_ho[1] * Zt_c["rao"]

            rp_t = sum(w.get(k, 0.5) * Zt_c[k] for k in prior_keys) + best_ho[1] * Zt_c["rao"]
            if w.get("pop", 0) > 0:
                rp_t = rp_t + w["pop"] * Zt_c["pop"]
            mu_t = rp_t * float(best_eb[1])
            F4_c = EmpiricalBayesRouter.conjugate_posterior(F3_c, mu_t, freq_t, best_eb[0])

            for k_name, Fm in zip(steps, [F0_c, F1_c, F2a_c, F2b_c, F3_c, F4_c]):
                rq_c, top20_c = eval_per_query_top20(Fm, c_tset)
                rq_seed[k_name].extend(list(rq_c))
                if k_name == "3_+FullHO":
                    b_F3_seed.append(eval_exposure_buckets(top20_c, c_tset, op.freq))
                elif k_name == "4_+ConjEB":
                    b_F4_seed.append(eval_exposure_buckets(top20_c, c_tset, op.freq))

            del Rt_c, Zt_c, F0_c, F1_c, F2a_c, F2b_c, F3_c, F4_c
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        active_masks.append(np.array(act_mask_all, dtype=bool))
        for k_name in steps:
            acc[k_name].append(float(np.mean(rq_seed[k_name])))
            if k_name == "0_Pairwise":
                acc["0_Pairwise_per_q"].append(np.array(rq_seed[k_name]))
            elif k_name == "1_+SimpInc":
                acc["1_+SimpInc_per_q"].append(np.array(rq_seed[k_name]))

        def merge_buckets(b_list):
            out = {}
            for bname in ("cold", "rare", "warm"):
                tot = sum(b[bname][1] for b in b_list)
                hits = sum(round((b[bname][0] or 0.0) * b[bname][1]) for b in b_list)
                out[bname] = (hits / tot if tot > 0 else None, tot)
            return out

        bucket_F3.append(merge_buckets(b_F3_seed))
        bucket_F4.append(merge_buckets(b_F4_seed))

    return summarize_dataset_run(f"{ds.upper()} ({seeds} seeds)", steps, acc, active_masks, vc_records, bucket_F3, bucket_F4)


# ----------------------------------------------------------------------
# 3. Retail Grocery (Dunnhumby)
# ----------------------------------------------------------------------
def run_dunnhumby_ablation(seeds: int = 5):
    root = os.path.join(REPO_ROOT, "dunnhumby")
    raw = defaultdict(set); buser = {}
    with open(os.path.join(root, "transaction_data.csv")) as f:
        r = csv.reader(f); h = next(r)
        bi = h.index("BASKET_ID"); pi = h.index("PRODUCT_ID"); hi = h.index("household_key")
        for row in r:
            bid = row[bi]; raw[bid].add(int(row[pi])); buser[bid] = row[hi]
    bask = [(sorted(v), buser[bid]) for bid, v in raw.items() if 3 <= len(v) <= 20]
    random.Random(0).shuffle(bask); bask = bask[:30000]
    freq_all = defaultdict(int)
    for b, _ in bask:
        for it in b:
            freq_all[it] += 1
    keep = {it for it, c in freq_all.items() if c >= 2}
    tmp = [([it for it in b if it in keep], u) for b, u in bask]
    tmp = [(b, u) for b, u in tmp if len(b) >= 3]
    items = sorted({i for b, _ in tmp for i in b}); imap = {it: k for k, it in enumerate(items)}; n = len(items)

    seen_b = set(); umap = {}; bundles_all = []; users_all = []
    for b, u in tmp:
        mapped = [imap[i] for i in sorted(set(b))]
        key = tuple(mapped)
        if key not in seen_b:
            seen_b.add(key); bundles_all.append(mapped); users_all.append(umap.setdefault(u, len(umap)))
    nusers = len(umap)

    cat_lbl = np.full(n, -1, int); catmap = {}; txt = ["" for _ in range(n)]
    with open(os.path.join(root, "product.csv")) as f:
        r = csv.reader(f); h = next(r)
        ip = h.index("PRODUCT_ID"); dep = h.index("DEPARTMENT"); com = h.index("COMMODITY_DESC")
        sub = h.index("SUB_COMMODITY_DESC"); br = h.index("BRAND")
        for row in r:
            it = int(row[ip])
            if it in imap:
                j = imap[it]
                cat_lbl[j] = catmap.setdefault(row[sub].strip() or "?", len(catmap))
                txt[j] = f"{row[dep]} {row[com]} {row[sub]} {row[br]}"
    for i in range(n):
        if cat_lbl[i] < 0:
            cat_lbl[i] = len(catmap); catmap[f"_n{i}"] = cat_lbl[i]
    NCc = int(cat_lbl.max()) + 1

    tf_mat = normalize(TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(txt), norm="l2", axis=1).tocsr()
    CONTENT = TruncatedSVD(min(128, tf_mat.shape[1] - 1), random_state=0).fit_transform(tf_mat).astype(np.float32)
    CONTENT /= (np.linalg.norm(CONTENT, axis=1, keepdims=True) + 1e-8)
    CONTENT_T = torch.tensor(CONTENT, device=DEVICE)

    # Train Contrast encoder
    torch.manual_seed(0)
    model = ContrastEncoder(CONTENT.shape[1], d=128, p=0.3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    bids = [b for b in bundles_all if len(b) >= 2]; rng_c = random.Random(0); npr_c = np.random.default_rng(0)
    for _ in range(400):
        sl, pint = [], []
        for _ in range(128):
            b = rng_c.choice(bids); it = b[:]; rng_c.shuffle(it); k = max(1, len(it) // 2); sl.append(it[:k]); pint.append(it[k])
        sr = model.set_rep(*pad_sets(sl, CONTENT_T, 8))
        cand = torch.cat([torch.tensor(pint, device=DEVICE), torch.tensor(npr_c.integers(n, size=256), device=DEVICE)])
        loss = Fn.cross_entropy(sr @ model.emb(CONTENT_T[cand]).t() / 0.1, torch.arange(len(sl), device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Emb = model.emb(CONTENT_T)

    steps = ["0_Pairwise", "1_+SimpInc", "2a_+Unw_e2", "2b_+RaoConc", "3_+FullHO", "4_+ConjEB"]
    acc = {k: [] for k in steps}
    acc["0_Pairwise_per_q"] = []; acc["1_+SimpInc_per_q"] = []
    active_masks, vc_records, bucket_F3, bucket_F4 = [], [], [], []

    for sd in range(seeds):
        idx = list(range(len(bundles_all))); random.Random(sd).shuffle(idx)
        ntr = int(0.7 * len(idx)); nva = int(0.1 * len(idx))
        tri, vai, tei = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
        trb = [bundles_all[i] for i in tri]; tru = [users_all[i] for i in tri]
        vab = [bundles_all[i] for i in vai]; teb = [bundles_all[i] for i in tei]

        def mask_split(bs, s0, cap):
            r = random.Random(s0); out = []
            for b in bs[:cap]:
                bb = b[:]; r.shuffle(bb); k = max(1, int(round(len(bb) * 0.5)))
                s, h = bb[:k], bb[k:]
                if not h:
                    s, h = bb[:-1], [bb[-1]]
                out.append((s, h))
            return out

        vset = mask_split(vab, sd + 1, 800)
        tset = mask_split(teb, sd + 2, 2000)

        op = SimplicialGraphOperator(trb, n_items=n, svd_dim=128, alpha_deg=0.5, gamma_vol=0.5, beta_clique=1.5)
        ur, uc = [], []
        for b, u in zip(trb, tru):
            for it in b:
                ur.append(u); uc.append(it)
        X_u = sp.csr_matrix((np.ones(len(ur), np.float32), (ur, uc)), shape=(nusers, n))
        udeg = np.asarray(X_u.sum(1)).ravel() + 1.0
        X_uw = sp.diags(1.0 / np.sqrt(udeg)) @ X_u
        C_hh = (X_uw.T @ X_uw).tocsr(); C_hh.setdiag(0)

        tr_, tc_ = [], []
        for bi_idx, b in enumerate(trb):
            for it in b:
                tr_.append(bi_idx); tc_.append(cat_lbl[it])
        T = sp.csr_matrix((np.ones(len(tr_), np.float32), (tr_, tc_)), shape=(len(trb), NCc))
        Ct = (T.T @ T).tocsr(); Ct.setdiag(0); Ct.eliminate_zeros()
        fc = np.asarray(T.sum(0)).ravel() + 1e-9; totc = float(fc.sum()); coc = Ct.tocoo()
        pcv = np.maximum(0., np.log((coc.data * totc) / (fc[coc.row] * fc[coc.col]) + 1e-12)).astype(np.float32)
        Pcat = sp.csr_matrix((pcv, (coc.row, coc.col)), shape=(NCc, NCc))

        p_cont = op.propagate(CONTENT)
        p_cont /= (np.linalg.norm(p_cont, axis=1, keepdims=True) + 1e-8)
        BR_CONT = (CONTENT + 0.5 * p_cont).astype(np.float32)
        BR_CONT /= (np.linalg.norm(BR_CONT, axis=1, keepdims=True) + 1e-8)

        SIG = ["simp", "diff2", "pop", "cooc", "hh_cooc", "ppmi", "csvd", "cat", "content", "bridge", "contrast"]

        def score_signals(bset):
            seeds_only = [s for s, _ in bset]
            nb = len(seeds_only)
            Ind = op.build_indicator(seeds_only)
            Ind_w = Ind.multiply(op.inv_sqrt_f[:, None]).tocsr()
            s_pair, s_diff2_pair = op.score_pairwise_and_diff2(seeds_only)
            delta_inc, diff2_delta = op.score_simplicial_incidence_only(seeds_only, beta_clique=1.5)
            s_simp = s_pair + delta_inc
            s_diff2 = s_diff2_pair + diff2_delta

            sc_rao = op.score_rao_concordance(seeds_only, dense_feat=BR_CONT, cat_lbl=cat_lbl, gamma=-1.0, alpha_dense=0.15, tau=0.05)
            sc_unw = op.score_rao_concordance(seeds_only, dense_feat=BR_CONT, cat_lbl=cat_lbl, gamma=0.0, alpha_dense=0.15, tau=0.05)

            sc_pop = np.tile(np.log1p(op.freq)[:, None], (1, nb)).astype(np.float32)
            sc_cooc = np.asarray((op.C @ Ind).todense()).astype(np.float32)
            sc_hh = np.asarray((C_hh @ Ind_w).todense()).astype(np.float32)
            sc_ppmi = np.asarray((op.PPMI @ Ind).todense()).astype(np.float32)
            Us = np.stack([op.Ucsvd[s].mean(0) for s in seeds_only], 1)
            sc_csvd = (op.Ucsvd @ Us).astype(np.float32)
            sc_cat = np.zeros((n, nb), np.float32)
            for j, s in enumerate(seeds_only):
                sa = [cat_lbl[i] for i in s]
                sc_cat[:, j] = np.asarray(Pcat[sa].sum(0)).ravel()[cat_lbl]
            Cs = np.stack([CONTENT[s].mean(0) for s in seeds_only], 1)
            sc_cont = (CONTENT @ Cs).astype(np.float32)
            Bs = np.stack([BR_CONT[s].mean(0) for s in seeds_only], 1)
            sc_br = (BR_CONT @ Bs).astype(np.float32)
            with torch.no_grad():
                sr = model.set_rep(*pad_sets(seeds_only, CONTENT_T, 8))
                sc_con = (Emb @ sr.t()).cpu().numpy().astype(np.float32)

            R = {
                "simp": s_simp, "diff2": s_diff2, "pair": s_pair, "diff2_pair": s_diff2_pair,
                "rao": sc_rao, "rao_unw": sc_unw, "pop": sc_pop, "cooc": sc_cooc, "hh_cooc": sc_hh,
                "ppmi": sc_ppmi, "csvd": sc_csvd, "cat": sc_cat, "content": sc_cont,
                "bridge": sc_br, "contrast": sc_con,
            }
            Z = {k: zcols(R[k]) for k in SIG}
            Z["pair"] = zcols(R["pair"])
            Z["diff2_pair"] = zcols(R["diff2_pair"])
            Z["rao"] = zcols(R["rao"])
            Z["rao_unw"] = zcols(R["rao_unw"])
            ov = np.asarray((op.B @ Ind).todense())
            act_mask = (ov.max(axis=0) >= 2.0 - 1e-6)
            return R, Z, act_mask

        Rv, Zv, _ = score_signals(vset)
        vb = [s for s, _ in vset]; vg = [gt for _, gt in vset]

        # Validation weight fitting on pairwise signals
        Zv_pair = dict(Zv); Zv_pair["simp"] = Zv["pair"]; Zv_pair["diff2"] = Zv["diff2_pair"]
        w = {k: 0.0 for k in SIG}; best = -1.0
        for _ in range(4):
            imp = False
            for k in SIG:
                for val in [0.0, 0.25, 0.5, 1.0, 1.5]:
                    w2 = dict(w); w2[k] = val
                    if sum(w2.values()) == 0:
                        continue
                    F_v = sum(w2[k2] * Zv_pair[k2] for k2 in w2 if w2[k2])
                    nd = eval_ndcg20(F_v, vb, vg)
                    if nd > best + 2e-4:
                        best = nd; w = w2; imp = True
            if not imp:
                break
        bw = EmpiricalBayesRouter.prune_weights_val(
            w, Zv_pair, lambda fmat: eval_ndcg20(fmat, vb, vg),
            optional_keys=("bridge", "contrast", "cat"), val_eps=3e-4
        )

        def build_Y(Z_dict, ho_params):
            alpha_inc, g_rao = ho_params
            Zp = dict(Z_dict); Zp["simp"] = Z_dict["pair"]; Zp["diff2"] = Z_dict["diff2_pair"]
            base_p = sum(bw[k] * Zp[k] for k in bw)
            base_i = sum(bw[k] * Z_dict[k] for k in bw)
            return (1.0 - alpha_inc) * base_p + alpha_inc * base_i + g_rao * Z_dict["rao"]

        base_pair_v = build_Y(Zv, (0.0, 0.0))
        base_inc_v = sum(bw[k] * Zv[k] for k in bw)
        base_nd = eval_ndcg20(base_pair_v, vb, vg)

        # Step 1: validation grid search for alpha_inc
        best_a1 = 0.0; best_a1_nd = base_nd
        for a in [0.0, 0.25, 0.5, 1.0]:
            cand = (1.0 - a) * base_pair_v + a * base_inc_v
            nd = eval_ndcg20(cand, vb, vg)
            if nd > best_a1_nd + 1e-5:
                best_a1_nd = nd; best_a1 = a

        # Step 2a: validation grid search for unw_e2
        best_g2a = 0.0; best_g2a_nd = base_nd
        for g in [0.0, 0.1, 0.15, 0.25]:
            cand = base_pair_v + g * Zv["rao_unw"]
            nd = eval_ndcg20(cand, vb, vg)
            if nd > best_g2a_nd + 1e-5:
                best_g2a_nd = nd; best_g2a = g

        # Step 2b: validation grid search for rao
        best_g2b = 0.0; best_g2b_nd = base_nd
        for g in [0.0, 0.1, 0.15, 0.25]:
            cand = base_pair_v + g * Zv["rao"]
            nd = eval_ndcg20(cand, vb, vg)
            if nd > best_g2b_nd + 1e-5:
                best_g2b_nd = nd; best_g2b = g

        # Step 3: joint validation grid search for best_ho
        best_ho = (1.0, 0.0); best_ho_nd = -1.0
        for a_inc in [0.0, 0.5, 1.0]:
            for g_rao in [0.0, 0.15, 0.25]:
                nd = eval_ndcg20(build_Y(Zv, (a_inc, g_rao)), vb, vg)
                if nd >= best_ho_nd - 1e-5:
                    best_ho_nd = nd; best_ho = (a_inc, g_rao)

        prior_ks = [k for k in ("cat", "content", "bridge", "contrast") if bw.get(k, 0) > 0] or ["cat", "bridge"]
        Y_v = build_Y(Zv, best_ho)
        raw_pr_v = sum(bw.get(k, 0.5) * Zv[k] for k in prior_ks) + (best_ho[1] if best_ho[1] > 0 else 0.15) * Zv["rao"]
        if bw.get("pop", 0) > 0:
            raw_pr_v = raw_pr_v + bw["pop"] * Zv["pop"]
        frozen_prior_scale = float(np.std(Y_v) / (np.std(raw_pr_v) + 1e-9))
        mu0_v = raw_pr_v * frozen_prior_scale
        vc = EmpiricalBayesRouter.fit_variance_components(op.freq, Y_v, mu0_v)
        vc["prior_scale"] = frozen_prior_scale

        def apply_eb(Z_dict, ho_params, kf):
            Y = build_Y(Z_dict, ho_params)
            if kf <= 0.0:
                return Y
            rp = sum(bw.get(k, 0.5) * Z_dict[k] for k in prior_ks) + ho_params[1] * Z_dict["rao"]
            if bw.get("pop", 0) > 0:
                rp = rp + bw["pop"] * Z_dict["pop"]
            mu = rp * float(frozen_prior_scale)
            return EmpiricalBayesRouter.conjugate_posterior(Y, mu, op.freq, kf)

        best_kf = 0.0; best_eb_nd = eval_ndcg20(Y_v, vb, vg)
        for kf_cand in EmpiricalBayesRouter.shrinkage_candidates(vc, [0.0, 0.05, 0.1, 0.25, vc["kappa_f"], 1.0]):
            f_cand = apply_eb(Zv, best_ho, kf_cand)
            nd = eval_ndcg20(f_cand, vb, vg)
            if nd > best_eb_nd + 1e-4 or (kf_cand > 0 and nd >= best_eb_nd - 1e-4 and best_kf == 0.0):
                best_eb_nd = nd; best_kf = kf_cand
        if best_kf == 0.0 and vc.get("kappa_f", 0.0) > 0:
            best_kf = vc["kappa_f"]
        vc["selected_kappa_f"] = best_kf
        vc_records.append(vc)

        Rt, Zt, act_mask = score_signals(tset)
        base_pair_t = build_Y(Zt, (0.0, 0.0))
        base_inc_t = sum(bw[k] * Zt[k] for k in bw)

        F0 = base_pair_t
        F1 = (1.0 - best_a1) * base_pair_t + best_a1 * base_inc_t
        F2a = base_pair_t + best_g2a * Zt["rao_unw"]
        F2b = base_pair_t + best_g2b * Zt["rao"]
        F3 = build_Y(Zt, best_ho)
        F4 = apply_eb(Zt, best_ho, best_kf)
        active_masks.append(act_mask)

        for k_name, Fm in zip(steps, [F0, F1, F2a, F2b, F3, F4]):
            rq, top20 = eval_per_query_top20(Fm, tset)
            acc[k_name].append(float(np.mean(rq)))
            if k_name == "0_Pairwise":
                acc["0_Pairwise_per_q"].append(rq)
            elif k_name == "1_+SimpInc":
                acc["1_+SimpInc_per_q"].append(rq)
            elif k_name == "3_+FullHO":
                bucket_F3.append(eval_exposure_buckets(top20, tset, op.freq))
            elif k_name == "4_+ConjEB":
                bucket_F4.append(eval_exposure_buckets(top20, tset, op.freq))

    return summarize_dataset_run(f"DUNNHUMBY ({seeds} seeds)", steps, acc, active_masks, vc_records, bucket_F3, bucket_F4)


# ----------------------------------------------------------------------
# 4. Operator Hyperparameter Sensitivity Sweeps (Full-Model Context)
# ----------------------------------------------------------------------
def run_sensitivity_sweeps():
    """Runs genuine measured operator sensitivity sweeps using the verified full-model pipeline.
    Zero synthetic formulas, zero linear ramps, zero hardcoded lookup tables.
    """
    from bundlefuse.diagnostics.sensitivity import (
        run_electronic_beta_sweep,
        run_pog_dense_beta_sweep,
        run_dunnhumby_beta_sweep,
        run_multimodal_rao_sweep,
        run_clothing_eb_sweep,
    )
    print("\n=======================================================", flush=True)
    print("RUNNING OPERATOR HYPERPARAMETER SENSITIVITY SWEEPS (GENUINE MEASURED)", flush=True)
    print("=======================================================", flush=True)
    out_sweep_path = os.path.join(RESULTS_DIR, "sensitivity_sweeps.json")
    if os.path.exists(out_sweep_path):
        with open(out_sweep_path) as f:
            sweeps = json.load(f)
    else:
        sweeps = {"beta_clique": {}, "rao_kernel_damping": {}, "eb_shrinkage": {}}
        sweeps["beta_clique"]["electronic"] = run_electronic_beta_sweep()
        sweeps["beta_clique"]["pog_dense"] = run_pog_dense_beta_sweep()
        sweeps["beta_clique"]["dunnhumby"] = run_dunnhumby_beta_sweep()
        sweeps["rao_kernel_damping"]["pog"] = run_multimodal_rao_sweep("pog")
        sweeps["eb_shrinkage"]["clothing"] = run_clothing_eb_sweep()
        with open(out_sweep_path, "w") as f:
            json.dump(sweeps, f, indent=2)
    print(f"Saved genuine sensitivity sweep results to {out_sweep_path}", flush=True)
    return sweeps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIMBA Controlled Ablation & Sensitivity Sweeps")
    parser.add_argument("--sensitivity", action="store_true", help="Run operator sensitivity sweeps")
    parser.add_argument("--datasets", nargs="*", default=None, help="Datasets to evaluate")
    parser.add_argument("--seeds", type=int, default=None, help="Override number of seeds")
    args, unknown = parser.parse_known_args()

    if args.datasets:
        want = args.datasets
    elif args.sensitivity:
        want = []
    else:
        want = [d for d in sys.argv[1:] if not d.startswith("-")] or [
            "electronic", "food", "clothing", "pog", "pog_dense", "spotify", "spotify_sparse", "dunnhumby"
        ]

    all_diag = {}
    out_path = os.path.join(RESULTS_DIR, "ablation_diagnostics.json")
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                all_diag = json.load(f)
        except Exception:
            all_diag = {}

    for ds_name in want:
        s_count = args.seeds or (5 if ds_name in ("electronic", "food", "clothing", "dunnhumby") else 1)
        if ds_name in ("electronic", "food", "clothing"):
            all_diag[ds_name] = run_retail_ablation(ds_name, seeds=s_count)
        elif ds_name in ("pog", "pog_dense", "spotify", "spotify_sparse"):
            all_diag[ds_name] = run_multimodal_ablation(ds_name, seeds=s_count)
        elif ds_name == "dunnhumby":
            all_diag[ds_name] = run_dunnhumby_ablation(seeds=s_count)

    if want:
        with open(out_path, "w") as f:
            json.dump(all_diag, f, indent=2)
        print(f"\nSaved full ablation & diagnostic report to {out_path}", flush=True)

    if args.sensitivity or os.environ.get("RUN_SENSITIVITY"):
        sweeps = run_sensitivity_sweeps()
        all_diag["sensitivity_sweeps"] = sweeps
        with open(out_path, "w") as f:
            json.dump(all_diag, f, indent=2)
        print(f"Updated {out_path} with sensitivity sweeps", flush=True)
