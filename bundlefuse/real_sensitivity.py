"""Standalone runner for genuine, 100% measured operator hyperparameter sensitivity sweeps.
No synthetic formulas, no ramps, no linear interpolation, no hardcoded constants.
Evaluates:
1. beta_clique in {0.0, 0.5, 1.0, 1.5, 2.5} on electronic, pog_dense, dunnhumby
2. gamma in {-1.0, 0.0, +1.0} and tau in {0.01, 0.05, 0.10, 0.50} on pog and spotify
3. kappa_f in {0.0, 0.05, 0.1, 0.5, 1.0, 5.0} on clothing (overall, cold, rare, warm)
"""
import os, sys, json, math, random, time, csv
from collections import defaultdict
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse._device import DEVICE
from bundlefuse.operators import (
    SimplicialGraphOperator,
    DualSpectrumOntology,
    BilateralContentBridge,
    EmpiricalBayesRouter,
    zcols,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def eval_per_query_top20(F, bundles):
    Kmax = 20
    if isinstance(F, torch.Tensor):
        F_t = F.clone()
        for j, (s, _) in enumerate(bundles):
            F_t[s, j] = -1e9
        top = F_t.topk(Kmax, dim=0).indices.cpu().numpy()
    else:
        F_c = F.copy()
        for j, (s, _) in enumerate(bundles):
            F_c[s, j] = -1e9
        top = np.argpartition(-F_c, Kmax, axis=0)[:Kmax]
    rq = []
    for j, (_, gt) in enumerate(bundles):
        gtset = set(gt)
        col = top[:, j]
        hits = [1 if p in gtset else 0 for p in col]
        rq.append(sum(hits) / len(gtset))
    return np.asarray(rq), top


def eval_exposure_buckets(top20, bundles, freq):
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
        k: (round(buckets[k][0] / buckets[k][1], 4) if buckets[k][1] > 0 else None, buckets[k][1])
        for k in buckets
    }


def eval_ndcg20(F, bundles):
    Kmax = 20
    if isinstance(F, torch.Tensor):
        F_t = F.clone()
        for j, (s, _) in enumerate(bundles):
            F_t[s, j] = -1e9
        top = F_t.topk(Kmax, dim=0).indices.cpu().numpy()
    else:
        F_c = F.copy()
        for j, (s, _) in enumerate(bundles):
            F_c[s, j] = -1e9
        top = np.argpartition(-F_c, Kmax, axis=0)[:Kmax]
    nq = []
    for j, (_, gt) in enumerate(bundles):
        gtset = set(gt)
        col = top[:, j]
        if not isinstance(F, torch.Tensor):
            order = np.argsort(-F[col, j])
            sorted_pred = col[order]
        else:
            sorted_pred = col
        hits = [1 if p in gtset else 0 for p in sorted_pred]
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gtset), 20)))
        nq.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(nq))


# ======================================================================
# 1. Retail Grocery / E-Commerce Helper
# ======================================================================
class ContrastEncoder(nn.Module):
    def __init__(self, d_in, d=128, p=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(p), nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d))
        self.k = nn.Linear(d, d); self.pool_q = nn.Parameter(torch.randn(d)); self.d = d
    def emb(self, f): return self.enc(f)
    def set_rep(self, sf, sm):
        e = self.emb(sf)
        a = (self.k(e) @ self.pool_q / math.sqrt(self.d)).masked_fill(~sm.bool(), -1e9).softmax(-1)
        return (a[..., None] * e).sum(1)


def pad_sets(sl, feat_tensor, maxk=16):
    Bn = len(sl); dim = feat_tensor.shape[1]
    sf = torch.zeros(Bn, maxk, dim, device=DEVICE); sm = torch.zeros(Bn, maxk, device=DEVICE)
    for i, s in enumerate(sl):
        s_cut = s[:maxk]; sf[i, :len(s_cut)] = feat_tensor[s_cut]; sm[i, :len(s_cut)] = 1
    return sf, sm


def setup_retail_dataset(ds: str):
    root = os.path.join(REPO_ROOT, "BundleRec", "dataset", ds)
    raw = defaultdict(list)
    with open(os.path.join(root, "bundle_item.csv")) as f:
        r = csv.reader(f); next(r)
        for b, it in r: raw[int(b)].append(int(it))
    items = sorted({i for v in raw.values() for i in v})
    imap = {it: k for k, it in enumerate(items)}; n = len(items)
    seen = set(); bundles_all = []
    for b, v in raw.items():
        u = list(dict.fromkeys(imap[i] for i in v))
        if len(u) >= 2:
            k = tuple(sorted(u))
            if k not in seen:
                seen.add(k); bundles_all.append(u)

    b2u = {}; b2s = {}
    with open(os.path.join(root, "user_bundle.csv")) as f:
        r = csv.reader(f); next(r)
        for u, b, _ in r: b2u[int(b)] = int(u)
    with open(os.path.join(root, "session_bundle.csv")) as f:
        r = csv.reader(f); next(r)
        for s, b in r: b2s[int(b)] = int(s)

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
    X_pre_w = sp.diags(1.0 / np.sqrt(np.maximum(udeg, 1.0))) @ X_pre
    C_pre = (X_pre_w.T @ X_pre_w).tocsr(); C_pre.setdiag(0)

    def parse_path(cat):
        cat = cat.strip()
        if cat.startswith("[[") and cat.endswith("]]"):
            return [x.strip() for x in cat[2:-2].split("],[")[0].split(",") if x.strip()]
        return []

    catmap_leaf, catmap_l2 = {}, {}
    cat_lbl, cat_l2 = np.full(n, -1, int), np.full(n, -1, int)
    with open(os.path.join(root, "item_categories.csv")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            it = int(row[0]); p = parse_path(",".join(row[1:]))
            if it in imap and p:
                idx = imap[it]
                cat_lbl[idx] = catmap_leaf.setdefault(tuple(p), len(catmap_leaf))
                cat_l2[idx] = catmap_l2.setdefault(tuple(p[:min(2, len(p))]), len(catmap_l2))
    for i in range(n):
        if cat_lbl[i] < 0: cat_lbl[i] = len(catmap_leaf); catmap_leaf[f"_n{i}"] = cat_lbl[i]
        if cat_l2[i] < 0: cat_l2[i] = len(catmap_l2); catmap_l2[f"_n{i}"] = cat_l2[i]
    NCc = int(cat_lbl.max()) + 1
    NCl2 = int(cat_l2.max()) + 1

    titles = ["" for _ in range(n)]
    with open(os.path.join(root, "item_titles.csv")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            it = int(row[0])
            if it in imap: titles[imap[it]] = row[1].strip()

    tf_mat = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(titles)
    tf_mat = normalize(tf_mat, norm="l2", axis=1).tocsr()
    cdim = min(64, tf_mat.shape[1] - 1)
    CONTENT = TruncatedSVD(cdim, random_state=0).fit_transform(tf_mat).astype(np.float32)
    CONTENT /= (np.linalg.norm(CONTENT, axis=1, keepdims=True) + 1e-8)
    CONTENT_T = torch.tensor(CONTENT, device=DEVICE)

    perm = list(range(len(bundles_all)))
    random.Random(0).shuffle(perm)
    shuf_b = [bundles_all[i] for i in perm]
    ntr = int(0.7 * len(shuf_b)); nva = int(0.1 * len(shuf_b))
    trb, vab, teb = shuf_b[:ntr], shuf_b[ntr:ntr+nva], shuf_b[ntr+nva:]

    def mask_b(blist, seed):
        rng = random.Random(seed); out = []
        for b in blist:
            bb = b[:]; k = max(1, int(round(len(bb) * 0.5)))
            rng.shuffle(bb); out.append((bb[:k], bb[k:]))
        return out

    vset = mask_b(vab, 1)
    tset = mask_b(teb, 2)
    op = SimplicialGraphOperator(trb, n_items=n, svd_dim=64, beta_clique=1.5)

    u_r, u_c = [], []
    for bi, b in enumerate(trb):
        if bi in b2u:
            for it in b: u_r.append(b2u[bi]); u_c.append(it)
    if u_r:
        X_u = sp.csr_matrix((np.ones(len(u_r), np.float32), (u_r, u_c)), shape=(max(u_r) + 1, n))
        udeg_tr = np.asarray(X_u.sum(1)).ravel()
        X_u_w = sp.diags(1.0 / np.sqrt(np.maximum(udeg_tr, 1.0))) @ X_u
        C_u = (X_u_w.T @ X_u_w).tocsr(); C_u.setdiag(0)
    else:
        C_u = sp.csr_matrix((n, n), dtype=np.float32)

    s_r, s_c = [], []
    for bi, b in enumerate(trb):
        if bi in b2s:
            for it in b: s_r.append(b2s[bi]); s_c.append(it)
    if s_r:
        X_s = sp.csr_matrix((np.ones(len(s_r), np.float32), (s_r, s_c)), shape=(max(s_r) + 1, n))
        sdeg_tr = np.asarray(X_s.sum(1)).ravel()
        X_s_w = sp.diags(1.0 / np.sqrt(np.maximum(sdeg_tr, 1.0))) @ X_s
        C_s = (X_s_w.T @ X_s_w).tocsr(); C_s.setdiag(0)
    else:
        C_s = sp.csr_matrix((n, n), dtype=np.float32)

    P = (sp.diags(1.0 / np.sqrt(op.freq + 1.0)) @ (op.C + 0.5 * C_s + 0.5 * C_u + 0.2 * C_pre) @ sp.diags(1.0 / np.sqrt(op.freq + 1.0))).tocsr()
    TF_bridge = normalize(tf_mat + 1.5 * (P @ tf_mat) + 0.5 * (P @ (P @ tf_mat)), norm="l2", axis=1).tocsr()
    BR_DENSE = (CONTENT + 1.5 * (P @ CONTENT) + 0.5 * (P @ (P @ CONTENT))).astype(np.float32)
    BR_DENSE /= (np.linalg.norm(BR_DENSE, axis=1, keepdims=True) + 1e-8)

    def build_pcat(lbl_arr, n_types, keep_diag=False):
        tr_, tc_ = [], []
        for bi, b in enumerate(trb):
            for it in b: tr_.append(bi); tc_.append(lbl_arr[it])
        T = sp.csr_matrix((np.ones(len(tr_), np.float32), (tr_, tc_)), shape=(len(trb), n_types))
        Ct = (T.T @ T).tocsr()
        if not keep_diag: Ct.setdiag(0); Ct.eliminate_zeros()
        fc = np.asarray(T.sum(0)).ravel() + 1e-9; totc = float(fc.sum()); coc = Ct.tocoo()
        pc_ = np.maximum(0., np.log((coc.data * totc) / (fc[coc.row] * fc[coc.col]) + 1e-12)).astype(np.float32)
        return sp.csr_matrix((pc_, (coc.row, coc.col)), shape=(n_types, n_types))

    Pcat_leaf = build_pcat(cat_lbl, NCc, keep_diag=False)
    Pcat_homo = build_pcat(cat_lbl, NCc, keep_diag=True)
    Pcat_l2 = build_pcat(cat_l2, NCl2, keep_diag=True)

    torch.manual_seed(0)
    m_con = ContrastEncoder(cdim).to(DEVICE)
    opt_con = torch.optim.AdamW(m_con.parameters(), lr=1e-3, weight_decay=1e-3)
    rng_c = random.Random(0); npr_c = np.random.default_rng(0)
    bids_c = [b for b in trb if len(b) >= 2]
    for _ in range(500):
        m_con.train(); sl, pi = [], []
        for _ in range(64):
            b = rng_c.choice(bids_c); it = b[:]; rng_c.shuffle(it); k = max(1, len(it) // 2)
            sl.append(it[:k]); pi.append(it[k])
        sr = m_con.set_rep(*pad_sets(sl, CONTENT_T, 8))
        cand = torch.cat([torch.tensor(pi, device=DEVICE), torch.tensor(npr_c.integers(n, size=128), device=DEVICE)])
        loss = Fn.cross_entropy(sr @ m_con.emb(CONTENT_T[cand]).t() / 0.07, torch.arange(len(sl), device=DEVICE))
        opt_con.zero_grad(); loss.backward(); opt_con.step()
    m_con.eval()

    SIG = ["simp", "diff2", "cooc", "ppmi", "csvd", "cat", "homo", "tfidf", "bridge", "pretrain", "contrast"]

    def score_split(bset, beta_clique_val=1.5):
        seeds_list = [s for s, _ in bset]; nb = len(seeds_list)
        Ind = op.build_indicator(seeds_list)
        s_pair_b, _ = op.score_pairwise_and_diff2(seeds_list)
        delta_inc, _ = op.score_simplicial_incidence_only(seeds_list, beta_clique=beta_clique_val)
        s_u = np.asarray((C_u @ Ind).todense()); s_s = np.asarray((C_s @ Ind).todense())
        s_pair = s_pair_b + 0.5 * s_u + 0.5 * s_s
        s_diff2_pair = np.asarray(P @ s_pair).astype(np.float32)
        s_simp = s_pair + delta_inc
        s_diff2 = np.asarray(P @ s_simp).astype(np.float32)
        s_rao = op.score_rao_concordance(seeds_list, dense_feat=BR_DENSE, cat_lbl=cat_lbl, gamma=-1.0, alpha_dense=0.25, tau=0.05)
        s_cooc = np.asarray((op.C @ Ind).todense()).astype(np.float32)
        s_ppmi = np.asarray((op.PPMI @ Ind).todense()).astype(np.float32)
        Us = np.stack([op.Ucsvd[s].mean(0) for s in seeds_list], 1)
        s_csvd = (op.Ucsvd @ Us).astype(np.float32)
        s_cat = np.zeros((n, nb), np.float32); s_homo = np.zeros((n, nb), np.float32)
        for j, s in enumerate(seeds_list):
            sa = [cat_lbl[i] for i in s]; sa2 = [cat_l2[i] for i in s]
            s_cat[:, j] = np.asarray(Pcat_leaf[sa].sum(0)).ravel()[cat_lbl]
            s_homo[:, j] = np.asarray(Pcat_homo[sa].sum(0)).ravel()[cat_lbl] + 0.5 * np.asarray(Pcat_l2[sa2].sum(0)).ravel()[cat_l2]
        s_tfidf = np.asarray((tf_mat @ (tf_mat.T @ Ind)).todense()).astype(np.float32)
        s_bridge = np.asarray((TF_bridge @ (TF_bridge.T @ Ind)).todense()).astype(np.float32)
        s_pre = np.asarray((C_pre @ Ind).todense()).astype(np.float32)
        with torch.no_grad():
            sf, sm = pad_sets(seeds_list, CONTENT_T, 8)
            sr = m_con.set_rep(sf, sm)
            s_con = (m_con.emb(CONTENT_T) @ sr.t()).cpu().numpy().astype(np.float32)
        R = {
            "simp": s_simp, "diff2": s_diff2, "pair": s_pair, "diff2_pair": s_diff2_pair,
            "rao": s_rao, "cooc": s_cooc, "ppmi": s_ppmi, "csvd": s_csvd,
            "cat": s_cat, "homo": s_homo, "tfidf": s_tfidf, "bridge": s_bridge,
            "pretrain": s_pre, "contrast": s_con,
        }
        Z = {k: zcols(R[k]) for k in SIG}
        Z["pair"] = zcols(R["pair"]); Z["diff2_pair"] = zcols(R["diff2_pair"]); Z["rao"] = zcols(R["rao"])
        return R, Z

    Rv, Zv = score_split(vset, beta_clique_val=1.5)
    Rt, Zt = score_split(tset, beta_clique_val=1.5)

    # Validation weight fit
    Zv_pair = dict(Zv); Zv_pair["simp"] = Zv["pair"]; Zv_pair["diff2"] = Zv["diff2_pair"]
    w = {k: 0.0 for k in SIG}; best = -1.0
    for _ in range(4):
        imp = False
        for k in SIG:
            for val in [0.0, 0.25, 0.5, 1.0, 1.5]:
                w2 = dict(w); w2[k] = val
                if sum(w2.values()) == 0: continue
                F_v = sum(w2[k2] * Zv_pair[k2] for k2 in w2 if w2[k2])
                nd = eval_ndcg20(F_v, vset)
                if nd > best + 2e-4: best = nd; w = w2; imp = True
        if not imp: break
    bw = EmpiricalBayesRouter.prune_weights_val(w, Zv_pair, lambda fmat: eval_ndcg20(fmat, vset), optional_keys=("contrast", "homo"), val_eps=3e-4)

    def build_Y(Z_dict, ho_params):
        alpha_inc, g_rao = ho_params
        Zp = dict(Z_dict); Zp["simp"] = Z_dict["pair"]; Zp["diff2"] = Z_dict["diff2_pair"]
        base_p = sum(bw[k] * Zp[k] for k in bw); base_i = sum(bw[k] * Z_dict[k] for k in bw)
        return (1.0 - alpha_inc) * base_p + alpha_inc * base_i + g_rao * Z_dict["rao"]

    best_ho = (0.5, 0.25); best_ho_nd = -1.0
    for alpha_inc in [0.0, 0.5, 1.0]:
        for g_rao in [0.0, 0.2, 0.35]:
            nd = eval_ndcg20(build_Y(Zv, (alpha_inc, g_rao)), vset)
            if nd > best_ho_nd + 1e-5: best_ho_nd = nd; best_ho = (alpha_inc, g_rao)

    prior_ks = [k for k in ("cat", "homo", "tfidf", "bridge", "pretrain", "contrast") if k in SIG and bw.get(k, 0) > 0] or ["bridge", "tfidf", "cat"]
    Y_v = build_Y(Zv, best_ho)
    raw_pr_v = sum(bw.get(k, 0.5) * Zv[k] for k in prior_ks) + best_ho[1] * Zv["rao"]
    frozen_prior_scale = float(np.std(Y_v) / (np.std(raw_pr_v) + 1e-9))
    mu0_v = raw_pr_v * frozen_prior_scale
    vc = EmpiricalBayesRouter.fit_variance_components(op.freq, Y_v, mu0_v)

    def apply_eb(Z_dict, ho_params, kf, p_scale):
        Y = build_Y(Z_dict, ho_params)
        if kf <= 0.0: return Y
        raw_pr = sum(bw.get(k, 0.5) * Z_dict[k] for k in prior_ks) + ho_params[1] * Z_dict["rao"]
        mu0 = raw_pr * float(p_scale)
        return EmpiricalBayesRouter.conjugate_posterior(Y, mu0, op.freq, kf)

    best_kf = 0.0; best_eb_nd = eval_ndcg20(Y_v, vset)
    for kf_cand in EmpiricalBayesRouter.shrinkage_candidates(vc, [0.0, 0.05, 0.1, 0.25, 0.5, vc["kappa_f"], 1.0, 2.0, 5.0]):
        nd = eval_ndcg20(apply_eb(Zv, best_ho, kf_cand, frozen_prior_scale), vset)
        if nd > best_eb_nd + 1e-5 or (kf_cand > 0 and nd >= best_eb_nd - 1e-4 and best_kf == 0.0):
            best_eb_nd = nd; best_kf = kf_cand
    if best_kf == 0.0 and vc.get("kappa_f", 0.0) > 0: best_kf = vc["kappa_f"]

    return {
        "op": op, "P": P, "tset": tset, "Zt": Zt, "bw": bw, "best_ho": best_ho,
        "best_kf": best_kf, "frozen_prior_scale": frozen_prior_scale,
        "build_Y": build_Y, "apply_eb": apply_eb, "score_split": score_split,
        "prior_ks": prior_ks, "C_u": C_u, "C_s": C_s, "SIG": SIG,
    }


def run_electronic_beta_sweep():
    print("\n--- Real Sweep 1: beta_clique on Electronic ---", flush=True)
    setup = setup_retail_dataset("electronic")
    tset = setup["tset"]; op = setup["op"]; P = setup["P"]
    seeds_list = [s for s, _ in tset]; Ind = op.build_indicator(seeds_list)
    s_pair_b, _ = op.score_pairwise_and_diff2(seeds_list)
    s_u = np.asarray((setup["C_u"] @ Ind).todense()); s_s = np.asarray((setup["C_s"] @ Ind).todense())
    s_pair = s_pair_b + 0.5 * s_u + 0.5 * s_s

    beta_vals = [0.0, 0.5, 1.0, 1.5, 2.5]
    res_el = {}
    for beta in beta_vals:
        if beta == 0.0:
            delta_inc = 0.0
        else:
            delta_inc, _ = op.score_simplicial_incidence_only(seeds_list, beta_clique=beta)
        s_simp = s_pair + delta_inc
        s_diff2 = np.asarray(P @ s_simp).astype(np.float32)
        Z_cur = dict(setup["Zt"])
        Z_cur["simp"] = zcols(s_simp)
        Z_cur["diff2"] = zcols(s_diff2)
        F_cand = setup["apply_eb"](Z_cur, setup["best_ho"], setup["best_kf"], setup["frozen_prior_scale"])
        rq, _ = eval_per_query_top20(F_cand, tset)
        r20 = float(np.mean(rq))
        res_el[str(beta)] = round(r20, 4)
        print(f"  electronic | beta_clique={beta:3.1f} -> Recall@20 = {100*r20:.2f}%", flush=True)
    return res_el


def run_clothing_eb_sweep():
    print("\n--- Real Sweep 3: EB Shrinkage kappa_f on Clothing ---", flush=True)
    setup = setup_retail_dataset("clothing")
    tset = setup["tset"]; op = setup["op"]
    Y_t = setup["build_Y"](setup["Zt"], setup["best_ho"])
    raw_pr_t = sum(setup["bw"].get(k, 0.5) * setup["Zt"][k] for k in setup["prior_ks"]) + setup["best_ho"][1] * setup["Zt"]["rao"]
    mu0_t = raw_pr_t * setup["frozen_prior_scale"]

    kf_vals = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]
    res_cl = {}
    for kf in kf_vals:
        if kf <= 0.0:
            F_cand = Y_t
        else:
            F_cand = EmpiricalBayesRouter.conjugate_posterior(Y_t, mu0_t, op.freq, kf)
        rq, top20 = eval_per_query_top20(F_cand, tset)
        ov = float(np.mean(rq))
        bk = eval_exposure_buckets(top20, tset, op.freq)
        res_cl[str(kf)] = {
            "overall_r20": round(ov, 4),
            "cold_r20": bk["cold"][0],
            "rare_r20": bk["rare"][0],
            "warm_r20": bk["warm"][0],
        }
        print(f"  clothing | kf={kf:4.2f} -> Overall: {100*ov:6.2f}% | Cold: {100*(bk['cold'][0] or 0):6.2f}% | Rare: {100*(bk['rare'][0] or 0):6.2f}% | Warm: {bk['warm'][0]}", flush=True)
    return res_cl


def run_dunnhumby_beta_sweep():
    print("\n--- Real Sweep 1: beta_clique on Dunnhumby ---", flush=True)
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
        for it in b: freq_all[it] += 1
    keep = {it for it, c in freq_all.items() if c >= 2}
    tmp = [([it for it in b if it in keep], u) for b, u in bask]
    tmp = [(b, u) for b, u in tmp if len(b) >= 3]
    items = sorted({i for b, _ in tmp for i in b}); imap = {it: k for k, it in enumerate(items)}; n = len(items)

    seen_b = set(); bundles_all = []
    for b, _ in tmp:
        mapped = [imap[i] for i in sorted(set(b))]
        key = tuple(mapped)
        if key not in seen_b:
            seen_b.add(key); bundles_all.append(mapped)

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
        if cat_lbl[i] < 0: cat_lbl[i] = len(catmap); catmap[f"_n{i}"] = cat_lbl[i]

    tf_mat = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(txt)
    tf_mat = normalize(tf_mat, norm="l2", axis=1).tocsr()
    cdim = min(128, tf_mat.shape[1] - 1)
    CONTENT = TruncatedSVD(cdim, random_state=0).fit_transform(tf_mat).astype(np.float32)
    CONTENT /= (np.linalg.norm(CONTENT, axis=1, keepdims=True) + 1e-8)

    perm = list(range(len(bundles_all)))
    random.Random(0).shuffle(perm)
    shuf_b = [bundles_all[i] for i in perm]
    ntr = int(0.7 * len(shuf_b)); nva = int(0.1 * len(shuf_b))
    trb, vab, teb = shuf_b[:ntr], shuf_b[ntr:ntr+nva], shuf_b[ntr+nva:]

    def mask_b(blist, seed):
        rng = random.Random(seed); out = []
        for b in blist:
            bb = b[:]; k = max(1, int(round(len(bb) * 0.5)))
            rng.shuffle(bb); out.append((bb[:k], bb[k:]))
        return out

    vset = mask_b(vab, 1)[:800]
    tset = mask_b(teb, 2)[:2000]
    op = SimplicialGraphOperator(trb, n_items=n, svd_dim=64, beta_clique=1.5)
    BR_DENSE = op.propagate_rw(CONTENT)
    BR_DENSE /= (np.linalg.norm(BR_DENSE, axis=1, keepdims=True) + 1e-8)

    SIG = ["simp", "diff2", "rao", "cooc", "ppmi", "csvd"]

    def score_split(bset, beta_val=1.5):
        seeds_list = [s for s, _ in bset]
        Ind = op.build_indicator(seeds_list)
        s_pair, s_diff2_pair = op.score_pairwise_and_diff2(seeds_list)
        delta_inc, diff2_delta = op.score_simplicial_incidence_only(seeds_list, beta_clique=beta_val)
        s_simp = s_pair + delta_inc; s_diff2 = s_diff2_pair + diff2_delta
        s_rao = op.score_rao_concordance(seeds_list, dense_feat=BR_DENSE, cat_lbl=cat_lbl, gamma=-1.0, alpha_dense=0.25, tau=0.05)
        s_cooc = np.asarray((op.C @ Ind).todense()).astype(np.float32)
        s_ppmi = np.asarray((op.PPMI @ Ind).todense()).astype(np.float32)
        Us = np.stack([op.Ucsvd[s].mean(0) for s in seeds_list], 1)
        s_csvd = (op.Ucsvd @ Us).astype(np.float32)
        R = {"simp": s_simp, "diff2": s_diff2, "pair": s_pair, "diff2_pair": s_diff2_pair, "rao": s_rao, "cooc": s_cooc, "ppmi": s_ppmi, "csvd": s_csvd}
        Z = {k: zcols(R[k]) for k in SIG}
        Z["pair"] = zcols(R["pair"]); Z["diff2_pair"] = zcols(R["diff2_pair"]); Z["rao"] = zcols(R["rao"])
        return R, Z

    Rv, Zv = score_split(vset, beta_val=1.5)
    Rt, Zt = score_split(tset, beta_val=1.5)

    Zv_pair = dict(Zv); Zv_pair["simp"] = Zv["pair"]; Zv_pair["diff2"] = Zv["diff2_pair"]
    w = {k: 0.0 for k in SIG}; best = -1.0
    for _ in range(4):
        imp = False
        for k in SIG:
            for val in [0.0, 0.25, 0.5, 1.0, 1.5]:
                w2 = dict(w); w2[k] = val
                if sum(w2.values()) == 0: continue
                F_v = sum(w2[k2] * Zv_pair[k2] for k2 in w2 if w2[k2])
                nd = eval_ndcg20(F_v, vset)
                if nd > best + 2e-4: best = nd; w = w2; imp = True
        if not imp: break
    bw = EmpiricalBayesRouter.prune_weights_val(w, Zv_pair, lambda fmat: eval_ndcg20(fmat, vset), optional_keys=("rao",), val_eps=3e-4)

    def build_Y(Z_dict, ho_params):
        alpha_inc, g_rao = ho_params
        Zp = dict(Z_dict); Zp["simp"] = Z_dict["pair"]; Zp["diff2"] = Z_dict["diff2_pair"]
        base_p = sum(bw[k] * Zp[k] for k in bw); base_i = sum(bw[k] * Z_dict[k] for k in bw)
        return (1.0 - alpha_inc) * base_p + alpha_inc * base_i + g_rao * Z_dict["rao"]

    best_ho = (0.5, 0.25); best_ho_nd = -1.0
    for alpha_inc in [0.0, 0.5, 1.0]:
        for g_rao in [0.0, 0.2, 0.35]:
            nd = eval_ndcg20(build_Y(Zv, (alpha_inc, g_rao)), vset)
            if nd > best_ho_nd + 1e-5: best_ho_nd = nd; best_ho = (alpha_inc, g_rao)

    prior_ks = [k for k in ("cooc", "ppmi", "csvd") if k in SIG and bw.get(k, 0) > 0] or ["cooc", "csvd"]
    Y_v = build_Y(Zv, best_ho)
    raw_pr_v = sum(bw.get(k, 0.5) * Zv[k] for k in prior_ks) + best_ho[1] * Zv["rao"]
    frozen_prior_scale = float(np.std(Y_v) / (np.std(raw_pr_v) + 1e-9))
    mu0_v = raw_pr_v * frozen_prior_scale
    vc = EmpiricalBayesRouter.fit_variance_components(op.freq, Y_v, mu0_v)

    def apply_eb(Z_dict, ho_params, kf, p_scale):
        Y = build_Y(Z_dict, ho_params)
        if kf <= 0.0: return Y
        raw_pr = sum(bw.get(k, 0.5) * Z_dict[k] for k in prior_ks) + ho_params[1] * Z_dict["rao"]
        mu0 = raw_pr * float(p_scale)
        return EmpiricalBayesRouter.conjugate_posterior(Y, mu0, op.freq, kf)

    best_kf = 0.0; best_eb_nd = eval_ndcg20(Y_v, vset)
    for kf_cand in EmpiricalBayesRouter.shrinkage_candidates(vc, [0.0, 0.05, 0.1, 0.25, 0.5, vc["kappa_f"], 1.0, 2.0, 5.0]):
        nd = eval_ndcg20(apply_eb(Zv, best_ho, kf_cand, frozen_prior_scale), vset)
        if nd > best_eb_nd + 1e-5 or (kf_cand > 0 and nd >= best_eb_nd - 1e-4 and best_kf == 0.0):
            best_eb_nd = nd; best_kf = kf_cand
    if best_kf == 0.0 and vc.get("kappa_f", 0.0) > 0: best_kf = vc["kappa_f"]

    seeds_list = [s for s, _ in tset]
    s_pair, s_diff2_pair = op.score_pairwise_and_diff2(seeds_list)
    beta_vals = [0.0, 0.5, 1.0, 1.5, 2.5]
    res_dh = {}
    for beta in beta_vals:
        if beta == 0.0:
            delta_inc = 0.0; diff2_delta = 0.0
        else:
            delta_inc, diff2_delta = op.score_simplicial_incidence_only(seeds_list, beta_clique=beta)
        s_simp = s_pair + delta_inc; s_diff2 = s_diff2_pair + diff2_delta
        Z_cur = dict(Zt)
        Z_cur["simp"] = zcols(s_simp); Z_cur["diff2"] = zcols(s_diff2)
        F_cand = apply_eb(Z_cur, best_ho, best_kf, frozen_prior_scale)
        rq, _ = eval_per_query_top20(F_cand, tset)
        r20 = float(np.mean(rq))
        res_dh[str(beta)] = round(r20, 4)
        print(f"  dunnhumby | beta_clique={beta:3.1f} -> Recall@20 = {100*r20:.2f}%", flush=True)
    return res_dh


# ======================================================================
# 2. Multimodal Helper: POG_dense, POG, and Spotify
# ======================================================================
def setup_multimodal_dataset(ds: str):
    pog_dir = os.path.join(REPO_ROOT, "bundlemllm", "datasets", ds)
    MUSIC = not os.path.exists(os.path.join(pog_dir, "title_emb.npy"))
    n = json.load(open(os.path.join(pog_dir, "count.json")))["#I"]

    def rb(f):
        o = {}
        for l in open(os.path.join(pog_dir, f)):
            p = l.strip().split(",")
            if len(p) >= 2: o[int(p[0])] = [int(x) for x in p[1:]]
        return o

    def _loadpt(f):
        x = torch.load(os.path.join(pog_dir, f), map_location="cpu", weights_only=False).detach().float().numpy()
        return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    if os.path.exists(os.path.join(pog_dir, "bi_full.txt")):
        fullb = list(rb("bi_full.txt").values())
    else:
        fullb = list(rb("bi_train.txt").values())

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
        cmap = {}; cat_lbl = np.zeros(n, int)
        for k, v in info.items():
            a = v.get("artist_uri", "unk")
            cmap.setdefault(a, len(cmap)); cat_lbl[int(k)] = cmap[a]
        NC = int(cat_lbl.max()) + 1
        almap = {}; alb_lbl = np.zeros(n, int)
        for k, v in info.items():
            al = v.get("album_uri", "unk")
            almap.setdefault(al, len(almap)); alb_lbl[int(k)] = almap[al]
    else:
        NC = 100; cat_lbl = np.zeros(n, int)

    random.Random(0).shuffle(allb_master)
    ntr = int(0.7 * len(allb_master)); nva = int(0.1 * len(allb_master))
    trb = allb_master[:ntr]; vab = allb_master[ntr:ntr+nva]; teb = allb_master[ntr+nva:]

    def mask_b(bundles, s0):
        r = random.Random(s0); inp, gt = {}, {}
        for i, b in enumerate(bundles):
            bb = b[:]; r.shuffle(bb); k = max(1, int(round(len(bb) * 0.5)))
            s, h = bb[:k], bb[k:]
            if not h: s, h = bb[:-1], [bb[-1]]
            inp[i], gt[i] = s, h
        return [(inp[i], gt[i]) for i in range(len(bundles))]

    vset = mask_b(vab, 1)[:400]
    tset = mask_b(teb, 2)[:2000]

    op = SimplicialGraphOperator(trb, n_items=n, svd_dim=64, beta_clique=1.5)
    onto = DualSpectrumOntology(
        {i: b for i, b in enumerate(trb)}, cat_lbl=cat_lbl, n_items=n, alb_lbl=alb_lbl,
        cluster_feat=(clusterfeat if not MUSIC else None), n_soft_clusters=400, soft_tau=0.05, device=DEVICE,
    )
    if MUSIC:
        feats["audiostyle"] = BilateralContentBridge.disentangle(feats["audio"], cat_lbl, NC)
        feats["bridge"] = BilateralContentBridge.build_dense_bridge(feats["audiostyle"], op, alpha_bridge=0.5)
        rao_dense_feat = feats["audiostyle"]
    else:
        feats["style"] = BilateralContentBridge.disentangle(title, cat_lbl, NC)
        if "image" in feats: feats["imgstyle"] = BilateralContentBridge.disentangle(feats["image"], cat_lbl, NC)
        if "desc" in feats: feats["descstyle"] = BilateralContentBridge.disentangle(feats["desc"], cat_lbl, NC)
        st_parts = [feats["style"]] + ([feats["imgstyle"]] if "imgstyle" in feats else []) + ([feats["descstyle"]] if "descstyle" in feats else [])
        st_all = np.concatenate(st_parts, axis=1).astype(np.float32)
        st_all /= (np.linalg.norm(st_all, axis=1, keepdims=True) + 1e-8)
        feats["bridge"] = BilateralContentBridge.build_dense_bridge(st_all, op, alpha_bridge=0.5)
        rao_dense_feat = feats["style"]

    base_f = feats["content"] if "content" in feats else feats["audio"]
    parts = [base_f] + ([feats["image"]] if "image" in feats else []) + ([feats["desc"]] if "desc" in feats else [])
    Fnp = np.concatenate(parts, 1).astype(np.float32)
    FEAT = torch.tensor(Fnp, device=DEVICE)

    torch.manual_seed(0)
    model = ContrastEncoder(Fnp.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    bids = list(range(len(trb))); rng = random.Random(0); npr = np.random.default_rng(0)
    for _ in range(500 if not MUSIC else 300):
        model.train(); sl, pi = [], []
        for _ in range(64):
            b = rng.choice(bids); it = trb[b][:]; rng.shuffle(it); k = max(1, len(it) // 2)
            sl.append(it[:k]); pi.append(it[k])
        sr = model.set_rep(*pad_sets(sl, FEAT, 40 if MUSIC else 4))
        cand = torch.cat([torch.tensor(pi, device=DEVICE), torch.tensor(npr.integers(n, size=128), device=DEVICE)])
        loss = Fn.cross_entropy(sr @ model.emb(FEAT[cand]).t() / 0.07, torch.arange(len(sl), device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): Emb = model.emb(FEAT)

    feats["csvd"] = op.Ucsvd
    FEAT_T = {k: torch.tensor(v, device=DEVICE) for k, v in feats.items()}
    SIG = (["simp", "diff2", "pop", "cooc", "ppmi", "csvd", "cat", "homo", "contrast", "audiostyle", "bridge"]) if MUSIC else \
          (["cat", "softcat", "style", "imgstyle", "bridge", "simp", "diff2", "pop", "cooc", "ppmi", "csvd", "contrast"])
    SIG = [s for s in SIG if (s not in ("imgstyle",) or "imgstyle" in FEAT_T)]

    def extract_signals(bundles, beta_val=1.5, gamma_val=(1.0 if MUSIC else -1.0), tau_val=0.05):
        nb = len(bundles); seeds_list = [s for s, _ in bundles]
        Ind = op.build_indicator(seeds_list)
        sc_pair, sc_diff2_pair = op.score_pairwise_and_diff2(seeds_list)
        delta_inc, diff2_delta = op.score_simplicial_incidence_only(seeds_list, beta_clique=beta_val)
        sc_simp = sc_pair + delta_inc; sc_diff2 = sc_diff2_pair + diff2_delta
        sc_rao = op.score_rao_concordance(seeds_list, dense_feat=rao_dense_feat, cat_lbl=cat_lbl, gamma=gamma_val, alpha_dense=0.15, tau=tau_val)
        R = {
            "simp": torch.tensor(sc_simp, dtype=torch.float32, device=DEVICE),
            "diff2": torch.tensor(sc_diff2, dtype=torch.float32, device=DEVICE),
            "pair": torch.tensor(sc_pair, dtype=torch.float32, device=DEVICE),
            "diff2_pair": torch.tensor(sc_diff2_pair, dtype=torch.float32, device=DEVICE),
            "delta_inc": torch.tensor(delta_inc, dtype=torch.float32, device=DEVICE),
            "rao": torch.tensor(sc_rao, dtype=torch.float32, device=DEVICE),
            "pop": torch.tensor(np.tile(np.log1p(op.freq)[:, None], (1, nb)), dtype=torch.float32, device=DEVICE),
            "cooc": torch.tensor(np.asarray((op.C @ Ind).todense()), dtype=torch.float32, device=DEVICE),
            "ppmi": torch.tensor(np.asarray((op.PPMI @ Ind).todense()), dtype=torch.float32, device=DEVICE),
        }
        out_cat = np.zeros((n, nb), np.float32)
        for j, s in enumerate(seeds_list):
            sa = [cat_lbl[i] for i in s]
            out_cat[:, j] = np.asarray(onto.Pcat[sa].sum(0)).ravel()[cat_lbl]
        if not MUSIC: out_cat = out_cat + 0.5 * onto.score_slot_saturation(seeds_list)
        R["cat"] = torch.tensor(out_cat, dtype=torch.float32, device=DEVICE)
        if MUSIC:
            art_sc = np.asarray((onto.I_art_w @ (onto.I_art_w.T @ Ind)).todense())
            alb_sc = np.asarray((onto.I_alb_w @ (onto.I_alb_w.T @ Ind)).todense())
            R["homo"] = torch.tensor((art_sc + 2.0 * alb_sc).astype(np.float32), device=DEVICE)
        else:
            Ms = torch.stack([onto.Msoft_t[s].sum(0) for s in seeds_list], 1)
            R["softcat"] = onto.MP_t @ Ms
        with torch.no_grad():
            sr = model.set_rep(*pad_sets(seeds_list, FEAT, 40 if MUSIC else 4))
            R["contrast"] = Emb @ sr.t()
        for k in ("csvd", "style", "imgstyle", "audiostyle", "bridge"):
            if k in FEAT_T:
                f_mat = FEAT_T[k]
                M = torch.stack([f_mat[s].mean(0) for s in seeds_list], 1)
                R[k] = f_mat @ M
        Z = {k: zcols(R[k]) for k in SIG}
        Z["pair"] = zcols(R["pair"]); Z["diff2_pair"] = zcols(R["diff2_pair"]); Z["rao"] = zcols(R["rao"])
        return R, Z

    Rv, Zv = extract_signals(vset)
    Z_pair_v = dict(Zv); Z_pair_v["simp"] = Zv["pair"]; Z_pair_v["diff2"] = Zv["diff2_pair"]
    GPS = {k: [0.0, 0.25, 0.5, 1.0, 1.5] for k in SIG}
    w = {k: 0.0 for k in SIG}; best_nd = -1.0
    for _ in range(4):
        improved = False
        for k in SIG:
            for v in GPS[k]:
                w2 = dict(w); w2[k] = v
                if sum(w2.values()) == 0: continue
                fused = sum(w2[k2] * Z_pair_v[k2] for k2 in SIG if w2[k2])
                nd = eval_ndcg20(fused, vset)
                if nd > best_nd + 2e-4: best_nd = nd; w = w2; improved = True
        if not improved: break
    bw = EmpiricalBayesRouter.prune_weights_val(w, Z_pair_v, lambda fmat: eval_ndcg20(fmat, vset), optional_keys=("bridge", "contrast", "softcat", "homo"), val_eps=3e-4)

    prior_cand = ("cat", "homo", "audiostyle", "bridge", "contrast") if MUSIC else ("cat", "softcat", "style", "imgstyle", "bridge", "contrast")
    prior_keys = [k for k in prior_cand if k in SIG and bw.get(k, 0) > 0] or [k for k in prior_cand if k in SIG]

    base_pair_v = sum(bw[k] * Z_pair_v[k] for k in bw)
    base_inc_v = sum(bw[k] * Zv[k] for k in bw)
    best_ho = (0.0, 0.0); best_ho_nd = eval_ndcg20(base_pair_v, vset)
    for alpha_inc in [0.0, 0.5, 1.0]:
        for g_rao in [0.0, 0.2, 0.35]:
            cand_v = (1.0 - alpha_inc) * base_pair_v + alpha_inc * base_inc_v + g_rao * Zv["rao"]
            nd = eval_ndcg20(cand_v, vset)
            if nd > best_ho_nd + 1e-5: best_ho_nd = nd; best_ho = (alpha_inc, g_rao)

    alpha_inc, g_rao = best_ho
    Y_v = (1.0 - alpha_inc) * base_pair_v + alpha_inc * base_inc_v + g_rao * Zv["rao"]
    raw_prior_v = sum(bw.get(k, 0.5) * Zv[k] for k in prior_keys) + g_rao * Zv["rao"]
    if bw.get("pop", 0) > 0: raw_prior_v = raw_prior_v + bw["pop"] * Zv["pop"]
    val_prior_scale = float(((Y_v.std() + 1e-9) / (raw_prior_v.std() + 1e-9)).item())
    mu0_scaled_v = raw_prior_v * val_prior_scale
    vc = EmpiricalBayesRouter.fit_variance_components(op.freq, Y_v, mu0_scaled_v)
    freq_t = torch.tensor(op.freq, dtype=torch.float32, device=DEVICE)[:, None]
    best_eb = (0.0, val_prior_scale); best_eb_nd = eval_ndcg20(Y_v, vset)
    for p_scale in (val_prior_scale, 1.0):
        mu0_cand = raw_prior_v * p_scale
        for kf in EmpiricalBayesRouter.shrinkage_candidates(vc, [0.0, 0.05, 0.1, 0.5, 1.0, vc["kappa_f"], 5.0]):
            f_eb = EmpiricalBayesRouter.conjugate_posterior(Y_v, mu0_cand, freq_t, kf)
            nd = eval_ndcg20(f_eb, vset)
            if nd > best_eb_nd + 1e-5 or (kf > 0 and nd >= best_eb_nd - 1e-4 and best_eb[0] == 0.0):
                best_eb_nd = nd; best_eb = (kf, p_scale)
    if best_eb[0] == 0.0 and vc.get("kappa_f", 0.0) > 0: best_eb = (vc["kappa_f"], val_prior_scale)

    def apply_model(Z_dict, ho_tuple=best_ho):
        Zp = dict(Z_dict); Zp["simp"] = Z_dict["pair"]; Zp["diff2"] = Z_dict["diff2_pair"]
        base_p = sum(bw[k] * Zp[k] for k in bw)
        base_i = sum(bw[k] * Z_dict[k] for k in bw)
        Y = (1.0 - ho_tuple[0]) * base_p + ho_tuple[0] * base_i + ho_tuple[1] * Z_dict["rao"]
        if best_eb[0] <= 0.0: return Y
        raw_pr = sum(bw.get(k, 0.5) * Z_dict[k] for k in prior_keys) + ho_tuple[1] * Z_dict["rao"]
        if bw.get("pop", 0) > 0: raw_pr = raw_pr + bw["pop"] * Z_dict["pop"]
        mu_t = raw_pr * float(best_eb[1])
        return EmpiricalBayesRouter.conjugate_posterior(Y, mu_t, freq_t, best_eb[0])

    return {
        "op": op, "tset": tset, "bw": bw, "best_ho": best_ho, "best_eb": best_eb,
        "extract_signals": extract_signals, "apply_model": apply_model,
        "rao_dense_feat": rao_dense_feat, "cat_lbl": cat_lbl,
    }


def run_pog_dense_beta_sweep():
    print("\n--- Real Sweep 1: beta_clique on POG_dense ---", flush=True)
    setup = setup_multimodal_dataset("pog_dense")
    tset = setup["tset"]; op = setup["op"]
    seeds_list = [s for s, _ in tset]
    Rt, Zt_base = setup["extract_signals"](tset, beta_val=0.0)

    beta_vals = [0.0, 0.5, 1.0, 1.5, 2.5]
    res_pd = {}
    for beta in beta_vals:
        if beta == 0.0:
            Zt = dict(Zt_base)
        else:
            delta_inc, diff2_delta = op.score_simplicial_incidence_only(seeds_list, beta_clique=beta)
            sc_simp = Rt["pair"] + torch.tensor(delta_inc, dtype=torch.float32, device=DEVICE)
            sc_diff2 = Rt["diff2_pair"] + torch.tensor(diff2_delta, dtype=torch.float32, device=DEVICE)
            Zt = dict(Zt_base)
            Zt["simp"] = zcols(sc_simp)
            Zt["diff2"] = zcols(sc_diff2)
        F_cand = setup["apply_model"](Zt)
        rq, _ = eval_per_query_top20(F_cand, tset)
        r20 = float(np.mean(rq))
        res_pd[str(beta)] = round(r20, 4)
        print(f"  pog_dense | beta_clique={beta:3.1f} -> Recall@20 = {100*r20:.2f}%", flush=True)
    return res_pd


def run_multimodal_rao_sweep(ds: str):
    print(f"\n--- Real Sweep 2: Rao gamma/tau on {ds} ---", flush=True)
    setup = setup_multimodal_dataset(ds)
    tset = setup["tset"]; op = setup["op"]
    seeds_list = [s for s, _ in tset]
    Rt, Zt_base = setup["extract_signals"](tset)
    rao_dense_feat = setup["rao_dense_feat"]
    cat_lbl = setup["cat_lbl"]

    gamma_vals = [-1.0, 0.0, 1.0]
    tau_vals = [0.01, 0.05, 0.10, 0.50]
    res = {}
    for g in gamma_vals:
        for t in tau_vals:
            key = f"gamma={g:+.1f}_tau={t:.2f}"
            sc_rao = op.score_rao_concordance(seeds_list, dense_feat=rao_dense_feat, cat_lbl=cat_lbl, gamma=g, alpha_dense=0.15, tau=t)
            Zt = dict(Zt_base)
            Zt["rao"] = zcols(torch.tensor(sc_rao, dtype=torch.float32, device=DEVICE))
            F_cand = setup["apply_model"](Zt, ho_tuple=(setup["best_ho"][0], 0.25))
            rq, _ = eval_per_query_top20(F_cand, tset)
            r20 = float(np.mean(rq))
            res[key] = round(r20, 4)
            print(f"  {ds} | {key} -> Recall@20 = {100*r20:.2f}%", flush=True)
    return res


if __name__ == "__main__":
    t_start = time.time()
    all_sweeps = {"beta_clique": {}, "rao_kernel_damping": {}, "eb_shrinkage": {}}

    # 1. beta_clique sweeps
    all_sweeps["beta_clique"]["electronic"] = run_electronic_beta_sweep()
    all_sweeps["beta_clique"]["pog_dense"] = run_pog_dense_beta_sweep()
    all_sweeps["beta_clique"]["dunnhumby"] = run_dunnhumby_beta_sweep()

    # 2. Rao sweeps
    all_sweeps["rao_kernel_damping"]["pog"] = run_multimodal_rao_sweep("pog")
    all_sweeps["rao_kernel_damping"]["spotify"] = run_multimodal_rao_sweep("spotify")

    # 3. EB shrinkage sweep
    all_sweeps["eb_shrinkage"]["clothing"] = run_clothing_eb_sweep()

    out_file = os.path.join(RESULTS_DIR, "sensitivity_sweeps.json")
    with open(out_file, "w") as f:
        json.dump(all_sweeps, f, indent=2)
    print(f"\nSaved 100% REAL measured sensitivity sweeps to {out_file} ({time.time()-t_start:.1f}s)", flush=True)
