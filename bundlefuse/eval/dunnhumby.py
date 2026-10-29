"""BundleFuse evaluator for Dunnhumby 'The Complete Journey' (retail grocery), 7:1:2 full-ranking.

Transaction baskets are treated as (implicit) bundles; there are no curated training bundles, so the
whole method is self-supervised over basket co-occurrence. Runs the shared closed-form algorithm:

1. Weighted fusion of the collaborative / taxonomic / content signals (:mod:`bundlefuse.operators`).
2. Higher-order terms: simplicial incidence surplus (Delta_inc) + Rao quadratic concordance (R_{-1}).
3. Cold-start shrinkage theta_hat = (1 - u_c)*Y + u_c*mu_0 toward a metadata prior, with the prior
   scale frozen on validation so test scoring is independent of test batching.

Weights/mixing/shrinkage are selected on validation via ``SEL_OBJ`` (see :mod:`bundlefuse.selection`).
Env modes (each exits after printing/writing): ``BASELINES`` (matched neighbourhood suite),
``ABLATE`` (stage ablation + exposure buckets), ``VALGAP`` (selection generalization gap),
``SELDUMP``/``MATRIX_STD`` (dump the matrix / per-seed std). Default prints the full-ranking matrix.
Run: ``python -m bundlefuse.eval.dunnhumby``  (see ``docs/EXPERIMENTS.md``).
"""
import os, sys, csv, math, random, time
from collections import defaultdict
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bundlefuse._device import DEVICE
from bundlefuse.operators import SimplicialGraphOperator, EmpiricalBayesRouter, zcols
from bundlefuse.paths import DATA_ROOT, RESULTS_DIR
from bundlefuse import selection, protocol

ROOT = os.environ.get("DUNN_DIR") or os.path.join(DATA_ROOT, "dunnhumby")
MASK_FRAC = float(os.environ.get("MASK_FRAC", "0.5")); SEEDS = int(os.environ.get("SEEDS", "5"))
SPLIT_MODE = os.environ.get("SPLIT_MODE", "random")
BASKET_CAP = int(os.environ.get("BASKET_CAP", "30000")); MIN_FREQ = 2
ENCODER = os.environ.get("ENCODER", "0") != "0"  # default off: zero-param closed-form. Set ENCODER=1 to add the optional set-attention encoder
VAL_N, TEST_N = 800, 2000
csv.field_size_limit(10**7)

t0 = time.time()
raw = defaultdict(set); buser = {}; bday = {}
with open(os.path.join(ROOT, "transaction_data.csv")) as f:
    r = csv.reader(f); h = next(r)
    bi = h.index("BASKET_ID"); pi = h.index("PRODUCT_ID"); hi = h.index("household_key"); di = h.index("DAY")
    for row in r:
        bid = row[bi]
        raw[bid].add(int(row[pi]))
        buser[bid] = row[hi]
        bday[bid] = int(row[di])

bask = [(sorted(v), buser[bid], bday[bid]) for bid, v in raw.items() if 3 <= len(v) <= 20]
random.Random(0).shuffle(bask); bask = bask[:BASKET_CAP]
freq_all = defaultdict(int)
for b, _, _ in bask:
    for it in b:
        freq_all[it] += 1
keep = {it for it, c in freq_all.items() if c >= MIN_FREQ}
tmp = [([it for it in b if it in keep], u, d) for b, u, d in bask]
tmp = [(b, u, d) for b, u, d in tmp if len(b) >= 3]
items = sorted({i for b, _, _ in tmp for i in b}); imap = {it: k for k, it in enumerate(items)}; n = len(items)

seen_b = set()
umap = {}; bundles_all = []; users_all = []; days_all = []
for b, u, d in tmp:
    mapped = [imap[i] for i in sorted(set(b))]
    key = tuple(mapped)
    if key not in seen_b:
        seen_b.add(key)
        bundles_all.append(mapped)
        users_all.append(umap.setdefault(u, len(umap)))
        days_all.append(d)
nusers = len(umap)

cat_lbl = np.full(n, -1, int); catmap = {}
txt = ["" for _ in range(n)]
with open(os.path.join(ROOT, "product.csv")) as f:
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

tf_mat = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(txt)
tf_mat = normalize(tf_mat, norm="l2", axis=1).tocsr()
cdim = min(128, tf_mat.shape[1] - 1)
CONTENT = TruncatedSVD(cdim, random_state=0).fit_transform(tf_mat).astype(np.float32)
CONTENT /= (np.linalg.norm(CONTENT, axis=1, keepdims=True) + 1e-8)
CONTENT_T = torch.tensor(CONTENT, device=DEVICE)


class Contrast(nn.Module):
    def __init__(self, d, hd=128, p=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(p), nn.Linear(d, hd), nn.ReLU(), nn.Dropout(p), nn.Linear(hd, hd))
        self.k = nn.Linear(hd, hd); self.q = nn.Parameter(torch.randn(hd)); self.h = hd

    def emb(self, f):
        return self.enc(f)

    def set_rep(self, sf, sm):
        e = self.emb(sf)
        a = (self.k(e) @ self.q / math.sqrt(self.h)).masked_fill(~sm.bool(), -1e9).softmax(-1)
        return (a[..., None] * e).sum(1)


def train_contrast(train):
    MK = 8; torch.manual_seed(0)
    m = Contrast(CONTENT.shape[1]).to(DEVICE); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    bids = [b for b in train if len(b) >= 2]; rng = random.Random(0); npr = np.random.default_rng(0)

    def pad(sl):
        sf = torch.zeros(len(sl), MK, CONTENT.shape[1], device=DEVICE); sm = torch.zeros(len(sl), MK, device=DEVICE)
        for i, s in enumerate(sl):
            s = s[:MK]; sf[i, :len(s)] = CONTENT_T[s]; sm[i, :len(s)] = 1
        return sf, sm

    for _ in range(400):
        sl, pint = [], []
        for _ in range(128):
            b = rng.choice(bids); it = b[:]; rng.shuffle(it); k = max(1, len(it) // 2); sl.append(it[:k]); pint.append(it[k])
        sr = m.set_rep(*pad(sl)); cand = torch.cat([torch.tensor(pint, device=DEVICE), torch.tensor(npr.integers(n, size=256), device=DEVICE)])
        loss = Fn.cross_entropy(sr @ m.emb(CONTENT_T[cand]).t() / 0.1, torch.arange(len(sl), device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        E = m.emb(CONTENT_T)
    return m, E, pad


def train_item2vec(train, dim=128, epochs=10, neg=5, bs=4096):
    pr, pc = [], []
    for b in train:
        for i in range(len(b)):
            for j in range(len(b)):
                if i != j:
                    pr.append(b[i]); pc.append(b[j])
    pr = np.asarray(pr, np.int64); pc = np.asarray(pc, np.int64)
    freq_i = np.bincount(np.concatenate([np.asarray(b) for b in train]), minlength=n).astype(np.float64)
    pn = (freq_i + 1e-9) ** 0.75; pn /= pn.sum()
    U = nn.Embedding(n, dim).to(DEVICE); V = nn.Embedding(n, dim).to(DEVICE)
    nn.init.normal_(U.weight, std=0.1); nn.init.normal_(V.weight, std=0.1)
    opt = torch.optim.Adam(list(U.parameters()) + list(V.parameters()), lr=0.01)
    npr = np.random.default_rng(0)
    for _ in range(epochs):
        perm = npr.permutation(len(pr))
        for s in range(0, len(pr), bs):
            idx = perm[s:s + bs]; c = torch.tensor(pr[idx], device=DEVICE); o = torch.tensor(pc[idx], device=DEVICE)
            ng = torch.tensor(npr.choice(n, size=(len(idx), neg), p=pn), device=DEVICE)
            uc = U(c); pos = Fn.logsigmoid((uc * V(o)).sum(-1))
            negl = Fn.logsigmoid(-(uc.unsqueeze(1) * V(ng)).sum(-1)).sum(-1)
            loss = -(pos + negl).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return U.weight.detach().cpu().numpy().astype(np.float32), V.weight.detach().cpu().numpy().astype(np.float32)


def build(trb, tru):
    graph_op = SimplicialGraphOperator(trb, n_items=n, svd_dim=128, alpha_deg=0.5, gamma_vol=0.5, beta_clique=1.5)
    freq = graph_op.freq

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

    p_cont = graph_op.propagate(CONTENT)
    p_cont /= (np.linalg.norm(p_cont, axis=1, keepdims=True) + 1e-8)
    BR_CONT = CONTENT + 0.5 * p_cont
    BR_CONT /= (np.linalg.norm(BR_CONT, axis=1, keepdims=True) + 1e-8)

    model, E, pad = train_contrast(trb) if ENCODER else (None, None, None)
    return graph_op, freq, C_hh, Pcat, BR_CONT, model, E, pad


SIG = ["simp", "diff2", "pop", "cooc", "hh_cooc", "ppmi", "csvd", "cat", "content", "bridge", "contrast"]
if not ENCODER:
    SIG = [s for s in SIG if s != "contrast"]


def score_all(bundles, built):
    graph_op, freq, C_hh, Pcat, BR_CONT, model, E, pad = built
    nb = len(bundles)
    seeds_list = [s for s, _ in bundles]
    Ind = graph_op.build_indicator(seeds_list)
    Ind_w = Ind.multiply(graph_op.inv_sqrt_f[:, None]).tocsr()

    # Pairwise base vs Pure Simplicial Incidence Surplus (`score_simplicial_incidence_only`)
    sc_pair, sc_diff2_pair = graph_op.score_pairwise_and_diff2(seeds_list)
    delta_inc, diff2_delta = graph_op.score_simplicial_incidence_only(seeds_list, beta_clique=1.5)
    sc_simp = sc_pair + delta_inc
    sc_diff2 = sc_diff2_pair + diff2_delta

    # Redundancy-Corrected Quadratic Concordance (`score_rao_concordance`)
    sc_rao = graph_op.score_rao_concordance(
        seeds_list, dense_feat=BR_CONT, cat_lbl=cat_lbl, gamma=-1.0, alpha_dense=0.15, tau=0.05
    )

    sc_pop = np.tile(np.log1p(freq)[:, None], (1, nb)).astype(np.float32)
    sc_cooc = np.asarray((graph_op.C @ Ind).todense()).astype(np.float32)
    sc_hh = np.asarray((C_hh @ Ind_w).todense()).astype(np.float32)
    sc_ppmi = np.asarray((graph_op.PPMI @ Ind).todense()).astype(np.float32)
    Us = np.stack([graph_op.Ucsvd[s].mean(0) for s in seeds_list], 1)
    sc_csvd = (graph_op.Ucsvd @ Us).astype(np.float32)
    sc_cat = np.zeros((n, nb), np.float32)
    for j, s in enumerate(seeds_list):
        sa = [cat_lbl[i] for i in s]
        sc_cat[:, j] = np.asarray(Pcat[sa].sum(0)).ravel()[cat_lbl]
    Cs = np.stack([CONTENT[s].mean(0) for s in seeds_list], 1)
    sc_cont = (CONTENT @ Cs).astype(np.float32)
    Bs = np.stack([BR_CONT[s].mean(0) for s in seeds_list], 1)
    sc_br = (BR_CONT @ Bs).astype(np.float32)
    sc_tf = np.asarray((tf_mat @ (tf_mat.T @ Ind)).todense()).astype(np.float32)
    sc_con = None
    if ENCODER:
        with torch.no_grad():
            sr = model.set_rep(*pad([s[:8] for s in seeds_list]))
            sc_con = (E @ sr.t()).cpu().numpy().astype(np.float32)
    R = {
        "simp": sc_simp, "diff2": sc_diff2, "pair": sc_pair, "diff2_pair": sc_diff2_pair,
        "rao": sc_rao, "pop": sc_pop, "cooc": sc_cooc, "hh_cooc": sc_hh,
        "ppmi": sc_ppmi, "csvd": sc_csvd, "cat": sc_cat, "content": sc_cont,
        "bridge": sc_br, "contrast": sc_con, "tfidf": sc_tf,
    }
    Z = {k: zcols(R[k]) for k in SIG}
    Z["pair"] = zcols(R["pair"])
    Z["diff2_pair"] = zcols(R["diff2_pair"])
    Z["rao"] = zcols(R["rao"])
    Z["tfidf"] = zcols(R["tfidf"])
    return R, Z


def eval_mat(F_mat, bundles, Ks=(10, 20, 50)):
    nb = len(bundles); rng = np.random.default_rng(0)
    F = F_mat + rng.random((n, nb)).astype(np.float32) * 1e-6
    for j, (s, _) in enumerate(bundles):
        F[s, j] = -1e9
    top = np.argpartition(-F, max(Ks), axis=0)[:max(Ks)]
    acc = {K2: ([], []) for K2 in Ks}
    for j, (_, gt) in enumerate(bundles):
        col = top[:, j]; order = col[np.argsort(-F[col, j])]; g = set(gt)
        for K2 in Ks:
            hits = [1 if p in g else 0 for p in order[:K2]]
            dcg = sum(x / math.log2(i + 2) for i, x in enumerate(hits))
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(g), K2)))
            acc[K2][0].append(sum(hits) / len(g)); acc[K2][1].append(dcg / idcg if idcg else 0.0)
    return {K2: (float(np.mean(acc[K2][0])), float(np.mean(acc[K2][1]))) for K2 in Ks}


# Validation selection objective, decoupled from the reported cutoff (see bundlefuse.selection).
SEL_OBJ = os.environ.get("SEL_OBJ", "ndcg20")


def sel_score(F, bundles):
    """Selection objective (higher=better); see :mod:`bundlefuse.selection`."""
    return selection.sel_score(F, bundles, SEL_OBJ, eval_mat)


def mask(bundles, seed):
    """Reveal-fraction masking into (seeds, held_out) pairs; see :mod:`bundlefuse.protocol`."""
    return protocol.mask(bundles, seed, MASK_FRAC)


def ease_test_scores(trb, vset, tset, n_items, lam_grid=(1.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)):
    """EASE closed-form item-item baseline (Steck WWW'19), lambda selected on val."""
    rows, cols = [], []
    for bi, b in enumerate(trb):
        for it in b:
            rows.append(bi); cols.append(it)
    Btr = sp.csr_matrix((np.ones(len(rows), np.float64), (rows, cols)), shape=(len(trb), n_items))
    G = (Btr.T @ Btr).toarray().astype(np.float64)
    di = np.diag_indices(n_items)

    def score_set(W, bundles):
        M = np.zeros((n_items, len(bundles)), np.float32)
        for j, (s, _) in enumerate(bundles):
            M[:, j] = W[s].sum(0)
        return M
    best_nd, best_W = -1.0, None
    for lam in lam_grid:
        Gl = G.copy(); Gl[di] += lam
        P = np.linalg.inv(Gl)
        W = -P / np.diag(P)[None, :]; W[di] = 0.0
        nd = eval_mat(score_set(W, vset), vset, Ks=(20,))[20][1]
        if nd > best_nd:
            best_nd, best_W = nd, W
    return score_set(best_W, tset)


METH = ["popularity", "co-occurrence", "content-kNN", "item2vec", "Pairwise-Repl", "SIMBA-NoRes", "Ours"]
EASE = os.environ.get("EASE") and n <= 10000  # EASE needs an n x n solve: only tractable on small catalogues
if EASE:
    METH = METH + ["EASE"]
agg = {m: {10: ([], []), 20: ([], []), 50: ([], [])} for m in METH}
cold_recalls = []
VALGAP = os.environ.get("VALGAP")  # validation->test generalization gap of greedy weight selection
vg_rows = []
AB = os.environ.get("ABLATE")  # operator OFAT ablation + exposure-bucket / variance-component diagnostics
ab_op, ab_vc = [], []
ab_bk = {"cold": [0, 0, 0], "rare": [0, 0, 0], "warm": [0, 0, 0]}  # [pre_hit, post_hit, tot]
BASELINES = os.environ.get("BASELINES")  # matched-protocol tuned-neighbourhood baselines
BL_SEEDS = []
if BASELINES:
    from bundlefuse import baselines as _BL

if SPLIT_MODE == "chrono":
    chrono_idx = sorted(range(len(bundles_all)), key=lambda i: (days_all[i], i))
    ntr = int(0.7 * len(chrono_idx)); nva = int(0.1 * len(chrono_idx))
    tri, vai, tei = chrono_idx[:ntr], chrono_idx[ntr:ntr + nva], chrono_idx[ntr + nva:]
    trb_c = [bundles_all[i] for i in tri]; tru_c = [users_all[i] for i in tri]
    vab_c = [bundles_all[i] for i in vai]; teb_c = [bundles_all[i] for i in tei]
    built_c = build(trb_c, tru_c); Ui_c, Vi_c = train_item2vec(trb_c)

for sd in range(SEEDS):
    if SPLIT_MODE == "chrono":
        trb, tru, vab, teb = trb_c, tru_c, vab_c, teb_c
        built, Ui, Vi = built_c, Ui_c, Vi_c
    else:
        idx = list(range(len(bundles_all))); random.Random(sd).shuffle(idx)
        ntr = int(0.7 * len(idx)); nva = int(0.1 * len(idx))
        tri, vai, tei = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
        trb = [bundles_all[i] for i in tri]; tru = [users_all[i] for i in tri]
        vab = [bundles_all[i] for i in vai]; teb = [bundles_all[i] for i in tei]
        built = build(trb, tru); Ui, Vi = train_item2vec(trb)

    freq = built[1]
    vset = mask(vab, sd + 1)[:VAL_N]; tset = mask(teb, sd + 2)[:TEST_N]
    if BASELINES:
        BL_SEEDS.append(_BL.run_matched_seed(built[0], freq, eval_mat, vset, tset))
        continue
    Rv, Zv = score_all(vset, built)
    Rt, Zt = score_all(tset, built)

    # 1. Fit base weights on Pairwise signals first + validation-tolerance pruning
    Zv_pair = dict(Zv); Zv_pair["simp"] = Zv["pair"]; Zv_pair["diff2"] = Zv["diff2_pair"]
    w = {k: 0.0 for k in SIG}; best = -1e18
    for _ in range(4):
        imp = False
        for k in SIG:
            for val in [0.0, 0.25, 0.5, 1.0, 1.5]:
                w2 = dict(w); w2[k] = val
                if sum(w2.values()) == 0:
                    continue
                F_v = sum(w2[k2] * Zv_pair[k2] for k2 in w2 if w2[k2])
                nd = sel_score(F_v, vset)
                if nd > best + 2e-4:
                    best = nd; w = w2; imp = True
        if not imp:
            break
    bw = EmpiricalBayesRouter.prune_weights_val(
        w, Zv_pair, lambda fmat: sel_score(fmat, vset),
        optional_keys=("bridge", "contrast", "cat"), val_eps=3e-4
    )

    if VALGAP:
        # Fusion-only (no higher-order, no shrinkage) NDCG@20 at SELECTED vs UNIFORM weights,
        # on val and test, so the gap isolates weight tuning. Same definition as eval/fullrank.py.
        Zt_pair = dict(Zt); Zt_pair["simp"] = Zt["pair"]; Zt_pair["diff2"] = Zt["diff2_pair"]
        def _nd(Zp, split, wd):
            F = sum(wd[k] * Zp[k] for k in SIG if wd.get(k, 0))
            return eval_mat(F, split, Ks=(20,))[20][1]
        w_un = {k: 1.0 for k in SIG}
        v_sel, t_sel = _nd(Zv_pair, vset, bw), _nd(Zt_pair, tset, bw)
        v_un, t_un = _nd(Zv_pair, vset, w_un), _nd(Zt_pair, tset, w_un)
        vg_rows.append((v_sel, t_sel, v_un, t_un, sum(1 for k in bw if bw[k] > 0)))
        continue

    # 2. Nested Higher-Order Mixing (alpha_inc in [0, 1], gamma_rao >= 0)
    def build_Y(Z_dict, ho_params):
        alpha_inc, g_rao = ho_params
        Zp = dict(Z_dict); Zp["simp"] = Z_dict["pair"]; Zp["diff2"] = Z_dict["diff2_pair"]
        base_p = sum(bw[k] * Zp[k] for k in bw)
        base_i = sum(bw[k] * Z_dict[k] for k in bw)
        return (1.0 - alpha_inc) * base_p + alpha_inc * base_i + g_rao * Z_dict["rao"]

    best_ho = (1.0, 0.0)
    best_ho_nd = -1e18
    for a_inc in [0.0, 0.5, 1.0]:
        for g_rao in [0.0, 0.15, 0.25]:
            nd = sel_score(build_Y(Zv, (a_inc, g_rao)), vset)
            if nd >= best_ho_nd - 1e-5:
                best_ho_nd = nd; best_ho = (a_inc, g_rao)
    best = max(best, best_ho_nd)

    prior_ks = [k for k in ("cat", "content", "bridge", "contrast") if bw.get(k, 0) > 0] or ["cat", "bridge"]

    # 3. Exact Conjugate Normal-Normal Posterior: theta_hat = (1 - u_c) * Y + u_c * mu_0
    # Freeze prior_scale on the validation split (`Zv`) so test inference is strictly independent of test batching
    Y_v = build_Y(Zv, best_ho)
    raw_pr_v = sum(bw.get(k, 0.5) * Zv[k] for k in prior_ks) + best_ho[1] * Zv["rao"]
    if bw.get("pop", 0) > 0:
        raw_pr_v = raw_pr_v + bw["pop"] * Zv["pop"]
    frozen_prior_scale = float(np.std(Y_v) / (np.std(raw_pr_v) + 1e-9))
    mu0_v = raw_pr_v * frozen_prior_scale
    vc = EmpiricalBayesRouter.fit_variance_components(freq, Y_v, mu0_v)

    def apply_eb(Z_dict, ho_params, kf, p_scale):
        Y = build_Y(Z_dict, ho_params)
        if kf <= 0.0:
            return Y
        raw_prior = sum(bw.get(k, 0.5) * Z_dict[k] for k in prior_ks) + ho_params[1] * Z_dict["rao"]
        if bw.get("pop", 0) > 0:
            raw_prior = raw_prior + bw["pop"] * Z_dict["pop"]
        mu_0 = raw_prior * float(p_scale)
        return EmpiricalBayesRouter.conjugate_posterior(Y, mu_0, freq, kf)

    best_kf = 0.0  # honest null: apply shrinkage only if a candidate beats no-shrinkage on validation
    for kf_cand in EmpiricalBayesRouter.shrinkage_candidates(vc, [0.0, 0.05, 0.25, vc["kappa_f"], 1.0]):
        nd = sel_score(apply_eb(Zv, best_ho, kf_cand, frozen_prior_scale), vset)
        if nd > best + 1e-4:
            best = nd; best_kf = kf_cand

    if AB:
        alpha, g = best_ho; scale = frozen_prior_scale
        st = {"base": build_Y(Zt, (0.0, 0.0)), "inc": build_Y(Zt, (alpha, 0.0)), "rao": build_Y(Zt, (0.0, g)),
              "both": build_Y(Zt, (alpha, g)), "full": apply_eb(Zt, best_ho, best_kf, scale)}
        ab_op.append({k: eval_mat(v, tset, Ks=(20,))[20][0] for k, v in st.items()})
        ab_vc.append((vc.get("raw_sigma0_sq"), vc.get("raw_sigma_eps_sq"), vc.get("kappa_f"), best_kf))
        for nm, F_mat in (("pre", st["base"]), ("post", st["full"])):
            F = np.asarray(F_mat) + np.random.default_rng(0).random(np.asarray(F_mat).shape).astype(np.float32) * 1e-6
            for j, (s, _) in enumerate(tset):
                F[s, j] = -1e9
            top = np.argpartition(-F, 20, axis=0)[:20]
            for j, (_, gt) in enumerate(tset):
                col = set(top[:, j])
                for it in gt:
                    b = "cold" if freq[it] == 0 else ("rare" if freq[it] <= 5 else "warm")
                    if nm == "pre":
                        ab_bk[b][2] += 1; ab_bk[b][0] += int(it in col)
                    else:
                        ab_bk[b][1] += int(it in col)
        continue

    Us_t = np.stack([Ui[s].mean(0) for s, _ in tset], 1)
    i2v_mat = zcols((Vi @ Us_t).astype(np.float32))
    F_ours = apply_eb(Zt, best_ho, best_kf, frozen_prior_scale)

    F_tmp = F_ours.copy()
    for j, (s, _) in enumerate(tset):
        F_tmp[s, j] = -1e9
    top20 = np.argpartition(-F_tmp, 20, axis=0)[:20]
    ch, ct = 0, 0
    for j, (_, gt) in enumerate(tset):
        p = set(top20[:, j])
        for g_it in gt:
            if freq[g_it] == 0:
                ct += 1
                if g_it in p:
                    ch += 1
    cold_recalls.append(ch / max(ct, 1))

    res = {
        "popularity": eval_mat(Zt["pop"], tset),
        "co-occurrence": eval_mat(Zt["cooc"], tset),
        "content-kNN": eval_mat(Zt["content"], tset),
        "item2vec": eval_mat(i2v_mat, tset),
        "Pairwise-Repl": eval_mat(apply_eb(Zt, (0.0, 0.0), best_kf, frozen_prior_scale), tset),
        "SIMBA-NoRes": eval_mat(build_Y(Zt, best_ho), tset),
        "Ours": eval_mat(F_ours, tset),
    }
    if EASE:
        res["EASE"] = eval_mat(ease_test_scores(trb, vset, tset, n), tset)
    for m in METH:
        for K2 in (10, 20, 50):
            agg[m][K2][0].append(res[m][K2][0]); agg[m][K2][1].append(res[m][K2][1])

if BASELINES:
    import json as _json
    order_m = ["ItemKNN", "P3alpha", "RP3beta", "SLIM", "iALS", "BundleKNN", "LinearCombiner"]
    present = [m for m in order_m if any(m in s for s in BL_SEEDS)]
    out = {"dataset": "dunnhumby", "n": int(n), "seeds": len(BL_SEEDS), "methods": {}}
    print(f"\n=== dunnhumby BASELINES ({len(BL_SEEDS)} seeds; matched protocol; {time.time()-t0:.0f}s) ===", flush=True)
    print(f"{'method':14s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8}", flush=True)
    for m in present:
        acc = {K2: ([], []) for K2 in (10, 20)}
        for s in BL_SEEDS:
            if m in s:
                for K2 in (10, 20):
                    acc[K2][0].append(s[m][K2][0]); acc[K2][1].append(s[m][K2][1])
        vals = {K2: (float(np.mean(acc[K2][0])), float(np.mean(acc[K2][1]))) for K2 in (10, 20)}
        out["methods"][m] = {"R@10": vals[10][0], "N@10": vals[10][1], "R@20": vals[20][0], "N@20": vals[20][1],
                             "seeds": len(acc[20][0])}
        print(f"{m:14s} {vals[10][0]:8.4f} {vals[10][1]:8.4f} {vals[20][0]:8.4f} {vals[20][1]:8.4f}", flush=True)
    if "SLIM" not in present:
        print(f"dunnhumby SLIM skipped (n={n:,} > n_max; EASE covers the sparse-linear family)", flush=True)
    _json.dump(out, open(os.path.join(RESULTS_DIR, "baselines_dunnhumby.json"), "w"), indent=2)
    sys.exit()

if AB:
    o = {k: float(np.mean([r[k] for r in ab_op])) for k in ("base", "inc", "rao", "both", "full")}
    print(f"ABLATE_OP dunnhumby: base={o['base']:.4f} inc={o['inc']:.4f} rao={o['rao']:.4f} both={o['both']:.4f} "
          f"full={o['full']:.4f} | d_inc={100*(o['inc']-o['base']):+.2f}pp d_rao={100*(o['rao']-o['base']):+.2f}pp "
          f"d_both={100*(o['both']-o['base']):+.2f}pp d_shrink={100*(o['full']-o['both']):+.2f}pp", flush=True)
    s0v = [x[0] for x in ab_vc if x[0] is not None]; sev = [x[1] for x in ab_vc if x[1] is not None]
    s0s = f"{np.mean(s0v):.3f}" if s0v else "NA"; ses = f"{np.mean(sev):.3f}" if sev else "NA"
    kpl = float(np.mean([x[2] for x in ab_vc])); kva = float(np.mean([x[3] for x in ab_vc]))
    def _bk(b):
        h0, h1, t = ab_bk[b]
        return (100*h0/t if t else 0.0, 100*h1/t if t else 0.0, t)
    cpre, cpost, ct = _bk("cold"); rpre, rpost, rt = _bk("rare"); wpre, wpost, wt = _bk("warm")
    print(f"ABLATE_VAR dunnhumby: sigma0_sq={s0s} sigma_eps_sq={ses} kappa_plug={kpl:.3f} kappa_val={kva:.3f} "
          f"| cold {cpre:.1f}->{cpost:.1f} (n={ct}) rare {rpre:.1f}->{rpost:.1f} (n={rt}) warm {wpre:.1f}->{wpost:.1f} (n={wt})", flush=True)
    sys.exit()

if VALGAP:
    a = np.array([r[:4] for r in vg_rows]); m = a.mean(0)
    v_sel, t_sel, v_un, t_un = m
    nsig = np.mean([r[4] for r in vg_rows])
    print(f"VALGAP dunnhumby: SELECTED val={v_sel:.4f} test={t_sel:.4f} gap={v_sel-t_sel:+.4f} | "
          f"UNIFORM val={v_un:.4f} test={t_un:.4f} gap={v_un-t_un:+.4f} | "
          f"selection_excess_gap={(v_sel-t_sel)-(v_un-t_un):+.4f} | "
          f"test_sel-test_unif={t_sel-t_un:+.4f} | nsig={nsig:.1f} | seeds={len(vg_rows)}", flush=True)
    sys.exit()

print(f"\n=== dunnhumby MATRIX (mode={SPLIT_MODE}; sel={SEL_OBJ}; {SEEDS} seeds; cold R@20={100*np.mean(cold_recalls):.2f}%; {time.time()-t0:.0f}s) ===", flush=True)
print(f"{'method':16s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8} {'R@50':>8} {'N@50':>8}", flush=True)
for m in METH:
    a = agg[m]
    print(f"{m:16s} {np.mean(a[10][0]):8.4f} {np.mean(a[10][1]):8.4f} {np.mean(a[20][0]):8.4f} {np.mean(a[20][1]):8.4f} {np.mean(a[50][0]):8.4f} {np.mean(a[50][1]):8.4f}", flush=True)
if os.environ.get("SELDUMP"):
    import json as _json
    outp = os.path.join(RESULTS_DIR, f"selcmp_dunnhumby_{SEL_OBJ}.json")
    dump = {"dataset": "dunnhumby", "sel_obj": SEL_OBJ, "n": int(n), "seeds": SEEDS,
            "methods": {m: {f"{M}@{K2}": float(np.mean(agg[m][K2][i]))
                            for K2 in (10, 20, 50) for i, M in enumerate(("R", "N"))} for m in METH}}
    _json.dump(dump, open(outp, "w"), indent=2)
    print(f"dunnhumby SELDUMP ({SEL_OBJ}) -> {outp}", flush=True)
if os.environ.get("MATRIX_STD", "0") == "1":
    # population std across the SEEDS held-out splits (matches fullrank error bars)
    for m in METH:
        print(f"STD {m} {np.std(agg[m][20][0]):.4f} n={len(agg[m][20][0])}", flush=True)
