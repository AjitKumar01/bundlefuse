"""EmbFuse: compact, GRAPH-FREE bundle completion. The signals are encoded as per-item EMBEDDINGS
(obtained by SVD -- the estimator that works on sparse data where gradient-SGNS starves), so INFERENCE
is embedding lookups + dot products + a 2-scalar shrinkage. No co-occurrence matrix, no graph, no 2GB
feature matrix at serve time -- just a small per-item embedding table.

  per item i:  e_i = [ u_i^csvd (SVD of cooc) | u_i^ppmi (SVD of PPMI) | t_i (title content) ]
  score(c|S) = (1-g(f_c)) * sum_block w_b <e_c^b, sum_{s in S} e_s^b>   +   g(f_c) * w_prior <t_c, sum t_s>
               \__ collaborative blocks (head/rare via csvd+ppmi) __/        \__ content prior (cold/rare) __/
  g(f) = kappa/(f+kappa)

Trainable parameters: block weights w_b, prior weight, kappa, prior scale (~6). The REPRESENTATION lives
in the embeddings (csvd/ppmi/title), which ARE the learned-signal parameters -- computed by SVD because
the data is too sparse to gradient-learn them (GatedBundle collapsed). Captures PAIRWISE co-occurrence
(csvd=head, ppmi=rare/mid) + content; it CANNOT capture BundleFuse's higher-order simplicial term, which
is not a pairwise dot -- that is the honest cost of graph-free inference. Protocol: fit on VAL (internal
hold-out early-stop), report TEST once. Env: DS, DIM, EPOCHS, LR. Run: DS=pog_dense python3 reports/embfuse.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import ranking_metrics, top_items

DS = os.environ.get("DS", "pog_dense")
DIM = int(os.environ.get("DIM", "128"))
EPOCHS = int(os.environ.get("EPOCHS", "120"))
LR = float(os.environ.get("LR", "0.05"))
SEED = int(os.environ.get("SEED", "0"))
EDGES = [(0, 0, "cold"), (1, 2, "rare"), (3, 10, "mid"), (11, 10**12, "head")]
REF = {"pog_dense": {"CLHE": (0.3815, 0.2771), "BundleFuse": (0.3969, 0.2914)},
       "spotify": {"CLHE": (0.1082, 0.1806), "RaMen": (0.1091, 0.1882), "BundleFuse": (0.1486, 0.2532)},
       "spotify_sparse": {"CLHE": (0.0969, 0.1150), "BundleFuse": (0.1620, 0.2265)}}


def log(*a):
    print(*a, flush=True)


def bkof(f):
    for i, (lo, hi, nm) in enumerate(EDGES):
        if lo <= f <= hi:
            return i
    return 3


def svd_emb(M, d, n):
    s = TruncatedSVD(n_components=min(d, n - 2), random_state=0)
    A = s.fit_transform(M).astype(np.float32)        # candidate side (n,d)
    B = s.components_.T.astype(np.float32)            # seed side (n,d)
    return torch.from_numpy(A), torch.from_numpy(B)


class EmbFuse(nn.Module):
    def __init__(self, blocks, freq_t):
        super().__init__()
        # blocks: dict name -> (Ucand (n,d), Vseed (n,d))
        self.names = list(blocks.keys())
        self.Uc = nn.ParameterList([nn.Parameter(blocks[k][0], requires_grad=False) for k in self.names])
        self.Vs = nn.ParameterList([nn.Parameter(blocks[k][1], requires_grad=False) for k in self.names])
        self.w = nn.Parameter(torch.full((len(self.names),), 0.5))
        self.w_prior = nn.Parameter(torch.tensor(0.3))
        self.prior_block = self.names.index("title") if "title" in self.names else len(self.names) - 1
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(np.expm1(5.0)))))
        self.freq_t = freq_t

    def seed_sums(self, seeds):
        return [self.Vs[b][seeds].sum(0) for b in range(len(self.names))]   # list of (d,)

    def score_all(self, seeds):
        ss = self.seed_sums(seeds)
        w = Fn.softplus(self.w)
        Y = sum(w[b] * (self.Uc[b] @ ss[b]) for b in range(len(self.names)))    # (n,)
        pb = self.prior_block
        mu0 = Fn.softplus(self.w_prior) * (self.Uc[pb] @ ss[pb])
        kappa = Fn.softplus(self.log_kappa); u = kappa / (self.freq_t + kappa)
        return (1 - u) * Y + u * mu0


def recall(model, pairs, Ks=(10, 20), subset=None):
    acc = {k: np.zeros(2) for k in Ks}; cnt = 0
    with torch.no_grad():
        items = pairs if subset is None else [pairs[i] for i in subset]
        for seeds, gt in items:
            sc = model.score_all(list(seeds))
            m = ranking_metrics(sc[:, None], [(list(seeds), list(gt))], Ks)
            for k in Ks:
                acc[k] += np.asarray(m[k])
            cnt += 1
    return {k: tuple(acc[k] / cnt) for k in Ks}


def buckets(model, pairs, freq, k=20):
    buck = {nm: [0, 0] for _, _, nm in EDGES}
    with torch.no_grad():
        for seeds, gt in pairs:
            sc = model.score_all(list(seeds)).numpy()
            top = set(top_items(sc, list(seeds), k).tolist())
            for t in set(gt):
                b = buck[EDGES[bkof(int(freq[t]))][2]]; b[1] += 1; b[0] += int(t in top)
    return {nm: (v[0] / v[1] if v[1] else float("nan")) for nm, v in buck.items()}


def main():
    t0 = time.time()
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", DS)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    prov = ProvidedSplits(dd, n)
    g = SimplicialGraphOperator(prov.train, n_items=n, svd_dim=2)
    freq = g.freq.astype(np.float64); freq_t = torch.tensor(freq, dtype=torch.float32)
    log(f"{DS}: building signal embeddings (SVD)... ({time.time()-t0:.0f}s)")
    Uc_cooc, Vc_cooc = svd_emb(g.W_pair.tocsr(), DIM, n)
    Uc_ppmi, Vc_ppmi = svd_emb(g.PPMI.tocsr(), DIM, n)
    # content: prefer title text (pog) else content_feature
    tpath = os.path.join(dd, "title_emb.npy")
    if os.path.exists(tpath):
        T = np.load(tpath)
    else:
        T = torch.load(os.path.join(dd, "content_feature.pt"), map_location="cpu", weights_only=False).detach().float().numpy()
    T = (T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)).astype(np.float32); T = torch.from_numpy(T)
    blocks = {"csvd": (Uc_cooc, Vc_cooc), "ppmi": (Uc_ppmi, Vc_ppmi), "title": (T, T)}
    foot = sum(b[0].numel() + b[1].numel() for b in blocks.values()) * 4 / 1e6
    model = EmbFuse(blocks, freq_t)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"{DS}: blocks={list(blocks.keys())} | embedding table {foot:.0f}MB (graph-free) | trainable params={trainable}")

    val = prov.pairs("valid"); test = prov.pairs("test")
    nv = len(val); perm = rng.permutation(nv); nfit = int(0.8 * nv)
    fit_idx, es_idx = perm[:nfit], perm[nfit:]
    vpos = []
    for seeds, gt in val:
        c = [x for x in gt if x not in set(seeds)]; vpos.append(c[0] if c else (gt[0] if gt else 0))

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    best, best_state, stale = -1, None, 0
    for ep in range(1, EPOCHS + 1):
        o = fit_idx.copy(); rng.shuffle(o); losses = []
        for j in o:
            seeds = list(val[j][0])
            sc = model.score_all(seeds)
            sc[seeds] = -1e9
            loss = Fn.cross_entropy(sc[None, :], torch.tensor([vpos[j]]))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        if ep % 3 == 0 or ep == EPOCHS:
            r = recall(model, val, Ks=(20,), subset=es_idx)[20][0]
            if r > best + 1e-5:
                best, best_state, stale = r, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            log(f"    [ep={ep}] es_R@20={r:.4f} best={best:.4f} w={[round(float(x),2) for x in Fn.softplus(model.w)]} kappa={float(Fn.softplus(model.log_kappa.detach())):.1f} ({time.time()-t0:.0f}s)")
            if stale >= 5:
                break
    if best_state:
        model.load_state_dict(best_state)

    rr = recall(model, test, Ks=(10, 20)); bk = buckets(model, test, freq)
    ref = REF.get(DS, {})
    log(f"\n  {DS}  EmbFuse (graph-free, embeddings-only)  ({time.time()-t0:.0f}s)")
    log(f"    {'method':12s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8}")
    for nm, v in ref.items():
        log(f"    {nm:12s} {'':8} {'':8} {v[0]:8.4f} {v[1]:8.4f}")
    log(f"    {'EmbFuse':12s} {rr[10][0]:8.4f} {rr[10][1]:8.4f} {rr[20][0]:8.4f} {rr[20][1]:8.4f}")
    log(f"    buckets: cold {bk['cold']:.4f}  rare {bk['rare']:.4f}  mid {bk['mid']:.4f}  head {bk['head']:.4f}")
    log(f"    INFERENCE: {foot:.0f}MB embedding table, {trainable} trainable scalars, graph-free dot products")


if __name__ == "__main__":
    main()
