"""GatedBundle: parameter-only, graph-free bundle completion. Learns the association SIGNAL (not a frozen
kernel) and a frequency GATE (the routing mechanism the input-swap ablation proved was the real missing
piece -- more kernels could not get head AND rare/mid at once without per-item routing).

  score(c|S) = <u_c, sum_s v_s>            (learned PMI association; SGNS => Levy-Goldberg PMI)
             + (1 - g(f_c)) * b_c          (gated popularity; g(f)=kappa/(f+kappa), learned kappa)

Head candidate (f large, g->0): full popularity bias -> behaves like raw co-occurrence -> head recall.
Rare candidate (f small, g->1): popularity stripped -> pure learned PMI -> rare/mid recall. A single
global fusion cannot do both; the per-candidate gate can (parametric empirical-Bayes shrinkage).

Two-stage training: (1) SGNS on co-occurrence pairs learns U,V (dense supervision -> fixes the earlier
data-starvation of a raw embedding table on sparse bundle CE); (2) freeze U,V, learn b and kappa on the
bundle listwise CE (the gate learns the head/rare routing from the task). Inference is one mat-vec + a
gated bias -- embeddings only, no graph. Eval: per-target Hit@20 by TRAIN frequency bucket, same buckets
as the BundleFuse/ MMFuse references. Env: DATASETS, DIM, SGNS_EPOCHS, GATE_EPOCHS, NEG, SGNS_NEG, LR,
INIT_KAPPA, JOINT (1=also fine-tune U,V in stage 2). Run: DATASETS=pog_dense python3 reports/gated_bundle.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reports.mmfuse as M
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT, PROJECT_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import top_items

DATASETS = os.environ.get("DATASETS", "pog_dense").split(",")
DIM = int(os.environ.get("DIM", "128"))
SGNS_EPOCHS = int(os.environ.get("SGNS_EPOCHS", "6"))
GATE_EPOCHS = int(os.environ.get("GATE_EPOCHS", "25"))
SGNS_NEG = int(os.environ.get("SGNS_NEG", "5"))
SGNS_BS = int(os.environ.get("SGNS_BS", "4096"))
SGNS_MAXPAIRS = int(os.environ.get("SGNS_MAXPAIRS", "3000000"))
SGNS_LR = float(os.environ.get("SGNS_LR", "0.01"))
NEG = int(os.environ.get("NEG", "32"))
HARD_FRAC = float(os.environ.get("HARD_FRAC", "0.5"))
BATCH = int(os.environ.get("BATCH", "128"))
SEED_CAP = int(os.environ.get("SEED_CAP", "20"))
LR = float(os.environ.get("LR", "0.01"))
INIT_KAPPA = float(os.environ.get("INIT_KAPPA", "5.0"))
COOC_SRC = os.environ.get("COOC_SRC", "svd_wpair")   # svd_wpair (count factorisation, head) | sgns_uniform
JOINT = os.environ.get("JOINT", "0") != "0"
VAL_N = int(os.environ.get("VAL_N", "400"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "5"))
EDGES = [(0, 0, "cold(f=0)"), (1, 2, "rare(1-2)"), (3, 10, "mid(3-10)"), (11, 10**12, "head(>10)")]
ORDER = ["cold(f=0)", "rare(1-2)", "mid(3-10)", "head(>10)"]
BF = {"pog_dense": {"cold(f=0)": 0.0000, "rare(1-2)": 0.0935, "mid(3-10)": 0.1578, "head(>10)": 0.7503}}


def log(*a):
    print(*a, flush=True)


def bucket(f):
    for lo, hi, name in EDGES:
        if lo <= f <= hi:
            return name
    return "head(>10)"


class GatedBundle(nn.Module):
    def __init__(self, n, d, init_kappa):
        super().__init__()
        # pmi pathway (unigram^0.75 negatives -> PMI, rare/mid) ; cooc pathway (uniform negatives ->
        # popularity-preserving log P(c|s), head). Both LEARNED by skip-gram; the only difference is the
        # negative distribution (the debiasing knob). A learned gate SELECTS between them per candidate.
        self.Up = nn.Embedding(n, d); self.Vp = nn.Embedding(n, d)
        self.Uc = nn.Embedding(n, d); self.Vc = nn.Embedding(n, d)
        self.gb = nn.Parameter(torch.ones(2))    # learned per-pathway weight (active from start)
        for e in (self.Up, self.Vp, self.Uc, self.Vc):
            nn.init.normal_(e.weight, std=0.1)
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(np.expm1(init_kappa)))))

    def kappa(self):
        return Fn.softplus(self.log_kappa)

    def gate(self, f):
        k = self.kappa()
        return k / (f + k)                        # ->1 for rare (pmi), ->0 for head (cooc)

    def score_batch(self, cidx, sidx, smask, fcand):
        vsp = (self.Vp(sidx) * smask[..., None]).sum(1)                 # (B,d)
        vsc = (self.Vc(sidx) * smask[..., None]).sum(1)
        s_pmi = (self.Up(cidx) * vsp.unsqueeze(1)).sum(-1)              # (B,ncand)
        s_cooc = (self.Uc(cidx) * vsc.unsqueeze(1)).sum(-1)
        g = self.gate(fcand)                                           # (B,ncand)
        return self.gb[0] * g * s_pmi + self.gb[1] * (1.0 - g) * s_cooc

    @torch.no_grad()
    def score_all(self, seeds, freq_t, n):
        s = list(seeds)
        s_pmi = self.Up.weight @ self.Vp.weight[s].sum(0)              # (n,)
        s_cooc = self.Uc.weight @ self.Vc.weight[s].sum(0)
        g = self.gate(freq_t)
        return (self.gb[0] * g * s_pmi + self.gb[1] * (1.0 - g) * s_cooc).cpu().numpy()


def build_pairs(bundles, rng):
    pr, pc = [], []
    order = list(range(len(bundles))); rng.shuffle(order)
    for idx in order:
        b = bundles[idx]
        for a in range(len(b)):
            for c in range(len(b)):
                if a != c:
                    pr.append(b[a]); pc.append(b[c])
        if len(pr) >= SGNS_MAXPAIRS:
            break
    return np.asarray(pr, np.int64), np.asarray(pc, np.int64)


def train_sgns(U, V, pr, pc, pn, n, rng, label):
    """Learn an association embedding by SGNS. The negative distribution `pn` is the debiasing knob:
    pn ~ freq^0.75 -> PMI (rare); pn ~ uniform -> log P(c|s) (popularity-preserving, head)."""
    opt = torch.optim.Adam([U.weight, V.weight], lr=SGNS_LR)
    t0 = time.time()
    for ep in range(1, SGNS_EPOCHS + 1):
        perm = rng.permutation(len(pr)); losses = []
        for s in range(0, len(pr), SGNS_BS):
            idx = perm[s:s + SGNS_BS]
            i = U(torch.from_numpy(pr[idx])); j = V(torch.from_numpy(pc[idx]))
            ng = V(torch.from_numpy(rng.choice(n, size=(len(idx), SGNS_NEG), p=pn)))
            pos = Fn.logsigmoid((i * j).sum(-1))
            negl = Fn.logsigmoid(-(i.unsqueeze(1) * ng).sum(-1)).sum(-1)
            loss = -(pos + negl).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        if ep == 1 or ep == SGNS_EPOCHS:
            log(f"    [sgns:{label} ep={ep}] loss={np.mean(losses):.4f} ({time.time()-t0:.0f}s)")


def train_gate(model, C, bundles, freq, freq_t, val_pairs, n, rng):
    """Stage 2: freeze pathways (unless JOINT), learn gate kappa + pathway weights gb on bundle CE."""
    params = [model.gb, model.log_kappa] + (
        [model.Up.weight, model.Vp.weight, model.Uc.weight, model.Vc.weight] if JOINT else [])
    opt = torch.optim.Adam(params, lr=LR)
    train_b = [list(b) for b in bundles if len(b) >= 2]
    steps = max(1, len(train_b) // BATCH)
    best, best_state, t0 = -1e9, None, time.time()
    for ep in range(1, GATE_EPOCHS + 1):
        order = rng.permutation(len(train_b)); losses = []
        for st in range(steps):
            idx = order[(st * BATCH) % len(order): (st * BATCH) % len(order) + BATCH]
            if len(idx) < 2:
                continue
            sidx, smask, cidx = M.make_batch(C, [train_b[i] for i in idx], rng, n, NEG, SEED_CAP, HARD_FRAC)
            fcand = freq_t[cidx]
            logits = model.score_batch(cidx, sidx, smask, fcand)
            loss = Fn.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        if ep % EVAL_EVERY == 0 or ep == GATE_EPOCHS:
            v = quick_warm_hit(model, val_pairs, freq_t, freq, n)
            if v > best + 1e-5:
                best = v; best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
            log(f"    [gate ep={ep}] loss={np.mean(losses):.4f} kappa={float(model.kappa().detach()):.2f} val_warmhit20={v:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")
    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def quick_warm_hit(model, pairs, freq_t, freq, n, k=20):
    hit, tot = 0, 0
    for seeds, gt in pairs:
        sc = model.score_all(seeds, freq_t, n)
        top = set(top_items(sc, list(seeds), k).tolist())
        for t in gt:
            if freq[t] > 2:
                tot += 1; hit += (t in top)
    return hit / max(1, tot)


def hits_by_bucket(model, pairs, freq, freq_t, n, k=20):
    buck = {name: [0, 0] for _, _, name in EDGES}
    for seeds, gt in pairs:
        sc = model.score_all(seeds, freq_t, n)
        top = set(top_items(sc, list(seeds), k).tolist())
        for t in set(gt):
            b = buck[bucket(int(freq[t]))]; b[1] += 1; b[0] += int(t in top)
    return {name: (v[0] / v[1] if v[1] else float("nan")) for name, v in buck.items()}


def run(dataset):
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", dataset)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    provided = ProvidedSplits(dd, n)
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    freq = graph.freq.astype(np.float64); C = graph.C.tocsr()
    freq_t = torch.tensor(freq, dtype=torch.float32)
    rng = np.random.default_rng(0); torch.manual_seed(0)
    bundles = list(provided.train.values())
    log(f"\n### {dataset}: n={n:,} DIM={DIM} JOINT={JOINT} | setup {time.time()-t0:.0f}s")

    model = GatedBundle(n, DIM, INIT_KAPPA)
    pr, pc = build_pairs(bundles, rng)
    pn_pmi = (freq + 1e-9) ** 0.75; pn_pmi = pn_pmi / pn_pmi.sum()
    log(f"  stage 1: pmi pathway = learned SGNS (pairs={len(pr):,}); cooc pathway = {COOC_SRC}...")
    train_sgns(model.Up, model.Vp, pr, pc, pn_pmi, n, rng, "pmi")
    if COOC_SRC == "svd_wpair":
        # head pathway: optimal count factorisation of W_pair (Eckart-Young == gradient WMF fixed point;
        # the raw-cooc dot product that gives head ~0.72). Load into the cooc embeddings, keep frozen.
        from sklearn.decomposition import TruncatedSVD
        sv = TruncatedSVD(n_components=min(DIM, n - 2), random_state=0)
        A = sv.fit_transform(graph.W_pair.tocsr()).astype(np.float32); B = sv.components_.T.astype(np.float32)
        with torch.no_grad():
            model.Uc.weight.copy_(torch.from_numpy(A)); model.Vc.weight.copy_(torch.from_numpy(B))
        model.Uc.weight.requires_grad_(False); model.Vc.weight.requires_grad_(False)
    else:  # sgns_uniform: fully-learned but weaker head pathway (v2)
        train_sgns(model.Uc, model.Vc, pr, pc, np.ones(n) / n, n, rng, "cooc")
    log("  stage 2: learn gate kappa + pathway weights gb on bundle CE...")
    val_pairs = provided.pairs("valid")[:VAL_N]
    train_gate(model, C, bundles, freq, freq_t, val_pairs, n, rng)

    test_pairs = provided.pairs("test")
    res = hits_by_bucket(model, test_pairs, freq, freq_t, n)
    log(f"\n  {dataset}  GatedBundle: per-target Hit@20 by TRAIN frequency ({len(test_pairs)} queries) | kappa={float(model.kappa().detach()):.2f}")
    log(f"    {'config':14s} " + " ".join(f"{b:>11}" for b in ORDER))
    bf = BF.get(dataset)
    if bf:
        log(f"    {'BundleFuse':14s} " + " ".join(f"{bf[b]:11.4f}" for b in ORDER) + "   (reference)")
    log(f"    {'GatedBundle':14s} " + " ".join(f"{res[b]:11.4f}" for b in ORDER))
    log(f"  ({dataset} done in {time.time()-t0:.0f}s)")
    return {"dataset": dataset, "kappa": float(model.kappa().detach()), "buckets": res, "bundlefuse": bf}


def main():
    out = [run(ds) for ds in DATASETS]
    d = os.path.join(PROJECT_ROOT, "results", "sefp"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "gated_bundle.json"); json.dump(out, open(p, "w"), indent=2)
    log(f"\nwrote {p}")


if __name__ == "__main__":
    main()
