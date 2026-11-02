"""BundleFuse -- a single, config-driven model class for bundle completion.

One class, one object per dataset. Which signals are active is decided by the ``families`` in the
:class:`BundleFuseConfig` passed to the constructor; the signals are organised by the conceptual
families (collaborative / content / taxonomy / popularity / personalization). The API mirrors a
scikit-learn / PyTorch model:

    model = BundleFuse(BundleFuseConfig(n_items=n, families=["collaborative", "content", ...]))
    model.fit(train_bundles, val_pairs, content=..., cat_labels=..., user_items=...)
    model.save("bf.pt")                 # checkpoint: all fitted artifacts + weights
    model = BundleFuse.load("bf.pt")    # restore
    items = model.predict(seeds, k=20, user=None)   # fast: one pass over the seed set

``fit`` does ALL the processing once (build the hypergraph operator, content bridge, taxonomy
co-occurrence, user history; select fusion weights + empirical-Bayes shrinkage on validation) and
stores it on the object. ``predict``/``score`` are fast: they only compute the active signals for the
given seed set (a few sparse mat-vecs and dot products), fuse, shrink, and rank. No training happens at
predict time. The heavy math is delegated to the verified operators in :mod:`bundlefuse.operators`.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp

from bundlefuse.operators import (
    SimplicialGraphOperator,
    BilateralContentBridge,
    EmpiricalBayesRouter,
    zcols,
)
from bundlefuse.metrics import ranking_metrics

# ---- the conceptual families -> the signals each provides -------------------------------------
FAMILY_SIGNALS = {
    "collaborative": ["cooc", "ppmi", "simp", "diff2", "csvd"],
    "content": ["content", "bridge"],
    "taxonomy": ["cat"],
    "popularity": ["pop"],
    "personalization": ["user_freq", "user_cooc"],
}
# signals that act as cold-start PRIORS (content/taxonomy are informative for rare items)
PRIOR_SIGNALS = {"content", "bridge", "cat"}
EPS = 1e-8


@dataclass
class BundleFuseConfig:
    """Everything needed to instantiate a model; ``families`` switches signal blocks on/off.

    A family is only truly active if (a) it is listed in ``families`` AND (b) the data it needs is
    supplied to ``fit`` (content array, category labels, or user histories). ``families=None`` means
    "enable every family whose data is provided."
    """
    n_items: int
    families: Optional[list] = None          # subset of FAMILY_SIGNALS keys, or None = auto
    svd_dim: int = 128
    alpha_deg: float = 0.5
    gamma_vol: float = 0.5
    beta_clique: float = 1.5
    alpha_bridge: float = 0.5
    weight_grid: tuple = (0.0, 0.25, 0.5, 1.0, 1.5)
    fit_rounds: int = 4
    shrink_grid: tuple = (0.0, 0.05, 0.25, 1.0)
    seed_cap: int = 0                        # 0 = use all seeds; else truncate to first `seed_cap`


class BundleFuse:
    def __init__(self, config: BundleFuseConfig):
        self.cfg = config
        self.graph: Optional[SimplicialGraphOperator] = None
        self.E = None            # (n, d) normalised content
        self.BR = None           # (n, d) dense content bridge
        self.cat_lbl = None      # (n,) int category labels
        self.Pcat = None         # (NC, NC) category co-occurrence
        self.UI = None           # (n, n_users) user purchase counts (personalization)
        self.user_items = None   # dict user -> list[int] (history item indices)
        self.signals_: list = []
        self.weights_: dict = {}
        self.prior_keys_: list = []
        self.kappa_f_: float = 0.0
        self.prior_scale_: float = 1.0
        self.fitted_ = False

    # ===================== signal families (organised blocks) =====================
    # Each block takes a list of seed-sets and returns {signal_name: (n, nb) raw score matrix}.

    def _collaborative(self, seeds_list):
        g = self.graph
        out = {}
        Ind = g.build_indicator(seeds_list)
        out["cooc"] = np.asarray((g.C @ Ind).todense(), dtype=np.float32)
        out["ppmi"] = np.asarray((g.PPMI @ Ind).todense(), dtype=np.float32)
        sc_pair, sc_diff2 = g.score_pairwise_and_diff2(seeds_list)
        d_inc, d_diff2 = g.score_simplicial_incidence_only(seeds_list, beta_clique=self.cfg.beta_clique)
        out["simp"] = (sc_pair + d_inc).astype(np.float32)
        out["diff2"] = (sc_diff2 + d_diff2).astype(np.float32)
        Us = np.stack([g.Ucsvd[s].mean(0) for s in seeds_list], 1)
        out["csvd"] = (g.Ucsvd @ Us).astype(np.float32)
        return out

    def _content(self, seeds_list):
        out = {}
        if self.E is not None:
            Cs = np.stack([self.E[s].mean(0) for s in seeds_list], 1)
            out["content"] = (self.E @ Cs).astype(np.float32)
        if self.BR is not None:
            Bs = np.stack([self.BR[s].mean(0) for s in seeds_list], 1)
            out["bridge"] = (self.BR @ Bs).astype(np.float32)
        return out

    def _taxonomy(self, seeds_list):
        if self.cat_lbl is None:
            return {}
        nb = len(seeds_list)
        sc = np.zeros((self.cfg.n_items, nb), np.float32)
        for j, s in enumerate(seeds_list):
            sc[:, j] = np.asarray(self.Pcat[[self.cat_lbl[i] for i in s]].sum(0)).ravel()[self.cat_lbl]
        return {"cat": sc}

    def _popularity(self, seeds_list):
        pop = np.log1p(self.graph.freq)[:, None]
        return {"pop": np.tile(pop, (1, len(seeds_list))).astype(np.float32)}

    def _personalization(self, seeds_list, users):
        if self.UI is None or users is None:
            return {}
        n = self.cfg.n_items; nb = len(seeds_list)
        uf = np.zeros((n, nb), np.float32)
        Hind = sp.lil_matrix((n, nb))
        for j, u in enumerate(users):
            if u is not None and u in self.user_items:
                hist = self.user_items[u]
                uf[:, j] = np.asarray(self.UI[:, u].todense()).ravel()
                if hist:
                    Hind[hist, j] = 1.0
        uc = np.asarray((self.graph.C @ Hind.tocsr()).todense(), dtype=np.float32)
        return {"user_freq": uf, "user_cooc": uc}

    def _signals(self, seeds_list, users=None):
        """Compute every ACTIVE signal, z-scored per column (candidate axis). Returns {name:(n,nb)}."""
        if self.cfg.seed_cap:
            seeds_list = [list(s)[: self.cfg.seed_cap] for s in seeds_list]
        raw = {}
        active_families = self._active_families()
        if "collaborative" in active_families:
            raw.update(self._collaborative(seeds_list))
        if "content" in active_families:
            raw.update(self._content(seeds_list))
        if "taxonomy" in active_families:
            raw.update(self._taxonomy(seeds_list))
        if "popularity" in active_families:
            raw.update(self._popularity(seeds_list))
        if "personalization" in active_families:
            raw.update(self._personalization(seeds_list, users))
        return {k: zcols(raw[k]) for k in raw if k in self.signals_}

    # ===================== fit =====================

    def _active_families(self):
        fams = self.cfg.families if self.cfg.families is not None else list(FAMILY_SIGNALS)
        avail = {"collaborative", "popularity"}
        if self.E is not None or self.BR is not None:
            avail.add("content")
        if self.cat_lbl is not None:
            avail.add("taxonomy")
        if self.UI is not None:
            avail.add("personalization")
        return [f for f in fams if f in avail]

    @staticmethod
    def _build_pcat(train_bundles, cat_lbl, n_items):
        nc = int(cat_lbl.max()) + 1
        rows, cols = [], []
        b_iter = train_bundles.values() if isinstance(train_bundles, dict) else train_bundles
        for bi, items in enumerate(b_iter):
            for it in dict.fromkeys(items):
                rows.append(bi); cols.append(int(cat_lbl[it]))
        Bm = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(max(bi + 1, 1), nc))
        P = (Bm.T @ Bm).toarray().astype(np.float32)
        np.fill_diagonal(P, 0.0)
        return P

    def _build_user(self, user_items):
        from collections import Counter
        self.user_items = {u: list(its) for u, its in user_items.items()}
        nusers = (max(user_items) + 1) if user_items else 1
        rows, cols, data = [], [], []
        for u, its in user_items.items():
            for it, c in Counter(its).items():
                rows.append(it); cols.append(u); data.append(float(c))
        self.UI = sp.csr_matrix((data, (rows, cols)), shape=(self.cfg.n_items, nusers))

    def fit(self, train_bundles, val_pairs, *, content=None, cat_labels=None,
            user_items=None, val_users=None):
        """Build all artifacts and select fusion weights + shrinkage on the validation pairs.

        train_bundles : list of item-index lists (the training bundles).
        val_pairs     : list of (seeds, ground_truth) for weight selection.
        content       : optional (n, d) float array   -> enables the content family.
        cat_labels    : optional (n,) int array        -> enables the taxonomy family.
        user_items    : optional dict user->list[int]  -> enables personalization.
        val_users     : optional list of user ids aligned to val_pairs (personalization fit).
        """
        cfg = self.cfg
        self.graph = SimplicialGraphOperator(
            train_bundles, n_items=cfg.n_items, svd_dim=cfg.svd_dim,
            alpha_deg=cfg.alpha_deg, gamma_vol=cfg.gamma_vol, beta_clique=cfg.beta_clique,
        )
        if content is not None:
            E = np.asarray(content, np.float32)
            self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + EPS)
            self.BR = BilateralContentBridge.build_dense_bridge(self.E, self.graph, alpha_bridge=cfg.alpha_bridge)
        if cat_labels is not None:
            self.cat_lbl = np.asarray(cat_labels, int)
            self.Pcat = self._build_pcat(train_bundles, self.cat_lbl, cfg.n_items)
        if user_items is not None:
            self._build_user(user_items)

        # active signal menu = signals of active families whose data exists
        self.signals_ = [s for f in self._active_families() for s in FAMILY_SIGNALS[f]]
        # content/personalization signals may be missing if a tensor wasn't built
        probe = self._signals([val_pairs[0][0]], [val_users[0] if val_users else None])
        self.signals_ = [s for s in self.signals_ if s in probe]

        val_seeds = [s for s, _ in val_pairs]
        Zv = self._signals(val_seeds, val_users)
        self._fit_fusion(Zv, val_pairs)
        self.fitted_ = True
        return self

    def _ndcg(self, F, pairs):
        return ranking_metrics(F, pairs, (20,))[20][1]

    def _fit_fusion(self, Zv, val_pairs):
        sigs = self.signals_
        w = {k: 0.0 for k in sigs}; best = -1.0
        for _ in range(self.cfg.fit_rounds):
            improved = False
            for k in sigs:
                for g in self.cfg.weight_grid:
                    w2 = dict(w); w2[k] = g
                    if sum(w2.values()) == 0:
                        continue
                    F = sum(w2[s] * Zv[s] for s in w2 if w2[s])
                    nd = self._ndcg(F, val_pairs)
                    if nd > best + 2e-4:
                        best, w, improved = nd, w2, True
            if not improved:
                break
        self.weights_ = {k: v for k, v in w.items() if v}
        self.prior_keys_ = [k for k in self.weights_ if k in PRIOR_SIGNALS] or \
                           [k for k in sigs if k in PRIOR_SIGNALS][:1]

        # empirical-Bayes shrinkage (conjugate normal-normal), frozen on validation
        freq = self.graph.freq
        Yv = sum(self.weights_[k] * Zv[k] for k in self.weights_)
        if self.prior_keys_:
            raw_pr = sum(self.weights_.get(k, 0.5) * Zv[k] for k in self.prior_keys_)
            self.prior_scale_ = float(np.std(Yv) / (np.std(raw_pr) + EPS))
            mu0 = raw_pr * self.prior_scale_
            vc = EmpiricalBayesRouter.fit_variance_components(freq, Yv, mu0)
            best_kf, best_nd = 0.0, best
            cands = EmpiricalBayesRouter.shrinkage_candidates(vc, list(self.cfg.shrink_grid) + [vc["kappa_f"]])
            for kf in cands:
                theta = EmpiricalBayesRouter.conjugate_posterior(Yv, mu0, freq, kf)
                nd = self._ndcg(theta, val_pairs)
                if nd > best_nd + 1e-4:
                    best_nd, best_kf = nd, kf
            self.kappa_f_ = best_kf

    # ===================== predict / score =====================

    def score(self, seeds, user=None):
        """Full-catalogue score vector (n,) for one seed set."""
        if not self.fitted_:
            raise RuntimeError("call fit() or load() first")
        Z = self._signals([list(seeds)], [user])
        Y = sum(self.weights_[k] * Z[k][:, 0] for k in self.weights_ if k in Z)
        if self.kappa_f_ > 0 and self.prior_keys_:
            raw_pr = sum(self.weights_.get(k, 0.5) * Z[k][:, 0] for k in self.prior_keys_ if k in Z)
            mu0 = raw_pr * self.prior_scale_
            Y = EmpiricalBayesRouter.conjugate_posterior(Y, mu0, self.graph.freq, self.kappa_f_)
        return np.asarray(Y, np.float32)

    def predict(self, seeds, k=20, user=None):
        """Top-k completion item indices for a seed set (seeds excluded, decreasing score)."""
        sc = self.score(seeds, user).copy()
        sc[list(seeds)] = -np.inf
        k = min(k, (sc > -np.inf).sum())
        top = np.argpartition(-sc, k - 1)[:k]
        return top[np.argsort(-sc[top])].tolist()

    # ===================== checkpoint =====================

    _STATE_KEYS = ("cfg", "graph", "E", "BR", "cat_lbl", "Pcat", "UI", "user_items",
                   "signals_", "weights_", "prior_keys_", "kappa_f_", "prior_scale_", "fitted_")

    def save(self, path):
        """Serialise all fitted artifacts + weights (a self-contained checkpoint)."""
        state = {k: getattr(self, k) for k in self._STATE_KEYS}
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(state["cfg"])
        for k in cls._STATE_KEYS:
            setattr(obj, k, state[k])
        return obj
