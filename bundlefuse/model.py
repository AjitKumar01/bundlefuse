"""BundleFuse -- the config-driven model class, implementing the SAME pipeline as the research
evaluators (bundlefuse/eval/fullrank.py etc. on the compressed-inference branch).

The scoring (``_score_all`` == the original ``score_all_signals``) and the fit/apply engine
(``fusion_selection.fit_fusion`` / ``apply_fusion``) are the verified originals -- this class only
assembles the artifacts the original built at module level (graph operator, DualSpectrumOntology,
disentangled content styles + bridge, Rao concordance, SIG) inside ``fit`` and reuses them, so the
numbers reproduce the research matrix.

Signals are grouped into families; ``config.families`` keeps a subset of the full research SIG
(default: all). ``fit`` does all processing once; ``predict``/``score`` run one scoring pass over a
seed set + the frozen fusion; ``save``/``load`` is a compressed checkpoint.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from bundlefuse.operators import (
    SimplicialGraphOperator, DualSpectrumOntology, BilateralContentBridge,
    EmpiricalBayesRouter, zcols,
)
from bundlefuse.fusion_selection import fit_fusion, apply_fusion, ValidationSignals
from bundlefuse.metrics import ranking_metrics

DEVICE = torch.device("cpu")
EPS = 1e-8

# full research signal menu; the active SIG = (music or non-music list) ∩ families ∩ available
SIG_MUSIC = ["simp", "diff2", "pop", "cooc", "ppmi", "csvd", "cat", "homo", "audiostyle", "bridge"]
SIG_FASHION = ["cat", "softcat", "style", "imgstyle", "bridge", "simp", "diff2", "pop", "cooc", "ppmi", "csvd"]
SIG_PERS = ["user_freq", "user_cooc"]

# which family each signal belongs to (for config.families filtering). Auxiliary co-occurrence signals
# (hh_cooc, pretrain, ...) are registered by the loader-chosen name at fit time -> collaborative.
FAMILY_OF = {}
for s in ("cooc", "ppmi", "simp", "diff2", "csvd"):
    FAMILY_OF[s] = "collaborative"
for s in ("content", "style", "imgstyle", "audiostyle", "descstyle", "bridge"):
    FAMILY_OF[s] = "content"
for s in ("cat", "softcat", "homo"):
    FAMILY_OF[s] = "taxonomy"
FAMILY_OF["pop"] = "popularity"
for s in SIG_PERS:
    FAMILY_OF[s] = "personalization"

AUX_KEYS = {"pair", "diff2_pair", "rao"}            # fed to the higher-order mix, never selected directly

# content-feature keys that get low-ranked for a compressed checkpoint (csvd stays -- it is collaborative)
_COMPRESS_KEYS = {"content", "style", "imgstyle", "audiostyle", "descstyle", "bridge"}
# derived-style source for each content modality the loader provides
_STYLE_OF = {"content": "style", "title": "style", "image": "imgstyle", "desc": "descstyle", "audio": "audiostyle"}


@dataclass
class BundleFuseConfig:
    n_items: int
    families: Optional[list] = None          # subset of family names, or None = all
    svd_dim: int = 128
    alpha_deg: float = 0.5
    gamma_vol: float = 0.5
    beta_clique: float = 1.5
    alpha_bridge: float = 0.5
    n_soft_clusters: int = 400
    soft_tau: float = 0.05
    sel_obj: str = "ndcg20"
    compressed: bool = True
    content_rank: int = 64


class BundleFuse:
    def __init__(self, config: BundleFuseConfig):
        self.cfg = config
        self.fitted_ = False

    # ===================== fit =====================
    def fit(self, train_bundles, val_pairs, *, features=None, cat_labels=None, music=False,
            alb_labels=None, cluster_feat=None, aux_incidences=None, group_labels=None,
            user_items=None, val_users=None):
        """Build the signal-pool artifacts, then select the fusion on validation. Each dataset supplies
        whatever inputs it has; the model computes every applicable signal and the fit selects them.

        features        : content modalities, e.g. {"content":title,"image":img,"desc":desc} / {"audio":a}
                          / {"content": tfidf}. -> raw `content` + disentangled `style`/`imgstyle`/... + `bridge`.
        cat_labels / cluster_feat : item categories, or a feature array to cluster -> `cat`, `softcat`.
        music / alb_labels : music path (`audiostyle`, artist/album homophily).
        aux_incidences  : {signal_name: (incidence_csr[items x contexts], weighted_bool)} -> a co-occurrence
                          signal per auxiliary incidence (e.g. "hh_cooc" household, "pretrain" user-item).
        group_labels    : {name: (labels, weight)} -> a `homo` homophily signal (e.g. session/user).
        user_items / val_users : personalization signals (`user_freq`, `user_cooc`).
        """
        cfg = self.cfg
        n = cfg.n_items
        self.music = bool(music)
        g = SimplicialGraphOperator(train_bundles, n_items=n, svd_dim=cfg.svd_dim,
                                    alpha_deg=cfg.alpha_deg, gamma_vol=cfg.gamma_vol, beta_clique=cfg.beta_clique)
        self.graph = g
        self.freq = g.freq
        self.freq_c = (np.asarray(g.C.sum(1)).ravel() + 1e-9).astype(np.float32)
        self.totc = float(g.C.sum())
        n_soft = max(2, min(cfg.n_soft_clusters, n - 1))   # clamp for tiny catalogues

        mods = {k: self._norm(np.asarray(v, np.float32)) for k, v in (features or {}).items()}

        # category labels (explicit) or k-means on the content/cluster features
        if cat_labels is not None:
            cat_lbl = np.asarray(cat_labels, int)
        else:
            cf = cluster_feat if cluster_feat is not None else (
                np.concatenate(list(mods.values()), 1) if mods else g.Ucsvd)
            cat_lbl = MiniBatchKMeans(n_clusters=n_soft, random_state=0, n_init=3,
                                      batch_size=4096).fit(np.asarray(cf, np.float32)).labels_.astype(int)
        self.cat_lbl = cat_lbl
        NC = int(cat_lbl.max()) + 1

        self.onto = DualSpectrumOntology(
            train_bundles, cat_lbl=cat_lbl, n_items=n, alb_lbl=alb_labels,
            cluster_feat=(np.concatenate(list(mods.values()), 1).astype(np.float32)
                          if (mods and not music) else None),
            n_soft_clusters=n_soft, soft_tau=cfg.soft_tau, device=DEVICE)

        # content pool: raw `content` (concat of modalities) + disentangled per-modality style + bridge.
        # disentangle labels = explicit category (fashion), a content clustering for music (artist cat
        # is too fine).
        feats = {}
        if mods:
            feats["content"] = self._norm(np.concatenate(list(mods.values()), 1).astype(np.float32))
            if music:
                style_lbl = MiniBatchKMeans(n_clusters=n_soft, random_state=0, n_init=3,
                                            batch_size=4096).fit(feats["content"]).labels_.astype(int)
                style_nc = int(style_lbl.max()) + 1
            else:
                style_lbl, style_nc = cat_lbl, NC
            styles = {_STYLE_OF.get(mod, "style"): BilateralContentBridge.disentangle(arr, style_lbl, style_nc)
                      for mod, arr in mods.items()}
            st_all = self._norm(np.concatenate(list(styles.values()), 1).astype(np.float32))
            feats.update(styles)
            feats["bridge"] = BilateralContentBridge.build_dense_bridge(st_all, g, alpha_bridge=cfg.alpha_bridge)
            self.rao_dense = styles.get("audiostyle" if music else "style")
        else:
            self.rao_dense = None
        feats["csvd"] = g.Ucsvd

        # auxiliary co-occurrence incidences (household, user-item pretrain, ...) + homophily groupings
        self.aux = {name: (I.tocsr(), bool(w)) for name, (I, w) in (aux_incidences or {}).items()}
        for name in self.aux:
            FAMILY_OF.setdefault(name, "collaborative")
        self.group_inc = []
        for _, (lbl, weight) in (group_labels or {}).items():
            lbl = np.asarray(lbl, int); ng = int(lbl.max()) + 1
            I = sp.csr_matrix((np.ones(n, np.float32), (np.arange(n), lbl)), shape=(n, ng))
            deg = np.asarray(I.sum(0)).ravel() + 1.0
            self.group_inc.append(((I @ sp.diags(1.0 / np.sqrt(deg))).tocsr(), float(weight)))

        if cfg.compressed:
            r = cfg.content_rank
            feats = {k: (TruncatedSVD(r, random_state=0).fit_transform(v).astype(np.float32)
                         if (k in _COMPRESS_KEYS and np.asarray(v).shape[1] > r) else np.asarray(v, np.float32))
                     for k, v in feats.items()}
        self.FEAT = {k: torch.tensor(v, device=DEVICE) for k, v in feats.items()}

        if user_items is not None:
            self._build_user(user_items)
        else:
            self.UI = None; self.user_items = None

        # compute every applicable signal on validation, then the active SIG = selectable signals
        # (co-occurrence, content, taxonomy, popularity, personalization) kept by config.families
        val_seeds = [s for s, _ in val_pairs]
        _, Zv = self._score_all(val_seeds, val_users)
        fams = cfg.families
        self.SIG = [k for k in Zv if k not in AUX_KEYS and (fams is None or FAMILY_OF.get(k) in fams)]

        # fit the fusion on validation (the verified engine)
        validation = ValidationSignals(lambda: iter([(0, Zv)]), list(val_pairs), self.freq, cfg.sel_obj)
        prior_cand = tuple(k for k in ("cat", "softcat", "homo", "content", "style", "imgstyle",
                                       "audiostyle", "bridge") if k in self.SIG)
        rao_grid = (0.0, 0.2, 0.35) if "rao" in Zv else (0.0,)
        self.w, self.prior_keys, self.best_eb, self.best_ho, _ = fit_fusion(
            validation, self.SIG, prior_cand, rao=rao_grid)
        self.freq_t = torch.tensor(self.freq, device=DEVICE)[:, None]
        self.fitted_ = True
        return self

    def _active_sig(self):
        base = SIG_MUSIC if self.music else SIG_FASHION
        if self.UI is not None:
            base = base + SIG_PERS
        fams = self.cfg.families
        out = []
        for s in base:
            if fams is not None and FAMILY_OF.get(s) not in fams:
                continue
            if s in ("style", "imgstyle", "audiostyle", "descstyle", "bridge") and s not in self.FEAT:
                continue
            if s in SIG_PERS and self.UI is None:
                continue
            out.append(s)
        return out

    @staticmethod
    def _norm(x):
        return (x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)).astype(np.float32)

    def _build_user(self, user_items):
        from collections import Counter
        self.user_items = {u: list(its) for u, its in user_items.items()}
        nu = (max(user_items) + 1) if user_items else 1
        r, c, d = [], [], []
        for u, its in user_items.items():
            for it, cnt in Counter(its).items():
                r.append(it); c.append(u); d.append(float(cnt))
        self.UI = sp.csr_matrix((d, (r, c)), shape=(self.cfg.n_items, nu))

    # ===================== scoring (verbatim score_all_signals) =====================
    def _score_all(self, bundles, users=None):
        g = self.graph; onto = self.onto; n = self.cfg.n_items; nb = len(bundles)
        Ind = g.build_indicator(bundles)
        sc_pair, sc_diff2_pair = g.score_pairwise_and_diff2(bundles)
        delta_inc, diff2_delta = g.score_simplicial_incidence_only(bundles, beta_clique=self.cfg.beta_clique)
        R = {
            "simp": torch.tensor(sc_pair + delta_inc, device=DEVICE),
            "diff2": torch.tensor(sc_diff2_pair + diff2_delta, device=DEVICE),
            "pair": torch.tensor(sc_pair, device=DEVICE),
            "diff2_pair": torch.tensor(sc_diff2_pair, device=DEVICE),
            "pop": torch.tensor(np.tile(np.log1p(self.freq)[:, None], (1, nb)), dtype=torch.float32, device=DEVICE),
        }
        if self.rao_dense is not None and getattr(g, "W_pair", None) is not None:
            sc_rao = g.score_rao_concordance(bundles, dense_feat=self.rao_dense, cat_lbl=self.cat_lbl,
                                             gamma=(1.0 if self.music else -1.0), alpha_dense=0.15, tau=0.05)
            R["rao"] = torch.tensor(sc_rao, device=DEVICE)
        # cooc / ppmi -- from materialised C/PPMI, or from the incidence B when compressed-stripped
        if getattr(g, "PPMI", None) is not None:
            R["cooc"] = torch.tensor(np.asarray((g.C @ Ind).todense()), dtype=torch.float32, device=DEVICE)
            R["ppmi"] = torch.tensor(np.asarray((g.PPMI @ Ind).todense()), dtype=torch.float32, device=DEVICE)
        else:
            B = g.B
            R["cooc"] = torch.tensor(np.asarray((B.T @ (B @ Ind)).todense()).astype(np.float32), device=DEVICE)
            ppmi = np.zeros((n, nb), np.float32)
            for j, s in enumerate(bundles):
                if not len(s):
                    continue
                Cs = np.asarray((B.T @ B[:, list(s)]).todense())
                with np.errstate(divide="ignore", invalid="ignore"):
                    pp = np.log(Cs * self.totc / (self.freq_c[:, None] * self.freq_c[list(s)][None, :] + 1e-12))
                pp = np.maximum(pp, 0.0); pp[Cs <= 0] = 0.0
                ppmi[:, j] = pp.sum(1)
            R["ppmi"] = torch.tensor(ppmi, device=DEVICE)
        # auxiliary co-occurrence (household / user-item pretrain / ...): I^T (I @ indicator)
        if self.aux:
            Ind_w = Ind.multiply(g.inv_sqrt_f[:, None]).tocsr()
            for name, (I, weighted) in self.aux.items():
                x = Ind_w if weighted else Ind
                R[name] = torch.tensor(np.asarray((I.T @ (I @ x)).todense()).astype(np.float32), device=DEVICE)
        # taxonomy
        out_cat = np.zeros((n, nb), np.float32)
        for j, s in enumerate(bundles):
            sa = [self.cat_lbl[i] for i in s]
            out_cat[:, j] = np.asarray(onto.Pcat[sa].sum(0)).ravel()[self.cat_lbl]
        if not self.music:
            out_cat = out_cat + 0.5 * onto.score_slot_saturation(bundles)
        R["cat"] = torch.tensor(out_cat, device=DEVICE)
        # homophily: artist/album for music (via the ontology), or supplied group labels (session/user)
        if self.music and getattr(onto, "I_art_w", None) is not None:
            art = np.asarray((onto.I_art_w @ (onto.I_art_w.T @ Ind)).todense())
            alb = np.asarray((onto.I_alb_w @ (onto.I_alb_w.T @ Ind)).todense()) if onto.I_alb_w is not None else 0.0
            R["homo"] = torch.tensor((art + 2.0 * alb).astype(np.float32), device=DEVICE)
        elif self.group_inc:
            homo = sum(w * np.asarray((I @ (I.T @ Ind)).todense()) for I, w in self.group_inc)
            R["homo"] = torch.tensor(homo.astype(np.float32), device=DEVICE)
        if (not self.music) and onto.MP_t is not None:
            Ms = torch.stack([onto.Msoft_t[s].sum(0) for s in bundles], 1)
            R["softcat"] = onto.MP_t @ Ms
        # content embeddings (dot with the mean seed vector) -- raw content + disentangled styles + bridge
        for k in list(self.FEAT):
            f = self.FEAT[k]
            R[k] = f @ torch.stack([f[s].mean(0) for s in bundles], 1)
        # personalization
        if self.UI is not None and users is not None:
            uf = np.zeros((n, nb), np.float32); Hind = sp.lil_matrix((n, nb))
            for j, u in enumerate(users):
                if u is not None and u in self.user_items:
                    uf[:, j] = np.asarray(self.UI[:, u].todense()).ravel()
                    h = self.user_items[u]
                    if h:
                        Hind[h, j] = 1.0
            R["user_freq"] = torch.tensor(uf, device=DEVICE)
            uc = (g.C @ Hind.tocsr()) if getattr(g, "C", None) is not None else (g.B.T @ (g.B @ Hind.tocsr()))
            R["user_cooc"] = torch.tensor(np.asarray(uc.todense()).astype(np.float32), device=DEVICE)
        Z = {k: zcols(R[k]) for k in R}      # z-score every computed signal (SIG is derived from these)
        return R, Z

    # ===================== predict / score =====================
    def score(self, seeds, user=None):
        if not self.fitted_:
            raise RuntimeError("call fit() or load() first")
        _, Z = self._score_all([list(seeds)], [user])
        F = apply_fusion(Z, self.w, self.prior_keys, self.best_eb, self.freq_t, self.best_ho)
        return np.asarray(F[:, 0].cpu().numpy() if hasattr(F, "cpu") else F[:, 0], np.float32)

    def predict(self, seeds, k=20, user=None):
        sc = self.score(seeds, user).copy()
        sc[list(seeds)] = -np.inf
        k = min(k, int((sc > -np.inf).sum()))
        top = np.argpartition(-sc, k - 1)[:k]
        return top[np.argsort(-sc[top])].tolist()

    # ===================== checkpoint =====================
    _DROP_WHEN_COMPRESSED = ("C", "C_bin", "PPMI", "W_pair_sq", "W_csc", "P_rw")  # W_pair kept (needed by rao)

    def save(self, path):
        if self.cfg.compressed and getattr(self, "graph", None) is not None:
            for attr in self._DROP_WHEN_COMPRESSED:
                if getattr(self.graph, attr, None) is not None:
                    setattr(self.graph, attr, None)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        return load_model(path)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
