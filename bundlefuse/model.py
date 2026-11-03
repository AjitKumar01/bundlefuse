"""Config-driven closed-form bundle scorer with validation-fitted fusion and exact checkpoints.

The full-signal method is the reference for selected-signal serving. Dataset protocol, signal
menu and content compression are explicit settings; historical benchmark parity is not implied
merely by sharing the mathematical operators with an earlier research evaluator.
"""
from __future__ import annotations

import math
import pickle
from copy import copy, deepcopy
import os
import tempfile
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
from bundlefuse.fusion_selection import fit_fusion, apply_fusion, ValidationSignals, required_fusion_signals
from bundlefuse.metrics import top_items
from bundlefuse.graph_state import ensure_graph_matrices
from bundlefuse.selected_signals import SelectedSignals
from bundlefuse.validation_cache import ValidationCache

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
    fit_rounds: int = 4
    seed_cap: int = 0
    validation_chunk: int = 32

    def __post_init__(self):
        if min(self.n_items, self.svd_dim, self.content_rank, self.fit_rounds, self.validation_chunk) < 1:
            raise ValueError('Catalogue size, dimensions, fit rounds and chunk size must be positive')
        if self.n_items < 3 or self.seed_cap < 0:
            raise ValueError('Need at least three items and a nonnegative seed cap')
        known = {'collaborative', 'content', 'taxonomy', 'popularity', 'personalization'}
        if self.families is not None and (not self.families or set(self.families) - known):
            raise ValueError('Unknown or empty signal families')


class BundleFuse:
    def __init__(self, config: BundleFuseConfig):
        self.cfg = config
        self.fitted_ = False

    # ===================== fit =====================
    def fit(self, train_bundles, val_pairs, *, features=None, cat_labels=None, music=False,
            alb_labels=None, cluster_feat=None, aux_incidences=None, group_labels=None,
            user_items=None, val_users=None, data_spec=None):
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
        cfg.__post_init__()
        self.fitted_ = False
        if not val_pairs:
            raise ValueError('Nonempty held-out validation queries are required')
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
        # category-homophily matrix (same-category affinity, diagonal kept) -> the non-music `homo` signal
        self.Pcat_homo = None
        if (not music) and not self.group_inc:
            b_iter = train_bundles.values() if isinstance(train_bundles, dict) else train_bundles
            rr, cc, bi = [], [], 0
            for bi, items in enumerate(b_iter):
                for it in dict.fromkeys(items):
                    rr.append(bi); cc.append(int(cat_lbl[it]))
            T = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(max(bi + 1, 1), NC))
            self.Pcat_homo = (T.T @ T).tocsr()

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
        with ValidationCache(val_pairs, val_users, self._score_all, n, cfg.validation_chunk) as cache:
            fams = cfg.families
            self.SIG = [k for k in cache.keys if k not in AUX_KEYS and (fams is None or FAMILY_OF.get(k) in fams)]
            validation = ValidationSignals(cache.blocks, list(val_pairs), self.freq, cfg.sel_obj)
            prior_cand = tuple(k for k in ('cat', 'softcat', 'homo', 'content', 'style', 'imgstyle',
                                           'audiostyle', 'bridge') if k in self.SIG)
            rao_grid = (0., .2, .35) if 'rao' in cache.keys else (0.,)
            self.w, self.prior_keys, self.best_eb, self.best_ho, self.variance_components_ = fit_fusion(
                validation, self.SIG, prior_cand, rao=rao_grid, rounds=cfg.fit_rounds)
        self.freq_t = torch.tensor(self.freq, device=DEVICE)[:, None]
        self.data_spec = deepcopy(data_spec)
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
        bundles = self._scoring_seeds(bundles)
        g = self.graph; onto = self.onto; n = self.cfg.n_items; nb = len(bundles)
        ensure_graph_matrices(g, {'cooc', 'ppmi', 'user_cooc'})
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
        R["cooc"] = torch.tensor(np.asarray((g.C @ Ind).todense()), dtype=torch.float32, device=DEVICE)
        R["ppmi"] = torch.tensor(np.asarray((g.PPMI @ Ind).todense()), dtype=torch.float32, device=DEVICE)
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
        elif self.Pcat_homo is not None:
            hh = np.zeros((n, nb), np.float32)
            for j, s in enumerate(bundles):
                hh[:, j] = np.asarray(self.Pcat_homo[[self.cat_lbl[i] for i in s]].sum(0)).ravel()[self.cat_lbl]
            R["homo"] = torch.tensor(hh, device=DEVICE)
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
            uc = g.C @ Hind.tocsr()
            R["user_cooc"] = torch.tensor(np.asarray(uc.todense()).astype(np.float32), device=DEVICE)
        Z = {k: zcols(R[k]) for k in R}      # z-score every computed signal (SIG is derived from these)
        return R, Z

    # ===================== predict / score =====================
    def _scoring_seeds(self, bundles):
        prepared = []
        for seeds in bundles:
            seeds = list(seeds)
            if not seeds or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= self.cfg.n_items for i in seeds):
                raise ValueError('Each query needs valid catalogue item IDs')
            if len(set(seeds)) != len(seeds):
                raise ValueError('Repeated seed IDs are not supported')
            cap = getattr(self.cfg, 'seed_cap', 0)
            prepared.append(sorted(seeds)[:cap] if cap else seeds)
        return prepared

    def score_batch(self, bundles, users=None, *, reference=False):
        if not self.fitted_:
            raise RuntimeError("call fit() or load() first")
        if not bundles or (users is not None and len(users) != len(bundles)):
            raise ValueError('Nonempty queries and aligned users are required')
        users = [None] * len(bundles) if users is None else users
        if reference:
            _, Z = self._score_all(bundles, users)
        else:
            required = required_fusion_signals(self.w, self.prior_keys, self.best_eb, self.best_ho)
            scorer = SelectedSignals(self.graph, self.onto, self.FEAT, self.cat_lbl, self.rao_dense,
                                     self.music, DEVICE, beta_clique=self.cfg.beta_clique,
                                     aux=self.aux, groups=self.group_inc, category_homo=self.Pcat_homo,
                                     user_incidence=self.UI, user_items=self.user_items)
            _, Z = scorer.score(self._scoring_seeds(bundles), required, users)
        F = apply_fusion(Z, self.w, self.prior_keys, self.best_eb, self.freq_t, self.best_ho)
        return F.detach().cpu().numpy().astype(np.float32, copy=False)

    def score(self, seeds, user=None):
        return self.score_batch([list(seeds)], [user])[:, 0]

    def predict(self, seeds, k=20, user=None):
        seeds = list(seeds)
        return top_items(self.score(seeds, user), seeds, k).tolist()

    # ===================== checkpoint =====================
    _DROP_WHEN_COMPRESSED = ("C", "C_bin", "PPMI", "W_pair_sq", "W_csc", "P_rw")  # W_pair kept (needed by rao)

    def save(self, path):
        if not self.fitted_:
            raise RuntimeError('Fit a model before saving it')
        snapshot = copy(self)
        snapshot.graph = copy(self.graph)
        if self.cfg.compressed:
            for attr in self._DROP_WHEN_COMPRESSED:
                setattr(snapshot.graph, attr, None)
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix='.bundlefuse_', dir=os.path.dirname(path) or '.')
        try:
            with os.fdopen(handle, 'wb') as stream:
                pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        return load_model(path)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
