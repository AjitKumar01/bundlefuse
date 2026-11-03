"""Baseline completion models, sharing BundleFuse's interface (fit / score / predict / save / load).

Every baseline is a lightweight, zero-gradient model that plugs into the same evaluator.py and
inference.py as BundleFuse: construct with ``n_items``, ``fit`` on train bundles (and optional content),
``score(seeds)`` returns a full-catalogue score vector, ``predict(seeds, k)`` the top-k items.

Models (all derived from the bundle incidence B, so they store only B + small vectors):
  popularity    rank by item log-frequency (ignores the seeds)
  cooccurrence  C @ 1_S  = Bᵀ(B·1_S)
  itemknn       cosine item-item neighbourhood:  sim(i,j)=C_ij/sqrt(f_i f_j)
  p3alpha       3-step random walk on the row-stochastic item graph
  rp3beta       p3alpha re-weighted by popularity^(-beta)
  contentknn    content cosine:  E @ mean(E[S])   (needs a content array)
"""
from __future__ import annotations

import pickle
from copy import deepcopy

import numpy as np

from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.metrics import top_items

EPS = 1e-8


class Baseline:
    """Common interface. Subclasses implement fit() and score()."""
    name = "baseline"

    def __init__(self, n_items, **params):
        self.n = n_items
        for k, v in params.items():
            setattr(self, k, v)

    # fit takes the same keyword arguments as BundleFuse.fit; each baseline uses only what it needs.
    def fit(self, train_bundles, val_pairs=None, *, content=None, cat_labels=None,
            user_items=None, val_users=None):
        raise NotImplementedError

    def score(self, seeds, user=None) -> np.ndarray:
        raise NotImplementedError

    def predict(self, seeds, k=20, user=None):
        seeds = list(seeds)
        return top_items(self.score(seeds, user), seeds, k).tolist()

    def score_batch(self, bundles, users=None):
        users = [None] * len(bundles) if users is None else users
        if not bundles or len(users) != len(bundles):
            raise ValueError('Nonempty queries and aligned users are required')
        return np.stack([self.score(s, u) for s, u in zip(bundles, users)], 1)

    def _ind(self, seeds):
        v = np.zeros(self.n, np.float32)
        v[list(seeds)] = 1.0
        return v

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


class Popularity(Baseline):
    name = "popularity"

    def fit(self, train_bundles, val_pairs=None, **kw):
        f = np.zeros(self.n, np.float64)
        for b in train_bundles:
            for i in set(b):
                f[i] += 1
        self.logf = np.log1p(f).astype(np.float32)
        self.data_spec = deepcopy(kw.get('data_spec'))
        return self

    def score(self, seeds, user=None):
        return self.logf.copy()


class _GraphBaseline(Baseline):
    """Baselines that read the bundle incidence B (co-occurrence family)."""

    def fit(self, train_bundles, val_pairs=None, **kw):
        g = SimplicialGraphOperator(train_bundles, n_items=self.n, svd_dim=2)
        self.B = g.B
        self.freq = g.freq.astype(np.float32)
        self.freq_c = (np.asarray(g.C.sum(1)).ravel() + 1e-9).astype(np.float32)
        self.data_spec = deepcopy(kw.get('data_spec'))
        return self

    def _cooc(self, w):
        """C @ w = Bᵀ(B·w), from the incidence only."""
        return np.asarray(self.B.T @ (self.B @ w)).ravel().astype(np.float32)


class CoOccurrence(_GraphBaseline):
    name = "cooccurrence"

    def score(self, seeds, user=None):
        return self._cooc(self._ind(seeds))


class ItemKNN(_GraphBaseline):
    name = "itemknn"

    def score(self, seeds, user=None):
        invsq = 1.0 / np.sqrt(self.freq + 1.0)
        return (invsq * self._cooc(self._ind(seeds) * invsq)).astype(np.float32)


class P3alpha(_GraphBaseline):
    name = "p3alpha"
    steps = 3

    def score(self, seeds, user=None):
        w = self._ind(seeds)
        for _ in range(self.steps):                 # row-stochastic step P@w = (C@w)/deg
            w = self._cooc(w) / self.freq_c
        return w.astype(np.float32)


class RP3beta(P3alpha):
    name = "rp3beta"
    beta = 0.5

    def score(self, seeds, user=None):
        w = super().score(seeds)
        return (w / np.power(self.freq + 1.0, self.beta)).astype(np.float32)


class ContentKNN(Baseline):
    name = "contentknn"

    def fit(self, train_bundles, val_pairs=None, *, content=None, features=None, **kw):
        if features:
            if content is not None:
                raise ValueError('Supply features or content, not both')
            parts = [np.asarray(value, np.float32) for value in features.values()]
            content = np.concatenate([v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS) for v in parts], 1)
        if content is None:
            raise ValueError("contentknn needs a content array")
        E = np.asarray(content, np.float32)
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + EPS)
        self.data_spec = deepcopy(kw.get('data_spec'))
        return self

    def score(self, seeds, user=None):
        return (self.E @ self.E[list(seeds)].mean(0)).astype(np.float32)


BASELINES = {c.name: c for c in (Popularity, CoOccurrence, ItemKNN, P3alpha, RP3beta, ContentKNN)}
