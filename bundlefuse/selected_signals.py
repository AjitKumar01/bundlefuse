"""Dependency-pruned fullrank scoring with the reference floating-point operation order.

No operator approximation or candidate pruning. BundleFuse._score_all remains the
independent all-signals reference. Sharing the indicator and bundle-overlap computation avoids
duplicate work; diffusion and Rao are evaluated only when a selected configuration uses them.
"""
import numpy as np
import scipy.sparse as sp
import torch

from .operators import zcols
from .graph_state import ensure_graph_matrices


class SelectedSignals:
    def __init__(self, graph, ontology, feature_tensors, categories, rao_features, music, device,
                 *, beta_clique=1.5, aux=None, groups=None, category_homo=None,
                 user_incidence=None, user_items=None):
        self.graph, self.ontology = graph, ontology
        self.features, self.categories = feature_tensors, categories
        self.rao_features, self.music, self.device = rao_features, music, device
        self.beta_clique, self.aux = beta_clique, aux or {}
        self.groups, self.category_homo = groups or [], category_homo
        self.user_incidence, self.user_items = user_incidence, user_items or {}

    def score(self, bundles, required, users=None):
        required = set(required)
        supported = {'pair', 'diff2_pair', 'simp', 'diff2', 'delta_inc', 'rao', 'pop',
                     'cooc', 'ppmi', 'cat', 'homo', 'softcat'} | set(self.features) | set(self.aux)
        if self.user_incidence is not None:
            supported.update(('user_freq', 'user_cooc'))
        if not required or required - supported:
            raise ValueError(f'Invalid selected signals: {required - supported or required}')
        graph, onto = self.graph, self.ontology
        ensure_graph_matrices(graph, required)
        raw = {}
        indicator = None
        def ind():
            nonlocal indicator
            if indicator is None:
                indicator = graph.build_indicator(bundles)
            return indicator
        def tensor(key, value):
            raw[key] = torch.tensor(value, dtype=torch.float32, device=self.device)

        pair_keys = {'pair', 'diff2_pair', 'simp', 'diff2'}
        incidence_keys = {'simp', 'diff2', 'delta_inc'}
        if required & (pair_keys | incidence_keys):
            weighted = ind().multiply(graph.inv_sqrt_f[:, None]).tocsr()
            overlap = np.asarray((graph.B @ weighted).todense()) * graph.inv_vol_pow[:, None]
            if required & pair_keys:
                pair = np.asarray(graph.B.T @ overlap).astype(np.float32)
                if 'pair' in required:
                    tensor('pair', pair)
                if required & {'diff2_pair', 'diff2'}:
                    diff_pair = graph.propagate(pair)
                    if 'diff2_pair' in required:
                        tensor('diff2_pair', diff_pair)
            if required & incidence_keys:
                overlap_count = np.asarray((graph.B_bin @ ind()).todense())
                activation = overlap * (self.beta_clique * np.maximum(0.0, overlap_count - 1.0))
                delta = np.asarray(graph.B.T @ activation).astype(np.float32)
                if 'delta_inc' in required:
                    tensor('delta_inc', delta)
                if 'simp' in required:
                    tensor('simp', pair + delta)
                if 'diff2' in required:
                    tensor('diff2', diff_pair + graph.propagate(delta))
        if 'rao' in required:
            tensor('rao', graph.score_rao_concordance(
                bundles, dense_feat=self.rao_features, cat_lbl=self.categories,
                gamma=1.0 if self.music else -1.0, alpha_dense=.15, tau=.05))
        if 'pop' in required:
            tensor('pop', np.tile(np.log1p(graph.freq)[:, None], (1, len(bundles))))
        for key in ('cooc', 'ppmi'):
            if key in required:
                operator = graph.C if key == 'cooc' else graph.PPMI
                tensor(key, np.asarray((operator @ ind()).todense()))
        for key, (incidence, weighted) in self.aux.items():
            if key in required:
                x = ind().multiply(graph.inv_sqrt_f[:, None]).tocsr() if weighted else ind()
                tensor(key, np.asarray((incidence.T @ (incidence @ x)).todense()).astype(np.float32))
        if 'cat' in required:
            score = np.zeros((graph.n, len(bundles)), np.float32)
            for j, seeds in enumerate(bundles):
                labels = [self.categories[i] for i in seeds]
                score[:, j] = np.asarray(onto.Pcat[labels].sum(0)).ravel()[self.categories]
            if not self.music:
                score = score + .5 * onto.score_slot_saturation(bundles)
            tensor('cat', score)
        if 'homo' in required:
            if self.music:
                artist = np.asarray((onto.I_art_w @ (onto.I_art_w.T @ ind())).todense())
                album = (np.asarray((onto.I_alb_w @ (onto.I_alb_w.T @ ind())).todense())
                         if onto.I_alb_w is not None else 0.)
                tensor('homo', (artist + 2.0 * album).astype(np.float32))
            elif self.groups:
                score = sum(w * np.asarray((I @ (I.T @ ind())).todense()) for I, w in self.groups)
                tensor('homo', score.astype(np.float32))
            elif self.category_homo is not None:
                score = np.zeros((graph.n, len(bundles)), np.float32)
                for j, seeds in enumerate(bundles):
                    labels = [self.categories[i] for i in seeds]
                    score[:, j] = np.asarray(self.category_homo[labels].sum(0)).ravel()[self.categories]
                tensor('homo', score)
        if 'softcat' in required:
            means = torch.stack([onto.Msoft_t[seeds].sum(0) for seeds in bundles], 1)
            raw['softcat'] = onto.MP_t @ means
        for key in self.features:
            if key in required:
                feature = self.features[key]
                means = torch.stack([feature[seeds].mean(0) for seeds in bundles], 1)
                raw[key] = feature @ means
        if required & {'user_freq', 'user_cooc'}:
            users = [None] * len(bundles) if users is None else users
            if len(users) != len(bundles):
                raise ValueError('Users and queries must align')
            frequency = np.zeros((graph.n, len(bundles)), np.float32)
            history = sp.lil_matrix((graph.n, len(bundles)))
            for j, user in enumerate(users):
                if user is not None and user in self.user_items:
                    if 'user_freq' in required:
                        frequency[:, j] = np.asarray(self.user_incidence[:, user].todense()).ravel()
                    if 'user_cooc' in required:
                        history[self.user_items[user], j] = 1.
            if 'user_freq' in required:
                tensor('user_freq', frequency)
            if 'user_cooc' in required:
                tensor('user_cooc', np.asarray((graph.C @ history.tocsr()).todense()).astype(np.float32))
        if set(raw) != required:
            raise ValueError(f'Unavailable selected signals: {required - set(raw)}')
        return raw, {key: zcols(value) for key, value in raw.items()}
