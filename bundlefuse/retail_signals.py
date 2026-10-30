"""Shared retail graph/content scoring, using only the supplied training interactions."""
import numpy as np
import scipy.sparse as sp

from .baselines.hybrid import SimpleHybrid, category_scores
from .operators import SimplicialGraphOperator, zcols


class RetailSignals:
    def __init__(self, content, tfidf, categories, n_users):
        self.content = np.asarray(content, dtype=np.float32)
        self.tfidf = tfidf
        self.categories = np.asarray(categories)
        self.n = len(categories)
        self.n_users = n_users
        self.keys = ['simp', 'diff2', 'pop', 'cooc', 'hh_cooc', 'ppmi', 'csvd', 'cat', 'content', 'bridge']

    def build(self, train, users):
        graph = SimplicialGraphOperator(train, self.n, svd_dim=128, alpha_deg=.5, gamma_vol=.5, beta_clique=1.5)
        rows, cols = [], []
        for bundle, user in zip(train, users):
            rows.extend([user] * len(bundle)); cols.extend(bundle)
        interactions = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(self.n_users, self.n))
        weighted = sp.diags(1 / np.sqrt(np.asarray(interactions.sum(1)).ravel() + 1)) @ interactions
        household = (weighted.T @ weighted).tocsr(); household.setdiag(0)
        rows, cols = [], []
        for j, bundle in enumerate(train):
            rows.extend([j] * len(bundle)); cols.extend(self.categories[bundle])
        nc = int(self.categories.max()) + 1
        membership = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(train), nc))
        cooc = (membership.T @ membership).tocsr(); cooc.setdiag(0); cooc.eliminate_zeros()
        counts = np.asarray(membership.sum(0)).ravel() + 1e-9
        pairs = cooc.tocoo()
        values = np.maximum(0., np.log(pairs.data * float(counts.sum()) / (counts[pairs.row] * counts[pairs.col]) + 1e-12)).astype(np.float32)
        category = sp.csr_matrix((values, (pairs.row, pairs.col)), shape=(nc, nc))
        propagated = graph.propagate(self.content)
        propagated /= np.linalg.norm(propagated, axis=1, keepdims=True) + 1e-8
        bridge = self.content + .5 * propagated
        bridge /= np.linalg.norm(bridge, axis=1, keepdims=True) + 1e-8
        self.graph, self.household, self.category, self.bridge = graph, household, category, bridge
        self.interactions = interactions
        return self

    def score(self, seeds):
        g = self.graph
        indicator = g.build_indicator(seeds)
        pair, diff2_pair = g.score_pairwise_and_diff2(seeds)
        incidence, diff2_incidence = g.score_simplicial_incidence_only(seeds, beta_clique=1.5)
        svd_query = np.stack([g.Ucsvd[s].mean(0) for s in seeds], 1)
        content_query = np.stack([self.content[s].mean(0) for s in seeds], 1)
        bridge_query = np.stack([self.bridge[s].mean(0) for s in seeds], 1)
        raw = {'simp': pair + incidence, 'diff2': diff2_pair + diff2_incidence,
               'pair': pair, 'diff2_pair': diff2_pair,
               'rao': g.score_rao_concordance(seeds, dense_feat=self.bridge, cat_lbl=self.categories,
                                              gamma=-1., alpha_dense=.15, tau=.05),
               'pop': np.tile(np.log1p(g.freq)[:, None], (1, len(seeds))),
               'cooc': (g.C @ indicator).toarray(),
               'hh_cooc': (self.household @ indicator.multiply(g.inv_sqrt_f[:, None]).tocsr()).toarray(),
               'ppmi': (g.PPMI @ indicator).toarray(), 'csvd': g.Ucsvd @ svd_query,
               'cat': category_scores(self.category, self.categories, seeds),
               'content': self.content @ content_query, 'bridge': self.bridge @ bridge_query}
        raw = {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}
        normalized = {key: zcols(value) for key, value in raw.items()}
        return raw, normalized

    def hybrid(self):
        return SimpleHybrid(self.graph, self.tfidf,
                            lambda seeds: category_scores(self.category, self.categories, seeds),
                            {'household': self.household})

    def auxiliary_bundles(self):
        return [self.interactions.indices[self.interactions.indptr[j]:self.interactions.indptr[j + 1]].tolist()
                for j in range(self.interactions.shape[0]) if self.interactions.indptr[j + 1] - self.interactions.indptr[j] >= 2]
