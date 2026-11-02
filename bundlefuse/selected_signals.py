"""Dependency-pruned fullrank scoring with the reference floating-point operation order.

No operator approximation or candidate pruning. The legacy fullrank function remains the
independent all-signals reference. Sharing the indicator and bundle-overlap computation avoids
duplicate work; diffusion and Rao are evaluated only when a selected configuration uses them.
"""
import numpy as np
import torch

from .operators import zcols


class SelectedSignals:
    def __init__(self, graph, ontology, feature_tensors, categories, rao_features, music, device):
        self.graph, self.ontology = graph, ontology
        self.features, self.categories = feature_tensors, categories
        self.rao_features, self.music, self.device = rao_features, music, device

    def score(self, bundles, required):
        required = set(required)
        supported = {'pair', 'diff2_pair', 'simp', 'diff2', 'delta_inc', 'rao', 'pop',
                     'cooc', 'ppmi', 'cat', 'homo', 'softcat'} | set(self.features)
        if not required or required - supported:
            raise ValueError(f'Invalid selected signals: {required - supported or required}')
        graph, onto = self.graph, self.ontology
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
                activation = overlap * (1.5 * np.maximum(0.0, overlap_count - 1.0))
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
        if __import__('os').environ.get('INCIDENCE') and ({'cooc', 'ppmi'} & required):
            # compute cooc/ppmi from the sparse incidence B (no materialized C/PPMI)
            Bm = graph.B; nb = len(bundles)
            if not hasattr(graph, '_fc'):
                graph._fc = np.asarray(graph.C.sum(1)).ravel() + 1e-9
                graph._totc = float(graph.C.sum())
            fc, totc = graph._fc, graph._totc
            if 'cooc' in required:
                tensor('cooc', np.asarray((Bm.T @ (Bm @ ind())).todense()).astype(np.float32))
            if 'ppmi' in required:
                pm = np.zeros((graph.n, nb), np.float32)
                for j, s in enumerate(bundles):
                    Cs = np.asarray((Bm.T @ Bm[:, list(s)]).todense())
                    with np.errstate(divide='ignore', invalid='ignore'):
                        pp = np.log(Cs * totc / (fc[:, None] * fc[list(s)][None, :] + 1e-12))
                    pp = np.maximum(pp, 0.0); pp[Cs <= 0] = 0.0
                    pm[:, j] = pp.sum(1)
                tensor('ppmi', pm)
        else:
            for key, operator in (('cooc', graph.C), ('ppmi', graph.PPMI)):
                if key in required:
                    tensor(key, np.asarray((operator @ ind()).todense()))
        if 'cat' in required:
            score = np.zeros((graph.n, len(bundles)), np.float32)
            for j, seeds in enumerate(bundles):
                labels = [self.categories[i] for i in seeds]
                score[:, j] = np.asarray(onto.Pcat[labels].sum(0)).ravel()[self.categories]
            if not self.music:
                score = score + .5 * onto.score_slot_saturation(bundles)
            tensor('cat', score)
        if 'homo' in required:
            artist = np.asarray((onto.I_art_w @ (onto.I_art_w.T @ ind())).todense())
            album = np.asarray((onto.I_alb_w @ (onto.I_alb_w.T @ ind())).todense())
            tensor('homo', (artist + 2.0 * album).astype(np.float32))
        if 'softcat' in required:
            means = torch.stack([onto.Msoft_t[seeds].sum(0) for seeds in bundles], 1)
            raw['softcat'] = onto.MP_t @ means
        for key in ('csvd', 'style', 'imgstyle', 'audiostyle', 'bridge'):
            if key in required:
                feature = self.features[key]
                means = torch.stack([feature[seeds].mean(0) for seeds in bundles], 1)
                raw[key] = feature @ means
        if set(raw) != required:
            raise ValueError(f'Unavailable selected signals: {required - set(raw)}')
        return raw, {key: zcols(value) for key, value in raw.items()}
