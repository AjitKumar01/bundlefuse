"""Retail and commerce signals, with full and dependency-pruned scoring.

The full scorer provides the reference path; serving computes only fitted dependencies.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from bundlefuse.operators import zcols, ensure_graph_matrices

class BasketSignals:

    def __init__(self, graph, categories, data, train_bundles):
        self.graph = graph
        self.kind = data['kind']
        self.n = graph.n
        self.cat_lbl = categories
        self.NCc = int(categories.max()) + 1
        self.CONTENT = data['content']
        self.tf_mat = data['tfidf']
        if self.kind == 'retail':
            self.nusers = data['n_users']
            self.keys = ['simp', 'diff2', 'pop', 'cooc', 'hh_cooc', 'ppmi', 'csvd', 'cat', 'content', 'bridge']
            self.built = self._build_retail(train_bundles, data['train_users'])[1:]
        else:
            self.cat_l2 = data['cat_l2']
            self.NCl2 = data['n_cat_l2']
            self.b2u = data['bundle_users']
            self.b2s = data['bundle_sessions']
            self.C_pre = data['pretrain']
            self.keys = ['simp', 'diff2', 'cooc', 'ppmi', 'csvd', 'cat', 'homo', 'tfidf', 'bridge', 'pretrain']
            self.built = self._build_commerce(train_bundles, data['train_keys'])[1:]

    def score(self, seeds, required=None):
        if required is not None:
            return score_selected(self,seeds,required)
        ensure_graph_matrices(self.graph, {'cooc', 'ppmi'})
        built = (self.graph,) + self.built
        function = self._full_retail if self.kind == 'retail' else self._full_commerce
        raw, normalized = function([(s, []) for s in seeds], built)
        return raw, normalized

    def _build_retail(self, trb, tru):
        graph_op = self.graph
        freq = graph_op.freq
        ur, uc = ([], [])
        for b, u in zip(trb, tru):
            for it in b:
                ur.append(u)
                uc.append(it)
        X_u = sp.csr_matrix((np.ones(len(ur), np.float32), (ur, uc)), shape=(self.nusers, self.n))
        udeg = np.asarray(X_u.sum(1)).ravel() + 1.0
        X_uw = sp.diags(1.0 / np.sqrt(udeg)) @ X_u
        C_hh = (X_uw.T @ X_uw).tocsr()
        C_hh.setdiag(0)
        tr_, tc_ = ([], [])
        for bi_idx, b in enumerate(trb):
            for it in b:
                tr_.append(bi_idx)
                tc_.append(self.cat_lbl[it])
        T = sp.csr_matrix((np.ones(len(tr_), np.float32), (tr_, tc_)), shape=(len(trb), self.NCc))
        Ct = (T.T @ T).tocsr()
        Ct.setdiag(0)
        Ct.eliminate_zeros()
        fc = np.asarray(T.sum(0)).ravel() + 1e-09
        totc = float(fc.sum())
        coc = Ct.tocoo()
        pcv = np.maximum(0.0, np.log(coc.data * totc / (fc[coc.row] * fc[coc.col]) + 1e-12)).astype(np.float32)
        Pcat = sp.csr_matrix((pcv, (coc.row, coc.col)), shape=(self.NCc, self.NCc))
        p_cont = graph_op.propagate(self.CONTENT)
        p_cont /= np.linalg.norm(p_cont, axis=1, keepdims=True) + 1e-08
        BR_CONT = self.CONTENT + 0.5 * p_cont
        BR_CONT /= np.linalg.norm(BR_CONT, axis=1, keepdims=True) + 1e-08
        model, E, pad = (None, None, None)
        return (graph_op, freq, C_hh, Pcat, BR_CONT, model, E, pad)

    def _full_retail(self, bundles, built):
        graph_op, freq, C_hh, Pcat, BR_CONT, model, E, pad = built
        nb = len(bundles)
        seeds_list = [s for s, _ in bundles]
        Ind = graph_op.build_indicator(seeds_list)
        Ind_w = Ind.multiply(graph_op.inv_sqrt_f[:, None]).tocsr()
        sc_pair, sc_diff2_pair = graph_op.score_pairwise_and_diff2(seeds_list)
        delta_inc, diff2_delta = graph_op.score_simplicial_incidence_only(seeds_list, beta_clique=graph_op.beta_clique)
        sc_simp = sc_pair + delta_inc
        sc_diff2 = sc_diff2_pair + diff2_delta
        sc_rao = graph_op.score_rao_concordance(seeds_list, dense_feat=BR_CONT, cat_lbl=self.cat_lbl, gamma=-1.0, alpha_dense=0.15, tau=0.05)
        sc_pop = np.tile(np.log1p(freq)[:, None], (1, nb)).astype(np.float32)
        sc_cooc = np.asarray((graph_op.C @ Ind).todense()).astype(np.float32)
        sc_hh = np.asarray((C_hh @ Ind_w).todense()).astype(np.float32)
        sc_ppmi = np.asarray((graph_op.PPMI @ Ind).todense()).astype(np.float32)
        Us = np.stack([graph_op.Ucsvd[s].mean(0) for s in seeds_list], 1)
        sc_csvd = (graph_op.Ucsvd @ Us).astype(np.float32)
        sc_cat = np.zeros((self.n, nb), np.float32)
        for j, s in enumerate(seeds_list):
            sa = [self.cat_lbl[i] for i in s]
            sc_cat[:, j] = np.asarray(Pcat[sa].sum(0)).ravel()[self.cat_lbl]
        Cs = np.stack([self.CONTENT[s].mean(0) for s in seeds_list], 1)
        sc_cont = (self.CONTENT @ Cs).astype(np.float32)
        Bs = np.stack([BR_CONT[s].mean(0) for s in seeds_list], 1)
        sc_br = (BR_CONT @ Bs).astype(np.float32)
        sc_tf = np.asarray((self.tf_mat @ (self.tf_mat.T @ Ind)).todense()).astype(np.float32)
        sc_con = None
        R = {'simp': sc_simp, 'diff2': sc_diff2, 'pair': sc_pair, 'diff2_pair': sc_diff2_pair, 'rao': sc_rao, 'pop': sc_pop, 'cooc': sc_cooc, 'hh_cooc': sc_hh, 'ppmi': sc_ppmi, 'csvd': sc_csvd, 'cat': sc_cat, 'content': sc_cont, 'bridge': sc_br, 'contrast': sc_con, 'tfidf': sc_tf}
        Z = {k: zcols(R[k]) for k in self.keys}
        Z['pair'] = zcols(R['pair'])
        Z['diff2_pair'] = zcols(R['diff2_pair'])
        Z['rao'] = zcols(R['rao'])
        Z['tfidf'] = zcols(R['tfidf'])
        return (R, Z)

    def _build_commerce(self, tr_bundles, tr_bkeys):
        graph_op = self.graph
        freq = graph_op.freq
        ur, uc = ([], [])
        umap_tr = {}
        sr, sc_ = ([], [])
        smap_tr = {}
        for bi_key, b in zip(tr_bkeys, tr_bundles):
            u = self.b2u.get(bi_key, -1)
            if u >= 0:
                ui = umap_tr.setdefault(u, len(umap_tr))
                for it in b:
                    ur.append(ui)
                    uc.append(it)
            s_id = self.b2s.get(bi_key, -1)
            if s_id >= 0:
                si = smap_tr.setdefault(s_id, len(smap_tr))
                for it in b:
                    sr.append(si)
                    sc_.append(it)
        X_u = sp.csr_matrix((np.ones(len(ur), np.float32), (ur, uc)), shape=(max(len(umap_tr), 1), self.n))
        C_u = (X_u.T @ X_u).tocsr()
        C_u.setdiag(0)
        X_s = sp.csr_matrix((np.ones(len(sr), np.float32), (sr, sc_)), shape=(max(len(smap_tr), 1), self.n))
        C_s = (X_s.T @ X_s).tocsr()
        C_s.setdiag(0)
        P = (sp.diags(1.0 / np.sqrt(freq + 1.0)) @ (graph_op.C + 0.5 * C_s + 0.5 * C_u + 0.2 * self.C_pre) @ sp.diags(1.0 / np.sqrt(freq + 1.0))).tocsr()
        TF_bridge = normalize(self.tf_mat + 1.5 * (P @ self.tf_mat) + 0.5 * (P @ (P @ self.tf_mat)), norm='l2', axis=1).tocsr()
        BR_DENSE = (self.CONTENT + 1.5 * (P @ self.CONTENT) + 0.5 * (P @ (P @ self.CONTENT))).astype(np.float32)
        BR_DENSE /= np.linalg.norm(BR_DENSE, axis=1, keepdims=True) + 1e-08

        def build_pcat(lbl_arr, n_types, keep_diag=False):
            tr_, tc_ = ([], [])
            for bi, b in enumerate(tr_bundles):
                for it in b:
                    tr_.append(bi)
                    tc_.append(lbl_arr[it])
            T = sp.csr_matrix((np.ones(len(tr_), np.float32), (tr_, tc_)), shape=(len(tr_bundles), n_types))
            Ct = (T.T @ T).tocsr()
            if not keep_diag:
                Ct.setdiag(0)
                Ct.eliminate_zeros()
            fc = np.asarray(T.sum(0)).ravel() + 1e-09
            totc = float(fc.sum())
            coc = Ct.tocoo()
            pc_ = np.maximum(0.0, np.log(coc.data * totc / (fc[coc.row] * fc[coc.col]) + 1e-12)).astype(np.float32)
            return sp.csr_matrix((pc_, (coc.row, coc.col)), shape=(n_types, n_types))
        Pcat_leaf = build_pcat(self.cat_lbl, self.NCc, keep_diag=False)
        Pcat_homo = build_pcat(self.cat_lbl, self.NCc, keep_diag=True)
        Pcat_l2 = build_pcat(self.cat_l2, self.NCl2, keep_diag=True)
        model, E, pad = (None, None, None)
        return (graph_op, C_u, C_s, P, TF_bridge, BR_DENSE, Pcat_leaf, Pcat_homo, Pcat_l2, model, E, pad)

    def _full_commerce(self, bset, built):
        graph_op, C_u, C_s, P, TF_bridge, BR_DENSE, Pcat_leaf, Pcat_homo, Pcat_l2, model, E, pad = built
        seeds_list = [s for s, _ in bset]
        nb = len(seeds_list)
        Ind = graph_op.build_indicator(seeds_list)
        s_pair_b, _ = graph_op.score_pairwise_and_diff2(seeds_list)
        delta_inc, _ = graph_op.score_simplicial_incidence_only(seeds_list, beta_clique=graph_op.beta_clique)
        s_u = np.asarray((C_u @ Ind).todense())
        s_s = np.asarray((C_s @ Ind).todense())
        s_pair = (s_pair_b + 0.35 * s_u + 0.35 * s_s).astype(np.float32)
        s_diff2_pair = np.asarray(P @ s_pair).astype(np.float32)
        s_simp = (s_pair + delta_inc).astype(np.float32)
        s_diff2 = np.asarray(P @ s_simp).astype(np.float32)
        s_rao = graph_op.score_rao_concordance(seeds_list, dense_feat=BR_DENSE, cat_lbl=self.cat_lbl, gamma=-1.0, alpha_dense=0.25, tau=0.05)
        s_cooc = np.asarray((graph_op.C @ Ind).todense()).astype(np.float32)
        s_ppmi = np.asarray((graph_op.PPMI @ Ind).todense()).astype(np.float32)
        Us = np.stack([graph_op.Ucsvd[s].mean(0) for s in seeds_list], 1)
        s_csvd = (graph_op.Ucsvd @ Us).astype(np.float32)
        s_cat = np.zeros((self.n, nb), np.float32)
        s_homo = np.zeros((self.n, nb), np.float32)
        for j, s in enumerate(seeds_list):
            sa = [self.cat_lbl[i] for i in s]
            sa2 = [self.cat_l2[i] for i in s]
            s_cat[:, j] = np.asarray(Pcat_leaf[sa].sum(0)).ravel()[self.cat_lbl]
            s_homo[:, j] = np.asarray(Pcat_homo[sa].sum(0)).ravel()[self.cat_lbl] + 0.5 * np.asarray(Pcat_l2[sa2].sum(0)).ravel()[self.cat_l2]
        s_tfidf = np.asarray((self.tf_mat @ (self.tf_mat.T @ Ind)).todense()).astype(np.float32)
        s_bridge = np.asarray((TF_bridge @ (TF_bridge.T @ Ind)).todense()).astype(np.float32)
        s_pre = np.asarray((self.C_pre @ Ind).todense()).astype(np.float32)
        s_con = None
        s_img = None
        R = {'simp': s_simp, 'diff2': s_diff2, 'pair': s_pair, 'diff2_pair': s_diff2_pair, 'rao': s_rao, 'cooc': s_cooc, 'ppmi': s_ppmi, 'csvd': s_csvd, 'cat': s_cat, 'homo': s_homo, 'tfidf': s_tfidf, 'bridge': s_bridge, 'pretrain': s_pre, 'contrast': s_con, 'img': s_img}
        Z = {k: zcols(R[k]) for k in self.keys}
        Z['pair'] = zcols(R['pair'])
        Z['diff2_pair'] = zcols(R['diff2_pair'])
        Z['rao'] = zcols(R['rao'])
        return (R, Z)


def score_selected(provider, bundles, required):
    required=set(required);g=provider.graph
    ensure_graph_matrices(g,required)
    indicator=g.build_indicator(bundles);raw={}
    retail=provider.kind=='retail'
    if retail:
        freq,household,category,bridge=provider.built[:4]
    else:
        user,session,transition,tf_bridge,dense_bridge,category,homo,category_l2=provider.built[:8]
    need_pair=required & {'pair','simp','diff2_pair','diff2'}
    need_inc=required & {'simp','diff2'}
    if need_pair:
        weighted=indicator.multiply(g.inv_sqrt_f[:,None]).tocsr()
        overlap=np.asarray((g.B@weighted).todense())*g.inv_vol_pow[:,None]
        pair=np.asarray(g.B.T@overlap).astype(np.float32)
        if not retail:
            pair=(pair+.35*np.asarray((user@indicator).todense())+.35*np.asarray((session@indicator).todense())).astype(np.float32)
        if 'pair' in required:raw['pair']=pair
        if 'diff2_pair' in required or (retail and 'diff2' in required):
            diff_pair=g.propagate(pair) if retail else np.asarray(transition@pair).astype(np.float32)
            if 'diff2_pair' in required:raw['diff2_pair']=diff_pair
        if need_inc:
            count=np.asarray((g.B_bin@indicator).todense())
            activation=overlap*(g.beta_clique*np.maximum(0.,count-1.))
            delta=np.asarray(g.B.T@activation).astype(np.float32)
            if 'simp' in required:raw['simp']=(pair+delta).astype(np.float32)
            if 'diff2' in required:
                raw['diff2']=(diff_pair+g.propagate(delta) if retail else transition@(pair+delta)).astype(np.float32)
    if 'rao' in required:
        raw['rao']=g.score_rao_concordance(bundles,dense_feat=bridge if retail else dense_bridge,
                         cat_lbl=provider.cat_lbl,gamma=-1.,alpha_dense=.15 if retail else .25,tau=.05)
    if 'pop' in required:
        raw['pop']=np.tile(np.log1p(g.freq)[:,None],(1,len(bundles))).astype(np.float32)
    for key,operator in (('cooc',g.C),('ppmi',g.PPMI)):
        if key in required:raw[key]=np.asarray((operator@indicator).todense()).astype(np.float32)
    if 'hh_cooc' in required:
        weighted=indicator.multiply(g.inv_sqrt_f[:,None]).tocsr()
        raw['hh_cooc']=np.asarray((household@weighted).todense()).astype(np.float32)
    for key,features in (('csvd',g.Ucsvd),('content',provider.CONTENT),('bridge',bridge if retail else None)):
        if key in required and features is not None:
            means=np.stack([features[s].mean(0) for s in bundles],1)
            raw[key]=(features@means).astype(np.float32)
    if 'cat' in required:
        values=np.zeros((g.n,len(bundles)),np.float32)
        for j,seeds in enumerate(bundles):
            labels=[provider.cat_lbl[i] for i in seeds]
            values[:,j]=np.asarray(category[labels].sum(0)).ravel()[provider.cat_lbl]
        raw['cat']=values
    if 'homo' in required:
        values=np.zeros((g.n,len(bundles)),np.float32)
        for j,seeds in enumerate(bundles):
            labels=[provider.cat_lbl[i] for i in seeds];second=[provider.cat_l2[i] for i in seeds]
            values[:,j]=np.asarray(homo[labels].sum(0)).ravel()[provider.cat_lbl]+.5*np.asarray(category_l2[second].sum(0)).ravel()[provider.cat_l2]
        raw['homo']=values
    if 'tfidf' in required:
        raw['tfidf']=np.asarray((provider.tf_mat@(provider.tf_mat.T@indicator)).todense()).astype(np.float32)
    if 'bridge' in required and not retail:
        raw['bridge']=np.asarray((tf_bridge@(tf_bridge.T@indicator)).todense()).astype(np.float32)
    if 'pretrain' in required:
        raw['pretrain']=np.asarray((provider.C_pre@indicator).todense()).astype(np.float32)
    if set(raw)!=required:
        raise ValueError(f'Unavailable basket signals: {required-set(raw)}')
    return raw,{key:zcols(value) for key,value in raw.items()}
