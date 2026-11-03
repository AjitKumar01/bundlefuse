"""Independent fullrank reference from 4516eab."""
def score_all_signals(bundles, required=None):
    """Computes raw score matrices R[k] and column-wise z-normalized Z[k] per query."""
    if required is not None:
        return selected_signal_scorer.score(bundles, required)
    nb = len(bundles)
    Ind = graph_op.build_indicator(bundles)
    sc_pair, sc_diff2_pair = graph_op.score_pairwise_and_diff2(bundles)
    delta_inc, diff2_delta = graph_op.score_simplicial_incidence_only(bundles, beta_clique=1.5)
    sc_rao = graph_op.score_rao_concordance(bundles, dense_feat=rao_dense_feat, cat_lbl=cat_lbl, gamma=1.0 if MUSIC else -1.0, alpha_dense=0.15, tau=0.05)
    sc_simp = sc_pair + delta_inc
    sc_diff2 = sc_diff2_pair + diff2_delta
    R = {'simp': torch.tensor(sc_simp, device=DEVICE), 'diff2': torch.tensor(sc_diff2, device=DEVICE), 'pair': torch.tensor(sc_pair, device=DEVICE), 'diff2_pair': torch.tensor(sc_diff2_pair, device=DEVICE), 'delta_inc': torch.tensor(delta_inc, device=DEVICE), 'rao': torch.tensor(sc_rao, device=DEVICE), 'pop': torch.tensor(np.tile(np.log1p(freq)[:, None], (1, nb)), dtype=torch.float32, device=DEVICE), 'cooc': torch.tensor(np.asarray((graph_op.C @ Ind).todense()), dtype=torch.float32, device=DEVICE), 'ppmi': torch.tensor(np.asarray((graph_op.PPMI @ Ind).todense()), dtype=torch.float32, device=DEVICE)}
    out_cat = np.zeros((n, nb), np.float32)
    for j, s in enumerate(bundles):
        sa = [cat_lbl[i] for i in s]
        out_cat[:, j] = np.asarray(onto_op.Pcat[sa].sum(0)).ravel()[cat_lbl]
    if not MUSIC:
        out_cat = out_cat + 0.5 * onto_op.score_slot_saturation(bundles)
    R['cat'] = torch.tensor(out_cat, device=DEVICE)
    if MUSIC:
        art_sc = np.asarray((onto_op.I_art_w @ (onto_op.I_art_w.T @ Ind)).todense())
        alb_sc = np.asarray((onto_op.I_alb_w @ (onto_op.I_alb_w.T @ Ind)).todense())
        R['homo'] = torch.tensor((art_sc + 2.0 * alb_sc).astype(np.float32), device=DEVICE)
    else:
        Ms = torch.stack([onto_op.Msoft_t[s].sum(0) for s in bundles], 1)
        R['softcat'] = onto_op.MP_t @ Ms
    if ENCODER:
        with torch.no_grad():
            sr = model.set_rep(*pad([s[:MAXK] for s in bundles]))
            R['contrast'] = Emb @ sr.t()
    for k in ('csvd', 'style', 'imgstyle', 'audiostyle', 'bridge'):
        if k in FEAT_T:
            f = FEAT_T[k]
            M = torch.stack([f[s].mean(0) for s in bundles], 1)
            R[k] = f @ M
    Z = {k: zcols(R[k]) for k in SIG}
    Z['pair'] = zcols(R['pair'])
    Z['diff2_pair'] = zcols(R['diff2_pair'])
    Z['rao'] = zcols(R['rao'])
    return (R, Z)
