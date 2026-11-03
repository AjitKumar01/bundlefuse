"""Independent selector reference preserved from 4516eab dunnhumby.py."""
Zv_pair = dict(Zv)
Zv_pair['simp'] = Zv['pair']
Zv_pair['diff2'] = Zv['diff2_pair']
w = {k: 0.0 for k in SIG}
best = -1e+18
for _ in range(4):
    imp = False
    for k in SIG:
        for val in [0.0, 0.25, 0.5, 1.0, 1.5]:
            w2 = dict(w)
            w2[k] = val
            if sum(w2.values()) == 0:
                continue
            F_v = sum((w2[k2] * Zv_pair[k2] for k2 in w2 if w2[k2]))
            nd = sel_score(F_v, vset)
            if nd > best + 0.0002:
                best = nd
                w = w2
                imp = True
    if not imp:
        break
bw = EmpiricalBayesRouter.prune_weights_val(w, Zv_pair, lambda fmat: sel_score(fmat, vset), optional_keys=('bridge', 'contrast', 'cat'), val_eps=0.0003)

def build_Y(Z_dict, ho_params):
    alpha_inc, g_rao = ho_params
    Zp = dict(Z_dict)
    Zp['simp'] = Z_dict['pair']
    Zp['diff2'] = Z_dict['diff2_pair']
    base_p = sum((bw[k] * Zp[k] for k in bw))
    base_i = sum((bw[k] * Z_dict[k] for k in bw))
    return (1.0 - alpha_inc) * base_p + alpha_inc * base_i + g_rao * Z_dict['rao']
best_ho = (1.0, 0.0)
best_ho_nd = -1e+18
for a_inc in [0.0, 0.5, 1.0]:
    for g_rao in [0.0, 0.15, 0.25]:
        nd = sel_score(build_Y(Zv, (a_inc, g_rao)), vset)
        if nd >= best_ho_nd - 1e-05:
            best_ho_nd = nd
            best_ho = (a_inc, g_rao)
best = max(best, best_ho_nd)
prior_ks = [k for k in ('cat', 'content', 'bridge', 'contrast') if bw.get(k, 0) > 0] or ['cat', 'bridge']
Y_v = build_Y(Zv, best_ho)
raw_pr_v = sum((bw.get(k, 0.5) * Zv[k] for k in prior_ks)) + best_ho[1] * Zv['rao']
if bw.get('pop', 0) > 0:
    raw_pr_v = raw_pr_v + bw['pop'] * Zv['pop']
frozen_prior_scale = float(np.std(Y_v) / (np.std(raw_pr_v) + 1e-09))
mu0_v = raw_pr_v * frozen_prior_scale
vc = EmpiricalBayesRouter.fit_variance_components(freq, Y_v, mu0_v)

def apply_eb(Z_dict, ho_params, kf, p_scale):
    Y = build_Y(Z_dict, ho_params)
    if kf <= 0.0:
        return Y
    raw_prior = sum((bw.get(k, 0.5) * Z_dict[k] for k in prior_ks)) + ho_params[1] * Z_dict['rao']
    if bw.get('pop', 0) > 0:
        raw_prior = raw_prior + bw['pop'] * Z_dict['pop']
    mu_0 = raw_prior * float(p_scale)
    return EmpiricalBayesRouter.conjugate_posterior(Y, mu_0, freq, kf)
best_kf = 0.0
for kf_cand in EmpiricalBayesRouter.shrinkage_candidates(vc, [0.0, 0.05, 0.25, vc['kappa_f'], 1.0]):
    nd = sel_score(apply_eb(Zv, best_ho, kf_cand, frozen_prior_scale), vset)
    if nd > best + 0.0001:
        best = nd
        best_kf = kf_cand
