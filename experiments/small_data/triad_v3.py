"""Three changes to the full-rank TRIAD (experiments.small_data.triad_v2) suggested by its trained parameters
(dials/TRIAD_DESIGN.md section 5). Each is a switch; all start the model exactly where TRIADv2 starts.

  sep_prior  The popularity prior rho(log(1+f_c)) leaves the graph term and becomes its own term with its own gate
             weight: s = w_C C + sum_l w_l L_l + w_G G_counts + w_P P. In TRIADv2 the prior and the counts share one
             per-query scaled term, so changing one changes the other's weight.
  taste      The content matrix M_zz = A_z'B_z is used only through its symmetric part (M_zz + M_zz')/2, and an
             explicit candidate preference u'z_c (u in R^d, starts at 0) is added to the prior. In TRIADv2 the
             antisymmetric part of M_zz acts only as the fixed direction K'mu; this makes that direction explicit, and
             lets the prior tell never-bundled items apart by their features.
  one_exp    The seed and candidate exponents are tied (alpha = beta): both start at 1/2 and their gradients are
             averaged, so they stay equal under AdamW.
"""
import math
import torch
import torch.nn as nn
from experiments.small_data.triad_v2 import TRIADv2


class TRIADv3(TRIADv2):
    def __init__(self, features, levels, train, sep_prior=False, taste=False, one_exp=False, **kw):
        super().__init__(features, levels, train, **kw)
        self.sep_prior, self.use_taste, self.one_exp = sep_prior, taste, one_exp
        if sep_prior:  # one more gate output (the prior term), starting at weight 1 like the others
            last = nn.Linear(self.gate[-1].in_features, self.gate[-1].out_features + 1)
            nn.init.zeros_(last.weight);nn.init.constant_(last.bias, math.log(math.e - 1));self.gate[-1] = last
        if taste:self.taste = nn.Parameter(torch.zeros(features.shape[1]))
        if one_exp:
            with torch.no_grad():self.exponents.fill_(float(self.exponents.mean()))
            self.exponents.register_hook(lambda g: g.mean().expand(2).clone())

    def fast_parameters(self):
        return super().fast_parameters() + ([self.taste] if self.use_taste else [])

    def content_scores(self, seeds, bundles, cache=None):
        out = super().content_scores(seeds, bundles, cache)
        if not self.use_taste:return out
        # Remove the antisymmetric part of the zz block: zbar'(A'B)z - 1/2 zbar'(A'B - B'A)z = zbar' sym(A'B) z.
        z = cache['content'] if cache is not None else self.content();zb = torch.stack([z[s].mean(0) for s in seeds])
        A, B = self.query_content, self.key_content
        return out - .5*((zb@A.T)@(z@B.T).T - (zb@B.T)@(z@A.T).T)

    def prior_scores(self, seeds, bundles, cache=None):
        """rho(log(1+f_c)) (+ u'z_c), leave-one-out frequencies for the bundle's own members in training (q x n)."""
        q = len(seeds)
        prior = (self.prior(self.frequency.log1p()[:, None]).squeeze(1) if cache is None else cache['prior']).expand(q, -1)
        if bundles is not None:
            rows = torch.cat([torch.full((len(b),), i) for i, b in enumerate(bundles)]);cols = torch.cat(bundles)
            left = (self.frequency[cols] - 1).clamp_min(0)
            prior = prior.index_put((rows, cols), self.prior(left.log1p()[:, None]).squeeze(1) - prior[rows, cols], accumulate=True)
        if self.use_taste:
            z = cache['content'] if cache is not None else self.content();prior = prior + (z@self.taste)[None, :]
        return prior

    def graph_scores(self, seeds, bundles, cache=None):
        full = super().graph_scores(seeds, bundles, cache)  # theta log(1 + counts) + rho
        if not (self.sep_prior or self.use_taste):return full
        q = len(seeds)
        rho = (self.prior(self.frequency.log1p()[:, None]).squeeze(1) if cache is None else cache['prior']).expand(q, -1)
        if bundles is not None:
            rows = torch.cat([torch.full((len(b),), i) for i, b in enumerate(bundles)]);cols = torch.cat(bundles)
            left = (self.frequency[cols] - 1).clamp_min(0)
            rho = rho.index_put((rows, cols), self.prior(left.log1p()[:, None]).squeeze(1) - rho[rows, cols], accumulate=True)
        counts = full - rho
        return counts if self.sep_prior else counts + self.prior_scores(seeds, bundles, cache)

    def forward(self, seeds, bundles=None, cache=None):
        if not self.sep_prior:return super().forward(seeds, bundles=bundles, cache=cache)
        seeds = [torch.as_tensor(sorted(set(s)), dtype=torch.long) for s in seeds]
        if bundles is not None:bundles = [torch.as_tensor(sorted(set(b)), dtype=torch.long) for b in bundles]
        parts = ([self.content_scores(seeds, bundles, cache)] + [self.label_scores(i, seeds) for i in range(self.levels)]
                 + [self.graph_scores(seeds, bundles, cache), self.prior_scores(seeds, bundles, cache)])
        parts = [v/v.abs().amax(1, keepdim=True).clamp_min(1e-6) for v in parts]
        w = self.gate_weights(seeds, bundles)
        return sum(w[:, j, None]*v for j, v in enumerate(parts))


VARIANTS = dict(v3_prior=dict(sep_prior=True), v3_taste=dict(taste=True), v3_oneexp=dict(one_exp=True),
                v3_all=dict(sep_prior=True, taste=True, one_exp=True))
