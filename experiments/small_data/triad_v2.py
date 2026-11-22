"""TRIAD with a principled starting point: one model, same training loop, so that at initialization it already is a
sensible simple scorer and training learns the relaxations from there.

Four changes to the architecture's start (each optional, for the diagnosis), none adds a separate model:
  norm     each term is rescaled per query before the gate (divided by its largest absolute value over the catalogue),
           so the gate mixes signals of comparable scale, as tuned blends do (max-abs scaling). Without it the gate
           must first learn the relative scales of content (x10 dot products), log counts and the label MLP.
  cosine   the degree exponents start at alpha = beta = 1/2 (symmetric normalization, close to ItemKNN's cosine)
           instead of 0 (raw counts).
  full     the content maps have rank d = 128 and start at sqrt(10) I, so the initial content score uses all feature
           directions, not only the first 64.
  quiet    the label score delta_l and the prior rho start at exactly 0 (last layer zero), so untrained small networks
           add no noise; their sign and shape are still learned.
  anchor_content  optional: content = 10 x full-d similarity of the seed mean (fixed, no parameters) + the learned
           rank-r maps as a correction starting at 0 (B_z = 0); keeps the parameter count of rank r.
  label_bonus  optional: delta_l starts at +1 for a shared label (the simple scorer's same-label bonus) with gate weight
           label_bonus; its sign is still learned.
"""
import math
import torch
import torch.nn as nn
from triad.model import TRIAD


class TRIADv2(TRIAD):
    def __init__(self, features, levels, train, rank=64, hidden=32, norm=True, cosine=True, full=True, quiet=True, label_bonus=0., anchor_content=False):
        d = features.shape[1]
        super().__init__(features, levels, train, rank=d if full else rank, hidden=hidden)
        self.norm = norm;self.anchor_content = anchor_content
        if anchor_content:  # fixed full-d similarity + learned rank-r correction that starts at 0 (B_z = 0)
            with torch.no_grad():self.key_content.zero_()
        if full:
            start = math.sqrt(10.)*torch.eye(d)
            with torch.no_grad():self.query_content.copy_(start);self.key_content.copy_(start)
        if cosine:
            with torch.no_grad():self.exponents.fill_(.5)
        if quiet:
            for net in [*self.same_label, self.prior]:
                nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
        if label_bonus:  # start the label rule as the simple scorer's same-label bonus: +1 for a shared label,
            # mixed in with a small gate weight (softplus = label_bonus); its sign and shape stay learnable
            with torch.no_grad():
                for net in self.same_label:net[-1].bias.fill_(1.)
                b = math.log(math.expm1(label_bonus));self.gate[-1].bias[1:1 + self.levels].fill_(b)

    def content_scores(self, seeds, bundles, cache=None):
        out = super().content_scores(seeds, bundles, cache)
        if not self.anchor_content:return out
        x = self.features;mean = torch.stack([x[s].mean(0) for s in seeds])
        return out + 10.*(mean@x.T)

    def forward(self, seeds, bundles=None, cache=None):
        if not self.norm:return super().forward(seeds, bundles=bundles, cache=cache)
        seeds = [torch.as_tensor(sorted(set(s)), dtype=torch.long) for s in seeds]
        if bundles is not None:bundles = [torch.as_tensor(sorted(set(b)), dtype=torch.long) for b in bundles]
        parts = ([self.content_scores(seeds, bundles, cache)] + [self.label_scores(i, seeds) for i in range(self.levels)]
                 + [self.graph_scores(seeds, bundles, cache)])
        parts = [v/v.abs().amax(1, keepdim=True).clamp_min(1e-6) for v in parts]
        w = self.gate_weights(seeds, bundles)
        return sum(w[:, j, None]*v for j, v in enumerate(parts))
