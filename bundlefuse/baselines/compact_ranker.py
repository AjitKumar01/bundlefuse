"""Small feature-only two tower: shared item projection and nonlinear pooled query tower.

No trainable item/category-ID tables, teacher supervision, frequency backoff, or sparse-input
augmentation. Frozen inputs are identical in width for the raw/engineered comparison.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .metadata_ranker import pad_seeds


GRID = tuple({'dim': 128, 'lr': lr, 'regularization': 1e-4, 'negatives': 32,
              'batch_size': 128, 'max_epochs': 100, 'evaluate_every': 5, 'patience': 4}
             for lr in (.001, .003))


class CompactRanker(nn.Module):
    def __init__(self, features, frequency, dim=128, hidden=64):
        super().__init__()
        self.n, self.dim = len(frequency), dim
        self.register_buffer('features', torch.as_tensor(features, dtype=torch.float32), persistent=False)
        logfreq = np.log1p(frequency).astype(np.float32)
        logfreq = (logfreq - logfreq.mean()) / (logfreq.std() + 1e-8)
        self.register_buffer('log_frequency', torch.as_tensor(logfreq), persistent=False)
        self.register_buffer('cold', torch.as_tensor(np.asarray(frequency) == 0), persistent=False)
        self.item_projection = nn.Linear(features.shape[1], dim, bias=False)
        self.query_encoder = nn.Sequential(nn.LayerNorm(2 * dim + 4), nn.Linear(2 * dim + 4, hidden),
                                           nn.GELU(), nn.Linear(hidden, dim))
        self.pop_weight = nn.Parameter(torch.tensor(0.0))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def optimizers(self, lr, regularization):
        return (torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=regularization),)

    def vectors(self, ids):
        projected = self.item_projection(self.features[ids])
        return F.normalize(projected, dim=-1, eps=1e-8), projected.square().mean()

    def query(self, vectors, seeds, mask):
        count = mask.sum(1, keepdim=True).clamp_min(1)
        mean = (vectors * mask[..., None]).sum(1) / count
        variance = ((vectors - mean[:, None]).square() * mask[..., None]).sum(1) / count
        logfreq = self.log_frequency[seeds]
        stats = torch.cat((torch.log1p(count), (logfreq * mask).sum(1, keepdim=True) / count,
                           logfreq.masked_fill(~mask.bool(), float('inf')).min(1, keepdim=True).values,
                           (self.cold[seeds] * mask).sum(1, keepdim=True) / count), 1)
        residual = self.query_encoder(torch.cat((mean, variance, stats), 1))
        return F.normalize(mean + .1 * residual, dim=-1, eps=1e-8)

    def loss(self, seeds, mask, candidates, regularization):
        vectors, seed_penalty = self.vectors(seeds)
        candidate_vectors, candidate_penalty = self.vectors(candidates)
        query = self.query(vectors, seeds, mask)
        scores = torch.einsum('bd,bkd->bk', query, candidate_vectors)
        scores = scores * self.log_scale.clamp(0, math.log(100)).exp()
        scores = scores + self.pop_weight * self.log_frequency[candidates]
        return (F.cross_entropy(scores, torch.zeros(len(seeds), dtype=torch.long))
                + regularization * (seed_penalty + candidate_penalty))

    @torch.no_grad()
    def catalogue(self, chunk=4096):
        self.eval()
        return torch.cat([self.vectors(torch.arange(start, min(self.n, start + chunk)))[0]
                          for start in range(0, self.n, chunk)])

    @torch.no_grad()
    def score(self, seed_lists, catalogue):
        self.eval()
        seeds, mask = pad_seeds(seed_lists)
        query = self.query(catalogue[seeds], seeds, mask)
        return (catalogue @ query.T * self.log_scale.clamp(0, math.log(100)).exp()
                + self.pop_weight * self.log_frequency[:, None])
