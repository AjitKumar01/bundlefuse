"""Candidate-conditioned set scorer and two controlled neural ablations."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    variant: str = 'contextual'
    dim: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = .1
    identity_dropout: float = .1


class BundleNetwork(nn.Module):
    """All inputs are raw catalogue attributes, training frequencies and item IDs.

    No count matrix, source score, BundleFuse checkpoint or teacher enters forward().
    Padding uses catalogue index n; padding and unseen identities use one unknown ID.
    """
    def __init__(self, features, categories, frequency, config=ModelConfig()):
        super().__init__()
        if config.variant not in ('dot', 'relations', 'contextual'):
            raise ValueError(config.variant)
        self.config = config
        self.n = len(frequency)
        d = config.dim
        features = F.normalize(torch.as_tensor(features, dtype=torch.float32), dim=1)
        cats = torch.as_tensor(categories, dtype=torch.long)
        freq = torch.as_tensor(frequency, dtype=torch.float32)
        self.register_buffer('features', F.pad(features, (0, 0, 0, 1)))
        self.register_buffer('categories', F.pad(cats, (0, 1), value=int(cats.max()) + 1))
        self.register_buffer('frequency', F.pad(freq, (0, 1)))
        self.identity = nn.Embedding(self.n + 1, d)
        nn.init.normal_(self.identity.weight, std=.02)
        self.category = nn.Embedding(int(cats.max()) + 2, 16)
        self.content = nn.Linear(features.shape[1], d)
        self.item = nn.Sequential(nn.Linear(2*d + 17, d), nn.GELU(), nn.LayerNorm(d))
        if config.variant != 'relations':
            block = nn.TransformerEncoderLayer(d, config.heads, 2*d,
                        dropout=config.dropout, activation='gelu', batch_first=True)
            self.context = nn.TransformerEncoder(block, config.layers, enable_nested_tensor=False)
        if config.variant == 'dot':
            self.query_projection = nn.Linear(d, d)
            self.candidate_projection = nn.Linear(d, d)
        else:
            self.relation = nn.Sequential(nn.Linear(4*d, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
            self.decoder = nn.Sequential(nn.Linear(3*d + 2, d), nn.GELU(), nn.Linear(d, 1))

    def encode_items(self, ids):
        known = self.frequency[ids] > 0
        if self.training and self.config.identity_dropout:
            known = known & (torch.rand(ids.shape, device=ids.device) >= self.config.identity_dropout)
        identity_ids = torch.where(known, ids, self.n)
        return self.item(torch.cat((self.identity(identity_ids), self.content(self.features[ids]),
                         self.category(self.categories[ids]),
                         torch.log1p(self.frequency[ids]).unsqueeze(-1)), dim=-1))

    def encode_query(self, seeds, valid, item_table=None):
        if not bool(valid.any(1).all()):
            raise ValueError('Each query needs at least one observed item')
        raw = self.encode_items(seeds) if item_table is None else item_table[seeds]
        contextual = raw if self.config.variant == 'relations' else self.context(raw, src_key_padding_mask=~valid)
        return raw, contextual, valid

    def score_candidates(self, query, candidates, item_table=None):
        raw, contextual, valid = query
        candidate = self.encode_items(candidates) if item_table is None else item_table[candidates]
        if self.config.variant == 'dot':
            pooled = (contextual * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            return (self.query_projection(pooled).unsqueeze(1) *
                    self.candidate_projection(candidate)).sum(-1) / self.config.dim**.5
        b, k, d = candidate.shape
        length = raw.shape[1]
        c = candidate.unsqueeze(2).expand(b, k, length, d)
        r = raw.unsqueeze(1).expand(b, k, length, d)
        h = contextual.unsqueeze(1).expand(b, k, length, d)
        messages = self.relation(torch.cat((c, r, h, c*h), dim=-1))
        mask = valid[:, None, :, None]
        summed = (messages * mask).sum(2)
        maximum = messages.masked_fill(~mask, -torch.inf).amax(2)
        size = torch.log1p(valid.sum(1).to(candidate.dtype))[:, None, None].expand(b, k, 1)
        freq = torch.log1p(self.frequency[candidates]).unsqueeze(-1)
        return self.decoder(torch.cat((candidate, summed, maximum, size, freq), dim=-1)).squeeze(-1)

    def forward(self, seeds, valid, candidates):
        return self.score_candidates(self.encode_query(seeds, valid), candidates)

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ids = self.identity.weight.numel()
        return dict(total=total, item_identity=ids, other=total-ids,
                    frozen_buffer_bytes=sum(b.numel()*b.element_size() for b in self.buffers()),
                    config=asdict(self.config))
