"""Train-fold-only hybrid two-tower ranker using raw metadata and collaborative item IDs."""
import math
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from ..artifacts import write_json


GRID = tuple({'dim': 64, 'lr': lr, 'regularization': reg, 'negatives': 32,
              'batch_size': 128, 'max_epochs': 100, 'evaluate_every': 5, 'patience': 4}
             for lr in (.001, .003) for reg in (1e-5, 1e-3))


class MetadataRanker(nn.Module):
    def __init__(self, content, categories, frequency, dim=64):
        super().__init__()
        self.n = len(frequency)
        self.dim = dim
        self.sparse_content = content.tocsr() if sp.issparse(content) else None
        if self.sparse_content is None:
            self.register_buffer('content', torch.as_tensor(content, dtype=torch.float32), persistent=False)
            self.content_encoder = nn.Linear(content.shape[1], dim, bias=False)
        else:
            self.content_encoder = nn.EmbeddingBag(content.shape[1], dim, mode='sum', sparse=True,
                                                    include_last_offset=True)
            nn.init.normal_(self.content_encoder.weight, std=.02)
        self.category_embeddings = nn.ModuleList()
        for j, values in enumerate(categories):
            _, labels = np.unique(np.asarray(values), return_inverse=True)
            self.register_buffer(f'category_{j}', torch.as_tensor(labels), persistent=False)
            embedding = nn.Embedding(int(labels.max()) + 1, dim, sparse=True)
            nn.init.normal_(embedding.weight, std=.02)
            self.category_embeddings.append(embedding)
        self.item_embedding = nn.Embedding(self.n, dim + 1, sparse=True)
        nn.init.normal_(self.item_embedding.weight, std=.01)
        with torch.no_grad():
            self.item_embedding.weight[:, -1] = 0
        self.register_buffer('warm', torch.as_tensor(np.asarray(frequency) > 0, dtype=torch.float32), persistent=False)
        logfreq = np.log1p(frequency).astype(np.float32)
        logfreq = (logfreq - logfreq.mean()) / (logfreq.std() + 1e-8)
        self.register_buffer('log_frequency', torch.as_tensor(logfreq), persistent=False)
        self.pop_weight = nn.Parameter(torch.tensor(0.0))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.query_encoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def optimizers(self, lr, regularization):
        sparse = [self.item_embedding.weight] + [embedding.weight for embedding in self.category_embeddings]
        if self.sparse_content is not None:
            sparse.append(self.content_encoder.weight)
        sparse_ids = {id(value) for value in sparse}
        dense = [value for value in self.parameters() if id(value) not in sparse_ids]
        return torch.optim.AdamW(dense, lr=lr, weight_decay=regularization), torch.optim.SparseAdam(sparse, lr=lr)

    def item_vectors(self, items):
        original_shape = items.shape
        unique, inverse = torch.unique(items.reshape(-1), sorted=True, return_inverse=True)
        if self.sparse_content is None:
            vector = self.content_encoder(self.content[unique])
        else:
            rows = self.sparse_content[unique.numpy()].tocsr()
            vector = self.content_encoder(torch.as_tensor(rows.indices, dtype=torch.long),
                                          torch.as_tensor(rows.indptr, dtype=torch.long),
                                          per_sample_weights=torch.as_tensor(rows.data, dtype=torch.float32))
        penalty = vector.square().mean()
        for j, embedding in enumerate(self.category_embeddings):
            categorical = embedding(getattr(self, f'category_{j}')[unique])
            vector = vector + categorical
            penalty = penalty + categorical.square().mean()
        identity = self.item_embedding(unique) * self.warm[unique, None]
        vector = F.normalize(vector + identity[:, :self.dim], dim=-1, eps=1e-8)
        bias = identity[:, -1] + self.pop_weight * self.log_frequency[unique]
        penalty = penalty + identity.square().mean()
        return (vector[inverse].reshape(*original_shape, self.dim), bias[inverse].reshape(original_shape), penalty)

    def query(self, vectors, mask):
        mean = (vectors * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(mean + .1 * self.query_encoder(mean), dim=-1, eps=1e-8)

    def loss(self, seeds, mask, candidates, regularization):
        encoded, _, seed_penalty = self.item_vectors(seeds[mask.bool()])
        seed_vectors = torch.zeros((*seeds.shape, self.dim), dtype=encoded.dtype)
        seed_vectors[mask.bool()] = encoded
        candidate_vectors, bias, candidate_penalty = self.item_vectors(candidates)
        query = self.query(seed_vectors, mask)
        logits = torch.einsum('bd,bkd->bk', query, candidate_vectors) * self.log_scale.clamp(0, math.log(100)).exp() + bias
        target = torch.zeros(len(seeds), dtype=torch.long)
        return F.cross_entropy(logits, target) + regularization * (seed_penalty + candidate_penalty)

    @torch.no_grad()
    def catalogue(self, chunk=4096):
        self.eval()
        vectors, biases = [], []
        for start in range(0, self.n, chunk):
            vector, bias, _ = self.item_vectors(torch.arange(start, min(self.n, start + chunk)))
            vectors.append(vector); biases.append(bias)
        return torch.cat(vectors), torch.cat(biases)

    @torch.no_grad()
    def score(self, seed_lists, catalogue):
        self.eval()
        vectors, bias = catalogue
        seeds, mask = pad_seeds(seed_lists)
        query = self.query(vectors[seeds], mask)
        return vectors @ query.T * self.log_scale.clamp(0, math.log(100)).exp() + bias[:, None]


def pad_seeds(seed_lists):
    if not seed_lists or any(not seeds for seeds in seed_lists):
        raise ValueError('Every query must have a revealed item')
    items = torch.zeros((len(seed_lists), max(map(len, seed_lists))), dtype=torch.long)
    mask = torch.zeros_like(items, dtype=torch.float32)
    for j, seeds in enumerate(seed_lists):
        items[j, :len(seeds)] = torch.as_tensor(seeds)
        mask[j, :len(seeds)] = 1
    return items, mask


def training_batch(bundles, n_items, rng, negatives, exclusions=None):
    seed_lists, candidates = [], []
    for index, bundle in enumerate(bundles):
        items = np.asarray(bundle, dtype=np.int64).copy()
        forbidden = np.asarray(bundle if exclusions is None else exclusions[index], dtype=np.int64)
        if len(items) < 2 or len(set(forbidden)) >= n_items:
            raise ValueError('Training bundles need at least two items and an available negative')
        rng.shuffle(items)
        split = max(1, min(len(items) - 1, round(len(items) / 2)))
        seed_lists.append(items[:split].tolist())
        positive = int(rng.choice(items[split:]))
        sampled = rng.integers(n_items, size=negatives)
        invalid = np.isin(sampled, forbidden)
        while invalid.any():
            sampled[invalid] = rng.integers(n_items, size=int(invalid.sum()))
            invalid = np.isin(sampled, forbidden)
        candidates.append([positive] + sampled.tolist())
    seeds, mask = pad_seeds(seed_lists)
    return seeds, mask, torch.as_tensor(candidates)


def train_ranker(content, categories, frequency, train_bundles, validate, hp, seed, output,
                 auxiliary_bundles=None, model_factory=None):
    """Validation callback never receives test queries; retain the best validation checkpoint."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = (MetadataRanker(content, categories, frequency, hp['dim']) if model_factory is None
             else model_factory())
    optimizers = model.optimizers(hp['lr'], hp['regularization'])
    train = [list(bundle) for bundle in train_bundles if len(bundle) >= 2]
    auxiliary = [list(bundle) for bundle in (auxiliary_bundles or []) if len(bundle) >= 2]
    if not train:
        raise ValueError('No eligible training bundles')
    best, best_epoch, stale = -float('inf'), 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, hp['max_epochs'] + 1):
        model.train()
        losses = []
        order = rng.permutation(len(train))
        for start in range(0, len(order), hp['batch_size']):
            bundles = [train[i] for i in order[start:start + hp['batch_size']]]
            exclusions = list(bundles)
            if auxiliary:
                for j in range(len(bundles)):
                    if rng.random() < .2:
                        exclusions[j] = auxiliary[int(rng.integers(len(auxiliary)))]
                        bundles[j] = rng.choice(exclusions[j], size=2, replace=False).tolist()
            seeds, mask, candidates = training_batch(bundles, len(frequency), rng, hp['negatives'], exclusions)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss = model.loss(seeds, mask, candidates, hp['regularization'])
            if not torch.isfinite(loss):
                raise RuntimeError('Learned-ranker training produced a nonfinite loss')
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            losses.append(float(loss.detach()))
        record = {'epoch': epoch, 'train_loss': float(np.mean(losses))}
        if epoch % hp['evaluate_every'] == 0 or epoch == hp['max_epochs']:
            value = float(validate(model))
            if not np.isfinite(value):
                raise RuntimeError('Learned-ranker validation objective is nonfinite')
            record['val_ndcg20'] = value
            if value > best + 1e-5:
                best, best_epoch, stale = value, epoch, 0
                torch.save(model.state_dict(), output / 'best.pt')
            else:
                stale += 1
            print(f'LEARNED seed={seed} epoch={epoch} loss={record["train_loss"]:.6f} val_N20={value:.6f} best={best:.6f}', flush=True)
        history.append(record)
        write_json(output / 'history.json', history)
        if stale >= hp['patience']:
            break
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=True))
    model.eval()
    result = {'hp': hp, 'seed': seed, 'best_epoch': best_epoch, 'val_ndcg20': best,
              'epochs_run': epoch, 'stopped': 'patience' if stale >= hp['patience'] else 'epoch_budget',
              'best_at_final_check': best_epoch == epoch, 'training_bundles': len(train),
              'auxiliary_bundles': len(auxiliary), 'parameter_count': sum(p.numel() for p in model.parameters()),
              'seconds': time.perf_counter() - started}
    write_json(output / 'training.json', result)
    return model, result
