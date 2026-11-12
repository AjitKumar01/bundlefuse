"""Full-catalogue ranking with the project's common metric and tie policy."""
import time
import numpy as np
import torch
from bundlefuse.metrics import ranking_metrics, top_items
from .data import pad_seeds


def summarize(columns, pairs):
    records = []
    for score, pair in zip(columns, pairs):
        metric = ranking_metrics(np.asarray(score)[:, None], [pair], Ks=(20,))[20]
        records.append(dict(recall20=float(metric[0]), ndcg20=float(metric[1]),
                            top20=top_items(score, pair[0], 20).tolist()))
    return dict(recall20=float(np.mean([r['recall20'] for r in records])),
                ndcg20=float(np.mean([r['ndcg20'] for r in records])), queries=records)


@torch.inference_mode()
def evaluate(model, pairs, batch_size=8, candidate_chunk=256):
    model.eval()
    started = time.monotonic()
    item_table = model.encode_items(torch.arange(model.n+1))
    encoding_seconds = time.monotonic()-started
    all_scores = []
    for start in range(0, len(pairs), batch_size):
        rows = pairs[start:start+batch_size]
        ids, valid = pad_seeds([s for s, _ in rows], model.n)
        query = model.encode_query(ids, valid, item_table)
        chunks = []
        for c in range(0, model.n, candidate_chunk):
            candidates = torch.arange(c, min(c+candidate_chunk, model.n)).expand(len(rows), -1)
            chunks.append(model.score_candidates(query, candidates, item_table))
        scores = torch.cat(chunks, 1).numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError('Nonfinite evaluation scores')
        all_scores.extend(scores)
    result = summarize(all_scores, pairs)
    result.update(seconds=time.monotonic()-started, item_encoding_seconds=encoding_seconds,
                  full_catalogue=True, catalogue_size=model.n)
    return result


def evaluate_classical(model, pairs, users=None):
    columns = []
    started = time.monotonic()
    for i, (seeds, _) in enumerate(pairs):
        columns.append(np.asarray(model.score(seeds, user=None if users is None else users[i])))
    result = summarize(columns, pairs)
    result.update(seconds=time.monotonic()-started, full_catalogue=True)
    return result
