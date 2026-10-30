"""Warm-cache scoring and ranking timings; report preprocessing and worker peak RSS separately."""
import resource
import statistics
import sys
import time

import numpy as np

from .metrics import ranking_metrics


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def profile_scorers(scorers, pairs, repeats=3, query_count=32):
    indices = np.random.default_rng(0).permutation(len(pairs))[:query_count]
    queries = [pairs[i] for i in indices]
    seeds = [s for s, _ in queries]
    report = {}
    for name, scorer in scorers.items():
        scorer(seeds)  # Explicit warmup; catalogue vectors / graph are already built.
        scoring, ranking = [], []
        for _ in range(repeats):
            start = time.perf_counter(); scores = scorer(seeds)
            end = time.perf_counter()
            ranking_metrics(scores, queries, Ks=(10, 20, 50))
            finish = time.perf_counter()
            scoring.append(1000 * (end - start) / len(queries))
            ranking.append(1000 * (finish - start) / len(queries))
        report[name] = {'scoring_ms_per_query': scoring, 'scoring_plus_metrics_ms_per_query': ranking,
                        'median_scoring_ms_per_query': statistics.median(scoring),
                        'median_scoring_plus_metrics_ms_per_query': statistics.median(ranking),
                        'queries': len(queries), 'repeats': repeats, 'query_indices': indices.tolist()}
    return report
