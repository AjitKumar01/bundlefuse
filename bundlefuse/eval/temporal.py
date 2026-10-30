"""Chronological fixed-catalogue retail experiment with training-only interaction graphs."""
import os
import time

STARTED = time.perf_counter()

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from bundlefuse._device import DEVICE
from bundlefuse.eval.generalization import run_generalization
from bundlefuse.paths import DATA_ROOT, RESULTS_DIR
from bundlefuse.retail_signals import RetailSignals
from bundlefuse.temporal_data import load_temporal


def main():
    root = os.environ.get('DUNN_DIR') or os.path.join(DATA_ROOT, 'dunnhumby')
    data = load_temporal(root)
    tfidf = TfidfVectorizer(max_features=8000, stop_words='english', ngram_range=(1, 2),
                           sublinear_tf=True).fit_transform(data['text']).tocsr()
    content = TruncatedSVD(min(128, tfidf.shape[1] - 1), random_state=0).fit_transform(tfidf).astype(np.float32)
    content /= np.linalg.norm(content, axis=1, keepdims=True) + 1e-8
    signals = RetailSignals(content, tfidf, data['categories'], data['n_users']).build(data['train'], data['users'])
    print(f'TEMPORAL device={DEVICE} catalogue={len(data["categories"])} protocol={data["protocol"]["day_ranges"]}', flush=True)
    run_generalization('dunnhumby_chrono', data['train'], data['valid'], data['test'],
                       data['test_ids'], data['groups'], data['protocol'], signals.score, signals.keys,
                       signals.graph.freq, signals.hybrid(), tfidf, [data['categories']], RESULTS_DIR,
                       preprocessing_seconds=time.perf_counter() - STARTED,
                       auxiliary=signals.auxiliary_bundles(), chunk=int(os.environ.get('PROVIDED_CHUNK', '64')))


if __name__ == '__main__':
    main()
