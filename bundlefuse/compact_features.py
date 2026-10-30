"""Fixed raw and engineered feature packs for the compact-model go/no-go study.

The shared 97 columns are identical (content 64, raw taxonomy 32, log-frequency 1).
The remaining 159 columns hold additional raw projections or engineered features. Random maps use fixed seeds;
collaborative SVD and ontology/bridge construction depend on training interactions only.
Metadata embeddings/catalogue taxonomy retain the existing transductive assumption.
"""
import hashlib
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)


def project(values, width, seed):
    rng = np.random.default_rng(seed)
    mapping = rng.choice(np.array([-1., 1.], np.float32), size=(values.shape[1], width)) / np.sqrt(width)
    return normalize(values @ mapping.astype(np.float32))


def taxonomy_vectors(categories, width=32):
    result = np.zeros((len(categories[0]), width), np.float32)
    for j, values in enumerate(categories):
        _, labels = np.unique(values, return_inverse=True)
        rng = np.random.default_rng(50 + j)
        prototypes = rng.standard_normal((labels.max() + 1, width)).astype(np.float32)
        result += normalize(prototypes)[labels]
    return normalize(result)


def build_features(content, categories, frequency, graph, ontology, features, music):
    started = time.perf_counter()
    n = len(frequency)
    raw = np.zeros((n, 256), np.float32)
    raw[:, :64] = project(normalize(content), 64, 10)
    raw[:, 64:96] = taxonomy_vectors(categories)
    logfreq = np.log1p(frequency).astype(np.float32)
    raw[:, 96] = (logfreq - logfreq.mean()) / (logfreq.std() + 1e-8)
    raw[:, 97:225] = project(normalize(content), 128, 14)
    raw[:, 225:256] = taxonomy_vectors(categories, width=31)
    engineered = raw.copy()
    engineered[:, 97:] = 0
    # Keep both directions of this asymmetric degree-normalized operator. This is a
    # deliberately lossy rank-32 approximation, not an exact replacement for its scores.
    svd = TruncatedSVD(n_components=min(32, n - 2), random_state=0)
    if graph.W_pair.nnz:
        left = svd.fit_transform(graph.W_pair).astype(np.float32)
        right = svd.components_.T.astype(np.float32)
        engineered[:, 97:97 + left.shape[1]] = normalize(left)
        engineered[:, 129:129 + right.shape[1]] = normalize(right)
        explained = float(svd.explained_variance_ratio_.sum())
    else:
        explained = 0.0
    style_keys = ['audiostyle'] if music else [k for k in ('style', 'imgstyle', 'descstyle') if k in features]
    style = normalize(np.concatenate([features[k] for k in style_keys], 1))
    engineered[:, 161:193] = project(style, 32, 11)
    engineered[:, 193:225] = project(features['bridge'], 32, 12)
    # Project the train-derived category-PMI rows, rather than learning category-ID tables.
    category_features = project(ontology.Pcat, 31, 13)
    engineered[:, 225:256] = category_features[categories[0]]
    packs = {'Raw': raw, 'Engineered': engineered}
    record = {'input_dimensions': 256, 'shared_raw_columns': 97,
              'shared_blocks': {'raw_content': [0, 64], 'raw_taxonomy': [64, 96], 'log_frequency': [96, 97]},
              'raw_additional_blocks': {'extra_content': [97, 225], 'extra_taxonomy': [225, 256]},
              'engineered_blocks': {
                         'pair_svd_left': [97, 129], 'pair_svd_right': [129, 161],
                         'style': [161, 193], 'bridge': [193, 225], 'ontology': [225, 256]},
              'pair_svd_explained_variance_ratio': explained,
              'seconds': time.perf_counter() - started,
              'sha256': {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in packs.items()},
              'frozen_input_bytes': {key: value.nbytes for key, value in packs.items()},
              'limitation': 'Fixed 256-column input budget; engineering may lose information under compression. No claim about every feature representation or rank.'}
    return packs, record
