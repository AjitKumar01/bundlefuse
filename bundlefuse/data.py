"""Dataset loaders for BundleFuse -- thin, dataset-specific I/O that feeds the model class.

``load_dataset(name)`` returns a :class:`Dataset` with everything ``BundleFuse.fit`` needs:
train bundles, validation / test (seeds, ground-truth) pairs, and the optional content array,
category labels and per-user histories that switch the content / taxonomy / personalization
families on. All modelling lives in :mod:`bundlefuse.model`; this module only reads files.

Supported names:
  bundlemllm  : pog, pog_dense, spotify, spotify_sparse   (provided completion splits; no users)
  dunnhumby   : dunnhumby                                  (grocery baskets; households as users)
  BundleRec   : electronic, food, clothing                (user_bundle + item_titles + pretrain)
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp

from bundlefuse.paths import DATA_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse import protocol

BUNDLEMLLM = {"pog", "pog_dense", "spotify", "spotify_sparse"}
BUNDLEREC = {"electronic", "food", "clothing"}


@dataclass
class Dataset:
    name: str
    n_items: int
    train: list
    val_pairs: list
    test_pairs: list
    families: list
    content: Optional[np.ndarray] = None
    cat_labels: Optional[np.ndarray] = None
    user_items: Optional[dict] = None
    val_users: Optional[list] = None
    test_users: Optional[list] = None


def load_dataset(name, *, val_n=800, test_n=2000, mask_frac=0.5, split_seed=0):
    if name in BUNDLEMLLM:
        return _load_bundlemllm(name, val_n, test_n)
    if name == "dunnhumby":
        return _load_dunnhumby(val_n, test_n, mask_frac, split_seed)
    if name in BUNDLEREC:
        return _load_bundlerec(name, val_n, test_n, mask_frac, split_seed)
    raise ValueError(f"unknown dataset {name!r}")


def _tfidf_svd(texts, dim=128):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import TruncatedSVD
    tf = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    tf = normalize(tf, norm="l2", axis=1)
    d = min(dim, tf.shape[1] - 1)
    return TruncatedSVD(d, random_state=0).fit_transform(tf).astype(np.float32)


def _kmeans(feat, k=400):
    from sklearn.cluster import MiniBatchKMeans
    return MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=4096).fit(feat).labels_.astype(int)


# ---------------------------------------------------------------- bundlemllm
def _load_bundlemllm(name, val_n, test_n):
    import torch
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", name)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    sp_ = ProvidedSplits(dd, n)
    info = json.load(open(os.path.join(dd, "item_info.json")))
    music = name in ("spotify", "spotify_sparse") or info["0"].get("artist_uri") is not None

    def _pt(f):
        return torch.load(os.path.join(dd, f), map_location="cpu", weights_only=False).detach().float().numpy()

    parts = []
    if music:
        if os.path.exists(os.path.join(dd, "content_feature.pt")):
            parts.append(_pt("content_feature.pt"))
    else:
        if os.path.exists(os.path.join(dd, "title_emb.npy")):
            parts.append(np.load(os.path.join(dd, "title_emb.npy")).astype(np.float32))
        for f in ("content_feature.pt", "description_feature.pt"):
            if os.path.exists(os.path.join(dd, f)):
                parts.append(_pt(f))
    content = None
    if parts:
        parts = [p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8) for p in parts]
        content = np.concatenate(parts, axis=1).astype(np.float32)

    # taxonomy: explicit category if present, else cluster the content
    cat = None
    cval = info["0"].get("cate", info["0"].get("cate_id"))
    if cval is not None:
        cmap = {}; cat = np.zeros(n, int)
        for k, v in info.items():
            c = v.get("cate", v.get("cate_id"))
            cat[int(k)] = cmap.setdefault(c, len(cmap))
    elif content is not None:
        cat = _kmeans(content)

    fams = ["collaborative", "popularity"] + (["content"] if content is not None else []) + (["taxonomy"] if cat is not None else [])
    return Dataset(name, n, list(sp_.train.values()), sp_.pairs("valid")[:val_n], sp_.pairs("test")[:test_n],
                   fams, content=content, cat_labels=cat)


# ---------------------------------------------------------------- dunnhumby
def _load_dunnhumby(val_n, test_n, mask_frac, split_seed, basket_cap=30000):
    root = os.environ.get("DUNN_DIR") or os.path.join(DATA_ROOT, "dunnhumby")
    raw = defaultdict(set); buser = {}
    with open(os.path.join(root, "transaction_data.csv")) as f:
        r = csv.reader(f); h = next(r); bi, pi, hi = h.index("BASKET_ID"), h.index("PRODUCT_ID"), h.index("household_key")
        for row in r:
            raw[row[bi]].add(int(row[pi])); buser[row[bi]] = row[hi]
    bask = [(sorted(v), buser[b]) for b, v in raw.items() if 3 <= len(v) <= 20]
    random.Random(0).shuffle(bask); bask = bask[:basket_cap]
    fa = defaultdict(int)
    for b, _ in bask:
        for it in b:
            fa[it] += 1
    keep = {it for it, c in fa.items() if c >= 2}
    tmp = [([it for it in b if it in keep], u) for b, u in bask]
    tmp = [(b, u) for b, u in tmp if len(b) >= 3]
    items = sorted({i for b, _ in tmp for i in b}); imap = {it: k for k, it in enumerate(items)}; n = len(items)
    umap = {}; seen = set(); bundles = []; busers = []
    for b, u in tmp:
        mp = tuple(sorted(imap[i] for i in set(b))); uu = umap.setdefault(u, len(umap))
        if (uu, mp) not in seen:
            seen.add((uu, mp)); bundles.append(list(mp)); busers.append(uu)

    # taxonomy + content from product.csv
    cat = np.full(n, -1, int); cmap = {}; txt = ["" for _ in range(n)]
    with open(os.path.join(root, "product.csv")) as f:
        r = csv.reader(f); h = next(r); ip = h.index("PRODUCT_ID")
        dep, com, sub, br = h.index("DEPARTMENT"), h.index("COMMODITY_DESC"), h.index("SUB_COMMODITY_DESC"), h.index("BRAND")
        for row in r:
            it = int(row[ip])
            if it in imap:
                j = imap[it]; cat[j] = cmap.setdefault(row[sub].strip() or "?", len(cmap))
                txt[j] = f"{row[dep]} {row[com]} {row[sub]} {row[br]}"
    cat[cat < 0] = len(cmap)
    content = _tfidf_svd(txt)

    idx = list(range(len(bundles))); random.Random(split_seed).shuffle(idx)
    ntr, nva = int(0.7 * len(idx)), int(0.1 * len(idx))
    tri, vai, tei = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
    train = [bundles[i] for i in tri]
    user_items = defaultdict(list)
    for i in tri:
        user_items[busers[i]] += bundles[i]
    vp = protocol.mask([bundles[i] for i in vai], split_seed + 1, mask_frac)[:val_n]
    tp = protocol.mask([bundles[i] for i in tei], split_seed + 2, mask_frac)[:test_n]
    vu = [busers[i] for i in vai][:val_n]; tu = [busers[i] for i in tei][:test_n]
    fams = ["collaborative", "content", "taxonomy", "popularity", "personalization"]
    return Dataset("dunnhumby", n, train, vp, tp, fams, content=content, cat_labels=cat,
                   user_items=dict(user_items), val_users=vu, test_users=tu)


# ---------------------------------------------------------------- BundleRec
def _load_bundlerec(name, val_n, test_n, mask_frac, split_seed):
    root = os.path.join(DATA_ROOT, "BundleRec", "dataset", name)
    raw = defaultdict(list)
    with open(os.path.join(root, "bundle_item.csv")) as f:
        r = csv.reader(f); next(r)
        for b, it in r:
            raw[int(b)].append(int(it))
    items = sorted({i for v in raw.values() for i in v}); imap = {it: k for k, it in enumerate(items)}; n = len(items)
    seen = set(); bkeys, bundles = [], []
    for b, v in raw.items():
        u = list(dict.fromkeys(imap[i] for i in v))
        if len(u) >= 2:
            key = tuple(sorted(u))
            if key not in seen:
                seen.add(key); bkeys.append(b); bundles.append(u)
    b2u = {}
    with open(os.path.join(root, "user_bundle.csv")) as f:
        r = csv.reader(f); next(r)
        for u, b, *_ in r:
            b2u[int(b)] = int(u)

    # content (item titles) + taxonomy (item categories)
    titles = {i: "" for i in range(n)}
    tp = os.path.join(root, "item_titles.csv")
    if os.path.exists(tp):
        with open(tp) as f:
            r = csv.reader(f); next(r)
            for row in r:
                it = int(row[0])
                if it in imap:
                    titles[imap[it]] = " ".join(row[1:])
    content = _tfidf_svd([titles[i] for i in range(n)])
    cat = np.zeros(n, int); cp = os.path.join(root, "item_categories.csv")
    if os.path.exists(cp):
        cmap = {}
        with open(cp) as f:
            r = csv.reader(f); next(r)
            for row in r:
                it = int(row[0])
                if it in imap:
                    cat[imap[it]] = cmap.setdefault(row[1], len(cmap))

    # per-user histories from user_item_pretrain
    user_items = defaultdict(list); up = os.path.join(root, "user_item_pretrain.csv")
    if os.path.exists(up):
        with open(up) as f:
            r = csv.reader(f); next(r)
            for u, it, *_ in r:
                if int(it) in imap:
                    user_items[int(u)].append(imap[int(it)])

    idx = list(range(len(bundles))); random.Random(split_seed).shuffle(idx)
    ntr, nva = int(0.7 * len(idx)), int(0.1 * len(idx))
    vai, tei = idx[ntr:ntr + nva], idx[ntr + nva:]
    train = [bundles[i] for i in idx[:ntr]]
    vp = protocol.mask([bundles[i] for i in vai], split_seed + 1, mask_frac)[:val_n]
    tp_ = protocol.mask([bundles[i] for i in tei], split_seed + 2, mask_frac)[:test_n]
    vu = [b2u.get(bkeys[i], -1) for i in vai][:val_n]; tu = [b2u.get(bkeys[i], -1) for i in tei][:test_n]
    fams = ["collaborative", "content", "taxonomy", "popularity", "personalization"]
    return Dataset(name, n, train, vp, tp_, fams, content=content, cat_labels=cat,
                   user_items=dict(user_items), val_users=vu, test_users=tu)
