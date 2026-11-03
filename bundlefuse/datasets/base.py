"""Shared dataset loaders for BundleFuse -- thin, dataset-family I/O that feeds the model class.

Each per-dataset module (``bundlefuse/datasets/<name>.py``) calls one of these family loaders and
returns a :class:`Dataset` with everything ``BundleFuse.fit`` needs: train bundles, validation / test
(seeds, ground-truth) pairs, and the optional content array, category labels and per-user histories
that switch the content / taxonomy / personalization families on. All modelling lives in
:mod:`bundlefuse.model`; this module only reads files.

Family loaders:
  load_bundlemllm : pog, pog_dense, spotify, spotify_sparse   (provided completion splits; no users)
  load_dunnhumby  : dunnhumby                                  (grocery baskets; households as users)
  load_bundlerec  : electronic, food, clothing                (user_bundle + item_titles + pretrain)
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
from bundlefuse.provenance import record_dataset

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
    features: Optional[dict] = None          # content modalities, e.g. {"content":title,"image":img,"desc":desc}
    cat_labels: Optional[np.ndarray] = None
    music: bool = False
    alb_labels: Optional[np.ndarray] = None
    cluster_feat: Optional[np.ndarray] = None
    aux_incidences: Optional[dict] = None
    group_labels: Optional[dict] = None
    user_items: Optional[dict] = None
    val_users: Optional[list] = None
    test_users: Optional[list] = None
    metadata: dict = field(default_factory=dict)
    val_ids: list = field(default_factory=list)
    test_ids: list = field(default_factory=list)

    def fit_kwargs(self):
        return {**{key: getattr(self, key) for key in ('features', 'cat_labels', 'music', 'alb_labels',
                'cluster_feat', 'aux_incidences', 'group_labels', 'user_items', 'val_users')},
                'data_spec': self.metadata}


def _limits(val_n, test_n, mask_frac):
    if any(value is not None and value < 0 for value in (val_n, test_n)):
        raise ValueError('Query limits must be nonnegative; 0/None means the full split')
    if not 0 < mask_frac < 1:
        raise ValueError('Mask fraction must be between zero and one')


def _limit(rows, count):
    return rows[:count] if count else rows


def _tfidf_svd(texts, dim=128):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import TruncatedSVD
    tf = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    tf = normalize(tf, norm="l2", axis=1)
    d = min(dim, tf.shape[1] - 1)
    return TruncatedSVD(d, random_state=0).fit_transform(tf).astype(np.float32)


def _user_incidence(user_items, n):
    """Degree-normalised (users x items) incidence for an auxiliary co-occurrence signal."""
    from collections import Counter
    nu = (max(user_items) + 1) if user_items else 1
    r, c, d = [], [], []
    for u, its in user_items.items():
        for it, cnt in Counter(its).items():
            r.append(u); c.append(it); d.append(float(cnt))
    X = sp.csr_matrix((d, (r, c)), shape=(nu, n))
    deg = np.asarray(X.sum(1)).ravel() + 1.0
    return (sp.diags(1.0 / np.sqrt(deg)) @ X).tocsr()


def _kmeans(feat, k=400):
    from sklearn.cluster import MiniBatchKMeans
    return MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=4096).fit(feat).labels_.astype(int)


# ---------------------------------------------------------------- bundlemllm
def load_bundlemllm(name, val_n=None, test_n=None, mask_frac=0.5, split_seed=0, protocol_mode="provided"):
    _limits(val_n, test_n, mask_frac)
    if protocol_mode not in ('provided', 'resplit'):
        raise ValueError('Protocol must be provided or resplit')
    if protocol_mode == 'provided' and (split_seed != 0 or mask_frac != .5):
        raise ValueError('Provided splits cannot be resplit or remasked; use protocol_mode="resplit"')
    import torch
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", name)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    sp_ = ProvidedSplits(dd, n)
    info = json.load(open(os.path.join(dd, "item_info.json")))
    music = name in ("spotify", "spotify_sparse") or info["0"].get("artist_uri") is not None

    def _pt(f):
        return torch.load(os.path.join(dd, f), map_location="cpu", weights_only=False).detach().float().numpy()

    # per-modality content features (kept separate so the model disentangles each, as the research build)
    feats = {}
    if music:
        if os.path.exists(os.path.join(dd, "content_feature.pt")):
            feats["audio"] = _pt("content_feature.pt")
    else:
        if os.path.exists(os.path.join(dd, "title_emb.npy")):
            feats["content"] = np.load(os.path.join(dd, "title_emb.npy")).astype(np.float32)
        if os.path.exists(os.path.join(dd, "content_feature.pt")):
            feats["image"] = _pt("content_feature.pt")
        if os.path.exists(os.path.join(dd, "description_feature.pt")):
            feats["desc"] = _pt("description_feature.pt")
    feats = {k: (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)).astype(np.float32) for k, v in feats.items()}
    cluster_feat = np.concatenate(list(feats.values()), 1).astype(np.float32) if feats else None

    # taxonomy labels: explicit category (fashion) or artist indices (music); album labels for homophily
    cat = None; alb = None
    cval = info["0"].get("cate", info["0"].get("cate_id"))
    if (not music) and cval is not None:
        cmap = {}; cat = np.zeros(n, int)
        for k, v in info.items():
            cat[int(k)] = cmap.setdefault(v.get("cate", v.get("cate_id")), len(cmap))
    elif music and info["0"].get("artist_uri") is not None:
        am = {}; cat = np.zeros(n, int); alm = {}; alb = np.zeros(n, int)
        for k, v in info.items():
            cat[int(k)] = am.setdefault(v.get("artist_uri", "?"), len(am))
            alb[int(k)] = alm.setdefault(v.get("album_uri", "?"), len(alm))

    fams = ["collaborative", "popularity", "content", "taxonomy"]
    if protocol_mode == "provided":
        train = list(sp_.train.values()); full_vp = sp_.pairs('valid'); full_tp = sp_.pairs('test')
        vp = _limit(full_vp, val_n); tp = _limit(full_tp, test_n)
        vi = _limit(sorted(sp_.splits['valid'][0]), val_n); ti = _limit(sorted(sp_.splits['test'][0]), test_n)
    else:
        # the research matrix protocol: reconstruct all bundles, resplit 70/10/20 by seed, reveal-mask
        fullb = list(sp_.train.values())
        for inp, gt in sp_.pairs("valid") + sp_.pairs("test"):
            fullb.append(list(inp) + list(gt))
        seen = set(); allb = []
        for v in fullb:
            u = list(dict.fromkeys(v))
            if len(u) >= 2 and tuple(sorted(u)) not in seen:
                seen.add(tuple(sorted(u))); allb.append(u)
        random.Random(split_seed).shuffle(allb)
        ntr, nva = int(0.7 * len(allb)), int(0.1 * len(allb))
        train, vab, teb = allb[:ntr], allb[ntr:ntr + nva], allb[ntr + nva:]
        full_vp = protocol.mask(vab, split_seed + 1, mask_frac)
        full_tp = protocol.mask(teb, split_seed + 2, mask_frac)
        vp, tp = _limit(full_vp, val_n), _limit(full_tp, test_n)
        vi, ti = list(range(len(vp))), list(range(len(tp)))
    ds = Dataset(name, n, train, vp, tp, fams, features=feats, cat_labels=cat,
                 music=music, alb_labels=alb, cluster_feat=cluster_feat)
    files = list(sp_.hashes) + ['item_info.json', 'title_emb.npy', 'content_feature.pt', 'description_feature.pt']
    return record_dataset(ds, dd, files, protocol=protocol_mode, split_seed=split_seed, mask_frac=mask_frac,
                          val_limit=val_n, test_limit=test_n, val_total=len(full_vp), test_total=len(full_tp),
                          val_ids=vi, test_ids=ti)


# ---------------------------------------------------------------- dunnhumby
def load_dunnhumby(val_n, test_n, mask_frac, split_seed, basket_cap=30000, protocol_mode='resplit'):
    _limits(val_n, test_n, mask_frac)
    if protocol_mode != 'resplit':
        raise ValueError('This retail loader supports resplit only; it is not the historical chronological study')
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
    vp = _limit(protocol.mask([bundles[i] for i in vai], split_seed + 1, mask_frac), val_n)
    tp = _limit(protocol.mask([bundles[i] for i in tei], split_seed + 2, mask_frac), test_n)
    vu = _limit([busers[i] for i in vai], val_n); tu = _limit([busers[i] for i in tei], test_n)
    fams = ["collaborative", "content", "taxonomy", "popularity", "personalization"]
    aux = {"hh_cooc": (_user_incidence(dict(user_items), n), True)}   # household co-occurrence (degree-weighted)
    ds = Dataset('dunnhumby', n, train, vp, tp, fams, features={'content': content}, cat_labels=cat,
                 aux_incidences=aux, user_items=dict(user_items), val_users=vu, test_users=tu)
    return record_dataset(ds, root, ['transaction_data.csv', 'product.csv'], protocol='resplit',
                          split_seed=split_seed, mask_frac=mask_frac, val_limit=val_n, test_limit=test_n,
                          val_total=len(vai), test_total=len(tei), val_ids=_limit(vai,val_n), test_ids=_limit(tei,test_n))


# ---------------------------------------------------------------- BundleRec
def load_bundlerec(name, val_n, test_n, mask_frac, split_seed, protocol_mode='resplit'):
    _limits(val_n, test_n, mask_frac)
    if protocol_mode != 'resplit':
        raise ValueError('BundleRec completion supports resplit only')
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
    vp = _limit(protocol.mask([bundles[i] for i in vai], split_seed + 1, mask_frac), val_n)
    tp_ = _limit(protocol.mask([bundles[i] for i in tei], split_seed + 2, mask_frac), test_n)
    vu = _limit([b2u.get(bkeys[i], -1) for i in vai], val_n); tu = _limit([b2u.get(bkeys[i], -1) for i in tei], test_n)
    fams = ["collaborative", "content", "taxonomy", "popularity", "personalization"]
    aux = {"pretrain": (_user_incidence(dict(user_items), n), False)}   # user-item pretrain co-occurrence
    ds = Dataset(name, n, train, vp, tp_, fams, features={'content': content}, cat_labels=cat,
                 aux_incidences=aux, user_items=dict(user_items), val_users=vu, test_users=tu)
    return record_dataset(ds, root, ['bundle_item.csv', 'user_bundle.csv', 'item_titles.csv',
                          'item_categories.csv', 'user_item_pretrain.csv'], protocol='resplit',
                          split_seed=split_seed, mask_frac=mask_frac, val_limit=val_n, test_limit=test_n,
                          val_total=len(vai), test_total=len(tei),
                          val_ids=[bkeys[i] for i in _limit(vai,val_n)], test_ids=[bkeys[i] for i in _limit(tei,test_n)])
