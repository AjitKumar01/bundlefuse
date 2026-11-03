"""Retail and commerce data construction for the published experiment recipe.

Extended-profile compatibility loading is dispatched from base.py.
"""
import csv, os, random
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from bundlefuse.paths import DATA_ROOT
from bundlefuse import splits
from bundlefuse.provenance import record_dataset
from bundlefuse.datasets.base import Dataset, _limit

def load_retail(val_n, test_n, mask_frac, split_seed, basket_cap=30000):
    ROOT = os.environ.get("DUNN_DIR") or os.path.join(DATA_ROOT,"dunnhumby")
    BASKET_CAP = basket_cap
    MIN_FREQ = 2
    raw = defaultdict(set); buser = {}; bday = {}
    with open(os.path.join(ROOT, "transaction_data.csv")) as f:
        r = csv.reader(f); h = next(r)
        bi = h.index("BASKET_ID"); pi = h.index("PRODUCT_ID"); hi = h.index("household_key"); di = h.index("DAY")
        for row in r:
            bid = row[bi]
            raw[bid].add(int(row[pi]))
            buser[bid] = row[hi]
            bday[bid] = int(row[di])

    bask = [(sorted(v), buser[bid], bday[bid]) for bid, v in raw.items() if 3 <= len(v) <= 20]
    random.Random(0).shuffle(bask); bask = bask[:BASKET_CAP]
    freq_all = defaultdict(int)
    for b, _, _ in bask:
        for it in b:
            freq_all[it] += 1
    keep = {it for it, c in freq_all.items() if c >= MIN_FREQ}
    tmp = [([it for it in b if it in keep], u, d) for b, u, d in bask]
    tmp = [(b, u, d) for b, u, d in tmp if len(b) >= 3]
    items = sorted({i for b, _, _ in tmp for i in b}); imap = {it: k for k, it in enumerate(items)}; n = len(items)

    seen_b = set()
    umap = {}; bundles_all = []; users_all = []; days_all = []
    for b, u, d in tmp:
        mapped = [imap[i] for i in sorted(set(b))]
        key = tuple(mapped)
        if key not in seen_b:
            seen_b.add(key)
            bundles_all.append(mapped)
            users_all.append(umap.setdefault(u, len(umap)))
            days_all.append(d)
    nusers = len(umap)

    cat_lbl = np.full(n, -1, int); catmap = {}
    txt = ["" for _ in range(n)]
    with open(os.path.join(ROOT, "product.csv")) as f:
        r = csv.reader(f); h = next(r)
        ip = h.index("PRODUCT_ID"); dep = h.index("DEPARTMENT"); com = h.index("COMMODITY_DESC")
        sub = h.index("SUB_COMMODITY_DESC"); br = h.index("BRAND")
        for row in r:
            it = int(row[ip])
            if it in imap:
                j = imap[it]
                cat_lbl[j] = catmap.setdefault(row[sub].strip() or "?", len(catmap))
                txt[j] = f"{row[dep]} {row[com]} {row[sub]} {row[br]}"
    for i in range(n):
        if cat_lbl[i] < 0:
            cat_lbl[i] = len(catmap); catmap[f"_n{i}"] = cat_lbl[i]
    NCc = int(cat_lbl.max()) + 1

    tf_mat = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(txt)
    tf_mat = normalize(tf_mat, norm="l2", axis=1).tocsr()
    cdim = min(128, tf_mat.shape[1] - 1)
    CONTENT = TruncatedSVD(cdim, random_state=0).fit_transform(tf_mat).astype(np.float32)
    CONTENT /= (np.linalg.norm(CONTENT, axis=1, keepdims=True) + 1e-8)


    idx=list(range(len(bundles_all))); random.Random(split_seed).shuffle(idx)
    ntr,nva=int(.7*len(idx)),int(.1*len(idx))
    tri,vai,tei=idx[:ntr],idx[ntr:ntr+nva],idx[ntr+nva:]
    train=[bundles_all[i] for i in tri]; train_users=[users_all[i] for i in tri]
    vp=_limit(splits.mask([bundles_all[i] for i in vai],split_seed+1,mask_frac),val_n)
    tp=_limit(splits.mask([bundles_all[i] for i in tei],split_seed+2,mask_frac),test_n)
    ds=Dataset('dunnhumby',n,train,vp,tp,['collaborative','content','taxonomy','popularity'],
     features={'content':CONTENT},cat_labels=cat_lbl,
     val_users=_limit([users_all[i] for i in vai],val_n),test_users=_limit([users_all[i] for i in tei],test_n))
    ds.research_data={'kind':'retail','content':CONTENT,'tfidf':tf_mat,'train_users':train_users,'n_users':nusers}
    return record_dataset(ds,ROOT,['transaction_data.csv','product.csv'],protocol='resplit',split_seed=split_seed,
     mask_frac=mask_frac,val_limit=val_n,test_limit=test_n,val_total=len(vai),test_total=len(tei),
     val_ids=_limit(vai,val_n),test_ids=_limit(tei,test_n),data_profile='research')


def load_commerce(name, val_n, test_n, mask_frac, split_seed):
    ROOT = os.path.join(DATA_ROOT,"BundleRec","dataset",name)
    USE_AUX = True
    raw = defaultdict(list)
    with open(os.path.join(ROOT, "bundle_item.csv")) as f:
        r = csv.reader(f); next(r)
        for b, it in r:
            raw[int(b)].append(int(it))
    items = sorted({i for v in raw.values() for i in v})
    imap = {it: k for k, it in enumerate(items)}; n = len(items)

    seen_bundles = set()
    bkeys_all, bundles_all = [], []
    for b, v in raw.items():
        u = list(dict.fromkeys(imap[i] for i in v))
        if len(u) >= 2:
            key = tuple(sorted(u))
            if key not in seen_bundles:
                seen_bundles.add(key)
                bkeys_all.append(b)
                bundles_all.append(u)

    b2u = {}
    with open(os.path.join(ROOT, "user_bundle.csv")) as f:
        r = csv.reader(f); next(r)
        for u, b, ts in r:
            b2u[int(b)] = int(u)
    b2s = {}
    with open(os.path.join(ROOT, "session_bundle.csv")) as f:
        r = csv.reader(f); next(r)
        for s, b in r:
            b2s[int(b)] = int(s)

    uip_r, uip_c = [], []; uip_map = {}
    if USE_AUX:
        with open(os.path.join(ROOT, "user_item_pretrain.csv")) as f:
            r = csv.reader(f); next(r)
            for u, it, _ in r:
                it_i = int(it)
                if it_i in imap:
                    uip_r.append(uip_map.setdefault(int(u), len(uip_map)))
                    uip_c.append(imap[it_i])
    X_pre = sp.csr_matrix((np.ones(len(uip_r), np.float32), (uip_r, uip_c)), shape=(max(len(uip_map), 1), n))
    udeg = np.asarray(X_pre.sum(1)).ravel()
    X_pre_w = sp.diags(1.0 / np.sqrt(np.maximum(udeg, 1.0))) @ X_pre
    C_pre = (X_pre_w.T @ X_pre_w).tocsr(); C_pre.setdiag(0)


    def parse_path(cat):
        cat = cat.strip()
        if cat.startswith("[[") and cat.endswith("]]"):
            return [x.strip() for x in cat[2:-2].split("],[")[0].split(",") if x.strip()]
        return []


    cat_paths = [[] for _ in range(n)]
    catmap_leaf = {}; cat_lbl = np.full(n, -1, int)
    catmap_l2 = {}; cat_l2 = np.full(n, -1, int)
    with open(os.path.join(ROOT, "item_categories.csv")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            it = int(row[0]); p = parse_path(",".join(row[1:]))
            if it in imap and p:
                idx = imap[it]; cat_paths[idx] = p
                cat_lbl[idx] = catmap_leaf.setdefault(tuple(p), len(catmap_leaf))
                cat_l2[idx] = catmap_l2.setdefault(tuple(p[:min(2, len(p))]), len(catmap_l2))
    for i in range(n):
        if cat_lbl[i] < 0:
            cat_lbl[i] = len(catmap_leaf); catmap_leaf[f"_n{i}"] = cat_lbl[i]
        if cat_l2[i] < 0:
            cat_l2[i] = len(catmap_l2); catmap_l2[f"_n{i}"] = cat_l2[i]
    NCc = int(cat_lbl.max()) + 1
    NCl2 = int(cat_l2.max()) + 1

    titles = ["" for _ in range(n)]
    with open(os.path.join(ROOT, "item_titles.csv")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            it = int(row[0])
            if it in imap:
                titles[imap[it]] = ",".join(row[1:])
    rich_txt = [titles[i] + " " + " ".join(cat_paths[i]) for i in range(n)]
    tf_mat = TfidfVectorizer(max_features=12000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(rich_txt)
    tf_mat = normalize(tf_mat, norm="l2", axis=1).tocsr()
    cdim = min(128, tf_mat.shape[1] - 1)
    CONTENT = TruncatedSVD(cdim, random_state=0).fit_transform(tf_mat).astype(np.float32)
    CONTENT /= (np.linalg.norm(CONTENT, axis=1, keepdims=True) + 1e-8)



    order=list(range(len(bundles_all)));random.Random(split_seed).shuffle(order)
    allb=[bundles_all[i] for i in order];allk=[bkeys_all[i] for i in order]
    ntr,nva=int(.7*len(allb)),int(.1*len(allb))
    train=allb[:ntr];trk=allk[:ntr];vab=allb[ntr:ntr+nva];teb=allb[ntr+nva:]
    vp=_limit(splits.mask(vab,split_seed+1,mask_frac),val_n)
    tp=_limit(splits.mask(teb,split_seed+2,mask_frac),test_n)
    ds=Dataset(name,n,train,vp,tp,['collaborative','content','taxonomy'],features={'content':CONTENT},cat_labels=cat_lbl)
    ds.research_data={'kind':'bundlerec','content':CONTENT,'tfidf':tf_mat,'cat_l2':cat_l2,'n_cat_l2':NCl2,
     'train_keys':trk,'bundle_users':b2u,'bundle_sessions':b2s,'pretrain':C_pre}
    return record_dataset(ds,ROOT,['bundle_item.csv','user_bundle.csv','session_bundle.csv','item_titles.csv',
     'item_categories.csv','user_item_pretrain.csv'],protocol='resplit',split_seed=split_seed,
     mask_frac=mask_frac,val_limit=val_n,test_limit=test_n,val_total=len(vab),test_total=len(teb),
     val_ids=_limit(allk[ntr:ntr+nva],val_n),test_ids=_limit(allk[ntr+nva:],test_n),data_profile='research')


