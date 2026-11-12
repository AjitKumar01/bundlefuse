"""Households of Dunnhumby baskets, in dials.data.load_baskets' item numbering and split.

Rebuilds the loader's basket list with each basket's household and day (the first basket of a duplicated item set is
kept, as the loader keeps it), applies the same 70/10/20 shuffle and the same seed hiding, and checks that training
bundles, validation pairs and test pairs all match the loader's exactly.

Histories use training baskets only, so a validation or test basket never enters one. A training basket is in its own
household's history, so training scores need leave-one-out (model.DIALSUser, repurchase.DIALSRepurchase).
"""
import csv,random
from dataclasses import dataclass
from dials import data


@dataclass
class Split:
    train:list   # (household, day) per training bundle, aligned with ds.train
    val:list     # (household, day) per validation pair, aligned with ds.val_pairs
    test:list    # (household, day) per test pair, aligned with ds.test_pairs


def rebuild(ds,seed=0,cap=30000):
    root=data.DATA_ROOT/'dunnhumby';raw={};owner={};day={}
    with open(root/'transaction_data.csv') as f:
        rows=csv.reader(f);h=next(rows);b,p,u,t=(h.index(k) for k in ('BASKET_ID','PRODUCT_ID','household_key','DAY'))
        for row in rows:raw.setdefault(row[b],set()).add(int(row[p]));owner[row[b]]=row[u];day[row[b]]=int(row[t])
    baskets=[(k,sorted(v)) for k,v in raw.items() if 3<=len(v)<=20]  # dict order equals the loader's raw.values() order
    random.Random(0).shuffle(baskets);baskets=baskets[:cap]
    counts={}
    for _,v in baskets:
        for i in v:counts[i]=counts.get(i,0)+1
    baskets=[(k,[i for i in v if counts[i]>=2]) for k,v in baskets];baskets=[(k,v) for k,v in baskets if len(v)>=3]
    items=sorted({i for _,v in baskets for i in v});index={item:j for j,item in enumerate(items)}
    seen=set();bundles=[];meta=[]
    for k,v in baskets:
        key=tuple(index[i] for i in v)
        if key not in seen:seen.add(key);bundles.append(list(key));meta.append((owner[k],day[k]))
    order=list(range(len(bundles)));random.Random(seed).shuffle(order)
    a=int(.7*len(order));c=a+int(.1*len(order))
    train=[bundles[i] for i in order[:a]]
    val=data.hide([bundles[i] for i in order[a:c]],seed+1);test=data.hide([bundles[i] for i in order[c:]],seed+2)
    if train!=ds.train or val!=ds.val_pairs or test!=ds.test_pairs:raise RuntimeError('Rebuilt split differs from the loader')
    return Split([meta[i] for i in order[:a]],[meta[i] for i in order[a:c]],[meta[i] for i in order[c:]])


def households(train,owners):
    """Per household: item -> number of its training baskets holding the item; and bundle key -> household index."""
    index={};counts=[];owner={}
    for (u,_),b in zip(owners,train):
        k=index.setdefault(u,len(index))
        if k==len(counts):counts.append({})
        for i in set(b):counts[k][i]=counts[k].get(i,0)+1
        owner[tuple(sorted(set(b)))]=k
    return counts,owner,index


def training_households(ds):
    counts,owner,_=households(ds.train,rebuild(ds).train);return counts,owner
