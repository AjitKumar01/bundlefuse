"""Leakage audit for a user co-interaction term on POG: how much do users' histories already contain the bundles?

For each split: share of bundles wholly inside one user's history; share of queries where some user holds every seed
and at least one held-out item; share of (seed, held-out) pairs with user co-interaction M > 0 against bundle
co-occurrence N > 0. Items: share of the catalogue, and of held-out test items, that any user interacted with.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_user.audit
"""
import numpy as np
from collections import defaultdict
from dials import data as dials_data
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import CACHE
from pathlib import Path


def histories(name):
    rows=[[int(v) for v in line.split(',')] for line in (Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/'ui_full.txt').read_text().split('\n') if line.strip()]
    return {r[0]:set(r[1:]) for r in rows}


def main(name='pog'):
    ds=dials_data.load(name,CACHE);users=histories(name);owners=defaultdict(set)
    for u,items in users.items():
        for i in items:owners[i].add(u)
    pairs=set()
    for b in ds.train:
        b=sorted(set(b));pairs.update((s,c) for s in b for c in b if s!=c)
    def inside(items):
        common=None
        for i in items:
            common=set(owners.get(i,())) if common is None else common&owners.get(i,set())
            if not common:return False
        return True
    def users_with_all(items):
        common=None
        for i in items:
            common=set(owners.get(i,())) if common is None else common&owners.get(i,set())
            if not common:return set()
        return common
    report=dict(users=len(users),interactions=sum(map(len,users.values())),
                catalogue_with_users=float(np.mean([i in owners for i in range(ds.n_items)])))
    report['train_bundles_inside_one_user']=float(np.mean([inside(sorted(set(b))) for b in ds.train]))
    for split,qs in (('valid',ds.val_pairs),('test',ds.test_pairs),('clean_test',[ds.test_pairs[i] for i in ds.clean_test])):
        whole=[];partial=[];m=[];n=[];cov=[]
        for s,g in qs:
            whole.append(inside(sorted(set(s)|set(g))))
            seedusers=users_with_all(s);partial.append(any(seedusers&owners.get(c,set()) for c in g))
            for c in g:
                cov.append(c in owners)
                m.append(any(owners.get(x,set())&owners.get(c,set()) for x in s));n.append(any((x,c) in pairs for x in s))
        report[split]=dict(queries=len(qs),bundle_inside_one_user=float(np.mean(whole)),seeds_and_a_target_in_one_user=float(np.mean(partial)),
                           targets_with_users=float(np.mean(cov)),pairs_with_user_coevidence=float(np.mean(m)),pairs_with_bundle_cooccurrence=float(np.mean(n)),
                           user_but_no_bundle_evidence=float(np.mean(np.array(m)&~np.array(n))))
    for k,v in report.items():print(k,v)
    out=Path('results/dials_user');out.mkdir(parents=True,exist_ok=True);write_json(out/f'audit_{name}.json',dict(note=__doc__,**report))


if __name__=='__main__':main()
