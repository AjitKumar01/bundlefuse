"""Simple baselines on one bundle-budget condition: tuned on validation NDCG@20 as in simple_baselines and
label_baselines, test scored once. Statistics come from the condition's bundles (all of them, not a sample).

Methods: popularity, co-occurrence, ItemKNN, RP3beta, EASE (at most 40,000 active items), the content + ItemKNN blend,
and label + ItemKNN + content. Writes results/bundle_budget/baselines/<dataset>/<condition>.json.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget.baselines --dataset pog --condition frac10
"""
import argparse
import dials  # pins BLAS threads before numpy loads
import time
from bundlefuse.provenance import write_json
from experiments.label_baselines.run import LabelBaselines
from experiments.simple_baselines.run import evaluate,mean
from experiments.neural_revision.run import log
from .conditions import dataset,condition,CONDITIONS,DATASETS,OUT


def main(args):
    ds=dataset(args.dataset);stats,_=condition(args.dataset,args.condition);keep=sorted(ds.clean_test)
    model=LabelBaselines(stats,ds.n_items,ds.levels,ds.features);report=dict(dataset=args.dataset,condition=args.condition,bundles=len(stats),methods={})
    def tune(name,grid,scorer=None):
        rows=[]
        for p in grid:
            v=mean(evaluate(model,name,p,ds.val_pairs));rows.append((v['ndcg20'],p));log(f'{args.dataset} {args.condition} {name} {p} VAL N20={v["ndcg20"]:.5f}')
        best=max(rows,key=lambda r:r[0]);records=evaluate(model,name,best[1],ds.test_pairs)
        report['methods'][name]=dict(params=best[1],validation_ndcg20=best[0],test=mean(records,keep),queries=[[records[i]['recall20'],records[i]['ndcg20']] for i in keep])
        log(f'{args.dataset} {args.condition} {name} TEST clean {report["methods"][name]["test"]}');return best[1]
    grid=model.grid(40000)
    for name in ('popularity','cooccurrence','itemknn','rp3beta','ease'):
        g=[p for n,p in grid if n==name]
        if g:tune(name,g)
    knn=report['methods']['itemknn']['params']
    tune('hybrid',[p for n,p in model.extended_grid(knn) if n=='hybrid'])
    lg=model.label_grid()
    for name in ('samelabel','transition'):tune(name,[p for n,p in lg if n==name])
    label=max(('samelabel','transition'),key=lambda n:report['methods'][n]['validation_ndcg20']);lp=report['methods'][label]['params']
    best=tune('label_knn',[dict(label=label,label_params=lp,itemknn=knn,w=w) for w in (0.,.1,.2,.3,.5,.7,.9,1.)])
    tune('label_knn_content',[dict(best,v=v) for v in (0.,.05,.1,.2,.3,.5)])
    write_json(OUT/'baselines'/args.dataset/f'{args.condition}.json',report);log(f'{args.dataset} {args.condition} baselines COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=DATASETS);p.add_argument('--condition',required=True,choices=CONDITIONS)
    main(p.parse_args())
