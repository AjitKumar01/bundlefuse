"""Changes to the frozen DIALS model (branch dials), each tested by retraining against the unchanged model.

Variants (results/dials_v2/<variant>):
  candidate_weight  a candidate-dependent content weight (below). Tested 2026-11-10: not adopted.
  prior_kappa       the popularity prior gets its own gate output, and a learned scale goes inside the graph log:
                        G'(S, c) = theta log(1 + kappa sum_s n_sc deg_s^-alpha deg_c^-beta),  kappa = exp(k), k = 0 at start
                        s(c | S) = ... + w_G(S) G'(S, c) + w_P(S) rho(f_c)
                    w_P is one more gate output (17 parameters) starting at 1, so the model starts exactly at the frozen
                    one (where the prior sits inside the graph term, scaled by w_G). Motivation (code review): a query
                    with unseen seeds has no graph evidence, and lowering w_G for it also lowers the popularity
                    fallback; and the degree exponents set both the discount and where log damping starts.
  reference         the unchanged model.

candidate_weight: the content term gets a factor that depends on the candidate's training frequency f_c:

    C'(S, c) = softplus(u(log(1 + f_c)) + log(e - 1)) * C(S, c)

where u is a small curve (1 -> 16 -> 1, 49 parameters) whose last layer starts at zero, so the factor starts at 1 and
the model starts exactly at the frozen one. Motivation (code review, 2026-11-10): candidates never seen in training get
no graph or history evidence, and the query-only gate gives content the same weight for them as for well-observed
candidates; on POG 59% of held-out items are unseen and are found 4% of the time. In training, a bundle member's
frequency excludes the bundle itself (leave-one-out), as for the prior. Everything else, including the training
configuration and seeds, is the frozen model's.

Selection on validation NDCG@20 only, as frozen. Training at 4 threads is not bit-identical to the frozen 1-thread runs
(POG seed 0 matches through epoch 13, then drifts by floating-point summation order), so the reference is the unchanged
model retrained here at the same thread count (train --reference, results/dials_v2/reference_4threads). compare: clean
test queries, three-seed per-query means, new minus reference, paired cluster bootstrap and sign flips, Holm over the
tests run.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_v2.run train --variant prior_kappa --dataset pog --seed 0 --threads 4
PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_v2.run compare --variant prior_kappa --datasets pog pog_dense
"""
import dials  # pins BLAS threads before numpy loads
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dials import data
from dials import train as T
from dials.evaluate import evaluate
from dials.model import DIALS,curve
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison,holm_adjust

RESULTS=Path('results/dials_v2');REFERENCE=RESULTS/'reference_4threads';CACHE=Path(DATA_ROOT)/'dials'/'cache'


class CandidateWeight(DIALS):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.content_weight=curve(1)
        nn.init.zeros_(self.content_weight[-1].weight);nn.init.zeros_(self.content_weight[-1].bias)

    def fast_parameters(self):
        return super().fast_parameters()+list(self.content_weight.parameters())

    def factor(self,frequency):
        return F.softplus(self.content_weight(frequency.log1p()[:,None]).squeeze(1)+math.log(math.e-1))

    def cache(self):
        return dict(super().cache(),content_factor=self.factor(self.frequency))

    def content_scores(self,seeds,bundles,cache=None):
        scores=super().content_scores(seeds,bundles,cache)
        factor=(self.factor(self.frequency) if cache is None else cache['content_factor']).expand(len(seeds),-1)
        if bundles is not None:  # leave-one-out: a member's frequency without its own bundle
            rows=torch.cat([torch.full((len(b),),i) for i,b in enumerate(bundles)]);cols=torch.cat(bundles)
            factor=factor.index_put((rows,cols),self.factor((self.frequency[cols]-1).clamp_min(0)))
        return scores*factor


class PriorKappa(DIALS):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gate_prior=nn.Linear(self.gate[-1].in_features,1)
        nn.init.zeros_(self.gate_prior.weight);nn.init.constant_(self.gate_prior.bias,math.log(math.e-1))  # w_P = 1
        self.log_kappa=nn.Parameter(torch.zeros(()))

    def fast_parameters(self):
        return super().fast_parameters()+[*self.gate_prior.parameters(),self.log_kappa]

    def gate_both(self,seeds,bundles):
        stats=[]
        for s in seeds:
            f=(self.frequency[s]-(1 if bundles is not None else 0)).clamp_min(0)
            stats.append(torch.stack((torch.tensor(float(len(s))).log(),f.log1p().mean(),(f==0).float().mean())))
        hidden=self.gate[1](self.gate[0](torch.stack(stats)))
        return F.softplus(self.gate[2](hidden)),F.softplus(self.gate_prior(hidden))

    def graph_parts(self,seeds,bundles,cache=None):
        """DIALS.graph_scores with kappa inside the log, returning the graph and prior parts separately."""
        alpha,beta=self.exponents;q=len(seeds)
        query,source,edge=self.expand(torch.cat([torch.full((len(s),),i) for i,s in enumerate(seeds)]),torch.cat(seeds))
        target=self.target[edge];n=self.counts[edge]
        ds=self.log_degree[source];dt=self.log_degree[target]
        if bundles is not None:
            member=torch.zeros(q,self.n,dtype=torch.bool)
            for i,b in enumerate(bundles):member[i,b]=True
            inside=member[query,target].float();n=n-inside
            others=torch.tensor([float(len(b)-1) for b in bundles])[query]
            ds=(self.history_mass[source]-others).clamp_min(1).log();dt=(self.history_mass[target]-inside*others).clamp_min(1).log()
        w=n*torch.exp(-alpha*ds-beta*dt)
        first=torch.zeros(q*self.n).scatter_add(0,query*self.n+target,w).view(q,self.n)
        graph=self.theta*(self.log_kappa.exp()*first).log1p()
        prior=(self.prior(self.frequency.log1p()[:,None]).squeeze(1) if cache is None else cache['prior']).expand(q,-1)
        if bundles is not None:
            rows=torch.cat([torch.full((len(b),),i) for i,b in enumerate(bundles)]);cols=torch.cat(bundles)
            left=(self.frequency[cols]-1).clamp_min(0)
            prior=prior.index_put((rows,cols),self.prior(left.log1p()[:,None]).squeeze(1)-prior[rows,cols],accumulate=True)
        return graph,prior

    def forward(self,seeds,bundles=None,cache=None):
        if cache is not None and bundles is not None:raise ValueError('The cache is for evaluation only')
        seeds=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s in seeds]
        if bundles is not None:bundles=[torch.as_tensor(sorted(set(b)),dtype=torch.long) for b in bundles]
        graph,prior=self.graph_parts(seeds,bundles,cache)
        parts=[self.content_scores(seeds,bundles,cache)]+[self.label_scores(i,seeds,bundles) for i in range(self.levels)]+[graph]
        w,wp=self.gate_both(seeds,bundles)
        return sum(w[:,j,None]*v for j,v in enumerate(parts))+wp*prior


VARIANTS=dict(reference=DIALS,candidate_weight=CandidateWeight,prior_kappa=PriorKappa)


def build(ds,cache,config):
    model=VARIANTS[config.get('variant','reference')]  # the 4-thread check run predates the variant key
    return model(ds.features,ds.levels,ds.train,T.spectral(ds,cache,config),rank=config['rank'],hidden=config['hidden'])


T.build=build  # dials.train.fit and dials.train.load construct models through this name


def train(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    variant=args.variant;run=(REFERENCE if variant=='reference' else RESULTS/variant)/args.dataset/f'seed{args.seed}'
    ds=data.load(args.dataset,CACHE);T.fit(ds,args.seed,run,CACHE,dict(T.CONFIG,variant=variant))
    if not (run/'test.json').exists():write_json(run/'test.json',evaluate(T.load(ds,run,CACHE),ds.test_pairs))
    T.log(f'{variant} {args.dataset} seed {args.seed} COMPLETE')


def compare(args):
    report=dict(note=__doc__,datasets={});records=[];groups=lambda pairs:[str(tuple(sorted(set(s)|set(g)))) for s,g in pairs]
    mean=lambda runs:np.mean([[[q['recall20'],q['ndcg20']] for q in json.loads((r/'test.json').read_text())['queries']] for r in runs],axis=0)
    for name in args.datasets:
        ds=data.load(name,CACHE);keep=ds.clean_test;clusters=[groups(ds.test_pairs)[i] for i in keep]
        new_runs=[RESULTS/args.variant/name/f'seed{s}' for s in (0,1,2)];old_runs=[REFERENCE/name/f'seed{s}' for s in (0,1,2)]
        new,old=mean(new_runs),mean(old_runs)
        val=lambda runs:float(np.mean([json.loads((r/'selected.json').read_text())['validation']['ndcg20'] for r in runs]))
        report['datasets'][name]=dict(reference=dict(validation_ndcg20=val(old_runs),test=old[keep].mean(0).tolist()),
                                      **{args.variant:dict(validation_ndcg20=val(new_runs),test=new[keep].mean(0).tolist())})
        for j,metric in ((1,'ndcg20'),(0,'recall20')):
            records.append(dict(paired_comparison(new[keep,j],old[keep,j],clusters,bootstrap=2000,permutations=10000,seed=20261110),dataset=name,metric=metric))
    holm_adjust(records,len(records));report['primary']=records
    write_json(RESULTS/args.variant/'summary.json',report)
    for name,v in report['datasets'].items():print(name,{k:(round(x['validation_ndcg20'],4),[round(t,4) for t in x['test']]) for k,x in v.items()})
    for r in records:print(r['dataset'],r['metric'],round(r['delta'],4),[round(x,4) for x in r['ci95']],round(r['p_holm'],4))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);sub=p.add_subparsers(required=True)
    t=sub.add_parser('train');t.set_defaults(run=train);t.add_argument('--dataset',required=True,choices=data.DATASETS)
    t.add_argument('--seed',type=int,required=True);t.add_argument('--threads',type=int,default=4)
    t.add_argument('--variant',required=True,choices=sorted(VARIANTS))
    t=sub.add_parser('compare');t.set_defaults(run=compare);t.add_argument('--datasets',nargs='+',required=True,choices=data.DATASETS)
    t.add_argument('--variant',required=True,choices=sorted(set(VARIANTS)-{'reference'}))
    args=p.parse_args();args.run(args)
