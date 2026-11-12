import unittest
import numpy as np
import torch
from experiments.neural_lowrank.model import LowRankRelation,pair_counts


def toy():
    rng=np.random.default_rng(0)
    train=[[0,1,2],[1,2,3],[2,3,4,5],[0,5],[1,4],[3,5,6],[0,2,6],[1,2,3]]
    return rng.normal(size=(8,16)).astype(np.float32),np.array([0,1,2,0,1,2,0,1]),train


class LowRankTest(unittest.TestCase):
    def test_counts(self):
        source,target,counts,frequency,pointer=pair_counts([[0,1],[1,0,2],[0,1]],3)
        table={(int(s),int(t)):int(n) for s,t,n in zip(source,target,counts)}
        self.assertEqual(table[(0,1)],3);self.assertEqual(table[(1,0)],3);self.assertEqual(table[(2,0)],1)
        self.assertEqual(frequency.tolist(),[3,3,1]);self.assertEqual(pointer.tolist(),[0,2,4,6])

    def test_leave_one_out_equals_refit_without_the_row(self):
        x,g,train=toy();torch.manual_seed(0)
        model=LowRankRelation(x,g,train,rank=8,group_rank=4,hidden=4)
        with torch.no_grad():
            for p in model.parameters():p.add_(.3*torch.randn_like(p))
        for i in (1,2,7):
            other=LowRankRelation(x,g,train[:i]+train[i+1:],rank=8,group_rank=4,hidden=4)
            state={k:v for k,v in model.state_dict().items() if k in dict(model.named_parameters())}
            other.load_state_dict(state,strict=False)
            row=train[i];seeds=[row[:2]]
            with torch.no_grad():
                loo=model(seeds,bundles=[row]);refit=other(seeds)
            self.assertTrue(torch.allclose(loo[0,row],refit[0,row],atol=1e-5),(i,loo[0,row],refit[0,row]))

    def test_cached_keys_match_and_cold_items_score(self):
        x,g,train=toy();model=LowRankRelation(x,g,train,rank=8,group_rank=4,hidden=4)
        with torch.no_grad():
            a=model([[0,1]]);b=model([[0,1]],keys=model.keys())
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.isfinite(a[0,7]))


class GroupMatchTest(unittest.TestCase):
    def build(self,train):
        from experiments.neural_lowrank2.model import GroupMatchRelation
        x,g,_=toy();album=np.array([0,0,1,1,2,2,3,3])
        return GroupMatchRelation(x,[g,album],train,rank=8,group_rank=4,hidden=4)

    def test_leave_one_out_equals_refit_without_the_row(self):
        _,_,train=toy();torch.manual_seed(0);model=self.build(train)
        with torch.no_grad():
            for p in model.parameters():p.add_(.3*torch.randn_like(p))
        for i in (1,2,7):
            other=self.build(train[:i]+train[i+1:])
            other.load_state_dict({k:v for k,v in model.state_dict().items() if k in dict(model.named_parameters())},strict=False)
            row=train[i]
            with torch.no_grad():loo=model([row[:2]],bundles=[row]);refit=other([row[:2]])
            self.assertTrue(torch.allclose(loo[0,row],refit[0,row],atol=1e-5))

    def test_match_counts(self):
        _,_,train=toy();model=self.build(train);seen=[]
        model.match[0]=SpyCurve(seen);model.match[1]=SpyCurve(seen)
        with torch.no_grad():model.group_match([[0,3]])
        # categories [0,1,2,0,1,2,0,1]: items 0,3,6 match twice; albums [0,0,1,1,...]: items 0,1,2,3 match once
        self.assertEqual(sorted(seen[0]),[2.,2.,2.]);self.assertEqual(sorted(seen[1]),[1.,1.,1.,1.])


class RelationFamilyTest(unittest.TestCase):
    mode='free'

    def build(self,train):
        from experiments.neural_relations.model import RelationFamily
        x,g,_=toy();album=np.array([0,0,1,1,2,2,3,3])
        return RelationFamily(x,[g,album],train,rank=8,group_rank=2,hidden=4,graph_rank=8,label_mode=self.mode)

    def test_leave_one_out_equals_refit_without_the_row(self):
        _,_,train=toy();torch.manual_seed(0);model=self.build(train)
        with torch.no_grad():
            for name,p in model.named_parameters():p.add_(.3*torch.randn_like(p))
            model.hops[1]=0  # the second hop is cross-fitted instead; tested below
        for i in (1,2,7):
            other=self.build(train[:i]+train[i+1:]);other.load_state_dict(model.state_dict())
            row=train[i]
            with torch.no_grad():loo=model([row[:2]],bundles=[row]);refit=other([row[:2]])
            self.assertTrue(torch.allclose(loo[0,row],refit[0,row],atol=1e-5),(i,loo[0,row],refit[0,row]))

    def test_cross_fitting_never_sees_the_bundle(self):
        from experiments.neural_relations.model import cross_fitted,fold_of
        _,g,train=toy();out=cross_fitted(train,8,8,[g],2,folds=3);full,folds=out['graph']
        self.assertEqual(fold_of([3,2,1],3),fold_of([1,2,3,3],3))
        for b in train:
            k=fold_of(b,3);rest=[r for r in train if fold_of(r,3)!=k]
            refit=cross_fitted(rest,8,8,[g],2,folds=3)
            self.assertTrue(np.allclose(folds[k][0]@folds[k][1].T,refit['graph'][0][0]@refit['graph'][0][1].T,atol=1e-4))
            a=out['labels'][0][1][k];b2=refit['labels'][0][0]
            self.assertTrue(np.allclose(a[0]@a[1].T,b2[0]@b2[1].T,atol=1e-4))

    def test_second_hop_matches_dense(self):
        _,_,train=toy();model=self.build(train)
        with torch.no_grad():
            model.exponents.copy_(torch.tensor([.3,.5,.7]));model.hops.copy_(torch.tensor([0.,1.]))
            model.disabled={'prior'};got=model.graph_scores([torch.tensor([0,1])],None,None)
        N=torch.zeros(8,8)
        for b in train:
            for a in set(b):
                for c in set(b):
                    if a!=c:N[a,c]+=1
        deg=N.sum(1).clamp_min(1).log();y=torch.zeros(8);y[[0,1]]=1
        first=(y*torch.exp(-.3*deg))@N*torch.exp(-.5*deg)
        second=(first*torch.exp(-.7*deg))@N*torch.exp(-.5*deg)
        self.assertTrue(torch.allclose(got[0],second.log1p(),atol=1e-5))


class SpectralLabelTest(RelationFamilyTest):
    mode='spectral'

    def test_leave_one_out_equals_refit_without_the_row(self):
        # the label part is cross-fitted like the second hop, so compare with both switched off
        _,_,train=toy();torch.manual_seed(0);model=self.build(train)
        with torch.no_grad():
            for p in model.parameters():p.add_(.3*torch.randn_like(p))
            model.hops[1]=0
        model.disabled={'lowrank'}
        for i in (1,2,7):
            other=self.build(train[:i]+train[i+1:]);other.load_state_dict(model.state_dict());other.disabled={'lowrank'}
            row=train[i]
            with torch.no_grad():loo=model([row[:2]],bundles=[row]);refit=other([row[:2]])
            self.assertTrue(torch.allclose(loo[0,row],refit[0,row],atol=1e-5))


class SpyCurve(torch.nn.Module):
    def __init__(self,seen):super().__init__();self.seen=seen
    def forward(self,x):self.seen.append(x[:,0].expm1().round().tolist());return torch.zeros(len(x),1)


if __name__=='__main__':
    unittest.main()
