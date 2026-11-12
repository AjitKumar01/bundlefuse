import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from experiments.neural_revision.assays import run_assays
from experiments.neural_revision.model import RelationNetwork,full_catalogue_loss
from experiments.neural_revision.training import fit,evaluate,project_parameters


class RevisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(13)
        self.features=np.random.default_rng(2).normal(size=(12,8)).astype('float32')
        self.cat=np.arange(12)%3
        self.train=[[0,1,2],[1,3],[4,5,6],[5,7],[8,9]]

    def test_mechanism_and_optimization_assays(self):
        result=run_assays()
        self.assertLess(result['synthetic_loss_final'],result['synthetic_loss_initial']*.1)

    def test_full_catalogue_loss_matches_explicit_positive_denominators(self):
        scores=torch.tensor([[1.,2.,3.,4.,5.]],requires_grad=True)
        loss=full_catalogue_loss(scores,[[0]],[[1,2]])
        expected=torch.stack([torch.logsumexp(scores[0,[i,3,4]],0)-scores[0,i] for i in [1,2]]).mean()
        torch.testing.assert_close(loss,expected)
        loss.backward();self.assertEqual(scores.grad[0,0],0.)
        self.assertTrue(bool((scores.grad[0,1:3]<0).all()))

    def test_cold_history_has_no_item_or_group_or_graph_path(self):
        model=RelationNetwork(self.features,self.cat,self.train,'graph_joint')
        keep=torch.zeros(12)
        before=model([[0,1],[4]],keep).detach()
        with torch.no_grad():
            model.pair_weights.normal_(mean=3.,std=5.)
            model.joint_weights.normal_(mean=2.,std=4.)
            model.graph_projection.weight.normal_()
        torch.testing.assert_close(before,model([[0,1],[4]],keep))

    def test_score_roundtrip_and_batching(self):
        model=RelationNetwork(self.features,self.cat,self.train,'graph_joint')
        with torch.no_grad():model.pair_weights.normal_();model.joint_weights.normal_()
        pairs=[([0,1],[2]),([4],[5,6])]
        a=evaluate(model,pairs,batch=1);b=evaluate(model,pairs,batch=2)
        self.assertEqual(a['queries'],b['queries'])
        restored=RelationNetwork(self.features,self.cat,self.train,'graph_joint')
        restored.load_state_dict(model.state_dict())
        self.assertEqual(a['queries'],evaluate(restored,pairs)['queries'])

    def test_unseen_categories_do_not_learn_negative_only_weights(self):
        cats=self.cat.copy();cats[10:]=9
        model=RelationNetwork(self.features,cats,self.train,'attributes')
        with torch.no_grad():model.category_weights.normal_()
        components=model.components([[0,1]])
        self.assertTrue(bool((components['category'][:,10:]==0).all()))
        loss=full_catalogue_loss(model([[0]]),[[0]],[[1]])
        loss.backward()
        self.assertTrue(bool((model.category_weights.grad[:,9]==0).all()))

    def test_constrained_updates_and_disabled_category(self):
        m=RelationNetwork(self.features,self.cat,self.train,'graph_joint',category_mode='nonnegative')
        with torch.no_grad():
            m.pair_weights.fill_(-1);m.joint_weights.fill_(-2);m.category_weights.fill_(-3)
        project_parameters(m,{'project_links':True})
        self.assertTrue(bool((m.pair_weights>=0).all()))
        self.assertTrue(bool((m.joint_weights>=0).all()))
        self.assertTrue(bool((m.category_weights>=0).all()))
        off=RelationNetwork(self.features,self.cat,self.train,'graph_pair',category_mode='off')
        self.assertEqual(off.category_weights.numel(),0)
        self.assertEqual(torch.count_nonzero(off.components([[0]])['category']),0)
        self.assertNotIn('category_weights',dict(off.named_parameters()))

    def test_indexed_pair_scoring_matches_sparse_forward_and_gradient(self):
        m=RelationNetwork(self.features,self.cat,self.train,'graph_pair',category_mode='off')
        with torch.no_grad():m.pair_weights.normal_()
        seeds=[[0,1],[4,5],[8]]
        indicator=torch.zeros(3,12)
        for q,s in enumerate(seeds):indicator[q,s]=1
        source,target=m.edges
        expected=torch.sparse.mm(torch.sparse_coo_tensor(torch.stack((target,source)),m.pair_weights,(12,12)),indicator.T).T
        actual=m.components(seeds)['pair']
        torch.testing.assert_close(actual,expected)
        weight=torch.randn_like(actual)
        a=torch.autograd.grad((actual*weight).sum(),m.pair_weights)[0]
        b=torch.autograd.grad((expected*weight).sum(),m.pair_weights)[0]
        torch.testing.assert_close(a,b)
        m.eval()
        with torch.no_grad():torch.testing.assert_close(m(seeds),m(seeds,item_vectors=m.item_vectors()))

    def test_resumed_training_is_identical(self):
        ds=SimpleNamespace(n_items=12,train=self.train,val_pairs=[([0],[1]),([4],[5])])
        condition=dict(name='graph_joint_cold',variant='graph_joint',cold_probability=.25)
        config=dict(epochs=4,batch=2,eval_every=2,patience=6,link_lr=.03,shared_lr=.001,weight_decay=.001)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            fit(ds,self.features,self.cat,condition,0,config,root/'out',root/'ckpt',lambda _:None)
            (root/'resume').mkdir();shutil.copy2(root/'ckpt/epoch_0002.pt',root/'resume/epoch_0002.pt')
            fit(ds,self.features,self.cat,condition,0,config,root/'out_resume',root/'resume',lambda _:None)
            a=torch.load(root/'ckpt/epoch_0004.pt',weights_only=False);b=torch.load(root/'resume/epoch_0004.pt',weights_only=False)
            for k in a['model']:self.assertTrue(torch.equal(a['model'][k],b['model'][k]),k)
            self.assertEqual(a['history'][-1]['loss'],b['history'][-1]['loss'])


if __name__=='__main__':unittest.main()
