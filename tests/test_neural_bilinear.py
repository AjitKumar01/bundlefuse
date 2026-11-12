import unittest
import numpy as np
import torch
from experiments.neural_bilinear.model import BilinearNetwork
from experiments.neural_revision.model import RelationNetwork


class BilinearTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.features=np.random.default_rng(2).normal(size=(12,8)).astype('float32')
        self.cat=np.arange(12)%3;self.train=[[0,1,2],[1,3],[4,5,6],[5,7],[8,9]];self.seeds=[[0,1],[4]]

    def build(self,**kw):
        torch.manual_seed(13);return BilinearNetwork(self.features,self.cat,self.train,**kw)

    def test_new_terms_start_as_baseline(self):
        torch.manual_seed(13);base=RelationNetwork(self.features,self.cat,self.train,variant='graph_pair',category_mode='off')
        with torch.no_grad():base.pair_weights.uniform_(0,2);base.graph_projection.weight.normal_()
        for kw in (dict(category_mode='free',bilinear=True),dict(category_mode='off',bilinear=True),dict(category_mode='free')):
            model=self.build(**kw);model.load_state_dict({k:v for k,v in base.state_dict().items() if k!='category_weights'},strict=False)
            torch.testing.assert_close(model(self.seeds).detach(),base(self.seeds).detach())
            keep=torch.ones(12);keep[1]=0
            torch.testing.assert_close(model(self.seeds,keep).detach(),base(self.seeds,keep).detach())

    def test_bilinear_can_be_asymmetric_and_learns(self):
        model=self.build(bilinear=True)
        model(self.seeds).sum().backward();self.assertGreater(float(model.bilinear.grad.abs().sum()),0)
        with torch.no_grad():model.bilinear.add_(torch.tril(torch.ones(8,8),-1)*3)
        vectors=model.item_vectors().detach();a=model([[0]],item_vectors=vectors)[0,1];b=model([[1]],item_vectors=vectors)[0,0]
        self.assertGreater(abs(float(a-b)),1e-4)

    def test_category_term_survives_cold_mask_and_reaches_cold_items(self):
        model=self.build(category_mode='free',bilinear=True)
        with torch.no_grad():model.category_weights[0,1]=5.;model.category_weights[0,0]=-5.
        before=model([[0]],torch.zeros(12)).detach();after=self.build(bilinear=True)([[0]],torch.zeros(12)).detach()
        difference=before-after
        self.assertAlmostEqual(float(difference[0,1]),5.,places=4);self.assertAlmostEqual(float(difference[0,0]),-5.,places=4);self.assertAlmostEqual(float(difference[0,2]),0.,places=4)


if __name__=='__main__':
    unittest.main()
