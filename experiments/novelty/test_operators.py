"""Small numerical checks for independent operators and leakage boundaries."""
import unittest
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from bundlefuse.operators import SimplicialGraphOperator
from .raw import AmplifiedNeighbors,standardize
from .fusion import score,train_linear,features
from .data import validation_groups


class StudyChecks(unittest.TestCase):
    def test_amplification_matches_explicit_bundle_sum(self):
        rows=[[0,1,2],[0,2],[1,3],[3,4]]
        g=SimplicialGraphOperator(rows,5,svd_dim=2)
        operator=AmplifiedNeighbors(g);seeds=[0,1]
        overlap=np.array([sum(g.inv_sqrt_f[s] for s in seeds if s in b)/np.sqrt(len(b)*2) for b in rows])
        for power in [1,2,3]:
            expected=np.array([sum(overlap[j]**power for j,b in enumerate(rows) if c in b) for c in range(5)])
            np.testing.assert_allclose(operator.score(seeds,power,None),expected,rtol=1e-6)
        best=int(np.argmax(overlap));expected=np.zeros(5);expected[rows[best]]=overlap[best]**2
        np.testing.assert_allclose(operator.score(seeds,2,1),expected,rtol=1e-6)

    def test_incidence_vanishes_without_joint_seed_matches(self):
        g=SimplicialGraphOperator([[0,2],[1,2],[3,4]],5,svd_dim=2)
        delta,_=g.score_simplicial_incidence_only([[0,1]])
        np.testing.assert_array_equal(delta,np.zeros_like(delta))

    def test_validation_partition_keeps_duplicate_complete_bundles_together(self):
        pairs=[([0],[1,2]),([2],[0,1]),([3],[4]),([5],[6]),([7],[8])]
        fit,tune=validation_groups(pairs,10,.7)
        self.assertEqual((0 in fit),(1 in fit))
        self.assertFalse(set(fit)&set(tune));self.assertEqual(set(fit)|set(tune),set(range(5)))

    def test_rrf_excludes_seed_from_source_ranks(self):
        z={'a':np.array([99.,3.,2.,1.])};cfg=dict(kind='rrf',offset=10.)
        actual=score(z,cfg,['a'],np.ones(4),seeds=[0])
        np.testing.assert_allclose(actual,[0,1/11,1/12,1/13])

    def test_frequency_expanded_linear_can_express_gate_exactly(self):
        z={'x':np.array([.2,.3,.4]),'y':np.array([.5,.6,.7])};f=np.array([0.,2.,20.])
        cfg=dict(kind='bundlefuse_gate',weights={'x':1.,'y':2.},prior_weights={'y':2.},scale=1.4,kappa=2.)
        gate=score(z,cfg,['x','y'],f)
        linear=dict(kind='frequency_linear',coefficients=[0,2.8,1.,-.8,0.],kappa=2.)
        np.testing.assert_allclose(gate,score(z,linear,['x','y'],f),rtol=1e-6)

    def test_train_ranker_improves_pair_order_without_target_feature(self):
        x=np.array([[2,1],[0,1],[3,2],[-1,2]],np.float32)
        method=train_linear(x,np.array([1,0,1,0]),[2,2],.01,1.,1)
        z={'a':np.array([2.,0.,3.,-1.])};scores=score(z,method,['a'],x[:,-1])
        self.assertGreater(scores[0],scores[1]);self.assertGreater(scores[2],scores[3])


if __name__=='__main__':unittest.main()
