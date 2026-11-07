import unittest
import numpy as np
from experiments.sufficiency.engine import Bank,score,prior_weights,rho


class SufficiencyChecks(unittest.TestCase):
    def setUp(self):
        self.freq=np.array([0.,1.,4.],np.float32)
        self.z={'hist':np.array([-1.,2.,3.]),'cat':np.array([2.,1.,-1.])}
        self.cfg=dict(kind='gate',weights={'hist':.5,'cat':1.},prior_weights={'cat':1.},scale=1.2,kappa=.5)

    def test_support_controls_reduce_to_original_and_zero_strength(self):
        expected=score(self.z,self.cfg,self.freq)
        for mode in ('history_frequency','ess','count'):
            c=dict(self.cfg,kind=mode,parameter=.5)
            np.testing.assert_allclose(score(self.z,c,self.freq,{'ess':self.freq,'count':self.freq}),expected)
        np.testing.assert_array_equal(rho(self.freq,0),np.ones(3))

    def test_constant_history_and_cold_fallback(self):
        c=dict(self.cfg,kind='history_constant',parameter=0.)
        h=.5*self.z['hist'];y=h+self.z['cat'];f=1.2*self.z['cat']
        np.testing.assert_allclose(score(self.z,c,self.freq),f+rho(self.freq,.5)*(y-f-h))
        self.assertAlmostEqual(score(self.z,self.cfg,self.freq)[0],f[0])

    def test_pruned_fallback_and_bank_aliases(self):
        source={'prior':['cat','bridge']}
        self.assertEqual(prior_weights({'hist':1.},source),{})
        a={'pair':np.array([[1.,2.]]),'simp':np.array([[3.,4.]])}
        np.testing.assert_array_equal(Bank(a,['hist'],'canonical',.5).query(0)['hist'],[2.,3.])
        np.testing.assert_array_equal(Bank({'pair':a['pair']},['hist'],'canonical',0).query(0)['hist'],[1.,2.])
        b=Bank(dict(a,ess=np.array([[1.,2.]]),amp_ess=np.array([[4.,5.]])),[],'amp_selected')
        np.testing.assert_array_equal(b.aux(0)['ess'],[4.,5.])

    def test_four_item_term_and_support_bounds(self):
        from scripts.check_score_sufficiency import corpus
        from bundlefuse.operators import SimplicialGraphOperator
        g=SimplicialGraphOperator(corpus(2,0),8,svd_dim=2)
        for seeds in ([0],[0,2],[0,2,4]):
            B=g.B_bin.toarray();k=B[:,seeds].sum(1)
            overlap=(B[:,seeds]*g.inv_sqrt_f[list(seeds)]).sum(1)*g.inv_vol_pow
            third=B.T@(overlap*np.maximum(k-1,0)*np.maximum(k-2,0)/2)
            if len(seeds)<3:np.testing.assert_array_equal(third,np.zeros(8))
            else:self.assertGreater(third[6],third[7])
            h=overlap*(1+1.5*np.maximum(k-1,0));a=B.T@h;q=B.T@(h*h)
            ess=np.divide(a*a,q,out=np.zeros_like(a),where=q>0)
            count=B.T@(h>0).astype(float)
            self.assertTrue(np.all(ess<=count+1e-5))
            self.assertTrue(np.all(count<=g.freq))


if __name__=='__main__':unittest.main()
