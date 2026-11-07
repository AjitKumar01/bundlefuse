"""Protect source-equivalence and personalization in the cross-dataset follow-up."""
import unittest
import numpy as np
import torch
from bundlefuse.fusion import apply_fusion
from experiments.amplified_fallback.data import effective, ScoreBank
from experiments.amplified_fallback.run import canonical_method
from experiments.amplified_fallback.fitting import score


class FollowupChecks(unittest.TestCase):
    def test_original_formula_preserves_incidence_diffusion_and_prior_terms(self):
        rng=np.random.default_rng(17)
        raw={k:rng.normal(size=(9,1)).astype(np.float32) for k in
             ['pair','simp','diff2_pair','diff2','cat','rao','pop']}
        frequency=np.array([0,0,1,2,3,5,10,20,100],dtype=np.float32)
        for alpha in (0.,.5,1.):
            recipe=dict(weights={'simp':1.5,'diff2':.25,'cat':.5,'pop':.25},
                        prior=['cat'],higher_order=[alpha,.35],backoff=[.5,1.4])
            expected=apply_fusion({k:torch.tensor(v) for k,v in raw.items()},recipe['weights'],
                recipe['prior'],recipe['backoff'],torch.tensor(frequency[:,None]),recipe['higher_order']).numpy().ravel()
            z={k:v.ravel() for k,v in raw.items()}
            z['hist']=effective(z,'hist',alpha);z['diff2']=effective(z,'diff2',alpha)
            np.testing.assert_allclose(score(z,canonical_method(recipe),frequency),expected,rtol=1e-6,atol=1e-6)

    def test_amplified_bank_changes_only_historical_column(self):
        history=np.array([[1,2,3],[4,5,6]],np.float32)
        user=np.array([[0,10,2],[1,0,9]],np.float32)
        amplified=-history.copy();arrays={'hist':history,'user_freq':user}
        ref=ScoreBank(arrays,amplified,list(arrays),'reference')
        amp=ScoreBank(arrays,amplified,list(arrays),'amplified')
        np.testing.assert_array_equal(ref.query(1)['user_freq'],amp.query(1)['user_freq'])
        np.testing.assert_array_equal(ref.query(1)['hist'],[4,5,6])
        np.testing.assert_array_equal(amp.query(1)['hist'],[-4,-5,-6])
        score(amp.query(1),dict(kind='weighted',weights={'hist':1.,'user_freq':.5}),np.ones(3))
        np.testing.assert_array_equal(history,[[1,2,3],[4,5,6]])
        np.testing.assert_array_equal(user,[[0,10,2],[1,0,9]])


if __name__=='__main__':unittest.main()
