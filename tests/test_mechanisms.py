"""Research controls must isolate routing and preserve validation multiplicity."""
import numpy as np
import torch
from experiments.mechanisms.selection import StudyValidation, score_recipe, cluster_bootstrap
from experiments.mechanisms.statistics import holm_adjust


def test_mixture_controls_and_cold_boundary():
    z={'cooc':torch.tensor([[9.],[2.],[3.]]),'cat':torch.tensor([[1.],[4.],[5.]])}
    method=dict(weights={'cooc':1.,'cat':1.},prior=['cat'],higher_order=[0.,0.],backoff=[-1.,2.])
    f=torch.tensor([[0.],[1.],[100.]])
    np.testing.assert_array_equal(score_recipe(z,method,f),2*z['cat'].numpy())
    method['backoff']=[0.,2.]
    np.testing.assert_array_equal(score_recipe(z,method,f),(z['cat']+z['cooc']).numpy())
    method['backoff']=[1.,2.]
    actual=score_recipe(z,method,f)
    assert actual[0,0]==2.
    assert abs(actual[2,0]-8.) < abs(actual[1,0]-6.)


def test_resampled_blocks_preserve_multiplicity_and_labels():
    pairs=[([i],[i+4]) for i in range(4)]
    def blocks():
        for start in (0,2):
            yield start,{'cat':torch.tensor([[start,start+1]],dtype=torch.float32)}
    v=StudyValidation(blocks,pairs,np.ones(1),indices=np.array([0,0,2,3]))
    actual=list(v.blocks())
    assert [a for a,_ in actual]==[0,2]
    np.testing.assert_array_equal(torch.cat([b['cat'] for _,b in actual],1).numpy(),[[0,0,2,3]])
    assert v.pairs==[pairs[i] for i in (0,0,2,3)]


def test_duplicate_bundles_bootstrap_as_cluster_and_holm_reserves_family():
    pairs=[([0],[1]),([1],[0]),([2],[3])]
    ix=cluster_bootstrap(pairs,20261106)
    assert np.count_nonzero(ix==0)==np.count_nonzero(ix==1)
    records=[{'p_value':.001}]
    holm_adjust(records,96)
    assert records[0]['p_holm']==.096
