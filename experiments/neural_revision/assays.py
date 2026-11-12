"""Behavior and capacity checks, independent of benchmark labels."""
from itertools import combinations
import numpy as np
import torch
from .model import RelationNetwork,full_catalogue_loss


def run_assays():
    torch.set_num_threads(1);torch.manual_seed(42)
    train=[[0,1,2],[0,1,3],[1,4,5],[0,4,6],[2,5,7]]
    x=np.random.default_rng(7).normal(size=(10,8)).astype('float32')
    cat=np.arange(10)%3
    m=RelationNetwork(x,cat,train,'graph_joint',hidden=8)
    # Exact architecture capability: arbitrary direct and observed triple coefficients,
    # explicitly installed ONLY into this synthetic model, never real-run checkpoints.
    with torch.no_grad():
        m.pair_weights.copy_(torch.linspace(-1,1,len(m.pair_weights)))
        m.joint_weights.copy_(torch.linspace(-.5,.5,len(m.joint_weights)))
    seeds=[[0,1],[1,4],[2]]
    c=m.components(seeds)
    expected_pair=torch.zeros(3,10);expected_joint=torch.zeros(3,10)
    for q,s in enumerate(seeds):
        for j,(a,b) in enumerate(m.edges.T.tolist()):
            if a in s:expected_pair[q,b]+=m.pair_weights[j].detach()
        for j,(key,b,a,t) in enumerate(m.joint_edges.T.tolist()):
            if a in s and t in s:expected_joint[q,b]+=m.joint_weights[j].detach()
    torch.testing.assert_close(c['pair'],expected_pair)
    torch.testing.assert_close(c['joint'],expected_joint)
    torch.testing.assert_close(m([[0,1]]),m([[1,0]]))
    keep=torch.ones(10);keep[0]=0
    hidden=m.components([[0,1]],keep)
    # All direct/group paths involving a hidden candidate or seed must disappear.
    assert hidden['pair'][0,0]==0 and hidden['joint'][0,0]==0
    assert torch.count_nonzero(hidden['joint'])==0
    raw=torch.nn.functional.normalize(torch.tensor(x),dim=1)
    initial=RelationNetwork(x,cat,train,'attributes',hidden=8)
    torch.testing.assert_close(initial.components([[0,1]])['text'][0]/10,raw[[0,1]].mean(0)@raw.T)
    # Can gradients actually fit a small completion problem with several candidates?
    learned=RelationNetwork(x,cat,train,'joint',hidden=8)
    opt=torch.optim.Adam(learned.parameters(),lr=.03)
    seeds=[[0],[4],[5]];targets=[[1],[6],[7]]
    losses=[]
    for _ in range(100):
        opt.zero_grad();loss=full_catalogue_loss(learned(seeds),seeds,targets)
        losses.append(float(loss.detach()));loss.backward();opt.step()
    assert losses[-1]<.1*losses[0],losses
    return dict(sparse_pair_lookup_exact=True,sparse_group_lookup_exact=True,set_order_invariant=True,
                simulated_cold_removes_all_history_paths=True,initial_text_geometry_preserved=True,
                synthetic_loss_initial=losses[0],synthetic_loss_final=losses[-1],
                scope='Synthetic architecture/optimization checks, not evidence of benchmark superiority or recovery of every BundleFuse source.')


if __name__=='__main__':
    print(run_assays())
