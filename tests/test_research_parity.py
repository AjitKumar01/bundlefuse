"""Independent historical recipe checks, beyond checkpoint or internal-path parity."""
from pathlib import Path
import runpy
import sys
from types import ModuleType

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from bundlefuse.model import BundleFuse, BundleFuseConfig
from bundlefuse.operators import SimplicialGraphOperator, EmpiricalBayesRouter
from bundlefuse.basket_signals import BasketSignals
from bundlefuse.fusion import fit_basket_fusion
from bundlefuse.metrics import ranking_metrics
from bundlefuse.operators import DualSpectrumOntology, BilateralContentBridge, zcols
from sklearn.cluster import MiniBatchKMeans
import torch


@pytest.mark.parametrize('kind',['retail','bundlerec'])
def test_historical_numpy_selection_signals_and_checkpoint(kind,tmp_path,monkeypatch):
    rng=np.random.default_rng(23);n=40
    train=[rng.choice(n,size=5,replace=False).tolist() for _ in range(70)]
    content=normalize(rng.normal(size=(n,8)).astype(np.float32))
    tfidf=sp.csr_matrix(normalize(np.abs(rng.normal(size=(n,12)))))
    categories=np.arange(n)%6
    data={'kind':kind,'content':content,'tfidf':tfidf}
    if kind=='retail':
        data.update(train_users=[j%5 for j in range(len(train))],n_users=5)
    else:
        pre=sp.csr_matrix(rng.integers(0,2,(4,n)).astype(np.float32));pre=pre.T@pre;pre.setdiag(0)
        data.update(cat_l2=np.arange(n)%3,n_cat_l2=3,train_keys=list(range(len(train))),
                    bundle_users={j:j%4 for j in range(len(train))},
                    bundle_sessions={j:j%7 for j in range(len(train))},pretrain=pre)
    g=SimplicialGraphOperator(train,n,svd_dim=8)
    provider=BasketSignals(g,categories,data,train)
    pairs=[([0,1],[8]),([3],[9]),([4,5,6],[10])];seeds=[s for s,_ in pairs]
    raw,z=provider.score(seeds)
    required=set(provider.keys)|{'pair','diff2_pair','rao'}
    reduced,z_reduced=provider.score(seeds,required)
    for key in required:
        np.testing.assert_array_equal(raw[key].view(np.uint32),reduced[key].view(np.uint32))
        np.testing.assert_array_equal(z[key].view(np.uint32),z_reduced[key].view(np.uint32))
    fixture='retail' if kind=='retail' else 'commerce'
    original=runpy.run_path(str(Path(__file__).parent/'fixtures'/f'{fixture}_selection.py'),init_globals={
        'np':np,'SIG':provider.keys,'Zv':z,'freq':g.freq,'vset':pairs,
        'EmpiricalBayesRouter':EmpiricalBayesRouter,
        'sel_score':lambda values,queries: ranking_metrics(values,queries,(20,))[20][1],
        'GATE_W':2e-4,'GATE_PRUNE':3e-4,'GATE_HO':1e-5,'GATE_EB':1e-4})
    w,prior,eb,ho,vc=fit_basket_fusion(z,provider.keys,pairs,g.freq,kind)
    assert w==original['bw'] and prior==original['prior_ks'] and ho==original['best_ho']
    assert eb==(original['best_kf'],original['frozen_prior_scale'])
    cfg=BundleFuseConfig(n_items=n,svd_dim=8,signal_profile='research')
    model=BundleFuse(cfg).fit(train,pairs,features={'content':content},cat_labels=categories,research_data=data)
    expected=original['apply_eb'](z,ho,eb[0],eb[1])
    np.testing.assert_array_equal(expected.view(np.uint32),model.score_batch(seeds).view(np.uint32))
    path=tmp_path/'research.pt';model.save(path);loaded=BundleFuse.load(path)
    assert loaded.graph is loaded.research_provider.graph and loaded.graph.C is None
    np.testing.assert_array_equal(expected.view(np.uint32),loaded.score_batch(seeds).view(np.uint32))
    # Checkpoint files written before the module rename must still load without shim files.
    old_module = ModuleType('bundlefuse.research_signals')
    old_module.BasketSignals = BasketSignals
    legacy_path = tmp_path/'legacy.pt'
    with monkeypatch.context() as legacy:
        legacy.setitem(sys.modules, old_module.__name__, old_module)
        legacy.setattr(BasketSignals, '__module__', old_module.__name__)
        model.save(legacy_path)
    restored = BundleFuse.load(legacy_path)
    assert restored.graph is restored.research_provider.graph
    np.testing.assert_array_equal(expected.view(np.uint32),restored.score_batch(seeds).view(np.uint32))


def test_basket_profile_rejects_silent_projection_override():
    cfg=BundleFuseConfig(n_items=40,svd_dim=8,signal_profile='research',content_rank=4)
    with pytest.raises(ValueError,match='sparse bridge'):
        BundleFuse(cfg).fit([[0,1,2]],[([0],[1])],cat_labels=np.arange(40)%4,
                            research_data={'kind':'retail'})


@pytest.mark.parametrize('music',[False,True])
def test_fullrank_preprocessing_and_signals_match_archived_reference(music):
    rng=np.random.default_rng(37);n=440
    train=[rng.choice(n,size=5,replace=False).tolist() for _ in range(100)]
    categories=np.arange(n)%8;albums=np.arange(n)%7 if music else None
    names=['audio'] if music else ['content','image','desc']
    features={key:rng.normal(size=(n,8)).astype(np.float32) for key in names}
    features={key:value/(np.linalg.norm(value,axis=1,keepdims=True)+1e-8) for key,value in features.items()}
    cluster=np.concatenate(list(features.values()),1).astype(np.float32)
    pairs=[([0,1],[20]),([2],[21]),([3,4,5],[22])]
    model=BundleFuse(BundleFuseConfig(n_items=n,svd_dim=8,signal_profile='research',fit_rounds=1))
    model.fit(train,pairs,features=features,cat_labels=categories,music=music,alb_labels=albums,
              cluster_feat=cluster,research_data={'kind':'fullrank','features_normalized':True})
    if music:
        labels=MiniBatchKMeans(n_clusters=400,random_state=0,n_init=3,batch_size=4096).fit(features['audio']).labels_
        style=BilateralContentBridge.disentangle(features['audio'],labels,int(labels.max())+1)
        reference_features={'audiostyle':style,'bridge':BilateralContentBridge.build_dense_bridge(style,model.graph,alpha_bridge=.5)}
        rao=style;cluster_for_onto=None
    else:
        reference_features={key:BilateralContentBridge.disentangle(features[source],categories,8)
                            for key,source in [('style','content'),('imgstyle','image'),('descstyle','desc')]}
        combined=np.concatenate(list(reference_features.values()),1).astype(np.float32)
        combined/=np.linalg.norm(combined,axis=1,keepdims=True)+1e-8
        reference_features['bridge']=BilateralContentBridge.build_dense_bridge(combined,model.graph,alpha_bridge=.5)
        cluster_for_onto=cluster/(np.linalg.norm(cluster,axis=1,keepdims=True)+1e-8)
        rao=reference_features['style']
    reference_features['csvd']=model.graph.Ucsvd
    onto=DualSpectrumOntology(train,categories,n,alb_lbl=albums,cluster_feat=cluster_for_onto,n_soft_clusters=400)
    namespace=runpy.run_path(str(Path(__file__).parent/'fixtures/fullrank_reference.py'),init_globals={
        'np':np,'torch':torch,'graph_op':model.graph,'onto_op':onto,'n':n,'freq':model.freq,
        'FEAT_T':{key:torch.as_tensor(value) for key,value in reference_features.items()},
        'cat_lbl':categories,'rao_dense_feat':rao,'MUSIC':music,'DEVICE':torch.device('cpu'),
        'ENCODER':False,'SIG':model.SIG,'zcols':zcols})
    expected_raw,expected_z=namespace['score_all_signals']([s for s,_ in pairs])
    raw,z=model._score_all([s for s,_ in pairs])
    for key in model.SIG+['pair','diff2_pair','rao']:
        np.testing.assert_array_equal(expected_raw[key].numpy().view(np.uint32),raw[key].numpy().view(np.uint32))
        np.testing.assert_array_equal(expected_z[key].numpy().view(np.uint32),z[key].numpy().view(np.uint32))
