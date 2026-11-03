import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import evaluator
import inference
import validate
from bundlefuse import datasets
from bundlefuse.baselines import BASELINES
from bundlefuse.evaluation import sparse_pairs
from bundlefuse.metrics import top_items
from bundlefuse.model import BundleFuse, BundleFuseConfig
from bundlefuse.splits import ProvidedSplits


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    from bundlefuse.datasets import base
    root=tmp_path/'data'; directory=root/'bundlemllm/datasets/pog_dense';directory.mkdir(parents=True)
    rng=np.random.default_rng(19); n=60
    train={i:rng.choice(n,size=5,replace=False).tolist() for i in range(100)}
    train[100]=train[0]  # Provided mode must retain duplicate bundles with distinct IDs.
    files={'train':train,'valid_input':{210:[0,1],203:[2,3,4],220:[5]},
           'valid_gt':{210:[12],203:[13],220:[14]},
           'test_input':{310:[6,7,8],303:[9],320:[10,11]},
           'test_gt':{310:[15,16],303:[17],320:[18]}}
    for name,rows in files.items():
        (directory/f'bi_{name}.txt').write_text(''.join(','.join(map(str,[i]+v))+'\n' for i,v in rows.items()))
    (directory/'count.json').write_text(json.dumps({'#I':n}))
    (directory/'item_info.json').write_text(json.dumps({str(i):{'cate':str(i%6)} for i in range(n)}))
    np.save(directory/'title_emb.npy',rng.normal(size=(n,8)).astype(np.float32))
    monkeypatch.setattr(base,'DATA_ROOT',str(root))
    return directory


def test_public_provided_and_resplit_preserve_explicit_protocol(catalogue):
    original=ProvidedSplits(catalogue,60)
    ds=datasets.load('pog_dense')
    assert len(ds.train)==101 and ds.train==list(original.train.values())
    assert ds.val_pairs==original.pairs('valid') and ds.test_pairs==original.pairs('test')
    assert ds.val_ids==[203,210,220] and ds.test_ids==[303,310,320]
    assert ds.metadata['protocol']=='provided' and ds.metadata['full_validation'] and ds.metadata['full_test']
    capped=datasets.load('pog_dense',val_n=1,test_n=1)
    assert len(capped.test_pairs)==1 and not capped.metadata['full_test']
    assert datasets.load('pog_dense',val_n=0,test_n=0).test_pairs==ds.test_pairs
    split=datasets.load('pog_dense',protocol_mode='resplit',split_seed=2,mask_frac=.25)
    assert split.metadata['protocol']=='resplit' and split.metadata['split_seed']==2
    assert split.metadata['train_sha256']!=ds.metadata['train_sha256']
    with pytest.raises(ValueError): datasets.load('pog_dense',mask_frac=.25)
    with pytest.raises(ValueError): datasets.load('pog_dense',split_seed=2)
    with pytest.raises(ValueError): datasets.load('pog_dense',test_n=-1)


@pytest.mark.parametrize('mode',['fashion','music','grouped'])
def test_selected_reference_and_checkpoint_all_signal_families(tmp_path,mode):
    rng=np.random.default_rng(3);n=40
    train=[rng.choice(n,size=5,replace=False).tolist() for _ in range(50)]
    valid=[([0,1],[2]),([3,4],[5]),([6],[7])]
    features={'audio' if mode=='music' else 'content':rng.normal(size=(n,8)).astype(np.float32)}
    cfg=BundleFuseConfig(n_items=n,svd_dim=8,n_soft_clusters=5,content_rank=4,beta_clique=.75,fit_rounds=1,validation_chunk=2)
    model=BundleFuse(cfg).fit(train,valid,features=features,cat_labels=np.arange(n)%5,music=mode=='music',
        alb_labels=np.arange(n)%7 if mode=='music' else None,
        group_labels={'household':(np.arange(n)%7,.5)} if mode=='grouped' else None,
        aux_incidences={'aux_test':(sp.csr_matrix(rng.integers(0,2,size=(5,n)).astype(np.float32)),True)},
        user_items={0:[0,1,1,3],1:[5,6]},val_users=[0,1,None])
    bundles=[[0,1],[4],[5,6,7]];users=[0,None,1]
    for alpha in (0.,.5,1.):
        for gamma in (0.,.2):
            model.w={key:.25 for key in model.SIG}
            model.prior_keys=['content','cat'];model.best_eb=(.5,1.3);model.best_ho=(alpha,gamma)
            expected=model.score_batch(bundles,users,reference=True)
            actual=model.score_batch(bundles,users)
            np.testing.assert_array_equal(expected.view(np.uint32),actual.view(np.uint32))
    before=model.score_batch(bundles,users).copy();c=model.graph.C
    path=tmp_path/'m.pt';model.save(path)
    assert model.graph.C is c
    loaded=BundleFuse.load(path)
    assert loaded.graph.C is None and loaded.graph.PPMI is None
    np.testing.assert_array_equal(before.view(np.uint32),loaded.score_batch(bundles,users).view(np.uint32))


def test_zero_weight_expensive_signals_are_skipped(catalogue,monkeypatch):
    ds=datasets.load('pog_dense');cfg=datasets.get('pog_dense').config(60,svd_dim=8);cfg.fit_rounds=1
    model=BundleFuse(cfg).fit(ds.train,ds.val_pairs,**ds.fit_kwargs())
    model.w={'pop':1.};model.best_eb=(0.,1.);model.best_ho=(0.,0.)
    def fail(*args,**kwargs):raise AssertionError('Unselected expensive operation ran')
    monkeypatch.setattr(model.graph,'propagate',fail)
    monkeypatch.setattr(model.graph,'score_rao_concordance',fail)
    model.score([0,1])


def test_configured_rounds_and_seed_cap_reach_the_computation(catalogue,monkeypatch):
    import bundlefuse.model as module
    from bundlefuse.fusion import fit_fusion
    seen=[]
    def capture(*args,**kwargs):
        seen.append(kwargs['rounds'])
        return fit_fusion(*args,**kwargs)
    monkeypatch.setattr(module,'fit_fusion',capture)
    ds=datasets.load('pog_dense')
    cfg=BundleFuseConfig(n_items=60,svd_dim=8,n_soft_clusters=5,seed_cap=1,fit_rounds=1)
    model=BundleFuse(cfg).fit(ds.train,ds.val_pairs,**ds.fit_kwargs())
    assert seen==[1]
    np.testing.assert_array_equal(model.score([3,1,2]),model.score([1]))
    assert not set(model.predict([3,1,2],20)).intersection([1,2,3])
    assert model.data_spec==ds.metadata
    ds.metadata['protocol']='changed'
    assert model.data_spec['protocol']=='provided'


def test_validation_cache_cleans_up_when_scoring_fails(monkeypatch):
    import bundlefuse.validation_cache as module
    from tempfile import TemporaryDirectory
    paths=[]
    def capture(*args,**kwargs):
        temporary=TemporaryDirectory(*args,**kwargs);paths.append(Path(temporary.name))
        return temporary
    monkeypatch.setattr(module,'TemporaryDirectory',capture)
    def fail(seeds,users):raise RuntimeError('scoring failed')
    with pytest.raises(RuntimeError,match='scoring failed'):
        module.ValidationCache([([0],[1])],None,fail,3)
    assert paths and all(not path.exists() for path in paths)


def test_atomic_save_failure_preserves_state(catalogue,tmp_path,monkeypatch):
    import bundlefuse.model as module
    ds=datasets.load('pog_dense');cfg=datasets.get('pog_dense').config(60,svd_dim=8);cfg.fit_rounds=1
    model=BundleFuse(cfg).fit(ds.train,ds.val_pairs,**ds.fit_kwargs())
    before=model.score([0,1]).copy();original_c=model.graph.C
    path=tmp_path/'saved.pt';path.write_bytes(b'previous checkpoint')
    def fail(*args,**kwargs):raise OSError('simulated failed write')
    monkeypatch.setattr(module.pickle,'dump',fail)
    with pytest.raises(OSError):model.save(path)
    assert path.read_bytes()==b'previous checkpoint' and model.graph.C is original_c
    np.testing.assert_array_equal(before,model.score([0,1]))
    assert not list(tmp_path.glob('.bundlefuse_*'))


@pytest.mark.parametrize('name',list(BASELINES))
def test_all_baseline_interfaces_receive_dataset_features(catalogue,name):
    ds=datasets.load('pog_dense');model=BASELINES[name](ds.n_items).fit(ds.train,ds.val_pairs,**ds.fit_kwargs())
    assert model.score([0,1]).shape==(60,)
    assert model.score_batch([[0,1],[3]]).shape==(60,2)
    assert not set(model.predict([0,1],10)).intersection([0,1])


def test_prediction_uses_evaluation_tie_order():
    scores=np.random.default_rng(0).integers(0,3,60).astype(np.float32)
    models=[BundleFuse(BundleFuseConfig(n_items=60)),BASELINES['popularity'](60)]
    for model in models:
        model.score=lambda seeds,user=None:scores.copy()
        assert model.predict([0],20)==top_items(scores,[0],20).tolist()
        assert model.predict([0],0)==[]


def test_seed_stress_preserves_bundle_membership():
    pairs=[([3,1,2],[8]),([4],[9])]
    reduced=sparse_pairs(pairs,1)
    assert reduced==[([1],[8,2,3]),([4],[9])]
    for before,after in zip(pairs,reduced):
        assert set(before[0]+before[1])==set(after[0]+after[1])


@pytest.mark.parametrize('name',['bundlefuse','contentknn','popularity'])
def test_fit_save_reload_report_cli_contract(catalogue,tmp_path,name):
    path=tmp_path/f'{name}.pt';out=tmp_path/f'{name}_report.json'
    evaluator.main(['pog_dense','--model',name,'--save',str(path),'--svd-dim','8',
                    '--fit-rounds','1','--seed-cap','2','--chunk','2'])
    saved=json.loads(Path(str(path)+'.json').read_text());assert saved['checkpoint']['score_bit_identity']
    model=BundleFuse.load(path)
    inference.main(['--ckpt',str(path),'report','pog_dense','--model',name,'--out',str(out),'--chunk','2'])
    result=json.loads(out.read_text())
    assert result['overall']==saved['test']
    assert result['data']['protocol']=='provided' and result['test_query_ids']==[303,310,320]
    if name=='bundlefuse':
        assert result['weights']==model.w and result['signals']==model.SIG
        assert result['kappa_f']==model.best_eb[0]
        assert model.cfg.fit_rounds==1 and model.cfg.seed_cap==2
    inference.main(['--ckpt',str(path),'predict','pog_dense','--model',name,'--seeds','0,1','--k','3'])
    with pytest.raises(ValueError,match='protocol'):
        inference.main(['--ckpt',str(path),'evaluate','pog_dense','--model',name,'--protocol','resplit'])
    (catalogue/'item_info.json').write_text((catalogue/'item_info.json').read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        inference.main(['--ckpt',str(path),'evaluate','pog_dense','--model',name])


def test_verification_cli_checks_all_queries(catalogue,tmp_path):
    config=tmp_path/'config.json';config.write_text(json.dumps({'svd_dim':8,'fit_rounds':1,'n_soft_clusters':5}))
    output=tmp_path/'verified.json'
    result=validate.main(['--datasets','pog_dense','--config',str(config),'--chunk','2','--out',str(output),
                          '--checkpoint-dir',str(tmp_path/'permanent')])
    assert result['status']=='complete' and result['runs'][0]['queries_verified']==3
    assert result['runs'][0]['checkpoint_score_bit_identity']
    checkpoint=result['runs'][0]['checkpoint']
    assert checkpoint['retained'] and Path(checkpoint['path']).is_file()
    loaded=BundleFuse.load(checkpoint['path'])
    assert loaded.fitted_ and loaded.score([0,1]).shape==(60,)
