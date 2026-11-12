"""Frozen-model validation diagnostics; never used in completion optimization."""
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from bundlefuse.metrics import top_items


@torch.inference_mode()
def compare_sources(model,reference,pairs):
    model.eval()
    raw,z=reference._score_all([s for s,_ in pairs])
    targets={k:z[k].numpy().T for k in ['pair','simp','cooc','ppmi','cat','homo','content','bridge','csvd','rao'] if k in z}
    delta=(raw['simp']-raw['pair']).numpy().T
    targets['joint_surplus']=(delta-delta.mean(1,keepdims=True))/(delta.std(1,keepdims=True)+1e-9)
    keys=list(targets)
    rng=np.random.default_rng(712);order=rng.permutation(len(pairs));fitting=set(order[:len(order)//2])
    xs=[];ys=[];fitted=[];alignment={};component_names=None
    for q,(s,_) in enumerate(pairs):
        components=model.components([s]);component_names=list(components)
        components={k:v[0].numpy() for k,v in components.items()}
        score=sum(components.values());components['final']=score
        ids=set(rng.choice(model.n,min(128,model.n),replace=False).tolist())
        for k,target in targets.items():
            if np.ptp(target[q])>0:ids.update(top_items(target[q],s,16).tolist())
        ids=np.array(sorted(ids-set(s)))
        xs.append(np.stack([components[k][ids] for k in component_names],1))
        ys.append(np.stack([targets[k][q,ids] for k in keys],1));fitted.extend([q in fitting]*len(ids))
        allowed=np.ones(model.n,bool);allowed[s]=False
        for component,values in components.items():
            for k,target in targets.items():
                if np.ptp(values[allowed])>0 and np.ptp(target[q,allowed])>0:
                    alignment.setdefault(component,{}).setdefault(k,[]).append(float(spearmanr(values[allowed],target[q,allowed]).statistic))
    x=np.concatenate(xs);y=np.concatenate(ys);fit=np.array(fitted)
    scaler=StandardScaler().fit(x[fit]);x=scaler.transform(x)
    ym=y[fit].mean(0);yscale=np.maximum(y[fit].std(0),1e-8)
    probe=Ridge(alpha=100.).fit(x[fit],(y[fit]-ym)/yscale)
    prediction=probe.predict(x[~fit])*yscale+ym
    r2=r2_score(y[~fit],prediction,multioutput='raw_values')
    return dict(scope='Validation-only linear probes on component scores and descriptive rank alignment. No teacher fitting of the neural model. Different representation dimension from v1; not a controlled capacity comparison.',
                component_names=component_names,
                heldout_probe_r2={k:float(r2[i]) for i,k in enumerate(keys)},
                rank_alignment={c:{k:float(np.mean(v)) for k,v in row.items()} for c,row in alignment.items()},
                warning='High rank agreement for sparse sources can reflect shared support. These probes do not establish causal recovery of BundleFuse mechanisms.')


def target_frequency_metrics(result,pairs,frequency):
    counts={k:[0,0] for k in ['unseen','one_two','three_plus']}
    for (_,truth),record in zip(pairs,result['queries']):
        predicted=set(record['top20'])
        for item in set(truth):
            key='unseen' if frequency[item]==0 else 'one_two' if frequency[item]<=2 else 'three_plus'
            counts[key][0]+=int(item in predicted);counts[key][1]+=1
    return {k:dict(hits=h,targets=n,micro_recall20=h/n if n else None) for k,(h,n) in counts.items()}
