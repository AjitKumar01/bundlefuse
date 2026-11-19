"""Train the paper version of RaMen (ramen_paper.py) on a dataset written by export.py; save per-query test metrics.

Runs in the RaMen environment, from the RaMen repository:
cd ../baselines/RaMen_repo && .venv/bin/python ../../bundlefuse/experiments/ramen_retail/ramen_paper_runner.py \
    --dataset electronic_clean --L1 1 --L2 1 --N 2 --Z 1 --H 4 --beta .5 --gamma .5 --lambda1 .1 --seed 2306 \
    --out ../../bundlefuse/results/ramen_paper
Add --check to verify, on one batch, that every part of the model is trained and that test scores are deterministic.

Training: Adam, lr 1e-3, batch 1024, weight decay 1e-5 (paper Sec. 4.1.3). The paper gives no epoch count; validation
runs after every epoch, training stops after --patience epochs without a new best, and the best epoch is the one
where validation Recall@20 and NDCG@20 both improved last (the released train.py rule). Test is scored once.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import setup_seed
from utility import Datasets
import ramen_paper
from ramen_runner import DEFAULTS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SPACE = dict(L1=range(1, 6), L2=range(1, 6), N=range(1, 6), Z=range(1, 6), H=(4, 8, 16, 32, 64),
             beta=np.round(np.arange(.1, 1, .1), 1), gamma=np.round(np.arange(.1, 1, .1), 1), lambda1=np.round(np.arange(.1, 1, .1), 1))


def load(a):
    conf = yaml.safe_load(open('config.yaml'))['electronic']
    conf.update(DEFAULTS, dataset=a.dataset, lr=1e-3, batch_size_train=1024, batch_size_test=1024, seed=a.seed,
                device=DEVICE, n_layer_gat=a.N, hyper_num=a.H, L1=a.L1, L2=a.L2, Z=a.Z, beta=a.beta,
                gamma=a.gamma, lambda1=a.lambda1)
    setup_seed(a.seed);ds = Datasets(conf)
    conf.update(num_users=ds.num_users, num_bundles=ds.num_bundles, num_items=ds.num_items, l2_reg=1e-5, embedding_size=64,
                num_layers=1)
    return conf, ds


def per_query(model, loader, k=20):
    """Recall@20 and NDCG@20 for every query (as ramen_runner.per_query, on the model's device)."""
    model.eval();out = [];disc = 1/torch.log2(torch.arange(2, k + 2, dtype=torch.float, device=DEVICE))
    with torch.no_grad():
        for index, b_i_input, seq, gt in loader:
            index, b_i_input, seq, gt = (x.to(DEVICE) for x in (index, b_i_input, seq, gt))
            top = (model.evaluate(None, (index, b_i_input, seq)) - 1e8*b_i_input).topk(k).indices;hit = gt.gather(1, top)
            pos = gt.sum(1);idcg = torch.stack([disc[:int(min(q, k))].sum() for q in pos])
            out += torch.stack([hit.sum(1)/pos, (hit*disc).sum(1)/idcg], 1).tolist()
    return np.asarray(out)


def check(model, ds):
    """Every registered parameter gets a gradient; the paper's parts are registered; test scores are deterministic."""
    names = dict(model.named_parameters())
    for part in ('bundle_encode.attn_encode.0.w_q.weight', 'encoder.mlp_multimodal.weight', 'encoder.iui_gat_conv.convs.0.att',
                 'encoder.v_hyper', 'encoder.t_hyper', 'decoder.items_embedding', 'decoder.item_embeddings'):
        assert part in names, f'{part} not registered'
    model.train();model.zero_grad();model([x.to(DEVICE) for x in next(iter(ds.train_loader))])['loss'].backward()
    dead = [n for n, p in names.items() if p.grad is None or p.grad.abs().sum() == 0]
    model.eval();b = next(iter(ds.test_loader))
    with torch.no_grad():
        b = [x.to(DEVICE) for x in b[:3]];s1, s2 = model.evaluate(None, b), model.evaluate(None, b)
    diff = (s1 - s2).abs().max().item()
    print(f'parameters {sum(p.numel() for p in names.values()):,} | without gradient: {dead or "none"} | '
          f'test scores identical: {torch.equal(s1, s2)} (max difference {diff:.2e}, scores up to {s1.abs().max().item():.2f})', flush=True)
    assert not dead and diff < 1e-4  # GPU scatter-add sums in a varying order: differences of ~1e-6 remain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True);p.add_argument('--seed', type=int, default=2306)
    for k in ('L1', 'L2', 'N', 'Z', 'H'):p.add_argument(f'--{k}', type=int, required=True)
    for k in ('beta', 'gamma', 'lambda1'):p.add_argument(f'--{k}', type=float, required=True)
    p.add_argument('--epochs', type=int, default=1000);p.add_argument('--patience', type=int, default=60)
    p.add_argument('--threads', type=int, default=2);p.add_argument('--out');p.add_argument('--check', action='store_true')
    a = p.parse_args();torch.set_num_threads(a.threads)
    conf, ds = load(a);model = ramen_paper.build(conf, ds.graphs, ds.features).to(DEVICE)
    if a.check:return check(model, ds)
    tag = f'{a.dataset}/L1{a.L1}_L2{a.L2}_N{a.N}_Z{a.Z}_H{a.H}_b{a.beta:g}_g{a.gamma:g}_l{a.lambda1:g}_seed{a.seed}'
    path = Path(a.out)/tag
    if (path/'result.json').exists():return
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    setup_seed(a.seed);best = (0, 0);state = None;chosen = 0;history = [];start = time.time()
    for epoch in range(a.epochs):
        model.train(True)
        for batch in ds.train_loader:
            optimizer.zero_grad();model([x.to(DEVICE) for x in batch])['loss'].backward();optimizer.step()
        val = per_query(model, ds.val_loader).mean(0);history.append(val.tolist())
        if val[0] > best[0] and val[1] > best[1]:best = (val[0], val[1]);state = copy.deepcopy(model.state_dict());chosen = epoch
        if epoch - chosen >= a.patience:break
    model.load_state_dict(state);test = per_query(model, ds.test_loader)
    path.mkdir(parents=True, exist_ok=True)
    (path/'result.json').write_text(json.dumps(dict(dataset=a.dataset, seed=a.seed, config={k: getattr(a, k) for k in SPACE},
        epoch=chosen, epochs_run=len(history), validation=list(best), test=test.mean(0).tolist(), queries=test.tolist(),
        history=history, seconds=time.time() - start)))
    print(f'{tag} epoch {chosen}/{len(history)} val R20 {best[0]:.4f} N20 {best[1]:.4f} test R20 {test[:, 0].mean():.4f} '
          f'N20 {test[:, 1].mean():.4f} ({time.time() - start:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
