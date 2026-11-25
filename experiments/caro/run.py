"""CaRo (Nguyen et al., SIGIR-AP 2025; github.com/L2R-UET/CaRo, commit db6cbad) on our clean e-commerce splits.

Data: the clean CLHE folders (baselines/CLHE/datasets/<d>_clean_blip: same splits, BLIP features and the user-item
file built from training bundles only, as for our CLHE reruns), plus what CaRo needs in addition: item_cate.txt (item,
category) from the category labels TRIAD uses (dials triad.data), and '#C' in count.json. Written to
baselines/CaRo/datasets/<d>_clean_blip; the other files are symbolic links.

Settings: the authors' run.sh for Electronic and Food (item_augment FN, bundle_augment ID, bundle_ratio .5,
bundle_cl_temp .01, bundle_cl_alpha .5, cl_temp .5, cl_alpha 2, cate_filter False) and their config.yaml e-commerce
block (lr 1e-4, weight decay 1e-5, embedding 64, batch 256, 50 epochs); their release has no Clothing block, so
Clothing uses the same (the Electronic and Food blocks are identical). Model selection as in our CLHE reruns: validation
Recall@20 and NDCG@20 after every epoch, the epoch kept when both improve (CaRo's own script checks every 5 epochs at
top 10). Test scored once, per query, seeds masked. Three runs (random seeds 2023, 0, 1). CPU, data loaders without
worker processes.

Output: results/caro/<d>/CaRo_seed<s>.json with per-query [bundle id, Recall@20, NDCG@20].
cd ../bundlefuse && VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=.:../dials .venv/bin/python -m experiments.caro.run --datasets electronic
"""
import argparse, copy, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CARO = ROOT/'baselines'/'CaRo';CLHE_DATA = ROOT/'baselines'/'CLHE'/'datasets'
SEEDS = (2023, 0, 1)
FLAGS = dict(item_augment='FN', bundle_augment='ID', bundle_ratio=.5, bundle_cl_temp=.01, bundle_cl_alpha=.5,
             cl_temp=.5, cl_alpha=2., cate_filter=False, dropout_rate=.2, noise_weight=.02, attention='', trans_layer=1,
             num_token=200)


def prepare(name):
    """baselines/CaRo/datasets/<name>_clean_blip: links to the clean CLHE folder + item_cate.txt + count.json with #C."""
    import triad
    from triad import data
    src = CLHE_DATA/f'{name}_clean_blip';dst = CARO/'datasets'/f'{name}_clean_blip'
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != 'count.json' and not (dst/f.name).exists():
            (dst/f.name).symlink_to(f.resolve())
    labels = data.load(name, ROOT/'dials'/'cache').levels[0]
    (dst/'item_cate.txt').write_text(''.join(f'{i}, {int(c)}\n' for i, c in enumerate(labels)))
    count = json.loads((src/'count.json').read_text());count['#C'] = int(max(labels)) + 1
    (dst/'count.json').write_text(json.dumps(count))
    return dst


def train(dataset, seed, out, epochs=50, log=print):
    path = Path(out)/dataset/f'CaRo_seed{seed}.json'
    if path.exists():
        return json.loads(path.read_text())
    os.chdir(CARO);sys.path.insert(0, str(CARO))
    import numpy as np, torch
    from torch.utils.data import DataLoader
    from utility import Datasets
    from train import setup_seed
    import models
    device = torch.device('cpu')

    def per_query(model, loader, k=20):
        model.eval();rows = []
        disc = 1/torch.log2(torch.arange(2, k + 2, dtype=torch.float))
        with torch.no_grad():
            for index, b_i_input, seq, gt in loader:
                pred = model.evaluate(None, (index, b_i_input, seq)) - 1e8*b_i_input
                hit = gt.gather(1, pred.topk(k).indices);pos = gt.sum(1)
                idcg = torch.stack([disc[:int(min(p, k))].sum() for p in pos])
                r = torch.stack([hit.sum(1)/pos, (hit*disc).sum(1)/idcg], 1)
                rows += [[int(i), float(a), float(b)] for i, (a, b) in zip(index.tolist(), r.tolist())]
        return rows

    conf = dict(data_path='./datasets', dataset=dataset, model='CaRo', batch_size_train=256, batch_size_test=1024,
                topk=[20], neg_num=1, embedding_sizes=[64], num_layerss=[1], lrs=[1e-4], l2_regs=[1e-5], epochs=epochs,
                test_interval=1, gpu='0', info='', lr=1e-4, reg=1e-5, seed=seed, epoch=-1, device=device, **FLAGS)
    setup_seed(seed);ds = Datasets(conf)
    ds.train_loader = DataLoader(ds.bundle_train_data, batch_size=256, shuffle=True, num_workers=0)
    ds.val_loader = DataLoader(ds.bundle_val_data, batch_size=1024, shuffle=False, num_workers=0)
    ds.test_loader = DataLoader(ds.bundle_test_data, batch_size=1024, shuffle=False, num_workers=0)
    conf.update(num_users=ds.num_users, num_bundles=ds.num_bundles, num_items=ds.num_items, num_cates=ds.num_cates,
                l2_reg=1e-5, embedding_size=64, num_layers=1)
    model = models.CaRo(conf, ds.graphs, ds.features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    setup_seed(seed);best = (0., 0.);state = copy.deepcopy(model.state_dict());chosen = -1;history = [];start = time.time()
    for epoch in range(epochs):
        model.train(True)
        for batch in ds.train_loader:
            opt.zero_grad();model(list(batch))['loss'].backward(retain_graph=True);opt.step()
        val = np.asarray([q[1:] for q in per_query(model, ds.val_loader)]).mean(0);history.append(val.tolist())
        if val[0] > best[0] and val[1] > best[1]:
            best = (float(val[0]), float(val[1]));state = copy.deepcopy(model.state_dict());chosen = epoch
        log(f'{dataset} CaRo seed {seed} epoch {epoch} val R20 {val[0]:.4f} N20 {val[1]:.4f} best epoch {chosen} '
            f'{time.time() - start:.0f}s')
    model.load_state_dict(state);test = per_query(model, ds.test_loader)
    t = np.asarray([q[1:] for q in test]).mean(0)
    result = dict(dataset=dataset, model='CaRo', seed=seed, epoch=chosen, validation=best, test=t.tolist(), queries=test,
                  history=history, seconds=time.time() - start,
                  parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
                  settings=dict(FLAGS, lr=1e-4, weight_decay=1e-5, epochs=epochs, caro_commit='db6cbad'))
    path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(result))
    log(f'RESULT {dataset} CaRo seed {seed} epoch {chosen} test R20 {t[0]:.4f} N20 {t[1]:.4f}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', required=True);p.add_argument('--out', default='results/caro')
    p.add_argument('--epochs', type=int, default=50);p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.out).resolve()
    for name in a.datasets:
        prepare(name)
        for seed in a.seeds:
            train(f'{name}_clean_blip', seed, out, a.epochs, log=lambda m: print(m, flush=True))


if __name__ == '__main__':
    main()
