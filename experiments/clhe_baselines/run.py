"""The neural baselines of CLHE (Ma et al., WSDM 2024): MultiDAE, MultiVAE, BiLSTM, HyperGraph and Transformer, trained
with CLHE's own code (github.com/Xiaohao-Liu/CLHE, commit 74eb236) on the same data and splits as our CLHE reruns.

CLHE's paper also reports TransformerCL, which is not in the released code and is therefore not run. The released
Transformer baseline needs one adapter to the released encoder interface (released_transformer_call).

Settings. CLHE's config.yaml (embedding 64, learning rate 1e-3, weight decay 1e-5, batch 256, Adam) and its epoch budget
(50 for POG sets, 100 otherwise). The released code reads two settings it never defines: `sigmoid` (False here: the
plain softmax-reconstruction of the standard models) and, for MultiVAE, `anneal`, the weight of the KL term. MultiVAE
is trained with anneal 0 and with 0.2 (the cap of the standard Mult-VAE schedule), and the analysis keeps the value with
the higher mean validation NDCG@20. As in our CLHE runner, every model trains for the full epoch budget, validation
Recall@20 and NDCG@20 are computed after every epoch, and the epoch is kept when both improve. Test is scored once, at
the kept epoch, per query. Three runs per model (random seeds 2023, 0, 1).

Output: <out>/<dataset>/<model>_seed<s>.json with per-query [bundle id, Recall@20, NDCG@20], as for CLHE.

Local (e-commerce, CPU):
  cd ../bundlefuse && .venv/bin/python -m experiments.clhe_baselines.run --clhe ../baselines/CLHE \
      --datasets electronic_clean_blip food_clean_blip clothing_clean_blip --out results/clhe_baselines
Kaggle: kaggle/clhe_baselines.py runs the same `train` function on POG, POG_DENSE, Spotify_sparse and Spotify.
"""
import argparse, copy, json, os, sys, time
from pathlib import Path

MODELS = ('MultiDAE', 'MultiVAE_a0', 'MultiVAE_a0.2', 'BiLSTM', 'HyperGraph', 'Transformer')
SEEDS = (2023, 0, 1)
EPOCHS = dict(pog=50, pog_dense=50, spotify=100, spotify_sparse=100)


def train(clhe, dataset, model_key, seed, out, device, epochs=None, log=print):
    """Train one model with one random seed; write and return the result. Skips finished runs.

    model_key is a model name, or MultiVAE_a<anneal> for MultiVAE with that KL weight."""
    path = Path(out)/dataset/f'{model_key}_seed{seed}.json'
    model_name, anneal = (model_key.split('_a')[0], float(model_key.split('_a')[1])) if '_a' in model_key else (model_key, 0.)
    if path.exists():
        return json.loads(path.read_text())
    os.chdir(clhe);sys.path.insert(0, str(clhe))
    import importlib
    import numpy as np, torch
    from utility import Datasets
    from train import setup_seed

    def per_query(model, loader, k=20):
        model.eval();rows = [];rs = model.propagate()
        disc = 1/torch.log2(torch.arange(2, k + 2, dtype=torch.float, device=device))
        with torch.no_grad():
            for index, b_i_input, seq, gt in loader:
                b_i_input, seq, gt = b_i_input.to(device), seq.to(device), gt.to(device)
                pred = model.evaluate(rs, (index.to(device), b_i_input, seq)) - 1e8*b_i_input
                hit = gt.gather(1, pred.topk(k).indices);pos = gt.sum(1)
                idcg = torch.stack([disc[:int(min(p, k))].sum() for p in pos])
                r = torch.stack([hit.sum(1)/pos, (hit*disc).sum(1)/idcg], 1).cpu()
                rows += [[int(i), float(a), float(b)] for i, (a, b) in zip(index.tolist(), r.tolist())]
        return rows

    epochs = epochs or EPOCHS.get(dataset, 100)
    conf = dict(data_path='./datasets/', topk=[20], neg_num=1, embedding_sizes=[64], num_layerss=[1], lrs=[1e-3],
                l2_regs=[1e-5], test_interval=1, batch_size_train=256, batch_size_test=256, epochs=epochs, gpu='0',
                dataset=dataset, model=model_name, info='', lr=1e-3, reg=1e-5, item_augment='NA', bundle_augment='ID',
                bundle_ratio=.5, dropout_rate=.2, noise_weight=.02, cl_temp=.2, cl_alpha=0, bundle_cl_temp=.2,
                bundle_cl_alpha=0, attention='', trans_layer=1, num_token=200, seed=seed, epoch=-1, device=device,
                sigmoid=False, anneal=anneal)
    setup_seed(seed);ds = Datasets(conf)
    conf.update(num_users=ds.num_users, num_bundles=ds.num_bundles, num_items=ds.num_items, l2_reg=1e-5,
                embedding_size=64, num_layers=1)
    # The released models/__init__.py registers only CLHE, so each baseline is imported from its own file.
    cls = getattr(importlib.import_module(f'models.{model_name}'), model_name)
    if model_name == 'Transformer':
        cls = released_transformer_call(cls, torch)
    model = cls(conf, ds.graphs, ds.features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    setup_seed(seed);best = (0., 0.);state = copy.deepcopy(model.state_dict());chosen = -1;history = [];start = time.time()
    for epoch in range(epochs):
        model.train(True)
        for batch in ds.train_loader:
            opt.zero_grad();model([x.to(device) for x in batch])['loss'].backward();opt.step()
        val = np.asarray([q[1:] for q in per_query(model, ds.val_loader)]).mean(0);history.append(val.tolist())
        if val[0] > best[0] and val[1] > best[1]:
            best = (float(val[0]), float(val[1]));state = copy.deepcopy(model.state_dict());chosen = epoch
        log(f'{dataset} {model_key} seed {seed} epoch {epoch} val R20 {val[0]:.4f} N20 {val[1]:.4f} '
            f'best epoch {chosen} {time.time() - start:.0f}s')
    model.load_state_dict(state);test = per_query(model, ds.test_loader)
    t = np.asarray([q[1:] for q in test]).mean(0)
    result = dict(dataset=dataset, model=model_key, seed=seed, epoch=chosen, validation=best, test=t.tolist(),
                  queries=test, history=history, seconds=time.time() - start,
                  parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
                  settings=dict(sigmoid=False, anneal=anneal if model_name == 'MultiVAE' else None, epochs=epochs,
                                clhe_commit='74eb236'))
    path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(result))
    log(f'RESULT {dataset} {model_key} seed {seed} epoch {chosen} test R20 {t[0]:.4f} N20 {t[1]:.4f}')
    del model, opt, state
    return result


def released_transformer_call(base, torch):
    """The released Transformer baseline calls `self.transformer(idx, x, item_feature)`, an encoder interface the released
    models/utils.py no longer has (its TransformerEncoder takes the token features and a padding mask). This subclass
    makes exactly the call CLHE itself makes (models/CLHE.py): the features of the bundle's item sequence, padded with
    index n, and the mask `seq == n`. Everything else (item encoders, decoder, loss) is the released code."""
    class Transformer(base):
        def encode(self, idx, x, seq):
            if 'id' in self.component:
                item_feature = self.item_embeddings
            else:
                item_feature = (torch.nn.functional.normalize(self.c_encoder(self.content_feature))
                                + torch.nn.functional.normalize(self.t_encoder(self.text_feature)))
            padded = torch.cat([item_feature, item_feature.new_zeros(1, item_feature.shape[1])])
            return self.transformer(padded[seq], seq == self.num_item)

        def forward(self, batch):
            idx, x, seq = batch[:3]
            recon_x = self.decode(self.encode(idx, x, seq))
            loss = -torch.mean(torch.sum(torch.nn.functional.log_softmax(recon_x, 1)*x, -1)/x.sum(dim=-1))
            return {'loss': loss}

        def evaluate(self, propagate_result, batch):
            idx, x, seq = batch[:3]
            return self.decode(self.encode(idx, x, seq))
    return Transformer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clhe', required=True);p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--out', required=True);p.add_argument('--models', nargs='+', default=list(MODELS))
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    clhe, out = Path(a.clhe).resolve(), Path(a.out).resolve()
    for dataset in a.datasets:
        for model_name in a.models:
            for seed in SEEDS:
                train(clhe, dataset, model_name, seed, out, 'cpu', log=lambda m: print(m, flush=True))


if __name__ == '__main__':
    main()
