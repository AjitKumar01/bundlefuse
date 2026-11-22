"""Small-data fixes for TRIAD on Electronic, Food and Clothing (bundle-budget conditions of experiments.bundle_budget_retail).

Why TRIAD loses with <= 100 bundles: at initialization it is an *untuned* blend (feature similarity + raw counts,
equal weights), while the baseline it is compared with is *tuned* on validation; with few bundles TRIAD overfits before
it catches up (dials/RESULTS.md section 6). Two variants, both trained with TRIAD's frozen code for the model itself:

  residual  s(c|S) = exp(kappa) * b(c|S) + gamma * TRIAD(c|S), gamma = 0 at the start, so epoch 0 IS the tuned simple
            scorer b: the best of the five main simple baselines with the frequency prior, chosen on validation for this
            condition (results/fair_prior/<dataset>[_<condition>].json, same method, setting and lambda). b is
            standardized per query. Training scores of b are cross-fitted: the training bundles are split into 5 folds
            and a bundle's seeds are scored by b fitted on the other four folds, so b never sees the bundle (as
            TRIAD's own leave-one-out statistics). Validation and test use b fitted on all training bundles.
  v2        one model: TRIAD with a principled start (experiments.small_data.triad_v2: per-query scaled terms,
            cosine-style exponents, full-rank content, label score and prior starting at 0); validation picks the
            epoch only (epoch 0 allowed).
  v2_64     v2 with rank-64 content maps (41,368 parameters, as the original TRIAD).
  v2_64a    v2_64 with content = fixed full-d similarity + learned rank-64 correction starting at 0.
  frozen    plain TRIAD with the content maps and encoder frozen at their initial values; only the ~600 relation
            parameters (label score, exponents, theta, prior, gate) are trained.

Both validate at epoch 0 and after every epoch (TRIAD's early stopping: validation NDCG@20, patience 6) and keep the
best epoch, so epoch 0 can be chosen. Seeds 0, 1, 2; metrics as the simple baselines (experiments.neural.evaluation).
  residual_tp  as residual, but the anchor's frequency prior is estimated from training bundles only (leave-one-out
               base rate) and its lambda re-chosen on validation, so validation is clean for model selection.
  residual_cv  as residual_tp, but the anchor (method, setting, lambda) is tuned by 5-fold cross-validation on the
               training bundles (experiments.small_data.anchor_cv): no validation query sets any anchor parameter;
               validation only picks the epoch.
Writes results/small_data/<variant>/<dataset>_<condition>.json with per-query test metrics.

cd ../bundlefuse && PYTHONPATH=.:../dials nice -n 10 .venv/bin/python -m experiments.small_data.residual residual electronic:n100 ...
"""
import triad  # pins BLAS threads before numpy loads
import json, math, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
from triad import train as T
from bundlefuse.provenance import write_json
from experiments.neural.evaluation import summarize
from experiments.fair_prior.baselines import WithPrior, base_rate, setup, EDGES, LAMBDAS
from experiments.bundle_budget_retail.conditions import with_train

OUT = Path('results/small_data');SAVE_TAG = 'full';DIALS = Path(__file__).resolve().parents[3]/'dials';CACHE = DIALS/'cache'
CONFIG = dict(T.CONFIG);FOLDS = 5


def baseline_spec(name, cond):
    f = Path('results/fair_prior')/(f'{name}.json' if cond == 'full' else f'{name}_{cond}.json')
    m = json.loads(f.read_text())['methods'];best = max(m, key=lambda k: m[k]['validation_ndcg20'])
    return best, m[best]['params'], m[best]['lam'], m[best]['validation_ndcg20']


def base_rate_train(f, bundles):
    """The base rate of targets by frequency, from training bundles only (leave-one-out: a bundle's item counts with
    frequency f - 1, as in TRIAD's training), against the catalogue's frequency distribution. Same bins as base_rate."""
    t = np.array([f[g] - 1 for b in bundles for g in set(b)]);n = len(f);out = np.zeros(n, np.float32)
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        items = (f >= lo) & (f < hi)
        if items.any():out[items] = np.log((((t >= lo) & (t < hi)).sum() + 1)/(len(t) + len(EDGES) - 1)) - np.log(items.sum()/n)
    return out


def retune_lambda(ds, spec):
    """Re-choose lambda on validation NDCG@20 for the anchor's method and setting, with the training-only prior."""
    method, params, _, _ = spec;rows = []
    for lam in LAMBDAS:
        sc = make_scorer(ds, ds.train, (method, params, lam, None), 'train')
        rows.append((mean(summarize(sc([s for s, _ in ds.val_pairs]), ds.val_pairs)['queries']), lam))
    v, lam = max(rows);return method, params, lam, v


def make_scorer(ds, bundles, spec, prior='validation'):
    method, params, lam, _ = spec
    model = WithPrior(bundles, ds.n_items, ds.levels, ds.features);model.lam = lam
    model.prior = base_rate(model.f, ds.val_pairs) if prior == 'validation' else base_rate_train(model.f, bundles)
    def score(seeds):
        s = np.asarray(model.scores(method, params, [sorted(set(x)) for x in seeds]), np.float64)
        return ((s - s.mean(1, keepdims=True))/(s.std(1, keepdims=True) + 1e-9)).astype(np.float32)
    return score


class Residual(nn.Module):
    def __init__(self, base_model):
        super().__init__();self.t = base_model
        self.kappa = nn.Parameter(torch.tensor(math.log(3.)));self.gamma = nn.Parameter(torch.tensor(0.))
    def forward(self, seeds, base, bundles=None, cache=None):
        return self.kappa.exp()*base + self.gamma*self.t(seeds, bundles=bundles, cache=cache)


def scores_for(model, variant, seeds, base, bundles=None, cache=None):
    if variant.startswith('residual'):return model(seeds, base, bundles=bundles, cache=cache)
    return model(seeds, bundles=bundles, cache=cache)


@torch.no_grad()
def evaluate(model, variant, pairs, base):
    model.eval();inner = model.t if variant.startswith('residual') else model;cache = inner.cache();out = []
    for a in range(0, len(pairs), 64):
        rows = pairs[a:a + 64];b = torch.as_tensor(base[a:a + 64]) if base is not None else None
        out.extend(summarize(scores_for(model, variant, [s for s, _ in rows], b, cache=cache).numpy(), rows)['queries'])
    return out


def mean(rows, k='ndcg20'):return float(np.mean([r[k] for r in rows]))


@torch.no_grad()
def validation_loss(model, variant, pairs, base):
    """TRIAD's training loss on the validation queries (statistics from all training bundles, as at test time)."""
    model.eval();inner = model.t if variant.startswith('residual') else model;cache = inner.cache();total = 0.
    for a in range(0, len(pairs), 64):
        rows = pairs[a:a + 64];b = torch.as_tensor(base[a:a + 64]) if base is not None else None
        sc = scores_for(model, variant, [s for s, _ in rows], b, cache=cache)
        total += float(T.loss(sc, [list(s) for s, _ in rows], [list(g) for _, g in rows]))*len(rows)
    return total/len(pairs)


def fit(ds, variant, seed, spec):
    torch.manual_seed(seed);rng = np.random.default_rng(seed)
    inner = T.build(ds, CACHE, CONFIG);train = [b for b in ds.train if len(set(b)) >= 2]
    if variant in ('residual', 'residual_tp', 'residual_cv'):
        prior = 'train' if variant in ('residual_tp', 'residual_cv') else 'validation'
        model = Residual(inner);full = make_scorer(ds, ds.train, spec, prior)
        fold = np.random.default_rng(20261121).permutation(len(train)) % FOLDS
        fold_scorers = [make_scorer(ds, [b for b, k in zip(train, fold) if k != j], spec, prior) for j in range(FOLDS)]
        val_base = full([s for s, _ in ds.val_pairs]);test_base = full([s for s, _ in ds.test_pairs])
        fast = inner.fast_parameters() + [model.kappa, model.gamma]
    elif variant in ('v2', 'v2_64', 'v2_64a'):
        from experiments.small_data.triad_v2 import TRIADv2
        model = TRIADv2(ds.features, ds.levels, ds.train, rank=CONFIG['rank'], hidden=CONFIG['hidden'], full=variant == 'v2',
                        anchor_content=variant == 'v2_64a')
        fold = None;val_base = test_base = None;fast = model.fast_parameters()
    elif variant.startswith('v3'):
        from experiments.small_data.triad_v3 import TRIADv3, VARIANTS
        model = TRIADv3(ds.features, ds.levels, ds.train, rank=CONFIG['rank'], hidden=CONFIG['hidden'], **VARIANTS[variant])
        fold = None;val_base = test_base = None;fast = model.fast_parameters()
    else:
        model = inner;fold = None;val_base = test_base = None
        for p in [*inner.encoder.parameters(), inner.query_content, inner.key_content, inner.query_history, inner.key_history]:p.requires_grad_(False)
        fast = inner.fast_parameters()
    shared = [p for p in model.parameters() if p.requires_grad and all(p is not t for t in fast)]
    groups = [dict(params=fast, lr=CONFIG['fast_lr'])] + ([dict(params=shared, lr=CONFIG['shared_lr'])] if shared else [])
    opt = torch.optim.AdamW(groups, weight_decay=CONFIG['weight_decay'])
    best = mean(evaluate(model, variant, ds.val_pairs, val_base));best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch, stale, history = 0, 0, [dict(epoch=0, val=best, val_loss=validation_loss(model, variant, ds.val_pairs, val_base))]
    for epoch in range(1, CONFIG['epochs'] + 1):
        if stale >= CONFIG['patience']:break
        model.train();order = rng.permutation(len(train));losses = []
        for off in range(0, len(order), CONFIG['batch']):
            idx = order[off:off + CONFIG['batch']];rows = [train[i] for i in idx];seeds, truth = T.hide(rows, rng)
            base = None
            if variant.startswith('residual'):
                base = np.zeros((len(rows), ds.n_items), np.float32)
                for j in range(FOLDS):
                    sel = [q for q, i in enumerate(idx) if fold[i] == j]
                    if sel:base[sel] = fold_scorers[j]([seeds[q] for q in sel])
                base = torch.as_tensor(base)
            opt.zero_grad(set_to_none=True)
            value = T.loss(scores_for(model, variant, seeds, base, bundles=[list(set(b)) for b in rows]), seeds, truth)
            value.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.);opt.step()
            losses.append((float(value.detach()), len(rows)))
        v = mean(evaluate(model, variant, ds.val_pairs, val_base))
        history.append(dict(epoch=epoch, val=v, train_loss=sum(a*n for a, n in losses)/sum(n for _, n in losses),
                            val_loss=validation_loss(model, variant, ds.val_pairs, val_base)))
        print(f"  {ds.name} seed {seed} epoch {epoch} train_loss {history[-1]['train_loss']:.4f} val_loss {history[-1]['val_loss']:.4f} val N20 {v:.4f}", flush=True)
        if v > best:best, best_epoch, stale = v, epoch, 0;best_state = {k: x.clone() for k, x in model.state_dict().items()}
        else:stale += 1
    model.load_state_dict(best_state);test = evaluate(model, variant, ds.test_pairs, test_base)
    if variant == 'v2' or variant.startswith('v3'):  # keep the selected model, for reading its parameters (experiments.small_data.readings)
        (OUT/f'{variant}_models').mkdir(parents=True, exist_ok=True);torch.save(best_state, OUT/f'{variant}_models'/f'{ds.name}_{SAVE_TAG}_seed{seed}.pt')
    extra = dict(kappa=float(model.kappa.exp()), gamma=float(model.gamma)) if variant.startswith('residual') else {}
    return dict(seed=seed, best_epoch=best_epoch, validation=best, history=history, test=[[q['recall20'], q['ndcg20']] for q in test], **extra)


def main(variant, jobs):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for job in jobs:
        global SAVE_TAG
        name, cond = job.split(':');SAVE_TAG = cond;ds0, train, _ = setup(name, cond);ds = with_train(ds0, train);spec = baseline_spec(name, cond)
        if variant == 'residual_tp':spec = retune_lambda(ds, spec)
        if variant == 'residual_cv':
            from experiments.small_data.anchor_cv import tune_anchor
            spec = tune_anchor(ds, ds.train, base_rate_train)
            print(name, cond, 'anchor (training CV):', spec[0], spec[1], 'lambda', spec[2], 'cv N20', round(spec[3], 4), flush=True)
        path = OUT/variant/f'{name}_{cond}.json'
        if path.exists():print('exists', path);continue
        started = time.monotonic();runs = [fit(ds, variant, s, spec) for s in (0, 1, 2)]
        q = np.mean([r['test'] for r in runs], 0)
        write_json(path, dict(note=__doc__, dataset=name, condition=cond, baseline=dict(method=spec[0], params=spec[1], lam=spec[2], validation_ndcg20=spec[3]),
                              runs=runs, test=q.mean(0).tolist(), queries=q.tolist(), seconds=time.monotonic() - started))
        print(f"{variant} {name} {cond}: test R/N {np.round(q.mean(0), 4)} best epochs {[r['best_epoch'] for r in runs]} "
              f"{[(round(r.get('kappa', 0), 2), round(r.get('gamma', 0), 3)) for r in runs] if variant.startswith('residual') else ''} ({time.monotonic() - started:.0f}s)", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
