"""TRIAD plus a purchase-log term, on Electronic, Food and Clothing.

Question: does a user-item signal from outside the bundles help TRIAD, used in addition to the training bundles?

The purchase log is BundleRec's user_item_pretrain.csv, a separate log that shares only 2-3 (user, item) pairs with the
bundles and contains no validation or test bundle (results/ramen_retail/export_pretrain.json). BundleRec's
user_item.csv is NOT used: it holds the items of every bundle, test bundles included.

With X the user x item 0/1 matrix of the log (catalogue items only, repeated rows once), P = X^T X off the diagonal:
P_sc is the number of users who bought both s and c. The added term mirrors the bundle graph term:

    U(S, c) = theta_U log(1 + sum_{s in S} P_sc deg_s^-alpha_U deg_c^-beta_U),   deg = P 1

and the gate gets one more output, w_U. Everything else is TRIAD unchanged (same split, features, configuration,
training, early stopping). The log contains no training bundle, so U needs no leave-one-out correction.
Initialization: alpha_U = beta_U = 0, theta_U = 1, w_U = 1 (the new gate row starts at zero weight, bias log(e-1),
like the others). New parameters: alpha_U, beta_U, theta_U and one gate output (16 + 1), 20 in all.

    PYTHONPATH=.:../dials .venv/bin/python -m experiments.triad_purchase.run train [datasets] [seeds]   # default: e-commerce, 0 1 2
    PYTHONPATH=.:../dials .venv/bin/python -m experiments.triad_purchase.run test [datasets]   # paired vs ../dials/runs, clean queries

On POG and POG_DENSE the log is the released ui_full.txt (users' interaction histories, the file CLHE's user-item graph
uses), split into pieces of at most 30 items as for the pseudo-bundles.

Writes results/triad_purchase/runs/<dataset>/seed<s>/ and results/triad_purchase/comparison.json.
"""
import triad  # pins BLAS threads before numpy loads
import collections, csv, json, math, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from triad import data, train as T
from triad.evaluate import evaluate
from triad.model import TRIAD, pair_counts

DIALS = Path(__file__).resolve().parents[3]/'dials';CACHE = DIALS/'cache'
OUT = Path('results/triad_purchase');DATASETS = ('electronic', 'food', 'clothing');SEEDS = (0, 1, 2)


def purchase_histories(name):
    """Each user's distinct catalogue items in the loader's item numbering, users with at least two items.

    Electronic, Food, Clothing: user_item_pretrain.csv. POG, POG_DENSE: the released ui_full.txt, histories over 30
    items split at random into pieces of at most 30 (experiments.bundle_budget.conditions.histories, as for the
    pseudo-bundles)."""
    if name in ('pog', 'pog_dense'):
        from experiments.bundle_budget.conditions import histories
        return histories(name)
    root = data.DATA_ROOT/'BundleRec'/'dataset'/name
    with open(root/'bundle_item.csv') as f:
        r = csv.reader(f);next(r);items = sorted({int(i) for _, i in r})
    index = {it: k for k, it in enumerate(items)};users = collections.defaultdict(set)
    with open(root/'user_item_pretrain.csv') as f:
        r = csv.reader(f);next(r)
        for u, i, _ in r:
            if int(i) in index:users[u].add(index[int(i)])
    return [sorted(v) for v in users.values() if len(v) >= 2]


class TRIADPurchase(TRIAD):
    def __init__(self, features, levels, train, purchase, rank=64, hidden=32):
        super().__init__(features, levels, train, rank=rank, hidden=hidden)
        source, target, counts, _, pointer = pair_counts(purchase, self.n)
        mass = np.bincount(source, weights=counts, minlength=self.n)  # deg = P 1
        for k, v in dict(p_pointer=torch.as_tensor(pointer, dtype=torch.long), p_target=torch.as_tensor(target, dtype=torch.long),
                         p_counts=torch.as_tensor(counts, dtype=torch.float32),
                         p_log_degree=torch.as_tensor(mass, dtype=torch.float32).clamp_min(1).log()).items():
            self.register_buffer(k, v, persistent=False)
        self.p_exponents = nn.Parameter(torch.zeros(2))
        self.p_theta = nn.Parameter(torch.tensor(1.))
        # Gate: keep TRIAD's first layer, widen the output by one (purchase), all outputs start at softplus = 1.
        last = nn.Linear(16, 3 + self.levels);nn.init.zeros_(last.weight);nn.init.constant_(last.bias, math.log(math.e - 1))
        self.gate[2] = last

    def fast_parameters(self):
        return [*super().fast_parameters()[:-len(list(self.gate.parameters()))], self.p_exponents, self.p_theta, *self.gate.parameters()]

    def purchase_scores(self, seeds):
        q = len(seeds);rows = torch.cat(seeds);query = torch.cat([torch.full((len(s),), i) for i, s in enumerate(seeds)])
        start = self.p_pointer[rows];length = self.p_pointer[rows + 1] - start;offsets = torch.cumsum(length, 0) - length
        edge = torch.repeat_interleave(start - offsets, length) + torch.arange(int(length.sum()))
        query = torch.repeat_interleave(query, length);source = torch.repeat_interleave(rows, length);target = self.p_target[edge]
        a, b = self.p_exponents
        w = self.p_counts[edge]*torch.exp(-a*self.p_log_degree[source] - b*self.p_log_degree[target])
        return self.p_theta*torch.zeros(q*self.n).scatter_add(0, query*self.n + target, w).view(q, self.n).log1p()

    def forward(self, seeds, bundles=None, cache=None):
        if cache is not None and bundles is not None:raise ValueError('The cache is for evaluation only')
        seeds = [torch.as_tensor(sorted(set(s)), dtype=torch.long) for s in seeds]
        if bundles is not None:bundles = [torch.as_tensor(sorted(set(b)), dtype=torch.long) for b in bundles]
        parts = ([self.content_scores(seeds, bundles, cache)] + [self.label_scores(i, seeds) for i in range(self.levels)]
                 + [self.graph_scores(seeds, bundles, cache), self.purchase_scores(seeds)])
        w = self.gate_weights(seeds, bundles)
        return sum(w[:, j, None]*v for j, v in enumerate(parts))


_purchase = {}


def build(ds, cache, config=T.CONFIG):
    if ds.name not in _purchase:_purchase[ds.name] = purchase_histories(ds.name)
    return TRIADPurchase(ds.features, ds.levels, ds.train, _purchase[ds.name], rank=config['rank'], hidden=config['hidden'])


T.build = build  # fit() and load() call train.build


def run_train(names=DATASETS, seeds=SEEDS):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for name in names:
        ds = data.load(name, CACHE)
        for s in seeds:T.fit(ds, s, OUT/'runs'/name/f'seed{s}', CACHE);T.log(f'{name} seed {s} COMPLETE')


def paired(a, b, rng, draws=10000):
    d = np.asarray(a) - np.asarray(b);n = len(d)
    boot = d[rng.integers(0, n, (draws, n))].mean(1)
    flips = (d[None, :]*rng.choice([-1, 1], (draws, n))).mean(1)
    return dict(delta=float(d.mean()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                p=float((1 + (np.abs(flips) >= abs(d.mean())).sum())/(draws + 1)))


@torch.no_grad()
def run_test(names=DATASETS):
    torch.set_num_threads(4);rng = np.random.default_rng(0)
    out = OUT/'comparison.json';report = json.loads(out.read_text())['datasets'] if out.exists() else {}
    for name in names:
        ds = data.load(name, CACHE);new, old, learned = [], [], []
        for s in SEEDS:
            run = OUT/'runs'/name/f'seed{s}';m = T.load(ds, run, CACHE);r = evaluate(m, ds.test_pairs)
            (run/'test.json').write_text(json.dumps(r));new.append(r)
            old.append(json.loads((DIALS/'runs'/name/f'seed{s}'/'test.json').read_text()))
            S = [torch.as_tensor(sorted(set(x))) for x, _ in ds.val_pairs[:512]]
            w = m.gate_weights(S, None).mean(0).tolist()
            learned.append(dict(alpha_U=m.p_exponents[0].item(), beta_U=m.p_exponents[1].item(), theta_U=m.p_theta.item(),
                                gate_mean=w, parameters=m.parameter_count(),
                                best_epoch=json.loads((run/'selected.json').read_text())['best_epoch']))
        row = dict(learned=learned)
        keep = sorted(ds.clean_test);row['test_queries'] = len(keep)  # clean queries; all of them on e-commerce
        for metric in ('ndcg20', 'recall20'):
            a = np.mean([[r['queries'][i][metric] for i in keep] for r in new], 0);b = np.mean([[r['queries'][i][metric] for i in keep] for r in old], 0)
            row[metric] = dict(triad_purchase=float(a.mean()), triad=float(b.mean()), **paired(a, b, rng))
        # How many test targets the purchase graph links to a seed (as in export_pretrain.json).
        p = m.p_pointer;t = m.p_target
        partners = lambda s: set(t[p[s]:p[s + 1]].tolist())
        row['test_targets_linked_by_purchase'] = float(np.mean([any(g in partners(s) for s in set(x)) for x, gt in ds.test_pairs for g in set(gt)]))
        report[name] = row
    ps = sorted((report[n]['ndcg20']['p'], n) for n in names)  # Holm within the datasets tested together
    for k, (pv, n) in enumerate(ps):report[n]['ndcg20']['p_holm'] = min(1., max([pv*(len(ps) - k)] + [report[m]['ndcg20']['p_holm'] for _, m in ps[:k]]))
    OUT.mkdir(parents=True, exist_ok=True);out.write_text(json.dumps(dict(note=__doc__, datasets=report), indent=1))
    for n in names:
        r = report[n];print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r['ndcg20'].items()},
                            'linked', round(r['test_targets_linked_by_purchase'], 3))
        for l in r['learned']:print('   ', {k: (round(v, 3) if isinstance(v, float) else [round(x, 2) for x in v] if isinstance(v, list) else v) for k, v in l.items()})


if __name__ == '__main__':
    names = tuple(a for a in sys.argv[2:] if not a.isdigit()) or DATASETS
    seeds = tuple(int(a) for a in sys.argv[2:] if a.isdigit()) or SEEDS
    run_train(names, seeds) if sys.argv[1] == 'train' else run_test(names)
