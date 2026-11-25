"""DDBC (Tu et al., ICLR 2026) on POG_DENSE: the authors' released checkpoint, scored on our test queries.

What is reused unchanged
  - checkpoint  baselines/ddbc_ckpt/pog_dense.ckpt (huggingface.co/liaialley/DDBC), EMA weights, as in their rec_eval
  - item codes  clhe_token.json, codebooks clhe_weight.npy, item vectors clhe.pt (their Google Drive release)
  - network     baselines/DDBC/models/dit.py (commit e1f4afc)
  - sampler     their ddpm_cache update (diffusion.py _ddpm_caching_update, sample_subs_guidance), 25 steps (checkpoint
                config), then the final argmax pass
Their training file is byte-identical to our bi_train.txt and their valid/test bundles are ours, so the checkpoint has
not seen our validation or test bundles.

What we had to supply
  - flash_attn is CUDA-only. Its two calls are replaced by the same computation in PyTorch: rotary embedding of q and k
    (non-interleaved) and full bidirectional softmax attention with scale 1/sqrt(head dim).
  - The released rec_eval cannot run (tokenizer.collate_fn is None), and two layouts are consistent with the code:
      single: [BOS] seeds, masked slots [EOS], 4 item slots in all (model.length 22; the 4-item training windows)
      split:  [BOS] seeds [EOS] [BOS] as many masked slots as seeds [EOS] (their sample_subs_guidance and position mask)
    The layout is chosen on validation NDCG@20. Each item slot is [BOI] + 4 codes; a masked code may only take a code
    of its own level. At most 3 seeds fit the single layout (2 test queries have 4; their first 3 are used).
  - Ranking the full catalogue. Their evaluator maps each generated code tuple to the nearest of ~100 candidate items
    (cosine on clhe.pt) and takes the union over k samples. We draw 20 samples per query (their top-k of 20), map each
    generated slot to its nearest catalogue item, seeds excluded, and score every item by
        (number of generated slots whose nearest item it is) + (its highest cosine to any generated slot),
    so items generated more often rank first and the rest follow by closeness. A code tuple that is no item's is mapped
    to the sum of its three codebook vectors (their _token_to_feature drops the deduplication code, as here).

Writes results/ddbc/pog_dense.json (validation per layout, test per query, clean test means).
cd bundlefuse && PYTHONPATH=.:../dials .venv/bin/python -m experiments.ddbc.run
"""
import json, sys, time, types, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
SCRATCH_LIB = Path('/private/tmp/claude-501/-Users-ajit-Projects-Career-change/2594a4a1-49ad-4214-9ef7-56481f666b97/scratchpad/pylib')
sys.path.insert(0, str(SCRATCH_LIB))   # omegaconf, einops
from triad import data, evaluate   # noqa: E402

CODE = ROOT/'baselines'/'DDBC'
DRIVE = ROOT/'baselines'/'ddbc_drive'/'datasets'
CKPT = ROOT/'baselines'/'ddbc_ckpt'/'pog_dense.ckpt'
OUT = Path('results/ddbc')
BOS, CB, SLOTS, SAMPLES, STEPS = 0, 128, 4, 20, 25
BOI, EOS = 4*CB+1, 4*CB+2          # 513, 514 (tokenizer.py)
MASK = EOS+1                         # vocab_size; diffusion.py mask_index


# ---- flash_attn replacement ------------------------------------------------------------------------------------------
def rotary_qkv_(qkv, cos, sin):
    """flash_attn.layers.rotary.apply_rotary_emb_qkv_: rotate q and k in place, non-interleaved halves."""
    r = cos.shape[-1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    for j in (0, 1):
        x = qkv[:, :, j]
        x1, x2 = x[..., :r].clone(), x[..., r:2*r].clone()
        x[..., :r] = x1*c - x2*s
        x[..., r:2*r] = x1*s + x2*c
    return qkv


def varlen_attention(qkv, cu_seqlens, max_seqlen, dropout, causal=False):
    """flash_attn_varlen_qkvpacked_func for equal-length sequences: (b*s, 3, h, d) -> (b*s, h, d)."""
    total, _, h, d = qkv.shape
    q, k, v = qkv.view(total//max_seqlen, max_seqlen, 3, h, d).permute(2, 0, 3, 1, 4)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.permute(0, 2, 1, 3).reshape(total, h, d)


def load_dit():
    fa = types.ModuleType('flash_attn')
    fa.layers = types.SimpleNamespace(rotary=types.SimpleNamespace(apply_rotary_emb_qkv_=rotary_qkv_))
    fa.flash_attn_interface = types.SimpleNamespace(flash_attn_varlen_qkvpacked_func=varlen_attention)
    sys.modules['flash_attn'] = fa
    sys.modules['flash_attn.layers'] = fa.layers
    sys.modules['flash_attn.layers.rotary'] = fa.layers.rotary
    source = (CODE/'models'/'dit.py').read_text().replace("f'/workspace/", f"f'{DRIVE}/")
    spec = importlib.util.spec_from_loader('ddbc_dit', loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(CODE/'models'/'dit.py'), 'exec'), module.__dict__)
    return module


def load_model():
    import pickle
    from omegaconf import OmegaConf

    class Stub:
        def __init__(self, *a, **k): pass
        def __setstate__(self, state): self.__dict__['_state'] = state

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:   # their tokenizer object, not needed
                return type(name, (Stub,), {})

    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False,
                      pickle_module=types.SimpleNamespace(Unpickler=Unpickler, load=pickle.load, __name__='pickle'))
    cfg = OmegaConf.create(dict(model=OmegaConf.to_container(ckpt['hyper_parameters']['config'].model),
                                dataset='pog_dense', feature_type='clhe'))
    dit = load_dit()
    net = dit.DIT(cfg, vocab_size=MASK+1)
    state = {k[len('backbone.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('backbone.')}
    net.load_state_dict(state)
    params = [p for p in net.parameters() if p.requires_grad]
    shadow = ckpt['ema']['shadow_params']
    assert len(params) == len(shadow) and all(p.shape == s.shape for p, s in zip(params, shadow))
    with torch.no_grad():
        for p, s in zip(params, shadow):
            p.copy_(s.float())
    return net.eval(), dict(epoch=ckpt['epoch'], step=ckpt['global_step'])


# ---- sequences -------------------------------------------------------------------------------------------------------
def layout(seeds, token, kind):
    """Token list and the positions of the masked item slots (index of each slot's BOI)."""
    item = lambda i: [BOI]+list(token[i])
    masked = [BOI]+[MASK]*4
    if kind == 'single':
        given = seeds[:SLOTS-1]
        seq = [BOS]+sum((item(i) for i in given), [])
        slots = [len(seq)+5*j for j in range(SLOTS-len(given))]
        seq += masked*(SLOTS-len(given))+[EOS]
    else:
        seq = [BOS]+sum((item(i) for i in seeds), [])+[EOS, BOS]
        slots = [len(seq)+5*j for j in range(len(seeds))]
        seq += masked*len(seeds)+[EOS]
    return seq, slots


def legal_mask(seq_len, slots):
    """(seq_len, vocab) True where a token is allowed: code level j of a masked slot takes codes j*CB+1..(j+1)*CB."""
    allowed = torch.ones(seq_len, MASK+1, dtype=torch.bool)
    for b in slots:
        for j in range(4):
            allowed[b+1+j] = False
            allowed[b+1+j, j*CB+1:(j+1)*CB+1] = True
    return allowed


@torch.no_grad()
def sample(net, x, allowed):
    """Their ddpm_cache sampler (loglinear noise, subs parameterization, no time conditioning) and final argmax."""
    n = x.shape[0]
    zero = torch.zeros(n)
    dt = 1/STEPS

    def log_p_x0(x):
        logits = net(x, zero).float()
        logits[:, :, MASK] = -1e6
        logits = logits-torch.logsumexp(logits, -1, keepdim=True)
        known = x != MASK
        logits[known] = -1e6
        logits[known, x[known]] = 0
        return logits

    p = None
    for i in range(STEPS+1):
        t = 1-i*dt
        if p is None:
            p = log_p_x0(x).exp()
        q = p*(t-(t-dt))
        q[:, :, MASK] = t-dt
        q = torch.where(allowed | (torch.arange(MASK+1) == MASK), q, torch.full_like(q, -1e9))
        gumbel = 1e-10-(torch.rand_like(q)+1e-10).log()
        new = (q/gumbel).argmax(-1)
        new = torch.where(x != MASK, x, new)
        if not torch.equal(new, x):
            p = None
        x = new
    return log_p_x0(x).argmax(-1)


# ---- scoring ---------------------------------------------------------------------------------------------------------
class Catalogue:
    def __init__(self):
        d = DRIVE/'pog_dense'
        tokens = json.load(open(d/'clhe_token.json'))
        self.token = [tuple(tokens[str(i)]) for i in range(len(tokens))]
        self.item_of = {t: i for i, t in enumerate(self.token)}
        self.W = torch.tensor(np.load(d/'clhe_weight.npy', allow_pickle=True), dtype=torch.float32)
        self.F = F.normalize(torch.load(d/'clhe.pt', map_location='cpu').float(), dim=1)

    def vectors(self, codes):
        """Feature of each generated 4-code tuple: the item's own vector, else the sum of its 3 codebook vectors."""
        out = []
        for c in codes:
            i = self.item_of.get(tuple(c))
            out.append(self.F[i] if i is not None else self.W[[c[0]-1, c[1]-1, c[2]-1]].sum(0))
        return F.normalize(torch.stack(out), dim=1)


def score_queries(net, cat, pairs, kind, batch=32, seed=0):
    torch.manual_seed(seed)
    records = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start+batch]
        groups = {}
        for q, (seeds, _) in enumerate(chunk):
            seq, slots = layout(list(seeds), cat.token, kind)
            groups.setdefault((len(seq), tuple(slots)), []).append((q, seq, slots))
        values = [None]*len(chunk)
        for (length, slots), members in groups.items():
            x = torch.tensor([m[1] for m in members]).repeat_interleave(SAMPLES, 0)
            out = sample(net, x, legal_mask(length, slots))
            for r, (q, _, _) in enumerate(members):
                rows = out[r*SAMPLES:(r+1)*SAMPLES]
                codes = [rows[s, b+1:b+5].tolist() for s in range(SAMPLES) for b in slots]
                sims = cat.vectors(codes)@cat.F.T                        # generated slots x items
                sims[:, list(chunk[q][0])] = -2                          # seeds excluded
                count = torch.bincount(sims.argmax(1), minlength=sims.shape[1]).float()
                values[q] = (count+sims.max(0).values).numpy()
        records.extend(evaluate.query_metrics(v, s, g) for v, (s, g) in zip(values, chunk))
    return records


def mean(records, idx=None):
    rows = records if idx is None else [records[i] for i in idx]
    return dict(queries=len(rows), recall20=float(np.mean([r['recall20'] for r in rows])),
                ndcg20=float(np.mean([r['ndcg20'] for r in rows])))


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    ds = data.load('pog_dense', ROOT/'dials'/'cache')
    net, meta = load_model()
    cat = Catalogue()
    assert len(cat.token) == ds.n_items
    log = open(OUT/'pog_dense.log', 'a')
    say = lambda *a: (print(*a, flush=True), print(*a, file=log, flush=True))
    say('checkpoint', meta)

    validation = {}
    for kind in ('single', 'split'):
        t0 = time.time()
        validation[kind] = mean(score_queries(net, cat, ds.val_pairs, kind))
        say('validation', kind, validation[kind], f'{time.time()-t0:.0f}s')
    kind = max(validation, key=lambda k: validation[k]['ndcg20'])
    say('chosen layout', kind)

    t0 = time.time()
    records = score_queries(net, cat, ds.test_pairs, kind, seed=1)
    result = dict(note=__doc__, checkpoint=meta, samples=SAMPLES, steps=STEPS, validation=validation, layout=kind,
                  test=mean(records), clean_test=mean(records, ds.clean_test),
                  queries=[dict(recall20=r['recall20'], ndcg20=r['ndcg20']) for r in records])
    say('test', result['test'], 'clean', result['clean_test'], f'{time.time()-t0:.0f}s')
    (OUT/'pog_dense.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
