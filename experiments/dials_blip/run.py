"""DIALS on Electronic, Food and Clothing with BLIP text features in place of TF-IDF.

What changes: only the text encoder. dials.data.load_resplit embeds each item title with TF-IDF and projects it to 128
dimensions with a truncated SVD. Here the TF-IDF step is replaced by BLIP's text encoder (Li et al., ICML 2022;
checkpoint Salesforce/blip-itm-base-coco, text-only, the 768-d [CLS] vector, the same size as the BLIP features
released for POG). Row normalization, the SVD to 128 dimensions, labels, split, model, training and seeds are those of
the frozen DIALS. Items without a title get a zero vector, as with TF-IDF. Images are not used: Electronic has image
URLs for 451 of 3,499 items (those tested do not resolve), Food and Clothing have none.

The released split's BLIP features cannot be reproduced exactly (CLHE released no extraction script), so these are
BLIP features computed by us, not the ones RaMen used.

Outputs, results/dials_blip/: <dataset>/seed<s>/{model.pt, selected.json, history.json, test.json}, and summary.json
with three-seed test means and paired tests against TF-IDF DIALS (../dials/runs), Holm over the six tests.

PYTHONPATH=.:../dials nice -n 10 .venv/bin/python -m experiments.dials_blip.run
"""
import argparse
import hashlib
import json
from pathlib import Path
import triad  # pins BLAS threads before numpy loads
import numpy as np
import torch
from triad import data
from triad import train as T
from triad.evaluate import evaluate
from bundlefuse.paths import DATA_ROOT
from experiments.mechanisms.statistics import paired_comparison, holm_adjust

DATASETS = ['electronic', 'food', 'clothing'];SEEDS = (0, 1, 2)
OUT = Path('results/dials_blip');CACHE = Path(DATA_ROOT)/'dials'/'cache_blip';FROZEN = Path(DATA_ROOT)/'dials'/'runs'
CHECKPOINT = 'Salesforce/blip-itm-base-coco'


def blip(titles, batch=64):
    """768-d [CLS] vectors of BLIP's text encoder for a list of titles; zero rows for empty titles."""
    from transformers import BlipForImageTextRetrieval, BlipProcessor
    processor = BlipProcessor.from_pretrained(CHECKPOINT);model = BlipForImageTextRetrieval.from_pretrained(CHECKPOINT).eval()
    out = np.zeros((len(titles), model.config.text_config.hidden_size), np.float32)
    keep = [i for i, t in enumerate(titles) if t.strip()]
    with torch.inference_mode():
        for start in range(0, len(keep), batch):
            rows = keep[start:start + batch]
            tokens = processor.tokenizer([titles[i] for i in rows], padding=True, truncation=True, max_length=128, return_tensors='pt')
            hidden = model.text_encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, return_dict=True).last_hidden_state
            out[rows] = hidden[:, 0].numpy()
    return out


class BlipVectorizer:
    """Stands in for TfidfVectorizer inside dials.data.load_resplit: fit_transform(titles) returns BLIP vectors."""

    def __init__(self, **_):pass

    def fit_transform(self, titles):
        key = CACHE/f"{hashlib.sha256(chr(0).join(titles).encode()).hexdigest()[:16]}.npy"  # one file per catalogue (titles in item order)
        if not key.exists():
            key.parent.mkdir(parents=True, exist_ok=True);np.save(key, blip(list(titles)))
        return np.load(key)


def load(name):
    original = data.TfidfVectorizer;data.TfidfVectorizer = BlipVectorizer
    try:return data.load(name, CACHE)
    finally:data.TfidfVectorizer = original


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--threads', type=int, default=2);a = p.parse_args()
    torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True)
    summary = dict(note=__doc__.split('\n\n')[1], encoder=CHECKPOINT, datasets={});records = []
    for name in DATASETS:
        ds = load(name);runs = [OUT/name/f'seed{s}' for s in SEEDS]
        for seed, run in zip(SEEDS, runs):
            T.fit(ds, seed, run, CACHE);T.log(f'blip {name} seed {seed} trained')
            if not (run/'test.json').exists():T.write_json(run/'test.json', evaluate(T.load(ds, run, CACHE), ds.test_pairs))
        q = lambda paths:np.mean([[[r[m] for m in ('recall20', 'ndcg20')] for r in json.loads(Path(f).read_text())['queries']]
                                  for f in paths], 0)
        blip_q = q([r/'test.json' for r in runs]);tfidf_q = q([FROZEN/name/f'seed{s}'/'test.json' for s in SEEDS])
        clusters = [str(tuple(sorted(set(s) | set(g)))) for s, g in ds.test_pairs]  # identical complete bundles move together
        summary['datasets'][name] = dict(blip=blip_q.mean(0).tolist(), tfidf=tfidf_q.mean(0).tolist(),
                                         seeds=[json.loads((r/'selected.json').read_text()) for r in runs])
        for j, metric in enumerate(('recall20', 'ndcg20')):
            records.append(dict(paired_comparison(blip_q[:, j], tfidf_q[:, j], clusters, bootstrap=2000, permutations=10000,
                                                  seed=20261116), dataset=name, metric=metric))
        T.log(f'{name} BLIP R/N {blip_q.mean(0).round(4)} TF-IDF R/N {tfidf_q.mean(0).round(4)}')
    holm_adjust(records, len(records));summary['tests'] = records;OUT.mkdir(parents=True, exist_ok=True)
    T.write_json(OUT/'summary.json', summary)


if __name__ == '__main__':
    main()
