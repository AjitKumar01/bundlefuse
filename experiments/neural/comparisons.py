"""Evaluation-only BundleFuse references; never imported by the neural scorer."""
from copy import deepcopy
import json
from pathlib import Path
from dataclasses import asdict
from bundlefuse.baselines import BASELINES
from bundlefuse.model import BundleFuse
from bundlefuse.provenance import write_json
from bundlefuse.splits import file_hash
from .evaluation import evaluate_classical


def run_comparisons(ds, output, checkpoint_dir, log):
    path = output/'comparisons.json'
    if path.exists():
        return json.loads(path.read_text())
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output/'comparison_queries').mkdir(parents=True, exist_ok=True)
    reference_path = Path('checkpoints/reorganized_162dd8c')/ds.name/'seed_0/model.pt'
    reference_hash = file_hash(reference_path)
    reference = BundleFuse.load(reference_path)
    for key in ('n_items', 'train_sha256', 'val_pairs_sha256', 'test_pairs_sha256', 'input_sha256'):
        if ds.metadata[key] != reference.data_spec[key]:
            raise RuntimeError(f'Reference split/input mismatch: {key}')
    models = {'bundlefuse_original_auxiliary': reference}
    matched_path = checkpoint_dir/'bundlefuse_matched.pt'
    if matched_path.exists():
        matched = BundleFuse.load(matched_path)
    else:
        cfg = deepcopy(reference.cfg)
        cfg.families = ['collaborative', 'content', 'taxonomy', 'popularity']
        cfg.signal_profile = 'extended'
        matched = BundleFuse(cfg)
        log('Fitting BundleFuse input-matched control (no auxiliary/user data)')
        matched.fit(ds.train, ds.val_pairs, features=ds.features, cat_labels=ds.cat_labels,
                    data_spec=ds.metadata)
        matched.save(matched_path)
    models['bundlefuse_matched'] = matched
    for name, cls in BASELINES.items():
        model_path = checkpoint_dir/f'{name}.pkl'
        if model_path.exists():
            model = cls.load(model_path)
        else:
            model = cls(ds.n_items).fit(ds.train, ds.val_pairs, features=ds.features,
                                       cat_labels=ds.cat_labels, data_spec=ds.metadata)
            model.save(model_path)
        models[name] = model
    result = {}
    for name, model in models.items():
        log(f'Evaluating comparator {name}')
        row = evaluate_classical(model, ds.test_pairs,
                                 ds.test_users if name == 'bundlefuse_original_auxiliary' else None)
        write_json(output/'comparison_queries'/f'{name}.json', row)
        result[name] = {k:v for k,v in row.items() if k != 'queries'}
        if isinstance(model, BundleFuse):
            result[name]['fitted_recipe'] = dict(weights=model.w, prior=model.prior_keys,
                backoff=list(model.best_eb), higher_order=list(model.best_ho), config=asdict(model.cfg))
        log(f'{name}: R20={row["recall20"]:.6f} N20={row["ndcg20"]:.6f}')
    expected = json.loads((Path('results/reorganized_run/original')/ds.name/'result.json').read_text())['metrics']
    for metric, old_key in [('recall20','recall@20'), ('ndcg20','ndcg@20')]:
        if abs(result['bundlefuse_original_auxiliary'][metric]-expected[old_key]) > 1e-6:
            raise RuntimeError(f'Original BundleFuse metric replay mismatch: {metric}')
    if file_hash(reference_path) != reference_hash:
        raise RuntimeError('Original checkpoint changed')
    write_json(output/'comparison_provenance.json', dict(original_checkpoint=str(reference_path),
        sha256_before=reference_hash, sha256_after=file_hash(reference_path),
        matched_checkpoint=str(matched_path), matched_sha256=file_hash(matched_path),
        note='Classical baselines use repository default settings, not an exhaustive tuning search. '
             'p3alpha/rp3beta are local named implementations, not paper reproductions. '
             'Original BundleFuse additionally uses auxiliary/user histories.'))
    write_json(path, result)
    return result
