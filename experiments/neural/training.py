"""Deterministic CPU training, validation-only selection, retained resumable epochs."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from bundlefuse.provenance import write_json
from .data import make_batch
from .evaluation import evaluate
from .model import BundleNetwork, ModelConfig


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def fit(ds, features, cats, freq, variant, seed, config, checkpoint_dir, output_dir, log):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir/'selected.json').exists():
        return json.loads((output_dir/'selected.json').read_text())
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model_config = ModelConfig(variant=variant, dim=config['dim'])
    model = BundleNetwork(features, cats, freq, model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    epochs = sorted(checkpoint_dir.glob('epoch_*.pt'))
    best, best_epoch, stale, start_epoch = -float('inf'), 0, 0, 1
    elapsed = 0.
    history = []
    if epochs:
        state = torch.load(epochs[-1], map_location='cpu', weights_only=False)
        if state['config'] != config or state['model_config'] != asdict(model_config):
            raise RuntimeError('Resume configuration mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']); rng.bit_generator.state = state['numpy_rng']
        best, best_epoch, stale = state['best'], state['best_epoch'], state['stale']
        start_epoch = state['epoch'] + 1
        history, elapsed = state['history'], state['elapsed_seconds']
        log(f'{variant} seed={seed}: resume after epoch {start_epoch-1}')
    log(f'{variant} seed={seed}: parameters={model.parameter_counts()}')
    begun = time.monotonic()
    for epoch in range(start_epoch, config['epochs']+1):
        if stale >= config['patience']:
            break
        model.train()
        order = rng.permutation(len(ds.train))
        total_loss, queries = 0., 0
        for start in range(0, len(order), config['batch_size']):
            bundles = [ds.train[i] for i in order[start:start+config['batch_size']]]
            seeds, valid, candidates, positive, negative = make_batch(bundles, ds.n_items, rng,
                                                                        config['negative_pool'])
            optimizer.zero_grad(set_to_none=True)
            scores = model(seeds, valid, candidates)
            from .data import completion_loss
            loss = completion_loss(scores, positive, negative, hard=epoch > 2,
                                   negatives=config['negatives'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach())*len(bundles); queries += len(bundles)
        record = dict(epoch=epoch, loss=total_loss/queries)
        should_evaluate = epoch % config['eval_every'] == 0 or epoch == config['epochs']
        if should_evaluate:
            validation = evaluate(model, ds.val_pairs, config['eval_batch'], config['candidate_chunk'])
            record['validation'] = {k:v for k,v in validation.items() if k != 'queries'}
            if validation['ndcg20'] > best:
                best, best_epoch, stale = validation['ndcg20'], epoch, 0
            else:
                stale += 1
            log(f'{variant} seed={seed} epoch={epoch} loss={record["loss"]:.5f} '
                f'VAL R20={validation["recall20"]:.6f} N20={validation["ndcg20"]:.6f} '
                f'best_epoch={best_epoch} patience={stale}/{config["patience"]}')
        else:
            log(f'{variant} seed={seed} epoch={epoch} loss={record["loss"]:.5f}')
        history.append(record)
        write_json(output_dir/'history.json', history)
        if should_evaluate:
            save_checkpoint(checkpoint_dir/f'epoch_{epoch:04d}.pt', dict(
                model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                best=best, best_epoch=best_epoch, stale=stale, history=history,
                config=config, model_config=asdict(model_config),
                torch_rng=torch.get_rng_state(), numpy_rng=rng.bit_generator.state,
                elapsed_seconds=elapsed+time.monotonic()-begun))
            write_json(output_dir/'status.json', dict(status='training', epoch=epoch,
                best_epoch=best_epoch, best_validation_ndcg20=best,
                elapsed_seconds=elapsed+time.monotonic()-begun))
    path = checkpoint_dir/f'epoch_{best_epoch:04d}.pt'
    if not path.exists():
        raise RuntimeError('No selected checkpoint')
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    validation = evaluate(model, ds.val_pairs, config['eval_batch'], config['candidate_chunk'])
    if abs(validation['ndcg20']-best) > 1e-12:
        raise RuntimeError('Selected checkpoint validation replay differs')
    write_json(output_dir/'validation_predictions.json', validation)
    selected = dict(variant=variant, seed=seed, checkpoint=str(path), best_epoch=best_epoch,
                    validation={k:v for k,v in validation.items() if k != 'queries'},
                    parameters=model.parameter_counts(), model_config=asdict(model_config),
                    elapsed_seconds=elapsed+time.monotonic()-begun,
                    epochs_run=history[-1]['epoch'], selection='validation NDCG@20 only')
    write_json(output_dir/'selected.json', selected)
    write_json(output_dir/'status.json', dict(status='selected', **selected))
    return selected


def load_selected(selected, features, cats, freq):
    model = BundleNetwork(features, cats, freq, ModelConfig(**selected['model_config']))
    state = torch.load(selected['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()
    return model
