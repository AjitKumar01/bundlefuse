"""Dataset options and checkpoint-bound split reconstruction for the command-line tools."""
import argparse

from . import datasets
from .datasets.base import BUNDLEMLLM


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('Use a nonnegative value (0 means all queries)')
    return value


def add_data_arguments(parser, training=False):
    parser.add_argument('--protocol', choices=('provided', 'resplit'), default=None,
                        help='default: provided for POG/Spotify, resplit otherwise; inference inherits checkpoint')
    parser.add_argument('--split-seed', type=int, default=None, help='resplit seed; inference inherits checkpoint')
    parser.add_argument('--mask-frac', type=float, default=None, help='resplit masking fraction')
    parser.add_argument('--test-n', type=nonnegative, default=None, help='0 means full test split')
    if training:
        parser.add_argument('--val-n', type=nonnegative, default=None, help='0 means full validation split')


def load_data(args, model=None, test_n=None):
    saved = getattr(model, 'data_spec', None) if model is not None else None
    if model is not None and not saved:
        raise ValueError('Checkpoint lacks dataset provenance; refit with evaluator.py before dataset evaluation')
    if saved and saved['dataset'] != args.dataset:
        raise ValueError('Checkpoint dataset differs from the requested dataset')
    default = 'provided' if args.dataset in BUNDLEMLLM else 'resplit'
    mode = args.protocol or (saved['protocol'] if saved else default)
    seed = args.split_seed if args.split_seed is not None else ((saved['split_seed'] or 0) if saved else 0)
    mask = args.mask_frac if args.mask_frac is not None else ((saved['mask_frac'] or .5) if saved else .5)
    if saved and (mode != saved['protocol'] or (mode == 'resplit' and
                 (seed != saved['split_seed'] or mask != saved['mask_frac']))):
        raise ValueError('Evaluation protocol/seed/mask differs from the fitted checkpoint')
    val_limit = saved['val_limit'] if saved else getattr(args, 'val_n', None)
    limit = test_n if test_n is not None else args.test_n
    if limit is None:
        limit = saved['test_limit'] if saved else None
    if not saved and args.dataset not in BUNDLEMLLM:
        val_limit = 800 if val_limit is None else val_limit
        limit = 2000 if limit is None else limit
    ds = datasets.load(args.dataset, val_n=val_limit, test_n=limit, split_seed=seed,
                       mask_frac=mask, protocol_mode=mode)
    if saved:
        for key in ('n_items', 'train_sha256', 'val_pairs_sha256', 'input_sha256'):
            if saved[key] != ds.metadata[key]:
                raise ValueError(f'Dataset changed since fitting ({key}); refusing mismatched evaluation')
    if not ds.val_pairs or not ds.test_pairs:
        raise ValueError('Dataset must have nonempty validation and test splits')
    return ds
