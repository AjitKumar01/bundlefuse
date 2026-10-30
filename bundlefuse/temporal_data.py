"""Whole-day retail folds, with no frequency filtering or interaction fitting on future baskets."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .protocol import mask
from .provided_splits import file_hash


def load_temporal(directory, caps=(21000, 3000, 6000), seed=0):
    root = Path(directory)
    products = {}
    with (root / 'product.csv').open() as stream:
        for row in csv.DictReader(stream):
            products[int(row['PRODUCT_ID'])] = row
    item_ids = sorted(products)
    index = {item: j for j, item in enumerate(item_ids)}
    category_map = {}
    categories, text = [], []
    for item in item_ids:
        row = products[item]
        label = row['SUB_COMMODITY_DESC'].strip() or '?'
        categories.append(category_map.setdefault(label, len(category_map)))
        text.append(' '.join(row[key] for key in ('DEPARTMENT', 'COMMODITY_DESC', 'SUB_COMMODITY_DESC', 'BRAND')))
    baskets = defaultdict(set); days = {}; households = {}
    with (root / 'transaction_data.csv').open() as stream:
        for row in csv.DictReader(stream):
            bid, day, household = row['BASKET_ID'], int(row['DAY']), row['household_key']
            if bid in days and (days[bid] != day or households[bid] != household):
                raise ValueError('A retail basket crosses day or household boundaries')
            item = int(row['PRODUCT_ID'])
            if item not in index:
                raise ValueError('Transaction product is absent from the fixed metadata catalogue')
            baskets[bid].add(index[item]); days[bid] = day; households[bid] = household
    eligible = [bid for bid, items in baskets.items() if 3 <= len(items) <= 20]
    by_day = defaultdict(list)
    for bid in eligible:
        by_day[days[bid]].append(bid)
    boundaries = []
    seen = 0
    for day in sorted(by_day):
        seen += len(by_day[day])
        if not boundaries and seen >= .7 * len(eligible):
            boundaries.append(day)
        if len(boundaries) == 1 and seen >= .8 * len(eligible):
            boundaries.append(day)
    if len(boundaries) != 2 or boundaries[0] >= boundaries[1]:
        raise ValueError('Not enough distinct days for chronological folds')
    folds = [[bid for bid in eligible if days[bid] <= boundaries[0]],
             [bid for bid in eligible if boundaries[0] < days[bid] <= boundaries[1]],
             [bid for bid in eligible if days[bid] > boundaries[1]]]
    rng = np.random.default_rng(seed)
    selected = []
    for fold, cap in zip(folds, caps):
        ordered = sorted(fold, key=lambda bid: (days[bid], int(bid)))
        if cap and len(ordered) > cap:
            positions = np.sort(rng.choice(len(ordered), size=cap, replace=False))
            ordered = [ordered[i] for i in positions]
        if not ordered:
            raise ValueError('Empty chronological fold')
        selected.append(ordered)
    user_index = {user: j for j, user in enumerate(sorted(set(households.values())))}
    train = [sorted(baskets[bid]) for bid in selected[0]]
    valid = mask([sorted(baskets[bid]) for bid in selected[1]], seed + 1, .5)
    test = mask([sorted(baskets[bid]) for bid in selected[2]], seed + 2, .5)
    ranges = [[min(days[bid] for bid in fold), max(days[bid] for bid in fold)] for fold in selected]
    if not (ranges[0][1] < ranges[1][0] and ranges[1][1] < ranges[2][0]):
        raise ValueError('Chronological day boundaries overlap')
    protocol = {'protocol': 'whole_day_chronological_fixed_catalogue', 'cutoff_days': boundaries,
                'day_ranges': dict(zip(('train', 'valid', 'test'), ranges)),
                'eligible_counts': dict(zip(('train', 'valid', 'test'), map(len, folds))),
                'train_bundles': len(train), 'valid_queries': len(valid), 'test_queries': len(test),
                'caps': list(caps), 'seed': seed, 'deduplicate_bundles': False,
                'masking': {'hidden_fraction': .5, 'validation_seed': seed + 1, 'test_seed': seed + 2},
                'query_ids': dict(zip(('train', 'valid', 'test'), selected)),
                'catalogue_items': len(item_ids), 'catalogue_sha256': hashlib.sha256(json.dumps(item_ids).encode()).hexdigest(),
                'catalogue_assumption': 'Product metadata catalogue is fixed and available transductively; product introduction dates are unavailable.',
                'sha256': {name: file_hash(root / name) for name in ('product.csv', 'transaction_data.csv')}}
    return {'train': train, 'users': [user_index[households[bid]] for bid in selected[0]],
            'n_users': len(user_index), 'valid': valid, 'test': test, 'test_ids': selected[2],
            'groups': [user_index[households[bid]] for bid in selected[2]],
            'categories': np.asarray(categories), 'text': text, 'protocol': protocol}
