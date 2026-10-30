"""Validate canonical matrix consistency; error bars now come from the SAME seed records.

The experiment runner writes means/stds together. This compatibility command no longer
reruns a different experiment and patches an existing table's error bars.
"""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'results'
lookup = json.loads((root / 'matrix_zeroparam.json').read_text())
for row in csv.DictReader((root / 'matrix_zeroparam.csv').open()):
    values = lookup[row['dataset']][row['method']]
    for index, key in enumerate(('R@10', 'N@10', 'R@20', 'N@20')):
        if abs(float(row[key]) - values[index][0]) > 1e-7:
            raise ValueError(f"CSV/JSON disagree: {row['dataset']} {row['method']} {key}")
    if abs(float(row['R@20_std']) - values[2][1]) > 1e-7:
        raise ValueError(f"CSV/JSON error bars disagree: {row['dataset']} {row['method']}")
print('Canonical CSV/JSON means and standard deviations agree.')
