"""Atomic experiment artifacts and shared aggregation of paired seed results."""
import json
import os
from pathlib import Path
import numpy as np
from .metrics import TIE_POLICY


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def flat_metrics(metrics):
    return {f"{label}@{k}": float(pair[i]) for k, pair in metrics.items()
            for i, label in enumerate(("R", "N"))}


def summarize_seeds(records):
    """Each record contains its seed ID and unrounded method->flat metrics."""
    if not records:
        raise ValueError("No successful experiment seeds")
    methods = set(records[0]["methods"])
    if any(set(r["methods"]) != methods for r in records):
        raise ValueError("Inconsistent methods across experiment seeds")
    output = {}
    for method in sorted(methods):
        keys = records[0]["methods"][method]
        output[method] = {}
        for key in keys:
            values = [r["methods"][method][key] for r in records]
            output[method][key] = float(np.mean(values))
            output[method][key + "_std"] = float(np.std(values))
        output[method]["seeds"] = len(records)
    return output


def baseline_artifact(dataset, n_items, seed_records):
    records = [{"seed": r["seed"], "methods": {
        name: flat_metrics(value["test"]) for name, value in r["methods"].items() if "test" in value
    }} for r in seed_records]
    return {"schema_version": 2, "dataset": dataset, "n": int(n_items),
            "seeds": len(records), "tie_policy": TIE_POLICY,
            "selection_objective": "ndcg20", "methods": summarize_seeds(records),
            "per_seed": seed_records}
