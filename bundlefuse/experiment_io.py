"""Fail-closed subprocess execution and parsing for experiment drivers."""
import os
from pathlib import Path
import re
import subprocess

ROW = re.compile(r"^(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)(?:\s|$)")


def run_checked(command, *, env=None, cwd=None, log_path=None):
    result = subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True)
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Experiment failed (exit {result.returncode}): {command}\n{result.stderr[-3000:]}")
    return result


def parse_matrix(stdout, required=("popularity", "co-occurrence", "content-kNN", "item2vec", "Ours")):
    rows = {}
    for line in stdout.splitlines():
        match = ROW.match(line.strip())
        if match:
            rows[match[1]] = [float(match[i]) for i in range(2, 6)]
    missing = set(required) - rows.keys()
    if missing:
        raise ValueError(f"Missing experiment rows: {sorted(missing)}")
    if any(not 0 <= value <= 1 for row in rows.values() for value in row):
        raise ValueError("Experiment returned a metric outside [0, 1]")
    return rows
