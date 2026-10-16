"""Unified Driver for the 8-Dataset SIMBA Benchmark Matrix.

Usage:
    python -m bundlefuse.run_matrix                 # run all 8 datasets -> results/matrix.csv
    python -m bundlefuse.run_matrix pog pog_dense   # run a subset
"""
import os, sys, json, subprocess, re, csv

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS, exist_ok=True)
PY = sys.executable

RUN = {
    "pog":            ("fullrank_eval.py",  {"BUNDLE_DS": "pog",            "RAMEN_SEED": "0", "MATRIX": "1"}),
    "pog_dense":      ("fullrank_eval.py",  {"BUNDLE_DS": "pog_dense",      "RAMEN_SEED": "0", "MATRIX": "1"}),
    "spotify":        ("fullrank_eval.py",  {"BUNDLE_DS": "spotify",        "RAMEN_SEED": "0", "MATRIX": "1"}),
    "spotify_sparse": ("fullrank_eval.py",  {"BUNDLE_DS": "spotify_sparse", "RAMEN_SEED": "0", "MATRIX": "1"}),
    "electronic":     ("bundlerec_eval.py", {"BRDS": "electronic", "MATRIX": "1"}),
    "food":           ("bundlerec_eval.py", {"BRDS": "food",       "MATRIX": "1"}),
    "clothing":       ("bundlerec_eval.py", {"BRDS": "clothing",   "MATRIX": "1"}),
    "dunnhumby":      ("dunnhumby_eval.py", {"MATRIX": "1"}),
}
ORDER = ["pog", "pog_dense", "spotify", "spotify_sparse", "electronic", "food", "clothing", "dunnhumby"]
PRINT_METHODS = ["popularity", "co-occurrence", "content-kNN", "item2vec", "CLHE", "RaMen", "Ours"]

# Published / reproduced heavy baselines (full-ranking R@10, N@10, R@20, N@20)
PUBLISHED = {
    "pog":            {"CLHE": [0.0213, 0.0160, 0.0284, 0.0193], "RaMen": [0.0264, 0.0191, 0.0375, 0.0226]},
    "pog_dense":      {"CLHE": [0.3438, 0.2662, 0.3815, 0.2771]},
    "spotify":        {"CLHE": [0.0689, 0.1950, 0.1082, 0.1806], "RaMen": [0.0695, 0.2060, 0.1091, 0.1882]},
    "spotify_sparse": {"CLHE": [0.0608, 0.1164, 0.0969, 0.1150]},
    "electronic":     {"CLHE": [0.4407, 0.3300, 0.4721, 0.3390], "RaMen": [0.7410, 0.5104, 0.8371, 0.5373]},
    "food":           {"CLHE": [0.4557, 0.3237, 0.5077, 0.3386], "RaMen": [0.7575, 0.5028, 0.8459, 0.5242]},
}

ROW_RE = re.compile(r"^(popularity|co-occurrence|content-kNN|item2vec|Ours)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def run_ds(ds):
    script, env = RUN[ds]
    e = dict(os.environ, **env)
    e.setdefault("PYTHONWARNINGS", "ignore")
    print(f"[run] {ds:15s} -> {script}", flush=True)
    p = subprocess.run([PY, "-u", os.path.join(HERE, script)], cwd=HERE, env=e, capture_output=True, text=True)
    rows = {}
    for line in p.stdout.splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            rows[m.group(1)] = [float(m.group(i)) for i in range(2, 6)]
    if not rows:
        print(f"  !! no rows parsed for {ds}. stderr tail:\n" + "\n".join(p.stderr.splitlines()[-8:]), flush=True)
    return rows


def main():
    want = [a for a in sys.argv[1:] if a in RUN] or ORDER
    table = {}
    for ds in want:
        rows = run_ds(ds)
        rows.update(PUBLISHED.get(ds, {}))
        table[ds] = rows

    cols = ["R@10", "N@10", "R@20", "N@20"]
    with open(os.path.join(RESULTS, "matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method"] + cols)
        for ds in ORDER:
            for meth in PRINT_METHODS:
                v = table.get(ds, {}).get(meth)
                if v is not None:
                    w.writerow([ds, meth] + ["%.4f" % x for x in v])

    print("\n================= SIMBA MASTER MATRIX (full-ranking) =================", flush=True)
    print(f"{'dataset':15s} {'method':14s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8}")
    for ds in ORDER:
        if ds not in table:
            continue
        for meth in PRINT_METHODS:
            v = table[ds].get(meth)
            if v is not None:
                print(f"{ds:15s} {meth:14s} {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.4f} {v[3]:8.4f}")


if __name__ == "__main__":
    main()
