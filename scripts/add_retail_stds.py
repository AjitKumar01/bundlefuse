"""Add cross-seed error bars to the e-commerce/retail rows of matrix_zeroparam.

These datasets' evaluators already run SEEDS (=5) held-out splits internally; the
matrix only stored their mean. This reruns them once with MATRIX_STD=1 (deterministic,
ENCODER=0, CPU), reads the per-method population std across those 5 splits, and patches
the R@20_std column of results/matrix_zeroparam.{csv,json} in place. Fullrank rows
(which already carry 3-seed std) are left untouched."""
import os, sys, subprocess, re, csv, json
HERE = "/Users/ajit/Projects/Career_change/projects/bundle-dpo/bundlefuse"
PY = sys.executable
LABELS = ("popularity", "co-occurrence", "content-kNN", "item2vec", "Ours")
RETAIL = [("electronic", "bundlefuse.bundlerec_eval", "BRDS"),
          ("food", "bundlefuse.bundlerec_eval", "BRDS"),
          ("clothing", "bundlefuse.bundlerec_eval", "BRDS"),
          ("dunnhumby", "bundlefuse.dunnhumby_eval", None)]
STD = re.compile(r"^STD\s+(\S+)\s+([\d.]+)\s+n=(\d+)")

stds = {}
for ds, mod, var in RETAIL:
    env = dict(os.environ, MATRIX="1", MATRIX_STD="1", ENCODER="0",
               FORCE_DEVICE="cpu", PYTHONWARNINGS="ignore")
    if var:
        env[var] = ds
    p = subprocess.run([PY, "-u", "-m", mod], cwd=HERE, env=env,
                       capture_output=True, text=True)
    got = {}
    nseed = None
    for ln in p.stdout.splitlines():
        m = STD.match(ln.strip())
        if m:
            got[m.group(1)] = float(m.group(2)); nseed = int(m.group(3))
    if "Ours" not in got:
        sys.stderr.write(f"!! no STD for {ds}\n" + "\n".join(p.stderr.splitlines()[-6:]) + "\n")
        continue
    stds[ds] = got
    print(f"{ds:12s} n={nseed} Ours R@20 std={got['Ours']:.4f} "
          f"(co-occ {got.get('co-occurrence', float('nan')):.4f})", flush=True)

# patch CSV
csv_path = os.path.join(HERE, "results", "matrix_zeroparam.csv")
rows = list(csv.reader(open(csv_path)))
head, body = rows[0], rows[1:]
for r in body:
    ds, lab = r[0], r[1]
    if ds in stds and lab in stds[ds]:
        r[6] = f"{stds[ds][lab]:.4f}"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(head); w.writerows(body)

# patch JSON (R@20 is index 2 of the per-method list -> (mean, std))
jpath = os.path.join(HERE, "results", "matrix_zeroparam.json")
J = json.load(open(jpath))
for ds in stds:
    if ds in J:
        for lab in stds[ds]:
            if lab in J[ds]:
                mean = J[ds][lab][2][0]
                J[ds][lab][2] = [mean, stds[ds][lab]]
json.dump(J, open(jpath, "w"), indent=2)
print("RETAIL_STD_DONE", flush=True)
