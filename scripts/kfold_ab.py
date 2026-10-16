"""A/B: single-val vs K-fold CV hyperparameter selection, pog + spotify_sparse, 3 seeds."""
import os, sys, subprocess, re, statistics as st, time
HERE = "/Users/ajit/Projects/Career_change/projects/bundle-dpo/bundlefuse"
PY = sys.executable
ROW = re.compile(r"^Ours\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
def run(ds, sd, kfold):
    e = dict(os.environ, MATRIX="1", ENCODER="0", FORCE_DEVICE="cpu",
             BUNDLE_DS=ds, RAMEN_SEED=str(sd), PYTHONWARNINGS="ignore")
    if kfold:
        e["KFOLD"] = "5"
    p = subprocess.run([PY, "-u", "-m", "bundlefuse.fullrank_eval"], cwd=HERE,
                       env=e, capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        m = ROW.match(ln.strip())
        if m:
            return float(m.group(3)), float(m.group(4))  # R@20, N@20
    sys.stderr.write(f"!! no Ours {ds} sd{sd} kfold={kfold}\n" + "\n".join(p.stderr.splitlines()[-5:]) + "\n")
    return None, None
for ds in ("pog", "spotify_sparse"):
    base, kf = [], []
    for sd in (0, 1, 2):
        t = time.time(); b = run(ds, sd, False); k = run(ds, sd, True)
        if b[0] is not None: base.append(b[0])
        if k[0] is not None: kf.append(k[0])
        print(f"{ds} sd{sd}: single-val R@20={b[0]} | kfold R@20={k[0]} ({time.time()-t:.0f}s)", flush=True)
    if base and kf:
        print(f">>> {ds}: single-val {st.mean(base):.4f}+/-{st.pstdev(base):.4f} | "
              f"kfold {st.mean(kf):.4f}+/-{st.pstdev(kf):.4f} | "
              f"delta={st.mean(kf)-st.mean(base):+.4f}", flush=True)
print("KFOLD_AB_DONE", flush=True)
