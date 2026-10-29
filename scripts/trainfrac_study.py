"""POG training-data learning curve: keep val(10%)/test(20%) fixed, subsample the
train(70%) fold to 100/90/80/70% of itself. 3 seeds each, mean+/-std. Runs sequentially."""
import os, sys, subprocess, re, statistics as st, time
HERE = "/Users/ajit/Projects/Career_change/projects/bundle-dpo/bundlefuse"
PY = sys.executable
ROW = re.compile(r"^Ours\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
FRACS = [1.0, 0.9, 0.8, 0.7]
SEEDS = [0, 1, 2]
results = {}
for frac in FRACS:
    r20s, n20s = [], []
    for sd in SEEDS:
        e = dict(os.environ, MATRIX="1", ENCODER="0", FORCE_DEVICE="cpu",
                 BUNDLE_DS="pog", RAMEN_SEED=str(sd), TRAIN_FRAC=str(frac),
                 PYTHONWARNINGS="ignore")
        t = time.time()
        p = subprocess.run([PY, "-u", "-m", "bundlefuse.eval.fullrank"], cwd=HERE,
                           env=e, capture_output=True, text=True)
        got = None
        for ln in p.stdout.splitlines():
            m = ROW.match(ln.strip())
            if m:
                got = (float(m.group(3)), float(m.group(4)))
        if got:
            r20s.append(got[0]); n20s.append(got[1])
            print(f"frac={frac} seed={sd}: R@20={got[0]:.4f} N@20={got[1]:.4f} ({time.time()-t:.0f}s)", flush=True)
        else:
            sys.stderr.write(f"!! no Ours frac={frac} sd={sd}\n" + "\n".join(p.stderr.splitlines()[-5:]) + "\n")
    if r20s:
        results[frac] = (st.mean(r20s), st.pstdev(r20s), st.mean(n20s), st.pstdev(n20s), len(r20s))
print("\n=== POG TRAINING-DATA LEARNING CURVE (val/test fixed; train subsampled) ===", flush=True)
print(f"{'train frac':>10} {'% of total':>10} {'R@20':>16} {'N@20':>16}", flush=True)
for frac in FRACS:
    if frac in results:
        r_m, r_s, n_m, n_s, k = results[frac]
        print(f"{frac:>10.2f} {70*frac:>9.1f}% {r_m:>8.4f}+/-{r_s:.4f} {n_m:>8.4f}+/-{n_s:.4f}", flush=True)
print("TRAINFRAC_DONE", flush=True)
