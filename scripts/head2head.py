import os, sys, subprocess, re, csv, time
HERE = "/Users/ajit/Projects/Career_change/projects/bundle-dpo/bundlefuse"
PY = sys.executable
# dataset -> (module, env)   fast first, spotify last
RUNS = [
    ("electronic",     "bundlefuse.eval.bundlerec", {"BRDS":"electronic"}),
    ("food",           "bundlefuse.eval.bundlerec", {"BRDS":"food"}),
    ("clothing",       "bundlefuse.eval.bundlerec", {"BRDS":"clothing"}),
    ("dunnhumby",      "bundlefuse.eval.dunnhumby", {"SPLIT_MODE":"random"}),
    ("pog",            "bundlefuse.eval.fullrank",  {"BUNDLE_DS":"pog","RAMEN_SEED":"0"}),
    ("pog_dense",      "bundlefuse.eval.fullrank",  {"BUNDLE_DS":"pog_dense","RAMEN_SEED":"0"}),
    ("spotify_sparse", "bundlefuse.eval.fullrank",  {"BUNDLE_DS":"spotify_sparse","RAMEN_SEED":"0"}),
    ("spotify",        "bundlefuse.eval.fullrank",  {"BUNDLE_DS":"spotify","RAMEN_SEED":"0"}),
]
ROW = re.compile(r"^Ours\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
def run(mod, env):
    e = dict(os.environ, **env); e["MATRIX"]="1"; e["PYTHONWARNINGS"]="ignore"; e["FORCE_DEVICE"]="cpu"; e["SEEDS"]="1"; e["TEST_N"]="2000"
    p = subprocess.run([PY,"-u","-m",mod], cwd=HERE, env=e, capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        m = ROW.match(ln.strip())
        if m: return [float(m.group(i)) for i in range(1,5)]
    sys.stderr.write(f"  !! no Ours row for {mod} {env}\n"+ "\n".join(p.stderr.splitlines()[-5:])+"\n")
    return None
out = SP = os.path.dirname(os.path.abspath(__file__))
rows = []
for name, mod, env in RUNS:
    for enc in ("1","0"):
        t=time.time()
        r = run(mod, dict(env, ENCODER=enc))
        dt=time.time()-t
        tag = "with-enc" if enc=="1" else "zero-param"
        r20 = r[2] if r else None
        print(f"{name:16s} {tag:11s} R@20={r20}  ({dt:.0f}s)", flush=True)
        rows.append([name, tag, *(r if r else ["","","",""])])
with open(os.path.join(SP,"head2head.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["dataset","variant","R@10","N@10","R@20","N@20"]); w.writerows(rows)
print("DONE head2head", flush=True)
