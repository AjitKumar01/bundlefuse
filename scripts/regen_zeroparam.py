"""Deterministic zero-parameter regeneration of the full matrix (citable numbers).
fullrank datasets: 3 seeds -> mean +/- std ; e-commerce/retail: internal 5 seeds (mean).
All runs ENCODER=0, FORCE_DEVICE=cpu."""
import os, sys, subprocess, re, csv, json, statistics as st, time
HERE = "/Users/ajit/Projects/Career_change/projects/bundle-dpo/bundlefuse"
PY = sys.executable
LABELS = ("popularity", "co-occurrence", "content-kNN", "item2vec", "Ours")
ROW = re.compile(r"^(popularity|co-occurrence|content-kNN|item2vec|Ours)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
PUBLISHED = {  # R@20 of heavy baselines, for the comparison column (not recomputed here)
  "pog":{"CLHE":0.0284,"RaMen":0.0375},"pog_dense":{"CLHE":0.3815},
  "spotify":{"CLHE":0.1082,"RaMen":0.1091},"spotify_sparse":{"CLHE":0.0969},
  "electronic":{"CLHE":0.4721,"RaMen":0.8371},"food":{"CLHE":0.5077,"RaMen":0.8459},
}
FULLRANK=[("pog","BUNDLE_DS"),("pog_dense","BUNDLE_DS"),("spotify_sparse","BUNDLE_DS"),("spotify","BUNDLE_DS")]
RETAIL=[("electronic","bundlefuse.eval.bundlerec","BRDS"),("food","bundlefuse.eval.bundlerec","BRDS"),
        ("clothing","bundlefuse.eval.bundlerec","BRDS"),("dunnhumby","bundlefuse.eval.dunnhumby",None)]
def run(mod, env):
    e=dict(os.environ,**env); e["MATRIX"]="1"; e["ENCODER"]="0"; e["FORCE_DEVICE"]="cpu"; e["PYTHONWARNINGS"]="ignore"
    p=subprocess.run([PY,"-u","-m",mod],cwd=HERE,env=e,capture_output=True,text=True)
    rows={}
    for ln in p.stdout.splitlines():
        m=ROW.match(ln.strip())
        if m: rows[m.group(1)]=[float(m.group(i)) for i in range(2,6)]  # R@10,N@10,R@20,N@20
    if "Ours" not in rows: sys.stderr.write(f"!! no Ours for {mod} {env}\n"+"\n".join(p.stderr.splitlines()[-6:])+"\n")
    return rows
table={}
for ds,var in FULLRANK:
    seeds=[]
    for sd in (0,1,2):
        t=time.time(); r=run("bundlefuse.eval.fullrank",{var:ds,"RAMEN_SEED":str(sd)})
        if "Ours" in r: seeds.append(r); print(f"{ds} seed{sd} Ours R@20={r['Ours'][2]:.4f} ({time.time()-t:.0f}s)",flush=True)
    if seeds:
        agg={}
        for lab in LABELS:
            vals=[s[lab] for s in seeds if lab in s]
            if vals: agg[lab]=[ (st.mean(c), (st.pstdev(c) if len(c)>1 else 0.0)) for c in zip(*vals)]
        table[ds]=agg
for ds,mod,var in RETAIL:
    env={var:ds} if var else {}
    t=time.time(); r=run(mod,env)
    if r:
        table[ds]={lab:[(v,0.0) for v in r[lab]] for lab in r}
        print(f"{ds} Ours R@20={r['Ours'][2]:.4f} ({time.time()-t:.0f}s)",flush=True)
# write CSV (R@20 mean/std) + json
with open(os.path.join(HERE,"results","matrix_zeroparam.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["dataset","method","R@10","N@10","R@20","N@20","R@20_std"])
    for ds in table:
        for lab in LABELS:
            if lab in table[ds]:
                c=table[ds][lab]; w.writerow([ds,lab,f"{c[0][0]:.4f}",f"{c[1][0]:.4f}",f"{c[2][0]:.4f}",f"{c[3][0]:.4f}",f"{c[2][1]:.4f}"])
json.dump({ds:{lab:table[ds][lab] for lab in table[ds]} for ds in table}, open(os.path.join(HERE,"results","matrix_zeroparam.json"),"w"), indent=2)
print("\n=== ZERO-PARAM MATRIX (R@20 mean+/-std; CLHE/RaMen published) ===",flush=True)
for ds in table:
    o=table[ds]["Ours"][2]; pub=PUBLISHED.get(ds,{})
    pubs=" ".join(f"{k}={v}" for k,v in pub.items())
    print(f"{ds:15s} Ours={o[0]:.4f}+/-{o[1]:.4f}   {pubs}",flush=True)
print("REGEN_DONE",flush=True)
