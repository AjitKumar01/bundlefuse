"""Run every ablation job, at most 9 at once and at most 6 on the Spotify datasets, longest first; then test.

PYTHONPATH=.:../dials nohup .venv/bin/python -m experiments.dials_ablations.schedule > results/dials_ablations/schedule.log 2>&1 &
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from .run import PARTS,OUT

HEAVY=['spotify','spotify_sparse'];ORDER=['spotify_sparse','pog_dense','pog','clothing','electronic','food']  # paper set (2026-11-16): full Spotify and Dunnhumby left out
TOTAL,HEAVY_MAX=9,6  # 13 and then 11 strained the machine; Spotify 6 once only Spotify_sparse was left (2026-11-17)
LOGS=OUT/'logs'


def command(*args):return [sys.executable,'-u','-m','experiments.dials_ablations.run',*args]


def elsewhere():
    """Training jobs left running by an earlier scheduler, so a restart neither duplicates nor over-fills."""
    found=set()
    for line in subprocess.run(['ps','-eo','command'],capture_output=True,text=True).stdout.splitlines():
        a=line.split()
        # Only python processes running the trainer (not shells whose command text merely mentions it).
        if a and 'python' in a[0] and 'experiments.dials_ablations.run' in a and 'train' in a:
            try:found.add((a[a.index('--dataset')+1],a[a.index('--part')+1],int(a[a.index('--seed')+1])))
            except (ValueError,IndexError):pass
    return found


def main():
    LOGS.mkdir(parents=True,exist_ok=True)
    jobs=[(name,part,seed) for name in ORDER for part in PARTS for seed in (0,1,2)
          if not (OUT/part/name/f'seed{seed}'/'selected.json').exists()]
    running=[]
    while jobs or running or elsewhere():
        running=[(j,p) for j,p in running if p.poll() is None]
        orphans=elsewhere()-{j for j,_ in running}
        jobs=[j for j in jobs if j not in orphans]
        heavy=sum(j[0] in HEAVY for j,_ in running)+sum(j[0] in HEAVY for j in orphans)
        for job in list(jobs):
            if len(running)+len(orphans)>=TOTAL:break
            if job[0] in HEAVY and heavy>=HEAVY_MAX:continue
            name,part,seed=job;log=open(LOGS/f'{part}_{name}_seed{seed}.log','a')
            running.append((job,subprocess.Popen(command('train','--part',part,'--dataset',name,'--seed',str(seed)),stdout=log,stderr=subprocess.STDOUT,env=os.environ)))
            jobs.remove(job);heavy+=job[0] in HEAVY
            print(time.strftime('%H:%M'),'started',part,name,seed,f'({len(jobs)} queued)',flush=True);time.sleep(5)
        time.sleep(30)
    for name in ORDER:
        if not (OUT/'untrained'/name/'selected.json').exists():
            subprocess.run(command('untrained','--dataset',name),stdout=open(LOGS/f'untrained_{name}.log','a'),stderr=subprocess.STDOUT)
    subprocess.run(command('test'),stdout=open(OUT/'test.log','w'),stderr=subprocess.STDOUT)
    print(time.strftime('%H:%M'),'ALL DONE',flush=True)


if __name__=='__main__':
    main()
