"""Run every ablation job, at most 12 at once and at most 6 on the Spotify datasets, longest first; then test.

PYTHONPATH=.:../dials nohup .venv/bin/python -m experiments.dials_ablations.schedule > results/dials_ablations/schedule.log 2>&1 &
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from .run import PARTS,OUT

HEAVY=['spotify','spotify_sparse'];ORDER=['spotify','spotify_sparse','pog','pog_dense','clothing','electronic','food']
TOTAL,HEAVY_MAX=12,6
LOGS=OUT/'logs'


def command(*args):return [sys.executable,'-u','-m','experiments.dials_ablations.run',*args]


def main():
    LOGS.mkdir(parents=True,exist_ok=True)
    jobs=[(name,part,seed) for name in ORDER for part in PARTS for seed in (0,1,2)
          if not (OUT/part/name/f'seed{seed}'/'selected.json').exists()]
    running=[]
    while jobs or running:
        running=[(j,p) for j,p in running if p.poll() is None]
        heavy=sum(j[0] in HEAVY for j,_ in running)
        for job in list(jobs):
            if len(running)>=TOTAL:break
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
