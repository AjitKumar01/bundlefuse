"""Run a file of module invocations (one per line, 'module args...'), --parallel at a time, each with nice 10.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.triad_ablations.queue results/triad_queue.txt --parallel 3
"""
import argparse, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def job(line):
    log = Path('results/triad_queue_logs')/(line.replace(' ', '_').replace('-', '')[:150] + '.log')
    code = subprocess.run(['nice', '-n', '10', sys.executable, '-m', *line.split()], stdout=open(log, 'w'), stderr=subprocess.STDOUT).returncode
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'exit', code, line, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('file');p.add_argument('--parallel', type=int, default=3);a = p.parse_args()
    Path('results/triad_queue_logs').mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.parallel) as pool:list(pool.map(job, [l.strip() for l in open(a.file) if l.strip()]))
    print('QUEUE DONE', flush=True)
