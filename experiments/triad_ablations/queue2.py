"""Like queue.py, but safe to run beside jobs started elsewhere: a job is skipped while an identical command is
running (checked with ps just before starting it). Finished jobs are cheap to repeat, since every runner returns at
once when its outputs exist.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.triad_ablations.queue2 results/triad_queue.txt --parallel 5
"""
import argparse, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
LOCK = threading.Lock()

def running(line):
    out = subprocess.run(['ps', '-Ao', 'command'], capture_output=True, text=True).stdout
    return any(l.rstrip().endswith('-m ' + line) for l in out.splitlines())

def job(line):
    while running(line):time.sleep(60)  # started by another scheduler: wait for it, then the rerun is a no-op
    log = Path('results/triad_queue_logs')/(line.replace(' ', '_').replace('-', '')[:150] + '.log')
    code = subprocess.run(['nice', '-n', '10', sys.executable, '-m', *line.split()], stdout=open(log, 'a'), stderr=subprocess.STDOUT).returncode
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'exit', code, line, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('file');p.add_argument('--parallel', type=int, default=5);a = p.parse_args()
    lines = [l.strip() for l in open(a.file) if l.strip()]
    with ThreadPoolExecutor(a.parallel) as pool:list(pool.map(job, lines))
    print('QUEUE DONE', flush=True)
