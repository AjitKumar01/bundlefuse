"""Regenerate paired matrix means/stds through the fail-closed experiment runner."""
import os
import subprocess
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
result = subprocess.run([sys.executable, '-m', 'bundlefuse.experiments', '--modes', 'matrix',
                         '--output', 'results/matrix_refresh'], cwd=root)
raise SystemExit(result.returncode)
