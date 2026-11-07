"""Build a live cross-dataset table without modifying prior results."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/amplified_fallback'
OUT.mkdir(parents=True,exist_ok=True)
lines=['# Amplified neighbours with frequency fallback: live results','',
       'The first three datasets reuse the completed novelty-attribution study. '
       'The remaining five run in a separate follow-up with constant-mixture controls. '
       'Input checkpoint profiles and query limits differ by dataset; see the '
       '[protocol](../../experiments/amplified_fallback/README.md). Original references '
       'used all earlier validation queries; controlled variants use grouped 70/30 '
       'fit/tuning splits. Comparisons with the originals are descriptive.', '',
       '| Dataset | Status | Original R@20 | Original N@20 | Amplified weighted R@20 | '
       'Amplified weighted N@20 | Amplified fallback R@20 | Amplified fallback N@20 | Historical weight in amplified fallback |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|']
rows=[]
for name in ['pog_dense','spotify','pog','electronic','food','clothing','dunnhumby','spotify_sparse']:
    earlier=name in ['pog_dense','spotify','pog']
    directory=ROOT/'results'/('novelty_attribution' if earlier else 'amplified_fallback')/name
    path=directory/'summary.json'
    row=dict(dataset=name,status='running/incomplete' if (directory/'run.log').exists() else 'pending')
    numbers=['-']*6; historical='-'
    if path.exists():
        summary=json.loads(path.read_text())
        if summary.get('status')=='complete' and not summary.get('smoke'):
            row['status']='complete';methods=summary['methods']
            row['methods']={k:methods[k] for k in ['original_reference','amplified:weighted',
                             'amplified:bundlefuse_gate' if earlier else 'amplified:gate']}
            numbers=[f'{row["methods"][k][m]:.6f}' for k in row['methods'] for m in ['recall20','ndcg20']]
            selected=json.loads((directory/'selection/amplified/selected.json').read_text())
            gate=selected['bundlefuse_gate' if earlier else 'gate']
            row['historical_weight']=gate['weights'].get('hist',0.)
            historical=f'{row["historical_weight"]:g}'
    lines.append('| '+' | '.join([name,row['status'],*numbers,historical])+' |');rows.append(row)
lines+=['','Follow the queue with `tail -n 100 -F results/amplified_fallback/queue.log`.',
        'Per-dataset logs and reports are under `results/amplified_fallback/<dataset>/`. '
        'All prior and new checkpoints/caches are retained.', '',
        'A historical weight of zero means the selected model does not use amplified neighbours. '
        'An improved metric in such a row cannot be attributed to amplification.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
(OUT/'overview.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Updated results/amplified_fallback/README.md')
