"""Pool the complete planned contrast family after all four workers finish."""
import json
from pathlib import Path
from .protocol import PLAN
from .statistics import holm_adjust
from bundlefuse.provenance import write_json

def main():
    root=Path('results/publication_mechanisms')
    records=[];summaries={}
    for dataset in PLAN['datasets']:
        path=root/dataset/'summary.json'
        if not path.exists():raise RuntimeError(f'Incomplete worker: {dataset}')
        summaries[dataset]=json.loads(path.read_text())
        records.extend(dict(row,dataset=dataset) for row in summaries[dataset]['contrasts'])
    assert len(records)==PLAN['planned_tests']
    holm_adjust(records)
    write_json(root/'paired.json',records)
    lines=['# Completed mechanism studies','',
           'Four supplied splits; all validation/test queries. Follow-up analysis on previously inspected benchmarks.',
           'Paired bootstrap/sign-flip inference conditions on fitted models and fixed folds.',
           'Holm correction jointly covers all 96 planned metric contrasts. Stability variants are descriptive.',
           '', '| Dataset | Method | Recall@20 | NDCG@20 |','|---|---|---:|---:|']
    for dataset,summary in summaries.items():
        for method,values in summary['methods'].items():
            lines.append(f"| {dataset} | {method} | {values['recall20']:.6f} | {values['ndcg20']:.6f} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'status.json',dict(status='complete',datasets=list(summaries),contrasts=len(records)))

if __name__=='__main__':main()
