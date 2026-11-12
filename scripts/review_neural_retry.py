"""Summarize independent neural retry without selecting models from test scores."""
from pathlib import Path
import json
import numpy as np


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    root=Path('results/neural_retry');root.mkdir(parents=True,exist_ok=True)
    rows={}
    for name,source in [('electronic',Path('results/neural_constrained/electronic')),
                        ('food',Path('results/neural_transfer/food')),
                        ('clothing',Path('results/neural_transfer/clothing'))]:
        p=source/'summary.json'
        if not p.exists() or json.loads(p.read_text())['status']!='complete':
            rows[name]={'status':'pending','source':str(source)};continue
        summary=json.loads(p.read_text());condition='graph_pair_no_category'
        keys=[f'{condition}_seed{s}' for s in [0,1,2]]
        values=np.array([[summary['test'][k][metric] for metric in ['recall20','ndcg20']] for k in keys])
        mean=values.mean(0);std=values.std(0,ddof=1)
        baseline=summary['baseline']['bundlefuse_matched']['test']
        result=dict(status='complete',source=str(source),condition=condition,
                    mean=dict(zip(['recall20','ndcg20'],mean.tolist())),
                    sample_std=dict(zip(['recall20','ndcg20'],std.tolist())),
                    relative_difference_vs_matched={k:float(mean[i]/baseline[k]-1) for i,k in enumerate(['recall20','ndcg20'])},
                    bundlefuse_matched={k:baseline[k] for k in ['recall20','ndcg20']},
                    bundlefuse_original={k:summary['baseline']['bundlefuse_original_auxiliary']['test'][k] for k in ['recall20','ndcg20']},
                    parameters=summary['selected'][keys[0]]['parameters'],
                    by_target_frequency={group:float(np.mean([summary['test'][k]['by_target_training_frequency'][group]['micro_recall20'] for k in keys]))
                                         for group in ['unseen','one_two','three_plus']})
        previous=Path(f'results/neural_small/{name}/summary.json')
        if previous.exists():
            old=json.loads(previous.read_text())
            if old.get('status')=='complete':result['previous_neural_means']=old['neural_means']
        probe=json.loads((source/keys[0]/'signal_probe.json').read_text())
        result['signal_probe_r2_seed0']=probe['heldout_probe_r2']
        rows[name]=result
    write_json(root/'summary.json',dict(datasets=rows,scope='Three fixed training seeds per dataset; architecture selected on Electronic validation then transferred unchanged. Same dataset-specific validation selects epochs. No test-based architecture selection; previously inspected splits.'))
    lines=['# Independent neural retry: completed evidence','',
           'The old queue was stopped during Clothing. All checkpoints were retained. Two diagnosis-driven revisions were tested on Electronic; the final architecture was selected using Electronic validation, then transferred unchanged to Food and Clothing.','',
           '| Dataset | Neural Recall@20 | Neural NDCG@20 | Matched BF Recall@20 | Matched BF NDCG@20 | Trainable parameters |',
           '|---|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        if r['status']!='complete':lines.append(f'| {name} | pending | pending | pending | pending | pending |');continue
        m=r['mean'];b=r['bundlefuse_matched']
        lines.append(f'| {name} | {m["recall20"]:.6f} | {m["ndcg20"]:.6f} | {b["recall20"]:.6f} | {b["ndcg20"]:.6f} | {r["parameters"]["trainable_parameters"]:,} |')
    lines += ['', 'Neural values are means across three seeds. All methods use the same test queries, catalogue, observed-item masks and metric implementation. The matched BF control also has category inputs, which the final neural architecture does not consume; the neural model uses a subset of those inputs. Original BF additionally uses auxiliary/user information. No RaMen comparison with unmatched published splits is made here.','',
              '## Why the model changed','',
              '- V1 compressed historical relationships into item vectors, exposed training frequency as a predictive input, and did not simulate genuinely unseen inputs when dropping identities.',
              '- Frozen validation probes found weak linear recovery of category, text, propagated-text and joint-history signals. Sparse learned link weights and neural attribute propagation improved source recovery, but did not by themselves match BF.',
              '- Validation interventions isolated harmful unrestricted category penalties. The final model removes that table and constrains direct relationship weights to be nonnegative during optimization.',
              '- The final model learns a small text-residual encoder, one graph propagation layer, and one scalar per observed directed training pair. It has no item-ID embedding table, no frequency input, no BF score or teacher, and no fixed frequency fallback.',
              '- Training uses every eligible catalogue item and simulates unseen items by masking all history paths incident on selected IDs. The final architecture has no separate joint-score term; that controlled candidate did not win Electronic validation.','',
              '## How signal recovery was verified','',
              'Synthetic assays verify exact supported pair/triple coefficient representation, gradient learning, set invariance, cold-history masking, checkpoint replay and deterministic resume. These are capacity/correctness checks, not accuracy guarantees. Separate Ridge probes fit BF source targets only to frozen representations on half of validation queries and evaluate the other half. They never train the completion network.','',
              'Probe dimension and representation type differ between revisions; source standardization and shared sparse support affect recoverability. Higher probe R2 does not prove causal mechanism recovery. The final scorer does not reproduce every BF raw score, nor does it need to copy each one to rank well. Spectral, Rao and category-saturation equivalence has not been established.','',
              '## Evidence and logs','',
              '- [Initial source probes](../neural_revision/diagnostics/electronic/SIGNAL_DIAGNOSTIC.md)',
              '- [Revision findings and validation interventions](../neural_revision/FINDINGS.md)',
              '- [Electronic final report](../neural_constrained/electronic/REPORT.md)',
              '- [Food transfer report](../neural_transfer/food/REPORT.md)',
              '- [Clothing transfer report](../neural_transfer/clothing/REPORT.md)',
              '- [Implementation and protocol](../../experiments/neural_revision/README.md)',
              '- [Machine-readable three-seed summaries](summary.json)','',
              'Checkpoints remain under checkpoints/neural_small, checkpoints/neural_revision, checkpoints/neural_constrained and checkpoints/neural_transfer. Every evaluated epoch retains optimizer/RNG state; source/protocol changes use separate roots and preserve the earlier source archive.','',
              'These are exploratory results on previously inspected splits. Numerical closeness, small seed variation or an interval crossing zero is not a non-inferiority certificate or evidence of universal superiority.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:({a:v[a] for a in ['mean','relative_difference_vs_matched','parameters']} if v['status']=='complete' else v) for k,v in rows.items()},indent=2))


if __name__=='__main__':main()
