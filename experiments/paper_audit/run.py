"""Check every number of the paper's tables (and the key numbers quoted in its text) against the result files.

For each expected row the cells are recomputed from the sources below, formatted as the paper prints them (three
decimals, leading zero dropped) and searched for in paper/main.tex after removing \\textbf, $...$ marks and footnote
symbols. Prints OK / MISMATCH per check and writes results/paper_audit.json; exits 1 on any mismatch.

Sources
  Table 1  dials/paper/figdata/datasets.json
  Table 2  results/fair_prior/comparison.json (simple baselines with the prior, TRIAD, TRIAD w/o bonus);
           results/dials_frozen/summary.json (CLHE and RaMen reruns: e-commerce RaMen and CLHE with BLIP titles,
           CLHE on POG_DENSE / POG / Spotify_sparse); CLHE on full Spotify: ../clhe_seeds_out (one seed, all queries);
           RaMen on POG and Spotify: published (RaMen, Table 2)
  Table 3  results/fair_prior/comparison.json tests (via dials/paper/tables/tests.tex)
  Table 4  results/bundlemllm_protocol/hit1*.json, hit1_simple.json; results/ddbc_rerun/comparison(_no_bonus).json;
           results/ddbc_protocol/simple_triad_features.json
  Table 5  results/triad_ablations/summary.json
  Text     results/dials_frozen/summary.json (neural tests), results/catalogue_check/summary.json

cd ../bundlefuse && python -m experiments.paper_audit.run
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3];BF = ROOT/'bundlefuse'/'results';PAPER = ROOT/'dials'/'paper'
J = lambda p: json.loads(Path(p).read_text())
f3 = lambda x: f'{x:.3f}'.replace('0.', '.', 1) if x >= 0 else '$-$' + f'{-x:.3f}'.replace('0.', '.', 1)
results = []


def norm(s):
    s = re.sub(r'\\textbf\{([^}]*)\}', r'\1', s);s = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\$\^\{?\\?(dagger|ddagger|\*|S)\}?\$', '', s);s = s.replace('$^{*}$', '')
    return re.sub(r'\s+', ' ', s)


TEX = norm((PAPER/'main.tex').read_text());TESTS = norm((PAPER/'tables'/'tests.tex').read_text())


def check(label, expected, text=None):
    ok = norm(expected) in (text or TEX);results.append(dict(check=label, expected=expected, ok=ok))
    print('OK      ' if ok else 'MISMATCH', label, '|', expected, flush=True)


def table1():
    d = J(PAPER/'figdata'/'datasets.json')
    for k, name in (('electronic', 'Electronic'), ('food', 'Food'), ('clothing', 'Clothing'), ('pog_dense', 'POG\\_DENSE'),
                    ('pog', 'POG'), ('spotify_sparse', 'Spotify\\_sparse'), ('spotify', 'Spotify')):
        r = d[k];test = f"{r['test']:,}" + (f" ({r['clean_test']:,})" if r['clean_test'] != r['test'] else '')
        check(f'T1 {k}', f"{name} & {r['items']:,} & {r['train']:,} & {r['mean_size']:.1f} & {test} & {'; '.join(f'{v:,}' for v in r['levels'])}\\\\")


def table2():
    c = J(BF/'fair_prior'/'comparison.json')['datasets'];fz = J(BF/'dials_frozen'/'summary.json')['datasets']
    for block in (('electronic', 'food', 'clothing'), ('pog_dense', 'pog', 'spotify_sparse', 'spotify')):
        for m, label in (('itemknn', 'ItemKNN'), ('rp3beta', 'RP3$\\beta$'), ('ease', 'EASE'), ('hybrid', 'Content+ItemKNN'),
                         ('label_knn_content', 'Label+ItemKNN+content')):
            cells = []
            for d in block:
                t = c[d]['methods'].get(m + '+prior');cells += ['--', '--'] if t is None else [f3(t['test'][0]), f3(t['test'][1])]
            check(f'T2 {m} {block[0]}', f"{label} & " + ' & '.join(cells) + '\\\\')
        for key, label in (('triad', 'TRIAD'), ('triad_no_bonus', '\\emph{TRIAD w/o bonus}')):
            check(f'T2 {key} {block[0]}', f"{label} & " + ' & '.join(f3(v) for d in block for v in c[d][key]) + '\\\\')
    clhe = []
    for d in ('electronic', 'food', 'clothing'):clhe += [f3(v) for v in fz[d]['clhe_clean']['test']]
    check('T2 CLHE e-commerce', 'CLHE & ' + ' & '.join(clhe) + '\\\\')
    ram = []
    for d in ('electronic', 'food', 'clothing'):ram += [f3(v) for v in fz[d]['ramen_clean']['test']]
    check('T2 RaMen e-commerce', 'RaMen & ' + ' & '.join(ram) + '\\\\')
    sp = J(ROOT/'clhe_seeds_out'/'spotify'/'clhe_out'/'spotify'/'seed2023.json')
    cells = []
    for d in ('pog_dense', 'pog', 'spotify_sparse'):cells += [f3(v) for v in fz[d]['clhe']['test']]
    check('T2 CLHE outfits/playlists', 'CLHE & ' + ' & '.join(cells + [f3(sp['test'][0]), f3(sp['test'][1])]) + '\\\\')
    for d in ('pog_dense', 'pog', 'spotify_sparse'):
        check(f'T2 CLHE parameters {d}', f"{fz[d]['clhe']['parameters']/1e6:.1f}\\,M" if fz[d]['clhe']['parameters'] < 2e7 else f"{fz[d]['clhe']['parameters']/1e6:.0f}\\,M")
    for d in ('electronic', 'food', 'clothing'):check(f'T2 CLHE parameters {d}', f"{fz[d]['clhe_clean']['parameters']/1e6:.1f}\\,M")


def table3():
    t = J(BF/'fair_prior'/'comparison.json')['tests']['triad_vs_best_simple_prior']
    for r in t:
        if r['metric'] != 'ndcg20':continue
        d = r['delta'];lo, hi = r['ci95']
        s = f"{'+' if d >= 0 else '$-$'}{f3(abs(d))} [{f3(lo)}, {f3(hi)}]"
        check(f"T3 {r['dataset']}", s, TESTS)


def table4():
    h = J(BF/'bundlemllm_protocol'/'hit1.json');hs = J(BF/'bundlemllm_protocol'/'hit1_spotify.json')
    hn = J(BF/'bundlemllm_protocol'/'hit1_no_bonus.json');hsn = J(BF/'bundlemllm_protocol'/'hit1_spotify_no_bonus.json')
    simple = J(BF/'bundlemllm_protocol'/'hit1_simple.json')['datasets']
    mean = lambda a: sum(map(sum, a))/sum(map(len, a))
    pub = dict(pog_dense=('.668', '.637'), pog=('.413', '.348'), spotify_sparse=('.796', '.723'), spotify=('.824', '.757'))
    for d, name, src, srcn in (('pog_dense', 'POG\\_DENSE', h, hn), ('pog', 'POG', h, hn), ('spotify_sparse', 'Spotify\\_sparse', hs, hsn), ('spotify', 'Spotify', hs, hsn)):
        check(f'T4 HitRate {d}', f"{name} & {f3(mean(src[d]['all']))} & {f3(mean(srcn[d]['all']))} & {f3(simple[d]['best_simple']['hit1'])} & {pub[d][0]} & {pub[d][1]}\\\\")
    c = J(BF/'ddbc_rerun'/'comparison.json')['cases'];cn = J(BF/'ddbc_rerun'/'comparison_no_bonus.json')['cases']
    s = J(BF/'ddbc_protocol'/'simple_triad_features.json')['cases']
    best = {'pog_dense_k4': '.213 / .157', 'spotify_k30': '.153 / .090', 'spotify_k60': '.129 / .076', 'spotify_k90': '.123 / .070'}
    for k, name in (('pog_dense_k4', 'POG\\_DENSE $k{=}4$'), ('spotify_k30', 'Spotify $k{=}30$'), ('spotify_k60', 'Spotify $k{=}60$'), ('spotify_k90', 'Spotify $k{=}90$')):
        pair = lambda r, a='pool_f1', b='pool_jaccard': f"{f3(r[a])} / {f3(r[b])}"
        check(f'T4 DDBC {k}', f"{name} & {pair(c[k]['dials'])} & {pair(cn[k]['dials'])} & {pair(s[k])} & {pair(c[k]['ddbc'])} & {best[k]}\\\\")


def table5():
    a = J(BF/'triad_ablations'/'summary.json')['parts'];sets = ('electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse')
    check('T5 full', '\\emph{full model (NDCG@20)} & ' + ' & '.join(f3(a['loo'][d]['full'][1]) for d in sets) + '\\\\')
    for part, label in (('loo', 'leave-one-out statistics'), ('gate', 'gate (all weights 1)'), ('content', 'content term'),
                        ('labels', 'label terms'), ('graph', 'graph term'), ('untrained', 'training (initial parameters)')):
        check(f'T5 {part}', f'{label} & ' + ' & '.join(f3(a[part][d]['full'][1] - a[part][d]['ablated'][1]) for d in sets) + '\\\\')


def text():
    fz = J(BF/'dials_frozen'/'summary.json')['primary']
    for who, key in (('CLHE', 'clhe'), ('RaMen', 'ramen')):
        d = [r['delta'] for r in fz if r['family'] == 'neural' and r['metric'] == 'ndcg20' and r['second'].startswith(key)]
        check(f'text neural {who} range', f'+{f3(min(d))} to +{f3(max(d))}')
        assert all(r['reject_05'] for r in fz if r['family'] == 'neural'), 'a neural test does not reject'


if __name__ == '__main__':
    for f in (table1, table2, table3, table4, table5, text):f()
    bad = [r for r in results if not r['ok']]
    (BF/'paper_audit.json').write_text(json.dumps(dict(note=__doc__, checks=results), indent=1))
    print(f'{len(results) - len(bad)} of {len(results)} checks match')
    sys.exit(1 if bad else 0)
