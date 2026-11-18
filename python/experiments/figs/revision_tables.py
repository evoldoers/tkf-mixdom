#!/usr/bin/env python3
"""Assemble every number for the MBE paper-1 revision tables from the result files:
held-out (validation) and test-split likelihoods per pair under both conventions, and
BAliBASE / SABmark / OXBench accuracies (core-block and all-column SP/TC, BAliBASE F1).
Missing results print as blanks, so it can be re-run as the queues fill.
Usage: revision_tables.py [--json out.json] [--latex out.tex]"""
import argparse, glob, json, os, re, numpy as np
PY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(PY)
C20 = os.environ.get('C20_DIR') or f'{PY}/experiments/profile_mixture/results/fsa_c20_balibase'   # fsa_c20_balibase.py runs

def grab(path, pat):
    if not os.path.exists(path): return None
    m = re.findall(pat, open(path, errors='replace').read()); return float(m[-1]) if m else None
def val_pc(tag): return grab(f'experiments/ordsum_val/pathcond_{tag}.log', r'LL/pair=(-[\d.]+)')
def val_os(tag): return grab(f'experiments/ordsum_val/score_{tag}.log', r'LL/pair  = (-[\d.]+)')
def test_pc(tag): return grab(f'experiments/test_ll/pathcond_{tag}.log', r'LL/pair  = (-[\d.]+)')
def test_os(tag): return grab(f'experiments/test_ll/ordsum_{tag}.log', r'LL/pair  = (-[\d.]+)')
def mf(F, kind, split): return grab(f'experiments/mixfrag_F/{kind}_F{F}_{split}.log', r'LL/pair  = (-[\d.]+)')

def rows_json(path):
    if not os.path.exists(path): return None
    d = json.load(open(path)); v = d.get('per_family') or d.get('rows') or d
    if isinstance(v, dict): v = list(v.values())
    v = [r for r in v if isinstance(r, dict) and 'family' in r]
    return v or None
def acc(rows, g):
    if not rows: return None
    out = {}
    for k in ('sp', 'tc', 'sp_full', 'tc_full'):
        vals = [r.get(f'msa_{k}_{g}') for r in rows]; vals = [x for x in vals if x is not None]
        out[k] = float(np.mean(vals)) if vals else None
    key = 'micro_fsa_sps' if g == 'g0' else 'micro_hard'   # the harness's MSA residue-pair F1 at gap factor 0 (fsa_sps) or 1 (hard)
    m = [r[key] for r in rows if isinstance(r.get(key), dict)]
    if m:
        tp = sum(x['e_tp'] for x in m); mass = sum(x['total_mass'] for x in m); gold = sum(x['gold'] for x in m)
        p, rc = tp / mass, tp / gold; out['f1'] = 2 * p * rc / (p + rc)
    out['n'] = len(rows); return out
def c20_rows(path):
    if not os.path.exists(path): return None
    rows, seen = [], set()   # one row per family (a run that was accidentally started twice appends duplicates)
    for l in open(path):
        if not l.strip(): continue
        r = json.loads(l)
        if r.get('family') in seen: continue
        seen.add(r.get('family')); rows.append(r)
    return rows or None
def bench_mixdom(tag, b):   # b in bali3pdbm|sabrem|oxm
    if b == 'bali3pdbm': return acc(rows_json(f'experiments/expected_balibase/expected_balibase_mixdom_{tag}_withsps.json'), 'g0')
    return acc(rows_json(f'experiments/benchmarks/{b}_mixdom_{tag}.json'), 'g0')
def bench_c20(name, b):
    for suffix in (('corefull', 'bali3pdbm') if b == 'bali3pdbm' else (b,)):
        r = c20_rows(f'{C20}/{name}_LGp_one_{suffix}.jsonl')
        if r: return acc(r, 'g0')
    return None
def bench_ext(name, b):
    if b == 'bali3pdbm': return acc(rows_json(f'experiments/expected_balibase/expected_balibase_{name}_withsps.json'), 'g1')
    return acc(rows_json(f'experiments/benchmarks/{b}_{name}.json'), 'g1')

MODELS = []   # (label, group, val_pc, val_os, test_pc, test_os, bench(b))
MODELS.append(('TKF92 (Table 2 params)', 'fixed', grab('experiments/test_ll/pathcond2d_tkf92_val.log', r'LL/pair  = (-[\d.]+)'), val_os('tkf92') if os.path.exists('experiments/ordsum_val/score_tkf92.log') else -721.29, test_pc('tkf92'), test_os('tkf92'), lambda b: bench_c20('tkf92', b)))
MODELS.append(('MixFrag F=2 (Table 2 params)', 'fixed', None, -719.35, test_pc('mixfrag'), test_os('mixfrag'), lambda b: bench_c20('mixfrag', b)))
for F in (2, 3, 4, 5):
    MODELS.append((f'MixFrag F={F} (refit)', 'mixfrag', mf(F, 'pathcond', 'val'), mf(F, 'ordsum', 'val'), mf(F, 'pathcond', 'test'), mf(F, 'ordsum', 'test'), (lambda F: lambda b: bench_c20(f'mixfrag{F}', b))(F)))
for d, f in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1), (5, 1)):
    base = f'd{d}f{f}_nobali'
    for suffix, lab in (('_1620', 'EM 1620'), ('_adam_from1620', '+Adam'), ('_lbfgs_from1620', '+L-BFGS')):
        tag = base + suffix; ttag = f'd{d}f{f}_nobali_1620' if suffix == '_1620' else (f'd{d}f{f}_nobali_adam' if 'adam' in suffix else None)
        btag = tag if suffix == '_1620' else tag
        MODELS.append((f'MixDom d{d}f{f} cleaned, {lab}', 'clean', val_pc(tag), val_os(tag), test_pc(ttag) if ttag else None, test_os(ttag) if ttag else None,
                       (lambda tag, suffix: lambda b: bench_mixdom(tag if b == 'bali3pdbm' else tag.replace('_adam_from1620', '_adam').replace('_from1620', ''), b))(tag, suffix)))
for name, tag, ttag in (('MixDom d3f1 Table 4 (full corpus, 540)', 'd3f1_v4_fix', 'd3f1_table4'),):
    MODELS.append((name, 'full', val_pc('svibw_d3f1_perdomain_v4_fix'), val_os('d3f1_v4_fix'), test_pc(ttag), test_os(ttag), lambda b: bench_mixdom('d3f1_v4_fix', b) if b != 'bali3pdbm' else acc(rows_json('experiments/expected_balibase/expected_balibase_mixdom_d3f1_v4_fix_withsps.json'), 'g0')))
for d in (3, 2):
    for suffix, lab, ttag in (('_1620', 'EM 1620', f'd{d}f1_full_1620'), ('_adam_from1620', '+Adam', f'd{d}f1_full_adam'), ('_lbfgs_from1620', '+L-BFGS', None)):
        tag = f'd{d}f1_full{suffix}'
        MODELS.append((f'MixDom d{d}f1 full corpus, {lab}', 'full', val_pc(tag), val_os(tag), test_pc(ttag) if ttag else None, test_os(ttag) if ttag else None,
                       (lambda tag: lambda b: bench_mixdom(tag if b == 'bali3pdbm' else tag.replace('_adam_from1620', '_adam').replace('_from1620', ''), b))(tag)))
MODELS.append(('MixDom d2f1 tied mixture (control), EM 1620', 'control', val_pc('d2f1_tied_nobali_1620'), val_os('d2f1_tied_nobali_1620'), test_pc('d2f1_tied_nobali_1620'), test_os('d2f1_tied_nobali_1620'), lambda b: bench_mixdom('d2f1_tied_nobali_1620', b)))
MODELS.append(('MixDom d2f1 tied indel (control), EM 1620', 'control', val_pc('d2f1_tiedindel_nobali_1620'), val_os('d2f1_tiedindel_nobali_1620'), test_pc('d2f1_tiedindel_nobali_1620'), test_os('d2f1_tiedindel_nobali_1620'), lambda b: bench_mixdom('d2f1_tiedindel_nobali_1620', b)))
MODELS.append(('MAFFT', 'ext', None, None, None, None, lambda b: bench_ext('mafft', b)))
MODELS.append(('MUSCLE', 'ext', None, None, None, None, lambda b: bench_ext('muscle', b)))

def f(x, nd=2): return '' if x is None else f'{x:.{nd}f}'
def fa(a, k): return '' if not a or a.get(k) is None else f'{a[k]:.3f}'
ap = argparse.ArgumentParser(); ap.add_argument('--json', default='experiments/figs/revision_tables.json'); ap.add_argument('--latex', default=None); args = ap.parse_args()
out = []
hdr = ['model', 'val pc', 'val os', 'test pc', 'test os', 'BALI F1', 'BALI SP', 'BALI TC', 'BALI SPfull', 'BALI TCfull', 'SAB SP', 'SAB TC', 'SAB SPfull', 'SAB TCfull', 'OX SP', 'OX TC', 'OX SPfull', 'OX TCfull']
print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
for label, group, vpc, vos, tpc, tos, bench in MODELS:
    B = {b: bench(b) for b in ('bali3pdbm', 'sabrem', 'oxm')}
    rec = dict(model=label, group=group, val_pathcond=vpc, val_ordsum=vos, test_pathcond=tpc, test_ordsum=tos, bench=B); out.append(rec)
    cells = [label, f(vpc), f(vos), f(tpc), f(tos), fa(B['bali3pdbm'], 'f1')] + [fa(B[b], k) for b in ('bali3pdbm', 'sabrem', 'oxm') for k in ('sp', 'tc', 'sp_full', 'tc_full')]
    print('| ' + ' | '.join(cells) + ' |')
json.dump(out, open(args.json, 'w'), indent=1, default=float)
if args.latex:
    with open(args.latex, 'w') as fh:
        for r in out:
            B = r['bench']; fh.write(' & '.join([r['model'], f(r['val_pathcond']), f(r['val_ordsum']), f(r['test_pathcond']), f(r['test_ordsum'])] + [fa(B[b], k) for b in ('bali3pdbm', 'sabrem', 'oxm') for k in ('sp', 'tc', 'sp_full', 'tc_full')]) + ' \\\\\n')
