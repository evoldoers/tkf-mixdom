#!/usr/bin/env python3
"""Emit the MBE paper-1 results tables as LaTeX row files, straight from the result files
(via revision_tables.py's collectors), plus the paired tests quoted in the text.

Two consolidated tables (every number in the paper's alignment/likelihood tables comes from here):
  models_table.tex   one row per model fit: Pfam validation / test likelihood and BAliBASE, SABmark,
                     OXBench accuracy under the paper's standard decoding (annealing, gap factor 0)
  decoding_table.tex posterior source x decoding (MUSCLE's HMM ablation; our models under annealing,
                     consistency, progressive MEA, refinement) on the three benchmarks
  tails_table.tex    appendix: EM 1,620 / +Adam / +L-BFGS for d1f1..d5f1 and MixFrag-S F=2,3 (likelihood + BAliBASE)
Writes to ~/tkf-dp/molbiolevol-paper/paper1-baumwelch/tables/ (committed with the paper)."""
import json, os, sys, numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.dirname(os.path.dirname(HERE)); os.chdir(PY)
sys.argv = ['x']; exec(open(os.path.join(HERE, 'revision_tables.py')).read().split("ap = argparse.ArgumentParser()")[0])   # collectors + MODELS
OUT = os.environ.get('PAPER_TABLES_OUT') or os.path.expanduser('~/tkf-dp/molbiolevol-paper/paper1-baumwelch/tables'); os.makedirs(OUT, exist_ok=True)
EB, BM = 'experiments/expected_balibase', 'experiments/benchmarks'
BENCHES = ('bali3pdbm', 'sabrem', 'oxm')


N_FAM = {'bali3pdbm': 120, 'sabrem': 303, 'oxm': 336}


def complete(rows, b):
    """A result file counts only when every family of the benchmark is in it (runs write incrementally)."""
    return rows if rows and len(rows) >= N_FAM[b] else None


def acc_any(rows, g, b=None):
    """revision_tables.acc, plus pooled F1 from the FSA-hard micro counts (decoding scripts: micro_hard_<g>;
    external aligners: micro_hard)."""
    if b is not None: rows = complete(rows, b)
    out = acc(rows, g)
    if out is not None and out.get('f1') is None:
        for key in (f'micro_hard_{g}', 'micro_hard'):
            m = [r[key] for r in rows if isinstance(r.get(key), dict)]
            if m:
                tp = sum(x['e_tp'] for x in m); mass = sum(x['total_mass'] for x in m); gold = sum(x['gold'] for x in m)
                p, rc = tp / mass, tp / gold; out['f1'] = 2 * p * rc / (p + rc) if p + rc else None
                break
    return out


def first(*paths):
    for p in paths:
        if os.path.exists(p): return p
    return None


# ---------------------------------------------------------------- benchmark accuracy by source
def bench_harness(name, b, g='g0'):
    """Models run through expected_pairwise_balibase.py (posteriors cached under <name>__<bench>)."""
    p = first(f'{EB}/expected_balibase_{name}_withsps.json') if b == 'bali3pdbm' else first(f'{BM}/{b}_{name}.json')
    return acc_any(rows_json(p), g, b) if p else None


def bench_c20_any(name, b, g='g0'):
    """tkf-dp fsa_c20_balibase.py runs (TKF92 / MixFrag with the published LG, one clock; the anchored CML20)."""
    for suffix in (('corefull', 'bali3pdbm', '') if b == 'bali3pdbm' else (b,)):
        p = f'{C20}/{name}_{suffix}.jsonl' if suffix else f'{C20}/{name}.jsonl'
        r = complete(c20_rows(p), b)
        if r: return acc_any(r, g)
    return None


def bench_mixdom_any(tag, b, g='g0'):
    p = f'{EB}/expected_balibase_mixdom_{tag}_withsps.json' if b == 'bali3pdbm' else f'{BM}/{b}_mixdom_{tag}.json'
    return acc_any(rows_json(p), g, b) if os.path.exists(p) else None


def bench_ext_any(name, b):
    p = f'{EB}/expected_balibase_{name}_withsps.json' if b == 'bali3pdbm' else f'{BM}/{b}_{name}.json'
    return acc_any(rows_json(p), 'g1', b) if os.path.exists(p) else None


# ---------------------------------------------------------------- matched decoding for the models table
# Every row of the models table is decoded by progressive MEA without consistency or refinement (MUSCLE's HMM by
# MUSCLE's own code with -consiters 0 -refineiters 0); the annealing results stay in the decoding table.
def bench_prog(m, b):
    """m = model stem of the cached posteriors (e.g. 'mixdom_d2f1_nobali_1620', 'tkf92_LGp_one')."""
    if b == 'bali3pdbm':
        p = first(f'{EB}/msa_{m}_progressive_c0_r0.json', f'{EB}/msa_{m}_prog_c0_r0.json')
    else:
        p = first(f'{BM}/{b}_{m}_progressive_c0_r0.json')
    return acc_any(rows_json(p), 'g0', b) if p else None


def stem_mixdom(tag, b):
    """cache stem of a MixDom model per benchmark (the Adam points were cached under different names)."""
    if tag.endswith('_adam_from1620'):
        return 'mixdom_' + tag.replace('_adam_from1620', '_adam')   # the decoding files use the short name on every benchmark
    return 'mixdom_' + tag


MODELS_DECODING = os.environ.get('MODELS_DECODING', 'progressive')   # 'progressive' (matched) or 'anneal' (the old tables)


def row_bench(stem_fn, anneal_fn):
    """bench(b) for a models-table row: progressive from cache (matched), or the annealing numbers of the main runs."""
    if MODELS_DECODING == 'anneal': return anneal_fn
    return lambda b: bench_prog(stem_fn(b), b)


# ---------------------------------------------------------------- likelihoods
# path-conditioned validation scores: one code path (the 2D scorer with PATHCOND=1) for every row once available;
# the fixed models' 1D-scorer values are the fallback until their 2D runs exist (MixDom's 1D and 2D values coincide)
def pc2d(tag, fallback):
    for d in ('experiments/ordsum_val', 'experiments/test_ll'):
        v = grab(f'{d}/pathcond2d_{tag}_val.log', r'LL/pair  = (-[\d.]+)')
        if v is not None: return v
    return fallback
TABLE2_PC = {'tkf92': pc2d('tkf92', -725.62), 'mixfrag_F2': pc2d('mixfrag', -725.70)}
def mix_ll(tag):
    """K=20 / C=20 / refitted fixed models: path-conditioned validation and test log-likelihood per pair from the
    launch_mixture_ll.sh / launch_pathcond_test.sh outputs ('--' while the test run is pending)."""
    return ll(grab(f'experiments/ordsum_val/pathcond_{tag}.log', r'LL/pair  = (-[\d.]+)'), test_pc(tag))


# every likelihood in the tables is path-conditioned: vpc = validation split, tpc = clan-disjoint test split
def ll(vpc, tpc): return dict(vpc=vpc, tpc=tpc)


# ---- MixFrag-S: MixFrag with one substitution matrix (S_f, pi_f) per fragtype, fitted by svi-BW on the
# BAliBASE-free split. Fixed naming: checkpoints pfam/svibw_<tag>_1620.npz with the gradient tails
# pfam/{adam,lbfgs}_<tag>_from1620.npz; scorer logs <dir>/{pathcond,score,ordsum}_<tag>_<suffix>.log; benchmark
# runs cache_runs/<tag>_<suffix>_<bench>.jsonl and decodings msa_<tag>_<suffix>_<decoding>.json. Missing files = '--'.
MFS_TAGS = {2: 'mixfrags_F2_nobali', 3: 'mixfrags_F3_nobali'}
MFS_SUFFIXES = (('1620', 'EM 1620'), ('adam_from1620', '+Adam'), ('lbfgs_from1620', '+L-BFGS'))
MFS_LL_PAT = r'LL/pair\s*=\s*(-[\d.]+)'


def mfs_stem(F, suffix='1620'): return f'{MFS_TAGS[F]}_{suffix}'


def mfs_ll(F, suffix='1620'):
    """path-conditioned validation and test log-likelihood from the held-out scorer logs (each prints 'LL/pair  = v')."""
    t = mfs_stem(F, suffix)
    return ll(grab(f'experiments/ordsum_val/pathcond_{t}.log', MFS_LL_PAT), grab(f'experiments/test_ll/pathcond_{t}.log', MFS_LL_PAT))


def mfs_anneal(F, suffix='1620'):
    """bench(b, g) of the model's own fsa_c20_balibase.py run (annealing), like the fixed models' cache_runs."""
    return lambda b, g='g0': bench_c20_any(f'cache_runs/{mfs_stem(F, suffix)}', b, g)


def mfs_label(F): return f'\\MixFrag-S $\\mathcal F{{=}}{F}$'


def mfs_params(F):
    """ins, del, F extensions, F-1 fragtype weights; one GTR (S_f, pi_f) per fragtype (F=2: 423, F=3: 634)."""
    return 2 + F + (F - 1) + F * GTR


def mfs_row(F, suffix, fit):
    """a models-table row: likelihoods from the scorer logs, accuracy via the matched decoding of the cached posteriors."""
    return (mfs_label(F), fit, mfs_params(F), mfs_ll(F, suffix), row_bench(lambda b: mfs_stem(F, suffix), mfs_anneal(F, suffix)))


# ---- free-parameter counts (A = 20 residues: a GTR has 190 exchangeabilities + 19 equilibrium frequencies = 209)
GTR = 190 + 19


def mixdom_params(D, F, tied_class=False, tied_indel=False):
    """MixDom-dDfF with C = D substitution matrices: domain-level ins/del (2); per domain type ins, del and F
    fragtype-extension probabilities (the diagonal of the FxF extension matrix; the F=2 fits freeze the off-diagonal
    entries at zero, --freeze-ext-offdiag; shared if tied_indel); domain weights (D-1); fragtype weights D(F-1); the class
    distribution, fixed to the identity (one matrix per domain type) in every fit except the class-tied control,
    which fits one mixture over the C matrices shared by all domain types (C-1 per fragtype); and C matrices (S, pi)."""
    C = D
    indel = (2 + F) * (1 if tied_indel else D)
    classdist = F * (C - 1) if tied_class else 0
    return 2 + indel + (D - 1) + D * (F - 1) + classdist + C * GTR


# (row label, fitting, free parameters, likelihoods, bench(b)); None likelihoods = block header text
def mixdom_tag(d, f):  # F=2 rows: the diagonal-extension refits (off-diagonal fragtype switching frozen at zero)
    return f'd{d}f{f}dg_nobali_1620' if f == 2 else f'd{d}f{f}_nobali_1620'


MODEL_ROWS = [
    ('Fixed gap-length distributions, full \\Pfam\\ split', None, None, None, None),
    ('TKF92', 'counts EM', 3 + GTR, mix_ll('tkf92_gtrrefit'), row_bench(lambda b: 'tkf92_GTR_one', lambda b: bench_c20_any('cache_runs/tkf92_GTR_one', b))),
    ('TKF92, $K{=}20$ mixture', 'family-level EM', 20 * (3 + GTR) + 19, mix_ll('tkf92_K20'), row_bench(lambda b: 'tkf92_K20', lambda b: bench_harness('tkf92_K20', b))),
    ('TKF92, $C{=}20$ site mixture', 'counts EM', 3 + 20 * GTR + 19, mix_ll('cherryml_C20pc'), row_bench(lambda b: 'tkf92_CML20pc_anchor', lambda b: bench_c20_any('cache_runs/tkf92_CML20pc_anchor', b))),
    ('TKF92, $K{=}20$ indel, $C{=}20$ site classes', 'family-level EM', 20 * 3 + 19 + 20 * GTR + 19, mix_ll('tkf92_K20tied'), row_bench(lambda b: 'tkf92_K20tied', lambda b: bench_harness('tkf92_K20tied', b))),
    ('TKF92, $K{=}20$ indel classes, one matrix', 'family-level EM', 20 * 3 + 19 + GTR, mix_ll('tkf92_K20tiedgtr'), row_bench(lambda b: 'tkf92_K20tiedgtr', lambda b: bench_harness('tkf92_K20tiedgtr', b))),
    ('TKF92, $K{=}20$ matrices, one indel process', 'family-level EM', 20 * GTR + 19 + 3, mix_ll('tkf92_K20tiedindel'), row_bench(lambda b: 'tkf92_K20tiedindel', lambda b: bench_harness('tkf92_K20tiedindel', b))),
    ('\\MixFrag\\ $\\mathcal F{=}2$', 'counts EM', 5 + GTR, mix_ll('mixfrag_gtrrefit'), row_bench(lambda b: 'mixfrag_GTR_one', lambda b: bench_c20_any('cache_runs/mixfrag_GTR_one', b))),
    ('\\MixFrag\\ $\\mathcal F{=}3$', 'counts EM', 7 + GTR, mix_ll('mixfrag3_gtrrefit'), row_bench(lambda b: 'mixfrag3_GTR_one', lambda b: bench_c20_any('cache_runs/mixfrag3_GTR_one', b))),
    ('\\MixFrag\\ with one substitution matrix per fragtype (\\MixFrag-S), BAliBASE-free \\Pfam\\ split', None, None, None, None),
    mfs_row(2, '1620', '\\svibw, 1{,}620 it.'),
    mfs_row(3, '1620', '\\svibw, 1{,}620 it.'),
    ('\\MixDom, BAliBASE-free \\Pfam\\ split', None, None, None, None),
] + [
    (f'\\MixDom-d{d}f{f}', '\\svibw, 1{,}620 it.', mixdom_params(d, f), ll(val_pc(mixdom_tag(d, f)), test_pc(mixdom_tag(d, f))),
     (lambda t: row_bench(lambda b: stem_mixdom(t, b), lambda b: bench_mixdom_any(t, b)))(mixdom_tag(d, f)))
    for d, f in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1), (5, 1))
] + [
    ('Controls: d2f1 with one parameter set tied across domain types, BAliBASE-free split', None, None, None, None),
    ('\\MixDom-d2f1, class mixture tied', '\\svibw, 1{,}620 it.', mixdom_params(2, 1, tied_class=True), ll(val_pc('d2f1_tied_nobali_1620'), test_pc('d2f1_tied_nobali_1620')), row_bench(lambda b: stem_mixdom('d2f1_tied_nobali_1620', b), lambda b: bench_mixdom_any('d2f1_tied_nobali_1620', b))),
    ('\\MixDom-d2f1, indel process tied', '\\svibw, 1{,}620 it.', mixdom_params(2, 1, tied_indel=True), ll(val_pc('d2f1_tiedindel_nobali_1620'), test_pc('d2f1_tiedindel_nobali_1620')), row_bench(lambda b: stem_mixdom('d2f1_tiedindel_nobali_1620', b), lambda b: bench_mixdom_any('d2f1_tiedindel_nobali_1620', b))),
    ('Gradient tails from the 1{,}620-iteration fit (2{,}000 Adam steps; see \\Cref{tab:tails})', None, None, None, None),
    ('\\MixDom-d2f1', '\\svibw\\ + Adam', mixdom_params(2, 1), ll(val_pc('d2f1_nobali_adam_from1620'), test_pc('d2f1_nobali_adam')), row_bench(lambda b: stem_mixdom('d2f1_nobali_adam_from1620', b), lambda b: bench_mixdom_any('d2f1_nobali_adam_from1620' if b == 'bali3pdbm' else 'd2f1_nobali_adam', b))),
    ('\\MixDom-d3f1', '\\svibw\\ + Adam', mixdom_params(3, 1), ll(val_pc('d3f1_nobali_adam_from1620'), test_pc('d3f1_nobali_adam')), row_bench(lambda b: stem_mixdom('d3f1_nobali_adam_from1620', b), lambda b: bench_mixdom_any('d3f1_nobali_adam_from1620' if b == 'bali3pdbm' else 'd3f1_nobali_adam', b))),
    mfs_row(2, 'adam_from1620', '\\svibw\\ + Adam'),
    mfs_row(3, 'adam_from1620', '\\svibw\\ + Adam'),
    ('\\MixDom, full \\Pfam\\ split (contains relatives of BAliBASE proteins)', None, None, None, None),
    ('\\MixDom-d3f1', '\\svibw, 540 it.', mixdom_params(3, 1), ll(val_pc('svibw_d3f1_perdomain_v4_fix'), test_pc('d3f1_table4')), row_bench(lambda b: 'mixdom_d3f1_v4_fix', lambda b: bench_mixdom_any('d3f1_v4_fix', b) if b != 'bali3pdbm' else acc_any(rows_json(f'{EB}/expected_balibase_mixdom_d3f1_v4_fix_withsps.json'), 'g0'))),
    ('\\MixDom-d3f1', '\\svibw, 1{,}620 it.', mixdom_params(3, 1), ll(val_pc('d3f1_full_1620'), test_pc('d3f1_full_1620')), row_bench(lambda b: 'mixdom_d3f1_full_1620', lambda b: bench_mixdom_any('d3f1_full_1620', b))),
    ('\\MixDom-d2f1', '\\svibw, 1{,}620 it.', mixdom_params(2, 1), ll(val_pc('d2f1_full_1620'), test_pc('d2f1_full_1620')), row_bench(lambda b: 'mixdom_d2f1_full_1620', lambda b: bench_mixdom_any('d2f1_full_1620', b))),
    ('Alignment programs (no fitting)', None, None, None, None),
    ('MAFFT (\\texttt{-{}-auto})', '--', None, ll(None, None), lambda b: bench_ext_any('mafft', b)),
    ('MUSCLE 5.3 HMM, progressive' if MODELS_DECODING == 'progressive' else 'MUSCLE 5.3 (default)', '--', None, ll(None, None), lambda b: bench_ext_any('muscle_cons0refine0' if MODELS_DECODING == 'progressive' else 'muscle', b)),
]

NCOL = 14   # label + fitting + params + 2 LL (val, test) + 3 x (SP, TC, F1)
NUMERIC = NCOL - 3   # columns ranked for bold / underline


def fmt(x, nd, best=None, second=None):
    """Bold the best value in a column, underline the second best (ties at the printed precision share a rank)."""
    if x is None: return '--'
    s = f'{x:.{nd}f}'; tol = 0.5 * 10 ** (-nd)
    if best is not None and abs(x - best) < tol: return f'\\textbf{{{s}}}'
    if second is not None and abs(x - second) < tol: return f'\\underline{{{s}}}'
    return s


def top2(values, nd):
    """(best, second best) among distinct printed values."""
    vs = sorted({round(v, nd) for v in values if v is not None}, reverse=True)
    return (vs[0] if vs else None), (vs[1] if len(vs) > 1 else None)


def model_cells(L, B):
    return [(L['vpc'], 2), (L['tpc'], 2),
            (B['bali3pdbm'] and B['bali3pdbm'].get('sp'), 3), (B['bali3pdbm'] and B['bali3pdbm'].get('tc'), 3), (B['bali3pdbm'] and B['bali3pdbm'].get('f1'), 3),
            (B['sabrem'] and B['sabrem'].get('sp'), 3), (B['sabrem'] and B['sabrem'].get('tc'), 3), (B['sabrem'] and B['sabrem'].get('f1'), 3),
            (B['oxm'] and B['oxm'].get('sp'), 3), (B['oxm'] and B['oxm'].get('tc'), 3), (B['oxm'] and B['oxm'].get('f1'), 3)]


table = []
for label, fit, npar, L, bench in MODEL_ROWS:
    if L is None: table.append((label, None)); continue
    B = {b: bench(b) for b in BENCHES}
    table.append((f'{label} & {fit} & ' + ('--' if npar is None else f'{npar:,}'), model_cells(L, B)))
best = [top2([c[j][0] for _, c in table if c], c0[j][1]) for j, c0 in ((j, next(c for _, c in table if c)) for j in range(NUMERIC))]
MODELS_HEAD = r'''\begin{tabular}{@{}llrccccccccccc@{}}
\toprule
 & & & \multicolumn{2}{c}{\Pfam\ log-likelihood per pair} & \multicolumn{3}{c}{BAliBASE} & \multicolumn{3}{c}{SABmark} & \multicolumn{3}{c}{OXBench} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}
Model & Fitting & Params & val. & test & SP & TC & $F_1$ & SP & TC & $F_1$ & SP & TC & $F_1$ \\
'''
with open(f'{OUT}/models_table.tex', 'w') as fh:
    fh.write(MODELS_HEAD)
    for label, cells in table:
        if cells is None:
            fh.write(f"\\midrule\n\\multicolumn{{{NCOL}}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\\n"); continue
        fh.write(label + ' & ' + ' & '.join(fmt(v, nd, *best[j]) for j, (v, nd) in enumerate(cells)) + ' \\\\\n')
    fh.write('\\bottomrule\n\\end{tabular}\n')

# ---------------------------------------------------------------- decoding table
DEC = [('progressive', 0, 0, 'P'), ('progressive', 0, 100, 'P+R'), ('progressive', 2, 0, 'C+P'),
       ('progressive', 2, 100, 'C+P+R'), ('anneal', 0, 0, 'A'), ('anneal1', 0, 0, 'G'), ('anneal', 2, 0, 'C+A'), ('anneal_g1', 2, 0, 'C+G')]


def dec_file(m, b, algo, c, r):
    if algo == 'anneal_g1':   # consistency, then annealing at gap factor 1
        return first(f'{EB}/msa_{m}_anneal_c{c}_r{r}_g1.json') if b == 'bali3pdbm' else first(f'{BM}/{b}_{m}_anneal_c{c}_r{r}_g1.json')
    if b == 'bali3pdbm':
        if (algo, c, r) == ('anneal', 0, 0): return None   # the standard decoding: taken from the model's main run
        if (algo, c, r) == ('anneal', 2, 0):
            return first(f'{EB}/msa_{m}_anneal_c2_r0.json', f'{EB}/consistency_{m}_c2_probcons.json')
        return first(f'{EB}/msa_{m}_{algo}_c{c}_r{r}.json', f'{EB}/msa_{m}_prog_c{c}_r{r}.json')
    if (algo, c, r) == ('anneal', 0, 0): return None
    return first(f'{BM}/{b}_{m}_{algo}_c{c}_r{r}.json')


def dec_acc(m, main_bench, b, algo, c, r):
    if algo == 'anneal1': return main_bench(b, 'g1')   # the main run's gap-factor-1 annealing (the submitted setting)
    if (algo, c, r) == ('anneal', 0, 0): return main_bench(b)
    p = dec_file(m, b, algo, c, r)
    return acc_any(rows_json(p), 'g1' if algo == 'anneal_g1' else 'g0', b) if p else None


FIXED_SUBSET = ('prog00', 'prog2100', 'anneal00', 'anneal1', 'anneal20')
DEC_MODELS = [('TKF92', 'tkf92_GTR_one', lambda b, g='g0': bench_c20_any('cache_runs/tkf92_GTR_one', b, g), None),
              ('\\MixFrag\\ $\\mathcal F{=}2$', 'mixfrag_GTR_one', lambda b, g='g0': bench_c20_any('cache_runs/mixfrag_GTR_one', b, g), None),
              (mfs_label(2), mfs_stem(2), mfs_anneal(2), None),
              (mfs_label(3), mfs_stem(3), mfs_anneal(3), None),
              ('\\MixDom-d1f1', 'mixdom_d1f1_nobali_1620', lambda b, g='g0': bench_mixdom_any('d1f1_nobali_1620', b, g), None),
              ('\\MixDom-d2f1', 'mixdom_d2f1_nobali_1620', lambda b, g='g0': bench_mixdom_any('d2f1_nobali_1620', b, g), None),
              ('\\MixDom-d3f1', 'mixdom_d3f1_nobali_1620', lambda b, g='g0': bench_mixdom_any('d3f1_nobali_1620', b, g), None),
              ('\\MixDom-d4f1', 'mixdom_d4f1_nobali_1620', lambda b, g='g0': bench_mixdom_any('d4f1_nobali_1620', b, g), None),
              ('\\MixDom-d5f1', 'mixdom_d5f1_nobali_1620', lambda b, g='g0': bench_mixdom_any('d5f1_nobali_1620', b, g), None),
              ('\\MixDom-d2f1 (full split)', 'mixdom_d2f1_full_1620', lambda b, g='g0': bench_mixdom_any('d2f1_full_1620', b, g), None)]
MUSCLE_ROWS = [('P', 'muscle_cons0refine0'), ('P+R', 'muscle_consiters0'), ('C+P', 'muscle_refineiters0'), ('C+P+R (default)', 'muscle')]


def dcells(B):
    return [(B['bali3pdbm'] and B['bali3pdbm'].get('sp'), 3), (B['bali3pdbm'] and B['bali3pdbm'].get('tc'), 3), (B['bali3pdbm'] and B['bali3pdbm'].get('f1'), 3),
            (B['sabrem'] and B['sabrem'].get('sp'), 3), (B['sabrem'] and B['sabrem'].get('tc'), 3), (B['sabrem'] and B['sabrem'].get('f1'), 3),
            (B['oxm'] and B['oxm'].get('sp'), 3), (B['oxm'] and B['oxm'].get('tc'), 3), (B['oxm'] and B['oxm'].get('f1'), 3)]


dtable = []
for lab, name in MUSCLE_ROWS:
    dtable.append(('MUSCLE HMM', lab, dcells({b: bench_ext_any(name, b) for b in BENCHES})))
dtable.append(None)
for lab, m, main, subset in DEC_MODELS:
    for algo, c, r, dlab in DEC:
        key = ('anneal00' if (algo, c, r) == ('anneal', 0, 0) else 'anneal1' if algo == 'anneal1' else 'anneal20' if (algo, c, r) == ('anneal', 2, 0) else 'anneal2g1' if algo == 'anneal_g1' else f'prog{c}{r}')
        if subset and key not in subset: continue
        dtable.append((lab, dlab, dcells({b: dec_acc(m, main, b, algo, c, r) for b in BENCHES})))
dbest = [top2([row[2][j][0] for row in dtable if row], 3) for j in range(9)]
DEC_HEAD = r'''\begin{tabular}{@{}llccccccccc@{}}
\toprule
 & & \multicolumn{3}{c}{BAliBASE} & \multicolumn{3}{c}{SABmark} & \multicolumn{3}{c}{OXBench} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
Posteriors & Decoding & SP & TC & $F_1$ & SP & TC & $F_1$ & SP & TC & $F_1$ \\
\midrule
'''
# the decoding table is split in two floats (it does not fit one page): MUSCLE's HMM and the fixed gap-length
# models, then the MixDom models
SPLIT_AT = '\\MixDom-d1f1'
def write_dec(path, rows):
    with open(path, 'w') as fh:
        fh.write(DEC_HEAD); prev = None
        for row in rows:
            if row is None: fh.write('\\midrule\n'); prev = None; continue
            src, dlab, cells = row
            if prev is not None and src != prev: fh.write('\\addlinespace[2pt]\n')
            fh.write(f"{src if src != prev else ''} & {dlab} & " + ' & '.join(fmt(v, nd, *dbest[j]) for j, (v, nd) in enumerate(cells)) + ' \\\\\n'); prev = src
        fh.write('\\bottomrule\n\\end{tabular}\n')
write_dec(f'{OUT}/decoding_table.tex', dtable)   # one table; the paper puts its legend on the following page
with open(f'{OUT}/decoding_table_body.tex', 'w') as fh:   # rows only, for the paper's longtable (spans pages)
    prev = None
    for row in dtable:
        if row is None: fh.write('\\midrule\n'); prev = None; continue
        src, dlab, cells = row
        if prev is not None and src != prev: fh.write('\\addlinespace[2pt]\n')
        fh.write(f"{src if src != prev else ''} & {dlab} & " + ' & '.join(fmt(v, nd, *dbest[j]) for j, (v, nd) in enumerate(cells)) + ' \\\\\n'); prev = src
for stale in ('decoding_table_mixdom.tex',):
    if os.path.exists(f'{OUT}/{stale}'): os.remove(f'{OUT}/{stale}')


# ---------------------------------------------------------------- appendix: optimizer tails
# (validation path-conditioned LL, test path-conditioned LL, bench); revision_tables.MODELS still carries the
# order-summed slots, which no table prints
rows = {label: (vpc, tpc, {b: bench(b) for b in BENCHES}) for label, _, vpc, _vos, tpc, _tos, bench in MODELS}
for F in (2, 3):
    for suffix, lab in MFS_SUFFIXES:
        L = mfs_ll(F, suffix)
        rows[f'MixFrag-S F={F}, {lab}'] = (L['vpc'], L['tpc'], {b: mfs_anneal(F, suffix)(b) for b in BENCHES})


def f(x, nd=2):
    return '--' if x is None else f'{x:.{nd}f}'


def accf(B, b, k, nd=3): return f(B[b][k], nd) if B.get(b) and B[b].get(k) is not None else '--'


TAILS_HEAD = r'''\begin{tabular}{@{}lccccccccccc@{}}
\toprule
 & \multicolumn{4}{c}{\svibw, 1{,}620 it.} & \multicolumn{4}{c}{\svibw\ + Adam} & \multicolumn{3}{c}{\svibw\ + L-BFGS} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-12}
Model & LL & SP & TC & h & LL & SP & TC & h & LL & SP & TC \\
\midrule
'''
# Training time (hours) per model: the uncontested wall-clock session of 2026-11-14 (experiments/timing/), per-iteration
# median over iterations 21-80 (Adam: steps 21-124) x 1,620 iterations (2,000 steps) + start-up, from block_stats.json.
_BS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'timing', 'scratch', 'block_stats.json')
_TIME = json.load(open(_BS)) if os.path.exists(_BS) else {}


def hrs(key):
    return f"{_TIME[key]['total_h']:.1f}" if key in _TIME else '--'

with open(f'{OUT}/tails_table.tex', 'w') as fh:
    fh.write(TAILS_HEAD)
    for d in (1, 2, 3, 4, 5):
        cells = []
        for suf, tkey in (('EM 1620', f'd{d}f1'), ('+Adam', f'adam_d{d}f1'), ('+L-BFGS', None)):
            vpc, tpc, B = rows[f'MixDom d{d}f1 cleaned, {suf}']
            cells.append(f"{f(vpc)} & {accf(B,'bali3pdbm','sp')} & {accf(B,'bali3pdbm','tc')}" + (f" & {hrs(tkey)}" if tkey else ''))
        fh.write(f"\\MixDom-d{d}f1 & " + ' & '.join(cells) + " \\\\\n")
    fh.write("\\midrule\n")
    for F in (2, 3):
        cells = []
        for i, (_, suf) in enumerate(MFS_SUFFIXES):
            vpc, tpc, B = rows[f'MixFrag-S F={F}, {suf}']
            tkey = (f'mixfrags_F{F}', f'adam_mixfrags_F{F}', None)[i]
            cells.append(f"{f(vpc)} & {accf(B,'bali3pdbm','sp')} & {accf(B,'bali3pdbm','tc')}" + (f" & {hrs(tkey)}" if tkey else ''))
        fh.write(f"{mfs_label(F)} & " + ' & '.join(cells) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
for stale in ('ladder_rows.tex', 'fullcorpus_rows.tex', 'test_rows.tex', 'models_rows.tex', 'decoding_rows.tex', 'tails_rows.tex'):
    p = f'{OUT}/{stale}'
    if os.path.exists(p): os.remove(p)

# ---------------------------------------------------------------- paired tests quoted in the text
def fam_rows(tag):
    if MODELS_DECODING == 'progressive':
        p = first(f'{EB}/msa_{stem_mixdom(tag, "bali3pdbm")}_progressive_c0_r0.json', f'{EB}/msa_{stem_mixdom(tag, "bali3pdbm")}_prog_c0_r0.json')
    else:
        p = f'{EB}/expected_balibase_mixdom_{tag}_withsps.json'
    rows = complete(rows_json(p), 'bali3pdbm') if p and os.path.exists(p) else None
    return {r['family']: r for r in rows} if rows else None


def ptest_rows(A, Bb):
    if not A or not Bb: return None
    c = [x for x in A if x in Bb]
    return {k: (float(np.mean([A[x][f'msa_{k}_g0'] for x in c]) - np.mean([Bb[x][f'msa_{k}_g0'] for x in c])),
                float(wilcoxon([A[x][f'msa_{k}_g0'] for x in c], [Bb[x][f'msa_{k}_g0'] for x in c]).pvalue)) for k in ('sp', 'tc')}


def ptest(a, b): return ptest_rows(fam_rows(a), fam_rows(b))


def fam_rows_c20(stem, c20name):
    """per-family BAliBASE rows of a c20-style model (fixed models, MixFrag-S): the progressive decoding of its cached
    posteriors, or (MODELS_DECODING=anneal) its fsa_c20_balibase.py run."""
    if MODELS_DECODING == 'progressive':
        p = first(f'{EB}/msa_{stem}_progressive_c0_r0.json', f'{EB}/msa_{stem}_prog_c0_r0.json')
        rows = complete(rows_json(p), 'bali3pdbm') if p else None
    else:
        rows = None
        for suffix in ('corefull', 'bali3pdbm', ''):
            rows = complete(c20_rows(f'{C20}/{c20name}_{suffix}.jsonl' if suffix else f'{C20}/{c20name}.jsonl'), 'bali3pdbm')
            if rows: break
    return {r['family']: r for r in rows} if rows else None


def ptest_c20(a, b): return ptest_rows(fam_rows_c20(a, f'cache_runs/{a}'), fam_rows_c20(b, f'cache_runs/{b}'))


def ptest_bench(bench, a, b):
    suf = '_progressive_c0_r0' if MODELS_DECODING == 'progressive' else ''
    pa, pb = f'{BM}/{bench}_mixdom_{a}{suf}.json', f'{BM}/{bench}_mixdom_{b}{suf}.json'
    A, Bb = (complete(rows_json(p), bench) if os.path.exists(p) else None for p in (pa, pb))
    if not (A and Bb): return None
    return ptest_rows({r['family']: r for r in A}, {r['family']: r for r in Bb})


tests = {}
for bench in ('sabrem', 'oxm'):
    for a, b in (('d2f1_nobali_1620', 'd1f1_nobali_1620'), ('d2f1_nobali_1620', 'd3f1_nobali_1620'), ('d2f1_nobali_1620', 'd4f1_nobali_1620'), ('d2f1_nobali_1620', 'd5f1_nobali_1620'), ('d2f1_nobali_1620', 'd3f1_v4_fix')):
        t = ptest_bench(bench, a, b)
        if t: tests[f'{bench}: {a} - {b}'] = t; print(f"{bench:6s} {a:18s} - {b:18s}  dSP {t['sp'][0]:+.4f} p={t['sp'][1]:.3f}   dTC {t['tc'][0]:+.4f} p={t['tc'][1]:.3f}")
for a, b in (('d2f1_nobali_1620', 'd1f1_nobali_1620'), ('d3f1_nobali_1620', 'd1f1_nobali_1620'), ('d2f1_nobali_1620', 'd3f1_nobali_1620'),
             ('d2f1_tied_nobali_1620', 'd1f1_nobali_1620'), ('d2f1_tied_nobali_1620', 'd2f1_nobali_1620'),
             ('d2f1_tiedindel_nobali_1620', 'd1f1_nobali_1620'), ('d2f1_tiedindel_nobali_1620', 'd2f1_nobali_1620'),
             ('d1f2dg_nobali_1620', 'd1f1_nobali_1620'), ('d2f2dg_nobali_1620', 'd2f1_nobali_1620'), ('d5f1_nobali_1620', 'd2f1_nobali_1620')):
    t = ptest(a, b)
    if t: tests[f'{a} - {b}'] = t; print(f"{a:26s} - {b:18s}  dSP {t['sp'][0]:+.4f} p={t['sp'][1]:.3f}   dTC {t['tc'][0]:+.4f} p={t['tc'][1]:.3f}")
for a, b in ((mfs_stem(2), 'mixfrag_GTR_one'), (mfs_stem(2), 'tkf92_GTR_one'), (mfs_stem(3), 'mixfrag3_GTR_one'), (mfs_stem(3), 'tkf92_GTR_one')):
    t = ptest_c20(a, b)
    if t: tests[f'{a} - {b}'] = t; print(f"{a:26s} - {b:18s}  dSP {t['sp'][0]:+.4f} p={t['sp'][1]:.3f}   dTC {t['tc'][0]:+.4f} p={t['tc'][1]:.3f}")
json.dump(tests, open(f'{OUT}/paired_tests.json', 'w'), indent=1)
print('wrote', OUT)
