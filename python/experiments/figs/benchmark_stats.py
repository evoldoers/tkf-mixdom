#!/usr/bin/env python3
"""Summary statistics of the three structural benchmarks as used in the paper (the MUSTANG-reference versions of
the MUSCLE benchmark collection: bali3pdbm, SABRE, OXBench), from the input sequences (in/) and the reference
alignments (ref/, core columns in upper case): families, sequences per family, sequence length, pairwise percent
identity over the aligned residue pairs of the reference, and the share of reference columns and of residues that
are in core blocks. Writes molbiolevol-paper/paper1-baumwelch/tables/benchmarks_table.tex."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tkfmixdom.util.msa_benchmark import parse_fasta

HOME = os.path.expanduser('~')
BENCH = [('BAliBASE (\\texttt{bali3pdbm})', f'{HOME}/bio-datasets/data/balibase/bali3pdbm'),
         ('SABmark (SABRE)', f'{HOME}/bio-datasets/data/bench1.0/sabrem'),
         ('OXBench', f'{HOME}/bio-datasets/data/bench1.0/oxm')]
OUT = '/home/yam/tkf-dp/molbiolevol-paper/paper1-baumwelch/tables/benchmarks_table.tex'


def stats(d):
    fams = sorted(f for f in os.listdir(f'{d}/ref') if not f.startswith('.'))
    nseq, lens, pid, pid_fam, core_cols, core_res, npairs = [], [], [], [], [], [], 0
    for fam in fams:
        ref = parse_fasta(f'{d}/ref/{fam}'); names = list(ref); rows = [ref[n] for n in names]
        nseq.append(len(names)); L = len(rows[0])
        for r in rows: lens.append(sum(c != '-' for c in r))
        # core columns: those in which every residue is upper case (the convention of the collection's scorer)
        col_core = [all(r[c] == '-' or r[c].isupper() for r in rows) and any(r[c] != '-' for r in rows) for c in range(L)]
        nonempty = [any(r[c] != '-' for r in rows) for c in range(L)]
        core_cols.append(sum(col_core) / max(1, sum(nonempty)))
        core_res.append(sum(1 for r in rows for c in range(L) if r[c] != '-' and r[c].isupper()) / max(1, sum(lens[-len(rows):])))
        fp = []
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                al = [(x, y) for x, y in zip(rows[a], rows[b]) if x != '-' and y != '-']
                if al: fp.append(sum(x.upper() == y.upper() for x, y in al) / len(al)); npairs += 1
        pid.extend(fp); pid_fam.append(float(np.mean(fp)) if fp else np.nan)
    q = lambda v, p: float(np.percentile(v, p))
    return dict(fams=len(fams), seqs=sum(nseq), pairs=npairs, nseq_med=np.median(nseq), nseq_min=min(nseq), nseq_max=max(nseq),
                len_med=np.median(lens), len_q1=q(lens, 25), len_q3=q(lens, 75),
                pid_med=100 * np.median(pid), pid_q1=100 * q(pid, 25), pid_q3=100 * q(pid, 75),
                core_cols=100 * np.mean(core_cols), core_res=100 * np.mean(core_res))


rows = [(lab, stats(d)) for lab, d in BENCH]
with open(OUT, 'w') as fh:
    fh.write('\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n')
    fh.write('Benchmark & Families & Pairs & Seqs./family & Length & Identity (\\%) & Core (\\%) \\\\\n')
    fh.write(' & & & median (range) & median (IQR) & median (IQR) & cols./res. \\\\\n\\midrule\n')
    for lab, s in rows:
        fh.write(f"{lab} & {s['fams']} & {s['pairs']:,} & {s['nseq_med']:.0f} ({s['nseq_min']}--{s['nseq_max']}) & "
                 f"{s['len_med']:.0f} ({s['len_q1']:.0f}--{s['len_q3']:.0f}) & {s['pid_med']:.0f} ({s['pid_q1']:.0f}--{s['pid_q3']:.0f}) & "
                 f"{s['core_cols']:.0f} / {s['core_res']:.0f} \\\\\n")
    fh.write('\\bottomrule\n\\end{tabular}\n')
for lab, s in rows: print(lab, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()})
print('wrote', OUT)
