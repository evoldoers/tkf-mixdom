#!/usr/bin/env python3
"""Paired Wilcoxon tests between two models' per-family BAliBASE / SABmark / OXBench scores under the paper's standard
decoding (progressive MEA, consistency 0, refinement 0), for the comparisons quoted in paper 1's text that
paper_tables.py does not compute. Reads experiments/expected_balibase/msa_<stem>_progressive_c0_r0.json (bali3pdbm)
and experiments/benchmarks/<bench>_<stem>_progressive_c0_r0.json (sabrem, oxm).
Usage: .venv/bin/python experiments/figs/adhoc_paired_tests.py [stemA-stemB ...]   (default: the paper's list)"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
EB, BM = 'experiments/expected_balibase', 'experiments/benchmarks'
DEFAULT = ['tkf92_K20-tkf92_GTR_one', 'tkf92_CML20pc_anchor-tkf92_GTR_one', 'tkf92_K20tied-tkf92_GTR_one', 'tkf92_K20tiedgtr-tkf92_GTR_one',
           'tkf92_K20tiedindel-tkf92_GTR_one', 'mixfrag_GTR_one-tkf92_GTR_one', 'mixfrag3_GTR_one-tkf92_GTR_one', 'mixfrag3_GTR_one-mixfrag_GTR_one',
           'mixdom_d1f1_nobali_1620-tkf92_GTR_one', 'mixdom_d2f1_nobali_1620-tkf92_GTR_one', 'mixdom_d2f1_nobali_1620-mixfrag_GTR_one',
           'mixfrags_F2_nobali_1620-tkf92_GTR_one', 'mixfrags_F3_nobali_1620-tkf92_GTR_one', 'mixfrags_F2_nobali_1620-mixfrag_GTR_one',
           'mixfrags_F3_nobali_1620-mixfrag3_GTR_one', 'mixdom_d2f1_nobali_1620-mixfrags_F2_nobali_1620']
def rows(stem, b):
    p = f'{EB}/msa_{stem}_progressive_c0_r0.json' if b == 'bali3pdbm' else f'{BM}/{b}_{stem}_progressive_c0_r0.json'
    if not os.path.exists(p): return None
    d = json.load(open(p)); out = {}
    for r in d['rows']:
        g = r.get('g0', r); fam = r.get('family', r.get('fam'))
        out[fam] = (float(g.get('sp', r.get('msa_sp_g0'))), float(g.get('tc', r.get('msa_tc_g0'))))
    return out
def main():
    pairs = sys.argv[1:] or DEFAULT
    print(f"{'A - B':54s} {'bench':9s} {'n':>4s} {'dSP':>8s} {'p':>8s} {'dTC':>8s} {'p':>8s}")
    for pr in pairs:
        a, b = pr.split('-', 1)
        for bench in ('bali3pdbm', 'sabrem', 'oxm'):
            ra, rb = rows(a, bench), rows(b, bench)
            if not ra or not rb: print(f'{pr:54s} {bench:9s}   missing'); continue
            fams = sorted(set(ra) & set(rb))
            dsp = np.array([ra[f][0] - rb[f][0] for f in fams]); dtc = np.array([ra[f][1] - rb[f][1] for f in fams])
            psp = wilcoxon(dsp).pvalue if np.any(dsp) else 1.0; ptc = wilcoxon(dtc).pvalue if np.any(dtc) else 1.0
            print(f'{pr:54s} {bench:9s} {len(fams):4d} {dsp.mean()*100:+8.2f} {psp:8.3g} {dtc.mean()*100:+8.2f} {ptc:8.3g}')
if __name__ == '__main__': main()
