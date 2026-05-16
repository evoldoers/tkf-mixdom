#!/usr/bin/env python3
"""TKF paper §7: Cherry-trained vs MSA-refined TKF92 parameter comparison.

Section sec:results-cherry-vs-msa.  Compares:
  * Cherry-trained: pfam/tkf92_K1_train.npz (from §3, run_tkf3_pfam_maraschino).
  * MSA-refined:    experiments/tkf_paper_runs/tkf6_svi_varanc_d1f1.json
                    + final iter checkpoint (from §6, tree_svi_vbem_pfam.py).

Reports side-by-side parameter table and held-out test-set metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cherry-fit', default='pfam/tkf92_K1_train.npz')
    p.add_argument('--svi-varanc-out',
                    default='pfam/tkf92_svi_varanc_pure_train.npz',
                    help='Output of run_tkf6_svi_varanc_pure_tkf92.py')
    args = p.parse_args()

    if not os.path.exists(args.cherry_fit):
        print(f'ERROR: cherry-trained fit {args.cherry_fit} not found.')
        sys.exit(1)
    cherry = np.load(args.cherry_fit, allow_pickle=True)

    if not os.path.exists(args.svi_varanc_out):
        print(f'svi-VarAnc result not found at {args.svi_varanc_out}.')
        return
    msa = np.load(args.svi_varanc_out, allow_pickle=True)

    cherry_lam = float(np.atleast_1d(cherry['dom_ins'])[0])
    cherry_mu = float(np.atleast_1d(cherry['dom_del'])[0])
    cherry_r = float(np.atleast_1d(cherry['ext_rates']).flatten()[0])

    # Pure-TKF92 svi-VarAnc result has 'ins_rate', 'del_rate', 'ext'.
    msa_lam = float(np.atleast_1d(msa['ins_rate'])[0])
    msa_mu = float(np.atleast_1d(msa['del_rate'])[0])
    msa_r = float(np.atleast_1d(msa['ext'])[0])

    print(f'\n{"":<25} {"Cherry":<14} {"MSA-refined":<14} {"Diff (rel)":<12}')
    print('-' * 65)
    print(f'{"λ (insertion rate)":<25} {cherry_lam:<14.5f} '
          f'{msa_lam:<14.5f} {(msa_lam - cherry_lam)/max(cherry_lam, 1e-9):+.2%}')
    print(f'{"μ (deletion rate)":<25} {cherry_mu:<14.5f} '
          f'{msa_mu:<14.5f} {(msa_mu - cherry_mu)/max(cherry_mu, 1e-9):+.2%}')
    print(f'{"r (frag extension)":<25} {cherry_r:<14.4f} '
          f'{msa_r:<14.4f} {(msa_r - cherry_r)/max(cherry_r, 1e-9):+.2%}')

    print()
    print('Interpretation (sec:results-cherry-vs-msa):')
    print('  Cherry-trained: §3 Maraschino, conditioned on Pfam alignment.')
    print('  MSA-refined: §6 svi-VarAnc on Pfam test families (200), warm-')
    print('    started from Cherry, MARGINALISES over ancestral indel state.')
    print('  Both fit the SAME indel rates in principle; differences')
    print('    reflect the alignment-given vs marginalised path bias.')


if __name__ == '__main__':
    main()
