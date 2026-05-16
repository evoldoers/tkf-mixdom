#!/usr/bin/env python3
"""TKF paper §4 runner: Maraschino (alignment-given) vs SVI-BW (marginalised).

Section sec:results-svibw.  Compare alignment-given Maraschino TKF92 fit
(from §3) against unaligned 2D SVI-BW on the same Pfam pair set.

This is a launcher / stub — the actual SVI-BW training is multi-hour,
so we leave the user to invoke ``train_pfam.py`` with the appropriate
arguments and then run the comparison post-hoc.

Steps:
  1. Run train_pfam.py with --n-dom 1 --n-frag 1 (plain TKF92 SVI-BW)
     on the same Pfam train families used in §3 (pfam/cherries_tkf92/).
     Recommended budget: --budget-hours 2 to 4.
  2. Once the run produces a final iterNNNN.npz checkpoint, load it
     here and compare to pfam/tkf92_K1_train.npz (the §3 result).
  3. Held-out log-likelihood: for both parameter sets, compute the FB
     log-likelihood on a Pfam val-split sibling pair set.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


SVIBW_LAUNCH_CMD = """
# Suggested SVI-BW launch command (plain TKF92 = n_dom 1, n_frag 1):
#
# cd python && JAX_PLATFORMS=cpu uv run python train_pfam.py \\
#     --msa-dir ~/bio-datasets/data/pfam-seed \\
#     --n-dom 1 --n-frag 1 --n-iter 30 \\
#     --budget-hours 4 \\
#     --checkpoint experiments/tkf92_svibw_run/iter
#
# Then this script can compare:
#   --maraschino-fit pfam/tkf92_K1_train.npz
#   --svibw-fit experiments/tkf92_svibw_run/iter0030.npz
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--maraschino-fit', default='pfam/tkf92_K1_train.npz',
                    help='Output of run_tkf3_pfam_maraschino.py')
    p.add_argument('--svibw-fit',
                    default='pfam/tkf92_svi_bw_pure_train_5k.npz',
                    help='Output of run_tkf4_svi_bw_pfam_pure_tkf92.py')
    args = p.parse_args()

    if not os.path.exists(args.maraschino_fit):
        print(f'ERROR: Maraschino fit {args.maraschino_fit} not found. '
              f'Run experiments/run_tkf3_pfam_maraschino.py first.')
        sys.exit(1)
    if not os.path.exists(args.svibw_fit):
        print(f'ERROR: SVI-BW fit {args.svibw_fit} not found. '
              f'Run experiments/run_tkf4_svi_bw_pfam_pure_tkf92.py first.')
        sys.exit(1)

    mar = np.load(args.maraschino_fit, allow_pickle=True)
    svi = np.load(args.svibw_fit, allow_pickle=True)
    # Maraschino: K=1 mixture stored in MixDom1 format (dom_ins/dom_del,
    # ext_rates as (1, 1, 1) tensor).
    lam_mar = float(np.atleast_1d(mar['dom_ins'])[0])
    mu_mar = float(np.atleast_1d(mar['dom_del'])[0])
    ext_mar = float(np.array(mar['ext_rates']).flatten()[0])
    # SVI-BW: pure-TKF92 npz (lam, mu, ext scalars).
    lam_svi = float(np.atleast_1d(svi['lam'])[0])
    mu_svi = float(np.atleast_1d(svi['mu'])[0])
    ext_svi = float(np.atleast_1d(svi['ext'])[0])

    print()
    print(f'{"":<32} {"Maraschino":<14} {"SVI-BW":<14} {"Δ (SVI − Mar)":<16}')
    print('-' * 80)
    print(f'{"λ (insertion rate)":<32} {lam_mar:<14.5f} {lam_svi:<14.5f} '
          f'{lam_svi-lam_mar:+.5f} ({(lam_svi-lam_mar)/lam_mar:+.1%})')
    print(f'{"μ (deletion rate)":<32} {mu_mar:<14.5f} {mu_svi:<14.5f} '
          f'{mu_svi-mu_mar:+.5f} ({(mu_svi-mu_mar)/mu_mar:+.1%})')
    print(f'{"ext (frag extension prob)":<32} {ext_mar:<14.4f} {ext_svi:<14.4f} '
          f'{ext_svi-ext_mar:+.4f} ({(ext_svi-ext_mar)/ext_mar:+.1%})')
    print()
    print('Interpretation (sec:results-svibw):')
    print('  Maraschino conditions on the given Pfam alignment, treating')
    print('  every aligned-shared-residue as a chain extension event.')
    print('  SVI-BW marginalises over alignments via the chi=tau92 Pair HMM')
    print('  forward-backward, so some "shared" residues are correctly')
    print('  attributed to alternative alignment paths rather than to a')
    print('  fragment extension.  The empirical signature is:')
    print('    Maraschino → higher ext, lower rates (λ, μ).')
    print('    SVI-BW     → lower ext, higher rates (λ, μ).')


if __name__ == '__main__':
    main()
