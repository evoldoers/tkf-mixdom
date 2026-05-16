#!/usr/bin/env python3
"""TKF §4 production: pure-TKF92 SVI-BW on Pfam precompiled pairs.

Section sec:results-svibw.  Counterpart to the §3 alignment-given
Maraschino fit (pfam/tkf92_K1_train.npz) with the alignment-MARGINALISED
2D pair-FB E-step on the same Pfam pair set.

Loads pfam/precompiled/ shards (the same cherry pairs used by §3),
runs ``svi_bw_tkf92`` (pure TKF92 — no MixDom hierarchy), and saves
the converged (λ, μ, ext, Q, π) to disk for the §4 comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.core.ctmc import rate_matrix_jc69
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.train.tkf92_svi_bw import svi_bw_tkf92


def load_precompiled_pairs(precompiled_dir, max_pairs=None,
                              max_alignment_len=300, seed=0):
    """Load (x_int, y_int, t) tuples from precompiled pair shards.

    Reuses the existing PrecompiledPairSource decoding path.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from train_pfam import PrecompiledPairSource

    src = PrecompiledPairSource(precompiled_dir,
                                  max_alignment_len=max_alignment_len)
    print(f'PrecompiledPairSource: {src.n_pairs} pairs, '
          f'{src.n_families} families', flush=True)
    all_decoded = src.load_all_pairs()
    print(f'  decoded {len(all_decoded)} pairs', flush=True)

    rng = np.random.default_rng(seed)
    if max_pairs is not None and max_pairs < len(all_decoded):
        idx = rng.choice(len(all_decoded), max_pairs, replace=False)
        all_decoded = [all_decoded[i] for i in idx]
        print(f'  subsampled to {len(all_decoded)} pairs', flush=True)

    pairs = []
    for x_int, y_int, _states, _ac, _dc, t_est in all_decoded:
        pairs.append((np.asarray(x_int), np.asarray(y_int), float(t_est)))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--precompiled-dir', default='pfam/precompiled')
    p.add_argument('--max-pairs', type=int, default=5000,
                    help='Subsample to this many pairs from Pfam corpus')
    p.add_argument('--max-aln-len', type=int, default=300)
    p.add_argument('--init-lam', type=float, default=0.04)
    p.add_argument('--init-mu', type=float, default=0.05)
    p.add_argument('--init-ext', type=float, default=0.5)
    p.add_argument('--n-iter', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=200)
    p.add_argument('--svi-tau', type=float, default=1.0)
    p.add_argument('--svi-kappa', type=float, default=0.7)
    p.add_argument('--Q', choices=['lg', 'jc'], default='lg')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out',
                    default='pfam/tkf92_svi_bw_pure_train.npz')
    args = p.parse_args()

    if args.Q == 'lg':
        Q, pi = rate_matrix_lg()
    else:
        Q, pi = rate_matrix_jc69(20)
    pi_np = np.asarray(pi)
    print(f'Q model: {args.Q}', flush=True)

    print(f'Loading {args.max_pairs} pairs from {args.precompiled_dir}...',
          flush=True)
    t0 = time.time()
    pairs = load_precompiled_pairs(args.precompiled_dir,
                                       max_pairs=args.max_pairs,
                                       max_alignment_len=args.max_aln_len,
                                       seed=args.seed)
    print(f'  loaded {len(pairs)} pairs in {time.time()-t0:.1f}s', flush=True)

    print(f'\nLaunching SVI-BW: '
          f'init=({args.init_lam}, {args.init_mu}, {args.init_ext}), '
          f'n_iter={args.n_iter}, batch={args.batch_size}, '
          f'tau={args.svi_tau}, kappa={args.svi_kappa}', flush=True)
    out = svi_bw_tkf92(
        lambda: iter(pairs), n_total_pairs=len(pairs),
        init_lam=args.init_lam, init_mu=args.init_mu, init_ext=args.init_ext,
        Q=Q, pi=pi_np,
        n_iter=args.n_iter, batch_size=args.batch_size,
        svi_tau=args.svi_tau, svi_kappa=args.svi_kappa,
        seed=args.seed, log_fn=print)

    print(f'\nFinal: λ={out["lam"]:.5f} μ={out["mu"]:.5f} ext={out["ext"]:.4f}')
    np.savez(args.out, lam=out['lam'], mu=out['mu'], ext=out['ext'],
              Q=np.asarray(Q), pi=pi_np,
              history=np.array(out['history'], dtype=object),
              n_pairs=len(pairs),
              n_iter=args.n_iter, batch_size=args.batch_size)
    print(f'Saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
