#!/usr/bin/env python3
"""TKF paper §3: Maraschino fit of plain TKF92 on Pfam sibling pairs.

Section sec:results-pfam-cherry. Fits a single-component TKF92 (no
mixdom hierarchy) to per-family cherry counts via Maraschino-around-EM
and reports stability across bootstrap subsets of the Pfam train split.

Pipeline:
  1. ``build_tkf92_cherry_counts.py`` (already run) produces per-family
     cherry-count tensors at ~/tkf-mixdom/python/pfam/cherries_tkf92/.
  2. This script loads bootstrap subsets of the Pfam train split and
     runs ``fit_mixture(K=1)`` on each.
  3. Reports recovered (lambda, mu, r) means and bootstrap rel_std.

Output: pfam/tkf92_K1_bootstrap_*.npz with per-seed parameter triples.

Usage:
    cd python && JAX_PLATFORMS=cpu uv run python \\
        experiments/run_tkf3_pfam_maraschino.py \\
        --n-seeds 5 --subsample-frac 0.8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.distill.tkf92_mixture import (  # noqa: E402
    fit_mixture, load_cherry_stack,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cherries-dir', default='pfam/cherries_tkf92',
                    help='Per-family TKF92 cherry-count directory')
    p.add_argument('--splits-file',
                    default=os.path.expanduser(
                        '~/bio-datasets/data/pfam-seed/splits/v1.json'))
    p.add_argument('--split', default='train',
                    choices=['train', 'val', 'test'])
    p.add_argument('--n-seeds', type=int, default=5)
    p.add_argument('--subsample-frac', type=float, default=0.8)
    p.add_argument('--outer-iter', type=int, default=20)
    p.add_argument('--inner-iter', type=int, default=15)
    p.add_argument('--out',
                    default='pfam/tkf92_K1_bootstrap_5seeds.npz')
    args = p.parse_args()

    with open(args.splits_file) as f:
        splits = json.load(f)
    fams = splits[args.split]
    print(f'Total {args.split} families: {len(fams)}', flush=True)

    results = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        n = int(len(fams) * args.subsample_frac)
        sample = list(rng.choice(fams, size=n, replace=False))
        t0 = time.time()
        stack = load_cherry_stack(args.cherries_dir, families=sample)
        thetas, log_mix, history = fit_mixture(
            stack, K=1, seed=seed,
            outer_n_iter_max=args.outer_iter,
            inner_n_iter_max=args.inner_iter,
            log_fn=lambda *a, **kw: None)
        elapsed = time.time() - t0
        lam = float(thetas[0][0])
        mu = float(thetas[0][1])
        r = float(thetas[0][2])
        n_fams_used = int(stack.match_counts.shape[0])
        print(f'seed={seed}: lam={lam:.5f}, mu={mu:.5f}, r={r:.4f} '
              f'({elapsed:.1f}s, n_fams={n_fams_used})', flush=True)
        results.append({
            'seed': seed, 'lam': lam, 'mu': mu, 'r': r,
            'n_fams': n_fams_used, 'elapsed_sec': elapsed,
        })

    lams = np.array([d['lam'] for d in results])
    mus = np.array([d['mu'] for d in results])
    rs = np.array([d['r'] for d in results])
    print()
    print(f'Bootstrap stats (n_seeds={args.n_seeds}, '
          f'subsample={args.subsample_frac*100:.0f}%):')
    print(f'  lam: mean={lams.mean():.5f} std={lams.std():.5f} '
          f'rel_std={lams.std() / lams.mean():.2%}')
    print(f'  mu : mean={mus.mean():.5f} std={mus.std():.5f} '
          f'rel_std={mus.std() / mus.mean():.2%}')
    print(f'  r  : mean={rs.mean():.4f} std={rs.std():.4f}')
    np.savez(args.out, lams=lams, mus=mus, rs=rs,
              results=np.array(results, dtype=object))
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
