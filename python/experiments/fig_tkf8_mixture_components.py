#!/usr/bin/env python3
"""TKF paper §8 figure: mixture-of-TKF92 components.

Loads pfam/tkf92_K{2,3,4,6,8}_train.npz (produced by fit_tkf92_mixture.py
launched in run_tkf8_em_around_maraschino_kspread.sh) and shows how the
recovered components partition Pfam by indel-rate / fragment-length /
equilibrium composition.

Output: experiments/figures/tkf8_mixture_components.pdf
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


KS = [1, 2, 3, 4, 6, 8]


def load_K(K):
    """Returns dict with lams (K,), mus (K,), rs (K,), weights (K,) or None."""
    path = f'pfam/tkf92_K{K}_train.npz'
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    # Format: dom_ins (K,), dom_del (K,), ext_rates (K, F=1, F=1), dom_weights (K,)
    return {
        'lams': np.array(d['dom_ins']),
        'mus': np.array(d['dom_del']),
        'rs': np.array(d['ext_rates']).reshape(-1),
        'weights': np.array(d['dom_weights']),
        'K': K,
    }


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)

    fits = {K: load_K(K) for K in KS if load_K(K) is not None}
    if not fits:
        print('No K_train fit files found in pfam/. Run fit_tkf92_mixture.py first.')
        sys.exit(1)

    print(f'Loaded fits for K = {sorted(fits)}')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for K, f in fits.items():
        # Panel 1: λ vs μ scatter, dot size = weight.
        sizes = 200 * np.array(f['weights'])
        axes[0].scatter(f['lams'], f['mus'], s=sizes, alpha=0.6,
                          edgecolor='k', linewidth=0.8, label=f'K={K}')
        # Panel 2: r (extension) vs λ.
        axes[1].scatter(f['lams'], f['rs'], s=sizes, alpha=0.6,
                          edgecolor='k', linewidth=0.8, label=f'K={K}')
        # Panel 3: weights as a horizontal bar.
        for ki, w in enumerate(f['weights']):
            axes[2].barh(K + (ki - 0.5) * 0.1, w, 0.08, alpha=0.7,
                          edgecolor='k', linewidth=0.5)

    axes[0].set_xlabel('λ (insertion rate)')
    axes[0].set_ylabel('μ (deletion rate)')
    axes[0].set_title('Components in (λ, μ) space (size = weight)')
    axes[0].plot([0.005, 0.05], [0.005, 0.05], 'k--', alpha=0.3,
                  label='λ = μ')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('λ (insertion rate)')
    axes[1].set_ylabel('r (fragment extension prob.)')
    axes[1].set_title('Components in (λ, r) space')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel('component weight')
    axes[2].set_ylabel('K')
    axes[2].set_title('Per-K mixture weights')
    axes[2].set_yticks(list(fits.keys()))
    axes[2].grid(alpha=0.3, axis='x')

    fig.suptitle(
        'TKF §8: mixture-of-TKF92 components fit by EM-around-Maraschino '
        'on Pfam train.', fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'tkf8_mixture_components.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

    # Also print a tabular summary.
    print()
    print('Component summary:')
    print(f'{"K":<3} {"k":<3} {"weight":<8} {"λ":<10} {"μ":<10} {"r":<8} {"κ=λ/μ":<8}')
    for K, f in fits.items():
        for k in range(K):
            w = f['weights'][k]
            lam = f['lams'][k]
            mu = f['mus'][k]
            r = f['rs'][k]
            kappa = lam / mu if mu > 0 else float('inf')
            print(f'{K:<3} {k:<3} {w:.4f}   {lam:.5f}    {mu:.5f}    '
                  f'{r:.4f}    {kappa:.4f}')


if __name__ == '__main__':
    main()
