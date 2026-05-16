#!/usr/bin/env python3
"""§9 figure: per-component (S, π) heatmaps for the C=10 mixture-of-sites
fit on Pfam-cherries.

Two panels per component (so 10 components -> 10 (S, π) panels), arranged
in a grid:

  Component k: pi panel (1x20 vector, color = log frequency)
              S panel  (20x20 exchangeability matrix, color = log rate)

Plus a top row showing class weights w_k.

Saves to experiments/figures/sec9_components_C10.pdf.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


AA = 'ARNDCQEGHILKMFPSTWYV'
assert len(AA) == 20


def main(C=10, out_path=None):
    fn = f'/home/yam/tkf-mixdom/python/pfam/cherryml_mixture_C{C}_n5000.npz'
    if out_path is None:
        out_path = f'experiments/figures/sec9_components_C{C}.pdf'
    d = np.load(fn, allow_pickle=True)
    S = np.asarray(d['S'])     # (C, 20+1, 20+1) — last row/col often gap
    pi = np.asarray(d['pi'])   # (C, 20+1)
    weights = np.asarray(d['weights'])  # (C,)

    # Drop the gap row/col (state 20).
    if S.shape[-1] == 21:
        S = S[:, :20, :20]
        pi = pi[:, :20]
    print(f'C={C}: S shape {S.shape}, pi shape {pi.shape}, weights={np.round(weights, 3)}')

    # Sort components by weight (descending) for readability.
    order = np.argsort(-weights)
    S = S[order]; pi = pi[order]; weights = weights[order]

    fig = plt.figure(figsize=(2.0 * C + 1, 4.5))
    gs = GridSpec(2, C, height_ratios=[0.8, 5.0], wspace=0.35, hspace=0.4)

    for c in range(C):
        # Top: pi as a vertical bar.
        ax_pi = fig.add_subplot(gs[0, c])
        ax_pi.imshow(pi[c:c+1], aspect='auto', cmap='viridis', vmin=0, vmax=0.15)
        ax_pi.set_xticks(range(20)); ax_pi.set_xticklabels(list(AA), fontsize=5)
        ax_pi.set_yticks([]);
        ax_pi.set_title(f'k={c+1}, w={weights[c]:.2f}', fontsize=8)

        # Bottom: S heatmap (log scale to compress).
        ax_S = fig.add_subplot(gs[1, c])
        log_S = np.log10(np.clip(S[c], 1e-3, None))
        im = ax_S.imshow(log_S, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
        ax_S.set_xticks(range(20)); ax_S.set_xticklabels(list(AA), fontsize=4)
        ax_S.set_yticks(range(20)); ax_S.set_yticklabels(list(AA), fontsize=4)
        if c == 0:
            ax_S.set_ylabel('from')
        ax_S.set_xlabel('to', fontsize=7)

    fig.suptitle(
        f'§9: Mixture-of-sites $C={C}$ components on $5{{,}}000$ Pfam '
        f'cherries (sorted by weight)', y=1.02)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.45])
    fig.colorbar(im, cax=cbar_ax, label='$\\log_{10}\\,S^{(c)}_{ij}$')
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main(C=10)
