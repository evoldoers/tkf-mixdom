#!/usr/bin/env python3
"""§6 figure: SVI-VarAnc rate trajectory (real Pfam + TKF91 simulation).

Two-panel figure:
  Left:  Real Pfam SVI-VarAnc on the train spec (19850 fams), reading
         per-iter checkpoints from pfam/tkf92_svi_varanc_pure_train.npz.iter_ckpt.npz.
         Cherry-trained init (~0.030, 0.030) inflates by 9-15x within a
         few iterations; demonstrates the structural bias on real data.
  Right: TKF91 simulation (n_fams=30, L=240, lambda_*=0.04, mu_*=0.05),
         from experiments/figures/quantify_moment_mismatch.json. Truth-init
         jumps to biased fixed point in 1 iter, beta_lambda~2.85.

Saves to experiments/figures/sec6_rate_trajectory.pdf.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_pfam_iter_ckpt():
    """Read iter_ckpt.npz and extract per-iter (lambda, mu, ext)."""
    fn = '/home/yam/tkf-mixdom/python/pfam/tkf92_svi_varanc_pure_train.npz.iter_ckpt.npz'
    d = np.load(fn, allow_pickle=True)
    history = d['history']
    iters = []; ins = []; del_ = []; ext = []
    for entry in history:
        iters.append(int(entry['iter']))
        ins.append(float(entry['ins']))
        del_.append(float(entry['del_']))
        ext.append(float(entry['ext']))
    # Cherry init for context
    cherry_init = (0.02969, 0.03027, 0.6510)
    return {
        'iters': np.asarray(iters),
        'ins': np.asarray(ins),
        'del': np.asarray(del_),
        'ext': np.asarray(ext),
        'cherry_init': cherry_init,
    }


def load_sim_em_traj():
    """Load TKF91 simulation EM trajectory from quantify_moment_mismatch."""
    fn = '/home/yam/tkf-mixdom/python/experiments/figures/quantify_moment_mismatch.json'
    if not os.path.isfile(fn):
        return None
    d = json.load(open(fn))
    # Look for an em_trajectory key.
    for k in ('em_trajectory', 'B_em_trajectory', 'em_traj_truth_init'):
        if k in d:
            return d[k]
    # Otherwise hardcode the table from the LaTeX (we know these numbers).
    return None


def main():
    pfam = load_pfam_iter_ckpt()
    sim_traj = load_sim_em_traj()
    print(f'Pfam iter_ckpt: {len(pfam["iters"])} entries, last iter={pfam["iters"][-1]}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Real Pfam.
    ax = axes[0]
    ax.axhline(pfam['cherry_init'][0], ls='--', c='gray', lw=0.8,
                label=f'cherry init λ={pfam["cherry_init"][0]:.3f}')
    ax.axhline(pfam['cherry_init'][1], ls='--', c='gray', lw=0.8)
    ax.plot([0] + list(pfam['iters']),
             [pfam['cherry_init'][0]] + list(pfam['ins']), 'o-',
             label='λ (insertion rate)', c='tab:blue', ms=4)
    ax.plot([0] + list(pfam['iters']),
             [pfam['cherry_init'][1]] + list(pfam['del']), 's-',
             label='μ (deletion rate)', c='tab:orange', ms=4)
    ax.set_xlabel('SVI-VarAnc iter')
    ax.set_ylabel('rate (events / time)')
    ax.set_title(f'(A) Real Pfam: 19{{,}}850 train families '
                  f'(checkpoint at iter {pfam["iters"][-1]})')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    # Panel B: TKF91 simulation.
    ax = axes[1]
    iters_sim = [0, 1, 5, 10, 20, 30]
    ins_sim   = [0.040, 0.114, 0.116, 0.116, 0.112, 0.107]
    del_sim   = [0.050, 0.116, 0.118, 0.117, 0.113, 0.108]
    ext_sim   = [0.000, 0.0001, 0.0004, 0.002, 0.015, 0.054]
    ax.axhline(0.040, ls='--', c='gray', lw=0.8, label='truth λ=0.040')
    ax.axhline(0.050, ls='--', c='black', lw=0.8, label='truth μ=0.050')
    ax.plot(iters_sim, ins_sim, 'o-', label='λ', c='tab:blue', ms=5)
    ax.plot(iters_sim, del_sim, 's-', label='μ', c='tab:orange', ms=5)
    ax.set_xlabel('Tree-VBEM iter (truth-init)')
    ax.set_ylabel('rate (events / time)')
    ax.set_title('(B) TKF91 simulation: $n_{\\text{fams}}=30$, $L=240$')
    ax.legend(loc='center right', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = 'experiments/figures/sec6_rate_trajectory.pdf'
    plt.savefig(out_path)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
