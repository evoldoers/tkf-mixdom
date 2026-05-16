#!/usr/bin/env python3
"""§2 follow-up: explore the ext-recovery bias as a function of t and the
substitution model's information content.

Hypothesis (§2 caption): the persistent ~20% bias in ext_hat at the
high-mu*t regime under JC20 emissions is caused by the JC20 model
having weak per-column discrimination (M-state self-emission probability
falls from ~0.74 at t=0.3 to ~0.38 at t=1.0), so the FB has limited
substitution leverage to lock down state assignments. A sharper
substitution model (LG08) should restore 1/sqrt(N) convergence on ext.

Sweep:
  - 3 values of t: 0.3, 1.0, 3.0
  - 2 substitution models: 20-state uniform Q (JC20), LG08 protein Q
  - Fixed (lam, mu, ext) = (0.08, 0.12, 0.5), N=1000, n_seeds=4

Saves results JSON + a small bar plot.
"""
from __future__ import annotations
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import sys
import json
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.core.ctmc import rate_matrix_jc69, transition_matrix
from tkfmixdom.jax.core.protein import rate_matrix_lg
from experiments.fig_tkf_fb_validation import tkf92_recover_at_N


def m_state_self_prob(Q, pi, t):
    """E[P(b=a | a, t)] = sum_a pi[a] * P(a|a, t) under the Q model."""
    P = np.asarray(transition_matrix(Q, t))
    pi = np.asarray(pi)
    return float(np.sum(pi * np.diag(P)))


def main():
    Q_jc, pi_jc = rate_matrix_jc69(20)
    Q_jc, pi_jc = np.asarray(Q_jc), np.asarray(pi_jc)
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg, pi_lg = np.asarray(Q_lg), np.asarray(pi_lg)

    lam, mu, ext = 0.08, 0.12, 0.5
    N = 1000
    n_seeds = 4
    seed = 2026

    t_values = [0.3, 1.0, 3.0]
    models = [
        ('JC20', Q_jc, pi_jc),
        ('LG08', Q_lg, pi_lg),
    ]

    print(f'Truth: lam={lam} mu={mu} ext={ext}; N={N}, n_seeds={n_seeds}')
    print(f'{"model":>5} | {"t":>4} | {"mu*t":>6} | {"P(b=a|a,t)":>11} | '
          f'{"mean(ext_hat)":>13} | {"std(ext_hat)":>12} | {"rel_bias":>8}')
    print('-' * 90)

    results = []
    for tag, Q, pi in models:
        for t in t_values:
            sub_mat_t = np.asarray(transition_matrix(jnp.asarray(Q), t))
            P_self = m_state_self_prob(jnp.asarray(Q), jnp.asarray(pi), t)
            ext_hats = []
            for ks in range(n_seeds):
                _, _, ext_h = tkf92_recover_at_N(
                    N, lam, mu, t, ext,
                    jnp.asarray(Q), pi, sub_mat_t,
                    ext_method='correct', seed=seed + ks * 1000)
                ext_hats.append(ext_h)
            arr = np.array(ext_hats)
            mean_h, std_h = arr.mean(), arr.std()
            rel_bias = (mean_h - ext) / ext  # signed relative bias
            print(f'{tag:>5} | {t:>4.1f} | {mu*t:>6.3f} | '
                  f'{P_self:>11.4f} | {mean_h:>13.4f} | {std_h:>12.4f} | '
                  f'{rel_bias:>+8.4f}')
            results.append({
                'model': tag, 't': t, 'mu_t': mu * t,
                'P_self': P_self,
                'ext_hats': ext_hats,
                'mean_ext_hat': float(mean_h),
                'std_ext_hat': float(std_h),
                'rel_bias': float(rel_bias),
            })

    out = {
        'simulation': {
            'lam': lam, 'mu': mu, 'ext_truth': ext,
            'N': N, 'n_seeds': n_seeds,
        },
        'results': results,
    }
    out_path = 'experiments/figures/ext_recovery_vs_subst_model.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')

    # Quick plot: x = mu*t, y = mean ext_hat ± std, colored by model.
    fig, ax = plt.subplots(figsize=(7, 5))
    by_model = {}
    for r in results:
        by_model.setdefault(r['model'], []).append(r)
    for tag, recs in by_model.items():
        xs = [r['mu_t'] for r in recs]
        ys = [r['mean_ext_hat'] for r in recs]
        es = [r['std_ext_hat'] / np.sqrt(n_seeds) for r in recs]
        ax.errorbar(xs, ys, yerr=es, fmt='o-', capsize=4, label=tag, ms=8,
                     alpha=0.85)
    ax.axhline(ext, ls='--', c='black', alpha=0.5, label=f'truth ext={ext}')
    ax.set_xlabel(r'$\mu \cdot t$ (per-pair expected-deletion fraction)')
    ax.set_ylabel(r'mean $\hat{\text{ext}}$ across $n_{\text{seeds}}$ replicates')
    ax.set_title(f'TKF92 ext recovery vs (t, Q model)\n'
                  f'(lam={lam}, mu={mu}, truth ext={ext}, N={N}, n_seeds={n_seeds})')
    ax.legend()
    ax.grid(alpha=0.3)
    out_pdf = 'experiments/figures/ext_recovery_vs_subst_model.pdf'
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f'Saved {out_pdf}')


if __name__ == '__main__':
    main()
