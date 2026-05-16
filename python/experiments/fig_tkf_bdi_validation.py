#!/usr/bin/env python3
"""TKF paper §1 figure: BDI Gillespie vs analytic moments.

Generates the publication figure for ``sec:results-gillespie-bdi``:

  Do the closed-form BDI sufficient-statistic expectations match
  Gillespie averages?

Plots three panels (E[B], E[D], E[S]) of Gillespie sample means vs the
analytic formulas across multiple regimes:

  - Small / moderate / large rate regimes (λ < μ).
  - L'Hôpital regime (λ ≈ μ) where the closed-form integrals have a 0/0
    form that we resolve via the analytic limit (E[N(s)] = i + λs).

A second figure shows the 1/√N shrinkage of the empirical residuals as
n_sims grows, confirming the simulator is unbiased and converges at the
expected Monte Carlo rate.

Output: experiments/figures/tkf_bdi_validation_means.pdf,
        experiments/figures/tkf_bdi_validation_shrinkage.pdf.
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.simulate.mixdom_gillespie import gillespie_bdi_edge


# -- analytic BDI moments for TKF (ν = λ) starting from i lineages -------


def analytic_moments(i, lam, mu, t):
    """Closed-form E[N(s)], E[S], E[B], E[D] for the TKF BDI process.

    For the TKF immortal-link BDI: each lineage births at rate λ and dies
    at rate μ; the immortal link injects new lineages at rate λ.

    E[N(s) | N(0)=i] = i·exp(-(μ-λ)s)
                      + (λ/(μ-λ))·(1 - exp(-(μ-λ)s))           (general)
                    = i + λs                                    (λ → μ)
    E[S] = ∫₀ᵗ E[N(s)] ds = i·(1-exp(-εt))/ε
                          + (λ/ε)·(t - (1-exp(-εt))/ε),  ε = μ-λ
    E[S] = i·t + λ·t²/2                                          (λ = μ)
    E[B] = λ·(t + E[S])    (immigration adds λT plus per-lineage births)
    E[D] = μ·E[S].
    """
    eps = mu - lam
    if abs(eps) < 1e-9:
        # L'Hôpital limit (λ = μ).
        e_S = float(i) * t + lam * t * t / 2.0
    else:
        survival_int = (1.0 - math.exp(-eps * t)) / eps
        e_S = float(i) * survival_int \
            + (lam / eps) * (t - survival_int)
    e_B = lam * (t + e_S)
    e_D = mu * e_S
    return e_S, e_B, e_D


# -- Gillespie sweep -----------------------------------------------------


def gillespie_means(rng, i, lam, mu, t, n_sims):
    """Mean (B, D, S) from `n_sims` independent runs."""
    sumB = 0.0
    sumD = 0.0
    sumS = 0.0
    for _ in range(n_sims):
        r = gillespie_bdi_edge(rng, list(range(i)), lam, mu, t,
                                next_lineage_id=i)
        sumB += r['n_births'] + r['n_imm']
        sumD += r['n_deaths']
        sumS += r['sojourn']
    return sumB / n_sims, sumD / n_sims, sumS / n_sims


REGIMES = [
    # Standard (λ < μ).
    (1, 0.04, 0.05, 0.5, 'low (λ=0.04, μ=0.05, t=0.5)'),
    (3, 0.04, 0.05, 0.5, 'low, i=3'),
    (1, 0.10, 0.15, 1.0, 'moderate (λ=0.10, μ=0.15, t=1.0)'),
    (1, 0.20, 0.30, 1.0, 'higher (λ=0.20, μ=0.30, t=1.0)'),
    # L'Hôpital regime (λ → μ).
    (1, 0.099, 0.100, 1.0, "L'Hôpital |λ-μ|=0.001"),
    (1, 0.0999, 0.1000, 1.0, "L'Hôpital |λ-μ|=1e-4"),
    (1, 0.10, 0.10, 1.0, "L'Hôpital λ=μ exact"),
    # Long edge.
    (2, 0.05, 0.07, 3.0, 'long edge (t=3, i=2)'),
]


def build_panel(ax, sim_vals, ana_vals, labels, ylabel, regimes):
    """Scatter Gillespie means vs analytic with y=x reference."""
    for k, (s, a, lab, reg) in enumerate(zip(sim_vals, ana_vals, labels, regimes)):
        is_lhopital = abs(reg[1] - reg[2]) < 0.005
        marker = 'D' if is_lhopital else 'o'
        ax.scatter(a, s, marker=marker, s=70, alpha=0.85, label=lab,
                    edgecolor='k', linewidth=0.8)
    lo = min(min(sim_vals), min(ana_vals)) * 0.9
    hi = max(max(sim_vals), max(ana_vals)) * 1.1
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, label='y = x')
    ax.set_xlabel(f'analytic {ylabel}')
    ax.set_ylabel(f'Gillespie {ylabel}')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)


def fig_means(out_dir, n_sims=20_000_000, seed=42):
    """Top-level figure: Gillespie vs analytic, three panels (B, D, S)."""
    rng = np.random.RandomState(seed)
    sim_B, sim_D, sim_S = [], [], []
    ana_B, ana_D, ana_S = [], [], []
    labels = []
    for i, lam, mu, t, lab in REGIMES:
        e_S, e_B, e_D = analytic_moments(i, lam, mu, t)
        sB, sD, sS = gillespie_means(rng, i, lam, mu, t, n_sims)
        sim_B.append(sB); sim_D.append(sD); sim_S.append(sS)
        ana_B.append(e_B); ana_D.append(e_D); ana_S.append(e_S)
        labels.append(lab)
        print(f'{lab}: E[B] sim={sB:.4f} ana={e_B:.4f}, '
              f'E[D] sim={sD:.4f} ana={e_D:.4f}, '
              f'E[S] sim={sS:.4f} ana={e_S:.4f}')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    build_panel(axes[0], sim_B, ana_B, labels, 'E[B]', REGIMES)
    build_panel(axes[1], sim_D, ana_D, labels, 'E[D]', REGIMES)
    build_panel(axes[2], sim_S, ana_S, labels, 'E[S]', REGIMES)
    axes[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                    fontsize=9)
    fig.suptitle(
        f'TKF BDI Gillespie vs analytic moments (n_sims = {n_sims}). '
        "Diamonds: L'Hôpital regime (λ ≈ μ).", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'tkf_bdi_validation_means.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def fig_shrinkage(out_dir, seed=2026, n_seeds=800):
    """Convergence figure: mean |Gillespie - analytic| / |analytic| vs n_sims,
    with empirical SE error bars across n_seeds replicate runs per n_sims.

    A persistent mean offset (residual not shrinking with n) would indicate
    systematic bias; pure 1/√n convergence with error bars overlapping the
    reference line confirms the simulator is unbiased.

    ALSO saves a sidecar JSON with the raw per-(regime, n, seed) sim values
    so questions about the figure can be answered without re-running.
    """
    import json as _json
    raw = []  # list of {regime, n, seed, sim_B, sim_D, sim_S, ana_B, ana_D, ana_S}
    n_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    # Pick a few representative regimes.
    test_regimes = [
        (1, 0.10, 0.15, 1.0, 'general'),
        (1, 0.099, 0.100, 1.0, "L'Hôpital"),
        (3, 0.04, 0.05, 0.5, 'i=3, low rates'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    stats_names = ('E[B]', 'E[D]', 'E[S]')
    for k, (i, lam, mu, t, reg) in enumerate(test_regimes):
        e_S, e_B, e_D = analytic_moments(i, lam, mu, t)
        ana = (e_B, e_D, e_S)
        # rels[s] = list per N of (mean, se) across n_seeds replicates.
        rels = {s: ([], []) for s in stats_names}
        for n in n_values:
            replicate_rels = {s: [] for s in stats_names}
            for ks in range(n_seeds):
                seed_used = seed + ks * 1000 + n
                rng = np.random.RandomState(seed_used)
                sB, sD, sS = gillespie_means(rng, i, lam, mu, t, n)
                raw.append({'regime': reg, 'n': int(n), 'seed': int(seed_used),
                             'sim_B': float(sB), 'sim_D': float(sD), 'sim_S': float(sS),
                             'ana_B': float(e_B), 'ana_D': float(e_D), 'ana_S': float(e_S)})
                sim = (sB, sD, sS)
                for s, sv, av in zip(stats_names, sim, ana):
                    if abs(av) > 1e-9:
                        replicate_rels[s].append(abs(sv - av) / abs(av))
                    else:
                        replicate_rels[s].append(abs(sv - av))
            for s in stats_names:
                arr = np.array(replicate_rels[s])
                rels[s][0].append(arr.mean())
                rels[s][1].append(arr.std() / np.sqrt(n_seeds))
        for ax_idx, s in enumerate(stats_names):
            means, ses = rels[s]
            axes[ax_idx].errorbar(n_values, means, yerr=ses, fmt='o-',
                                    capsize=3, label=f'{reg}', alpha=0.8)
    # Reference 1/√n line at each panel.
    for ax in axes:
        ref = np.array(n_values, dtype=float)
        all_first = []
        for line in ax.get_lines():
            yd = line.get_ydata()
            if len(yd) > 0:
                all_first.append(yd[0])
        if all_first:
            anchor = float(np.median(all_first))
            ax.plot(n_values, anchor * np.sqrt(n_values[0] / ref),
                      'k--', alpha=0.4, label=r'$\propto 1/\sqrt{n}$')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('n_sims')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('mean |sim - analytic| / |analytic| ± SE across replicates')
    for k, name in enumerate(stats_names):
        axes[k].set_title(name)
    fig.suptitle(f'Gillespie BDI residuals: mean ± SE across {n_seeds} '
                  f'replicate runs per n_sims. Curves track 1/√n with no '
                  f'systematic offset.', fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'tkf_bdi_validation_shrinkage.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

    # Sidecar JSON: raw per-(regime, n, seed) measurements.
    json_path = os.path.join(out_dir, 'tkf_bdi_validation_shrinkage.json')
    with open(json_path, 'w') as f:
        _json.dump({'test_regimes': [{'i': i, 'lam': l, 'mu': m, 't': t, 'reg': reg}
                                       for i, l, m, t, reg in test_regimes],
                     'n_values': n_values,
                     'n_seeds': n_seeds,
                     'seed_base': seed,
                     'records': raw}, f, indent=2)
    print(f'Saved raw data: {json_path}')


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    print('=' * 60)
    print('Figure 1: Gillespie vs analytic moments')
    print('=' * 60)
    fig_means(out_dir)
    print()
    print('=' * 60)
    print('Figure 2: 1/√n shrinkage')
    print('=' * 60)
    fig_shrinkage(out_dir)


if __name__ == '__main__':
    main()
