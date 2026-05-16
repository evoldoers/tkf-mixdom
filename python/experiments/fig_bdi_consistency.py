#!/usr/bin/env python3
"""Task 1: BDI Consistency Scatterplots (Paper Figures).

Generates publication-quality scatterplots showing statistical consistency
of BDI sufficient statistics recovery:
  - X-axis: analytic E[statistic] from score function (recover_bdi_stats)
  - Y-axis: mean of true counts from Gillespie simulation, with std dev error bars
  - Points aggregated by X(T) value (descendant length), labeled with X(T)
  - Multiple colors for different (λ, μ, T) parameter regimes

All simulations start from X(0) = 1 (single ancestor residue).

Output: experiments/figures/bdi_consistency_{B,D,S}.pdf

Usage:
    cd python && python experiments/fig_bdi_consistency.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from tkfmixdom.jax.simulate.simulate import simulate_bdi_gillespie


# Parameter regimes: (λ, μ, T, label, category)
REGIMES = [
    # κ < 1 (λ < μ)
    (0.01, 0.03, 1.0, r"$\lambda{=}0.01,\mu{=}0.03,T{=}1$", "kappa_lt_1"),
    (0.05, 0.10, 2.0, r"$\lambda{=}0.05,\mu{=}0.10,T{=}2$", "kappa_lt_1"),
    # κ = 1 (λ = μ)
    (0.02, 0.02, 1.0, r"$\lambda{=}\mu{=}0.02,T{=}1$", "kappa_eq_1"),
    (0.05, 0.05, 2.0, r"$\lambda{=}\mu{=}0.05,T{=}2$", "kappa_eq_1"),
    # κ > 1 (λ > μ)
    (0.03, 0.01, 1.0, r"$\lambda{=}0.03,\mu{=}0.01,T{=}1$", "kappa_gt_1"),
]

N_SIMS = 10_000_000  # Gillespie replicates per regime (was 10_000; 1000x scale-up for tight error bars)
MIN_COUNT = 10  # minimum simulations per X(T) bin for inclusion
MAX_J = 15  # max X(T) to include


# --- Numpy BDI log-probability (works at any parameters) ---

def bdi_logprob(i, j, ins_rate, del_rate, t):
    """Compute log P(j|i) for the BDI process via forward DP.

    Uses numpy. Handles the λ=μ singularity via L'Hôpital limits for β and γ.
    """
    alpha = np.exp(-del_rate * t)
    kappa = ins_rate / del_rate if del_rate > 0 else 0.0

    if abs(1.0 - kappa) < 1e-6:
        # L'Hôpital limits: β → s/(1+s), γ → 1 - 1/((1+s)φ)
        s = del_rate * t
        beta = s / (1.0 + s) if s > 0 else 0.0
        phi = (1.0 - alpha) / s if s > 1e-10 else 1.0 - s / 2.0
        gamma = 1.0 - 1.0 / ((1.0 + s) * phi) if s > 1e-10 else 0.0
    else:
        eta = np.exp(-ins_rate * t)
        denom = del_rate * eta - ins_rate * alpha
        if abs(denom) < 1e-30:
            beta = 0.0
        else:
            beta = ins_rate * (eta - alpha) / denom
        if abs(ins_rate * (1 - alpha)) < 1e-30:
            gamma = 0.0
        else:
            gamma = 1.0 - del_rate * beta / (ins_rate * (1 - alpha))

    j_max = j + 1
    F = np.zeros(j_max)
    for d in range(j_max):
        F[d] = (1 - beta) * beta**d

    for a in range(i):
        F_new = np.zeros(j_max)
        for d in range(j_max):
            F_new[d] += (1 - alpha) * (1 - gamma) * F[d]
            if d > 0:
                survive_or_orphan = alpha + (1 - alpha) * gamma
                for dp in range(d):
                    k = d - dp - 1
                    F_new[d] += F[dp] * survive_or_orphan * (1 - beta) * beta**k
        F = F_new

    if j >= j_max or F[j] <= 0:
        return -np.inf
    return np.log(F[j])


def _score_derivs(i, j, ins_rate, del_rate, t, eps=1e-6):
    """Compute d(log P)/d(lambda) and d(log P)/d(mu) via central finite differences."""
    d_lam = (bdi_logprob(i, j, ins_rate + eps, del_rate, t)
             - bdi_logprob(i, j, ins_rate - eps, del_rate, t)) / (2 * eps)
    d_mu = (bdi_logprob(i, j, ins_rate, del_rate + eps, t)
            - bdi_logprob(i, j, ins_rate, del_rate - eps, t)) / (2 * eps)
    return d_lam, d_mu


def _score_2nd_derivs(i, j, lam, mu, t, h=1e-4):
    """Second-order partial derivatives of log P(j|i; lam, mu, t) via
    central finite differences. Returns (d_λλ, d_λμ, d_μμ).

    Step h=1e-4 is the optimum for 4th-order central differences in
    float64 (round-off ~ε/h², truncation ~h²·f''''/12 balance at h ~ ε^{1/4}).
    """
    lp = bdi_logprob(i, j, lam, mu, t)
    # ∂²/∂λ² via 3-point central FD
    d_ll = (bdi_logprob(i, j, lam + h, mu, t)
            - 2 * lp
            + bdi_logprob(i, j, lam - h, mu, t)) / (h * h)
    # ∂²/∂μ²
    d_mm = (bdi_logprob(i, j, lam, mu + h, t)
            - 2 * lp
            + bdi_logprob(i, j, lam, mu - h, t)) / (h * h)
    # ∂²/(∂λ ∂μ) via 4-point cross FD
    d_lm = (bdi_logprob(i, j, lam + h, mu + h, t)
            - bdi_logprob(i, j, lam + h, mu - h, t)
            - bdi_logprob(i, j, lam - h, mu + h, t)
            + bdi_logprob(i, j, lam - h, mu - h, t)) / (4 * h * h)
    return d_ll, d_lm, d_mm


def analytic_stats(lam, mu, t, j, i=1):
    """Compute analytic E[B], E[D], E[S] for the BDI process.

    Uses the eq:ES-formula (lhopital-limits.tex eq:ES-formula / body-tkf91.tex
    eq:exposure-tkf) with finite-difference score derivatives.

    At λ=μ=ρ the (j-i + μ d_μ - λ d_λ - λt)/(λ-μ) form is 0/0; the
    L'Hôpital resolution (eq:ES-limit in lhopital-limits.tex) is
        E[S]|_{λ=μ=ρ} = dN/dλ|_{λ=ρ}
                      = ρ·∂²_{λμ} log p - ∂_λ log p - ρ·∂²_{λλ} log p - t
    where N(λ) = j-i + μ·∂_μ log p - λ·∂_λ log p - λt is the formula's
    numerator. Implemented here directly via central finite differences on
    `bdi_logprob` -- no ε-clamp / off-critical shift required.
    """
    lp = bdi_logprob(i, j, lam, mu, t)
    if not np.isfinite(lp):
        return None

    kappa = lam / mu if mu > 0 else np.inf
    if abs(1.0 - kappa) < 1e-3:
        # Direct L'Hôpital second-derivative form (eq:ES-limit). At λ=μ=ρ:
        #   E[S] = ρ·∂²_λμ log p - ∂_λ log p - ρ·∂²_λλ log p - t
        # No off-critical clamp; computed at the true λ=μ via second-order
        # central FDs on bdi_logprob.
        rho = mu
        d_lam, d_mu = _score_derivs(i, j, rho, rho, t)
        d_ll, d_lm, _ = _score_2nd_derivs(i, j, rho, rho, t)
        E_S = rho * d_lm - d_lam - rho * d_ll - t
        E_B = rho * (d_lam + E_S + t)
        E_D = rho * (d_mu + E_S)
        return E_B, E_D, E_S

    # General case
    d_lam, d_mu = _score_derivs(i, j, lam, mu, t)
    E_S = (j - i + mu * d_mu - lam * d_lam - lam * t) / (lam - mu)
    E_B = lam * (d_lam + E_S + t)
    E_D = mu * (d_mu + E_S)
    return E_B, E_D, E_S


def run_gillespie_regime(lam, mu, t, n_sims, seed=42):
    """Run Gillespie simulations from X(0)=1, grouped by final state X(T)."""
    np_rng = np.random.RandomState(seed)
    results = defaultdict(lambda: {'births': [], 'deaths': [], 'sojourn': []})

    for _ in range(n_sims):
        j, nb, nd, ni, soj = simulate_bdi_gillespie(np_rng, 1, lam, mu, t)
        total_births = nb + ni
        results[j]['births'].append(total_births)
        results[j]['deaths'].append(nd)
        results[j]['sojourn'].append(soj)

    return results


def make_figures():
    """Generate 3 publication-quality scatterplots: bdi_consistency_{B,D,S}.pdf."""
    stat_keys = ['births', 'deaths', 'sojourn']
    stat_labels = ['E[B]', 'E[D]', 'E[S]']
    stat_filenames = ['bdi_consistency_B.pdf', 'bdi_consistency_D.pdf', 'bdi_consistency_S.pdf']

    cat_colors = {
        'kappa_lt_1': '#2176AE',
        'kappa_eq_1': '#D4A029',
        'kappa_gt_1': '#D32F2F',
    }
    markers = ['o', 's', '^', 'D', 'v']

    # Run all regimes
    all_data = []
    for r_idx, (lam, mu, t, label, cat) in enumerate(REGIMES):
        print(f"Running regime: λ={lam}, μ={mu}, T={t} ...")
        sim_data = run_gillespie_regime(lam, mu, t, N_SIMS, seed=r_idx * 1000)

        regime_points = {}
        for j in sorted(sim_data.keys()):
            if j > MAX_J or len(sim_data[j]['births']) < MIN_COUNT:
                continue
            result = analytic_stats(lam, mu, t, j)
            if result is not None:
                eb, ed, es = result
                regime_points[j] = {
                    'analytic': (eb, ed, es),
                    'sim': sim_data[j],
                }
                print(f"  j={j} (n={len(sim_data[j]['births'])}): "
                      f"E[B]={eb:.4f}, E[D]={ed:.4f}, E[S]={es:.4f}")

        all_data.append((lam, mu, t, label, cat, regime_points))

    # Generate one figure per statistic
    figdir = os.path.join(os.path.dirname(__file__), 'figures')

    for s_idx, (stat_key, stat_label, filename) in enumerate(
            zip(stat_keys, stat_labels, stat_filenames)):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))

        for r_idx, (lam, mu, t, label, cat, regime_points) in enumerate(all_data):
            if not regime_points:
                continue

            analytic_vals = []
            sim_means = []
            sim_stds = []
            j_vals = []

            for j, data in sorted(regime_points.items()):
                ana = data['analytic'][s_idx]
                sim_arr = np.array(data['sim'][stat_key])
                analytic_vals.append(ana)
                sim_means.append(np.mean(sim_arr))
                sim_stds.append(np.std(sim_arr))
                j_vals.append(j)

            analytic_vals = np.array(analytic_vals)
            sim_means = np.array(sim_means)
            sim_stds = np.array(sim_stds)

            color = cat_colors[cat]
            marker = markers[r_idx]

            ax.errorbar(analytic_vals, sim_means, yerr=sim_stds, fmt=marker,
                        color=color, ms=6, capsize=3, alpha=0.8,
                        label=label, markeredgecolor='white',
                        markeredgewidth=0.5, linewidth=1)

            # Label points with X(T) value
            for k, j in enumerate(j_vals):
                ax.annotate(str(j), (analytic_vals[k], sim_means[k]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=6, color=color, alpha=0.7)

        # y=x identity line
        all_vals = []
        for _, _, _, _, _, rp in all_data:
            for j, data in rp.items():
                all_vals.append(data['analytic'][s_idx])
                all_vals.append(np.mean(data['sim'][stat_key]))
        if all_vals:
            lo = min(all_vals)
            hi = max(all_vals)
            margin = (hi - lo) * 0.1 + 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    'k--', alpha=0.3, lw=1, label='$y = x$')
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

        ax.set_xlabel(f'Analytic {stat_label}', fontsize=11)
        ax.set_ylabel(f'Simulated {stat_label} (mean $\\pm$ std)', fontsize=11)
        ax.set_title(f'BDI Consistency: {stat_label}', fontsize=13)
        ax.legend(fontsize=7, loc='upper left')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=10)

        fig.tight_layout()
        outpath = os.path.join(figdir, filename)
        fig.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved {outpath}")


if __name__ == '__main__':
    make_figures()
