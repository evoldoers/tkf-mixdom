#!/usr/bin/env python3
"""Scatterplot validation: BDI sufficient statistics vs Gillespie simulations.

For each parameter regime (general and L'Hôpital), simulates many BDI
trajectories from X(0)=1, groups by final state X(T)=j, and compares:
  - Simulated mean E[B], E[D], E[S] (from Gillespie)
  - Analytic E[B], E[D], E[S] (from score function + conservation law)

Produces scatterplots with y=x reference line and regression fit.

Usage:
    python experiments/fig_bdi_lhopital.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from tkfmixdom.jax.core.bdi import (
    tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa,
)
from tkfmixdom.jax.simulate.simulate import simulate_bdi_gillespie


def recover_bdi_stats(d_lam, d_mu, lam, mu, t, i=1, j=1):
    """Recover E[B], E[D], E[S] from score derivatives via the conservation
    law of the BDI process. Inlined here since it was removed from the
    bdi module; equivalent to the formula used in fig_bdi_consistency.
    """
    if abs(lam - mu) < 1e-9:
        # L'Hôpital limit handled by caller via finite difference if needed.
        E_S = (j - i + mu * d_mu - lam * d_lam - lam * t) / max(lam - mu, 1e-30)
    else:
        E_S = (j - i + mu * d_mu - lam * d_lam - lam * t) / (lam - mu)
    E_B = lam * (d_lam + E_S + t)
    E_D = mu * (d_mu + E_S)
    return E_B, E_D, E_S


# Parameter regimes: (λ, μ, T, label)
REGIMES = [
    # General regimes (λ < μ, away from singularity)
    (0.10, 0.20, 1.0, "general_moderate"),
    (0.05, 0.15, 0.5, "general_low"),
    (0.15, 0.30, 1.5, "general_high"),
    # Moderate κ regimes
    (0.08, 0.12, 1.0, "moderate_kappa_067"),
    (0.05, 0.20, 1.5, "low_kappa_025"),
    # High κ
    (0.15, 0.25, 1.0, "high_kappa_06"),
]


def log_p_bdi(lam, mu, t, i, j):
    """Log P(X(T)=j | X(0)=i) for the TKF BDI process (ν=λ).

    Direct closed-form, JAX-differentiable. Only i=1 supported.
    P(1→j) = [α·(1-β) + (1-α)·γ·(1-β)] · β^{j-1}  for j≥1
    P(1→0) = (1-α)·(1-γ)·(1-β)
    where the (1-β) accounts for termination and β^{j-1} for the geometric
    distribution over additional births from the survivor/orphan's offspring.
    """
    alpha = jnp.exp(-mu * t)
    beta = tkf_beta(lam, mu, t)
    gamma = tkf_gamma(lam, mu, t)

    if j == 0:
        return jnp.log((1.0 - alpha) * (1.0 - gamma) * (1.0 - beta))
    else:
        # P(1→j) = [α(1-β) + (1-α)γ(1-β)] β^{j-1}
        # = (1-β) [α + (1-α)γ] β^{j-1}
        return (jnp.log(1.0 - beta) + jnp.log(alpha + (1.0 - alpha) * gamma)
                + (j - 1) * jnp.log(beta))


def score_based_expected_stats(lam, mu, t, j):
    """Compute E[B], E[D], E[S] for the TKF BDI transition i=1→j.

    Uses JAX autodiff on the BDI transition probability (not the pair HMM).
    The score function identity relates d(log P)/d(λ,μ) to BDI statistics.
    """
    lam_jnp = jnp.float64(lam)
    mu_jnp = jnp.float64(mu)

    d_lam = float(jax.grad(lambda l: log_p_bdi(l, mu_jnp, t, 1, j))(lam_jnp))
    d_mu = float(jax.grad(lambda m: log_p_bdi(lam_jnp, m, t, 1, j))(mu_jnp))

    E_B, E_D, E_S = recover_bdi_stats(d_lam, d_mu, lam, mu, t, i=1, j=j)
    return float(E_B), float(E_D), float(E_S)


def run_gillespie(lam, mu, t, n_sims=10_000_000):
    """Run Gillespie simulations from i=1, grouped by final state j."""
    np_rng = np.random.RandomState(42)
    results = defaultdict(lambda: {'births': [], 'deaths': [], 'sojourn': []})

    for _ in range(n_sims):
        j, nb, nd, ni, soj = simulate_bdi_gillespie(np_rng, 1, lam, mu, t)
        total_births = nb + ni  # offspring births + immigrations
        results[j]['births'].append(total_births)
        results[j]['deaths'].append(nd)
        results[j]['sojourn'].append(soj)

    return results


def make_scatterplot(ax, sim_data, analytic_fn, stat_key, regime_label,
                     stat_label=None, max_j=10):
    """Make a single scatterplot panel."""
    if stat_label is None:
        stat_label = stat_key
    sim_means = []
    sim_stds = []
    analytic_vals = []
    j_vals = []

    for j in sorted(sim_data.keys()):
        if j > max_j or len(sim_data[j][stat_key]) < 20:
            continue
        sim_mean = np.mean(sim_data[j][stat_key])
        sim_std = np.std(sim_data[j][stat_key]) / np.sqrt(len(sim_data[j][stat_key]))
        ana_val = analytic_fn(j)
        if ana_val is None:
            continue
        sim_means.append(sim_mean)
        sim_stds.append(sim_std)
        analytic_vals.append(ana_val)
        j_vals.append(j)

    if not sim_means:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return

    sim_means = np.array(sim_means)
    sim_stds = np.array(sim_stds)
    analytic_vals = np.array(analytic_vals)

    ax.errorbar(analytic_vals, sim_means, yerr=2*sim_stds, fmt='o', ms=4,
                capsize=2, alpha=0.7)

    # y=x reference line
    mn = min(analytic_vals.min(), sim_means.min())
    mx = max(analytic_vals.max(), sim_means.max())
    margin = (mx - mn) * 0.1 + 0.1
    ax.plot([mn-margin, mx+margin], [mn-margin, mx+margin], 'k--', alpha=0.3, lw=1)

    # Regression
    if len(analytic_vals) > 2:
        m, c = np.polyfit(analytic_vals, sim_means, 1)
        ax.plot([mn-margin, mx+margin],
                [m*(mn-margin)+c, m*(mx+margin)+c], 'r-', alpha=0.5, lw=1)
        ax.set_title(f"{stat_label} ({regime_label})\nm={m:.3f}, c={c:.3f}", fontsize=8)
    else:
        ax.set_title(f"{stat_label} ({regime_label})", fontsize=8)

    ax.set_xlabel("Analytic", fontsize=7)
    ax.set_ylabel("Simulated", fontsize=7)
    ax.tick_params(labelsize=6)


def main():
    jax.config.update("jax_enable_x64", True)

    n_regimes = len(REGIMES)
    fig, axes = plt.subplots(n_regimes, 3, figsize=(12, 3 * n_regimes))
    if n_regimes == 1:
        axes = axes[np.newaxis, :]

    for r_idx, (lam, mu, t, label) in enumerate(REGIMES):
        print(f"Regime: {label} (λ={lam}, μ={mu}, t={t})")

        # Run Gillespie
        sim_data = run_gillespie(lam, mu, t, n_sims=10_000_000)

        # Compute analytic values for each j
        analytic_cache = {}
        for j in sorted(sim_data.keys()):
            if j <= 10 and len(sim_data[j]['births']) >= 20:
                try:
                    eb, ed, es = score_based_expected_stats(lam, mu, t, j)
                    analytic_cache[j] = (eb, ed, es)
                    print(f"  j={j}: E[B]={eb:.3f}, E[D]={ed:.3f}, E[S]={es:.3f}")
                except Exception as e:
                    print(f"  j={j}: {e}")

        for s_idx, stat_name in enumerate(['births', 'deaths', 'sojourn']):
            stat_label = ['E[B]', 'E[D]', 'E[S]'][s_idx]

            def analytic_fn(j, idx=s_idx):
                if j in analytic_cache:
                    return analytic_cache[j][idx]
                return None

            make_scatterplot(axes[r_idx, s_idx], sim_data, analytic_fn,
                             stat_name, label, stat_label=stat_label)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), 'fig_bdi_lhopital.pdf')
    plt.savefig(outpath, bbox_inches='tight')
    print(f"Saved to {outpath}")


if __name__ == '__main__':
    main()
