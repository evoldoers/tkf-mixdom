#!/usr/bin/env python3
"""Figure: BDI stats — estimated vs true.

For each parameter regime (lambda, mu, t):
  1. Sample initial population i from TKF91 stationary distribution Geometric(1-kappa)
  2. Simulate the continuous-time BDI process (Gillespie), recording:
     - births (from existing links), deaths, immigrations, sojourn time
     - final population j
  3. Compute score-function estimates E[B], E[D], E[S] given (i, j, lambda, mu, t)
     using numerical finite differences on the forward DP log P(j|i)
  4. Average true and estimated values over N trials
  5. Plot estimated (y) vs true (x) with error bars, annotated by parameters

Produces four panels: births, deaths, immigrations, dwell time.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tkfmixdom.jax.core.params import tkf_kappa, recover_bdi_stats
from tkfmixdom.jax.simulate.simulate import simulate_bdi_gillespie


# Parameter regimes: (lambda, mu, t)
REGIMES = [
    (0.05, 0.10, 0.5),
    (0.10, 0.20, 0.5),
    (0.10, 0.20, 1.0),
    (0.10, 0.20, 2.0),
    (0.15, 0.25, 1.0),
    (0.15, 0.25, 2.0),
    (0.05, 0.30, 1.0),
    (0.05, 0.30, 2.0),
    (0.20, 0.25, 0.5),
    (0.20, 0.25, 1.0),
    (0.01, 0.05, 1.0),
    (0.01, 0.05, 3.0),
]

N_TRIALS = 200  # trials per regime


def bdi_logprob(i, j, ins_rate, del_rate, t):
    """Compute log P(j|i) for the BDI process underlying TKF91.

    P(j|i) = probability of j mortal links at time t given i mortal links at time 0,
    with birth rate lambda per link, death rate mu per link, immigration rate lambda.

    Uses a DP that processes ancestor links one by one:
      F[a, d] = P(d descendants after processing a of i ancestor links)

    Transitions:
      - Link survives (prob alpha): contributes 1 descendant + geometric(beta) offspring
      - Link dies (prob 1-alpha):
          - No orphan (prob 1-gamma): no new descendants
          - Orphan (prob gamma): 1 descendant + geometric(beta) offspring
      - Before first ancestor link: immortal link spawns geometric(beta) offspring

    Args:
        i: ancestor population (number of mortal links)
        j: descendant population
        ins_rate, del_rate, t: BDI/TKF91 parameters

    Returns:
        log P(j | i)
    """
    alpha = np.exp(-del_rate * t)
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

    # F[d] = P(d descendants so far) after processing some ancestor links
    # Max possible descendants: j (we only need up to j)
    j_max = j + 1

    # Initialize: immortal link spawns geometric(beta) offspring
    # P(d offspring) = (1-beta) * beta^d
    F = np.zeros(j_max)
    for d in range(j_max):
        F[d] = (1 - beta) * beta**d

    # Process each of i ancestor links
    for a in range(i):
        F_new = np.zeros(j_max)
        for d in range(j_max):
            # Case 1: link dies, no orphan
            F_new[d] += (1 - alpha) * (1 - gamma) * F[d]

            # Case 2: link survives OR orphan: contributes 1 + geometric(beta) offspring
            # d = d' + 1 + k where k ~ geometric(beta), P(k) = (1-beta)*beta^k
            # So for each d' < d: contribution = F[d'] * [alpha + (1-alpha)*gamma] * (1-beta) * beta^{d-d'-1}
            if d > 0:
                survive_or_orphan = alpha + (1 - alpha) * gamma
                for dp in range(d):
                    k = d - dp - 1  # number of offspring (k >= 0)
                    F_new[d] += F[dp] * survive_or_orphan * (1 - beta) * beta**k

        F = F_new

    if j >= j_max or F[j] <= 0:
        return -np.inf
    return np.log(F[j])


def bdi_mortal_logprob(i, m, ins_rate, del_rate, t):
    """Compute log P_mortal(m | i): probability that i mortal links produce m descendants.

    This excludes the immortal link's contribution. Descendants include surviving
    links plus orphan offspring.

    Args:
        i: number of mortal links
        m: number of descendants from mortal links
        ins_rate, del_rate, t: parameters

    Returns:
        log P_mortal(m | i)
    """
    alpha = np.exp(-del_rate * t)
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

    # DP without immortal link: F[d] after processing a ancestor links
    m_max = m + 1
    F = np.zeros(m_max)
    F[0] = 1.0  # No immortal link contribution

    for a in range(i):
        F_new = np.zeros(m_max)
        for d in range(m_max):
            # Link dies, no orphan
            F_new[d] += (1 - alpha) * (1 - gamma) * F[d]
            # Link survives or orphan: 1 + geometric(beta)
            if d > 0:
                survive_or_orphan = alpha + (1 - alpha) * gamma
                for dp in range(d):
                    k = d - dp - 1
                    F_new[d] += F[dp] * survive_or_orphan * (1 - beta) * beta**k
        F = F_new

    if m >= m_max or F[m] <= 0:
        return -np.inf
    return np.log(F[m])


def score_function_bdi(i, j, ins_rate, del_rate, t):
    """Compute E[births], E[deaths], E[immigrations], E[sojourn] from score function.

    Uses the decomposition P(j|i) = sum_k P_imm(k) * P_mortal(j-k|i) to
    compute E[immigrations|i,j] exactly, then recovers E[B_total], E[D], E[S]
    from the full score function and deduces E[births] = E[B_total] - E[imm].
    """
    eps = 1e-6

    lp_base = bdi_logprob(i, j, ins_rate, del_rate, t)
    if not np.isfinite(lp_base):
        return None

    # BDI parameters
    alpha = np.exp(-del_rate * t)
    eta = np.exp(-ins_rate * t)
    denom = del_rate * eta - ins_rate * alpha
    beta = ins_rate * (eta - alpha) / denom if abs(denom) > 1e-30 else 0.0

    # E[immigrations | i,j] = sum_k k * P(k from immortal) * P_mortal(j-k|i) / P(j|i)
    P_j_i = np.exp(lp_base)
    E_imm = 0.0
    for k in range(j + 1):
        p_imm_k = (1 - beta) * beta**k  # P(k offspring from immortal)
        m = j - k  # remaining descendants from mortal links
        lp_mortal = bdi_mortal_logprob(i, m, ins_rate, del_rate, t)
        if np.isfinite(lp_mortal):
            E_imm += k * p_imm_k * np.exp(lp_mortal) / P_j_i

    # Score derivatives for total BDI stats
    lp_lam_plus = bdi_logprob(i, j, ins_rate + eps, del_rate, t)
    lp_lam_minus = bdi_logprob(i, j, ins_rate - eps, del_rate, t)
    d_lam = (lp_lam_plus - lp_lam_minus) / (2 * eps)

    lp_mu_plus = bdi_logprob(i, j, ins_rate, del_rate + eps, t)
    lp_mu_minus = bdi_logprob(i, j, ins_rate, del_rate - eps, t)
    d_mu = (lp_mu_plus - lp_mu_minus) / (2 * eps)

    E_B, E_D, E_S = recover_bdi_stats(d_lam, d_mu, ins_rate, del_rate, t, i, j)
    E_B, E_D, E_S = float(E_B), float(E_D), float(E_S)

    # E_B = births_from_existing + immigrations
    E_births = E_B - E_imm

    return E_births, float(E_D), E_imm, float(E_S)


def run_regime(ins_rate, del_rate, t, n_trials, seed=0):
    """Run one parameter regime: simulate + estimate."""
    kappa = float(tkf_kappa(ins_rate, del_rate))
    np_rng = np.random.RandomState(seed)

    true_births = []
    true_deaths = []
    true_immigrations = []
    true_sojourn = []

    est_births = []
    est_deaths = []
    est_immigrations = []
    est_sojourn = []

    for trial in range(n_trials):
        # Sample i from stationary distribution
        if kappa < 1e-10:
            i = 0
        else:
            u = np_rng.random()
            if u < 1e-15:
                u = 1e-15
            i = int(np.floor(np.log(u) / np.log(kappa)))
            i = min(i, 100)  # cap to keep DP tractable

        # Gillespie simulation
        j, nb, nd, ni, soj = simulate_bdi_gillespie(np_rng, i, ins_rate, del_rate, t)
        j = min(j, 100)  # cap for DP

        # Score function estimates
        result = score_function_bdi(i, j, ins_rate, del_rate, t)
        if result is None:
            continue

        eb, ed, ei, es = result

        true_births.append(nb)
        true_deaths.append(nd)
        true_immigrations.append(ni)
        true_sojourn.append(soj)

        est_births.append(eb)
        est_deaths.append(ed)
        est_immigrations.append(ei)
        est_sojourn.append(es)

    return {
        'true': (np.array(true_births), np.array(true_deaths),
                 np.array(true_immigrations), np.array(true_sojourn)),
        'est': (np.array(est_births), np.array(est_deaths),
                np.array(est_immigrations), np.array(est_sojourn)),
    }


def make_figure(output_path='fig_bdi_stats.pdf'):
    """Generate the 4-panel BDI stats figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    stat_names = ['Births', 'Deaths', 'Immigrations', 'Dwell time']

    colors = plt.cm.tab10(np.linspace(0, 1, len(REGIMES)))

    for regime_idx, (lam, mu, t) in enumerate(REGIMES):
        print(f"Running regime lam={lam}, mu={mu}, t={t}...")
        result = run_regime(lam, mu, t, N_TRIALS, seed=regime_idx * 1000)

        if len(result['true'][0]) == 0:
            print(f"  Skipped (no valid trials)")
            continue

        label = f"({lam},{mu},{t})"
        color = colors[regime_idx]

        for panel_idx in range(4):
            ax = axes[panel_idx // 2][panel_idx % 2]
            true_vals = result['true'][panel_idx]
            est_vals = result['est'][panel_idx]

            true_mean = np.mean(true_vals)
            est_mean = np.mean(est_vals)
            true_se = np.std(true_vals) / np.sqrt(len(true_vals))
            est_se = np.std(est_vals) / np.sqrt(len(est_vals))

            ax.errorbar(true_mean, est_mean,
                        xerr=true_se, yerr=est_se,
                        fmt='o', color=color, markersize=5, capsize=3,
                        label=label if panel_idx == 0 else None)

    # Formatting
    for panel_idx in range(4):
        ax = axes[panel_idx // 2][panel_idx % 2]
        ax.set_xlabel(f'True (Gillespie) mean {stat_names[panel_idx].lower()}')
        ax.set_ylabel(f'Score function E[{stat_names[panel_idx]}]')
        ax.set_title(stat_names[panel_idx])

        # Identity line
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')

    # Legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.22, 0.5),
               fontsize=7, title=r'$(\lambda,\mu,t)$')

    fig.suptitle('BDI Statistics: Score Function vs Gillespie Simulation', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved figure to {output_path}")
    return output_path


if __name__ == '__main__':
    make_figure()
