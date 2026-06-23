#!/usr/bin/env python3
"""Q4: fit TKF92 and GGI-flowed-TKF92 using GAP COUNTS instead of 5x5
transition counts.

For each cherry alignment, we collect gap-type counts (i, j) — number of
deletions i, insertions j — for each of the four gap types (SM, MM, ME, SE).
The conditional LL uses the closed-form-DP gap probabilities (gap_counts_lib.
all_four_gap_probs) and subtracts the TKF92 ancestor singlet log-marginal
just as the transition-count fitters do.

Two models, both with 3 free params + 1 derived:
  - TKF92 constant (lam, mu, r) via (log mu, logit kappa, logit ext).
  - GGI-flowed (lam0, x_del, y_ins) with mu0 by reversibility, plus the
    closed-form GGI -> TKF92(theta(t)) trajectory.

Outputs the conditional LL for each and the delta vs the existing
transition-count fits.
"""
import sys, os, time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate_ggi_cherries import (
    simulate_one_pair, tally_transitions, S, M, I, D, E
)
from gap_counts_lib import (
    cherry_to_gap_counts, all_four_gap_probs,
    GAP_SM, GAP_MM, GAP_ME, GAP_SE,
)
from fit_ggi_cherryml import (
    tkf92_trans_full, tkf92_singlet_log_marginal, ggi_to_tkf92_at_t,
    ggi_stationary_log_marginal,
)
from fit_tkf92_cherryml import _tkf92_singlet_lm_aggregate

from tkfmixdom.jax.core.bdi import tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa


# -----------------------------------------------------------------
# Simulation: same protocol as compare_ggi_vs_tkf92_on_sim.py
# -----------------------------------------------------------------

LAM0_TRUE, MU0_TRUE = 0.04, 0.04
X_DEL_TRUE, Y_INS_TRUE = 0.6, 0.4
L_MEAN = 50
N_CHERRIES = 5000
TAU = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
SEED = 42
LMAX = 20   # max gap size before clipping (rare large gaps fold into [Lmax, Lmax])


def simulate_and_tally(lam0, mu0, x_del, y_ins, tau_centers, n_cherries, l_mean,
                        seed, Lmax):
    """Simulate cherries and return (gap_counts, trans_counts) tensors.

    gap_counts: shape (n_tau, 4, Lmax+1, Lmax+1)  -- 4 gap types SM/MM/ME/SE
    trans_counts: shape (n_tau, 5, 5)              -- for cross-check / singlet
    """
    rng = np.random.default_rng(seed)
    r_anc = max(0.0, 1.0 - 1.0 / l_mean)
    n_tau = len(tau_centers)
    gap_counts = np.zeros((n_tau, 4, Lmax + 1, Lmax + 1), dtype=np.float64)
    trans_counts = np.zeros((n_tau, 5, 5), dtype=np.float64)
    t_start = time.monotonic()
    for ti, tau in enumerate(tau_centers):
        for ci in range(n_cherries):
            L0 = int(rng.geometric(1.0 - r_anc))
            seq = simulate_one_pair(L0, lam0, mu0, x_del, y_ins, tau, rng)
            cherry_to_gap_counts(seq, gap_counts, ti, Lmax)
            trans_counts[ti] += tally_transitions(seq)
        print(f"  tau={tau:.4f}: {n_cherries} cherries  "
              f"(elapsed {time.monotonic()-t_start:.1f}s)")
    return gap_counts, trans_counts


# -----------------------------------------------------------------
# Gap-conditional LL for one (lam, mu, r, t) combination
# -----------------------------------------------------------------

def gap_joint_ll_one_tau(lam, mu, t, r, gap_counts_tau, Lmax):
    """Sum_{g in SM,MM,ME,SE} sum_{i, j} counts[g, i, j] * log G[g, i, j]."""
    T = tkf92_trans_full(lam, mu, t, r)
    probs = all_four_gap_probs(T, Lmax)
    log_SM = jnp.log(jnp.maximum(probs['SM'], 1e-30))
    log_MM = jnp.log(jnp.maximum(probs['MM'], 1e-30))
    log_ME = jnp.log(jnp.maximum(probs['ME'], 1e-30))
    log_SE = jnp.log(jnp.maximum(probs['SE'], 1e-30))
    ll = (jnp.sum(gap_counts_tau[GAP_SM] * log_SM)
          + jnp.sum(gap_counts_tau[GAP_MM] * log_MM)
          + jnp.sum(gap_counts_tau[GAP_ME] * log_ME)
          + jnp.sum(gap_counts_tau[GAP_SE] * log_SE))
    return ll


def conditional_ll_tkf92_constant(log_del, logit_kappa, logit_ext,
                                    gap_counts_j, trans_counts_j, tau_j, Lmax):
    """Constant TKF92(lam, mu, r) under the gap-counts LL, minus the ancestor
    singlet log-marginal."""
    mu_ = jnp.exp(log_del)
    kappa_ = jax.nn.sigmoid(logit_kappa)
    lam_ = kappa_ * mu_
    r_ = jax.nn.sigmoid(logit_ext)

    def per_tau(ti):
        return gap_joint_ll_one_tau(lam_, mu_, tau_j[ti], r_,
                                      gap_counts_j[ti], Lmax)

    n_tau = tau_j.shape[0]
    joint_ll = jnp.sum(jax.vmap(per_tau)(jnp.arange(n_tau)))
    # Ancestor singlet (constant theta): aggregate over all bins.
    t_anc_total = (jnp.sum(trans_counts_j[:, :, M])
                    + jnp.sum(trans_counts_j[:, :, D]))
    n_cherries_total = jnp.sum(trans_counts_j[:, :, E])
    singlet = _tkf92_singlet_lm_aggregate(
        kappa_, r_, n_cherries_total, t_anc_total)
    return joint_ll - singlet


def joint_ll_tkf92_constant(log_del, logit_kappa, logit_ext,
                              gap_counts_j, trans_counts_j, tau_j, Lmax):
    """Joint LL (no singlet subtraction): the FULL log P(joint pair) under
    TKF92, computed by summing gap_joint_ll over tau bins WITHOUT subtracting
    the ancestor singlet log-marginal.

    At kappa=1 the joint diverges via the (1-kappa) terminator factor and
    is NOT cancelled by anything, so Adam is repelled from the kappa=1
    boundary.  This is the model-appropriate joint likelihood.
    """
    mu_ = jnp.exp(log_del)
    kappa_ = jax.nn.sigmoid(logit_kappa)
    lam_ = kappa_ * mu_
    r_ = jax.nn.sigmoid(logit_ext)

    def per_tau(ti):
        return gap_joint_ll_one_tau(lam_, mu_, tau_j[ti], r_,
                                      gap_counts_j[ti], Lmax)

    n_tau = tau_j.shape[0]
    return jnp.sum(jax.vmap(per_tau)(jnp.arange(n_tau)))


def joint_ll_ggi_flowed_native(log_lam0, logit_x_del, logit_y_ins,
                                gap_counts_j, trans_counts_j, tau_j, Lmax):
    """log P(joint pair) under GGI-flowed with NATIVE GGI ancestor prior.

    Equals (current conditional_ll_ggi_flowed) + GGI_native_marginal
    (where conditional uses TKF92-at-flowed-params singlet, which we add
    back via the existing formula, then add the GGI-native marginal).

    Equivalently: replace the per-tau TKF92(flowed) singlet with the
    GGI-native (lam0, mu0, x, y) geometric marginal aggregated across all
    tau bins.
    """
    lam0 = jnp.exp(log_lam0)
    xv = jax.nn.sigmoid(logit_x_del)
    yv = jax.nn.sigmoid(logit_y_ins)
    mu0 = lam0 * yv * (1 - yv) / jnp.maximum(xv * (1 - xv), 1e-30)

    def per_tau(ti):
        t = tau_j[ti]
        lam_t, mu_t, r_t = ggi_to_tkf92_at_t(lam0, mu0, xv, yv, t)
        return gap_joint_ll_one_tau(lam_t, mu_t, t, r_t, gap_counts_j[ti], Lmax)

    n_tau = tau_j.shape[0]
    joint_total = jnp.sum(jax.vmap(per_tau)(jnp.arange(n_tau)))

    # GGI-native ancestor log-marginal (aggregated, t-independent).
    t_anc_total = jnp.sum(trans_counts_j[:, :, M]) + jnp.sum(trans_counts_j[:, :, D])
    n_cherries_total = jnp.sum(trans_counts_j[:, :, E])
    log_p_anc = ggi_stationary_log_marginal(
        lam0, mu0, xv, yv, n_cherries_total, t_anc_total)
    return joint_total + log_p_anc


def joint_ll_ggi_flowed_stable_native(log_mu0, logit_rho, logit_y_ins,
                                       gap_counts_j, trans_counts_j, tau_j, Lmax,
                                       x_branch_upper=False):
    """STABLE-parameterized GGI-flowed joint LL with native prior."""
    lam0, mu0, xv, yv = unpack_stable_ggi(
        log_mu0, logit_rho, logit_y_ins, x_branch_upper=x_branch_upper)

    def per_tau(ti):
        t = tau_j[ti]
        lam_t, mu_t, r_t = ggi_to_tkf92_at_t(lam0, mu0, xv, yv, t)
        return gap_joint_ll_one_tau(lam_t, mu_t, t, r_t, gap_counts_j[ti], Lmax)

    n_tau = tau_j.shape[0]
    joint_total = jnp.sum(jax.vmap(per_tau)(jnp.arange(n_tau)))

    t_anc_total = jnp.sum(trans_counts_j[:, :, M]) + jnp.sum(trans_counts_j[:, :, D])
    n_cherries_total = jnp.sum(trans_counts_j[:, :, E])
    log_p_anc = ggi_stationary_log_marginal(
        lam0, mu0, xv, yv, n_cherries_total, t_anc_total)
    return joint_total + log_p_anc


def joint_ll_ggi_frozen_stable_native(log_mu0, logit_rho, logit_y_ins,
                                       gap_counts_j, trans_counts_j, tau_j, Lmax,
                                       x_branch_upper=False):
    """STABLE-parameterized GGI-FROZEN joint LL with native prior."""
    lam0, mu0, xv, yv = unpack_stable_ggi(
        log_mu0, logit_rho, logit_y_ins, x_branch_upper=x_branch_upper)

    def per_tau(ti):
        t = tau_j[ti]
        lam_t, mu_t, r_t = _ggi_to_tkf92_at_t_frozen(lam0, mu0, xv, yv, t)
        return gap_joint_ll_one_tau(lam_t, mu_t, t, r_t, gap_counts_j[ti], Lmax)

    n_tau = tau_j.shape[0]
    joint_total = jnp.sum(jax.vmap(per_tau)(jnp.arange(n_tau)))

    t_anc_total = jnp.sum(trans_counts_j[:, :, M]) + jnp.sum(trans_counts_j[:, :, D])
    n_cherries_total = jnp.sum(trans_counts_j[:, :, E])
    log_p_anc = ggi_stationary_log_marginal(
        lam0, mu0, xv, yv, n_cherries_total, t_anc_total)
    return joint_total + log_p_anc


def conditional_ll_ggi_flowed(log_lam0, logit_x_del, logit_y_ins,
                                gap_counts_j, trans_counts_j, tau_j, Lmax):
    """GGI-flowed TKF92(theta(t)) under the gap-counts LL, minus the
    TKF92-at-theta(t) ancestor singlet per tau bin (matches the GGI
    transition-count conditional convention).

    UNCONSTRAINED parameterization: (log_lam0, logit_x, logit_y). mu0 is
    derived from reversibility. Allows ρ = lam0/mu0 > 1 — closed-form
    extrapolates beyond the valid GGI region in that case.
    """
    lam0 = jnp.exp(log_lam0)
    xv = jax.nn.sigmoid(logit_x_del)
    yv = jax.nn.sigmoid(logit_y_ins)
    mu0 = lam0 * yv * (1 - yv) / jnp.maximum(xv * (1 - xv), 1e-30)

    def per_tau(ti):
        t = tau_j[ti]
        lam_t, mu_t, r_t = ggi_to_tkf92_at_t(lam0, mu0, xv, yv, t)
        joint = gap_joint_ll_one_tau(
            lam_t, mu_t, t, r_t, gap_counts_j[ti], Lmax)
        # Per-bin singlet uses theta(t) — same convention as fit_ggi_cherryml
        n_ij = trans_counts_j[ti]
        t_anc = jnp.sum(n_ij[:, M]) + jnp.sum(n_ij[:, D])
        n_cherries = jnp.sum(n_ij[:, E])
        singlet = tkf92_singlet_log_marginal(lam_t, mu_t, r_t, n_cherries, t_anc)
        return joint - singlet

    n_tau = tau_j.shape[0]
    return jnp.sum(jax.vmap(per_tau)(jnp.arange(n_tau)))


def unpack_stable_ggi(log_mu0, logit_rho, logit_y_ins, x_branch_upper=False):
    """Stable (mu0, rho, y) -> (lam0, mu0, x, y).

    rho = sigmoid(logit_rho) in (0, 1)   -- enforced stability lam0 <= mu0
    mu0 = exp(log_mu0) > 0
    y   = sigmoid(logit_y_ins) in (0, 1)
    lam0 = rho * mu0
    x from reversibility: x(1-x) = rho * y(1-y).  Two branches:
      x_lower = (1 - sqrt(1 - 4q))/2  in (0, 1/2]
      x_upper = (1 + sqrt(1 - 4q))/2  in [1/2, 1)
    x_branch_upper is a STATIC Python bool; pick the upper branch when True.

    Domain: rho in (0,1), y in (0,1) ensures q = rho * y(1-y) <= 1/4, so
    the discriminant is >= 0.
    """
    mu0 = jnp.exp(log_mu0)
    rho = jax.nn.sigmoid(logit_rho)
    yv = jax.nn.sigmoid(logit_y_ins)
    lam0 = rho * mu0
    q = rho * yv * (1.0 - yv)
    disc = jnp.maximum(1.0 - 4.0 * q, 0.0)
    sqrt_disc = jnp.sqrt(disc)
    if x_branch_upper:
        xv = (1.0 + sqrt_disc) / 2.0
    else:
        xv = (1.0 - sqrt_disc) / 2.0
    return lam0, mu0, xv, yv


def _ggi_to_tkf92_at_t_frozen(lam0, mu0, x, y, t):
    """Like ggi_to_tkf92_at_t but lam(t)=lam0 and mu(t)=mu0 stay frozen at
    their t=0 values.  r(t) still flows via the closed form.

    Skips the renormalization lam(t) = lam0/(1-r(t)), mu(t) = mu0/(1-r(t)).
    """
    num = lam0 * y * (1 - x) + mu0 * x * (1 - y)
    den = lam0 * (1 - y) + mu0 * (1 - x)
    r_boundary = num / jnp.maximum(den, 1e-30)
    r_inf = r_boundary / (2 - r_boundary)
    k = (lam0 + mu0) * (2 - r_boundary) / jnp.maximum(1 - r_boundary, 1e-30)
    r_t = r_inf + (r_boundary - r_inf) * jnp.exp(-k * t)
    # FROZEN: do NOT divide by (1 - r(t))
    return lam0, mu0, r_t


def conditional_ll_ggi_frozen_stable(log_mu0, logit_rho, logit_y_ins,
                                     gap_counts_j, trans_counts_j, tau_j, Lmax,
                                     x_branch_upper=False):
    """GGI-FROZEN variant: lam(t)=lam0 and mu(t)=mu0 constant; only r(t) flows.

    Same (log_mu0, logit_rho, logit_y, x_branch) interface as
    conditional_ll_ggi_flowed_stable.  Differs only in skipping the
    lam(t) = lam0/(1-r(t)) renormalization.
    """
    lam0, mu0, xv, yv = unpack_stable_ggi(
        log_mu0, logit_rho, logit_y_ins, x_branch_upper=x_branch_upper)

    def per_tau(ti):
        t = tau_j[ti]
        lam_t, mu_t, r_t = _ggi_to_tkf92_at_t_frozen(lam0, mu0, xv, yv, t)
        joint = gap_joint_ll_one_tau(
            lam_t, mu_t, t, r_t, gap_counts_j[ti], Lmax)
        n_ij = trans_counts_j[ti]
        t_anc = jnp.sum(n_ij[:, M]) + jnp.sum(n_ij[:, D])
        n_cherries = jnp.sum(n_ij[:, E])
        singlet = tkf92_singlet_log_marginal(lam_t, mu_t, r_t, n_cherries, t_anc)
        return joint - singlet

    n_tau = tau_j.shape[0]
    return jnp.sum(jax.vmap(per_tau)(jnp.arange(n_tau)))


def conditional_ll_ggi_flowed_stable(log_mu0, logit_rho, logit_y_ins,
                                     gap_counts_j, trans_counts_j, tau_j, Lmax,
                                     x_branch_upper=False):
    """GGI-flowed conditional LL, STABLE-parameterized.

    Free params (3 continuous): log_mu0, logit_rho, logit_y.
    Static branch flag: x_branch_upper (False = lower root x<=1/2, True =
    upper root x>=1/2).  Run once per branch to cover both halves of model
    space; reversibility (lam0 y(1-y) = mu0 x(1-x)) is enforced by
    construction.

    Enforces rho = lam0/mu0 <= 1 (closed-form GGI valid region).
    """
    lam0, mu0, xv, yv = unpack_stable_ggi(
        log_mu0, logit_rho, logit_y_ins, x_branch_upper=x_branch_upper)

    def per_tau(ti):
        t = tau_j[ti]
        lam_t, mu_t, r_t = ggi_to_tkf92_at_t(lam0, mu0, xv, yv, t)
        joint = gap_joint_ll_one_tau(
            lam_t, mu_t, t, r_t, gap_counts_j[ti], Lmax)
        n_ij = trans_counts_j[ti]
        t_anc = jnp.sum(n_ij[:, M]) + jnp.sum(n_ij[:, D])
        n_cherries = jnp.sum(n_ij[:, E])
        singlet = tkf92_singlet_log_marginal(lam_t, mu_t, r_t, n_cherries, t_anc)
        return joint - singlet

    n_tau = tau_j.shape[0]
    return jnp.sum(jax.vmap(per_tau)(jnp.arange(n_tau)))


# -----------------------------------------------------------------
# Adam loop
# -----------------------------------------------------------------

def adam_max(val_and_grad, init_params, lr, n_steps, log_interval, label=''):
    params = list(init_params)
    m = [jnp.zeros_like(p) for p in params]
    v = [jnp.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = -float('inf')
    best_p = [float(p) for p in params]
    t0 = time.monotonic()
    for step in range(n_steps):
        ll, grads = val_and_grad(*params)
        llv = float(ll)
        if not np.isfinite(llv):
            print(f"  [{label}] step {step}: non-finite LL ({llv}), stopping")
            break
        if llv > best:
            best = llv
            best_p = [float(p) for p in params]
        for i in range(len(params)):
            g = grads[i]
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * (g * g)
            mh = m[i] / (1 - b1 ** (step + 1))
            vh = v[i] / (1 - b2 ** (step + 1))
            params[i] = params[i] + lr * mh / (jnp.sqrt(vh) + eps)
        if log_interval and step % log_interval == 0:
            elapsed = time.monotonic() - t0
            print(f"  [{label}] step {step:5d}  LL={llv:14.2f}  best={best:14.2f}"
                  f"  ({elapsed:.1f}s)")
    return best_p, best


def main():
    print(f"Sim params: lam0={LAM0_TRUE}, mu0={MU0_TRUE}, "
          f"x_del={X_DEL_TRUE}, y_ins={Y_INS_TRUE}, Lmax={LMAX}")
    print(f"Simulating + tallying gap counts...")
    gap_counts, trans_counts = simulate_and_tally(
        LAM0_TRUE, MU0_TRUE, X_DEL_TRUE, Y_INS_TRUE,
        TAU, N_CHERRIES, L_MEAN, SEED, LMAX)
    total = int(trans_counts.sum())
    print(f"\nTotal transitions: {total}")
    print(f"Gap count totals per type (summed over tau, i, j):")
    for g_name, g_idx in zip(('SM', 'MM', 'ME', 'SE'), range(4)):
        print(f"  {g_name}: {int(gap_counts[:, g_idx].sum())}")

    gap_counts_j = jnp.asarray(gap_counts, jnp.float32)
    trans_counts_j = jnp.asarray(trans_counts, jnp.float32)
    tau_j = jnp.asarray(TAU, jnp.float32)
    Lmax_j = LMAX

    # ----- TKF92 constant fit -----
    print(f"\n{'='*70}")
    print("TKF92 CONSTANT fit on GAP counts")
    print(f"{'='*70}")
    def tkf_loss(log_del, logit_kappa, logit_ext):
        return conditional_ll_tkf92_constant(
            log_del, logit_kappa, logit_ext,
            gap_counts_j, trans_counts_j, tau_j, Lmax_j)
    tkf_vag = jax.jit(jax.value_and_grad(tkf_loss, argnums=(0, 1, 2)))

    init_tkf = [
        jnp.log(jnp.float32(0.08)),
        jnp.log(jnp.float32(0.95 / 0.05)),
        jnp.log(jnp.float32(0.5 / 0.5)),
    ]
    t0 = time.monotonic()
    ll0, _ = tkf_vag(*init_tkf)
    jax.block_until_ready(ll0)
    print(f"  JIT + first eval: {time.monotonic()-t0:.1f}s, initial LL = {float(ll0):.2f}")
    bp_tkf, bll_tkf = adam_max(tkf_vag, init_tkf, lr=0.005, n_steps=2000,
                                log_interval=300, label='TKF92')

    mu_t = float(np.exp(bp_tkf[0]))
    kappa_t = 1 / (1 + float(np.exp(-bp_tkf[1])))
    r_t = 1 / (1 + float(np.exp(-bp_tkf[2])))
    lam_t = kappa_t * mu_t
    print(f"\n  TKF92 best LL = {bll_tkf:.2f}")
    print(f"    lam={lam_t:.5f}, mu={mu_t:.5f}, kappa={kappa_t:.4f}, r={r_t:.4f}")

    # ----- GGI-flowed fit -----
    print(f"\n{'='*70}")
    print("GGI-FLOWED (closed-form) fit on GAP counts")
    print(f"{'='*70}")
    def ggi_loss(log_lam0, logit_x, logit_y):
        return conditional_ll_ggi_flowed(
            log_lam0, logit_x, logit_y,
            gap_counts_j, trans_counts_j, tau_j, Lmax_j)
    ggi_vag = jax.jit(jax.value_and_grad(ggi_loss, argnums=(0, 1, 2)))

    ggi_inits = [
        ('truth',    LAM0_TRUE, X_DEL_TRUE, Y_INS_TRUE),
        ('generic',  0.05, 0.5, 0.5),
        ('symmetric',0.05, 0.65, 0.65),
        ('ins-heavy',0.05, 0.3, 0.7),
    ]
    best_ggi_ll = -float('inf')
    best_ggi_p = None
    best_ggi_lbl = None
    for label, l0, xi, yi in ggi_inits:
        init = [jnp.log(jnp.float32(l0)),
                jnp.log(jnp.float32(xi / (1 - xi))),
                jnp.log(jnp.float32(yi / (1 - yi)))]
        print(f"\n  init: {label} (lam0={l0}, x={xi}, y={yi})")
        bp, bll = adam_max(ggi_vag, init, lr=0.002, n_steps=1500,
                            log_interval=300, label=f'GGI/{label}')
        if bll > best_ggi_ll:
            best_ggi_ll = bll
            best_ggi_p = bp
            best_ggi_lbl = label

    lam0_g = float(np.exp(best_ggi_p[0]))
    x_g = 1 / (1 + float(np.exp(-best_ggi_p[1])))
    y_g = 1 / (1 + float(np.exp(-best_ggi_p[2])))
    mu0_g = lam0_g * y_g * (1 - y_g) / max(x_g * (1 - x_g), 1e-30)
    print(f"\n  GGI-flowed best LL = {best_ggi_ll:.2f} (init: {best_ggi_lbl})")
    print(f"    lam0={lam0_g:.5f}, mu0={mu0_g:.5f}, x={x_g:.4f}, y={y_g:.4f}")

    # ----- Summary -----
    print(f"\n{'='*70}")
    print("SUMMARY (gap-counts conditional LL)")
    print(f"{'='*70}")
    print(f"   TKF92 constant best       = {bll_tkf:>14.2f}")
    print(f"   GGI-flowed (closed-form)  = {best_ggi_ll:>14.2f}")
    delta = best_ggi_ll - bll_tkf
    print(f"   delta(GGI - TKF92)        = {delta:>+14.2f}  "
          f"(per cherry: {delta/(N_CHERRIES*len(TAU)):.3f} nats)")
    print()
    print("   Reference (TRANSITION-counts conditional LL on same sim, from")
    print("   compare_ggi_vs_tkf92_on_sim.py):")
    print("     TKF92 constant best       = -689,619")
    print("     GGI @ truth                = -664,563")
    print("     GGI closed-form best       = -651,116")
    print()
    print("   The gap-counts LL is over a *richer* representation (full (i,j)")
    print("   tables vs marginal 5x5 transitions), so the magnitude differs")
    print("   from the transition-count LL.  Only relative comparisons within")
    print("   the same representation are meaningful.")


if __name__ == "__main__":
    main()
