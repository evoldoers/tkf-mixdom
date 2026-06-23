#!/usr/bin/env python3
"""Fit GGI parameters (lambda_0, mu_0, x, y) to CherryML cherry-counts tensor
via Adam on the complete-data log-likelihood.

Unlike fit_tkf92_cherryml.py, this version:
  - Estimates the *underlying GGI* generator parameters, not the TKF92 surrogate.
  - Imposes GGI reversibility:   lambda_0 * y * (1 - y) = mu_0 * x * (1 - x).
  - At each tau bin, derives the TKF92 surrogate (lambda(t), mu(t), r(t)) via
    the closed-form approximations from composition-renormalization.tex:

       r*(0)  = (lam0 * y * (1-x) + mu0 * x * (1-y)) /
                (lam0 * (1-x)       + mu0 * (1-y))
       r_inf  = r*(0) / (2 - r*(0))
       k      = (lam0 + mu0) * (2 - r*(0)) / (1 - r*(0))
       r(t)   = r_inf + (r*(0) - r_inf) * exp(-k * t)
       lam(t) = lam0 / (1 - r(t))
       mu(t)  = mu0  / (1 - r(t))

    and uses (lam(t), mu(t), r(t)) as the TKF92 pair-HMM transition-matrix
    parameters at branch length t.
  - The Adam objective is the composite log-likelihood
       sum_{tau, i, j}  n_{tau, i, j} * log P_{TKF92(lam(t), mu(t), r(t))}[i, j]
    summed over the (S, M, I, D, E) transition pairs.

The free parameters are (log_lam0, logit_x, logit_y); mu_0 is derived from the
reversibility constraint.

Usage:
    cd python && uv run python experiments/fit_ggi_cherryml.py
"""

import sys
import time
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tkfmixdom.jax.core.bdi import tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa


# State indices (matches fit_tkf92_cherryml.py and build_tkf92_cherry_counts.py)
S, M, I, D, E = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# TKF92 pair HMM transition matrix (JAX, jittable)
# ---------------------------------------------------------------------------

def tkf92_trans_full(ins_rate, del_rate, t, ext):
    """Build 5x5 TKF92 Pair HMM transition matrix (rows sum to 1).

    Row/col order: S, M, I, D, E.
    """
    alpha = tkf_alpha(del_rate, t)
    beta = tkf_beta(ins_rate, del_rate, t)
    gamma = tkf_gamma(ins_rate, del_rate, t)
    kappa = tkf_kappa(ins_rate, del_rate)
    # Stable 1-kappa: subtraction-first so it is exact 0 at ins_rate == del_rate.
    # Using `1 - kappa` with kappa = lam/mu drifts to ~1e-8 under XLA fusion
    # even when lam == mu, polluting the E-column probabilities and the
    # log-likelihood (see Pfam Lmax=20 GGI fit, 2026-05-29).
    one_minus_kappa = jnp.maximum(
        0.0, (del_rate - ins_rate) / jnp.maximum(del_rate, 1e-30))

    tau = jnp.zeros((5, 5))

    # S, M, I rows use beta
    for src in (S, M, I):
        tau = tau.at[src, M].set((1 - beta) * kappa * alpha)
        tau = tau.at[src, I].set(beta)
        tau = tau.at[src, D].set((1 - beta) * kappa * (1 - alpha))
        tau = tau.at[src, E].set((1 - beta) * one_minus_kappa)

    # D row uses gamma
    tau = tau.at[D, M].set((1 - gamma) * kappa * alpha)
    tau = tau.at[D, I].set(gamma)
    tau = tau.at[D, D].set((1 - gamma) * kappa * (1 - alpha))
    tau = tau.at[D, E].set((1 - gamma) * one_minus_kappa)

    # Apply fragment-extension self-loops on M, I, D
    for src in (M, I, D):
        row = tau[src]
        tau = tau.at[src].set((1 - ext) * row)
        tau = tau.at[src, src].set(ext + (1 - ext) * row[src])

    return tau


# ---------------------------------------------------------------------------
# GGI -> TKF92 closed-form mapping
# ---------------------------------------------------------------------------

def ggi_to_tkf92_at_t(lam0, mu0, x, y, t):
    """Closed-form TKF92(lambda(t), mu(t), r(t)) from GGI params at branch t.

    Uses the leading-order approximations from composition-renormalization.tex:
      r*(0)  = (lam0 y (1-x) + mu0 x (1-y)) / (lam0 (1-x) + mu0 (1-y))
      r_inf  = r*(0) / (2 - r*(0))
      k      = (lam0 + mu0) (2 - r*(0)) / (1 - r*(0))
      r(t)   = r_inf + (r*(0) - r_inf) exp(-k t)
      lam(t) = lam0 / (1 - r(t)),    mu(t) = mu0 / (1 - r(t))

    All inputs scalar (or broadcasting); output is (lam_t, mu_t, r_t).
    """
    # Boundary r*(0)
    num = lam0 * y * (1 - x) + mu0 * x * (1 - y)
    den = lam0 * (1 - y) + mu0 * (1 - x)
    r_boundary = num / jnp.maximum(den, 1e-30)

    # Closed-form fixed point and decay rate
    r_inf = r_boundary / (2 - r_boundary)
    k = (lam0 + mu0) * (2 - r_boundary) / jnp.maximum(1 - r_boundary, 1e-30)

    # r(t)
    r_t = r_inf + (r_boundary - r_inf) * jnp.exp(-k * t)

    # Fixed-rate (lam, mu) per wideboy_to_lambda.md 2026-06-03 — paper's
    # recommended approximation: lam_t, mu_t held constant at their
    # boundary values, only r_t evolves.  Was slaved-rate previously.
    one_minus_r0 = jnp.maximum(1 - r_boundary, 1e-30)
    lam_t = lam0 / one_minus_r0
    mu_t = mu0 / one_minus_r0

    return lam_t, mu_t, r_t


# ---------------------------------------------------------------------------
# Parameter unpacking
# ---------------------------------------------------------------------------

def unpack_ggi_params(log_lam0, logit_x, logit_y):
    """Return (lam0, mu0, x, y) with mu0 derived from GGI reversibility.

    Reversibility:   lam0 * y * (1 - y) = mu0 * x * (1 - x)
    so                mu0 = lam0 * y(1-y) / (x(1-x))

    To keep the system well-defined we require x, y in (0, 1) strictly; the
    logit transforms ensure that.
    """
    lam0 = jnp.exp(log_lam0)
    x = jax.nn.sigmoid(logit_x)
    y = jax.nn.sigmoid(logit_y)
    # Reversibility-derived mu0
    mu0 = lam0 * y * (1 - y) / jnp.maximum(x * (1 - x), 1e-30)
    return lam0, mu0, x, y


# ---------------------------------------------------------------------------
# Composite log-likelihood
# ---------------------------------------------------------------------------

def tkf92_singlet_log_marginal(lam_t, mu_t, r_t, n_cherries, t_anc):
    """Aggregated TKF92 singlet log-marginal of the ancestor for one tau bin.

    TKF92 stationary length distribution is compound geometric:
      L = sum_{i=1..N} X_i,  with  N ~ Geom_0(kappa)   (P(N=n) = (1-kappa) kappa^n, n >= 0)
      and  X_i ~ Geom_1(r)    (P(X=k) = (1-r) r^{k-1}, k >= 1).

    The marginal probability of ancestor length L is:
      P(L=0)   = 1 - kappa
      P(L=ell) = (1-kappa) * kappa(1-r)/r * (r + kappa(1-r))^(ell - 1),  ell >= 1.

    Aggregated over a tau bin with N_cherries cherries and total ancestor
    residues T_anc (assuming all cherries have L >= 1):
      Sum_log_P =  N_cherries * log((1-kappa) * kappa (1-r) / r)
                +  (T_anc - N_cherries) * log(r + kappa (1-r)).

    Args:
      lam_t, mu_t, r_t: TKF92 params at this tau (kappa derived as lam_t/mu_t).
      n_cherries: number of cherries (== sum of E-column counts in this bin).
      t_anc: total ancestor residues = sum over (i, j in {M,D}) of n_{ij}.

    Returns: scalar log-marginal contribution.
    """
    kappa = lam_t / jnp.maximum(mu_t, 1e-30)
    # Stable 1-kappa: subtraction-first; exact 0 at lam_t==mu_t (XLA fusion
    # otherwise drifts to ~1e-8, see fix in tkf92_trans_full 2026-05-29).
    one_m_k = jnp.maximum(
        0.0, (mu_t - lam_t) / jnp.maximum(mu_t, 1e-30))
    k_1mr_over_r = kappa * (1 - r_t) / jnp.maximum(r_t, 1e-30)
    rho = r_t + kappa * (1 - r_t)
    log_first = jnp.log(jnp.maximum(one_m_k, 1e-30)) + jnp.log(jnp.maximum(k_1mr_over_r, 1e-30))
    log_rho = jnp.log(jnp.maximum(rho, 1e-30))
    return n_cherries * log_first + (t_anc - n_cherries) * log_rho


def ggi_stationary_log_marginal(lam0, mu0, x, y, n_cherries_total, t_anc_total):
    """Aggregated GGI stationary log-marginal of the ancestor.

    Plain geometric stationary with parameter
       rho_GGI = lam0 (1 - x) / [mu0 (1 - y)]
    (mean length = rho_GGI / (1 - rho_GGI); requires mu0(1-y) > lam0(1-x), i.e.
    rho_GGI < 1, for stability).  Aggregated over a sample of cherries:

       Sum_log_P = N_cherries_total * log(1 - rho_GGI)
                 + T_anc_total      * log(rho_GGI).

    For rho_GGI >= 1 the stationary does not exist; we add a smooth quadratic
    barrier so the optimiser stays in the stable region.
    """
    rho_raw = lam0 * (1 - x) / jnp.maximum(mu0 * (1 - y), 1e-30)
    # Clamp for the geometric formula
    rho = jnp.minimum(rho_raw, 0.99999)
    log_rho = jnp.log(jnp.maximum(rho, 1e-30))
    log_one_m_rho = jnp.log(1 - rho)
    ll = n_cherries_total * log_one_m_rho + t_anc_total * log_rho
    # Quadratic barrier for rho_raw > 0.99 (strong push back; scale picked to
    # dominate the linear gain from t_anc * log(rho) at Pfam scale).
    barrier = -1e10 * jnp.square(jax.nn.relu(rho_raw - 0.99))
    return ll + barrier


def composite_log_likelihood_3param(log_lam0, logit_x, logit_y,
                                     trans_counts, tau_centers,
                                     conditional_flag, ggi_stat_flag):
    """Composite LL over all tau bins, given GGI (3-free-parameter) params.

    Three modes (set via the two flag arguments):
      (0, 0):  joint LL  =  sum_t  n_{t,i,j}  *  log tau(theta(t), t)[i,j].
        Equivalent to TKF92-style joint LL where the surrogate's stationary
        appears implicitly through the kappa factors in tau(...).
      (1, 0):  conditional LL  =  joint  -  TKF92_singlet_log_marginal(theta(t))
        for each tau bin.  Objective is P(descendant|ancestor) only.
      (1, 1):  GGI-stationary joint  =  conditional + GGI_singlet_log_marginal.
        Apples-to-apples vs the TKF92-only joint fit: both have a stationary
        ancestor distribution + a conditional descendant likelihood; the
        difference is that here the ancestor stationary is GGI's plain
        geometric, not TKF92's compound geometric.

    Args:
      log_lam0, logit_x, logit_y: scalar JAX arrays (free GGI params).
      trans_counts: jax array (n_tau, 5, 5) of TKF92 transition counts.
      tau_centers: jax array (n_tau,) of branch-length bin centers.
      conditional_flag: scalar (0.0 or 1.0); subtract TKF92 ancestor singlet.
      ggi_stat_flag:   scalar (0.0 or 1.0); add the GGI ancestor singlet.

    Returns: scalar LL.
    """
    lam0, mu0, x, y = unpack_ggi_params(log_lam0, logit_x, logit_y)

    # Aggregate ancestor counts (constant inputs, but recomputed for JIT clarity)
    t_anc_total = jnp.sum(trans_counts[:, :, M]) + jnp.sum(trans_counts[:, :, D])
    n_cherries_total = jnp.sum(trans_counts[:, :, E])

    def ll_one_tau(tau_idx):
        t = tau_centers[tau_idx]
        lam_t, mu_t, r_t = ggi_to_tkf92_at_t(lam0, mu0, x, y, t)
        T = tkf92_trans_full(lam_t, mu_t, t, r_t)
        log_T = jnp.log(jnp.maximum(T, 1e-30))
        n_ij = trans_counts[tau_idx]
        joint_ll = jnp.sum(n_ij * log_T)
        t_anc = jnp.sum(n_ij[:, M]) + jnp.sum(n_ij[:, D])
        n_cherries = jnp.sum(n_ij[:, E])
        singlet_lm = tkf92_singlet_log_marginal(lam_t, mu_t, r_t, n_cherries, t_anc)
        return joint_ll - conditional_flag * singlet_lm

    n_tau = tau_centers.shape[0]
    lls = jax.vmap(ll_one_tau)(jnp.arange(n_tau))
    total_ll = jnp.sum(lls)

    # GGI stationary ancestor singlet (time-independent, single scalar)
    ggi_stat = ggi_stationary_log_marginal(
        lam0, mu0, x, y, n_cherries_total, t_anc_total)
    total_ll = total_ll + ggi_stat_flag * ggi_stat

    return total_ll


# JIT-compile the LL & gradient
val_and_grad = jax.jit(jax.value_and_grad(
    composite_log_likelihood_3param, argnums=(0, 1, 2)))


# ---------------------------------------------------------------------------
# I/O: support both Maraschino-style (C_MM, ...) and (n_tau, 5, 5) tensor
# ---------------------------------------------------------------------------

# Mapping from Maraschino transition-name keys to (from_state, to_state)
_MARCH_KEY_TO_IJ = {
    'C_SM': (S, M), 'C_SI': (S, I), 'C_SD': (S, D), 'C_SE': (S, E),
    'C_MM': (M, M), 'C_MI': (M, I), 'C_MD': (M, D), 'C_ME': (M, E),
    'C_IM': (I, M), 'C_II': (I, I), 'C_ID': (I, D), 'C_IE': (I, E),
    'C_DM': (D, M), 'C_DI': (D, I), 'C_DD': (D, D), 'C_DE': (D, E),
}


def load_counts_tensor(counts_path):
    """Load counts file (.npz). Returns (trans_counts, tau_centers).

    Supports:
      - Maraschino format: C_MM, C_MI, ..., possibly with extra AA dims to sum
        over (last 2-4 dims are AA contexts).
      - Direct format: 'transition_counts' (n_tau, 5, 5) tensor.

    Returns:
      trans_counts: numpy array of shape (n_tau, 5, 5), dtype float64.
      tau_centers: numpy array of shape (n_tau,).
    """
    d = np.load(counts_path, allow_pickle=True)
    tau_centers = np.asarray(d['tau_centers'], dtype=np.float64)
    n_tau = len(tau_centers)

    if 'transition_counts' in d.files:
        # Direct (n_tau, 5, 5) tensor
        trans = np.asarray(d['transition_counts'], dtype=np.float64)
        assert trans.shape == (n_tau, 5, 5), \
            f"transition_counts shape {trans.shape} != ({n_tau}, 5, 5)"
        return trans, tau_centers

    # Maraschino style: C_xx keys, possibly with extra AA dimensions
    trans = np.zeros((n_tau, 5, 5), dtype=np.float64)
    for key, (i, j) in _MARCH_KEY_TO_IJ.items():
        if key not in d.files:
            continue
        v = np.asarray(d[key], dtype=np.float64)
        # Sum over all amino-acid dimensions (keep only the tau axis 0)
        if v.ndim > 1:
            v = v.sum(axis=tuple(range(1, v.ndim)))
        assert v.shape == (n_tau,), \
            f"{key} sum-shape {v.shape} != ({n_tau},)"
        trans[:, i, j] = v
    return trans, tau_centers


# ---------------------------------------------------------------------------
# Adam loop
# ---------------------------------------------------------------------------

def fit_ggi(counts_path, n_steps=5000, lr=0.02, save_path=None,
             init_lam0=0.05, init_x=0.5, init_y=0.5, log_interval=500,
             mode='joint'):
    """Fit GGI (lam0, x, y) -- and mu0 by reversibility -- via Adam on the
    composite log-likelihood.

    Args:
      mode: one of:
        'joint'         - joint LL (TKF92(t) stationary * TKF92(t) conditional).
        'conditional'   - conditional LL (subtracts the TKF92 ancestor singlet
                          log-marginal at each tau).
        'ggi_stationary'- conditional + GGI plain-geometric stationary on the
                          ancestor.  Apples-to-apples vs the TKF92-only joint
                          fit: both have a stationary ancestor distribution
                          plus a conditional descendant likelihood, but the
                          stationary here is GGI's (not TKF92's).
    """
    assert mode in ('joint', 'conditional', 'ggi_stationary'), f"unknown mode {mode}"
    conditional = mode in ('conditional', 'ggi_stationary')
    ggi_stat = mode == 'ggi_stationary'
    print(f"Loading counts from {counts_path}...")
    trans_counts_np, tau_centers_np = load_counts_tensor(counts_path)
    total = trans_counts_np.sum()
    print(f"Total transition counts: {total:.0f}")
    print(f"Tau bins: {len(tau_centers_np)}, "
          f"range [{tau_centers_np[0]:.4f}, {tau_centers_np[-1]:.4f}]")

    trans_counts = jnp.asarray(trans_counts_np, dtype=jnp.float32)
    tau_centers = jnp.asarray(tau_centers_np, dtype=jnp.float32)
    conditional_flag = jnp.float32(1.0 if conditional else 0.0)
    ggi_stat_flag = jnp.float32(1.0 if ggi_stat else 0.0)
    if mode == 'joint':
        print("Objective: joint LL  =  log P(ancestor, descendant; TKF92(theta(t))).")
    elif mode == 'conditional':
        print("Objective: conditional LL  =  log P(descendant | ancestor; TKF92(theta(t)))"
              " (subtracts TKF92 stationary at each tau).")
    elif mode == 'ggi_stationary':
        print("Objective: GGI-stationary joint LL  =  log P(ancestor; GGI plain geometric)"
              " + log P(descendant | ancestor; TKF92(theta(t))).  Apples-to-apples vs the"
              " TKF92-only joint fit.")

    # Initial parameters
    log_lam0 = jnp.float32(np.log(init_lam0))
    logit_x = jnp.float32(np.log(init_x / (1 - init_x)))
    logit_y = jnp.float32(np.log(init_y / (1 - init_y)))

    # JIT warmup
    print("JIT compiling...")
    t0 = time.monotonic()
    ll, grads = val_and_grad(log_lam0, logit_x, logit_y,
                              trans_counts, tau_centers,
                              conditional_flag, ggi_stat_flag)
    jax.block_until_ready(ll)
    print(f"JIT done in {time.monotonic()-t0:.1f}s, initial LL = {float(ll):.2f}")

    # Adam optimizer
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    params = [log_lam0, logit_x, logit_y]
    m = [jnp.zeros_like(p) for p in params]
    v = [jnp.zeros_like(p) for p in params]

    best_ll = -float('inf')
    best_params = None

    t0 = time.monotonic()
    for step in range(n_steps):
        ll, grads = val_and_grad(params[0], params[1], params[2],
                                  trans_counts, tau_centers,
                                  conditional_flag, ggi_stat_flag)
        ll_val = float(ll)
        if ll_val > best_ll:
            best_ll = ll_val
            best_params = [float(p) for p in params]

        for i in range(3):
            g = grads[i]
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            params[i] = params[i] + lr * m_hat / (jnp.sqrt(v_hat) + eps)

        if step % log_interval == 0 or step == n_steps - 1:
            lam0_v = float(jnp.exp(params[0]))
            x_v = float(jax.nn.sigmoid(params[1]))
            y_v = float(jax.nn.sigmoid(params[2]))
            mu0_v = lam0_v * y_v * (1 - y_v) / max(x_v * (1 - x_v), 1e-30)
            # Boundary
            num = lam0_v * y_v * (1 - x_v) + mu0_v * x_v * (1 - y_v)
            den = lam0_v * (1 - y_v) + mu0_v * (1 - x_v)
            r_bound = num / max(den, 1e-30)
            elapsed = time.monotonic() - t0
            print(f"Step {step:5d}: LL={ll_val:15.2f}  "
                  f"lam0={lam0_v:.5f}  mu0={mu0_v:.5f}  x={x_v:.4f}  y={y_v:.4f}  "
                  f"r*(0)={r_bound:.4f}  ({elapsed:.1f}s)")

    # Best params
    lam0 = float(np.exp(best_params[0]))
    x = float(1 / (1 + np.exp(-best_params[1])))
    y = float(1 / (1 + np.exp(-best_params[2])))
    mu0 = lam0 * y * (1 - y) / max(x * (1 - x), 1e-30)

    # Recompute summary quantities
    num = lam0 * y * (1 - x) + mu0 * x * (1 - y)
    den = lam0 * (1 - y) + mu0 * (1 - x)
    r_boundary = num / max(den, 1e-30)
    r_inf = r_boundary / (2 - r_boundary)
    k = (lam0 + mu0) * (2 - r_boundary) / max(1 - r_boundary, 1e-30)
    lam_T_boundary = lam0 / max(1 - r_boundary, 1e-30)
    mu_T_boundary = mu0 / max(1 - r_boundary, 1e-30)

    print(f"\n{'='*70}")
    print(f"FITTED GGI PARAMETERS (CherryML composite likelihood)")
    print(f"{'='*70}")
    print(f"  lambda_0 (GGI insertion rate per link)  = {lam0:.6f}")
    print(f"  mu_0     (GGI deletion rate per residue) = {mu0:.6f}")
    print(f"  x        (GGI deletion length geom)     = {x:.6f}")
    print(f"  y        (GGI insertion length geom)    = {y:.6f}")
    print(f"  Reversibility check: lam0*y(1-y) = {lam0*y*(1-y):.6e}, "
          f"mu0*x(1-x) = {mu0*x*(1-x):.6e}")
    print(f"")
    print(f"  Derived TKF92 boundary:")
    print(f"    r*(0)        = {r_boundary:.6f}")
    print(f"    lambda*(0)   = {lam_T_boundary:.6f}")
    print(f"    mu*(0)       = {mu_T_boundary:.6f}")
    print(f"  Closed-form trajectory:")
    print(f"    r_inf        = {r_inf:.6f}")
    print(f"    decay k      = {k:.6f}  (half-life {np.log(2)/max(k,1e-30):.2f})")
    print(f"  Best LL        = {best_ll:.2f}")
    print(f"{'='*70}")

    # Tabulate trajectory at representative branch lengths
    print(f"\nTrajectory at representative branch lengths:")
    print(f"  {'t':>7}  {'lam(t)':>10}  {'mu(t)':>10}  {'r(t)':>9}")
    for t_val in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        r_t = r_inf + (r_boundary - r_inf) * np.exp(-k * t_val)
        # Fixed-rate (paper's recommended surrogate form) — lam_t, mu_t
        # constant at their boundary values; only r_t evolves.
        lam_t = lam0 / max(1 - r_boundary, 1e-30)
        mu_t = mu0 / max(1 - r_boundary, 1e-30)
        print(f"  {t_val:>7.3f}  {lam_t:>10.6f}  {mu_t:>10.6f}  {r_t:>9.4f}")

    if save_path:
        save_dict = dict(
            lambda_0=lam0, mu_0=mu0, x=x, y=y,
            r_boundary=float(r_boundary),
            lambda_T_boundary=float(lam_T_boundary),
            mu_T_boundary=float(mu_T_boundary),
            r_inf=float(r_inf), decay_k=float(k),
            best_ll=float(best_ll),
            n_steps=n_steps,
            counts_path=str(counts_path),
        )
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        print(f"\nSaved GGI fit -> {save_path}")

    return dict(
        lambda_0=lam0, mu_0=mu0, x=x, y=y,
        r_boundary=float(r_boundary),
        r_inf=float(r_inf), decay_k=float(k),
        best_ll=float(best_ll),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--counts', default='../data/seed_counts.npz',
                        help='Path to counts .npz (Maraschino or direct format)')
    parser.add_argument('--out', default=None,
                        help='Path to save JSON of fitted params (optional)')
    parser.add_argument('--n-steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--init-lam0', type=float, default=0.05)
    parser.add_argument('--init-x', type=float, default=0.5)
    parser.add_argument('--init-y', type=float, default=0.5)
    parser.add_argument('--log-interval', type=int, default=500)
    parser.add_argument('--mode', default='joint',
                        choices=('joint', 'conditional', 'ggi_stationary'),
                        help="Objective: 'joint' (default), 'conditional' (subtract TKF92"
                             " ancestor singlet log-marginal at each tau), or"
                             " 'ggi_stationary' (conditional + add the GGI plain-geometric"
                             " stationary on the ancestor; apples-to-apples vs the TKF92-only"
                             " joint fit).")
    parser.add_argument('--conditional', action='store_true',
                        help='Alias for --mode conditional (kept for backward compat).')
    args = parser.parse_args()

    mode = args.mode
    if args.conditional and mode == 'joint':
        mode = 'conditional'

    fit_ggi(
        args.counts,
        n_steps=args.n_steps, lr=args.lr,
        save_path=args.out,
        init_lam0=args.init_lam0,
        init_x=args.init_x, init_y=args.init_y,
        log_interval=args.log_interval,
        mode=mode,
    )
