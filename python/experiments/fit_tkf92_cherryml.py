#!/usr/bin/env python3
"""Fit TKF92 parameters (λ, μ, r) from CherryML cherry counts tensor.

Uses the same seed_counts.npz that maraschino uses, but fits a bare TKF92
model (no domains, no mixture) via composite likelihood over discretized
branch-length bins.

The TKF92 transition matrix is 5×5: S, M, I, D, E.
Fragment extension parameter r creates self-loops on M, I, D.

Usage:
    cd python && uv run python experiments/fit_tkf92_cherryml.py

Outputs fitted (λ, μ, r) and optionally saves to a JSON file.
"""

import sys
import time
import json

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, '.')
from tkfmixdom.jax.core.bdi import tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa


# State indices
S, M, I, D, E = 0, 1, 2, 3, 4


def tkf92_trans_full(ins_rate, del_rate, t, ext):
    """Build 5×5 TKF92 Pair HMM transition matrix (rows sum to 1)."""
    alpha = tkf_alpha(del_rate, t)
    beta = tkf_beta(ins_rate, del_rate, t)
    gamma = tkf_gamma(ins_rate, del_rate, t)
    kappa = tkf_kappa(ins_rate, del_rate)

    tau = jnp.zeros((5, 5))

    for src in [S, M, I]:
        tau = tau.at[src, M].set((1 - beta) * kappa * alpha)
        tau = tau.at[src, I].set(beta)
        tau = tau.at[src, D].set((1 - beta) * kappa * (1 - alpha))
        tau = tau.at[src, E].set((1 - beta) * (1 - kappa))

    tau = tau.at[D, M].set((1 - gamma) * kappa * alpha)
    tau = tau.at[D, I].set(gamma)
    tau = tau.at[D, D].set((1 - gamma) * kappa * (1 - alpha))
    tau = tau.at[D, E].set((1 - gamma) * (1 - kappa))

    # Apply fragment extension self-loops
    for src in [M, I, D]:
        row = tau[src]
        tau = tau.at[src].set((1 - ext) * row)
        tau = tau.at[src, src].set(ext + (1 - ext) * row[src])

    return tau


def load_gap_counts(counts_path):
    """Load counts and marginalize over amino acids to get gap transition counts.

    Returns dict of {transition_name: array of shape (n_tau,)} and tau_centers.
    """
    d = np.load(counts_path, allow_pickle=True)
    tau_centers = d['tau_centers']

    gap_counts = {}
    for k in ['C_MM', 'C_MI', 'C_MD', 'C_IM', 'C_II', 'C_ID',
              'C_DM', 'C_DD', 'C_DI', 'C_SM', 'C_SI', 'C_SD',
              'C_ME', 'C_IE', 'C_DE', 'C_SE']:
        v = d[k]
        # Sum over all amino acid dimensions (keep only tau bin axis 0)
        gap_counts[k] = v.sum(axis=tuple(range(1, v.ndim))) if v.ndim > 1 else v

    return gap_counts, tau_centers


@jax.jit
def composite_log_likelihood(log_del, logit_kappa, logit_ext, gap_counts_jax, tau_centers):
    """Composite log-likelihood of TKF92 from cherry gap counts.

    Parameters in unconstrained space:
        log_del: log(μ)
        logit_kappa: logit(κ) where κ = λ/μ ∈ (0, 1)
        logit_ext: logit(r) = log(r/(1-r))
    """
    del_rate = jnp.exp(log_del)
    kappa = jax.nn.sigmoid(logit_kappa)
    ins_rate = kappa * del_rate
    ext = jax.nn.sigmoid(logit_ext)

    def ll_one_tau(tau_idx):
        t = tau_centers[tau_idx]
        T = tkf92_trans_full(ins_rate, del_rate, t, ext)
        log_T = jnp.log(jnp.maximum(T, 1e-30))

        ll = 0.0
        # Post-S transitions
        ll += gap_counts_jax['C_SM'][tau_idx] * log_T[S, M]
        ll += gap_counts_jax['C_SI'][tau_idx] * log_T[S, I]
        ll += gap_counts_jax['C_SD'][tau_idx] * log_T[S, D]
        ll += gap_counts_jax['C_SE'][tau_idx] * log_T[S, E]

        # Post-M transitions
        ll += gap_counts_jax['C_MM'][tau_idx] * log_T[M, M]
        ll += gap_counts_jax['C_MI'][tau_idx] * log_T[M, I]
        ll += gap_counts_jax['C_MD'][tau_idx] * log_T[M, D]
        ll += gap_counts_jax['C_ME'][tau_idx] * log_T[M, E]

        # Post-I transitions
        ll += gap_counts_jax['C_IM'][tau_idx] * log_T[I, M]
        ll += gap_counts_jax['C_II'][tau_idx] * log_T[I, I]
        ll += gap_counts_jax['C_ID'][tau_idx] * log_T[I, D]
        ll += gap_counts_jax['C_IE'][tau_idx] * log_T[I, E]

        # Post-D transitions
        ll += gap_counts_jax['C_DM'][tau_idx] * log_T[D, M]
        ll += gap_counts_jax['C_DD'][tau_idx] * log_T[D, D]
        ll += gap_counts_jax['C_DI'][tau_idx] * log_T[D, I]
        ll += gap_counts_jax['C_DE'][tau_idx] * log_T[D, E]

        return ll

    n_tau = tau_centers.shape[0]
    lls = jax.vmap(ll_one_tau)(jnp.arange(n_tau))
    return jnp.sum(lls)


def fit_tkf92(counts_path='../data/seed_counts.npz', n_steps=5000, lr=0.01,
              save_path=None):
    """Fit TKF92 (λ, μ, r) via Adam on CherryML composite likelihood."""
    print(f"Loading counts from {counts_path}...")
    gap_counts, tau_centers = load_gap_counts(counts_path)

    # Print summary
    total = sum(v.sum() for v in gap_counts.values())
    print(f"Total gap transition counts: {total:.0f}")
    print(f"Tau bins: {len(tau_centers)}, range [{tau_centers[0]:.4f}, {tau_centers[-1]:.4f}]")

    # Convert to JAX
    gap_counts_jax = {k: jnp.array(v, dtype=jnp.float32) for k, v in gap_counts.items()}
    tau_centers_jax = jnp.array(tau_centers, dtype=jnp.float32)

    # Initialize: reasonable protein evolution rates
    # μ ≈ 0.06, κ = λ/μ ≈ 0.85, r ≈ 0.5
    log_del = jnp.float32(jnp.log(0.06))
    logit_kappa = jnp.float32(1.7)  # sigmoid(1.7) ≈ 0.845
    logit_ext = jnp.float32(0.0)    # sigmoid(0) = 0.5

    val_and_grad = jax.value_and_grad(composite_log_likelihood, argnums=(0, 1, 2))

    # JIT warmup
    print("JIT compiling...")
    t0 = time.monotonic()
    ll, grads = val_and_grad(log_del, logit_kappa, logit_ext,
                             gap_counts_jax, tau_centers_jax)
    jax.block_until_ready(ll)
    print(f"JIT done in {time.monotonic() - t0:.1f}s, initial LL = {float(ll):.2f}")

    # Adam optimizer
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = [jnp.zeros_like(log_del)] * 3
    v = [jnp.zeros_like(log_del)] * 3

    params = [log_del, logit_kappa, logit_ext]
    best_ll = float('-inf')
    best_params = None

    t0 = time.monotonic()
    for step in range(n_steps):
        ll, grads = val_and_grad(params[0], params[1], params[2],
                                 gap_counts_jax, tau_centers_jax)
        ll_val = float(ll)

        if ll_val > best_ll:
            best_ll = ll_val
            best_params = [float(p) for p in params]

        for i in range(3):
            g = grads[i]
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            params[i] = params[i] + lr * m_hat / (jnp.sqrt(v_hat) + eps)

        if step % 500 == 0 or step == n_steps - 1:
            dl = float(jnp.exp(params[0]))
            kappa = float(jax.nn.sigmoid(params[1]))
            ins = kappa * dl
            ext = float(jax.nn.sigmoid(params[2]))
            elapsed = time.monotonic() - t0
            print(f"Step {step:5d}: LL={ll_val:15.2f}  λ={ins:.6f}  μ={dl:.6f}  "
                  f"κ={kappa:.4f}  r={ext:.4f}  ({elapsed:.1f}s)")

    # Best params
    del_rate = float(np.exp(best_params[0]))
    kappa = float(1 / (1 + np.exp(-best_params[1])))
    ins_rate = kappa * del_rate
    ext_rate = float(1 / (1 + np.exp(-best_params[2])))

    print(f"\n{'='*60}")
    print(f"FITTED TKF92 PARAMETERS (CherryML composite likelihood)")
    print(f"{'='*60}")
    print(f"  λ (insertion rate) = {ins_rate:.6f}")
    print(f"  μ (deletion rate)  = {del_rate:.6f}")
    print(f"  κ = λ/μ            = {kappa:.6f}")
    print(f"  r (fragment ext)   = {ext_rate:.6f}")
    print(f"  Best LL            = {best_ll:.2f}")
    print(f"{'='*60}")

    # Print transition matrix at a few representative branch lengths
    for t_val in [0.1, 0.5, 1.0, 2.0]:
        T = tkf92_trans_full(ins_rate, del_rate, t_val, ext_rate)
        T_np = np.array(T)
        print(f"\nTKF92 transition matrix at t={t_val}:")
        print(f"  P(M|M)={T_np[M,M]:.4f}  P(I|M)={T_np[M,I]:.4f}  "
              f"P(D|M)={T_np[M,D]:.4f}  P(E|M)={T_np[M,E]:.4f}")
        print(f"  P(M|I)={T_np[I,M]:.4f}  P(I|I)={T_np[I,I]:.4f}  "
              f"P(D|I)={T_np[I,D]:.4f}  P(E|I)={T_np[I,E]:.4f}")
        print(f"  P(M|D)={T_np[D,M]:.4f}  P(I|D)={T_np[D,I]:.4f}  "
              f"P(D|D)={T_np[D,D]:.4f}  P(E|D)={T_np[D,E]:.4f}")

    result = {
        'ins_rate': ins_rate,
        'del_rate': del_rate,
        'ext_rate': ext_rate,
        'kappa': kappa,
        'best_ll': best_ll,
        'counts_path': counts_path,
        'n_steps': n_steps,
    }

    if save_path is None:
        save_path = 'experiments/tkf92_fitted_params.json'
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {save_path}")

    return result


if __name__ == '__main__':
    fit_tkf92()
