"""TKF91/TKF92/MixDom parameter computations.

All functions are JAX-compatible (differentiable, jit-compilable).

BDI parameter functions (tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa),
score derivatives, and related functions are defined in indel.py and
re-exported here for backward compatibility.
"""

import jax.numpy as jnp

from .bdi import (
    tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa,
    score_derivatives, transition_count_groups,
    EQUAL_RATE_THRESHOLD,
)


# --- State constants ---
# State order: S=0, M=1, I=2, D=3, E=4

S, M, I, D, E = 0, 1, 2, 3, 4
STATE_NAMES = ['S', 'M', 'I', 'D', 'E']
N_STATES = 5


# --- TKF91 Pair HMM transition matrix ---

def tkf91_trans(ins_rate, del_rate, t):
    """Build 5x5 TKF91 Pair HMM transition matrix.

    Row/col order: S, M, I, D, E.
    """
    alpha = tkf_alpha(del_rate, t)
    beta = tkf_beta(ins_rate, del_rate, t)
    gamma = tkf_gamma(ins_rate, del_rate, t)
    kappa = tkf_kappa(ins_rate, del_rate)

    tau = jnp.zeros((5, 5))

    # S, M, I rows use beta
    for src in [S, M, I]:
        tau = tau.at[src, M].set((1 - beta) * kappa * alpha)
        tau = tau.at[src, I].set(beta)
        tau = tau.at[src, D].set((1 - beta) * kappa * (1 - alpha))
        tau = tau.at[src, E].set((1 - beta) * (1 - kappa))

    # D row uses gamma
    tau = tau.at[D, M].set((1 - gamma) * kappa * alpha)
    tau = tau.at[D, I].set(gamma)
    tau = tau.at[D, D].set((1 - gamma) * kappa * (1 - alpha))
    tau = tau.at[D, E].set((1 - gamma) * (1 - kappa))

    return tau


def tkf91_trans_cond(ins_rate, del_rate, t):
    """Build 5x5 TKF91 transition matrix conditioned on ancestor length.

    Removes κ^i·(1-κ) geometric prior factors from the transition matrix:
    - κ factor removed from M/D entries (ancestor position transitions)
    - (1-κ) factor removed from E entries (geometric normalization)

    The resulting matrix is unnormalized (rows don't sum to 1) but gives
    the same FB posteriors as the full matrix, and the DP computes
    log P(x,y | |ancestor|=i) + const instead of log P(x,y).

    Use when κ ≈ 1 (λ ≈ μ) to avoid numerical issues with log(1-κ).
    """
    alpha = tkf_alpha(del_rate, t)
    beta = tkf_beta(ins_rate, del_rate, t)
    gamma = tkf_gamma(ins_rate, del_rate, t)

    tau = jnp.zeros((5, 5))

    # S, M, I rows: κ removed from M/D, (1-κ) removed from E
    for src in [S, M, I]:
        tau = tau.at[src, M].set((1 - beta) * alpha)
        tau = tau.at[src, I].set(beta)
        tau = tau.at[src, D].set((1 - beta) * (1 - alpha))
        tau = tau.at[src, E].set((1 - beta))

    # D row: same treatment with gamma
    tau = tau.at[D, M].set((1 - gamma) * alpha)
    tau = tau.at[D, I].set(gamma)
    tau = tau.at[D, D].set((1 - gamma) * (1 - alpha))
    tau = tau.at[D, E].set((1 - gamma))

    return tau


def tkf92_trans(ins_rate, del_rate, t, ext):
    """Build 5x5 TKF92 Pair HMM transition matrix.

    Like TKF91 but M, I, D have fragment extension self-loops with probability ext.
    """
    tau91 = tkf91_trans(ins_rate, del_rate, t)
    tau = jnp.zeros((5, 5))

    # S row: same as TKF91 (first character of first fragment)
    tau = tau.at[S].set(tau91[S])

    # M, I, D rows: self-loop = ext + (1-ext)*tau91[a,a], others = (1-ext)*tau91[a,b]
    for src in [M, I, D]:
        tau = tau.at[src].set((1 - ext) * tau91[src])
        tau = tau.at[src, src].set(ext + (1 - ext) * tau91[src, src])

    return tau


