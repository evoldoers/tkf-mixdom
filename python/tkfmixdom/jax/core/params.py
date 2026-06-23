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


# --- MixFrag (TKF92 with fragment mixtures) transition matrices ---
#
# MixFrag promotes the TKF92 fragment-extension parameter to a per-fragment
# categorical latent variable: when a fragment is created it draws a fragtype
# f ~ Categorical(weights), then extends geometrically with fragtype-specific
# probability exts[f].  Substitution and indel (BDI) processes are shared
# across fragtypes; the only new parameters are exts (r_f) and weights (w_f).
# See the supplement section "The TKF92 Model with Fragment Mixtures (MixFrag)"
# (tkf/mixfrag.tex) for the derivation.  Reduces to TKF92 at F=1, weights=[1].
#
# Construction from the TKF91 matrix tau (identical rule for Pair and Singlet):
#   * a transition OUT of an emitting state of fragtype f  -> x (1 - exts[f])
#   * a transition INTO an emitting state of fragtype g     -> x weights[g]
#   * an added self-loop exts[f] on the same state AND same fragtype.


def mixfrag_pair_index(F):
    """Base indices (S, M0, I0, D0, E) for the (3F+2)-state MixFrag Pair HMM.

    State order: S, M_1..M_F, I_1..I_F, D_1..D_F, E, with M_f at index M0+f
    (f = 0..F-1), I_f at I0+f, D_f at D0+f.
    """
    return 0, 1, 1 + F, 1 + 2 * F, 3 * F + 1


def mixfrag_trans(ins_rate, del_rate, t, exts, weights):
    """Build the (3F+2, 3F+2) MixFrag joint Pair HMM transition matrix.

    Args:
        ins_rate, del_rate, t: TKF91 indel rates and branch length.
        exts:    (F,) fragtype extension probabilities r_f in [0, 1).
        weights: (F,) fragtype weights w_f (>= 0, sum to 1).

    Returns:
        (3F+2, 3F+2) transition matrix; state order
        S, M_1..M_F, I_1..I_F, D_1..D_F, E.

    Reduces to ``tkf92_trans(ins_rate, del_rate, t, ext)`` when F=1,
    exts=[ext], weights=[1].  Each emitting row sums to 1 (the joint TKF91
    matrix is row-stochastic and the w_g collapse via sum_g w_g = 1).
    """
    exts = jnp.asarray(exts)
    weights = jnp.asarray(weights)
    tau91 = tkf91_trans(ins_rate, del_rate, t)            # 5x5 TKF91 base
    F = int(exts.shape[0])
    Sx, M0, I0, D0, Ex = mixfrag_pair_index(F)
    n = 3 * F + 2
    blocks = ((M0, M), (I0, I), (D0, D))                  # (block base, TKF91 type)

    tau = jnp.zeros((n, n))
    # S row: S is non-emitting (no (1 - ext) factor, no self-loop).
    for (db, dty) in blocks:
        for g in range(F):
            tau = tau.at[Sx, db + g].set(weights[g] * tau91[S, dty])
    tau = tau.at[Sx, Ex].set(tau91[S, E])

    # Emitting rows M_f, I_f, D_f.
    for (sb, sty) in blocks:
        for f in range(F):
            src = sb + f
            for (db, dty) in blocks:
                for g in range(F):
                    val = (1.0 - exts[f]) * weights[g] * tau91[sty, dty]
                    if sty == dty and f == g:             # same state, same fragtype
                        val = val + exts[f]               # fragment-extension self-loop
                    tau = tau.at[src, db + g].set(val)
            tau = tau.at[src, Ex].set((1.0 - exts[f]) * tau91[sty, E])
    # E row is absorbing (all zeros).
    return tau


def mixfrag_singlet_trans(ins_rate, del_rate, exts, weights):
    """Build the (F+2, F+2) MixFrag Singlet (stationary) HMM transition matrix.

    State order S, I_1..I_F, E.  Built from the TKF91 stationary HMM (emit
    n ~ Geom(kappa), kappa = ins_rate/del_rate) by the same fragtype rule as
    the Pair HMM.  Each I_f row sums to 1.
    """
    exts = jnp.asarray(exts)
    weights = jnp.asarray(weights)
    kappa = tkf_kappa(ins_rate, del_rate)
    F = int(exts.shape[0])
    n = F + 2
    Sx, I0, Ex = 0, 1, F + 1
    tau = jnp.zeros((n, n))
    # S row.
    for g in range(F):
        tau = tau.at[Sx, I0 + g].set(kappa * weights[g])
    tau = tau.at[Sx, Ex].set(1.0 - kappa)
    # I_f rows.
    for f in range(F):
        src = I0 + f
        for g in range(F):
            val = (1.0 - exts[f]) * kappa * weights[g]
            if f == g:
                val = val + exts[f]
            tau = tau.at[src, I0 + g].set(val)
        tau = tau.at[src, Ex].set((1.0 - exts[f]) * (1.0 - kappa))
    return tau


