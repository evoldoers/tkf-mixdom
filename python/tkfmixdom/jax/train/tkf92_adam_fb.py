"""Adam on the 2D pair-HMM forward-backward log-likelihood for plain TKF92
and GGI-steered TKF92.

Counterpart to ``tkf92_svi_bw.py`` (analytic gradients via Holmes-Rubin score
identity).  This module differentiates ``forward_backward_2d``'s log_prob
directly via autograd, which:

* admits any smooth re-parameterization of (lam, mu, ext) as a function of
  the per-pair branch length t (the GGI-steered case below), and
* never reads alignments — the gradient comes from the marginalized
  posterior over alignments, so heavily-substituted "false-gap" stretches
  do not bias the indel-rate gradient the way a fixed CherryML alignment
  can.

Two model modes are supported via the ``ParamPack`` interface:

* ``"tkf92"``: free (log_lam, log_mu, logit_ext); constant across all t.
* ``"ggi"``:   free (log_lam0, logit_x, logit_y); per-pair
              (lam(t), mu(t), r(t)) derived via the closed-form GGI flow
              ``ggi_to_tkf92_at_t``.

Memory note: 2D F-B activations are remat'd via ``@jax.checkpoint`` on
``tkf92_log_prob_fb`` so the backward pass recomputes the lattice rather
than storing it.  Combined with bin-bucketed minibatching, this scales
well past the historical 5k-pair ceiling on 11 GB GPUs.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, Optional

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Core: log P(x, y | TKF92(lam, mu, ext), Q, t) via 2D F-B, autodiff-friendly
# ---------------------------------------------------------------------------


def _tkf92_log_trans(lam, mu, ext, t, pi, A):
    """Build the TKF92 5-state log-transition matrix as a JAX array.

    Pure-JAX reimplementation of ``make_tkf92_pair_hmm`` that avoids the
    numpy/jax-array casts in the cherry-counts SVI-BW path so the result
    is differentiable w.r.t. (lam, mu, ext, t).
    """
    from ..core.bdi import tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa

    alpha = tkf_alpha(mu, t)
    beta = tkf_beta(lam, mu, t)
    gamma = tkf_gamma(lam, mu, t)
    kappa = tkf_kappa(lam, mu)
    one_m_k = jnp.maximum(0.0, (mu - lam) / jnp.maximum(mu, 1e-30))
    # State indices: S, M, I, D, E
    S, M, I, D, E = 0, 1, 2, 3, 4
    T = jnp.zeros((5, 5))
    for src in (S, M, I):
        T = T.at[src, M].set((1 - beta) * kappa * alpha)
        T = T.at[src, I].set(beta)
        T = T.at[src, D].set((1 - beta) * kappa * (1 - alpha))
        T = T.at[src, E].set((1 - beta) * one_m_k)
    T = T.at[D, M].set((1 - gamma) * kappa * alpha)
    T = T.at[D, I].set(gamma)
    T = T.at[D, D].set((1 - gamma) * kappa * (1 - alpha))
    T = T.at[D, E].set((1 - gamma) * one_m_k)
    # Fragment-extension self-loops
    for src in (M, I, D):
        row = T[src]
        T = T.at[src].set((1 - ext) * row)
        T = T.at[src, src].set(ext + (1 - ext) * row[src])
    return jnp.log(jnp.maximum(T, 1e-30))


@jax.checkpoint
def tkf92_log_prob_fb(lam, mu, ext, t, Q, pi, x_seq, y_seq,
                      real_Lx=None, real_Ly=None):
    """log P(x_seq, y_seq | TKF92(lam, mu, ext), Q, pi, t) via 2D F-B.

    Autograd-friendly w.r.t. (lam, mu, ext, t).  ``@jax.checkpoint`` remats
    the (Lx · Ly · 5) F-B lattice so the backward pass recomputes it instead
    of storing it.

    If real_Lx / real_Ly are given, x_seq / y_seq are assumed to be
    pre-padded (e.g. by `_stack_bucket` for vmap'd batching) and the F-B
    masks emissions outside the real region to NEG_INF and reads log_p
    at the (real_Lx, real_Ly) endpoint.  Without these args, the F-B
    treats x_seq / y_seq as unpadded — using their .shape[0] as the
    real length — which is correct ONLY when the caller passes
    unpadded sequences.  Failure to thread real_Lx / real_Ly through a
    pre-padded batched call quietly silently computes log P over the
    PADDED sequences (padding bytes are residue 0 of the alphabet),
    adding spurious −10–60 nats/pair to the loss; see commit log.
    """
    from ..core.ctmc import transition_matrix
    from ..dp.hmm import forward_backward_2d
    log_trans = _tkf92_log_trans(lam, mu, ext, t, pi, Q.shape[0])
    state_types = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)  # S, M, I, D, E
    sub = transition_matrix(Q, t)
    log_p, _, _ = forward_backward_2d(
        log_trans, state_types, x_seq, y_seq, sub, pi,
        real_Lx=real_Lx, real_Ly=real_Ly)
    return log_p


def _log_p_tkf92_stationary(kappa, r, L):
    """log P(L_anc=L) under TKF92 compound geometric stationary.

    P(L=0)   = 1 - kappa
    P(L>=1) = (1-kappa) * kappa(1-r)/r * (r + kappa(1-r))^(L-1)
    Stable subtraction-first 1-kappa.
    """
    one_m_k = jnp.maximum(0.0, 1 - kappa)
    log_one_m_k = jnp.log(jnp.maximum(one_m_k, 1e-30))
    k_1mr_over_r = kappa * (1 - r) / jnp.maximum(r, 1e-30)
    log_term2 = jnp.log(jnp.maximum(k_1mr_over_r, 1e-30))
    rho_int = r + kappa * (1 - r)
    log_rho = jnp.log(jnp.maximum(rho_int, 1e-30))
    log_p_pos = log_one_m_k + log_term2 + (L - 1) * log_rho
    return jnp.where(L > 0, log_p_pos, log_one_m_k)


def _log_p_ggi_stationary(lam0, mu0, x, y, L):
    """log P(L_anc=L) under the GGI native geometric stationary:
        rho_GGI = lam0(1-x) / [mu0(1-y)]
        P(L) = (1-rho_GGI) * rho_GGI^L
    Barrier-clamped to rho_GGI in (1e-30, 1-1e-9); if the optimizer wanders
    into rho_GGI >= 1 (model non-stationary), the barrier will push back.
    """
    rho = lam0 * (1 - x) / jnp.maximum(mu0 * (1 - y), 1e-30)
    rho = jnp.minimum(jnp.maximum(rho, 1e-30), 1.0 - 1e-9)
    return jnp.log(1 - rho) + L * jnp.log(rho)


def ggi_steered_log_prob_fb(lam0, mu0, x_geom, y_geom, t, Q, pi,
                              x_seq, y_seq, Lx_real, Ly_real=None,
                              prior_swap=True):
    """log P(anc, des) under GGI-steered TKF92 with GGI-native ancestor prior.

    The objective for GGI-steered model fitting is
        P_GGI(anc) * P_steeredTKF92(des | anc)
    which equals
        P_TKF92_joint(anc, des; steered) - P_TKF92_stationary(anc; steered)
                                          + P_GGI_stationary(anc)
    in log-space.  Subtracting the TKF92-at-steered marginal cancels the
    ancestor-prior factor that forward_backward_2d implicitly includes,
    and re-adds the model-appropriate GGI native geometric prior.

    Lx_real is the UNPADDED ancestor length (the actual cherry's |x|);
    x_seq, y_seq are bin-padded so the F-B can be batched per shape.
    """
    # Closed-form GGI flow (same algebra as
    # experiments/fit_ggi_cherryml.ggi_to_tkf92_at_t):
    num = lam0 * y_geom * (1 - x_geom) + mu0 * x_geom * (1 - y_geom)
    den = lam0 * (1 - y_geom) + mu0 * (1 - x_geom)
    r_boundary = num / jnp.maximum(den, 1e-30)
    r_inf = r_boundary / (2 - r_boundary)
    k = (lam0 + mu0) * (2 - r_boundary) / jnp.maximum(1 - r_boundary, 1e-30)
    r_t = r_inf + (r_boundary - r_inf) * jnp.exp(-k * t)
    # Fixed-rate flow (paper's recommended approximation; see
    # composition-renormalization.tex eq:r-closedform-tkf and
    # eq:frozen-rate-r): hold (lam, mu) at their boundary values, only
    # r(t) evolves.  Per wideboy_to_lambda.md 2026-06-03.
    one_minus_r0 = jnp.maximum(1 - r_boundary, 1e-30)
    lam_t = lam0 / one_minus_r0
    mu_t = mu0 / one_minus_r0

    # 2D F-B joint LL using TKF92 dynamics with steered (lam_t, mu_t, r_t).
    # Lx_real / Ly_real thread through to mask padded positions.
    joint_log_p = tkf92_log_prob_fb(
        lam_t, mu_t, r_t, t, Q, pi, x_seq, y_seq,
        real_Lx=Lx_real, real_Ly=Ly_real)

    if not prior_swap:
        # GGI-flowed dynamics with TKF92's OWN ancestor prior (no swap).
        # Isolates "do the GGI transition dynamics fit data?" from
        # "does the GGI native geometric prior fit data?".
        return joint_log_p

    # Swap the implicit TKF92 ancestor prior for the GGI native prior.
    kappa_t = lam_t / jnp.maximum(mu_t, 1e-30)
    log_p_tkf_anc = _log_p_tkf92_stationary(kappa_t, r_t, Lx_real)
    log_p_ggi_anc = _log_p_ggi_stationary(lam0, mu0, x_geom, y_geom, Lx_real)
    return joint_log_p - log_p_tkf_anc + log_p_ggi_anc


# ---------------------------------------------------------------------------
# Parameter packing / unpacking for Adam
# ---------------------------------------------------------------------------


def unpack_tkf92(params):
    """(log_mu, logit_kappa, logit_ext) -> (lam, mu, ext) with λ < μ strict.

    Pure-TKF92 requires λ < μ for the stationary distribution to exist
    (compound-geometric length with success probability 1-κ).  Independent
    (log_lam, log_mu) lets Adam walk into κ ≥ 1, hitting the (1-κ)=0 wall
    in tkf92_trans_full and NaN-poisoning the gradient via safe_log(0).
    Sigmoid-parameterised κ keeps the optimizer strictly inside (0, 1).
    """
    log_mu, logit_kappa, logit_ext = params
    mu = jnp.exp(log_mu)
    kappa = jax.nn.sigmoid(logit_kappa)
    lam = kappa * mu
    return lam, mu, jax.nn.sigmoid(logit_ext)


def unpack_ggi(params, segment="lower"):
    """(log_mu0, logit_rho, logit_x_raw) -> (lam0, mu0, x, y) with the
    symmetric root of y matching the chosen x-segment.

    Feasibility for the reversibility equation y(1-y) = x(1-x)/ρ requires
    x(1-x) ≤ ρ/4, i.e., x ∈ (0, x_min) ∪ (1-x_min, 1) where
    x_min = (1 - √(1-ρ))/2.  The two segments are DISJOINT (no smooth
    interpolation), so `segment` is a structural choice fixed per run:

      segment="lower":  x ∈ (0, x_min),       y ∈ (0, ½)  (lower-lower)
      segment="upper":  x ∈ (1-x_min, 1),     y ∈ (½, 1)  (upper-upper)

    Symmetric root convention: both x and y are on the SAME side of ½.
    At ρ → 1, x_min → ½ and y = x exactly (the mirror line x = y).

    The lower segment is the historical default (small indel/extension
    rates).  The upper segment is needed when r_boundary (the GGI flow's
    initial extension probability at t→0) must exceed ~0.4 — e.g., when
    warm-starting at a TKF92 optimum with ext > x_min.
    """
    log_mu0, logit_rho, logit_x_raw = params
    mu0 = jnp.exp(log_mu0)
    rho = jax.nn.sigmoid(logit_rho)
    lam0 = rho * mu0
    x_min = (1.0 - jnp.sqrt(jnp.maximum(1.0 - rho, 0.0))) / 2.0
    raw_x = jax.nn.sigmoid(logit_x_raw)
    if segment == "lower":
        xv = raw_x * x_min  # x ∈ (0, x_min)
    elif segment == "upper":
        xv = 1.0 - raw_x * x_min  # x ∈ (1-x_min, 1)
    else:
        raise ValueError(f"unknown segment {segment!r}; "
                          "use 'lower' or 'upper'")
    q = xv * (1.0 - xv) / jnp.maximum(rho, 1e-30)
    disc = jnp.maximum(1.0 - 4.0 * q, 0.0)
    sqrt_disc = jnp.sqrt(disc)
    if segment == "lower":
        yv = (1.0 - sqrt_disc) / 2.0  # lower (symmetric) root
    else:
        yv = (1.0 + sqrt_disc) / 2.0  # upper (symmetric) root
    return lam0, mu0, xv, yv


def _ggi_xraw_from_x(x, rho, segment="lower"):
    """Inverse of unpack: given target x and ρ, return the logit_x_raw
    that unpacks to it under the chosen segment."""
    x_min = (1.0 - np.sqrt(max(1.0 - rho, 0.0))) / 2.0
    if segment == "lower":
        raw_x = x / x_min  # x ∈ (0, x_min) → raw_x ∈ (0, 1)
    elif segment == "upper":
        raw_x = (1.0 - x) / x_min  # x ∈ (1-x_min, 1) → raw_x ∈ (0, 1)
    else:
        raise ValueError(f"unknown segment {segment!r}")
    raw_x = np.clip(raw_x, 1e-6, 1 - 1e-6)
    return np.log(raw_x / (1 - raw_x))


def init_tkf92(lam=0.04, mu=0.05, ext=0.5):
    """Init in the constrained (log_mu, logit_kappa, logit_ext) basis,
    with κ = lam/mu derived from the user's (lam, mu) request."""
    if not (0 < lam < mu):
        raise ValueError(f"init_tkf92 requires 0 < lam < mu, got "
                          f"lam={lam}, mu={mu}")
    if not (0 < ext < 1):
        raise ValueError(f"init_tkf92 requires 0 < ext < 1, got ext={ext}")
    kappa = lam / mu
    return [
        jnp.asarray(np.log(mu), jnp.float32),
        jnp.asarray(np.log(kappa / (1 - kappa)), jnp.float32),
        jnp.asarray(np.log(ext / (1 - ext)), jnp.float32),
    ]


def init_ggi(mu0=0.05, rho=0.8, x=0.3, segment="lower"):
    """Init in the constrained (log_mu0, logit_rho, logit_x_scaled) basis.

    Constraints (enforced by parameterization):
      - ρ = λ₀/μ₀ ∈ (0, 1) strict (so TKF92 stationary at all t)
      - x ∈ (0, x_min) (lower segment) or x ∈ (1-x_min, 1) (upper segment)
        feasible for the reversibility constraint at ρ
      - y derived from x by the symmetric root in the chosen segment.
    """
    if not (0 < rho < 1):
        raise ValueError(f"init_ggi requires 0 < rho < 1, got rho={rho}")
    x_min = (1.0 - np.sqrt(max(1.0 - rho, 0.0))) / 2.0
    if segment == "lower":
        if not (0 < x < x_min):
            raise ValueError(
                f"init_ggi[lower]: x={x} outside (0, {x_min:.4f}) for "
                f"ρ={rho}.  Pass segment='upper' for x > 1-x_min.")
    elif segment == "upper":
        if not ((1 - x_min) < x < 1):
            raise ValueError(
                f"init_ggi[upper]: x={x} outside ({1-x_min:.4f}, 1) for "
                f"ρ={rho}.  Pass segment='lower' for x < x_min.")
    else:
        raise ValueError(f"unknown segment {segment!r}")
    return [
        jnp.asarray(np.log(mu0), jnp.float32),
        jnp.asarray(np.log(rho / (1 - rho)), jnp.float32),
        jnp.asarray(_ggi_xraw_from_x(x, rho, segment), jnp.float32),
    ]


def report_tkf92(params):
    lam, mu, ext = (float(v) for v in unpack_tkf92(params))
    return {"lam": lam, "mu": mu, "ext": ext,
            "kappa": lam / max(mu, 1e-30)}


def report_ggi(params, segment="lower"):
    lam0, mu0, x, y = (float(v) for v in unpack_ggi(params, segment))
    rho = lam0 / max(mu0, 1e-30)
    return {"lam0": lam0, "mu0": mu0, "x": x, "y": y, "rho": rho}


# ---------------------------------------------------------------------------
# Per-minibatch loss (sum log_prob over pairs in one bucket, same shape)
# ---------------------------------------------------------------------------


def _make_loss_tkf92(Q, pi):
    """Build a loss function that sums TKF92-constant joint pair log LL
    over a minibatch.  forward_backward_2d's output is exactly
    log P_TKF92(anc, des) under the constant model, no prior swap needed.
    Pairs assumed bin-bucketed to the SAME (Lx_pad, Ly_pad).

    Per-pair log_p is clamped to ≥ LOG_P_FLOOR before summation, so any
    single pair hitting NEG_INF (= -1e30, from safe_log of a zero-probability
    transition in forward_backward_2d) can't NaN-poison Adam's gradient.
    The clamp is differentiable (jnp.maximum has a non-zero gradient on
    the active branch), so pairs that ARE below the floor will still
    contribute a gradient pushing them back up.
    """
    Qj, pij = jnp.asarray(Q), jnp.asarray(pi)
    LOG_P_FLOOR = -1e6

    def loss(params, x_batch, y_batch, t_batch, Lx_batch, Ly_batch):
        lam, mu, ext = unpack_tkf92(params)

        def per_pair(x, y, t, Lx, Ly):
            return tkf92_log_prob_fb(
                lam, mu, ext, t, Qj, pij, x, y,
                real_Lx=Lx, real_Ly=Ly)

        log_ps = jax.vmap(per_pair)(
            x_batch, y_batch, t_batch, Lx_batch, Ly_batch)
        # jnp.where(isfinite, ...) replaces NaN/Inf with LOG_P_FLOOR.
        # jnp.maximum(NaN, x) returns NaN, which then poisons Adam's m/v
        # via the gradient — the where is mandatory, not equivalent.
        log_ps_clamped = jnp.where(
            jnp.isfinite(log_ps), log_ps, LOG_P_FLOOR)
        return -jnp.sum(log_ps_clamped), log_ps  # report unclamped for diag

    return loss


def _make_loss_ggi(Q, pi, segment="lower", prior_swap=True):
    """Build a loss function for GGI-steered TKF92 with GGI native ancestor
    prior:  -log P_GGI(anc) - log P_steered_TKF92(des|anc) summed over pairs.

    Equivalent to  -joint_TKF92(steered) + tkf92_marginal(steered) -
    ggi_marginal(native), per pair.

    `segment` is the x-feasibility segment (lower or upper) and is closed
    over here so the JIT'd loss has no Python branching.
    """
    Qj, pij = jnp.asarray(Q), jnp.asarray(pi)
    LOG_P_FLOOR = -1e6

    def loss(params, x_batch, y_batch, t_batch, Lx_batch, Ly_batch):
        lam0, mu0, x_geom, y_geom = unpack_ggi(params, segment)

        def per_pair(x, y, t, Lx, Ly):
            return ggi_steered_log_prob_fb(
                lam0, mu0, x_geom, y_geom, t, Qj, pij, x, y, Lx, Ly,
                prior_swap=prior_swap)

        log_ps = jax.vmap(per_pair)(
            x_batch, y_batch, t_batch, Lx_batch, Ly_batch)
        # See _make_loss_tkf92: where(isfinite,...), not maximum.
        log_ps_clamped = jnp.where(
            jnp.isfinite(log_ps), log_ps, LOG_P_FLOOR)
        return -jnp.sum(log_ps_clamped), log_ps  # unclamped for diag

    return loss


# ---------------------------------------------------------------------------
# Val LL evaluator (just sums log_prob, no gradient)
# ---------------------------------------------------------------------------


def _make_val_eval(Q, pi, mode, segment="lower", prior_swap=True):
    """Per-pair val LL evaluator with the SAME objective as training:
    constant TKF92 joint LL for 'tkf92', GGI-prior-swapped joint LL for 'ggi'.
    """
    Qj, pij = jnp.asarray(Q), jnp.asarray(pi)

    if mode == "tkf92":
        def eval_one(params, x, y, t, Lx, Ly=None):
            # Lx/Ly may be None (unpadded scalar path: shape = real length)
            # or jnp scalars (vmap'd over a pre-padded batch).
            lam, mu, ext = unpack_tkf92(params)
            return tkf92_log_prob_fb(lam, mu, ext, t, Qj, pij, x, y,
                                       real_Lx=Lx, real_Ly=Ly)
    elif mode == "ggi":
        def eval_one(params, x, y, t, Lx, Ly=None):
            lam0, mu0, x_geom, y_geom = unpack_ggi(params, segment)
            return ggi_steered_log_prob_fb(
                lam0, mu0, x_geom, y_geom, t, Qj, pij, x, y, Lx, Ly,
                prior_swap=prior_swap)
    else:
        raise ValueError(f"unknown mode {mode}")

    return eval_one


# ---------------------------------------------------------------------------
# Bin-bucketed pair handling
# ---------------------------------------------------------------------------


def _bin_bucket_pairs(pairs):
    """Group pairs by (Lx_pad, Ly_pad) bin key.  Returns {(Lx_pad, Ly_pad):
    [(x_pad, y_pad, t, Lx_real, Ly_real), ...]} where x, y are padded
    with zeros to bin size; Lx/Ly_real are the original UNPADDED lengths
    (needed by the GGI-steered objective's ancestor-prior term).
    """
    from ..dp.hmm import _pad_to_bin
    buckets = {}
    for x, y, t in pairs:
        Lx_real = int(x.shape[0])
        Ly_real = int(y.shape[0])
        Lx_pad = _pad_to_bin(Lx_real)
        Ly_pad = _pad_to_bin(Ly_real)
        x_pad = np.zeros(Lx_pad, dtype=x.dtype)
        x_pad[:Lx_real] = x
        y_pad = np.zeros(Ly_pad, dtype=y.dtype)
        y_pad[:Ly_real] = y
        buckets.setdefault((Lx_pad, Ly_pad), []).append(
            (x_pad, y_pad, float(t), Lx_real, Ly_real))
    return buckets


def _stack_bucket(pairs_in_bucket):
    """Stack a list of (x_pad, y_pad, t, Lx_real, Ly_real) tuples into
    batched jax arrays."""
    xs = jnp.asarray(np.stack([p[0] for p in pairs_in_bucket]))
    ys = jnp.asarray(np.stack([p[1] for p in pairs_in_bucket]))
    ts = jnp.asarray(np.array([p[2] for p in pairs_in_bucket], np.float32))
    Lx = jnp.asarray(np.array([p[3] for p in pairs_in_bucket], np.int32))
    Ly = jnp.asarray(np.array([p[4] for p in pairs_in_bucket], np.int32))
    return xs, ys, ts, Lx, Ly


# ---------------------------------------------------------------------------
# Adam training loop with val + early stop + bin-bucketed sampling
# ---------------------------------------------------------------------------


def adam_fb_train(
    pairs,
    val_pairs,
    *,
    Q, pi,
    mode="tkf92",               # "tkf92" or "ggi"
    init_params=None,
    n_iter=200, batch_size=100,
    val_every=5, patience=20,
    lr=1e-2,
    seed=0,
    log_fn=print,
    ggi_segment="lower",        # "lower" or "upper" — only used if mode=="ggi"
    ggi_prior_swap=True,        # if False, train GGI with TKF92's joint LL
                                  # (GGI flow's dynamics but TKF92 stationary
                                  # prior) instead of the GGI-native joint.
):
    """Adam-on-FB training with bin-bucketed minibatching + val + early stop.

    pairs / val_pairs: list of (x_int, y_int, t) tuples.  x_int, y_int are
        1-D numpy int arrays of UNPADDED residue indices; t is a python float.
    Q, pi: substitution model.
    mode:   "tkf92" or "ggi".
    init_params: list of jax arrays (log_lam, log_mu, logit_ext) for tkf92
        or (log_lam0, logit_x, logit_y) for ggi.  None -> default inits.
    n_iter, batch_size, val_every, patience, lr: standard knobs.

    Returns dict with 'best_params', 'best_val_ll_per_pair', 'history' list.
    """
    rng = np.random.default_rng(seed)

    # ----- Bin-bucket training pairs -----
    log_fn(f"adam_fb_train ({mode}): {len(pairs)} train pairs, "
           f"{len(val_pairs)} val pairs.")
    train_buckets = _bin_bucket_pairs(pairs)
    bucket_keys = list(train_buckets.keys())
    bucket_sizes = np.array([len(train_buckets[k]) for k in bucket_keys])
    bucket_probs = bucket_sizes / bucket_sizes.sum()
    log_fn(f"  pre-bucketed train into {len(bucket_keys)} (Lx_pad, Ly_pad) "
           f"cells (largest {int(bucket_sizes.max())}, smallest "
           f"{int(bucket_sizes.min())}).")

    # Val is processed in CHUNKS of exactly `batch_size` pairs, matching the
    # training-batched JIT-compiled shapes.  A naive "feed the whole val
    # bucket as one batch" approach compiles a new (Lx_pad, Ly_pad, N_val_bucket)
    # shape per bucket — fine at batch=16 / n_val=50 (~50 pairs total), OOMs
    # at batch=64 / n_val=500 (~50-pair buckets × 256×256 × 5 states × float64).
    # Chunking to `batch_size` ensures val reuses the EXACT compiled shapes
    # training already cached, so adding val doesn't add JIT workspace.
    # Each (bucket_key, chunk_of_batch_size) is stacked and the last chunk
    # is padded with-replacement to keep shape consistent.
    val_buckets = _bin_bucket_pairs(val_pairs)
    n_val_total = sum(len(lst) for lst in val_buckets.values())

    val_chunks = []   # list of (xs, ys, ts, Lx, Ly, n_real)
    for lst in val_buckets.values():
        n = len(lst)
        for i in range(0, n, batch_size):
            chunk = lst[i:i + batch_size]
            n_real = len(chunk)
            if n_real < batch_size:
                # Pad chunk by repeating the last entry; the per-pair LL of
                # the repeated entries will be SUBTRACTED OUT via n_real.
                chunk = chunk + [chunk[-1]] * (batch_size - n_real)
            xs, ys, ts, Lx, Ly = _stack_bucket(chunk)
            val_chunks.append((xs, ys, ts, Lx, Ly, n_real))
    log_fn(f"  pre-bucketed val into {len(val_buckets)} cells "
           f"(total {n_val_total} pairs); {len(val_chunks)} chunks of "
           f"size {batch_size} for val_eval (matches training shapes).")

    # ----- Loss + Adam state -----
    if init_params is None:
        init_params = (init_tkf92() if mode == "tkf92"
                       else init_ggi(segment=ggi_segment))
    if mode == "tkf92":
        loss_fn = _make_loss_tkf92(Q, pi)
        report = report_tkf92
    elif mode == "ggi":
        loss_fn = _make_loss_ggi(Q, pi, segment=ggi_segment,
                                  prior_swap=ggi_prior_swap)
        report = lambda p: report_ggi(p, segment=ggi_segment)
    else:
        raise ValueError(f"unknown mode {mode}")

    # JIT the loss separately per bucket-shape (auto-cached by JAX).
    loss_vag = jax.jit(jax.value_and_grad(loss_fn, has_aux=True, argnums=0))
    # Forward-only JIT for val (no backward → reuses fewer cache slots).
    loss_value_only = jax.jit(loss_fn)

    params = list(init_params)
    m = [jnp.zeros_like(p) for p in params]
    v = [jnp.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    def adam_step(params, grads, m, v, t):
        new_m = [b1 * mi + (1 - b1) * gi for mi, gi in zip(m, grads)]
        new_v = [b2 * vi + (1 - b2) * (gi * gi) for vi, gi in zip(v, grads)]
        mh = [mi / (1 - b1 ** t) for mi in new_m]
        vh = [vi / (1 - b2 ** t) for vi in new_v]
        new_p = [pi - lr * mhi / (jnp.sqrt(vhi) + eps)
                 for pi, mhi, vhi in zip(params, mh, vh)]
        return new_p, new_m, new_v

    def eval_val():
        """Sum log_prob over all val pairs, in chunks of size `batch_size`
        matching the training-cached JIT shapes.  Each chunk's last n_real
        entries are real; the rest are repeated padding whose per-pair
        log_p is excluded from the sum via the per-pair log_ps return."""
        total = 0.0
        for x_b, y_b, t_b, Lx_b, Ly_b, n_real in val_chunks:
            # loss_value_only returns (-sum_log_p, per_pair_log_p);
            # we use the per-pair array to sum only the n_real real entries.
            _, log_ps = loss_value_only(
                params, x_b, y_b, t_b, Lx_b, Ly_b)
            total += float(jnp.sum(log_ps[:n_real]))
        return total

    history = []
    best_val_total = -float("inf")
    best_params = [float(p) for p in params]
    val_no_improve = 0

    t0 = time.time()
    for it in range(1, n_iter + 1):
        # Pick a bucket (weighted), draw batch_size pairs from it
        bk = int(rng.choice(len(bucket_keys), p=bucket_probs))
        pool = train_buckets[bucket_keys[bk]]
        if len(pool) >= batch_size:
            ix = rng.choice(len(pool), batch_size, replace=False)
        else:
            ix = rng.choice(len(pool), batch_size, replace=True)
        sub = [pool[i] for i in ix]
        x_batch, y_batch, t_batch, Lx_batch, Ly_batch = _stack_bucket(sub)

        (loss_val, log_ps), grads = loss_vag(
            params, x_batch, y_batch, t_batch, Lx_batch, Ly_batch)
        # Belt-and-suspenders NaN protection.  Even with the loss-side
        # where(isfinite,...), 0*NaN = NaN in IEEE 754 can sneak NaN through
        # the backward pass when the upstream forward was NaN.  Zeroing
        # non-finite gradients before Adam prevents the m/v stats from
        # being permanently poisoned by a single bad pair.
        grads_clean = [
            jnp.where(jnp.isfinite(g), g, 0.0) for g in grads]
        params, m, v = adam_step(params, grads_clean, m, v, it)

        if it % val_every == 0 or it == n_iter:
            val_ll = eval_val()
            val_per_pair = val_ll / max(n_val_total, 1)
            train_per_pair = -float(loss_val) / batch_size
            rep = report(params)
            elapsed = time.time() - t0
            entry = {"iter": it, "train_ll_per_pair": train_per_pair,
                     "val_ll_total": val_ll,
                     "val_ll_per_pair": val_per_pair, "elapsed": elapsed,
                     **rep}
            history.append(entry)
            param_str = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in rep.items())
            log_fn(f"  iter {it:4d}/{n_iter}  "
                   f"train_ll/pair={train_per_pair:9.2f}  "
                   f"val_ll/pair={val_per_pair:9.2f}  {param_str}  "
                   f"({elapsed:.1f}s)")
            if val_ll > best_val_total:
                best_val_total = val_ll
                best_params = [float(p) for p in params]
                val_no_improve = 0
            else:
                val_no_improve += 1
                if val_no_improve >= patience:
                    log_fn(f"  early stop: val LL no improvement for "
                           f"{val_no_improve} consecutive val checks "
                           f"(patience={patience}).")
                    break

    return {
        "best_params": best_params,
        "best_val_ll_total": best_val_total,
        "best_val_ll_per_pair": best_val_total / max(n_val_total, 1),
        "history": history,
        "mode": mode,
        "ggi_segment": ggi_segment if mode == "ggi" else None,
        "ggi_prior_swap": ggi_prior_swap if mode == "ggi" else None,
    }
