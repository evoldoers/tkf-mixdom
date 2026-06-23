"""FSA-style sequence annealing for multiple sequence alignment.

Implements the sequence annealing approach from Bradley et al. (2009, PLoS
Comp Biol) using TKF92 or MixDom pair HMMs instead of the standard 3/5-state
model.

Algorithm:
1. Select sequence pairs (full O(N^2) or Erdos-Renyi O(N log N))
2. For each pair: FB at tau=1, optimize tau, re-run FB at optimal tau
3. Extract pairwise residue alignment posteriors P(i aligned to j)
4. Build MSA via sequence annealing (iterative greedy refinement)

References:
    Bradley RK, Roberts A, Smoot M, Juvekar S, Do J, Dewey C, Holmes I,
    Pachter L. "Fast Statistical Alignment." PLoS Computational Biology
    5(5): e1000392 (2009).
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from ..core.params import S, M, I, D, E
from ..core.ctmc import transition_matrix
from ..dp.hmm import (
    forward_backward_2d, pair_hmm_emissions, safe_log,
    _forward_2d_core, _backward_2d_core, _find_e_idx,
    _pad_to_bin, _pad_seq, NEG_INF,
)


# ============================================================
# Canonical per-domain substitution matrix builder
# ============================================================

def build_per_domain_sub_matrices(params, t, n_dom):
    """Build per-domain substitution matrices P_n(t) from model params.

    This is the ONE canonical implementation. All code that needs
    per-domain P(t) matrices should call this function.

    Args:
        params: model parameter dict. Must contain 'pi' (N, A).
            If 'S_exch' is present and 3D (N, A, A): per-domain exchangeability.
            If 'S_exch' is present and 2D (A, A): shared exchangeability.
            If neither: falls back to 'Q' + 'pi' (single rate matrix).
        t: evolutionary time (float or scalar JAX array)
        n_dom: number of domains

    Returns:
        sub_matrices: (N, A, A) per-domain substitution matrices
        pis: (N, A) per-domain equilibrium distributions
    """
    from ..core.ctmc import transition_matrix
    from ..core.ctmc import build_Q_from_S_pi as build_rate_matrix

    pis = jnp.array(params['pi'])
    if pis.ndim == 1:
        pis = jnp.tile(pis[None], (n_dom, 1))

    if 'S_exch' in params:
        S_exch = jnp.array(params['S_exch'])
        if S_exch.ndim == 3:
            # Per-domain exchangeability
            sub_matrices = jax.vmap(
                lambda S, pi: transition_matrix(build_rate_matrix(S, pi), t)
            )(S_exch, pis)
        else:
            # Shared exchangeability, per-domain equilibrium
            sub_matrices = jax.vmap(
                lambda pi: transition_matrix(build_rate_matrix(S_exch, pi), t)
            )(pis)
    elif 'Q' in params:
        # Legacy: single Q matrix
        sub_matrix = transition_matrix(params['Q'], t)
        sub_matrices = jnp.tile(sub_matrix[None], (n_dom, 1, 1))
    else:
        raise ValueError("params must contain 'S_exch' or 'Q'")

    return sub_matrices, pis


# ============================================================
# Pairwise posterior computation
# ============================================================

# Module-level JIT-cached E[LL(tau)] for TKF92 + LG: defining at module
# scope (rather than as an inner closure) lets the JIT cache survive
# across pairs. All pair-specific tensors are explicit arguments.
def _tkf92_expected_ll(log_tau, n_trans_fixed, match_W_fixed,
                        ins_rate, del_rate, ext, Q, pi):
    from ..core.params import tkf92_trans
    tau = jnp.exp(log_tau)
    chi = tkf92_trans(ins_rate, del_rate, tau, ext)
    log_chi = jnp.log(jnp.maximum(chi, 1e-300))
    trans_term = jnp.sum(n_trans_fixed * log_chi)
    P_t = transition_matrix(Q, tau)
    log_P = jnp.log(jnp.maximum(P_t, 1e-300))
    emit_term = jnp.sum(
        match_W_fixed * (jnp.log(jnp.maximum(pi, 1e-300))[:, None] + log_P))
    return trans_term + emit_term


_tkf92_tau_grad = jax.jit(jax.grad(_tkf92_expected_ll, argnums=0))
_tkf92_tau_hess = jax.jit(jax.grad(jax.grad(_tkf92_expected_ll, argnums=0),
                                    argnums=0))


# Module-level mixture E[LL(tau)] for K-component TKF92.
# CRITICAL invariants (rigor-auditor watch list):
#   - This is responsibility-weighted SUM of per-component log-likelihoods:
#         Σ_k R_k · ll_{k,i}(τ)
#     NOT log of weighted prob-sum. The correct EM Q-function for HMM
#     mixtures is the former; the latter is a different (and incorrect)
#     auxiliary objective.
#   - Per-component (n_trans_K[k], match_W_K[k]) are kept SEPARATE; never
#     summed into "mixture counts" before the call.
#   - Per-component (ins_K[k], del_K[k], ext_K[k], Q_K[k], pi_K[k]) are
#     kept SEPARATE; never averaged into a "mixture-Q" before the call.
def _tkf92_mixture_expected_ll_per_pair(
        log_tau,                                        # scalar
        R,                                              # (K,) family responsibilities
        n_trans_K, match_W_K,                           # (K,5,5), (K,A,A)
        ins_K, del_K, ext_K, Q_K, pi_K):                # (K,), (K,), (K,), (K,A,A), (K,A)
    """Mixture Q-function for τ-refit at one pair.

    Σ_k R_k · _tkf92_expected_ll(τ; n_trans_K[k], match_W_K[k],
                                  ins_K[k], del_K[k], ext_K[k], Q_K[k], pi_K[k]).

    The vmap distributes over k; per-component sufficient statistics and
    parameters are passed separately; no averaging of params or counts.
    """
    ll_k = jax.vmap(
        _tkf92_expected_ll,
        in_axes=(None, 0, 0, 0, 0, 0, 0, 0)
    )(log_tau, n_trans_K, match_W_K, ins_K, del_K, ext_K, Q_K, pi_K)  # (K,)
    return jnp.sum(R * ll_k)


_tkf92_mixture_tau_grad = jax.jit(jax.grad(
    _tkf92_mixture_expected_ll_per_pair, argnums=0))
_tkf92_mixture_tau_hess = jax.jit(jax.grad(
    jax.grad(_tkf92_mixture_expected_ll_per_pair, argnums=0), argnums=0))


def _mixdom_expected_ll(
        log_tau, n_trans_fixed, match_counts, insert_counts, delete_counts,
        main_ins, main_del, dom_ins, dom_del, dom_weights,
        frag_weights, ext_rates, S_exch_3d, pis):
    """E[LL(tau)] for MixDom with reduced sufficient statistics. All inputs
    are explicit so the JIT cache can be reused across pairs (per shape)."""
    from ..models.mixdom import build_nested_trans
    from ..core.ctmc import build_Q_from_S_pi as build_rate_matrix
    tau = jnp.exp(log_tau)
    chi, _ = build_nested_trans(
        main_ins, main_del, tau,
        dom_ins, dom_del, dom_weights, frag_weights, ext_rates)
    log_chi = jnp.log(jnp.maximum(chi, 1e-300))
    trans_term = jnp.sum(n_trans_fixed * log_chi)

    sub_mats = jax.vmap(
        lambda S, pi: transition_matrix(build_rate_matrix(S, pi), tau)
    )(S_exch_3d, pis)
    log_sub = jnp.log(jnp.maximum(sub_mats, 1e-300))   # (n_dom, A, A)
    log_pi = jnp.log(jnp.maximum(pis, 1e-300))         # (n_dom, A)

    emit_match = jnp.sum(match_counts * (log_pi[:, :, None] + log_sub))
    emit_insert = jnp.sum(insert_counts * log_pi)
    emit_delete = jnp.sum(delete_counts * log_pi)
    return trans_term + emit_match + emit_insert + emit_delete


_mixdom_tau_grad = jax.jit(jax.grad(_mixdom_expected_ll, argnums=0))
_mixdom_tau_hess = jax.jit(jax.grad(jax.grad(_mixdom_expected_ll, argnums=0),
                                     argnums=0))


def pairwise_posteriors_tkf92(x_seq, y_seq, ins_rate, del_rate, ext,
                               Q, pi, n_newton=5, tau_init=1.0):
    """Compute pairwise residue alignment posteriors using TKF92.

    Same architecture as pairwise_posteriors_mixdom:
    1. One FB at tau_init → expected transition and emission counts
    2. Reduce to sufficient statistics (5×5 trans counts + 20×20 match counts)
    3. NR on E[LL(tau)] via autograd on small matrices (no sequence-length JIT)
    4. Re-run FB at optimal tau for final posteriors

    Args:
        x_seq, y_seq: (Lx,), (Ly,) integer sequence arrays
        ins_rate, del_rate, ext: TKF92 parameters
        Q: (A, A) rate matrix
        pi: (A,) equilibrium distribution
        n_newton: Newton-Raphson steps for tau optimization
        tau_init: initial evolutionary time

    Returns:
        match_posteriors: (Lx, Ly) P(residue i aligned to residue j)
        tau_opt: optimized evolutionary time
        log_prob: log-probability at optimal tau
    """
    x_j = jnp.asarray(x_seq)
    y_j = jnp.asarray(y_seq)
    mp, tau, lp = _pairwise_posteriors_tkf92_jax(
        x_j, y_j,
        jnp.int32(x_j.shape[0]), jnp.int32(y_j.shape[0]),
        ins_rate, del_rate, ext, jnp.asarray(Q), jnp.asarray(pi),
        n_newton=n_newton, tau_init=tau_init)
    return np.asarray(mp), float(tau), float(lp)


def _pairwise_posteriors_tkf92_jax(x, y, real_Lx, real_Ly,
                                    ins_rate, del_rate, ext, Q, pi,
                                    n_newton=5, tau_init=1.0):
    """vmap-safe core of pairwise_posteriors_tkf92. x, y are jnp arrays
    of shape (Lx_pad,), (Ly_pad,); real_Lx, real_Ly are traced scalars
    giving the unpadded lengths."""
    from ..models.left_regular import make_tkf92_pair_hmm

    A = Q.shape[0]
    Lx, Ly = x.shape[0], y.shape[0]

    # Step 1: FB at tau_init
    log_trans0, st0, sub0, pi0 = make_tkf92_pair_hmm(
        ins_rate, del_rate, tau_init, ext, Q, pi)
    _, posteriors0, n_trans0 = forward_backward_2d(
        log_trans0, st0, x, y, sub0, pi0,
        real_Lx=real_Lx, real_Ly=real_Ly)

    # Step 2: Reduce to sufficient statistics (pure JAX).
    n_trans_fixed = jax.lax.stop_gradient(n_trans0)
    post_fixed = jax.lax.stop_gradient(posteriors0)
    st_j = jnp.asarray(st0)
    is_M_j = (st_j == M).astype(jnp.float64)
    X_oh = jax.nn.one_hot(x, A, dtype=jnp.float64)
    Y_oh = jax.nn.one_hot(y, A, dtype=jnp.float64)
    post_MM = post_fixed[1:Lx + 1, 1:Ly + 1, :]
    match_post = jnp.einsum('ijs,s->ij', post_MM, is_M_j)
    match_W_fixed = jnp.einsum('ij,ia,jb->ab', match_post, X_oh, Y_oh)

    # Step 3: NR via module-level JIT-cached grad/hess (no Python branches).
    log_tau = jnp.log(jnp.float64(tau_init))
    for _ in range(n_newton):
        g = _tkf92_tau_grad(log_tau, n_trans_fixed, match_W_fixed,
                            ins_rate, del_rate, ext, Q, pi)
        h = _tkf92_tau_hess(log_tau, n_trans_fixed, match_W_fixed,
                            ins_rate, del_rate, ext, Q, pi)
        safe_neg_h = jnp.where(jnp.abs(h) > 1e-10, -h, 1.0)
        step = jnp.clip(g / safe_neg_h, -1.0, 1.0)
        log_tau = log_tau + step
    tau_opt = jnp.exp(jnp.clip(log_tau, jnp.log(1e-4), jnp.log(10.0)))

    # Step 4: FB at optimal tau.
    log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
        ins_rate, del_rate, tau_opt, ext, Q, pi)
    log_prob, posteriors, _ = forward_backward_2d(
        log_trans, state_types, x, y, sub_matrix, pi_out,
        real_Lx=real_Lx, real_Ly=real_Ly)

    is_match = (state_types == M)
    match_posteriors = jnp.sum(
        posteriors[1:Lx + 1, 1:Ly + 1, :] * is_match[None, None, :],
        axis=-1)
    return match_posteriors, tau_opt, log_prob


def pairwise_posteriors_tkf92_batched(xs, ys, real_Lxs, real_Lys,
                                       ins_rate, del_rate, ext, Q, pi,
                                       n_newton=5, tau_init=1.0):
    """Batched (vmap'd) pairwise posteriors for TKF92. xs, ys: (B, Lx_pad),
    (B, Ly_pad). real_Lxs, real_Lys: (B,) int32 arrays of unpadded lengths.
    Returns (match_post (B, Lx_pad, Ly_pad), tau (B,), lp (B,))."""
    Q_j = jnp.asarray(Q)
    pi_j = jnp.asarray(pi)
    return jax.vmap(
        lambda x, y, lx, ly: _pairwise_posteriors_tkf92_jax(
            x, y, lx, ly, ins_rate, del_rate, ext, Q_j, pi_j,
            n_newton=n_newton, tau_init=tau_init),
        in_axes=(0, 0, 0, 0),
    )(xs, ys, real_Lxs, real_Lys)


# ---------------------------------------------------------------------------
# Mixture-of-K-TKF92 family-level pair posteriors.
#
# CRITICAL design invariants (rigor-auditor-monitored):
#
#   1. The single-TKF92 anchor is the ONLY consensus model in this pipeline.
#      It is used ONCE, in step 1, for the initial per-pair τ̂_i. Every
#      downstream FB / E-step / posterior / τ-refit uses per-component
#      (S_k, π_k, λ_k, μ_k, r_k) without averaging.
#
#   2. Family responsibility R_k is computed once per outer round from the
#      JOINT family LL: log w_k = log mix_weights[k] + Σ_i log p_{k,i}; then
#      softmax over k. Not per-pair averaged.
#
#   3. τ-refit uses the responsibility-weighted log-sum
#         Σ_k R_k · log p_{k,i}(τ)
#      (the correct EM Q-function for HMM mixtures). NOT log of weighted
#      prob-sum log Σ_k R_k · p_{k,i}(τ).
#
#   4. Mixture posterior mp_i = Σ_k R_k · mp_{k,i} mixes DISTRIBUTIONS, not
#      parameters. Per-pair mp_{k,i} is computed under component k's full
#      params via its own FB.
#
#   5. Per-pair τ̂_i (one τ per pair) — NOT shared across the family.
#
#   6. Padding: callers must pre-pad sequences to bins via _pad_to_bin /
#      _pad_seq before invoking this function (pad-to-bin discipline from
#      pairwise_posteriors_mixdom is enforced at the caller level).
# ---------------------------------------------------------------------------
@jax.jit
def _tkf92_fb_at_tau_one(x, y, real_Lx, real_Ly, tau,
                          ins_rate, del_rate, ext, Q, pi):
    """Forward-backward for one (x, y) pair with one TKF92 component at fixed τ.

    Returns (log_p, mp, n_trans, match_W) — same suff-stat decomposition
    as `_pairwise_posteriors_tkf92_jax` but at a CALLER-supplied τ
    (no NR inside).

    Invariant: takes per-call (ins, del, ext, Q, pi) — there is no
    averaging or pooling across components inside this function.
    """
    from ..models.left_regular import make_tkf92_pair_hmm
    A = Q.shape[0]
    Lx, Ly = x.shape[0], y.shape[0]
    log_trans, st, sub_mat, pi_out = make_tkf92_pair_hmm(
        ins_rate, del_rate, tau, ext, Q, pi)
    log_p, posteriors, n_trans = forward_backward_2d(
        log_trans, st, x, y, sub_mat, pi_out,
        real_Lx=real_Lx, real_Ly=real_Ly)
    is_M = (jnp.asarray(st) == M).astype(jnp.float64)
    X_oh = jax.nn.one_hot(x, A, dtype=jnp.float64)
    Y_oh = jax.nn.one_hot(y, A, dtype=jnp.float64)
    post_MM = posteriors[1:Lx + 1, 1:Ly + 1, :]
    match_post = jnp.einsum('ijs,s->ij', post_MM, is_M)
    match_W = jnp.einsum('ij,ia,jb->ab', match_post, X_oh, Y_oh)
    is_match = (st == M)
    mp = jnp.sum(
        posteriors[1:Lx + 1, 1:Ly + 1, :] * is_match[None, None, :],
        axis=-1)
    return log_p, mp, n_trans, match_W


def pairwise_posteriors_tkf92_mixture_family(
        pairs_x, pairs_y, real_Lxs, real_Lys,
        *,
        anchor_ins, anchor_del, anchor_ext, anchor_Q, anchor_pi,
        mix_weights, mix_ins, mix_del, mix_ext, mix_Q, mix_pi,
        n_outer_rounds=1, n_newton=5, tau_init=1.0,
        k_chunk=None, pair_chunk=None):
    """Family-level mixture-of-K-TKF92s pairwise residue posteriors.

    For one BAliBase-style family with `n_pairs` pairs, computes per-pair
    match posteriors under a K-component TKF92 mixture, with a single
    family-level responsibility shared across all pairs in the family.

    Approach selector via `n_outer_rounds`:
      0 → approach (1): anchor τ̂_i once (single-TKF92), then per-component
        FB at τ̂_i, then family R, then mixture posterior. No τ refit.
      1 → approach (2): approach (1) + one round of τ refit using the
        responsibility-weighted mixture E[ll], then a final per-component
        FB at the refit τ + mixture posterior.
      ≥2 → approach (3): iterate (FB → R → τ refit) `n_outer_rounds` times.

    Args:
      pairs_x, pairs_y: (n_pairs, Lx_pad), (n_pairs, Ly_pad) padded
        sequence arrays. CALLER must pad to geometric bins for JIT cache
        reuse (see `tkfmixdom.jax.dp.hmm._pad_to_bin`).
      real_Lxs, real_Lys: (n_pairs,) int32 unpadded lengths.
      anchor_*: single-TKF92 anchor params for the initial τ̂_i fit (the
        ONLY consensus use; every downstream computation is per-component).
      mix_weights: (K,) component prior probabilities.
      mix_ins, mix_del, mix_ext: (K,) per-component TKF92 indel params.
      mix_Q: (K, A, A) per-component rate matrices.
      mix_pi: (K, A) per-component equilibrium distributions.
      n_outer_rounds: see selector above.
      n_newton: Newton-Raphson steps per τ optimisation.
      tau_init: initial τ for the anchor NR.

    Returns:
      mp_mixture: (n_pairs, Lx_pad, Ly_pad) mixture match posteriors.
      tau_per_pair: (n_pairs,) final τ estimates.
      R_family: (K,) family responsibilities used in the final mixture.
    """
    n_pairs = pairs_x.shape[0]
    K = mix_weights.shape[0]

    # Stage 1: anchor τ̂_i per pair via single-TKF92 NR. ONLY consensus use.
    # Pair-chunked too (full vmap over n_pairs blew up GPU memory on
    # the largest BAliBase families even though it's single-component).
    def _anchor_tau_one(x, y, rL, rR):
        _, tau_opt, _ = _pairwise_posteriors_tkf92_jax(
            x, y, rL, rR,
            jnp.float64(anchor_ins), jnp.float64(anchor_del),
            jnp.float64(anchor_ext),
            jnp.asarray(anchor_Q), jnp.asarray(anchor_pi),
            n_newton=n_newton, tau_init=tau_init)
        return tau_opt
    _anchor_tau_vmap = jax.vmap(_anchor_tau_one)
    if pair_chunk is None or pair_chunk >= n_pairs:
        tau_curr = _anchor_tau_vmap(
            pairs_x, pairs_y, real_Lxs, real_Lys)        # (n_pairs,)
    else:
        tau_chunks = []
        for s in range(0, n_pairs, pair_chunk):
            e = min(s + pair_chunk, n_pairs)
            tau_chunks.append(_anchor_tau_vmap(
                pairs_x[s:e], pairs_y[s:e],
                real_Lxs[s:e], real_Lys[s:e]))
        tau_curr = jnp.concatenate(tau_chunks)            # (n_pairs,)

    # Pre-cast mixture params to JAX arrays.
    mix_weights_j = jnp.asarray(mix_weights, dtype=jnp.float64)
    mix_ins_j = jnp.asarray(mix_ins, dtype=jnp.float64)
    mix_del_j = jnp.asarray(mix_del, dtype=jnp.float64)
    mix_ext_j = jnp.asarray(mix_ext, dtype=jnp.float64)
    mix_Q_j = jnp.asarray(mix_Q, dtype=jnp.float64)
    mix_pi_j = jnp.asarray(mix_pi, dtype=jnp.float64)
    log_mix_weights = jnp.log(jnp.maximum(mix_weights_j, 1e-300))

    # FB at fixed τ over (K, n_pairs):
    #
    # Two independent chunking knobs control memory:
    #   k_chunk:    chunk K-component dim (default: K = full vmap).
    #               Process B_K = k_chunk components in parallel via
    #               vmap inside a `lax.map` over (K / k_chunk) chunks.
    #   pair_chunk: chunk n_pairs dim (default: n_pairs = full vmap).
    #               Process B_P = pair_chunk pairs in parallel via vmap
    #               inside a Python for-loop over (n_pairs / pair_chunk)
    #               chunks. (Python loop instead of lax.map because the
    #               last chunk may have fewer pairs and lax.map needs
    #               uniform shapes; the Python loop is OK because each
    #               chunk dispatches a single GPU call.)
    #
    # Both default to None (full vmap) for max throughput; the caller
    # lowers them when peak memory exceeds the GPU's. For BAliBase with
    # max n_pairs=66 and Lx_pad up to 1024, k_chunk=1 + pair_chunk=8
    # keeps peak memory under ~3 GB on an 11 GiB GPU.

    n_pairs = int(pairs_x.shape[0])
    fb_per_pair = jax.vmap(
        _tkf92_fb_at_tau_one,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None))
    fb_per_pair_K_inner = jax.vmap(
        fb_per_pair,
        in_axes=(None, None, None, None, None, 0, 0, 0, 0, 0))

    # Build a low-level "FB on a pair-slice with a K-component-slice"
    # primitive. Outer chunking iterates over (K-chunk × pair-chunk)
    # combinations. Note `tau_slice` is an EXPLICIT argument (not
    # closed-over `tau_curr`), so the pair-chunk dispatcher can pass
    # the matching tau-slice to each pair chunk.
    def _fb_pair_slice_for_K_slice(px, py, lx, ly, tau_slice,
                                    ins_b, del_b, ext_b, Q_b, pi_b):
        # px: (B_P, Lx_pad), py: (B_P, Ly_pad), lx/ly/tau_slice: (B_P,).
        # ins_b..pi_b: (B_K, ...).
        return fb_per_pair_K_inner(px, py, lx, ly, tau_slice,
                                    ins_b, del_b, ext_b, Q_b, pi_b)
        # → (B_K, B_P, ...)

    # Build the K-axis chunked dispatcher (returns a function that takes
    # px, py, lx, ly, tau_slice and returns full-K results for that pair-slice).
    if k_chunk is None or k_chunk >= K:
        def _full_K_for_pair_slice(px, py, lx, ly, tau_slice):
            return _fb_pair_slice_for_K_slice(
                px, py, lx, ly, tau_slice,
                mix_ins_j, mix_del_j, mix_ext_j, mix_Q_j, mix_pi_j)
    else:
        if K % k_chunk != 0:
            raise ValueError(
                f"k_chunk={k_chunk} must divide K={K}; "
                f"pad mixture to a multiple of k_chunk or pick a divisor.")
        nKc = K // k_chunk
        mix_ins_C = mix_ins_j.reshape(nKc, k_chunk)
        mix_del_C = mix_del_j.reshape(nKc, k_chunk)
        mix_ext_C = mix_ext_j.reshape(nKc, k_chunk)
        mix_Q_C = mix_Q_j.reshape(nKc, k_chunk, *mix_Q_j.shape[1:])
        mix_pi_C = mix_pi_j.reshape(nKc, k_chunk, *mix_pi_j.shape[1:])

        def _full_K_for_pair_slice(px, py, lx, ly, tau_slice):
            def _fb_one_K_chunk(chunk_args):
                ins_b, del_b, ext_b, Q_b, pi_b = chunk_args
                return _fb_pair_slice_for_K_slice(
                    px, py, lx, ly, tau_slice,
                    ins_b, del_b, ext_b, Q_b, pi_b)
            log_p_C, mp_C, n_trans_C, match_W_C = jax.lax.map(
                _fb_one_K_chunk,
                (mix_ins_C, mix_del_C, mix_ext_C, mix_Q_C, mix_pi_C))
            # Reshape (nKc, k_chunk, B_P, ...) → (K, B_P, ...)
            return (log_p_C.reshape(K, *log_p_C.shape[2:]),
                    mp_C.reshape(K, *mp_C.shape[2:]),
                    n_trans_C.reshape(K, *n_trans_C.shape[2:]),
                    match_W_C.reshape(K, *match_W_C.shape[2:]))

    # Pair-chunking dispatcher — calls _full_K_for_pair_slice once per
    # pair-chunk and concatenates along the n_pairs axis. tau_curr is
    # closed-over (changes each round); we slice it with the pair window.
    def fb_K_pairs_packaged():
        if pair_chunk is None or pair_chunk >= n_pairs:
            return _full_K_for_pair_slice(
                pairs_x, pairs_y, real_Lxs, real_Lys, tau_curr)
        log_p_chunks = []
        mp_chunks = []
        n_trans_chunks = []
        match_W_chunks = []
        for s in range(0, n_pairs, pair_chunk):
            e = min(s + pair_chunk, n_pairs)
            log_p_c, mp_c, n_trans_c, match_W_c = _full_K_for_pair_slice(
                pairs_x[s:e], pairs_y[s:e],
                real_Lxs[s:e], real_Lys[s:e], tau_curr[s:e])
            log_p_chunks.append(log_p_c)
            mp_chunks.append(mp_c)
            n_trans_chunks.append(n_trans_c)
            match_W_chunks.append(match_W_c)
        return (jnp.concatenate(log_p_chunks, axis=1),
                jnp.concatenate(mp_chunks, axis=1),
                jnp.concatenate(n_trans_chunks, axis=1),
                jnp.concatenate(match_W_chunks, axis=1))

    R = jnp.full((K,), 1.0 / K, dtype=jnp.float64)
    log_p_KP = mp_KP = n_trans_KP = match_W_KP = None

    for round_idx in range(n_outer_rounds + 1):
        # FB at current τ for every (k, pair). Per-component params kept
        # SEPARATE — no averaging.
        log_p_KP, mp_KP, n_trans_KP, match_W_KP = fb_K_pairs_packaged()
        # Shapes: (K, n_pairs), (K, n_pairs, Lx_pad, Ly_pad),
        #         (K, n_pairs, 5, 5), (K, n_pairs, A, A)

        # Family responsibility from JOINT family LL — single softmax over K.
        # log w_k = log mix_weights[k] + Σ_i log p_{k,i}.
        log_w_k = log_mix_weights + jnp.sum(log_p_KP, axis=1)        # (K,)
        R = jax.nn.softmax(log_w_k)                                  # (K,)

        if round_idx == n_outer_rounds:
            break

        # τ refit per pair via mixture-Q NR. Counts (n_trans_K, match_W_K)
        # frozen at this round's FB — `stop_gradient` makes them M-step
        # constants so the NR ∂Q/∂log_τ doesn't back-prop through the FB.
        # Mixture Q = Σ_k R_k · ll_{k,i}(τ) (responsibility-weighted
        # log-sum, NOT log of weighted prob-sum). Vmap over pairs; per-pair
        # NR uses the full K-vector counts and params.
        n_trans_PK = jax.lax.stop_gradient(
            jnp.swapaxes(n_trans_KP, 0, 1))                    # (n_pairs, K, 5, 5)
        match_W_PK = jax.lax.stop_gradient(
            jnp.swapaxes(match_W_KP, 0, 1))                     # (n_pairs, K, A, A)
        # Stop-gradient on R too: under multi-round iteration `R` is a
        # JAX traced node (softmax of summed log_p_KP), and we want each
        # outer round's NR to treat R as a fixed M-step constant.
        R_stop = jax.lax.stop_gradient(R)

        # Pass R as an explicit vmap arg with in_axes=None instead of
        # closure-capturing it. Without this, `R` (a fresh traced node
        # each outer round) re-traces the vmapped scan at every round
        # → silent JIT recompile per round for n_outer_rounds >= 2.
        def _tau_refit_one(log_tau_i, R_in, n_trans_K_i, match_W_K_i):
            def step(carry, _):
                lt = carry
                g = _tkf92_mixture_tau_grad(
                    lt, R_in, n_trans_K_i, match_W_K_i,
                    mix_ins_j, mix_del_j, mix_ext_j, mix_Q_j, mix_pi_j)
                h = _tkf92_mixture_tau_hess(
                    lt, R_in, n_trans_K_i, match_W_K_i,
                    mix_ins_j, mix_del_j, mix_ext_j, mix_Q_j, mix_pi_j)
                safe_neg_h = jnp.where(jnp.abs(h) > 1e-10, -h, 1.0)
                stp = jnp.clip(g / safe_neg_h, -1.0, 1.0)
                return lt + stp, None
            log_tau_out, _ = jax.lax.scan(
                step, log_tau_i, None, length=n_newton)
            return jnp.exp(jnp.clip(log_tau_out,
                                    jnp.log(1e-4), jnp.log(10.0)))

        log_tau_init_arr = jnp.log(jnp.maximum(tau_curr, 1e-12))
        tau_curr = jax.vmap(_tau_refit_one,
                            in_axes=(0, None, 0, 0))(
            log_tau_init_arr, R_stop, n_trans_PK, match_W_PK)  # (n_pairs,)

    # Mixture posterior: R (K,) · mp_KP (K, n_pairs, Lx_pad, Ly_pad).
    # Mixes DISTRIBUTIONS not parameters.
    mp_mixture = jnp.sum(
        R[:, None, None, None] * mp_KP, axis=0)              # (n_pairs, Lx_pad, Ly_pad)

    return mp_mixture, tau_curr, R


def pairwise_posteriors_tkf92_mixture_streaming(
        pairs_x, pairs_y, real_Lxs, real_Lys, tau_anchor,
        mix_ins, mix_del, mix_ext, mix_Q, mix_pi, mix_weights,
        pair_chunk=None):
    """Streaming-accumulate mixture FSA: never materialise the full
    (K, n_pairs, Lx, Ly) tensor.  Two scans over K:
      pass 1: compute log_p_K (K, n_pairs) — small.
      pass 2: accumulate R-weighted mp into running total.
    Memory: O(1 component's pair-marg + running total + log_p_K).

    Approach 1 (n_outer_rounds=0) only — uses anchored tau (no NR refit).

    Args:
        pairs_x, pairs_y: (n_pairs, Lx_pad), (n_pairs, Ly_pad).
        real_Lxs, real_Lys: (n_pairs,).
        tau_anchor: (n_pairs,) — fixed tau per pair (e.g. from single-TKF92 anchor).
        mix_ins, mix_del, mix_ext: (K,).
        mix_Q: (K, A, A); mix_pi: (K, A); mix_weights: (K,).

    Returns:
        mp_mixture: (n_pairs, Lx_pad, Ly_pad) — R-weighted posterior.
        R: (K,) — family responsibility.
    """
    K = mix_ins.shape[0]
    log_mix_weights = jnp.log(jnp.maximum(mix_weights, 1e-30))

    # Shape guard: tau_anchor must be per-pair (n_pairs,).  Passing a
    # scalar or family-mean would silently re-introduce the tau_rep bug.
    n_pairs = pairs_x.shape[0]
    if tau_anchor.shape != (n_pairs,):
        raise ValueError(
            f"tau_anchor must be per-pair shape ({n_pairs},); "
            f"got {tau_anchor.shape}.  Each pair needs its own anchor "
            f"tau_i — passing a scalar or family-mean would re-introduce "
            f"the tau_rep regression.")

    fb_per_pair = jax.vmap(
        _tkf92_fb_at_tau_one,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None))

    # Pair-chunked variant: process pairs in slices of pair_chunk
    # (sequential), vmap within each slice (parallel).  This bounds the
    # peak per-component working set to ~pair_chunk × per-pair-tensor
    # rather than n_pairs × per-pair-tensor.  Returns only (log_p, mp)
    # — n_trans and match_W are not used by the streaming caller and
    # accumulating them would waste compute on per-chunk concatenation.
    if pair_chunk is None or pair_chunk >= n_pairs:
        def fb_for_one_K(ins_k, del_k, ext_k, Q_k, pi_k):
            log_p, mp, _nt, _mw = fb_per_pair(
                pairs_x, pairs_y, real_Lxs, real_Lys, tau_anchor,
                ins_k, del_k, ext_k, Q_k, pi_k)
            return log_p, mp
    else:
        def fb_for_one_K(ins_k, del_k, ext_k, Q_k, pi_k):
            log_p_chunks = []
            mp_chunks = []
            for s in range(0, n_pairs, pair_chunk):
                e = min(s + pair_chunk, n_pairs)
                lp_c, mp_c, _nt, _mw = fb_per_pair(
                    pairs_x[s:e], pairs_y[s:e],
                    real_Lxs[s:e], real_Lys[s:e],
                    tau_anchor[s:e], ins_k, del_k, ext_k, Q_k, pi_k)
                log_p_chunks.append(lp_c)
                mp_chunks.append(mp_c)
            return (jnp.concatenate(log_p_chunks, axis=0),
                    jnp.concatenate(mp_chunks, axis=0))

    # Pass 1: per-component log_p only.  Loop over K explicitly (Python
    # loop, not lax.map) so each component is fully released between
    # iterations.  We keep only the (K, n_pairs) log_p stack in memory.
    log_p_KP = []
    for k in range(K):
        lp, _mp = fb_for_one_K(
            mix_ins[k], mix_del[k], mix_ext[k], mix_Q[k], mix_pi[k])
        log_p_KP.append(lp)
    log_p_KP = jnp.stack(log_p_KP, axis=0)

    # Family responsibility from joint LL across pairs.
    log_w_k = log_mix_weights + jnp.sum(log_p_KP, axis=1)
    R = jax.nn.softmax(log_w_k)

    # Pass 2: accumulate R-weighted mp via Python loop over K.
    Lx_pad = pairs_x.shape[1]
    Ly_pad = pairs_y.shape[1]
    mp_mixture = jnp.zeros((n_pairs, Lx_pad, Ly_pad), dtype=jnp.float64)
    for k in range(K):
        _lp, mp = fb_for_one_K(
            mix_ins[k], mix_del[k], mix_ext[k], mix_Q[k], mix_pi[k])
        mp_mixture = mp_mixture + R[k] * mp
    return mp_mixture, R


def pairwise_posteriors_distilled(x_seq, y_seq, maraschino_params, n_classes,
                                   precomp=None, n_newton=3, tau_init=1.0):
    """Compute pairwise residue alignment posteriors using distilled order-1 model.

    Uses the effective_pair_hmm from algebraic distillation (maraschino)
    to build a 5-state pair HMM with domain-mixture emissions, then runs
    Forward-Backward to extract match posteriors.

    This is a diagnostic tool for evaluating whether the distilled model
    retains alignment-relevant information. The transitions come from the
    emission-marginalized order-1 WFST (context-free approximation), and
    the emissions use the full domain-mixture substitution model.

    Args:
        x_seq, y_seq: (Lx,), (Ly,) integer sequence arrays
        maraschino_params: constrained parameter dict from maraschino.load_params
        n_classes: number of gamma rate classes
        precomp: optional precomputed eigendecompositions
        n_newton: Newton-Raphson steps for tau optimization
        tau_init: initial evolutionary time

    Returns:
        match_posteriors: (Lx, Ly) P(residue i aligned to residue j)
        tau_opt: optimized evolutionary time
        log_prob: log-probability at optimal tau
    """
    from ..distill.maraschino import (
        effective_pair_hmm, precompute_mixdom, distill_mixdom,
        normalize_freqs_hmm,
    )
    from .guide_tree import estimate_pairwise_distance

    x_seq = jnp.asarray(x_seq)
    y_seq = jnp.asarray(y_seq)
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]

    if precomp is None:
        precomp = precompute_mixdom(maraschino_params, n_classes)

    # Step 1: estimate tau using TKF92 approximation (fast)
    # INTENTIONAL: average across domains to obtain a single TKF92 proxy
    # for the per-pair tau anchor. This is a diagnostic-only function;
    # downstream uses do NOT consume the averaged rates. DO NOT copy this
    # pattern into mixture-aware code paths -- there each component must
    # retain its own (lam, mu, r) per the rigor-auditor invariants in the
    # mixture pipeline (see CRITICAL design invariants block above).
    avg_lam = float(jnp.mean(maraschino_params['lam']))
    avg_mu = float(jnp.mean(maraschino_params['mu']))
    avg_ext = float(jnp.mean(maraschino_params['r']))

    from ..core.protein import rate_matrix_lg
    Q_lg, pi_lg = rate_matrix_lg()

    tau_opt, _ = estimate_pairwise_distance(
        x_seq, y_seq, avg_lam, avg_mu, avg_ext,
        Q_lg, pi_lg, n_newton=n_newton, tau_init=tau_init)

    # Step 2: get effective pair HMM at optimal tau
    eph = effective_pair_hmm(maraschino_params, n_classes, tau_opt, precomp)

    log_trans = eph['log_trans']            # (5, 5) [S, M, I, D, E]
    P_domains = eph['P_domains']            # (N, AA, AA)
    pis = eph['pis']                        # (N, AA)
    domain_weights = eph['domain_weights']  # (N,)
    pi_mix = eph['pi_mix']                  # (AA,)

    # Step 3: build emission table
    # Match: log sum_n w_n * pi_n[a] * P_n[a, b]
    # Insert: log pi_mix[b]
    # Delete: log pi_mix[a]
    state_types = jnp.array([S, M, I, D, E])

    # Domain-mixture match emission: (AA, AA)
    e_M = jnp.einsum('n,na,nab->ab', domain_weights, pis, P_domains)
    log_e_M = safe_log(e_M)  # (AA, AA)

    log_pi_mix = safe_log(pi_mix)

    # Build emission table: (Lx+1, Ly+1, 5)
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)
    x_pad = _pad_seq(x_seq, Lx_pad)
    y_pad = _pad_seq(y_seq, Ly_pad)

    x_padded = jnp.concatenate([jnp.array([0]), x_pad])  # (Lx_pad+1,)
    y_padded = jnp.concatenate([jnp.array([0]), y_pad])  # (Ly_pad+1,)

    match_emit = log_e_M[x_padded[:, None], y_padded[None, :]]  # (Lx_pad+1, Ly_pad+1)
    ins_emit = log_pi_mix[y_padded]    # (Ly_pad+1,)
    del_emit = log_pi_mix[x_padded]    # (Lx_pad+1,)

    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)
    is_emit = is_M | is_I | is_D

    emit = (is_M[None, None, :] * match_emit[:, :, None] +
            is_I[None, None, :] * ins_emit[None, :, None] +
            is_D[None, None, :] * del_emit[:, None, None])
    emit = jnp.where(is_emit[None, None, :], emit, 0.0)

    # Mask padded positions
    from ..dp.hmm import _emit_mask
    mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, 5)
    emit = jnp.where(mask, emit, NEG_INF)

    # Step 4: Forward-Backward
    _, F_pad = _forward_2d_core(log_trans, state_types, emit, Lx_pad, Ly_pad)
    B = _backward_2d_core(log_trans, state_types, emit, Lx, Ly)

    e_idx = _find_e_idx(state_types)
    log_prob = jax.nn.logsumexp(F_pad[Lx, Ly, :] + log_trans[:, e_idx])
    F = F_pad[:Lx + 1, :Ly + 1, :]

    posteriors = jnp.exp(F + B - log_prob)

    # Sum over M-type states
    match_posteriors = jnp.sum(
        posteriors[1:Lx+1, 1:Ly+1, :] * is_M[None, None, :],
        axis=-1)

    return np.asarray(match_posteriors), float(tau_opt), float(log_prob)


def pairwise_posteriors_wfst(x_seq, y_seq, maraschino_params, n_classes,
                              precomp=None, n_newton=3, tau_init=1.0):
    """Compute pairwise posteriors using full order-1 WFST (context-dependent).

    Uses a custom Forward-Backward that works directly with the HMM-normalized
    order-1 distilled WFST tensors, tracking (pair_state, ancestor_context,
    descendant_context) at each cell. State space per cell: 3*AA*AA = 1200.

    This is a diagnostic tool to determine whether the order-1 WFST retains
    alignment-relevant information. If FSA with these posteriors gives good
    SP scores, then the ProgRec compose_intersect has a bug; if SP is near
    zero, then the distillation itself loses information.

    Args:
        x_seq, y_seq: (Lx,), (Ly,) integer sequence arrays
        maraschino_params: constrained parameter dict from maraschino.load_params
        n_classes: number of gamma rate classes
        precomp: optional precomputed eigendecompositions
        n_newton: Newton-Raphson steps for tau optimization
        tau_init: initial evolutionary time

    Returns:
        match_posteriors: (Lx, Ly) P(residue i aligned to residue j)
        tau_opt: optimized evolutionary time
        log_prob: log-probability at optimal tau
    """
    from ..distill.maraschino import (
        distill_mixdom, precompute_mixdom, normalize_freqs_hmm, AA,
    )
    from .guide_tree import estimate_pairwise_distance
    from scipy.special import logsumexp as sp_logsumexp

    x_seq = jnp.asarray(x_seq)
    y_seq = jnp.asarray(y_seq)
    Lx = int(x_seq.shape[0])
    Ly = int(y_seq.shape[0])

    if precomp is None:
        precomp = precompute_mixdom(maraschino_params, n_classes)

    # Step 1: estimate tau using TKF92 approximation
    avg_lam = float(jnp.mean(maraschino_params['lam']))
    avg_mu = float(jnp.mean(maraschino_params['mu']))
    avg_ext = float(jnp.mean(maraschino_params['r']))

    from ..core.protein import rate_matrix_lg
    Q_lg, pi_lg = rate_matrix_lg()

    tau_opt, _ = estimate_pairwise_distance(
        x_seq, y_seq, avg_lam, avg_mu, avg_ext,
        Q_lg, pi_lg, n_newton=n_newton, tau_init=tau_init)

    # Step 2: distill at optimal tau and normalize as HMM
    dist = distill_mixdom(maraschino_params, tau_opt, n_classes, precomp)
    hmm = normalize_freqs_hmm(dist)

    # Step 3: custom Forward-Backward with compact WFST representation
    # Forward table: F[i, j] is a (3, AA, AA) array in log space
    # where F[i,j,s,a,b] = log P(x_1..x_i, y_1..y_j, last_state=(s,a,b))
    # s: 0=M, 1=I, 2=D

    # Extract log transition tensors
    NI = float(NEG_INF)
    lpmm = np.asarray(safe_log(jnp.asarray(hmm['p_mm'])))  # (AA,AA,AA,AA)
    lpmi = np.asarray(safe_log(jnp.asarray(hmm['p_mi'])))  # (AA,AA,AA)
    lpmd = np.asarray(safe_log(jnp.asarray(hmm['p_md'])))  # (AA,AA,AA)
    lpme = np.asarray(safe_log(jnp.asarray(hmm['p_me'])))  # (AA,AA)

    lpim = np.asarray(safe_log(jnp.asarray(hmm['p_im'])))  # (AA,AA,AA,AA)
    lpii = np.asarray(safe_log(jnp.asarray(hmm['p_ii'])))  # (AA,AA,AA)
    lpid = np.asarray(safe_log(jnp.asarray(hmm['p_id'])))  # (AA,AA,AA)
    lpie = np.asarray(safe_log(jnp.asarray(hmm['p_ie'])))  # (AA,AA)

    lpdm = np.asarray(safe_log(jnp.asarray(hmm['p_dm'])))  # (AA,AA,AA,AA)
    lpdi = np.asarray(safe_log(jnp.asarray(hmm['p_di'])))  # (AA,AA,AA)
    lpdd = np.asarray(safe_log(jnp.asarray(hmm['p_dd'])))  # (AA,AA,AA)
    lpde = np.asarray(safe_log(jnp.asarray(hmm['p_de'])))  # (AA,AA)

    lpsm = np.asarray(safe_log(jnp.asarray(hmm['p_sm'])))  # (AA,AA)
    lpsi = np.asarray(safe_log(jnp.asarray(hmm['p_si'])))  # (AA,AA)
    lpsd = np.asarray(safe_log(jnp.asarray(hmm['p_sd'])))  # (AA,AA)
    lpse = float(safe_log(jnp.asarray(np.float64(hmm['p_se']))))

    x_np = np.asarray(x_seq)
    y_np = np.asarray(y_seq)

    # Forward table: (Lx+1, Ly+1, 3, AA, AA)
    F = np.full((Lx + 1, Ly + 1, 3, AA, AA), NI)

    # Forward pass
    # Row 0: only I-type transitions (no x consumed)
    for j in range(1, Ly + 1):
        bj = y_np[j - 1]
        if j == 1:
            # S -> I(a_pass, bj)
            F[0, 1, 1, :, bj] = np.maximum(F[0, 1, 1, :, bj], lpsi[:, bj])
        if j >= 2:
            # I(a, b_prev) -> I(a, bj)
            for a in range(AA):
                vals = F[0, j - 1, 1, a, :] + lpii[a, :, bj]
                F[0, j, 1, a, bj] = sp_logsumexp(vals)

    # Rows 1..Lx
    for i in range(1, Lx + 1):
        ai = x_np[i - 1]

        # Column 0: only D-type transitions
        if i == 1:
            F[1, 0, 2, ai, :] = np.maximum(F[1, 0, 2, ai, :], lpsd[ai, :])
        if i >= 2:
            for b in range(AA):
                vals = F[i - 1, 0, 2, :, b] + lpdd[:, b, ai]
                F[i, 0, 2, ai, b] = sp_logsumexp(vals)

        for j in range(1, Ly + 1):
            bj = y_np[j - 1]

            # --- M-update: predecessor at (i-1, j-1) ---
            m_vals = []
            if i == 1 and j == 1:
                m_vals.append(lpsm[ai, bj])
            # From M, I, D sources at (i-1, j-1)
            src = F[i - 1, j - 1]  # (3, AA, AA)
            v_m = src[0, :, :] + lpmm[:, :, ai, bj]  # (AA, AA)
            m_vals.append(sp_logsumexp(v_m))
            v_i = src[1, :, :] + lpim[:, :, ai, bj]
            m_vals.append(sp_logsumexp(v_i))
            v_d = src[2, :, :] + lpdm[:, :, ai, bj]
            m_vals.append(sp_logsumexp(v_d))
            F[i, j, 0, ai, bj] = sp_logsumexp(np.array(m_vals))

            # --- I-update: predecessor at (i, j-1) ---
            src_ij1 = F[i, j - 1]  # (3, AA, AA)
            # Vectorized over a: for each a, sum over b
            # M(a,b) -> I(a, bj): lpmi[a, b, bj]
            contrib_m = sp_logsumexp(src_ij1[0, :, :] + lpmi[:, :, bj], axis=1)  # (AA,)
            contrib_i = sp_logsumexp(src_ij1[1, :, :] + lpii[:, :, bj], axis=1)  # (AA,)
            contrib_d = sp_logsumexp(src_ij1[2, :, :] + lpdi[:, :, bj], axis=1)  # (AA,)
            # Stack and logsumexp over sources for each a
            stacked = np.stack([contrib_m, contrib_i, contrib_d], axis=0)  # (3, AA)
            F[i, j, 1, :, bj] = sp_logsumexp(stacked, axis=0)  # (AA,)

            # --- D-update: predecessor at (i-1, j) ---
            src_i1j = F[i - 1, j]  # (3, AA, AA)
            # Vectorized over b: for each b, sum over a
            contrib_m = sp_logsumexp(src_i1j[0, :, :] + lpmd[:, :, ai], axis=0)  # (AA,)
            contrib_i = sp_logsumexp(src_i1j[1, :, :] + lpid[:, :, ai], axis=0)  # (AA,)
            contrib_d = sp_logsumexp(src_i1j[2, :, :] + lpdd[:, :, ai], axis=0)  # (AA,)
            stacked = np.stack([contrib_m, contrib_i, contrib_d], axis=0)  # (3, AA)
            F[i, j, 2, ai, :] = sp_logsumexp(stacked, axis=0)  # (AA,)

    # Terminal: log_prob = logsumexp over all states at (Lx, Ly) transitioning to E
    # S->E is only valid at (0,0) but we don't track S in the F table.
    # For Lx>0 or Ly>0, the empty alignment contributes nothing.
    term_vals = []
    term_m = sp_logsumexp(F[Lx, Ly, 0, :, :] + lpme)
    term_vals.append(term_m)
    term_i = sp_logsumexp(F[Lx, Ly, 1, :, :] + lpie)
    term_vals.append(term_i)
    term_d = sp_logsumexp(F[Lx, Ly, 2, :, :] + lpde)
    term_vals.append(term_d)
    # Only include S->E for empty sequences
    if Lx == 0 and Ly == 0:
        term_vals.append(lpse)
    log_prob = sp_logsumexp(np.array(term_vals))

    # Backward pass: B[i,j,s,a,b] = log P(rest, end | state=(s,a,b) at (i,j))
    B = np.full((Lx + 1, Ly + 1, 3, AA, AA), NI)

    # Terminal
    B[Lx, Ly, 0, :, :] = lpme
    B[Lx, Ly, 1, :, :] = lpie
    B[Lx, Ly, 2, :, :] = lpde

    # Last row (i=Lx): only I-type successors
    for j in range(Ly - 1, -1, -1):
        if j >= Ly:
            continue
        bj1 = y_np[j]  # char at position j+1 (0-indexed)
        # s(a,b) -> I(a, bj1): passthrough a, new b'=bj1
        b_succ = B[Lx, j + 1, 1, :, bj1]  # (AA,) indexed by a
        for b in range(AA):
            B[Lx, j, 0, :, b] = np.logaddexp(B[Lx, j, 0, :, b], lpmi[:, b, bj1] + b_succ)
            B[Lx, j, 1, :, b] = np.logaddexp(B[Lx, j, 1, :, b], lpii[:, b, bj1] + b_succ)
            B[Lx, j, 2, :, b] = np.logaddexp(B[Lx, j, 2, :, b], lpdi[:, b, bj1] + b_succ)

    # Last column (j=Ly): only D-type successors
    for i in range(Lx - 1, -1, -1):
        if i >= Lx:
            continue
        ai1 = x_np[i]  # char at position i+1 (0-indexed)
        # s(a,b) -> D(ai1, b): passthrough b, new a'=ai1
        b_succ = B[i + 1, Ly, 2, ai1, :]  # (AA,) indexed by b
        for a in range(AA):
            B[i, Ly, 0, a, :] = np.logaddexp(B[i, Ly, 0, a, :], lpmd[a, :, ai1] + b_succ)
            B[i, Ly, 1, a, :] = np.logaddexp(B[i, Ly, 1, a, :], lpid[a, :, ai1] + b_succ)
            B[i, Ly, 2, a, :] = np.logaddexp(B[i, Ly, 2, a, :], lpdd[a, :, ai1] + b_succ)

    # Interior cells (vectorized over a,b)
    for i in range(Lx - 1, -1, -1):
        for j in range(Ly - 1, -1, -1):
            ai1 = x_np[i] if i < Lx else -1
            bj1 = y_np[j] if j < Ly else -1

            if ai1 >= 0 and bj1 >= 0:
                # M-type successor at (i+1, j+1): fully vectorized
                b_m = B[i + 1, j + 1, 0, ai1, bj1]  # scalar
                B[i, j, 0] = np.logaddexp(B[i, j, 0], lpmm[:, :, ai1, bj1] + b_m)
                B[i, j, 1] = np.logaddexp(B[i, j, 1], lpim[:, :, ai1, bj1] + b_m)
                B[i, j, 2] = np.logaddexp(B[i, j, 2], lpdm[:, :, ai1, bj1] + b_m)

            if bj1 >= 0:
                # I-type successor at (i, j+1): vectorized
                b_i = B[i, j + 1, 1, :, bj1]  # (AA,) indexed by a
                # lpmi[:, :, bj1] is (AA, AA) = (a, b), b_i is (AA,) = (a,)
                # Result B[i,j,s,a,b] += lpmi[a, b, bj1] + b_i[a]
                B[i, j, 0] = np.logaddexp(B[i, j, 0], lpmi[:, :, bj1] + b_i[:, None])
                B[i, j, 1] = np.logaddexp(B[i, j, 1], lpii[:, :, bj1] + b_i[:, None])
                B[i, j, 2] = np.logaddexp(B[i, j, 2], lpdi[:, :, bj1] + b_i[:, None])

            if ai1 >= 0:
                # D-type successor at (i+1, j): vectorized
                b_d = B[i + 1, j, 2, ai1, :]  # (AA,) indexed by b
                # lpmd[:, :, ai1] is (AA, AA) = (a, b), b_d is (AA,) = (b,)
                B[i, j, 0] = np.logaddexp(B[i, j, 0], lpmd[:, :, ai1] + b_d[None, :])
                B[i, j, 1] = np.logaddexp(B[i, j, 1], lpid[:, :, ai1] + b_d[None, :])
                B[i, j, 2] = np.logaddexp(B[i, j, 2], lpdd[:, :, ai1] + b_d[None, :])

    # Extract match posteriors: vectorized
    fb_m = F[1:Lx+1, 1:Ly+1, 0, :, :] + B[1:Lx+1, 1:Ly+1, 0, :, :]  # (Lx, Ly, AA, AA)
    match_posteriors = np.exp(sp_logsumexp(fb_m.reshape(Lx, Ly, -1), axis=2) - log_prob)

    return match_posteriors, float(tau_opt), float(log_prob)


def pairwise_posteriors_mixdom(x_seq, y_seq, params, n_dom, n_frag,
                                n_newton=5, tau_init=1.0):
    """Compute pairwise residue alignment posteriors using MixDom.

    Args:
        x_seq, y_seq: (Lx,), (Ly,) integer sequence arrays
        params: MixDom parameter dict with keys:
            main_ins, main_del, dom_ins, dom_del, dom_weights,
            frag_weights, ext_rates, Q, pi
        n_dom, n_frag: model dimensions
        n_newton: Newton-Raphson steps for tau optimization
        tau_init: initial evolutionary time

    Returns:
        match_posteriors: (Lx, Ly) P(residue i aligned to residue j)
        tau_opt: optimized evolutionary time
        log_prob: log-probability at optimal tau
    """
    # Geometric-bin padding: every distinct (Lx, Ly) pair triggers a
    # JIT recompile, and the cumulative CUBIN cache fills GPU memory
    # within ~10 family alignments. Pad to the next geometric bin so
    # JAX reuses compiled functions across pairs of similar length.
    from ..dp.hmm import _pad_to_bin, _pad_seq
    x_j = jnp.asarray(x_seq)
    y_j = jnp.asarray(y_seq)
    real_Lx = int(x_j.shape[0])
    real_Ly = int(y_j.shape[0])
    Lx_pad = _pad_to_bin(real_Lx)
    Ly_pad = _pad_to_bin(real_Ly)
    x_padded = _pad_seq(x_j, Lx_pad)
    y_padded = _pad_seq(y_j, Ly_pad)
    mp, tau, lp = _pairwise_posteriors_mixdom_jax(
        x_padded, y_padded,
        jnp.int32(real_Lx), jnp.int32(real_Ly),
        params, n_dom, n_frag,
        n_newton=n_newton, tau_init=tau_init)
    # Trim posterior back to (real_Lx, real_Ly).
    mp_real = np.asarray(mp)[:real_Lx, :real_Ly]
    return mp_real, float(tau), float(lp)


def _pairwise_posteriors_mixdom_jax(x, y, real_Lx, real_Ly, params, n_dom, n_frag,
                                     n_newton=5, tau_init=1.0):
    """vmap-safe core of pairwise_posteriors_mixdom.

    x, y: jnp int arrays of shape (Lx,), (Ly,). Under vmap they will be the
    PADDED-bin sizes (all batch elements in a bucket share the same shape).
    real_Lx, real_Ly: traced jnp scalars giving the actual (unpadded)
    sequence lengths. Used to mask padded emission cells to NEG_INF and to
    extract log_prob at the real endpoint. Pass jnp.int32(x.shape[0]) for
    the scalar (unpadded) case.

    Returns jnp arrays: (match_posteriors (Lx, Ly), tau_opt (), log_prob ()).
    Does NOT call np.asarray or float() — safe for jax.vmap.
    """
    from ..models.mixdom import build_nested_trans, state_types as mixdom_state_types
    from ..dp.hmm import pair_hmm_emissions_per_domain

    st = mixdom_state_types(n_dom, n_frag)
    Lx = x.shape[0]
    Ly = y.shape[0]
    is_match = (st == M)

    # Step 1: tau optimization via FB counts + NR (returns jnp scalar).
    tau_opt = _optimize_tau_mixdom(
        x, y, real_Lx, real_Ly, params, n_dom, n_frag,
        n_newton=n_newton, tau_init=tau_init)

    # Step 2: FB at optimal tau for per-domain posteriors.
    chi_opt, _ = build_nested_trans(
        params['main_ins'], params['main_del'], tau_opt,
        jnp.array(params['dom_ins']), jnp.array(params['dom_del']),
        jnp.array(params['dom_weights']),
        jnp.array(params['frag_weights']),
        jnp.array(params['ext_rates']))
    log_chi_opt = safe_log(chi_opt)

    # Per-class emission when MixDom2 class data is present (Annabel-style).
    # Otherwise fall back to per-domain class-marginal emission.
    if all(k in params for k in ('class_pi', 'class_S_exch', 'class_dist')):
        from ..dp.hmm import pair_hmm_emissions_per_class
        from ..core.ctmc import build_Q_from_S_pi as build_rate_matrix
        cpi = jnp.asarray(params['class_pi'])
        cS = jnp.asarray(params['class_S_exch'])
        cdist = jnp.asarray(params['class_dist'])
        # Build per-class P(t) at tau_opt.
        class_subs = jax.vmap(
            lambda S, pi: transition_matrix(build_rate_matrix(S, pi), tau_opt))(cS, cpi)
        log_emit = pair_hmm_emissions_per_class(
            st, x, y, class_subs, cpi, cdist, n_dom, n_frag)
    else:
        sub_matrices, pis = build_per_domain_sub_matrices(params, tau_opt, n_dom)
        log_emit = pair_hmm_emissions_per_domain(
            st, x, y, sub_matrices, pis, n_dom, n_frag)
    log_prob, posteriors, _ = forward_backward_2d(
        log_chi_opt, st, x, y, None, None, log_emit_table=log_emit,
        real_Lx=real_Lx, real_Ly=real_Ly)

    match_posteriors = jnp.sum(
        posteriors[1:Lx + 1, 1:Ly + 1, :] * is_match[None, None, :],
        axis=-1)
    return match_posteriors, tau_opt, log_prob


def pairwise_posteriors_mixdom_batched(xs, ys, real_Lxs, real_Lys,
                                        params, n_dom, n_frag,
                                        n_newton=5, tau_init=1.0):
    """Batched (vmap'd) pairwise posteriors for MixDom.

    xs, ys: jnp int arrays of shape (B, Lx_pad), (B, Ly_pad). All batch
    elements MUST share the same padded shape (caller buckets by padded shape).
    real_Lxs, real_Lys: (B,) jnp int32 arrays of unpadded lengths per pair.

    Returns (match_posteriors (B, Lx_pad, Ly_pad), tau (B,), log_prob (B,)).
    Padded cells have match_posterior == 0.
    """
    return jax.vmap(
        lambda x, y, lx, ly: _pairwise_posteriors_mixdom_jax(
            x, y, lx, ly, params, n_dom, n_frag,
            n_newton=n_newton, tau_init=tau_init),
        in_axes=(0, 0, 0, 0),
    )(xs, ys, real_Lxs, real_Lys)


def _optimize_tau_mixdom(x_seq, y_seq, real_Lx, real_Ly, params, n_dom, n_frag,
                          n_newton=5, tau_init=1.0):
    """Optimize tau for MixDom via FB counts + Newton-Raphson with autograd.

    x_seq, y_seq may be padded to bin sizes; real_Lx, real_Ly give the
    unpadded lengths for emission masking and endpoint extraction.

    1. Run one FB pass at tau_init to get expected transition and emission counts
    2. Define E[LL(tau)] = sum n_trans * log chi(tau) + sum post * log emit(tau)
       where chi(tau) and emit(tau) are differentiable functions of tau
    3. Maximize E[LL(tau)] w.r.t. tau via Newton-Raphson using jax.grad
    """
    from ..models.mixdom import build_nested_trans, state_types as mixdom_state_types
    from ..core.ctmc import transition_matrix
    from ..dp.hmm import pair_hmm_emissions_per_domain
    from ..core.ctmc import build_Q_from_S_pi as build_rate_matrix

    st = mixdom_state_types(n_dom, n_frag)
    ns = len(st)
    x = jnp.asarray(x_seq)
    y = jnp.asarray(y_seq)
    Lx = x.shape[0]
    Ly = y.shape[0]

    # Pre-convert params
    main_ins = jnp.float64(params['main_ins'])
    main_del = jnp.float64(params['main_del'])
    dom_ins = jnp.array(params['dom_ins'], dtype=jnp.float64)
    dom_del = jnp.array(params['dom_del'], dtype=jnp.float64)
    dom_weights = jnp.array(params['dom_weights'], dtype=jnp.float64)
    frag_weights = jnp.array(params['frag_weights'], dtype=jnp.float64)
    ext_rates = jnp.array(params['ext_rates'], dtype=jnp.float64)

    S_exch_arr = jnp.array(params['S_exch'], dtype=jnp.float64)
    pis_arr = jnp.array(params['pi'], dtype=jnp.float64)
    if pis_arr.ndim == 1:
        pis_arr = jnp.tile(pis_arr[None], (n_dom, 1))

    # Step 1: FB at tau_init to get counts
    tau0 = jnp.float64(tau_init)
    chi0, _ = build_nested_trans(
        main_ins, main_del, tau0,
        dom_ins, dom_del, dom_weights, frag_weights, ext_rates)
    log_chi0 = safe_log(chi0)

    if S_exch_arr.ndim == 3:
        sub0 = jax.vmap(
            lambda S, pi: transition_matrix(build_rate_matrix(S, pi), tau0)
        )(S_exch_arr, pis_arr)
    else:
        sub0 = jax.vmap(
            lambda pi: transition_matrix(build_rate_matrix(S_exch_arr, pi), tau0)
        )(pis_arr)

    emit0 = pair_hmm_emissions_per_domain(
        st, x, y, sub0, pis_arr, n_dom, n_frag)
    _, posteriors0, n_trans0 = forward_backward_2d(
        log_chi0, st, x, y, None, None, log_emit_table=emit0,
        real_Lx=real_Lx, real_Ly=real_Ly)

    # Fix the counts (detach from computation graph)
    n_trans_fixed = jax.lax.stop_gradient(n_trans0)  # (ns, ns)
    post_fixed = jax.lax.stop_gradient(posteriors0)   # (Lx+1, Ly+1, ns)

    # Step 2: Reduce FB counts to sufficient statistics for tau optimization
    # (pure JAX, stays on device — no host transfers).
    A = int(pis_arr.shape[-1])
    st_j = jnp.asarray(st)
    is_M_j = (st_j == 1).astype(jnp.float64)  # (ns,)
    is_I_j = (st_j == 2).astype(jnp.float64)
    is_D_j = (st_j == 3).astype(jnp.float64)
    # domain of body state s (s>=2): (s-2) // (5*n_frag); SS/EE pinned to 0
    body_idx = jnp.maximum((jnp.arange(ns) - 2) // (5 * n_frag), 0)
    state_dom = jax.nn.one_hot(body_idx, n_dom, dtype=jnp.float64)  # (ns, n_dom)

    # One-hot of x, y into A columns; wildcard (index A) → all-zero row.
    X_oh = jax.nn.one_hot(x, A, dtype=jnp.float64)  # (Lx, A)
    Y_oh = jax.nn.one_hot(y, A, dtype=jnp.float64)  # (Ly, A)

    # Match counts: (n_dom, A, A). Split into three small einsums
    # rather than one 4-index contraction — cuBLAS autotuner chokes on
    # the 4D contraction under vmap when the product of batch x Lx x Ly
    # is large.
    mask_M = state_dom * is_M_j[:, None]                   # (ns, n_dom)
    post_MM = post_fixed[1:Lx+1, 1:Ly+1, :]                # (Lx, Ly, ns)
    W_M = jnp.einsum('ijs,sd->dij', post_MM, mask_M)       # (n_dom, Lx, Ly)
    # (d, a, j) = sum_i W_M[d, i, j] * X_oh[i, a]
    tmp_M = jnp.einsum('dij,ia->daj', W_M, X_oh)           # (n_dom, A, Ly)
    match_counts = jnp.einsum('daj,jb->dab', tmp_M, Y_oh)  # (n_dom, A, A)

    # Insert counts: (n_dom, A), sum over i∈[0,Lx] then project via y
    mask_I = state_dom * is_I_j[:, None]
    post_I = post_fixed[0:Lx+1, 1:Ly+1, :]                 # (Lx+1, Ly, ns)
    I_per_j = jnp.einsum('ijs,sd->dj', post_I, mask_I)     # (n_dom, Ly)
    insert_counts = jnp.einsum('dj,ja->da', I_per_j, Y_oh)

    # Delete counts: (n_dom, A), sum over j∈[0,Ly] then project via x
    mask_D = state_dom * is_D_j[:, None]
    post_D = post_fixed[1:Lx+1, 0:Ly+1, :]                 # (Lx, Ly+1, ns)
    D_per_i = jnp.einsum('ijs,sd->di', post_D, mask_D)     # (n_dom, Lx)
    delete_counts = jnp.einsum('di,ia->da', D_per_i, X_oh)

    # Detach all counts
    match_counts = jax.lax.stop_gradient(match_counts)
    insert_counts = jax.lax.stop_gradient(insert_counts)
    delete_counts = jax.lax.stop_gradient(delete_counts)

    # Step 3: NR via module-level JIT-cached grad/hess (same trick as
    # the TKF92 path: explicit args, no per-pair closures).
    # Tile S_exch to ndim==3 for a uniform cache key.
    if S_exch_arr.ndim == 2:
        S_exch_arr_3d = jnp.tile(S_exch_arr[None], (n_dom, 1, 1))
    else:
        S_exch_arr_3d = S_exch_arr

    log_tau = jnp.log(jnp.float64(tau_init))
    for _ in range(n_newton):
        g = _mixdom_tau_grad(
            log_tau, n_trans_fixed, match_counts, insert_counts, delete_counts,
            main_ins, main_del, dom_ins, dom_del, dom_weights,
            frag_weights, ext_rates, S_exch_arr_3d, pis_arr)
        h = _mixdom_tau_hess(
            log_tau, n_trans_fixed, match_counts, insert_counts, delete_counts,
            main_ins, main_del, dom_ins, dom_del, dom_weights,
            frag_weights, ext_rates, S_exch_arr_3d, pis_arr)
        # Always run full n_newton steps. Guard against degenerate Hessians
        # via where() (no Python branch — vmap-safe).
        safe_neg_h = jnp.where(jnp.abs(h) > 1e-10, -h, 1.0)
        step = jnp.clip(g / safe_neg_h, -1.0, 1.0)
        log_tau = log_tau + step

    # Return a jnp scalar (no float() host sync) — vmap-safe.
    return jnp.exp(jnp.clip(log_tau, jnp.log(1e-4), jnp.log(10.0)))


# ============================================================
# Pair selection strategies
# ============================================================

def select_pairs_full(n_seqs):
    """Select all O(N^2) pairs."""
    pairs = []
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            pairs.append((i, j))
    return pairs


def select_pairs_erdos_renyi(n_seqs, rng=None, c=2.0):
    """Select O(N log N) random pairs (Erdos-Renyi random graph).

    Each pair is included independently with probability p = c * ln(N) / N.
    This ensures connectivity with high probability when c > 1.

    Args:
        n_seqs: number of sequences
        rng: numpy RandomState (default: seed 42)
        c: connectivity constant (default 2.0 for robust connectivity)

    Returns:
        list of (i, j) pairs with i < j
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if n_seqs <= 3:
        return select_pairs_full(n_seqs)

    p = c * np.log(n_seqs) / n_seqs
    p = min(p, 1.0)

    pairs = []
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            if rng.random() < p:
                pairs.append((i, j))

    # Ensure graph is connected: add minimum spanning chain
    connected = set()
    if pairs:
        connected.add(pairs[0][0])
        connected.add(pairs[0][1])
        for i, j in pairs:
            connected.add(i)
            connected.add(j)

    # Add edges to connect isolated nodes
    for i in range(n_seqs):
        if i not in connected:
            # Connect to nearest included node
            best_j = min(connected, key=lambda j: abs(j - i))
            pair = (min(i, best_j), max(i, best_j))
            if pair not in set(pairs):
                pairs.append(pair)
            connected.add(i)

    # Ensure at least one pair per sequence for good coverage
    node_degree = np.zeros(n_seqs, dtype=int)
    for i, j in pairs:
        node_degree[i] += 1
        node_degree[j] += 1
    for i in range(n_seqs):
        if node_degree[i] == 0:
            j = rng.randint(0, n_seqs)
            while j == i:
                j = rng.randint(0, n_seqs)
            pair = (min(i, j), max(i, j))
            if pair not in set(pairs):
                pairs.append(pair)

    return sorted(set(pairs))


# ============================================================
# Compute all pairwise posteriors
# ============================================================

def compute_pairwise_posteriors(sequences, pairs, model='tkf92',
                                 ins_rate=0.02, del_rate=0.05, ext=0.5,
                                 Q=None, pi=None, mixdom_params=None,
                                 n_dom=3, n_frag=2,
                                 maraschino_params=None, n_classes=4,
                                 maraschino_precomp=None,
                                 n_newton=5, tau_init=1.0, verbose=False):
    """Compute pairwise residue alignment posteriors for selected pairs.

    Args:
        sequences: dict of {name: integer_array}
        pairs: list of (i, j) index pairs
        model: 'tkf92', 'mixdom', 'distilled', or 'wfst'
        ins_rate, del_rate, ext: TKF92 parameters (used if model='tkf92')
        Q, pi: substitution model
        mixdom_params: MixDom parameter dict (used if model='mixdom')
        n_dom, n_frag: MixDom dimensions
        maraschino_params: constrained params from maraschino.load_params
            (used if model='distilled' or model='wfst')
        n_classes: number of gamma rate classes (used if model='distilled' or 'wfst')
        maraschino_precomp: precomputed eigendecompositions (optional)
        n_newton: Newton-Raphson steps for tau
        tau_init: initial tau
        verbose: print progress

    Returns:
        pair_posteriors: dict of {(i,j): (Lx, Ly) match posterior array}
        pair_taus: dict of {(i,j): optimized tau}
    """
    if Q is None or pi is None:
        from ..core.protein import rate_matrix_lg
        Q, pi = rate_matrix_lg()

    names = list(sequences.keys())
    pair_posteriors = {}
    pair_taus = {}

    # Precompute eigendecompositions once for distilled/wfst model
    if model in ('distilled', 'wfst') and maraschino_precomp is None:
        from ..distill.maraschino import precompute_mixdom
        maraschino_precomp = precompute_mixdom(maraschino_params, n_classes)

    # Note: each unique (Lx_pad, Ly_pad) pair triggers JIT compilation.
    # For proteins of similar length (~80aa), the geometric binning
    # groups most pairs into the same bin, limiting recompilations.

    for idx, (i, j) in enumerate(pairs):
        x_seq = jnp.asarray(sequences[names[i]])
        y_seq = jnp.asarray(sequences[names[j]])

        if verbose and (idx + 1) % 10 == 0:
            print(f"  Pair {idx+1}/{len(pairs)}: {names[i]} vs {names[j]}")

        if model == 'tkf92':
            mp, tau, lp = pairwise_posteriors_tkf92(
                x_seq, y_seq, ins_rate, del_rate, ext,
                Q, pi, n_newton=n_newton, tau_init=tau_init)
        elif model == 'mixdom':
            mp, tau, lp = pairwise_posteriors_mixdom(
                x_seq, y_seq, mixdom_params, n_dom, n_frag,
                n_newton=n_newton, tau_init=tau_init)
        elif model == 'distilled':
            mp, tau, lp = pairwise_posteriors_distilled(
                x_seq, y_seq, maraschino_params, n_classes,
                precomp=maraschino_precomp,
                n_newton=n_newton, tau_init=tau_init)
        elif model == 'wfst':
            mp, tau, lp = pairwise_posteriors_wfst(
                x_seq, y_seq, maraschino_params, n_classes,
                precomp=maraschino_precomp,
                n_newton=n_newton, tau_init=tau_init)
        else:
            raise ValueError(f"Unknown model: {model}")

        pair_posteriors[(i, j)] = mp
        pair_taus[(i, j)] = tau

    return pair_posteriors, pair_taus


# ============================================================
# Sequence annealing
# ============================================================

def _score_alignment(col_assignments, seq_lengths, pair_posteriors):
    """Score an alignment by sum of pairwise posterior probabilities.

    For each pair (i,j) with computed posteriors, sum the posterior
    probability of the implied residue pairings.

    Args:
        col_assignments: list of arrays, col_assignments[i][k] = column
        seq_lengths: list of sequence lengths
        pair_posteriors: dict {(i,j): (Li, Lj) posterior array}

    Returns:
        score: total alignment score
    """
    score = 0.0
    for (i, j), post in pair_posteriors.items():
        Li, Lj = post.shape
        # Build column -> residue mapping for each sequence
        col_to_res_i = {}
        for k in range(Li):
            col_to_res_i[col_assignments[i][k]] = k
        col_to_res_j = {}
        for k in range(Lj):
            col_to_res_j[col_assignments[j][k]] = k

        # Sum posteriors for aligned residue pairs
        for col in set(col_to_res_i.keys()) & set(col_to_res_j.keys()):
            ri = col_to_res_i[col]
            rj = col_to_res_j[col]
            score += post[ri, rj]

    return score


def sequence_annealing(n_seqs, seq_lengths, pair_posteriors,
                        n_iterations=5, verbose=False,
                        gap_factor=1.0, edge_weight_threshold=0.0,
                        seed=42):
    """Build MSA via AMAP sequence annealing (DAG column merging).

    Port of the AMAP algorithm from DART (MultiSequenceDag.h). Each residue
    starts in its own column. Candidate edges (column merges) are created
    from all pairwise posteriors above a threshold. Edges are processed in
    descending weight order using a priority queue. Before each merge:
      - Dynamic weight recalculation (TGF formula) accounts for merged columns
      - If recalculated weight is lower than the next edge, re-insert and retry
      - Same-sequence conflicts and DAG cycles are detected and rejected
    Column merging uses the Pearce-Kelly online topological ordering algorithm
    for cycle detection, matching the DART C++ implementation.

    The n_iterations parameter controls refinement sweeps after the initial
    AMAP build (remove each sequence and reinsert optimally). The initial
    AMAP merge is the primary alignment step.

    The refinement-sweep sequence-order is a random permutation seeded from
    ``seed``; different seeds typically produce different MSAs of varying
    quality on hard families. Call ``sequence_annealing_multi_seed`` for a
    multi-seed sweep with cached pair_posteriors.

    Args:
        n_seqs: number of sequences
        seq_lengths: list of sequence lengths
        pair_posteriors: dict {(i,j): (Li, Lj) posterior array}
        n_iterations: number of refinement sweeps after initial build
        verbose: print progress
        gap_factor: TGF gap factor (1.0 = AMA accuracy, >1 = fewer gaps)
        edge_weight_threshold: minimum edge weight to consider
        seed: int seed for the refinement-sweep permutation (default 42).

    Returns:
        col_assignments: list of arrays, col_assignments[i][k] = column
            assigned to residue k of sequence i
        msa_length: total number of columns in the alignment
    """
    col_assignments = _amap_align(
        n_seqs, seq_lengths, pair_posteriors,
        gap_factor=gap_factor,
        edge_weight_threshold=edge_weight_threshold,
        verbose=verbose)

    if all(len(ca) == 0 for ca in col_assignments):
        return col_assignments, 0

    n_cols = max(max(ca) for ca in col_assignments if len(ca) > 0) + 1

    if verbose:
        score = _score_alignment(col_assignments, seq_lengths, pair_posteriors)
        print(f"  AMAP build: {n_cols} cols, score={score:.4f}")

    rng_local = np.random.RandomState(seed)
    # Refinement: remove-and-reinsert each sequence
    for iteration in range(n_iterations):
        improved = False
        best_score = _score_alignment(
            col_assignments, seq_lengths, pair_posteriors)

        order = rng_local.permutation(n_seqs)
        for seq_idx in order:
            if seq_lengths[seq_idx] == 0:
                continue

            col_assignments_new, n_cols_new = _refine_one_sequence(
                col_assignments, seq_idx, n_seqs, seq_lengths,
                pair_posteriors)

            new_score = _score_alignment(
                col_assignments_new, seq_lengths, pair_posteriors)

            if new_score > best_score + 1e-10:
                col_assignments = col_assignments_new
                n_cols = n_cols_new
                best_score = new_score
                improved = True

        if verbose:
            print(f"  Refinement {iteration+1}: {n_cols} cols, "
                  f"score={best_score:.4f}, improved={improved}")

        if not improved:
            break

    return col_assignments, n_cols


def _amap_align(n_seqs, seq_lengths, pair_posteriors,
                gap_factor=1.0, edge_weight_threshold=0.0, verbose=False):
    """Core AMAP DAG column-merging algorithm.

    Port of MultiSequenceDag::AlignDag from DART. Uses TGF (transformed
    gap factor) edge weighting with dynamic recalculation and Pearce-Kelly
    online topological ordering for cycle detection.

    Args:
        n_seqs: number of sequences
        seq_lengths: list of sequence lengths
        pair_posteriors: dict {(i,j): (Li, Lj) posterior array}
        gap_factor: TGF gap factor
        edge_weight_threshold: minimum edge weight
        verbose: print progress

    Returns:
        col_assignments: list of arrays, col_assignments[i][k] = column
            assigned to residue k of sequence i
    """
    import heapq
    from collections import deque

    if sum(seq_lengths) == 0:
        return [np.array([], dtype=int) for _ in range(n_seqs)]

    # ================================================================
    # Data structures
    # ================================================================

    # Precompute gap posteriors for each (seq, position):
    # gap_post[i][ri] = 1 - sum_j P(ri ~ any residue in seq j)
    # In practice: gap_post[(i,j)][ri] = 1 - sum_rj post[ri, rj]
    # and gap_post[(j,i)][rj] = 1 - sum_ri post[ri, rj]
    # We store per-pair gap posteriors like AMAP's SparseMatrix.
    gap_post_0 = {}  # (i,j) -> array of length Li: gap posterior for seq i vs seq j
    gap_post_1 = {}  # (i,j) -> array of length Lj: gap posterior for seq j vs seq i

    for (i, j), post in pair_posteriors.items():
        Li, Lj = post.shape
        row_sums = np.sum(post, axis=1)  # sum over j for each ri
        col_sums = np.sum(post, axis=0)  # sum over i for each rj
        gp0 = np.maximum(1.0 - row_sums, 1e-4)
        gp1 = np.maximum(1.0 - col_sums, 1e-4)
        gap_post_0[(i, j)] = gp0
        gap_post_1[(i, j)] = gp1

    # Column data structures
    # col_seqs[col] = dict {seq_id: residue_pos} (1-based positions like AMAP)
    # seqpos_to_col[(seq, pos)] -> col_id (1-based positions)
    # col_index[col] = topological ordering index

    total_residues = sum(seq_lengths)
    col_seqs = {}       # col_id -> {seq: pos_1based}
    seqpos_to_col = {}  # (seq, pos_1based) -> col_id
    col_index = {}      # col_id -> topological index
    merged_into = {}    # col_id -> col_id (for tracking merges)

    col_id = 0
    idx = 0
    for si in range(n_seqs):
        for k in range(seq_lengths[si]):
            pos = k + 1  # 1-based like AMAP
            col_seqs[col_id] = {si: pos}
            seqpos_to_col[(si, pos)] = col_id
            col_index[col_id] = idx
            merged_into[col_id] = col_id
            col_id += 1
            idx += 1

    live_cols = set(range(total_residues))
    next_col_id = total_residues

    def find_col(c):
        """Follow merge chain to find current live column."""
        while merged_into[c] != c:
            # Path compression
            merged_into[c] = merged_into[merged_into[c]]
            c = merged_into[c]
        return c

    def get_successors(col):
        """Get successor columns via sequence ordering edges."""
        succs = set()
        for seq, pos in col_seqs[col].items():
            nxt_key = (seq, pos + 1)
            if nxt_key in seqpos_to_col:
                sc = find_col(seqpos_to_col[nxt_key])
                if sc != col and sc in live_cols:
                    succs.add(sc)
        return succs

    def get_predecessors(col):
        """Get predecessor columns via sequence ordering edges."""
        preds = set()
        for seq, pos in col_seqs[col].items():
            prev_key = (seq, pos - 1)
            if prev_key in seqpos_to_col:
                pc = find_col(seqpos_to_col[prev_key])
                if pc != col and pc in live_cols:
                    preds.add(pc)
        return preds

    # ================================================================
    # Pearce-Kelly cycle detection (from DART MultiSequenceDag)
    # ================================================================

    def dfs_forward(node, upper_bound, r_forward):
        """DFS forward from node, collecting visited nodes.
        Returns True if upper_bound is reachable (cycle detected)."""
        node_visited.add(node)
        r_forward.append(node)
        for seq, pos in col_seqs[node].items():
            nxt_key = (seq, pos + 1)
            if nxt_key not in seqpos_to_col:
                continue
            w = find_col(seqpos_to_col[nxt_key])
            if w not in live_cols or w == node:
                continue
            if col_index[w] == col_index[upper_bound]:
                return True  # cycle
            if w not in node_visited and col_index[w] < col_index[upper_bound]:
                if dfs_forward(w, upper_bound, r_forward):
                    return True
        return False

    def dfs_backward(node, lower_bound, r_backward):
        """DFS backward from node, collecting visited nodes."""
        node_visited.add(node)
        r_backward.append(node)
        for seq, pos in col_seqs[node].items():
            prev_key = (seq, pos - 1)
            if prev_key not in seqpos_to_col:
                continue
            w = find_col(seqpos_to_col[prev_key])
            if w not in live_cols or w == node:
                continue
            if w not in node_visited and col_index[lower_bound] < col_index[w]:
                dfs_backward(w, lower_bound, r_backward)

    def reorder(r_forward, r_backward):
        """Reorder column indices after edge addition (Pearce-Kelly)."""
        # Collect all indices
        all_indices = sorted(
            [col_index[c] for c in r_backward] +
            [col_index[c] for c in r_forward])

        # Assign: backward nodes get lower indices, forward get higher
        r_backward.sort(key=lambda c: col_index[c])
        r_forward.sort(key=lambda c: col_index[c])

        idx_iter = iter(all_indices)
        for c in r_backward:
            col_index[c] = next(idx_iter)
        for c in r_forward:
            col_index[c] = next(idx_iter)

    def try_add_edge(col1, col2):
        """Try to merge col1 and col2. Returns 0 on success, 1 if same-seq
        conflict, 2 if cycle detected."""
        nonlocal node_visited

        col1 = find_col(col1)
        col2 = find_col(col2)

        if col1 == col2:
            return 0  # already merged

        # Same-sequence conflict
        if set(col_seqs[col1].keys()) & set(col_seqs[col2].keys()):
            return 1

        # Determine lower and upper bound by topological index
        if col_index[col1] < col_index[col2]:
            l_bound, u_bound = col1, col2
        else:
            l_bound, u_bound = col2, col1

        # Cycle detection via DFS
        node_visited = set()
        r_forward = []
        if dfs_forward(l_bound, u_bound, r_forward):
            return 2  # cycle

        r_backward = []
        dfs_backward(u_bound, l_bound, r_backward)
        node_visited = set()  # clear for next use

        # Reorder if needed
        if len(r_forward) == 1:
            # l_bound has no forward deps in range, merge into u_bound
            col1, col2 = u_bound, l_bound
        elif len(r_backward) == 1:
            col1, col2 = l_bound, u_bound
        else:
            reorder(r_forward, r_backward)
            col1, col2 = l_bound, u_bound

        # Merge col2 into col1
        for seq, pos in list(col_seqs[col2].items()):
            col_seqs[col1][seq] = pos
            seqpos_to_col[(seq, pos)] = col1
        merged_into[col2] = col1
        live_cols.discard(col2)
        del col_seqs[col2]
        # col1 keeps its index

        return 0

    node_visited = set()

    # ================================================================
    # Compute TGF edge weight (from DART Edge::calcTgfWeight)
    # ================================================================

    def calc_tgf_weight(src_col, tgt_col):
        """Calculate TGF weight for merging two columns.

        Returns (weight, delta) where weight is the TGF ratio and
        delta is the expected accuracy improvement.
        Returns (INVALID, INVALID) if the edge is invalid (same seq).
        """
        src_col = find_col(src_col)
        tgt_col = find_col(tgt_col)

        if src_col == tgt_col:
            return -1e10, -1e10

        c1pos = col_seqs.get(src_col, {})
        c2pos = col_seqs.get(tgt_col, {})

        sum_pmatch = 0.0
        sum_pgap = 0.0

        for seq_i, pos_i in c1pos.items():
            for seq_j, pos_j in c2pos.items():
                if seq_i == seq_j:
                    return -1e10, -1e10  # invalid

                # Look up posterior P(pos_i ~ pos_j)
                if (seq_i, seq_j) in pair_posteriors:
                    post = pair_posteriors[(seq_i, seq_j)]
                    pmatch = float(post[pos_i - 1, pos_j - 1])
                    gp_i = float(gap_post_0[(seq_i, seq_j)][pos_i - 1])
                    gp_j = float(gap_post_1[(seq_i, seq_j)][pos_j - 1])
                elif (seq_j, seq_i) in pair_posteriors:
                    post = pair_posteriors[(seq_j, seq_i)]
                    pmatch = float(post[pos_j - 1, pos_i - 1])
                    gp_i = float(gap_post_1[(seq_j, seq_i)][pos_i - 1])
                    gp_j = float(gap_post_0[(seq_j, seq_i)][pos_j - 1])
                else:
                    continue

                sum_pmatch += 2 * pmatch
                sum_pgap += gp_i + gp_j

        if sum_pgap < 1e-10:
            if sum_pmatch > 0:
                weight = 1e10
            else:
                return -1e10, -1e10
        else:
            weight = sum_pmatch / sum_pgap

        delta = sum_pmatch - sum_pgap
        return weight, delta

    # ================================================================
    # Create initial candidate edges (from DART AlignDag)
    # ================================================================

    edge_counter = 0  # for breaking ties in the heap
    edges = []

    for si in range(n_seqs):
        for sj in range(si + 1, n_seqs):
            if (si, sj) in pair_posteriors:
                post = pair_posteriors[(si, sj)]
                gp0 = gap_post_0[(si, sj)]
                gp1 = gap_post_1[(si, sj)]
            elif (sj, si) in pair_posteriors:
                post = pair_posteriors[(sj, si)].T
                gp0 = gap_post_1[(sj, si)]
                gp1 = gap_post_0[(sj, si)]
            else:
                continue

            Li, Lj = post.shape
            for ri in range(Li):
                pgap_i = float(gp0[ri])
                for rj in range(Lj):
                    pmatch = float(post[ri, rj])
                    if pmatch < 0.01:
                        continue
                    pgap_j = float(gp1[rj])
                    # TGF weight: 2*Pmatch / (Pgap_i + Pgap_j)
                    denom = pgap_i + pgap_j
                    if denom < 1e-10:
                        weight = 1e10 if pmatch > 0 else 0.0
                    else:
                        weight = 2 * pmatch / denom

                    if weight < edge_weight_threshold or weight < gap_factor:
                        continue

                    col_i = seqpos_to_col[(si, ri + 1)]
                    col_j = seqpos_to_col[(sj, rj + 1)]
                    heapq.heappush(edges,
                                   (-weight, edge_counter, col_i, col_j))
                    edge_counter += 1

    # ================================================================
    # Process edges in descending weight order (from DART AlignDag)
    # ================================================================

    n_merged = 0
    while edges:
        neg_w, _, e_col1, e_col2 = heapq.heappop(edges)

        c1 = find_col(e_col1)
        c2 = find_col(e_col2)

        if c1 == c2:
            continue
        if c1 not in live_cols or c2 not in live_cols:
            continue

        # Dynamic weight recalculation (enableEdgeReordering in DART)
        new_weight, delta = calc_tgf_weight(c1, c2)

        if new_weight <= -1e9:  # invalid edge
            continue

        # Check if weight dropped below threshold
        if new_weight < edge_weight_threshold or new_weight < gap_factor:
            continue

        # Check if weight dropped below the next edge in queue
        if edges and new_weight < -edges[0][0]:
            heapq.heappush(edges,
                           (-new_weight, edge_counter, c1, c2))
            edge_counter += 1
            continue

        # Try to add the edge (merge columns)
        result = try_add_edge(c1, c2)
        if result == 0:
            n_merged += 1

    if verbose:
        print(f"  AMAP: merged {n_merged} edges, "
              f"{len(live_cols)} columns remain")

    # ================================================================
    # Topological sort of remaining columns
    # ================================================================

    # Build adjacency from sequence ordering
    adj = {c: set() for c in live_cols}
    in_degree = {c: 0 for c in live_cols}
    for c in live_cols:
        for succ in get_successors(c):
            if succ in live_cols and succ != c:
                if succ not in adj[c]:
                    adj[c].add(succ)
                    in_degree[succ] += 1

    # Kahn's algorithm, breaking ties by col_index for stability
    topo_queue = []
    for c in live_cols:
        if in_degree[c] == 0:
            heapq.heappush(topo_queue, (col_index[c], c))

    topo_order = []
    while topo_queue:
        _, node = heapq.heappop(topo_queue)
        topo_order.append(node)
        for succ in adj[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                heapq.heappush(topo_queue, (col_index[succ], succ))

    if len(topo_order) != len(live_cols):
        # Fallback: sort by topological index
        topo_order = sorted(live_cols, key=lambda c: col_index[c])

    col_remap = {c: idx for idx, c in enumerate(topo_order)}

    col_assignments = []
    for si in range(n_seqs):
        ca = np.zeros(seq_lengths[si], dtype=int)
        for k in range(seq_lengths[si]):
            pos = k + 1
            live_c = find_col(seqpos_to_col[(si, pos)])
            ca[k] = col_remap[live_c]
        col_assignments.append(ca)

    return col_assignments


def _refine_one_sequence(col_assignments, seq_idx, n_seqs, seq_lengths,
                          pair_posteriors):
    """Remove one sequence and reinsert it optimally via DP.

    This is a refinement step applied after the main AMAP build. For the
    given sequence, we remove it from the alignment, then use DP to find
    the column assignment that maximizes the sum of pairwise posteriors
    with all other (fixed) sequences, subject to the constraint that
    columns must be strictly increasing.

    Args:
        col_assignments: current alignment (list of arrays)
        seq_idx: which sequence to refine
        n_seqs: number of sequences
        seq_lengths: list of sequence lengths
        pair_posteriors: dict {(i,j): (Li, Lj) posterior array}

    Returns:
        new_col_assignments: updated alignment
        new_n_cols: number of columns
    """
    Li = seq_lengths[seq_idx]
    if Li == 0:
        return list(col_assignments), max(
            max(ca) + 1 for ca in col_assignments if len(ca) > 0)

    # Build column -> residue mapping for each other sequence
    col_to_res = {}  # {j: {col: residue_idx}}
    for j in range(n_seqs):
        if j == seq_idx or len(col_assignments[j]) == 0:
            continue
        mapping = {}
        for k, c in enumerate(col_assignments[j]):
            mapping[int(c)] = k
        col_to_res[j] = mapping

    # Find columns used by other sequences
    used_cols = set()
    for j, mapping in col_to_res.items():
        used_cols.update(mapping.keys())
    sorted_cols = sorted(used_cols)

    if not sorted_cols:
        # No other sequences, just assign contiguous columns
        new_ca = list(col_assignments)
        new_ca[seq_idx] = np.arange(Li, dtype=int)
        return new_ca, Li

    n_cols_other = len(sorted_cols)
    col_remap = {c: idx for idx, c in enumerate(sorted_cols)}

    # Remap other sequences to contiguous columns
    remapped = list(col_assignments)
    for j in range(n_seqs):
        if j == seq_idx:
            continue
        if len(remapped[j]) > 0:
            remapped[j] = np.array([col_remap[int(c)] for c in remapped[j]])

    # Recompute col_to_res with remapped columns
    col_to_res_r = {}
    for j in range(n_seqs):
        if j == seq_idx or len(remapped[j]) == 0:
            continue
        mapping = {}
        for k, c in enumerate(remapped[j]):
            mapping[int(c)] = k
        col_to_res_r[j] = mapping

    # Compute score_gain[k][c] = sum of posteriors from placing
    # residue k of seq_idx in (remapped) column c
    max_col = n_cols_other
    score_gain = np.zeros((Li, max_col))

    for j, res_map in col_to_res_r.items():
        if (seq_idx, j) in pair_posteriors:
            post = pair_posteriors[(seq_idx, j)]
            for c, rj in res_map.items():
                score_gain[:, c] += post[:, rj]
        elif (j, seq_idx) in pair_posteriors:
            post = pair_posteriors[(j, seq_idx)]
            for c, rj in res_map.items():
                score_gain[:, c] += post[rj, :]

    # DP: assign each residue to a column (existing or new),
    # with strictly increasing column indices.
    # State: (residue k, column choice c)
    # Transition: c_k > c_{k-1}
    # New columns are indexed max_col, max_col+1, ...

    # Candidate columns: existing cols with nonzero gain + new cols
    candidate_cols = set()
    for k in range(Li):
        for c in range(max_col):
            if score_gain[k, c] > 1e-10:
                candidate_cols.add(c)
    # New columns for gap positions
    for i in range(Li):
        candidate_cols.add(max_col + i)

    sorted_candidates = sorted(candidate_cols)
    n_cand = len(sorted_candidates)

    # DP arrays
    dp = np.full((Li, n_cand), -np.inf)
    bp = np.full((Li, n_cand), -1, dtype=int)

    # Base case: residue 0
    for ci, c in enumerate(sorted_candidates):
        dp[0, ci] = score_gain[0, c] if c < max_col else 0.0

    # Fill DP
    for k in range(1, Li):
        # Running max of dp[k-1, :] up to ci-1
        best_prev = -np.inf
        best_prev_ci = -1
        for ci, c in enumerate(sorted_candidates):
            # The constraint is strictly increasing: c_k > c_{k-1}
            # So for candidate ci, we can use any previous candidate < ci
            if ci > 0 and dp[k-1, ci-1] > best_prev:
                best_prev = dp[k-1, ci-1]
                best_prev_ci = ci - 1

            if best_prev > -np.inf:
                gain = score_gain[k, c] if c < max_col else 0.0
                val = best_prev + gain
                if val > dp[k, ci]:
                    dp[k, ci] = val
                    bp[k, ci] = best_prev_ci

    # Traceback
    best_ci = int(np.argmax(dp[Li - 1]))
    result_cols = []
    ci = best_ci
    for k in range(Li - 1, -1, -1):
        result_cols.append(sorted_candidates[ci])
        ci = bp[k, ci]
    result_cols.reverse()

    # Build new col_assignments
    # Merge result_cols (may include new cols beyond max_col) with existing
    all_cols = set()
    for j in range(n_seqs):
        if j == seq_idx and len(remapped[j]) > 0:
            continue
        if len(remapped[j]) > 0:
            for c in remapped[j]:
                all_cols.add(int(c))
    for c in result_cols:
        all_cols.add(c)

    sorted_all = sorted(all_cols)
    final_remap = {c: idx for idx, c in enumerate(sorted_all)}
    new_n_cols = len(sorted_all)

    new_ca = []
    for j in range(n_seqs):
        if j == seq_idx:
            new_ca.append(np.array([final_remap[c] for c in result_cols],
                                   dtype=int))
        else:
            if len(remapped[j]) > 0:
                new_ca.append(np.array(
                    [final_remap[int(c)] for c in remapped[j]], dtype=int))
            else:
                new_ca.append(np.array([], dtype=int))

    return new_ca, new_n_cols


# ============================================================
# Backward-compatible aliases for old API
# ============================================================

def _build_initial_alignment(n_seqs, seq_lengths, pair_posteriors):
    """Backward-compatible alias for _amap_align."""
    return _amap_align(n_seqs, seq_lengths, pair_posteriors)


def _remove_sequence(col_assignments, seq_idx, n_seqs, seq_lengths):
    """Remove a sequence from the alignment and compact columns."""
    new_assignments = list(col_assignments)
    new_assignments[seq_idx] = None

    used_cols = set()
    for i in range(n_seqs):
        if i == seq_idx:
            continue
        if new_assignments[i] is not None:
            for c in new_assignments[i]:
                used_cols.add(c)

    sorted_cols = sorted(used_cols)
    col_remap = {c: idx for idx, c in enumerate(sorted_cols)}
    next_col = len(sorted_cols)

    for i in range(n_seqs):
        if i == seq_idx or new_assignments[i] is None:
            continue
        new_assignments[i] = np.array([col_remap[c] for c in new_assignments[i]])

    return new_assignments, next_col


def _reinsert_sequence(col_assignments, seq_idx, n_seqs, seq_lengths,
                        pair_posteriors, n_cols):
    """Backward-compatible wrapper around _refine_one_sequence."""
    # First put dummy assignment for seq_idx, then refine
    Li = seq_lengths[seq_idx]
    ca = list(col_assignments)
    if ca[seq_idx] is None:
        ca[seq_idx] = np.arange(n_cols, n_cols + Li, dtype=int)
    return _refine_one_sequence(ca, seq_idx, n_seqs, seq_lengths,
                                 pair_posteriors)


# ============================================================
# MSA construction
# ============================================================

def sequence_annealing_multi_seed(n_seqs, seq_lengths, pair_posteriors,
                                    seeds, n_iterations=5, verbose=False,
                                    gap_factor=1.0, edge_weight_threshold=0.0,
                                    return_all=False):
    """Run sequence_annealing for many seeds with the same pair_posteriors.

    The pairwise-posterior computation is the expensive step; the annealing
    refinement that depends on ``seed`` is cheap, so calling this with a
    list of seeds is a near-free way to explore the FSA solution distribution.

    Args:
        n_seqs, seq_lengths, pair_posteriors: as in ``sequence_annealing``.
        seeds: iterable of int seeds.
        n_iterations, verbose, gap_factor, edge_weight_threshold: as above.
        return_all: if True, return list of (seed, col_assignments, msa_length,
            score) tuples for every seed; if False, return only the best by
            internal score.

    Returns:
        If return_all=False: (best_col_assignments, best_msa_length, best_seed, best_score).
        If return_all=True:  list of (seed, col_assignments, msa_length, score).
    """
    runs = []
    for seed in seeds:
        col_assignments, msa_length = sequence_annealing(
            n_seqs, seq_lengths, pair_posteriors,
            n_iterations=n_iterations, verbose=verbose,
            gap_factor=gap_factor,
            edge_weight_threshold=edge_weight_threshold,
            seed=int(seed))
        score = _score_alignment(col_assignments, seq_lengths, pair_posteriors)
        runs.append((int(seed), col_assignments, msa_length, float(score)))
    if return_all:
        return runs
    best = max(runs, key=lambda r: r[3])
    return best[1], best[2], best[0], best[3]


def fsa_align(sequences, model='tkf92', pair_selection='erdos_renyi',
              ins_rate=0.02, del_rate=0.05, ext=0.5,
              Q=None, pi=None, mixdom_params=None,
              n_dom=3, n_frag=2,
              maraschino_params=None, n_classes=4,
              n_newton=5, tau_init=1.0,
              n_anneal_iterations=5,
              verbose=False,
              seed=42):
    """Compute MSA using FSA-style sequence annealing.

    Args:
        sequences: dict of {name: integer_array}
        model: 'tkf92', 'mixdom', 'distilled', or 'wfst'
        pair_selection: 'full' (O(N^2)) or 'erdos_renyi' (O(N log N))
        ins_rate, del_rate, ext: TKF92 parameters
        Q, pi: substitution model (default: LG)
        mixdom_params: MixDom parameter dict (if model='mixdom')
        n_dom, n_frag: MixDom model dimensions
        maraschino_params: constrained params from maraschino.load_params
            (if model='distilled')
        n_classes: number of gamma rate classes (if model='distilled')
        n_newton: Newton-Raphson steps for tau
        tau_init: initial tau
        n_anneal_iterations: number of annealing sweeps
        verbose: print progress

    Returns:
        msa_dict: dict of {name: aligned_integer_array (with -1 for gaps)}
        msa_length: alignment length
    """
    if Q is None or pi is None:
        from ..core.protein import rate_matrix_lg
        Q, pi = rate_matrix_lg()

    names = list(sequences.keys())
    n_seqs = len(names)
    seq_lengths = [len(sequences[n]) for n in names]

    if n_seqs < 2:
        # Trivial case
        msa_dict = {n: sequences[n] for n in names}
        return msa_dict, seq_lengths[0] if seq_lengths else 0

    # Step 1: Select pairs
    if pair_selection == 'full':
        pairs = select_pairs_full(n_seqs)
    elif pair_selection == 'erdos_renyi':
        pairs = select_pairs_erdos_renyi(n_seqs)
    else:
        raise ValueError(f"Unknown pair selection: {pair_selection}")

    if verbose:
        print(f"FSA: {n_seqs} sequences, {len(pairs)} pairs "
              f"({pair_selection}), model={model}")

    # Step 2: Compute pairwise posteriors
    pair_posteriors, pair_taus = compute_pairwise_posteriors(
        sequences, pairs, model=model,
        ins_rate=ins_rate, del_rate=del_rate, ext=ext,
        Q=Q, pi=pi, mixdom_params=mixdom_params,
        n_dom=n_dom, n_frag=n_frag,
        maraschino_params=maraschino_params, n_classes=n_classes,
        n_newton=n_newton, tau_init=tau_init, verbose=verbose)

    if verbose:
        tau_vals = list(pair_taus.values())
        print(f"  Tau range: [{min(tau_vals):.4f}, {max(tau_vals):.4f}]")

    # Step 3: Sequence annealing
    col_assignments, msa_length = sequence_annealing(
        n_seqs, seq_lengths, pair_posteriors,
        n_iterations=n_anneal_iterations, verbose=verbose, seed=seed)

    # Step 4: Convert to MSA dict
    msa_dict = {}
    for i, name in enumerate(names):
        row = np.full(msa_length, -1, dtype=np.int32)
        seq = sequences[name]
        for k in range(len(seq)):
            col = col_assignments[i][k]
            row[col] = int(seq[k])
        msa_dict[name] = row

    return msa_dict, msa_length


# ============================================================
# CLI-compatible runner (for benchmark integration)
# ============================================================

def run_fsa_alignment(input_fasta_path, output_fasta_path,
                       model='tkf92', pair_selection='erdos_renyi',
                       params_path=None, verbose=False):
    """Run FSA alignment from FASTA files.

    Args:
        input_fasta_path: path to input unaligned FASTA
        output_fasta_path: path to write aligned FASTA
        model: 'tkf92', 'mixdom', 'distilled', or 'wfst'
        pair_selection: 'full' or 'erdos_renyi'
        params_path: path to MixDom params .npz (required if model='mixdom',
            'distilled', or 'wfst')
        verbose: print progress

    Returns:
        True on success, False on failure
    """
    from ..core.protein import rate_matrix_lg

    AA_MAP = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"

    # Parse input
    seqs = {}
    name = None
    parts = []
    with open(input_fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    seqs[name] = ''.join(parts)
                name = line[1:].split()[0]
                parts = []
            elif name is not None:
                parts.append(line)
    if name is not None:
        seqs[name] = ''.join(parts)

    if len(seqs) < 2:
        return False

    Q, pi = rate_matrix_lg()

    int_seqs = {}
    for n, seq in seqs.items():
        int_seq = [AA_MAP.get(c.upper(), 0) for c in seq if c.upper() not in '.-']
        int_seqs[n] = np.array(int_seq, dtype=np.int32)

    mixdom_params = None
    maraschino_params = None
    n_dom, n_frag = 3, 2
    n_classes = 4

    if model == 'mixdom' and params_path is not None:
        data = np.load(params_path, allow_pickle=True)
        n_dom = int(data.get('n_domains', 3)) if 'n_domains' in data else len(data['dom_ins'])
        n_frag = data['frag_weights'].shape[1] if data['frag_weights'].ndim > 1 else 1
        mixdom_params = {
            'main_ins': float(data['main_ins']),
            'main_del': float(data['main_del']),
            'dom_ins': np.array(data['dom_ins']),
            'dom_del': np.array(data['dom_del']),
            'dom_weights': np.array(data['dom_weights']),
            'frag_weights': np.array(data['frag_weights']),
            'ext_rates': np.array(data['ext_rates']),
            'Q': Q, 'pi': pi,
        }
    elif model in ('distilled', 'wfst') and params_path is not None:
        from ..distill.maraschino import load_params
        maraschino_params, _, n_classes = load_params(params_path)

    try:
        msa_dict, msa_length = fsa_align(
            int_seqs, model=model, pair_selection=pair_selection,
            Q=Q, pi=pi, mixdom_params=mixdom_params,
            n_dom=n_dom, n_frag=n_frag,
            maraschino_params=maraschino_params, n_classes=n_classes,
            verbose=verbose)
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return False

    # Write output
    with open(output_fasta_path, 'w') as f:
        for name in seqs:
            if name in msa_dict:
                row = msa_dict[name]
                aln_str = ''.join(
                    AA_CHARS[c] if 0 <= c < 20 else '-'
                    for c in row)
                f.write(f'>{name}\n{aln_str}\n')

    return True
