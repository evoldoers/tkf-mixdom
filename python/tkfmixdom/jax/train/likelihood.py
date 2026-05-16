"""Uniform differentiable likelihood interface for TKF models.

Provides three functions:
  1. e_step: Forward-Backward, returns counts and match pairs
  2. observed_ll: log P(x, y | θ) with custom VJP (structural params)
  3. expected_ll: Q(θ) = Q_struct + Q_subst, differentiable w.r.t. ALL params

Sufficient statistics:
  - Structural (λ, μ, ext, weights): BDI score function identity
  - Substitution (Q): Holmes-Rubin expected CTMC path counts (u_hat, w_hat)
    pre-computed in E-step, then Q_subst = Σ u_hat·log Q + Σ w_hat·Q_ii

Usage:
    from tkfmixdom.jax.train.likelihood import e_step, observed_ll, expected_ll

    # E-step (returns dict suff stats; parallels MixDom2 case)
    log_p, n_trans, ss = e_step('tkf91', params, x, y)

    # Q-function (differentiable w.r.t. ALL params including Q and π)
    agg_u, agg_w = substitution_suffstats(
        params['Q'], params['pi'], params['t'], ss['match_pairs'])
    q = expected_ll('tkf91', params, n_trans,
                     subst_suffstats=(agg_u, agg_w),
                     pi_obs_counts=ss['pi_obs_counts'])
"""

import jax
import jax.numpy as jnp
import numpy as np

from .vjp import (
    tkf91_log_prob, tkf91_log_prob_cond,
    tkf92_log_prob,
    mixdom_log_prob, mixdom2_log_prob, _chi_weighted_loglik,
    _build_class_subs,
)
from ..core.params import S, M, I, D, E, tkf91_trans, tkf92_trans
from ..core.ctmc import (transition_matrix,
                          holmes_rubin_expected_stats,
                          holmes_rubin_weighted_stats)
from ..dp.hmm import forward_backward_2d, safe_log, pair_hmm_emissions_per_class
from ..models.mixdom import build_nested_trans, state_types as mixdom_state_types


# ================================================================
# E-step
# ================================================================

def e_step(model, params, x, y):
    """E-step: Forward-Backward on the Pair HMM.

    Returns:
        log_prob: log P(x, y | θ)
        n_trans: (N, N) expected transition counts
        ss: For single-Q models (TKF91/TKF92/MixDom1), a dict with
            - 'match_pairs':   list of (anc_char, desc_char, weight)
            - 'pi_obs_counts': dict with keys 'match_anc_count' (A,),
                                'insert_count' (A,), 'delete_count' (A,)
            For MixDom2, the full per-class dict (see
            `e_step_mixdom2`).

    The `pi_obs_counts` enables `expected_ll`'s `Σ_a V[a] · log π[a]`
    observation-side π term — required for the wrapper-grad ↔
    expected_ll-grad cross-check on π.
    """
    if model == 'mixdom2':
        return e_step_mixdom2(params, x, y)

    Q, pi, t = params['Q'], params['pi'], params['t']
    sub = transition_matrix(Q, t)
    x_jnp, y_jnp = jnp.asarray(x), jnp.asarray(y)

    if model in ('tkf91', 'tkf91_cond'):
        tau = tkf91_trans(params['ins_rate'], params['del_rate'], t)
        st = jnp.array([S, M, I, D, E])
    elif model == 'tkf92':
        tau = tkf92_trans(params['ins_rate'], params['del_rate'], t, params['ext'])
        st = jnp.array([S, M, I, D, E])
    elif model == 'mixdom':
        tau, _ = build_nested_trans(
            params['main_ins'], params['main_del'], t,
            jnp.array(params['dom_ins']), jnp.array(params['dom_del']),
            jnp.array(params['dom_weights']),
            jnp.array(params['frag_weights']),
            jnp.array(params['ext_rates']))
        K = len(params['dom_ins'])
        F = np.asarray(params['frag_weights']).shape[1]
        st = mixdom_state_types(K, F)
    else:
        raise ValueError(f"Unknown model: {model}")

    log_prob, posteriors, n_trans = forward_backward_2d(
        safe_log(tau), st, x_jnp, y_jnp, sub, pi)

    # Vectorised match-pair extraction. Replaces the prior per-(i, j)
    # double Python loop. For each grid cell (i, j) ∈ (1..Lx, 1..Ly),
    # m_post[i, j] = Σ_s posteriors[i, j, s] · 1[state_type[s] == M].
    # Single einsum, then a single np.where + fancy-indexing gather to
    # produce the match_pairs list-of-tuples for the public API.
    Lx, Ly = len(x), len(y)
    is_M_state = (st == M).astype(posteriors.dtype)
    m_post_full = jnp.einsum('ijs,s->ij', posteriors, is_M_state)
    m_post = np.asarray(m_post_full[1:Lx + 1, 1:Ly + 1])             # (Lx, Ly)
    mask = m_post > 1e-8
    ii, jj = np.where(mask)
    if ii.size == 0:
        match_pairs = []
    else:
        anc_chars_arr = np.asarray(x, dtype=np.int32)[ii]
        desc_chars_arr = np.asarray(y, dtype=np.int32)[jj]
        weights = m_post[mask].astype(np.float64)
        # match_pairs is a (n_match, 3) ndarray (a, b, w). The single
        # `list(zip(...))` does a vectorised gather + tuple-construction
        # at the I/O boundary; no per-position Python iteration over
        # the (Lx, Ly) lattice.
        match_pairs = list(zip(
            anc_chars_arr.tolist(),
            desc_chars_arr.tolist(),
            weights.tolist()))

    # forward_backward_2d pads x, y to geometric bins for JIT cache
    # reuse, so `posteriors` shape is (Lx_pad+1, Ly_pad+1, ns). Use
    # padded x/y for indexing; the boundary mask in FB zeroes posterior
    # mass beyond real lengths, so padded cells contribute 0 to V.
    A = int(np.asarray(pi).shape[0])
    from ..dp.hmm import _pad_to_bin, _pad_seq
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)
    x_pad = _pad_seq(x_jnp, Lx_pad)
    y_pad = _pad_seq(y_jnp, Ly_pad)
    pi_obs_counts_jnp = observation_counts_2d(
        posteriors, st, x_pad, y_pad, A)
    pi_obs_counts = {k: np.asarray(v) for k, v in pi_obs_counts_jnp.items()}

    ss = {'match_pairs': match_pairs, 'pi_obs_counts': pi_obs_counts}
    return float(log_prob), n_trans, ss


# ================================================================
# E-step for MixDom2 (per-class suff stats, dict return)
# ================================================================

def e_step_mixdom2(params, x, y):
    """E-step for MixDom2 (per-fragment site-class mixture).

    Returns the dict of per-class sufficient statistics that
    `expected_ll(model='mixdom2', ...)` consumes:

        {
          'class_W':              (C, A),
          'class_U':              (C, A, A),
          'class_match_counts':   (C, A, A),
          'class_insert_counts':  (C, A),
          'class_delete_counts':  (C, A),
          'classdist_counts':     (D, F, C),
        }

    plus log_prob and n_trans (the chi-side suff stat, shape (N, N)).

    Implementation:
      1. Build chi (top-level transitions) and per-class P_c(t).
      2. Build per-class emission table (logsumexp over c).
      3. Run forward_backward_2d → posteriors[i, j, s], n_trans.
      4. For each (i, j) and each body state s in (d, f) of type τ ∈ {M, I, D}:
           q[c | i, j, s] = classdist[d, f, c] · emit_factor_τ[c, i, j]
                            / Σ_c' classdist · emit_factor
         joint[c, i, j, s] = posteriors[i, j, s] · q[c | i, j, s]
      5. Aggregate via einsum:
           class_match_counts[c, a, b]  = Σ_{i, j, s∈M-type, x[i-1]=a, y[j-1]=b} joint
           class_insert_counts[c, b]    = Σ_{i, j, s∈I-type, y[j-1]=b} joint
           class_delete_counts[c, a]    = Σ_{i, j, s∈D-type, x[i-1]=a} joint
           classdist_counts[d, f, c]    = Σ_{i, j, s∈(d,f)} joint
      6. Per-class HR via vmap of `holmes_rubin_weighted_stats` →
         class_W (C, A), class_U (C, A, A).

    Sums are explicit per-(c, d, f, i, j); never averaged over latent
    axes. No consensus parameters. The single legitimate aggregation
    is the FB sum-over-paths producing posteriors, which is just
    standard EM.

    Args:
        params: dict with `main_ins`, `main_del`, `t`, `dom_ins`,
            `dom_del`, `dom_weights`, `frag_weights`, `ext_rates`,
            `class_S_exch` (C, A, A), `class_pis` (C, A),
            `classdist` (D, F, C).
        x, y: integer sequences (Lx,), (Ly,).

    Returns:
        log_prob: scalar log P(x, y | θ).
        n_trans: (N, N) expected transition counts (chi suff stat).
        suffstats: dict of per-class suff stats (see above).
    """
    t = params['t']
    n_dom = jnp.asarray(params['dom_ins']).shape[0]
    n_frag = jnp.asarray(params['frag_weights']).shape[1]
    A = jnp.asarray(params['class_pis']).shape[1]
    n_cls = jnp.asarray(params['class_pis']).shape[0]

    x_jnp = jnp.asarray(x, dtype=jnp.int32)
    y_jnp = jnp.asarray(y, dtype=jnp.int32)

    # Build chi (top-level transitions over the 5DF+2 collapsed state space).
    chi, _ = build_nested_trans(
        params['main_ins'], params['main_del'], t,
        jnp.asarray(params['dom_ins']), jnp.asarray(params['dom_del']),
        jnp.asarray(params['dom_weights']),
        jnp.asarray(params['frag_weights']),
        jnp.asarray(params['ext_rates']))
    log_chi = safe_log(chi)
    st = mixdom_state_types(n_dom, n_frag)

    # Per-class P_c(t) — built via _build_class_subs (eigh-based; for
    # the inner suff-stat extraction we treat it as a leaf since the
    # downstream HR computes its OWN expm/integrals on Q_c).
    class_subs = _build_class_subs(
        jnp.asarray(params['class_S_exch']),
        jnp.asarray(params['class_pis']),
        t)                                                             # (C, A, A)

    # Per-class emission table (FB-ready). pair_hmm_emissions_per_class
    # marginalises over c via logsumexp internally per (i, j, state) —
    # the result is the (Lx+1, Ly+1, ns) log emission used by FB.
    log_emit = pair_hmm_emissions_per_class(
        st, x_jnp, y_jnp,
        class_subs,
        jnp.asarray(params['class_pis']),
        jnp.asarray(params['classdist']),
        n_dom, n_frag)

    # FB on the per-class-aware emission table. sub_matrix and pi are
    # ignored on this code path (forward_backward_2d uses log_emit_table
    # when present); we pass dummy values.
    log_prob, posteriors, n_trans = forward_backward_2d(
        log_chi, st, x_jnp, y_jnp,
        sub_matrix=jnp.eye(1), pi=jnp.ones(1),
        log_emit_table=log_emit)

    # forward_backward_2d pads x, y to geometric bin sizes for JIT cache
    # reuse, so `posteriors` shape is (Lx_pad+1, Ly_pad+1, ns), NOT
    # (Lx+1, Ly+1, ns). At padded positions the FB has zero posterior
    # mass (boundary mask zeroes out beyond real lengths), so
    # contributions from those cells are 0 — but we must use the
    # padded character vectors x_pad, y_pad for indexing to keep all
    # einsum dims consistent.
    from ..dp.hmm import _pad_to_bin, _pad_seq
    Lx, Ly = x_jnp.shape[0], y_jnp.shape[0]
    Lx_pad = _pad_to_bin(int(Lx))
    Ly_pad = _pad_to_bin(int(Ly))
    x_pad_full = _pad_seq(x_jnp, Lx_pad)
    y_pad_full = _pad_seq(y_jnp, Ly_pad)
    x_at_i = jnp.concatenate([jnp.array([0], dtype=x_pad_full.dtype),
                                x_pad_full])                            # (Lx_pad+1,)
    y_at_j = jnp.concatenate([jnp.array([0], dtype=y_pad_full.dtype),
                                y_pad_full])                            # (Ly_pad+1,)
    class_pis = jnp.asarray(params['class_pis'])                          # (C, A)
    cd = jnp.asarray(params['classdist'])                                 # (D, F, C)

    # Per-class per-(i, j) emission factors:
    #   match: π_c[x[i]] · P_c[x[i], y[j]]            shape (C, Lx+1, Ly+1)
    #   ins:   π_c[y[j]]                                shape (C, Ly+1)
    #   del:   π_c[x[i]]                                shape (C, Lx+1)
    pi_x = class_pis[:, x_at_i]                                           # (C, Lx+1)
    pi_y = class_pis[:, y_at_j]                                           # (C, Ly+1)
    P_xy = class_subs[:, x_at_i, :][:, :, y_at_j]                         # (C, Lx+1, Ly+1)
    f_M = pi_x[:, :, None] * P_xy                                         # (C, Lx+1, Ly+1)
    f_I = pi_y                                                            # (C, Ly+1)
    f_D = pi_x                                                            # (C, Lx+1)

    # Per-state (d, f, type) lookup for body states (s ≥ 2).
    ns = st.shape[0]
    body = jnp.arange(ns - 2)
    dom_idx = body // (5 * n_frag)                                         # (ns-2,)
    within_dom = body % (5 * n_frag)
    frag_idx = within_dom % n_frag                                         # (ns-2,)
    uv_idx = within_dom // n_frag           # 0..4 → MM, MI, MD, II, DD
    is_M_body = (uv_idx == 0).astype(posteriors.dtype)
    is_I_body = ((uv_idx == 1) | (uv_idx == 3)).astype(posteriors.dtype)
    is_D_body = ((uv_idx == 2) | (uv_idx == 4)).astype(posteriors.dtype)

    # Per-(i, j, d, f) posterior aggregates of body states by type.
    body_post = posteriors[:, :, 2:]                                       # (Lx+1, Ly+1, ns-2)
    flat_df = (dom_idx * n_frag + frag_idx).astype(jnp.int32)
    one_hot_df = jax.nn.one_hot(flat_df, n_dom * n_frag,
                                  dtype=posteriors.dtype)                  # (ns-2, DF)

    def _agg_by_df_and_type(body_mask):
        weighted = body_post * body_mask[None, None, :]                    # (Lx+1, Ly+1, ns-2)
        flat = jnp.einsum('ijb,bk->ijk', weighted, one_hot_df)             # (Lx+1, Ly+1, DF)
        return flat.reshape(posteriors.shape[0], posteriors.shape[1],
                             n_dom, n_frag)                                # (Lx+1, Ly+1, D, F)

    post_dfM = _agg_by_df_and_type(is_M_body)                              # (Lx+1, Ly+1, D, F)
    post_dfI = _agg_by_df_and_type(is_I_body)
    post_dfD = _agg_by_df_and_type(is_D_body)

    # Per-(i, j) per-class assignment posterior: q[c | i, j, type, d, f]
    # ∝ classdist[d, f, c] · emit_factor[c, i, j, type]. Normalise per
    # (d, f, type, i, j). The denominator is the per-state-type emit
    # at (i, j) under the (d, f) location.
    # Compute (D, F, C, Lx+1, Ly+1) joint factor for each type, then
    # contract with body posteriors.
    cd_DFC_M = cd[:, :, :, None, None] * f_M[None, None, :, :, :]          # (D, F, C, Lx+1, Ly+1)
    cd_DFC_I = cd[:, :, :, None] * f_I[None, None, :, :]                   # (D, F, C, Ly+1)
    cd_DFC_D = cd[:, :, :, None] * f_D[None, None, :, :]                   # (D, F, C, Lx+1)
    # Denominator (sum over c): emit_per_(d, f, type, i, j) — equals
    # the emission used by the FB at this (d, f) location.
    denom_M = jnp.sum(cd_DFC_M, axis=2) + 1e-300                           # (D, F, Lx+1, Ly+1)
    denom_I = jnp.sum(cd_DFC_I, axis=2) + 1e-300                           # (D, F, Ly+1)
    denom_D = jnp.sum(cd_DFC_D, axis=2) + 1e-300                           # (D, F, Lx+1)
    # Per-class assignment posterior q[c | type, d, f, i, j]:
    q_M = cd_DFC_M / denom_M[:, :, None, :, :]                             # (D, F, C, Lx+1, Ly+1)
    q_I = cd_DFC_I / denom_I[:, :, None, :]                                # (D, F, C, Ly+1)
    q_D = cd_DFC_D / denom_D[:, :, None, :]                                # (D, F, C, Lx+1)

    # joint[d, f, c, i, j] = post_df_type[i, j, d, f] · q[c | type, d, f, i, j]
    joint_M = jnp.einsum('ijdf,dfcij->dfcij', post_dfM, q_M)               # (D, F, C, Lx+1, Ly+1)
    # For I joint we ignore i (only depends on j):
    joint_I = jnp.einsum('ijdf,dfcj->dfcij', post_dfI, q_I)                # (D, F, C, Lx+1, Ly+1)
    joint_D = jnp.einsum('ijdf,dfci->dfcij', post_dfD, q_D)                # (D, F, C, Lx+1, Ly+1)

    # Aggregate by character. one_hot at i=0 / j=0 maps to char 0 which
    # is masked because for real positions i ∈ [1, Lx] (so x_at_i[i]=x[i-1]).
    # `joint_M[d, f, c, 0, *]` would correspond to a non-existent grid
    # row — but post_dfM[0, :, :, :] = 0 because no body state lives at
    # grid row 0 (S/E states are non-emitting). So this is safe.
    one_hot_x = jax.nn.one_hot(x_at_i, A, dtype=posteriors.dtype)          # (Lx+1, A)
    one_hot_y = jax.nn.one_hot(y_at_j, A, dtype=posteriors.dtype)          # (Ly+1, A)

    # Sum over (d, f) → per-class only (for class_match/insert/delete_counts).
    class_match_counts = jnp.einsum(
        'dfcij,ia,jb->cab',
        joint_M, one_hot_x, one_hot_y)                                     # (C, A, A)
    class_insert_counts = jnp.einsum(
        'dfcij,jb->cb',
        joint_I, one_hot_y)                                                # (C, A)
    class_delete_counts = jnp.einsum(
        'dfcij,ia->ca',
        joint_D, one_hot_x)                                                # (C, A)

    # classdist_counts[d, f, c] = Σ_(i, j) over all types at that (d, f).
    classdist_counts = (joint_M.sum(axis=(3, 4))
                        + joint_I.sum(axis=(3, 4))
                        + joint_D.sum(axis=(3, 4)))                        # (D, F, C)

    # Per-class HR: build Q_c, then call holmes_rubin_weighted_stats per c.
    def _build_Q_c(S_c, pi_c):
        Q = S_c * pi_c[None, :]
        Q = Q - jnp.diag(jnp.diag(Q))
        Q = Q - jnp.diag(Q.sum(axis=1))
        return Q
    Q_per_class = jax.vmap(_build_Q_c)(
        jnp.asarray(params['class_S_exch']),
        jnp.asarray(params['class_pis']))                                  # (C, A, A)

    def _hr_per_class(Q_c, pi_c, mc_c):
        return holmes_rubin_weighted_stats(Q_c, pi_c, t, mc_c)
    class_W, class_U = jax.vmap(_hr_per_class)(
        Q_per_class, jnp.asarray(params['class_pis']), class_match_counts)
    # class_W: (C, A); class_U: (C, A, A)

    suffstats = {
        'class_W': np.asarray(class_W),
        'class_U': np.asarray(class_U),
        'class_match_counts': np.asarray(class_match_counts),
        'class_insert_counts': np.asarray(class_insert_counts),
        'class_delete_counts': np.asarray(class_delete_counts),
        'classdist_counts': np.asarray(classdist_counts),
    }
    return float(log_prob), n_trans, suffstats


# ================================================================
# Substitution sufficient statistics
# ================================================================

def substitution_suffstats(Q, pi, t, match_pairs):
    """Pre-compute Holmes-Rubin sufficient statistics for the substitution Q-function.

    Aggregates expected CTMC dwell times and transition counts across
    all match pairs, weighted by FB posteriors. Vectorised: builds an
    (A, A) per-(a, b) match-count matrix `mc` from match_pairs in one
    shot, then calls `holmes_rubin_weighted_stats(Q, pi, t, mc)` which
    shares a single eigendecomposition + integral-tensor computation
    across all (a, b) entries. Replaces the prior per-pair Python loop
    that called `holmes_rubin_expected_stats` once per match.

    Returns:
        agg_u: (n, n) aggregated expected transition counts
        agg_w: (n,) aggregated expected dwell times
    """
    n = int(Q.shape[0])
    if not match_pairs:
        return np.zeros((n, n)), np.zeros(n)
    # Vectorised mc construction. `np.asarray(list_of_3-tuples)` is a
    # single C-level copy (not a Python loop). `np.add.at` is a
    # vectorised scatter-add — it handles repeated (a, b) keys
    # correctly via index-grouped accumulation.
    mp_arr = np.asarray(match_pairs)                                  # (n_match, 3)
    anc = mp_arr[:, 0].astype(np.int32)
    desc = mp_arr[:, 1].astype(np.int32)
    weights = mp_arr[:, 2].astype(np.float64)
    mc = np.zeros((n, n), dtype=np.float64)
    np.add.at(mc, (anc, desc), weights)
    # One-shot vectorised HR — shares the eigh + integral tensor across
    # all (a, b) entries.
    W, U = holmes_rubin_weighted_stats(
        jnp.asarray(Q), jnp.asarray(pi), t, jnp.asarray(mc))
    return np.asarray(U), np.asarray(W)


# ================================================================
# Observed-data log-likelihood (structural params only)
# ================================================================

def observed_ll(model, params, x, y):
    """log P(x, y | θ) with custom VJP for structural parameters.

    Differentiable w.r.t. indel rates, weights, ext via BDI score identity.
    The custom VJP returns None for Q and π gradients — use expected_ll
    with substitution_suffstats for Q gradients (via Holmes-Rubin).

    The Q gradient from expected_ll equals the observed-data gradient
    at the current parameters (EM gradient identity), so expected_ll
    can be used for substitution parameter optimization too.
    """
    if model == 'tkf91':
        return tkf91_log_prob(params['ins_rate'], params['del_rate'],
                               params['t'], params['Q'], params['pi'], x, y)
    elif model == 'tkf91_cond':
        return tkf91_log_prob_cond(params['ins_rate'], params['del_rate'],
                                    params['t'], params['Q'], params['pi'], x, y)
    elif model == 'tkf92':
        return tkf92_log_prob(params['ins_rate'], params['del_rate'],
                               params['t'], params['ext'],
                               params['Q'], params['pi'], x, y)
    elif model == 'mixdom':
        return mixdom_log_prob(
            params['main_ins'], params['main_del'], params['t'],
            jnp.array(params['dom_ins']), jnp.array(params['dom_del']),
            jnp.array(params['dom_weights']),
            jnp.array(params['frag_weights']),
            jnp.array(params['ext_rates']),
            params['Q'], params['pi'], x, y)
    elif model == 'mixdom2':
        return mixdom2_log_prob(
            params['main_ins'], params['main_del'], params['t'],
            jnp.array(params['dom_ins']), jnp.array(params['dom_del']),
            jnp.array(params['dom_weights']),
            jnp.array(params['frag_weights']),
            jnp.array(params['ext_rates']),
            jnp.array(params['class_pis']),
            jnp.array(params['class_S_exch']),
            jnp.array(params['classdist']),
            x, y)
    else:
        raise ValueError(f"Unknown model: {model}")


# ================================================================
# Expected complete-data log-likelihood (ALL params)
# ================================================================

def expected_ll(model, params, n_trans, subst_suffstats=None,
                pi_obs_counts=None):
    """Q(θ) = Q_struct(θ) + Q_subst(θ) + Q_π(θ), differentiable w.r.t. ALL params.

    Q_struct = Σ_ij E[n_ij] · log τ_ij(θ)     (indel, weights, ext)
    Q_subst depends on model:
      tkf91 / tkf91_cond / tkf92 / mixdom (single-Q):
        Σ_{i≠j} agg_u_ij · log Q_ij + Σ_i agg_w_i · Q_ii
        + Σ_a V[a] · log π[a]      (linear-π observation term, optional)
      mixdom2 (per-class):
        Σ_c [Σ_{a≠b} U_c[a,b] · log Q_c[a,b] + Σ_a W_c[a] · Q_c[a,a]]
        + Σ_c Σ_a V_c[c, a] · log π_c[a]   (linear-π factor)
        + Σ_d Σ_f Σ_c classdist_counts[d,f,c] · log classdist[d,f,c]

    The n_trans and subst_suffstats are FIXED from the E-step.
    Only the parameters in `params` are differentiated.

    The EM/score-function identity guarantees
    `∇_θ log P(data | θ)|_{θ_old} = ∇_θ expected_ll(θ; δ@θ_old)|_{θ_old}`
    so this function delivers the observed-data gradient at the current
    parameters via a closed-form-in-θ expression — no FB inside the
    bwd graph. JIT signature is parameter-shape only.

    Args:
        model: 'tkf91', 'tkf91_cond', 'tkf92', 'mixdom', or 'mixdom2'.
        params: dict with ALL model parameters. For mixdom2 must include
            `class_S_exch` (C, A, A), `class_pis` (C, A),
            `classdist` (D, F, C).
        n_trans: (N, N) expected transition counts (fixed).
        subst_suffstats:
            For tkf91/92/mixdom: tuple `(agg_u, agg_w)` from
                `substitution_suffstats()`.
            For mixdom2: dict with keys `class_W` (C, A),
                `class_U` (C, A, A), `class_match_counts` (C, A, A),
                `class_insert_counts` (C, A), `class_delete_counts` (C, A),
                `classdist_counts` (D, F, C).
            If None, only Q_struct is returned.
        pi_obs_counts:
            For tkf91/92/mixdom only. Optional dict with keys
                `match_anc_count`  (A,)
                `insert_count`     (A,)
                `delete_count`     (A,)
            from `observation_counts_2d`. When provided, adds the
            `Σ_a V[a] · log π[a]` term (V = sum of the three counts).
            Required for the Q-function to match the OBSERVATION
            log-likelihood w.r.t. π (π appears as the per-position
            emission factor at all M/I/D states); without it,
            `expected_ll`'s π-gradient is zero, while the wrapper's
            `_pi_emission_gradient` is non-zero. Ignored for mixdom2,
            where `class_pis` is handled by `_class_pis_linear_q_term`.

    Returns: scalar Q(θ; δ).
    """
    # ---- Structural (chi/τ) Q term — same shape for all 5 models ----
    if model in ('tkf91', 'tkf91_cond'):
        tau = tkf91_trans(params['ins_rate'], params['del_rate'], params['t'])
        q_struct = jnp.sum(n_trans * safe_log(tau))
    elif model == 'tkf92':
        tau = tkf92_trans(params['ins_rate'], params['del_rate'],
                          params['t'], params['ext'])
        q_struct = jnp.sum(n_trans * safe_log(tau))
    elif model in ('mixdom', 'mixdom2'):
        # Both MixDom1 and MixDom2 share the same chi term — chi is built
        # from the structural params (main rates, per-domain rates,
        # weights, ext) regardless of whether site classes are present.
        q_struct = _chi_weighted_loglik(
            params['main_ins'], params['main_del'], params['t'],
            jnp.array(params['dom_ins']), jnp.array(params['dom_del']),
            jnp.array(params['dom_weights']),
            jnp.array(params['frag_weights']),
            jnp.array(params['ext_rates']),
            n_trans)
    else:
        raise ValueError(f"Unknown model: {model}")

    if subst_suffstats is None:
        return q_struct

    # ---- Substitution + per-class terms (model-dependent) ----
    if model == 'mixdom2':
        # subst_suffstats is a dict for MixDom2.
        ss = subst_suffstats
        q_subst = _subst_q_from_class_suffstats(
            jnp.asarray(params['class_S_exch']),
            jnp.asarray(params['class_pis']),
            jnp.asarray(ss['class_W']),
            jnp.asarray(ss['class_U']))
        q_classpi = _class_pis_linear_q_term(
            jnp.asarray(params['class_pis']),
            jnp.asarray(ss['class_match_counts']),
            jnp.asarray(ss['class_insert_counts']),
            jnp.asarray(ss['class_delete_counts']))
        q_classdist = _classdist_q_term(
            jnp.asarray(params['classdist']),
            jnp.asarray(ss['classdist_counts']))
        return q_struct + q_subst + q_classpi + q_classdist

    # ---- tkf91 / tkf91_cond / tkf92 / mixdom: single-Q substitution ----
    #
    # Phase 5.6 Sub-phase B note: this branch keeps the legacy single-Q
    # parameterization for TKF91/TKF92 (structurally single-Q) and
    # external `expected_ll('mixdom', {Q, pi, ...})` consumers (FSA /
    # benchmarks / EM-identity tests). The PRODUCTION Adam pipeline
    # for MixDom1 uses per-domain (`dom_S_exch`, `dom_pis`, `dom_Qs`)
    # via `_subst_q_from_dom_suffstats` + `_dom_pis_linear_q_term` —
    # NOT this branch. If a future caller wires production training
    # through `expected_ll('mixdom', ...)` with a `to_constrained(uc)`
    # output, they'll hit a KeyError on `params['Q']` here, surfacing
    # the parameterization mismatch loud rather than silently
    # consuming a stale Q.
    if 'Q' not in params or 'pi' not in params:
        raise KeyError(
            f"expected_ll(model={model!r}) requires the legacy single-Q "
            f"parameterization (params['Q'], params['pi']). The "
            f"production Adam loss for MixDom1 uses per-domain "
            f"(dom_S_exch, dom_pis) via adam_train.adam_loss_from_delta; "
            f"do not route per-domain training through this path.")
    agg_u, agg_w = subst_suffstats
    q_subst = _subst_q_from_suffstats(params['Q'], agg_u, agg_w)
    total = q_struct + q_subst

    if pi_obs_counts is not None:
        # V[a] = match_anc[a] + insert[a] + delete[a] (observation count
        # at character a across M, I, D positions).
        V = (jnp.asarray(pi_obs_counts['match_anc_count'])
             + jnp.asarray(pi_obs_counts['insert_count'])
             + jnp.asarray(pi_obs_counts['delete_count']))
        q_pi = _pi_linear_q_term(jnp.asarray(params['pi']), V)
        total = total + q_pi

    return total


def _subst_q_from_suffstats(Q, agg_u, agg_w):
    """Q_subst = Σ_{i≠j} u_ij · log Q_ij + Σ_i w_i · Q_ii

    agg_u, agg_w are FIXED numpy arrays (from Holmes-Rubin E-step).
    Only Q is traced by JAX, making this differentiable to all orders
    w.r.t. Q (the function is analytic in Q's off-diagonal entries).

    Note: if differentiability w.r.t. the E-step parameters were needed
    (e.g. for the EM Hessian), one could replace this with the marginal
    emission log-likelihood Σ w_ab · log expm(Qt)_{ab} via Padé
    approximation, which is fully JAX-differentiable but gives a
    different mathematical object (observed emission LL, not the
    expected complete-data LL).
    """
    n = Q.shape[0]
    u = jnp.array(agg_u)
    w = jnp.array(agg_w)
    is_diag = jnp.eye(n, dtype=bool)

    # Off-diagonal: u_ij · log Q_ij (masked to avoid log(0) on diagonal)
    log_Q_safe = jnp.where(is_diag, 0.0, jnp.log(jnp.maximum(Q, 1e-30)))
    q_off = jnp.sum(jnp.where(is_diag, 0.0, u * log_Q_safe))

    # Diagonal: w_i · Q_ii
    q_diag = jnp.sum(w * jnp.diag(Q))

    return q_off + q_diag


# ================================================================
# MixDom2 substitution Q-function (per-class HR + classdist + linear-π)
# ================================================================

def _build_class_Q_from_S_pi(class_S_exch, class_pis):
    """Per-class CTMC rate matrix Q_c built from (S_c, π_c) without expm.

    Q_c[a, b] = S_c[a, b] · π_c[b]  for  a ≠ b
    Q_c[a, a] = -Σ_{b≠a} Q_c[a, b]

    Returns: (C, A, A) full Q stack.

    Construction is purely algebraic — NOT a representative / averaged /
    consensus Q. Each class c has its own Q_c, fully differentiable
    w.r.t. (class_S_exch[c], class_pis[c]) via standard JAX autograd.
    """
    A = class_pis.shape[-1]
    eye_A = jnp.eye(A, dtype=class_S_exch.dtype)
    is_diag_C = eye_A[None, :, :].astype(bool)                       # (1, A, A)

    # Off-diagonal entries; diagonal forced to 0.
    Q_off = class_S_exch * class_pis[:, None, :]                     # (C, A, A)
    Q_off = jnp.where(is_diag_C, 0.0, Q_off)
    # Diagonal as -row-sum.
    Q_diag = -Q_off.sum(axis=-1)                                     # (C, A)
    Q_c = Q_off + Q_diag[:, :, None] * eye_A[None, :, :]              # (C, A, A)
    return Q_c


def _subst_q_from_class_suffstats(class_S_exch, class_pis,
                                    class_W, class_U):
    """Per-class HR substitution Q-function (no expm, no eigh).

    Σ_c [ Σ_{a≠b} U_c[a, b] · log Q_c[a, b]
        + Σ_a    W_c[a]    · Q_c[a, a] ]

    where  Q_c[a, b] = S_c[a, b] · π_c[b]  (a ≠ b),
           Q_c[a, a] = -Σ_{b≠a} Q_c[a, b].

    Per-class summation is explicit (sum over c is the outermost sum,
    NOT an average — each class c contributes its own dwell-time and
    transition-count tensor weighted by its own posterior assignment
    counts). class_W and class_U are FIXED suff-stat arrays from the
    HR E-step (`holmes_rubin_weighted_stats` per class, accumulated
    across pairs at each pair's own t_p in `_process_pairs_batched`).

    HR term uses log Q (full off-diagonal rate, including the π factor)
    NOT log S (exchangeability alone). The "U-feedback" on log π_c[b]
    enters automatically through `log Q_c[a,b] = log S_c[a,b] + log π_c[b]`,
    so V_c_linear in `_class_pis_linear_q_term` is the OBSERVATION-only
    count V_c (NOT V'_c with the U feedback added — that would
    double-count).
    """
    A = class_pis.shape[-1]
    eye_A = jnp.eye(A, dtype=class_S_exch.dtype)
    is_diag_C = eye_A[None, :, :].astype(bool)                       # (1, A, A)

    Q_off = class_S_exch * class_pis[:, None, :]                     # (C, A, A)
    Q_off = jnp.where(is_diag_C, 0.0, Q_off)
    Q_diag = -Q_off.sum(axis=-1)                                     # (C, A)

    # Off-diagonal: Σ_c Σ_{a≠b} U_c[a,b] · log Q_c[a,b].
    log_Q_off_safe = jnp.where(is_diag_C, 0.0,
                                jnp.log(jnp.maximum(Q_off, 1e-30)))
    u_safe = jnp.where(is_diag_C, 0.0, class_U)
    q_off = jnp.sum(u_safe * log_Q_off_safe)

    # Diagonal: Σ_c Σ_a W_c[a] · Q_c[a,a].
    q_diag = jnp.sum(class_W * Q_diag)

    return q_off + q_diag


def _class_pis_linear_q_term(class_pis,
                               class_match_counts,
                               class_insert_counts,
                               class_delete_counts):
    """Linear-π factor term: Σ_c Σ_a V_c[c, a] · log π_c[a].

    V_c[c, a] = class_match_counts[c, a, :].sum(-1)   (M, anc factor)
              + class_insert_counts[c, a]             (I, desc factor)
              + class_delete_counts[c, a]             (D, anc factor)

    NOTE on V_c vs V'_c: this is V_c (observation-only); V'_c =
    V_c + Σ_{j≠i} U_c[j, i] would DOUBLE-COUNT, because the HR term in
    `_subst_q_from_class_suffstats` already provides the U feedback via
    log Q_c[a,b] = log S_c[a,b] + log π_c[b].

    See memory: project_q_function_v_c_linear_trap.md.
    """
    V_c = (class_match_counts.sum(axis=-1)                           # (C, A)
           + class_insert_counts
           + class_delete_counts)
    log_pi_safe = jnp.log(jnp.maximum(class_pis, 1e-30))
    return jnp.sum(V_c * log_pi_safe)


def _classdist_q_term(classdist, classdist_counts):
    """Per-(domain, fragment) class-mixture term:
    Σ_d Σ_f Σ_c classdist_counts[d, f, c] · log classdist[d, f, c].

    Sum is explicit per-(d, f, c); NOT averaged. Each (d, f) keeps its
    own categorical class distribution.
    """
    log_cd_safe = jnp.log(jnp.maximum(classdist, 1e-30))
    return jnp.sum(classdist_counts * log_cd_safe)


# ================================================================
# Per-domain HR + linear-π (Phase 5.6 Sub-phase B; parallels the
# per-class versions above for MixDom2).
# ================================================================

def _subst_q_from_dom_suffstats(dom_Qs, dom_W, dom_U):
    """Per-domain HR substitution Q-function (no expm, no eigh).

    Σ_d [ Σ_{a≠b} U_d[a, b] · log Q_d[a, b]
        + Σ_a    W_d[a]    · Q_d[a, a] ]

    Per-domain summation is explicit (sum over d is the outermost sum,
    NOT an average — each domain d contributes its own dwell-time and
    transition-count tensor). dom_W and dom_U are FIXED suff-stat
    arrays from the HR E-step (`holmes_rubin_weighted_stats` per
    domain, accumulated across pairs at each pair's own t_p inside
    `_process_pairs_batched`'s vmapped HR path).

    Args:
        dom_Qs: (D, A, A) per-domain rate matrix stack. Built from
            `dom_S_exch` and `dom_pis` via
            `jax.vmap(build_Q_from_S_pi)(dom_S_exch, dom_pis)` upstream.
        dom_W: (D, A) per-domain dwell-time aggregates.
        dom_U: (D, A, A) per-domain off-diagonal transition-count
            aggregates (diagonal is zero / ignored).

    Returns: scalar Σ_d HR(Q_d).

    The HR term uses `log Q_d[a, b]` (full off-diagonal rate, including
    the π_d factor); the U-feedback to `log π_d[b]` is implicit through
    the chain `log Q_d[a, b] = log S_d[a, b] + log π_d[b]`. Adding it
    again in `_dom_pis_linear_q_term` would double-count — that linear
    term uses the OBSERVATION-only count V_d, not V'_d.
    """
    A = dom_Qs.shape[-1]
    eye_A = jnp.eye(A, dtype=dom_Qs.dtype)
    is_diag_D = eye_A[None, :, :].astype(bool)                       # (1, A, A)

    # Off-diagonal: Σ_d Σ_{a≠b} U_d[a, b] · log Q_d[a, b].
    log_Q_off_safe = jnp.where(is_diag_D, 0.0,
                                jnp.log(jnp.maximum(dom_Qs, 1e-30)))
    u_safe = jnp.where(is_diag_D, 0.0, dom_U)
    q_off = jnp.sum(u_safe * log_Q_off_safe)

    # Diagonal: Σ_d Σ_a W_d[a] · Q_d[a, a].
    Q_diag = jnp.diagonal(dom_Qs, axis1=-2, axis2=-1)                # (D, A)
    q_diag = jnp.sum(dom_W * Q_diag)

    return q_off + q_diag


def _dom_pis_linear_q_term(dom_pis,
                            dom_match_counts,
                            dom_insert_counts,
                            dom_delete_counts):
    """Per-domain linear-π factor: Σ_d Σ_a V_d[a] · log π_d[a].

    V_d[a] = dom_match_counts[d, a, :].sum(-1)   (M, anc factor)
           + dom_insert_counts[d, a]             (I, desc factor)
           + dom_delete_counts[d, a]             (D, anc factor)

    OBSERVATION-only V_d (NOT V'_d). The U feedback to log π_d[b]
    enters automatically through `_subst_q_from_dom_suffstats`'s
    `log Q_d[a,b] = log S_d[a,b] + log π_d[b]`. See
    `project_q_function_v_c_linear_trap.md`.

    Args:
        dom_pis: (D, A) per-domain stationary distributions.
        dom_match_counts: (D, A, A) per-domain match emission counts.
        dom_insert_counts: (D, A) per-domain insert emission counts.
        dom_delete_counts: (D, A) per-domain delete emission counts.

    Returns: scalar Σ_d Σ_a V_d[a] log π_d[a].
    """
    V_d = (dom_match_counts.sum(axis=-1)                              # (D, A)
           + dom_insert_counts
           + dom_delete_counts)
    log_pi_safe = jnp.log(jnp.maximum(dom_pis, 1e-30))
    return jnp.sum(V_d * log_pi_safe)


def chi_q_from_bdi(params, exact_ss):
    """Chi-axis Q-function in the BDI form (per-pair-t correct, t-free).

    Q_chi(θ) = ℓ_top(λ_main, μ_main; B_top, D_top, L_top, M_top, S_top, T_top)
             + Σ_d ℓ_dom(λ_d, μ_d; B_d, D_d, L_d, M_d, S_d, T_d)
             + Σ_d dom_w[d] · log dom_weights[d]
             + Σ_d Σ_f frag_w[d, f] · log frag_weights[d, f]
             + Σ_d Σ_f Σ_g ext[d, f, g] · log ext_rates[d, f, g]
                + Σ_d Σ_f term[d, f] · log(1 - Σ_g ext_rates[d, f, g])

    where ℓ(λ, μ; B, D, L, M, S, T) is the TKF Baum-Welch BDI form
    (tkf.tex eq lines 783-786, sec `bw-tkf91`):

        ℓ(λ, μ; B, D, L, M, S, T)
          = (B + L) log λ + (D - L - M) log μ + M log(μ - λ)
          - (λ + μ) S - λ T

    **t-parameter-free**: t enters only through the aggregated S and T,
    each of which is a sum of per-pair-t-correct integrals computed
    inside `exact_suffstats_per_pair_batch`. Differentiating in (λ, μ)
    and setting to zero recovers the κ-quadratic of
    `m_step_indel_quadratic` (eq:kappa-quadratic), confirming this is
    the EM Q-function for the chi axis.

    Args:
        params: dict with `main_ins`, `main_del`, `dom_ins` (D,),
            `dom_del` (D,), `dom_weights` (D,), `frag_weights` (D, F),
            `ext_rates` (D, F, F).
        exact_ss: dict from `exact_suffstats_per_pair_batch` (also
            available at `delta['exact_ss']` from the SVI-BW E-step).
            Required keys:
              - `top_5x5` (5, 5)
              - `dom_M_5x5` (D, 5, 5) (or list of D (5, 5) tensors)
              - `top_E_B`, `top_E_D`, `top_E_S`, `top_T_obs` (scalars)
              - `dom_E_B`, `dom_E_D`, `dom_E_S`, `dom_T_obs` (D,)
              - `dom_w` (D,), `frag_w` (D, F)
              - `ext` (D, F, F), `term` (D, F)

    Returns: scalar Q_chi.

    NOTE: this function does NOT consume `delta['agg_n_chi']` or any
    other diagnostic aggregate. It uses the per-pair-t-correct BDI
    suff stats directly. See
    `.claude/examples/per_pair_t_chi_axis_recidivism.md` for the
    backstory on the agg_n_chi trap.
    """
    from ..core.params import M as M_idx, D as D_idx, E as E_idx

    lam_main = jnp.asarray(params['main_ins'])
    mu_main = jnp.asarray(params['main_del'])
    dom_lam = jnp.asarray(params['dom_ins'])           # (D,)
    dom_mu = jnp.asarray(params['dom_del'])            # (D,)
    dom_weights = jnp.asarray(params['dom_weights'])   # (D,)
    frag_weights = jnp.asarray(params['frag_weights']) # (D, F)
    ext_rates = jnp.asarray(params['ext_rates'])       # (D, F, F)

    # ---- Top-level (λ_main, μ_main) ----
    # L = n_kappa = (col M sum) + (col D sum) — every transition that
    # CONSUMES an ancestor character (κ-event), regardless of source row.
    # M = n_{1-κ} = (col E sum) — terminations.
    # See bdi.py:369-372 (n_log_kappa) and compiled.py:553 (top_nk).
    # Earlier audit caught a row-S-sum bug here.
    top_5x5 = jnp.asarray(exact_ss['top_5x5'])
    top_L = jnp.sum(top_5x5[:, M_idx]) + jnp.sum(top_5x5[:, D_idx])
    top_M = jnp.sum(top_5x5[:, E_idx])
    top_B = jnp.asarray(exact_ss['top_E_B'])
    top_D = jnp.asarray(exact_ss['top_E_D'])
    top_S = jnp.asarray(exact_ss['top_E_S'])
    top_T = jnp.asarray(exact_ss['top_T_obs'])
    q_top = (
        (top_B + top_L) * jnp.log(jnp.maximum(lam_main, 1e-30))
        + (top_D - top_L - top_M) * jnp.log(jnp.maximum(mu_main, 1e-30))
        + top_M * jnp.log(jnp.maximum(mu_main - lam_main, 1e-30))
        - (lam_main + mu_main) * top_S
        - lam_main * top_T)

    # ---- Per-domain (dom_ins[d], dom_del[d]) ----
    # `dom_M_5x5` may be (D, 5, 5) ndarray OR list of D (5, 5) tensors.
    # Per-domain L = (col M + col D) of dom_M_5x5 PLUS dom_kappa
    # (the I/D-type fragment contribution tracked separately).
    # Per-domain M = (col E) PLUS dom_1mkappa.
    # See compiled.py:576-579 (n_kappa, n_1mkappa).
    dom_5x5 = jnp.asarray(exact_ss['dom_M_5x5'])        # (D, 5, 5)
    dom_kappa = jnp.asarray(exact_ss['dom_kappa'])       # (D,)
    dom_1mkappa = jnp.asarray(exact_ss['dom_1mkappa'])   # (D,)
    dom_L = (jnp.sum(dom_5x5[:, :, M_idx], axis=-1)
             + jnp.sum(dom_5x5[:, :, D_idx], axis=-1)
             + dom_kappa)                                 # (D,)
    dom_M = jnp.sum(dom_5x5[:, :, E_idx], axis=-1) + dom_1mkappa  # (D,)
    dom_B = jnp.asarray(exact_ss['dom_E_B'])             # (D,)
    dom_D = jnp.asarray(exact_ss['dom_E_D'])             # (D,)
    dom_S = jnp.asarray(exact_ss['dom_E_S'])             # (D,)
    dom_T = jnp.asarray(exact_ss['dom_T_obs'])           # (D,)
    q_dom = jnp.sum(
        (dom_B + dom_L) * jnp.log(jnp.maximum(dom_lam, 1e-30))
        + (dom_D - dom_L - dom_M) * jnp.log(jnp.maximum(dom_mu, 1e-30))
        + dom_M * jnp.log(jnp.maximum(dom_mu - dom_lam, 1e-30))
        - (dom_lam + dom_mu) * dom_S
        - dom_lam * dom_T)

    # ---- Domain mixture weights (multinomial / Dirichlet form) ----
    log_dw = jnp.log(jnp.maximum(dom_weights, 1e-30))
    q_dom_w = jnp.sum(jnp.asarray(exact_ss['dom_w']) * log_dw)

    # ---- Fragment weights per domain (multinomial per row) ----
    log_fw = jnp.log(jnp.maximum(frag_weights, 1e-30))
    q_frag_w = jnp.sum(jnp.asarray(exact_ss['frag_w']) * log_fw)

    # ---- ext_rates: (D, F, F) extension probs + termination ----
    # ext_rates[d, f, :] sums to <1; term_prob[d, f] = 1 - row_sum.
    log_ext = jnp.log(jnp.maximum(ext_rates, 1e-30))                  # (D, F, F)
    term_prob = 1.0 - jnp.sum(ext_rates, axis=-1)                     # (D, F)
    log_term = jnp.log(jnp.maximum(term_prob, 1e-30))                 # (D, F)
    q_ext = (jnp.sum(jnp.asarray(exact_ss['ext']) * log_ext)
             + jnp.sum(jnp.asarray(exact_ss['term']) * log_term))

    return q_top + q_dom + q_dom_w + q_frag_w + q_ext


def _pi_linear_q_term(pi, V):
    """Linear-π factor for single-Q models (TKF91/TKF92/MixDom1):

        Σ_a V[a] · log π[a]

    where  V[a] = match_anc_count[a] + insert_count[a] + delete_count[a]

    is the total observed-character count (across M, I, D positions
    weighted by FB posteriors) at character a.

    This term captures the π's appearance in the OBSERVATION
    log-likelihood — log π[a] for every emitted character — that the
    HR substitution Q-function `Σ U log Q + Σ W Q_diag` does NOT
    include for a free-Q parameterization. Without it, expected_ll's
    gradient w.r.t. π would be zero from the substitution side; the
    wrapper's gradient (via `_pi_emission_gradient`) includes V/π,
    so the two would disagree on π until this term is added.

    NOTE: this is the SAME observation-only V (NOT V'), analogous to
    `_class_pis_linear_q_term`'s V_c. The U feedback is automatically
    included via `log Q_ij = log S_ij + log π[j]` IF Q is parameterised
    via (S, π). For a free-Q parameterisation Q has no S/π split,
    log Q is a free function, and U feedback to π does not occur
    through this term. (See feedback_q_function_v_c_linear_trap memory.)
    """
    log_pi_safe = jnp.log(jnp.maximum(pi, 1e-30))
    return jnp.sum(V * log_pi_safe)


def observation_counts_2d(posteriors, state_types, x_seq, y_seq, alphabet_size):
    """Aggregate per-character observation counts from FB posteriors.

    For each (i, j) grid cell with body state s of type τ ∈ {M, I, D}:
      - τ = M: x[i-1] is the anc emission, y[j-1] is the desc.
      - τ = I: y[j-1] is the desc emission only.
      - τ = D: x[i-1] is the anc emission only.

    Aggregates posterior weight onto per-character counts:
      match_anc_count[a] = Σ_(i, j, s ∈ M-type, x[i-1]=a) posteriors[i, j, s]
      insert_count[a]    = Σ_(i, j, s ∈ I-type, y[j-1]=a) posteriors[i, j, s]
      delete_count[a]    = Σ_(i, j, s ∈ D-type, x[i-1]=a) posteriors[i, j, s]

    Reachability ranges in the (Lx+1) × (Ly+1) FB grid:
      M: i ∈ [1, Lx], j ∈ [1, Ly]
      I: i ∈ [0, Lx], j ∈ [1, Ly]   (no x consumed at i=0; insert can
                                      happen at any prefix of x)
      D: i ∈ [1, Lx], j ∈ [0, Ly]   (no y consumed at j=0; delete can
                                      happen at any prefix of y)

    Each state-type uses its OWN slicing — matches the legacy
    `_pi_emission_gradient` in `vjp.py:58-118`. The earlier version of
    this function (Phase 4.5 introduction) used the M-slicing for all
    three types and dropped the i=0 / j=0 boundary contributions for
    I and D — silently corrupting the V·log π linear factor in
    `expected_ll` by a few characters' worth of posterior weight per
    pair (specifically, by `≈ #inserts at i=0 + #deletes at j=0`).
    Caught 2026-04-30 via FD-vs-HR EM-identity probe on log_pi_logits.

    Returns a dict with three (A,) arrays.
    """
    from ..core.params import M as M_T, I as I_T, D as D_T

    Lx = int(x_seq.shape[0])
    Ly = int(y_seq.shape[0])
    is_M_state = (state_types == M_T).astype(posteriors.dtype)
    is_I_state = (state_types == I_T).astype(posteriors.dtype)
    is_D_state = (state_types == D_T).astype(posteriors.dtype)

    # Per-(i, j) sums of posteriors over states of each type.
    M_post_full = jnp.einsum('ijs,s->ij', posteriors, is_M_state)
    I_post_full = jnp.einsum('ijs,s->ij', posteriors, is_I_state)
    D_post_full = jnp.einsum('ijs,s->ij', posteriors, is_D_state)

    # Per-state-type slicing (different for I and D from M).
    M_post = M_post_full[1:Lx + 1, 1:Ly + 1]   # (Lx, Ly)
    I_post = I_post_full[0:Lx + 1, 1:Ly + 1]   # (Lx+1, Ly) — i=0 included
    D_post = D_post_full[1:Lx + 1, 0:Ly + 1]   # (Lx, Ly+1) — j=0 included

    one_hot_x = jax.nn.one_hot(jnp.asarray(x_seq, dtype=jnp.int32),
                                 alphabet_size, dtype=posteriors.dtype)  # (Lx, A)
    one_hot_y = jax.nn.one_hot(jnp.asarray(y_seq, dtype=jnp.int32),
                                 alphabet_size, dtype=posteriors.dtype)  # (Ly, A)

    # Match: x[i-1] (anc) for i ∈ [1, Lx], y[j-1] for j ∈ [1, Ly].
    match_anc_count = jnp.einsum('ij,ia->a', M_post, one_hot_x)
    # Insert: y[j-1] for j ∈ [1, Ly]; sum over all i (no x consumed).
    I_per_j = jnp.sum(I_post, axis=0)                         # (Ly,)
    insert_count = jnp.einsum('j,ja->a', I_per_j, one_hot_y)
    # Delete: x[i-1] for i ∈ [1, Lx]; sum over all j (no y consumed).
    D_per_i = jnp.sum(D_post, axis=1)                         # (Lx,)
    delete_count = jnp.einsum('i,ia->a', D_per_i, one_hot_x)
    return {
        'match_anc_count': match_anc_count,
        'insert_count': insert_count,
        'delete_count': delete_count,
    }
