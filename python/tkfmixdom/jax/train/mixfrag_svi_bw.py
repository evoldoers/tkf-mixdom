"""MixFrag (TKF92 with fragment mixtures) SVI-Baum-Welch training on pair data.

MixFrag promotes the TKF92 fragment-extension parameter to a per-fragment
categorical latent variable: each fragment draws a fragtype f ~ Cat(weights)
and then extends geometrically with fragtype-specific probability exts[f].
Substitution and indel (BDI) processes are shared across fragtypes; the only
new parameters are exts (r_f) and weights (w_f).  See tkf/mixfrag.tex.

Because the fragtypes are LATENT (not recoverable from the alignment), the
per-fragtype fragment-extension/end counts F_f, E_f are not functions of the
observed alignment alone, so the cherry-count memoization of TKF92/Maraschino
is unavailable.  Training therefore marginalises the alignment AND the
fragtypes inside a 2D pair-HMM forward-backward E-step on the (3F+2)-state
MixFrag chi matrix, exactly the SVI-BW pattern of ``tkf92_svi_bw`` generalised
to F fragtypes.

This module is the F-fragtype analogue of ``tkf92_svi_bw``; it reuses that
module's bin-bucketing, BDI-stats, and (lambda, mu) kappa-quadratic M-step,
and adds a per-fragtype Beta M-step for ext_f and a Dirichlet M-step for the
fragtype weights w_f.

Public API:
  svi_bw_mixfrag(pair_iter, n_total_pairs, n_fragtypes, ...) -> dict
  estep_batch_mixfrag(xs_pad, ys_pad, ts, Lx_real, Ly_real,
                      lam, mu, exts, weights, Q, pi) -> dict
"""

from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

from ..core.ctmc import transition_matrix
from ..core.params import (
    S as TYPE_S, M as TYPE_M, I as TYPE_I, D as TYPE_D, E as TYPE_E,
    tkf91_trans, mixfrag_trans, mixfrag_pair_index,
)
from ..dp.hmm import forward_backward_2d
from .tkf92_svi_bw import (
    _bin_bucket_pairs, _stack_bucket, _bdi_stats_batch, m_step_lam_mu,
)


# -------------------------------------------------------------------------
# Chi decomposition (per-fragtype) — JAX-traceable.
# -------------------------------------------------------------------------


def _mixfrag_state_type_ids(F):
    """numpy (3F+2,) array of TKF91 type ids for the MixFrag Pair HMM states
    (S, M_1..M_F, I_1..I_F, D_1..D_F, E)."""
    return np.array([TYPE_S] + [TYPE_M] * F + [TYPE_I] * F + [TYPE_D] * F
                    + [TYPE_E])


def _aggregate_chi_to_5x5(mat, F):
    """Sum a (3F+2, 3F+2) MixFrag count matrix down to a 5x5 TKF91-type matrix
    (S, M, I, D, E), mapping M_f->M, I_f->I, D_f->D, summing over fragtypes.
    The indel/substitution processes are shared across fragtypes, so the
    BDI sufficient statistics consume the fragtype-summed counts."""
    type_ids = _mixfrag_state_type_ids(F)
    out = jnp.zeros((5, 5))
    for it in range(5):
        mi = (type_ids == it).astype(np.float64)
        for jt in range(5):
            mj = (type_ids == jt).astype(np.float64)
            out = out.at[it, jt].set(jnp.sum(mat * mi[:, None] * mj[None, :]))
    return out


def _mixfrag_chi_resolve_core(n_chi, ins_rate, del_rate, t, exts, weights):
    """Resolve the (3F+2)^2 chi count matrix into per-fragtype fragment
    statistics and the fragtype-summed 5x5 TKF91-type matrix.

    Per mixfrag.tex sec:bw-mixfrag:
      * F_{af} = ext_frac[a, f] * n_chi[a_f, a_f],  with
        ext_frac[a, f] = ext_f / (ext_f + (1-ext_f) * tau91[a, a] * w_f)
        the fraction of the a_f->a_f self-loop attributable to fragment
        extension (the new-link branch re-draws the same fragtype, hence w_f).
      * F_f = sum_a F_{af}  (fragment-extension count for fragtype f).
      * E_f = sum_a sum_dest n_chi[a_f, dest] - F_f  (fragment-end count =
        number of type-f fragments).
      * resolved n̂_{a_f a_f} = n_chi[a_f, a_f] - F_{af}; off-diagonals
        unchanged.  Summed over fragtypes -> 5x5 TKF91-level counts.

    Returns (resolved5, F_f, E_f, L_coef, M_coef).
    """
    tau91 = tkf91_trans(ins_rate, del_rate, t)
    F = int(exts.shape[0])
    _, M0, I0, D0, _ = mixfrag_pair_index(F)
    blocks = ((M0, TYPE_M), (I0, TYPE_I), (D0, TYPE_D))

    resolved = n_chi
    F_f = jnp.zeros(F)
    E_f = jnp.zeros(F)
    for (base, ty) in blocks:
        diag91 = tau91[ty, ty]
        for f in range(F):
            idx = base + f
            selfloop_w = exts[f] + (1.0 - exts[f]) * diag91 * weights[f]
            ext_frac = jnp.where(selfloop_w > 1e-30,
                                 exts[f] / jnp.maximum(selfloop_w, 1e-30), 0.0)
            ext_frac = jnp.clip(ext_frac, 0.0, 1.0)
            F_af = ext_frac * n_chi[idx, idx]
            F_f = F_f.at[f].add(F_af)
            E_f = E_f.at[f].add(jnp.sum(n_chi[idx, :]))   # total out from a_f
            resolved = resolved.at[idx, idx].add(-F_af)
    E_f = E_f - F_f

    resolved5 = _aggregate_chi_to_5x5(resolved, F)
    # L (log_kappa coef): dest in {M, D}; M (log_1mkappa coef): dest = E.
    L_coef = (resolved5[TYPE_S, TYPE_M] + resolved5[TYPE_S, TYPE_D]
              + resolved5[TYPE_M, TYPE_M] + resolved5[TYPE_M, TYPE_D]
              + resolved5[TYPE_I, TYPE_M] + resolved5[TYPE_I, TYPE_D]
              + resolved5[TYPE_D, TYPE_M] + resolved5[TYPE_D, TYPE_D])
    M_coef = (resolved5[TYPE_S, TYPE_E] + resolved5[TYPE_M, TYPE_E]
              + resolved5[TYPE_I, TYPE_E] + resolved5[TYPE_D, TYPE_E])
    return resolved5, F_f, E_f, L_coef, M_coef


# -------------------------------------------------------------------------
# Per-pair E-step core (JAX-only, vmappable, jittable).
# -------------------------------------------------------------------------


def _estep_pair_mixfrag_core(x_pad, y_pad, t, Lx_real, Ly_real,
                             lam, mu, exts, weights, Q, pi):
    """Per-pair MixFrag E-step core.  Returns dict with log_p, resolved5,
    F_f, E_f, L, M.  Padding semantics match _estep_pair_tkf92_core."""
    tau = mixfrag_trans(lam, mu, t, exts, weights)
    log_trans = jnp.log(jnp.maximum(tau, 1e-30))
    F = int(exts.shape[0])
    state_types = jnp.array(
        [TYPE_S] + [TYPE_M] * F + [TYPE_I] * F + [TYPE_D] * F + [TYPE_E],
        dtype=jnp.int32)
    sub = transition_matrix(Q, t)
    log_p, _, n_chi = forward_backward_2d(
        log_trans, state_types, x_pad, y_pad, sub, pi,
        real_Lx=Lx_real, real_Ly=Ly_real)
    resolved5, F_f, E_f, L_coef, M_coef = _mixfrag_chi_resolve_core(
        n_chi, lam, mu, t, exts, weights)
    return {
        'log_p': log_p,
        'resolved5': resolved5,
        'F_f': F_f,
        'E_f': E_f,
        'L': L_coef,
        'M': M_coef,
    }


def _estep_pair_mixfrag_forward_only_core(x_pad, y_pad, t, Lx_real, Ly_real,
                                          lam, mu, exts, weights, Q, pi):
    """Forward-only variant for val_eval (log_p only)."""
    tau = mixfrag_trans(lam, mu, t, exts, weights)
    log_trans = jnp.log(jnp.maximum(tau, 1e-30))
    F = int(exts.shape[0])
    state_types = jnp.array(
        [TYPE_S] + [TYPE_M] * F + [TYPE_I] * F + [TYPE_D] * F + [TYPE_E],
        dtype=jnp.int32)
    sub = transition_matrix(Q, t)
    return forward_backward_2d(
        log_trans, state_types, x_pad, y_pad, sub, pi,
        real_Lx=Lx_real, real_Ly=Ly_real, forward_only=True)


_ESTEP_MF_FULL_JIT = None
_ESTEP_MF_FWD_JIT = None


def _get_estep_mf_full_jit():
    global _ESTEP_MF_FULL_JIT
    if _ESTEP_MF_FULL_JIT is None:
        vmapped = jax.vmap(
            _estep_pair_mixfrag_core,
            in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None))
        _ESTEP_MF_FULL_JIT = jax.jit(vmapped)
    return _ESTEP_MF_FULL_JIT


def _get_estep_mf_fwd_jit():
    global _ESTEP_MF_FWD_JIT
    if _ESTEP_MF_FWD_JIT is None:
        vmapped = jax.vmap(
            _estep_pair_mixfrag_forward_only_core,
            in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None))
        _ESTEP_MF_FWD_JIT = jax.jit(vmapped)
    return _ESTEP_MF_FWD_JIT


def estep_batch_mixfrag(xs_pad, ys_pad, ts, Lx_real, Ly_real,
                        lam, mu, exts, weights, Q, pi):
    """Batched (vmap'd) MixFrag E-step on a bin-bucketed minibatch.

    Returns dict of jnp arrays:
        'log_p' (B,), 'B','D','S' (B,), 'L','M' (B,), 'T' (B,),
        'F_f' (B, F)  per-fragtype fragment-extension counts,
        'E_f' (B, F)  per-fragtype fragment-end counts (# type-f fragments).
    """
    xs_pad = jnp.asarray(xs_pad)
    ys_pad = jnp.asarray(ys_pad)
    ts = jnp.asarray(ts, dtype=jnp.float64)
    Lx_real = jnp.asarray(Lx_real, dtype=jnp.int32)
    Ly_real = jnp.asarray(Ly_real, dtype=jnp.int32)
    Qj = jnp.asarray(Q)
    pij = jnp.asarray(pi)
    lam_j = jnp.asarray(float(lam))
    mu_j = jnp.asarray(float(mu))
    exts_j = jnp.asarray(exts, dtype=jnp.float64)
    weights_j = jnp.asarray(weights, dtype=jnp.float64)

    fn = _get_estep_mf_full_jit()
    out = fn(xs_pad, ys_pad, ts, Lx_real, Ly_real,
             lam_j, mu_j, exts_j, weights_j, Qj, pij)

    T_batch = ts  # T = t for the shared single-process indel model
    E_B, E_D, E_S = _bdi_stats_batch(out['resolved5'], lam, mu, ts, T_batch)
    E_B = jnp.maximum(E_B, 0.0)
    E_D = jnp.maximum(E_D, 0.0)
    E_S = jnp.maximum(E_S, 0.0)

    return {
        'log_p': out['log_p'],
        'B': E_B, 'D': E_D, 'S': E_S,
        'L': out['L'], 'M': out['M'], 'T': T_batch,
        'F_f': out['F_f'], 'E_f': out['E_f'],
    }


def estep_batch_mixfrag_forward_only(xs_pad, ys_pad, ts, Lx_real, Ly_real,
                                     lam, mu, exts, weights, Q, pi):
    """Forward-only batched E-step (per-pair log_p only) for val_eval."""
    fn = _get_estep_mf_fwd_jit()
    return fn(jnp.asarray(xs_pad), jnp.asarray(ys_pad),
              jnp.asarray(ts, dtype=jnp.float64),
              jnp.asarray(Lx_real, dtype=jnp.int32),
              jnp.asarray(Ly_real, dtype=jnp.int32),
              jnp.asarray(float(lam)), jnp.asarray(float(mu)),
              jnp.asarray(exts, dtype=jnp.float64),
              jnp.asarray(weights, dtype=jnp.float64),
              jnp.asarray(Q), jnp.asarray(pi))


# -------------------------------------------------------------------------
# M-step helpers (lambda/mu reuse tkf92's kappa-quadratic via m_step_lam_mu).
# -------------------------------------------------------------------------


def m_step_ext_per_fragtype(F_f, E_f, prior_alpha=2.0, prior_beta=3.0):
    """Per-fragtype Beta(alpha, beta) MAP for the extension probabilities
    ext_f = (F_f + a - 1) / (F_f + E_f + a + b - 2)."""
    F_f = np.asarray(F_f, dtype=np.float64)
    E_f = np.asarray(E_f, dtype=np.float64)
    a = F_f + prior_alpha - 1.0
    b = E_f + prior_beta - 1.0
    denom = a + b
    return np.where(denom > 1e-9, a / np.maximum(denom, 1e-9), 0.5)


def m_step_weights(E_f, prior_alpha=1.5):
    """Dirichlet(prior_alpha) MAP for the fragtype weights.

    E_f is the expected number of fragments of each fragtype (each fragment
    terminates exactly once), so this is the categorical MLE/MAP:
    w_f = (E_f + alpha - 1) / sum_f' (E_f' + alpha - 1)  (alpha > 1)."""
    E_f = np.asarray(E_f, dtype=np.float64)
    a = np.maximum(E_f + prior_alpha - 1.0, 1e-9)
    return a / a.sum()


# -------------------------------------------------------------------------
# SVI suff-stat accumulator (B,D,S,L,M,T scalars + F_f,E_f arrays).
# -------------------------------------------------------------------------


def _empty_suff_mf(F):
    return {'B': 0.0, 'D': 0.0, 'S': 0.0, 'L': 0.0, 'M': 0.0, 'T': 0.0,
            'F_f': np.zeros(F), 'E_f': np.zeros(F)}


def _scaled_add_mf(blend, mb, eta, scale):
    return {k: (1.0 - eta) * blend[k] + eta * scale * mb[k] for k in blend}


def _aggregate_batch_suff_mf(bd):
    return {
        'B': float(jnp.sum(bd['B'])), 'D': float(jnp.sum(bd['D'])),
        'S': float(jnp.sum(bd['S'])), 'L': float(jnp.sum(bd['L'])),
        'M': float(jnp.sum(bd['M'])), 'T': float(jnp.sum(bd['T'])),
        'F_f': np.asarray(jnp.sum(bd['F_f'], axis=0)),
        'E_f': np.asarray(jnp.sum(bd['E_f'], axis=0)),
    }


# -------------------------------------------------------------------------
# Training loop.
# -------------------------------------------------------------------------


def svi_bw_mixfrag(pair_iter, *, n_total_pairs, n_fragtypes,
                   init_lam=0.05, init_mu=0.05,
                   init_exts=None, init_weights=None,
                   Q, pi,
                   n_iter=200, batch_size=50,
                   svi_tau=1.0, svi_kappa=0.7,
                   prior_alpha_lam=2.0, prior_alpha_mu=2.0, prior_beta=10.0,
                   ext_prior_alpha=2.0, ext_prior_beta=3.0,
                   weight_prior_alpha=1.5,
                   bin_bucketed=False, pre_warm=False,
                   val_pairs=None, val_every=0, patience=0,
                   log_fn=print, seed=0):
    """SVI-BW on MixFrag pair data.

    Args mirror ``svi_bw_tkf92`` but with F = n_fragtypes; ``init_exts`` and
    ``init_weights`` are length-F (defaults: spread exts, uniform weights).
    ``pair_iter`` is a callable() -> generator of (x_int, y_int, t).

    Returns dict with final 'lam', 'mu', 'exts' (F,), 'weights' (F,) and
    'history'; plus best_* if val_pairs given.
    """
    F = int(n_fragtypes)
    rng = np.random.default_rng(seed)
    lam, mu = float(init_lam), float(init_mu)
    if init_exts is None:
        exts = np.linspace(0.3, 0.7, F) if F > 1 else np.array([0.5])
    else:
        exts = np.asarray(init_exts, dtype=np.float64)
    weights = (np.asarray(init_weights, dtype=np.float64)
               if init_weights is not None else np.full(F, 1.0 / F))
    weights = weights / weights.sum()
    assert exts.shape == (F,) and weights.shape == (F,)

    suff_blend = _empty_suff_mf(F)
    history = []

    pairs = list(pair_iter())
    if not pairs:
        raise ValueError('Pair iterator produced no pairs.')
    n_actual = len(pairs)
    log_fn(f'svi_bw_mixfrag: {n_actual} pairs loaded; F={F}; '
           f'n_total_pairs={n_total_pairs}, batch_size={batch_size}, '
           f'n_iter={n_iter}.')

    train_buckets = _bin_bucket_pairs(pairs)
    bucket_keys = list(train_buckets.keys())
    bucket_sizes = np.array([len(train_buckets[k]) for k in bucket_keys])
    bucket_weights = bucket_sizes / bucket_sizes.sum()
    flat_pairs = [(key, j) for key in bucket_keys
                  for j in range(len(train_buckets[key]))]
    n_flat = len(flat_pairs)

    if pre_warm:
        for key in bucket_keys:
            xs, ys, ts, Lx, Ly = _stack_bucket(train_buckets[key][:1])
            _ = estep_batch_mixfrag(xs, ys, ts, Lx, Ly, lam, mu, exts, weights,
                                    Q, pi)

    best_val_ll = -float('inf')
    best_params = (lam, mu, exts.copy(), weights.copy())
    val_no_improve = 0
    val_chunks = None
    if val_pairs:
        val_buckets = _bin_bucket_pairs(val_pairs)
        val_chunks = []
        for lst in val_buckets.values():
            for i in range(0, len(lst), batch_size):
                chunk = lst[i:i + batch_size]
                n_real = len(chunk)
                if n_real < batch_size:
                    chunk = chunk + [chunk[-1]] * (batch_size - n_real)
                val_chunks.append((*_stack_bucket(chunk), n_real))

    scale = float(n_total_pairs) / float(batch_size)
    t0 = time.time()
    for k in range(n_iter):
        eta_k = (svi_tau + k) ** (-svi_kappa)
        if bin_bucketed:
            bk = int(rng.choice(len(bucket_keys), p=bucket_weights))
            pool = train_buckets[bucket_keys[bk]]
            ix = rng.choice(len(pool), batch_size, replace=len(pool) < batch_size)
            sub_buckets = {bucket_keys[bk]: [pool[i] for i in ix]}
        else:
            ix = rng.choice(n_flat, batch_size, replace=False)
            sub_buckets = {}
            for i in ix:
                key, local_idx = flat_pairs[i]
                sub_buckets.setdefault(key, []).append(
                    train_buckets[key][local_idx])

        suff_mb = _empty_suff_mf(F)
        ll_mb = 0.0
        for key, sub in sub_buckets.items():
            x_b, y_b, t_b, Lx_b, Ly_b = _stack_bucket(sub)
            batch_out = estep_batch_mixfrag(
                x_b, y_b, t_b, Lx_b, Ly_b, lam, mu, exts, weights, Q, pi)
            agg = _aggregate_batch_suff_mf(batch_out)
            for kk in suff_mb:
                suff_mb[kk] = suff_mb[kk] + agg[kk]
            ll_mb += float(jnp.sum(batch_out['log_p']))

        if k == 0:
            suff_blend = {kk: scale * suff_mb[kk] for kk in suff_mb}
        else:
            suff_blend = _scaled_add_mf(suff_blend, suff_mb, eta_k, scale)

        lam, mu = m_step_lam_mu(
            suff_blend, prior_alpha_lam=prior_alpha_lam,
            prior_alpha_mu=prior_alpha_mu, prior_beta=prior_beta)
        exts = m_step_ext_per_fragtype(
            suff_blend['F_f'], suff_blend['E_f'], ext_prior_alpha, ext_prior_beta)
        weights = m_step_weights(suff_blend['E_f'], weight_prior_alpha)

        history.append({
            'iter': k + 1, 'eta': eta_k, 'mb_log_p_mean': ll_mb / batch_size,
            'lam': lam, 'mu': mu, 'exts': np.asarray(exts).copy(),
            'weights': np.asarray(weights).copy(),
        })

        if (k + 1) % 10 == 0 or k == 0 or k == n_iter - 1:
            log_fn(f'  iter {k+1:>4}/{n_iter}: lam={lam:.5f} mu={mu:.5f} '
                   f'exts={np.array2string(np.asarray(exts), precision=3)} '
                   f'w={np.array2string(np.asarray(weights), precision=3)} '
                   f'mb_ll/pair={ll_mb/batch_size:.2f} ({time.time()-t0:.1f}s)')

        if val_chunks and val_every and ((k + 1) % val_every == 0
                                         or k == n_iter - 1):
            val_ll = 0.0
            for xs, ys, ts, Lx, Ly, n_real in val_chunks:
                log_ps = estep_batch_mixfrag_forward_only(
                    xs, ys, ts, Lx, Ly, lam, mu, exts, weights, Q, pi)
                val_ll += float(jnp.sum(log_ps[:n_real]))
            history[-1]['val_ll_total'] = val_ll
            history[-1]['val_ll_per_pair'] = val_ll / max(len(val_pairs), 1)
            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_params = (lam, mu, np.asarray(exts).copy(),
                               np.asarray(weights).copy())
                val_no_improve = 0
            else:
                val_no_improve += 1
                if patience and val_no_improve >= patience:
                    log_fn(f'  early stop at iter {k+1} '
                           f'(no val improvement x{val_no_improve}).')
                    break

    out = {'lam': lam, 'mu': mu, 'exts': np.asarray(exts), 'weights': np.asarray(weights),
           'history': history}
    if val_pairs:
        out['best_lam'], out['best_mu'], out['best_exts'], out['best_weights'] = best_params
        out['best_val_ll_total'] = best_val_ll
    return out


# =========================================================================
# MixFrag-S: per-fragtype substitution matrices.
#
# Same MixFrag pair HMM (S, M_1..M_F, I_1..I_F, D_1..D_F, E) and the same
# shared indel process, but fragtype f carries ITS OWN reversible
# substitution model (S_f, pi_f) -> Q_f = build_Q_from_S_pi(S_f, pi_f).
# Emissions come from ``pair_hmm_emissions_per_fragtype``; the indel-side
# statistics are the shared-matrix ones above; the substitution side adds
# per-fragtype match counts mc_f, inserted-residue counts ins_f and
# deleted-residue counts del_f, from which per-pair Holmes-Rubin dwell /
# transition statistics (W_f, U_f) at the pair's own t feed
# ``m_step_subst_option1`` with LG-centred Dirichlet priors.
#
# Everything below is additive: the shared-matrix API above is unchanged.
# =========================================================================

from ..core.ctmc import (
    holmes_rubin_weighted_stats, m_step_subst_option1, build_Q_from_S_pi,
)
from ..dp.hmm import pair_hmm_emissions_per_fragtype


def lg_prior_S_pi():
    """(S_lg, pi_lg) for the LG-centred priors: S_lg = Q_lg / pi_lg[None, :]
    off the diagonal, symmetrised (the ``train_pfam.py`` convention)."""
    from ..core.protein import rate_matrix_lg
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg = np.asarray(Q_lg, dtype=np.float64)
    pi_lg = np.asarray(pi_lg, dtype=np.float64)
    A = pi_lg.shape[0]
    S_lg = Q_lg / np.maximum(pi_lg[None, :], 1e-30) * (1.0 - np.eye(A))
    S_lg = (S_lg + S_lg.T) / 2.0
    S_lg = np.maximum(S_lg, 1e-30) * (1.0 - np.eye(A))
    return S_lg, pi_lg


def _mixfrag_s_residue_stats(post, x_pad, y_pad, Lx_real, Ly_real, F, A):
    """Per-fragtype residue statistics from the FB posteriors.

    post: (Lx_pad+1, Ly_pad+1, 3F+2) posterior state probabilities on the
    1-indexed grid (grid row i <-> x[i-1], column j <-> y[j-1]).
    Returns (mc (F,A,A), ins (F,A), dele (F,A)):
        mc[f,a,b]  = sum_{i,j} post[i,j,M_f] 1[x_i=a] 1[y_j=b]
        ins[f,b]   = sum_{i,j} post[i,j,I_f] 1[y_j=b]
        dele[f,a]  = sum_{i,j} post[i,j,D_f] 1[x_i=a]
    Residues with index >= A (wildcards) are routed to a dropped one-hot
    column; grid positions beyond the real lengths are masked (the
    posteriors are already 0 there, the mask is belt-and-braces).
    """
    _, M0, I0, D0, _ = mixfrag_pair_index(F)
    Lx_pad = x_pad.shape[0]
    Ly_pad = y_pad.shape[0]
    dt = post.dtype
    ox = jax.nn.one_hot(jnp.clip(x_pad, 0, A), A + 1, dtype=dt)     # (Lx_pad, A+1)
    oy = jax.nn.one_hot(jnp.clip(y_pad, 0, A), A + 1, dtype=dt)     # (Ly_pad, A+1)
    ox = jnp.concatenate([jnp.zeros((1, A + 1), dtype=dt), ox], axis=0)  # row 0: no residue
    oy = jnp.concatenate([jnp.zeros((1, A + 1), dtype=dt), oy], axis=0)
    ox = ox * (jnp.arange(Lx_pad + 1) <= Lx_real)[:, None]
    oy = oy * (jnp.arange(Ly_pad + 1) <= Ly_real)[:, None]
    post_M = post[:, :, M0:M0 + F]
    post_I = post[:, :, I0:I0 + F]
    post_D = post[:, :, D0:D0 + F]
    # (A+1, Ly_pad+1, F): contract the ancestor axis first.
    tmp = jnp.einsum('ia,ijf->ajf', ox, post_M)
    mc = jnp.einsum('ajf,jb->fab', tmp, oy)[:, :A, :A]
    ins = jnp.einsum('ijf,jb->fb', post_I, oy)[:, :A]
    dele = jnp.einsum('ijf,ia->fa', post_D, ox)[:, :A]
    return mc, ins, dele


def _mixfrag_s_log_emit_table(state_types, x_pad, y_pad, t, Qs, pis):
    """(Lx_pad+1, Ly_pad+1, 3F+2) log-emission table at the pair's own t:
    P_f(t) = expm(Q_f t) for each fragtype, then the per-fragtype emissions."""
    subs = jax.vmap(transition_matrix, in_axes=(0, None))(Qs, t)   # (F, A, A)
    return pair_hmm_emissions_per_fragtype(state_types, x_pad, y_pad, subs, pis)


def _estep_pair_mixfrag_s_core(x_pad, y_pad, t, Lx_real, Ly_real,
                               lam, mu, exts, weights, Qs, pis):
    """Per-pair MixFrag-S E-step core.

    Qs: (F, A, A) per-fragtype rate matrices; pis: (F, A).  Returns the
    shared-matrix outputs (log_p, resolved5, F_f, E_f, L, M) plus the
    per-fragtype residue statistics 'mc' (F,A,A), 'ins' (F,A), 'del' (F,A).
    Padding semantics match ``_estep_pair_mixfrag_core``.
    """
    tau = mixfrag_trans(lam, mu, t, exts, weights)
    log_trans = jnp.log(jnp.maximum(tau, 1e-30))
    F = int(exts.shape[0])
    A = int(pis.shape[-1])
    state_types = jnp.array(
        [TYPE_S] + [TYPE_M] * F + [TYPE_I] * F + [TYPE_D] * F + [TYPE_E],
        dtype=jnp.int32)
    table = _mixfrag_s_log_emit_table(state_types, x_pad, y_pad, t, Qs, pis)
    log_p, post, n_chi = forward_backward_2d(
        log_trans, state_types, x_pad, y_pad, None, None,
        log_emit_table=table, real_Lx=Lx_real, real_Ly=Ly_real)
    resolved5, F_f, E_f, L_coef, M_coef = _mixfrag_chi_resolve_core(
        n_chi, lam, mu, t, exts, weights)
    mc, ins, dele = _mixfrag_s_residue_stats(
        post, x_pad, y_pad, Lx_real, Ly_real, F, A)
    # Holmes-Rubin dwell / transition statistics at THIS pair's t, per
    # fragtype (vmapped over f): W (F,A), U (F,A,A).  Done inside the jitted
    # core so a chunk costs one dispatch and mc never round-trips the host.
    W, U = jax.vmap(holmes_rubin_weighted_stats,
                    in_axes=(0, 0, None, 0))(Qs, pis, t, mc)
    return {
        'log_p': log_p,
        'resolved5': resolved5,
        'F_f': F_f,
        'E_f': E_f,
        'L': L_coef,
        'M': M_coef,
        'mc': mc,
        'ins': ins,
        'del': dele,
        'W': W,
        'U': U,
    }


def _estep_pair_mixfrag_s_forward_only_core(x_pad, y_pad, t, Lx_real, Ly_real,
                                            lam, mu, exts, weights, Qs, pis):
    """Forward-only MixFrag-S variant (log_p only)."""
    tau = mixfrag_trans(lam, mu, t, exts, weights)
    log_trans = jnp.log(jnp.maximum(tau, 1e-30))
    F = int(exts.shape[0])
    state_types = jnp.array(
        [TYPE_S] + [TYPE_M] * F + [TYPE_I] * F + [TYPE_D] * F + [TYPE_E],
        dtype=jnp.int32)
    table = _mixfrag_s_log_emit_table(state_types, x_pad, y_pad, t, Qs, pis)
    return forward_backward_2d(
        log_trans, state_types, x_pad, y_pad, None, None,
        log_emit_table=table, real_Lx=Lx_real, real_Ly=Ly_real,
        forward_only=True)


_ESTEP_MFS_FULL_JIT = None
_ESTEP_MFS_FWD_JIT = None
_HR_PAIR_FRAG_JIT = None


def _get_estep_mfs_full_jit():
    global _ESTEP_MFS_FULL_JIT
    if _ESTEP_MFS_FULL_JIT is None:
        vmapped = jax.vmap(
            _estep_pair_mixfrag_s_core,
            in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None))
        _ESTEP_MFS_FULL_JIT = jax.jit(vmapped)
    return _ESTEP_MFS_FULL_JIT


def _get_estep_mfs_fwd_jit():
    global _ESTEP_MFS_FWD_JIT
    if _ESTEP_MFS_FWD_JIT is None:
        vmapped = jax.vmap(
            _estep_pair_mixfrag_s_forward_only_core,
            in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None))
        _ESTEP_MFS_FWD_JIT = jax.jit(vmapped)
    return _ESTEP_MFS_FWD_JIT


def _get_hr_pair_frag_jit():
    """vmap ``holmes_rubin_weighted_stats`` over (pairs, fragtypes): inner
    over fragtypes (Q_f, pi_f, mc_f; shared t), outer over pairs (own t)."""
    global _HR_PAIR_FRAG_JIT
    if _HR_PAIR_FRAG_JIT is None:
        per_frag = jax.vmap(holmes_rubin_weighted_stats,
                            in_axes=(0, 0, None, 0))
        per_pair = jax.vmap(per_frag, in_axes=(None, None, 0, 0))
        _HR_PAIR_FRAG_JIT = jax.jit(per_pair)
    return _HR_PAIR_FRAG_JIT


def _as_mfs_params(lam, mu, exts, weights, Qs, pis):
    return (jnp.asarray(float(lam)), jnp.asarray(float(mu)),
            jnp.asarray(exts, dtype=jnp.float64),
            jnp.asarray(weights, dtype=jnp.float64),
            jnp.asarray(Qs, dtype=jnp.float64),
            jnp.asarray(pis, dtype=jnp.float64))


def estep_batch_mixfrag_s(xs_pad, ys_pad, ts, Lx_real, Ly_real,
                          lam, mu, exts, weights, Qs, pis):
    """Batched (vmap'd) MixFrag-S E-step on a bin-bucketed minibatch.

    Qs: (F, A, A) per-fragtype rate matrices, pis: (F, A).

    Returns dict of jnp arrays: the ``estep_batch_mixfrag`` entries
    ('log_p', 'B','D','S', 'L','M','T' (B,), 'F_f','E_f' (B,F)) plus
    'mc' (B,F,A,A), 'ins' (B,F,A), 'del' (B,F,A) and the per-pair
    Holmes-Rubin statistics at each pair's own t: 'W' (B,F,A), 'U' (B,F,A,A).
    """
    xs_pad = jnp.asarray(xs_pad)
    ys_pad = jnp.asarray(ys_pad)
    ts = jnp.asarray(ts, dtype=jnp.float64)
    Lx_real = jnp.asarray(Lx_real, dtype=jnp.int32)
    Ly_real = jnp.asarray(Ly_real, dtype=jnp.int32)
    lam_j, mu_j, exts_j, weights_j, Qs_j, pis_j = _as_mfs_params(
        lam, mu, exts, weights, Qs, pis)

    fn = _get_estep_mfs_full_jit()
    out = fn(xs_pad, ys_pad, ts, Lx_real, Ly_real,
             lam_j, mu_j, exts_j, weights_j, Qs_j, pis_j)

    T_batch = ts
    E_B, E_D, E_S = _bdi_stats_batch(out['resolved5'], lam, mu, ts, T_batch)
    return {
        'log_p': out['log_p'],
        'B': jnp.maximum(E_B, 0.0), 'D': jnp.maximum(E_D, 0.0),
        'S': jnp.maximum(E_S, 0.0),
        'L': out['L'], 'M': out['M'], 'T': T_batch,
        'F_f': out['F_f'], 'E_f': out['E_f'],
        'mc': out['mc'], 'ins': out['ins'], 'del': out['del'],
        'W': out['W'], 'U': out['U'],
    }


def estep_batch_mixfrag_s_forward_only(xs_pad, ys_pad, ts, Lx_real, Ly_real,
                                       lam, mu, exts, weights, Qs, pis):
    """Forward-only batched MixFrag-S E-step (per-pair log_p only)."""
    fn = _get_estep_mfs_fwd_jit()
    return fn(jnp.asarray(xs_pad), jnp.asarray(ys_pad),
              jnp.asarray(ts, dtype=jnp.float64),
              jnp.asarray(Lx_real, dtype=jnp.int32),
              jnp.asarray(Ly_real, dtype=jnp.int32),
              *_as_mfs_params(lam, mu, exts, weights, Qs, pis))


def hr_stats_per_pair_fragtype(Qs, pis, ts, mc_stack, chunk_size=128):
    """Per-pair, per-fragtype Holmes-Rubin weighted statistics at each pair's
    own t.  Qs (F,A,A), pis (F,A), ts (B,), mc_stack (B,F,A,A).
    Returns (W (B,F,A), U (B,F,A,A)) as numpy.  Chunked over B purely for
    memory (the HR integral tensor is (chunk, F, A, A, A, A) float64)."""
    fn = _get_hr_pair_frag_jit()
    Qs_j = jnp.asarray(Qs, dtype=jnp.float64)
    pis_j = jnp.asarray(pis, dtype=jnp.float64)
    ts = np.asarray(ts, dtype=np.float64)
    mc_stack = np.asarray(mc_stack, dtype=np.float64)
    B = int(ts.shape[0])
    F, A = int(Qs_j.shape[0]), int(Qs_j.shape[-1])
    if B == 0:
        return np.zeros((0, F, A)), np.zeros((0, F, A, A))
    W_chunks, U_chunks = [], []
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        W_c, U_c = fn(Qs_j, pis_j, jnp.asarray(ts[s:e]),
                      jnp.asarray(mc_stack[s:e]))
        W_chunks.append(np.asarray(W_c))
        U_chunks.append(np.asarray(U_c))
    return np.concatenate(W_chunks, axis=0), np.concatenate(U_chunks, axis=0)


# -------------------------------------------------------------------------
# MixFrag-S M-step and accumulator.
# -------------------------------------------------------------------------


def m_step_subst_per_fragtype(W, U, V, S_prior=None, pi_prior=None,
                              pi_pseudo=10.0, S_pseudo=5.0):
    """Per-fragtype exact substitution M-step: for each f,
    (S_f, pi_f, Q_f) = m_step_subst_option1(W[f], U[f], V[f], ...) with the
    (LG-centred) Dirichlet pseudocount priors.  Returns
    (S (F,A,A), pi (F,A), Q (F,A,A)) in raw GTR units (not unit-normalised)."""
    W = np.asarray(W, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    F, A = V.shape
    S_out = np.zeros((F, A, A))
    pi_out = np.zeros((F, A))
    Q_out = np.zeros((F, A, A))
    for f in range(F):
        S_f, pi_f, Q_f = m_step_subst_option1(
            W[f], U[f], V[f], S_prior=S_prior, pi_prior=pi_prior,
            pi_pseudo=pi_pseudo, S_pseudo=S_pseudo)
        S_out[f], pi_out[f], Q_out[f] = S_f, pi_f, Q_f
    return S_out, pi_out, Q_out


def _empty_suff_mfs(F, A):
    d = _empty_suff_mf(F)
    d.update({'W': np.zeros((F, A)), 'U': np.zeros((F, A, A)),
              'V': np.zeros((F, A))})
    return d


def _aggregate_chunk_suff_mfs(bd, n_real, W_stack, U_stack):
    """Sum a chunk's per-pair statistics (first ``n_real`` entries) into a
    suff dict; W_stack/U_stack are the per-pair HR stats for the chunk."""
    sl = slice(0, n_real)
    mc = np.asarray(bd['mc'][sl])
    ins = np.asarray(bd['ins'][sl])
    dele = np.asarray(bd['del'][sl])
    anc = mc.sum(axis=-1)                                  # (n, F, A)
    return {
        'B': float(jnp.sum(bd['B'][sl])), 'D': float(jnp.sum(bd['D'][sl])),
        'S': float(jnp.sum(bd['S'][sl])), 'L': float(jnp.sum(bd['L'][sl])),
        'M': float(jnp.sum(bd['M'][sl])), 'T': float(jnp.sum(bd['T'][sl])),
        'F_f': np.asarray(jnp.sum(bd['F_f'][sl], axis=0)),
        'E_f': np.asarray(jnp.sum(bd['E_f'][sl], axis=0)),
        'W': W_stack[sl].sum(axis=0),
        'U': U_stack[sl].sum(axis=0),
        'V': (anc + ins + dele).sum(axis=0),
    }


# -------------------------------------------------------------------------
# Lazy bucketing (unpadded storage; padding at stack time) and chunk sizing.
# -------------------------------------------------------------------------


def _coarse_pad_to_bin(n, stride):
    """Pad n to the next member of every ``stride``-th geometric bin (always
    itself a member of the fine bin list, so the FB's own ``_pad_to_bin`` of
    the padded length is the identity and no second padding happens)."""
    from ..dp.hmm import _pad_to_bin, _GEOM_BINS
    n_fine = _pad_to_bin(n)
    if stride <= 1:
        return n_fine
    coarse = [b for i, b in enumerate(_GEOM_BINS) if b >= 4 and i % stride == 0]
    for b in coarse:
        if b >= n_fine:
            return b
    return n_fine


def bucket_pairs_lazy(pairs, bin_stride=1):
    """Group (x, y, t) pairs by (Lx_pad, Ly_pad) bin key WITHOUT padding the
    stored arrays (the full corpus is ~1M pairs; padding up front would
    multiply its footprint).  Returns {(Lx_pad, Ly_pad): [(x, y, t), ...]}.

    bin_stride=1 uses the FB's geometric bins (ratio 1.2) exactly as the
    shared-matrix loop does; bin_stride=k keeps every k-th bin, trading a
    little extra padding for ~k^2 fewer (Lx, Ly) buckets, i.e. far fewer
    distinct JIT shapes to compile on the first pass over a large corpus."""
    buckets = {}
    for x, y, t in pairs:
        key = (_coarse_pad_to_bin(int(x.shape[0]), bin_stride),
               _coarse_pad_to_bin(int(y.shape[0]), bin_stride))
        buckets.setdefault(key, []).append((x, y, float(t)))
    return buckets


def stack_bucket_lazy(items, key, n_pad_to=None):
    """Pad+stack [(x, y, t), ...] of one bucket to (xs, ys, ts, Lx, Ly).
    If ``n_pad_to`` > len(items), the last item is repeated so that the
    batch dimension hits a cached JIT shape (callers slice [:n_real])."""
    Lx_pad, Ly_pad = key
    n = len(items)
    n_out = n if n_pad_to is None else max(n, int(n_pad_to))
    xs = np.zeros((n_out, Lx_pad), dtype=np.int32)
    ys = np.zeros((n_out, Ly_pad), dtype=np.int32)
    ts = np.zeros(n_out, dtype=np.float64)
    Lx = np.zeros(n_out, dtype=np.int32)
    Ly = np.zeros(n_out, dtype=np.int32)
    for i in range(n_out):
        x, y, t = items[min(i, n - 1)]
        xs[i, :x.shape[0]] = x
        ys[i, :y.shape[0]] = y
        ts[i] = t
        Lx[i] = x.shape[0]
        Ly[i] = y.shape[0]
    return xs, ys, ts, Lx, Ly


def chunk_size_for_bucket(key, n_states, chunk_max=256, mem_bytes=2.0e9):
    """Pairs per vmapped E-step call for a (Lx_pad, Ly_pad) bucket.  The 2D
    FB materialises several (Lx_pad, Ly_pad, ns, ns) float64 tensors for the
    expected transition counts; ~8 such tensors per pair is the measured
    envelope, so the chunk is mem_bytes / (8 * Lx * Ly * ns^2 * 8B), capped
    at chunk_max and floored at 1."""
    Lx_pad, Ly_pad = key
    per_pair = 8.0 * (Lx_pad + 1) * (Ly_pad + 1) * n_states * n_states * 8.0
    return int(max(1, min(chunk_max, mem_bytes // per_pair)))


def _round_up_pow2(n, cap):
    """Pad a chunk's batch size up to a power of two (<= cap) so each bucket
    compiles at most log2(cap)+1 shapes."""
    p = 1
    while p < n:
        p *= 2
    return min(p, cap)


def estep_bucket_chunks_mixfrag_s(items, key, lam, mu, exts, weights, Qs, pis,
                                  *, chunk_max=256, mem_bytes=2.0e9,
                                  hr_chunk=128, with_hr=True):
    """Run the MixFrag-S E-step over one bucket's items in memory-sized
    chunks; returns (suff dict summed over items, total log_p, n_items).
    The HR (W_f, U_f) come out of the jitted E-step at each pair's own t
    (``hr_chunk`` is kept for API compatibility; ``with_hr=False`` zeroes
    them)."""
    F, A = int(np.asarray(pis).shape[0]), int(np.asarray(pis).shape[1])
    n_states = 3 * F + 2
    chunk = chunk_size_for_bucket(key, n_states, chunk_max, mem_bytes)
    suff = _empty_suff_mfs(F, A)
    ll = 0.0
    for s in range(0, len(items), chunk):
        sub = items[s:s + chunk]
        n_real = len(sub)
        n_pad = _round_up_pow2(n_real, chunk)
        xs, ys, ts, Lx, Ly = stack_bucket_lazy(sub, key, n_pad_to=n_pad)
        bd = estep_batch_mixfrag_s(xs, ys, ts, Lx, Ly,
                                   lam, mu, exts, weights, Qs, pis)
        if with_hr:
            W_st, U_st = np.asarray(bd['W']), np.asarray(bd['U'])
        else:
            W_st = np.zeros((n_real, F, A))
            U_st = np.zeros((n_real, F, A, A))
        agg = _aggregate_chunk_suff_mfs(bd, n_real, W_st, U_st)
        for k in suff:
            suff[k] = suff[k] + agg[k]
        ll += float(jnp.sum(bd['log_p'][:n_real]))
    return suff, ll, len(items)


def loglik_bucketed_mixfrag_s(buckets, lam, mu, exts, weights, Qs, pis,
                              *, chunk_max=256, mem_bytes=2.0e9):
    """Total forward-only log-likelihood over bucketed pairs."""
    F = int(np.asarray(pis).shape[0])
    n_states = 3 * F + 2
    total = 0.0
    n = 0
    for key, items in buckets.items():
        chunk = chunk_size_for_bucket(key, n_states, chunk_max,
                                      mem_bytes * 4.0)  # forward-only: no (ns,ns) tensors
        for s in range(0, len(items), chunk):
            sub = items[s:s + chunk]
            n_real = len(sub)
            xs, ys, ts, Lx, Ly = stack_bucket_lazy(
                sub, key, n_pad_to=_round_up_pow2(n_real, chunk))
            lp = estep_batch_mixfrag_s_forward_only(
                xs, ys, ts, Lx, Ly, lam, mu, exts, weights, Qs, pis)
            total += float(jnp.sum(lp[:n_real]))
            n += n_real
    return total, n


# -------------------------------------------------------------------------
# MixFrag-S training loop.
# -------------------------------------------------------------------------


def _fmt_vec(v, prec=3):
    return np.array2string(np.asarray(v, dtype=np.float64), precision=prec,
                           separator=' ', max_line_width=10 ** 6)


def svi_bw_mixfrag_s(pairs, *, n_total_pairs, n_fragtypes,
                     init_lam, init_mu, init_exts, init_weights,
                     init_S, init_pi,
                     n_iter=1620, batch_size=2000,
                     svi_tau=1.0, svi_kappa=0.7,
                     prior_alpha_lam=2.0, prior_alpha_mu=2.0, prior_beta=10.0,
                     ext_prior_alpha=2.0, ext_prior_beta=3.0,
                     weight_prior_alpha=1.5,
                     S_prior=None, pi_prior=None, pi_pseudo=10.0, S_pseudo=5.0,
                     full_batch=False,
                     start_iter=0, suff_blend_init=None,
                     chunk_max=256, mem_bytes=2.0e9, hr_chunk=128,
                     bin_stride=1,
                     dp_mode='path', fb_scan_mode='associative',
                     checkpoint_fn=None, ckpt_every=0,
                     val_pairs=None, val_every=0,
                     log_fn=print, log_every=1, seed=42):
    """SVI Baum-Welch for MixFrag-S (per-fragtype substitution matrices).

    pairs: list of pairs, each carrying its own t, in any of the formats
    ``as_mode_pairs`` accepts: the decoded precompiled tuple
    (x_int, y_int, states, anc_chars, desc_chars, t), the compact 'path'
    tuple (states, anc_full, desc_full, t), (x, y, t, mi, mj), or the
    alignment-free (x_int, y_int, t) (that one only serves
    ``dp_mode='unaligned'``).
    dp_mode: E-step mode — 'path' (default; 1D alignment-conditioned
    forward-backward along the training alignment, the paper's svi-BW
    convention), 'match_aligned' (2D, matches pinned to the alignment,
    indel orderings summed) or 'unaligned' (fully unaligned 2D).
    fb_scan_mode: 'associative' | 'sequential' for the 1D path FB.
    init_S (F,A,A), init_pi (F,A): per-fragtype exchangeabilities /
    equilibria; Q_f is rebuilt as build_Q_from_S_pi(S_f, pi_f).
    S_prior / pi_prior: default to the LG-centred prior (``lg_prior_S_pi``).

    Per iteration k (0-based, global; ``start_iter`` on resume):
      * minibatch of ``batch_size`` pairs drawn without replacement with
        rng = default_rng([seed, k]) (so a resumed run re-draws the same
        minibatch sequence as an uninterrupted one);
      * E-step on the minibatch, bucketed by padded (Lx, Ly) bin and
        processed in memory-sized chunks; HR stats at each pair's own t;
      * suff = (1-eta_k) suff + eta_k (N_total/batch) suff_mb with
        eta_k = (svi_tau + k)^(-svi_kappa) (k == 0 or a resume without
        saved suff stats initialises suff = (N_total/batch) suff_mb);
      * M-steps for (lam, mu), exts, weights and the F matrices on the
        blended stats.
    ``full_batch=True``: minibatch = all pairs, eta = 1, scale = 1 (plain
    EM; the logged minibatch log-likelihood is then the total LL at the
    parameters BEFORE that iteration's M-step).

    bin_stride: see ``bucket_pairs_lazy``.

    checkpoint_fn(state) is called every ``ckpt_every`` iterations and at
    the end with state = {'lam','mu','exts','weights','S','pi','Q',
    'em_iter','suff_blend','history'}.

    Returns dict: final 'lam','mu','exts','weights','S','pi','Q',
    'em_iter', 'suff_blend', 'history'.
    """
    F = int(n_fragtypes)
    init_S = np.asarray(init_S, dtype=np.float64)
    init_pi = np.asarray(init_pi, dtype=np.float64)
    A = int(init_pi.shape[-1])
    assert init_S.shape == (F, A, A) and init_pi.shape == (F, A)
    if S_prior is None or pi_prior is None:
        S_lg, pi_lg = lg_prior_S_pi()
        S_prior = S_lg if S_prior is None else S_prior
        pi_prior = pi_lg if pi_prior is None else pi_prior

    lam, mu = float(init_lam), float(init_mu)
    exts = np.asarray(init_exts, dtype=np.float64).copy()
    weights = np.asarray(init_weights, dtype=np.float64).copy()
    weights = weights / weights.sum()
    assert exts.shape == (F,) and weights.shape == (F,)
    S_cur = init_S.copy()
    pi_cur = init_pi.copy()
    Q_cur = np.stack([np.asarray(build_Q_from_S_pi(jnp.asarray(S_cur[f]),
                                                   jnp.asarray(pi_cur[f])))
                      for f in range(F)])

    pairs = as_mode_pairs(list(pairs), dp_mode)
    if not pairs:
        raise ValueError('No pairs.')
    n_actual = len(pairs)
    if full_batch:
        batch_size = n_actual
    bucket_fn, estep_fn, loglik_fn = mixfrag_s_mode_fns(
        dp_mode, chunk_max=chunk_max, mem_bytes=mem_bytes,
        bin_stride=bin_stride, scan_mode=fb_scan_mode)
    buckets = bucket_fn(pairs)
    bucket_keys = list(buckets.keys())
    flat = [(key, j) for key in bucket_keys for j in range(len(buckets[key]))]
    n_flat = len(flat)
    scale = 1.0 if full_batch else float(n_total_pairs) / float(batch_size)
    log_fn(f'svi_bw_mixfrag_s: dp_mode={dp_mode}; {n_actual} pairs in '
           f'{len(bucket_keys)} '
           f'{"L_pad" if dp_mode == "path" else "(Lx_pad, Ly_pad)"} buckets; '
           f'F={F}; A={A}; '
           f'n_total_pairs={n_total_pairs}, batch_size={batch_size}, '
           f'scale={scale:.3f}, n_iter={n_iter}, start_iter={start_iter}, '
           f'full_batch={full_batch}.')

    val_buckets = (bucket_fn(as_mode_pairs(list(val_pairs), dp_mode))
                   if val_pairs else None)

    suff_blend = None
    if suff_blend_init is not None:
        suff_blend = {k: (np.asarray(v, dtype=np.float64).copy()
                          if np.ndim(v) else float(v))
                      for k, v in suff_blend_init.items()}
        log_fn('  resuming with saved blended sufficient statistics.')
    history = []

    def _state(k_done):
        return {'lam': lam, 'mu': mu, 'exts': exts.copy(),
                'weights': weights.copy(), 'S': S_cur.copy(),
                'pi': pi_cur.copy(), 'Q': Q_cur.copy(), 'em_iter': int(k_done),
                'suff_blend': suff_blend, 'history': history}

    t0 = time.time()
    k = int(start_iter)
    while k < n_iter:
        eta_k = 1.0 if full_batch else (svi_tau + k) ** (-svi_kappa)
        if full_batch:
            sub_buckets = buckets
        else:
            rng = np.random.default_rng([int(seed), int(k)])
            ix = rng.choice(n_flat, batch_size, replace=n_flat < batch_size)
            sub_buckets = {}
            for i in ix:
                key, j = flat[i]
                sub_buckets.setdefault(key, []).append(buckets[key][j])

        t_e = time.time()
        suff_mb = _empty_suff_mfs(F, A)
        ll_mb = 0.0
        n_mb = 0
        for key, items in sub_buckets.items():
            agg, ll, n = estep_fn(
                items, key, lam, mu, exts, weights, Q_cur, pi_cur)
            for kk in suff_mb:
                suff_mb[kk] = suff_mb[kk] + agg[kk]
            ll_mb += ll
            n_mb += n
        t_e = time.time() - t_e

        if suff_blend is None or full_batch:
            suff_blend = {kk: scale * suff_mb[kk] for kk in suff_mb}
        else:
            suff_blend = _scaled_add_mf(suff_blend, suff_mb, eta_k, scale)

        lam, mu = m_step_lam_mu(
            suff_blend, prior_alpha_lam=prior_alpha_lam,
            prior_alpha_mu=prior_alpha_mu, prior_beta=prior_beta)
        exts = np.asarray(m_step_ext_per_fragtype(
            suff_blend['F_f'], suff_blend['E_f'], ext_prior_alpha,
            ext_prior_beta), dtype=np.float64)
        weights = np.asarray(m_step_weights(suff_blend['E_f'],
                                            weight_prior_alpha),
                             dtype=np.float64)
        S_cur, pi_cur, Q_cur = m_step_subst_per_fragtype(
            suff_blend['W'], suff_blend['U'], suff_blend['V'],
            S_prior=S_prior, pi_prior=pi_prior,
            pi_pseudo=pi_pseudo, S_pseudo=S_pseudo)
        k += 1

        rec = {'iter': k, 'eta': eta_k, 'mb_log_p_total': ll_mb,
               'mb_n_pairs': n_mb, 'mb_log_p_mean': ll_mb / max(n_mb, 1),
               'lam': lam, 'mu': mu, 'exts': exts.copy(),
               'weights': weights.copy(), 'pi': pi_cur.copy(),
               'estep_sec': t_e}
        history.append(rec)
        if log_every and (k % log_every == 0 or k == n_iter):
            log_fn(f'  iter {k:>4}/{n_iter}: eta={eta_k:.5f} lam={lam:.6f} '
                   f'mu={mu:.6f} exts={_fmt_vec(exts, 4)} '
                   f'w={_fmt_vec(weights, 4)} '
                   f'mb_ll/pair={ll_mb / max(n_mb, 1):.3f} '
                   f'(mb_ll={ll_mb:.2f}, n={n_mb}; estep {t_e:.1f}s; '
                   f'elapsed {time.time() - t0:.0f}s)')
            for f in range(F):
                log_fn(f'      pi[{f}]={_fmt_vec(pi_cur[f], 4)}')

        if val_buckets is not None and val_every and (k % val_every == 0
                                                      or k == n_iter):
            val_ll, n_val = loglik_fn(
                val_buckets, lam, mu, exts, weights, Q_cur, pi_cur)
            rec['val_ll_total'] = val_ll
            rec['val_ll_per_pair'] = val_ll / max(n_val, 1)
            log_fn(f'      val: ll={val_ll:.2f} ll/pair={val_ll / max(n_val, 1):.3f} '
                   f'(n={n_val})')

        if checkpoint_fn is not None and ckpt_every and k % ckpt_every == 0 \
                and k < n_iter:
            checkpoint_fn(_state(k))

    out = _state(k)
    if checkpoint_fn is not None:
        checkpoint_fn(out)
    return out


# =========================================================================
# MixFrag-S E-step modes.
#
#   'path'          1D alignment-conditioned forward-backward along each
#                   training alignment's column sequence (the paper's svi-BW
#                   convention, cf. train_pfam._process_pairs_batched): at an
#                   M column only the M_f states may fire (pi_f[a] P_f(t)[a,b]),
#                   at an I column only I_f (pi_f[b]), at a D column only D_f
#                   (pi_f[a]); transitions mixfrag_trans(lam, mu, t, exts, w);
#                   S at the start, E at the end.  Fragtype assignments are
#                   summed; the indel ordering is fixed by the alignment.
#                   O(L_cols * ns^2) per pair, no (Lx, Ly) tables.
#   'match_aligned' 2D forward-backward with the matches pinned to the
#                   alignment's (i, j) pairs (mask_emissions_match_pinned:
#                   every path is forced through each match cell by its M
#                   step, so E[#matches] = n_match exactly; stricter than
#                   the held-out scorer's mask_emissions_match_aligned, which
#                   lets an insert + delete bypass a match) and the indel
#                   orderings between matches summed.
#   'unaligned'     the fully unaligned 2D forward-backward (round 1, above).
#
# All three produce exactly the sufficient statistics of
# ``estep_batch_mixfrag_s``: log_p, B/D/S/L/M/T, F_f, E_f, mc, ins, del, W, U.
#
# Pair formats (``as_mode_pairs`` converts between them):
#   decoded     (x_int, y_int, states, anc_chars, desc_chars, t)   — the
#               train_pfam.PrecompiledPairSource tuple;
#   'path'      (states, anc_full, desc_full, t): per-column state type
#               (M=1, I=2, D=3), ancestor residue at M/D columns, descendant
#               residue at M/I columns (0 elsewhere); compact int8 arrays;
#   'match_aligned'  (x_int, y_int, t, mi, mj): 0-indexed match positions;
#   'unaligned' (x_int, y_int, t).
# =========================================================================

from ..dp.hmm import (
    forward_backward_1d_padded, mask_emissions_match_pinned, NEG_INF,
)

DP_MODES = ('path', 'match_aligned', 'unaligned')


def _mixfrag_s_frag_index(F):
    """numpy (3F+2,) int32: fragtype of each state (0 for S and E)."""
    return np.array([0] + list(range(F)) * 3 + [0], dtype=np.int32)


# -------------------------------------------------------------------------
# Pair-format helpers.
# -------------------------------------------------------------------------


def path_pair_from_decoded(x_int, y_int, states, anc_chars, desc_chars, t,
                           dtype=np.int8):
    """(states, anc_full, desc_full, t) from a decoded precompiled tuple.
    ``anc_full[l]`` is the ancestor residue at M/D columns (0 at I columns);
    ``desc_full[l]`` the descendant residue at M/I columns (0 at D columns).
    Stored as compact ``dtype`` (int8 holds residues 0..19 + wildcard 20)."""
    states_arr = np.asarray(states, dtype=np.int8)
    L = int(states_arr.shape[0])
    anc_full = np.zeros(L, dtype=dtype)
    desc_full = np.zeros(L, dtype=dtype)
    m_or_d = states_arr != TYPE_I
    m_or_i = states_arr != TYPE_D
    anc_full[m_or_d] = np.asarray(anc_chars, dtype=dtype)
    desc_full[m_or_i] = np.asarray(desc_chars, dtype=dtype)
    return (states_arr, anc_full, desc_full, float(t))


def xy_from_path_pair(states, anc_full, desc_full):
    """(x_int, y_int) int32 from a 'path' pair's column arrays."""
    states = np.asarray(states)
    x = np.asarray(anc_full)[states != TYPE_I].astype(np.int32)
    y = np.asarray(desc_full)[states != TYPE_D].astype(np.int32)
    return x, y


def match_positions_from_states(states):
    """0-indexed (mi, mj) match positions of a column state sequence."""
    states = np.asarray(states)
    i_pos = np.cumsum(states != TYPE_I) - 1      # ancestor index at M/D columns
    j_pos = np.cumsum(states != TYPE_D) - 1      # descendant index at M/I columns
    m = states == TYPE_M
    return i_pos[m].astype(np.int32), j_pos[m].astype(np.int32)


def _pair_kind(p):
    """Classify a pair tuple: 'decoded' (6), 'path' (4), 'match_aligned' (5),
    'unaligned' (3)."""
    n = len(p)
    if n == 6:
        return 'decoded'
    if n == 4:
        return 'path'
    if n == 5:
        return 'match_aligned'
    if n == 3:
        return 'unaligned'
    raise ValueError(f'unrecognised pair tuple of length {n}')


def as_mode_pairs(pairs, dp_mode):
    """Convert a list of pairs (any of the formats above) to ``dp_mode``'s
    format.  Alignment-free (x, y, t) pairs can only serve 'unaligned'."""
    if dp_mode not in DP_MODES:
        raise ValueError(f'dp_mode must be one of {DP_MODES}, got {dp_mode!r}')
    out = []
    for p in pairs:
        kind = _pair_kind(p)
        if kind == dp_mode:
            out.append(p)
            continue
        if kind == 'decoded':
            x, y, states, anc, desc, t = p
            if dp_mode == 'path':
                out.append(path_pair_from_decoded(x, y, states, anc, desc, t))
            elif dp_mode == 'match_aligned':
                mi, mj = match_positions_from_states(states)
                out.append((np.asarray(x, dtype=np.int32),
                            np.asarray(y, dtype=np.int32), float(t), mi, mj))
            else:
                out.append((np.asarray(x, dtype=np.int32),
                            np.asarray(y, dtype=np.int32), float(t)))
        elif kind == 'path':
            states, anc_full, desc_full, t = p
            x, y = xy_from_path_pair(states, anc_full, desc_full)
            if dp_mode == 'match_aligned':
                mi, mj = match_positions_from_states(states)
                out.append((x, y, float(t), mi, mj))
            else:
                out.append((x, y, float(t)))
        elif kind == 'match_aligned':
            x, y, t, mi, mj = p
            if dp_mode == 'unaligned':
                out.append((np.asarray(x, dtype=np.int32),
                            np.asarray(y, dtype=np.int32), float(t)))
            else:
                raise ValueError("dp_mode 'path' needs the alignment columns; "
                                 "got (x, y, t, mi, mj) pairs")
        else:
            raise ValueError(f"dp_mode {dp_mode!r} needs the training alignment; "
                             "got alignment-free (x, y, t) pairs")
    return out


# -------------------------------------------------------------------------
# 'path': 1D alignment-conditioned E-step.
# -------------------------------------------------------------------------


def _mixfrag_s_log_emit_columns(states_pad, anc_pad, desc_pad, L_real, t,
                                Qs, pis):
    """(L_pad, 3F+2) column log-emission table under the per-fragtype models.

    M columns: M_f -> log pi_f[a] + log P_f(t)[a, b];  I columns: I_f ->
    log pi_f[b];  D columns: D_f -> log pi_f[a]; every other (column, state)
    entry and every column >= L_real is NEG_INF.  Residue index >= A
    (wildcard) emits with probability 1 (log 0), matching
    ``pair_hmm_emissions_per_fragtype``.
    """
    F = int(Qs.shape[0])
    A = int(pis.shape[-1])
    L_pad = int(states_pad.shape[0])
    subs = jax.vmap(transition_matrix, in_axes=(0, None))(Qs, t)       # (F, A, A)
    log_subs = jnp.log(jnp.maximum(subs, 1e-30))
    log_pis = jnp.log(jnp.maximum(pis, 1e-30))
    log_subs = jnp.pad(log_subs, ((0, 0), (0, 1), (0, 1)), constant_values=0.0)
    log_pis = jnp.pad(log_pis, ((0, 0), (0, 1)), constant_values=0.0)
    frag_idx = jnp.asarray(_mixfrag_s_frag_index(F))                      # (ns,) int32
    state_types = jnp.asarray(_mixfrag_state_type_ids(F), dtype=jnp.int32)
    anc = jnp.clip(jnp.asarray(anc_pad, dtype=jnp.int32), 0, A)
    desc = jnp.clip(jnp.asarray(desc_pad, dtype=jnp.int32), 0, A)
    st_log_pi = log_pis[frag_idx]                                         # (ns, A+1)
    st_log_sub = log_subs[frag_idx]                                       # (ns, A+1, A+1)
    match_emit = (st_log_pi[:, anc] + st_log_sub[:, anc, desc]).T         # (L_pad, ns)
    ins_emit = st_log_pi[:, desc].T
    del_emit = st_log_pi[:, anc].T
    col = jnp.asarray(states_pad, dtype=jnp.int32)
    M_mask = (col == TYPE_M)[:, None] & (state_types == TYPE_M)[None, :]
    I_mask = (col == TYPE_I)[:, None] & (state_types == TYPE_I)[None, :]
    D_mask = (col == TYPE_D)[:, None] & (state_types == TYPE_D)[None, :]
    emit = jnp.where(M_mask, match_emit, NEG_INF)
    emit = jnp.where(I_mask, ins_emit, emit)
    emit = jnp.where(D_mask, del_emit, emit)
    real = jnp.arange(L_pad, dtype=jnp.int32)[:, None] < jnp.asarray(L_real, dtype=jnp.int32)
    return jnp.where(real, emit, NEG_INF)


def _mixfrag_s_residue_stats_1d(post, states_pad, anc_pad, desc_pad, L_real,
                                F, A):
    """Per-fragtype residue statistics from the (L_pad, 3F+2) column
    posteriors: mc[f,a,b] over M columns, ins[f,b] over I columns, del[f,a]
    over D columns.  Wildcards (index >= A) go to a dropped one-hot slot."""
    _, M0, I0, D0, _ = mixfrag_pair_index(F)
    L_pad = int(states_pad.shape[0])
    dt = post.dtype
    col = jnp.asarray(states_pad, dtype=jnp.int32)
    real = jnp.arange(L_pad, dtype=jnp.int32) < jnp.asarray(L_real, dtype=jnp.int32)
    oa = jax.nn.one_hot(jnp.clip(jnp.asarray(anc_pad, dtype=jnp.int32), 0, A),
                        A + 1, dtype=dt)                                  # (L_pad, A+1)
    ob = jax.nn.one_hot(jnp.clip(jnp.asarray(desc_pad, dtype=jnp.int32), 0, A),
                        A + 1, dtype=dt)
    post_M = post[:, M0:M0 + F] * ((col == TYPE_M) & real)[:, None]
    post_I = post[:, I0:I0 + F] * ((col == TYPE_I) & real)[:, None]
    post_D = post[:, D0:D0 + F] * ((col == TYPE_D) & real)[:, None]
    tmp = jnp.einsum('lf,la->lfa', post_M, oa)
    mc = jnp.einsum('lfa,lb->fab', tmp, ob)[:, :A, :A]
    ins = jnp.einsum('lf,lb->fb', post_I, ob)[:, :A]
    dele = jnp.einsum('lf,la->fa', post_D, oa)[:, :A]
    return mc, ins, dele


def _estep_pair_mixfrag_s_path_core(states_pad, anc_pad, desc_pad, L_real, t,
                                    lam, mu, exts, weights, Qs, pis,
                                    scan_mode='associative'):
    """Per-pair 1D alignment-conditioned MixFrag-S E-step core.  Returns the
    same dict as ``_estep_pair_mixfrag_s_core``."""
    tau = mixfrag_trans(lam, mu, t, exts, weights)
    log_trans = jnp.log(jnp.maximum(tau, 1e-30))
    F = int(exts.shape[0])
    A = int(pis.shape[-1])
    ns = 3 * F + 2
    emit = _mixfrag_s_log_emit_columns(states_pad, anc_pad, desc_pad, L_real,
                                       t, Qs, pis)
    log_p, post, n_chi = forward_backward_1d_padded(
        log_trans, emit, L_real, init_state=0, final_state=ns - 1,
        scan_mode=scan_mode)
    resolved5, F_f, E_f, L_coef, M_coef = _mixfrag_chi_resolve_core(
        n_chi, lam, mu, t, exts, weights)
    mc, ins, dele = _mixfrag_s_residue_stats_1d(
        post, states_pad, anc_pad, desc_pad, L_real, F, A)
    W, U = jax.vmap(holmes_rubin_weighted_stats,
                    in_axes=(0, 0, None, 0))(Qs, pis, t, mc)
    return {'log_p': log_p, 'resolved5': resolved5, 'F_f': F_f, 'E_f': E_f,
            'L': L_coef, 'M': M_coef, 'mc': mc, 'ins': ins, 'del': dele,
            'W': W, 'U': U}


def _estep_pair_mixfrag_s_path_forward_only_core(states_pad, anc_pad, desc_pad,
                                                 L_real, t, lam, mu, exts,
                                                 weights, Qs, pis,
                                                 scan_mode='associative'):
    tau = mixfrag_trans(lam, mu, t, exts, weights)
    log_trans = jnp.log(jnp.maximum(tau, 1e-30))
    F = int(exts.shape[0])
    ns = 3 * F + 2
    emit = _mixfrag_s_log_emit_columns(states_pad, anc_pad, desc_pad, L_real,
                                       t, Qs, pis)
    return forward_backward_1d_padded(
        log_trans, emit, L_real, init_state=0, final_state=ns - 1,
        forward_only=True, scan_mode=scan_mode)


_ESTEP_MFS_PATH_JIT = {}
_ESTEP_MFS_PATH_FWD_JIT = {}


def _get_estep_mfs_path_jit(scan_mode='associative'):
    if scan_mode not in _ESTEP_MFS_PATH_JIT:
        core = lambda *a: _estep_pair_mixfrag_s_path_core(*a, scan_mode=scan_mode)
        vmapped = jax.vmap(
            core, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None))
        _ESTEP_MFS_PATH_JIT[scan_mode] = jax.jit(vmapped)
    return _ESTEP_MFS_PATH_JIT[scan_mode]


def _get_estep_mfs_path_fwd_jit(scan_mode='associative'):
    if scan_mode not in _ESTEP_MFS_PATH_FWD_JIT:
        core = lambda *a: _estep_pair_mixfrag_s_path_forward_only_core(
            *a, scan_mode=scan_mode)
        vmapped = jax.vmap(
            core, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None))
        _ESTEP_MFS_PATH_FWD_JIT[scan_mode] = jax.jit(vmapped)
    return _ESTEP_MFS_PATH_FWD_JIT[scan_mode]


def estep_batch_mixfrag_s_path(states_pad, anc_pad, desc_pad, L_real, ts,
                               lam, mu, exts, weights, Qs, pis,
                               scan_mode='associative'):
    """Batched (vmap'd) 1D alignment-conditioned MixFrag-S E-step.

    states_pad / anc_pad / desc_pad: (B, L_pad) int; L_real (B,) real column
    counts; ts (B,).  Returns the ``estep_batch_mixfrag_s`` dict ('log_p',
    'B','D','S','L','M','T', 'F_f','E_f', 'mc','ins','del', 'W','U')."""
    ts = jnp.asarray(ts, dtype=jnp.float64)
    lam_j, mu_j, exts_j, weights_j, Qs_j, pis_j = _as_mfs_params(
        lam, mu, exts, weights, Qs, pis)
    fn = _get_estep_mfs_path_jit(scan_mode)
    out = fn(jnp.asarray(states_pad, dtype=jnp.int32),
             jnp.asarray(anc_pad, dtype=jnp.int32),
             jnp.asarray(desc_pad, dtype=jnp.int32),
             jnp.asarray(L_real, dtype=jnp.int32), ts,
             lam_j, mu_j, exts_j, weights_j, Qs_j, pis_j)
    T_batch = ts
    E_B, E_D, E_S = _bdi_stats_batch(out['resolved5'], lam, mu, ts, T_batch)
    return {
        'log_p': out['log_p'],
        'B': jnp.maximum(E_B, 0.0), 'D': jnp.maximum(E_D, 0.0),
        'S': jnp.maximum(E_S, 0.0),
        'L': out['L'], 'M': out['M'], 'T': T_batch,
        'F_f': out['F_f'], 'E_f': out['E_f'],
        'mc': out['mc'], 'ins': out['ins'], 'del': out['del'],
        'W': out['W'], 'U': out['U'],
    }


def estep_batch_mixfrag_s_path_forward_only(states_pad, anc_pad, desc_pad,
                                            L_real, ts, lam, mu, exts, weights,
                                            Qs, pis, scan_mode='associative'):
    fn = _get_estep_mfs_path_fwd_jit(scan_mode)
    return fn(jnp.asarray(states_pad, dtype=jnp.int32),
              jnp.asarray(anc_pad, dtype=jnp.int32),
              jnp.asarray(desc_pad, dtype=jnp.int32),
              jnp.asarray(L_real, dtype=jnp.int32),
              jnp.asarray(ts, dtype=jnp.float64),
              *_as_mfs_params(lam, mu, exts, weights, Qs, pis))


def bucket_pairs_lazy_path(pairs, bin_stride=1):
    """Group 'path' pairs (states, anc_full, desc_full, t) by the padded
    alignment length L_pad (FB geometric bins; see ``bucket_pairs_lazy``).
    Returns {L_pad: [pair, ...]}."""
    buckets = {}
    for p in pairs:
        key = _coarse_pad_to_bin(int(np.asarray(p[0]).shape[0]), bin_stride)
        buckets.setdefault(key, []).append(p)
    return buckets


def stack_bucket_lazy_path(items, L_pad, n_pad_to=None):
    """Pad+stack 'path' pairs of one bucket to (states, anc, desc, L_real, ts)
    int32/float64 arrays of shape (n, L_pad); the last item is repeated up to
    ``n_pad_to`` (callers slice [:n_real])."""
    n = len(items)
    n_out = n if n_pad_to is None else max(n, int(n_pad_to))
    st = np.zeros((n_out, L_pad), dtype=np.int32)
    an = np.zeros((n_out, L_pad), dtype=np.int32)
    de = np.zeros((n_out, L_pad), dtype=np.int32)
    Ls = np.zeros(n_out, dtype=np.int32)
    ts = np.zeros(n_out, dtype=np.float64)
    for i in range(n_out):
        s, a, d, t = items[min(i, n - 1)]
        L = int(np.asarray(s).shape[0])
        st[i, :L] = s
        an[i, :L] = a
        de[i, :L] = d
        Ls[i] = L
        ts[i] = t
    return st, an, de, Ls, ts


def chunk_size_for_bucket_path(L_pad, n_states, F, A, chunk_max=1024,
                               mem_bytes=2.0e9):
    """Pairs per vmapped 1D E-step call: the associative FB materialises a
    few (L_pad, ns, ns) float64 tensors per pair (prefix/suffix products and
    the per-position transition counts; ~8 is a safe envelope) and the HR
    step an (F, A, A, A, A) integral tensor per pair."""
    per_pair = 8.0 * L_pad * n_states * n_states * 8.0 + 2.0 * F * A ** 4 * 8.0
    return int(max(1, min(chunk_max, mem_bytes // per_pair)))


def estep_bucket_chunks_mixfrag_s_path(items, L_pad, lam, mu, exts, weights,
                                       Qs, pis, *, chunk_max=1024,
                                       mem_bytes=2.0e9, scan_mode='associative'):
    """1D E-step over one L_pad bucket in memory-sized chunks; returns
    (suff dict summed over items, total log_p, n_items)."""
    F, A = int(np.asarray(pis).shape[0]), int(np.asarray(pis).shape[1])
    chunk = chunk_size_for_bucket_path(L_pad, 3 * F + 2, F, A, chunk_max,
                                       mem_bytes)
    suff = _empty_suff_mfs(F, A)
    ll = 0.0
    for s in range(0, len(items), chunk):
        sub = items[s:s + chunk]
        n_real = len(sub)
        n_pad = _round_up_pow2(n_real, chunk)
        st, an, de, Ls, ts = stack_bucket_lazy_path(sub, L_pad, n_pad_to=n_pad)
        bd = estep_batch_mixfrag_s_path(st, an, de, Ls, ts, lam, mu, exts,
                                        weights, Qs, pis, scan_mode=scan_mode)
        agg = _aggregate_chunk_suff_mfs(bd, n_real, np.asarray(bd['W']),
                                        np.asarray(bd['U']))
        for k in suff:
            suff[k] = suff[k] + agg[k]
        ll += float(jnp.sum(bd['log_p'][:n_real]))
    return suff, ll, len(items)


def loglik_bucketed_mixfrag_s_path(buckets, lam, mu, exts, weights, Qs, pis,
                                   *, chunk_max=1024, mem_bytes=2.0e9,
                                   scan_mode='associative'):
    """Total forward-only 1D log-likelihood over {L_pad: [path pairs]}."""
    F, A = int(np.asarray(pis).shape[0]), int(np.asarray(pis).shape[1])
    total, n = 0.0, 0
    for L_pad, items in buckets.items():
        chunk = chunk_size_for_bucket_path(L_pad, 3 * F + 2, F, A, chunk_max,
                                           mem_bytes * 4.0)
        for s in range(0, len(items), chunk):
            sub = items[s:s + chunk]
            n_real = len(sub)
            st, an, de, Ls, ts = stack_bucket_lazy_path(
                sub, L_pad, n_pad_to=_round_up_pow2(n_real, chunk))
            lp = estep_batch_mixfrag_s_path_forward_only(
                st, an, de, Ls, ts, lam, mu, exts, weights, Qs, pis,
                scan_mode=scan_mode)
            total += float(jnp.sum(lp[:n_real]))
            n += n_real
    return total, n


# -------------------------------------------------------------------------
# 'match_aligned': 2D with the matches pinned to the alignment.
# -------------------------------------------------------------------------


def _estep_pair_mixfrag_s_ma_core(x_pad, y_pad, t, Lx_real, Ly_real, mi, mj,
                                  lam, mu, exts, weights, Qs, pis):
    """``_estep_pair_mixfrag_s_core`` with ``mask_emissions_match_pinned``
    applied to the per-fragtype table: every path is forced through each
    alignment match cell (mi[k]+1, mj[k]+1) by an M_f step (M_f may fire
    nowhere else, and no I/D path may bypass a match cell); padding slots in
    mi/mj hold -1 and constrain nothing."""
    tau = mixfrag_trans(lam, mu, t, exts, weights)
    log_trans = jnp.log(jnp.maximum(tau, 1e-30))
    F = int(exts.shape[0])
    A = int(pis.shape[-1])
    state_types = jnp.array(
        [TYPE_S] + [TYPE_M] * F + [TYPE_I] * F + [TYPE_D] * F + [TYPE_E],
        dtype=jnp.int32)
    table = _mixfrag_s_log_emit_table(state_types, x_pad, y_pad, t, Qs, pis)
    table = mask_emissions_match_pinned(table, state_types,
                                        jnp.asarray(mi, dtype=jnp.int32),
                                        jnp.asarray(mj, dtype=jnp.int32))
    log_p, post, n_chi = forward_backward_2d(
        log_trans, state_types, x_pad, y_pad, None, None,
        log_emit_table=table, real_Lx=Lx_real, real_Ly=Ly_real)
    resolved5, F_f, E_f, L_coef, M_coef = _mixfrag_chi_resolve_core(
        n_chi, lam, mu, t, exts, weights)
    mc, ins, dele = _mixfrag_s_residue_stats(
        post, x_pad, y_pad, Lx_real, Ly_real, F, A)
    W, U = jax.vmap(holmes_rubin_weighted_stats,
                    in_axes=(0, 0, None, 0))(Qs, pis, t, mc)
    return {'log_p': log_p, 'resolved5': resolved5, 'F_f': F_f, 'E_f': E_f,
            'L': L_coef, 'M': M_coef, 'mc': mc, 'ins': ins, 'del': dele,
            'W': W, 'U': U}


def _estep_pair_mixfrag_s_ma_forward_only_core(x_pad, y_pad, t, Lx_real,
                                               Ly_real, mi, mj, lam, mu, exts,
                                               weights, Qs, pis):
    tau = mixfrag_trans(lam, mu, t, exts, weights)
    log_trans = jnp.log(jnp.maximum(tau, 1e-30))
    F = int(exts.shape[0])
    state_types = jnp.array(
        [TYPE_S] + [TYPE_M] * F + [TYPE_I] * F + [TYPE_D] * F + [TYPE_E],
        dtype=jnp.int32)
    table = _mixfrag_s_log_emit_table(state_types, x_pad, y_pad, t, Qs, pis)
    table = mask_emissions_match_pinned(table, state_types,
                                        jnp.asarray(mi, dtype=jnp.int32),
                                        jnp.asarray(mj, dtype=jnp.int32))
    return forward_backward_2d(
        log_trans, state_types, x_pad, y_pad, None, None,
        log_emit_table=table, real_Lx=Lx_real, real_Ly=Ly_real,
        forward_only=True)


_ESTEP_MFS_MA_JIT = None
_ESTEP_MFS_MA_FWD_JIT = None


def _get_estep_mfs_ma_jit():
    global _ESTEP_MFS_MA_JIT
    if _ESTEP_MFS_MA_JIT is None:
        vmapped = jax.vmap(
            _estep_pair_mixfrag_s_ma_core,
            in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None))
        _ESTEP_MFS_MA_JIT = jax.jit(vmapped)
    return _ESTEP_MFS_MA_JIT


def _get_estep_mfs_ma_fwd_jit():
    global _ESTEP_MFS_MA_FWD_JIT
    if _ESTEP_MFS_MA_FWD_JIT is None:
        vmapped = jax.vmap(
            _estep_pair_mixfrag_s_ma_forward_only_core,
            in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None))
        _ESTEP_MFS_MA_FWD_JIT = jax.jit(vmapped)
    return _ESTEP_MFS_MA_FWD_JIT


def estep_batch_mixfrag_s_match_aligned(xs_pad, ys_pad, ts, Lx_real, Ly_real,
                                        mis, mjs, lam, mu, exts, weights,
                                        Qs, pis):
    """Batched match-aligned 2D MixFrag-S E-step; mis/mjs (B, n_match_pad)
    int32 with -1 in padding slots.  Returns the ``estep_batch_mixfrag_s``
    dict."""
    ts = jnp.asarray(ts, dtype=jnp.float64)
    lam_j, mu_j, exts_j, weights_j, Qs_j, pis_j = _as_mfs_params(
        lam, mu, exts, weights, Qs, pis)
    fn = _get_estep_mfs_ma_jit()
    out = fn(jnp.asarray(xs_pad), jnp.asarray(ys_pad), ts,
             jnp.asarray(Lx_real, dtype=jnp.int32),
             jnp.asarray(Ly_real, dtype=jnp.int32),
             jnp.asarray(mis, dtype=jnp.int32), jnp.asarray(mjs, dtype=jnp.int32),
             lam_j, mu_j, exts_j, weights_j, Qs_j, pis_j)
    T_batch = ts
    E_B, E_D, E_S = _bdi_stats_batch(out['resolved5'], lam, mu, ts, T_batch)
    return {
        'log_p': out['log_p'],
        'B': jnp.maximum(E_B, 0.0), 'D': jnp.maximum(E_D, 0.0),
        'S': jnp.maximum(E_S, 0.0),
        'L': out['L'], 'M': out['M'], 'T': T_batch,
        'F_f': out['F_f'], 'E_f': out['E_f'],
        'mc': out['mc'], 'ins': out['ins'], 'del': out['del'],
        'W': out['W'], 'U': out['U'],
    }


def estep_batch_mixfrag_s_match_aligned_forward_only(xs_pad, ys_pad, ts,
                                                     Lx_real, Ly_real, mis, mjs,
                                                     lam, mu, exts, weights,
                                                     Qs, pis):
    fn = _get_estep_mfs_ma_fwd_jit()
    return fn(jnp.asarray(xs_pad), jnp.asarray(ys_pad),
              jnp.asarray(ts, dtype=jnp.float64),
              jnp.asarray(Lx_real, dtype=jnp.int32),
              jnp.asarray(Ly_real, dtype=jnp.int32),
              jnp.asarray(mis, dtype=jnp.int32), jnp.asarray(mjs, dtype=jnp.int32),
              *_as_mfs_params(lam, mu, exts, weights, Qs, pis))


def bucket_pairs_lazy_match_aligned(pairs, bin_stride=1):
    """Group (x, y, t, mi, mj) pairs by (Lx_pad, Ly_pad)."""
    buckets = {}
    for p in pairs:
        key = (_coarse_pad_to_bin(int(p[0].shape[0]), bin_stride),
               _coarse_pad_to_bin(int(p[1].shape[0]), bin_stride))
        buckets.setdefault(key, []).append(p)
    return buckets


def stack_bucket_lazy_match_aligned(items, key, n_pad_to=None):
    """Pad+stack (x, y, t, mi, mj) pairs to (xs, ys, ts, Lx, Ly, mis, mjs);
    match arrays are padded with -1 to a power-of-two slot count (fewer JIT
    shapes)."""
    xs, ys, ts, Lx, Ly = stack_bucket_lazy([(x, y, t) for x, y, t, _, _ in items],
                                           key, n_pad_to)
    n = len(items)
    n_out = xs.shape[0]
    nm = max(1, max(int(np.asarray(p[3]).shape[0]) for p in items))
    nm = _round_up_pow2(nm, 1 << 30)
    mis = np.full((n_out, nm), -1, dtype=np.int32)
    mjs = np.full((n_out, nm), -1, dtype=np.int32)
    for i in range(n_out):
        _, _, _, mi, mj = items[min(i, n - 1)]
        k = int(np.asarray(mi).shape[0])
        mis[i, :k] = mi
        mjs[i, :k] = mj
    return xs, ys, ts, Lx, Ly, mis, mjs


def estep_bucket_chunks_mixfrag_s_match_aligned(items, key, lam, mu, exts,
                                                weights, Qs, pis, *,
                                                chunk_max=256, mem_bytes=2.0e9):
    """Match-aligned 2D E-step over one (Lx_pad, Ly_pad) bucket in chunks;
    returns (suff dict, total log_p, n_items)."""
    F, A = int(np.asarray(pis).shape[0]), int(np.asarray(pis).shape[1])
    chunk = chunk_size_for_bucket(key, 3 * F + 2, chunk_max, mem_bytes)
    suff = _empty_suff_mfs(F, A)
    ll = 0.0
    for s in range(0, len(items), chunk):
        sub = items[s:s + chunk]
        n_real = len(sub)
        n_pad = _round_up_pow2(n_real, chunk)
        xs, ys, ts, Lx, Ly, mis, mjs = stack_bucket_lazy_match_aligned(
            sub, key, n_pad_to=n_pad)
        bd = estep_batch_mixfrag_s_match_aligned(
            xs, ys, ts, Lx, Ly, mis, mjs, lam, mu, exts, weights, Qs, pis)
        agg = _aggregate_chunk_suff_mfs(bd, n_real, np.asarray(bd['W']),
                                        np.asarray(bd['U']))
        for k in suff:
            suff[k] = suff[k] + agg[k]
        ll += float(jnp.sum(bd['log_p'][:n_real]))
    return suff, ll, len(items)


def loglik_bucketed_mixfrag_s_match_aligned(buckets, lam, mu, exts, weights,
                                            Qs, pis, *, chunk_max=256,
                                            mem_bytes=2.0e9):
    F = int(np.asarray(pis).shape[0])
    total, n = 0.0, 0
    for key, items in buckets.items():
        chunk = chunk_size_for_bucket(key, 3 * F + 2, chunk_max, mem_bytes * 4.0)
        for s in range(0, len(items), chunk):
            sub = items[s:s + chunk]
            n_real = len(sub)
            xs, ys, ts, Lx, Ly, mis, mjs = stack_bucket_lazy_match_aligned(
                sub, key, n_pad_to=_round_up_pow2(n_real, chunk))
            lp = estep_batch_mixfrag_s_match_aligned_forward_only(
                xs, ys, ts, Lx, Ly, mis, mjs, lam, mu, exts, weights, Qs, pis)
            total += float(jnp.sum(lp[:n_real]))
            n += n_real
    return total, n


# -------------------------------------------------------------------------
# Mode dispatch used by the training loop.
# -------------------------------------------------------------------------


def mixfrag_s_mode_fns(dp_mode, *, chunk_max, mem_bytes, bin_stride,
                       scan_mode='associative'):
    """(bucket_fn(pairs), estep_fn(items, key, lam, mu, exts, w, Qs, pis)
    -> (suff, ll, n), loglik_fn(buckets, lam, mu, exts, w, Qs, pis)
    -> (ll, n)) for ``dp_mode``."""
    if dp_mode == 'path':
        return (
            lambda pairs: bucket_pairs_lazy_path(pairs, bin_stride),
            lambda items, key, *p: estep_bucket_chunks_mixfrag_s_path(
                items, key, *p, chunk_max=chunk_max, mem_bytes=mem_bytes,
                scan_mode=scan_mode),
            lambda buckets, *p: loglik_bucketed_mixfrag_s_path(
                buckets, *p, chunk_max=chunk_max, mem_bytes=mem_bytes,
                scan_mode=scan_mode),
        )
    if dp_mode == 'match_aligned':
        return (
            lambda pairs: bucket_pairs_lazy_match_aligned(pairs, bin_stride),
            lambda items, key, *p: estep_bucket_chunks_mixfrag_s_match_aligned(
                items, key, *p, chunk_max=chunk_max, mem_bytes=mem_bytes),
            lambda buckets, *p: loglik_bucketed_mixfrag_s_match_aligned(
                buckets, *p, chunk_max=chunk_max, mem_bytes=mem_bytes),
        )
    if dp_mode == 'unaligned':
        return (
            lambda pairs: bucket_pairs_lazy(pairs, bin_stride),
            lambda items, key, *p: estep_bucket_chunks_mixfrag_s(
                items, key, *p, chunk_max=chunk_max, mem_bytes=mem_bytes),
            lambda buckets, *p: loglik_bucketed_mixfrag_s(
                buckets, *p, chunk_max=chunk_max, mem_bytes=mem_bytes),
        )
    raise ValueError(f'dp_mode must be one of {DP_MODES}, got {dp_mode!r}')
