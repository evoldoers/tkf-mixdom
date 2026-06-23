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
