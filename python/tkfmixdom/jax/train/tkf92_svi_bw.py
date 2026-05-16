"""Pure-TKF92 SVI-Baum-Welch training on pair data.

This is the alignment-MARGINALISED training path for plain TKF92 (no
MixDom hierarchy).  Each pair (ancestor, descendant, t) contributes its
expected sufficient statistics via 2D pair-HMM forward-backward on the
TKF92 chi=tau92 5x5 matrix; per-pair-t correctness is preserved by
calling ``tkf92_stats_from_counts`` at each pair's own t and summing
the resulting (B, D, S, ext_count, notext_count) into a global
accumulator.

This module is the TKF92-first-class analogue of MixDom2's
``train.tree_vbem`` / ``train_pfam.train_svi_bw``: same SVI EMA scheme,
same κ-quadratic M-step + Beta-posterior ext M-step, but with a single
(λ, μ, ext, Q, π) parameter set and no hierarchy.

For the §3 comparison Maraschino is alignment-GIVEN: cherry counts are
read off the Pfam alignment directly.  This module is the
alignment-MARGINALISED counterpart: the FB E-step sums over alignments
inside each per-pair invocation.

Public API:

  svi_bw_tkf92(pair_iter, init_lam, init_mu, init_ext, Q, pi, ...)
    Training loop.  ``pair_iter`` yields (x_int, y_int, t) tuples.
    Returns (final_params, history).

  estep_pair_tkf92(x, y, t, lam, mu, ext, Q, pi)
    Single-pair E-step: returns dict with B, D, S, L (log_kappa coef),
    M (log_1mkappa coef), T, ext_count, notext_count plus log_p.

The 1D / 1.5D variants (alignment-given and gap-ordering-marginal) are
NOT implemented here; for 1D use the cherry-count infrastructure (§3
Maraschino path).  The 1.5D mode (sum over gap orderings via the
hypergeometric gap probabilities of evolmoves/doc/tex/main.tex
sec:gapprob) is a future extension.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
import jax
import jax.numpy as jnp

from ..core.bdi import (
    tkf92_stats_from_counts, transition_count_groups, m_step_indel_quadratic,
)
from ..core.params import S as TYPE_S, M as TYPE_M, I as TYPE_I, D as TYPE_D, E as TYPE_E
from ..dp.hmm import forward_backward_2d
from ..models.left_regular import make_tkf92_pair_hmm


# -------------------------------------------------------------------------
# Single-pair E-step
# -------------------------------------------------------------------------


def estep_pair_tkf92(x_int, y_int, t, lam, mu, ext, Q, pi):
    """Single-pair TKF92 forward-backward E-step + per-pair-t suff stats.

    Args:
        x_int:  (Lx,) int array — ancestor residues (0..A-1).
        y_int:  (Ly,) int array — descendant residues (0..A-1).
        t:      scalar — branch length.
        lam:    insertion rate λ.
        mu:     deletion rate μ.
        ext:    fragment-extension probability ext ∈ [0, 1).
        Q:      (A, A) substitution rate matrix.
        pi:     (A,) stationary distribution.

    Returns:
        dict with:
          'log_p':         scalar.
          'B', 'D', 'S':   BDI suff stats.
          'L', 'M':        log_kappa / log_1mkappa coefficients.
          'T':             time × n_trans.sum() (for the pair).
          'ext_count':     fragment-extension event count.
          'notext_count':  body→body non-extension event count.
          'n_chi':         (5, 5) FB-derived chi count matrix.
    """
    from ..core.ctmc import transition_matrix
    sub_matrix = np.asarray(transition_matrix(Q, float(t)))
    log_trans, state_types, _, _ = make_tkf92_pair_hmm(
        float(lam), float(mu), float(t), float(ext), Q, np.asarray(pi))
    x_jax = jnp.asarray(x_int)
    y_jax = jnp.asarray(y_int)
    log_p, _, n_chi = forward_backward_2d(
        log_trans, state_types, x_jax, y_jax,
        sub_matrix, np.asarray(pi))
    n_chi_np = np.asarray(n_chi)
    # Per body-tkf92.tex sec:bw-tkf92: TKF92 BW has TWO key requirements:
    # (1) T is the IMMORTAL-LINK observation time per pair = t (one BDI
    #     process per pair at the fragment level — fragments are the TKF91
    #     "links" of TKF92).  Total T across pairs = sum of t per pair.
    #     The MixDom2 t * n_trans.sum() convention does NOT apply.
    # (2) L (log_kappa coef) and M (log_1mkappa coef) must be computed on
    #     the RESOLVED count matrix n̂_{ab} = ñ'_{ab} - δ_{ab} F_a, where
    #     F_a = ñ'_{aa} · ext / (ext + (1-ext)·tau91_{aa}) are the fragment
    #     extensions to be subtracted from chi self-loops.
    T_pair = float(t)
    r = tkf92_stats_from_counts(
        n_chi_np, lam, mu, float(t), float(ext), T=T_pair)
    groups = transition_count_groups(r['n_trans_resolved'])
    # Numerical guard: when (λ, μ) drift into the exact L'Hôpital regime
    # the smooth-β/γ formula occasionally produces tiny negative E_S /
    # E_B / E_D from cancellation.  Clamp at 0 to keep the M-step quadratic
    # well-defined; the bias this introduces is O(1e-10) and does not
    # affect convergence.
    return {
        'log_p': float(log_p),
        'B': float(max(r['E_B'], 0.0)),
        'D': float(max(r['E_D'], 0.0)),
        'S': float(max(r['E_S'], 0.0)),
        'L': float(groups['log_kappa']),
        'M': float(groups['log_1mkappa']),
        'T': T_pair,
        'ext_count': float(r['ext_count']),
        'notext_count': float(r['notext_count']),
        'n_chi': n_chi_np,
    }


# -------------------------------------------------------------------------
# M-step helpers
# -------------------------------------------------------------------------


def m_step_lam_mu(suff, prior_alpha_lam=2.0, prior_alpha_mu=2.0,
                    prior_beta=10.0):
    """κ-quadratic joint M-step on (λ, μ) given a suff dict."""
    return m_step_indel_quadratic(
        B=suff['B'], D=suff['D'], S=suff['S'],
        L=suff['L'], M=suff['M'], T=suff['T'],
        prior_alpha_lam=prior_alpha_lam,
        prior_alpha_mu=prior_alpha_mu,
        prior_beta=prior_beta)


def m_step_ext(suff, prior_alpha=2.0, prior_beta=3.0):
    """Beta(α, β) posterior on the extension probability ext."""
    a = suff['ext_count'] + prior_alpha - 1.0
    b = suff['notext_count'] + prior_beta - 1.0
    if a <= 0 and b <= 0:
        return 0.5
    if a + b <= 1e-9:
        return 0.5
    return float(a / (a + b))


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------


def _empty_suff():
    return {'B': 0.0, 'D': 0.0, 'S': 0.0, 'L': 0.0, 'M': 0.0, 'T': 0.0,
            'ext_count': 0.0, 'notext_count': 0.0}


def _scaled_add(suff_blend, suff_minibatch, eta, scale):
    """SVI EMA blend: suff_blend ← (1 - eta) suff_blend + eta · scale · suff_minibatch."""
    out = {}
    for k in suff_blend:
        out[k] = (1.0 - eta) * suff_blend[k] \
            + eta * scale * suff_minibatch[k]
    return out


def svi_bw_tkf92(pair_iter, *, n_total_pairs,
                   init_lam=0.05, init_mu=0.05, init_ext=0.5,
                   Q, pi,
                   n_iter=200, batch_size=50,
                   svi_tau=1.0, svi_kappa=0.7,
                   prior_alpha_lam=2.0, prior_alpha_mu=2.0, prior_beta=10.0,
                   ext_prior_alpha=2.0, ext_prior_beta=3.0,
                   log_fn=print, seed=0):
    """SVI-BW on plain TKF92 pair data.

    Args:
        pair_iter: callable() -> generator yielding (x_int, y_int, t)
                   tuples.  Each call produces an INDEPENDENT pass.
        n_total_pairs: int — total number of pairs in the corpus
                       (used to scale per-minibatch suff stats).
        init_lam, init_mu, init_ext: initial parameters.
        Q, pi:     fixed substitution model (rate matrix + stationary).
        n_iter:    number of SVI iterations.
        batch_size: per-iter minibatch size.
        svi_tau, svi_kappa: SVI step-size schedule
                           η_k = (svi_tau + k)^(-svi_kappa).
        prior_*:   M-step priors.
        log_fn:    progress logger.
        seed:      RNG seed for sampler.

    Returns:
        dict with final 'lam', 'mu', 'ext', plus 'history' (list of
        per-iter dicts).
    """
    rng = np.random.default_rng(seed)
    lam, mu, ext = float(init_lam), float(init_mu), float(init_ext)
    suff_blend = _empty_suff()
    history = []

    # Materialise pairs once (small enough to fit in memory for Pfam-cherry
    # data; ~1.1M pairs at ~1KB each = 1GB, manageable).
    pairs = list(pair_iter())
    if not pairs:
        raise ValueError('Pair iterator produced no pairs.')
    n_actual = len(pairs)
    log_fn(f'svi_bw_tkf92: {n_actual} pairs loaded; '
            f'n_total_pairs (scale)={n_total_pairs}, '
            f'batch_size={batch_size}, n_iter={n_iter}.')

    scale = float(n_total_pairs) / float(batch_size)
    t0 = time.time()
    for k in range(n_iter):
        eta_k = (svi_tau + k) ** (-svi_kappa)
        idx = rng.choice(n_actual, batch_size, replace=False)
        suff_mb = _empty_suff()
        ll_mb = 0.0
        for i in idx:
            x, y, t_pair = pairs[i]
            r = estep_pair_tkf92(x, y, t_pair, lam, mu, ext, Q, pi)
            for kk in suff_mb:
                suff_mb[kk] += r[kk]
            ll_mb += r['log_p']

        if k == 0:
            # First iteration: no prior blend; fully replace.
            suff_blend = {kk: scale * suff_mb[kk] for kk in suff_mb}
        else:
            suff_blend = _scaled_add(suff_blend, suff_mb, eta_k, scale)

        # M-step on the blended suff stats.
        lam_new, mu_new = m_step_lam_mu(
            suff_blend, prior_alpha_lam=prior_alpha_lam,
            prior_alpha_mu=prior_alpha_mu, prior_beta=prior_beta)
        ext_new = m_step_ext(suff_blend, ext_prior_alpha, ext_prior_beta)

        history.append({
            'iter': k + 1,
            'eta': eta_k,
            'mb_log_p_mean': ll_mb / batch_size,
            'lam': lam_new,
            'mu': mu_new,
            'ext': ext_new,
            'B_blend': suff_blend['B'],
            'D_blend': suff_blend['D'],
            'S_blend': suff_blend['S'],
        })
        lam, mu, ext = lam_new, mu_new, ext_new

        if (k + 1) % 10 == 0 or k == 0 or k == n_iter - 1:
            log_fn(f'  iter {k+1:>4}/{n_iter}: λ={lam:.5f} μ={mu:.5f} '
                    f'ext={ext:.4f} eta={eta_k:.4f} '
                    f'mb_ll/pair={ll_mb/batch_size:.2f} '
                    f'({time.time()-t0:.1f}s)')

    return {'lam': lam, 'mu': mu, 'ext': ext, 'history': history}
