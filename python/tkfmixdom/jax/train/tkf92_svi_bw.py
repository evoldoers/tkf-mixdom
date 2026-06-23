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

  estep_batch_tkf92(xs_pad, ys_pad, ts, Lx_real, Ly_real,
                    lam, mu, ext, Q, pi)
    Vmap'd over a bin-bucketed minibatch (all pairs share the same
    Lx_pad, Ly_pad).  Returns jnp arrays for log_p, B, D, S, T,
    ext_count, notext_count, L (log_kappa coef), M (log_1mkappa coef),
    and n_chi_diag.  This replaces the Python-loop / per-pair-FB cost
    in svi_bw_tkf92's E-step.  The `n_chi_diag` array is DIAGNOSTIC
    ONLY — production likelihood / gradient / M-step paths must consume
    B, D, S, L, M, T (the t-parameter-free BDI form).  See the
    docstring of `estep_batch_tkf92` for the rationale.

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
    m_step_indel_quadratic, ES_LHOPITAL_THRESHOLD,
)
from ..core.ctmc import transition_matrix
from ..core.params import (
    S as TYPE_S, M as TYPE_M, I as TYPE_I, D as TYPE_D, E as TYPE_E,
    tkf92_trans,
)
from ..dp.hmm import forward_backward_2d


# -------------------------------------------------------------------------
# JAX-traceable chi decomposition (replaces the numpy block in
# tkf92_stats_from_counts so it can run under vmap).
# -------------------------------------------------------------------------


def _tkf92_chi_resolve_core(n_chi, ins_rate, del_rate, t, ext):
    """JAX-traceable analogue of the numpy chi-decomposition done at the
    top of tkf92_stats_from_counts.

    Given a (5, 5) chi count matrix in TKF92 semantics (chi self-loops
    mix fragment-extensions with new-fragment-with-s events), produce:

      * n_trans_resolved (5, 5): the TKF91-style count matrix obtained
        by subtracting ext_part from each diagonal.
      * ext_count: scalar — total expected fragment-extension events.
      * notext_count: scalar — body-row sum on the resolved matrix
        (denominator for the ext-Beta M-step).

    Per body-tkf92.tex sec:bw-tkf92 (the resolved n̂_{ab} = ñ'_{ab} -
    δ_{ab}·F_a, with F_a = ext_frac[s]·ñ'_{aa}).
    """
    tau92 = tkf92_trans(ins_rate, del_rate, t, ext)

    # ext_frac[s] = ext / tau92[s, s] for s in {M, I, D}; 0 if diag tiny.
    # Use clamp like the numpy version (max(0, min(1, frac))) to be safe.
    diag92 = jnp.array([tau92[TYPE_M, TYPE_M],
                         tau92[TYPE_I, TYPE_I],
                         tau92[TYPE_D, TYPE_D]])
    # ext_frac defined where diag92 > tiny; else 0 (no extension to subtract).
    ext_frac = jnp.where(diag92 > 1e-30, ext / jnp.maximum(diag92, 1e-30), 0.0)
    ext_frac = jnp.clip(ext_frac, 0.0, 1.0)

    # ext_part[s] = ext_frac[s] * n_chi[s, s]; subtract from diagonal.
    loop_counts = jnp.array([n_chi[TYPE_M, TYPE_M],
                              n_chi[TYPE_I, TYPE_I],
                              n_chi[TYPE_D, TYPE_D]])
    ext_part = ext_frac * loop_counts
    ext_count = jnp.sum(ext_part)

    # Build resolved n_trans by replacing the M, I, D diagonals.
    n_trans_resolved = n_chi
    n_trans_resolved = n_trans_resolved.at[TYPE_M, TYPE_M].set(
        (1.0 - ext_frac[0]) * loop_counts[0])
    n_trans_resolved = n_trans_resolved.at[TYPE_I, TYPE_I].set(
        (1.0 - ext_frac[1]) * loop_counts[1])
    n_trans_resolved = n_trans_resolved.at[TYPE_D, TYPE_D].set(
        (1.0 - ext_frac[2]) * loop_counts[2])

    # notext_count = body-row sum (rows M, I, D over all dest) of resolved.
    notext_count = (n_trans_resolved[TYPE_M, :].sum()
                    + n_trans_resolved[TYPE_I, :].sum()
                    + n_trans_resolved[TYPE_D, :].sum())

    return n_trans_resolved, ext_count, notext_count


# -------------------------------------------------------------------------
# Per-pair core (JAX-only, vmappable, jittable)
# -------------------------------------------------------------------------


def _estep_pair_tkf92_core(x_pad, y_pad, t, Lx_real, Ly_real,
                            lam, mu, ext, Q, pi):
    """Per-pair E-step core that runs entirely in JAX (no numpy).

    Computes:
      * log_p — joint pair log-LL.
      * n_chi (5, 5) — expected transition counts under TKF92 chi.
      * resolved (5, 5) — chi counts with extensions subtracted off the
        diagonal.
      * ext_count, notext_count — extension-Beta M-step counts.
      * L, M — log_kappa / log_1mkappa coefficients from the resolved
        matrix.

    Padding semantics: x_pad / y_pad are pre-padded to bin sizes, and
    Lx_real / Ly_real are the unpadded real lengths.
    forward_backward_2d masks emissions outside (Lx_real, Ly_real) to
    NEG_INF and reads log_p / chi at the (Lx_real, Ly_real) endpoint;
    padded positions contribute zero to both log_p and the chi matrix.
    """
    # Build log_trans (5x5 tau92) and substitution matrix per-pair (t varies).
    tau = tkf92_trans(lam, mu, t, ext)
    log_trans = jnp.log(jnp.maximum(tau, 1e-30))
    state_types = jnp.array([TYPE_S, TYPE_M, TYPE_I, TYPE_D, TYPE_E],
                             dtype=jnp.int32)
    sub = transition_matrix(Q, t)
    log_p, _, n_chi = forward_backward_2d(
        log_trans, state_types, x_pad, y_pad, sub, pi,
        real_Lx=Lx_real, real_Ly=Ly_real)

    # Chi decomposition (JAX-traceable).
    n_trans_resolved, ext_count, notext_count = _tkf92_chi_resolve_core(
        n_chi, lam, mu, t, ext)

    # L (log_kappa coef) and M (log_1mkappa coef) from RESOLVED matrix
    # — sec:bw-tkf92 requires the resolved matrix here.
    # log_kappa group: sum over (a, b) with b ∈ {M, D} (eight entries).
    L_coef = (n_trans_resolved[TYPE_S, TYPE_M] + n_trans_resolved[TYPE_S, TYPE_D]
              + n_trans_resolved[TYPE_M, TYPE_M] + n_trans_resolved[TYPE_M, TYPE_D]
              + n_trans_resolved[TYPE_I, TYPE_M] + n_trans_resolved[TYPE_I, TYPE_D]
              + n_trans_resolved[TYPE_D, TYPE_M] + n_trans_resolved[TYPE_D, TYPE_D])
    # log_1mkappa group: sum over (a, E) for a ∈ {S, M, I, D}.
    M_coef = (n_trans_resolved[TYPE_S, TYPE_E] + n_trans_resolved[TYPE_M, TYPE_E]
              + n_trans_resolved[TYPE_I, TYPE_E] + n_trans_resolved[TYPE_D, TYPE_E])

    return {
        'log_p': log_p,
        'n_chi': n_chi,
        'n_trans_resolved': n_trans_resolved,
        'ext_count': ext_count,
        'notext_count': notext_count,
        'L': L_coef,
        'M': M_coef,
    }


def _estep_pair_tkf92_forward_only_core(x_pad, y_pad, t, Lx_real, Ly_real,
                                         lam, mu, ext, Q, pi):
    """Forward-only variant used for val_eval (no chi matrix, no backward).

    Returns just log_p; cheaper than the full E-step.
    """
    tau = tkf92_trans(lam, mu, t, ext)
    log_trans = jnp.log(jnp.maximum(tau, 1e-30))
    state_types = jnp.array([TYPE_S, TYPE_M, TYPE_I, TYPE_D, TYPE_E],
                             dtype=jnp.int32)
    sub = transition_matrix(Q, t)
    log_p = forward_backward_2d(
        log_trans, state_types, x_pad, y_pad, sub, pi,
        real_Lx=Lx_real, real_Ly=Ly_real, forward_only=True)
    return log_p


# -------------------------------------------------------------------------
# Batched (vmap'd + JIT'd) wrapper
# -------------------------------------------------------------------------


def _estep_batch_jit_compile(forward_only=False):
    """Return a jit(vmap(_estep_pair_tkf92_core)) function. Closed-form
    re-jit per (forward_only, bucket-shape) since JAX caches on input
    shape automatically.
    """
    core = (_estep_pair_tkf92_forward_only_core if forward_only
            else _estep_pair_tkf92_core)
    # in_axes: x, y, t, Lx_real, Ly_real are batched; lam, mu, ext, Q, pi shared.
    vmapped = jax.vmap(core, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None))
    return jax.jit(vmapped)


# Cache one jitted function per (forward_only) flag; JAX shape-caches the
# rest automatically.  We can't easily cache by Q, pi shape because they're
# closure-captured kwargs to the loss, but vmap captures them as the
# `None` axes so the JIT only re-compiles on (shape, dtype) changes.
_ESTEP_FULL_JIT = None
_ESTEP_FWD_JIT = None


def _get_estep_full_jit():
    global _ESTEP_FULL_JIT
    if _ESTEP_FULL_JIT is None:
        _ESTEP_FULL_JIT = _estep_batch_jit_compile(forward_only=False)
    return _ESTEP_FULL_JIT


def _get_estep_fwd_jit():
    global _ESTEP_FWD_JIT
    if _ESTEP_FWD_JIT is None:
        _ESTEP_FWD_JIT = _estep_batch_jit_compile(forward_only=True)
    return _ESTEP_FWD_JIT


def _bdi_stats_batch(n_trans_resolved_batch, ins_rate, del_rate, t_batch,
                      T_batch):
    """Vmapped per-pair BDI stats on the resolved (TKF91-style) matrices.

    Picks general / L'Hôpital branch on the shared (ins_rate, del_rate)
    scalars (single Python boolean for the whole batch — same logic as
    the existing `tkf91_stats_from_counts_batch`).
    """
    from ..core.bdi import (
        _get_bdi_general_batch_jit, _get_bdi_limit_batch_jit,
    )
    kappa = float(ins_rate) / float(del_rate)
    near_equal = abs(1.0 - kappa) < ES_LHOPITAL_THRESHOLD
    fn = _get_bdi_limit_batch_jit() if near_equal else _get_bdi_general_batch_jit()
    E_B, E_D, E_S = fn(n_trans_resolved_batch,
                        jnp.asarray(float(ins_rate)),
                        jnp.asarray(float(del_rate)),
                        t_batch, T_batch)
    return E_B, E_D, E_S


def estep_batch_tkf92(xs_pad, ys_pad, ts, Lx_real, Ly_real,
                       lam, mu, ext, Q, pi):
    """Batched (vmap'd) TKF92 E-step on a bin-bucketed minibatch.

    All pairs in the batch share the same (Lx_pad, Ly_pad) bin shape
    so a single JIT-compiled invocation handles the whole batch.

    Args:
        xs_pad: (B, Lx_pad) int32 — pre-padded ancestor sequences.
        ys_pad: (B, Ly_pad) int32 — pre-padded descendant sequences.
        ts:     (B,) — per-pair branch lengths.
        Lx_real, Ly_real: (B,) — unpadded real sequence lengths.
        lam, mu, ext: scalar TKF92 parameters (shared across batch).
        Q, pi: substitution model (shared across batch).

    Returns:
        dict of jnp arrays:
            'log_p'        (B,)   joint pair log-LL.
            'B', 'D', 'S'  (B,)   BDI suff stats (E[B], E[D], E[S]).
            'L', 'M'       (B,)   log_kappa / log_1mkappa coefficients.
            'T'            (B,)   per-pair BDI observation time (= ts).
            'ext_count'    (B,)   fragment-extension event count.
            'notext_count' (B,)   body→body non-extension event count.
            'n_chi_diag'   (B, 5, 5) per-pair chi count matrices.
                           Diagnostic-only output; do NOT consume in any
                           likelihood / gradient / M-step path. The correct
                           sufficient statistics for downstream consumption
                           are B, D, S, L, M, T (BDI form). The chi-axis
                           Q-function has no `t` parameter; using an
                           aggregated chi matrix as a suff-stat
                           re-introduces the t_rep approximation pattern
                           documented in
                           .claude/examples/per_pair_t_chi_axis_recidivism.md.
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
    ext_j = jnp.asarray(float(ext))

    fn = _get_estep_full_jit()
    out = fn(xs_pad, ys_pad, ts, Lx_real, Ly_real,
              lam_j, mu_j, ext_j, Qj, pij)

    # BDI stats are computed via a second vmapped+jitted call on the
    # already-resolved per-pair matrices.  Per-pair T defaults to per-pair t.
    T_batch = ts  # T = t for the single-process TKF92 case
    E_B, E_D, E_S = _bdi_stats_batch(
        out['n_trans_resolved'], lam, mu, ts, T_batch)

    # Numerical guard: clamp small negatives from L'Hôpital cancellation.
    E_B = jnp.maximum(E_B, 0.0)
    E_D = jnp.maximum(E_D, 0.0)
    E_S = jnp.maximum(E_S, 0.0)

    return {
        'log_p': out['log_p'],
        'B': E_B,
        'D': E_D,
        'S': E_S,
        'L': out['L'],
        'M': out['M'],
        'T': T_batch,
        'ext_count': out['ext_count'],
        'notext_count': out['notext_count'],
        'n_chi_diag': out['n_chi'],
    }


def estep_batch_tkf92_forward_only(xs_pad, ys_pad, ts, Lx_real, Ly_real,
                                     lam, mu, ext, Q, pi):
    """Forward-only batched E-step for val_eval.  Returns per-pair log_p only.

    Re-uses the same JIT cache as the full E-step when called with the same
    (Lx_pad, Ly_pad) bucket shapes, by sharing the forward path inside
    forward_backward_2d.
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
    ext_j = jnp.asarray(float(ext))

    fn = _get_estep_fwd_jit()
    return fn(xs_pad, ys_pad, ts, Lx_real, Ly_real,
               lam_j, mu_j, ext_j, Qj, pij)


# -------------------------------------------------------------------------
# Single-pair E-step (back-compat shim, equivalent to the old
# numpy-using version — preserved for callers that pass unpadded
# sequences without bin-bucketing).
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

    This is now a thin shim over `estep_batch_tkf92` with a batch of 1.
    The padding-correctness path is the same as the batched version so
    bugs in either auto-affect both.
    """
    from ..dp.hmm import _pad_to_bin
    x_arr = np.asarray(x_int)
    y_arr = np.asarray(y_int)
    Lx_real = int(x_arr.shape[0])
    Ly_real = int(y_arr.shape[0])
    Lx_pad = _pad_to_bin(Lx_real)
    Ly_pad = _pad_to_bin(Ly_real)
    x_pad = np.zeros(Lx_pad, dtype=x_arr.dtype)
    x_pad[:Lx_real] = x_arr
    y_pad = np.zeros(Ly_pad, dtype=y_arr.dtype)
    y_pad[:Ly_real] = y_arr

    out = estep_batch_tkf92(
        xs_pad=x_pad[None, :], ys_pad=y_pad[None, :],
        ts=np.array([float(t)]),
        Lx_real=np.array([Lx_real], dtype=np.int32),
        Ly_real=np.array([Ly_real], dtype=np.int32),
        lam=lam, mu=mu, ext=ext, Q=Q, pi=pi)

    # Unpack the batch-of-1 result and convert to python floats / numpy.
    return {
        'log_p': float(out['log_p'][0]),
        'B': float(out['B'][0]),
        'D': float(out['D'][0]),
        'S': float(out['S'][0]),
        'L': float(out['L'][0]),
        'M': float(out['M'][0]),
        'T': float(out['T'][0]),
        'ext_count': float(out['ext_count'][0]),
        'notext_count': float(out['notext_count'][0]),
        'n_chi': np.asarray(out['n_chi_diag'][0]),
    }


# -------------------------------------------------------------------------
# Bin-bucketed batching helpers (mirror tkf92_adam_fb._bin_bucket_pairs /
# _stack_bucket so train + val share the same compiled shapes).
# -------------------------------------------------------------------------


def _bin_bucket_pairs(pairs):
    """Group (x, y, t) pairs by (Lx_pad, Ly_pad) bin key.  Returns
    {(Lx_pad, Ly_pad): [(x_pad, y_pad, t, Lx_real, Ly_real), ...]}.
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
    batched jax arrays (xs, ys, ts, Lx, Ly)."""
    xs = np.stack([p[0] for p in pairs_in_bucket])
    ys = np.stack([p[1] for p in pairs_in_bucket])
    ts = np.array([p[2] for p in pairs_in_bucket], np.float64)
    Lx = np.array([p[3] for p in pairs_in_bucket], np.int32)
    Ly = np.array([p[4] for p in pairs_in_bucket], np.int32)
    return xs, ys, ts, Lx, Ly


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


def _aggregate_batch_suff(batch_dict):
    """Sum the batched per-pair suff stats into the scalar accumulator
    layout that _empty_suff / _scaled_add expects.
    """
    return {
        'B': float(jnp.sum(batch_dict['B'])),
        'D': float(jnp.sum(batch_dict['D'])),
        'S': float(jnp.sum(batch_dict['S'])),
        'L': float(jnp.sum(batch_dict['L'])),
        'M': float(jnp.sum(batch_dict['M'])),
        'T': float(jnp.sum(batch_dict['T'])),
        'ext_count': float(jnp.sum(batch_dict['ext_count'])),
        'notext_count': float(jnp.sum(batch_dict['notext_count'])),
    }


def svi_bw_tkf92(pair_iter, *, n_total_pairs,
                   init_lam=0.05, init_mu=0.05, init_ext=0.5,
                   Q, pi,
                   n_iter=200, batch_size=50,
                   svi_tau=1.0, svi_kappa=0.7,
                   prior_alpha_lam=2.0, prior_alpha_mu=2.0, prior_beta=10.0,
                   ext_prior_alpha=2.0, ext_prior_beta=3.0,
                   bin_bucketed=False, pre_warm=False,
                   val_pairs=None, val_every=0, patience=0,
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
        bin_bucketed: stratified-by-shape sampling.  Pre-bucket pairs by
                   (Lx_pad, Ly_pad) and draw each minibatch from one bucket
                   per iter; minibatch then hits only one JIT-compiled
                   shape instead of N independent shapes.  Drastically
                   reduces unique compiled CUDA graphs and dodges the
                   ~97-graph cap.  The vmap'd E-step (`estep_batch_tkf92`)
                   then computes all `batch_size` pairs' suff stats in
                   one JIT call rather than via a Python loop.
        pre_warm:  before the training loop, run one batched E-step for
                   each distinct (Lx_pad, Ly_pad) bin shape so the JIT
                   cache is fully warm.  Any OOM happens up-front, not
                   mid-run.
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

    # Pre-bucket train pairs into (Lx_pad, Ly_pad) cells; we always
    # stack-batch within a bucket whether or not bin_bucketed sampling
    # is on (so the vmap'd E-step has consistent shapes).
    #
    # train_buckets[(Lx_pad, Ly_pad)] = list of (x_pad, y_pad, t,
    #                                              Lx_real, Ly_real)
    # entries.  For minibatch construction we ALSO keep a per-pair
    # mapping from the corpus index → (bucket_key, idx_within_bucket)
    # so a minibatch sampled with arbitrary `rng.choice(n_actual, ...)`
    # can be grouped by bucket and dispatched as one batched call per
    # bucket shape.
    train_buckets = _bin_bucket_pairs(pairs)
    bucket_keys = list(train_buckets.keys())
    bucket_sizes = np.array([len(train_buckets[k]) for k in bucket_keys])
    bucket_weights = bucket_sizes / bucket_sizes.sum()
    log_fn(f'svi_bw_tkf92: pre-bucketed {n_actual} pairs into '
            f'{len(bucket_keys)} unique (Lx_pad, Ly_pad) cells '
            f'(largest {int(bucket_sizes.max())} pairs, smallest '
            f'{int(bucket_sizes.min())}).')

    # Build a "flat" index over the bucketed pairs so the
    # non-bin-bucketed sampling path can sample over the whole corpus
    # then group results by bucket.  flat_pairs[i] = (bucket_key,
    # local_idx) for the i-th pair after bucketing (order: bucket-by-
    # bucket).  This isn't quite the same as the original corpus order
    # but for IID `rng.choice` sampling the order is irrelevant.
    flat_pairs = []  # (bucket_key, local_idx)
    for key in bucket_keys:
        for j in range(len(train_buckets[key])):
            flat_pairs.append((key, j))
    n_flat = len(flat_pairs)
    assert n_flat == n_actual, "pair counting mismatch in bucketing"

    # ----- Pre-warm: run one batched E-step per bucket so any OOM hits up-front -----
    if pre_warm:
        log_fn(f'svi_bw_tkf92: pre-warming JIT cache for {len(bucket_keys)} '
                f'(Lx_pad, Ly_pad) shapes ...')
        t_warm = time.time()
        for key in bucket_keys:
            sub = train_buckets[key][:1]
            xs, ys, ts, Lx, Ly = _stack_bucket(sub)
            _ = estep_batch_tkf92(
                xs, ys, ts, Lx, Ly, lam, mu, ext, Q, pi)
        log_fn(f'  pre-warm complete in {time.time()-t_warm:.1f}s.')

    # ----- Val LL + early-stop state -----
    best_val_ll = -float('inf')
    best_params = (lam, mu, ext)
    val_no_improve = 0

    # Pre-bucket val pairs into chunks of size `batch_size` matching the
    # training JIT shapes.  (Mirror tkf92_adam_fb pattern.)
    val_chunks = None
    if val_pairs:
        val_buckets = _bin_bucket_pairs(val_pairs)
        val_chunks = []  # list of (xs, ys, ts, Lx, Ly, n_real)
        for lst in val_buckets.values():
            n = len(lst)
            for i in range(0, n, batch_size):
                chunk = lst[i:i + batch_size]
                n_real = len(chunk)
                if n_real < batch_size:
                    chunk = chunk + [chunk[-1]] * (batch_size - n_real)
                xs, ys, ts, Lx, Ly = _stack_bucket(chunk)
                val_chunks.append((xs, ys, ts, Lx, Ly, n_real))
        n_val_total = len(val_pairs)
        log_fn(f'svi_bw_tkf92: pre-bucketed {n_val_total} val pairs into '
                f'{len(val_chunks)} chunks of size {batch_size}.')

    scale = float(n_total_pairs) / float(batch_size)
    t0 = time.time()
    for k in range(n_iter):
        eta_k = (svi_tau + k) ** (-svi_kappa)
        if bin_bucketed:
            # Pick one bucket per iter (weighted by bucket size), then
            # sample batch_size pairs WITHIN that bucket.  All pairs in
            # the minibatch share one shape — one batched call.
            bk = int(rng.choice(len(bucket_keys), p=bucket_weights))
            pool = train_buckets[bucket_keys[bk]]
            replace = len(pool) < batch_size
            ix = rng.choice(len(pool), batch_size, replace=replace)
            sub_buckets = {bucket_keys[bk]: [pool[i] for i in ix]}
        else:
            # Plain IID sample over the whole corpus, then group by
            # bucket — one batched call per distinct shape in the
            # minibatch (most minibatches hit a handful of shapes,
            # which is still far cheaper than a 50-pair Python loop).
            ix = rng.choice(n_flat, batch_size, replace=False)
            sub_buckets = {}
            for i in ix:
                key, local_idx = flat_pairs[i]
                sub_buckets.setdefault(key, []).append(
                    train_buckets[key][local_idx])

        suff_mb = _empty_suff()
        ll_mb = 0.0
        for key, sub in sub_buckets.items():
            x_b, y_b, t_b, Lx_b, Ly_b = _stack_bucket(sub)
            batch_out = estep_batch_tkf92(
                x_b, y_b, t_b, Lx_b, Ly_b, lam, mu, ext, Q, pi)
            agg = _aggregate_batch_suff(batch_out)
            for kk in suff_mb:
                suff_mb[kk] += agg[kk]
            ll_mb += float(jnp.sum(batch_out['log_p']))

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

        # ----- Val LL + early stopping -----
        if val_chunks and val_every and ((k + 1) % val_every == 0
                                          or k == n_iter - 1):
            val_ll = 0.0
            for xs, ys, ts, Lx, Ly, n_real in val_chunks:
                log_ps = estep_batch_tkf92_forward_only(
                    xs, ys, ts, Lx, Ly, lam, mu, ext, Q, pi)
                # Only the first n_real entries are real; the rest are
                # padding-with-replacement repeats whose log_p we exclude.
                val_ll += float(jnp.sum(log_ps[:n_real]))
            val_per_pair = val_ll / max(len(val_pairs), 1)
            history[-1]['val_ll_total'] = val_ll
            history[-1]['val_ll_per_pair'] = val_per_pair
            log_fn(f'    val iter {k+1}: val_ll/pair={val_per_pair:.2f} '
                    f'(n_val={len(val_pairs)})')
            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_params = (lam, mu, ext)
                val_no_improve = 0
            else:
                val_no_improve += 1
                if patience and val_no_improve >= patience:
                    log_fn(f'  early stop at iter {k+1}: val LL no improvement '
                            f'for {val_no_improve} val checks '
                            f'(patience={patience}).')
                    break

    out = {'lam': lam, 'mu': mu, 'ext': ext, 'history': history}
    if val_pairs:
        out['best_lam'], out['best_mu'], out['best_ext'] = best_params
        out['best_val_ll_total'] = best_val_ll
        out['best_val_ll_per_pair'] = (
            best_val_ll / max(len(val_pairs), 1)
            if best_val_ll > -float('inf') else float('nan'))
    return out
