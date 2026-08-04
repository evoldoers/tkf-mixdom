"""MixFrag summarised-count exact EM (the path-2 trainer of supplement B.6).

This is the *alignment-given* MixFrag trainer.  Unlike ``mixfrag_svi_bw`` (which
marginalises the alignment AND the fragtypes inside a per-pair 2D
forward-backward, so it must revisit every cherry on every iteration), this
module consumes the fixed alignment-summary count tensors built by
``build_mixfrag_cherry_counts`` and runs EM in time independent of the number of
cherries.  The latent fragtypes are still marginalised exactly; the trick
(supplement "MixFrag Training from Alignment-Summary Counts",
sec:mixfrag-cherrytrain) is that the fragtype-augmented transfer matrices have a
rank-one off-diagonal, so each pairwise alignment factorises -- per discretised
time bin -- into match-run factors phi_M(k) and ordering-summed gap factors
G_Y(i,j), and the corpus enters the indel likelihood only through the counts
N_match(k), N_gap->M(i,j), N_gap->E(i,j), N_start/N_end/N_empty and the
substitution counts N_sub(a,b).

Pieces (all per time bin tau, with t = the bin centre):

  * factor functions
      m(ell)        mixture-of-geometrics fragment-length marginal
                    m(ell) = sum_f w_f r_f^{ell-1} (1-r_f)            (eq:mixfrag-mell)
      phi_a(k)      run factor; GF Phi_a(z)=M(z)/(1-tau_aa M(z))      (eq:mixfrag-gf)
      G_M, G_E      gap factors; alternating I/D-run 2D recursion     (eq:mixfrag-gapprefix)
  * log-likelihood  eq:mixfrag-summary-ll (indel) + the shared-(Q,pi) substitution term.
  * E-step          the expected complete-data sufficient statistics are read off
                    the gradient of the indel log-factor by the score-function
                    identity (eq:score-identity).  Concretely we parameterise the
                    log-likelihood by the *free* TKF91 transition slots tau_ab and
                    the *split* fragtype atoms (start weight s_f = w_f, extension
                    r_f, stop q_f = 1-r_f), and JAX autodiff returns
                        n_hat_ab = d logZ / d log tau_ab    (resolved 5x5; extensions
                                                             live in r, not tau_aa)
                        F_f       = d logZ / d log r_f       (type-f extension count)
                        E_f       = d logZ / d log s_f       (type-f fragment count)
                    -- this is the count-tensor inside algorithm's expectation
                    semiring, computed automatically.
  * M-step          CLOSED FORM (project rule -- never gradient ascent): n_hat ->
                    tkf91_stats_from_counts -> (B,D,S); then the kappa-quadratic
                    m_step_indel_quadratic for (lam,mu), the Beta
                    m_step_ext_per_fragtype for r_f, the Dirichlet m_step_weights
                    for w_f.  (Q,pi) are held fixed, exactly as in svi_bw_mixfrag.

The defining correctness oracle is F=1: every factor, the log-likelihood, and the
E-step suff-stats reduce to plain TKF92 (see tests/test_mixfrag_cherry_em.py).

Public API:
  prep_bins(agg)                                  -> list of per-bin count structs
  corpus_indel_loglik(bins, lam, mu, exts, weights)         -> float
  corpus_substitution_loglik(bins, Q, pi)                   -> float
  estep(bins, lam, mu, exts, weights)             -> suff dict (B,D,S,L,M,T,F_f,E_f)
  em_fit(agg, n_fragtypes, Q, pi, ...)            -> dict (lam,mu,exts,weights,history)
"""

from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

from ..core.params import S, M, I, D, E, tkf91_trans
from ..core.ctmc import transition_matrix
from ..core.bdi import (
    tkf91_stats_from_counts_batch, m_step_indel_quadratic,
    transition_count_groups,
)
from .mixfrag_svi_bw import m_step_ext_per_fragtype, m_step_weights


# =========================================================================
# Factor functions (JAX-differentiable; the E-step autodiffs through these).
# =========================================================================


def m_full(s, r, q, Lmax):
    """Fragment-length marginal m(ell) for ell = 0..Lmax (eq:mixfrag-mell).

    m(ell) = sum_f s_f r_f^{ell-1} q_f  for ell >= 1; m(0) := 0 (unused).
    The fragtype atoms are kept SPLIT -- start weight s_f, extension r_f, stop
    q_f -- so that d/d log s_f and d/d log r_f isolate the fragment-count E_f and
    the extension-count F_f respectively.  At the physical point s=w, q=1-r.

    Returns a (Lmax+1,) array indexed by ell (index 0 = 0).
    """
    ell = jnp.arange(Lmax + 1)                    # 0..Lmax  (power of z)
    expo = jnp.maximum(ell - 1, 0)                # r exponent (ell-1), clipped
    rpow = r[None, :] ** expo[:, None]            # (Lmax+1, F)
    mf = (s[None, :] * q[None, :] * rpow).sum(axis=1)   # (Lmax+1,)
    return jnp.where(ell == 0, 0.0, mf)           # m(0) := 0 (mask, no scatter)


def _poly_mul_trunc(a, b, Lmax):
    """Truncated polynomial product: (a*b) coefficients for powers 0..Lmax.

    a[l], b[l] are coefficients of z^l; jnp.convolve gives the product whose
    n-th entry is [z^n] of a*b.
    """
    return jnp.convolve(a, b)[:Lmax + 1]


def run_factor(tau_aa, mf):
    """Run factor phi_a(k) for k = 0..Lmax (eq:mixfrag-gf).

    Phi_a(z) = M(z)/(1 - tau_aa M(z)) = sum_{p>=1} tau_aa^{p-1} M(z)^p, so
        phi_a(k) = sum_{p=1}^{k} tau_aa^{p-1} [z^k] M(z)^p,
    accumulated by repeated truncated convolution of the M-series.  mf is the
    m(ell) array from ``m_full``; returns phi indexed by k (index 0 = 0).
    """
    Lmax = mf.shape[0] - 1
    phi = mf                              # p = 1 term (tau^0 * M^1)
    cp = mf                              # running M^p coefficients
    tau_pow = 1.0
    for _p in range(2, Lmax + 1):
        tau_pow = tau_pow * tau_aa
        cp = _poly_mul_trunc(cp, mf, Lmax)
        phi = phi + tau_pow * cp
    return phi


def run_factor_scan(tau_aa, s, r, q, Lmax):
    """Run factor phi_a(k) for k=0..Lmax via the fragtype transfer matrix -- the
    fast, compile-cheap equivalent of ``run_factor`` (verified identical, see
    tests/test_mixfrag_cherry_em.py::test_run_factor_scan_matches_powsum).

    A match-run at residue a is a chain of fragments; tracking the fragtype f of
    the current column's fragment gives a per-column transfer

        T_{gf} = r_f delta_{gf} + s_g tau_aa q_f

    (continue the same type f with extension r_f, OR stop type f with q_f, cross
    a tau_aa link and redraw a new type g with start weight s_g).  Entering the
    run draws s, leaving multiplies q, so

        phi_a(k) = q^T T^{k-1} s          (k >= 1;  phi_a(0) := 0).

    The prefix powers {T^{k-1}} are obtained by an ASSOCIATIVE SCAN of matmul
    (parallel prefix, O(log Lmax) depth).  This matters twofold over the
    O(Lmax^2) truncated-convolution power sum in ``run_factor``: (a) F is tiny
    (2x2 here) so the work is O(Lmax F^3); (b) the graph is O(log Lmax) instead
    of unrolling Lmax steps -- decisive because real Pfam match-runs reach
    kmax ~ 1800, where the unrolled power sum makes XLA compile (and run) for
    minutes per bin.  Same split atoms (s=w, r, q=1-r) as ``m_full``/
    ``run_factor``, so the score-identity E-step gradients are unchanged.
    Returns phi indexed by k (index 0 = 0), length Lmax+1.
    """
    F = s.shape[0]
    T = jnp.diag(r) + tau_aa * jnp.outer(s, q)         # T_{gf} = r_f d_gf + s_g tau q_f
    eye = jnp.eye(F, dtype=T.dtype)
    if Lmax <= 0:
        return jnp.zeros((1,), dtype=T.dtype)          # only phi(0) := 0
    if Lmax == 1:
        mats = eye[None]                               # T^0 only
    else:
        Ts = jnp.broadcast_to(T, (Lmax - 1, F, F))
        powers = jax.lax.associative_scan(jnp.matmul, Ts)   # powers[i] = T^{i+1}
        mats = jnp.concatenate([eye[None], powers], axis=0)  # T^0 .. T^{Lmax-1}
    phi_pos = jnp.einsum('g,kgf,f->k', q, mats, s)     # phi(k), k = 1..Lmax
    return jnp.concatenate([jnp.zeros((1,), dtype=phi_pos.dtype), phi_pos])


def _causal_toeplitz(phi):
    """(n,n) upper-triangular Toeplitz T[m, c] = phi[c-m] (phi[0]=0, so the
    diagonal is 0 and only strictly-causal taps appear).  Built with a gather +
    mask (no scatter), so XLA-CPU codegen stays on the matmul/select path.
    A @ T convolves each row of A with phi along axis 1; T.T @ A along axis 0."""
    n = phi.shape[0]
    idx = jnp.arange(n)
    diff = idx[None, :] - idx[:, None]                  # c - m
    return jnp.where(diff >= 0, phi[jnp.clip(diff, 0, n - 1)], 0.0)


def gap_factors(tau, phi_I, phi_D, i_max, j_max):
    """Ordering-summed gap factors G_M(i,j), G_E(i,j) (eq:mixfrag-gapgf).

    A gap entered from a match (== start, by the TKF91 row coincidence) is a
    maximal block of alternating I- and D-runs.  a_I(i,j) / a_D(i,j) are the
    summed weight of gap prefixes with i deletions and j insertions ending in an
    I- / D-run (the coefficients of P_I, P_D in eq:mixfrag-gapprefix):

        a_I = base_I + tau_DI * conv_j(phi_I, a_D)
        a_D = base_D + tau_ID * conv_i(phi_D, a_I)

    with base_I[0,j] = tau_MI phi_I(j) (a single I-run requires i=0 since runs
    alternate) and base_D[i,0] = tau_MD phi_D(i).  Both convolutions strictly
    lower the total degree i+j (phi has no k=0 term), so the linear system is
    triangular in i+j and Jacobi/Gauss-Seidel iteration is EXACT after
    i_max+j_max sweeps (the max number of alternating runs).  Exits:
        G_M(i,j) = a_I tau_IM + a_D tau_DM,   G_E(i,j) = a_I tau_IE + a_D tau_DE.

    phi_I is indexed 0..j_max, phi_D 0..i_max.  Returns (G_M, G_E), each an
    (i_max+1, j_max+1) array (the (0,0) cell is 0 and never referenced).
    """
    tMI, tMD = tau[M, I], tau[M, D]
    tID, tDI = tau[I, D], tau[D, I]
    tIM, tDM = tau[I, M], tau[D, M]
    tIE, tDE = tau[I, E], tau[D, E]

    # base_I[0, j] = tMI phi_I(j); base_D[i, 0] = tMD phi_D(i) -- via broadcast
    # masks (no scatter).
    row0 = (jnp.arange(i_max + 1) == 0).astype(phi_I.dtype)[:, None]
    col0 = (jnp.arange(j_max + 1) == 0).astype(phi_D.dtype)[None, :]
    base_I = row0 * (tMI * phi_I)[None, :]
    base_D = col0 * (tMD * phi_D)[:, None]

    # Causal-convolution Toeplitz operators: conv_j(phi_I, A) = A @ T_I (axis 1),
    # conv_i(phi_D, A) = T_D.T @ A (axis 0).
    T_I = _causal_toeplitz(phi_I)
    T_D = _causal_toeplitz(phi_D)

    # The prefix recursion lowers total degree i+j every step, so it is exact
    # after i_max+j_max sweeps.  Run them as a lax.scan (static length) rather
    # than a Python-unrolled loop: the compiled graph is O(1) in i_max+j_max
    # (one scan body) instead of unrolling up to 2*max_gap matmuls per bin
    # shape -- this is what dominates startup compile.  Reverse-mode
    # differentiable, numerically identical to the unrolled loop.
    def _sweep(carry, _):
        aI, aD = carry
        aI = base_I + tDI * (aD @ T_I)
        aD = base_D + tID * (T_D.T @ aI)
        return (aI, aD), None

    (aI, aD), _ = jax.lax.scan(_sweep, (base_I, base_D), None,
                               length=i_max + j_max)

    G_M = aI * tIM + aD * tDM
    G_E = aI * tIE + aD * tDE
    return G_M, G_E


def _xlogy(count, factor):
    """count * log(factor), defined as 0 where count == 0 (and grad-safe).

    Uses the double-where trick so JAX never differentiates log at a masked
    (count == 0) entry where ``factor`` may legitimately be 0.
    """
    safe = jnp.where(count > 0, factor, 1.0)
    return jnp.where(count > 0, count * jnp.log(safe), 0.0)


# =========================================================================
# Per-bin count preparation (sparse dicts -> dense per-bin arrays, once).
# =========================================================================


def prep_bins(agg, max_gap=None, log_fn=None):
    """Reorganise the aggregated count tensors into a per-bin list.

    ``agg`` is the dict returned by
    ``build_mixfrag_cherry_counts.aggregate_counts`` (or ``load_family_counts``).
    Each returned struct holds the (dense, numpy) count arrays for one
    non-empty tau bin plus the bin time t:

        t                 bin centre (discretised evolutionary time)
        Nmatch  (kmax+1,) N_match(k), index k (k=0 unused)
        NgapM, NgapE      (imax+1, jmax+1) N_gap->M / N_gap->E (i=del, j=ins)
        Nstart,Nend,Nempty  scalars
        Nsub    (A, A)    substitution match counts
        ncher             # cherries in the bin (= Nend + #gaps->E + Nempty)
        kmax, imax, jmax, Lmax

    ``max_gap`` (optional): drop gap entries whose total size i+j exceeds this
    cutoff.  Such gaps are vanishingly rare on real data (i+j>50 is ~0.13% of
    Pfam-train gaps, >100 is ~0.02%) but each unrolls ``gap_factors`` to depth
    i+j, so a modest cap bounds the gap grid (and thus compile + runtime) at
    negligible likelihood cost.  Match runs are NEVER capped -- ``run_factor_scan``
    handles kmax~1800 in O(log kmax) graph.  The dropped gap mass is logged.
    """
    n_bins = int(agg["n_tau_bins"])
    centers = np.asarray(agg["tau_centers"], dtype=np.float64)
    scalars = np.asarray(agg["scalars"])
    match_counts = np.asarray(agg["match_counts"])

    def _gap_ok(i, j):
        return max_gap is None or (i + j) <= max_gap

    n_drop = n_keep = 0
    per_bin_match = [dict() for _ in range(n_bins)]
    per_bin_gm = [dict() for _ in range(n_bins)]
    per_bin_ge = [dict() for _ in range(n_bins)]
    for (b, k), v in agg["match_run"].items():
        per_bin_match[b][k] = per_bin_match[b].get(k, 0) + v
    for (b, i, j), v in agg["gap_to_match"].items():
        if _gap_ok(i, j):
            per_bin_gm[b][(i, j)] = per_bin_gm[b].get((i, j), 0) + v
            n_keep += v
        else:
            n_drop += v
    for (b, i, j), v in agg["gap_to_end"].items():
        if _gap_ok(i, j):
            per_bin_ge[b][(i, j)] = per_bin_ge[b].get((i, j), 0) + v
            n_keep += v
        else:
            n_drop += v
    if max_gap is not None and log_fn is not None and (n_keep + n_drop) > 0:
        log_fn(f"prep_bins: max_gap={max_gap}; dropped {n_drop} gaps "
               f"({100.0 * n_drop / (n_keep + n_drop):.4f}% of {n_keep + n_drop}).")

    bins = []
    for b in range(n_bins):
        n_start = int(scalars[b, 0])
        n_end = int(scalars[b, 1])
        n_empty = int(scalars[b, 2])
        mr, gm, ge = per_bin_match[b], per_bin_gm[b], per_bin_ge[b]
        n_gap_to_end = int(sum(ge.values()))
        ncher = n_end + n_gap_to_end + n_empty
        has_segments = bool(mr) or bool(gm) or bool(ge)
        if not has_segments and ncher == 0:
            continue                       # genuinely empty bin

        kmax = max(mr.keys(), default=0)
        imax = max([i for (i, _j) in gm] + [i for (i, _j) in ge], default=0)
        jmax = max([j for (_i, j) in gm] + [j for (_i, j) in ge], default=0)

        Nmatch = np.zeros(kmax + 1, np.float64)
        for k, v in mr.items():
            Nmatch[k] = v
        NgapM = np.zeros((imax + 1, jmax + 1), np.float64)
        NgapE = np.zeros((imax + 1, jmax + 1), np.float64)
        for (i, j), v in gm.items():
            NgapM[i, j] = v
        for (i, j), v in ge.items():
            NgapE[i, j] = v

        bins.append({
            "t": float(centers[b]),
            "Nmatch": Nmatch, "NgapM": NgapM, "NgapE": NgapE,
            "Nstart": n_start, "Nend": n_end, "Nempty": n_empty,
            "Nsub": match_counts[b].astype(np.float64),
            "ncher": ncher,
            "kmax": kmax, "imax": imax, "jmax": jmax,
            "Lmax": max(kmax, imax, jmax),
        })
    return bins


# =========================================================================
# Indel log-likelihood (eq:mixfrag-summary-ll) and its E-step gradients.
# =========================================================================


def _bin_indel_logZ(tau, s, r, q, Nmatch, NgapM, NgapE, Nstart, Nend, Nempty):
    """Indel log-likelihood of one bin as a function of the free atoms
    (tau 5x5, s, r, q), with the bin's count tensors as arguments.

    The grid sizes kmax/imax/jmax come from the (static, under jit) count-array
    shapes, so one compiled version serves every bin of a given shape.  The match
    run factor uses ``run_factor_scan`` (associative-scan matrix powers, O(log
    kmax) graph -- kmax reaches ~1800 on real data); the bounded gap grid
    (capped by ``prep_bins(max_gap=...)``) keeps ``gap_factors`` small.  This is
    the function the E-step autodiffs (gradient gives the expected counts,
    eq:score-identity).
    """
    kmax = Nmatch.shape[0] - 1
    imax = NgapM.shape[0] - 1
    jmax = NgapM.shape[1] - 1
    z = 0.0

    if kmax >= 1:
        phiM = run_factor_scan(tau[M, M], s, r, q, kmax)
        z = z + jnp.sum(_xlogy(Nmatch, phiM))

    if imax >= 1 or jmax >= 1:
        phiI = run_factor_scan(tau[I, I], s, r, q, jmax)
        phiD = run_factor_scan(tau[D, D], s, r, q, imax)
        G_M, G_E = gap_factors(tau, phiI, phiD, imax, jmax)
        z = z + jnp.sum(_xlogy(NgapM, G_M))
        z = z + jnp.sum(_xlogy(NgapE, G_E))

    # Nstart/Nend/Nempty multiply transition logs; _xlogy keeps a zero count at
    # exactly 0 even if its transition probability is 0 (e.g. tau[.,E]=0 at the
    # degenerate kappa=1), avoiding 0*log(0)=NaN.  (em_fit keeps kappa<1 anyway.)
    z = z + _xlogy(Nstart, tau[S, M])
    z = z + _xlogy(Nend, tau[M, E])
    z = z + _xlogy(Nempty, tau[S, E])
    return z


_BIN_VALGRAD = jax.jit(jax.value_and_grad(_bin_indel_logZ, argnums=(0, 1, 2, 3)))


def _bin_args(bn):
    """The (Nmatch, NgapM, NgapE, Nstart, Nend, Nempty) jnp arguments for a
    prep_bins struct."""
    return (jnp.asarray(bn["Nmatch"]), jnp.asarray(bn["NgapM"]),
            jnp.asarray(bn["NgapE"]),
            jnp.asarray(float(bn["Nstart"])), jnp.asarray(float(bn["Nend"])),
            jnp.asarray(float(bn["Nempty"])))


def corpus_indel_loglik(bins, lam, mu, exts, weights):
    """Total indel log-likelihood (eq:mixfrag-summary-ll, indel terms) summed
    over bins, at parameters (lam, mu, exts, weights)."""
    exts = np.asarray(exts, np.float64)
    weights = np.asarray(weights, np.float64)
    s = jnp.asarray(weights)
    r = jnp.asarray(exts)
    qv = jnp.asarray(1.0 - exts)
    total = 0.0
    for bn in bins:
        tau = tkf91_trans(float(lam), float(mu), bn["t"])
        logZ, _ = _BIN_VALGRAD(tau, s, r, qv, *_bin_args(bn))
        total += float(logZ)
    return total


def corpus_substitution_loglik(bins, Q, pi):
    """Shared-substitution match-emission log-likelihood,
    sum_bin sum_ab N_sub(a,b;t) log(pi_a exp(Q t)_ab).  Independent of the indel
    parameters (Q, pi fixed), so it is a constant offset for the indel M-step but
    part of the reported / validation log-likelihood."""
    pi = np.asarray(pi, np.float64)
    log_pi = jnp.log(jnp.asarray(pi))
    total = 0.0
    for bn in bins:
        Nsub = jnp.asarray(bn["Nsub"])
        if float(np.sum(bn["Nsub"])) == 0.0:
            continue
        P = transition_matrix(jnp.asarray(Q), bn["t"])          # (A,A) exp(Qt)
        logemit = log_pi[:, None] + jnp.log(jnp.maximum(P, 1e-300))
        total += float(jnp.sum(Nsub * logemit))
    return total


def estep(bins, lam, mu, exts, weights):
    """E-step: corpus expected complete-data sufficient statistics.

    For each bin the score-function identity turns the gradient of the indel
    log-factor into expected counts: n_hat_ab (resolved 5x5 TKF91 transition
    counts, extensions already split off into r), and the per-fragtype
    extension/fragment counts F_f, E_f.  n_hat is mapped to (B,D,S) by the exact
    TKF91 BDI machinery at the bin time t (with T = t x #cherries), and to the
    kappa-quadratic coefficients (L, M).  Everything sums over bins.

    Returns a dict with scalars B,D,S,L,M,T and arrays F_f,E_f (length F), plus
    the indel log-likelihood 'loglik'.
    """
    exts = np.asarray(exts, np.float64)
    weights = np.asarray(weights, np.float64)
    F = exts.shape[0]
    s = jnp.asarray(weights)
    r = jnp.asarray(exts)
    qv = jnp.asarray(1.0 - exts)

    n_hats, t_list, T_list = [], [], []
    L_sum = M_sum = loglik = 0.0
    F_acc, E_acc = np.zeros(F), np.zeros(F)

    for bn in bins:
        tau = tkf91_trans(float(lam), float(mu), bn["t"])
        logZ, (g_tau, g_s, g_r, _g_q) = _BIN_VALGRAD(tau, s, r, qv, *_bin_args(bn))

        # Expected counts via the expectation-semiring identity
        # E[n_e] = d logZ / d log p_e = p_e * d logZ / d p_e.
        n_hat = np.asarray(tau * g_tau)                 # resolved 5x5
        F_acc += np.asarray(r * g_r)                     # extension counts
        E_acc += np.asarray(s * g_s)                     # fragment counts
        loglik += float(logZ)

        groups = transition_count_groups(n_hat)
        L_sum += float(groups["log_kappa"])             # dest in {M, D}
        M_sum += float(groups["log_1mkappa"])           # dest = E (trajectory ends)

        n_hats.append(n_hat)
        t_list.append(bn["t"])
        T_list.append(bn["t"] * bn["ncher"])

    # BDI (B,D,S) batched over bins in a single vmapped+jitted call.
    E_B, E_D, E_S = tkf91_stats_from_counts_batch(
        np.stack(n_hats), float(lam), float(mu),
        np.asarray(t_list), np.asarray(T_list))

    return {"B": float(np.sum(E_B)), "D": float(np.sum(E_D)),
            "S": float(np.sum(E_S)), "L": L_sum, "M": M_sum,
            "T": float(np.sum(T_list)),
            "F_f": F_acc, "E_f": E_acc, "loglik": loglik}


# =========================================================================
# EM driver.
# =========================================================================


def _ckpt_path(path):
    """Normalise a checkpoint path to end in .npz."""
    p = str(path)
    return p if p.endswith(".npz") else p + ".npz"


def _save_checkpoint(path, lam, mu, exts, weights, it, history, sub_ll):
    """Write a resume checkpoint atomically (.tmp.npz then os.replace)."""
    import os as _os
    final = _ckpt_path(path)
    tmp = final[:-4] + ".tmp.npz"
    np.savez(tmp, lam=np.float64(lam), mu=np.float64(mu),
             exts=np.asarray(exts, np.float64), weights=np.asarray(weights, np.float64),
             next_iter=np.int64(it), sub_ll=np.float64(sub_ll),
             hist_iter=np.array([h["iter"] for h in history], np.int64),
             hist_total_ll=np.array([h["total_ll"] for h in history], np.float64),
             hist_indel_ll=np.array([h["indel_ll"] for h in history], np.float64),
             hist_lam=np.array([h["lam"] for h in history], np.float64),
             hist_mu=np.array([h["mu"] for h in history], np.float64),
             hist_exts=np.array([h["exts"] for h in history], np.float64),
             hist_weights=np.array([h["weights"] for h in history], np.float64))
    _os.replace(tmp, final)


def _load_checkpoint(path):
    """Load a resume checkpoint written by ``_save_checkpoint``; returns
    (lam, mu, exts, weights, next_iter, history) or None if absent."""
    import os as _os
    p = _ckpt_path(path)
    if not _os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    history = [{"iter": int(i), "total_ll": float(t), "indel_ll": float(il),
                "lam": float(l), "mu": float(m), "exts": e.copy(), "weights": w.copy()}
               for i, t, il, l, m, e, w in zip(
                   z["hist_iter"], z["hist_total_ll"], z["hist_indel_ll"],
                   z["hist_lam"], z["hist_mu"], z["hist_exts"], z["hist_weights"])]
    return (float(z["lam"]), float(z["mu"]), np.asarray(z["exts"], np.float64),
            np.asarray(z["weights"], np.float64), int(z["next_iter"]), history)


def em_fit(agg, *, n_fragtypes, Q, pi,
           init_lam=0.05, init_mu=0.06, init_exts=None, init_weights=None,
           n_iter=100, tol=1e-7,
           prior_alpha_lam=2.0, prior_alpha_mu=2.0, prior_beta=10.0,
           ext_prior_alpha=2.0, ext_prior_beta=3.0, weight_prior_alpha=1.5,
           max_gap=None, checkpoint_path=None, checkpoint_every=0, resume=True,
           log_fn=print, log_every=10):
    """Exact EM on the MixFrag alignment-summary count tensors (supplement B.6).

    ``agg`` is the aggregated count dict (build_mixfrag_cherry_counts).  (Q, pi)
    are FIXED (shared substitution model, e.g. LG08), as in svi_bw_mixfrag.
    M-steps are closed form: kappa-quadratic for (lam,mu), Beta for r_f,
    Dirichlet for w_f.  Iterates to convergence of the total log-likelihood.

    ``max_gap`` caps the gap grid (see ``prep_bins``).  ``checkpoint_path`` +
    ``checkpoint_every`` (>0) write a resume checkpoint every N iters; with
    ``resume`` (default True) an existing checkpoint is loaded and EM continues
    from it.  Returns a dict with final lam, mu, exts (F,), weights (F,), and
    'history'.
    """
    F = int(n_fragtypes)
    bins = prep_bins(agg, max_gap=max_gap, log_fn=log_fn)
    if not bins:
        raise ValueError("No non-empty tau bins in the count tensors.")

    lam, mu = float(init_lam), float(init_mu)
    # TKF91 requires kappa = lam/mu < 1 (a finite expected sequence length): at
    # kappa = 1 every termination probability tau[.,E] = (1-beta)(1-kappa) is 0,
    # so the indel log-likelihood is -inf and its gradient is NaN.  The
    # kappa-quadratic M-step keeps kappa < 1 thereafter, but a kappa >= 1 INIT
    # (e.g. lam = mu) would NaN the very first E-step; nudge mu up to keep
    # kappa <= 0.9 at the start.
    if lam >= 0.9 * mu:
        mu = lam / 0.9
        log_fn(f"em_fit: nudged init mu to {mu:.5f} to keep kappa=lam/mu<1 "
               f"(model is improper at lam=mu).")
    exts = (np.linspace(0.3, 0.7, F) if F > 1 else np.array([0.5])) \
        if init_exts is None else np.asarray(init_exts, np.float64)
    weights = (np.full(F, 1.0 / F) if init_weights is None
               else np.asarray(init_weights, np.float64))
    weights = weights / weights.sum()
    assert exts.shape == (F,) and weights.shape == (F,)

    sub_ll = corpus_substitution_loglik(bins, Q, pi)   # constant offset
    n_cher = sum(bn["ncher"] for bn in bins)
    log_fn(f"em_fit: {len(bins)} bins, {n_cher} cherries, F={F}; "
           f"fixed-substitution loglik={sub_ll:.4f}")

    history = []
    prev_ll = -np.inf
    start_iter = 0
    if checkpoint_path is not None and resume:
        ck = _load_checkpoint(checkpoint_path)
        if ck is not None:
            lam, mu, exts, weights, start_iter, history = ck
            if history:
                prev_ll = history[-1]["total_ll"]
            log_fn(f"em_fit: resumed from checkpoint at iter {start_iter} "
                   f"(ll={prev_ll:.4f}).")

    t0 = time.time()
    for it in range(start_iter, n_iter):
        suff = estep(bins, lam, mu, exts, weights)
        total_ll = suff["loglik"] + sub_ll

        lam, mu = m_step_indel_quadratic(
            B=suff["B"], D=suff["D"], S=suff["S"],
            L=suff["L"], M=suff["M"], T=suff["T"],
            prior_alpha_lam=prior_alpha_lam, prior_alpha_mu=prior_alpha_mu,
            prior_beta=prior_beta)
        exts = m_step_ext_per_fragtype(
            suff["F_f"], suff["E_f"], ext_prior_alpha, ext_prior_beta)
        weights = m_step_weights(suff["E_f"], weight_prior_alpha)

        history.append({
            "iter": it + 1, "indel_ll": suff["loglik"], "total_ll": total_ll,
            "lam": lam, "mu": mu, "exts": exts.copy(), "weights": weights.copy(),
        })
        done = it + 1 - start_iter
        if (it + 1) % log_every == 0 or it == start_iter:
            el = time.time() - t0
            eta = el / done * (n_iter - (it + 1)) if done > 0 else float("nan")
            log_fn(f"  iter {it+1:>4}/{n_iter}: ll={total_ll:.4f} "
                   f"lam={lam:.5f} mu={mu:.5f} "
                   f"exts={np.array2string(exts, precision=3)} "
                   f"w={np.array2string(weights, precision=3)} "
                   f"({el:.1f}s, {el/done:.2f}s/it, ETA {eta/60:.1f}m)")
        if checkpoint_path is not None and checkpoint_every > 0 \
                and (it + 1) % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, lam, mu, exts, weights,
                             it + 1, history, sub_ll)
        if total_ll - prev_ll < tol and it > start_iter:
            log_fn(f"  converged at iter {it+1} (dll={total_ll-prev_ll:.2e}).")
            break
        prev_ll = total_ll

    return {"lam": lam, "mu": mu, "exts": np.asarray(exts),
            "weights": np.asarray(weights), "history": history,
            "final_ll": history[-1]["total_ll"] if history else float("nan"),
            "sub_ll": sub_ll}
