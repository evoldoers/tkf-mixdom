"""Hypergeometric gap-length log-probability under the Pair HMM.

Implements `gapprob_{X,Y}(i, j)` from tkf/body-tkf92.tex § "Gap
probabilities summed over indel orderings".  For an alignment state
path X → (i deletes and j inserts in ANY order) → Y, the marginal
over orderings is given by hypergeometric sums.

We work in numpy (no autograd needed for the easy eval).  The
hypergeometric arguments are non-positive integers, so each 2F1 is a
finite polynomial of degree min(i-1, j-1) or similar — direct sum is
fast.

Convention: state types are S=0, M=1, I=2, D=3, E=4 (matches
tkfmixdom.jax.core.params).  τ is the 5×5 TKF92 transition matrix.
The hypergeometric formula uses:

  a = τ[X, Y]   b = τ[X, I]   c = τ[X, D]
  f = τ[I, Y]   g = τ[I, I]   h = τ[I, D]
  p = τ[D, Y]   q = τ[D, I]   r = τ[D, D]

  z = h q / (g r)
"""
from __future__ import annotations

import math

import numpy as np

S, M, I, D, E = 0, 1, 2, 3, 4
NEG_INF = -1e30


def _hyp2f1_polynomial(a, b, c, z):
    """2F1(a, b; c; z) when at least one of a, b is a non-positive integer
    (so the series terminates).  Computed as a direct finite sum:

        2F1(a, b; c; z) = Σ_{k=0}^{K} (a)_k (b)_k / [(c)_k k!] · z^k

    K is set to the smaller of `-a` (if a≤0) and `-b` (if b≤0).
    """
    K = None
    if isinstance(a, int) and a <= 0:
        K = -a
    if isinstance(b, int) and b <= 0:
        if K is None or -b < K:
            K = -b
    if K is None:
        # No termination — shouldn't happen in our use here.
        raise ValueError(f"hyp2f1({a}, {b}; {c}; {z}) does not terminate "
                          "(needs a or b a non-positive integer)")

    total = 1.0
    term = 1.0
    for k in range(1, K + 1):
        # (a)_k / (a)_{k-1} = a + k - 1, similarly for (b), (c)
        term *= ((a + k - 1) * (b + k - 1)) / ((c + k - 1) * k) * z
        total += term
    return total


def gap_logprob(X, Y, i, j, tau):
    """log P(gap of i I-states and j D-states connecting X to Y | tau).

    All four ordering cases:
      (0,0): direct transition X→Y         = τ[X, Y]
      (i,0): X→D, D→D × (i-1), D→Y         = c · r^{i-1} · p
      (0,j): X→I, I→I × (j-1), I→Y         = b · g^{j-1} · f
      (i,j) both ≥ 1: hypergeometric sum   (body-tkf92.tex eq sec:tkf92-gapprob)

    Returns -inf if the gap probability is zero or unrepresentable
    (e.g. transition matrix entry zero on a required edge).
    """
    a = tau[X, Y]; b = tau[X, I]; c = tau[X, D]
    f = tau[I, Y]; g = tau[I, I]; h = tau[I, D]
    p = tau[D, Y]; q = tau[D, I]; r = tau[D, D]

    if i == 0 and j == 0:
        return _safe_log(a)
    if j == 0:  # only deletes
        return _safe_log(c) + (i - 1) * _safe_log(r) + _safe_log(p)
    if i == 0:  # only inserts
        return _safe_log(b) + (j - 1) * _safe_log(g) + _safe_log(f)

    # Both i ≥ 1 and j ≥ 1.  Hypergeometric sum.
    if g <= 0 or r <= 0:
        return NEG_INF
    z = (h * q) / (g * r) if (g * r) > 0 else 0.0

    # Three hypergeometric terms.  brf branch (i deletes mostly, j inserts mostly):
    F1 = _hyp2f1_polynomial(1 - i, 2 - j, 2, z) if (i >= 1 and j >= 1) else 0.0
    F2 = _hyp2f1_polynomial(2 - i, 1 - j, 2, z) if (i >= 1 and j >= 1) else 0.0
    F3 = _hyp2f1_polynomial(1 - i, 1 - j, 1, z)

    # Coefficients
    term1 = z * (j - 1) * b * r * f * F1
    term2 = z * (i - 1) * c * g * p * F2
    term3 = (b * h * p + c * q * f) * F3
    gp = (g ** (j - 1)) * (r ** (i - 1)) * (term1 + term2 + term3)
    return _safe_log(gp)


def _safe_log(x):
    if x <= 0:
        return NEG_INF
    return math.log(x)


def alignment_to_gap_list(state_seq):
    """Convert a state sequence (list of 0..4 ints, with S=0 first and E=4 last)
    into a list of (X, i, j, Y) tuples — one per gap between consecutive
    "anchor" states (S, M, or E).

    Anchor states are S, M, E.  Inserts (I) and deletes (D) accumulate
    until the next anchor.
    """
    gaps = []
    anchor_X = None  # last anchor state index
    i_count = 0      # accumulated I's since last anchor
    j_count = 0      # accumulated D's since last anchor (j = D, by convention)
    # NOTE: the body-tkf92 formula uses i = #D (the row), j = #I (the col).
    # We re-check on the call site.  For now: i_count = # D, j_count = # I.

    for s in state_seq:
        if s == I:
            j_count += 1  # j = number of inserts
        elif s == D:
            i_count += 1  # i = number of deletes
        elif s in (S, M, E):
            if anchor_X is not None:
                gaps.append((anchor_X, i_count, j_count, s))
            anchor_X = s
            i_count = 0
            j_count = 0
        # else: skip
    return gaps


def per_pair_gap_loglike(state_seq, tau):
    """log P(state_seq | tau) using the GAP-MARGINALIZED hypergeometric
    decomposition — sums over I/D orderings within each gap.

    Compare to the simpler SMIDE-path LL Σ log τ_{s_i, s_{i+1}} which
    keeps the ordering.
    """
    gaps = alignment_to_gap_list(state_seq)
    return sum(gap_logprob(X, Y, i, j, tau) for X, i, j, Y in gaps)


def tkf92_trans_np(lam, mu, t, ext):
    """Pure-numpy TKF92 transition matrix, matching params.tkf92_trans
    but without JAX overhead (for fast batch eval over many pairs).

    Includes the L'Hôpital limit for |1-κ| < 1e-4.
    """
    s = mu * t
    alpha = math.exp(-s)
    kappa = lam / max(mu, 1e-30)
    eps = abs(1.0 - kappa)
    if eps < 1e-4:
        # L'Hôpital limits
        beta = s / (1.0 + s)
        Phi = 1.0 - alpha
        phi = (1.0 - s / 2.0) if s < 1e-10 else (Phi / max(s, 1e-30))
        gamma = 1.0 - 1.0 / ((1.0 + s) * phi)
    else:
        eta = math.exp(-lam * t)
        denom = mu * eta - lam * alpha
        beta = (lam * (eta - alpha)) / denom if abs(denom) > 1e-30 else 0.0
        Phi = 1.0 - alpha
        gamma = 1.0 - mu * beta / (lam * max(Phi, 1e-30))

    one_m_kappa = max(0.0, (mu - lam) / max(mu, 1e-30))

    # Build tau91 row-by-row, then extend to TKF92 with ext self-loops.
    tau91 = np.zeros((5, 5))
    for src in (S, M, I):
        tau91[src, M] = (1 - beta) * kappa * alpha
        tau91[src, I] = beta
        tau91[src, D] = (1 - beta) * kappa * (1 - alpha)
        tau91[src, E] = (1 - beta) * one_m_kappa
    tau91[D, M] = (1 - gamma) * kappa * alpha
    tau91[D, I] = gamma
    tau91[D, D] = (1 - gamma) * kappa * (1 - alpha)
    tau91[D, E] = (1 - gamma) * one_m_kappa

    tau = tau91.copy()
    for src in (M, I, D):
        tau[src] = (1 - ext) * tau91[src]
        tau[src, src] = ext + (1 - ext) * tau91[src, src]
    return tau


def ggi_flow_at_t_np(lam0, mu0, x, y, t):
    """Pure-numpy GGI flow — FIXED-RATE form (paper's recommended
    closed-form GGI -> TKF92 surrogate).  Per wideboy_to_lambda.md
    2026-06-03; previously used slaved-rate form (lam_t = lam0/(1-r_t))."""
    num = lam0 * y * (1 - x) + mu0 * x * (1 - y)
    den = lam0 * (1 - y) + mu0 * (1 - x)
    r_b = num / max(den, 1e-30)
    r_inf = r_b / (2 - r_b)
    k = (lam0 + mu0) * (2 - r_b) / max(1 - r_b, 1e-30)
    r_t = r_inf + (r_b - r_inf) * math.exp(-k * t)
    one_m_r0 = max(1 - r_b, 1e-30)
    lam_t = lam0 / one_m_r0
    mu_t = mu0 / one_m_r0
    return lam_t, mu_t, r_t
