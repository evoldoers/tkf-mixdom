"""Posterior pseudocounts for SVI Baum-Welch.

Implements the α̃-space representation of SVB state described in
``tkf/svb-convergence.tex`` §Pseudocount representation and its advantages.
In this view, the per-iteration EMA update operates directly on posterior
pseudocounts

    α̃_k = α_prior + EMA_k(N/|B| · s_batch),

i.e.

    α̃_k = (1 − η_k) α̃_{k−1} + η_k (α_prior + scale · s_batch).

This is equivalent to the "data-only EMA + prior-at-M-step" formulation at
every iteration (see ``tests/test_svi_pseudocount_equivalence.py``), but
exposes α̃ directly so that:

  * the complete-data log joint J(θ | α̃) = α̃·log θ − Z(θ) is a single
    pytree readout (used by the M-step monotonicity tests);
  * per-group effective sample size ESS_K is available as a first-class
    quantity (used by the convergence diagnostics instrumentation in
    train_pfam.py); and
  * the M-step reduces to one call per parameter group with no
    re-addition of priors.

This module does NOT replace ``train_pfam.svi_bw``'s EMA machinery — it
adds a layered representation on top of ``svi_running_stats`` (which stays
on disk unchanged for backward compat with all existing Pfam checkpoints).
Conversion is symbolic: α̃ = running + prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Prior construction from CLI args
# ---------------------------------------------------------------------------

def build_prior_pseudocounts(
    args: Any,
    n_dom: int,
    n_frag: int,
    n_classes: int = 1,
    AA: int = 20,
) -> dict:
    """Build a pseudocount dict with the same keys as ``svi_running_stats``
    populated from CLI prior hyperparameters.

    The prior contribution to each key matches ``_log_prior`` in
    train_pfam.py (which is itself aligned with the M-step pseudocount
    additions). Any key whose M-step is not a simple conjugate-Dirichlet /
    conjugate-Gamma formula (substitution mixture) is set to zero here;
    those parameters have their priors handled directly in the M-step.

    Keys produced (mirroring ``svi_running_stats``):
      * ``top_5x5`` — (5,5) zeros (Gamma prior on λ, μ handled via
        ``top_prior_gamma`` returned alongside).
      * ``dom_M_5x5`` — list of (5,5) zeros × n_dom.
      * ``dom_kappa`` — (n_dom,) zeros.
      * ``dom_1mkappa`` — (n_dom,) zeros.
      * ``ext`` — (n_dom, n_frag, n_frag) with value (α_ext − 1).
      * ``term`` — (n_dom, n_frag) with value (β_ext − 1).
      * ``dom_w`` — (n_dom,) with value (α_dom − 1).
      * ``frag_w`` — list of (n_frag,) with value (α_frag − 1).
      * ``classdist_counts`` — (n_dom, n_frag, n_classes) with value
        (α_classdist − 1), if n_classes > 1.

    Returns:
        dict with keys matching svi_running_stats. The Gamma-prior
        components for indel rates do NOT fit this pseudocount form
        cleanly (they contribute to B, D, S separately); use
        ``indel_prior_pseudocounts`` for those.
    """
    prior: dict = {}

    # Dirichlet prior on dom_weights (symmetric α_dom):
    alpha_dom = float(args.dom_dirichlet)
    prior['dom_w'] = np.full(n_dom, alpha_dom - 1.0, dtype=float)

    # Dirichlet prior on frag_weights (symmetric α_frag per domain):
    alpha_frag = float(args.frag_dirichlet)
    prior['frag_w'] = [np.full(n_frag, alpha_frag - 1.0, dtype=float)
                       for _ in range(n_dom)]

    # Dirichlet prior on ext rows:
    #   transition cells use α_ext, termination uses β_ext.
    alpha_ext = float(args.ext_alpha)
    beta_ext = float(args.ext_beta)
    prior['ext'] = np.full((n_dom, n_frag, n_frag),
                           alpha_ext - 1.0, dtype=float)
    prior['term'] = np.full((n_dom, n_frag), beta_ext - 1.0, dtype=float)

    # BDI suff stats: the Gamma prior doesn't add to (B, D, S) directly as
    # a pseudocount shift. Emit zeros here; the BDI-specific prior is
    # returned by indel_prior_pseudocounts().
    prior['top_5x5'] = np.zeros((5, 5), dtype=float)
    prior['dom_M_5x5'] = [np.zeros((5, 5), dtype=float) for _ in range(n_dom)]
    prior['dom_kappa'] = np.zeros(n_dom, dtype=float)
    prior['dom_1mkappa'] = np.zeros(n_dom, dtype=float)

    # Classdist: symmetric Dirichlet(α_classdist) per (d, f)
    if n_classes > 1:
        alpha_cd = float(getattr(args, 'classdist_dirichlet', 1.0))
        prior['classdist_counts'] = np.full(
            (n_dom, n_frag, n_classes), alpha_cd - 1.0, dtype=float)

    return prior


@dataclass
class IndelPrior:
    """Gamma prior on a single (λ, μ) pair: Gamma(α_λ, β), Gamma(α_μ, β)."""
    alpha_lam: float
    alpha_mu: float
    beta: float

    def as_bdi_shift(self) -> tuple[float, float, float]:
        """Return (ΔB, ΔD, ΔS) that when added to data (B, D, S) gives
        the augmentations used by ``m_step_indel_quadratic``:
            B_aug = B + α_λ − 1 = (B + α_λ) − 1 → α̃_B = B + α_λ, so ΔB = α_λ.
            D_aug = D + α_μ − 1 = (D + α_μ) − 1 → α̃_D = D + α_μ, so ΔD = α_μ.
            S_aug = S + β                      → α̃_S = S + β,   so ΔS = β.
        """
        return (self.alpha_lam, self.alpha_mu, self.beta)


def indel_prior_pseudocounts(args: Any) -> IndelPrior:
    """Build Gamma-prior pseudocounts from CLI args.ins_prior, args.del_prior.

    Assumes args.ins_prior = (α_λ, β) and args.del_prior = (α_μ, β') with
    β = β' (consistent with ``m_step_indel_quadratic``'s single
    ``prior_beta`` parameter).
    """
    a_ins, b_ins = args.ins_prior
    a_del, b_del = args.del_prior
    # m_step_indel_quadratic uses prior_beta from ins_prior only; warn if
    # del_prior's beta differs.
    if abs(float(b_ins) - float(b_del)) > 1e-9:
        import warnings
        warnings.warn(
            f"ins_prior rate {b_ins} != del_prior rate {b_del}; "
            f"using ins_prior rate in BDI M-step.")
    return IndelPrior(alpha_lam=float(a_ins),
                       alpha_mu=float(a_del),
                       beta=float(b_ins))


# ---------------------------------------------------------------------------
# Lift / lower between data-running-stats and posterior-pseudocount reps
# ---------------------------------------------------------------------------

def lift_to_pseudocount(running_stats: dict, prior: dict) -> dict:
    """Compute α̃ = running_stats + prior, elementwise across matching keys.

    Keys present in only one side are passed through untouched; this is
    important because ``running_stats`` may contain keys (e.g.
    ``class_match_counts``) for which we don't define a prior pseudocount
    here — those are left alone for the non-conjugate M-step to handle.
    """
    out: dict = {}
    for k, v in running_stats.items():
        p = prior.get(k)
        out[k] = _add_like(v, p) if p is not None else _copy(v)
    # Keys in prior but not in running_stats are effectively "data = 0":
    for k in prior:
        if k not in out:
            out[k] = _copy(prior[k])
    return out


def lower_to_running(alpha_tilde: dict, prior: dict) -> dict:
    """Inverse of lift_to_pseudocount: data = α̃ − prior."""
    out: dict = {}
    for k, v in alpha_tilde.items():
        p = prior.get(k)
        out[k] = _sub_like(v, p) if p is not None else _copy(v)
    return out


def _copy(v):
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, list):
        return [_copy(x) for x in v]
    return v


def _add_like(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [_add_like(x, y) for x, y in zip(a, b)]
    return np.asarray(a, dtype=float) + np.asarray(b, dtype=float)


def _sub_like(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [_sub_like(x, y) for x, y in zip(a, b)]
    return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)


# ---------------------------------------------------------------------------
# Pseudocount-space EMA update (paper's eq:svi-update form)
# ---------------------------------------------------------------------------

def ema_update_pseudocount(
    alpha_tilde_prev: dict,
    prior: dict,
    batch: dict,
    scale: float,
    eta: float,
) -> dict:
    """Paper-form EMA update directly on posterior pseudocounts:

        α̃_new = (1 − η) α̃_prev + η (α_prior + scale · batch).

    Only keys present in both ``batch`` and ``alpha_tilde_prev`` are
    updated; any key unique to ``batch`` is initialized as
    ``α_prior + scale · batch`` (first-batch convention).
    """
    out: dict = {}
    for k, batch_v in batch.items():
        p = prior.get(k, _zero_like(batch_v))
        rhs = _add_like(p, _scaled(batch_v, scale))
        if k in alpha_tilde_prev:
            out[k] = _blend(alpha_tilde_prev[k], rhs, eta)
        else:
            out[k] = rhs
    # Preserve pre-existing keys that didn't appear in this batch:
    for k, v in alpha_tilde_prev.items():
        if k not in out:
            out[k] = _copy(v)
    return out


def _zero_like(v):
    if isinstance(v, list):
        return [_zero_like(x) for x in v]
    return np.zeros_like(np.asarray(v, dtype=float))


def _scaled(v, factor):
    if isinstance(v, list):
        return [_scaled(x, factor) for x in v]
    return np.asarray(v, dtype=float) * factor


def _blend(prev, rhs, eta):
    """Compute (1 − η) prev + η rhs, elementwise."""
    if isinstance(prev, list) and isinstance(rhs, list):
        return [_blend(p, r, eta) for p, r in zip(prev, rhs)]
    return (1.0 - eta) * np.asarray(prev, dtype=float) + eta * np.asarray(rhs, dtype=float)


# ---------------------------------------------------------------------------
# Effective sample size and EMA weight history
# ---------------------------------------------------------------------------

def ema_weight_history(etas: list) -> list:
    """Compute the EMA weight history

        w_{j,K} = η_j ∏_{i=j+1..K} (1 − η_i)

    given the list of step sizes η_1, …, η_K. Returns a list of length K.
    Note that ``sum(w_{j,K}) = 1 − ∏(1 − η_i)`` which approaches 1 as K→∞
    but is strictly < 1 for finite K when all η_i < 1.
    """
    K = len(etas)
    w = np.zeros(K, dtype=float)
    for j in range(K):
        wj = etas[j]
        for i in range(j + 1, K):
            wj *= (1.0 - etas[i])
        w[j] = wj
    return list(w)


def ess_from_weights(weights: list) -> float:
    """Effective sample size of a weight vector:

        ESS = (Σ w_j)^2 / Σ w_j^2.

    For SVB's EMA with step size η_k = (k + τ)^(−κ), the ESS at iteration
    K quantifies how many batch-level "virtual observations" contribute
    non-negligibly to α̃_K.
    """
    w = np.asarray(weights, dtype=float)
    num = float(w.sum()) ** 2
    den = float((w * w).sum())
    return num / den if den > 1e-30 else 0.0


# ---------------------------------------------------------------------------
# Log joint J(θ | α̃) per parameter group (for monotonicity tests)
# ---------------------------------------------------------------------------

def log_J_dirichlet(alpha_tilde: np.ndarray, theta: np.ndarray) -> float:
    """J(θ|α̃) = Σ_i (α̃_i − 1) log θ_i, Dirichlet log-density up to const.

    Applied to a single simplex (a vector α̃ of length K and θ of length K).
    """
    a = np.asarray(alpha_tilde, dtype=float)
    t = np.maximum(np.asarray(theta, dtype=float), 1e-300)
    return float(np.sum((a - 1.0) * np.log(t)))


def log_J_gamma_pair(alpha_B: float, alpha_D: float, alpha_S: float,
                     L: float, M: float, T: float,
                     ins: float, mu: float) -> float:
    """J(λ, μ | α̃) for a BDI process with joint geometric length prior:

        J = (α̃_B − 1) log λ + (α̃_D − 1) log μ
            − λ (α̃_S + T) − μ α̃_S
            + L log κ + M log(1 − κ)

    where κ = λ/μ. This is the objective maximized by
    ``m_step_indel_quadratic`` (eq:kappa-quadratic).
    """
    import math
    ins_s = max(float(ins), 1e-300)
    mu_s = max(float(mu), 1e-300)
    kappa = ins_s / mu_s
    kappa = min(max(kappa, 1e-300), 1.0 - 1e-15)
    return float(
        (alpha_B - 1.0) * math.log(ins_s)
        + (alpha_D - 1.0) * math.log(mu_s)
        - ins_s * (alpha_S + T)
        - mu_s * alpha_S
        + L * math.log(kappa)
        + M * math.log(1.0 - kappa))
