"""Restricted substitution M-step variants.

Implements the closed-form M-step variants documented in
`tkf/substitution-mstep.tex`, sections sec:mstep-rescaling and
sec:mstep-tied-pi:

  * Rate-rescaling (sec:mstep-rescaling): hold (S, pi) fixed up to a
    scalar multiplier sigma; sigma = U_dot / D maximises ell_2.
  * Tied-pi block (sec:mstep-tied-pi): partition C classes into
    blocks of size N_tied; classes within a block share a pooled pi
    while keeping per-class S exchangeabilities free.

Both routines preserve the closed-form, monotone-ascent character of
the standard M-step and do not require gradient-based optimisation.
They are used by `train_pfam.py` when --subst-mode selects the
corresponding restricted regime.
"""

from __future__ import annotations

import numpy as np


def m_step_subst_rescaling(W, U, S_old, pi_old,
                           sigma_prior_a=1.0, sigma_prior_b=0.0):
    """Rate-rescaling M-step.

    Holds (S_old, pi_old) fixed and returns (S_new, sigma) with
    S_new = sigma * S_old such that sigma maximises ell_2.

    Closed form (eq:rescale-sigma in substitution-mstep.tex):
        sigma_hat = (U_dot + a - 1) / (D + b)
    where U_dot = sum_{i!=j} U_ij and
          D = sum_{j>i} S_old_ij * (W_i pi_old_j + W_j pi_old_i).

    The default Gamma(a=1, b=0) is the improper uniform prior on
    [0, infinity) which reduces to plain MLE: sigma_hat = U_dot / D.

    Args:
        W: (A,) expected dwell times.
        U: (A, A) expected substitution counts (off-diagonal).
        S_old: (A, A) symmetric exchangeability with zero diagonal.
        pi_old: (A,) equilibrium distribution.
        sigma_prior_a: Gamma shape pseudocount for sigma (default 1.0
            -> plain MLE).
        sigma_prior_b: Gamma rate pseudocount for sigma (default 0).

    Returns:
        (S_new, pi_new, sigma): pi_new is just pi_old (unchanged).
    """
    W = np.asarray(W, dtype=float)
    U = np.asarray(U, dtype=float)
    S_old = np.asarray(S_old, dtype=float)
    pi_old = np.asarray(pi_old, dtype=float)
    # U_dot: total off-diagonal substitution count.
    U_dot = float(U.sum() - np.diag(U).sum())
    # D: dwell-weighted opportunity for substitution under fixed shape.
    # D = sum_{j>i} S_ij (W_i pi_j + W_j pi_i)
    #   = (1/2) sum_{i!=j} S_ij W_i pi_j   (S symmetric, swap i,j)
    Wpi = W[:, None] * pi_old[None, :]      # (A, A)
    D = 0.5 * float((S_old * (Wpi + Wpi.T)).sum())
    # MAP closed form.
    numer = U_dot + (sigma_prior_a - 1.0)
    denom = D + sigma_prior_b
    if denom < 1e-30:
        # No dwell-weighted opportunity → keep S unchanged.
        sigma = 1.0
    else:
        sigma = numer / denom
    sigma = max(sigma, 1e-30)
    S_new = sigma * S_old
    np.fill_diagonal(S_new, 0.0)
    return S_new, pi_old.copy(), float(sigma)


def m_step_subst_rescaling_pi(W, U, V, S_old, sigma_old, pi_old,
                              n_newton=15, tol=1e-12):
    """Joint rate-rescaling and equilibrium M-step.

    Holds S_old fixed (shape) and returns (S_new, pi_new, sigma) that
    jointly maximise ell_2 in (sigma, pi) for the parameterisation
    Q = sigma * S_old * diag(pi) (eq:rescale-pi-class in
    substitution-mstep.tex sec:mstep-rescaling-pi). Closed-form
    Lagrange-multiplier reduction collapses the joint problem to a
    1-D root in sigma.

    Math (substitution-mstep.tex sec:mstep-rescaling-pi):
        V'_b   = V_b + sum_{a != b} U_{ab}                  (eq above)
        N      = sum_b V'_b = V_dot + U_dot
        V_dot  = sum_b V_b   (boundary count)
        U_dot  = sum_{a != b} U_{ab}                        (jump total)
        r_b    = sum_{a != b} S_ab W_a   (= (S_old @ W)_b)
        Lagrange multiplier lambda = V_dot                  (eq:rescale-pi-lambda)
        pi_b(sigma) = V'_b / (V_dot + sigma * r_b)          (eq:rescale-pi-pi-of-sigma)
        sigma satisfies                                     (eq:rescale-pi-sigma-eq)
            g(sigma) := sum_b sigma * r_b * V'_b / (V_dot + sigma * r_b)
                       - U_dot   = 0,
            with g monotone increasing on (0, infty).

    Args:
        W: (A,) expected dwell times.
        U: (A, A) expected substitution counts (off-diagonal).
        V: (A,) expected boundary state-counts (e.g. ins+del+match-anc).
        S_old: (A, A) symmetric exchangeability with zero diagonal.
        sigma_old: previous sigma (warm-start for Newton; >= 0).
        pi_old: (A,) previous equilibrium (only used as a degenerate-case
            fallback when V_dot == 0 or U_dot == 0).
        n_newton: max log-space Newton iterations on g.
        tol: relative |g| tolerance for early stopping.

    Returns:
        (S_new, pi_new, sigma): pi_new is the joint-MLE pi; S_new = sigma * S_old.
    """
    W = np.asarray(W, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    S_old = np.asarray(S_old, dtype=float)
    pi_old = np.asarray(pi_old, dtype=float)
    A = W.shape[0]

    # Sufficient statistics
    U_dot = float(U.sum() - np.diag(U).sum())          # total off-diag jump
    U_in = U.sum(axis=0) - np.diag(U)                  # (A,) U_{. b}: jumps INTO b
    V_prime = V + U_in                                  # (A,) V'_b = V_b + U_{. b}
    V_dot = float(V.sum())                              # boundary count

    # r_b = sum_{a != b} S_ab W_a. S has zero diagonal so this is just S^T W = S W.
    r = S_old.T @ W                                     # (A,)

    # Degenerate cases
    if V_dot < 1e-30 or U_dot < 1e-30:
        # No data → keep pi at warm, sigma neutral.
        return S_old.copy(), pi_old.copy(), 1.0

    if not np.any(r > 0) or not np.any(V_prime > 0):
        # No dwell-weighted opportunity OR no boundary count; keep warm.
        return S_old.copy(), pi_old.copy(), 1.0

    # Solve g(sigma) = 0 via Newton on log-sigma. g(0) = -U_dot < 0,
    # g(infty) = V_dot > 0, g' > 0 → unique positive root.
    log_sigma = np.log(max(float(sigma_old), 1e-12))
    converged = False
    for _ in range(n_newton):
        sigma = float(np.exp(log_sigma))
        denom = V_dot + sigma * r              # (A,)
        # g and g'
        g = float(np.sum(sigma * r * V_prime / denom)) - U_dot
        # dg/d log_sigma = sigma * dg/dsigma
        # dg/dsigma = sum_b r_b * V'_b * V_dot / (V_dot + sigma r_b)^2
        dg_dsigma = float(np.sum(r * V_prime * V_dot / (denom ** 2)))
        dg_dlog = sigma * dg_dsigma
        if abs(g) < tol * max(U_dot, 1.0):
            converged = True
            break
        if dg_dlog <= 0:
            # Should not happen given strict monotonicity; bisect-style step.
            log_sigma -= np.sign(g) * 0.5
            continue
        # Newton step on log_sigma; clamp step magnitude for safety.
        step = -g / dg_dlog
        step = float(np.clip(step, -2.0, 2.0))
        log_sigma += step

    sigma = float(np.exp(log_sigma))
    if not converged:
        # Fall back to bisection in log_sigma over a wide bracket.
        lo, hi = -30.0, 30.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            sigma_mid = float(np.exp(mid))
            denom = V_dot + sigma_mid * r
            g_mid = float(np.sum(sigma_mid * r * V_prime / denom)) - U_dot
            if g_mid > 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-12:
                break
        log_sigma = 0.5 * (lo + hi)
        sigma = float(np.exp(log_sigma))

    # Reconstruct pi and S
    denom = V_dot + sigma * r
    pi_new = V_prime / denom
    pi_new = pi_new / max(pi_new.sum(), 1e-300)        # safety renorm
    S_new = sigma * S_old
    np.fill_diagonal(S_new, 0.0)
    return S_new, pi_new, sigma


def m_step_subst_tied_pi_rescaling_block(
        W_list, U_list, V_list,
        S_old_list, sigma_old_list, pi_block_old,
        n_iter=20, tol=1e-12):
    """Tied-pi-rescaling block M-step.

    For a block of classes sharing one equilibrium pi but with per-class
    rate multipliers sigma_c, jointly maximise ell_2 over
    ({sigma_c}_{c in block}, pi_block) with S_old shape frozen per class.

    Math (substitution-mstep.tex sec:mstep-tied-pi-rescaling):
        ell_block = sum_c [ V'_c log pi_block + U_dot_c log sigma_c
                            - sigma_c r_c^T pi_block ] + const
        with V'_c[b] = V_c[b] + sum_{a != b} U_c[a, b],
             r_c     = (S_old_c)^T W_c.
        Lagrange multiplier on sum(pi_block) = 1: lambda = V_dot_block
                                                  (constant; analogous
                                                  to single-class case).
        sigma_c   = U_dot_c / (r_c^T pi_block)                          (closed)
        pi_block_b = (sum_c V'_c_b) / (V_dot_block + sum_c sigma_c r_c_b)  (closed)

    Coordinate ascent alternates the two closed-form updates; each step
    is a monotone ell_2 ascent in its own coordinate, so convergence to
    the unique block-optimum is guaranteed.

    Args:
        W_list: list of (A,) per-class expected dwell times.
        U_list: list of (A, A) per-class expected substitution counts.
        V_list: list of (A,) per-class boundary state counts.
        S_old_list: list of (A, A) per-class fixed exchangeability shapes.
        sigma_old_list: list of warm-start sigmas (one per class in block).
        pi_block_old: (A,) shared block equilibrium (warm-start).
        n_iter: max coordinate-ascent rounds.
        tol: convergence tolerance on max |Delta pi_block|.

    Returns:
        (S_new_list, pi_block_new, sigma_new_list):
            S_new_list[c] = sigma_c * S_old_list[c]
            pi_block_new is the joint-optimum shared pi for the block
            sigma_new_list[c] is the joint-optimum scalar.
    """
    n_cls = len(W_list)
    pi_block = np.asarray(pi_block_old, dtype=float).copy()
    A = pi_block.shape[0]

    W = [np.asarray(W_list[c], dtype=float) for c in range(n_cls)]
    U = [np.asarray(U_list[c], dtype=float) for c in range(n_cls)]
    V = [np.asarray(V_list[c], dtype=float) for c in range(n_cls)]
    S_old = [np.asarray(S_old_list[c], dtype=float) for c in range(n_cls)]

    # Pre-compute per-class fixed sufficient stats
    U_dot = np.zeros(n_cls)
    V_prime = [np.zeros(A) for _ in range(n_cls)]
    r = [np.zeros(A) for _ in range(n_cls)]
    for c in range(n_cls):
        Up = U[c].copy()
        np.fill_diagonal(Up, 0.0)
        U_dot[c] = float(Up.sum())
        V_prime[c] = V[c] + Up.sum(axis=0)
        # r_c[b] = sum_{a != b} S_old_c[a, b] W_c[a] = (S_old_c.T @ W_c)[b]
        r[c] = S_old[c].T @ W[c]
    V_dot_block = sum(float(V[c].sum()) for c in range(n_cls))
    V_prime_block = np.sum(V_prime, axis=0)               # (A,) shared numerator

    # Degenerate handling: if no class has dwell-weighted opportunity,
    # keep warm-start values.
    if V_dot_block < 1e-30 or U_dot.sum() < 1e-30:
        sigmas = [float(s) for s in sigma_old_list]
        S_new = [sigma_old_list[c] * S_old[c] for c in range(n_cls)]
        for s in S_new:
            np.fill_diagonal(s, 0.0)
        return S_new, pi_block, sigmas

    sigma = np.array([float(s) for s in sigma_old_list], dtype=float)
    sigma = np.maximum(sigma, 1e-12)

    for _ in range(n_iter):
        pi_block_prev = pi_block.copy()

        # Step 1: per-class sigma_c = U_dot_c / (r_c . pi_block)
        for c in range(n_cls):
            denom_c = float(r[c] @ pi_block)
            if denom_c > 1e-30:
                sigma[c] = U_dot[c] / denom_c
            sigma[c] = max(sigma[c], 1e-30)

        # Step 2: pi_block_b = V'_block_b / (V_dot_block + sum_c sigma_c r_c_b)
        denom_pi = V_dot_block + np.sum(
            [sigma[c] * r[c] for c in range(n_cls)], axis=0)        # (A,)
        # Guard against any zero entries (unlikely with positive r_c and V_dot)
        denom_pi = np.maximum(denom_pi, 1e-300)
        pi_block = V_prime_block / denom_pi
        pi_block = pi_block / max(pi_block.sum(), 1e-300)

        if float(np.max(np.abs(pi_block - pi_block_prev))) < tol:
            break

    S_new = []
    for c in range(n_cls):
        S_c = sigma[c] * S_old[c]
        np.fill_diagonal(S_c, 0.0)
        S_new.append(S_c)
    return S_new, pi_block, [float(s) for s in sigma]


def m_step_subst_tied_pi_block(
        W_list, U_list, V_list,
        S_old_list, pi_block_old,
        S_prior=None, pi_prior=None,
        pi_pseudo=0.0, S_pseudo=0.0,
        n_iter=50, tol=1e-10):
    """Block tied-pi M-step.

    Run coordinate ascent that alternates per-class S updates with
    a single pooled-pi update for the block. Mirrors
    `m_step_subst_option1` from `core/ctmc.py` but with pi shared
    across classes in the block.

    Args:
        W_list: list of (A,) per-class expected dwell times.
        U_list: list of (A, A) per-class expected substitution counts.
        V_list: list of (A,) per-class character counts (match-anc +
            insert + delete in the joint pair HMM).
        S_old_list: list of (A, A) per-class exchangeabilities (used as
            warm-start for coordinate ascent).
        pi_block_old: (A,) shared block equilibrium (warm-start).
        S_prior: (A, A) optional symmetric exchangeability prior.
        pi_prior: (A,) optional equilibrium prior (for pi_pseudo).
        pi_pseudo: pseudocount weight for pi_prior.
        S_pseudo: pseudocount weight for S_prior (applied independently
            per class).
        n_iter: max coordinate-ascent iterations.
        tol: convergence tolerance on max |Delta pi_block|.

    Returns:
        (S_new_list, pi_block_new): updated per-class S and pooled pi.
    """
    n_cls = len(W_list)
    A = len(pi_block_old)
    W = [np.asarray(W_list[c], dtype=float).copy() for c in range(n_cls)]
    U = [np.asarray(U_list[c], dtype=float).copy() for c in range(n_cls)]
    V = [np.asarray(V_list[c], dtype=float).copy() for c in range(n_cls)]
    S = [np.asarray(S_old_list[c], dtype=float).copy() for c in range(n_cls)]

    # Apply pseudocounts (same prescription as m_step_subst_option1):
    # V_c <- V_c + pi_pseudo * pi_prior  (added once per class --
    # the pooled pi treats the pseudocount per class as data evidence).
    if pi_prior is not None and pi_pseudo > 0:
        pi_pr = np.asarray(pi_prior, dtype=float)
        for c in range(n_cls):
            V[c] = V[c] + pi_pseudo * pi_pr
    if S_prior is not None and pi_prior is not None and S_pseudo > 0:
        S_pr = np.asarray(S_prior, dtype=float)
        pi_pr = np.asarray(pi_prior, dtype=float)
        U_pseudo = S_pseudo * S_pr * pi_pr[None, :]
        W_pseudo = S_pseudo * np.sum(S_pr * pi_pr[None, :], axis=1)
        for c in range(n_cls):
            U[c] = U[c] + U_pseudo
            W[c] = W[c] + W_pseudo

    # V'_c[i] = V_c[i] + sum_{j!=i} U_c[j, i]
    V_prime = []
    for c in range(n_cls):
        col_sum = U[c].sum(axis=0)  # sum_j U_c[j, i]
        V_prime.append(V[c] + col_sum - np.diag(U[c]))

    # Pooled V_prime (eq:tied-pi-Vprime in tex).
    V_prime_pool = np.zeros(A)
    for vp in V_prime:
        V_prime_pool += vp

    pi_block = np.asarray(pi_block_old, dtype=float).copy()
    pi_block = pi_block / max(pi_block.sum(), 1e-30)

    U_sym = [U[c] + U[c].T for c in range(n_cls)]

    for it in range(n_iter):
        # 1. Per-class S update with current pi_block (eq:tied-pi-S-update).
        S_new = []
        for c in range(n_cls):
            denom = (W[c][:, None] * pi_block[None, :]
                     + W[c][None, :] * pi_block[:, None])
            denom = np.maximum(denom, 1e-30)
            Sc = U_sym[c] / denom
            np.fill_diagonal(Sc, 0.0)
            S_new.append(Sc)
        S = S_new

        # 2. Pooled pi update (eq:tied-pi-pi-update).
        # tilde c_i = sum_c sum_{j!=i} S_c[i, j] * W_c[j]
        c_pool = np.zeros(A)
        for c in range(n_cls):
            c_pool += (S[c] * W[c][None, :]).sum(axis=1)

        # Bisection on eta with f(eta) = sum_i V'_i / (c_i - eta) - 1 = 0.
        eta_max = c_pool.min() - 1e-10
        eta_min = eta_max - 10.0 * V_prime_pool.sum()
        for _ in range(60):
            eta_mid = 0.5 * (eta_min + eta_max)
            f_mid = float(np.sum(
                V_prime_pool / np.maximum(c_pool - eta_mid, 1e-30)) - 1.0)
            if f_mid > 0:
                eta_max = eta_mid
            else:
                eta_min = eta_mid
        eta = 0.5 * (eta_min + eta_max)

        pi_candidate = V_prime_pool / np.maximum(c_pool - eta, 1e-30)
        if (np.all(pi_candidate > 0)
                and np.isfinite(pi_candidate.sum())):
            pi_new = pi_candidate / pi_candidate.sum()
        else:
            pi_new = pi_block.copy()

        delta = float(np.max(np.abs(pi_new - pi_block)))
        pi_block = pi_new
        if delta < tol:
            break

    return S, pi_block


def banded_3fc_init(n_dom, p_ext, dtype=float):
    """Banded init for fragdist and ext_rates with n_frag=3.

    Implements the user-specified banded structure where the three
    fragchars are FragStart (0), FragMid (1), and FragEnd (2):

        fragdist[d, 0] = 1, fragdist[d, f>0] = 0  (always start at f=0)
        ext[d, 0] = [0, p^2, p(1-p)]   (term = 1-p)
        ext[d, 1] = [0, p,   1-p   ]   (term = 0)
        ext[d, 2] = [0, 0,   0     ]   (term = 1)

    so multi-character fragments always begin in FragStart and end in
    FragEnd; single-character fragments use FragStart as both endpoints.

    Args:
        n_dom: number of domains.
        p_ext: extension probability (in (0, 1)).
        dtype: numpy dtype for the returned arrays.

    Returns:
        (frag_weights, ext_rates) with shapes (n_dom, 3) and (n_dom, 3, 3).
    """
    if not (0.0 < p_ext < 1.0):
        raise ValueError(f"p_ext must be in (0, 1), got {p_ext}")
    frag_weights = np.zeros((n_dom, 3), dtype=dtype)
    frag_weights[:, 0] = 1.0
    ext = np.zeros((n_dom, 3, 3), dtype=dtype)
    p = float(p_ext)
    ext[:, 0, 1] = p * p
    ext[:, 0, 2] = p * (1.0 - p)
    ext[:, 1, 1] = p
    ext[:, 1, 2] = 1.0 - p
    # Row 2 stays all zeros; termination = 1.
    return frag_weights, ext


def banded_3fc_ext_mask():
    """Boolean masks identifying which (f, g) entries / term entries
    are structurally nonzero in the banded 3-fragchar parametrisation.

    Returns:
        (ext_mask, term_mask):
          ext_mask[f, g] = True iff ext[d, f, g] is allowed nonzero.
          term_mask[f]   = True iff term[d, f]   is allowed nonzero.
    """
    ext_mask = np.zeros((3, 3), dtype=bool)
    ext_mask[0, 1] = True
    ext_mask[0, 2] = True
    ext_mask[1, 1] = True
    ext_mask[1, 2] = True
    term_mask = np.array([True, False, True], dtype=bool)
    return ext_mask, term_mask


# --------------------------------------------------------------------
# Mode-string validation
# --------------------------------------------------------------------

VALID_SUBST_MODES = {
    'standard',
    'frozen-pi',
    'rescaling-rates',
    'rescaling-rates-and-pi',
    'tied-pi',
    'tied-pi-rescaling',
    'alt-tied-pi-rescaling',
}


def validate_subst_mode(mode, n_tied, n_classes):
    """Validate subst-mode CLI args. Raises ValueError on misuse."""
    if mode not in VALID_SUBST_MODES:
        raise ValueError(
            f"--subst-mode={mode!r} not in {sorted(VALID_SUBST_MODES)}")
    needs_tied = mode in ('tied-pi', 'tied-pi-rescaling',
                          'alt-tied-pi-rescaling')
    if needs_tied:
        if n_classes <= 1:
            raise ValueError(
                f"--subst-mode={mode} requires n_classes > 1, "
                f"got n_classes={n_classes}")
        if n_tied is None or n_tied <= 0:
            raise ValueError(
                f"--subst-mode={mode} requires --n-tied >= 1")
        if n_classes % n_tied != 0:
            raise ValueError(
                f"--subst-mode={mode} requires --n-tied to divide "
                f"n_classes; got n_tied={n_tied}, n_classes={n_classes}")


def tied_pi_blocks(n_classes, n_tied):
    """Return list of arrays of class indices in each tied-pi block."""
    n_blocks = n_classes // n_tied
    return [np.arange(b * n_tied, (b + 1) * n_tied)
            for b in range(n_blocks)]


def class_mixture_mstep(
        mode, em_iter,
        W_per_class, U_per_class, V_per_class,
        S_old, pi_old,
        S_prior, pi_prior,
        pi_pseudo, S_pseudo,
        n_tied=1,
        log_fn=None):
    """Dispatch to the appropriate per-class substitution M-step.

    Args:
        mode: one of standard / frozen-pi / rescaling-rates / tied-pi /
            alt-tied-pi-rescaling.
        em_iter: current EM iteration index (0-based). Only used by
            alt-tied-pi-rescaling: even iters use tied-pi, odd iters use
            rescaling-rates.
        W_per_class: (C, A) per-class expected dwell times.
        U_per_class: (C, A, A) per-class expected substitution counts.
        V_per_class: (C, A) per-class char counts (V = match-anc + ins + del).
        S_old: (C, A, A) current per-class exchangeabilities.
        pi_old: (C, A) current per-class equilibrium.
        S_prior: (A, A) symmetric exchangeability prior (e.g. LG).
        pi_prior: (A,) equilibrium prior (e.g. LG).
        pi_pseudo: pseudocount weight for pi_prior.
        S_pseudo: pseudocount weight for S_prior.
        n_tied: tied-pi block size; required for tied-pi /
            alt-tied-pi-rescaling.
        log_fn: optional callable for progress logging.

    Returns:
        (S_new, pi_new) with shape (C, A, A) and (C, A).
    """
    from tkfmixdom.jax.core.ctmc import m_step_subst_option1

    n_cls = S_old.shape[0]
    A = pi_old.shape[1]
    S_new = np.zeros_like(S_old)
    pi_new = np.zeros_like(pi_old)

    effective_mode = mode
    if mode == 'alt-tied-pi-rescaling':
        # Alternating: even iterations run tied-pi, odd run rescaling.
        effective_mode = 'tied-pi' if (em_iter % 2 == 0) else 'rescaling-rates'
        if log_fn is not None:
            log_fn(f"    [alt-tied-pi-rescaling] em_iter={em_iter} → "
                   f"{effective_mode} step")

    if effective_mode == 'standard':
        for cc in range(n_cls):
            S_c, pi_c, _ = m_step_subst_option1(
                W_per_class[cc], U_per_class[cc], V_per_class[cc],
                S_prior=S_prior, pi_prior=pi_prior,
                pi_pseudo=pi_pseudo, S_pseudo=S_pseudo)
            S_new[cc] = S_c
            pi_new[cc] = pi_c

    elif effective_mode == 'frozen-pi':
        # Hold pi at pi_old; only update S. Use the standard option1
        # solver to get S, but discard its pi update.
        for cc in range(n_cls):
            S_c, _pi_drop, _ = m_step_subst_option1(
                W_per_class[cc], U_per_class[cc], V_per_class[cc],
                S_prior=S_prior, pi_prior=pi_prior,
                pi_pseudo=pi_pseudo, S_pseudo=S_pseudo)
            S_new[cc] = S_c
            pi_new[cc] = pi_old[cc]

    elif effective_mode == 'rescaling-rates':
        # Hold (S, pi) shape fixed; only update sigma_c with closed form.
        # No pseudocount on (S, pi); rate-only Gamma prior absent here
        # (default flat).
        sigmas = []
        for cc in range(n_cls):
            S_c, pi_c, sigma_c = m_step_subst_rescaling(
                W_per_class[cc], U_per_class[cc], S_old[cc], pi_old[cc])
            S_new[cc] = S_c
            pi_new[cc] = pi_c
            sigmas.append(sigma_c)
        if log_fn is not None:
            log_fn(f"    [rescaling-rates] sigma_c per class: "
                   f"min={min(sigmas):.4f} max={max(sigmas):.4f} "
                   f"mean={float(np.mean(sigmas)):.4f}")

    elif effective_mode == 'rescaling-rates-and-pi':
        # Joint (sigma_c, pi_c) closed-form M-step with S shape frozen.
        # Same DOF per class as a Maraschino fit with --freeze-class-S-shape.
        # See substitution-mstep.tex sec:mstep-rescaling-pi.
        sigmas = []
        # Recover sigma_c warm-start as ||S_old[c]|| / ||S_prior|| (default 1
        # if S_prior is None, treating warm S_old as already at sigma=1 scale).
        if S_prior is not None:
            sp = np.asarray(S_prior, dtype=float)
            sp_norm = float(np.linalg.norm(sp.ravel()))
        else:
            sp_norm = 0.0
        for cc in range(n_cls):
            so = np.asarray(S_old[cc], dtype=float)
            sigma_warm = (float(np.linalg.norm(so.ravel())) / sp_norm
                          if sp_norm > 1e-30 else 1.0)
            S_c, pi_c, sigma_c = m_step_subst_rescaling_pi(
                W_per_class[cc], U_per_class[cc], V_per_class[cc],
                S_old[cc], sigma_warm, pi_old[cc])
            S_new[cc] = S_c
            pi_new[cc] = pi_c
            sigmas.append(sigma_c)
        if log_fn is not None:
            log_fn(f"    [rescaling-rates-and-pi] sigma_c per class: "
                   f"min={min(sigmas):.4f} max={max(sigmas):.4f} "
                   f"mean={float(np.mean(sigmas)):.4f}")

    elif effective_mode == 'tied-pi':
        if n_cls % n_tied != 0:
            raise ValueError(
                f"tied-pi requires n_tied to divide n_classes; "
                f"got n_tied={n_tied}, n_classes={n_cls}")
        blocks = tied_pi_blocks(n_cls, n_tied)
        for blk in blocks:
            # Pool old pi across the block for warm-start.
            pi_block_old = np.mean(pi_old[blk], axis=0)
            pi_block_old = pi_block_old / max(pi_block_old.sum(), 1e-30)
            S_blk_new, pi_blk_new = m_step_subst_tied_pi_block(
                [W_per_class[c] for c in blk],
                [U_per_class[c] for c in blk],
                [V_per_class[c] for c in blk],
                [S_old[c] for c in blk],
                pi_block_old,
                S_prior=S_prior, pi_prior=pi_prior,
                pi_pseudo=pi_pseudo, S_pseudo=S_pseudo)
            for k, c in enumerate(blk):
                S_new[c] = S_blk_new[k]
                pi_new[c] = pi_blk_new
        if log_fn is not None:
            log_fn(f"    [tied-pi] {len(blocks)} blocks of {n_tied} "
                   f"classes each (C={n_cls})")

    elif effective_mode == 'tied-pi-rescaling':
        # pi tied within blocks; per-class sigma_c (S_c shape frozen).
        # Same DOF as a Maraschino fit with --freeze-class-S-shape +
        # block-tied class_pi.
        if n_cls % n_tied != 0:
            raise ValueError(
                f"tied-pi-rescaling requires n_tied to divide n_classes; "
                f"got n_tied={n_tied}, n_classes={n_cls}")
        blocks = tied_pi_blocks(n_cls, n_tied)
        # Recover per-class warm sigma from ||S_old[c]|| / ||S_prior||.
        if S_prior is not None:
            sp_norm = float(np.linalg.norm(np.asarray(S_prior).ravel()))
        else:
            sp_norm = 0.0
        all_sigmas = []
        for blk in blocks:
            pi_block_old = np.mean(pi_old[blk], axis=0)
            pi_block_old = pi_block_old / max(pi_block_old.sum(), 1e-30)
            sigma_old_blk = []
            for c in blk:
                so_norm = float(np.linalg.norm(
                    np.asarray(S_old[c]).ravel()))
                sigma_old_blk.append(so_norm / sp_norm
                                     if sp_norm > 1e-30 else 1.0)
            S_blk_new, pi_blk_new, sigma_blk = (
                m_step_subst_tied_pi_rescaling_block(
                    [W_per_class[c] for c in blk],
                    [U_per_class[c] for c in blk],
                    [V_per_class[c] for c in blk],
                    [S_old[c] for c in blk],
                    sigma_old_blk, pi_block_old))
            for k, c in enumerate(blk):
                S_new[c] = S_blk_new[k]
                pi_new[c] = pi_blk_new
            all_sigmas.extend(sigma_blk)
        if log_fn is not None:
            log_fn(f"    [tied-pi-rescaling] {len(blocks)} blocks of "
                   f"{n_tied} classes (C={n_cls}); sigma_c "
                   f"min={min(all_sigmas):.4f} max={max(all_sigmas):.4f} "
                   f"mean={float(np.mean(all_sigmas)):.4f}")

    else:
        raise ValueError(f"Unknown subst-mode: {mode!r}")

    return S_new, pi_new


def banded_3fc_pseudocounts(n_frag, ext_alpha, ext_beta):
    """Return ext-row pseudocount and term pseudocount selection for the
    banded 3-fragchar parametrisation.

    Used inside the per-row Dirichlet MAP M-step:
        posterior[g] = ext_count[d, f, g] + (ext_alpha-1 if mask else 0)
        posterior_term = term_count[d, f] + (ext_beta-1 if mask else 0)
    Pseudocounts of 0 on structurally-zero entries pin them at 0
    through training (matching the chi log-floor behaviour).

    Args:
        n_frag: number of fragchars (must be 3 to use this helper).
        ext_alpha: Dirichlet alpha for nonzero ext entries.
        ext_beta: Dirichlet beta for nonzero term entries.

    Returns:
        (ext_pseudo[F, F], term_pseudo[F]): pseudocount tensors.
    """
    if n_frag != 3:
        raise ValueError(
            f"banded_3fc_pseudocounts requires n_frag==3, got {n_frag}")
    ext_mask, term_mask = banded_3fc_ext_mask()
    ext_pseudo = np.where(ext_mask, ext_alpha - 1.0, 0.0)
    term_pseudo = np.where(term_mask, ext_beta - 1.0, 0.0)
    return ext_pseudo, term_pseudo
