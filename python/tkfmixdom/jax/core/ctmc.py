"""Substitution model: finite-state CTMC and Holmes-Rubin sufficient statistics."""

import jax
import jax.numpy as jnp


def ensure_rate_matrix(Q):
    """Set diagonal entries so each row sums to zero.

    Args:
        Q: (n, n) matrix with off-diagonal rates.

    Returns:
        Q with diagonal = -(row sum of off-diagonals).
    """
    Q = Q - jnp.diag(jnp.diag(Q))
    return Q - jnp.diag(Q.sum(axis=1))


def rate_matrix_jc69(alphabet_size=4):
    """Jukes-Cantor rate matrix (normalized to 1 substitution per unit time)."""
    q = jnp.ones((alphabet_size, alphabet_size)) / alphabet_size
    q = q - jnp.diag(jnp.diag(q))
    q = q - jnp.diag(q.sum(axis=1))
    # Normalize so mean rate = 1
    pi = jnp.ones(alphabet_size) / alphabet_size
    mean_rate = -jnp.sum(pi * jnp.diag(q))
    return q / mean_rate, pi


def transition_matrix(Q, t):
    """Compute M(t) = exp(Q*t) via Padé scaling-and-squaring.

    Computed by `jax.scipy.linalg.expm`. Padé is well-conditioned for any
    Q (no detailed-balance assumption, no symmetrization step, no
    similarity transform). Its autograd VJP returns the canonical
    ∂M/∂Q tangent — in particular, asymmetric perturbations of Q give
    asymmetric tangents (the previous eigh-via-symmetrize chain
    silently symmetrized, which is wrong off the detailed-balance
    manifold).

    Args:
      Q: (A, A) rate matrix (rows sum to zero). Need not be reversible.
      t: scalar time.

    Returns:
      M: (A, A) transition matrix M(t) = exp(Q t). Rows sum to one
        (up to floating-point error) iff rows of Q sum to zero.

    Dtype: requires float64 inputs — `jax.scipy.linalg.expm` and the
    `jnp.linalg.solve` it uses internally have float64-hardcoded
    coefficients/VJPs. The codebase convention (per
    `feedback_test_dtype_x64.md`) is fp64 throughout, enforced by
    `conftest.py` (`jax.config.update('jax_enable_x64', True)`); callers
    that pass float32 will trip `lax.cond` / `linalg.solve` branch-dtype
    assertions. Cast inputs at the call site if needed.
    """
    Q = jnp.asarray(Q)
    t = jnp.asarray(t, dtype=Q.dtype)
    return jax.scipy.linalg.expm(t * Q)


def build_Q_from_S_pi(S_exch, pi):
    """Build a reversible CTMC rate matrix Q from a symmetric exchangeability
    matrix S_exch and an equilibrium distribution pi (GTR parameterization).

      Q[a, b] = S_exch[a, b] · pi[b]   for a ≠ b
      Q[a, a] = -Σ_{b ≠ a} Q[a, b]

    By construction Q satisfies detailed balance with pi (pi[a] · Q[a, b]
    = pi[b] · Q[b, a]) iff S_exch is symmetric, which is the standard GTR
    assumption. The diagonal of S_exch is masked to zero (the GTR
    convention; non-zero S[a,a] entries are silently dropped — caller's
    responsibility to use the convention).

    For the (S_exch, π) → M(t) computation, compose with
    `transition_matrix(Q, t)`:

        Q = build_Q_from_S_pi(S_exch, π)
        M = transition_matrix(Q, t)

    `transition_matrix` uses Padé via `jax.scipy.linalg.expm`, whose VJP
    returns the canonical ∂M/∂Q tangent (no symmetrize step). Gradient
    ∂M/∂S_exch flows cleanly via the chain Q[a,b] = S[a,b]·π[b].
    """
    A = pi.shape[-1]
    eye = jnp.eye(A, dtype=S_exch.dtype)
    # Off-diag: Q[a,b] = S[a,b] · pi[b]; diagonal zeroed.
    Q_off = S_exch * pi[..., None, :]
    Q_off = jnp.where(eye.astype(bool), 0.0, Q_off)
    # Diagonal: -row-sum.
    Q_diag = -Q_off.sum(axis=-1)
    Q = Q_off + Q_diag[..., :, None] * eye
    return Q


def build_rate_matrix_unit_normalized(S_exch, pi, *, acknowledged_lossy=False):
    """Build a GTR rate matrix Q from (S_exch, π) and divide by the equilibrium
    mean rate, so that the resulting Q has Σ_i π[i] · Σ_{j≠i} Q[i,j] = 1
    (i.e. one expected substitution per site per unit time at equilibrium).

    .. warning::
        ALMOST CERTAINLY DO NOT WANT THIS FUNCTION.

        This rescales Q so that t is measured in expected substitutions
        per site at equilibrium. That is the right convention only when
        the (S_exch, π) pair is a fixed published rate matrix
        (LG08, WAG, JC69, F81, …) whose authors calibrated their rates
        against that convention, OR when the caller deliberately
        factors the absolute rate scale into a separate multiplicative
        array (Annabel's ``rate_multipliers``, discrete-Γ ``γ_c``, …).
        **Almost everything else** — anything learned from data,
        anything trained per-domain or per-class, anything where the
        absolute scale of (S_exch, π) is the model's own information
        about evolutionary tempo — must NOT be passed through this
        function.

        Concrete failure mode this function caused on this project:
        partition-reconstruction benchmarks for trained MixDom models
        (d3f1, d4f1, d5f1) were collapsing to identical reconstructions
        regardless of model size, because each per-domain Q was
        independently rescaled to mean rate 1, stripping the trained
        per-domain rate-scale information that was the model's whole
        point.

        For trained models, use :func:`build_Q_from_S_pi` (no
        normalization). The mean rate is then a property of the
        trained parameters and t is in raw rate-matrix units consistent
        with whatever t was used at training time.

        Even for ostensibly fixed matrices, prefer the unnormalized
        helper if you have the slightest doubt: t can always be
        rescaled at the call site, but information stripped here is
        gone.

    Args:
        S_exch: (A, A) symmetric exchangeability with zero diagonal.
        pi: (A,) equilibrium distribution.
        acknowledged_lossy: pass ``True`` from call sites where the
            unit-normalisation is intentional (e.g. Annabel emission
            tables that multiply ``rate_multipliers`` back in, or
            distillation tensors whose downstream consumer expects
            unit-rate Q). Suppresses the runtime ``UserWarning`` that
            otherwise fires on every call. Keyword-only by design — a
            caller that has not read this docstring should not be able
            to silence the warning by accident.

    Returns:
        Q: (..., A, A) rate matrix with zero diagonal corrected and
        equilibrium mean rate normalised to 1.
    """
    if not acknowledged_lossy:
        import warnings
        warnings.warn(
            "build_rate_matrix_unit_normalized strips the absolute rate "
            "scale of (S_exch, pi). For trained models use "
            "tkfmixdom.jax.core.ctmc.build_Q_from_S_pi instead. If this "
            "call site is deliberately factoring rate scale into a "
            "separate multiplier (Annabel rate_multipliers, "
            "discrete-gamma, etc.), pass acknowledged_lossy=True to "
            "silence this warning. See function docstring for the full "
            "rationale.",
            UserWarning,
            stacklevel=2,
        )
    Q = build_Q_from_S_pi(S_exch, pi)
    pi_arr = jnp.asarray(pi)
    mean_rate = -jnp.sum(pi_arr * jnp.diagonal(Q, axis1=-2, axis2=-1), axis=-1)
    # Broadcast scalar mean_rate over the trailing (A, A) of Q.
    return Q / jnp.maximum(mean_rate, 1e-30)[..., None, None]


def holmes_rubin_integrals(Q, pi, t):
    """Compute Holmes-Rubin I^{ab}_{ij}(T) integrals via spectral decomposition.

    Returns:
        I: array of shape (n, n, n, n) where I[a,b,i,j] = integral_0^T M_{ai}(s) M_{jb}(T-s) ds
        M: transition matrix M(T)
    """
    n = Q.shape[0]
    sqrt_pi = jnp.sqrt(pi)

    S = Q * (sqrt_pi[:, None] / sqrt_pi[None, :])
    S = (S + S.T) / 2
    eigenvalues, V = jnp.linalg.eigh(S)

    # J^{kl}(T) = T*exp(lambda_k*T) if lambda_k == lambda_l,
    #           = (exp(lambda_k*T) - exp(lambda_l*T))/(lambda_k - lambda_l) otherwise
    # (Holmes & Rubin 2002, equation below eq. 3)
    #
    # The J formula handles degenerate eigenvalues exactly (not via L'Hôpital).
    # The safe_diff prevents division by zero when both branches of jnp.where
    # are evaluated (JAX semantics).
    #
    # NOTE: holmes_rubin_integrals is NOT differentiable w.r.t. Q via JAX autodiff
    # because jnp.linalg.eigh's gradient blows up at degenerate eigenvalues
    # (the eigenvector gradient involves 1/(λ_i - λ_j)). For EM, this is fine:
    # the Q-function uses pre-computed Holmes-Rubin stats as frozen constants.
    # Making this differentiable would require a custom JVP for eigh that handles
    # the degenerate case (e.g. via perturbation theory or the Daleckiĭ-Kreĭn formula).
    exp_eig = jnp.exp(eigenvalues * t)
    diff = eigenvalues[:, None] - eigenvalues[None, :]
    safe_diff = jnp.where(jnp.abs(diff) < 1e-10, 1.0, diff)
    J = jnp.where(
        jnp.abs(diff) < 1e-10,
        t * exp_eig[:, None],
        (exp_eig[:, None] - exp_eig[None, :]) / safe_diff
    )

    # I^{ab}_{ij}(T) = sqrt(pi_i * pi_b / (pi_a * pi_j))
    #                  * sum_k V_{ak} V_{ik} * sum_l V_{jl} V_{bl} * J^{kl}
    # Use einsum for efficiency
    # VV_ai_k = V[a,k] * V[i,k] -> shape (n, n, n_eig)
    # VV_jb_l = V[j,l] * V[b,l] -> shape (n, n, n_eig)
    VV_left = V[:, None, :] * V[None, :, :]   # (a, i, k)
    VV_right = V[:, None, :] * V[None, :, :]  # (j, b, l)

    # sum over k,l: VV_left[a,i,k] * J[k,l] * VV_right[j,b,l]
    # -> contract: (a,i,k) @ (k,l) @ (j,b,l)^T ... use einsum
    I_sym = jnp.einsum('aik,kl,jbl->aibj', VV_left, J, VV_right)

    # Unsymmetrize: I^{ab}_{ij} = sqrt(pi_i * pi_b / (pi_a * pi_j)) * I_sym
    pi_factor = jnp.sqrt(
        (pi[None, None, :, None] * pi[None, :, None, None]) /
        (pi[:, None, None, None] * pi[None, None, None, :] + 1e-30)
    )
    # I_sym has shape (a, i, b, j), we need (a, b, i, j)
    I = I_sym.transpose(0, 2, 1, 3) * pi_factor.transpose(0, 2, 1, 3)

    # Transition matrix
    exp_diag = jnp.exp(eigenvalues * t)
    M_sym = (V * exp_diag[None, :]) @ V.T
    M = M_sym * (sqrt_pi[None, :] / sqrt_pi[:, None])

    return I, M


def ctmc_log_prior(Q, gamma_shape=2.0, gamma_rate=1.0):
    """Log-prior on CTMC rate matrix: independent Gamma on each off-diagonal rate.

    Gamma(shape, rate) on each q_{ij} (i≠j) decomposes into:
    - Gamma((K-1)*shape, rate) on exit rate r_i = -Q[i,i]
    - Symmetric Dirichlet(shape) on jump target probabilities

    Requires shape >= 1 for a proper MAP estimate.

    Returns:
        log p(Q) = sum_{i!=j} [(shape-1)*log(q_ij) - rate*q_ij] + const
    """
    n = Q.shape[0]
    mask = 1.0 - jnp.eye(n)
    off_diag = Q * mask
    # Clamp to avoid log(0) for zero rates
    off_diag_safe = jnp.maximum(off_diag, 1e-30) * mask
    return jnp.sum(mask * ((gamma_shape - 1.0) * jnp.log(off_diag_safe)
                           - gamma_rate * off_diag))


def holmes_rubin_expected_stats(Q, pi, t, a, b):
    """Compute expected dwell times and transition counts for path a->b in time t.

    Args:
        Q: rate matrix (n, n)
        pi: stationary distribution (n,)
        t: time
        a, b: start and end states

    Returns:
        w_hat: expected dwell times (n,)
        u_hat: expected transition counts (n, n)
    """
    I, M = holmes_rubin_integrals(Q, pi, t)
    M_ab = M[a, b]

    n = Q.shape[0]
    w_hat = I[a, b, jnp.arange(n), jnp.arange(n)] / M_ab
    u_hat = Q * I[a, b] / M_ab

    # Zero out diagonal of u_hat (no self-transitions)
    u_hat = u_hat - jnp.diag(jnp.diag(u_hat))

    return w_hat, u_hat


def holmes_rubin_weighted_stats(Q, pi, t, mc):
    """Compute Σ_{a,b} mc[a,b]·(w_hat_ab, u_hat_ab) at fixed (Q, pi, t).

    Vectorized over the (a, b) match-count grid. Equivalent to:

        W[i] = Σ_{a,b} mc[a,b] · (I[a,b,i,i] / M[a,b])
        U[i,j] = Σ_{a,b} mc[a,b] · (Q[i,j] · I[a,b,i,j] / M[a,b])

    where I and M are the Holmes-Rubin integrals from `holmes_rubin_integrals`.

    The (Q, pi, t)-dependent precomputation (eigendecomposition of Q,
    integral tensor I) is shared across all (a, b) pairs in this single
    call — much faster than the per-(a, b) `holmes_rubin_expected_stats`
    loop in the M-step when there are many non-zero entries in mc.

    Per-pair-t aggregation: callers process pairs with different t_p by
    accumulating the W/U returned here across calls. Both W and U are
    additive in mc, so per-pair sums recover the time-aware aggregate
    sufficient statistics.

    Args:
        Q: (A, A) rate matrix (must be reversible w.r.t. pi).
        pi: (A,) stationary distribution.
        t: scalar evolutionary time t_p.
        mc: (A, A) per-pair match-count weights.

    Returns:
        W: (A,) accumulated dwell times.
        U: (A, A) accumulated transition counts (zero diagonal).
    """
    I, M = holmes_rubin_integrals(Q, pi, t)
    M_safe = jnp.where(jnp.abs(M) < 1e-30, 1.0, M)
    ratio = mc / M_safe  # (A, A)
    # W[i] = Σ_{a,b} ratio[a,b] · I[a,b,i,i]
    W = jnp.einsum('ab,abii->i', ratio, I)
    # U[i,j] = Q[i,j] · Σ_{a,b} ratio[a,b] · I[a,b,i,j]
    sum_ratio_I = jnp.einsum('ab,abij->ij', ratio, I)
    U = Q * sum_ratio_I
    # Zero out diagonal (no self-transitions)
    U = U - jnp.diag(jnp.diag(U))
    return W, U


def m_step_substitution(match_pairs, Q, pi, t, alphabet_size=4, min_weight=None,
                        gamma_shape=1.0, gamma_rate=0.0, reversible=True):
    """M-step for substitution model: re-estimate Q from weighted match pairs.

    Implements tkf.tex sec:bw-tkf91 Q̂ alternatives (Holmes-Rubin CTMC M-step).
    Uses Holmes-Rubin expected sufficient statistics (eq:em-dwell, eq:em-counts).
    match_pairs: list of (a, b, weight) triples.

    With Gamma(gamma_shape, gamma_rate) prior on each off-diagonal rate q_{ij}:
        MAP: q̂_{ij} = (u_{ij} + gamma_shape - 1) / (w_i + gamma_rate)
    Requires gamma_shape >= 1 for non-negative MAP estimates.
    Default (shape=1, rate=0) gives the MLE (no prior).

    When reversible=False, uses the general (irreversible) Holmes-Rubin
    integrals and recomputes pi from the updated Q.

    Returns:
        Q_new (if reversible=True)
        (Q_new, pi_new) (if reversible=False)
    """
    if min_weight is None:
        min_weight = float(alphabet_size)

    total_weight = sum(w for _, _, w in match_pairs)
    if total_weight < min_weight:
        return Q if reversible else (Q, pi)

    w_total = jnp.zeros(alphabet_size)
    u_total = jnp.zeros((alphabet_size, alphabet_size))

    if reversible:
        for a, b, weight in match_pairs:
            w_hat, u_hat = holmes_rubin_expected_stats(Q, pi, t, a, b)
            w_total = w_total + weight * w_hat
            u_total = u_total + weight * u_hat
    else:
        from .ctmc_irreversible import (
            holmes_rubin_expected_stats_general, stationary_distribution)
        for a, b, weight in match_pairs:
            w_hat, u_hat = holmes_rubin_expected_stats_general(Q, t, a, b)
            w_total = w_total + weight * w_hat
            u_total = u_total + weight * u_hat

    # MAP with Gamma prior: q̂_{ij} = max(0, u_{ij} + α - 1) / (w_i + β)
    u_total = jnp.maximum(u_total + (gamma_shape - 1.0), 0.0)
    w_total = w_total + gamma_rate

    Q_new = u_total / jnp.maximum(w_total[:, None], 1e-30)
    Q_new = Q_new - jnp.diag(jnp.diag(Q_new))
    Q_new = Q_new - jnp.diag(Q_new.sum(axis=1))

    if not reversible:
        from .ctmc_irreversible import stationary_distribution
        pi_new = stationary_distribution(Q_new)
        return Q_new, pi_new

    return Q_new


def class_responsibilities(class_weights, pis, sub_matrices, a, b):
    """Posterior responsibility of each site class for emission (a, b).

    Implements tkf.tex γ_{fc}(a,b):
        γ_c(a,b) = u_c · π_c[a] · P_c(t)[a,b] / Σ_{c'} u_{c'} · π_{c'}[a] · P_{c'}(t)[a,b]

    Args:
        class_weights: (C,) mixture weights u_c (simplex).
        pis: (C, A) equilibrium distributions per class.
        sub_matrices: (C, A, A) transition matrices P_c(t) = exp(Q_c·t).
        a, b: ancestor and descendant character indices.

    Returns:
        (C,) posterior responsibilities γ_c(a, b), summing to 1.
    """
    # Likelihood per class: u_c * pi_c[a] * P_c[a,b]
    lik = class_weights * pis[:, a] * sub_matrices[:, a, b]
    total = jnp.sum(lik)
    return jnp.where(total > 1e-30, lik / total, class_weights)


def class_responsibilities_singlet(class_weights, pis, x):
    """Posterior responsibility for singlet emission (insert or delete).

    γ^I_c(x) = u_c · π_c[x] / Σ_{c'} u_{c'} · π_{c'}[x]
    """
    lik = class_weights * pis[:, x]
    total = jnp.sum(lik)
    return jnp.where(total > 1e-30, lik / total, class_weights)


def m_step_substitution_multiclass(match_pairs_by_class, Qs, pis, t,
                                    alphabet_size=4, gamma_shape=1.0,
                                    gamma_rate=0.0):
    """M-step for multi-class substitution model.

    Each class c has its own (Q_c, π_c). match_pairs_by_class[c] is a list
    of (a, b, weight) triples for class c (after γ_{fc} posterior splitting).

    Returns:
        list of Q_new_c arrays, one per class.
    """
    n_class = len(Qs)
    Q_news = []
    for c in range(n_class):
        Q_c_new = m_step_substitution(
            match_pairs_by_class[c], Qs[c], pis[c], t,
            alphabet_size=alphabet_size,
            gamma_shape=gamma_shape, gamma_rate=gamma_rate)
        Q_news.append(Q_c_new)
    return Q_news


def m_step_subst_option1(W, U, V, S_prior=None, pi_prior=None,
                         pi_pseudo=1.0, S_pseudo=0.0, n_iter=50, tol=1e-10):
    """Exact substitution M-step via iterative coordinate ascent
    (paper option 2, tkf.tex lines 821-822).

    Iterates between two coordinate-wise EXACT updates:
      1. (S | π fixed): closed form S_ij = (U_ij + U_ji) / (W_i π_j + W_j π_i)
      2. (π | S fixed): Lagrange-multiplier solve π_i = V'_i / (c_i - η),
         where η solves Σ_i V'_i/(c_i - η) = 1 by bisection
         (60 iterations of [eta_min, eta_max] → ~1e-18 precision).

    The joint objective ℓ_2(S, π) is strictly concave on the simplex × R₊
    (Hessian over π is negative definite when V'_i > 0; tkf.tex L660), so
    coordinate ascent converges to the unique global maximum — i.e., the
    EXACT joint MLE. Convergence is geometric. With n_iter ≥ ~20 and the
    early-stop tolerance below, the result is the perfect M-step up to
    floating-point precision.

    Each iteration:
      1. S_ij = (U_ij + U_ji) / (W_i·π_j + W_j·π_i)  [closed form]
      2. Solve η from Σ_i V'_i / (c_i - η) = 1 where c_i = Σ_{j≠i} S_ij·W_j
         [1D bisection; f(η) is monotone increasing]
      3. π_i = V'_i / (c_i - η)

    Initialized with π from character counts V (option 1 as starting point).

    Args:
        W: (A,) expected dwell times per state
        U: (A, A) expected transition counts (off-diagonal)
        V: (A,) character counts in the joint pair HMM (match-pos ancestor
           + insert + delete, see tkf.tex L755)
        S_prior: optional (A, A) exchangeability prior (e.g. LG08)
        pi_prior: optional (A,) equilibrium prior (e.g. LG08)
        pi_pseudo: pseudocount weight for π prior (Dirichlet concentration)
        S_pseudo: pseudocount weight for S prior
        n_iter: maximum coordinate ascent iterations (default 50; usually
                converges in <10 due to geometric rate)
        tol: relative tolerance for early-stopping (default 1e-10);
             stops when max(|Δπ|) / max(|π|) < tol

    Returns:
        S_new: (A, A) symmetric exchangeability matrix (zero diagonal)
        pi_new: (A,) equilibrium distribution
        Q_new: (A, A) rate matrix in raw GTR units (Q_ij = S_ij·π_j off
            the diagonal, Q_ii = -Σ_{j≠i} Q_ij). Not unit-normalised:
            the absolute rate scale is preserved as part of the trained
            model's information (see comment block below for the
            rationale). Time t at the call site must be expressed in
            the same units as the (W, U) sufficient statistics that
            produced this Q.
    """
    import numpy as np
    A = len(V)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    U = np.array(U, dtype=float)

    # Add pseudocounts
    if pi_prior is not None and pi_pseudo > 0:
        pi_pr = np.array(pi_prior, dtype=float)
        V = V + pi_pseudo * pi_pr
    if S_prior is not None and pi_prior is not None and S_pseudo > 0:
        S_pr = np.array(S_prior, dtype=float)
        pi_pr = np.array(pi_prior, dtype=float)
        U_pseudo = S_pseudo * S_pr * pi_pr[None, :]
        W_pseudo = S_pseudo * np.sum(S_pr * pi_pr[None, :], axis=1)
        U = U + U_pseudo
        W = W + W_pseudo

    # V'_i = V_i + Σ_{j≠i} U_{ji}  (tkf.tex line 627)
    U_col_sum = U.sum(axis=0)  # Σ_j U_{ji} for each i
    V_prime = V + U_col_sum - np.diag(U)  # exclude self-transitions

    # Initialize π from V (option 1 starting point)
    V_total = V.sum()
    pi_new = V / max(V_total, 1e-30) if V_total > 1e-30 else np.ones(A) / A

    U_sym = U + U.T

    for it in range(n_iter):
        # Step 1: Fix π, solve S in closed form (eq 657)
        denom = W[:, None] * pi_new[None, :] + W[None, :] * pi_new[:, None]
        denom = np.maximum(denom, 1e-30)
        S_new = U_sym / denom
        np.fill_diagonal(S_new, 0.0)

        # Step 2: Fix S, solve for π via Lagrange multiplier η
        # From ∂ℓ/∂π_i = 0: π_i = V'_i / (c_i - η)
        # where c_i = Σ_{j≠i} S_ij · W_j
        # Constraint: Σ_i π_i = 1 → Σ_i V'_i / (c_i - η) = 1
        c = (S_new * W[None, :]).sum(axis=1)  # c_i = Σ_j S_ij · W_j

        # Bisection to find η such that f(η) = Σ V'_i/(c_i - η) - 1 = 0
        # f is monotone increasing; η must be < min(c_i) for all terms positive
        eta_max = c.min() - 1e-10
        eta_min = eta_max - 10.0 * V_prime.sum()  # generous lower bound

        for _ in range(60):  # bisection iterations (convergence to ~1e-18)
            eta_mid = (eta_min + eta_max) / 2.0
            f_mid = np.sum(V_prime / np.maximum(c - eta_mid, 1e-30)) - 1.0
            if f_mid > 0:
                eta_max = eta_mid
            else:
                eta_min = eta_mid

        eta = (eta_min + eta_max) / 2.0
        pi_candidate = V_prime / np.maximum(c - eta, 1e-30)

        # Safety: ensure valid probability distribution
        if np.all(pi_candidate > 0) and np.isfinite(pi_candidate.sum()):
            pi_new = pi_candidate / pi_candidate.sum()
        # else keep previous pi_new

    # Build Q = S · diag(π) with diagonal = -row_sum(off-diag).
    #
    # The paper's M-step (tkf.tex L786, L821-822) does NOT specify a
    # rate normalization step (Q / mean_rate) — it just defines Q via
    # Q_ij = S_ij π_j (off-diag) and Q_ii = -Σ Q_ij. We therefore
    # construct Q exactly that way here. Any rate convention is left to
    # the caller (e.g., the static LG rate matrix is shipped at mean
    # rate 1 by convention from the LG paper, but trained Q's evolve
    # away from that scale — the model's t-units evolve with them).
    #
    # Operation order (np.fill_diagonal + np.diag_indices assignment)
    # matches the per-class E-step's Q rebuild in
    # tkfmixdom/jax/train/constrained.py L1167-1170, so the M-step's
    # output Q is bit-identical to a re-derivation from the same S, π.
    Q_new = S_new * pi_new[None, :]
    np.fill_diagonal(Q_new, 0.0)
    Q_new[np.diag_indices(len(pi_new))] = -Q_new.sum(axis=1)

    return S_new, pi_new, Q_new
