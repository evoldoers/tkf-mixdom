"""Substitution model for irreversible (non-reversible) CTMCs.

This module provides variants of the functions in subst.py that work with
general (irreversible) rate matrices. The key difference is that the
eigendecomposition is performed on Q directly (not a symmetrized version),
yielding potentially complex eigenvalues and eigenvectors.

Note: The eigendecomposition-based functions (stationary_distribution,
transition_matrix_general, holmes_rubin_integrals_general) do NOT support
JAX autodiff through jnp.linalg.eig. Use transition_matrix_expm for
optimization paths that require gradients (e.g. L-BFGS).
"""

import jax.numpy as jnp
import jax.scipy.linalg


def stationary_distribution(Q):
    """Compute the stationary distribution pi as the left eigenvector of Q.

    Finds the eigenvector of Q^T corresponding to eigenvalue 0 (the eigenvalue
    closest to zero), normalizes it to sum to 1, and returns the real part.

    Note: Does NOT support JAX autodiff (uses jnp.linalg.eig).

    Args:
        Q: (n, n) rate matrix (rows sum to zero).

    Returns:
        pi: (n,) stationary distribution (real, non-negative, sums to 1).
    """
    eigenvalues, eigenvectors = jnp.linalg.eig(Q.T)
    # Find the eigenvalue closest to 0
    idx = jnp.argmin(jnp.abs(eigenvalues))
    pi = eigenvectors[:, idx]
    # Normalize to sum to 1, taking real part
    pi = pi.real
    pi = pi / jnp.sum(pi)
    # Ensure non-negative (numerical noise can produce tiny negatives)
    pi = jnp.maximum(pi, 0.0)
    pi = pi / jnp.sum(pi)
    return pi


def transition_matrix_general(Q, t):
    """Compute M(t) = exp(Q*t) via eigendecomposition for a general rate matrix.

    Uses V @ diag(exp(eigenvalues * t)) @ V^{-1}, handling complex eigenvalues.
    The result is real (imaginary parts arise only from numerical noise).

    Note: Does NOT support JAX autodiff (uses jnp.linalg.eig).

    Args:
        Q: (n, n) rate matrix (not necessarily reversible).
        t: scalar time.

    Returns:
        M: (n, n) transition matrix exp(Q*t), real-valued.
    """
    eigenvalues, V = jnp.linalg.eig(Q)
    V_inv = jnp.linalg.inv(V)
    exp_diag = jnp.exp(eigenvalues * t)
    M = (V * exp_diag[None, :]) @ V_inv
    return M.real


def transition_matrix_expm(Q, t):
    """Compute M(t) = exp(Q*t) using jax.scipy.linalg.expm.

    This version supports JAX autodiff and can be used for gradient-based
    optimization (e.g. L-BFGS). It works for any rate matrix, reversible
    or irreversible.

    Args:
        Q: (n, n) rate matrix.
        t: scalar time.

    Returns:
        M: (n, n) transition matrix exp(Q*t).
    """
    return jax.scipy.linalg.expm(Q * t)


def holmes_rubin_integrals_general(Q, t):
    """Compute Holmes-Rubin I^{ab}_{ij}(T) integrals for a general rate matrix.

    Uses the eigendecomposition of Q directly (not a symmetrized version),
    so this works for irreversible rate matrices. The formula is:

        I^{ab}_{ij}(T) = sum_{k,l} V_{a,k} V_inv_{k,i} J^{kl}(T) V_{j,l} V_inv_{l,b}

    where J^{kl}(T) = (exp(lam_k * T) - exp(lam_l * T)) / (lam_k - lam_l),
    with L'Hopital's rule for near-equal eigenvalues giving J^{kk}(T) = T * exp(lam_k * T).

    Complex eigenvalues are handled correctly; the final result is real.

    Note: Does NOT support JAX autodiff (uses jnp.linalg.eig).

    Args:
        Q: (n, n) rate matrix.
        t: scalar time.

    Returns:
        I: (n, n, n, n) array where I[a, b, i, j] = integral_0^T M_{ai}(s) M_{jb}(T-s) ds
        M: (n, n) transition matrix M(T) = exp(Q*T).
    """
    eigenvalues, V = jnp.linalg.eig(Q)
    V_inv = jnp.linalg.inv(V)

    # J^{kl}(T): shape (n_eig, n_eig), possibly complex
    exp_eig = jnp.exp(eigenvalues * t)
    diff = eigenvalues[:, None] - eigenvalues[None, :]
    J = jnp.where(
        jnp.abs(diff) < 1e-10,
        t * exp_eig[:, None],
        (exp_eig[:, None] - exp_eig[None, :]) / jnp.where(jnp.abs(diff) < 1e-10, 1.0, diff)
    )

    # I[a, b, i, j] = sum_{k,l} V[a,k] * V_inv[k,i] * J[k,l] * V[j,l] * V_inv[l,b]
    I = jnp.einsum('ak,ki,kl,jl,lb->abij', V, V_inv, J, V, V_inv)

    # Transition matrix
    M = ((V * exp_eig[None, :]) @ V_inv).real

    return I.real, M


def holmes_rubin_expected_stats_general(Q, t, a, b):
    """Compute expected dwell times and transition counts for a->b in time t.

    This is the irreversible analogue of holmes_rubin_expected_stats in subst.py.
    It uses the general eigendecomposition (not symmetrized), so it works for
    any rate matrix.

    Note: Does NOT support JAX autodiff (uses jnp.linalg.eig internally).

    Args:
        Q: (n, n) rate matrix (not necessarily reversible).
        t: scalar time.
        a, b: start and end states (integers).

    Returns:
        w_hat: (n,) expected dwell times in each state.
        u_hat: (n, n) expected transition counts (diagonal is zero).
    """
    I, M = holmes_rubin_integrals_general(Q, t)
    M_ab = M[a, b]

    n = Q.shape[0]
    w_hat = I[a, b, jnp.arange(n), jnp.arange(n)] / M_ab
    u_hat = Q * I[a, b] / M_ab

    # Zero out diagonal (no self-transitions)
    u_hat = u_hat - jnp.diag(jnp.diag(u_hat))

    return w_hat, u_hat
