"""Generic EM training for any CompiledModel.

Works with TKF91, TKF92, MixDom, TKFST — any model that implements
the CompiledModel interface (build_trans, e_step, extract_stats, m_step).

For model-specific training with full features (L-BFGS switching,
conditioned mode, safeguards), use the specialized em_tkf.py / em_mixdom.py.

Also provides generic pytree operations for sufficient statistics:
- _add_stats: aggregate stats across sequence pairs (CherryML-style)
- score_from_counts: differentiable score function for gradient-based optimization
"""

from __future__ import annotations

from typing import Any, Protocol, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import BDISuffStats, SimplexSuffStats, BernoulliSuffStats
from .optimizer import em_optimize, OptimizeResult
from ..dp.hmm import safe_log


# ---------------------------------------------------------------------------
# Pytree operations for sufficient statistics
# ---------------------------------------------------------------------------

def _add_stats(a, b):
    """Recursively add two stats pytrees (dicts, lists, dataclasses, arrays, scalars)."""
    if isinstance(a, BDISuffStats):
        return BDISuffStats(E_B=a.E_B + b.E_B, E_D=a.E_D + b.E_D,
                            E_S=a.E_S + b.E_S,
                            n_kappa=a.n_kappa + b.n_kappa,
                            n_1mkappa=a.n_1mkappa + b.n_1mkappa,
                            T=a.T + b.T)
    if isinstance(a, SimplexSuffStats):
        return SimplexSuffStats(counts=a.counts + b.counts)
    if isinstance(a, BernoulliSuffStats):
        return BernoulliSuffStats(n_success=a.n_success + b.n_success,
                                  n_failure=a.n_failure + b.n_failure)
    if isinstance(a, dict):
        return {k: _add_stats(a[k], b[k]) for k in a if k in b}
    if isinstance(a, list):
        return [_add_stats(ai, bi) for ai, bi in zip(a, b)]
    if hasattr(a, 'shape'):
        return np.array(a) + np.array(b)
    if isinstance(a, (int, float)):
        return a + b
    return a  # unknown type, keep first


def score_from_counts(build_trans_fn: Callable, n_counts: jax.Array,
                      *args) -> float:
    """Compute Σ n[i,j] × log(trans(*args)[i,j]).

    This is the score function whose gradient w.r.t. *args gives
    d(log P)/d(params) via the score function identity:
        d(log P)/d(θ) = Σ_ij E[n_ij] × d(log χ_ij)/d(θ)

    Use with jax.grad to get parameter gradients for Adam/L-BFGS:
        grad_fn = jax.grad(score_from_counts, argnums=2)
        d_params = grad_fn(model.build_trans, n_counts, params)

    Args:
        build_trans_fn: callable(*args) -> (trans_matrix, state_types)
        n_counts: expected transition counts from E-step
        *args: parameters passed to build_trans_fn (differentiable)

    Returns:
        scalar: Σ n[i,j] × log(trans[i,j])
    """
    trans, _ = build_trans_fn(*args)
    log_trans = safe_log(trans)
    return jnp.sum(n_counts * log_trans)


class TrainableModel(Protocol):
    """Model that supports generic EM training.

    Extends CompiledModel with e_step for running Forward-Backward.
    """
    def e_step(self, params: dict, x: jax.Array, y: jax.Array
               ) -> tuple[float, jax.Array, jax.Array]:
        """Run Forward-Backward.

        Returns:
            log_prob: scalar log-likelihood
            n_trans: expected transition counts
            posteriors: posterior state probabilities
        """
        ...

    def extract_stats(self, n_trans: jax.Array, params: dict) -> dict:
        """Extract sufficient statistics from transition counts."""
        ...

    def m_step(self, stats: dict, params: dict) -> dict:
        """Update parameters from sufficient statistics."""
        ...


def em_single_pair(model: TrainableModel,
                   params: dict,
                   x: jax.Array,
                   y: jax.Array,
                   n_iter: int = 50,
                   convergence_tol: float = 0.1,
                   verbose: bool = False) -> OptimizeResult:
    """Run EM on a single sequence pair.

    Args:
        model: any TrainableModel (TKF91Model, TKF92Model, MixDomModel, etc.)
        params: initial parameter dict
        x, y: integer sequences
        n_iter: max iterations
        convergence_tol: stop when |ΔLL| < tol
        verbose: print progress

    Returns:
        OptimizeResult with final params and log-probability history.
    """
    def e_step(p):
        ll, n_trans, _ = model.e_step(p, x, y)
        stats = model.extract_stats(n_trans, p)
        return ll, stats

    def m_step(p, stats):
        return model.m_step(stats, p)

    return em_optimize(
        params=params,
        e_step=e_step,
        m_step=m_step,
        n_iter=n_iter,
        convergence_tol=convergence_tol,
        verbose=verbose,
    )


def em_aggregate(model: TrainableModel,
                 params: dict,
                 pairs: list[tuple[jax.Array, jax.Array]],
                 n_iter: int = 50,
                 convergence_tol: float = 0.1,
                 verbose: bool = False) -> OptimizeResult:
    """Run aggregate EM over multiple sequence pairs.

    CherryML-style: aggregates E-step counts from all pairs before M-step.

    Args:
        model: any TrainableModel
        params: initial parameter dict (shared across all pairs)
        pairs: list of (x, y) sequence pairs
        n_iter: max iterations
        convergence_tol: stop when |ΔLL| < tol
        verbose: print progress

    Returns:
        OptimizeResult with final params and total log-probability history.
    """
    def e_step(p):
        total_ll = 0.0
        agg_stats = None

        for x, y in pairs:
            ll, n_trans, _ = model.e_step(p, x, y)
            total_ll += ll
            stats = model.extract_stats(n_trans, p)

            if agg_stats is None:
                agg_stats = stats
            else:
                agg_stats = _add_stats(agg_stats, stats)

        return total_ll, agg_stats

    def m_step(p, stats):
        return model.m_step(stats, p)

    return em_optimize(
        params=params,
        e_step=e_step,
        m_step=m_step,
        n_iter=n_iter,
        convergence_tol=convergence_tol,
        verbose=verbose,
    )
