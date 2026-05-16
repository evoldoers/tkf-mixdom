"""Abstract EM optimizer.

Provides a generic optimization loop that:
1. Runs EM iterations using user-supplied E-step and M-step callables
2. Collects timing data per iteration

The E-step returns (log_likelihood, sufficient_statistics).
The M-step takes sufficient_statistics and returns updated parameters.

The M-step is always a closed-form update from sufficient statistics.
No line search, damping, or numerical optimization is used.
"""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

import numpy as np
from ..util.timing import Timer

P = TypeVar('P')  # parameter type
S = TypeVar('S')  # sufficient statistics type


class OptimizeResult:
    """Result of an optimization run."""
    __slots__ = ['params', 'log_probs', 'converged', 'n_iter', 'timer']

    def __init__(self, params: Any, log_probs: list[float],
                 converged: bool, n_iter: int, timer: Timer) -> None:
        self.params = params
        self.log_probs = log_probs
        self.converged = converged
        self.n_iter = n_iter
        self.timer = timer


def em_optimize(params: Any,
                e_step: Callable[[Any], tuple[float, Any]],
                m_step: Callable[[Any, Any], Any],
                n_iter: int = 50,
                convergence_tol: float = 0.1,
                verbose: bool = False,
                timer: Timer | None = None) -> OptimizeResult:
    """Generic EM loop with closed-form M-step.

    Args:
        params: initial parameter dict/object (passed to e_step and m_step)
        e_step: callable(params) -> (log_likelihood: float, stats: any)
            Runs the E-step (e.g. Forward-Backward on all pairs, aggregating
            sufficient statistics).
        m_step: callable(params, stats) -> params
            Runs the M-step (closed-form update from sufficient statistics).
        n_iter: maximum EM iterations
        convergence_tol: declare convergence when |ΔLL| < this.
        verbose: print progress
        timer: Timer instance for collecting timing data (created if None)

    Returns:
        OptimizeResult with final params, log_probs history, timing data
    """
    if timer is None:
        timer = Timer()

    log_probs = []
    converged = False

    for it in range(n_iter):
        # E-step
        with timer.section('e_step') as t_e:
            ll, stats = e_step(params)
        log_probs.append(ll)

        if verbose:
            delta = ll - log_probs[-2] if len(log_probs) >= 2 else float('nan')
            print(f"  iter {it+1}: LL={ll:.2f} (ΔLL={delta:+.4f}, "
                  f"E={t_e.elapsed:.1f}s)", end='')

        # Check convergence
        if len(log_probs) >= 2:
            delta_ll = abs(log_probs[-1] - log_probs[-2])

            if delta_ll < convergence_tol:
                if verbose:
                    print(f" — converged")
                converged = True
                break

        # M-step (closed-form, exact)
        with timer.section('m_step') as t_m:
            params = m_step(params, stats)

        if verbose:
            print(f", M={t_m.elapsed:.1f}s")

    timer.record('n_iter', len(log_probs))
    timer.record('final_ll', log_probs[-1] if log_probs else float('nan'))
    timer.record('converged', converged)

    return OptimizeResult(
        params=params,
        log_probs=log_probs,
        converged=converged,
        n_iter=len(log_probs),
        timer=timer,
    )
