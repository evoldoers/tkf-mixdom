"""Pairwise fitting pipeline.

High-level interface for fitting TKF91 parameters to sequence pairs,
scoring pairs under fitted models, and batch operations.
"""

import jax.numpy as jnp
import numpy as np

from ..core.params import tkf91_trans, S, M, I, D, E
from ..core.ctmc import transition_matrix
from ..dp.hmm import forward_2d, viterbi_2d, safe_log
from .em import em_single_pair
from ..models.compiled import TKF91Model


def fit_pairwise(x, y, Q, pi, init_ins=0.10, init_del=0.20, init_t=1.0,
                 n_iter=20, verbose=False):
    """Fit TKF91 indel parameters to a single sequence pair.

    Uses exact closed-form M-step via CompiledModel protocol.

    Args:
        x, y: integer sequence arrays (ancestor, descendant)
        Q: substitution rate matrix
        pi: equilibrium frequencies
        init_ins, init_del: initial indel rate guesses
        init_t: branch length (fixed during EM)
        n_iter: number of EM iterations
        verbose: print per-iteration info

    Returns:
        dict with keys: ins_rate, del_rate, t, Q, log_probs, final_log_prob
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    model = TKF91Model()
    params = {
        'ins_rate': init_ins, 'del_rate': init_del, 't': init_t,
        'Q': Q, 'pi': pi,
    }

    result = em_single_pair(model, params, x, y,
                            n_iter=n_iter, verbose=verbose)

    return {
        "ins_rate": result.params['ins_rate'],
        "del_rate": result.params['del_rate'],
        "t": init_t,
        "Q": result.params['Q'],
        "log_probs": result.log_probs,
        "final_log_prob": result.log_probs[-1] if result.log_probs else float("nan"),
    }


def score_pairwise(x, y, ins_rate, del_rate, t, Q, pi):
    """Compute log P(x, y | params) via forward algorithm.

    Args:
        x, y: integer sequence arrays
        ins_rate, del_rate, t: TKF91 parameters
        Q: rate matrix
        pi: equilibrium frequencies

    Returns:
        float log-probability
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    sub_matrix = transition_matrix(Q, t)
    tau = tkf91_trans(ins_rate, del_rate, t)
    log_trans = safe_log(tau)
    state_types = jnp.array([S, M, I, D, E])
    log_prob, _ = forward_2d(log_trans, state_types, x, y, sub_matrix, pi)
    return float(log_prob)


def align_pairwise(x, y, ins_rate, del_rate, t, Q, pi):
    """Compute Viterbi alignment for a sequence pair.

    Args:
        x, y: integer sequence arrays
        ins_rate, del_rate, t: TKF91 parameters
        Q: rate matrix
        pi: equilibrium frequencies

    Returns:
        log_prob: float, Viterbi log-probability
        path: list of (i, j, state) tuples
        alignment: list of (x_char_or_None, y_char_or_None) tuples
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    sub_matrix = transition_matrix(Q, t)
    tau = tkf91_trans(ins_rate, del_rate, t)
    log_trans = safe_log(tau)
    state_types = jnp.array([S, M, I, D, E])
    log_prob, path = viterbi_2d(log_trans, state_types, x, y, sub_matrix, pi)

    alignment = []
    for i, j, s in path:
        if s == M and i > 0 and j > 0:
            alignment.append((int(x[i - 1]), int(y[j - 1])))
        elif s == I and j > 0:
            alignment.append((None, int(y[j - 1])))
        elif s == D and i > 0:
            alignment.append((int(x[i - 1]), None))

    return float(log_prob), path, alignment


def fit_branch_length(x, y, ins_rate, del_rate, Q, pi,
                      t_grid=None):
    """Estimate branch length by grid search over t values.

    Args:
        x, y: integer sequence arrays
        ins_rate, del_rate: indel rates
        Q: rate matrix
        pi: equilibrium frequencies
        t_grid: array of t values to evaluate (default: 0.01 to 5.0)

    Returns:
        dict with keys: best_t, log_probs, t_grid
    """
    if t_grid is None:
        t_grid = np.concatenate([
            np.arange(0.01, 0.1, 0.01),
            np.arange(0.1, 1.0, 0.1),
            np.arange(1.0, 5.1, 0.5),
        ])

    log_probs = []
    for t in t_grid:
        lp = score_pairwise(x, y, ins_rate, del_rate, float(t), Q, pi)
        log_probs.append(lp)

    best_idx = np.argmax(log_probs)
    return {
        "best_t": float(t_grid[best_idx]),
        "log_probs": log_probs,
        "t_grid": t_grid,
    }


def fit_all_pairs(sequences, Q, pi, init_ins=0.10, init_del=0.20,
                  init_t=1.0, n_iter=15):
    """Fit TKF91 parameters to all pairs from a dict of sequences.

    Args:
        sequences: dict of {name: integer_array}
        Q, pi: substitution model
        init_ins, init_del, init_t: initial parameter guesses
        n_iter: EM settings

    Returns:
        results: dict of {(name_i, name_j): fit_result}
    """
    names = sorted(sequences.keys())
    results = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            x = sequences[n1]
            y = sequences[n2]
            if len(x) == 0 or len(y) == 0:
                continue
            result = fit_pairwise(x, y, Q, pi, init_ins, init_del,
                                  init_t, n_iter)
            results[(n1, n2)] = result
    return results
