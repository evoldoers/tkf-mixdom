"""Singlet HMM forward algorithm for scoring single sequences.

Given a MixDom singlet HMM (N domain states with domain-specific
emission distributions), computes log P(sequence) via forward DP.

Usage:
    from tkfmixdom.jax.dp.singlet_forward import singlet_log_prob
    lp = singlet_log_prob(seq, sing_start, sing_trans, sing_end, pis)
"""

import numpy as np


def _logsumexp_rows(a):
    """logsumexp along axis=0 of a 2D array, pure numpy."""
    mx = np.max(a, axis=0)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return mx + np.log(np.sum(np.exp(a - mx[None, :]), axis=0))


def singlet_log_prob(seq, sing_start, sing_trans, sing_end, pis):
    """Compute log P(sequence) under the singlet HMM.

    The singlet HMM has N emitting states (one per domain type).
    State n emits character c with probability pis[n, c].
    Transitions: start -> n with prob sing_start[n],
                 n -> m with prob sing_trans[n, m],
                 n -> end with prob sing_end[n].

    Args:
        seq: (L,) integer array of character indices
        sing_start: (N,) start probabilities
        sing_trans: (N, N) transition matrix between emitting states
        sing_end: (N,) end probabilities
        pis: (N, AA) emission distributions per domain

    Returns:
        log P(sequence) scalar
    """
    seq = np.asarray(seq, dtype=int)
    L = len(seq)

    if L == 0:
        p_empty = 1.0 - np.sum(sing_start)
        return float(np.log(max(p_empty, 1e-30)))

    # Log-space forward
    log_start = np.log(np.maximum(np.asarray(sing_start, dtype=np.float64), 1e-300))
    log_trans = np.log(np.maximum(np.asarray(sing_trans, dtype=np.float64), 1e-300))
    log_end = np.log(np.maximum(np.asarray(sing_end, dtype=np.float64), 1e-300))
    log_pis = np.log(np.maximum(np.asarray(pis, dtype=np.float64), 1e-300))

    # Precompute emission log-probs for the sequence: (L, N)
    log_emit = log_pis[:, seq].T  # (L, N)

    # Forward pass (vectorized over states)
    alpha = log_start + log_emit[0]  # (N,)

    for t in range(1, L):
        # alpha[:, None] + log_trans is (N_from, N_to)
        # max/logsumexp over axis 0 -> (N_to,)
        a = alpha[:, None] + log_trans
        mx = np.max(a, axis=0)
        alpha = mx + np.log(np.sum(np.exp(a - mx[None, :]), axis=0)) + log_emit[t]

    # Terminate
    final = alpha + log_end
    mx = np.max(final)
    return float(mx + np.log(np.sum(np.exp(final - mx))))


def singlet_log_prob_batch(seqs_int, sing_start, sing_trans, sing_end, pis):
    """Score a batch of integer-encoded sequences.

    Args:
        seqs_int: list of (L_i,) integer arrays
        sing_start, sing_trans, sing_end, pis: HMM parameters

    Returns:
        list of log P(seq) floats
    """
    return [singlet_log_prob(s, sing_start, sing_trans, sing_end, pis)
            for s in seqs_int]
