"""Beam-pruned forward/backward algorithms for Pair HMMs.

Implements the beam Inside algorithm from the paper (Section 5) specialized
to regular grammars (Pair HMMs), where it reduces to beam Forward/Backward.

Features:
- Beam pruning: retain only states within log(Delta) of the max at each cell
- MSA envelope constraints: restrict DP to cells near a guide alignment
- Iterative envelope refinement: use output of one round as guide for next

When beam_width=inf and no envelope, reduces to exact forward_2d.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..core.params import S, M, I, D, E
from .hmm import NEG_INF, _pair_hmm_emission


def msa_to_envelope(x_len, y_len, msa_x_row, msa_y_row, band_width, gap_token=-1):
    """Convert an MSA guide alignment to a DP envelope.

    Given two rows of a guide MSA, extract the implied pairwise alignment
    and compute the set of (i, j) cells within band_width of the alignment
    diagonal.

    Args:
        x_len: length of sequence x (ungapped)
        y_len: length of sequence y (ungapped)
        msa_x_row: (L_msa,) aligned row for x (gap_token for gaps)
        msa_y_row: (L_msa,) aligned row for y (gap_token for gaps)
        band_width: half-width of band around guide alignment
        gap_token: gap character value

    Returns:
        envelope: (x_len+1, y_len+1) boolean array, True for allowed cells
    """
    msa_x_row = np.asarray(msa_x_row)
    msa_y_row = np.asarray(msa_y_row)
    L_msa = len(msa_x_row)

    # Build mapping from MSA column to (i, j) position
    # i = number of non-gap chars in x up to this column
    # j = number of non-gap chars in y up to this column
    guide_points = []
    xi, yj = 0, 0
    guide_points.append((0, 0))
    for col in range(L_msa):
        x_has = int(msa_x_row[col]) != gap_token
        y_has = int(msa_y_row[col]) != gap_token
        if x_has:
            xi += 1
        if y_has:
            yj += 1
        guide_points.append((xi, yj))
    guide_points.append((x_len, y_len))

    # Build envelope: all cells (i, j) within band_width of any guide point
    envelope = np.zeros((x_len + 1, y_len + 1), dtype=bool)

    # For efficiency, compute the guide diagonal and band around it
    # At each x position i, find the range of allowed y positions
    for i in range(x_len + 1):
        # Find guide y positions at this x
        y_min_guide = y_len
        y_max_guide = 0
        for gi, gj in guide_points:
            if abs(gi - i) <= band_width:
                y_min_guide = min(y_min_guide, gj)
                y_max_guide = max(y_max_guide, gj)

        j_lo = max(0, y_min_guide - band_width)
        j_hi = min(y_len, y_max_guide + band_width)
        envelope[i, j_lo:j_hi + 1] = True

    # Always include (0,0) and (x_len, y_len)
    envelope[0, 0] = True
    envelope[x_len, y_len] = True

    return envelope


def beam_forward_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi,
                    beam_log_width=np.inf, envelope=None):
    """Beam-pruned forward algorithm for a Pair HMM.

    Like forward_2d but with two optimizations:
    1. Beam pruning: at each cell, only retains states within beam_log_width
       of the maximum forward probability
    2. Envelope constraint: only computes cells where envelope[i,j] is True

    When beam_log_width=inf and envelope=None, equivalent to forward_2d.

    Args:
        log_trans: (n_states, n_states) log transition matrix
        state_types: (n_states,) state type codes
        x_seq: (Lx,) ancestor sequence
        y_seq: (Ly,) descendant sequence
        sub_matrix: (A, A) substitution probability matrix
        pi: (A,) equilibrium distribution
        beam_log_width: log beam width (prune states below max - beam_log_width)
        envelope: optional (Lx+1, Ly+1) boolean array

    Returns:
        log_prob: total log probability
        F: (Lx+1, Ly+1, n_states) forward table
        n_cells_computed: number of cells actually computed
    """
    x_seq = np.asarray(x_seq)
    y_seq = np.asarray(y_seq)
    log_trans = np.asarray(log_trans)
    state_types = np.asarray(state_types)
    sub_matrix = np.asarray(sub_matrix)
    pi_arr = np.asarray(pi)

    Lx = len(x_seq)
    Ly = len(y_seq)
    ns = log_trans.shape[0]
    log_sub = np.log(np.maximum(sub_matrix, 1e-30))
    log_pi = np.log(np.maximum(pi_arr, 1e-30))

    F = np.full((Lx + 1, Ly + 1, ns), NEG_INF)
    F[0, 0, S] = 0.0
    n_cells = 0

    def _emission(st, xi, yj):
        if st == M:
            return log_pi[xi] + log_sub[xi, yj]
        elif st == I:
            return log_pi[yj]
        elif st == D:
            return log_pi[xi]
        return 0.0

    # Process cells in anti-diagonal order
    for d in range(1, Lx + Ly + 1):
        i_min = max(0, d - Ly)
        i_max = min(d, Lx)

        for i in range(i_min, i_max + 1):
            j = d - i
            if i == 0 and j == 0:
                continue

            # Check envelope constraint
            if envelope is not None and not envelope[i, j]:
                continue

            xi = x_seq[max(i - 1, 0)] if i > 0 else 0
            yj = y_seq[max(j - 1, 0)] if j > 0 else 0
            n_cells += 1

            for k in range(ns):
                st = int(state_types[k])

                if st == M and i > 0 and j > 0:
                    # Check predecessor is in envelope
                    if envelope is not None and not envelope[i - 1, j - 1]:
                        continue
                    pred = F[i - 1, j - 1, :]
                elif st == I and j > 0:
                    if envelope is not None and not envelope[i, j - 1]:
                        continue
                    pred = F[i, j - 1, :]
                elif st == D and i > 0:
                    if envelope is not None and not envelope[i - 1, j]:
                        continue
                    pred = F[i - 1, j, :]
                else:
                    continue

                # logsumexp(pred + log_trans[:, k])
                vals = pred + log_trans[:, k]
                max_val = np.max(vals)
                if max_val > NEG_INF + 100:
                    lse = max_val + np.log(np.sum(np.exp(vals - max_val)))
                else:
                    lse = NEG_INF

                emit = _emission(st, xi, yj)
                F[i, j, k] = lse + emit

            # Beam pruning: zero out states below threshold
            if beam_log_width < np.inf:
                max_f = np.max(F[i, j, :])
                if max_f > NEG_INF + 100:
                    threshold = max_f - beam_log_width
                    for k in range(ns):
                        if F[i, j, k] < threshold:
                            F[i, j, k] = NEG_INF

    # Terminal transition to E
    e_idx = int(np.argmax(state_types == E))
    final_vals = F[Lx, Ly, :] + log_trans[:, e_idx]
    max_final = np.max(final_vals)
    if max_final > NEG_INF + 100:
        log_prob = max_final + np.log(np.sum(np.exp(final_vals - max_final)))
    else:
        log_prob = NEG_INF

    return log_prob, F, n_cells


def beam_forward_backward_2d(log_trans, state_types, x_seq, y_seq,
                              sub_matrix, pi,
                              beam_log_width=np.inf, envelope=None):
    """Beam-pruned forward-backward for a Pair HMM.

    Runs beam forward, then backward restricted to cells that survived
    the forward beam. Returns posterior state probabilities.

    Args:
        Same as beam_forward_2d

    Returns:
        log_prob: total log probability
        posteriors: (Lx+1, Ly+1, n_states) posterior probabilities
        F: forward table
    """
    x_seq = np.asarray(x_seq)
    y_seq = np.asarray(y_seq)
    log_trans = np.asarray(log_trans)
    state_types = np.asarray(state_types)
    sub_matrix = np.asarray(sub_matrix)
    pi_arr = np.asarray(pi)

    Lx = len(x_seq)
    Ly = len(y_seq)
    ns = log_trans.shape[0]
    log_sub = np.log(np.maximum(sub_matrix, 1e-30))
    log_pi = np.log(np.maximum(pi_arr, 1e-30))

    # Forward pass
    log_prob, F, n_cells = beam_forward_2d(
        log_trans, state_types, x_seq, y_seq, sub_matrix, pi,
        beam_log_width, envelope)

    if log_prob <= NEG_INF + 100:
        posteriors = np.zeros((Lx + 1, Ly + 1, ns))
        return log_prob, posteriors, F

    # Build active cell mask from forward pass
    active = np.any(F > NEG_INF + 100, axis=2)

    # Backward pass (restricted to active cells)
    B = np.full((Lx + 1, Ly + 1, ns), NEG_INF)
    e_idx = int(np.argmax(state_types == E))
    B[Lx, Ly, :] = log_trans[:, e_idx]

    def _emission(st, xi, yj):
        if st == M:
            return log_pi[xi] + log_sub[xi, yj]
        elif st == I:
            return log_pi[yj]
        elif st == D:
            return log_pi[xi]
        return 0.0

    # Process in reverse anti-diagonal order
    for d in range(Lx + Ly, 0, -1):
        i_min = max(0, d - Ly)
        i_max = min(d, Lx)

        for i in range(i_min, i_max + 1):
            j = d - i
            if not active[i, j]:
                continue

            # For each state k at (i,j), compute backward value
            # B[i,j,k] = logsumexp over successors s:
            #   log_trans[k,s] + emission(s, i', j') + B[i',j',s]
            for k in range(ns):
                # Successors: states that can follow k
                total = NEG_INF
                for s in range(ns):
                    st_s = int(state_types[s])
                    if st_s == M:
                        ni, nj = i + 1, j + 1
                    elif st_s == I:
                        ni, nj = i, j + 1
                    elif st_s == D:
                        ni, nj = i + 1, j
                    else:
                        continue

                    if ni > Lx or nj > Ly:
                        continue
                    if not active[ni, nj]:
                        continue

                    xi = x_seq[max(ni - 1, 0)] if ni > 0 else 0
                    yj = y_seq[max(nj - 1, 0)] if nj > 0 else 0
                    emit = _emission(st_s, xi, yj)

                    val = log_trans[k, s] + emit + B[ni, nj, s]
                    if val > NEG_INF + 100:
                        if total > NEG_INF + 100:
                            total = np.logaddexp(total, val)
                        else:
                            total = val

                if total > NEG_INF + 100:
                    if B[i, j, k] > NEG_INF + 100:
                        B[i, j, k] = np.logaddexp(B[i, j, k], total)
                    else:
                        B[i, j, k] = total

    # Posteriors: P(state=k at (i,j)) = exp(F[i,j,k] + B[i,j,k] - log_prob)
    posteriors = np.zeros((Lx + 1, Ly + 1, ns))
    for i in range(Lx + 1):
        for j in range(Ly + 1):
            if active[i, j]:
                log_post = F[i, j, :] + B[i, j, :] - log_prob
                posteriors[i, j, :] = np.exp(np.minimum(log_post, 0))

    return log_prob, posteriors, F


def iterative_beam_refinement(log_trans, state_types, x_seq, y_seq,
                               sub_matrix, pi,
                               beam_log_width=10.0, initial_band=5,
                               n_iterations=3, band_shrink=0.7):
    """Iterative beam refinement with tightening envelope.

    Starts with a wide band, runs beam forward to find the high-probability
    region, then uses the result to construct a tighter envelope for the
    next iteration.

    Args:
        log_trans, state_types, x_seq, y_seq, sub_matrix, pi: HMM params
        beam_log_width: log beam width for pruning
        initial_band: initial band width around the diagonal
        n_iterations: number of refinement iterations
        band_shrink: factor to shrink band each iteration

    Returns:
        log_prob: final log probability
        F: final forward table
        envelope: final envelope used
    """
    Lx = len(x_seq)
    Ly = len(y_seq)

    # Initial envelope: band around the diagonal
    envelope = np.zeros((Lx + 1, Ly + 1), dtype=bool)
    for i in range(Lx + 1):
        # Expected j on diagonal
        j_center = int(round(i * Ly / max(Lx, 1)))
        j_lo = max(0, j_center - initial_band)
        j_hi = min(Ly, j_center + initial_band)
        envelope[i, j_lo:j_hi + 1] = True

    band = initial_band
    log_prob = NEG_INF
    F = None

    for iteration in range(n_iterations):
        log_prob, F, n_cells = beam_forward_2d(
            log_trans, state_types, x_seq, y_seq, sub_matrix, pi,
            beam_log_width, envelope)

        if log_prob <= NEG_INF + 100:
            # Expand envelope and retry
            band = int(band * 2)
            envelope = np.zeros((Lx + 1, Ly + 1), dtype=bool)
            for i in range(Lx + 1):
                j_center = int(round(i * Ly / max(Lx, 1)))
                j_lo = max(0, j_center - band)
                j_hi = min(Ly, j_center + band)
                envelope[i, j_lo:j_hi + 1] = True
            continue

        if iteration < n_iterations - 1:
            # Build new envelope from active cells
            active = np.any(F > NEG_INF + 100, axis=2)
            new_band = max(2, int(band * band_shrink))
            new_envelope = np.zeros((Lx + 1, Ly + 1), dtype=bool)

            for i in range(Lx + 1):
                for j in range(Ly + 1):
                    if active[i, j]:
                        # Expand around active cells
                        i_lo = max(0, i - new_band)
                        i_hi = min(Lx, i + new_band)
                        j_lo = max(0, j - new_band)
                        j_hi = min(Ly, j + new_band)
                        new_envelope[i_lo:i_hi + 1, j_lo:j_hi + 1] = True

            envelope = new_envelope
            band = new_band

    return log_prob, F, envelope


def beam_viterbi_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi,
                    envelope=None):
    """Band-constrained Viterbi for a Pair HMM.

    Like viterbi_2d but restricted to cells within the envelope.

    Args:
        log_trans, state_types, x_seq, y_seq, sub_matrix, pi: HMM params
        envelope: optional (Lx+1, Ly+1) boolean constraint

    Returns:
        log_prob: log probability of best path
        path: list of (i, j, state) tuples
    """
    x_seq = np.asarray(x_seq)
    y_seq = np.asarray(y_seq)
    log_trans = np.asarray(log_trans)
    state_types = np.asarray(state_types)
    sub_matrix = np.asarray(sub_matrix)
    pi_arr = np.asarray(pi)

    Lx = len(x_seq)
    Ly = len(y_seq)
    ns = log_trans.shape[0]
    log_sub = np.log(np.maximum(sub_matrix, 1e-30))
    log_pi = np.log(np.maximum(pi_arr, 1e-30))

    V = np.full((Lx + 1, Ly + 1, ns), NEG_INF)
    V[0, 0, S] = 0.0
    TB = np.zeros((Lx + 1, Ly + 1, ns, 3), dtype=np.int32)

    def _emission(st, xi, yj):
        if st == M:
            return log_pi[xi] + log_sub[xi, yj]
        elif st == I:
            return log_pi[yj]
        elif st == D:
            return log_pi[xi]
        return 0.0

    for d in range(1, Lx + Ly + 1):
        i_min = max(0, d - Ly)
        i_max = min(d, Lx)
        for i in range(i_min, i_max + 1):
            j = d - i
            if i == 0 and j == 0:
                continue
            if envelope is not None and not envelope[i, j]:
                continue

            xi = x_seq[max(i - 1, 0)] if i > 0 else 0
            yj = y_seq[max(j - 1, 0)] if j > 0 else 0

            for k in range(ns):
                st = int(state_types[k])
                if st == M and i > 0 and j > 0:
                    if envelope is not None and not envelope[i - 1, j - 1]:
                        continue
                    scores = V[i - 1, j - 1, :] + log_trans[:, k]
                    best = int(np.argmax(scores))
                    V[i, j, k] = scores[best] + _emission(st, xi, yj)
                    TB[i, j, k] = [i - 1, j - 1, best]
                elif st == I and j > 0:
                    if envelope is not None and not envelope[i, j - 1]:
                        continue
                    scores = V[i, j - 1, :] + log_trans[:, k]
                    best = int(np.argmax(scores))
                    V[i, j, k] = scores[best] + _emission(st, xi, yj)
                    TB[i, j, k] = [i, j - 1, best]
                elif st == D and i > 0:
                    if envelope is not None and not envelope[i - 1, j]:
                        continue
                    scores = V[i - 1, j, :] + log_trans[:, k]
                    best = int(np.argmax(scores))
                    V[i, j, k] = scores[best] + _emission(st, xi, yj)
                    TB[i, j, k] = [i - 1, j, best]

    # Terminal
    e_idx = int(np.argmax(state_types == E))
    final_scores = V[Lx, Ly, :] + log_trans[:, e_idx]
    best_final = int(np.argmax(final_scores))
    log_prob = float(final_scores[best_final])

    # Traceback
    path = []
    ci, cj, ck = Lx, Ly, best_final
    while ci > 0 or cj > 0:
        path.append((ci, cj, ck))
        pi_, pj_, pk_ = TB[ci, cj, ck]
        ci, cj, ck = int(pi_), int(pj_), int(pk_)
    path.append((0, 0, int(S)))
    path.reverse()

    return log_prob, path


def beam_sample_traceback(log_trans, state_types, F, log_prob, x_seq, y_seq,
                          sub_matrix, pi, rng, envelope=None):
    """Stochastic traceback from a beam-pruned forward table.

    Samples an alignment path proportional to its posterior probability,
    restricted to cells that are active in the forward table.

    Args:
        log_trans: (n_states, n_states) log transition matrix
        state_types: (n_states,) state type codes
        F: (Lx+1, Ly+1, n_states) forward table from beam_forward_2d
        log_prob: total log probability
        x_seq, y_seq: sequences
        sub_matrix, pi: emission parameters
        rng: numpy RandomState for sampling
        envelope: optional (Lx+1, Ly+1) boolean constraint

    Returns:
        path: list of (i, j, state) tuples from start to end
    """
    x_seq = np.asarray(x_seq)
    y_seq = np.asarray(y_seq)
    log_trans = np.asarray(log_trans)
    state_types = np.asarray(state_types)

    Lx = len(x_seq)
    Ly = len(y_seq)
    ns = log_trans.shape[0]

    # Sample terminal state at (Lx, Ly)
    e_idx = int(np.argmax(state_types == E))
    terminal_scores = F[Lx, Ly, :] + log_trans[:, e_idx]
    max_ts = np.max(terminal_scores)
    if max_ts <= NEG_INF + 100:
        return []
    terminal_probs = np.exp(terminal_scores - max_ts)
    terminal_probs /= terminal_probs.sum()
    ck = rng.choice(ns, p=terminal_probs)

    path = []
    ci, cj = Lx, Ly

    while ci > 0 or cj > 0:
        path.append((ci, cj, int(ck)))
        st = int(state_types[ck])

        if st == M and ci > 0 and cj > 0:
            pi_, pj_ = ci - 1, cj - 1
        elif st == I and cj > 0:
            pi_, pj_ = ci, cj - 1
        elif st == D and ci > 0:
            pi_, pj_ = ci - 1, cj
        else:
            break

        if envelope is not None and not envelope[pi_, pj_]:
            break

        pred_scores = F[pi_, pj_, :] + log_trans[:, ck]
        max_ps = np.max(pred_scores)
        if max_ps <= NEG_INF + 100:
            break
        pred_probs = np.exp(pred_scores - max_ps)
        pred_probs /= pred_probs.sum()
        pk = rng.choice(ns, p=pred_probs)

        ci, cj, ck = pi_, pj_, pk

    path.append((0, 0, int(S)))
    path.reverse()
    return path
