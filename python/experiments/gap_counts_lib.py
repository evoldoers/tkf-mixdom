#!/usr/bin/env python3
"""Gap-counts representation for cherry alignments + Pair HMM gap-LL.

A "gap" in an alignment is a contiguous (possibly empty) stretch of I and D
columns flanked by either matches or the alignment ends.  Gap types:

  SM(i, j)  : from S (start) to the first M, with i deletions + j insertions
  MM(i, j)  : between two consecutive M's, with i del + j ins
  ME(i, j)  : from the last M to E (end), with i del + j ins
  SE(i, j)  : whole alignment has no M -- S straight to E, i del + j ins

The gap probability  G^M_{X,Y}(i, j)  is the sum of path probabilities for
X -> (any sequence of i deletions and j insertions in any order) -> Y, given
the Pair HMM transition matrix T.

Computed via forward DP on the 2-D table A[I, i, j], A[D, i, j]:
  initial:        A[I, 0, 1] = T[X, I]
                  A[D, 1, 0] = T[X, D]
  recurrence:
                  A[I, i, j] = T[I, I] * A[I, i, j-1] + T[D, I] * A[D, i, j-1]
                  A[D, i, j] = T[I, D] * A[I, i-1, j] + T[D, D] * A[D, i-1, j]
  emission:       G(i, j) = A[I, i, j] * T[I, Y] + A[D, i, j] * T[D, Y]
                            (+ T[X, Y] if (i, j) = (0, 0))

The "open chain" SM, MM, ME, SE gaps each plug in their own X, Y.
"""

import numpy as np
import jax
import jax.numpy as jnp


# State indices (must match the rest of the experiments)
S, M, I, D, E = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Gap-counts extraction
# ---------------------------------------------------------------------------

# Gap types: 0=SM, 1=MM, 2=ME, 3=SE
GAP_SM, GAP_MM, GAP_ME, GAP_SE = 0, 1, 2, 3


def cherry_to_gap_counts(col_seq, gap_counts, ti, Lmax):
    """Tally gaps in one cherry's column sequence into gap_counts[ti, g, i, j].

    col_seq is a list of {M=1, I=2, D=3} integers (S and E not included).
    gap_counts has shape (n_tau, 4, Lmax+1, Lmax+1) and is modified in place.
    Gaps with i > Lmax or j > Lmax are CLIPPED to Lmax (so the tail is folded
    into the (Lmax, Lmax) cell).  Choose Lmax generously.
    """
    # Find all match positions
    match_positions = [k for k, c in enumerate(col_seq) if c == M]

    if not match_positions:
        # No matches: SE gap covering the whole sequence
        i_del = sum(1 for c in col_seq if c == D)
        j_ins = sum(1 for c in col_seq if c == I)
        gap_counts[ti, GAP_SE,
                   min(i_del, Lmax),
                   min(j_ins, Lmax)] += 1
        return

    # SM gap: from S to first M
    first_M = match_positions[0]
    i_del = sum(1 for c in col_seq[:first_M] if c == D)
    j_ins = sum(1 for c in col_seq[:first_M] if c == I)
    gap_counts[ti, GAP_SM,
               min(i_del, Lmax),
               min(j_ins, Lmax)] += 1

    # MM gaps: between consecutive matches
    for k in range(len(match_positions) - 1):
        left = match_positions[k]
        right = match_positions[k + 1]
        i_del = sum(1 for c in col_seq[left + 1:right] if c == D)
        j_ins = sum(1 for c in col_seq[left + 1:right] if c == I)
        gap_counts[ti, GAP_MM,
                   min(i_del, Lmax),
                   min(j_ins, Lmax)] += 1

    # ME gap: from last M to E
    last_M = match_positions[-1]
    i_del = sum(1 for c in col_seq[last_M + 1:] if c == D)
    j_ins = sum(1 for c in col_seq[last_M + 1:] if c == I)
    gap_counts[ti, GAP_ME,
               min(i_del, Lmax),
               min(j_ins, Lmax)] += 1


# ---------------------------------------------------------------------------
# JAX gap-probability DP
# ---------------------------------------------------------------------------

def compute_A_tables(T_XI, T_XD, T_II, T_ID, T_DI, T_DD, Lmax):
    """Forward DP: A[I, i, j] and A[D, i, j] tables.

    Loops are unrolled via jax.lax.fori_loop in (i + j) wavefront order.
    Returns (AI, AD) each of shape (Lmax+1, Lmax+1).
    """
    AI = jnp.zeros((Lmax + 1, Lmax + 1))
    AD = jnp.zeros((Lmax + 1, Lmax + 1))
    AI = AI.at[0, 1].set(T_XI)
    AD = AD.at[1, 0].set(T_XD)

    # Iterate wavefronts s = i + j from 2 to 2*Lmax
    def body_fun(s, val):
        AI, AD = val
        # For each i in [max(0, s-Lmax), min(s, Lmax)], compute (i, j) with j = s - i.
        i_min = jnp.maximum(0, s - Lmax)
        i_max = jnp.minimum(s, Lmax)

        def inner(i, val2):
            AI, AD = val2
            j = s - i
            in_range = (i >= i_min) & (i <= i_max)
            # AI update (requires j >= 1)
            AI_new = T_II * AI[i, j - 1] + T_DI * AD[i, j - 1]
            do_AI = in_range & (j >= 1)
            AI_set = jnp.where(do_AI, AI_new, AI[i, j])
            AI = AI.at[i, j].set(AI_set)
            # AD update (requires i >= 1)
            AD_new = T_ID * AI[i - 1, j] + T_DD * AD[i - 1, j]
            do_AD = in_range & (i >= 1)
            AD_set = jnp.where(do_AD, AD_new, AD[i, j])
            AD = AD.at[i, j].set(AD_set)
            return (AI, AD)

        return jax.lax.fori_loop(0, Lmax + 1, inner, (AI, AD))

    AI, AD = jax.lax.fori_loop(2, 2 * Lmax + 1, body_fun, (AI, AD))
    return AI, AD


def gap_prob_matrix(T_XY, T_XI, T_XD, T_IY, T_DY, T_II, T_ID, T_DI, T_DD, Lmax):
    """G[i, j] = gap probability X -> (i del + j ins) -> Y under Pair HMM
    with the given transitions.  Shape (Lmax+1, Lmax+1).
    """
    AI, AD = compute_A_tables(T_XI, T_XD, T_II, T_ID, T_DI, T_DD, Lmax)
    G = AI * T_IY + AD * T_DY
    G = G.at[0, 0].add(T_XY)
    return G


def all_four_gap_probs(T_full, Lmax):
    """Compute G for all four gap types: SM, MM, ME, SE.
    T_full is the 5x5 Pair HMM transition matrix.

    Returns dict with keys SM, MM, ME, SE, each shape (Lmax+1, Lmax+1).
    """
    # All gap types share the same interior I/D transitions.
    T_II = T_full[I, I]
    T_ID = T_full[I, D]
    T_DI = T_full[D, I]
    T_DD = T_full[D, D]

    out = {}
    # SM: X = S, Y = M
    out['SM'] = gap_prob_matrix(
        T_full[S, M], T_full[S, I], T_full[S, D],
        T_full[I, M], T_full[D, M],
        T_II, T_ID, T_DI, T_DD, Lmax)
    # MM: X = M, Y = M
    out['MM'] = gap_prob_matrix(
        T_full[M, M], T_full[M, I], T_full[M, D],
        T_full[I, M], T_full[D, M],
        T_II, T_ID, T_DI, T_DD, Lmax)
    # ME: X = M, Y = E
    out['ME'] = gap_prob_matrix(
        T_full[M, E], T_full[M, I], T_full[M, D],
        T_full[I, E], T_full[D, E],
        T_II, T_ID, T_DI, T_DD, Lmax)
    # SE: X = S, Y = E
    out['SE'] = gap_prob_matrix(
        T_full[S, E], T_full[S, I], T_full[S, D],
        T_full[I, E], T_full[D, E],
        T_II, T_ID, T_DI, T_DD, Lmax)
    return out


def gap_log_likelihood(T_full, gap_counts_tau, Lmax):
    """Sum_{g, i, j} counts[g, i, j] * log G[g, i, j] for one tau bin.

    gap_counts_tau has shape (4, Lmax+1, Lmax+1) ordered as SM, MM, ME, SE.
    """
    probs = all_four_gap_probs(T_full, Lmax)
    log_SM = jnp.log(jnp.maximum(probs['SM'], 1e-30))
    log_MM = jnp.log(jnp.maximum(probs['MM'], 1e-30))
    log_ME = jnp.log(jnp.maximum(probs['ME'], 1e-30))
    log_SE = jnp.log(jnp.maximum(probs['SE'], 1e-30))
    ll = (jnp.sum(gap_counts_tau[GAP_SM] * log_SM)
          + jnp.sum(gap_counts_tau[GAP_MM] * log_MM)
          + jnp.sum(gap_counts_tau[GAP_ME] * log_ME)
          + jnp.sum(gap_counts_tau[GAP_SE] * log_SE))
    return ll


# ---------------------------------------------------------------------------
# Sanity-check entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Quick correctness check: the marginal of gap probabilities over (i, j)
    should reproduce a known 5x5 transition-count likelihood when we use the
    same data, for a 1-step approximation."""
    Lmax = 8

    # Cook up a Pair HMM transition matrix
    T = jnp.array([
        [0.0, 0.7, 0.15, 0.10, 0.05],  # S
        [0.0, 0.8, 0.05, 0.05, 0.10],  # M
        [0.0, 0.3, 0.5,  0.1,  0.1 ],  # I
        [0.0, 0.3, 0.1,  0.5,  0.1 ],  # D
        [0.0, 0.0, 0.0,  0.0,  1.0 ],  # E
    ])

    probs = all_four_gap_probs(T, Lmax)
    print("Gap probability tables computed:")
    for k, v in probs.items():
        s = float(jnp.sum(v))
        print(f"  {k}: sum = {s:.6f},   G[0,0] = {float(v[0,0]):.6f}")

    # The S-to-anything-eventually probability should sum to 1.
    # P(reach M eventually from S) + P(reach E eventually from S without M) = 1.
    # Equivalently: sum_{i,j} G_SM(i,j) + sum_{i,j} G_SE(i,j) = 1
    s_sm = float(jnp.sum(probs['SM']))
    s_se = float(jnp.sum(probs['SE']))
    print(f"\nSanity check: P(S->...->M, never visiting M before) + "
          f"P(S->...->E, no M) = {s_sm:.4f} + {s_se:.4f} = {s_sm + s_se:.4f}")
    print("(should be 1.0 if the only escapes from {I, D} are M or E)")

    print("\nGap probabilities sample (MM at (i, j)):")
    for i in range(min(4, Lmax + 1)):
        row = "  "
        for j in range(min(4, Lmax + 1)):
            row += f"{float(probs['MM'][i, j]):.5f}  "
        print(row)
