"""MSA (Multiple Sequence Alignment) likelihood computation.

Computes P(MSA | tree, params) by scanning columns of the alignment.
Each column specifies which sequences have a character (non-gap) at that position.
The likelihood is computed by a product of per-column factors along the tree.

For a pair of sequences (ancestor, descendant), the alignment columns correspond
to the state sequence of the Pair HMM: M (both present), I (descendant only),
D (ancestor only).
"""

import jax
import jax.numpy as jnp

from ..core.params import S, M, I, D, E, N_STATES
from ..dp.hmm import NEG_INF


def alignment_to_states(ancestor_chars, descendant_chars, gap_token=-1):
    """Convert aligned sequences to state sequence.

    Args:
        ancestor_chars: (L,) array of character indices (-1 for gap)
        descendant_chars: (L,) array of character indices (-1 for gap)
        gap_token: value indicating a gap

    Returns:
        states: list of state codes (M, I, D)
        anc_chars: list of ancestor characters (for M, D states)
        desc_chars: list of descendant characters (for M, I states)
    """
    L = ancestor_chars.shape[0]
    states = []
    anc_chars = []
    desc_chars = []

    for pos in range(L):
        anc = int(ancestor_chars[pos])
        desc = int(descendant_chars[pos])
        has_anc = (anc != gap_token)
        has_desc = (desc != gap_token)

        if has_anc and has_desc:
            states.append(M)
            anc_chars.append(anc)
            desc_chars.append(desc)
        elif has_desc:
            states.append(I)
            desc_chars.append(desc)
        elif has_anc:
            states.append(D)
            anc_chars.append(anc)
        # Both gaps: skip (shouldn't happen in valid MSA)

    return states, anc_chars, desc_chars


def pairwise_log_likelihood(log_trans, state_types, states, anc_chars, desc_chars,
                            sub_matrix, pi):
    """Compute log P(alignment | params) for a pairwise alignment.

    Given a fixed alignment (state sequence), computes the probability
    without the DP — just multiplies transition and emission probabilities.

    Args:
        log_trans: (n_states, n_states) log transition matrix
        state_types: (n_states,) state type codes
        states: list of state indices (M=1, I=2, D=3)
        anc_chars: list of ancestor characters (for M, D)
        desc_chars: list of descendant characters (for M, I)
        sub_matrix: (A, A) substitution probability matrix
        pi: (A,) equilibrium distribution

    Returns:
        log_prob: log probability of the alignment
    """
    log_sub = jnp.log(sub_matrix + 1e-30)
    log_pi = jnp.log(pi + 1e-30)

    log_prob = 0.0
    prev_state = S
    ai, di = 0, 0  # indices into anc_chars, desc_chars

    for state in states:
        # Transition probability
        log_prob += log_trans[prev_state, state]

        # Emission probability
        if state == M:
            a, d = anc_chars[ai], desc_chars[di]
            log_prob += log_pi[a] + log_sub[a, d]
            ai += 1
            di += 1
        elif state == I:
            d = desc_chars[di]
            log_prob += log_pi[d]
            di += 1
        elif state == D:
            a = anc_chars[ai]
            log_prob += log_pi[a]
            ai += 1

        prev_state = state

    # Terminal transition
    log_prob += log_trans[prev_state, E]

    return float(log_prob)


def msa_column_log_likelihood(log_trans, state_types, msa, tree_edges,
                              sub_matrices, pi, gap_token=-1):
    """Compute log P(MSA | tree, params) by scanning columns.

    For each edge in the tree, extracts the pairwise alignment from the MSA
    and computes the log-likelihood. The total is the sum over all edges.

    This is the alignment-conditioned likelihood (alignment is fixed/given).

    Args:
        log_trans: (n_edges, n_states, n_states) per-edge log transition matrices
        state_types: (n_states,) state type codes
        msa: (n_seqs, L) integer array of aligned sequences (gap_token for gaps)
        tree_edges: list of (parent_idx, child_idx) pairs
        sub_matrices: (n_edges, A, A) per-edge substitution matrices
        pi: (A,) equilibrium distribution
        gap_token: gap character value

    Returns:
        total_log_prob: sum of per-edge log-likelihoods
    """
    total = 0.0

    for edge_idx, (parent, child) in enumerate(tree_edges):
        anc_seq = msa[parent]
        desc_seq = msa[child]
        states, anc_chars, desc_chars = alignment_to_states(anc_seq, desc_seq, gap_token)

        lp = pairwise_log_likelihood(
            log_trans[edge_idx], state_types, states, anc_chars, desc_chars,
            sub_matrices[edge_idx], pi
        )
        total += lp

    return total
