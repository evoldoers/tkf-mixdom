"""Generic HMM/Pair HMM data structures and construction from TKF models.

An HMM is represented as:
- trans: (n_states, n_states) log-transition matrix
- emit: (n_states, n_emit) log-emission matrix (or None for silent states)
- state_types: (n_states,) array of state type codes
  S=0 (start, silent), M=1 (match, emits pair), I=2 (insert, emits y),
  D=3 (delete, emits x), E=4 (end, silent)

For Pair HMMs, emissions depend on state type:
- M states emit (x, y) pairs
- I states emit y only
- D states emit x only
- S, E are silent
"""

import jax.numpy as jnp
from ..core import params
from ..dp.hmm import safe_log


# State type codes
S, M, I, D, E = 0, 1, 2, 3, 4


def make_tkf91_pair_hmm(ins_rate, del_rate, t, Q, pi, condition_geometric=False):
    """Construct TKF91 Pair HMM.

    Args:
        ins_rate, del_rate: indel rates
        t: evolutionary time
        Q: substitution rate matrix (alphabet_size, alphabet_size)
        pi: equilibrium distribution (alphabet_size,)
        condition_geometric: if True, factor out κ^i·(1-κ) from the
            transition matrix. The DP then computes log P(x,y | |ancestor|=i)
            instead of the joint log P(x,y). Required when κ ≈ 1.

    Returns:
        trans: (5, 5) log-transition matrix
        state_types: (5,) state type codes
        sub_matrix: (alphabet_size, alphabet_size) substitution probability matrix exp(Qt)
        pi: equilibrium distribution
    """
    from ..core.ctmc import transition_matrix
    if condition_geometric:
        tau = params.tkf91_trans_cond(ins_rate, del_rate, t)
    else:
        tau = params.tkf91_trans(ins_rate, del_rate, t)
    log_trans = safe_log(tau)
    state_types = jnp.array([S, M, I, D, E])
    sub_matrix = transition_matrix(Q, t)
    return log_trans, state_types, sub_matrix, pi


def make_tkf92_pair_hmm(ins_rate, del_rate, t, ext, Q, pi):
    """Construct TKF92 Pair HMM (joint distribution).

    Use ``varanc_presence.tkf92_wfst_T`` for the conditional WFST
    representation of P(descendant | ancestor).
    """
    from ..core.ctmc import transition_matrix
    tau = params.tkf92_trans(ins_rate, del_rate, t, ext)
    log_trans = safe_log(tau)
    state_types = jnp.array([S, M, I, D, E])
    sub_matrix = transition_matrix(Q, t)
    return log_trans, state_types, sub_matrix, pi


def make_mixfrag_pair_hmm(ins_rate, del_rate, t, exts, weights, Q, pi):
    """Construct MixFrag (TKF92 fragment-mixture) Pair HMM (joint distribution).

    State order S, M_1..M_F, I_1..I_F, D_1..D_F, E (F = len(exts)). Emissions
    are fragtype-independent (shared substitution model), so ``state_types``
    maps every M_f -> M, I_f -> I, D_f -> D and the generic Pair-HMM emission
    code (``pair_emission_logprob``) applies unchanged. Reduces to
    ``make_tkf92_pair_hmm`` at F=1, weights=[1].

    Args:
        exts:    (F,) fragtype extension probabilities r_f.
        weights: (F,) fragtype weights w_f (sum to 1).
    Returns:
        (log_trans, state_types, sub_matrix, pi).
    """
    from ..core.ctmc import transition_matrix
    tau = params.mixfrag_trans(ins_rate, del_rate, t, exts, weights)
    log_trans = safe_log(tau)
    F = int(jnp.asarray(exts).shape[0])
    state_types = jnp.array([S] + [M] * F + [I] * F + [D] * F + [E],
                            dtype=jnp.int32)
    sub_matrix = transition_matrix(Q, t)
    return log_trans, state_types, sub_matrix, pi


def make_mixfrag_singlet_hmm(ins_rate, del_rate, exts, weights, pi):
    """Construct MixFrag Singlet (stationary) HMM.

    State order S, I_1..I_F, E; each I_f emits one residue from pi
    (fragtype-independent). Returns (log_trans, state_types, pi).
    """
    tau = params.mixfrag_singlet_trans(ins_rate, del_rate, exts, weights)
    log_trans = safe_log(tau)
    F = int(jnp.asarray(exts).shape[0])
    state_types = jnp.array([S] + [I] * F + [E], dtype=jnp.int32)
    return log_trans, state_types, pi


def pair_emission_logprob(state_type, x_char, y_char, sub_matrix, pi):
    """Log emission probability for a Pair HMM state.

    Args:
        state_type: S=0, M=1, I=2, D=3, E=4
        x_char: ancestor character index (or -1 if none)
        y_char: descendant character index (or -1 if none)
        sub_matrix: (A, A) substitution matrix
        pi: (A,) equilibrium distribution

    Returns:
        log probability of emission
    """
    log_match = jnp.log(pi[x_char] * sub_matrix[x_char, y_char] + 1e-30)
    log_insert = jnp.log(pi[y_char] + 1e-30)
    log_delete = jnp.log(pi[x_char] + 1e-30)

    return jnp.where(
        state_type == M, log_match,
        jnp.where(
            state_type == I, log_insert,
            jnp.where(
                state_type == D, log_delete,
                0.0  # S, E are silent
            )
        )
    )


def pair_emission_matrix(state_types, x_seq, y_seq, sub_matrix, pi):
    """Compute emission log-probabilities for all states at all alignment positions.

    For a Pair HMM, "positions" are (i, j) where i indexes x and j indexes y.
    This function returns per-state emission probabilities that can be
    combined with the position-dependent state usage in the DP.

    Args:
        state_types: (n_states,) state type codes
        x_seq: (Lx,) ancestor sequence (integer indices)
        y_seq: (Ly,) descendant sequence (integer indices)
        sub_matrix: (A, A) substitution probability matrix
        pi: (A,) equilibrium distribution

    Returns:
        match_emit: (Lx, Ly, n_states) log emission for match states
        ins_emit: (Ly, n_states) log emission for insert states
        del_emit: (Lx, n_states) log emission for delete states
    """
    A = pi.shape[0]
    n_states = state_types.shape[0]
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]

    # Match emissions: pi[x] * P(y|x) for each (x_i, y_j)
    log_pi = jnp.log(pi + 1e-30)
    log_sub = jnp.log(sub_matrix + 1e-30)

    # (Lx, Ly) match log probs
    match_lp = log_pi[x_seq][:, None] + log_sub[x_seq][:, None, y_seq]  # wrong shape
    # Actually: for position (i,j), log P = log pi[x[i]] + log sub[x[i], y[j]]
    match_lp = log_pi[x_seq[:, None].repeat(Ly, axis=1)] + log_sub[x_seq[:, None], y_seq[None, :]]

    # Insert emissions: pi[y] for each y_j
    ins_lp = log_pi[y_seq]

    # Delete emissions: pi[x] for each x_i
    del_lp = log_pi[x_seq]

    # Broadcast to (positions, n_states)
    is_match = (state_types == M).astype(jnp.float32)
    is_ins = (state_types == I).astype(jnp.float32)
    is_del = (state_types == D).astype(jnp.float32)

    return match_lp, ins_lp, del_lp
