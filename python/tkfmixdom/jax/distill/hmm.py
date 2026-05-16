"""Order-1 distillation: compress structured HMM/transducer to character-level.

Distills a MixDom-style Pair HMM (with domains, fragments, site classes)
into compact order-1 machines whose transition probabilities depend only
on the most recently emitted characters.

Two distillations:
1. Order-1 HMM: generates single sequences from stationary distribution
2. Order-1 WFST: transducer for ancestor->descendant pairs

The WFST distillation follows §9.4.3 of the paper: compute adjacency
frequencies from the full Pair HMM, then normalize per WFST source state
× context (X, Y) to obtain transducer weights. This is the MLE solution
for the WFST component given a fixed singlet HMM (see §9.4.3 remark).
"""

import jax.numpy as jnp
from ..core.params import S, M, I, D, E


def null_closure(trans, null_states):
    """Compute (I - T_null)^{-1} for null state transitions.

    Args:
        trans: (N, N) transition matrix (linear)
        null_states: list of null state indices

    Returns:
        closure: (len(null_states), len(null_states)) null closure matrix
    """
    n_null = len(null_states)
    if n_null == 0:
        return jnp.zeros((0, 0))
    T_null = trans[jnp.array(null_states)][:, jnp.array(null_states)]
    return jnp.linalg.inv(jnp.eye(n_null) - T_null)


def effective_emit_trans(trans, emit_states, null_states):
    """Effective transition matrix between emitting states, marginalizing nulls.

    T_eff = T_EE + T_EN · (I - T_NN)^{-1} · T_NE

    Args:
        trans: (N, N) transition matrix
        emit_states: list of emitting state indices
        null_states: list of null state indices

    Returns:
        T_eff: (n_emit, n_emit) effective transition matrix
    """
    e_idx = jnp.array(emit_states)
    T_EE = trans[e_idx][:, e_idx]

    if len(null_states) == 0:
        return T_EE

    n_idx = jnp.array(null_states)
    T_EN = trans[e_idx][:, n_idx]
    T_NE = trans[n_idx][:, e_idx]
    closure = null_closure(trans, null_states)

    return T_EE + T_EN @ closure @ T_NE


def distill_singlet_hmm(trans, emit_probs, pi, alphabet_size):
    """Distill structured HMM to order-1 HMM.

    Computes adjacency frequencies f(a, b) from the full HMM and normalizes
    to get order-1 transition probabilities.

    Args:
        trans: (N, N) transition matrix (linear, not log)
        emit_probs: (N, alphabet_size) emission probability for each state
            (0 for silent states)
        pi: (N,) stationary distribution over states (for weighting)
        alphabet_size: size of output alphabet

    Returns:
        order1_trans: (alphabet_size+2, alphabet_size+2) transition matrix
            States: 0=start, 1..A=characters, A+1=end
        order1_emit: deterministic (state i emits character i-1)
    """
    N = trans.shape[0]
    A = alphabet_size

    # Identify emitting vs silent states
    emit_states = []
    null_states = []
    for i in range(N):
        if emit_probs[i].sum() > 1e-30:
            emit_states.append(i)
        else:
            null_states.append(i)

    # Effective transition between emitting states
    T_eff = effective_emit_trans(trans, emit_states, null_states)
    n_emit = len(emit_states)

    # Start -> emitting: effective transition from S through nulls
    # Assuming S is state 0 and E is the last state
    s_idx = 0  # start state index
    e_idx_val = N - 1  # end state index (E)

    # Start -> emit transitions (through nulls)
    s_to_emit = trans[s_idx, jnp.array(emit_states)]
    if len(null_states) > 0:
        n_idx = jnp.array(null_states)
        closure = null_closure(trans, null_states)
        s_to_null = trans[s_idx, n_idx]
        null_to_emit = trans[n_idx][:, jnp.array(emit_states)]
        s_to_emit = s_to_emit + s_to_null @ closure @ null_to_emit

    # Emit -> end transitions (through nulls)
    emit_to_e = trans[jnp.array(emit_states), e_idx_val]
    if len(null_states) > 0:
        emit_to_null = trans[jnp.array(emit_states)][:, n_idx]
        null_to_e = trans[n_idx, e_idx_val]
        emit_to_e = emit_to_e + emit_to_null @ closure @ null_to_e

    # Adjacency frequencies: f(a, b) = Σ_{s1, s2} π_eff(s1) * emit(s1, a) * T_eff(s1, s2) * emit(s2, b)
    # where π_eff is the marginal distribution reaching each emitting state
    emit_p = emit_probs[jnp.array(emit_states)]  # (n_emit, A)

    # f(a, b) = Σ_{s1,s2} emit_p[s1,a] * T_eff[s1,s2] * emit_p[s2,b]
    # = emit_p.T @ T_eff @ emit_p
    f_ab = emit_p.T @ T_eff @ emit_p  # (A, A)

    # Build order-1 transition matrix: states 0=S, 1..A=chars, A+1=E
    order1 = jnp.zeros((A + 2, A + 2))

    # S -> character a: proportional to Σ_s s_to_emit[s] * emit_p[s, a]
    start_probs = s_to_emit @ emit_p  # (A,)
    order1 = order1.at[0, 1:A+1].set(start_probs)

    # S -> E: direct
    s_to_e = trans[s_idx, e_idx_val]
    if len(null_states) > 0:
        s_to_e = s_to_e + s_to_null @ closure @ null_to_e
    order1 = order1.at[0, A+1].set(s_to_e)

    # Character a -> character b: normalize f(a,b)
    for a in range(A):
        row_sum = f_ab[a].sum()
        if row_sum > 1e-30:
            # Also need a -> E probability
            # emit -> E weighted by emission: Σ_s emit_p[s,a] * emit_to_e[s]
            a_to_e = jnp.sum(emit_p[:, a] * emit_to_e)
            total = row_sum + a_to_e
            if total > 1e-30:
                order1 = order1.at[1+a, 1:A+1].set(f_ab[a] / total)
                order1 = order1.at[1+a, A+1].set(a_to_e / total)

    return order1


def _compute_adjacency_frequencies(trans, state_types, sub_matrix, pi,
                                   alphabet_size):
    """Compute all adjacency frequencies from a Pair HMM.

    Returns raw (unnormalized) frequencies for every adjacency type
    described in §9.4.3 of the paper. Each frequency tensor is indexed
    by context characters as follows:

    - Start → *: indexed by destination characters only
    - Source → Dest: indexed by (source_context, dest_characters)
    - For insert sources: ancestor context X is propagated from preceding
      match/delete (for single-state-per-type models, factorizes as pi[X])
    - For delete sources: descendant context Y is propagated from preceding
      match/insert (for single-state-per-type models, factorizes as pi[Y])

    Args:
        trans: (N, N) Pair HMM transition matrix (linear)
        state_types: (N,) state type codes
        sub_matrix: (A, A) substitution probability matrix P(Y|X)
        pi: (A,) equilibrium distribution
        alphabet_size: size of alphabet

    Returns:
        freqs: dict with all adjacency frequency tensors
    """
    N = trans.shape[0]
    A = alphabet_size

    # Identify emitting states by type
    match_states = [i for i in range(N) if state_types[i] == M]
    ins_states = [i for i in range(N) if state_types[i] == I]
    del_states = [i for i in range(N) if state_types[i] == D]
    emit_states = match_states + ins_states + del_states
    null_states = [i for i in range(N)
                   if state_types[i] in (S, E) and i not in emit_states]

    n_match = len(match_states)
    n_ins = len(ins_states)
    n_del = len(del_states)

    # Effective transitions between emitting states
    T_eff = effective_emit_trans(trans, emit_states, null_states)

    # Index helpers: position of each state in emit_states list
    def _eidx(s):
        return emit_states.index(s)

    # Start -> emit (through nulls)
    s_idx = 0
    e_idx_val = N - 1
    s_to_emit = trans[s_idx, jnp.array(emit_states)]
    emit_to_e = trans[jnp.array(emit_states), e_idx_val]
    if len(null_states) > 0:
        n_idx = jnp.array(null_states)
        closure = null_closure(trans, null_states)
        s_to_null = trans[s_idx, n_idx]
        null_to_emit = trans[n_idx][:, jnp.array(emit_states)]
        s_to_emit = s_to_emit + s_to_null @ closure @ null_to_emit
        emit_to_null = trans[jnp.array(emit_states)][:, n_idx]
        null_to_e = trans[n_idx, e_idx_val]
        emit_to_e = emit_to_e + emit_to_null @ closure @ null_to_e

    # --- Start → * ---
    # f_SM(X', Y') = Σ_{s∈match} s_to_emit[s] * P(X',Y'|s)
    f_SM = jnp.zeros((A, A))
    for s in match_states:
        w = s_to_emit[_eidx(s)]
        f_SM += w * jnp.einsum('x,xy->xy', pi, sub_matrix)

    # f_SI(Y') = Σ_{s∈ins} s_to_emit[s] * P(Y'|s)
    f_SI = jnp.zeros((A,))
    for s in ins_states:
        w = s_to_emit[_eidx(s)]
        f_SI += w * pi

    # f_SE = Σ path from start to end without emitting
    f_SE = trans[s_idx, e_idx_val]
    if len(null_states) > 0:
        f_SE = f_SE + float(s_to_null @ closure @ null_to_e)
    f_SE = float(f_SE)

    # --- Match emissions: P(X,Y|s) = pi[X] * sub[X,Y] ---
    # match_emit[X,Y] = Σ_{s∈match} pi[X]*sub[X,Y]  (for single-state: just pi*sub)
    match_emit_xy = jnp.einsum('x,xy->xy', pi, sub_matrix)  # (A, A)

    # --- Match → * ---
    # f_MM(X,Y,X',Y') = Σ_{s1∈M,s2∈M} T_eff[s1,s2] * P(X,Y|s1) * P(X',Y'|s2)
    f_MM = jnp.zeros((A, A, A, A))
    for s1 in match_states:
        for s2 in match_states:
            w = T_eff[_eidx(s1), _eidx(s2)]
            f_MM += w * jnp.einsum('xy,ab->xyab', match_emit_xy, match_emit_xy)

    # f_MI(X,Y,Y') = Σ_{s1∈M,s2∈I} T_eff * P(X,Y|s1) * P(Y'|s2)
    f_MI = jnp.zeros((A, A, A))
    for s1 in match_states:
        for s2 in ins_states:
            w = T_eff[_eidx(s1), _eidx(s2)]
            f_MI += w * jnp.einsum('xy,b->xyb', match_emit_xy, pi)

    # f_MD(X,Y,X') = Σ_{s1∈M,s2∈D} T_eff * P(X,Y|s1) * P(X'|s2)
    f_MD = jnp.zeros((A, A, A))
    for s1 in match_states:
        for s2 in del_states:
            w = T_eff[_eidx(s1), _eidx(s2)]
            f_MD += w * jnp.einsum('xy,a->xya', match_emit_xy, pi)

    # f_ME(X,Y) = Σ_{s∈M} emit_to_e[s] * P(X,Y|s)
    f_ME = jnp.zeros((A, A))
    for s in match_states:
        w = emit_to_e[_eidx(s)]
        f_ME += w * match_emit_xy

    # --- Insert → * (with propagated ancestor X) ---
    # For single-state-per-type: propagated X is pi[X]-distributed
    # f_IM(X,Y,X',Y') = Σ_{s1∈I,s2∈M} T_eff * pi[X] * P(Y|s1) * P(X',Y'|s2)
    f_IM = jnp.zeros((A, A, A, A))
    for s1 in ins_states:
        for s2 in match_states:
            w = T_eff[_eidx(s1), _eidx(s2)]
            f_IM += w * jnp.einsum('x,y,ab->xyab', pi, pi, match_emit_xy)

    # f_II(X,Y,Y') = Σ_{s1∈I,s2∈I} T_eff * pi[X] * P(Y|s1) * P(Y'|s2)
    f_II = jnp.zeros((A, A, A))
    for s1 in ins_states:
        for s2 in ins_states:
            w = T_eff[_eidx(s1), _eidx(s2)]
            f_II += w * jnp.einsum('x,y,b->xyb', pi, pi, pi)

    # f_ID(X,Y,X') = Σ_{s1∈I,s2∈D} T_eff * pi[X] * P(Y|s1) * P(X'|s2)
    f_ID = jnp.zeros((A, A, A))
    for s1 in ins_states:
        for s2 in del_states:
            w = T_eff[_eidx(s1), _eidx(s2)]
            f_ID += w * jnp.einsum('x,y,a->xya', pi, pi, pi)

    # f_IE(X,Y) = Σ_{s∈I} emit_to_e[s] * pi[X] * P(Y|s)
    f_IE = jnp.zeros((A, A))
    for s in ins_states:
        w = emit_to_e[_eidx(s)]
        f_IE += w * jnp.einsum('x,y->xy', pi, pi)

    # --- Delete → * (with propagated descendant Y) ---
    # For single-state-per-type: propagated Y is pi[Y]-distributed
    # f_DM(X,Y,X',Y') = Σ_{s1∈D,s2∈M} T_eff * P(X|s1) * pi[Y] * P(X',Y'|s2)
    f_DM = jnp.zeros((A, A, A, A))
    for s1 in del_states:
        for s2 in match_states:
            w = T_eff[_eidx(s1), _eidx(s2)]
            f_DM += w * jnp.einsum('x,y,ab->xyab', pi, pi, match_emit_xy)

    # f_DD(X,Y,X') = Σ_{s1∈D,s2∈D} T_eff * P(X|s1) * pi[Y] * P(X'|s2)
    f_DD = jnp.zeros((A, A, A))
    for s1 in del_states:
        for s2 in del_states:
            w = T_eff[_eidx(s1), _eidx(s2)]
            f_DD += w * jnp.einsum('x,y,a->xya', pi, pi, pi)

    # f_DI(X,Y,Y') = Σ_{s1∈D,s2∈I} T_eff * P(X|s1) * pi[Y] * P(Y'|s2)
    f_DI = jnp.zeros((A, A, A))
    for s1 in del_states:
        for s2 in ins_states:
            w = T_eff[_eidx(s1), _eidx(s2)]
            f_DI += w * jnp.einsum('x,y,b->xyb', pi, pi, pi)

    # f_DE(X,Y) = Σ_{s∈D} emit_to_e[s] * P(X|s) * pi[Y]
    f_DE = jnp.zeros((A, A))
    for s in del_states:
        w = emit_to_e[_eidx(s)]
        f_DE += w * jnp.einsum('x,y->xy', pi, pi)

    return {
        # Start → *
        'f_SM': f_SM, 'f_SI': f_SI, 'f_SE': f_SE,
        # Match → *
        'f_MM': f_MM, 'f_MI': f_MI, 'f_MD': f_MD, 'f_ME': f_ME,
        # Insert → * (propagated ancestor X)
        'f_IM': f_IM, 'f_II': f_II, 'f_ID': f_ID, 'f_IE': f_IE,
        # Delete → * (propagated descendant Y)
        'f_DM': f_DM, 'f_DD': f_DD, 'f_DI': f_DI, 'f_DE': f_DE,
    }


def distill_pair_hmm(trans, state_types, sub_matrix, pi, alphabet_size):
    """Distill Pair HMM to order-1 WFST (weighted finite state transducer).

    The order-1 transducer has 7 machine states {S, M, I, D, V, W, E}
    where V (post-M/I wait) and W (post-D wait) are waiting states.
    Transition weights depend on (last_anc_char, last_desc_char).

    Computes adjacency frequencies from the Pair HMM and normalizes
    per WFST source state × context (X, Y) following §9.4.3.

    Args:
        trans: (N, N) Pair HMM transition matrix (linear)
        state_types: (N,) state type codes
        sub_matrix: (A, A) substitution probability matrix
        pi: (A,) equilibrium distribution
        alphabet_size: size of alphabet

    Returns:
        wfst: dict with normalized transducer weight arrays:
            Start transitions:
                'p_S_waitm': scalar
                'p_S_ins': (A,) per output nucleotide Y'
                'p_S_end': scalar
            Wait-after-match/insert (V) transitions (consuming input X'):
                'p_V_mat': (A, A, A, A) [X, Y, X', Y']
                'p_V_del': (A, A, A) [X, Y, X']
            Wait-after-delete (W) transitions (consuming input X'):
                'p_W_mat': (A, A, A, A) [X, Y, X', Y']
                'p_W_del': (A, A, A) [X, Y, X']
            Post-match transitions:
                'p_mat_waitm': (A, A) [X, Y]
                'p_mat_ins': (A, A, A) [X, Y, Y']
                'p_mat_end': (A, A) [X, Y]
            Post-insert transitions:
                'p_ins_waitm': (A, A) [X, Y]
                'p_ins_ins': (A, A, A) [X, Y, Y']
                'p_ins_end': (A, A) [X, Y]
            Post-delete transitions:
                'p_del_waitd': (A, A) [X, Y]
                'p_del_ins': (A, A, A) [X, Y, Y']
                'p_del_end': (A, A) [X, Y]
        freqs: dict with raw adjacency frequency tensors (for diagnostics)
    """
    A = alphabet_size
    freqs = _compute_adjacency_frequencies(trans, state_types, sub_matrix,
                                           pi, alphabet_size)

    # --- Normalization (§9.4.3) ---

    # Start transitions: normalize over {waitm, ins(Y'), end}
    Z_S = freqs['f_SM'].sum() + freqs['f_SI'].sum() + freqs['f_SE']
    if Z_S > 1e-30:
        p_S_waitm = float(freqs['f_SM'].sum() / Z_S)
        p_S_ins = freqs['f_SI'] / Z_S
        p_S_end = float(freqs['f_SE'] / Z_S)
    else:
        p_S_waitm = 1.0
        p_S_ins = jnp.zeros(A)
        p_S_end = 0.0

    # Wait-after-match/insert (V): aggregate mat + ins sources
    # f^{·→mat}(X,Y,X',Y') = f_MM + f_IM
    # f^{·→del}(X,Y,X') = f_MD + f_ID
    f_dot_mat = freqs['f_MM'] + freqs['f_IM']  # (A, A, A, A)
    f_dot_del = freqs['f_MD'] + freqs['f_ID']  # (A, A, A)
    # Denominator: sum over X', Y' for mat + sum over X' for del
    Z_V = f_dot_mat.sum(axis=(2, 3)) + f_dot_del.sum(axis=2)  # (A, A)
    Z_V_safe = jnp.where(Z_V > 1e-30, Z_V, 1.0)
    p_V_mat = f_dot_mat / Z_V_safe[:, :, None, None]
    p_V_del = f_dot_del / Z_V_safe[:, :, None]

    # Wait-after-delete (W): only from delete sources
    # f^{del→mat}(X,Y,X',Y'), f^{del→del}(X,Y,X')
    Z_W = freqs['f_DM'].sum(axis=(2, 3)) + freqs['f_DD'].sum(axis=2)  # (A, A)
    Z_W_safe = jnp.where(Z_W > 1e-30, Z_W, 1.0)
    p_W_mat = freqs['f_DM'] / Z_W_safe[:, :, None, None]
    p_W_del = freqs['f_DD'] / Z_W_safe[:, :, None]

    # Post-match transitions: normalize {waitm, ins(Y'), end}
    # waitm numerator: Σ_{X',Y'} f_MM(X,Y,X',Y') + Σ_{X'} f_MD(X,Y,X')
    n_mat_waitm = freqs['f_MM'].sum(axis=(2, 3)) + freqs['f_MD'].sum(axis=2)
    n_mat_ins = freqs['f_MI']  # (A, A, A): [X, Y, Y']
    n_mat_end = freqs['f_ME']  # (A, A)
    Z_mat = n_mat_waitm + n_mat_ins.sum(axis=2) + n_mat_end  # (A, A)
    Z_mat_safe = jnp.where(Z_mat > 1e-30, Z_mat, 1.0)
    p_mat_waitm = n_mat_waitm / Z_mat_safe
    p_mat_ins = n_mat_ins / Z_mat_safe[:, :, None]
    p_mat_end = n_mat_end / Z_mat_safe

    # Post-insert transitions: analogous using f_I* frequencies
    n_ins_waitm = freqs['f_IM'].sum(axis=(2, 3)) + freqs['f_ID'].sum(axis=2)
    n_ins_ins = freqs['f_II']  # (A, A, A)
    n_ins_end = freqs['f_IE']  # (A, A)
    Z_ins = n_ins_waitm + n_ins_ins.sum(axis=2) + n_ins_end  # (A, A)
    Z_ins_safe = jnp.where(Z_ins > 1e-30, Z_ins, 1.0)
    p_ins_waitm = n_ins_waitm / Z_ins_safe
    p_ins_ins = n_ins_ins / Z_ins_safe[:, :, None]
    p_ins_end = n_ins_end / Z_ins_safe

    # Post-delete transitions: {waitd, ins(Y'), end}
    n_del_waitd = freqs['f_DM'].sum(axis=(2, 3)) + freqs['f_DD'].sum(axis=2)
    n_del_ins = freqs['f_DI']  # (A, A, A)
    n_del_end = freqs['f_DE']  # (A, A)
    Z_del = n_del_waitd + n_del_ins.sum(axis=2) + n_del_end  # (A, A)
    Z_del_safe = jnp.where(Z_del > 1e-30, Z_del, 1.0)
    p_del_waitd = n_del_waitd / Z_del_safe
    p_del_ins = n_del_ins / Z_del_safe[:, :, None]
    p_del_end = n_del_end / Z_del_safe

    wfst = {
        # Start
        'p_S_waitm': p_S_waitm,
        'p_S_ins': p_S_ins,
        'p_S_end': p_S_end,
        # Wait-after-match/insert (V)
        'p_V_mat': p_V_mat,
        'p_V_del': p_V_del,
        # Wait-after-delete (W)
        'p_W_mat': p_W_mat,
        'p_W_del': p_W_del,
        # Post-match
        'p_mat_waitm': p_mat_waitm,
        'p_mat_ins': p_mat_ins,
        'p_mat_end': p_mat_end,
        # Post-insert
        'p_ins_waitm': p_ins_waitm,
        'p_ins_ins': p_ins_ins,
        'p_ins_end': p_ins_end,
        # Post-delete
        'p_del_waitd': p_del_waitd,
        'p_del_ins': p_del_ins,
        'p_del_end': p_del_end,
    }

    return wfst, freqs
