"""Generic pair HMM simulator via backward-guided forward sampling.

Works with any CompiledModel that provides a transition matrix and state types.
Correctly samples from P(y|x) for arbitrary pair HMMs (TKF91, TKF92, MixDom).

The algorithm:
1. Compute backward structural probabilities B(s, ai) — the probability of
   consuming all remaining ancestor characters and reaching E, starting from
   state s at ancestor position ai.
2. Forward sample: at each step, weight transitions by B to get exact
   conditional distribution P(next_state | current_state, remaining_ancestor).
"""

import numpy as np
import jax.random as jr

from ..core.params import S, M, I, D, E


def _classify_states(state_types):
    """Classify states into sets by emission type.

    Returns:
        consumes_x: set of state indices that consume an ancestor character (M, D types)
        emits_y: set of state indices that emit a descendant character (M, I types)
        e_idx: index of the End state
        s_idx: index of the Start state
    """
    st = np.array(state_types)
    consumes_x = set(np.where((st == M) | (st == D))[0])
    emits_y = set(np.where((st == M) | (st == I))[0])
    e_idx = int(np.where(st == E)[0][0])
    s_idx = int(np.where(st == S)[0][0])
    # I-type states (emit y but don't consume x) — may have self-loops
    i_type = set(np.where(st == I)[0])
    return consumes_x, emits_y, i_type, e_idx, s_idx


def backward_structural(tau, state_types, ancestor, sub_pi):
    """Compute backward structural probabilities for generic pair HMM.

    B(s, ai) = probability of consuming ancestor[ai:] and reaching E,
    starting from state s at position ai.

    For states consuming x (M/D-type): transition advances ai.
    For I-type states: transition stays at ai (self-loops possible).
    For E state: B(E, L) = 1.

    Handles I-type self-loops by solving the linear system directly.

    Args:
        tau: (N, N) transition matrix
        state_types: (N,) array of S/M/I/D/E codes
        ancestor: (L,) ancestor character indices
        sub_pi: (A,) equilibrium distribution (for marginalizing x emissions)

    Returns:
        B: (L+1, N) backward structural probabilities
    """
    tau_np = np.array(tau)
    pi_np = np.array(sub_pi)
    N = tau_np.shape[0]
    L = len(ancestor)

    consumes_x, emits_y, i_type, e_idx, s_idx = _classify_states(state_types)
    # Non-I body states (M, D, S types that are not I-type and not E)
    non_i_body = [s for s in range(N) if s not in i_type and s != e_idx]
    i_list = sorted(i_type)

    B = np.zeros((L + 1, N))
    B[L, e_idx] = 1.0

    # At ai=L: must reach E without consuming more x.
    # Only I-type states (which don't consume x) and E are reachable.
    # Solve for I-type states: B_I = (I - T_II)^{-1} @ T_IE
    # where T_II is the I→I submatrix and T_IE is I→E column.
    if i_list:
        n_i = len(i_list)
        T_II = np.zeros((n_i, n_i))
        T_IE = np.zeros(n_i)
        for ii, si in enumerate(i_list):
            T_IE[ii] = tau_np[si, e_idx]
            for jj, sj in enumerate(i_list):
                T_II[ii, jj] = tau_np[si, sj]
        # Solve (I - T_II) @ B_I = T_IE
        B_I = np.linalg.solve(np.eye(n_i) - T_II, T_IE)
        for ii, si in enumerate(i_list):
            B[L, si] = B_I[ii]

    # Non-I states at ai=L: can only go to I or E (can't consume more x)
    for s in non_i_body:
        val = tau_np[s, e_idx]
        for ii, si in enumerate(i_list):
            val += tau_np[s, si] * B[L, si]
        B[L, s] = val

    # Backward recurrence for ai < L
    for ai in range(L - 1, -1, -1):
        # Emission factor for consuming ancestor[ai]
        px = pi_np[ancestor[ai]]

        # First solve for I-type states at this position.
        # B_I(ai) depends on B(x-consuming, ai+1) and B_I(ai) (self-loops).
        # System: B_I = T_II @ B_I + rhs_I
        # where rhs_I[i] = Σ_{s consumes x} tau[i,s] * px * B(s, ai+1)
        #                 + tau[i, E] * 0  (can't end with remaining x)
        if i_list:
            rhs_I = np.zeros(n_i)
            for ii, si in enumerate(i_list):
                for sj in range(N):
                    if sj in consumes_x:
                        rhs_I[ii] += tau_np[si, sj] * px * B[ai + 1, sj]
            B_I = np.linalg.solve(np.eye(n_i) - T_II, rhs_I)
            for ii, si in enumerate(i_list):
                B[ai, si] = B_I[ii]

        # Non-I states
        for s in non_i_body:
            val = 0.0
            for sj in range(N):
                if sj in consumes_x:
                    val += tau_np[s, sj] * px * B[ai + 1, sj]
                elif sj in i_type:
                    val += tau_np[s, sj] * B[ai, sj]
                # E: 0 (can't end with remaining x)
            B[ai, s] = val

    return B


def simulate_pair_hmm(rng, ancestor, tau, state_types, sub_matrix, sub_pi):
    """Simulate a descendant sequence from a generic pair HMM.

    Uses backward-guided forward sampling to sample from P(y|x).

    Args:
        rng: JAX PRNG key
        ancestor: (L,) integer array of ancestor characters
        tau: (N, N) transition matrix (rows sum to 1 for non-E states)
        state_types: (N,) array of S/M/I/D/E codes
        sub_matrix: (A, A) substitution probability matrix P(y|x)
        sub_pi: (A,) equilibrium distribution

    Returns:
        descendant: (L',) integer array
        alignment: list of (anc_idx_or_None, desc_idx_or_None) pairs
        state_path: list of state indices visited (excluding S and E)
    """
    tau_np = np.array(tau)
    sub_np = np.array(sub_matrix)
    pi_np = np.array(sub_pi)
    ancestor_np = np.array(ancestor, dtype=int)
    N = tau_np.shape[0]
    A = pi_np.shape[0]
    L = len(ancestor_np)

    consumes_x, emits_y, i_type, e_idx, s_idx = _classify_states(state_types)

    # Compute backward structural probabilities
    B = backward_structural(tau_np, state_types, ancestor_np, sub_pi)

    # Seed numpy RNG
    rng_np = int(jr.fold_in(rng, 0)[0])
    np_rng = np.random.RandomState(rng_np % (2**31))

    descendant = []
    alignment = []
    state_path = []
    ai = 0  # ancestor position
    state = s_idx

    while True:
        # Compute backward-guided transition weights
        weights = np.zeros(N)
        for s_next in range(N):
            if s_next == s_idx:
                continue  # can't go back to S
            if s_next == e_idx:
                if ai == L:
                    weights[s_next] = tau_np[state, s_next]  # B(E,L) = 1
            elif s_next in consumes_x:
                if ai < L:
                    px = pi_np[ancestor_np[ai]]
                    weights[s_next] = tau_np[state, s_next] * px * B[ai + 1, s_next]
            elif s_next in i_type:
                weights[s_next] = tau_np[state, s_next] * B[ai, s_next]

        w_sum = weights.sum()
        if w_sum < 1e-30:
            break
        probs = weights / w_sum
        next_state = np_rng.choice(N, p=probs)

        if next_state == e_idx:
            break

        st_type = int(np.array(state_types)[next_state])
        state_path.append(next_state)

        if st_type == M:
            # Match: consume x, emit y
            old_char = ancestor_np[ai]
            new_char = np_rng.choice(A, p=sub_np[old_char])
            desc_idx = len(descendant)
            descendant.append(new_char)
            alignment.append((ai, desc_idx))
            ai += 1
        elif st_type == D:
            # Delete: consume x, no y
            alignment.append((ai, None))
            ai += 1
        elif st_type == I:
            # Insert: emit y, don't consume x
            new_char = np_rng.choice(A, p=pi_np)
            desc_idx = len(descendant)
            descendant.append(new_char)
            alignment.append((None, desc_idx))

        state = next_state

    import jax.numpy as jnp
    return jnp.array(descendant, dtype=jnp.int32), alignment, state_path


def simulate_pair_from_model(rng, model, params, max_len=1000):
    """Simulate an ancestor-descendant pair from a CompiledModel.

    1. Sample ancestor from stationary distribution (geometric length, chars from pi)
    2. Sample descendant via backward-guided forward sampling through pair HMM

    Args:
        rng: JAX PRNG key
        model: CompiledModel instance (has build_trans method)
        params: parameter dict for the model
        max_len: maximum ancestor length

    Returns:
        ancestor: (L,) integer array
        descendant: (L',) integer array
        alignment: list of (anc_idx_or_None, desc_idx_or_None) pairs
        state_path: list of state indices visited
    """
    import jax.numpy as jnp
    from ..core.ctmc import transition_matrix

    rng1, rng2 = jr.split(rng)

    # Get model matrices
    tau, state_types = model.build_trans(params)
    sub_matrix = transition_matrix(params['Q'], params['t'])
    pi = params['pi']

    # For stationary ancestor length, need κ = λ/μ
    # Use top-level rates (main_ins/main_del for MixDom, ins_rate/del_rate for TKF91/92)
    if 'main_ins' in params:
        ins_rate = params['main_ins']
        del_rate = params['main_del']
    else:
        ins_rate = params['ins_rate']
        del_rate = params['del_rate']

    from .simulate import simulate_stationary_sequence
    ancestor = simulate_stationary_sequence(rng1, ins_rate, del_rate, pi, max_len)

    # Simulate descendant
    descendant, alignment, state_path = simulate_pair_hmm(
        rng2, ancestor, tau, state_types, sub_matrix, pi)

    return ancestor, descendant, alignment, state_path
