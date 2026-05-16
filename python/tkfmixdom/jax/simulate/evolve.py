"""Sequence evolution via stochastic Forward traceback.

Given an ancestor sequence and a model P(descendant|ancestor,t), evolves
the sequence by running a sequential stochastic process: at each step,
sample the next HMM state and (if emitting) the next descendant character.

This works for any plain Pair HMM (TKF91, TKF92). A MixDom-aware
simulator was removed because its spec diverged from the MixDom2
model; it can be rebuilt against the corrected spec.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.params import S, M, I, D, E, tkf91_trans, tkf92_trans
from ..core.ctmc import transition_matrix


def evolve_pair_hmm(rng, ancestor, log_trans, state_types, sub_matrix, pi):
    """Evolve a sequence through a Pair HMM by direct simulation.

    Sequentially samples states from the Pair HMM. At each step:
    - From the current state, sample the next state from the transition distribution
    - M state: consume next ancestor char, emit descendant char from sub_matrix
    - I state: emit descendant char from pi
    - D state: consume next ancestor char, no descendant emission
    - E state: done

    The process ensures all ancestor characters are consumed (via M or D).

    Args:
        rng: JAX PRNG key
        ancestor: (Lx,) integer array of ancestor characters
        log_trans: (n_states, n_states) log transition matrix
        state_types: (n_states,) state type codes (S=0, M=1, I=2, D=3, E=4)
        sub_matrix: (A, A) substitution probability matrix
        pi: (A,) equilibrium distribution

    Returns:
        descendant: integer array of descendant characters
        alignment: list of (anc_idx_or_None, desc_idx_or_None) pairs
        log_prob: log probability of the sampled path
    """
    Lx = len(ancestor)
    ns = log_trans.shape[0]
    ancestor = np.array(ancestor)
    trans_np = np.array(log_trans)
    sub_np = np.array(sub_matrix)
    pi_np = np.array(pi)
    A = len(pi_np)

    rng_np = int(jr.fold_in(rng, 0)[0]) % (2**31)
    np_rng = np.random.RandomState(rng_np)

    # Start in state S
    current_state = S
    anc_pos = 0  # next ancestor position to consume
    descendant = []
    alignment = []
    log_prob = 0.0

    max_steps = Lx * 10 + 100  # safety limit

    for step in range(max_steps):
        # Compute transition probabilities from current state
        # But constrain based on what's possible:
        # - M and D require anc_pos < Lx (ancestor chars remaining)
        # - E requires anc_pos == Lx (all ancestor consumed)
        log_probs = trans_np[current_state].copy()

        # Mask impossible transitions
        for k in range(ns):
            st = state_types[k]
            if st == M or st == D:
                if anc_pos >= Lx:
                    log_probs[k] = -np.inf  # can't consume more ancestor
            elif st == E:
                if anc_pos < Lx:
                    log_probs[k] = -np.inf  # must consume all ancestor first

        # Normalize
        probs = _softmax(log_probs)
        next_state = np_rng.choice(ns, p=probs)
        log_prob += np.log(max(probs[next_state], 1e-300))

        st = state_types[next_state]

        if st == E:
            break
        elif st == M:
            # Consume ancestor char, emit descendant char
            x_char = ancestor[anc_pos]
            y_char = np_rng.choice(A, p=sub_np[x_char])
            log_prob += np.log(max(sub_np[x_char, y_char], 1e-300))
            desc_idx = len(descendant)
            descendant.append(y_char)
            alignment.append((anc_pos, desc_idx))
            anc_pos += 1
        elif st == I:
            # Emit descendant char from pi
            y_char = np_rng.choice(A, p=pi_np)
            log_prob += np.log(max(pi_np[y_char], 1e-300))
            desc_idx = len(descendant)
            descendant.append(y_char)
            alignment.append((None, desc_idx))
        elif st == D:
            # Consume ancestor char, no descendant
            alignment.append((anc_pos, None))
            anc_pos += 1

        current_state = next_state

    return np.array(descendant, dtype=np.int32), alignment, log_prob


def _softmax(v):
    """Softmax of a vector, handling -inf."""
    v = np.array(v, dtype=np.float64)
    m = np.max(v)
    if not np.isfinite(m):
        # All -inf: uniform
        return np.ones(len(v)) / len(v)
    e = np.exp(v - m)
    s = np.sum(e)
    if s < 1e-300:
        return np.ones(len(v)) / len(v)
    return e / s


def evolve_tkf91(rng, ancestor, ins_rate, del_rate, t, Q, pi):
    """Evolve a sequence under TKF91.

    Args:
        rng: JAX PRNG key
        ancestor: integer array
        ins_rate, del_rate: indel rates
        t: evolutionary time
        Q: substitution rate matrix
        pi: equilibrium distribution

    Returns:
        descendant, alignment, log_prob
    """
    sub_matrix = np.array(transition_matrix(Q, t))
    tau = np.array(tkf91_trans(ins_rate, del_rate, t))
    log_trans = np.log(np.maximum(tau, 1e-300))
    state_types = np.array([S, M, I, D, E])
    return evolve_pair_hmm(rng, ancestor, log_trans, state_types, sub_matrix, pi)


def evolve_tkf92(rng, ancestor, ins_rate, del_rate, t, ext, Q, pi):
    """Evolve a sequence under TKF92.

    Args:
        rng: JAX PRNG key
        ancestor: integer array
        ins_rate, del_rate: indel rates
        t: evolutionary time
        ext: fragment extension probability
        Q: substitution rate matrix
        pi: equilibrium distribution

    Returns:
        descendant, alignment, log_prob
    """
    sub_matrix = np.array(transition_matrix(Q, t))
    tau = np.array(tkf92_trans(ins_rate, del_rate, t, ext))
    log_trans = np.log(np.maximum(tau, 1e-300))
    state_types = np.array([S, M, I, D, E])
    return evolve_pair_hmm(rng, ancestor, log_trans, state_types, sub_matrix, pi)


