"""TKF91/TKF92 sequence evolution simulator.

Simulates ancestor-descendant sequence pairs under the TKF model
with a given substitution model. Used for statistical consistency tests.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..core.params import tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa


def simulate_stationary_sequence(rng, ins_rate, del_rate, sub_pi,
                                   max_len=1000, ext=0.0):
    """Sample a sequence from the TKF91 / TKF92 stationary distribution.

    TKF91 (ext=0): length ~ Geometric(1-κ) where κ = λ/μ.
    TKF92 (ext>0): N_links ~ Geometric(1-κ); each link emits a fragment of
        length 1+Geometric(ext); total length is the zero-inflated
        geometric distribution of body-tkf92.tex with parameters κ, p.

    Characters ~ pi (equilibrium distribution).

    Args:
        rng: JAX PRNG key
        ins_rate, del_rate: indel rates (κ = ins_rate / del_rate)
        sub_pi: (A,) equilibrium distribution
        max_len: maximum sequence length (truncation)
        ext: TKF92 fragment-extension probability (0 for TKF91)

    Returns:
        seq: integer array of character indices
    """
    kappa = float(ins_rate / del_rate)
    rng1, rng2 = jr.split(rng)
    ext_f = float(ext)

    # Number of TKF links: N ~ Geom(1-κ) with support {0, 1, 2, ...}.
    u = float(jr.uniform(rng1))
    if kappa < 1e-10:
        n_links = 0
    else:
        n_links = int(np.floor(np.log(u) / np.log(kappa)))

    if ext_f <= 0.0:
        length = n_links
    else:
        # Each link emits a fragment of length 1+Geom(ext) with support
        # {1, 2, ...}. Generate fragment-length contributions and sum.
        if n_links == 0:
            length = 0
        else:
            rng_frag = jr.fold_in(rng1, 1)
            us = np.asarray(jr.uniform(rng_frag, shape=(n_links,)))
            # Geom(1-ext) on support {0, 1, ...}: floor(log(u)/log(ext))
            if ext_f < 1.0:
                extra = np.floor(np.log(np.maximum(us, 1e-30)) / np.log(ext_f))
            else:
                extra = np.full(n_links, max_len, dtype=np.float64)
            length = int(n_links + extra.sum())

    length = min(length, max_len)
    if length <= 0:
        return jnp.array([], dtype=jnp.int32)

    # Characters from pi.
    A = sub_pi.shape[0]
    chars = jr.choice(rng2, A, shape=(length,), p=sub_pi)
    return chars


def simulate_descendant(rng, ancestor, ins_rate, del_rate, t, sub_matrix, sub_pi):
    """Simulate descendant sequence given ancestor under TKF91.

    For each ancestral link:
    - Survives with probability α = exp(-μt)
    - If survives, character substitutes via sub_matrix
    - Offspring links inserted with geometric(β) distribution

    For orphan links (from dead ancestors):
    - New links created with probability γ (then geometric(β) more)

    Args:
        rng: JAX PRNG key
        ancestor: (L,) integer array of ancestor characters
        ins_rate, del_rate: indel rates
        t: evolutionary time
        sub_matrix: (A, A) substitution probability matrix P(y|x)
        sub_pi: (A,) equilibrium distribution

    Returns:
        descendant: integer array
        alignment: list of (anc_idx_or_None, desc_idx_or_None) pairs
    """
    alpha = float(tkf_alpha(del_rate, t))
    beta = float(tkf_beta(ins_rate, del_rate, t))
    gamma = float(tkf_gamma(ins_rate, del_rate, t))
    A = sub_pi.shape[0]

    # Use numpy for variable-length operations
    rng_np = int(jr.fold_in(rng, 0)[0])
    np_rng = np.random.RandomState(rng_np % (2**31))
    ancestor_np = np.array(ancestor, dtype=int)
    sub_matrix_np = np.array(sub_matrix)
    pi_np = np.array(sub_pi)
    L = len(ancestor_np)

    descendant = []
    alignment = []

    def insert_offspring(char_idx):
        """Insert geometric(β) offspring after a character."""
        while np_rng.random() < beta:
            new_char = np_rng.choice(A, p=pi_np)
            desc_idx = len(descendant)
            descendant.append(new_char)
            alignment.append((None, desc_idx))

    # Process immortal link (before first ancestor position)
    # In TKF91, the immortal link always survives; offspring are born from it
    insert_offspring(None)

    for i in range(L):
        # Does this mortal link survive?
        if np_rng.random() < alpha:
            # Survives: substitute character
            old_char = ancestor_np[i]
            new_char = np_rng.choice(A, p=sub_matrix_np[old_char])
            desc_idx = len(descendant)
            descendant.append(new_char)
            alignment.append((i, desc_idx))
            # Offspring after this surviving link
            insert_offspring(desc_idx)
        else:
            # Dies: alignment records deletion
            alignment.append((i, None))
            # Orphan: new links with probability γ, then geometric(β)
            if np_rng.random() < gamma:
                new_char = np_rng.choice(A, p=pi_np)
                desc_idx = len(descendant)
                descendant.append(new_char)
                alignment.append((None, desc_idx))
                insert_offspring(desc_idx)

    return jnp.array(descendant, dtype=jnp.int32), alignment


def simulate_pair(rng, ins_rate, del_rate, t, sub_matrix, sub_pi, max_len=1000):
    """Simulate an ancestor-descendant pair under TKF91.

    Returns:
        ancestor: integer array
        descendant: integer array
        alignment: list of (anc_idx_or_None, desc_idx_or_None) pairs
    """
    rng1, rng2 = jr.split(rng)
    ancestor = simulate_stationary_sequence(rng1, ins_rate, del_rate, sub_pi, max_len)
    descendant, alignment = simulate_descendant(rng2, ancestor, ins_rate, del_rate, t,
                                                 sub_matrix, sub_pi)
    return ancestor, descendant, alignment


def _tkf92_backward_structural_log(tau, ancestor, sub_pi):
    """Log-space backward structural probabilities for TKF92 guided sampling.

    logB[ai, s] = log P(eventually consume rest of ancestor from position ai
    in state s, reaching E; marginalising over descendant characters).

    Computed in log space (numpy.logaddexp) so the recurrence is stable for
    arbitrarily long ancestors.  The earlier float64 version underflowed for
    ancestors longer than ~22 characters at low-rate regimes, silently
    truncating descendants.
    """
    from ..core.params import S, M, I, D, E

    L = len(ancestor)
    pi_np = np.asarray(sub_pi)
    tau_np = np.asarray(tau)
    NEG = -np.inf
    LOG_ZERO_GUARD = 1e-300
    log_tau = np.log(np.where(tau_np > 0, tau_np, LOG_ZERO_GUARD))
    log_pi = np.log(np.where(pi_np > 0, pi_np, LOG_ZERO_GUARD))
    one_minus_II = 1.0 - tau_np[I, I]
    log_one_minus_II = (np.log(one_minus_II) if one_minus_II > 0
                        else NEG)

    logB = np.full((L + 1, 5), NEG)
    logB[L, E] = 0.0
    # B[L, I] = tau[I, E] / (1 - tau[I, I]); rest combine via logaddexp.
    logB[L, I] = log_tau[I, E] - log_one_minus_II
    logB[L, M] = np.logaddexp(log_tau[M, E], log_tau[M, I] + logB[L, I])
    logB[L, D] = np.logaddexp(log_tau[D, E], log_tau[D, I] + logB[L, I])
    logB[L, S] = np.logaddexp(log_tau[S, E], log_tau[S, I] + logB[L, I])

    for ai in range(L - 1, -1, -1):
        log_px = log_pi[ancestor[ai]]
        # Solve B(I, ai) first (has self-loop):
        # B(I, ai) = [tau[I, M]*px*B(M, ai+1) + tau[I, D]*px*B(D, ai+1)] / (1-tau[I, I])
        log_numer_I = np.logaddexp(
            log_tau[I, M] + log_px + logB[ai + 1, M],
            log_tau[I, D] + log_px + logB[ai + 1, D],
        )
        logB[ai, I] = log_numer_I - log_one_minus_II
        for s in [S, M, D]:
            terms = np.array([
                log_tau[s, M] + log_px + logB[ai + 1, M],
                log_tau[s, I] + logB[ai, I],
                log_tau[s, D] + log_px + logB[ai + 1, D],
            ])
            # log-sum-exp over the three branches.
            m = terms.max()
            if m == NEG:
                logB[ai, s] = NEG
            else:
                logB[ai, s] = m + np.log(np.sum(np.exp(terms - m)))
    return logB


def simulate_descendant_tkf92(rng, ancestor, ins_rate, del_rate, t, ext, sub_matrix, sub_pi):
    """Simulate descendant sequence given ancestor under TKF92.

    Uses backward-guided forward sampling to correctly sample from the pair HMM's
    conditional distribution P(y|x). At each step, transitions are weighted by the
    probability of eventually consuming all remaining ancestor characters (backward
    structural probability), giving the exact conditional distribution.

    Args:
        rng: JAX PRNG key
        ancestor: (L,) integer array of ancestor characters
        ins_rate, del_rate: indel rates
        t: evolutionary time
        ext: fragment extension probability
        sub_matrix: (A, A) substitution probability matrix P(y|x)
        sub_pi: (A,) equilibrium distribution

    Returns:
        descendant: integer array
        alignment: list of (anc_idx_or_None, desc_idx_or_None) pairs
    """
    from ..core.params import tkf92_trans, S, M, I, D, E

    tau = np.array(tkf92_trans(ins_rate, del_rate, t, ext))
    A = sub_pi.shape[0]

    rng_np = int(jr.fold_in(rng, 0)[0])
    np_rng = np.random.RandomState(rng_np % (2**31))
    ancestor_np = np.array(ancestor, dtype=int)
    sub_matrix_np = np.array(sub_matrix)
    pi_np = np.array(sub_pi)
    L = len(ancestor_np)

    # Backward structural probabilities in log space (numerically stable for
    # arbitrarily long ancestors).
    logB = _tkf92_backward_structural_log(tau, ancestor_np, sub_pi)
    log_pi_local = np.log(np.where(pi_np > 0, pi_np, 1e-300))
    NEG = -np.inf

    descendant = []
    alignment = []
    ai = 0  # ancestor index (how many ancestor chars consumed)
    state = S

    while True:
        # Backward-guided transition log-weights.
        log_weights = np.full(5, NEG)
        with np.errstate(divide='ignore'):
            log_tau_state = np.where(tau[state] > 0, np.log(tau[state]), NEG)
        if ai == L:
            log_weights[E] = log_tau_state[E]  # logB[E]=0
        if ai < L:
            log_weights[M] = log_tau_state[M] + log_pi_local[ancestor_np[ai]] + logB[ai + 1, M]
            log_weights[D] = log_tau_state[D] + log_pi_local[ancestor_np[ai]] + logB[ai + 1, D]
        # Insert emission sums to 1 over y chars; no px factor.
        log_weights[I] = log_tau_state[I] + logB[ai, I]

        m = log_weights.max()
        if m == NEG:
            raise RuntimeError(
                f'simulate_descendant_tkf92: all transition log-weights -inf '
                f'at state={state}, ai={ai}, L={L}; backward DP exhausted '
                f'(should not happen with log-space DP)')
        log_norm = m + np.log(np.sum(np.exp(log_weights - m)))
        probs = np.exp(log_weights - log_norm)
        # Numerical hygiene before np.random.choice.
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()
        next_state = np_rng.choice(5, p=probs)

        if next_state == E:
            break
        elif next_state == M:
            old_char = ancestor_np[ai]
            new_char = np_rng.choice(A, p=sub_matrix_np[old_char])
            desc_idx = len(descendant)
            descendant.append(new_char)
            alignment.append((ai, desc_idx))
            ai += 1
        elif next_state == I:
            new_char = np_rng.choice(A, p=pi_np)
            desc_idx = len(descendant)
            descendant.append(new_char)
            alignment.append((None, desc_idx))
        elif next_state == D:
            alignment.append((ai, None))
            ai += 1

        state = next_state

    # Invariant: alignment must consume exactly |ancestor| ancestor chars and
    # produce exactly |descendant| descendant chars.
    n_anc_consumed = sum(1 for a, _ in alignment if a is not None)
    n_desc_produced = sum(1 for _, d in alignment if d is not None)
    assert n_anc_consumed == L, (
        f'simulate_descendant_tkf92: alignment consumed {n_anc_consumed} of '
        f'{L} ancestor chars; truncation bug')
    assert n_desc_produced == len(descendant), (
        f'simulate_descendant_tkf92: alignment produced {n_desc_produced} '
        f'descendant marks but descendant has {len(descendant)} chars')

    return jnp.array(descendant, dtype=jnp.int32), alignment


def simulate_pair_tkf92(rng, ins_rate, del_rate, t, ext, sub_matrix, sub_pi, max_len=1000):
    """Simulate an ancestor-descendant pair under TKF92.

    Returns:
        ancestor: integer array
        descendant: integer array
        alignment: list of (anc_idx_or_None, desc_idx_or_None) pairs
    """
    rng1, rng2 = jr.split(rng)
    ancestor = simulate_stationary_sequence(
        rng1, ins_rate, del_rate, sub_pi, max_len, ext=ext)
    descendant, alignment = simulate_descendant_tkf92(
        rng2, ancestor, ins_rate, del_rate, t, ext, sub_matrix, sub_pi
    )
    return ancestor, descendant, alignment


def alignment_to_state_sequence(alignment):
    """Convert alignment pairs to HMM state sequence.

    Returns list of states (M=1, I=2, D=3).
    """
    from ..core.params import M, I, D
    states = []
    for anc_idx, desc_idx in alignment:
        if anc_idx is not None and desc_idx is not None:
            states.append(M)
        elif desc_idx is not None:
            states.append(I)
        elif anc_idx is not None:
            states.append(D)
    return states


def count_alignment_states(alignment):
    """Count M, I, D states in an alignment.

    Returns (n_match, n_insert, n_delete).
    """
    n_m = sum(1 for a, d in alignment if a is not None and d is not None)
    n_i = sum(1 for a, d in alignment if a is None and d is not None)
    n_d = sum(1 for a, d in alignment if a is not None and d is None)
    return n_m, n_i, n_d


def simulate_bdi_gillespie(np_rng, i, ins_rate, del_rate, t):
    """Simulate a linear BDI process using the Gillespie algorithm.

    Models the continuous-time birth-death-immigration process underlying TKF91:
    - Birth rate per individual: λ (ins_rate)
    - Death rate per individual: μ (del_rate)
    - Immigration rate: λ (from the immortal link)

    Args:
        np_rng: numpy RandomState
        i: initial population (number of mortal links)
        ins_rate: birth/immigration rate (λ)
        del_rate: death rate (μ)
        t: total time

    Returns:
        j: final population
        n_births: births from existing individuals
        n_deaths: total deaths
        n_immigrations: total immigrations
        sojourn: total sojourn time (∫n(s)ds over [0,t])
    """
    n = i
    s = 0.0
    n_births = 0
    n_deaths = 0
    n_immigrations = 0
    sojourn = 0.0

    while s < t:
        birth_rate = ins_rate * n
        death_rate = del_rate * n
        imm_rate = ins_rate  # immigration from immortal link
        total_rate = birth_rate + death_rate + imm_rate

        if total_rate < 1e-30:
            sojourn += n * (t - s)
            break

        dt = np_rng.exponential(1.0 / total_rate)

        if s + dt > t:
            sojourn += n * (t - s)
            break

        sojourn += n * dt
        s += dt

        u = np_rng.random() * total_rate
        if u < birth_rate:
            n += 1
            n_births += 1
        elif u < birth_rate + death_rate:
            n -= 1
            n_deaths += 1
        else:
            n += 1
            n_immigrations += 1

    return n, n_births, n_deaths, n_immigrations, sojourn
