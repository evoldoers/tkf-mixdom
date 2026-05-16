"""TKF grammar construction via elaboration rules.

Constructs the singlet and pair grammars for TKF91, TKF92, and MixDom
using the formal elaboration rules described in the paper's Appendix C.

Each elaboration step transforms a grammar:
1. CTMC expansion: replace character emissions with substitution model
2. Fragment expansion: extend single-character to geometric-length fragments
3. Mixture expansion: add mixture components to substitution
4. Concatenation: link sequences with start/end transitions
5. Non-recursive nesting: nest fragment grammars inside domain grammars
6. Evolution: transform singlet grammar to pair grammar (M/I/D states)

The resulting grammars can be used with the generic Inside/Outside/CYK
algorithms in grammar.py. For regular grammars (TKF91/TKF92), the system
detects this and delegates to the optimized HMM DP algorithms.
"""

import jax
import numpy as np
from ..grammar.scfg import WCFG, Production, build_grammar, inside_logprob
from ..core.params import (
    tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa,
    tkf91_trans, tkf92_trans, S, M, I, D, E,
)
from ..core.ctmc import transition_matrix
from ..dp.hmm import forward_2d, viterbi_2d, safe_log


def build_tkf91_singlet_grammar(ins_rate, del_rate, n_chars):
    """Build the TKF91 singlet grammar (generates one sequence).

    The grammar is regular (right-linear):
        S -> a L   for each character a, with weight pi[a] * kappa
        S -> END   with weight (1 - kappa)
        L -> a L   for each character a, with weight pi[a] * kappa
        L -> END   with weight (1 - kappa)

    This generates geometric-length sequences from the stationary distribution.
    """
    kappa = float(ins_rate / del_rate)
    # Uniform pi for now (can be parameterized)
    pi = 1.0 / n_chars

    nonterminals = ['S', 'L', 'END']
    rules = []

    # S -> a L (start generating)
    for a in range(n_chars):
        rules.append(('S', [(a, 'T'), ('L', 'N')], pi * kappa))
    # S -> END (empty sequence)
    rules.append(('S', [('END', 'N')], 1 - kappa))

    # L -> a L (continue)
    for a in range(n_chars):
        rules.append(('L', [(a, 'T'), ('L', 'N')], pi * kappa))
    # L -> END (stop)
    rules.append(('L', [('END', 'N')], 1 - kappa))

    # END -> epsilon
    rules.append(('END', [], 1.0))

    return build_grammar(nonterminals, n_chars, rules, start='S')


def build_tkf91_pair_grammar(ins_rate, del_rate, t, Q, pi):
    """Build the TKF91 pair grammar (Pair HMM as a regular grammar).

    States: Start, Match, Insert, Delete, End
    The grammar is regular — each production emits a pair (or gap) and
    transitions to next state.

    This constructs the grammar equivalent of the TKF91 Pair HMM transition
    matrix, suitable for the generic Inside algorithm. For efficiency, use
    the dedicated forward_2d when possible.
    """
    n_chars = Q.shape[0]
    sub_matrix = np.array(transition_matrix(Q, t))
    tau = np.array(tkf91_trans(ins_rate, del_rate, t))
    pi = np.array(pi)

    # Pair terminals: encode as (anc_char, desc_char) or gaps
    # For the grammar framework, we need a 1D terminal alphabet.
    # Encode: M(a,b) = a * n_chars + b
    #         I(b) = n_chars^2 + b
    #         D(a) = n_chars^2 + n_chars + a
    n_pair_terminals = n_chars * n_chars + 2 * n_chars

    nonterminals = ['Start', 'Match', 'Insert', 'Delete', 'End']
    rules = []

    # From Start: same transitions as S row
    # Start -> M(a,b) Match  with weight tau[S,M] * pi[a] * sub[a,b]
    for a in range(n_chars):
        for b in range(n_chars):
            t_idx = a * n_chars + b
            w = float(tau[S, M]) * float(pi[a]) * float(sub_matrix[a, b])
            if w > 1e-30:
                rules.append(('Start', [(t_idx, 'T'), ('Match', 'N')], w))

    # Start -> I(b) Insert
    for b in range(n_chars):
        t_idx = n_chars * n_chars + b
        w = float(tau[S, I]) * float(pi[b])
        if w > 1e-30:
            rules.append(('Start', [(t_idx, 'T'), ('Insert', 'N')], w))

    # Start -> D(a) Delete
    for a in range(n_chars):
        t_idx = n_chars * n_chars + n_chars + a
        w = float(tau[S, D]) * float(pi[a])
        if w > 1e-30:
            rules.append(('Start', [(t_idx, 'T'), ('Delete', 'N')], w))

    # Start -> End
    w = float(tau[S, E])
    if w > 1e-30:
        rules.append(('Start', [('End', 'N')], w))

    # From Match, Insert, Delete: similar transitions
    for src_name, src_idx in [('Match', M), ('Insert', I), ('Delete', D)]:
        # -> M(a,b) Match
        for a in range(n_chars):
            for b in range(n_chars):
                t_idx = a * n_chars + b
                w = float(tau[src_idx, M]) * float(pi[a]) * float(sub_matrix[a, b])
                if w > 1e-30:
                    rules.append((src_name, [(t_idx, 'T'), ('Match', 'N')], w))

        # -> I(b) Insert
        for b in range(n_chars):
            t_idx = n_chars * n_chars + b
            w = float(tau[src_idx, I]) * float(pi[b])
            if w > 1e-30:
                rules.append((src_name, [(t_idx, 'T'), ('Insert', 'N')], w))

        # -> D(a) Delete
        for a in range(n_chars):
            t_idx = n_chars * n_chars + n_chars + a
            w = float(tau[src_idx, D]) * float(pi[a])
            if w > 1e-30:
                rules.append((src_name, [(t_idx, 'T'), ('Delete', 'N')], w))

        # -> End
        w = float(tau[src_idx, E])
        if w > 1e-30:
            rules.append((src_name, [('End', 'N')], w))

    # End -> epsilon
    rules.append(('End', [], 1.0))

    return build_grammar(nonterminals, n_pair_terminals, rules, start='Start')


def encode_pair_sequence(x, y, alignment, n_chars):
    """Encode an alignment as a sequence of pair terminals.

    Args:
        x: ancestor sequence (int array)
        y: descendant sequence (int array)
        alignment: list of (anc_pos_or_None, desc_pos_or_None)
        n_chars: alphabet size

    Returns:
        pair_seq: integer array of encoded pair terminals
    """
    pair_seq = []
    for anc_idx, desc_idx in alignment:
        if anc_idx is not None and desc_idx is not None:
            # Match
            pair_seq.append(int(x[anc_idx]) * n_chars + int(y[desc_idx]))
        elif desc_idx is not None:
            # Insert
            pair_seq.append(n_chars * n_chars + int(y[desc_idx]))
        elif anc_idx is not None:
            # Delete
            pair_seq.append(n_chars * n_chars + n_chars + int(x[anc_idx]))
    return np.array(pair_seq, dtype=np.int32)


def pair_grammar_logprob(grammar, x, y, alignment, n_chars):
    """Compute log P(alignment | pair grammar).

    Encodes the alignment as pair terminals and runs Inside.
    """
    pair_seq = encode_pair_sequence(x, y, alignment, n_chars)
    return inside_logprob(grammar, pair_seq)


def build_tkf92_pair_grammar(ins_rate, del_rate, t, ext, Q, pi):
    """Build the TKF92 pair grammar (Pair HMM with fragment extension).

    Like TKF91 but M, I, D states have self-loops with probability ext,
    modeling geometric-length fragments.
    """
    n_chars = Q.shape[0]
    sub_matrix = np.array(transition_matrix(Q, t))
    tau = np.array(tkf92_trans(ins_rate, del_rate, t, ext))
    pi = np.array(pi)

    n_pair_terminals = n_chars * n_chars + 2 * n_chars

    nonterminals = ['Start', 'Match', 'Insert', 'Delete', 'End']
    rules = []

    # From each source state, emit pair terminal and transition
    for src_name, src_idx in [('Start', S), ('Match', M), ('Insert', I), ('Delete', D)]:
        # -> M(a,b) Match
        for a in range(n_chars):
            for b in range(n_chars):
                t_idx = a * n_chars + b
                w = float(tau[src_idx, M]) * float(pi[a]) * float(sub_matrix[a, b])
                if w > 1e-30:
                    rules.append((src_name, [(t_idx, 'T'), ('Match', 'N')], w))

        # -> I(b) Insert
        for b in range(n_chars):
            t_idx = n_chars * n_chars + b
            w = float(tau[src_idx, I]) * float(pi[b])
            if w > 1e-30:
                rules.append((src_name, [(t_idx, 'T'), ('Insert', 'N')], w))

        # -> D(a) Delete
        for a in range(n_chars):
            t_idx = n_chars * n_chars + n_chars + a
            w = float(tau[src_idx, D]) * float(pi[a])
            if w > 1e-30:
                rules.append((src_name, [(t_idx, 'T'), ('Delete', 'N')], w))

        # -> End
        w = float(tau[src_idx, E])
        if w > 1e-30:
            rules.append((src_name, [('End', 'N')], w))

    # End -> epsilon
    rules.append(('End', [], 1.0))

    return build_grammar(nonterminals, n_pair_terminals, rules, start='Start')


def tkf91_forward_logprob(x, y, ins_rate, del_rate, t, Q, pi):
    """Compute log P(x, y | TKF91) using the dedicated forward algorithm.

    This is the optimized path; the grammar version should give the same result.
    """
    import jax.numpy as jnp
    sub_matrix = transition_matrix(Q, t)
    tau = tkf91_trans(ins_rate, del_rate, t)
    log_trans = safe_log(tau)
    state_types = jnp.array([S, M, I, D, E])
    log_prob, _ = forward_2d(log_trans, state_types, jnp.array(x), jnp.array(y), sub_matrix, pi)
    return float(log_prob)


def tkf92_forward_logprob(x, y, ins_rate, del_rate, t, ext, Q, pi):
    """Compute log P(x, y | TKF92) using the dedicated forward algorithm."""
    import jax.numpy as jnp
    sub_matrix = transition_matrix(Q, t)
    tau = tkf92_trans(ins_rate, del_rate, t, ext)
    log_trans = safe_log(tau)
    state_types = jnp.array([S, M, I, D, E])
    log_prob, _ = forward_2d(log_trans, state_types, jnp.array(x), jnp.array(y), sub_matrix, pi)
    return float(log_prob)


def build_mixdom_pair_grammar(main_ins_rate, main_del_rate, t,
                               dom_ins_rates, dom_del_rates, dom_weights,
                               frag_weights, ext_rates, Q, pi):
    """Build the MixDom nested pair grammar.

    This constructs a regular grammar (Pair HMM as grammar) from the
    nested MixDom transition matrix chi. The grammar has compound states
    SS, EE, and UV_df for each (compound_state, domain, fragment).

    Since MixDom is still a Pair HMM (regular), the grammar is right-linear.

    Args:
        main_ins_rate, main_del_rate: top-level domain indel rates
        t: evolutionary time
        dom_ins_rates: (n_dom,) per-domain insertion rates
        dom_del_rates: (n_dom,) per-domain deletion rates
        dom_weights: (n_dom,) domain mixture weights
        frag_weights: (n_dom, n_frag) fragment mixture weights
        ext_rates: (n_dom, n_frag) fragment extension probabilities
        Q: substitution rate matrix
        pi: equilibrium distribution

    Returns:
        WCFG (regular grammar)
    """
    import jax.numpy as jnp
    from .mixdom import build_nested_trans, n_states, MM, MI, MD, II, DD

    dom_ins_rates = jnp.array(dom_ins_rates)
    dom_del_rates = jnp.array(dom_del_rates)
    dom_weights = jnp.array(dom_weights)
    frag_weights = jnp.array(frag_weights)
    ext_rates = jnp.array(ext_rates)

    chi, state_map = build_nested_trans(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_weights,
        frag_weights, ext_rates)
    chi = np.array(chi)

    n_dom = int(dom_ins_rates.shape[0])
    n_frag = int(frag_weights.shape[1])
    N = n_states(n_dom, n_frag)
    n_chars = Q.shape[0]

    sub_matrix = np.array(transition_matrix(Q, t))
    pi_arr = np.array(pi)

    # Determine state types for compound states
    # SS=0: start (silent), EE=1: end (silent)
    # UV states: MM=match, MI=match-insert (ancestor match + desc insert?),
    # etc. For pair terminals, we use the same encoding as TKF91/92.
    # State type mapping for pair HMM advancement:
    # MM: match domain, match fragment -> emit (x,y) pair -> M
    # MI: match domain, insert fragment -> emit y only -> I
    # MD: match domain, delete fragment -> emit x only -> D
    # II: insert domain -> emit y only -> I
    # DD: delete domain -> emit x only -> D
    uv_to_pair_type = {
        MM: 'M',   # both ancestor and descendant advance
        MI: 'I',   # descendant only (insertion within matched domain)
        MD: 'D',   # ancestor only (deletion within matched domain)
        II: 'I',   # insert (descendant only)
        DD: 'D',   # delete (ancestor only)
    }

    n_pair_terminals = n_chars * n_chars + 2 * n_chars

    # Build nonterminal names
    nonterminals = ['SS', 'EE']
    nt_state_type = [S, E]  # S for start, E for end
    for d in range(n_dom):
        for uv in range(5):
            for f in range(n_frag):
                uv_names = ['MM', 'MI', 'MD', 'II', 'DD']
                nt_name = f'{uv_names[uv]}_d{d}_f{f}'
                nonterminals.append(nt_name)
                nt_state_type.append(uv_to_pair_type[uv])

    rules = []

    # For each source -> dest transition in chi with weight > 0
    SS_idx = 0
    EE_idx = 1

    def flat_idx(uv, d, f):
        return 2 + d * 5 * n_frag + uv * n_frag + f

    for src in range(N):
        for dst in range(N):
            w = float(chi[src, dst])
            if w < 1e-30:
                continue

            src_name = nonterminals[src]
            dst_name = nonterminals[dst]

            if dst == EE_idx:
                # Transition to End: unary production
                rules.append((src_name, [(dst_name, 'N')], w))
                continue

            if dst < 2:
                # dst is SS or EE (silent): unary
                rules.append((src_name, [(dst_name, 'N')], w))
                continue

            # dst is a compound state UV_df
            # Determine pair type of destination
            dst_type = nt_state_type[dst]

            if dst_type == 'M':
                # Match: emit pair terminal M(a,b)
                for a in range(n_chars):
                    for b in range(n_chars):
                        t_idx = a * n_chars + b
                        emit_w = float(pi_arr[a]) * float(sub_matrix[a, b])
                        total_w = w * emit_w
                        if total_w > 1e-30:
                            rules.append((src_name, [(t_idx, 'T'), (dst_name, 'N')], total_w))
            elif dst_type == 'I':
                # Insert: emit I(b)
                for b in range(n_chars):
                    t_idx = n_chars * n_chars + b
                    emit_w = float(pi_arr[b])
                    total_w = w * emit_w
                    if total_w > 1e-30:
                        rules.append((src_name, [(t_idx, 'T'), (dst_name, 'N')], total_w))
            elif dst_type == 'D':
                # Delete: emit D(a)
                for a in range(n_chars):
                    t_idx = n_chars * n_chars + n_chars + a
                    emit_w = float(pi_arr[a])
                    total_w = w * emit_w
                    if total_w > 1e-30:
                        rules.append((src_name, [(t_idx, 'T'), (dst_name, 'N')], total_w))

    # EE -> epsilon
    rules.append(('EE', [], 1.0))

    return build_grammar(nonterminals, n_pair_terminals, rules, start='SS')


def regular_grammar_to_pair_hmm(grammar, n_chars):
    """Convert a regular pair grammar to Pair HMM transition matrix + state types.

    Extracts the HMM structure from a right-linear grammar. This allows
    fast O(Lx*Ly*N²) DP via forward_2d instead of O(L³) grammar Inside.

    For each right-linear production A -> t B with weight w, accumulates
    trans[A, B] += w. The sum over all terminals for a given (A, B) pair
    gives the HMM transition weight (since emission weights sum to 1 for
    properly normalized pair terminals).

    Args:
        grammar: WCFG that is regular (right-linear)
        n_chars: alphabet size

    Returns:
        log_trans: (N, N) log transition matrix
        state_types: (N,) array of S/M/I/D/E codes
        e_idx: index of the E-type state (for terminal transition)
    """
    import jax.numpy as jnp

    assert grammar.is_regular(), "Grammar must be regular"
    N = grammar.n_nonterminals
    n_sq = n_chars * n_chars

    trans = np.zeros((N, N))
    state_type_votes = [set() for _ in range(N)]

    for p in grammar.productions:
        if p.is_right_linear:
            term, B = p.rhs
            trans[p.lhs, B] += p.weight
            if term < n_sq:
                state_type_votes[B].add(M)
            elif term < n_sq + n_chars:
                state_type_votes[B].add(I)
            else:
                state_type_votes[B].add(D)
        elif p.is_unary:
            trans[p.lhs, p.rhs[0]] += p.weight

    state_types = np.full(N, -1, dtype=np.int32)
    state_types[grammar.start] = S

    for i in range(N):
        if i == grammar.start:
            continue
        votes = state_type_votes[i]
        if len(votes) == 0:
            state_types[i] = E
        elif len(votes) == 1:
            state_types[i] = list(votes)[0]
        else:
            raise ValueError(
                f"Nonterminal {grammar.nonterminals[i]} has mixed emission types: {votes}")

    log_trans = np.log(np.maximum(trans, 1e-300))
    e_idx = int(np.argmax(state_types == E))

    return jnp.array(log_trans), jnp.array(state_types), e_idx


def grammar_forward_2d(grammar, x, y, n_chars, Q, pi, t):
    """Compute log P(x, y) using a pair grammar — dispatch to HMM if regular.

    For regular grammars, converts to Pair HMM and uses forward_2d.
    For general CFGs, uses the grammar Inside algorithm.

    Args:
        grammar: WCFG (pair grammar)
        x, y: ancestor/descendant sequences (int arrays)
        n_chars: alphabet size
        Q: substitution rate matrix
        pi: equilibrium distribution
        t: evolutionary time

    Returns:
        log_prob: log P(x, y | grammar)
    """
    import jax.numpy as jnp

    if grammar.is_regular():
        log_trans, state_types, e_idx = regular_grammar_to_pair_hmm(grammar, n_chars)
        sub_matrix = transition_matrix(Q, t)
        _, F = forward_2d(log_trans, state_types,
                          jnp.array(x), jnp.array(y), sub_matrix, pi)
        Lx, Ly = len(x), len(y)
        log_prob = float(jax.nn.logsumexp(F[Lx, Ly, :] + log_trans[:, e_idx]))
        return log_prob
    else:
        # Fall back to grammar Inside (requires alignment)
        raise NotImplementedError(
            "grammar_forward_2d for non-regular grammars requires "
            "pair SCFG Inside algorithm (not yet implemented)")


def grammar_em_step(grammar, x, y, n_chars, Q, pi, t, ins_rate, del_rate, lr=0.01):
    """One EM step using a pair grammar — delegates to HMM DP for regular grammars.

    Args:
        grammar: WCFG (pair grammar)
        x, y: ancestor/descendant sequences
        n_chars: alphabet size
        Q: substitution rate matrix
        pi: equilibrium distribution
        t: evolutionary time
        ins_rate, del_rate: current indel parameters
        lr: learning rate for M-step

    Returns:
        ins_new, del_new, Q_new, log_prob
    """
    import jax.numpy as jnp

    if grammar.is_regular():
        from .compiled import TKF91Model
        from ..core.ctmc import m_step_substitution
        model = TKF91Model()
        params = {'ins_rate': ins_rate, 'del_rate': del_rate, 't': t,
                  'Q': Q, 'pi': pi}
        log_prob, n_trans, posteriors = model.e_step(params,
            jnp.array(x), jnp.array(y))
        stats = model.extract_stats(n_trans, params)
        new_params = model.m_step(stats, params)
        # Substitution M-step from match pairs (extracted from posteriors)
        from ..core.params import M as _M
        match_pairs = []
        x_arr, y_arr = jnp.array(x), jnp.array(y)
        for ix in range(x_arr.shape[0]):
            for iy in range(y_arr.shape[0]):
                w = float(posteriors[ix + 1, iy + 1, _M])
                if w > 1e-15:
                    match_pairs.append((int(x_arr[ix]), int(y_arr[iy]), w))
        Q_new = m_step_substitution(match_pairs, Q, pi, t, n_chars)
        return float(new_params['ins_rate']), float(new_params['del_rate']), Q_new, float(log_prob)
    else:
        raise NotImplementedError(
            "grammar_em_step for non-regular grammars not yet implemented")


def mixdom_forward_logprob(x, y, main_ins_rate, main_del_rate, t,
                            dom_ins_rates, dom_del_rates, dom_weights,
                            frag_weights, ext_rates, Q, pi):
    """Compute log P(x, y | MixDom) using the dedicated forward algorithm."""
    import jax.numpy as jnp
    from .mixdom import build_nested_trans, MM, MI, MD, II, DD

    dom_ins_rates = jnp.array(dom_ins_rates)
    dom_del_rates = jnp.array(dom_del_rates)
    dom_weights = jnp.array(dom_weights)
    frag_weights = jnp.array(frag_weights)
    ext_rates = jnp.array(ext_rates)

    chi, state_map = build_nested_trans(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_weights,
        frag_weights, ext_rates)

    N = chi.shape[0]
    # State types: SS=S(0), EE=E(4), compound UV states:
    # MM/MI/MD -> M(1), II -> I(2), DD -> D(3)
    state_types = jnp.zeros(N, dtype=jnp.int32)
    state_types = state_types.at[0].set(S)  # SS
    state_types = state_types.at[1].set(E)  # EE

    n_dom = int(dom_ins_rates.shape[0])
    n_frag = int(frag_weights.shape[1])
    uv_to_st = {MM: M, MI: I, MD: D, II: I, DD: D}
    for d in range(n_dom):
        for uv in range(5):
            for f in range(n_frag):
                idx = 2 + d * 5 * n_frag + uv * n_frag + f
                state_types = state_types.at[idx].set(uv_to_st[uv])

    sub_matrix = transition_matrix(Q, t)
    log_trans = safe_log(chi)

    # forward_2d hardcodes E=4 as the end state index.
    # For MixDom, E state (EE) is at index 1. We need to compute
    # the terminal transition manually.
    _, F = forward_2d(log_trans, state_types, jnp.array(x), jnp.array(y),
                      sub_matrix, pi)
    Lx, Ly = len(x), len(y)
    # Find the E-type state index
    e_idx = int(jnp.argmax(state_types == E))
    log_prob = float(jax.nn.logsumexp(F[Lx, Ly, :] + log_trans[:, e_idx]))
    return log_prob
