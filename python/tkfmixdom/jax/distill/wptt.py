"""Weighted Parse-tree Pair Transducer (WPTT) for RNA evolution.

Implements the order-1 WPTT described in ParseTreeTransducers.md.
The WPTT transduces an ancestor parse tree into a descendant parse tree,
modeling substitution, insertion, and deletion of structural elements.

State space (246 nonterminals):
  - 1 Start state
  - 7 * 7 * 5 = 245 states X-ab-pq where:
      X = alignment context: M (match), I (insert), D (delete),
          V (ready post-M/I), W (ready post-D)
      ab = input context: one of 7 base pair classes (AU,CG,...,NN)
      pq = output context: one of 7 base pair classes

  Two distinct ready/wait states (V and W) are needed because TKF91
  transition probabilities differ after match/insert vs after delete:
    - V (post-M/I): insert probability = beta
    - W (post-D): insert probability = gamma

Terminal alphabet:
  24 input tokens + 24 output tokens = 48 total tokens.
  Input tokens are indexed 0-23, output tokens 24-47.

  Input/output token types (matching rna_grammar.py):
    Pair (a,b): a*4+b      (0-15)
    Left a:     16+a        (16-19)
    Right a:    20+a        (20-23)

Ready states:
  - A state is "ready" if its alignment context is V, W, or Start.
  - Ready states may only produce rules that consume an input token,
    bifurcate, or go to epsilon.
  - Non-ready states may NOT consume input tokens.

Rule types:
  Match transitions (from V/W/Start): consume input, produce output, go to M
  Insert transitions (from M,I,D): produce output only, go to I
  Delete transitions (from V/W/Start): consume input only, go to D
  Ready transitions: M -> V, I -> V, D -> W (no tokens)
  Bifurcation (from V/W/Start): split into two children at Start
  Epsilon (from V/W/Start): terminate
"""

import numpy as np
from ..core.rna import (
    N_CONTEXT, N_NUC, CTX_NN, CONTEXT_NAMES, classify_basepair,
)
from ..models.rna_grammar import (
    N_TOTAL_TERMINALS, left_terminal, right_terminal, pair_terminal,
    decode_terminal,
)


# Alignment context indices
ALN_M = 0   # Match: just consumed input and produced output
ALN_I = 1   # Insert: just produced output without consuming input
ALN_D = 2   # Delete: just consumed input without producing output
ALN_V = 3   # Ready post-M/I: waiting for input (uses beta for insert prob)
ALN_W = 4   # Ready post-D: waiting for input (uses gamma for insert prob)
ALN_R = ALN_V  # Backward compat alias — V is the "default" ready type
N_ALN = 5
ALN_NAMES = ['M', 'I', 'D', 'V', 'W']

# Special state
IDX_START = 0
N_WPTT_STATES = 1 + N_CONTEXT * N_CONTEXT * N_ALN  # 1 + 7*7*5 = 246

# Input/output token offsets
INPUT_OFFSET = 0
OUTPUT_OFFSET = N_TOTAL_TERMINALS  # 24
N_WPTT_TERMINALS = 2 * N_TOTAL_TERMINALS  # 48


def wptt_state_name(aln, in_ctx, out_ctx):
    """Human-readable name for a WPTT state."""
    return f'{ALN_NAMES[aln]}-{CONTEXT_NAMES[in_ctx]}-{CONTEXT_NAMES[out_ctx]}'


def wptt_state_index(aln, in_ctx, out_ctx):
    """Flat index for a WPTT state (1-based, 0 is Start)."""
    return 1 + aln * N_CONTEXT * N_CONTEXT + in_ctx * N_CONTEXT + out_ctx


def decode_wptt_state(idx):
    """Decode a flat state index to (aln, in_ctx, out_ctx). Start returns None."""
    if idx == IDX_START:
        return None
    idx -= 1
    aln = idx // (N_CONTEXT * N_CONTEXT)
    rem = idx % (N_CONTEXT * N_CONTEXT)
    in_ctx = rem // N_CONTEXT
    out_ctx = rem % N_CONTEXT
    return (aln, in_ctx, out_ctx)


def is_ready_state(idx):
    """Check if a state is a ready state (can consume input).

    Ready states are: Start, V (post-M/I), W (post-D).
    """
    if idx == IDX_START:
        return True
    info = decode_wptt_state(idx)
    return info[0] in (ALN_V, ALN_W)


def _emit_type_and_ctx(terminal_idx):
    """Given a terminal index (0-23), return (emit_type, new_context).

    emit_type: 'L', 'R', or 'LR'
    new_context: context class after this emission
    """
    etype, nucs = decode_terminal(terminal_idx)
    if etype == 'LR':
        ctx = classify_basepair(nucs[0], nucs[1])
    else:
        ctx = CTX_NN  # single-base emission resets context
    return etype, ctx



class WPTTRule:
    """A single WPTT transition rule.

    Attributes:
        src: source state index
        rule_type: 'match', 'insert', 'delete', 'ready', 'bifurcation', 'epsilon'
        input_token: input terminal consumed (None for insert/ready/epsilon)
        output_token: output terminal produced (None for delete/ready/epsilon)
        dst: destination state index (for linear rules)
        dst_left, dst_right: destination states (for bifurcation)
        weight: rule weight/probability
    """
    __slots__ = ['src', 'rule_type', 'input_token', 'output_token',
                 'dst', 'dst_left', 'dst_right', 'weight']

    def __init__(self, src, rule_type, weight,
                 input_token=None, output_token=None,
                 dst=None, dst_left=None, dst_right=None):
        self.src = src
        self.rule_type = rule_type
        self.input_token = input_token
        self.output_token = output_token
        self.dst = dst
        self.dst_left = dst_left
        self.dst_right = dst_right
        self.weight = weight


class WPTT:
    """Weighted Parse-tree Pair Transducer.

    Operates on parse trees rather than strings. Each node in the input
    parse tree is transduced to produce an output parse tree node.

    Attributes:
        n_states: number of states (246)
        state_names: list of state name strings
        rules: list of WPTTRule objects
        rules_from: dict mapping src_state -> list of WPTTRule
    """

    def __init__(self, state_names, rules):
        self.n_states = len(state_names)
        self.state_names = state_names
        self.rules = rules
        self.rules_from = {}
        for r in rules:
            self.rules_from.setdefault(r.src, []).append(r)

    def ready_states(self):
        """Return set of state indices that are ready states."""
        return {i for i in range(self.n_states) if is_ready_state(i)}

    def rules_for(self, src):
        """Get all rules from a source state."""
        return self.rules_from.get(src, [])


def build_wptt_transducer(weights=None):
    """Build the full 246-state WPTT.

    Args:
        weights: optional dict mapping rule keys to weights.
                 Default: uniform weights, normalized per source state.

    Returns:
        WPTT object.
    """
    # Build state names
    state_names = ['Start']
    for aln in range(N_ALN):
        for in_ctx in range(N_CONTEXT):
            for out_ctx in range(N_CONTEXT):
                state_names.append(wptt_state_name(aln, in_ctx, out_ctx))

    all_rules = []

    for src in range(N_WPTT_STATES):
        src_ready = is_ready_state(src)
        if src == IDX_START:
            src_in = CTX_NN
            src_out = CTX_NN
        else:
            _, src_in, src_out = decode_wptt_state(src)

        raw = []  # collect (WPTTRule, key) for normalization

        if src_ready:
            # --- Match: consume input token, produce output token -> M ---
            for in_tok in range(N_TOTAL_TERMINALS):
                in_etype, in_ctx = _emit_type_and_ctx(in_tok)
                for out_tok in range(N_TOTAL_TERMINALS):
                    out_etype, out_ctx = _emit_type_and_ctx(out_tok)
                    if in_etype != out_etype:
                        continue
                    dst = wptt_state_index(ALN_M, in_ctx, out_ctx)
                    key = ('match', src, in_tok, out_tok)
                    w = weights.get(key, 1.0) if weights else 1.0
                    raw.append(WPTTRule(src, 'match', w,
                                       input_token=in_tok, output_token=out_tok,
                                       dst=dst))

            # --- Delete: consume input token, no output -> D ---
            for in_tok in range(N_TOTAL_TERMINALS):
                in_etype, in_ctx = _emit_type_and_ctx(in_tok)
                dst = wptt_state_index(ALN_D, in_ctx, src_out)
                key = ('delete', src, in_tok)
                w = weights.get(key, 1.0) if weights else 1.0
                raw.append(WPTTRule(src, 'delete', w,
                                   input_token=in_tok, dst=dst))

            # --- Bifurcation: both children get Start ---
            key = ('bif', src)
            w = weights.get(key, 1.0) if weights else 1.0
            raw.append(WPTTRule(src, 'bifurcation', w,
                                dst_left=IDX_START, dst_right=IDX_START))

            # --- Epsilon: terminate ---
            key = ('eps', src)
            w = weights.get(key, 1.0) if weights else 1.0
            raw.append(WPTTRule(src, 'epsilon', w))

        # Non-ready states (M, I, D) can insert or transition to ready
        if not src_ready:
            aln, _, _ = decode_wptt_state(src)

            # --- Insert: produce output token, no input -> I ---
            for out_tok in range(N_TOTAL_TERMINALS):
                out_etype, out_ctx = _emit_type_and_ctx(out_tok)
                dst = wptt_state_index(ALN_I, src_in, out_ctx)
                key = ('insert', src, out_tok)
                w = weights.get(key, 1.0) if weights else 1.0
                raw.append(WPTTRule(src, 'insert', w,
                                   output_token=out_tok, dst=dst))

            # --- Ready transition: M,I -> V; D -> W ---
            if aln in (ALN_M, ALN_I):
                dst = wptt_state_index(ALN_V, src_in, src_out)
            else:  # ALN_D
                dst = wptt_state_index(ALN_W, src_in, src_out)
            key = ('ready', src)
            w = weights.get(key, 1.0) if weights else 1.0
            raw.append(WPTTRule(src, 'ready', w, dst=dst))

        # Normalize weights
        total = sum(r.weight for r in raw)
        if total > 0:
            for r in raw:
                r.weight /= total

        all_rules.extend(raw)

    return WPTT(state_names, all_rules)


def tkf_wptt_weights(ins_rate, del_rate, t, subst_matrix=None):
    """Compute WPTT rule weights from TKF91 parameters.

    Maps TKF91 Pair HMM transition probabilities and substitution model
    to WPTT rule weights. The two ready states (V, W) get different
    transition probabilities reflecting TKF's state-dependent behavior:
      V (post-M/I): uses beta for insert, tau[M,*] for next column
      W (post-D): uses gamma for insert, tau[D,*] for next column

    For ready states (V, W, Start):
      match:  P(M|prev) * P(out_nuc | in_nuc, t) — with substitution
      delete: P(D|prev)  — marginalizing over ancestor nucleotide
      bif:    P(M|prev) * (bifurcation fraction)
      eps:    P(E|prev) — end probability

    For non-ready states (M, I, D):
      insert: P(I|prev) * P(out_nuc) — insertion with stationary freq
      ready:  1 - P(I|prev) — transition to V or W

    Args:
        ins_rate: TKF insertion rate (lambda)
        del_rate: TKF deletion rate (mu)
        t: branch length
        subst_matrix: (4, 4) array P(out | in, t), or None for JC69

    Returns:
        weights: dict mapping WPTT rule keys to weight values
    """
    import jax.numpy as jnp
    from ..core.bdi import tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa

    # Compute TKF91 parameters directly
    alpha = float(tkf_alpha(del_rate, t))    # survival prob: exp(-mu*t)
    beta = float(tkf_beta(ins_rate, del_rate, t))   # insert prob post-M/I
    gamma = float(tkf_gamma(ins_rate, del_rate, t))  # insert prob post-D
    kappa = float(tkf_kappa(ins_rate, del_rate))     # lambda/mu

    # WPTT decomposes TKF91 transitions into two layers:
    #   Non-ready (M,I,D): insert or become ready
    #   Ready (V,W,Start): match, delete, bifurcate, or epsilon
    #
    # From the 5×5 TKF91 matrix (e.g. M row):
    #   tau[M,M] = (1-beta)*kappa*alpha    → non-ready P(ready)=1-beta, ready P(match)=kappa*alpha
    #   tau[M,I] = beta                    → non-ready P(insert)=beta
    #   tau[M,D] = (1-beta)*kappa*(1-alpha)→ ready P(delete)=kappa*(1-alpha)
    #   tau[M,E] = (1-beta)*(1-kappa)      → ready P(eps)=1-kappa
    #
    # When kappa=1 (lambda=mu), P(eps)=0 which prevents termination.
    # Use a floor to ensure the WPTT can always terminate (the SCFG
    # and recognizer determine actual structure, not the WPTT eps weight).
    p_end = max(1.0 - kappa, 0.01)

    # Ready-state weights (same for V, W, and Start — the V/W distinction
    # only affects the insert probability at non-ready states)
    p_match = kappa * alpha        # ancestor position survives and is matched
    p_delete = kappa * (1 - alpha) # ancestor position dies (deleted)
    # build_wptt_transducer normalizes per source state, so these are
    # relative weights — no need to sum to exactly 1

    # Non-ready insert probabilities
    v_insert = beta   # post-M/I uses beta
    w_insert = gamma  # post-D uses gamma

    # Substitution matrix (4x4): P(out_nuc | in_nuc, t)
    if subst_matrix is None:
        from ..core.ctmc import rate_matrix_jc69, transition_matrix
        Q, pi = rate_matrix_jc69()
        subst_matrix = np.array(transition_matrix(Q, t))
        pi = np.array(pi)
    else:
        subst_matrix = np.asarray(subst_matrix)
        pi = np.ones(4) / 4  # assume uniform if not provided

    weights = {}
    bif_frac = 0.1  # fraction of match probability allocated to bifurcation

    for src in range(N_WPTT_STATES):
        src_ready = is_ready_state(src)

        if src_ready:
            # Ready-state weights are the same for V, W, Start
            # (V/W distinction only affects insert at non-ready states)

            # Match: weight = p_match * P(out_nuc | in_nuc)
            for in_tok in range(N_TOTAL_TERMINALS):
                in_etype, _ = _emit_type_and_ctx(in_tok)
                _, in_nucs = decode_terminal(in_tok)

                for out_tok in range(N_TOTAL_TERMINALS):
                    out_etype, _ = _emit_type_and_ctx(out_tok)
                    if in_etype != out_etype:
                        continue
                    _, out_nucs = decode_terminal(out_tok)

                    if in_etype == 'LR':
                        sp = (subst_matrix[in_nucs[0], out_nucs[0]] *
                              subst_matrix[in_nucs[1], out_nucs[1]])
                    else:
                        sp = subst_matrix[in_nucs[0], out_nucs[0]]

                    key = ('match', src, in_tok, out_tok)
                    weights[key] = p_match * (1 - bif_frac) * sp

            # Delete
            for in_tok in range(N_TOTAL_TERMINALS):
                key = ('delete', src, in_tok)
                weights[key] = p_delete / N_TOTAL_TERMINALS

            # Bifurcation
            key = ('bif', src)
            weights[key] = p_match * bif_frac

            # Epsilon
            key = ('eps', src)
            weights[key] = p_end

        else:
            # Non-ready: insert or transition to ready
            # Insert probability depends on which ready state we came from
            aln, _, _ = decode_wptt_state(src)
            if aln == ALN_D:
                p_ins = w_insert  # post-D uses gamma
            else:
                p_ins = v_insert  # post-M/I uses beta

            for out_tok in range(N_TOTAL_TERMINALS):
                _, out_nucs = decode_terminal(out_tok)
                out_etype, _ = _emit_type_and_ctx(out_tok)
                if out_etype == 'LR':
                    ip = pi[out_nucs[0]] * pi[out_nucs[1]]
                else:
                    ip = pi[out_nucs[0]]
                key = ('insert', src, out_tok)
                weights[key] = p_ins * ip

            # Ready transition: 1 - p_insert
            key = ('ready', src)
            weights[key] = 1.0 - p_ins

    return weights


def wptt_rule_counts(wptt):
    """Count rules by type."""
    counts = {}
    for r in wptt.rules:
        counts[r.rule_type] = counts.get(r.rule_type, 0) + 1
    return counts
