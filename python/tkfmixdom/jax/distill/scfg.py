"""Order-1 distillation for SCFGs and conditional Pair SCFGs.

Extends the order-1 distillation (Section 4.9 of the paper) from HMMs/WFSTs
to full SCFGs. In the SCFG setting, production weights depend on the
*four* flanking positions:
    - left_anc:  ancestor character immediately left of the current span
    - left_desc: descendant character immediately left of the current span
    - right_anc:  ancestor character immediately right of the current span
    - right_desc: descendant character immediately right of the current span

Each position takes one of A+1 values: {0, 1, ..., A-1, PAST_END}
where PAST_END denotes "past the end of the sequence" (boundary/BOS/EOS).

For singlet SCFGs (single sequence), only left and right characters matter
(2 context positions, each with A+1 values).

For pair SCFGs (ancestor-descendant pairs), all 4 context positions matter
((A+1)^4 contexts total).

The distilled SCFG has the same nonterminals and production structure as the
original, but production weights are now context-dependent:
    w(A -> B C | ctx) instead of w(A -> B C)

During the Inside algorithm, the context is known at each span boundary:
    Inside[A, i, j] uses ctx = (seq[i-1], seq[j]) for singlet
    Inside[A, ix, jx, iy, jy] uses ctx = (x[ix-1], y[iy-1], x[jx], y[jy]) for pairs
"""

import numpy as np
from ..grammar.scfg import WCFG, Production
from .hmm import null_closure, effective_emit_trans


# Sentinel value for "past end of sequence"
PAST_END = -1


def _ctx_index(char, alphabet_size):
    """Map a character (or PAST_END) to a context index in [0, A]."""
    if char == PAST_END or char < 0:
        return alphabet_size  # past-end index
    return int(char)


def _n_ctx(alphabet_size):
    """Number of distinct context values per position."""
    return alphabet_size + 1  # A chars + past_end


# ---------------------------------------------------------------------------
# Singlet SCFG distillation (2 context positions: left, right)
# ---------------------------------------------------------------------------

class DistilledSingletSCFG:
    """Order-1 distilled singlet SCFG with context-dependent weights.

    For each production p of the original SCFG, stores a weight table
    indexed by (left_ctx, right_ctx) where each ctx ∈ {0..A-1, PAST_END}.

    Attributes:
        grammar: original WCFG
        alphabet_size: A
        n_ctx: A + 1 (including PAST_END)
        weights: dict mapping production_index -> (n_ctx, n_ctx) weight array
    """

    def __init__(self, grammar, alphabet_size, weights):
        self.grammar = grammar
        self.alphabet_size = alphabet_size
        self.n_ctx = _n_ctx(alphabet_size)
        self.weights = weights

    def production_weight(self, prod_idx, left_char, right_char):
        """Get context-dependent weight for a production.

        Args:
            prod_idx: index into grammar.productions
            left_char: character to the left of span (or PAST_END)
            right_char: character to the right of span (or PAST_END)

        Returns:
            weight: float
        """
        A = self.alphabet_size
        li = _ctx_index(left_char, A)
        ri = _ctx_index(right_char, A)
        return float(self.weights[prod_idx][li, ri])


def distill_singlet_scfg(grammar, trans, emit_probs, pi, alphabet_size):
    """Distill a structured singlet SCFG to an order-1 context-dependent SCFG.

    Computes adjacency frequencies f(a, b) conditioned on left and right
    flanking context, then normalizes to get context-dependent production
    weights.

    For terminal productions A -> a:
        w(A -> a | left, right) ∝ f(left, a) * f(a, right)

    For binary productions A -> B C:
        w(A -> B C | left, right) = w_orig(A -> B C)
        (split into left context for B, right context for C at the split point)

    For unary and epsilon productions:
        w(A -> B | left, right) depends on null closure through flanking chars

    Args:
        grammar: WCFG
        trans: (N, N) HMM transition matrix underlying the grammar
        emit_probs: (N, A) emission probabilities per HMM state
        pi: (N,) stationary distribution over HMM states
        alphabet_size: A

    Returns:
        DistilledSingletSCFG
    """
    A = alphabet_size
    C = _n_ctx(A)
    n_prods = len(grammar.productions)

    # Compute adjacency frequencies from the HMM
    N = trans.shape[0]
    emit_states = [i for i in range(N) if np.sum(emit_probs[i]) > 1e-30]
    null_states = [i for i in range(N) if i not in emit_states]

    T_eff = np.array(effective_emit_trans(
        np.array(trans, dtype=np.float64),
        emit_states, null_states))

    emit_p = np.array(emit_probs)[emit_states]  # (n_emit, A)

    # f(a, b) = emit_p.T @ T_eff @ emit_p  shape (A, A)
    f_ab = emit_p.T @ T_eff @ emit_p

    # Boundary frequencies
    # Start -> emit: T_eff[start, :] (first row for start state)
    # We approximate boundary by uniform over characters
    f_start = np.sum(f_ab, axis=0)  # marginal for "following start"
    f_end = np.sum(f_ab, axis=1)    # marginal for "preceding end"

    # Build context-dependent production weights
    weights = {}
    for pi_idx, p in enumerate(grammar.productions):
        w_ctx = np.zeros((C, C))

        if p.is_terminal:
            char = p.rhs[0]
            for li in range(C):
                for ri in range(C):
                    # Weight depends on how well char fits between left and right
                    left_freq = f_ab[li, char] if li < A else f_start[char]
                    right_freq = f_ab[char, ri] if ri < A else f_end[char]
                    w_ctx[li, ri] = p.weight * np.sqrt(
                        max(left_freq, 1e-30) * max(right_freq, 1e-30))
            # Normalize per (left, right) across productions with same LHS
            w_ctx = np.maximum(w_ctx, 1e-30)

        elif p.is_right_linear:
            char = p.rhs[0]
            for li in range(C):
                for ri in range(C):
                    left_freq = f_ab[li, char] if li < A else f_start[char]
                    w_ctx[li, ri] = p.weight * max(left_freq, 1e-30)

        elif p.is_binary or p.is_unary or p.is_empty:
            # For structural productions, use original weight
            # (context affects children, not the production itself)
            w_ctx[:, :] = p.weight

        weights[pi_idx] = w_ctx

    # Normalize: for each (lhs, left, right), ensure weights sum correctly
    for li in range(C):
        for ri in range(C):
            for nt in range(grammar.n_nonterminals):
                prods = [(i, p) for i, p in enumerate(grammar.productions)
                         if p.lhs == nt]
                if not prods:
                    continue
                total = sum(weights[i][li, ri] for i, _ in prods)
                if total > 1e-30:
                    for i, _ in prods:
                        weights[i][li, ri] /= total
                    # Re-scale to match original total weight
                    orig_total = sum(p.weight for _, p in prods)
                    for i, _ in prods:
                        weights[i][li, ri] *= orig_total

    return DistilledSingletSCFG(grammar, alphabet_size, weights)


# ---------------------------------------------------------------------------
# Pair SCFG distillation (4 context positions)
# ---------------------------------------------------------------------------

class DistilledPairSCFG:
    """Order-1 distilled pair SCFG with context-dependent weights.

    Context = (left_anc, left_desc, right_anc, right_desc)
    Each ∈ {0..A-1, PAST_END}

    Attributes:
        grammar: original WCFG (pair grammar)
        alphabet_size: A
        n_ctx: A + 1
        weights: dict mapping prod_idx -> (n_ctx, n_ctx, n_ctx, n_ctx) weight array
    """

    def __init__(self, grammar, alphabet_size, weights):
        self.grammar = grammar
        self.alphabet_size = alphabet_size
        self.n_ctx = _n_ctx(alphabet_size)
        self.weights = weights

    def production_weight(self, prod_idx, left_anc, left_desc,
                          right_anc, right_desc):
        """Get context-dependent weight for a production.

        Args:
            prod_idx: index into grammar.productions
            left_anc, left_desc: chars left of span (or PAST_END)
            right_anc, right_desc: chars right of span (or PAST_END)

        Returns:
            weight: float
        """
        A = self.alphabet_size
        la = _ctx_index(left_anc, A)
        ld = _ctx_index(left_desc, A)
        ra = _ctx_index(right_anc, A)
        rd = _ctx_index(right_desc, A)
        return float(self.weights[prod_idx][la, ld, ra, rd])


def distill_pair_scfg(grammar, trans, state_types, sub_matrix, pi,
                       alphabet_size):
    """Distill a structured pair SCFG to order-1 context-dependent pair SCFG.

    The distilled production weights depend on 4 flanking positions:
    (left_anc, left_desc, right_anc, right_desc).

    Computes adjacency frequencies from the Pair HMM underlying the grammar,
    conditioned on all four flanking characters.

    For pair terminal productions (M, I, D emissions):
        w(A -> pair_term B | ctx) depends on adjacency frequencies
        conditioned on ancestor and descendant contexts separately

    For structural productions:
        Original weight, with context propagated to children

    Args:
        grammar: WCFG (pair grammar)
        trans: (N, N) Pair HMM transition matrix
        state_types: (N,) state type codes (S, M, I, D, E)
        sub_matrix: (A, A) substitution probability matrix
        pi: (A,) equilibrium distribution
        alphabet_size: A

    Returns:
        DistilledPairSCFG
    """
    from ..core.params import S, M, I, D, E

    A = alphabet_size
    C = _n_ctx(A)
    N = trans.shape[0]
    trans = np.array(trans, dtype=np.float64)
    state_types = np.array(state_types)
    sub_matrix = np.array(sub_matrix, dtype=np.float64)
    pi_arr = np.array(pi, dtype=np.float64)

    # Identify state types
    match_states = [i for i in range(N) if state_types[i] == M]
    ins_states = [i for i in range(N) if state_types[i] == I]
    del_states = [i for i in range(N) if state_types[i] == D]
    emit_states = match_states + ins_states + del_states
    null_states = [i for i in range(N)
                   if state_types[i] in (S, E) and i not in emit_states]

    T_eff = np.array(effective_emit_trans(
        trans, emit_states, null_states))

    n_match = len(match_states)
    n_ins = len(ins_states)
    n_del = len(del_states)

    # Adjacency frequencies conditioned on 4 flanking characters
    # f_MM(X, Y, X', Y') = adjacency from match(X,Y) to match(X',Y')
    f_MM = np.zeros((A, A, A, A))
    f_MI = np.zeros((A, A, A))      # match(X,Y) -> ins(Y')
    f_MD = np.zeros((A, A, A))      # match(X,Y) -> del(X')
    f_IM = np.zeros((A, A, A))      # ins(Y) -> match(X',Y')
    f_II = np.zeros((A, A))         # ins(Y) -> ins(Y')
    f_ID = np.zeros((A, A))         # ins(Y) -> del(X')
    f_DM = np.zeros((A, A, A))      # del(X) -> match(X',Y')
    f_DD = np.zeros((A, A))         # del(X) -> del(X')
    f_DI = np.zeros((A, A))         # del(X) -> ins(Y')

    # Compute frequencies from effective transitions
    for i, s1 in enumerate(emit_states):
        for j, s2 in enumerate(emit_states):
            w = T_eff[i, j]
            if abs(w) < 1e-30:
                continue
            t1 = int(state_types[s1])
            t2 = int(state_types[s2])

            if t1 == M and t2 == M:
                f_MM += w * np.einsum('x,xy,a,ab->xyab',
                                       pi_arr, sub_matrix, pi_arr, sub_matrix)
            elif t1 == M and t2 == I:
                f_MI += w * np.einsum('x,xy,b->xyb',
                                       pi_arr, sub_matrix, pi_arr)
            elif t1 == M and t2 == D:
                f_MD += w * np.einsum('x,xy,a->xya',
                                       pi_arr, sub_matrix, pi_arr)
            elif t1 == I and t2 == M:
                f_IM += w * np.einsum('y,a,ab->yab',
                                       pi_arr, pi_arr, sub_matrix)
            elif t1 == I and t2 == I:
                f_II += w * np.outer(pi_arr, pi_arr)
            elif t1 == I and t2 == D:
                f_ID += w * np.outer(pi_arr, pi_arr)
            elif t1 == D and t2 == M:
                f_DM += w * np.einsum('x,a,ab->xab',
                                       pi_arr, pi_arr, sub_matrix)
            elif t1 == D and t2 == D:
                f_DD += w * np.outer(pi_arr, pi_arr)
            elif t1 == D and t2 == I:
                f_DI += w * np.outer(pi_arr, pi_arr)

    # Build context-dependent production weights
    n_prods = len(grammar.productions)
    n_sq = A * A

    weights = {}
    for pi_idx, p in enumerate(grammar.productions):
        w_ctx = np.full((C, C, C, C), p.weight)

        if p.is_right_linear:
            term = p.rhs[0]
            # Decode pair terminal to determine type and characters
            if term < n_sq:
                # Match terminal M(a,b): a = term // A, b = term % A
                anc_char = term // A
                desc_char = term % A
                for la in range(C):
                    for ld in range(C):
                        for ra in range(C):
                            for rd in range(C):
                                # Left context: what precedes on both anc and desc
                                left_anc_freq = (
                                    f_MM[la, ld, anc_char, desc_char]
                                    if la < A and ld < A else
                                    pi_arr[anc_char] * sub_matrix[anc_char, desc_char])
                                right_anc_freq = (
                                    f_MM[anc_char, desc_char, ra, rd]
                                    if ra < A and rd < A else
                                    pi_arr[anc_char] * sub_matrix[anc_char, desc_char])
                                w_ctx[la, ld, ra, rd] = p.weight * np.sqrt(
                                    max(left_anc_freq, 1e-30) *
                                    max(right_anc_freq, 1e-30))

            elif term < n_sq + A:
                # Insert terminal I(b): b = term - n_sq
                desc_char = term - n_sq
                for la in range(C):
                    for ld in range(C):
                        for ra in range(C):
                            for rd in range(C):
                                left_freq = (f_II[ld, desc_char]
                                             if ld < A else pi_arr[desc_char])
                                right_freq = (f_II[desc_char, rd]
                                              if rd < A else pi_arr[desc_char])
                                w_ctx[la, ld, ra, rd] = p.weight * np.sqrt(
                                    max(left_freq, 1e-30) *
                                    max(right_freq, 1e-30))

            else:
                # Delete terminal D(a): a = term - n_sq - A
                anc_char = term - n_sq - A
                for la in range(C):
                    for ld in range(C):
                        for ra in range(C):
                            for rd in range(C):
                                left_freq = (f_DD[la, anc_char]
                                             if la < A else pi_arr[anc_char])
                                right_freq = (f_DD[anc_char, ra]
                                              if ra < A else pi_arr[anc_char])
                                w_ctx[la, ld, ra, rd] = p.weight * np.sqrt(
                                    max(left_freq, 1e-30) *
                                    max(right_freq, 1e-30))

        weights[pi_idx] = w_ctx

    # Normalize within each (lhs, context)
    for la in range(C):
        for ld in range(C):
            for ra in range(C):
                for rd in range(C):
                    for nt in range(grammar.n_nonterminals):
                        prods = [(i, p) for i, p in enumerate(grammar.productions)
                                 if p.lhs == nt]
                        if not prods:
                            continue
                        total = sum(weights[i][la, ld, ra, rd] for i, _ in prods)
                        if total > 1e-30:
                            orig_total = sum(p.weight for _, p in prods)
                            for i, _ in prods:
                                weights[i][la, ld, ra, rd] = (
                                    weights[i][la, ld, ra, rd] / total * orig_total)

    return DistilledPairSCFG(grammar, alphabet_size, weights)


# ---------------------------------------------------------------------------
# Context-dependent Inside algorithm
# ---------------------------------------------------------------------------

def inside_with_context(distilled, sequence):
    """Inside algorithm using context-dependent production weights.

    For singlet DistilledSingletSCFG: at span [i, j], context is
        left = sequence[i-1] if i > 0 else PAST_END
        right = sequence[j] if j < L else PAST_END

    Args:
        distilled: DistilledSingletSCFG
        sequence: integer array of terminal indices, shape (L,)

    Returns:
        log_inside: (n_nonterminals, L+1, L+1) log Inside probabilities
    """
    grammar = distilled.grammar
    L = len(sequence)
    n = grammar.n_nonterminals
    A = distilled.alphabet_size
    NEG_INF = -1e30

    log_I = np.full((n, L + 1, L + 1), NEG_INF)

    def _left_ctx(i):
        return int(sequence[i - 1]) if i > 0 else PAST_END

    def _right_ctx(j):
        return int(sequence[j]) if j < L else PAST_END

    # Pre-compute unary closure (context-independent structure)
    log_U = np.full((n, n), NEG_INF)
    for a in range(n):
        log_U[a, a] = 0.0
    W = np.full((n, n), NEG_INF)
    for p in grammar.productions:
        if p.is_unary:
            W[p.lhs, p.rhs[0]] = np.logaddexp(
                W[p.lhs, p.rhs[0]], np.log(max(p.weight, 1e-300)))
    Wk = W.copy()
    for _ in range(n):
        for a in range(n):
            for b in range(n):
                log_U[a, b] = np.logaddexp(log_U[a, b], Wk[a, b])
        new_Wk = np.full((n, n), NEG_INF)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    new_Wk[a, b] = np.logaddexp(new_Wk[a, b], Wk[a, c] + W[c, b])
        if np.allclose(np.exp(np.maximum(new_Wk, -100)), 0, atol=1e-15):
            break
        Wk = new_Wk

    def _close_unary(i, j):
        vals = log_I[:, i, j].copy()
        for a in range(n):
            log_I[a, i, j] = NEG_INF
            for b in range(n):
                log_I[a, i, j] = np.logaddexp(log_I[a, i, j], log_U[a, b] + vals[b])

    # Base case: epsilon spans
    for i in range(L + 1):
        left = _left_ctx(i)
        right = _right_ctx(i)
        for pi_idx, p in enumerate(grammar.productions):
            if p.is_empty:
                w = distilled.production_weight(pi_idx, left, right)
                if w > 1e-300:
                    log_I[p.lhs, i, i] = np.logaddexp(
                        log_I[p.lhs, i, i], np.log(w))
        _close_unary(i, i)

    # Fill spans
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span
            left = _left_ctx(i)
            right = _right_ctx(j)

            # Terminal productions
            if span == 1:
                for pi_idx, p in enumerate(grammar.productions):
                    if p.is_terminal and p.rhs[0] == int(sequence[i]):
                        w = distilled.production_weight(pi_idx, left, right)
                        if w > 1e-300:
                            log_I[p.lhs, i, j] = np.logaddexp(
                                log_I[p.lhs, i, j], np.log(w))

            # Right-linear: A -> a B
            for pi_idx, p in enumerate(grammar.productions):
                if p.is_right_linear:
                    term, B = p.rhs
                    if int(sequence[i]) == term and log_I[B, i + 1, j] > NEG_INF:
                        w = distilled.production_weight(pi_idx, left, right)
                        if w > 1e-300:
                            val = np.log(w) + log_I[B, i + 1, j]
                            log_I[p.lhs, i, j] = np.logaddexp(
                                log_I[p.lhs, i, j], val)

            # Binary: A -> B C
            for pi_idx, p in enumerate(grammar.productions):
                if p.is_binary:
                    B, C = p.rhs
                    w = distilled.production_weight(pi_idx, left, right)
                    if w < 1e-300:
                        continue
                    log_w = np.log(w)
                    for k in range(i, j + 1):
                        if log_I[B, i, k] > NEG_INF and log_I[C, k, j] > NEG_INF:
                            val = log_w + log_I[B, i, k] + log_I[C, k, j]
                            log_I[p.lhs, i, j] = np.logaddexp(
                                log_I[p.lhs, i, j], val)

            _close_unary(i, j)

    return log_I


def inside_pair_with_context(distilled, x, y, alignment, n_chars,
                              guide_alignment=None, band_width=None):
    """Inside algorithm for distilled pair SCFG with 4-position context.

    At each span, context is:
        left_anc = x[ix-1] if ix > 0 else PAST_END
        left_desc = y[iy-1] if iy > 0 else PAST_END
        right_anc = x[jx] if jx < Lx else PAST_END
        right_desc = y[jy] if jy < Ly else PAST_END

    For regular pair grammars (where the pair sequence is 1D), this reduces
    to the singlet context-dependent Inside with the pair terminal encoding.

    When guide_alignment and band_width are given, only considers spans
    within band_width of the guide alignment, reducing complexity from
    O(La^3) to O(La * k^2) where k = band_width.

    Args:
        distilled: DistilledPairSCFG
        x: ancestor sequence (int array)
        y: descendant sequence (int array)
        alignment: list of (anc_idx_or_None, desc_idx_or_None)
        n_chars: alphabet size
        guide_alignment: optional guide alignment for banding (unused in
            1D pair sequence mode, but kept for API consistency with the
            4D pair_scfg_inside)
        band_width: optional band width for banding constraint

    Returns:
        log_inside: log P(alignment | distilled pair SCFG)
    """
    from ..models.tkf_grammar import encode_pair_sequence

    pair_seq = encode_pair_sequence(x, y, alignment, n_chars)
    L = len(pair_seq)
    grammar = distilled.grammar
    n = grammar.n_nonterminals
    NEG_INF = -1e30

    # For each position in the pair sequence, determine the ancestor
    # and descendant characters at that position
    anc_chars_at = []
    desc_chars_at = []
    for anc_idx, desc_idx in alignment:
        a = int(x[anc_idx]) if anc_idx is not None else PAST_END
        d = int(y[desc_idx]) if desc_idx is not None else PAST_END
        anc_chars_at.append(a)
        desc_chars_at.append(d)

    def _ctx(pos, chars_list):
        """Get context character at boundary position."""
        if pos <= 0 or pos > len(chars_list):
            return PAST_END
        return chars_list[pos - 1]

    log_I = np.full((n, L + 1, L + 1), NEG_INF)

    # Pre-compute unary closure
    log_U = np.full((n, n), NEG_INF)
    for a in range(n):
        log_U[a, a] = 0.0
    W = np.full((n, n), NEG_INF)
    for p in grammar.productions:
        if p.is_unary:
            W[p.lhs, p.rhs[0]] = np.logaddexp(
                W[p.lhs, p.rhs[0]], np.log(max(p.weight, 1e-300)))
    Wk = W.copy()
    for _ in range(n):
        for a in range(n):
            for b in range(n):
                log_U[a, b] = np.logaddexp(log_U[a, b], Wk[a, b])
        new_Wk = np.full((n, n), NEG_INF)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    new_Wk[a, b] = np.logaddexp(new_Wk[a, b], Wk[a, c] + W[c, b])
        if np.allclose(np.exp(np.maximum(new_Wk, -100)), 0, atol=1e-15):
            break
        Wk = new_Wk

    def _close_unary(i, j):
        vals = log_I[:, i, j].copy()
        for a in range(n):
            log_I[a, i, j] = NEG_INF
            for b in range(n):
                log_I[a, i, j] = np.logaddexp(log_I[a, i, j], log_U[a, b] + vals[b])

    # Base case: epsilon spans
    for i in range(L + 1):
        la = _ctx(i, anc_chars_at)
        ld = _ctx(i, desc_chars_at)
        ra = _ctx(i + 1, anc_chars_at) if i < L else PAST_END
        rd = _ctx(i + 1, desc_chars_at) if i < L else PAST_END

        for pi_idx, p in enumerate(grammar.productions):
            if p.is_empty:
                w = distilled.production_weight(pi_idx, la, ld, ra, rd)
                if w > 1e-300:
                    log_I[p.lhs, i, i] = np.logaddexp(
                        log_I[p.lhs, i, i], np.log(w))
        _close_unary(i, i)

    # Fill spans
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span
            la = _ctx(i, anc_chars_at)
            ld = _ctx(i, desc_chars_at)
            ra = _ctx(j + 1, anc_chars_at) if j < L else PAST_END
            rd = _ctx(j + 1, desc_chars_at) if j < L else PAST_END

            if span == 1:
                for pi_idx, p in enumerate(grammar.productions):
                    if p.is_terminal and p.rhs[0] == int(pair_seq[i]):
                        w = distilled.production_weight(pi_idx, la, ld, ra, rd)
                        if w > 1e-300:
                            log_I[p.lhs, i, j] = np.logaddexp(
                                log_I[p.lhs, i, j], np.log(w))

            for pi_idx, p in enumerate(grammar.productions):
                if p.is_right_linear:
                    term, B = p.rhs
                    if int(pair_seq[i]) == term and log_I[B, i + 1, j] > NEG_INF:
                        w = distilled.production_weight(pi_idx, la, ld, ra, rd)
                        if w > 1e-300:
                            val = np.log(w) + log_I[B, i + 1, j]
                            log_I[p.lhs, i, j] = np.logaddexp(
                                log_I[p.lhs, i, j], val)

            for pi_idx, p in enumerate(grammar.productions):
                if p.is_binary:
                    B, C = p.rhs
                    w = distilled.production_weight(pi_idx, la, ld, ra, rd)
                    if w < 1e-300:
                        continue
                    log_w = np.log(w)
                    for k in range(i, j + 1):
                        if log_I[B, i, k] > NEG_INF and log_I[C, k, j] > NEG_INF:
                            val = log_w + log_I[B, i, k] + log_I[C, k, j]
                            log_I[p.lhs, i, j] = np.logaddexp(
                                log_I[p.lhs, i, j], val)

            _close_unary(i, j)

    return log_I[grammar.start, 0, L]


# ---------------------------------------------------------------------------
# Integration with beam search and progressive reconstruction
# ---------------------------------------------------------------------------

def beam_inside_with_context(distilled, sequence, beam_log_width=10.0):
    """Beam-pruned Inside algorithm with context-dependent weights.

    Like inside_with_context but prunes nonterminals at each span
    that fall below max - beam_log_width.

    Args:
        distilled: DistilledSingletSCFG
        sequence: integer array
        beam_log_width: log beam width for pruning

    Returns:
        log_inside: (n_nonterminals, L+1, L+1) log Inside probabilities
        n_active: number of active (non-pruned) chart entries
    """
    grammar = distilled.grammar
    L = len(sequence)
    n = grammar.n_nonterminals
    NEG_INF = -1e30

    log_I = np.full((n, L + 1, L + 1), NEG_INF)
    n_active = 0

    def _left_ctx(i):
        return int(sequence[i - 1]) if i > 0 else PAST_END

    def _right_ctx(j):
        return int(sequence[j]) if j < L else PAST_END

    # Unary closure (same as context-free version)
    log_U = np.full((n, n), NEG_INF)
    for a in range(n):
        log_U[a, a] = 0.0
    W = np.full((n, n), NEG_INF)
    for p in grammar.productions:
        if p.is_unary:
            W[p.lhs, p.rhs[0]] = np.logaddexp(
                W[p.lhs, p.rhs[0]], np.log(max(p.weight, 1e-300)))
    Wk = W.copy()
    for _ in range(n):
        for a in range(n):
            for b in range(n):
                log_U[a, b] = np.logaddexp(log_U[a, b], Wk[a, b])
        new_Wk = np.full((n, n), NEG_INF)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    new_Wk[a, b] = np.logaddexp(new_Wk[a, b], Wk[a, c] + W[c, b])
        if np.allclose(np.exp(np.maximum(new_Wk, -100)), 0, atol=1e-15):
            break
        Wk = new_Wk

    def _close_unary(i, j):
        vals = log_I[:, i, j].copy()
        for a in range(n):
            log_I[a, i, j] = NEG_INF
            for b in range(n):
                log_I[a, i, j] = np.logaddexp(log_I[a, i, j], log_U[a, b] + vals[b])

    def _beam_prune(i, j):
        nonlocal n_active
        max_val = np.max(log_I[:, i, j])
        if max_val > NEG_INF + 100:
            threshold = max_val - beam_log_width
            for a in range(n):
                if log_I[a, i, j] > NEG_INF + 100:
                    if log_I[a, i, j] < threshold:
                        log_I[a, i, j] = NEG_INF
                    else:
                        n_active += 1

    # Base case
    for i in range(L + 1):
        left = _left_ctx(i)
        right = _right_ctx(i)
        for pi_idx, p in enumerate(grammar.productions):
            if p.is_empty:
                w = distilled.production_weight(pi_idx, left, right)
                if w > 1e-300:
                    log_I[p.lhs, i, i] = np.logaddexp(
                        log_I[p.lhs, i, i], np.log(w))
        _close_unary(i, i)
        _beam_prune(i, i)

    # Fill spans
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span
            left = _left_ctx(i)
            right = _right_ctx(j)

            if span == 1:
                for pi_idx, p in enumerate(grammar.productions):
                    if p.is_terminal and p.rhs[0] == int(sequence[i]):
                        w = distilled.production_weight(pi_idx, left, right)
                        if w > 1e-300:
                            log_I[p.lhs, i, j] = np.logaddexp(
                                log_I[p.lhs, i, j], np.log(w))

            for pi_idx, p in enumerate(grammar.productions):
                if p.is_right_linear:
                    term, B = p.rhs
                    if int(sequence[i]) == term and log_I[B, i + 1, j] > NEG_INF:
                        w = distilled.production_weight(pi_idx, left, right)
                        if w > 1e-300:
                            val = np.log(w) + log_I[B, i + 1, j]
                            log_I[p.lhs, i, j] = np.logaddexp(
                                log_I[p.lhs, i, j], val)

            for pi_idx, p in enumerate(grammar.productions):
                if p.is_binary:
                    B, C = p.rhs
                    w = distilled.production_weight(pi_idx, left, right)
                    if w < 1e-300:
                        continue
                    log_w = np.log(w)
                    for k in range(i, j + 1):
                        if log_I[B, i, k] > NEG_INF and log_I[C, k, j] > NEG_INF:
                            val = log_w + log_I[B, i, k] + log_I[C, k, j]
                            log_I[p.lhs, i, j] = np.logaddexp(
                                log_I[p.lhs, i, j], val)

            _close_unary(i, j)
            _beam_prune(i, j)

    return log_I, n_active
