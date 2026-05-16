"""Parse tree transducer for context-free sequence evolution.

A tree transducer maps a source parse tree (from a singlet SCFG) to a
target parse tree, modeling structure-preserving sequence evolution.
This is the context-free generalization of the pair HMM / WFST evolver.

For each node in the source parse tree:
- Terminal nodes: apply substitution model (match, insert, delete)
- Unary/binary nodes: recurse on children, preserving tree structure

The transducer defines P(target_tree | source_tree, t) and thus
P(descendant | ancestor, t) for structured sequences (e.g., RNA).

Key components:
- ParseTree: recursive tree data structure
- PairSCFG: pair stochastic context-free grammar
- evolve_tree: stochastic tree transducer (forward sampling)
- pair_scfg_inside: Inside algorithm for pair SCFGs
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class ParseTree:
    """A parse tree node.

    Attributes:
        symbol: nonterminal index or terminal value
        is_terminal: True if this is a leaf (terminal symbol)
        children: list of child ParseTree nodes
        weight: production weight used at this node
    """
    symbol: int
    is_terminal: bool = False
    children: list = field(default_factory=list)
    weight: float = 1.0

    def leaves(self):
        """Return the sequence of terminal symbols (left-to-right)."""
        if self.is_terminal:
            return [self.symbol]
        if not self.children:
            return []  # internal node with no children (epsilon)
        result = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def size(self):
        """Number of nodes in the tree."""
        if self.is_terminal:
            return 1
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        """Maximum depth of the tree."""
        if self.is_terminal:
            return 0
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)


def tree_from_cyk(parse_tuple, grammar):
    """Convert a CYK parse tuple to a ParseTree.

    The CYK traceback returns nested tuples like:
    - (lhs, terminal)  for terminal productions
    - (lhs, child)     for unary productions
    - (lhs, left, right) for binary productions
    - (lhs, terminal, child) for right-linear productions

    Args:
        parse_tuple: from grammar.cyk() traceback
        grammar: the WCFG used for parsing

    Returns:
        ParseTree
    """
    if parse_tuple is None:
        # Empty production node (no children, no terminals)
        return ParseTree(0)
    if len(parse_tuple) == 2:
        lhs, second = parse_tuple
        if isinstance(second, tuple):
            # Unary: (lhs, child_tuple)
            child = tree_from_cyk(second, grammar)
            return ParseTree(lhs, children=[child] if child else [])
        elif second is None:
            # Empty production: (lhs, None)
            return ParseTree(lhs)
        else:
            # Terminal: (lhs, terminal_value)
            return ParseTree(lhs, children=[ParseTree(second, is_terminal=True)])
    elif len(parse_tuple) == 3:
        lhs, second, third = parse_tuple
        if isinstance(second, int) and not isinstance(second, bool):
            if isinstance(third, tuple):
                # Right-linear: (lhs, terminal, child_tuple)
                term = ParseTree(second, is_terminal=True)
                child = tree_from_cyk(third, grammar)
                children = [term]
                if child:
                    children.append(child)
                return ParseTree(lhs, children=children)
            elif third is None:
                # Right-linear with None child: (lhs, terminal, None)
                term = ParseTree(second, is_terminal=True)
                return ParseTree(lhs, children=[term])
        if isinstance(second, tuple) and isinstance(third, tuple):
            # Binary: (lhs, left_tuple, right_tuple)
            left = tree_from_cyk(second, grammar)
            right = tree_from_cyk(third, grammar)
            children = []
            if left:
                children.append(left)
            if right:
                children.append(right)
            return ParseTree(lhs, children=children)
    return ParseTree(0)


def evolve_tree(rng, source_tree, sub_matrix, pi, indel_model=None):
    """Evolve a parse tree via a tree transducer.

    Traverses the source parse tree top-down. At each node:
    - Terminal leaf: apply substitution model to produce descendant character
    - Internal node: recurse on children, preserving tree structure

    With indel_model, can also insert/delete subtrees:
    - Before/after each child, may insert new terminal nodes
    - Each child may be deleted (skipped in output)

    Args:
        rng: numpy RandomState
        source_tree: ParseTree of ancestor sequence
        sub_matrix: (A, A) substitution probability matrix P(b|a)
        pi: (A,) equilibrium distribution
        indel_model: optional dict with:
            'ins_prob': probability of inserting before each position
            'del_prob': probability of deleting a terminal
            'ext_prob': probability of extending an insertion run

    Returns:
        target_tree: ParseTree of descendant sequence
        alignment: list of (source_leaf_idx, target_leaf_idx) or None
        log_prob: log probability of the transformation
    """
    A = len(pi)
    alignment = []
    log_prob = 0.0
    src_leaf_counter = [0]
    tgt_leaf_counter = [0]

    def _insert_terminals(rng_state):
        """Sample a run of inserted terminals."""
        nonlocal log_prob
        if indel_model is None:
            return []
        ins_prob = indel_model.get('ins_prob', 0.0)
        ext_prob = indel_model.get('ext_prob', 0.0)
        inserted = []
        if rng_state.random() < ins_prob:
            log_prob += np.log(max(ins_prob, 1e-300))
            while True:
                char = rng_state.choice(A, p=pi)
                log_prob += np.log(max(pi[char], 1e-300))
                inserted.append(ParseTree(int(char), is_terminal=True))
                alignment.append((None, tgt_leaf_counter[0]))
                tgt_leaf_counter[0] += 1
                if rng_state.random() >= ext_prob:
                    log_prob += np.log(max(1 - ext_prob, 1e-300))
                    break
                log_prob += np.log(max(ext_prob, 1e-300))
        else:
            if indel_model is not None:
                log_prob += np.log(max(1 - ins_prob, 1e-300))
        return inserted

    def _evolve_node(node, rng_state):
        nonlocal log_prob

        if node.is_terminal:
            # Check for deletion
            if indel_model is not None:
                del_prob = indel_model.get('del_prob', 0.0)
                if rng_state.random() < del_prob:
                    log_prob += np.log(max(del_prob, 1e-300))
                    alignment.append((src_leaf_counter[0], None))
                    src_leaf_counter[0] += 1
                    return None  # deleted
                log_prob += np.log(max(1 - del_prob, 1e-300))

            # Substitution
            src_char = node.symbol
            probs = sub_matrix[src_char]
            tgt_char = rng_state.choice(A, p=probs)
            log_prob += np.log(max(probs[tgt_char], 1e-300))

            alignment.append((src_leaf_counter[0], tgt_leaf_counter[0]))
            src_leaf_counter[0] += 1
            tgt_leaf_counter[0] += 1
            return ParseTree(int(tgt_char), is_terminal=True)

        # Internal node: evolve children
        new_children = []
        for child in node.children:
            # Insert before child
            new_children.extend(_insert_terminals(rng_state))
            evolved = _evolve_node(child, rng_state)
            if evolved is not None:
                new_children.append(evolved)

        # Insert after last child
        new_children.extend(_insert_terminals(rng_state))

        return ParseTree(node.symbol, children=new_children)

    target_tree = _evolve_node(source_tree, rng)
    return target_tree, alignment, log_prob


def build_pair_scfg_from_singlet(singlet_grammar, sub_matrix, pi,
                                  ins_prob=0.0, del_prob=0.0):
    """Build a pair SCFG from a singlet grammar and evolution model.

    For each singlet production, creates corresponding pair productions:
    - A -> a : creates A_pair -> M(a,b) for each b (substitution)
    - A -> B : creates A_pair -> B_pair (unary pass-through)
    - A -> B C : creates A_pair -> B_pair C_pair (binary pass-through)
    - A -> a B : creates A_pair -> M(a,b) B_pair (right-linear match)

    When ins_prob > 0 or del_prob > 0, additional productions model
    insertions and deletions at each position.

    Args:
        singlet_grammar: WCFG for ancestor sequences
        sub_matrix: (A, A) substitution probability matrix
        pi: (A,) equilibrium distribution
        ins_prob: insertion probability per position
        del_prob: deletion probability per position

    Returns:
        pair_grammar: WCFG (pair grammar)
        n_chars: alphabet size
    """
    from ..grammar.scfg import WCFG, Production, build_grammar

    n_chars = len(pi)
    n_pair_terminals = n_chars * n_chars + 2 * n_chars

    # Create pair nonterminals (one per singlet nonterminal)
    pair_nts = [f"{name}_pair" for name in singlet_grammar.nonterminals]
    rules = []

    for p in singlet_grammar.productions:
        if p.is_terminal:
            # A -> a : create A_pair -> M(a,b) for each b
            a = p.rhs[0]
            for b in range(n_chars):
                t_idx = a * n_chars + b
                w = p.weight * float(sub_matrix[a, b])
                if w > 1e-30:
                    rules.append((pair_nts[p.lhs], [(t_idx, 'T')], w))

        elif p.is_unary:
            # A -> B : A_pair -> B_pair
            B_pair = pair_nts[p.rhs[0]]
            rules.append((pair_nts[p.lhs], [(B_pair, 'N')], p.weight))

        elif p.is_binary:
            # A -> B C : A_pair -> B_pair C_pair
            B_pair = pair_nts[p.rhs[0]]
            C_pair = pair_nts[p.rhs[1]]
            rules.append((pair_nts[p.lhs],
                         [(B_pair, 'N'), (C_pair, 'N')], p.weight))

        elif p.is_right_linear:
            # A -> a B : A_pair -> M(a,b) B_pair
            a = p.rhs[0]
            B_pair = pair_nts[p.rhs[1]]
            for b in range(n_chars):
                t_idx = a * n_chars + b
                w = p.weight * float(sub_matrix[a, b])
                if w > 1e-30:
                    rules.append((pair_nts[p.lhs],
                                 [(t_idx, 'T'), (B_pair, 'N')], w))

        elif p.is_empty:
            # A -> ε : A_pair -> ε
            rules.append((pair_nts[p.lhs], [], p.weight))

    return build_grammar(pair_nts, n_pair_terminals, rules,
                        start=pair_nts[singlet_grammar.start]), n_chars


def _build_guide_mapping(guide_alignment, Lx, Ly):
    """Build monotone mapping from x-boundary to y-boundary positions.

    Walks through the guide alignment and records the y-boundary position
    at each x-boundary position. The result g satisfies:
        g[0] = 0, g[Lx] = Ly, and g is monotone non-decreasing.

    Args:
        guide_alignment: list of (x_idx_or_None, y_idx_or_None) pairs
        Lx, Ly: sequence lengths

    Returns:
        g: array of shape (Lx+1,) mapping x-boundary to y-boundary
    """
    g = np.zeros(Lx + 1, dtype=int)
    xb, yb = 0, 0
    for x_idx, y_idx in guide_alignment:
        if x_idx is not None and y_idx is not None:
            xb += 1
            yb += 1
            g[xb] = yb
        elif x_idx is not None:
            xb += 1
            g[xb] = yb
        elif y_idx is not None:
            yb += 1
    for i in range(xb + 1, Lx + 1):
        g[i] = Ly
    return g


def _in_band(ix, iy, g, k, Ly):
    """Check if boundary (ix, iy) is within band k of guide mapping g."""
    return max(0, g[ix] - k) <= iy <= min(Ly, g[ix] + k)


def pair_scfg_inside(grammar, x_seq, y_seq, n_chars,
                     guide_alignment=None, band_width=None):
    """Inside algorithm for a pair SCFG, optionally band-constrained.

    Computes I[A, ix, jx, iy, jy] = P(x[ix:jx], y[iy:jy] | A)
    for all nonterminals A and span pairs.

    Without banding: O(N * Lx^3 * Ly^3) — expensive but correct.
    With banding: O(N * La^3 * k^3) where La <= Lx + Ly, k = band_width.

    The band constraint restricts which cells are computed: a span
    (ix, jx, iy, jy) is only computed if both boundary points (ix, iy)
    and (jx, jy) are within band_width of the guide alignment path.
    Setting band_width >= Lx + Ly recovers the full unconstrained DP.

    Args:
        grammar: WCFG (pair SCFG)
        x_seq: (Lx,) ancestor sequence
        y_seq: (Ly,) descendant sequence
        n_chars: alphabet size
        guide_alignment: optional list of (x_idx_or_None, y_idx_or_None)
            pairs defining a guide alignment for banding
        band_width: optional int, band half-width k around guide alignment

    Returns:
        log_prob: log P(x_seq, y_seq | grammar)
    """
    Lx = len(x_seq)
    Ly = len(y_seq)
    N = grammar.n_nonterminals
    NEG_INF = -1e30
    n_sq = n_chars * n_chars

    # Build guide mapping for band constraint
    banded = guide_alignment is not None and band_width is not None
    if banded:
        g = _build_guide_mapping(guide_alignment, Lx, Ly)
        k = band_width
    else:
        g = None
        k = max(Lx, Ly) + 1  # effectively unbanded

    def in_band(ix, iy):
        if not banded:
            return True
        return _in_band(ix, iy, g, k, Ly)

    def iy_range(ix, lo, hi):
        """Valid iy values at x-boundary ix within [lo, hi]."""
        if not banded:
            return range(lo, hi)
        y_lo = max(lo, g[ix] - k)
        y_hi = min(hi, g[ix] + k + 1)
        if y_lo >= y_hi:
            return range(0, 0)
        return range(y_lo, y_hi)

    # Use sparse dict for banded case, dense array otherwise
    if banded:
        log_I_dict = {}

        def get_I(a, ix, jx, iy, jy):
            return log_I_dict.get((a, ix, jx, iy, jy), NEG_INF)

        def set_I(a, ix, jx, iy, jy, val):
            log_I_dict[(a, ix, jx, iy, jy)] = val

        def logaddexp_I(a, ix, jx, iy, jy, val):
            old = log_I_dict.get((a, ix, jx, iy, jy), NEG_INF)
            log_I_dict[(a, ix, jx, iy, jy)] = np.logaddexp(old, val)
    else:
        log_I = np.full((N, Lx + 1, Lx + 1, Ly + 1, Ly + 1), NEG_INF)

        def get_I(a, ix, jx, iy, jy):
            return log_I[a, ix, jx, iy, jy]

        def set_I(a, ix, jx, iy, jy, val):
            log_I[a, ix, jx, iy, jy] = val

        def logaddexp_I(a, ix, jx, iy, jy, val):
            log_I[a, ix, jx, iy, jy] = np.logaddexp(
                log_I[a, ix, jx, iy, jy], val)

    def decode_terminal(t):
        if t < n_sq:
            return 'M', t // n_chars, t % n_chars
        elif t < n_sq + n_chars:
            return 'I', None, t - n_sq
        else:
            return 'D', t - n_sq - n_chars, None

    # Precompute unary closure matrix U = (I - W)^{-1}
    log_U = np.full((N, N), NEG_INF)
    for a in range(N):
        log_U[a, a] = 0.0
    W = np.full((N, N), NEG_INF)
    for p in grammar.productions:
        if p.is_unary:
            W[p.lhs, p.rhs[0]] = np.logaddexp(
                W[p.lhs, p.rhs[0]], np.log(max(p.weight, 1e-300)))
    Wk = W.copy()
    for _ in range(N):
        for a in range(N):
            for b in range(N):
                log_U[a, b] = np.logaddexp(log_U[a, b], Wk[a, b])
        new_Wk = np.full((N, N), NEG_INF)
        for a in range(N):
            for b in range(N):
                for c in range(N):
                    new_Wk[a, b] = np.logaddexp(new_Wk[a, b], Wk[a, c] + W[c, b])
        if np.allclose(np.exp(np.maximum(new_Wk, -100)), 0, atol=1e-15):
            break
        Wk = new_Wk

    def _unary_close(ix, jx, iy, jy):
        vals = np.array([get_I(a, ix, jx, iy, jy) for a in range(N)])
        for a in range(N):
            new_val = NEG_INF
            for b in range(N):
                new_val = np.logaddexp(new_val, log_U[a, b] + vals[b])
            set_I(a, ix, jx, iy, jy, new_val)

    # Base case: empty spans — (ix, ix, iy, iy) where both boundaries valid
    for ix in range(Lx + 1):
        for iy in iy_range(ix, 0, Ly + 1):
            for p in grammar.productions:
                if p.is_empty:
                    logaddexp_I(p.lhs, ix, ix, iy, iy,
                                np.log(max(p.weight, 1e-300)))
            _unary_close(ix, ix, iy, iy)

    # Terminal productions — only at band-valid cells
    for p in grammar.productions:
        if p.is_terminal:
            typ, xc, yc = decode_terminal(p.rhs[0])
            log_w = np.log(max(p.weight, 1e-300))
            if typ == 'M':
                for ix in range(Lx):
                    if int(x_seq[ix]) == xc:
                        for iy in iy_range(ix, 0, Ly):
                            if int(y_seq[iy]) == yc and in_band(ix + 1, iy + 1):
                                logaddexp_I(p.lhs, ix, ix+1, iy, iy+1, log_w)
            elif typ == 'I':
                for iy in range(Ly):
                    if int(y_seq[iy]) == yc:
                        for ix in range(Lx + 1):
                            if in_band(ix, iy) and in_band(ix, iy + 1):
                                logaddexp_I(p.lhs, ix, ix, iy, iy+1, log_w)
            elif typ == 'D':
                for ix in range(Lx):
                    if int(x_seq[ix]) == xc:
                        for iy in iy_range(ix, 0, Ly + 1):
                            if in_band(ix + 1, iy):
                                logaddexp_I(p.lhs, ix, ix+1, iy, iy, log_w)

    # Unary close terminal spans
    for ix in range(Lx + 1):
        for jx in range(ix, min(ix + 2, Lx + 1)):
            for iy in iy_range(ix, 0, Ly + 1):
                for jy in range(iy, min(iy + 2, Ly + 1)):
                    if jx == ix and jy == iy:
                        continue
                    if not in_band(jx, jy):
                        continue
                    _unary_close(ix, jx, iy, jy)

    # Fill by increasing total span (sx + sy)
    for total_span in range(1, Lx + Ly + 1):
        for sx in range(min(total_span, Lx) + 1):
            sy = total_span - sx
            if sy < 0 or sy > Ly:
                continue

            for ix in range(Lx - sx + 1):
                jx = ix + sx
                for iy in iy_range(ix, 0, Ly - sy + 1):
                    jy = iy + sy
                    if not in_band(jx, jy):
                        continue

                    # Right-linear: A -> t B
                    for p in grammar.productions:
                        if p.is_right_linear:
                            term, B = p.rhs
                            typ, xc, yc = decode_terminal(term)
                            log_w = np.log(max(p.weight, 1e-300))

                            if typ == 'M' and sx >= 1 and sy >= 1:
                                if int(x_seq[ix]) == xc and int(y_seq[iy]) == yc:
                                    val = log_w + get_I(B, ix+1, jx, iy+1, jy)
                                    if val > NEG_INF:
                                        logaddexp_I(p.lhs, ix, jx, iy, jy, val)
                            elif typ == 'I' and sy >= 1:
                                if int(y_seq[iy]) == yc:
                                    val = log_w + get_I(B, ix, jx, iy+1, jy)
                                    if val > NEG_INF:
                                        logaddexp_I(p.lhs, ix, jx, iy, jy, val)
                            elif typ == 'D' and sx >= 1:
                                if int(x_seq[ix]) == xc:
                                    val = log_w + get_I(B, ix+1, jx, iy, jy)
                                    if val > NEG_INF:
                                        logaddexp_I(p.lhs, ix, jx, iy, jy, val)

                    # Binary: A -> B C (split constrained to band)
                    for p in grammar.productions:
                        if p.is_binary:
                            B, C = p.rhs
                            log_w = np.log(max(p.weight, 1e-300))
                            for kx in range(ix, jx + 1):
                                for ky in iy_range(kx, iy, jy + 1):
                                    val = (log_w +
                                           get_I(B, ix, kx, iy, ky) +
                                           get_I(C, kx, jx, ky, jy))
                                    if val > NEG_INF:
                                        logaddexp_I(p.lhs, ix, jx, iy, jy, val)

                    _unary_close(ix, jx, iy, jy)

    return get_I(grammar.start, 0, Lx, 0, Ly)
