"""Weighted Context-Free Grammar (WCFG) framework.

Provides data structures and algorithms for WCFGs:
- Grammar construction with typed productions
- Nullability fixed-point iteration with expected counts
- Epsilon elimination with inverse count-mapping
- Inside/Outside algorithms (CYK-style, banded)
- Conversion to/from regular grammars (HMMs)

Grammars are specified as:
  - nonterminals: list of str
  - terminals: list of str (or int indices)
  - productions: list of (lhs, rhs, weight) where rhs is a tuple
  - start: str (start nonterminal)

Production types:
  - Terminal: A -> a  (rhs is a single terminal)
  - Unary: A -> B  (rhs is a single nonterminal)
  - Binary: A -> B C  (rhs is two nonterminals)
  - Right-linear: A -> a B  (terminal followed by nonterminal, L-emit)
  - Left-linear: A -> B a  (nonterminal followed by terminal, R-emit)
  - LR-linear: A -> a B b  (terminal, nonterminal, terminal, LR-emit)
  - Empty: A -> epsilon  (rhs is empty tuple)
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Production:
    """A single production rule."""
    lhs: int       # nonterminal index
    rhs: tuple     # (idx, ...) — nonterminal or terminal indices
    weight: float   # production weight (probability)
    rhs_types: tuple  # ('N', 'N') for binary, ('T',) for terminal, etc.

    @property
    def is_terminal(self):
        return len(self.rhs_types) == 1 and self.rhs_types[0] == 'T'

    @property
    def is_unary(self):
        return len(self.rhs_types) == 1 and self.rhs_types[0] == 'N'

    @property
    def is_binary(self):
        return len(self.rhs_types) == 2 and all(t == 'N' for t in self.rhs_types)

    @property
    def is_right_linear(self):
        """A -> a B (terminal followed by nonterminal, L-emit)."""
        return len(self.rhs_types) == 2 and self.rhs_types[0] == 'T' and self.rhs_types[1] == 'N'

    @property
    def is_left_linear(self):
        """A -> B a (nonterminal followed by terminal, R-emit)."""
        return len(self.rhs_types) == 2 and self.rhs_types[0] == 'N' and self.rhs_types[1] == 'T'

    @property
    def is_lr_linear(self):
        """A -> a B b (terminal, nonterminal, terminal, LR-emit)."""
        return len(self.rhs_types) == 3 and self.rhs_types == ('T', 'N', 'T')

    @property
    def is_empty(self):
        return len(self.rhs) == 0


@dataclass
class WCFG:
    """Weighted Context-Free Grammar.

    Attributes:
        nonterminals: list of nonterminal names
        terminals: list of terminal symbols (or alphabet size)
        productions: list of Production objects
        start: index of start nonterminal
    """
    nonterminals: list
    n_terminals: int
    productions: list
    start: int = 0

    @property
    def n_nonterminals(self):
        return len(self.nonterminals)

    def nt_index(self, name):
        return self.nonterminals.index(name)

    def productions_for(self, lhs):
        """Get all productions with given LHS."""
        return [p for p in self.productions if p.lhs == lhs]

    def is_regular(self):
        """Check if the grammar is regular (right-linear, unary, terminal, or empty)."""
        for p in self.productions:
            if p.is_binary or p.is_left_linear or p.is_lr_linear:
                return False
            if not (p.is_terminal or p.is_unary or p.is_empty or p.is_right_linear):
                return False
        return True

    def is_cnf(self):
        """Check if the grammar is in Chomsky Normal Form."""
        for p in self.productions:
            if p.is_empty or p.is_unary:
                return False
            if not (p.is_terminal or p.is_binary):
                return False
        return True


def build_grammar(nonterminals, n_terminals, rules, start=0):
    """Build a WCFG from a list of rules.

    Args:
        nonterminals: list of nonterminal names
        n_terminals: number of terminal symbols
        rules: list of (lhs_name, rhs_names, weight) where rhs_names is
               a list of (name, type) pairs, type='N' or 'T'
        start: index or name of start nonterminal

    Returns:
        WCFG
    """
    nt_map = {name: i for i, name in enumerate(nonterminals)}
    if isinstance(start, str):
        start = nt_map[start]

    productions = []
    for lhs_name, rhs_spec, weight in rules:
        lhs = nt_map[lhs_name]
        rhs = tuple()
        rhs_types = tuple()
        for name, typ in rhs_spec:
            if typ == 'N':
                rhs = rhs + (nt_map[name],)
            else:
                rhs = rhs + (name,)  # terminal index
            rhs_types = rhs_types + (typ,)
        productions.append(Production(lhs, rhs, weight, rhs_types))

    return WCFG(nonterminals, n_terminals, productions, start)


# --- Nullability fixed-point iteration ---

def compute_nullability(grammar):
    """Compute nullability for each nonterminal via fixed-point iteration.

    A nonterminal A is nullable if it can derive epsilon.
    Nullability(A) = sum over epsilon-deriving productions of their weights.

    Returns:
        nullability: array of shape (n_nonterminals,)
    """
    n = grammar.n_nonterminals
    null = np.zeros(n)

    for iteration in range(100):
        old_null = null.copy()
        new_null = np.zeros(n)
        for p in grammar.productions:
            if p.is_empty:
                new_null[p.lhs] += p.weight
            elif all(t == 'N' for t in p.rhs_types):
                child_null = np.prod([null[c] for c in p.rhs])
                new_null[p.lhs] += p.weight * child_null
        null = np.minimum(new_null, 1.0)
        if np.allclose(null, old_null, atol=1e-12):
            break

    return null


def compute_null_counts(grammar, nullability):
    """Compute expected rule counts conditioned on deriving epsilon.

    For each production p, compute the expected number of times p would be
    used in a null derivation of p.lhs.

    Returns:
        null_counts: dict mapping production index to expected count
    """
    n = grammar.n_nonterminals
    null_counts = np.zeros(len(grammar.productions))

    for iteration in range(100):
        old_counts = null_counts.copy()
        # For each nonterminal, sum the expected counts from its null derivations
        for pi, p in enumerate(grammar.productions):
            if p.is_empty:
                null_counts[pi] = p.weight
            elif all(t == 'N' for t in p.rhs_types):
                child_null_prod = np.prod([nullability[c] for c in p.rhs])
                if nullability[p.lhs] > 1e-30:
                    # This production contributes proportionally
                    null_counts[pi] = p.weight * child_null_prod

        if np.allclose(null_counts, old_counts, atol=1e-12):
            break

    return null_counts


def eliminate_epsilon(grammar, nullability, null_counts):
    """Eliminate epsilon productions and return the new grammar + count mapping.

    For each binary rule A -> B C where B is nullable:
      add A -> C with weight w * nullability(B)
    For each binary rule A -> B C where C is nullable:
      add A -> B with weight w * nullability(C)

    Returns:
        new_grammar: WCFG with no epsilon productions
        count_map: function that maps counts in new grammar to original grammar counts
    """
    new_prods = []
    # Map from new production index to (original_index, factor)
    origin_map = []

    for pi, p in enumerate(grammar.productions):
        if p.is_empty:
            continue  # skip epsilon productions

        if p.is_terminal:
            new_prods.append(p)
            origin_map.append([(pi, 1.0)])
        elif p.is_unary:
            new_prods.append(p)
            origin_map.append([(pi, 1.0)])
        elif p.is_binary:
            B, C = p.rhs
            # Keep original if both children can be non-null
            new_prods.append(p)
            origin_map.append([(pi, 1.0)])

            # Add A -> C if B is nullable
            if nullability[B] > 1e-30:
                new_prod = Production(
                    p.lhs, (C,), p.weight * nullability[B], ('N',)
                )
                new_prods.append(new_prod)
                origin_map.append([(pi, nullability[B])])

            # Add A -> B if C is nullable
            if nullability[C] > 1e-30:
                new_prod = Production(
                    p.lhs, (B,), p.weight * nullability[C], ('N',)
                )
                new_prods.append(new_prod)
                origin_map.append([(pi, nullability[C])])

    new_grammar = WCFG(
        grammar.nonterminals, grammar.n_terminals,
        new_prods, grammar.start
    )

    def map_counts(new_counts):
        """Map expected counts from epsilon-eliminated grammar to original."""
        orig_counts = np.zeros(len(grammar.productions))
        for new_pi, (new_count, origins) in enumerate(zip(new_counts, origin_map)):
            for orig_pi, factor in origins:
                orig_counts[orig_pi] += new_count * factor
        return orig_counts

    return new_grammar, map_counts


# --- Inside algorithm ---

def inside(grammar, sequence):
    """Inside algorithm for a WCFG.

    Computes the inside probabilities I[A, i, j] = P(sequence[i:j] | A)
    for all nonterminals A and spans [i, j).

    Uses CYK-style bottom-up filling in log-space.

    Args:
        grammar: WCFG (should be in CNF or near-CNF)
        sequence: integer array of terminal indices, shape (L,)

    Returns:
        log_inside: array of shape (n_nonterminals, L+1, L+1)
                    where log_inside[A, i, j] = log P(seq[i:j] | A)
    """
    L = len(sequence)
    n = grammar.n_nonterminals
    NEG_INF = -1e30

    log_I = np.full((n, L + 1, L + 1), NEG_INF)

    # Pre-compute unary closure matrix: U[A,B] = total weight of paths A =>* B
    # via unary productions only. This is (I - W)^{-1} in probability space,
    # computed as a geometric series W + W^2 + W^3 + ...
    log_U = np.full((n, n), NEG_INF)
    for a in range(n):
        log_U[a, a] = 0.0  # identity
    W = np.full((n, n), NEG_INF)
    for p in grammar.productions:
        if p.is_unary:
            W[p.lhs, p.rhs[0]] = np.logaddexp(
                W[p.lhs, p.rhs[0]], np.log(max(p.weight, 1e-300))
            )
    # Iterate W^k and accumulate
    Wk = W.copy()
    for _ in range(n):
        for a in range(n):
            for b in range(n):
                log_U[a, b] = np.logaddexp(log_U[a, b], Wk[a, b])
        # Wk = Wk @ W (in log-space)
        new_Wk = np.full((n, n), NEG_INF)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    new_Wk[a, b] = np.logaddexp(new_Wk[a, b], Wk[a, c] + W[c, b])
        if np.allclose(np.exp(np.maximum(new_Wk, -100)), 0, atol=1e-15):
            break
        Wk = new_Wk

    def _close_unary(i, j):
        """Apply unary closure to log_I[:, i, j]."""
        # new_I[A, i, j] = logsumexp_B(log_U[A, B] + log_I[B, i, j])
        vals = log_I[:, i, j].copy()
        for a in range(n):
            log_I[a, i, j] = NEG_INF
            for b in range(n):
                log_I[a, i, j] = np.logaddexp(log_I[a, i, j], log_U[a, b] + vals[b])

    # Base case: spans of length 0 (epsilon productions)
    for i in range(L + 1):
        for p in grammar.productions:
            if p.is_empty:
                log_I[p.lhs, i, i] = np.logaddexp(
                    log_I[p.lhs, i, i], np.log(max(p.weight, 1e-300))
                )
        _close_unary(i, i)

    # Fill spans of increasing length
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span
            # Terminal productions for span=1
            if span == 1:
                for p in grammar.productions:
                    if p.is_terminal and p.rhs[0] == int(sequence[i]):
                        log_I[p.lhs, i, j] = np.logaddexp(
                            log_I[p.lhs, i, j], np.log(max(p.weight, 1e-300))
                        )
            # Right-linear productions: A -> a B (terminal at i, nonterminal covers [i+1, j))
            for p in grammar.productions:
                if p.is_right_linear:
                    term, B = p.rhs
                    if int(sequence[i]) == term and log_I[B, i + 1, j] > NEG_INF:
                        log_w = np.log(max(p.weight, 1e-300))
                        val = log_w + log_I[B, i + 1, j]
                        log_I[p.lhs, i, j] = np.logaddexp(log_I[p.lhs, i, j], val)
            # Left-linear productions: A -> B a (nonterminal covers [i, j-1), terminal at j-1)
            for p in grammar.productions:
                if p.is_left_linear:
                    B, term = p.rhs
                    if int(sequence[j - 1]) == term and log_I[B, i, j - 1] > NEG_INF:
                        log_w = np.log(max(p.weight, 1e-300))
                        val = log_w + log_I[B, i, j - 1]
                        log_I[p.lhs, i, j] = np.logaddexp(log_I[p.lhs, i, j], val)
            # LR-linear productions: A -> a B b (terminal at i, nonterminal [i+1, j-1), terminal at j-1)
            if span >= 2:
                for p in grammar.productions:
                    if p.is_lr_linear:
                        term_l, B, term_r = p.rhs
                        if (int(sequence[i]) == term_l and
                                int(sequence[j - 1]) == term_r and
                                log_I[B, i + 1, j - 1] > NEG_INF):
                            log_w = np.log(max(p.weight, 1e-300))
                            val = log_w + log_I[B, i + 1, j - 1]
                            log_I[p.lhs, i, j] = np.logaddexp(log_I[p.lhs, i, j], val)
            # Binary productions (including splits at epsilon spans).
            # We need values from both smaller spans (already closed) and
            # current span (just filled by terminal/linear rules, not yet
            # closed). Apply unary closure to a temporary copy for the current
            # span so binary rules can use unary-derived values (e.g. LOOPLINK
            # from LFRAG), without double-counting.
            vals_before_binary = log_I[:, i, j].copy()
            # Temporarily close for binary lookup
            _close_unary(i, j)
            closed_vals = log_I[:, i, j].copy()
            # Restore pre-closure values (binary results add to these)
            log_I[:, i, j] = vals_before_binary
            for p in grammar.productions:
                if p.is_binary:
                    B, C = p.rhs
                    log_w = np.log(max(p.weight, 1e-300))
                    for k in range(i, j + 1):
                        # For splits at current span boundaries, use closed vals
                        B_val = closed_vals[B] if k == j else log_I[B, i, k]
                        C_val = closed_vals[C] if k == i else log_I[C, k, j]
                        if B_val > NEG_INF and C_val > NEG_INF:
                            val = log_w + B_val + C_val
                            log_I[p.lhs, i, j] = np.logaddexp(log_I[p.lhs, i, j], val)
            _close_unary(i, j)

    return log_I


def inside_logprob(grammar, sequence):
    """Compute log P(sequence | grammar)."""
    log_I = inside(grammar, sequence)
    return log_I[grammar.start, 0, len(sequence)]


# --- Outside algorithm ---

def outside(grammar, sequence, log_inside):
    """Outside algorithm for a WCFG.

    Computes O[A, i, j] = P(seq[0:i], seq[j:L] | A at [i,j))

    Args:
        grammar: WCFG
        sequence: integer array
        log_inside: from inside()

    Returns:
        log_outside: array of shape (n_nonterminals, L+1, L+1)
    """
    L = len(sequence)
    n = grammar.n_nonterminals
    NEG_INF = -1e30

    log_O = np.full((n, L + 1, L + 1), NEG_INF)
    log_O[grammar.start, 0, L] = 0.0  # O(S, 0, L) = 1

    # Top-down: spans of decreasing length (including span=0 for epsilon)
    for span in range(L, -1, -1):
        # Handle unary productions first
        for _ in range(n):
            changed = False
            for i in range(L - span + 1):
                j = i + span
                for p in grammar.productions:
                    if p.is_unary:
                        B = p.rhs[0]
                        log_w = np.log(max(p.weight, 1e-300))
                        val = log_w + log_O[p.lhs, i, j]
                        old = log_O[B, i, j]
                        log_O[B, i, j] = np.logaddexp(old, val)
                        if log_O[B, i, j] > old + 1e-10:
                            changed = True
            if not changed:
                break

        for i in range(L - span + 1):
            j = i + span
            for p in grammar.productions:
                if p.is_right_linear:
                    term, B = p.rhs
                    log_w = np.log(max(p.weight, 1e-300))
                    # A -> a B: O(B, i+1, j) += w * O(A, i, j) if seq[i] == a
                    if i < L and int(sequence[i]) == term:
                        val = log_w + log_O[p.lhs, i, j]
                        log_O[B, i + 1, j] = np.logaddexp(log_O[B, i + 1, j], val)
                elif p.is_left_linear:
                    B, term = p.rhs
                    log_w = np.log(max(p.weight, 1e-300))
                    # A -> B a: O(B, i, j-1) += w * O(A, i, j) if seq[j-1] == a
                    if j > 0 and int(sequence[j - 1]) == term:
                        val = log_w + log_O[p.lhs, i, j]
                        log_O[B, i, j - 1] = np.logaddexp(log_O[B, i, j - 1], val)
                elif p.is_lr_linear:
                    term_l, B, term_r = p.rhs
                    log_w = np.log(max(p.weight, 1e-300))
                    # A -> a B b: O(B, i+1, j-1) += w * O(A, i, j) if seq[i]==a and seq[j-1]==b
                    if (span >= 2 and i < L and j > 0 and
                            int(sequence[i]) == term_l and int(sequence[j - 1]) == term_r):
                        val = log_w + log_O[p.lhs, i, j]
                        log_O[B, i + 1, j - 1] = np.logaddexp(
                            log_O[B, i + 1, j - 1], val)
                elif p.is_binary:
                    B, C = p.rhs
                    log_w = np.log(max(p.weight, 1e-300))

                    # B is left child: O(B, i, k) += w * O(A, i, j) * I(C, k, j)
                    for k in range(i + 1, j):
                        val = log_w + log_O[p.lhs, i, j] + log_inside[C, k, j]
                        log_O[B, i, k] = np.logaddexp(log_O[B, i, k], val)

                    # C is right child: O(C, k, j) += w * O(A, i, j) * I(B, i, k)
                    for k in range(i + 1, j):
                        val = log_w + log_O[p.lhs, i, j] + log_inside[B, i, k]
                        log_O[C, k, j] = np.logaddexp(log_O[C, k, j], val)

    return log_O


def expected_counts(grammar, sequence, log_inside, log_outside):
    """Compute expected production counts from Inside-Outside.

    Returns:
        counts: array of shape (n_productions,) with E[count of each production]
    """
    L = len(sequence)
    log_total = log_inside[grammar.start, 0, L]
    NEG_INF = -1e30
    counts = np.zeros(len(grammar.productions))

    for pi, p in enumerate(grammar.productions):
        if p.is_terminal:
            for i in range(L):
                if p.rhs[0] == int(sequence[i]):
                    log_w = np.log(max(p.weight, 1e-300))
                    val = log_w + log_outside[p.lhs, i, i + 1] - log_total
                    if val > NEG_INF:
                        counts[pi] += np.exp(val)

        elif p.is_right_linear:
            term, B = p.rhs
            log_w = np.log(max(p.weight, 1e-300))
            for span in range(2, L + 1):
                for i in range(L - span + 1):
                    j = i + span
                    if int(sequence[i]) == term:
                        val = (log_w + log_outside[p.lhs, i, j]
                               + log_inside[B, i + 1, j] - log_total)
                        if val > NEG_INF:
                            counts[pi] += np.exp(val)

        elif p.is_left_linear:
            B, term = p.rhs
            log_w = np.log(max(p.weight, 1e-300))
            for span in range(2, L + 1):
                for i in range(L - span + 1):
                    j = i + span
                    if int(sequence[j - 1]) == term:
                        val = (log_w + log_outside[p.lhs, i, j]
                               + log_inside[B, i, j - 1] - log_total)
                        if val > NEG_INF:
                            counts[pi] += np.exp(val)

        elif p.is_lr_linear:
            term_l, B, term_r = p.rhs
            log_w = np.log(max(p.weight, 1e-300))
            for span in range(2, L + 1):
                for i in range(L - span + 1):
                    j = i + span
                    if int(sequence[i]) == term_l and int(sequence[j - 1]) == term_r:
                        val = (log_w + log_outside[p.lhs, i, j]
                               + log_inside[B, i + 1, j - 1] - log_total)
                        if val > NEG_INF:
                            counts[pi] += np.exp(val)

        elif p.is_unary:
            B = p.rhs[0]
            log_w = np.log(max(p.weight, 1e-300))
            for span in range(1, L + 1):
                for i in range(L - span + 1):
                    j = i + span
                    val = log_w + log_outside[p.lhs, i, j] + log_inside[B, i, j] - log_total
                    if val > NEG_INF:
                        counts[pi] += np.exp(val)

        elif p.is_binary:
            B, C = p.rhs
            log_w = np.log(max(p.weight, 1e-300))
            for span in range(2, L + 1):
                for i in range(L - span + 1):
                    j = i + span
                    for k in range(i + 1, j):
                        val = (log_w + log_outside[p.lhs, i, j]
                               + log_inside[B, i, k] + log_inside[C, k, j]
                               - log_total)
                        if val > NEG_INF:
                            counts[pi] += np.exp(val)

        elif p.is_empty:
            log_w = np.log(max(p.weight, 1e-300))
            for i in range(L + 1):
                val = log_w + log_outside[p.lhs, i, i] - log_total
                if val > NEG_INF:
                    counts[pi] += np.exp(val)

    return counts


# --- CYK (Viterbi) algorithm ---

def cyk(grammar, sequence):
    """CYK algorithm: find the most probable parse.

    Returns:
        log_prob: log probability of best parse
        parse: nested tuple representation of the parse tree
    """
    L = len(sequence)
    n = grammar.n_nonterminals
    NEG_INF = -1e30

    log_V = np.full((n, L + 1, L + 1), NEG_INF)
    # Backpointer: (production_index, split_point_or_None)
    bp = [[None] * (L + 1) for _ in range(n)]
    bp = np.empty((n, L + 1, L + 1), dtype=object)

    def _close_unary_cyk(i, j):
        """Apply unary closure for CYK at span [i, j)."""
        for _ in range(n):
            changed = False
            for pi, p in enumerate(grammar.productions):
                if p.is_unary:
                    B = p.rhs[0]
                    log_w = np.log(max(p.weight, 1e-300))
                    val = log_w + log_V[B, i, j]
                    if val > log_V[p.lhs, i, j]:
                        log_V[p.lhs, i, j] = val
                        bp[p.lhs, i, j] = (pi, None)
                        changed = True
            if not changed:
                break

    # Base case: epsilon spans
    for i in range(L + 1):
        for pi, p in enumerate(grammar.productions):
            if p.is_empty:
                log_w = np.log(max(p.weight, 1e-300))
                if log_w > log_V[p.lhs, i, i]:
                    log_V[p.lhs, i, i] = log_w
                    bp[p.lhs, i, i] = (pi, None)
        _close_unary_cyk(i, i)

    # Fill spans
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span
            # Terminal productions for span=1
            if span == 1:
                for pi, p in enumerate(grammar.productions):
                    if p.is_terminal and p.rhs[0] == int(sequence[i]):
                        log_w = np.log(max(p.weight, 1e-300))
                        if log_w > log_V[p.lhs, i, j]:
                            log_V[p.lhs, i, j] = log_w
                            bp[p.lhs, i, j] = (pi, None)
            # Right-linear: A -> a B
            for pi, p in enumerate(grammar.productions):
                if p.is_right_linear:
                    term, B = p.rhs
                    if int(sequence[i]) == term and log_V[B, i + 1, j] > NEG_INF:
                        log_w = np.log(max(p.weight, 1e-300))
                        val = log_w + log_V[B, i + 1, j]
                        if val > log_V[p.lhs, i, j]:
                            log_V[p.lhs, i, j] = val
                            bp[p.lhs, i, j] = (pi, i + 1)
            # Left-linear: A -> B a
            for pi, p in enumerate(grammar.productions):
                if p.is_left_linear:
                    B, term = p.rhs
                    if int(sequence[j - 1]) == term and log_V[B, i, j - 1] > NEG_INF:
                        log_w = np.log(max(p.weight, 1e-300))
                        val = log_w + log_V[B, i, j - 1]
                        if val > log_V[p.lhs, i, j]:
                            log_V[p.lhs, i, j] = val
                            bp[p.lhs, i, j] = (pi, j - 1)
            # LR-linear: A -> a B b
            if span >= 2:
                for pi, p in enumerate(grammar.productions):
                    if p.is_lr_linear:
                        term_l, B, term_r = p.rhs
                        if (int(sequence[i]) == term_l and
                                int(sequence[j - 1]) == term_r and
                                log_V[B, i + 1, j - 1] > NEG_INF):
                            log_w = np.log(max(p.weight, 1e-300))
                            val = log_w + log_V[B, i + 1, j - 1]
                            if val > log_V[p.lhs, i, j]:
                                log_V[p.lhs, i, j] = val
                                bp[p.lhs, i, j] = (pi, (i + 1, j - 1))
            # Close unary before binary, so binary rules can see
            # unary-derived values at current span (e.g. LOOPLINK from LFRAG).
            # CYK uses max (idempotent), so double-closure is safe.
            _close_unary_cyk(i, j)
            # Binary: A -> B C
            for pi, p in enumerate(grammar.productions):
                if p.is_binary:
                    B, C = p.rhs
                    log_w = np.log(max(p.weight, 1e-300))
                    for k in range(i, j + 1):
                        if log_V[B, i, k] > NEG_INF and log_V[C, k, j] > NEG_INF:
                            val = log_w + log_V[B, i, k] + log_V[C, k, j]
                            if val > log_V[p.lhs, i, j]:
                                log_V[p.lhs, i, j] = val
                                bp[p.lhs, i, j] = (pi, k)
            _close_unary_cyk(i, j)

    log_prob = log_V[grammar.start, 0, L]

    # Traceback
    def traceback(nt, i, j):
        if bp[nt, i, j] is None:
            return None
        pi_idx, k = bp[nt, i, j]
        p = grammar.productions[pi_idx]
        if p.is_terminal:
            return (p.lhs, p.rhs[0])
        elif p.is_right_linear:
            child = traceback(p.rhs[1], k, j)
            return (p.lhs, p.rhs[0], child)
        elif p.is_left_linear:
            child = traceback(p.rhs[0], i, k)
            return (p.lhs, child, p.rhs[1])
        elif p.is_lr_linear:
            inner_i, inner_j = k
            child = traceback(p.rhs[1], inner_i, inner_j)
            return (p.lhs, p.rhs[0], child, p.rhs[2])
        elif p.is_unary:
            child = traceback(p.rhs[0], i, j)
            return (p.lhs, child)
        elif p.is_binary:
            left = traceback(p.rhs[0], i, k)
            right = traceback(p.rhs[1], k, j)
            return (p.lhs, left, right)
        return None

    parse = traceback(grammar.start, 0, L)
    return log_prob, parse


# --- Banded Inside algorithm ---

def inside_banded(grammar, sequence, band_center=None, band_width=None):
    """Inside algorithm with banding constraint.

    Only fills cells (i, j) where |center(i,j) - band_center(i,j)| <= band_width.
    If no banding is specified, falls back to regular Inside.

    For alignment envelopes (Holmes 2005), band_center and band_width
    constrain which spans are computed, reducing O(L^3) to O(L * k^2).

    Args:
        grammar: WCFG
        sequence: integer array
        band_center: optional function(i, j) -> float giving expected split
        band_width: optional int, maximum deviation from band_center

    Returns:
        log_inside: array of shape (n_nonterminals, L+1, L+1)
    """
    if band_center is None or band_width is None:
        return inside(grammar, sequence)

    L = len(sequence)
    n = grammar.n_nonterminals
    NEG_INF = -1e30

    log_I = np.full((n, L + 1, L + 1), NEG_INF)

    # Base case
    for i in range(L):
        for p in grammar.productions:
            if p.is_terminal and p.rhs[0] == int(sequence[i]):
                log_I[p.lhs, i, i + 1] = np.logaddexp(
                    log_I[p.lhs, i, i + 1], np.log(max(p.weight, 1e-300))
                )

    # Fill spans within band
    for span in range(2, L + 1):
        for i in range(L - span + 1):
            j = i + span
            # Check if this span is within the band
            center = band_center(i, j)
            if abs((i + j) / 2.0 - center) > band_width:
                continue

            for p in grammar.productions:
                if p.is_binary:
                    B, C = p.rhs
                    log_w = np.log(max(p.weight, 1e-300))
                    # Only consider split points within band
                    k_min = max(i + 1, int(center - band_width))
                    k_max = min(j, int(center + band_width) + 1)
                    for k in range(k_min, k_max):
                        val = log_w + log_I[B, i, k] + log_I[C, k, j]
                        log_I[p.lhs, i, j] = np.logaddexp(log_I[p.lhs, i, j], val)

    return log_I
