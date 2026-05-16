"""Leaf recognizer: grammar representing parses of a nucleotide sequence.

A recognizer is a grammar (WCFG-like structure) that accepts input tokens
matching a specific leaf nucleotide sequence. It represents all possible
(or selected) parse trees that can generate the leaf sequence.

Recognizer states have the same types as the order-1 SCFG:
  - L-inputting: consumes a left nucleotide
  - R-inputting: consumes a right nucleotide
  - LR-inputting: consumes a pair of nucleotides (left + right)
  - Ready: waiting for input (can accept tokens, bifurcate, or epsilon)
  - Start: initial state

For composition with a WPTT, the recognizer's input tokens must match
the WPTT's output tokens.

Banding:
  An alignment guide (from an MSA) restricts which spans [i,j) are
  considered valid. Each leaf position maps to an MSA column; a span
  [i,j) is valid if the MSA columns for positions i..j-1 are
  "structurally compatible" with the guide. A band_width parameter
  controls how far from the guide a span can deviate.

  Without banding: O(L^2) states.
  With banding: O(L * B) states where B is the band width.

Consensus fallback path:
  When building a recognizer, a consensus structure (from MSA) can be
  provided. This creates a guaranteed "fallback path" through the
  recognizer that is protected from pruning/compression. Multiple
  fallback paths can be marked (e.g. from stochastic Inside tracebacks).
  This ensures sibling recognizers always share a compatible structural
  skeleton, preventing empty intersections during progressive
  reconstruction.
"""

import numpy as np
from ..models.rna_grammar import (
    N_NUC, N_TOTAL_TERMINALS, left_terminal, right_terminal,
    pair_terminal, decode_terminal,
)
from ..core.rna import classify_basepair, CTX_NN, N_CONTEXT


class RecognizerState:
    """A state in the leaf recognizer.

    Attributes:
        index: integer state index
        name: human-readable name
        span: (i, j) tuple for the span this state represents
        is_ready: whether this is a ready state (can accept input)
        is_fallback: whether this state is on the consensus fallback path
    """
    __slots__ = ['index', 'name', 'span', 'is_ready', 'is_fallback']

    def __init__(self, index, name, span=None, is_ready=False, is_fallback=False):
        self.index = index
        self.name = name
        self.span = span
        self.is_ready = is_ready
        self.is_fallback = is_fallback


class RecognizerRule:
    """A transition rule in the leaf recognizer.

    Attributes:
        src: source state index
        rule_type: 'match' (consume input), 'ready', 'bifurcation', 'epsilon'
        input_token: input terminal consumed (None for ready/epsilon)
        dst: destination state index (for linear rules)
        dst_left, dst_right: destination states (for bifurcation)
        weight: rule weight/probability
        is_fallback: whether this rule is on the consensus fallback path
    """
    __slots__ = ['src', 'rule_type', 'input_token',
                 'dst', 'dst_left', 'dst_right', 'weight', 'is_fallback']

    def __init__(self, src, rule_type, weight,
                 input_token=None, dst=None,
                 dst_left=None, dst_right=None,
                 is_fallback=False):
        self.src = src
        self.rule_type = rule_type
        self.input_token = input_token
        self.dst = dst
        self.dst_left = dst_left
        self.dst_right = dst_right
        self.weight = weight
        self.is_fallback = is_fallback


class Recognizer:
    """A leaf recognizer grammar.

    Represents possible parses of a specific nucleotide sequence.
    States are indexed 0..n_states-1.

    Attributes:
        n_states: number of states
        states: list of RecognizerState objects
        rules: list of RecognizerRule objects
        rules_from: dict mapping src_state -> list of RecognizerRule
        start: start state index
        span_to_state: dict mapping (i, j) -> state index
    """

    def __init__(self, states, rules, start=0):
        self.states = states
        self.n_states = len(states)
        self.rules = rules
        self.rules_from = {}
        for r in rules:
            self.rules_from.setdefault(r.src, []).append(r)
        self.start = start
        self.span_to_state = {}
        for s in states:
            if s.span is not None:
                self.span_to_state[s.span] = s.index

    def rules_for(self, src):
        return self.rules_from.get(src, [])

    def ready_states(self):
        return {s.index for s in self.states if s.is_ready}

    def fallback_states(self):
        return {s.index for s in self.states if s.is_fallback}

    def fallback_rules(self):
        return [r for r in self.rules if r.is_fallback]


def parse_dot_bracket(structure):
    """Parse dot-bracket notation into a set of base pairs.

    Returns:
        pairs: dict mapping left_pos -> right_pos (and right_pos -> left_pos)
        unpaired: set of unpaired positions
    """
    pairs = {}
    unpaired = set()
    stack = []
    for pos, ch in enumerate(structure):
        if ch == '(':
            stack.append(pos)
        elif ch == ')':
            if stack:
                left = stack.pop()
                pairs[left] = pos
                pairs[pos] = left
        else:
            unpaired.add(pos)
    return pairs, unpaired


def compute_allowed_spans(L, band_width=None, guide_pairs=None,
                          fallback_spans=None):
    """Compute which spans [i,j) are allowed under banding constraints.

    Args:
        L: sequence length
        band_width: maximum span width to consider. If None, all spans allowed.
            For spans involving guide pairs, the pair's span width is used
            regardless of band_width.
        guide_pairs: dict mapping left_pos -> right_pos for guide structure pairs.
            Spans derived from guide pairs are always included.
        fallback_spans: set of (i, j) tuples that must be included regardless
            of banding. These are the consensus fallback spans.

    Returns:
        set of (i, j) tuples representing allowed spans.
    """
    allowed = set()

    if fallback_spans is None:
        fallback_spans = set()

    # Always include empty spans [i, i)
    for i in range(L + 1):
        allowed.add((i, i))

    if band_width is None:
        # No banding: all spans allowed
        for span in range(1, L + 1):
            for i in range(L - span + 1):
                allowed.add((i, i + span))
    else:
        # Banded: only spans with width <= band_width
        for span in range(1, min(band_width + 1, L + 1)):
            for i in range(L - span + 1):
                allowed.add((i, i + span))

        # Always include the full span [0, L)
        allowed.add((0, L))

        # Include spans derived from guide pairs (regardless of width)
        if guide_pairs:
            for left, right in guide_pairs.items():
                if left < right:  # only process each pair once
                    # The pair span [left, right+1)
                    allowed.add((left, right + 1))
                    # The inner span [left+1, right)
                    allowed.add((left + 1, right))

            # Expand around guide pairs: for each guide pair (l, r),
            # include spans that strip one nucleotide from either side
            for left, right in guide_pairs.items():
                if left < right:
                    for di in range(-band_width, band_width + 1):
                        for dj in range(-band_width, band_width + 1):
                            ni = left + di
                            nj = right + 1 + dj
                            if 0 <= ni < nj <= L:
                                allowed.add((ni, nj))

    # Always include fallback spans
    allowed.update(fallback_spans)

    return allowed


def _find_fallback_spans(L, consensus_structures):
    """Find all spans on any consensus fallback path.

    Args:
        L: sequence length
        consensus_structures: list of dot-bracket strings (multiple fallback paths)

    Returns:
        set of (i, j) tuples that are fallback spans
    """
    fallback = set()
    for structure in consensus_structures:
        pairs, _ = parse_dot_bracket(structure)

        # Full span is always fallback
        fallback.add((0, L))

        # Recursively find consensus spans
        def _trace(i, j):
            fallback.add((i, j))
            if i >= j:
                return
            # If i and j-1 form a consensus pair, inner span is also fallback
            if j - 1 in pairs and pairs[j - 1] == i:
                _trace(i + 1, j - 1)
            # Unpaired positions: L-strip and R-strip are fallback
            # (these represent loop/bulge content on the fallback path)
            # We mark intermediate spans created by stripping unpaired
            # positions from either end
            if i not in pairs or (i in pairs and pairs[i] < i):
                # i is unpaired or right-half of a pair: can L-strip
                if (i + 1, j) not in fallback:
                    _trace(i + 1, j)
            if j - 1 not in pairs or (j - 1 in pairs and pairs[j - 1] > j - 1):
                # j-1 is unpaired or left-half of a pair: can R-strip
                if (i, j - 1) not in fallback:
                    _trace(i, j - 1)

        if L > 0:
            _trace(0, L)

        # All empty spans are fallback
        for pos in range(L + 1):
            fallback.add((pos, pos))

    return fallback


def build_leaf_recognizer(leaf_seq, consensus_structure=None,
                          consensus_structures=None,
                          band_width=None, guide_pairs=None):
    """Build a recognizer for a leaf nucleotide sequence.

    Creates a grammar that accepts parses of the given nucleotide sequence,
    restricted to spans allowed by the banding constraints.

    Args:
        leaf_seq: numpy array of nucleotide indices (0-3), shape (L,)
        consensus_structure: optional single dot-bracket string of length L.
            Shorthand for consensus_structures=[consensus_structure].
        consensus_structures: optional list of dot-bracket strings.
            Each one marks a fallback path. Multiple paths supported.
        band_width: optional int. Maximum span width for non-guide spans.
            If None, all spans are included (no banding).
        guide_pairs: optional dict mapping left_pos -> right_pos for
            guide structure pairs. Spans involving guide pairs are always
            included regardless of band_width.

    Returns:
        Recognizer object.
    """
    L = len(leaf_seq)

    # Normalize consensus structures
    all_consensus = []
    if consensus_structure is not None:
        all_consensus.append(consensus_structure)
    if consensus_structures is not None:
        all_consensus.extend(consensus_structures)

    # Compute fallback spans from consensus structures
    fallback_spans = set()
    if all_consensus:
        fallback_spans = _find_fallback_spans(L, all_consensus)

    # If guide_pairs not given but we have consensus structures,
    # derive guide pairs from the first consensus structure
    if guide_pairs is None and all_consensus:
        guide_pairs, _ = parse_dot_bracket(all_consensus[0])

    # Compute allowed spans (with banding)
    allowed = compute_allowed_spans(
        L, band_width=band_width, guide_pairs=guide_pairs,
        fallback_spans=fallback_spans)

    # Create states for allowed spans
    states = []
    rules = []
    state_index = {}  # (i, j) -> state index

    # Sort spans for deterministic ordering: empty spans first, then by width
    sorted_spans = sorted(allowed, key=lambda s: (s[1] - s[0], s[0]))

    for idx, (i, j) in enumerate(sorted_spans):
        name = f'S_{i}_{j}'
        is_fb = (i, j) in fallback_spans
        states.append(RecognizerState(
            idx, name, span=(i, j), is_ready=True, is_fallback=is_fb))
        state_index[(i, j)] = idx

    # Find the start state (span [0, L))
    if (0, L) in state_index:
        start = state_index[(0, L)]
    elif L == 0 and (0, 0) in state_index:
        start = state_index[(0, 0)]
    else:
        # Shouldn't happen if allowed spans include [0, L)
        start = 0

    # Build rules only for allowed spans

    # 1. Epsilon rules: span [i, i) -> epsilon
    for i in range(L + 1):
        if (i, i) not in state_index:
            continue
        si = state_index[(i, i)]
        is_fb = states[si].is_fallback
        rules.append(RecognizerRule(si, 'epsilon', 1.0, is_fallback=is_fb))

    # 2. L-input rules: span [i, j) -> left_terminal(leaf[i]) + span [i+1, j)
    for (i, j) in allowed:
        if j <= i:
            continue
        if (i + 1, j) not in state_index:
            continue
        si = state_index[(i, j)]
        si_next = state_index[(i + 1, j)]
        tok = left_terminal(int(leaf_seq[i]))
        is_fb = states[si].is_fallback and states[si_next].is_fallback
        rules.append(RecognizerRule(
            si, 'match', 1.0,
            input_token=tok, dst=si_next,
            is_fallback=is_fb))

    # 3. R-input rules: span [i, j) -> right_terminal(leaf[j-1]) + span [i, j-1)
    for (i, j) in allowed:
        if j <= i:
            continue
        if (i, j - 1) not in state_index:
            continue
        si = state_index[(i, j)]
        si_next = state_index[(i, j - 1)]
        tok = right_terminal(int(leaf_seq[j - 1]))
        is_fb = states[si].is_fallback and states[si_next].is_fallback
        rules.append(RecognizerRule(
            si, 'match', 1.0,
            input_token=tok, dst=si_next,
            is_fallback=is_fb))

    # 4. LR-input rules: span [i, j) with span >= 2 ->
    #    pair_terminal(leaf[i], leaf[j-1]) + span [i+1, j-1)
    for (i, j) in allowed:
        if j - i < 2:
            continue
        if (i + 1, j - 1) not in state_index:
            continue
        si = state_index[(i, j)]
        si_next = state_index[(i + 1, j - 1)]
        tok = pair_terminal(int(leaf_seq[i]), int(leaf_seq[j - 1]))
        is_fb = states[si].is_fallback and states[si_next].is_fallback
        rules.append(RecognizerRule(
            si, 'match', 1.0,
            input_token=tok, dst=si_next,
            is_fallback=is_fb))

    # 5. Bifurcation rules: span [i, j) splits at k into
    #    span [i, k) and span [k, j)
    for (i, j) in allowed:
        if j <= i:
            continue
        for k in range(i + 1, j):
            if (i, k) not in state_index or (k, j) not in state_index:
                continue
            si = state_index[(i, j)]
            si_l = state_index[(i, k)]
            si_r = state_index[(k, j)]
            is_fb = (states[si].is_fallback and
                     states[si_l].is_fallback and
                     states[si_r].is_fallback)
            rules.append(RecognizerRule(
                si, 'bifurcation', 1.0,
                dst_left=si_l, dst_right=si_r,
                is_fallback=is_fb))

    return Recognizer(states, rules, start=start)


def recognizer_state_count(L):
    """Number of states in an unbanded recognizer for sequence length L."""
    return (L + 1) * (L + 2) // 2


def banded_state_count(L, band_width):
    """Approximate number of states in a banded recognizer.

    This is a rough upper bound: L * band_width + guide pair expansions.
    """
    # Empty spans + banded spans + full span
    n = (L + 1)  # empty spans
    for span in range(1, min(band_width + 1, L + 1)):
        n += L - span + 1
    if L > band_width:
        n += 1  # full span [0, L)
    return n
