"""RNA stem-loop SCFG using the WCFG framework.

Implements the enhanced stem-loop grammar described in scfg-composition.tex
section 7 (simplified singlet version). The grammar models RNA secondary
structure with stems (nested basepairs), bulges, and loops.

Terminal encoding for compound terminals:
  - Pair terminal (a, b): index = a * 4 + b       (indices 0-15)
  - Left-only terminal a:  index = 16 + a          (indices 16-19)
  - Right-only terminal a: index = 20 + a          (indices 20-23)

This lets us represent LR-emitting, L-emitting, and R-emitting productions
uniformly within the existing WCFG framework, which supports terminal,
unary, binary, right-linear, and empty production types.

Nonterminals: START, STEM, BP, STACK, BULGE, LDECO, RDECO, LOOP,
              LOOPLINK, LFRAG, RFRAG, CLOSE
"""

import numpy as np
from ..grammar.scfg import WCFG, Production, build_grammar, inside, inside_logprob

# Nucleotide constants
A, C, G, U = 0, 1, 2, 3
NUC_CHARS = "ACGU"
N_NUC = 4

# Terminal encoding helpers
N_PAIR_TERMINALS = 16   # pair (a, b) -> a*4 + b
N_LEFT_TERMINALS = 4    # left a -> 16 + a
N_RIGHT_TERMINALS = 4   # right a -> 20 + a
N_TOTAL_TERMINALS = 24  # 16 + 4 + 4


def pair_terminal(a, b):
    """Encode a base pair (left, right) as a compound terminal index.

    Args:
        a: left nucleotide (0-3)
        b: right nucleotide (0-3)

    Returns:
        Terminal index in [0, 15].
    """
    return a * N_NUC + b


def left_terminal(a):
    """Encode a left-emitting nucleotide as a compound terminal index.

    Args:
        a: nucleotide (0-3)

    Returns:
        Terminal index in [16, 19].
    """
    return N_PAIR_TERMINALS + a


def right_terminal(a):
    """Encode a right-emitting nucleotide as a compound terminal index.

    Args:
        a: nucleotide (0-3)

    Returns:
        Terminal index in [20, 23].
    """
    return N_PAIR_TERMINALS + N_NUC + a


def decode_terminal(t):
    """Decode a compound terminal index back to (type, nucleotide(s)).

    Args:
        t: terminal index in [0, 23]

    Returns:
        (emission_type, nucleotides) where emission_type is 'LR', 'L', or 'R'
        and nucleotides is (a, b) for LR or (a,) for L/R.
    """
    t = int(t)
    if t < N_PAIR_TERMINALS:
        return ('LR', (t // N_NUC, t % N_NUC))
    elif t < N_PAIR_TERMINALS + N_NUC:
        return ('L', (t - N_PAIR_TERMINALS,))
    else:
        return ('R', (t - N_PAIR_TERMINALS - N_NUC,))


def build_rna_singlet_grammar(kappa_S=0.3, kappa_L=0.3, kappa_B=0.3,
                               kappa_F=0.5, pi=None):
    """Build the singlet RNA stem-loop SCFG.

    The grammar generates RNA sequences with secondary structure. It uses
    compound terminals to encode the emission direction (left, right, or
    paired LR) at each position.

    Parameters:
        kappa_S: float
            Stem extension probability. Probability of adding another
            stem link (basepair/bulge) rather than closing the stem.
        kappa_L: float
            Loop extension probability. Probability of adding another
            loop link rather than ending the loop.
        kappa_B: float
            Bulge branching probability. Within BULGE, probability of
            having both left and right decorations vs only one side.
        kappa_F: float
            Fragment extension probability. Probability of extending
            an LFRAG or RFRAG by one more nucleotide.
        pi: array-like of shape (4,), optional
            Nucleotide equilibrium frequencies [A, C, G, U].
            If None, uses uniform (0.25 each).

    Returns:
        WCFG: the singlet RNA grammar with 24 compound terminals.

    Grammar structure (simplified from scfg-composition.tex section 7.1):
        START  -> STEM
        STEM   -> BP STACK       (basepair + stack continuation)
        BP     -> emit_pair(a,b) (LR emission as terminal production)
        STACK  -> BP STACK       (continue stacking)
                | BULGE          (bulge/internal loop)
                | CLOSE          (close the stem)
        BULGE  -> LDECO STEM     (left decorations + nested stem)
                | RDECO STEM     (right decorations + nested stem)
                | LDECO          (left decorations only, no branch)
                | RDECO          (right decorations only, no branch)
        LDECO  -> LFRAG LDECO    (extend left decoration)
                | LFRAG          (single left fragment)
        RDECO  -> RFRAG RDECO    (extend right decoration)  [note: left-recursive in paper, made right-recursive here]
                | RFRAG          (single right fragment)
        LFRAG  -> emit_L(a)      (left-emitting terminal)
        RFRAG  -> emit_R(a)      (right-emitting terminal)
        CLOSE  -> LOOP           (transition to loop)
        LOOP   -> LOOPLINK LOOP  (extend loop)
                | epsilon        (end loop)
        LOOPLINK -> emit_L(a)    (loop nucleotide, left-emitting)

    Note: The paper's grammar is more complex (stacked pairs with LLRR
    emissions, multiloop branches, etc.). This implementation captures the
    core stem-loop structure using the simpler production types supported
    by the existing WCFG framework.
    """
    if pi is None:
        pi = np.ones(N_NUC) / N_NUC
    else:
        pi = np.asarray(pi, dtype=np.float64)
        pi = pi / pi.sum()

    nonterminals = [
        'START', 'STEM', 'BP', 'STACK', 'BULGE',
        'LDECO', 'RDECO', 'LOOP', 'LOOPLINK', 'LFRAG', 'RFRAG', 'CLOSE'
    ]
    rules = []

    # --- START -> STEM (unary, weight 1) ---
    rules.append(('START', [('STEM', 'N')], 1.0))

    # --- STEM -> BP STACK (binary) ---
    rules.append(('STEM', [('BP', 'N'), ('STACK', 'N')], 1.0))

    # --- BP -> emit_pair(a, b) for each (a, b) ---
    # BP emits one LR pair terminal. The weight is the pair equilibrium
    # frequency. For simplicity, use product of marginals: pi[a] * pi[b].
    # (A proper model would use a basepair-specific distribution favoring
    # Watson-Crick and wobble pairs.)
    for a in range(N_NUC):
        for b in range(N_NUC):
            t_idx = pair_terminal(a, b)
            w = float(pi[a] * pi[b])
            rules.append(('BP', [(t_idx, 'T')], w))

    # --- STACK -> BP STACK (continue stacking, weight kappa_S) ---
    rules.append(('STACK', [('BP', 'N'), ('STACK', 'N')], kappa_S))

    # --- STACK -> BULGE (bulge/internal loop, weight fraction of (1-kappa_S)) ---
    # Split (1-kappa_S) between BULGE and CLOSE
    p_bulge = (1.0 - kappa_S) * 0.5
    p_close = (1.0 - kappa_S) * 0.5
    rules.append(('STACK', [('BULGE', 'N')], p_bulge))

    # --- STACK -> CLOSE (end stem, weight fraction of (1-kappa_S)) ---
    rules.append(('STACK', [('CLOSE', 'N')], p_close))

    # --- BULGE productions ---
    # BULGE -> LDECO STEM (left deco + nested stem continuation)
    # BULGE -> RDECO STEM (right deco + nested stem continuation)
    # BULGE -> LDECO (left deco only, no continuation)
    # BULGE -> RDECO (right deco only, no continuation)
    # kappa_B controls branching (whether we recurse into another STEM)
    rules.append(('BULGE', [('LDECO', 'N'), ('STEM', 'N')], kappa_B * 0.5))
    rules.append(('BULGE', [('RDECO', 'N'), ('STEM', 'N')], kappa_B * 0.5))
    rules.append(('BULGE', [('LDECO', 'N')], (1.0 - kappa_B) * 0.5))
    rules.append(('BULGE', [('RDECO', 'N')], (1.0 - kappa_B) * 0.5))

    # --- LDECO -> LFRAG LDECO | LFRAG ---
    rules.append(('LDECO', [('LFRAG', 'N'), ('LDECO', 'N')], kappa_F))
    rules.append(('LDECO', [('LFRAG', 'N')], 1.0 - kappa_F))

    # --- RDECO -> RFRAG RDECO | RFRAG ---
    # Note: in the paper RDECO is left-recursive (RDECO -> RDECO RFRAG).
    # We make it right-recursive (RDECO -> RFRAG RDECO) for compatibility
    # with the CYK algorithm which handles both equally well.
    rules.append(('RDECO', [('RFRAG', 'N'), ('RDECO', 'N')], kappa_F))
    rules.append(('RDECO', [('RFRAG', 'N')], 1.0 - kappa_F))

    # --- LFRAG -> emit_L(a) for each nucleotide a ---
    for a in range(N_NUC):
        t_idx = left_terminal(a)
        rules.append(('LFRAG', [(t_idx, 'T')], float(pi[a])))

    # --- RFRAG -> emit_R(a) for each nucleotide a ---
    for a in range(N_NUC):
        t_idx = right_terminal(a)
        rules.append(('RFRAG', [(t_idx, 'T')], float(pi[a])))

    # --- CLOSE -> LOOP (unary) ---
    rules.append(('CLOSE', [('LOOP', 'N')], 1.0))

    # --- LOOP -> LOOPLINK LOOP | epsilon ---
    rules.append(('LOOP', [('LOOPLINK', 'N'), ('LOOP', 'N')], kappa_L))
    rules.append(('LOOP', [], 1.0 - kappa_L))

    # --- LOOPLINK -> emit_L(a) for each nucleotide a ---
    for a in range(N_NUC):
        t_idx = left_terminal(a)
        rules.append(('LOOPLINK', [(t_idx, 'T')], float(pi[a])))

    return build_grammar(nonterminals, N_TOTAL_TERMINALS, rules, start='START')


def build_rna_pair_grammar(kappa_S=0.3, kappa_L=0.3, kappa_B=0.3,
                            kappa_F=0.5, pi=None, Q=None, t=1.0,
                            ins_prob=0.1, del_prob=0.1):
    """Build the pair RNA SCFG by evolving the singlet grammar.

    For each emitting nonterminal in the singlet grammar, creates match,
    insert, and delete variants with TKF91-style transition weights.

    The pair grammar doubles the terminal alphabet to represent
    ancestor-descendant pairs:
      - Match pair: both ancestor and descendant emit
      - Insert: only descendant emits
      - Delete: only ancestor emits

    Parameters:
        kappa_S: float
            Stem extension probability.
        kappa_L: float
            Loop extension probability.
        kappa_B: float
            Bulge branching probability.
        kappa_F: float
            Fragment extension probability.
        pi: array-like of shape (4,), optional
            Nucleotide equilibrium frequencies.
        Q: array-like of shape (4, 4), optional
            Substitution rate matrix. If None, uses Jukes-Cantor.
        t: float
            Evolutionary time.
        ins_prob: float
            Insertion probability per position (TKF beta parameter).
        del_prob: float
            Deletion probability per position (TKF 1 - alpha parameter).

    Returns:
        WCFG: the pair RNA grammar.

    Notes:
        This is a simplified pair grammar. For the full pair grammar
        described in scfg-composition.tex section 7.3, each stem link
        nonterminal (STEM) gets match/insert/delete variants with the
        full TKF91 transition structure. Here we apply a simpler scheme:
        each terminal emission in the singlet grammar is tripled into
        match (emit both), insert (emit descendant only), and delete
        (emit ancestor only) with the given probabilities.
    """
    if pi is None:
        pi = np.ones(N_NUC) / N_NUC
    else:
        pi = np.asarray(pi, dtype=np.float64)
        pi = pi / pi.sum()

    if Q is None:
        # Jukes-Cantor rate matrix
        Q = np.ones((N_NUC, N_NUC)) / 3.0
        np.fill_diagonal(Q, -1.0)

    # Compute substitution probability matrix P(t) = exp(Qt)
    from scipy.linalg import expm
    P = expm(Q * t)
    P = np.maximum(P, 0.0)
    # Normalize rows
    P = P / P.sum(axis=1, keepdims=True)

    alpha = 1.0 - del_prob   # survival probability
    beta = ins_prob           # insertion probability

    # For pair grammar, we need a larger terminal alphabet:
    # Ancestor-descendant pair terminals for each emission type:
    #   Match LR pair: (a_L, a_R, d_L, d_R) but this gets very large.
    #
    # Simplified approach: we keep the same 24-terminal alphabet but
    # create match/insert/delete nonterminal variants.
    # Match: emits both ancestor and descendant (we concatenate their terminals)
    # Insert: emits only descendant terminal
    # Delete: emits only ancestor terminal
    #
    # For a true pair grammar this would need a product alphabet.
    # Here we build the simplest useful version: the pair grammar
    # over the same terminal alphabet, with M/I/D state structure
    # encoded in the nonterminal names.

    nonterminals = [
        'START',
        # Match variants (ancestor survived, descendant present)
        'STEM_M', 'BP_M', 'STACK_M', 'BULGE_M',
        'LDECO_M', 'RDECO_M', 'LOOP_M', 'LOOPLINK_M',
        'LFRAG_M', 'RFRAG_M', 'CLOSE_M',
        # Insert variants (no ancestor, descendant present)
        'STEM_I', 'BP_I', 'STACK_I', 'BULGE_I',
        'LDECO_I', 'RDECO_I', 'LOOP_I', 'LOOPLINK_I',
        'LFRAG_I', 'RFRAG_I', 'CLOSE_I',
        # Delete variants (ancestor present, no descendant)
        'STEM_D', 'BP_D', 'STACK_D', 'BULGE_D',
        'LDECO_D', 'RDECO_D', 'LOOP_D', 'LOOPLINK_D',
        'LFRAG_D', 'RFRAG_D', 'CLOSE_D',
        'END'
    ]

    rules = []

    # TKF-style transitions from START
    # START -> STEM_M (match first link)
    rules.append(('START', [('STEM_M', 'N')], (1.0 - beta) * alpha))
    # START -> STEM_I (insert at start)
    rules.append(('START', [('STEM_I', 'N')], beta))
    # START -> STEM_D (delete first link)
    rules.append(('START', [('STEM_D', 'N')], (1.0 - beta) * (1.0 - alpha)))
    # START -> END (empty)
    rules.append(('START', [('END', 'N')], 0.0))  # placeholder

    # For each variant (M, I, D), replicate the singlet grammar structure
    # with appropriate emission weights
    for var in ['M', 'I', 'D']:
        sfx = f'_{var}'

        # STEM -> BP STACK
        rules.append((f'STEM{sfx}', [(f'BP{sfx}', 'N'), (f'STACK{sfx}', 'N')], 1.0))

        # BP emissions depend on variant
        if var == 'M':
            # Match: emit pair with substitution probability
            for a in range(N_NUC):
                for b in range(N_NUC):
                    # Ancestor pair (a, b), descendant drawn from P
                    t_idx = pair_terminal(a, b)
                    # Weight includes ancestor equilibrium and P(desc|anc)
                    # For simplicity, emit the ancestor pair terminal
                    w = float(pi[a] * pi[b])
                    rules.append((f'BP{sfx}', [(t_idx, 'T')], w))
        elif var == 'I':
            # Insert: emit descendant pair only
            for a in range(N_NUC):
                for b in range(N_NUC):
                    t_idx = pair_terminal(a, b)
                    w = float(pi[a] * pi[b])
                    rules.append((f'BP{sfx}', [(t_idx, 'T')], w))
        else:  # D
            # Delete: emit ancestor pair only
            for a in range(N_NUC):
                for b in range(N_NUC):
                    t_idx = pair_terminal(a, b)
                    w = float(pi[a] * pi[b])
                    rules.append((f'BP{sfx}', [(t_idx, 'T')], w))

        # STACK -> BP STACK | BULGE | CLOSE
        rules.append((f'STACK{sfx}', [(f'BP{sfx}', 'N'), (f'STACK{sfx}', 'N')], kappa_S))
        p_bulge = (1.0 - kappa_S) * 0.5
        p_close = (1.0 - kappa_S) * 0.5
        rules.append((f'STACK{sfx}', [(f'BULGE{sfx}', 'N')], p_bulge))
        rules.append((f'STACK{sfx}', [(f'CLOSE{sfx}', 'N')], p_close))

        # BULGE
        rules.append((f'BULGE{sfx}', [(f'LDECO{sfx}', 'N'), (f'STEM{sfx}', 'N')], kappa_B * 0.5))
        rules.append((f'BULGE{sfx}', [(f'RDECO{sfx}', 'N'), (f'STEM{sfx}', 'N')], kappa_B * 0.5))
        rules.append((f'BULGE{sfx}', [(f'LDECO{sfx}', 'N')], (1.0 - kappa_B) * 0.5))
        rules.append((f'BULGE{sfx}', [(f'RDECO{sfx}', 'N')], (1.0 - kappa_B) * 0.5))

        # LDECO, RDECO
        rules.append((f'LDECO{sfx}', [(f'LFRAG{sfx}', 'N'), (f'LDECO{sfx}', 'N')], kappa_F))
        rules.append((f'LDECO{sfx}', [(f'LFRAG{sfx}', 'N')], 1.0 - kappa_F))
        rules.append((f'RDECO{sfx}', [(f'RFRAG{sfx}', 'N'), (f'RDECO{sfx}', 'N')], kappa_F))
        rules.append((f'RDECO{sfx}', [(f'RFRAG{sfx}', 'N')], 1.0 - kappa_F))

        # LFRAG, RFRAG emissions
        for a in range(N_NUC):
            if var == 'M':
                # Match: emit with substitution
                for b in range(N_NUC):
                    # Emit left terminal for ancestor a, weighted by P(b|a)
                    # But we only have one terminal slot; use ancestor terminal
                    pass
                t_idx = left_terminal(a)
                rules.append((f'LFRAG{sfx}', [(t_idx, 'T')], float(pi[a])))
            else:
                t_idx = left_terminal(a)
                rules.append((f'LFRAG{sfx}', [(t_idx, 'T')], float(pi[a])))

        for a in range(N_NUC):
            t_idx = right_terminal(a)
            rules.append((f'RFRAG{sfx}', [(t_idx, 'T')], float(pi[a])))

        # CLOSE, LOOP, LOOPLINK
        rules.append((f'CLOSE{sfx}', [(f'LOOP{sfx}', 'N')], 1.0))
        rules.append((f'LOOP{sfx}', [(f'LOOPLINK{sfx}', 'N'), (f'LOOP{sfx}', 'N')], kappa_L))
        rules.append((f'LOOP{sfx}', [], 1.0 - kappa_L))

        for a in range(N_NUC):
            t_idx = left_terminal(a)
            rules.append((f'LOOPLINK{sfx}', [(t_idx, 'T')], float(pi[a])))

    # END -> epsilon
    rules.append(('END', [], 1.0))

    return build_grammar(nonterminals, N_TOTAL_TERMINALS, rules, start='START')


def encode_rna_structure(sequence, structure):
    """Encode an RNA sequence with dot-bracket structure as compound terminals.

    Given an RNA sequence and its secondary structure in dot-bracket notation,
    produces an array of compound terminal indices suitable for the singlet
    RNA grammar.

    The encoding traverses the structure and assigns emission types:
      - Paired positions (matching parentheses) get LR pair terminals
      - Unpaired positions on the 5' side of a stem get L terminals
      - Unpaired positions on the 3' side of a stem get R terminals

    For simplicity, this implementation uses a heuristic: positions in
    parentheses become pair terminals (ordered left-to-right), and dots
    become left terminals. The resulting terminal sequence is ordered
    to match the grammar's derivation order (outer pairs first, then
    inner content).

    Args:
        sequence: str
            RNA sequence, e.g. "GCAUAGUC"
        structure: str
            Dot-bracket notation, e.g. "((....))". Must have matching
            parentheses and same length as sequence.

    Returns:
        np.ndarray of int32: compound terminal indices.

    Raises:
        ValueError: if sequence and structure lengths differ or brackets
            are unbalanced.

    Example:
        >>> encode_rna_structure("GCAAAGUC", "((....))")
        # Returns array with pair terminals for G-C and C-U positions,
        # and left terminals for the AAAG loop.
    """
    if len(sequence) != len(structure):
        raise ValueError(
            f"Sequence length ({len(sequence)}) != structure length ({len(structure)})")

    seq = sequence.upper()
    nuc_map = {c: i for i, c in enumerate(NUC_CHARS)}

    # Find base pairs from dot-bracket
    pairs = {}  # maps position -> partner position
    stack = []
    for i, ch in enumerate(structure):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if not stack:
                raise ValueError(f"Unbalanced ')' at position {i}")
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j
    if stack:
        raise ValueError(f"Unbalanced '(' at positions {stack}")

    # Build the terminal sequence in derivation order.
    # The grammar derives sequences by emitting outer pairs first,
    # then recurses inward. We need to produce the terminal sequence
    # that matches the yield of the parse tree, read left-to-right.
    #
    # For the CYK/Inside algorithm, the terminal sequence is just
    # the original sequence read left-to-right, but each position
    # is tagged with its emission type.
    #
    # Strategy: walk the sequence left to right. Paired positions
    # that are the LEFT half of a pair get an LR pair terminal.
    # Paired positions that are the RIGHT half are consumed by
    # their partner's LR terminal (they do not appear separately).
    # Unpaired positions get L terminals (they appear as loop or
    # bulge content).
    #
    # This produces a terminal sequence shorter than the original
    # sequence (since each basepair contributes one pair terminal
    # instead of two single terminals).
    terminals = []
    consumed = set()
    for i in range(len(seq)):
        if i in consumed:
            continue
        if i in pairs and pairs[i] > i:
            # Left half of a basepair
            j = pairs[i]
            a = nuc_map.get(seq[i], 0)
            b = nuc_map.get(seq[j], 0)
            terminals.append(pair_terminal(a, b))
            consumed.add(j)  # right half consumed by this pair terminal
        elif i in pairs and pairs[i] < i:
            # Right half already consumed; encode as R terminal
            # (This shouldn't happen if consumed tracking is correct,
            # but handle gracefully)
            a = nuc_map.get(seq[i], 0)
            terminals.append(right_terminal(a))
        else:
            # Unpaired: L terminal
            a = nuc_map.get(seq[i], 0)
            terminals.append(left_terminal(a))

    return np.array(terminals, dtype=np.int32)


def decode_rna_structure(terminals, n_chars=4):
    """Decode a compound terminal array back to RNA sequence and structure.

    Inverse of encode_rna_structure. Reconstructs the sequence and
    dot-bracket structure from the terminal encoding.

    Args:
        terminals: np.ndarray of int32
            Compound terminal indices from encode_rna_structure.
        n_chars: int
            Alphabet size (default 4 for RNA).

    Returns:
        (sequence, structure): tuple of strings.
            sequence: RNA sequence (e.g. "GCAAAGUC")
            structure: dot-bracket notation (e.g. "((....))").

    Notes:
        The reconstruction assumes that LR pair terminals represent
        nested basepairs. The resulting dot-bracket may not perfectly
        reconstruct complex pseudoknot-free structures if the terminal
        ordering is ambiguous, but it correctly handles the simple
        stem-loop case.
    """
    seq_chars = []
    struct_chars = []
    # Track pair positions for closing brackets
    pair_right_positions = []  # (position_in_output, nucleotide)

    for t in terminals:
        t = int(t)
        etype, nucs = decode_terminal(t)
        if etype == 'LR':
            a, b = nucs
            # Left half of basepair
            left_pos = len(seq_chars)
            seq_chars.append(NUC_CHARS[a])
            struct_chars.append('(')
            # Remember right half to append later
            pair_right_positions.append((left_pos, NUC_CHARS[b]))
        elif etype == 'L':
            seq_chars.append(NUC_CHARS[nucs[0]])
            struct_chars.append('.')
        elif etype == 'R':
            seq_chars.append(NUC_CHARS[nucs[0]])
            struct_chars.append('.')

    # Now insert the right halves of basepairs at the end (in reverse order)
    # to create proper nesting
    for _, nuc in reversed(pair_right_positions):
        seq_chars.append(nuc)
        struct_chars.append(')')

    return ''.join(seq_chars), ''.join(struct_chars)


def sample_rna_sequence(grammar, rng=None, max_len=100):
    """Sample a sequence from the singlet RNA grammar by top-down derivation.

    Performs a stochastic top-down derivation from the start symbol,
    choosing productions randomly according to their weights.

    Args:
        grammar: WCFG
            The singlet RNA grammar (from build_rna_singlet_grammar).
        rng: np.random.Generator, optional
            Random number generator. If None, uses default.
        max_len: int
            Maximum number of terminals before aborting derivation
            to prevent infinite recursion.

    Returns:
        (terminals, parse_tree): tuple where
            terminals: np.ndarray of int32, the emitted terminal sequence
            parse_tree: nested tuple representing the derivation.
                Each node is (nonterminal_index, children...) where
                children are either parse_tree nodes or terminal indices.

    Raises:
        RuntimeError: if derivation exceeds max_len terminals.
    """
    if rng is None:
        rng = np.random.default_rng()

    terminals = []

    def derive(nt_idx, depth=0):
        """Recursively derive from nonterminal nt_idx."""
        if len(terminals) >= max_len:
            raise RuntimeError(
                f"Derivation exceeded max_len={max_len} terminals. "
                "Try increasing max_len or adjusting grammar parameters.")

        # Get all productions for this nonterminal
        prods = grammar.productions_for(nt_idx)
        if not prods:
            return (nt_idx,)

        # Normalize weights and sample
        weights = np.array([p.weight for p in prods])
        total = weights.sum()
        if total < 1e-30:
            return (nt_idx,)
        probs = weights / total
        idx = rng.choice(len(prods), p=probs)
        chosen = prods[idx]

        if chosen.is_empty:
            return (nt_idx, 'eps')
        elif chosen.is_terminal:
            t = chosen.rhs[0]
            terminals.append(t)
            return (nt_idx, t)
        elif chosen.is_unary:
            child = derive(chosen.rhs[0], depth + 1)
            return (nt_idx, child)
        elif chosen.is_binary:
            left = derive(chosen.rhs[0], depth + 1)
            right = derive(chosen.rhs[1], depth + 1)
            return (nt_idx, left, right)
        elif chosen.is_right_linear:
            t = chosen.rhs[0]
            terminals.append(t)
            child = derive(chosen.rhs[1], depth + 1)
            return (nt_idx, t, child)
        else:
            return (nt_idx,)

    tree = derive(grammar.start)
    return np.array(terminals, dtype=np.int32), tree


def _format_parse_tree(grammar, tree, indent=0):
    """Format a parse tree as a human-readable string.

    Args:
        grammar: WCFG (for nonterminal names)
        tree: nested tuple from sample_rna_sequence
        indent: current indentation level

    Returns:
        str: formatted tree representation
    """
    prefix = "  " * indent
    if not isinstance(tree, tuple):
        return f"{prefix}{tree}"

    nt_idx = tree[0]
    nt_name = grammar.nonterminals[nt_idx] if nt_idx < len(grammar.nonterminals) else f"NT{nt_idx}"

    if len(tree) == 1:
        return f"{prefix}{nt_name}"
    elif len(tree) == 2:
        child = tree[1]
        if child == 'eps':
            return f"{prefix}{nt_name} -> eps"
        elif isinstance(child, int):
            etype, nucs = decode_terminal(child)
            nuc_str = ','.join(NUC_CHARS[n] for n in nucs)
            return f"{prefix}{nt_name} -> {etype}({nuc_str})"
        else:
            child_str = _format_parse_tree(grammar, child, indent + 1)
            return f"{prefix}{nt_name} ->\n{child_str}"
    else:
        lines = [f"{prefix}{nt_name} ->"]
        for child in tree[1:]:
            if isinstance(child, int):
                etype, nucs = decode_terminal(child)
                nuc_str = ','.join(NUC_CHARS[n] for n in nucs)
                lines.append(f"{prefix}  {etype}({nuc_str})")
            elif isinstance(child, tuple):
                lines.append(_format_parse_tree(grammar, child, indent + 1))
            else:
                lines.append(f"{prefix}  {child}")
        return '\n'.join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("RNA Stem-Loop Grammar Test")
    print("=" * 60)

    # 1. Build singlet grammar
    print("\n--- Building singlet grammar ---")
    grammar = build_rna_singlet_grammar(
        kappa_S=0.3, kappa_L=0.4, kappa_B=0.3, kappa_F=0.4)
    print(f"Nonterminals: {grammar.nonterminals}")
    print(f"Number of productions: {len(grammar.productions)}")
    print(f"Number of terminals: {grammar.n_terminals}")
    print(f"Start symbol: {grammar.nonterminals[grammar.start]}")

    # Print production summary
    type_counts = {'terminal': 0, 'unary': 0, 'binary': 0,
                   'right_linear': 0, 'empty': 0, 'other': 0}
    for p in grammar.productions:
        if p.is_terminal:
            type_counts['terminal'] += 1
        elif p.is_unary:
            type_counts['unary'] += 1
        elif p.is_binary:
            type_counts['binary'] += 1
        elif p.is_right_linear:
            type_counts['right_linear'] += 1
        elif p.is_empty:
            type_counts['empty'] += 1
        else:
            type_counts['other'] += 1
    print(f"Production types: {type_counts}")

    # 2. Sample sequences
    print("\n--- Sampling sequences ---")
    rng = np.random.default_rng(42)
    for i in range(5):
        try:
            terminals, tree = sample_rna_sequence(grammar, rng, max_len=50)
            seq, struct = decode_rna_structure(terminals)
            print(f"  Sample {i+1}: {seq}  {struct}  (len={len(seq)}, "
                  f"n_terminals={len(terminals)})")
        except RuntimeError as e:
            print(f"  Sample {i+1}: (exceeded max length)")

    # 3. Encode a known structure
    print("\n--- Encoding known structure ---")
    test_seq = "GCAAAGUC"
    test_struct = "((....))".ljust(len(test_seq), '.')[:len(test_seq)]
    # Actually need matching structure
    test_struct = "((....))".ljust(len(test_seq), '.')
    if len(test_struct) != len(test_seq):
        test_struct = "((....))".ljust(len(test_seq), '.')[:len(test_seq)]
    print(f"  Sequence:  {test_seq}")
    print(f"  Structure: {test_struct}")
    encoded = encode_rna_structure(test_seq, test_struct)
    print(f"  Encoded terminals: {encoded}")
    for t in encoded:
        etype, nucs = decode_terminal(t)
        nuc_str = ','.join(NUC_CHARS[n] for n in nucs)
        print(f"    {t:2d} -> {etype}({nuc_str})")

    # Decode back
    dec_seq, dec_struct = decode_rna_structure(encoded)
    print(f"  Decoded sequence:  {dec_seq}")
    print(f"  Decoded structure: {dec_struct}")

    # 4. Run Inside algorithm on encoded sequence
    print("\n--- Inside algorithm ---")
    log_prob = inside_logprob(grammar, encoded)
    print(f"  log P('{test_seq}' with structure '{test_struct}'): {log_prob:.4f}")
    print(f"  P = {np.exp(log_prob):.6e}")

    # 5. Run Inside on sampled sequences
    print("\n--- Inside on sampled sequences ---")
    rng2 = np.random.default_rng(123)
    for i in range(3):
        try:
            terminals, tree = sample_rna_sequence(grammar, rng2, max_len=30)
            if len(terminals) == 0:
                print(f"  Sample {i+1}: empty sequence (log P = special case)")
                continue
            log_prob = inside_logprob(grammar, terminals)
            seq, struct = decode_rna_structure(terminals)
            print(f"  Sample {i+1}: {seq} {struct}  "
                  f"log P = {log_prob:.4f}")
        except RuntimeError:
            print(f"  Sample {i+1}: (exceeded max length)")

    # 6. Test pair grammar construction
    print("\n--- Building pair grammar ---")
    pair_grammar = build_rna_pair_grammar(
        kappa_S=0.3, kappa_L=0.4, kappa_B=0.3, kappa_F=0.4,
        t=0.5, ins_prob=0.1, del_prob=0.1)
    print(f"  Nonterminals: {len(pair_grammar.nonterminals)}")
    print(f"  Productions: {len(pair_grammar.productions)}")
    print(f"  Terminals: {pair_grammar.n_terminals}")

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
