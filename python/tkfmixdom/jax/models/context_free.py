"""TKF Structure Tree (TKFST/TKFStack) singlet and pair SCFGs.

Implements the full TKFStack grammar from paper Section 7.3, including:
- Stems with single basepairs (LR emission) and stacked pairs (LLRR via
  factored lr_linear)
- Loops with left/right fragments and multiloop branches
- Bulges with left/right decorations
- Closing basepairs

Terminal encoding (8 terminals for singlet, extended for pair):
  Left-emitting:  L(a) = a       (indices 0-3)
  Right-emitting: R(a) = 4 + a   (indices 4-7)

The grammar uses lr_linear productions (A -> a B b) for basepair emissions,
right_linear (A -> a B) for left-emitting fragments, and left_linear (A -> B a)
for right-emitting fragments.

Stacked pairs (LLRR) are factored into two nested lr_linear productions via
context nonterminals STACKED_{pair}, one per canonical basepair class.
"""

import numpy as np
from ..grammar.scfg import WCFG, Production, build_grammar, inside, inside_logprob

# Nucleotide constants
A, C, G, U = 0, 1, 2, 3
NUC_CHARS = "ACGU"
N_NUC = 4

# Canonical basepairs (Watson-Crick + wobble)
CANONICAL_PAIRS = [(A, U), (C, G), (G, C), (U, A), (G, U), (U, G)]
N_CANONICAL = len(CANONICAL_PAIRS)
_CANONICAL_SET = set(CANONICAL_PAIRS)


def _canonical_index(a, b):
    """Return index of canonical pair (a,b) in CANONICAL_PAIRS, or -1."""
    for i, (ca, cb) in enumerate(CANONICAL_PAIRS):
        if ca == a and cb == b:
            return i
    return -1


def is_canonical(a, b):
    """True if (a,b) is a canonical basepair."""
    return (a, b) in _CANONICAL_SET


# --- Terminal encoding ---

N_SINGLET_TERMINALS = 8  # 4 left + 4 right


def left_terminal(a):
    """Left-emitting terminal for nucleotide a."""
    return a


def right_terminal(a):
    """Right-emitting terminal for nucleotide a."""
    return N_NUC + a


def decode_terminal(t):
    """Decode terminal index to (direction, nucleotide).

    Returns:
        ('L', a) or ('R', a)
    """
    t = int(t)
    if t < N_NUC:
        return ('L', t)
    else:
        return ('R', t - N_NUC)


# --- Default equilibrium distributions ---

def _default_bp_equilibrium(pi):
    """Basepair equilibrium: canonical pairs weighted by marginals, normalized."""
    bp_eq = np.zeros((N_NUC, N_NUC))
    for a, b in CANONICAL_PAIRS:
        bp_eq[a, b] = pi[a] * pi[b]
    total = bp_eq.sum()
    if total > 0:
        bp_eq /= total
    return bp_eq


def _default_stack_equilibrium(pi_bp):
    """Stacked-pair equilibrium: independent canonical pairs."""
    stack_eq = np.zeros((N_CANONICAL, N_CANONICAL))
    for i, (a1, b1) in enumerate(CANONICAL_PAIRS):
        for j, (a2, b2) in enumerate(CANONICAL_PAIRS):
            stack_eq[i, j] = pi_bp[a1, b1] * pi_bp[a2, b2]
    total = stack_eq.sum()
    if total > 0:
        stack_eq /= total
    return stack_eq


# --- Singlet grammar construction ---

def _pair_name(a, b):
    """Short name for a basepair, e.g. 'AU'."""
    return f"{NUC_CHARS[a]}{NUC_CHARS[b]}"


def build_tkfst_singlet_grammar(
    kappa_S=0.4, kappa_L=0.3,
    p_bp=0.6, p_st=0.3, p_bu=0.1,
    ext_K=0.5,
    p_lf=0.4, p_rf=0.4, p_sl=0.2,
    ext_B=0.3,
    pi=None, pi_bp=None, pi_stack=None,
):
    """Build the TKFST singlet SCFG (Section 7.3).

    Parameters:
        kappa_S: stem link extension probability (TKF91 κ for stems)
        kappa_L: loop link extension probability (TKF91 κ for loops)
        p_bp: probability of single basepair stem link
        p_st: probability of stacked pair stem link
        p_bu: probability of bulge stem link
        ext_K: stacked pair extension probability (geometric)
        p_lf: probability of left fragment in loop link
        p_rf: probability of right fragment in loop link
        p_sl: probability of nested stem-loop in loop link
        ext_B: fragment extension probability in bulges
        pi: (4,) nucleotide equilibrium
        pi_bp: (4,4) basepair equilibrium (rows=left, cols=right)
        pi_stack: (6,6) stacked pair equilibrium (outer × inner canonical pairs)

    Returns:
        WCFG with 8 terminals (4 left + 4 right)
    """
    assert abs(p_bp + p_st + p_bu - 1.0) < 1e-10, "p_bp + p_st + p_bu must = 1"
    assert abs(p_lf + p_rf + p_sl - 1.0) < 1e-10, "p_lf + p_rf + p_sl must = 1"

    if pi is None:
        pi = np.ones(N_NUC) / N_NUC
    else:
        pi = np.asarray(pi, dtype=np.float64)
    if pi_bp is None:
        pi_bp = _default_bp_equilibrium(pi)
    else:
        pi_bp = np.asarray(pi_bp, dtype=np.float64)
    if pi_stack is None:
        pi_stack = _default_stack_equilibrium(pi_bp)
    else:
        pi_stack = np.asarray(pi_stack, dtype=np.float64)

    # Marginal distribution over outer canonical pair (for factored stacking)
    pi_outer = pi_stack.sum(axis=1)  # (N_CANONICAL,) sum over inner
    # Conditional: pi_inner_given_outer[i, j] = pi_stack[i, j] / pi_outer[i]
    pi_inner_given_outer = np.zeros_like(pi_stack)
    for i in range(N_CANONICAL):
        if pi_outer[i] > 0:
            pi_inner_given_outer[i] = pi_stack[i] / pi_outer[i]

    # Build nonterminal list
    # Note: CLOSE wraps a LOOP inside the closing basepair (lr_linear),
    # so there is no separate CLOSE_INNER. STEM -> CLOSE (not CLOSE LOOP).
    nonterminals = [
        'SL',               # 0: stem-loop entry
        'STEM',             # 1: stem link sequence
        'STEM_AFTER_BULGE', # 2: continuation after bulge's left decoration
        'CLOSE',            # 3: closing basepair wrapping loop
        'LOOP',             # 4: loop link sequence
        'LOOPLINK',         # 5: loop link type
        'BULGE_L',          # 6: left decoration
        'BULGE_R',          # 7: right decoration
        'LFRAG',            # 8: left unpaired fragment
        'RFRAG',            # 9: right unpaired fragment
    ]
    # Add STACKED_{pair} context nonterminals (one per canonical pair)
    for a, b in CANONICAL_PAIRS:
        nonterminals.append(f'STACKED_{_pair_name(a, b)}')

    rules = []

    # --- SL -> STEM ---
    rules.append(('SL', [('STEM', 'N')], 1.0))

    # --- STEM productions ---
    # 1. Single basepair: STEM -> L(a) STEM R(b)  [lr_linear]
    for a in range(N_NUC):
        for b in range(N_NUC):
            if pi_bp[a, b] > 0:
                rules.append(('STEM', [
                    (left_terminal(a), 'T'),
                    ('STEM', 'N'),
                    (right_terminal(b), 'T'),
                ], kappa_S * p_bp * float(pi_bp[a, b])))

    # 2. Stacked pair (start): STEM -> L(a1) STACKED_{a1b1} R(b1)  [lr_linear]
    for idx, (a1, b1) in enumerate(CANONICAL_PAIRS):
        if pi_outer[idx] > 0:
            rules.append(('STEM', [
                (left_terminal(a1), 'T'),
                (f'STACKED_{_pair_name(a1, b1)}', 'N'),
                (right_terminal(b1), 'T'),
            ], kappa_S * p_st * float(pi_outer[idx])))

    # 3. Bulge: STEM -> BULGE_L STEM_AFTER_BULGE  [binary]
    rules.append(('STEM', [('BULGE_L', 'N'), ('STEM_AFTER_BULGE', 'N')],
                  kappa_S * p_bu))

    # STEM_AFTER_BULGE -> STEM BULGE_R  [binary]
    rules.append(('STEM_AFTER_BULGE', [('STEM', 'N'), ('BULGE_R', 'N')], 1.0))

    # 4. End stem: STEM -> CLOSE  [unary; CLOSE contains the loop inside]
    rules.append(('STEM', [('CLOSE', 'N')], 1.0 - kappa_S))

    # --- STACKED_{outer} productions ---
    for idx_o, (a1, b1) in enumerate(CANONICAL_PAIRS):
        nt_name = f'STACKED_{_pair_name(a1, b1)}'
        for idx_i, (a2, b2) in enumerate(CANONICAL_PAIRS):
            cond_w = float(pi_inner_given_outer[idx_o, idx_i])
            if cond_w > 0:
                # Terminal stack: STACKED_{a1b1} -> L(a2) STEM R(b2)
                rules.append((nt_name, [
                    (left_terminal(a2), 'T'),
                    ('STEM', 'N'),
                    (right_terminal(b2), 'T'),
                ], (1.0 - ext_K) * cond_w))

                # Extended stack: STACKED_{a1b1} -> L(a2) STACKED_{a2b2} R(b2)
                nt_inner = f'STACKED_{_pair_name(a2, b2)}'
                rules.append((nt_name, [
                    (left_terminal(a2), 'T'),
                    (nt_inner, 'N'),
                    (right_terminal(b2), 'T'),
                ], ext_K * cond_w))

    # --- CLOSE -> L(a) LOOP R(b) ---
    # Closing basepair wraps the loop inside it (lr_linear).
    for a in range(N_NUC):
        for b in range(N_NUC):
            if pi_bp[a, b] > 0:
                rules.append(('CLOSE', [
                    (left_terminal(a), 'T'),
                    ('LOOP', 'N'),
                    (right_terminal(b), 'T'),
                ], float(pi_bp[a, b])))

    # --- LOOP ---
    # LOOP -> LOOPLINK LOOP  [binary]
    rules.append(('LOOP', [('LOOPLINK', 'N'), ('LOOP', 'N')], kappa_L))
    # LOOP -> epsilon
    rules.append(('LOOP', [], 1.0 - kappa_L))

    # --- LOOPLINK ---
    rules.append(('LOOPLINK', [('LFRAG', 'N')], p_lf))
    rules.append(('LOOPLINK', [('RFRAG', 'N')], p_rf))
    rules.append(('LOOPLINK', [('SL', 'N')], p_sl))  # multiloop branch

    # --- BULGE_L (left decoration) ---
    # BULGE_L -> LFRAG BULGE_L  | SL BULGE_L  | epsilon
    rules.append(('BULGE_L', [('LFRAG', 'N'), ('BULGE_L', 'N')], 0.4))
    rules.append(('BULGE_L', [('SL', 'N'), ('BULGE_L', 'N')], 0.1))
    rules.append(('BULGE_L', [], 0.5))

    # --- BULGE_R (right decoration, made right-recursive) ---
    # Paper: RDECO -> RDECO RFRAG | RDECO SL | epsilon
    # Right-recursive: BULGE_R -> RFRAG BULGE_R | SL BULGE_R | epsilon
    rules.append(('BULGE_R', [('RFRAG', 'N'), ('BULGE_R', 'N')], 0.4))
    rules.append(('BULGE_R', [('SL', 'N'), ('BULGE_R', 'N')], 0.1))
    rules.append(('BULGE_R', [], 0.5))

    # --- LFRAG (left fragment, right-linear: L-emission) ---
    for a in range(N_NUC):
        # Extend: LFRAG -> L(a) LFRAG
        rules.append(('LFRAG', [
            (left_terminal(a), 'T'),
            ('LFRAG', 'N'),
        ], ext_B * float(pi[a])))
        # End: LFRAG -> L(a)
        rules.append(('LFRAG', [
            (left_terminal(a), 'T'),
        ], (1.0 - ext_B) * float(pi[a])))

    # --- RFRAG (right fragment, left-linear: R-emission) ---
    for a in range(N_NUC):
        # Extend: RFRAG -> RFRAG R(a)
        rules.append(('RFRAG', [
            ('RFRAG', 'N'),
            (right_terminal(a), 'T'),
        ], ext_B * float(pi[a])))
        # End: RFRAG -> R(a)
        rules.append(('RFRAG', [
            (right_terminal(a), 'T'),
        ], (1.0 - ext_B) * float(pi[a])))

    return build_grammar(nonterminals, N_SINGLET_TERMINALS, rules, start='SL')


# --- Sequence encoding ---

def encode_rna_sequence(sequence, structure):
    """Encode RNA sequence + dot-bracket as terminal array for TKFST grammar.

    Each position gets a left_terminal or right_terminal based on its structural
    role. The grammar's lr_linear rules determine which positions pair.

    Heuristic for L vs R assignment:
    - Paired left-half positions: L(a)
    - Paired right-half positions: R(a)
    - Unpaired positions before the midpoint of their enclosing context: L(a)
    - Unpaired positions after the midpoint: R(a)

    For simple stem-loops this produces the natural assignment.

    Args:
        sequence: str, RNA sequence (e.g. "GCAAAGUC")
        structure: str, dot-bracket notation (e.g. "((....))"), same length

    Returns:
        np.ndarray of int32, terminal indices of length len(sequence)
    """
    if len(sequence) != len(structure):
        raise ValueError(f"Length mismatch: seq={len(sequence)}, struct={len(structure)}")

    seq = sequence.upper()
    nuc_map = {c: i for i, c in enumerate(NUC_CHARS)}
    L = len(seq)

    # Find basepairs from dot-bracket
    pairs = {}
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

    terminals = np.zeros(L, dtype=np.int32)
    for i in range(L):
        a = nuc_map.get(seq[i], 0)
        if i in pairs:
            if pairs[i] > i:
                # Left half of basepair
                terminals[i] = left_terminal(a)
            else:
                # Right half of basepair
                terminals[i] = right_terminal(a)
        else:
            # Unpaired: determine L vs R by position relative to enclosing pair
            # Find innermost enclosing pair
            is_right = False
            for j in range(i - 1, -1, -1):
                if j in pairs and pairs[j] > i:
                    # j is left-half of a pair enclosing i
                    mid = (j + pairs[j]) / 2.0
                    is_right = (i > mid)
                    break
            terminals[i] = right_terminal(a) if is_right else left_terminal(a)

    return terminals


def decode_rna_sequence(terminals):
    """Decode terminal array back to RNA sequence.

    Returns:
        str: RNA sequence (structural information is lost)
    """
    chars = []
    for t in terminals:
        direction, a = decode_terminal(t)
        chars.append(NUC_CHARS[a])
    return ''.join(chars)


# --- Pair grammar elaboration ---

# Pair terminal encoding for the pair SCFG:
#   Match L:  M_L(ax, ay) = ax * 4 + ay                    (0-15)
#   Match R:  M_R(ax, ay) = 16 + ax * 4 + ay               (16-31)
#   Insert L: I_L(ay) = 32 + ay                             (32-35)
#   Insert R: I_R(ay) = 36 + ay                             (36-39)
#   Delete L: D_L(ax) = 40 + ax                             (40-43)
#   Delete R: D_R(ax) = 44 + ax                             (44-47)
N_PAIR_TERMINALS = 48


def match_left_terminal(ax, ay):
    """Match left terminal: ancestor ax, descendant ay."""
    return ax * N_NUC + ay


def match_right_terminal(ax, ay):
    """Match right terminal: ancestor ax, descendant ay."""
    return 16 + ax * N_NUC + ay


def insert_left_terminal(ay):
    """Insert left terminal: descendant ay only."""
    return 32 + ay


def insert_right_terminal(ay):
    """Insert right terminal: descendant ay only."""
    return 36 + ay


def delete_left_terminal(ax):
    """Delete left terminal: ancestor ax only."""
    return 40 + ax


def delete_right_terminal(ax):
    """Delete right terminal: ancestor ax only."""
    return 44 + ax


def decode_pair_terminal(t):
    """Decode pair terminal to (type, direction, nucleotides).

    Returns:
        (type, direction, nucs) where type is 'M'/'I'/'D',
        direction is 'L'/'R', nucs is (ax, ay) or (a,).
    """
    t = int(t)
    if t < 16:
        return ('M', 'L', (t // N_NUC, t % N_NUC))
    elif t < 32:
        t2 = t - 16
        return ('M', 'R', (t2 // N_NUC, t2 % N_NUC))
    elif t < 36:
        return ('I', 'L', (t - 32,))
    elif t < 40:
        return ('I', 'R', (t - 36,))
    elif t < 44:
        return ('D', 'L', (t - 40,))
    else:
        return ('D', 'R', (t - 44,))


def build_tkfst_pair_grammar(
    kappa_S=0.4, kappa_L=0.3,
    p_bp=0.6, p_st=0.3, p_bu=0.1,
    ext_K=0.5,
    p_lf=0.4, p_rf=0.4, p_sl=0.2,
    ext_B=0.3,
    pi=None, pi_bp=None, pi_stack=None,
    alpha_S=0.9, beta_S=0.1, gamma_S=0.15,
    alpha_L=0.9, beta_L=0.1, gamma_L=0.15,
    sub_matrix=None,
    sub_bp=None,
    sub_stack=None,
):
    """Build the TKFST pair SCFG via evolution elaboration (Section 7.3).

    Each emitting nonterminal in the singlet grammar is tripled into M/I/D
    variants with TKF91 transition weights. Separate TKF parameters for
    stems (α_S, β_S, γ_S) and loops (α_L, β_L, γ_L).

    Parameters:
        (structural params same as singlet)
        alpha_S, beta_S, gamma_S: TKF91 survival/insertion/post-delete for stems
        alpha_L, beta_L, gamma_L: TKF91 survival/insertion/post-delete for loops
        sub_matrix: (4,4) nucleotide substitution probability matrix P(y|x)
        sub_bp: (16,16) basepair substitution P((yL,yR)|(xL,xR)), or None for
                product of marginals
        sub_stack: (36,36) stacked-pair substitution, or None for product

    Returns:
        WCFG with 48 pair terminals
    """
    assert abs(p_bp + p_st + p_bu - 1.0) < 1e-10
    assert abs(p_lf + p_rf + p_sl - 1.0) < 1e-10

    if pi is None:
        pi = np.ones(N_NUC) / N_NUC
    else:
        pi = np.asarray(pi, dtype=np.float64)
    if pi_bp is None:
        pi_bp = _default_bp_equilibrium(pi)
    else:
        pi_bp = np.asarray(pi_bp, dtype=np.float64)
    if pi_stack is None:
        pi_stack = _default_stack_equilibrium(pi_bp)
    else:
        pi_stack = np.asarray(pi_stack, dtype=np.float64)
    if sub_matrix is None:
        sub_matrix = np.eye(N_NUC)  # identity (no substitution)
    else:
        sub_matrix = np.asarray(sub_matrix, dtype=np.float64)
    if sub_bp is None:
        # Product of marginal substitutions for basepairs
        sub_bp = np.zeros((N_NUC * N_NUC, N_NUC * N_NUC))
        for aL in range(N_NUC):
            for aR in range(N_NUC):
                for bL in range(N_NUC):
                    for bR in range(N_NUC):
                        sub_bp[aL * N_NUC + aR, bL * N_NUC + bR] = (
                            sub_matrix[aL, bL] * sub_matrix[aR, bR])
    else:
        sub_bp = np.asarray(sub_bp, dtype=np.float64)

    pi_outer = pi_stack.sum(axis=1)
    pi_inner_given_outer = np.zeros_like(pi_stack)
    for i in range(N_CANONICAL):
        if pi_outer[i] > 0:
            pi_inner_given_outer[i] = pi_stack[i] / pi_outer[i]

    # Build nonterminal list: triplicate emitting NTs into _M, _I, _D
    # Non-emitting NTs (structural only) also need M/I/D to track TKF state
    base_nts = [
        'SL', 'STEM', 'STEM_AFTER_BULGE', 'CLOSE',
        'LOOP', 'LOOPLINK', 'BULGE_L', 'BULGE_R', 'LFRAG', 'RFRAG',
    ]
    stacked_nts = [f'STACKED_{_pair_name(a, b)}' for a, b in CANONICAL_PAIRS]

    nonterminals = ['START']
    for nt in base_nts + stacked_nts:
        nonterminals.append(f'{nt}_M')
        nonterminals.append(f'{nt}_I')
        nonterminals.append(f'{nt}_D')

    rules = []

    # --- START -> SL_M / SL_I / SL_D ---
    # Initial TKF transition from immortal link (use stem parameters)
    rules.append(('START', [('SL_M', 'N')], (1.0 - beta_S) * alpha_S))
    rules.append(('START', [('SL_I', 'N')], beta_S))
    rules.append(('START', [('SL_D', 'N')], (1.0 - beta_S) * (1.0 - alpha_S)))

    # Helper: add a rule for each M/I/D variant
    def _add_structural_rules(lhs, rhs_list, weight):
        """Add structural (non-emitting) rules for all M/I/D variants."""
        for var in ['M', 'I', 'D']:
            rhs_var = [(f'{r}_{var}' if isinstance(r, str) else r, t)
                       for r, t in rhs_list]
            rules.append((f'{lhs}_{var}', rhs_var, weight))

    # --- SL -> STEM (unary, structural) ---
    _add_structural_rules('SL', [('STEM', 'N')], 1.0)

    # --- STEM productions ---
    # For STEM, the TKF link extension uses stem parameters.
    # The M/I/D on STEM refers to the alignment state of the current link.
    #
    # From STEM_M (post-match/insert state), transitions use beta_S:
    #   continue: (1-beta_S)*alpha_S for next M, beta_S for next I, (1-beta_S)*(1-alpha_S) for next D
    # From STEM_D (post-delete state), transitions use gamma_S:
    #   continue: (1-gamma_S)*alpha_S for next M, gamma_S for next I, (1-gamma_S)*(1-alpha_S) for next D

    for src_var in ['M', 'I', 'D']:
        beta_eff = beta_S if src_var in ('M', 'I') else gamma_S
        src = f'STEM_{src_var}'

        # 1. Single basepair match: STEM -> ML(ax) STEM_M MR(bx)
        #    with descendant drawn from substitution
        for ax in range(N_NUC):
            for bx in range(N_NUC):
                if pi_bp[ax, bx] <= 0:
                    continue
                for ay in range(N_NUC):
                    for by in range(N_NUC):
                        w = (kappa_S * p_bp * (1.0 - beta_eff) * alpha_S *
                             float(pi_bp[ax, bx]) *
                             float(sub_matrix[ax, ay]) * float(sub_matrix[bx, by]))
                        if w > 0:
                            rules.append((src, [
                                (match_left_terminal(ax, ay), 'T'),
                                ('STEM_M', 'N'),
                                (match_right_terminal(bx, by), 'T'),
                            ], w))

        # Single basepair insert: STEM -> IL(ay) STEM_I IR(by)
        for ay in range(N_NUC):
            for by in range(N_NUC):
                w = kappa_S * p_bp * beta_eff * float(pi_bp[ay, by])
                if w > 0:
                    rules.append((src, [
                        (insert_left_terminal(ay), 'T'),
                        ('STEM_I', 'N'),
                        (insert_right_terminal(by), 'T'),
                    ], w))

        # Single basepair delete: STEM -> DL(ax) STEM_D DR(bx)
        for ax in range(N_NUC):
            for bx in range(N_NUC):
                w = (kappa_S * p_bp * (1.0 - beta_eff) * (1.0 - alpha_S) *
                     float(pi_bp[ax, bx]))
                if w > 0:
                    rules.append((src, [
                        (delete_left_terminal(ax), 'T'),
                        ('STEM_D', 'N'),
                        (delete_right_terminal(bx), 'T'),
                    ], w))

        # 2. Stacked pair (start): STEM -> terminals STACKED_{pair}_var terminals
        for idx_o, (a1, b1) in enumerate(CANONICAL_PAIRS):
            if pi_outer[idx_o] <= 0:
                continue

            # Match stacked: emit ancestor outer pair + descendant outer pair
            for a1y in range(N_NUC):
                for b1y in range(N_NUC):
                    w = (kappa_S * p_st * (1.0 - beta_eff) * alpha_S *
                         float(pi_outer[idx_o]) *
                         float(sub_matrix[a1, a1y]) * float(sub_matrix[b1, b1y]))
                    if w > 0:
                        rules.append((src, [
                            (match_left_terminal(a1, a1y), 'T'),
                            (f'STACKED_{_pair_name(a1, b1)}_M', 'N'),
                            (match_right_terminal(b1, b1y), 'T'),
                        ], w))

            # Insert stacked: descendant outer pair only
            for a1y in range(N_NUC):
                for b1y in range(N_NUC):
                    # For inserted stacked pairs, use marginal bp equilibrium
                    w_bp = float(pi_bp[a1y, b1y]) if is_canonical(a1y, b1y) else 0.0
                    w = kappa_S * p_st * beta_eff * w_bp
                    if w > 0:
                        # Map inserted pair to nearest canonical for STACKED context
                        idx_ins = _canonical_index(a1y, b1y)
                        if idx_ins >= 0:
                            ca, cb = CANONICAL_PAIRS[idx_ins]
                            rules.append((src, [
                                (insert_left_terminal(a1y), 'T'),
                                (f'STACKED_{_pair_name(ca, cb)}_I', 'N'),
                                (insert_right_terminal(b1y), 'T'),
                            ], w))

            # Delete stacked: ancestor outer pair only
            w = (kappa_S * p_st * (1.0 - beta_eff) * (1.0 - alpha_S) *
                 float(pi_outer[idx_o]))
            if w > 0:
                rules.append((src, [
                    (delete_left_terminal(a1), 'T'),
                    (f'STACKED_{_pair_name(a1, b1)}_D', 'N'),
                    (delete_right_terminal(b1), 'T'),
                ], w))

        # 3. Bulge: STEM -> BULGE_L STEM_AFTER_BULGE  [structural, same var]
        w_bulge = kappa_S * p_bu
        # Bulge is structural (no emission), so just propagate alignment state
        rules.append((src, [
            (f'BULGE_L_{src_var}', 'N'),
            (f'STEM_AFTER_BULGE_{src_var}', 'N'),
        ], w_bulge))

        # 4. End stem: STEM -> CLOSE (CLOSE wraps LOOP inside)
        rules.append((src, [
            (f'CLOSE_{src_var}', 'N'),
        ], 1.0 - kappa_S))

    # --- STEM_AFTER_BULGE: structural passthrough ---
    _add_structural_rules('STEM_AFTER_BULGE', [('STEM', 'N'), ('BULGE_R', 'N')], 1.0)

    # --- STACKED_{pair} productions ---
    for idx_o, (a1, b1) in enumerate(CANONICAL_PAIRS):
        for var in ['M', 'I', 'D']:
            nt_src = f'STACKED_{_pair_name(a1, b1)}_{var}'
            for idx_i, (a2, b2) in enumerate(CANONICAL_PAIRS):
                cond_w = float(pi_inner_given_outer[idx_o, idx_i])
                if cond_w <= 0:
                    continue

                if var == 'M':
                    # Match inner: emit anc + desc
                    for a2y in range(N_NUC):
                        for b2y in range(N_NUC):
                            w = ((1.0 - ext_K) * cond_w *
                                 float(sub_matrix[a2, a2y]) * float(sub_matrix[b2, b2y]))
                            if w > 0:
                                rules.append((nt_src, [
                                    (match_left_terminal(a2, a2y), 'T'),
                                    ('STEM_M', 'N'),
                                    (match_right_terminal(b2, b2y), 'T'),
                                ], w))
                            # Extended
                            w_ext = (ext_K * cond_w *
                                     float(sub_matrix[a2, a2y]) * float(sub_matrix[b2, b2y]))
                            if w_ext > 0:
                                rules.append((nt_src, [
                                    (match_left_terminal(a2, a2y), 'T'),
                                    (f'STACKED_{_pair_name(a2, b2)}_M', 'N'),
                                    (match_right_terminal(b2, b2y), 'T'),
                                ], w_ext))
                elif var == 'I':
                    # Insert inner: desc only
                    for a2y in range(N_NUC):
                        for b2y in range(N_NUC):
                            if not is_canonical(a2y, b2y):
                                continue
                            w_ins = float(pi_bp[a2y, b2y])
                            if w_ins > 0:
                                idx_ins = _canonical_index(a2y, b2y)
                                ca2, cb2 = CANONICAL_PAIRS[idx_ins]
                                # Terminal
                                rules.append((nt_src, [
                                    (insert_left_terminal(a2y), 'T'),
                                    ('STEM_I', 'N'),
                                    (insert_right_terminal(b2y), 'T'),
                                ], (1.0 - ext_K) * cond_w * w_ins))
                                # Extended
                                rules.append((nt_src, [
                                    (insert_left_terminal(a2y), 'T'),
                                    (f'STACKED_{_pair_name(ca2, cb2)}_I', 'N'),
                                    (insert_right_terminal(b2y), 'T'),
                                ], ext_K * cond_w * w_ins))
                else:  # D
                    # Delete inner: anc only
                    w_del = (1.0 - ext_K) * cond_w
                    if w_del > 0:
                        rules.append((nt_src, [
                            (delete_left_terminal(a2), 'T'),
                            ('STEM_D', 'N'),
                            (delete_right_terminal(b2), 'T'),
                        ], w_del))
                    w_del_ext = ext_K * cond_w
                    if w_del_ext > 0:
                        rules.append((nt_src, [
                            (delete_left_terminal(a2), 'T'),
                            (f'STACKED_{_pair_name(a2, b2)}_D', 'N'),
                            (delete_right_terminal(b2), 'T'),
                        ], w_del_ext))

    # --- CLOSE (wraps LOOP inside closing basepair) ---
    for var in ['M', 'I', 'D']:
        src = f'CLOSE_{var}'
        if var == 'M':
            for ax in range(N_NUC):
                for bx in range(N_NUC):
                    if pi_bp[ax, bx] <= 0:
                        continue
                    for ay in range(N_NUC):
                        for by in range(N_NUC):
                            w = (float(pi_bp[ax, bx]) *
                                 float(sub_matrix[ax, ay]) * float(sub_matrix[bx, by]))
                            if w > 0:
                                rules.append((src, [
                                    (match_left_terminal(ax, ay), 'T'),
                                    (f'LOOP_{var}', 'N'),
                                    (match_right_terminal(bx, by), 'T'),
                                ], w))
        elif var == 'I':
            for ay in range(N_NUC):
                for by in range(N_NUC):
                    w = float(pi_bp[ay, by])
                    if w > 0:
                        rules.append((src, [
                            (insert_left_terminal(ay), 'T'),
                            (f'LOOP_{var}', 'N'),
                            (insert_right_terminal(by), 'T'),
                        ], w))
        else:
            for ax in range(N_NUC):
                for bx in range(N_NUC):
                    w = float(pi_bp[ax, bx])
                    if w > 0:
                        rules.append((src, [
                            (delete_left_terminal(ax), 'T'),
                            (f'LOOP_{var}', 'N'),
                            (delete_right_terminal(bx), 'T'),
                        ], w))

    # --- LOOP (uses loop TKF parameters) ---
    for src_var in ['M', 'I', 'D']:
        beta_eff = beta_L if src_var in ('M', 'I') else gamma_L
        src = f'LOOP_{src_var}'

        # Continue loop: need M/I/D transitions for next link
        for dest_var, tw in [('M', (1.0 - beta_eff) * alpha_L),
                             ('I', beta_eff),
                             ('D', (1.0 - beta_eff) * (1.0 - alpha_L))]:
            rules.append((src, [
                (f'LOOPLINK_{dest_var}', 'N'),
                (f'LOOP_{dest_var}', 'N'),
            ], kappa_L * tw))

        # End loop
        rules.append((src, [], (1.0 - kappa_L)))

    # --- LOOPLINK ---
    for var in ['M', 'I', 'D']:
        rules.append((f'LOOPLINK_{var}', [(f'LFRAG_{var}', 'N')], p_lf))
        rules.append((f'LOOPLINK_{var}', [(f'RFRAG_{var}', 'N')], p_rf))
        rules.append((f'LOOPLINK_{var}', [(f'SL_{var}', 'N')], p_sl))

    # --- BULGE_L / BULGE_R (structural, propagate alignment state) ---
    for var in ['M', 'I', 'D']:
        rules.append((f'BULGE_L_{var}', [
            (f'LFRAG_{var}', 'N'), (f'BULGE_L_{var}', 'N')], 0.4))
        rules.append((f'BULGE_L_{var}', [
            (f'SL_{var}', 'N'), (f'BULGE_L_{var}', 'N')], 0.1))
        rules.append((f'BULGE_L_{var}', [], 0.5))

        rules.append((f'BULGE_R_{var}', [
            (f'RFRAG_{var}', 'N'), (f'BULGE_R_{var}', 'N')], 0.4))
        rules.append((f'BULGE_R_{var}', [
            (f'SL_{var}', 'N'), (f'BULGE_R_{var}', 'N')], 0.1))
        rules.append((f'BULGE_R_{var}', [], 0.5))

    # --- LFRAG (L-emission, uses right_linear) ---
    for var in ['M', 'I', 'D']:
        src = f'LFRAG_{var}'
        if var == 'M':
            for ax in range(N_NUC):
                for ay in range(N_NUC):
                    w_e = ext_B * float(pi[ax]) * float(sub_matrix[ax, ay])
                    w_t = (1.0 - ext_B) * float(pi[ax]) * float(sub_matrix[ax, ay])
                    if w_e > 0:
                        rules.append((src, [
                            (match_left_terminal(ax, ay), 'T'),
                            (f'LFRAG_M', 'N'),
                        ], w_e))
                    if w_t > 0:
                        rules.append((src, [
                            (match_left_terminal(ax, ay), 'T'),
                        ], w_t))
        elif var == 'I':
            for ay in range(N_NUC):
                w_e = ext_B * float(pi[ay])
                w_t = (1.0 - ext_B) * float(pi[ay])
                if w_e > 0:
                    rules.append((src, [
                        (insert_left_terminal(ay), 'T'),
                        (f'LFRAG_I', 'N'),
                    ], w_e))
                if w_t > 0:
                    rules.append((src, [
                        (insert_left_terminal(ay), 'T'),
                    ], w_t))
        else:  # D
            for ax in range(N_NUC):
                w_e = ext_B * float(pi[ax])
                w_t = (1.0 - ext_B) * float(pi[ax])
                if w_e > 0:
                    rules.append((src, [
                        (delete_left_terminal(ax), 'T'),
                        (f'LFRAG_D', 'N'),
                    ], w_e))
                if w_t > 0:
                    rules.append((src, [
                        (delete_left_terminal(ax), 'T'),
                    ], w_t))

    # --- RFRAG (R-emission, uses left_linear) ---
    for var in ['M', 'I', 'D']:
        src = f'RFRAG_{var}'
        if var == 'M':
            for ax in range(N_NUC):
                for ay in range(N_NUC):
                    w_e = ext_B * float(pi[ax]) * float(sub_matrix[ax, ay])
                    w_t = (1.0 - ext_B) * float(pi[ax]) * float(sub_matrix[ax, ay])
                    if w_e > 0:
                        rules.append((src, [
                            (f'RFRAG_M', 'N'),
                            (match_right_terminal(ax, ay), 'T'),
                        ], w_e))
                    if w_t > 0:
                        rules.append((src, [
                            (match_right_terminal(ax, ay), 'T'),
                        ], w_t))
        elif var == 'I':
            for ay in range(N_NUC):
                w_e = ext_B * float(pi[ay])
                w_t = (1.0 - ext_B) * float(pi[ay])
                if w_e > 0:
                    rules.append((src, [
                        (f'RFRAG_I', 'N'),
                        (insert_right_terminal(ay), 'T'),
                    ], w_e))
                if w_t > 0:
                    rules.append((src, [
                        (insert_right_terminal(ay), 'T'),
                    ], w_t))
        else:  # D
            for ax in range(N_NUC):
                w_e = ext_B * float(pi[ax])
                w_t = (1.0 - ext_B) * float(pi[ax])
                if w_e > 0:
                    rules.append((src, [
                        (f'RFRAG_D', 'N'),
                        (delete_right_terminal(ax), 'T'),
                    ], w_e))
                if w_t > 0:
                    rules.append((src, [
                        (delete_right_terminal(ax), 'T'),
                    ], w_t))

    return build_grammar(nonterminals, N_PAIR_TERMINALS, rules, start='START')




# --- Rate EM ---

# Parameter factor tags used to annotate rules for EM M-step.
# Each tag represents a multiplicative factor in the rule weight.
# Binary parameter pairs: (tag, complement_tag) -> M-step is ratio of counts.
_FACTOR_TAGS = [
    'alpha_S', '1-alpha_S',     # stem survival
    'beta_S',  '1-beta_S',      # stem post-M/I insertion
    'gamma_S', '1-gamma_S',     # stem post-D insertion
    'alpha_L', '1-alpha_L',     # loop survival
    'beta_L',  '1-beta_L',      # loop post-M/I insertion
    'gamma_L', '1-gamma_L',     # loop post-D insertion
    'kappa_S', '1-kappa_S',     # stem extension
    'kappa_L', '1-kappa_L',     # loop extension
    'p_bp', 'p_st', 'p_bu',    # stem type (Dirichlet)
    'p_lf', 'p_rf', 'p_sl',    # loop link type (Dirichlet)
    'ext_K', '1-ext_K',        # stacked pair extension
    'ext_B', '1-ext_B',        # fragment extension
]


def _build_rule_annotations(
    kappa_S, kappa_L, p_bp, p_st, p_bu, ext_K, p_lf, p_rf, p_sl, ext_B,
    alpha_S, beta_S, gamma_S, alpha_L, beta_L, gamma_L,
):
    """Build rule factor annotations parallel to build_tkfst_pair_grammar.

    Returns a list of sets, one per rule. Each set contains the factor tags
    that appear as multiplicative factors in that rule's weight formula.

    This must exactly mirror the rule construction order in
    build_tkfst_pair_grammar.
    """
    annotations = []

    def ann(tags):
        """Record annotation for the current rule."""
        annotations.append(set(tags))

    # --- START -> SL_M / SL_I / SL_D ---
    ann(['1-beta_S', 'alpha_S'])    # SL_M
    ann(['beta_S'])                  # SL_I
    ann(['1-beta_S', '1-alpha_S'])  # SL_D

    # --- SL -> STEM (structural, 3 variants M/I/D) ---
    for _ in range(3):
        ann([])

    # --- STEM productions ---
    for src_var in ['M', 'I', 'D']:
        beta_tag = 'beta_S' if src_var in ('M', 'I') else 'gamma_S'
        not_beta_tag = '1-beta_S' if src_var in ('M', 'I') else '1-gamma_S'

        # 1. Single basepair match
        for ax in range(N_NUC):
            for bx in range(N_NUC):
                # skip if pi_bp[ax,bx] <= 0 — but we need to match the grammar
                # builder exactly. We use uniform pi_bp by default.
                for ay in range(N_NUC):
                    for by in range(N_NUC):
                        ann(['kappa_S', 'p_bp', not_beta_tag, 'alpha_S'])

        # 2. Single basepair insert
        for ay in range(N_NUC):
            for by in range(N_NUC):
                ann(['kappa_S', 'p_bp', beta_tag])

        # 3. Single basepair delete
        for ax in range(N_NUC):
            for bx in range(N_NUC):
                ann(['kappa_S', 'p_bp', not_beta_tag, '1-alpha_S'])

        # 4. Stacked pair (start) - Match
        for idx_o in range(N_CANONICAL):
            for a1y in range(N_NUC):
                for b1y in range(N_NUC):
                    ann(['kappa_S', 'p_st', not_beta_tag, 'alpha_S'])

        # Stacked pair - Insert
        for idx_o in range(N_CANONICAL):
            for a1y in range(N_NUC):
                for b1y in range(N_NUC):
                    # Only canonical pairs get rules
                    pass  # handled below

        # Stacked pair - Delete
        for idx_o in range(N_CANONICAL):
            ann(['kappa_S', 'p_st', not_beta_tag, '1-alpha_S'])

        # 5. Bulge
        ann(['kappa_S', 'p_bu'])

        # 6. End stem
        ann(['1-kappa_S'])

    # This approach of manually mirroring is too fragile.
    # Let me use a different approach.
    annotations.clear()
    return None  # Signal to use the runtime approach instead


def _classify_rule_factors(grammar):
    """Classify each rule's parameter factors by inspecting NT names and structure.

    Returns a list of sets of factor tags, one per rule.
    """
    annotations = []
    nt_names = grammar.nonterminals

    for p in grammar.productions:
        tags = set()
        lhs_name = nt_names[p.lhs]

        # Determine context: stem or loop
        is_stem_ctx = any(x in lhs_name for x in
                         ['STEM', 'CLOSE', 'STACKED', 'SL_', 'START'])
        is_loop_ctx = any(x in lhs_name for x in
                         ['LOOP', 'LOOPLINK', 'LFRAG', 'RFRAG', 'BULGE'])

        # Determine source alignment state
        if lhs_name == 'START':
            src_var = None  # Special
        elif lhs_name.endswith('_M'):
            src_var = 'M'
        elif lhs_name.endswith('_I'):
            src_var = 'I'
        elif lhs_name.endswith('_D'):
            src_var = 'D'
        else:
            src_var = None

        # Determine target alignment state (from RHS NTs)
        rhs_nts = [nt_names[r] for r in p.rhs
                    if isinstance(r, int) and r < grammar.n_nonterminals]
        # Filter: in lr_linear/right_linear/left_linear, rhs has terminal ints too
        # Actually p.rhs contains terminal indices as well. Need to use production type
        if p.is_binary:
            rhs_nts = [nt_names[p.rhs[0]], nt_names[p.rhs[1]]]
        elif p.is_unary:
            rhs_nts = [nt_names[p.rhs[0]]]
        elif p.is_right_linear:
            rhs_nts = [nt_names[p.rhs[1]]]  # A -> t B
        elif p.is_left_linear:
            rhs_nts = [nt_names[p.rhs[0]]]  # A -> B t
        elif p.is_lr_linear:
            rhs_nts = [nt_names[p.rhs[1]]]  # A -> t B t
        elif p.is_terminal or p.is_empty:
            rhs_nts = []
        else:
            rhs_nts = []

        dest_var = None
        if rhs_nts:
            main_nt = rhs_nts[0]
            if main_nt.endswith('_M'):
                dest_var = 'M'
            elif main_nt.endswith('_I'):
                dest_var = 'I'
            elif main_nt.endswith('_D'):
                dest_var = 'D'

        # Classify based on LHS
        if lhs_name == 'START':
            # START -> SL_{M,I,D}: TKF initial transition (stem params)
            if dest_var == 'M':
                tags.update(['1-beta_S', 'alpha_S'])
            elif dest_var == 'I':
                tags.add('beta_S')
            elif dest_var == 'D':
                tags.update(['1-beta_S', '1-alpha_S'])

        elif lhs_name.startswith('SL_'):
            # Structural passthrough, no parameter factors
            pass

        elif lhs_name.startswith('STEM_'):
            beta_tag = 'beta_S' if src_var in ('M', 'I') else 'gamma_S'
            not_beta_tag = '1-beta_S' if src_var in ('M', 'I') else '1-gamma_S'

            # Determine what type of STEM production
            has_terminals = p.is_lr_linear or p.is_terminal or p.is_right_linear or p.is_left_linear
            is_end = p.is_unary and rhs_nts and 'CLOSE' in rhs_nts[0]
            is_bulge = p.is_binary and rhs_nts and 'BULGE' in rhs_nts[0]

            if is_end:
                tags.add('1-kappa_S')
            elif is_bulge:
                tags.update(['kappa_S', 'p_bu'])
            elif has_terminals and rhs_nts:
                # Basepair or stacked pair emission
                main_rhs = rhs_nts[0]
                is_stacked = 'STACKED' in main_rhs

                if is_stacked:
                    tags.add('p_st')
                else:
                    tags.add('p_bp')

                tags.add('kappa_S')

                if dest_var == 'M':
                    tags.update([not_beta_tag, 'alpha_S'])
                elif dest_var == 'I':
                    tags.add(beta_tag)
                elif dest_var == 'D':
                    tags.update([not_beta_tag, '1-alpha_S'])

        elif lhs_name.startswith('STEM_AFTER_BULGE_'):
            # Structural passthrough
            pass

        elif lhs_name.startswith('STACKED_'):
            # Inner basepair of stacked pair
            main_rhs = rhs_nts[0] if rhs_nts else ''
            is_extended = 'STACKED' in main_rhs
            if is_extended:
                tags.add('ext_K')
            else:
                tags.add('1-ext_K')

        elif lhs_name.startswith('CLOSE_'):
            # Closing basepair — emission only, no TKF transition tags
            pass

        elif lhs_name.startswith('LOOP_'):
            beta_tag = 'beta_L' if src_var in ('M', 'I') else 'gamma_L'
            not_beta_tag = '1-beta_L' if src_var in ('M', 'I') else '1-gamma_L'

            if p.is_empty:
                # End loop
                tags.add('1-kappa_L')
            elif p.is_binary:
                # Continue: LOOP -> LOOPLINK LOOP
                tags.add('kappa_L')
                if dest_var == 'M':
                    tags.update([not_beta_tag, 'alpha_L'])
                elif dest_var == 'I':
                    tags.add(beta_tag)
                elif dest_var == 'D':
                    tags.update([not_beta_tag, '1-alpha_L'])

        elif lhs_name.startswith('LOOPLINK_'):
            # LOOPLINK -> LFRAG / RFRAG / SL
            if rhs_nts:
                child = rhs_nts[0]
                if 'LFRAG' in child:
                    tags.add('p_lf')
                elif 'RFRAG' in child:
                    tags.add('p_rf')
                elif 'SL' in child:
                    tags.add('p_sl')

        elif lhs_name.startswith('BULGE_L_') or lhs_name.startswith('BULGE_R_'):
            # Bulge extension — no parameter tags for now
            # (bulge extension probs are fixed structural params)
            pass

        elif lhs_name.startswith('LFRAG_') or lhs_name.startswith('RFRAG_'):
            # Fragment rules: extension vs termination
            if p.is_right_linear or p.is_left_linear:
                # A -> t B or A -> B t: extension
                if rhs_nts and ('LFRAG' in rhs_nts[0] or 'RFRAG' in rhs_nts[0]):
                    tags.add('ext_B')
            elif p.is_terminal:
                # A -> t: termination
                tags.add('1-ext_B')

        annotations.append(tags)

    return annotations


def tkfst_e_step(x_seq, y_seq, alignment):
    """Build an E-step callable for TKFST pair SCFG.

    Returns a function e_step(params) -> (log_likelihood, stats) suitable
    for em_optimize().

    Args:
        x_seq, y_seq: integer nucleotide arrays
        alignment: list of (x_idx_or_None, y_idx_or_None) pairs

    Returns:
        e_step: callable(params_dict) -> (float, dict)
    """
    col_types, col_x, col_y = alignment_to_columns(x_seq, y_seq, alignment)

    def e_step(params):
        grammar = build_tkfst_pair_grammar(**params)
        log_prob, log_I = tkfst_pair_inside_aligned(
            grammar, col_types, col_x, col_y, return_table=True)
        if log_prob <= -1e29:
            return log_prob, None

        log_O = tkfst_pair_outside_aligned(grammar, col_types, col_x, col_y, log_I)
        counts = tkfst_pair_expected_counts_aligned(
            grammar, col_types, col_x, col_y, log_I, log_O)

        annotations = _classify_rule_factors(grammar)

        stats = {}
        for tag in _FACTOR_TAGS:
            stats[tag] = 0.0
        for pi_idx, (count, tags) in enumerate(zip(counts, annotations)):
            for tag in tags:
                if tag in stats:
                    stats[tag] += count

        return log_prob, stats

    return e_step


def tkfst_m_step(params, stats):
    """M-step for TKFST pair SCFG: update params from sufficient statistics.

    Args:
        params: current parameter dict
        stats: dict of factor tag counts from E-step

    Returns:
        new_params: updated parameter dict
    """
    if stats is None:
        return params

    new_params = dict(params)
    _EPS = 1e-10

    def _binary_mstep(tag, comp_tag, key):
        n = stats[tag]
        d = stats[tag] + stats[comp_tag]
        if d > _EPS:
            new_params[key] = np.clip(n / d, _EPS, 1.0 - _EPS)

    _binary_mstep('alpha_S', '1-alpha_S', 'alpha_S')
    _binary_mstep('beta_S', '1-beta_S', 'beta_S')
    _binary_mstep('gamma_S', '1-gamma_S', 'gamma_S')
    _binary_mstep('alpha_L', '1-alpha_L', 'alpha_L')
    _binary_mstep('beta_L', '1-beta_L', 'beta_L')
    _binary_mstep('gamma_L', '1-gamma_L', 'gamma_L')
    _binary_mstep('kappa_S', '1-kappa_S', 'kappa_S')
    _binary_mstep('kappa_L', '1-kappa_L', 'kappa_L')
    _binary_mstep('ext_K', '1-ext_K', 'ext_K')
    _binary_mstep('ext_B', '1-ext_B', 'ext_B')

    def _dirichlet_mstep(tags, keys):
        total = sum(stats[t] for t in tags)
        if total > _EPS:
            for tag, key in zip(tags, keys):
                new_params[key] = np.clip(stats[tag] / total, _EPS, 1.0 - _EPS)
            s = sum(new_params[k] for k in keys)
            for k in keys:
                new_params[k] /= s

    _dirichlet_mstep(['p_bp', 'p_st', 'p_bu'], ['p_bp', 'p_st', 'p_bu'])
    _dirichlet_mstep(['p_lf', 'p_rf', 'p_sl'], ['p_lf', 'p_rf', 'p_sl'])

    return new_params


def tkfst_em_step(x_seq, y_seq, params, alignment=None):
    """One EM step for TKFST pair SCFG.

    Convenience wrapper combining tkfst_e_step and tkfst_m_step.

    Args:
        x_seq, y_seq: integer nucleotide arrays
        params: dict with keys matching build_tkfst_pair_grammar kwargs
        alignment: list of (x_idx_or_None, y_idx_or_None) pairs. Required.

    Returns:
        log_prob: log P(alignment | params)
        new_params: updated parameter dict
    """
    if alignment is None:
        raise ValueError("alignment is required for tkfst_em_step")

    e_step = tkfst_e_step(x_seq, y_seq, alignment)
    log_prob, stats = e_step(params)
    new_params = tkfst_m_step(params, stats)
    return log_prob, new_params


def tkfst_em_loop(x_seq, y_seq, params, alignment=None,
                  n_iter=10, tol=1e-4, verbose=False):
    """Run multiple EM iterations for TKFST pair SCFG.

    Args:
        x_seq, y_seq: integer nucleotide arrays
        params: initial parameter dict
        alignment: list of (x_idx_or_None, y_idx_or_None) pairs. Required.
        n_iter: maximum iterations
        tol: convergence tolerance on log-prob improvement
        verbose: print progress

    Returns:
        log_prob: final log P(alignment | params)
        params: final parameters
        history: list of (iteration, log_prob) tuples
    """
    history = []
    prev_lp = -np.inf

    for i in range(n_iter):
        lp, params = tkfst_em_step(x_seq, y_seq, params, alignment=alignment)
        history.append((i, lp))

        if verbose:
            print(f"  EM iter {i}: log_prob = {lp:.6f}")

        if i > 0 and abs(lp - prev_lp) < tol:
            if verbose:
                print(f"  Converged at iter {i}")
            break

        prev_lp = lp

    return lp, params, history


def tkfst_em_optimize(x_seq, y_seq, params, alignment,
                      n_iter=50, convergence_tol=0.1, verbose=False):
    """Train TKFST pair SCFG using em_optimize().

    Wraps tkfst_e_step/tkfst_m_step into the em_optimize() interface
    for EM with optional L-BFGS switching.

    Args:
        x_seq, y_seq: integer nucleotide arrays
        params: initial parameter dict
        alignment: list of (x_idx_or_None, y_idx_or_None) pairs
        n_iter: maximum EM iterations
        convergence_tol: convergence tolerance
        verbose: print progress

    Returns:
        OptimizeResult with final params, log_probs, timing
    """
    from ..train.optimizer import em_optimize

    e_step = tkfst_e_step(x_seq, y_seq, alignment)
    return em_optimize(
        params, e_step, tkfst_m_step,
        n_iter=n_iter, convergence_tol=convergence_tol,
        verbose=verbose,
    )


# --- Distillation ---

def tkfst_distill_counts(grammar, counts):
    """Extract emission statistics from IO expected counts.

    Classifies each rule by its emitted terminals and aggregates expected
    counts into substitution and indel frequency tables.

    Args:
        grammar: WCFG (pair grammar)
        counts: list of expected rule counts from expected_counts_aligned

    Returns:
        dict with:
            'match_counts': (4, 4) expected match count for each (anc, desc) pair
            'insert_counts': (4,) expected insert count for each desc nucleotide
            'delete_counts': (4,) expected delete count for each anc nucleotide
            'stem_match_bp': (4, 4, 4, 4) basepair match (aL, aR, dL, dR)
            'stem_insert_bp': (4, 4) basepair insert (dL, dR)
            'stem_delete_bp': (4, 4) basepair delete (aL, aR)
            'loop_match': (4, 4) singlet match in loops
            'loop_insert': (4,) singlet insert in loops
            'loop_delete': (4,) singlet delete in loops
    """
    nt_names = grammar.nonterminals
    n_rules = len(grammar.productions)

    # Aggregate tables
    match_counts = np.zeros((N_NUC, N_NUC))
    insert_counts = np.zeros(N_NUC)
    delete_counts = np.zeros(N_NUC)

    stem_match_bp = np.zeros((N_NUC, N_NUC, N_NUC, N_NUC))
    stem_insert_bp = np.zeros((N_NUC, N_NUC))
    stem_delete_bp = np.zeros((N_NUC, N_NUC))

    loop_match = np.zeros((N_NUC, N_NUC))
    loop_insert = np.zeros(N_NUC)
    loop_delete = np.zeros(N_NUC)

    for pi_idx, p in enumerate(grammar.productions):
        c = counts[pi_idx]
        if c < 1e-30:
            continue

        lhs_name = nt_names[p.lhs]
        is_loop = any(x in lhs_name for x in ['LFRAG', 'RFRAG'])
        is_stem = any(x in lhs_name for x in ['STEM', 'CLOSE', 'STACKED'])

        # Extract emitted terminals
        if p.is_lr_linear:
            # A -> tL B tR: basepair emission
            tL, _, tR = p.rhs
            typL, dirL, nucsL = decode_pair_terminal(tL)
            typR, dirR, nucsR = decode_pair_terminal(tR)

            if typL == 'M' and typR == 'M':
                aL, dL = nucsL
                aR, dR = nucsR
                stem_match_bp[aL, aR, dL, dR] += c
                match_counts[aL, dL] += c
                match_counts[aR, dR] += c
            elif typL == 'I' and typR == 'I':
                dL = nucsL[0]
                dR = nucsR[0]
                stem_insert_bp[dL, dR] += c
                insert_counts[dL] += c
                insert_counts[dR] += c
            elif typL == 'D' and typR == 'D':
                aL = nucsL[0]
                aR = nucsR[0]
                stem_delete_bp[aL, aR] += c
                delete_counts[aL] += c
                delete_counts[aR] += c

        elif p.is_right_linear:
            # A -> t B: left emission (LFRAG)
            t = p.rhs[0]
            typ, direction, nucs = decode_pair_terminal(t)
            if typ == 'M':
                ax, ay = nucs
                loop_match[ax, ay] += c
                match_counts[ax, ay] += c
            elif typ == 'I':
                ay = nucs[0]
                loop_insert[ay] += c
                insert_counts[ay] += c
            elif typ == 'D':
                ax = nucs[0]
                loop_delete[ax] += c
                delete_counts[ax] += c

        elif p.is_left_linear:
            # A -> B t: right emission (RFRAG)
            t = p.rhs[1]
            typ, direction, nucs = decode_pair_terminal(t)
            if typ == 'M':
                ax, ay = nucs
                loop_match[ax, ay] += c
                match_counts[ax, ay] += c
            elif typ == 'I':
                ay = nucs[0]
                loop_insert[ay] += c
                insert_counts[ay] += c
            elif typ == 'D':
                ax = nucs[0]
                loop_delete[ax] += c
                delete_counts[ax] += c

        elif p.is_terminal:
            # A -> t: terminal emission (fragment termination)
            t = p.rhs[0]
            typ, direction, nucs = decode_pair_terminal(t)
            if typ == 'M':
                ax, ay = nucs
                if is_loop:
                    loop_match[ax, ay] += c
                match_counts[ax, ay] += c
            elif typ == 'I':
                ay = nucs[0]
                if is_loop:
                    loop_insert[ay] += c
                insert_counts[ay] += c
            elif typ == 'D':
                ax = nucs[0]
                if is_loop:
                    loop_delete[ax] += c
                delete_counts[ax] += c

    return {
        'match_counts': match_counts,
        'insert_counts': insert_counts,
        'delete_counts': delete_counts,
        'stem_match_bp': stem_match_bp,
        'stem_insert_bp': stem_insert_bp,
        'stem_delete_bp': stem_delete_bp,
        'loop_match': loop_match,
        'loop_insert': loop_insert,
        'loop_delete': loop_delete,
    }


def tkfst_distill(grammar, x_seq, y_seq, alignment):
    """Distill TKFST pair SCFG to order-1 statistics from a single sequence pair.

    Runs alignment-constrained Inside-Outside and extracts emission frequency
    tables that characterize the order-1 substitution and indel behavior.

    Args:
        grammar: WCFG (pair grammar from build_tkfst_pair_grammar)
        x_seq, y_seq: integer nucleotide arrays
        alignment: list of (x_idx_or_None, y_idx_or_None) pairs

    Returns:
        dict with distillation statistics (see tkfst_distill_counts)
        plus 'log_prob' and normalized substitution matrix 'sub_est'
    """
    col_types, col_x, col_y = alignment_to_columns(x_seq, y_seq, alignment)
    log_prob, log_I = tkfst_pair_inside_aligned(
        grammar, col_types, col_x, col_y, return_table=True)
    if log_prob <= -1e29:
        return None

    log_O = tkfst_pair_outside_aligned(grammar, col_types, col_x, col_y, log_I)
    counts = tkfst_pair_expected_counts_aligned(
        grammar, col_types, col_x, col_y, log_I, log_O)
    stats = tkfst_distill_counts(grammar, counts)

    # Estimate substitution matrix from match counts
    mc = stats['match_counts']
    row_sums = mc.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 1e-30, row_sums, 1.0)
    sub_est = np.where(row_sums > 1e-30, mc / safe_sums, np.eye(N_NUC))

    stats['log_prob'] = log_prob
    stats['sub_est'] = sub_est

    return stats


def aggregate_distill_stats(stats_list):
    """Aggregate distillation statistics across multiple sequence pairs.

    Args:
        stats_list: list of dicts from tkfst_distill() (None entries skipped)

    Returns:
        dict with summed counts and re-normalized substitution estimate
    """
    agg = {
        'match_counts': np.zeros((N_NUC, N_NUC)),
        'insert_counts': np.zeros(N_NUC),
        'delete_counts': np.zeros(N_NUC),
        'stem_match_bp': np.zeros((N_NUC, N_NUC, N_NUC, N_NUC)),
        'stem_insert_bp': np.zeros((N_NUC, N_NUC)),
        'stem_delete_bp': np.zeros((N_NUC, N_NUC)),
        'loop_match': np.zeros((N_NUC, N_NUC)),
        'loop_insert': np.zeros(N_NUC),
        'loop_delete': np.zeros(N_NUC),
        'total_log_prob': 0.0,
        'n_pairs': 0,
    }
    for s in stats_list:
        if s is None:
            continue
        for k in ['match_counts', 'insert_counts', 'delete_counts',
                   'stem_match_bp', 'stem_insert_bp', 'stem_delete_bp',
                   'loop_match', 'loop_insert', 'loop_delete']:
            agg[k] = agg[k] + s[k]
        agg['total_log_prob'] += s['log_prob']
        agg['n_pairs'] += 1

    # Estimate substitution matrix with Laplace smoothing
    mc = agg['match_counts'] + 0.01  # pseudocount avoids zero entries
    row_sums = mc.sum(axis=1, keepdims=True)
    agg['sub_est'] = mc / row_sums

    return agg


def distill_to_wptt_weights(agg_stats, t=1.0):
    """Convert aggregated distillation stats to WPTT rule weights.

    Estimates TKF91 parameters from match/insert/delete counts, then
    calls tkf_wptt_weights() to produce the WPTT rule weight dict.

    Args:
        agg_stats: dict from aggregate_distill_stats()
        t: branch length (default 1.0; counts are insensitive to t
           but the WPTT parameterization requires it)

    Returns:
        wptt_weights: dict of WPTT rule weights
        estimated_params: dict with estimated TKF91 parameters
    """
    from ..distill.wptt import tkf_wptt_weights

    n_match = float(agg_stats['match_counts'].sum())
    n_insert = float(agg_stats['insert_counts'].sum())
    n_delete = float(agg_stats['delete_counts'].sum())
    n_total = n_match + n_insert + n_delete

    if n_total < 1e-30:
        # No counts; use defaults
        ins_rate, del_rate = 0.05, 0.10
    else:
        # Estimate birth/death fractions
        birth_frac = n_insert / n_total
        death_frac = n_delete / n_total

        # Simple moment-matching: ins_rate ≈ birth_frac * scale,
        # del_rate ≈ death_frac * scale. Use t to set scale.
        ins_rate = max(birth_frac / t, 1e-6)
        del_rate = max(death_frac / t, 1e-6)

    sub_matrix = agg_stats['sub_est']
    wptt_weights = tkf_wptt_weights(ins_rate, del_rate, t,
                                     subst_matrix=sub_matrix)

    estimated_params = {
        'ins_rate': ins_rate,
        'del_rate': del_rate,
        't': t,
        'n_match': n_match,
        'n_insert': n_insert,
        'n_delete': n_delete,
    }

    return wptt_weights, estimated_params


# ---------------------------------------------------------------------------
# Alignment-constrained Pair SCFG Inside/Outside ("cheap/fast" 1D version)
# ---------------------------------------------------------------------------
#
# Given a fixed pairwise alignment, the 4D pair SCFG DP collapses to 1D
# over alignment columns.  Each column is Match (x,y), Insert (-,y), or
# Delete (x,-).  The DP table is I[nt, i, j] for alignment-column spans
# [i, j), giving O(N * L_aln^3) time and O(N * L_aln^2) memory.

# Column type constants
COL_M, COL_I, COL_D = 0, 1, 2


def alignment_to_columns(x_seq, y_seq, alignment):
    """Convert a pairwise alignment to column arrays.

    Args:
        x_seq: (Lx,) ancestor sequence (int array)
        y_seq: (Ly,) descendant sequence (int array)
        alignment: list of (x_idx_or_None, y_idx_or_None) pairs, e.g.
            [(0,0), (1,1), (2,None), (None,2), (3,3)]
            Match = (int,int), Insert = (None,int), Delete = (int,None)

    Returns:
        col_types: (L_aln,) int array — COL_M, COL_I, or COL_D
        col_x: (L_aln,) int array — ancestor nucleotide (-1 for inserts)
        col_y: (L_aln,) int array — descendant nucleotide (-1 for deletes)
    """
    L = len(alignment)
    col_types = np.empty(L, dtype=np.int32)
    col_x = np.full(L, -1, dtype=np.int32)
    col_y = np.full(L, -1, dtype=np.int32)
    for c, (xi, yi) in enumerate(alignment):
        if xi is not None and yi is not None:
            col_types[c] = COL_M
            col_x[c] = int(x_seq[xi])
            col_y[c] = int(y_seq[yi])
        elif xi is None and yi is not None:
            col_types[c] = COL_I
            col_y[c] = int(y_seq[yi])
        elif xi is not None and yi is None:
            col_types[c] = COL_D
            col_x[c] = int(x_seq[xi])
        else:
            raise ValueError(f"Column {c}: both x and y are None")
    return col_types, col_x, col_y


def _precompute_term_col_match(col_types, col_x, col_y):
    """Precompute which terminals match at each alignment column.

    Returns:
        match_at: (L_aln,) list of sets of matching terminal indices
    """
    L = len(col_types)
    match_at = [set() for _ in range(L)]
    for c in range(L):
        ct = col_types[c]
        xc = col_x[c]
        yc = col_y[c]
        if ct == COL_M:
            # Match terminals: M_L(ax,ay) and M_R(ax,ay)
            t_l = match_left_terminal(xc, yc)
            t_r = match_right_terminal(xc, yc)
            match_at[c].add(t_l)
            match_at[c].add(t_r)
        elif ct == COL_I:
            # Insert terminals: I_L(ay) and I_R(ay)
            t_l = insert_left_terminal(yc)
            t_r = insert_right_terminal(yc)
            match_at[c].add(t_l)
            match_at[c].add(t_r)
        elif ct == COL_D:
            # Delete terminals: D_L(ax) and D_R(ax)
            t_l = delete_left_terminal(xc)
            t_r = delete_right_terminal(xc)
            match_at[c].add(t_l)
            match_at[c].add(t_r)
    return match_at


def tkfst_pair_inside_aligned(grammar, col_types, col_x, col_y,
                              return_table=False):
    """Alignment-constrained Inside for TKFST pair SCFG.

    Given a fixed alignment (as column arrays from alignment_to_columns()),
    computes I[A, i, j] = P(columns[i:j] | A) for all nonterminals A and
    alignment column spans [i, j).

    This is the "cheap" O(N * L^3) algorithm for training, where L = L_aln.

    Args:
        grammar: WCFG (TKFST pair SCFG with 48 terminals)
        col_types: (L,) int array of column types (COL_M/COL_I/COL_D)
        col_x: (L,) int array of ancestor nucleotides (-1 for insert)
        col_y: (L,) int array of descendant nucleotides (-1 for delete)
        return_table: if True, also return the full Inside table

    Returns:
        log_prob: log P(alignment | grammar)
        (optional) log_I: dict mapping (A, i, j) -> log prob
    """
    L = len(col_types)
    N = grammar.n_nonterminals
    NEG_INF = -1e30

    log_I = {}

    def get_I(a, i, j):
        return log_I.get((a, i, j), NEG_INF)

    def logaddexp_I(a, i, j, val):
        old = log_I.get((a, i, j), NEG_INF)
        log_I[(a, i, j)] = np.logaddexp(old, val)

    # Precompute unary closure U = (I - W)^{-1}
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
                    new_Wk[a, b] = np.logaddexp(
                        new_Wk[a, b], Wk[a, c] + W[c, b])
        if np.allclose(np.exp(np.maximum(new_Wk, -100)), 0, atol=1e-15):
            break
        Wk = new_Wk

    def _unary_close(i, j):
        vals = np.array([get_I(a, i, j) for a in range(N)])
        for a in range(N):
            new_val = NEG_INF
            for b in range(N):
                new_val = np.logaddexp(new_val, log_U[a, b] + vals[b])
            if new_val > NEG_INF:
                log_I[(a, i, j)] = new_val

    # Precompute terminal matches at each column
    match_at = _precompute_term_col_match(col_types, col_x, col_y)

    # Pre-categorize productions
    prods_empty = [p for p in grammar.productions if p.is_empty]
    prods_terminal = [p for p in grammar.productions if p.is_terminal]
    prods_right_linear = [p for p in grammar.productions if p.is_right_linear]
    prods_left_linear = [p for p in grammar.productions if p.is_left_linear]
    prods_lr_linear = [p for p in grammar.productions if p.is_lr_linear]
    prods_binary = [p for p in grammar.productions if p.is_binary]

    # Index productions by terminal for fast lookup
    from collections import defaultdict
    lr_by_terms = defaultdict(list)
    for p in prods_lr_linear:
        tL, _, tR = p.rhs
        lr_by_terms[(tL, tR)].append(p)

    term_by_t = defaultdict(list)
    for p in prods_terminal:
        term_by_t[p.rhs[0]].append(p)

    rl_by_t = defaultdict(list)
    for p in prods_right_linear:
        rl_by_t[p.rhs[0]].append(p)

    ll_by_t = defaultdict(list)
    for p in prods_left_linear:
        ll_by_t[p.rhs[1]].append(p)

    # Base case: empty spans
    for i in range(L + 1):
        for p in prods_empty:
            logaddexp_I(p.lhs, i, i, np.log(max(p.weight, 1e-300)))
        _unary_close(i, i)

    # Fill by increasing span size
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span

            # Terminal: A -> t (span = 1 column)
            if span == 1:
                for t_idx in match_at[i]:
                    for p in term_by_t.get(t_idx, []):
                        logaddexp_I(p.lhs, i, j,
                                    np.log(max(p.weight, 1e-300)))

            # Right-linear: A -> t B (left edge terminal at column i)
            for t_idx in match_at[i]:
                for p in rl_by_t.get(t_idx, []):
                    B = p.rhs[1]
                    val = get_I(B, i + 1, j)
                    if val > NEG_INF:
                        logaddexp_I(p.lhs, i, j,
                                    np.log(max(p.weight, 1e-300)) + val)

            # Left-linear: A -> B t (right edge terminal at column j-1)
            for t_idx in match_at[j - 1]:
                for p in ll_by_t.get(t_idx, []):
                    B = p.rhs[0]
                    val = get_I(B, i, j - 1)
                    if val > NEG_INF:
                        logaddexp_I(p.lhs, i, j,
                                    np.log(max(p.weight, 1e-300)) + val)

            # LR-linear: A -> tL B tR (span >= 2)
            if span >= 2:
                for tL in match_at[i]:
                    for tR in match_at[j - 1]:
                        prods = lr_by_terms.get((tL, tR), [])
                        for p in prods:
                            B = p.rhs[1]
                            val = get_I(B, i + 1, j - 1)
                            if val > NEG_INF:
                                logaddexp_I(
                                    p.lhs, i, j,
                                    np.log(max(p.weight, 1e-300)) + val)

            # Binary: A -> B C (split at alignment column k)
            vals_before = {a: get_I(a, i, j) for a in range(N)}
            _unary_close(i, j)
            closed_vals = {a: get_I(a, i, j) for a in range(N)}
            for a in range(N):
                if vals_before[a] > NEG_INF:
                    log_I[(a, i, j)] = vals_before[a]
                elif (a, i, j) in log_I:
                    del log_I[(a, i, j)]

            for p in prods_binary:
                B, C = p.rhs
                log_w = np.log(max(p.weight, 1e-300))
                for k in range(i, j + 1):
                    if k == j:
                        b_val = closed_vals[B]
                    else:
                        b_val = get_I(B, i, k)
                    if k == i:
                        c_val = closed_vals[C]
                    else:
                        c_val = get_I(C, k, j)
                    val = b_val + c_val
                    if val > NEG_INF + 1e20:
                        logaddexp_I(p.lhs, i, j, log_w + val)

            _unary_close(i, j)

    log_prob = get_I(grammar.start, 0, L)
    if return_table:
        return log_prob, log_I
    return log_prob


def tkfst_pair_outside_aligned(grammar, col_types, col_x, col_y, log_I):
    """Alignment-constrained Outside for TKFST pair SCFG.

    Computes O[A, i, j] = log P(rest of derivation | A generates columns[i:j]).

    Args:
        grammar: WCFG
        col_types, col_x, col_y: from alignment_to_columns()
        log_I: Inside table from tkfst_pair_inside_aligned(return_table=True)

    Returns:
        log_O: dict mapping (A, i, j) -> log outside prob
    """
    L = len(col_types)
    N = grammar.n_nonterminals
    NEG_INF = -1e30

    log_O = {}

    def get_O(a, i, j):
        return log_O.get((a, i, j), NEG_INF)

    def logaddexp_O(a, i, j, val):
        old = log_O.get((a, i, j), NEG_INF)
        log_O[(a, i, j)] = np.logaddexp(old, val)

    def get_In(a, i, j):
        return log_I.get((a, i, j), NEG_INF)

    # Precompute unary closure (transposed for outside)
    log_U_T = np.full((N, N), NEG_INF)
    for a in range(N):
        log_U_T[a, a] = 0.0
    W = np.full((N, N), NEG_INF)
    for p in grammar.productions:
        if p.is_unary:
            W[p.lhs, p.rhs[0]] = np.logaddexp(
                W[p.lhs, p.rhs[0]], np.log(max(p.weight, 1e-300)))
    Wk = W.copy()
    for _ in range(N):
        for a in range(N):
            for b in range(N):
                log_U_T[a, b] = np.logaddexp(log_U_T[a, b], Wk[b, a])
        new_Wk = np.full((N, N), NEG_INF)
        for a in range(N):
            for b in range(N):
                for c in range(N):
                    new_Wk[a, b] = np.logaddexp(
                        new_Wk[a, b], Wk[a, c] + W[c, b])
        if np.allclose(np.exp(np.maximum(new_Wk, -100)), 0, atol=1e-15):
            break
        Wk = new_Wk

    def _outside_unary_close(i, j):
        """Apply transposed unary closure to outside values."""
        vals = np.array([get_O(a, i, j) for a in range(N)])
        for a in range(N):
            new_val = NEG_INF
            for b in range(N):
                new_val = np.logaddexp(new_val, log_U_T[a, b] + vals[b])
            if new_val > NEG_INF:
                log_O[(a, i, j)] = new_val

    # Pre-categorize productions
    prods_right_linear = [p for p in grammar.productions if p.is_right_linear]
    prods_left_linear = [p for p in grammar.productions if p.is_left_linear]
    prods_lr_linear = [p for p in grammar.productions if p.is_lr_linear]
    prods_binary = [p for p in grammar.productions if p.is_binary]

    match_at = _precompute_term_col_match(col_types, col_x, col_y)

    # Initialize: O[start, 0, L] = 0 (log 1)
    log_O[(grammar.start, 0, L)] = 0.0
    _outside_unary_close(0, L)

    # Fill by decreasing span
    for span in range(L, -1, -1):
        for i in range(L - span + 1):
            j = i + span

            _outside_unary_close(i, j)

            for p in prods_right_linear:
                # A -> t B: parent span [i, j], child B has span [i+1, j]
                # Contributes to O[B, i+1, j] from O[A, i, j]
                t_idx = p.rhs[0]
                B = p.rhs[1]
                if i < L and t_idx in match_at[i]:
                    o_val = get_O(p.lhs, i, j)
                    if o_val > NEG_INF:
                        log_w = np.log(max(p.weight, 1e-300))
                        logaddexp_O(B, i + 1, j, o_val + log_w)

            for p in prods_left_linear:
                # A -> B t: parent span [i, j], child B has span [i, j-1]
                t_idx = p.rhs[1]
                B = p.rhs[0]
                if j > 0 and t_idx in match_at[j - 1]:
                    o_val = get_O(p.lhs, i, j)
                    if o_val > NEG_INF:
                        log_w = np.log(max(p.weight, 1e-300))
                        logaddexp_O(B, i, j - 1, o_val + log_w)

            for p in prods_lr_linear:
                # A -> tL B tR: parent [i,j], child B has [i+1, j-1]
                tL, B, tR = p.rhs
                if span >= 2 and tL in match_at[i] and tR in match_at[j-1]:
                    o_val = get_O(p.lhs, i, j)
                    if o_val > NEG_INF:
                        log_w = np.log(max(p.weight, 1e-300))
                        logaddexp_O(B, i + 1, j - 1, o_val + log_w)

            for p in prods_binary:
                # A -> B C: parent [i,j], children [i,k] and [k,j]
                B, C = p.rhs
                o_val = get_O(p.lhs, i, j)
                if o_val <= NEG_INF:
                    continue
                log_w = np.log(max(p.weight, 1e-300))
                for k in range(i, j + 1):
                    # O[B, i, k] += O[A, i, j] * w * I[C, k, j]
                    c_val = get_In(C, k, j)
                    if c_val > NEG_INF:
                        logaddexp_O(B, i, k, o_val + log_w + c_val)
                    # O[C, k, j] += O[A, i, j] * w * I[B, i, k]
                    b_val = get_In(B, i, k)
                    if b_val > NEG_INF:
                        logaddexp_O(C, k, j, o_val + log_w + b_val)

    # Final unary close on all spans
    for span in range(L, -1, -1):
        for i in range(L - span + 1):
            j = i + span
            _outside_unary_close(i, j)

    return log_O


def tkfst_pair_expected_counts_aligned(grammar, col_types, col_x, col_y,
                                       log_I, log_O):
    """Alignment-constrained expected rule counts from Inside-Outside.

    Args:
        grammar: WCFG
        col_types, col_x, col_y: from alignment_to_columns()
        log_I: Inside table from tkfst_pair_inside_aligned
        log_O: Outside table from tkfst_pair_outside_aligned

    Returns:
        counts: list of expected counts, one per production
    """
    L = len(col_types)
    N = grammar.n_nonterminals
    NEG_INF = -1e30

    log_Z = log_I.get((grammar.start, 0, L), NEG_INF)
    if log_Z <= NEG_INF:
        return [0.0] * len(grammar.productions)

    def get_In(a, i, j):
        return log_I.get((a, i, j), NEG_INF)

    def get_O(a, i, j):
        return log_O.get((a, i, j), NEG_INF)

    match_at = _precompute_term_col_match(col_types, col_x, col_y)
    counts = [0.0] * len(grammar.productions)

    for span in range(L + 1):
        for i in range(L - span + 1):
            j = i + span

            for pi, p in enumerate(grammar.productions):
                o_val = get_O(p.lhs, i, j)
                if o_val <= NEG_INF:
                    continue
                log_w = np.log(max(p.weight, 1e-300))

                if p.is_empty:
                    if span == 0:
                        counts[pi] += np.exp(o_val + log_w - log_Z)

                elif p.is_terminal:
                    if span == 1:
                        t = p.rhs[0]
                        if t in match_at[i]:
                            counts[pi] += np.exp(o_val + log_w - log_Z)

                elif p.is_unary:
                    B = p.rhs[0]
                    i_val = get_In(B, i, j)
                    if i_val > NEG_INF:
                        counts[pi] += np.exp(
                            o_val + log_w + i_val - log_Z)

                elif p.is_right_linear:
                    t_idx = p.rhs[0]
                    B = p.rhs[1]
                    if span >= 1 and t_idx in match_at[i]:
                        i_val = get_In(B, i + 1, j)
                        if i_val > NEG_INF:
                            counts[pi] += np.exp(
                                o_val + log_w + i_val - log_Z)

                elif p.is_left_linear:
                    t_idx = p.rhs[1]
                    B = p.rhs[0]
                    if span >= 1 and t_idx in match_at[j - 1]:
                        i_val = get_In(B, i, j - 1)
                        if i_val > NEG_INF:
                            counts[pi] += np.exp(
                                o_val + log_w + i_val - log_Z)

                elif p.is_lr_linear:
                    tL, B, tR = p.rhs
                    if (span >= 2 and tL in match_at[i]
                            and tR in match_at[j - 1]):
                        i_val = get_In(B, i + 1, j - 1)
                        if i_val > NEG_INF:
                            counts[pi] += np.exp(
                                o_val + log_w + i_val - log_Z)

                elif p.is_binary:
                    B, C = p.rhs
                    for k in range(i, j + 1):
                        b_val = get_In(B, i, k)
                        c_val = get_In(C, k, j)
                        if b_val > NEG_INF and c_val > NEG_INF:
                            counts[pi] += np.exp(
                                o_val + log_w + b_val + c_val - log_Z)

    return counts


def encode_pair_sequences(x_seq_str, y_seq_str, x_struct, y_struct):
    """Encode ancestor/descendant RNA sequences for pair SCFG Inside.

    Converts string sequences to integer arrays.

    Args:
        x_seq_str: ancestor RNA sequence string
        y_seq_str: descendant RNA sequence string
        x_struct: ancestor dot-bracket structure (determines L/R assignment)
        y_struct: descendant dot-bracket structure

    Returns:
        x_seq: np.int32 array of ancestor nucleotide indices
        y_seq: np.int32 array of descendant nucleotide indices
    """
    nuc_map = {c: i for i, c in enumerate(NUC_CHARS)}
    x_seq = np.array([nuc_map.get(c.upper(), 0) for c in x_seq_str],
                     dtype=np.int32)
    y_seq = np.array([nuc_map.get(c.upper(), 0) for c in y_seq_str],
                     dtype=np.int32)
    return x_seq, y_seq
