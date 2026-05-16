"""Order-1 Single SCFG for RNA secondary structure.

Implements the 10-nonterminal order-1 singlet SCFG described in
ParseTreeTransducers.md. This grammar captures adjacency correlations
from the full TKFStack model via coarse-grained 7-class context.

Nonterminals (10):
  - Start: initial state, context unknown
  - L: emits a single base to the left (context becomes NN)
  - R: emits a single base to the right (context becomes NN)
  - LR_AU, LR_CG, LR_GC, LR_UA, LR_GU, LR_UG, LR_NN:
    emit a base pair (left + right), context set to pair class

Each nonterminal has the same rule template:
  A -> X B         (L-emit: right-linear, next state B)
  A -> B Y         (R-emit: left-linear, next state B)
  A -> X B Y       (LR-emit: lr-linear, next state B)
  A -> Start Start (bifurcation: binary, resets context)
  A -> epsilon     (end)

The key insight is that context flows through the nonterminal names:
after emitting an LR pair (a,b), the next nonterminal is LR_classify(a,b).
After emitting L or R (single base), the next nonterminal is L or R.
Bifurcations reset both children to Start.

Terminal encoding (matching rna_grammar.py):
  Pair (a,b): a*4 + b  (indices 0-15)
  Left a:     16 + a    (indices 16-19)
  Right a:    20 + a    (indices 20-23)
"""

import numpy as np
from ..grammar.scfg import WCFG, Production, build_grammar
from ..core.rna import (
    classify_basepair, N_CONTEXT, N_NUC, CTX_NN, CONTEXT_NAMES,
)
from .rna_grammar import (
    N_TOTAL_TERMINALS, pair_terminal, left_terminal, right_terminal,
)


# Nonterminal names
_NT_NAMES = ['Start', 'L', 'R',
             'LR_AU', 'LR_CG', 'LR_GC', 'LR_UA', 'LR_GU', 'LR_UG', 'LR_NN']
N_SINGLET_STATES = len(_NT_NAMES)  # 10

# State indices
IDX_START = 0
IDX_L = 1
IDX_R = 2
IDX_LR_BASE = 3  # LR_AU=3, LR_CG=4, ..., LR_NN=9

def _lr_nt_name(ctx):
    """Get LR nonterminal name for a context class."""
    return f'LR_{CONTEXT_NAMES[ctx]}'

def _lr_nt_index(ctx):
    """Get LR nonterminal index for a context class."""
    return IDX_LR_BASE + ctx


def build_order1_singlet_scfg(weights=None):
    """Build the 10-nonterminal order-1 singlet SCFG.

    Args:
        weights: optional dict specifying production weights. If None,
                 uses uniform defaults. Keys are tuples:
                   ('L', src_nt_idx, nuc) -> weight for L-emit of nucleotide `nuc`
                   ('R', src_nt_idx, nuc) -> weight for R-emit
                   ('LR', src_nt_idx, left_nuc, right_nuc) -> weight for LR-emit
                   ('bif', src_nt_idx) -> weight for bifurcation
                   ('eps', src_nt_idx) -> weight for epsilon

    Returns:
        WCFG with 10 nonterminals, 24 terminals, and the full rule set.
    """
    rules = []

    for src_idx, src_name in enumerate(_NT_NAMES):
        # Collect all outgoing rules for this nonterminal to normalize later
        raw_rules = []

        # L-emit: A -> X L  (right-linear, for each nucleotide X)
        # After L-emit, context becomes NN, so next state is L
        for nuc in range(N_NUC):
            t = left_terminal(nuc)
            key = ('L', src_idx, nuc)
            w = weights.get(key, 1.0) if weights else 1.0
            raw_rules.append((src_name, [(t, 'T'), ('L', 'N')], w, key))

        # R-emit: A -> R Y  (left-linear, for each nucleotide Y)
        # After R-emit, context becomes NN, so next state is R
        for nuc in range(N_NUC):
            t = right_terminal(nuc)
            key = ('R', src_idx, nuc)
            w = weights.get(key, 1.0) if weights else 1.0
            raw_rules.append((src_name, [('R', 'N'), (t, 'T')], w, key))

        # LR-emit: A -> X LR_ctx Y  (lr-linear, for each pair X,Y)
        # After LR-emit of (X,Y), context is classify(X,Y)
        for left_nuc in range(N_NUC):
            for right_nuc in range(N_NUC):
                t_left = left_terminal(left_nuc)
                t_right = right_terminal(right_nuc)
                ctx = classify_basepair(left_nuc, right_nuc)
                next_nt = _lr_nt_name(ctx)
                key = ('LR', src_idx, left_nuc, right_nuc)
                w = weights.get(key, 1.0) if weights else 1.0
                raw_rules.append((src_name,
                                  [(t_left, 'T'), (next_nt, 'N'), (t_right, 'T')],
                                  w, key))

        # Bifurcation: A -> Start Start  (resets context)
        key = ('bif', src_idx)
        w = weights.get(key, 1.0) if weights else 1.0
        raw_rules.append((src_name, [('Start', 'N'), ('Start', 'N')], w, key))

        # Epsilon: A -> epsilon
        key = ('eps', src_idx)
        w = weights.get(key, 1.0) if weights else 1.0
        raw_rules.append((src_name, [], w, key))

        # Normalize weights for this nonterminal
        total_w = sum(r[2] for r in raw_rules)
        if total_w > 0:
            for lhs, rhs_spec, w, key in raw_rules:
                rules.append((lhs, rhs_spec, w / total_w))

    return build_grammar(_NT_NAMES, N_TOTAL_TERMINALS, rules, start='Start')


def singlet_scfg_rule_index(grammar):
    """Build an index mapping (rule_type, src_nt, ...) to production indices.

    Returns:
        dict mapping key tuples to production indices in grammar.productions
    """
    index = {}
    pi = 0
    for src_idx in range(N_SINGLET_STATES):
        # L-emit: 4 rules
        for nuc in range(N_NUC):
            index[('L', src_idx, nuc)] = pi
            pi += 1
        # R-emit: 4 rules
        for nuc in range(N_NUC):
            index[('R', src_idx, nuc)] = pi
            pi += 1
        # LR-emit: 16 rules
        for left_nuc in range(N_NUC):
            for right_nuc in range(N_NUC):
                index[('LR', src_idx, left_nuc, right_nuc)] = pi
                pi += 1
        # Bifurcation: 1 rule
        index[('bif', src_idx)] = pi
        pi += 1
        # Epsilon: 1 rule
        index[('eps', src_idx)] = pi
        pi += 1
    return index


def count_rules():
    """Return the expected number of production rules.

    Per nonterminal: 4 L-emit + 4 R-emit + 16 LR-emit + 1 bif + 1 eps = 26
    Total: 10 * 26 = 260
    """
    return N_SINGLET_STATES * (N_NUC + N_NUC + N_NUC * N_NUC + 1 + 1)
