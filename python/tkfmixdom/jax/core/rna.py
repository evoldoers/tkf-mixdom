"""Coarse-grained base pair context for order-1 RNA SCFGs and WPTTs.

Defines 7 context equivalence classes for the order-1 distillation:
  - 6 canonical (Watson-Crick + wobble) base pairs: AU, CG, GC, UA, GU, UG
  - 1 catch-all class NN: non-canonical pairs, unpaired bases, boundaries

Context tracks the most recently emitted LR pair. Emitting a single L or R
base (unpaired) resets context to NN. Sequence boundaries are also NN.
"""

import numpy as np

# Nucleotide indices (matching rna_grammar.py)
_A, _C, _G, _U = 0, 1, 2, 3
N_NUC = 4

# Context class indices
CTX_AU = 0
CTX_CG = 1
CTX_GC = 2
CTX_UA = 3
CTX_GU = 4
CTX_UG = 5
CTX_NN = 6
N_CONTEXT = 7

CONTEXT_NAMES = ['AU', 'CG', 'GC', 'UA', 'GU', 'UG', 'NN']

# Lookup table: (left_nuc, right_nuc) -> context class
_PAIR_TO_CTX = np.full((N_NUC, N_NUC), CTX_NN, dtype=np.int32)
_PAIR_TO_CTX[_A, _U] = CTX_AU
_PAIR_TO_CTX[_C, _G] = CTX_CG
_PAIR_TO_CTX[_G, _C] = CTX_GC
_PAIR_TO_CTX[_U, _A] = CTX_UA
_PAIR_TO_CTX[_G, _U] = CTX_GU
_PAIR_TO_CTX[_U, _G] = CTX_UG


def classify_basepair(left_nuc, right_nuc):
    """Classify a base pair into one of 7 context classes.

    Args:
        left_nuc: left nucleotide index (0-3 for A,C,G,U)
        right_nuc: right nucleotide index (0-3 for A,C,G,U)

    Returns:
        Context class index (0-6). Returns CTX_NN for non-canonical pairs.
    """
    if 0 <= left_nuc < N_NUC and 0 <= right_nuc < N_NUC:
        return int(_PAIR_TO_CTX[left_nuc, right_nuc])
    return CTX_NN


def is_canonical(left_nuc, right_nuc):
    """Check if a base pair is canonical (Watson-Crick or wobble)."""
    return classify_basepair(left_nuc, right_nuc) != CTX_NN


def context_name(ctx):
    """Return human-readable name for a context class."""
    return CONTEXT_NAMES[ctx]


def pair_for_context(ctx):
    """Return (left_nuc, right_nuc) for a canonical context class.

    Args:
        ctx: context class index (0-5 for canonical, 6 for NN)

    Returns:
        (left_nuc, right_nuc) tuple, or None for CTX_NN.
    """
    if ctx == CTX_NN:
        return None
    _pairs = [(_A, _U), (_C, _G), (_G, _C), (_U, _A), (_G, _U), (_U, _G)]
    return _pairs[ctx]
