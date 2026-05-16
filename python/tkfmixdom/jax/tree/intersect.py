"""Intersection of sibling recognizer profiles.

Given two sibling branches, each with a WPTT×Recognizer composite,
intersects them to produce a parent recognizer. The intersection
ensures both siblings accept the same input tokens simultaneously.

Synchronization rules (from ParseTreeTransducers.md):
  - If Sibling2 isn't in a ready state, Sibling1 can't update
  - If both are in a ready state, both make a simultaneous transition
    accepting the same input token (or simultaneous bifurcations, or
    simultaneous epsilon productions)

The intersected states are (sib1_state, sib2_state) tuples, where each
sib_state is a (wptt_st, rec_st) pair from the WPTT×Recognizer composition.
"""

import numpy as np
from ..distill.wptt import (
    WPTT, is_ready_state, IDX_START as WPTT_START,
)
from .recognizer import Recognizer, RecognizerState, RecognizerRule
from ..models.rna_grammar import (
    left_terminal, right_terminal, pair_terminal, decode_terminal,
    N_NUC, N_TOTAL_TERMINALS,
)


def intersect_wptt_recognizers(
        trans1, rec1, seq1,
        trans2, rec2, seq2):
    """Intersect two WPTT×Recognizer compositions at a parent node.

    Both compositions share the same WPTT (same evolutionary model),
    but have different recognizers (different leaf sequences).

    The intersection produces a parent recognizer where states are
    tuples of composed states from each sibling. The parent recognizer
    accepts input tokens that both siblings can process simultaneously.

    For simplicity and efficiency, this function computes the
    intersection via a factored Inside algorithm over the combined
    state space, returning the log probability and Inside table.

    Args:
        trans1: WPTT for sibling 1
        rec1: Recognizer for sibling 1's leaf
        seq1: leaf sequence for sibling 1
        trans2: WPTT for sibling 2
        rec2: Recognizer for sibling 2's leaf
        seq2: leaf sequence for sibling 2

    Returns:
        log_prob: log probability of the intersection
        log_I: dict mapping state tuples to log probabilities
    """
    # For now, implement a simplified version that just verifies
    # both children can produce valid parses and computes the
    # combined probability.
    #
    # The full intersection would enumerate (wptt1_st, rec1_st, wptt2_st, rec2_st)
    # state tuples, which is very large. We need the factored DP approach.
    #
    # Since both siblings share the same generator (order-1 SCFG) input,
    # the intersection enforces that the generator produces tokens
    # consumed simultaneously by both WPTTs.

    from .compose_wptt_rec import compose_wptt_recognizer

    # Compute each sibling's WPTT×Recognizer probability independently
    lp1, table1 = compose_wptt_recognizer(trans1, rec1, seq1)
    lp2, table2 = compose_wptt_recognizer(trans2, rec2, seq2)

    # The intersection probability is approximately the product
    # (sum in log space) of the two sibling probabilities,
    # but this is only exact when the WPTT states are independent.
    # For the full factored intersection, we'd need to jointly enumerate.
    log_prob = lp1 + lp2

    return log_prob, {'sib1': table1, 'sib2': table2}


def intersect_factored(
        transducer, rec1, seq1, rec2, seq2):
    """Full factored intersection of two WPTT×Recognizer siblings.

    Both siblings use the same WPTT (shared evolutionary model).
    The intersection state space is:
        (wptt_st_1, rec_st_1, wptt_st_2, rec_st_2)

    Since both WPTTs must accept the same generator input simultaneously,
    and both children of a bifurcation start from WPTT_START, we only
    need states where:
        - Both WPTTs start from WPTT_START (after bifurcation)
        - Generator input tokens are consumed by both WPTTs

    For now, we implement the approximate version. The full factored
    version will be needed for accurate progressive reconstruction.

    Args:
        transducer: WPTT object (shared between siblings)
        rec1: Recognizer for left sibling
        seq1: left sibling's leaf sequence
        rec2: Recognizer for right sibling
        seq2: right sibling's leaf sequence

    Returns:
        log_prob: approximate log probability
        info: dict with sibling tables
    """
    return intersect_wptt_recognizers(
        transducer, rec1, seq1,
        transducer, rec2, seq2)
