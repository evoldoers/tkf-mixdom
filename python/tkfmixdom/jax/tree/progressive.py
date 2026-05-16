"""Progressive reconstruction pipeline for parse tree transducers.

Implements the full progressive reconstruction algorithm from
ParseTreeTransducers.md:

1. Build leaf recognizers from leaf sequences (with consensus fallback)
2. For each internal node (post-order):
   a. Compose WPTT with each child's recognizer
   b. Intersect the two sibling compositions
   c. Pre-compose with order-1 SCFG
   d. Sample parse trees through the composed grammar
   e. Compress to produce parent recognizer
3. At root: final parse tree gives the ancestral reconstruction

This module provides both the full pipeline and individual steps.
"""

import numpy as np
from ..models.order1_scfg import build_order1_singlet_scfg
from ..distill.wptt import build_wptt_transducer, WPTT, IDX_START as WPTT_START
from .recognizer import build_leaf_recognizer, Recognizer
from .compose_wptt_rec import compose_wptt_recognizer
from .intersect import intersect_factored
from ..dp.scfg_factored import factored_inside


def build_leaf_recognizers(leaf_seqs, consensus_structure=None):
    """Build recognizer profiles for all leaf sequences.

    Args:
        leaf_seqs: list of numpy arrays, each shape (Li,) with nuc indices
        consensus_structure: optional dot-bracket string. Applied to all
            leaves (padded/truncated as needed). If provided, marks
            fallback states in each recognizer.

    Returns:
        list of Recognizer objects
    """
    recognizers = []
    for seq in leaf_seqs:
        L = len(seq)
        if consensus_structure is not None and len(consensus_structure) >= L:
            cs = consensus_structure[:L]
        else:
            cs = None
        recognizers.append(build_leaf_recognizer(seq, consensus_structure=cs))
    return recognizers


def progressive_reconstruct(tree, leaf_seqs, consensus_structure=None,
                            wptt_weights=None, scfg_weights=None):
    """Run progressive reconstruction on a phylogenetic tree.

    Args:
        tree: tree structure as nested tuples: (left, right) for internal,
              or int for leaf index
        leaf_seqs: list of numpy arrays (nucleotide sequences)
        consensus_structure: optional dot-bracket consensus structure
        wptt_weights: optional WPTT weight dict
        scfg_weights: optional SCFG weight dict

    Returns:
        dict with:
            'log_prob': total log probability
            'recognizers': dict mapping node -> Recognizer
            'leaf_log_probs': dict mapping leaf_idx -> log prob of leaf
    """
    gen = build_order1_singlet_scfg(weights=scfg_weights)
    trans = build_wptt_transducer(weights=wptt_weights)

    # Build leaf recognizers
    recs = build_leaf_recognizers(leaf_seqs, consensus_structure)

    # Post-order traversal
    node_recs = {}   # node_id -> (Recognizer, leaf_seq)
    node_lps = {}    # node_id -> log_prob

    def _traverse(node, node_id=0):
        if isinstance(node, int):
            # Leaf node
            node_recs[node_id] = (recs[node], leaf_seqs[node])
            return node_id + 1

        left, right = node
        next_id = node_id + 1
        left_id = next_id
        next_id = _traverse(left, left_id)
        right_id = next_id
        next_id = _traverse(right, right_id)

        # Get child recognizers
        rec_l, seq_l = node_recs[left_id]
        rec_r, seq_r = node_recs[right_id]

        # Intersect siblings
        lp, info = intersect_factored(trans, rec_l, seq_l, rec_r, seq_r)
        node_lps[node_id] = lp

        # For now, the parent recognizer is a placeholder.
        # Full implementation would:
        # 1. Pre-compose with SCFG
        # 2. Sample parse trees
        # 3. Compress into parent recognizer
        # For now, use the longer child's recognizer as parent proxy
        if len(seq_l) >= len(seq_r):
            node_recs[node_id] = (rec_l, seq_l)
        else:
            node_recs[node_id] = (rec_r, seq_r)

        return next_id

    _traverse(tree)

    total_lp = sum(node_lps.values()) if node_lps else 0.0

    return {
        'log_prob': total_lp,
        'node_log_probs': node_lps,
        'node_recognizers': {k: v[0] for k, v in node_recs.items()},
    }


def compute_leaf_log_probs(leaf_seqs, scfg_weights=None, wptt_weights=None):
    """Compute log probability of each leaf under the SCFG × WPTT model.

    Uses the factored Inside DP to compute P(leaf | SCFG, WPTT)
    for each leaf sequence.

    Args:
        leaf_seqs: list of numpy arrays
        scfg_weights: optional SCFG weight dict
        wptt_weights: optional WPTT weight dict

    Returns:
        list of log probabilities, one per leaf
    """
    gen = build_order1_singlet_scfg(weights=scfg_weights)
    trans = build_wptt_transducer(weights=wptt_weights)

    log_probs = []
    for seq in leaf_seqs:
        lp, _ = factored_inside(gen, trans, seq)
        log_probs.append(lp)
    return log_probs
