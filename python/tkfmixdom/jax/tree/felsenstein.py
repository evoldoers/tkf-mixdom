"""Tree operations for phylogenetic inference.

Felsenstein pruning (substitution-only), tree likelihood with TKF91,
and utilities for working with phylogenetic trees.
"""

import jax.numpy as jnp
import numpy as np

from ..util.io import TreeNode, parse_newick, tree_to_adjacency
from ..core.ctmc import transition_matrix
from ..dp.hmm import forward_2d, safe_log
from ..core.params import tkf91_trans, S, M, I, D, E


def felsenstein_pruning(tree_root, leaf_seqs, Q, pi):
    """Felsenstein pruning algorithm for a single alignment column.

    Computes P(column | tree, Q, pi) for substitution-only model.
    Uses rescaling to prevent underflow on deep trees: at each internal
    node, the conditional likelihood vector is divided by its max element,
    and the log of the scale factor is accumulated.

    Args:
        tree_root: TreeNode (root of tree)
        leaf_seqs: dict of {leaf_name: character_index} for this column
        Q: rate matrix
        pi: equilibrium frequencies

    Returns:
        log_prob: float, log-likelihood of this column
    """
    n = Q.shape[0]

    def _prune(node):
        """Returns (cond_likelihood, log_scale) pair."""
        if node.is_leaf:
            char = leaf_seqs.get(node.name)
            if char is None or char < 0:
                return jnp.ones(n), 0.0  # missing data: uniform
            cond = jnp.zeros(n)
            cond = cond.at[char].set(1.0)
            return cond, 0.0

        partial = jnp.ones(n)
        log_scale = 0.0
        for child in node.children:
            child_cond, child_log_scale = _prune(child)
            M_t = transition_matrix(Q, child.branch_length)
            partial = partial * (M_t @ child_cond)
            log_scale += child_log_scale

        # Rescale to prevent underflow
        max_val = jnp.max(partial)
        max_val = jnp.maximum(max_val, 1e-300)
        partial = partial / max_val
        log_scale += jnp.log(max_val)

        return partial, log_scale

    root_cond, log_scale = _prune(tree_root)
    prob = jnp.sum(pi * root_cond)
    return float(log_scale + jnp.log(jnp.maximum(prob, 1e-300)))


def felsenstein_msa(tree_root, alignment, Q, pi, alphabet="protein"):
    """Felsenstein log-likelihood for an entire MSA (substitution-only).

    Args:
        tree_root: TreeNode
        alignment: dict of {leaf_name: aligned_sequence_string}
        Q: rate matrix
        pi: equilibrium frequencies
        alphabet: "protein" or "dna"

    Returns:
        total_log_prob: float
    """
    from ..util.io import AA_TO_INT, NT_TO_INT
    mapping = AA_TO_INT if alphabet == "protein" else NT_TO_INT

    names = list(alignment.keys())
    aln_len = len(next(iter(alignment.values())))

    total_lp = 0.0
    for col in range(aln_len):
        col_chars = {}
        all_gap = True
        for name in names:
            c = alignment[name][col]
            if c in "-." or c not in mapping:
                col_chars[name] = -1
            else:
                col_chars[name] = mapping[c]
                all_gap = False
        if all_gap:
            continue
        lp = felsenstein_pruning(tree_root, col_chars, Q, pi)
        total_lp += lp

    return total_lp


def tree_pairwise_logprob(x, y, ins_rate, del_rate, t, Q, pi):
    """Compute log P(x, y | TKF91 params) for a single edge.

    Wrapper around forward_2d for tree likelihood computation.
    """
    sub_matrix = transition_matrix(Q, t)
    tau = tkf91_trans(ins_rate, del_rate, t)
    log_trans = safe_log(tau)
    state_types = jnp.array([S, M, I, D, E])
    log_prob, _ = forward_2d(log_trans, state_types, x, y, sub_matrix, pi)
    return float(log_prob)


def tree_likelihood_pairwise(tree_root, leaf_seqs, ins_rate, del_rate, Q, pi):
    """Approximate tree likelihood as sum of pairwise edge likelihoods.

    For each edge (parent, child) in the tree, computes log P(x_parent, x_child)
    using the parent's sequence as ancestor. At the root, uses the stationary
    distribution. Leaves must have observed sequences.

    This is an approximation — the true tree likelihood requires the full
    multidimensional DP or progressive alignment approach.

    Args:
        tree_root: TreeNode
        leaf_seqs: dict of {leaf_name: integer_array}
        ins_rate, del_rate: indel rates
        Q: rate matrix
        pi: equilibrium frequencies

    Returns:
        total_log_prob: float (sum of edge log-likelihoods)
        edge_log_probs: dict of {(parent_name, child_name): log_prob}
    """
    # Assign sequences to leaves
    node_seqs = {}
    for node in tree_root.preorder():
        if node.is_leaf and node.name in leaf_seqs:
            node_seqs[id(node)] = jnp.asarray(leaf_seqs[node.name])

    # Bottom-up: for internal nodes, use the longer child sequence as proxy
    for node in tree_root.postorder():
        if id(node) not in node_seqs and not node.is_leaf:
            child_seqs = [node_seqs[id(c)] for c in node.children
                          if id(c) in node_seqs]
            if child_seqs:
                node_seqs[id(node)] = max(child_seqs, key=lambda s: len(s))

    total_lp = 0.0
    edge_lps = {}

    for node in tree_root.preorder():
        for child in node.children:
            if id(node) in node_seqs and id(child) in node_seqs:
                x = node_seqs[id(node)]
                y = node_seqs[id(child)]
                if len(x) > 0 and len(y) > 0:
                    t = max(child.branch_length, 1e-6)
                    lp = tree_pairwise_logprob(x, y, ins_rate, del_rate,
                                               t, Q, pi)
                    pname = node.name or f"node_{id(node)}"
                    cname = child.name or f"node_{id(child)}"
                    edge_lps[(pname, cname)] = lp
                    total_lp += lp

    return total_lp, edge_lps
