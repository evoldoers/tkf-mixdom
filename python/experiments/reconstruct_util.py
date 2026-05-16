"""Utility module for held-out leaf reconstruction at the correct node.

The bug in existing benchmarks: they reconstruct at the ROOT of the tree,
but should reconstruct at the internal node where the held-out leaf was
attached (the target's parent). When the held-out leaf is not directly
off the root (~65% of cases), root reconstruction gives the wrong answer.

Correct approach:
  1. Build tree from full MSA (target included)
  2. Find target leaf and its parent
  3. Re-root tree at target's parent
  4. Remove target leaf (now a child of the new root)
  5. Run Felsenstein at the new root = the original parent node
  6. Use Fitch parsimony for gap pattern at the same node

This module provides `reconstruct_held_out_felsenstein` which implements
this pipeline end-to-end.
"""

import os
import copy
import time
import shutil
import tempfile
import subprocess
import numpy as np


def _find_fasttree():
    """Locate FastTree binary."""
    ft = shutil.which('FastTree')
    if ft:
        return ft
    home_bin = os.path.expanduser('~/bin/FastTree')
    if os.path.isfile(home_bin) and os.access(home_bin, os.X_OK):
        return home_bin
    raise FileNotFoundError(
        "FastTree not found. Install it or place it in ~/bin/FastTree")


def _find_leaf(tree, name):
    """Find a leaf node by name. Returns None if not found."""
    for node in tree.preorder():
        if node.is_leaf and node.name == name:
            return node
    return None


def _find_node_by_id(tree, target_id):
    """Find a node by python id in a tree."""
    for node in tree.preorder():
        if id(node) == target_id:
            return node
    return None


def reroot_at_node(tree, target_node):
    """Re-root a tree at a given internal node.

    The target_node must be an internal node (not a leaf) in the tree.
    This reverses parent-child relationships along the path from root
    to target_node, making target_node the new root.

    Operates on a deep copy; the original tree is not modified.

    Args:
        tree: TreeNode root of the original tree
        target_node: the node (by name) to become the new root

    Returns:
        new_root: TreeNode that is the re-rooted tree
    """
    from tkfmixdom.jax.util.io import TreeNode

    # Deep copy to avoid mutating the original
    new_tree = copy.deepcopy(tree)

    # Find the target in the copied tree by name match
    target = None
    for node in new_tree.preorder():
        if node.name == target_node.name:
            target = node
            break
    if target is None:
        raise ValueError(f"Node '{target_node.name}' not found in tree")

    if target.parent is None:
        # Already root
        return new_tree

    # Collect path from target to root
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = current.parent

    # path = [target, parent, grandparent, ..., root]
    # Reverse the parent-child links along this path
    for i in range(len(path) - 1):
        child_node = path[i]
        parent_node = path[i + 1]

        # Remove child_node from parent_node's children
        parent_node.children = [c for c in parent_node.children
                                if c is not child_node]

        # Add parent_node as a child of child_node
        # The branch length stays with the edge (which was parent->child,
        # now becomes child->parent), so parent_node gets child_node's
        # original branch_length
        parent_node.branch_length = child_node.branch_length
        child_node.children.append(parent_node)
        parent_node.parent = child_node

    # The target is now root
    target.parent = None
    target.branch_length = 0.0

    return target


def _remove_child_leaf(root, leaf_name):
    """Remove a direct child leaf from root. Returns the modified tree.

    Only removes leaves that are direct children of root.
    If the removed leaf was the only child, returns None.
    """
    root.children = [c for c in root.children
                     if not (c.is_leaf and c.name == leaf_name)]
    return root if root.children else None


def _name_internal_nodes(tree):
    """Assign names to unnamed internal nodes."""
    counter = 0
    for node in tree.preorder():
        if node.name is None or node.name == '':
            node.name = f'_int_{counter}'
            counter += 1


def reconstruct_held_out_felsenstein(msa_strings, target_name, Q=None, pi=None):
    """Reconstruct a held-out leaf at its parent node using Felsenstein.

    Args:
        msa_strings: dict {name: aligned_string} -- ALL sequences including
            target, aligned (with gaps as '-')
        target_name: name of the held-out sequence
        Q: (20, 20) substitution rate matrix (default: LG08)
        pi: (20,) equilibrium frequencies (default: LG08)

    Returns:
        pred_seq: integer array of predicted residues (gap-stripped)
        elapsed: wall time in seconds
    """
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT, write_fasta
    from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
    from tkfmixdom.jax.tree.tree_varanc import infer_internal_presence

    t0 = time.time()

    if Q is None or pi is None:
        Q_lg, pi_lg = rate_matrix_lg()
        if Q is None:
            Q = np.asarray(Q_lg)
        if pi is None:
            pi = np.asarray(pi_lg)

    if target_name not in msa_strings:
        raise ValueError(f"Target '{target_name}' not found in MSA")

    # --- Step 1: Write MSA to temp FASTA and run FastTree ---
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        tmpfa = f.name
        for name, seq in msa_strings.items():
            f.write(f">{name}\n{seq}\n")

    try:
        fasttree = _find_fasttree()
        result = subprocess.run(
            [fasttree, '-quiet', '-lg', '-nosupport'],
            stdin=open(tmpfa),
            capture_output=True, text=True, timeout=120)
        newick_str = result.stdout.strip()
        if not newick_str:
            raise RuntimeError(
                f"FastTree produced no output. stderr: {result.stderr[:500]}")
    finally:
        os.unlink(tmpfa)

    # --- Step 2: Parse tree and find target ---
    tree = parse_newick(newick_str)
    _name_internal_nodes(tree)

    target_leaf = _find_leaf(tree, target_name)
    if target_leaf is None:
        raise ValueError(
            f"Target '{target_name}' not found in FastTree output")

    parent = target_leaf.parent
    if parent is None:
        raise ValueError(
            f"Target '{target_name}' is the root (single-sequence tree?)")

    # --- Step 3: Re-root tree at target's parent ---
    # Name the parent if needed (already done by _name_internal_nodes)
    rerooted = reroot_at_node(tree, parent)

    # --- Step 4: Remove target leaf (now a direct child of the new root) ---
    _remove_child_leaf(rerooted, target_name)

    # Handle degenerate case: if root now has a single child, it's still
    # a valid tree for Felsenstein (unary internal node is fine)

    # --- Step 5: Build leaf observations and run Felsenstein ---
    # Convert aligned strings to integer arrays (gaps as -1)
    L = len(next(iter(msa_strings.values())))
    leaf_seqs_aligned = {}
    for name, seq in msa_strings.items():
        if name == target_name:
            continue
        arr = np.full(L, -1, dtype=np.int32)
        for col, ch in enumerate(seq):
            if ch != '-' and ch != '.' and ch != '~':
                idx = AA_TO_INT.get(ch.upper(), -1)
                if idx >= 0:
                    arr[col] = idx
        leaf_seqs_aligned[name] = arr

    ancestor, posteriors = marginal_ancestor_all_columns_jax(
        rerooted, leaf_seqs_aligned, Q, pi)

    # --- Step 6: Fitch parsimony for gap pattern at the reconstruction node ---
    # Build leaf presence (True = residue present, False = gap)
    leaf_presence = {}
    for name, seq in msa_strings.items():
        if name == target_name:
            continue
        pres = np.array([ch not in '-.~' for ch in seq], dtype=bool)
        leaf_presence[name] = pres

    # Need all internal nodes named for infer_internal_presence
    _name_internal_nodes(rerooted)
    presence = infer_internal_presence(rerooted, leaf_presence)

    # The root of the rerooted tree IS the target's parent node.
    # Get gap pattern at root.
    root_name = rerooted.name
    if root_name in presence:
        root_presence = presence[root_name]
    else:
        # Fallback: if root not in presence dict, use union of leaves
        root_presence = np.zeros(L, dtype=bool)
        for p in leaf_presence.values():
            root_presence |= p

    # Build predicted sequence: MAP residue where present, skip gaps
    pred_list = []
    for col in range(L):
        if root_presence[col]:
            pred_list.append(int(ancestor[col]) if ancestor[col] >= 0
                             else int(np.argmax(posteriors[col])))
        # else: gap at this node, skip
    pred_seq = np.array(pred_list, dtype=np.int32)

    elapsed = time.time() - t0
    return pred_seq, elapsed


def reconstruct_held_out_partition(msa_strings, target_name, model=None,
                                   config=None):
    """Reconstruct a held-out leaf using partition reconstruction.

    NOTE: The partition reconstruction adapter (`partition_recon_adapter.py`)
    uses `prune_leaf` which collapses the target's parent node. This means
    it reconstructs at the ROOT of the pruned tree, not at the target's
    parent. This function wraps the partition adapter with the same
    re-rooting fix applied in `reconstruct_held_out_felsenstein`.

    Args:
        msa_strings: dict {name: aligned_string} -- ALL sequences including
            target, aligned (with gaps as '-')
        target_name: name of the held-out sequence
        model: PartitionReconModel (default: single-domain TKF92 + LG08)
        config: PartitionReconConfig (default: use_jax=True)

    Returns:
        pred_seq: integer array of predicted residues (gap-stripped)
        elapsed: wall time in seconds
    """
    from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT, write_fasta
    from tkfmixdom.jax.tree.tree_varanc import infer_internal_presence
    from experiments.partition_recon_adapter import (
        PartitionReconConfig, default_single_domain_model, build_inputs,
        partition_recon_forward_backward_jax,
        partition_recon_forward_backward,
    )
    from tkfmixdom.jax.tree.tree_varanc import name_internal_nodes

    t0 = time.time()

    if config is None:
        config = PartitionReconConfig()
    if model is None:
        model = default_single_domain_model(kappa_top=config.kappa_top)

    if target_name not in msa_strings:
        raise ValueError(f"Target '{target_name}' not found in MSA")

    # --- Build tree from full MSA ---
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        tmpfa = f.name
        for name, seq in msa_strings.items():
            f.write(f">{name}\n{seq}\n")

    try:
        fasttree = _find_fasttree()
        result = subprocess.run(
            [fasttree, '-quiet', '-lg', '-nosupport'],
            stdin=open(tmpfa),
            capture_output=True, text=True, timeout=120)
        newick_str = result.stdout.strip()
        if not newick_str:
            raise RuntimeError(
                f"FastTree produced no output. stderr: {result.stderr[:500]}")
    finally:
        os.unlink(tmpfa)

    tree = parse_newick(newick_str)
    _name_internal_nodes(tree)

    target_leaf = _find_leaf(tree, target_name)
    if target_leaf is None:
        raise ValueError(
            f"Target '{target_name}' not found in FastTree output")

    parent = target_leaf.parent
    if parent is None:
        raise ValueError(
            f"Target '{target_name}' is the root (single-sequence tree?)")

    # Re-root at target's parent
    rerooted = reroot_at_node(tree, parent)
    _remove_child_leaf(rerooted, target_name)
    name_internal_nodes(rerooted)

    # Build MSA dict (integer arrays, -1 for gaps)
    L = len(next(iter(msa_strings.values())))
    msa_int = {}
    for name, seq in msa_strings.items():
        if name == target_name:
            continue
        arr = np.full(L, -1, dtype=np.int32)
        for col, ch in enumerate(seq):
            if ch != '-' and ch != '.' and ch != '~':
                idx = AA_TO_INT.get(ch.upper(), -1)
                if idx >= 0:
                    arr[col] = idx
        msa_int[name] = arr

    # Run partition reconstruction on the re-rooted tree
    inputs = build_inputs(rerooted, msa_int)

    if config.use_jax:
        recon_result = partition_recon_forward_backward_jax(inputs, model)
    else:
        recon_result = partition_recon_forward_backward(inputs, model)

    root_map = recon_result.root_residue_map
    pred_seq = np.array(
        [int(root_map[c]) for c in range(len(root_map)) if root_map[c] >= 0],
        dtype=np.int32)

    elapsed = time.time() - t0
    return pred_seq, elapsed
