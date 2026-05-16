#!/usr/bin/env python3
"""Leaf-holdout presence/absence reconstruction benchmark.

For each entry in ``unified_benchmark_spec.json`` /
``unified_benchmark_long_spec.json``:
  1. Load the MSA + FastTree tree.
  2. Hold out the leaf specified in the spec.
  3. Predict per-column presence/absence for the held-out leaf with each
     method, using only the remaining leaves' clamps.
  4. Score predicted vs ground-truth presence with F1 / precision /
     recall (column-wise binary).

Methods:
  - varanc_presence: variational TKF92 (appendix L of tkf.tex).
    Tied per-edge variational conditionals (irreversible Felsenstein-3
    in the appendix's special-case nomenclature).
  - fitch: Fitch parsimony for the binary {present, absent} labelling.
  - fels21: Felsenstein-21 with gap as the 21st character; presence
    prediction = 1 - posterior(state = gap).

Per the project full-coverage rule (benchmark-curator), every method
runs on every entry; F1 is the headline metric; we save predicted
presence per column for re-scoring.

Usage:
  cd python && JAX_ENABLE_X64=1 uv run python -u \\
      experiments/varanc_presence_benchmark.py --dataset unified_short
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tkfmixdom.jax.tree.varanc_presence import (
    parse_binary_tree, edge_lookup, BinaryTree,
    make_q_conditionals, make_root_dist, leaf_clamp_to_beta,
    bp_pair_marginals, expected_branch_LL, entropy_per_column,
    elbo as elbo_fn, NYI, PRESENT, DELETED, N_Z,
    tkf92_wfst_log_T,
)
from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT
from experiments.ancrec_benchmark import parse_sto
from experiments.fels21_reconstruction_benchmark import (
    load_pfam_family, run_fels21,
)
from tkfmixdom.jax.tree.tree_varanc import name_internal_nodes


t0 = time.time()
def log(msg): print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Tree handling: convert generic newick TreeNode to BinaryTree, prune
# the held-out leaf, binarise polytomies.
# ---------------------------------------------------------------------------

def _binarise(tree_node):
    """In-place: replace every node with k>2 children by a left-comb of
    binary nodes with zero-length internal edges. Two-children nodes are
    left alone. Leaves are unchanged."""
    if not tree_node.children:
        return
    while len(tree_node.children) > 2:
        first = tree_node.children[0]
        rest = tree_node.children[1:]
        from tkfmixdom.jax.util.io import TreeNode
        new = TreeNode()
        new.children = rest
        new.branch_length = 0.0
        new.name = None
        tree_node.children = [first, new]
    for c in tree_node.children:
        _binarise(c)


def _to_newick(node):
    if not node.children:
        return f"{node.name or ''}:{node.branch_length:.6f}"
    inner = ",".join(_to_newick(c) for c in node.children)
    return f"({inner}){node.name or ''}:{node.branch_length:.6f}"


def build_binary_tree_from_node(root_node):
    """Walk a TreeNode root, binarise polytomies in-place, then build
    the BinaryTree static representation by serialising to newick and
    parsing with parse_binary_tree."""
    _binarise(root_node)
    return parse_binary_tree(_to_newick(root_node) + ";")


def _prune_leaf(node, name_to_remove):
    """Return a fresh tree with the named leaf removed (and any
    resulting unary internal node spliced out). Modifies a deep copy."""
    import copy as cp
    root = cp.deepcopy(node)

    # Find and remove the leaf.
    def visit(n):
        n.children = [c for c in n.children if not (
            (not c.children) and (c.name or "") == name_to_remove)]
        for c in n.children:
            visit(c)
    visit(root)

    # Splice out unary internal nodes.
    def splice(n):
        for i, c in enumerate(n.children):
            if c.children and len(c.children) == 1:
                gc = c.children[0]
                gc.branch_length += c.branch_length
                n.children[i] = gc
            splice(n.children[i])
    splice(root)

    # If root has one child, promote it.
    while len(root.children) == 1:
        c = root.children[0]
        c.branch_length = 0.0
        root = c

    return root


# ---------------------------------------------------------------------------
# VarAnc-presence prediction.
# ---------------------------------------------------------------------------

def _predict_one_holdout_varanc(binary_tree, leaf_present_remaining,
                                holdout_leaf_idx_in_full,
                                holdout_leaf_name, full_leaf_names,
                                ins_rate, del_rate, ext,
                                n_iter=150, lr=0.05, seed=0,
                                tie_edge_logits=True):
    """Run varanc-presence on the tree with the held-out leaf
    integrated as a leaf with uniform clamp. Returns predicted P(present)
    per column for the held-out leaf.

    Args:
        binary_tree: BinaryTree of the FULL tree (held-out leaf still in).
        leaf_present_remaining: (num_leaves, L) {0,1}; held-out leaf row
            will be overwritten with uniform clamp.
        holdout_leaf_idx_in_full: index of the held-out leaf in
            binary_tree.leaf_names.
        tie_edge_logits: True (default) → 2 free logits per edge,
            broadcast across columns.  False → 2 free logits per
            (edge, column), letting each column have its own
            transition q.
    """
    L = leaf_present_remaining.shape[1]
    le, re = edge_lookup(binary_tree)

    # Clip degenerate (zero or near-zero) branch lengths to keep the WFST
    # well-defined (alpha=1, beta=0 at t=0 gives log T entries of -inf).
    edge_lengths = np.maximum(np.asarray(binary_tree.edge_length), 1e-3)
    binary_tree = binary_tree._replace(edge_length=edge_lengths)

    # Leaf clamp: uniform on the held-out leaf so it carries no evidence;
    # observed elsewhere.
    leaf_clamp = np.array(leaf_clamp_to_beta(leaf_present_remaining))
    leaf_clamp[holdout_leaf_idx_in_full, :, :] = 1.0
    leaf_clamp = jnp.asarray(leaf_clamp)

    # Initialise edge logits + root logit from Fitch parsimony labels
    # (treating the held-out leaf as missing data). This seeds q close
    # to the Fitch reconstruction, from which Adam refines the ELBO.
    edge_logits, root_logit = fitch_seeded_init(
        binary_tree, leaf_present_remaining, holdout_leaf_idx_in_full)
    # If untied, broadcast the (E, 2) Fitch seed to (E, L, 2) so each
    # column starts with the same Fitch-seeded init but Adam can move
    # them independently.
    if not tie_edge_logits:
        edge_logits = jnp.broadcast_to(
            edge_logits[:, None, :], (binary_tree.num_edges, L, 2))
    # Add a small jitter so seed > 0 produces a different trajectory.
    rng = np.random.default_rng(seed)
    edge_logits = edge_logits + jnp.asarray(
        rng.standard_normal(edge_logits.shape) * 0.05, dtype=jnp.float64)

    # `elbo_fn` builds leaf_clamp internally from leaf_present.
    # The uniform-clamp on the held-out leaf needs a custom path that
    # consumes ``leaf_clamp_jnp`` directly:
    from tkfmixdom.jax.tree.varanc_presence import singlet_root_log_prior

    @jax.jit
    def loss_with_clamp(edge_logits, root_logit, leaf_clamp_jnp):
        if tie_edge_logits:
            logits = jnp.broadcast_to(
                edge_logits[:, None, :], (binary_tree.num_edges, L, 2))
        else:
            logits = edge_logits  # already (E, L, 2)
        root_logits = jnp.broadcast_to(root_logit, (L,))
        q_cond = make_q_conditionals(logits)
        root_dist = make_root_dist(root_logits)
        pair_marg, log_Z = bp_pair_marginals(
            q_cond, root_dist, leaf_clamp_jnp, binary_tree, le, re)
        edge_lens = jnp.asarray(binary_tree.edge_length)
        log_T_per_edge = jax.vmap(
            lambda t: tkf92_wfst_log_T(ins_rate, del_rate, t, ext))(edge_lens)
        branch_LLs = jax.vmap(expected_branch_LL)(pair_marg, log_T_per_edge)
        sum_branch_LL = jnp.sum(branch_LLs)
        H = entropy_per_column(
            pair_marg, root_dist, beta_root=None,
            node_marg_internal=None, q_cond=q_cond, tree=binary_tree)
        H_total = jnp.sum(H)
        # TKF92 stationary singlet root prior with L_root=0 special case.
        log_prior_root = singlet_root_log_prior(
            root_dist, ins_rate, del_rate, ext)
        sum_log_Z = jnp.sum(log_Z)
        return -(sum_branch_LL + log_prior_root + H_total + sum_log_Z)

    optimizer = optax.adam(lr)
    state = optimizer.init((edge_logits, root_logit))
    grad_fn = jax.jit(jax.grad(loss_with_clamp, argnums=(0, 1)))

    for step in range(n_iter):
        grads = grad_fn(edge_logits, root_logit, leaf_clamp)
        updates, state = optimizer.update(grads, state)
        edge_logits, root_logit = optax.apply_updates(
            (edge_logits, root_logit), updates)

    # Extract held-out leaf marginal from BP pair marginals.
    if tie_edge_logits:
        logits = jnp.broadcast_to(
            edge_logits[:, None, :], (binary_tree.num_edges, L, 2))
    else:
        logits = edge_logits
    root_logits = jnp.broadcast_to(root_logit, (L,))
    q_cond = make_q_conditionals(logits)
    root_dist = make_root_dist(root_logits)
    pair_marg, _ = bp_pair_marginals(
        q_cond, root_dist, leaf_clamp, binary_tree, le, re)

    holdout_node = binary_tree.num_internal + holdout_leaf_idx_in_full
    edge_to_holdout = None
    for e in range(binary_tree.num_edges):
        if int(binary_tree.edge_child[e]) == holdout_node:
            edge_to_holdout = e
            break
    assert edge_to_holdout is not None

    # P(leaf = PRESENT | column) = sum over parent state of pair_marg(parent, PRESENT)
    p_present = pair_marg[edge_to_holdout, :, :, PRESENT].sum(axis=-1)

    # Per-internal-node MAP profile for the model-probability selector.
    # For each internal v != root: marginal = sum over parent state of pair_marg
    # at any edge whose CHILD is v. For root: sum over child state at any edge
    # whose PARENT is root.
    n_internal = binary_tree.num_internal
    node_marg = np.zeros((n_internal, L, N_Z))
    edge_parent_np = np.asarray(binary_tree.edge_parent)
    edge_child_np = np.asarray(binary_tree.edge_child)
    for v in range(n_internal):
        # As-child marginal (sum over parent_state) for any edge into v
        as_child = np.where(edge_child_np == v)[0]
        as_parent = np.where(edge_parent_np == v)[0]
        if len(as_child):
            e = int(as_child[0])
            node_marg[v] = np.asarray(pair_marg[e]).sum(axis=-2)  # (L, 3) sum over parent
        elif len(as_parent):
            e = int(as_parent[0])
            node_marg[v] = np.asarray(pair_marg[e]).sum(axis=-1)  # (L, 3) sum over child
    # Per-node MAP: argmax over {NYI, P, D}, binary mask = (state == PRESENT)
    map_state = node_marg.argmax(axis=-1)            # (n_internal, L)
    internal_MAP_binary = (map_state == PRESENT).astype(np.int32)

    return np.asarray(p_present), internal_MAP_binary


def log_p_profile(binary_tree, internal_labels, leaf_states_binary,
                  ins_rate, del_rate, ext, eps=1e-30):
    """log p̃(MSA, internals) for a deterministic configuration.

    Args:
        internal_labels: (num_internal, L) binary 0/1 (1=Present at internal).
        leaf_states_binary: (num_leaves, L) binary 0/1 (1=Present at leaf).
        ins_rate, del_rate, ext: TKF92 params.

    Returns:
        scalar log probability = log p_singlet(root) + sum_branches log P_branch.
    """
    from tkfmixdom.jax.tree.varanc_presence import (
        tkf92_wfst_log_T, expected_branch_LL,
    )
    L = leaf_states_binary.shape[1]
    n_internal = binary_tree.num_internal

    # Convert binary -> 3-state {NYI, P, D} via top-down inheritance.
    state_int = np.zeros((n_internal, L), dtype=np.int32)
    parent_arr = np.asarray(binary_tree.parent)
    for v in binary_tree.preorder_internal:
        v = int(v)
        for n in range(L):
            if internal_labels[v, n] == 1:
                state_int[v, n] = 1  # P
            else:
                par = int(parent_arr[v])
                if par == -1:
                    state_int[v, n] = 0  # root absent -> NYI
                else:
                    par_st = state_int[par, n]
                    state_int[v, n] = 0 if par_st == 0 else 2

    state_leaf = np.zeros((binary_tree.num_leaves, L), dtype=np.int32)
    for li in range(binary_tree.num_leaves):
        node = binary_tree.num_internal + li
        par = int(parent_arr[node])
        for n in range(L):
            if leaf_states_binary[li, n] == 1:
                state_leaf[li, n] = 1
            else:
                par_st = state_int[par, n] if par >= 0 else 0
                state_leaf[li, n] = 0 if par_st == 0 else 2

    # Singlet root term.
    kappa = ins_rate / del_rate
    p = ext + (1.0 - ext) * kappa
    root = binary_tree.root
    L_root = int((state_int[root] == 1).sum())
    if L_root == 0:
        log_singlet = float(np.log(1.0 - kappa))
    else:
        log_singlet = float(
            np.log(kappa) + (L_root - 1) * np.log(p)
            + np.log((1.0 - ext) * (1.0 - kappa)))

    # Build deterministic per-branch pair_marg (one-hot at the realised states).
    n_edges = binary_tree.num_edges
    pair_marg_det = np.zeros((n_edges, L, N_Z, N_Z))
    for e in range(n_edges):
        p_idx = int(binary_tree.edge_parent[e])
        c_idx = int(binary_tree.edge_child[e])
        for n in range(L):
            ps = state_int[p_idx, n]
            cs = (state_int[c_idx, n] if c_idx < n_internal
                  else state_leaf[c_idx - n_internal, n])
            pair_marg_det[e, n, ps, cs] = 1.0
    pair_marg_det = jnp.asarray(pair_marg_det)

    # Per-branch log T.
    edge_lens = jnp.asarray(binary_tree.edge_length)
    log_T_per_edge = jax.vmap(
        lambda t: tkf92_wfst_log_T(ins_rate, del_rate, t, ext))(edge_lens)
    branch_LLs = jax.vmap(expected_branch_LL)(pair_marg_det, log_T_per_edge)
    return float(log_singlet) + float(jnp.sum(branch_LLs))


# ---------------------------------------------------------------------------
# Fitch parsimony for binary presence/absence.
# ---------------------------------------------------------------------------

def fitch_labels(binary_tree, leaf_present, missing_leaf_idx=None):
    """Run Fitch parsimony for binary {present, absent} labels at every
    (internal node, column).

    If ``missing_leaf_idx`` is provided, that leaf is treated as missing
    data (its set is {0, 1} at every column).

    Returns:
        labels: (num_internal, L) int array of 0/1 Fitch assignments.
    """
    L = leaf_present.shape[1]
    left_child = np.asarray(binary_tree.left_child)
    right_child = np.asarray(binary_tree.right_child)
    labels = np.zeros((binary_tree.num_internal, L), dtype=np.int32)

    for n in range(L):
        sets = [None] * binary_tree.num_nodes
        for li in range(binary_tree.num_leaves):
            node_idx = binary_tree.num_internal + li
            if missing_leaf_idx is not None and li == missing_leaf_idx:
                sets[node_idx] = {0, 1}
            else:
                sets[node_idx] = {int(leaf_present[li, n])}
        for v in binary_tree.postorder_internal:
            v = int(v)
            l, r = int(left_child[v]), int(right_child[v])
            inter = sets[l] & sets[r]
            sets[v] = inter if inter else (sets[l] | sets[r])
        # Down pass: assign root then propagate.
        assignments = [None] * binary_tree.num_nodes
        root = binary_tree.root
        assignments[root] = 1 if 1 in sets[root] else 0
        for v in binary_tree.preorder_internal:
            v = int(v)
            l, r = int(left_child[v]), int(right_child[v])
            for child in (l, r):
                if assignments[child] is not None:
                    continue
                if assignments[v] in sets[child]:
                    assignments[child] = assignments[v]
                else:
                    assignments[child] = next(iter(sets[child]))
        for v in range(binary_tree.num_internal):
            labels[v, n] = assignments[v]
    return labels


def fitch_seeded_init(binary_tree, leaf_present, holdout_leaf_idx,
                      smoothing=1.0):
    """Initialise tied edge logits + root logit from a Fitch parsimony
    labelling of the (leaves \\ {holdout}) tree.

    For each (parent, child) edge we count the empirical
    (parent-state, child-state) frequencies in 3-state space
    {NotYetInserted, Present, Deleted}, then convert to row-conditional
    probabilities for the two free variational rows (NYI parent and P
    parent). Laplace smoothing avoids zero/inf.

    Returns:
        edge_logits: (num_edges, 2) jnp array.
        root_logit: scalar jnp.
    """
    L = leaf_present.shape[1]
    fitch = fitch_labels(binary_tree, leaf_present, missing_leaf_idx=holdout_leaf_idx)

    # Convert binary labels to 3-state {N, P, D} via top-down pass.
    # Internal node states.
    state_int = np.zeros_like(fitch, dtype=np.int32)
    parent_arr = np.asarray(binary_tree.parent)
    for v in binary_tree.preorder_internal:
        v = int(v)
        for n in range(L):
            if fitch[v, n] == 1:
                state_int[v, n] = 1  # P
            else:
                par = int(parent_arr[v])
                if par == -1:  # root, absent
                    state_int[v, n] = 0  # N
                else:
                    par_st = state_int[par, n]
                    state_int[v, n] = 0 if par_st == 0 else 2  # N before insertion, D after

    # Leaf states (clamped from data; convert absent → N or D from parent's state).
    state_leaf = np.zeros((binary_tree.num_leaves, L), dtype=np.int32)
    for li in range(binary_tree.num_leaves):
        node = binary_tree.num_internal + li
        par = int(parent_arr[node])
        for n in range(L):
            if li == holdout_leaf_idx:
                # Treat as observed for init purposes: pick the state
                # consistent with the parent (no information from this leaf).
                par_st = state_int[par, n] if par >= 0 else 0
                state_leaf[li, n] = par_st if par_st != 2 else 2
            elif leaf_present[li, n] == 1:
                state_leaf[li, n] = 1  # P
            else:
                par_st = state_int[par, n] if par >= 0 else 0
                state_leaf[li, n] = 0 if par_st == 0 else 2

    # Count (parent_state, child_state) per edge across columns.
    edge_counts = np.zeros((binary_tree.num_edges, 3, 3))
    for e in range(binary_tree.num_edges):
        p_idx = int(binary_tree.edge_parent[e])
        c_idx = int(binary_tree.edge_child[e])
        for n in range(L):
            ps = state_int[p_idx, n]
            cs = (state_int[c_idx, n] if c_idx < binary_tree.num_internal
                  else state_leaf[c_idx - binary_tree.num_internal, n])
            edge_counts[e, ps, cs] += 1.0

    edge_counts += smoothing
    edge_logits = np.zeros((binary_tree.num_edges, 2))
    for e in range(binary_tree.num_edges):
        c = edge_counts[e]
        # NYI parent: q(NYI|NYI) vs q(P|NYI)
        sN = c[NYI, NYI] + c[NYI, PRESENT]
        p_NN = c[NYI, NYI] / sN
        # P parent: q(P|P) vs q(D|P)
        sP = c[PRESENT, PRESENT] + c[PRESENT, DELETED]
        p_PP = c[PRESENT, PRESENT] / sP
        edge_logits[e, 0] = float(np.log(p_NN / max(1.0 - p_NN, 1e-10)))
        edge_logits[e, 1] = float(np.log(p_PP / max(1.0 - p_PP, 1e-10)))

    # Root logit: q(root=NYI) under Fitch counts.
    root = binary_tree.root
    n_NYI = float((state_int[root] == 0).sum()) + smoothing
    n_P = float((state_int[root] == 1).sum()) + smoothing
    p_root_NYI = n_NYI / (n_NYI + n_P)
    root_logit = float(np.log(p_root_NYI / max(1.0 - p_root_NYI, 1e-10)))

    return jnp.asarray(edge_logits, dtype=jnp.float64), \
           jnp.asarray(root_logit, dtype=jnp.float64)


def predict_holdout_fitch(binary_tree, leaf_present, holdout_leaf_idx):
    """Fitch parsimony: per-column independent {0,1} reconstruction.

    For each column:
      Up pass: each internal node's set = intersection of child sets;
        if empty, set = union.
      Down pass: assign each internal node deterministically. Root
        prefers majority of its set; each child prefers parent's value
        if compatible, else any value in its own set.
      Held-out leaf prediction: project from the held-out leaf's parent
        following the same rule.
    """
    L = leaf_present.shape[1]
    out = np.zeros(L, dtype=np.float64)

    # Build child lookup for postorder processing.
    parent_arr = np.asarray(binary_tree.parent)
    left_child = np.asarray(binary_tree.left_child)
    right_child = np.asarray(binary_tree.right_child)
    holdout_node = binary_tree.num_internal + holdout_leaf_idx

    # Find parent of holdout leaf.
    holdout_parent = int(parent_arr[holdout_node])
    if holdout_parent == -1:
        # Holdout is the root (one-leaf tree?); cannot happen.
        return out

    # For each column, run Fitch.
    for n in range(L):
        sets = [None] * binary_tree.num_nodes
        # Leaves
        for li in range(binary_tree.num_leaves):
            node_idx = binary_tree.num_internal + li
            if li == holdout_leaf_idx:
                # Treat as missing data: set = {0, 1}.
                sets[node_idx] = {0, 1}
            else:
                sets[node_idx] = {int(leaf_present[li, n])}

        # Up pass in postorder.
        for v in binary_tree.postorder_internal:
            v = int(v)
            l, r = int(left_child[v]), int(right_child[v])
            inter = sets[l] & sets[r]
            sets[v] = inter if inter else (sets[l] | sets[r])

        # Down pass: assign root, then propagate.
        assignments = [None] * binary_tree.num_nodes
        root = binary_tree.root
        # Assign root: prefer 1 if 1 in set, else 0.
        assignments[root] = 1 if 1 in sets[root] else 0
        for v in binary_tree.preorder_internal:
            v = int(v)
            l, r = int(left_child[v]), int(right_child[v])
            for child in (l, r):
                if assignments[child] is not None:
                    continue
                # Prefer parent's value if compatible, else pick from set.
                if assignments[v] in sets[child]:
                    assignments[child] = assignments[v]
                else:
                    assignments[child] = next(iter(sets[child]))

        # Predict held-out leaf state: same rule from its parent.
        if assignments[holdout_parent] in sets[holdout_node]:
            out[n] = assignments[holdout_parent]
        else:
            out[n] = next(iter(sets[holdout_node]))

    return out


# ---------------------------------------------------------------------------
# Fels21-derived presence prediction.
# ---------------------------------------------------------------------------

def predict_holdout_fels21(family, held_out, msa_int, tree_node, Q21, pi21):
    """Run fels21 reconstruction and convert to presence/absence per column.

    Returns (p_present, elapsed) where p_present[n] in {0, 1}.
    The fels21 prediction sequence drops gap columns; we recover presence
    by intersecting predicted-AA columns with the original column index.
    """
    from experiments.unified_reconstruction_benchmark import prune_leaf_keep_parent
    from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax

    pruned_tree, _ = prune_leaf_keep_parent(tree_node, held_out)
    if pruned_tree is None:
        return None, 0.0
    name_internal_nodes(pruned_tree)
    remaining = [n for n in msa_int.keys() if n != held_out]
    msa21 = {}
    for name in remaining:
        seq = msa_int[name].copy()
        seq[seq < 0] = 20
        seq[seq >= 20] = 20
        msa21[name] = seq

    if not msa21:
        return None, 0.0

    tf = time.time()
    # marginal_ancestor_all_columns_jax returns (argmax, posteriors)
    # where posteriors has shape (num_internal_nodes, n_cols, n_states).
    # The "ancestor" output is at the prediction node (parent of holdout).
    # We need posteriors at THE PREDICTION node.
    ancestor, posteriors = marginal_ancestor_all_columns_jax(
        pruned_tree, msa21, Q21, pi21)
    elapsed = time.time() - tf

    # ancestor is an int sequence of length n_cols giving argmax aa per col.
    # P(present) = 1 if argmax in 0..19, 0 if argmax == 20.
    p_present = np.zeros(len(ancestor), dtype=np.float64)
    for c in range(len(ancestor)):
        a = int(ancestor[c])
        if 0 <= a < 20:
            p_present[c] = 1.0
    return p_present, elapsed


# ---------------------------------------------------------------------------
# F1 / precision / recall.
# ---------------------------------------------------------------------------

def f1_pr(predicted_prob, ground_truth, threshold=0.5):
    pred = predicted_prob >= threshold
    gt = ground_truth > 0
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return f1, precision, recall, tp, fp, fn, tn


# ---------------------------------------------------------------------------
# Confidence: log posterior of predicted presence/absence sequence.
# ---------------------------------------------------------------------------
#
# logp_target math (per-method confidence for ancestral reconstruction):
#   For presence/absence methods (varanc, fitch, mixdom, fels21-presence),
#   the prediction at the held-out leaf is a per-column binary vector
#   pred[c] = 1{p_present[c] >= 0.5}. We record the joint log posterior
#   of the predicted sequence under the method's posterior:
#       logp_target = sum_{c=0..L-1} log P(pred[c] | MSA \ target, tree)
#                   = sum_{c: pred[c]=1} log p_present[c]
#                   + sum_{c: pred[c]=0} log (1 - p_present[c])
#   p_present here is the method's own per-column posterior on the
#   held-out leaf (i.e. the same vector saved as 'p_present').
#
#   This is the user-requested "log posterior of the target leaf's
#   predicted sequence" — *not* the old `best_p` whole-MSA log probability.

def logp_binary_pred(p_present, threshold=0.5, eps=1e-30):
    """Joint log posterior of binary presence-prediction per column.

    Args:
        p_present: 1D array of P(present) per column.
        threshold: prediction cutoff (default 0.5).
        eps: floor to keep log finite at near-deterministic posteriors.

    Returns:
        float scalar logp_target.
    """
    p = np.asarray(p_present, dtype=np.float64)
    pred_present = p >= threshold
    log_p = np.log(np.maximum(p, eps))
    log_1mp = np.log(np.maximum(1.0 - p, eps))
    return float(np.sum(np.where(pred_present, log_p, log_1mp)))


def logp_binary_true(p_present, gt_present, eps=1e-30):
    """Joint log posterior of the GROUND-TRUTH binary presence vector
    under the method's per-column posterior.

    For binary methods (varanc, mixdom, fitch, fels21-as-presence):
        logp_true = sum_c [gt[c]=1] log p[c] + [gt[c]=0] log (1-p[c])

    Calibrated methods (varanc, mixdom) yield finite-negative values.
    Hard-label methods (fitch) have p in {0, 1}; logp_true = 0 if every
    column matches truth, otherwise -inf at any mismatch (the right
    answer — these methods cannot represent uncertainty). NumPy's
    log(0) returns -inf which propagates through the sum, so no special
    handling is required (we floor BOTH p and 1-p only for the
    calibrated case via `eps`; for hard 0/1 inputs the floor is hit
    only at the mismatching column and produces a finite-but-very-
    negative value rather than literal -inf — to faithfully report
    -inf we threshold p at exactly 0 / 1 first).

    Args:
        p_present: 1D array of P(present) per column.
        gt_present: 1D array of ground-truth {0, 1} per column.
        eps: floor for calibrated posteriors. Hard 0 / 1 entries are
            preserved (yielding -inf at any mismatch).

    Returns:
        float scalar logp_true.
    """
    p = np.asarray(p_present, dtype=np.float64)
    gt = np.asarray(gt_present, dtype=np.int32)
    # Hard-label inputs (p exactly in {0, 1}) preserve -inf at mismatches;
    # calibrated inputs are floored to keep finite.
    log_p = np.where(p == 0.0, -np.inf, np.log(np.maximum(p, eps)))
    log_1mp = np.where(p == 1.0, -np.inf, np.log(np.maximum(1.0 - p, eps)))
    return float(np.sum(np.where(gt > 0, log_p, log_1mp)))


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def _load_trained_tkf92_params():
    """Load TKF92 params fit on Pfam-seed train counts (preferred default)."""
    p = Path(__file__).parent / 'tkf92_fitted_params.json'
    try:
        with open(p) as f:
            d = json.load(f)
        return {
            'ins_rate': float(d['ins_rate']),
            'del_rate': float(d['del_rate']),
            'ext': float(d.get('ext_rate', d.get('ext', 0.6))),
        }
    except Exception:
        return None


# Trained TKF92 params (from python/experiments/tkf92_fitted_params.json,
# fit on Pfam-seed train counts). These are the canonical defaults — the
# DEFAULT_TKF_PARAMS hardcoded values are kept only as a fallback if the
# fitted-params JSON is missing.
DEFAULT_TKF_PARAMS = _load_trained_tkf92_params() or {
    'ins_rate': 0.05,
    'del_rate': 0.06,
    'ext': 0.6,
}


def load_treefam_family(fspec, treefam_dir):
    """Load a TreeFam family: aa.fasta MSA + .nh.emf newick tree.

    Returns (msa_int, tree_node, n_cols).
    """
    from tkfmixdom.util.msa_benchmark import parse_fasta
    from experiments.fels21_reconstruction_benchmark import parse_treefam_tree
    fam = fspec['family']
    fasta_path = os.path.join(treefam_dir, f'{fam}.aa.fasta')
    tree_path = os.path.join(treefam_dir, f'{fam}.nh.emf')
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f'TreeFam fasta missing: {fasta_path}')
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f'TreeFam tree missing: {tree_path}')

    raw_seqs = parse_fasta(fasta_path)
    if not raw_seqs:
        raise ValueError(f'No sequences in {fasta_path}')
    C = len(next(iter(raw_seqs.values())))

    msa_int = {}
    for name, seq in raw_seqs.items():
        arr = np.full(C, -1, dtype=np.int32)
        for j, ch in enumerate(seq[:C]):
            cu = ch.upper()
            if cu in AA_TO_INT and AA_TO_INT[cu] < 20:
                arr[j] = AA_TO_INT[cu]
        msa_int[name] = arr

    tree = parse_treefam_tree(tree_path)
    # Prune tree to MSA leaves only (some TreeFam trees have extra leaves
    # not in the MSA fasta).
    import copy as cp
    tree = cp.deepcopy(tree)
    msa_names = set(raw_seqs.keys())

    def prune_node(node):
        node.children = [c for c in node.children if (c.children or (c.name or '') in msa_names)]
        for c in node.children:
            prune_node(c)
    prune_node(tree)
    # Splice unary internals.
    def splice(node):
        for i, c in enumerate(node.children):
            if c.children and len(c.children) == 1:
                gc = c.children[0]
                gc.branch_length += c.branch_length
                node.children[i] = gc
            splice(node.children[i])
    splice(tree)
    while len(tree.children) == 1:
        c = tree.children[0]
        c.branch_length = 0.0
        tree = c

    return msa_int, tree, C


def load_balibase_family(fspec, balibase_ref_dir):
    """Load a BAliBASE family: ref MSA + FastTree-built tree.

    Returns (msa_int, tree_node, n_cols) compatible with load_pfam_family.
    """
    import subprocess
    import tempfile
    from tkfmixdom.util.msa_benchmark import parse_fasta
    fam = fspec['family']
    ref_path = os.path.join(balibase_ref_dir, fam)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f'BAliBASE ref not found: {ref_path}')

    raw_seqs = parse_fasta(ref_path)
    if not raw_seqs:
        raise ValueError(f'No sequences in {ref_path}')
    C = len(next(iter(raw_seqs.values())))

    msa_int = {}
    for name, seq in raw_seqs.items():
        arr = np.full(C, -1, dtype=np.int32)
        for j, ch in enumerate(seq[:C]):
            cu = ch.upper()
            if cu in AA_TO_INT and AA_TO_INT[cu] < 20:
                arr[j] = AA_TO_INT[cu]
        msa_int[name] = arr

    # Build FastTree on the ref MSA (uppercase, dashes preserved).
    fasttree_bin = os.path.expanduser('~/bin/FastTree')
    if not os.path.exists(fasttree_bin):
        fasttree_bin = 'FastTree'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        for name, seq in raw_seqs.items():
            f.write(f'>{name}\n{seq.upper()}\n')
        msa_path = f.name
    try:
        result = subprocess.run(
            [fasttree_bin, '-quiet', '-lg'],
            stdin=open(msa_path), capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f'FastTree failed: {result.stderr}')
        tree = parse_newick(result.stdout.strip())
    finally:
        os.unlink(msa_path)

    return msa_int, tree, C


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',
                        choices=['unified_short_test', 'unified_long_test',
                                 'unified_hard_test', 'unified_xhard_test',
                                 'unified_short', 'unified_long',
                                 'balibase', 'treefam'],
                        default='unified_short_test')
    parser.add_argument('--n-families', type=int, default=0,
                        help='Limit to N families (0 = all).')
    parser.add_argument('--methods', type=str,
                        default='varanc,fitch,fels21',
                        help='Comma-separated subset of '
                             '{varanc,varanc_untied,fitch,fels21}. '
                             'varanc=tied edge logits (E, 2); '
                             'varanc_untied=per-(edge, col) (E, L, 2).')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--n-iter', type=int, default=150,
                        help='Adam iterations for varanc.')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--ins-rate', type=float,
                        default=DEFAULT_TKF_PARAMS['ins_rate'])
    parser.add_argument('--del-rate', type=float,
                        default=DEFAULT_TKF_PARAMS['del_rate'])
    parser.add_argument('--ext', type=float, default=DEFAULT_TKF_PARAMS['ext'])
    parser.add_argument('--skip-existing-fams', action='store_true',
                        help='Load existing --out JSON and skip families '
                        'already present in its results list. Used for '
                        'partial re-runs after some entries have been '
                        'deleted (see empty_col_cleanup.py).')
    args = parser.parse_args()

    methods = set(args.methods.split(','))
    if args.dataset == 'unified_short_test':
        spec_path = Path(__file__).parent / 'unified_benchmark_test_spec.json'
    elif args.dataset == 'unified_long_test':
        spec_path = Path(__file__).parent / 'unified_benchmark_long_test_spec.json'
    elif args.dataset == 'unified_hard_test':
        spec_path = Path(__file__).parent / 'unified_benchmark_hard_test_spec.json'
    elif args.dataset == 'unified_xhard_test':
        spec_path = Path(__file__).parent / 'unified_benchmark_xhard_test_spec.json'
    elif args.dataset == 'unified_short':
        # Contaminated val-split spec (kept as a way to reproduce
        # pre-2026-05-04 numbers). Spec file moved to triage dir.
        spec_path = Path(__file__).parent / 'contaminated_val_split_triage_for_deletion' / 'unified_benchmark_spec.json'
    elif args.dataset == 'unified_long':
        spec_path = Path(__file__).parent / 'contaminated_val_split_triage_for_deletion' / 'unified_benchmark_long_spec.json'
    elif args.dataset == 'balibase':
        spec_path = Path(__file__).parent / 'balibase_reconstruction_spec.json'
    else:  # treefam
        spec_path = Path(__file__).parent / 'treefam_reconstruction_spec.json'

    with open(spec_path) as f:
        spec = json.load(f)
    families = spec['families']
    pfam_dir = tree_dir = balibase_ref_dir = treefam_dir = None
    if args.dataset.startswith('unified_'):
        pfam_dir = os.path.expanduser(spec['pfam_dir'])
        tree_dir = os.path.expanduser(spec['tree_dir'])
    elif args.dataset == 'balibase':
        balibase_ref_dir = os.path.expanduser(
            "~/bio-datasets/data/balibase/bali3pdbm/ref")
    else:  # treefam
        treefam_dir = os.path.expanduser(spec.get('treefam_dir',
            "~/bio-datasets/data/treefam/treefam_family_data"))
    if args.n_families > 0:
        families = families[:args.n_families]

    # Pre-load fels21 Q matrix if needed.
    Q21 = pi21 = None
    if 'fels21' in methods:
        # Fetch fitted fels21 params; default to params/fels21_lg08_init.npz.
        from tkfmixdom.jax.core.protein import rate_matrix_lg
        Q20, pi20 = rate_matrix_lg()
        Q20, pi20 = np.asarray(Q20), np.asarray(pi20)
        # Build a simple 21-state extension: stationary gap freq = 0.05, gap rate = 0.05.
        Q21 = np.zeros((21, 21))
        Q21[:20, :20] = Q20 * 0.95
        gap_rate = 0.05
        Q21[:20, 20] = gap_rate
        Q21[20, :20] = gap_rate * pi20 / pi20.sum()
        Q21[range(21), range(21)] = 0.0
        for i in range(21):
            Q21[i, i] = -np.sum(Q21[i, :])
        pi21 = np.zeros(21)
        pi21[:20] = pi20 * 0.95
        pi21[20] = 0.05

    out_path = args.out or f'experiments/varanc_presence_{args.dataset}.json'

    existing_results = []
    skip_fams = set()
    if args.skip_existing_fams and os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing_results = json.load(f).get('results', [])
            skip_fams = {r['family'] for r in existing_results
                          if r.get('family')}
            log(f"--skip-existing-fams: loaded {len(existing_results)} "
                f"existing entries from {out_path}; will skip those")
        except Exception as e:
            log(f"--skip-existing-fams: failed to load {out_path}: {e}")

    log(f"Running on {len(families)} families ({args.dataset}); methods={methods}")
    print(f"{'family':<12} {'L':>3} {'V':>3} | "
          + " | ".join(f"{m:>10} F1" for m in sorted(methods))
          + " | time")
    print("-" * (40 + 14 * len(methods)))

    results = list(existing_results)
    save_every = 5
    for fi, fspec in enumerate(families):
        if fspec['family'] in skip_fams:
            continue
        try:
            if args.dataset.startswith('unified_'):
                msa_int, tree_node, C = load_pfam_family(fspec, pfam_dir, tree_dir)
            elif args.dataset == 'balibase':
                msa_int, tree_node, C = load_balibase_family(fspec, balibase_ref_dir)
            else:  # treefam
                msa_int, tree_node, C = load_treefam_family(fspec, treefam_dir)
        except Exception as e:
            log(f"[{fi+1}/{len(families)}] {fspec['family']}: load failed: {e}")
            continue

        held_out = fspec['held_out']
        if held_out not in msa_int:
            log(f"[{fi+1}/{len(families)}] {fspec['family']}: held_out absent from MSA")
            continue

        # Convert MSA to presence/absence per column for all leaves.
        leaf_names = sorted(msa_int.keys())
        present_arr = np.zeros((len(leaf_names), C), dtype=np.int32)
        for i, name in enumerate(leaf_names):
            present_arr[i] = (msa_int[name] >= 0).astype(np.int32)

        # Truth: held-out leaf's per-column presence.
        gt_present = present_arr[leaf_names.index(held_out)]

        # Build BinaryTree from the tree_node (binarised).
        try:
            binary_tree = build_binary_tree_from_node(tree_node)
        except Exception as e:
            log(f"[{fi+1}/{len(families)}] {fspec['family']}: tree binarise failed: {e}")
            continue

        # Map BinaryTree.leaf_names to present_arr rows.
        bt_leaf_to_present_row = {}
        for li, lname in enumerate(binary_tree.leaf_names):
            if lname in leaf_names:
                bt_leaf_to_present_row[li] = leaf_names.index(lname)

        # Re-build present_arr in BinaryTree leaf order.
        bt_present = np.zeros((binary_tree.num_leaves, C), dtype=np.int32)
        for li in range(binary_tree.num_leaves):
            row = bt_leaf_to_present_row.get(li)
            if row is not None:
                bt_present[li] = present_arr[row]

        if held_out not in binary_tree.leaf_names:
            log(f"[{fi+1}/{len(families)}] {fspec['family']}: held_out missing in binary tree")
            continue
        holdout_idx_bt = binary_tree.leaf_names.index(held_out)

        entry = {
            'family': fspec['family'],
            'held_out': held_out,
            'n_cols': int(C),
            'n_leaves': int(binary_tree.num_leaves),
            'n_internal': int(binary_tree.num_internal),
            'gt_present': gt_present.tolist(),
            'methods': {},
        }

        line = f"{fspec['family']:<12} {C:>3} {binary_tree.num_internal:>3} |"

        # Stash internal-node MAP profiles for the model-probability selector.
        # These are only used to compute log p̃ at the end; they're not the
        # same as the leaf prediction.
        varanc_internal_MAP = None
        fitch_internal_MAP = None

        # Method: VarAnc-presence (TIED edge logits — (E, 2))
        if 'varanc' in methods:
            try:
                tv0 = time.time()
                p_pred, varanc_internal_MAP = _predict_one_holdout_varanc(
                    binary_tree, bt_present, holdout_idx_bt, held_out,
                    binary_tree.leaf_names,
                    args.ins_rate, args.del_rate, args.ext,
                    n_iter=args.n_iter, lr=args.lr, seed=fi,
                    tie_edge_logits=True)
                tv = time.time() - tv0
                f1, prec, rec, *_ = f1_pr(p_pred, gt_present)
                entry['methods']['varanc'] = {
                    'p_present': p_pred.tolist(),
                    'f1': f1, 'precision': prec, 'recall': rec, 'time': tv,
                    'logp_target': logp_binary_pred(p_pred),
                    'logp_true': logp_binary_true(p_pred, gt_present),
                }
                line += f"  varanc {f1:.3f} |"
            except Exception as e:
                line += f"  varanc ERR |"
                entry['methods']['varanc'] = {'error': str(e)}

        # Method: VarAnc-presence (UNTIED edge logits — (E, L, 2))
        # — extra per-column flexibility, more parameters per family.
        if 'varanc_untied' in methods:
            try:
                tv0 = time.time()
                p_pred, _ = _predict_one_holdout_varanc(
                    binary_tree, bt_present, holdout_idx_bt, held_out,
                    binary_tree.leaf_names,
                    args.ins_rate, args.del_rate, args.ext,
                    n_iter=args.n_iter, lr=args.lr, seed=fi,
                    tie_edge_logits=False)
                tv = time.time() - tv0
                f1, prec, rec, *_ = f1_pr(p_pred, gt_present)
                entry['methods']['varanc_untied'] = {
                    'p_present': p_pred.tolist(),
                    'f1': f1, 'precision': prec, 'recall': rec, 'time': tv,
                    'logp_target': logp_binary_pred(p_pred),
                    'logp_true': logp_binary_true(p_pred, gt_present),
                }
                line += f"  varanc_untied {f1:.3f} |"
            except Exception as e:
                line += f"  varanc_untied ERR |"
                entry['methods']['varanc_untied'] = {'error': str(e)}

        # Method: Fitch
        if 'fitch' in methods:
            try:
                tv0 = time.time()
                p_pred = predict_holdout_fitch(
                    binary_tree, bt_present, holdout_idx_bt)
                fitch_internal_MAP = fitch_labels(
                    binary_tree, bt_present, missing_leaf_idx=holdout_idx_bt)
                tv = time.time() - tv0
                f1, prec, rec, *_ = f1_pr(p_pred, gt_present)
                # Hard-label predictor: no probabilistic posterior, so
                # logp_target / logp_true are category errors here. Omit.
                entry['methods']['fitch'] = {
                    'p_present': p_pred.tolist(),
                    'f1': f1, 'precision': prec, 'recall': rec, 'time': tv,
                }
                line += f"   fitch {f1:.3f} |"
            except Exception as e:
                line += f"   fitch ERR |"
                entry['methods']['fitch'] = {'error': str(e)}

        # Method: Fels21
        if 'fels21' in methods:
            try:
                tv0 = time.time()
                p_pred, _ = predict_holdout_fels21(
                    fspec['family'], held_out, msa_int, tree_node, Q21, pi21)
                tv = time.time() - tv0
                if p_pred is None:
                    raise ValueError('fels21 returned None')
                # Pad/truncate to match C if needed.
                if len(p_pred) != C:
                    if len(p_pred) > C:
                        p_pred = p_pred[:C]
                    else:
                        p_pred = np.concatenate([p_pred, np.zeros(C - len(p_pred))])
                f1, prec, rec, *_ = f1_pr(p_pred, gt_present)
                # fels21-as-presence is hard-label here (argmax over
                # 21 states then thresholded to {present, absent}); the
                # per-column posterior is degenerate 0/1 so logp_target /
                # logp_true would be 0 / -inf — category error, omit.
                # The standalone fels21 residue launcher (real posteriors)
                # still emits both fields.
                entry['methods']['fels21'] = {
                    'p_present': p_pred.tolist(),
                    'f1': f1, 'precision': prec, 'recall': rec, 'time': tv,
                }
                line += f"  fels21 {f1:.3f} |"
            except Exception as e:
                line += f"  fels21 ERR |"
                entry['methods']['fels21'] = {'error': str(e)}

        # Method: best_p — model-probability-selected reconstruction.
        # Compute log p̃(MSA, internals) for fitch and varanc reconstructions,
        # using each method's leaf prediction (binary-thresholded at 0.5) as
        # the held-out leaf's state. Pick whichever has higher log p̃.
        if (varanc_internal_MAP is not None and fitch_internal_MAP is not None
                and 'varanc' in entry['methods'] and 'fitch' in entry['methods']
                and 'p_present' in entry['methods']['varanc']
                and 'p_present' in entry['methods']['fitch']):
            try:
                varanc_pred_bin = (np.asarray(
                    entry['methods']['varanc']['p_present']) >= 0.5).astype(np.int32)
                fitch_pred_bin = (np.asarray(
                    entry['methods']['fitch']['p_present']) >= 0.5).astype(np.int32)
                # Substitute method's prediction for held-out leaf's row.
                varanc_leaves = bt_present.copy()
                varanc_leaves[holdout_idx_bt] = varanc_pred_bin
                fitch_leaves = bt_present.copy()
                fitch_leaves[holdout_idx_bt] = fitch_pred_bin

                lp_varanc = log_p_profile(
                    binary_tree, varanc_internal_MAP, varanc_leaves,
                    args.ins_rate, args.del_rate, args.ext)
                lp_fitch = log_p_profile(
                    binary_tree, fitch_internal_MAP, fitch_leaves,
                    args.ins_rate, args.del_rate, args.ext)
                pick = 'varanc' if lp_varanc >= lp_fitch else 'fitch'
                p_best = np.asarray(entry['methods'][pick]['p_present'])
                f1, prec, rec, *_ = f1_pr(p_best, gt_present)
                entry['methods']['best_p'] = {
                    'p_present': p_best.tolist(),
                    'f1': f1, 'precision': prec, 'recall': rec,
                    'log_p_varanc': lp_varanc,
                    'log_p_fitch': lp_fitch,
                    'picked': pick,
                }
                line += f"  best_p {f1:.3f}/{pick[0]} |"
            except Exception as e:
                line += f"  best_p ERR |"
                entry['methods']['best_p'] = {'error': str(e)}

        line += f" {time.time()-t0:.0f}s"
        print(line)
        results.append(entry)

        if (fi + 1) % save_every == 0:
            with open(out_path, 'w') as f:
                json.dump({'spec': args.dataset, 'tkf_params': {
                    'ins_rate': args.ins_rate, 'del_rate': args.del_rate,
                    'ext': args.ext}, 'results': results}, f)

    # Summary.
    print("\n" + "=" * 60)
    print(f"Final: n={len(results)} entries")
    for m in sorted(methods):
        f1s = [r['methods'][m]['f1'] for r in results
               if m in r['methods'] and 'f1' in r['methods'][m]]
        if f1s:
            print(f"  {m:>8}: F1 mean={np.mean(f1s):.4f}, median={np.median(f1s):.4f}, n={len(f1s)}")

    with open(out_path, 'w') as f:
        json.dump({'spec': args.dataset, 'tkf_params': {
            'ins_rate': args.ins_rate, 'del_rate': args.del_rate,
            'ext': args.ext}, 'results': results}, f)
    log(f"Saved {out_path}")


if __name__ == '__main__':
    main()
