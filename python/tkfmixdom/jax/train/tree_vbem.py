"""Tree-VBEM training for MixDom.

Per varanc-vbem.tex (appendix N): outer loop alternates between
per-family E-step (Adam ascent on the variational q_i at fixed θ) and
a global M-step (closed-form θ update from aggregated sufficient
statistics).

Sufficient statistics extracted per family:

  * W^(v→w)_{ss', τ τ'} : per-branch reduced expected counts
        (5×T×5×T tensor); driver of indel/Markov M-steps.
  * q^(τ)_n(τ) : per-column tuple categorical; driver of dom_w/frag_w.
  * q^(c|f)_n(c) : per-column class posterior; weights HR substitution counts.
  * Per-class Felsenstein expected substitution counts on Fitch subtree
        (deferred — implemented as TODO; smoke-training uses
        existing class params unchanged).

The M-step decomposes into route-attributed BDI counts per domain
(driving m_step_indel_quadratic), Dirichlet-conjugate updates for
dom_w/frag_w/ext, and (when implemented) class-weighted HR for
substitution params.

For now the smoke-training pipeline updates ONLY the indel-side
parameters (main rates, per-domain rates, dom_w, frag_w, ext_rates),
holding the substitution params (classdist, class_pis, class_S_exch)
fixed at the warm-start values. Class M-steps require Felsenstein-
posterior expected substitution counts, which is a separate piece of
machinery and is left as a TODO marked clearly below.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..core.bdi import (
    transition_count_groups, m_step_indel_quadratic,
    bdi_stats_from_counts,
)
from ..core.params import S as _S, M as _M, I as _I, D as _D, E as _E
from ..tree.varanc_presence import (
    parse_binary_tree, edge_lookup, BinaryTree,
    make_q_conditionals, make_root_dist, leaf_clamp_to_beta,
    bp_pair_marginals, entropy_per_column,
    NYI, PRESENT, DELETED, N_Z,
)
# MixDom WFST is 5-state (S, M, I, D, E); the TKF92 WFST in varanc_presence
# is now 6-state (with the Ins0/Ins1 split). MixDom currently uses the 5-state
# form, so we keep a local N_WFST=5 here. If MixDom adopts the Ins0/Ins1
# split, switch to importing N_WFST from varanc_presence.
N_WFST = 5
from ..tree.varanc_presence_mixdom import (
    mixdom_reduced_T_pair, mixdom_reduced_T_per_route, mixdom_omega,
    make_tuple_dist, fragchar_marginal_from_tuple, domain_marginal_from_tuple,
    expected_branch_LL_mixdom, class_marginalised_sub_LL_per_tuple,
    singlet_root_log_prior_mixdom,
    WFST_S, WFST_M, WFST_I, WFST_D, WFST_E,
)


# ---------------------------------------------------------------------------
# Geometric padding for JIT cache reuse.
#
# Both the column count (L) and the leaf count (and hence num_edges,
# num_internal) feed into the JIT compile shape for fit_family_estep.
# Without padding, JAX recompiles for every distinct (n_leaves, L) pair
# encountered, which costs ~30s per unique shape. Pfam has 19,850
# distinct families with widely varying tree sizes and MSA widths, so
# unbinned compilation can dominate runtime entirely (the in-flight val
# pass spends > 2 hours just compiling).
#
# Strategy: pad both axes to the next geometric bin size, with masks
# zeroing out the padded contributions everywhere they enter the ELBO
# so the math is unchanged (only wall-time differs).
#
# Tree padding: insert ghost leaves under a new root, whose leaf clamps
# are [1, 1, 1] (no evidence). Combined with row-stochastic q_cond and
# an explicit edge_mask zeroing branch-LL + edge entropy on ghost
# edges, the ELBO contribution of the ghost subtree is exactly zero.
#
# Column padding: pad leaf_present with absent leaves and sub_LL with
# uniform-1 (so log_L = 0). A col_mask zeros entropy/prior/branch-LL
# contributions on padded columns; the W tensor cumulant trick is
# masked by zeroing P_M/P_I/P_D and forcing P_Ig = 1 on padded cols.
# ---------------------------------------------------------------------------


# Bins for leaf count: 4, 8, 16, 32, 64, 128, 256, 512, 1024, ...
_LEAF_BINS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Bins for MSA width (columns): 16, 32, 64, 128, 256, 512, 1024, ...
_COL_BINS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _pad_to_bin(n, bins):
    """Round n up to next geometric bin size from `bins`."""
    n = max(int(n), 1)
    for b in bins:
        if b >= n:
            return int(b)
    # Beyond precomputed bins: round up to next power of 2.
    return int(2 ** int(np.ceil(np.log2(max(n, 1)))))


def _pad_columns(leaf_present, sub_LL_per_class, target_L):
    """Pad column axis to target_L. Returns padded arrays + col_mask.

    leaf_present: (n_leaves, L) -> (n_leaves, target_L). Padded cols set
        to 0 (absent everywhere).
    sub_LL_per_class: (L, n_classes) -> (target_L, n_classes). Padded
        rows set to 1.0 (uniform; log = 0).
    Returns (leaf_present_pad, sub_LL_pad, col_mask) where col_mask
    is (target_L,) with 1 on real columns, 0 on padded.
    """
    L = leaf_present.shape[1]
    n_classes = sub_LL_per_class.shape[1]
    if L > target_L:
        raise ValueError(f"L={L} > target_L={target_L}")
    if L == target_L:
        col_mask = np.ones(L, dtype=np.float64)
        return (np.asarray(leaf_present), np.asarray(sub_LL_per_class),
                col_mask)
    pad_L = target_L - L
    leaf_present_pad = np.concatenate([
        np.asarray(leaf_present),
        np.zeros((leaf_present.shape[0], pad_L), dtype=leaf_present.dtype),
    ], axis=1)
    sub_LL_pad = np.concatenate([
        np.asarray(sub_LL_per_class),
        np.ones((pad_L, n_classes), dtype=sub_LL_per_class.dtype),
    ], axis=0)
    col_mask = np.zeros(target_L, dtype=np.float64)
    col_mask[:L] = 1.0
    return leaf_present_pad, sub_LL_pad, col_mask


def _pad_tree_with_ghosts(tree, target_n_leaves):
    """Pad a binary tree to have target_n_leaves leaves via ghost subtree.

    Returns:
        padded_tree: BinaryTree with target_n_leaves leaves.
        edge_mask: (num_edges_pad,) {0, 1}; 1 on real edges (those that
            exist in the original tree), 0 on ghost edges.
        ghost_leaf_indices: (n_ghost,) array of leaf-node indices in the
            padded tree corresponding to ghost leaves (used to override
            leaf_clamp to [1, 1, 1] for "no evidence").
        leaf_index_map: (orig_n_leaves,) array; leaf_index_map[i] is the
            leaf-node index in the padded tree corresponding to original
            leaf i. Used to remap leaf_present (n_leaves_orig, L) to the
            padded tree's leaf node ordering.
        n_real_edges: number of real (non-ghost) edges (= original tree's
            num_edges).

    Layout convention (chosen for JIT cache reuse — all FIXED indices
    depend only on target_n_leaves, not on the original tree topology):

      Internals: ghost internals at 0..n_ghost_int-1, then original
                 internals at n_ghost_int..n_ghost_int+orig_n_int-1
                 (REINDEXED by adding n_ghost_int to original indices),
                 with new top-level root at pad_n_internal-1.
      Leaves:    pad_n_internal..pad_n_nodes-1 — original leaves first,
                 then ghost leaves.

    With this layout:
      - tree.root = pad_n_internal - 1 (FIXED for a given bin)
      - tree.num_internal = pad_n_internal (FIXED)
      - tree.num_edges = pad_n_edges (FIXED)
    so JAX's JIT cache can reuse compiled functions across all families
    in the same bin (only array contents differ, not shapes/scalars).

    Ghost edges always last in the edge array, ensuring n_real_edges
    real-edge prefix is well-defined.
    """
    orig_n_leaves = int(tree.num_leaves)
    target_n_leaves = int(target_n_leaves)
    if orig_n_leaves > target_n_leaves:
        raise ValueError(
            f"orig_n_leaves={orig_n_leaves} > target={target_n_leaves}")

    pad_n_internal = target_n_leaves - 1
    pad_n_nodes = 2 * target_n_leaves - 1
    pad_n_edges = 2 * target_n_leaves - 2

    if orig_n_leaves == target_n_leaves:
        # Already at bin size — but we still need the layout convention
        # (root = pad_n_internal - 1) for JIT cache reuse. Since the
        # original tree's root is the LAST internal in postorder = the
        # last index after parse_binary_tree, we can usually just return
        # the tree as-is. But the original convention has root as the
        # FIRST internal (postorder index 0)... actually no — postorder
        # visits children before parents, so root IS the last in postorder.
        # parse_binary_tree's `node_idx` enumerates postorder, so root
        # has the highest index = num_internal - 1. So tree.root is
        # ALREADY at pad_n_internal - 1. Good — no remapping needed.
        edge_mask = np.ones(int(tree.num_edges), dtype=np.float64)
        ghost_leaf_indices = np.zeros(0, dtype=np.int32)
        leaf_index_map = np.arange(
            int(tree.num_internal),
            int(tree.num_internal) + orig_n_leaves, dtype=np.int32)
        return tree, edge_mask, ghost_leaf_indices, leaf_index_map, int(tree.num_edges)

    n_ghost = target_n_leaves - orig_n_leaves
    # Ghost-internal count: total - orig - 1 (the new top root).
    orig_n_internal = int(tree.num_internal)
    n_ghost_internals = pad_n_internal - orig_n_internal - 1

    new_root_idx = pad_n_internal - 1  # FIXED for this bin

    # Reindex original internals: add n_ghost_internals to all original
    # internal indices.
    def remap_internal(orig_idx):
        return orig_idx + n_ghost_internals

    # Ghost internals: 0..n_ghost_internals-1.
    # Leaves: pad_n_internal..pad_n_nodes-1.
    # Original leaves first: pad_n_internal..pad_n_internal+orig_n_leaves-1.
    # Ghost leaves last: pad_n_internal+orig_n_leaves..pad_n_nodes-1.
    leaf_index_map = np.arange(
        pad_n_internal, pad_n_internal + orig_n_leaves, dtype=np.int32)
    ghost_leaf_indices = np.arange(
        pad_n_internal + orig_n_leaves, pad_n_nodes, dtype=np.int32)

    parent = np.full(pad_n_nodes, -1, dtype=np.int32)
    left_child = np.full(pad_n_internal, -1, dtype=np.int32)
    right_child = np.full(pad_n_internal, -1, dtype=np.int32)

    # Copy original tree's internal structure, remapping all internal
    # indices by adding n_ghost_internals; remap leaves via leaf_index_map.
    orig_root = int(tree.root)
    orig_root_pad = remap_internal(orig_root)
    for v in range(orig_n_internal):
        v_pad = remap_internal(v)
        l = int(tree.left_child[v])
        r = int(tree.right_child[v])
        if l >= orig_n_internal:
            l_pad = int(leaf_index_map[l - orig_n_internal])
        else:
            l_pad = remap_internal(l)
        if r >= orig_n_internal:
            r_pad = int(leaf_index_map[r - orig_n_internal])
        else:
            r_pad = remap_internal(r)
        left_child[v_pad] = l_pad
        right_child[v_pad] = r_pad
        parent[l_pad] = v_pad
        parent[r_pad] = v_pad

    # Build ghost subtree: comb of (n_ghost_internals) internal nodes
    # with ghost leaves attached, terminating at the deepest ghost
    # internal (or directly at a leaf for n_ghost == 1).
    if n_ghost == 1:
        ghost_subtree_root = int(ghost_leaf_indices[0])
    else:
        # ghost_internals list: indices 0..n_ghost_internals-1 in the
        # padded tree. Comb structure: ghost_int[0] is at the top
        # (closest to new_root), ghost_int[k-1] is the deepest.
        # ghost_int[i] has children (ghost_leaf[i], ghost_int[i+1])
        # where ghost_int[k-1] has children (ghost_leaf[k-1], ghost_leaf[k]).
        ghost_internals = list(range(n_ghost_internals))
        for i, gi in enumerate(ghost_internals):
            l_pad = int(ghost_leaf_indices[i])
            if i + 1 < n_ghost_internals:
                r_pad = ghost_internals[i + 1]
            else:
                r_pad = int(ghost_leaf_indices[i + 1])
            left_child[gi] = l_pad
            right_child[gi] = r_pad
            parent[l_pad] = gi
            parent[r_pad] = gi
        ghost_subtree_root = ghost_internals[0]

    # Wire the new top-level root: children = (orig_root_pad, ghost_subtree_root).
    left_child[new_root_idx] = orig_root_pad
    right_child[new_root_idx] = ghost_subtree_root
    parent[orig_root_pad] = new_root_idx
    parent[ghost_subtree_root] = new_root_idx

    # Build edge arrays: real edges first (preserving original edge
    # ordering, just remapping internal/leaf indices), then ghost edges.
    n_real_edges = int(tree.num_edges)
    edge_parent = np.zeros(pad_n_edges, dtype=np.int32)
    edge_child = np.zeros(pad_n_edges, dtype=np.int32)
    edge_length = np.zeros(pad_n_edges, dtype=np.float64)

    for e in range(n_real_edges):
        ep = int(tree.edge_parent[e])
        ec = int(tree.edge_child[e])
        ep_pad = remap_internal(ep)  # parent is always internal
        if ec >= orig_n_internal:
            ec_pad = int(leaf_index_map[ec - orig_n_internal])
        else:
            ec_pad = remap_internal(ec)
        edge_parent[e] = ep_pad
        edge_child[e] = ec_pad
        edge_length[e] = float(tree.edge_length[e])

    eps_branch = 1e-3
    next_e = n_real_edges
    # Ghost edge: new_root -> orig_root_pad.
    edge_parent[next_e] = new_root_idx
    edge_child[next_e] = orig_root_pad
    edge_length[next_e] = eps_branch
    next_e += 1
    # Ghost edge: new_root -> ghost_subtree_root.
    edge_parent[next_e] = new_root_idx
    edge_child[next_e] = ghost_subtree_root
    edge_length[next_e] = eps_branch
    next_e += 1
    if n_ghost >= 2:
        for i, gi in enumerate(ghost_internals):
            l_pad = int(ghost_leaf_indices[i])
            edge_parent[next_e] = gi
            edge_child[next_e] = l_pad
            edge_length[next_e] = eps_branch
            next_e += 1
            if i + 1 < n_ghost_internals:
                r_pad = ghost_internals[i + 1]
            else:
                r_pad = int(ghost_leaf_indices[i + 1])
            edge_parent[next_e] = gi
            edge_child[next_e] = r_pad
            edge_length[next_e] = eps_branch
            next_e += 1
    assert next_e == pad_n_edges, f"edge count mismatch: {next_e} != {pad_n_edges}"

    edge_mask = np.zeros(pad_n_edges, dtype=np.float64)
    edge_mask[:n_real_edges] = 1.0

    # Postorder of internals: ghosts first (deepest -> shallowest), then
    # original internals (in their original postorder, remapped), then
    # new_root. This visits children-before-parent for all internals.
    orig_postorder = np.asarray(tree.postorder_internal, dtype=np.int32)
    orig_postorder_pad = orig_postorder + n_ghost_internals

    if n_ghost >= 2:
        # Ghost internals: ghost_int[k-1] (deepest) ... ghost_int[0] (shallowest).
        ghost_postorder = np.array(ghost_internals[::-1], dtype=np.int32)
        postorder_internal = np.concatenate([
            ghost_postorder, orig_postorder_pad,
            np.array([new_root_idx], dtype=np.int32)
        ])
    else:
        postorder_internal = np.concatenate([
            orig_postorder_pad, np.array([new_root_idx], dtype=np.int32)
        ])
    preorder_internal = postorder_internal[::-1].copy()

    leaf_names = list(tree.leaf_names) + [
        f"__ghost_{i}__" for i in range(n_ghost)]

    padded = BinaryTree(
        num_internal=pad_n_internal,
        num_leaves=target_n_leaves,
        num_nodes=pad_n_nodes,
        num_edges=pad_n_edges,
        parent=parent,
        left_child=left_child,
        right_child=right_child,
        edge_parent=edge_parent,
        edge_child=edge_child,
        edge_length=edge_length,
        postorder_internal=postorder_internal,
        preorder_internal=preorder_internal,
        leaf_names=leaf_names,
        root=new_root_idx,
    )
    return padded, edge_mask, ghost_leaf_indices, leaf_index_map, n_real_edges


# ---------------------------------------------------------------------------
# Masked variants of branch-LL / entropy / root-prior / W-tensor.
# ---------------------------------------------------------------------------


def _expected_branch_LL_mixdom_masked(pair_marg_branch, q_tau,
                                       log_T_branch, col_mask, eps=1e-30):
    """expected_branch_LL_mixdom with column masking.

    On padded columns (col_mask[n] = 0), force P_M = P_I = P_D = 0 and
    P_Ig = 1 so the cumulant integral skips those columns entirely.
    """
    P_M = pair_marg_branch[:, PRESENT, PRESENT] * col_mask
    P_I = pair_marg_branch[:, NYI, PRESENT] * col_mask
    P_D = pair_marg_branch[:, PRESENT, DELETED] * col_mask
    P_Ig_real = pair_marg_branch[:, NYI, NYI] + pair_marg_branch[:, DELETED, DELETED]
    # On padded cols: force P_Ig = 1. On real cols: keep value (floored).
    P_Ig = jnp.where(col_mask > 0, jnp.maximum(P_Ig_real, eps), 1.0)
    L = pair_marg_branch.shape[0]
    T = q_tau.shape[1]

    P_state_inner = jnp.zeros((L + 2, N_WFST))
    P_state_inner = P_state_inner.at[0, WFST_S].set(1.0)
    P_state_inner = P_state_inner.at[L + 1, WFST_E].set(1.0)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_M].set(P_M)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_I].set(P_I)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_D].set(P_D)

    q_tau_full = jnp.zeros((L + 2, T))
    q_tau_full = q_tau_full.at[0, 0].set(1.0)
    q_tau_full = q_tau_full.at[L + 1, 0].set(1.0)
    q_tau_full = q_tau_full.at[1:L + 1, :].set(q_tau)
    P_state_tau = P_state_inner[:, :, None] * q_tau_full[:, None, :]

    P_Ig_full = jnp.ones(L + 2)
    P_Ig_full = P_Ig_full.at[1:L + 1].set(P_Ig)
    log_P_Ig = jnp.log(P_Ig_full)
    C = jnp.cumsum(log_P_Ig)
    C_prev = jnp.concatenate([jnp.zeros(1), C[:-1]])

    log_P_tau = jnp.log(jnp.maximum(P_state_tau, eps))
    log_v = log_P_tau - C[:, None, None]
    log_v = jnp.where(P_state_tau > 0, log_v, -jnp.inf)

    def lae_step(carry, x):
        new = jnp.logaddexp(carry, x)
        return new, new

    init = jnp.full((N_WFST, T), -jnp.inf)
    _, cum_log_v = jax.lax.scan(lae_step, init, log_v)
    cum_log_v_prev = jnp.concatenate(
        [jnp.full((1, N_WFST, T), -jnp.inf), cum_log_v[:-1]], axis=0)
    inner = jnp.exp(cum_log_v_prev + C_prev[:, None, None])
    W = jnp.einsum('nsa,ntb->satb', inner[1:], P_state_tau[1:])
    return jnp.sum(log_T_branch * W)


def _entropy_per_column_masked(pair_marg, root_dist, q_cond, tree,
                                edge_mask, col_mask, root_edge_idx):
    """entropy_per_column with edge-axis and column-axis masking.

    Edge mask zeros ghost edges' contribution to the per-edge entropy
    sum. Column mask zeros padded columns of the per-column root term
    AND the per-edge term before summing.

    H_root: extracted from pair_marg[root_edge_idx] (first edge whose
    parent is tree.root — i.e. one of the new-root's outgoing ghost
    edges in the padded case). Under our ghost-leaf clamp convention
    ([1, 1, 1], no evidence) the BP-up-pass from the ghost subtree
    contributes m=1 multiplicatively, so the BP-posterior root marginal
    at the new root equals the BP-posterior at the original root in
    the unpadded tree. The cross-entropy with root_dist is therefore
    the same as in the unpadded computation.
    """
    root_marg_post = pair_marg[root_edge_idx].sum(axis=-1)  # (L, 3)
    safe_root_dist = jnp.where(root_dist > 0.0, root_dist, 1.0)
    log_root_dist = jnp.where(root_dist > 0.0, jnp.log(safe_root_dist), 0.0)
    H_root = -(root_marg_post * log_root_dist).sum(axis=-1)  # (L,) per-col

    # Edge term, with edge mask on the per-edge sum and col mask on per-col.
    safe_q = jnp.where(q_cond > 0.0, q_cond, 1.0)
    log_q = jnp.where(q_cond > 0.0, jnp.log(safe_q), 0.0)
    edge_term = -(pair_marg * log_q).sum(axis=(-1, -2))  # (E, L)
    edge_term_masked = edge_term * edge_mask[:, None]  # (E, L)
    H_edges = edge_term_masked.sum(axis=0)  # (L,)

    H_per_col = (H_root + H_edges) * col_mask
    return H_per_col


def _singlet_root_log_prior_mixdom_masked(root_dist, q_tau, params,
                                           n_dom, n_frag, col_mask,
                                           eps=1e-30):
    """singlet_root_log_prior_mixdom with column masking.

    The original sums root presence/absence indicators and per-column
    tuple log-priors over all L. With padding, padded columns must
    contribute zero. Mask before summing.
    """
    main_lam = params['main_ins']
    main_mu = params['main_del']
    kappa_main = main_lam / main_mu
    dom_lam = params['dom_ins']
    dom_mu = params['dom_del']
    kappa_dom = dom_lam / dom_mu
    dom_w = params['dom_weights']
    frag_w = params['frag_weights']
    ext = params['ext_rates']
    if ext.ndim == 2:
        ext = jax.vmap(jnp.diag)(ext)

    notext = 1.0 - ext.sum(axis=-1)
    p_continue = (frag_w * (1.0 - notext)).sum(axis=-1)
    p_continue_mean = (dom_w * p_continue).sum()
    kappa_eff = kappa_main * (dom_w * kappa_dom).sum()

    log_p_present = jnp.log(jnp.maximum(kappa_eff, eps))
    log_p_absent = jnp.log(jnp.maximum(1.0 - kappa_eff, eps))

    # Mask: only real columns contribute.
    sum_root_P = jnp.sum(root_dist[:, PRESENT] * col_mask)
    sum_root_N = jnp.sum(root_dist[:, NYI] * col_mask)
    presence_term = log_p_present * sum_root_P + log_p_absent * sum_root_N

    pi_tau = (dom_w[:, None] * frag_w).reshape(-1)
    log_pi_tau = jnp.log(jnp.maximum(pi_tau, eps))
    # Per-col tuple term, masked.
    tuple_per_col = (root_dist[:, PRESENT, None] * q_tau
                      * log_pi_tau[None, :]).sum(axis=-1)  # (L,)
    tuple_term = jnp.sum(tuple_per_col * col_mask)

    return presence_term + tuple_term


def _bp_pair_marginals_jax(q_cond, root_dist, leaf_clamp,
                            num_internal, num_nodes, num_edges, root,
                            left_child, right_child,
                            left_edge, right_edge,
                            edge_parent, edge_child,
                            postorder, preorder):
    """Module-level BP that takes tree topology as JIT-traced arguments
    rather than closure-captured numpy arrays. This is critical for
    JIT cache reuse: with the original `bp_pair_marginals(.., tree, le, re)`,
    JAX captures the `tree` NamedTuple's numpy arrays as concrete data,
    making the cache key vary per family. Passing them as arguments
    means JAX traces them as abstract values (shape/dtype only) and
    reuses the compiled function across families with the same shape.
    """
    L = q_cond.shape[1]

    beta = jnp.ones((num_nodes, L, N_Z))
    beta = beta.at[num_internal:].set(leaf_clamp)
    m_edge = jnp.zeros((num_edges, L, N_Z))

    def up_step(carry, v):
        beta, m_edge = carry
        le = left_edge[v]
        re = right_edge[v]
        l_child = left_child[v]
        r_child = right_child[v]
        ml = jnp.einsum('nij,nj->ni', q_cond[le], beta[l_child])
        mr = jnp.einsum('nij,nj->ni', q_cond[re], beta[r_child])
        beta = beta.at[v].set(ml * mr)
        m_edge = m_edge.at[le].set(ml)
        m_edge = m_edge.at[re].set(mr)
        return (beta, m_edge), None

    (beta, m_edge), _ = jax.lax.scan(up_step, (beta, m_edge), postorder)

    log_Z = jnp.log(jnp.einsum('nz,nz->n', root_dist, beta[root]) + 1e-300)

    eta = jnp.zeros((num_nodes, L, N_Z))
    eta = eta.at[root].set(root_dist)

    def down_step(carry, v):
        eta, = carry
        le = left_edge[v]
        re = right_edge[v]
        l_child = left_child[v]
        r_child = right_child[v]
        eta_v = eta[v]
        out_for_left = eta_v * m_edge[re]
        out_for_right = eta_v * m_edge[le]
        eta_l = jnp.einsum('nz,nzj->nj', out_for_left, q_cond[le])
        eta_r = jnp.einsum('nz,nzj->nj', out_for_right, q_cond[re])
        eta = eta.at[l_child].set(eta_l)
        eta = eta.at[r_child].set(eta_r)
        return (eta,), None

    (eta,), _ = jax.lax.scan(down_step, (eta,), preorder)

    eta_p = eta[edge_parent]
    beta_p = beta[edge_parent]
    beta_c = beta[edge_child]
    m_c = m_edge

    safe_m = jnp.where(m_c > 0.0, m_c, 1.0)
    outgoing = eta_p * (beta_p / safe_m) * (m_c > 0.0)
    joint = (outgoing[..., None] * q_cond * beta_c[..., None, :])
    joint_sum = joint.sum(axis=(-1, -2), keepdims=True)
    pair_marg = joint / jnp.where(joint_sum > 0.0, joint_sum, 1.0)
    return pair_marg, log_Z


def _bp_pair_marginals_jax_log(q_cond, root_dist, leaf_clamp,
                                  num_internal, num_nodes, num_edges, root,
                                  left_child, right_child,
                                  left_edge, right_edge,
                                  edge_parent, edge_child,
                                  postorder, preorder):
    """Log-space BP — numerically stable for deep trees / many cols.

    Equivalent to ``_bp_pair_marginals_jax`` but operates on log
    probabilities throughout the up- and down-passes, avoiding the
    underflow / catastrophic-cancellation issues that broke the
    backward pass on big Pfam families (e.g. 537-leaf PF00003 padded
    to bin 1024).

    Returns the SAME (pair_marg, log_Z) shapes as the linear-space
    version: pair_marg is a (num_edges, L, N_Z, N_Z) array of
    NORMALISED posterior pair-marginals (real-space probabilities, not
    logs), and log_Z is (L,) — log evidence per column.
    """
    NEG_INF = -1e10  # large finite negative; gradient-safe substitute for -inf
    L = q_cond.shape[1]

    safe_q = jnp.where(q_cond > 0.0, q_cond, 1.0)
    log_q_cond = jnp.where(q_cond > 0.0, jnp.log(safe_q), NEG_INF)

    safe_clamp = jnp.where(leaf_clamp > 0.0, leaf_clamp, 1.0)
    log_clamp = jnp.where(leaf_clamp > 0.0, jnp.log(safe_clamp), NEG_INF)

    safe_root = jnp.where(root_dist > 0.0, root_dist, 1.0)
    log_root_dist = jnp.where(root_dist > 0.0, jnp.log(safe_root), NEG_INF)

    log_beta = jnp.full((num_nodes, L, N_Z), NEG_INF)
    log_beta = log_beta.at[num_internal:].set(log_clamp)
    log_m_edge = jnp.full((num_edges, L, N_Z), NEG_INF)

    def up_step(carry, v):
        log_beta, log_m_edge = carry
        le = left_edge[v]
        re = right_edge[v]
        l_child = left_child[v]
        r_child = right_child[v]
        # log_ml[col, parent_state]
        #   = logsumexp_{cs}(log_q_cond[le, col, ps, cs] + log_beta[l_child, col, cs])
        log_ml = jax.scipy.special.logsumexp(
            log_q_cond[le] + log_beta[l_child][:, None, :], axis=-1)
        log_mr = jax.scipy.special.logsumexp(
            log_q_cond[re] + log_beta[r_child][:, None, :], axis=-1)
        log_beta = log_beta.at[v].set(log_ml + log_mr)
        log_m_edge = log_m_edge.at[le].set(log_ml)
        log_m_edge = log_m_edge.at[re].set(log_mr)
        return (log_beta, log_m_edge), None

    (log_beta, log_m_edge), _ = jax.lax.scan(
        up_step, (log_beta, log_m_edge), postorder)

    # log_Z[col] = logsumexp_{state}(log_root_dist[col, state] + log_beta[root, col, state])
    log_Z = jax.scipy.special.logsumexp(
        log_root_dist + log_beta[root], axis=-1)

    log_eta = jnp.full((num_nodes, L, N_Z), NEG_INF)
    log_eta = log_eta.at[root].set(log_root_dist)

    def down_step(carry, v):
        log_eta, = carry
        le = left_edge[v]
        re = right_edge[v]
        l_child = left_child[v]
        r_child = right_child[v]
        log_eta_v = log_eta[v]
        # out_for_left = eta_v * m_edge[re] in log space.
        log_out_for_left = log_eta_v + log_m_edge[re]    # (L, parent_state)
        log_out_for_right = log_eta_v + log_m_edge[le]
        # eta_l[col, cs] = sum_{ps} out_for_left[col, ps] * q_cond[le, col, ps, cs].
        log_eta_l = jax.scipy.special.logsumexp(
            log_out_for_left[:, :, None] + log_q_cond[le], axis=1)
        log_eta_r = jax.scipy.special.logsumexp(
            log_out_for_right[:, :, None] + log_q_cond[re], axis=1)
        log_eta = log_eta.at[l_child].set(log_eta_l)
        log_eta = log_eta.at[r_child].set(log_eta_r)
        return (log_eta,), None

    (log_eta,), _ = jax.lax.scan(down_step, (log_eta,), preorder)

    # Pair marginals: joint[e, col, ps, cs] (log) = log_eta[parent]
    #   + log_beta[parent] - log_m_edge[e] + log_q_cond + log_beta[child]
    # The (log_beta_p - log_m_c) part = log of the OPPOSITE sibling's
    # message, which is what should propagate down this edge.
    log_eta_p = log_eta[edge_parent]              # (E, L, ps)
    log_beta_p = log_beta[edge_parent]            # (E, L, ps)
    log_beta_c = log_beta[edge_child]             # (E, L, cs)
    log_m_c = log_m_edge                          # (E, L, ps)
    log_outgoing = log_eta_p + log_beta_p - log_m_c   # (E, L, ps)
    log_joint = (log_outgoing[..., None]
                  + log_q_cond
                  + log_beta_c[..., None, :])      # (E, L, ps, cs)
    # Normalise per (edge, col).  If logsumexp = NEG_INF (no valid path),
    # pair_marg = 0.  Use double-where to keep gradient finite.
    log_norm = jax.scipy.special.logsumexp(log_joint, axis=(-1, -2),
                                              keepdims=True)
    log_pair = log_joint - jnp.where(log_norm > NEG_INF / 2,
                                          log_norm, 0.0)
    pair_marg = jnp.where(log_norm > NEG_INF / 2,
                            jnp.exp(log_pair),
                            0.0)
    return pair_marg, log_Z


def _entropy_per_column_masked_argonly(pair_marg, root_dist, q_cond,
                                        edge_mask, col_mask, root_edge_idx_arr):
    """Same as _entropy_per_column_masked but with root_edge_idx as a JIT
    argument (jnp scalar) rather than a Python int.
    """
    root_marg_post = pair_marg[root_edge_idx_arr].sum(axis=-1)  # (L, 3)
    safe_root_dist = jnp.where(root_dist > 0.0, root_dist, 1.0)
    log_root_dist = jnp.where(root_dist > 0.0, jnp.log(safe_root_dist), 0.0)
    H_root = -(root_marg_post * log_root_dist).sum(axis=-1)

    safe_q = jnp.where(q_cond > 0.0, q_cond, 1.0)
    log_q = jnp.where(q_cond > 0.0, jnp.log(safe_q), 0.0)
    edge_term = -(pair_marg * log_q).sum(axis=(-1, -2))
    edge_term_masked = edge_term * edge_mask[:, None]
    H_edges = edge_term_masked.sum(axis=0)

    H_per_col = (H_root + H_edges) * col_mask
    return H_per_col


def _make_neg_elbo_jit(L, n_dom, n_frag,
                        num_internal, num_nodes, num_edges, root,
                        n_iter, lr):
    """Build a JIT-compiled neg_elbo + Adam loop function.

    Cache key depends only on the static Python scalars (n_iter, lr,
    L, n_dom, n_frag, num_internal, num_edges, root) and the abstract
    shapes of array arguments. The number of "real" vs "ghost" edges
    is conveyed via `edge_mask` (an array argument), so the cache
    reuses across all families in the same (n_leaves_bin, L_bin) combo
    regardless of the original tree's leaf count.

    edge_logits has shape (num_edges, L, 2) — per-(edge, col, 2) full
    per-column flexibility (no longer tied across columns).  Tying
    introduced variational bias because the data-conditioned posterior
    varies per column with the leaf observations.

    For ghost edges the per-edge logit is unused (q_cond is forced to
    identity inside the function). To prevent Adam from drifting these
    unused entries, the gradient is masked by edge_mask before the
    update.
    """
    eye_3_const = jnp.eye(3, dtype=jnp.float64)

    def _build_q_cond(edge_logits, edge_mask):
        """Build the per-(edge, col) q_cond. For real edges (mask=1), use the
        learned logits; for ghost edges (mask=0), use identity."""
        # edge_logits: (E, L, 2) — per-(edge, col) free logits.
        learned_q = make_q_conditionals(edge_logits)  # (E, L, 3, 3)
        # Identity for ghost edges.
        identity_q = jnp.broadcast_to(eye_3_const, (num_edges, L, 3, 3))
        em = edge_mask[:, None, None, None]
        q_cond = em * learned_q + (1.0 - em) * identity_q
        return q_cond

    def neg_elbo_inner(edge_logits, root_logit, tuple_logits,
                       leaf_clamp, edge_lengths,
                       L_sub, classdist, params_flat,
                       edge_mask, col_mask, root_edge_idx_arr,
                       left_child, right_child, left_edge, right_edge,
                       edge_parent, edge_child, postorder, preorder):
        params = _params_from_flat(params_flat, n_dom, n_frag)

        q_cond = _build_q_cond(edge_logits, edge_mask)

        root_logits = jnp.broadcast_to(root_logit, (L,))
        root_dist = make_root_dist(root_logits)

        pair_marg, log_Z = _bp_pair_marginals_jax(
            q_cond, root_dist, leaf_clamp,
            num_internal, num_nodes, num_edges, root,
            left_child, right_child, left_edge, right_edge,
            edge_parent, edge_child, postorder, preorder)

        q_tau = make_tuple_dist(tuple_logits)

        def log_T_for(t):
            T_pair = mixdom_reduced_T_pair(params, t)
            return jnp.log(jnp.maximum(T_pair, 1e-300))
        log_T_per_edge = jax.vmap(log_T_for)(edge_lengths)
        branch_LLs = jax.vmap(
            lambda pm, lt: _expected_branch_LL_mixdom_masked(
                pm, q_tau, lt, col_mask))(pair_marg, log_T_per_edge)
        sum_branch_LL = jnp.sum(branch_LLs * edge_mask)

        log_L_per_tuple = class_marginalised_sub_LL_per_tuple(L_sub, classdist)
        q_tau_d = q_tau.reshape(L, n_dom, n_frag)
        sub_per_col = (q_tau_d * log_L_per_tuple).sum(axis=(-1, -2))
        sum_sub_LL = jnp.sum(sub_per_col * col_mask)

        H_inner = jnp.sum(_entropy_per_column_masked_argonly(
            pair_marg, root_dist, q_cond,
            edge_mask, col_mask, root_edge_idx_arr))
        log_q_tau_arr = jnp.log(jnp.maximum(q_tau, 1e-30))
        H_tau_per_col = -(q_tau * log_q_tau_arr).sum(axis=-1)
        H_tau = jnp.sum(H_tau_per_col * col_mask)

        log_prior = _singlet_root_log_prior_mixdom_masked(
            root_dist, q_tau, params, n_dom, n_frag, col_mask)

        return -(sum_branch_LL + sum_sub_LL + log_prior + H_inner + H_tau
                 + jnp.sum(log_Z * col_mask))

    @jax.jit
    def run_adam(edge_logits, root_logit, tuple_logits,
                  leaf_clamp, edge_lengths,
                  L_sub, classdist, params_flat,
                  edge_mask, col_mask, root_edge_idx_arr,
                  left_child, right_child, left_edge, right_edge,
                  edge_parent, edge_child, postorder, preorder):
        optimizer = optax.adam(lr)
        state = optimizer.init((edge_logits, root_logit, tuple_logits))

        def step(carry, _):
            p, state = carry
            grad_fn = jax.grad(neg_elbo_inner, argnums=(0, 1, 2))
            g = grad_fn(*p, leaf_clamp, edge_lengths,
                         L_sub, classdist, params_flat,
                         edge_mask, col_mask, root_edge_idx_arr,
                         left_child, right_child, left_edge, right_edge,
                         edge_parent, edge_child, postorder, preorder)
            # Mask out gradient on ghost edges — they're "frozen" at 0.
            g_edge, g_root, g_tuple = g
            g_edge_masked = g_edge * edge_mask[:, None, None]
            g = (g_edge_masked, g_root, g_tuple)
            u, state = optimizer.update(g, state)
            new_p = optax.apply_updates(p, u)
            return (new_p, state), None

        (final_p, _), _ = jax.lax.scan(
            step, ((edge_logits, root_logit, tuple_logits), state),
            None, length=n_iter)
        return final_p

    @jax.jit
    def compute_neg_elbo(edge_logits, root_logit, tuple_logits,
                          leaf_clamp, edge_lengths,
                          L_sub, classdist, params_flat,
                          edge_mask, col_mask, root_edge_idx_arr,
                          left_child, right_child, left_edge, right_edge,
                          edge_parent, edge_child, postorder, preorder):
        return neg_elbo_inner(edge_logits, root_logit, tuple_logits,
                               leaf_clamp, edge_lengths,
                               L_sub, classdist, params_flat,
                               edge_mask, col_mask, root_edge_idx_arr,
                               left_child, right_child, left_edge, right_edge,
                               edge_parent, edge_child, postorder, preorder)

    @jax.jit
    def compute_pair_marg_q_tau(edge_logits, root_logit, tuple_logits,
                                  leaf_clamp, edge_mask,
                                  left_child, right_child, left_edge, right_edge,
                                  edge_parent, edge_child, postorder, preorder):
        q_cond = _build_q_cond(edge_logits, edge_mask)
        root_logits = jnp.broadcast_to(root_logit, (L,))
        root_dist = make_root_dist(root_logits)
        pair_marg, _ = _bp_pair_marginals_jax(
            q_cond, root_dist, leaf_clamp,
            num_internal, num_nodes, num_edges, root,
            left_child, right_child, left_edge, right_edge,
            edge_parent, edge_child, postorder, preorder)
        q_tau = make_tuple_dist(tuple_logits)
        return pair_marg, q_tau

    return run_adam, compute_neg_elbo, compute_pair_marg_q_tau


# Cache JIT-compiled functions per (bin, n_iter, lr) signature so
# repeated calls with the same shape reuse the compiled artifact.
_JIT_CACHE = {}


def _params_to_flat(params, n_dom, n_frag):
    """Flatten the params dict into a stable tuple of jax arrays for JIT."""
    return (
        jnp.asarray(params['main_ins'], dtype=jnp.float64),
        jnp.asarray(params['main_del'], dtype=jnp.float64),
        jnp.asarray(params['dom_ins'], dtype=jnp.float64),
        jnp.asarray(params['dom_del'], dtype=jnp.float64),
        jnp.asarray(params['dom_weights'], dtype=jnp.float64),
        jnp.asarray(params['frag_weights'], dtype=jnp.float64),
        jnp.asarray(params['ext_rates'], dtype=jnp.float64),
        jnp.asarray(params['classdist'], dtype=jnp.float64),
        jnp.asarray(params['class_pis'], dtype=jnp.float64),
        jnp.asarray(params['class_S_exch'], dtype=jnp.float64),
    )


def _params_from_flat(flat, n_dom, n_frag):
    """Restore the params dict from the flat tuple."""
    (main_ins, main_del, dom_ins, dom_del, dom_weights, frag_weights,
     ext_rates, classdist, class_pis, class_S_exch) = flat
    return {
        'main_ins': main_ins,
        'main_del': main_del,
        'dom_ins': dom_ins,
        'dom_del': dom_del,
        'dom_weights': dom_weights,
        'frag_weights': frag_weights,
        'ext_rates': ext_rates,
        'classdist': classdist,
        'class_pis': class_pis,
        'class_S_exch': class_S_exch,
    }


def _compute_W_tensor_masked(pair_marg_branch, q_tau, T, col_mask, eps=1e-30):
    """Column-masked _compute_W_tensor for post-hoc per-branch W extraction.

    Same masking convention as _expected_branch_LL_mixdom_masked.
    """
    P_M = pair_marg_branch[:, PRESENT, PRESENT] * col_mask
    P_I = pair_marg_branch[:, NYI, PRESENT] * col_mask
    P_D = pair_marg_branch[:, PRESENT, DELETED] * col_mask
    P_Ig_real = pair_marg_branch[:, NYI, NYI] + pair_marg_branch[:, DELETED, DELETED]
    P_Ig = jnp.where(col_mask > 0, jnp.maximum(P_Ig_real, eps), 1.0)
    L = pair_marg_branch.shape[0]

    P_state_inner = jnp.zeros((L + 2, 5))
    P_state_inner = P_state_inner.at[0, WFST_S].set(1.0)
    P_state_inner = P_state_inner.at[L + 1, WFST_E].set(1.0)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_M].set(P_M)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_I].set(P_I)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_D].set(P_D)

    q_tau_full = jnp.zeros((L + 2, T))
    q_tau_full = q_tau_full.at[0, 0].set(1.0)
    q_tau_full = q_tau_full.at[L + 1, 0].set(1.0)
    q_tau_full = q_tau_full.at[1:L + 1, :].set(q_tau)
    P_state_tau = P_state_inner[:, :, None] * q_tau_full[:, None, :]

    P_Ig_full = jnp.ones(L + 2)
    P_Ig_full = P_Ig_full.at[1:L + 1].set(P_Ig)
    log_P_Ig = jnp.log(P_Ig_full)
    C = jnp.cumsum(log_P_Ig)
    C_prev = jnp.concatenate([jnp.zeros(1), C[:-1]])

    log_P_tau = jnp.log(jnp.maximum(P_state_tau, eps))
    log_v = log_P_tau - C[:, None, None]
    log_v = jnp.where(P_state_tau > 0, log_v, -jnp.inf)

    def lae_step(carry, x):
        new = jnp.logaddexp(carry, x)
        return new, new

    init = jnp.full((5, T), -jnp.inf)
    _, cum_log_v = jax.lax.scan(lae_step, init, log_v)
    cum_log_v_prev = jnp.concatenate(
        [jnp.full((1, 5, T), -jnp.inf), cum_log_v[:-1]], axis=0)
    inner = jnp.exp(cum_log_v_prev + C_prev[:, None, None])
    return jnp.einsum('nsa,ntb->satb', inner[1:], P_state_tau[1:])


# ---------------------------------------------------------------------------
# Per-family E-step.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FamilyEStepStats:
    """Sufficient statistics from one family's E-step."""
    elbo: float
    n_cols: int
    n_edges: int
    # Per-branch W tensor (E, 5, T, 5, T).
    W_per_branch: np.ndarray
    # Per-column tuple categorical (L, T).
    q_tau: np.ndarray
    # Per-column class posterior (L, n_classes).
    q_c: np.ndarray
    # Per-column root marginal (L, 3) for {N, P, D}.
    root_marg: np.ndarray
    # Per-edge branch lengths (E,).
    edge_lengths: np.ndarray


def fit_family_estep(binary_tree, leaf_present, params, sub_LL_per_class,
                      n_iter=100, lr=0.05, seed=0,
                      tuple_init_from_sub=True, use_padding=True):
    """Run the per-family E-step: Adam ascent on q at fixed θ.

    Returns FamilyEStepStats including the per-branch W tensor + q_tau.

    Args:
        binary_tree: BinaryTree (leaf clamps determine presence pattern).
        leaf_present: (n_leaves, L) {0, 1} indicator.
        params: MixDom params dict.
        sub_LL_per_class: (L, n_classes) per-column per-class Felsenstein
            up-pass likelihoods (treated as constants during E-step).
        n_iter: number of Adam steps.
        lr: learning rate.
        seed: rng seed for jitter.
        tuple_init_from_sub: bias initial tuple logits toward
            substitution-likelihood-favoured fragchars.
        use_padding: pad both tree (leaf count) and column count to
            geometric bin sizes for JIT cache reuse. Padded entries
            contribute exactly zero to the ELBO (verified by smoke
            tests). Default True; set to False to disable for testing.

    Returns:
        FamilyEStepStats with n_cols and n_edges set to the ORIGINAL
        (unpadded) values, and W_per_branch / q_tau / q_c / root_marg
        sliced to the real columns / real edges.
    """
    L_real = int(leaf_present.shape[1])
    n_leaves_real = int(binary_tree.num_leaves)
    n_edges_real = int(binary_tree.num_edges)
    n_dom = params['dom_ins'].shape[0]
    n_frag = params['frag_weights'].shape[1]
    T = n_dom * n_frag

    # Clip degenerate branch lengths.
    edge_lengths_real = np.maximum(np.asarray(binary_tree.edge_length), 1e-3)
    binary_tree = binary_tree._replace(edge_length=edge_lengths_real)

    if use_padding:
        target_leaves = _pad_to_bin(n_leaves_real, _LEAF_BINS)
        target_L = _pad_to_bin(L_real, _COL_BINS)
        padded_tree, edge_mask, ghost_leaf_indices, leaf_index_map, _ = \
            _pad_tree_with_ghosts(binary_tree, target_leaves)
        leaf_present_pad, sub_LL_pad, col_mask = _pad_columns(
            leaf_present, sub_LL_per_class, target_L)
    else:
        target_leaves = n_leaves_real
        target_L = L_real
        padded_tree = binary_tree
        edge_mask = np.ones(n_edges_real, dtype=np.float64)
        ghost_leaf_indices = np.zeros(0, dtype=np.int32)
        leaf_index_map = np.arange(
            int(binary_tree.num_internal),
            int(binary_tree.num_internal) + n_leaves_real, dtype=np.int32)
        leaf_present_pad = np.asarray(leaf_present)
        sub_LL_pad = np.asarray(sub_LL_per_class)
        col_mask = np.ones(L_real, dtype=np.float64)

    L = target_L  # all internal computations use the padded width.

    le, re = edge_lookup(padded_tree)

    # Build leaf clamps in the padded-tree leaf ordering.
    pad_n_internal = int(padded_tree.num_internal)
    leaf_clamp_full = np.zeros((target_leaves, L, 3), dtype=np.float64)
    orig_clamp = np.asarray(leaf_clamp_to_beta(leaf_present_pad))
    for i in range(n_leaves_real):
        leaf_local = int(leaf_index_map[i]) - pad_n_internal
        leaf_clamp_full[leaf_local] = orig_clamp[i]
    for gi in ghost_leaf_indices:
        leaf_local = int(gi) - pad_n_internal
        leaf_clamp_full[leaf_local] = 1.0
    leaf_clamp_jnp = jnp.asarray(leaf_clamp_full)

    edge_mask_jnp = jnp.asarray(edge_mask)
    col_mask_jnp = jnp.asarray(col_mask)

    n_ghost_edges = int(padded_tree.num_edges) - n_edges_real

    rng = np.random.default_rng(seed)
    # edge_logits has shape (pad_n_edges, L, 2) — per-(edge, col, 2) full
    # per-column flexibility on the presence-state axis (NYI/P/D).
    # Tying across columns (formerly (E, 2)) introduced variational bias
    # because the data-conditioned posterior varies per column.  The
    # (d, f) tuple axis is already per-column via tuple_logits.
    edge_logits_real = rng.standard_normal((n_edges_real, L, 2)) * 0.1
    edge_logits_full = np.zeros((int(padded_tree.num_edges), L, 2),
                                  dtype=np.float64)
    edge_logits_full[:n_edges_real] = edge_logits_real
    edge_logits = jnp.asarray(edge_logits_full, dtype=jnp.float64)
    root_logit = jnp.asarray(rng.standard_normal(()) * 0.1, dtype=jnp.float64)

    if tuple_init_from_sub:
        L_sub_jnp = jnp.asarray(sub_LL_pad)
        log_L_per_tuple_init = class_marginalised_sub_LL_per_tuple(
            L_sub_jnp, params['classdist'])
        tuple_logits_init = (log_L_per_tuple_init / 2.0).reshape(L, T)
        tuple_logits = tuple_logits_init + jnp.asarray(
            rng.standard_normal((L, T)) * 0.05, dtype=jnp.float64)
    else:
        tuple_logits = jnp.asarray(
            rng.standard_normal((L, T)) * 0.1, dtype=jnp.float64)

    L_sub_jnp = jnp.asarray(sub_LL_pad)
    classdist_jnp = jnp.asarray(params['classdist'])

    # Tree-topology arrays as jnp (passed as JIT arguments — keeps the
    # cache key shape-only across families with the same bin).
    left_child_jnp = jnp.asarray(padded_tree.left_child)
    right_child_jnp = jnp.asarray(padded_tree.right_child)
    left_edge_jnp = jnp.asarray(le)
    right_edge_jnp = jnp.asarray(re)
    edge_parent_jnp = jnp.asarray(padded_tree.edge_parent)
    edge_child_jnp = jnp.asarray(padded_tree.edge_child)
    postorder_jnp = jnp.asarray(padded_tree.postorder_internal)
    preorder_jnp = jnp.asarray(padded_tree.preorder_internal)
    edge_lengths_jnp = jnp.asarray(padded_tree.edge_length)

    # Find root_edge_idx for the H_root term: first edge whose parent is
    # the root. Pass as jnp scalar to keep cache shape-only.
    edge_parent_arr_np = np.asarray(padded_tree.edge_parent)
    root_edge_idx_int = int(
        np.where(edge_parent_arr_np == padded_tree.root)[0][0])
    root_edge_idx_arr = jnp.asarray(root_edge_idx_int)

    params_flat = _params_to_flat(params, n_dom, n_frag)

    # Cache key: scalars that determine the JIT compile (shapes/sizes).
    # Note: n_edges_real / n_ghost_edges are NOT in the key — they're
    # conveyed via edge_mask (an array argument), so families with
    # different real-edge counts but the same padded shape reuse the
    # same compiled function.
    jit_key = (
        L, n_dom, n_frag,
        int(padded_tree.num_internal), int(padded_tree.num_nodes),
        int(padded_tree.num_edges), int(padded_tree.root),
        int(n_iter), float(lr),
    )
    if jit_key not in _JIT_CACHE:
        _JIT_CACHE[jit_key] = _make_neg_elbo_jit(
            L, n_dom, n_frag,
            int(padded_tree.num_internal), int(padded_tree.num_nodes),
            int(padded_tree.num_edges), int(padded_tree.root),
            int(n_iter), float(lr))
    run_adam, compute_neg_elbo, compute_pair_marg_q_tau = _JIT_CACHE[jit_key]

    edge_logits_f, root_logit_f, tuple_logits_f = run_adam(
        edge_logits, root_logit, tuple_logits,
        leaf_clamp_jnp, edge_lengths_jnp,
        L_sub_jnp, classdist_jnp, params_flat,
        edge_mask_jnp, col_mask_jnp, root_edge_idx_arr,
        left_child_jnp, right_child_jnp, left_edge_jnp, right_edge_jnp,
        edge_parent_jnp, edge_child_jnp, postorder_jnp, preorder_jnp)

    final_elbo = -float(compute_neg_elbo(
        edge_logits_f, root_logit_f, tuple_logits_f,
        leaf_clamp_jnp, edge_lengths_jnp,
        L_sub_jnp, classdist_jnp, params_flat,
        edge_mask_jnp, col_mask_jnp, root_edge_idx_arr,
        left_child_jnp, right_child_jnp, left_edge_jnp, right_edge_jnp,
        edge_parent_jnp, edge_child_jnp, postorder_jnp, preorder_jnp))

    pair_marg, q_tau = compute_pair_marg_q_tau(
        edge_logits_f, root_logit_f, tuple_logits_f,
        leaf_clamp_jnp, edge_mask_jnp,
        left_child_jnp, right_child_jnp, left_edge_jnp, right_edge_jnp,
        edge_parent_jnp, edge_child_jnp, postorder_jnp, preorder_jnp)

    # Per-branch W tensors. Compute only for REAL edges; pad at the
    # original (unpadded) edge ordering. The W tensors are returned at
    # the real-column dimension (sliced from L_real to drop padding).
    W_per_branch = []
    for e in range(n_edges_real):
        W = _compute_W_tensor_masked(
            pair_marg[e], q_tau, T, col_mask_jnp)
        W_per_branch.append(np.asarray(W))
    W_per_branch = np.stack(W_per_branch, axis=0)  # (E_real, 5, T, 5, T)

    # Slice all per-column outputs back to the real-column dimension.
    q_tau_real = np.asarray(q_tau)[:L_real]
    q_tau_d = q_tau_real.reshape(L_real, n_dom, n_frag)
    cd = np.asarray(params['classdist'])
    L_sub = np.asarray(sub_LL_per_class)[:L_real]
    n_classes = L_sub.shape[1]
    joint = cd[None, :, :, :] * L_sub[:, None, None, :]
    denom = joint.sum(axis=-1, keepdims=True)
    safe_denom = np.where(denom > 0, denom, 1.0)
    q_c_given_df = np.where(denom > 0, joint / safe_denom, 0.0)
    q_c = (q_tau_d[..., None] * q_c_given_df).sum(axis=(1, 2))

    root_marg = np.asarray(pair_marg[root_edge_idx_int]).sum(axis=-1)[:L_real]

    return FamilyEStepStats(
        elbo=final_elbo,
        n_cols=L_real,
        n_edges=n_edges_real,
        W_per_branch=W_per_branch,
        q_tau=q_tau_real,
        q_c=q_c,
        root_marg=root_marg,
        edge_lengths=np.asarray(edge_lengths_real),
    )


# ---------------------------------------------------------------------------
# Hold-out leaf prediction (used by varanc_presence_mixdom_benchmark.py).
# Reuses fit_family_estep's shape-keyed JIT machinery (same _JIT_CACHE
# key) so a sweep over benchmark families pays JIT compile cost only
# once per (target_L, target_n_leaves) bucket — replaces the legacy
# closure-JIT path that re-traced from scratch on every family.
# ---------------------------------------------------------------------------


def predict_holdout_mixdom(binary_tree, leaf_present, holdout_idx, params,
                             sub_LL_per_class, n_iter=150, lr=0.05, seed=0,
                             init_edge_logits=None, init_root_logit=None):
    """Predict P(held-out leaf is present | rest, params) per MSA column.

    Mirrors the legacy `_predict_one_holdout_mixdom` in
    `experiments/varanc_presence_mixdom_benchmark.py`, but routes through
    the shape-keyed JIT factory _make_neg_elbo_jit so the JIT cache
    reuses across all families in the same padded bucket.

    The leaf_clamp for the held-out leaf is set to uniform [1,1,1] (no
    observation), matching the benchmark's "treated as missing" setup.

    The ELBO uses the per-tuple substitution-LL aggregation
    (class_marginalised_sub_LL_per_tuple), not the legacy per-fragchar
    aggregation. For models where classdist is constant across domains
    (e.g. d3f1 auto-promoted to dummy C=3) the two formulations are
    mathematically equivalent up to floating-point rounding. For
    genuinely class-aware MixDom2 models (d3f1c3, d3f3c27, …) the
    per-tuple version is strictly more accurate.

    Args:
        binary_tree: BinaryTree (full tree including held-out leaf).
        leaf_present: (n_leaves, L) {0, 1} per column. Held-out leaf row
            is OVERRIDDEN to uniform inside this function — its actual
            value is irrelevant.
        holdout_idx: index of the held-out leaf in binary_tree.leaf_names.
        params: MixDom params dict (parse_mixdom_params_npz output).
        sub_LL_per_class: (L, n_classes) per-column per-class Felsenstein
            up-pass likelihood at the holdout-as-missing setting.
        n_iter, lr, seed: Adam knobs. Default n_iter=150 / lr=0.05
            matches the legacy benchmark.
        init_edge_logits: optional (n_edges_real, 2) seed for the inner
            3-state q. Caller (typically the benchmark) supplies the
            Fitch-seeded init from `fitch_seeded_init` for fast
            convergence; default random init only converges with many
            more Adam steps.
        init_root_logit: optional scalar seed for the root logit.

    Returns:
        p_present: (L_real,) per-column P(held-out = present).
    """
    L_real = int(leaf_present.shape[1])
    n_leaves_real = int(binary_tree.num_leaves)
    n_edges_real = int(binary_tree.num_edges)
    n_dom = params['dom_ins'].shape[0]
    n_frag = params['frag_weights'].shape[1]
    T = n_dom * n_frag

    edge_lengths_real = np.maximum(np.asarray(binary_tree.edge_length), 1e-3)
    binary_tree = binary_tree._replace(edge_length=edge_lengths_real)

    target_leaves = _pad_to_bin(n_leaves_real, _LEAF_BINS)
    target_L = _pad_to_bin(L_real, _COL_BINS)
    padded_tree, edge_mask, ghost_leaf_indices, leaf_index_map, _ = \
        _pad_tree_with_ghosts(binary_tree, target_leaves)
    leaf_present_pad, sub_LL_pad, col_mask = _pad_columns(
        leaf_present, sub_LL_per_class, target_L)
    L = target_L
    le, re = edge_lookup(padded_tree)

    pad_n_internal = int(padded_tree.num_internal)
    leaf_clamp_full = np.zeros((target_leaves, L, 3), dtype=np.float64)
    orig_clamp = np.asarray(leaf_clamp_to_beta(leaf_present_pad))
    holdout_padded_leaf = int(leaf_index_map[holdout_idx])  # node id in padded tree
    for i in range(n_leaves_real):
        leaf_local = int(leaf_index_map[i]) - pad_n_internal
        if i == holdout_idx:
            leaf_clamp_full[leaf_local] = 1.0  # uniform = no observation
        else:
            leaf_clamp_full[leaf_local] = orig_clamp[i]
    for gi in ghost_leaf_indices:
        leaf_local = int(gi) - pad_n_internal
        leaf_clamp_full[leaf_local] = 1.0
    leaf_clamp_jnp = jnp.asarray(leaf_clamp_full)

    edge_mask_jnp = jnp.asarray(edge_mask)
    col_mask_jnp = jnp.asarray(col_mask)

    rng = np.random.default_rng(seed)
    if init_edge_logits is None:
        edge_logits_real = rng.standard_normal((n_edges_real, L, 2)) * 0.1
    else:
        init_arr = np.asarray(init_edge_logits)
        # Accept either tied (E, 2) — broadcast — or untied (E, L, 2).
        if init_arr.shape == (n_edges_real, 2):
            edge_logits_real = np.broadcast_to(
                init_arr[:, None, :], (n_edges_real, L, 2)).copy()
        elif init_arr.shape == (n_edges_real, L, 2):
            edge_logits_real = init_arr.copy()
        else:
            raise ValueError(
                f"init_edge_logits shape {init_arr.shape} not in "
                f"{{ ({n_edges_real}, 2), ({n_edges_real}, {L}, 2) }}")
        # Add small jitter on top so repeated seeds aren't identical traces.
        edge_logits_real = (edge_logits_real
                             + rng.standard_normal((n_edges_real, L, 2)) * 0.05)
    edge_logits_full = np.zeros((int(padded_tree.num_edges), L, 2),
                                  dtype=np.float64)
    edge_logits_full[:n_edges_real] = edge_logits_real
    edge_logits = jnp.asarray(edge_logits_full, dtype=jnp.float64)
    if init_root_logit is None:
        root_logit = jnp.asarray(
            rng.standard_normal(()) * 0.1, dtype=jnp.float64)
    else:
        root_logit = jnp.asarray(float(init_root_logit), dtype=jnp.float64)

    L_sub_jnp = jnp.asarray(sub_LL_pad)
    log_L_per_tuple_init = class_marginalised_sub_LL_per_tuple(
        L_sub_jnp, params['classdist'])
    tuple_logits_init = (log_L_per_tuple_init / 2.0).reshape(L, T)
    tuple_logits = tuple_logits_init + jnp.asarray(
        rng.standard_normal((L, T)) * 0.05, dtype=jnp.float64)

    classdist_jnp = jnp.asarray(params['classdist'])

    left_child_jnp = jnp.asarray(padded_tree.left_child)
    right_child_jnp = jnp.asarray(padded_tree.right_child)
    left_edge_jnp = jnp.asarray(le)
    right_edge_jnp = jnp.asarray(re)
    edge_parent_jnp = jnp.asarray(padded_tree.edge_parent)
    edge_child_jnp = jnp.asarray(padded_tree.edge_child)
    postorder_jnp = jnp.asarray(padded_tree.postorder_internal)
    preorder_jnp = jnp.asarray(padded_tree.preorder_internal)
    edge_lengths_jnp = jnp.asarray(padded_tree.edge_length)

    edge_parent_arr_np = np.asarray(padded_tree.edge_parent)
    root_edge_idx_int = int(
        np.where(edge_parent_arr_np == padded_tree.root)[0][0])
    root_edge_idx_arr = jnp.asarray(root_edge_idx_int)

    params_flat = _params_to_flat(params, n_dom, n_frag)

    # Same JIT key as fit_family_estep — cache hit when both have run on
    # this bucket already. Default n_iter/lr also match fit_family_estep
    # so a training run can warm the cache used by the benchmark.
    jit_key = (L, n_dom, n_frag,
                int(padded_tree.num_internal), int(padded_tree.num_nodes),
                int(padded_tree.num_edges), int(padded_tree.root),
                int(n_iter), float(lr))
    if jit_key not in _JIT_CACHE:
        _JIT_CACHE[jit_key] = _make_neg_elbo_jit(
            L, n_dom, n_frag,
            int(padded_tree.num_internal), int(padded_tree.num_nodes),
            int(padded_tree.num_edges), int(padded_tree.root),
            int(n_iter), float(lr))
    run_adam, _, compute_pair_marg_q_tau = _JIT_CACHE[jit_key]

    edge_logits_f, root_logit_f, tuple_logits_f = run_adam(
        edge_logits, root_logit, tuple_logits,
        leaf_clamp_jnp, edge_lengths_jnp,
        L_sub_jnp, classdist_jnp, params_flat,
        edge_mask_jnp, col_mask_jnp, root_edge_idx_arr,
        left_child_jnp, right_child_jnp, left_edge_jnp, right_edge_jnp,
        edge_parent_jnp, edge_child_jnp, postorder_jnp, preorder_jnp)

    pair_marg, _ = compute_pair_marg_q_tau(
        edge_logits_f, root_logit_f, tuple_logits_f,
        leaf_clamp_jnp, edge_mask_jnp,
        left_child_jnp, right_child_jnp, left_edge_jnp, right_edge_jnp,
        edge_parent_jnp, edge_child_jnp, postorder_jnp, preorder_jnp)

    # Locate the edge whose CHILD is the held-out leaf in the padded tree.
    edge_child_np = np.asarray(padded_tree.edge_child)
    candidates = np.where(edge_child_np == holdout_padded_leaf)[0]
    if len(candidates) == 0:
        raise ValueError(
            f"holdout leaf node {holdout_padded_leaf} has no incoming edge "
            f"in padded tree")
    edge_to_holdout = int(candidates[0])

    # Slice to real columns and sum out the parent dimension to get
    # per-column P(holdout = PRESENT).
    p_present = np.asarray(
        pair_marg[edge_to_holdout, :L_real, :, PRESENT].sum(axis=-1))
    return p_present


# ---------------------------------------------------------------------------
# Batched val ELBO: vmap across same-bucket families. Same JIT cache key as
# fit_family_estep, plus an extra batch-size key for the vmapped runner.
# ---------------------------------------------------------------------------

_VMAP_VAL_CACHE = {}


def _make_vmapped_val_runner(L, n_dom, n_frag,
                               num_internal, num_nodes, num_edges, root,
                               n_iter, lr):
    """Build a JIT-compiled vmapped run_adam-then-elbo for fixed bucket shape.

    CRITICAL: cache the returned function in _VMAP_VAL_CACHE. Without this
    cache, every val pass redefines `vmapped_run_then_elbo` (a fresh
    @jax.jit-decorated closure), and JAX caches by function identity so it
    re-traces from scratch — paying the full compile cost (tens of minutes
    on big buckets) every val.
    """
    full_key = (L, n_dom, n_frag, num_internal, num_nodes, num_edges, root,
                int(n_iter), float(lr))
    if full_key in _VMAP_VAL_CACHE:
        return _VMAP_VAL_CACHE[full_key]
    if full_key not in _JIT_CACHE:
        _JIT_CACHE[full_key] = _make_neg_elbo_jit(
            L, n_dom, n_frag,
            num_internal, num_nodes, num_edges, root,
            int(n_iter), float(lr))
    run_adam, compute_neg_elbo, _ = _JIT_CACHE[full_key]

    @jax.jit
    def vmapped_run_then_elbo(edge_logits, root_logit, tuple_logits,
                                leaf_clamp, edge_lengths, L_sub,
                                classdist, params_flat,
                                edge_mask, col_mask, root_edge_idx_arr,
                                left_child, right_child, left_edge, right_edge,
                                edge_parent, edge_child, postorder, preorder):
        def per_family(el, rl, tl, lc, edl, ls, em, cm, rei,
                        lcc, rcc, lee, ree, epp, ecc, po, pr):
            final_p = run_adam(el, rl, tl, lc, edl, ls,
                                classdist, params_flat,
                                em, cm, rei, lcc, rcc, lee, ree,
                                epp, ecc, po, pr)
            ne = compute_neg_elbo(final_p[0], final_p[1], final_p[2],
                                    lc, edl, ls, classdist, params_flat,
                                    em, cm, rei, lcc, rcc, lee, ree,
                                    epp, ecc, po, pr)
            return ne
        return jax.vmap(per_family, in_axes=(0,) * 17)(
            edge_logits, root_logit, tuple_logits,
            leaf_clamp, edge_lengths, L_sub,
            edge_mask, col_mask, root_edge_idx_arr,
            left_child, right_child, left_edge, right_edge,
            edge_parent, edge_child, postorder, preorder)

    _VMAP_VAL_CACHE[full_key] = vmapped_run_then_elbo
    return vmapped_run_then_elbo


def _prepare_padded_family_for_val(bt, lp, sub_LL, params,
                                     target_leaves, target_L, seed):
    """Pad one family's inputs to bucket shape for batched val.

    Returns dict of arrays plus '_jit_key' for bucket grouping.
    """
    n_dom = params['dom_ins'].shape[0]
    n_frag = params['frag_weights'].shape[1]
    T = n_dom * n_frag
    L_real = int(lp.shape[1])
    n_leaves_real = int(bt.num_leaves)
    n_edges_real = int(bt.num_edges)

    edge_lengths_real = np.maximum(np.asarray(bt.edge_length), 1e-3)
    bt = bt._replace(edge_length=edge_lengths_real)

    padded_tree, edge_mask, ghost_leaf_indices, leaf_index_map, _ = \
        _pad_tree_with_ghosts(bt, target_leaves)
    leaf_present_pad, sub_LL_pad, col_mask = _pad_columns(
        lp, sub_LL, target_L)

    le, re = edge_lookup(padded_tree)

    pad_n_internal = int(padded_tree.num_internal)
    leaf_clamp_full = np.zeros((target_leaves, target_L, 3), dtype=np.float64)
    orig_clamp = np.asarray(leaf_clamp_to_beta(leaf_present_pad))
    for i in range(n_leaves_real):
        leaf_local = int(leaf_index_map[i]) - pad_n_internal
        leaf_clamp_full[leaf_local] = orig_clamp[i]
    for gi in ghost_leaf_indices:
        leaf_local = int(gi) - pad_n_internal
        leaf_clamp_full[leaf_local] = 1.0

    rng = np.random.default_rng(seed)
    edge_logits_real = rng.standard_normal((n_edges_real, 2)) * 0.1
    edge_logits_full = np.zeros((int(padded_tree.num_edges), 2),
                                  dtype=np.float64)
    edge_logits_full[:n_edges_real] = edge_logits_real

    root_logit = float(rng.standard_normal(()) * 0.1)

    L_sub_jnp = jnp.asarray(sub_LL_pad)
    log_L_per_tuple_init = class_marginalised_sub_LL_per_tuple(
        L_sub_jnp, params['classdist'])
    tuple_logits_init = (np.asarray(log_L_per_tuple_init) / 2.0).reshape(
        target_L, T)
    tuple_logits = tuple_logits_init + (
        rng.standard_normal((target_L, T)) * 0.05)

    edge_parent_arr = np.asarray(padded_tree.edge_parent)
    root_edge_idx_int = int(
        np.where(edge_parent_arr == padded_tree.root)[0][0])

    return {
        'edge_logits': edge_logits_full.astype(np.float64),
        'root_logit': np.asarray(root_logit, dtype=np.float64),
        'tuple_logits': tuple_logits.astype(np.float64),
        'leaf_clamp': leaf_clamp_full,
        'edge_lengths': np.asarray(padded_tree.edge_length, dtype=np.float64),
        'L_sub': sub_LL_pad.astype(np.float64),
        'edge_mask': edge_mask.astype(np.float64),
        'col_mask': col_mask.astype(np.float64),
        'root_edge_idx_arr': np.asarray(root_edge_idx_int, dtype=np.int32),
        'left_child': np.asarray(padded_tree.left_child, dtype=np.int32),
        'right_child': np.asarray(padded_tree.right_child, dtype=np.int32),
        'left_edge': np.asarray(le, dtype=np.int32),
        'right_edge': np.asarray(re, dtype=np.int32),
        'edge_parent': np.asarray(padded_tree.edge_parent, dtype=np.int32),
        'edge_child': np.asarray(padded_tree.edge_child, dtype=np.int32),
        'postorder': np.asarray(padded_tree.postorder_internal, dtype=np.int32),
        'preorder': np.asarray(padded_tree.preorder_internal, dtype=np.int32),
        '_jit_key': (target_L, n_dom, n_frag,
                      int(padded_tree.num_internal),
                      int(padded_tree.num_nodes),
                      int(padded_tree.num_edges),
                      int(padded_tree.root)),
    }


def eval_val_elbo_batched(params, val_provider, val_n, val_n_inner, lr,
                            seed_offset=10**6, verbose=False):
    """Batched val ELBO across same-shape buckets. Same semantics as the
    sequential per-family loop in svi_vbem_train, but vmapped per bucket.

    Returns (mean_elbo_per_family, list_of_elbos) or (None, None).
    """
    if val_provider is None or val_n <= 0:
        return None, None

    n_dom = params['dom_ins'].shape[0]
    n_frag = params['frag_weights'].shape[1]
    classdist_jnp = jnp.asarray(params['classdist'])
    params_flat = _params_to_flat(params, n_dom, n_frag)

    buckets = {}
    for vi in range(val_n):
        try:
            bt, lp, sub_LL = val_provider(int(vi))
        except Exception as exc:
            if verbose:
                print(f"  [val] family {vi} load failed: {exc}", flush=True)
            continue
        L_real = int(lp.shape[1])
        n_leaves_real = int(bt.num_leaves)
        target_leaves = _pad_to_bin(n_leaves_real, _LEAF_BINS)
        target_L = _pad_to_bin(L_real, _COL_BINS)
        try:
            inputs = _prepare_padded_family_for_val(
                bt, lp, sub_LL, params, target_leaves, target_L,
                seed_offset + vi)
        except Exception as exc:
            if verbose:
                print(f"  [val] family {vi} pad failed: {exc}", flush=True)
            continue
        key = inputs['_jit_key']
        buckets.setdefault(key, []).append(inputs)

    elbos = []
    keys_to_stack = ['edge_logits', 'root_logit', 'tuple_logits',
                      'leaf_clamp', 'edge_lengths', 'L_sub',
                      'edge_mask', 'col_mask', 'root_edge_idx_arr',
                      'left_child', 'right_child', 'left_edge', 'right_edge',
                      'edge_parent', 'edge_child', 'postorder', 'preorder']
    for key, inputs_list in buckets.items():
        L, _, _, num_internal, num_nodes, num_edges, root = key
        runner = _make_vmapped_val_runner(
            L, n_dom, n_frag,
            num_internal, num_nodes, num_edges, root,
            int(val_n_inner), float(lr))
        stacked = tuple(
            jnp.stack([jnp.asarray(it[k]) for it in inputs_list])
            for k in keys_to_stack)
        neg_elbos = runner(
            stacked[0], stacked[1], stacked[2],
            stacked[3], stacked[4], stacked[5],
            classdist_jnp, params_flat,
            stacked[6], stacked[7], stacked[8],
            stacked[9], stacked[10], stacked[11], stacked[12],
            stacked[13], stacked[14], stacked[15], stacked[16])
        elbos.extend([float(-e) for e in np.asarray(neg_elbos)])

    if not elbos:
        return None, None
    return sum(elbos) / len(elbos), elbos


def _compute_W_tensor(pair_marg_branch, q_tau, T):
    """Compute the (5, T, 5, T) W tensor for one branch (cumulant trick)."""
    P_M = pair_marg_branch[:, PRESENT, PRESENT]
    P_I = pair_marg_branch[:, NYI, PRESENT]
    P_D = pair_marg_branch[:, PRESENT, DELETED]
    P_Ig = pair_marg_branch[:, NYI, NYI] + pair_marg_branch[:, DELETED, DELETED]
    L = pair_marg_branch.shape[0]

    P_state_inner = jnp.zeros((L + 2, 5))
    P_state_inner = P_state_inner.at[0, WFST_S].set(1.0)
    P_state_inner = P_state_inner.at[L + 1, WFST_E].set(1.0)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_M].set(P_M)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_I].set(P_I)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_D].set(P_D)

    q_tau_full = jnp.zeros((L + 2, T))
    q_tau_full = q_tau_full.at[0, 0].set(1.0)
    q_tau_full = q_tau_full.at[L + 1, 0].set(1.0)
    q_tau_full = q_tau_full.at[1:L + 1, :].set(q_tau)
    P_state_tau = P_state_inner[:, :, None] * q_tau_full[:, None, :]

    eps = 1e-30
    P_Ig_full = jnp.ones(L + 2)
    P_Ig_full = P_Ig_full.at[1:L + 1].set(jnp.maximum(P_Ig, eps))
    log_P_Ig = jnp.log(P_Ig_full)
    C = jnp.cumsum(log_P_Ig)
    C_prev = jnp.concatenate([jnp.zeros(1), C[:-1]])

    log_P_tau = jnp.log(jnp.maximum(P_state_tau, eps))
    log_v = log_P_tau - C[:, None, None]
    log_v = jnp.where(P_state_tau > 0, log_v, -jnp.inf)

    def lae_step(carry, x):
        new = jnp.logaddexp(carry, x)
        return new, new

    init = jnp.full((5, T), -jnp.inf)
    _, cum_log_v = jax.lax.scan(lae_step, init, log_v)
    cum_log_v_prev = jnp.concatenate(
        [jnp.full((1, 5, T), -jnp.inf), cum_log_v[:-1]], axis=0)
    inner = jnp.exp(cum_log_v_prev + C_prev[:, None, None])
    return jnp.einsum('nsa,ntb->satb', inner[1:], P_state_tau[1:])


# ---------------------------------------------------------------------------
# Per-route soft-count attribution (M-step preparation).
# ---------------------------------------------------------------------------


def _per_route_soft_counts(W, params, t, n_dom, n_frag, eps=1e-30):
    """Decompose W tensor into per-route soft counts (EXACT decomposition).

    Computes the per-route Pair HMM contributions T^(r) = ω^(r) · W^(r)
    via mixdom_reduced_T_per_route, then route posteriors

      ρ^(r)_{ss', τ τ'} = T^(r)_{ss', τ τ'} / (T^(R1) + T^(R2) + T^(R3))_{ss', τ τ'}

    and the per-route soft counts

      tilde_W^(r)_{ss', τ τ'} = W_{ss', τ τ'} · ρ^(r)_{ss', τ τ'}.

    This replaces the earlier proportional-ω approximation that assumed
    W^(r) was constant across routes (which is wrong: W^(R1) = 1, W^(R2) =
    (1-β_d)α_d, W^(R3) = cross-domain entry weight, all distinct).

    Args:
        W: (5, T, 5, T) per-branch expected counts.
        params: MixDom params.
        t: branch length.
        n_dom, n_frag: tuple shape.

    Returns:
        dict with keys 'R1', 'R2', 'R3'; each is (5, T, 5, T).
    """
    T = n_dom * n_frag

    # Per-route Pair HMM contributions (each is the relevant block of chi
    # from build_nested_trans, reshaped to (5, T, 5, T)).
    T_R1, T_R2, T_R3 = mixdom_reduced_T_per_route(params, t)
    T_R1 = np.asarray(T_R1)
    T_R2 = np.asarray(T_R2)
    T_R3 = np.asarray(T_R3)

    # Route posterior at body entries: ρ^(r) = T^(r) / Σ_r T^(r).
    T_total = T_R1 + T_R2 + T_R3 + eps
    rho_R1 = T_R1 / T_total
    rho_R2 = T_R2 / T_total
    rho_R3 = T_R3 / T_total

    tilde_R1 = np.zeros_like(W)
    tilde_R2 = np.zeros_like(W)
    tilde_R3 = np.zeros_like(W)
    for s in [WFST_M, WFST_I, WFST_D]:
        for sp in [WFST_M, WFST_I, WFST_D]:
            tilde_R1[s, :, sp, :] = W[s, :, sp, :] * rho_R1[s, :, sp, :]
            tilde_R2[s, :, sp, :] = W[s, :, sp, :] * rho_R2[s, :, sp, :]
            tilde_R3[s, :, sp, :] = W[s, :, sp, :] * rho_R3[s, :, sp, :]
    # Boundary entries (S row, E column): attribute to R3 (entry/exit
    # are mediated by the cross-domain machinery). The S row at body
    # entries belongs to "first character emission" and the E column
    # to "last character departure"; both involve the top-level TKF91
    # transitions which are part of the cross-domain block in chi.
    tilde_R3[WFST_S, :, :, :] += W[WFST_S, :, :, :]
    tilde_R3[:, :, WFST_E, :] += W[:, :, WFST_E, :]
    # Avoid double-counting if (S, E) is already added.
    tilde_R3[WFST_S, :, WFST_E, :] -= W[WFST_S, :, WFST_E, :]
    return {'R1': tilde_R1, 'R2': tilde_R2, 'R3': tilde_R3}


# ---------------------------------------------------------------------------
# Sufficient-statistic extraction (separated from M-step for SVI-VBEM EMA).
# ---------------------------------------------------------------------------


def _empty_suff_stats(n_dom, n_frag):
    """Return a fresh suff-stats pytree with zeros."""
    return {
        'bdi_per_dom': [{'B': 0.0, 'D': 0.0, 'S': 0.0,
                         'L': 0.0, 'M': 0.0, 'T': 0.0} for _ in range(n_dom)],
        'bdi_top':     {'B': 0.0, 'D': 0.0, 'S': 0.0,
                        'L': 0.0, 'M': 0.0, 'T': 0.0},
        'ext_counts':   np.zeros((n_dom, n_frag, n_frag)),
        'notext_counts': np.zeros((n_dom, n_frag)),
        'dom_counts':    np.zeros(n_dom),
        'frag_counts':   np.zeros((n_dom, n_frag)),
    }


def extract_suff_stats(stats_list, params):
    """Aggregate per-family E-step stats into a sufficient-statistic pytree.

    The aggregate (over a (mini)batch of families) is a flat dict whose
    leaves are scalars or numpy arrays. All leaves are LINEAR in the
    per-family contributions, so the dict can be summed across batches
    or blended via EMA (\\cref{sec:vbem-svi} of varanc-vbem.tex).

    BDI stats (B, D, S, L, M, T) are accumulated PER BRANCH at the
    branch's own t (because bdi_stats_from_counts uses t-dependent
    derivatives) and only the resulting scalars are summed.

    Args:
        stats_list: list of FamilyEStepStats from E-step.
        params: current MixDom params (used for route attribution and BDI
            conversion at this iteration's θ — standard SVI setup).

    Returns:
        suff dict (see _empty_suff_stats for shape).
    """
    n_dom = params['dom_ins'].shape[0]
    n_frag = params['frag_weights'].shape[1]
    suff = _empty_suff_stats(n_dom, n_frag)

    bdi_per_dom = suff['bdi_per_dom']
    bdi_top = suff['bdi_top']
    ext_counts = suff['ext_counts']
    notext_counts = suff['notext_counts']
    dom_counts = suff['dom_counts']
    frag_counts = suff['frag_counts']

    for fs in stats_list:
        for e in range(fs.n_edges):
            t = float(fs.edge_lengths[e])
            W = fs.W_per_branch[e]  # (5, T, 5, T)
            tilde = _per_route_soft_counts(W, params, t, n_dom, n_frag)

            # Per-branch n_trans accumulation: each per-character W entry
            # contributes to multiple n_trans matrices depending on route.
            #
            # R1 (intra-fragment ext): NOT a TKF91 event for any level.
            #     Only contributes to fragment-extension counts (handled
            #     separately below in the ext_counts accumulation).
            #
            # R2 (intra-domain new-fragment): one within-domain TKF91
            #     transition (s -> s') for the source's domain.
            #     Per-domain n_trans_d[d][s, s'] += tilde_R2[s, τ_src, s', τ_dst].
            #
            # R3 (cross-domain or self-recurrence): represents three
            #     simultaneous micro-events:
            #       (a) source-domain exit: s -> E in the source domain's TKF91.
            #       (b) top-level transition: s -> s' in the top-level TKF91.
            #       (c) dest-domain entry: S -> s' in the dest domain's TKF91.
            #     Each R3 W entry contributes to all three n_trans matrices.
            #
            # Boundary entries (W[S, ...] and W[..., E, ...]) represent
            # the chain start/end:
            #     W[S, 0, s', τ_dst]: chain start. Contributes to top-level
            #         n_trans_t[S, s'] AND per-domain (d_dst) n_trans_d[d_dst][S, s'].
            #     W[s, τ_src, E, 0]: chain end. Contributes to top-level
            #         n_trans_t[s, E] AND per-domain (d_src) n_trans_d[d_src][s, E].

            n_trans_per_dom_branch = [np.zeros((5, 5)) for _ in range(n_dom)]
            n_trans_top_branch = np.zeros((5, 5))

            # R2 contributions to per-domain (within-domain new-fragment).
            tilde_R2 = tilde['R2']
            for d in range(n_dom):
                src_taus = [d * n_frag + f for f in range(n_frag)]
                dst_taus = [d * n_frag + fp for fp in range(n_frag)]
                for s in [WFST_M, WFST_I, WFST_D]:
                    for sp in [WFST_M, WFST_I, WFST_D]:
                        # Sum over τ_src in domain d, τ_dst in domain d.
                        n_trans_per_dom_branch[d][s, sp] += float(
                            tilde_R2[s, :, sp, :][src_taus, :][:, dst_taus].sum())

            # R3 contributions:
            tilde_R3 = tilde['R3']
            # Body entries: source domain d_src exit + dest domain d_dst entry
            #               + top-level cross-transition.
            for d_src in range(n_dom):
                src_taus = [d_src * n_frag + f for f in range(n_frag)]
                for d_dst in range(n_dom):
                    dst_taus = [d_dst * n_frag + fp for fp in range(n_frag)]
                    for s in [WFST_M, WFST_I, WFST_D]:
                        for sp in [WFST_M, WFST_I, WFST_D]:
                            mass = float(
                                tilde_R3[s, :, sp, :][src_taus, :][:, dst_taus].sum())
                            if mass > 0:
                                # Source domain: s -> E (fragment ends, domain exits)
                                n_trans_per_dom_branch[d_src][s, WFST_E] += mass
                                # Dest domain: S -> s' (entry into new domain)
                                n_trans_per_dom_branch[d_dst][WFST_S, sp] += mass
                                # Top level: s -> s' transition
                                n_trans_top_branch[s, sp] += mass

            # Boundary: W[S, *, s', τ_dst] (chain start). The boundary
            # τ_src is meaningless (no domain at the chain start), so sum
            # over all τ_src — _compute_W_tensor currently concentrates this
            # mass at τ_src=0 but we sum to be defensive.
            for d_dst in range(n_dom):
                dst_taus = [d_dst * n_frag + fp for fp in range(n_frag)]
                for sp in [WFST_M, WFST_I, WFST_D]:
                    mass = float(tilde_R3[WFST_S, :, sp, :][:, dst_taus].sum())
                    if mass > 0:
                        n_trans_per_dom_branch[d_dst][WFST_S, sp] += mass
                        n_trans_top_branch[WFST_S, sp] += mass

            # Boundary: W[s, τ_src, E, *] (chain end). Sum over all τ_dst
            # (boundary τ has no meaning) so τ_dst≠0 mass is not silently
            # dropped if the W boundary convention ever changes.
            for d_src in range(n_dom):
                src_taus = [d_src * n_frag + f for f in range(n_frag)]
                for s in [WFST_M, WFST_I, WFST_D]:
                    mass = float(tilde_R3[s, src_taus, WFST_E, :].sum())
                    if mass > 0:
                        n_trans_per_dom_branch[d_src][s, WFST_E] += mass
                        n_trans_top_branch[s, WFST_E] += mass

            # Convert per-branch n_trans -> BDI sufficient stats at THIS branch's t.
            for d in range(n_dom):
                n_trans_d = n_trans_per_dom_branch[d]
                if n_trans_d.sum() > 1e-9:
                    E_B, E_D, E_S = bdi_stats_from_counts(
                        n_trans_d, float(params['dom_ins'][d]),
                        float(params['dom_del'][d]), t,
                        T=t * n_trans_d.sum())
                    groups = transition_count_groups(n_trans_d)
                    bdi_per_dom[d]['B'] += float(E_B)
                    bdi_per_dom[d]['D'] += float(E_D)
                    bdi_per_dom[d]['S'] += float(E_S)
                    bdi_per_dom[d]['L'] += float(groups['log_kappa'])
                    bdi_per_dom[d]['M'] += float(groups['log_1mkappa'])
                    bdi_per_dom[d]['T'] += t * n_trans_d.sum()

            if n_trans_top_branch.sum() > 1e-9:
                E_B, E_D, E_S = bdi_stats_from_counts(
                    n_trans_top_branch, float(params['main_ins']),
                    float(params['main_del']), t,
                    T=t * n_trans_top_branch.sum())
                groups = transition_count_groups(n_trans_top_branch)
                bdi_top['B'] += float(E_B)
                bdi_top['D'] += float(E_D)
                bdi_top['S'] += float(E_S)
                bdi_top['L'] += float(groups['log_kappa'])
                bdi_top['M'] += float(groups['log_1mkappa'])
                bdi_top['T'] += t * n_trans_top_branch.sum()

            # Within-fragment extension counts (R1 only).
            for d in range(n_dom):
                for f in range(n_frag):
                    for fp in range(n_frag):
                        tau_src = d * n_frag + f
                        tau_dst = d * n_frag + fp
                        ext_counts[d, f, fp] += tilde['R1'][:, tau_src, :, tau_dst].sum()
                    # Notext = R2 + R3 contributions from this (d, f) source.
                    for sp in [WFST_M, WFST_I, WFST_D, WFST_E]:
                        notext_counts[d, f] += (
                            tilde['R2'][:, tau_src, sp, :].sum()
                            + tilde['R3'][:, tau_src, sp, :].sum())

        # Dom/frag tuple counts from q^(τ).
        q_tau_d = fs.q_tau.reshape(fs.n_cols, n_dom, n_frag)
        dom_counts += q_tau_d.sum(axis=(0, 2))
        frag_counts += q_tau_d.sum(axis=0)

    return suff


# ---------------------------------------------------------------------------
# Sufficient-statistic blending (EMA + scaling for SVI-VBEM).
# ---------------------------------------------------------------------------


def _scale_suff_stats(suff, scale):
    """Multiply every leaf of a suff-stats pytree by `scale`."""
    out = {
        'bdi_per_dom': [{k: scale * v for k, v in d.items()} for d in suff['bdi_per_dom']],
        'bdi_top':      {k: scale * v for k, v in suff['bdi_top'].items()},
        'ext_counts':   scale * np.asarray(suff['ext_counts']),
        'notext_counts': scale * np.asarray(suff['notext_counts']),
        'dom_counts':    scale * np.asarray(suff['dom_counts']),
        'frag_counts':   scale * np.asarray(suff['frag_counts']),
    }
    return out


def ema_blend_suff_stats(prev, batch, eta, batch_scale=1.0):
    """Apply (1-eta) prev + eta * batch_scale * batch elementwise.

    The pseudocount EMA of \\eqref{eq:svi-vbem-update} (varanc-vbem.tex
    sec:vbem-svi). For full-batch VBEM, set eta=1.0 and batch_scale=1.0
    to recover the plain M-step (prev contributes nothing). For SVI-VBEM,
    batch_scale should be N_total / |batch| (so the rescaled batch is an
    unbiased estimate of the full-corpus suff stats).
    """
    if prev is None:
        # No previous state — the EMA target IS the (rescaled) batch.
        return _scale_suff_stats(batch, batch_scale)
    out = {
        'bdi_per_dom': [
            {k: (1.0 - eta) * pv[k] + eta * batch_scale * bv[k]
             for k in pv}
            for pv, bv in zip(prev['bdi_per_dom'], batch['bdi_per_dom'])],
        'bdi_top': {
            k: (1.0 - eta) * prev['bdi_top'][k] + eta * batch_scale * batch['bdi_top'][k]
            for k in prev['bdi_top']},
        'ext_counts':    (1.0 - eta) * prev['ext_counts']    + eta * batch_scale * np.asarray(batch['ext_counts']),
        'notext_counts': (1.0 - eta) * prev['notext_counts'] + eta * batch_scale * np.asarray(batch['notext_counts']),
        'dom_counts':    (1.0 - eta) * prev['dom_counts']    + eta * batch_scale * np.asarray(batch['dom_counts']),
        'frag_counts':   (1.0 - eta) * prev['frag_counts']   + eta * batch_scale * np.asarray(batch['frag_counts']),
    }
    return out


# ---------------------------------------------------------------------------
# M-step from sufficient statistics.
# ---------------------------------------------------------------------------


def m_step_from_suff_stats(suff, params, prior=None, ext_dirichlet=1.5,
                            dom_dirichlet=1.5, frag_dirichlet=1.5):
    """Apply closed-form parameter updates given a suff-stats pytree.

    Substitution params (classdist, class_pis, class_S_exch) are HELD
    FIXED — class M-step requires Felsenstein expected substitution
    counts, deferred (TODO).
    """
    if prior is None:
        prior = {'alpha_lam': 2.0, 'alpha_mu': 2.0, 'beta': 10.0}

    n_dom = params['dom_ins'].shape[0]
    n_frag = params['frag_weights'].shape[1]

    bdi_per_dom = suff['bdi_per_dom']
    bdi_top = suff['bdi_top']
    ext_counts = np.asarray(suff['ext_counts'])
    notext_counts = np.asarray(suff['notext_counts'])
    dom_counts = np.asarray(suff['dom_counts'])
    frag_counts = np.asarray(suff['frag_counts'])

    new_params = dict(params)

    new_dom_ins = np.array(params['dom_ins'])
    new_dom_del = np.array(params['dom_del'])
    for d in range(n_dom):
        bdi = bdi_per_dom[d]
        if bdi['T'] < 1e-9 or bdi['S'] <= 0 or bdi['S'] + bdi['L'] <= 0:
            continue
        try:
            ins_new, del_new = m_step_indel_quadratic(
                bdi['B'], bdi['D'], bdi['S'],
                bdi['L'], bdi['M'], bdi['T'],
                prior_alpha_lam=prior['alpha_lam'],
                prior_alpha_mu=prior['alpha_mu'],
                prior_beta=prior['beta'])
            new_dom_ins[d] = ins_new
            new_dom_del[d] = del_new
        except (ValueError, FloatingPointError, ArithmeticError) as exc:
            warnings.warn(
                f"m_step_indel_quadratic failed for domain {d} "
                f"(B={bdi['B']:.4g}, D={bdi['D']:.4g}, S={bdi['S']:.4g}, "
                f"L={bdi['L']:.4g}, M={bdi['M']:.4g}, T={bdi['T']:.4g}): "
                f"{exc!r}. Keeping previous rates.",
                RuntimeWarning, stacklevel=2)
    new_params['dom_ins'] = jnp.asarray(new_dom_ins)
    new_params['dom_del'] = jnp.asarray(new_dom_del)

    if bdi_top['T'] > 1e-9 and bdi_top['S'] > 0:
        try:
            ins_new, del_new = m_step_indel_quadratic(
                bdi_top['B'], bdi_top['D'], bdi_top['S'],
                bdi_top['L'], bdi_top['M'], bdi_top['T'],
                prior_alpha_lam=prior['alpha_lam'],
                prior_alpha_mu=prior['alpha_mu'],
                prior_beta=prior['beta'])
            new_params['main_ins'] = float(ins_new)
            new_params['main_del'] = float(del_new)
        except (ValueError, FloatingPointError, ArithmeticError) as exc:
            warnings.warn(
                f"m_step_indel_quadratic failed for top-level "
                f"(B={bdi_top['B']:.4g}, D={bdi_top['D']:.4g}, "
                f"S={bdi_top['S']:.4g}, L={bdi_top['L']:.4g}, "
                f"M={bdi_top['M']:.4g}, T={bdi_top['T']:.4g}): "
                f"{exc!r}. Keeping previous rates.",
                RuntimeWarning, stacklevel=2)

    dom_count_aug = dom_counts + dom_dirichlet - 1.0
    dom_count_aug = np.maximum(dom_count_aug, 1e-6)
    new_params['dom_weights'] = jnp.asarray(dom_count_aug / dom_count_aug.sum())

    new_frag_w = np.zeros_like(np.asarray(params['frag_weights']))
    for d in range(n_dom):
        c = frag_counts[d] + frag_dirichlet - 1.0
        c = np.maximum(c, 1e-6)
        new_frag_w[d] = c / c.sum()
    new_params['frag_weights'] = jnp.asarray(new_frag_w)

    new_ext = np.zeros((n_dom, n_frag, n_frag))
    for d in range(n_dom):
        for f in range(n_frag):
            ext_row = ext_counts[d, f] + ext_dirichlet - 1.0
            notext_aug = notext_counts[d, f] + ext_dirichlet - 1.0
            ext_row = np.maximum(ext_row, 1e-6)
            notext_aug = max(notext_aug, 1e-6)
            total = ext_row.sum() + notext_aug
            new_ext[d, f] = ext_row / total
    new_params['ext_rates'] = jnp.asarray(new_ext)

    return new_params


# ---------------------------------------------------------------------------
# Convenience wrapper: full-batch M-step (extract + apply).
# ---------------------------------------------------------------------------


def m_step(stats_list, params, prior=None, ext_dirichlet=1.5,
           dom_dirichlet=1.5, frag_dirichlet=1.5):
    """Full-batch M-step. Equivalent to extract_suff_stats followed by
    m_step_from_suff_stats; preserves the existing API used by tests +
    vbem_train."""
    suff = extract_suff_stats(stats_list, params)
    return m_step_from_suff_stats(suff, params, prior=prior,
                                   ext_dirichlet=ext_dirichlet,
                                   dom_dirichlet=dom_dirichlet,
                                   frag_dirichlet=frag_dirichlet)


# ---------------------------------------------------------------------------
# Outer driver.
# ---------------------------------------------------------------------------


_MIXDOM_T_BUG_WARNED = False


def _warn_mixdom_T_convention_bug():
    """Print a prominent one-time warning about MixDom2 T conventions.

    Diagnosed 2026-05-09 in plain TKF92 tree-VBEM (tkf92_vbem.py) and
    confirmed to apply identically to MixDom2 (tree_vbem.py:1810,1817,
    1823,1830).
    """
    global _MIXDOM_T_BUG_WARNED
    if _MIXDOM_T_BUG_WARNED:
        return
    _MIXDOM_T_BUG_WARNED = True
    msg = (
        "\n" + "=" * 78 + "\n"
        "  WARNING: MixDom2 tree-VBEM T conventions disagree with body-mixdom.tex\n"
        "  sec:bw-mixdom (lines 502-545):\n"
        "    * Top-level (TKF91 inter-domain) should use T += t per branch.\n"
        "      Code uses T += t · n_trans_top_branch.sum().  WRONG.\n"
        "    * Per-domain (TKF92 intra-domain) should use T += t · n_hat_notkappa.\n"
        "      Code uses T += t · n_trans_d.sum().  WRONG (uses chain length\n"
        "      instead of fragment-termination count).\n"
        "  Combined effect: rate-recovery bias of ~+25% (smaller than the\n"
        "  +110% gap that paper-correct T would expose, because the inflated\n"
        "  T partially masks the column-factorised q's variational error).\n"
        "  Fix is gated on completing the TKF paper (per user policy).\n"
        + "=" * 78 + "\n")
    import warnings
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def vbem_train(families, init_params, n_outer=5, n_inner=100, lr=0.05,
               verbose=True, iter_callback=None):
    """Run VBEM training loop.

    Args:
        families: list of (binary_tree, leaf_present, sub_LL_per_class) tuples.
        init_params: starting MixDom params.
        n_outer: number of outer EM iterations.
        n_inner: Adam steps per family E-step.
        lr: Adam learning rate.
        iter_callback: optional fn(outer_idx, params_after_M, history) called
            after every outer iteration's M-step. Use for checkpointing.

    Returns:
        (final_params, history) where history is list of (iter, mean_elbo, params_snapshot).
    """
    _warn_mixdom_T_convention_bug()
    params = init_params
    history = []
    for outer in range(n_outer):
        t0 = time.time()
        stats_list = []
        for fi, (bt, lp, sub_LL) in enumerate(families):
            stats = fit_family_estep(bt, lp, params, sub_LL,
                                      n_iter=n_inner, lr=lr, seed=outer * 1000 + fi)
            stats_list.append(stats)
        mean_elbo = np.mean([s.elbo for s in stats_list])
        if verbose:
            print(f"[outer {outer}] mean ELBO/family = {mean_elbo:.2f}, "
                  f"E-step time = {time.time()-t0:.1f}s, n_families = {len(stats_list)}")
        history.append({'iter': outer, 'mean_elbo': mean_elbo,
                        'main_ins': float(params['main_ins']),
                        'main_del': float(params['main_del']),
                        'dom_ins': np.asarray(params['dom_ins']).tolist(),
                        'dom_del': np.asarray(params['dom_del']).tolist()})
        # M-step.
        params = m_step(stats_list, params)
        if verbose:
            print(f"  after M: main_ins={float(params['main_ins']):.5f}, "
                  f"main_del={float(params['main_del']):.5f}, "
                  f"dom_ins={[float(x) for x in params['dom_ins']]}")
        if iter_callback is not None:
            iter_callback(outer, params, history)

    return params, history


# ---------------------------------------------------------------------------
# SVI-VBEM: stochastic variational EM with EMA on suff stats and
# breadth-first minibatch sampling.
# ---------------------------------------------------------------------------


class BreadthFirstSampler:
    """Visit-count-prioritised minibatch sampler.

    Each call to `sample(batch_size)` returns the `batch_size` family
    indices with the smallest visit counts so far, breaking ties at
    random. Visit counts then increment for the chosen indices. This
    deterministic round-robin guarantees every family is visited at
    least once every ⌈N/B⌉ iterations (one "epoch") and that the
    per-family contribution arrives in Θ(K·B/N) of the first K
    iterations — no rare-family starvation.
    """

    def __init__(self, n_families, seed=0):
        self.n_families = int(n_families)
        self.visit_counts = np.zeros(self.n_families, dtype=np.int64)
        self.rng = np.random.default_rng(int(seed))

    def sample(self, batch_size):
        if batch_size > self.n_families:
            raise ValueError(
                f"batch_size {batch_size} > n_families {self.n_families}")
        # Random permutation breaks ties; argsort with stable sort then picks
        # the smallest-count entries (with random tie-breaking).
        perm = self.rng.permutation(self.n_families)
        order = perm[np.argsort(self.visit_counts[perm], kind='stable')]
        chosen = order[:batch_size]
        self.visit_counts[chosen] += 1
        return chosen.copy()

    def stats(self):
        c = self.visit_counts
        return {
            'min': int(c.min()), 'max': int(c.max()),
            'mean': float(c.mean()), 'unseen': int((c == 0).sum()),
        }

    def get_state(self):
        """Return a dict that fully captures the sampler's internal state."""
        return {
            'n_families': self.n_families,
            'visit_counts': self.visit_counts.copy(),
            'rng_state': self.rng.bit_generator.state,
        }

    def set_state(self, state):
        """Restore sampler from a get_state() snapshot."""
        if int(state['n_families']) != self.n_families:
            raise ValueError(
                f"sampler n_families mismatch: got {state['n_families']}, "
                f"sampler was initialised with {self.n_families}")
        self.visit_counts = np.asarray(state['visit_counts'], dtype=np.int64).copy()
        self.rng.bit_generator.state = state['rng_state']


# ---------------------------------------------------------------------------
# Resumption helpers: serialize/deserialize EMA suff-stats so a paused run
# can resume with all accumulator state intact (mirrors SVI-BW pairwise
# training's _save_checkpoint pattern in train_pfam.py — without this,
# warm-start from an iter checkpoint discards every batch contribution
# accumulated up to that iter).
# ---------------------------------------------------------------------------


def serialize_ema_stats(ema_stats, n_dom):
    """Flatten the suff-stats pytree into an npz-friendly dict.

    Returns a dict mapping str -> np.ndarray; reverses with
    deserialize_ema_stats(..., n_dom). Returns {} if ema_stats is None
    (i.e. no batches blended yet).
    """
    if ema_stats is None:
        return {}
    out = {}
    for d, b in enumerate(ema_stats['bdi_per_dom']):
        for k, v in b.items():
            out[f'ema_bdi_dom{d}_{k}'] = np.float64(v)
    for k, v in ema_stats['bdi_top'].items():
        out[f'ema_bdi_top_{k}'] = np.float64(v)
    out['ema_ext_counts']    = np.asarray(ema_stats['ext_counts'])
    out['ema_notext_counts'] = np.asarray(ema_stats['notext_counts'])
    out['ema_dom_counts']    = np.asarray(ema_stats['dom_counts'])
    out['ema_frag_counts']   = np.asarray(ema_stats['frag_counts'])
    return out


def deserialize_ema_stats(flat, n_dom):
    """Reconstruct the suff-stats pytree from a flat npz dict.

    Returns None if no ema_* keys are present (fresh start).
    """
    if not any(k.startswith('ema_') for k in flat):
        return None
    bdi_per_dom = []
    for d in range(int(n_dom)):
        bdi_per_dom.append({
            k: float(flat[f'ema_bdi_dom{d}_{k}'])
            for k in ('B', 'D', 'S', 'L', 'M', 'T')
        })
    bdi_top = {
        k: float(flat[f'ema_bdi_top_{k}'])
        for k in ('B', 'D', 'S', 'L', 'M', 'T')
    }
    return {
        'bdi_per_dom': bdi_per_dom,
        'bdi_top': bdi_top,
        'ext_counts':    np.asarray(flat['ema_ext_counts']),
        'notext_counts': np.asarray(flat['ema_notext_counts']),
        'dom_counts':    np.asarray(flat['ema_dom_counts']),
        'frag_counts':   np.asarray(flat['ema_frag_counts']),
    }


def serialize_sampler_state(sampler):
    """Pull the BreadthFirstSampler's state into npz-friendly arrays.

    PCG64's state['state']/state['inc'] are 128-bit Python ints that
    don't fit in any numpy integer dtype, so we store them as decimal
    strings (round-trip exact via int(...)).
    """
    s = sampler.get_state()
    rng = s['rng_state']
    return {
        'sampler_n_families': np.int64(s['n_families']),
        'sampler_visit_counts': s['visit_counts'],
        'sampler_rng_bit_generator': np.array(str(rng['bit_generator'])),
        'sampler_rng_state_state':
            np.array(str(int(rng['state']['state']))),
        'sampler_rng_state_inc':
            np.array(str(int(rng['state']['inc']))),
        'sampler_rng_has_uint32':
            np.int32(int(rng.get('has_uint32', 0))),
        'sampler_rng_uinteger':
            np.uint32(int(rng.get('uinteger', 0))),
    }


def deserialize_sampler_state(flat):
    """Inverse of serialize_sampler_state. Returns None if no sampler_* keys."""
    if 'sampler_n_families' not in flat:
        return None
    return {
        'n_families': int(flat['sampler_n_families']),
        'visit_counts': np.asarray(
            flat['sampler_visit_counts'], dtype=np.int64),
        'rng_state': {
            'bit_generator': str(flat['sampler_rng_bit_generator']),
            'state': {
                'state': int(str(flat['sampler_rng_state_state'])),
                'inc': int(str(flat['sampler_rng_state_inc'])),
            },
            'has_uint32': int(flat['sampler_rng_has_uint32']),
            'uinteger': int(flat['sampler_rng_uinteger']),
        },
    }


def svi_vbem_train(family_provider, init_params, n_total_families,
                   batch_size=10, n_iter=100, n_inner=100, lr=0.05,
                   tau=10.0, kappa=0.7,
                   prior=None, ext_dirichlet=1.5,
                   dom_dirichlet=1.5, frag_dirichlet=1.5,
                   sampler_seed=0, verbose=True, iter_callback=None,
                   val_provider=None, val_n=0, val_every_k=5,
                   val_n_inner=None, val_batched=True,
                   init_ema_stats=None, init_sampler_state=None,
                   start_iter=0, init_history=None):
    """Stochastic VBEM (\\cref{sec:vbem-svi} of varanc-vbem.tex).

    Each iteration:
      1. Breadth-first sample a minibatch B_k of `batch_size` families.
      2. Per-family Adam E-step → variational posteriors q_i.
      3. Extract aggregated suff stats from the batch.
      4. EMA-blend with previous stats:
           ω̄_k = (1 - η_k) ω̄_{k-1} + η_k · (N/|B|) · ω_batch
         with η_k = (k + tau)^(-kappa).
      5. Closed-form M-step from the EMA stats.

    The pseudocount-EMA convention matches SVI-BW
    (\\cref{sec:pseudocount-view} of svb-convergence.tex); the only
    difference is that the per-batch suff stats are family-aggregate
    Tree-VBEM stats rather than pair-aggregate FB stats.

    Args:
        family_provider: callable index -> (binary_tree, leaf_present,
            sub_LL_per_class) — fetches a family by integer index in
            [0, n_total_families). Caller controls caching strategy.
        init_params: starting MixDom params dict.
        n_total_families: corpus size N.
        batch_size: minibatch size |B|.
        n_iter: total outer iterations.
        n_inner: Adam steps per per-family E-step.
        lr: Adam learning rate.
        tau, kappa: step-size schedule η_k = (k+τ)^(-κ).
        prior: BDI pseudocount priors (default {alpha_lam=2, alpha_mu=2,
            beta=10}).
        ext_dirichlet, dom_dirichlet, frag_dirichlet: Dirichlet pseudocounts.
        sampler_seed: seed for the breadth-first sampler's tie-break RNG.
        verbose: print per-iteration progress.
        iter_callback: optional fn(iter, params_after_M, history) for
            checkpointing.

    Returns:
        (final_params, history).
    """
    if batch_size > n_total_families:
        raise ValueError(
            f"batch_size {batch_size} > n_total_families {n_total_families}")
    _warn_mixdom_T_convention_bug()
    sampler = BreadthFirstSampler(n_total_families, seed=sampler_seed)
    if init_sampler_state is not None:
        sampler.set_state(init_sampler_state)
    scale = float(n_total_families) / float(batch_size)
    params = init_params
    ema_stats = init_ema_stats
    history = list(init_history) if init_history else []
    start_iter = int(start_iter)

    if val_n_inner is None:
        val_n_inner = n_inner

    def _eval_val_elbo(p):
        """Sweep through val_n val families at fixed θ=p, return mean
        ELBO/fam and per-family list. Caller controls when this fires
        (typically every val_every_k iterations) — it's the proper
        convergence signal because each call sees the SAME val set.

        If val_batched=True (default), uses the bucketed-vmap path
        (eval_val_elbo_batched) which groups same-padded-shape families
        and runs the Adam q-fit + ELBO via a single vmapped JIT call per
        bucket. Empirically 3-5x faster on GPU than the per-family loop."""
        if val_provider is None or val_n <= 0:
            return None, None
        if val_batched:
            return eval_val_elbo_batched(
                p, val_provider, val_n, val_n_inner, lr, verbose=False)
        elbos = []
        for vi in range(val_n):
            try:
                bt, lp, sub_LL = val_provider(int(vi))
            except Exception:
                continue
            try:
                stats = fit_family_estep(bt, lp, p, sub_LL,
                                          n_iter=val_n_inner, lr=lr,
                                          seed=10**6 + vi)
            except Exception:
                continue
            elbos.append(float(stats.elbo))
        if not elbos:
            return None, None
        return sum(elbos) / len(elbos), elbos
    for k in range(start_iter, n_iter):
        eta_k = (k + tau) ** (-kappa)
        t0 = time.time()
        indices = sampler.sample(batch_size)
        stats_list = []
        used_indices = []
        for slot, idx in enumerate(indices):
            try:
                bt, lp, sub_LL = family_provider(int(idx))
            except Exception as exc:
                warnings.warn(
                    f"[iter {k}] skipping family idx {int(idx)}: {exc}",
                    RuntimeWarning, stacklevel=2)
                continue
            try:
                stats = fit_family_estep(bt, lp, params, sub_LL,
                                          n_iter=n_inner, lr=lr,
                                          seed=int(k) * 100000 + int(idx))
            except Exception as exc:
                warnings.warn(
                    f"[iter {k}] E-step failed for family idx {int(idx)}: {exc}",
                    RuntimeWarning, stacklevel=2)
                continue
            stats_list.append(stats)
            used_indices.append(int(idx))
        if not stats_list:
            warnings.warn(
                f"[iter {k}] no families succeeded — skipping M-step",
                RuntimeWarning, stacklevel=2)
            continue
        e_time = time.time() - t0

        # Per-batch ELBO (scaled by N/|effective_B| for cross-iter
        # comparison). When some families fail to load, the effective
        # batch is smaller — rescale so the EMA target remains an
        # unbiased estimate of the full-corpus stats.
        effective_batch = len(stats_list)
        effective_scale = float(n_total_families) / float(effective_batch)
        batch_elbo_sum = float(np.sum([s.elbo for s in stats_list]))
        scaled_elbo = effective_scale * batch_elbo_sum
        mean_elbo_per_family = batch_elbo_sum / float(effective_batch)

        suff_batch = extract_suff_stats(stats_list, params)
        ema_stats = ema_blend_suff_stats(ema_stats, suff_batch, eta_k,
                                          batch_scale=effective_scale)
        params = m_step_from_suff_stats(
            ema_stats, params, prior=prior,
            ext_dirichlet=ext_dirichlet,
            dom_dirichlet=dom_dirichlet,
            frag_dirichlet=frag_dirichlet)

        sstats = sampler.stats()
        if verbose:
            print(f"[iter {k}] eta={eta_k:.4f}, "
                  f"batch_size={len(stats_list)}, "
                  f"mean ELBO/fam={mean_elbo_per_family:.2f}, "
                  f"scaled ELBO={scaled_elbo:.2f}, "
                  f"E-step time={e_time:.1f}s, "
                  f"visits={sstats['min']}-{sstats['max']} "
                  f"(unseen={sstats['unseen']})")
            print(f"  after M: main_ins={float(params['main_ins']):.5f}, "
                  f"main_del={float(params['main_del']):.5f}, "
                  f"dom_ins={[float(x) for x in params['dom_ins']]}")
        # Optional val ELBO at fixed θ — fires every val_every_k iters
        # (and on the final iter so the run always reports a final val).
        val_mean = None
        if val_provider is not None and val_n > 0 and (
                ((k + 1) % val_every_k == 0) or (k + 1 == n_iter)):
            t_val = time.time()
            val_mean, _val_list = _eval_val_elbo(params)
            v_time = time.time() - t_val
            if verbose and val_mean is not None:
                print(f"  val ELBO/fam @ θ_after_M = {val_mean:.2f} "
                      f"(n={val_n}, time={v_time:.1f}s)")

        history.append({
            'iter': k,
            'eta': float(eta_k),
            'batch_indices': [int(i) for i in indices],
            'batch_size': int(len(stats_list)),
            'mean_elbo_per_family': mean_elbo_per_family,
            'scaled_elbo': scaled_elbo,
            'val_elbo_per_family': val_mean,
            'main_ins': float(params['main_ins']),
            'main_del': float(params['main_del']),
            'dom_ins': np.asarray(params['dom_ins']).tolist(),
            'dom_del': np.asarray(params['dom_del']).tolist(),
            'visits_min': sstats['min'],
            'visits_max': sstats['max'],
            'visits_unseen': sstats['unseen'],
        })
        if iter_callback is not None:
            try:
                iter_callback(k, params, history, ema_stats, sampler)
            except TypeError:
                # Backward-compat: callbacks with the legacy 3-arg signature.
                iter_callback(k, params, history)

    return params, history
