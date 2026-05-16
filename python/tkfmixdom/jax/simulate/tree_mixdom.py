"""Tree-based simulator for MixDom2 / TKF92 evolution.

For testing Tree-VBEM parameter recovery: simulate per-leaf sequences down a
phylogenetic tree using the labeled MixDom2 generative process (equivalently
the chi-matrix Pair HMM at each branch), then run Tree-VBEM on the result and
check whether it recovers the input rates.

V1 supports only TKF92 (no MixDom hierarchy) — main_ins, main_del, ext_rate,
single substitution Q. This is enough to diagnose whether Tree-VBEM has a
fundamental indel-rate calibration bug; per-domain / per-class structure can
be added once V1 passes.

The labeled approach assigns each residue a unique lineage id during
simulation; the MSA is reconstructed as columns indexed by lineage id, with
residues filled in for leaves that have that lineage and gaps elsewhere.
Column order is the order in which lineage ids appear during a left-to-right
preorder traversal of the simulation.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.params import S, M, I, D, E, tkf92_trans
from ..core.ctmc import transition_matrix


def _state_machinery():
    return np.array([S, M, I, D, E])


def _normalised_log_trans(ins_rate, del_rate, t, ext):
    """5x5 log-transition matrix for TKF92 with extension."""
    tau = np.array(tkf92_trans(ins_rate, del_rate, t, ext))
    return np.log(np.maximum(tau, 1e-300))


def _softmax_masked(log_probs):
    """Softmax over a row that may have -inf entries. Returns 0 for all-inf."""
    arr = np.asarray(log_probs, dtype=np.float64)
    m = arr.max()
    if not np.isfinite(m):
        return np.zeros_like(arr)
    e = np.exp(arr - m)
    s = e.sum()
    if s < 1e-300:
        return np.zeros_like(arr)
    return e / s


def evolve_branch_tkf92(rng, ancestor_residues, ancestor_lineage,
                          ins_rate, del_rate, t, ext, Q, pi,
                          next_lineage_id, max_steps_per_anc=10):
    """Evolve a sequence down one branch under TKF92, tracking lineage IDs.

    Args:
        rng: JAX PRNG key for this branch.
        ancestor_residues: (Lx,) np.int32 array of ancestor amino acid codes.
        ancestor_lineage:  (Lx,) np.int32 array — lineage id per ancestor pos.
        ins_rate, del_rate, t, ext, Q, pi: TKF92 model.
        next_lineage_id: integer counter — first available lineage id for
            new (inserted) residues. The function returns the updated counter.
        max_steps_per_anc: safety bound (steps per ancestor position).

    Returns:
        descendant_residues: (Ly,) np.int32 — descendant amino acid codes.
        descendant_lineage:  (Ly,) np.int32 — lineage id per descendant pos.
            For Match: inherited from ancestor. For Insert: a new id.
        descendant_after:    (Ly,) np.int32 — for each descendant residue, the
            ancestor lineage id that immediately precedes it in the chain (i.e.
            the lineage id this residue comes "after" in the parent's order).
            -1 for residues inserted at the very start (before any ancestor).
            For Match residues, this equals their own lineage id minus the
            "after" relation: convention here is that descendant_after[k] is
            the ANCESTOR position they Match (so insertions ordered between
            position p-1 and p have descendant_after = ancestor_lineage[p-1]).
        next_lineage_id: integer — first unused lineage id after this branch.
    """
    sub_matrix = np.asarray(transition_matrix(Q, t))
    log_trans = _normalised_log_trans(ins_rate, del_rate, t, ext)
    state_types = _state_machinery()
    Lx = len(ancestor_residues)
    A = len(pi)
    rng_int = int(jr.fold_in(rng, 0)[0]) % (2**31)
    np_rng = np.random.RandomState(rng_int)

    descendant_residues = []
    descendant_lineage = []
    descendant_after = []

    current_state = S
    anc_pos = 0
    last_consumed_lineage = -1  # ancestor lineage id of the most recently
                                  # M-or-D-consumed parent residue
    safety_cap = (Lx + 1) * max_steps_per_anc + 100

    for _step in range(safety_cap):
        # Build per-state log-probs masked by which transitions are legal.
        log_probs = log_trans[current_state].copy()
        for k, st in enumerate(state_types):
            if st in (M, D) and anc_pos >= Lx:
                log_probs[k] = -np.inf
            elif st == E and anc_pos < Lx:
                log_probs[k] = -np.inf
            elif st == S:
                log_probs[k] = -np.inf  # can't return to start
        probs = _softmax_masked(log_probs)
        if probs.sum() == 0:
            # Degenerate — terminate.
            break
        next_state = np_rng.choice(len(state_types), p=probs)
        st = state_types[next_state]
        if st == E:
            break
        elif st == M:
            x = int(ancestor_residues[anc_pos])
            y = int(np_rng.choice(A, p=sub_matrix[x]))
            descendant_residues.append(y)
            descendant_lineage.append(int(ancestor_lineage[anc_pos]))
            descendant_after.append(int(ancestor_lineage[anc_pos]))
            last_consumed_lineage = int(ancestor_lineage[anc_pos])
            anc_pos += 1
        elif st == I:
            y = int(np_rng.choice(A, p=pi))
            descendant_residues.append(y)
            descendant_lineage.append(next_lineage_id)
            descendant_after.append(last_consumed_lineage)
            next_lineage_id += 1
        elif st == D:
            last_consumed_lineage = int(ancestor_lineage[anc_pos])
            anc_pos += 1
        current_state = next_state

    return (np.asarray(descendant_residues, dtype=np.int32),
            np.asarray(descendant_lineage, dtype=np.int32),
            np.asarray(descendant_after, dtype=np.int32),
            next_lineage_id)


def simulate_tkf92_tree(rng, tree_root, ins_rate, del_rate, ext, Q, pi,
                          root_length_mean=100, fixed_root_residues=None):
    """Simulate per-leaf sequences down a tree under TKF92.

    Args:
        rng: JAX PRNG key.
        tree_root: TreeNode root from parse_newick (must have .children,
            .branch_length, .name, .leaves()).
        ins_rate, del_rate, ext, Q, pi: TKF92 model parameters.
        root_length_mean: Poisson mean for the root chain length (ignored
            if fixed_root_residues is provided).
        fixed_root_residues: optional (L,) np.int32 — if given, use as the
            root chain instead of sampling from pi.

    Returns:
        leaf_seqs: dict[leaf_name] -> np.int32 array (un-aligned residue
            sequence in chain order).
        leaf_lineages: dict[leaf_name] -> np.int32 array (lineage id per
            residue).
        msa: dict[leaf_name] -> np.int32 array (length n_cols; -1 for gap).
        n_cols: total alignment width.
        true_branch_alignments: list of (parent_name_or_id, child_name_or_id,
            t, np.int32 alignment_path) — useful for diagnostics. Each
            alignment_path is a flat list of ('M'/'I'/'D' codes) emitted
            during that branch.
    """
    rng_int = int(jr.fold_in(rng, 0)[0]) % (2**31)
    np_rng = np.random.RandomState(rng_int)
    A = len(pi)

    # Sample root chain.
    if fixed_root_residues is None:
        L0 = max(1, np_rng.poisson(root_length_mean))
        root_residues = np_rng.choice(A, size=L0, p=np.asarray(pi))
    else:
        root_residues = np.asarray(fixed_root_residues, dtype=np.int32)
        L0 = len(root_residues)
    root_lineage = np.arange(L0, dtype=np.int32)
    next_lineage = L0

    # Walk the tree preorder; identify each non-root node and its parent.
    # Each node is identified by id(node). Use Python id() since TreeNode
    # objects don't carry stable IDs.
    node_chain = {id(tree_root): (root_residues, root_lineage)}
    branches_in_order = []

    def walk(node):
        for c in node.children:
            branches_in_order.append((node, c, float(c.branch_length or 0.0)))
            walk(c)
    walk(tree_root)

    # Use a JAX subkey per branch for determinism.
    keys = jr.split(rng, len(branches_in_order) + 1)
    true_branch_alignments = []

    for bi, (parent, child, t) in enumerate(branches_in_order):
        p_res, p_lin = node_chain[id(parent)]
        c_res, c_lin, c_after, next_lineage = evolve_branch_tkf92(
            keys[bi], p_res, p_lin, ins_rate, del_rate, t, ext, Q, pi,
            next_lineage)
        node_chain[id(child)] = (c_res, c_lin)
        # Track per-branch event tally for diagnostics.
        n_match = int(np.sum(np.isin(c_lin, p_lin)))
        n_insert = int(len(c_lin) - n_match)
        n_delete = int(len(p_lin) - n_match)
        true_branch_alignments.append({
            'parent_name': parent.name or f'<int_{id(parent) & 0xffff}>',
            'child_name': child.name or f'<int_{id(child) & 0xffff}>',
            't': t,
            'n_match': n_match,
            'n_insert': n_insert,
            'n_delete': n_delete,
            'parent_len': len(p_res),
            'child_len': len(c_res),
            # Full per-branch trace, sufficient to reconstruct the
            # M/I/D transition trajectory (Phase 1 oracle suff-stat tests).
            'parent_lineage': np.asarray(p_lin, dtype=np.int64).copy(),
            'child_lineage': np.asarray(c_lin, dtype=np.int64).copy(),
            'child_after': np.asarray(c_after, dtype=np.int64).copy(),
        })

    # ---- Build the MSA ----
    # Collect all lineage IDs that appear at any leaf.
    leaf_seqs = {}
    leaf_lineages = {}
    leaves = []
    def collect_leaves(node):
        if not node.children:
            leaves.append(node)
        else:
            for c in node.children:
                collect_leaves(c)
    collect_leaves(tree_root)

    for leaf in leaves:
        res, lin = node_chain[id(leaf)]
        leaf_seqs[leaf.name] = res
        leaf_lineages[leaf.name] = lin

    # Order columns by lineage id (creation order: root first, then
    # branches in tree-traversal order). This is monotonically increasing
    # by construction since we incremented next_lineage per insertion in
    # tree-preorder. Gaps are filled with -1 for leaves missing that
    # lineage.
    # WARNING: this ordering does NOT preserve "natural" left-to-right
    # chain order across the tree. The MSA columns will be in CREATION
    # order, not in chain order. For Tree-VBEM that's fine because TKF
    # treats columns as iid; the test only needs a valid MSA, not a
    # biologically nice one.
    all_lineages = sorted({int(x) for leaf in leaves
                            for x in leaf_lineages[leaf.name]})
    col_idx = {lin: i for i, lin in enumerate(all_lineages)}
    n_cols = len(all_lineages)

    msa = {}
    for leaf in leaves:
        row = np.full(n_cols, -1, dtype=np.int32)
        res = leaf_seqs[leaf.name]
        lin = leaf_lineages[leaf.name]
        for r, l in zip(res, lin):
            row[col_idx[int(l)]] = int(r)
        msa[leaf.name] = row

    return leaf_seqs, leaf_lineages, msa, n_cols, true_branch_alignments
