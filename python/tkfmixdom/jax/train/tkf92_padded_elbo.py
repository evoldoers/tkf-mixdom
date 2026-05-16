"""Padded ELBO for plain TKF92 svi-VarAnc (GPU-friendly, JIT cache reuse).

Mirrors the MixDom2 padding infrastructure in tree_vbem.py
(_bp_pair_marginals_jax, _expected_branch_LL_mixdom_masked,
_entropy_per_column_masked_argonly, _singlet_root_log_prior_mixdom_masked)
but for the plain TKF92 ELBO (no domain/fragment/class hierarchy).

Cache-key strategy: tree topology arrays are passed as JIT arguments,
not closure-captured, so the cache reuses across all families with the
same (n_leaves_bin, n_cols_bin) shape.

Mask discipline:
  * edge_mask gates ghost edges (real -> learned q_cond, ghost -> identity).
  * col_mask gates padded columns inside the masked branch-LL / entropy /
    root-prior / log_Z terms.
  * Ghost-edge logits get zero gradient via edge_mask in the Adam step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..tree.varanc_presence import (
    NYI, PRESENT, DELETED,
    edge_lookup,
    leaf_clamp_to_beta,
    make_q_conditionals,
    make_root_dist,
    tkf92_wfst_log_T,
)
from .tree_vbem import (
    _pad_to_bin, _pad_tree_with_ghosts,
    _bp_pair_marginals_jax_log as _bp_pair_marginals_jax,
    _entropy_per_column_masked_argonly,
    _LEAF_BINS, _COL_BINS,
)


N_Z = 3  # NYI, PRESENT, DELETED
# 6-state TKF92 WFST with Ins0/Ins1 split; see tkf/tkf92-wfst-derivation.tex
# eq:tkf92-wfst. State indices match varanc_presence.py.
N_WFST = 6  # S, M, I=I1, D, E, I0
WFST_S, WFST_M, WFST_I, WFST_D, WFST_E, WFST_I0 = 0, 1, 2, 3, 4, 5
WFST_I1 = WFST_I


def _pad_leaf_present(leaf_present, target_L):
    """Pad column axis of a (n_leaves, L) presence array to target_L.

    Padded columns set to 0 (absent everywhere). Returns
    (padded, col_mask) with col_mask 1 on real, 0 on padded columns.
    """
    n_leaves, L = leaf_present.shape
    if L > target_L:
        raise ValueError(f"L={L} > target_L={target_L}")
    if L == target_L:
        return np.asarray(leaf_present), np.ones(L, dtype=np.float64)
    pad_L = target_L - L
    out = np.concatenate([
        np.asarray(leaf_present),
        np.zeros((n_leaves, pad_L), dtype=leaf_present.dtype),
    ], axis=1)
    col_mask = np.zeros(target_L, dtype=np.float64)
    col_mask[:L] = 1.0
    return out, col_mask


def _expected_branch_LL_masked(pair_marg_branch, log_T_branch, col_mask,
                                eps=1e-30):
    """Plain-TKF92 expected_branch_LL with column masking.

    On padded columns (col_mask[n] = 0), force P_M = P_I = P_D = 0 and
    P_Ig = 1 so the cumulant integral skips those columns entirely.
    """
    P_M = pair_marg_branch[:, PRESENT, PRESENT] * col_mask
    P_I = pair_marg_branch[:, NYI, PRESENT] * col_mask
    P_D = pair_marg_branch[:, PRESENT, DELETED] * col_mask
    P_Ig_real = (pair_marg_branch[:, NYI, NYI]
                  + pair_marg_branch[:, DELETED, DELETED])
    P_Ig = jnp.where(col_mask > 0, jnp.maximum(P_Ig_real, eps), 1.0)
    # Leading-NYI prefix product on real columns only: padded columns
    # contribute factor 1, since col_mask*log(P_NYI) is masked to 0.
    P_NYI_real = (pair_marg_branch[:, NYI, NYI]
                   + pair_marg_branch[:, NYI, PRESENT])
    log_P_NYI = jnp.where(col_mask > 0, jnp.log(jnp.maximum(P_NYI_real, eps)), 0.0)
    cum_log_NYI = jnp.cumsum(log_P_NYI)
    log_pi = jnp.concatenate([jnp.zeros(1), cum_log_NYI[:-1]])
    pi = jnp.exp(log_pi)
    P_I0 = P_I * pi
    P_I1 = P_I * (1.0 - pi)
    L = pair_marg_branch.shape[0]

    P_state = jnp.zeros((L + 2, N_WFST))
    P_state = P_state.at[0, WFST_S].set(1.0)
    P_state = P_state.at[L + 1, WFST_E].set(1.0)
    P_state = P_state.at[1:L + 1, WFST_M].set(P_M)
    P_state = P_state.at[1:L + 1, WFST_I1].set(P_I1)
    P_state = P_state.at[1:L + 1, WFST_I0].set(P_I0)
    P_state = P_state.at[1:L + 1, WFST_D].set(P_D)

    P_Ig_full = jnp.ones(L + 2)
    P_Ig_full = P_Ig_full.at[1:L + 1].set(P_Ig)
    log_P_Ig = jnp.log(P_Ig_full)
    C = jnp.cumsum(log_P_Ig)

    # Safe-gradient pattern: avoid log(max(x, eps)) which has gradient
    # 1/eps at x=0, producing NaN when multiplied by downstream zeros
    # in the backward pass on large trees.  Use double-where: replace
    # P_state=0 with 1.0 inside log (gradient = 0), then mask back.
    safe_P_state = jnp.where(P_state > 0, P_state, 1.0)
    # Use a large negative finite value (not -inf) for the False branch
    # so that the cumulant scan's backward pass via logaddexp stays
    # finite. -1e10 is large enough that exp(-1e10) underflows to 0
    # for all practical purposes, but its gradient is well-defined.
    LARGE_NEG = -1e10
    log_P_state = jnp.where(P_state > 0, jnp.log(safe_P_state), LARGE_NEG)
    log_v = jnp.where(P_state > 0, log_P_state - C[:, None], LARGE_NEG)

    def lae_step(carry, x):
        new = jnp.logaddexp(carry, x)
        return new, new

    init = jnp.full((N_WFST,), LARGE_NEG)
    _, cum_log_v = jax.lax.scan(lae_step, init, log_v)
    cum_log_v_prev = jnp.concatenate(
        [jnp.full((1, N_WFST), LARGE_NEG), cum_log_v[:-1]], axis=0)
    C_prev = jnp.concatenate([jnp.zeros(1), C[:-1]])
    inner = jnp.exp(cum_log_v_prev + C_prev[:, None])
    W = jnp.einsum('ns,nt->st', inner[1:], P_state[1:])
    return jnp.sum(log_T_branch * W)


def _singlet_root_log_prior_tkf92_masked(root_dist, ins_rate, del_rate, ext,
                                            col_mask, eps=1e-30):
    """Plain-TKF92 singlet_root_log_prior with column masking.

    Mirrors varanc_presence.singlet_root_log_prior but multiplies the
    presence sum (sum_n q(root_n=P)) by col_mask before summing, and
    computes P_q(L=0) over real cols only (padded cols contribute 0
    log-prob => factor 1 multiplicatively, equivalent to ignoring them).
    """
    kappa = ins_rate / del_rate
    p = ext + (1.0 - ext) * kappa
    log_p = jnp.log(p)
    log_1mk = jnp.log(1.0 - kappa)
    c1 = jnp.log(kappa * (1.0 - ext) * (1.0 - kappa) / p)

    sum_root_P = jnp.sum(root_dist[:, PRESENT] * col_mask)

    # P_q(L=0) = prod_{real n} q(root_n = NYI). Padded cols contribute
    # factor 1 (equivalent log = 0), achieved by multiplying log term by
    # col_mask before summing.
    safe_NYI = jnp.maximum(root_dist[:, NYI], eps)
    log_P_L_zero = jnp.sum(jnp.log(safe_NYI) * col_mask)
    P_L_zero = jnp.exp(log_P_L_zero)

    return (log_p * sum_root_P
            + (1.0 - P_L_zero) * c1
            + P_L_zero * log_1mk)


def pad_tkf92_inputs(binary_tree, leaf_present,
                       target_leaves=None, target_L=None):
    """Bundle: pad tree + columns; build all the masks and arrays needed.

    Returns dict keys:
      'padded_tree', 'edge_mask', 'col_mask', 'leaf_clamp_full',
      'edge_logits_init', 'le', 're', 'n_real_edges', 'n_real_leaves',
      'L_real', 'leaf_index_map', 'ghost_leaf_indices', 'target_leaves',
      'target_L', 'edge_length', 'root_edge_idx'.
    """
    n_leaves_real = int(binary_tree.num_leaves)
    L_real = int(leaf_present.shape[1])
    if target_leaves is None:
        target_leaves = _pad_to_bin(n_leaves_real, _LEAF_BINS)
    if target_L is None:
        target_L = _pad_to_bin(L_real, _COL_BINS)

    # Floor short branch lengths.
    edge_lengths_real = np.maximum(np.asarray(binary_tree.edge_length), 1e-3)
    binary_tree = binary_tree._replace(edge_length=edge_lengths_real)

    padded_tree, edge_mask, ghost_leaf_indices, leaf_index_map, n_real_edges \
        = _pad_tree_with_ghosts(binary_tree, target_leaves)
    leaf_present_pad, col_mask = _pad_leaf_present(leaf_present, target_L)

    le, re = edge_lookup(padded_tree)

    # Build leaf clamp on the padded leaf-node ordering. Real leaves at
    # leaf_index_map positions; ghost leaves at uniform [1,1,1].
    pad_n_internal = int(padded_tree.num_internal)
    leaf_clamp_full = np.zeros((target_leaves, target_L, N_Z),
                                  dtype=np.float64)
    orig_clamp = np.asarray(leaf_clamp_to_beta(leaf_present_pad))
    for i in range(n_leaves_real):
        leaf_local = int(leaf_index_map[i]) - pad_n_internal
        leaf_clamp_full[leaf_local] = orig_clamp[i]
    for gi in ghost_leaf_indices:
        leaf_local = int(gi) - pad_n_internal
        leaf_clamp_full[leaf_local] = 1.0

    # Index of the FIRST edge whose parent is the (new) root — used by
    # entropy_per_column to extract root marginals.
    root_idx = int(padded_tree.root)
    root_edges = np.where(np.asarray(padded_tree.edge_parent) == root_idx)[0]
    if len(root_edges) == 0:
        raise ValueError("No root edge found")
    root_edge_idx = int(root_edges[0])

    # Per-(edge, column, 2) variational logits — full per-column flexibility.
    # Tying across columns would introduce variational bias because the
    # data-conditioned posterior P(z_p, z_c | leaves) varies per column
    # with the leaf observations.  See feedback_varanc_tied_logits.md.
    edge_logits_init = np.zeros(
        (int(padded_tree.num_edges), int(target_L), 2),
        dtype=np.float64)
    return {
        'padded_tree': padded_tree,
        'edge_mask': np.asarray(edge_mask, dtype=np.float64),
        'col_mask': col_mask,
        'leaf_clamp_full': leaf_clamp_full,
        'edge_logits_init': edge_logits_init,
        'le': np.asarray(le, dtype=np.int32),
        're': np.asarray(re, dtype=np.int32),
        'n_real_edges': int(n_real_edges),
        'n_real_leaves': int(n_leaves_real),
        'L_real': int(L_real),
        'leaf_index_map': np.asarray(leaf_index_map),
        'ghost_leaf_indices': np.asarray(ghost_leaf_indices),
        'target_leaves': int(target_leaves),
        'target_L': int(target_L),
        'edge_length': np.asarray(padded_tree.edge_length, dtype=np.float64),
        'root_edge_idx': root_edge_idx,
    }


def _make_neg_elbo_tkf92_jit(target_L, num_internal, num_nodes, num_edges,
                               root, n_iter, lr):
    """Build JIT-compiled neg_elbo + Adam loop for plain TKF92.

    Cache key depends only on (target_L, num_internal, num_nodes,
    num_edges, root, n_iter, lr) — all Python scalars. Tree topology
    arrays go in as JIT-traced arguments.
    """
    eye_3_const = jnp.eye(N_Z, dtype=jnp.float64)

    def _build_q_cond(edge_logits, edge_mask):
        # edge_logits: (E, L, 2) — per-(edge, col) free logits.
        learned_q = make_q_conditionals(edge_logits)  # (E, L, 3, 3)
        identity_q = jnp.broadcast_to(eye_3_const, (num_edges, target_L, N_Z, N_Z))
        em = edge_mask[:, None, None, None]
        return em * learned_q + (1.0 - em) * identity_q

    def neg_elbo_inner(edge_logits, root_logit,
                        leaf_clamp, edge_lengths,
                        ins_rate, del_rate, ext,
                        edge_mask, col_mask, root_edge_idx_arr,
                        left_child, right_child, left_edge, right_edge,
                        edge_parent, edge_child, postorder, preorder):
        q_cond = _build_q_cond(edge_logits, edge_mask)
        root_logits = jnp.broadcast_to(root_logit, (target_L,))
        root_dist = make_root_dist(root_logits)

        pair_marg, log_Z = _bp_pair_marginals_jax(
            q_cond, root_dist, leaf_clamp,
            num_internal, num_nodes, num_edges, root,
            left_child, right_child, left_edge, right_edge,
            edge_parent, edge_child, postorder, preorder)

        log_T_per_edge = jax.vmap(
            lambda t: tkf92_wfst_log_T(ins_rate, del_rate, t, ext))(edge_lengths)
        branch_LLs = jax.vmap(
            lambda pm, lt: _expected_branch_LL_masked(pm, lt, col_mask)
        )(pair_marg, log_T_per_edge)
        sum_branch_LL = jnp.sum(branch_LLs * edge_mask)

        H_per_col = _entropy_per_column_masked_argonly(
            pair_marg, root_dist, q_cond,
            edge_mask, col_mask, root_edge_idx_arr)
        H_total = jnp.sum(H_per_col)

        log_prior = _singlet_root_log_prior_tkf92_masked(
            root_dist, ins_rate, del_rate, ext, col_mask)

        sum_log_Z = jnp.sum(log_Z * col_mask)

        return -(sum_branch_LL + log_prior + H_total + sum_log_Z)

    @jax.jit
    def run_adam(edge_logits, root_logit,
                  leaf_clamp, edge_lengths,
                  ins_rate, del_rate, ext,
                  edge_mask, col_mask, root_edge_idx_arr,
                  left_child, right_child, left_edge, right_edge,
                  edge_parent, edge_child, postorder, preorder):
        optimizer = optax.adam(lr)
        state = optimizer.init((edge_logits, root_logit))

        def step(carry, _):
            p, st = carry
            grad_fn = jax.grad(neg_elbo_inner, argnums=(0, 1))
            g = grad_fn(*p, leaf_clamp, edge_lengths,
                         ins_rate, del_rate, ext,
                         edge_mask, col_mask, root_edge_idx_arr,
                         left_child, right_child, left_edge, right_edge,
                         edge_parent, edge_child, postorder, preorder)
            g_edge, g_root = g
            # Mask gradient on ghost edges (edge_logits shape: E, L, 2).
            g_edge = g_edge * edge_mask[:, None, None]
            g = (g_edge, g_root)
            u, st = optimizer.update(g, st)
            new_p = optax.apply_updates(p, u)
            return (new_p, st), None

        (final_p, _), _ = jax.lax.scan(
            step, ((edge_logits, root_logit), state), None, length=n_iter)
        return final_p

    @jax.jit
    def compute_neg_elbo(edge_logits, root_logit,
                          leaf_clamp, edge_lengths,
                          ins_rate, del_rate, ext,
                          edge_mask, col_mask, root_edge_idx_arr,
                          left_child, right_child, left_edge, right_edge,
                          edge_parent, edge_child, postorder, preorder):
        return neg_elbo_inner(edge_logits, root_logit,
                               leaf_clamp, edge_lengths,
                               ins_rate, del_rate, ext,
                               edge_mask, col_mask, root_edge_idx_arr,
                               left_child, right_child, left_edge, right_edge,
                               edge_parent, edge_child, postorder, preorder)

    @jax.jit
    def compute_pair_marg(edge_logits, root_logit,
                            leaf_clamp, edge_mask,
                            left_child, right_child, left_edge, right_edge,
                            edge_parent, edge_child, postorder, preorder):
        q_cond = _build_q_cond(edge_logits, edge_mask)
        root_logits = jnp.broadcast_to(root_logit, (target_L,))
        root_dist = make_root_dist(root_logits)
        pair_marg, _ = _bp_pair_marginals_jax(
            q_cond, root_dist, leaf_clamp,
            num_internal, num_nodes, num_edges, root,
            left_child, right_child, left_edge, right_edge,
            edge_parent, edge_child, postorder, preorder)
        return pair_marg

    return run_adam, compute_neg_elbo, compute_pair_marg


# Cache (target_L, num_internal, num_nodes, num_edges, root, n_iter, lr)
# -> (run_adam, compute_neg_elbo, compute_pair_marg).
_TKF92_JIT_CACHE = {}


def get_tkf92_padded_jit(target_L, num_internal, num_nodes, num_edges,
                            root, n_iter, lr):
    key = (int(target_L), int(num_internal), int(num_nodes),
            int(num_edges), int(root), int(n_iter), float(lr))
    if key not in _TKF92_JIT_CACHE:
        _TKF92_JIT_CACHE[key] = _make_neg_elbo_tkf92_jit(
            target_L, num_internal, num_nodes, num_edges, root, n_iter, lr)
    return _TKF92_JIT_CACHE[key]


def fit_family_padded_tkf92(binary_tree, leaf_present,
                                ins_rate, del_rate, ext,
                                n_iter=30, lr=0.05,
                                target_leaves=None, target_L=None):
    """Run padded-ELBO Adam for one family.

    Returns dict with 'pair_marg', 'edge_logits', 'root_logit',
    'neg_elbo_final', plus the padded inputs for downstream use.
    """
    inputs = pad_tkf92_inputs(binary_tree, leaf_present,
                                  target_leaves=target_leaves,
                                  target_L=target_L)
    pt = inputs['padded_tree']

    run_adam, compute_neg_elbo, compute_pair_marg = get_tkf92_padded_jit(
        inputs['target_L'], int(pt.num_internal), int(pt.num_nodes),
        int(pt.num_edges), int(pt.root), n_iter, lr)

    edge_logits_j = jnp.asarray(inputs['edge_logits_init'])
    root_logit_j = jnp.zeros((), dtype=jnp.float64)
    leaf_clamp_j = jnp.asarray(inputs['leaf_clamp_full'])
    edge_length_j = jnp.asarray(inputs['edge_length'])

    edge_mask_j = jnp.asarray(inputs['edge_mask'])
    col_mask_j = jnp.asarray(inputs['col_mask'])
    root_edge_idx_arr = jnp.asarray(inputs['root_edge_idx'])

    left_child_j = jnp.asarray(pt.left_child)
    right_child_j = jnp.asarray(pt.right_child)
    le_j = jnp.asarray(inputs['le'])
    re_j = jnp.asarray(inputs['re'])
    edge_parent_j = jnp.asarray(pt.edge_parent)
    edge_child_j = jnp.asarray(pt.edge_child)
    postorder_j = jnp.asarray(pt.postorder_internal)
    preorder_j = jnp.asarray(pt.preorder_internal)

    final_logits, final_root = run_adam(
        edge_logits_j, root_logit_j,
        leaf_clamp_j, edge_length_j,
        jnp.asarray(ins_rate), jnp.asarray(del_rate), jnp.asarray(ext),
        edge_mask_j, col_mask_j, root_edge_idx_arr,
        left_child_j, right_child_j, le_j, re_j,
        edge_parent_j, edge_child_j, postorder_j, preorder_j)

    neg_elbo = compute_neg_elbo(
        final_logits, final_root,
        leaf_clamp_j, edge_length_j,
        jnp.asarray(ins_rate), jnp.asarray(del_rate), jnp.asarray(ext),
        edge_mask_j, col_mask_j, root_edge_idx_arr,
        left_child_j, right_child_j, le_j, re_j,
        edge_parent_j, edge_child_j, postorder_j, preorder_j)

    pair_marg = compute_pair_marg(
        final_logits, final_root,
        leaf_clamp_j, edge_mask_j,
        left_child_j, right_child_j, le_j, re_j,
        edge_parent_j, edge_child_j, postorder_j, preorder_j)

    return {
        'pair_marg': np.asarray(pair_marg),
        'edge_logits': np.asarray(final_logits),
        'root_logit': float(final_root),
        'neg_elbo': float(neg_elbo),
        'inputs': inputs,
    }
