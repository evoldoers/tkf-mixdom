"""JAX-vectorised partition-conditioned ancestral reconstruction.

This is the fully vectorised counterpart to `partition_recon.py`. The
algorithm is specified in tkf/partition-recon.tex and a correctness-
oriented plain-NumPy reference lives in `partition_recon.py`.

Design
------

The heavy lifting is the computation of log G(k+1, l, n) for every
(k, l, n). We vectorise by:

- Precomputing per-edge log-transition tensors (n_edges, N, 5, 5).
- Precomputing per-column branch types (n_edges, L) with -1 for
  untouched columns.
- Precomputing per-column per-domain Felsenstein log-likelihoods and
  per-column per-domain log root posteriors (these are cached and
  reused by both the Python and JAX drivers).
- For each of L starting columns k we run a `jax.lax.scan` over
  columns l = k..L-1 that carries (last_state, lp_branch, fels_sum)
  across columns and produces log G(k, l, n) at every position. This
  inner scan is vmapped across the N domain classes.
- The outer loop over starting columns k is itself a vmap, yielding
  the full (padded_L, padded_L, N) table.

Padding
-------

To reuse JIT-compiled functions across MSAs of different L we pad L
to the next geometric bin (shared with `dp.hmm`). Padded columns
have branch types set to -1 (untouched) and Felsenstein log-
likelihoods set to 0, so they contribute nothing to the DP. The
close-to-E factor at padded positions is still computed, but those
`G[k, l, n]` entries for k >= seq_length or l >= seq_length are
masked to -inf before being fed into the Forward/Backward.

The Forward/Backward over blocks is a cheap O(L^2 N) sequential DP
and is kept in plain NumPy (pulled out of JIT); the JAX cost is
concentrated in the G computation.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Optional, Tuple

from ..core.params import S, M, I, D, E
from ..core.ctmc import transition_matrix
from ..core.params import tkf92_trans
from ..dp.hmm import _pad_to_bin, NEG_INF as _DP_NEG_INF
from ..util.io import TreeNode
from .tree_varanc import infer_internal_presence, name_internal_nodes

from .partition_recon import (
    PartitionReconInputs, PartitionReconModel, PartitionReconResult,
    build_inputs, _enumerate_edges, _compute_branch_types,
    _build_log_trans_per_edge_tkf91,
    _compute_presence_profile, _felsenstein_column,
    _forward, _backward,
    _posterior_class_per_column, _mix_root_posterior,
    _compute_log_emission,
    _safe_log, _logsumexp_np, NEG_INF,
)


# ---------------------------------------------------------------------------
# JAX G kernel
# ---------------------------------------------------------------------------

def _g_inner_scan(log_trans: jnp.ndarray,
                  branch_types: jnp.ndarray,
                  fels_logliks: jnp.ndarray,
                  start_k: jnp.ndarray,
                  padded_L: int,
                  ) -> jnp.ndarray:
    """Compute log G(k+1, l, n) for every l >= k and every n, for one fixed k.

    Args:
        log_trans: (n_b, N, 5, 5) log-transition matrices per edge per domain.
        branch_types: (n_b, padded_L) int array in {-1, M, I, D}.
        fels_logliks: (padded_L, N) log column likelihoods.
        start_k: scalar traced integer, the starting column index (0-indexed;
            the block covers columns start_k..l in 0-indexed terms).
        padded_L: Python int, padded MSA length.

    Returns:
        G_row: (padded_L, N) array. Entry [l, n] = log G(k+1, l+1, n) for
            l >= start_k; entries with l < start_k are -inf.
    """
    n_b = log_trans.shape[0]
    N = log_trans.shape[1]

    # Initial state: every branch at S, lp_branch = 0, fels_sum = 0.
    init_last = jnp.full((N, n_b), S, dtype=jnp.int32)
    init_lp = jnp.zeros((N, n_b), dtype=jnp.float64)
    init_fels = jnp.zeros((N,), dtype=jnp.float64)

    close_lp = log_trans[:, :, :, E]  # (n_b, N, 5)

    b_idx = jnp.arange(n_b)
    n_range = jnp.arange(N)

    def step(carry, l):
        last_state, lp_branch, fels_sum = carry

        active = l >= start_k

        types_l = branch_types[:, l]  # (n_b,)
        touched = (types_l >= 0) & active  # (n_b,)

        new_state = jnp.where(types_l >= 0, types_l, 0)  # (n_b,) int

        def per_n(n):
            ls_n = last_state[n]  # (n_b,)
            vals = log_trans[b_idx, n, ls_n, new_state]  # (n_b,)
            return vals
        trans_inc = jax.vmap(per_n)(n_range)  # (N, n_b)

        trans_inc = jnp.where(touched[None, :], trans_inc, 0.0)

        new_lp = lp_branch + trans_inc
        new_last = jnp.where(touched[None, :], new_state[None, :], last_state)

        fels_add = jnp.where(active, fels_logliks[l], jnp.zeros_like(fels_sum))
        new_fels = fels_sum + fels_add

        def close_per_n(n):
            nl = new_last[n]  # (n_b,)
            vals = close_lp[b_idx, n, nl]  # (n_b,)
            return jnp.sum(vals)
        close_sum = jax.vmap(close_per_n)(n_range)  # (N,)

        block_open = new_fels + jnp.sum(new_lp, axis=1)  # (N,)
        log_G_l_n = block_open + close_sum  # (N,)
        log_G_l_n = jnp.where(active, log_G_l_n,
                              jnp.full_like(log_G_l_n, _DP_NEG_INF))

        new_carry = (new_last, new_lp, new_fels)
        return new_carry, log_G_l_n

    ls = jnp.arange(padded_L)
    _, G_row = jax.lax.scan(step, (init_last, init_lp, init_fels), ls)
    return G_row  # (padded_L, N)


@partial(jax.jit, static_argnums=(3,))
def _compute_G_jax(log_trans: jnp.ndarray,
                   branch_types: jnp.ndarray,
                   fels_logliks: jnp.ndarray,
                   padded_L: int,
                   ) -> jnp.ndarray:
    """Compute (padded_L, padded_L, N) block log-likelihoods via nested scans.

    Entry [k, l, n] = log G(k+1, l+1, n) for l >= k; -inf otherwise.

    JIT-compiled with `padded_L` as a static argument so the compiled
    function is reused across inputs that pad to the same bin.
    """
    def per_k(k):
        return _g_inner_scan(log_trans, branch_types, fels_logliks,
                             k, padded_L)

    G = jax.vmap(per_k)(jnp.arange(padded_L))  # (padded_L, padded_L, N)
    return G


# ---------------------------------------------------------------------------
# JAX G kernel for D x F fragment tracking (n_frag > 1)
# ---------------------------------------------------------------------------

def _g_inner_scan_frag(log_trans_tkf91: jnp.ndarray,
                       branch_types: jnp.ndarray,
                       log_emission: jnp.ndarray,
                       profile_ids: jnp.ndarray,
                       log_ext: jnp.ndarray,
                       log_frag_wts: jnp.ndarray,
                       log_notext: jnp.ndarray,
                       start_k: jnp.ndarray,
                       padded_L: int,
                       n_frag: int,
                       ) -> jnp.ndarray:
    """Compute log G(k+1, l, n) for every l >= k and n, for one fixed k.

    D x F algorithm: tracks fragment state within blocks.

    Args:
        log_trans_tkf91: (n_b, N, 5, 5) TKF91 (ext=0) log-transitions.
        branch_types: (n_b, padded_L) int array in {-1, M, I, D}.
        log_emission: (padded_L, N, F) log column emission. For
            class-marginalised models, this is just the per-domain
            Felsenstein loglik broadcast across F. For MixDom2 with a
            per-fragment class mixture, entry [l, n, g] = log sum_c
            classdist_{n,g,c} * U(l, c) where U(l, c) is the per-class
            column likelihood.
        profile_ids: (padded_L,) int array of presence profile ids.
        log_ext: (N, F, F) log fragment extension matrix.
        log_frag_wts: (N, F) log fragment weights.
        log_notext: (N, F) log notext probabilities.
        start_k: scalar traced integer.
        padded_L: Python int.
        n_frag: Python int (F).

    Returns:
        G_row: (padded_L, N) array.
    """
    n_b = log_trans_tkf91.shape[0]
    N = log_trans_tkf91.shape[1]
    F = n_frag

    init_last = jnp.full((N, n_b), S, dtype=jnp.int32)
    # F_intra: (N, F) intra-block forward in log-space.
    init_F_intra = jnp.full((N, F), _DP_NEG_INF, dtype=jnp.float64)
    init_prev_profile = jnp.array(-1, dtype=jnp.int32)

    close_lp = log_trans_tkf91[:, :, :, E]  # (n_b, N, 5)
    b_idx = jnp.arange(n_b)
    n_range = jnp.arange(N)

    def step(carry, l):
        last_state, F_intra, prev_profile = carry

        active = l >= start_k
        is_first = (l == start_k)

        types_l = branch_types[:, l]
        touched = (types_l >= 0) & active
        new_state = jnp.where(types_l >= 0, types_l, 0)

        # Per-branch TKF91 log-transition increment.
        def per_n_branch(n):
            ls_n = last_state[n]
            vals = log_trans_tkf91[b_idx, n, ls_n, new_state]
            return vals
        trans_inc = jax.vmap(per_n_branch)(n_range)  # (N, n_b)
        trans_inc = jnp.where(touched[None, :], trans_inc, 0.0)
        log_branch_inc = jnp.sum(trans_inc, axis=1)  # (N,)

        new_last = jnp.where(touched[None, :], new_state[None, :], last_state)

        cur_profile = profile_ids[l]
        same_profile = (cur_profile == prev_profile) & (~is_first)

        # Emission per (n, g): log sum_c classdist[n,g,c] * U(l,c).
        # When the model has no class structure, log_emission[l,n,g] is the
        # per-domain Felsenstein loglik broadcast across g.
        log_emit = jnp.where(
            active,
            log_emission[l],
            jnp.zeros((N, F), dtype=jnp.float64),
        )  # (N, F)

        # Intra-block forward recurrence.
        def first_col_F(n):
            # Base case: F_{k,k,d,g} = T_start * fragdist_g * E(d,k,g)
            return log_branch_inc[n] + log_frag_wts[n] + log_emit[n]
            # Returns (F,)

        def recurse_F(n):
            # F_intra[n, :] is the old F (previous column).
            # new_F[g] = logsumexp_f(F_old[f] + T_col(f,g)) + E(d,l,g)
            # T_col(f,g) = same_profile * ext_{fg} + notext_f * branch_inc * fragdist_g
            def per_g(g):
                # For each f, compute log(T_col(f,g))
                t_ext = log_ext[n, :, g]  # (F,)
                t_notext = log_notext[n, :] + log_branch_inc[n] + log_frag_wts[n, g]  # (F,)

                # When same_profile: logsumexp(t_ext, t_notext) per f
                # When different profile: just t_notext per f
                t_col_same = jnp.logaddexp(t_ext, t_notext)  # (F,)
                t_col = jnp.where(same_profile, t_col_same, t_notext)

                log_terms = F_intra[n, :] + t_col  # (F,)
                return jax.scipy.special.logsumexp(log_terms) + log_emit[n, g]
            return jax.vmap(per_g)(jnp.arange(F))  # (F,)

        def compute_new_F(n):
            first = first_col_F(n)   # (F,)
            recur = recurse_F(n)     # (F,)
            return jnp.where(is_first, first, recur)

        new_F_intra = jax.vmap(compute_new_F)(n_range)  # (N, F)
        new_F_intra = jnp.where(active, new_F_intra,
                                jnp.full_like(new_F_intra, _DP_NEG_INF))

        # Block close: G = sum_f F_intra[n,f] * notext[n,f] * T_end
        def close_per_n(n):
            nl = new_last[n]
            close_vals = close_lp[b_idx, n, nl]
            log_t_end = jnp.sum(close_vals)

            log_terms = new_F_intra[n, :] + log_notext[n, :]  # (F,)
            return jax.scipy.special.logsumexp(log_terms) + log_t_end

        log_G_l_n = jax.vmap(close_per_n)(n_range)  # (N,)
        log_G_l_n = jnp.where(active, log_G_l_n,
                              jnp.full_like(log_G_l_n, _DP_NEG_INF))

        new_carry = (new_last, new_F_intra, cur_profile)
        return new_carry, log_G_l_n

    ls = jnp.arange(padded_L)
    _, G_row = jax.lax.scan(
        step,
        (init_last, init_F_intra, init_prev_profile),
        ls)
    return G_row  # (padded_L, N)


@partial(jax.jit, static_argnums=(4, 5))
def _compute_G_jax_frag(log_trans_tkf91: jnp.ndarray,
                        branch_types: jnp.ndarray,
                        log_emission: jnp.ndarray,
                        frag_params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                        padded_L: int,
                        n_frag: int,
                        profile_ids: jnp.ndarray,
                        ) -> jnp.ndarray:
    """Compute (padded_L, padded_L, N) block log-likelihoods for D x F.

    JIT-compiled with padded_L and n_frag as static arguments.

    ``log_emission`` has shape (padded_L, N, F). For class-marginalised
    models pass ``fels_logliks[:, :, None].repeat(F, axis=-1)``; for
    MixDom2 with per-fragment class mixture pass the result of
    `_compute_log_emission` (broadcast to padded_L).
    """
    log_ext, log_frag_wts, log_notext = frag_params

    def per_k(k):
        return _g_inner_scan_frag(
            log_trans_tkf91, branch_types, log_emission,
            profile_ids, log_ext, log_frag_wts, log_notext,
            k, padded_L, n_frag)

    G = jax.vmap(per_k)(jnp.arange(padded_L))
    return G


# ---------------------------------------------------------------------------
# Felsenstein precomputation
#
# Two implementations:
#   _precompute_felsenstein_np: reference, per-(col, n, child) eigh call.
#   _precompute_felsenstein_pt_cached: optimisation — batched eigh once
#     per (model, edge), then per-column peel via Pt lookup. Numerically
#     identical to the reference (asserts in regression test).
#   _precompute_felsenstein_jax_vmap: full JAX vectorisation — vmaps the
#     per-column peel using the same Pt cache plus a fixed-tree topology,
#     with absent children masked out. Numerically identical (asserts).
# ---------------------------------------------------------------------------


def _batched_log_pts(model_Q: np.ndarray, model_pi: np.ndarray,
                      edge_lengths: np.ndarray) -> np.ndarray:
    """Compute log_Pts[m, edge_idx] = log expm(Q[m] · t[edge_idx]) for all (m, edge).

    Uses one batched ``jax.vmap`` over the (M × n_edges) flattened axis,
    reducing the eigendecomposition count from L · M · n_edges (per-column
    loop in the reference) to M · n_edges. ``transition_matrix``
    is exact (eigendecomposition of the symmetrised generator); calling
    it inside vmap is exact too.

    Edges with t=0 (e.g. the virtual above-root edge) get Pt = identity,
    which has -inf off-diagonal in log-space. Such edges are never used in
    the Felsenstein peel (the virtual edge is skipped).
    """
    M = int(model_Q.shape[0])
    n_edges = int(len(edge_lengths))
    A = int(model_Q.shape[1])

    Q_j = jnp.asarray(model_Q)             # (M, A, A)
    pi_j = jnp.asarray(model_pi)           # (M, A)
    t_j = jnp.asarray(edge_lengths)        # (n_edges,)

    Q_batch = jnp.broadcast_to(
        Q_j[:, None, :, :], (M, n_edges, A, A)).reshape(M * n_edges, A, A)
    t_batch = jnp.repeat(t_j[None, :], M, axis=0).reshape(M * n_edges)

    Pts_flat = jax.vmap(transition_matrix,
                        in_axes=(0, 0))(Q_batch, t_batch)
    Pts = Pts_flat.reshape(M, n_edges, A, A)
    log_Pts = jnp.log(jnp.maximum(Pts, 1e-300))
    return np.asarray(log_Pts)


def _felsenstein_column_with_pts(tree, col, presence, leaf_seqs_aligned,
                                  log_pi, log_Pts, child_to_edge_idx):
    """Felsenstein peel at one column with cached log_Pt[edge_idx, A, A].

    Numerically identical to ``_felsenstein_column`` (including wildcard /
    absent-leaf / shallowest-present-root logic). The only difference is
    that ``log_Pt`` is looked up by child name via ``child_to_edge_idx``
    instead of recomputed via eigendecomposition each call.
    """
    A = int(log_pi.shape[0])
    present = {name for name in presence if presence[name][col]}

    if not present:
        return 0.0, None

    log_partial = {}

    for node in tree.postorder():
        if node.name not in present:
            continue
        if node.is_leaf:
            char = int(leaf_seqs_aligned.get(node.name,
                                              np.full(col + 1, -1))[col])
            log_L = np.full(A, NEG_INF)
            if char < 0:
                log_L = np.zeros(A)
            elif char >= A:
                log_L = np.zeros(A)
            else:
                log_L[char] = 0.0
            log_partial[node.name] = log_L
        else:
            log_L = np.zeros(A)
            for child in node.children:
                if child.name not in present:
                    continue
                edge_idx = child_to_edge_idx[child.name]
                log_Pt = log_Pts[edge_idx]
                child_log_L = log_partial[child.name]
                contrib = _logsumexp_np(
                    log_Pt + child_log_L[None, :], axis=1)
                log_L = log_L + contrib
            log_partial[node.name] = log_L

    root_of_present = None
    for node in tree.preorder():
        if node.name in present:
            root_of_present = node
            break

    log_L_present_root = log_partial[root_of_present.name]
    log_joint_at_pres_root = log_pi + log_L_present_root
    log_col = _logsumexp_np(log_joint_at_pres_root)

    if tree.name in present:
        log_root_joint = log_pi + log_partial[tree.name]
        log_root_Z = _logsumexp_np(log_root_joint)
        log_root_post = log_root_joint - log_root_Z
    else:
        log_root_post = None

    return float(log_col), log_root_post


def _precompute_felsenstein_pt_cached(inputs: PartitionReconInputs,
                                       model: PartitionReconModel,
                                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  Optional[np.ndarray]]:
    """Pt-cache version of ``_precompute_felsenstein_np`` (Stage 1).

    Numerically identical output to the reference; uses one batched eigh
    per (model, edge) up-front (via ``_batched_log_pts``), then runs the
    same per-column tree peel via Pt lookup. Avoids the O(L) re-evaluation
    of the matrix exponential per branch.
    """
    L = inputs.L
    N = model.n_dom
    A = model.A

    edges = _enumerate_edges(inputs.tree)
    edge_lengths = np.array([
        max(float(e[2]), 1e-8) if e[2] is not None else 0.0
        for e in edges
    ], dtype=np.float64)
    child_to_edge_idx = {e[1]: ei for ei, e in enumerate(edges)}

    # Per-domain Pt cache
    log_Pts_dom = _batched_log_pts(
        np.asarray(model.Q), np.asarray(model.pi), edge_lengths)
    # (N, n_edges, A, A)

    fels_logliks = np.zeros((L, N), dtype=np.float64)
    root_log_post = np.zeros((L, N, A), dtype=np.float64)
    root_present = np.asarray(inputs.presence[inputs.tree.name], dtype=bool)

    pi_np = np.asarray(model.pi)
    for n in range(N):
        log_pi_n = _safe_log(pi_np[n])
        for col in range(L):
            ll, rp = _felsenstein_column_with_pts(
                inputs.tree, col, inputs.presence,
                inputs.leaf_seqs_aligned,
                log_pi_n, log_Pts_dom[n], child_to_edge_idx)
            fels_logliks[col, n] = ll
            if rp is not None:
                root_log_post[col, n] = rp
            else:
                root_log_post[col, n] = -np.log(A)

    # Per-class
    class_logliks = None
    class_root_log_post = None
    if model.class_Q is not None and model.class_pi is not None:
        C = int(model.n_class)
        class_pi_np = np.asarray(model.class_pi)
        log_Pts_cls = _batched_log_pts(
            np.asarray(model.class_Q), class_pi_np, edge_lengths)
        # (C, n_edges, A, A)
        class_logliks = np.zeros((L, C), dtype=np.float64)
        class_root_log_post = np.zeros((L, C, A), dtype=np.float64)
        for c in range(C):
            log_pi_c = _safe_log(class_pi_np[c])
            for col in range(L):
                ll, rp = _felsenstein_column_with_pts(
                    inputs.tree, col, inputs.presence,
                    inputs.leaf_seqs_aligned,
                    log_pi_c, log_Pts_cls[c], child_to_edge_idx)
                class_logliks[col, c] = ll
                if rp is not None:
                    class_root_log_post[col, c] = rp
                else:
                    class_root_log_post[col, c] = -np.log(A)

    return (fels_logliks, root_log_post, root_present, class_logliks,
            class_root_log_post)


# ---------------------------------------------------------------------------
# Stage 2: full JAX vectorisation of the per-column peel
#
# Reuses the Pt cache from Stage 1 (one batched eigh per (model, edge)),
# then vmaps the per-column tree peel using a fixed topology with
# presence-mask gating of absent children. Numerically identical to the
# reference; verified by regression test.
# ---------------------------------------------------------------------------


def _build_tree_topology(tree):
    """Flatten the tree into static arrays/lists used by the JAX kernel.

    Returns dict with:
      n_total: total node count
      node_names: (n_total,) list of names — preorder
      name_to_idx: dict name → idx
      leaf_indices: list of leaf indices
      internal_postorder: list of internal-node indices in postorder
      children_of: dict {internal_idx: [(child_idx, edge_idx), ...]}
      preorder_indices: (n_total,) np.int32 (= range(n_total) since we ordered preorder)
      root_idx: int
      edges: list of edge tuples (matches _enumerate_edges)
      edge_lengths: (n_edges,) np.float64
      child_to_edge_idx: dict child_name → edge_idx
    """
    nodes_preorder = list(tree.preorder())
    name_to_idx = {n.name: i for i, n in enumerate(nodes_preorder)}
    leaf_indices = [i for i, n in enumerate(nodes_preorder) if n.is_leaf]

    internal_postorder = []
    for n in tree.postorder():
        if not n.is_leaf:
            internal_postorder.append(name_to_idx[n.name])

    edges = _enumerate_edges(tree)
    child_to_edge_idx = {e[1]: ei for ei, e in enumerate(edges)}
    edge_lengths = np.array(
        [max(float(e[2]), 1e-8) if e[2] is not None else 0.0 for e in edges],
        dtype=np.float64)

    children_of = {}
    for v in internal_postorder:
        node = nodes_preorder[v]
        kids = []
        for child in node.children:
            kids.append((name_to_idx[child.name],
                         child_to_edge_idx[child.name]))
        children_of[v] = kids

    return dict(
        n_total=len(nodes_preorder),
        node_names=[n.name for n in nodes_preorder],
        name_to_idx=name_to_idx,
        leaf_indices=leaf_indices,
        internal_postorder=internal_postorder,
        children_of=children_of,
        preorder_indices=np.arange(len(nodes_preorder), dtype=np.int32),
        root_idx=name_to_idx[tree.name],
        edges=edges,
        edge_lengths=edge_lengths,
        child_to_edge_idx=child_to_edge_idx,
    )


def _build_leaf_log_partial(topology, presence_arr, leaf_chars_arr, A: int,
                              L: int):
    """Build (n_total, L, A) initial log_partial table for leaves.

    For each leaf:
      - col absent OR char < 0 OR char >= A → log_partial[leaf, col, :] = 0
      - char ∈ [0, A) AND col present → log_partial[leaf, col, char] = 0,
                                          others = NEG_INF

    Internal nodes are placeholders (zeros); they're overwritten by the peel.
    """
    n_total = topology['n_total']
    log_partial = np.zeros((n_total, L, A), dtype=np.float64)
    for leaf_local_i, leaf_idx in enumerate(topology['leaf_indices']):
        chars = leaf_chars_arr[leaf_local_i]  # (L,) int
        pres = presence_arr[leaf_idx]  # (L,) bool
        for col in range(L):
            char = int(chars[col])
            if not pres[col]:
                # absent → uniform contribution (zeros)
                continue
            if char < 0 or char >= A:
                # wildcard / weird → uniform
                continue
            log_partial[leaf_idx, col, :] = NEG_INF
            log_partial[leaf_idx, col, char] = 0.0
    return log_partial


def _root_of_present_per_col(topology, presence_arr, L: int) -> np.ndarray:
    """For each col, the index of the shallowest preorder node that is present.

    Returns (L,) int. Cols with no present nodes get index 0 (caller masks).
    """
    preorder_idx = topology['preorder_indices']  # (n_total,)
    presence_in_preorder = presence_arr[preorder_idx, :]  # (n_total, L)
    # First True per col along axis=0
    first_present = np.argmax(presence_in_preorder, axis=0)  # (L,)
    return preorder_idx[first_present]


def _vec_fels_kernel_one_model(log_pi: jnp.ndarray,
                                 log_Pts: jnp.ndarray,
                                 leaf_log_partial: jnp.ndarray,
                                 presence_jax: jnp.ndarray,
                                 root_of_present: jnp.ndarray,
                                 root_idx: int,
                                 root_present: jnp.ndarray,
                                 internal_postorder,
                                 children_of,
                                 A: int,
                                 ):
    """Vectorised per-column Felsenstein for one (Q, pi) model.

    log_pi: (A,)
    log_Pts: (n_edges, A, A)
    leaf_log_partial: (n_total, L, A) — leaf entries set; internals = 0 placeholder
    presence_jax: (n_total, L) bool
    root_of_present: (L,) int — shallowest preorder present node per col
    root_idx: int (overall tree root)
    root_present: (L,) bool — whether overall root is present per col
    internal_postorder, children_of: static topology (Python objects)

    Returns:
      log_col: (L,)
      log_root_post: (L, A) — for cols where root is absent, set to -log(A)
    """
    log_partial = leaf_log_partial

    for v in internal_postorder:
        contribs = []
        for child_idx, edge_idx in children_of[v]:
            log_pt = log_Pts[edge_idx]  # (A, A)
            child_lp = log_partial[child_idx]  # (L, A)
            # contrib[col, a] = logsumexp_b(log_pt[a, b] + child_lp[col, b])
            combined = log_pt[None, :, :] + child_lp[:, None, :]
            # (L, A, A) — last axis = b
            contrib = jax.scipy.special.logsumexp(combined, axis=2)  # (L, A)
            child_pres = presence_jax[child_idx]  # (L,) bool
            contrib_masked = jnp.where(child_pres[:, None], contrib, 0.0)
            contribs.append(contrib_masked)
        if contribs:
            new_v = sum(contribs[1:], contribs[0])  # (L, A)
        else:
            new_v = jnp.zeros_like(log_partial[v])
        log_partial = log_partial.at[v].set(new_v)

    # log_col: gather log_partial[root_of_present[col], col, :]
    L = log_partial.shape[1]
    cols = jnp.arange(L)
    gathered = log_partial[root_of_present, cols, :]  # (L, A)
    log_joint = log_pi[None, :] + gathered  # (L, A)
    log_col = jax.scipy.special.logsumexp(log_joint, axis=1)  # (L,)

    # Cols with no present nodes: log_col = 0
    any_present = jnp.any(presence_jax, axis=0)  # (L,)
    log_col = jnp.where(any_present, log_col, 0.0)

    # Root posterior at overall root
    root_lp = log_partial[root_idx]  # (L, A)
    log_root_joint = log_pi[None, :] + root_lp  # (L, A)
    log_root_Z = jax.scipy.special.logsumexp(log_root_joint, axis=1)  # (L,)
    log_root_post = log_root_joint - log_root_Z[:, None]  # (L, A)
    log_root_post = jnp.where(root_present[:, None],
                               log_root_post, -jnp.log(float(A)))

    return log_col, log_root_post


def _precompute_felsenstein_jax_vmap(inputs: PartitionReconInputs,
                                       model: PartitionReconModel,
                                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  Optional[np.ndarray]]:
    """Stage 2 — fully vectorised per-column Felsenstein with Pt cache.

    Replaces the per-column Python loop in
    ``_precompute_felsenstein_pt_cached`` with a JAX-traced peel that
    runs the postorder pass over an (L, A) tensor for all cols at once,
    using presence-mask gating for absent direct children.

    Numerically identical to the reference (asserts via regression test):
      - same Pt = expm(Q · t) per (model, edge), batched
      - same per-internal-node aggregation: sum over PRESENT direct children
        of logsumexp_b(log_Pt + child_log_partial)
      - same shallowest-preorder-present-root for log_col
      - same overall-root posterior for log_root_post (uniform when absent)

    No averaging, no consensus, no fallback.
    """
    L = inputs.L
    N = model.n_dom
    A = model.A

    topology = _build_tree_topology(inputs.tree)
    n_edges = len(topology['edges'])

    # Build (n_total, L) presence array.
    n_total = topology['n_total']
    presence_arr = np.zeros((n_total, L), dtype=bool)
    for name, idx in topology['name_to_idx'].items():
        presence_arr[idx] = inputs.presence[name]

    # Build (n_leaves, L) leaf chars array.
    leaf_chars_arr = np.full((len(topology['leaf_indices']), L),
                              -1, dtype=np.int32)
    for li, leaf_idx in enumerate(topology['leaf_indices']):
        name = topology['node_names'][leaf_idx]
        seq = inputs.leaf_seqs_aligned.get(name, np.full(L, -1, dtype=np.int32))
        # `_felsenstein_column` indexes seq[col]; ensure length matches.
        seq_arr = np.asarray(seq, dtype=np.int32)
        if seq_arr.shape[0] < L:
            seq_arr = np.concatenate(
                [seq_arr, np.full(L - seq_arr.shape[0], -1, dtype=np.int32)])
        leaf_chars_arr[li] = seq_arr[:L]

    leaf_log_partial = _build_leaf_log_partial(
        topology, presence_arr, leaf_chars_arr, A, L)

    root_of_present = _root_of_present_per_col(topology, presence_arr, L)
    root_present = presence_arr[topology['root_idx']]  # (L,) bool

    # Per-domain Pt cache (batched eigh).
    Q_np = np.asarray(model.Q)        # (N, A, A)
    pi_np = np.asarray(model.pi)      # (N, A)
    log_Pts_dom_np = _batched_log_pts(
        Q_np, pi_np, topology['edge_lengths'])  # (N, n_edges, A, A)

    # JAX-side vmap over models.
    leaf_lp_j = jnp.asarray(leaf_log_partial)
    presence_j = jnp.asarray(presence_arr)
    rop_j = jnp.asarray(root_of_present, dtype=jnp.int32)
    rp_j = jnp.asarray(root_present)

    def _kernel_one(log_pi_v, log_Pts_v):
        return _vec_fels_kernel_one_model(
            log_pi_v, log_Pts_v, leaf_lp_j, presence_j,
            rop_j, topology['root_idx'], rp_j,
            topology['internal_postorder'], topology['children_of'], A)

    log_pi_dom_j = jnp.log(jnp.maximum(jnp.asarray(pi_np), 1e-300))  # (N, A)
    log_Pts_dom_j = jnp.asarray(log_Pts_dom_np)  # (N, n_edges, A, A)

    log_col_dom, log_root_post_dom = jax.vmap(
        _kernel_one, in_axes=(0, 0))(log_pi_dom_j, log_Pts_dom_j)
    # log_col_dom: (N, L)
    # log_root_post_dom: (N, L, A)

    fels_logliks = np.asarray(log_col_dom).T  # (L, N)
    root_log_post = np.transpose(np.asarray(log_root_post_dom),
                                  (1, 0, 2))  # (L, N, A)

    # Per-class
    class_logliks = None
    class_root_log_post = None
    if model.class_Q is not None and model.class_pi is not None:
        class_Q_np = np.asarray(model.class_Q)
        class_pi_np = np.asarray(model.class_pi)
        log_Pts_cls_np = _batched_log_pts(
            class_Q_np, class_pi_np, topology['edge_lengths'])
        log_pi_cls_j = jnp.log(jnp.maximum(jnp.asarray(class_pi_np), 1e-300))
        log_Pts_cls_j = jnp.asarray(log_Pts_cls_np)

        log_col_cls, log_root_post_cls = jax.vmap(
            _kernel_one, in_axes=(0, 0))(log_pi_cls_j, log_Pts_cls_j)
        # log_col_cls: (C, L); log_root_post_cls: (C, L, A)
        class_logliks = np.asarray(log_col_cls).T  # (L, C)
        class_root_log_post = np.transpose(
            np.asarray(log_root_post_cls), (1, 0, 2))  # (L, C, A)

    return (fels_logliks, root_log_post, root_present, class_logliks,
            class_root_log_post)


# ---------------------------------------------------------------------------
# Felsenstein precomputation: original reference (kept for regression tests)
# ---------------------------------------------------------------------------

def _precompute_felsenstein_np(inputs: PartitionReconInputs,
                               model: PartitionReconModel,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                          Optional[np.ndarray]]:
    """Per-column per-domain Felsenstein loglik + per-column per-class loglik.

    ``class_logliks`` is None when the model has no per-class rate
    matrices; otherwise (L, C). The driver uses it (combined with
    ``model.class_dist``) to build a (L, N, F) per-fragment emission
    table that the JAX D x F kernel consumes.
    """
    L = inputs.L
    N = model.n_dom
    A = model.A
    fels_logliks = np.zeros((L, N), dtype=np.float64)
    root_log_post = np.zeros((L, N, A), dtype=np.float64)
    root_present = np.asarray(inputs.presence[inputs.tree.name], dtype=bool)
    for col in range(L):
        for n in range(N):
            ll, rp = _felsenstein_column(
                inputs.tree, col, inputs.presence,
                inputs.leaf_seqs_aligned,
                np.asarray(model.Q[n]), np.asarray(model.pi[n]))
            fels_logliks[col, n] = ll
            if rp is not None:
                root_log_post[col, n] = rp
            else:
                root_log_post[col, n] = -np.log(A)

    # Per-class column log-likelihoods + root posterior.
    class_logliks = None
    class_root_log_post = None
    if model.class_Q is not None and model.class_pi is not None:
        C = int(model.n_class)
        class_logliks = np.zeros((L, C), dtype=np.float64)
        class_root_log_post = np.zeros((L, C, A), dtype=np.float64)
        for col in range(L):
            for c in range(C):
                ll, rp = _felsenstein_column(
                    inputs.tree, col, inputs.presence,
                    inputs.leaf_seqs_aligned,
                    np.asarray(model.class_Q[c]),
                    np.asarray(model.class_pi[c]))
                class_logliks[col, c] = ll
                if rp is not None:
                    class_root_log_post[col, c] = rp
                else:
                    class_root_log_post[col, c] = -np.log(A)

    return (fels_logliks, root_log_post, root_present, class_logliks,
            class_root_log_post)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def partition_recon_forward_backward_jax(inputs: PartitionReconInputs,
                                          model: PartitionReconModel,
                                          padded_L: Optional[int] = None,
                                          ) -> PartitionReconResult:
    """JAX-backed partition-conditioned Forward-Backward.

    Numerically identical to `partition_recon_forward_backward` on
    unpadded inputs (up to floating-point round-off). Pads the MSA
    length to the next geometric bin so that repeated calls with
    similar L reuse the same JIT-compiled function.

    Args:
        inputs: PartitionReconInputs.
        model: PartitionReconModel.
        padded_L: optional override. If None, uses `dp.hmm._pad_to_bin(L)`.

    Returns:
        PartitionReconResult (same shape as the Python reference).
    """
    L = inputs.L
    if padded_L is None:
        padded_L = int(_pad_to_bin(L))
    padded_L = max(padded_L, L, 1)

    edges = _enumerate_edges(inputs.tree)
    n_b = len(edges)
    N = model.n_dom
    F = model.n_frag

    branch_types_np = _compute_branch_types(edges, inputs.presence, L)
    (fels_logliks_np, root_log_post_np, root_present,
     class_logliks_np,
     class_root_log_post_np) = _precompute_felsenstein_jax_vmap(inputs, model)

    # Build per-(L, N, F) log-emission table: combines class-aware
    # mixing when the model carries class_dist, else broadcasts the
    # per-domain Felsenstein loglik across the F axis.
    log_emission_np = _compute_log_emission(
        model, fels_logliks_np, class_logliks_np)  # (L, N, F)

    if padded_L > L:
        pad_cols = padded_L - L
        branch_types_padded = np.concatenate(
            [branch_types_np,
             np.full((n_b, pad_cols), -1, dtype=np.int32)],
            axis=1)
        log_emission_padded = np.concatenate(
            [log_emission_np, np.zeros((pad_cols, N, F))], axis=0)
    else:
        branch_types_padded = branch_types_np
        log_emission_padded = log_emission_np

    branch_types_j = jnp.asarray(branch_types_padded, dtype=jnp.int32)
    log_emission_j = jnp.asarray(log_emission_padded, dtype=jnp.float64)

    # Always use the D x F algorithm with TKF91 (ext=0) per-branch
    # transitions and column-level fragment extension via ext_matrix.
    log_trans_tkf91_np = _build_log_trans_per_edge_tkf91(edges, model)
    log_trans_tkf91_j = jnp.asarray(log_trans_tkf91_np, dtype=jnp.float64)

    ext_mat = model.get_ext_matrix()      # (N, F, F)
    frag_wts = model.get_frag_weights()   # (N, F)
    notext = model.get_notext()           # (N, F)

    log_ext_j = jnp.asarray(_safe_log(ext_mat), dtype=jnp.float64)
    log_frag_wts_j = jnp.asarray(_safe_log(frag_wts), dtype=jnp.float64)
    log_notext_j = jnp.asarray(_safe_log(notext), dtype=jnp.float64)

    profile_ids_np = _compute_presence_profile(branch_types_np, L)
    if padded_L > L:
        max_id = int(profile_ids_np.max()) + 1 if L > 0 else 0
        pad_ids = np.full(padded_L - L, max_id, dtype=np.int32)
        profile_ids_padded = np.concatenate([profile_ids_np, pad_ids])
    else:
        profile_ids_padded = profile_ids_np
    profile_ids_j = jnp.asarray(profile_ids_padded, dtype=jnp.int32)

    frag_params = (log_ext_j, log_frag_wts_j, log_notext_j)
    G_padded = _compute_G_jax_frag(
        log_trans_tkf91_j, branch_types_j, log_emission_j,
        frag_params, padded_L, F, profile_ids_j)

    G_padded_np = np.asarray(G_padded)

    # Extract the unpadded block.
    G_closed = G_padded_np[:L, :L, :].copy()
    # Mask (k > l) entries to -inf (safety).
    idx_k, idx_l = np.indices((L, L))
    mask = idx_k <= idx_l
    G_closed = np.where(mask[:, :, None], G_closed, NEG_INF)

    # NumPy Forward/Backward (cheap).
    F, bar_F, log_Z_forward = _forward(G_closed, model.dom_weights,
                                       model.kappa_top)
    beta, log_Z_backward = _backward(G_closed, model.dom_weights,
                                     model.kappa_top)
    class_post = _posterior_class_per_column(
        G_closed, bar_F, beta, model.dom_weights, model.kappa_top,
        log_Z_forward)

    # Intra-block FB. Computes:
    #   - `frag_posterior` (L, F): per-column marginal fragment-state posterior
    #   - `site_class_posterior` (L, C): per-column marginal site-class
    #     posterior (zeros when the model has no per-class structure)
    # The reference Python driver runs this whenever F >= 1; the JAX driver
    # had silently been skipping it entirely, which (a) returned
    # `frag_posterior = None` to callers expecting it for non-class
    # MixDom1 models with F >= 2, and (b) forced `_mix_root_posterior` to
    # fall back to per-domain mixing even when per-class root posteriors
    # were available, degrading MixDom2 (Annabel) reconstruction quality.
    # `_compute_frag_class_posteriors` handles the no-class case
    # internally (sets `site_class_posterior` to zeros if class data is
    # missing); `_mix_root_posterior` then sees `class_root_log_post=None`
    # and takes its per-domain fallback branch. Both branches stay
    # consistent with the reference.
    site_class_post = None
    frag_post = None
    if model.n_frag >= 1:
        from .partition_recon import (
            _compute_intra_block_forward,
            _compute_intra_block_backward,
            _compute_frag_class_posteriors,
        )
        # `log_emission_np` was already computed above (per-(L, N, F)).
        F_table, log_branch_inc_table, last_state_table, profile_ids = \
            _compute_intra_block_forward(inputs, model, log_emission_np)
        B_table = _compute_intra_block_backward(
            inputs, model, log_emission_np,
            log_branch_inc_table, last_state_table, profile_ids)
        frag_post, site_class_post = _compute_frag_class_posteriors(
            F_table, B_table, G_closed, bar_F, beta, model,
            log_Z_forward, class_logliks_np)

    root_post, root_map = _mix_root_posterior(
        class_post, root_log_post_np, root_present,
        site_class_posterior=site_class_post,
        class_root_log_post=class_root_log_post_np)

    return PartitionReconResult(
        log_Z_forward=float(log_Z_forward),
        log_Z_backward=float(log_Z_backward),
        class_posterior=class_post,
        root_residue_posterior=root_post,
        root_residue_map=root_map,
        root_is_present=root_present,
        F=F,
        beta=beta,
        G_closed=G_closed,
        frag_posterior=frag_post,
        site_class_posterior=site_class_post,
    )