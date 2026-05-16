"""Mixture-of-trees variational TKF92 ancestral presence/absence.

See ``tkf/varanc-presence.tex`` (appendix L of the main paper) for the
math derivation. This module implements:

- ``tkf92_wfst_T``: closed-form 6x6 TKF92 conditional WFST transition
  matrix (eq:tkf92-wfst), with Ins0/Ins1 split.
- ``parse_binary_tree``: parse a newick string to a static binary-tree
  structure suitable for vmapped BP.
- ``make_q_conditionals``: map free per-(edge, column) logits to
  3x3 row-stochastic edge conditionals respecting the irreversibility
  constraints of section L.3.
- ``bp_pair_marginals``: vmapped Felsenstein-style up-down BP that
  returns pairwise (parent_state, child_state) marginals at every
  (edge, column).
- ``expected_branch_LL``: cumulant-trick computation of E_q[L_branch]
  per branch (eq:E-branch-LL + eq:W-prefix). Splits the I-column
  probability into an Ins0 (leading-insert, divisor kappa) and Ins1
  (post-immortal-link, divisor p) contribution per the corrected
  6x6 WFST.
- ``elbo``: total ELBO = sum branches E[LL] + entropy + log P(root).

State conventions:

- WFST states (6): ``S=0, M=1, I=I1=2, D=3, E=4, I0=5``. The classical
  ``S, M, I, D, E`` indices 0..4 are preserved for backward compatibility
  with the rest of the codebase; the additional ``I0`` state is appended
  at index 5 to keep diff churn minimal. ``WFST_I`` is an alias for
  ``WFST_I1`` (the post-immortal-link insert).
- Variational Z states (3): ``NYI=0, P=1, D=2`` (NotYetInserted,
  Present, Deleted).
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from ..core.bdi import tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa
from ..util.io import parse_newick


WFST_S, WFST_M, WFST_I, WFST_D, WFST_E, WFST_I0 = 0, 1, 2, 3, 4, 5
WFST_I1 = WFST_I  # alias: I and I1 are the same post-immortal-link insert
N_WFST = 6

NYI, PRESENT, DELETED = 0, 1, 2
N_Z = 3


# ---------------------------------------------------------------------------
# TKF92 conditional WFST transition matrix (eq:tkf92-wfst).
# ---------------------------------------------------------------------------

def tkf92_wfst_T(ins_rate, del_rate, t, ext):
    """Compute the TKF92 conditional WFST 6x6 transition matrix.

    Closed-form from ``tkf/tkf92-wfst-derivation.tex`` eq:tkf92-wfst.
    State order: S, M, I=I1, D, E, I0 (indices 0..5; I0 is at index 5).
    The Ins state is split into Ins0 (leading insert before any ancestor
    consumption; divides ancestor exits by kappa) and Ins1 (post-immortal-link
    insert; divides ancestor exits by p = ext + (1-ext)*kappa). At ext=0,
    p = kappa, and the two rows collapse, recovering the TKF91 5x5 WFST.

    Rows do not sum to 1 in the local single-step sense (the singlet
    factor varies by destination column), but the WFST is conditionally
    normalised: for each state and each non-epsilon input symbol, the
    geometric chain of insert self-loops followed by a single
    input-consuming edge has total weight 1.

    Args:
        ins_rate, del_rate: scalar lambda, mu.
        t: scalar branch length.
        ext: scalar fragment-extension probability r.

    Returns:
        T: (6, 6) array; ``T[s_M, s_N]`` is the WFST transition
        probability from state ``s_M`` to state ``s_N``.
    """
    a = tkf_alpha(del_rate, t)
    b = tkf_beta(ins_rate, del_rate, t)
    g = tkf_gamma(ins_rate, del_rate, t)
    k = tkf_kappa(ins_rate, del_rate)
    r = ext
    p = r + (1.0 - r) * k

    one_minus_b = 1.0 - b
    one_minus_g = 1.0 - g
    one_minus_r = 1.0 - r
    one_minus_a = 1.0 - a

    T = jnp.zeros((N_WFST, N_WFST))

    # S row: insert goes to Ins0; structural zero on Ins1.
    T = T.at[WFST_S, WFST_M].set(one_minus_b * a)
    T = T.at[WFST_S, WFST_I0].set(b)
    T = T.at[WFST_S, WFST_D].set(one_minus_b * one_minus_a)
    T = T.at[WFST_S, WFST_E].set(one_minus_b)

    # M row: insert goes to Ins1; structural zero on Ins0.
    T = T.at[WFST_M, WFST_M].set((r + one_minus_r * one_minus_b * k * a) / p)
    T = T.at[WFST_M, WFST_I1].set(one_minus_r * b)
    T = T.at[WFST_M, WFST_D].set(one_minus_r * one_minus_b * k * one_minus_a / p)
    T = T.at[WFST_M, WFST_E].set(one_minus_b)

    # Ins1 row: post-immortal-link inserts; ancestor exits divide by p.
    T = T.at[WFST_I1, WFST_M].set(one_minus_r * one_minus_b * k * a / p)
    T = T.at[WFST_I1, WFST_I1].set(r + one_minus_r * b)
    T = T.at[WFST_I1, WFST_D].set(one_minus_r * one_minus_b * k * one_minus_a / p)
    T = T.at[WFST_I1, WFST_E].set(one_minus_b)

    # D row: insert goes to Ins1; structural zero on Ins0.
    T = T.at[WFST_D, WFST_M].set(one_minus_r * one_minus_g * k * a / p)
    T = T.at[WFST_D, WFST_I1].set(one_minus_r * g)
    T = T.at[WFST_D, WFST_D].set((r + one_minus_r * one_minus_g * k * one_minus_a) / p)
    T = T.at[WFST_D, WFST_E].set(one_minus_g)

    # Ins0 row: leading inserts; ancestor exits divide by kappa, end by 1-kappa.
    # Algebra: (1-r)(1-b)*k*a / k = (1-r)(1-b)*a, etc.; (1-r)(1-b)*(1-k) / (1-k) = (1-r)(1-b).
    T = T.at[WFST_I0, WFST_M].set(one_minus_r * one_minus_b * a)
    T = T.at[WFST_I0, WFST_I0].set(r + one_minus_r * b)
    T = T.at[WFST_I0, WFST_D].set(one_minus_r * one_minus_b * one_minus_a)
    T = T.at[WFST_I0, WFST_E].set(one_minus_r * one_minus_b)

    # E row stays zero (no outgoing transitions).
    return T


def tkf92_wfst_log_T(ins_rate, del_rate, t, ext, eps=1e-300):
    """Log of ``tkf92_wfst_T`` with safe handling of structural zeros."""
    T = tkf92_wfst_T(ins_rate, del_rate, t, ext)
    return jnp.log(jnp.where(T > 0.0, T, eps))


def singlet_root_log_prior(root_dist, ins_rate, del_rate, ext, eps=1e-30):
    """Compute E_q[log p_singlet(root presence profile)] under TKF92.

    Implements equation (E-singlet-root) of the paper appendix:
    under per-column factorised q over root states (only N and P
    have positive mass; D is impossible at the root by irreversibility),

        E_q[log p_singlet] = log(p) * sum_n q(root_n = P)
                            + (1 - P_q(L=0)) * c1
                            + P_q(L=0) * log(1 - kappa)

    with p = ext + (1-ext)*kappa, c1 = log(kappa(1-ext)(1-kappa)/p),
    and P_q(L=0) = prod_n q(root_n = N).

    The L=0 special case must be retained for TKF92; for TKF91 (ext=0)
    the second and third terms collapse to the q-independent constant
    log(1-kappa).

    Args:
        root_dist: (L, 3) categorical q(root_n) per column. The D
            entry should be zero in our parameterisation.
        ins_rate, del_rate, ext: TKF92 parameters.

    Returns:
        scalar; the expected log singlet probability summed over
        columns.
    """
    kappa = ins_rate / del_rate
    p = ext + (1.0 - ext) * kappa
    log_p = jnp.log(p)
    log_1mk = jnp.log(1.0 - kappa)
    c1 = jnp.log(kappa * (1.0 - ext) * (1.0 - kappa) / p)

    sum_root_P = jnp.sum(root_dist[:, PRESENT])

    # P_q(L=0) = prod_n q(root_n != P) = prod_n q(root_n = N) here
    # (since root D mass is forced to 0). Compute in log-space to avoid
    # underflow when L is large.
    safe_NYI = jnp.maximum(root_dist[:, NYI], eps)
    log_P_L_zero = jnp.sum(jnp.log(safe_NYI))
    P_L_zero = jnp.exp(log_P_L_zero)

    return (log_p * sum_root_P
            + (1.0 - P_L_zero) * c1
            + P_L_zero * log_1mk)


# ---------------------------------------------------------------------------
# Tree topology.
# ---------------------------------------------------------------------------

class BinaryTree(NamedTuple):
    """Static binary-tree representation for vmapped BP.

    Convention: nodes 0..num_internal-1 are internal; nodes
    num_internal..num_nodes-1 are leaves. Root is the last internal
    node in postorder.
    """
    num_internal: int
    num_leaves: int
    num_nodes: int
    num_edges: int
    parent: np.ndarray         # (num_nodes,) int; -1 at root
    left_child: np.ndarray     # (num_internal,) int; child index
    right_child: np.ndarray    # (num_internal,) int; child index
    edge_parent: np.ndarray    # (num_edges,) int; parent node of edge e
    edge_child: np.ndarray     # (num_edges,) int; child node of edge e
    edge_length: np.ndarray    # (num_edges,) float; branch length
    postorder_internal: np.ndarray  # (num_internal,) postorder indices of internals
    preorder_internal: np.ndarray   # (num_internal,) preorder indices of internals
    leaf_names: list           # leaf names in node-index order
    root: int                  # root node index


def parse_binary_tree(newick: str) -> BinaryTree:
    """Parse a newick string into a BinaryTree.

    Polytomies are rejected (require strictly binary trees).
    """
    root_node = parse_newick(newick)

    # Walk the tree to collect all nodes and validate binary-ness.
    leaves = []
    internals = []
    parent_of = {}

    def visit(node, parent):
        parent_of[id(node)] = parent
        if not node.children:
            leaves.append(node)
        else:
            if len(node.children) != 2:
                raise ValueError(
                    f"varanc_presence requires strictly binary trees; "
                    f"node {node.name!r} has {len(node.children)} children")
            internals.append(node)
            for c in node.children:
                visit(c, node)

    visit(root_node, None)

    # Postorder traversal of internals.
    postorder = []
    visited_int = set()
    def post(node):
        if not node.children:
            return
        for c in node.children:
            post(c)
        if id(node) not in visited_int:
            visited_int.add(id(node))
            postorder.append(node)
    post(root_node)

    num_internal = len(internals)
    num_leaves = len(leaves)
    num_nodes = num_internal + num_leaves

    # Index internals by postorder position; leaves indexed in encounter
    # order after internals.
    node_idx = {}
    for i, node in enumerate(postorder):
        node_idx[id(node)] = i
    for i, node in enumerate(leaves):
        node_idx[id(node)] = num_internal + i

    parent = np.full(num_nodes, -1, dtype=np.int32)
    left_child = np.full(num_internal, -1, dtype=np.int32)
    right_child = np.full(num_internal, -1, dtype=np.int32)
    edge_parent_list = []
    edge_child_list = []
    edge_length_list = []

    for node in postorder:
        cidx_self = node_idx[id(node)]
        left, right = node.children
        left_child[cidx_self] = node_idx[id(left)]
        right_child[cidx_self] = node_idx[id(right)]
        for child in (left, right):
            cidx = node_idx[id(child)]
            parent[cidx] = cidx_self
            edge_parent_list.append(cidx_self)
            edge_child_list.append(cidx)
            edge_length_list.append(float(child.branch_length))
    for leaf in leaves:
        # leaf parent already set via internals' children loop; nothing
        # to do here.
        pass

    edge_parent = np.array(edge_parent_list, dtype=np.int32)
    edge_child = np.array(edge_child_list, dtype=np.int32)
    edge_length = np.array(edge_length_list, dtype=np.float64)
    num_edges = edge_parent.shape[0]

    postorder_internal = np.array(
        [node_idx[id(n)] for n in postorder], dtype=np.int32)
    # preorder = reverse postorder of internals (since each parent comes
    # after its children in postorder, so before them when reversed).
    preorder_internal = postorder_internal[::-1].copy()

    leaf_names = [(leaf.name or f"leaf{i}") for i, leaf in enumerate(leaves)]
    root_idx = node_idx[id(root_node)]

    return BinaryTree(
        num_internal=num_internal,
        num_leaves=num_leaves,
        num_nodes=num_nodes,
        num_edges=num_edges,
        parent=parent,
        left_child=left_child,
        right_child=right_child,
        edge_parent=edge_parent,
        edge_child=edge_child,
        edge_length=edge_length,
        postorder_internal=postorder_internal,
        preorder_internal=preorder_internal,
        leaf_names=leaf_names,
        root=root_idx,
    )


# ---------------------------------------------------------------------------
# Variational q parameterisation.
# ---------------------------------------------------------------------------

def make_q_conditionals(logits):
    """Build per-(edge, column) 3x3 row-stochastic conditionals.

    The 3x3 matrix has the irreversibility structure:

        [[a, 1-a, 0],
         [0, b, 1-b],
         [0, 0, 1]]

    where a = sigmoid(logits[..., 0]) is q(NYI | NYI parent),
    b = sigmoid(logits[..., 1]) is q(P | P parent), and the rest are
    pinned to 0 or 1.

    Args:
        logits: (..., 2) array of free variational logits.

    Returns:
        q_cond: (..., 3, 3) array. ``q_cond[..., parent, child]`` is
        the variational P(child state | parent state).
    """
    a = jax.nn.sigmoid(logits[..., 0])
    b = jax.nn.sigmoid(logits[..., 1])
    one_minus_a = 1.0 - a
    one_minus_b = 1.0 - b
    zeros = jnp.zeros_like(a)
    ones = jnp.ones_like(a)

    row_NYI = jnp.stack([a, one_minus_a, zeros], axis=-1)
    row_P = jnp.stack([zeros, b, one_minus_b], axis=-1)
    row_D = jnp.stack([zeros, zeros, ones], axis=-1)
    return jnp.stack([row_NYI, row_P, row_D], axis=-2)


def make_root_dist(root_logit):
    """Build per-column root distribution over {NYI, P}.

    The root cannot be in state D (D requires a prior P). We use a
    single logit per column for q(root = NYI) vs q(root = P).

    Args:
        root_logit: (..., ) array; q(root=NYI) = sigmoid(root_logit).

    Returns:
        root_dist: (..., 3) categorical, with mass on D pinned to 0.
    """
    p_NYI = jax.nn.sigmoid(root_logit)
    p_P = 1.0 - p_NYI
    p_D = jnp.zeros_like(p_NYI)
    return jnp.stack([p_NYI, p_P, p_D], axis=-1)


def leaf_clamp_to_beta(leaf_present):
    """Convert leaf indicator (0/1) to a 3-vector of valid-state masks.

    Present leaves: only Z=PRESENT is consistent. Absent leaves: Z=NYI
    and Z=DELETED both consistent (the leaf carries no mark of which).
    """
    present = jnp.asarray(leaf_present, dtype=jnp.float64)
    absent = 1.0 - present
    return jnp.stack([absent, present, absent], axis=-1)


# ---------------------------------------------------------------------------
# Belief propagation (per-column, then vmapped across columns).
# ---------------------------------------------------------------------------

def _bp_up_single_column(q_cond_col, beta_init, postorder_internal,
                         left_child, right_child):
    """Postorder up pass for one column.

    Args:
        q_cond_col: (num_edges, 3, 3) variational conditionals for this column.
        beta_init: (num_nodes, 3) initial beta (leaves clamped, internals zero).
        postorder_internal: (num_internal,) postorder indices.
        left_child, right_child: (num_internal,) child node indices.

    Returns:
        beta: (num_nodes, 3) up-pass beta values.
        log_Z: scalar log partition function for this column.
    """
    # Edge convention: edges are ordered such that edge e connects
    # internal node ``edge_parent[e]`` to ``edge_child[e]``. We need to
    # look up, for each (parent, child) pair, the edge index.
    # We package this in the calling convention: q_cond_col is already
    # gathered per-edge in a way the BP can consume.
    #
    # For binary trees with internal node v having children left=L, right=R,
    # we need q_cond on edges (v, L) and (v, R). We pre-index those by
    # left_edge[v] and right_edge[v] (built in caller).
    raise NotImplementedError("Use bp_pair_marginals; this helper is internal.")


def bp_pair_marginals(q_cond, root_dist, leaf_clamp, tree,
                      left_edge_of_internal, right_edge_of_internal):
    """Vectorised up-down BP returning pairwise (parent, child) marginals.

    Args:
        q_cond: (num_edges, L, 3, 3) variational conditionals.
        root_dist: (L, 3) per-column root distribution (D pinned to 0).
        leaf_clamp: (num_leaves, L, 3) leaf beta vectors.
        tree: BinaryTree.
        left_edge_of_internal: (num_internal,) edge index of left child.
        right_edge_of_internal: (num_internal,) edge index of right child.

    Returns:
        pair_marg: (num_edges, L, 3, 3); pair_marg[e, n, z_p, z_c] is
            q_n(parent=z_p, child=z_c) for edge e, column n.
        log_Z: (L,) log partition function per column.
    """
    L = q_cond.shape[1]
    num_internal = tree.num_internal
    num_nodes = tree.num_nodes
    num_edges = tree.num_edges

    # Convert tree's static arrays to jnp for tracer-safe indexing.
    left_edge = jnp.asarray(left_edge_of_internal)
    right_edge = jnp.asarray(right_edge_of_internal)
    left_child = jnp.asarray(tree.left_child)
    right_child = jnp.asarray(tree.right_child)
    edge_parent = jnp.asarray(tree.edge_parent)
    edge_child = jnp.asarray(tree.edge_child)
    postorder = jnp.asarray(tree.postorder_internal)
    preorder = jnp.asarray(tree.preorder_internal)

    # Initialise beta: leaves get the clamps; internals start at ones
    # (will be overwritten in postorder).
    beta = jnp.ones((num_nodes, L, N_Z))
    beta = beta.at[num_internal:].set(leaf_clamp)

    # Per-edge upward messages m_v(z_p) = sum_z q(z|z_p) beta_v(z).
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

    # Partition function per column: Z_n = sum_z root_dist(z) * beta[root, z]
    log_Z = jnp.log(jnp.einsum('nz,nz->n', root_dist, beta[tree.root]) + 1e-300)

    # Down pass: eta_v(z) = msg into v from outside its subtree.
    eta = jnp.zeros((num_nodes, L, N_Z))
    eta = eta.at[tree.root].set(root_dist)

    def down_step(carry, v):
        eta, = carry
        le = left_edge[v]
        re = right_edge[v]
        l_child = left_child[v]
        r_child = right_child[v]
        eta_v = eta[v]
        # outgoing for left child = eta_v * m_right (msg from right sibling).
        out_for_left = eta_v * m_edge[re]
        out_for_right = eta_v * m_edge[le]
        eta_l = jnp.einsum('nz,nzj->nj', out_for_left, q_cond[le])
        eta_r = jnp.einsum('nz,nzj->nj', out_for_right, q_cond[re])
        eta = eta.at[l_child].set(eta_l)
        eta = eta.at[r_child].set(eta_r)
        return (eta,), None

    (eta,), _ = jax.lax.scan(down_step, (eta,), preorder)

    # Pairwise marginals on each edge (parent p, child c):
    # P(z_p, z_c) propto outgoing(z_p) * q(z_c | z_p) * beta_c(z_c)
    # where outgoing(z_p) = eta_p(z_p) * (beta_p(z_p) / m_c(z_p))
    #                     = eta_p(z_p) * (msgs from p's other children)
    eta_p = eta[edge_parent]                 # (E, L, 3)
    beta_p = beta[edge_parent]               # (E, L, 3)
    beta_c = beta[edge_child]                # (E, L, 3)
    m_c = m_edge                             # (E, L, 3); msg c sent up

    safe_m = jnp.where(m_c > 0.0, m_c, 1.0)
    outgoing = eta_p * (beta_p / safe_m) * (m_c > 0.0)

    joint = (outgoing[..., None] * q_cond * beta_c[..., None, :])
    joint_sum = joint.sum(axis=(-1, -2), keepdims=True)
    pair_marg = joint / jnp.where(joint_sum > 0.0, joint_sum, 1.0)

    return pair_marg, log_Z


# ---------------------------------------------------------------------------
# Per-branch expected log-likelihood (cumulant trick, eq:W-prefix).
# ---------------------------------------------------------------------------

def expected_branch_LL(pair_marg_branch, log_T_branch, eps=1e-30):
    """Compute E_q[L_branch] for a single branch via the cumulant trick.

    Args:
        pair_marg_branch: (L, 3, 3) pairwise marginals q_n(z_p, z_c)
            for this branch over the L MSA columns.
        log_T_branch: (6, 6) log of TKF92 conditional WFST. State order
            S=0, M=1, I=I1=2, D=3, E=4, I0=5; see ``tkf92_wfst_T``.
        eps: floor for P(Ig) to avoid -inf cumulants.

    Returns:
        E[L_branch] scalar.
    """
    # Map pairwise (z_p, z_c) -> WFST state probabilities at each column.
    # P(M)   = q(P, P)
    # P(I)   = q(NYI, P)              [splits into I0 + I1; see below]
    # P(D)   = q(P, D)
    # P(Ig)  = q(NYI, NYI) + q(D, D)
    # P(NYI) = q(NYI, NYI) + q(NYI, P)  [parent has not yet appeared; for I0 prefix]
    P_M = pair_marg_branch[:, PRESENT, PRESENT]
    P_I = pair_marg_branch[:, NYI, PRESENT]
    P_D = pair_marg_branch[:, PRESENT, DELETED]
    P_Ig = pair_marg_branch[:, NYI, NYI] + pair_marg_branch[:, DELETED, DELETED]
    P_NYI = pair_marg_branch[:, NYI, NYI] + pair_marg_branch[:, NYI, PRESENT]
    L = pair_marg_branch.shape[0]

    # Split P_I per column into a leading-insert (I0) component and a
    # post-immortal-link (I1) component using the prefix product
    #     pi[n] = prod_{m<n} P_NYI[m] = P(parent never appeared before col n).
    # Under the column-factorised q, P(s_n = I0) = P_I[n] * pi[n] and
    # P(s_n = I1) = P_I[n] * (1 - pi[n]). The cumulant trick treats them
    # as separate WFST states (indices WFST_I0 and WFST_I1) carrying the
    # 6x6 log_T's per-row divisors (kappa for I0 ancestor exits, p for I1).
    log_P_NYI = jnp.log(jnp.maximum(P_NYI, eps))
    cum_log_NYI = jnp.cumsum(log_P_NYI)
    log_pi = jnp.concatenate([jnp.zeros(1), cum_log_NYI[:-1]])  # log_pi[n] = cum up to n-1
    pi = jnp.exp(log_pi)
    P_I0 = P_I * pi
    P_I1 = P_I * (1.0 - pi)

    # State-probability tensor: shape (L+2, 6) for columns 0..L+1.
    # Boundary: column 0 is S (deterministic), column L+1 is E (deterministic).
    P_state = jnp.zeros((L + 2, N_WFST))
    P_state = P_state.at[0, WFST_S].set(1.0)
    P_state = P_state.at[L + 1, WFST_E].set(1.0)
    P_state = P_state.at[1:L + 1, WFST_M].set(P_M)
    P_state = P_state.at[1:L + 1, WFST_I1].set(P_I1)
    P_state = P_state.at[1:L + 1, WFST_I0].set(P_I0)
    P_state = P_state.at[1:L + 1, WFST_D].set(P_D)

    # P(Ig) per column 0..L+1; boundaries set to 1 so log -> 0 and they
    # contribute nothing to the cumulant.
    P_Ig_full = jnp.ones(L + 2)
    P_Ig_full = P_Ig_full.at[1:L + 1].set(jnp.maximum(P_Ig, eps))

    # C[n] = sum_{k=1..n} log P(Ig at col k). cumsum over P_Ig_full is
    # exactly that since boundary log=0.
    log_P_Ig = jnp.log(P_Ig_full)
    C = jnp.cumsum(log_P_Ig)              # (L+2,)
    C_prev = jnp.concatenate([jnp.zeros(1), C[:-1]])  # C_prev[N] = C[N-1]

    # W[s, s'] = sum_{N=1..L+1} sum_{M<N}
    #     P_state[M, s] * P_state[N, s'] * exp(C[N-1] - C[M])
    #
    # Stable form: work with log_v[M, s] = log P_state[M, s] - C[M].
    # cum_log_v[N, s] = logsumexp_{M <= N} log_v[M, s].
    # inner[N, s] = exp(cum_log_v_prev[N, s] + C[N-1])
    #             = sum_{M < N} P_state[M, s] * exp(C[N-1] - C[M])
    # which is bounded in [0, N] regardless of the cumulants' magnitude.
    log_P = jnp.log(jnp.maximum(P_state, eps))                 # (L+2, 5)
    log_v = log_P - C[:, None]                                 # (L+2, 5)
    # Mask entries where P_state = 0: log_v -> -inf so they don't enter
    # the running logaddexp.
    log_v = jnp.where(P_state > 0, log_v, -jnp.inf)

    def lae_step(carry, x):
        new = jnp.logaddexp(carry, x)
        return new, new

    init = jnp.full((N_WFST,), -jnp.inf)
    _, cum_log_v = jax.lax.scan(lae_step, init, log_v)         # (L+2, 5)
    cum_log_v_prev = jnp.concatenate(
        [jnp.full((1, N_WFST), -jnp.inf), cum_log_v[:-1]], axis=0)

    inner = jnp.exp(cum_log_v_prev + C_prev[:, None])          # (L+2, 5)

    # W[s, s'] = sum_{N >= 1} P_state[N, s'] * inner[N, s]
    W = jnp.einsum('ns,nt->st', inner[1:], P_state[1:])

    return jnp.sum(log_T_branch * W)


def expected_branch_LL_naive(pair_marg_branch, log_T_branch, eps=1e-30):
    """O(L^2) reference implementation of E_q[L_branch] for testing."""
    P_M = pair_marg_branch[:, PRESENT, PRESENT]
    P_I = pair_marg_branch[:, NYI, PRESENT]
    P_D = pair_marg_branch[:, PRESENT, DELETED]
    P_Ig = pair_marg_branch[:, NYI, NYI] + pair_marg_branch[:, DELETED, DELETED]
    P_NYI = pair_marg_branch[:, NYI, NYI] + pair_marg_branch[:, NYI, PRESENT]
    L = pair_marg_branch.shape[0]

    log_P_NYI = jnp.log(jnp.maximum(P_NYI, eps))
    cum_log_NYI = jnp.cumsum(log_P_NYI)
    log_pi = jnp.concatenate([jnp.zeros(1), cum_log_NYI[:-1]])
    pi = jnp.exp(log_pi)
    P_I0 = P_I * pi
    P_I1 = P_I * (1.0 - pi)

    P_state = jnp.zeros((L + 2, N_WFST))
    P_state = P_state.at[0, WFST_S].set(1.0)
    P_state = P_state.at[L + 1, WFST_E].set(1.0)
    P_state = P_state.at[1:L + 1, WFST_M].set(P_M)
    P_state = P_state.at[1:L + 1, WFST_I1].set(P_I1)
    P_state = P_state.at[1:L + 1, WFST_I0].set(P_I0)
    P_state = P_state.at[1:L + 1, WFST_D].set(P_D)
    P_Ig_full = jnp.ones(L + 2)
    P_Ig_full = P_Ig_full.at[1:L + 1].set(jnp.maximum(P_Ig, eps))

    total = 0.0
    for N in range(1, L + 2):
        for M in range(0, N):
            # Product of intervening Ig probabilities (interior cols only).
            if M + 1 <= N - 1:
                ig_prod = jnp.prod(P_Ig_full[M + 1:N])
            else:
                ig_prod = 1.0
            # Sum over (s, s').
            term = jnp.einsum(
                's,t,st->', P_state[M], P_state[N], log_T_branch) * ig_prod
            total = total + term
    return total


# ---------------------------------------------------------------------------
# Entropy of a tree-structured q (per column, sum over edges).
# ---------------------------------------------------------------------------

def entropy_per_column(pair_marg, root_dist, beta_root,
                       node_marg_internal, q_cond, tree):
    """Compute -E_q[log q_joint(internals, leaves)] per column.

    This is the cross-entropy piece of the leaf-conditioned q entropy:
    the full entropy is H[q(internals|MSA)] = -E_q[log q_joint] + log Z_q
    (the caller in ``elbo`` adds the log Z_q correction).

    Per equation (cross-entropy-decomp) of the appendix, the result is

        -E[log q_root(z_root)]
            - sum_edges sum_{z_p, z_c} pair_marg(z_p, z_c) * log q^{p->c}(z_c | z_p)

    using BP-derived joint pair marginals.
    Note: this is NOT the parent-marginal-weighted conditional entropy
    sum_z_p q(z_p) * H[q^{p->c}(.|z_p)]; the two coincide only when the
    leaves impose no evidence on the edge, which is generically false.

    Args:
        pair_marg: (num_edges, L, 3, 3) BP pairwise joint marginals.
        root_dist: (L, 3) root distribution under q.
        beta_root: unused (kept for API symmetry).
        node_marg_internal: unused.
        q_cond: (num_edges, L, 3, 3) prior edge conditionals.
        tree: BinaryTree.

    Returns:
        H: (L,) per-column cross-entropy (>= 0).
    """
    # We compute -E_q(internals|leaves)[log q_joint(z_root, edge conditionals)],
    # which is one of the two pieces of H[q(internals|leaves)] (the other
    # being log Z_q, added by the caller in ``elbo``).
    #
    # Root term: -E_q(internals|leaves)[log q_root(z_root)]
    #          = -sum_z q(z_root | leaves) * log q_root(z_root)
    # where q(z_root | leaves) is the BP-post-conditioned root marginal,
    # NOT the prior root_dist. We extract q(z_root | leaves) from
    # pair_marg by summing over the child state on any edge whose parent
    # is the root: q(z_root | leaves) = sum_z_c pair_marg(root_edge, z_root, z_c).
    # We use the root's left child edge.
    edge_parent_arr = jnp.asarray(tree.edge_parent)
    root_edge = jnp.argmax((edge_parent_arr == tree.root).astype(jnp.int32))
    root_marg_post = pair_marg[root_edge].sum(axis=-1)  # (L, 3)
    safe_root_dist = jnp.where(root_dist > 0.0, root_dist, 1.0)
    log_root_dist = jnp.where(root_dist > 0.0, jnp.log(safe_root_dist), 0.0)
    H_root = -(root_marg_post * log_root_dist).sum(axis=-1)  # (L,)

    # Edge term: -E[log q^{p->c}(z_c | z_p)] under the leaf-conditioned
    # joint marginal pair_marg(z_p, z_c).
    # = -sum_{z_p, z_c} pair_marg(z_p, z_c) * log q^{p->c}(z_c | z_p)
    # NB this is a CROSS-entropy of the BP-conditional vs q^{p->c}, weighted
    # by the parent marginal. Critically NOT the entropy of q^{p->c}.
    safe_q = jnp.where(q_cond > 0.0, q_cond, 1.0)
    log_q = jnp.where(q_cond > 0.0, jnp.log(safe_q), 0.0)
    edge_term = -(pair_marg * log_q).sum(axis=(-1, -2))  # (E, L)
    H_edges = edge_term.sum(axis=0)                      # (L,)

    return H_root + H_edges


# ---------------------------------------------------------------------------
# Driver: ELBO.
# ---------------------------------------------------------------------------

def elbo(logits, root_logits, leaf_present, tree, ins_rate, del_rate, ext,
         left_edge_of_internal, right_edge_of_internal,
         root_log_prior=None):
    """Total variational lower bound (ELBO) on the indel log-likelihood.

    Args:
        logits: (num_edges, L, 2) variational logits.
        root_logits: (L,) root-distribution logits.
        leaf_present: (num_leaves, L) {0, 1} indicators.
        tree: BinaryTree.
        ins_rate, del_rate, ext: TKF92 parameters (scalars).
        left_edge_of_internal, right_edge_of_internal: (num_internal,)
            edge indices for left/right children, precomputed from tree.
        root_log_prior: (3,) optional log prior on root state. Defaults
            to log uniform on {NYI, P} (D mass forced to 0 anyway).

    Returns:
        elbo_total: scalar.
        diagnostics: dict with per-column ELBO, log_Z etc.
    """
    L = logits.shape[1]

    q_cond = make_q_conditionals(logits)
    root_dist = make_root_dist(root_logits)
    leaf_clamp = leaf_clamp_to_beta(leaf_present)

    pair_marg, log_Z = bp_pair_marginals(
        q_cond, root_dist, leaf_clamp, tree,
        left_edge_of_internal, right_edge_of_internal)

    # Build per-branch log_T using each edge's branch length.
    edge_lengths = jnp.asarray(tree.edge_length)
    log_T_per_edge = jax.vmap(
        lambda t: tkf92_wfst_log_T(ins_rate, del_rate, t, ext))(edge_lengths)

    # Sum E[L_branch] over edges.
    branch_LLs = jax.vmap(expected_branch_LL)(pair_marg, log_T_per_edge)
    sum_branch_LL = jnp.sum(branch_LLs)

    # Node marginals for entropy.
    # Per-internal-node marginal: q_n(z_v) = beta_v(z_v) * eta_v(z_v) / Z.
    # We can recover it from the pair marginal by summing over child state
    # for any edge whose PARENT is v. For simplicity (and because the
    # entropy uses parent marginals per edge), we pass pair_marg directly.
    # Pass dummy `node_marg_internal` since the entropy fn computes it.
    H_per_col = entropy_per_column(
        pair_marg, root_dist, beta_root=None,
        node_marg_internal=None, q_cond=q_cond, tree=tree)
    H_total = jnp.sum(H_per_col)

    # Root prior contribution: TKF92 stationary singlet HMM (eq:E-singlet-root).
    # Override possible by passing root_log_prior explicitly (kept for tests).
    if root_log_prior is None:
        log_prior_root = singlet_root_log_prior(
            root_dist, ins_rate, del_rate, ext)
    else:
        log_prior_root = jnp.sum(root_dist * root_log_prior)

    # Entropy of q(internals | leaves):
    # H[q(internals | leaves)] = -E_q[log q_joint(internals, leaves)] + log Z_q
    # The first term is computed by entropy_per_column using leaf-conditioned
    # pair marginals (so it accounts for leaf-edge factors with mass only on
    # the clamped leaf state). The log Z_q correction converts from
    # joint-q-entropy to leaf-conditioned-q-entropy.
    sum_log_Z = jnp.sum(log_Z)
    elbo_total = sum_branch_LL + log_prior_root + H_total + sum_log_Z

    return elbo_total, {
        'sum_branch_LL': sum_branch_LL,
        'log_prior_root': log_prior_root,
        'entropy_joint': H_total,
        'log_Z': sum_log_Z,
        'log_Z_per_col': log_Z,
        'pair_marg': pair_marg,
    }


def edge_lookup(tree: BinaryTree):
    """Compute (left_edge_of_internal, right_edge_of_internal).

    For each internal node, find the edge index that connects it to its
    left and right child.
    """
    le = np.full(tree.num_internal, -1, dtype=np.int32)
    re = np.full(tree.num_internal, -1, dtype=np.int32)
    for e in range(tree.num_edges):
        p = tree.edge_parent[e]
        c = tree.edge_child[e]
        if c == tree.left_child[p]:
            le[p] = e
        elif c == tree.right_child[p]:
            re[p] = e
    return le, re
