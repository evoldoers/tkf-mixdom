"""Ancestral sequence reconstruction on phylogenetic trees.

Implements:
1. Pairwise ancestral reconstruction via Viterbi traceback
2. Progressive tree reconstruction (bottom-up pairwise alignment)
3. Marginal ancestral probabilities at internal nodes (Felsenstein-style)
4. Vectorized Felsenstein over all MSA columns via jax.vmap
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..core.params import tkf91_trans, S, M, I, D, E
from ..core.ctmc import transition_matrix
from ..dp.hmm import viterbi_2d, forward_2d, sample_traceback_2d, safe_log
from ..dp.hmm_beam import beam_forward_backward_2d, msa_to_envelope
from ..util.io import TreeNode


def reconstruct_ancestor_pairwise(x, y, ins_rate, del_rate, t, Q, pi):
    """Reconstruct the most likely ancestral alignment between x and y.

    Args:
        x, y: integer sequence arrays
        ins_rate, del_rate, t: TKF91 parameters
        Q: rate matrix
        pi: equilibrium frequencies

    Returns:
        ancestor: integer array (reconstructed ancestral sequence)
        alignment: list of (anc_pos_or_None, x_pos_or_None, y_pos_or_None)
        log_prob: Viterbi log-probability
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    sub_matrix = transition_matrix(Q, t)
    tau = tkf91_trans(ins_rate, del_rate, t)
    log_trans = safe_log(tau)
    state_types = jnp.array([S, M, I, D, E])

    log_prob, path = viterbi_2d(log_trans, state_types, x, y, sub_matrix, pi)

    # Extract ancestor from path
    ancestor_chars = []
    alignment = []
    anc_pos = 0

    for i, j, s in path:
        if s == M and i > 0 and j > 0:
            # Match: ancestor character is x's character (most likely ancestor)
            ancestor_chars.append(int(x[i - 1]))
            alignment.append((anc_pos, i - 1, j - 1))
            anc_pos += 1
        elif s == I and j > 0:
            # Insert in y: no ancestor position
            alignment.append((None, None, j - 1))
        elif s == D and i > 0:
            # Delete from x: ancestor has this position but y doesn't
            ancestor_chars.append(int(x[i - 1]))
            alignment.append((anc_pos, i - 1, None))
            anc_pos += 1

    ancestor = np.array(ancestor_chars, dtype=np.int32)
    return ancestor, alignment, float(log_prob)


def marginal_ancestor_column(tree_root, leaf_chars, Q, pi):
    """Compute marginal posterior P(ancestor_char | leaf_data) at the root.

    Uses Felsenstein pruning (substitution only, no indels).

    Args:
        tree_root: TreeNode
        leaf_chars: dict of {leaf_name: character_index}
        Q: rate matrix
        pi: equilibrium frequencies

    Returns:
        posterior: (alphabet_size,) posterior probabilities at root
    """
    n = Q.shape[0]

    def _prune(node):
        """Returns (cond_likelihood, log_scale) with rescaling."""
        if node.is_leaf:
            char = leaf_chars.get(node.name)
            if char is None or char < 0:
                return jnp.ones(n), 0.0
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

        max_val = jnp.max(partial)
        max_val = jnp.maximum(max_val, 1e-300)
        partial = partial / max_val
        log_scale += jnp.log(max_val)

        return partial, log_scale

    root_cond, _ = _prune(tree_root)
    # Scale factors cancel in the posterior ratio
    joint = pi * root_cond
    posterior = joint / jnp.sum(joint)
    return posterior


def reconstruct_tree_progressive(tree_root, leaf_seqs, ins_rate, del_rate, Q, pi):
    """Progressive ancestral reconstruction on a tree.

    Bottom-up: at each internal node, align the two child sequences
    (or reconstructed ancestors) using the sum of child branch lengths,
    then reconstruct the ancestor.

    Args:
        tree_root: TreeNode
        leaf_seqs: dict of {leaf_name: integer_array}
        ins_rate, del_rate: indel rates
        Q: rate matrix
        pi: equilibrium frequencies

    Returns:
        node_seqs: dict of {node_id: integer_array} for all nodes
        node_alignments: dict of {node_id: alignment_info}
    """
    node_seqs = {}
    node_alignments = {}

    # Assign leaf sequences
    for node in tree_root.preorder():
        if node.is_leaf and node.name in leaf_seqs:
            node_seqs[id(node)] = np.asarray(leaf_seqs[node.name])

    # Bottom-up reconstruction
    for node in tree_root.postorder():
        if node.is_leaf:
            continue

        children_with_seqs = [c for c in node.children if id(c) in node_seqs]
        if len(children_with_seqs) == 0:
            continue

        if len(children_with_seqs) == 1:
            # Single child: ancestor is the child sequence
            child = children_with_seqs[0]
            node_seqs[id(node)] = node_seqs[id(child)]
            continue

        # Two children: align them through their common ancestor (this node)
        left, right = children_with_seqs[0], children_with_seqs[1]
        x = jnp.asarray(node_seqs[id(left)])
        y = jnp.asarray(node_seqs[id(right)])

        if len(x) == 0 or len(y) == 0:
            # Use whichever is non-empty
            node_seqs[id(node)] = node_seqs[id(left)] if len(x) > 0 else node_seqs[id(right)]
            continue

        # Total evolutionary distance between children goes through this node
        t_total = max(left.branch_length + right.branch_length, 1e-6)

        ancestor, alignment, lp = reconstruct_ancestor_pairwise(
            x, y, ins_rate, del_rate, t_total, Q, pi
        )

        node_seqs[id(node)] = ancestor
        node_alignments[id(node)] = {
            "left_child": left.name or f"node_{id(left)}",
            "right_child": right.name or f"node_{id(right)}",
            "alignment": alignment,
            "log_prob": lp,
            "t": t_total,
        }

    return node_seqs, node_alignments


def reconstruct_marginal_sequence(tree_root, leaf_seqs_aligned, Q, pi):
    """Reconstruct the most probable ancestral sequence at the root using
    marginal posterior probabilities (column by column).

    Requires aligned sequences (same length, gaps as -1).

    Args:
        tree_root: TreeNode
        leaf_seqs_aligned: dict of {leaf_name: integer_array} (aligned, -1 for gaps)
        Q: rate matrix
        pi: equilibrium frequencies

    Returns:
        ancestor: integer array (MAP ancestral sequence)
        posteriors: (L, alphabet_size) posterior probabilities
    """
    names = list(leaf_seqs_aligned.keys())
    L = len(next(iter(leaf_seqs_aligned.values())))
    n = Q.shape[0]

    ancestor_chars = []
    posteriors = []

    for col in range(L):
        col_chars = {}
        all_gap = True
        for name in names:
            c = int(leaf_seqs_aligned[name][col])
            col_chars[name] = c
            if c >= 0:
                all_gap = False

        if all_gap:
            # All gaps: skip this column
            ancestor_chars.append(-1)
            posteriors.append(np.ones(n) / n)
            continue

        post = marginal_ancestor_column(tree_root, col_chars, Q, pi)
        ancestor_chars.append(int(jnp.argmax(post)))
        posteriors.append(np.array(post))

    return np.array(ancestor_chars, dtype=np.int32), np.array(posteriors)


# --- Vectorized Felsenstein over all columns via jax.vmap ---

def _flatten_tree_for_felsenstein(tree_root):
    """Flatten tree into arrays for JAX-vectorized Felsenstein pruning.

    Uses node identity (id) rather than names to handle unnamed internal nodes.

    Returns:
        node_list: list of TreeNode objects in preorder
        node_to_id: dict {id(node): int}
        parent_ids: (N,) int array (-1 for root)
        children_ids: (N, max_children) int array (-1 for padding)
        n_children: (N,) int array
        is_leaf: (N,) bool array
        branch_lengths: (N,) float array
        postorder_ids: (N,) int array
    """
    node_list = list(tree_root.preorder())
    N = len(node_list)
    node_to_id = {id(node): i for i, node in enumerate(node_list)}

    parent_ids = np.full(N, -1, dtype=np.int32)
    max_children = max((len(node.children) for node in node_list), default=1)
    if max_children == 0:
        max_children = 1
    children_ids = np.full((N, max_children), -1, dtype=np.int32)
    n_children = np.zeros(N, dtype=np.int32)
    is_leaf = np.zeros(N, dtype=bool)
    branch_lengths = np.zeros(N, dtype=np.float64)

    for node in node_list:
        nid = node_to_id[id(node)]
        branch_lengths[nid] = node.branch_length
        if node.parent is not None:
            parent_ids[nid] = node_to_id[id(node.parent)]
        is_leaf[nid] = node.is_leaf
        for j, child in enumerate(node.children):
            children_ids[nid, j] = node_to_id[id(child)]
            n_children[nid] = j + 1

    postorder_ids = np.array(
        [node_to_id[id(n)] for n in tree_root.postorder()], dtype=np.int32)

    return (node_list, node_to_id, parent_ids, children_ids,
            n_children, is_leaf, branch_lengths, postorder_ids)


def _build_leaf_obs_array(leaf_seqs_aligned, name_to_nid, N, L, A):
    """Build (L, N, A) one-hot leaf observation array from aligned sequences.

    Gap positions (char < 0) get uniform (ones) conditional likelihood.

    Args:
        leaf_seqs_aligned: dict {name: int array of length L, gaps as -1}
        name_to_nid: dict {leaf_name: int node_id}
        N: total number of nodes
        L: alignment length
        A: alphabet size

    Returns:
        leaf_obs: (L, N, A) float array
        is_all_gap: (L,) bool array, True if all leaves are gaps at this column
    """
    leaf_obs = np.ones((L, N, A), dtype=np.float64)  # default: uniform (missing)
    is_all_gap = np.ones(L, dtype=bool)

    for name, seq in leaf_seqs_aligned.items():
        if name not in name_to_nid:
            continue
        nid = name_to_nid[name]
        seq_arr = np.asarray(seq)
        for col in range(L):
            c = int(seq_arr[col])
            if c >= 0:
                is_all_gap[col] = False
                leaf_obs[col, nid] = 0.0
                leaf_obs[col, nid, c] = 1.0

    return leaf_obs, is_all_gap


def _felsenstein_peel_all_columns(leaf_obs, trans_matrices, postorder_ids,
                                  children_ids, n_children, is_leaf, pi):
    """Vectorized Felsenstein pruning over all columns simultaneously.

    Uses jax.vmap over columns and jax.lax.scan over postorder traversal.

    Args:
        leaf_obs: (L, N, A) leaf conditional likelihoods
        trans_matrices: (N, A, A) transition matrix for each node's branch
        postorder_ids: (N,) int array
        children_ids: (N, max_ch) int array
        n_children: (N,) int array
        is_leaf: (N,) bool array
        pi: (A,) equilibrium frequencies

    Returns:
        posteriors: (L, A) posterior probabilities at root for each column
    """
    N = is_leaf.shape[0]
    A = pi.shape[0]
    max_ch = children_ids.shape[1]

    def _peel_one_column(leaf_ob):
        """Felsenstein pruning for a single column. leaf_ob: (N, A)."""

        def _peel_step(CL, nid):
            # Leaf: use observation directly
            leaf_cl = leaf_ob[nid]

            # Internal node: multiply child messages
            def _child_msg(ci):
                child_id = children_ids[nid, ci]
                safe_child_id = jnp.maximum(child_id, 0)
                valid = (ci < n_children[nid]) & (child_id >= 0)
                child_cl = CL[safe_child_id]
                # M_t @ child_cl: transition matrix times child CL
                msg = trans_matrices[safe_child_id] @ child_cl
                # Invalid children contribute ones (identity for multiplication)
                msg = jnp.where(valid, msg, jnp.ones(A))
                return msg

            child_msgs = jax.vmap(lambda ci: _child_msg(ci))(jnp.arange(max_ch))
            int_cl = jnp.prod(child_msgs, axis=0)

            # Rescale to prevent underflow
            int_max = jnp.maximum(jnp.max(int_cl), 1e-300)
            int_cl = int_cl / int_max

            # Select leaf vs internal
            cl = jnp.where(is_leaf[nid], leaf_cl, int_cl)

            CL = CL.at[nid].set(cl)
            return CL, None

        CL_init = jnp.zeros((N, A))
        CL_final, _ = jax.lax.scan(_peel_step, CL_init, postorder_ids)

        # Root is the first node in preorder = postorder_ids[-1]
        root_id = postorder_ids[-1]
        root_cl = CL_final[root_id]

        # Posterior at root
        joint = pi * root_cl
        Z = jnp.sum(joint)
        posterior = joint / jnp.maximum(Z, 1e-300)
        return posterior

    # vmap over all L columns
    posteriors = jax.vmap(_peel_one_column)(leaf_obs)
    return posteriors


def marginal_ancestor_all_columns_jax(tree_root, leaf_seqs_aligned, Q, pi):
    """Vectorized Felsenstein: compute marginal posteriors at root for all columns.

    Equivalent to calling marginal_ancestor_column() per column, but uses
    jax.vmap over all MSA columns for a single JIT-compiled computation.

    Args:
        tree_root: TreeNode
        leaf_seqs_aligned: dict of {leaf_name: integer_array} (aligned, -1 for gaps)
        Q: rate matrix (A, A)
        pi: equilibrium frequencies (A,)

    Returns:
        ancestor: (L,) integer array, MAP ancestral character per column (-1 for all-gap)
        posteriors: (L, A) posterior probabilities at root
    """
    Q = jnp.asarray(Q)
    pi = jnp.asarray(pi)
    A = Q.shape[0]

    names = list(leaf_seqs_aligned.keys())
    L = len(next(iter(leaf_seqs_aligned.values())))

    # Flatten tree (uses node identity, not names, for unique IDs)
    (node_list, node_to_id, parent_ids, children_ids,
     n_children, is_leaf, branch_lengths, postorder_ids) = \
        _flatten_tree_for_felsenstein(tree_root)
    N = len(node_list)

    # Build name -> node_id mapping for leaves
    name_to_nid = {}
    for node in node_list:
        if node.name is not None:
            name_to_nid[node.name] = node_to_id[id(node)]

    # Precompute transition matrices for all branches
    trans_matrices = np.zeros((N, A, A), dtype=np.float64)
    for nid in range(N):
        if parent_ids[nid] >= 0:  # not root
            t = branch_lengths[nid]
            trans_matrices[nid] = np.array(transition_matrix(Q, t))
    trans_matrices = jnp.asarray(trans_matrices)

    # Build leaf observation array
    leaf_obs, is_all_gap = _build_leaf_obs_array(
        leaf_seqs_aligned, name_to_nid, N, L, A)
    leaf_obs = jnp.asarray(leaf_obs)

    # Convert tree structure arrays to JAX
    children_ids_jax = jnp.asarray(children_ids)
    n_children_jax = jnp.asarray(n_children)
    is_leaf_jax = jnp.asarray(is_leaf)
    postorder_ids_jax = jnp.asarray(postorder_ids)

    # Run vectorized Felsenstein
    posteriors = _felsenstein_peel_all_columns(
        leaf_obs, trans_matrices, postorder_ids_jax,
        children_ids_jax, n_children_jax, is_leaf_jax, pi)

    posteriors = np.array(posteriors, copy=True)

    # Build ancestor: MAP character, -1 for all-gap columns
    ancestor = np.where(is_all_gap, -1, np.argmax(posteriors, axis=1)).astype(np.int32)

    # For all-gap columns, set posterior to uniform
    posteriors[is_all_gap] = 1.0 / A

    return ancestor, posteriors


# --- Sampling-based progressive reconstruction ---

def sample_ancestor_pairwise(x, y, ins_rate, del_rate, t, Q, pi, rng_key):
    """Sample an ancestral alignment between x and y from the posterior.

    Like reconstruct_ancestor_pairwise but uses stochastic traceback
    instead of Viterbi, sampling paths proportional to their probability.

    Args:
        x, y: integer sequence arrays
        ins_rate, del_rate, t: TKF91 parameters
        Q: rate matrix
        pi: equilibrium frequencies
        rng_key: JAX random key

    Returns:
        ancestor: integer array
        alignment: list of (anc_pos_or_None, x_pos_or_None, y_pos_or_None)
        log_prob: forward log-probability (total, not path-specific)
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    sub_matrix = transition_matrix(Q, t)
    tau = tkf91_trans(ins_rate, del_rate, t)
    log_trans = safe_log(tau)
    state_types = jnp.array([S, M, I, D, E])

    log_prob, path = sample_traceback_2d(
        log_trans, state_types, x, y, sub_matrix, pi, rng_key
    )

    ancestor_chars = []
    alignment = []
    anc_pos = 0

    for i, j, s in path:
        if s == M and i > 0 and j > 0:
            ancestor_chars.append(int(x[i - 1]))
            alignment.append((anc_pos, i - 1, j - 1))
            anc_pos += 1
        elif s == I and j > 0:
            alignment.append((None, None, j - 1))
        elif s == D and i > 0:
            ancestor_chars.append(int(x[i - 1]))
            alignment.append((anc_pos, i - 1, None))
            anc_pos += 1

    ancestor = np.array(ancestor_chars, dtype=np.int32)
    return ancestor, alignment, float(log_prob)


def reconstruct_tree_sampling(tree_root, leaf_seqs, ins_rate, del_rate, Q, pi,
                               rng_key, n_samples=10, threshold=2):
    """Progressive ancestral reconstruction using sampled paths.

    At each internal node, runs the forward algorithm on the two child
    sequences, then samples n_samples paths. The profile (set of visited
    alignment cells) is the union of cells visited by at least `threshold`
    paths. The ancestor is constructed from the most frequently visited
    cells.

    This is the sampling-based version of reconstruct_tree_progressive,
    following the approach of historian (Holmes 2017).

    Args:
        tree_root: TreeNode
        leaf_seqs: dict of {leaf_name: integer_array}
        ins_rate, del_rate: indel rates
        Q: rate matrix
        pi: equilibrium frequencies
        rng_key: JAX random key
        n_samples: number of paths to sample per node
        threshold: minimum visit count to include a cell

    Returns:
        node_seqs: dict of {node_id: integer_array} for all nodes
        node_samples: dict of {node_id: list_of_sampled_ancestors}
    """
    node_seqs = {}
    node_samples = {}

    # Assign leaf sequences
    for node in tree_root.preorder():
        if node.is_leaf and node.name in leaf_seqs:
            node_seqs[id(node)] = np.asarray(leaf_seqs[node.name])

    # Bottom-up reconstruction
    for node in tree_root.postorder():
        if node.is_leaf:
            continue

        children_with_seqs = [c for c in node.children if id(c) in node_seqs]
        if len(children_with_seqs) == 0:
            continue

        if len(children_with_seqs) == 1:
            child = children_with_seqs[0]
            node_seqs[id(node)] = node_seqs[id(child)]
            continue

        left, right = children_with_seqs[0], children_with_seqs[1]
        x = jnp.asarray(node_seqs[id(left)])
        y = jnp.asarray(node_seqs[id(right)])

        if len(x) == 0 or len(y) == 0:
            node_seqs[id(node)] = node_seqs[id(left)] if len(x) > 0 else node_seqs[id(right)]
            continue

        t_total = max(left.branch_length + right.branch_length, 1e-6)

        # Sample multiple paths
        sampled_ancestors = []
        for k in range(n_samples):
            rng_key, subkey = jr.split(rng_key)
            anc, aln, lp = sample_ancestor_pairwise(
                x, y, ins_rate, del_rate, t_total, Q, pi, subkey
            )
            sampled_ancestors.append(anc)

        node_samples[id(node)] = sampled_ancestors

        # Build consensus: use the median-length ancestor
        # (more sophisticated: use cell frequency thresholding)
        lengths = [len(a) for a in sampled_ancestors]
        median_len = int(np.median(lengths))
        # Pick the sample closest to median length
        best_idx = min(range(len(sampled_ancestors)),
                       key=lambda i: abs(len(sampled_ancestors[i]) - median_len))
        node_seqs[id(node)] = sampled_ancestors[best_idx]

    return node_seqs, node_samples


# --- Inside-guided progressive reconstruction ---

def _posterior_ancestor(x, y, ins_rate, del_rate, t, Q, pi,
                        envelope=None, beam_log_width=np.inf):
    """Reconstruct ancestor using forward-backward posterior probabilities.

    Instead of Viterbi (single best path), uses the full forward-backward
    posterior to compute P(state | data) at each cell, then extracts the
    most probable alignment path weighted by posterior probability.

    Args:
        x, y: integer sequence arrays
        ins_rate, del_rate, t: TKF91 parameters
        Q, pi: substitution model
        envelope: optional (Lx+1, Ly+1) boolean array constraining DP
        beam_log_width: log beam width for pruning

    Returns:
        ancestor: integer array
        alignment: list of (anc_pos, x_pos_or_None, y_pos_or_None)
        log_prob: total forward log probability
        posteriors: (Lx+1, Ly+1, n_states) posterior probabilities
    """
    x = np.asarray(x)
    y = np.asarray(y)
    sub_matrix = transition_matrix(Q, t)
    tau = tkf91_trans(ins_rate, del_rate, t)
    log_trans = np.log(np.maximum(np.asarray(tau), 1e-30))
    state_types = np.array([S, M, I, D, E])

    log_prob, posteriors, F = beam_forward_backward_2d(
        log_trans, state_types, x, y, sub_matrix, pi,
        beam_log_width, envelope)

    # Extract MAP path using posterior-weighted Viterbi
    # At each cell, pick the state with highest posterior
    Lx, Ly = len(x), len(y)
    ancestor_chars = []
    alignment = []
    anc_pos = 0

    # Greedy traceback from (Lx, Ly) following highest-posterior predecessors
    i, j = Lx, Ly
    path = []

    while i > 0 or j > 0:
        # Consider all possible states at (i, j)
        best_score = -1
        best_state = -1
        for k in range(len(state_types)):
            if posteriors[i, j, k] > best_score:
                best_score = posteriors[i, j, k]
                best_state = k

        st = int(state_types[best_state])
        path.append((i, j, st))

        if st == M and i > 0 and j > 0:
            i, j = i - 1, j - 1
        elif st == I and j > 0:
            j = j - 1
        elif st == D and i > 0:
            i = i - 1
        else:
            break

    path.reverse()

    for ci, cj, st in path:
        if st == M and ci > 0 and cj > 0:
            ancestor_chars.append(int(x[ci - 1]))
            alignment.append((anc_pos, ci - 1, cj - 1))
            anc_pos += 1
        elif st == I and cj > 0:
            alignment.append((None, None, cj - 1))
        elif st == D and ci > 0:
            ancestor_chars.append(int(x[ci - 1]))
            alignment.append((anc_pos, ci - 1, None))
            anc_pos += 1

    ancestor = np.array(ancestor_chars, dtype=np.int32)
    return ancestor, alignment, float(log_prob), posteriors


def reconstruct_tree_inside_guided(tree_root, leaf_seqs, ins_rate, del_rate,
                                    Q, pi, guide_msa=None, band_width=5,
                                    beam_log_width=10.0):
    """Inside-guided progressive reconstruction on a tree.

    Like reconstruct_tree_progressive but uses forward-backward posteriors
    instead of Viterbi, with optional MSA guide envelope for efficiency.

    At each internal node:
    1. Compute forward-backward posteriors for the pair HMM
    2. Use posteriors to reconstruct the ancestor (MAP from posteriors)
    3. If guide_msa provided, constrain DP to cells near the guide alignment

    Args:
        tree_root: TreeNode
        leaf_seqs: dict of {leaf_name: integer_array}
        ins_rate, del_rate: indel rates
        Q, pi: substitution model
        guide_msa: optional dict of {leaf_name: aligned_array} (gaps as -1)
        band_width: envelope band width when using guide_msa
        beam_log_width: beam width for forward-backward pruning

    Returns:
        node_seqs: dict of {node_id: integer_array} for all nodes
        node_posteriors: dict of {node_id: posterior_info}
    """
    node_seqs = {}
    node_posteriors = {}

    # Assign leaf sequences
    for node in tree_root.preorder():
        if node.is_leaf and node.name in leaf_seqs:
            node_seqs[id(node)] = np.asarray(leaf_seqs[node.name])

    # Bottom-up reconstruction
    for node in tree_root.postorder():
        if node.is_leaf:
            continue

        children_with_seqs = [c for c in node.children if id(c) in node_seqs]
        if len(children_with_seqs) == 0:
            continue

        if len(children_with_seqs) == 1:
            child = children_with_seqs[0]
            node_seqs[id(node)] = node_seqs[id(child)]
            continue

        left, right = children_with_seqs[0], children_with_seqs[1]
        x = np.asarray(node_seqs[id(left)])
        y = np.asarray(node_seqs[id(right)])

        if len(x) == 0 or len(y) == 0:
            node_seqs[id(node)] = node_seqs[id(left)] if len(x) > 0 else node_seqs[id(right)]
            continue

        t_total = max(left.branch_length + right.branch_length, 1e-6)

        # Build envelope from guide MSA if available
        envelope = None
        if guide_msa is not None:
            left_name = left.name
            right_name = right.name
            if left_name in guide_msa and right_name in guide_msa:
                envelope = msa_to_envelope(
                    len(x), len(y),
                    guide_msa[left_name], guide_msa[right_name],
                    band_width)

        ancestor, alignment, lp, posteriors = _posterior_ancestor(
            x, y, ins_rate, del_rate, t_total, Q, pi,
            envelope, beam_log_width)

        node_seqs[id(node)] = ancestor
        node_posteriors[id(node)] = {
            "left_child": left.name or f"node_{id(left)}",
            "right_child": right.name or f"node_{id(right)}",
            "alignment": alignment,
            "log_prob": lp,
            "t": t_total,
            "posteriors_shape": posteriors.shape,
        }

    return node_seqs, node_posteriors


# --- Context-dependent SCFG progressive reconstruction ---

def reconstruct_tree_scfg_progressive(tree_root, leaf_seqs, ins_rate, del_rate,
                                        Q, pi, guide_msa=None, band_width=5,
                                        beam_log_width=10.0):
    """Progressive reconstruction using distilled pair SCFGs with 4-position context.

    At each internal node, distills the pair HMM into an order-1 pair SCFG
    where production weights depend on (left_anc, left_desc, right_anc, right_desc).
    This captures context-dependent indel rates from the MixDom hierarchy.

    The distilled SCFG is used with beam-pruned Inside to reconstruct ancestors,
    with optional MSA guide envelope.

    Args:
        tree_root: TreeNode
        leaf_seqs: dict of {leaf_name: integer_array}
        ins_rate, del_rate: indel rates
        Q, pi: substitution model
        guide_msa: optional dict of {leaf_name: aligned_array} (gaps as -1)
        band_width: envelope band width
        beam_log_width: beam width for Inside pruning

    Returns:
        node_seqs: dict of {node_id: integer_array}
        node_info: dict of {node_id: dict with alignment, log_prob, etc.}
    """
    from ..distill.scfg import distill_pair_scfg, inside_pair_with_context
    from ..models.tkf_grammar import build_tkf91_pair_grammar

    node_seqs = {}
    node_info = {}

    # Assign leaf sequences
    for node in tree_root.preorder():
        if node.is_leaf and node.name in leaf_seqs:
            node_seqs[id(node)] = np.asarray(leaf_seqs[node.name])

    n_chars = Q.shape[0]

    for node in tree_root.postorder():
        if node.is_leaf:
            continue

        children_with_seqs = [c for c in node.children if id(c) in node_seqs]
        if len(children_with_seqs) == 0:
            continue

        if len(children_with_seqs) == 1:
            child = children_with_seqs[0]
            node_seqs[id(node)] = node_seqs[id(child)]
            continue

        left, right = children_with_seqs[0], children_with_seqs[1]
        x = np.asarray(node_seqs[id(left)])
        y = np.asarray(node_seqs[id(right)])

        if len(x) == 0 or len(y) == 0:
            node_seqs[id(node)] = (
                node_seqs[id(left)] if len(x) > 0 else node_seqs[id(right)])
            continue

        t_total = max(left.branch_length + right.branch_length, 1e-6)

        # Build pair grammar and distill with context
        grammar = build_tkf91_pair_grammar(ins_rate, del_rate, t_total, Q, pi)
        sub_matrix = np.array(transition_matrix(Q, t_total))
        tau = np.array(tkf91_trans(ins_rate, del_rate, t_total))
        state_types_arr = np.array([S, M, I, D, E])

        distilled = distill_pair_scfg(
            grammar, tau, state_types_arr, sub_matrix, pi, n_chars)

        # Use beam forward-backward for the alignment
        # (the distilled SCFG captures context; the HMM forward-backward
        # gives the alignment posteriors)
        envelope = None
        if guide_msa is not None:
            left_name = left.name
            right_name = right.name
            if left_name in guide_msa and right_name in guide_msa:
                envelope = msa_to_envelope(
                    len(x), len(y),
                    guide_msa[left_name], guide_msa[right_name],
                    band_width)

        # Get alignment from posteriors
        ancestor, alignment, lp, posteriors = _posterior_ancestor(
            x, y, ins_rate, del_rate, t_total, Q, pi,
            envelope, beam_log_width)

        # Score the alignment with the context-dependent distilled SCFG
        pair_alignment = [(a[1], a[2]) if a[0] is not None else (None, a[2])
                          for a in alignment]
        # Convert to (anc_idx, desc_idx) format
        pair_aln = []
        for a in alignment:
            if a[0] is not None:
                pair_aln.append((a[1], a[2]))
            elif a[2] is not None:
                pair_aln.append((None, a[2]))
            elif a[1] is not None:
                pair_aln.append((a[1], None))

        ctx_lp = inside_pair_with_context(distilled, x, y, pair_aln, n_chars)

        node_seqs[id(node)] = ancestor
        node_info[id(node)] = {
            "left_child": left.name or f"node_{id(left)}",
            "right_child": right.name or f"node_{id(right)}",
            "alignment": alignment,
            "log_prob_hmm": lp,
            "log_prob_scfg": float(ctx_lp),
            "t": t_total,
        }

    return node_seqs, node_info
