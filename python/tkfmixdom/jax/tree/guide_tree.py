"""Guide tree construction from pairwise TKF92 distances.

Estimates evolutionary time (tau) between all pairs of sequences using
Forward-Backward E-step + Newton-Raphson optimization on tau, then builds
a Neighbor-Joining tree from the distance matrix.

Method 3 from GuidanceNeeded-response.md.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..core.params import tkf92_trans, S, M, I, D, E
from ..core.ctmc import transition_matrix
from ..util.io import TreeNode
from ..dp.hmm import safe_log


def _expected_loglik_tau(tau, n_trans, match_weight_matrix, Q, pi,
                         ins_rate, del_rate, ext, tau_reg=0.01):
    """Expected complete-data log-likelihood as a function of tau.

    After the E-step (Forward-Backward at some tau_0), the expected
    log-likelihood is:
        Q(tau) = sum_ij n_ij * log(chi_ij(tau))
               + sum_ab w_ab * log(P(tau)_ab)
               - tau_reg * tau

    where chi is the TKF92 transition matrix and P(tau) is the
    substitution matrix.

    Args:
        tau: evolutionary time (scalar, positive)
        n_trans: (5, 5) expected transition counts from E-step
        match_weight_matrix: (A, A) weighted match pair counts
        Q: (A, A) rate matrix
        pi: (A,) stationary distribution
        ins_rate, del_rate, ext: TKF92 parameters
        tau_reg: regularization strength (Exponential prior on tau)

    Returns:
        scalar log-likelihood value
    """
    # Transition contribution
    chi = tkf92_trans(ins_rate, del_rate, tau, ext)
    log_chi = safe_log(chi)
    trans_ll = jnp.sum(n_trans * log_chi)

    # Emission contribution from match pairs
    P_t = transition_matrix(Q, tau)
    log_P = safe_log(P_t)
    emit_ll = jnp.sum(match_weight_matrix * log_P)

    # Regularizer (weak exponential prior on tau)
    reg = -tau_reg * tau

    return trans_ll + emit_ll + reg


def _match_pairs_to_matrix(match_pairs, alphabet_size):
    """Convert list of (a, b, weight) triples to (A, A) weight matrix."""
    W = np.zeros((alphabet_size, alphabet_size))
    for a, b, w in match_pairs:
        W[a, b] += w
    return W


def estimate_pairwise_distance(x_seq, y_seq, ins_rate, del_rate, ext,
                                Q, pi, n_newton=5, tau_init=1.0,
                                tau_reg=0.01):
    """Estimate TKF92 evolutionary distance between two sequences.

    Does one Forward-Backward E-step at tau_init, then optimizes tau
    via Newton-Raphson on the expected complete-data log-likelihood.

    Args:
        x_seq, y_seq: integer sequence arrays
        ins_rate, del_rate, ext: TKF92 parameters
        Q: (A, A) rate matrix
        pi: (A,) stationary distribution
        n_newton: number of Newton-Raphson steps
        tau_init: initial evolutionary time estimate
        tau_reg: regularization strength

    Returns:
        tau: estimated evolutionary time (float)
        log_prob: forward log-probability at initial tau
    """
    from ..models.compiled import TKF92Model

    A = Q.shape[0]

    # E-step: Forward-Backward at initial tau
    model = TKF92Model()
    params = {'ins_rate': ins_rate, 'del_rate': del_rate, 't': tau_init,
              'ext': ext, 'Q': Q, 'pi': pi}
    log_prob, n_trans, posteriors = model.e_step(params,
        jnp.asarray(x_seq), jnp.asarray(y_seq))

    # Extract match pairs from posteriors
    from ..core.params import M as _M
    x_arr = jnp.asarray(x_seq)
    y_arr = jnp.asarray(y_seq)
    Lx, Ly = x_arr.shape[0], y_arr.shape[0]
    # Vectorized match weight matrix: sum posteriors by (x_char, y_char)
    match_post = posteriors[1:Lx+1, 1:Ly+1, _M]  # (Lx, Ly)
    # Use einsum-style scatter: for each (ix, iy), add match_post[ix,iy] to W[x[ix], y[iy]]
    match_W = jnp.zeros((A, A))
    match_W = match_W.at[x_arr[:, None], y_arr[None, :]].add(match_post)
    n_trans_jnp = jnp.array(n_trans)

    # Newton-Raphson on z where tau = softplus(z)
    # softplus(z) = log(1 + exp(z))
    # Initialize z so that softplus(z) = tau_init
    z = float(jnp.log(jnp.expm1(jnp.array(tau_init))))

    # Use JAX value_and_grad for efficient Newton steps.
    # Compute gradient analytically: d(LL)/d(tau) from the expected LL formula,
    # then convert to d/dz via chain rule (softplus).
    # This avoids JAX tracing overhead from jax.grad on a closure.
    from functools import partial

    @partial(jax.jit, static_argnames=('ins_r', 'del_r', 'ext_r', 'reg'))
    def _val_and_grad(z_val, n_trans, match_w, Q_mat, pi_vec,
                      ins_r, del_r, ext_r, reg):
        def obj(z):
            tau_val = jax.nn.softplus(z)
            return _expected_loglik_tau(
                tau_val, n_trans, match_w, Q_mat, pi_vec,
                ins_r, del_r, ext_r, reg)
        val, g = jax.value_and_grad(obj)(z_val)
        return val, g

    for _ in range(n_newton):
        val, g = _val_and_grad(
            jnp.float64(z), n_trans_jnp, match_W, Q, pi,
            ins_r=float(ins_rate), del_r=float(del_rate),
            ext_r=float(ext), reg=float(tau_reg))
        g = float(g)
        # Approximate Hessian via finite difference
        eps = 1e-4
        _, g_plus = _val_and_grad(
            jnp.float64(z + eps), n_trans_jnp, match_W, Q, pi,
            ins_r=float(ins_rate), del_r=float(del_rate),
            ext_r=float(ext), reg=float(tau_reg))
        h = float(g_plus - g) / eps

        if h < -1e-10:
            # Newton step (concave region)
            step = -g / h
            # Clip step size
            step = max(min(step, 2.0), -2.0)
            z = z + step
        else:
            # Gradient step with small learning rate (non-concave region)
            z = z + 0.1 * g

    tau = float(jax.nn.softplus(jnp.array(z)))
    # Clamp to reasonable range
    tau = max(min(tau, 10.0), 1e-4)

    return tau, log_prob


def pairwise_distance_matrix(sequences, ins_rate, del_rate, ext,
                              Q, pi, n_newton=5, tau_init=1.0,
                              verbose=False):
    """Compute all-pairs TKF92 distance matrix.

    Args:
        sequences: dict of {name: integer_array}
        ins_rate, del_rate, ext: TKF92 parameters
        Q, pi: substitution model
        n_newton: Newton-Raphson steps per pair
        tau_init: initial tau for all pairs
        verbose: print progress

    Returns:
        names: list of sequence names
        dist_matrix: (n, n) symmetric distance matrix
    """
    names = list(sequences.keys())
    n = len(names)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if verbose:
                print(f"  Distance {names[i]} vs {names[j]}...", end=" ", flush=True)
            tau, _ = estimate_pairwise_distance(
                sequences[names[i]], sequences[names[j]],
                ins_rate, del_rate, ext, Q, pi,
                n_newton=n_newton, tau_init=tau_init
            )
            D[i, j] = D[j, i] = tau
            if verbose:
                print(f"tau={tau:.4f}")

    return names, D


def neighbor_joining(dist_matrix, names):
    """Build a Neighbor-Joining tree from a distance matrix.

    Args:
        dist_matrix: (n, n) symmetric distance matrix
        names: list of n sequence names

    Returns:
        TreeNode: root of the NJ tree
    """
    n = len(names)
    if n == 1:
        return TreeNode(names[0], branch_length=0.1)
    if n == 2:
        root = TreeNode(None)
        d = max(dist_matrix[0, 1] / 2, 1e-4)
        root.add_child(TreeNode(names[0], branch_length=d))
        root.add_child(TreeNode(names[1], branch_length=d))
        return root

    D = dist_matrix.copy()
    active = list(range(n))
    # Each label is either a leaf name (str) or a TreeNode
    labels = [TreeNode(name, branch_length=0.0) for name in names]

    while len(active) > 2:
        m = len(active)
        # Compute r_i
        r = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += D[active[i], active[j]]

        # Find pair with minimum Q criterion
        best_q = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * D[active[i], active[j]] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j

        ai, aj = active[best_i], active[best_j]
        d_ij = D[ai, aj]

        # Branch lengths
        if m > 2:
            bl_i = d_ij / 2.0 + (r[best_i] - r[best_j]) / (2.0 * (m - 2))
            bl_j = d_ij - bl_i
        else:
            bl_i = bl_j = d_ij / 2.0

        bl_i = max(bl_i, 1e-4)
        bl_j = max(bl_j, 1e-4)

        # Create internal node
        new_node = TreeNode(None)
        child_i = labels[ai]
        child_j = labels[aj]
        child_i.branch_length = bl_i
        child_j.branch_length = bl_j
        new_node.add_child(child_i)
        new_node.add_child(child_j)

        # Update distance matrix
        new_idx = len(D)
        D_new = np.zeros((new_idx + 1, new_idx + 1))
        D_new[:new_idx, :new_idx] = D
        for k_idx in range(m):
            k = active[k_idx]
            if k != ai and k != aj:
                d_new = (D[ai, k] + D[aj, k] - d_ij) / 2.0
                D_new[new_idx, k] = D_new[k, new_idx] = max(d_new, 0.0)
        D = D_new

        # Extend labels
        labels.append(new_node)

        # Update active
        active.remove(ai)
        active.remove(aj)
        active.append(new_idx)

    # Final join
    ai, aj = active[0], active[1]
    d = max(D[ai, aj] / 2, 1e-4)
    root = TreeNode(None)
    child_i = labels[ai]
    child_j = labels[aj]
    child_i.branch_length = d
    child_j.branch_length = d
    root.add_child(child_i)
    root.add_child(child_j)

    return root


def build_guide_tree(sequences, ins_rate=0.05, del_rate=0.10, ext=0.5,
                     Q=None, pi=None, n_newton=5, tau_init=1.0,
                     verbose=False):
    """Build a Neighbor-Joining guide tree from TKF92 pairwise distances.

    Args:
        sequences: dict of {name: integer_array}
        ins_rate, del_rate, ext: TKF92 parameters
        Q, pi: substitution model (defaults to LG if None)
        n_newton: Newton-Raphson steps per pair
        tau_init: initial tau for all pairs
        verbose: print progress

    Returns:
        TreeNode: root of the guide tree
    """
    if Q is None or pi is None:
        from ..core.protein import rate_matrix_lg
        Q, pi = rate_matrix_lg()

    names, D = pairwise_distance_matrix(
        sequences, ins_rate, del_rate, ext, Q, pi,
        n_newton=n_newton, tau_init=tau_init, verbose=verbose
    )

    return neighbor_joining(D, names)


def upgma(dist_matrix, names):
    """Build a UPGMA (Unweighted Pair Group Method with Arithmetic Mean) tree.

    Produces an ultrametric tree (all root-to-leaf distances are equal).

    Args:
        dist_matrix: (n, n) symmetric distance matrix
        names: list of n sequence names

    Returns:
        TreeNode: root of the UPGMA tree
    """
    n = len(names)
    if n == 1:
        return TreeNode(names[0], branch_length=0.1)
    if n == 2:
        root = TreeNode(None)
        d = max(dist_matrix[0, 1] / 2, 1e-4)
        root.add_child(TreeNode(names[0], branch_length=d))
        root.add_child(TreeNode(names[1], branch_length=d))
        return root

    D = dist_matrix.copy()
    active = list(range(n))
    # Each entry: (TreeNode, height) where height is the UPGMA node height
    nodes = [(TreeNode(name, branch_length=0.0), 0.0) for name in names]
    # Cluster sizes for weighted averaging
    sizes = [1] * n

    while len(active) > 2:
        m = len(active)
        # Find closest pair
        best_d = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                if D[active[i], active[j]] < best_d:
                    best_d = D[active[i], active[j]]
                    best_i, best_j = i, j

        ai, aj = active[best_i], active[best_j]
        new_height = best_d / 2.0

        # Create merged node
        new_node = TreeNode(None)
        child_i, h_i = nodes[ai]
        child_j, h_j = nodes[aj]
        child_i.branch_length = max(new_height - h_i, 1e-4)
        child_j.branch_length = max(new_height - h_j, 1e-4)
        new_node.add_child(child_i)
        new_node.add_child(child_j)

        # Update distance matrix (weighted average)
        new_idx = len(D)
        D_new = np.zeros((new_idx + 1, new_idx + 1))
        D_new[:new_idx, :new_idx] = D
        si, sj = sizes[ai], sizes[aj]
        new_size = si + sj
        for k in active:
            if k != ai and k != aj:
                d_new = (si * D[ai, k] + sj * D[aj, k]) / new_size
                D_new[new_idx, k] = D_new[k, new_idx] = d_new
        D = D_new

        nodes.append((new_node, new_height))
        sizes.append(new_size)

        active.remove(ai)
        active.remove(aj)
        active.append(new_idx)

    # Final join
    ai, aj = active[0], active[1]
    new_height = D[ai, aj] / 2.0
    root = TreeNode(None)
    child_i, h_i = nodes[ai]
    child_j, h_j = nodes[aj]
    child_i.branch_length = max(new_height - h_i, 1e-4)
    child_j.branch_length = max(new_height - h_j, 1e-4)
    root.add_child(child_i)
    root.add_child(child_j)

    return root


def star_tree(names, branch_length=0.5):
    """Build a star tree where all leaves radiate from a single root.

    Args:
        names: list of sequence names
        branch_length: branch length for all edges (default 0.5)

    Returns:
        TreeNode: root of the star tree
    """
    if len(names) == 1:
        return TreeNode(names[0], branch_length=branch_length)

    root = TreeNode(None)
    for name in names:
        root.add_child(TreeNode(name, branch_length=branch_length))
    return root
