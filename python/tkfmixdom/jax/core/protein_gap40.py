"""Fels40: 40-state substitution model with hidden gapped states.

40 states = 20 "present" states (emit their amino acid) + 20 "gapped" states
(emit gap character). When a position becomes a gap, it retains a "memory"
of which amino acid it was. This allows the model to track hidden substitution
history through indel events.

State layout:
  - States 0-19:  "present" states -- state i emits amino acid i
  - States 20-39: "gapped" states -- state 20+i is the gapped version of
                  amino acid i, always emits gap character

Rate matrix Q40:
  Q40 = [ Q_aa    Q_del  ]     Q_aa:  20x20 (LG08 for present->present)
         [ Q_ins   Q_gap  ]     Q_del: 20x20 (present->gapped)
                                 Q_ins: 20x20 (gapped->present)
                                 Q_gap: 20x20 (gapped->gapped, hidden subst)

Emission matrix E (40 x 21):
  E[i, a]     = delta(i, a)   for i in 0..19, a in 0..19  (present emits AA)
  E[20+i, 20] = 1             for i in 0..19               (gapped emits gap)
  E[20+i, a]  = 0             for a in 0..19               (gapped never emits AA)

This is NOT a standard reversible substitution model in the observed space --
it has hidden states. Felsenstein pruning sums over the hidden gapped states
when computing likelihoods at gap-observed leaves.
"""

import jax.numpy as jnp
import numpy as np

from .protein import rate_matrix_lg


def build_Q40(r_del=0.03, r_ins=0.03, gap_subst_scale=0.0):
    """Build the 40x40 rate matrix and equilibrium distribution.

    Parameterization:
      - Q_aa = LG08 Q matrix (present -> present, standard AA substitution)
      - Q_del = diag(r_del) (present_i -> gapped_i, deletion)
      - Q_ins = diag(r_ins) (gapped_i -> present_i, insertion)
      - Q_gap = LG08 Q * gap_subst_scale (hidden substitution while gapped)

    For simplicity, present_i -> gapped_j (i != j) is set to 0 (no simultaneous
    substitution + deletion). Similarly gapped_i -> present_j (i != j) is 0.

    Detailed balance at equilibrium:
      pi40[i] * Q40[i, 20+i] = pi40[20+i] * Q40[20+i, i]
      pi_aa[i] * r_del = pi_gap[i] * r_ins
      => pi_gap[i] / pi_aa[i] = r_del / r_ins

    So pi40[i]    = pi_lg[i] * r_ins / (r_ins + r_del)  (present)
       pi40[20+i] = pi_lg[i] * r_del / (r_ins + r_del)  (gapped)

    Args:
        r_del: deletion rate (present -> gapped). Default 0.03.
        r_ins: insertion rate (gapped -> present). Default 0.03.
        gap_subst_scale: scaling factor for hidden substitution in gapped states.
                         0.0 = no hidden substitution (simplest model).
                         1.0 = same rate as present-state substitution.

    Returns:
        Q40: (40, 40) rate matrix with rows summing to 0
        pi40: (40,) equilibrium distribution
    """
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg = np.asarray(Q_lg)
    pi_lg = np.asarray(pi_lg)

    Q40 = np.zeros((40, 40))

    # Top-left: present -> present (LG08 substitution)
    Q40[:20, :20] = Q_lg

    # Top-right: present -> gapped (deletion)
    # Only present_i -> gapped_i (diagonal): deletion preserves AA identity
    for i in range(20):
        Q40[i, 20 + i] = r_del

    # Bottom-left: gapped -> present (insertion)
    # Only gapped_i -> present_i (diagonal): insertion restores same AA
    for i in range(20):
        Q40[20 + i, i] = r_ins

    # Bottom-right: gapped -> gapped (hidden substitution while deleted)
    if gap_subst_scale > 0:
        Q40[20:, 20:] = Q_lg * gap_subst_scale

    # Fix diagonals so rows sum to 0
    np.fill_diagonal(Q40, 0.0)
    np.fill_diagonal(Q40, -Q40.sum(axis=1))

    # Equilibrium distribution
    kappa = r_ins / (r_ins + r_del)
    pi40 = np.zeros(40)
    pi40[:20] = pi_lg * kappa        # present states
    pi40[20:] = pi_lg * (1 - kappa)  # gapped states
    pi40 = pi40 / pi40.sum()         # normalize

    # Normalize Q40 so mean rate = 1 over pi40
    mean_rate = -np.sum(pi40 * np.diag(Q40))
    if mean_rate > 0:
        Q40 = Q40 / mean_rate

    return jnp.array(Q40), jnp.array(pi40)


def build_emission_matrix():
    """Build the 40 x 21 emission matrix.

    Observed alphabet: 20 amino acids (indices 0-19) + gap (index 20).

    Returns:
        E: (40, 21) emission matrix
           E[s, o] = P(observed = o | hidden state = s)
    """
    E = np.zeros((40, 21))
    # Present states emit their amino acid deterministically
    E[:20, :20] = np.eye(20)
    # Gapped states emit gap deterministically
    E[20:, 20] = 1.0
    return jnp.array(E)


def leaf_conditional(observed_char, emission=None):
    """Compute leaf conditional likelihood vector for Felsenstein pruning.

    For an observed character, returns a 40-dimensional vector where entry s
    gives the probability of observing that character if hidden state is s.

    Args:
        observed_char: int. 0-19 for amino acid, 20 or -1 for gap.
        emission: optional (40, 21) emission matrix. Built if not provided.

    Returns:
        cond: (40,) conditional likelihood vector
    """
    if emission is None:
        emission = build_emission_matrix()
    emission = np.asarray(emission)

    if observed_char < 0:
        observed_char = 20  # map -1 to gap index

    return emission[:, observed_char]


def felsenstein_40(tree_root, msa, Q40, pi40, emission=None):
    """Felsenstein pruning with the 40-state model.

    At each leaf, the conditional likelihood is set by the emission matrix:
    - If observed AA a (0-19): cond[a] = 1, cond[20+i] = 0 for all i
      (only present state a is compatible)
    - If observed gap (-1 or 20): cond[i] = 0 for i in 0..19,
      cond[20+i] = 1 for all i (any gapped state is compatible)

    At internal nodes: standard Felsenstein peeling with P(t) = expm(Q40 * t).
    Root likelihood: sum over s of pi40[s] * CL_root[s].

    Args:
        tree_root: TreeNode (root of phylogenetic tree)
        msa: dict {leaf_name: list_of_ints} where values are 0-19 (AA) or -1 (gap)
        Q40: (40, 40) rate matrix
        pi40: (40,) equilibrium distribution
        emission: optional (40, 21) emission matrix

    Returns:
        total_log_prob: float, log-likelihood of the MSA
    """
    from ..core.ctmc import transition_matrix

    if emission is None:
        emission = build_emission_matrix()
    emission_np = np.asarray(emission)

    leaf_names = list(msa.keys())
    msa_len = len(next(iter(msa.values())))
    N = 40

    # Precompute transition matrices P(t) = expm(Q40 * t)
    sub_matrices = {}
    for node in tree_root.preorder():
        for child in node.children:
            t = max(child.branch_length, 1e-6)
            P = np.asarray(transition_matrix(Q40, t))
            sub_matrices[id(child)] = P

    pi_np = np.asarray(pi40)
    total_lp = 0.0

    for col in range(msa_len):
        # Build leaf observations for this column
        col_chars = {}
        for name in leaf_names:
            c = msa[name][col]
            col_chars[name] = 20 if c < 0 else c

        def _prune(node):
            """Returns (cond_likelihood_40, log_scale)."""
            if node.is_leaf:
                char = col_chars.get(node.name, 20)
                cond = emission_np[:, char].copy()
                return cond, 0.0

            partial = np.ones(N)
            log_scale = 0.0
            for child in node.children:
                child_cond, child_ls = _prune(child)
                P = sub_matrices[id(child)]
                partial = partial * (P @ child_cond)
                log_scale += child_ls

            max_val = max(np.max(partial), 1e-300)
            partial = partial / max_val
            log_scale += np.log(max_val)
            return partial, log_scale

        root_cond, log_scale = _prune(tree_root)
        prob = np.sum(pi_np * root_cond)
        total_lp += log_scale + np.log(max(prob, 1e-300))

    return total_lp


def reconstruct_ancestors_40(tree_root, msa, Q40, pi40, emission=None):
    """Reconstruct MAP ancestral states at root using the 40-state model.

    For each column:
    1. Felsenstein pruning (inside pass) to get root conditional likelihoods
    2. Multiply by pi40, normalize to get root posterior over 40 states
    3. MAP state: if present (0-19), predict that AA; if gapped (20-39), predict gap

    Args:
        tree_root: TreeNode
        msa: dict {leaf_name: list_of_ints} where values are 0-19 (AA) or -1 (gap)
        Q40: (40, 40) rate matrix
        pi40: (40,) equilibrium distribution
        emission: optional (40, 21) emission matrix

    Returns:
        root_seq: list of ints (0-19 for AA, 20 for gap)
        posteriors: (L, 40) posterior probabilities at root over hidden states
    """
    from ..core.ctmc import transition_matrix

    if emission is None:
        emission = build_emission_matrix()
    emission_np = np.asarray(emission)

    leaf_names = list(msa.keys())
    msa_len = len(next(iter(msa.values())))
    N = 40

    # Precompute transition matrices
    sub_matrices = {}
    for node in tree_root.preorder():
        for child in node.children:
            t = max(child.branch_length, 1e-6)
            P = np.asarray(transition_matrix(Q40, t))
            sub_matrices[id(child)] = P

    pi_np = np.asarray(pi40)
    root_seq = []
    all_posteriors = []

    for col in range(msa_len):
        col_chars = {}
        for name in leaf_names:
            c = msa[name][col]
            col_chars[name] = 20 if c < 0 else c

        def _prune(node):
            if node.is_leaf:
                char = col_chars.get(node.name, 20)
                cond = emission_np[:, char].copy()
                return cond, 0.0

            partial = np.ones(N)
            log_scale = 0.0
            for child in node.children:
                child_cond, child_ls = _prune(child)
                P = sub_matrices[id(child)]
                partial = partial * (P @ child_cond)
                log_scale += child_ls

            max_val = max(np.max(partial), 1e-300)
            partial = partial / max_val
            log_scale += np.log(max_val)
            return partial, log_scale

        root_cond, _ = _prune(tree_root)
        joint = pi_np * root_cond
        posterior = joint / max(np.sum(joint), 1e-300)

        # MAP state
        map_state = int(np.argmax(posterior))
        if map_state >= 20:
            root_seq.append(20)  # gapped -> predict gap
        else:
            root_seq.append(map_state)  # present -> predict AA

        all_posteriors.append(posterior)

    return root_seq, np.array(all_posteriors)


def reconstruct_leaf_40(tree_root, msa, target_name, Q40, pi40, emission=None):
    """Predict a withheld leaf's sequence using Felsenstein inside-outside.

    1. Remove target leaf from tree
    2. Run Felsenstein pruning on pruned tree to get root posterior
    3. Propagate root posterior down to the target attachment point
    4. For each column, predict the MAP observed character

    For simplicity, this uses the root posterior + total branch length to target
    as an approximation (rather than full inside-outside).

    Args:
        tree_root: TreeNode
        msa: dict {leaf_name: list_of_ints}
        target_name: name of leaf to predict
        Q40: (40, 40) rate matrix
        pi40: (40,) equilibrium distribution
        emission: optional (40, 21) emission matrix

    Returns:
        predicted_seq: list of ints (0-19 for AA, 20 for gap)
        posteriors_observed: (L, 21) posterior over observed characters
    """
    from ..core.ctmc import transition_matrix

    if emission is None:
        emission = build_emission_matrix()
    emission_np = np.asarray(emission)

    N = 40

    # Find target leaf and its branch length
    target_leaf = None
    for node in tree_root.preorder():
        if node.is_leaf and node.name == target_name:
            target_leaf = node
            break
    if target_leaf is None:
        raise ValueError(f"Leaf {target_name} not found in tree")

    # Build pruned MSA (without target)
    pruned_msa = {k: v for k, v in msa.items() if k != target_name}
    msa_len = len(next(iter(msa.values())))

    # Remove target from tree (deep copy first)
    def _deep_copy(node, parent=None):
        new_node = type(node)(node.name, node.branch_length)
        new_node.parent = parent
        for c in node.children:
            new_child = _deep_copy(c, new_node)
            new_node.children.append(new_child)
        return new_node

    pruned_tree = _deep_copy(tree_root)

    # Find and remove target in the copy
    target_bl = target_leaf.branch_length
    target_in_copy = None
    for node in pruned_tree.preorder():
        if node.is_leaf and node.name == target_name:
            target_in_copy = node
            break

    if target_in_copy is not None:
        parent = target_in_copy.parent
        if parent is not None:
            parent.children = [c for c in parent.children if c.name != target_name]
            # If parent now has 1 child and is not root, merge
            if len(parent.children) == 1 and parent.parent is not None:
                remaining = parent.children[0]
                grandparent = parent.parent
                remaining.branch_length += parent.branch_length
                remaining.parent = grandparent
                grandparent.children = [
                    remaining if c is parent else c
                    for c in grandparent.children
                ]
                # Track total distance from attachment point
                target_bl += parent.branch_length
            # If root has 1 child, promote
            if parent.parent is None and len(parent.children) == 1:
                new_root = parent.children[0]
                new_root.parent = None
                pruned_tree = new_root

    # Compute root posterior on pruned tree, then propagate to target
    root_seq, root_posteriors = reconstruct_ancestors_40(
        pruned_tree, pruned_msa, Q40, pi40, emission
    )

    # Propagate from root to target attachment point
    # P_target[s] = sum_r P(t_total)[r, s] * posterior_root[r]
    t_total = target_bl
    P_t = np.asarray(transition_matrix(Q40, max(t_total, 1e-6)))

    predicted_seq = []
    posteriors_observed = []

    for col in range(msa_len):
        # Root posterior over 40 hidden states
        post_root = root_posteriors[col]
        # Propagate to target: P(target_state | data) ~ sum_r P(t)[r,s] * post[r]
        post_target = P_t.T @ post_root
        post_target = post_target / max(np.sum(post_target), 1e-300)

        # Map to observed space via emission matrix: P(obs=o) = sum_s E[s,o] * post[s]
        post_obs = emission_np.T @ post_target  # (21,)
        post_obs = post_obs / max(np.sum(post_obs), 1e-300)

        map_obs = int(np.argmax(post_obs))
        predicted_seq.append(map_obs)
        posteriors_observed.append(post_obs)

    return predicted_seq, np.array(posteriors_observed)
