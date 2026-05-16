"""LG08+gap: 21-state protein substitution model treating gap as a character.

Extends the LG08 20-state amino acid model with a 21st "gap" state,
adding learnable rates for substitution-to-gap and gap-to-substitution
transitions, constrained to maintain reversibility.

This model treats indels as a special type of substitution, allowing
standard Felsenstein pruning on a fixed MSA (with gaps treated as
observed characters rather than missing data). This provides a simple
baseline for phylogenetic reconstruction that doesn't require a
separate indel model.

The extended rate matrix Q21 is:
- Q21[i,j] = Q_lg[i,j]    for i,j in 0..19  (amino acid to amino acid)
- Q21[i,20] = r_del * pi_gap   for i in 0..19  (amino acid to gap)
- Q21[20,j] = r_ins * pi_j     for j in 0..19  (gap to amino acid)
- Q21[20,20] = -sum(Q21[20,:])

where r_del and r_ins are learnable parameters and pi_gap is the
equilibrium gap frequency, determined by detailed balance:
    pi_gap * r_ins * pi_j = pi_j * r_del * pi_gap   (for each j)
which simplifies to r_ins = r_del, so we use a single rate parameter r_gap.

The equilibrium distribution is:
    pi21[i] = (1 - f_gap) * pi_lg[i]   for i in 0..19
    pi21[20] = f_gap

where f_gap is the equilibrium gap frequency.
"""

import os

import jax.numpy as jnp
import numpy as np

from .protein import rate_matrix_lg
from ..util.io import AA_TO_INT as _AA_TO_INT


def rate_matrix_lg_gap(r_gap=0.1, f_gap=0.05):
    """Build LG08+gap 21-state rate matrix.

    Args:
        r_gap: rate of gap-to-aa and aa-to-gap transitions (symmetric for
               reversibility). Higher = more indels.
        f_gap: equilibrium gap frequency. Typical values: 0.01-0.10.

    Returns:
        Q21: (21, 21) rate matrix
        pi21: (21,) equilibrium frequencies
    """
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg = np.asarray(Q_lg)
    pi_lg = np.asarray(pi_lg)

    # Scale pi_lg to account for gap frequency
    pi_aa = (1.0 - f_gap) * pi_lg  # (20,) amino acid frequencies
    pi21 = np.zeros(21)
    pi21[:20] = pi_aa
    pi21[20] = f_gap

    # Build Q21
    Q21 = np.zeros((21, 21))

    # AA-AA block: scale LG08 rates
    # The LG08 Q already satisfies Q_ij = S_ij * pi_j and is normalized.
    # We keep the same exchangeabilities but scale the pi in Q_ij.
    # Actually, for simplicity, we keep Q_lg unchanged for the AA block
    # and just add the gap row/col. The normalization will be adjusted.
    Q21[:20, :20] = Q_lg

    # AA -> gap: Q21[i, 20] = r_gap * pi21[20] / pi21[i] * pi21[i]
    # For reversibility: pi21[i] * Q21[i,20] = pi21[20] * Q21[20,i]
    # Choose Q21[i,20] = r_gap * f_gap and Q21[20,i] = r_gap * pi_aa[i]
    # Check: pi_aa[i] * r_gap * f_gap = f_gap * r_gap * pi_aa[i] ✓
    Q21[:20, 20] = r_gap * f_gap
    Q21[20, :20] = r_gap * pi_aa

    # Set diagonal
    np.fill_diagonal(Q21, 0.0)
    np.fill_diagonal(Q21, -Q21.sum(axis=1))

    # Renormalize to mean rate of 1
    mean_rate = -np.sum(pi21 * np.diag(Q21))
    if mean_rate > 0:
        Q21 = Q21 / mean_rate

    return jnp.array(Q21), jnp.array(pi21)


def felsenstein_pruning_gap(tree_root, msa, Q21, pi21):
    """Felsenstein pruning treating gaps as the 21st character.

    Unlike standard Felsenstein pruning where gaps are missing data,
    this treats gap as an observed character with index 20.

    Args:
        tree_root: TreeNode
        msa: dict of {leaf_name: list_of_ints} where -1 = gap (mapped to 20)
        Q21: (21, 21) rate matrix
        pi21: (21,) equilibrium frequencies

    Returns:
        total_log_prob: total log-likelihood of the MSA
    """
    from ..core.ctmc import transition_matrix

    leaf_names = list(msa.keys())
    msa_len = len(next(iter(msa.values())))
    A = 21

    # Precompute transition matrices
    sub_matrices = {}
    for node in tree_root.preorder():
        for child in node.children:
            t = max(child.branch_length, 1e-6)
            P = np.asarray(transition_matrix(Q21, t))
            sub_matrices[id(child)] = P

    pi_np = np.asarray(pi21)
    total_lp = 0.0

    for col in range(msa_len):
        col_chars = {}
        for name in leaf_names:
            c = msa[name][col]
            # Map -1 (gap) to 20 (gap character)
            col_chars[name] = 20 if c < 0 else c

        def _prune(node):
            if node.is_leaf:
                char = col_chars.get(node.name, 20)
                cond = np.zeros(A)
                cond[char] = 1.0
                return cond, 0.0

            partial = np.ones(A)
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


def reconstruct_root_gap(tree_root, msa, Q21, pi21):
    """Reconstruct MAP root sequence treating gaps as 21st character.

    For each column, compute Felsenstein posterior and take argmax.
    If argmax = 20 (gap), the position is deleted in the ancestor.

    Args:
        tree_root: TreeNode
        msa: dict of {leaf_name: list_of_ints} where -1 = gap
        Q21: (21, 21) rate matrix
        pi21: (21,) equilibrium frequencies

    Returns:
        root_seq: list of ints (0-19 for amino acids, 20 for gap)
        posteriors: (L, 21) posterior probabilities at root
    """
    from ..core.ctmc import transition_matrix

    leaf_names = list(msa.keys())
    msa_len = len(next(iter(msa.values())))
    A = 21

    sub_matrices = {}
    for node in tree_root.preorder():
        for child in node.children:
            t = max(child.branch_length, 1e-6)
            P = np.asarray(transition_matrix(Q21, t))
            sub_matrices[id(child)] = P

    pi_np = np.asarray(pi21)
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
                cond = np.zeros(A)
                cond[char] = 1.0
                return cond, 0.0

            partial = np.ones(A)
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
        root_seq.append(int(np.argmax(posterior)))
        all_posteriors.append(posterior)

    return root_seq, np.array(all_posteriors)


# ── Full GTR 21-state model (fitted to data) ──────────────────────────

def load_fels21_model(path=None):
    """Load a fitted 21-state GTR model from disk.

    Args:
        path: path to .npz file (default: pfam/fels21_fitted.npz
              relative to the python/ directory)

    Returns:
        Q21: (21, 21) rate matrix
        pi21: (21,) equilibrium frequencies
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "pfam", "fels21_fitted.npz")
    data = np.load(path)
    return jnp.array(data["Q21"]), jnp.array(data["pi21"])


def rate_matrix_gtr21(S21, pi21):
    """Build a 21-state GTR rate matrix from exchangeabilities and frequencies.

    Args:
        S21: (21, 21) symmetric exchangeability matrix (zero diagonal)
        pi21: (21,) equilibrium frequencies

    Returns:
        Q21: (21, 21) rate matrix, normalized to mean rate 1
        pi21: (21,) equilibrium frequencies (unchanged)
    """
    Q21 = np.asarray(S21) * np.asarray(pi21)[None, :]
    np.fill_diagonal(Q21, 0.0)
    np.fill_diagonal(Q21, -Q21.sum(axis=1))
    mean_rate = -np.sum(np.asarray(pi21) * np.diag(Q21))
    if mean_rate > 1e-30:
        Q21 = Q21 / mean_rate
    return jnp.array(Q21), jnp.array(pi21)


def _reroot_at_node(tree, target_node):
    """Re-root a tree at a given node (by name). Returns new root."""
    import copy
    from ..util.io import TreeNode

    new_tree = copy.deepcopy(tree)
    target = None
    for node in new_tree.preorder():
        if node.name == target_node.name:
            target = node
            break
    if target is None:
        raise ValueError(f"Node '{target_node.name}' not found in tree")
    if target.parent is None:
        return new_tree

    path = []
    current = target
    while current is not None:
        path.append(current)
        current = current.parent

    for i in range(len(path) - 1):
        child_node = path[i]
        parent_node = path[i + 1]
        parent_node.children = [c for c in parent_node.children
                                if c is not child_node]
        parent_node.branch_length = child_node.branch_length
        child_node.children.append(parent_node)
        parent_node.parent = child_node

    target.parent = None
    target.branch_length = 0.0
    return target


def reconstruct_held_out_fels21(msa_strings, target_name, Q21=None, pi21=None,
                                model_path=None):
    """Reconstruct a held-out leaf using the full GTR 21-state model.

    This function:
    1. Builds a FastTree from the full MSA (including target)
    2. Re-roots the tree at the target's parent
    3. Removes the target leaf
    4. Runs Felsenstein pruning with the 21-state model (gaps = char 20)
    5. Returns the predicted sequence (residues where MAP != gap)

    Unlike reconstruct_root_gap which uses a 2-parameter gap model,
    this uses a full GTR model with independent AA<->gap rates fitted
    to Pfam data.

    Args:
        msa_strings: dict {name: aligned_string} -- ALL sequences
            including target, aligned (with gaps as '-')
        target_name: name of the held-out sequence
        Q21: (21, 21) rate matrix (default: load from pfam/fels21_fitted.npz)
        pi21: (21,) equilibrium frequencies
        model_path: path to fitted model .npz (used if Q21/pi21 not given)

    Returns:
        pred_seq: integer array of predicted residues (gap-stripped)
        posteriors_full: (L, 21) full posteriors at each column
        elapsed: wall time in seconds
    """
    import shutil
    import tempfile
    import subprocess
    import time

    from ..util.io import parse_newick

    t0 = time.time()

    # Load model if not provided
    if Q21 is None or pi21 is None:
        Q21, pi21 = load_fels21_model(model_path)
    Q21 = np.asarray(Q21)
    pi21 = np.asarray(pi21)

    if target_name not in msa_strings:
        raise ValueError(f"Target '{target_name}' not found in MSA")

    # Step 1: Build tree via FastTree
    ft = shutil.which("FastTree")
    if ft is None:
        home_bin = os.path.expanduser("~/bin/FastTree")
        if os.path.isfile(home_bin) and os.access(home_bin, os.X_OK):
            ft = home_bin
        else:
            raise FileNotFoundError("FastTree not found")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
        tmpfa = f.name
        for name, seq in msa_strings.items():
            f.write(f">{name}\n{seq}\n")

    try:
        result = subprocess.run(
            [ft, "-quiet", "-lg", "-nosupport"],
            stdin=open(tmpfa),
            capture_output=True, text=True, timeout=120)
        newick_str = result.stdout.strip()
        if not newick_str:
            raise RuntimeError(
                f"FastTree produced no output. stderr: {result.stderr[:500]}")
    finally:
        os.unlink(tmpfa)

    # Step 2: Parse tree and find target
    tree = parse_newick(newick_str)
    counter = 0
    for node in tree.preorder():
        if node.name is None or node.name == "":
            node.name = f"_int_{counter}"
            counter += 1

    target_leaf = None
    for node in tree.preorder():
        if node.is_leaf and node.name == target_name:
            target_leaf = node
            break
    if target_leaf is None:
        raise ValueError(f"Target '{target_name}' not found in tree")

    parent = target_leaf.parent
    if parent is None:
        raise ValueError(f"Target '{target_name}' is the root")

    # Step 3: Re-root at parent and remove target
    rerooted = _reroot_at_node(tree, parent)
    rerooted.children = [c for c in rerooted.children
                         if not (c.is_leaf and c.name == target_name)]

    # Step 4: Build leaf observations (gaps = 20, not missing)
    L = len(next(iter(msa_strings.values())))
    leaf_seqs_21 = {}
    for name, seq in msa_strings.items():
        if name == target_name:
            continue
        arr = np.full(L, 20, dtype=np.int32)
        for col, ch in enumerate(seq):
            if ch in "-.~":
                arr[col] = 20
            else:
                idx = _AA_TO_INT.get(ch.upper(), -1)
                if 0 <= idx < 20:
                    arr[col] = idx
                else:
                    arr[col] = 20
        leaf_seqs_21[name] = arr

    # Step 5: Felsenstein pruning with 21-state model
    root_seq, posteriors = reconstruct_root_gap(
        rerooted, leaf_seqs_21, jnp.array(Q21), jnp.array(pi21))

    # Step 6: Extract predicted residues (skip gap positions)
    pred_list = []
    for col in range(L):
        char = root_seq[col]
        if char < 20:
            pred_list.append(char)
    pred_seq = np.array(pred_list, dtype=np.int32)

    elapsed = time.time() - t0
    return pred_seq, np.array(posteriors), elapsed


def reconstruct_fels21_at_node(tree_root, msa_int21, Q21, pi21):
    """Felsenstein reconstruction at root using 21-state model.

    Simpler interface for when tree and MSA are already prepared.

    Args:
        tree_root: TreeNode (root of tree, target already removed)
        msa_int21: dict {name: int32 array} with gaps as 20
        Q21: (21, 21) rate matrix
        pi21: (21,) equilibrium frequencies

    Returns:
        pred_seq: integer array of predicted residues (gap-stripped)
        posteriors: (L, 21) posteriors at root
    """
    root_seq, posteriors = reconstruct_root_gap(
        tree_root, msa_int21,
        jnp.array(Q21), jnp.array(pi21))

    pred_list = []
    for col in range(len(root_seq)):
        char = root_seq[col]
        if char < 20:
            pred_list.append(char)
    pred_seq = np.array(pred_list, dtype=np.int32)

    return pred_seq, posteriors
