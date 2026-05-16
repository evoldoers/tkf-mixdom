"""1D Triad gap inference: MSA-column-indexed progressive reconstruction.

Unlike the 2D triad (triad_gap_inference.py) which builds its own pairwise
alignments via Viterbi, the 1D triad operates directly on MSA columns:
- Pairwise alignment is determined by the seed MSA column structure
- The triad HMM decides parent presence at each column
- A Fitch floor can guarantee that Fitch-present columns are always present
- Profiles are indexed by MSA column, avoiding coordinate mismatches

The Fitch floor ensures consistency: at columns where both child subtrees
have at least one leaf (Fitch-present), the parent is forced present.
The triad only decides at columns where Fitch would say absent.
"""

import numpy as np
from ..core.bdi import tkf_alpha, tkf_kappa
from ..core.ctmc import transition_matrix
from ..core.params import tkf92_trans, tkf91_trans, S, M, I, D, E


def reconstruct_1d_triad(tree_root, leaf_presence, leaf_profiles,
                          ins_rate, del_rate, Q, pi,
                          use_tkf92=True, ext=0.5,
                          triad_method='viterbi',
                          fitch_floor=True):
    """Progressive reconstruction using 1D triad on MSA columns.

    All profiles and presence arrays are indexed by MSA column.

    Args:
        tree_root: TreeNode (names assigned to all internal nodes)
        leaf_presence: {leaf_name: bool array of length L_msa}
        leaf_profiles: {leaf_name: (L_chars, A) conditional likelihoods}
            where L_chars = sum(leaf_presence[name])
        ins_rate, del_rate: TKF insertion/deletion rates
        Q: (A, A) rate matrix
        pi: (A,) equilibrium distribution
        use_tkf92: use TKF92 (with extension) or TKF91
        ext: TKF92 fragment extension probability
        triad_method: 'viterbi' or 'forward_backward'
        fitch_floor: if True, force parent present wherever Fitch says so

    Returns:
        node_presence: {node_name: bool array of length L_msa} for all nodes
        node_profiles: {node_name: (n_present, A) conditional likelihoods}
        root_sequence: int array — MAP root sequence at present positions
    """
    from .triad_gap_inference import (
        _column_types_from_path,
        _compute_triad_emissions,
        _compute_triad_transitions,
        _viterbi_1d,
        _forward_backward_1d,
        infer_parent_gaps_triad,
    )

    A = pi.shape[0]
    pi = np.asarray(pi)
    L_msa = len(next(iter(leaf_presence.values())))

    # Compute Fitch presence for all nodes (for the floor)
    fitch_presence = {}
    for node in tree_root.postorder():
        if node.is_leaf:
            fitch_presence[node.name] = np.array(leaf_presence.get(node.name,
                                                  np.zeros(L_msa, dtype=bool)), dtype=bool)
        else:
            ch_pres = [fitch_presence[c.name] for c in node.children if c.name in fitch_presence]
            if len(ch_pres) >= 2:
                fitch_presence[node.name] = ch_pres[0] & ch_pres[1]
            elif len(ch_pres) == 1:
                fitch_presence[node.name] = ch_pres[0].copy()
            else:
                fitch_presence[node.name] = np.zeros(L_msa, dtype=bool)
    # Preorder: if parent present, child present
    for node in tree_root.preorder():
        if node.is_root:
            continue
        fitch_presence[node.name] = fitch_presence[node.name] | fitch_presence[node.parent.name]

    # Build full MSA-indexed profiles for leaves: (L_msa, A) with zeros at absent columns
    full_profiles = {}
    for name, pres in leaf_presence.items():
        prof = leaf_profiles[name]
        full = np.zeros((L_msa, A))
        present_idx = np.where(pres)[0]
        full[present_idx[:len(prof)]] = prof[:len(present_idx)]
        full_profiles[name] = full

    # Node results
    node_presence = dict(leaf_presence)  # copy leaf presence
    node_profiles_out = {}

    # Bottom-up reconstruction
    for node in tree_root.postorder():
        if node.is_leaf:
            continue

        children_with_data = [c for c in node.children if c.name in full_profiles]
        if len(children_with_data) == 0:
            full_profiles[node.name] = np.zeros((L_msa, A))
            node_presence[node.name] = np.zeros(L_msa, dtype=bool)
            continue

        if len(children_with_data) == 1:
            # Pass through: inherit child's profile (with substitution)
            child = children_with_data[0]
            t = max(child.branch_length, 1e-6)
            sub = np.asarray(transition_matrix(Q, t))
            full_profiles[node.name] = full_profiles[child.name] @ sub.T
            node_presence[node.name] = node_presence[child.name].copy()
            continue

        left, right = children_with_data[0], children_with_data[1]
        t_left = max(left.branch_length, 1e-6)
        t_right = max(right.branch_length, 1e-6)
        t_total = t_left + t_right

        sub_left = np.asarray(transition_matrix(Q, t_left))
        sub_right = np.asarray(transition_matrix(Q, t_right))

        left_pres = node_presence[left.name]
        right_pres = node_presence[right.name]

        # Build pairwise path from MSA column presence
        # Columns where neither child is present are skipped
        path_cols = []  # (msa_col, type) where type is 'MM', 'X_', '_Y'
        for c in range(L_msa):
            lp = left_pres[c]
            rp = right_pres[c]
            if lp and rp:
                path_cols.append((c, 'MM'))
            elif lp and not rp:
                path_cols.append((c, 'X_'))
            elif not lp and rp:
                path_cols.append((c, '_Y'))
            # neither: skip

        if len(path_cols) == 0:
            full_profiles[node.name] = np.zeros((L_msa, A))
            node_presence[node.name] = np.zeros(L_msa, dtype=bool)
            continue

        # Extract child profiles at path columns
        left_full = full_profiles[left.name]
        right_full = full_profiles[right.name]

        # Build the alignment path in triad format: (i, j, state)
        # i = left profile position, j = right profile position
        # We need to map MSA columns to child profile positions
        left_idx = np.cumsum(left_pres) - 1  # MSA col → left profile pos
        right_idx = np.cumsum(right_pres) - 1

        path = [(0, 0, S)]
        for msa_c, col_type in path_cols:
            if col_type == 'MM':
                path.append((int(left_idx[msa_c]), int(right_idx[msa_c]), M))
            elif col_type == 'X_':
                path.append((int(left_idx[msa_c]), -1, I))
            elif col_type == '_Y':
                path.append((-1, int(right_idx[msa_c]), D))
        li = int(np.sum(left_pres))
        ri = int(np.sum(right_pres))
        path.append((li, ri, E))

        # Extract compact child profiles (only at present positions)
        left_present_idx = np.where(left_pres)[0]
        right_present_idx = np.where(right_pres)[0]
        profile_x = left_full[left_present_idx]
        profile_y = right_full[right_present_idx]

        # Fitch floor for this node: parent present at MM columns
        pf = None
        if fitch_floor:
            pf = np.array([ct == 'MM' for _, ct in path_cols], dtype=bool)

        # Run triad gap inference
        parent_present, parent_profile, columns = infer_parent_gaps_triad(
            path, profile_x, profile_y,
            ins_rate, del_rate, t_left, t_right, Q, pi,
            ext=ext if use_tkf92 else None,
            method=triad_method, presence_floor=pf)

        # Map parent_present back to MSA columns
        parent_pres_msa = np.zeros(L_msa, dtype=bool)
        parent_full = np.zeros((L_msa, A))
        prof_idx = 0
        for k, (msa_c, col_type) in enumerate(path_cols):
            if k < len(parent_present) and parent_present[k]:
                parent_pres_msa[msa_c] = True
                if prof_idx < len(parent_profile):
                    parent_full[msa_c] = parent_profile[prof_idx]
                    prof_idx += 1

        node_presence[node.name] = parent_pres_msa
        full_profiles[node.name] = parent_full

    # Extract compact profiles for output
    for node in tree_root.preorder():
        if node.is_leaf:
            node_profiles_out[node.name] = leaf_profiles[node.name]
        else:
            pres = node_presence.get(node.name, np.zeros(L_msa, dtype=bool))
            fp = full_profiles.get(node.name, np.zeros((L_msa, A)))
            node_profiles_out[node.name] = fp[pres]

    # Root MAP sequence
    root_pres = node_presence.get(tree_root.name, np.zeros(L_msa, dtype=bool))
    root_prof = full_profiles.get(tree_root.name, np.zeros((L_msa, A)))
    root_vals = root_prof[root_pres]
    if len(root_vals) > 0:
        weighted = pi[None, :] * root_vals
        root_sequence = np.argmax(weighted, axis=1).astype(np.int32)
    else:
        root_sequence = np.array([], dtype=np.int32)

    return node_presence, node_profiles_out, root_sequence
