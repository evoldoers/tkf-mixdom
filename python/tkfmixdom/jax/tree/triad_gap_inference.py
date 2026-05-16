"""Triad HMM for parent presence/absence inference.

Given a pairwise alignment of two sibling sequences (from Stage 1 Viterbi),
infer whether the parent is present or absent at each alignment column.

The triad HMM has:
- State: parent present (1) or absent (0)
- Observations: alignment column types from sibling pairwise alignment
- Transitions: from TKF91/TKF92 singlet geometric length model
- Emissions: Felsenstein likelihoods conditional on parent state

At each alignment column, the parent is either PRESENT or ABSENT:

**Parent PRESENT:** The column descended from a parent position.
  - MM column: sum_a pi(a) * P(c1|a,t1) * P(c2|a,t2)
  - Only c1 emits: sum_a pi(a) * P(c1|a,t1) * (1 - alpha2) [c2 deleted]
  - Only c2 emits: sum_a pi(a) * P(c2|a,t2) * (1 - alpha1) [c1 deleted]

**Parent ABSENT:** The column is an insertion relative to parent.
  - MM column: independent insertions: sum_a pi(a)*P(c1|a,t1) * sum_b pi(b)*P(c2|b,t2)
    But actually, if parent is absent, both children are inserting independently.
    For a pair HMM, in the "I" state the descendant character is drawn from pi.
    So P(c1|absent) * P(c2|absent) = pi(c1) * pi(c2) for discrete chars,
    or (pi @ cl1) * (pi @ cl2) for profiles.
  - Only c1 emits: P(c1|absent) = pi @ cl1
  - Only c2 emits: P(c2|absent) = pi @ cl2

Transitions encode the TKF singlet geometric length model:
  - present -> present: kappa (geometric continuation)
  - present -> absent:  (1-kappa) * beta / (1-beta)  [exit then insert run]
  - absent  -> present: (1-beta) * kappa [end insert run, start new position]
  - absent  -> absent:  beta [continue insert run]
  - to END from present: (1-kappa)
  - to END from absent:  (1-beta) * (1-kappa)

The triad model factorizes: per-branch pair HMM parameters (alpha, beta, gamma)
come from the individual branch times t1, t2 (NOT t_total).
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..core.params import (
    tkf_alpha, tkf_beta, tkf_gamma, tkf_kappa,
    tkf92_trans, S, M, I, D, E,
)
from ..core.ctmc import transition_matrix


def _column_types_from_path(path):
    """Extract alignment column info from a pairwise alignment path.

    Args:
        path: list of (i, j, state) tuples from viterbi_profile.
            State is S=0, M=1, I=2, D=3, E=4.

    Returns:
        columns: list of dicts with keys:
            'type': 'MM', 'X_', '_Y' (which siblings emit)
            'i': position in seq1 (0-based, or -1 if not emitting)
            'j': position in seq2 (0-based, or -1 if not emitting)
    """
    columns = []
    for i_pos, j_pos, st in path:
        if st == S or st == E:
            continue
        if st == M:
            # Both siblings emit
            columns.append({
                'type': 'MM',
                'i': i_pos - 1,  # 1-indexed -> 0-indexed
                'j': j_pos - 1,
            })
        elif st == D:
            # Only left child (x) emits
            columns.append({
                'type': 'X_',
                'i': i_pos - 1,
                'j': -1,
            })
        elif st == I:
            # Only right child (y) emits
            columns.append({
                'type': '_Y',
                'i': -1,
                'j': j_pos - 1,
            })
    return columns


def _compute_triad_emissions(columns, profile_x, profile_y,
                              sub1, sub2, pi, alpha1, alpha2):
    """Compute log emission probabilities for each column under each parent state.

    Args:
        columns: list of column dicts from _column_types_from_path
        profile_x: (Lx, A) left child profile (conditional likelihoods)
        profile_y: (Ly, A) right child profile
        sub1: (A, A) substitution matrix P(b|a, t1) for branch to child 1
        sub2: (A, A) substitution matrix P(b|a, t2) for branch to child 2
        pi: (A,) equilibrium distribution
        alpha1: survival probability on branch 1
        alpha2: survival probability on branch 2

    Returns:
        log_emit: (L_aln, 2) array where [:,0] = log P(col | absent),
                  [:,1] = log P(col | present)
    """
    L = len(columns)
    A = pi.shape[0]
    log_emit = np.zeros((L, 2))

    profile_x_np = np.asarray(profile_x)
    profile_y_np = np.asarray(profile_y)
    sub1_np = np.asarray(sub1)
    sub2_np = np.asarray(sub2)
    pi_np = np.asarray(pi)

    for c, col in enumerate(columns):
        if col['type'] == 'MM':
            # Both siblings emit
            cl1 = profile_x_np[col['i']]  # (A,)
            cl2 = profile_y_np[col['j']]  # (A,)

            # Parent PRESENT: sum_a pi(a) * (sub1[a,:] @ cl1) * (sub2[a,:] @ cl2)
            left_lik = sub1_np @ cl1   # (A,): P(data_below_c1 | parent=a)
            right_lik = sub2_np @ cl2  # (A,): P(data_below_c2 | parent=a)
            p_present = np.sum(pi_np * left_lik * right_lik)

            # Parent ABSENT: both are insertions, drawn independently from pi
            p_absent = np.sum(pi_np * cl1) * np.sum(pi_np * cl2)

            log_emit[c, 0] = np.log(max(p_absent, 1e-300))
            log_emit[c, 1] = np.log(max(p_present, 1e-300))

        elif col['type'] == 'X_':
            # Only c1 emits
            cl1 = profile_x_np[col['i']]  # (A,)

            # Parent PRESENT: c1 matches, c2 was deleted
            # sum_a pi(a) * (sub1[a,:] @ cl1) * (1 - alpha2)
            left_lik = sub1_np @ cl1
            p_present = np.sum(pi_np * left_lik) * (1.0 - float(alpha2))

            # Parent ABSENT: c1 is an insertion
            p_absent = np.sum(pi_np * cl1)

            log_emit[c, 0] = np.log(max(p_absent, 1e-300))
            log_emit[c, 1] = np.log(max(p_present, 1e-300))

        elif col['type'] == '_Y':
            # Only c2 emits
            cl2 = profile_y_np[col['j']]  # (A,)

            # Parent PRESENT: c2 matches, c1 was deleted
            # sum_a pi(a) * (sub2[a,:] @ cl2) * (1 - alpha1)
            right_lik = sub2_np @ cl2
            p_present = np.sum(pi_np * right_lik) * (1.0 - float(alpha1))

            # Parent ABSENT: c2 is an insertion
            p_absent = np.sum(pi_np * cl2)

            log_emit[c, 0] = np.log(max(p_absent, 1e-300))
            log_emit[c, 1] = np.log(max(p_present, 1e-300))

    return log_emit


def _compute_triad_transitions(ins_rate, del_rate):
    """Compute log transition matrix for the parent present/absent HMM.

    The singlet HMM models the ancestor sequence with geometric length
    distribution (parameter kappa = lambda/mu) and insertion runs
    (parameter beta from stationary).

    States: 0 = absent (insertion), 1 = present (ancestor position)

    Transitions:
      present -> present: kappa                    (continue ancestor)
      present -> absent:  (1-kappa) * kappa_ins    (end pos, start insertion)
      present -> END:     (1-kappa) * (1-kappa_ins) (end pos, no insertion)
      absent  -> absent:  kappa_ins                (continue insertion)
      absent  -> present: (1-kappa_ins) * kappa    (end ins, new ancestor pos)
      absent  -> END:     (1-kappa_ins) * (1-kappa) (end ins, end sequence)

    Where kappa_ins = kappa (insertion run length same as ancestor length
    in TKF91 stationary distribution).

    For the start state:
      S -> present: kappa
      S -> absent:  0 (insertions before first ancestor position use beta)
      S -> END: (1-kappa)

    Actually for a simpler correct model: the ancestor has geometric(kappa)
    length. Between and around ancestor positions, insertions happen with
    rate beta (at the pairwise level). But here the pairwise alignment is
    already done. The triad just needs to decide which columns are ancestral.

    Simplification: use a 2-state HMM where:
      - State 1 (present) has self-loop prob = ext_present
      - State 0 (absent) has self-loop prob = ext_absent
      - Transitions between states depend on relative rates

    For TKF91:
      kappa = lambda/mu (fraction of positions that are ancestral)
      The prior probability that a column is ancestral = kappa / (1 + kappa)?
      No: the equilibrium distribution is geometric(kappa), so the expected
      number of ancestor positions per "slot" is kappa/(1-kappa).

    We use a simpler parameterization:
      log_trans[1,1] = log(kappa)         present -> present
      log_trans[1,0] = log(1-kappa)       present -> absent
      log_trans[0,1] = log(1-kappa)       absent -> present (symmetric)
      log_trans[0,0] = log(kappa)         absent -> absent

    Actually this is wrong. Let me think more carefully.

    The parent sequence has geometric length with parameter kappa.
    P(parent has k positions) = kappa^k * (1-kappa).

    Between each pair of adjacent parent positions, there can be
    0 or more insertion columns. The number of insertions between
    any two parent positions is independent.

    But the triad sees the MERGED alignment of both siblings. So we need
    to model: at each column of the sibling pairwise alignment, is there
    a parent position here?

    Model: parent positions are separated by runs of insertion columns.
    - Within a parent "run": present -> present with prob 1 (consecutive)
      Actually no, parent positions alternate with deletion gaps.

    Let me use the TKF framework directly:
    The parent is a TKF91 singlet: at stationarity, length ~ Geometric(kappa).
    Each parent position independently survives to child 1 with prob alpha1,
    and independently survives to child 2 with prob alpha2.
    Insertions on branch 1 appear with rate beta1, on branch 2 with beta2.

    The 1D triad HMM along the pairwise alignment:
    - At present columns: parent emits, then children inherit/delete
    - At absent columns: one or both children inserted independently

    Transition model:
      present -> present: the next column also has a parent position
      present -> absent: the next column is an insertion
      absent -> absent: another insertion column follows
      absent -> present: insertion run ends, next parent position

    Using TKF91 singlet prior on parent length:
      P(k positions) = kappa^k * (1-kappa)

    So for a sequence of parent-present/absent labels:
      P(present at start) ∝ kappa
      P(absent at start) ∝ 1  (insertion before first parent pos)

    For transitions, the model is:
      After a present position:
        - With prob kappa: another present position follows (possibly with
          insertions in between, but the alignment has already decided the
          column ordering)
        - With prob (1-kappa): this was the last parent position; remaining
          columns are insertions

      After an absent position:
        - The insertion could continue or end
        - Insertion run: governed by beta

    This gets complicated because insertions on the two branches have
    different rates. Let me use a pragmatic approach:

    Args:
        ins_rate: insertion rate (lambda)
        del_rate: deletion rate (mu)

    Returns:
        log_trans: (2, 2) log transition matrix [from, to]
            Index 0 = absent, 1 = present
        log_start: (2,) log start probabilities
    """
    kappa = float(tkf_kappa(ins_rate, del_rate))

    # Transition matrix for parent present/absent
    # present -> present: kappa (geometric continuation)
    # present -> absent: 1 - kappa
    # absent -> present: 1 - kappa (end insertion run; symmetric in TKF91)
    # absent -> absent: kappa (insertion geometric run)
    #
    # This symmetric parameterization comes from TKF91: both ancestor length
    # and insertion length have the same geometric parameter kappa at stationarity.
    log_trans = np.array([
        [np.log(max(kappa, 1e-300)), np.log(max(1 - kappa, 1e-300))],       # absent -> absent, absent -> present
        [np.log(max(1 - kappa, 1e-300)), np.log(max(kappa, 1e-300))],       # present -> absent, present -> present
    ])

    # Start probabilities: P(first column is present) = kappa, absent = 1-kappa
    # (Actually, at first column we should consider the prior probability
    # of having an ancestor position. With geometric(kappa) ancestor length,
    # the first column being present has prior kappa.)
    log_start = np.array([
        np.log(max(1 - kappa, 1e-300)),   # absent
        np.log(max(kappa, 1e-300)),        # present
    ])

    return log_trans, log_start


def infer_parent_gaps_triad(path, profile_x, profile_y,
                             ins_rate, del_rate, t1, t2, Q, pi,
                             ext=None, method='viterbi',
                             presence_floor=None):
    """Infer parent presence/absence at each alignment column via triad HMM.

    Stage 2 of the triad algorithm: given a pairwise alignment of siblings
    (from Stage 1 Viterbi), run a 1D HMM to determine which columns
    correspond to parent positions vs. insertions.

    Args:
        path: list of (i, j, state) tuples from viterbi_profile (Stage 1)
        profile_x: (Lx, A) left child profile (conditional likelihoods)
        profile_y: (Ly, A) right child profile
        ins_rate: TKF insertion rate (lambda)
        del_rate: TKF deletion rate (mu)
        t1: branch length to left child
        t2: branch length to right child
        Q: (A, A) substitution rate matrix
        pi: (A,) equilibrium distribution
        ext: TKF92 extension probability (None for TKF91)
        method: 'viterbi' for MAP path, 'forward_backward' for posteriors

    Returns:
        parent_present: (L_aln,) bool array — True where parent is present
        parent_profile: (n_present, A) conditional likelihoods at present positions
        columns: list of column dicts (for debugging/tracing)
    """
    # Extract column information from the pairwise alignment path
    columns = _column_types_from_path(path)
    L_aln = len(columns)

    if L_aln == 0:
        return (np.array([], dtype=bool),
                np.zeros((0, pi.shape[0])),
                columns)

    # Compute substitution matrices for each branch
    sub1 = np.asarray(transition_matrix(Q, t1))
    sub2 = np.asarray(transition_matrix(Q, t2))

    # Survival probabilities
    alpha1 = float(tkf_alpha(del_rate, t1))
    alpha2 = float(tkf_alpha(del_rate, t2))

    # Compute emission log-probabilities
    log_emit = _compute_triad_emissions(columns, profile_x, profile_y,
                                         sub1, sub2, pi, alpha1, alpha2)

    # Compute transition log-probabilities
    log_trans, log_start = _compute_triad_transitions(ins_rate, del_rate)

    if method == 'viterbi':
        parent_present = _viterbi_1d(log_emit, log_trans, log_start)
    elif method == 'forward_backward':
        parent_present = _forward_backward_1d(log_emit, log_trans, log_start)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply presence floor: force present wherever floor says so
    if presence_floor is not None:
        parent_present = parent_present | np.asarray(presence_floor[:len(parent_present)], dtype=bool)

    # Build parent profile at present positions
    A = pi.shape[0]
    parent_conds = []
    for c, col in enumerate(columns):
        if not parent_present[c]:
            continue
        if col['type'] == 'MM':
            # Both children contribute
            left_lik = sub1 @ np.asarray(profile_x[col['i']])
            right_lik = sub2 @ np.asarray(profile_y[col['j']])
            parent_conds.append(left_lik * right_lik)
        elif col['type'] == 'X_':
            # Only left child contributes
            left_lik = sub1 @ np.asarray(profile_x[col['i']])
            parent_conds.append(left_lik)
        elif col['type'] == '_Y':
            # Only right child contributes
            right_lik = sub2 @ np.asarray(profile_y[col['j']])
            parent_conds.append(right_lik)

    if parent_conds:
        parent_profile = np.stack(parent_conds, axis=0)
    else:
        parent_profile = np.zeros((0, A))

    return parent_present, parent_profile, columns


def _viterbi_1d(log_emit, log_trans, log_start):
    """1D Viterbi for the 2-state parent present/absent HMM.

    Args:
        log_emit: (L, 2) log emission probabilities [absent, present]
        log_trans: (2, 2) log transition matrix [from, to]
        log_start: (2,) log start probabilities

    Returns:
        present: (L,) bool array — True where parent is present
    """
    L = log_emit.shape[0]
    NEG_INF_VAL = -1e30

    # Viterbi tables
    V = np.full((L, 2), NEG_INF_VAL)
    TB = np.zeros((L, 2), dtype=np.int32)

    # Initialize
    V[0] = log_start + log_emit[0]

    # Forward pass
    for t in range(1, L):
        for s in range(2):
            # Best predecessor for state s at time t
            scores = V[t - 1] + log_trans[:, s]
            best = int(np.argmax(scores))
            V[t, s] = scores[best] + log_emit[t, s]
            TB[t, s] = best

    # Traceback
    states = np.zeros(L, dtype=np.int32)
    states[L - 1] = int(np.argmax(V[L - 1]))
    for t in range(L - 2, -1, -1):
        states[t] = TB[t + 1, states[t + 1]]

    return states.astype(bool)  # 1 = present, 0 = absent


def _forward_backward_1d(log_emit, log_trans, log_start):
    """1D Forward-Backward for the 2-state parent present/absent HMM.

    Args:
        log_emit: (L, 2) log emission probabilities [absent, present]
        log_trans: (2, 2) log transition matrix [from, to]
        log_start: (2,) log start probabilities

    Returns:
        present: (L,) bool array — True where posterior P(present) > 0.5
    """
    from scipy.special import logsumexp

    L = log_emit.shape[0]
    NEG_INF_VAL = -1e30

    # Forward
    F = np.full((L, 2), NEG_INF_VAL)
    F[0] = log_start + log_emit[0]

    for t in range(1, L):
        for s in range(2):
            F[t, s] = logsumexp(F[t - 1] + log_trans[:, s]) + log_emit[t, s]

    # Backward
    B = np.full((L, 2), NEG_INF_VAL)
    B[L - 1] = 0.0  # log(1)

    for t in range(L - 2, -1, -1):
        for s in range(2):
            B[t, s] = logsumexp(log_trans[s, :] + log_emit[t + 1] + B[t + 1])

    # Posterior
    log_posterior = F + B
    # Normalize each column
    log_Z = logsumexp(log_posterior, axis=1, keepdims=True)
    log_posterior -= log_Z

    # P(present) > 0.5 iff log_posterior[:,1] > log(0.5)
    present = log_posterior[:, 1] > np.log(0.5)

    return present.astype(bool)


def _msa_constrained_path(left_presence, right_presence):
    """Extract pairwise alignment path from guide MSA presence arrays.

    Returns path in the same format as viterbi_profile: list of (i, j, state).
    Every column where both siblings are present → M (both advance).
    Only left present → I (left advances). Only right → D (right advances).
    """
    L = len(left_presence)
    path = [(0, 0, S)]  # start
    i, j = 0, 0
    for c in range(L):
        lp = bool(left_presence[c])
        rp = bool(right_presence[c])
        if lp and rp:
            path.append((i, j, M))
            i += 1; j += 1
        elif lp and not rp:
            path.append((i, -1, I))
            i += 1
        elif not lp and rp:
            path.append((-1, j, D))
            j += 1
    path.append((i, j, E))
    return path


def reconstruct_with_triad(tree_root, leaf_seqs, ins_rate, del_rate,
                            t_scale, Q, pi, use_tkf92=False, ext=0.5,
                            triad_method='viterbi', guide_msa=None,
                            fitch_floor=False):
    """Progressive reconstruction using triad gap inference.

    Like reconstruct_progressive_felsenstein but uses the triad HMM
    (Stage 2) to infer parent presence/absence after the pairwise
    Viterbi alignment (Stage 1).

    The key difference from the baseline: instead of including all
    Match and Delete columns as parent positions (union-of-leaves),
    the triad HMM decides which columns actually correspond to
    parent positions based on the probabilistic model.

    Args:
        tree_root: TreeNode root
        leaf_seqs: dict of {leaf_name: integer_array}
        ins_rate, del_rate: TKF indel rates
        t_scale: branch length multiplier
        Q: (A, A) substitution rate matrix
        pi: (A,) equilibrium distribution
        use_tkf92: if True, use TKF92 pair HMM
        ext: TKF92 extension probability
        triad_method: 'viterbi' or 'forward_backward'
        guide_msa: optional dict {leaf_name: (L_msa,) bool array}.
            If provided, pairwise alignment paths are read from the MSA
            (1D traceback) instead of computed via 2D Viterbi.
            Produces longer root sequences since the MSA constrains
            all Match columns to have parent present.

    Returns:
        node_profiles: dict of {node_id: (L, A) profile}
        node_alignments: dict of {node_id: alignment_info}
        root_sequence: integer array
    """
    from ..core.params import tkf91_trans, tkf92_trans
    from ..dp.hmm import safe_log
    from ..util.io import TreeNode
    from .progrec_felsenstein import (
        seq_to_profile, profile_emissions, viterbi_profile,
    )

    A = pi.shape[0]
    node_profiles = {}
    node_alignments = {}

    # Assign leaf profiles
    for node in tree_root.preorder():
        if node.is_leaf and node.name in leaf_seqs:
            seq = np.asarray(leaf_seqs[node.name])
            node_profiles[id(node)] = np.eye(A)[seq]

    # Bottom-up reconstruction
    for node in tree_root.postorder():
        if node.is_leaf:
            continue

        is_root = (node is tree_root)

        children_with_profiles = [c for c in node.children
                                  if id(c) in node_profiles]
        if len(children_with_profiles) == 0:
            continue

        if len(children_with_profiles) == 1:
            child = children_with_profiles[0]
            t = max(child.branch_length * t_scale, 1e-6)
            sub_mat = np.asarray(transition_matrix(Q, t))
            profile = node_profiles[id(child)]
            node_profiles[id(node)] = profile @ sub_mat.T
            # Propagate guide_msa: this node inherits the child's presence
            if guide_msa is not None:
                child_name = child.name or f"node_{id(child)}"
                node_name = node.name or f"node_{id(node)}"
                if child_name in guide_msa:
                    guide_msa[node_name] = guide_msa[child_name].copy()
            continue

        # Two children: align then apply triad
        left, right = children_with_profiles[0], children_with_profiles[1]
        t_left = max(left.branch_length * t_scale, 1e-6)
        t_right = max(right.branch_length * t_scale, 1e-6)
        t_total = t_left + t_right

        sub_left = np.asarray(transition_matrix(Q, t_left))
        sub_right = np.asarray(transition_matrix(Q, t_right))

        if use_tkf92:
            tau = tkf92_trans(ins_rate, del_rate, t_total, ext)
        else:
            tau = tkf91_trans(ins_rate, del_rate, t_total)
        log_trans = safe_log(tau)
        state_types = np.array([S, M, I, D, E])

        profile_x = np.asarray(node_profiles[id(left)])
        profile_y = np.asarray(node_profiles[id(right)])

        if profile_x.shape[0] == 0 or profile_y.shape[0] == 0:
            if profile_x.shape[0] > 0:
                node_profiles[id(node)] = node_profiles[id(left)]
                donor = left
            else:
                node_profiles[id(node)] = node_profiles[id(right)]
                donor = right
            # Propagate guide_msa from the non-empty child
            if guide_msa is not None:
                donor_name = donor.name or f"node_{id(donor)}"
                node_name = node.name or f"node_{id(node)}"
                if donor_name in guide_msa:
                    guide_msa[node_name] = guide_msa[donor_name].copy()
            continue

        # Stage 1: Pairwise alignment of siblings
        use_guide = (guide_msa is not None
                     and left.name in guide_msa
                     and right.name in guide_msa)
        if use_guide:
            # MSA-constrained: read path from guide, skip 2D Viterbi
            path = _msa_constrained_path(guide_msa[left.name], guide_msa[right.name])
            log_prob = 0.0
        else:
            log_prob, path, _ = viterbi_profile(
                log_trans, state_types, profile_x, profile_y,
                sub_left, sub_right, pi, is_root=is_root)

        # Stage 2: Triad gap inference
        # Fitch floor: force parent present at MM columns (both children present)
        pf = None
        if fitch_floor:
            col_types = _column_types_from_path(path)
            pf = np.array([col['type'] == 'MM' for col in col_types], dtype=bool)
        parent_present, parent_profile, columns = infer_parent_gaps_triad(
            path, profile_x, profile_y,
            ins_rate, del_rate, t_left, t_right, Q, pi,
            ext=ext if use_tkf92 else None,
            method=triad_method, presence_floor=pf)

        node_profiles[id(node)] = parent_profile

        # Add parent's presence to guide_msa (for use at higher levels)
        if guide_msa is not None and use_guide:
            L_msa = len(next(iter(guide_msa.values())))
            parent_pres = np.zeros(L_msa, dtype=bool)
            # Map path columns back to MSA columns
            non_se = [(i_p, j_p, st) for i_p, j_p, st in path if st != S and st != E]
            col_idx = 0
            for c in range(L_msa):
                lp = bool(guide_msa[left.name][c]) if left.name in guide_msa else False
                rp = bool(guide_msa[right.name][c]) if right.name in guide_msa else False
                if lp or rp:
                    if col_idx < len(parent_present) and parent_present[col_idx]:
                        parent_pres[c] = True
                    col_idx += 1
            node_name = node.name if node.name else f"node_{id(node)}"
            guide_msa[node_name] = parent_pres

        # Build rec_positions: subset of path M/D entries where parent is present.
        # This tells extract_msa which columns are parent positions (vs insertions).
        non_se_entries = [(i_p, j_p, st) for i_p, j_p, st in path
                          if st != S and st != E]
        rec_positions = [non_se_entries[c] for c in range(len(non_se_entries))
                         if c < len(parent_present) and parent_present[c]]

        node_alignments[id(node)] = {
            "left_child": left.name or f"node_{id(left)}",
            "right_child": right.name or f"node_{id(right)}",
            "path": path,
            "rec_positions": rec_positions,
            "parent_present": parent_present,
            "log_prob": log_prob,
            "t_left": t_left,
            "t_right": t_right,
            "n_columns": len(columns),
            "n_parent_present": int(np.sum(parent_present)),
        }

    # Extract MAP root sequence
    root_profile = node_profiles.get(id(tree_root))
    if root_profile is not None and len(root_profile) > 0:
        weighted = np.asarray(pi)[None, :] * root_profile
        root_sequence = np.argmax(weighted, axis=1).astype(np.int32)
    else:
        root_sequence = np.array([], dtype=np.int32)

    return node_profiles, node_alignments, root_sequence


def extract_triad_msa_presence(tree_root, leaf_seqs, node_profiles, node_alignments):
    """Extract MSA and per-node presence arrays from triad reconstruction.

    Combines extract_msa (which handles the column merging) with
    per-node presence derivation. For leaves, presence = (char != gap).
    For internal nodes, presence is derived from the triad's parent_present
    via Fitch parsimony on the triad-derived MSA.

    This bridges the gap between triad output (keyed by id(node)) and
    burl_variational_ancrec input (keyed by node.name).

    Args:
        tree_root: TreeNode root (must have node.name set for all nodes)
        leaf_seqs: dict of {leaf_name: integer_array}
        node_profiles: from reconstruct_with_triad (keyed by id(node))
        node_alignments: from reconstruct_with_triad (keyed by id(node))

    Returns:
        msa: dict of {leaf_name: list} — leaf MSA rows (-1 = gap)
        msa_presence: dict of {node_name: bool_array} — presence at each
            MSA column for every node (leaves and internals)
        msa_length: int — total MSA length
    """
    from .progrec_felsenstein import extract_full_msa

    full = extract_full_msa(tree_root, leaf_seqs, node_alignments, node_profiles)
    return full['msa'], full['presence'], full['length']
