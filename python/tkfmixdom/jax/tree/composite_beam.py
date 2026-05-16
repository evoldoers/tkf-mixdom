"""Composite likelihood beam search for ancestral sequence reconstruction.

Given K observed descendant sequences D_1,...,D_K on a star phylogeny
with unknown root A and branch lengths T_1,...,T_K, find:

    A* = argmax_A  log Q(A)

where:
    Q(A) = P(A | theta) * prod_k P(D_k | A, T_k, theta)
         = Singlet(A)^{1-K} * prod_k PairForward(A, D_k | T_k, theta)

The (1-K) weighting on the singlet corrects for the K copies of
the ancestor prior embedded in the pair HMM forward probabilities.

The beam iterates column-by-column through the ancestor A. At each
position, for each candidate residue a:
  - Advance each pair HMM forward table by one ancestor row
  - Advance the singlet HMM forward by one step
  - Score = (1-K) * singletForward(A[0..j]) + sum_k terminalScore_k(j)

Uses the full MixDom pair HMM with per-domain substitution matrices.
"""

import numpy as np
from scipy.special import logsumexp

from ..dp.hmm import M, I, D, S, E

NEG_INF = -1e30


# ============================================================
# Reference (slow) implementations — kept for testing
# ============================================================

def _build_emit_row_ref(anc_char, desc_seq, state_types, dom_idx,
                         log_subs, log_pis, L_desc):
    """Reference: build emission row with Python loops."""
    ns = len(state_types)
    emit_row = np.zeros((L_desc + 1, ns))
    for s in range(ns):
        stype = state_types[s]
        d = dom_idx[s]
        if stype == M:
            for j in range(1, L_desc + 1):
                emit_row[j, s] = log_pis[d, anc_char] + log_subs[d, anc_char, desc_seq[j - 1]]
        elif stype == I:
            for j in range(1, L_desc + 1):
                emit_row[j, s] = log_pis[d, desc_seq[j - 1]]
        elif stype == D:
            emit_row[:, s] = log_pis[d, anc_char]
    return emit_row


def _advance_ancestor_row_ref(fwd_row_prev, emit_row, log_chi, state_types, L_desc):
    """Reference: advance one ancestor row with partial vectorization."""
    ns = len(state_types)
    fwd_row = np.full((L_desc + 1, ns), NEG_INF)
    m_mask = np.where(state_types == M)[0]
    i_mask = np.where(state_types == I)[0]
    d_mask = np.where(state_types == D)[0]

    if len(d_mask) > 0:
        chi_D = log_chi[:, d_mask]
        vals = fwd_row_prev[:, :, None] + chi_D[None, :, :]
        fwd_row[:, d_mask] = logsumexp(vals, axis=1) + emit_row[:, d_mask]

    if len(m_mask) > 0 and L_desc > 0:
        chi_M = log_chi[:, m_mask]
        vals = fwd_row_prev[:L_desc, :, None] + chi_M[None, :, :]
        m_scores = logsumexp(vals, axis=1) + emit_row[1:, m_mask]
        fwd_row[1:, m_mask] = np.logaddexp(fwd_row[1:, m_mask], m_scores)

    if len(i_mask) > 0:
        chi_I = log_chi[:, i_mask]
        for j in range(1, L_desc + 1):
            vals = fwd_row[j - 1, :, None] + chi_I
            i_scores = logsumexp(vals, axis=0) + emit_row[j, i_mask]
            fwd_row[j, i_mask] = np.logaddexp(fwd_row[j, i_mask], i_scores)

    return fwd_row


# ============================================================
# Vectorized implementations
# ============================================================

def _precompute_all_emit_rows(desc_seq, state_types, dom_idx,
                               log_subs, log_pis, L_desc, A=20,
                               class_log_subs=None, class_log_pis=None,
                               class_dist=None, frag_idx=None):
    """Precompute emission rows for ALL ancestor characters at once.

    Args:
        desc_seq: (L_desc,) descendant integer sequence.
        state_types: (ns,) M/I/D/S/E codes.
        dom_idx: (ns,) per-state domain index.
        log_subs: (n_dom, A, A) class-marginal P(t) (used when no class data).
        log_pis: (n_dom, A) class-marginal equilibrium.
        L_desc: descendant length.
        A: alphabet size (default 20 for proteins).
        class_log_subs: optional (C, A, A) per-class P(t). When provided
            together with class_log_pis and class_dist, emissions become
            log sum_c classdist[dom, frag, c] * class_pi[c, anc] *
            class_P[c, anc, desc] for M states (and the analogous form
            for I/D). class_dist is in linear space, expected to sum to
            1 over its last axis.
        class_log_pis: optional (C, A) per-class equilibrium.
        class_dist: optional (D, F, C) per-(dom, frag) class distribution.
        frag_idx: optional (ns,) per-state fragment index. Required when
            class data is provided.

    Returns:
        emit_all: (A, L_desc+1, ns) — emit_all[a] is the emission row
                  for ancestor character a.
    """
    use_class = (class_log_subs is not None and class_log_pis is not None
                 and class_dist is not None and frag_idx is not None)

    ns = len(state_types)
    emit_all = np.zeros((A, L_desc + 1, ns))

    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    desc_chars = np.asarray(desc_seq)

    if not use_class:
        # Pad log_subs and log_pis to size A+1 for wildcard (index 20) support
        A_orig = log_subs.shape[1]
        if A_orig <= A:
            log_subs = np.pad(log_subs,
                              ((0, 0), (0, A + 1 - A_orig), (0, A + 1 - A_orig)),
                              constant_values=0.0)
            log_pis = np.pad(log_pis, ((0, 0), (0, A + 1 - A_orig)),
                             constant_values=0.0)

        # Per-state domain lookup
        state_dom = dom_idx  # (ns,)

        # Match: emit[a, j, s] = log_pi[dom[s], a] + log_sub[dom[s], a, desc[j-1]]
        if is_M.any():
            m_idx = np.where(is_M)[0]
            m_doms = state_dom[m_idx]
            m_pi = log_pis[m_doms, :A].T  # (A, n_M)
            m_sub = log_subs[m_doms][:, :A, :][:, :, desc_chars]  # (n_M, A, L_desc)
            emit_match = m_pi[:, None, :] + m_sub.transpose(1, 2, 0)
            emit_all[:, 1:, m_idx] = emit_match

        # Insert: emit[a, j, s] = log_pi[dom[s], desc[j-1]] — independent of a
        if is_I.any():
            i_idx = np.where(is_I)[0]
            i_doms = state_dom[i_idx]
            i_emit = log_pis[i_doms][:, desc_chars].T
            emit_all[:, 1:, i_idx] = i_emit[None, :, :]

        # Delete: emit[a, j, s] = log_pi[dom[s], a] — independent of j
        if is_D.any():
            d_idx = np.where(is_D)[0]
            d_doms = state_dom[d_idx]
            d_pi = log_pis[d_doms, :A].T
            emit_all[:, :, d_idx] = d_pi[:, None, :]

        return emit_all

    # ── Class-aware path ─────────────────────────────────────────────────
    # class_log_subs: (C, A, A) — pad to A+1 for wildcard support.
    # class_log_pis:  (C, A) — pad to A+1.
    # class_dist:     (D, F, C) — linear, normalised over last axis.
    A_orig_c = class_log_subs.shape[1]
    if A_orig_c <= A:
        class_log_subs = np.pad(
            class_log_subs,
            ((0, 0), (0, A + 1 - A_orig_c), (0, A + 1 - A_orig_c)),
            constant_values=0.0)
        class_log_pis = np.pad(
            class_log_pis,
            ((0, 0), (0, A + 1 - A_orig_c)),
            constant_values=0.0)

    # log classdist of shape (D, F, C). Use NEG_INF for zero entries.
    log_class_dist = np.where(
        class_dist > 0, np.log(np.maximum(class_dist, 1e-300)), NEG_INF)

    # Per-state (dom, frag) lookups → (ns,)
    state_dom = np.asarray(dom_idx)
    state_frag = np.asarray(frag_idx)

    # log w[s, c] = log classdist[dom[s], frag[s], c] — shape (ns, C)
    state_log_w = log_class_dist[state_dom, state_frag, :]  # (ns, C)

    # Helper: logsumexp over class axis with broadcasting.
    def _lse_over_c(arr, axis):
        return logsumexp(arr, axis=axis)

    # Match: emit[a, j+1, s] = logsumexp_c(log w[s,c] + log class_pi[c,a]
    #                          + log class_P[c, a, desc[j]])
    if is_M.any():
        m_idx = np.where(is_M)[0]
        # (n_M, C)
        log_w_m = state_log_w[m_idx]
        # Per-class pi at ancestor: (C, A) → (A, C)
        cpi = class_log_pis[:, :A].T  # (A, C)
        # Per-class sub at (anc, desc): (C, A, L_desc)
        csub_full = class_log_subs[:, :A, :][:, :, desc_chars]  # (C, A, L_desc)
        # Combine: (n_M, A, L_desc, C) — too big to materialise.
        # Iterate over m states (typically ~3-9 of them) to keep memory in check.
        for ii, s_m in enumerate(m_idx):
            # log_terms: (A, L_desc, C)
            terms = (log_w_m[ii][None, None, :]   # (1,1,C)
                     + cpi[:, None, :]             # (A,1,C)
                     + csub_full.transpose(1, 2, 0))  # (A, L_desc, C)
            # logsumexp over C → (A, L_desc)
            emit_all[:, 1:, s_m] = _lse_over_c(terms, axis=2)

    # Insert: emit[a, j+1, s] = logsumexp_c(log w[s,c] + log class_pi[c, desc[j]])
    if is_I.any():
        i_idx = np.where(is_I)[0]
        log_w_i = state_log_w[i_idx]  # (n_I, C)
        cpi_desc = class_log_pis[:, desc_chars]  # (C, L_desc)
        for ii, s_i in enumerate(i_idx):
            # (L_desc, C) = (1, C) + (L_desc, C)
            terms = log_w_i[ii][None, :] + cpi_desc.T
            row = _lse_over_c(terms, axis=1)  # (L_desc,)
            emit_all[:, 1:, s_i] = row[None, :]

    # Delete: emit[a, j, s] = logsumexp_c(log w[s,c] + log class_pi[c, a])
    if is_D.any():
        d_idx = np.where(is_D)[0]
        log_w_d = state_log_w[d_idx]  # (n_D, C)
        cpi_a = class_log_pis[:, :A].T  # (A, C)
        for ii, s_d in enumerate(d_idx):
            terms = log_w_d[ii][None, :] + cpi_a  # (A, C)
            row = _lse_over_c(terms, axis=1)      # (A,)
            emit_all[:, :, s_d] = row[:, None]

    return emit_all


def _advance_ancestor_row(fwd_row_prev, emit_row, log_chi, state_types, L_desc):
    """Vectorized ancestor row advance (numpy).

    Same semantics as _advance_ancestor_row_ref but uses batch operations.
    The I-type sweep is still sequential over j (inherent dependency).
    """
    ns = len(state_types)
    fwd_row = np.full((L_desc + 1, ns), NEG_INF)

    m_mask = np.where(state_types == M)[0]
    i_mask = np.where(state_types == I)[0]
    d_mask = np.where(state_types == D)[0]

    # D-type: vectorized over all j
    if len(d_mask) > 0:
        chi_D = log_chi[:, d_mask]
        vals = fwd_row_prev[:, :, None] + chi_D[None, :, :]
        fwd_row[:, d_mask] = logsumexp(vals, axis=1) + emit_row[:, d_mask]

    # M-type: shifted vectorized
    if len(m_mask) > 0 and L_desc > 0:
        chi_M = log_chi[:, m_mask]
        vals = fwd_row_prev[:L_desc, :, None] + chi_M[None, :, :]
        m_scores = logsumexp(vals, axis=1) + emit_row[1:, m_mask]
        fwd_row[1:, m_mask] = np.logaddexp(fwd_row[1:, m_mask], m_scores)

    # I-type: sequential (inherent j→j dependency)
    if len(i_mask) > 0:
        chi_I = log_chi[:, i_mask]
        for j in range(1, L_desc + 1):
            vals = fwd_row[j - 1, :, None] + chi_I
            i_scores = logsumexp(vals, axis=0) + emit_row[j, i_mask]
            fwd_row[j, i_mask] = np.logaddexp(fwd_row[j, i_mask], i_scores)

    return fwd_row


def _terminal_score(fwd_row, log_chi, state_types, L_desc):
    """Score transitioning to End after processing all descendant positions."""
    e_idx = np.where(state_types == E)[0]
    if len(e_idx) == 0:
        return NEG_INF
    vals = fwd_row[L_desc, :] + log_chi[:, e_idx[0]]
    return logsumexp(vals)


# ============================================================
# Main beam search
# ============================================================

def compute_unique_weights(tree, target_name, desc_names):
    """Compute unique path fraction weights for composite likelihood.

    For each descendant, the weight is the fraction of its path to the
    target (root) that is NOT shared with any other descendant. If a
    descendant has total distance d_k to root and shares a common
    ancestor at distance s_k from root with some other descendant,
    its unique share is (d_k - s_k) / d_k.

    Args:
        tree: TreeNode root
        target_name: name of the target node (ancestor to reconstruct)
        desc_names: list of descendant node names

    Returns:
        weights: list of floats, one per descendant
    """
    # Find target node
    target = None
    nodes_by_name = {}
    for node in tree.preorder():
        nodes_by_name[node.name] = node
        if node.name == target_name:
            target = node

    if target is None:
        return [1.0] * len(desc_names)

    # For each descendant, compute path to target and shared prefix
    def path_to_target(node):
        """Return (total_dist, list of (ancestor, dist_from_target))."""
        path = []
        n = node
        d = 0.0
        while n is not None:
            path.append((n.name, d))
            if n.name == target_name:
                return d, path
            d += n.branch_length
            n = n.parent
        return d, path

    desc_paths = {}
    for name in desc_names:
        if name in nodes_by_name:
            total_d, path = path_to_target(nodes_by_name[name])
            desc_paths[name] = (total_d, {n: d for n, d in path})

    weights = []
    for i, name in enumerate(desc_names):
        if name not in desc_paths:
            weights.append(1.0)
            continue

        total_d, path_dict = desc_paths[name]
        if total_d < 1e-10:
            weights.append(1.0)
            continue

        # Find the deepest shared ancestor with ANY other descendant
        max_shared = 0.0
        for j, other_name in enumerate(desc_names):
            if i == j or other_name not in desc_paths:
                continue
            _, other_path = desc_paths[other_name]
            # Shared ancestors = nodes on both paths
            for anc_name in path_dict:
                if anc_name in other_path and anc_name != target_name:
                    # Distance from target to this shared ancestor
                    shared_d = path_dict[anc_name]
                    # The shared portion is total_d - shared_d
                    # (from the descendant up to the shared ancestor)
                    # Actually: shared_d is dist from THIS descendant to ancestor
                    # We want: dist from target to shared ancestor
                    # path_dict[anc_name] = dist from descendant to ancestor
                    # dist from target to shared ancestor = total_d - path_dict[anc_name]
                    shared_from_target = total_d - path_dict[anc_name]
                    max_shared = max(max_shared, shared_from_target)

        # unique fraction = (total_d - max_shared_from_target) / total_d
        # = unique_portion / total_d
        # But max_shared is measured from the descendant side
        # Let me reconsider:
        # total_d = distance from descendant to target
        # shared_from_target = distance from target to the deepest shared node
        # unique = total_d - shared_from_target (the part from shared node to descendant)
        # unique_frac = unique / total_d = (total_d - shared_from_target) / total_d
        unique_frac = max(0.0, (total_d - max_shared) / total_d)
        weights.append(unique_frac)

    return weights


def composite_beam_reconstruct(
    desc_seqs,
    distances,
    log_chi_list,
    state_types_list,
    sub_matrices_list,
    pis_list,
    n_dom,
    n_frag,
    singlet_log_trans,
    singlet_log_pi,
    singlet_log_end,
    beam_width=100,
    max_len=500,
    desc_weights=None,
    fixed_len=None,
    class_sub_matrices_list=None,
    class_pis_list=None,
    class_dist=None,
):
    """Composite likelihood beam search for ancestor reconstruction.

    Maximizes:
        log Q(A) = (1 - sum w_k) * logSinglet(A)
                   + sum_k w_k * logPairForward(A, D_k)

    If fixed_len is set, produces a sequence of exactly that length
    (no early stopping, no length optimization). The beam runs for
    fixed_len positions and returns the best sequence at that length.

    When desc_weights is None, all weights are 1.0 and this reduces to:
        log Q(A) = (1-K) * logSinglet(A) + sum_k logPairForward(A, D_k)

    Use compute_unique_weights() to get weights that downweight
    descendants sharing common ancestry (avoiding double-counting
    shared phylogenetic signal).

    Args:
        desc_seqs: list of K integer arrays (descendant sequences)
        distances: list of K floats (branch lengths)
        log_chi_list: list of K (ns, ns) log transition matrices
        state_types_list: list of K (ns,) state type arrays
        sub_matrices_list: list of K (n_dom, A, A) per-domain P(t) matrices
        pis_list: list of K (n_dom, A) per-domain equilibrium
        n_dom, n_frag: MixDom dimensions
        singlet_log_trans: (A, A) singlet HMM log transition
        singlet_log_pi: (A,) singlet HMM log start distribution
        singlet_log_end: (A,) singlet HMM log end probability
        beam_width: beam size
        max_len: max ancestor length
        desc_weights: optional list of K floats. Weight for each descendant
            in the composite score. Default None = all 1.0 (standard).
            Use compute_unique_weights() for phylogeny-aware weighting.
        class_sub_matrices_list: optional list of K (C, A, A) per-class
            P(t) matrices, one per descendant branch length.
        class_pis_list: optional list of K (C, A) per-class equilibrium
            distributions (typically the same array repeated K times).
        class_dist: optional (D, F, C) per-(domain, fragment) class
            distribution. When provided together with the above two
            args, emissions become full per-class mixtures (MixDom2);
            when omitted (or any of the three is None), the function
            falls back to the class-marginal per-domain emission.

    Returns:
        ancestor_seq: integer array
        log_score: float
    """
    # Strict end-to-end mode: if any class arg is provided, all must be.
    if (class_sub_matrices_list is not None or class_pis_list is not None
            or class_dist is not None):
        if (class_sub_matrices_list is None or class_pis_list is None
                or class_dist is None):
            raise ValueError(
                "composite_beam_reconstruct: per-class plumbing requires all of "
                "class_sub_matrices_list, class_pis_list, and class_dist; "
                "got partial inputs.")
    K = len(desc_seqs)
    A = singlet_log_pi.shape[0]
    desc_seqs = [np.asarray(s, dtype=np.int32) for s in desc_seqs]
    desc_lens = [len(s) for s in desc_seqs]
    singlet_log_trans = np.asarray(singlet_log_trans)
    singlet_log_pi = np.asarray(singlet_log_pi)
    singlet_log_end = np.asarray(singlet_log_end)

    if desc_weights is None:
        desc_weights = [1.0] * K
    singlet_weight = 1.0 - sum(desc_weights)

    use_class = (class_sub_matrices_list is not None)
    cdist_np = (np.asarray(class_dist, dtype=np.float64)
                if use_class else None)

    # Precompute per-descendant data and emission tables
    desc_data = []
    for k in range(K):
        st = np.asarray(state_types_list[k])
        ns = len(st)
        subs = np.asarray(sub_matrices_list[k])
        pis_k = np.asarray(pis_list[k])
        dom_idx = np.zeros(ns, dtype=np.int32)
        frag_idx = np.zeros(ns, dtype=np.int32)
        for s_idx in range(2, ns):
            body_i = s_idx - 2
            dom_idx[s_idx] = body_i // (5 * n_frag)
            within_dom = body_i % (5 * n_frag)
            frag_idx[s_idx] = within_dom % n_frag

        log_subs = np.log(np.maximum(subs, 1e-300))
        log_pis = np.log(np.maximum(pis_k, 1e-300))

        if use_class:
            csub = np.asarray(class_sub_matrices_list[k])  # (C, A, A)
            cpi = np.asarray(class_pis_list[k])            # (C, A)
            class_log_subs = np.log(np.maximum(csub, 1e-300))
            class_log_pis = np.log(np.maximum(cpi, 1e-300))
            emit_all = _precompute_all_emit_rows(
                desc_seqs[k], st, dom_idx, log_subs, log_pis, desc_lens[k], A,
                class_log_subs=class_log_subs,
                class_log_pis=class_log_pis,
                class_dist=cdist_np,
                frag_idx=frag_idx)
        else:
            emit_all = _precompute_all_emit_rows(
                desc_seqs[k], st, dom_idx, log_subs, log_pis, desc_lens[k], A)

        desc_data.append({
            'st': st, 'ns': ns, 'dom_idx': dom_idx,
            'log_chi': np.asarray(log_chi_list[k]),
            'L': desc_lens[k],
            'emit_all': emit_all,  # (A, L+1, ns)
        })

    # Initialize: forward row 0 for each pair HMM
    def _init_row(k):
        dd = desc_data[k]
        row0 = np.full((dd['L'] + 1, dd['ns']), NEG_INF)
        row0[0, 0] = 0.0
        # Sweep I-type at row 0
        i_mask = np.where(dd['st'] == I)[0]
        if len(i_mask) > 0:
            chi_I = dd['log_chi'][:, i_mask]
            for j in range(1, dd['L'] + 1):
                emit_ij = dd['emit_all'][0, j, i_mask]  # use anc_char=0 for I (doesn't matter)
                vals = row0[j - 1, :, None] + chi_I
                row0[j, i_mask] = logsumexp(vals, axis=0) + emit_ij
        return row0

    init_rows = [_init_row(k) for k in range(K)]

    # Beam entries: list of dicts
    beam = [{
        'seq': [],
        'fwd_rows': [r.copy() for r in init_rows],
        'singlet_state': -1,  # -1 = no previous character
        'singlet_fwd': 0.0,
    }]

    best_complete = None
    target_len = fixed_len if fixed_len is not None else max_len

    for anc_pos in range(target_len):
        next_beam = []

        for entry in beam:
            prev_char = entry['singlet_state']
            prev_singlet = entry['singlet_fwd']
            prev_rows = entry['fwd_rows']

            for a in range(A):
                # Singlet forward
                if prev_char < 0:
                    new_singlet = prev_singlet + singlet_log_pi[a]
                else:
                    new_singlet = prev_singlet + singlet_log_trans[prev_char, a]

                # Singlet end score at this position
                singlet_end = new_singlet + singlet_log_end[a]

                # Pair HMM: advance each descendant (weighted)
                pair_term = 0.0
                new_rows = []
                for k in range(K):
                    dd = desc_data[k]
                    emit_row = dd['emit_all'][a]
                    new_row = _advance_ancestor_row(
                        prev_rows[k], emit_row, dd['log_chi'],
                        dd['st'], dd['L'])
                    new_rows.append(new_row)
                    pair_term += desc_weights[k] * _terminal_score(
                        new_row, dd['log_chi'], dd['st'], dd['L'])

                total_score = singlet_weight * singlet_end + pair_term

                next_beam.append({
                    'seq': entry['seq'] + [a],
                    'fwd_rows': new_rows,
                    'singlet_state': a,
                    'singlet_fwd': new_singlet,
                    'score': total_score,
                })

        if not next_beam:
            break

        # Prune
        next_beam.sort(key=lambda e: -e['score'])
        beam = next_beam[:beam_width]

        # Track best
        if best_complete is None or beam[0]['score'] > best_complete['score']:
            best_complete = {
                'seq': beam[0]['seq'][:],
                'score': beam[0]['score'],
            }

        # Early stop (disabled when fixed_len is set)
        if fixed_len is None:
            if anc_pos >= 5 and best_complete is not None:
                if beam[0]['score'] < best_complete['score'] - 20.0:
                    break

    if fixed_len is not None:
        # Return the best beam entry at exactly fixed_len positions
        if beam:
            return np.array(beam[0]['seq'], dtype=np.int32), beam[0]['score']
    elif best_complete is not None:
        return np.array(best_complete['seq'], dtype=np.int32), best_complete['score']

    return np.array([], dtype=np.int32), -np.inf
