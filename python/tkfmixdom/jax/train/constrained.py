"""Alignment-constrained training for TKF91, TKF92, and MixDom.

When a guide alignment (e.g. from Pfam MSA) is available, the pairwise
alignment is fixed and the Pair HMM DP reduces to a 1D problem:

- TKF91/TKF92: The state sequence (M/I/D) is fully observed. Transition
  and emission counts are direct tallies — no Forward-Backward needed.

- MixDom: The alignment column types (M/I/D) are fixed, but the domain
  and fragment assignments are still latent. A 1D Forward-Backward over
  the N-state MixDom HMM is needed, with emissions constrained by the
  observed alignment column type. Uses associative_scan for O(log L).

All functions aggregate sufficient statistics across pairs for CherryML-
style composite likelihood training.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from ..core.params import S, M, I, D, E, N_STATES, tkf91_trans, tkf92_trans
from ..simulate.msa import alignment_to_states
from ..core.ctmc import transition_matrix, holmes_rubin_expected_stats
from ..dp.hmm import forward_backward_1d_associative, NEG_INF


# --- TKF91/TKF92: fully observed counts ---

def observed_counts(states, anc_chars, desc_chars):
    """Tally transition counts and match pairs from an observed alignment.

    When the pairwise alignment is fixed, the state sequence is fully
    determined and counts are literal (not expected).

    Args:
        states: list of state codes (M, I, D)
        anc_chars: list of ancestor character indices (for M, D states)
        desc_chars: list of descendant character indices (for M, I states)

    Returns:
        n_trans: (5, 5) transition count matrix
        match_pairs: list of (anc_char, desc_char, 1.0) triples
    """
    n_trans = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    match_pairs = []

    prev = S
    ai, di = 0, 0
    for state in states:
        n_trans[prev, state] += 1.0

        if state == M:
            match_pairs.append((int(anc_chars[ai]), int(desc_chars[di]), 1.0))
            ai += 1
            di += 1
        elif state == I:
            di += 1
        elif state == D:
            ai += 1

        prev = state

    n_trans[prev, E] += 1.0

    return jnp.array(n_trans), match_pairs


def observed_log_likelihood(log_trans, sub_matrix, pi, states, anc_chars, desc_chars):
    """Compute log P(alignment | params) from observed state sequence.

    Same as msa.pairwise_log_likelihood but returns a float.
    """
    log_sub = jnp.log(sub_matrix + 1e-30)
    log_pi = jnp.log(pi + 1e-30)

    lp = 0.0
    prev = S
    ai, di = 0, 0
    for state in states:
        lp += float(log_trans[prev, state])
        if state == M:
            a, d = int(anc_chars[ai]), int(desc_chars[di])
            lp += float(log_pi[a]) + float(log_sub[a, d])
            ai += 1
            di += 1
        elif state == I:
            d = int(desc_chars[di])
            lp += float(log_pi[d])
            di += 1
        elif state == D:
            a = int(anc_chars[ai])
            lp += float(log_pi[a])
            ai += 1
        prev = state
    lp += float(log_trans[prev, E])
    return lp


def aggregate_observed_counts(aligned_pairs, log_trans, sub_matrix, pi):
    """Aggregate observed counts across all pairs (CherryML-style).

    Args:
        aligned_pairs: list of (name_i, name_j, aligned_i, aligned_j) where
            aligned_i/j are gapped integer arrays (-1 = gap)
        log_trans: (5, 5) log transition matrix
        sub_matrix: (A, A) substitution probability matrix
        pi: (A,) equilibrium distribution

    Returns:
        total_ll: total log-likelihood across all pairs
        agg_n_trans: (5, 5) aggregated transition counts
        all_match_pairs: list of (a, b, weight) triples
        total_anc_len: total ancestor length across pairs
    """
    agg_n_trans = jnp.zeros((N_STATES, N_STATES))
    all_match_pairs = []
    total_ll = 0.0
    total_anc_len = 0

    for _, _, aln_i, aln_j in aligned_pairs:
        states, anc_chars, desc_chars = alignment_to_states(aln_i, aln_j)
        if not states:
            continue

        n_trans, match_pairs = observed_counts(states, anc_chars, desc_chars)
        ll = observed_log_likelihood(log_trans, sub_matrix, pi, states, anc_chars, desc_chars)

        agg_n_trans = agg_n_trans + n_trans
        all_match_pairs.extend(match_pairs)
        total_ll += ll
        total_anc_len += len(anc_chars)

    return total_ll, agg_n_trans, all_match_pairs, total_anc_len


# --- MixDom: 1D Forward-Backward with constrained emission type ---

def mixdom_constrained_emissions(states, anc_chars, desc_chars,
                                  state_types, sub_matrix, pi, n_states,
                                  sub_matrices=None, pis=None, n_frag=1,
                                  gamma_labels=None, sub_matrices_per_gamma=None):
    """Build per-column emission log-probs for MixDom 1D FB.

    WARNING: This uses Python loops over positions × states and is ~100x
    slower than ``mixdom_constrained_emissions_vectorized()``. Use the
    vectorized version for all production/experiment code.

    Each alignment column has a known type (M/I/D). Only MixDom states
    whose type matches the column type get non-NEG_INF emissions.

    When sub_matrices and pis are provided (per-domain), each MixDom state
    gets domain-specific emission probabilities. Otherwise all states of
    the same type share a single emission (backward compatible).

    When gamma_labels and sub_matrices_per_gamma are provided, match emissions
    use the gamma-class-specific substitution matrix for each column.

    Args:
        states: list of alignment column states (M, I, D)
        anc_chars: ancestor characters for M/D columns
        desc_chars: descendant characters for M/I columns
        state_types: (n_states,) MixDom state type array
        sub_matrix: (A, A) substitution probability matrix (shared fallback)
        pi: (A,) equilibrium distribution (shared fallback)
        n_states: number of MixDom states
        sub_matrices: optional (n_dom, A, A) per-domain substitution matrices
        pis: optional (n_dom, A) per-domain equilibrium distributions
        n_frag: fragments per domain (for state→domain mapping)
        gamma_labels: optional list of int (0..G-1), per alignment column.
            Selects which gamma-class substitution matrix to use per column.
        sub_matrices_per_gamma: optional (G, n_dom, A, A) or (G, A, A)
            gamma-class-specific substitution matrices.
            When per-domain: (G, n_dom, A, A).
            When shared: (G, A, A).

    Returns:
        log_emissions: (L, n_states) log emission probs
    """
    per_domain = sub_matrices is not None and pis is not None
    use_gamma = gamma_labels is not None and sub_matrices_per_gamma is not None

    if use_gamma:
        log_subs_gamma = np.log(np.maximum(np.asarray(sub_matrices_per_gamma), 1e-30))
        gamma_per_domain = log_subs_gamma.ndim == 4  # (G, n_dom, A, A)

    if per_domain:
        log_subs = np.log(np.maximum(np.asarray(sub_matrices), 1e-30))
        log_pis = np.log(np.maximum(np.asarray(pis), 1e-30))
        n_dom = log_subs.shape[0]
    else:
        log_sub = np.asarray(jnp.log(sub_matrix + 1e-30))
        log_pi = np.asarray(jnp.log(pi + 1e-30))

    L = len(states)
    st_np = np.asarray(state_types)

    log_emit = np.full((L, n_states), NEG_INF)
    ai, di = 0, 0

    for t, state in enumerate(states):
        # Get gamma class for this column (default 0 if absent or -1)
        g = 0
        if use_gamma and t < len(gamma_labels):
            g = max(0, gamma_labels[t])

        if state == M:
            a, d = int(anc_chars[ai]), int(desc_chars[di])
            ai += 1
            di += 1
            for s in range(n_states):
                if st_np[s] == M:
                    if use_gamma and per_domain and gamma_per_domain:
                        dom = (s - 2) // (5 * n_frag)
                        log_emit[t, s] = log_pis[dom, a] + log_subs_gamma[g, dom, a, d]
                    elif use_gamma and not gamma_per_domain:
                        if per_domain:
                            dom = (s - 2) // (5 * n_frag)
                            log_emit[t, s] = log_pis[dom, a] + log_subs_gamma[g, a, d]
                        else:
                            log_emit[t, s] = log_pi[a] + log_subs_gamma[g, a, d]
                    elif per_domain:
                        dom = (s - 2) // (5 * n_frag)
                        log_emit[t, s] = log_pis[dom, a] + log_subs[dom, a, d]
                    else:
                        log_emit[t, s] = log_pi[a] + log_sub[a, d]
        elif state == I:
            d = int(desc_chars[di])
            di += 1
            for s in range(n_states):
                if st_np[s] == I:
                    if per_domain:
                        dom = (s - 2) // (5 * n_frag)
                        log_emit[t, s] = log_pis[dom, d]
                    else:
                        log_emit[t, s] = log_pi[d]
        elif state == D:
            a = int(anc_chars[ai])
            ai += 1
            for s in range(n_states):
                if st_np[s] == D:
                    if per_domain:
                        dom = (s - 2) // (5 * n_frag)
                        log_emit[t, s] = log_pis[dom, a]
                    else:
                        log_emit[t, s] = log_pi[a]

    return jnp.array(log_emit)


def mixdom_constrained_emissions_vectorized(states_arr, anc_full, desc_full,
                                             state_types, n_states,
                                             log_subs, log_pis, n_frag=1,
                                             classdist=None,
                                             class_log_subs=None,
                                             class_log_pis=None):
    """Vectorized emission construction for per-domain MixDom 1D FB.

    Same semantics as mixdom_constrained_emissions but uses NumPy array
    operations instead of Python loops over positions × states. ~100x faster
    for large N (e.g. d10f1 with N=52).

    When classdist is provided, computes per-fragment-per-class emissions:
        E(s, t) = logsumexp_c(log classdist[d,f,c] + log_class_sub_c[anc, desc])
    instead of per-domain:
        E(s, t) = log_sub[dom, anc, desc]

    Args:
        states_arr: (L,) int array with M=1, I=2, D=3 per alignment position
        anc_full: (L,) int array of ancestor chars (valid at M/D positions,
            0 elsewhere)
        desc_full: (L,) int array of descendant chars (valid at M/I positions,
            0 elsewhere)
        state_types: (n_states,) state type array
        n_states: number of MixDom states
        log_subs: (n_dom, A, A) log substitution matrices (per-domain fallback)
        log_pis: (n_dom, A) log equilibrium distributions (per-domain fallback)
        n_frag: fragments per domain
        classdist: optional (N, F, C) per-fragment class distribution
        class_log_subs: optional (C, A, A) per-class log substitution matrices
        class_log_pis: optional (C, A) per-class log equilibrium distributions

    Returns:
        log_emissions: (L, n_states) log emission probs (np.float32 array)
    """
    L = len(states_arr)
    st = np.asarray(state_types)

    # Domain and fragment index per state
    dom_idx = np.maximum((np.arange(n_states) - 2) // (5 * n_frag), 0)
    frag_idx = np.maximum(((np.arange(n_states) - 2) % (5 * n_frag)) % n_frag, 0)

    is_M_state = (st == M)
    is_I_state = (st == I)
    is_D_state = (st == D)

    if classdist is not None and class_log_subs is not None and class_log_pis is not None:
        # Per-fragment-per-class emissions via logsumexp over classes.
        # Replaces the prior `for c in range(C)` and `for s in range(n_states)`
        # Python loops with vectorised numpy ops:
        #   - Per-class per-position log-probs are built by fancy indexing
        #     (no `for c` loop).
        #   - Per-state logsumexp over c uses the proper PER-(s, l) max-
        #     shift normalisation — bit-equivalent to the prior per-state
        #     loop on float64 (the discrepant-argmax-c case the matmul
        #     shortcut would slightly mishandle is fully recovered here).
        # No averaging across states or classes; no consensus value;
        # `if s < 2: continue` is preserved by writing NEG_INF into states
        # 0 and 1 (state-0/1 dom_idx is clamped to 0 by the np.maximum
        # guard above so the logsumexp returns a finite number — we
        # discard it here for safety; downstream `is_M_state` / etc.
        # masks would gate it out anyway).
        classdist = np.asarray(classdist)
        class_log_subs = np.asarray(class_log_subs)
        class_log_pis = np.asarray(class_log_pis)
        _Nc, _Fc, C = classdist.shape

        log_cd = np.log(np.maximum(classdist, 1e-300))  # (n_dom, n_frag, C)

        # Per-class per-position log-probs (vectorised fancy indexing).
        # class_log_subs[:, anc_full, desc_full] picks (anc[l], desc[l]) for
        # each c and l → (C, L). class_log_pis[:, anc_full] is (C, L).
        class_match_emit = (class_log_pis[:, anc_full]
                            + class_log_subs[:, anc_full, desc_full])  # (C, L)
        class_ins_emit = class_log_pis[:, desc_full]                   # (C, L)
        class_del_emit = class_log_pis[:, anc_full]                    # (C, L)

        # Per-state log_cd lookup: log_cd_per_s[s, c] = log_cd[dom_idx[s],
        # frag_idx[s], c]. dom_idx, frag_idx have shape (n_states,) so
        # advanced indexing produces (n_states, C).
        log_cd_per_s = log_cd[dom_idx, frag_idx, :]  # (n_states, C)

        def _logsumexp_classes(class_emit):
            """match/ins/del emit[s, l] = logsumexp_c(log_cd_per_s[s, c]
            + class_emit[c, l]).

            Builds a (n_states, C, L) combined-log tensor and reduces over
            the c axis using the standard subtract-max trick; numerically
            equivalent to the per-state Python loop in the prior code.
            """
            # combined[s, c, l] = log_cd_per_s[s, c] + class_emit[c, l]
            combined = (log_cd_per_s[:, :, None]
                        + class_emit[None, :, :])  # (n_states, C, L)
            # Per-(s, l) max over c — same as the prior per-state loop.
            cmax = np.max(combined, axis=1, keepdims=True)  # (n_states, 1, L)
            shifted = combined - cmax
            return cmax[:, 0, :] + np.log(np.sum(np.exp(shifted), axis=1))

        match_emit = _logsumexp_classes(class_match_emit)
        ins_emit = _logsumexp_classes(class_ins_emit)
        del_emit = _logsumexp_classes(class_del_emit)

        # Mask state 0 (SS) and state 1 (EE): they have no emission. The
        # logsumexp above produced a finite number for them (dom_idx is
        # clamped to 0 by np.maximum at line 291), but the downstream
        # mask_MM / mask_II / mask_DD already gate them out via
        # is_M_state[0:2] = False. Setting NEG_INF here for safety.
        if n_states >= 2:
            match_emit[:2] = NEG_INF
            ins_emit[:2] = NEG_INF
            del_emit[:2] = NEG_INF
    else:
        # Per-domain emissions (original path)
        match_emit = (log_pis[dom_idx[:, None], anc_full[None, :]] +
                      log_subs[dom_idx[:, None], anc_full[None, :], desc_full[None, :]])
        ins_emit = log_pis[dom_idx[:, None], desc_full[None, :]]
        del_emit = log_pis[dom_idx[:, None], anc_full[None, :]]

    # Position type masks
    is_M_pos = (states_arr == M)
    is_I_pos = (states_arr == I)
    is_D_pos = (states_arr == D)

    mask_MM = is_M_state[:, None] & is_M_pos[None, :]
    mask_II = is_I_state[:, None] & is_I_pos[None, :]
    mask_DD = is_D_state[:, None] & is_D_pos[None, :]

    log_emit = np.full((n_states, L), NEG_INF, dtype=np.float32)
    log_emit = np.where(mask_MM, match_emit, log_emit)
    log_emit = np.where(mask_II, ins_emit, log_emit)
    log_emit = np.where(mask_DD, del_emit, log_emit)

    return log_emit.T  # (L, n_states)


def mixdom_constrained_e_step(aligned_pairs, log_chi, state_types,
                               sub_matrix, pi, n_states):
    """E-step for constrained MixDom: 1D FB along guide alignment.

    The alignment fixes which columns are M/I/D, but domain/fragment
    assignments are latent. Forward-Backward over the N-state MixDom
    HMM yields posterior domain assignments and expected transition counts.

    Uses sequential scan (O(L)). Associative scan (O(log L)) is available
    for forward only; backward associative scan is TODO.

    Args:
        aligned_pairs: list of (name_i, name_j, aligned_i, aligned_j)
        log_chi: (N, N) log transition matrix for MixDom
        state_types: (N,) state type codes
        sub_matrix, pi: substitution model
        n_states: number of MixDom states

    Returns:
        total_ll: total log-likelihood
        agg_n_chi: (N, N) aggregated expected transition counts
        all_match_info: list of (states, posteriors, anc_chars, desc_chars) per pair
    """
    SS, EE = 0, 1  # MixDom start/end state indices

    agg_n_chi = jnp.zeros((n_states, n_states))
    total_ll = 0.0
    all_match_info = []

    for _, _, aln_i, aln_j in aligned_pairs:
        states, anc_chars, desc_chars = alignment_to_states(aln_i, aln_j)
        if not states:
            continue

        log_emit = mixdom_constrained_emissions(
            states, anc_chars, desc_chars, state_types, sub_matrix, pi, n_states)

        log_prob, posteriors, n_chi = forward_backward_1d_associative(
            log_chi, log_emit, init_state=SS, final_state=EE)
        total_ll += float(log_prob)
        agg_n_chi = agg_n_chi + n_chi
        all_match_info.append((states, np.asarray(posteriors), anc_chars, desc_chars))

    return total_ll, agg_n_chi, all_match_info


# --- Pairwise alignment extraction from MSA ---

def prepare_aligned_pairs(aligned_seqs, max_pairs=None):
    """Extract pairwise alignments from MSA-aligned sequences.

    Args:
        aligned_seqs: dict of {name: gapped_int_array} (from fetch_balibase_reference)
        max_pairs: maximum number of pairs (None = all)

    Returns:
        list of (name_i, name_j, aligned_i, aligned_j) tuples
    """
    names = sorted(aligned_seqs.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((
                names[i], names[j],
                np.asarray(aligned_seqs[names[i]]),
                np.asarray(aligned_seqs[names[j]]),
            ))
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs
