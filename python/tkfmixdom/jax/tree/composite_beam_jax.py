"""JAX-vectorized composite likelihood beam search for ancestral reconstruction.

Same algorithm as composite_beam.py but with JAX-accelerated inner loops:
  - jax.lax.scan for the I-type sequential sweep
  - jax.vmap over ancestor residues (20-way parallelism)
  - jax.vmap over beam entries (B-way parallelism)
  - Geometric bin padding for JIT cache reuse

Function signature matches composite_beam_reconstruct() exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from ..core.params import S, M, I, D, E
from ..dp.hmm import _pad_to_bin

# Force float64
jax.config.update("jax_enable_x64", True)

NEG_INF = -1e30


# ============================================================
# Geometric bin sizes (same as hmm.py)
# ============================================================

_GEOM_BINS = np.unique(np.concatenate([
    np.array([0, 1, 2, 3, 4]),
    np.geomspace(4, 4096, num=40, dtype=int),
])).tolist()


# ============================================================
# JIT-compiled row advance
# ============================================================


@partial(jax.jit, static_argnums=(6, 7))
def _advance_row_jax_impl(fwd_row_prev, emit_row, log_chi, is_M, is_I, is_D, ns, L_pad):
    """Advance one ancestor row. JIT-compiled with scan for I-type.

    Args:
        fwd_row_prev: (L_pad+1, ns) previous forward row
        emit_row: (L_pad+1, ns) emission log-probs
        log_chi: (ns, ns) log transition matrix
        is_M: (ns,) boolean mask for M-type states
        is_I: (ns,) boolean mask for I-type states
        is_D: (ns,) boolean mask for D-type states
        ns: number of states (static)
        L_pad: padded descendant length (static)

    Returns:
        fwd_row: (L_pad+1, ns) new forward row
    """

    # --- D-type: fwd[j,s] = logsumexp_r(prev[j,r] + chi[r,s]) + emit[j,s] ---
    # Vectorized over all j simultaneously
    # prev: (L_pad+1, ns), chi: (ns, ns) → (L_pad+1, ns)
    d_scores = jax.nn.logsumexp(
        fwd_row_prev[:, :, None] + log_chi[None, :, :], axis=1
    ) + emit_row  # (L_pad+1, ns)

    # --- M-type: fwd[j,s] = logsumexp_r(prev[j-1,r] + chi[r,s]) + emit[j,s] ---
    # Shifted: use prev[0..L_pad-1] for positions 1..L_pad
    m_incoming = jax.nn.logsumexp(
        fwd_row_prev[:-1, :, None] + log_chi[None, :, :], axis=1
    ) + emit_row[1:]  # (L_pad, ns)
    # Pad position 0 with NEG_INF
    m_scores_full = jnp.concatenate(
        [jnp.full((1, ns), NEG_INF), m_incoming], axis=0
    )  # (L_pad+1, ns)

    # Initialize: D-type and M-type combined
    # States that are neither M, I, nor D (i.e. S, E) stay at NEG_INF
    init_row = jnp.where(
        is_D[None, :], d_scores,
        jnp.where(is_M[None, :], m_scores_full, jnp.full((L_pad + 1, ns), NEG_INF))
    )

    # --- I-type: sequential scan over j ---
    # fwd[j, s] = logsumexp_r(fwd[j-1, r] + chi[r, s]) + emit[j, s]
    # This has a j→j dependency, so we use lax.scan

    def i_scan_step(prev_cell, j):
        """prev_cell: (ns,) = fwd_row[j-1]. Compute I-type for position j."""
        # I-type incoming from previous cell
        i_incoming = jax.nn.logsumexp(
            prev_cell[:, None] + log_chi, axis=0
        ) + emit_row[j]  # (ns,)

        # For I-type states: use i_incoming (logaddexp with init_row[j] for M/D contributions)
        # For non-I-type states: use init_row[j] (already computed)
        cell = jnp.where(
            is_I,
            jnp.logaddexp(init_row[j], i_incoming),
            init_row[j]
        )
        return cell, cell

    # Position 0: no I-type contribution (no j-1 exists)
    cell0 = init_row[0]

    _, scanned = jax.lax.scan(i_scan_step, cell0, jnp.arange(1, L_pad + 1))
    fwd_row = jnp.concatenate([cell0[None, :], scanned], axis=0)  # (L_pad+1, ns)

    return fwd_row


@partial(jax.jit, static_argnums=(6, 7))
def _advance_row_batch(fwd_rows_prev, emit_rows, log_chi, is_M, is_I, is_D, ns, L_pad):
    """Advance rows for a batch of beam entries and all ancestor residues.

    Args:
        fwd_rows_prev: (B, L_pad+1, ns) previous forward rows for B beam entries
        emit_rows: (A, L_pad+1, ns) precomputed emission rows for A ancestor chars
        log_chi: (ns, ns) log transition matrix
        is_M, is_I, is_D: (ns,) boolean masks
        ns, L_pad: static ints

    Returns:
        fwd_rows_new: (B, A, L_pad+1, ns) new forward rows
    """
    # vmap over ancestor residues (A=20)
    def advance_one_residue(emit_row):
        # vmap over beam entries (B)
        def advance_one_entry(fwd_prev):
            return _advance_row_jax_impl(fwd_prev, emit_row, log_chi, is_M, is_I, is_D, ns, L_pad)
        return jax.vmap(advance_one_entry)(fwd_rows_prev)  # (B, L_pad+1, ns)

    # Result: (A, B, L_pad+1, ns) → transpose to (B, A, L_pad+1, ns)
    result = jax.vmap(advance_one_residue)(emit_rows)  # (A, B, L_pad+1, ns)
    return jnp.transpose(result, (1, 0, 2, 3))  # (B, A, L_pad+1, ns)


@partial(jax.jit, static_argnums=(3,))
def _terminal_score_batch(fwd_rows, log_chi, e_idx, L_real):
    """Compute terminal scores for a batch of forward rows.

    Args:
        fwd_rows: (B, A, L_pad+1, ns) forward rows
        log_chi: (ns, ns) log transition matrix
        e_idx: int, index of End state
        L_real: int (static), real descendant length

    Returns:
        scores: (B, A) terminal log-probabilities
    """
    # Extract row at position L_real: (B, A, ns)
    final_cells = fwd_rows[:, :, L_real, :]
    # Score to end: logsumexp over states
    # chi[:, e_idx] is (ns,)
    chi_to_end = log_chi[:, e_idx]  # (ns,)
    scores = jax.nn.logsumexp(final_cells + chi_to_end[None, None, :], axis=-1)
    return scores  # (B, A)


@partial(jax.jit, static_argnums=(4, 5))
def _init_row_jax(log_chi, emit_all_0, is_I, e_idx_unused, ns, L_pad):
    """Initialize forward row 0 for a pair HMM.

    Args:
        log_chi: (ns, ns) log transition matrix
        emit_all_0: (L_pad+1, ns) emission row for I-type at row 0
        is_I: (ns,) boolean mask
        e_idx_unused: not used, kept for signature consistency
        ns, L_pad: static ints

    Returns:
        row0: (L_pad+1, ns) initial forward row
    """
    row0_cell0 = jnp.full(ns, NEG_INF)
    row0_cell0 = row0_cell0.at[S].set(0.0)

    def scan_step(prev_cell, j):
        incoming = jax.nn.logsumexp(
            prev_cell[:, None] + log_chi, axis=0
        ) + emit_all_0[j]
        cell = jnp.where(is_I, incoming, NEG_INF)
        return cell, cell

    _, rest = jax.lax.scan(scan_step, row0_cell0, jnp.arange(1, L_pad + 1))
    row0 = jnp.concatenate([row0_cell0[None, :], rest], axis=0)
    return row0


# ============================================================
# Precompute emissions (JAX version)
# ============================================================

def _precompute_all_emit_rows_jax(desc_seq, state_types, dom_idx,
                                   log_subs, log_pis, L_desc, L_pad, A=20):
    """Precompute emission rows for all ancestor chars, with padding.

    Class-marginal version (MixDom1 / class_dist=None).

    Returns:
        emit_all: (A, L_pad+1, ns) — padded emission table.
                  Positions beyond L_desc are set to NEG_INF for M/I states.
    """
    ns = len(state_types)

    # Pad log_subs and log_pis to size A+1 for wildcard (index 20) support
    log_subs = jnp.pad(log_subs, ((0, 0), (0, 1), (0, 1)), constant_values=0.0)
    log_pis = jnp.pad(log_pis, ((0, 0), (0, 1)), constant_values=0.0)

    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    state_log_pi = log_pis[dom_idx]    # (ns, A+1)
    state_log_sub = log_subs[dom_idx]  # (ns, A+1, A+1)

    # Pad desc_seq to L_pad
    desc_padded = jnp.zeros(L_pad, dtype=jnp.int32)
    desc_padded = desc_padded.at[:L_desc].set(desc_seq)

    # Match: emit[a, j, s] = log_pi[s, a] + log_sub[s, a, desc[j-1]] for j>=1
    # Ancestor chars are 0..A-1 only (not wildcard); desc may contain wildcard (20)
    pi_anc = state_log_pi[:, :A].T  # (A, ns) — pi_anc[a, s] = log_pi[s, a]
    sub_anc_desc = state_log_sub[:, :A, :][:, :, desc_padded]  # (ns, A, L_pad)
    # sub_anc_desc[s, a, j] = log_sub[s, a, desc_padded[j]]
    # match[a, j+1, s] = pi_anc[a, s] + sub_anc_desc[s, a, j]
    match_body = pi_anc[:, None, :] + sub_anc_desc.transpose(1, 2, 0)  # (A, L_pad, ns)

    # Insert: emit[a, j, s] = log_pi[s, desc_padded[j-1]] (independent of a)
    ins_body = state_log_pi[:, desc_padded].T  # (L_pad, ns)

    # Delete: emit[a, j, s] = log_pi[s, a] (independent of j)
    del_all = pi_anc  # (A, ns)

    # Build full (A, L_pad+1, ns) table
    emit_all = jnp.full((A, L_pad + 1, ns), 0.0)

    # Position 0: only D-type gets emission
    emit_all = emit_all.at[:, 0, :].set(
        jnp.where(is_D[None, :], del_all, 0.0)
    )

    # Positions 1..L_pad:
    # M-type: match_body
    # I-type: ins_body (broadcast over A)
    # D-type: del_all (broadcast over j)
    body = (is_M[None, None, :] * match_body +
            is_I[None, None, :] * ins_body[None, :, :] +
            is_D[None, None, :] * del_all[:, None, :])

    # Non-emitting states get 0
    is_emit = is_M | is_I | is_D
    body = jnp.where(is_emit[None, None, :], body, 0.0)

    emit_all = emit_all.at[:, 1:, :].set(body)

    # Mask positions beyond L_desc to NEG_INF for M and I states
    # (D states can still be at any j, but M/I beyond L_desc are impossible)
    mask_beyond = jnp.arange(L_pad + 1) > L_desc  # (L_pad+1,)
    needs_mask = is_M | is_I  # (ns,)
    penalty = jnp.where(
        mask_beyond[:, None] & needs_mask[None, :],
        NEG_INF, 0.0
    )  # (L_pad+1, ns)
    emit_all = emit_all + penalty[None, :, :]

    return emit_all


def _precompute_all_emit_rows_jax_class(
        desc_seq, state_types, dom_idx, frag_idx,
        class_log_subs, class_log_pis, class_log_dist,
        L_desc, L_pad, A=20):
    """Class-aware emission table with padding.

    Computes the per-state, per-(ancestor, descendant_pos) emission as a
    log-mixture over per-fragment site classes:

        M[a, j, s] = log sum_c w[s, c] * class_pi[c, a] * class_P[c, a, desc[j-1]]
        I[a, j, s] = log sum_c w[s, c] * class_pi[c, desc[j-1]]
        D[a, j, s] = log sum_c w[s, c] * class_pi[c, a]

    where ``w[s, c] = classdist[dom[s], frag[s], c]``.

    Args:
        desc_seq: (L_desc,) descendant sequence.
        state_types: (ns,) M/I/D/S/E codes.
        dom_idx, frag_idx: (ns,) per-state lookups.
        class_log_subs: (C, A, A) log P(t) per class.
        class_log_pis:  (C, A) log equilibrium per class.
        class_log_dist: (D, F, C) log classdist (NEG_INF where dist==0).
        L_desc, L_pad: real and padded descendant length.

    Returns:
        emit_all: (A, L_pad+1, ns).
    """
    ns = len(state_types)
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    # Pad class arrays to A+1 for wildcard support.
    class_log_subs = jnp.pad(class_log_subs, ((0, 0), (0, 1), (0, 1)),
                             constant_values=0.0)
    class_log_pis = jnp.pad(class_log_pis, ((0, 0), (0, 1)),
                            constant_values=0.0)

    # Pad descendant.
    desc_padded = jnp.zeros(L_pad, dtype=jnp.int32)
    desc_padded = desc_padded.at[:L_desc].set(desc_seq)

    # Per-state log w[s, c] = log classdist[dom[s], frag[s], c]
    state_log_w = class_log_dist[dom_idx, frag_idx, :]  # (ns, C)

    # Match: (A, L_pad, ns) — each entry is logsumexp_c(log w[s,c] +
    # log class_pi[c, a] + log class_P[c, a, desc[j]]).
    # Build (C, A, L_pad): class_P at (anc, desc).
    csub_anc_desc = class_log_subs[:, :A, :][:, :, desc_padded]  # (C, A, L_pad)
    cpi_anc = class_log_pis[:, :A].T  # (A, C)
    # combined_match[a, j, c] = cpi_anc[a, c] + csub_anc_desc[c, a, j]
    combined_match = cpi_anc[:, None, :] + csub_anc_desc.transpose(1, 2, 0)
    # Now add per-state log_w[s, c] and logsumexp over c → (A, L_pad, ns).
    # state_log_w: (ns, C); combined_match: (A, L_pad, C).
    # Result[a, j, s] = logsumexp_c(combined_match[a, j, c] + state_log_w[s, c]).
    match_body = jax.nn.logsumexp(
        combined_match[:, :, None, :] + state_log_w[None, None, :, :],
        axis=-1)  # (A, L_pad, ns)

    # Insert: (L_pad, ns) — logsumexp_c(log w[s,c] + log class_pi[c, desc[j]]).
    cpi_desc = class_log_pis[:, desc_padded]  # (C, L_pad)
    # ins_body[j, s] = logsumexp_c(state_log_w[s, c] + cpi_desc[c, j]).
    ins_body = jax.nn.logsumexp(
        state_log_w[None, :, :] + cpi_desc.T[:, None, :],
        axis=-1)  # (L_pad, ns)

    # Delete: (A, ns) — logsumexp_c(log w[s,c] + log class_pi[c, a]).
    del_all = jax.nn.logsumexp(
        state_log_w[None, :, :] + cpi_anc[:, None, :],
        axis=-1)  # (A, ns)

    emit_all = jnp.full((A, L_pad + 1, ns), 0.0)
    # Position 0: only D-type gets emission.
    emit_all = emit_all.at[:, 0, :].set(
        jnp.where(is_D[None, :], del_all, 0.0))

    body = (is_M[None, None, :] * match_body +
            is_I[None, None, :] * ins_body[None, :, :] +
            is_D[None, None, :] * del_all[:, None, :])
    is_emit = is_M | is_I | is_D
    body = jnp.where(is_emit[None, None, :], body, 0.0)
    emit_all = emit_all.at[:, 1:, :].set(body)

    mask_beyond = jnp.arange(L_pad + 1) > L_desc  # (L_pad+1,)
    needs_mask = is_M | is_I
    penalty = jnp.where(
        mask_beyond[:, None] & needs_mask[None, :],
        NEG_INF, 0.0)
    emit_all = emit_all + penalty[None, :, :]
    return emit_all


# ============================================================
# Main beam search (JAX-accelerated)
# ============================================================

def composite_beam_reconstruct_jax(
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
    """JAX-accelerated composite likelihood beam search for ancestor reconstruction.

    Same interface as composite_beam_reconstruct() in composite_beam.py.
    If fixed_len is set, produces a sequence of exactly that length.
    Uses JAX for vectorized row advances and terminal scoring.

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
        class_sub_matrices_list: optional list of K (C, A, A) per-class
            P(t) matrices.
        class_pis_list: optional list of K (C, A) per-class equilibrium.
        class_dist: optional (D, F, C) per-(domain, fragment) class
            distribution. When all three are provided, emissions become
            full per-class log-mixtures (MixDom2). Otherwise the function
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
                "composite_beam_reconstruct_jax: per-class plumbing requires "
                "all of class_sub_matrices_list, class_pis_list, and "
                "class_dist; got partial inputs.")

    use_class = class_sub_matrices_list is not None
    K = len(desc_seqs)
    A = singlet_log_pi.shape[0]
    desc_seqs_np = [np.asarray(s, dtype=np.int32) for s in desc_seqs]
    desc_lens = [len(s) for s in desc_seqs_np]

    if use_class:
        class_log_dist_np = np.where(
            np.asarray(class_dist) > 0,
            np.log(np.maximum(np.asarray(class_dist), 1e-300)),
            NEG_INF)
        class_log_dist_j = jnp.asarray(class_log_dist_np, dtype=jnp.float64)

    singlet_log_trans = jnp.asarray(singlet_log_trans, dtype=jnp.float64)
    singlet_log_pi = jnp.asarray(singlet_log_pi, dtype=jnp.float64)
    singlet_log_end = jnp.asarray(singlet_log_end, dtype=jnp.float64)

    if desc_weights is None:
        desc_weights = [1.0] * K
    desc_weights_arr = jnp.array(desc_weights, dtype=jnp.float64)
    singlet_weight = 1.0 - float(sum(desc_weights))

    # Precompute per-descendant data
    desc_data = []
    for k in range(K):
        st = np.asarray(state_types_list[k], dtype=np.int32)
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

        log_subs = jnp.log(jnp.maximum(jnp.asarray(subs, dtype=jnp.float64), 1e-300))
        log_pis_k = jnp.log(jnp.maximum(jnp.asarray(pis_k, dtype=jnp.float64), 1e-300))

        L_desc = desc_lens[k]
        L_pad = _pad_to_bin(L_desc)

        # State type masks
        is_M = jnp.asarray(st == M)
        is_I = jnp.asarray(st == I)
        is_D = jnp.asarray(st == D)

        # End state index
        e_idx = int(np.argmax(st == E))

        # Log transition matrix
        log_chi = jnp.asarray(log_chi_list[k], dtype=jnp.float64)

        # Precompute emission rows for all ancestor chars: (A, L_pad+1, ns)
        desc_seq_jax = jnp.asarray(desc_seqs_np[k], dtype=jnp.int32)
        dom_idx_jax = jnp.asarray(dom_idx, dtype=jnp.int32)
        frag_idx_jax = jnp.asarray(frag_idx, dtype=jnp.int32)

        if use_class:
            class_subs_k = jnp.asarray(class_sub_matrices_list[k],
                                       dtype=jnp.float64)
            class_pis_k = jnp.asarray(class_pis_list[k], dtype=jnp.float64)
            class_log_subs = jnp.log(jnp.maximum(class_subs_k, 1e-300))
            class_log_pis_arr = jnp.log(jnp.maximum(class_pis_k, 1e-300))
            emit_all = _precompute_all_emit_rows_jax_class(
                desc_seq_jax, st, dom_idx_jax, frag_idx_jax,
                class_log_subs, class_log_pis_arr, class_log_dist_j,
                L_desc, L_pad, A,
            )
        else:
            emit_all = _precompute_all_emit_rows_jax(
                desc_seq_jax, st, dom_idx_jax,
                log_subs, log_pis_k, L_desc, L_pad, A
            )

        # Initialize forward row 0
        # Use emit from any ancestor char for I-type (I emissions don't depend on ancestor)
        row0 = _init_row_jax(log_chi, emit_all[0], is_I, e_idx, ns, L_pad)

        desc_data.append({
            'ns': ns,
            'L_desc': L_desc,
            'L_pad': L_pad,
            'log_chi': log_chi,
            'is_M': is_M,
            'is_I': is_I,
            'is_D': is_D,
            'e_idx': e_idx,
            'emit_all': emit_all,  # (A, L_pad+1, ns)
            'row0': row0,          # (L_pad+1, ns)
        })

    # Beam entries
    beam = [{
        'seq': [],
        'fwd_rows': [np.asarray(dd['row0']) for dd in desc_data],
        'singlet_state': -1,
        'singlet_fwd': 0.0,
    }]

    best_complete = None
    target_len = fixed_len if fixed_len is not None else max_len

    for anc_pos in range(target_len):
        B = len(beam)

        # --- Pair HMM: advance all beam entries x all residues for each descendant ---
        # For each descendant k: stack beam forward rows → (B, L_pad+1, ns)
        # Then call _advance_row_batch → (B, A, L_pad+1, ns)

        pair_term_scores = jnp.zeros((B, A))  # accumulate terminal scores

        new_fwd_rows_all = []  # list of K arrays, each (B, A, L_pad+1, ns)

        for k in range(K):
            dd = desc_data[k]
            ns_k = dd['ns']
            L_pad_k = dd['L_pad']
            L_desc_k = dd['L_desc']

            # Stack beam entries' forward rows for descendant k.
            # Stack at numpy level first (cheap), then a single host->device
            # transfer rather than B individual jnp.asarray conversions.
            beam_rows_np = np.stack(
                [beam[b]['fwd_rows'][k] for b in range(B)], axis=0)
            beam_rows = jnp.asarray(beam_rows_np)  # (B, L_pad+1, ns)

            # Advance: (B, A, L_pad+1, ns)
            new_rows = _advance_row_batch(
                beam_rows, dd['emit_all'], dd['log_chi'],
                dd['is_M'], dd['is_I'], dd['is_D'], ns_k, L_pad_k
            )
            new_fwd_rows_all.append(new_rows)

            # Terminal scores: (B, A)
            term = _terminal_score_batch(
                new_rows, dd['log_chi'], dd['e_idx'], L_desc_k
            )
            pair_term_scores = pair_term_scores + desc_weights_arr[k] * term

        # --- Singlet scores: (B, A) — vectorised over beam dim ---
        prev_chars = jnp.array([beam[b]['singlet_state'] for b in range(B)],
                               dtype=jnp.int32)               # (B,)
        prev_fwds = jnp.array([beam[b]['singlet_fwd'] for b in range(B)],
                              dtype=jnp.float64)              # (B,)
        idx_safe = jnp.maximum(prev_chars, 0)                 # (B,) safe gather idx
        log_inc_trans = singlet_log_trans[idx_safe]           # (B, A)
        log_inc_pi = jnp.broadcast_to(singlet_log_pi[None, :], (B, A))
        is_init = (prev_chars < 0)[:, None]                   # (B, 1)
        log_inc = jnp.where(is_init, log_inc_pi, log_inc_trans)  # (B, A)
        singlet_fwds = prev_fwds[:, None] + log_inc           # (B, A)
        singlet_scores = singlet_fwds + singlet_log_end[None, :]  # (B, A)

        # --- Total scores: (B, A) ---
        total_scores = singlet_weight * singlet_scores + pair_term_scores

        # Flatten to (B*A,) for pruning
        total_flat = total_scores.reshape(-1)

        # Top-k pruning
        n_candidates = B * A
        k_select = min(beam_width, n_candidates)
        top_vals, top_idx = jax.lax.top_k(total_flat, k_select)

        # Convert flat indices to (beam_idx, residue_idx)
        top_vals_np = np.asarray(top_vals)
        top_idx_np = np.asarray(top_idx)
        beam_indices = top_idx_np // A
        residue_indices = top_idx_np % A

        # Extract singlet forward values (single host transfer)
        singlet_fwds_np = np.asarray(singlet_fwds)

        # Batch-gather forward rows for all top-k entries per descendant.
        # Replaces an inner k_select x K Python loop of single-element gathers
        # (each of which forced a JAX -> numpy device-host round trip).
        beam_idx_j = jnp.asarray(beam_indices, dtype=jnp.int32)
        res_idx_j = jnp.asarray(residue_indices, dtype=jnp.int32)
        gathered_per_k = []
        for k_desc in range(K):
            # new_fwd_rows_all[k_desc] has shape (B, A, L_pad+1, ns)
            gathered = new_fwd_rows_all[k_desc][beam_idx_j, res_idx_j]
            # Single host transfer per descendant rather than k_select transfers
            gathered_per_k.append(np.asarray(gathered))

        # Build new beam
        next_beam = []
        for rank in range(k_select):
            if top_vals_np[rank] <= NEG_INF / 2:
                continue
            b = int(beam_indices[rank])
            a = int(residue_indices[rank])

            new_fwd_list = [gathered_per_k[k_desc][rank] for k_desc in range(K)]

            next_beam.append({
                'seq': beam[b]['seq'] + [a],
                'fwd_rows': new_fwd_list,
                'singlet_state': a,
                'singlet_fwd': float(singlet_fwds_np[b, a]),
                'score': float(top_vals_np[rank]),
            })

        if not next_beam:
            break

        beam = next_beam

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
        if beam:
            return np.array(beam[0]['seq'], dtype=np.int32), beam[0]['score']
    elif best_complete is not None:
        return np.array(best_complete['seq'], dtype=np.int32), best_complete['score']

    return np.array([], dtype=np.int32), -np.inf


# ============================================================
# Self-test: compare JAX vs numpy on a small example
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/yam/tkf-mixdom/python")

    from tkfmixdom.jax.tree.composite_beam import (
        composite_beam_reconstruct,
        _precompute_all_emit_rows,
        _advance_ancestor_row,
        _terminal_score,
    )

    np.random.seed(42)

    # Small problem: 2 descendants, 1 domain, 1 fragment, alphabet=4
    A_size = 4
    n_dom = 1
    n_frag = 1
    # States: S(0), E(1), M(2), I(3), D(4)
    ns = 5
    state_types = np.array([S, E, M, I, D], dtype=np.int32)

    # Random sequences
    desc1 = np.array([0, 1, 2, 3], dtype=np.int32)
    desc2 = np.array([1, 2, 0], dtype=np.int32)

    # Random transition matrix (log-space)
    def make_log_trans(ns):
        raw = np.random.dirichlet(np.ones(ns), size=ns)
        return np.log(raw + 1e-300)

    log_chi1 = make_log_trans(ns)
    log_chi2 = make_log_trans(ns)

    # Substitution matrices (uniform-ish)
    subs1 = np.random.dirichlet(np.ones(A_size), size=(n_dom, A_size))
    subs2 = np.random.dirichlet(np.ones(A_size), size=(n_dom, A_size))

    # Equilibrium
    pi1 = np.random.dirichlet(np.ones(A_size), size=n_dom)
    pi2 = np.random.dirichlet(np.ones(A_size), size=n_dom)

    # Singlet HMM
    singlet_trans = np.log(np.random.dirichlet(np.ones(A_size), size=A_size) + 1e-300)
    singlet_pi = np.log(np.random.dirichlet(np.ones(A_size)) + 1e-300)
    singlet_end = np.log(np.random.uniform(0.01, 0.1, size=A_size))

    print("=" * 60)
    print("Self-test: composite_beam_reconstruct_jax vs numpy")
    print("=" * 60)

    # Test 1: Single row advance comparison
    print("\n--- Test 1: Single row advance ---")
    dom_idx = np.zeros(ns, dtype=np.int32)
    log_subs_np = np.log(np.maximum(subs1, 1e-300))
    log_pis_np = np.log(np.maximum(pi1, 1e-300))

    emit_np = _precompute_all_emit_rows(
        desc1, state_types, dom_idx, log_subs_np, log_pis_np, len(desc1), A_size)

    # Init row (numpy)
    from scipy.special import logsumexp as sp_logsumexp
    row0_np = np.full((len(desc1) + 1, ns), -1e30)
    row0_np[0, S] = 0.0
    i_mask = np.where(state_types == I)[0]
    chi_I = log_chi1[:, i_mask]
    for j in range(1, len(desc1) + 1):
        vals = row0_np[j-1, :, None] + chi_I
        row0_np[j, i_mask] = sp_logsumexp(vals, axis=0) + emit_np[0, j, i_mask]

    # Advance row (numpy)
    row1_np = _advance_ancestor_row(row0_np, emit_np[1], log_chi1, state_types, len(desc1))

    # Advance row (JAX)
    L_pad = _pad_to_bin(len(desc1))
    is_M_j = jnp.asarray(state_types == M)
    is_I_j = jnp.asarray(state_types == I)

    log_subs_j = jnp.log(jnp.maximum(jnp.asarray(subs1, dtype=jnp.float64), 1e-300))
    log_pis_j = jnp.log(jnp.maximum(jnp.asarray(pi1, dtype=jnp.float64), 1e-300))
    dom_idx_j = jnp.asarray(dom_idx)

    emit_jax = _precompute_all_emit_rows_jax(
        jnp.asarray(desc1), state_types, dom_idx_j,
        log_subs_j, log_pis_j, len(desc1), L_pad, A_size)

    # Compare emissions
    emit_jax_np = np.asarray(emit_jax[:, :len(desc1)+1, :])
    emit_diff = np.max(np.abs(emit_np - emit_jax_np))
    print(f"  Emission table max diff: {emit_diff:.2e}")
    assert emit_diff < 1e-12, f"Emission mismatch: {emit_diff}"

    # Init row (JAX)
    row0_jax = _init_row_jax(
        jnp.asarray(log_chi1, dtype=jnp.float64),
        emit_jax[0], is_I_j, 1, ns, L_pad)
    row0_jax_np = np.asarray(row0_jax[:len(desc1)+1, :])
    row0_diff = np.max(np.abs(row0_np - row0_jax_np))
    print(f"  Init row max diff: {row0_diff:.2e}")
    assert row0_diff < 1e-10, f"Init row mismatch: {row0_diff}"

    # Advance (JAX)
    # Pad row0
    row0_padded = jnp.full((L_pad + 1, ns), NEG_INF)
    row0_padded = row0_padded.at[:len(desc1)+1, :].set(jnp.asarray(row0_np, dtype=jnp.float64))

    is_D_j = jnp.asarray(state_types == D)
    row1_jax = _advance_row_jax_impl(
        row0_padded, emit_jax[1],
        jnp.asarray(log_chi1, dtype=jnp.float64),
        is_M_j, is_I_j, is_D_j, ns, L_pad)
    row1_jax_np = np.asarray(row1_jax[:len(desc1)+1, :])

    row1_diff = np.max(np.abs(row1_np - row1_jax_np))
    print(f"  Row advance max diff: {row1_diff:.2e}")
    assert row1_diff < 1e-8, f"Row advance mismatch: {row1_diff}"

    # Test 2: Full beam search comparison
    print("\n--- Test 2: Full beam search (beam_width=5, max_len=8) ---")

    args = dict(
        desc_seqs=[desc1, desc2],
        distances=[0.5, 0.3],
        log_chi_list=[log_chi1, log_chi2],
        state_types_list=[state_types, state_types],
        sub_matrices_list=[subs1, subs2],
        pis_list=[pi1, pi2],
        n_dom=n_dom,
        n_frag=n_frag,
        singlet_log_trans=singlet_trans,
        singlet_log_pi=singlet_pi,
        singlet_log_end=singlet_end,
        beam_width=5,
        max_len=8,
    )

    seq_np, score_np = composite_beam_reconstruct(**args)
    seq_jax, score_jax = composite_beam_reconstruct_jax(**args)

    print(f"  Numpy:  seq={seq_np}, score={score_np:.6f}")
    print(f"  JAX:    seq={seq_jax}, score={score_jax:.6f}")

    score_diff = abs(score_np - score_jax)
    print(f"  Score diff: {score_diff:.2e}")

    if np.array_equal(seq_np, seq_jax):
        print("  Sequences MATCH")
    else:
        print(f"  Sequences DIFFER (may be OK if scores are close)")

    assert score_diff < 1e-6, f"Score mismatch too large: {score_diff}"

    print("\n" + "=" * 60)
    print("All self-tests PASSED")
    print("=" * 60)
