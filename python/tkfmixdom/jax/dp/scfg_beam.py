"""Beam-pruned phylogenetic Inside for SCFGs via WPTT state pushing.

Computes the phylogenetic likelihood of an MSA under an order-1 singlet
SCFG, by pushing WPTT state up the phylogenetic tree for each Inside
span. This correctly handles the context-sensitivity of WPTTs (basepair
stacking) which prevents per-column Felsenstein pruning.

Architecture:
  - SCFG structural weights are folded into WPTT rule weights via
    scfg_weighted_wptt_weights(). This avoids tracking SCFG nonterminal
    state explicitly, at the cost of averaging over nonterminals that
    map to the same basepair context class.

  - Leaf k: CL_k[w, i, j] = WPTT×Recognizer composition, marginalizing
    out recognizer states. w indexes WPTT states. i, j are MSA column
    indices. The WPTT includes folded SCFG structural weights.

  - Internal node B with children L, R: SCFG bifurcation resets both
    children to START state. CL_B[START, i, j] = CL_L[START, i, j] ×
    CL_R[START, i, j]. Only START is populated at internal nodes.

  - Root: log P(MSA) = CL_root[START, 0, L].

  - Beam pruning: at each span (i, j), keep only top B WPTT states
    by log-probability. This controls the explosion from 246 WPTT states.

  - MSA-constrained: the alignment is fixed (columns from MSA). Beam
    searches over ancestral nucleotide identities and M/I/D alignment
    type, not over structural alignment.

MSA convention:
  msa[seq, col] = nucleotide index (0-3) or -1 for gap.
  Rows 0..N-1 are leaf sequences, ordered to match phylo-tree leaf indices.
  Columns 0..L-1 are alignment columns.
"""

import numpy as np
from ..distill.wptt import (
    WPTT, WPTTRule, is_ready_state, decode_wptt_state,
    IDX_START as WPTT_START, N_WPTT_STATES,
    ALN_M, ALN_I, ALN_D, ALN_R, ALN_NAMES,
)
from ..models.rna_grammar import (
    N_NUC, N_TOTAL_TERMINALS, left_terminal, right_terminal,
    pair_terminal, decode_terminal,
)
from ..core.rna import classify_basepair, CTX_NN, N_CONTEXT
from ..tree.recognizer import build_leaf_recognizer, Recognizer


def _build_ctx_lookup():
    """Build lookup tables for WPTT state context constraints.

    Returns:
        states_by_in_ctx: dict ctx -> list of WPTT state indices with that in_ctx
        out_ctx_of: dict w -> out_ctx for state w (CTX_NN for Start)
    """
    states_by_in_ctx = {c: [] for c in range(N_CONTEXT)}
    out_ctx_of = {}

    # Start state: in_ctx = out_ctx = CTX_NN
    states_by_in_ctx[CTX_NN].append(WPTT_START)
    out_ctx_of[WPTT_START] = CTX_NN

    for w in range(1, N_WPTT_STATES):
        aln, in_ctx, out_ctx = decode_wptt_state(w)
        states_by_in_ctx[in_ctx].append(w)
        out_ctx_of[w] = out_ctx

    return states_by_in_ctx, out_ctx_of


# Module-level cached lookup (computed once)
_CTX_LOOKUP = None


def _get_ctx_lookup():
    """Get or compute the cached context lookup tables."""
    global _CTX_LOOKUP
    if _CTX_LOOKUP is None:
        _CTX_LOOKUP = _build_ctx_lookup()
    return _CTX_LOOKUP


def msa_to_columns(msa, gap_char=-1):
    """Convert MSA to column profiles.

    Args:
        msa: (N, L) array, values are nucleotide indices (0-3) or gap_char.

    Returns:
        profiles: list of L dicts, each mapping nuc -> count
        ungapped_counts: (L,) array of non-gap entries per column
    """
    msa = np.asarray(msa)
    N, L = msa.shape
    profiles = []
    ungapped_counts = np.zeros(L, dtype=int)
    for col in range(L):
        profile = {}
        for seq in range(N):
            nuc = int(msa[seq, col])
            if nuc != gap_char:
                profile[nuc] = profile.get(nuc, 0) + 1
                ungapped_counts[col] += 1
        profiles.append(profile)
    return profiles, ungapped_counts


def column_emission_logprob(profile, pi=None):
    """Log probability of a column profile under equilibrium distribution."""
    if pi is None:
        log_pi = np.full(N_NUC, np.log(1.0 / N_NUC))
    else:
        log_pi = np.log(np.maximum(pi, 1e-30))
    lp = 0.0
    for nuc, count in profile.items():
        if 0 <= nuc < N_NUC:
            lp += count * log_pi[nuc]
    return lp


def _index_wptt_rules(transducer):
    """Pre-index WPTT rules by type and source state.

    Returns dict of rule indices keyed by type.
    """
    idx = {
        'match': {},      # src -> [(in_tok, out_tok, dst, weight)]
        'delete': {},     # src -> [(in_tok, dst, weight)]
        'insert': {},     # src -> [(out_tok, dst, weight)]
        'ready': {},      # src -> [(dst, weight)]
        'bif': {},        # src -> [(dst_l, dst_r, weight)]
        'eps': {},        # src -> [(weight,)]
        'match_by_in': {},  # (src, in_tok) -> [(out_tok, dst, weight)]
        'match_by_out': {},  # (src, out_tok) -> [(in_tok, dst, weight)]
        'insert_by_out': {},  # (src, out_tok) -> [(dst, weight)]
        # Token-only indices (no src): avoids iterating over all WPTT states
        'all_match_by_out': {},  # out_tok -> [(src, in_tok, dst, weight)]
        'all_insert_by_out': {},  # out_tok -> [(src, dst, weight)]
        # Ready/delete as flat lists (small, iterate directly)
        'ready_list': [],  # [(src, dst, weight)]
        'delete_list': [],  # [(src, in_tok, dst, weight)]
        # Delete rules indexed by input token type (L, R, LR)
        'delete_by_type': {'L': [], 'R': [], 'LR': []},
    }
    for r in transducer.rules:
        if r.rule_type == 'match':
            idx['match'].setdefault(r.src, []).append(
                (r.input_token, r.output_token, r.dst, r.weight))
            idx['match_by_in'].setdefault((r.src, r.input_token), []).append(
                (r.output_token, r.dst, r.weight))
            idx['match_by_out'].setdefault((r.src, r.output_token), []).append(
                (r.input_token, r.dst, r.weight))
            idx['all_match_by_out'].setdefault(r.output_token, []).append(
                (r.src, r.input_token, r.dst, r.weight))
        elif r.rule_type == 'delete':
            idx['delete'].setdefault(r.src, []).append(
                (r.input_token, r.dst, r.weight))
            idx['delete_list'].append((r.src, r.input_token, r.dst, r.weight))
            del_type = decode_terminal(r.input_token)[0]
            idx['delete_by_type'][del_type].append(
                (r.src, r.input_token, r.dst, r.weight))
        elif r.rule_type == 'insert':
            idx['insert'].setdefault(r.src, []).append(
                (r.output_token, r.dst, r.weight))
            idx['insert_by_out'].setdefault((r.src, r.output_token), []).append(
                (r.dst, r.weight))
            idx['all_insert_by_out'].setdefault(r.output_token, []).append(
                (r.src, r.dst, r.weight))
        elif r.rule_type == 'ready':
            idx['ready'].setdefault(r.src, []).append(
                (r.dst, r.weight))
            idx['ready_list'].append((r.src, r.dst, r.weight))
        elif r.rule_type == 'bifurcation':
            idx['bif'].setdefault(r.src, []).append(
                (r.dst_left, r.dst_right, r.weight))
        elif r.rule_type == 'epsilon':
            idx['eps'].setdefault(r.src, []).append(
                (r.weight,))
    return idx


def leaf_conditional_likelihoods(transducer, leaf_seq, wptt_idx,
                                 band_width=None, guide_pairs=None,
                                 consensus_structure=None,
                                 mode='inside'):
    """Compute conditional likelihoods at a leaf node.

    CL[w, i, j] = sum (or max) over recognizer states and WPTT output of
    P(leaf_seq[i:j] | WPTT input state w)

    Args:
        transducer: WPTT object
        leaf_seq: (L_leaf,) array of nucleotide indices for this leaf
        wptt_idx: pre-indexed WPTT rules from _index_wptt_rules
        band_width: optional span banding width
        guide_pairs: optional guide structure pairs
        consensus_structure: optional dot-bracket string
        mode: 'inside' (logsumexp) or 'cyk' (max)

    Returns:
        cl: dict mapping (w, i, j) -> log probability
    """
    rec = build_leaf_recognizer(
        leaf_seq, band_width=band_width, guide_pairs=guide_pairs,
        consensus_structure=consensus_structure)

    L = len(leaf_seq)
    NEG_INF = -1e30
    n_trans = transducer.n_states

    # Full WPTT×Recognizer Inside table
    log_I = {}
    # Per-span index: (i, j) -> set of (w, r) pairs with entries
    span_entries = {}
    use_max = (mode == 'cyk')

    def _get(w, r, i, j):
        return log_I.get((w, r, i, j), NEG_INF)

    def _set(w, r, i, j, val):
        old = log_I.get((w, r, i, j), NEG_INF)
        if use_max:
            log_I[(w, r, i, j)] = max(old, val)
        else:
            log_I[(w, r, i, j)] = np.logaddexp(old, val)
        span_entries.setdefault((i, j), set()).add((w, r))

    rec_match = {}
    rec_bif = {}
    rec_eps = {}
    for r in rec.rules:
        if r.rule_type == 'match':
            rec_match.setdefault(r.src, []).append(
                (r.input_token, r.dst, r.weight))
        elif r.rule_type == 'bifurcation':
            rec_bif.setdefault(r.src, []).append(
                (r.dst_left, r.dst_right, r.weight))
        elif r.rule_type == 'epsilon':
            rec_eps.setdefault(r.src, []).append((r.weight,))

    trans_ready = wptt_idx['ready']
    trans_delete = wptt_idx['delete']
    trans_match_by_out = wptt_idx['match_by_out']
    trans_insert_by_out = wptt_idx['insert_by_out']
    trans_bif = wptt_idx['bif']
    trans_eps = wptt_idx['eps']
    all_match_by_out = wptt_idx['all_match_by_out']
    all_insert_by_out = wptt_idx['all_insert_by_out']
    ready_list = wptt_idx['ready_list']
    delete_list = wptt_idx['delete_list']
    n_rec = rec.n_states
    rec_state_by_span = rec.span_to_state

    # Pre-compute log weights for ready and delete rules
    ready_log = [(s, d, np.log(w)) for s, d, w in ready_list]
    delete_log = [(s, it, d, np.log(w)) for s, it, d, w in delete_list]

    # Track populated (w, r) pairs per span for faster closure
    def _unary_closure(i, j):
        # Collect populated entries for this span
        populated = set(span_entries.get((i, j), set()))

        # Index populated entries by WPTT state for fast lookup
        by_wptt = {}
        for w, r in populated:
            by_wptt.setdefault(w, []).append(r)

        for iteration in range(50):
            changed = False
            for w_src, w_dst, log_wt in ready_log:
                for r in by_wptt.get(w_dst, []):
                    val = log_I.get((w_dst, r, i, j), NEG_INF)
                    if val <= NEG_INF:
                        continue
                    new_val = log_wt + val
                    old = log_I.get((w_src, r, i, j), NEG_INF)
                    if new_val > old + 1e-10 or old <= NEG_INF:
                        _set(w_src, r, i, j, new_val)
                        if (w_src, r) not in populated:
                            populated.add((w_src, r))
                            by_wptt.setdefault(w_src, []).append(r)
                        changed = True
            for w_src, in_tok, w_dst, log_wt in delete_log:
                for r in by_wptt.get(w_dst, []):
                    val = log_I.get((w_dst, r, i, j), NEG_INF)
                    if val <= NEG_INF:
                        continue
                    new_val = log_wt + val
                    old = log_I.get((w_src, r, i, j), NEG_INF)
                    if new_val > old + 1e-10 or old <= NEG_INF:
                        _set(w_src, r, i, j, new_val)
                        if (w_src, r) not in populated:
                            populated.add((w_src, r))
                            by_wptt.setdefault(w_src, []).append(r)
                        changed = True
            if not changed:
                break

    # Pre-compute log weights for token-indexed WPTT rules
    match_log_by_out = {}
    for out_tok, rules in all_match_by_out.items():
        match_log_by_out[out_tok] = [
            (src, in_tok, dst, np.log(w)) for src, in_tok, dst, w in rules
        ]
    insert_log_by_out = {}
    for out_tok, rules in all_insert_by_out.items():
        insert_log_by_out[out_tok] = [
            (src, dst, np.log(w)) for src, dst, w in rules
        ]

    # Pre-compute epsilon log weights
    eps_log = {}
    for w, rules in trans_eps.items():
        for w_t, in rules:
            eps_log[w] = np.log(w_t)

    # Pre-compute bif log weights: flat list
    bif_log = []
    for w, rules in trans_bif.items():
        for w_l, w_r, w_t in rules:
            bif_log.append((w, w_l, w_r, np.log(w_t)))

    # Base case: epsilon spans
    for pos in range(L + 1):
        r_st = rec_state_by_span.get((pos, pos))
        if r_st is None:
            continue
        for r_w, in rec_eps.get(r_st, []):
            log_rw = np.log(r_w)
            for w, log_wt in eps_log.items():
                _set(w, r_st, pos, pos, log_rw + log_wt)
        _unary_closure(pos, pos)

    # Fill spans
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span

            r_st = rec_state_by_span.get((i, j))
            if r_st is None:
                continue

            # Match + Insert: WPTT produces output, recognizer consumes
            for rec_tok, r_dst, r_w in rec_match.get(r_st, []):
                out_type, out_nucs = decode_terminal(rec_tok)
                if out_type == 'L':
                    if int(leaf_seq[i]) != out_nucs[0]:
                        continue
                    ci, cj = i + 1, j
                elif out_type == 'R':
                    if int(leaf_seq[j - 1]) != out_nucs[0]:
                        continue
                    ci, cj = i, j - 1
                elif out_type == 'LR':
                    if span < 2:
                        continue
                    if (int(leaf_seq[i]) != out_nucs[0] or
                            int(leaf_seq[j - 1]) != out_nucs[1]):
                        continue
                    ci, cj = i + 1, j - 1
                else:
                    continue

                log_rw = np.log(r_w)

                # Match rules (WPTT ready -> match)
                for w_src, in_tok, w_dst, log_wt in match_log_by_out.get(
                        rec_tok, []):
                    val_child = _get(w_dst, r_dst, ci, cj)
                    if val_child <= NEG_INF:
                        continue
                    _set(w_src, r_st, i, j,
                         log_rw + log_wt + val_child)

                # Insert rules (WPTT non-ready -> insert)
                for w_src, w_dst, log_wt in insert_log_by_out.get(
                        rec_tok, []):
                    val_child = _get(w_dst, r_dst, ci, cj)
                    if val_child <= NEG_INF:
                        continue
                    _set(w_src, r_st, i, j,
                         log_rw + log_wt + val_child)

            # Bifurcation
            for r_l, r_r, r_w in rec_bif.get(r_st, []):
                span_l = rec.states[r_l].span
                k = span_l[1]
                log_rw = np.log(r_w)
                for w, w_l, w_r, log_wt in bif_log:
                    vl = _get(w_l, r_l, i, k)
                    vr = _get(w_r, r_r, k, j)
                    if vl <= NEG_INF or vr <= NEG_INF:
                        continue
                    _set(w, r_st, i, j,
                         log_rw + log_wt + vl + vr)

            _unary_closure(i, j)

    # Marginalize out recognizer states to get CL[w, i, j]
    cl = {}
    combine = max if use_max else np.logaddexp
    for (w, r, i, j), val in log_I.items():
        if val <= NEG_INF:
            continue
        key = (w, i, j)
        old = cl.get(key, NEG_INF)
        cl[key] = combine(old, val)

    return cl


def leaf_cl_msa(transducer, msa_row, wptt_idx,
                band_width=None, guide_pairs=None,
                consensus_structure=None, mode='inside',
                return_inside_table=False):
    """Compute conditional likelihoods at a leaf, indexed by MSA columns.

    Unlike leaf_conditional_likelihoods which indexes spans by ungapped leaf
    positions, this function indexes spans by MSA columns, properly handling
    gap columns via WPTT delete transitions.

    For non-gap boundary columns: WPTT match/insert + recognizer validation.
    For gap boundary columns: WPTT delete (marginalizing over ancestor
    nucleotide), span shrinks by one column, recognizer unchanged.

    Args:
        transducer: WPTT object
        msa_row: (L,) array of nucleotide indices (-1 for gaps)
        wptt_idx: pre-indexed WPTT rules from _index_wptt_rules
        band_width: optional span banding width (in MSA columns)
        guide_pairs: optional guide structure pairs (in MSA columns)
        consensus_structure: optional dot-bracket string
        mode: 'inside' (logsumexp) or 'cyk' (max)
        return_inside_table: if True, also return the full (w, r, i, j)
            Inside table and auxiliary data needed for Outside computation

    Returns:
        cl: dict mapping (w, i, j) -> log probability,
            where i, j are MSA column indices
        tb: (only if mode='cyk') dict mapping (w, i, j) -> traceback entry.
        inside_info: (only if return_inside_table=True) dict with:
            'log_I': full Inside table (w, r, i, j) -> log_prob
            'rec': Recognizer object
            'nongap_before': array mapping column -> leaf position
            'is_gap': boolean array of gap columns
    """
    msa_row = np.asarray(msa_row)
    L = len(msa_row)

    # Build gap mapping
    is_gap = (msa_row < 0)
    ungapped = msa_row[~is_gap]
    L_leaf = len(ungapped)

    # nongap_before[c] = number of non-gap columns in [0, c)
    nongap_before = np.zeros(L + 1, dtype=int)
    for c in range(L):
        nongap_before[c + 1] = nongap_before[c] + (0 if is_gap[c] else 1)

    # leaf_to_col[k] = MSA column of k-th non-gap nucleotide
    nongap_cols = np.where(~is_gap)[0]
    # Append L as sentinel for k = L_leaf
    leaf_to_col = np.append(nongap_cols, L)

    # Build recognizer from ungapped sequence (spans over leaf positions)
    rec = build_leaf_recognizer(
        ungapped, band_width=band_width, guide_pairs=guide_pairs,
        consensus_structure=consensus_structure)

    NEG_INF = -1e30
    n_trans = transducer.n_states

    use_max = (mode == 'cyk')
    log_I = {}
    tb_I = {} if use_max else None
    span_entries = {}

    def _get(w, r, i, j):
        return log_I.get((w, r, i, j), NEG_INF)

    def _set(w, r, i, j, val, tb_entry=None):
        old = log_I.get((w, r, i, j), NEG_INF)
        if use_max:
            if val > old:
                log_I[(w, r, i, j)] = val
                if tb_entry is not None:
                    tb_I[(w, r, i, j)] = tb_entry
        else:
            log_I[(w, r, i, j)] = np.logaddexp(old, val)
        span_entries.setdefault((i, j), set()).add((w, r))

    # Index recognizer rules
    rec_match = {}
    rec_bif = {}
    rec_eps = {}
    for r in rec.rules:
        if r.rule_type == 'match':
            rec_match.setdefault(r.src, []).append(
                (r.input_token, r.dst, r.weight))
        elif r.rule_type == 'bifurcation':
            rec_bif.setdefault(r.src, []).append(
                (r.dst_left, r.dst_right, r.weight))
        elif r.rule_type == 'epsilon':
            rec_eps.setdefault(r.src, []).append((r.weight,))

    trans_bif = wptt_idx['bif']
    trans_eps = wptt_idx['eps']
    all_match_by_out = wptt_idx['all_match_by_out']
    all_insert_by_out = wptt_idx['all_insert_by_out']
    ready_list = wptt_idx['ready_list']
    delete_by_type = wptt_idx['delete_by_type']
    n_rec = rec.n_states
    rec_state_by_span = rec.span_to_state

    # Pre-compute log weights
    ready_log = [(s, d, np.log(w)) for s, d, w in ready_list]
    match_log_by_out = {}
    for out_tok, rules in all_match_by_out.items():
        match_log_by_out[out_tok] = [
            (src, in_tok, dst, np.log(w)) for src, in_tok, dst, w in rules
        ]
    insert_log_by_out = {}
    for out_tok, rules in all_insert_by_out.items():
        insert_log_by_out[out_tok] = [
            (src, dst, np.log(w)) for src, dst, w in rules
        ]
    eps_log = {}
    for w, rules in trans_eps.items():
        for w_t, in rules:
            eps_log[w] = np.log(w_t)
    bif_log = []
    for w, rules in trans_bif.items():
        for w_l, w_r, w_t in rules:
            bif_log.append((w, w_l, w_r, np.log(w_t)))

    # Pre-compute delete rules by type with log weights
    del_L_log = [(s, it, d, np.log(w)) for s, it, d, w in delete_by_type['L']]
    del_R_log = [(s, it, d, np.log(w)) for s, it, d, w in delete_by_type['R']]
    del_LR_log = [(s, it, d, np.log(w)) for s, it, d, w in delete_by_type['LR']]

    def _ready_closure(i, j):
        """Apply only ready transitions (NOT delete — deletes are explicit)."""
        populated = set(span_entries.get((i, j), set()))
        by_wptt = {}
        for w, r in populated:
            by_wptt.setdefault(w, []).append(r)

        for iteration in range(50):
            changed = False
            for w_src, w_dst, log_wt in ready_log:
                for r in by_wptt.get(w_dst, []):
                    val = log_I.get((w_dst, r, i, j), NEG_INF)
                    if val <= NEG_INF:
                        continue
                    new_val = log_wt + val
                    old = log_I.get((w_src, r, i, j), NEG_INF)
                    if use_max:
                        if new_val > old:
                            _set(w_src, r, i, j, new_val,
                                 ('ready', w_src, r, i, j, w_dst))
                            if (w_src, r) not in populated:
                                populated.add((w_src, r))
                                by_wptt.setdefault(w_src, []).append(r)
                            changed = True
                    elif new_val > old + 1e-10 or old <= NEG_INF:
                        _set(w_src, r, i, j, new_val)
                        if (w_src, r) not in populated:
                            populated.add((w_src, r))
                            by_wptt.setdefault(w_src, []).append(r)
                        changed = True
            if not changed:
                break

    def _leaf_span(i, j):
        """Map MSA column span [i, j) to leaf position span."""
        return nongap_before[i], nongap_before[j]

    # === Base case: empty MSA spans ===
    for pos in range(L + 1):
        li, lj = _leaf_span(pos, pos)
        r_st = rec_state_by_span.get((li, lj))
        if r_st is None:
            continue
        for r_w, in rec_eps.get(r_st, []):
            log_rw = np.log(r_w)
            for w, log_wt in eps_log.items():
                _set(w, r_st, pos, pos, log_rw + log_wt,
                     ('eps', w, r_st, pos, pos))
        _ready_closure(pos, pos)

    # === Fill spans ===
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span

            li, lj = _leaf_span(i, j)
            r_st = rec_state_by_span.get((li, lj))
            if r_st is None:
                continue

            # --- Gap delete at left boundary (column i) ---
            if is_gap[i]:
                # Ancestor emitted L(a) at column i, WPTT deletes.
                # Span shrinks to [i+1, j), recognizer stays at same state.
                for w_src, in_tok, w_dst, log_wt in del_L_log:
                    val_child = _get(w_dst, r_st, i + 1, j)
                    if val_child <= NEG_INF:
                        continue
                    _set(w_src, r_st, i, j, log_wt + val_child,
                         ('gap_del_L', w_src, r_st, i, j, w_dst))

            # --- Gap delete at right boundary (column j-1) ---
            if is_gap[j - 1]:
                for w_src, in_tok, w_dst, log_wt in del_R_log:
                    val_child = _get(w_dst, r_st, i, j - 1)
                    if val_child <= NEG_INF:
                        continue
                    _set(w_src, r_st, i, j, log_wt + val_child,
                         ('gap_del_R', w_src, r_st, i, j, w_dst))

            # --- Gap delete at both boundaries (pair delete) ---
            if span >= 2 and is_gap[i] and is_gap[j - 1]:
                for w_src, in_tok, w_dst, log_wt in del_LR_log:
                    val_child = _get(w_dst, r_st, i + 1, j - 1)
                    if val_child <= NEG_INF:
                        continue
                    _set(w_src, r_st, i, j, log_wt + val_child,
                         ('gap_del_LR', w_src, r_st, i, j, w_dst))

            # --- Non-gap: match/insert (recognizer validates) ---
            for rec_tok, r_dst, r_w in rec_match.get(r_st, []):
                out_type, out_nucs = decode_terminal(rec_tok)

                if out_type == 'L':
                    if is_gap[i]:
                        continue
                    if int(ungapped[li]) != out_nucs[0]:
                        continue
                    ci, cj = i + 1, j
                elif out_type == 'R':
                    if is_gap[j - 1]:
                        continue
                    if int(ungapped[lj - 1]) != out_nucs[0]:
                        continue
                    ci, cj = i, j - 1
                elif out_type == 'LR':
                    if span < 2 or is_gap[i] or is_gap[j - 1]:
                        continue
                    if (int(ungapped[li]) != out_nucs[0] or
                            int(ungapped[lj - 1]) != out_nucs[1]):
                        continue
                    ci, cj = i + 1, j - 1
                else:
                    continue

                log_rw = np.log(r_w)

                # Match rules
                for w_src, in_tok, w_dst, log_wt in match_log_by_out.get(
                        rec_tok, []):
                    val_child = _get(w_dst, r_dst, ci, cj)
                    if val_child <= NEG_INF:
                        continue
                    _set(w_src, r_st, i, j,
                         log_rw + log_wt + val_child,
                         ('match', w_src, r_st, i, j, rec_tok,
                          w_dst, r_dst, ci, cj))

                # Insert rules
                for w_src, w_dst, log_wt in insert_log_by_out.get(
                        rec_tok, []):
                    val_child = _get(w_dst, r_dst, ci, cj)
                    if val_child <= NEG_INF:
                        continue
                    _set(w_src, r_st, i, j,
                         log_rw + log_wt + val_child,
                         ('insert', w_src, r_st, i, j, rec_tok,
                          w_dst, r_dst, ci, cj))

            # --- Bifurcation ---
            # Recognizer bifurcates at leaf position k.
            # MSA split at leaf_to_col[k].
            for r_l, r_r, r_w in rec_bif.get(r_st, []):
                leaf_k = rec.states[r_l].span[1]  # end of left child
                c = int(leaf_to_col[leaf_k])  # MSA column for split
                log_rw = np.log(r_w)
                for w, w_l, w_r, log_wt in bif_log:
                    vl = _get(w_l, r_l, i, c)
                    vr = _get(w_r, r_r, c, j)
                    if vl <= NEG_INF or vr <= NEG_INF:
                        continue
                    _set(w, r_st, i, j,
                         log_rw + log_wt + vl + vr,
                         ('bif', w, r_st, i, j, w_l, w_r,
                          r_l, r_r, c))

            _ready_closure(i, j)

    # Marginalize out recognizer states
    cl = {}
    tb = {} if use_max else None
    combine = max if use_max else np.logaddexp
    for (w, r, i, j), val in log_I.items():
        if val <= NEG_INF:
            continue
        key = (w, i, j)
        old = cl.get(key, NEG_INF)
        if use_max:
            if val > old:
                cl[key] = val
                tb[key] = tb_I.get((w, r, i, j))
        else:
            cl[key] = np.logaddexp(old, val)

    inside_info = None
    if return_inside_table:
        inside_info = {
            'log_I': log_I,
            'rec': rec,
            'nongap_before': nongap_before,
            'is_gap': is_gap,
        }

    if use_max:
        if return_inside_table:
            return cl, tb, inside_info
        return cl, tb
    if return_inside_table:
        return cl, inside_info
    return cl


def _beam_prune_cl(cl, beam_width, L):
    """Prune conditional likelihoods to keep top beam_width WPTT states per span.

    Args:
        cl: dict (w, i, j) -> log_prob
        beam_width: max number of WPTT states to keep per (i, j)
        L: sequence length (for span enumeration)

    Returns:
        pruned cl dict
    """
    if beam_width is None or beam_width >= N_WPTT_STATES:
        return cl

    # Group by span
    by_span = {}
    for (w, i, j), val in cl.items():
        by_span.setdefault((i, j), []).append((val, w))

    pruned = {}
    for (i, j), entries in by_span.items():
        entries.sort(reverse=True)
        for rank, (val, w) in enumerate(entries):
            if rank >= beam_width:
                break
            pruned[(w, i, j)] = val

    return pruned


def leaf_outside_msa(transducer, msa_row, wptt_idx, inside_info):
    """Compute Outside table for a leaf, given Inside table from leaf_cl_msa.

    The Outside O[w, r, i, j] represents the probability of all data
    outside span [i, j) given WPTT state w and recognizer state r at
    the boundary of that span.

    Posterior probability of state (w, r) at span [i, j):
        P(w, r, i, j | data) = exp(I[w,r,i,j] + O[w,r,i,j] - total)

    Args:
        transducer: WPTT object (same as used in Inside)
        msa_row: (L,) array of nucleotide indices (-1 for gaps)
        wptt_idx: pre-indexed WPTT rules from _index_wptt_rules
        inside_info: dict from leaf_cl_msa(return_inside_table=True)

    Returns:
        log_O: dict mapping (w, r, i, j) -> log outside probability
    """
    msa_row = np.asarray(msa_row)
    L = len(msa_row)
    NEG_INF = -1e30

    log_I = inside_info['log_I']
    rec = inside_info['rec']
    nongap_before = inside_info['nongap_before']
    is_gap = inside_info['is_gap']

    n_trans = transducer.n_states

    log_O = {}

    def _get_I(w, r, i, j):
        return log_I.get((w, r, i, j), NEG_INF)

    def _get_O(w, r, i, j):
        return log_O.get((w, r, i, j), NEG_INF)

    def _set_O(w, r, i, j, val):
        old = log_O.get((w, r, i, j), NEG_INF)
        log_O[(w, r, i, j)] = np.logaddexp(old, val)

    # Index recognizer rules
    rec_match = {}
    rec_bif = {}
    rec_eps = {}
    for r in rec.rules:
        if r.rule_type == 'match':
            rec_match.setdefault(r.src, []).append(
                (r.input_token, r.dst, r.weight))
        elif r.rule_type == 'bifurcation':
            rec_bif.setdefault(r.src, []).append(
                (r.dst_left, r.dst_right, r.weight))
        elif r.rule_type == 'epsilon':
            rec_eps.setdefault(r.src, []).append((r.weight,))

    # Also need reverse recognizer index: which src states have dst as target
    rec_match_by_dst = {}
    for src, rules in rec_match.items():
        for in_tok, dst, wt in rules:
            rec_match_by_dst.setdefault(dst, []).append((src, in_tok, wt))

    rec_bif_by_children = {}
    for src, rules in rec_bif.items():
        for dl, dr, wt in rules:
            rec_bif_by_children.setdefault((dl, dr), []).append((src, wt))

    trans_bif = wptt_idx['bif']
    trans_eps = wptt_idx['eps']
    all_match_by_out = wptt_idx['all_match_by_out']
    all_insert_by_out = wptt_idx['all_insert_by_out']
    ready_list = wptt_idx['ready_list']
    delete_by_type = wptt_idx['delete_by_type']
    rec_state_by_span = rec.span_to_state

    # Pre-compute log weights (same as Inside)
    ready_log = [(s, d, np.log(w)) for s, d, w in ready_list]
    match_log_by_out = {}
    for out_tok, rules in all_match_by_out.items():
        match_log_by_out[out_tok] = [
            (src, in_tok, dst, np.log(w)) for src, in_tok, dst, w in rules
        ]
    insert_log_by_out = {}
    for out_tok, rules in all_insert_by_out.items():
        insert_log_by_out[out_tok] = [
            (src, dst, np.log(w)) for src, dst, w in rules
        ]
    eps_log = {}
    for w, rules in trans_eps.items():
        for w_t, in rules:
            eps_log[w] = np.log(w_t)
    bif_log = []
    for w, rules in trans_bif.items():
        for w_l, w_r, w_t in rules:
            bif_log.append((w, w_l, w_r, np.log(w_t)))

    del_L_log = [(s, it, d, np.log(w)) for s, it, d, w in delete_by_type['L']]
    del_R_log = [(s, it, d, np.log(w)) for s, it, d, w in delete_by_type['R']]
    del_LR_log = [(s, it, d, np.log(w)) for s, it, d, w in delete_by_type['LR']]

    def _leaf_span(i, j):
        return nongap_before[i], nongap_before[j]

    # Collect populated entries from Inside, grouped by span
    inside_by_span = {}
    for (w, r, i, j), val in log_I.items():
        if val > NEG_INF:
            inside_by_span.setdefault((i, j), set()).add((w, r))

    def _reverse_ready_closure(i, j):
        """Propagate Outside through ready transitions (reverse direction).
        Ready: w_src(non-ready) -> w_dst(ready), weight w_t
        Outside reverse: O[w_dst] += w_t * O[w_src]
        """
        populated = set()
        for (w, r, ci, cj), val in log_O.items():
            if ci == i and cj == j and val > NEG_INF:
                populated.add((w, r))

        for iteration in range(50):
            changed = False
            for w_src, w_dst, log_wt in ready_log:
                # In Inside: I[w_src, r, i, j] += w_t * I[w_dst, r, i, j]
                # In Outside: O[w_dst, r, i, j] += w_t * O[w_src, r, i, j]
                for r in {r for ww, r in populated if ww == w_src}:
                    oval_src = _get_O(w_src, r, i, j)
                    if oval_src <= NEG_INF:
                        continue
                    new_val = log_wt + oval_src
                    old = _get_O(w_dst, r, i, j)
                    if new_val > old + 1e-10 or old <= NEG_INF:
                        _set_O(w_dst, r, i, j, new_val)
                        if (w_dst, r) not in populated:
                            populated.add((w_dst, r))
                        changed = True
            if not changed:
                break

    # === Base case: Outside for root span [0, L) ===
    # Only START state has O = 0 (log 1) at root span.
    # The model starts from WPTT_START; other states are only reachable
    # through ready transitions from START.
    for (w, r) in inside_by_span.get((0, L), set()):
        if w == WPTT_START:
            log_O[(w, r, 0, L)] = 0.0

    _reverse_ready_closure(0, L)

    # === Fill spans from largest to smallest ===
    # For each span: O values are already complete (all larger spans
    # have contributed). Apply reverse ready closure, then propagate
    # to sub-spans.
    for span in range(L, 0, -1):
        for i in range(L - span + 1):
            j = i + span

            li, lj = _leaf_span(i, j)
            r_st = rec_state_by_span.get((li, lj))
            if r_st is None:
                continue

            # Apply reverse ready closure: O values at this span from
            # all larger spans are now accumulated; propagate through
            # span-preserving ready transitions.
            if span < L:
                _reverse_ready_closure(i, j)

            # For each populated (w, r) at this span with nonzero Outside,
            # propagate to sub-spans

            populated = inside_by_span.get((i, j), set())
            if not populated:
                continue

            # --- Gap delete at left boundary ---
            if is_gap[i]:
                for w_src, in_tok, w_dst, log_wt in del_L_log:
                    for r in {r for ww, r in populated if ww == w_src}:
                        oval = _get_O(w_src, r, i, j)
                        if oval <= NEG_INF:
                            continue
                        _set_O(w_dst, r, i + 1, j, log_wt + oval)

            # --- Gap delete at right boundary ---
            if j > i and is_gap[j - 1]:
                for w_src, in_tok, w_dst, log_wt in del_R_log:
                    for r in {r for ww, r in populated if ww == w_src}:
                        oval = _get_O(w_src, r, i, j)
                        if oval <= NEG_INF:
                            continue
                        _set_O(w_dst, r, i, j - 1, log_wt + oval)

            # --- Gap delete LR ---
            if span >= 2 and is_gap[i] and is_gap[j - 1]:
                for w_src, in_tok, w_dst, log_wt in del_LR_log:
                    for r in {r for ww, r in populated if ww == w_src}:
                        oval = _get_O(w_src, r, i, j)
                        if oval <= NEG_INF:
                            continue
                        _set_O(w_dst, r, i + 1, j - 1, log_wt + oval)

            if not is_gap[i]:
                target_nuc_i = int(msa_row[i])
                target_out_l = left_terminal(target_nuc_i)

                # --- L-match: I[w_src, r_src, i, j] += w_m * I[w_dst, r_dst, i+1, j]
                # Outside: O[w_dst, r_dst, i+1, j] += w_m * O[w_src, r_src, i, j]
                for w_src, in_tok, w_dst, log_wt in match_log_by_out.get(
                        target_out_l, []):
                    for r_src in {r for ww, r in populated if ww == w_src}:
                        oval = _get_O(w_src, r_src, i, j)
                        if oval <= NEG_INF:
                            continue
                        # Find recognizer transitions from r_src matching target_out_l
                        for rec_tok, r_dst, r_wt in rec_match.get(r_src, []):
                            if rec_tok != target_out_l:
                                continue
                            total_wt = log_wt + np.log(r_wt) + oval
                            _set_O(w_dst, r_dst, i + 1, j, total_wt)

                # --- L-insert: I[w_src, r_src, i, j] += w_ins * I[w_dst, r_dst, i+1, j]
                for w_src, w_dst, log_wt in insert_log_by_out.get(
                        target_out_l, []):
                    for r_src in {r for ww, r in populated if ww == w_src}:
                        oval = _get_O(w_src, r_src, i, j)
                        if oval <= NEG_INF:
                            continue
                        for rec_tok, r_dst, r_wt in rec_match.get(r_src, []):
                            if rec_tok != target_out_l:
                                continue
                            total_wt = log_wt + np.log(r_wt) + oval
                            _set_O(w_dst, r_dst, i + 1, j, total_wt)

            if j > i and not is_gap[j - 1]:
                target_nuc_j = int(msa_row[j - 1])
                target_out_r = right_terminal(target_nuc_j)

                # --- R-match ---
                for w_src, in_tok, w_dst, log_wt in match_log_by_out.get(
                        target_out_r, []):
                    for r_src in {r for ww, r in populated if ww == w_src}:
                        oval = _get_O(w_src, r_src, i, j)
                        if oval <= NEG_INF:
                            continue
                        for rec_tok, r_dst, r_wt in rec_match.get(r_src, []):
                            if rec_tok != target_out_r:
                                continue
                            total_wt = log_wt + np.log(r_wt) + oval
                            _set_O(w_dst, r_dst, i, j - 1, total_wt)

                # --- R-insert ---
                for w_src, w_dst, log_wt in insert_log_by_out.get(
                        target_out_r, []):
                    for r_src in {r for ww, r in populated if ww == w_src}:
                        oval = _get_O(w_src, r_src, i, j)
                        if oval <= NEG_INF:
                            continue
                        for rec_tok, r_dst, r_wt in rec_match.get(r_src, []):
                            if rec_tok != target_out_r:
                                continue
                            total_wt = log_wt + np.log(r_wt) + oval
                            _set_O(w_dst, r_dst, i, j - 1, total_wt)

            # --- LR-match ---
            if span >= 2 and not is_gap[i] and not is_gap[j - 1]:
                target_nuc_i = int(msa_row[i])
                target_nuc_j = int(msa_row[j - 1])
                target_pair = pair_terminal(target_nuc_i, target_nuc_j)

                for w_src, in_tok, w_dst, log_wt in match_log_by_out.get(
                        target_pair, []):
                    for r_src in {r for ww, r in populated if ww == w_src}:
                        oval = _get_O(w_src, r_src, i, j)
                        if oval <= NEG_INF:
                            continue
                        for rec_tok, r_dst, r_wt in rec_match.get(r_src, []):
                            if rec_tok != target_pair:
                                continue
                            total_wt = log_wt + np.log(r_wt) + oval
                            _set_O(w_dst, r_dst, i + 1, j - 1, total_wt)

            # --- Bifurcation ---
            # I[w, r, i, j] += w_bif * w_rec_bif * I[w_l, r_l, i, k] * I[w_r, r_r, k, j]
            # O[w_l, r_l, i, k] += w_bif * w_rec_bif * O[w, r, i, j] * I[w_r, r_r, k, j]
            # O[w_r, r_r, k, j] += w_bif * w_rec_bif * O[w, r, i, j] * I[w_l, r_l, i, k]
            for w_src, w_l, w_r, log_wt in bif_log:
                for r_src in {r for ww, r in populated if ww == w_src}:
                    oval = _get_O(w_src, r_src, i, j)
                    if oval <= NEG_INF:
                        continue
                    for r_l, r_r, r_wt in rec_bif.get(r_src, []):
                        log_combo = log_wt + np.log(r_wt) + oval
                        for k in range(i, j + 1):
                            ival_l = _get_I(w_l, r_l, i, k)
                            ival_r = _get_I(w_r, r_r, k, j)
                            if ival_l > NEG_INF and ival_r > NEG_INF:
                                _set_O(w_l, r_l, i, k,
                                       log_combo + ival_r)
                                _set_O(w_r, r_r, k, j,
                                       log_combo + ival_l)

    # Apply reverse ready closure for empty spans (span=0)
    for i in range(L + 1):
        _reverse_ready_closure(i, i)

    return log_O


def posterior_basepair_probs(log_I, log_O, msa_row, total_log_prob,
                            wptt_idx, inside_info):
    """Compute posterior basepair probabilities from Inside-Outside tables.

    P(columns i and j-1 are base-paired | data) is the probability that
    an LR-match transition produced output at positions i and j-1.

    This is computed as:
        sum_{w,r} O[w,r,i,j] × w_LR_match × w_rec × I[w',r',i+1,j-1] / total

    for all WPTT LR-match transitions from (w,r) → (w',r').

    Args:
        log_I: Inside table from leaf_cl_msa (w, r, i, j) -> log_prob
        log_O: Outside table from leaf_outside_msa
        msa_row: (L,) array of nucleotide indices
        total_log_prob: log P(data) for normalization
        wptt_idx: pre-indexed WPTT rules
        inside_info: from leaf_cl_msa(return_inside_table=True)

    Returns:
        bp_probs: dict mapping (i, j-1) -> posterior probability that
            column i and column j-1 are base-paired (0-indexed)
    """
    msa_row = np.asarray(msa_row)
    L = len(msa_row)
    NEG_INF = -1e30
    is_gap = inside_info['is_gap']
    rec = inside_info['rec']
    nongap_before = inside_info['nongap_before']

    # Index recognizer match rules
    rec_match = {}
    for r in rec.rules:
        if r.rule_type == 'match':
            rec_match.setdefault(r.src, []).append(
                (r.input_token, r.dst, r.weight))

    all_match_by_out = wptt_idx['all_match_by_out']
    rec_state_by_span = rec.span_to_state

    # Pre-compute LR-match transitions with log weights
    lr_match_log = {}
    for out_tok, rules in all_match_by_out.items():
        tok_type, _ = decode_terminal(out_tok)
        if tok_type == 'LR':
            lr_match_log[out_tok] = [
                (src, in_tok, dst, np.log(w))
                for src, in_tok, dst, w in rules
            ]

    # Collect Inside entries grouped by span
    inside_by_span = {}
    for (w, r, i, j), val in log_I.items():
        if val > NEG_INF:
            inside_by_span.setdefault((i, j), set()).add((w, r))

    bp_log_probs = {}  # (i, j-1) -> log posterior probability

    for span in range(2, L + 1):
        for i in range(L - span + 1):
            j = i + span

            # LR-match requires both boundaries to be non-gap
            if is_gap[i] or is_gap[j - 1]:
                continue

            target_nuc_i = int(msa_row[i])
            target_nuc_j = int(msa_row[j - 1])
            target_pair = pair_terminal(target_nuc_i, target_nuc_j)

            if target_pair not in lr_match_log:
                continue

            # Get recognizer state for inner span
            li_inner = nongap_before[i + 1]
            lj_inner = nongap_before[j - 1]
            r_inner = rec_state_by_span.get((li_inner, lj_inner))

            populated = inside_by_span.get((i, j), set())
            if not populated:
                continue

            for w_src, in_tok, w_dst, log_wt in lr_match_log[target_pair]:
                for r_src in {r for ww, r in populated if ww == w_src}:
                    oval = log_O.get((w_src, r_src, i, j), NEG_INF)
                    if oval <= NEG_INF:
                        continue

                    # Find recognizer transitions matching the pair token
                    for rec_tok, r_dst, r_wt in rec_match.get(r_src, []):
                        if rec_tok != target_pair:
                            continue

                        # Inner span Inside value
                        ival_inner = log_I.get(
                            (w_dst, r_dst, i + 1, j - 1), NEG_INF)
                        if ival_inner <= NEG_INF:
                            continue

                        # Posterior contribution
                        log_post = (oval + log_wt + np.log(r_wt)
                                    + ival_inner - total_log_prob)

                        key = (i, j - 1)
                        old = bp_log_probs.get(key, NEG_INF)
                        bp_log_probs[key] = np.logaddexp(old, log_post)

    return {k: np.exp(v) for k, v in bp_log_probs.items()
            if v > NEG_INF}


def posterior_span_probs(log_I, log_O, total_log_prob):
    """Compute posterior span probabilities from Inside-Outside tables.

    P(span [i,j) active | data) = sum_{w,r} exp(I[w,r,i,j] + O[w,r,i,j] - total)

    Args:
        log_I: Inside table
        log_O: Outside table
        total_log_prob: log P(data) for normalization

    Returns:
        span_probs: dict mapping (i, j) -> posterior probability
    """
    NEG_INF = -1e30
    span_log = {}

    for (w, r, i, j), ival in log_I.items():
        if ival <= NEG_INF:
            continue
        oval = log_O.get((w, r, i, j), NEG_INF)
        if oval <= NEG_INF:
            continue
        posterior = ival + oval - total_log_prob
        key = (i, j)
        old = span_log.get(key, NEG_INF)
        span_log[key] = np.logaddexp(old, posterior)

    return {k: np.exp(v) for k, v in span_log.items()
            if v > NEG_INF}


def push_up_internal(cl_left, cl_right, wptt_left, wptt_right,
                     wptt_left_idx, wptt_right_idx,
                     L, beam_width=None):
    """Compute conditional likelihoods at an internal phylo-node.

    CL_k[w, i, j] means "Inside probability for subtree below k, for
    SCFG span [i,j), given the WPTT entering node k is in state w."

    At internal node B with children L, R: SCFG bifurcation resets both
    child WPTTs to START. So the parent's CL at START is simply the
    product of children's CL at START (independent given the same span).

    CL_B[START, i, j] = CL_L[START, i, j] * CL_R[START, i, j]

    Only the START state is populated at the parent. This is correct
    because bifurcation always resets children to START.

    Args:
        cl_left: dict (w, i, j) -> log_prob for left child
        cl_right: dict (w, i, j) -> log_prob for right child
        wptt_left: WPTT for left branch (B→L)
        wptt_right: WPTT for right branch (B→R)
        wptt_left_idx: pre-indexed rules for left WPTT
        wptt_right_idx: pre-indexed rules for right WPTT
        L: MSA length (number of columns)
        beam_width: max WPTT states per span

    Returns:
        cl_parent: dict (w, i, j) -> log_prob for this node
    """
    NEG_INF = -1e30
    cl_parent = {}

    # Collect spans present in left child at START
    spans_left = {}
    for (w, i, j), val in cl_left.items():
        if w == WPTT_START and val > NEG_INF:
            spans_left[(i, j)] = val

    # For each span, multiply left and right START values
    for (i, j), val_l in spans_left.items():
        val_r = cl_right.get((WPTT_START, i, j), NEG_INF)
        if val_r > NEG_INF:
            cl_parent[(WPTT_START, i, j)] = val_l + val_r

    if beam_width is not None:
        cl_parent = _beam_prune_cl(cl_parent, beam_width, L)

    return cl_parent


def beam_inside_phylo(grammar, msa, phylo_tree, transducers,
                      guide_structure=None, beam_width=None,
                      band_width=None, consensus_structure=None):
    """Beam-pruned phylogenetic Inside for SCFG on MSA via WPTT pushing.

    Computes the phylogenetic likelihood of an MSA under the grammar,
    pushing WPTT state up the tree for each Inside span.

    Args:
        grammar: WCFG (order-1 singlet SCFG)
        msa: (N, L) array of nucleotide indices (-1 for gaps)
        phylo_tree: nested tuple representing the phylogenetic tree.
            Leaves are integer indices into msa rows.
            Internal nodes are tuples: (left_child, right_child).
        transducers: dict mapping (parent, child) edges to WPTT objects,
            or a single WPTT for all branches.
        guide_structure: optional dot-bracket string for banding
        beam_width: max WPTT states per span (None = no pruning)
        band_width: max span width for recognizer banding
        consensus_structure: dot-bracket for recognizer fallback paths

    Returns:
        log_prob: log P(MSA | grammar, tree)
        cl_root: dict (w, i, j) -> log_prob at root
        info: dict with per-node CL tables
    """
    msa = np.asarray(msa)
    N_seq, L = msa.shape
    NEG_INF = -1e30

    # Parse guide structure for banding
    guide_pairs = None
    if guide_structure is not None:
        from ..tree.recognizer import parse_dot_bracket
        guide_pairs, _ = parse_dot_bracket(guide_structure)

    # Get a WPTT for a branch (handles both dict and single-WPTT cases)
    def _get_wptt(parent_node, child_node):
        if isinstance(transducers, dict):
            return transducers.get((parent_node, child_node), transducers.get('default'))
        return transducers

    # Cache WPTT indices
    wptt_idx_cache = {}

    def _get_wptt_idx(wptt):
        wid = id(wptt)
        if wid not in wptt_idx_cache:
            wptt_idx_cache[wid] = _index_wptt_rules(wptt)
        return wptt_idx_cache[wid]

    # Postorder traversal to compute CL tables
    node_cls = {}  # node_id -> CL table
    next_id = [0]

    def _assign_id(node):
        nid = next_id[0]
        next_id[0] += 1
        return nid

    def _postorder(node, parent_id=None):
        """Compute CL table for this node via postorder traversal."""
        node_id = _assign_id(node)

        if isinstance(node, int):
            # Leaf node — use MSA-column-indexed CL
            msa_row = msa[node]
            wptt = _get_wptt(parent_id, node_id)
            wptt_i = _get_wptt_idx(wptt)
            cl = leaf_cl_msa(
                wptt, msa_row, wptt_i,
                band_width=band_width, guide_pairs=guide_pairs,
                consensus_structure=consensus_structure)
            if beam_width is not None:
                cl = _beam_prune_cl(cl, beam_width, L)
            node_cls[node_id] = cl
            return node_id

        # Internal node
        left_child, right_child = node
        left_id = _postorder(left_child, node_id)
        right_id = _postorder(right_child, node_id)

        wptt_left = _get_wptt(node_id, left_id)
        wptt_right = _get_wptt(node_id, right_id)
        wptt_left_idx = _get_wptt_idx(wptt_left)
        wptt_right_idx = _get_wptt_idx(wptt_right)

        cl = push_up_internal(
            node_cls[left_id], node_cls[right_id],
            wptt_left, wptt_right,
            wptt_left_idx, wptt_right_idx,
            L, beam_width=beam_width)
        node_cls[node_id] = cl
        return node_id

    root_id = _postorder(phylo_tree)
    cl_root = node_cls[root_id]

    # At root: read START state for full span
    log_prob = cl_root.get((WPTT_START, 0, L), NEG_INF)

    return log_prob, cl_root, node_cls


def beam_cyk_phylo(grammar, msa, phylo_tree, transducers,
                   guide_structure=None, beam_width=None,
                   band_width=None, consensus_structure=None):
    """Beam-pruned phylogenetic CYK (Viterbi) for SCFG on MSA.

    Like beam_inside_phylo but uses max instead of sum, and returns
    traceback information for Viterbi ancestral reconstruction.

    Returns:
        log_prob: Viterbi log P(MSA | grammar, tree)
        cl_root: dict (w, i, j) -> log_prob at root
        node_info: dict with per-node CL tables and tracebacks
    """
    msa = np.asarray(msa)
    N_seq, L = msa.shape
    NEG_INF = -1e30

    guide_pairs = None
    if guide_structure is not None:
        from ..tree.recognizer import parse_dot_bracket
        guide_pairs, _ = parse_dot_bracket(guide_structure)

    def _get_wptt(parent_node, child_node):
        if isinstance(transducers, dict):
            return transducers.get((parent_node, child_node),
                                   transducers.get('default'))
        return transducers

    wptt_idx_cache = {}

    def _get_wptt_idx(wptt):
        wid = id(wptt)
        if wid not in wptt_idx_cache:
            wptt_idx_cache[wid] = _index_wptt_rules(wptt)
        return wptt_idx_cache[wid]

    node_cls = {}
    node_tbs = {}
    leaf_msa_rows = {}
    next_id = [0]

    def _assign_id(node):
        nid = next_id[0]
        next_id[0] += 1
        return nid

    def _postorder(node, parent_id=None):
        node_id = _assign_id(node)

        if isinstance(node, int):
            msa_row = msa[node]
            wptt = _get_wptt(parent_id, node_id)
            wptt_i = _get_wptt_idx(wptt)
            cl, tb = leaf_cl_msa(
                wptt, msa_row, wptt_i,
                band_width=band_width, guide_pairs=guide_pairs,
                consensus_structure=consensus_structure, mode='cyk')
            if beam_width is not None:
                cl = _beam_prune_cl(cl, beam_width, L)
            node_cls[node_id] = cl
            node_tbs[node_id] = tb
            leaf_msa_rows[node_id] = msa_row
            return node_id

        left_child, right_child = node
        left_id = _postorder(left_child, node_id)
        right_id = _postorder(right_child, node_id)

        wptt_left = _get_wptt(node_id, left_id)
        wptt_right = _get_wptt(node_id, right_id)
        wptt_left_idx = _get_wptt_idx(wptt_left)
        wptt_right_idx = _get_wptt_idx(wptt_right)

        cl = push_up_internal(
            node_cls[left_id], node_cls[right_id],
            wptt_left, wptt_right,
            wptt_left_idx, wptt_right_idx,
            L, beam_width=beam_width)
        node_cls[node_id] = cl
        node_tbs[node_id] = None  # internal nodes don't have leaf tb
        return node_id

    root_id = _postorder(phylo_tree)
    cl_root = node_cls[root_id]

    # At root: read START state for full span
    log_prob = cl_root.get((WPTT_START, 0, L), NEG_INF)

    node_info = {
        'cls': node_cls,
        'tbs': node_tbs,
        'leaf_msa_rows': leaf_msa_rows,
        'root_id': root_id,
    }
    return log_prob, cl_root, node_info


def reconstruct_phylo_structure(msa, node_info):
    """Extract per-leaf structure and alignment from beam_cyk_phylo results.

    Args:
        msa: (N, L) array of nucleotide indices (-1 for gaps)
        node_info: dict from beam_cyk_phylo with 'tbs' and node mapping

    Returns:
        dict mapping leaf_node_id -> {
            'structure': dot-bracket string (L columns, MSA-indexed),
            'aln_type': alignment annotation per column (M/I/D/.),
        }
    """
    msa = np.asarray(msa)
    N, L = msa.shape
    tbs = node_info['tbs']
    results = {}

    for node_id, tb in tbs.items():
        if tb is None:
            continue
        # Find which MSA row this leaf corresponds to.
        # node_info doesn't directly store leaf->msa_row mapping,
        # so we search for the leaf index from the traceback.
        # The leaf_cl_msa was called with msa[leaf_idx], so the
        # traceback is MSA-column-indexed.
        # We need the msa_row to call format_cyk_parse_msa.
        # For now, we try each MSA row to find one that's consistent.
        # This is stored in node_info if available.
        msa_row = node_info.get('leaf_msa_rows', {}).get(node_id)
        if msa_row is None:
            continue
        struct, aln = format_cyk_parse_msa(tb, msa_row)
        results[node_id] = {
            'structure': struct,
            'aln_type': aln,
        }

    return results


def consensus_structure_from_phylo(msa, node_info):
    """Compute consensus dot-bracket structure from beam_cyk_phylo results.

    Extracts per-leaf structures and computes majority-vote consensus.

    Args:
        msa: (N, L) array of nucleotide indices (-1 for gaps)
        node_info: dict from beam_cyk_phylo

    Returns:
        structure: consensus dot-bracket string (L columns)
        per_leaf: dict of per-leaf annotations from reconstruct_phylo_structure
    """
    per_leaf = reconstruct_phylo_structure(msa, node_info)
    L = msa.shape[1]

    if not per_leaf:
        return '.' * L, per_leaf

    # Count basepair votes
    pair_votes = {}  # (i, j) -> count
    for leaf_info in per_leaf.values():
        struct = leaf_info['structure']
        stack = []
        for k, c in enumerate(struct):
            if c == '(':
                stack.append(k)
            elif c == ')' and stack:
                left = stack.pop()
                pair_votes[(left, k)] = pair_votes.get((left, k), 0) + 1

    # Accept pairs supported by majority of leaves
    n_leaves = len(per_leaf)
    threshold = n_leaves / 2.0
    consensus = ['.'] * L
    accepted_pairs = sorted(pair_votes.items(), key=lambda x: -x[1])

    # Greedily accept non-conflicting pairs by vote count
    paired = set()
    for (left, right), count in accepted_pairs:
        if count >= threshold and left not in paired and right not in paired:
            consensus[left] = '('
            consensus[right] = ')'
            paired.add(left)
            paired.add(right)

    return ''.join(consensus), per_leaf


def leaf_cyk_likelihoods(transducer, leaf_seq, wptt_idx,
                         band_width=None, guide_pairs=None,
                         consensus_structure=None):
    """CYK (Viterbi) version of leaf_conditional_likelihoods.

    Returns the best (max) log-probability path for each (w, i, j),
    plus a traceback table recording which rule/split produced each entry.

    Returns:
        cl: dict (w, i, j) -> log_prob (max, not sum)
        tb: dict (w, i, j) -> traceback_entry
            traceback_entry is one of:
              ('eps', w, r, i, j)
              ('match', w, r_st, i, j, rec_tok, w_dst, r_dst, ci, cj)
              ('insert', w, r_st, i, j, rec_tok, w_dst, r_dst, ci, cj)
              ('bif', w, r_st, i, j, w_l, w_r, r_l, r_r, k)
              ('ready', w_src, r, i, j, w_dst)
              ('delete', w_src, r, i, j, w_dst)
    """
    rec = build_leaf_recognizer(
        leaf_seq, band_width=band_width, guide_pairs=guide_pairs,
        consensus_structure=consensus_structure)

    L = len(leaf_seq)
    NEG_INF = -1e30
    n_trans = transducer.n_states

    log_I = {}     # (w, r, i, j) -> log_prob
    tb_I = {}      # (w, r, i, j) -> traceback entry
    span_entries_cyk = {}  # (i, j) -> set of (w, r)

    def _get(w, r, i, j):
        return log_I.get((w, r, i, j), NEG_INF)

    def _set(w, r, i, j, val, tb_entry):
        old = log_I.get((w, r, i, j), NEG_INF)
        if val > old:
            log_I[(w, r, i, j)] = val
            tb_I[(w, r, i, j)] = tb_entry
            span_entries_cyk.setdefault((i, j), set()).add((w, r))

    rec_match = {}
    rec_bif = {}
    rec_eps = {}
    for r in rec.rules:
        if r.rule_type == 'match':
            rec_match.setdefault(r.src, []).append(
                (r.input_token, r.dst, r.weight))
        elif r.rule_type == 'bifurcation':
            rec_bif.setdefault(r.src, []).append(
                (r.dst_left, r.dst_right, r.weight))
        elif r.rule_type == 'epsilon':
            rec_eps.setdefault(r.src, []).append((r.weight,))

    trans_bif = wptt_idx['bif']
    trans_eps = wptt_idx['eps']
    all_match_by_out = wptt_idx['all_match_by_out']
    all_insert_by_out = wptt_idx['all_insert_by_out']
    ready_list = wptt_idx['ready_list']
    delete_list = wptt_idx['delete_list']
    n_rec = rec.n_states
    rec_state_by_span = rec.span_to_state

    ready_log = [(s, d, np.log(w)) for s, d, w in ready_list]
    delete_log = [(s, it, d, np.log(w)) for s, it, d, w in delete_list]
    match_log_by_out = {}
    for out_tok, rules in all_match_by_out.items():
        match_log_by_out[out_tok] = [
            (src, in_tok, dst, np.log(w)) for src, in_tok, dst, w in rules
        ]
    insert_log_by_out = {}
    for out_tok, rules in all_insert_by_out.items():
        insert_log_by_out[out_tok] = [
            (src, dst, np.log(w)) for src, dst, w in rules
        ]
    eps_log = {}
    for w, rules in trans_eps.items():
        for w_t, in rules:
            eps_log[w] = np.log(w_t)
    bif_log = []
    for w, rules in trans_bif.items():
        for w_l, w_r, w_t in rules:
            bif_log.append((w, w_l, w_r, np.log(w_t)))

    def _unary_closure(i, j):
        populated = set(span_entries_cyk.get((i, j), set()))
        by_wptt = {}
        for w, r in populated:
            by_wptt.setdefault(w, []).append(r)

        for iteration in range(50):
            changed = False
            for w_src, w_dst, log_wt in ready_log:
                for r in by_wptt.get(w_dst, []):
                    val = log_I.get((w_dst, r, i, j), NEG_INF)
                    if val <= NEG_INF:
                        continue
                    new_val = log_wt + val
                    old = log_I.get((w_src, r, i, j), NEG_INF)
                    if new_val > old:
                        _set(w_src, r, i, j, new_val,
                             ('ready', w_src, r, i, j, w_dst))
                        if (w_src, r) not in populated:
                            populated.add((w_src, r))
                            by_wptt.setdefault(w_src, []).append(r)
                        changed = True
            for w_src, in_tok, w_dst, log_wt in delete_log:
                for r in by_wptt.get(w_dst, []):
                    val = log_I.get((w_dst, r, i, j), NEG_INF)
                    if val <= NEG_INF:
                        continue
                    new_val = log_wt + val
                    old = log_I.get((w_src, r, i, j), NEG_INF)
                    if new_val > old:
                        _set(w_src, r, i, j, new_val,
                             ('delete', w_src, r, i, j, w_dst))
                        if (w_src, r) not in populated:
                            populated.add((w_src, r))
                            by_wptt.setdefault(w_src, []).append(r)
                        changed = True
            if not changed:
                break

    # Base case: epsilon spans
    for pos in range(L + 1):
        r_st = rec_state_by_span.get((pos, pos))
        if r_st is None:
            continue
        for r_w, in rec_eps.get(r_st, []):
            log_rw = np.log(r_w)
            for w, log_wt in eps_log.items():
                _set(w, r_st, pos, pos, log_rw + log_wt,
                     ('eps', w, r_st, pos, pos))
        _unary_closure(pos, pos)

    # Fill spans
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span
            r_st = rec_state_by_span.get((i, j))
            if r_st is None:
                continue

            # Match + Insert
            for rec_tok, r_dst, r_w in rec_match.get(r_st, []):
                out_type, out_nucs = decode_terminal(rec_tok)
                if out_type == 'L':
                    if int(leaf_seq[i]) != out_nucs[0]:
                        continue
                    ci, cj = i + 1, j
                elif out_type == 'R':
                    if int(leaf_seq[j - 1]) != out_nucs[0]:
                        continue
                    ci, cj = i, j - 1
                elif out_type == 'LR':
                    if span < 2:
                        continue
                    if (int(leaf_seq[i]) != out_nucs[0] or
                            int(leaf_seq[j - 1]) != out_nucs[1]):
                        continue
                    ci, cj = i + 1, j - 1
                else:
                    continue

                log_rw = np.log(r_w)

                for w_src, in_tok, w_dst, log_wt in match_log_by_out.get(
                        rec_tok, []):
                    val_child = _get(w_dst, r_dst, ci, cj)
                    if val_child <= NEG_INF:
                        continue
                    val = log_rw + log_wt + val_child
                    _set(w_src, r_st, i, j, val,
                         ('match', w_src, r_st, i, j, rec_tok,
                          w_dst, r_dst, ci, cj))

                for w_src, w_dst, log_wt in insert_log_by_out.get(
                        rec_tok, []):
                    val_child = _get(w_dst, r_dst, ci, cj)
                    if val_child <= NEG_INF:
                        continue
                    val = log_rw + log_wt + val_child
                    _set(w_src, r_st, i, j, val,
                         ('insert', w_src, r_st, i, j, rec_tok,
                          w_dst, r_dst, ci, cj))

            # Bifurcation
            for r_l, r_r, r_w in rec_bif.get(r_st, []):
                span_l = rec.states[r_l].span
                k = span_l[1]
                log_rw = np.log(r_w)
                for w, w_l, w_r, log_wt in bif_log:
                    vl = _get(w_l, r_l, i, k)
                    vr = _get(w_r, r_r, k, j)
                    if vl <= NEG_INF or vr <= NEG_INF:
                        continue
                    val = log_rw + log_wt + vl + vr
                    _set(w, r_st, i, j, val,
                         ('bif', w, r_st, i, j, w_l, w_r,
                          r_l, r_r, k))

            _unary_closure(i, j)

    # Marginalize out recognizer states (max)
    cl = {}
    tb = {}
    for (w, r, i, j), val in log_I.items():
        if val <= NEG_INF:
            continue
        key = (w, i, j)
        old = cl.get(key, NEG_INF)
        if val > old:
            cl[key] = val
            tb[key] = tb_I[(w, r, i, j)]

    return cl, tb


def cyk_traceback(tb, w_start, i, j):
    """Extract the Viterbi parse from a CYK traceback table.

    Args:
        tb: traceback dict from leaf_cyk_likelihoods
        w_start: starting WPTT state (typically WPTT_START)
        i, j: span

    Returns:
        list of (rule_type, details) tuples describing the parse
    """
    result = []
    stack = [(w_start, i, j)]

    while stack:
        w, ci, cj = stack.pop()
        entry = tb.get((w, ci, cj))
        if entry is None:
            continue

        rule_type = entry[0]
        result.append(entry)

        if rule_type == 'eps':
            pass  # terminal
        elif rule_type == 'ready':
            # ready: w_src -> w_dst at same span
            _, w_src, r, ii, jj, w_dst = entry
            stack.append((w_dst, ii, jj))
        elif rule_type == 'delete':
            _, w_src, r, ii, jj, w_dst = entry
            stack.append((w_dst, ii, jj))
        elif rule_type == 'match':
            _, w, r_st, ii, jj, rec_tok, w_dst, r_dst, child_i, child_j = entry
            stack.append((w_dst, child_i, child_j))
        elif rule_type == 'insert':
            _, w, r_st, ii, jj, rec_tok, w_dst, r_dst, child_i, child_j = entry
            stack.append((w_dst, child_i, child_j))
        elif rule_type == 'bif':
            _, w, r_st, ii, jj, w_l, w_r, r_l, r_r, k = entry
            stack.append((w_l, ii, k))
            stack.append((w_r, k, jj))

    return result


def format_cyk_parse(tb, leaf_seq, w_start=WPTT_START):
    """Convert CYK traceback to structural annotation.

    Returns a dot-bracket string and alignment annotation for the leaf sequence.

    Args:
        tb: traceback dict from leaf_cyk_likelihoods
        leaf_seq: nucleotide sequence array
        w_start: starting WPTT state

    Returns:
        structure: dot-bracket string (length L)
        aln_type: alignment type string ('M'=match, 'I'=insert, 'D'=delete,
                  '.'=unpaired, each char corresponds to a sequence position)
    """
    L = len(leaf_seq)
    if L == 0:
        return '', ''

    structure = ['.'] * L
    aln_type = ['.'] * L
    pairs = []  # list of (i, j) basepairs

    NUC_CHARS = 'ACGU'

    def _trace(w, i, j):
        entry = tb.get((w, i, j))
        if entry is None:
            return

        rule_type = entry[0]

        if rule_type == 'eps':
            pass
        elif rule_type == 'ready':
            _, w_src, r, ii, jj, w_dst = entry
            _trace(w_dst, ii, jj)
        elif rule_type == 'delete':
            _, w_src, r, ii, jj, w_dst = entry
            _trace(w_dst, ii, jj)
        elif rule_type in ('match', 'insert'):
            _, w, r_st, ii, jj, rec_tok, w_dst, r_dst, ci, cj = entry
            out_type, out_nucs = decode_terminal(rec_tok)

            # Record alignment type from WPTT state
            wptt_info = decode_wptt_state(w)
            aln_char = 'S'  # Start
            if wptt_info is not None:
                aln_idx = wptt_info[0]
                aln_char = ALN_NAMES[aln_idx]

            if rule_type == 'insert':
                aln_char = 'I'

            if out_type == 'L':
                aln_type[ii] = aln_char
            elif out_type == 'R':
                aln_type[jj - 1] = aln_char
            elif out_type == 'LR':
                pairs.append((ii, jj - 1))
                structure[ii] = '('
                structure[jj - 1] = ')'
                aln_type[ii] = aln_char
                aln_type[jj - 1] = aln_char

            _trace(w_dst, ci, cj)
        elif rule_type == 'bif':
            _, w, r_st, ii, jj, w_l, w_r, r_l, r_r, k = entry
            _trace(w_l, ii, k)
            _trace(w_r, k, jj)

    _trace(w_start, 0, L)
    return ''.join(structure), ''.join(aln_type)


def cyk_traceback_msa(tb, w_start, i, j):
    """Extract the Viterbi parse from an MSA-indexed CYK traceback table.

    Like cyk_traceback but handles gap_del_L/R/LR entries from leaf_cl_msa.

    Args:
        tb: traceback dict from leaf_cl_msa(mode='cyk')
        w_start: starting WPTT state
        i, j: MSA column span

    Returns:
        list of traceback entry tuples
    """
    result = []
    stack = [(w_start, i, j)]

    while stack:
        w, ci, cj = stack.pop()
        entry = tb.get((w, ci, cj))
        if entry is None:
            continue

        rule_type = entry[0]
        result.append(entry)

        if rule_type == 'eps':
            pass
        elif rule_type == 'ready':
            _, w_src, r, ii, jj, w_dst = entry
            stack.append((w_dst, ii, jj))
        elif rule_type == 'gap_del_L':
            _, w_src, r, ii, jj, w_dst = entry
            stack.append((w_dst, ii + 1, jj))
        elif rule_type == 'gap_del_R':
            _, w_src, r, ii, jj, w_dst = entry
            stack.append((w_dst, ii, jj - 1))
        elif rule_type == 'gap_del_LR':
            _, w_src, r, ii, jj, w_dst = entry
            stack.append((w_dst, ii + 1, jj - 1))
        elif rule_type == 'match':
            _, w, r_st, ii, jj, rec_tok, w_dst, r_dst, child_i, child_j = entry
            stack.append((w_dst, child_i, child_j))
        elif rule_type == 'insert':
            _, w, r_st, ii, jj, rec_tok, w_dst, r_dst, child_i, child_j = entry
            stack.append((w_dst, child_i, child_j))
        elif rule_type == 'bif':
            _, w, r_st, ii, jj, w_l, w_r, r_l, r_r, k = entry
            stack.append((w_l, ii, k))
            stack.append((w_r, k, jj))

    return result


def format_cyk_parse_msa(tb, msa_row, w_start=WPTT_START):
    """Convert MSA-indexed CYK traceback to structural annotation.

    Like format_cyk_parse but for MSA-indexed tracebacks from leaf_cl_msa.
    Produces per-MSA-column annotations (including gap columns).

    Args:
        tb: traceback dict from leaf_cl_msa(mode='cyk')
        msa_row: (L,) array of nucleotide indices (-1 for gaps)
        w_start: starting WPTT state

    Returns:
        structure: dot-bracket string (length L, MSA columns)
        aln_type: alignment type per column ('M'=match, 'I'=insert,
                  'D'=delete/gap, '.'=unpaired/unvisited)
    """
    msa_row = np.asarray(msa_row)
    L = len(msa_row)
    if L == 0:
        return '', ''

    structure = ['.'] * L
    aln_type = ['.'] * L

    def _aln_char(w, rule_type):
        """Determine alignment character from WPTT state."""
        if rule_type == 'insert':
            return 'I'
        wptt_info = decode_wptt_state(w)
        if wptt_info is not None:
            aln_idx = wptt_info[0]
            return ALN_NAMES[aln_idx]
        return 'S'

    def _trace(w, i, j):
        entry = tb.get((w, i, j))
        if entry is None:
            return

        rule_type = entry[0]

        if rule_type == 'eps':
            pass
        elif rule_type == 'ready':
            _, w_src, r, ii, jj, w_dst = entry
            _trace(w_dst, ii, jj)
        elif rule_type == 'gap_del_L':
            _, w_src, r, ii, jj, w_dst = entry
            aln_type[ii] = 'D'
            _trace(w_dst, ii + 1, jj)
        elif rule_type == 'gap_del_R':
            _, w_src, r, ii, jj, w_dst = entry
            aln_type[jj - 1] = 'D'
            _trace(w_dst, ii, jj - 1)
        elif rule_type == 'gap_del_LR':
            _, w_src, r, ii, jj, w_dst = entry
            aln_type[ii] = 'D'
            aln_type[jj - 1] = 'D'
            structure[ii] = '('
            structure[jj - 1] = ')'
            _trace(w_dst, ii + 1, jj - 1)
        elif rule_type in ('match', 'insert'):
            _, w, r_st, ii, jj, rec_tok, w_dst, r_dst, ci, cj = entry
            out_type, out_nucs = decode_terminal(rec_tok)
            ac = _aln_char(w, rule_type)

            if out_type == 'L':
                aln_type[ii] = ac
            elif out_type == 'R':
                aln_type[jj - 1] = ac
            elif out_type == 'LR':
                structure[ii] = '('
                structure[jj - 1] = ')'
                aln_type[ii] = ac
                aln_type[jj - 1] = ac

            _trace(w_dst, ci, cj)
        elif rule_type == 'bif':
            _, w, r_st, ii, jj, w_l, w_r, r_l, r_r, k = entry
            _trace(w_l, ii, k)
            _trace(w_r, k, jj)

    _trace(w_start, 0, L)
    return ''.join(structure), ''.join(aln_type)


def scfg_weighted_wptt_weights(scfg_weights=None, tkf_weights=None,
                               ins_rate=0.02, del_rate=0.04, t=0.5):
    """Build WPTT weights that incorporate SCFG structural priors.

    Multiplies SCFG production weights into the WPTT rule weights so that
    the SCFG structural preferences are reflected in the CL computation.

    The SCFG nonterminal maps to WPTT in_ctx:
      Start, L, R -> CTX_NN (averaged)
      LR_AU -> CTX_AU, LR_CG -> CTX_CG, etc.

    For each WPTT rule from a ready state with in_ctx = c:
      - Match/delete consuming input_tok: multiply by SCFG emit weight
        for that token type from nonterminals mapping to c
      - Bifurcation: multiply by SCFG bifurcation weight
      - Epsilon: multiply by SCFG epsilon weight

    For non-ready state rules (insert, ready): no SCFG weight needed
    (these are within-column transitions, not structural choices).

    Args:
        scfg_weights: optional SCFG weight dict (see build_order1_singlet_scfg)
        tkf_weights: optional pre-computed TKF WPTT weight dict
        ins_rate, del_rate, t: TKF params (used if tkf_weights not provided)

    Returns:
        WPTT weight dict with SCFG weights folded in
    """
    from ..distill.wptt import tkf_wptt_weights
    from ..models.order1_scfg import (
        build_order1_singlet_scfg, N_SINGLET_STATES,
        IDX_START as SCFG_START, IDX_L, IDX_R, IDX_LR_BASE,
    )

    # Get base TKF weights
    if tkf_weights is None:
        tkf_weights = tkf_wptt_weights(
            ins_rate=ins_rate, del_rate=del_rate, t=t)

    # Build SCFG to extract normalized production weights
    scfg = build_order1_singlet_scfg(weights=scfg_weights)

    # Build SCFG weight lookup: for each (ctx, rule_type, token_info) -> weight
    scfg_emit_weight = {}  # key -> list of weights (for averaging)

    for prod in scfg.productions:
        src_idx = prod.lhs
        if src_idx in (SCFG_START, IDX_L, IDX_R):
            ctx = CTX_NN
        else:
            ctx = src_idx - IDX_LR_BASE

        if prod.is_right_linear:
            tok = prod.rhs[0]
            scfg_emit_weight.setdefault(
                (ctx, 'L', tok), []).append(prod.weight)
        elif prod.is_left_linear:
            tok = prod.rhs[1]
            scfg_emit_weight.setdefault(
                (ctx, 'R', tok), []).append(prod.weight)
        elif prod.is_lr_linear:
            tok_l, tok_r = prod.rhs[0], prod.rhs[2]
            lt, lt_nucs = decode_terminal(tok_l)
            rt, rt_nucs = decode_terminal(tok_r)
            if lt == 'L' and rt == 'R':
                pair_tok = pair_terminal(lt_nucs[0], rt_nucs[0])
                scfg_emit_weight.setdefault(
                    (ctx, 'LR', pair_tok), []).append(prod.weight)
        elif prod.is_binary:
            scfg_emit_weight.setdefault(
                (ctx, 'bif', None), []).append(prod.weight)
        elif prod.is_empty:
            scfg_emit_weight.setdefault(
                (ctx, 'eps', None), []).append(prod.weight)

    # Average weights for contexts with multiple nonterminals (CTX_NN)
    scfg_lookup = {}
    for key, weight_list in scfg_emit_weight.items():
        scfg_lookup[key] = sum(weight_list) / len(weight_list)

    # Multiply SCFG weights into TKF WPTT weights
    combined = dict(tkf_weights)

    for key, tkf_w in tkf_weights.items():
        rule_type = key[0]

        if rule_type in ('match', 'delete'):
            src = key[1]
            in_tok = key[2]
            if not is_ready_state(src):
                continue
            in_ctx = CTX_NN if src == 0 else decode_wptt_state(src)[1]
            tok_type, _ = decode_terminal(in_tok)
            if tok_type in ('L', 'R', 'LR'):
                scfg_key = (in_ctx, tok_type, in_tok)
                scfg_w = scfg_lookup.get(scfg_key, 1.0)
                combined[key] = tkf_w * scfg_w

        elif rule_type == 'bif':
            src = key[1]
            if not is_ready_state(src):
                continue
            in_ctx = CTX_NN if src == 0 else decode_wptt_state(src)[1]
            scfg_w = scfg_lookup.get((in_ctx, 'bif', None), 1.0)
            combined[key] = tkf_w * scfg_w

        elif rule_type == 'eps':
            src = key[1]
            if not is_ready_state(src):
                continue
            in_ctx = CTX_NN if src == 0 else decode_wptt_state(src)[1]
            scfg_w = scfg_lookup.get((in_ctx, 'eps', None), 1.0)
            combined[key] = tkf_w * scfg_w

    return combined


NUC_CHARS = 'ACGU'


def format_msa_with_structure(msa, consensus, per_leaf=None, seq_names=None):
    """Format MSA with structure annotation for display.

    Args:
        msa: (N, L) array of nucleotide indices (-1 for gaps)
        consensus: dot-bracket string (L chars)
        per_leaf: optional per-leaf annotations from reconstruct_phylo_structure
        seq_names: optional list of sequence names

    Returns:
        string: formatted multi-line display
    """
    msa = np.asarray(msa)
    N, L = msa.shape
    lines = []

    # Header
    if seq_names is None:
        seq_names = [f'seq{i}' for i in range(N)]
    max_name = max(len(n) for n in seq_names)

    # Sequence lines
    for i in range(N):
        name = seq_names[i].ljust(max_name)
        seq_str = ''.join(
            NUC_CHARS[msa[i, j]] if msa[i, j] >= 0 else '-'
            for j in range(L))
        lines.append(f'{name}  {seq_str}')

    # Consensus structure
    lines.append(f'{"cons".ljust(max_name)}  {consensus}')

    # Per-leaf alignment types
    if per_leaf:
        for node_id, ann in sorted(per_leaf.items()):
            label = f'aln{node_id}'.ljust(max_name)
            lines.append(f'{label}  {ann["aln_type"]}')

    return '\n'.join(lines)


def leaf_posterior_structure(seq, ins_rate=0.02, del_rate=0.04, t=0.5,
                            wptt_weights=None, scfg_weights=None,
                            band_width=None, consensus_structure=None):
    """Compute posterior basepair probabilities for a single sequence.

    Runs Inside-Outside on the WPTT×Recognizer composition and returns
    posterior basepair probabilities and span posteriors.

    Args:
        seq: numpy array of nucleotide indices (0-3), shape (L,)
        ins_rate, del_rate, t: TKF parameters
        wptt_weights: optional pre-computed WPTT weight dict
        scfg_weights: optional SCFG weight dict
        band_width: optional span banding width
        consensus_structure: optional dot-bracket guide structure

    Returns:
        dict with:
            'log_prob': log P(seq | model)
            'basepair_probs': dict (i, j) -> P(i paired with j | seq)
            'span_probs': dict (i, j) -> P(span [i,j) active | seq)
    """
    from ..distill.wptt import build_wptt_transducer, tkf_wptt_weights as _tkf_wts

    seq = np.asarray(seq)

    if wptt_weights is not None:
        wts = wptt_weights
    else:
        wts = _tkf_wts(ins_rate=ins_rate, del_rate=del_rate, t=t)
        if scfg_weights is not None:
            wts = scfg_weighted_wptt_weights(
                scfg_weights=scfg_weights, tkf_weights=wts)

    wptt = build_wptt_transducer(weights=wts)
    idx = _index_wptt_rules(wptt)

    # Inside pass
    cl, inside_info = leaf_cl_msa(
        wptt, seq, idx,
        band_width=band_width,
        consensus_structure=consensus_structure,
        return_inside_table=True)

    total = cl.get((WPTT_START, 0, len(seq)), -1e30)
    log_I = inside_info['log_I']

    # Outside pass
    log_O = leaf_outside_msa(wptt, seq, idx, inside_info)

    # Posteriors
    bp = posterior_basepair_probs(
        log_I, log_O, seq, total, idx, inside_info)
    sp = posterior_span_probs(log_I, log_O, total)

    return {
        'log_prob': total,
        'basepair_probs': bp,
        'span_probs': sp,
    }


def phylo_structure(msa, tree, ins_rate=0.02, del_rate=0.04, t=0.5,
                    branch_lengths=None,
                    beam_width=None, band_width=None,
                    consensus_structure=None,
                    wptt_weights=None, scfg_weights=None,
                    mode='inside'):
    """High-level interface: compute phylogenetic structure for an MSA.

    Runs the beam SCFG pipeline (Inside or CYK) on the given MSA and tree,
    with optional TKF-parameterized WPTT weights.

    Args:
        msa: (N, L) array of nucleotide indices (0-3, -1 for gaps)
        tree: phylogenetic tree as nested tuples ((0, 1), 2) or (0, 1)
        ins_rate: TKF insertion rate (used if wptt_weights not provided)
        del_rate: TKF deletion rate
        t: default branch length (used if branch_lengths not provided)
        branch_lengths: optional dict mapping (parent_id, child_id) -> t,
            or a single float for uniform branches
        beam_width: max WPTT states per span (None = no pruning)
        band_width: max span width in MSA columns (None = no banding)
        consensus_structure: optional dot-bracket guide structure
        wptt_weights: optional pre-computed WPTT weight dict (single WPTT
            for all branches; overrides TKF params and branch_lengths)
        scfg_weights: optional SCFG weight dict
        mode: 'inside' (log marginal) or 'cyk' (Viterbi + traceback)

    Returns:
        dict with:
            'log_prob': log P(MSA | model)
            'consensus': (CYK only) consensus dot-bracket structure
            'per_leaf': (CYK only) per-leaf {structure, aln_type}
            'cl_root': root CL table
            'node_info': per-node info dict
    """
    from ..models.order1_scfg import build_order1_singlet_scfg
    from ..distill.wptt import build_wptt_transducer, tkf_wptt_weights

    msa = np.asarray(msa)
    grammar = build_order1_singlet_scfg(weights=scfg_weights)

    def _make_wptt_weights(bl):
        """Build WPTT weights for a given branch length, with SCFG if provided."""
        tkf_wts = tkf_wptt_weights(
            ins_rate=ins_rate, del_rate=del_rate, t=bl)
        if scfg_weights is not None:
            return scfg_weighted_wptt_weights(
                scfg_weights=scfg_weights, tkf_weights=tkf_wts)
        return tkf_wts

    if wptt_weights is not None:
        # Single WPTT for all branches (caller manages SCFG integration)
        wptt = build_wptt_transducer(weights=wptt_weights)
    elif branch_lengths is not None:
        if isinstance(branch_lengths, (int, float)):
            wts = _make_wptt_weights(float(branch_lengths))
            wptt = build_wptt_transducer(weights=wts)
        else:
            wptt = {}
            for edge, bl in branch_lengths.items():
                wts = _make_wptt_weights(bl)
                wptt[edge] = build_wptt_transducer(weights=wts)
            default_wts = _make_wptt_weights(t)
            wptt['default'] = build_wptt_transducer(weights=default_wts)
    else:
        wts = _make_wptt_weights(t)
        wptt = build_wptt_transducer(weights=wts)

    if mode == 'cyk':
        lp, cl_root, info = beam_cyk_phylo(
            grammar, msa, tree, wptt,
            beam_width=beam_width, band_width=band_width,
            consensus_structure=consensus_structure)
        consensus, per_leaf = consensus_structure_from_phylo(msa, info)
        return {
            'log_prob': lp,
            'consensus': consensus,
            'per_leaf': per_leaf,
            'cl_root': cl_root,
            'node_info': info,
        }
    else:
        lp, cl_root, info = beam_inside_phylo(
            grammar, msa, tree, wptt,
            beam_width=beam_width, band_width=band_width,
            consensus_structure=consensus_structure)
        return {
            'log_prob': lp,
            'cl_root': cl_root,
            'node_info': info,
        }


def iterative_phylo_structure(msa, tree, n_iter=3, **kwargs):
    """Iterative refinement: CYK -> extract consensus -> re-run with guide.

    Each iteration:
    1. Run CYK to get Viterbi structure
    2. Extract consensus structure
    3. Use consensus as guide for next iteration

    The guide structure biases the recognizer towards the consensus,
    improving convergence and runtime (via banding).

    Args:
        msa: (N, L) array of nucleotide indices
        tree: phylogenetic tree
        n_iter: number of refinement iterations
        **kwargs: passed to phylo_structure (ins_rate, del_rate, t, etc.)

    Returns:
        dict with final iteration results plus 'history' of log_probs
    """
    history = []
    consensus = kwargs.pop('consensus_structure', None)

    for iteration in range(n_iter):
        result = phylo_structure(
            msa, tree, mode='cyk',
            consensus_structure=consensus, **kwargs)
        history.append(result['log_prob'])
        new_consensus = result['consensus']

        # Check convergence: if consensus didn't change, stop
        if consensus is not None and new_consensus == consensus:
            break
        consensus = new_consensus

    result['history'] = history
    result['n_iterations'] = len(history)
    return result
