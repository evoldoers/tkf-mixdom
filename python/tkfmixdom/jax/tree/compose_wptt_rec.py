"""Factored DP for WPTT × Recognizer composition.

Composes a WPTT (transducer) with a downstream recognizer by computing
transitions on the fly from the component machines.

State: (wptt_st, rec_st) — WPTT state × recognizer state
The WPTT's output tokens must match the recognizer's input tokens.

Synchronization rules (from ParseTreeTransducers.md):
  - If recognizer is not ready, transducer cannot update
  - If transducer doesn't produce output, recognizer cannot update
  - When transducer produces output, recognizer must consume it as input
  - Epsilon and bifurcation must happen simultaneously
  - Composite weight = product of component weights
"""

import numpy as np
from ..distill.wptt import (
    WPTT, is_ready_state, IDX_START as WPTT_START,
    ALN_M, ALN_I, ALN_D, ALN_R,
)
from ..models.rna_grammar import (
    left_terminal, right_terminal, pair_terminal, decode_terminal,
    N_NUC, N_TOTAL_TERMINALS,
)


def compose_wptt_recognizer(transducer, recognizer, leaf_seq):
    """Factored Inside for WPTT × Recognizer on a nucleotide leaf sequence.

    Computes the probability that the WPTT, when its output is constrained
    to match the recognizer (which accepts parses of leaf_seq), generates
    any valid transduction.

    This is used in the progressive reconstruction pipeline: given a WPTT
    and a child's recognizer profile, compute the composed probability
    over all alignments and structural assignments.

    Args:
        transducer: WPTT object (246 states)
        recognizer: Recognizer object (span-based, all states ready)
        leaf_seq: numpy array of nucleotide indices (0-3), shape (L,)

    Returns:
        log_prob: log probability at (WPTT_START, rec_start, 0, L)
        log_I: dict mapping (wptt_st, rec_st, i, j) -> log probability
    """
    L = len(leaf_seq)
    n_trans = transducer.n_states
    n_rec = recognizer.n_states
    NEG_INF = -1e30

    # Sparse Inside table: (wptt_st, rec_st, i, j) -> log_prob
    log_I = {}

    def _get(w, r, i, j):
        return log_I.get((w, r, i, j), NEG_INF)

    def _set(w, r, i, j, val):
        old = log_I.get((w, r, i, j), NEG_INF)
        log_I[(w, r, i, j)] = np.logaddexp(old, val)

    # Pre-index transducer rules by source state and type
    trans_match = {}   # src -> list of (input_tok, output_tok, dst, weight)
    trans_delete = {}  # src -> list of (input_tok, dst, weight)
    trans_insert = {}  # src -> list of (output_tok, dst, weight)
    trans_ready = {}   # src -> list of (dst, weight)
    trans_bif = {}     # src -> list of (dst_l, dst_r, weight)
    trans_eps = {}     # src -> list of (weight,)

    for r in transducer.rules:
        if r.rule_type == 'match':
            trans_match.setdefault(r.src, []).append(
                (r.input_token, r.output_token, r.dst, r.weight))
        elif r.rule_type == 'delete':
            trans_delete.setdefault(r.src, []).append(
                (r.input_token, r.dst, r.weight))
        elif r.rule_type == 'insert':
            trans_insert.setdefault(r.src, []).append(
                (r.output_token, r.dst, r.weight))
        elif r.rule_type == 'ready':
            trans_ready.setdefault(r.src, []).append(
                (r.dst, r.weight))
        elif r.rule_type == 'bifurcation':
            trans_bif.setdefault(r.src, []).append(
                (r.dst_left, r.dst_right, r.weight))
        elif r.rule_type == 'epsilon':
            trans_eps.setdefault(r.src, []).append(
                (r.weight,))

    # Pre-index recognizer rules by source state and type
    rec_match = {}    # src -> list of (input_tok, dst, weight)
    rec_bif = {}      # src -> list of (dst_l, dst_r, weight)
    rec_eps = {}      # src -> list of (weight,)

    # Also build token-indexed lookup for recognizer matches
    rec_match_by_tok = {}  # (src, input_tok) -> list of (dst, weight)

    for r in recognizer.rules:
        if r.rule_type == 'match':
            rec_match.setdefault(r.src, []).append(
                (r.input_token, r.dst, r.weight))
            rec_match_by_tok.setdefault((r.src, r.input_token), []).append(
                (r.dst, r.weight))
        elif r.rule_type == 'bifurcation':
            rec_bif.setdefault(r.src, []).append(
                (r.dst_left, r.dst_right, r.weight))
        elif r.rule_type == 'epsilon':
            rec_eps.setdefault(r.src, []).append(
                (r.weight,))

    # Pre-index: WPTT match rules by (src, output_tok)
    trans_match_by_out = {}  # (src, out_tok) -> list of (in_tok, dst, weight)
    for src, rules in trans_match.items():
        for in_tok, out_tok, dst, w in rules:
            trans_match_by_out.setdefault((src, out_tok), []).append(
                (in_tok, dst, w))

    # Pre-index: WPTT insert rules by (src, output_tok)
    trans_insert_by_out = {}
    for src, rules in trans_insert.items():
        for out_tok, dst, w in rules:
            trans_insert_by_out.setdefault((src, out_tok), []).append(
                (dst, w))

    # Build recognizer state index from span
    rec_state_by_span = recognizer.span_to_state

    def _unary_closure(i, j):
        """Apply span-preserving transitions until convergence.

        Span-preserving: WPTT ready, WPTT delete (consumes no leaf nucleotides).
        Note: recognizer states don't change for span-preserving transitions
        because they are span-indexed and the span doesn't change.
        """
        for iteration in range(50):
            changed = False

            # WPTT ready transition: w_nonready -> w_ready
            for w_src, rules in trans_ready.items():
                for w_dst, w_t in rules:
                    for r in range(n_rec):
                        val = _get(w_dst, r, i, j)
                        if val <= NEG_INF:
                            continue
                        new_val = np.log(w_t) + val
                        old = _get(w_src, r, i, j)
                        if new_val > old + 1e-10 or old <= NEG_INF:
                            _set(w_src, r, i, j, new_val)
                            changed = True

            # WPTT delete: consumes WPTT input but produces no output
            # This is span-preserving because the recognizer doesn't advance.
            # The WPTT goes from ready to D state.
            # But wait — delete consumes an *input* token from the generator,
            # not from the recognizer. In this composition, the WPTT's input
            # comes from the *generator* (which we don't have here — that's
            # the SCFG×WPTT composition, not WPTT×Recognizer).
            #
            # In WPTT×Recognizer, the WPTT's *output* goes to the recognizer.
            # Deletes consume WPTT input (from generator) but don't produce
            # WPTT output, so the recognizer doesn't advance.
            # However, we don't have the generator input tokens here.
            #
            # In the progressive pipeline, WPTT×Recognizer is done WITHOUT
            # the generator. The WPTT delete rules are about consuming
            # generator input tokens. Since we don't have the generator,
            # delete transitions are marginalized over all possible input
            # tokens. So: sum over all delete rules from a ready WPTT state.
            for w in range(n_trans):
                if not is_ready_state(w):
                    continue
                for rules in [trans_delete.get(w, [])]:
                    for in_tok, w_dst, w_t in rules:
                        for r in range(n_rec):
                            val = _get(w_dst, r, i, j)
                            if val <= NEG_INF:
                                continue
                            new_val = np.log(w_t) + val
                            old = _get(w, r, i, j)
                            if new_val > old + 1e-10 or old <= NEG_INF:
                                _set(w, r, i, j, new_val)
                                changed = True

            if not changed:
                break

    # === Base case: span 0 (epsilon) ===
    for pos in range(L + 1):
        r_st = rec_state_by_span.get((pos, pos))
        if r_st is None:
            continue
        for r_w, in rec_eps.get(r_st, []):
            for w in range(n_trans):
                if not is_ready_state(w):
                    continue
                for w_t, in trans_eps.get(w, []):
                    val = np.log(r_w * w_t)
                    _set(w, r_st, pos, pos, val)

        _unary_closure(pos, pos)

    # === Fill spans of increasing length ===
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span
            r_st = rec_state_by_span.get((i, j))
            if r_st is None:
                continue

            # --- Match: WPTT produces output, recognizer consumes it ---
            # For each recognizer match rule from r_st that accepts a token,
            # find WPTT match rules that produce that token.
            for rec_tok, r_dst, r_w in rec_match.get(r_st, []):
                out_type, out_nucs = decode_terminal(rec_tok)

                # Determine which leaf nucleotides this token consumes
                if out_type == 'L':
                    # Consumes leaf[i], child span is [i+1, j)
                    if int(leaf_seq[i]) != out_nucs[0]:
                        continue
                    child_i, child_j = i + 1, j
                elif out_type == 'R':
                    # Consumes leaf[j-1], child span is [i, j-1)
                    if int(leaf_seq[j - 1]) != out_nucs[0]:
                        continue
                    child_i, child_j = i, j - 1
                elif out_type == 'LR':
                    if span < 2:
                        continue
                    if (int(leaf_seq[i]) != out_nucs[0] or
                            int(leaf_seq[j - 1]) != out_nucs[1]):
                        continue
                    child_i, child_j = i + 1, j - 1
                else:
                    continue

                # Find WPTT rules that produce rec_tok as output
                for w in range(n_trans):
                    if not is_ready_state(w):
                        continue
                    for in_tok, w_dst, w_t in trans_match_by_out.get(
                            (w, rec_tok), []):
                        val_child = _get(w_dst, r_dst, child_i, child_j)
                        if val_child <= NEG_INF:
                            continue
                        val = np.log(r_w * w_t) + val_child
                        _set(w, r_st, i, j, val)

            # --- Insert: WPTT inserts (produces output without input) ---
            # WPTT is in non-ready state, recognizer consumes the output
            for rec_tok, r_dst, r_w in rec_match.get(r_st, []):
                out_type, out_nucs = decode_terminal(rec_tok)

                if out_type == 'L':
                    if int(leaf_seq[i]) != out_nucs[0]:
                        continue
                    child_i, child_j = i + 1, j
                elif out_type == 'R':
                    if int(leaf_seq[j - 1]) != out_nucs[0]:
                        continue
                    child_i, child_j = i, j - 1
                elif out_type == 'LR':
                    if span < 2:
                        continue
                    if (int(leaf_seq[i]) != out_nucs[0] or
                            int(leaf_seq[j - 1]) != out_nucs[1]):
                        continue
                    child_i, child_j = i + 1, j - 1
                else:
                    continue

                for w in range(n_trans):
                    if is_ready_state(w):
                        continue  # inserts only from non-ready
                    for w_dst, w_t in trans_insert_by_out.get(
                            (w, rec_tok), []):
                        val_child = _get(w_dst, r_dst, child_i, child_j)
                        if val_child <= NEG_INF:
                            continue
                        val = np.log(r_w * w_t) + val_child
                        _set(w, r_st, i, j, val)

            # --- Bifurcation: both WPTT and recognizer bifurcate ---
            for r_l, r_r, r_w in rec_bif.get(r_st, []):
                for w in range(n_trans):
                    if not is_ready_state(w):
                        continue
                    for w_l, w_r, w_t in trans_bif.get(w, []):
                        log_wgt = np.log(r_w * w_t)
                        # Split point k from recognizer state spans
                        span_l = recognizer.states[r_l].span
                        k = span_l[1]  # end of left child span

                        val_l = _get(w_l, r_l, i, k)
                        val_r = _get(w_r, r_r, k, j)
                        if val_l <= NEG_INF or val_r <= NEG_INF:
                            continue
                        val = log_wgt + val_l + val_r
                        _set(w, r_st, i, j, val)

            # Apply span-preserving closure
            _unary_closure(i, j)

    # Result at (WPTT_START, rec_start, 0, L)
    rec_start = recognizer.start
    log_prob = _get(WPTT_START, rec_start, 0, L)
    return log_prob, log_I
