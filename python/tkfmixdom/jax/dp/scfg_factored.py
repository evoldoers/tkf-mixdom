"""Factored DP for SCFG × WPTT composition.

Instead of materializing the full composed grammar (~230K rules, 2460 NTs),
runs Inside/CYK directly on the factored state space (gen_nt, wptt_st),
computing transitions on the fly from the component machines.

The DP operates over a NUCLEOTIDE leaf sequence. WPTT output tokens
consume nucleotides from the leaf:
  - L-output: consumes nucleotide at position i (left end of span)
  - R-output: consumes nucleotide at position j-1 (right end of span)
  - LR-output: consumes nucleotides at both i and j-1

State: (gen_nt, wptt_st) — 10 × 246 = 2460 composite states
Span: [i, j) over leaf nucleotide sequence y

Transitions:
  - Match: gen emits input token, WPTT consumes it and outputs to leaf
  - Insert: WPTT outputs to leaf without consuming gen input
  - Delete: WPTT consumes gen input without outputting
  - Ready: WPTT transitions to ready state (no tokens)
  - Unary: gen unary transition (WPTT unchanged)
  - Bifurcation: both gen and WPTT bifurcate (children start from Start)
  - Epsilon: both go to epsilon simultaneously

Two implementations:
  - factored_inside(): original sparse dict-based (slow but correct reference)
  - factored_inside_jax(): dense numpy array-based (~100x faster)
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp
from ..models.rna_grammar import (
    left_terminal, right_terminal, pair_terminal, decode_terminal,
    N_NUC, N_TOTAL_TERMINALS,
)
from ..core.rna import classify_basepair, CTX_NN
from ..distill.wptt import (
    WPTT, is_ready_state, decode_wptt_state, IDX_START as WPTT_START,
    ALN_M, ALN_I, ALN_D, ALN_V, ALN_W, ALN_R,
)


# ---------------------------------------------------------------------------
# Pre-compiled transition tables for dense DP
# ---------------------------------------------------------------------------

class _CompiledTransitions:
    """Pre-compiled transition index arrays for dense factored Inside.

    Converts the generator WCFG and WPTT transducer rules into flat numpy
    arrays suitable for vectorized DP operations.
    """
    __slots__ = [
        'n_gen', 'n_wptt', 'n_composite',
        # Closure matrix: (I - U)^{-1} for span-preserving transitions
        'closure_dense',  # dense numpy array (for JIT)
        'closure',  # (N_C, N_C) in probability space
        # Delete sub-matrix of U (for extracting delete counts in EM)
        'U_del',  # sparse (N_C, N_C) — composed delete transitions only
        # Epsilon base case: composite states with nonzero epsilon weight
        'eps_states', 'eps_log_weights',  # (n_eps,)
        # L-match: gen right-linear + WPTT match with L output
        # For each output nucleotide y: arrays of (src_c, dst_c, log_w)
        'l_match',   # dict: out_nuc -> (src_c, dst_c, log_w) arrays
        # R-match: gen left-linear + WPTT match with R output
        'r_match',   # dict: out_nuc -> (src_c, dst_c, log_w) arrays
        # LR-match: gen lr-linear + WPTT match with LR output
        'lr_match',  # dict: (out_nuc_l, out_nuc_r) -> (src_c, dst_c, log_w)
        # L-insert: WPTT insert with L output (gen unchanged)
        'l_insert',  # dict: out_nuc -> (src_c, dst_c, log_w) arrays
        # R-insert: WPTT insert with R output
        'r_insert',  # dict: out_nuc -> (src_c, dst_c, log_w)
        # Bifurcation: gen binary + WPTT bifurcation
        'bif_src', 'bif_dst_l', 'bif_dst_r', 'bif_log_w',  # (n_bif,)
        # EM metadata: V/W source flags and input tokens for match transitions
        'l_match_from_V',   # dict: nuc -> bool array (True if src is V/Start)
        'r_match_from_V',   # dict: nuc -> bool array
        'lr_match_from_V',  # dict: (nuc_l, nuc_r) -> bool array
        'l_match_in_nuc',   # dict: out_nuc -> int array of input nucleotides
        'r_match_in_nuc',   # dict: out_nuc -> int array of input nucleotides
        'lr_match_in_pair', # dict: (out_l, out_r) -> (int array in_l, int array in_r)
    ]

    def __init__(self, generator, transducer):
        n_gen = generator.n_nonterminals
        n_wptt = transducer.n_states
        n_c = n_gen * n_wptt
        self.n_gen = n_gen
        self.n_wptt = n_wptt
        self.n_composite = n_c

        def _ci(g, w):
            """Composite index."""
            return g * n_wptt + w

        # --- Build span-preserving transition matrix U ---
        # U[src_c, dst_c] = weight of span-preserving transition src -> dst
        # Includes: gen unary, WPTT ready, composed deletes
        U = np.zeros((n_c, n_c), dtype=np.float64)

        # Gen unary: (g, w) -> (g', w)
        for p in generator.productions:
            if p.is_unary:
                g, g_next = p.lhs, p.rhs[0]
                for w in range(n_wptt):
                    U[_ci(g, w), _ci(g_next, w)] += p.weight

        # WPTT ready: (g, w_src) -> (g, w_dst)
        for r in transducer.rules:
            if r.rule_type == 'ready':
                for g in range(n_gen):
                    U[_ci(g, r.src), _ci(g, r.dst)] += r.weight

        # Composed delete: gen emitting rule + WPTT delete
        wptt_del = {}  # input_tok -> [(src, dst, weight)]
        for r in transducer.rules:
            if r.rule_type == 'delete':
                wptt_del.setdefault(r.input_token, []).append(
                    (r.src, r.dst, r.weight))

        U_del = np.zeros((n_c, n_c), dtype=np.float64)
        for p in generator.productions:
            if p.is_right_linear:
                term, g_next = p.rhs[0], p.rhs[1]
                for w_src, w_dst, w_t in wptt_del.get(term, []):
                    w = p.weight * w_t
                    U[_ci(p.lhs, w_src), _ci(g_next, w_dst)] += w
                    U_del[_ci(p.lhs, w_src), _ci(g_next, w_dst)] += w
            elif p.is_left_linear:
                g_next, term = p.rhs[0], p.rhs[1]
                for w_src, w_dst, w_t in wptt_del.get(term, []):
                    w = p.weight * w_t
                    U[_ci(p.lhs, w_src), _ci(g_next, w_dst)] += w
                    U_del[_ci(p.lhs, w_src), _ci(g_next, w_dst)] += w
            elif p.is_lr_linear:
                term_l, g_next, term_r = p.rhs
                lt, lt_nucs = decode_terminal(term_l)
                rt, rt_nucs = decode_terminal(term_r)
                if lt == 'L' and rt == 'R':
                    compound = pair_terminal(lt_nucs[0], rt_nucs[0])
                    for w_src, w_dst, w_t in wptt_del.get(compound, []):
                        w = p.weight * w_t
                        U[_ci(p.lhs, w_src), _ci(g_next, w_dst)] += w
                        U_del[_ci(p.lhs, w_src), _ci(g_next, w_dst)] += w

        # Closure: (I - U)^{-1} in probability space
        import scipy.sparse as sp
        self.closure_dense = np.linalg.solve(np.eye(n_c) - U, np.eye(n_c))
        self.closure = sp.csr_matrix(self.closure_dense)
        self.U_del = sp.csr_matrix(U_del)

        # --- Epsilon base case ---
        # (g, w) with gen epsilon + WPTT epsilon, both from ready states
        wptt_eps = {}  # src -> weight
        for r in transducer.rules:
            if r.rule_type == 'epsilon':
                wptt_eps[r.src] = wptt_eps.get(r.src, 0.0) + r.weight

        eps_states_list = []
        eps_log_w_list = []
        for p in generator.productions:
            if p.is_empty:
                for w, w_t in wptt_eps.items():
                    c = _ci(p.lhs, w)
                    eps_states_list.append(c)
                    eps_log_w_list.append(np.log(p.weight * w_t))
        self.eps_states = np.array(eps_states_list, dtype=np.int32)
        self.eps_log_weights = np.array(eps_log_w_list, dtype=np.float64)

        # --- Match transitions ---
        # Index WPTT match rules: (src, input_tok) -> [(output_tok, dst, w)]
        wptt_match = {}
        for r in transducer.rules:
            if r.rule_type == 'match':
                wptt_match.setdefault((r.src, r.input_token), []).append(
                    (r.output_token, r.dst, r.weight))

        # L-match: gen right-linear g->term g' + WPTT match(term, out_L) -> w'
        # Group by output nucleotide
        # Extra metadata: from_V (bool), in_nuc (input nucleotide)
        l_match_lists = {nuc: ([], [], [], [], []) for nuc in range(N_NUC)}
        for p in generator.productions:
            if p.is_right_linear:
                term, g_next = p.rhs[0], p.rhs[1]
                _, term_nucs = decode_terminal(term)
                in_nuc = term_nucs[0] if term_nucs else -1
                for w in range(n_wptt):
                    if not is_ready_state(w):
                        continue
                    w_is_V = (w == WPTT_START) or (
                        decode_wptt_state(w) is not None and
                        decode_wptt_state(w)[0] == ALN_V)
                    for out_tok, w_dst, w_t in wptt_match.get(
                            (w, term), []):
                        # out_tok must be a left terminal
                        ot, ot_nucs = decode_terminal(out_tok)
                        if ot != 'L':
                            continue
                        out_nuc = ot_nucs[0]
                        src_c, dst_c = _ci(p.lhs, w), _ci(g_next, w_dst)
                        l_match_lists[out_nuc][0].append(src_c)
                        l_match_lists[out_nuc][1].append(dst_c)
                        l_match_lists[out_nuc][2].append(
                            np.log(p.weight * w_t))
                        l_match_lists[out_nuc][3].append(w_is_V)
                        l_match_lists[out_nuc][4].append(in_nuc)

        self.l_match = {}
        self.l_match_from_V = {}
        self.l_match_in_nuc = {}
        for nuc in range(N_NUC):
            if l_match_lists[nuc][0]:
                self.l_match[nuc] = (
                    np.array(l_match_lists[nuc][0], dtype=np.int32),
                    np.array(l_match_lists[nuc][1], dtype=np.int32),
                    np.array(l_match_lists[nuc][2], dtype=np.float64),
                )
                self.l_match_from_V[nuc] = np.array(
                    l_match_lists[nuc][3], dtype=bool)
                self.l_match_in_nuc[nuc] = np.array(
                    l_match_lists[nuc][4], dtype=np.int32)

        # R-match: gen left-linear g->g' term + WPTT match(term, out_R) -> w'
        r_match_lists = {nuc: ([], [], [], [], []) for nuc in range(N_NUC)}
        for p in generator.productions:
            if p.is_left_linear:
                g_next, term = p.rhs[0], p.rhs[1]
                _, term_nucs = decode_terminal(term)
                in_nuc = term_nucs[0] if term_nucs else -1
                for w in range(n_wptt):
                    if not is_ready_state(w):
                        continue
                    w_is_V = (w == WPTT_START) or (
                        decode_wptt_state(w) is not None and
                        decode_wptt_state(w)[0] == ALN_V)
                    for out_tok, w_dst, w_t in wptt_match.get(
                            (w, term), []):
                        ot, ot_nucs = decode_terminal(out_tok)
                        if ot != 'R':
                            continue
                        out_nuc = ot_nucs[0]
                        src_c, dst_c = _ci(p.lhs, w), _ci(g_next, w_dst)
                        r_match_lists[out_nuc][0].append(src_c)
                        r_match_lists[out_nuc][1].append(dst_c)
                        r_match_lists[out_nuc][2].append(
                            np.log(p.weight * w_t))
                        r_match_lists[out_nuc][3].append(w_is_V)
                        r_match_lists[out_nuc][4].append(in_nuc)

        self.r_match = {}
        self.r_match_from_V = {}
        self.r_match_in_nuc = {}
        for nuc in range(N_NUC):
            if r_match_lists[nuc][0]:
                self.r_match[nuc] = (
                    np.array(r_match_lists[nuc][0], dtype=np.int32),
                    np.array(r_match_lists[nuc][1], dtype=np.int32),
                    np.array(r_match_lists[nuc][2], dtype=np.float64),
                )
                self.r_match_from_V[nuc] = np.array(
                    r_match_lists[nuc][3], dtype=bool)
                self.r_match_in_nuc[nuc] = np.array(
                    r_match_lists[nuc][4], dtype=np.int32)

        # LR-match: gen lr-linear g->termL g' termR + WPTT match(compound, outLR)
        lr_match_lists = {}  # key -> (src, dst, lw, from_V, in_l, in_r)
        for p in generator.productions:
            if p.is_lr_linear:
                term_l, g_next, term_r = p.rhs
                lt, lt_nucs = decode_terminal(term_l)
                rt, rt_nucs = decode_terminal(term_r)
                if lt != 'L' or rt != 'R':
                    continue
                in_nuc_l, in_nuc_r = lt_nucs[0], rt_nucs[0]
                compound_in = pair_terminal(in_nuc_l, in_nuc_r)
                for w in range(n_wptt):
                    if not is_ready_state(w):
                        continue
                    w_is_V = (w == WPTT_START) or (
                        decode_wptt_state(w) is not None and
                        decode_wptt_state(w)[0] == ALN_V)
                    for out_tok, w_dst, w_t in wptt_match.get(
                            (w, compound_in), []):
                        ot, ot_nucs = decode_terminal(out_tok)
                        if ot != 'LR':
                            continue
                        key = (ot_nucs[0], ot_nucs[1])
                        if key not in lr_match_lists:
                            lr_match_lists[key] = ([], [], [], [], [], [])
                        src_c = _ci(p.lhs, w)
                        dst_c = _ci(g_next, w_dst)
                        lr_match_lists[key][0].append(src_c)
                        lr_match_lists[key][1].append(dst_c)
                        lr_match_lists[key][2].append(
                            np.log(p.weight * w_t))
                        lr_match_lists[key][3].append(w_is_V)
                        lr_match_lists[key][4].append(in_nuc_l)
                        lr_match_lists[key][5].append(in_nuc_r)

        self.lr_match = {}
        self.lr_match_from_V = {}
        self.lr_match_in_pair = {}
        for key, (s, d, lw, fv, il, ir) in lr_match_lists.items():
            self.lr_match[key] = (
                np.array(s, dtype=np.int32),
                np.array(d, dtype=np.int32),
                np.array(lw, dtype=np.float64),
            )
            self.lr_match_from_V[key] = np.array(fv, dtype=bool)
            self.lr_match_in_pair[key] = (
                np.array(il, dtype=np.int32),
                np.array(ir, dtype=np.int32),
            )

        # --- Insert transitions ---
        # WPTT insert: w_src -> out_tok, w_dst (gen state unchanged)
        # Only from non-ready states
        l_ins_lists = {nuc: ([], [], []) for nuc in range(N_NUC)}
        r_ins_lists = {nuc: ([], [], []) for nuc in range(N_NUC)}
        for r in transducer.rules:
            if r.rule_type != 'insert':
                continue
            if is_ready_state(r.src):
                continue
            ot, ot_nucs = decode_terminal(r.output_token)
            if ot == 'L':
                out_nuc = ot_nucs[0]
                for g in range(n_gen):
                    l_ins_lists[out_nuc][0].append(_ci(g, r.src))
                    l_ins_lists[out_nuc][1].append(_ci(g, r.dst))
                    l_ins_lists[out_nuc][2].append(np.log(r.weight))
            elif ot == 'R':
                out_nuc = ot_nucs[0]
                for g in range(n_gen):
                    r_ins_lists[out_nuc][0].append(_ci(g, r.src))
                    r_ins_lists[out_nuc][1].append(_ci(g, r.dst))
                    r_ins_lists[out_nuc][2].append(np.log(r.weight))

        self.l_insert = {}
        for nuc in range(N_NUC):
            if l_ins_lists[nuc][0]:
                self.l_insert[nuc] = (
                    np.array(l_ins_lists[nuc][0], dtype=np.int32),
                    np.array(l_ins_lists[nuc][1], dtype=np.int32),
                    np.array(l_ins_lists[nuc][2], dtype=np.float64),
                )
        self.r_insert = {}
        for nuc in range(N_NUC):
            if r_ins_lists[nuc][0]:
                self.r_insert[nuc] = (
                    np.array(r_ins_lists[nuc][0], dtype=np.int32),
                    np.array(r_ins_lists[nuc][1], dtype=np.int32),
                    np.array(r_ins_lists[nuc][2], dtype=np.float64),
                )

        # --- Bifurcation ---
        # gen binary g->g_l g_r + WPTT bif w->w_l w_r
        # Children always start from (gen_start=Start, wptt_start=WPTT_START)?
        # No — gen binary produces (g_l, g_r) and WPTT bif produces (w_l, w_r)
        # In practice, WPTT bif always goes to (WPTT_START, WPTT_START)
        # and gen binary always goes to (Start, Start) for order-1 SCFG
        bif_src_list = []
        bif_dst_l_list = []
        bif_dst_r_list = []
        bif_log_w_list = []
        for p in generator.productions:
            if p.is_binary:
                g_l, g_r = p.rhs
                for w in range(n_wptt):
                    if not is_ready_state(w):
                        continue
                    for r in transducer.rules:
                        if r.rule_type == 'bifurcation' and r.src == w:
                            bif_src_list.append(_ci(p.lhs, w))
                            bif_dst_l_list.append(_ci(g_l, r.dst_left))
                            bif_dst_r_list.append(_ci(g_r, r.dst_right))
                            bif_log_w_list.append(
                                np.log(p.weight * r.weight))
        self.bif_src = np.array(bif_src_list, dtype=np.int32)
        self.bif_dst_l = np.array(bif_dst_l_list, dtype=np.int32)
        self.bif_dst_r = np.array(bif_dst_r_list, dtype=np.int32)
        self.bif_log_w = np.array(bif_log_w_list, dtype=np.float64)


# Module-level cache for compiled transitions
_compiled_cache = {}


def _get_compiled(generator, transducer):
    """Get or create compiled transitions (cached by object identity)."""
    key = (id(generator), id(transducer))
    if key not in _compiled_cache:
        _compiled_cache[key] = _CompiledTransitions(generator, transducer)
    return _compiled_cache[key]


# ---------------------------------------------------------------------------
# Dense numpy Inside algorithm
# ---------------------------------------------------------------------------

def factored_inside_jax(generator, transducer, leaf_seq):
    """Dense factored Inside for (SCFG × WPTT) on a nucleotide leaf sequence.

    Probability-space DP with per-cell scaling and sparse closure.
    Uses np.bincount for scatter-add (~9× faster than logaddexp.at)
    and scipy sparse matmul for closure (~20× faster than dense).

    Args:
        generator: WCFG (the order-1 singlet SCFG, 10 NTs)
        transducer: WPTT object (246 states)
        leaf_seq: numpy array of nucleotide indices (0-3), shape (L,)

    Returns:
        log_prob: log P(leaf_seq | generator, transducer)
        log_I: (N_C, L+1, L+1) dense numpy array of log probabilities
    """
    ct = _get_compiled(generator, transducer)
    L = len(leaf_seq)
    N_C = ct.n_composite
    NEG_INF = -1e30

    # Probability-space tables with per-cell log scaling
    # True probability = P[c, i, j] * exp(S[i, j])
    P = np.zeros((N_C, L + 1, L + 1), dtype=np.float64)
    S = np.full((L + 1, L + 1), NEG_INF, dtype=np.float64)

    # Pre-compute probability weights from log weights
    pw = {}  # cache for exp(log_weights)
    for name in ('l_match', 'r_match', 'l_insert', 'r_insert'):
        d = getattr(ct, name)
        pw[name] = {}
        for nuc, (src, dst, lw) in d.items():
            pw[name][nuc] = np.exp(lw)
    pw['lr_match'] = {}
    for key, (src, dst, lw) in ct.lr_match.items():
        pw['lr_match'][key] = np.exp(lw)
    bif_pw = np.exp(ct.bif_log_w)
    eps_pw = np.exp(ct.eps_log_weights)

    # === Base case: span 0 (epsilon) ===
    eps_raw = np.bincount(ct.eps_states, weights=eps_pw, minlength=N_C)
    eps_closed = ct.closure @ eps_raw
    eps_mx = eps_closed.max()
    if eps_mx > 0:
        eps_normed = eps_closed / eps_mx
        eps_scale = np.log(eps_mx)
        for i in range(L + 1):
            P[:, i, i] = eps_normed
            S[i, i] = eps_scale

    # === Fill spans of increasing length ===
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span
            nuc_i = int(leaf_seq[i])
            nuc_j = int(leaf_seq[j - 1])

            # Reference scale for this cell (max over child scales)
            ref = S[i + 1, j]                    # L-match/insert child
            ref = max(ref, S[i, j - 1])           # R-match/insert child
            if span >= 2:
                ref = max(ref, S[i + 1, j - 1])   # LR-match child
            # Bifurcation children
            for k in range(i, j + 1):
                ref = max(ref, S[i, k] + S[k, j])
            if ref <= NEG_INF:
                continue
            ref = float(ref)

            raw = np.zeros(N_C, dtype=np.float64)

            # --- L-match ---
            if nuc_i in ct.l_match:
                src, dst, _ = ct.l_match[nuc_i]
                w = pw['l_match'][nuc_i]
                child = P[dst, i + 1, j] * np.exp(S[i + 1, j] - ref)
                raw += np.bincount(src, weights=w * child, minlength=N_C)

            # --- R-match ---
            if nuc_j in ct.r_match:
                src, dst, _ = ct.r_match[nuc_j]
                w = pw['r_match'][nuc_j]
                child = P[dst, i, j - 1] * np.exp(S[i, j - 1] - ref)
                raw += np.bincount(src, weights=w * child, minlength=N_C)

            # --- LR-match ---
            if span >= 2:
                key = (nuc_i, nuc_j)
                if key in ct.lr_match:
                    src, dst, _ = ct.lr_match[key]
                    w = pw['lr_match'][key]
                    child = P[dst, i + 1, j - 1] * np.exp(
                        S[i + 1, j - 1] - ref)
                    raw += np.bincount(src, weights=w * child, minlength=N_C)

            # --- L-insert ---
            if nuc_i in ct.l_insert:
                src, dst, _ = ct.l_insert[nuc_i]
                w = pw['l_insert'][nuc_i]
                child = P[dst, i + 1, j] * np.exp(S[i + 1, j] - ref)
                raw += np.bincount(src, weights=w * child, minlength=N_C)

            # --- R-insert ---
            if nuc_j in ct.r_insert:
                src, dst, _ = ct.r_insert[nuc_j]
                w = pw['r_insert'][nuc_j]
                child = P[dst, i, j - 1] * np.exp(S[i, j - 1] - ref)
                raw += np.bincount(src, weights=w * child, minlength=N_C)

            # --- Bifurcation (vectorized over split points) ---
            if len(ct.bif_src) > 0:
                k_range = np.arange(i, j + 1)  # (span+1,)
                # Gather left/right Inside values for all splits
                l_vals = P[ct.bif_dst_l[:, None],
                           i, k_range[None, :]]          # (n_bif, span+1)
                r_vals = P[ct.bif_dst_r[:, None],
                           k_range[None, :], j]          # (n_bif, span+1)
                bscales = np.exp(
                    S[i, k_range] + S[k_range, j] - ref) # (span+1,)
                # Sum over split points, then scatter to src
                bif_contrib = bif_pw * np.sum(
                    l_vals * r_vals * bscales[None, :], axis=1)  # (n_bif,)
                raw += np.bincount(
                    ct.bif_src, weights=bif_contrib, minlength=N_C)

            # Apply closure (sparse matmul)
            closed = ct.closure @ raw

            # Normalize and store
            mx = closed.max()
            if mx > 0:
                P[:, i, j] = closed / mx
                S[i, j] = np.log(mx) + ref

    # Convert to log-probability table
    log_I = np.full((N_C, L + 1, L + 1), NEG_INF, dtype=np.float64)
    for span in range(L + 1):
        for i in range(L - span + 1):
            j = i + span
            if S[i, j] > NEG_INF:
                pos = P[:, i, j] > 0
                log_I[pos, i, j] = np.log(P[pos, i, j]) + S[i, j]

    start_c = generator.start * ct.n_wptt + WPTT_START
    log_prob = log_I[start_c, 0, L]
    return log_prob, log_I


# ---------------------------------------------------------------------------
# JIT-compiled factored Inside (JAX) — span-vectorized
# ---------------------------------------------------------------------------

def _compile_jax_arrays(ct):
    """Convert _CompiledTransitions to JAX arrays for JIT compilation.

    Exploits that L/R match/insert share src/dst indices across nucleotides
    (only weights differ), enabling batch gather-scatter operations.

    Returns a flat tuple of JAX arrays (valid pytree for JIT).
    """
    # Closure matrix
    closure = jnp.array(ct.closure_dense)

    # Epsilon base case: pre-scatter into a (N_C,) vector
    eps_vec = np.zeros(ct.n_composite, dtype=np.float64)
    for idx, lw in zip(ct.eps_states, ct.eps_log_weights):
        eps_vec[idx] += np.exp(lw)
    eps_vec = jnp.array(eps_vec)

    def _extract_shared(d):
        """Extract shared src/dst and per-nuc weights from transition dict.

        Returns (src, dst, w) where src/dst are (n_trans,) shared indices
        and w is (N_NUC, n_trans) per-nucleotide weights.
        """
        if not d:
            return (jnp.zeros(0, dtype=jnp.int32),
                    jnp.zeros(0, dtype=jnp.int32),
                    jnp.zeros((N_NUC, 0)))
        s0, d0, _ = d[0]
        n = len(s0)
        w = np.zeros((N_NUC, n), dtype=np.float64)
        for nuc in range(N_NUC):
            if nuc in d:
                w[nuc] = np.exp(d[nuc][2])
        return jnp.array(s0), jnp.array(d0), jnp.array(w)

    lm_src, lm_dst, lm_w = _extract_shared(ct.l_match)
    rm_src, rm_dst, rm_w = _extract_shared(ct.r_match)
    li_src, li_dst, li_w = _extract_shared(ct.l_insert)
    ri_src, ri_dst, ri_w = _extract_shared(ct.r_insert)

    # LR-match: (4, 4, n_trans) — indices vary by nuc pair
    n_lr = max((len(v[0]) for v in ct.lr_match.values()), default=0)
    lr_src = np.zeros((N_NUC, N_NUC, n_lr), dtype=np.int32)
    lr_dst = np.zeros((N_NUC, N_NUC, n_lr), dtype=np.int32)
    lr_w = np.zeros((N_NUC, N_NUC, n_lr), dtype=np.float64)
    for (nl, nr), (s, d2, lw) in ct.lr_match.items():
        k = len(s)
        lr_src[nl, nr, :k] = s
        lr_dst[nl, nr, :k] = d2
        lr_w[nl, nr, :k] = np.exp(lw)
    lr_src = jnp.array(lr_src)
    lr_dst = jnp.array(lr_dst)
    lr_w = jnp.array(lr_w)

    # Bifurcation
    bif_src = jnp.array(ct.bif_src)
    bif_dst_l = jnp.array(ct.bif_dst_l)
    bif_dst_r = jnp.array(ct.bif_dst_r)
    bif_w = jnp.exp(jnp.array(ct.bif_log_w))

    return (closure, eps_vec,
            lm_src, lm_dst, lm_w,
            rm_src, rm_dst, rm_w,
            li_src, li_dst, li_w,
            ri_src, ri_dst, ri_w,
            lr_src, lr_dst, lr_w,
            bif_src, bif_dst_l, bif_dst_r, bif_w)


_jax_cache = {}


def _get_jax_arrays(generator, transducer):
    """Get or create JAX transition arrays (cached)."""
    key = (id(generator), id(transducer))
    if key not in _jax_cache:
        ct = _get_compiled(generator, transducer)
        _jax_cache[key] = _compile_jax_arrays(ct)
    return _jax_cache[key]


# Geometric bin sizes for JIT cache reuse
_GEOM_BINS_DP = [0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 21, 26, 32, 39, 48, 59,
                 72, 88, 108, 132, 162, 198, 243, 297, 364, 446, 512]


def _pad_to_bin_dp(n):
    """Round n up to next geometric bin for JIT reuse."""
    for b in _GEOM_BINS_DP:
        if b >= n:
            return b
    return int(2 ** np.ceil(np.log2(max(n, 1))))


@functools.partial(jax.jit, static_argnums=(2, 3))
def _factored_inside_jit_core(leaf_seq, jax_arrays, N_C, L):
    """JIT-compiled factored Inside — span-vectorized.

    For each span length, processes ALL positions in parallel using
    batch gather, nucleotide-indexed weight lookup, batch scatter-add,
    and a single closure matrix-matrix multiply.

    Args:
        leaf_seq: (L,) int32 nucleotide indices (padded)
        jax_arrays: tuple of JAX arrays from _compile_jax_arrays
        N_C: static int, number of composite states
        L: static int, padded sequence length

    Returns:
        I: (N_C, L+1, L+1) scaled probability table
        S: (L+1, L+1) log scale factors
    """
    (closure, eps_vec,
     lm_src, lm_dst, lm_w,
     rm_src, rm_dst, rm_w,
     li_src, li_dst, li_w,
     ri_src, ri_dst, ri_w,
     lr_src, lr_dst, lr_w,
     bif_src, bif_dst_l, bif_dst_r, bif_w) = jax_arrays

    NEG = -1e30

    # Base case: closure @ eps_vec (same for all empty spans)
    init_col = closure @ eps_vec
    init_mx = jnp.max(init_col)
    init_col_normed = jnp.where(init_mx > 0, init_col / init_mx, 0.0)
    init_log_scale = jnp.where(init_mx > 0, jnp.log(init_mx), NEG)

    I = jnp.zeros((N_C, L + 1, L + 1))
    S = jnp.full((L + 1, L + 1), NEG)

    # Set base cases: I[:, i, i] for all i
    def base_body(i, carry):
        I_c, S_c = carry
        I_c = I_c.at[:, i, i].set(init_col_normed)
        S_c = S_c.at[i, i].set(init_log_scale)
        return (I_c, S_c)

    I, S = jax.lax.fori_loop(0, L + 1, base_body, (I, S))

    # Span loop (Python loop, unrolled by JIT tracer)
    for span in range(1, L + 1):
        n_pos = L - span + 1
        i_range = jnp.arange(n_pos)
        j_range = i_range + span
        nucs_i = leaf_seq[i_range]          # (n_pos,)
        nucs_j = leaf_seq[j_range - 1]      # (n_pos,)

        # Child scale factors (vectorized across positions)
        s_L = S[i_range + 1, j_range]       # (n_pos,)
        s_R = S[i_range, j_range - 1]       # (n_pos,)
        s_LR = jnp.where(span >= 2,
                         S[i_range + 1, j_range - 1], NEG)  # (n_pos,)

        # Bifurcation max scale (loop over split offsets, vectorized per-offset)
        bif_max = jnp.full(n_pos, NEG)
        for d in range(span + 1):
            k = i_range + d
            bif_max = jnp.maximum(bif_max, S[i_range, k] + S[k, j_range])

        # Reference scale per position
        ref = jnp.maximum(jnp.maximum(s_L, s_R),
                          jnp.maximum(s_LR, bif_max))
        ref = jnp.where(ref > NEG, ref, 0.0)

        # Accumulate raw contributions: (N_C, n_pos)
        raw = jnp.zeros((N_C, n_pos))

        # L-match: shared src/dst, per-nuc weights
        # child_L[t, p] = I[lm_dst[t], i_range[p]+1, j_range[p]]
        child_L = I[lm_dst[:, None],
                    (i_range + 1)[None, :],
                    j_range[None, :]]              # (n_lm, n_pos)
        scale_L = jnp.exp(s_L - ref)[None, :]     # (1, n_pos)
        w_lm = lm_w[nucs_i].T                     # (n_lm, n_pos)
        raw = raw.at[lm_src[:, None],
                     jnp.arange(n_pos)[None, :]].add(
            w_lm * child_L * scale_L)

        # L-insert: same child column as L-match
        child_LI = I[li_dst[:, None],
                     (i_range + 1)[None, :],
                     j_range[None, :]]             # (n_li, n_pos)
        w_li = li_w[nucs_i].T                     # (n_li, n_pos)
        raw = raw.at[li_src[:, None],
                     jnp.arange(n_pos)[None, :]].add(
            w_li * child_LI * scale_L)

        # R-match
        child_R = I[rm_dst[:, None],
                    i_range[None, :],
                    (j_range - 1)[None, :]]        # (n_rm, n_pos)
        scale_R = jnp.exp(s_R - ref)[None, :]
        w_rm = rm_w[nucs_j].T
        raw = raw.at[rm_src[:, None],
                     jnp.arange(n_pos)[None, :]].add(
            w_rm * child_R * scale_R)

        # R-insert
        child_RI = I[ri_dst[:, None],
                     i_range[None, :],
                     (j_range - 1)[None, :]]
        w_ri = ri_w[nucs_j].T
        raw = raw.at[ri_src[:, None],
                     jnp.arange(n_pos)[None, :]].add(
            w_ri * child_RI * scale_R)

        # LR-match: indices vary by nuc pair, loop over 16 pairs
        if span >= 2:
            scale_LR = jnp.exp(s_LR - ref)[None, :]
            for nl in range(N_NUC):
                for nr in range(N_NUC):
                    mask = ((nucs_i == nl) & (nucs_j == nr)).astype(
                        jnp.float64)[None, :]  # (1, n_pos)
                    child_lr = I[lr_dst[nl, nr, :, None],
                                 (i_range + 1)[None, :],
                                 (j_range - 1)[None, :]]  # (n_lr, n_pos)
                    raw = raw.at[lr_src[nl, nr, :, None],
                                 jnp.arange(n_pos)[None, :]].add(
                        lr_w[nl, nr, :, None] * child_lr * scale_LR * mask)

        # Bifurcation: loop over split offsets (vectorized per-offset)
        for d in range(span + 1):
            k = i_range + d
            l_vals = I[bif_dst_l[:, None],
                       i_range[None, :],
                       k[None, :]]                 # (n_bif, n_pos)
            r_vals = I[bif_dst_r[:, None],
                       k[None, :],
                       j_range[None, :]]           # (n_bif, n_pos)
            bscale = jnp.exp(S[i_range, k] + S[k, j_range] - ref)  # (n_pos,)
            raw = raw.at[bif_src[:, None],
                         jnp.arange(n_pos)[None, :]].add(
                bif_w[:, None] * l_vals * r_vals * bscale[None, :])

        # Batch closure: (N_C, N_C) @ (N_C, n_pos) = (N_C, n_pos)
        closed = closure @ raw

        # Normalize per position
        mx = jnp.max(closed, axis=0)  # (n_pos,)
        safe_mx = jnp.where(mx > 0, mx, 1.0)
        I = I.at[:, i_range, j_range].set(closed / safe_mx[None, :])
        S = S.at[i_range, j_range].set(
            jnp.where(mx > 0, jnp.log(mx) + ref, NEG))

    return I, S


def factored_inside_jit(generator, transducer, leaf_seq):
    """JIT-compiled factored Inside for (SCFG × WPTT).

    Span-vectorized: processes all positions within a span in parallel
    using batch gather-scatter and a single closure matrix-matrix multiply.
    Padded to geometric bins for JIT cache reuse.

    Args:
        generator: WCFG (the order-1 singlet SCFG, 10 NTs)
        transducer: WPTT object (246 states)
        leaf_seq: numpy array of nucleotide indices (0-3), shape (L,)

    Returns:
        log_prob: log P(leaf_seq | generator, transducer)
        log_I: (N_C, L+1, L+1) log-probability table (at real dimensions)
    """
    ct = _get_compiled(generator, transducer)
    jax_arrays = _get_jax_arrays(generator, transducer)
    L_real = len(leaf_seq)
    L_pad = max(_pad_to_bin_dp(L_real), 1)  # min 1 so JAX can trace indexing

    # Pad sequence
    if L_pad > L_real:
        leaf_seq_pad = jnp.concatenate([
            jnp.array(leaf_seq, dtype=jnp.int32),
            jnp.zeros(L_pad - L_real, dtype=jnp.int32)
        ])
    else:
        leaf_seq_pad = jnp.array(leaf_seq, dtype=jnp.int32)

    N_C = ct.n_composite
    I, S = _factored_inside_jit_core(leaf_seq_pad, jax_arrays, N_C, L_pad)

    # Extract start state log probability
    start_c = generator.start * ct.n_wptt + WPTT_START
    log_prob = jnp.log(I[start_c, 0, L_real]) + S[0, L_real]

    # Convert to log-probability table at real dimensions
    log_I_real = jnp.log(jnp.where(I[:, :L_real + 1, :L_real + 1] > 0,
                                    I[:, :L_real + 1, :L_real + 1],
                                    1e-300))
    log_I_real = log_I_real + S[:L_real + 1, :L_real + 1][None, :, :]

    return float(log_prob), np.array(log_I_real)


# ---------------------------------------------------------------------------
# Original sparse dict-based Inside (preserved as reference)
# ---------------------------------------------------------------------------

def factored_inside(generator, transducer, leaf_seq):
    """Factored Inside for (SCFG × WPTT) on a nucleotide leaf sequence.

    Computes the total probability that the composed system generates
    the given leaf sequence, marginalized over all ancestor parse trees
    and all alignment structures.

    Args:
        generator: WCFG (the order-1 singlet SCFG, 10 NTs)
        transducer: WPTT object (246 states)
        leaf_seq: numpy array of nucleotide indices (0-3), shape (L,)

    Returns:
        log_prob: log P(leaf_seq | generator, transducer)
        log_I: dict mapping (gen_nt, wptt_st, i, j) -> log probability
    """
    L = len(leaf_seq)
    n_gen = generator.n_nonterminals
    n_trans = transducer.n_states
    NEG_INF = -1e30

    # Sparse Inside table: (gen_nt, wptt_st, i, j) -> log_prob
    log_I = {}

    def _get(g, w, i, j):
        return log_I.get((g, w, i, j), NEG_INF)

    def _set(g, w, i, j, val):
        old = log_I.get((g, w, i, j), NEG_INF)
        new = np.logaddexp(old, val)
        log_I[(g, w, i, j)] = new

    # Pre-index generator productions by LHS and type
    gen_eps = {}      # gen_nt -> list of (weight,)
    gen_unary = {}    # gen_nt -> list of (next_nt, weight)
    gen_right = {}    # gen_nt -> list of (terminal, next_nt, weight)
    gen_left = {}     # gen_nt -> list of (next_nt, terminal, weight)
    gen_lr = {}       # gen_nt -> list of (term_l, next_nt, term_r, weight)
    gen_binary = {}   # gen_nt -> list of (left_nt, right_nt, weight)

    for p in generator.productions:
        if p.is_empty:
            gen_eps.setdefault(p.lhs, []).append((p.weight,))
        elif p.is_unary:
            gen_unary.setdefault(p.lhs, []).append((p.rhs[0], p.weight))
        elif p.is_right_linear:
            gen_right.setdefault(p.lhs, []).append(
                (p.rhs[0], p.rhs[1], p.weight))
        elif p.is_left_linear:
            gen_left.setdefault(p.lhs, []).append(
                (p.rhs[0], p.rhs[1], p.weight))
        elif p.is_lr_linear:
            gen_lr.setdefault(p.lhs, []).append(
                (p.rhs[0], p.rhs[1], p.rhs[2], p.weight))
        elif p.is_binary:
            gen_binary.setdefault(p.lhs, []).append(
                (p.rhs[0], p.rhs[1], p.weight))

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

    # Pre-compute: for each WPTT ready state, build a lookup
    # from input_token -> list of (output_tok, dst, weight) for matches
    # and input_token -> list of (dst, weight) for deletes
    match_by_input = {}   # (src, input_tok) -> [(output_tok, dst, weight)]
    delete_by_input = {}  # (src, input_tok) -> [(dst, weight)]

    for src, rules in trans_match.items():
        for in_tok, out_tok, dst, w in rules:
            match_by_input.setdefault((src, in_tok), []).append(
                (out_tok, dst, w))

    for src, rules in trans_delete.items():
        for in_tok, dst, w in rules:
            delete_by_input.setdefault((src, in_tok), []).append(
                (dst, w))

    def _unary_closure(i, j):
        """Apply unary closure at span [i,j): iterate span-preserving
        transitions (gen unary, WPTT ready, delete) until convergence."""
        for iteration in range(50):
            changed = False

            # Generator unary: (g, w) -> (g', w) with gen unary rule
            for g in range(n_gen):
                for g_next, w_g in gen_unary.get(g, []):
                    for w in range(n_trans):
                        val_child = _get(g_next, w, i, j)
                        if val_child <= NEG_INF:
                            continue
                        val = np.log(w_g) + val_child
                        old = _get(g, w, i, j)
                        if val > old + 1e-10 or old <= NEG_INF:
                            _set(g, w, i, j, val)
                            changed = True

            # WPTT ready transition: (g, w_nonready) -> (g, w_ready)
            for w_src, rules in trans_ready.items():
                for w_dst, w_t in rules:
                    for g in range(n_gen):
                        val_child = _get(g, w_dst, i, j)
                        if val_child <= NEG_INF:
                            continue
                        val = np.log(w_t) + val_child
                        old = _get(g, w_src, i, j)
                        if val > old + 1e-10 or old <= NEG_INF:
                            _set(g, w_src, i, j, val)
                            changed = True

            # Delete: (g, w_ready) -> (g', w') consuming gen token, no output
            for g in range(n_gen):
                # Right-linear gen rule: g -> left_X g'
                for term, g_next, w_g in gen_right.get(g, []):
                    for w in range(n_trans):
                        if not is_ready_state(w):
                            continue
                        for w_dst, w_t in delete_by_input.get((w, term), []):
                            val_child = _get(g_next, w_dst, i, j)
                            if val_child <= NEG_INF:
                                continue
                            val = np.log(w_g * w_t) + val_child
                            old = _get(g, w, i, j)
                            if val > old + 1e-10 or old <= NEG_INF:
                                _set(g, w, i, j, val)
                                changed = True

                # Left-linear gen rule: g -> g' right_X
                for g_next, term, w_g in gen_left.get(g, []):
                    for w in range(n_trans):
                        if not is_ready_state(w):
                            continue
                        for w_dst, w_t in delete_by_input.get((w, term), []):
                            val_child = _get(g_next, w_dst, i, j)
                            if val_child <= NEG_INF:
                                continue
                            val = np.log(w_g * w_t) + val_child
                            old = _get(g, w, i, j)
                            if val > old + 1e-10 or old <= NEG_INF:
                                _set(g, w, i, j, val)
                                changed = True

                # LR-linear gen rule: g -> left_X g' right_Z
                for term_l, g_next, term_r, w_g in gen_lr.get(g, []):
                    # Compound input token for LR
                    lt, lt_nucs = decode_terminal(term_l)
                    rt, rt_nucs = decode_terminal(term_r)
                    if lt == 'L' and rt == 'R':
                        compound = pair_terminal(lt_nucs[0], rt_nucs[0])
                    else:
                        continue
                    for w in range(n_trans):
                        if not is_ready_state(w):
                            continue
                        for w_dst, w_t in delete_by_input.get(
                                (w, compound), []):
                            val_child = _get(g_next, w_dst, i, j)
                            if val_child <= NEG_INF:
                                continue
                            val = np.log(w_g * w_t) + val_child
                            old = _get(g, w, i, j)
                            if val > old + 1e-10 or old <= NEG_INF:
                                _set(g, w, i, j, val)
                                changed = True

            if not changed:
                break

    # === Base case: span 0 (epsilon) ===
    for i in range(L + 1):
        for g in range(n_gen):
            for w_g, in gen_eps.get(g, []):
                for w in range(n_trans):
                    if not is_ready_state(w):
                        continue
                    for w_t, in trans_eps.get(w, []):
                        val = np.log(w_g * w_t)
                        _set(g, w, i, i, val)

        _unary_closure(i, i)

    # === Fill spans of increasing length ===
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span

            # --- L-output match: gen L-emit + WPTT match with L output ---
            # gen: g -> left_X g'  (right-linear)
            # WPTT: w (ready) -> match(input=left_X, output=left_Y) -> w'
            # constraint: Y == leaf_seq[i]
            target_out = left_terminal(int(leaf_seq[i]))
            for g in range(n_gen):
                for term, g_next, w_g in gen_right.get(g, []):
                    for w in range(n_trans):
                        if not is_ready_state(w):
                            continue
                        for out_tok, w_dst, w_t in match_by_input.get(
                                (w, term), []):
                            if out_tok != target_out:
                                continue
                            val_child = _get(g_next, w_dst, i + 1, j)
                            if val_child <= NEG_INF:
                                continue
                            val = np.log(w_g * w_t) + val_child
                            _set(g, w, i, j, val)

            # --- R-output match: gen R-emit + WPTT match with R output ---
            # gen: g -> g' right_X  (left-linear)
            # WPTT: w -> match(input=right_X, output=right_Y) -> w'
            # constraint: Y == leaf_seq[j-1]
            target_out_r = right_terminal(int(leaf_seq[j - 1]))
            for g in range(n_gen):
                for g_next, term, w_g in gen_left.get(g, []):
                    for w in range(n_trans):
                        if not is_ready_state(w):
                            continue
                        for out_tok, w_dst, w_t in match_by_input.get(
                                (w, term), []):
                            if out_tok != target_out_r:
                                continue
                            val_child = _get(g_next, w_dst, i, j - 1)
                            if val_child <= NEG_INF:
                                continue
                            val = np.log(w_g * w_t) + val_child
                            _set(g, w, i, j, val)

            # --- LR-output match: gen LR-emit + WPTT match with LR output ---
            if span >= 2:
                target_pair = pair_terminal(
                    int(leaf_seq[i]), int(leaf_seq[j - 1]))
                for g in range(n_gen):
                    for term_l, g_next, term_r, w_g in gen_lr.get(g, []):
                        lt, lt_nucs = decode_terminal(term_l)
                        rt, rt_nucs = decode_terminal(term_r)
                        if lt == 'L' and rt == 'R':
                            compound_in = pair_terminal(
                                lt_nucs[0], rt_nucs[0])
                        else:
                            continue
                        for w in range(n_trans):
                            if not is_ready_state(w):
                                continue
                            for out_tok, w_dst, w_t in match_by_input.get(
                                    (w, compound_in), []):
                                if out_tok != target_pair:
                                    continue
                                val_child = _get(
                                    g_next, w_dst, i + 1, j - 1)
                                if val_child <= NEG_INF:
                                    continue
                                val = np.log(w_g * w_t) + val_child
                                _set(g, w, i, j, val)

            # --- L-output insert: WPTT inserts left ---
            for w_src, rules in trans_insert.items():
                if is_ready_state(w_src):
                    continue  # inserts only from non-ready states
                for out_tok, w_dst, w_t in rules:
                    if out_tok != target_out:
                        continue
                    for g in range(n_gen):
                        val_child = _get(g, w_dst, i + 1, j)
                        if val_child <= NEG_INF:
                            continue
                        val = np.log(w_t) + val_child
                        _set(g, w_src, i, j, val)

            # --- R-output insert: WPTT inserts right ---
            for w_src, rules in trans_insert.items():
                if is_ready_state(w_src):
                    continue
                for out_tok, w_dst, w_t in rules:
                    if out_tok != target_out_r:
                        continue
                    for g in range(n_gen):
                        val_child = _get(g, w_dst, i, j - 1)
                        if val_child <= NEG_INF:
                            continue
                        val = np.log(w_t) + val_child
                        _set(g, w_src, i, j, val)

            # --- Bifurcation: gen + WPTT both bifurcate ---
            for g in range(n_gen):
                for g_left, g_right, w_g in gen_binary.get(g, []):
                    for w in range(n_trans):
                        if not is_ready_state(w):
                            continue
                        for w_l, w_r, w_t in trans_bif.get(w, []):
                            log_wgt = np.log(w_g * w_t)
                            for k in range(i, j + 1):
                                val_l = _get(g_left, w_l, i, k)
                                val_r = _get(g_right, w_r, k, j)
                                if val_l <= NEG_INF or val_r <= NEG_INF:
                                    continue
                                val = log_wgt + val_l + val_r
                                _set(g, w, i, j, val)

            # Apply span-preserving closure
            _unary_closure(i, j)

    log_prob = _get(generator.start, WPTT_START, 0, L)
    return log_prob, log_I


# ---------------------------------------------------------------------------
# Dense numpy Outside algorithm
# ---------------------------------------------------------------------------

def _add_scaled(P_O, T_O, i, j, contrib, log_scale, NEG_INF=-1e30):
    """Add contrib (scaled by exp(log_scale)) to P_O[:, i, j] / T_O[i, j].

    Handles the running reference scale so that
        P_O[:, i, j] * exp(T_O[i, j])
    always equals the true accumulated outside probability.
    """
    if np.all(contrib == 0):
        return
    if T_O[i, j] <= NEG_INF:
        P_O[:, i, j] = contrib
        T_O[i, j] = log_scale
    elif log_scale > T_O[i, j]:
        P_O[:, i, j] *= np.exp(T_O[i, j] - log_scale)
        P_O[:, i, j] += contrib
        T_O[i, j] = log_scale
    else:
        P_O[:, i, j] += contrib * np.exp(log_scale - T_O[i, j])


def factored_outside_jax(generator, transducer, leaf_seq, log_I):
    """Dense factored Outside for (SCFG × WPTT) on a nucleotide leaf sequence.

    Probability-space DP with per-cell scaling, sparse closure^T, and bincount.
    Processes spans top-down: at each cell, first applies closure^T in-place
    to finalize, then scatters to children.

    The stored P_O/T_O values are the CLOSED Outside probabilities (after
    closure^T), matching the convention that Inside stores CLOSED values
    (after closure). This ensures the IO identity holds:
        sum_c I[c,i,j] * O[c,i,j] = Z  for all spans [i,j).

    Degenerate bifurcation (k=i, k=j) is excluded: in Inside, these
    contribute zero because the current cell hasn't been filled yet when
    the bif loop runs. For consistency, Outside also excludes them.

    Args:
        generator: WCFG (the order-1 singlet SCFG, 10 NTs)
        transducer: WPTT object (246 states)
        leaf_seq: numpy array of nucleotide indices (0-3), shape (L,)
        log_I: (N_C, L+1, L+1) log inside table from factored_inside_jax

    Returns:
        log_O: (N_C, L+1, L+1) log outside probabilities (closed)
    """
    ct = _get_compiled(generator, transducer)
    L = len(leaf_seq)
    N_C = ct.n_composite
    NEG_INF = -1e30

    # Convert log_I to probability-space (P_I, S_I)
    P_I = np.zeros((N_C, L + 1, L + 1), dtype=np.float64)
    S_I = np.full((L + 1, L + 1), NEG_INF, dtype=np.float64)
    for span in range(L + 1):
        for i in range(L - span + 1):
            j = i + span
            col = log_I[:, i, j]
            mx = col.max()
            if mx > NEG_INF:
                P_I[:, i, j] = np.exp(col - mx)
                S_I[i, j] = mx

    # Closure^T (sparse)
    closure_T = ct.closure.T.tocsr()

    # Outside tables: O_true[c, i, j] = P_O[c, i, j] * exp(T_O[i, j])
    # Values accumulate raw contributions from parents, then get closed.
    P_O = np.zeros((N_C, L + 1, L + 1), dtype=np.float64)
    T_O = np.full((L + 1, L + 1), NEG_INF, dtype=np.float64)

    # Initialize: O[start, 0, L] = 1.0 (pre-closure raw)
    start_c = generator.start * ct.n_wptt + WPTT_START
    P_O[start_c, 0, L] = 1.0
    T_O[0, L] = 0.0

    # Pre-compute probability weights (same as Inside)
    pw = {}
    for name in ('l_match', 'r_match', 'l_insert', 'r_insert'):
        d = getattr(ct, name)
        pw[name] = {}
        for nuc, (src, dst, lw) in d.items():
            pw[name][nuc] = np.exp(lw)
    pw['lr_match'] = {}
    for key, (src, dst, lw) in ct.lr_match.items():
        pw['lr_match'][key] = np.exp(lw)
    bif_pw = np.exp(ct.bif_log_w)

    # Top-down: decreasing span length
    for span in range(L, -1, -1):
        for i in range(L - span + 1):
            j = i + span

            if T_O[i, j] <= NEG_INF:
                continue

            # === Step 1: Apply closure^T in place to finalize this cell ===
            closed = closure_T @ P_O[:, i, j]
            mx = closed.max()
            if mx > 0:
                P_O[:, i, j] = closed / mx
                T_O[i, j] += np.log(mx)
            else:
                P_O[:, i, j] = 0.0
                T_O[i, j] = NEG_INF
                continue

            # === Step 2: Scatter closed values to children ===
            o_closed = P_O[:, i, j]
            parent_scale = T_O[i, j]

            if span >= 1:
                nuc_i = int(leaf_seq[i])
                nuc_j = int(leaf_seq[j - 1])

                # --- Contributions to child (i+1, j): L-match + L-insert ---
                contrib_ij1 = np.zeros(N_C, dtype=np.float64)
                if nuc_i in ct.l_match:
                    src, dst, _ = ct.l_match[nuc_i]
                    w = pw['l_match'][nuc_i]
                    contrib_ij1 += np.bincount(
                        dst, weights=w * o_closed[src], minlength=N_C)
                if nuc_i in ct.l_insert:
                    src, dst, _ = ct.l_insert[nuc_i]
                    w = pw['l_insert'][nuc_i]
                    contrib_ij1 += np.bincount(
                        dst, weights=w * o_closed[src], minlength=N_C)
                _add_scaled(P_O, T_O, i + 1, j, contrib_ij1, parent_scale)

                # --- Contributions to child (i, j-1): R-match + R-insert ---
                contrib_ij2 = np.zeros(N_C, dtype=np.float64)
                if nuc_j in ct.r_match:
                    src, dst, _ = ct.r_match[nuc_j]
                    w = pw['r_match'][nuc_j]
                    contrib_ij2 += np.bincount(
                        dst, weights=w * o_closed[src], minlength=N_C)
                if nuc_j in ct.r_insert:
                    src, dst, _ = ct.r_insert[nuc_j]
                    w = pw['r_insert'][nuc_j]
                    contrib_ij2 += np.bincount(
                        dst, weights=w * o_closed[src], minlength=N_C)
                _add_scaled(P_O, T_O, i, j - 1, contrib_ij2, parent_scale)

                # --- Contributions to child (i+1, j-1): LR-match ---
                if span >= 2:
                    key = (nuc_i, nuc_j)
                    if key in ct.lr_match:
                        src, dst, _ = ct.lr_match[key]
                        w = pw['lr_match'][key]
                        contrib_lr = np.bincount(
                            dst, weights=w * o_closed[src], minlength=N_C)
                        _add_scaled(P_O, T_O, i + 1, j - 1,
                                    contrib_lr, parent_scale)

            # --- Bifurcation: scatter to left and right children ---
            # Exclude degenerate splits (k=j for left, k=i for right):
            # in Inside these contribute zero (cell unfilled when bif runs).
            if len(ct.bif_src) > 0:
                o_bif = bif_pw * o_closed[ct.bif_src]  # (n_bif,)

                # Left children (i, k) for k in [i, j):
                # vectorize gather, per-k scatter
                k_left = np.arange(i, j)
                if len(k_left) > 0:
                    r_vals_all = P_I[ct.bif_dst_r[:, None],
                                     k_left[None, :], j]    # (n_bif, j-i)
                    s_right = S_I[k_left, j]                 # (j-i,)
                    for d, k in enumerate(k_left):
                        if s_right[d] <= NEG_INF:
                            continue
                        contrib_l = np.bincount(
                            ct.bif_dst_l,
                            weights=o_bif * r_vals_all[:, d],
                            minlength=N_C)
                        _add_scaled(P_O, T_O, i, k, contrib_l,
                                    parent_scale + s_right[d])

                # Right children (k, j) for k in (i, j]:
                k_right = np.arange(i + 1, j + 1)
                if len(k_right) > 0:
                    l_vals_all = P_I[ct.bif_dst_l[:, None],
                                     i, k_right[None, :]]   # (n_bif, j-i)
                    s_left = S_I[i, k_right]                 # (j-i,)
                    for d, k in enumerate(k_right):
                        if s_left[d] <= NEG_INF:
                            continue
                        contrib_r = np.bincount(
                            ct.bif_dst_r,
                            weights=o_bif * l_vals_all[:, d],
                            minlength=N_C)
                        _add_scaled(P_O, T_O, k, j, contrib_r,
                                    parent_scale + s_left[d])

    # Final closure^T for span=0 cells (already done in loop above)
    # Convert to log-probability table
    log_O = np.full((N_C, L + 1, L + 1), NEG_INF, dtype=np.float64)
    for span in range(L + 1):
        for i in range(L - span + 1):
            j = i + span
            if T_O[i, j] > NEG_INF:
                pos = P_O[:, i, j] > 0
                log_O[pos, i, j] = np.log(P_O[pos, i, j]) + T_O[i, j]

    return log_O


def factored_expected_counts(generator, transducer, leaf_seq,
                             log_I, log_O):
    """Expected rule usage counts from Inside-Outside.

    Computes E[count of each transition type] for EM training.

    Args:
        generator, transducer, leaf_seq: as for factored_inside_jax
        log_I: (N_C, L+1, L+1) log inside table
        log_O: (N_C, L+1, L+1) log outside table

    Returns:
        counts: dict with keys:
            'l_match': (N_NUC,) expected L-match counts per output nucleotide
            'r_match': (N_NUC,) expected R-match counts per output nucleotide
            'lr_match': dict (nuc_l, nuc_r) -> count
            'l_insert': (N_NUC,) expected L-insert counts per output nucleotide
            'r_insert': (N_NUC,) expected R-insert counts per output nucleotide
            'bif': float, total expected bifurcation count
            'eps': float, total expected epsilon count
    """
    ct = _get_compiled(generator, transducer)
    L = len(leaf_seq)
    N_C = ct.n_composite
    NEG_INF = -1e30

    start_c = generator.start * ct.n_wptt + WPTT_START
    log_Z = log_I[start_c, 0, L]

    counts = {
        'l_match': np.zeros(N_NUC, dtype=np.float64),
        'r_match': np.zeros(N_NUC, dtype=np.float64),
        'lr_match': {},
        'l_insert': np.zeros(N_NUC, dtype=np.float64),
        'r_insert': np.zeros(N_NUC, dtype=np.float64),
        'bif': 0.0,
        'eps': 0.0,
    }

    # Epsilon counts: sum over all empty spans
    for i in range(L + 1):
        for idx, lw in zip(ct.eps_states, ct.eps_log_weights):
            val = lw + log_O[idx, i, i] - log_Z
            if val > NEG_INF:
                counts['eps'] += np.exp(val)

    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span
            nuc_i = int(leaf_seq[i])
            nuc_j = int(leaf_seq[j - 1])

            # L-match: O[src, i, j] * w * I[dst, i+1, j] / Z
            if nuc_i in ct.l_match:
                src, dst, lw = ct.l_match[nuc_i]
                vals = log_O[src, i, j] + lw + log_I[dst, i + 1, j] - log_Z
                counts['l_match'][nuc_i] += np.sum(np.exp(
                    np.where(vals > NEG_INF, vals, NEG_INF)))

            # R-match
            if nuc_j in ct.r_match:
                src, dst, lw = ct.r_match[nuc_j]
                vals = log_O[src, i, j] + lw + log_I[dst, i, j - 1] - log_Z
                counts['r_match'][nuc_j] += np.sum(np.exp(
                    np.where(vals > NEG_INF, vals, NEG_INF)))

            # LR-match
            if span >= 2:
                key = (nuc_i, nuc_j)
                if key in ct.lr_match:
                    src, dst, lw = ct.lr_match[key]
                    vals = (log_O[src, i, j] + lw +
                            log_I[dst, i + 1, j - 1] - log_Z)
                    c = np.sum(np.exp(
                        np.where(vals > NEG_INF, vals, NEG_INF)))
                    counts['lr_match'][key] = (
                        counts['lr_match'].get(key, 0.0) + c)

            # L-insert
            if nuc_i in ct.l_insert:
                src, dst, lw = ct.l_insert[nuc_i]
                vals = log_O[src, i, j] + lw + log_I[dst, i + 1, j] - log_Z
                counts['l_insert'][nuc_i] += np.sum(np.exp(
                    np.where(vals > NEG_INF, vals, NEG_INF)))

            # R-insert
            if nuc_j in ct.r_insert:
                src, dst, lw = ct.r_insert[nuc_j]
                vals = log_O[src, i, j] + lw + log_I[dst, i, j - 1] - log_Z
                counts['r_insert'][nuc_j] += np.sum(np.exp(
                    np.where(vals > NEG_INF, vals, NEG_INF)))

            # Bifurcation
            if len(ct.bif_src) > 0:
                for k in range(i, j + 1):
                    vals = (log_O[ct.bif_src, i, j] + ct.bif_log_w +
                            log_I[ct.bif_dst_l, i, k] +
                            log_I[ct.bif_dst_r, k, j] - log_Z)
                    counts['bif'] += np.sum(np.exp(
                        np.where(vals > NEG_INF, vals, NEG_INF)))

    return counts


def factored_expected_counts_detailed(generator, transducer, leaf_seq,
                                      log_I, log_O):
    """Extended expected counts for EM training.

    Returns everything from factored_expected_counts, plus:
    - Delete counts extracted via U_del
    - Match counts split by V (post-M/I) vs W (post-D) source
    - Match pairs as (input_nuc, output_nuc) -> weight for substitution M-step

    Args:
        generator, transducer, leaf_seq: as for factored_inside_jax
        log_I: (N_C, L+1, L+1) log inside table
        log_O: (N_C, L+1, L+1) log outside table

    Returns:
        counts: dict with keys from factored_expected_counts plus:
            'delete': float, total expected delete count
            'l_match_V': (N_NUC,) L-match counts from V/Start sources
            'l_match_W': (N_NUC,) L-match counts from W sources
            'r_match_V': (N_NUC,) R-match counts from V/Start sources
            'r_match_W': (N_NUC,) R-match counts from W sources
            'lr_match_V': dict (nuc_l, nuc_r) -> count from V/Start
            'lr_match_W': dict (nuc_l, nuc_r) -> count from W
            'subst_pairs': dict (in_nuc, out_nuc) -> weight
            'subst_pairs_lr': dict (in_l, in_r, out_l, out_r) -> weight
    """
    ct = _get_compiled(generator, transducer)
    L = len(leaf_seq)
    N_C = ct.n_composite
    NEG_INF = -1e30

    start_c = generator.start * ct.n_wptt + WPTT_START
    log_Z = log_I[start_c, 0, L]

    counts = {
        'l_match': np.zeros(N_NUC, dtype=np.float64),
        'r_match': np.zeros(N_NUC, dtype=np.float64),
        'lr_match': {},
        'l_insert': np.zeros(N_NUC, dtype=np.float64),
        'r_insert': np.zeros(N_NUC, dtype=np.float64),
        'bif': 0.0,
        'eps': 0.0,
        # Detailed counts
        'delete': 0.0,
        'l_match_V': np.zeros(N_NUC, dtype=np.float64),
        'l_match_W': np.zeros(N_NUC, dtype=np.float64),
        'r_match_V': np.zeros(N_NUC, dtype=np.float64),
        'r_match_W': np.zeros(N_NUC, dtype=np.float64),
        'lr_match_V': {},
        'lr_match_W': {},
        'subst_pairs': {},
        'subst_pairs_lr': {},
    }

    # --- Delete counts via U_del ---
    # E[del] = sum_{i,j} sum_{c1,c2} O_closed[c1,i,j] * U_del[c1,c2]
    #          * I_closed[c2,i,j] / Z
    # The closed tables are: closure^T @ O and closure @ I
    # But log_I and log_O already include closure (applied during DP).
    # U_del transitions are span-preserving, so c1 and c2 share same [i,j].
    U_del_coo = ct.U_del.tocoo()
    if U_del_coo.nnz > 0:
        del_src = U_del_coo.row  # c1
        del_dst = U_del_coo.col  # c2
        del_w = U_del_coo.data   # weight
        del_lw = np.log(np.maximum(del_w, 1e-300))
        for span in range(0, L + 1):
            for i in range(L - span + 1):
                j = i + span
                # O[c1, i, j] * U_del[c1,c2] * I[c2, i, j] / Z
                vals = (log_O[del_src, i, j] + del_lw +
                        log_I[del_dst, i, j] - log_Z)
                mask = vals > NEG_INF
                if np.any(mask):
                    counts['delete'] += np.sum(np.exp(vals[mask]))

    # --- Epsilon counts ---
    for i in range(L + 1):
        for idx, lw in zip(ct.eps_states, ct.eps_log_weights):
            val = lw + log_O[idx, i, i] - log_Z
            if val > NEG_INF:
                counts['eps'] += np.exp(val)

    # --- Token-consuming transitions ---
    for span in range(1, L + 1):
        for i in range(L - span + 1):
            j = i + span
            nuc_i = int(leaf_seq[i])
            nuc_j = int(leaf_seq[j - 1])

            # L-match
            if nuc_i in ct.l_match:
                src, dst, lw = ct.l_match[nuc_i]
                vals = log_O[src, i, j] + lw + log_I[dst, i + 1, j] - log_Z
                safe = np.where(vals > NEG_INF, vals, NEG_INF)
                weights = np.exp(safe)
                total = np.sum(weights)
                counts['l_match'][nuc_i] += total

                # V/W split
                from_V = ct.l_match_from_V[nuc_i]
                counts['l_match_V'][nuc_i] += np.sum(weights[from_V])
                counts['l_match_W'][nuc_i] += np.sum(weights[~from_V])

                # Substitution pairs
                in_nucs = ct.l_match_in_nuc[nuc_i]
                for in_n in np.unique(in_nucs):
                    mask = in_nucs == in_n
                    key = (int(in_n), nuc_i)
                    counts['subst_pairs'][key] = (
                        counts['subst_pairs'].get(key, 0.0) +
                        np.sum(weights[mask]))

            # R-match
            if nuc_j in ct.r_match:
                src, dst, lw = ct.r_match[nuc_j]
                vals = log_O[src, i, j] + lw + log_I[dst, i, j - 1] - log_Z
                safe = np.where(vals > NEG_INF, vals, NEG_INF)
                weights = np.exp(safe)
                total = np.sum(weights)
                counts['r_match'][nuc_j] += total

                from_V = ct.r_match_from_V[nuc_j]
                counts['r_match_V'][nuc_j] += np.sum(weights[from_V])
                counts['r_match_W'][nuc_j] += np.sum(weights[~from_V])

                in_nucs = ct.r_match_in_nuc[nuc_j]
                for in_n in np.unique(in_nucs):
                    mask = in_nucs == in_n
                    key = (int(in_n), nuc_j)
                    counts['subst_pairs'][key] = (
                        counts['subst_pairs'].get(key, 0.0) +
                        np.sum(weights[mask]))

            # LR-match
            if span >= 2:
                key = (nuc_i, nuc_j)
                if key in ct.lr_match:
                    src, dst, lw = ct.lr_match[key]
                    vals = (log_O[src, i, j] + lw +
                            log_I[dst, i + 1, j - 1] - log_Z)
                    safe = np.where(vals > NEG_INF, vals, NEG_INF)
                    weights = np.exp(safe)
                    total = np.sum(weights)
                    counts['lr_match'][key] = (
                        counts['lr_match'].get(key, 0.0) + total)

                    from_V = ct.lr_match_from_V[key]
                    counts['lr_match_V'][key] = (
                        counts['lr_match_V'].get(key, 0.0) +
                        np.sum(weights[from_V]))
                    counts['lr_match_W'][key] = (
                        counts['lr_match_W'].get(key, 0.0) +
                        np.sum(weights[~from_V]))

                    in_l, in_r = ct.lr_match_in_pair[key]
                    for idx in range(len(weights)):
                        if weights[idx] > 0:
                            pkey = (int(in_l[idx]), int(in_r[idx]),
                                    nuc_i, nuc_j)
                            counts['subst_pairs_lr'][pkey] = (
                                counts['subst_pairs_lr'].get(pkey, 0.0) +
                                weights[idx])

            # L-insert
            if nuc_i in ct.l_insert:
                src, dst, lw = ct.l_insert[nuc_i]
                vals = log_O[src, i, j] + lw + log_I[dst, i + 1, j] - log_Z
                counts['l_insert'][nuc_i] += np.sum(np.exp(
                    np.where(vals > NEG_INF, vals, NEG_INF)))

            # R-insert
            if nuc_j in ct.r_insert:
                src, dst, lw = ct.r_insert[nuc_j]
                vals = log_O[src, i, j] + lw + log_I[dst, i, j - 1] - log_Z
                counts['r_insert'][nuc_j] += np.sum(np.exp(
                    np.where(vals > NEG_INF, vals, NEG_INF)))

            # Bifurcation
            if len(ct.bif_src) > 0:
                for k in range(i, j + 1):
                    vals = (log_O[ct.bif_src, i, j] + ct.bif_log_w +
                            log_I[ct.bif_dst_l, i, k] +
                            log_I[ct.bif_dst_r, k, j] - log_Z)
                    counts['bif'] += np.sum(np.exp(
                        np.where(vals > NEG_INF, vals, NEG_INF)))

    return counts
