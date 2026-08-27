"""Dynamic programming algorithms for Pair HMMs.

1D (alignment-conditioned): uses jax.lax.associative_scan for O(log L) parallel prefix.
2D (full Pair HMM): uses anti-diagonal wavefront scan.

All algorithms work in log-space for numerical stability.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from ..core.params import S, M, I, D, E


def _find_e_idx(state_types):
    """Find the column index of the End state in the transition matrix.

    For TKF91/TKF92 (5 states), E is at index 4 (= E constant).
    For MixDom (22+ states), E is at index 1.
    """
    return jnp.argmax(state_types == E)


# --- Geometric binning for JIT cache reuse ---
#
# Bins are a nearest-int-rounded pure geomspace b_k = round(BIN_BASE *
# BIN_MULTIPLIER^k), strictly increasing, terminating at BIN_MAX. The
# only hyperparameter is BIN_MULTIPLIER:
#   smaller (e.g. 1.2)  -> denser bins, less padding waste per pair,
#                          more JIT compile events across the run
#   larger (e.g. 1.5)   -> fewer bins, more padding waste (worst case
#                          factor BIN_MULTIPLIER per axis), fewer JIT
#                          compiles
#
# At BIN_MULTIPLIER=1.2 (default, matches the historical 40-point
# np.geomspace(4, 4096) spec to within int-rounding noise) we get
# ~40 bins in [4, 4096] and worst-case 2D padding factor BIN_MULTIPLIER^2
# ~= 1.44; for the structurally O(L^4) M_obs gather the worst-case
# factor is BIN_MULTIPLIER^4 ~= 2.07. Pre-pended [0,1,2,3] for tiny
# inputs (forward_backward_1d_padded handles them via seq_length=0).
BIN_BASE = 4
BIN_MAX = 4096
BIN_MULTIPLIER = 1.2


def _make_geom_bins(base: int = BIN_BASE, max_bin: int = BIN_MAX,
                     multiplier: float = BIN_MULTIPLIER) -> list:
    """Strictly-increasing nearest-int-rounded geomspaced bins."""
    bins = [int(base)]
    val = float(base)
    while val < max_bin:
        val *= multiplier
        b = int(round(val))
        if b > bins[-1]:
            bins.append(b)
    if bins[-1] < max_bin:
        bins.append(int(max_bin))
    return bins


_GEOM_BINS = [0, 1, 2, 3] + _make_geom_bins()


def _pad_to_bin(n):
    """Round n up to next geometric bin size for JIT cache reuse.

    Returns at least 1 (0-length sequences are padded to 1 so that
    forward_backward_1d_padded can handle them via seq_length=0).

    Hack: TKFMIXDOM_MAX_PAD env var (if set) caps the result so that
    pad-bin can fall below the next geometric bin. Useful to fit OOM-
    prone families on small GPUs. NOTE: setting this disables JIT cache
    reuse between calls with different n (each lands on its own bin).
    """
    import os
    n = max(n, 1)
    cap = os.environ.get('TKFMIXDOM_MAX_PAD')
    if cap is not None:
        cap = int(cap)
        if n > cap:
            raise ValueError(f"_pad_to_bin: n={n} exceeds TKFMIXDOM_MAX_PAD={cap}")
        # Round n up to nearest multiple of 32 (or cap), whichever smaller
        rounded = min(((n + 31) // 32) * 32, cap)
        return max(rounded, 1)
    for b in _GEOM_BINS:
        if b >= n:
            return b
    # Beyond precomputed bins: round up to next power of 2
    return int(2 ** np.ceil(np.log2(max(n, 1))))


def _pad_seq(seq, target_len):
    """Pad a sequence with zeros to target_len."""
    if seq.shape[0] >= target_len:
        return seq
    return jnp.concatenate([seq, jnp.zeros(target_len - seq.shape[0], dtype=seq.dtype)])


# --- Log-space utilities ---

def logsumexp_pair(a, b):
    """Stable log(exp(a) + exp(b))."""
    return jnp.logaddexp(a, b)


NEG_INF = -1e30


def safe_log(x):
    """Log that maps exact zeros to NEG_INF instead of -inf or log(1e-30)≈-69.

    Use this instead of jnp.log(jnp.maximum(x, 1e-30)) to get true NEG_INF
    for structurally-zero entries. This matters for padded Forward-Backward
    where NEG_INF emissions must fully block probability flow.

    JAX evaluates both branches of jnp.where, so the inner maximum(x, 1e-300)
    prevents domain errors in jnp.log without affecting the output (the where
    selects NEG_INF for those entries).
    """
    return jnp.where(x > 0, jnp.log(jnp.maximum(x, 1e-300)), NEG_INF)


# --- 1D alignment-conditioned Forward ---

def forward_1d(log_trans, log_emissions, init_state=S, final_state=E):
    """Forward algorithm along a fixed alignment path.

    Handles non-emitting start/end states: init_state transitions into the
    first emitting position, and final_state is reached by a terminal
    transition (no emission) after the last position.

    Convention:
        alpha_t[k] = log P(o_0, ..., o_t, s_t = k)
        log_prob = logsumexp_k(alpha_{L-1}[k] + log_trans[k, final_state])

    Args:
        log_trans: (n_states, n_states) log transition matrix
        log_emissions: (L, n_states) log emission probabilities at each position
        init_state: start state index (non-emitting)
        final_state: end state index (non-emitting)

    Returns:
        log_prob: total log probability
        alphas: (L, n_states) forward log probabilities
    """
    n_states = log_trans.shape[0]

    def scan_fn(alpha_prev, log_emit):
        alpha_new = jax.nn.logsumexp(alpha_prev[:, None] + log_trans, axis=0) + log_emit
        return alpha_new, alpha_new

    alpha_init = jnp.full(n_states, NEG_INF).at[init_state].set(0.0)
    alpha_first = jax.nn.logsumexp(alpha_init[:, None] + log_trans, axis=0) + log_emissions[0]

    _, alphas = jax.lax.scan(scan_fn, alpha_first, log_emissions[1:])
    alphas = jnp.concatenate([alpha_first[None, :], alphas], axis=0)

    # Terminal transition to final_state (no emission)
    log_prob = jax.nn.logsumexp(alphas[-1] + log_trans[:, final_state])
    return log_prob, alphas


def backward_1d(log_trans, log_emissions, final_state=E):
    """Backward algorithm along a fixed alignment path.

    Convention:
        beta_t[k] = log P(o_{t+1}, ..., o_{L-1}, end | s_t = k)
        beta_{L-1}[k] = log_trans[k, final_state]  (terminal transition)

    Args:
        log_trans: (n_states, n_states) log transition matrix
        log_emissions: (L, n_states) log emission probabilities at each position
        final_state: end state index (non-emitting)

    Returns:
        betas: (L, n_states) backward log probabilities
    """
    n_states = log_trans.shape[0]
    L = log_emissions.shape[0]

    def scan_fn(beta_next, log_emit):
        # beta_t[i] = logsumexp_j(trans[i,j] + emit[t+1,j] + beta_{t+1}[j])
        beta_new = jax.nn.logsumexp(log_trans + log_emit[None, :] + beta_next[None, :], axis=1)
        return beta_new, beta_new

    # Base case: terminal transition to final_state (no emission)
    beta_last = log_trans[:, final_state]

    if L == 1:
        return beta_last[None, :]

    # Scan backward: beta_t uses emit[t+1], so process emit[L-1]..emit[1]
    _, betas_rev = jax.lax.scan(scan_fn, beta_last, log_emissions[1:][::-1])
    betas = jnp.concatenate([betas_rev[::-1], beta_last[None, :]], axis=0)

    return betas


def forward_backward_1d(log_trans, log_emissions, init_state=S, final_state=E):
    """Forward-Backward algorithm for 1D alignment-conditioned DP.

    WARNING: This is a plain Python reference implementation (~0.3s per call).
    Not JIT-compiled. For production use, prefer forward_backward_1d_associative
    (O(log L) via jax.lax.associative_scan). For alignment-constrained BW where
    the state path is known, FB is unnecessary — direct counting of transition
    and emission events from the known path is sufficient and instantaneous.

    Convention:
        alpha_t[k] = log P(o_0..o_t, s_t=k)
        beta_t[k] = log P(o_{t+1}..o_{L-1}, end | s_t=k)
        P(s_t=k | data) = exp(alpha_t[k] + beta_t[k] - log_prob)

    Returns:
        log_prob: total log probability
        posteriors: (L, n_states) posterior state probabilities
        expected_trans: (n_states, n_states) expected transition counts
    """
    log_prob, alphas = forward_1d(log_trans, log_emissions, init_state, final_state)
    betas = backward_1d(log_trans, log_emissions, final_state)

    posteriors = jnp.exp(alphas + betas - log_prob)

    L = log_emissions.shape[0]
    n_states = log_trans.shape[0]

    # Interior transition counts: P(s_t=i, s_{t+1}=j | data) for t=0..L-2
    def trans_count(t):
        log_counts = (alphas[t][:, None] + log_trans +
                      log_emissions[t + 1][None, :] + betas[t + 1][None, :] - log_prob)
        return jnp.exp(log_counts)

    expected_trans = jnp.zeros((n_states, n_states))
    if L > 1:
        expected_trans = jnp.sum(jax.vmap(trans_count)(jnp.arange(L - 1)), axis=0)

    # Terminal: P(s_{L-1}=i, end | data)
    log_term = alphas[-1] + log_trans[:, final_state] - log_prob
    expected_trans = expected_trans.at[:, final_state].add(jnp.exp(log_term))

    # Initial: P(start, s_0=j | data)
    alpha_init = jnp.full(n_states, NEG_INF).at[init_state].set(0.0)
    log_init = (alpha_init[:, None] + log_trans +
                log_emissions[0][None, :] + betas[0][None, :] - log_prob)
    expected_trans = expected_trans + jnp.exp(log_init)

    return log_prob, posteriors, expected_trans


# --- 1D associative-scan Forward-Backward (O(log L) depth) ---

def _logsemiring_matmul(A, B):
    """Matrix multiply in log-semiring: C[...,i,j] = logsumexp_k(A[...,i,k] + B[...,k,j]).

    Supports batched inputs (extra leading dimensions) from associative_scan.
    """
    return jax.nn.logsumexp(A[..., :, :, None] + B[..., None, :, :], axis=-2)


def _logsemiring_combine(a, b):
    """Associative combine: compose transfer matrices in log-semiring."""
    return _logsemiring_matmul(a, b)


def _logsemiring_combine_rev(a, b):
    """Reversed associative combine: b @ a (for suffix products)."""
    return _logsemiring_matmul(b, a)


def forward_1d_associative(log_trans, log_emissions, init_state=S, final_state=E,
                           seq_length=None):
    """Forward algorithm using associative scan for O(log L) parallel depth.

    Transfer matrix: T_t[i,j] = log_trans[i,j] + log_emit[t,j]
    Prefix product: P_t = T_0 @ T_1 @ ... @ T_t
    alpha_t[j] = logsumexp_i(P_t[i,j] + alpha_init[i])
    log_prob = logsumexp_k(alpha_{L-1}[k] + log_trans[k, final_state])

    Args:
        log_trans: (n_states, n_states) log transition matrix
        log_emissions: (L, n_states) log emission probs per position
        init_state: start state index (non-emitting)
        final_state: end state index (non-emitting)
        seq_length: real sequence length (for padded emissions). If None,
            uses full emission array length.

    Returns:
        log_prob: total log probability
        alphas: (L, n_states) forward log probabilities
    """
    n_states = log_trans.shape[0]

    # T_t[i,j] = log_trans[i,j] + log_emit[t,j], shape: (L, n, n)
    transfers = log_trans[None, :, :] + log_emissions[:, None, :]

    # Prefix product: P_t = T_0 @ T_1 @ ... @ T_t
    prefix_products = jax.lax.associative_scan(_logsemiring_combine, transfers, axis=0)

    alpha_init = jnp.full(n_states, NEG_INF).at[init_state].set(0.0)

    # alpha_t[j] = logsumexp_i(P_t[i,j] + alpha_init[i])
    alphas = jax.nn.logsumexp(prefix_products + alpha_init[None, :, None], axis=1)

    # Terminal transition (no emission) — use real length when padded
    last_alpha = alphas[seq_length - 1] if seq_length is not None else alphas[-1]
    log_prob = jax.nn.logsumexp(last_alpha + log_trans[:, final_state])
    return log_prob, alphas


def backward_1d_associative(log_trans, log_emissions, final_state=E):
    """Backward algorithm using associative scan for O(log L) parallel depth.

    Transfer matrix: T_t[i,j] = log_trans[i,j] + log_emit[t,j]
    Suffix product: S_t = T_{t+1} @ T_{t+2} @ ... @ T_{L-1}
    beta_t[i] = logsumexp_j(S_t[i,j] + beta_term[j])
    beta_{L-1}[i] = beta_term[i] = log_trans[i, final_state]

    Uses reversed associative combine to get correct matrix ordering.

    Returns:
        betas: (L, n_states) backward log probabilities
    """
    n_states = log_trans.shape[0]
    L = log_emissions.shape[0]

    beta_term = log_trans[:, final_state]

    if L == 1:
        return beta_term[None, :]

    # T_t[i,j] = log_trans[i,j] + log_emit[t,j]
    transfers = log_trans[None, :, :] + log_emissions[:, None, :]  # (L, n, n)

    # We need suffix products S_t = T_{t+1} @ ... @ T_{L-1} for t=0..L-2
    # Use transfers[1:] (positions 1..L-1), reverse, scan with reversed combine, reverse back.
    # suffix_transfers has positions 1..L-1 (L-1 elements)
    suffix_transfers = transfers[1:]  # (L-1, n, n)
    reversed_t = suffix_transfers[::-1]  # [T_{L-1}, T_{L-2}, ..., T_1]

    # With _logsemiring_combine_rev(a, b) = b @ a:
    # prefix_rev[0] = T_{L-1}
    # prefix_rev[1] = T_{L-2} @ T_{L-1}
    # prefix_rev[k] = T_{L-1-k} @ ... @ T_{L-1}
    prefix_rev = jax.lax.associative_scan(_logsemiring_combine_rev, reversed_t, axis=0)

    # After reversing: suffix_arr[s] = T_{s+1} @ ... @ T_{L-1} for s=0..L-2
    suffix_arr = prefix_rev[::-1]  # (L-1, n, n)

    # beta_t[i] = logsumexp_j(S_t[i,j] + beta_term[j]) for t=0..L-2
    betas_prefix = jax.nn.logsumexp(suffix_arr + beta_term[None, None, :], axis=2)

    # Concatenate with beta_{L-1} = beta_term
    betas = jnp.concatenate([betas_prefix, beta_term[None, :]], axis=0)

    return betas


def forward_backward_1d_associative(log_trans, log_emissions,
                                     init_state=S, final_state=E):
    """Forward-Backward using associative scan for O(log L) parallel depth.

    Same conventions as forward_backward_1d (sequential) but O(log L) depth.

    Returns:
        log_prob: total log probability
        posteriors: (L, n_states) posterior state probabilities
        expected_trans: (n_states, n_states) expected transition counts
    """
    log_prob, alphas = forward_1d_associative(log_trans, log_emissions,
                                               init_state, final_state)
    betas = backward_1d_associative(log_trans, log_emissions, final_state)

    posteriors = jnp.exp(alphas + betas - log_prob)

    L = log_emissions.shape[0]
    n_states = log_trans.shape[0]

    # Interior transitions: P(s_t=i, s_{t+1}=j | data)
    def trans_count(t):
        log_counts = (alphas[t][:, None] + log_trans +
                      log_emissions[t + 1][None, :] + betas[t + 1][None, :] - log_prob)
        return jnp.exp(log_counts)

    expected_trans = jnp.zeros((n_states, n_states))
    if L > 1:
        expected_trans = jnp.sum(jax.vmap(trans_count)(jnp.arange(L - 1)), axis=0)

    # Terminal: P(s_{L-1}=i, end | data)
    log_term = alphas[-1] + log_trans[:, final_state] - log_prob
    expected_trans = expected_trans.at[:, final_state].add(jnp.exp(log_term))

    # Initial: P(start, s_0=j | data)
    alpha_init = jnp.full(n_states, NEG_INF).at[init_state].set(0.0)
    log_init = (alpha_init[:, None] + log_trans +
                log_emissions[0][None, :] + betas[0][None, :] - log_prob)
    expected_trans = expected_trans + jnp.exp(log_init)

    return log_prob, posteriors, expected_trans


def forward_1d_scan(log_trans, log_emissions, init_state=S, final_state=E,
                     seq_length=None):
    """Forward via jax.lax.scan (sequential, O(L) depth, O(L·n + n²) memory).

    Counterpart to forward_1d_associative which uses associative_scan
    (O(log L) depth, O(L·n²) memory). For long sequences and large
    state spaces (e.g. d3f3 with ns=47, L=1024) the associative path's
    (L, n, n) prefix-product tensor is the dominant memory consumer
    when vmapped over a batch — this scan variant trades parallel
    depth for memory and lets larger configurations fit.

    Same I/O as forward_1d_associative.

    Returns:
        (log_prob, alphas) — log_prob is the total log probability;
        alphas has shape (L, n_states).
    """
    n_states = log_trans.shape[0]
    L = log_emissions.shape[0]

    alpha_init = jnp.full(n_states, NEG_INF).at[init_state].set(0.0)
    alpha0 = jax.nn.logsumexp(
        alpha_init[:, None] + log_trans, axis=0) + log_emissions[0]

    def _step(alpha_prev, log_emit_t):
        # alpha_t[j] = logsumexp_i(alpha_{t-1}[i] + log_trans[i, j]) + log_emit[t, j]
        alpha_t = jax.nn.logsumexp(
            alpha_prev[:, None] + log_trans, axis=0) + log_emit_t
        return alpha_t, alpha_t

    _, alphas_rest = jax.lax.scan(_step, alpha0, log_emissions[1:])
    alphas = jnp.concatenate([alpha0[None, :], alphas_rest], axis=0)

    last_alpha = alphas[seq_length - 1] if seq_length is not None else alphas[-1]
    log_prob = jax.nn.logsumexp(last_alpha + log_trans[:, final_state])
    return log_prob, alphas


def backward_1d_scan(log_trans, log_emissions, final_state=E):
    """Backward via jax.lax.scan (O(L) depth, O(L·n + n²) memory).

    Counterpart to backward_1d_associative. Same I/O.

    Returns:
        betas: (L, n_states).
    """
    n_states = log_trans.shape[0]
    L = log_emissions.shape[0]

    beta_term = log_trans[:, final_state]                       # (n,)
    if L == 1:
        return beta_term[None, :]

    def _step(beta_next, log_emit_next):
        # beta_t[i] = logsumexp_j(log_trans[i, j] + log_emit_{t+1}[j] + beta_{t+1}[j])
        beta_t = jax.nn.logsumexp(
            log_trans + log_emit_next[None, :] + beta_next[None, :], axis=1)
        return beta_t, beta_t

    # Reverse scan: process positions L-1..0. The "log_emit_next"
    # consumed at iteration k corresponds to position L-1-k.
    log_emit_rev = log_emissions[::-1]                          # (L, n)
    _, betas_rev = jax.lax.scan(
        _step, beta_term, log_emit_rev[:L - 1])                  # (L-1, n)
    betas_prefix = betas_rev[::-1]                              # positions 0..L-2
    betas = jnp.concatenate([betas_prefix, beta_term[None, :]], axis=0)
    return betas


def forward_backward_1d_padded(log_trans, log_emissions, seq_length,
                                init_state=S, final_state=E,
                                forward_only=False,
                                scan_mode='associative'):
    """Padded Forward-Backward for vmap batching over variable-length sequences.

    Same as forward_backward_1d_associative but supports padding:
    - log_emissions has shape (padded_L, n_states) where padded_L >= seq_length
    - Positions 0..seq_length-1 are real; seq_length..padded_L-1 are padding
    - Padding positions must have NEG_INF emissions (standard convention)
    - seq_length is a dynamic (traced) integer for vmap compatibility

    The forward pass uses seq_length to pick the terminal alpha.
    The backward pass uses neutral (zero) emissions at padded positions
    so the terminal condition propagates cleanly through padding.

    Args:
        scan_mode: 'associative' (default) uses jax.lax.associative_scan
            for O(log L) parallel depth at the cost of O(L·n²) memory
            per pair (the materialised prefix/suffix product tensors).
            'sequential' uses jax.lax.scan: O(L) depth but O(L·n + n²)
            memory. Pick 'sequential' when n is large enough that the
            associative tensor dominates the GPU budget — e.g. MixDom's
            d3f3 with ns=47 and L=1024 vmapped over B=200 spent ~1.8 GiB
            on the prefix-product tensor alone, and on a 11 GiB GPU
            that combined with class-aware accumulators caused d3f3c27
            to OOM at JIT-compile time.

    Returns:
        log_prob: total log probability
        posteriors: (padded_L, n_states) posterior probs (zero at padded positions)
        expected_trans: (n_states, n_states) expected transition counts
    """
    if forward_only:
        # Fast path: only compute log_prob, skip backward pass
        if scan_mode == 'sequential':
            return forward_1d_scan(log_trans, log_emissions,
                                    init_state, final_state,
                                    seq_length=seq_length)[0]
        return forward_1d_associative(log_trans, log_emissions,
                                       init_state, final_state,
                                       seq_length=seq_length)[0]

    padded_L = log_emissions.shape[0]
    n_states = log_trans.shape[0]

    if scan_mode == 'sequential':
        # Sequential lax.scan: O(L) depth, O(L·n + n²) memory per pair.
        # Avoids the (L, n, n) prefix/suffix tensors that dominate the
        # vmapped associative path's footprint at large n (e.g. d3f3
        # ns=47 with B=200, L=1024: ~1.8 GiB → ~8 MiB).
        log_prob, alphas = forward_1d_scan(log_trans, log_emissions,
                                            init_state, final_state,
                                            seq_length=seq_length)
    else:
        # Forward: O(log L) associative scan with seq_length terminal support
        log_prob, alphas = forward_1d_associative(log_trans, log_emissions,
                                                   init_state, final_state,
                                                   seq_length=seq_length)

    # Backward: same self-loop padding trick across both modes.
    log_trans_bwd = log_trans.at[final_state, final_state].set(0.0)

    pos_idx = jnp.arange(padded_L)
    state_idx = jnp.arange(n_states)
    is_padded = pos_idx[:, None] >= seq_length
    is_final = state_idx[None, :] == final_state
    bwd_emissions = jnp.where(is_padded & is_final, 0.0,
                     jnp.where(is_padded, NEG_INF, log_emissions))

    if scan_mode == 'sequential':
        betas = backward_1d_scan(log_trans_bwd, bwd_emissions, final_state)
    else:
        betas = backward_1d_associative(log_trans_bwd, bwd_emissions, final_state)

    # Posteriors (zero at padded positions via forward NEG_INF → exp gives 0)
    posteriors = jnp.exp(alphas + betas - log_prob)

    if scan_mode == 'sequential':
        # Sequential accumulation of expected transition counts: avoids
        # the (L-1, n, n) materialised tensor of the vmap path. Memory
        # collapses from O(L·n²) to O(n²) for the count accumulator.
        def _step_trans(carry, t):
            valid = (t < seq_length - 1).astype(jnp.float32)
            log_counts = (alphas[t][:, None] + log_trans
                          + log_emissions[t + 1][None, :]
                          + betas[t + 1][None, :] - log_prob)
            return carry + jnp.exp(log_counts) * valid, None

        if padded_L > 1:
            expected_trans, _ = jax.lax.scan(
                _step_trans, jnp.zeros((n_states, n_states)),
                jnp.arange(padded_L - 1))
        else:
            expected_trans = jnp.zeros((n_states, n_states))
    else:
        # Interior transitions: only count real positions (0..seq_length-2)
        def trans_count(t):
            # Zero out contribution if t >= seq_length - 1 (past real sequence)
            valid = (t < seq_length - 1).astype(jnp.float32)
            log_counts = (alphas[t][:, None] + log_trans +
                          log_emissions[t + 1][None, :] + betas[t + 1][None, :] - log_prob)
            return jnp.exp(log_counts) * valid

        expected_trans = jnp.zeros((n_states, n_states))
        if padded_L > 1:
            expected_trans = jnp.sum(jax.vmap(trans_count)(jnp.arange(padded_L - 1)), axis=0)

    # Terminal at real end: P(s_{seq_length-1}=i, end | data).
    # Pin the start-index dtype to seq_length's so dynamic_slice doesn't
    # mix int32 and int64 — that mismatch fires lax.concatenate when
    # seq_length is int32 (from a vmapped int32 lengths array) and the
    # literal 0 promotes to int64.
    seq_idx_dtype = jnp.asarray(seq_length).dtype
    last_alpha = jax.lax.dynamic_slice(
        alphas,
        (seq_length - 1, jnp.zeros((), dtype=seq_idx_dtype)),
        (1, n_states))[0]
    log_term = last_alpha + log_trans[:, final_state] - log_prob
    expected_trans = expected_trans.at[:, final_state].add(jnp.exp(log_term))

    # Initial: P(start, s_0=j | data)
    alpha_init = jnp.full(n_states, NEG_INF).at[init_state].set(0.0)
    log_init = (alpha_init[:, None] + log_trans +
                log_emissions[0][None, :] + betas[0][None, :] - log_prob)
    expected_trans = expected_trans + jnp.exp(log_init)

    # Handle 0-length sequences: log_prob = log_trans[init, final],
    # expected_trans has only the init→final count.
    is_empty = (seq_length == 0)
    empty_lp = log_trans[init_state, final_state]
    empty_trans = jnp.zeros((n_states, n_states)).at[init_state, final_state].set(1.0)
    log_prob = jnp.where(is_empty, empty_lp, log_prob)
    expected_trans = jnp.where(is_empty, empty_trans, expected_trans)
    posteriors = jnp.where(is_empty, jnp.zeros_like(posteriors), posteriors)

    return log_prob, posteriors, expected_trans


# --- 2D Pair HMM Forward-Backward ---
#
# Generic implementation for arbitrary Pair HMMs with M (match), I (insert),
# D (delete) state types. Takes precomputed emission log-probabilities.
# Works for TKF91, TKF92, MixDom, etc.


def _pair_hmm_emission(state_type, xi, yj, log_sub, log_pi):
    """Log emission probability for state_type at position (xi, yj)."""
    log_match = log_pi[xi] + log_sub[xi, yj]
    log_insert = log_pi[yj]
    log_delete = log_pi[xi]
    return jnp.where(
        state_type == M, log_match,
        jnp.where(state_type == I, log_insert,
                  jnp.where(state_type == D, log_delete, 0.0)))


def pair_hmm_emissions(state_types, x_seq, y_seq, sub_matrix, pi):
    """Precompute emission log-probabilities for a Pair HMM.

    Args:
        state_types: (ns,) state type codes (M=1, I=2, D=3)
        x_seq: (Lx,) ancestor sequence
        y_seq: (Ly,) descendant sequence
        sub_matrix: (A, A) substitution probability matrix
        pi: (A,) equilibrium distribution

    Returns:
        emit: (Lx+1, Ly+1, ns) log emission for each state at each grid position.
              Position (0,*) and (*,0) use dummy emissions (index 0).
    """
    ns = state_types.shape[0]
    log_sub = jnp.log(sub_matrix + 1e-30)
    log_pi = jnp.log(pi + 1e-30)

    # Pad to size 21 for wildcard support: index 20 gets log-prob 0 (prob 1)
    log_sub = jnp.pad(log_sub, ((0, 1), (0, 1)), constant_values=0.0)  # (21, 21)
    log_pi = jnp.pad(log_pi, (0, 1), constant_values=0.0)              # (21,)

    x_pad = jnp.concatenate([jnp.array([0]), x_seq])
    y_pad = jnp.concatenate([jnp.array([0]), y_seq])

    match_emit = log_pi[x_pad[:, None]] + log_sub[x_pad[:, None], y_pad[None, :]]  # (Lx+1, Ly+1)
    ins_emit = log_pi[y_pad]   # (Ly+1,)
    del_emit = log_pi[x_pad]   # (Lx+1,)

    # Build (Lx+1, Ly+1, ns) emission table
    is_M = (state_types == M)  # (ns,)
    is_I = (state_types == I)
    is_D = (state_types == D)

    emit = (is_M[None, None, :] * match_emit[:, :, None] +
            is_I[None, None, :] * ins_emit[None, :, None] +
            is_D[None, None, :] * del_emit[:, None, None])

    # Non-emitting states (S, E) get 0 emission
    is_emit = is_M | is_I | is_D
    emit = jnp.where(is_emit[None, None, :], emit, 0.0)

    return emit


def pair_hmm_emissions_per_class(state_types, x_seq, y_seq,
                                  class_sub_matrices, class_pis, class_dist,
                                  n_dom, n_frag):
    """Precompute emission log-probabilities with per-fragment class mixture.

    Each (domain d, fragment f) location's emissions are a log-mixture
    over per-fragment site classes:

        M[i, j, s] = log sum_c classdist[d, f, c] * class_pi[c, x[i]] *
                              class_P[c, x[i], y[j]]
        I[i, j, s] = log sum_c classdist[d, f, c] * class_pi[c, y[j]]
        D[i, j, s] = log sum_c classdist[d, f, c] * class_pi[c, x[i]]

    where d, f are derived from state s via the MixDom2 state layout
    (body_idx = s - 2, d = body_idx // (5 * n_frag),
     f = (body_idx % (5 * n_frag)) % n_frag).

    Args:
        state_types: (ns,) state type codes (M=1, I=2, D=3).
        x_seq, y_seq: (Lx,), (Ly,) integer sequences.
        class_sub_matrices: (C, A, A) per-class P(t) at the relevant t.
        class_pis: (C, A) per-class equilibrium.
        class_dist: (D, F, C) per-(domain, fragment) class distribution.
        n_dom, n_frag: model dimensions.

    Returns:
        emit: (Lx+1, Ly+1, ns) log emission for each state at each grid position.
    """
    ns = state_types.shape[0]
    A = class_sub_matrices.shape[-1]

    log_csubs = jnp.log(jnp.maximum(class_sub_matrices, 1e-30))  # (C, A, A)
    log_cpis = jnp.log(jnp.maximum(class_pis, 1e-30))            # (C, A)
    log_cd = jnp.where(class_dist > 0,
                       jnp.log(jnp.maximum(class_dist, 1e-300)),
                       NEG_INF)  # (D, F, C)

    # Pad to size 21 for wildcard support.
    log_csubs = jnp.pad(log_csubs, ((0, 0), (0, 1), (0, 1)),
                        constant_values=0.0)  # (C, 21, 21)
    log_cpis = jnp.pad(log_cpis, ((0, 0), (0, 1)),
                       constant_values=0.0)  # (C, 21)

    x_pad = jnp.concatenate([jnp.array([0]), x_seq])  # (Lx+1,)
    y_pad = jnp.concatenate([jnp.array([0]), y_seq])  # (Ly+1,)

    # Per-state (dom, frag) lookups → (ns,).
    dom_idx = jnp.zeros(ns, dtype=jnp.int32)
    frag_idx = jnp.zeros(ns, dtype=jnp.int32)
    body = jnp.arange(ns - 2)
    dom_idx = dom_idx.at[2:].set(body // (5 * n_frag))
    within_dom = body % (5 * n_frag)
    frag_idx = frag_idx.at[2:].set(within_dom % n_frag)

    # Per-state log w[s, c]: shape (ns, C). For S/E states the lookup
    # uses (0, 0) which is fine because is_emit masks them.
    state_log_w = log_cd[dom_idx, frag_idx, :]  # (ns, C)

    # Build per-class match emission table at each (i, j): log_cpis[c, x[i]]
    # + log_csubs[c, x[i], y[j]]. Shape (C, Lx+1, Ly+1).
    log_cpi_x = log_cpis[:, x_pad]                       # (C, Lx+1)
    log_csub_xy = log_csubs[:, x_pad][:, :, y_pad]       # (C, Lx+1, Ly+1)
    match_per_class = log_cpi_x[:, :, None] + log_csub_xy  # (C, Lx+1, Ly+1)
    log_cpi_y = log_cpis[:, y_pad]                       # (C, Ly+1)

    # logsumexp over c axis, computed via scan to avoid materialising the
    # (Lx+1, Ly+1, ns, C) intermediate tensor that the naïve broadcast
    # would build (108-class Annabel + 600x600 BAliBASE pair → 14.7 GiB,
    # OOMs an 11 GiB GPU). Scan reduces peak memory to O(Lx·Ly·ns).
    Lx_p = x_pad.shape[0]
    Ly_p = y_pad.shape[0]
    state_log_w_t = state_log_w.T  # (C, ns)

    def _scan_match(carry, inputs):
        match_c, log_w_c = inputs  # (Lx+1, Ly+1), (ns,)
        contrib = match_c[:, :, None] + log_w_c[None, None, :]  # (Lx+1, Ly+1, ns)
        return jnp.logaddexp(carry, contrib), None

    match_init = jnp.full((Lx_p, Ly_p, ns), NEG_INF, dtype=match_per_class.dtype)
    match_emit, _ = jax.lax.scan(
        _scan_match, match_init,
        (match_per_class, state_log_w_t))                # (Lx+1, Ly+1, ns)

    def _scan_ins(carry, inputs):
        cpi_y_c, log_w_c = inputs  # (Ly+1,), (ns,)
        contrib = cpi_y_c[:, None] + log_w_c[None, :]  # (Ly+1, ns)
        return jnp.logaddexp(carry, contrib), None

    ins_init = jnp.full((Ly_p, ns), NEG_INF, dtype=match_per_class.dtype)
    ins_emit, _ = jax.lax.scan(
        _scan_ins, ins_init, (log_cpi_y, state_log_w_t))  # (Ly+1, ns)

    def _scan_del(carry, inputs):
        cpi_x_c, log_w_c = inputs  # (Lx+1,), (ns,)
        contrib = cpi_x_c[:, None] + log_w_c[None, :]  # (Lx+1, ns)
        return jnp.logaddexp(carry, contrib), None

    del_init = jnp.full((Lx_p, ns), NEG_INF, dtype=match_per_class.dtype)
    del_emit, _ = jax.lax.scan(
        _scan_del, del_init, (log_cpi_x, state_log_w_t))  # (Lx+1, ns)

    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    emit = (is_M[None, None, :] * match_emit +
            is_I[None, None, :] * ins_emit[None, :, :] +
            is_D[None, None, :] * del_emit[:, None, :])
    is_emit = is_M | is_I | is_D
    emit = jnp.where(is_emit[None, None, :], emit, 0.0)
    return emit


def pair_hmm_emissions_per_domain(state_types, x_seq, y_seq,
                                   sub_matrices, pis, n_dom, n_frag):
    """Precompute emission log-probabilities with per-domain substitution.

    Each domain d uses its own substitution matrix sub_matrices[d] and
    equilibrium distribution pis[d]. States are assigned to domains by
    position in the state vector: body state s has domain (s-2) // (5*n_frag).

    Args:
        state_types: (ns,) state type codes (M=1, I=2, D=3)
        x_seq: (Lx,) ancestor sequence
        y_seq: (Ly,) descendant sequence
        sub_matrices: (n_dom, A, A) per-domain substitution probability matrices
        pis: (n_dom, A) per-domain equilibrium distributions
        n_dom: number of domains
        n_frag: number of fragments per domain

    Returns:
        emit: (Lx+1, Ly+1, ns) log emission for each state at each grid position.
    """
    ns = state_types.shape[0]
    A = sub_matrices.shape[-1]

    log_subs = jnp.log(jnp.maximum(sub_matrices, 1e-30))  # (n_dom, A, A)
    log_pis = jnp.log(jnp.maximum(pis, 1e-30))           # (n_dom, A)

    # Pad to size 21 for wildcard support: index 20 gets log-prob 0 (prob 1)
    log_subs = jnp.pad(log_subs, ((0, 0), (0, 1), (0, 1)), constant_values=0.0)  # (n_dom, 21, 21)
    log_pis = jnp.pad(log_pis, ((0, 0), (0, 1)), constant_values=0.0)             # (n_dom, 21)

    x_pad = jnp.concatenate([jnp.array([0]), x_seq])
    y_pad = jnp.concatenate([jnp.array([0]), y_seq])

    # Domain index for each state: SS=0, EE=0, body state s → (s-2)//(5*n_frag)
    dom_idx = jnp.zeros(ns, dtype=jnp.int32)
    body = jnp.arange(ns - 2)
    dom_idx = dom_idx.at[2:].set(body // (5 * n_frag))

    # Per-state log_pi and log_sub via domain lookup
    state_log_pi = log_pis[dom_idx]      # (ns, A)
    state_log_sub = log_subs[dom_idx]    # (ns, A, A)

    # Match: log_pi[dom, x[i]] + log_sub[dom, x[i], y[j]]
    # = state_log_pi[s, x[i]] + state_log_sub[s, x[i], y[j]]
    match_emit = (state_log_pi[:, x_pad][:, :, None] +
                  state_log_sub[:, x_pad][:, :, y_pad])  # (ns, Lx+1, Ly+1)
    match_emit = match_emit.transpose(1, 2, 0)  # (Lx+1, Ly+1, ns)

    # Insert: log_pi[dom, y[j]]
    ins_emit = state_log_pi[:, y_pad].T  # (Ly+1, ns)

    # Delete: log_pi[dom, x[i]]
    del_emit = state_log_pi[:, x_pad].T  # (Lx+1, ns)

    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    emit = (is_M[None, None, :] * match_emit +
            is_I[None, None, :] * ins_emit[None, :, :] +
            is_D[None, None, :] * del_emit[:, None, :])

    is_emit = is_M | is_I | is_D
    emit = jnp.where(is_emit[None, None, :], emit, 0.0)

    return emit


def pair_hmm_emissions_constrained_per_domain(state_types, state_seq,
                                                anc_chars, desc_chars,
                                                sub_matrices, pis,
                                                n_dom, n_frag):
    """JAX-native 1D constrained per-domain emissions for the alignment-
    aware (MSA-guided) DP.

    The alignment fixes per-column state type (M=1, I=2, D=3); only states
    of matching type get non-NEG_INF emissions at each column. This is the
    JAX-jittable, autograd-friendly counterpart to
    `mixdom_constrained_emissions_vectorized` (NumPy) in the train.constrained
    module — it yields the same per-(column, state) log-prob table but
    runs inside @jax.jit and supports custom_vjp.

    Inputs are filled-length: anc_chars[ℓ] is meaningful at columns where
    state_seq[ℓ] ∈ {M, D}; desc_chars[ℓ] at columns where state_seq[ℓ] ∈
    {M, I}. At the other type's positions the array entry is ignored (still
    indexed safely; the type mask zeroes its contribution).

    Args:
        state_types: (ns,) MixDom state type codes (M=1, I=2, D=3)
        state_seq: (L,) per-column state-type code (M=1, I=2, D=3)
        anc_chars: (L,) ancestor char per column (valid at M / D)
        desc_chars: (L,) descendant char per column (valid at M / I)
        sub_matrices: (n_dom, A, A) per-domain substitution probability matrices
        pis: (n_dom, A) per-domain equilibrium distributions
        n_dom: number of domains
        n_frag: fragments per domain

    Returns:
        log_emit: (L, ns) per-column-per-state log emission probability,
                  with NEG_INF at type-mismatched (column-state) entries.
    """
    ns = state_types.shape[0]

    log_subs = jnp.log(jnp.maximum(sub_matrices, 1e-30))   # (n_dom, A, A)
    log_pis = jnp.log(jnp.maximum(pis, 1e-30))             # (n_dom, A)

    # Wildcard padding (index 20 → log-prob 0 = prob 1) for parity with the
    # 2D path.
    log_subs = jnp.pad(log_subs, ((0, 0), (0, 1), (0, 1)),
                        constant_values=0.0)               # (n_dom, 21, 21)
    log_pis = jnp.pad(log_pis, ((0, 0), (0, 1)),
                       constant_values=0.0)                # (n_dom, 21)

    # Per-state domain lookup. SS=0, EE=0; body s → (s-2)//(5*n_frag).
    dom_idx = jnp.zeros(ns, dtype=jnp.int32)
    body = jnp.arange(ns - 2)
    dom_idx = dom_idx.at[2:].set(body // (5 * n_frag))

    state_log_pi = log_pis[dom_idx]                        # (ns, A_pad)
    state_log_sub = log_subs[dom_idx]                      # (ns, A_pad, A_pad)

    # Index into anc/desc per-column. Out-of-domain chars are clamped via the
    # subsequent type mask, so we don't need to filter here.
    anc_int = jnp.asarray(anc_chars, dtype=jnp.int32)
    desc_int = jnp.asarray(desc_chars, dtype=jnp.int32)

    # match_emit[ℓ, s] = log_pi[dom(s), anc[ℓ]] + log_sub[dom(s), anc[ℓ], desc[ℓ]].
    # Direct fancy indexing avoids the (ns, L, L) intermediate that the
    # naive cross-product would build.
    match_pi = state_log_pi[:, anc_int]                     # (ns, L)
    match_sub_diag = state_log_sub[:, anc_int, desc_int]    # (ns, L)
    match_emit = (match_pi + match_sub_diag).T              # (L, ns)

    # Insert: log_pi[dom(s), desc[ℓ]] at I-type columns.
    ins_emit = state_log_pi[:, desc_int].T                  # (L, ns)

    # Delete: log_pi[dom(s), anc[ℓ]] at D-type columns.
    del_emit = state_log_pi[:, anc_int].T                   # (L, ns)

    is_M_state = (state_types == M)                         # (ns,)
    is_I_state = (state_types == I)
    is_D_state = (state_types == D)

    is_M_col = (state_seq == M)                             # (L,)
    is_I_col = (state_seq == I)
    is_D_col = (state_seq == D)

    # (state, column) match: column type == state type.
    M_mask = is_M_col[:, None] & is_M_state[None, :]        # (L, ns)
    I_mask = is_I_col[:, None] & is_I_state[None, :]
    D_mask = is_D_col[:, None] & is_D_state[None, :]

    log_emit = jnp.where(M_mask, match_emit, NEG_INF)
    log_emit = jnp.where(I_mask, ins_emit, log_emit)
    log_emit = jnp.where(D_mask, del_emit, log_emit)

    return log_emit


def pair_hmm_emissions_constrained(state_types, state_seq,
                                     anc_chars, desc_chars,
                                     sub_matrices, pis, n_dom, n_frag,
                                     classdist=None,
                                     class_sub_matrices=None,
                                     class_pis=None):
    """Unified JAX 1D constrained emission builder. Dispatches between
    `pair_hmm_emissions_constrained_per_domain` (MixDom1) and
    `pair_hmm_emissions_constrained_per_class` (MixDom2 with site-class
    mixture) based on whether the classdist + per-class arguments are
    provided. Same role as `mixdom_constrained_emissions_vectorized` in
    `train.constrained` but JAX-native (jit + grad compatible).

    Args:
        state_types, state_seq, anc_chars, desc_chars, n_dom, n_frag:
            see pair_hmm_emissions_constrained_per_domain.
        sub_matrices, pis: per-domain (n_dom, A, A) / (n_dom, A) — used
            when classdist is None.
        classdist: optional (n_dom, n_frag, C). When provided, the
            per-class arguments must also be provided; the per-domain
            arguments are ignored.
        class_sub_matrices: (C, A, A).
        class_pis: (C, A).

    Returns:
        log_emit: (L, ns).
    """
    if classdist is not None and class_sub_matrices is not None and class_pis is not None:
        return pair_hmm_emissions_constrained_per_class(
            state_types, state_seq, anc_chars, desc_chars,
            class_sub_matrices, class_pis, classdist, n_dom, n_frag)
    return pair_hmm_emissions_constrained_per_domain(
        state_types, state_seq, anc_chars, desc_chars,
        sub_matrices, pis, n_dom, n_frag)


def pair_hmm_emissions_constrained_per_class(state_types, state_seq,
                                               anc_chars, desc_chars,
                                               class_sub_matrices, class_pis,
                                               class_dist, n_dom, n_frag):
    """JAX-native 1D constrained per-fragment-per-class emissions for the
    MixDom2 alignment-aware DP.

    Same I/O shape as `pair_hmm_emissions_constrained_per_domain` but
    each (domain, fragment) location's emissions are a log-mixture over
    per-fragment site classes:

        M[ℓ, s] = logsumexp_c(log classdist[d, f, c] +
                               log class_pis[c, anc[ℓ]] +
                               log class_subs[c, anc[ℓ], desc[ℓ]])
        I[ℓ, s] = logsumexp_c(log classdist[d, f, c] +
                               log class_pis[c, desc[ℓ]])
        D[ℓ, s] = logsumexp_c(log classdist[d, f, c] +
                               log class_pis[c, anc[ℓ]])

    where (d, f) are derived from state s via the MixDom2 state layout.

    Args:
        state_types: (ns,) state type codes (M=1, I=2, D=3)
        state_seq:   (L,) per-column state-type code (M=1, I=2, D=3)
        anc_chars:   (L,) ancestor char per column (valid at M / D)
        desc_chars:  (L,) descendant char per column (valid at M / I)
        class_sub_matrices: (C, A, A) per-class P(t)
        class_pis:   (C, A) per-class equilibrium
        class_dist:  (n_dom, n_frag, C) per-(domain, fragment) class distribution
        n_dom, n_frag: model dimensions

    Returns:
        log_emit: (L, ns)
    """
    ns = state_types.shape[0]
    n_cls = class_pis.shape[0]
    A = class_sub_matrices.shape[-1]

    log_csubs = jnp.log(jnp.maximum(class_sub_matrices, 1e-30))    # (C, A, A)
    log_cpis = jnp.log(jnp.maximum(class_pis, 1e-30))              # (C, A)
    log_cd = jnp.where(class_dist > 0,
                        jnp.log(jnp.maximum(class_dist, 1e-300)),
                        NEG_INF)                                    # (D, F, C)

    log_csubs = jnp.pad(log_csubs, ((0, 0), (0, 1), (0, 1)),
                         constant_values=0.0)                       # (C, 21, 21)
    log_cpis = jnp.pad(log_cpis, ((0, 0), (0, 1)),
                        constant_values=0.0)                        # (C, 21)

    # Per-state (dom, frag) lookup
    dom_idx = jnp.zeros(ns, dtype=jnp.int32)
    frag_idx = jnp.zeros(ns, dtype=jnp.int32)
    body = jnp.arange(ns - 2)
    dom_idx = dom_idx.at[2:].set(body // (5 * n_frag))
    within_dom = body % (5 * n_frag)
    frag_idx = frag_idx.at[2:].set(within_dom % n_frag)

    state_log_w = log_cd[dom_idx, frag_idx, :]                     # (ns, C)

    anc_int = jnp.asarray(anc_chars, dtype=jnp.int32)
    desc_int = jnp.asarray(desc_chars, dtype=jnp.int32)

    # Per-class per-column log-prob (vectorised fancy indexing).
    cls_match_emit = (log_cpis[:, anc_int]
                      + log_csubs[:, anc_int, desc_int])           # (C, L)
    cls_ins_emit = log_cpis[:, desc_int]                            # (C, L)
    cls_del_emit = log_cpis[:, anc_int]                             # (C, L)

    def _logsumexp_c(class_emit):
        """logsumexp over c of (state_log_w[s, c] + class_emit[c, ℓ]).

        Builds a (ns, C, L) combined-log tensor and reduces over c using
        the standard subtract-max trick. For ns·C·L ≈ 47·27·1024 = 1.3 M
        entries × 4 bytes = 5 MB this is small; the only concern would
        be unusually huge ns·C·L which is not a regime we care about
        for Pfam.
        """
        combined = (state_log_w[:, :, None]
                    + class_emit[None, :, :])                      # (ns, C, L)
        cmax = jnp.max(combined, axis=1, keepdims=True)            # (ns, 1, L)
        shifted = combined - cmax
        return cmax[:, 0, :] + jnp.log(jnp.sum(jnp.exp(shifted), axis=1))

    match_emit = _logsumexp_c(cls_match_emit).T                    # (L, ns)
    ins_emit = _logsumexp_c(cls_ins_emit).T
    del_emit = _logsumexp_c(cls_del_emit).T

    is_M_state = (state_types == M)
    is_I_state = (state_types == I)
    is_D_state = (state_types == D)
    is_M_col = (state_seq == M)
    is_I_col = (state_seq == I)
    is_D_col = (state_seq == D)

    M_mask = is_M_col[:, None] & is_M_state[None, :]
    I_mask = is_I_col[:, None] & is_I_state[None, :]
    D_mask = is_D_col[:, None] & is_D_state[None, :]

    log_emit = jnp.where(M_mask, match_emit, NEG_INF)
    log_emit = jnp.where(I_mask, ins_emit, log_emit)
    log_emit = jnp.where(D_mask, del_emit, log_emit)
    return log_emit


def mask_emissions_match_aligned(emit, state_types, match_positions_i, match_positions_j):
    """Mask 2D emission table to constrain matches to training alignment positions.

    For match-aligned training: match states can only fire at (i,j) positions
    from the training alignment. Insert/delete states are unconstrained.

    Args:
        emit: (Lx+1, Ly+1, ns) log emission table (already computed)
        state_types: (ns,) state type array (S=0, M=1, I=2, D=3, E=4)
        match_positions_i: (n_matches,) int array of x-sequence positions (0-indexed)
        match_positions_j: (n_matches,) int array of y-sequence positions (0-indexed)

    Returns:
        masked_emit: (Lx+1, Ly+1, ns) with match emissions at non-match positions set to NEG_INF
    """
    Lx1, Ly1, ns = emit.shape

    # Build boolean mask: True at grid positions (i+1, j+1) for each match
    # (grid is 1-indexed: position 0 is the boundary row/col)
    match_mask = jnp.zeros((Lx1, Ly1), dtype=bool)
    match_mask = match_mask.at[match_positions_i + 1, match_positions_j + 1].set(True)

    # For match states (state_types == M): set emission to NEG_INF where match_mask is False
    is_match_state = (state_types == M)  # (ns,)

    # Penalty: NEG_INF where (is_match_state AND NOT match_mask)
    penalty = jnp.where(
        is_match_state[None, None, :] & ~match_mask[:, :, None],
        NEG_INF,
        0.0
    )

    return emit + penalty


def _forward_2d_core(log_trans, state_types, emit, Lx, Ly):
    """Core forward algorithm for a Pair HMM.

    Generic over state types. States are classified by state_types:
      M-type (state_types==1): predecessor at (i-1, j-1)
      I-type (state_types==2): predecessor at (i, j-1)
      D-type (state_types==3): predecessor at (i-1, j)

    Args:
        log_trans: (ns, ns) log transition matrix
        state_types: (ns,) state type codes
        emit: (Lx+1, Ly+1, ns) log emission probabilities
        Lx, Ly: sequence lengths

    Returns:
        log_prob: total log probability
        F: (Lx+1, Ly+1, ns) forward table
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    # Row 0: only I-type transitions from (0,0)
    row0 = jnp.full((Ly + 1, ns), NEG_INF)
    row0 = row0.at[0, S].set(0.0)

    def row0_step(prev_cell, j):
        # For each I-type state k: F[0,j,k] = logsumexp(F[0,j-1,:] + trans[:,k]) + emit[0,j,k]
        raw = jax.nn.logsumexp(prev_cell[:, None] + log_trans, axis=0) + emit[0, j]
        cell = jnp.where(is_I, raw, NEG_INF)
        return cell, cell

    _, row0_rest = jax.lax.scan(row0_step, row0[0], jnp.arange(1, Ly + 1))
    row0 = jnp.concatenate([row0[0:1], row0_rest], axis=0)

    # Rows 1..Lx
    def row_step(prev_row, i):
        # Column 0: only D-type states
        raw0 = jax.nn.logsumexp(prev_row[0][:, None] + log_trans, axis=0) + emit[i, 0]
        cell0 = jnp.where(is_D, raw0, NEG_INF)

        def col_step(prev_cell, j):
            # M-type: from prev_row[j-1]  (i-1, j-1)
            m_val = jax.nn.logsumexp(prev_row[j - 1][:, None] + log_trans, axis=0) + emit[i, j]
            # I-type: from prev_cell      (i, j-1)
            i_val = jax.nn.logsumexp(prev_cell[:, None] + log_trans, axis=0) + emit[i, j]
            # D-type: from prev_row[j]    (i-1, j)
            d_val = jax.nn.logsumexp(prev_row[j][:, None] + log_trans, axis=0) + emit[i, j]

            cell = jnp.where(is_M, m_val, jnp.where(is_I, i_val, jnp.where(is_D, d_val, NEG_INF)))
            return cell, cell

        _, row_rest = jax.lax.scan(col_step, cell0, jnp.arange(1, Ly + 1))
        curr_row = jnp.concatenate([cell0[None], row_rest], axis=0)
        return curr_row, curr_row

    _, all_rows = jax.lax.scan(row_step, row0, jnp.arange(1, Lx + 1))

    F = jnp.concatenate([row0[None], all_rows], axis=0)
    e_idx = _find_e_idx(state_types)
    log_prob = jax.nn.logsumexp(F[Lx, Ly, :] + log_trans[:, e_idx])

    return log_prob, F


def _forward_2d_core_diag(log_trans, state_types, emit, Lx, Ly):
    """Anti-diagonal wavefront forward for 2D pair HMM.

    Same recursion as _forward_2d_core but parallelizes over each
    anti-diagonal d = i + j: cells on diagonal d depend only on cells on
    diagonals d-1 (I, D) and d-2 (M), so the cells on a single diagonal
    are independent and can be processed by jax.vmap. Sequential depth
    drops from O(Lx*Ly) to O(Lx+Ly).

    Carry is only two diagonals (d-1 and d-2), not the full grid.
    Diagonal cells are emitted as scan output and scattered into the
    full grid after the scan.

    Lx and Ly must be Python ints (typically the geometric-bin padded
    sizes used by forward_backward_2d).
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    # Max width of any anti-diagonal is min(Lx, Ly) + 1.
    D_max = min(Lx, Ly) + 1
    n_diags = Lx + Ly  # diagonals 1..Lx+Ly

    # Diagonal d=0 has one cell: (0, 0) with F[0,0,S]=0.
    diag0 = jnp.full((D_max, ns), NEG_INF)
    diag0 = diag0.at[0, S].set(0.0)

    def _i_min(d):
        return jnp.maximum(0, d - Ly)

    def compute_cell(prev, prev_prev, d, k):
        """Compute cell k on diagonal d from prev (d-1) and prev_prev (d-2)."""
        i = _i_min(d) + k
        j = d - i
        # Predecessor local indices on their diagonals
        # M: (i-1, j-1) on d-2, local idx = (i-1) - i_min(d-2)
        m_k = (i - 1) - _i_min(d - 2)
        m_k = jnp.clip(m_k, 0, D_max - 1)
        m_pred = prev_prev[m_k]
        m_val = jax.nn.logsumexp(m_pred[:, None] + log_trans, axis=0) + emit[i, j]

        # I: (i, j-1) on d-1, local idx = i - i_min(d-1)
        i_k = i - _i_min(d - 1)
        i_k = jnp.clip(i_k, 0, D_max - 1)
        i_pred = prev[i_k]
        i_val = jax.nn.logsumexp(i_pred[:, None] + log_trans, axis=0) + emit[i, j]

        # D: (i-1, j) on d-1, local idx = (i-1) - i_min(d-1)
        d_k = (i - 1) - _i_min(d - 1)
        d_k = jnp.clip(d_k, 0, D_max - 1)
        d_pred = prev[d_k]
        d_val = jax.nn.logsumexp(d_pred[:, None] + log_trans, axis=0) + emit[i, j]

        # Boundary masks
        m_val = jnp.where((i >= 1) & (j >= 1), m_val, NEG_INF)
        i_val = jnp.where(j >= 1, i_val, NEG_INF)
        d_val = jnp.where(i >= 1, d_val, NEG_INF)
        return jnp.where(is_M, m_val,
                jnp.where(is_I, i_val,
                jnp.where(is_D, d_val, NEG_INF)))

    def scan_fn(carry, d):
        prev, prev_prev = carry  # (D_max, ns) each
        ks = jnp.arange(D_max)
        i_vals = _i_min(d) + ks
        j_vals = d - i_vals
        valid = (i_vals <= Lx) & (j_vals >= 0) & (j_vals <= Ly)
        cells = jax.vmap(lambda k: compute_cell(prev, prev_prev, d, k))(ks)
        cells = jnp.where(valid[:, None], cells, NEG_INF)
        return (cells, prev), cells

    (_, _), all_diags = jax.lax.scan(
        scan_fn, (diag0, jnp.full((D_max, ns), NEG_INF)),
        jnp.arange(1, n_diags + 1))
    # all_diags: (n_diags, D_max, ns) — diags 1..Lx+Ly

    # Scatter diagonals into full (Lx+1, Ly+1, ns) grid via fori_loop.
    # One diagonal per iteration keeps each scatter small (≤D_max entries).
    F = jnp.full((Lx + 1, Ly + 1, ns), NEG_INF)
    F = F.at[0, 0, S].set(0.0)

    # Scatter via flat indexing with fori_loop: one diagonal per step.
    # Use (n_flat+1) buffer with dummy overflow slot to avoid collisions.
    n_flat = (Lx + 1) * (Ly + 1)
    F_flat = jnp.concatenate([F.reshape(n_flat, ns),
                               jnp.full((1, ns), NEG_INF)], axis=0)

    def _scatter_one_diag(idx, F_flat):
        d = idx + 1
        diag_data = all_diags[idx]
        i_min_d = jnp.maximum(0, d - Ly)
        ks = jnp.arange(D_max)
        i_vals = i_min_d + ks
        j_vals = d - i_vals
        valid = (i_vals <= Lx) & (j_vals >= 0) & (j_vals <= Ly)
        lin = i_vals * (Ly + 1) + j_vals
        lin_safe = jnp.where(valid, lin, n_flat)  # invalid → dummy
        return F_flat.at[lin_safe].set(diag_data)

    F_flat = jax.lax.fori_loop(0, n_diags, _scatter_one_diag, F_flat)
    F = F_flat[:n_flat].reshape(Lx + 1, Ly + 1, ns)

    e_idx = _find_e_idx(state_types)
    log_prob = jax.nn.logsumexp(F[Lx, Ly, :] + log_trans[:, e_idx])
    return log_prob, F


def _backward_2d_core_diag(log_trans, state_types, emit, Lx, Ly,
                            real_Lx=None, real_Ly=None):
    """Anti-diagonal wavefront backward for 2D pair HMM.

    Mirror of _forward_2d_core_diag — scans diagonals from high to low.
    Successor of cell (i, j, k) is at (i+1, j+1) for M-type k, (i, j+1)
    for I-type k, (i+1, j) for D-type k. So cells on diagonal d depend
    only on cells on diagonals d+1 (I, D) and d+2 (M).

    Carry is only two diagonals (d+1 and d+2), not the full grid.

    real_Lx/real_Ly (optional traced jnp scalars): when provided, the
    terminal condition B[real_Lx, real_Ly, k] = log_trans[k, E_idx] is
    placed at the real endpoint instead of the padded corner (Lx, Ly).
    Used under vmap when sequences are pre-padded to Lx_pad, Ly_pad.
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    e_idx = _find_e_idx(state_types)
    term_Lx = Lx if real_Lx is None else real_Lx
    term_Ly = Ly if real_Ly is None else real_Ly
    D_max = min(Lx, Ly) + 1
    n_diags = Lx + Ly  # diagonals 0..Lx+Ly-1

    def _i_min(d):
        return jnp.maximum(0, d - Ly)

    # Terminal diagonal: d_term = term_Lx + term_Ly
    # Place B[term_Lx, term_Ly] = log_trans[:, E_idx] at the right local position.
    term_d = term_Lx + term_Ly
    term_diag = jnp.full((D_max, ns), NEG_INF)
    term_k = term_Lx - _i_min(term_d)
    term_diag = term_diag.at[term_k].set(log_trans[:, e_idx])

    def compute_cell(next_diag, next_next_diag, d, k):
        """Compute backward cell k on diagonal d from next (d+1) and next_next (d+2)."""
        i = _i_min(d) + k
        j = d - i

        # M successor at (i+1, j+1) on d+2, local idx = (i+1) - i_min(d+2)
        m_k = (i + 1) - _i_min(d + 2)
        m_k = jnp.clip(m_k, 0, D_max - 1)
        succ_M = emit[jnp.clip(i+1, 0, Lx), jnp.clip(j+1, 0, Ly)] + next_next_diag[m_k]
        contrib_M = log_trans + succ_M[None, :]
        contrib_M = jnp.where(is_M[None, :], contrib_M, NEG_INF)
        m_val = jax.nn.logsumexp(contrib_M, axis=1)

        # I successor at (i, j+1) on d+1, local idx = i - i_min(d+1)
        i_k = i - _i_min(d + 1)
        i_k = jnp.clip(i_k, 0, D_max - 1)
        succ_I = emit[jnp.clip(i, 0, Lx), jnp.clip(j+1, 0, Ly)] + next_diag[i_k]
        contrib_I = log_trans + succ_I[None, :]
        contrib_I = jnp.where(is_I[None, :], contrib_I, NEG_INF)
        i_val = jax.nn.logsumexp(contrib_I, axis=1)

        # D successor at (i+1, j) on d+1, local idx = (i+1) - i_min(d+1)
        d_k = (i + 1) - _i_min(d + 1)
        d_k = jnp.clip(d_k, 0, D_max - 1)
        succ_D = emit[jnp.clip(i+1, 0, Lx), jnp.clip(j, 0, Ly)] + next_diag[d_k]
        contrib_D = log_trans + succ_D[None, :]
        contrib_D = jnp.where(is_D[None, :], contrib_D, NEG_INF)
        d_val = jax.nn.logsumexp(contrib_D, axis=1)

        # Boundary masks
        m_val = jnp.where((i < Lx) & (j < Ly), m_val, NEG_INF)
        i_val = jnp.where(j < Ly, i_val, NEG_INF)
        d_val = jnp.where(i < Lx, d_val, NEG_INF)

        return jnp.logaddexp(m_val, jnp.logaddexp(i_val, d_val))

    def scan_fn(carry, d):
        next_diag, next_next_diag = carry
        ks = jnp.arange(D_max)
        i_vals = _i_min(d) + ks
        j_vals = d - i_vals
        valid = ((i_vals <= Lx) & (j_vals >= 0) & (j_vals <= Ly)
                 & ~((i_vals == term_Lx) & (j_vals == term_Ly)))
        cells = jax.vmap(lambda k: compute_cell(next_diag, next_next_diag, d, k))(ks)
        # Keep terminal cell value from init; mask invalid cells
        is_term = (i_vals == term_Lx) & (j_vals == term_Ly)
        term_vals = jnp.where(is_term[:, None],
                              jnp.broadcast_to(log_trans[:, e_idx], (D_max, ns)),
                              NEG_INF)
        cells = jnp.where(valid[:, None], cells, term_vals)
        return (cells, next_diag), cells

    # Highest diagonal is Lx+Ly. We need to initialize properly:
    # The terminal diagonal term_d has the terminal condition.
    # Diagonals above term_d are all NEG_INF.
    # We scan from d = Lx+Ly-1 down to 0 (n_diags = Lx+Ly steps).
    # But we need to handle term_d correctly. The scan needs the
    # two diagonals above the current one. For d = Lx+Ly-1, we need
    # diag at Lx+Ly (= term_d when real_Lx=Lx, real_Ly=Ly) and Lx+Ly+1 (empty).

    # Build initial "next" and "next_next" for d = Lx+Ly-1:
    # next = diagonal Lx+Ly (contains terminal if term_d == Lx+Ly)
    # next_next = diagonal Lx+Ly+1 (always empty)
    diag_top = jnp.full((D_max, ns), NEG_INF)
    # If term_d == Lx+Ly, place terminal there
    top_k = term_Lx - _i_min(jnp.int32(Lx + Ly))
    top_k = jnp.clip(top_k, 0, D_max - 1)
    is_top_term = (term_d == Lx + Ly)
    diag_top = jnp.where(
        is_top_term,
        diag_top.at[top_k].set(log_trans[:, e_idx]),
        diag_top)
    empty = jnp.full((D_max, ns), NEG_INF)

    # Scan from d = Lx+Ly-1 down to 0
    (_, _), all_diags = jax.lax.scan(
        scan_fn, (diag_top, empty),
        jnp.arange(n_diags)[::-1])
    # all_diags: (n_diags, D_max, ns) for diags Lx+Ly-1 down to 0

    # Scatter into full grid via fori_loop (one diagonal per step).
    B = jnp.full((Lx + 1, Ly + 1, ns), NEG_INF)
    B = B.at[term_Lx, term_Ly].set(log_trans[:, e_idx])

    # all_diags[idx] corresponds to diagonal d = (Lx+Ly-1) - idx
    n_flat = (Lx + 1) * (Ly + 1)
    B_flat = jnp.concatenate([B.reshape(n_flat, ns),
                               jnp.full((1, ns), NEG_INF)], axis=0)

    def _scatter_one_diag(idx, B_flat):
        d = (Lx + Ly - 1) - idx
        diag_data = all_diags[idx]
        i_min_d = jnp.maximum(0, d - Ly)
        ks = jnp.arange(D_max)
        i_vals = i_min_d + ks
        j_vals = d - i_vals
        valid = (i_vals <= Lx) & (j_vals >= 0) & (j_vals <= Ly)
        lin = i_vals * (Ly + 1) + j_vals
        lin_safe = jnp.where(valid, lin, n_flat)
        return B_flat.at[lin_safe].set(diag_data)

    B_flat = jax.lax.fori_loop(0, n_diags, _scatter_one_diag, B_flat)
    B = B_flat[:n_flat].reshape(Lx + 1, Ly + 1, ns)
    B = B.at[term_Lx, term_Ly].set(log_trans[:, e_idx])

    return B


def _forward_2d_core_rowscan(log_trans, state_types, emit, Lx, Ly):
    """Row-scan forward for 2D pair HMM.

    Memory-efficient alternative to _forward_2d_core_diag for large state
    counts (e.g. 502-state MixDom). Uses O(Ly * ns) carry instead of
    O(D_max * ns) diagonal carry, but avoids the vmap-over-D_max logsumexp
    that causes XLA compilation blowup.

    Outer jax.lax.scan over rows (i=0..Lx), inner jax.lax.scan over
    columns (j) for I-type sequential dependency.

    Lx and Ly must be Python ints (padded bin sizes).
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    # trans_to_M[src] = log_trans[src, k] for M-type k, else NEG_INF
    # We need: for each destination state k, the contribution from each type.
    # log_trans is (ns, ns): log_trans[src, dst]
    # For M-type destinations: trans_M[:, k] = log_trans[:, k] if is_M[k] else NEG_INF
    trans_M = jnp.where(is_M[None, :], log_trans, NEG_INF)  # (ns, ns)
    trans_I = jnp.where(is_I[None, :], log_trans, NEG_INF)  # (ns, ns)
    trans_D = jnp.where(is_D[None, :], log_trans, NEG_INF)  # (ns, ns)

    # Row 0: F[0, 0, S] = 0, then I-type transitions along j
    row0 = jnp.full((Ly + 1, ns), NEG_INF)
    row0 = row0.at[0, S].set(0.0)

    # Fill row 0 columns 1..Ly via I-type scan
    def _col_scan_fwd(prev_cell, j):
        """Propagate I-type transitions from column j-1 to j."""
        # I contribution: prev_cell @ trans_I + emit[i, j]
        # prev_cell is (ns,), trans_I is (ns, ns)
        i_contrib = jax.nn.logsumexp(prev_cell[:, None] + trans_I, axis=0) + emit[0, j]
        # For non-I states, this should be NEG_INF (already handled by trans_I mask)
        return i_contrib, i_contrib

    # For row 0, there are no M or D contributions (no previous row), only I
    _, row0_cols = jax.lax.scan(_col_scan_fwd, row0[0], jnp.arange(1, Ly + 1))
    row0 = row0.at[1:].set(row0_cols)

    def _process_row(prev_row, i):
        """Compute row i given row i-1 (prev_row).

        For each column j:
          M-contrib[j, k] = logsumexp_src(prev_row[j-1, src] + trans_M[src, k]) + emit[i, j, k]
          D-contrib[j, k] = logsumexp_src(prev_row[j, src] + trans_D[src, k]) + emit[i, j, k]
        These are "free inputs" from the previous row.
        I-contrib propagates sequentially along j.
        """
        # Precompute M contributions for columns 1..Ly (from prev_row[0..Ly-1])
        # m_free[j-1] for j=1..Ly: prev_row[j-1] @ trans_M + emit[i, j]
        def _m_contrib(j_minus_1):
            j = j_minus_1 + 1
            return jax.nn.logsumexp(
                prev_row[j_minus_1][:, None] + trans_M, axis=0) + emit[i, j]
        m_free = jax.vmap(_m_contrib)(jnp.arange(Ly))  # (Ly, ns)

        # Precompute D contributions for columns 0..Ly (from prev_row[0..Ly])
        # d_free[j] for j=0..Ly: prev_row[j] @ trans_D + emit[i, j]
        def _d_contrib(j):
            return jax.nn.logsumexp(
                prev_row[j][:, None] + trans_D, axis=0) + emit[i, j]
        d_free = jax.vmap(_d_contrib)(jnp.arange(Ly + 1))  # (Ly+1, ns)

        # Column 0: only D contribution (no M or I from left)
        cell0 = d_free[0]

        # Columns 1..Ly: M+D free input, then propagate I along j
        def _col_scan_row(prev_cell, idx):
            """Process column j = idx+1."""
            j = idx + 1
            # M+D free input
            free = jnp.logaddexp(m_free[idx], d_free[j])
            # I contribution from prev_cell
            i_contrib = jax.nn.logsumexp(
                prev_cell[:, None] + trans_I, axis=0) + emit[i, j]
            cell = jnp.logaddexp(free, i_contrib)
            return cell, cell

        _, row_cols = jax.lax.scan(_col_scan_row, cell0, jnp.arange(Ly))
        new_row = jnp.concatenate([cell0[None, :], row_cols], axis=0)  # (Ly+1, ns)
        return new_row, new_row

    # Scan over rows 1..Lx
    _, all_rows = jax.lax.scan(_process_row, row0, jnp.arange(1, Lx + 1))
    # all_rows: (Lx, Ly+1, ns) for rows 1..Lx
    F = jnp.concatenate([row0[None, :, :], all_rows], axis=0)  # (Lx+1, Ly+1, ns)

    e_idx = _find_e_idx(state_types)
    log_prob = jax.nn.logsumexp(F[Lx, Ly, :] + log_trans[:, e_idx])
    return log_prob, F


def _backward_2d_core_rowscan(log_trans, state_types, emit, Lx, Ly,
                                real_Lx=None, real_Ly=None):
    """Row-scan backward for 2D pair HMM.

    Memory-efficient alternative to _backward_2d_core_diag for large state
    counts. Scans rows from Lx down to 0.

    For cell (i, j, k):
      B[i,j,k] = logsumexp over successor states k' of:
        M-type k': trans[k,k'] + emit[i+1,j+1,k'] + B[i+1,j+1,k']
        D-type k': trans[k,k'] + emit[i+1,j,k'] + B[i+1,j,k']
        I-type k': trans[k,k'] + emit[i,j+1,k'] + B[i,j+1,k']

    M and D successors are on the next row (known). I successors are on the
    same row at j+1 (sequential dependency, scan j from Ly-1 down to 0).

    real_Lx/real_Ly: optional traced jnp scalars for terminal placement.
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    e_idx = _find_e_idx(state_types)
    term_Lx = Lx if real_Lx is None else real_Lx
    term_Ly = Ly if real_Ly is None else real_Ly

    # Masks for successor types applied to log_trans
    # For M-type successors: trans[k, k'] where k' is M-type
    trans_M = jnp.where(is_M[None, :], log_trans, NEG_INF)  # (ns, ns)
    trans_I = jnp.where(is_I[None, :], log_trans, NEG_INF)  # (ns, ns)
    trans_D = jnp.where(is_D[None, :], log_trans, NEG_INF)  # (ns, ns)

    # Terminal row: B[term_Lx, term_Ly, k] = log_trans[k, e_idx]
    # Initialize last row to NEG_INF, place terminal condition
    last_row = jnp.full((Ly + 1, ns), NEG_INF)
    last_row = last_row.at[term_Ly].set(log_trans[:, e_idx])

    # For the terminal row (row = term_Lx = Lx in unpadded case),
    # we need to fill backward values for j < term_Ly using I-type successors.
    # B[term_Lx, j, k] = logsumexp_k'(trans[k,k'] + emit[term_Lx,j+1,k'] + B[term_Lx,j+1,k'])
    # for I-type k' only (no M or D successors since we're at the last row).

    # Scan j from term_Ly-1 down to 0 on the terminal row
    def _col_scan_bwd_terminal(next_cell, j):
        """Backward I-type propagation on the terminal row."""
        # I successor at (i, j+1): trans[k, k'] + emit[i, j+1, k'] + B[i, j+1, k']
        succ_I = emit[term_Lx, j + 1] + next_cell  # (ns,)
        i_contrib = jax.nn.logsumexp(trans_I + succ_I[None, :], axis=1)  # (ns,)
        # Only valid if j+1 <= term_Ly (which it is since we scan from term_Ly-1)
        # But we need to mask: this row is at i=term_Lx, only I transitions valid
        return i_contrib, i_contrib

    # We need to handle variable term_Ly (traced). Use full scan over Ly columns
    # and mask.

    # Actually, for the terminal row, we need to propagate backward from term_Ly.
    # For j >= term_Ly: B = NEG_INF (except j = term_Ly which is the terminal)
    # For j < term_Ly: only I-type successors on same row
    # Also at i = Lx (padded), for i < real_Lx there could be D successors too.
    # But at the terminal row (i = term_Lx = Lx in padded case), only I is valid.

    # Fill terminal row: scan from Ly-1 down to 0
    def _fill_terminal_col(next_cell, rev_j):
        """Fill terminal row column by column from right to left."""
        j = rev_j
        succ_I = emit[term_Lx, j + 1] + next_cell  # (ns,)
        i_contrib = jax.nn.logsumexp(trans_I + succ_I[None, :], axis=1)  # (ns,)
        # Only valid if j < term_Ly (j+1 <= term_Ly)
        cell = jnp.where(j < term_Ly, i_contrib, NEG_INF)
        # Keep terminal value at j == term_Ly
        cell = jnp.where(j == term_Ly, log_trans[:, e_idx], cell)
        return cell, cell

    if Ly > 0:
        _, term_row_cols = jax.lax.scan(
            _fill_terminal_col, last_row[term_Ly],
            jnp.arange(Ly)[::-1])  # scan from Ly-1 down to 0
        # term_row_cols is (Ly, ns) in reverse order (Ly-1, Ly-2, ..., 0)
        # Reverse to get (0, 1, ..., Ly-1)
        term_row_cols = term_row_cols[::-1]
        last_row = jnp.concatenate([term_row_cols, last_row[Ly:Ly+1]], axis=0)  # (Ly+1, ns)
    # else: Ly=0, last_row already has terminal at j=0

    def _process_row_bwd(next_row, i):
        """Compute backward row i given row i+1 (next_row).

        For cell (i, j, k):
          M-contrib: logsumexp_k'(trans_M[k,k'] + emit[i+1,j+1,k'] + next_row[j+1,k'])
          D-contrib: logsumexp_k'(trans_D[k,k'] + emit[i+1,j,k'] + next_row[j,k'])
        These are free inputs from the next row.
          I-contrib: logsumexp_k'(trans_I[k,k'] + emit[i,j+1,k'] + B[i,j+1,k'])
        Sequential dependency scanning j from right to left.
        """
        # Precompute M contributions for j=0..Ly-1 (successor at (i+1, j+1))
        def _m_contrib(j):
            succ = emit[i + 1, j + 1] + next_row[j + 1]  # (ns,)
            return jax.nn.logsumexp(trans_M + succ[None, :], axis=1)  # (ns,)
        m_free = jax.vmap(_m_contrib)(jnp.arange(Ly)) if Ly > 0 else jnp.empty((0, ns))  # (Ly, ns)

        # Precompute D contributions for j=0..Ly (successor at (i+1, j))
        def _d_contrib(j):
            succ = emit[i + 1, j] + next_row[j]  # (ns,)
            return jax.nn.logsumexp(trans_D + succ[None, :], axis=1)  # (ns,)
        d_free = jax.vmap(_d_contrib)(jnp.arange(Ly + 1))  # (Ly+1, ns)

        # Column Ly: only D contribution (no M successor at j+1=Ly+1, no I to right)
        cell_Ly = d_free[Ly]

        # Columns Ly-1 down to 0: M+D free input + I propagation
        def _col_scan_bwd(next_cell, rev_idx):
            """Process column j = rev_idx (scanning right to left)."""
            j = rev_idx
            # M+D free input
            free = jnp.logaddexp(m_free[j], d_free[j])
            # I contribution from next_cell (which is B[i, j+1])
            succ_I = emit[i, j + 1] + next_cell  # (ns,)
            i_contrib = jax.nn.logsumexp(trans_I + succ_I[None, :], axis=1)  # (ns,)
            cell = jnp.logaddexp(free, i_contrib)
            return cell, cell

        if Ly > 0:
            _, row_cols = jax.lax.scan(
                _col_scan_bwd, cell_Ly,
                jnp.arange(Ly)[::-1])  # scan from Ly-1 down to 0
            row_cols = row_cols[::-1]  # (Ly, ns) now in order 0..Ly-1
            new_row = jnp.concatenate([row_cols, cell_Ly[None, :]], axis=0)
        else:
            new_row = cell_Ly[None, :]
        return new_row, new_row

    # Wrap _process_row_bwd to handle real_Lx < Lx: rows beyond term_Lx
    # are all NEG_INF, and row term_Lx is the terminal row (last_row).
    def _process_row_bwd_masked(next_row, i):
        # For i > term_Lx: beyond real sequence → NEG_INF, carry = last_row
        # For i == term_Lx: this IS the terminal row → output last_row
        # For i < term_Lx: normal backward computation
        computed_row, _ = _process_row_bwd(next_row, i)
        is_below = (i < term_Lx)
        is_at = (i == term_Lx)
        out_row = jnp.where(is_below, computed_row,
                  jnp.where(is_at, last_row, jnp.full_like(computed_row, NEG_INF)))
        carry = jnp.where(is_below, computed_row,
                 jnp.where(is_at, last_row, last_row))
        return carry, out_row

    # Scan over rows Lx-1 down to 0
    if Lx > 0:
        _, all_rows = jax.lax.scan(
            _process_row_bwd_masked, last_row,
            jnp.arange(Lx)[::-1])  # scan from Lx-1 down to 0
        all_rows = all_rows[::-1]  # (Lx, Ly+1, ns) now in order 0..Lx-1
        B = jnp.concatenate([all_rows, last_row[None, :, :]], axis=0)
    else:
        B = last_row[None, :, :]

    return B


def _backward_2d_core(log_trans, state_types, emit, Lx, Ly):
    """Core backward algorithm for a Pair HMM.

    At cell (i,j), state k:
      β(i,j,k) = logsumexp over successor states k' of
                  [trans(k,k') + emit(succ,k') + β(succ,k')]
    where successor position depends on state type of k':
      M-type: (i+1, j+1),  I-type: (i, j+1),  D-type: (i+1, j)

    Returns:
        B: (Lx+1, Ly+1, ns) backward table
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    # Terminal: β(Lx, Ly, k) = log_trans[k, E_idx]
    e_idx = _find_e_idx(state_types)
    beta_final = log_trans[:, e_idx]

    # Last row (i=Lx): only I-type successors valid
    last_row = jnp.full((Ly + 1, ns), NEG_INF)
    last_row = last_row.at[Ly].set(beta_final)

    def last_row_step(beta_next, j):
        # Only I-type states at (Lx, j+1) are reachable
        # β(Lx,j,k) = Σ_{k' is I-type} trans[k,k'] * emit(Lx,j+1,k') * β(Lx,j+1,k')
        succ_emit = emit[Lx, j + 1]  # (ns,)
        contrib = log_trans + succ_emit[None, :] + beta_next[None, :]  # (ns, ns)
        # Zero out non-I successor contributions
        contrib = jnp.where(is_I[None, :], contrib, NEG_INF)
        cell = jax.nn.logsumexp(contrib, axis=1)  # (ns,)
        return cell, cell

    _, last_row_rest = jax.lax.scan(last_row_step, last_row[Ly],
                                     jnp.arange(Ly - 1, -1, -1))
    last_row = last_row.at[:Ly].set(last_row_rest[::-1])

    # Rows Lx-1 down to 0
    def row_step_bwd(next_row, i):
        # Column Ly: only D-type successors at (i+1, Ly) are valid
        succ_emit_D = emit[i + 1, Ly]
        contrib_D = log_trans + succ_emit_D[None, :] + next_row[Ly][None, :]
        contrib_D = jnp.where(is_D[None, :], contrib_D, NEG_INF)
        cell_Ly = jax.nn.logsumexp(contrib_D, axis=1)

        def col_step_bwd(beta_right, j):
            # M-type successor at (i+1, j+1)
            succ_emit_M = emit[i + 1, j + 1]
            contrib_M = log_trans + succ_emit_M[None, :] + next_row[j + 1][None, :]
            contrib_M = jnp.where(is_M[None, :], contrib_M, NEG_INF)

            # I-type successor at (i, j+1)
            succ_emit_I = emit[i, j + 1]
            contrib_I = log_trans + succ_emit_I[None, :] + beta_right[None, :]
            contrib_I = jnp.where(is_I[None, :], contrib_I, NEG_INF)

            # D-type successor at (i+1, j)
            succ_emit_D = emit[i + 1, j]
            contrib_D = log_trans + succ_emit_D[None, :] + next_row[j][None, :]
            contrib_D = jnp.where(is_D[None, :], contrib_D, NEG_INF)

            # Sum all successor contributions
            all_contrib = jnp.logaddexp(contrib_M, jnp.logaddexp(contrib_I, contrib_D))
            cell = jax.nn.logsumexp(all_contrib, axis=1)
            return cell, cell

        _, row_rest = jax.lax.scan(col_step_bwd, cell_Ly,
                                    jnp.arange(Ly - 1, -1, -1))
        curr_row = jnp.concatenate([row_rest[::-1], cell_Ly[None]], axis=0)
        return curr_row, curr_row

    _, all_rows_bwd = jax.lax.scan(row_step_bwd, last_row,
                                    jnp.arange(Lx - 1, -1, -1))

    B = jnp.concatenate([all_rows_bwd[::-1], last_row[None]], axis=0)
    return B


@jax.jit
def forward_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi):
    """Forward algorithm for a Pair HMM.

    Convenience wrapper that computes emissions then runs the forward DP.
    Pads inputs to geometric bins for JIT cache reuse.

    Returns:
        log_prob: total log probability P(x, y)
        F: (Lx+1, Ly+1, n_states) forward table
    """
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)
    x_pad = _pad_seq(x_seq, Lx_pad)
    y_pad = _pad_seq(y_seq, Ly_pad)
    emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi)
    # Mask padded emissions to NEG_INF
    mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)
    log_prob_pad, F_pad = _forward_2d_core_diag(log_trans, state_types, emit, Lx_pad, Ly_pad)
    # Extract result at real lengths
    e_idx = _find_e_idx(state_types)
    log_prob = jax.nn.logsumexp(F_pad[Lx, Ly, :] + log_trans[:, e_idx])
    F = F_pad[:Lx + 1, :Ly + 1, :]
    return log_prob, F


def backward_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi):
    """Backward algorithm for a Pair HMM.

    Convenience wrapper that computes emissions then runs the backward DP.
    Pads inputs to geometric bins for JIT cache reuse.

    Returns:
        B: (Lx+1, Ly+1, n_states) backward table
    """
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)
    x_pad = _pad_seq(x_seq, Lx_pad)
    y_pad = _pad_seq(y_seq, Ly_pad)
    emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi)
    mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)
    # Use real dimensions so terminal condition is at (Lx, Ly)
    B = _backward_2d_core(log_trans, state_types, emit, Lx, Ly)
    return B


def _emit_mask(Lx, Ly, Lx_pad, Ly_pad, ns):
    """Boolean mask: True for real positions, False for padding."""
    i_ok = jnp.arange(Lx_pad + 1) <= Lx  # (Lx_pad+1,)
    j_ok = jnp.arange(Ly_pad + 1) <= Ly  # (Ly_pad+1,)
    return (i_ok[:, None, None] & j_ok[None, :, None]) * jnp.ones(ns, dtype=bool)[None, None, :]


@partial(jax.jit, static_argnames=('forward_only',))
def forward_backward_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi,
                        log_emit_table=None, real_Lx=None, real_Ly=None,
                        forward_only=False):
    """Forward-Backward for a Pair HMM.

    Pads inputs to geometric bins for JIT cache reuse.

    Args:
        log_trans: (n_states, n_states) log transition matrix
        state_types: (n_states,) state types (S/M/I/D/E)
        x_seq: (Lx,) ancestor sequence (integer indices)
        y_seq: (Ly,) descendant sequence (integer indices)
        sub_matrix: (A, A) substitution matrix P(t)
        pi: (A,) equilibrium frequencies
        log_emit_table: optional (Lx+1, Ly+1, n_states) pre-computed log emission
            table. When provided, x_seq/y_seq/sub_matrix/pi are ignored for
            emission computation but x_seq/y_seq shapes are still used for padding.
            This enables rate-multiplier-annotated or soft-observation models.
        real_Lx, real_Ly: optional traced jnp scalars giving the real (unpadded)
            sequence lengths. When provided, the caller is passing sequences
            that are ALREADY padded to the JIT bin sizes (e.g. under vmap over
            a batch of pairs); the real lengths are used (a) to mask the
            computed/provided emission table to NEG_INF outside the real
            region and (b) for endpoint extraction of log_prob at
            F[real_Lx, real_Ly, :]. When None, falls back to the scalar
            assumption that x_seq/y_seq are of real length.

    Returns:
        log_prob: total log probability
        posteriors: (Lx_pad+1, Ly_pad+1, n_states) posterior state probabilities
            (when real_Lx/real_Ly given, posteriors are 0 outside the real
            region; caller trims to real shape if needed)
        expected_trans: (n_states, n_states) expected transition counts
    """
    Lx_arr = x_seq.shape[0]
    Ly_arr = y_seq.shape[0]
    Lx_pad = _pad_to_bin(Lx_arr)
    Ly_pad = _pad_to_bin(Ly_arr)
    # Lx/Ly are the real lengths used for masking and endpoint extraction.
    # When real_Lx/real_Ly are None (scalar path), they equal the array shape.
    Lx = Lx_arr if real_Lx is None else real_Lx
    Ly = Ly_arr if real_Ly is None else real_Ly

    if log_emit_table is not None:
        # Pre-computed emission table. Its shape is (Lx_arr+1, Ly_arr+1, ns);
        # we place it into a (Lx_pad+1, Ly_pad+1) NEG_INF-filled tensor (no-op
        # when Lx_arr == Lx_pad, as in the vmap case). Then, if real lengths
        # were given, mask the region outside (real_Lx, real_Ly) to NEG_INF.
        ns = state_types.shape[0]
        emit = jnp.full((Lx_pad + 1, Ly_pad + 1, ns), NEG_INF)
        emit = emit.at[:Lx_arr + 1, :Ly_arr + 1, :].set(log_emit_table)
        if real_Lx is not None or real_Ly is not None:
            mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, state_types.shape[0])
            emit = jnp.where(mask, emit, NEG_INF)
    else:
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
        emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi)
        mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, state_types.shape[0])
        emit = jnp.where(mask, emit, NEG_INF)
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    # Choose between diagonal wavefront and row-scan based on state count.
    # Diagonal wavefront vmaps logsumexp over (ns, ns) across D_max cells,
    # which causes XLA compilation blowup for large ns (e.g. 502-state MixDom).
    # Row-scan uses sequential inner scan, avoiding the vmap blowup.
    _use_rowscan = ns > 50

    if _use_rowscan:
        _, F_pad = _forward_2d_core_rowscan(log_trans, state_types, emit, Lx_pad, Ly_pad)
    else:
        _, F_pad = _forward_2d_core_diag(log_trans, state_types, emit, Lx_pad, Ly_pad)

    # Extract forward at real lengths (dynamic index — works under vmap).
    e_idx = _find_e_idx(state_types)
    log_prob = jax.nn.logsumexp(F_pad[Lx, Ly, :] + log_trans[:, e_idx])

    if forward_only:
        return log_prob

    # Backward terminal condition: place it at (Lx, Ly) (real lengths). When
    # real_Lx/real_Ly are None (scalar path), Lx == Lx_arr == Lx_pad-at-
    # most-equal; when batched, Lx/Ly are traced scalars within
    # [0, Lx_pad]/[0, Ly_pad].
    if _use_rowscan:
        B = _backward_2d_core_rowscan(log_trans, state_types, emit, Lx_pad, Ly_pad,
                                       real_Lx=Lx, real_Ly=Ly)
    else:
        B = _backward_2d_core_diag(log_trans, state_types, emit, Lx_pad, Ly_pad,
                                    real_Lx=Lx, real_Ly=Ly)
    F = F_pad

    posteriors = jnp.exp(F + B - log_prob)

    # Expected transition counts via vectorized computation over the full
    # padded range. Positions outside the real (Lx, Ly) region have F=B=
    # NEG_INF (by masking + terminal placement), so exp(F+B+... - log_prob)
    # is 0 there and naturally excluded from the sum. No static slicing
    # required — safe under vmap with traced Lx/Ly.
    #
    # n[k, k'] = Σ_{(i,j)} exp(F[i,j,k] + trans[k,k'] + emit(succ,k') + B(succ,k') - log_prob)

    expected_trans = jnp.zeros((ns, ns))

    # M-type destinations: source (i,j), dest (i+1,j+1)
    F_src_M = F[:-1, :-1, :]                     # (Lx_pad, Ly_pad, ns)
    emit_dest_M = emit[1:, 1:, :]
    B_dest_M = B[1:, 1:, :]
    log_n_M = (F_src_M[:, :, :, None] + log_trans[None, None, :, :] +
               emit_dest_M[:, :, None, :] + B_dest_M[:, :, None, :] - log_prob)
    log_n_M = jnp.where(is_M[None, None, None, :], log_n_M, NEG_INF)
    expected_trans = expected_trans + jnp.exp(log_n_M).sum(axis=(0, 1))

    # I-type destinations: source (i,j), dest (i, j+1)
    F_src_I = F[:, :-1, :]
    emit_dest_I = emit[:, 1:, :]
    B_dest_I = B[:, 1:, :]
    log_n_I = (F_src_I[:, :, :, None] + log_trans[None, None, :, :] +
               emit_dest_I[:, :, None, :] + B_dest_I[:, :, None, :] - log_prob)
    log_n_I = jnp.where(is_I[None, None, None, :], log_n_I, NEG_INF)
    expected_trans = expected_trans + jnp.exp(log_n_I).sum(axis=(0, 1))

    # D-type destinations: source (i,j), dest (i+1, j)
    F_src_D = F[:-1, :, :]
    emit_dest_D = emit[1:, :, :]
    B_dest_D = B[1:, :, :]
    log_n_D = (F_src_D[:, :, :, None] + log_trans[None, None, :, :] +
               emit_dest_D[:, :, None, :] + B_dest_D[:, :, None, :] - log_prob)
    log_n_D = jnp.where(is_D[None, None, None, :], log_n_D, NEG_INF)
    expected_trans = expected_trans + jnp.exp(log_n_D).sum(axis=(0, 1))

    # E transitions: at (Lx, Ly) where Lx, Ly are the real lengths.
    log_count_E = F[Lx, Ly, :] + log_trans[:, e_idx] - log_prob
    expected_trans = expected_trans.at[:, e_idx].add(jnp.exp(log_count_E))

    return log_prob, posteriors, expected_trans


def viterbi_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi):
    """Viterbi algorithm for a Pair HMM.

    Returns:
        log_prob: log probability of best path
        traceback: list of (i, j, state) tuples
    """
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]
    n_states = log_trans.shape[0]
    log_sub = jnp.log(sub_matrix + 1e-30)
    log_pi = jnp.log(pi + 1e-30)

    V = jnp.full((Lx + 1, Ly + 1, n_states), NEG_INF)
    V = V.at[0, 0, S].set(0.0)
    # Traceback pointers stored as (prev_i, prev_j, prev_state)
    TB = jnp.zeros((Lx + 1, Ly + 1, n_states, 3), dtype=jnp.int32)

    for d in range(1, Lx + Ly + 1):
        i_min = max(0, d - Ly)
        i_max = min(d, Lx)
        for i in range(i_min, i_max + 1):
            j = d - i
            if i == 0 and j == 0:
                continue

            log_emit = jax.vmap(
                lambda st: _pair_hmm_emission(st, x_seq[max(i - 1, 0)], y_seq[max(j - 1, 0)], log_sub, log_pi)
            )(state_types)

            for k in range(n_states):
                st = state_types[k]
                if st == M and i > 0 and j > 0:
                    scores = V[i - 1, j - 1, :] + log_trans[:, k]
                    best = jnp.argmax(scores)
                    V = V.at[i, j, k].set(scores[best] + log_emit[k])
                    TB = TB.at[i, j, k].set(jnp.array([i - 1, j - 1, best], dtype=jnp.int32))
                elif st == I and j > 0:
                    scores = V[i, j - 1, :] + log_trans[:, k]
                    best = jnp.argmax(scores)
                    V = V.at[i, j, k].set(scores[best] + log_emit[k])
                    TB = TB.at[i, j, k].set(jnp.array([i, j - 1, best], dtype=jnp.int32))
                elif st == D and i > 0:
                    scores = V[i - 1, j, :] + log_trans[:, k]
                    best = jnp.argmax(scores)
                    V = V.at[i, j, k].set(scores[best] + log_emit[k])
                    TB = TB.at[i, j, k].set(jnp.array([i - 1, j, best], dtype=jnp.int32))

    # Terminal
    e_idx = _find_e_idx(state_types)
    final_scores = V[Lx, Ly, :] + log_trans[:, e_idx]
    best_final = jnp.argmax(final_scores)
    log_prob = final_scores[best_final]

    # Traceback (not jit-compatible, use for debugging)
    path = []
    ci, cj, ck = int(Lx), int(Ly), int(best_final)
    while ci > 0 or cj > 0:
        path.append((ci, cj, ck))
        pi_, pj_, pk_ = TB[ci, cj, ck]
        ci, cj, ck = int(pi_), int(pj_), int(pk_)
    path.append((0, 0, int(S)))
    path.reverse()

    return log_prob, path


def sample_traceback_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi, rng_key):
    """Stochastic Forward traceback: sample a path proportional to probability.

    Runs the forward algorithm, then traces back from (Lx, Ly) to (0, 0),
    sampling predecessor states proportional to forward probability × transition.

    Args:
        log_trans, state_types, x_seq, y_seq, sub_matrix, pi: same as forward_2d
        rng_key: JAX random key

    Returns:
        log_prob: total forward log-probability
        path: sampled path as list of (i, j, state) tuples
    """
    import jax.random as jr

    log_prob, F = forward_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi)
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]
    n_states = log_trans.shape[0]
    log_sub = jnp.log(sub_matrix + 1e-30)
    log_pi = jnp.log(pi + 1e-30)

    # Sample terminal state
    e_idx = _find_e_idx(state_types)
    terminal_scores = F[Lx, Ly, :] + log_trans[:, e_idx]
    terminal_probs = jnp.exp(terminal_scores - jax.nn.logsumexp(terminal_scores))
    rng_key, subkey = jr.split(rng_key)
    ck = int(jr.choice(subkey, n_states, p=terminal_probs))

    path = []
    ci, cj = int(Lx), int(Ly)

    while ci > 0 or cj > 0:
        path.append((ci, cj, ck))
        st = state_types[ck]

        # Determine predecessor position based on current state type
        if st == M and ci > 0 and cj > 0:
            pi_, pj_ = ci - 1, cj - 1
        elif st == I and cj > 0:
            pi_, pj_ = ci, cj - 1
        elif st == D and ci > 0:
            pi_, pj_ = ci - 1, cj
        else:
            break

        # Sample predecessor state
        pred_scores = F[pi_, pj_, :] + log_trans[:, ck]
        pred_probs = jnp.exp(pred_scores - jax.nn.logsumexp(pred_scores))
        rng_key, subkey = jr.split(rng_key)
        pk = int(jr.choice(subkey, n_states, p=pred_probs))

        ci, cj, ck = pi_, pj_, pk

    path.append((0, 0, int(S)))
    path.reverse()

    return log_prob, path


# --- Banded 2D Pair HMM DP ---
#
# JIT-compiled banded variants that restrict computation to cells within
# a band around a guide alignment.  At row i the band covers
# j in [band_center[i] - W, band_center[i] + W] where W = band_width.
# Total width per row is BW = 2W+1.
#
# Storage:  F_bands / B_bands are (Lx_pad+1, BW, ns) arrays where position
# (i, b, k) stores the value for global j = band_center[i] - W + b.

def _get_band_val(band, center, j, W, BW, Ly):
    """Look up forward/backward value at global j from a band."""
    b = j - center + W
    valid = (b >= 0) & (b < BW) & (j >= 0) & (j <= Ly)
    return jnp.where(valid, band[jnp.clip(b, 0, BW - 1)], NEG_INF)


def _forward_2d_banded_core(log_trans, state_types, emit, Lx, Ly,
                             band_center, W):
    """Core banded forward algorithm for a Pair HMM.

    Args:
        log_trans: (ns, ns) log transition matrix
        state_types: (ns,) state type codes
        emit: (Lx_pad+1, Ly_pad+1, ns) log emission probabilities
        Lx, Ly: real sequence lengths
        band_center: (Lx_pad+1,) int array, expected j for each i
        W: int, half band width

    Returns:
        log_prob: total log probability
        F_bands: (Lx_pad+1, BW, ns) banded forward table
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)
    BW = 2 * W + 1

    gv = partial(_get_band_val, W=W, BW=BW, Ly=Ly)

    # Row 0: Start at (0,0), then I-type moves along j
    c0 = band_center[0]

    def row0_step(prev_cell, b):
        j = c0 - W + b
        valid = (j >= 0) & (j <= Ly)

        start_cell = jnp.where(jnp.arange(ns) == S, 0.0, NEG_INF)

        e = emit[0, jnp.clip(j, 0, Ly)]
        raw = jax.nn.logsumexp(prev_cell[:, None] + log_trans, axis=0) + e
        i_cell = jnp.where(is_I, raw, NEG_INF)

        cell = jnp.where(
            j == 0, start_cell,
            jnp.where(valid & (j > 0), i_cell, NEG_INF))

        carry = jnp.where(valid, cell, prev_cell)
        return carry, cell

    _, row0_band = jax.lax.scan(row0_step, jnp.full(ns, NEG_INF),
                                 jnp.arange(BW))

    def row_step(carry, i):
        prev_band, prev_center = carry
        center = band_center[i]

        def col_step(prev_cell, b):
            j = center - W + b
            valid = (j >= 0) & (j <= Ly) & (i <= Lx)

            e = emit[jnp.clip(i, 0, Lx), jnp.clip(j, 0, Ly)]

            m_src = gv(prev_band, prev_center, j - 1)
            m_val = jax.nn.logsumexp(
                m_src[:, None] + log_trans, axis=0) + e

            i_val = jax.nn.logsumexp(
                prev_cell[:, None] + log_trans, axis=0) + e

            d_src = gv(prev_band, prev_center, j)
            d_val = jax.nn.logsumexp(
                d_src[:, None] + log_trans, axis=0) + e

            cell = jnp.where(
                is_M & valid & (j > 0), m_val,
                jnp.where(
                    is_I & valid & (j > 0), i_val,
                    jnp.where(
                        is_D & valid, d_val, NEG_INF)))

            carry_out = jnp.where(valid, cell, prev_cell)
            return carry_out, cell

        _, curr_band = jax.lax.scan(col_step, jnp.full(ns, NEG_INF),
                                     jnp.arange(BW))
        return (curr_band, center), curr_band

    Lx_pad = band_center.shape[0] - 1
    _, all_bands = jax.lax.scan(
        row_step, (row0_band, c0), jnp.arange(1, Lx_pad + 1))

    F_bands = jnp.concatenate([row0_band[None], all_bands], axis=0)

    f_final = gv(F_bands[Lx], band_center[Lx], Ly)
    e_idx = _find_e_idx(state_types)
    log_prob = jax.nn.logsumexp(f_final + log_trans[:, e_idx])

    return log_prob, F_bands


def _backward_2d_banded_core(log_trans, state_types, emit, Lx, Ly,
                              band_center, W):
    """Core banded backward algorithm for a Pair HMM.

    Returns:
        B_bands: (Lx_pad+1, BW, ns) banded backward table
    """
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)
    BW = 2 * W + 1

    gv = partial(_get_band_val, W=W, BW=BW, Ly=Ly)
    Lx_pad = band_center.shape[0] - 1

    cLx = band_center[Lx]

    def last_row_step(beta_right, b_rev):
        b = BW - 1 - b_rev
        j = cLx - W + b
        valid = (j >= 0) & (j <= Ly)

        e_idx = _find_e_idx(state_types)
        beta_final = log_trans[:, e_idx]

        succ_j = jnp.clip(j + 1, 0, Ly)
        succ_emit = emit[Lx, succ_j]
        contrib_I = log_trans + succ_emit[None, :] + beta_right[None, :]
        contrib_I = jnp.where(is_I[None, :], contrib_I, NEG_INF)
        beta_j = jax.nn.logsumexp(contrib_I, axis=1)

        cell = jnp.where(
            j == Ly, beta_final,
            jnp.where(valid & (j < Ly), beta_j, NEG_INF))

        carry = jnp.where(valid, cell, beta_right)
        return carry, cell

    _, last_band_rev = jax.lax.scan(
        last_row_step, jnp.full(ns, NEG_INF), jnp.arange(BW))
    last_band = last_band_rev[::-1]

    def row_step_bwd(carry, i):
        next_band, next_center = carry
        center = band_center[i]

        def col_step_bwd(beta_right, b_rev):
            b = BW - 1 - b_rev
            j = center - W + b
            valid = (j >= 0) & (j <= Ly) & (i >= 0)

            succ_i = jnp.clip(i + 1, 0, Lx)
            succ_j = jnp.clip(j + 1, 0, Ly)

            succ_emit_M = emit[succ_i, succ_j]
            beta_M = gv(next_band, next_center, j + 1)
            contrib_M = log_trans + succ_emit_M[None, :] + beta_M[None, :]
            contrib_M = jnp.where(is_M[None, :] & (j < Ly), contrib_M,
                                  NEG_INF)

            succ_emit_I = emit[jnp.clip(i, 0, Lx), succ_j]
            contrib_I = log_trans + succ_emit_I[None, :] + beta_right[None, :]
            contrib_I = jnp.where(is_I[None, :] & (j < Ly), contrib_I,
                                  NEG_INF)

            succ_emit_D = emit[succ_i, jnp.clip(j, 0, Ly)]
            beta_D = gv(next_band, next_center, j)
            contrib_D = log_trans + succ_emit_D[None, :] + beta_D[None, :]
            contrib_D = jnp.where(is_D[None, :] & (i < Lx), contrib_D,
                                  NEG_INF)

            all_c = jnp.logaddexp(contrib_M,
                                  jnp.logaddexp(contrib_I, contrib_D))
            cell = jnp.where(valid, jax.nn.logsumexp(all_c, axis=1),
                             NEG_INF)

            carry = jnp.where(valid, cell, beta_right)
            return carry, cell

        _, band_rev = jax.lax.scan(
            col_step_bwd, jnp.full(ns, NEG_INF), jnp.arange(BW))
        curr_band = band_rev[::-1]
        return (curr_band, center), curr_band

    _, all_bands_bwd = jax.lax.scan(
        row_step_bwd,
        (last_band, band_center[Lx]),
        jnp.arange(Lx_pad - 1, -1, -1))

    B_bands = jnp.concatenate([all_bands_bwd[::-1], last_band[None]], axis=0)
    return B_bands


def _bands_to_dense(bands, band_center, Ly, W):
    """Convert banded table (N, BW, ns) to dense (N, Ly+1, ns)."""
    N, BW, ns = bands.shape
    dense = jnp.full((N, Ly + 1, ns), NEG_INF)
    for i in range(N):
        c = int(band_center[i])
        for b in range(BW):
            j = c - W + b
            if 0 <= j <= Ly:
                dense = dense.at[i, j].set(bands[i, b])
    return dense


def forward_2d_banded(log_trans, state_types, x_seq, y_seq, sub_matrix, pi,
                      band_center, band_width, log_emit_table=None):
    """Banded forward algorithm for a Pair HMM.

    Args:
        log_trans: (n_states, n_states) log transition matrix
        state_types: (n_states,) state type codes
        x_seq: (Lx,) ancestor sequence
        y_seq: (Ly,) descendant sequence
        sub_matrix: (A, A) substitution probability matrix (ignored if log_emit_table)
        pi: (A,) equilibrium distribution (ignored if log_emit_table)
        band_center: (Lx+1,) array, for each i the expected j position
        band_width: int, half-width of band (total width = 2*band_width+1)
        log_emit_table: optional (Lx+1, Ly+1, n_states) pre-computed log emissions.

    Returns:
        log_prob: total log probability P(x, y)
        F_bands: (Lx_pad+1, BW, n_states) banded forward table
    """
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]
    W = _pad_to_bin(band_width)
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)

    if log_emit_table is not None:
        ns = state_types.shape[0]
        emit = jnp.full((Lx_pad + 1, Ly_pad + 1, ns), NEG_INF)
        emit = emit.at[:Lx + 1, :Ly + 1, :].set(log_emit_table)
    else:
        x_pad = _pad_seq(x_seq, Lx_pad)
        y_pad = _pad_seq(y_seq, Ly_pad)
        emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi)
        mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, state_types.shape[0])
        emit = jnp.where(mask, emit, NEG_INF)

    bc_pad = jnp.zeros(Lx_pad + 1, dtype=jnp.int32)
    bc_pad = bc_pad.at[:Lx + 1].set(band_center)
    bc_pad = bc_pad.at[Lx + 1:].set(Ly)

    log_prob, F_bands = _forward_2d_banded_core(
        log_trans, state_types, emit, Lx, Ly, bc_pad, W)
    return log_prob, F_bands


def backward_2d_banded(log_trans, state_types, x_seq, y_seq, sub_matrix, pi,
                       band_center, band_width):
    """Banded backward algorithm for a Pair HMM.

    Returns:
        B_bands: (Lx_pad+1, BW, n_states) banded backward table
    """
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]
    W = _pad_to_bin(band_width)
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)
    x_pad = _pad_seq(x_seq, Lx_pad)
    y_pad = _pad_seq(y_seq, Ly_pad)
    emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi)
    mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)

    bc_pad = jnp.zeros(Lx_pad + 1, dtype=jnp.int32)
    bc_pad = bc_pad.at[:Lx + 1].set(band_center)
    bc_pad = bc_pad.at[Lx + 1:].set(Ly)

    B_bands = _backward_2d_banded_core(
        log_trans, state_types, emit, Lx, Ly, bc_pad, W)
    return B_bands


def forward_backward_2d_banded(log_trans, state_types, x_seq, y_seq,
                               sub_matrix, pi, band_center, band_width):
    """Banded forward-backward for a Pair HMM.

    Args:
        band_center: (Lx+1,) array, for each i the expected j position
        band_width: int, half-width of band

    Returns:
        log_prob: total log probability
        posteriors_bands: (Lx_pad+1, BW, n_states) posterior probabilities
        expected_trans: (n_states, n_states) expected transition counts
    """
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]
    W = _pad_to_bin(band_width)
    BW = 2 * W + 1
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)
    x_pad = _pad_seq(x_seq, Lx_pad)
    y_pad = _pad_seq(y_seq, Ly_pad)
    emit = pair_hmm_emissions(state_types, x_pad, y_pad, sub_matrix, pi)
    mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)
    ns = log_trans.shape[0]
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    bc_pad = jnp.zeros(Lx_pad + 1, dtype=jnp.int32)
    bc_pad = bc_pad.at[:Lx + 1].set(band_center)
    bc_pad = bc_pad.at[Lx + 1:].set(Ly)

    log_prob, F_bands = _forward_2d_banded_core(
        log_trans, state_types, emit, Lx, Ly, bc_pad, W)
    B_bands = _backward_2d_banded_core(
        log_trans, state_types, emit, Lx, Ly, bc_pad, W)

    posteriors_bands = jnp.exp(F_bands + B_bands - log_prob)

    gv = partial(_get_band_val, W=W, BW=BW, Ly=Ly)

    def row_trans_counts(i):
        center = bc_pad[i]
        next_center = bc_pad[jnp.clip(i + 1, 0, Lx_pad)]
        f_band = F_bands[i]
        b_next = B_bands[jnp.clip(i + 1, 0, Lx_pad)]
        b_curr = B_bands[i]

        def band_trans(b):
            j = center - W + b
            valid_ij = (j >= 0) & (j <= Ly) & (i <= Lx)
            f_src = f_band[b]

            succ_i = jnp.clip(i + 1, 0, Lx_pad)
            succ_j = jnp.clip(j + 1, 0, Ly_pad)

            e_M = emit[succ_i, succ_j]
            beta_M = gv(b_next, next_center, j + 1)
            log_n_M = (f_src[:, None] + log_trans +
                       e_M[None, :] + beta_M[None, :] - log_prob)
            log_n_M = jnp.where(
                is_M[None, :] & valid_ij & (i < Lx) & (j >= 0) & (j < Ly),
                log_n_M, NEG_INF)

            e_I = emit[jnp.clip(i, 0, Lx_pad), succ_j]
            beta_I = gv(b_curr, center, j + 1)
            log_n_I = (f_src[:, None] + log_trans +
                       e_I[None, :] + beta_I[None, :] - log_prob)
            log_n_I = jnp.where(
                is_I[None, :] & valid_ij & (j < Ly),
                log_n_I, NEG_INF)

            e_D = emit[succ_i, jnp.clip(j, 0, Ly_pad)]
            beta_D = gv(b_next, next_center, j)
            log_n_D = (f_src[:, None] + log_trans +
                       e_D[None, :] + beta_D[None, :] - log_prob)
            log_n_D = jnp.where(
                is_D[None, :] & valid_ij & (i < Lx),
                log_n_D, NEG_INF)

            counts = (jnp.exp(log_n_M) + jnp.exp(log_n_I) +
                      jnp.exp(log_n_D))
            return counts

        all_counts = jax.vmap(band_trans)(jnp.arange(BW))
        return all_counts.sum(axis=0)

    all_row_counts = jax.vmap(row_trans_counts)(jnp.arange(Lx_pad + 1))
    expected_trans = all_row_counts.sum(axis=0)

    f_final = gv(F_bands[Lx], bc_pad[Lx], Ly)
    e_idx = _find_e_idx(state_types)
    log_count_E = f_final + log_trans[:, e_idx] - log_prob
    expected_trans = expected_trans.at[:, e_idx].add(jnp.exp(log_count_E))

    return log_prob, posteriors_bands, expected_trans


def msa_to_band_center(msa_x_row, msa_y_row, Lx, gap_token=-1):
    """Convert MSA guide alignment to band center array for banded DP.

    Args:
        msa_x_row: (L_msa,) aligned row for x (gap_token for gaps)
        msa_y_row: (L_msa,) aligned row for y (gap_token for gaps)
        Lx: length of sequence x (ungapped)
        gap_token: gap character value (default -1)

    Returns:
        band_center: (Lx+1,) int array, for each i the expected j position
    """
    msa_x_row = np.asarray(msa_x_row)
    msa_y_row = np.asarray(msa_y_row)

    guide = [(0, 0)]
    xi, yj = 0, 0
    for col in range(len(msa_x_row)):
        if int(msa_x_row[col]) != gap_token:
            xi += 1
        if int(msa_y_row[col]) != gap_token:
            yj += 1
        guide.append((xi, yj))

    band_center = np.zeros(Lx + 1, dtype=np.int32)
    guide_idx = 0
    for i in range(Lx + 1):
        while guide_idx < len(guide) - 1 and guide[guide_idx][0] < i:
            guide_idx += 1
        band_center[i] = guide[guide_idx][1]

    return jnp.array(band_center)
