"""Anti-diagonal (wavefront) 2D Pair HMM dynamic programming.

Cells (i, j) on anti-diagonal d = i + j depend only on diagonals d-1 (I, D
predecessors) and d-2 (M predecessor), so each diagonal is one vectorized
step and the sequential depth is Lx + Ly. The layout follows
tkfmixdom.jax.dp.hmm._forward_2d_core_diag: local index k on diagonal d is
i - i_min(d), i_min(d) = max(0, d - Ly), and D_max = min(Lx, Ly) + 1.

Emissions are given as three pieces rather than an (Lx+1, Ly+1, ns) table:
  eM(i, j)  match emission of cell (i, j), i, j >= 1 (a callable on index arrays)
  eI[j]     insert emission of y_j, j >= 1          (length Ly + 1)
  eD[i]     delete emission of x_i, i >= 1          (length Lx + 1)
and apply to every state of that type (emissions do not depend on fragtype).

Lx, Ly are the padded (static) sizes; the true lengths real_Lx, real_Ly may be
traced, so one compiled kernel serves a batch of pairs under jax.vmap. Cells
beyond the true lengths are computed but never reach the captured endpoint.
"""
import jax
import jax.numpy as jnp

S, M, I, D, E = 0, 1, 2, 3, 4
NEG_INF = -1e30


def _i_min(d, Ly):
    return jnp.maximum(0, d - Ly)


def _diag_step(prev, prev_prev, d, logT, is_M, is_I, is_D, eM, eI, eD, Lx, Ly, D_max, viterbi):
    ks = jnp.arange(D_max)
    i = _i_min(d, Ly) + ks
    j = d - i
    valid = (i <= Lx) & (j >= 0) & (j <= Ly)
    ic, jc = jnp.clip(i, 0, Lx), jnp.clip(j, 0, Ly)
    m_pred = prev_prev[jnp.clip((i - 1) - _i_min(d - 2, Ly), 0, D_max - 1)]
    i_pred = prev[jnp.clip(i - _i_min(d - 1, Ly), 0, D_max - 1)]
    d_pred = prev[jnp.clip((i - 1) - _i_min(d - 1, Ly), 0, D_max - 1)]

    def combine(pred):                       # (D_max, ns) -> (D_max, ns), and argmax
        x = pred[:, :, None] + logT[None, :, :]
        if viterbi:
            return jnp.max(x, axis=1), jnp.argmax(x, axis=1).astype(jnp.int8)
        return jax.nn.logsumexp(x, axis=1), None

    vM, bM = combine(m_pred)
    vI, bI = combine(i_pred)
    vD, bD = combine(d_pred)
    vM = vM + eM(jnp.clip(ic, 1, Lx), jnp.clip(jc, 1, Ly))[:, None]
    vI = vI + eI[jc][:, None]
    vD = vD + eD[ic][:, None]
    vM = jnp.where(((i >= 1) & (j >= 1))[:, None], vM, NEG_INF)
    vI = jnp.where((j >= 1)[:, None], vI, NEG_INF)
    vD = jnp.where((i >= 1)[:, None], vD, NEG_INF)
    cells = jnp.where(is_M, vM, jnp.where(is_I, vI, jnp.where(is_D, vD, NEG_INF)))
    cells = jnp.maximum(jnp.where(valid[:, None], cells, NEG_INF), NEG_INF)
    if viterbi:
        bp = jnp.where(is_M, bM, jnp.where(is_I, bI, bD)).astype(jnp.int8)
        return cells, bp
    return cells, None


def forward(logT, state_types, eM, eI, eD, Lx, Ly, real_Lx, real_Ly):
    """Log Forward probability. Linear space; the scan body is checkpointed so
    reverse-mode gradients (expected counts) do not store per-cell residuals."""
    ns = logT.shape[0]
    is_M, is_I, is_D = state_types == M, state_types == I, state_types == D
    e_idx = jnp.argmax(state_types == E)
    D_max = min(Lx, Ly) + 1
    diag0 = jnp.full((D_max, ns), NEG_INF).at[0, S].set(0.0)
    d_end = real_Lx + real_Ly
    k_end = real_Lx - _i_min(d_end, Ly)

    @jax.checkpoint
    def body(carry, d):
        prev, prev_prev, saved = carry
        cells, _ = _diag_step(prev, prev_prev, d, logT, is_M, is_I, is_D,
                              eM, eI, eD, Lx, Ly, D_max, viterbi=False)
        saved = jnp.where(d == d_end, cells[k_end], saved)
        return (cells, prev, saved), None

    (_, _, saved), _ = jax.lax.scan(
        body, (diag0, jnp.full((D_max, ns), NEG_INF), diag0[0]), jnp.arange(1, Lx + Ly + 1))
    return jax.nn.logsumexp(saved + logT[:, e_idx])


def viterbi(logT, state_types, eM, eI, eD, Lx, Ly, real_Lx, real_Ly):
    """Viterbi score, final state, and back-pointers (Lx+Ly+1, D_max, ns) int8:
    bp[d, k, s] is the best predecessor state of state s at cell k of diagonal d."""
    ns = logT.shape[0]
    is_M, is_I, is_D = state_types == M, state_types == I, state_types == D
    e_idx = jnp.argmax(state_types == E)
    D_max = min(Lx, Ly) + 1
    diag0 = jnp.full((D_max, ns), NEG_INF).at[0, S].set(0.0)
    d_end = real_Lx + real_Ly
    k_end = real_Lx - _i_min(d_end, Ly)

    def body(carry, d):
        prev, prev_prev, saved = carry
        cells, bp = _diag_step(prev, prev_prev, d, logT, is_M, is_I, is_D,
                               eM, eI, eD, Lx, Ly, D_max, viterbi=True)
        saved = jnp.where(d == d_end, cells[k_end], saved)
        return (cells, prev, saved), bp

    (_, _, saved), bps = jax.lax.scan(
        body, (diag0, jnp.full((D_max, ns), NEG_INF), diag0[0]), jnp.arange(1, Lx + Ly + 1))
    fin = saved + logT[:, e_idx]
    bps = jnp.concatenate([jnp.zeros((1, D_max, ns), jnp.int8), bps], axis=0)
    return jnp.max(fin), jnp.argmax(fin), bps


def traceback(bps, state_types, last_state, Lx_real, Ly_real, Ly_pad):
    """Host-side traceback. Returns the column types (M/I/D codes) in order."""
    import numpy as np
    st = np.asarray(state_types)
    bps = np.asarray(bps)
    i, j, s = int(Lx_real), int(Ly_real), int(last_state)
    cols = []
    while st[s] != S:
        d = i + j
        k = i - max(0, d - Ly_pad)
        p = int(bps[d, k, s])
        t = st[s]
        cols.append(t)
        if t == M:
            i, j = i - 1, j - 1
        elif t == I:
            j -= 1
        else:
            i -= 1
        s = p
    assert i == 0 and j == 0, (i, j)
    return cols[::-1]
