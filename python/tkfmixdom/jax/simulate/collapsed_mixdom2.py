"""Simulator A — Collapsed Pair HMM categorical sampler for MixDom2 / MixDom1.

Walks the collapsed MixDom2 pair HMM (built via
:func:`tkfmixdom.jax.models.mixdom.build_nested_trans`) as a Markov chain.

At each step:
  1. Sample ``next_state`` from ``chi[curr_state, :]`` where ``chi`` is built
     once per pair at that pair's branch length ``t``.
  2. Sample the emitted ancestor/descendant pair from the per-state emission
     distribution:
       - M (match)  : per-class joint draw ``(a, b) ~ pi_c[a] · P_c(t)[a, b]``
                      with class index sampled from ``classdist[d, f, :]``.
       - I (insert) : ``b ~ pi_c[b]`` with class from ``classdist[d, f, :]``.
       - D (delete) : ``a ~ pi_c[a]`` with class from ``classdist[d, f, :]``.

No DP is involved — this is a direct categorical walk through chi, with one
``np.random.choice`` per transition and one per emission. It is fast enough
to drive high-throughput parameter recovery tests with thousands of pairs.

MixDom1 is the special case ``n_classes = 1``; the per-class indirection
collapses to a single-class draw using a per-domain ``Q`` built from
``class_pis[0]`` and ``class_S_exch[0]`` (or a per-domain Q if supplied).

The output is the precompiled-pair tuple format consumed by
``train_pfam._process_pairs_batched``:
    ``(x_int, y_int, states, anc_chars, desc_chars, t)``
plus the underlying ``state_path`` (a list of MixDom flat state indices).
"""

from __future__ import annotations

import numpy as np

from ..core.params import S as TYPE_S, M as TYPE_M, I as TYPE_I, D as TYPE_D, E as TYPE_E
from ..models.mixdom import (
    build_nested_trans,
    n_states as mixdom_n_states,
    state_types as mixdom_state_types,
)
from .labeled_utils import per_class_transition_matrices


# Special MixDom flat-state indices (matches mixdom.py).
SS_INDEX = 0
EE_INDEX = 1


# Module-level cache: build_nested_trans is JAX-traced; we wrap it in a
# JIT'd function so successive calls at different t reuse the compiled
# function instead of re-tracing per call.
_BUILD_CHI_JIT_CACHE = None


def _get_build_chi_jit():
    global _BUILD_CHI_JIT_CACHE
    if _BUILD_CHI_JIT_CACHE is None:
        import jax
        @jax.jit
        def _build(t, main_ins, main_del, dom_ins, dom_del, dom_w, frag_w, ext):
            chi, _ = build_nested_trans(
                main_ins_rate=main_ins, main_del_rate=main_del, t=t,
                dom_ins_rates=dom_ins, dom_del_rates=dom_del,
                dom_weights=dom_w, frag_weights=frag_w, ext_rates=ext)
            return chi
        _BUILD_CHI_JIT_CACHE = _build
    return _BUILD_CHI_JIT_CACHE


def _state_to_dom_frag_uv(s, n_frag):
    """Decompose a body state index ``s`` into ``(uv, d, f)``.

    Layout from ``mixdom.py``: ``index 2 + d * 5 * n_frag + uv * n_frag + f``.
    """
    if s < 2:
        return None  # SS or EE
    body = s - 2
    block_size = 5 * n_frag
    d = body // block_size
    rem = body % block_size
    uv = rem // n_frag
    f = rem % n_frag
    return (uv, int(d), int(f))


def _build_emission_tables(params, t):
    """Pre-compute per-(d, f) emission distributions used at every step.

    Returns a dict with keys:
      - ``Pmats``: (C, A, A) per-class transition probability matrices at time t.
      - ``class_pis``: (C, A) per-class equilibrium distributions.
      - ``classdist``: (D, F, C) per-(domain, fragment) class distributions.
      - ``joint_match[d, f, :, :]``: (A, A) marginal P(a, b | s in M-state of (d, f)),
            already summed over classes. Used for fast match emission sampling
            when the sampler doesn't care about which class generated the pair.
    """
    n_dom = int(np.asarray(params['dom_ins']).shape[0])
    n_frag = int(np.asarray(params['frag_weights']).shape[1])

    # Choose class machinery: explicit classdist if provided, else MixDom1
    # singleton equivalent built from a per-domain Q (LG by default).
    if 'classdist' in params and 'class_pis' in params and 'class_S_exch' in params:
        classdist = np.asarray(params['classdist'], dtype=np.float64)
        class_pis = np.asarray(params['class_pis'], dtype=np.float64)
        class_S_exch = np.asarray(params['class_S_exch'], dtype=np.float64)
        Pmats, _ = per_class_transition_matrices(class_pis, class_S_exch, t)
    else:
        # Fallback to MixDom1: single class per (d, f), using LG Q/pi
        from ..core.protein import rate_matrix_lg
        Q_lg, pi_lg = rate_matrix_lg()
        pi_lg = np.asarray(pi_lg, dtype=np.float64)
        Q_lg = np.asarray(Q_lg, dtype=np.float64)
        from ..core.ctmc import transition_matrix
        import jax.numpy as jnp
        P_lg = np.array(transition_matrix(jnp.array(Q_lg), t))
        class_pis = pi_lg[None, :]                       # (1, A)
        class_S_exch = (Q_lg / np.maximum(pi_lg[None, :], 1e-30))[None]  # (1, A, A)
        np.fill_diagonal(class_S_exch[0], 0.0)
        classdist = np.ones((n_dom, n_frag, 1), dtype=np.float64)
        Pmats = P_lg[None, :, :]
    A = class_pis.shape[1]

    # Pre-compute per-(d, f) joint match distribution: P(a, b | (d, f))
    # = Σ_c classdist[d, f, c] * pi_c[a] * Pmats[c][a, b].
    # Shape (D, F, A, A).
    joint_match = np.zeros((n_dom, n_frag, A, A), dtype=np.float64)
    for d in range(n_dom):
        for f in range(n_frag):
            cd = classdist[d, f]                                         # (C,)
            # weights[c, a, b] = cd[c] * pi_c[a] * Pmats[c, a, b]
            jm = (cd[:, None, None]
                  * class_pis[:, :, None]
                  * Pmats).sum(axis=0)
            jm = jm / max(jm.sum(), 1e-300)
            joint_match[d, f] = jm

    # Per-(d, f) marginal residue distribution for I/D states:
    # = Σ_c classdist[d, f, c] * pi_c[a]
    pi_marg = (classdist[:, :, :, None] * class_pis[None, None, :, :]).sum(axis=2)
    pi_marg = pi_marg / np.maximum(pi_marg.sum(axis=-1, keepdims=True), 1e-300)

    return {
        'classdist': classdist,
        'class_pis': class_pis,
        'class_S_exch': class_S_exch,
        'Pmats': Pmats,
        'joint_match': joint_match,
        'pi_marg': pi_marg,                               # (D, F, A)
        'A': A,
        'n_dom': n_dom,
        'n_frag': n_frag,
    }


def _categorical(np_rng, p):
    """Sample one integer from a 1D probability vector ``p`` (numerically robust)."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0.0:
        # Defensive: degenerate row — return argmax as a fallback.
        return int(np.argmax(p))
    return int(np_rng.choice(p.shape[0], p=p / s))


def simulate_collapsed_mixdom2_pair(np_rng, params, t, max_len=2000):
    """Sample one (anc, desc) pair from the collapsed MixDom2 pair HMM at ``t``.

    Args:
        np_rng:     ``np.random.RandomState`` (or compatible).
        params:     MixDom2 params dict (keys main_ins, main_del, dom_ins,
                    dom_del, dom_weights, frag_weights, ext_rates; optionally
                    classdist, class_pis, class_S_exch for n_classes > 1).
        t:          float branch length for this pair.
        max_len:    safety bound on alignment length (excludes SS/EE).

    Returns:
        x_int (np.ndarray int32) :  ancestor sequence
        y_int (np.ndarray int32) :  descendant sequence
        states (list[int])       :  per-column state codes (M=1, I=2, D=3)
        anc_chars (list[int])    :  residues at M and D positions, in column order
        desc_chars (list[int])   :  residues at M and I positions, in column order
        t (float)                :  branch length used to generate this pair
        state_path (list[int])   :  flat MixDom state indices visited (excl. SS/EE)
    """
    import jax.numpy as jnp

    # Build chi at this pair's t (JIT'd, so successive calls reuse the
    # compiled function).
    build_chi = _get_build_chi_jit()
    chi = np.asarray(build_chi(
        jnp.float32(t),
        jnp.asarray(params['main_ins'], dtype=jnp.float32),
        jnp.asarray(params['main_del'], dtype=jnp.float32),
        jnp.asarray(params['dom_ins'], dtype=jnp.float32),
        jnp.asarray(params['dom_del'], dtype=jnp.float32),
        jnp.asarray(params['dom_weights'], dtype=jnp.float32),
        jnp.asarray(params['frag_weights'], dtype=jnp.float32),
        jnp.asarray(params['ext_rates'], dtype=jnp.float32),
    ), dtype=np.float64)

    n_dom = int(np.asarray(params['dom_ins']).shape[0])
    n_frag = int(np.asarray(params['frag_weights']).shape[1])
    N = mixdom_n_states(n_dom, n_frag)
    st = np.asarray(mixdom_state_types(n_dom, n_frag))
    assert chi.shape == (N, N)

    emit = _build_emission_tables(params, t)
    A = emit['A']

    # Walk the chain
    states = []           # per-column M/I/D codes
    anc_chars = []        # residues at M+D positions
    desc_chars = []       # residues at M+I positions
    state_path = []       # flat MixDom state indices visited (body only)

    curr = SS_INDEX
    for step in range(max_len):
        row = chi[curr]
        nxt = _categorical(np_rng, row)
        if nxt == EE_INDEX:
            break
        if nxt == SS_INDEX:
            # Should never happen by construction — defensive.
            break
        # Body state: emit
        ts = int(st[nxt])
        info = _state_to_dom_frag_uv(nxt, n_frag)
        if info is None:
            break
        uv, d, f = info
        if ts == TYPE_M:
            jm = emit['joint_match'][d, f]
            ab_flat = _categorical(np_rng, jm.reshape(-1))
            a, b = divmod(ab_flat, A)
            states.append(TYPE_M)
            anc_chars.append(int(a))
            desc_chars.append(int(b))
        elif ts == TYPE_I:
            b = _categorical(np_rng, emit['pi_marg'][d, f])
            states.append(TYPE_I)
            desc_chars.append(int(b))
        elif ts == TYPE_D:
            a = _categorical(np_rng, emit['pi_marg'][d, f])
            states.append(TYPE_D)
            anc_chars.append(int(a))
        else:
            # Unexpected (S/E hit in body) — defensive end.
            break
        state_path.append(int(nxt))
        curr = nxt

    x_int = np.asarray(anc_chars, dtype=np.int32)
    y_int = np.asarray(desc_chars, dtype=np.int32)
    return x_int, y_int, states, anc_chars, desc_chars, float(t), state_path


def simulate_collapsed_mixdom2_batch(np_rng, params, t_array, max_len=2000):
    """Vectorised over a batch of branch lengths.

    Builds chi (and per-class P_c) ONCE per unique t in t_array, so that
    repeated draws at the same t do not pay the JAX trace + matrix-exponential
    cost per pair. This is essential for high-throughput tests.

    Args:
        np_rng:   ``np.random.RandomState``.
        params:   MixDom2 params dict.
        t_array:  iterable/np.ndarray of per-pair branch lengths.
        max_len:  safety bound per pair.

    Returns:
        list of (x_int, y_int, states, anc_chars, desc_chars, t, state_path) tuples
        — one per t in ``t_array``.
    """
    import jax.numpy as jnp

    n_dom = int(np.asarray(params['dom_ins']).shape[0])
    n_frag = int(np.asarray(params['frag_weights']).shape[1])
    N = mixdom_n_states(n_dom, n_frag)
    st = np.asarray(mixdom_state_types(n_dom, n_frag))

    build_chi = _get_build_chi_jit()
    main_ins = jnp.asarray(params['main_ins'], dtype=jnp.float32)
    main_del = jnp.asarray(params['main_del'], dtype=jnp.float32)
    dom_ins = jnp.asarray(params['dom_ins'], dtype=jnp.float32)
    dom_del = jnp.asarray(params['dom_del'], dtype=jnp.float32)
    dw = jnp.asarray(params['dom_weights'], dtype=jnp.float32)
    fw = jnp.asarray(params['frag_weights'], dtype=jnp.float32)
    ext = jnp.asarray(params['ext_rates'], dtype=jnp.float32)

    cache = {}    # t -> (chi, emit_tables)
    out = []
    for t_raw in t_array:
        t = float(t_raw)
        if t not in cache:
            chi = np.asarray(
                build_chi(jnp.float32(t), main_ins, main_del,
                          dom_ins, dom_del, dw, fw, ext),
                dtype=np.float64)
            emit = _build_emission_tables(params, t)
            cache[t] = (chi, emit)
        chi, emit = cache[t]
        # Inline a streamlined sampling loop that reuses chi/emit (no rebuild).
        result = _sample_one_pair_with_cache(np_rng, chi, emit, st, n_frag, t,
                                              max_len=max_len)
        out.append(result)
    return out


def _sample_one_pair_with_cache(np_rng, chi, emit, st, n_frag, t, max_len=2000):
    """Inner sampler: chi and emission tables are pre-built (cached by caller)."""
    A = emit['A']

    states = []
    anc_chars = []
    desc_chars = []
    state_path = []

    curr = SS_INDEX
    for step in range(max_len):
        row = chi[curr]
        nxt = _categorical(np_rng, row)
        if nxt == EE_INDEX:
            break
        if nxt == SS_INDEX:
            break
        ts = int(st[nxt])
        info = _state_to_dom_frag_uv(nxt, n_frag)
        if info is None:
            break
        uv, d, f = info
        if ts == TYPE_M:
            jm = emit['joint_match'][d, f]
            ab_flat = _categorical(np_rng, jm.reshape(-1))
            a, b = divmod(ab_flat, A)
            states.append(TYPE_M)
            anc_chars.append(int(a))
            desc_chars.append(int(b))
        elif ts == TYPE_I:
            b = _categorical(np_rng, emit['pi_marg'][d, f])
            states.append(TYPE_I)
            desc_chars.append(int(b))
        elif ts == TYPE_D:
            a = _categorical(np_rng, emit['pi_marg'][d, f])
            states.append(TYPE_D)
            anc_chars.append(int(a))
        else:
            break
        state_path.append(int(nxt))
        curr = nxt
    x_int = np.asarray(anc_chars, dtype=np.int32)
    y_int = np.asarray(desc_chars, dtype=np.int32)
    return x_int, y_int, states, anc_chars, desc_chars, float(t), state_path


def empirical_state_distribution(samples, n_states_total):
    """Marginal frequency of each MixDom state across a list of sample tuples.

    Useful for sanity-check that the empirical state distribution matches the
    chi stationary distribution (run a long sample and compare).

    Args:
        samples: iterable of tuples returned by simulate_collapsed_mixdom2_*.
                 Each tuple's 7th element (index 6) must be the ``state_path``.
        n_states_total: ``mixdom_n_states(n_dom, n_frag)`` — the size of the
                        empirical distribution vector to return.

    Returns:
        (N,) numpy array of state frequencies summing to 1.
    """
    counts = np.zeros(n_states_total, dtype=np.float64)
    total = 0
    for s in samples:
        path = s[6]
        for k in path:
            counts[k] += 1.0
            total += 1
    if total == 0:
        return counts
    return counts / total
