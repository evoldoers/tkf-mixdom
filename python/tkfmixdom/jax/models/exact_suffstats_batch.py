"""Batched (per-pair-vectorised) version of `exact_suffstats`.

`exact_suffstats_per_pair_batch` calls `exact_suffstats` once per pair in a
Python loop. For B=2000 pairs at d3f1, that loop dominates the iteration
wall time after the BDI vmap fix (≈ 130 sec/iter). This module batches the
chain restoration over a leading B axis so the elimination, restoration,
and stat-extraction all happen on (B, N, N)-shaped numpy arrays in a
single set of numpy ops.

Per-pair-t coherence is preserved: each batch element b uses its own
`t_array[b]` and `n_chi_stack[b]`. The (params, state-index) data are
shared across the batch — same as in the serial version.

The batched ops are all numpy (np.linalg.inv handles the leading B axis;
matmul broadcasts; fancy indexing with the same keep_idx / elim_idx works
across the batch). No JAX is required for this module — the work is done
on CPU with vectorised numpy.
"""

import numpy as np
import jax
import jax.numpy as jnp

from ..core.params import S, M, I, D, E, tkf91_trans
from .fully_exploded import build_fully_exploded


# ============================================================
# Vectorised TKF tau matrices over a t-batch.
# ============================================================

_tau_batch_jit = None


def _get_tau_batch_jit():
    """Lazy-init: vmapped JIT'd `tkf91_trans` over (lam, mu, t) where t is (B,)."""
    global _tau_batch_jit
    if _tau_batch_jit is None:
        # vmap over t only (lam, mu shared)
        _tau_batch_jit = jax.jit(
            jax.vmap(tkf91_trans, in_axes=(None, None, 0)))
    return _tau_batch_jit


def _tau_stack(ins_rate, del_rate, t_array):
    """Compute tau_b[b] = tkf91_trans(ins_rate, del_rate, t_array[b]).

    Returns np.ndarray (B, 5, 5).
    """
    fn = _get_tau_batch_jit()
    return np.asarray(fn(jnp.asarray(float(ins_rate)),
                         jnp.asarray(float(del_rate)),
                         jnp.asarray(t_array)))


# ============================================================
# Batched fully exploded transition matrix.
# ============================================================


def build_fully_exploded_batch(main_ins, main_del, t_array,
                                dom_ins, dom_del, dom_weights,
                                frag_weights, ext_rates):
    """Vectorised version of `build_fully_exploded` over a leading t-batch.

    Equivalent to looping
        T_full_stack[b], _, _, names, idx = build_fully_exploded(
            main_ins, main_del, t_array[b], dom_ins, dom_del, ...)
    but with T_full_stack assembled in one shot per (src, dst) entry —
    no per-pair Python interpretation overhead and no per-pair dict lookups.

    Returns:
        T_full_stack: (B, N_full, N_full)
        emit_indices, null_indices, names, idx: same as `build_fully_exploded`
                                                (these are batch-independent).
    """
    B = int(np.asarray(t_array).shape[0])
    n_dom = len(dom_ins)
    n_frag = (frag_weights.shape[1] if hasattr(frag_weights, 'shape')
              else len(frag_weights[0]))

    dom_ins = np.asarray(dom_ins)
    dom_del = np.asarray(dom_del)
    dom_weights = np.asarray(dom_weights)
    frag_weights = np.asarray(frag_weights)
    ext_rates = np.asarray(ext_rates)

    # Per-pair tau matrices (vmapped JAX call, single dispatch each).
    # Shapes: (B, 5, 5) for tau_0_b; (B, K, 5, 5) for tau_k_b.
    tau_0_b = _tau_stack(float(main_ins), float(main_del), t_array)
    tau_k_b = np.stack(
        [_tau_stack(float(dom_ins[k]), float(dom_del[k]), t_array)
         for k in range(n_dom)], axis=1)
    kappa_k = dom_ins / dom_del  # t-independent, shape (K,)

    # MixDom1 (D, F) ext -> MixDom2 (D, F, F)
    if ext_rates.ndim == 2:
        ext_rates_3d = np.zeros((n_dom, n_frag, n_frag))
        for d in range(n_dom):
            ext_rates_3d[d] = np.diag(ext_rates[d])
        ext_rates = ext_rates_3d

    # Build the state-index map (identical to scalar `build_fully_exploded`).
    names = []
    idx = {}

    def add(name):
        i = len(names)
        names.append(name)
        idx[name] = i
        return i

    add('Start')
    add('End')
    add('MatDom')
    add('InsDom')
    add('DelDom')
    for k in range(n_dom):
        add(f'MatDomType[{k}]')
        add(f'InsDomType[{k}]')
        add(f'DelDomType[{k}]')
    for k in range(n_dom):
        add(f'MatFrag[{k}]')
        add(f'InsFrag[{k}]')
        add(f'DelFrag[{k}]')
        add(f'IFrag[{k}]')
        add(f'DFrag[{k}]')
    for k in range(n_dom):
        for f in range(n_frag):
            add(f'MatFragType[{k},{f}]')
            add(f'InsFragType[{k},{f}]')
            add(f'DelFragType[{k},{f}]')
            add(f'IFragType[{k},{f}]')
            add(f'DFragType[{k},{f}]')
    emit_indices = []
    _emit_names = ['MatEmit', 'InsEmit', 'DelEmit', 'IEmit', 'DEmit']
    for k in range(n_dom):
        for uv_idx, emit_prefix in enumerate(_emit_names):
            for f in range(n_frag):
                emit_indices.append(add(f'{emit_prefix}[{k},{f}]'))
    for k in range(n_dom):
        for f in range(n_frag):
            add(f'MatFragEnd[{k},{f}]')
            add(f'InsFragEnd[{k},{f}]')
            add(f'DelFragEnd[{k},{f}]')
            add(f'IFragEnd[{k},{f}]')
            add(f'DFragEnd[{k},{f}]')
    add('MatDomEnd')
    add('InsDomEnd')
    add('DelDomEnd')

    N = len(names)
    null_indices = [i for i in range(N) if i not in emit_indices]

    T = np.zeros((B, N, N))

    # Top-level transitions: rows {Start, MatDomEnd, InsDomEnd} use beta;
    # DelDomEnd row uses gamma. Each entry is per-pair (B,).
    def top_row(src_idx, gamma_row):
        bg = tau_0_b[:, D, :] if gamma_row else tau_0_b[:, S, :]  # (B, 5)
        T[:, src_idx, idx['MatDom']] = bg[:, M]
        T[:, src_idx, idx['InsDom']] = bg[:, I]
        T[:, src_idx, idx['DelDom']] = bg[:, D]
        T[:, src_idx, idx['End']] = bg[:, E]

    top_row(idx['Start'], False)
    top_row(idx['MatDomEnd'], False)
    top_row(idx['InsDomEnd'], False)
    top_row(idx['DelDomEnd'], True)

    # Domain type selection (t-independent dom_weights)
    for k in range(n_dom):
        T[:, idx['MatDom'], idx[f'MatDomType[{k}]']] = dom_weights[k]
        T[:, idx['InsDom'], idx[f'InsDomType[{k}]']] = dom_weights[k]
        T[:, idx['DelDom'], idx[f'DelDomType[{k}]']] = dom_weights[k]

    # M-type domain entry (TKF91_k S row)
    for k in range(n_dom):
        T[:, idx[f'MatDomType[{k}]'], idx[f'MatFrag[{k}]']] = tau_k_b[:, k, S, M]
        T[:, idx[f'MatDomType[{k}]'], idx[f'InsFrag[{k}]']] = tau_k_b[:, k, S, I]
        T[:, idx[f'MatDomType[{k}]'], idx[f'DelFrag[{k}]']] = tau_k_b[:, k, S, D]
        T[:, idx[f'MatDomType[{k}]'], idx['MatDomEnd']] = tau_k_b[:, k, S, E]

    # I/D-type domain entry (kappa loop) — t-independent
    for k in range(n_dom):
        T[:, idx[f'InsDomType[{k}]'], idx[f'IFrag[{k}]']] = kappa_k[k]
        T[:, idx[f'InsDomType[{k}]'], idx['InsDomEnd']] = 1.0 - kappa_k[k]
        T[:, idx[f'DelDomType[{k}]'], idx[f'DFrag[{k}]']] = kappa_k[k]
        T[:, idx[f'DelDomType[{k}]'], idx['DelDomEnd']] = 1.0 - kappa_k[k]

    # Fragment type selection (t-independent frag_weights)
    for k in range(n_dom):
        for f in range(n_frag):
            T[:, idx[f'MatFrag[{k}]'], idx[f'MatFragType[{k},{f}]']] = frag_weights[k, f]
            T[:, idx[f'InsFrag[{k}]'], idx[f'InsFragType[{k},{f}]']] = frag_weights[k, f]
            T[:, idx[f'DelFrag[{k}]'], idx[f'DelFragType[{k},{f}]']] = frag_weights[k, f]
            T[:, idx[f'IFrag[{k}]'], idx[f'IFragType[{k},{f}]']] = frag_weights[k, f]
            T[:, idx[f'DFrag[{k}]'], idx[f'DFragType[{k},{f}]']] = frag_weights[k, f]

    # Emission transitions and Emit -> FragEnd (weight 1, t-independent).
    for k in range(n_dom):
        for f in range(n_frag):
            T[:, idx[f'MatFragType[{k},{f}]'], idx[f'MatEmit[{k},{f}]']] = 1.0
            T[:, idx[f'InsFragType[{k},{f}]'], idx[f'InsEmit[{k},{f}]']] = 1.0
            T[:, idx[f'DelFragType[{k},{f}]'], idx[f'DelEmit[{k},{f}]']] = 1.0
            T[:, idx[f'IFragType[{k},{f}]'], idx[f'IEmit[{k},{f}]']] = 1.0
            T[:, idx[f'DFragType[{k},{f}]'], idx[f'DEmit[{k},{f}]']] = 1.0
            T[:, idx[f'MatEmit[{k},{f}]'], idx[f'MatFragEnd[{k},{f}]']] = 1.0
            T[:, idx[f'InsEmit[{k},{f}]'], idx[f'InsFragEnd[{k},{f}]']] = 1.0
            T[:, idx[f'DelEmit[{k},{f}]'], idx[f'DelFragEnd[{k},{f}]']] = 1.0
            T[:, idx[f'IEmit[{k},{f}]'], idx[f'IFragEnd[{k},{f}]']] = 1.0
            T[:, idx[f'DEmit[{k},{f}]'], idx[f'DFragEnd[{k},{f}]']] = 1.0

    # Fragment end: extension (f->g) or TKF transition out.
    for k in range(n_dom):
        for f in range(n_frag):
            notext = 1.0 - ext_rates[k, f, :].sum()  # scalar (t-independent)

            # Extension transitions (t-independent ext_rates)
            for g in range(n_frag):
                T[:, idx[f'MatFragEnd[{k},{f}]'], idx[f'MatFragType[{k},{g}]']] = ext_rates[k, f, g]
                T[:, idx[f'InsFragEnd[{k},{f}]'], idx[f'InsFragType[{k},{g}]']] = ext_rates[k, f, g]
                T[:, idx[f'DelFragEnd[{k},{f}]'], idx[f'DelFragType[{k},{g}]']] = ext_rates[k, f, g]
                T[:, idx[f'IFragEnd[{k},{f}]'], idx[f'IFragType[{k},{g}]']] = ext_rates[k, f, g]
                T[:, idx[f'DFragEnd[{k},{f}]'], idx[f'DFragType[{k},{g}]']] = ext_rates[k, f, g]

            # M-type MatFragEnd: TKF91_k M-row (t-dependent)
            T[:, idx[f'MatFragEnd[{k},{f}]'], idx[f'MatFrag[{k}]']] = notext * tau_k_b[:, k, M, M]
            T[:, idx[f'MatFragEnd[{k},{f}]'], idx[f'InsFrag[{k}]']] = notext * tau_k_b[:, k, M, I]
            T[:, idx[f'MatFragEnd[{k},{f}]'], idx[f'DelFrag[{k}]']] = notext * tau_k_b[:, k, M, D]
            T[:, idx[f'MatFragEnd[{k},{f}]'], idx['MatDomEnd']] = notext * tau_k_b[:, k, M, E]

            # InsFragEnd: TKF91_k I-row
            T[:, idx[f'InsFragEnd[{k},{f}]'], idx[f'MatFrag[{k}]']] = notext * tau_k_b[:, k, I, M]
            T[:, idx[f'InsFragEnd[{k},{f}]'], idx[f'InsFrag[{k}]']] = notext * tau_k_b[:, k, I, I]
            T[:, idx[f'InsFragEnd[{k},{f}]'], idx[f'DelFrag[{k}]']] = notext * tau_k_b[:, k, I, D]
            T[:, idx[f'InsFragEnd[{k},{f}]'], idx['MatDomEnd']] = notext * tau_k_b[:, k, I, E]

            # DelFragEnd: TKF91_k D-row
            T[:, idx[f'DelFragEnd[{k},{f}]'], idx[f'MatFrag[{k}]']] = notext * tau_k_b[:, k, D, M]
            T[:, idx[f'DelFragEnd[{k},{f}]'], idx[f'InsFrag[{k}]']] = notext * tau_k_b[:, k, D, I]
            T[:, idx[f'DelFragEnd[{k},{f}]'], idx[f'DelFrag[{k}]']] = notext * tau_k_b[:, k, D, D]
            T[:, idx[f'DelFragEnd[{k},{f}]'], idx['MatDomEnd']] = notext * tau_k_b[:, k, D, E]

            # I-type IFragEnd: kappa loop (t-independent)
            T[:, idx[f'IFragEnd[{k},{f}]'], idx[f'IFrag[{k}]']] = notext * kappa_k[k]
            T[:, idx[f'IFragEnd[{k},{f}]'], idx['InsDomEnd']] = notext * (1.0 - kappa_k[k])

            # D-type DFragEnd: kappa loop
            T[:, idx[f'DFragEnd[{k},{f}]'], idx[f'DFrag[{k}]']] = notext * kappa_k[k]
            T[:, idx[f'DFragEnd[{k},{f}]'], idx['DelDomEnd']] = notext * (1.0 - kappa_k[k])

    return T, emit_indices, null_indices, names, idx


# ============================================================
# Batched eliminate / restore.
# ============================================================


def _partition_batch(T, keep_idx, elim_idx):
    """Partition batched T (B, N, N) into the four block stacks."""
    keep_arr = np.asarray(keep_idx)
    elim_arr = np.asarray(elim_idx)
    T_KK = T[:, keep_arr[:, None], keep_arr[None, :]]
    T_KZ = T[:, keep_arr[:, None], elim_arr[None, :]]
    T_ZK = T[:, elim_arr[:, None], keep_arr[None, :]]
    T_ZZ = T[:, elim_arr[:, None], elim_arr[None, :]]
    return T_KK, T_KZ, T_ZK, T_ZZ


def _eliminate_batch(T, keep_idx, elim_idx):
    """Batched null-elimination. T is (B, N, N).

    Returns (T_reduced, C) with shapes (B, n_keep, n_keep), (B, n_elim, n_elim).
    """
    T_KK, T_KZ, T_ZK, T_ZZ = _partition_batch(T, keep_idx, elim_idx)
    n_elim = len(elim_idx)
    eye = np.eye(n_elim)[None]  # (1, n_elim, n_elim) broadcast
    C = np.linalg.inv(eye - T_ZZ)
    T_reduced = T_KK + T_KZ @ C @ T_ZK
    return T_reduced, C


def _restore_counts_general_batch(n_reduced, T, keep_idx, elim_idx, C):
    """Batched count restoration (per-pair).

    n_reduced : (B, n_keep, n_keep) — counts on the reduced model
    T         : (B, N, N) — pre-elimination transition matrix
    keep_idx  : list[int] — kept-state indices in T
    elim_idx  : list[int] — eliminated-state indices in T
    C         : (B, n_elim, n_elim) — null-closure matrix from `_eliminate_batch`

    Returns:
        n_full: (B, N, N) — counts on the pre-elimination model.

    Same ghost-usage formula as the scalar `_restore_counts_general`, but
    every block multiplication is over the leading B axis.
    """
    T_KK, T_KZ, T_ZK, T_ZZ = _partition_batch(T, keep_idx, elim_idx)
    B, N_full, _ = T.shape
    N_keep = len(keep_idx)
    N_elim = len(elim_idx)

    T_reduced = T_KK + T_KZ @ C @ T_ZK  # (B, N_keep, N_keep)

    # Element-wise n / chi with safe zero where chi=0.
    with np.errstate(divide='ignore', invalid='ignore'):
        Scale = np.where(np.abs(T_reduced) > 1e-30,
                         n_reduced / T_reduced, 0.0)

    # H[b, a, sj] = (C @ T_ZK)[b, a, sj] = expected flow from elim a to kept sj.
    H = C @ T_ZK  # (B, n_elim, n_keep)

    # n_KZ[b, si, a] = T_KZ[b, si, a] * (Scale @ H.T)[b, si, a]
    SH = Scale @ np.swapaxes(H, -1, -2)  # (B, n_keep, n_elim)
    n_KZ = T_KZ * SH  # (B, n_keep, n_elim)

    # n_ZK[b, a, sj] = T_ZK[b, a, sj] * (Ct.T @ Scale)[b, a, sj]
    Ct = T_KZ @ C  # (B, n_keep, n_elim)
    CtS = np.swapaxes(Ct, -1, -2) @ Scale  # (B, n_elim, n_keep)
    n_ZK = T_ZK * CtS  # (B, n_elim, n_keep)

    # n_ZZ[b, a, b'] = T_ZZ[b, a, b'] * (CtS @ H.T)[b, a, b']
    G_factor = CtS @ np.swapaxes(H, -1, -2)  # (B, n_elim, n_elim)
    n_ZZ = T_ZZ * G_factor

    n_KK = Scale * T_KK

    # Assemble full
    n_full = np.zeros((B, N_full, N_full))
    keep_arr = np.asarray(keep_idx)
    elim_arr = np.asarray(elim_idx)
    n_full[:, keep_arr[:, None], keep_arr[None, :]] = n_KK
    n_full[:, keep_arr[:, None], elim_arr[None, :]] = n_KZ
    n_full[:, elim_arr[:, None], keep_arr[None, :]] = n_ZK
    n_full[:, elim_arr[:, None], elim_arr[None, :]] = n_ZZ

    return n_full


# ============================================================
# Batched suff-stat extraction (per-pair).
# ============================================================


def _extract_suffstats_batch(n_full, idx, K, F):
    """Batched per-pair suff-stat extraction.

    n_full: (B, N_full, N_full) — restored counts per pair.
    idx:    str→int state-index map.
    K, F:   n_dom, n_frag (state-index map's structure).

    Returns dict with arrays whose leading axis is B (per-pair stacks).
    Layout matches `exact_suffstats` element-wise: e.g. result['top_5x5']
    has shape (B, 5, 5), result['dom_M_5x5'] is a list of K arrays each
    (B, 5, 5).
    """
    # Top-level TKF91 5×5
    top_5x5 = np.zeros((n_full.shape[0], 5, 5))
    dom_map = {'MatDom': M, 'InsDom': I, 'DelDom': D}
    de_map = {'MatDomEnd': M, 'InsDomEnd': I, 'DelDomEnd': D}

    for v_name, V in dom_map.items():
        top_5x5[:, S, V] += n_full[:, idx['Start'], idx[v_name]]
    top_5x5[:, S, E] += n_full[:, idx['Start'], idx['End']]

    for u_name, U in de_map.items():
        for v_name, V in dom_map.items():
            top_5x5[:, U, V] += n_full[:, idx[u_name], idx[v_name]]
        top_5x5[:, U, E] += n_full[:, idx[u_name], idx['End']]

    # Per-domain M-type TKF91 5×5
    dom_M_5x5 = [np.zeros((n_full.shape[0], 5, 5)) for _ in range(K)]
    for k in range(K):
        mdt = idx[f'MatDomType[{k}]']
        dom_M_5x5[k][:, S, M] = n_full[:, mdt, idx[f'MatFrag[{k}]']]
        dom_M_5x5[k][:, S, I] = n_full[:, mdt, idx[f'InsFrag[{k}]']]
        dom_M_5x5[k][:, S, D] = n_full[:, mdt, idx[f'DelFrag[{k}]']]
        dom_M_5x5[k][:, S, E] = n_full[:, mdt, idx['MatDomEnd']]
        for f in range(F):
            for x_name, X in [('MatFragEnd', M), ('InsFragEnd', I),
                               ('DelFragEnd', D)]:
                fe = idx[f'{x_name}[{k},{f}]']
                dom_M_5x5[k][:, X, M] += n_full[:, fe, idx[f'MatFrag[{k}]']]
                dom_M_5x5[k][:, X, I] += n_full[:, fe, idx[f'InsFrag[{k}]']]
                dom_M_5x5[k][:, X, D] += n_full[:, fe, idx[f'DelFrag[{k}]']]
                dom_M_5x5[k][:, X, E] += n_full[:, fe, idx['MatDomEnd']]

    # Per-domain kappa / 1-kappa
    dom_kappa = np.zeros((n_full.shape[0], K))
    dom_1mkappa = np.zeros((n_full.shape[0], K))
    for k in range(K):
        for dt_name, frag_name, de_name in [
            (f'InsDomType[{k}]', f'IFrag[{k}]', 'InsDomEnd'),
            (f'DelDomType[{k}]', f'DFrag[{k}]', 'DelDomEnd'),
        ]:
            dom_kappa[:, k] += n_full[:, idx[dt_name], idx[frag_name]]
            dom_1mkappa[:, k] += n_full[:, idx[dt_name], idx[de_name]]
        for f in range(F):
            for fe_name, frag_name, de_name in [
                (f'IFragEnd[{k},{f}]', f'IFrag[{k}]', 'InsDomEnd'),
                (f'DFragEnd[{k},{f}]', f'DFrag[{k}]', 'DelDomEnd'),
            ]:
                dom_kappa[:, k] += n_full[:, idx[fe_name], idx[frag_name]]
                dom_1mkappa[:, k] += n_full[:, idx[fe_name], idx[de_name]]

    # Extension / termination (MixDom2 (K, F, F))
    ext_counts = np.zeros((n_full.shape[0], K, F, F))
    term_counts = np.zeros((n_full.shape[0], K, F))
    for k in range(K):
        for f in range(F):
            for fe_prefix, ft_prefix in [
                ('MatFragEnd', 'MatFragType'),
                ('InsFragEnd', 'InsFragType'),
                ('DelFragEnd', 'DelFragType'),
                ('IFragEnd', 'IFragType'),
                ('DFragEnd', 'DFragType'),
            ]:
                fe = idx[f'{fe_prefix}[{k},{f}]']
                ext_total = np.zeros(n_full.shape[0])
                for g in range(F):
                    ft = idx[f'{ft_prefix}[{k},{g}]']
                    ext_counts[:, k, f, g] += n_full[:, fe, ft]
                    ext_total += n_full[:, fe, ft]
                term_counts[:, k, f] += n_full[:, fe, :].sum(axis=-1) - ext_total

    # Domain weights (sum of incoming to MatDomType, InsDomType, DelDomType)
    dom_w = np.zeros((n_full.shape[0], K))
    for k in range(K):
        for dt_prefix in ['MatDomType', 'InsDomType', 'DelDomType']:
            dom_w[:, k] += n_full[:, :, idx[f'{dt_prefix}[{k}]']].sum(axis=-1)

    # Fragment weights (Frag → FragType only, not FragEnd → FragType)
    frag_w = np.zeros((n_full.shape[0], K, F))
    for k in range(K):
        for f in range(F):
            for frag_name, ft_prefix in [
                (f'MatFrag[{k}]', 'MatFragType'),
                (f'InsFrag[{k}]', 'InsFragType'),
                (f'DelFrag[{k}]', 'DelFragType'),
                (f'IFrag[{k}]', 'IFragType'),
                (f'DFrag[{k}]', 'DFragType'),
            ]:
                frag_w[:, k, f] += n_full[:,
                                          idx[frag_name],
                                          idx[f'{ft_prefix}[{k},{f}]']]

    return {
        'top_5x5': top_5x5,
        'dom_M_5x5': dom_M_5x5,
        'dom_kappa': dom_kappa,
        'dom_1mkappa': dom_1mkappa,
        'ext': ext_counts,
        'term': term_counts,
        'dom_w': dom_w,
        'frag_w': frag_w,
    }


# ============================================================
# Top-level batched suffstats.
# ============================================================


# These are the elimination groups in chain order — same as the scalar path.
# Computed as a function of (K, F).
def _elim_groups(K, F):
    return [
        [f'{p}[{k},{f}]' for k in range(K) for f in range(F)
         for p in ['MatFragType', 'InsFragType', 'DelFragType',
                   'IFragType', 'DFragType']],
        [f'{p}[{k},{f}]' for k in range(K) for f in range(F)
         for p in ['MatFragEnd', 'InsFragEnd', 'DelFragEnd',
                   'IFragEnd', 'DFragEnd']],
        [f'{p}[{k}]' for k in range(K)
         for p in ['MatFrag', 'InsFrag', 'DelFrag', 'IFrag', 'DFrag']],
        [f'{p}[{k}]' for k in range(K)
         for p in ['MatDomType', 'InsDomType', 'DelDomType']],
        ['MatDom', 'InsDom', 'DelDom'],
        ['MatDomEnd', 'InsDomEnd', 'DelDomEnd'],
    ]


def exact_suffstats_batch(n_chi_stack, main_ins, main_del, t_array,
                           dom_ins, dom_del, dom_weights, frag_weights,
                           ext_rates):
    """Batched per-pair `exact_suffstats`.

    Equivalent to looping
        ss[b] = exact_suffstats(n_chi_stack[b], main_ins, main_del,
                                 t_array[b], dom_ins, dom_del, ...)
    but with all chain restoration done on (B, ...) stacks.

    Per-pair-t coherence: each batch element b uses its own (n_chi_stack[b],
    t_array[b]) — no representative-t shortcut, no averaging.

    Args:
        n_chi_stack: (B, N_collapsed, N_collapsed) per-pair collapsed counts.
        main_ins, main_del: scalars (frozen for batch).
        t_array:    (B,) per-pair evolutionary times.
        dom_ins, dom_del, dom_weights, frag_weights, ext_rates: shared params.

    Returns:
        Dict matching the scalar `exact_suffstats` layout but with leading
        B axes on every array (and dom_M_5x5 as a list of K arrays each
        (B, 5, 5) instead of K arrays each (5, 5)).
    """
    K = len(np.asarray(dom_ins))
    F = (np.asarray(frag_weights).shape[1]
         if hasattr(frag_weights, 'shape') else len(frag_weights[0]))
    B = int(np.asarray(t_array).shape[0])

    # Step 1: build T_full stack (B, N, N) and state-index data.
    T_full, _emit_idx, _null_idx, names, idx = build_fully_exploded_batch(
        main_ins, main_del, t_array, dom_ins, dom_del,
        dom_weights, frag_weights, ext_rates)

    # Step 2: 6 elimination steps.
    elim_groups = _elim_groups(K, F)
    steps = []  # list of (ke_idx_in_T, el_idx_in_T, C_stack, T_pre_stack)
    T = T_full
    nm = names
    for eg in elim_groups:
        m = {n: i for i, n in enumerate(nm)}
        el_local = sorted([m[n] for n in eg if n in m])
        ke_local = sorted([i for i in range(T.shape[-1])
                            if i not in el_local])
        Tr, C = _eliminate_batch(T, ke_local, el_local)
        steps.append((ke_local, el_local, C, T))
        T = Tr
        nm = [nm[i] for i in ke_local]

    # Step 3: 6 restoration steps (in reverse).
    n_current = np.asarray(n_chi_stack).copy()
    for ke_local, el_local, C, T_pre in reversed(steps):
        n_current = _restore_counts_general_batch(
            n_current, T_pre, ke_local, el_local, C)

    # Step 4: extract suff stats.
    return _extract_suffstats_batch(n_current, idx, K, F)
