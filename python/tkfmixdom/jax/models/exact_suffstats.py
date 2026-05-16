"""Exact sufficient statistics for MixDom Baum-Welch.

Uses the 6-step null elimination chain (fully_exploded.py +
elimination_steps.py) to compute exact BDI sufficient statistics.

The chain:
  1. Build fully exploded model (all null states explicit)
  2. Forward chain: 6 elimination steps → collapsed model
  3. Reverse chain: 6 restoration steps → exploded counts
  4. Extract suff stats by summing exploded counts per parameter group

Steps 1-5 have C=I (trivial). Step 6 has a 3×3 analytic closure.
Total cost: O(N²) where N = 5KF+2. Verified to give gradients
matching autodiff to 1e-14.
"""

import numpy as np
from ..core.params import S, M, I, D, E, tkf91_trans

from .fully_exploded import build_fully_exploded
from .elimination_steps import _eliminate, _restore_counts_general


# Compound state mappings
MM, MI, MD, II, DD = 0, 1, 2, 3, 4
_UV_U = np.array([M, M, M, I, D])
_UV_X = np.array([M, I, D, I, D])
_IS_M = np.array([True, True, True, False, False])


def mixdom_stats_from_collapsed_counts(n_chi, main_ins, main_del, t,
                                          dom_ins, dom_del, dom_weights,
                                          frag_weights, ext_rates):
    """Compute exact MixDom2 sufficient statistics via 6-step chain
    restoration of collapsed→exploded counts.

    This is the MIXDOM2-SPECIFIC chain-restoration path. Walks the
    elimination chain (forward) to collapse the fully-exploded model to
    the standard chi matrix, then reverses (restoration) to recover
    exploded transition counts from the collapsed n_chi sufficient
    statistics. Per-parameter-group counts (per-route TKF91 stats,
    classdist counts, ext counts, frag/dom weights) are extracted from
    the restored exploded counts.

    .. note:: MixDom-specific. For plain TKF92 (no domain hierarchy)
       use `tkf92_stats_from_counts` directly. For TKF91, use
       `tkf91_stats_from_counts`.

    Returns dict with per-parameter-group counts.
    """
    dom_ins = np.asarray(dom_ins)
    dom_del = np.asarray(dom_del)
    K = len(dom_ins)
    F = np.asarray(frag_weights).shape[1]

    # Step 1: Build fully exploded model
    T_full, emit_idx, null_idx, names, idx = build_fully_exploded(
        main_ins, main_del, t, dom_ins, dom_del,
        np.asarray(dom_weights), np.asarray(frag_weights), np.asarray(ext_rates))

    # Step 2: Forward chain (6 eliminations)
    def _do_elim(T, nm, elim_names):
        m = {n: i for i, n in enumerate(nm)}
        el = sorted([m[n] for n in elim_names if n in m])
        ke = sorted([i for i in range(T.shape[0]) if i not in el])
        Tr, C = _eliminate(T, ke, el)
        return Tr, [nm[i] for i in ke], ke, el, C, T

    elim_groups = [
        [f'{p}[{k},{f}]' for k in range(K) for f in range(F)
         for p in ['MatFragType', 'InsFragType', 'DelFragType', 'IFragType', 'DFragType']],
        [f'{p}[{k},{f}]' for k in range(K) for f in range(F)
         for p in ['MatFragEnd', 'InsFragEnd', 'DelFragEnd', 'IFragEnd', 'DFragEnd']],
        [f'{p}[{k}]' for k in range(K)
         for p in ['MatFrag', 'InsFrag', 'DelFrag', 'IFrag', 'DFrag']],
        [f'{p}[{k}]' for k in range(K)
         for p in ['MatDomType', 'InsDomType', 'DelDomType']],
        ['MatDom', 'InsDom', 'DelDom'],
        ['MatDomEnd', 'InsDomEnd', 'DelDomEnd'],
    ]

    steps = []
    T = T_full
    nm = names
    for eg in elim_groups:
        T, nm, ke, el, C, Tpre = _do_elim(T, nm, eg)
        steps.append((ke, el, C, Tpre))

    # Step 3: Reverse chain (6 restorations)
    n_current = np.asarray(n_chi).copy()
    for ke, el, C, Tpre in reversed(steps):
        n_current = _restore_counts_general(n_current, Tpre, ke, el, C)

    # Step 4: Extract suff stats from fully exploded counts
    return _extract_suffstats(n_current, names, idx, K, F,
                               main_ins, main_del, t,
                               dom_ins, dom_del, dom_weights,
                               frag_weights, ext_rates)


# Backwards-compat alias (90+ call sites; new code should call
# `mixdom_stats_from_collapsed_counts` directly).
exact_suffstats = mixdom_stats_from_collapsed_counts


def _extract_suffstats(n_full, names, idx, K, F,
                        main_ins, main_del, t,
                        dom_ins, dom_del, dom_weights,
                        frag_weights, ext_rates):
    """Extract per-parameter sufficient statistics from exploded counts.

    Each transition in the exploded model has a known parameter factor.
    We sum the restored counts by parameter group.
    """
    tau_0 = np.asarray(tkf91_trans(main_ins, main_del, t))
    tau_k = np.array([np.asarray(tkf91_trans(dom_ins[k], dom_del[k], t))
                      for k in range(K)])
    kappa = np.asarray(dom_ins) / np.asarray(dom_del)

    # Top-level TKF91 5×5 (from DomEnd transitions and Start)
    top_5x5 = np.zeros((5, 5))
    de_map = {'MatDomEnd': M, 'InsDomEnd': I, 'DelDomEnd': D}
    dom_map = {'MatDom': M, 'InsDom': I, 'DelDom': D}

    # Start → Dom
    for v_name, V in dom_map.items():
        top_5x5[S, V] += n_full[idx['Start'], idx[v_name]]
    top_5x5[S, E] += n_full[idx['Start'], idx['End']]

    # DomEnd → Dom and DomEnd → End
    for u_name, U in de_map.items():
        for v_name, V in dom_map.items():
            top_5x5[U, V] += n_full[idx[u_name], idx[v_name]]
        top_5x5[U, E] += n_full[idx[u_name], idx['End']]

    # Per-domain M-type TKF91 5×5
    dom_M_5x5 = [np.zeros((5, 5)) for _ in range(K)]
    for k in range(K):
        mdt = idx[f'MatDomType[{k}]']
        # S row: MatDomType → {MatFrag, InsFrag, DelFrag, MatDomEnd}
        dom_M_5x5[k][S, M] = n_full[mdt, idx[f'MatFrag[{k}]']]
        dom_M_5x5[k][S, I] = n_full[mdt, idx[f'InsFrag[{k}]']]
        dom_M_5x5[k][S, D] = n_full[mdt, idx[f'DelFrag[{k}]']]
        dom_M_5x5[k][S, E] = n_full[mdt, idx['MatDomEnd']]

        # Body rows: FragEnd → {Frag, DomEnd}
        for f in range(F):
            for x_name, X in [('MatFragEnd', M), ('InsFragEnd', I), ('DelFragEnd', D)]:
                fe = idx[f'{x_name}[{k},{f}]']
                dom_M_5x5[k][X, M] += n_full[fe, idx[f'MatFrag[{k}]']]
                dom_M_5x5[k][X, I] += n_full[fe, idx[f'InsFrag[{k}]']]
                dom_M_5x5[k][X, D] += n_full[fe, idx[f'DelFrag[{k}]']]
                dom_M_5x5[k][X, E] += n_full[fe, idx['MatDomEnd']]

    # Per-domain kappa/1-kappa (I/D-type)
    dom_kappa = np.zeros(K)
    dom_1mkappa = np.zeros(K)
    for k in range(K):
        for dt_name, frag_name, de_name in [
            (f'InsDomType[{k}]', f'IFrag[{k}]', 'InsDomEnd'),
            (f'DelDomType[{k}]', f'DFrag[{k}]', 'DelDomEnd'),
        ]:
            dom_kappa[k] += n_full[idx[dt_name], idx[frag_name]]
            dom_1mkappa[k] += n_full[idx[dt_name], idx[de_name]]

        for f in range(F):
            for fe_name, frag_name, de_name in [
                (f'IFragEnd[{k},{f}]', f'IFrag[{k}]', 'InsDomEnd'),
                (f'DFragEnd[{k},{f}]', f'DFrag[{k}]', 'DelDomEnd'),
            ]:
                dom_kappa[k] += n_full[idx[fe_name], idx[frag_name]]
                dom_1mkappa[k] += n_full[idx[fe_name], idx[de_name]]

    # Extension / termination: MixDom2 produces (K, F, F) transition counts
    # ext_counts[k,f,g] = expected fragment f -> g transition count in domain k
    # term_counts[k,f] = expected fragment f termination count in domain k
    ext_counts = np.zeros((K, F, F))
    term_counts = np.zeros((K, F))
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
                # Extension counts: f -> g for each destination fragment g
                ext_total = 0.0
                for g in range(F):
                    ft = idx[f'{ft_prefix}[{k},{g}]']
                    ext_counts[k, f, g] += n_full[fe, ft]
                    ext_total += n_full[fe, ft]
                # Termination = all FragEnd outgoing except extension
                term_counts[k, f] += n_full[fe, :].sum() - ext_total

    # Domain weights
    dom_w = np.zeros(K)
    for k in range(K):
        for dt_prefix in ['MatDomType', 'InsDomType', 'DelDomType']:
            dom_w[k] += n_full[:, idx[f'{dt_prefix}[{k}]']].sum()

    # Fragment weights: only Frag → FragType transitions (not FragEnd → FragType,
    # which is the extension self-loop and belongs to ext counts)
    frag_w = np.zeros((K, F))
    for k in range(K):
        for f in range(F):
            for frag_name, ft_prefix in [
                (f'MatFrag[{k}]', 'MatFragType'),
                (f'InsFrag[{k}]', 'InsFragType'),
                (f'DelFrag[{k}]', 'DelFragType'),
                (f'IFrag[{k}]', 'IFragType'),
                (f'DFrag[{k}]', 'DFragType'),
            ]:
                frag_w[k, f] += n_full[idx[frag_name], idx[f'{ft_prefix}[{k},{f}]']]

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


def zero_suffstats(K, F):
    """Build a zero-initialized suff-stats accumulator with the same shape /
    dtype layout that `exact_suffstats` returns.

    Use this when accumulating per-pair exact suff stats across a batch.
    """
    return {
        'top_5x5': np.zeros((5, 5)),
        'dom_M_5x5': [np.zeros((5, 5)) for _ in range(K)],
        'dom_kappa': np.zeros(K),
        'dom_1mkappa': np.zeros(K),
        'ext': np.zeros((K, F, F)),
        'term': np.zeros((K, F)),
        'dom_w': np.zeros(K),
        'frag_w': np.zeros((K, F)),
    }


def add_suffstats(acc, ss):
    """In-place add per-pair suff stats `ss` into accumulator `acc`.

    Both must come from `exact_suffstats` / `zero_suffstats` (same layout).
    Mutates `acc`. Returns `acc` for chainability.
    """
    acc['top_5x5'] += ss['top_5x5']
    for k in range(len(acc['dom_M_5x5'])):
        acc['dom_M_5x5'][k] += ss['dom_M_5x5'][k]
    acc['dom_kappa'] += ss['dom_kappa']
    acc['dom_1mkappa'] += ss['dom_1mkappa']
    acc['ext'] += ss['ext']
    acc['term'] += ss['term']
    acc['dom_w'] += ss['dom_w']
    acc['frag_w'] += ss['frag_w']
    return acc


def exact_suffstats_per_pair_batch(per_pair_records, main_ins, main_del,
                                    dom_ins, dom_del, dom_weights,
                                    frag_weights, ext_rates,
                                    accumulate_bdi=True):
    """Run exact_suffstats per-pair and sum into a single accumulator.

    Args:
        per_pair_records: iterable of (n_chi_p, t_p) tuples.
        main_ins, main_del, ...: shared model parameters (frozen during E-step).
        accumulate_bdi: if True, also accumulate per-pair POST-DIVIDE BDI
            sufficient statistics (E[B]_p, E[D]_p, E[S]_p) at top level and
            per domain. These quantities (rather than the upstream pre-divide
            score scalars) are what the downstream EMA / M-step consumes.
            Pushing the (λ-μ) divide INSIDE the per-pair loop ensures that
            BDI is evaluated at the same (λ_K, μ_K) used for that pair's
            score evaluation; the EMA then operates on natural sufficient
            statistics, eliminating cross-iteration θ-staleness.

            Adds the following keys to the output:
                top_E_B, top_E_D, top_E_S, top_T_obs (scalars)
                dom_E_B (K,), dom_E_D (K,), dom_E_S (K,), dom_T_obs (K,)

    Returns:
        Dict with the same layout as `exact_suffstats` but containing the
        sum across all pairs (and, if `accumulate_bdi`, also the
        post-divide BDI sufficient-stat sums).
    """
    from ..core.params import S, M, I, D, E
    from ..core.bdi import bdi_stats_from_counts_batch
    from .exact_suffstats_batch import exact_suffstats_batch
    K = len(np.asarray(dom_ins))
    F = np.asarray(frag_weights).shape[1]
    acc = zero_suffstats(K, F)

    if accumulate_bdi:
        main_ins_f = float(main_ins)
        main_del_f = float(main_del)
        dom_ins_arr = np.asarray(dom_ins, dtype=float)
        dom_del_arr = np.asarray(dom_del, dtype=float)

    if not per_pair_records:
        if accumulate_bdi:
            acc['top_E_B'] = 0.0
            acc['top_E_D'] = 0.0
            acc['top_E_S'] = 0.0
            acc['top_T_obs'] = 0.0
            acc['dom_E_B'] = np.zeros(K)
            acc['dom_E_D'] = np.zeros(K)
            acc['dom_E_S'] = np.zeros(K)
            acc['dom_T_obs'] = np.zeros(K)
        return acc

    # Phase 1: stack per-pair (n_chi_p, t_p) and run the chain restoration
    # in one vectorised numpy call. This replaces the Python `for n_chi_p,
    # t_p in per_pair_records:` loop over `exact_suffstats` (≈ 65 ms/pair
    # at d3f1, dominated by the per-pair `build_fully_exploded` Python
    # interpretation cost — many idx[name] dict lookups). Each batch
    # element b uses its own (n_chi_stack[b], t_arr[b]); shared params are
    # the same scalars used in the serial path. Per-pair-t coherence is
    # preserved bit-for-bit (verified ≤ 1e-12 abs diff vs serial).
    n_chi_stack = np.stack([np.asarray(rec[0]) for rec in per_pair_records])
    t_arr = np.array([float(rec[1]) for rec in per_pair_records], dtype=float)
    ss_batch = exact_suffstats_batch(
        n_chi_stack, main_ins, main_del, t_arr,
        dom_ins, dom_del, dom_weights, frag_weights, ext_rates)

    # Sum per-pair stats into acc. ss_batch arrays have leading axis B.
    acc['top_5x5'] += ss_batch['top_5x5'].sum(axis=0)
    for k in range(K):
        acc['dom_M_5x5'][k] += ss_batch['dom_M_5x5'][k].sum(axis=0)
    acc['dom_kappa'] += ss_batch['dom_kappa'].sum(axis=0)
    acc['dom_1mkappa'] += ss_batch['dom_1mkappa'].sum(axis=0)
    acc['ext'] += ss_batch['ext'].sum(axis=0)
    acc['term'] += ss_batch['term'].sum(axis=0)
    acc['dom_w'] += ss_batch['dom_w'].sum(axis=0)
    acc['frag_w'] += ss_batch['frag_w'].sum(axis=0)

    if accumulate_bdi:
        # Phase 2: batched per-pair-t BDI sufficient-stat conversion.
        # 1 + K vmapped JAX calls (top-level + per-domain), each operating
        # on the full per-pair stack. Per-pair-t coherence preserved.
        top_stack = ss_batch['top_5x5']                       # (B, 5, 5)
        dom_stack = np.stack(ss_batch['dom_M_5x5'], axis=1)   # (B, K, 5, 5)
        n1k_stack = ss_batch['dom_1mkappa']                   # (B, K)
        B_proc = t_arr.shape[0]

        # Top-level: M_p = sum(top_5x5_p[:, E]); T_p = t_p · M_p.
        top_M = top_stack[:, :, E].sum(axis=1)
        top_T = t_arr * top_M
        EB_top, ED_top, ES_top = bdi_stats_from_counts_batch(
            top_stack, main_ins_f, main_del_f, t_arr, T_batch=top_T)

        acc['top_E_B'] = float(EB_top.sum())
        acc['top_E_D'] = float(ED_top.sum())
        acc['top_E_S'] = float(ES_top.sum())
        acc['top_T_obs'] = float(top_T.sum())

        # Per-domain: T_d_p = t_p · (n_entries_M_d_p + n1k_ID_d_p).
        n_entries_M = dom_stack[:, :, S, :].sum(axis=2)  # (B, K)
        T_d_stack = t_arr[:, None] * (n_entries_M + n1k_stack)  # (B, K)

        dom_E_B_total = np.zeros(K)
        dom_E_D_total = np.zeros(K)
        dom_E_S_total = np.zeros(K)
        dom_T_total = np.zeros(K)
        for d in range(K):
            EB_d, ED_d, ES_d = bdi_stats_from_counts_batch(
                dom_stack[:, d], float(dom_ins_arr[d]),
                float(dom_del_arr[d]), t_arr,
                T_batch=T_d_stack[:, d])
            dom_E_B_total[d] = float(EB_d.sum())
            dom_E_D_total[d] = float(ED_d.sum())
            dom_E_S_total[d] = float(ES_d.sum())
            dom_T_total[d] = float(T_d_stack[:, d].sum())

        acc['dom_E_B'] = dom_E_B_total
        acc['dom_E_D'] = dom_E_D_total
        acc['dom_E_S'] = dom_E_S_total
        acc['dom_T_obs'] = dom_T_total

    return acc
