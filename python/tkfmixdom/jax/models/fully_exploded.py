"""Fully exploded MixDom Pair HMM with ALL null states explicit.

This constructs the complete exploded model from ExplodedMixDom.txt,
with null states for domain type selection, fragment type selection,
fragment end, domain end, etc. The null elimination of this model
recovers the collapsed 5N+2 state MixDom Pair HMM exactly.

The primary purpose is to enable EXACT count restoration: given FB
expected transition counts on the collapsed model, restore counts
on the exploded model by inverting the null elimination. Each exploded
transition depends on exactly one parameter group, enabling exact
closed-form BDI M-steps at every level.
"""

import numpy as np
import jax.numpy as jnp
from ..core.params import S, M, I, D, E, tkf91_trans
from ..dp.hmm import NEG_INF


def build_fully_exploded(main_ins_rate, main_del_rate, t,
                          dom_ins_rates, dom_del_rates, dom_weights,
                          frag_weights, ext_rates):
    """Build the fully exploded MixDom transition matrix.

    States (all non-emitting except Emit states):
      0: Start
      1: End
      2: MatDom, 3: InsDom, 4: DelDom
      5+3k .. 5+3k+2: MatDomType[k], InsDomType[k], DelDomType[k]
      Then for each domain k, fragment-level states...
      Then Emit states (the only emitting states)
      Then FragEnd states
      Then DomEnd states: MatDomEnd, InsDomEnd, DelDomEnd

    Returns:
        T: (N_full, N_full) transition matrix
        emit_indices: list of indices of emitting states
        null_indices: list of indices of null states
        state_names: list of string names for each state
        param_tags: dict mapping (src, dst) -> parameter tag string
    """
    n_dom = len(dom_ins_rates)
    n_frag = frag_weights.shape[1] if hasattr(frag_weights, 'shape') else len(frag_weights[0])
    dom_ins_rates = np.asarray(dom_ins_rates)
    dom_del_rates = np.asarray(dom_del_rates)
    dom_weights = np.asarray(dom_weights)
    frag_weights = np.asarray(frag_weights)
    ext_rates = np.asarray(ext_rates)

    # BDI parameters
    tau_0 = np.asarray(tkf91_trans(main_ins_rate, main_del_rate, t))
    tau_k = np.array([np.asarray(tkf91_trans(dom_ins_rates[k], dom_del_rates[k], t))
                      for k in range(n_dom)])
    kappa_k = dom_ins_rates / dom_del_rates

    # Build state index map
    names = []
    idx = {}

    def add(name):
        i = len(names)
        names.append(name)
        idx[name] = i
        return i

    # Fixed states
    add('Start')
    add('End')
    add('MatDom')
    add('InsDom')
    add('DelDom')

    # Per-domain type selection states
    for k in range(n_dom):
        add(f'MatDomType[{k}]')
        add(f'InsDomType[{k}]')
        add(f'DelDomType[{k}]')

    # Per-domain fragment-level states (M-type domains have 3 frag states)
    for k in range(n_dom):
        add(f'MatFrag[{k}]')
        add(f'InsFrag[{k}]')
        add(f'DelFrag[{k}]')
        add(f'IFrag[{k}]')   # single frag for InsDom
        add(f'DFrag[{k}]')   # single frag for DelDom

    # Per-domain, per-fragment type selection
    for k in range(n_dom):
        for f in range(n_frag):
            add(f'MatFragType[{k},{f}]')
            add(f'InsFragType[{k},{f}]')
            add(f'DelFragType[{k},{f}]')
            add(f'IFragType[{k},{f}]')
            add(f'DFragType[{k},{f}]')

    # Emit states (emitting!) — ordered to match collapsed model:
    # state_index(uv, dom, frag, n_frag) = 2 + uv*n_frag + dom*5*n_frag + frag
    # So: domain outermost, then compound type (uv), then fragment innermost.
    # uv: MM=0 (MatEmit), MI=1 (InsEmit), MD=2 (DelEmit), II=3 (IEmit), DD=4 (DEmit)
    emit_indices = []
    _emit_names = ['MatEmit', 'InsEmit', 'DelEmit', 'IEmit', 'DEmit']
    for k in range(n_dom):
        for uv_idx, emit_prefix in enumerate(_emit_names):
            for f in range(n_frag):
                emit_indices.append(add(f'{emit_prefix}[{k},{f}]'))

    # Fragment end states
    for k in range(n_dom):
        for f in range(n_frag):
            add(f'MatFragEnd[{k},{f}]')
            add(f'InsFragEnd[{k},{f}]')
            add(f'DelFragEnd[{k},{f}]')
            add(f'IFragEnd[{k},{f}]')
            add(f'DFragEnd[{k},{f}]')

    # Domain end states
    add('MatDomEnd')
    add('InsDomEnd')
    add('DelDomEnd')

    N = len(names)
    T = np.zeros((N, N))
    null_indices = [i for i in range(N) if i not in emit_indices]

    # Helper: top-level TKF row (S/M/I rows use beta, D row uses gamma)
    def top_row(src, is_del=False):
        bg = tau_0[D, :] if is_del else tau_0[S, :]  # gamma vs beta row
        T[idx[src], idx['MatDom']] = bg[M]      # (1-β)κα or (1-γ)κα
        T[idx[src], idx['InsDom']] = bg[I]       # β or γ
        T[idx[src], idx['DelDom']] = bg[D]       # (1-β)κ(1-α) or (1-γ)κ(1-α)
        T[idx[src], idx['End']] = bg[E]          # (1-β)(1-κ) or (1-γ)(1-κ)

    # Top-level transitions
    top_row('Start')
    top_row('MatDomEnd')
    top_row('InsDomEnd')
    top_row('DelDomEnd', is_del=True)

    # Domain type selection
    for k in range(n_dom):
        T[idx['MatDom'], idx[f'MatDomType[{k}]']] = dom_weights[k]
        T[idx['InsDom'], idx[f'InsDomType[{k}]']] = dom_weights[k]
        T[idx['DelDom'], idx[f'DelDomType[{k}]']] = dom_weights[k]

    # M-type domain entry (TKF91 first transition)
    for k in range(n_dom):
        T[idx[f'MatDomType[{k}]'], idx[f'MatFrag[{k}]']] = tau_k[k][S, M]
        T[idx[f'MatDomType[{k}]'], idx[f'InsFrag[{k}]']] = tau_k[k][S, I]
        T[idx[f'MatDomType[{k}]'], idx[f'DelFrag[{k}]']] = tau_k[k][S, D]
        T[idx[f'MatDomType[{k}]'], idx['MatDomEnd']] = tau_k[k][S, E]  # phantom

    # I/D-type domain entry (kappa loop)
    for k in range(n_dom):
        T[idx[f'InsDomType[{k}]'], idx[f'IFrag[{k}]']] = kappa_k[k]
        T[idx[f'InsDomType[{k}]'], idx['InsDomEnd']] = 1.0 - kappa_k[k]
        T[idx[f'DelDomType[{k}]'], idx[f'DFrag[{k}]']] = kappa_k[k]
        T[idx[f'DelDomType[{k}]'], idx['DelDomEnd']] = 1.0 - kappa_k[k]

    # Fragment type selection
    for k in range(n_dom):
        for f in range(n_frag):
            T[idx[f'MatFrag[{k}]'], idx[f'MatFragType[{k},{f}]']] = frag_weights[k, f]
            T[idx[f'InsFrag[{k}]'], idx[f'InsFragType[{k},{f}]']] = frag_weights[k, f]
            T[idx[f'DelFrag[{k}]'], idx[f'DelFragType[{k},{f}]']] = frag_weights[k, f]
            T[idx[f'IFrag[{k}]'], idx[f'IFragType[{k},{f}]']] = frag_weights[k, f]
            T[idx[f'DFrag[{k}]'], idx[f'DFragType[{k},{f}]']] = frag_weights[k, f]

    # Emission transitions (weight 1)
    for k in range(n_dom):
        for f in range(n_frag):
            T[idx[f'MatFragType[{k},{f}]'], idx[f'MatEmit[{k},{f}]']] = 1.0
            T[idx[f'InsFragType[{k},{f}]'], idx[f'InsEmit[{k},{f}]']] = 1.0
            T[idx[f'DelFragType[{k},{f}]'], idx[f'DelEmit[{k},{f}]']] = 1.0
            T[idx[f'IFragType[{k},{f}]'], idx[f'IEmit[{k},{f}]']] = 1.0
            T[idx[f'DFragType[{k},{f}]'], idx[f'DEmit[{k},{f}]']] = 1.0

    # Emit -> FragEnd (weight 1)
    for k in range(n_dom):
        for f in range(n_frag):
            T[idx[f'MatEmit[{k},{f}]'], idx[f'MatFragEnd[{k},{f}]']] = 1.0
            T[idx[f'InsEmit[{k},{f}]'], idx[f'InsFragEnd[{k},{f}]']] = 1.0
            T[idx[f'DelEmit[{k},{f}]'], idx[f'DelFragEnd[{k},{f}]']] = 1.0
            T[idx[f'IEmit[{k},{f}]'], idx[f'IFragEnd[{k},{f}]']] = 1.0
            T[idx[f'DEmit[{k},{f}]'], idx[f'DFragEnd[{k},{f}]']] = 1.0

    # Auto-convert MixDom1 ext_rates (D, F) -> MixDom2 (D, F, F)
    if ext_rates.ndim == 2:
        ext_rates_3d = np.zeros((n_dom, n_frag, n_frag))
        for d in range(n_dom):
            ext_rates_3d[d] = np.diag(ext_rates[d])
        ext_rates = ext_rates_3d

    # Fragment end: extension (f->g transitions) or TKF transition
    for k in range(n_dom):
        for f in range(n_frag):
            notext = 1.0 - ext_rates[k, f, :].sum()  # 1 - sum_g ext[k,f,g]

            # M-type MatFragEnd: extension transitions f->g, or TKF91_k
            for g in range(n_frag):
                T[idx[f'MatFragEnd[{k},{f}]'], idx[f'MatFragType[{k},{g}]']] = ext_rates[k, f, g]
            T[idx[f'MatFragEnd[{k},{f}]'], idx[f'MatFrag[{k}]']] = notext * tau_k[k][M, M]
            T[idx[f'MatFragEnd[{k},{f}]'], idx[f'InsFrag[{k}]']] = notext * tau_k[k][M, I]
            T[idx[f'MatFragEnd[{k},{f}]'], idx[f'DelFrag[{k}]']] = notext * tau_k[k][M, D]
            T[idx[f'MatFragEnd[{k},{f}]'], idx['MatDomEnd']] = notext * tau_k[k][M, E]

            # InsFragEnd (beta row)
            for g in range(n_frag):
                T[idx[f'InsFragEnd[{k},{f}]'], idx[f'InsFragType[{k},{g}]']] = ext_rates[k, f, g]
            T[idx[f'InsFragEnd[{k},{f}]'], idx[f'MatFrag[{k}]']] = notext * tau_k[k][I, M]
            T[idx[f'InsFragEnd[{k},{f}]'], idx[f'InsFrag[{k}]']] = notext * tau_k[k][I, I]
            T[idx[f'InsFragEnd[{k},{f}]'], idx[f'DelFrag[{k}]']] = notext * tau_k[k][I, D]
            T[idx[f'InsFragEnd[{k},{f}]'], idx['MatDomEnd']] = notext * tau_k[k][I, E]

            # DelFragEnd (gamma row)
            for g in range(n_frag):
                T[idx[f'DelFragEnd[{k},{f}]'], idx[f'DelFragType[{k},{g}]']] = ext_rates[k, f, g]
            T[idx[f'DelFragEnd[{k},{f}]'], idx[f'MatFrag[{k}]']] = notext * tau_k[k][D, M]
            T[idx[f'DelFragEnd[{k},{f}]'], idx[f'InsFrag[{k}]']] = notext * tau_k[k][D, I]
            T[idx[f'DelFragEnd[{k},{f}]'], idx[f'DelFrag[{k}]']] = notext * tau_k[k][D, D]
            T[idx[f'DelFragEnd[{k},{f}]'], idx['MatDomEnd']] = notext * tau_k[k][D, E]

            # I-type IFragEnd: extension f->g or kappa loop
            for g in range(n_frag):
                T[idx[f'IFragEnd[{k},{f}]'], idx[f'IFragType[{k},{g}]']] = ext_rates[k, f, g]
            T[idx[f'IFragEnd[{k},{f}]'], idx[f'IFrag[{k}]']] = notext * kappa_k[k]
            T[idx[f'IFragEnd[{k},{f}]'], idx['InsDomEnd']] = notext * (1-kappa_k[k])

            # D-type DFragEnd: extension f->g or kappa loop
            for g in range(n_frag):
                T[idx[f'DFragEnd[{k},{f}]'], idx[f'DFragType[{k},{g}]']] = ext_rates[k, f, g]
            T[idx[f'DFragEnd[{k},{f}]'], idx[f'DFrag[{k}]']] = notext * kappa_k[k]
            T[idx[f'DFragEnd[{k},{f}]'], idx['DelDomEnd']] = notext * (1-kappa_k[k])

    return T, emit_indices, null_indices, names, idx


def null_eliminate(T, emit_indices, null_indices):
    """Eliminate null states to get the collapsed transition matrix.

    chi = T_EN (I - T_NN)^{-1} T_NE + T_EE

    where E = emit + Start + End, N = null states.
    But Start and End are not emitting — they are kept as visible states.
    In the collapsed model, Start = state 0, End = state 1, then body states.

    Actually, Start and End ARE kept states (not eliminated). So:
    kept = {Start, End} ∪ {emit states}
    eliminated = null states ∖ {Start, End}

    But Start (0) and End (1) are in null_indices. We need to keep them.
    Let's separate: keep_indices = [Start, End] + emit_indices,
    elim_indices = null_indices ∖ {Start, End}.

    Wait — Start and End are non-emitting but are NOT eliminated. They are
    boundary states of the collapsed model. So:
    """
    keep = sorted([0, 1] + list(emit_indices))  # Start, End, emit states
    elim = sorted([i for i in null_indices if i not in (0, 1)])

    T_KK = T[np.ix_(keep, keep)]
    T_KE = T[np.ix_(keep, elim)]
    T_EK = T[np.ix_(elim, keep)]
    T_EE = T[np.ix_(elim, elim)]

    # Null closure
    C = np.linalg.inv(np.eye(len(elim)) - T_EE)

    chi = T_KK + T_KE @ C @ T_EK
    return chi, keep, elim, C


def restore_counts(n_chi, T, emit_indices, null_indices):
    """Restore expected counts on the exploded model from collapsed counts.

    Given n_chi on the collapsed model (keep states), returns n_full on
    the full exploded model.
    """
    keep = sorted([0, 1] + list(emit_indices))
    elim = sorted([i for i in null_indices if i not in (0, 1)])

    T_KK = T[np.ix_(keep, keep)]
    T_KE = T[np.ix_(keep, elim)]
    T_EK = T[np.ix_(elim, keep)]
    T_EE = T[np.ix_(elim, elim)]

    C = np.linalg.inv(np.eye(len(elim)) - T_EE)
    chi = T_KK + T_KE @ C @ T_EK

    N_full = T.shape[0]
    N_keep = len(keep)
    N_elim = len(elim)

    n_full = np.zeros((N_full, N_full))

    # For each kept transition (s, s'), distribute count to exploded paths
    for si, s in enumerate(keep):
        for spi, sp in enumerate(keep):
            nc = n_chi[si, spi]
            if abs(nc) < 1e-30 or abs(chi[si, spi]) < 1e-30:
                continue

            scale = nc / chi[si, spi]

            # Direct path: s -> s' (if T[s, s'] > 0)
            n_full[s, sp] += scale * T_KK[si, spi]

            # Null-mediated paths: s -> elim chain -> s'
            # Count for s -> first null state a:
            # n_full[s, a] += scale * T_KE[si, a_idx] * (C @ T_EK[:, spi])[a_idx]
            # ... but we need per-transition counts, not per-state.

            # For each null transition (a -> b) where a, b ∈ elim:
            # Expected count = scale * (path through a,b)
            # = scale * T_KE[si, :] diag-weights through C

            # Use the Doob h-transform approach:
            # h[a] = (C @ T_EK[:, spi])[a_idx] = expected flow from a to s'
            h = C @ T_EK[:, spi]  # (N_elim,)

            # Entry: s -> a (for each null state a)
            for ai in range(N_elim):
                a = elim[ai]
                if T_KE[si, ai] < 1e-30 or h[ai] < 1e-30:
                    continue
                n_s_a = scale * T_KE[si, ai] * h[ai]
                n_full[s, a] += n_s_a

                # Interior null transitions: a -> b (within elim)
                for bi in range(N_elim):
                    b = elim[bi]
                    if T_EE[ai, bi] < 1e-30 or h[bi] < 1e-30:
                        continue
                    # Expected count of a->b conditional on entering at a and exiting to s'
                    n_full[a, b] += n_s_a * C[ai, ai] * T_EE[ai, bi] * h[bi] / h[ai]
                    # Wait, this double-counts. Need the proper formula.
                    # Actually: n(a->b) = entry_weight * C[entry, a] * T[a,b] * h[b] / h[entry]
                    pass

                # Exit: last null state -> s'
                for bi in range(N_elim):
                    b = elim[bi]
                    if T_EK[bi, spi] < 1e-30:
                        continue
                    n_full[b, sp] += n_s_a * C[ai, bi] * T_EK[bi, spi] / h[ai]

            # Interior null-null transitions using the ghost-usage formula:
            # n_full(a, b) for a, b ∈ elim, summed over all (s, s') paths
            # This is equation (eq:ghost-hmm) from ghost-usage.tex

    # Redo interior null transitions using the compact formula from ghost-usage.tex:
    # G[a,b] = Σ_{s,s'} (n_chi[s,s']/chi[s,s']) * diag(C^T t_s) * T_EE * diag(C t_{s'})
    # where t_s = T_KE[s, :] and t_{s'} = T_EK[:, s']

    # Clear previous null-null counts (we'll recompute)
    for ai in range(N_elim):
        for bi in range(N_elim):
            n_full[elim[ai], elim[bi]] = 0.0

    for si, s in enumerate(keep):
        for spi, sp in enumerate(keep):
            nc = n_chi[si, spi]
            if abs(nc) < 1e-30 or abs(chi[si, spi]) < 1e-30:
                continue
            scale = nc / chi[si, spi]

            t_s = T_KE[si, :]    # (N_elim,) row: s -> null states
            t_sp = T_EK[:, spi]  # (N_elim,) col: null states -> s'

            # h[a] = (C @ t_sp)[a] = expected flow to s' from null state a
            h = C @ t_sp

            for ai in range(N_elim):
                if t_s[ai] < 1e-30:
                    continue
                # Entry s -> a
                entry_a = scale * t_s[ai] * h[ai]

                # a -> b for each null transition
                for bi in range(N_elim):
                    if T_EE[ai, bi] < 1e-30 or h[bi] < 1e-30:
                        continue
                    # Doob h-transform: conditional on entering chain at a,
                    # expected count of transition a->b is
                    # C[a,a] * T[a,b] * h[b] / h[a]  ... no, that's wrong too.
                    # The correct formula for expected count of transition a->b
                    # in a chain that starts at entry a and ends at exit to s':
                    # sum over all visits to a, times T[a,b], times prob of
                    # eventually reaching s' from b.
                    # = C[entry, a] * T[a,b] * h[b] / h[entry]
                    # But we sum over all entry points.
                    pass

    # Actually, let me just use the matrix formula from ghost-usage.tex.
    # It's cleaner.

    # Reset all null transition counts
    for ai in range(N_elim):
        for bi in range(N_elim):
            n_full[elim[ai], elim[bi]] = 0.0
    # Reset keep->null and null->keep counts too
    for si_idx, s in enumerate(keep):
        for ai in range(N_elim):
            a = elim[ai]
            n_full[s, a] = 0.0
            n_full[a, s] = 0.0

    # Recompute ALL exploded counts using the general formula:
    # For each (s, s') in kept:
    #   chi[s,s'] = T_KK[s,s'] + sum_a sum_b T_KE[s,a] * C[a,b] * T_EK[b,s']
    #   For the direct part: n_full[s,s'] = n_chi[s,s'] * T_KK[s,s'] / chi[s,s']
    #   For each null-mediated path entering at a and exiting at b:
    #     n_enter(s,a) += n_chi[s,s'] * T_KE[s,a] * C[a,:] @ T_EK[:,s'] / chi[s,s']
    #     n_exit(b,s') += n_chi[s,s'] * T_KE[s,:] @ C[:,b] * T_EK[b,s'] / chi[s,s']
    #     For null-null (a->b): use ghost count matrix

    for si, s in enumerate(keep):
        for spi, sp in enumerate(keep):
            nc = n_chi[si, spi]
            if abs(nc) < 1e-30 or abs(chi[si, spi]) < 1e-30:
                continue
            scale = nc / chi[si, spi]

            # Direct
            n_full[s, sp] += scale * T_KK[si, spi]

            # Null-mediated: s -> null chain -> s'
            t_s = T_KE[si, :]
            t_sp = T_EK[:, spi]
            h = C @ t_sp  # (N_elim,)

            # Keep -> null entries
            for ai in range(N_elim):
                a = elim[ai]
                n_full[s, a] += scale * t_s[ai] * h[ai]

            # Null -> keep exits
            Ct_s = C.T @ t_s  # (N_elim,): expected visits weighted by entry
            for bi in range(N_elim):
                b = elim[bi]
                n_full[b, sp] += scale * Ct_s[bi] * t_sp[bi]

            # Null -> null: ghost count matrix (eq:ghost-hmm)
            # G[a,b] = scale * diag(C^T t_s) @ T_EE @ diag(C t_{s'})
            # But this gives a matrix for each (s,s'). Sum over all (s,s').
            Ct_s_diag = Ct_s  # C^T @ t_s
            h_diag = h         # C @ t_sp
            for ai in range(N_elim):
                a = elim[ai]
                if Ct_s_diag[ai] < 1e-30:
                    continue
                for bi in range(N_elim):
                    b = elim[bi]
                    if T_EE[ai, bi] < 1e-30 or h_diag[bi] < 1e-30:
                        continue
                    n_full[a, b] += scale * Ct_s_diag[ai] * T_EE[ai, bi] * h_diag[bi]

    return n_full, keep, elim
