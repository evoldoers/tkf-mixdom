"""Reference implementations via grammar elaboration with explicit null states.

Constructs TKF92 (and later MixDom) by building a null-rich Pair HMM from
atomic TKF91 pieces, then eliminating null states via matrix inversion.

These are NOT the production implementations — they exist to validate the
closed-form transition matrices and null count restoration against a
systematically constructed reference.

TKF92 elaboration:
    States: S, M, I, D, E  (visible, same as TKF91)
            M_end, I_end, D_end  (null fragment-boundary states)

    After emitting in state X ∈ {M, I, D}:
      - X → X  with prob ext      (fragment continues)
      - X → X_end with prob 1-ext (fragment ends, null transition)
    X_end transitions via TKF91:
      - X_end → Y  with prob tau91[X, Y]  for Y ∈ {M, I, D, E}

    Null elimination of {M_end, I_end, D_end} recovers TKF92's 5×5 matrix.

    Count restoration decomposes effective transition counts into:
    - Direct counts on visible edges (fragment extension self-loops)
    - Null exit counts X_end → Y (TKF91 transitions at fragment boundaries)
    The null exit counts are the TKF91-equivalent counts needed for BDI stats.
"""

import numpy as np
import jax.numpy as jnp

from ..core.params import S, M, I, D, E, tkf91_trans
from ..grammar.compile import build_null_info_hmm, effective_trans_from_null_info, restore_null_counts


# Null state indices in the 8-state exploded matrix
# Layout: [S=0, M=1, I=2, D=3, E=4, M_end=5, I_end=6, D_end=7]
M_END, I_END, D_END = 5, 6, 7
_NULL_INDICES = [M_END, I_END, D_END]

# visible_map: maps null state local index (within null sub-array) to visible state
# M_end (local 0) → M (1), I_end (local 1) → I (2), D_end (local 2) → D (3)
_VISIBLE_MAP = {0: M, 1: I, 2: D}

# Mapping from emitting state to its null (fragment-end) state index
_EMIT_TO_END = {M: M_END, I: I_END, D: D_END}
# Mapping from null state index to the emitting state it came from
_END_TO_EMIT = {M_END: M, I_END: I, D_END: D}


def build_tkf92_exploded(ins_rate, del_rate, t, ext):
    """Build the 8x8 exploded TKF92 matrix with null fragment-boundary states.

    Args:
        ins_rate, del_rate: TKF91 indel rates
        t: evolutionary time
        ext: fragment extension probability

    Returns:
        upsilon: (8, 8) exploded transition matrix
        null_indices: [5, 6, 7]
        visible_map: {0: M, 1: I, 2: D}
    """
    tau91 = np.asarray(tkf91_trans(ins_rate, del_rate, t))
    upsilon = np.zeros((8, 8))

    # S row: transitions to emitting states via TKF91 (no fragment extension from S)
    for dst in [M, I, D, E]:
        upsilon[S, dst] = tau91[S, dst]

    # Emitting states M, I, D:
    #   X → X with prob ext (fragment self-loop)
    #   X → X_end with prob (1-ext) (fragment ends)
    for x, x_end in [(M, M_END), (I, I_END), (D, D_END)]:
        upsilon[x, x] = ext
        upsilon[x, x_end] = 1.0 - ext

    # Null fragment-boundary states X_end:
    #   X_end → Y via tau91[X, Y]
    for x, x_end in [(M, M_END), (I, I_END), (D, D_END)]:
        for dst in [M, I, D, E]:
            upsilon[x_end, dst] = tau91[x, dst]

    return upsilon, _NULL_INDICES, _VISIBLE_MAP


def tkf92_from_elaboration(ins_rate, del_rate, t, ext):
    """Build TKF92 transition matrix via null state elaboration + elimination.

    Returns:
        T_eff: (5, 5) effective transition matrix (should match tkf92_trans)
        null_info: NullInfo for count restoration
        upsilon: (8, 8) exploded matrix
    """
    upsilon, null_indices, visible_map = build_tkf92_exploded(ins_rate, del_rate, t, ext)
    null_info = build_null_info_hmm(upsilon, null_indices, visible_map)
    T_eff = effective_trans_from_null_info(null_info)
    return T_eff, null_info, upsilon


def restore_tkf92_counts_full(n_chi, null_info):
    """Restore counts to the full 8-state exploded matrix.

    Returns the full (8, 8) count matrix on the exploded matrix, WITHOUT
    folding null states back to visible states. This allows extracting:
    - Direct visible→visible counts (fragment extension self-loops)
    - Visible→null entry counts (X → X_end, fragment boundary decisions)
    - Null→visible exit counts (X_end → Y, TKF91 transitions)

    Args:
        n_chi: (5, 5) expected transition counts on effective T
        null_info: NullInfo from tkf92_from_elaboration

    Returns:
        n_full: (8, 8) counts on the exploded matrix
    """
    return restore_null_counts(np.asarray(n_chi), null_info, _VISIBLE_MAP,
                               return_full=True)


def decompose_tkf92_counts(n_chi, ins_rate, del_rate, t, ext):
    """Decompose TKF92 FB counts into fragment-extension and TKF91 components.

    This is the grammar-elaboration equivalent of vjp.py's manual fragExt
    correction. It uses null state restoration to systematically decompose
    each effective transition count into:
    1. Fragment extension (direct self-loop, proportional to ext)
    2. TKF91 boundary transitions (through null X_end states)

    Args:
        n_chi: (5, 5) expected transition counts on effective TKF92 matrix
        ins_rate, del_rate, t, ext: TKF92 parameters

    Returns:
        dict with:
            'frag_ext': (5, 5) fragment extension counts (only self-loops nonzero)
            'tkf91': (5, 5) TKF91-equivalent transition counts at fragment boundaries
            'n_full': (8, 8) full exploded matrix counts
            'tkf91_from_source': dict mapping source state X → (5,) TKF91 exit counts
    """
    n_chi = np.asarray(n_chi)
    _, null_info, _ = tkf92_from_elaboration(ins_rate, del_rate, t, ext)
    n_full = restore_tkf92_counts_full(n_chi, null_info)

    # Fragment extension counts: direct visible→visible (only self-loops)
    frag_ext = np.zeros((5, 5))
    for i in range(5):
        frag_ext[i, i] = n_full[i, i]

    # TKF91 counts: null→visible exits (X_end → Y)
    # These are the TKF91 transitions at fragment boundaries.
    tkf91 = np.zeros((5, 5))
    tkf91_from_source = {}
    for x_end, x in _END_TO_EMIT.items():
        exits = np.zeros(5)
        for y in range(5):
            exits[y] = n_full[x_end, y]
        # Attribute X_end exits to source state X
        tkf91[x, :] += exits
        tkf91_from_source[x] = exits

    # S row: S transitions are already direct TKF91 (S has no fragment extension)
    tkf91[S, :] = n_full[S, :5]

    return {
        'frag_ext': frag_ext,
        'tkf91': tkf91,
        'n_full': n_full,
        'tkf91_from_source': tkf91_from_source,
    }


# ===================================================================
# MixDom elaboration: chi with explicit domain-level null states
# ===================================================================

# Compound state UV → top-level state U
_UV_U = [M, M, M, I, D]     # MM→M, MI→M, MD→M, II→I, DD→D
# Compound state UV → fragment-level state X (second component)
_UV_X = [M, I, D, I, D]     # MM→M, MI→I, MD→D, II→I, DD→D
# Which compound states are M-type (top-level U = M)?
_IS_M_TYPE = [True, True, True, False, False]


def _body_index(uv, dom, frag, n_frag):
    """Map (compound_state, domain, fragment) to flat body index (0-based)."""
    return dom * 5 * n_frag + uv * n_frag + frag


def _chi_index(uv, dom, frag, n_frag):
    """Map (compound_state, domain, fragment) to flat chi index (SS=0, EE=1, then body)."""
    return 2 + _body_index(uv, dom, frag, n_frag)


def build_mixdom_upsilon(main_ins_rate, main_del_rate, t,
                         dom_ins_rates, dom_del_rates, dom_weights,
                         frag_weights, ext_rates):
    """Build 7x7 exploded domain-level transition matrix (numpy version).

    Same as mixdom.effective_trans but returns the full upsilon and sub-blocks
    instead of just the effective T.

    Returns:
        upsilon: (7, 7) exploded matrix with null states at indices 5, 6
        T_NN: (5, 5) visible-to-visible block
        T_Nnull: (5, 2) visible-to-null block
        T_nullN: (2, 5) null-to-visible block
        T_nullnull: (2, 2) null-to-null block
    """
    from ..core.bdi import tkf_beta, tkf_kappa
    tau = np.asarray(tkf91_trans(main_ins_rate, main_del_rate, t))

    # Nullability
    kappas = np.asarray(dom_ins_rates / dom_del_rates)
    betas = np.asarray(tkf_beta(dom_ins_rates, dom_del_rates, t))
    dom_w = np.asarray(dom_weights)
    z_0 = float(np.sum(dom_w * (1.0 - kappas)))
    z_t = float(np.sum(dom_w * (1.0 - kappas) * (1.0 - betas)))

    # Row mapping: S→tau[S], M→tau[M], I→tau[I], D→tau[D], E→0, M_null→tau[M], D_null→tau[D]
    tau_rows = np.zeros((7, 5))
    tau_rows[0] = tau[S]
    tau_rows[1] = tau[M]
    tau_rows[2] = tau[I]
    tau_rows[3] = tau[D]
    # tau_rows[4] = 0  (E row)
    tau_rows[5] = tau[M]
    tau_rows[6] = tau[D]

    upsilon = np.zeros((7, 7))
    # col S: no transitions to S (stays 0)
    upsilon[:, M] = (1.0 - z_t) * tau_rows[:, M]
    upsilon[:, I] = (1.0 - z_0) * tau_rows[:, I]
    upsilon[:, D] = (1.0 - z_0) * tau_rows[:, D]
    upsilon[:, E] = tau_rows[:, E]
    upsilon[:, 5] = z_t * tau_rows[:, M] + z_0 * tau_rows[:, I]   # col M_null
    upsilon[:, 6] = z_0 * tau_rows[:, D]                           # col D_null

    T_NN = upsilon[:5, :5]
    T_Nnull = upsilon[:5, 5:7]
    T_nullN = upsilon[5:7, :5]
    T_nullnull = upsilon[5:7, 5:7]

    return upsilon, T_NN, T_Nnull, T_nullN, T_nullnull


def build_mixdom_elaborated(main_ins_rate, main_del_rate, t,
                            dom_ins_rates, dom_del_rates, dom_weights,
                            frag_weights, ext_rates):
    """Build reference MixDom chi with explicit domain-level null states.

    Adds 2 null states (M_null_chi, D_null_chi) at indices N and N+1,
    representing top-level empty-domain paths that are normally eliminated
    in the closed-form build_nested_trans.

    After null elimination, chi_full[:N,:N] + null contribution should
    exactly match build_nested_trans output.

    Args:
        Same as build_nested_trans.

    Returns:
        chi_full: (N+2, N+2) transition matrix with null states
        null_indices: [N, N+1] — indices of null states in chi_full
        upsilon: (7, 7) domain-level exploded matrix
        entry_factor: (n_dom, 5) per-domain entry factors
        exit_full: (n_dom, 5, n_frag) per-domain exit factors
    """
    from ..core.bdi import tkf_beta, tkf_kappa

    dom_ins_rates = np.asarray(dom_ins_rates)
    dom_del_rates = np.asarray(dom_del_rates)
    dom_w = np.asarray(dom_weights)
    frag_w = np.asarray(frag_weights)
    ext = np.asarray(ext_rates)
    # Auto-convert MixDom1 (D, F) -> MixDom2 (D, F, F)
    if ext.ndim == 2:
        ext_3d = np.zeros((ext.shape[0], ext.shape[1], ext.shape[1]))
        for d in range(ext.shape[0]):
            ext_3d[d] = np.diag(ext[d])
        ext = ext_3d

    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_w.shape[1]
    n_body = n_dom * 5 * n_frag
    N = 2 + n_body
    N_full = N + 2
    M_NULL_CHI = N       # index of M_null in chi_full
    D_NULL_CHI = N + 1   # index of D_null in chi_full

    # Build 7×7 upsilon and extract blocks
    upsilon, T_NN, T_Nnull, T_nullN, T_nullnull = build_mixdom_upsilon(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_w, frag_w, ext)

    # Per-domain TKF91 matrices
    tau_all = np.array([np.asarray(tkf91_trans(dom_ins_rates[d], dom_del_rates[d], t))
                        for d in range(n_dom)])
    kappas = np.asarray(dom_ins_rates / dom_del_rates)

    # Nullability normalization
    from .mixdom import nullability
    z_0, z_t = nullability(
        jnp.asarray(dom_ins_rates), jnp.asarray(dom_del_rates),
        jnp.asarray(dom_w), t)
    z_0, z_t = float(z_0), float(z_t)
    norm_M = max(1.0 - z_t, 1e-30)
    norm_ID = max(1.0 - z_0, 1e-30)

    # Entry factor: how each UV_df is entered from top level
    entry_factor = np.zeros((n_dom, 5))
    for d in range(n_dom):
        for uv in range(5):
            if _IS_M_TYPE[uv]:
                entry_factor[d, uv] = tau_all[d, S, _UV_X[uv]] / norm_M
            else:
                entry_factor[d, uv] = kappas[d] / norm_ID

    # Exit factor: how each UV_df exits its domain
    exit_inner = np.zeros((n_dom, 5))
    for d in range(n_dom):
        for uv in range(5):
            if _IS_M_TYPE[uv]:
                exit_inner[d, uv] = tau_all[d, _UV_X[uv], E]
            else:
                exit_inner[d, uv] = 1.0 - kappas[d]

    # notext[d,f] = 1 - sum_g ext[d,f,g] (MixDom2: row sums)
    notext = np.zeros((n_dom, n_frag))
    for d in range(n_dom):
        for f in range(n_frag):
            notext[d, f] = 1.0 - ext[d, f, :].sum()

    exit_full = np.zeros((n_dom, 5, n_frag))
    for d in range(n_dom):
        for uv in range(5):
            for f in range(n_frag):
                exit_full[d, uv, f] = notext[d, f] * exit_inner[d, uv]

    # --- Build chi_full ---
    chi_full = np.zeros((N_full, N_full))

    # === Intra-domain transitions (block-diagonal, same as build_nested_trans) ===
    for d in range(n_dom):
        tau_d = tau_all[d]
        kappa_d = kappas[d]
        tau_MID = tau_d[np.ix_([M, I, D], [M, I, D])]

        for suv in range(5):
            for sf in range(n_frag):
                src = _chi_index(suv, d, sf, n_frag)
                for duv in range(5):
                    for df in range(n_frag):
                        dst = _chi_index(duv, d, df, n_frag)

                        # Fragment extension: ext[d, sf, df] with delta(suv=duv)
                        # (extension preserves the MID state type)
                        if suv == duv:
                            chi_full[src, dst] += ext[d, sf, df]

                        # Non-extension intra-domain
                        if _IS_M_TYPE[suv] and _IS_M_TYPE[duv]:
                            # M-block: notext[d,sf]*tau_MID[x,y]*frag_w[d,df]
                            x_src = _UV_X[suv]  # M, I, or D
                            x_dst = _UV_X[duv]
                            # Map M,I,D → 0,1,2 for tau_MID indexing
                            mid_map = {M: 0, I: 1, D: 2}
                            chi_full[src, dst] += (notext[d, sf] *
                                                   tau_MID[mid_map[x_src], mid_map[x_dst]] *
                                                   frag_w[d, df])
                        elif not _IS_M_TYPE[suv] and not _IS_M_TYPE[duv]:
                            # I/D block: notext[d,sf]*kappa*frag_w[d,df]
                            if _UV_U[suv] == _UV_U[duv]:  # II→II or DD→DD
                                chi_full[src, dst] += (notext[d, sf] *
                                                       kappa_d * frag_w[d, df])

    # === Inter-domain transitions (using T_NN, direct paths only) ===
    for sd in range(n_dom):
        for suv in range(5):
            for sf in range(n_frag):
                src = _chi_index(suv, sd, sf, n_frag)
                U_src = _UV_U[suv]
                ef = exit_full[sd, suv, sf]

                for dd in range(n_dom):
                    for duv in range(5):
                        for df in range(n_frag):
                            dst = _chi_index(duv, dd, df, n_frag)
                            U_dst = _UV_U[duv]
                            dest = dom_w[dd] * frag_w[dd, df] * entry_factor[dd, duv]
                            chi_full[src, dst] += ef * T_NN[U_src, U_dst] * dest

    # === Body → null_chi ===
    for sd in range(n_dom):
        for suv in range(5):
            for sf in range(n_frag):
                src = _chi_index(suv, sd, sf, n_frag)
                U_src = _UV_U[suv]
                ef = exit_full[sd, suv, sf]
                chi_full[src, M_NULL_CHI] = ef * T_Nnull[U_src, 0]
                chi_full[src, D_NULL_CHI] = ef * T_Nnull[U_src, 1]

    # === SS row ===
    # SS → body (direct via T_NN)
    for dd in range(n_dom):
        for duv in range(5):
            for df in range(n_frag):
                dst = _chi_index(duv, dd, df, n_frag)
                U_dst = _UV_U[duv]
                dest = dom_w[dd] * frag_w[dd, df] * entry_factor[dd, duv]
                chi_full[0, dst] = T_NN[S, U_dst] * dest

    # SS → EE (direct)
    chi_full[0, 1] = T_NN[S, E]

    # SS → null_chi
    chi_full[0, M_NULL_CHI] = T_Nnull[S, 0]
    chi_full[0, D_NULL_CHI] = T_Nnull[S, 1]

    # === EE column from body (direct via T_NN) ===
    for sd in range(n_dom):
        for suv in range(5):
            for sf in range(n_frag):
                src = _chi_index(suv, sd, sf, n_frag)
                U_src = _UV_U[suv]
                ef = exit_full[sd, suv, sf]
                chi_full[src, 1] = ef * T_NN[U_src, E]

    # === Null_chi → body ===
    for dd in range(n_dom):
        for duv in range(5):
            for df in range(n_frag):
                dst = _chi_index(duv, dd, df, n_frag)
                U_dst = _UV_U[duv]
                dest = dom_w[dd] * frag_w[dd, df] * entry_factor[dd, duv]
                chi_full[M_NULL_CHI, dst] = T_nullN[0, U_dst] * dest
                chi_full[D_NULL_CHI, dst] = T_nullN[1, U_dst] * dest

    # === Null_chi → EE ===
    chi_full[M_NULL_CHI, 1] = T_nullN[0, E]
    chi_full[D_NULL_CHI, 1] = T_nullN[1, E]

    # === Null_chi → null_chi ===
    chi_full[M_NULL_CHI, M_NULL_CHI] = T_nullnull[0, 0]
    chi_full[M_NULL_CHI, D_NULL_CHI] = T_nullnull[0, 1]
    chi_full[D_NULL_CHI, M_NULL_CHI] = T_nullnull[1, 0]
    chi_full[D_NULL_CHI, D_NULL_CHI] = T_nullnull[1, 1]

    null_indices = [M_NULL_CHI, D_NULL_CHI]
    return chi_full, null_indices, upsilon, entry_factor, exit_full


def mixdom_from_elaboration(main_ins_rate, main_del_rate, t,
                            dom_ins_rates, dom_del_rates, dom_weights,
                            frag_weights, ext_rates):
    """Build MixDom chi via null state elaboration + elimination.

    Returns:
        chi_eff: (N, N) effective chi matrix (should match build_nested_trans)
        null_info: NullInfo for count restoration
        chi_full: (N+2, N+2) elaborated chi with null states
    """
    chi_full, null_indices, upsilon, _, _ = build_mixdom_elaborated(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_weights,
        frag_weights, ext_rates)

    N_full = chi_full.shape[0]
    N = N_full - 2

    # Null elimination
    T_NN_chi = chi_full[:N, :N]
    T_Nnull_chi = chi_full[:N, N:]
    T_nullN_chi = chi_full[N:, :N]
    T_nullnull_chi = chi_full[N:, N:]

    closure = np.linalg.inv(np.eye(2) - T_nullnull_chi)
    null_contrib = T_Nnull_chi @ closure @ T_nullN_chi
    chi_eff = T_NN_chi + null_contrib

    # Build NullInfo (for potential count restoration in Phase 3)
    null_info = build_null_info_hmm(chi_full, null_indices, visible_map=None)

    return chi_eff, null_info, chi_full


def resolve_counts_elaborated(n_chi, main_ins_rate, main_del_rate, t,
                               dom_ins_rates, dom_del_rates, dom_weights,
                               frag_weights, ext_rates):
    """Reference count resolution via chi_full null restoration.

    Given n_chi (N×N expected counts from FB on effective chi), uses the
    elaborated chi_full to restore null counts and decompose into components.

    This serves as a reference implementation for validating the production
    proportional decomposition in em_mixdom.resolve_counts.

    Returns dict with same keys as em_mixdom.resolve_counts:
        top_counts: (5, 5) inter-domain counts on T_eff
        top_counts_restored: (5, 5) null-restored counts (with phantom events)
        dom_counts: list of (5, 5) per-domain intra-domain counts
        dom_occupancy: (n_dom,) posterior mass per domain
        frag_occupancy: (n_dom, n_frag) posterior mass per fragment
        ext_self: (n_dom, n_frag) extension self-loop counts
        ext_exit: (n_dom, n_frag) non-extension intra-fragment counts
    """
    n_chi = np.asarray(n_chi)
    dom_ins_rates = np.asarray(dom_ins_rates)
    dom_del_rates = np.asarray(dom_del_rates)
    dom_w = np.asarray(dom_weights)
    frag_w = np.asarray(frag_weights)
    ext = np.asarray(ext_rates)
    # Auto-convert MixDom1 (D, F) -> MixDom2 (D, F, F)
    if ext.ndim == 2:
        ext_3d = np.zeros((ext.shape[0], ext.shape[1], ext.shape[1]))
        for d in range(ext.shape[0]):
            ext_3d[d] = np.diag(ext[d])
        ext = ext_3d

    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_w.shape[1]
    n_body = n_dom * 5 * n_frag
    N = 2 + n_body

    # Build chi_full and restore null counts
    chi_full, null_indices, upsilon, entry_factor, exit_full = \
        build_mixdom_elaborated(
            main_ins_rate, main_del_rate, t,
            dom_ins_rates, dom_del_rates, dom_w, frag_w, ext)

    null_info = build_null_info_hmm(chi_full, null_indices, visible_map=None)
    n_chi_full = restore_null_counts(n_chi, null_info, visible_map=None,
                                      return_full=True)

    # The visible block n_chi_full[:N,:N] has counts on chi_NN (which uses T_NN).
    # The null block has phantom null-mediated counts.
    # We need to decompose the visible block into inter-direct / intra / frag-ext.

    N_full = N + 2
    M_NULL_CHI = N
    D_NULL_CHI = N + 1

    # chi_NN is the visible block of chi_full
    chi_NN = chi_full[:N, :N]

    # Per-domain TKF91 matrices and kappas (same as build_mixdom_elaborated)
    tau_all = np.array([np.asarray(tkf91_trans(dom_ins_rates[d], dom_del_rates[d], t))
                        for d in range(n_dom)])
    kappas = dom_ins_rates / dom_del_rates

    _, T_NN, T_Nnull, T_nullN, T_nullnull = build_mixdom_upsilon(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_w, frag_w, ext)

    # Null fold map: D_null always folds to D.
    # M_null is a mixture z_t·τ[u,M] + z_0·τ[u,I], so we decompose into M and I.
    tau_main = np.asarray(tkf91_trans(main_ins_rate, main_del_rate, t))
    from ..models.mixdom import nullability as _nullability
    _z_0, _z_t = _nullability(
        jnp.array(dom_ins_rates), jnp.array(dom_del_rates),
        jnp.array(dom_w), t)
    _z_0, _z_t = float(_z_0), float(_z_t)

    # Row fold map: M_null_chi → M, D_null_chi → D (rows are unambiguous)
    _null_fold = {0: M, 1: D}

    def _fold_null_col(U_src, k, n, top_counts):
        """Fold count n from source state U_src to null column k into top_counts.

        k=0 (M_null): decompose mixture z_t·τ[U_src,M] + z_0·τ[U_src,I]
        k=1 (D_null): fold directly to D column
        """
        if k == 1:
            top_counts[U_src, D] += n
        else:
            # M_null mixture decomposition
            w_M = _z_t * tau_main[U_src, M]
            w_I = _z_0 * tau_main[U_src, I]
            denom = w_M + w_I
            if denom < 1e-30:
                top_counts[U_src, M] += n  # fallback
            else:
                top_counts[U_src, M] += n * w_M / denom
                top_counts[U_src, I] += n * w_I / denom

    # Initialize outputs
    top_counts = np.zeros((5, 5))
    dom_counts = [np.zeros((5, 5)) for _ in range(n_dom)]
    dom_occupancy = np.zeros(n_dom)
    frag_occupancy = np.zeros((n_dom, n_frag))
    ext_self = np.zeros((n_dom, n_frag))
    ext_exit = np.zeros((n_dom, n_frag))

    # --- Process SS row (row 0) ---
    # SS → EE
    top_counts[S, E] += n_chi_full[0, 1]

    # SS → body: all direct (T_NN path), attribute to top_counts
    for dd in range(n_dom):
        for duv in range(5):
            for df in range(n_frag):
                dst = _chi_index(duv, dd, df, n_frag)
                n = n_chi_full[0, dst]
                if n < 1e-15:
                    continue
                U_dst = _UV_U[duv]
                top_counts[S, U_dst] += n
                dom_occupancy[dd] += n
                frag_occupancy[dd, df] += n

    # SS → null_chi: attribute to top_counts via null fold (with mixture decomp)
    for k in range(2):
        n = n_chi_full[0, N + k]
        if n > 1e-15:
            _fold_null_col(S, k, n, top_counts)

    # --- Process null_chi → visible ---
    for k in range(2):
        U_null = _null_fold[k]
        # null → EE
        n = n_chi_full[N + k, 1]
        if n > 1e-15:
            top_counts[U_null, E] += n

        # null → body
        for dd in range(n_dom):
            for duv in range(5):
                for df in range(n_frag):
                    dst = _chi_index(duv, dd, df, n_frag)
                    n = n_chi_full[N + k, dst]
                    if n < 1e-15:
                        continue
                    U_dst = _UV_U[duv]
                    top_counts[U_null, U_dst] += n
                    dom_occupancy[dd] += n
                    frag_occupancy[dd, df] += n

        # null → null (with mixture decomp for M_null destination)
        for kp in range(2):
            n = n_chi_full[N + k, N + kp]
            if n > 1e-15:
                _fold_null_col(U_null, kp, n, top_counts)

    # --- Process body → null_chi (with mixture decomp for M_null) ---
    for sd in range(n_dom):
        for suv in range(5):
            for sf in range(n_frag):
                src = _chi_index(suv, sd, sf, n_frag)
                U_src = _UV_U[suv]
                for k in range(2):
                    n = n_chi_full[src, N + k]
                    if n > 1e-15:
                        _fold_null_col(U_src, k, n, top_counts)

    # --- Process body → EE (direct via T_NN) ---
    for sd in range(n_dom):
        for suv in range(5):
            for sf in range(n_frag):
                src = _chi_index(suv, sd, sf, n_frag)
                n = n_chi_full[src, 1]
                if n > 1e-15:
                    U_src = _UV_U[suv]
                    top_counts[U_src, E] += n

    # --- Process body → body (visible block) ---
    # Need to split into inter-domain-direct vs intra-domain vs frag-ext.
    # Use chi_NN weights for proportional decomposition.
    for sd in range(n_dom):
        for suv in range(5):
            for sf in range(n_frag):
                src = _chi_index(suv, sd, sf, n_frag)
                U_src = _UV_U[suv]
                X_src = _UV_X[suv]
                ef = exit_full[sd, suv, sf]

                for dd in range(n_dom):
                    for duv in range(5):
                        for df in range(n_frag):
                            dst = _chi_index(duv, dd, df, n_frag)
                            n_obs = n_chi_full[src, dst]
                            if n_obs < 1e-15:
                                continue

                            U_dst = _UV_U[duv]
                            X_dst = _UV_X[duv]
                            same_dom = (sd == dd)

                            # Compute chi_NN weights for this transition
                            # Inter-domain direct (T_NN)
                            dest = dom_w[dd] * frag_w[dd, df] * entry_factor[dd, duv]
                            w_inter = ef * T_NN[U_src, U_dst] * dest

                            # Intra-domain (notext * tau * frag_w path)
                            w_intra = 0.0
                            notext_sf = 1.0 - ext[sd, sf, :].sum()
                            if same_dom and U_src == U_dst:
                                mid_map = {M: 0, I: 1, D: 2}
                                if _IS_M_TYPE[suv] and _IS_M_TYPE[duv]:
                                    tau_d = tau_all[sd]
                                    tau_MID = tau_d[np.ix_([M, I, D], [M, I, D])]
                                    w_intra = (notext_sf *
                                               tau_MID[mid_map[X_src], mid_map[X_dst]] *
                                               frag_w[sd, df])
                                elif not _IS_M_TYPE[suv] and not _IS_M_TYPE[duv]:
                                    w_intra = (notext_sf *
                                               kappas[sd] * frag_w[sd, df])

                            # Fragment extension: ext[d, sf, df] with delta(suv=duv)
                            w_frag = 0.0
                            if same_dom and suv == duv:
                                w_frag = ext[sd, sf, df]

                            w_total = w_inter + w_intra + w_frag
                            if w_total < 1e-30:
                                continue

                            r_inter = w_inter / w_total
                            r_intra = w_intra / w_total
                            r_frag = w_frag / w_total

                            # Attribute inter-direct to top_counts
                            top_counts[U_src, U_dst] += n_obs * r_inter

                            # Attribute intra to dom_counts
                            if r_intra > 0:
                                dom_counts[dd][X_src, X_dst] += n_obs * r_intra

                            # Attribute frag-ext
                            if r_frag > 0:
                                ext_self[dd, df] += n_obs * r_frag

                            # All paths contribute to occupancy
                            dom_occupancy[dd] += n_obs
                            frag_occupancy[dd, df] += n_obs
                            if r_intra > 0:
                                ext_exit[dd, df] += n_obs * r_intra

    # Compute null-restored top_counts by folding null events
    # The top_counts already include null events from the null block above.
    # For a separate "top_counts on T_eff" (without phantom events),
    # we'd need to subtract the null contributions.
    # Here top_counts = direct_inter + null_phantom, which is the
    # null-restored version.
    top_counts_restored = top_counts.copy()

    # Compute top_counts on T_eff (un-restored): subtract null phantom events
    top_counts_eff = np.zeros((5, 5))
    # Re-derive: the non-null inter-domain contribution
    # SS → body and body → EE via T_NN are already direct
    # We just need to separate the direct and null contributions in top_counts
    # Actually, it's simpler to compute top_counts_eff directly from n_chi_full
    # by only counting the visible-block inter-domain and SS/EE paths.

    # SS → EE
    top_counts_eff[S, E] = n_chi_full[0, 1]
    # SS → body (direct)
    for dd in range(n_dom):
        for duv in range(5):
            for df in range(n_frag):
                dst = _chi_index(duv, dd, df, n_frag)
                n = n_chi_full[0, dst]
                if n > 1e-15:
                    top_counts_eff[S, _UV_U[duv]] += n
    # body → EE (direct)
    for sd in range(n_dom):
        for suv in range(5):
            for sf in range(n_frag):
                src = _chi_index(suv, sd, sf, n_frag)
                n = n_chi_full[src, 1]
                if n > 1e-15:
                    top_counts_eff[_UV_U[suv], E] += n
    # body → body inter-domain direct
    for sd in range(n_dom):
        for suv in range(5):
            for sf in range(n_frag):
                src = _chi_index(suv, sd, sf, n_frag)
                for dd in range(n_dom):
                    for duv in range(5):
                        for df in range(n_frag):
                            dst = _chi_index(duv, dd, df, n_frag)
                            n_obs = n_chi_full[src, dst]
                            if n_obs < 1e-15:
                                continue
                            ef = exit_full[sd, suv, sf]
                            dest = dom_w[dd] * frag_w[dd, df] * entry_factor[dd, duv]
                            w_inter = ef * T_NN[_UV_U[suv], _UV_U[duv]] * dest
                            chi_nn_val = chi_NN[src, dst]
                            if chi_nn_val < 1e-30:
                                continue
                            top_counts_eff[_UV_U[suv], _UV_U[duv]] += (
                                n_obs * w_inter / chi_nn_val)

    return {
        'top_counts': top_counts_eff,
        'top_counts_restored': top_counts_restored,
        'dom_counts': dom_counts,
        'dom_occupancy': dom_occupancy,
        'frag_occupancy': frag_occupancy,
        'ext_self': ext_self,
        'ext_exit': ext_exit,
    }
