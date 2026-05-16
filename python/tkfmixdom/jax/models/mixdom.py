"""MixDom: Nested Pair HMM with domains, fragments, and site classes.

Constructs the hierarchically nested Pair HMM transition matrix chi
from the paper's Section on "Nested Pair HMM".

States: {SS, EE} ∪ {UV_df : UV ∈ {MM, MI, MD, II, DD}, d ∈ domains, f ∈ fragments}
Total states: 2 + 5 * n_dom * n_frag

The transition matrix chi has entries combining:
- Top-level TKF91 (domain-level) transitions T (5x5 effective matrix with nulls removed)
- Nested TKF92 (fragment-level) transitions tau^(d)
- Fragment extension self-loops ext_f
- Domain/fragment mixture weights
"""

import jax
import jax.numpy as jnp
from jax.nn import logsumexp
from ..core.params import (
    S, M, I, D, E,
    tkf91_trans, tkf_beta, tkf_kappa,
)


def normalize_log_trans(log_chi):
    """Normalize rows of log transition matrix, returning log-norms as diagnostic.

    Computes per-row logsumexp and subtracts it so each row sums to 1 in
    probability space.  Rows that are all -inf (e.g. the EE/end row at
    index 1) are left untouched; their reported norm is 0.0.

    Args:
        log_chi: (N, N) log-space transition matrix.

    Returns:
        normalized_log_chi: (N, N) row-normalized log transition matrix.
        row_log_norms: (N,) per-row log normalization constants (0.0 for
            all-inf rows).
    """
    row_log_norms = logsumexp(log_chi, axis=1)
    # Don't normalize all-inf rows (e.g. EE row) — treat their norm as 0.0
    safe_norms = jnp.where(jnp.isfinite(row_log_norms), row_log_norms, 0.0)
    normalized = log_chi - safe_norms[:, None]
    return normalized, safe_norms


# Nested state types (compound states UV)
MM, MI, MD, II, DD = 0, 1, 2, 3, 4
SS_STATE, EE_STATE = 5, 6  # special start/end


def state_index(uv, dom, frag, n_frag):
    """Map (compound_state, domain, fragment) to flat index.

    Index 0 = SS, index 1 = EE, then 2 + (uv * n_dom * n_frag + dom * n_frag + frag).
    """
    return 2 + uv * n_frag + dom * 5 * n_frag + frag


def n_states(n_dom, n_frag):
    """Total number of states in nested Pair HMM."""
    return 2 + 5 * n_dom * n_frag


# Emission type for each compound state UV: MM→M, MI→I, MD→D, II→I, DD→D
_UV_EMIT_TYPE = jnp.array([M, I, D, I, D])


def state_types(n_dom, n_frag):
    """Build state_types array mapping MixDom states to M/I/D/S/E emission types.

    Returns:
        (N,) array where N = 2 + 5*n_dom*n_frag.
        Entry 0=S (start), 1=E (end), body states map to M/I/D.
    """
    body = jnp.tile(jnp.repeat(_UV_EMIT_TYPE, n_frag), n_dom)
    return jnp.concatenate([jnp.array([S, E]), body])


def nullability(dom_ins_rates, dom_del_rates, dom_weights, t):
    """Compute nullability probabilities z_0 and z_t.

    z_0 = Σ_d ω_d (1 - κ_d)       — prob empty segment from stationary dist
    z_t = Σ_d ω_d (1 - κ_d)(1-β_d) — prob empty segment from pair HMM

    Fully vectorized — tkf_beta accepts array inputs.

    Args:
        dom_ins_rates: (n_dom,) per-domain insertion rates
        dom_del_rates: (n_dom,) per-domain deletion rates
        dom_weights: (n_dom,) domain mixture weights
        t: evolutionary time

    Returns:
        z_0, z_t: scalar nullability probabilities
    """
    kappas = dom_ins_rates / dom_del_rates
    betas = tkf_beta(dom_ins_rates, dom_del_rates, t)  # element-wise
    z_0 = jnp.sum(dom_weights * (1.0 - kappas))
    z_t = jnp.sum(dom_weights * (1.0 - kappas) * (1.0 - betas))
    return z_0, z_t


def effective_trans_per_type(main_ins_rate, main_del_rate, t,
                             dom_ins_rates, dom_del_rates, dom_weights):
    """Build type-specific effective transitions from the expanded upsilon.

    Uses the expanded upsilon with explicit per-type null fan-out states
    (5 + 3K states). Each domain type choice is a null fan-out transition,
    so dom_weights ONLY appear through these fan-outs.

    Uses EXIT-type decomposition: T_exit_k[k, u, v] is the probability
    that a path from visible state u exits through a type-k null state to
    visible state v. This correctly assigns the domain type of the
    destination (the domain that actually produced the non-null pair).

    Returns:
        T_exit_k: (K, 5, 5) exit-type-specific effective transition contribution.
                  T_exit_k[k, u, v] = Σ paths from u that exit through type-k to v.
                  Σ_k T_exit_k[k, u, v] = null_contrib[u, v].
        T_eff: (5, 5) full effective transition matrix (T_NN + Σ_k T_exit_k).
    """
    K = dom_ins_rates.shape[0]
    tau = tkf91_trans(main_ins_rate, main_del_rate, t)

    # Per-type nullabilities
    kappas = dom_ins_rates / dom_del_rates              # (K,)
    betas = tkf_beta(dom_ins_rates, dom_del_rates, t)   # (K,)
    z_t_k = (1.0 - kappas) * (1.0 - betas)             # M-entry null prob
    z_0_k = 1.0 - kappas                                # I/D-entry null prob

    # tau rows for visible sources: S,M,I use tau[S]; D uses tau[D]; E=0
    tau_rows = jnp.stack([tau[S], tau[M], tau[I], tau[D], jnp.zeros(5)])  # (5, 5)

    # tau[M,I,D] columns from tau_rows: (5, 3)
    _MID_idx = jnp.array([M, I, D])
    tau_MID = tau_rows[:, _MID_idx]  # (5, 3) — for each source, the M/I/D dest probs

    # --- Build expanded upsilon submatrices ---

    # T_Nnull: (5, 3K) — visible source → type-k null states
    # T_Nnull[u, 3k+e] = w_k * tau_rows[u, MID[e]]
    w_rep = jnp.repeat(dom_weights, 3)                   # (3K,)
    T_Nnull = jnp.tile(tau_MID, (1, K)) * w_rep[None, :] # (5, 3K)

    # T_nullN: (3K, 5) — type-k null states → visible
    # For Mk (e=0): col M = 1-z_t_k, col E = z_t_k*tau[M,E], rest 0
    # For Ik (e=1): col I = kappa_k, col E = z_0_k*tau[M,E], rest 0
    # For Dk (e=2): col D = kappa_k, col E = z_0_k*tau[D,E], rest 0

    # Non-null exit probabilities per entry type: (K, 3)
    nonnull = jnp.stack([1.0 - z_t_k, kappas, kappas], axis=1)  # (K, 3)
    # Which visible state each exits to: M=1 for e=0, I=2 for e=1, D=3 for e=2
    exit_col = jnp.array([M, I, D])

    # Null prob per entry type: (K, 3)
    null_prob = jnp.stack([z_t_k, z_0_k, z_0_k], axis=1)  # (K, 3)
    # tau row for E destination per entry: tau[M,E] for Mk/Ik, tau[D,E] for Dk
    tau_E = jnp.stack([
        jnp.broadcast_to(tau[M, E], (K,)),
        jnp.broadcast_to(tau[M, E], (K,)),  # Ik uses tau[I]=tau[M]
        jnp.broadcast_to(tau[D, E], (K,)),
    ], axis=1)  # (K, 3)

    # Build T_nullN row by row: (3K, 5)
    T_nullN = jnp.zeros((3 * K, 5))
    for e in range(3):
        # Rows 3k+e for all k: non-null exit to col exit_col[e]
        rows = jnp.arange(K) * 3 + e
        T_nullN = T_nullN.at[rows, exit_col[e]].set(nonnull[:, e])
        T_nullN = T_nullN.at[rows, E].set(null_prob[:, e] * tau_E[:, e])

    # T_nullnull: (3K, 3K) — null state → null state
    # T_nullnull[3k1+e1, 3k2+e2] = z_{k1,e1} * w_{k2} * tau_r[e1, MID[e2]]
    # tau_r: Mk/Ik use tau[M], Dk uses tau[D]
    tau_null_rows = jnp.stack([tau[M], tau[M], tau[D]])  # (3, 5) — indexed by entry e
    tau_null_MID = tau_null_rows[:, _MID_idx]             # (3, 3) — tau[e1, MID[e2]]

    # z_flat[3k+e] = null_prob[k, e]
    z_flat = null_prob.reshape(3 * K)  # (3K,)

    # T_nullnull[i, j] = z_flat[i] * w[j//3] * tau_null_MID[i%3, j%3]
    i_e = jnp.arange(3 * K) % 3  # entry type of source
    j_k = jnp.arange(3 * K) // 3  # domain type of dest
    j_e = jnp.arange(3 * K) % 3   # entry type of dest
    T_nullnull = (z_flat[:, None] *
                  dom_weights[j_k[None, :]] *
                  tau_null_MID[i_e[:, None], j_e[None, :]])

    # --- Null elimination ---
    I_null = jnp.eye(3 * K)
    closure = jnp.linalg.inv(I_null - T_nullnull)    # (3K, 3K)
    closure_nullN = closure @ T_nullN                  # (3K, 5)

    # EXIT-type decomposition: T_exit_k[k, u, v] = T_Nnull @ closure_to_k @ T_nullN_k
    # where closure_to_k = closure[:, 3k:3k+3], T_nullN_k = T_nullN[3k:3k+3, :]
    # This decomposes by which domain type's non-null realization we used (exit type),
    # correctly matching destination body state domain types in chi.
    closure_3d = closure.reshape(3 * K, K, 3)     # closure_3d[i, k, e] = closure[i, 3k+e]
    T_nullN_3d = T_nullN.reshape(K, 3, 5)         # T_nullN_3d[k, e, v] = T_nullN[3k+e, v]

    # closure_exit[k, i, v] = Σ_e closure[i, 3k+e] * T_nullN[3k+e, v]
    closure_exit = jnp.einsum('ike,kev->kiv', closure_3d, T_nullN_3d)  # (K, 3K, 5)

    # T_exit_k[k, u, v] = Σ_i T_Nnull[u, i] * closure_exit[k, i, v]
    T_exit_k = jnp.einsum('ui,kiv->kuv', T_Nnull, closure_exit)  # (K, 5, 5)

    # T_NN: only direct u→E transitions
    T_NN = jnp.zeros((5, 5)).at[:, E].set(tau_rows[:, E])

    # Full T_eff
    T_eff = T_NN + T_exit_k.sum(axis=0)

    return T_exit_k, T_eff


def effective_trans(main_ins_rate, main_del_rate, t,
                    dom_ins_rates, dom_del_rates, dom_weights,
                    frag_weights, ext_rates):
    """Build the effective 5x5 top-level transition matrix T.

    Wrapper around effective_trans_per_type for backward compatibility.

    Returns:
        T: (5, 5) effective transition matrix (linear-space)
    """
    _, T = effective_trans_per_type(main_ins_rate, main_del_rate, t,
                                    dom_ins_rates, dom_del_rates, dom_weights)
    return T


def _build_state_map(n_dom, n_frag):
    """Build state map dict for backward compatibility."""
    state_map = {(-1, -1, -1): 0, (-2, -2, -2): 1}
    for d in range(n_dom):
        for f in range(n_frag):
            for uv in range(5):
                state_map[(uv, d, f)] = 2 + d * 5 * n_frag + uv * n_frag + f
    return state_map


# Compound state UV → top-level state U
_UV_U = jnp.array([M, M, M, I, D])    # MM→M, MI→M, MD→M, II→I, DD→D
# Compound state UV → fragment-level state X (second component)
_UV_X = jnp.array([M, I, D, I, D])    # MM→M, MI→I, MD→D, II→I, DD→D
# Which compound states are M-type (top-level U = M)?
_IS_M_TYPE = jnp.array([True, True, True, False, False])
# Indices of M, I, D within the 5-state space
_MID = jnp.array([M, I, D])


def build_nested_trans(main_ins_rate, main_del_rate, t,
                       dom_ins_rates, dom_del_rates, dom_weights,
                       frag_weights, ext_rates):
    """Build the full nested Pair HMM transition matrix chi.

    Uses type-specific effective transitions T_exit_k from the expanded
    upsilon, so dom_weights appear ONLY through the null fan-out transitions.
    This makes the weight M-step a pure Dirichlet conjugate update.

    The inter-domain block uses T_exit_k[d_dst] (exit-type decomposition)
    to correctly assign domain types to destination body states. Entry
    normalization is per-type: tau_d[S, X(uv)] / beta_d for M-mode,
    1.0 for I/D-mode (kappa already in T_exit_k).

    Fully vectorized — no Python loops or at[].set over body states.
    Compatible with JAX autodiff and JIT.

    Args:
        main_ins_rate, main_del_rate: top-level indel rates
        t: evolutionary time
        dom_ins_rates: (n_dom,) per-domain insertion rates
        dom_del_rates: (n_dom,) per-domain deletion rates
        dom_weights: (n_dom,) domain type distribution
        frag_weights: (n_dom, n_frag) fragment type distribution per domain
        ext_rates: (n_dom, n_frag, n_frag) fragment transition matrices per domain
                   (MixDom2 format). Also accepts (n_dom, n_frag) for backward
                   compatibility (MixDom1 format, auto-converted to diagonal).

    Returns:
        chi: (N, N) transition matrix (linear, not log)
        state_map: dict mapping (uv, d, f) -> flat index
    """
    # Auto-convert MixDom1 ext_rates (D, F) -> MixDom2 (D, F, F) diagonal
    ext_rates = jnp.asarray(ext_rates)
    if ext_rates.ndim == 2:
        ext_rates = jax.vmap(jnp.diag)(ext_rates)  # (D, F) -> (D, F, F)
    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_weights.shape[1]
    n_body = n_dom * 5 * n_frag
    N = 2 + n_body

    # Type-specific effective transitions T_exit_k (K, 5, 5) and full T_eff (5, 5)
    # T_exit_k[k, u, v] = prob of going from visible u through type-k exit to v
    T_exit_k, T = effective_trans_per_type(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_weights)

    # Per-domain TKF91 transition matrices: (n_dom, 5, 5)
    tau_all = jax.vmap(lambda lr, dr: tkf91_trans(lr, dr, t))(
        dom_ins_rates, dom_del_rates)
    kappas = dom_ins_rates / dom_del_rates  # (n_dom,)

    # --- Entry factor (per-type normalization) ---
    # M-type (MM,MI,MD): tau_d[S, X(uv)] / beta_d  where beta_d = 1 - tau_d[S, E]
    #   (per-type survival normalization; distributes among compound M-states)
    # I/D-type (II,DD):  1.0  (kappa already captured in T_exit_k)
    # Shape: (n_dom, 5)
    beta_d = jnp.maximum(1.0 - tau_all[:, S, E], 1e-30)  # (n_dom,)
    entry_M = tau_all[:, S, :][:, _UV_X] / beta_d[:, None]      # (n_dom, 5)
    entry_ID = jnp.ones((n_dom, 5))                              # (n_dom, 5)
    entry_factor = jnp.where(_IS_M_TYPE[None, :], entry_M, entry_ID)

    # --- Exit inner factor (before multiplying by 1-ext) ---
    # M-type: tau_d[X(uv), E]
    # I/D-type: (1 - kappa_d)
    # Shape: (n_dom, 5)
    exit_inner_M = tau_all[:, :, E][:, _UV_X]               # (n_dom, 5)
    exit_inner_ID = (1.0 - kappas)[:, None] * jnp.ones(5)[None, :]  # (n_dom, 5)
    exit_inner = jnp.where(_IS_M_TYPE[None, :], exit_inner_M, exit_inner_ID)

    # notext[d,f] = 1 - sum_g ext[d,f,g] (row sums of fragment transition matrix)
    # Shape: (n_dom, n_frag)
    notext = 1.0 - ext_rates.sum(axis=-1)

    # Full exit: notext[d,f] * exit_inner[d, uv]
    # Shape: (n_dom, 5, n_frag)
    exit_full = notext[:, None, :] * exit_inner[:, :, None]

    # --- Inter-domain block (using T_exit_k) ---
    # chi_inter[src, dst] = exit[src] * T_exit_k[d_dst, U(src), U(dst)]
    #                       * frag_weights[d_dst, f_dst] * entry_factor[d_dst, uv_dst]
    # (No dom_weights here — they're inside T_exit_k via the fan-out transitions)
    exit_flat = exit_full.reshape(n_body)                    # (n_body,)
    dest_factor = (frag_weights[:, None, :] *
                   entry_factor[:, :, None])                 # (n_dom, 5, n_frag)
    dest_flat = dest_factor.reshape(n_body)                  # (n_body,)

    # Per-destination-domain compound T: T_k_compound[k, suv, duv] = T_exit_k[k, U(suv), U(duv)]
    T_k_compound = T_exit_k[:, _UV_U][:, :, _UV_U]          # (K, 5, 5)

    # Build T_expanded using destination domain index
    block_size = 5 * n_frag
    d_indices = jnp.arange(n_body) // block_size             # domain of each body state
    suv_indices = jnp.tile(jnp.repeat(jnp.arange(5), n_frag), n_dom)  # (n_body,)

    # T_expanded[i, j] = T_k_compound[d_j, suv_i, suv_j]
    # = T_exit_k[domain_of_dest, U(src_compound), U(dst_compound)]
    T_expanded = T_k_compound[d_indices[None, :],
                              suv_indices[:, None],
                              suv_indices[None, :]]          # (n_body, n_body)

    chi_body_inter = exit_flat[:, None] * T_expanded * dest_flat[None, :]

    # --- Intra-domain block (block-diagonal over domains) ---
    # M-block for all domains: (n_dom, 3, 3) inner tau submatrix
    tau_MID_all = tau_all[:, _MID][:, :, _MID]  # (n_dom, 3, 3)

    # non_ext_M[d, x, f, y, g] = notext[d,f] * tau_MID[d,x,y] * frag_w[d,g]
    non_ext_M_all = jnp.einsum('df,dxy,dg->dxfyg',
                               notext, tau_MID_all, frag_weights
                               ).reshape(n_dom, 3 * n_frag, 3 * n_frag)

    # Extension block: delta(x=y) * ext[d,f,g] for the M-block
    # The M-block has 3F x 3F structure: (MID_type x frag) -> (MID_type x frag)
    # Extension preserves the MID state type (delta(x=y)), so the extension
    # block is block-diagonal of 3 copies of ext[d,:,:].
    ext_M_all = jax.vmap(
        lambda e: jax.scipy.linalg.block_diag(e, e, e)
    )(ext_rates)  # (n_dom, 3*n_frag, 3*n_frag)
    intra_M_all = non_ext_M_all + ext_M_all

    # I-block for all domains: (n_dom, n_frag, n_frag)
    # non_ext_I[d,f,g] = notext[d,f] * kappa_d * frag_w[d,g]
    non_ext_I_all = (notext[:, :, None] *
                     kappas[:, None, None] * frag_weights[:, None, :])
    # ext_I[d,f,g] = ext[d,f,g] (full F x F matrix per domain)
    intra_I_all = non_ext_I_all + ext_rates

    # D-block: same formula as I-block
    intra_D_all = non_ext_I_all + ext_rates

    # Assemble per-domain (5*n_frag, 5*n_frag) blocks
    block_all = jax.vmap(
        lambda m, i, d: jax.scipy.linalg.block_diag(m, i, d)
    )(intra_M_all, intra_I_all, intra_D_all)  # (n_dom, block_size, block_size)

    # Place domain blocks on the block-diagonal via advanced indexing
    local_idx = jnp.arange(n_body) % block_size   # position within domain block
    same_dom = (d_indices[:, None] == d_indices[None, :])  # (n_body, n_body)
    chi_body_intra = (block_all[d_indices[:, None],
                                local_idx[:, None],
                                local_idx[None, :]] * same_dom)

    # --- Combine body ---
    chi_body = chi_body_inter + chi_body_intra

    # --- SS row (using T_exit_k) ---
    # chi[SS, dst] = T_exit_k[d, S, U(uv)] * frag_weights[d, f] * entry_factor[d, uv]
    # (No dom_weights — they're inside T_exit_k)
    ss_T_k = T_exit_k[:, S, :][:, _UV_U]                     # (n_dom, 5)
    ss_body_3d = (ss_T_k[:, :, None] *
                  frag_weights[:, None, :] *
                  entry_factor[:, :, None])                    # (n_dom, 5, n_frag)
    ss_body_flat = ss_body_3d.reshape(n_body)

    # --- EE column (uses full T_eff, unchanged) ---
    # chi[UV_df, EE] = exit_full[d,uv,f] * T[U(uv), E]
    ee_body_3d = exit_full * T[_UV_U, E][None, :, None]     # (n_dom, 5, n_frag)
    ee_body_flat = ee_body_3d.reshape(n_body)

    # --- Assemble full chi via concatenation (no at[].set) ---
    row_ss = jnp.concatenate([jnp.zeros(1), jnp.array([T[S, E]]), ss_body_flat])
    row_ee = jnp.zeros((1, N))
    body_rows = jnp.concatenate([
        jnp.zeros((n_body, 1)),      # col SS (no transitions to SS)
        ee_body_flat[:, None],       # col EE
        chi_body,                     # body-to-body
    ], axis=1)
    chi = jnp.concatenate([row_ss[None, :], row_ee, body_rows], axis=0)

    state_map = _build_state_map(n_dom, n_frag)
    return chi, state_map
