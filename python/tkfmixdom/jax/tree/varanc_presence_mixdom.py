r"""Mixture-of-trees variational MixDom ancestral presence + tuple inference.

See ``tkf/varanc-presence-mixdom.tex`` (appendix M of the main paper) for
the math derivation. Generalises ``varanc_presence.py`` from TKF92 to
MixDom: each (column, internal-node) variational state is in
$\{N, D\} \cup \mathcal{T}$ where $\mathcal{T} = [N_{\text{dom}}] \times
[N_{\text{fr}}]$, with per-column tuple sharing invariant.

Key objects:

- ``mixdom_reduced_T_pair``: per-character marginal Pair HMM at the
  reduced (d, f) state space (eq:T-hat). Built by reshaping the output
  of ``models.mixdom.build_nested_trans``, which is verified against
  the route-sum form in
  ``python/verify_reduced_wfst_routes.py`` to 1.1e-16 at $t = 0.1$.

- ``mixdom_reduced_T_cond``: per-character conditional WFST at reduced
  state space (= Pair HMM / singlet emission), used in the variational
  path LL.

- ``parse_mixdom_params_npz``: load an SVI-BW checkpoint (e.g.
  ``svi_bw_d3f2c3_diag_postfix_best_val.npz``) into a parameter dict
  consumable by the kernel functions.

State conventions (mirror ``varanc_presence.py``):

- WFST states: ``S=0, M=1, I=2, D=3, E=4``.
- Variational Z states (reduced): ``NYI=0, PRESENT=1, DELETED=2``,
  where ``PRESENT`` carries an additional column-wide tuple $\tau \in
  \mathcal{T}$. The inner 3-state irreversible q is identical to the
  TKF92 case; the tuple is a separate per-column free categorical.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from ..models.mixdom import (
    build_nested_trans, effective_trans_per_type, _UV_U,
)
from ..core.params import tkf91_trans
from ..core.bdi import tkf_alpha, tkf_beta, tkf_kappa


WFST_S, WFST_M, WFST_I, WFST_D, WFST_E = 0, 1, 2, 3, 4
N_WFST = 5

NYI, PRESENT, DELETED = 0, 1, 2
N_Z = 3

# Pair HMM compound state indices (per build_nested_trans / mixdom.py:
# UV order is MM=0, MI=1, MD=2, II=3, DD=4).
UV_MM, UV_MI, UV_MD, UV_II, UV_DD = 0, 1, 2, 3, 4

# Mapping from variational (parent_state, child_state) to WFST
# transition type. (NYI, PRESENT) -> WFST insert state, etc.
# The labelled compound state UV in build_nested_trans uses:
#   MM = match-match, MI = match-insert (=> our I),
#   MD = match-delete (=> our D), II = insert-insert (=> our I again
#   if we treat domain-insert as an "I", otherwise it's a separate
#   variational route), DD = delete-delete (=> our D).
# For the variational the simplification is to map the compound UV
# states to the 5-state WFST {S, M, I, D, E} by using the descendant
# (top-level v) only: MM->M, MI->I (descendant insert), MD->D
# (descendant delete), II->I, DD->D.


# ---------------------------------------------------------------------------
# Reduced WFST kernel.
# ---------------------------------------------------------------------------


def mixdom_reduced_T_pair(params, t):
    """Compute the reduced per-character marginal Pair HMM at (d, f) state space.

    Wraps ``models.mixdom.build_nested_trans`` and reshapes its (N, N)
    output into a (5, T, 5, T) tensor where T = N_dom * N_fr is the
    reduced tuple count.

    The Pair HMM joint at reduced state space is:

        T_pair[s, tau, s', tau']
            = P(next compound state = (s', tau') | current = (s, tau))

    with s, s' in {S=0, M=1, I=2, D=3, E=4} and tau = (d, f) flattened
    as ``tau = d * N_fr + f``.

    Args:
        params: dict with keys main_ins, main_del, dom_ins, dom_del,
            dom_weights, frag_weights, ext_rates (per build_nested_trans).
        t: scalar branch length.

    Returns:
        T_pair: (5, T, 5, T) Pair HMM joint at reduced state space.
        Boundary (S row, E col) is at indices (s=S, tau=0) and
        (s=E, tau=0) by convention; these special entries do not
        depend on tau.
    """
    chi, state_map = build_nested_trans(
        params['main_ins'], params['main_del'], t,
        params['dom_ins'], params['dom_del'], params['dom_weights'],
        params['frag_weights'], params['ext_rates'])

    n_dom = params['dom_ins'].shape[0]
    n_frag = params['frag_weights'].shape[1]
    T = n_dom * n_frag
    N = chi.shape[0]
    assert N == 5 * T + 2, f"chi shape {chi.shape} mismatch with 5*{T}+2"

    # Body of chi (rows/cols 2..N-1): laid out as (uv, d, f) with
    # uv ∈ {0..4} = {MM, MI, MD, II, DD}, d ∈ {0..D-1}, f ∈ {0..F-1}.
    # Per state_map: index = 2 + d * 5 * F + uv * F + f.
    body = chi[2:, 2:].reshape(n_dom, 5, n_frag, n_dom, 5, n_frag)
    # Reorder to (uv_src, d_src, f_src, uv_dst, d_dst, f_dst).
    body = jnp.transpose(body, (1, 0, 2, 4, 3, 5))
    # Flatten (d, f) to tau, giving (5, T, 5, T).
    body_flat = body.reshape(5, T, 5, T)

    # Map UV -> WFST state via the descendant (V) component:
    # MM->M (V=M), MI->I (V=I), MD->D (V=D), II->I (V=I), DD->D (V=D).
    # For compound UV the WFST descendant state is _UV_X (V), but the
    # variational ELBO uses the "external" 5-state index where the
    # ancestor side is the source and the descendant side is the dest.
    # In our variational, we treat the per-character WFST state as a
    # SINGLE label that captures the (parent, child) pair via the
    # variational mapping (NYI, P)->I, (P, P)->M, (P, D)->D. So we
    # need to assign each compound UV to a single WFST state index for
    # the purpose of the variational path LL.
    #
    # Convention: WFST state index = the variational mapping's column
    # state. Compound UV components:
    #   MM -> WFST_M (column state matches between parent/child)
    #   MI -> WFST_I (descendant inserted; ancestor was NYI)
    #   MD -> WFST_D (descendant deleted; ancestor was P)
    #   II -> WFST_I (descendant inserts a whole domain)
    #   DD -> WFST_D (ancestor deletes a whole domain)
    #
    # Both II and MI map to WFST_I; both DD and MD map to WFST_D. So
    # the reduced kernel at the variational state space SUMS the
    # contributions from these compound states.
    uv_to_wfst = (WFST_M, WFST_I, WFST_D, WFST_I, WFST_D)

    # Build (5, T, 5, T) reduced kernel by summing compound UV entries
    # that map to the same WFST state.
    T_red_body = jnp.zeros((N_WFST, T, N_WFST, T))
    for uv_src in range(5):
        s_src = uv_to_wfst[uv_src]
        for uv_dst in range(5):
            s_dst = uv_to_wfst[uv_dst]
            T_red_body = T_red_body.at[s_src, :, s_dst, :].add(
                body_flat[uv_src, :, uv_dst, :])

    # Build (5, T, 5, T) full kernel including S row and E column.
    # Convention: T_pair[S, *, s, tau] = chi[STA, body_flat_idx_s_tau]
    # but we need to pool S onto a single row (s_src=S, tau_src ignored).
    T_pair = jnp.zeros((N_WFST, T, N_WFST, T))

    # Body block.
    T_pair = T_pair.at[1:4, :, 1:4, :].set(T_red_body[1:4, :, 1:4, :])

    # S row: chi[0, 2:] gives transitions from STA into body (= entry
    # transitions). Reshape to (D, 5, F), sum over compound UV by the
    # descendant mapping, flatten to (5, T) as the dest.
    s_row_body = chi[0, 2:].reshape(n_dom, 5, n_frag)  # (D, uv, F)
    s_row_red = jnp.zeros((N_WFST, T))
    for uv_dst in range(5):
        s_dst = uv_to_wfst[uv_dst]
        s_row_red = s_row_red.at[s_dst, :].add(
            s_row_body[:, uv_dst, :].reshape(T))
    # Place S row at all source-tuple slots (S has no source tuple,
    # but we place it at tau_src=0 by convention; consumers should index
    # T_pair[S, 0, s_dst, tau_dst]).
    T_pair = T_pair.at[WFST_S, 0, :, :].set(s_row_red)
    # S -> E special entry.
    T_pair = T_pair.at[WFST_S, 0, WFST_E, 0].set(chi[0, 1])

    # E column: chi[2:, 1] gives transitions from body to FIN. Reshape
    # to (D, 5, F), sum over UV by the descendant mapping (in fact for
    # E column the source's compound state matters — pool by source).
    e_col_body = chi[2:, 1].reshape(n_dom, 5, n_frag)  # (D, uv, F)
    e_col_red = jnp.zeros((N_WFST, T))
    for uv_src in range(5):
        s_src = uv_to_wfst[uv_src]
        e_col_red = e_col_red.at[s_src, :].add(
            e_col_body[:, uv_src, :].reshape(T))
    # Place E column at all dest-tuple slots (E has no dest tuple).
    T_pair = T_pair.at[1:4, :, WFST_E, 0].set(e_col_red[1:4, :])

    return T_pair


def mixdom_reduced_T_cond(params, t, eps=1e-30):
    """Compute the reduced conditional WFST = Pair HMM / singlet emission.

    The conditional WFST entry $\\hat T^{cond}_{ss'}((d, f), (d', f'))$ is
    the Pair HMM joint divided by the marginal singlet emission
    $\\omega^{(d, f, d', f')}$ (eq:omega in the appendix).

    For boundary entries (S row, E column), the convention here matches
    the simple TKF92 case: the S->X entries are themselves the WFST
    conditional entry (no division), and X->E entries also.

    Args:
        params: see ``mixdom_reduced_T_pair``.
        t: scalar branch length.
        eps: floor for omega to avoid div-by-zero on impossible routes.

    Returns:
        T_cond: (5, T, 5, T) conditional WFST at reduced state space.
    """
    T_pair = mixdom_reduced_T_pair(params, t)
    omega = mixdom_omega(params)

    n_dom = params['dom_ins'].shape[0]
    n_frag = params['frag_weights'].shape[1]
    T = n_dom * n_frag

    safe_omega = jnp.maximum(omega, eps)
    # Body entries: divide T_pair[s, tau, s', tau'] by omega[tau, tau']
    # for s, s' in {M, I, D}.
    T_cond = jnp.zeros_like(T_pair)
    body_pair = T_pair[1:4, :, 1:4, :]  # (3, T, 3, T)
    body_cond = body_pair / safe_omega[None, :, None, :]
    T_cond = T_cond.at[1:4, :, 1:4, :].set(body_cond)
    # S row and E column: keep as-is (Pair HMM joint = WFST cond for
    # the boundary entries by convention; the singlet's S/FIN transitions
    # are accounted for in the path LL boundary handling).
    T_cond = T_cond.at[WFST_S, 0, :, :].set(T_pair[WFST_S, 0, :, :])
    T_cond = T_cond.at[1:4, :, WFST_E, 0].set(T_pair[1:4, :, WFST_E, 0])
    T_cond = T_cond.at[WFST_S, 0, WFST_E, 0].set(T_pair[WFST_S, 0, WFST_E, 0])

    return T_cond


def mixdom_reduced_T_per_route(params, t):
    """Compute per-route (5, T, 5, T) Pair HMM contributions.

    Decomposes the labelled per-character Pair HMM (build_nested_trans) into
    three additive route contributions:

      * R1 (intra-fragment, g=0): chi entries from the ext-only block
        (ext[d, f, g] * delta(uv_src=uv_dst), only same domain).
      * R2 (new-fragment, same-domain, g=1, e=0): chi entries from the
        non-ext intra-domain block (notext × tau_inner × frag_w).
      * R3 (cross-domain or self-recurrence, g=1, e=1): chi entries from
        the cross-domain (chi_body_inter) block.

    Returns (T_R1, T_R2, T_R3), each of shape (5, T, 5, T) where T = D*F.
    By construction, T_R1 + T_R2 + T_R3 == mixdom_reduced_T_pair(...) at
    body entries.

    Used by tree_vbem.py for exact route attribution of expected counts.
    """
    from ..core.params import S as _S, M as _M, I as _I, D as _D, E as _E
    from ..models.mixdom import (
        effective_trans_per_type, _UV_U, _UV_X, _IS_M_TYPE, _MID,
    )
    from ..core.params import tkf91_trans

    main_lam = params['main_ins']
    main_mu = params['main_del']
    dom_lam = params['dom_ins']
    dom_mu = params['dom_del']
    dom_w = params['dom_weights']
    frag_w = params['frag_weights']
    ext = jnp.asarray(params['ext_rates'])
    if ext.ndim == 2:
        ext = jax.vmap(jnp.diag)(ext)

    n_dom = dom_lam.shape[0]
    n_frag = frag_w.shape[1]
    T = n_dom * n_frag
    n_body = n_dom * 5 * n_frag
    block_size = 5 * n_frag
    d_indices = jnp.arange(n_body) // block_size

    T_exit_k, _T_eff = effective_trans_per_type(
        main_lam, main_mu, t, dom_lam, dom_mu, dom_w)
    tau_all = jax.vmap(lambda lr, dr: tkf91_trans(lr, dr, t))(dom_lam, dom_mu)
    kappas = dom_lam / dom_mu
    notext = 1.0 - ext.sum(axis=-1)

    # ===== R1: intra-fragment ext block =====
    ext_M_all = jax.vmap(
        lambda e: jax.scipy.linalg.block_diag(e, e, e))(ext)  # (D, 3F, 3F)
    chi_R1 = _block_diag_intra_chi(
        ext_M_all, ext, ext, n_dom, n_frag, d_indices, n_body)

    # ===== R2: non-ext intra-domain (new-fragment-same-domain) =====
    tau_MID_all = tau_all[:, _MID][:, :, _MID]
    non_ext_M_all = jnp.einsum(
        'df,dxy,dg->dxfyg', notext, tau_MID_all, frag_w
    ).reshape(n_dom, 3 * n_frag, 3 * n_frag)
    non_ext_I_all = (notext[:, :, None] *
                     kappas[:, None, None] * frag_w[:, None, :])
    chi_R2 = _block_diag_intra_chi(
        non_ext_M_all, non_ext_I_all, non_ext_I_all,
        n_dom, n_frag, d_indices, n_body)

    # ===== R3: cross-domain block (chi_body_inter from build_nested_trans) =====
    beta_d = jnp.maximum(1.0 - tau_all[:, _S, _E], 1e-30)
    entry_M = tau_all[:, _S, :][:, _UV_X] / beta_d[:, None]
    entry_ID = jnp.ones((n_dom, 5))
    entry_factor = jnp.where(_IS_M_TYPE[None, :], entry_M, entry_ID)
    exit_inner_M = tau_all[:, :, _E][:, _UV_X]
    exit_inner_ID = (1.0 - kappas)[:, None] * jnp.ones(5)[None, :]
    exit_inner = jnp.where(_IS_M_TYPE[None, :], exit_inner_M, exit_inner_ID)
    exit_full = notext[:, None, :] * exit_inner[:, :, None]
    exit_flat = exit_full.reshape(n_body)

    dest_factor = (frag_w[:, None, :] * entry_factor[:, :, None])
    dest_flat = dest_factor.reshape(n_body)

    T_k_compound = T_exit_k[:, _UV_U][:, :, _UV_U]
    suv_indices = jnp.tile(jnp.repeat(jnp.arange(5), n_frag), n_dom)
    T_expanded = T_k_compound[d_indices[None, :],
                              suv_indices[:, None],
                              suv_indices[None, :]]
    chi_R3 = exit_flat[:, None] * T_expanded * dest_flat[None, :]

    T_R1 = _reshape_to_reduced(chi_R1, n_dom, n_frag)
    T_R2 = _reshape_to_reduced(chi_R2, n_dom, n_frag)
    T_R3 = _reshape_to_reduced(chi_R3, n_dom, n_frag)
    return T_R1, T_R2, T_R3


def _block_diag_intra_chi(M_block, I_block, D_block, n_dom, n_frag,
                           d_indices, n_body):
    """Assemble intra-domain block-diagonal chi (n_body, n_body) from
    per-block (D, ...) tensors. Mirrors build_nested_trans's chi_body_intra
    construction for a given choice of (M, I, D)-block contents."""
    block_size = 5 * n_frag
    block_all = jax.vmap(
        lambda m, i, d: jax.scipy.linalg.block_diag(m, i, d)
    )(M_block, I_block, D_block)
    local_idx = jnp.arange(n_body) % block_size
    same_dom = (d_indices[:, None] == d_indices[None, :])
    return (block_all[d_indices[:, None],
                       local_idx[:, None],
                       local_idx[None, :]] * same_dom)


def _reshape_to_reduced(chi_body, n_dom, n_frag):
    """Reshape (n_body, n_body) chi_body to (5, T, 5, T) reduced kernel
    via uv_to_wfst pooling (same convention as mixdom_reduced_T_pair)."""
    T = n_dom * n_frag
    body = chi_body.reshape(n_dom, 5, n_frag, n_dom, 5, n_frag)
    body = jnp.transpose(body, (1, 0, 2, 4, 3, 5)).reshape(5, T, 5, T)
    uv_to_wfst = (WFST_M, WFST_I, WFST_D, WFST_I, WFST_D)
    T_red = jnp.zeros((N_WFST, T, N_WFST, T))
    for uv_src in range(5):
        s_src = uv_to_wfst[uv_src]
        for uv_dst in range(5):
            s_dst = uv_to_wfst[uv_dst]
            T_red = T_red.at[s_src, :, s_dst, :].add(
                body[uv_src, :, uv_dst, :])
    return T_red


def mixdom_omega(params):
    """Marginal singlet emission probability $\\omega(d, f, d', f')$ (eq:omega).

    Returns:
        omega: (T, T) array; omega[tau, tau'] = sum over routes R1..R3.
    """
    n_dom = params['dom_ins'].shape[0]
    n_frag = params['frag_weights'].shape[1]
    T = n_dom * n_frag

    main_lam = params['main_ins']
    main_mu = params['main_del']
    dom_lam = params['dom_ins']
    dom_mu = params['dom_del']
    dom_w = params['dom_weights']
    frag_w = params['frag_weights']
    ext = params['ext_rates']

    # Auto-convert MixDom ext (D, F) -> MixDom (D, F, F) diagonal.
    ext = jnp.asarray(ext)
    if ext.ndim == 2:
        ext = jax.vmap(jnp.diag)(ext)

    kappa_main = main_lam / main_mu
    kappa_dom = dom_lam / dom_mu  # (D,)
    notext = 1.0 - ext.sum(axis=-1)  # (D, F)
    emptyseg = (dom_w * (1.0 - kappa_dom)).sum()
    zeta = kappa_main * emptyseg

    # Build omega via outer products.
    # omega(d, f, d', f') = delta(d=d') * [ext[d, f, f'] + notext[d, f] * kappa_d * frag_w[d, f']]
    #                     + notext[d, f] * (1-kappa_d) * kappa_main * dom_w[d'] * kappa_d' * frag_w[d', f'] / (1-zeta)
    omega = jnp.zeros((n_dom, n_frag, n_dom, n_frag))

    # R1 + R2: same domain (d' = d).
    intra_R1 = ext  # (D, F, F): ext[d, f, f']
    intra_R2 = (notext[:, :, None] * kappa_dom[:, None, None] *
                frag_w[:, None, :])  # (D, F, F)
    intra = intra_R1 + intra_R2  # (D, F, F)
    # Place on diagonal (d=d').
    omega = omega.at[jnp.arange(n_dom), :, jnp.arange(n_dom), :].set(intra)

    # R3: cross-domain (any d, d').
    R3 = (notext[:, :, None, None] * (1.0 - kappa_dom)[:, None, None, None] *
          kappa_main * dom_w[None, None, :, None] *
          kappa_dom[None, None, :, None] *
          frag_w[None, None, :, :] / (1.0 - zeta))  # (D, F, D, F)
    omega = omega + R3

    return omega.reshape(T, T)


# ---------------------------------------------------------------------------
# Parameter loader.
# ---------------------------------------------------------------------------


def parse_mixdom_params_npz(path):
    """Load an SVI-BW checkpoint into a params dict for the kernel.

    Args:
        path: path to e.g. ``svi_bw_d3f2c3_diag_postfix_best_val.npz``.

    Returns:
        dict with keys main_ins, main_del, dom_ins, dom_del, dom_weights,
        frag_weights, ext_rates, classdist, class_pis, class_S_exch.
    """
    data = np.load(path, allow_pickle=True)
    params = {
        'main_ins': float(data['main_ins']),
        'main_del': float(data['main_del']),
        'dom_ins': jnp.asarray(data['dom_ins']),
        'dom_del': jnp.asarray(data['dom_del']),
        'dom_weights': jnp.asarray(data['dom_weights']),
        'frag_weights': jnp.asarray(data['frag_weights']),
        'ext_rates': jnp.asarray(data['ext_rates']),
    }
    if 'classdist' in data.files:
        params['classdist'] = jnp.asarray(data['classdist'])
    if 'class_pis' in data.files:
        params['class_pis'] = jnp.asarray(data['class_pis'])
    if 'class_S_exch' in data.files:
        params['class_S_exch'] = jnp.asarray(data['class_S_exch'])
    if 'dom_pis' in data.files:
        params['dom_pis'] = jnp.asarray(data['dom_pis'])
    if 'dom_Qs' in data.files:
        params['dom_Qs'] = jnp.asarray(data['dom_Qs'])
    if 'dom_S_exch' in data.files:
        params['dom_S_exch'] = jnp.asarray(data['dom_S_exch'])

    # If checkpoint has no class layer (e.g. plain dN_F1 models), synthesize
    # a 1-class-per-domain structure: classdist[d, f, c] = delta(c == d),
    # with class_pis and class_S_exch reusing the per-domain matrices.
    if 'classdist' not in params and 'dom_pis' in params:
        n_dom = params['dom_ins'].shape[0]
        n_frag = params['frag_weights'].shape[1]
        eye_D = jnp.eye(n_dom)
        params['classdist'] = jnp.broadcast_to(
            eye_D[:, None, :], (n_dom, n_frag, n_dom))
        params['class_pis'] = params['dom_pis']
        params['class_S_exch'] = params['dom_S_exch']

    return params


# ---------------------------------------------------------------------------
# Variational q over tuples (per-column free categorical).
# ---------------------------------------------------------------------------


def make_tuple_dist(tuple_logits):
    """Build per-column tuple categorical from logits.

    Args:
        tuple_logits: (L, T) free logits over T = N_dom * N_fr tuples.

    Returns:
        q_tau: (L, T) categorical, softmax over T.
    """
    return jax.nn.softmax(tuple_logits, axis=-1)


def fragchar_marginal_from_tuple(q_tau, n_dom, n_frag):
    """Marginalise tuple distribution to fragchar marginal q^{(f)}_n(f).

    Args:
        q_tau: (L, T) tuple categorical.
        n_dom, n_frag: tuple shape.

    Returns:
        q_f: (L, n_frag) fragchar marginal.
    """
    L = q_tau.shape[0]
    return q_tau.reshape(L, n_dom, n_frag).sum(axis=1)


def domain_marginal_from_tuple(q_tau, n_dom, n_frag):
    """Marginalise tuple to domain marginal."""
    L = q_tau.shape[0]
    return q_tau.reshape(L, n_dom, n_frag).sum(axis=2)


# ---------------------------------------------------------------------------
# Per-branch expected indel LL with tuple weighting.
# ---------------------------------------------------------------------------


def expected_branch_LL_mixdom(pair_marg_branch, q_tau, log_T_branch, eps=1e-30):
    """E_q[branch indel LL] using cumulant trick + tuple weighting.

    Per varanc-presence-mixdom.tex eq:E-branch-LL-mixdom-reduced:

        E_q[log P(X^w | X^v)] = sum_{s, s'} sum_{tau, tau'}
            W_{ss', tau tau'} log T_hat_{ss'}(tau, tau')

    with reduced expected counts via the same cumulant prefix trick as
    the simple TreeVarAnc, but with state-prob tensors weighted by the
    per-column tuple categorical.

    Args:
        pair_marg_branch: (L, 3, 3) inner 3-state pairwise marginals per
            column for this branch.
        q_tau: (L, T) tuple categorical per column.
        log_T_branch: (5, T, 5, T) log of reduced WFST kernel for this branch.

    Returns:
        E[L_branch] scalar.
    """
    # Compute inner WFST state probabilities per column (same as simple
    # TreeVarAnc: M = q(P, P), I = q(N, P), D = q(P, D), Ig = rest).
    P_M = pair_marg_branch[:, PRESENT, PRESENT]
    P_I = pair_marg_branch[:, NYI, PRESENT]
    P_D = pair_marg_branch[:, PRESENT, DELETED]
    P_Ig = pair_marg_branch[:, NYI, NYI] + pair_marg_branch[:, DELETED, DELETED]
    L = pair_marg_branch.shape[0]
    T = q_tau.shape[1]

    # Build (L+2, 5) inner state probability with sentinels.
    P_state_inner = jnp.zeros((L + 2, N_WFST))
    P_state_inner = P_state_inner.at[0, WFST_S].set(1.0)
    P_state_inner = P_state_inner.at[L + 1, WFST_E].set(1.0)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_M].set(P_M)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_I].set(P_I)
    P_state_inner = P_state_inner.at[1:L + 1, WFST_D].set(P_D)

    # Tuple-weighted state probability: P_state_tau[M, s, tau].
    # Boundaries (M=0 is S, M=L+1 is E) have tuple = 0 by convention,
    # since S and E rows of the kernel only use tau=0.
    q_tau_full = jnp.zeros((L + 2, T))
    q_tau_full = q_tau_full.at[0, 0].set(1.0)        # boundary tuple
    q_tau_full = q_tau_full.at[L + 1, 0].set(1.0)
    q_tau_full = q_tau_full.at[1:L + 1, :].set(q_tau)
    # P_state_tau[M, s, tau] = P_state_inner[M, s] * q_tau_full[M, tau]
    P_state_tau = P_state_inner[:, :, None] * q_tau_full[:, None, :]

    # P_Ig per column (with boundary = 1 for log = 0 contribution).
    P_Ig_full = jnp.ones(L + 2)
    P_Ig_full = P_Ig_full.at[1:L + 1].set(jnp.maximum(P_Ig, eps))
    log_P_Ig = jnp.log(P_Ig_full)
    C = jnp.cumsum(log_P_Ig)
    C_prev = jnp.concatenate([jnp.zeros(1), C[:-1]])

    # Stable cumulant prefix on (s, tau).
    log_P_tau = jnp.log(jnp.maximum(P_state_tau, eps))   # (L+2, 5, T)
    log_v = log_P_tau - C[:, None, None]
    log_v = jnp.where(P_state_tau > 0, log_v, -jnp.inf)

    def lae_step(carry, x):
        new = jnp.logaddexp(carry, x)
        return new, new

    init = jnp.full((N_WFST, T), -jnp.inf)
    _, cum_log_v = jax.lax.scan(lae_step, init, log_v)
    cum_log_v_prev = jnp.concatenate(
        [jnp.full((1, N_WFST, T), -jnp.inf), cum_log_v[:-1]], axis=0)

    # inner[N, s, tau] = sum_{M < N} P_state_tau[M, s, tau] * exp(C[N-1] - C[M])
    inner = jnp.exp(cum_log_v_prev + C_prev[:, None, None])  # (L+2, 5, T)

    # W[s, tau, s', tau'] = sum_{N >= 1} inner[N, s, tau] * P_state_tau[N, s', tau']
    W = jnp.einsum('nsa,ntb->satb', inner[1:], P_state_tau[1:])

    return jnp.sum(log_T_branch * W)


# ---------------------------------------------------------------------------
# Class-marginalised substitution likelihood (Felsenstein on Fitch subtree).
# ---------------------------------------------------------------------------


def felsenstein_uppass(Q_class, residue_at_leaves, leaf_present, tree,
                       branch_lengths):
    """Per-column Felsenstein up-pass under class-c rate matrix Q.

    For a single class c, computes for each internal node v and each
    residue a: beta_v(a; c) = P(observed leaves descended from v |
    z_v = a, class = c).

    Args:
        Q_class: (20, 20) class-c rate matrix (rows sum to 0 in
            generator form; we'll exponentiate per branch).
        residue_at_leaves: (n_leaves, L) leaf residues (int 0..19) or
            -1 for absent.
        leaf_present: (n_leaves, L) {0, 1}.
        tree: BinaryTree.
        branch_lengths: (n_edges,) precomputed branch lengths.

    Returns:
        L_per_column: (L,) per-column likelihood under class c.
    """
    # Use existing felsenstein machinery; for now this is a placeholder
    # that the benchmark wrapper will fill in. The MixDom sub LL
    # involves running this for each class and then marginalising.
    raise NotImplementedError(
        "felsenstein_uppass: use class_marginalised_sub_LL_per_column "
        "via the benchmark's existing Felsenstein helpers.")


def class_marginalised_sub_LL_per_tuple(L_sub_per_class, classdist):
    r"""Compute log L^{sub,tot}_n(d, f; F_n) = log sum_c classdist[d, f, c] L^sub_n(c; F_n).

    The expected substitution log-LL under the variational tuple
    distribution q^(τ)(d, f) is then
    $\sum_{d, f} q^(τ)_n(d, f) \cdot \mathrm{log\_L\_per\_tuple}[n, d, f]$,
    keeping the joint (d, f) structure properly weighted by `q_tau`
    rather than collapsing classdist over the d axis (which was the
    earlier `class_marginalised_sub_LL_per_column` that mean-averaged
    over d — a Category A averaging-across-latent-variables bug).

    Args:
        L_sub_per_class: (L, n_classes) per-column per-class Felsenstein
            up-pass likelihoods on the Fitch subtree.
        classdist: (n_dom, n_frag, n_classes) class distribution.

    Returns:
        log_L_per_tuple: (L, n_dom, n_frag) per-column per-(d, f) log-likelihoods.
    """
    log_classdist = jnp.log(jnp.maximum(classdist, 1e-30))     # (D, F, C)
    log_L = jnp.log(jnp.maximum(L_sub_per_class, 1e-30))       # (L, C)
    # log_L_per_tuple[n, d, f] = logsumexp_c log_classdist[d, f, c] + log_L[n, c]
    return jax.scipy.special.logsumexp(
        log_classdist[None, :, :, :] + log_L[:, None, None, :], axis=-1)


def class_marginalised_sub_LL_per_column(L_sub_per_class, classdist):
    """DEPRECATED: returns (L, F) by domain-mean of classdist. Use
    class_marginalised_sub_LL_per_tuple instead — the d-mean is a
    Category A averaging-across-latent-variables bias when D > 1 and
    classdist has heterogeneous d-rows.

    Kept for backward compatibility with the inference benchmark that
    only needs a fragchar marginal; new code should use the per-tuple
    version.
    """
    classdist_f = classdist.mean(axis=0)
    log_classdist_f = jnp.log(jnp.maximum(classdist_f, 1e-30))
    log_L = jnp.log(jnp.maximum(L_sub_per_class, 1e-30))
    return jax.scipy.special.logsumexp(
        log_classdist_f[None, :, :] + log_L[:, None, :], axis=-1)


# ---------------------------------------------------------------------------
# Root prior (simplified: stationary tuple + TKF92 presence).
# ---------------------------------------------------------------------------


def singlet_root_log_prior_mixdom(root_dist, q_tau, params, n_dom, n_frag,
                                    eps=1e-30):
    r"""Simplified root log prior under MixDom reduced singlet.

    Approximation: presence prior uses TKF92 with effective kappa
    derived from main + per-domain rates; tuple prior uses stationary
    marginal pi_tau ≈ dom_w * frag_w.

    Args:
        root_dist: (L, 3) per-column root distribution {N, P, D}.
        q_tau: (L, T) per-column tuple categorical.
        params: MixDom params.
        n_dom, n_frag: tuple shape.

    Returns:
        scalar log prior.
    """
    # Effective kappa for presence under MixDom.
    # Use main_kappa as a proxy for "probability sequence is non-empty",
    # plus expected dom_kappa as a within-domain factor.
    main_lam = params['main_ins']
    main_mu = params['main_del']
    kappa_main = main_lam / main_mu
    dom_lam = params['dom_ins']
    dom_mu = params['dom_del']
    kappa_dom = dom_lam / dom_mu
    dom_w = params['dom_weights']
    frag_w = params['frag_weights']
    ext = params['ext_rates']
    if ext.ndim == 2:
        ext = jax.vmap(jnp.diag)(ext)

    # Effective per-character continuation prob (rough analogue of
    # TKF92's `p = ext + (1-ext)*kappa`).
    notext = 1.0 - ext.sum(axis=-1)
    # Average ext diagonal weighted by frag_w.
    p_continue = (frag_w * (1.0 - notext)).sum(axis=-1)  # (D,)
    p_continue_mean = (dom_w * p_continue).sum()
    # Effective kappa for the per-column presence indicator:
    kappa_eff = kappa_main * (dom_w * kappa_dom).sum()

    # Use TKF92-style presence prior with kappa_eff and ext_eff.
    # log P(present at this col | present sequence) = log p_continue.
    # log P(absent at this col, no extension) = log (1 - p_continue).
    # For simplicity, use kappa_eff for each column independently (per-column
    # marginal under stationary singlet).
    log_p_present = jnp.log(jnp.maximum(kappa_eff, eps))
    log_p_absent = jnp.log(jnp.maximum(1.0 - kappa_eff, eps))

    sum_root_P = jnp.sum(root_dist[:, PRESENT])
    sum_root_N = jnp.sum(root_dist[:, NYI])

    presence_term = log_p_present * sum_root_P + log_p_absent * sum_root_N

    # Tuple prior (per-column, conditional on present): pi_tau = dom_w * frag_w.
    pi_tau = (dom_w[:, None] * frag_w).reshape(-1)  # (T,)
    log_pi_tau = jnp.log(jnp.maximum(pi_tau, eps))
    # Expected log tuple prior weighted by P(root_n = present): contributes
    # only when the root is present.
    # Sum over n: q^{(τ)}_n(τ) * P(root_n = P) * log pi_tau(τ).
    tuple_term = jnp.sum(
        root_dist[:, PRESENT, None] * q_tau * log_pi_tau[None, :])

    return presence_term + tuple_term


# ---------------------------------------------------------------------------
# ELBO.
# ---------------------------------------------------------------------------


def elbo_mixdom(logits, root_logits, tuple_logits, leaf_present,
                  tree, params, sub_LL_per_class,
                  left_edge_of_internal, right_edge_of_internal):
    r"""Total variational lower bound under MixDom.

    Per varanc-presence-mixdom.tex Section M:
        ELBO = sum_branches E_q[log P_indel] + sum_cols E_q[log L^sub_tot]
             + log P_root + H[q^(tau)] + H[q^(pi|tau)] + log Z

    where H[q^(pi|tau)] is the inner 3-state graph entropy (same as
    simple TreeVarAnc) and H[q^(tau)] is the per-column tuple categorical
    entropy.

    Args:
        logits: (num_edges, L, 2) inner 3-state q logits.
        root_logits: (L,) root presence logits.
        tuple_logits: (L, T) per-column tuple logits.
        leaf_present: (num_leaves, L) {0, 1} indicators.
        tree: BinaryTree.
        params: MixDom params dict.
        sub_LL_per_class: (L, n_classes) precomputed per-column,
            per-class Felsenstein up-pass likelihoods on the Fitch
            subtree (computed once outside the optimisation loop).
        left_edge_of_internal, right_edge_of_internal: from edge_lookup.

    Returns:
        elbo_total: scalar.
        diagnostics: dict.
    """
    from .varanc_presence import (
        make_q_conditionals, make_root_dist, leaf_clamp_to_beta,
        bp_pair_marginals, entropy_per_column,
    )

    n_dom = params['dom_ins'].shape[0]
    n_frag = params['frag_weights'].shape[1]

    L = logits.shape[1]
    T = tuple_logits.shape[1]

    # Inner 3-state q.
    q_cond = make_q_conditionals(logits)
    root_dist = make_root_dist(root_logits)
    leaf_clamp = leaf_clamp_to_beta(leaf_present)
    pair_marg, log_Z = bp_pair_marginals(
        q_cond, root_dist, leaf_clamp, tree,
        left_edge_of_internal, right_edge_of_internal)

    # Tuple q.
    q_tau = make_tuple_dist(tuple_logits)

    # Per-branch reduced log_T at each branch length.
    edge_lengths = jnp.asarray(tree.edge_length)
    # Build log T_pair per branch (vmap over t).
    def log_T_for(t):
        T_pair = mixdom_reduced_T_pair(params, t)
        return jnp.log(jnp.maximum(T_pair, 1e-300))

    log_T_per_edge = jax.vmap(log_T_for)(edge_lengths)  # (E, 5, T, 5, T)

    # Sum E[branch LL] over edges.
    branch_LLs = jax.vmap(
        lambda pm, lt: expected_branch_LL_mixdom(pm, q_tau, lt))(
            pair_marg, log_T_per_edge)
    sum_branch_LL = jnp.sum(branch_LLs)

    # Per-column substitution log-likelihood expectation.
    # Use the per-tuple version that keeps the (d, f) joint structure
    # (the .mean(axis=0) in the per-fragchar version is a Category A
    # averaging-across-latent-variables bias when D > 1).
    log_L_per_tuple = class_marginalised_sub_LL_per_tuple(
        sub_LL_per_class, params['classdist'])  # (L, D, F)
    q_tau_d = q_tau.reshape(-1, n_dom, n_frag)  # (L, D, F)
    sum_sub_LL = jnp.sum(q_tau_d * log_L_per_tuple)

    # Inner 3-state entropy (same machinery).
    H_per_col_inner = entropy_per_column(
        pair_marg, root_dist, beta_root=None,
        node_marg_internal=None, q_cond=q_cond, tree=tree)
    H_inner_total = jnp.sum(H_per_col_inner)

    # Tuple categorical entropy.
    log_q_tau = jnp.log(jnp.maximum(q_tau, 1e-30))
    H_tau_per_col = -jnp.sum(q_tau * log_q_tau, axis=-1)  # (L,)
    H_tau_total = jnp.sum(H_tau_per_col)

    # Root prior.
    log_prior_root = singlet_root_log_prior_mixdom(
        root_dist, q_tau, params, n_dom, n_frag)

    sum_log_Z = jnp.sum(log_Z)
    elbo_total = (sum_branch_LL + sum_sub_LL + log_prior_root +
                  H_inner_total + H_tau_total + sum_log_Z)

    return elbo_total, {
        'sum_branch_LL': sum_branch_LL,
        'sum_sub_LL': sum_sub_LL,
        'log_prior_root': log_prior_root,
        'entropy_inner': H_inner_total,
        'entropy_tau': H_tau_total,
        'log_Z': sum_log_Z,
        'pair_marg': pair_marg,
        'q_tau': q_tau,
        'q_f': fragchar_marginal_from_tuple(q_tau, n_dom, n_frag),
    }

