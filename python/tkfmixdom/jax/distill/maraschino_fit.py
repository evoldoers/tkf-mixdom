"""MixDom2 cherry-count fit (Maraschino fit-mode rewrite).

Implements the MixDom2 cherry-count log-likelihood per
``tkf/maraschino.tex`` § sec:maraschino-fit:

    L_cherry(theta) = L_singlet(theta) + sum_b L_pair,b(theta)

Where, for each tau-bin b, the model-side adjacency frequencies F^{uv}_b
are derived from the collapsed Pair HMM transition matrix
``chi = build_nested_trans(...)`` of MixDom2 and the per-(d, f)
class-mixture emissions

    phi^M_{d,f}(X, Y) = sum_c classdist[d,f,c] * pi^c[X] * P^c(t)[X,Y]
    phi^I_{d,f}(Y)   = phi^D_{d,f}(Y) = sum_c classdist[d,f,c] * pi^c[Y]

Counts are scored against the row-normalised conditional probabilities.

This module owns the *fit-mode* MixDom2 likelihood; the legacy
``distill/maraschino.py`` (MixDom1-style distill) is left intact and is
the production order-1 distillation path. The two share neither
``constrain_params`` nor the cherry-count likelihood implementation.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.models.mixdom import build_nested_trans

AA = 20
S_IDX, M_IDX, I_IDX, D_IDX, E_IDX = 0, 1, 2, 3, 4
# UV ordering used by build_nested_trans body states: MM, MI, MD, II, DD
UV_MM, UV_MI, UV_MD, UV_II, UV_DD = 0, 1, 2, 3, 4
UV_IDS = ("MM", "MI", "MD", "II", "DD")
COUNT_KEYS = ("MM", "MI", "MD", "IM", "II", "ID", "DM", "DD", "DI",
              "SM", "SI", "SD", "ME", "IE", "DE", "SE")


# ============================================================
# Linear / unconstrained parameter conversion
# ============================================================
def linear_to_raw(linear_params: dict, n_dom: int, n_frag: int,
                  n_classes: int) -> dict:
    """Convert linear-space MixDom2 params to unconstrained (raw) params.

    Inverse of :func:`raw_to_linear` (modulo softmax shift invariance).

    Linear-space keys (as produced by :func:`init_mixdom2_params_from_args`)::

        main_ins (scalar), main_del (scalar)
        dom_ins[D], dom_del[D]
        dom_weights[D], frag_weights[D, F]
        ext_rates[D, F, F]              (row sums ≤ 1)
        class_pis[C, A], class_S_exch[C, A, A], classdist[D, F, C]

    Raw-space keys (returned)::

        log_main_ins, log_main_del      (log of positive scalar)
        log_dom_ins[D], log_dom_del[D]
        logit_dom_weights[D]            (softmax over D → dom_weights)
        logit_frag_weights[D, F]        (softmax over F)
        logit_ext_rates[D, F, F+1]      (softmax over last axis;
                                         first F slots → ext_rates,
                                         last slot → notext = 1 - row sum)
        log_class_pis[C, A]             (softmax over A)
        raw_class_S_lower[C, K]         (lower-triangular off-diagonal, K = A*(A-1)/2,
                                         linear; symmetrised on constrain)
        logit_classdist[D, F, C]        (softmax over C)
    """
    raw: dict = {}
    eps = 1e-30

    raw["log_main_ins"] = jnp.log(jnp.asarray(float(linear_params["main_ins"]), dtype=jnp.float32))
    raw["log_main_del"] = jnp.log(jnp.asarray(float(linear_params["main_del"]), dtype=jnp.float32))

    raw["log_dom_ins"] = jnp.log(jnp.asarray(linear_params["dom_ins"], dtype=jnp.float32) + eps)
    raw["log_dom_del"] = jnp.log(jnp.asarray(linear_params["dom_del"], dtype=jnp.float32) + eps)

    raw["logit_dom_weights"] = jnp.log(jnp.asarray(linear_params["dom_weights"], dtype=jnp.float32) + eps)
    raw["logit_frag_weights"] = jnp.log(jnp.asarray(linear_params["frag_weights"], dtype=jnp.float32) + eps)

    # ext_rates -> logit (with notext as the F+1-th slot for softmax)
    ext = jnp.asarray(linear_params["ext_rates"], dtype=jnp.float32)
    notext = 1.0 - ext.sum(axis=-1)  # (D, F)
    ext_full = jnp.concatenate([ext, notext[:, :, None]], axis=-1)  # (D, F, F+1)
    raw["logit_ext_rates"] = jnp.log(ext_full + eps)

    if n_classes > 1:
        raw["log_class_pis"] = jnp.log(jnp.asarray(linear_params["class_pis"], dtype=jnp.float32) + eps)
        # Symmetric S_exch with zero diagonal: store strict lower triangle entries (linear scale)
        Sx = jnp.asarray(linear_params["class_S_exch"], dtype=jnp.float32)  # (C, A, A)
        tril_idx = jnp.tril_indices(AA, k=-1)
        # Average over the (i,j) and (j,i) entries to enforce symmetry on input
        S_sym = (Sx + jnp.swapaxes(Sx, -2, -1)) * 0.5
        raw["raw_class_S_lower"] = S_sym[:, tril_idx[0], tril_idx[1]]  # (C, AA*(AA-1)/2)
        raw["logit_classdist"] = jnp.log(jnp.asarray(linear_params["classdist"], dtype=jnp.float32) + eps)

    return raw


def raw_to_linear(raw: dict, n_dom: int, n_frag: int, n_classes: int,
                  freeze_init: dict | None = None,
                  freeze_main_rates: bool = False) -> dict:
    """Convert raw (unconstrained) params back to linear-space MixDom2 params.

    Args:
        raw: dict produced by :func:`linear_to_raw` (after gradient updates).
            For the rate-rescale mode (``class_S_exch_shape`` in
            ``freeze_init``), an extra raw key ``log_class_sigma`` of shape
            ``(C,)`` must be present; ``class_S_exch[c] = exp(log_sigma[c]) *
            class_S_exch_shape[c]`` then carries all per-class S information,
            with the per-entry ``raw_class_S_lower`` ignored (gradient stops
            naturally because the call site does not use it).
        n_dom, n_frag, n_classes: model shape.
        freeze_init: optional dict with linear-space initial values for any
            of ``classdist``, ``frag_weights``, ``ext_rates``, ``class_pis``,
            and ``class_S_exch_shape``. When ``class_S_exch_shape`` is
            present, the rescale parameterisation above is used. When
            ``class_pis`` is present, π is held at the supplied value.

    Returns linear params with keys matching what
    :func:`init_mixdom2_params_from_args` produces.
    """
    softmax = lambda x: jax.nn.softmax(x, axis=-1)

    out: dict = {}
    if freeze_main_rates and freeze_init is not None and \
            'main_ins' in freeze_init and 'main_del' in freeze_init:
        # Hold main_ins / main_del at supplied init (typically very small,
        # so the top-level TKF91 contributes ~0 cross-component switching
        # within a family — useful when each component is meant to be a
        # single-domain model).
        out["main_ins"] = jnp.asarray(freeze_init['main_ins'],
                                       dtype=jnp.float32)
        out["main_del"] = jnp.asarray(freeze_init['main_del'],
                                       dtype=jnp.float32)
    else:
        out["main_ins"] = jnp.exp(raw["log_main_ins"])
        out["main_del"] = jnp.maximum(jnp.exp(raw["log_main_del"]),
                                       out["main_ins"] + 1e-6)

    out["dom_ins"] = jnp.exp(raw["log_dom_ins"])
    out["dom_del"] = jnp.maximum(jnp.exp(raw["log_dom_del"]), out["dom_ins"] + 1e-6)

    out["dom_weights"] = softmax(raw["logit_dom_weights"])
    if freeze_init is not None and freeze_init.get("frag_weights") is not None:
        out["frag_weights"] = jnp.asarray(freeze_init["frag_weights"], dtype=jnp.float32)
    else:
        out["frag_weights"] = softmax(raw["logit_frag_weights"])

    if freeze_init is not None and freeze_init.get("ext_rates") is not None:
        out["ext_rates"] = jnp.asarray(freeze_init["ext_rates"], dtype=jnp.float32)
    else:
        # softmax over (F+1) → first F slots are ext_rates rows, last is notext
        ext_full = softmax(raw["logit_ext_rates"])  # (D, F, F+1)
        out["ext_rates"] = ext_full[..., :n_frag]

    if n_classes > 1:
        # ----- π_c handling -----
        if freeze_init is not None and freeze_init.get("class_pis") is not None:
            # π frozen at supplied init (rate-rescale mode or freeze-π mode).
            out["class_pis"] = jnp.asarray(freeze_init["class_pis"],
                                           dtype=jnp.float32)
        else:
            out["class_pis"] = softmax(raw["log_class_pis"])

        # ----- S^c handling -----
        if freeze_init is not None and freeze_init.get("class_S_exch_shape") is not None:
            # Rate-rescale mode: S_c = exp(log_sigma_c) * S_shape_c.
            # Only log_class_sigma carries gradient; raw_class_S_lower is
            # ignored at this call site (the optimiser sees no signal on it).
            S_shape = jnp.asarray(freeze_init["class_S_exch_shape"],
                                   dtype=jnp.float32)
            log_sigma = raw["log_class_sigma"]   # (C,)
            sigma = jnp.exp(log_sigma)            # (C,)
            S = sigma[:, None, None] * S_shape
        else:
            # Standard: reconstruct symmetric S_exch from lower triangle
            tril_idx = jnp.tril_indices(AA, k=-1)
            S = jnp.zeros((n_classes, AA, AA), dtype=jnp.float32)
            S = S.at[:, tril_idx[0], tril_idx[1]].set(raw["raw_class_S_lower"])
            S = S + jnp.swapaxes(S, -2, -1)  # symmetrise (diagonal stays 0)
            # Enforce non-negativity (the optimizer may drive entries slightly < 0)
            S = jnp.maximum(S, 0.0)
        out["class_S_exch"] = S

        if freeze_init is not None and freeze_init.get("classdist") is not None:
            out["classdist"] = jnp.asarray(freeze_init["classdist"], dtype=jnp.float32)
        else:
            out["classdist"] = softmax(raw["logit_classdist"])
        out["n_classes"] = n_classes

    return out


# ============================================================
# Per-class transition kernels
# ============================================================
def _per_class_P(class_pis, class_S_exch, t):
    """Compute per-class P^c(t) = exp(R^c * t) where R^c = S^c diag(pi^c)."""

    def one_class(pi_c, S_c):
        Q_off = S_c * pi_c[None, :]  # off-diagonal of rate matrix
        Q = Q_off - jnp.diag(jnp.diag(Q_off))  # zero diagonal
        Q = Q - jnp.diag(Q.sum(axis=1))  # set diagonal so rows sum to 0
        return transition_matrix(Q, t)

    return jax.vmap(one_class)(class_pis, class_S_exch)  # (C, A, A)


def _emit_tensors(class_pis, class_S_exch, classdist, t):
    """Build per-(d, f) match and singlet emission tensors at time t.

    Returns:
        Emat: (D, F, A, A) match emission (anc X, desc Y)
              Emat[d, f, X, Y] = sum_c classdist[d,f,c] * pi^c[X] * P^c(t)[X,Y]
        Esng: (D, F, A) singlet emission character (single-tape)
              Esng[d, f, X] = sum_c classdist[d,f,c] * pi^c[X]
    """
    P = _per_class_P(class_pis, class_S_exch, t)  # (C, A, A)
    pi_P = class_pis[:, :, None] * P  # (C, A, A) -- row x: pi^c[X] * P^c[X,Y]
    Emat = jnp.einsum("dfc,cxy->dfxy", classdist, pi_P)
    Esng = jnp.einsum("dfc,ca->dfa", classdist, class_pis)
    return Emat, Esng


# ============================================================
# Visit-count vector from SS in chi
# ============================================================
def _visit_counts(chi):
    """Expected number of visits to each state from SS in the absorbing
    Markov chain on chi (EE absorbing).

    Returns:
        visits: (N,) where visits[s] = expected #(visits to s) starting at SS,
                until absorption at EE.
    """
    N = chi.shape[0]
    # Treat EE (index 1) as absorbing: zero its outgoing row.
    Q = chi.at[1, :].set(0.0)
    # Solve (I - Q^T) p = e_SS for the visit row of (I - Q)^{-1}, i.e. the SS row.
    # Equivalently, solve (I - Q^T) v = e_SS giving v[s] = N[SS, s].
    e_ss = jnp.zeros(N).at[0].set(1.0)
    visits = jnp.linalg.solve(jnp.eye(N) - Q.T, e_ss)
    return visits


# ============================================================
# Adjacency frequencies F^{uv} for one tau bin
# ============================================================
def _adjacency_freqs(linear_params, t, n_dom: int, n_frag: int, n_classes: int):
    """Compute the model's per-(u,v) adjacency frequency tensors at time t.

    Returns a dict of frequency tensors with the same axis convention as
    the count tensors (ancestor first, descendant second; only the
    relevant character axes for each adjacency type).
    """
    F = n_frag
    D = n_dom
    chi, _ = build_nested_trans(
        linear_params["main_ins"], linear_params["main_del"], t,
        linear_params["dom_ins"], linear_params["dom_del"],
        linear_params["dom_weights"],
        linear_params["frag_weights"], linear_params["ext_rates"])
    # chi: (N, N) where N = 2 + 5*D*F
    n_body = 5 * D * F
    # body slice
    chi_body = chi[2:, 2:].reshape(D, 5, F, D, 5, F)
    chi_ss = chi[0, 2:].reshape(D, 5, F)              # SS -> body
    chi_to_ee = chi[2:, 1].reshape(D, 5, F)           # body -> EE
    chi_ss_ee = chi[0, 1]                             # SS -> EE

    # Visit counts from SS
    visits = _visit_counts(chi)
    visits_body = visits[2:].reshape(D, 5, F)         # (D, 5, F)
    # visits[0] is # visits to SS = 1 (start state, never re-entered if chi[*, SS] = 0).

    # Class-mixture emissions
    if n_classes > 1:
        Emat, Esng = _emit_tensors(
            linear_params["class_pis"], linear_params["class_S_exch"],
            linear_params["classdist"], t)
        # Emat: (D, F, A, A), Esng: (D, F, A)
    else:
        # Fallback: use dom_pis if present; else uniform pi
        pi = linear_params.get("dom_pis", jnp.ones((D, AA)) / AA)
        # No per-domain Q: cannot compute proper match emission without sub model.
        # In MixDom2 we always have n_classes >= 1; treat n_classes == 1 as a
        # single class with class_pis[0] = pi_lg, class_S_exch[0] = LG.
        raise ValueError(
            "n_classes must be >= 1 with class_pis/class_S_exch/classdist set; "
            "got n_classes=1 without class params")

    # Indices of u-states in body
    M_state = UV_MM   # 0
    I_states = (UV_MI, UV_II)  # 1, 3
    D_states = (UV_MD, UV_DD)  # 2, 4

    # ===== Build per-(u, d, f) outgoing emission factors =====
    # For source emission with character context (X, Y):
    #   - MM_{d,f}: emit (X, Y) with prob Emat[d, f, X, Y]
    #   - MI_{d,f}, II_{d,f}: emit only Y with prob Esng[d, f, Y]; X is propagated
    #   - MD_{d,f}, DD_{d,f}: emit only X with prob Esng[d, f, X]; Y is propagated
    #
    # For destination emission (same per type) — used for incoming chars (X', Y').

    # ===== Pair adjacencies: F^{uv}(X, Y; X', Y') =====
    # We compute the joint frequency of (s_1, s_2) pairs weighted by their
    # emissions. Sums are over (d_1, f_1) latent for source and (d_2, f_2)
    # latent for destination.
    #
    # Indices: chi_body[d_1, uv_1, f_1, d_2, uv_2, f_2].

    out: dict[str, jnp.ndarray] = {}

    # Helper: weighted sum over (d_1, f_1) and (d_2, f_2) of
    #   visits[d_1, uv_1, f_1] * chi_body[d_1, uv_1, f_1, d_2, uv_2, f_2]
    # contracted with destination emission to give a (..., chars_dst) tensor,
    # then contracted with source emission to give a (chars_src, chars_dst).

    # Letters in einsums:
    #   a, b = source (d1, f1)
    #   c, e = destination (d2, f2)
    #   p, q = source ancestor / descendant character
    #   r, s = destination ancestor / descendant character

    # === Match-sourced (uv_1 = MM) ===
    Vsrc_M = visits_body[:, M_state, :]  # (D, F)

    # f^MM(p, q, r, s) = sum_{abce} V[a,b] * Emat[a,b,p,q] * chi[a, MM, b, c, MM, e] * Emat[c,e,r,s]
    chi_MM = chi_body[:, M_state, :, :, M_state, :]  # (D, F, D, F)
    T_M_to_M = Vsrc_M[:, :, None, None] * chi_MM     # (D, F, D, F)
    out["MM"] = jnp.einsum("abce,abpq,cers->pqrs",
                           T_M_to_M, Emat, Emat,
                           optimize=True)

    # f^MI(p, q, s) = sum * Esng[c, e, s]   (insert destination emits descendant s)
    chi_M_to_I = (chi_body[:, M_state, :, :, UV_MI, :]
                  + chi_body[:, M_state, :, :, UV_II, :])  # (D, F, D, F)
    T_M_to_I = Vsrc_M[:, :, None, None] * chi_M_to_I
    out["MI"] = jnp.einsum("abce,abpq,ces->pqs",
                           T_M_to_I, Emat, Esng,
                           optimize=True)

    # f^MD(p, q, r) = sum * Esng[c, e, r]   (delete destination emits ancestor r)
    chi_M_to_D = (chi_body[:, M_state, :, :, UV_MD, :]
                  + chi_body[:, M_state, :, :, UV_DD, :])
    T_M_to_D = Vsrc_M[:, :, None, None] * chi_M_to_D
    out["MD"] = jnp.einsum("abce,abpq,cer->pqr",
                           T_M_to_D, Emat, Esng,
                           optimize=True)

    # f^ME(p, q) = sum_{ab} V[a,b] * Emat[a,b,p,q] * chi_to_ee[a, MM, b]
    out["ME"] = jnp.einsum("ab,abpq,ab->pq",
                           Vsrc_M, Emat, chi_to_ee[:, M_state, :],
                           optimize=True)

    # === Insert-sourced (uv_1 in {MI, II}) ===
    # Source emits descendant q; ancestor is propagated and we marginalise
    # over it (counts only carry q).
    Vsrc_I = visits_body[:, UV_MI, :] + visits_body[:, UV_II, :]  # (D, F)

    chi_I_to_M = chi_body[:, UV_MI, :, :, M_state, :] + chi_body[:, UV_II, :, :, M_state, :]
    T_I_to_M = Vsrc_I[:, :, None, None] * chi_I_to_M
    out["IM"] = jnp.einsum("abce,abq,cers->qrs",
                           T_I_to_M, Esng, Emat,
                           optimize=True)

    chi_I_to_I = (chi_body[:, UV_MI, :, :, UV_MI, :] + chi_body[:, UV_MI, :, :, UV_II, :]
                  + chi_body[:, UV_II, :, :, UV_MI, :] + chi_body[:, UV_II, :, :, UV_II, :])
    T_I_to_I = Vsrc_I[:, :, None, None] * chi_I_to_I
    out["II"] = jnp.einsum("abce,abq,ces->qs",
                           T_I_to_I, Esng, Esng,
                           optimize=True)

    chi_I_to_D = (chi_body[:, UV_MI, :, :, UV_MD, :] + chi_body[:, UV_MI, :, :, UV_DD, :]
                  + chi_body[:, UV_II, :, :, UV_MD, :] + chi_body[:, UV_II, :, :, UV_DD, :])
    T_I_to_D = Vsrc_I[:, :, None, None] * chi_I_to_D
    out["ID"] = jnp.einsum("abce,abq,cer->qr",
                           T_I_to_D, Esng, Esng,
                           optimize=True)

    out["IE"] = jnp.einsum("ab,abq,ab->q",
                           Vsrc_I, Esng,
                           chi_to_ee[:, UV_MI, :] + chi_to_ee[:, UV_II, :],
                           optimize=True)

    # === Delete-sourced (uv_1 in {MD, DD}) ===
    # Source emits ancestor p; descendant is propagated and we marginalise
    # over it (counts only carry p).
    Vsrc_D = visits_body[:, UV_MD, :] + visits_body[:, UV_DD, :]

    chi_D_to_M = chi_body[:, UV_MD, :, :, M_state, :] + chi_body[:, UV_DD, :, :, M_state, :]
    T_D_to_M = Vsrc_D[:, :, None, None] * chi_D_to_M
    out["DM"] = jnp.einsum("abce,abp,cers->prs",
                           T_D_to_M, Esng, Emat,
                           optimize=True)

    chi_D_to_D = (chi_body[:, UV_MD, :, :, UV_MD, :] + chi_body[:, UV_MD, :, :, UV_DD, :]
                  + chi_body[:, UV_DD, :, :, UV_MD, :] + chi_body[:, UV_DD, :, :, UV_DD, :])
    T_D_to_D = Vsrc_D[:, :, None, None] * chi_D_to_D
    out["DD"] = jnp.einsum("abce,abp,cer->pr",
                           T_D_to_D, Esng, Esng,
                           optimize=True)

    chi_D_to_I = (chi_body[:, UV_MD, :, :, UV_MI, :] + chi_body[:, UV_MD, :, :, UV_II, :]
                  + chi_body[:, UV_DD, :, :, UV_MI, :] + chi_body[:, UV_DD, :, :, UV_II, :])
    T_D_to_I = Vsrc_D[:, :, None, None] * chi_D_to_I
    out["DI"] = jnp.einsum("abce,abp,ces->ps",
                           T_D_to_I, Esng, Esng,
                           optimize=True)

    out["DE"] = jnp.einsum("ab,abp,ab->p",
                           Vsrc_D, Esng,
                           chi_to_ee[:, UV_MD, :] + chi_to_ee[:, UV_DD, :],
                           optimize=True)

    # === Start-sourced ===
    # f^SM(r, s) = sum_{c, e} chi[SS, c, MM, e] * Emat[c, e, r, s]
    out["SM"] = jnp.einsum("ce,cers->rs",
                           chi_ss[:, M_state, :], Emat,
                           optimize=True)
    out["SI"] = jnp.einsum("ce,ces->s",
                           chi_ss[:, UV_MI, :] + chi_ss[:, UV_II, :], Esng,
                           optimize=True)
    out["SD"] = jnp.einsum("ce,cer->r",
                           chi_ss[:, UV_MD, :] + chi_ss[:, UV_DD, :], Esng,
                           optimize=True)
    out["SE"] = chi_ss_ee

    return out


# ============================================================
# Cherry log-likelihood: pair LL at one tau, full LL across tau
# ============================================================
def _pair_ll_one_tau(linear_params, t, counts_t,
                     n_dom: int, n_frag: int, n_classes: int):
    """Compute the pair log-likelihood term for a single tau bin."""
    F = _adjacency_freqs(linear_params, t, n_dom, n_frag, n_classes)
    eps = 1e-30

    # === Post-Match ===
    # Z_M(X, Y) = sum_{X', Y'} F^MM(X, Y, X', Y') + sum_{Y'} F^MI(X, Y, Y')
    #             + sum_{X'} F^MD(X, Y, X') + F^ME(X, Y)
    f_mm = jnp.maximum(F["MM"], eps)
    f_mi = jnp.maximum(F["MI"], eps)
    f_md = jnp.maximum(F["MD"], eps)
    f_me = jnp.maximum(F["ME"], eps)
    Z_M = (f_mm.sum(axis=(2, 3)) + f_mi.sum(axis=2) + f_md.sum(axis=2) + f_me)
    Z_M = jnp.maximum(Z_M, eps)
    log_ZM = jnp.log(Z_M)
    ll = (jnp.sum(counts_t["MM"] * (jnp.log(f_mm) - log_ZM[:, :, None, None]))
          + jnp.sum(counts_t["MI"] * (jnp.log(f_mi) - log_ZM[:, :, None]))
          + jnp.sum(counts_t["MD"] * (jnp.log(f_md) - log_ZM[:, :, None]))
          + jnp.sum(counts_t["ME"] * (jnp.log(f_me) - log_ZM)))

    # === Post-Insert (reduced context: only Y) ===
    # The model F^I* tensors carry Y as the source character.
    f_im = jnp.maximum(F["IM"], eps)  # (Y, X', Y')
    f_ii = jnp.maximum(F["II"], eps)  # (Y, Y')
    f_id = jnp.maximum(F["ID"], eps)  # (Y, X')
    f_ie = jnp.maximum(F["IE"], eps)  # (Y,)
    Z_I = (f_im.sum(axis=(1, 2)) + f_ii.sum(axis=1) + f_id.sum(axis=1) + f_ie)
    Z_I = jnp.maximum(Z_I, eps)
    log_ZI = jnp.log(Z_I)
    ll += (jnp.sum(counts_t["IM"] * (jnp.log(f_im) - log_ZI[:, None, None]))
           + jnp.sum(counts_t["II"] * (jnp.log(f_ii) - log_ZI[:, None]))
           + jnp.sum(counts_t["ID"] * (jnp.log(f_id) - log_ZI[:, None]))
           + jnp.sum(counts_t["IE"] * (jnp.log(f_ie) - log_ZI)))

    # === Post-Delete (reduced context: only X) ===
    f_dm = jnp.maximum(F["DM"], eps)  # (X, X', Y')
    f_dd = jnp.maximum(F["DD"], eps)  # (X, X')
    f_di = jnp.maximum(F["DI"], eps)  # (X, Y')
    f_de = jnp.maximum(F["DE"], eps)  # (X,)
    Z_D = (f_dm.sum(axis=(1, 2)) + f_dd.sum(axis=1) + f_di.sum(axis=1) + f_de)
    Z_D = jnp.maximum(Z_D, eps)
    log_ZD = jnp.log(Z_D)
    ll += (jnp.sum(counts_t["DM"] * (jnp.log(f_dm) - log_ZD[:, None, None]))
           + jnp.sum(counts_t["DD"] * (jnp.log(f_dd) - log_ZD[:, None]))
           + jnp.sum(counts_t["DI"] * (jnp.log(f_di) - log_ZD[:, None]))
           + jnp.sum(counts_t["DE"] * (jnp.log(f_de) - log_ZD)))

    # === Start ===
    f_sm = jnp.maximum(F["SM"], eps)  # (X', Y')
    f_si = jnp.maximum(F["SI"], eps)  # (Y',)
    f_sd = jnp.maximum(F["SD"], eps)  # (X',)
    f_se = jnp.maximum(F["SE"], eps)  # ()
    Z_S = jnp.maximum(f_sm.sum() + f_si.sum() + f_sd.sum() + f_se, eps)
    log_ZS = jnp.log(Z_S)
    ll += (jnp.sum(counts_t["SM"] * (jnp.log(f_sm) - log_ZS))
           + jnp.sum(counts_t["SI"] * (jnp.log(f_si) - log_ZS))
           + jnp.sum(counts_t["SD"] * (jnp.log(f_sd) - log_ZS))
           + jnp.sum(counts_t["SE"] * (jnp.log(f_se) - log_ZS)))

    return ll


def cherry_log_likelihood(linear_params, counts, tau_centers,
                          n_dom: int, n_frag: int, n_classes: int):
    """Total MixDom2 cherry-count log-likelihood.

    Args:
        linear_params: dict with keys main_ins, main_del, dom_ins[D],
            dom_del[D], dom_weights[D], frag_weights[D, F], ext_rates[D, F, F],
            class_pis[C, A], class_S_exch[C, A, A], classdist[D, F, C].
        counts: dict mapping COUNT_KEY -> jax array (with leading n_tau axis),
            plus 'B' for bigram singlet counts (n_tau, 22, 22).
        tau_centers: (n_tau,) tau-bin centers.

    Returns:
        Scalar log-likelihood.
    """
    n_tau = tau_centers.shape[0]

    # --- Pair LL (vmapped over tau) ---
    def pair_ll_t(t_idx):
        counts_t = {key: counts[f"C_{key}"][t_idx] for key in COUNT_KEYS}
        return _pair_ll_one_tau(linear_params, tau_centers[t_idx], counts_t,
                                n_dom, n_frag, n_classes)

    pair_lls = jax.vmap(pair_ll_t)(jnp.arange(n_tau))
    total = jnp.sum(pair_lls)

    return total


# ============================================================
# MixDom2 order-1 distillation: full-context adjacency frequencies
# and order-1 Pair WFST / Singlet HMM transition probabilities.
# ============================================================
def adjacency_freqs_full(linear_params, t, n_dom: int, n_frag: int,
                         n_classes: int):
    """MixDom2 adjacency frequencies with FULL propagated context.

    Like :func:`_adjacency_freqs` but does NOT marginalise the carried
    ancestor (for I-source) or descendant (for D-source) — the WFST
    representation of an order-1 transducer needs both characters in
    every context. The match-source rows are unchanged (already 4D).

    Returns dict with keys:
      MM (A, A, A, A), MI (A, A, A), MD (A, A, A), ME (A, A)
      IM (A, A, A, A), II (A, A, A), ID (A, A, A), IE (A, A)
      DM (A, A, A, A), DD (A, A, A), DI (A, A, A), DE (A, A)
      SM (A, A), SI (A,), SD (A,), SE scalar

    Indexing:
      For a state with prev (X, Y) and emission (X', Y') (or single-char
      where applicable), the tensor entry is the expected adjacency
      frequency under the MixDom2 collapsed Pair HMM at evol time t.
    """
    F = n_frag
    D = n_dom
    chi, _ = build_nested_trans(
        linear_params["main_ins"], linear_params["main_del"], t,
        linear_params["dom_ins"], linear_params["dom_del"],
        linear_params["dom_weights"],
        linear_params["frag_weights"], linear_params["ext_rates"])
    chi_body = chi[2:, 2:].reshape(D, 5, F, D, 5, F)
    chi_ss = chi[0, 2:].reshape(D, 5, F)
    chi_to_ee = chi[2:, 1].reshape(D, 5, F)
    chi_ss_ee = chi[0, 1]

    visits = _visit_counts(chi)
    visits_body = visits[2:].reshape(D, 5, F)

    Emat, Esng = _emit_tensors(
        linear_params["class_pis"], linear_params["class_S_exch"],
        linear_params["classdist"], t)

    out: dict[str, jnp.ndarray] = {}

    # ---- Match-sourced rows (4D match emission) ----
    Vsrc_M = visits_body[:, UV_MM, :]

    chi_MM = chi_body[:, UV_MM, :, :, UV_MM, :]
    T_M_to_M = Vsrc_M[:, :, None, None] * chi_MM
    out["MM"] = jnp.einsum("abce,abpq,cers->pqrs",
                           T_M_to_M, Emat, Emat, optimize=True)

    chi_M_to_I = (chi_body[:, UV_MM, :, :, UV_MI, :]
                  + chi_body[:, UV_MM, :, :, UV_II, :])
    T_M_to_I = Vsrc_M[:, :, None, None] * chi_M_to_I
    out["MI"] = jnp.einsum("abce,abpq,ces->pqs",
                           T_M_to_I, Emat, Esng, optimize=True)

    chi_M_to_D = (chi_body[:, UV_MM, :, :, UV_MD, :]
                  + chi_body[:, UV_MM, :, :, UV_DD, :])
    T_M_to_D = Vsrc_M[:, :, None, None] * chi_M_to_D
    out["MD"] = jnp.einsum("abce,abpq,cer->pqr",
                           T_M_to_D, Emat, Esng, optimize=True)

    out["ME"] = jnp.einsum("ab,abpq,ab->pq",
                           Vsrc_M, Emat, chi_to_ee[:, UV_MM, :], optimize=True)

    # ---- Insert-sourced rows: keep both propagated ancestor (p) and
    # emitted-at-insert descendant (q). To do this we treat the source
    # I-state's emission tensor as Eins[d, f, p, q] = pi^c[p] · pi^c[q]
    # marginalised over c (the propagated ancestor and the next-emitted
    # descendant are independent draws from the same per-(d,f) class-
    # mixture). This is the standard "ancestor-context-propagated"
    # treatment for an insert in MixDom2's order-1 distillation.
    Eins_full = jnp.einsum("dfc,cp,cq->dfpq",
                           linear_params["classdist"],
                           linear_params["class_pis"],
                           linear_params["class_pis"])  # (D, F, A, A)

    Vsrc_I = visits_body[:, UV_MI, :] + visits_body[:, UV_II, :]
    chi_I_to_M = chi_body[:, UV_MI, :, :, UV_MM, :] + chi_body[:, UV_II, :, :, UV_MM, :]
    T_I_to_M = Vsrc_I[:, :, None, None] * chi_I_to_M
    out["IM"] = jnp.einsum("abce,abpq,cers->pqrs",
                           T_I_to_M, Eins_full, Emat, optimize=True)

    chi_I_to_I = (chi_body[:, UV_MI, :, :, UV_MI, :] + chi_body[:, UV_MI, :, :, UV_II, :]
                  + chi_body[:, UV_II, :, :, UV_MI, :] + chi_body[:, UV_II, :, :, UV_II, :])
    T_I_to_I = Vsrc_I[:, :, None, None] * chi_I_to_I
    out["II"] = jnp.einsum("abce,abpq,ces->pqs",
                           T_I_to_I, Eins_full, Esng, optimize=True)

    chi_I_to_D = (chi_body[:, UV_MI, :, :, UV_MD, :] + chi_body[:, UV_MI, :, :, UV_DD, :]
                  + chi_body[:, UV_II, :, :, UV_MD, :] + chi_body[:, UV_II, :, :, UV_DD, :])
    T_I_to_D = Vsrc_I[:, :, None, None] * chi_I_to_D
    out["ID"] = jnp.einsum("abce,abpq,cer->pqr",
                           T_I_to_D, Eins_full, Esng, optimize=True)

    out["IE"] = jnp.einsum("ab,abpq,ab->pq",
                           Vsrc_I, Eins_full,
                           chi_to_ee[:, UV_MI, :] + chi_to_ee[:, UV_II, :],
                           optimize=True)

    # ---- Delete-sourced rows: propagated descendant + emitted ancestor.
    # Same trick as for inserts.
    Edel_full = jnp.einsum("dfc,cp,cq->dfpq",
                           linear_params["classdist"],
                           linear_params["class_pis"],
                           linear_params["class_pis"])  # (D, F, A, A) — pi^c[p_anc] * pi^c[q_desc]

    Vsrc_D = visits_body[:, UV_MD, :] + visits_body[:, UV_DD, :]

    chi_D_to_M = chi_body[:, UV_MD, :, :, UV_MM, :] + chi_body[:, UV_DD, :, :, UV_MM, :]
    T_D_to_M = Vsrc_D[:, :, None, None] * chi_D_to_M
    out["DM"] = jnp.einsum("abce,abpq,cers->pqrs",
                           T_D_to_M, Edel_full, Emat, optimize=True)

    chi_D_to_D = (chi_body[:, UV_MD, :, :, UV_MD, :] + chi_body[:, UV_MD, :, :, UV_DD, :]
                  + chi_body[:, UV_DD, :, :, UV_MD, :] + chi_body[:, UV_DD, :, :, UV_DD, :])
    T_D_to_D = Vsrc_D[:, :, None, None] * chi_D_to_D
    out["DD"] = jnp.einsum("abce,abpq,cer->pqr",
                           T_D_to_D, Edel_full, Esng, optimize=True)

    chi_D_to_I = (chi_body[:, UV_MD, :, :, UV_MI, :] + chi_body[:, UV_MD, :, :, UV_II, :]
                  + chi_body[:, UV_DD, :, :, UV_MI, :] + chi_body[:, UV_DD, :, :, UV_II, :])
    T_D_to_I = Vsrc_D[:, :, None, None] * chi_D_to_I
    out["DI"] = jnp.einsum("abce,abpq,ces->pqs",
                           T_D_to_I, Edel_full, Esng, optimize=True)

    out["DE"] = jnp.einsum("ab,abpq,ab->pq",
                           Vsrc_D, Edel_full,
                           chi_to_ee[:, UV_MD, :] + chi_to_ee[:, UV_DD, :],
                           optimize=True)

    # ---- Start-sourced ----
    out["SM"] = jnp.einsum("ce,cers->rs",
                           chi_ss[:, UV_MM, :], Emat, optimize=True)
    out["SI"] = jnp.einsum("ce,ces->s",
                           chi_ss[:, UV_MI, :] + chi_ss[:, UV_II, :], Esng,
                           optimize=True)
    out["SD"] = jnp.einsum("ce,cer->r",
                           chi_ss[:, UV_MD, :] + chi_ss[:, UV_DD, :], Esng,
                           optimize=True)
    out["SE"] = chi_ss_ee

    return out


def normalize_freqs_to_wfst_probs(F):
    """Normalise full-context adjacency frequencies to order-1 Pair WFST
    transition probabilities, in the form expected by
    :func:`maraschino._wfst_to_machineboss_json`.

    For each (source, context) → distribution over (next state type, next chars).
    For source M with context (X, Y): normalise across {MM, MI, MD, ME}.
    For source I with context (X, Y) (X = propagated ancestor, Y = last desc):
        normalise across {IM, II, ID, IE}.
    For source D with context (X, Y): {DM, DD, DI, DE}.
    Start has no context: single softmax over {SM, SI, SD, SE}.

    Each "axis" of the propagated-context I/D rows is broadcast from the
    full 4D F^I*[p, q, ·] tensors (p = propagated ancestor, q = last desc).
    """
    eps = 1e-30
    A = F["MM"].shape[0]
    out = {}

    # Post-Match
    f_mm = jnp.maximum(F["MM"], eps)              # (A, A, A, A)
    f_mi = jnp.maximum(F["MI"], eps)              # (A, A, A)
    f_md = jnp.maximum(F["MD"], eps)              # (A, A, A)
    f_me = jnp.maximum(F["ME"], eps)              # (A, A)
    Z_M = (f_mm.sum(axis=(2, 3)) + f_mi.sum(axis=2)
           + f_md.sum(axis=2) + f_me)
    Z_M = jnp.maximum(Z_M, eps)
    out["p_mm"] = f_mm / Z_M[:, :, None, None]
    out["p_mi"] = f_mi / Z_M[:, :, None]
    out["p_md"] = f_md / Z_M[:, :, None]
    out["p_me"] = f_me / Z_M

    # Post-Insert (full context p=propagated anc, q=last desc)
    f_im = jnp.maximum(F["IM"], eps)              # (A, A, A, A)
    f_ii = jnp.maximum(F["II"], eps)              # (A, A, A)
    f_id = jnp.maximum(F["ID"], eps)              # (A, A, A)
    f_ie = jnp.maximum(F["IE"], eps)              # (A, A)
    Z_I = (f_im.sum(axis=(2, 3)) + f_ii.sum(axis=2)
           + f_id.sum(axis=2) + f_ie)
    Z_I = jnp.maximum(Z_I, eps)
    out["p_im"] = f_im / Z_I[:, :, None, None]
    out["p_ii"] = f_ii / Z_I[:, :, None]
    out["p_id"] = f_id / Z_I[:, :, None]
    out["p_ie"] = f_ie / Z_I

    # Post-Delete
    f_dm = jnp.maximum(F["DM"], eps)              # (A, A, A, A)
    f_dd = jnp.maximum(F["DD"], eps)              # (A, A, A)
    f_di = jnp.maximum(F["DI"], eps)              # (A, A, A)
    f_de = jnp.maximum(F["DE"], eps)              # (A, A)
    Z_D = (f_dm.sum(axis=(2, 3)) + f_dd.sum(axis=2)
           + f_di.sum(axis=2) + f_de)
    Z_D = jnp.maximum(Z_D, eps)
    out["p_dm"] = f_dm / Z_D[:, :, None, None]
    out["p_dd"] = f_dd / Z_D[:, :, None]
    out["p_di"] = f_di / Z_D[:, :, None]
    out["p_de"] = f_de / Z_D

    # Start (single context)
    f_sm = jnp.maximum(F["SM"], eps)              # (A, A)
    f_si = jnp.maximum(F["SI"], eps)              # (A,)
    f_sd = jnp.maximum(F["SD"], eps)              # (A,)
    f_se = jnp.maximum(F["SE"], eps)              # ()
    Z_S = jnp.maximum(f_sm.sum() + f_si.sum() + f_sd.sum() + f_se, eps)
    p_sm_2d = f_sm / Z_S                            # (A, A)
    # The MachineBoss WFST has S→I_{x,y} for each (x, y) pair, where x
    # is a propagated ancestor (uniform over the alphabet at start) and
    # y is the inserted character. We split the 1-D F^SI mass evenly
    # across x: p_si[x, y] = p_si_1d[y] / A. Similarly for SD.
    out["p_sm"] = p_sm_2d
    out["p_si"] = jnp.broadcast_to((f_si / Z_S)[None, :], (A, A)) / float(A)
    out["p_sd"] = jnp.broadcast_to((f_sd / Z_S)[:, None], (A, A)) / float(A)
    out["p_se"] = float(f_se / Z_S)

    return out


def normalize_freqs_to_singlet_probs(F):
    """Normalise to order-1 singlet HMM transitions for the descendant
    sequence (just the per-character bigram model).

    Returns dict with:
      singlet_start: (A,) start-row distribution over first character
      singlet_trans: (A, A) per-character transition probabilities
      singlet_end: (A,) per-character end probabilities
    """
    eps = 1e-30
    A = F["MM"].shape[0]
    # Singlet adjacency f(a, b) = total expected (a, b) bigrams in the
    # descendant sequence under the model. For the descendant we sum
    # over all source/destination pair-HMM types whose emit produces a
    # descendant character (M, I — both emit a descendant). The
    # descendant emitted by MM at (X, Y) is Y; by MI/II it's the
    # second character q. We sum across both. For end/start we use the
    # marginal per-char.

    # Bigram f[a, b] = expected (prev_desc=a, next_desc=b) pairs.
    # Source state has prev_desc=Y; next state has next_desc=Y'.
    # Marginalise over X, X' (ancestor) and Z (delete pass-through).
    f_mm = F["MM"]                  # (X, Y, X', Y')
    f_mi = F["MI"]                  # (X, Y, Y')
    f_im = F["IM"]                  # (Xprop, Y, X', Y')   I→M
    f_ii = F["II"]                  # (Xprop, Y, Y')        I→I

    # Bigram (a=prev_desc, b=next_desc)
    bigram = (jnp.einsum("xayb->ab", f_mm)
              + jnp.einsum("xab->ab", f_mi)
              + jnp.einsum("xayb->ab", f_im)
              + jnp.einsum("xab->ab", f_ii))                # (A, A)

    # End frequencies (prev_desc emit, then transition to E)
    f_me = F["ME"]                  # (X, Y)
    f_ie = F["IE"]                  # (Xprop, Y)
    end_freq = (jnp.einsum("xa->a", f_me) + jnp.einsum("xa->a", f_ie))   # (A,)

    # Start frequencies (S → first_desc)
    f_sm = F["SM"]                  # (X', Y')
    f_si = F["SI"]                  # (Y',)
    start_freq = jnp.einsum("xa->a", f_sm) + f_si                          # (A,)

    eps_a = jnp.maximum(bigram.sum(axis=1) + end_freq, eps)
    p_trans = bigram / eps_a[:, None]
    p_end = end_freq / eps_a
    p_start = start_freq / jnp.maximum(start_freq.sum(), eps)
    return {"singlet_trans": p_trans,
            "singlet_start": p_start,
            "singlet_end": p_end}


def distill_mixdom2_probs(linear_params, tau, n_dom, n_frag, n_classes):
    """End-to-end MixDom2 distillation at evolutionary time tau.

    Returns a single dict with:
      WFST keys: p_mm, p_mi, p_md, p_me, p_im, p_ii, p_id, p_ie,
                 p_dm, p_dd, p_di, p_de, p_sm, p_si, p_sd, p_se
      Singlet keys: singlet_trans, singlet_start, singlet_end
    suitable for `_wfst_to_machineboss_json` /
    `_singlet_to_machineboss_json` in maraschino.py.
    """
    F = adjacency_freqs_full(linear_params, tau, n_dom, n_frag, n_classes)
    probs = normalize_freqs_to_wfst_probs(F)
    probs.update(normalize_freqs_to_singlet_probs(F))
    return probs


# ============================================================
# Save / load (train_pfam .npz format)
# ============================================================
def save_checkpoint(path, linear_params: dict, n_dom: int, n_frag: int,
                    n_classes: int, t: float, em_iter: int = 0,
                    config: dict | None = None) -> None:
    """Write a MixDom2 checkpoint in the same .npz layout as train_pfam.py.

    Key layout for n_classes > 1::

        main_ins, main_del             scalars
        dom_ins[D], dom_del[D]
        dom_weights[D], frag_weights[D, F], ext_rates[D, F, F]
        class_pis[C, A], class_S_exch[C, A, A], classdist[D, F, C]
        n_classes_frag                 int
        t                              float (training time)
        em_iter                        int
        _config                        json string with model+training metadata
    """
    import json

    save_dict = {
        "main_ins": np.asarray(float(linear_params["main_ins"]), dtype=np.float32),
        "main_del": np.asarray(float(linear_params["main_del"]), dtype=np.float32),
        "dom_ins": np.asarray(linear_params["dom_ins"], dtype=np.float32),
        "dom_del": np.asarray(linear_params["dom_del"], dtype=np.float32),
        "dom_weights": np.asarray(linear_params["dom_weights"], dtype=np.float32),
        "frag_weights": np.asarray(linear_params["frag_weights"], dtype=np.float32),
        "ext_rates": np.asarray(linear_params["ext_rates"], dtype=np.float32),
        "t": np.asarray(t, dtype=np.float32),
        "em_iter": np.asarray(em_iter, dtype=np.int32),
    }
    if n_classes > 1:
        save_dict["class_pis"] = np.asarray(linear_params["class_pis"], dtype=np.float32)
        save_dict["class_S_exch"] = np.asarray(linear_params["class_S_exch"], dtype=np.float32)
        save_dict["classdist"] = np.asarray(linear_params["classdist"], dtype=np.float32)
        save_dict["n_classes_frag"] = np.asarray(n_classes, dtype=np.int32)
    if config is not None:
        save_dict["_config"] = np.asarray(json.dumps(config))

    np.savez(path, **save_dict)


def load_checkpoint_linear(path) -> dict:
    """Load a train_pfam-style .npz into a linear-space MixDom2 param dict.

    Recognises both n_classes=1 and n_classes>1 layouts. Strips numpy 0-d
    wrappers on scalar entries.
    """
    data = np.load(path, allow_pickle=True)
    out: dict = {
        "main_ins": float(data["main_ins"]),
        "main_del": float(data["main_del"]),
        "dom_ins": np.asarray(data["dom_ins"]),
        "dom_del": np.asarray(data["dom_del"]),
        "dom_weights": np.asarray(data["dom_weights"]),
        "frag_weights": np.asarray(data["frag_weights"]),
        "ext_rates": np.asarray(data["ext_rates"]),
    }
    if "ext_rates" in data and out["ext_rates"].ndim == 2:
        # MixDom1 (D, F) → MixDom2 (D, F, F) diagonal expansion
        D, F = out["ext_rates"].shape
        ext = np.zeros((D, F, F), dtype=np.float32)
        for d in range(D):
            ext[d] = np.diag(out["ext_rates"][d])
        out["ext_rates"] = ext
    if "class_pis" in data:
        out["class_pis"] = np.asarray(data["class_pis"])
        out["class_S_exch"] = np.asarray(data["class_S_exch"])
        out["classdist"] = np.asarray(data["classdist"])
    return out
