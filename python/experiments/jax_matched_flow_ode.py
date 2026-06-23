"""JAX/diffrax port of the matched-flow ODE for the closed-form Triad,
with L'Hopital handling at lam ~ mu via bdi.score_derivatives gating.

Translation of scratch_ggi_cond_kl_quad.py to JAX so we can:
  - integrate (lam(t), mu(t), r(t)) via diffrax along each branch length;
  - take gradients w.r.t. the GGI parameters (lam0, mu0, x_del, y_ins);
  - plug into an Adam loop for end-to-end fitting.

Only stateless pieces — no scipy, no kl_fit, no Gillespie.

State indices:
  S, M, I, D, E   = 0, 1, 2, 3, 4
  Triad 13:       SS=0, sI=1, MM=2, mI=3, MD=4, IM=5, iI=6, ID=7,
                  Ds=8, Dm=9, Di=10, Dd=11, EE=12

This file is intended for offline experiments, not production.
"""
import jax
import jax.numpy as jnp

S, M, I, D, E = 0, 1, 2, 3, 4
SS, sI_, MM, mI, MD, IM, iI, ID_, Ds, Dm, Di, Dd, EE = range(13)
C_map = (S, S, M, M, M, I, I, I, D, D, D, D, E)  # coarse-graining (13 -> 5)


# ---------------------------------------------------------------------------
# JAX TKF92 5x5 transition matrix (Y) and the z-table
# ---------------------------------------------------------------------------

def jax_tau5(lam_s, mu_s, r_s, t):
    """TKF92 5x5 Pair HMM transition matrix Y[src, dst].  Uses bdi.tkf_beta
    and tkf_gamma which have proper L'Hopital limits at lam=mu (their naive
    formulas return 0/0 = 0 there instead of the correct s/(1+s) and
    associated gamma).
    """
    from tkfmixdom.jax.core.bdi import tkf_beta, tkf_gamma
    kap = lam_s / jnp.maximum(mu_s, 1e-30)
    alpha = jnp.exp(-mu_s * t)
    b = tkf_beta(lam_s, mu_s, t)
    g = tkf_gamma(lam_s, mu_s, t)
    ob, og, ok = 1 - b, 1 - g, 1 - kap
    T = jnp.zeros((5, 5))
    # S row
    T = T.at[S, M].set(ob * kap * alpha)
    T = T.at[S, I].set(b)
    T = T.at[S, D].set(ob * kap * (1 - alpha))
    T = T.at[S, E].set(ob * ok)
    # M, I row (with fragment extension)
    for src in (M, I):
        row_no_ext = jnp.array(
            [0.0, ob * kap * alpha, b, ob * kap * (1 - alpha), ob * ok])
        row = (1 - r_s) * row_no_ext
        T = T.at[src].set(row.at[src].add(r_s))
    # D row
    rowd = (1 - r_s) * jnp.array(
        [0.0, og * kap * alpha, g, og * kap * (1 - alpha), og * ok])
    rowd = rowd.at[D].add(r_s)
    T = T.at[D].set(rowd)
    return T


def jax_z_tables(lam_g, mu_g, x_del, y_ins):
    """Z0[src, dst] and Z1[src, dst] tables (5x5 each).  Z0 = z|_{dt=0},
    Z1 = dz/d(dt)|_{dt=0}.  APPENDIX convention: x_del = deletion length
    geom, y_ins = insertion length geom.  (scratch_ggi_*.py uses the
    opposite convention; do not mix them up.)"""
    Z0 = jnp.zeros((5, 5))
    Z1 = jnp.zeros((5, 5))
    # src in (S, M)
    for src in (S, M):
        Z0 = Z0.at[src, M].set(1.0); Z1 = Z1.at[src, M].set(-(lam_g + mu_g))
        Z0 = Z0.at[src, I].set(0.0); Z1 = Z1.at[src, I].set(lam_g)
        Z0 = Z0.at[src, D].set(0.0); Z1 = Z1.at[src, D].set(mu_g)
        Z0 = Z0.at[src, E].set(1.0); Z1 = Z1.at[src, E].set(-lam_g)
    # src = I  --- uses y_ins (insertion length geom)
    Z0 = Z0.at[I, I].set(y_ins)
    Z0 = Z0.at[I, M].set(1 - y_ins)
    Z0 = Z0.at[I, E].set(1 - y_ins)
    # src = D  --- uses x_del (deletion length geom)
    Z0 = Z0.at[D, D].set(x_del)
    Z0 = Z0.at[D, M].set(1 - x_del)
    Z0 = Z0.at[D, E].set(1.0)
    return Z0, Z1


# ---------------------------------------------------------------------------
# build_AB in JAX
# Entries (row, col, fac_y, z, is_sety) describe each non-zero in A/B.
#   fac_y is None (=> 1) or a (Y_row, Y_col) pair.
#   z is None (sety) or a (src_z, dst_z) pair (setyz).
# ---------------------------------------------------------------------------

# (row, col, fac_y or None, z = (src_z, dst_z) or None)
_TRIAD_ENTRIES = [
    # ----- Row SS -----
    (SS, sI_, None,    (S, I)),
    (SS, MM,  (S, M),  (S, M)),
    (SS, MD,  (S, M),  (S, D)),
    (SS, IM,  (S, I),  (S, M)),
    (SS, ID_, (S, I),  (S, D)),
    (SS, Ds,  (S, D),  None),
    (SS, EE,  (S, E),  (S, E)),
    # ----- Row sI -----
    (sI_, sI_, None,    (I, I)),
    (sI_, MM,  (S, M),  (I, M)),
    (sI_, MD,  (S, M),  (I, D)),
    (sI_, IM,  (S, I),  (I, M)),
    (sI_, ID_, (S, I),  (I, D)),
    (sI_, Di,  (S, D),  None),
    (sI_, EE,  (S, E),  (I, E)),
    # ----- Row MM -----
    (MM, MM,  (M, M), (M, M)),
    (MM, mI,  None,   (M, I)),
    (MM, MD,  (M, M), (M, D)),
    (MM, IM,  (M, I), (M, M)),
    (MM, ID_, (M, I), (M, D)),
    (MM, Dm,  (M, D), None),
    (MM, EE,  (M, E), (M, E)),
    # ----- Row mI -----
    (mI, MM,  (M, M), (I, M)),
    (mI, mI,  None,   (I, I)),
    (mI, MD,  (M, M), (I, D)),
    (mI, IM,  (M, I), (I, M)),
    (mI, ID_, (M, I), (I, D)),
    (mI, Di,  (M, D), None),
    (mI, EE,  (M, E), (I, E)),
    # ----- Row MD -----
    (MD, MM,  (M, M), (D, M)),
    (MD, mI,  None,   (D, I)),
    (MD, MD,  (M, M), (D, D)),
    (MD, IM,  (M, I), (D, M)),
    (MD, ID_, (M, I), (D, D)),
    (MD, Dd,  (M, D), None),
    (MD, EE,  (M, E), (D, E)),
    # ----- Row IM -----
    (IM, MM,  (I, M), (M, M)),
    (IM, MD,  (I, M), (M, D)),
    (IM, IM,  (I, I), (M, M)),
    (IM, iI,  None,   (M, I)),
    (IM, ID_, (I, I), (M, D)),
    (IM, Dm,  (I, D), None),
    (IM, EE,  (I, E), (M, E)),
    # ----- Row iI -----
    (iI, MM,  (I, M), (I, M)),
    (iI, MD,  (I, M), (I, D)),
    (iI, IM,  (I, I), (I, M)),
    (iI, iI,  None,   (I, I)),
    (iI, ID_, (I, I), (I, D)),
    (iI, Di,  (I, D), None),
    (iI, EE,  (I, E), (I, E)),
    # ----- Row ID -----
    (ID_, MM,  (I, M), (D, M)),
    (ID_, MD,  (I, M), (D, D)),
    (ID_, IM,  (I, I), (D, M)),
    (ID_, iI,  None,   (D, I)),
    (ID_, ID_, (I, I), (D, D)),
    (ID_, Dd,  (I, D), None),
    (ID_, EE,  (I, E), (D, E)),
    # ----- Row Ds -----
    (Ds, MM,  (D, M), (S, M)),
    (Ds, MD,  (D, M), (S, D)),
    (Ds, IM,  (D, I), (S, M)),
    (Ds, ID_, (D, I), (S, D)),
    (Ds, Ds,  (D, D), None),
    (Ds, EE,  (D, E), (S, E)),
    # ----- Row Dm -----
    (Dm, MM,  (D, M), (M, M)),
    (Dm, MD,  (D, M), (M, D)),
    (Dm, IM,  (D, I), (M, M)),
    (Dm, ID_, (D, I), (M, D)),
    (Dm, Dm,  (D, D), None),
    (Dm, EE,  (D, E), (M, E)),
    # ----- Row Di -----
    (Di, MM,  (D, M), (I, M)),
    (Di, MD,  (D, M), (I, D)),
    (Di, IM,  (D, I), (I, M)),
    (Di, ID_, (D, I), (I, D)),
    (Di, Di,  (D, D), None),
    (Di, EE,  (D, E), (I, E)),
    # ----- Row Dd -----
    (Dd, MM,  (D, M), (D, M)),
    (Dd, MD,  (D, M), (D, D)),
    (Dd, IM,  (D, I), (D, M)),
    (Dd, ID_, (D, I), (D, D)),
    (Dd, Dd,  (D, D), None),
    (Dd, EE,  (D, E), (D, E)),
]


def jax_build_AB(lam_s, mu_s, r_s, lam_g, mu_g, x_del, y_ins, t):
    """Construct the 13x13 A and B matrices.  A = T|_{dt=0}, B = dT/d(dt)|_{dt=0}.
    APPENDIX convention: x_del = deletion length geom, y_ins = insertion.
    """
    Y = jax_tau5(lam_s, mu_s, r_s, t)
    Z0, Z1 = jax_z_tables(lam_g, mu_g, x_del, y_ins)
    # Build (rows, cols, A_values, B_values) flat lists from the entry table.
    rows = [e[0] for e in _TRIAD_ENTRIES]
    cols = [e[1] for e in _TRIAD_ENTRIES]
    A_vals = []
    B_vals = []
    for (_, _, fac, z) in _TRIAD_ENTRIES:
        fac_y = 1.0 if fac is None else Y[fac[0], fac[1]]
        if z is None:
            A_vals.append(fac_y)
            B_vals.append(jnp.zeros_like(fac_y))
        else:
            A_vals.append(fac_y * Z0[z[0], z[1]])
            B_vals.append(fac_y * Z1[z[0], z[1]])
    A_vals = jnp.stack(A_vals)
    B_vals = jnp.stack(B_vals)
    A = jnp.zeros((13, 13)).at[jnp.array(rows), jnp.array(cols)].add(A_vals)
    B = jnp.zeros((13, 13)).at[jnp.array(rows), jnp.array(cols)].add(B_vals)
    return A, B


# ---------------------------------------------------------------------------
# Coarse-grain 13x13 -> 5x5 (sums over the C_map)
# ---------------------------------------------------------------------------

_CMAP_JAX = jnp.array(C_map)


def coarse_grain_jax(n):
    """Sum entries n[a, b] into m[C_map[a], C_map[b]]."""
    # n has shape (13, 13).  Use segment_sum over rows then columns.
    # Step 1: collapse rows by C_map.
    row_collapsed = jax.ops.segment_sum(n, _CMAP_JAX, num_segments=5)
    # Step 2: transpose, collapse rows (= original cols), transpose back.
    col_collapsed = jax.ops.segment_sum(row_collapsed.T, _CMAP_JAX, num_segments=5)
    return col_collapsed.T


# ---------------------------------------------------------------------------
# tkf91/tkf92 BDI stats from 5x5 m matrix (JAX port of _tkf91_bdi_from_m)
# ---------------------------------------------------------------------------

# Use bdi.score_derivatives which has L'Hopital gating built in.
from tkfmixdom.jax.core.bdi import score_derivatives as _bdi_score_derivatives


def _score_derivs_jax(lam, mu, t):
    """6-slot list [log alpha, log(1-alpha), log beta, log(1-beta),
    log gamma, log(1-gamma)] of (d/d lam, d/d mu) pairs, computed via
    bdi.score_derivatives so that the L'Hopital limit at lam ~ mu is
    handled automatically."""
    d = _bdi_score_derivatives(lam, mu, t)
    return [
        d['log_alpha'],
        d['log_1malpha'],
        d['log_beta'],
        d['log_1mbeta'],
        d['log_gamma'],
        d['log_1mgamma'],
    ]


def _count_groups_jax(m):
    n_a = m[S, M] + m[M, M] + m[I, M] + m[D, M]
    n_1a = m[S, D] + m[M, D] + m[I, D] + m[D, D]
    n_b = m[S, I] + m[M, I] + m[I, I]
    n_1b = (m[S, M] + m[S, D] + m[S, E]
            + m[M, M] + m[M, D] + m[M, E]
            + m[I, M] + m[I, D] + m[I, E])
    n_g = m[D, I]
    n_1g = m[D, M] + m[D, D] + m[D, E]
    return [n_a, n_1a, n_b, n_1b, n_g, n_1g]


from tkfmixdom.jax.core.bdi import _logP_wfst_smooth as _bdi_logP_smooth


def _es_lhopital_jax(groups, cons, mu, t, T_total):
    """Pure-JAX L'Hopital limit of E[S] at lam=mu, differentiable in mu.

    Reuses bdi._logP_wfst_smooth (which has analytic L'Hopital limits in
    its tkf_beta_smooth, tkf_gamma_smooth via expm1_ratio) and computes
        E[S] = mu * d^2 logP/(d lam d mu) - d logP/d lam
                - mu * d^2 logP/d lam^2 - T_total
    via nested jax.grad.  All operations are JAX-traceable.
    """
    n_a, n_1a, n_b, n_1b, n_g, n_1g = groups

    def _numerator(lam):
        dlogP_dlam = jax.grad(
            lambda l: _bdi_logP_smooth(l, mu, t, n_a, n_1a, n_b, n_1b, n_g, n_1g))(lam)
        dlogP_dmu = jax.grad(
            lambda m: _bdi_logP_smooth(lam, m, t, n_a, n_1a, n_b, n_1b, n_g, n_1g))(mu)
        return cons + mu * dlogP_dmu - lam * dlogP_dlam - lam * T_total

    return jax.grad(_numerator)(mu)


def _tkf91_bdi_from_m_jax(m, lam, mu, t):
    """JAX BDI counts.  Conservation-law identity for |lam - mu| > threshold;
    proper L'Hopital limit (eq:ES-limit) via _es_lhopital_jax for |lam - mu|
    <= threshold.  Both branches are JAX-traceable; we always compute both
    and select via jnp.where (so jax.grad through this function does the
    right thing in both regimes)."""
    groups = _count_groups_jax(m)
    derivs = _score_derivs_jax(lam, mu, t)
    lam_score = sum(g * lam * d[0] for g, d in zip(groups, derivs))
    mu_score = sum(g * mu * d[1] for g, d in zip(groups, derivs))
    cons = jnp.sum(m[:, I]) - jnp.sum(m[:, D])
    eps = lam - mu
    threshold = 1e-4
    is_near_equal = jnp.abs(eps) < threshold

    # General branch (works for |eps| > threshold)
    eps_safe_general = jnp.where(
        jnp.abs(eps) < 1e-12, 1e-12 * jnp.where(eps >= 0, 1.0, -1.0), eps)
    E_S_general = (cons + mu_score - lam_score - lam * t) / eps_safe_general

    # L'Hopital branch (works at lam = mu; reuses bdi smooth logP)
    E_S_limit = _es_lhopital_jax(groups, cons, mu, t, t)

    E_S = jnp.where(is_near_equal, E_S_limit, E_S_general)
    E_B = lam_score + lam * E_S + lam * t
    E_D = mu_score + mu * E_S
    return E_B, E_D, E_S


def tkf92_stats_jax(m, lam, mu, t, ext):
    """E[B], E[D], E[S], F, En from raw TKF92 (5,5) counts."""
    tau92 = jax_tau5(lam, mu, ext, t)
    F = jnp.zeros(())
    n_corr = m
    for s in (M, I, D):
        diag92 = tau92[s, s]
        ext_frac = jnp.clip(ext / jnp.maximum(diag92, 1e-30), 0.0, 1.0)
        loop = m[s, s]
        F = F + ext_frac * loop
        n_corr = n_corr.at[s, s].set((1 - ext_frac) * loop)
    notF = (jnp.sum(n_corr[M, :])
            + jnp.sum(n_corr[I, :])
            + jnp.sum(n_corr[D, :]))
    E_B, E_D, E_S = _tkf91_bdi_from_m_jax(n_corr, lam, mu, t)
    return E_B, E_D, E_S, F, notF


# ---------------------------------------------------------------------------
# Stats and increments via the Triad
# ---------------------------------------------------------------------------

def stats_and_increments_jax(lam, mu, r, t, lam0, mu0, x_del, y_ins):
    A, B = jax_build_AB(lam, mu, r, lam0, mu0, x_del, y_ins, t)
    C = jnp.linalg.inv(jnp.eye(13) - A)
    CBC = C @ B @ C
    SS_, EE_ = 0, 12
    n0 = jnp.outer(C[SS_], C[:, EE_]) * A
    n1 = (jnp.outer(CBC[SS_], C[:, EE_]) * A
          + jnp.outer(C[SS_], C[:, EE_]) * B
          + jnp.outer(C[SS_], CBC[:, EE_]) * A)
    m0 = coarse_grain_jax(n0)
    m1 = coarse_grain_jax(n1)
    s0 = tkf92_stats_jax(m0, lam, mu, t, r)
    s1 = tkf92_stats_jax(m1, lam, mu, t, r)
    return jnp.stack(s0), jnp.stack(s1)


def dtheta_dt_jax(t, theta, lam0, mu0, x_del, y_ins):
    """Right-hand side of the matched-flow ODE.  All-JAX, differentiable in
    (lam0, mu0, x_del, y_ins) and theta."""
    lam, mu, r = theta[0], theta[1], theta[2]
    s0, s1 = stats_and_increments_jax(lam, mu, r, t, lam0, mu0, x_del, y_ins)
    B_, D_, S_, F_, En_ = s0
    bG, dG, sG, fG, eG = s1
    dlam = (bG - lam * (sG + 1.0)) / jnp.maximum(S_ + t, 1e-30)
    dmu = (dG - mu * sG) / jnp.maximum(S_, 1e-30)
    dr = ((1 - r) * fG - r * eG) / jnp.maximum(F_ + En_, 1e-30)
    return jnp.array([dlam, dmu, dr])


def boundary_jax(lam0, mu0, x_del, y_ins):
    """JAX boundary -- appendix convention (x=del, y=ins).

    r*(0) = (lam0 y_ins (1 - x_del) + mu0 x_del (1 - y_ins)) /
            (lam0 (1 - x_del)       + mu0 (1 - y_ins))
    lam*(0) = lam0 / (1 - r*(0)), mu*(0) = mu0 / (1 - r*(0)).
    """
    num = lam0 * y_ins * (1 - x_del) + mu0 * x_del * (1 - y_ins)
    den = jnp.maximum(lam0 * (1 - x_del) + mu0 * (1 - y_ins), 1e-30)
    r_star = num / den
    one_m_r = jnp.maximum(1 - r_star, 1e-30)
    return jnp.array([lam0 / one_m_r, mu0 / one_m_r, r_star])
