"""Canonical GGI -> TKF92 matched-flow ODE integrator.

Implements equation (eq:ode-params) of tkf/composition-renormalization.tex
(the "boxed" parameter ODE, lines ~222-237):

    d lambda / dt = (Delta[B] - lambda * (Delta[S] + 1))   / (E_T[S] + t)
    d mu     / dt = (Delta[D] - mu     *  Delta[S])        /  E_T[S]
    d r      / dt = ((1 - r) * Delta[F] - r * Delta[E])    / (E_T[F] + E_T[E])

with the BO-slaved per-residue boundary condition (eq:boundary-r):

    r*(0)    = [lam0 * y * (1-x) + mu0 * x * (1-y)] / [lam0 * (1-x) + mu0 * (1-y)]
    lam_T(0) = lam0 / (1 - r*(0))
    mu_T(0)  = mu0  / (1 - r*(0))

(Convention swap inherited from the rest of the repo: code's `x_ins` is the
GGI insertion-extension `y`, code's `y_del` is the GGI deletion-extension
`x`.)

Per ODE step:
  1. Build the 13x13 Triad matrices A, B via `build_AB`.
  2. Compute C = (I - A)^{-1} once (numpy.linalg.inv).
  3. Read off n^(0) (= C[SS,:].outer(C[:,EE]) * A) and n^(1) (the CBC-decorated
     leading GGI perturbation) -- both rationally algebraic in (A, B, C).
  4. Coarse-grain 13 -> 5 to get m^(0), m^(1).
  5. Apply the closed-form TKF92 BDI map to both, with PROPER analytic
     L'Hopital handling at lambda = mu (NOT a recursive epsilon-clamp).
  6. Substitute into the boxed parameter ODE.

NO kl_fit, NO Gillespie, NO L-BFGS, NO JAX -- pure-numpy algebraic ODE.

The thin re-export shims at `scratch_ggi_cond_kl_ode.py` and
`scratch_ggi_triad_eliminated.py` forward to this module: they previously
held kl_fit / null-elimination alternative implementations that existed as
workarounds for the original (recursive, broken) L'Hopital fallback in this
file -- now fixed, those workarounds are obsolete.

History: the previous in-file `_tkf91_bdi_from_m` had a recursive
"L'Hopital fallback" that perturbed lambda by lambda*1e-4 whenever
|lambda-mu| < 1e-4.  For symmetric GGI (Pfam x ~ y), the perturbation was
smaller than the gap, so the recursion never escaped the threshold and
either stack-overflowed or returned garbage.  The fix here uses the general
formula for |lambda-mu| > 1e-10 (the float64-stability threshold, far below
any realistic GGI regime) and a non-recursive central-FD L'Hopital limit at
lambda = mu exactly.
"""
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, '/Users/yam/tkf-mixdom/python')

from scratch_ggi_triad_algebraic import build_AB
from scratch_ggi_flow import tau5

# 13-state Triad index map: SS=0, sI=1, MM=2, mI=3, MD=4, IM=5, iI=6,
# ID=7, Ds=8, Dm=9, Di=10, Dd=11, EE=12.
# We null-eliminate the ID state (a ghost insertion immediately followed
# by a deletion -- produces no observable column in the X->Z alignment)
# before coarse-graining, then use CMAP_12 to project onto the 5-state
# pair-HMM (the C_map in scratch_ggi_triad_algebraic.py mis-labels sI,
# mI, MD; see CMAP_12 below for the correct mapping).
SS_IDX, ID_IDX, EE_IDX = 0, 7, 12
KEEP_13 = [i for i in range(13) if i != ID_IDX]    # 12 indices ascending
S_, M_, I_, D_, E_ = 0, 1, 2, 3, 4
# Position-by-position: SS, sI, MM, mI, MD, IM, iI, Ds, Dm, Di, Dd, EE
CMAP_12 = [S_, I_, M_, I_, D_, I_, I_, D_, D_, D_, D_, E_]
SS_IDX_12 = 0
EE_IDX_12 = 11   # EE was at 12 in the 13-state, now at 11 after removing ID

# Threshold below which we switch from the general E_S = numerator/eps formula
# to a central-FD L'Hopital limit.  Set to 1e-10 -- well below any realistic
# GGI |lam-mu| (typically >= 1e-5), but above float64 round-off in the eps
# division (which only matters at |eps| < 1e-15 or so).
L_HOPITAL_EPS = 1e-10


# ---------------------------------------------------------------------------
# Pure-numpy TKF91/TKF92 BDI map
# ---------------------------------------------------------------------------

def _tkf_beta(lam, mu, t):
    eta = np.exp(-lam * t)
    alpha = np.exp(-mu * t)
    delta = mu * eta - lam * alpha
    if abs(delta) < 1e-30:
        return 0.0
    return lam * (eta - alpha) / delta


def _tkf_gamma(lam, mu, t):
    alpha = np.exp(-mu * t)
    if abs(1 - alpha) < 1e-30:
        return 0.5
    kap = lam / mu
    return 1.0 - _tkf_beta(lam, mu, t) / (kap * (1 - alpha))


def _safe(v, eps=1e-30):
    return v if abs(v) >= eps else (eps if v >= 0 else -eps)


def _score_derivs(lam, mu, t):
    """Pure-numpy port of _score_derivatives_general from bdi.py.

    Returns list of 6 tuples (d/dlam, d/dmu) for log-elasticities of
    (alpha, 1-alpha, beta, 1-beta, gamma, 1-gamma).
    """
    eta = np.exp(-lam * t)
    alpha = np.exp(-mu * t)
    beta = _tkf_beta(lam, mu, t)
    gamma = _tkf_gamma(lam, mu, t)
    delta = mu * eta - lam * alpha
    Phi = 1 - alpha

    eta_m_alpha = _safe(eta - alpha)
    delta_s = _safe(delta)
    Phi_s = max(Phi, 1e-30)

    # log alpha:  ∂/∂λ = 0, ∂/∂μ = -t
    da = (0.0, -t)
    # log(1-alpha)
    d1a = (0.0, t * alpha / Phi_s)
    # log beta
    db_dl = 1 / lam - t * eta / eta_m_alpha + (t * mu * eta + alpha) / delta_s
    db_dm = t * alpha / eta_m_alpha - (eta + t * lam * alpha) / delta_s
    # log(1-beta)
    rb = beta / max(1 - beta, 1e-30)
    d1b = (-rb * db_dl, -rb * db_dm)
    # log(1-gamma)
    d1g_dl = db_dl - 1 / lam
    d1g_dm = 1 / mu + db_dm - t * alpha / Phi_s
    # log gamma
    rg = (1 - gamma) / max(gamma, 1e-30)
    dg = (-rg * d1g_dl, -rg * d1g_dm)
    return [da, d1a, (db_dl, db_dm), d1b, dg, (d1g_dl, d1g_dm)]


def _count_groups(m):
    """Return list of 6 WFST count groups
    (log_alpha, log_1ma, log_b, log_1mb, log_g, log_1mg)."""
    n_a = m[S_, M_] + m[M_, M_] + m[I_, M_] + m[D_, M_]
    n_1a = m[S_, D_] + m[M_, D_] + m[I_, D_] + m[D_, D_]
    n_b = m[S_, I_] + m[M_, I_] + m[I_, I_]
    n_1b = (m[S_, M_] + m[S_, D_] + m[S_, E_]
            + m[M_, M_] + m[M_, D_] + m[M_, E_]
            + m[I_, M_] + m[I_, D_] + m[I_, E_])
    n_g = m[D_, I_]
    n_1g = m[D_, M_] + m[D_, D_] + m[D_, E_]
    return [n_a, n_1a, n_b, n_1b, n_g, n_1g]


def _general_bdi(m, lam, mu, t, T_total):
    """General-case E[B], E[D], E[S] from a (5,5) TKF91 count matrix.

    Assumes lambda != mu (caller is responsible for the L'Hopital branch).
    """
    groups = _count_groups(m)
    derivs = _score_derivs(lam, mu, t)
    lam_score = sum(g * lam * d[0] for g, d in zip(groups, derivs))
    mu_score = sum(g * mu * d[1] for g, d in zip(groups, derivs))
    cons = m[:, I_].sum() - m[:, D_].sum()
    eps = lam - mu
    E_S = (cons + mu_score - lam_score - lam * T_total) / eps
    E_B = lam_score + lam * E_S + lam * T_total
    E_D = mu_score + mu * E_S
    return E_B, E_D, E_S


def _tkf91_bdi_from_m(m, lam, mu, t, T_total=None):
    """E[B], E[D], E[S] from (5,5) TKF91 transition counts.

    Uses the general formula for |lambda - mu| > L_HOPITAL_EPS.  At
    lambda = mu exactly (within L_HOPITAL_EPS, which is far below any
    realistic GGI regime), uses a non-recursive central-FD L'Hopital limit:
    evaluates the general formula at (mu+h, mu) and (mu-h, mu) and averages.
    """
    if T_total is None:
        T_total = t
    eps = lam - mu
    if abs(eps) > L_HOPITAL_EPS:
        return _general_bdi(m, lam, mu, t, T_total)
    # Central-FD L'Hopital limit at lambda = mu.
    h = max(abs(mu) * 1e-4, 1e-8)
    Bp, Dp, Sp = _general_bdi(m, mu + h, mu, t, T_total)
    Bm, Dm, Sm = _general_bdi(m, mu - h, mu, t, T_total)
    return 0.5 * (Bp + Bm), 0.5 * (Dp + Dm), 0.5 * (Sp + Sm)


def tkf92_stats_full(m, lam, mu, t, ext):
    """E[B], E[D], E[S], F, E, L, M from raw TKF92 chi counts (5,5).

    L = n_kappa = sum of m[:, M] + m[:, D]  (expected ancestor-residue count)
    M = n_{1-kappa} = sum of m[:, E]        (expected trajectory-termination count)

    These are needed for the joint M-step (eq:kappa-quadratic) that uses
    the singlet ancestor likelihood in addition to the conditional descendant
    likelihood.  See bdi.m_step_indel_quadratic.
    """
    E_B, E_D, E_S, F, E_term = tkf92_stats(m, lam, mu, t, ext)
    L = float(m[:, M_].sum() + m[:, D_].sum())
    M_n = float(m[:, E_].sum())
    return E_B, E_D, E_S, F, E_term, L, M_n


def tkf92_stats(m, lam, mu, t, ext):
    """E[B], E[D], E[S], F, E from raw TKF92 chi counts (5,5).

    Decomposes chi self-loops into (extension, new-fragment-with-same-state)
    components, then applies the TKF91 BDI map to the cleaned count matrix.
    """
    tau92 = tau5(lam / mu, np.exp(-mu * t), ext)
    n_corr = m.copy()
    F = 0.0
    for s in (M_, I_, D_):
        diag92 = tau92[s, s]
        if diag92 < 1e-30:
            continue
        ext_frac = max(0.0, min(1.0, ext / diag92))
        loop = m[s, s]
        F += ext_frac * loop
        n_corr[s, s] = (1 - ext_frac) * loop
    notF = n_corr[(M_, I_, D_), :].sum()
    E_B, E_D, E_S = _tkf91_bdi_from_m(n_corr, lam, mu, t)
    return E_B, E_D, E_S, F, notF


# ---------------------------------------------------------------------------
# Triad ODE
# ---------------------------------------------------------------------------

def _null_eliminate_ID(A_full, B_full, atol=1e-12):
    """Eliminate the ID state from the 13x13 (A', B') -> 12x12 (U, V).

    Paths through ID (i -> ID -> ID -> ... -> ID -> j) are geometric-summed
    and folded back into the (i, j) transitions:

        U_{ij} = A'_{ij} + A'_{i, ID} * A'_{ID, j} / (1 - A'_{ID, ID})
        V_{ij} = B'_{ij} + B'_{i, ID} * A'_{ID, j} / (1 - A'_{ID, ID})

    Since B' has no nonzero ID row (z_{II}, z_{ID}, z_{DD} have no dt
    derivative), Q'_{ID, j} = A'_{ID, j} exactly, so the elimination splits
    cleanly into the closed forms above with no implicit equation.
    """
    if not np.allclose(B_full[ID_IDX, :], 0, atol=atol):
        raise ValueError(f"B' has nonzero ID row: {B_full[ID_IDX, :]}")
    g = A_full[ID_IDX, ID_IDX]
    one_minus_g = 1.0 - g
    if abs(one_minus_g) < 1e-14:
        raise ValueError(f"1 - A'_{{ID,ID}} = {one_minus_g}, cannot eliminate")
    A_iID = A_full[KEEP_13, ID_IDX]
    A_IDj = A_full[ID_IDX, KEEP_13]
    B_iID = B_full[KEEP_13, ID_IDX]
    A_keep = A_full[np.ix_(KEEP_13, KEEP_13)]
    B_keep = B_full[np.ix_(KEEP_13, KEEP_13)]
    U = A_keep + np.outer(A_iID, A_IDj) / one_minus_g
    V = B_keep + np.outer(B_iID, A_IDj) / one_minus_g
    return U, V


def _coarse_grain_12(n12):
    """Coarse-grain a 12x12 transition count to 5x5 (S, M, I, D, E)
    using the CORRECTED CMAP_12."""
    m = np.zeros((5, 5))
    for a in range(12):
        for b in range(12):
            m[CMAP_12[a], CMAP_12[b]] += n12[a, b]
    return m


def stats_and_increments(lam, mu, r, t, lam0, mu0, x_ins, y_del):
    """Build the 13x13 Triad (A, B), null-eliminate the ID state to get
    (U, V) on 12 states, then read off n^(0), n^(1), coarse-grain to (5,5)
    via the CORRECTED CMAP_12, and apply the closed-form TKF92 BDI map.

    The BDI map is affine (linear-in-m PLUS a constant offset that comes
    from the immortal-link `+lam*T` term in the TKF91 BDI formulas).  For
    Delta[X] we want the LINEAR part only -- so we subtract the m=0
    baseline `tkf92_stats(0, ...)` from the m1 evaluation.

    Returns (s0, s1) as (5,) arrays of (E_B, E_D, E_S, F, E).
    """
    Afull, Bfull, _Y = build_AB(lam, mu, r, lam0, mu0, x_ins, y_del, t)
    U, V = _null_eliminate_ID(Afull, Bfull)
    C = np.linalg.inv(np.eye(12) - U)
    CVC = C @ V @ C
    n0 = np.outer(C[SS_IDX_12, :], C[:, EE_IDX_12]) * U
    n1 = (np.outer(CVC[SS_IDX_12, :], C[:, EE_IDX_12]) * U
          + np.outer(C[SS_IDX_12, :], C[:, EE_IDX_12]) * V
          + np.outer(C[SS_IDX_12, :], CVC[:, EE_IDX_12]) * U)
    m0 = _coarse_grain_12(n0)
    m1 = _coarse_grain_12(n1)
    s0 = tkf92_stats(m0, lam, mu, t, r)
    # Subtract the m=0 baseline of the BDI map (the affine offset from
    # the +lam*T immortal-link clock).  This makes the perturbation
    # evaluation `tkf92_stats(m1) - tkf92_stats(0)` purely linear in m1,
    # which is what Delta[X] is supposed to be.
    s_zero = tkf92_stats(np.zeros((5, 5)), lam, mu, t, r)
    s1 = tuple(s1i - sz for s1i, sz in
               zip(tkf92_stats(m1, lam, mu, t, r), s_zero))
    return np.array(s0), np.array(s1)


def dtheta_dt(t, theta, lam0, mu0, x_ins, y_del):
    """RHS of the boxed parameter ODE (composition-renormalization.tex
    eq:ode-params).

    Only guards positivity: lambda may equal OR exceed mu, since the
    L'Hopital handling in `_tkf91_bdi_from_m` covers the singular case.
    """
    lam, mu, r = theta
    if not (lam > 1e-12 and mu > 1e-12 and 1e-9 < r < 1 - 1e-9):
        return np.zeros(3)
    try:
        s0, s1 = stats_and_increments(lam, mu, r, t, lam0, mu0, x_ins, y_del)
        _, _, S, F, En = s0
        bG, dG, sG, fG, eG = s1
        if S + t <= 0 or S <= 0 or F + En <= 0:
            return np.zeros(3)
        dlam = (bG - lam * (sG + 1.0)) / (S + t)
        dmu = (dG - mu * sG) / S
        dr = ((1 - r) * fG - r * eG) / (F + En)
        return np.array([dlam, dmu, dr])
    except Exception as ex:
        print(f"  [warn] dtheta_dt at t={t:.4f}: {ex}")
        return np.zeros(3)


def boundary_condition(lam0, mu0, x_ins, y_del):
    """Per-residue BO-slaved boundary at t -> 0+ (eq:boundary-r).

    Returns:
        r*(0)    = [lam0 y (1-x) + mu0 x (1-y)] / [lam0 (1-x) + mu0 (1-y)]
        lam_T(0) = lam0 / (1 - r*(0))
        mu_T(0)  = mu0  / (1 - r*(0))

    NOTE (2026-05-31): An earlier attempt minimized the per-residue
    conditional KL on per-link outcomes (insertion-runs etc.) and obtained
    (lam_G, mu_G, r*) instead of the BO slaving.  Per-tau M-projection of
    Gillespie SMIDE paths via kl_fit empirically matches the BO slaving,
    not that joint-KL minimum.  The discrepancy: kl_fit fits the compiled
    5x5 transition Markov chain, not the per-outcome distribution; the
    two M-projections agree on r but differ on the rate scale.  Restored
    to BO slaving for empirical consistency.

    Convention: x_ins = GGI insertion-extension (y in writeup),
                y_del = deletion-extension (x in writeup).
    """
    num = lam0 * x_ins * (1 - y_del) + mu0 * y_del * (1 - x_ins)
    den = lam0 * (1 - y_del) + mu0 * (1 - x_ins)
    r_star = num / den
    return lam0 / (1 - r_star), mu0 / (1 - r_star), r_star


def run_flow(lam0, mu0, x_ins, y_del, t_eps=5e-3, t_max=5.0, t_eval=None,
             rtol=1e-5, atol=1e-7, method='RK45', max_step=0.2):
    """Integrate the matched-flow ODE from t = t_eps to t = t_max.

    Returns (scipy ivp solution, (lam_T(0), mu_T(0), r*(0))).
    """
    lam_T0, mu_T0, r0 = boundary_condition(lam0, mu0, x_ins, y_del)
    if t_eval is None:
        t_eval = np.geomspace(t_eps, t_max, 40)
    sol = solve_ivp(
        fun=lambda t, theta: dtheta_dt(t, theta, lam0, mu0, x_ins, y_del),
        t_span=(t_eps, t_max),
        y0=np.array([lam_T0, mu_T0, r0]),
        t_eval=t_eval,
        method=method, rtol=rtol, atol=atol,
        max_step=max_step, dense_output=True,
    )
    return sol, (lam_T0, mu_T0, r0)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 72)
    print("Canonical matched-flow ODE -- self-test")
    print("=" * 72)

    # 1. Asymmetric GGI -- the regime where this ODE was originally
    #    validated against Gillespie (commit 47d4d4efc).  r should DECREASE.
    print("\n[1] Asymmetric (lam0=0.035, mu0=0.045, x_del=0.6, y_ins=0.4):")
    LAM0, MU0, X_DEL, Y_INS = 0.035, 0.045, 0.6, 0.4
    tic = time.time()
    sol, bc = run_flow(LAM0, MU0, x_ins=Y_INS, y_del=X_DEL,
                       t_eps=0.005, t_max=10.0,
                       t_eval=np.geomspace(0.005, 10.0, 15))
    print(f"  integration: {time.time()-tic:.1f}s, status={sol.status}")
    print(f"  boundary: lam={bc[0]:.4f}, mu={bc[1]:.4f}, r={bc[2]:.4f}")
    print(f"  {'t':>7}  {'lam':>9}  {'mu':>9}  {'r':>8}")
    for i in range(0, len(sol.t), 2):
        print(f"  {sol.t[i]:>7.3f}  {sol.y[0][i]:>9.5f}  "
              f"{sol.y[1][i]:>9.5f}  {sol.y[2][i]:>8.4f}")
    direction = "decreasing" if sol.y[2][-1] < sol.y[2][0] else "increasing"
    print(f"  r(t) is {direction} from {sol.y[2][0]:.4f} to {sol.y[2][-1]:.4f}")

    # 2. Pfam-symmetric -- the regime where the old recursive L'Hopital
    #    fallback stalled.  Should now run through lambda ~ mu cleanly.
    print("\n[2] Pfam joint-native (lam0~mu0~0.0107, x~y~0.64):")
    LAM0, MU0, X_DEL, Y_INS = 0.01061206, 0.01065096, 0.64072311, 0.63769192
    tic = time.time()
    sol, bc = run_flow(LAM0, MU0, x_ins=Y_INS, y_del=X_DEL,
                       t_eps=0.005, t_max=50.0,
                       t_eval=np.geomspace(0.005, 50.0, 20))
    print(f"  integration: {time.time()-tic:.1f}s, status={sol.status}")
    print(f"  boundary: lam={bc[0]:.5f}, mu={bc[1]:.5f}, r={bc[2]:.4f}")
    print(f"  {'t':>7}  {'lam':>9}  {'mu':>9}  {'r':>8}")
    for i in range(0, len(sol.t), 2):
        print(f"  {sol.t[i]:>7.3f}  {sol.y[0][i]:>9.5f}  "
              f"{sol.y[1][i]:>9.5f}  {sol.y[2][i]:>8.4f}")
    direction = "decreasing" if sol.y[2][-1] < sol.y[2][0] else "increasing"
    print(f"  r(t) is {direction} from {sol.y[2][0]:.4f} to {sol.y[2][-1]:.4f}")
