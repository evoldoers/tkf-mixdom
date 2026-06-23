"""6-panel evaluation: lam, mu, r, D(GGI||TKF92) via Gillespie (with error bars),
D(GGI||TKF92) via analytical integrand (Triad-based), D(TKF92_ode||TKF92_*).
"""
import sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d

sys.path.insert(0, '/Users/yam/tkf-mixdom/python')
sys.path.insert(0, '/Users/yam/tkf-mixdom/python/experiments')

from scratch_ggi_cond_kl_quad import (
    run_flow, boundary_condition, dtheta_dt, stats_and_increments,
    _null_eliminate_ID, _coarse_grain_12, SS_IDX_12, EE_IDX_12)
from scratch_ggi_triad_algebraic import build_AB
from scratch_ggi_flow import tau5, tau6_wfst, ml_fit, ml_fit_wfst, kar_to_lmr
from simulate_ggi_cherries import simulate_cherries


def get_m0_m1(lam, mu, r, t, lam0, mu0, x_ins, y_del):
    """Returns (m0, m1) — the surrogate and GGI-perturbation 5x5 count matrices."""
    Afull, Bfull, _ = build_AB(lam, mu, r, lam0, mu0, x_ins, y_del, t)
    U, V = _null_eliminate_ID(Afull, Bfull)
    C = np.linalg.inv(np.eye(12) - U)
    CVC = C @ V @ C
    n0 = np.outer(C[SS_IDX_12, :], C[:, EE_IDX_12]) * U
    n1 = (np.outer(CVC[SS_IDX_12, :], C[:, EE_IDX_12]) * U
          + np.outer(C[SS_IDX_12, :], C[:, EE_IDX_12]) * V
          + np.outer(C[SS_IDX_12, :], CVC[:, EE_IDX_12]) * U)
    return _coarse_grain_12(n0), _coarse_grain_12(n1)

LAM0, MU0 = 0.01061206, 0.01065096
X_DEL, Y_INS = 0.64072311, 0.63769192
T_MIN, T_MAX = 0.005, 50.0


def closed_form_r(lam0, mu0, x_del, y_ins, t):
    _, _, r0 = boundary_condition(lam0, mu0, y_ins, x_del)
    r_inf = r0 / (2.0 - r0)
    k = (lam0 + mu0) * (2.0 - r0) / (1.0 - r0)
    return r_inf + (r0 - r_inf) * np.exp(-k * t), r0, r_inf, k


def kl_transitions(N_emp, tau_th, eps=1e-12):
    """Joint Markov chain KL: empirical 5x5 vs 5x5 Pair HMM.  Includes ancestor
    singlet contribution (X->E and S->X factors), so will be contaminated by any
    ancestor-length mismatch between Gillespie (geom L_mean) and surrogate
    (geom 1/(1-kappa)).  Used only for legacy comparison."""
    kl_total = 0.0
    for i in range(5):
        row_total = N_emp[i].sum()
        if row_total < eps:
            continue
        p = N_emp[i] / row_total
        q = tau_th[i]
        mask = p > eps
        kl_total += row_total * np.sum(
            p[mask] * np.log(p[mask] / np.maximum(q[mask], eps)))
    return kl_total


def kl_anc_singlet(L_mean, lam, mu, r, eps=1e-12):
    """Analytic D( geom(1/L_mean) || tau_singlet(kappa, p) ) over ancestor
    length L >= 1.

    Empirical (sampling) ancestor:  P_emp(L=k) = (1-1/L_mean)^(k-1) * (1/L_mean)
    TKF92 singlet, conditioned on L>=1: P_tau(L=k|L>=1) = p^(k-1) * (1-p)
        with p = r + (1-r) * kappa, kappa = lam/mu.

    P_emp(L=0) = 0 (we sample L>=1).  P_tau(L=0) = 1-kappa; we condition on L>=1
    so the singlet contribution is the geometric on L>=1 with parameter (1-p).
    The (1-p) here ABSORBS the surrogate's "X->E from emit state" weight; the
    "S->A" weight kappa contributes a per-cherry constant log kappa - log P(L>=1)
    = log kappa - log kappa = 0, which is why it's not in this formula.
    """
    p_emp = 1.0 / L_mean
    kap = lam / mu
    p_tau = r + (1 - r) * kap
    # KL between two geometrics on k=1, 2, ... with success probs p_emp, 1-p_tau.
    return (np.log(p_emp) - np.log(1 - p_tau)
            + (1.0 / p_emp - 1.0) * (np.log(1 - p_emp) - np.log(p_tau)))


def kl_conditional_5state(N5_emp, lam, mu, r, t, L_mean, eps=1e-12):
    """Conditional D(P_GGI(desc|anc) || tau_WFST(desc|anc)) per cherry:
        D_cond = D_joint(emp PHMM || tau PHMM) - D_anc(geom(L_mean) || tau_singlet)
    The second term factors out the ancestor distribution analytically (the
    'divide P(anc,desc) by P(anc)' part)."""
    tau_phmm = tau5(lam / mu, np.exp(-mu * t), r)
    d_joint = 0.0
    for i in range(5):
        row_total = N5_emp[i].sum()
        if row_total < eps:
            continue
        p = N5_emp[i] / row_total
        q = tau_phmm[i]
        mask = (p > eps) & (q > eps)
        d_joint += row_total * np.sum(p[mask] * np.log(p[mask] / q[mask]))
    # N_cherries used in caller; here we return per-row-summed (will be divided).
    d_anc = kl_anc_singlet(L_mean, lam, mu, r)
    return d_joint, d_anc


def kl_two_tkf(theta_a, theta_b, t):
    lam_a, mu_a, r_a = theta_a
    lam_b, mu_b, r_b = theta_b
    tau_a = tau5(lam_a / mu_a, np.exp(-mu_a * t), r_a)
    tau_b = tau5(lam_b / mu_b, np.exp(-mu_b * t), r_b)
    eps = 1e-12
    kl = 0.0
    for i in range(4):  # weight rows S, M, I, D equally; skip E
        p = tau_a[i]; q = tau_b[i]
        mask = p > eps
        kl += np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps)))
    return kl


def analytical_kl_integrand(t, lam, mu, r, lam0, mu0, x_ins, y_del, dt=1e-3):
    """Per-alignment matched-flow projection rate D(m_source || m_target) / dt.

    NOTE on what this measures.  This is the LOCAL one-step matched-flow
    projection error rate: at time t, take the surrogate, apply one infinitesimal
    GGI step, then project back onto TKF92.  The KL of the projection is k(t)*dt.

    This is NOT D(GGI(t) || tau(t)) — it only equals D(GGI(t) || tau(t)) in the
    limit where the matched flow tracks GGI exactly.  When the matched-flow
    surrogate drifts (e.g. the Pfam ODE at moderate-to-large t), the integral
    of k stays small while D(GGI || tau) grows.  Compare to the empirical
    conditional-KL panel for the cumulative drift quantity.
    """
    dlam, dmu, dr = dtheta_dt(t, [lam, mu, r], lam0, mu0, x_ins, y_del)
    if not np.isfinite(dlam):
        return 0.0
    m0, m1 = get_m0_m1(lam, mu, r, t, lam0, mu0, x_ins, y_del)
    m_source = m0 + dt * m1
    lam_n = lam + dt * dlam
    mu_n = mu + dt * dmu
    r_n = r + dt * dr
    try:
        m_target, _ = get_m0_m1(lam_n, mu_n, r_n, t + dt, lam0, mu0, x_ins, y_del)
    except Exception:
        return 0.0
    eps = 1e-15
    kl = 0.0
    for i in range(5):
        rs = m_source[i].sum()
        rt = m_target[i].sum()
        if rs < eps or rt < eps:
            continue
        for j in range(5):
            if m_source[i, j] > eps and m_target[i, j] > eps:
                ps = m_source[i, j] / rs
                pt = m_target[i, j] / rt
                kl += m_source[i, j] * np.log(ps / pt)
    return kl / dt


# ============================================================
print(f"Pfam params: lam0={LAM0}, mu0={MU0}, x_del={X_DEL}, y_ins={Y_INS}")
t_grid = np.geomspace(T_MIN, T_MAX, 120)
t_gill = np.geomspace(T_MIN, T_MAX, 24)   # double the original 12

# --- Closed form ---
r_cf, r0, r_inf, k_decay = closed_form_r(LAM0, MU0, X_DEL, Y_INS, t_grid)
lam_cf = LAM0 / (1.0 - r_cf)
mu_cf  = MU0  / (1.0 - r_cf)
print(f"Closed form: r0={r0:.4f}, r_inf={r_inf:.4f}, k={k_decay:.4f}")

# --- Frozen-rates matched-flow r-ODE ---
lam_T0 = LAM0 / (1.0 - r0); mu_T0 = MU0 / (1.0 - r0)
lam_fz = np.full_like(t_grid, lam_T0); mu_fz = np.full_like(t_grid, mu_T0)
def dr_dt_frozen(t, r_arr):
    return [dtheta_dt(t, [lam_T0, mu_T0, float(r_arr[0])],
                       LAM0, MU0, Y_INS, X_DEL)[2]]
sol_fz = solve_ivp(dr_dt_frozen, (T_MIN, T_MAX), [r0], t_eval=t_grid,
                   method='RK45', rtol=1e-7, atol=1e-9, max_step=0.5)
r_fz = sol_fz.y[0]
print(f"Frozen-rate r-ODE: r_fz(0)={r_fz[0]:.4f}, r_fz(t_max)={r_fz[-1]:.4f}")

# --- Integrated ODE (full matched flow) ---
print("Integrating matched-flow ODE...")
tic = time.time()
sol, _ = run_flow(LAM0, MU0, x_ins=Y_INS, y_del=X_DEL,
                   t_eps=T_MIN, t_max=T_MAX, t_eval=t_grid,
                   rtol=1e-8, atol=1e-10)
lam_ode, mu_ode, r_ode = sol.y
print(f"  done in {time.time()-tic:.1f}s, status={sol.status}")

# --- Gillespie with error bars (multiple seeds) ---
# We compute the CONDITIONAL Gillespie KL on the 6-state recoded counts, so
# the ancestor distribution is divided out and L_MEAN can be anything (we
# pick the GGI-equilibrium length so the descendant statistics are at
# steady state).
ggi_eq_L = (LAM0 / (1 - Y_INS)) / (MU0 / (1 - X_DEL) - LAM0 / (1 - Y_INS))
L_MEAN = max(20, int(round(ggi_eq_L)))
print(f"L_MEAN set to GGI equilibrium length = {ggi_eq_L:.1f} -> {L_MEAN}")
N_CHERRIES = 3000
N_REPS = 5  # for error bars
print(f"Running Gillespie ({N_CHERRIES} cherries x {len(t_gill)} taus x {N_REPS} reps)...")
tic = time.time()
all_counts  = np.zeros((N_REPS, len(t_gill), 5, 5))
all_counts6 = np.zeros((N_REPS, len(t_gill), 6, 6))
for rep in range(N_REPS):
    all_counts[rep], all_counts6[rep] = simulate_cherries(
        LAM0, MU0, X_DEL, Y_INS, t_gill, N_CHERRIES, L_MEAN, seed=rep,
        with_6state=True)
print(f"  done in {time.time()-tic:.1f}s")

# --- KL panel 4: Gillespie -> 3 surrogates (mean + std over reps) ---
lam_ode_at = interp1d(t_grid, lam_ode); mu_ode_at = interp1d(t_grid, mu_ode); r_ode_at = interp1d(t_grid, r_ode)
lam_cf_at = interp1d(t_grid, lam_cf); mu_cf_at = interp1d(t_grid, mu_cf); r_cf_at = interp1d(t_grid, r_cf)
r_fz_at = interp1d(t_grid, r_fz)

def kl_for(theta_fn, t):
    th = theta_fn(t)
    tau = tau5(th[0]/th[1], np.exp(-th[1]*t), th[2])
    return tau

kl_ode_rep = np.zeros((N_REPS, len(t_gill)))
kl_cf_rep  = np.zeros((N_REPS, len(t_gill)))
kl_fz_rep  = np.zeros((N_REPS, len(t_gill)))
for rep in range(N_REPS):
    for ti, tt in enumerate(t_gill):
        th_o = (float(lam_ode_at(tt)), float(mu_ode_at(tt)), float(r_ode_at(tt)))
        th_c = (float(lam_cf_at(tt)),  float(mu_cf_at(tt)),  float(r_cf_at(tt)))
        th_f = (lam_T0, mu_T0, float(r_fz_at(tt)))
        # Conditional KL = D_joint(Markov chain) - D_anc(geom L_mean || tau_singlet).
        # D_joint is row-weighted by row counts (so it's "total joint LL diff per
        # cherry block"); D_anc is per-cherry, multiplied by N_CHERRIES to match
        # scale, then both get divided by N_CHERRIES at the end.
        for kk, th in zip((kl_ode_rep, kl_cf_rep, kl_fz_rep), (th_o, th_c, th_f)):
            d_joint, d_anc = kl_conditional_5state(all_counts[rep, ti],
                                                   th[0], th[1], th[2], tt, L_MEAN)
            kk[rep, ti] = d_joint - N_CHERRIES * d_anc
kl_ode_m, kl_ode_s = kl_ode_rep.mean(0), kl_ode_rep.std(0)
kl_cf_m,  kl_cf_s  = kl_cf_rep.mean(0),  kl_cf_rep.std(0)
kl_fz_m,  kl_fz_s  = kl_fz_rep.mean(0),  kl_fz_rep.std(0)

# --- Per-tau TKF92 ML fit to Gillespie (joint Pair HMM vs conditional WFST) ---
print("Fitting TKF92 per tau (joint Pair HMM + conditional WFST) on Gillespie...")
lam_gi_j = np.zeros((N_REPS, len(t_gill)));  mu_gi_j = np.zeros_like(lam_gi_j);  r_gi_j = np.zeros_like(lam_gi_j)
lam_gi_c = np.zeros_like(lam_gi_j);          mu_gi_c = np.zeros_like(lam_gi_j);  r_gi_c = np.zeros_like(lam_gi_j)
for rep in range(N_REPS):
    for ti, tt in enumerate(t_gill):
        N5_emp = all_counts[rep, ti]  / N_CHERRIES
        N6_emp = all_counts6[rep, ti] / N_CHERRIES
        try:
            kap_j, alph_j, r_j = ml_fit(N5_emp)
            lam_gi_j[rep, ti], mu_gi_j[rep, ti], r_gi_j[rep, ti] = kar_to_lmr(kap_j, alph_j, r_j, tt)
        except Exception:
            lam_gi_j[rep, ti] = mu_gi_j[rep, ti] = r_gi_j[rep, ti] = np.nan
        try:
            kap_c, alph_c, r_c = ml_fit_wfst(N6_emp)
            lam_gi_c[rep, ti], mu_gi_c[rep, ti], r_gi_c[rep, ti] = kar_to_lmr(kap_c, alph_c, r_c, tt)
        except Exception:
            lam_gi_c[rep, ti] = mu_gi_c[rep, ti] = r_gi_c[rep, ti] = np.nan
lam_gi_j_m, lam_gi_j_s = np.nanmean(lam_gi_j, 0), np.nanstd(lam_gi_j, 0)
mu_gi_j_m,  mu_gi_j_s  = np.nanmean(mu_gi_j,  0), np.nanstd(mu_gi_j,  0)
r_gi_j_m,   r_gi_j_s   = np.nanmean(r_gi_j,   0), np.nanstd(r_gi_j,   0)
lam_gi_c_m, lam_gi_c_s = np.nanmean(lam_gi_c, 0), np.nanstd(lam_gi_c, 0)
mu_gi_c_m,  mu_gi_c_s  = np.nanmean(mu_gi_c,  0), np.nanstd(mu_gi_c,  0)
r_gi_c_m,   r_gi_c_s   = np.nanmean(r_gi_c,   0), np.nanstd(r_gi_c,   0)
# Normalize KL per cherry (divide by N_CHERRIES)
kl_ode_m /= N_CHERRIES; kl_ode_s /= N_CHERRIES
kl_cf_m  /= N_CHERRIES; kl_cf_s  /= N_CHERRIES
kl_fz_m  /= N_CHERRIES; kl_fz_s  /= N_CHERRIES

# --- Panel 5: analytical KL integrand integrated ---
print("Computing analytical KL integrand along trajectories...")
def dtheta_at_t(t, lam_fn, mu_fn, r_fn):
    th = [float(lam_fn(t)), float(mu_fn(t)), float(r_fn(t))]
    return dtheta_dt(t, th, LAM0, MU0, Y_INS, X_DEL)

t_kl_grid = np.geomspace(T_MIN, T_MAX, 60)
kl_an_ode = np.zeros(len(t_kl_grid))
kl_an_cf  = np.zeros(len(t_kl_grid))
kl_an_fz  = np.zeros(len(t_kl_grid))
for ti, tt in enumerate(t_kl_grid):
    kl_an_ode[ti] = analytical_kl_integrand(
        tt, float(lam_ode_at(tt)), float(mu_ode_at(tt)), float(r_ode_at(tt)),
        LAM0, MU0, Y_INS, X_DEL)
    kl_an_cf[ti] = analytical_kl_integrand(
        tt, float(lam_cf_at(tt)), float(mu_cf_at(tt)), float(r_cf_at(tt)),
        LAM0, MU0, Y_INS, X_DEL)
    kl_an_fz[ti] = analytical_kl_integrand(
        tt, lam_T0, mu_T0, float(r_fz_at(tt)),
        LAM0, MU0, Y_INS, X_DEL)

D_an_ode = cumulative_trapezoid(kl_an_ode, t_kl_grid, initial=0)
D_an_cf  = cumulative_trapezoid(kl_an_cf,  t_kl_grid, initial=0)
D_an_fz  = cumulative_trapezoid(kl_an_fz,  t_kl_grid, initial=0)

# --- Panel 6: D(TKF92_ODE || TKF92_*) ---
kl_ode_vs_cf = np.zeros(len(t_grid))
kl_ode_vs_fz = np.zeros(len(t_grid))
for ti, tt in enumerate(t_grid):
    kl_ode_vs_cf[ti] = kl_two_tkf(
        (lam_ode[ti], mu_ode[ti], r_ode[ti]),
        (lam_cf[ti], mu_cf[ti], r_cf[ti]), tt)
    kl_ode_vs_fz[ti] = kl_two_tkf(
        (lam_ode[ti], mu_ode[ti], r_ode[ti]),
        (lam_T0, mu_T0, r_fz[ti]), tt)

# --- Plot ---
fig, axes = plt.subplots(6, 1, figsize=(10, 17), sharex=True)

PANEL_DATA = [
    (axes[0], lam_ode, lam_cf, lam_fz, lam_gi_j_m, lam_gi_j_s, lam_gi_c_m, lam_gi_c_s, r'$\lambda(t)$', LAM0,    'orange'),
    (axes[1], mu_ode,  mu_cf,  mu_fz,  mu_gi_j_m,  mu_gi_j_s,  mu_gi_c_m,  mu_gi_c_s,  r'$\mu(t)$',    MU0,     'orange'),
    (axes[2], r_ode,   r_cf,   r_fz,   r_gi_j_m,   r_gi_j_s,   r_gi_c_m,   r_gi_c_s,   r'$r(t)$',      None,    None)]
for (ax, val_ode, val_cf, val_fz, val_gi_j_m, val_gi_j_s, val_gi_c_m, val_gi_c_s,
     name, const, const_col) in PANEL_DATA:
    ax.plot(t_grid, val_ode, 'k-', lw=2, label='ODE (full)')
    ax.plot(t_grid, val_cf,  'b--', lw=1.5, label='closed form')
    ax.plot(t_grid, val_fz,  'r:',  lw=1.8, label=r'$\lambda,\mu$ frozen, r-ODE')
    ax.errorbar(t_gill, val_gi_j_m, yerr=val_gi_j_s, fmt='m^', ms=5, lw=1.0,
                capsize=2, alpha=0.8,
                label='ML fit (joint Pair HMM)')
    ax.errorbar(t_gill, val_gi_c_m, yerr=val_gi_c_s, fmt='cv', ms=5, lw=1.0,
                capsize=2, alpha=0.8,
                label='ML fit (conditional WFST)')
    if const is not None:
        ax.axhline(const, color=const_col, ls='-.', lw=0.9,
                   label=fr'$\lambda_0$' if name == r'$\lambda(t)$' else r'$\mu_0$')
    if name == r'$r(t)$':
        ax.axhline(r0,    color='darkgreen', ls='-.', lw=0.9, label=fr'$r^*(0)={r0:.3f}$')
        ax.axhline(r_inf, color='purple',     ls='-.', lw=0.9, label=fr'$r_\infty={r_inf:.3f}$')
    ax.set_ylabel(name, fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')

ax = axes[3]
ax.errorbar(t_gill, kl_ode_m, yerr=kl_ode_s, fmt='ko-', ms=5, lw=1.2, capsize=3,
            label=r'$D_\text{cond}(\text{GGI}\,\Vert\,\text{ODE})$, Gillespie')
ax.errorbar(t_gill, kl_cf_m,  yerr=kl_cf_s,  fmt='bs--', ms=5, lw=1.2, capsize=3,
            label=r'$D_\text{cond}(\text{GGI}\,\Vert\,\text{closed})$, Gillespie')
ax.errorbar(t_gill, kl_fz_m,  yerr=kl_fz_s,  fmt='rD:', ms=5, lw=1.2, capsize=3,
            label=r'$D_\text{cond}(\text{GGI}\,\Vert\,\lambda,\mu\text{ frozen})$, Gillespie')
ax.set_ylabel('cond. KL (nats/cherry)', fontsize=11)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3, which='both')
ax.set_yscale('log')

ax = axes[4]
ax.plot(t_kl_grid, D_an_ode, 'k-', lw=2,
        label=r'$\int_0^t k_\text{ODE}\,ds$')
ax.plot(t_kl_grid, D_an_cf,  'b--', lw=1.5,
        label=r'$\int_0^t k_\text{closed}\,ds$')
ax.plot(t_kl_grid, D_an_fz,  'r:',  lw=1.8,
        label=r'$\int_0^t k_\text{frozen}\,ds$')
ax.set_ylabel(r'analytical KL', fontsize=11)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3, which='both')
ax.set_yscale('log')

ax = axes[5]
ax.plot(t_grid, kl_ode_vs_cf, 'b-', lw=2,
        label=r'$D(\text{ODE}\,\Vert\,\text{closed})$')
ax.plot(t_grid, kl_ode_vs_fz, 'r-', lw=2,
        label=r'$D(\text{ODE}\,\Vert\,\lambda,\mu\text{ frozen})$')
ax.set_ylabel('KL', fontsize=11)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3, which='both')
ax.set_yscale('log')
ax.set_xlabel(r'time $t$ (log)', fontsize=12)

fig.suptitle(
    f'Pfam joint-native, $t \\in [0.005, 50]$: '
    f'$\\lambda_0={LAM0}$, $\\mu_0={MU0}$, $x={X_DEL:.3f}$, $y={Y_INS:.3f}$', fontsize=10)
plt.tight_layout()
out = '/tmp/closed_form_eval_pfam6.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f'Saved {out}')

# Also save trajectories for the LaTeX
np.savez('/tmp/closed_form_eval_pfam.npz',
         t_grid=t_grid, lam_ode=lam_ode, mu_ode=mu_ode, r_ode=r_ode,
         lam_cf=lam_cf, mu_cf=mu_cf, r_cf=r_cf, r_fz=r_fz,
         t_gill=t_gill, kl_ode_m=kl_ode_m, kl_ode_s=kl_ode_s,
         kl_cf_m=kl_cf_m, kl_cf_s=kl_cf_s, kl_fz_m=kl_fz_m, kl_fz_s=kl_fz_s,
         t_kl_grid=t_kl_grid, D_an_ode=D_an_ode, D_an_cf=D_an_cf, D_an_fz=D_an_fz,
         kl_ode_vs_cf=kl_ode_vs_cf, kl_ode_vs_fz=kl_ode_vs_fz)
print('Saved /tmp/closed_form_eval_pfam.npz')
