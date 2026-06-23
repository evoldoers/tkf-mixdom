"""4-panel figure for the 1-parameter (fragment-length) matched-flow ODE on Pfam
joint-native GGI params.  Compares the single-parameter flow (lambda, mu held
fixed at the BO boundary; only fragext = r evolves) to a closed-form
approximation, with Gillespie ground truth from per-tau ML fits to the WFST.

Panels:
  1. lambda(t): closed-form approximation, fragment-length flow, ML fit to
     Gillespie simulation.  (lambda is constant under the 1-param flow.)
  2. mu(t): same.
  3. r(t): same three curves plus r*(0) and r_inf asymptotes.
  4. Conditional KL D(P_GGI(desc|anc) || tau(desc|anc)) per cherry, linear y:
     Gillespie counts compared to the 1-param flow and the closed form, with
     error bars from N_REPS independent simulations.

This is the paper figure for the matched-flow-renormalisation section.  See
pfam_flow_6panel.py for the diagnostic 6-panel version that also shows the
3-parameter (triple-parameter) flow and the matched-flow projection-rate
integrand k(t).
"""
import sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

sys.path.insert(0, '/Users/yam/tkf-mixdom/python')
sys.path.insert(0, '/Users/yam/tkf-mixdom/python/experiments')

from scratch_ggi_cond_kl_quad import (
    run_flow, boundary_condition, dtheta_dt)
from scratch_ggi_flow import tau5, ml_fit_wfst, kar_to_lmr
from simulate_ggi_cherries import simulate_cherries


LAM0, MU0 = 0.01061206, 0.01065096
X_DEL, Y_INS = 0.64072311, 0.63769192
T_MIN, T_MAX = 0.005, 50.0


def closed_form_r(lam0, mu0, x_del, y_ins, t):
    _, _, r0 = boundary_condition(lam0, mu0, y_ins, x_del)
    r_inf = r0 / (2.0 - r0)
    k = (lam0 + mu0) * (2.0 - r0) / (1.0 - r0)
    return r_inf + (r0 - r_inf) * np.exp(-k * t), r0, r_inf, k


def kl_anc_singlet(L_mean, lam, mu, r):
    p_emp = 1.0 / L_mean
    kap = lam / mu
    p_tau = r + (1 - r) * kap
    return (np.log(p_emp) - np.log(1 - p_tau)
            + (1.0 / p_emp - 1.0) * (np.log(1 - p_emp) - np.log(p_tau)))


def kl_conditional_5state(N5_emp, lam, mu, r, t, L_mean, n_cherries, eps=1e-12):
    """D_cond(GGI || tau) per cherry = D_joint(emp || tau_PHMM)/n_cherries - D_anc."""
    tau_phmm = tau5(lam / mu, np.exp(-mu * t), r)
    d_joint_total = 0.0
    for i in range(5):
        rt = N5_emp[i].sum()
        if rt < eps:
            continue
        p = N5_emp[i] / rt
        q = tau_phmm[i]
        m = (p > eps) & (q > eps)
        d_joint_total += rt * np.sum(p[m] * np.log(p[m] / q[m]))
    d_anc = kl_anc_singlet(L_mean, lam, mu, r)
    return d_joint_total / n_cherries - d_anc


# ============================================================
print(f"Pfam params: lam0={LAM0}, mu0={MU0}, x_del={X_DEL}, y_ins={Y_INS}")
t_grid = np.geomspace(T_MIN, T_MAX, 120)
t_gill = np.geomspace(T_MIN, T_MAX, 24)

# --- Closed-form approximation ---
r_cf, r0, r_inf, k_decay = closed_form_r(LAM0, MU0, X_DEL, Y_INS, t_grid)
lam_cf = LAM0 / (1.0 - r_cf)
mu_cf  = MU0  / (1.0 - r_cf)
print(f"Closed-form approximation: r*(0)={r0:.4f}, r_inf={r_inf:.4f}, k={k_decay:.4f}")

# --- Fragment-length flow (single-parameter, lambda, mu held at boundary) ---
lam_T0 = LAM0 / (1.0 - r0); mu_T0 = MU0 / (1.0 - r0)
lam_fz = np.full_like(t_grid, lam_T0); mu_fz = np.full_like(t_grid, mu_T0)
def dr_dt_frozen(t, r_arr):
    return [dtheta_dt(t, [lam_T0, mu_T0, float(r_arr[0])],
                       LAM0, MU0, Y_INS, X_DEL)[2]]
sol_fz = solve_ivp(dr_dt_frozen, (T_MIN, T_MAX), [r0], t_eval=t_grid,
                   method='RK45', rtol=1e-7, atol=1e-9, max_step=0.5)
r_fz = sol_fz.y[0]
print(f"Fragment-length flow: r(0)={r_fz[0]:.4f}, r(t_max)={r_fz[-1]:.4f}")

# --- Gillespie with adaptive per-tau cherry counts ---
# Schedule: N_cherries(t) = max(N_floor, r(1-r) / (sigma^2 * R * t))
# where R is the per-cherry decision rate at the GGI equilibrium and sigma is
# the target absolute SE on the per-tau ML estimate of r.
ggi_eq_L = (LAM0 / (1 - Y_INS)) / (MU0 / (1 - X_DEL) - LAM0 / (1 - Y_INS))
L_MEAN = max(20, int(round(ggi_eq_L)))

# Effective per-cherry decision rate (insertion events + deletion events
# per unit time, weighted by fragment-length geometric extension probabilities
# so each event contributes ~1 decision on average): see derivation in the
# LaTeX section sec:comp-adaptive-sampling.
R_eff = ((L_MEAN + 1) * LAM0 / (1 - Y_INS)
         + L_MEAN * MU0 / (1 - X_DEL))
SIGMA_R = 0.005           # target absolute SE on r_hat per tau
N_FLOOR = 3000            # floor: at least the previous uniform budget
N_CAP = 400_000           # don't blow up at tiniest tau in case R_eff is wrong

def schedule(t, r_guess=0.5):
    return np.clip(np.round(r_guess * (1 - r_guess) / (SIGMA_R ** 2 * R_eff * t)),
                   N_FLOOR, N_CAP).astype(int)

N_per_tau = schedule(t_gill, r_guess=r0)   # r_guess at boundary
N_REPS = 5
print(f"L_MEAN = {L_MEAN} (GGI equilibrium length)")
print(f"R_eff = {R_eff:.3f} decisions/time/cherry (per-cherry decision rate)")
print(f"Adaptive schedule (target SE(r_hat) = {SIGMA_R}, floor = {N_FLOOR}):")
for tt, nn in zip(t_gill, N_per_tau):
    print(f"  t={tt:8.4f}  N_cherries={nn:6d}")
print(f"  total per rep = {N_per_tau.sum():,}; total over {N_REPS} reps = {N_REPS * N_per_tau.sum():,}")
print()

print(f"Running Gillespie ({N_REPS} reps)...")
tic = time.time()
all_counts  = np.zeros((N_REPS, len(t_gill), 5, 5))
all_counts6 = np.zeros((N_REPS, len(t_gill), 6, 6))
for rep in range(N_REPS):
    all_counts[rep], all_counts6[rep] = simulate_cherries(
        LAM0, MU0, X_DEL, Y_INS, t_gill, N_per_tau, L_MEAN, seed=rep,
        with_6state=True)
print(f"  done in {time.time()-tic:.1f}s")

# --- ML fit to Gillespie (conditional WFST) ---
print("ML fit (conditional WFST) per tau on Gillespie...")
lam_ml = np.zeros((N_REPS, len(t_gill)))
mu_ml  = np.zeros_like(lam_ml)
r_ml   = np.zeros_like(lam_ml)
for rep in range(N_REPS):
    for ti, tt in enumerate(t_gill):
        N6_emp = all_counts6[rep, ti] / N_per_tau[ti]
        try:
            kap, alph, rr = ml_fit_wfst(N6_emp)
            lam_ml[rep, ti], mu_ml[rep, ti], r_ml[rep, ti] = kar_to_lmr(kap, alph, rr, tt)
        except Exception:
            lam_ml[rep, ti] = mu_ml[rep, ti] = r_ml[rep, ti] = np.nan
lam_ml_m, lam_ml_s = np.nanmean(lam_ml, 0), np.nanstd(lam_ml, 0)
mu_ml_m,  mu_ml_s  = np.nanmean(mu_ml,  0), np.nanstd(mu_ml,  0)
r_ml_m,   r_ml_s   = np.nanmean(r_ml,   0), np.nanstd(r_ml,   0)

# --- Conditional KL panels (Gillespie counts vs. each surrogate) ---
lam_cf_at = interp1d(t_grid, lam_cf); mu_cf_at = interp1d(t_grid, mu_cf); r_cf_at = interp1d(t_grid, r_cf)
r_fz_at  = interp1d(t_grid, r_fz)

kl_cf_rep = np.zeros((N_REPS, len(t_gill)))
kl_fz_rep = np.zeros((N_REPS, len(t_gill)))
for rep in range(N_REPS):
    for ti, tt in enumerate(t_gill):
        th_c = (float(lam_cf_at(tt)), float(mu_cf_at(tt)), float(r_cf_at(tt)))
        th_f = (lam_T0, mu_T0, float(r_fz_at(tt)))
        kl_cf_rep[rep, ti] = kl_conditional_5state(
            all_counts[rep, ti], th_c[0], th_c[1], th_c[2], tt, L_MEAN, N_per_tau[ti])
        kl_fz_rep[rep, ti] = kl_conditional_5state(
            all_counts[rep, ti], th_f[0], th_f[1], th_f[2], tt, L_MEAN, N_per_tau[ti])
kl_cf_m, kl_cf_s = kl_cf_rep.mean(0), kl_cf_rep.std(0)
kl_fz_m, kl_fz_s = kl_fz_rep.mean(0), kl_fz_rep.std(0)

# --- Plot ---
t_half = np.log(2.0) / k_decay
print(f"r(t) half-life: t_half = ln(2)/k = {t_half:.4f}")

NATS_TO_BITS = 1.0 / np.log(2.0)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

HALF_LIFE_COLOR = 'darkorange'

# Panel 1: r(t)
ax = axes[0]
ax.plot(t_grid, r_fz, 'r-',  lw=2,   label='numerically integrated flow')
ax.plot(t_grid, r_cf, 'b--', lw=1.5, label='closed-form approximation')
ax.errorbar(t_gill, r_ml_m, yerr=r_ml_s, fmt='cv', ms=5, lw=1.0,
            capsize=2, alpha=0.8, label='ML fit to Gillespie simulation')
ax.axvline(t_half, color=HALF_LIFE_COLOR, ls='-.', lw=1.6,
           label=fr'$t_{{1/2}}^{{\mathrm{{frag}}}} = \ln 2/k = {t_half:.3f}$')
ax.set_ylabel(r'$r(t)$', fontsize=12)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3, which='both')
ax.set_xscale('log')

# Panel 2: D(GGI simulation || surrogate), linear y axis, in bits.
ax = axes[1]
ax.errorbar(t_gill, kl_fz_m * NATS_TO_BITS, yerr=kl_fz_s * NATS_TO_BITS,
            fmt='ro-',  ms=6, lw=1.5, capsize=4, elinewidth=1.8,
            label=r'$D(\text{GGI simulation}\,\Vert\,\text{integrated flow})$')
ax.errorbar(t_gill, kl_cf_m * NATS_TO_BITS, yerr=kl_cf_s * NATS_TO_BITS,
            fmt='bs--', ms=6, lw=1.5, capsize=4, elinewidth=1.8,
            label=r'$D(\text{GGI simulation}\,\Vert\,\text{closed form})$')
ax.axvline(t_half, color=HALF_LIFE_COLOR, ls='-.', lw=1.6)
ax.set_ylabel('cond. KL (bits/cherry)', fontsize=11)
ax.set_xlabel(r'time $t$ (log)', fontsize=12)
ax.set_xscale('log')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3, which='both')

fig.suptitle(
    f'Pfam joint-native, fragment-length flow: '
    f'$\\lambda_0={LAM0}$, $\\mu_0={MU0}$, $x={X_DEL:.3f}$, $y={Y_INS:.3f}$',
    fontsize=10)
plt.tight_layout()
out = '/tmp/pfam_flow_1param.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f'Saved {out}')

np.savez('/tmp/pfam_flow_1param.npz',
         t_grid=t_grid, lam_fz=lam_fz, mu_fz=mu_fz, r_fz=r_fz,
         lam_cf=lam_cf, mu_cf=mu_cf, r_cf=r_cf,
         t_gill=t_gill,
         lam_ml_m=lam_ml_m, lam_ml_s=lam_ml_s,
         mu_ml_m=mu_ml_m,  mu_ml_s=mu_ml_s,
         r_ml_m=r_ml_m,   r_ml_s=r_ml_s,
         kl_cf_m=kl_cf_m, kl_cf_s=kl_cf_s,
         kl_fz_m=kl_fz_m, kl_fz_s=kl_fz_s,
         r0=r0, r_inf=r_inf, lam_T0=lam_T0, mu_T0=mu_T0)
print('Saved /tmp/pfam_flow_1param.npz')
