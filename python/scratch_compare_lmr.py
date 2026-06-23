#!/usr/bin/env python3
"""compare_lmr_closed_vs_ode.py
Overlay closed-form lambda(t), mu(t), r(t) slavings against the full 3-D
matched-flow ODE.

Existing closed-form r(t)  (kept fixed across both slaving variants):
    r_inf  = r*(0) / (2 - r*(0))
    k      = (lam0 + mu0)(2 - r*(0)) / (1 - r*(0))
    r(t)   = r_inf + (r*(0) - r_inf) exp(-k t)

Slaving (BO) — the current Born-Oppenheimer heuristic:
    lam(t) = lam0 / (1 - r(t))
    mu(t)  = mu0  / (1 - r(t))

Slaving (S) — ENFORCES nu = x/y constant AND matches GGI's total per-residue
indel turnover rate:
    nu0   = x/y
    kap(t) = (nu0 - r(t)) / (1 - r(t))
    rho_G  = lam0/(1 - y) + mu0/(1 - x)
    mu(t)  = (1 - r(t))**2 rho_G / (1 + nu0 - 2 r(t))
    lam(t) = kap(t) mu(t)

Reference: 3-D matched-flow ODE from scratch_ggi_cond_kl_quad.run_flow.
Defaults: Pfam joint-native ML fit (closest to the whole-Pfam TKF92 CherryML
fit; both collapse to lam0 = mu0 to within 0.5%, x ~ y).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/yam/tkf-mixdom/python')
from scratch_ggi_cond_kl_quad import run_flow, boundary_condition


# ---------------------------------------------------------------------------
# Closed-form pieces
# ---------------------------------------------------------------------------
def closed_form_r(lam0, mu0, x_del, y_ins, t):
    # boundary_condition arg order is (lam0, mu0, x_ins, y_del).
    # User notation has x = x_del (deletion extension), y = y_ins (insertion).
    # Code's x_ins == user's y_ins; code's y_del == user's x_del.
    _, _, r0 = boundary_condition(lam0, mu0, y_ins, x_del)
    r_inf = r0 / (2.0 - r0)
    k = (lam0 + mu0) * (2.0 - r0) / (1.0 - r0)
    r_t = r_inf + (r0 - r_inf) * np.exp(-k * t)
    return r_t, r0, r_inf, k


def slaving_BO(lam0, mu0, x_del, y_ins, t):
    r_t, *_ = closed_form_r(lam0, mu0, x_del, y_ins, t)
    return lam0 / (1.0 - r_t), mu0 / (1.0 - r_t), r_t


def slaving_S(lam0, mu0, x_del, y_ins, t):
    r_t, *_ = closed_form_r(lam0, mu0, x_del, y_ins, t)
    nu0 = x_del / y_ins
    rho_G = lam0 / (1.0 - y_ins) + mu0 / (1.0 - x_del)
    denom = 1.0 + nu0 - 2.0 * r_t
    mu_t = (1.0 - r_t) ** 2 * rho_G / denom
    kap_t = (nu0 - r_t) / (1.0 - r_t)
    lam_t = kap_t * mu_t
    return lam_t, mu_t, r_t


# ---------------------------------------------------------------------------
# GGI params: Pfam joint-native ML fit (s3 .../pfam-gap-counts/2026-05-29/)
# ---------------------------------------------------------------------------
LAM0   = 0.01061206
MU0    = 0.01065096
X_DEL  = 0.64072311
Y_INS  = 0.63769192

print(f"GGI params (Pfam joint-native ML fit):")
print(f"  lam0  = {LAM0:.6f}")
print(f"  mu0   = {MU0:.6f}")
print(f"  x_del = {X_DEL:.4f}")
print(f"  y_ins = {Y_INS:.4f}")
print(f"  nu0   = x/y     = {X_DEL/Y_INS:.4f}")
print(f"  lam0/mu0        = {LAM0/MU0:.4f}")
print(f"  rho_G           = {LAM0/(1-Y_INS) + MU0/(1-X_DEL):.4f}")

T_EPS = 0.005
T_MAX = 50.0
N_T   = 200
t_grid = np.geomspace(T_EPS, T_MAX, N_T)

# closed-form bookkeeping
_, r0, r_inf, k_decay = closed_form_r(LAM0, MU0, X_DEL, Y_INS, T_EPS)
print(f"\nClosed-form r(t):")
print(f"  r*(0)  = {r0:.4f}")
print(f"  r_inf  = {r_inf:.4f}")
print(f"  k      = {k_decay:.4f}  (decay scale 1/k = {1/k_decay:.2f})")

lam_BO, mu_BO, r_cf = slaving_BO(LAM0, MU0, X_DEL, Y_INS, t_grid)
lam_S,  mu_S,  _    = slaving_S (LAM0, MU0, X_DEL, Y_INS, t_grid)

print(f"\nBoundary values at t -> 0:")
print(f"  BO        : lam(0) = {lam_BO[0]:.5f}, mu(0) = {mu_BO[0]:.5f}")
print(f"  Scheme (S): lam(0) = {lam_S[0]:.5f}, mu(0) = {mu_S[0]:.5f}")
print(f"\nLong-time limits (t = {T_MAX}):")
print(f"  BO        : lam = {lam_BO[-1]:.5f}, mu = {mu_BO[-1]:.5f}")
print(f"  Scheme (S): lam = {lam_S[-1]:.5f}, mu = {mu_S[-1]:.5f}")

# ---------------------------------------------------------------------------
# 3-D matched-flow ODE
# ---------------------------------------------------------------------------
print(f"\nIntegrating 3-D matched-flow ODE ... ", end='', flush=True)
sol, bc = run_flow(LAM0, MU0, x_ins=Y_INS, y_del=X_DEL,
                   t_eps=T_EPS, t_max=T_MAX, t_eval=t_grid)
print(f"done.  status={sol.status} ('{sol.message}')")
print(f"  ODE boundary (lam_T(0), mu_T(0), r(0)) = "
      f"({bc[0]:.5f}, {bc[1]:.5f}, {bc[2]:.4f})")
lam_ode = np.asarray(sol.y[0])
mu_ode  = np.asarray(sol.y[1])
r_ode   = np.asarray(sol.y[2])
# scipy may stop early; pad with NaN past the stop time
t_ode = sol.t
if len(t_ode) < N_T:
    keep = len(t_ode)
    pad = np.full(N_T - keep, np.nan)
    lam_ode = np.concatenate([lam_ode, pad])
    mu_ode  = np.concatenate([mu_ode,  pad])
    r_ode   = np.concatenate([r_ode,   pad])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)

ax = axes[0]
ax.plot(t_grid, lam_ode, 'k-',  lw=2.0, label='3-D ODE flow')
ax.plot(t_grid, lam_BO,  'b--', lw=1.5,
        label=r'closed form, BO slaving: $\lambda_0/(1-r(t))$')
ax.plot(t_grid, lam_S,   'r--', lw=1.5,
        label=r'closed form, scheme (S): $\nu_0 = x/y$, $\rho$ matched')
ax.axhline(LAM0, color='gray', ls=':', lw=0.8,
           label=fr'$\lambda_0 = {LAM0:.5f}$')
ax.set_ylabel(r'$\lambda(t)$')
ax.set_xscale('log')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
ax.plot(t_grid, mu_ode, 'k-',  lw=2.0, label='3-D ODE flow')
ax.plot(t_grid, mu_BO,  'b--', lw=1.5,
        label=r'closed form, BO slaving: $\mu_0/(1-r(t))$')
ax.plot(t_grid, mu_S,   'r--', lw=1.5,
        label=r'closed form, scheme (S)')
ax.axhline(MU0, color='gray', ls=':', lw=0.8,
           label=fr'$\mu_0 = {MU0:.5f}$')
ax.set_ylabel(r'$\mu(t)$')
ax.set_xscale('log')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3, which='both')

ax = axes[2]
ax.plot(t_grid, r_ode, 'k-',  lw=2.0, label='3-D ODE flow')
ax.plot(t_grid, r_cf,  'b--', lw=1.5,
        label='closed form (same for BO and (S))')
ax.axhline(r0,    color='gray',      ls=':', lw=0.8,
           label=fr'$r^*(0) = {r0:.3f}$')
ax.axhline(r_inf, color='lightgray', ls=':', lw=0.8,
           label=fr'$r_\infty = {r_inf:.3f}$')
ax.set_ylabel(r'$r(t)$')
ax.set_xlabel(r'time $t$')
ax.set_xscale('log')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3, which='both')

fig.suptitle(
    f'Closed-form $\\lambda(t), \\mu(t), r(t)$ vs full 3-D matched-flow ODE\n'
    f'GGI Pfam joint-native ML fit: '
    f'$\\lambda_0={LAM0:.5f}$, $\\mu_0={MU0:.5f}$, '
    f'$x={X_DEL:.3f}$, $y={Y_INS:.3f}$, $\\nu_0=x/y={X_DEL/Y_INS:.3f}$',
    fontsize=10)
plt.tight_layout()
out_png = '/tmp/lmr_closed_vs_ode_pfam.png'
plt.savefig(out_png, dpi=120, bbox_inches='tight')
print(f"\nSaved {out_png}")
