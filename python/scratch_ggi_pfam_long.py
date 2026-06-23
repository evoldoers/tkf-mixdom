"""Pfam (x=0.65, y=0.71) flow out to t=10^3:
   stacked (lambda, mu) over r, full 3D and slaved 1D only.
"""
import sys, time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/yam/tkf-mixdom/python')

from scratch_ggi_triad_eliminated import (
    dtheta_dt, boundary_condition, run_flow as run_full,
)


def dr_dt_slaved(t, r, lam0, mu0, x_ins, y_del, du=2e-4):
    r = float(r)
    if not (0 < r < 0.999): return 0.0
    lam, mu = lam0/(1 - r), mu0/(1 - r)
    if lam >= mu or lam <= 0: return 0.0
    _, _, dr = dtheta_dt(t, np.array([lam, mu, r]), lam0, mu0, x_ins, y_del, du=du)
    return float(dr)


def run_slaved(lam0, mu0, x_ins, y_del, t_eps, t_max, t_eval):
    _, _, r0 = boundary_condition(lam0, mu0, x_ins, y_del)
    return solve_ivp(
        fun=lambda t, r: [dr_dt_slaved(t, r[0], lam0, mu0, x_ins, y_del)],
        t_span=(t_eps, t_max), y0=[r0], t_eval=t_eval,
        method='RK23', rtol=1e-4, atol=1e-7, max_step=50.0,
    ), r0


# Pfam config
LAM_PFAM, MU_PFAM, R_PFAM = 0.0458, 0.0468, 0.683
LAM0 = LAM_PFAM * (1 - R_PFAM)
MU0 = MU_PFAM * (1 - R_PFAM)
x_ins, y_del = 0.65, 0.7098

bc_l, bc_m, bc_r = boundary_condition(LAM0, MU0, x_ins, y_del)
print(f"Pfam GGI: lam0={LAM0:.5f}, mu0={MU0:.5f}, x={x_ins}, y={y_del}")
print(f"Boundary: lam(0)={bc_l:.5f}, mu(0)={bc_m:.5f}, r(0)={bc_r:.4f}")

t_eps = 1e-2
t_max = 1000.0
t_eval = np.geomspace(t_eps, t_max, 80)

tic = time.time()
full, _ = run_full(LAM0, MU0, x_ins, y_del, t_eps=t_eps, t_max=t_max, t_eval=t_eval)
print(f"Full 3-D: {time.time()-tic:.1f}s, status={full.status}, reached t={full.t[-1]:.1f}")
print(f"  r min at t={full.t[np.argmin(full.y[2])]:.2f}: r={full.y[2].min():.4f}")
print(f"  r end at t={full.t[-1]:.2f}: r={full.y[2,-1]:.4f}")

tic = time.time()
slv, _ = run_slaved(LAM0, MU0, x_ins, y_del, t_eps, t_max, t_eval)
print(f"Slaved:   {time.time()-tic:.1f}s, status={slv.status}, reached t={slv.t[-1]:.1f}")
print(f"  r min at t={slv.t[np.argmin(slv.y[0])]:.2f}: r={slv.y[0].min():.4f}")
print(f"  r end at t={slv.t[-1]:.2f}: r={slv.y[0,-1]:.4f}")

# Stacked plot
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax = axes[0]
ax.semilogx(full.t, full.y[0], 'b-', lw=2, label=r'$\lambda(t)$ full 3-D')
slv_lam = LAM0 / (1 - slv.y[0])
ax.semilogx(slv.t, slv_lam, 'r--', lw=2, label=r'$\lambda(t)=\lambda_0/(1{-}r(t))$ slaved')
ax.axhline(bc_l, color='gray', ls=':', lw=1, label=fr'$\lambda^*(0)={bc_l:.4f}$')
ax.set_ylabel(r'$\lambda(t)$')
ax.set_title(f"Pfam GGI ($\\lambda_0$={LAM0:.4f}, $\\mu_0$={MU0:.4f}, $x$={x_ins}, $y$={y_del}): flow out to $t = 10^3$",
             fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc='best', fontsize=10)

ax = axes[1]
ax.semilogx(full.t, full.y[1], 'b-', lw=2, label=r'$\mu(t)$ full 3-D')
slv_mu = MU0 / (1 - slv.y[0])
ax.semilogx(slv.t, slv_mu, 'r--', lw=2, label=r'$\mu(t)=\mu_0/(1{-}r(t))$ slaved')
ax.axhline(bc_m, color='gray', ls=':', lw=1, label=fr'$\mu^*(0)={bc_m:.4f}$')
ax.set_ylabel(r'$\mu(t)$')
ax.grid(alpha=0.3)
ax.legend(loc='best', fontsize=10)

ax = axes[2]
ax.semilogx(full.t, full.y[2], 'b-', lw=2, label=r'$r(t)$ full 3-D')
ax.semilogx(slv.t, slv.y[0], 'r--', lw=2, label=r'$r(t)$ slaved')
ax.axhline(bc_r, color='gray', ls=':', lw=1, label=fr'$r^*(0)={bc_r:.4f}$')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$r(t)$')
ax.grid(alpha=0.3)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('/tmp/pfam_longtime_stacked.png', dpi=110, bbox_inches='tight')
print(f"\nSaved /tmp/pfam_longtime_stacked.png")
