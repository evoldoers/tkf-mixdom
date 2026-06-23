"""Long-time r(t) trajectories: full 3D vs slaved vs frozen, out to t=100.

The (0.4, 0.55) case shows non-monotonic 'u-turn' in r(t) in the
null-eliminated Triad ODE — this is to investigate whether that kink is
real (occurring even when lam, mu are allowed to evolve), and where the
flow eventually settles.
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


def dr_dt_frozen(t, r, lam_fix, mu_fix, lam0, mu0, x_ins, y_del, du=2e-4):
    r = float(r)
    if not (0 < r < 0.999): return 0.0
    _, _, dr = dtheta_dt(t, np.array([lam_fix, mu_fix, r]), lam0, mu0, x_ins, y_del, du=du)
    return float(dr)


def run_slaved(lam0, mu0, x_ins, y_del, t_eps, t_max, t_eval):
    _, _, r0 = boundary_condition(lam0, mu0, x_ins, y_del)
    return solve_ivp(
        fun=lambda t, r: [dr_dt_slaved(t, r[0], lam0, mu0, x_ins, y_del)],
        t_span=(t_eps, t_max), y0=[r0], t_eval=t_eval,
        method='RK23', rtol=1e-4, atol=1e-7, max_step=5.0,
    ), r0


def run_frozen(lam0, mu0, x_ins, y_del, t_eps, t_max, t_eval):
    lam_T0, mu_T0, r0 = boundary_condition(lam0, mu0, x_ins, y_del)
    return solve_ivp(
        fun=lambda t, r: [dr_dt_frozen(t, r[0], lam_T0, mu_T0, lam0, mu0, x_ins, y_del)],
        t_span=(t_eps, t_max), y0=[r0], t_eval=t_eval,
        method='RK23', rtol=1e-4, atol=1e-7, max_step=5.0,
    ), r0, (lam_T0, mu_T0)


cases = [
    ("(0.4, 0.55)", 1.0*0.4*0.45/(0.55*0.6), 1.0, 0.4, 0.55),
    ("(0.2, 0.4)",  1.0*0.2*0.6/(0.4*0.8),   1.0, 0.2, 0.4),
    ("(0.3, 0.6)",  1.0*0.3*0.4/(0.6*0.7),   1.0, 0.3, 0.6),
    ("Pfam(0.65,0.71)", 0.0145, 0.0148, 0.65, 0.7098),
]

t_eps = 5e-3
t_max = 100.0
t_eval = np.geomspace(t_eps, t_max, 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (label, lam0, mu0, x, y) in zip(axes.flat, cases):
    print(f"\n{'='*72}")
    print(f"{label}: lam0={lam0:.4f}, mu0={mu0:.4f}")
    bc_l, bc_m, bc_r = boundary_condition(lam0, mu0, x, y)
    print(f"  Boundary: lam={bc_l:.4f}, mu={bc_m:.4f}, r={bc_r:.4f}")

    tic = time.time()
    full, _ = run_full(lam0, mu0, x, y, t_eps=t_eps, t_max=t_max, t_eval=t_eval)
    print(f"  Full 3-D: {time.time()-tic:.1f}s  status={full.status}  reached t={full.t[-1]:.2f}")

    tic = time.time()
    slv, _ = run_slaved(lam0, mu0, x, y, t_eps, t_max, t_eval)
    print(f"  Slaved:   {time.time()-tic:.1f}s  status={slv.status}  reached t={slv.t[-1]:.2f}")

    tic = time.time()
    frz, _, _ = run_frozen(lam0, mu0, x, y, t_eps, t_max, t_eval)
    print(f"  Frozen:   {time.time()-tic:.1f}s  status={frz.status}  reached t={frz.t[-1]:.2f}")

    # Plot r(t)
    if full.status == 0:
        ax.semilogx(full.t, full.y[2], 'b-', lw=2, label='full 3-D')
    if slv.status == 0:
        ax.semilogx(slv.t, slv.y[0], 'r--', lw=2, label=r'slaved $\lambda=\lambda_0/(1{-}r)$')
    if frz.status == 0:
        ax.semilogx(frz.t, frz.y[0], 'g:', lw=2, label=r'frozen $\lambda=\lambda^*(0)$')
    ax.set_xlabel('t'); ax.set_ylabel('r(t)')
    ax.set_title(f"{label}: lam0={lam0:.3f}, mu0={mu0:.3f}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0, max(bc_r * 1.05, 0.1))
    # Print final r and any local min/max
    if full.status == 0:
        r = full.y[2]
        i_min = np.argmin(r)
        print(f"  Full r min at t={full.t[i_min]:.2f}: r={r[i_min]:.4f}")
        print(f"  Full r end at t={full.t[-1]:.2f}: r={r[-1]:.4f}")
    if slv.status == 0:
        r = slv.y[0]
        i_min = np.argmin(r)
        print(f"  Slaved r min at t={slv.t[i_min]:.2f}: r={r[i_min]:.4f}")
        print(f"  Slaved r end at t={slv.t[-1]:.2f}: r={r[-1]:.4f}")
    if frz.status == 0:
        r = frz.y[0]
        i_min = np.argmin(r)
        print(f"  Frozen r min at t={frz.t[i_min]:.2f}: r={r[i_min]:.4f}")
        print(f"  Frozen r end at t={frz.t[-1]:.2f}: r={r[-1]:.4f}")

plt.suptitle("Long-time r(t): does the matched flow settle?  (t up to 100)", fontsize=12)
plt.tight_layout()
plt.savefig('/tmp/rt_longtime.png', dpi=110, bbox_inches='tight')
print("\nSaved /tmp/rt_longtime.png")
