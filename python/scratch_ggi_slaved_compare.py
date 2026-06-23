"""Compare three approximations of the matched flow:
  (1) FULL 3-parameter (lam, mu, r) flow.
  (2) SLAVED: lam(t) = lam_0/(1-r(t)), mu(t) = mu_0/(1-r(t)), only r evolves.
  (3) FROZEN: lam(t) = lam*(0), mu(t) = mu*(0) (boundary values), only r evolves.
"""
import sys, time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/yam/tkf-mixdom/python')

from scratch_ggi_triad_eliminated import (
    triad_counts_eliminated, dtheta_dt, boundary_condition, run_flow as run_full,
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
        method='RK23', rtol=3e-4, atol=1e-6, max_step=0.5,
    ), r0


def run_frozen(lam0, mu0, x_ins, y_del, t_eps, t_max, t_eval):
    lam_T0, mu_T0, r0 = boundary_condition(lam0, mu0, x_ins, y_del)
    return solve_ivp(
        fun=lambda t, r: [dr_dt_frozen(t, r[0], lam_T0, mu_T0, lam0, mu0, x_ins, y_del)],
        t_span=(t_eps, t_max), y0=[r0], t_eval=t_eval,
        method='RK23', rtol=3e-4, atol=1e-6, max_step=0.5,
    ), r0, (lam_T0, mu_T0)


def compare(lam0, mu0, x_ins, y_del, t_eps, t_max, label):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    bc_l, bc_m, bc_r = boundary_condition(lam0, mu0, x_ins, y_del)
    print(f"  GGI: lam0={lam0:.4f}, mu0={mu0:.4f}, x={x_ins}, y={y_del}")
    print(f"  Boundary: lam={bc_l:.4f}, mu={bc_m:.4f}, r={bc_r:.4f}")
    t_eval = np.geomspace(t_eps, t_max, 40)
    tic = time.time()
    full, _ = run_full(lam0, mu0, x_ins, y_del, t_eps=t_eps, t_max=t_max, t_eval=t_eval)
    print(f"  Full 3-D:  {time.time()-tic:.1f}s")
    tic = time.time()
    slv, _ = run_slaved(lam0, mu0, x_ins, y_del, t_eps, t_max, t_eval)
    print(f"  Slaved:    {time.time()-tic:.1f}s")
    tic = time.time()
    frz, _, _ = run_frozen(lam0, mu0, x_ins, y_del, t_eps, t_max, t_eval)
    print(f"  Frozen:    {time.time()-tic:.1f}s")

    print(f"\n  {'t':>7} | {'full r':>8} {'slv r':>8} {'frz r':>8} | "
          f"{'slv-err':>8} {'frz-err':>8}")
    for tt in [0.05, 0.1, 0.3, 1.0, 2.0, 5.0]:
        if tt > t_max: continue
        if tt > full.t[-1] or tt > slv.t[-1] or tt > frz.t[-1]: continue
        i_f = np.argmin(np.abs(full.t - tt))
        i_s = np.argmin(np.abs(slv.t - tt))
        i_z = np.argmin(np.abs(frz.t - tt))
        fr, sr, zr = full.y[2, i_f], slv.y[0, i_s], frz.y[0, i_z]
        print(f"  {full.t[i_f]:>7.3f} | {fr:>8.4f} {sr:>8.4f} {zr:>8.4f} | "
              f"{(sr-fr)/fr*100:>7.1f}% {(zr-fr)/fr*100:>7.1f}%")
    return full, slv, frz, (lam0, mu0, x_ins, y_del, bc_l, bc_m, bc_r)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    case1 = compare(
        lam0=1.0 * 0.4 * (1 - 0.55) / (0.55 * (1 - 0.4)),
        mu0=1.0, x_ins=0.4, y_del=0.55, t_eps=5e-3, t_max=5.0,
        label="Case 1: GGI (x=0.4, y=0.55), strong indels"
    )

    LAM_PFAM, MU_PFAM, R_PFAM = 0.0458, 0.0468, 0.683
    LAM0 = LAM_PFAM * (1 - R_PFAM); MU0 = MU_PFAM * (1 - R_PFAM)
    case2 = compare(
        lam0=LAM0, mu0=MU0, x_ins=0.65, y_del=0.7098,
        t_eps=0.01, t_max=10.0,
        label="Case 2: Pfam scale (x=0.65, y=0.71)"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (case, name) in enumerate(zip([case1, case2], ["(0.4, 0.55)", "Pfam scale"])):
        full, slv, frz, (lam0, mu0, x_ins, y_del, bc_l, bc_m, bc_r) = case
        ax = axes[row, 0]
        ax.semilogx(full.t, full.y[2], 'b-', lw=2, label='r (full 3-D)')
        ax.semilogx(slv.t, slv.y[0], 'r--', lw=2, label=r'r (slaved: $\lambda=\lambda_0/(1-r)$)')
        ax.semilogx(frz.t, frz.y[0], 'g:', lw=2, label='r (frozen at boundary)')
        ax.set_xlabel('t'); ax.set_ylabel('r')
        ax.set_title(f'{name}: r(t) trajectories'); ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)
        ax = axes[row, 1]
        i_max = min(len(full.t), len(slv.t), len(frz.t))
        slv_at_full_t = np.interp(full.t, slv.t, slv.y[0])
        frz_at_full_t = np.interp(full.t, frz.t, frz.y[0])
        ax.semilogx(full.t, (slv_at_full_t - full.y[2])/full.y[2]*100, 'r-', lw=2, label='slaved')
        ax.semilogx(full.t, (frz_at_full_t - full.y[2])/full.y[2]*100, 'g-', lw=2, label='frozen')
        ax.axhline(0, color='gray', ls='-', lw=0.5)
        ax.set_xlabel('t'); ax.set_ylabel('relative r error (%)')
        ax.set_title(f'{name}: relative error in r(t)'); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig('/tmp/three_way_compare.png', dpi=110, bbox_inches='tight')
    print(f"\nSaved /tmp/three_way_compare.png")
