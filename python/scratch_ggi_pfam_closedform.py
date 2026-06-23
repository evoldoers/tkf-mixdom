"""Pfam (x=0.65, y=0.71) flow to t=10^3: full 3D vs the closed-form approximation
   r(t) ~ r_inf + (r0 - r_inf) exp(-kt) with
      r_inf = r0/(2-r0),  k = (lam0+mu0)(2-r0)/(1-r0)
   and the slaved relations lam(t) ~ lam0/(1-r(t)) and mu(t) ~ mu0/(1-r(t))
   using the closed-form r(t).
"""
import sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/yam/tkf-mixdom/python')

from scratch_ggi_triad_eliminated import boundary_condition, run_flow as run_full

# Pfam config
LAM_PFAM, MU_PFAM, R_PFAM = 0.0458, 0.0468, 0.683
LAM0 = LAM_PFAM * (1 - R_PFAM)
MU0 = MU_PFAM * (1 - R_PFAM)
x_ins, y_del = 0.65, 0.7098

bc_l, bc_m, bc_r = boundary_condition(LAM0, MU0, x_ins, y_del)
print(f"Pfam GGI: lam0={LAM0:.5f}, mu0={MU0:.5f}, x={x_ins}, y={y_del}")
print(f"Boundary: lam(0)={bc_l:.5f}, mu(0)={bc_m:.5f}, r(0)={bc_r:.4f}")

# Closed-form parameters
r0 = bc_r
r_inf = r0 / (2 - r0)
k_decay = (LAM0 + MU0) * (2 - r0) / (1 - r0)
print(f"Closed-form: r_inf = r0/(2-r0) = {r_inf:.4f},  k = (lam0+mu0)(2-r0)/(1-r0) = {k_decay:.4f}")

# Run full 3D
t_eps, t_max = 1e-2, 1e3
t_eval = np.geomspace(t_eps, t_max, 80)
tic = time.time()
full, _ = run_full(LAM0, MU0, x_ins, y_del, t_eps=t_eps, t_max=t_max, t_eval=t_eval)
print(f"Full 3-D: {time.time()-tic:.1f}s")

# Closed-form r(t)
t_fine = np.geomspace(t_eps, t_max, 200)
r_closed = r_inf + (r0 - r_inf) * np.exp(-k_decay * t_fine)
lam_closed = LAM0 / (1 - r_closed)
mu_closed = MU0 / (1 - r_closed)

# Stacked plot
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ax = axes[0]
ax.semilogx(full.t, full.y[0], 'b-', lw=2, label=r'$\lambda(t)$  full 3-D')
ax.semilogx(t_fine, lam_closed, 'r--', lw=2,
            label=r'$\lambda(t) = \lambda_0/(1-\tilde r(t))$ closed form')
ax.axhline(bc_l, color='gray', ls=':', lw=1, label=fr'$\lambda^*(0)={bc_l:.4f}$')
ax.set_ylabel(r'$\lambda(t)$')
ax.set_title(f"Pfam GGI ($\\lambda_0$={LAM0:.4f}, $\\mu_0$={MU0:.4f}, $x$={x_ins}, $y$={y_del}):"
             f"  closed-form approximation vs full 3-D flow",
             fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc='best', fontsize=10)

ax = axes[1]
ax.semilogx(full.t, full.y[1], 'b-', lw=2, label=r'$\mu(t)$  full 3-D')
ax.semilogx(t_fine, mu_closed, 'r--', lw=2,
            label=r'$\mu(t) = \mu_0/(1-\tilde r(t))$ closed form')
ax.axhline(bc_m, color='gray', ls=':', lw=1, label=fr'$\mu^*(0)={bc_m:.4f}$')
ax.set_ylabel(r'$\mu(t)$')
ax.grid(alpha=0.3)
ax.legend(loc='best', fontsize=10)

ax = axes[2]
ax.semilogx(full.t, full.y[2], 'b-', lw=2, label=r'$r(t)$  full 3-D')
ax.semilogx(t_fine, r_closed, 'r--', lw=2,
            label=fr'$\tilde r(t) = r_\infty + (r^*-r_\infty)e^{{-kt}}$'
                  fr',  $r_\infty={r_inf:.3f}$, $k={k_decay:.3f}$')
ax.axhline(bc_r, color='gray', ls=':', lw=1, label=fr'$r^*(0)={bc_r:.4f}$')
ax.axhline(r_inf, color='red', ls=':', lw=1, label=fr'$r_\infty={r_inf:.3f}$')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$r(t)$')
ax.grid(alpha=0.3)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/yam/tkf-mixdom/tkf/pfam_closedform.pdf', bbox_inches='tight')
plt.savefig('/tmp/pfam_closedform.png', dpi=110, bbox_inches='tight')
print(f"\nSaved /Users/yam/tkf-mixdom/tkf/pfam_closedform.pdf")
print(f"Saved /tmp/pfam_closedform.png")

# Tabulate at key points
print(f"\n{'t':>8} | {'full lam':>10} {'closed lam':>10} | {'full mu':>10} {'closed mu':>10} | {'full r':>9} {'closed r':>9}")
for tt in [0.01, 0.1, 1, 10, 50, 100, 500, 1000]:
    if tt > full.t[-1] or tt > t_fine[-1]: continue
    i_f = np.argmin(np.abs(full.t - tt))
    rc = r_inf + (r0 - r_inf) * np.exp(-k_decay * tt)
    lc = LAM0 / (1 - rc)
    mc = MU0 / (1 - rc)
    print(f"  {tt:>7.2f} | {full.y[0,i_f]:>10.5f} {lc:>10.5f} | {full.y[1,i_f]:>10.5f} {mc:>10.5f} | "
          f"{full.y[2,i_f]:>9.4f} {rc:>9.4f}")
