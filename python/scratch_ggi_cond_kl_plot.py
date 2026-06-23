"""Plot the conditional-KL Triad ODE trajectory vs empirical Gillespie fits."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load('/tmp/cond_kl_ode_traj.npz', allow_pickle=True)
t = d['t']; lam = d['lam']; mu = d['mu']; r = d['r']
x_ins, y_del = float(d['x_ins']), float(d['y_del'])
lam0, mu0 = float(d['lam0']), float(d['mu0'])
bc = d['boundary']

# Empirical Gillespie data the user collected (from NOTES_ggi_dr_dt.md table)
emp_t  = np.array([0.1,   0.5,   1.0,   2.0,   4.0])
emp_r  = np.array([0.284, 0.256, 0.221, 0.191, 0.167])
emp_lam = None  # not in the writeup table
emp_mu  = None
# From the user's earlier task output `byvywhsiv.output`:
#   (0.4,0.55)  kappa  mu-lam   k_fit  r_inf k/(mu-lam)
#   (0.4,0.55)  0.679  0.3409  0.6022  0.144      1.766
# So mu-lam = 0.34, kappa(fit) = 0.60, r_inf ~ 0.14 (asymptotic empirical r)
# i.e. at large t the empirical fit suggests mu - lam ~ 0.34 and r ~ 0.14

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.semilogx(t, r, 'b-', lw=2, label='ODE (cond-KL Triad)')
ax.semilogx(emp_t, emp_r, 'ro', ms=8, label='Empirical Gillespie fits')
ax.axhline(0.144, color='gray', ls=':', lw=1, label='Empirical r_inf (~0.14)')
ax.set_xlabel('t (branch length)')
ax.set_ylabel('r')
ax.set_title('Fragment-extension parameter r(t)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.semilogx(t, lam, 'b-', lw=2, label='lambda_T (ODE)')
ax.semilogx(t, mu,  'r-', lw=2, label='mu_T (ODE)')
ax.semilogx(t, lam/mu, 'g--', lw=1.5, label='kappa = lam/mu')
ax.axhline(0.6022, color='g', ls=':', lw=1, alpha=0.7, label='empirical kappa(~0.60)')
ax.axhline(0.34,   color='purple', ls=':', lw=1, alpha=0.7, label='empirical mu-lam(~0.34)')
ax.set_xlabel('t (branch length)')
ax.set_ylabel('rate')
ax.set_title('TKF92 rate parameters')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
kap = lam / mu
mean_L = kap / ((1 - kap) * (1 - r))
ax.semilogx(t, mean_L, 'b-', lw=2, label='TKF92 mean L (ODE)')
ggi_eq_L = x_ins / (y_del - x_ins) if y_del > x_ins else float('inf')
# code: x = ins, y = del. equilibrium length from balance:
#   lam (L+1)/(1-x) = mu L/(1-y)
#   L* = lam(1-y) / [mu(1-x) - lam(1-y)]
L_star = lam0 * (1 - y_del) / (mu0 * (1 - x_ins) - lam0 * (1 - y_del))
ax.axhline(L_star, color='r', ls='--', lw=1.5, label=f'GGI eq. L (~{L_star:.2f})')
ax.set_xlabel('t (branch length)')
ax.set_ylabel('mean length')
ax.set_title('TKF92 mean sequence length')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
# log scale param drift relative to t->0+ boundary
lam_drift = (lam - bc[0]) / bc[0]
mu_drift = (mu - bc[1]) / bc[1]
r_drift = (r - bc[2]) / bc[2]
ax.semilogx(t, lam_drift, 'b-', lw=2, label='Delta lam / lam(0)')
ax.semilogx(t, mu_drift,  'r-', lw=2, label='Delta mu / mu(0)')
ax.semilogx(t, r_drift,   'g-', lw=2, label='Delta r / r(0)')
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.set_xlabel('t (branch length)')
ax.set_ylabel('relative drift from boundary')
ax.set_title('Parameter drift relative to t->0+ boundary')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Conditional-KL Triad flow:  GGI(x_ins={x_ins}, y_del={y_del}) -> TKF92(t)\n'
             f'GGI rev: lam0={lam0:.3f}, mu0={mu0:.3f}.  Boundary: '
             f'lam(0)={bc[0]:.3f}, mu(0)={bc[1]:.3f}, r(0)={bc[2]:.3f}',
             fontsize=10)
plt.tight_layout()
plt.savefig('/tmp/cond_kl_ode_traj.png', dpi=110, bbox_inches='tight')
print('Saved /tmp/cond_kl_ode_traj.png')

# Diagnostic table
print('\nTrajectory summary:')
print(f"  {'t':>7}  {'lam_T':>8}  {'mu_T':>8}  {'mu-lam':>8}  {'r':>8}  {'kappa':>8}  {'mean_L':>8}")
for tt in [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0]:
    if tt > t[-1]:
        continue
    idx = np.argmin(np.abs(t - tt))
    print(f"  {t[idx]:>7.4f}  {lam[idx]:>8.4f}  {mu[idx]:>8.4f}  "
          f"{mu[idx]-lam[idx]:>8.4f}  {r[idx]:>8.4f}  {lam[idx]/mu[idx]:>8.4f}  "
          f"{kap[idx]/((1-kap[idx])*(1-r[idx])):>8.3f}")

# Check tameness of lam, mu after initial transient
mask = t > 0.3
print(f"\nPost-transient (t > 0.3) tameness check:")
print(f"  lam_T range: [{lam[mask].min():.3f}, {lam[mask].max():.3f}]  (relative spread {(lam[mask].max()-lam[mask].min())/lam[mask].mean()*100:.1f}%)")
print(f"  mu_T  range: [{mu[mask].min():.3f}, {mu[mask].max():.3f}]   (relative spread {(mu[mask].max()-mu[mask].min())/mu[mask].mean()*100:.1f}%)")
print(f"  r     range: [{r[mask].min():.3f}, {r[mask].max():.3f}]   (relative spread {(r[mask].max()-r[mask].min())/r[mask].mean()*100:.1f}%)")
