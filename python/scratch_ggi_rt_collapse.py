"""Test whether r(t)/r0 vs. a rescaled "indel time" collapses across regimes.

Hypothesis: the natural time scale of the matched flow is set by the
GGI per-residue event rate.  Candidates:
   (a) mu_0 * t                                 (per-residue deletion rate)
   (b) mu_0/(1-x) * t                           (per-residue lengthwise deletion rate)
   (c) [lam_0/(1-y) + mu_0/(1-x)] * t           (per-residue total indel rate)
   (d) (mu_0 - lam_0) * t                       (BDI net death rate)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/yam/tkf-mixdom/python')

from scratch_ggi_triad_eliminated import run_flow, boundary_condition

cases = [
    ("(0.4, 0.55)",      1.0*0.4*0.45/(0.55*0.6), 1.0, 0.4, 0.55, 5e-3, 5.0),
    ("(0.2, 0.4)",       1.0*0.2*0.6/(0.4*0.8),   1.0, 0.2, 0.4,  5e-3, 5.0),
    ("(0.3, 0.6)",       1.0*0.3*0.4/(0.6*0.7),   1.0, 0.3, 0.6,  5e-3, 5.0),
    ("(0.5, 0.7)",       1.0*0.5*0.3/(0.7*0.5),   1.0, 0.5, 0.7,  5e-3, 5.0),
    ("Pfam(0.65,0.71)",  0.0145, 0.0148, 0.65, 0.7098, 0.01, 10.0),
    ("Pfam(0.50,0.77)",  0.0145, 0.0148, 0.50, 0.7666, 0.01, 10.0),
]

results = []
for label, lam0, mu0, x, y, t_eps, t_max in cases:
    bc_l, bc_m, bc_r = boundary_condition(lam0, mu0, x, y)
    t_eval = np.geomspace(t_eps, t_max, 40)
    sol, _ = run_flow(lam0, mu0, x, y, t_eps=t_eps, t_max=t_max, t_eval=t_eval)
    if sol.status != 0:
        continue
    rate_total = lam0/(1-y) + mu0/(1-x)
    rate_del_persidue = mu0/(1-x)
    results.append(dict(label=label, lam0=lam0, mu0=mu0, x=x, y=y,
                         t=sol.t, r=sol.y[2], r0=bc_r,
                         rate_total=rate_total, rate_del=rate_del_persidue,
                         rate_mu=mu0, rate_diff=mu0-lam0))

# 4-panel: r/r0 vs rescaled time, using 4 candidate rates
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (key, rate_lbl) in zip(axes.flat, [
        ('rate_mu', r'$\mu_0\, t$'),
        ('rate_del', r'$\mu_0/(1{-}x)\cdot t$'),
        ('rate_total', r'$[\lambda_0/(1{-}y) + \mu_0/(1{-}x)]\cdot t$'),
        ('rate_diff', r'$(\mu_0{-}\lambda_0)\, t$'),
    ]):
    for res in results:
        ax.semilogx(res[key] * res['t'], res['r'] / res['r0'], '-', lw=1.5,
                    label=res['label'])
    ax.set_xlabel(rate_lbl); ax.set_ylabel(r'$r(t)/r_0$')
    ax.set_title(f"rescaled time = {rate_lbl}")
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
plt.suptitle("Curve collapse test: which dimensionless time unifies the matched flow r(t)?", fontsize=11)
plt.tight_layout()
plt.savefig('/tmp/rt_collapse.png', dpi=110, bbox_inches='tight')
print("Saved /tmp/rt_collapse.png")

# Print where the trajectories cross r/r0 = 0.5  (mid-decay) — the "turnover"
print("\nTime at which r(t) drops to r0/2  (potential 'turnover' marker):")
print(f"{'case':>20} | {'mu*t_half':>10} | {'mu_eff*t_half':>14} | {'(mu-lam)*t_half':>16} | {'(rates)*t_half':>15}")
for res in results:
    r_target = res['r0'] / 2
    if res['r'].min() > r_target:
        print(f"{res['label']:>20} |  r never reaches r0/2 (min r = {res['r'].min():.3f}, target = {r_target:.3f})")
        continue
    # Interpolate
    idx = np.where(res['r'] <= r_target)[0]
    if len(idx) == 0: continue
    i = idx[0]
    if i == 0:
        t_half = res['t'][0]
    else:
        # Linear interp in log t
        lt0, r0_pt = np.log(res['t'][i-1]), res['r'][i-1]
        lt1, r1_pt = np.log(res['t'][i]), res['r'][i]
        frac = (r0_pt - r_target) / (r0_pt - r1_pt)
        t_half = np.exp(lt0 + frac * (lt1 - lt0))
    print(f"{res['label']:>20} |  {res['rate_mu']*t_half:>10.3f} |  "
          f"{res['rate_del']*t_half:>14.3f} |  "
          f"{res['rate_diff']*t_half:>16.3f} |  "
          f"{res['rate_total']*t_half:>15.3f}")
