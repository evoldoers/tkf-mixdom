"""Simulate the conditional-KL Triad ODE flow at Pfam-fitted scales,
up to t=10, and report (lambda_T(t), mu_T(t)) ranges.

Pfam-fitted TKF92 params (from tkf92_fitted_params.json):
  lambda_T = 0.0458
  mu_T     = 0.0468
  r        = 0.683
  kappa    = 0.979

Inferring the GGI process this surrogates (via the t=0 boundary condition):
  lambda_0 = lambda_T (1-r) = 0.0145
  mu_0     = mu_T (1-r)     = 0.0148

The (x_ins, y_del) geometric params are not uniquely determined by the
boundary -- only the ratio constraint
  r* = [lam0 x_ins (1-y_del) + mu0 y_del (1-x_ins)]
     / [lam0 (1-y_del)       + mu0 (1-x_ins)]
fixes one combination.  We probe a small spread of (x_ins, y_del) values
consistent with r* in [0.65, 0.71] to see how the flow at Pfam scale
behaves out to t=10.
"""
import sys
import numpy as np
sys.path.insert(0, '/Users/yam/tkf-mixdom/python')
from scipy.integrate import solve_ivp

import scratch_ggi_flow as G
import scratch_ggi_suffstat as SS
import scratch_ggi_del as DEL
from scratch_ggi_cond_kl_ode import boundary_condition, dtheta_dt, run_flow

# ----------------------------------------------------------------------------
# Pfam-fitted TKF92 anchor
# ----------------------------------------------------------------------------
LAM_PFAM = 0.0458
MU_PFAM  = 0.0468
R_PFAM   = 0.683
LAM0_PFAM = LAM_PFAM * (1 - R_PFAM)
MU0_PFAM  = MU_PFAM  * (1 - R_PFAM)
print("=" * 72)
print("Pfam-scale flow at TKF92 anchor")
print("=" * 72)
print(f"  Pfam fit:  lam_T={LAM_PFAM} mu_T={MU_PFAM} r={R_PFAM}")
print(f"  Inferred:  lam_0={LAM0_PFAM:.5f} mu_0={MU0_PFAM:.5f}")
print()

# ----------------------------------------------------------------------------
# Spread of (x_ins, y_del): keep lam_0, mu_0 fixed, vary (x_ins, y_del) on the
# r*=0.68 curve.
# ----------------------------------------------------------------------------
def r_star(lam0, mu0, x_ins, y_del):
    num = lam0 * x_ins * (1 - y_del) + mu0 * y_del * (1 - x_ins)
    den = lam0 * (1 - y_del)         + mu0 * (1 - x_ins)
    return num / den

# Try a small range of x_ins, with y_del chosen so r* sits near R_PFAM
configs = []
for x_ins in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
    # Solve r_star(lam0, mu0, x_ins, y) = R_PFAM for y_del
    # Bisection in y
    lo, hi = 0.01, 0.99
    target = R_PFAM
    for _ in range(60):
        mid = (lo + hi) / 2
        v = r_star(LAM0_PFAM, MU0_PFAM, x_ins, mid)
        if v < target:
            lo = mid
        else:
            hi = mid
    y_del = (lo + hi) / 2
    rs = r_star(LAM0_PFAM, MU0_PFAM, x_ins, y_del)
    if 0.01 < y_del < 0.99 and abs(rs - R_PFAM) < 0.002:
        configs.append((x_ins, y_del))

print("  Probed (x_ins, y_del) configurations matching r*(0) = r_Pfam:")
print(f"  {'x_ins':>8} {'y_del':>8} {'r*':>8} {'(check)':>10}")
for x_ins, y_del in configs:
    rs = r_star(LAM0_PFAM, MU0_PFAM, x_ins, y_del)
    print(f"  {x_ins:>8.3f} {y_del:>8.3f} {rs:>8.4f} {rs - R_PFAM:>10.2e}")
print()

# ----------------------------------------------------------------------------
# Run the flow at each config up to t=10
# ----------------------------------------------------------------------------
print("  Flow trajectory ranges for t in [0.05, 10]:")
print(f"  {'x_ins':>8} {'y_del':>8} {'lam(0)':>8} {'mu(0)':>8} {'r(0)':>8} "
      f"{'lam[min,max]':>16} {'mu[min,max]':>16} {'r[min,max]':>16}")

results = []
for x_ins, y_del in configs:
    try:
        sol, bc = run_flow(LAM0_PFAM, MU0_PFAM, x_ins, y_del,
                            t_eps=5e-3, t_max=10.0,
                            t_eval=np.geomspace(5e-3, 10.0, 50))
        if sol.status != 0 or len(sol.t) < 10:
            print(f"  {x_ins:>8.3f} {y_del:>8.3f}   [solve_ivp failed: {sol.message}]")
            continue
        # Mask: take t >= 0.05 (post-transient is typically much later than this)
        mask = sol.t >= 0.05
        lam_min, lam_max = sol.y[0, mask].min(), sol.y[0, mask].max()
        mu_min,  mu_max  = sol.y[1, mask].min(), sol.y[1, mask].max()
        r_min,   r_max   = sol.y[2, mask].min(), sol.y[2, mask].max()
        results.append(dict(x_ins=x_ins, y_del=y_del, bc=bc, sol=sol,
                            lam_range=(lam_min, lam_max),
                            mu_range=(mu_min, mu_max),
                            r_range=(r_min, r_max)))
        print(f"  {x_ins:>8.3f} {y_del:>8.3f} {bc[0]:>8.4f} {bc[1]:>8.4f} {bc[2]:>8.4f} "
              f"[{lam_min:.4f},{lam_max:.4f}] "
              f"[{mu_min:.4f},{mu_max:.4f}] "
              f"[{r_min:.4f},{r_max:.4f}]")
    except Exception as ex:
        print(f"  {x_ins:>8.3f} {y_del:>8.3f}   [failed: {ex}]")

# ----------------------------------------------------------------------------
# Aggregate across configs
# ----------------------------------------------------------------------------
if results:
    all_lam_min = min(r['lam_range'][0] for r in results)
    all_lam_max = max(r['lam_range'][1] for r in results)
    all_mu_min  = min(r['mu_range'][0]  for r in results)
    all_mu_max  = max(r['mu_range'][1]  for r in results)
    all_r_min   = min(r['r_range'][0]   for r in results)
    all_r_max   = max(r['r_range'][1]   for r in results)
    print()
    print("  ----------------------------------------------------------------")
    print("  Aggregate envelope across Pfam-consistent GGIs, t in [0.05, 10]:")
    print(f"    lam_T:  [{all_lam_min:.4f}, {all_lam_max:.4f}]   "
          f"relative spread {(all_lam_max-all_lam_min)/((all_lam_max+all_lam_min)/2)*100:.1f}%")
    print(f"    mu_T :  [{all_mu_min:.4f}, {all_mu_max:.4f}]   "
          f"relative spread {(all_mu_max-all_mu_min)/((all_mu_max+all_mu_min)/2)*100:.1f}%")
    print(f"    r    :  [{all_r_min:.4f}, {all_r_max:.4f}]   "
          f"relative spread {(all_r_max-all_r_min)/((all_r_max+all_r_min)/2)*100:.1f}%")

    np.savez('/tmp/pfam_tameness.npz',
             configs=np.array(configs),
             pfam_anchor=np.array([LAM_PFAM, MU_PFAM, R_PFAM]),
             pfam_inferred=np.array([LAM0_PFAM, MU0_PFAM]),
             agg_lam_range=np.array([all_lam_min, all_lam_max]),
             agg_mu_range=np.array([all_mu_min, all_mu_max]),
             agg_r_range=np.array([all_r_min, all_r_max]),
             # one representative trajectory
             rep_t=results[len(results)//2]['sol'].t,
             rep_lam=results[len(results)//2]['sol'].y[0],
             rep_mu=results[len(results)//2]['sol'].y[1],
             rep_r=results[len(results)//2]['sol'].y[2])
    print("\n  Saved /tmp/pfam_tameness.npz")
