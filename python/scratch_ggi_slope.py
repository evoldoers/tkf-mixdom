"""Measure d(log r)/dt in the early phylogenetic regime for several GGI configs."""
import sys, numpy as np
sys.path.insert(0, '/Users/yam/tkf-mixdom/python')
from scratch_ggi_triad_eliminated import run_flow, boundary_condition

cases = [
    ("Pfam (0.65, 0.71)", 0.0145, 0.0148, 0.65, 0.7098, 1e-2, 50.0),
    ("Pfam (0.5, 0.77)",  0.0145, 0.0148, 0.50, 0.7666, 1e-2, 50.0),
    ("(0.4, 0.55)",       1.0*0.4*0.45/(0.55*0.6), 1.0, 0.4, 0.55, 5e-3, 0.5),
    ("(0.2, 0.4)",        1.0*0.2*0.6/(0.4*0.8),   1.0, 0.2, 0.4,  5e-3, 0.5),
    ("(0.3, 0.6)",        1.0*0.3*0.4/(0.6*0.7),   1.0, 0.3, 0.6,  5e-3, 0.5),
]

print(f"{'case':>22} | {'lam0':>7} {'mu0':>7} | {'r*(0)':>7} {'lam*(0)':>9} {'mu*(0)':>9} | {'slope_early':>11} {'(slope)/(mu0-lam0)':>20} {'(slope)/(mu*(0))':>17}")
for label, lam0, mu0, x, y, t_eps, t_max in cases:
    bc_l, bc_m, bc_r = boundary_condition(lam0, mu0, x, y)
    sol, _ = run_flow(lam0, mu0, x, y, t_eps=t_eps, t_max=t_max,
                       t_eval=np.geomspace(t_eps, t_max, 40))
    if sol.status != 0:
        continue
    # Take slope d(log r)/dt across a mid-range window
    t_lo, t_hi = t_eps * 5, t_max / 5
    mask = (sol.t >= t_lo) & (sol.t <= t_hi)
    if mask.sum() < 3:
        continue
    log_r = np.log(sol.y[2][mask])
    t = sol.t[mask]
    # Linear fit
    slope, intercept = np.polyfit(t, log_r, 1)
    slope_per_mu_minus_lam = slope / (mu0 - lam0)
    slope_per_mu_T0 = slope / bc_m
    print(f"{label:>22} | {lam0:>7.4f} {mu0:>7.4f} | {bc_r:>7.4f} {bc_l:>9.4f} {bc_m:>9.4f} | "
          f"{slope:>11.5f} {slope_per_mu_minus_lam:>20.3f} {slope_per_mu_T0:>17.4f}")
