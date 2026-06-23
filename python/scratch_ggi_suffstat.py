"""!!! WRONG ROUTE -- retained to DOCUMENT the failure mode. !!!

This script computes the suff-stat increments (b_G,..,e_G) as the bare
count-Jacobian of the BDI map in the Ndot direction AT FIXED (theta, T=t).
That is INCORRECT: the true increments are TOTAL time-derivatives of the
read-off and additionally carry (i) the immortal-link clock term dT/dt=1
(dominant for B,D,S) and (ii) the parameter-drift term d(stat)/dtheta * thetadot.
The consistency gate below FAILS (maxdiff up to ~3.5) precisely because of the
missing clock + drift terms.

The CORRECT validations live in:
  * scratch_ggi_true_incr.py  -- true total-derivative increments, param ODE OK
  * scratch_ggi_flow.py       -- natural-gradient / kl_fit flow (the engine)
  * scratch_ggi_del.py        -- exact deletion resolvent
See tkf/composition-renormalization.tex sec:comp-ggi-flow ("From the count
increment to the flow").  ins_increment() below IS correct and is imported by
the other scripts; only the count-Jacobian increment recipe here is wrong.

----
Assemble the suff-stat ODE system  d(B,D,S,F,E)/dt = (b_G,d_G,s_G,f_G,e_G)
and the equivalent (lambda,mu,r) ODE, and validate against the kl_fit flow.

  surrogate counts  N0 = expected_counts5(kappa,alpha,r)       [exact, per pair]
  GGI increment     Ndot = ins_increment + del_increment       [exact, validated]
  BDI map           tkf92_stats_from_counts(N, lam,mu,t,r,T)    [project code]

The suff-stat increments are the count-Jacobian of the BDI map in the Ndot
direction at the CURRENT params (fixed theta, T=t); the immortal-link clock
T=t advances separately (Tdot=1), entering the lambda ODE as the '+1'.
"""
import numpy as np
import scratch_ggi_flow as G
import scratch_ggi_del as DEL
from tkfmixdom.jax.core.bdi import tkf92_stats_from_counts
S, M, I, D, E = 0, 1, 2, 3, 4


def ins_increment(kap, alpha, r, lam0, x):
    """Closed-form insertion increment to Nbar (validated)."""
    Ns = G.expected_counts5(kap, alpha, r)
    cx = x / (1 - x)
    dN = np.zeros((5, 5))
    for p in (S, M, I):
        for q in (M, I, D, E):
            w = Ns[p, q]
            if w == 0:
                continue
            dN[p, q] -= lam0 * w
            dN[p, I] += lam0 * w
            dN[I, I] += lam0 * w * cx
            dN[I, q] += lam0 * w
    return dN


def bdi_stats(N, lam, mu, t, r, T):
    d = tkf92_stats_from_counts(N, lam, mu, t, r, T=T)
    return np.array([d['E_B'], d['E_D'], d['E_S'], d['ext_count'], d['notext_count']])


def suffstat_and_increments(lam, mu, r, t, lam0, mu0, x, y, eps=1e-5):
    kap, alpha = lam / mu, np.exp(-mu * t)
    N0 = G.expected_counts5(kap, alpha, r)
    Ndot = ins_increment(kap, alpha, r, lam0, x) + DEL.del_increment_exact(kap, alpha, r, mu0, y)
    stats0 = bdi_stats(N0, lam, mu, t, r, T=t)             # B,D,S,F,E
    # count-Jacobian of the BDI map in the Ndot direction, fixed theta, fixed T=t
    statsp = bdi_stats(N0 + eps * Ndot, lam, mu, t, r, T=t)
    incr = (statsp - stats0) / eps                          # b_G,d_G,s_G,f_G,e_G
    return stats0, incr, Ndot


def param_ode_from_suffstats(lam, mu, r, t, stats, incr):
    B, Dd, Sg, F, Enot = stats
    bG, dG, sG, fG, eG = incr
    dlam = (bG - lam * (sG + 1.0)) / (Sg + t)
    dmu = (dG - mu * sG) / Sg
    dr = ((1 - r) * fG - r * eG) / (F + Enot)
    return np.array([dlam, dmu, dr])


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    # reversible GGI
    x, y = 0.4, 0.55
    mu0 = 1.0
    lam0 = mu0 * x * (1 - y) / (y * (1 - x))
    print(f"GGI lam0={lam0:.4f} mu0={mu0} x={x} y={y}")

    print("\n=== consistency: suff-stat-derived param ODE vs kl_fit ground truth ===")
    print(f"{'(lam,mu,r,t)':>26}  {'dlam,dmu,dr [suffstat]':>30}  {'[kl_fit GT]':>26}  maxdiff")
    for (lam, mu, r, t) in [(0.7, 1.07, 0.25, 0.5), (0.73, 1.09, 0.22, 1.0),
                            (0.71, 1.03, 0.19, 2.0), (1.0, 1.5, 0.35, 0.3)]:
        stats, incr, Ndot = suffstat_and_increments(lam, mu, r, t, lam0, mu0, x, y)
        ode = param_ode_from_suffstats(lam, mu, r, t, stats, incr)
        # ground truth: kl_fit flow (route A Ndot already exact; reuse Ndot here)
        kap, alpha = lam / mu, np.exp(-mu * t)
        N0 = G.expected_counts5(kap, alpha, r)
        du = 2e-3
        kar0 = G.kl_fit(N0, x0=(kap, alpha, r))
        karp = G.kl_fit(N0 + du * Ndot, x0=tuple(kar0))
        lmr0 = np.array(G.kar_to_lmr(*kar0, t))
        lmrp = np.array(G.kar_to_lmr(*karp, t + du))
        gt = (lmrp - lmr0) / du
        md = np.max(np.abs(ode - gt))
        print(f"{str((lam,mu,round(r,3),t)):>26}  {str(np.round(ode,4)):>30}  {str(np.round(gt,4)):>26}  {md:.4f}")
        print(f"      stats B,D,S,F,E = {np.round(stats,4)}")
        print(f"      incr  b,d,s,f,e = {np.round(incr,4)}")
