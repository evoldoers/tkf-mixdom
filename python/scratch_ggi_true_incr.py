"""True total-derivative suff-stat increments along the GGI->TKF92 flow.

At (t,u) the composite is fit by kl_fit -> theta*(t,u); its suff stats are read by
the BDI map at (theta*, T=t+u).  We finite-difference in u to get the TRUE
dB/dt,...,dE/dt, then check the param ODE dlam/dt=[dB/dt-lam(dS/dt+1)]/(S+t) etc.
matches the kl_fit flow, and look for clean closed forms.
"""
import numpy as np
import scratch_ggi_flow as G
import scratch_ggi_del as DEL
import scratch_ggi_suffstat as SS
from tkfmixdom.jax.core.bdi import tkf92_stats_from_counts




def composite_stats(lam, mu, r, t, lam0, mu0, x, y, u):
    """suff stats of the composite at offset u: fit theta*, read BDI map at T=t+u."""
    kap, alpha = lam / mu, np.exp(-mu * t)
    N0 = G.expected_counts5(kap, alpha, r)
    Ndot = SS.ins_increment(kap, alpha, r, lam0, x) + DEL.del_increment_exact(kap, alpha, r, mu0, y)
    Nu = N0 + u * Ndot
    kar = G.kl_fit(Nu, x0=(kap, alpha, r))
    lamu, muu, ru = G.kar_to_lmr(*kar, t + u)
    d = tkf92_stats_from_counts(Nu, lamu, muu, t + u, ru, T=t + u)
    return (np.array([d['E_B'], d['E_D'], d['E_S'], d['ext_count'], d['notext_count']]),
            np.array([lamu, muu, ru]))


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    x, y = 0.4, 0.55; mu0 = 1.0
    lam0 = mu0 * x * (1 - y) / (y * (1 - x))
    du = 1e-3
    print(f"GGI lam0={lam0:.4f} mu0={mu0} x={x} y={y}\n")
    print("TRUE total-derivative increments and ODE-form check:")
    for (lam, mu, r, t) in [(0.7, 1.07, 0.25, 0.5), (0.73, 1.09, 0.22, 1.0), (0.71, 1.03, 0.19, 2.0)]:
        s0, th0 = composite_stats(lam, mu, r, t, lam0, mu0, x, y, 0.0)
        sp, thp = composite_stats(lam, mu, r, t, lam0, mu0, x, y, du)
        incr = (sp - s0) / du            # dB,dD,dS,dF,dE /dt (TRUE)
        dth = (thp - th0) / du           # dlam,dmu,dr /dt  (kl_fit flow)
        B, Dd, Sg, F, Enot = s0
        bG, dG, sG, fG, eG = incr
        ode = np.array([(bG - lam*(sG+1))/(Sg+t), (dG - mu*sG)/Sg, ((1-r)*fG - r*eG)/(F+Enot)])
        kap = lam/mu
        print(f"\n (lam,mu,r,t)=({lam},{mu},{r},{t})  kappa={kap:.4f}")
        print(f"   stats B,D,S,F,E = {np.round(s0,4)}")
        print(f"   TRUE incr dB,dD,dS,dF,dE = {np.round(incr,4)}")
        print(f"   clean-form guesses: dS=kappa/(1-kappa)={kap/(1-kap):.4f}?  "
              f"dB=dD? {abs(bG-dG)<0.05}")
        print(f"   param ODE [from incr] = {np.round(ode,4)}")
        print(f"   param flow [kl_fit GT]= {np.round(dth,4)}   match={np.max(np.abs(ode-dth)):.2e}")
