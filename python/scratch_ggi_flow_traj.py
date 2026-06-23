"""
GGI->TKF92 flow: trajectory, limits, and the moment-match connection.

Uses the VALIDATED route-A increment (scratch_ggi_flow.ggi_Ndot_for_sample) to
build the flow RHS, integrates theta_flow(t), and compares against:
  * theta_direct(t) = argmin_theta D( GGI(t) || TKF92(theta,t) )  [true GGI, Gillespie]
  * the Holmes-2020 equilibrium moment-match (t-independent).
"""
import numpy as np
import scratch_ggi_flow as G

rng = G.rng

# ---------------------------------------------------------------- moment-match
def moment_match(lam0, mu0, x, y):
    """Holmes-2020 GGI->TKF92 equilibrium moment map (varanc-presence / GGI doc):
       r=x,  mu=mu0/(1-y),  equilibrium-length match fixes kappa, lambda=kappa*mu."""
    r = x
    mu = mu0 / (1 - y)
    L = x / (y - x)                      # GGI equilibrium length ell_GGI
    # ell_TKF = kappa/((1-kappa)(1-r)) = L  ->  solve kappa
    kap = L * (1 - r) / (1 + L * (1 - r))
    lam = kap * mu
    return lam, mu, r

# --------------------------------------------------- flow RHS d(lam,mu,r)/dt
def flow_rhs(lam, mu, r, t, lam0, mu0, x, y, nsamp=30000, du=2e-3):
    kap, alpha, _ = G.lmr_to_kar(lam, mu, r, t)
    Ndot = np.zeros((5, 5)); N0 = np.zeros((5, 5))
    for _ in range(nsamp):
        cols = G.sample_alignment(kap, alpha, r)
        nd, n0 = G.ggi_Ndot_for_sample(cols, lam0, mu0, x, y)
        Ndot += nd; N0 += n0
    Ndot /= nsamp; N0 /= nsamp
    kar0 = G.kl_fit(N0, x0=(kap, alpha, r))
    karp = G.kl_fit(N0 + du * Ndot, x0=tuple(kar0))
    lmr0 = np.array(G.kar_to_lmr(*kar0, t))
    lmrp = np.array(G.kar_to_lmr(*karp, t + du))
    return (lmrp - lmr0) / du, lmr0

# ----------------------------------------------- direct best-fit to true GGI(t)
def ggi_equilibrium_len(x, y):
    """Sample X length ~ GGI equilibrium geometric(x/y): P(n)=(x/y)^n (1-x/y)."""
    rho = x / y
    return rng.geometric(1 - rho) - 1     # support {0,1,2,...}

def direct_fit_ggi(lam0, mu0, x, y, t, nsamp=40000):
    Nsum = np.zeros((5, 5))
    for _ in range(nsamp):
        n = ggi_equilibrium_len(x, y)
        cols = [G.M] * n
        colsZ = G.gillespie_leg2(cols, lam0, mu0, x, y, t)
        Nsum += G.count_transitions(colsZ)
    return np.array(G.kar_to_lmr(*G.kl_fit(Nsum / nsamp), t))

# ============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    # a representative reversible GGI
    x, y = 0.4, 0.55
    mu0 = 1.0
    lam0 = mu0 * x * (1 - y) / (y * (1 - x))
    print(f"GGI: lam0={lam0:.4f} mu0={mu0} x={x} y={y}  (mean ins len 1/(1-x)={1/(1-x):.2f}, "
          f"mean del len 1/(1-y)={1/(1-y):.2f}, ell_GGI={x/(y-x):.3f})")
    lam_mm, mu_mm, r_mm = moment_match(lam0, mu0, x, y)
    print(f"Holmes-2020 moment-match TKF92: lam={lam_mm:.4f} mu={mu_mm:.4f} r={r_mm:.4f}\n")

    print("=== direct best-fit theta_direct(t)=argmin D(GGI(t)||TKF92) vs moment-match ===")
    print(f"{'t':>6} {'lam':>8} {'mu':>8} {'r':>8}    (kappa)")
    for t in [0.05, 0.2, 0.5, 1.0, 2.0, 4.0]:
        lam, mu, r = direct_fit_ggi(lam0, mu0, x, y, t, nsamp=60000)
        print(f"{t:6.2f} {lam:8.4f} {mu:8.4f} {r:8.4f}    ({lam/mu:.4f})")
    print(f"{'mm':>6} {lam_mm:8.4f} {mu_mm:8.4f} {r_mm:8.4f}    ({lam_mm/mu_mm:.4f})")

    print("\n=== flow RHS d(lam,mu,r)/dt evaluated AT the moment-match params ===")
    print("(if mm were the exact fixed point of the flow, these would be ~0)")
    for t in [0.1, 0.3, 0.6, 1.2, 2.5]:
        rhs, lmr0 = flow_rhs(lam_mm, mu_mm, r_mm, t, lam0, mu0, x, y, nsamp=30000)
        print(f"  t={t:5.2f}: d(lam,mu,r)/dt = {np.round(rhs,4)}")

    print("\n=== integrate flow from small t0, compare theta_flow(t) to theta_direct(t) ===")
    t0 = 0.1
    # initial condition: direct fit at t0 (small-t generator match)
    th = direct_fit_ggi(lam0, mu0, x, y, t0, nsamp=120000)
    print(f"  init theta_flow({t0}) = {np.round(th,4)}  (from direct fit)")
    t = t0; dt = 0.1
    targets = [0.2, 0.5, 1.0, 2.0]
    ti = 0
    while t < 2.0 + 1e-9:
        rhs, _ = flow_rhs(th[0], th[1], th[2], t, lam0, mu0, x, y, nsamp=20000)
        th = th + dt * rhs
        t += dt
        if ti < len(targets) and abs(t - targets[ti]) < 1e-6:
            thd = direct_fit_ggi(lam0, mu0, x, y, t, nsamp=80000)
            print(f"  t={t:4.2f}: flow={np.round(th,4)}  direct={np.round(thd,4)}  "
                  f"|diff|={np.max(np.abs(th-thd)):.4f}")
            ti += 1
