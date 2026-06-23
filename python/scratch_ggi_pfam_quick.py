"""Quick Pfam-scale tameness — kl_fit-based dtheta_dt, coarse tolerance."""
import sys, time
import numpy as np
sys.path.insert(0, '/Users/yam/tkf-mixdom/python')

from scipy.integrate import solve_ivp
import scratch_ggi_flow as G
import scratch_ggi_suffstat as SS
import scratch_ggi_del as DEL


def boundary_condition(lam0, mu0, x_ins, y_del):
    num = lam0 * x_ins * (1 - y_del) + mu0 * y_del * (1 - x_ins)
    den = lam0 * (1 - y_del) + mu0 * (1 - x_ins)
    r_star = num / den
    return lam0 / (1 - r_star), mu0 / (1 - r_star), r_star


def dtheta_dt(t, theta, lam0, mu0, x_ins, y_del, du=5e-4):
    lam, mu, r = theta
    if not (0 < lam < mu and 0 < r < 0.999):
        return np.zeros(3)
    try:
        kap, alpha = lam/mu, np.exp(-mu*t)
        if not (1e-6 < kap < 1-1e-6) or not (1e-6 < alpha < 1-1e-6):
            return np.zeros(3)
        N0 = G.expected_counts5(kap, alpha, r)
        Ndot = (SS.ins_increment(kap, alpha, r, lam0, x_ins)
                + DEL.del_increment_exact(kap, alpha, r, mu0, y_del))
        Nu = N0 + du*Ndot
        kar = G.kl_fit(Nu, x0=(kap, alpha, r))
        lamu, muu, ru = G.kar_to_lmr(*kar, t+du)
        return np.array([(lamu-lam)/du, (muu-mu)/du, (ru-r)/du])
    except Exception:
        return np.zeros(3)


def run_pfam_traj(lam0, mu0, x_ins, y_del, t_eps=0.01, t_max=10.0):
    lam_T0, mu_T0, r0 = boundary_condition(lam0, mu0, x_ins, y_del)
    t_eval = np.geomspace(t_eps, t_max, 25)
    sol = solve_ivp(
        fun=lambda t, theta: dtheta_dt(t, theta, lam0, mu0, x_ins, y_del),
        t_span=(t_eps, t_max),
        y0=np.array([lam_T0, mu_T0, r0]),
        t_eval=t_eval,
        method='RK23', rtol=3e-3, atol=1e-5,
        max_step=1.0,
    )
    return sol, (lam_T0, mu_T0, r0)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    LAM_PFAM, MU_PFAM, R_PFAM = 0.0458, 0.0468, 0.683
    LAM0 = LAM_PFAM * (1 - R_PFAM)
    MU0 = MU_PFAM * (1 - R_PFAM)
    print(f"Pfam anchor: lam_T={LAM_PFAM} mu_T={MU_PFAM} r={R_PFAM}")
    print(f"Inferred GGI: lam_0={LAM0:.5f} mu_0={MU0:.5f}")

    def r_star(lam0, mu0, x_ins, y_del):
        num = lam0 * x_ins * (1 - y_del) + mu0 * y_del * (1 - x_ins)
        den = lam0 * (1 - y_del) + mu0 * (1 - x_ins)
        return num / den

    # Probe a few (x_ins, y_del) consistent with r* = R_PFAM
    configs = []
    for x_ins in [0.5, 0.6, 0.7, 0.75]:
        lo, hi = 0.01, 0.99
        for _ in range(60):
            mid = (lo + hi) / 2
            if r_star(LAM0, MU0, x_ins, mid) < R_PFAM: lo = mid
            else: hi = mid
        y_del = (lo + hi) / 2
        if 0.05 < y_del < 0.95:
            configs.append((x_ins, y_del))
    print(f"\nProbing {len(configs)} GGI shapes consistent with r*=R_Pfam:")

    results = []
    print(f"\n{'x_ins':>6} {'y_del':>6} | {'lam[min,max]':>17} | {'mu[min,max]':>17} | {'r[min,max]':>16} | {'time':>5}")
    for x_ins, y_del in configs:
        tic = time.time()
        sol, bc = run_pfam_traj(LAM0, MU0, x_ins, y_del, t_eps=0.01, t_max=10.0)
        elapsed = time.time() - tic
        if sol.status != 0 or len(sol.t) < 3:
            print(f"  {x_ins:>6.3f} {y_del:>6.3f}: failed (status={sol.status})")
            continue
        mask = sol.t >= 0.05
        if mask.sum() < 2:
            print(f"  {x_ins:>6.3f} {y_del:>6.3f}: no data past t=0.05")
            continue
        results.append(dict(x_ins=x_ins, y_del=y_del, sol=sol, bc=bc,
                            lam_range=(sol.y[0,mask].min(), sol.y[0,mask].max()),
                            mu_range=(sol.y[1,mask].min(), sol.y[1,mask].max()),
                            r_range=(sol.y[2,mask].min(), sol.y[2,mask].max())))
        r = results[-1]
        print(f"  {x_ins:>6.3f} {y_del:>6.3f} | "
              f"[{r['lam_range'][0]:.5f},{r['lam_range'][1]:.5f}] | "
              f"[{r['mu_range'][0]:.5f},{r['mu_range'][1]:.5f}] | "
              f"[{r['r_range'][0]:.3f},{r['r_range'][1]:.3f}] | {elapsed:.0f}s")

    if results:
        all_lam = (min(r['lam_range'][0] for r in results), max(r['lam_range'][1] for r in results))
        all_mu = (min(r['mu_range'][0] for r in results), max(r['mu_range'][1] for r in results))
        all_r = (min(r['r_range'][0] for r in results), max(r['r_range'][1] for r in results))
        print()
        print(f"Aggregate envelope, t in [0.05, 10]:")
        print(f"  lam_T:  [{all_lam[0]:.5f}, {all_lam[1]:.5f}]   spread {(all_lam[1]-all_lam[0])/np.mean(all_lam)*100:.1f}%")
        print(f"  mu_T :  [{all_mu[0]:.5f}, {all_mu[1]:.5f}]   spread {(all_mu[1]-all_mu[0])/np.mean(all_mu)*100:.1f}%")
        print(f"  r    :  [{all_r[0]:.3f}, {all_r[1]:.3f}]    spread {(all_r[1]-all_r[0])/np.mean(all_r)*100:.1f}%")
        np.savez('/tmp/pfam_tameness.npz',
                 pfam_anchor=np.array([LAM_PFAM, MU_PFAM, R_PFAM]),
                 lam_range=np.array(all_lam),
                 mu_range=np.array(all_mu),
                 r_range=np.array(all_r),
                 configs=np.array(configs))
        print(f"\nSaved /tmp/pfam_tameness.npz")
