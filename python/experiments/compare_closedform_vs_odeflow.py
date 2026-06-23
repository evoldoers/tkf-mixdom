#!/usr/bin/env python3
"""Compare conditional LL under two flow predictions of (lam(t), mu(t), r(t))
given fixed GGI generative parameters:

  - CLOSED FORM (first-order-exact heuristic from composition-renormalization.tex):
        r_inf = r*(0)/(2 - r*(0))
        k = (lam0 + mu0)(2 - r*(0))/(1 - r*(0))
        r(t) = r_inf + (r*(0) - r_inf) exp(-k t)
        lam(t) = lam0/(1 - r(t)),  mu(t) = mu0/(1 - r(t))

  - NUMERICAL ODE  (scratch_ggi_triad_eliminated.run_flow): integrate the full
        3-d ODE for (lam(t), mu(t), r(t)) from the boundary at t=eps to t_max
        using scipy.solve_ivp (RK23).

For each tau bin in the simulation, evaluate the TKF92 pair-HMM at the closed-
form (lam(t), mu(t), r(t)) and at the numerically-integrated (lam(t), mu(t),
r(t)), and report the conditional LL under each.  Tells us whether the
first-order-exact heuristic is leaving meaningful LL on the table.
"""
import sys, os, time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulate_ggi_cherries import simulate_cherries, S, M, I, D, E
from fit_ggi_cherryml import (
    tkf92_trans_full, tkf92_singlet_log_marginal, ggi_to_tkf92_at_t,
)
# Canonical algebraic matched-flow ODE.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/python')
sys.path.insert(0, '/Users/yam/tkf-mixdom/python')
from scratch_ggi_cond_kl_quad import run_flow, boundary_condition


# GGI generative params (slightly asymmetric so the existing ODE integrator
# in scratch_ggi_triad_eliminated does not trip the kappa->1 NaN path).
# Reversibility is exact when lam0*y(1-y) = mu0*x(1-x).
LAM0, MU0 = 0.035, 0.045
X_DEL, Y_INS = 0.6, 0.4
L_MEAN = 50
N_CHERRIES = 5000
TAU = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
SEED = 42


def conditional_ll_for_theta_trajectory(lam_arr, mu_arr, r_arr, counts, tau_centers):
    """Given (lam(t), mu(t), r(t)) at each tau bin, compute the conditional LL
    summed over bins:  sum_t [n_ij log tau(theta(t), t)[i,j] - singlet(theta(t))]
    """
    counts_j = jnp.asarray(counts, jnp.float32)
    total = 0.0
    for ti, t in enumerate(tau_centers):
        lam_t = jnp.float32(lam_arr[ti])
        mu_t = jnp.float32(mu_arr[ti])
        r_t = jnp.float32(r_arr[ti])
        T = tkf92_trans_full(lam_t, mu_t, jnp.float32(t), r_t)
        log_T = jnp.log(jnp.maximum(T, 1e-30))
        n_ij = counts_j[ti]
        joint = float(jnp.sum(n_ij * log_T))
        t_anc = float(jnp.sum(n_ij[:, M]) + jnp.sum(n_ij[:, D]))
        n_cherries = float(jnp.sum(n_ij[:, E]))
        singlet = float(tkf92_singlet_log_marginal(lam_t, mu_t, r_t, n_cherries, t_anc))
        total += joint - singlet
    return total


def main():
    print(f"Sim params: lam0={LAM0}, mu0={MU0}, x_del={X_DEL}, y_ins={Y_INS}")
    print(f"  Tau: {TAU}")
    print(f"  N_cherries/bin = {N_CHERRIES}, L_mean = {L_MEAN}\n")

    counts = simulate_cherries(LAM0, MU0, X_DEL, Y_INS, TAU, N_CHERRIES, L_MEAN, seed=SEED)
    total = int(counts.sum())
    print(f"\nTotal transitions: {total}")

    # -----------------------------------------------------------------
    # Trajectory A: closed-form
    # -----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Trajectory A: CLOSED FORM (first-order-exact heuristic)")
    print(f"{'='*70}")
    lam_cf = np.zeros_like(TAU)
    mu_cf = np.zeros_like(TAU)
    r_cf = np.zeros_like(TAU)
    for ti, t in enumerate(TAU):
        # fit_ggi_cherryml.ggi_to_tkf92_at_t is JAX; call with floats
        lam_t, mu_t, r_t = ggi_to_tkf92_at_t(
            jnp.float32(LAM0), jnp.float32(MU0),
            jnp.float32(X_DEL), jnp.float32(Y_INS), jnp.float32(t))
        lam_cf[ti] = float(lam_t)
        mu_cf[ti] = float(mu_t)
        r_cf[ti] = float(r_t)
    print(f"  {'tau':>7}  {'lam_cf':>10}  {'mu_cf':>10}  {'r_cf':>9}")
    for ti, t in enumerate(TAU):
        print(f"  {t:>7.3f}  {lam_cf[ti]:>10.5f}  {mu_cf[ti]:>10.5f}  {r_cf[ti]:>9.4f}")

    # -----------------------------------------------------------------
    # Trajectory B: numerical ODE
    # scratch_ggi_triad_eliminated uses (x_ins, y_del) naming:
    #   x_ins = insertion length geom (= my Y_INS)
    #   y_del = deletion length geom  (= my X_DEL)
    # -----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Trajectory B: NUMERICAL ODE (scipy solve_ivp RK23)")
    print(f"{'='*70}")
    t_eps = TAU[0] * 0.5  # integrate from slightly before the smallest tau
    t_max = TAU[-1]
    t0 = time.monotonic()
    sol, bc = run_flow(LAM0, MU0, x_ins=Y_INS, y_del=X_DEL,
                              t_eps=t_eps, t_max=t_max, t_eval=TAU.copy(),
                              rtol=1e-6, atol=1e-8)
    print(f"  Integration done in {time.monotonic()-t0:.1f}s, "
          f"status={sol.status} ('{sol.message}')")
    print(f"  Boundary (lam*(0), mu*(0), r*(0)) = "
          f"({bc[0]:.4f}, {bc[1]:.4f}, {bc[2]:.4f})")
    lam_num = sol.y[0]
    mu_num = sol.y[1]
    r_num = sol.y[2]
    print(f"  {'tau':>7}  {'lam_num':>10}  {'mu_num':>10}  {'r_num':>9}")
    for ti, t in enumerate(TAU):
        print(f"  {t:>7.3f}  {lam_num[ti]:>10.5f}  {mu_num[ti]:>10.5f}  {r_num[ti]:>9.4f}")

    # -----------------------------------------------------------------
    # Conditional LL under each trajectory
    # -----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CONDITIONAL LL UNDER EACH TRAJECTORY")
    print(f"{'='*70}")
    ll_cf = conditional_ll_for_theta_trajectory(lam_cf, mu_cf, r_cf, counts, TAU)
    ll_num = conditional_ll_for_theta_trajectory(lam_num, mu_num, r_num, counts, TAU)
    print(f"  Closed-form trajectory:   LL = {ll_cf:.2f}")
    print(f"  Numerical ODE trajectory: LL = {ll_num:.2f}")
    print(f"  Delta (numerical - closed-form): {ll_num - ll_cf:+.2f}  "
          f"(per cherry: {(ll_num - ll_cf)/(N_CHERRIES*len(TAU)):.4f} nats)")

    # -----------------------------------------------------------------
    # Per-tau breakdown
    # -----------------------------------------------------------------
    print(f"\n  Per-tau LL breakdown:")
    print(f"  {'tau':>7}  {'LL_cf':>14}  {'LL_num':>14}  {'delta':>10}")
    for ti, t in enumerate(TAU):
        ll_cf_t = conditional_ll_for_theta_trajectory(
            lam_cf[ti:ti+1], mu_cf[ti:ti+1], r_cf[ti:ti+1],
            counts[ti:ti+1], TAU[ti:ti+1])
        ll_num_t = conditional_ll_for_theta_trajectory(
            lam_num[ti:ti+1], mu_num[ti:ti+1], r_num[ti:ti+1],
            counts[ti:ti+1], TAU[ti:ti+1])
        print(f"  {t:>7.3f}  {ll_cf_t:>14.2f}  {ll_num_t:>14.2f}  "
              f"{ll_num_t - ll_cf_t:>+10.2f}")


if __name__ == "__main__":
    main()
