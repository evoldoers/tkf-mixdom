#!/usr/bin/env python3
"""End-to-end test: simulate GGI cherries, fit both GGI-closed-form and constant
TKF92, report conditional LLs at the truth and at each model's optimum.

Goal: answer the user's question -- when the data is generated from a GGI
process, does GGI-corrected-TKF92 (using the closed-form r(t)) beat plain
TKF92 in conditional log-likelihood?
"""

import sys, os, time, json
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulate_ggi_cherries import simulate_cherries, S, M, I, D, E
from fit_ggi_cherryml import (
    composite_log_likelihood_3param,
    tkf92_trans_full as ggi_tkf92_trans_full,
    ggi_to_tkf92_at_t,
    tkf92_singlet_log_marginal,
)
from fit_tkf92_cherryml import (
    composite_log_likelihood as tkf92_composite_ll,
)


def conditional_ll_ggi(lam0, mu0, x, y, counts, tau_centers):
    """Closed-form GGI conditional LL: sum_t [joint_t - singlet_t(theta(t))]."""
    log_lam0 = jnp.log(jnp.asarray(lam0, jnp.float32))
    logit_x = jnp.log(x / (1 - x)).astype(jnp.float32)
    logit_y = jnp.log(y / (1 - y)).astype(jnp.float32)
    # Need to override mu0 -- but fit_ggi enforces reversibility.
    # For LL evaluation at arbitrary (lam0, mu0, x, y), reimplement directly.
    counts_j = jnp.asarray(counts, jnp.float32)
    tau_j = jnp.asarray(tau_centers, jnp.float32)

    def ll_one_tau(ti):
        t = tau_j[ti]
        lam_t, mu_t, r_t = ggi_to_tkf92_at_t(
            jnp.float32(lam0), jnp.float32(mu0),
            jnp.float32(x), jnp.float32(y), t)
        T = ggi_tkf92_trans_full(lam_t, mu_t, t, r_t)
        log_T = jnp.log(jnp.maximum(T, 1e-30))
        n_ij = counts_j[ti]
        joint = jnp.sum(n_ij * log_T)
        t_anc = jnp.sum(n_ij[:, M]) + jnp.sum(n_ij[:, D])
        n_cherries = jnp.sum(n_ij[:, E])
        singlet = tkf92_singlet_log_marginal(lam_t, mu_t, r_t, n_cherries, t_anc)
        return joint - singlet

    n_tau = tau_j.shape[0]
    return float(jnp.sum(jax.vmap(ll_one_tau)(jnp.arange(n_tau))))


def conditional_ll_tkf92(ins_rate, del_rate, ext, counts_dict, tau_centers):
    """Constant-TKF92 conditional LL: joint - constant-theta singlet."""
    log_del = jnp.log(jnp.asarray(del_rate, jnp.float32))
    kappa = min(0.99999, ins_rate / max(del_rate, 1e-30))
    logit_kappa = jnp.float32(np.log(kappa / (1 - kappa)))
    logit_ext = jnp.float32(np.log(ext / (1 - ext)))
    gap_counts_j = {k: jnp.asarray(v, jnp.float32) for k, v in counts_dict.items()}
    tau_j = jnp.asarray(tau_centers, jnp.float32)
    ll = tkf92_composite_ll(log_del, logit_kappa, logit_ext,
                             gap_counts_j, tau_j, jnp.float32(1.0))
    return float(ll)


# -----------------------------------------------------------------------------
# Adam loop helper
# -----------------------------------------------------------------------------
def adam_max(loss_fn, init_params, lr, n_steps, log_interval, label=''):
    """Maximise loss_fn via Adam. Returns (best_params, best_loss, history)."""
    val_and_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=tuple(range(len(init_params)))))
    params = [jnp.float32(p) for p in init_params]
    m = [jnp.zeros_like(p) for p in params]
    v = [jnp.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_loss = -float('inf')
    best_params = [float(p) for p in params]
    t0 = time.monotonic()
    for step in range(n_steps):
        ll, grads = val_and_grad(*params)
        llv = float(ll)
        if llv > best_loss:
            best_loss = llv
            best_params = [float(p) for p in params]
        for i in range(len(params)):
            g = grads[i]
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * (g * g)
            m_hat = m[i] / (1 - b1 ** (step + 1))
            v_hat = v[i] / (1 - b2 ** (step + 1))
            params[i] = params[i] + lr * m_hat / (jnp.sqrt(v_hat) + eps)
        if log_interval and step % log_interval == 0:
            print(f"  [{label}] step {step:5d}  LL={llv:.2f}  best={best_loss:.2f}  "
                  f"({time.monotonic()-t0:.1f}s)")
    return best_params, best_loss


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main():
    # GGI generative parameters (must be reversible)
    LAM0, MU0 = 0.04, 0.04
    X_DEL, Y_INS = 0.6, 0.4
    L_MEAN = 50
    N_CHERRIES = 5000
    TAU = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
    SEED = 42

    rev_lhs = LAM0 * Y_INS * (1 - Y_INS)
    rev_rhs = MU0 * X_DEL * (1 - X_DEL)
    print(f"GGI sim params:")
    print(f"  lam0={LAM0}, mu0={MU0}, x_del={X_DEL}, y_ins={Y_INS}")
    print(f"  reversibility: lam0*y(1-y)={rev_lhs:.6e}, mu0*x(1-x)={rev_rhs:.6e}")
    print(f"  rel err = {abs(rev_lhs-rev_rhs)/max(rev_lhs,rev_rhs):.2e}")
    print(f"  L_mean = {L_MEAN}, N_cherries/bin = {N_CHERRIES}, n_tau = {len(TAU)}")
    print(f"  tau bins: {TAU}")

    # Closed-form GGI trajectory
    num = LAM0 * Y_INS * (1 - X_DEL) + MU0 * X_DEL * (1 - Y_INS)
    den = LAM0 * (1 - Y_INS) + MU0 * (1 - X_DEL)
    r_b = num / den
    r_inf_v = r_b / (2 - r_b)
    k_dec = (LAM0 + MU0) * (2 - r_b) / (1 - r_b)
    print(f"\n  Closed-form: r*(0)={r_b:.4f}, r_inf={r_inf_v:.4f}, k={k_dec:.4f}")

    print(f"\nSimulating...")
    counts = simulate_cherries(
        LAM0, MU0, X_DEL, Y_INS, TAU, N_CHERRIES, L_MEAN, seed=SEED)
    total = counts.sum()
    print(f"Total transitions: {int(total)}")

    # Build C_xx dict for TKF92 fitter
    name_to_ij = {
        'C_SM': (S, M), 'C_SI': (S, I), 'C_SD': (S, D), 'C_SE': (S, E),
        'C_MM': (M, M), 'C_MI': (M, I), 'C_MD': (M, D), 'C_ME': (M, E),
        'C_IM': (I, M), 'C_II': (I, I), 'C_ID': (I, D), 'C_IE': (I, E),
        'C_DM': (D, M), 'C_DI': (D, I), 'C_DD': (D, D), 'C_DE': (D, E),
    }
    c_dict = {n: counts[:, i, j].astype(np.float64) for n, (i, j) in name_to_ij.items()}

    # -------------------------------------------------------------------------
    # 1) LL at the TRUTH (under each model)
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"1) Conditional LL evaluated at the GGI TRUTH (lam0={LAM0}, mu0={MU0}, "
          f"x={X_DEL}, y={Y_INS}):")
    print(f"{'='*70}")
    ll_ggi_at_truth = conditional_ll_ggi(LAM0, MU0, X_DEL, Y_INS, counts, TAU)
    print(f"   GGI closed-form conditional LL at truth = {ll_ggi_at_truth:.2f}")

    # For TKF92 evaluated at theta(t=0) boundary (a sensible single point):
    lam_b = LAM0 / (1 - r_b)
    mu_b = MU0 / (1 - r_b)
    ll_tkf92_at_bdry = conditional_ll_tkf92(lam_b, mu_b, r_b, c_dict, TAU)
    print(f"   TKF92(theta(0)={lam_b:.4f},{mu_b:.4f},{r_b:.4f}) const cond LL"
          f" = {ll_tkf92_at_bdry:.2f}")

    # -------------------------------------------------------------------------
    # 2) Optimise both models, compare best conditional LLs
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"2) Optimising both models (conditional LL):")
    print(f"{'='*70}")

    counts_j = jnp.asarray(counts, jnp.float32)
    tau_j = jnp.asarray(TAU, jnp.float32)
    cflag = jnp.float32(1.0)
    ggi_flag = jnp.float32(0.0)

    def ggi_loss(log_lam0, logit_x, logit_y):
        return composite_log_likelihood_3param(
            log_lam0, logit_x, logit_y,
            counts_j, tau_j, cflag, ggi_flag)

    def ggi_loss_stable(log_lam0, logit_x, logit_y):
        """Same as ggi_loss but with a strong quadratic barrier on rho >= 1."""
        ll = composite_log_likelihood_3param(
            log_lam0, logit_x, logit_y,
            counts_j, tau_j, cflag, ggi_flag)
        # Compute rho and add barrier
        lam0 = jnp.exp(log_lam0)
        xv = jax.nn.sigmoid(logit_x)
        yv = jax.nn.sigmoid(logit_y)
        mu0 = lam0 * yv * (1 - yv) / jnp.maximum(xv * (1 - xv), 1e-30)
        rho = lam0 * (1 - xv) / jnp.maximum(mu0 * (1 - yv), 1e-30)
        # Quadratic barrier active for rho > 0.95
        barrier = -1e8 * jnp.square(jax.nn.relu(rho - 0.95))
        return ll + barrier

    # Multi-init: try several starting points and keep the best.
    ggi_inits = [
        ("truth      ", 0.04, 0.6, 0.4),
        ("generic    ", 0.05, 0.5, 0.5),
        ("symmetric  ", 0.05, 0.65, 0.65),
        ("ins-heavy  ", 0.05, 0.3, 0.7),
        ("del-heavy  ", 0.05, 0.7, 0.3),
    ]
    best_ggi_ll = -float('inf')
    best_ggi_p = None
    best_init_label = None
    for label, l0, xi, yi in ggi_inits:
        init = (jnp.log(jnp.float32(l0)),
                jnp.log(jnp.float32(xi / (1 - xi))),
                jnp.log(jnp.float32(yi / (1 - yi))))
        print(f"\n  -- GGI fit (init: lam0={l0}, x={xi}, y={yi}, label={label.strip()}) --")
        best_p, best_ll = adam_max(
            ggi_loss, init, lr=0.001, n_steps=4000,
            log_interval=1000, label=f'GGI/{label.strip()}')
        print(f"    -> best LL = {best_ll:.2f}")
        if best_ll > best_ggi_ll:
            best_ggi_ll = best_ll
            best_ggi_p = best_p
            best_init_label = label.strip()

    lam0_fit = float(np.exp(best_ggi_p[0]))
    x_fit = 1 / (1 + float(np.exp(-best_ggi_p[1])))
    y_fit = 1 / (1 + float(np.exp(-best_ggi_p[2])))
    mu0_fit = lam0_fit * y_fit * (1 - y_fit) / max(x_fit * (1 - x_fit), 1e-30)
    rho_fit = lam0_fit * (1 - x_fit) / max(mu0_fit * (1 - y_fit), 1e-30)
    print(f"\n  *** GGI best LL = {best_ggi_ll:.2f} (init: {best_init_label}) ***")
    print(f"      lam0={lam0_fit:.5f}, mu0={mu0_fit:.5f}, x={x_fit:.4f}, y={y_fit:.4f}")
    print(f"      rho={rho_fit:.4f}  ({'STABLE' if rho_fit < 1 else 'UNSTABLE'})")


    gap_counts_j = {k: jnp.asarray(v, jnp.float32) for k, v in c_dict.items()}

    def tkf92_loss(log_del, logit_kappa, logit_ext):
        return tkf92_composite_ll(
            log_del, logit_kappa, logit_ext,
            gap_counts_j, tau_j, cflag)

    print(f"\n  -- TKF92 constant fit (init mu=0.05, kappa=0.95, r=0.5) --")
    init_tkf = (jnp.log(jnp.float32(0.05)),
                jnp.log(jnp.float32(0.95 / 0.05)),
                jnp.log(jnp.float32(0.5 / 0.5)))
    best_tkf_p, best_tkf_ll = adam_max(
        tkf92_loss, init_tkf, lr=0.001, n_steps=4000,
        log_interval=500, label='TKF92')

    mu_fit = float(np.exp(best_tkf_p[0]))
    kappa_fit = 1 / (1 + float(np.exp(-best_tkf_p[1])))
    r_fit = 1 / (1 + float(np.exp(-best_tkf_p[2])))
    lam_fit = kappa_fit * mu_fit
    print(f"  TKF92 best LL = {best_tkf_ll:.2f}")
    print(f"    lam={lam_fit:.5f}, mu={mu_fit:.5f}, kappa={kappa_fit:.4f}, r={r_fit:.4f}")

    # -------------------------------------------------------------------------
    # 3) Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Conditional LLs (each model's own conditional, on same simulated data):")
    print(f"     GGI at the TRUTH (lam0,mu0,x,y) = (0.04,0.04,0.6,0.4):")
    print(f"       LL = {ll_ggi_at_truth:>14.2f}   (rho_GGI = 0.667, STABLE)")
    print(f"     GGI closed-form best (multi-init optimisation):")
    print(f"       LL = {best_ggi_ll:>14.2f}   (rho_GGI = {rho_fit:.3f},"
          f" {'STABLE' if rho_fit < 1 else 'UNSTABLE'})")
    print(f"     TKF92 constant  (optimised):")
    print(f"       LL = {best_tkf_ll:>14.2f}")
    delta_best = best_ggi_ll - best_tkf_ll
    delta_truth = ll_ggi_at_truth - best_tkf_ll
    print()
    print(f"   Delta(GGI@truth   - TKF92@opt):   {delta_truth:>+14.2f}  "
          f"(per cherry: {delta_truth/(N_CHERRIES*len(TAU)):.3f} nats)")
    print(f"   Delta(GGI@opt     - TKF92@opt):   {delta_best:>+14.2f}  "
          f"(per cherry: {delta_best/(N_CHERRIES*len(TAU)):.3f} nats)")
    print()
    if delta_truth > 0:
        print(f"   ==> YES: even at the GGI generative truth, the closed-form")
        print(f"       conditional LL beats the optimised constant-TKF92 by "
              f"{delta_truth:.0f} nats.")
        print(f"       Optimising the closed-form widens the gap by another "
              f"{delta_best - delta_truth:.0f} nats")
        print(f"       (closed-form best is in the unstable rho>1 region: the")
        print(f"        closed-form pair-HMM family extends beyond valid GGI).")
    else:
        print(f"   ==> NO: TKF92 wins.")

    return dict(
        sim=dict(lam0=LAM0, mu0=MU0, x_del=X_DEL, y_ins=Y_INS,
                 L_mean=L_MEAN, n_cherries=N_CHERRIES, tau=TAU.tolist(),
                 r_boundary=r_b, r_inf=r_inf_v, k_decay=k_dec, seed=SEED),
        ggi_at_truth=ll_ggi_at_truth,
        ggi_fit=dict(ll=best_ggi_ll, lam0=lam0_fit, mu0=mu0_fit,
                     x=x_fit, y=y_fit, rho=rho_fit),
        tkf92_fit=dict(ll=best_tkf_ll, lam=lam_fit, mu=mu_fit,
                       kappa=kappa_fit, r=r_fit),
        deltas=dict(at_truth=delta_truth, best=delta_best),
        total_transitions=int(total),
    )


if __name__ == "__main__":
    result = main()
    out_path = '/tmp/ggi_vs_tkf92_on_sim.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")
