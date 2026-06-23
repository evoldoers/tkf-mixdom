#!/usr/bin/env python3
"""Differentiable diffrax-based fit of GGI parameters via the full matched-flow
ODE for (lam(t), mu(t), r(t)).

Same data and same conditional-LL objective as fit_ggi_cherryml.py, but with
the closed-form trajectory replaced by numerical diffrax integration of the
JAX matched-flow ODE in jax_matched_flow_ode.py.

Three free GGI params (log_lam0, logit_x_del, logit_y_ins); mu0 derived from
reversibility lam0 y(1-y) = mu0 x(1-x).
"""
import sys, os, time, json
import numpy as np
# Enable float64 BEFORE importing jax-using modules.
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import diffrax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate_ggi_cherries import simulate_cherries, S, M, I, D, E
from jax_matched_flow_ode import (
    boundary_jax, dtheta_dt_jax, jax_tau5,
)
from fit_ggi_cherryml import (
    tkf92_trans_full, tkf92_singlet_log_marginal,
)


# ---------------------------------------------------------------------------
# diffrax integration: returns (lam(t), mu(t), r(t)) at each tau center.
# ---------------------------------------------------------------------------

def integrate_flow(lam0, mu0, x_del, y_ins, tau_centers, t_eps=1e-3):
    """Integrate (lam, mu, r) ODE from t_eps to max(tau).  Returns shape
    (n_tau, 3) array of (lam, mu, r) at each tau center."""
    y0 = boundary_jax(lam0, mu0, x_del, y_ins)
    term = diffrax.ODETerm(
        lambda t, y, args: dtheta_dt_jax(t, y, lam0, mu0, x_del, y_ins))
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=tau_centers)
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=t_eps, t1=jnp.max(tau_centers),
        dt0=1e-3, y0=y0, saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        max_steps=16384,
        throw=False,  # don't throw on max-steps; return whatever was computed
    )
    return sol.ys


# ---------------------------------------------------------------------------
# Conditional LL
# ---------------------------------------------------------------------------

def unpack_ggi(log_lam0, logit_x_del, logit_y_ins):
    lam0 = jnp.exp(log_lam0)
    x = jax.nn.sigmoid(logit_x_del)
    y = jax.nn.sigmoid(logit_y_ins)
    # Reversibility: lam0 * y(1-y) = mu0 * x(1-x)  =>  mu0 = lam0 y(1-y) / (x(1-x))
    mu0 = lam0 * y * (1 - y) / jnp.maximum(x * (1 - x), 1e-30)
    return lam0, mu0, x, y


def conditional_ll_diffrax(log_lam0, logit_x_del, logit_y_ins,
                            counts_jax, tau_centers):
    lam0, mu0, x_del, y_ins = unpack_ggi(log_lam0, logit_x_del, logit_y_ins)
    ys = integrate_flow(lam0, mu0, x_del, y_ins, tau_centers)
    n_tau = tau_centers.shape[0]

    def ll_one_tau(ti):
        t = tau_centers[ti]
        lam_t, mu_t, r_t = ys[ti, 0], ys[ti, 1], ys[ti, 2]
        T = tkf92_trans_full(lam_t, mu_t, t, r_t)
        log_T = jnp.log(jnp.maximum(T, 1e-30))
        n_ij = counts_jax[ti]
        joint = jnp.sum(n_ij * log_T)
        t_anc = jnp.sum(n_ij[:, M]) + jnp.sum(n_ij[:, D])
        n_cherries = jnp.sum(n_ij[:, E])
        singlet = tkf92_singlet_log_marginal(lam_t, mu_t, r_t, n_cherries, t_anc)
        return joint - singlet

    return jnp.sum(jax.vmap(ll_one_tau)(jnp.arange(n_tau)))


# ---------------------------------------------------------------------------
# Adam loop
# ---------------------------------------------------------------------------

def adam_max(val_and_grad, init_params, lr, n_steps, log_interval, label=''):
    params = [jnp.float64(p) for p in init_params]
    m = [jnp.zeros_like(p) for p in params]
    v = [jnp.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = -float('inf')
    best_p = [float(p) for p in params]
    t0 = time.monotonic()
    for step in range(n_steps):
        ll, grads = val_and_grad(*params)
        llv = float(ll)
        if llv > best:
            best = llv
            best_p = [float(p) for p in params]
        for i in range(len(params)):
            g = grads[i]
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * (g * g)
            mh = m[i] / (1 - b1 ** (step + 1))
            vh = v[i] / (1 - b2 ** (step + 1))
            params[i] = params[i] + lr * mh / (jnp.sqrt(vh) + eps)
        if log_interval and step % log_interval == 0:
            elapsed = time.monotonic() - t0
            print(f"  [{label}] step {step:5d}  LL={llv:12.2f}  best={best:12.2f}"
                  f"  ({elapsed:.1f}s)")
    return best_p, best


def main():
    # Pick reversible GGI with x_del != y_ins.  Under reversibility
    #   lam0 y(1-y) = mu0 x(1-x),
    # so x(1-x) != y(1-y) is what lets lam0 != mu0.
    # For x_del=0.55, y_ins=0.45 we get x(1-x)=0.2475, y(1-y)=0.2475 (equal!).
    # We need asymmetric (x(1-x), y(1-y)).  Use x_del=0.7, y_ins=0.4:
    #   x(1-x) = 0.21,  y(1-y) = 0.24,  ratio lam0/mu0 = 0.21/0.24 = 0.875.
    LAM0_TRUE, MU0_TRUE = 0.0350, 0.0400  # ratio 0.875, reversible up to rounding
    X_DEL_TRUE, Y_INS_TRUE = 0.7, 0.4
    L_MEAN = 50
    N_CHERRIES = 5000
    # Reduced tau set so diffrax-+-grad JIT compilation finishes in <2 min.
    TAU = np.array([0.05, 0.2, 1.0, 5.0], dtype=np.float64)
    SEED = 42

    print(f"GGI sim params (asymmetric so the JAX ODE is in the easy regime):")
    print(f"  lam0={LAM0_TRUE}, mu0={MU0_TRUE}, x_del={X_DEL_TRUE}, y_ins={Y_INS_TRUE}")
    print(f"  Tau: {TAU}")

    counts = simulate_cherries(
        LAM0_TRUE, MU0_TRUE, X_DEL_TRUE, Y_INS_TRUE,
        TAU, N_CHERRIES, L_MEAN, seed=SEED)
    print(f"\nTotal transitions: {int(counts.sum())}")

    counts_j = jnp.asarray(counts, jnp.float64)
    tau_j = jnp.asarray(TAU, jnp.float64)

    def loss(log_lam0, logit_x_del, logit_y_ins):
        return conditional_ll_diffrax(
            log_lam0, logit_x_del, logit_y_ins, counts_j, tau_j)

    val_and_grad = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))

    # ----- Evaluate at the GGI TRUTH -----
    print(f"\n{'='*70}")
    print(f"Evaluating LL at the GGI generative TRUTH...")
    print(f"{'='*70}")
    init_truth = (
        jnp.log(jnp.float64(LAM0_TRUE)),
        jnp.log(jnp.float64(X_DEL_TRUE / (1 - X_DEL_TRUE))),
        jnp.log(jnp.float64(Y_INS_TRUE / (1 - Y_INS_TRUE))),
    )
    t0 = time.monotonic()
    ll_truth, _ = val_and_grad(*init_truth)
    jax.block_until_ready(ll_truth)
    print(f"  Diffrax ODE LL at TRUTH = {float(ll_truth):.2f}  "
          f"(JIT compile + first eval: {time.monotonic()-t0:.1f}s)")

    # ----- Multi-init Adam -----
    print(f"\n{'='*70}")
    print(f"Multi-init Adam fit (diffrax-driven LL)")
    print(f"{'='*70}")
    inits = [
        ("truth      ", LAM0_TRUE, X_DEL_TRUE, Y_INS_TRUE),
        ("generic    ", 0.05, 0.5, 0.5),
        ("ins-heavy  ", 0.05, 0.5, 0.6),
    ]
    best_ll = -float('inf')
    best_p = None
    best_init = None
    for label, l0, xi, yi in inits:
        init = (jnp.log(jnp.float64(l0)),
                jnp.log(jnp.float64(xi / (1 - xi))),
                jnp.log(jnp.float64(yi / (1 - yi))))
        print(f"\n  -- init: lam0={l0}, x_del={xi}, y_ins={yi} ({label.strip()}) --")
        bp, bll = adam_max(val_and_grad, init, lr=0.005, n_steps=300,
                            log_interval=50, label=label.strip())
        if bll > best_ll:
            best_ll = bll
            best_p = bp
            best_init = label.strip()

    lam0_fit = float(np.exp(best_p[0]))
    x_fit = 1 / (1 + float(np.exp(-best_p[1])))
    y_fit = 1 / (1 + float(np.exp(-best_p[2])))
    mu0_fit = lam0_fit * y_fit * (1 - y_fit) / max(x_fit * (1 - x_fit), 1e-30)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Diffrax ODE LL at TRUTH                       = {float(ll_truth):14.2f}")
    print(f"  Diffrax ODE LL best (multi-init Adam)         = {best_ll:14.2f}  "
          f"(init: {best_init})")
    print(f"    lam0={lam0_fit:.5f}, mu0={mu0_fit:.5f}, "
          f"x_del={x_fit:.4f}, y_ins={y_fit:.4f}")
    print()
    # For reference, the closed-form fit on the same simulated data:
    # see compare_ggi_vs_tkf92_on_sim.py / compare_closedform_vs_odeflow.py.
    print(f"  Reference numbers (from previous experiments on the same sim):")
    print(f"    closed-form @ truth (eval only): see compare_closedform_vs_odeflow.py")
    print(f"    closed-form best fit            : see compare_ggi_vs_tkf92_on_sim.py")


if __name__ == "__main__":
    main()
