#!/usr/bin/env python3
"""Settle Q2 (diffrax-based ODE fit vs closed-form fit) on GGI-simulated data.

Plan:
  1. Simulate GGI cherries with asymmetric, reversible parameters.
  2. Run closed-form Adam fit (the existing fit_ggi_cherryml conditional
     objective).  Get best closed-form params.
  3. Evaluate the diffrax-ODE conditional LL at:
       - the GGI generative truth
       - the closed-form best fit
  4. Run a SHORT diffrax-driven Adam from the closed-form best fit, to see
     whether the ODE objective improves beyond what the closed-form found.

Conclusion: tells us whether the closed-form heuristic is leaving LL on the
table that a full-ODE fit could pick up.
"""
import sys, os, time, json
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate_ggi_cherries import simulate_cherries, S, M, I, D, E
from fit_ggi_cherryml import (
    composite_log_likelihood_3param, tkf92_trans_full,
    tkf92_singlet_log_marginal, ggi_to_tkf92_at_t,
)
from fit_ggi_diffrax import conditional_ll_diffrax


# Use 4 tau bins so diffrax+grad JIT fits in ~25 s
TAU = np.array([0.05, 0.2, 1.0, 5.0], dtype=np.float64)
LAM0_TRUE, MU0_TRUE = 0.0350, 0.0400  # ratio 0.875, reversible
X_DEL_TRUE, Y_INS_TRUE = 0.7, 0.4
L_MEAN = 50
N_CHERRIES = 5000
SEED = 42


def adam_max(val_and_grad, init_params, lr, n_steps, log_interval, label='', tol=None):
    params = list(init_params)
    m = [jnp.zeros_like(p) for p in params]
    v = [jnp.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = -float('inf')
    best_p = [float(p) for p in params]
    t0 = time.monotonic()
    for step in range(n_steps):
        ll, grads = val_and_grad(*params)
        llv = float(ll)
        if not np.isfinite(llv):
            print(f"  [{label}] step {step}: non-finite LL ({llv}), stopping")
            break
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
            lam0v = float(jnp.exp(params[0]))
            xv = float(jax.nn.sigmoid(params[1]))
            yv = float(jax.nn.sigmoid(params[2]))
            print(f"  [{label}] step {step:5d}  LL={llv:14.2f}  best={best:14.2f}  "
                  f"lam0={lam0v:.4f} x={xv:.4f} y={yv:.4f}  ({elapsed:.1f}s)")
    return best_p, best


def main():
    print(f"GGI sim params (asymmetric reversible):")
    print(f"  lam0={LAM0_TRUE}, mu0={MU0_TRUE}, x_del={X_DEL_TRUE}, y_ins={Y_INS_TRUE}")
    print(f"  Tau: {TAU}")
    print()

    counts = simulate_cherries(
        LAM0_TRUE, MU0_TRUE, X_DEL_TRUE, Y_INS_TRUE,
        TAU, N_CHERRIES, L_MEAN, seed=SEED)
    print(f"\nTotal transitions: {int(counts.sum())}")
    counts_j = jnp.asarray(counts, jnp.float64)
    tau_j = jnp.asarray(TAU, jnp.float64)

    # ---- 1) Closed-form Adam fit ----
    print(f"\n{'='*70}")
    print(f"1) Closed-form Adam fit (existing fit_ggi_cherryml.composite_log_likelihood_3param)")
    print(f"{'='*70}")
    cflag = jnp.float64(1.0)
    ggi_flag = jnp.float64(0.0)
    def cf_loss(p0, p1, p2):
        return composite_log_likelihood_3param(p0, p1, p2, counts_j, tau_j,
                                                  cflag, ggi_flag)
    cf_vag = jax.jit(jax.value_and_grad(cf_loss, argnums=(0, 1, 2)))

    cf_inits = [
        ('truth',    LAM0_TRUE, X_DEL_TRUE, Y_INS_TRUE),
        ('generic',  0.05, 0.5, 0.5),
        ('ins-heavy',0.05, 0.5, 0.6),
    ]
    cf_best_ll = -float('inf')
    cf_best_p = None
    for label, l0, xi, yi in cf_inits:
        init = [jnp.log(jnp.float64(l0)),
                jnp.log(jnp.float64(xi / (1 - xi))),
                jnp.log(jnp.float64(yi / (1 - yi)))]
        print(f"\n  init: {label} (lam0={l0}, x={xi}, y={yi})")
        bp, bll = adam_max(cf_vag, init, lr=0.002, n_steps=2000,
                            log_interval=500, label=f'cf/{label}')
        if bll > cf_best_ll:
            cf_best_ll = bll
            cf_best_p = bp

    lam0_cf = float(np.exp(cf_best_p[0]))
    x_cf = 1 / (1 + float(np.exp(-cf_best_p[1])))
    y_cf = 1 / (1 + float(np.exp(-cf_best_p[2])))
    mu0_cf = lam0_cf * y_cf * (1 - y_cf) / max(x_cf * (1 - x_cf), 1e-30)
    print(f"\n  *** closed-form best LL = {cf_best_ll:.2f} ***")
    print(f"      lam0={lam0_cf:.5f}, mu0={mu0_cf:.5f}, x={x_cf:.4f}, y={y_cf:.4f}")

    # ---- 2) ODE LL at the truth ----
    print(f"\n{'='*70}")
    print(f"2) Diffrax ODE LL evaluations")
    print(f"{'='*70}")
    print(f"   (each call needs ~25 s JIT compilation on first invocation)\n")
    def ode_loss(p0, p1, p2):
        return conditional_ll_diffrax(p0, p1, p2, counts_j, tau_j)
    ode_vag = jax.jit(jax.value_and_grad(ode_loss, argnums=(0, 1, 2)))

    truth = [jnp.log(jnp.float64(LAM0_TRUE)),
             jnp.log(jnp.float64(X_DEL_TRUE / (1 - X_DEL_TRUE))),
             jnp.log(jnp.float64(Y_INS_TRUE / (1 - Y_INS_TRUE)))]
    t0 = time.monotonic()
    ll_ode_truth, _ = ode_vag(*truth)
    jax.block_until_ready(ll_ode_truth)
    print(f"   ODE LL at TRUTH                = {float(ll_ode_truth):.2f}  "
          f"(first call: {time.monotonic()-t0:.1f}s)")

    cf_best = [jnp.asarray(p, jnp.float64) for p in cf_best_p]
    t0 = time.monotonic()
    ll_ode_at_cf, _ = ode_vag(*cf_best)
    jax.block_until_ready(ll_ode_at_cf)
    print(f"   ODE LL at closed-form best     = {float(ll_ode_at_cf):.2f}  "
          f"({time.monotonic()-t0:.1f}s)")

    # ---- 3) Short ODE-driven Adam from closed-form best ----
    print(f"\n{'='*70}")
    print(f"3) Short ODE-driven Adam from the closed-form best fit")
    print(f"{'='*70}")
    bp, bll = adam_max(ode_vag, list(cf_best), lr=0.001, n_steps=150,
                        log_interval=10, label='ode')
    lam0_ode = float(np.exp(bp[0]))
    x_ode = 1 / (1 + float(np.exp(-bp[1])))
    y_ode = 1 / (1 + float(np.exp(-bp[2])))
    print(f"\n   ODE Adam best LL = {bll:.2f}")
    print(f"      lam0={lam0_ode:.5f}, x={x_ode:.4f}, y={y_ode:.4f}")

    # ---- 4) Summary ----
    print(f"\n{'='*70}")
    print(f"SUMMARY (all conditional LLs on the same {int(counts.sum())} transitions)")
    print(f"{'='*70}")
    print(f"   ODE LL @ truth                 = {float(ll_ode_truth):>14.2f}")
    print(f"   closed-form LL @ truth         = (see closed-form Adam first step)")
    print(f"   closed-form BEST LL            = {cf_best_ll:>14.2f}")
    print(f"   ODE LL @ closed-form best      = {float(ll_ode_at_cf):>14.2f}")
    print(f"   ODE LL @ ODE-Adam best         = {bll:>14.2f}")
    print()
    print(f"   delta(ODE@truth - cf@best)         = "
          f"{float(ll_ode_truth) - cf_best_ll:+.2f}")
    print(f"   delta(ODE@cf-best - cf@best)       = "
          f"{float(ll_ode_at_cf) - cf_best_ll:+.2f}")
    print(f"   delta(ODE-Adam best - cf@best)     = "
          f"{bll - cf_best_ll:+.2f}")


if __name__ == "__main__":
    main()
