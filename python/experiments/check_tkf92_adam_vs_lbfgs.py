#!/usr/bin/env python3
"""Rule out Adam-vs-EM optimization noise for the TKF92-only conditional fit.

Runs:
  1. Multi-init Adam (small LR, many steps) — same protocol as the comparison.
  2. scipy L-BFGS (deterministic) on the JAX gradient.
Then reports the best LL each finds.  If L-BFGS finds the same LL as
multi-init Adam, the GGI advantage over TKF92 isn't an optimization artefact.
"""
import sys, os, time
import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate_ggi_cherries import simulate_cherries, S, M, I, D, E
from fit_tkf92_cherryml import composite_log_likelihood as tkf92_cll


# Sim params (must match compare_ggi_vs_tkf92_on_sim.py for reproducibility)
LAM0, MU0 = 0.04, 0.04
X_DEL, Y_INS = 0.6, 0.4
L_MEAN = 50
N_CHERRIES = 5000
TAU = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
SEED = 42


def main():
    print(f"Simulating GGI cherries (same protocol as compare_ggi_vs_tkf92_on_sim.py)...")
    counts = simulate_cherries(LAM0, MU0, X_DEL, Y_INS, TAU, N_CHERRIES, L_MEAN, seed=SEED)
    print(f"  Total transitions: {int(counts.sum())}")

    # Build C_xx dict for TKF92 loss
    name_to_ij = {
        'C_SM': (S, M), 'C_SI': (S, I), 'C_SD': (S, D), 'C_SE': (S, E),
        'C_MM': (M, M), 'C_MI': (M, I), 'C_MD': (M, D), 'C_ME': (M, E),
        'C_IM': (I, M), 'C_II': (I, I), 'C_ID': (I, D), 'C_IE': (I, E),
        'C_DM': (D, M), 'C_DI': (D, I), 'C_DD': (D, D), 'C_DE': (D, E),
    }
    gap_counts = {n: jnp.asarray(counts[:, i, j], jnp.float64)
                  for n, (i, j) in name_to_ij.items()}
    tau_j = jnp.asarray(TAU, jnp.float64)
    cflag = jnp.float64(1.0)

    def loss(params):
        return tkf92_cll(params[0], params[1], params[2], gap_counts, tau_j, cflag)

    val_and_grad = jax.jit(jax.value_and_grad(loss))
    # Compile
    init = jnp.asarray([np.log(0.05), np.log(0.95/0.05), 0.0], dtype=jnp.float64)
    ll0, g0 = val_and_grad(init)
    jax.block_until_ready(ll0)

    # ------------------------------------------------------------------
    # 1) Multi-init Adam
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("1) Multi-init Adam")
    print(f"{'='*70}")
    inits_tkf = [
        ("init (κ=0.95, r=0.5)",  0.05, 0.95, 0.5),
        ("init (κ=0.9, r=0.3)",   0.04, 0.9,  0.3),
        ("init (κ=0.99, r=0.7)",  0.04, 0.99, 0.7),
        ("init (κ=0.8, r=0.6)",   0.06, 0.8,  0.6),
        ("init (κ=0.95, r=0.504)",0.083, 0.95, 0.504),  # near previous fit
    ]
    best_adam_ll = -float('inf')
    best_adam_params = None
    best_adam_init = None
    for label, mu_init, kappa_init, r_init in inits_tkf:
        params = jnp.asarray([
            np.log(mu_init),
            np.log(kappa_init / (1 - kappa_init)),
            np.log(r_init / (1 - r_init)),
        ], dtype=jnp.float64)
        m = jnp.zeros_like(params); v = jnp.zeros_like(params)
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.001
        n_steps = 10000
        local_best = -float('inf')
        local_best_p = params
        t0 = time.monotonic()
        for step in range(n_steps):
            ll, g = val_and_grad(params)
            llv = float(ll)
            if llv > local_best:
                local_best = llv
                local_best_p = params
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            m_hat = m / (1 - b1 ** (step + 1))
            v_hat = v / (1 - b2 ** (step + 1))
            params = params + lr * m_hat / (jnp.sqrt(v_hat) + eps)
        print(f"  {label}: best LL = {local_best:.2f}  ({time.monotonic()-t0:.1f}s)")
        if local_best > best_adam_ll:
            best_adam_ll = local_best
            best_adam_params = local_best_p
            best_adam_init = label

    mu_a = float(jnp.exp(best_adam_params[0]))
    kappa_a = 1 / (1 + float(jnp.exp(-best_adam_params[1])))
    r_a = 1 / (1 + float(jnp.exp(-best_adam_params[2])))
    lam_a = kappa_a * mu_a
    print(f"\n  *** Best Adam LL = {best_adam_ll:.2f} ({best_adam_init}) ***")
    print(f"      lam={lam_a:.5f}, mu={mu_a:.5f}, kappa={kappa_a:.4f}, r={r_a:.4f}")

    # ------------------------------------------------------------------
    # 2) scipy L-BFGS (deterministic, second-order)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("2) scipy L-BFGS (deterministic, multi-init)")
    print(f"{'='*70}")
    def neg_ll_and_grad_np(p_np):
        p = jnp.asarray(p_np, dtype=jnp.float64)
        ll, g = val_and_grad(p)
        return -float(ll), -np.asarray(g, dtype=np.float64)

    best_lbfgs_ll = -float('inf')
    best_lbfgs_x = None
    best_lbfgs_init = None
    for label, mu_init, kappa_init, r_init in inits_tkf:
        x0 = np.array([
            np.log(mu_init),
            np.log(kappa_init / (1 - kappa_init)),
            np.log(r_init / (1 - r_init)),
        ])
        t0 = time.monotonic()
        res = scipy.optimize.minimize(
            neg_ll_and_grad_np, x0, jac=True, method='L-BFGS-B',
            options=dict(ftol=1e-10, gtol=1e-8, maxiter=1000))
        ll = -res.fun
        print(f"  {label}: LL = {ll:.2f}  (converged={res.success}, "
              f"nit={res.nit}, {time.monotonic()-t0:.1f}s)")
        if ll > best_lbfgs_ll:
            best_lbfgs_ll = ll
            best_lbfgs_x = res.x
            best_lbfgs_init = label

    mu_l = float(np.exp(best_lbfgs_x[0]))
    kappa_l = 1 / (1 + float(np.exp(-best_lbfgs_x[1])))
    r_l = 1 / (1 + float(np.exp(-best_lbfgs_x[2])))
    lam_l = kappa_l * mu_l
    print(f"\n  *** Best L-BFGS LL = {best_lbfgs_ll:.2f} ({best_lbfgs_init}) ***")
    print(f"      lam={lam_l:.5f}, mu={mu_l:.5f}, kappa={kappa_l:.4f}, r={r_l:.4f}")

    # ------------------------------------------------------------------
    # Compare to prior (GGI@truth) result
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Multi-init Adam  best LL:   {best_adam_ll:>14.2f}")
    print(f"  scipy L-BFGS     best LL:   {best_lbfgs_ll:>14.2f}")
    print(f"  Delta (L-BFGS - Adam):       {best_lbfgs_ll - best_adam_ll:>+14.2f}")
    print()
    print(f"  (For reference, from compare_ggi_vs_tkf92_on_sim.py:)")
    print(f"    GGI@truth conditional LL:  -664563.38")
    print(f"    -> Even if Adam left some TKF92 LL on the table, the gap")
    print(f"       to GGI@truth is what really matters.")
    print()
    print(f"  Delta(GGI@truth - TKF92@L-BFGS-best):   "
          f"{-664563.38 - best_lbfgs_ll:>+14.2f}")


if __name__ == "__main__":
    main()
