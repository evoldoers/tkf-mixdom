#!/usr/bin/env python3
"""Fit TKF92 to the 5x5 transition counts that come from the SAME cherry
extraction as our gap counts (pfam_gap_counts.npz's trans_counts), via
joint LL.  This gives a fair comparison with the joint-native gap-counts
TKF92 fit.
"""
import argparse, json, os, sys, time
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.dirname(THIS_DIR))

from cross_eval_gap_vs_trans import joint_ll_trans_tkf92, _logit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="pfam_gap_counts.npz")
    p.add_argument("--out", default="experiments/tkf92_fitted_on_gap_trans.json")
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--n-steps", type=int, default=5000)
    args = p.parse_args()

    d = np.load(args.npz)
    t = d["trans_counts"].astype(np.float32)
    tau = d["tau_centers"].astype(np.float32)
    tj = jnp.asarray(t); taj = jnp.asarray(tau)
    print(f"trans_counts shape: {tj.shape}  total: {int(tj.sum()):,}")

    def loss(log_del, logit_kappa, logit_ext):
        return joint_ll_trans_tkf92(log_del, logit_kappa, logit_ext, tj, taj)

    vg = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))

    # Multi-init grid
    inits = [
        ("mu0.05_k0.95_r0.7", 0.05, 0.95, 0.7),
        ("mu0.03_k0.98_r0.7", 0.03, 0.98, 0.7),
        ("mu0.05_k0.99_r0.5", 0.05, 0.99, 0.5),
    ]
    best = {"ll": -np.inf}
    for label, mu_i, k_i, r_i in inits:
        p_ = [
            jnp.asarray(np.log(mu_i), jnp.float32),
            jnp.asarray(_logit(k_i), jnp.float32),
            jnp.asarray(_logit(r_i), jnp.float32),
        ]
        m = [jnp.zeros_like(x) for x in p_]
        v = [jnp.zeros_like(x) for x in p_]
        b1, b2, eps = 0.9, 0.999, 1e-8
        best_ll, best_p = -np.inf, [float(x) for x in p_]
        t0 = time.monotonic()
        for step in range(args.n_steps):
            ll, gs = vg(*p_)
            llv = float(ll)
            if llv > best_ll:
                best_ll = llv
                best_p = [float(x) for x in p_]
            for i in range(3):
                m[i] = b1 * m[i] + (1 - b1) * gs[i]
                v[i] = b2 * v[i] + (1 - b2) * (gs[i] * gs[i])
                mh = m[i] / (1 - b1 ** (step + 1))
                vh = v[i] / (1 - b2 ** (step + 1))
                p_[i] = p_[i] + args.lr * mh / (jnp.sqrt(vh) + eps)
            if step % 500 == 0:
                mu_v = float(jnp.exp(p_[0])); kap = float(jax.nn.sigmoid(p_[1]))
                rr = float(jax.nn.sigmoid(p_[2]))
                print(f"  [{label}] step {step:5d}: LL={llv:15,.0f}  μ={mu_v:.5f}  κ={kap:.4f}  r={rr:.4f}  ({time.monotonic()-t0:.1f}s)")
        mu_f = float(np.exp(best_p[0]))
        kap_f = 1 / (1 + float(np.exp(-best_p[1])))
        r_f = 1 / (1 + float(np.exp(-best_p[2])))
        lam_f = kap_f * mu_f
        print(f"  => best LL={best_ll:,.0f}  lam={lam_f:.5f}  mu={mu_f:.5f}  κ={kap_f:.4f}  r={r_f:.4f}\n")
        if best_ll > best["ll"]:
            best = {"ll": best_ll, "init": label,
                    "lam": lam_f, "mu": mu_f, "kappa": kap_f, "r": r_f}

    out = {"meta": {"npz": args.npz, "n_total_trans": int(tj.sum())},
           "best": best}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
