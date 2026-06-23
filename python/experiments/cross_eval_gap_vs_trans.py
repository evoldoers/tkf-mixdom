#!/usr/bin/env python3
"""Cross-evaluate TKF92 params trained on gap counts vs trained on 5x5
transition counts, scoring each on BOTH val representations.

Hypothesis: gap-trained params score higher on gap LL (which they were
optimised on); transition-trained params score higher on transition LL.
"""
import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.dirname(THIS_DIR))

from fit_via_gap_counts import joint_ll_tkf92_constant
from fit_ggi_cherryml import tkf92_trans_full
from fit_pfam_gap_counts import _fold_clip

S, M, I, D, E = 0, 1, 2, 3, 4


def joint_ll_trans_tkf92(log_del, logit_kappa, logit_ext, trans_counts_j, tau_j):
    """sum_{tau, src, dst} trans_counts[tau, src, dst] * log T[src, dst].

    Joint LL summed using the 5x5 transition representation (NOT summed
    over I/D orderings — keeps every M->I, I->I, etc. as separate counts).
    """
    mu_ = jnp.exp(log_del)
    kappa_ = jax.nn.sigmoid(logit_kappa)
    lam_ = kappa_ * mu_
    r_ = jax.nn.sigmoid(logit_ext)

    def per_tau(ti):
        T = tkf92_trans_full(lam_, mu_, tau_j[ti], r_)
        # Clamp first then log — guards against small negative T from
        # numerical noise in tkf_gamma (occasionally produces T[D,I] = -3e-4
        # at small tau & κ near 1).
        log_T = jnp.log(jnp.maximum(T, 1e-30))
        return jnp.sum(trans_counts_j[ti] * log_T)

    n_tau = tau_j.shape[0]
    return jnp.sum(jax.vmap(per_tau)(jnp.arange(n_tau)))


def _logit(p): return float(np.log(p / max(1 - p, 1e-30)))


def score(lam, mu, kappa, r, gap_j, trans_j, tau_j, Lmax):
    log_del = jnp.asarray(np.log(mu), jnp.float32)
    logit_kappa = jnp.asarray(_logit(kappa), jnp.float32)
    logit_ext = jnp.asarray(_logit(r), jnp.float32)
    gap_ll = float(joint_ll_tkf92_constant(log_del, logit_kappa, logit_ext,
                                            gap_j, trans_j, tau_j, Lmax))
    trans_ll = float(joint_ll_trans_tkf92(log_del, logit_kappa, logit_ext,
                                            trans_j, tau_j))
    return gap_ll, trans_ll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-npz", default="pfam_gap_counts_val.npz")
    parser.add_argument("--Lmax-fit", type=int, default=20)
    parser.add_argument("--symmetrize", action="store_true", default=True)
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    d = np.load(args.val_npz)
    g = _fold_clip(d["gap_counts"].astype(np.float32), args.Lmax_fit)
    if args.symmetrize:
        g = ((g + g.transpose(0, 1, 3, 2)) / 2.0).astype(np.float32)
    t = d["trans_counts"].astype(np.float32)
    tau = d["tau_centers"].astype(np.float32)
    gj = jnp.asarray(g)
    tj = jnp.asarray(t)
    taj = jnp.asarray(tau)
    n_cherries = int(d["n_cherries_per_bin"].sum())
    print(f"  val cherries: {n_cherries:,}")
    print(f"  gap_counts shape: {gj.shape}  trans_counts shape: {tj.shape}")
    print(f"  trans_counts total: {int(tj.sum()):,}")

    # Two param sets to cross-compare:
    # Load fitted params from JSON files for reproducibility.
    # NOTE: tkf92_fitted_params_train.json was fit on seed_counts_train.npz
    # (OLD cherry extraction, 585k cherries / 21,667 families) — comparing to
    # gap-trained from pfam_gap_counts.npz (NEW cherry extraction, 364k /
    # 19,850) is unfair: different data sets.
    with open("experiments/tkf92_fitted_params_train.json") as f:
        tp_old = json.load(f)
    old_trans_lam, old_trans_mu = tp_old["ins_rate"], tp_old["del_rate"]
    old_trans_r, old_trans_kappa = tp_old["ext_rate"], tp_old["kappa"]
    # FAIR comparison: trans-trained on the SAME cherry extraction as the gap fit
    with open("experiments/tkf92_fitted_on_gap_trans.json") as f:
        tp = json.load(f)["best"]
    trans_lam = tp["lam"]; trans_mu = tp["mu"]
    trans_r = tp["r"]; trans_kappa = tp["kappa"]
    # Gap-trained joint-native (init #1 of currently-running fit)
    gap_lam, gap_mu, gap_kappa, gap_r = 0.03137, 0.03200, 0.9804, 0.6624
    # Conditional gap-trained (kappa=1 exact axis)
    cond_lam, cond_mu, cond_kappa, cond_r = 0.05261, 0.05261, 1.0000, 0.8107

    methods = [
        ("gap-trained (joint-native)", gap_lam, gap_mu, gap_kappa, gap_r),
        ("trans-trained (same chry)",  trans_lam, trans_mu, trans_kappa, trans_r),
        ("trans-trained (OLD chry)",   old_trans_lam, old_trans_mu, old_trans_kappa, old_trans_r),
        ("gap-trained (conditional)",  cond_lam, cond_mu, cond_kappa, cond_r),
    ]

    print(f"\n{'method':30s}  {'gap-LL/cherry':>15s}  {'trans-LL/cherry':>17s}  "
          f"{'gap-LL':>14s}  {'trans-LL':>14s}")
    print("-" * 100)
    results = {}
    for name, lam, mu, kappa, r in methods:
        gap_ll, trans_ll = score(lam, mu, kappa, r, gj, tj, taj, args.Lmax_fit)
        # Per-cherry normalization
        gap_per_cherry = gap_ll / max(n_cherries, 1)
        trans_per_cherry = trans_ll / max(n_cherries, 1)
        print(f"{name:30s}  {gap_per_cherry:>15.4f}  {trans_per_cherry:>17.4f}  "
              f"{gap_ll:>14,.0f}  {trans_ll:>14,.0f}")
        results[name] = {
            "lam": lam, "mu": mu, "kappa": kappa, "r": r,
            "val_gap_ll": gap_ll, "val_trans_ll": trans_ll,
            "val_gap_per_cherry": gap_per_cherry,
            "val_trans_per_cherry": trans_per_cherry,
        }

    print()
    print("Hypothesis check:")
    gap_trained = "gap-trained (joint-native)"
    trans_trained = "trans-count-trained"
    trans_trained = "trans-trained (same chry)"
    print(f"  gap-LL:    {gap_trained:30s} vs {trans_trained:30s}  "
          f"delta = {results[gap_trained]['val_gap_ll'] - results[trans_trained]['val_gap_ll']:+,.0f}  "
          f"({(results[gap_trained]['val_gap_ll'] - results[trans_trained]['val_gap_ll'])/n_cherries:+.4f} /cherry)")
    print(f"  trans-LL:  {gap_trained:30s} vs {trans_trained:30s}  "
          f"delta = {results[gap_trained]['val_trans_ll'] - results[trans_trained]['val_trans_ll']:+,.0f}  "
          f"({(results[gap_trained]['val_trans_ll'] - results[trans_trained]['val_trans_ll'])/n_cherries:+.4f} /cherry)")

    with open("experiments/pfam_gap_fit/cross_eval_gap_vs_trans.json", "w") as f:
        json.dump({"n_val_cherries": n_cherries, "results": results}, f, indent=2)
    print("\nWrote experiments/pfam_gap_fit/cross_eval_gap_vs_trans.json")


if __name__ == "__main__":
    main()
