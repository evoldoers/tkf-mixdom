#!/usr/bin/env python3
"""Fit a K-component mixture-of-TKF92 to per-family cherry counts.

Reads ~/tkf-mixdom/python/pfam/cherries_tkf92/PFXXXXX.npz files (one
per Pfam seed family with both an MSA and a tree, produced by
build_tkf92_cherry_counts.py), runs an outer EM that assigns each
family a single hidden component k_n, and an inner TKF92 Baum–Welch
M-step on the responsibility-weighted aggregate counts (HR strategy
#2 for the substitution model + κ-quadratic for the indel rates).

Output: a train_pfam-loadable .npz with K MixDom1 domains (one per
component), suitable as a `train_pfam.py --checkpoint <path>` warm
start.

Usage:
    uv run python fit_tkf92_mixture.py --K 3 \\
        --cherries-dir pfam/cherries_tkf92 \\
        --out pfam/tkf92_mixture_K3.npz

To restrict to the v1 train split (so val/test families don't leak
into the fit):
    uv run python fit_tkf92_mixture.py --K 3 \\
        --split-file ~/bio-datasets/data/pfam/seed/splits/v1.json \\
        --split train \\
        --out pfam/tkf92_mixture_K3_train.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# Default to CPU — the model is small enough that GPU contention with
# other training runs (e.g. step 6 SVI-BW) is the dominant cost.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from tkfmixdom.jax.distill.tkf92_mixture import (  # noqa: E402
    fit_mixture, load_cherry_stack, to_mixdom1_checkpoint,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, required=True,
                   help="Number of mixture components")
    p.add_argument("--cherries-dir", type=str, default="pfam/cherries_tkf92",
                   help="Directory of per-family .npz cherry-count files")
    p.add_argument("--out", type=str, required=True,
                   help="Output .npz (MixDom1-loadable: train_pfam --checkpoint)")
    p.add_argument("--split-file", type=str, default=None,
                   help="Optional JSON split file with train/val/test family lists")
    p.add_argument("--split", type=str, default=None,
                   choices=["train", "val", "test", "all"],
                   help="Which split to fit (requires --split-file). all = use everything")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for symmetry-breaking init")
    p.add_argument("--outer-n-iter", type=int, default=50,
                   help="Max outer EM iterations")
    p.add_argument("--outer-rel-tol", type=float, default=1e-5,
                   help="Outer relative-LL tolerance for early stop")
    p.add_argument("--inner-n-iter", type=int, default=30,
                   help="Max inner BW iterations per component per outer step")
    p.add_argument("--inner-rel-tol", type=float, default=1e-4,
                   help="Inner relative-LL tolerance")
    p.add_argument("--init-perturb", type=float, default=0.05,
                   help="Symmetry-breaking perturbation amplitude on init θ_k")
    p.add_argument("--base-lam", type=float, default=0.02,
                   help="Initial mean λ across components")
    p.add_argument("--base-mu", type=float, default=0.025,
                   help="Initial mean μ across components")
    p.add_argument("--base-r", type=float, default=0.5,
                   help="Initial mean fragment-extension probability")
    p.add_argument("--main-ins", type=float, default=0.014,
                   help="Top-level main_ins for output MixDom1 ckpt (default ~ d3f1)")
    p.add_argument("--main-del", type=float, default=0.015,
                   help="Top-level main_del for output MixDom1 ckpt")
    p.add_argument("--pi-pseudo", type=float, default=1.0,
                   help="LG-π pseudocount weight (Dirichlet concentration)")
    p.add_argument("--S-pseudo", type=float, default=0.0,
                   help="LG-S pseudocount weight")
    p.add_argument("--ext-prior-alpha", type=float, default=2.0,
                   help="Beta(α,β) prior on r: α (default 2)")
    p.add_argument("--ext-prior-beta", type=float, default=3.0,
                   help="Beta(α,β) prior on r: β (default 3)")
    p.add_argument("--prior-alpha-lam", type=float, default=2.0,
                   help="Gamma(α_λ,β) prior shape for λ")
    p.add_argument("--prior-alpha-mu", type=float, default=2.0,
                   help="Gamma(α_μ,β) prior shape for μ")
    p.add_argument("--prior-beta", type=float, default=10.0,
                   help="Gamma rate β for both λ and μ priors")
    args = p.parse_args()

    # ---- Resolve which families to fit ----
    families = None
    if args.split_file is not None:
        with open(os.path.expanduser(args.split_file)) as f:
            sd = json.load(f)
        if args.split is None or args.split == "all":
            families = sorted(set().union(*[set(sd.get(s, []))
                                            for s in ("train", "val", "test")]))
        else:
            families = sorted(sd.get(args.split, []))
        print(f"[load] split={args.split or 'all'} → {len(families)} families requested")

    print(f"[load] cherry stack from {args.cherries_dir} ...")
    t0 = time.monotonic()
    stack = load_cherry_stack(args.cherries_dir, families=families)
    dt_load = time.monotonic() - t0
    print(f"[load] N={len(stack.family_ids)} families, T={stack.tau_centers.shape[0]} bins, "
          f"total cherries={int(stack.n_pairs.sum())} ({dt_load:.1f}s)")

    print(f"[fit ] K={args.K}, seed={args.seed}, "
          f"outer ≤ {args.outer_n_iter}, inner ≤ {args.inner_n_iter}")
    t0 = time.monotonic()
    thetas, log_mix, history = fit_mixture(
        stack, K=args.K,
        seed=args.seed,
        outer_n_iter_max=args.outer_n_iter,
        outer_rel_tol=args.outer_rel_tol,
        inner_n_iter_max=args.inner_n_iter,
        inner_rel_tol=args.inner_rel_tol,
        perturb=args.init_perturb,
        base_lam=args.base_lam, base_mu=args.base_mu, base_r=args.base_r,
        pi_pseudo=args.pi_pseudo, S_pseudo=args.S_pseudo,
        ext_prior_alpha=args.ext_prior_alpha, ext_prior_beta=args.ext_prior_beta,
        prior_alpha_lam=args.prior_alpha_lam,
        prior_alpha_mu=args.prior_alpha_mu,
        prior_beta=args.prior_beta,
        log_fn=print)
    dt_fit = time.monotonic() - t0
    print(f"[fit ] done in {dt_fit:.1f}s, final total_ll = {history[-1]['total_ll']:.4f}")

    # ---- Print final per-component params ----
    print("[done] component summary:")
    mix = np.exp(np.asarray(log_mix))
    mix /= mix.sum()
    for k, (lam, mu, r, S, pi) in enumerate(thetas):
        print(f"  k={k}: weight={mix[k]:.3f}, λ={lam:.4g}, μ={mu:.4g}, r={r:.3f}, "
              f"π_AA mean L1 from LG = {float(np.abs(np.asarray(pi) - 1/20).sum()):.3f}")

    # ---- Save MixDom1-compatible ckpt ----
    config = {
        "trainer": "fit_tkf92_mixture",
        "K": int(args.K),
        "seed": int(args.seed),
        "split": args.split,
        "split_file": args.split_file,
        "n_families": int(len(stack.family_ids)),
        "n_pairs": int(stack.n_pairs.sum()),
        "n_dom": int(args.K),  # train_pfam needs this in _config
        "n_frag": 1,
        "main_ins": float(args.main_ins),
        "main_del": float(args.main_del),
        "init_perturb": float(args.init_perturb),
        "base_lam": float(args.base_lam),
        "base_mu": float(args.base_mu),
        "base_r": float(args.base_r),
        "outer_iters_run": int(history[-1]["iter"]),
        "inner_n_iter": int(args.inner_n_iter),
        "final_total_ll": float(history[-1]["total_ll"]),
        "history": history,
    }
    out = to_mixdom1_checkpoint(
        thetas, log_mix,
        main_ins=args.main_ins, main_del=args.main_del,
        n_frag=1, em_iter=0, config=config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"[save] {out_path} (size {out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
