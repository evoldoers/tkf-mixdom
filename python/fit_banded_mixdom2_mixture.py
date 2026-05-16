#!/usr/bin/env python3
"""Fit a K-component banded MixDom2 mixture to per-family Maraschino counts.

Phase 2 of the FragStart/FragMid/FragEnd pipeline (see
`tkf/substitution-mstep.tex` sec:mstep-tied-pi). Reads per-family
Maraschino .marcounts.npz files (produced by
``maraschino.py count --out-suffix .marcounts.npz``), runs an outer EM
that hard-assigns each family to a single hidden component k, and an
inner Adam loop on the banded MixDom2 cherry log-likelihood per
component on the responsibility-aggregated counts.

Output: a `train_pfam`-loadable .npz with K MixDom2 banded "domains"
(one per component), 3K site classes (one per (k, fragchar)), and
block-diagonal classdist. Suitable as a `train_pfam.py --init <path>`
warm start.

Why hard EM (not soft): order-1 Maraschino correlations alone are weak
at resolving mixture components; soft EM tends to collapse all
components onto the same mode. Hard assignment + symmetry-breaking
inits + per-iter orphan rescue keeps the K components separated.

Usage:
    # 1. Build per-family Maraschino counts
    uv run python maraschino.py count \\
        --msa-dir ~/bio-datasets/data/pfam/seed/ \\
        --out-suffix .marcounts.npz \\
        --n-tau-bins 8

    # 2. Fit K=3 banded mixture
    uv run python fit_banded_mixdom2_mixture.py --K 3 \\
        --counts-dir ~/bio-datasets/data/pfam/seed/ \\
        --counts-suffix .marcounts.npz \\
        --out pfam/banded_mixture_K3.npz

    # 3. Warm-start train_pfam
    uv run python train_pfam.py --init pfam/banded_mixture_K3.npz \\
        --n-dom 3 --n-frag 3 --n-classes 9 --banded-frag-init ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# Default to CPU — the model is small (small K, small T) and CPU avoids
# contention with concurrent training jobs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from tkfmixdom.jax.distill.banded_mixture import (  # noqa: E402
    fit_banded_mixture, load_per_family_marcounts, to_mixdom2_checkpoint,
)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, required=True,
                   help="Number of mixture components (= number of "
                   "output MixDom2 'domains').")
    p.add_argument("--counts-dir", type=str, required=True,
                   help="Directory of per-family .marcounts.npz files "
                   "(produced by `maraschino.py count --out-suffix`).")
    p.add_argument("--counts-suffix", type=str, default=".marcounts.npz",
                   help="Filename suffix for per-family counts "
                   "(default: .marcounts.npz).")
    p.add_argument("--out", type=str, required=True,
                   help="Output .npz (train_pfam-loadable: --init <path>).")
    p.add_argument("--split-file", type=str, default=None,
                   help="Optional JSON split file (train/val/test).")
    p.add_argument("--split", type=str, default=None,
                   choices=["train", "val", "test", "all"],
                   help="Which split to fit (requires --split-file).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for symmetry-breaking init.")
    # Outer EM
    p.add_argument("--outer-n-iter", type=int, default=30,
                   help="Max outer EM iterations.")
    p.add_argument("--outer-rel-tol", type=float, default=1e-5,
                   help="Outer relative-LL tolerance for early stop.")
    # Inner Adam
    p.add_argument("--inner-n-steps", type=int, default=200,
                   help="Adam steps per inner per-component fit "
                   "per outer iter.")
    p.add_argument("--inner-lr", type=float, default=1e-2,
                   help="Adam learning rate for inner fit.")
    # Banded init
    p.add_argument("--p-ext", type=float, default=0.6,
                   help="Initial extension probability for banded init.")
    p.add_argument("--init-ins", type=float, default=0.014,
                   help="Initial top-level / dom indel rate λ.")
    p.add_argument("--init-del", type=float, default=0.015,
                   help="Initial top-level / dom indel rate μ.")
    p.add_argument("--init-perturb", type=float, default=0.05,
                   help="(unused; reserved for future symmetry-breaking).")
    # ----- Per-class S/π freezing (mirrors maraschino fit) -----
    p.add_argument("--rescale-class-S-only", action='store_true',
                   help="Hold (S^c, π^c) shape fixed at LG init; only "
                   "per-class log_class_sigma_c is free per component. "
                   "Implies --freeze-class-S-shape and --freeze-class-pi.")
    p.add_argument("--freeze-class-S-shape", action='store_true',
                   help="Hold class_S_exch shape at LG init; let per-class "
                   "log_class_sigma_c (S_c = exp(σ) · S_LG) and π_c vary.")
    p.add_argument("--freeze-class-pi", action='store_true',
                   help="Hold class_pis at init across the inner Adam loop.")
    p.add_argument("--freeze-main-rates", action='store_true',
                   help="Hold (main_ins, main_del) at init across the inner "
                   "Adam loop. Each component is conceptually a single-domain "
                   "model (D=1), so the top-level TKF91 should not absorb "
                   "indel signal. Recommended.")
    p.add_argument("--family-batch-size", type=int, default=256,
                   help="E-step vmap batch size over families (default: 256).")
    p.add_argument("--max-families", type=int, default=None,
                   help="Limit to a random subset of N training families "
                   "(uses --seed). Useful for sanity runs.")
    # Output
    p.add_argument("--main-ins", type=float, default=0.014,
                   help="Output top-level main_ins (for train_pfam ckpt).")
    p.add_argument("--main-del", type=float, default=0.015,
                   help="Output top-level main_del (for train_pfam ckpt).")
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
        print(f"[load] split={args.split or 'all'} → "
              f"{len(families)} families requested")

    # If --max-families is set, pre-sample the family ID list BEFORE
    # loading any tensors. With ~7 MB/family uncompressed, loading all
    # 21k train families can hit ~150 GB RAM; pre-sampling avoids this.
    if args.max_families is not None and families is not None:
        if args.max_families < len(families):
            rng = np.random.RandomState(args.seed)
            # Filter to families that actually have counts files first.
            from pathlib import Path as _Path
            counted = {p.name.replace(args.counts_suffix, '')
                       for p in _Path(args.counts_dir).glob(
                           f"*{args.counts_suffix}")}
            fams_with_counts = sorted([f for f in families if f in counted])
            if len(fams_with_counts) < args.max_families:
                args.max_families = len(fams_with_counts)
            sample = sorted(rng.choice(fams_with_counts, args.max_families,
                                         replace=False))
            print(f"[load] pre-subsampled to {len(sample)} families "
                  f"(seed={args.seed}; from {len(fams_with_counts)} train "
                  f"families with counts)")
            families = sample

    print(f"[load] per-family counts from {args.counts_dir} "
          f"(suffix={args.counts_suffix}) ...")
    t0 = time.monotonic()
    fams, tau_centers = load_per_family_marcounts(
        args.counts_dir, families=families, suffix=args.counts_suffix)
    dt_load = time.monotonic() - t0
    total_pairs = sum(f.n_pairs for f in fams)
    print(f"[load] N={len(fams)} families, T={tau_centers.shape[0]} bins, "
          f"total cherries={total_pairs} ({dt_load:.1f}s)")

    # Post-load fallback subsample (only used when families list was
    # not pre-filterable, e.g. when --split-file was not supplied).
    if args.max_families is not None and args.max_families < len(fams):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(fams), size=args.max_families, replace=False)
        fams = [fams[int(i)] for i in idx]
        total_pairs = sum(f.n_pairs for f in fams)
        print(f"[load] subsampled to {len(fams)} families "
              f"(seed={args.seed}, total cherries={total_pairs})")

    # ---- Resolve umbrella flag ----
    if args.rescale_class_S_only:
        args.freeze_class_S_shape = True
        args.freeze_class_pi = True

    # ---- Fit ----
    print(f"[fit ] K={args.K}, seed={args.seed}, "
          f"outer ≤ {args.outer_n_iter}, inner Adam {args.inner_n_steps} "
          f"steps × lr={args.inner_lr}")
    if args.freeze_class_S_shape:
        print("       freeze class_S_shape: only log_class_sigma per "
              "component varies for substitution rate")
    if args.freeze_class_pi:
        print("       freeze class_pis: held at init")
    if args.freeze_main_rates:
        print("       freeze main_rates: top-level TKF91 (main_ins, "
              "main_del) held at init (recommended for K-component "
              "single-domain mixture).")
    t0 = time.monotonic()
    components, mix_weights, history = fit_banded_mixture(
        fams, tau_centers, K=args.K,
        seed=args.seed,
        outer_n_iter=args.outer_n_iter,
        outer_rel_tol=args.outer_rel_tol,
        inner_n_steps=args.inner_n_steps,
        inner_lr=args.inner_lr,
        p_ext=args.p_ext,
        init_ins=args.init_ins, init_del=args.init_del,
        init_perturb=args.init_perturb,
        freeze_class_S_shape=args.freeze_class_S_shape,
        freeze_class_pi=args.freeze_class_pi,
        freeze_main_rates=args.freeze_main_rates,
        family_batch_size=args.family_batch_size,
        log_fn=print)
    dt_fit = time.monotonic() - t0
    final_ll = history[-1]['total_ll'] if history else float('nan')
    print(f"[fit ] done in {dt_fit:.1f}s, final total_ll = {final_ll:.4f}")

    # ---- Per-component summary ----
    print("[done] component summary:")
    for k, comp in enumerate(components):
        ext = np.asarray(comp['ext_rates'])[0]
        lam = float(np.asarray(comp['dom_ins'])[0])
        mu = float(np.asarray(comp['dom_del'])[0])
        # Approximate "mean extension" diagnostic: probability of any
        # forward transition from FragStart row.
        p_extend_from_start = float(ext[0, 1] + ext[0, 2])
        print(f"  k={k}: weight={mix_weights[k]:.3f}, "
              f"λ={lam:.4g}, μ={mu:.4g}, "
              f"p(extend|FragStart)={p_extend_from_start:.3f}, "
              f"ext[FragStart]→FragEnd={ext[0, 2]:.3f}, "
              f"ext[FragMid]→FragEnd={ext[1, 2]:.3f}")

    # ---- Save train_pfam-compatible ckpt ----
    config = {
        "trainer": "fit_banded_mixdom2_mixture",
        "K": int(args.K),
        "seed": int(args.seed),
        "split": args.split,
        "split_file": args.split_file,
        "n_families": int(len(fams)),
        "n_pairs": int(total_pairs),
        "n_dom": int(args.K),
        "n_frag": 3,
        "n_classes": 3 * int(args.K),
        "p_ext": float(args.p_ext),
        "main_ins": float(args.main_ins),
        "main_del": float(args.main_del),
        "outer_iters_run": int(history[-1]["iter"]) if history else 0,
        "inner_n_steps": int(args.inner_n_steps),
        "inner_lr": float(args.inner_lr),
        "final_total_ll": final_ll,
        "history": history,
        "banded_frag_init": True,
    }
    out = to_mixdom2_checkpoint(
        components, mix_weights,
        main_ins=args.main_ins, main_del=args.main_del,
        t=1.0, em_iter=0, config=config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"[save] {out_path} (size {out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
