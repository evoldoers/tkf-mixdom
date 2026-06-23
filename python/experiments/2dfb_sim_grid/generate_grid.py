#!/usr/bin/env python3
"""Stage 1 of the model-recovery grid experiment.

For each TKF92 grid point (λ, μ, ext), project to GGI upper-segment
parameters (the configuration that matches the TKF92 dynamics at t=0)
and simulate 20k train + 500 val cherry pairs from THAT GGI generator.
Output a per-grid-point sim directory + a grid_meta.json with all
grid points and their projected GGI params.

Grid: 4 × 3 = 12 points spanning the K=20 mixture's range.
  λ ∈ {0.015, 0.025, 0.040, 0.060}
  ext ∈ {0.55, 0.62, 0.70}
  κ = 0.98 fixed (μ = λ/κ)

Projection TKF92 → GGI (upper segment):
  μ₀ = μ · (1 − ext)
  ρ  = λ / μ  (= κ; pinned to 0.98)
  x  = ext  (upper segment for ext > 1−x_min)
  y  = upper-root of reversibility: y(1−y) = x(1−x)/ρ

This projection matches the TKF92 dynamics at t = 0:
  λ_t(t=0) = λ₀/(1−x) = (μ(1−ext)·ρ)/(1−ext) = λ
  μ_t(t=0) = μ₀/(1−x) = μ
  r_t(t=0) = r_boundary = ext (when x=y, which is the case at ρ=1)
So at t=0 the GGI generator MATCHES the TKF92 generator exactly;
as t grows, r_t drifts toward r_inf < ext, and (λ_t, μ_t) drift too.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "2dfb_sim"))

import jax.random as jr

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.simulate.simulate import (
    simulate_descendant_tkf92)
from simulate_pfam_like import (
    load_pfam_t_distribution, sample_ts_from_empirical,
    y_from_reversibility, sample_ggi_ancestor, ggi_flow_at_t,
    simulate_ggi_cherries)


LAM_GRID = [0.015, 0.025, 0.040, 0.060]
EXT_GRID = [0.55, 0.62, 0.70]
KAPPA = 0.98


def project_tkf92_to_ggi_upper(lam, mu, ext):
    """Project a TKF92 (λ, μ, ext) point into the GGI upper-segment
    parameterisation that MATCHES TKF92 dynamics at t = 0."""
    mu0 = mu * (1.0 - ext)
    rho = lam / max(mu, 1e-30)
    if not (0 < rho < 1):
        raise ValueError(f"GGI projection requires ρ ∈ (0,1), got {rho}")
    x_min = (1.0 - math.sqrt(max(1.0 - rho, 0.0))) / 2.0
    # x = ext, but must lie in upper segment (1-x_min, 1).
    # At ρ=0.98 the lower bound is ≈ 0.5707; ext=0.55 falls just below.
    # In that edge case, snap x up to the segment edge so the projection
    # remains in upper segment.  The mismatch from intended ext is ≤0.02.
    if ext <= 1 - x_min:
        x = (1 - x_min) + 1e-3
    else:
        x = ext
    y = y_from_reversibility(x, rho, "upper")
    lam0 = rho * mu0  # = lam * (1-ext)
    return {"mu0": mu0, "lam0": lam0, "rho": rho, "x": x, "y": y}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precompiled-dir", default="pfam/precompiled")
    p.add_argument("--max-aln-len", type=int, default=256)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--out-dir", default="experiments/2dfb_sim_grid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=600)
    p.add_argument("--t-scale", type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Pfam empirical t-distribution ...", flush=True)
    t_pool = load_pfam_t_distribution(args.precompiled_dir, args.max_aln_len)
    print(f"  {len(t_pool):,} Pfam t-values "
          f"(median {np.median(t_pool):.3f}, mean {t_pool.mean():.3f}, "
          f"max {t_pool.max():.3f})", flush=True)

    Q, pi = rate_matrix_lg()

    # Pre-sample t-vectors once per grid point (so all grid points share
    # the SAME t draws — controls for t-distribution noise).
    rng_t = np.random.default_rng(args.seed)
    ts_train = sample_ts_from_empirical(t_pool, args.n_train, rng_t)
    ts_val   = sample_ts_from_empirical(t_pool, args.n_val,   rng_t)
    if args.t_scale != 1.0:
        ts_train = ts_train * args.t_scale
        ts_val   = ts_val   * args.t_scale
        print(f"  t-scale={args.t_scale}: train mean {ts_train.mean():.3f}, "
              f"val mean {ts_val.mean():.3f}", flush=True)

    grid_pts = []
    for i_lam, lam in enumerate(LAM_GRID):
        for i_ext, ext in enumerate(EXT_GRID):
            mu = lam / KAPPA
            idx = i_lam * len(EXT_GRID) + i_ext
            grid_pts.append({
                "idx": idx, "lam": lam, "mu": mu, "ext": ext, "kappa": KAPPA,
                "ggi": project_tkf92_to_ggi_upper(lam, mu, ext),
            })

    grid_meta = {
        "lam_grid": LAM_GRID, "ext_grid": EXT_GRID, "kappa_fixed": KAPPA,
        "n_train": args.n_train, "n_val": args.n_val, "seed": args.seed,
        "t_scale": args.t_scale,
        "t_stats": {
            "train_mean": float(ts_train.mean()),
            "train_median": float(np.median(ts_train)),
            "train_max": float(ts_train.max()),
            "val_mean": float(ts_val.mean()),
            "val_median": float(np.median(ts_val)),
            "val_max": float(ts_val.max()),
        },
        "grid_points": grid_pts,
    }
    json.dump(grid_meta, open(os.path.join(args.out_dir, "grid_meta.json"), "w"),
              indent=2)

    for pt in grid_pts:
        idx = pt["idx"]
        lam, mu, ext = pt["lam"], pt["mu"], pt["ext"]
        ggi = pt["ggi"]
        rho_geom = ggi["lam0"] * (1 - ggi["x"]) / max(
            ggi["mu0"] * (1 - ggi["y"]), 1e-30)
        mean_anc = rho_geom / max(1 - rho_geom, 1e-30)

        cell_dir = os.path.join(args.out_dir, f"grid{idx:02d}")
        os.makedirs(cell_dir, exist_ok=True)

        print(f"\n=== grid{idx:02d}: TKF92 truth λ={lam} μ={mu:.5f} ext={ext} "
              f"⇒ GGI λ₀={ggi['lam0']:.5f} μ₀={ggi['mu0']:.5f} "
              f"x={ggi['x']:.4f} y={ggi['y']:.4f} "
              f"ρ_GGI={rho_geom:.4f} mean_anc≈{mean_anc:.1f} ===",
              flush=True)

        t0 = time.time()
        train_pairs = simulate_ggi_cherries(
            args.n_train, ggi["lam0"], ggi["mu0"], ggi["x"], ggi["y"],
            ts_train, Q, pi, seed=args.seed + 1000 + idx,
            max_len=args.max_len)
        val_pairs = simulate_ggi_cherries(
            args.n_val, ggi["lam0"], ggi["mu0"], ggi["x"], ggi["y"],
            ts_val, Q, pi, seed=args.seed + 2000 + idx,
            max_len=args.max_len)
        print(f"  {len(train_pairs)} train + {len(val_pairs)} val pairs "
              f"in {time.time()-t0:.1f}s", flush=True)

        pickle.dump(train_pairs,
                    open(os.path.join(cell_dir, "train.pkl"), "wb"))
        pickle.dump(val_pairs,
                    open(os.path.join(cell_dir, "val.pkl"), "wb"))

    print(f"\nAll {len(grid_pts)} grid points generated.")


if __name__ == "__main__":
    main()
