#!/usr/bin/env python3
"""Simulate Pfam-like cherry pair datasets from two ground-truth models:

  TKF92 ground truth: λ=0.058, μ=0.060, ext=0.75   (Adam-tkf92 long converged)
  GGI   ground truth: λ₀=0.014, μ₀=0.014, x=0.81, y=0.81
                       (Adam-GGI native upper long2400 converged, ρ≈0.99)

For each model, simulate 20,000 train + 500 val cherry pairs.  t-values
are sampled from the EMPIRICAL distribution of Pfam cherry t-values
(uniformly over the precompiled corpus's t-values, not stratified by
family) so the t-diversity matches what real training has seen.

Outputs:
  experiments/2dfb_sim/sim_tkf92/{train,val}.pkl   — list of (x_int, y_int, t)
  experiments/2dfb_sim/sim_ggi/{train,val}.pkl     — same format
  experiments/2dfb_sim/sim_meta.json                — generator params + summary

These pickles are loaded by run_tkf92_2dfb_pfam.py via --sim-pairs-file.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import jax.random as jr
import jax.numpy as jnp

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.simulate.simulate import (
    simulate_stationary_sequence, simulate_descendant_tkf92)


def load_pfam_t_distribution(precompiled_dir, max_aln_len=256):
    """Read every Pfam cherry's t-value out of the precompiled corpus."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from train_pfam import PrecompiledPairSource
    src = PrecompiledPairSource(precompiled_dir,
                                  max_alignment_len=max_aln_len)
    all_decoded = src._decode_all()
    ts = np.array([float(item[5]) for item in all_decoded])
    return ts


def sample_ts_from_empirical(empirical_ts, n, rng):
    """Sample n t-values uniformly from the empirical distribution."""
    idx = rng.integers(0, empirical_ts.shape[0], size=n)
    return empirical_ts[idx]


def ggi_flow_at_t(lam0, mu0, x, y, t):
    """Compute (λ_t, μ_t, r_t) from GGI flow at branch length t.

    FIXED-RATE form (paper's recommended closed-form GGI → TKF92
    surrogate) per wideboy_to_lambda.md round 2 (2026-06-03).  λ_t
    and μ_t held constant at their boundary values; only r(t) evolves.
    """
    num = lam0 * y * (1 - x) + mu0 * x * (1 - y)
    den = lam0 * (1 - y) + mu0 * (1 - x)
    r_boundary = num / max(den, 1e-30)
    r_inf = r_boundary / (2 - r_boundary)
    k = (lam0 + mu0) * (2 - r_boundary) / max(1 - r_boundary, 1e-30)
    r_t = r_inf + (r_boundary - r_inf) * np.exp(-k * t)
    one_minus_r0 = max(1 - r_boundary, 1e-30)
    lam_t = lam0 / one_minus_r0
    mu_t = mu0 / one_minus_r0
    return lam_t, mu_t, r_t


def y_from_reversibility(x, rho, segment="upper"):
    """Solve y(1-y) = x(1-x)/ρ for y, picking the segment-symmetric root.

    Lower segment: x < x_min ⇒ y ∈ (0, ½) (lower root).
    Upper segment: x > 1-x_min ⇒ y ∈ (½, 1) (upper root).
    """
    q = x * (1.0 - x) / max(rho, 1e-30)
    disc = max(1.0 - 4.0 * q, 0.0)
    sqrt_disc = float(np.sqrt(disc))
    if segment == "lower":
        return (1.0 - sqrt_disc) / 2.0
    elif segment == "upper":
        return (1.0 + sqrt_disc) / 2.0
    raise ValueError(f"unknown segment {segment!r}")


def sample_ggi_ancestor(lam0, mu0, x, y, pi, rng):
    """Sample an ancestor sequence from the GGI native geometric stationary.

    L ~ Geom(1-ρ_GGI) where ρ_GGI = λ₀(1-x)/(μ₀(1-y));
    residues iid from pi.
    """
    rho = lam0 * (1 - x) / max(mu0 * (1 - y), 1e-30)
    rho = min(max(rho, 1e-30), 1.0 - 1e-9)
    # np.random.geometric returns L≥1; for L≥0 geometric, subtract 1.
    L = rng.geometric(p=1 - rho) - 1
    if L == 0:
        return np.zeros(0, dtype=np.int32)
    chars = rng.choice(pi.shape[0], size=L, p=pi)
    return chars.astype(np.int32)


def simulate_tkf92_cherries(n_pairs, lam, mu, ext, ts, Q, pi, seed,
                              max_len=600, log_fn=print):
    """Generate n_pairs cherries from TKF92(lam, mu, ext) with the
    given (length-n) t-values.

    Returns list of (x_int, y_int, t).  ancestor and descendant trimmed
    to max_len each (cherries longer than that are RESIMULATED with a
    fresh key so the dataset stays at n_pairs).
    """
    pi_np = np.asarray(pi)
    sub_matrix_at_t = {}  # cache by rounded t to speed up
    out = []
    rng_key = jr.PRNGKey(seed)
    rejections = 0
    t0 = time.time()
    for i in range(n_pairs):
        t = float(ts[i])
        # Cache substitution matrix by rounded t
        t_key = round(t, 4)
        if t_key not in sub_matrix_at_t:
            sub_matrix_at_t[t_key] = np.asarray(transition_matrix(Q, t_key))
        sub_matrix = sub_matrix_at_t[t_key]

        # Try until we get a pair within max_len
        attempt = 0
        while True:
            rng_key, sub_key = jr.split(rng_key)
            ancestor = simulate_stationary_sequence(
                sub_key, lam, mu, pi_np, max_len, ext=ext)
            if len(ancestor) == 0 or len(ancestor) > max_len:
                rejections += 1
                attempt += 1
                if attempt > 10:
                    raise RuntimeError(
                        f"TKF92 sim: 10 consecutive ancestor-length rejections "
                        f"(L0={len(ancestor)}); params may be misspecified.")
                continue
            rng_key, sub_key = jr.split(rng_key)
            descendant, _alignment = simulate_descendant_tkf92(
                sub_key, ancestor, lam, mu, t, ext, sub_matrix, pi_np)
            if len(descendant) > max_len or len(descendant) == 0:
                rejections += 1
                attempt += 1
                if attempt > 10:
                    raise RuntimeError(
                        f"TKF92 sim: 10 consecutive desc rejections at t={t}.")
                continue
            break
        out.append((np.asarray(ancestor, dtype=np.int32),
                    np.asarray(descendant, dtype=np.int32), t))
        if (i + 1) % 1000 == 0:
            log_fn(f"  TKF92 sim: {i+1}/{n_pairs} done "
                   f"(elapsed {time.time()-t0:.1f}s, "
                   f"rej {rejections})", flush=True)
    log_fn(f"  TKF92 sim done: {n_pairs} pairs, {rejections} rejections, "
           f"{time.time()-t0:.1f}s", flush=True)
    return out


def simulate_ggi_cherries(n_pairs, lam0, mu0, x, y, ts, Q, pi, seed,
                            max_len=600, log_fn=print):
    """Generate n_pairs cherries from GGI(lam0, mu0, x, y) — ancestor
    from GGI native geometric stationary, descendant from TKF92 at the
    per-pair flowed (λ_t, μ_t, r_t)."""
    pi_np = np.asarray(pi)
    out = []
    np_rng = np.random.default_rng(seed)
    rng_key = jr.PRNGKey(seed + 9999)
    sub_matrix_cache = {}
    t0 = time.time()
    rejections = 0

    for i in range(n_pairs):
        t = float(ts[i])
        lam_t, mu_t, r_t = ggi_flow_at_t(lam0, mu0, x, y, t)
        t_key = round(t, 4)
        if t_key not in sub_matrix_cache:
            sub_matrix_cache[t_key] = np.asarray(transition_matrix(Q, t_key))
        sub_matrix = sub_matrix_cache[t_key]

        attempt = 0
        while True:
            ancestor = sample_ggi_ancestor(lam0, mu0, x, y, pi_np, np_rng)
            if len(ancestor) == 0 or len(ancestor) > max_len:
                rejections += 1
                attempt += 1
                if attempt > 10:
                    raise RuntimeError(
                        f"GGI sim: 10 consecutive ancestor rejections "
                        f"(L0={len(ancestor)}); params may be misspecified.")
                continue
            rng_key, sub_key = jr.split(rng_key)
            descendant, _alignment = simulate_descendant_tkf92(
                sub_key, ancestor, lam_t, mu_t, t, r_t, sub_matrix, pi_np)
            if len(descendant) > max_len or len(descendant) == 0:
                rejections += 1
                attempt += 1
                if attempt > 10:
                    raise RuntimeError(
                        f"GGI sim: 10 consecutive desc rejections at t={t}.")
                continue
            break
        out.append((np.asarray(ancestor, dtype=np.int32),
                    np.asarray(descendant, dtype=np.int32), t))
        if (i + 1) % 1000 == 0:
            log_fn(f"  GGI sim: {i+1}/{n_pairs} done "
                   f"(elapsed {time.time()-t0:.1f}s, "
                   f"rej {rejections})", flush=True)
    log_fn(f"  GGI sim done: {n_pairs} pairs, {rejections} rejections, "
           f"{time.time()-t0:.1f}s", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--precompiled-dir", default="pfam/precompiled")
    p.add_argument("--max-aln-len", type=int, default=256)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--out-dir", default="experiments/2dfb_sim")
    p.add_argument("--seed", type=int, default=42)
    # TKF92 gen params (Adam-tkf92 long converged)
    p.add_argument("--tkf92-lam", type=float, default=0.058)
    p.add_argument("--tkf92-mu", type=float, default=0.060)
    p.add_argument("--tkf92-ext", type=float, default=0.75)
    # GGI gen params: small perturbation from the Adam-GGI native upper
    # long2400 MLE (λ₀=0.014, μ₀=0.014, x=y=0.81) which sits exactly at
    # the ρ_GGI=1 degenerate boundary.  λ₀=0.01393, μ₀=0.014 gives
    # ρ=0.995, and reversibility (upper segment) gives y≈0.8087,
    # ρ_GGI≈0.988, mean ancestor length ≈ 83.
    p.add_argument("--ggi-lam0", type=float, default=0.01393)
    p.add_argument("--ggi-mu0", type=float, default=0.014)
    p.add_argument("--ggi-x", type=float, default=0.81)
    p.add_argument("--ggi-segment", choices=["lower", "upper"], default="upper",
                    help="GGI x-feasibility segment (determines which y-root "
                         "of the reversibility constraint to use).")
    p.add_argument("--t-scale", type=float, default=1.0,
                    help="Multiplicative scaling applied to ALL sampled t "
                         "values.  Default 1.0 = match Pfam.  >1 rolls out the "
                         "evolution time, making GGI's t-flow more distinct "
                         "from a constant-rate TKF92.")
    p.add_argument("--max-len", type=int, default=600,
                    help="Reject simulated cherries longer than this on either "
                         "side. Set generously since TKF92 long-tail lengths "
                         "would otherwise force resampling.")
    args = p.parse_args()

    os.makedirs(os.path.join(args.out_dir, "sim_tkf92"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "sim_ggi"), exist_ok=True)

    print("Loading Pfam empirical t-distribution ...", flush=True)
    t_pool = load_pfam_t_distribution(args.precompiled_dir, args.max_aln_len)
    print(f"  {len(t_pool):,} Pfam t-values "
          f"(min={t_pool.min():.4f}, max={t_pool.max():.4f}, "
          f"mean={t_pool.mean():.4f}, median={np.median(t_pool):.4f})",
          flush=True)

    Q, pi = rate_matrix_lg()
    rng_t = np.random.default_rng(args.seed)
    ts_train = sample_ts_from_empirical(t_pool, args.n_train, rng_t)
    ts_val = sample_ts_from_empirical(t_pool, args.n_val, rng_t)
    if args.t_scale != 1.0:
        ts_train = ts_train * args.t_scale
        ts_val = ts_val * args.t_scale
        print(f"  t-scale={args.t_scale}: scaled train mean t = "
              f"{ts_train.mean():.4f}, val mean t = {ts_val.mean():.4f}",
              flush=True)
    print(f"  sampled t-vectors: train {ts_train.shape}, val {ts_val.shape}",
          flush=True)

    # TKF92 dataset
    print(f"\n=== TKF92 ground truth: "
          f"λ={args.tkf92_lam} μ={args.tkf92_mu} ext={args.tkf92_ext} ===",
          flush=True)
    print("Train ...", flush=True)
    tkf92_train = simulate_tkf92_cherries(
        args.n_train, args.tkf92_lam, args.tkf92_mu, args.tkf92_ext,
        ts_train, Q, pi, seed=args.seed + 1, max_len=args.max_len)
    print("Val ...", flush=True)
    tkf92_val = simulate_tkf92_cherries(
        args.n_val, args.tkf92_lam, args.tkf92_mu, args.tkf92_ext,
        ts_val, Q, pi, seed=args.seed + 2, max_len=args.max_len)
    pickle.dump(tkf92_train,
                open(os.path.join(args.out_dir, "sim_tkf92", "train.pkl"), "wb"))
    pickle.dump(tkf92_val,
                open(os.path.join(args.out_dir, "sim_tkf92", "val.pkl"), "wb"))

    # GGI dataset — y derived from reversibility (matches unpack_ggi production logic)
    ggi_rho = args.ggi_lam0 / max(args.ggi_mu0, 1e-30)
    ggi_y = y_from_reversibility(args.ggi_x, ggi_rho, args.ggi_segment)
    ggi_rho_geom = (args.ggi_lam0 * (1 - args.ggi_x)
                     / max(args.ggi_mu0 * (1 - ggi_y), 1e-30))
    print(f"\n=== GGI ground truth: λ₀={args.ggi_lam0} μ₀={args.ggi_mu0} "
          f"x={args.ggi_x} y={ggi_y:.4f} ({args.ggi_segment}); "
          f"ρ={ggi_rho:.4f} ρ_GGI={ggi_rho_geom:.4f} "
          f"mean_L={ggi_rho_geom/max(1-ggi_rho_geom,1e-30):.1f} ===",
          flush=True)
    print("Train ...", flush=True)
    ggi_train = simulate_ggi_cherries(
        args.n_train, args.ggi_lam0, args.ggi_mu0, args.ggi_x, ggi_y,
        ts_train, Q, pi, seed=args.seed + 3, max_len=args.max_len)
    print("Val ...", flush=True)
    ggi_val = simulate_ggi_cherries(
        args.n_val, args.ggi_lam0, args.ggi_mu0, args.ggi_x, ggi_y,
        ts_val, Q, pi, seed=args.seed + 4, max_len=args.max_len)
    pickle.dump(ggi_train,
                open(os.path.join(args.out_dir, "sim_ggi", "train.pkl"), "wb"))
    pickle.dump(ggi_val,
                open(os.path.join(args.out_dir, "sim_ggi", "val.pkl"), "wb"))

    # Meta
    meta = {
        "n_train": args.n_train, "n_val": args.n_val, "seed": args.seed,
        "max_len": args.max_len, "max_aln_len": args.max_aln_len,
        "n_pfam_t_pool": int(t_pool.shape[0]),
        "t_train_stats": {
            "mean": float(ts_train.mean()), "median": float(np.median(ts_train)),
            "min": float(ts_train.min()), "max": float(ts_train.max())},
        "t_val_stats": {
            "mean": float(ts_val.mean()), "median": float(np.median(ts_val)),
            "min": float(ts_val.min()), "max": float(ts_val.max())},
        "tkf92_gen": {"lam": args.tkf92_lam, "mu": args.tkf92_mu,
                       "ext": args.tkf92_ext},
        "ggi_gen": {"lam0": args.ggi_lam0, "mu0": args.ggi_mu0,
                     "x": args.ggi_x, "y": ggi_y,
                     "segment": args.ggi_segment,
                     "rho": ggi_rho, "rho_GGI": ggi_rho_geom,
                     "mean_anc_len_GGI_prior": ggi_rho_geom / max(1 - ggi_rho_geom, 1e-30)},
        "tkf92_train_lengths": {
            "mean_anc": float(np.mean([len(p[0]) for p in tkf92_train])),
            "mean_des": float(np.mean([len(p[1]) for p in tkf92_train])),
            "max_anc": int(max(len(p[0]) for p in tkf92_train)),
            "max_des": int(max(len(p[1]) for p in tkf92_train))},
        "ggi_train_lengths": {
            "mean_anc": float(np.mean([len(p[0]) for p in ggi_train])),
            "mean_des": float(np.mean([len(p[1]) for p in ggi_train])),
            "max_anc": int(max(len(p[0]) for p in ggi_train)),
            "max_des": int(max(len(p[1]) for p in ggi_train))},
    }
    json.dump(meta, open(os.path.join(args.out_dir, "sim_meta.json"), "w"),
              indent=2)
    print(f"\nSaved to {args.out_dir}/.  Meta:\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
