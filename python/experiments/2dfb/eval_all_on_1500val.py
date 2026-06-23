#!/usr/bin/env python3
"""Evaluate every converged 2D-FB Pfam model on a 1500-pair canonical
val set (3× the 500 pair eval).  Same val families as n_val=500 (still
the first 5% of sorted families) — so the 500 set is a strict prefix
of the 1500 set, and existing models trained against the n_val=500
canonical pool retain their train/val disjointness.

Bin-bucketed + vmap'd evaluation reusing the batched 2D F-B from
estep_batch_tkf92_forward_only (the SVI-BW rewrite path).  Pair-by-pair
JIT was OOM'ing on accumulated command buffers.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["TKFMIXDOM_MAX_PAD"] = "256"
# Disable CUDA command buffers — pair-by-pair JIT was hitting the
# ~1.6k-graph alive-graphs limit and OOMing the GPU.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_gpu_enable_command_buffer=").strip()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax
import jax.numpy as jnp

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.train.tkf92_adam_fb import (
    tkf92_log_prob_fb, ggi_steered_log_prob_fb,
    unpack_tkf92, unpack_ggi, _bin_bucket_pairs, _stack_bucket)

# Reuse the loader from the 500-pair eval — same logic, just larger n_val.
from eval_all_on_500val import load_val_pairs


def eval_tkf92(lam, mu, ext, val_pairs, Q, pi):
    """Bin-bucketed, vmap'd 2D F-B sum log P over val pairs.

    Replaces the pair-by-pair pair-iter that OOMs on accumulated CUDA
    command buffers at 1500+ pairs.
    """
    Qj, pij = jnp.asarray(Q), jnp.asarray(pi)
    lam_j, mu_j, ext_j = (jnp.asarray(v) for v in (lam, mu, ext))

    def per_pair(x, y, t, Lx, Ly):
        return tkf92_log_prob_fb(lam_j, mu_j, ext_j, t, Qj, pij, x, y,
                                  real_Lx=Lx, real_Ly=Ly)
    per_pair_vmap = jax.jit(jax.vmap(per_pair))

    buckets = _bin_bucket_pairs(val_pairs)
    total = 0.0
    for bk, lst in buckets.items():
        xs, ys, ts, Lx, Ly = _stack_bucket(lst)
        log_ps = per_pair_vmap(xs, ys, ts, Lx, Ly)
        total += float(jnp.sum(log_ps))
    return total


def eval_ggi(lam0, mu0, x_geom, y_geom, val_pairs, Q, pi, prior_swap=True):
    """Bin-bucketed, vmap'd GGI-steered 2D F-B sum log P over val pairs."""
    Qj, pij = jnp.asarray(Q), jnp.asarray(pi)
    lam0_j, mu0_j = jnp.asarray(lam0), jnp.asarray(mu0)
    xg_j, yg_j = jnp.asarray(x_geom), jnp.asarray(y_geom)

    def per_pair(x, y, t, Lx, Ly):
        return ggi_steered_log_prob_fb(
            lam0_j, mu0_j, xg_j, yg_j, t, Qj, pij, x, y, Lx, Ly,
            prior_swap=prior_swap)
    per_pair_vmap = jax.jit(jax.vmap(per_pair))

    buckets = _bin_bucket_pairs(val_pairs)
    total = 0.0
    for bk, lst in buckets.items():
        xs, ys, ts, Lx, Ly = _stack_bucket(lst)
        log_ps = per_pair_vmap(xs, ys, ts, Lx, Ly)
        total += float(jnp.sum(log_ps))
    return total


def main():
    Q, pi = rate_matrix_lg()
    n_val = 1500
    print(f"Loading {n_val}-pair canonical val set ...", flush=True)
    val_pairs = load_val_pairs("pfam/precompiled", 256, 20000, n_val, seed=0)
    print(f"  loaded {len(val_pairs)} val pairs", flush=True)

    results = {"n_val": len(val_pairs), "models": {}}

    # Helper to load + unpack each model.
    models = []
    def add(label, path, mode, segment="lower", prior_swap=True):
        models.append((label, path, mode, segment, prior_swap))

    # SVI-BW: 3 seeds (canonical chain) + original
    for seed in (0, 1, 2):
        add(f"svi_bw_canonical_seed{seed}",
            f"experiments/2dfb/svi_bw_canonical_seed{seed}.json", "svi_bw")

    # Adam-tkf92: 3 seeds
    for seed in (0, 1, 2):
        out = f"adam_tkf92_long.json" if seed == 0 else f"adam_tkf92_seed{seed}_long.json"
        add(f"adam_tkf92_seed{seed}", f"experiments/2dfb/{out}", "tkf92")

    # Adam-GGI upper, native prior, n_iter=800 (3 seeds)
    for seed in (0, 1, 2):
        out = "adam_ggi_upper_long.json" if seed == 0 else f"adam_ggi_upper_seed{seed}_long.json"
        add(f"adam_ggi_upper_seed{seed}_n800",
            f"experiments/2dfb/{out}", "ggi", "upper", True)

    # Adam-GGI upper, native prior, n_iter=2400 (2 seeds)
    for seed in (0, 1):
        add(f"adam_ggi_upper_long2400_seed{seed}",
            f"experiments/2dfb/adam_ggi_upper_long2400_seed{seed}.json",
            "ggi", "upper", True)

    # Adam-GGI upper, no-prior-swap (2 seeds)
    for seed in (0, 1):
        path = f"experiments/2dfb/adam_ggi_upper_noswap_seed{seed}_long.json"
        if os.path.exists(path):
            add(f"adam_ggi_upper_noswap_seed{seed}", path, "ggi", "upper", False)

    # Adam-GGI lower (1 seed)
    add("adam_ggi_lower_seed0", "experiments/2dfb/adam_ggi_long.json",
        "ggi", "lower", True)

    print(f"\n{len(models)} models to evaluate.\n", flush=True)
    for label, path, mode, segment, prior_swap in models:
        if not os.path.exists(path):
            print(f"  {label}: SKIP (file not found: {path})", flush=True)
            continue
        d = json.load(open(path))

        if mode == "svi_bw":
            lam, mu, ext = d["best_lam"], d["best_mu"], d["best_ext"]
            t0 = time.time()
            ll = eval_tkf92(lam, mu, ext, val_pairs, Q, pi)
            print(f"  {label:40s} svi_bw  "
                  f"lam={lam:.4f} mu={mu:.4f} ext={ext:.4f}  "
                  f"val_ll/pair = {ll/len(val_pairs):10.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
            results["models"][label] = {
                "mode": "svi_bw", "lam": lam, "mu": mu, "ext": ext,
                "val_ll_total": ll, "val_ll_per_pair": ll / len(val_pairs)}

        elif mode == "tkf92":
            lam, mu, ext = (float(v) for v in unpack_tkf92(
                [jnp.asarray(p) for p in d["best_params"]]))
            t0 = time.time()
            ll = eval_tkf92(lam, mu, ext, val_pairs, Q, pi)
            print(f"  {label:40s} adam_tk lam={lam:.4f} mu={mu:.4f} "
                  f"ext={ext:.4f}  val_ll/pair = {ll/len(val_pairs):10.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
            results["models"][label] = {
                "mode": "adam_tkf92", "lam": lam, "mu": mu, "ext": ext,
                "val_ll_total": ll, "val_ll_per_pair": ll / len(val_pairs)}

        else:  # ggi
            lam0, mu0, xg, yg = (float(v) for v in unpack_ggi(
                [jnp.asarray(p) for p in d["best_params"]], segment))
            t0 = time.time()
            ll = eval_ggi(lam0, mu0, xg, yg, val_pairs, Q, pi,
                           prior_swap=prior_swap)
            print(f"  {label:40s} adam_gg lam0={lam0:.4f} mu0={mu0:.4f} "
                  f"x={xg:.4f} y={yg:.4f} ρ={lam0/mu0:.4f}  "
                  f"val_ll/pair = {ll/len(val_pairs):10.4f} "
                  f"{'[noswap]' if not prior_swap else '[swap]'}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
            results["models"][label] = {
                "mode": "adam_ggi", "segment": segment,
                "prior_swap": prior_swap,
                "lam0": lam0, "mu0": mu0, "x": xg, "y": yg, "rho": lam0/mu0,
                "val_ll_total": ll, "val_ll_per_pair": ll / len(val_pairs)}

    json.dump(results, open("experiments/2dfb/eval_all_on_1500val.json", "w"),
              indent=2)

    print("\n=== SUMMARY (val_ll/pair on 1500-pair canonical val) ===")
    by_val = sorted(results["models"].items(),
                     key=lambda kv: -kv[1]["val_ll_per_pair"])
    for label, m in by_val:
        print(f"  {label:42s}  {m['val_ll_per_pair']:10.4f}")


if __name__ == "__main__":
    main()
