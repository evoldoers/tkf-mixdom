#!/usr/bin/env python3
"""Stage 2: at each grid point, evaluate the simulated val data under
both ground-truth models:

  * TKF92(λ, μ, ext) — the TKF92 ANCESTOR of this grid point.  Constant
    rates; ignores the t-flow that the GGI generator actually had.
  * GGI(λ₀, μ₀, x, y) — the GGI truth, with prior_swap=True
    (i.e. P_GGI(anc) · P_TKF92(des|anc; flowed)).
    Also report prior_swap=False as a side metric.

Answers: AT TRUTH, does closed-form GGI better represent
GGI-simulated data?  (Should be yes if the t-flow has any signal at
all and the TKF92 truth is a strict approximation.)

Output: experiments/2dfb_sim_grid/eval_at_truth.json
"""
from __future__ import annotations

import os
import sys
import time
import json

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["TKFMIXDOM_MAX_PAD"] = "256"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_gpu_enable_command_buffer=").strip()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "2dfb"))

import pickle
import numpy as np
import jax.numpy as jnp

from tkfmixdom.jax.core.protein import rate_matrix_lg
from eval_all_on_1500val import eval_tkf92, eval_ggi  # bin-bucketed vmap


def main():
    out_dir = "experiments/2dfb_sim_grid"
    # Poll for grid_meta.json (Stage 1 emits it before any sim data).
    meta_path = os.path.join(out_dir, "grid_meta.json")
    while not os.path.exists(meta_path):
        print(f"  waiting for {meta_path} ...", flush=True)
        time.sleep(30)
    grid_meta = json.load(open(meta_path))
    Q, pi = rate_matrix_lg()

    results = {"grid_meta": grid_meta, "evals": []}

    for pt in grid_meta["grid_points"]:
        idx = pt["idx"]
        lam, mu, ext = pt["lam"], pt["mu"], pt["ext"]
        ggi = pt["ggi"]
        cell_dir = os.path.join(out_dir, f"grid{idx:02d}")
        # Stream: wait for this cell's val.pkl to appear (Stage 1 writes it).
        val_path = os.path.join(cell_dir, "val.pkl")
        while not os.path.exists(val_path):
            print(f"  grid{idx:02d}: waiting for sim data ...", flush=True)
            time.sleep(30)
        val_pairs = pickle.load(open(val_path, "rb"))
        n_val = len(val_pairs)
        print(f"\n=== grid{idx:02d}: TKF92 truth ({lam}, {mu:.5f}, {ext}) "
              f"⇒ GGI ({ggi['lam0']:.5f}, {ggi['mu0']:.5f}, "
              f"{ggi['x']:.4f}, {ggi['y']:.4f}); n_val={n_val} ===", flush=True)

        t0 = time.time()
        ll_tkf = eval_tkf92(lam, mu, ext, val_pairs, Q, pi)
        v_tkf = ll_tkf / n_val
        print(f"  TKF92(truth) val_ll/pair = {v_tkf:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        ll_ggi_swap = eval_ggi(
            ggi["lam0"], ggi["mu0"], ggi["x"], ggi["y"],
            val_pairs, Q, pi, prior_swap=True)
        v_ggi_swap = ll_ggi_swap / n_val
        print(f"  GGI(truth, swap=T) val_ll/pair = {v_ggi_swap:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        ll_ggi_noswap = eval_ggi(
            ggi["lam0"], ggi["mu0"], ggi["x"], ggi["y"],
            val_pairs, Q, pi, prior_swap=False)
        v_ggi_noswap = ll_ggi_noswap / n_val
        print(f"  GGI(truth, swap=F) val_ll/pair = {v_ggi_noswap:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

        print(f"  Δ (ggi_swap − tkf92) = {v_ggi_swap - v_tkf:+.4f}  "
              f"(positive ⇒ GGI better)", flush=True)

        results["evals"].append({
            "idx": idx, "lam_truth": lam, "mu_truth": mu, "ext_truth": ext,
            "ggi_truth": ggi,
            "tkf92_at_truth": {"val_ll_per_pair": v_tkf, "val_ll_total": ll_tkf},
            "ggi_at_truth_swap":   {"val_ll_per_pair": v_ggi_swap,
                                      "val_ll_total": ll_ggi_swap},
            "ggi_at_truth_noswap": {"val_ll_per_pair": v_ggi_noswap,
                                      "val_ll_total": ll_ggi_noswap},
            "delta_ggi_minus_tkf92": v_ggi_swap - v_tkf,
        })
        # Re-save after each cell so partial results are always on disk.
        json.dump(results,
                  open(os.path.join(out_dir, "eval_at_truth.json"), "w"),
                  indent=2, default=float)

    print(f"\nSaved {out_dir}/eval_at_truth.json")

    print("\n=== Summary (val_ll/pair) ===")
    print(f"{'idx':>3} {'lam':>7} {'ext':>5}  {'TKF92':>10} "
          f"{'GGI swap':>10} {'GGI noswap':>11} {'Δ swap-tkf':>11}")
    for e in results["evals"]:
        print(f"{e['idx']:>3} {e['lam_truth']:>7.4f} {e['ext_truth']:>5.2f}  "
              f"{e['tkf92_at_truth']['val_ll_per_pair']:>10.4f} "
              f"{e['ggi_at_truth_swap']['val_ll_per_pair']:>10.4f} "
              f"{e['ggi_at_truth_noswap']['val_ll_per_pair']:>11.4f} "
              f"{e['delta_ggi_minus_tkf92']:>+11.4f}")


if __name__ == "__main__":
    main()
