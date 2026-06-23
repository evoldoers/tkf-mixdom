#!/usr/bin/env python3
"""Stage 4: gather grid results into one summary table.

For each grid cell, reports:
  * Truth: TKF92 (λ, μ, ext), projected GGI (λ₀, μ₀, x, y)
  * At-truth val LL/pair: TKF92(truth), GGI(truth, swap), GGI(truth, noswap)
  * Fitted: Adam-TKF92 (λ̂, μ̂, ext̂) + val LL, Adam-GGI (λ̂₀, μ̂₀, x̂, ŷ) + val LL
  * Parameter recovery: |Δlam|, |Δmu|, |Δext| for TKF92, |Δlam0|, |Δmu0|,
    |Δx|, |Δy| for GGI
  * Three "does GGI better X" booleans:
      better_at_truth, better_at_fit, better_param_recovery_(rho_geom)

Outputs CSV (analyze.csv) + JSON (analyze.json) + a pretty stdout table.
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import jax.numpy as jnp

from tkfmixdom.jax.train.tkf92_adam_fb import unpack_tkf92, unpack_ggi


def main():
    out_dir = "experiments/2dfb_sim_grid"
    grid_meta = json.load(open(os.path.join(out_dir, "grid_meta.json")))
    eval_truth = json.load(open(os.path.join(out_dir, "eval_at_truth.json")))
    truth_by_idx = {e["idx"]: e for e in eval_truth["evals"]}

    rows = []
    for pt in grid_meta["grid_points"]:
        idx = pt["idx"]
        cell_dir = os.path.join(out_dir, f"grid{idx:02d}")
        et = truth_by_idx.get(idx, {})

        row = {
            "idx": idx, "lam_truth": pt["lam"], "mu_truth": pt["mu"],
            "ext_truth": pt["ext"],
            "ggi_lam0_truth": pt["ggi"]["lam0"],
            "ggi_mu0_truth": pt["ggi"]["mu0"],
            "ggi_x_truth": pt["ggi"]["x"], "ggi_y_truth": pt["ggi"]["y"],
            "ll_at_truth_tkf92":      et.get("tkf92_at_truth", {}).get("val_ll_per_pair"),
            "ll_at_truth_ggi_swap":   et.get("ggi_at_truth_swap", {}).get("val_ll_per_pair"),
            "ll_at_truth_ggi_noswap": et.get("ggi_at_truth_noswap", {}).get("val_ll_per_pair"),
            "delta_at_truth_swap_minus_tkf92": et.get("delta_ggi_minus_tkf92"),
        }
        if row["ll_at_truth_tkf92"] is not None and row["ll_at_truth_ggi_swap"] is not None:
            row["better_at_truth_ggi_swap"] = row["ll_at_truth_ggi_swap"] > row["ll_at_truth_tkf92"]

        # Adam-TKF92 fit
        atk = os.path.join(cell_dir, "adam_tkf92.json")
        if os.path.exists(atk):
            d = json.load(open(atk))
            lam_fit, mu_fit, ext_fit = (float(v) for v in unpack_tkf92(
                [jnp.asarray(p) for p in d["best_params"]]))
            row["adam_tkf92_lam_fit"] = lam_fit
            row["adam_tkf92_mu_fit"] = mu_fit
            row["adam_tkf92_ext_fit"] = ext_fit
            row["adam_tkf92_val_ll_per_pair"] = d["best_val_ll_per_pair"]
            row["adam_tkf92_delta_lam"] = lam_fit - pt["lam"]
            row["adam_tkf92_delta_mu"]  = mu_fit  - pt["mu"]
            row["adam_tkf92_delta_ext"] = ext_fit - pt["ext"]

        # Adam-GGI fit
        agi = os.path.join(cell_dir, "adam_ggi.json")
        if os.path.exists(agi):
            d = json.load(open(agi))
            seg = d.get("args", {}).get("ggi_segment", "upper")
            lam0_fit, mu0_fit, x_fit, y_fit = (float(v) for v in unpack_ggi(
                [jnp.asarray(p) for p in d["best_params"]], seg))
            rho_fit = lam0_fit / max(mu0_fit, 1e-30)
            rho_geom_fit = (lam0_fit * (1 - x_fit)
                             / max(mu0_fit * (1 - y_fit), 1e-30))
            row["adam_ggi_lam0_fit"]  = lam0_fit
            row["adam_ggi_mu0_fit"]   = mu0_fit
            row["adam_ggi_x_fit"]     = x_fit
            row["adam_ggi_y_fit"]     = y_fit
            row["adam_ggi_rho_fit"]   = rho_fit
            row["adam_ggi_rho_geom_fit"] = rho_geom_fit
            row["adam_ggi_val_ll_per_pair"] = d["best_val_ll_per_pair"]
            row["adam_ggi_delta_lam0"] = lam0_fit - pt["ggi"]["lam0"]
            row["adam_ggi_delta_mu0"]  = mu0_fit  - pt["ggi"]["mu0"]
            row["adam_ggi_delta_x"]    = x_fit    - pt["ggi"]["x"]
            row["adam_ggi_delta_y"]    = y_fit    - pt["ggi"]["y"]

        # At-fit comparison
        if "adam_tkf92_val_ll_per_pair" in row and "adam_ggi_val_ll_per_pair" in row:
            row["delta_at_fit_ggi_minus_tkf92"] = (
                row["adam_ggi_val_ll_per_pair"] -
                row["adam_tkf92_val_ll_per_pair"])
            row["better_at_fit_ggi"] = (
                row["adam_ggi_val_ll_per_pair"] >
                row["adam_tkf92_val_ll_per_pair"])

        rows.append(row)

    # Write JSON + CSV
    json.dump({"rows": rows}, open(os.path.join(out_dir, "analyze.json"), "w"),
              indent=2, default=float)

    if rows:
        keys = sorted(set().union(*[r.keys() for r in rows]))
        with open(os.path.join(out_dir, "analyze.csv"), "w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Pretty print summary
    print(f"\n{'idx':>3} {'lam':>7} {'ext':>5}  "
          f"{'truth_TKF':>10} {'truth_GGI':>10} {'Δ':>8}  "
          f"{'fit_TKF':>10} {'fit_GGI':>10} {'Δ':>8}  "
          f"{'GGI@truth':>9} {'GGI@fit':>9}")
    for r in rows:
        truth_diff = r.get("delta_at_truth_swap_minus_tkf92", float("nan"))
        fit_diff = r.get("delta_at_fit_ggi_minus_tkf92", float("nan"))
        ggi_truth_better = r.get("better_at_truth_ggi_swap", False)
        ggi_fit_better = r.get("better_at_fit_ggi", False)

        def fmt(v, w=10, d=4):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f"{'-':>{w}}"
            return f"{v:>{w}.{d}f}"

        flag_truth = "GGI" if ggi_truth_better else "TKF"
        flag_fit = "GGI" if ggi_fit_better else "TKF"

        print(f"{r['idx']:>3} {r['lam_truth']:>7.4f} {r['ext_truth']:>5.2f}  "
              f"{fmt(r.get('ll_at_truth_tkf92'))} {fmt(r.get('ll_at_truth_ggi_swap'))} "
              f"{fmt(truth_diff, 8, 4)}  "
              f"{fmt(r.get('adam_tkf92_val_ll_per_pair'))} "
              f"{fmt(r.get('adam_ggi_val_ll_per_pair'))} "
              f"{fmt(fit_diff, 8, 4)}  "
              f"{flag_truth:>9} {flag_fit:>9}")


if __name__ == "__main__":
    main()
