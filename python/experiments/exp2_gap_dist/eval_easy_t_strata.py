#!/usr/bin/env python3
"""Easy version of Exp #2 (real-Pfam, gap-length distribution).

Stratify the precompiled cherry corpus by t-bin.  For each bin and
each candidate param set, compute gap-length log-likelihood using the
HYPERGEOMETRIC formula (gap_logprob.per_pair_gap_loglike) — NOT the
simpler SMIDE-path product of transition weights.

Param sets evaluated:
  - SVI-BW canonical (2D-FB unaligned MLE)        TKF92
  - Adam-tkf92 cold (2D-FB unaligned MLE seed=0)  TKF92
  - Aligned Pfam K=1 (CherryML)                   TKF92
  - Adam-GGI upper no-prior-swap                  GGI-flowed per pair
  - Adam-GGI upper native long2400                GGI-flowed per pair

Output: experiments/exp2_gap_dist/easy_t_strata.json + console table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from gap_logprob import (
    per_pair_gap_loglike, tkf92_trans_np, ggi_flow_at_t_np,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompiled-dir", default="pfam/precompiled")
    ap.add_argument("--max-aln-len", type=int, default=None,
                     help="Cap alignment length.  Default None (no cap) — "
                          "the gap-LL eval does no 2D DP so the 256 cap "
                          "inherited from the F-B path is unnecessary and "
                          "silently drops ~14%% of cherries, biasing the "
                          "high-t bins.")
    ap.add_argument("--out",
                     default="experiments/exp2_gap_dist/easy_t_strata.json")
    ap.add_argument("--t-bin-edges", default="0,0.3,0.7,1.3,2.0,5.0",
                     help="Comma-separated list of t-bin edges.")
    args = ap.parse_args()

    bin_edges = np.array([float(s) for s in args.t_bin_edges.split(",")])
    n_bins = len(bin_edges) - 1
    print(f"t-bins: {bin_edges}  ({n_bins} bins)", flush=True)

    # Decode all cherries from precompiled corpus.
    from train_pfam import PrecompiledPairSource
    src = PrecompiledPairSource(args.precompiled_dir,
                                  max_alignment_len=args.max_aln_len)
    t0 = time.time()
    all_decoded = src._decode_all()
    print(f"Decoded {len(all_decoded):,} cherries from {args.precompiled_dir} "
          f"in {time.time()-t0:.1f}s", flush=True)

    # Group by t-bin
    by_bin = [[] for _ in range(n_bins)]
    for item in all_decoded:
        x_int, y_int, states, anc_chars, desc_chars, t, *_ = item
        bin_i = np.searchsorted(bin_edges, t, side="right") - 1
        if 0 <= bin_i < n_bins:
            by_bin[bin_i].append((states, float(t)))
    print("Per-bin counts:",
          [len(b) for b in by_bin], flush=True)

    # GGI params loaded from JSON at FULL precision (rounded params hit
    # κ=1 degeneracy).  TKF92 baselines still inline (they're from the
    # historical SVI-BW / Adam-tkf92 / aligned-K1 fits).
    import math as _math
    def _unpack_ggi(path):
        d = json.load(open(path))
        log_mu0, logit_rho, logit_x_raw = d['best_params']
        mu0 = _math.exp(log_mu0)
        rho = 1.0 / (1 + _math.exp(-logit_rho))
        lam0 = rho * mu0
        x_min = (1 - _math.sqrt(max(1 - rho, 0))) / 2
        raw_x = 1.0 / (1 + _math.exp(-logit_x_raw))
        x = 1.0 - raw_x * x_min  # upper segment
        q = x * (1 - x) / max(rho, 1e-30)
        sd = _math.sqrt(max(1 - 4 * q, 0))
        y = (1 + sd) / 2  # upper root
        return {"lam0": lam0, "mu0": mu0, "x": x, "y": y}

    param_sets = {
        "svi_bw_2dfb":           ("tkf92", {"lam": 0.03268, "mu": 0.03331,
                                              "ext": 0.5798}),
        "adam_tkf92_cold_seed0": ("tkf92", {"lam": 0.05786, "mu": 0.06280,
                                              "ext": 0.7534}),
        "aligned_K1_cherryml":   ("tkf92", {"lam": 0.02969, "mu": 0.03027,
                                              "ext": 0.6510}),
        "adam_ggi_noswap_s0":    ("ggi", _unpack_ggi(
            "experiments/2dfb/adam_ggi_upper_noswap_seed0_fixedrate.json")),
        "adam_ggi_native_l2400": ("ggi", _unpack_ggi(
            "experiments/2dfb/adam_ggi_upper_long2400_seed0_fixedrate.json")),
    }

    results = {"bin_edges": bin_edges.tolist(),
                "bin_counts": [len(b) for b in by_bin],
                "models": {}}

    for name, (kind, p) in param_sets.items():
        print(f"\nEvaluating {name} ({kind}) ...", flush=True)
        t0 = time.time()
        per_bin_ll = []
        per_bin_n = []
        for bin_i, pairs in enumerate(by_bin):
            if not pairs:
                per_bin_ll.append(0.0); per_bin_n.append(0); continue
            total = 0.0
            n = 0
            for states, t in pairs:
                if kind == "tkf92":
                    tau = tkf92_trans_np(p["lam"], p["mu"], t, p["ext"])
                else:
                    lam_t, mu_t, r_t = ggi_flow_at_t_np(
                        p["lam0"], p["mu0"], p["x"], p["y"], t)
                    tau = tkf92_trans_np(lam_t, mu_t, t, r_t)
                total += per_pair_gap_loglike(states, tau)
                n += 1
            per_bin_ll.append(total)
            per_bin_n.append(n)
        print(f"  done in {time.time()-t0:.1f}s")
        print(f"  per-bin LL/pair: ", end="")
        for bin_i in range(n_bins):
            if per_bin_n[bin_i]:
                pp = per_bin_ll[bin_i] / per_bin_n[bin_i]
                print(f"[{bin_edges[bin_i]}-{bin_edges[bin_i+1]}]:"
                       f"{pp:.3f}  ", end="")
        print()
        results["models"][name] = {
            "kind": kind, "params": p,
            "per_bin_ll_total": per_bin_ll,
            "per_bin_n": per_bin_n,
            "per_bin_ll_per_pair": [
                per_bin_ll[i] / max(per_bin_n[i], 1) for i in range(n_bins)],
            "total_ll": float(sum(per_bin_ll)),
            "total_n": sum(per_bin_n),
            "total_ll_per_pair": (sum(per_bin_ll)
                                    / max(sum(per_bin_n), 1)),
        }
        json.dump(results, open(args.out, "w"), indent=2, default=float)

    # Pretty diff table
    print("\n=== Δ (GGI − TKF92) gap-LL/pair per t-bin ===")
    print(f"{'t-bin':>14}",
           f" {'#pairs':>8}",
           f" {'svi_bw':>10}",
           f" {'aligned':>10}",
           f" {'GGI_noswap':>12}",
           f" {'Δ noswap-svi':>14}",
           f" {'Δ noswap-aln':>14}", sep="")
    svi = results["models"]["svi_bw_2dfb"]["per_bin_ll_per_pair"]
    aln = results["models"]["aligned_K1_cherryml"]["per_bin_ll_per_pair"]
    ggi = results["models"]["adam_ggi_noswap_s0"]["per_bin_ll_per_pair"]
    nat = results["models"]["adam_ggi_native_l2400"]["per_bin_ll_per_pair"]
    for bin_i in range(n_bins):
        n = results["bin_counts"][bin_i]
        if not n: continue
        print(f"{bin_edges[bin_i]:>5}-{bin_edges[bin_i+1]:<6}",
               f" {n:>8}",
               f" {svi[bin_i]:>10.3f}",
               f" {aln[bin_i]:>10.3f}",
               f" {ggi[bin_i]:>12.3f}",
               f" {ggi[bin_i] - svi[bin_i]:>+14.3f}",
               f" {ggi[bin_i] - aln[bin_i]:>+14.3f}", sep="")


if __name__ == "__main__":
    main()
