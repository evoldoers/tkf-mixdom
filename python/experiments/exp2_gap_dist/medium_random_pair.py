#!/usr/bin/env python3
"""Exp #2 medium — gap-LL on random pair per Pfam family from full MSAs.

For each Pfam-seed family, sample ONE random ordered pair of sequences
from the MSA (not just cherries) and extract its alignment state
sequence (M / I / D) directly from the column structure.  Estimate
each pair's branch length t from a JC69 inversion of %identity in
match columns:

    p = identity in M columns
    d = 1 - p  (fraction mismatched)
    t = -3/4 ln(1 - 4/3 d)        # JC69 distance; cap at 10 if saturated

Each family contributes exactly one pair → ~20k pairs total
(one per family, no double counting, all families weighted equally).
No alignment-length cap.

Then compute gap-LL per pair via the hypergeometric formula
(experiments.exp2_gap_dist.gap_logprob.per_pair_gap_loglike) under
each of the candidate param sets.  Strata by t-bin, including bins
extending past t=5 (where cherries cap out).

Output: experiments/exp2_gap_dist/medium_random_pair.json + table.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
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
from train_pfam import parse_stockholm


# State codes (match gap_logprob.S/M/I/D/E)
S, M, I, D, E = 0, 1, 2, 3, 4
T_MAX_JC = 10.0  # cap when JC69 saturates


def alignment_to_state_seq_from_msa(s_anc, s_des):
    """Convert two aligned sequence strings (MSA columns) to a Pair-HMM
    state sequence: [S, M/I/D, ..., M/I/D, E].
    """
    gap_chars = set("-.")
    state_seq = [S]
    for ca, cd in zip(s_anc, s_des):
        a_gap = ca in gap_chars
        d_gap = cd in gap_chars
        if a_gap and d_gap:
            continue  # both gap: skip (common in MSAs with many sequences)
        elif (not a_gap) and (not d_gap):
            state_seq.append(M)
        elif a_gap and (not d_gap):
            state_seq.append(I)
        else:
            state_seq.append(D)
    state_seq.append(E)
    return state_seq


def jc69_t_from_identity(p_identity, t_max=T_MAX_JC):
    """JC69 distance from fraction identical in match columns."""
    d = 1.0 - p_identity
    if d >= 0.75 - 1e-9:
        return t_max
    return min(t_max, max(0.0, -0.75 * math.log(1.0 - (4.0 / 3.0) * d)))


def count_match_identity(s_anc, s_des):
    """Return (n_match_cols, n_identical_match) over the alignment."""
    gap_chars = set("-.")
    n_m = 0
    n_id = 0
    for ca, cd in zip(s_anc, s_des):
        if ca in gap_chars or cd in gap_chars:
            continue
        n_m += 1
        if ca.upper() == cd.upper():
            n_id += 1
    return n_m, n_id


def load_random_pair_per_family(msa_dir, seed=0, max_families=None):
    """One random ordered pair per Pfam family.  Yields tuples
       (family_id, state_seq, t_est, n_match).
    """
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(msa_dir, "PF*.sto")))
    if max_families:
        files = files[:max_families]
    print(f"  found {len(files):,} Pfam-seed MSA files", flush=True)

    out = []
    for i, f in enumerate(files):
        try:
            names, seqs = parse_stockholm(f)
        except Exception as e:
            continue
        if len(seqs) < 2:
            continue
        # Random ordered pair (anc, des)
        idx_a, idx_d = rng.choice(len(seqs), size=2, replace=False)
        s_anc = seqs[idx_a]
        s_des = seqs[idx_d]
        state_seq = alignment_to_state_seq_from_msa(s_anc, s_des)
        n_m, n_id = count_match_identity(s_anc, s_des)
        if n_m < 5:  # not enough match columns to estimate t
            continue
        p_id = n_id / n_m
        t_est = jc69_t_from_identity(p_id)
        fam = os.path.basename(f).replace(".sto", "")
        out.append((fam, state_seq, t_est, n_m))
        if (i + 1) % 1000 == 0:
            print(f"  parsed {i+1}/{len(files)}; kept {len(out)} pairs",
                  flush=True)
    print(f"  total pairs kept: {len(out):,}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--msa-dir", default="/home/yam/bio-datasets/data/pfam-seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-families", type=int, default=None,
                     help="Limit number of Pfam families (for debugging).")
    ap.add_argument("--out",
                     default="experiments/exp2_gap_dist/medium_random_pair.json")
    ap.add_argument("--t-bin-edges",
                     default="0,0.3,0.7,1.3,2.0,3.0,5.0,8.0,10.0",
                     help="Comma-separated list of t-bin edges; default goes "
                          "well past cherry corpus's t=5 cap.")
    args = ap.parse_args()

    bin_edges = np.array([float(s) for s in args.t_bin_edges.split(",")])
    n_bins = len(bin_edges) - 1
    print(f"t-bins: {bin_edges}  ({n_bins} bins)", flush=True)

    t0 = time.time()
    pairs = load_random_pair_per_family(args.msa_dir, seed=args.seed,
                                          max_families=args.max_families)
    print(f"  loading + parsing took {time.time()-t0:.1f}s", flush=True)

    # Group by t-bin
    by_bin = [[] for _ in range(n_bins)]
    for fam, state_seq, t, n_m in pairs:
        bin_i = int(np.searchsorted(bin_edges, t, side="right") - 1)
        if 0 <= bin_i < n_bins:
            by_bin[bin_i].append((state_seq, t, n_m))
    print(f"Per-bin counts: {[len(b) for b in by_bin]}", flush=True)

    # Full-precision GGI params loaded from JSON to avoid the κ=1
    # degeneracy that 4-decimal rounding produces.
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
        # Cherry-trained Adam-GGI (the old "GGI fits don't transfer" rows).
        "adam_ggi_noswap_cherry":    ("ggi", _unpack_ggi(
            "experiments/2dfb/adam_ggi_upper_noswap_seed0_fixedrate.json")),
        "adam_ggi_native_cherry": ("ggi", _unpack_ggi(
            "experiments/2dfb/adam_ggi_upper_long2400_seed0_fixedrate.json")),
    }
    # Medium-trained Adam-GGI (the OOD-removed comparison).  Only added
    # if the medium-fit JSONs exist (skip native if not yet finished).
    for key, path in [
        ("adam_ggi_noswap_medium",
         "experiments/exp2_gap_dist/medium_fb/adam_ggi_noswap_n800.json"),
        ("adam_ggi_native_medium",
         "experiments/exp2_gap_dist/medium_fb/adam_ggi_native_long2400.json"),
    ]:
        if os.path.exists(path):
            param_sets[key] = ("ggi", _unpack_ggi(path))

    results = {"bin_edges": bin_edges.tolist(),
                "bin_counts": [len(b) for b in by_bin],
                "models": {}}

    for name, (kind, p) in param_sets.items():
        print(f"\nEvaluating {name} ({kind}) ...", flush=True)
        t0 = time.time()
        per_bin_ll = []
        per_bin_n = []
        for bin_i, bin_pairs in enumerate(by_bin):
            if not bin_pairs:
                per_bin_ll.append(0.0); per_bin_n.append(0); continue
            total = 0.0
            n = 0
            for state_seq, t, n_m in bin_pairs:
                if kind == "tkf92":
                    tau = tkf92_trans_np(p["lam"], p["mu"], t, p["ext"])
                else:
                    lam_t, mu_t, r_t = ggi_flow_at_t_np(
                        p["lam0"], p["mu0"], p["x"], p["y"], t)
                    tau = tkf92_trans_np(lam_t, mu_t, t, r_t)
                total += per_pair_gap_loglike(state_seq, tau)
                n += 1
            per_bin_ll.append(total)
            per_bin_n.append(n)
        print(f"  done in {time.time()-t0:.1f}s")
        print(f"  per-bin LL/pair: ", end="")
        for bin_i in range(n_bins):
            if per_bin_n[bin_i]:
                pp = per_bin_ll[bin_i] / per_bin_n[bin_i]
                print(f"[{bin_edges[bin_i]}-{bin_edges[bin_i+1]}]:"
                      f"{pp:.2f}  ", end="")
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

    # Diff table — auto-detect which models are present.
    print("\n=== Δ vs aligned-K1 (per t-bin), random pair per family ===")
    aln = results["models"]["aligned_K1_cherryml"]["per_bin_ll_per_pair"]
    ggi_keys = [k for k in results["models"] if k.startswith("adam_ggi_")]
    print(f"{'t-bin':>14} {'#pairs':>8} {'aligned':>10}  "
          + "  ".join(f"{k.replace('adam_ggi_', ''):>16}" for k in ggi_keys))
    print(f"{'':>14} {'':>8} {'':>10}  "
          + "  ".join(f"{'Δ vs aligned':>16}" for _ in ggi_keys))
    for bin_i in range(n_bins):
        n = results["bin_counts"][bin_i]
        if not n: continue
        a = aln[bin_i]
        ggi_vals = [results["models"][k]["per_bin_ll_per_pair"][bin_i]
                     for k in ggi_keys]
        print(f"{bin_edges[bin_i]:>5}-{bin_edges[bin_i+1]:<6}  "
              f"{n:>8} {a:>10.2f}  "
              + "  ".join(f"{v - a:>+16.3f}" for v in ggi_vals))


if __name__ == "__main__":
    main()
