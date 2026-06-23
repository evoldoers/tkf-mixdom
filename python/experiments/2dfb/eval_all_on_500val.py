#!/usr/bin/env python3
"""Evaluate ALL three converged points (SVI-BW, Adam-tkf92 cold, Adam-GGI cold,
Adam-tkf92 warmstart) on the SAME 500-pair val set the warmstart used, so we
get a fair comparison free of n_val=50 sample noise.

Outputs cross_eval_500.json.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["TKFMIXDOM_MAX_PAD"] = "256"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax.numpy as jnp


def load_val_pairs(precompiled_dir, max_aln_len, n_train, n_val, seed=0):
    """Load the CANONICAL val pair set (independent of seed).

    Must match exactly load_breadth_first_pairs in run_tkf92_2dfb_pfam.py:
    sorted families, take first n_val_fam, breadth-sample with fixed
    canonical RNG (seed=0xCAFE).
    """
    from run_tkf92_2dfb_pfam import CANONICAL_VAL_SEED
    from train_pfam import PrecompiledPairSource
    src = PrecompiledPairSource(precompiled_dir, max_alignment_len=max_aln_len)
    all_decoded = src._decode_all()

    from collections import defaultdict
    by_family = defaultdict(list)
    for item in all_decoded:
        x_int, y_int, _states, _ac, _dc, t_est, fam = item
        by_family[fam].append((x_int, y_int, float(t_est)))

    families = sorted(by_family.keys())  # canonical alphabetical order
    n_val_fam = int(round(len(families) * 0.05))  # fixed at 5%, independent of n_val
    val_families = families[:n_val_fam]

    rng_val = np.random.default_rng(CANONICAL_VAL_SEED)

    def breadth_sample(families_in_order, by_fam, target_n, rng):
        per_family_q = {f: list(by_fam[f]) for f in families_in_order}
        for f in per_family_q:
            rng.shuffle(per_family_q[f])
        out = []
        any_left = True
        while any_left and len(out) < target_n:
            any_left = False
            for f in families_in_order:
                if not per_family_q[f]:
                    continue
                out.append(per_family_q[f].pop())
                any_left = True
                if len(out) >= target_n:
                    break
        return out

    val_pairs = breadth_sample(val_families, by_family, n_val, rng=rng_val)
    return val_pairs


def eval_tkf92(lam, mu, ext, val_pairs, Q, pi):
    """Sum log P_TKF92(joint anc, des) over val pairs via F-B."""
    from tkfmixdom.jax.train.tkf92_adam_fb import tkf92_log_prob_fb
    Qj, pij = jnp.asarray(Q), jnp.asarray(pi)
    total = 0.0
    for x, y, t in val_pairs:
        log_p = tkf92_log_prob_fb(
            jnp.asarray(lam), jnp.asarray(mu), jnp.asarray(ext),
            jnp.asarray(t), Qj, pij, jnp.asarray(x), jnp.asarray(y))
        total += float(log_p)
    return total


def eval_ggi(lam0, mu0, x_geom, y_geom, val_pairs, Q, pi):
    """Sum log P_GGI-steered(joint anc, des) with GGI native ancestor prior."""
    from tkfmixdom.jax.train.tkf92_adam_fb import ggi_steered_log_prob_fb
    Qj, pij = jnp.asarray(Q), jnp.asarray(pi)
    total = 0.0
    for x, y, t in val_pairs:
        Lx_real = jnp.asarray(int(x.shape[0]))
        log_p = ggi_steered_log_prob_fb(
            jnp.asarray(lam0), jnp.asarray(mu0),
            jnp.asarray(x_geom), jnp.asarray(y_geom),
            jnp.asarray(t), Qj, pij,
            jnp.asarray(x), jnp.asarray(y), Lx_real)
        total += float(log_p)
    return total


def main():
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    Q, pi = rate_matrix_lg()
    print("Loading 500-pair val set (same as Adam-tkf92 warmstart) ...",
          flush=True)
    val_pairs = load_val_pairs("pfam/precompiled", 256, 20000, 500, seed=0)
    print(f"  loaded {len(val_pairs)} val pairs", flush=True)

    results = {"n_val": len(val_pairs), "models": {}}

    # SVI-BW solution (TKF92 model)
    svi = json.load(open("experiments/2dfb/svi_bw.json"))
    lam_svi, mu_svi, ext_svi = (svi["best_lam"], svi["best_mu"],
                                 svi["best_ext"])
    print(f"\nEvaluating SVI-BW @ (λ={lam_svi:.4f}, μ={mu_svi:.4f}, "
          f"ext={ext_svi:.4f}) ...", flush=True)
    t0 = time.time()
    ll = eval_tkf92(lam_svi, mu_svi, ext_svi, val_pairs, Q, pi)
    print(f"  val_ll/pair = {ll/len(val_pairs):.4f}  ({time.time()-t0:.1f}s)")
    results["models"]["svi_bw"] = {
        "lam": lam_svi, "mu": mu_svi, "ext": ext_svi,
        "val_ll_total": ll, "val_ll_per_pair": ll / len(val_pairs)}

    # Adam-tkf92 cold solution
    adam_t = json.load(open("experiments/2dfb/adam_tkf92.json"))
    from tkfmixdom.jax.train.tkf92_adam_fb import unpack_tkf92
    lam_at, mu_at, ext_at = (float(v) for v in unpack_tkf92(
        [jnp.asarray(p) for p in adam_t["best_params"]]))
    print(f"\nEvaluating Adam-tkf92 cold @ (λ={lam_at:.4f}, μ={mu_at:.4f}, "
          f"ext={ext_at:.4f}) ...", flush=True)
    t0 = time.time()
    ll = eval_tkf92(lam_at, mu_at, ext_at, val_pairs, Q, pi)
    print(f"  val_ll/pair = {ll/len(val_pairs):.4f}  ({time.time()-t0:.1f}s)")
    results["models"]["adam_tkf92_cold"] = {
        "lam": lam_at, "mu": mu_at, "ext": ext_at,
        "val_ll_total": ll, "val_ll_per_pair": ll / len(val_pairs)}

    # Adam-tkf92 warmstart solution
    ws_path = "experiments/2dfb/adam_tkf92_warmstart.json"
    if os.path.exists(ws_path):
        ws = json.load(open(ws_path))
        lam_w, mu_w, ext_w = (float(v) for v in unpack_tkf92(
            [jnp.asarray(p) for p in ws["best_params"]]))
        print(f"\nEvaluating Adam-tkf92 warmstart @ (λ={lam_w:.4f}, "
              f"μ={mu_w:.4f}, ext={ext_w:.4f}) ...", flush=True)
        t0 = time.time()
        ll = eval_tkf92(lam_w, mu_w, ext_w, val_pairs, Q, pi)
        print(f"  val_ll/pair = {ll/len(val_pairs):.4f}  "
              f"({time.time()-t0:.1f}s)")
        results["models"]["adam_tkf92_warmstart"] = {
            "lam": lam_w, "mu": mu_w, "ext": ext_w,
            "val_ll_total": ll, "val_ll_per_pair": ll / len(val_pairs)}

    # Adam-GGI cold solution (lower segment)
    adam_g = json.load(open("experiments/2dfb/adam_ggi.json"))
    from tkfmixdom.jax.train.tkf92_adam_fb import unpack_ggi
    lam0, mu0, xg, yg = (float(v) for v in unpack_ggi(
        [jnp.asarray(p) for p in adam_g["best_params"]], "lower"))
    print(f"\nEvaluating Adam-GGI cold (lower) @ (λ₀={lam0:.4f}, "
          f"μ₀={mu0:.4f}, x={xg:.4f}, y={yg:.4f}, ρ={lam0/mu0:.4f}) ...",
          flush=True)
    t0 = time.time()
    ll = eval_ggi(lam0, mu0, xg, yg, val_pairs, Q, pi)
    print(f"  val_ll/pair = {ll/len(val_pairs):.4f}  ({time.time()-t0:.1f}s)")
    results["models"]["adam_ggi_cold"] = {
        "segment": "lower", "lam0": lam0, "mu0": mu0, "x": xg, "y": yg,
        "rho": lam0/mu0,
        "val_ll_total": ll, "val_ll_per_pair": ll / len(val_pairs)}

    # Adam-GGI upper-segment warmstart solution
    ws_up_path = "experiments/2dfb/adam_ggi_warmstart_upper.json"
    if os.path.exists(ws_up_path):
        ws_up = json.load(open(ws_up_path))
        seg = ws_up.get("args", {}).get("ggi_segment", "upper")
        lam0, mu0, xg, yg = (float(v) for v in unpack_ggi(
            [jnp.asarray(p) for p in ws_up["best_params"]], seg))
        print(f"\nEvaluating Adam-GGI warmstart ({seg}) @ (λ₀={lam0:.4f}, "
              f"μ₀={mu0:.4f}, x={xg:.4f}, y={yg:.4f}, ρ={lam0/mu0:.4f}) ...",
              flush=True)
        t0 = time.time()
        ll = eval_ggi(lam0, mu0, xg, yg, val_pairs, Q, pi)
        print(f"  val_ll/pair = {ll/len(val_pairs):.4f}  "
              f"({time.time()-t0:.1f}s)")
        results["models"]["adam_ggi_warmstart_upper"] = {
            "segment": seg, "lam0": lam0, "mu0": mu0, "x": xg, "y": yg,
            "rho": lam0/mu0,
            "val_ll_total": ll, "val_ll_per_pair": ll / len(val_pairs)}

    json.dump(results, open("experiments/2dfb/eval_all_on_500val.json", "w"),
              indent=2)
    print("\n=== SUMMARY (val_ll/pair on 500-pair val) ===")
    for name, m in results["models"].items():
        print(f"  {name:25s}  {m['val_ll_per_pair']:10.4f}")


if __name__ == "__main__":
    main()
