#!/usr/bin/env python3
"""Score the best params from each fit_pfam_gap_counts result on a SEPARATE
validation gap-counts npz.

For each of {TKF92-constant, GGI-flowed-stable, GGI-frozen-stable}, load the
JSON history, evaluate the conditional LL on the val npz at each history
entry's params, and report (train_LL, val_LL, val_nats_per_cherry) per init.

Runs on CPU; eval per init is ~ a few seconds.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.dirname(THIS_DIR))

# Force JAX to CPU to avoid grabbing the GPUs that are still in use.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp

from fit_via_gap_counts import (
    conditional_ll_tkf92_constant,
    conditional_ll_ggi_flowed,
    conditional_ll_ggi_flowed_stable,
    conditional_ll_ggi_frozen_stable,
)
from fit_pfam_gap_counts import _fold_clip


def _logit(p):
    return float(np.log(p / max(1 - p, 1e-30)))


def _load_counts(npz_path, Lmax_fit, symmetrize):
    d = np.load(npz_path)
    g = d["gap_counts"].astype(np.float32)
    if Lmax_fit > 0 and Lmax_fit < int(d["Lmax"]):
        g = _fold_clip(g, Lmax_fit)
    if symmetrize:
        g = ((g + g.transpose(0, 1, 3, 2)) / 2.0).astype(np.float32)
    t = d["trans_counts"].astype(np.float32)
    tau = d["tau_centers"].astype(np.float32)
    return (jnp.asarray(g), jnp.asarray(t), jnp.asarray(tau),
            int(d.get("n_cherries_per_bin", np.zeros(1)).sum()))


def score_tkf92(hist, gj, tj, taj, Lmax):
    out = []
    for h in hist:
        if not np.isfinite(h.get("best_ll", -np.inf)):
            out.append({"init": h.get("init"), "train_ll": float("nan"),
                        "val_ll": float("nan")})
            continue
        log_del = jnp.asarray(np.log(h["mu"]), jnp.float32)
        logit_kappa = jnp.asarray(_logit(h["kappa"]), jnp.float32)
        logit_ext = jnp.asarray(_logit(h["r"]), jnp.float32)
        ll = float(conditional_ll_tkf92_constant(
            log_del, logit_kappa, logit_ext, gj, tj, taj, Lmax))
        out.append({
            "init": h.get("init"),
            "train_ll": float(h["best_ll"]),
            "val_ll": ll,
            "params": {k: h[k] for k in ("lam", "mu", "kappa", "r")},
        })
    return out


def score_ggi_unconstrained(hist, gj, tj, taj, Lmax):
    """Score using conditional_ll_ggi_flowed (UNCONSTRAINED). For history
    entries from the unconstrained run; takes (lam0, x, y) directly."""
    out = []
    for h in hist:
        if not np.isfinite(h.get("best_ll", -np.inf)):
            out.append({"init": h.get("init"), "val_ll": float("nan")})
            continue
        log_lam0 = jnp.asarray(np.log(h["lam0"]), jnp.float32)
        logit_x = jnp.asarray(_logit(h["x"]), jnp.float32)
        logit_y = jnp.asarray(_logit(h["y"]), jnp.float32)
        ll = float(conditional_ll_ggi_flowed(
            log_lam0, logit_x, logit_y, gj, tj, taj, Lmax))
        out.append({
            "init": h.get("init"),
            "train_ll": float(h["best_ll"]),
            "val_ll": ll,
            "params": {k: h[k] for k in ("lam0", "mu0", "x", "y",
                                          "rho_lam0_over_mu0")},
        })
    return out


def score_ggi_stable(hist, gj, tj, taj, Lmax, kind):
    """Score the STABLE GGI params using either conditional_ll_ggi_flowed_stable
    or conditional_ll_ggi_frozen_stable.

    History entries store (mu0, rho, x, y) which we can recover via the
    unpack_stable_ggi formula. We need the SAME (log_mu0, logit_rho, logit_y)
    that produced these.  Convenient route: read the saved (log_mu0,
    logit_rho, logit_y) from the best entry; for the history rows, the
    params are the converged-from-init endpoints and we re-derive logit_y
    from the saved y, log_mu0 from mu0, logit_rho from rho_lam0_over_mu0.
    x_branch is decided by whether x > 0.5.
    """
    out = []
    loss_fn = (conditional_ll_ggi_flowed_stable if kind == "flowed"
               else conditional_ll_ggi_frozen_stable)
    for h in hist:
        if not np.isfinite(h.get("best_ll", -np.inf)):
            out.append({"init": h.get("init"), "val_ll": float("nan")})
            continue
        mu0 = float(h["mu0"])
        rho = float(h["rho_lam0_over_mu0"])
        y = float(h["y"])
        x = float(h["x"])
        x_branch_upper = x > 0.5
        log_mu0 = jnp.asarray(np.log(max(mu0, 1e-30)), jnp.float32)
        # Clamp rho to (epsilon, 1-epsilon) for finite logit
        rho_c = min(max(rho, 1e-9), 1.0 - 1e-9)
        logit_rho = jnp.asarray(_logit(rho_c), jnp.float32)
        logit_y = jnp.asarray(_logit(min(max(y, 1e-9), 1 - 1e-9)), jnp.float32)
        ll = float(loss_fn(log_mu0, logit_rho, logit_y, gj, tj, taj, Lmax,
                            x_branch_upper=x_branch_upper))
        out.append({
            "init": h.get("init"),
            "train_ll": float(h["best_ll"]),
            "val_ll": ll,
            "params": {k: h.get(k) for k in (
                "lam0", "mu0", "x", "y", "rho_lam0_over_mu0", "x_branch")},
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-npz", default="pfam_gap_counts.npz")
    p.add_argument("--val-npz", default="pfam_gap_counts_val.npz")
    p.add_argument("--fit-dir", default="experiments/pfam_gap_fit")
    p.add_argument("--Lmax-fit", type=int, default=20)
    p.add_argument("--symmetrize", action="store_true", default=True)
    p.add_argument("--no-symmetrize", dest="symmetrize", action="store_false")
    p.add_argument("--out", default="experiments/pfam_gap_fit/pfam_val_scores.json")
    args = p.parse_args()

    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    print(f"Loading val counts from {args.val_npz} ...")
    gj_v, tj_v, taj_v, n_val_cherries = _load_counts(
        args.val_npz, args.Lmax_fit, args.symmetrize)
    print(f"  val total cherries: {n_val_cherries:,}")
    print(f"  Lmax_fit = {args.Lmax_fit}  symmetrize = {args.symmetrize}")

    Lmax = args.Lmax_fit

    out = {"val_n_cherries": n_val_cherries, "Lmax_fit": Lmax,
           "symmetrize": args.symmetrize}

    from fit_via_gap_counts import (
        joint_ll_tkf92_constant, joint_ll_ggi_flowed_stable_native,
        joint_ll_ggi_frozen_stable_native,
    )

    def _logit_local(p):
        return float(np.log(p / max(1 - p, 1e-30)))

    def score_jt_tkf(hist, gj, tj, taj, Lmax):
        out = []
        for h in hist:
            if not np.isfinite(h.get("best_ll", -np.inf)):
                out.append({"init": h.get("init"), "train_ll": float("nan"),
                            "val_ll": float("nan")})
                continue
            ll = float(joint_ll_tkf92_constant(
                jnp.asarray(np.log(h["mu"]), jnp.float32),
                jnp.asarray(_logit_local(h["kappa"]), jnp.float32),
                jnp.asarray(_logit_local(h["r"]), jnp.float32),
                gj, tj, taj, Lmax))
            out.append({"init": h.get("init"), "train_ll": float(h["best_ll"]),
                        "val_ll": ll,
                        "params": {k: h[k] for k in ("lam","mu","kappa","r")}})
        return out

    def score_jt_ggi(hist, gj, tj, taj, Lmax, kind):
        loss_fn = (joint_ll_ggi_flowed_stable_native if kind == "flowed"
                   else joint_ll_ggi_frozen_stable_native)
        out = []
        for h in hist:
            if not np.isfinite(h.get("best_ll", -np.inf)):
                out.append({"init": h.get("init"), "val_ll": float("nan")})
                continue
            ll = float(loss_fn(
                jnp.asarray(np.log(max(h["mu0"], 1e-30)), jnp.float32),
                jnp.asarray(_logit_local(min(max(h["rho_lam0_over_mu0"], 1e-9), 1 - 1e-9)), jnp.float32),
                jnp.asarray(_logit_local(min(max(h["y"], 1e-9), 1 - 1e-9)), jnp.float32),
                gj, tj, taj, Lmax,
                x_branch_upper=(h["x"] > 0.5)))
            out.append({"init": h.get("init"), "train_ll": float(h["best_ll"]),
                        "val_ll": ll,
                        "params": {k: h.get(k) for k in (
                            "lam0","mu0","x","y","rho_lam0_over_mu0","x_branch")}})
        return out

    # JOINT-NATIVE fits
    for fname, key, score_fn in [
        ("pfam_fit_tkf92_jointnative.json",       "tkf92_jt",       score_jt_tkf),
        ("pfam_fit_ggi_flowed_jointnative.json",  "ggi_flowed_jt",  lambda h, *a: score_jt_ggi(h, *a, kind="flowed")),
        ("pfam_fit_ggi_frozen_jointnative.json",  "ggi_frozen_jt",  lambda h, *a: score_jt_ggi(h, *a, kind="frozen")),
    ]:
        fpath = os.path.join(args.fit_dir, fname)
        if not os.path.exists(fpath):
            continue
        print(f"\nScoring {key} (joint-native) from {fpath} ...")
        j = json.load(open(fpath))
        hist = j.get("history", [])
        t0 = time.monotonic()
        scored = score_fn(hist, gj_v, tj_v, taj_v, Lmax)
        print(f"  {len(scored)} entries scored in {time.monotonic()-t0:.1f}s")
        out[key] = scored
        for s in scored:
            ll = s.get("val_ll"); tl = s.get("train_ll", float("nan"))
            ll_s = f"{ll:>15,.0f}" if np.isfinite(ll) else "      nan      "
            tl_s = f"{tl:>15,.0f}" if np.isfinite(tl) else "      nan      "
            nat = ll / max(n_val_cherries, 1) if np.isfinite(ll) else float("nan")
            print(f"    {s['init']:25s} train={tl_s}  val={ll_s}  ({nat:+.4f} nats/cherry)")

    # TKF92
    f_tkf = os.path.join(args.fit_dir, "pfam_fit_tkf92.json")
    if os.path.exists(f_tkf):
        print(f"\nScoring TKF92 from {f_tkf} ...")
        j = json.load(open(f_tkf))
        hist = j.get("history", [])
        t0 = time.monotonic()
        scored = score_tkf92(hist, gj_v, tj_v, taj_v, Lmax)
        print(f"  {len(scored)} entries scored in {time.monotonic()-t0:.1f}s")
        out["tkf92"] = scored
        for s in scored:
            ll = s.get("val_ll")
            ll_str = f"{ll:>15,.0f}" if np.isfinite(ll) else "      nan      "
            tl = s.get("train_ll", float("nan"))
            tl_str = f"{tl:>15,.0f}" if np.isfinite(tl) else "      nan      "
            nat = ll / max(n_val_cherries, 1) if np.isfinite(ll) else float("nan")
            print(f"    {s['init']:25s} train={tl_str}  val={ll_str}  "
                  f"({nat:+.4f} nats/cherry)")

    # GGI flowed (stable)
    f_ggi = os.path.join(args.fit_dir, "pfam_fit_ggi_flowed.json")
    if os.path.exists(f_ggi):
        print(f"\nScoring GGI-flowed (stable) from {f_ggi} ...")
        j = json.load(open(f_ggi))
        hist = j.get("history", [])
        t0 = time.monotonic()
        scored = score_ggi_stable(hist, gj_v, tj_v, taj_v, Lmax, "flowed")
        print(f"  {len(scored)} entries scored in {time.monotonic()-t0:.1f}s")
        out["ggi_flowed"] = scored
        for s in scored:
            ll = s.get("val_ll")
            ll_str = f"{ll:>15,.0f}" if np.isfinite(ll) else "      nan      "
            tl = s.get("train_ll", float("nan"))
            tl_str = f"{tl:>15,.0f}" if np.isfinite(tl) else "      nan      "
            nat = ll / max(n_val_cherries, 1) if np.isfinite(ll) else float("nan")
            print(f"    {s['init']:25s} train={tl_str}  val={ll_str}  "
                  f"({nat:+.4f} nats/cherry)")

    # GGI frozen (stable)
    f_ggi_frozen = os.path.join(args.fit_dir, "pfam_fit_ggi_frozen.json")
    if os.path.exists(f_ggi_frozen):
        print(f"\nScoring GGI-frozen (stable) from {f_ggi_frozen} ...")
        j = json.load(open(f_ggi_frozen))
        hist = j.get("history", [])
        t0 = time.monotonic()
        scored = score_ggi_stable(hist, gj_v, tj_v, taj_v, Lmax, "frozen")
        print(f"  {len(scored)} entries scored in {time.monotonic()-t0:.1f}s")
        out["ggi_frozen"] = scored
        for s in scored:
            ll = s.get("val_ll")
            ll_str = f"{ll:>15,.0f}" if np.isfinite(ll) else "      nan      "
            tl = s.get("train_ll", float("nan"))
            tl_str = f"{tl:>15,.0f}" if np.isfinite(tl) else "      nan      "
            nat = ll / max(n_val_cherries, 1) if np.isfinite(ll) else float("nan")
            print(f"    {s['init']:25s} train={tl_str}  val={ll_str}  "
                  f"({nat:+.4f} nats/cherry)")

    # Summary: best val LL per family
    print(f"\n{'='*72}\nSUMMARY (val LL @ best-by-train, per method)\n{'='*72}")
    summary = {}
    for method in ("tkf92", "ggi_flowed", "ggi_frozen"):
        if method not in out:
            continue
        scored = out[method]
        finite = [s for s in scored if np.isfinite(s.get("val_ll", -np.inf))]
        if not finite:
            continue
        # Best by val LL
        best_val = max(finite, key=lambda s: s["val_ll"])
        # Best by train LL
        best_train = max(finite, key=lambda s: s["train_ll"])
        print(f"  {method:13s}  best-by-train  val={best_val['val_ll']:>15,.0f}  "
              f"({best_val['val_ll']/max(n_val_cherries,1):+.4f} nats/cherry)  "
              f"init={best_val['init']}")
        print(f"  {method:13s}  best-by-val    val={best_val['val_ll']:>15,.0f}  "
              f"init={best_val['init']}")
        summary[method] = {
            "best_by_train": best_train, "best_by_val": best_val
        }
    out["summary"] = summary

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
