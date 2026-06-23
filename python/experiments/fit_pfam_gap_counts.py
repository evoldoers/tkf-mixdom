#!/usr/bin/env python3
"""Fit TKF92-constant and GGI-flowed on a Pfam gap-counts npz built by
build_gap_counts.py.

Reuses conditional_ll_tkf92_constant / conditional_ll_ggi_flowed from
fit_via_gap_counts. Adam loop is replaced by a JIT'd lax.scan version
(adam_max_scan) so each (init, 4000-step) run executes entirely on device.

Supports --only {tkf,ggi,both} so you can split the two halves across
two GPUs simultaneously (set CUDA_VISIBLE_DEVICES=0 / =1 on each process).

Writes per-half JSONs and (when both halves are present on disk) a
combined summary JSON.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.dirname(THIS_DIR))

from fit_via_gap_counts import (
    conditional_ll_tkf92_constant,
    conditional_ll_ggi_flowed,
    conditional_ll_ggi_flowed_stable,
    conditional_ll_ggi_frozen_stable,
    joint_ll_tkf92_constant,
    joint_ll_ggi_flowed_stable_native,
    joint_ll_ggi_frozen_stable_native,
    unpack_stable_ggi,
)


def _make_scan_adam(loss_fn, n_steps: int, lr: float):
    """JIT-compiled Adam loop that runs n_steps without host sync.

    loss_fn: (*params) -> scalar LL to maximise. Three scalar params.
    Returns a jitted function: init_params -> (best_params, best_ll, ll_trace).
    """
    b1, b2, eps = 0.9, 0.999, 1e-8
    val_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))

    def body(carry, step):
        p, m, v, best_ll, best_p = carry
        ll, grads = val_and_grad(*p)
        new_m = tuple(b1 * mi + (1 - b1) * gi for mi, gi in zip(m, grads))
        new_v = tuple(b2 * vi + (1 - b2) * (gi * gi) for vi, gi in zip(v, grads))
        dtype = p[0].dtype
        bias_m = 1.0 - b1 ** (step + 1).astype(dtype)
        bias_v = 1.0 - b2 ** (step + 1).astype(dtype)
        new_p = tuple(
            pi + lr * (mi / bias_m) / (jnp.sqrt(vi / bias_v) + eps)
            for pi, mi, vi in zip(p, new_m, new_v))
        better = ll > best_ll
        nb_ll = jnp.where(better, ll, best_ll)
        nb_p = tuple(jnp.where(better, pi, bp) for pi, bp in zip(p, best_p))
        return (new_p, new_m, new_v, nb_ll, nb_p), ll

    def run(p0):
        p = tuple(p0)
        m = tuple(jnp.zeros_like(pi) for pi in p)
        v = tuple(jnp.zeros_like(pi) for pi in p)
        best_ll = jnp.asarray(-jnp.inf, p[0].dtype)
        best_p = tuple(jnp.asarray(pi) for pi in p)
        (p_f, m_f, v_f, best_ll_f, best_p_f), ll_trace = jax.lax.scan(
            body, (p, m, v, best_ll, best_p), jnp.arange(n_steps))
        return best_p_f, best_ll_f, ll_trace

    return jax.jit(run)


def adam_max_scan(loss_fn, init_params, lr, n_steps, log_interval, label=""):
    """Drop-in replacement that runs entirely in a single JIT'd lax.scan."""
    run = _make_scan_adam(loss_fn, n_steps, lr)
    t0 = time.monotonic()
    best_p, best_ll, ll_trace = run(init_params)
    jax.block_until_ready(best_ll)
    t1 = time.monotonic()
    ll_arr = np.asarray(ll_trace)
    bll = float(best_ll)
    bp = [float(x) for x in best_p]
    if log_interval:
        for s in range(0, n_steps, log_interval):
            print(f"  [{label}] step {s:5d}  LL={float(ll_arr[s]):14.2f}  "
                  f"best_so_far={float(ll_arr[:s+1].max()):14.2f}", flush=True)
        print(f"  [{label}] step {n_steps-1:5d}  LL={float(ll_arr[-1]):14.2f}  "
              f"best={bll:14.2f}  ({t1 - t0:.1f}s wall, jit+scan)", flush=True)
    return bp, bll


def _fold_clip(gap_counts: np.ndarray, Lmax_new: int) -> np.ndarray:
    """Fold mass at i > Lmax_new into i=Lmax_new (and same for j).

    gap_counts: (n_tau, 4, Lmax_old+1, Lmax_old+1).
    Returns:    (n_tau, 4, Lmax_new+1, Lmax_new+1).
    Preserves total mass exactly.
    """
    n_tau, n_gap, L1, L2 = gap_counts.shape
    Lmax_old = L1 - 1
    assert L1 == L2 == Lmax_old + 1
    if Lmax_new >= Lmax_old:
        return gap_counts.copy()
    out = np.zeros((n_tau, n_gap, Lmax_new + 1, Lmax_new + 1), dtype=gap_counts.dtype)
    out[:, :, :Lmax_new, :Lmax_new] = gap_counts[:, :, :Lmax_new, :Lmax_new]
    out[:, :, Lmax_new, :Lmax_new] = gap_counts[:, :, Lmax_new:, :Lmax_new].sum(axis=2)
    out[:, :, :Lmax_new, Lmax_new] = gap_counts[:, :, :Lmax_new, Lmax_new:].sum(axis=3)
    out[:, :, Lmax_new, Lmax_new] = gap_counts[:, :, Lmax_new:, Lmax_new:].sum(axis=(2, 3))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="pfam_gap_counts.npz")
    p.add_argument("--out-dir", default=".", help="Where to write fit JSONs")
    p.add_argument("--tkf-lr", type=float, default=0.005)
    p.add_argument("--tkf-steps", type=int, default=6000)
    p.add_argument("--ggi-lr", type=float, default=0.002)
    p.add_argument("--ggi-steps", type=int, default=6000)
    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--Lmax-fit", type=int, default=0,
                   help="Clip Lmax to this value (overflow folded to boundary). "
                        "0 = use npz Lmax.")
    p.add_argument("--only", default="both",
                   choices=["tkf", "ggi", "ggi-frozen", "both"],
                   help="Which fit to run. Use 'tkf' + 'ggi' on separate GPUs. "
                        "'ggi-frozen' = lam(t)=lam0, mu(t)=mu0 constant; "
                        "only r(t) flows per closed form.")
    p.add_argument("--symmetrize", action="store_true", default=True,
                   help="Symmetrize gap counts in (i, j) before the fit. "
                        "Removes the artifact of which leaf is labeled 'ancestor' "
                        "in cherry extraction.")
    p.add_argument("--no-symmetrize", dest="symmetrize", action="store_false")
    p.add_argument("--loss-mode", default="conditional",
                   choices=["conditional", "joint-native"],
                   help="conditional = joint - TKF92-singlet (current default). "
                        "joint-native = joint + model-appropriate ancestor "
                        "prior: TKF92 marginal for TKF92, GGI native geometric "
                        "for GGI variants.  joint-native penalizes kappa=1 / "
                        "rho_GGI>=1 boundaries that conditional cancels out.")
    return p.parse_args()


def fit_tkf92(args, gap_j, trans_j, tau_j, Lmax, j_dtype):
    """Run TKF92-constant Adam over multiple inits. Returns (best, history)."""
    title = (f"TKF92 CONSTANT fit on Pfam gap counts ({args.loss_mode})")
    print(f"\n{'='*72}\n{title}\n{'='*72}", flush=True)

    if args.loss_mode == "joint-native":
        loss_fn = joint_ll_tkf92_constant
    else:
        loss_fn = conditional_ll_tkf92_constant

    def tkf_loss(log_del, logit_kappa, logit_ext):
        return loss_fn(
            log_del, logit_kappa, logit_ext,
            gap_j, trans_j, tau_j, Lmax)

    # Use kappa strictly < 1 to avoid sigmoid saturation at logit_kappa = +inf
    # (kappa=1.0 -> _logit(1.0) clamps to 69, where d sigmoid/d logit ~ 0 in
    # float32, so Adam can't move logit_kappa even when the loss says it
    # should).  kappa=0.99 -> logit=4.6, well within the gradient zone.
    tkf_inits = [
        ("mu0.08_k0.95_r0.5", 0.08, 0.95, 0.5),
        ("mu0.04_k0.99_r0.7", 0.04, 0.99, 0.7),
        ("mu0.04_k0.99_r0.3", 0.04, 0.99, 0.3),
        ("mu0.02_k0.99_r0.9", 0.02, 0.99, 0.9),
        ("mu0.2_k0.5_r0.5",   0.20, 0.50, 0.5),
    ]

    best = {"ll": -np.inf}
    history = []
    for label, mu, kappa, r in tkf_inits:
        init = [
            jnp.asarray(np.log(mu), j_dtype),
            jnp.asarray(_logit(kappa), j_dtype),
            jnp.asarray(_logit(r), j_dtype),
        ]
        print(f"\n  init [{label}] mu={mu} kappa={kappa} r={r}", flush=True)
        bp, bll = adam_max_scan(
            tkf_loss, init,
            lr=args.tkf_lr, n_steps=args.tkf_steps,
            log_interval=args.log_interval, label=f"TKF92/{label}")
        mu_f = float(np.exp(bp[0]))
        kappa_f = 1 / (1 + float(np.exp(-bp[1])))
        r_f = 1 / (1 + float(np.exp(-bp[2])))
        lam_f = kappa_f * mu_f
        history.append(dict(
            init=label, init_mu=mu, init_kappa=kappa, init_r=r,
            best_ll=float(bll),
            lam=float(lam_f), mu=float(mu_f), kappa=float(kappa_f), r=float(r_f),
        ))
        print(f"    => best LL = {bll:.2f}  lam={lam_f:.5f} mu={mu_f:.5f} "
              f"kappa={kappa_f:.4f} r={r_f:.4f}", flush=True)
        if bll > best["ll"]:
            best = dict(
                ll=float(bll), init=label,
                lam=float(lam_f), mu=float(mu_f),
                kappa=float(kappa_f), r=float(r_f),
                log_del=float(bp[0]),
                logit_kappa=float(bp[1]),
                logit_ext=float(bp[2]),
            )
    return best, history


def fit_ggi(args, gap_j, trans_j, tau_j, Lmax, j_dtype, kind="flowed"):
    """Run GGI Adam over multiple inits. Returns (best, history).

    kind: "flowed" -> conditional_ll_ggi_flowed_stable (closed-form lam(t),
                      mu(t), r(t) all flow with t)
          "frozen" -> conditional_ll_ggi_frozen_stable (lam(t)=lam0,
                      mu(t)=mu0 constant; only r(t) flows)

    STABLE parameterization (log_mu0, logit_rho, logit_y): rho =
    sigmoid(logit_rho) in (0, 1] enforces lam0 <= mu0 (closed-form GGI
    valid region). x is derived from reversibility via x(1-x) =
    rho*y(1-y) — two roots in (0, 1/2] (lower) and [1/2, 1) (upper).
    Both branches are explored via a static x_branch flag.
    """
    title = "GGI-FLOWED (closed-form, stable rho<=1)" if kind == "flowed" else \
            "GGI-FROZEN (lam,mu constant; r(t) flows; stable rho<=1)"
    print(f"\n{'='*72}\n{title} fit on Pfam gap counts ({args.loss_mode})\n  "
          f"params: (log_mu0, logit_rho, logit_y); "
          f"x derived from reversibility (lower/upper branch)\n{'='*72}",
          flush=True)

    if args.loss_mode == "joint-native":
        loss_fn = (joint_ll_ggi_flowed_stable_native if kind == "flowed"
                   else joint_ll_ggi_frozen_stable_native)
    else:
        loss_fn = (conditional_ll_ggi_flowed_stable if kind == "flowed"
                   else conditional_ll_ggi_frozen_stable)

    def make_loss(x_branch_upper):
        def ggi_loss(log_mu0, logit_rho, logit_y):
            return loss_fn(
                log_mu0, logit_rho, logit_y,
                gap_j, trans_j, tau_j, Lmax,
                x_branch_upper=x_branch_upper)
        return ggi_loss

    loss_lower = make_loss(False)
    loss_upper = make_loss(True)

    # Inits cover both x branches (lower: x in (0, 1/2], upper: x in [1/2, 1)).
    # Each init = (label, mu0, rho, y, branch_upper).
    # Filter: skip inits whose r*(0) > 1 (the closed-form GGI mapping is only
    # valid for r*(0) in (0, 1)).  Even with rho<=1, the upper branch combined
    # with extreme y can push r*(0) > 1; those inits NaN immediately.
    import math
    def _compute_r_boundary(mu0, rho, y, upper):
        lam0 = rho * mu0
        disc = max(1.0 - 4.0 * rho * y * (1 - y), 0.0)
        x = ((1.0 + math.sqrt(disc)) / 2.0 if upper
             else (1.0 - math.sqrt(disc)) / 2.0)
        num = lam0 * y * (1 - x) + mu0 * x * (1 - y)
        den = lam0 * (1 - y) + mu0 * (1 - x)
        return num / max(den, 1e-30)

    raw_inits = [
        # Symmetric / boundary (rho ~ 1) — both branches
        ("sym-half-lo",     0.04, 0.999, 0.40, False),  # truth-guess equiv (x~0.40)
        ("sym-half-up",     0.04, 0.999, 0.40, True),   # truth-guess (x~0.60)
        # Generic, rho-mid
        ("generic-lo",      0.05, 0.50,  0.40, False),
        # rho-hi-up replaces the original generic-up which had r*(0) = 1.24
        ("rho-hi-up",       0.05, 0.80,  0.40, True),
        # Asymmetric (rho ~ 1)
        ("ins-heavy-lo",    0.05, 0.999, 0.70, False),  # x~0.30 = ins-heavy
        ("del-heavy-up",    0.05, 0.999, 0.30, True),   # x~0.70 = del-heavy
        # Low rho
        ("rho-low-lo",      0.02, 0.30,  0.50, False),
        # rho-mid-up replaces the original rho-low-up which had r*(0) = 2.03
        ("rho-mid-up",      0.05, 0.70,  0.60, True),
    ]
    ggi_inits = []
    for entry in raw_inits:
        label, mu0_i, rho_i, y_i, br_up = entry
        rb = _compute_r_boundary(mu0_i, rho_i, y_i, br_up)
        if rb >= 1.0:
            print(f"  SKIP init [{label}]: r*(0) = {rb:.4f} >= 1 (closed-form "
                  f"invalid)", flush=True)
            continue
        ggi_inits.append(entry)

    best = {"ll": -np.inf}
    history = []
    for label, mu0_i, rho_i, y_i, x_branch_upper in ggi_inits:
        init = [
            jnp.asarray(np.log(mu0_i), j_dtype),
            jnp.asarray(_logit(rho_i), j_dtype),
            jnp.asarray(_logit(y_i), j_dtype),
        ]
        l0_eff, m0_eff, x_eff, y_eff = unpack_stable_ggi(
            *init, x_branch_upper=x_branch_upper)
        branch_str = "upper" if x_branch_upper else "lower"
        print(f"\n  init [{label}] mu0={mu0_i} rho={rho_i} y={y_i}  "
              f"branch={branch_str}   (derived: lam0={float(l0_eff):.5f}, "
              f"x={float(x_eff):.4f})", flush=True)
        loss = loss_upper if x_branch_upper else loss_lower
        bp, bll = adam_max_scan(
            loss, init,
            lr=args.ggi_lr, n_steps=args.ggi_steps,
            log_interval=args.log_interval, label=f"GGI/{label}")
        l_e, m_e, x_e, y_e = unpack_stable_ggi(
            jnp.asarray(bp[0], j_dtype),
            jnp.asarray(bp[1], j_dtype),
            jnp.asarray(bp[2], j_dtype),
            x_branch_upper=x_branch_upper)
        lam0_f = float(l_e); mu0_f = float(m_e); x_f = float(x_e); y_f = float(y_e)
        rho_f = lam0_f / max(mu0_f, 1e-30)
        num = lam0_f * y_f * (1 - x_f) + mu0_f * x_f * (1 - y_f)
        den = lam0_f * (1 - y_f) + mu0_f * (1 - x_f)
        r_boundary = num / max(den, 1e-30)
        r_inf = r_boundary / (2 - r_boundary)
        k = (lam0_f + mu0_f) * (2 - r_boundary) / max(1 - r_boundary, 1e-30)
        lam_T_boundary = lam0_f / max(1 - r_boundary, 1e-30)
        mu_T_boundary = mu0_f / max(1 - r_boundary, 1e-30)
        history.append(dict(
            init=label, init_mu0=mu0_i, init_rho=rho_i, init_y=y_i,
            x_branch=branch_str,
            best_ll=float(bll),
            lam0=float(lam0_f), mu0=float(mu0_f),
            x=float(x_f), y=float(y_f),
            rho_lam0_over_mu0=float(rho_f),
            r_boundary=float(r_boundary),
            r_inf=float(r_inf), decay_k=float(k),
            lam_T_boundary=float(lam_T_boundary),
            mu_T_boundary=float(mu_T_boundary),
        ))
        print(f"    => best LL = {bll:.2f}  rho={rho_f:.4f}  "
              f"lam0={lam0_f:.5f} mu0={mu0_f:.5f}  x={x_f:.4f} y={y_f:.4f}  "
              f"r*(0)={r_boundary:.4f}  r_inf={r_inf:.4f}  k={k:.4f}",
              flush=True)
        if bll > best["ll"]:
            best = dict(
                ll=float(bll), init=label,
                x_branch=branch_str,
                lam0=float(lam0_f), mu0=float(mu0_f),
                x=float(x_f), y=float(y_f),
                rho_lam0_over_mu0=float(rho_f),
                r_boundary=float(r_boundary),
                r_inf=float(r_inf), decay_k=float(k),
                lam_T_boundary=float(lam_T_boundary),
                mu_T_boundary=float(mu_T_boundary),
                log_mu0=float(bp[0]),
                logit_rho=float(bp[1]),
                logit_y=float(bp[2]),
                param="stable_mu0_rho_y_branch",
            )
    return best, history


def _logit(p):
    return float(np.log(p / max(1 - p, 1e-30)))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    np_dtype = np.float64 if args.dtype == "float64" else np.float32
    j_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32

    print(f"Loading {args.npz} ...", flush=True)
    d = np.load(args.npz)
    gap_counts_raw = d["gap_counts"].astype(np_dtype)
    trans_counts = d["trans_counts"].astype(np_dtype)
    tau_centers = d["tau_centers"].astype(np_dtype)
    Lmax_npz = int(d["Lmax"])
    Lmax = Lmax_npz if args.Lmax_fit == 0 else int(args.Lmax_fit)
    if Lmax < Lmax_npz:
        gap_counts = _fold_clip(gap_counts_raw, Lmax)
        kept_inner = float(
            gap_counts_raw[:, :, :Lmax, :Lmax].sum() / max(gap_counts_raw.sum(), 1)
        ) * 100
        print(f"  Lmax: npz={Lmax_npz}  fit={Lmax}  (inner-block mass = "
              f"{kept_inner:.4f}%; overflow folded to boundary)", flush=True)
    else:
        gap_counts = gap_counts_raw

    if args.symmetrize:
        # gap_counts: (n_tau, 4, Lmax+1, Lmax+1). Symmetrize over last two axes.
        gap_counts = (
            gap_counts + gap_counts.transpose(0, 1, 3, 2)) / 2.0
        gap_counts = gap_counts.astype(np_dtype)
        print(f"  symmetrized gap counts over (i,j) axes "
              f"(removes leaf-labeling artifact)", flush=True)

    n_cherries_per_bin = d["n_cherries_per_bin"].astype(np.int64)
    n_total_cherries = int(n_cherries_per_bin.sum())
    n_total_trans = int(trans_counts.sum())
    gap_totals = {
        "SM": int(gap_counts[:, 0].sum()),
        "MM": int(gap_counts[:, 1].sum()),
        "ME": int(gap_counts[:, 2].sum()),
        "SE": int(gap_counts[:, 3].sum()),
    }

    print(f"  gap_counts shape: {gap_counts.shape}  Lmax={Lmax}", flush=True)
    print(f"  trans_counts shape: {trans_counts.shape}", flush=True)
    print(f"  tau_centers: [{tau_centers.min():.4g}, {tau_centers.max():.4g}], "
          f"n_bins={tau_centers.shape[0]}", flush=True)
    print(f"  total cherries: {n_total_cherries:,}", flush=True)
    print(f"  total transitions: {n_total_trans:,}", flush=True)
    print(f"  gap totals: SM={gap_totals['SM']:,}  MM={gap_totals['MM']:,}  "
          f"ME={gap_totals['ME']:,}  SE={gap_totals['SE']:,}", flush=True)
    print(f"  jax devices: {jax.devices()}", flush=True)

    gap_j = jnp.asarray(gap_counts, j_dtype)
    trans_j = jnp.asarray(trans_counts, j_dtype)
    tau_j = jnp.asarray(tau_centers, j_dtype)

    n_tau = int(tau_centers.shape[0])
    meta = dict(
        npz=os.path.abspath(args.npz),
        n_families_ok=int(d["n_families_ok"]),
        n_families_failed=int(d["n_families_failed"]),
        n_cherries_skipped=int(d["n_cherries_skipped"]),
        n_total_cherries=n_total_cherries,
        n_total_transitions=n_total_trans,
        gap_totals=gap_totals,
        n_tau_bins=n_tau,
        Lmax_npz=Lmax_npz,
        Lmax_fit=Lmax,
        tau_min=float(tau_centers.min()),
        tau_max=float(tau_centers.max()),
        dtype=args.dtype,
        n_steps={"tkf": args.tkf_steps, "ggi": args.ggi_steps},
        lr={"tkf": args.tkf_lr, "ggi": args.ggi_lr},
    )

    suffix = "" if args.loss_mode == "conditional" else "_jointnative"
    out_tkf = os.path.join(args.out_dir, f"pfam_fit_tkf92{suffix}.json")
    out_ggi = os.path.join(args.out_dir, f"pfam_fit_ggi_flowed{suffix}.json")
    out_ggi_frozen = os.path.join(args.out_dir, f"pfam_fit_ggi_frozen{suffix}.json")
    out_sum = os.path.join(args.out_dir, f"pfam_fit_summary{suffix}.json")

    if args.only in ("tkf", "both"):
        best_tkf, tkf_history = fit_tkf92(args, gap_j, trans_j, tau_j, Lmax, j_dtype)
        with open(out_tkf, "w") as f:
            json.dump({"meta": meta, "best": best_tkf, "history": tkf_history},
                      f, indent=2)
        print(f"\nWrote {out_tkf}", flush=True)

    if args.only in ("ggi", "both"):
        best_ggi, ggi_history = fit_ggi(
            args, gap_j, trans_j, tau_j, Lmax, j_dtype, kind="flowed")
        with open(out_ggi, "w") as f:
            json.dump({"meta": meta, "best": best_ggi, "history": ggi_history},
                      f, indent=2)
        print(f"\nWrote {out_ggi}", flush=True)

    if args.only == "ggi-frozen":
        best_ggi_frozen, ggi_frozen_history = fit_ggi(
            args, gap_j, trans_j, tau_j, Lmax, j_dtype, kind="frozen")
        with open(out_ggi_frozen, "w") as f:
            json.dump({"meta": meta, "best": best_ggi_frozen,
                       "history": ggi_frozen_history}, f, indent=2)
        print(f"\nWrote {out_ggi_frozen}", flush=True)

    # Combined summary: only when both halves are on disk.
    if os.path.exists(out_tkf) and os.path.exists(out_ggi):
        with open(out_tkf) as f:
            tkf_json = json.load(f)
        with open(out_ggi) as f:
            ggi_json = json.load(f)
        bt = tkf_json["best"]
        bg = ggi_json["best"]
        delta = bg["ll"] - bt["ll"]
        nats = delta / max(meta["n_total_cherries"], 1)
        print(f"\n{'='*72}\nSUMMARY (Pfam gap-counts conditional LL)\n{'='*72}",
              flush=True)
        print(f"  total cherries      = {meta['n_total_cherries']:,}", flush=True)
        print(f"  TKF92 constant best       LL = {bt['ll']:>16.2f}  "
              f"(init {bt['init']})", flush=True)
        print(f"  GGI-flowed (closed-form)  LL = {bg['ll']:>16.2f}  "
              f"(init {bg['init']})", flush=True)
        print(f"  delta(GGI - TKF92)           = {delta:>+16.2f}  "
              f"({nats:+.4f} nats/cherry)", flush=True)
        print(f"\n  TKF92 best params:  lam={bt['lam']:.5f}  mu={bt['mu']:.5f}  "
              f"r={bt['r']:.4f}  kappa={bt['kappa']:.4f}", flush=True)
        print(f"  GGI   best params:  lam0={bg['lam0']:.5f}  "
              f"mu0={bg['mu0']:.5f}  x={bg['x']:.4f}  y={bg['y']:.4f}", flush=True)
        print(f"  GGI boundary:        lam*(0)={bg['lam_T_boundary']:.5f}  "
              f"mu*(0)={bg['mu_T_boundary']:.5f}  r*(0)={bg['r_boundary']:.4f}",
              flush=True)
        print(f"  GGI trajectory:      r_inf={bg['r_inf']:.4f}  "
              f"decay k={bg['decay_k']:.4f}", flush=True)
        print(f"  rho = lam0/mu0     = {bg['rho_lam0_over_mu0']:.4f}  "
              f"(>1 = unstable / outside valid GGI region)", flush=True)
        if bg["rho_lam0_over_mu0"] > 1.0:
            print(f"  ** NOTE: best GGI fit lies in unstable rho>1 region; "
                  f"closed form extrapolates beyond valid GGI.", flush=True)
        with open(out_sum, "w") as f:
            json.dump({
                "meta": meta,
                "tkf92_best": bt,
                "ggi_best": bg,
                "delta_ll_ggi_minus_tkf92": float(delta),
                "delta_nats_per_cherry": float(nats),
            }, f, indent=2)
        print(f"\nWrote {out_sum}", flush=True)
    else:
        missing = []
        if not os.path.exists(out_tkf):
            missing.append("pfam_fit_tkf92.json")
        if not os.path.exists(out_ggi):
            missing.append("pfam_fit_ggi_flowed.json")
        print(f"\nSummary deferred (missing: {missing}).  Re-run with --only=both "
              f"or run the other half on a second GPU, then summary will land "
              f"automatically.", flush=True)


if __name__ == "__main__":
    main()
