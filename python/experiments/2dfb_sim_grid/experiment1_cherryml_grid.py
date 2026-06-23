#!/usr/bin/env python3
"""Experiment #1 — alignment-given (CherryML-style) grid fit.

For each grid point (λ, ext, t-scale):
  1. Project TKF92 truth → GGI upper-segment.
  2. Simulate K cherry pair ALIGNMENTS (no residues) from the GGI generator
     with t-values drawn from the Pfam empirical distribution scaled by
     t-scale.  For each pair keep the (5×5) transition count matrix.
  3. At-truth eval: log P(counts | TKF92 truth) and
     log P(counts | GGI truth, per-pair-flowed).
  4. Fit by Adam over the count-based LL:
     TKF92  → params (λ, μ, ext); shared across pairs.
     GGI    → params (λ₀, μ₀, x, y); flowed per pair via the GGI ODE.
     Closed-form gradients through `tkf92_trans` + GGI flow + jax.grad.
  5. Eval fitted params, record (truth vs fit) for both models.

Runtime: ~5 min per cell on a single GPU (sim is alignment-only; fit
is scalar LL with no F-B over sequences).

Output: experiments/2dfb_sim_grid/experiment1_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "2dfb_sim"))

import jax
import jax.numpy as jnp
import jax.random as jr

from tkfmixdom.jax.core.params import tkf92_trans
from tkfmixdom.jax.core.protein import rate_matrix_lg
from simulate_pfam_like import (
    load_pfam_t_distribution, sample_ts_from_empirical,
    y_from_reversibility, ggi_flow_at_t)


# Grid (matches generate_grid.py)
LAM_GRID = [0.015, 0.025, 0.040, 0.060]
EXT_GRID = [0.55, 0.62, 0.70]
KAPPA = 0.98


def project_tkf92_to_ggi_upper(lam, mu, ext):
    """Same projection as generate_grid.py: TKF92(λ, μ, ext) → GGI(λ₀, μ₀, x, y)
    matching dynamics at t=0."""
    mu0 = mu * (1.0 - ext)
    lam0 = lam * (1.0 - ext)
    rho = lam / max(mu, 1e-30)
    x_min = (1.0 - math.sqrt(max(1.0 - rho, 0.0))) / 2.0
    x = (1 - x_min) + 1e-3 if ext <= 1 - x_min else ext
    y = y_from_reversibility(x, rho, "upper")
    return lam0, mu0, x, y, rho


# Alignment simulation: reuse the existing simulator and extract the
# transition count matrix from the alignment.  Residues are sampled
# but discarded; this is the cheap step.  ~75 ms/pair on a single CPU.
def simulate_alignment_counts(lam, mu, t, ext, rng):
    """Returns (5,5) int transition count matrix for one cherry alignment."""
    from tkfmixdom.jax.simulate.simulate import (
        simulate_stationary_sequence, simulate_descendant_tkf92)
    from tkfmixdom.jax.core.ctmc import transition_matrix
    rng1, rng2 = jr.split(rng)
    # Dummy sub_matrix; we don't use the residues but the existing fn
    # needs it for the emission step.  Pick the identity-ish 20×20 LG.
    Q, pi = rate_matrix_lg()
    pi_np = np.asarray(pi)
    sub_matrix = np.asarray(transition_matrix(Q, float(t)))
    ancestor = simulate_stationary_sequence(rng1, lam, mu, pi_np, 600, ext=ext)
    if len(ancestor) == 0 or len(ancestor) > 600:
        return None
    _desc, alignment = simulate_descendant_tkf92(
        rng2, ancestor, lam, mu, float(t), ext, sub_matrix, pi_np)

    # Convert alignment to state sequence: 0=S (only at start), 1=M, 2=I, 3=D, 4=E
    S, M, I, D, E = 0, 1, 2, 3, 4
    state_seq = [S]
    for anc, des in alignment:
        if anc is not None and des is not None:
            state_seq.append(M)
        elif des is not None:
            state_seq.append(I)
        elif anc is not None:
            state_seq.append(D)
    state_seq.append(E)

    # Count transitions
    n_chi = np.zeros((5, 5), dtype=np.int64)
    for a, b in zip(state_seq[:-1], state_seq[1:]):
        n_chi[a, b] += 1
    return n_chi


def log_p_alignment_tkf92(lam, mu, t, ext, n_chi):
    """log P(alignment | TKF92(lam,mu,t,ext)) = Σ_{ij} n_chi[ij] log τ_{ij}.

    Differentiable w.r.t. (lam, mu, ext, t) for any-t-fitting.
    """
    tau = tkf92_trans(lam, mu, t, ext)  # (5, 5) JAX array
    log_tau = jnp.log(jnp.maximum(tau, 1e-30))
    return jnp.sum(n_chi * log_tau)


def log_p_alignment_ggi(lam0, mu0, x, y, t, n_chi):
    """log P(alignment | GGI(lam0,mu0,x,y) flowed to t)."""
    lam_t, mu_t, r_t = ggi_flow_at_t_jax(lam0, mu0, x, y, t)
    return log_p_alignment_tkf92(lam_t, mu_t, t, r_t, n_chi)


def ggi_flow_at_t_jax(lam0, mu0, x, y, t):
    """JAX version of GGI flow — FIXED-RATE form (paper's recommended
    closed-form GGI -> TKF92 surrogate).  Per wideboy_to_lambda.md
    2026-06-03; previously used slaved-rate form (lam_t = lam0/(1-r_t))."""
    num = lam0 * y * (1 - x) + mu0 * x * (1 - y)
    den = lam0 * (1 - y) + mu0 * (1 - x)
    r_b = num / jnp.maximum(den, 1e-30)
    r_inf = r_b / (2 - r_b)
    k = (lam0 + mu0) * (2 - r_b) / jnp.maximum(1 - r_b, 1e-30)
    r_t = r_inf + (r_b - r_inf) * jnp.exp(-k * t)
    one_m_r0 = jnp.maximum(1 - r_b, 1e-30)
    lam_t = lam0 / one_m_r0
    mu_t = mu0 / one_m_r0
    return lam_t, mu_t, r_t


def total_ll_tkf92(params_log, n_chis, ts):
    """params_log = (log_mu, logit_kappa, logit_ext) with κ<1 strict
    (λ = κ μ) so TKF92 stays in the regime where the smooth β/γ formulas
    give valid probabilities.  Without this, Adam happily walks λ > μ
    and ends up with τ entries > 1 / negative probabilities, yielding
    spurious positive log-LL.
    """
    log_mu, logit_kappa, logit_ext = params_log
    mu = jnp.exp(log_mu)
    kappa = jax.nn.sigmoid(logit_kappa)
    lam = kappa * mu
    ext = jax.nn.sigmoid(logit_ext)

    def per_pair(n_chi, t):
        return log_p_alignment_tkf92(lam, mu, t, ext, n_chi)
    lls = jax.vmap(per_pair)(n_chis, ts)
    return jnp.sum(lls)


def total_ll_ggi(params_log, n_chis, ts):
    """params_log = (log_mu0, logit_rho, logit_x_raw); reuses unpack_ggi (upper)."""
    log_mu0, logit_rho, logit_x_raw = params_log
    mu0 = jnp.exp(log_mu0)
    rho = jax.nn.sigmoid(logit_rho)
    lam0 = rho * mu0
    x_min = (1.0 - jnp.sqrt(jnp.maximum(1.0 - rho, 0.0))) / 2.0
    raw_x = jax.nn.sigmoid(logit_x_raw)
    xv = 1.0 - raw_x * x_min  # upper segment
    q = xv * (1.0 - xv) / jnp.maximum(rho, 1e-30)
    disc = jnp.maximum(1.0 - 4.0 * q, 0.0)
    yv = (1.0 + jnp.sqrt(disc)) / 2.0

    def per_pair(n_chi, t):
        return log_p_alignment_ggi(lam0, mu0, xv, yv, t, n_chi)
    lls = jax.vmap(per_pair)(n_chis, ts)
    return jnp.sum(lls)


def init_tkf92_params(lam, mu, ext):
    """Init in (log_mu, logit_kappa, logit_ext) basis with κ = λ/μ < 1."""
    kappa = lam / max(mu, 1e-30)
    kappa = min(max(kappa, 1e-6), 1 - 1e-6)
    return jnp.array([
        math.log(mu),
        math.log(kappa / (1 - kappa)),
        math.log(ext / (1 - ext))])


def init_ggi_params(mu0, rho, x):
    x_min = (1.0 - math.sqrt(max(1.0 - rho, 0.0))) / 2.0
    raw_x = (1 - x) / x_min  # upper segment inverse
    raw_x = max(min(raw_x, 1 - 1e-6), 1e-6)
    return jnp.array([
        math.log(mu0), math.log(rho / (1 - rho)),
        math.log(raw_x / (1 - raw_x))])


def adam_fit(init_params, total_ll_fn, n_chis, ts, lr=0.05, n_iter=400):
    """Plain Adam on the negative-LL of the counts."""
    loss = jax.jit(lambda p: -total_ll_fn(p, n_chis, ts))
    grad_fn = jax.jit(jax.grad(loss))

    params = init_params
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_loss = float('inf')
    best_params = params
    for it in range(1, n_iter + 1):
        g = grad_fn(params)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** it)
        vh = v / (1 - b2 ** it)
        params = params - lr * mh / (jnp.sqrt(vh) + eps)
        l = float(loss(params))
        if l < best_loss:
            best_loss = l
            best_params = params
    return best_params, best_loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precompiled-dir", default="pfam/precompiled")
    ap.add_argument("--max-aln-len", type=int, default=256)
    ap.add_argument("--n-pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/2dfb_sim_grid/experiment1_results.json")
    ap.add_argument("--n-iter", type=int, default=400, help="Adam iters/cell")
    ap.add_argument("--lr", type=float, default=0.05)
    # Optional extra "fast" point (K=20 component 9 with t × 10)
    ap.add_argument("--include-fast-point", action="store_true",
                     help="Append the K=20-component-9 cell with t-scale=10 "
                          "as cell 'fast' (the strong-flow test point).")
    args = ap.parse_args()

    print("Loading Pfam empirical t-distribution ...", flush=True)
    t_pool = load_pfam_t_distribution(args.precompiled_dir, args.max_aln_len)

    cells = []
    for lam in LAM_GRID:
        for ext in EXT_GRID:
            mu = lam / KAPPA
            cells.append({"label": f"grid_lam{lam}_ext{ext}",
                          "lam": lam, "mu": mu, "ext": ext, "t_scale": 1.0})
    if args.include_fast_point:
        cells.append({"label": "fast_comp9_tx10",
                      "lam": 0.0709, "mu": 0.0720, "ext": 0.5882,
                      "t_scale": 10.0})

    rng_master = np.random.default_rng(args.seed)
    results = {"cells": [], "config": vars(args), "grid_meta": {
        "lam_grid": LAM_GRID, "ext_grid": EXT_GRID, "kappa": KAPPA,
        "n_pairs": args.n_pairs,
    }}

    for cell_idx, cell in enumerate(cells):
        t0 = time.time()
        lam, mu, ext = cell["lam"], cell["mu"], cell["ext"]
        ts_all = sample_ts_from_empirical(t_pool, args.n_pairs, rng_master) * cell["t_scale"]
        lam0, mu0, x, y, rho = project_tkf92_to_ggi_upper(lam, mu, ext)
        print(f"\n=== {cell['label']} ({cell_idx+1}/{len(cells)}): "
              f"TKF92 truth λ={lam} μ={mu:.5f} ext={ext} → "
              f"GGI λ₀={lam0:.5f} μ₀={mu0:.5f} x={x:.4f} y={y:.4f} "
              f"(t_scale={cell['t_scale']}) ===", flush=True)

        # Stage 1+2: simulate K alignments under GGI flow, count transitions per pair.
        t_sim = time.time()
        rng_jax = jr.PRNGKey(args.seed * 100 + cell_idx)
        n_chis_list = []
        ts_list = []
        n_dropped = 0
        for k in range(args.n_pairs):
            rng_jax, sub = jr.split(rng_jax)
            t = float(ts_all[k])
            lam_t = lam0 / (1 - y_from_reversibility(x, rho, "upper"))  # initial proxy
            # Actually we need the per-pair-t flow:
            lam_t_v, mu_t_v, r_t_v = ggi_flow_at_t(lam0, mu0, x, y, t)
            n_chi = simulate_alignment_counts(lam_t_v, mu_t_v, t, r_t_v, sub)
            if n_chi is None:
                n_dropped += 1
                continue
            n_chis_list.append(n_chi)
            ts_list.append(t)
        n_chis = jnp.asarray(np.stack(n_chis_list))
        ts_arr = jnp.asarray(np.array(ts_list, dtype=np.float64))
        print(f"  simulated {len(n_chis_list)} alignments "
              f"(dropped {n_dropped}) in {time.time()-t_sim:.1f}s", flush=True)

        # At-truth eval
        ll_tkf_truth = float(total_ll_tkf92(
            init_tkf92_params(lam, mu, ext), n_chis, ts_arr))
        ll_ggi_truth = float(total_ll_ggi(
            init_ggi_params(mu0, rho, x), n_chis, ts_arr))
        per_pair_tkf = ll_tkf_truth / len(n_chis_list)
        per_pair_ggi = ll_ggi_truth / len(n_chis_list)
        delta = per_pair_ggi - per_pair_tkf
        print(f"  at-truth val_ll/pair  TKF92={per_pair_tkf:.4f}  "
              f"GGI={per_pair_ggi:.4f}  Δ(GGI−TKF)={delta:+.4f}", flush=True)

        # Adam-TKF92 fit
        t_fit = time.time()
        best_p_tkf, best_l_tkf = adam_fit(
            init_tkf92_params(0.029, 0.030, 0.6),  # κ ≈ 0.97 init
            total_ll_tkf92, n_chis, ts_arr, lr=args.lr, n_iter=args.n_iter)
        # (log_mu, logit_kappa, logit_ext) basis
        mu_fit = float(jnp.exp(best_p_tkf[0]))
        kappa_fit = float(jax.nn.sigmoid(best_p_tkf[1]))
        lam_fit = kappa_fit * mu_fit
        ext_fit = float(jax.nn.sigmoid(best_p_tkf[2]))
        ll_tkf_fit = -best_l_tkf
        per_pair_tkf_fit = ll_tkf_fit / len(n_chis_list)
        print(f"  Adam-TKF92 fit:  λ̂={lam_fit:.5f} μ̂={mu_fit:.5f} "
              f"ext̂={ext_fit:.4f}  ll/pair={per_pair_tkf_fit:.4f}  "
              f"({time.time()-t_fit:.1f}s)", flush=True)

        # Adam-GGI fit (starting from truth projection)
        t_fit = time.time()
        best_p_ggi, best_l_ggi = adam_fit(
            init_ggi_params(mu0, rho, x),
            total_ll_ggi, n_chis, ts_arr, lr=args.lr, n_iter=args.n_iter)
        mu0_fit = float(jnp.exp(best_p_ggi[0]))
        rho_fit = float(jax.nn.sigmoid(best_p_ggi[1]))
        lam0_fit = rho_fit * mu0_fit
        x_min_fit = (1.0 - math.sqrt(max(1.0 - rho_fit, 0.0))) / 2.0
        raw_x_fit = float(jax.nn.sigmoid(best_p_ggi[2]))
        x_fit = 1.0 - raw_x_fit * x_min_fit
        q_fit = x_fit * (1.0 - x_fit) / max(rho_fit, 1e-30)
        y_fit = (1.0 + math.sqrt(max(1.0 - 4 * q_fit, 0.0))) / 2.0
        ll_ggi_fit = -best_l_ggi
        per_pair_ggi_fit = ll_ggi_fit / len(n_chis_list)
        print(f"  Adam-GGI fit:    λ₀̂={lam0_fit:.5f} μ₀̂={mu0_fit:.5f} "
              f"x̂={x_fit:.4f} ŷ={y_fit:.4f}  ll/pair={per_pair_ggi_fit:.4f}  "
              f"({time.time()-t_fit:.1f}s)", flush=True)

        delta_fit = per_pair_ggi_fit - per_pair_tkf_fit
        print(f"  Δ at fit (GGI−TKF) = {delta_fit:+.4f}", flush=True)

        results["cells"].append({
            "label": cell["label"],
            "truth": {"lam": lam, "mu": mu, "ext": ext, "t_scale": cell["t_scale"],
                       "ggi": {"lam0": lam0, "mu0": mu0, "x": x, "y": y, "rho": rho}},
            "n_pairs_kept": len(n_chis_list), "n_dropped": n_dropped,
            "at_truth": {"ll_tkf_per_pair": per_pair_tkf,
                          "ll_ggi_per_pair": per_pair_ggi,
                          "delta_ggi_minus_tkf": delta},
            "fit_tkf92": {"lam": lam_fit, "mu": mu_fit, "ext": ext_fit,
                            "ll_per_pair": per_pair_tkf_fit},
            "fit_ggi":   {"lam0": lam0_fit, "mu0": mu0_fit, "x": x_fit, "y": y_fit,
                            "rho": rho_fit, "ll_per_pair": per_pair_ggi_fit},
            "delta_fit_ggi_minus_tkf": delta_fit,
            "wall_seconds": time.time() - t0,
        })
        # Save partial results after each cell.
        json.dump(results, open(args.out, "w"), indent=2, default=float)
        print(f"  cell wall = {time.time()-t0:.1f}s; partial results saved.",
              flush=True)

    print(f"\nDone. Final results in {args.out}.")
    # Pretty summary
    print(f"\n{'cell':>22} {'truth: TKF':>11} {'truth: GGI':>11} "
          f"{'Δtruth':>8} {'fit: TKF':>10} {'fit: GGI':>10} {'Δfit':>8}")
    for c in results["cells"]:
        print(f"{c['label']:>22} "
              f"{c['at_truth']['ll_tkf_per_pair']:>11.4f} "
              f"{c['at_truth']['ll_ggi_per_pair']:>11.4f} "
              f"{c['at_truth']['delta_ggi_minus_tkf']:>+8.4f} "
              f"{c['fit_tkf92']['ll_per_pair']:>10.4f} "
              f"{c['fit_ggi']['ll_per_pair']:>10.4f} "
              f"{c['delta_fit_ggi_minus_tkf']:>+8.4f}")


if __name__ == "__main__":
    main()
