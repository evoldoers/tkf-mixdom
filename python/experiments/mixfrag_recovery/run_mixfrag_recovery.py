#!/usr/bin/env python3
"""MixFrag (F=2) parameter recovery by exact summarised-count EM.

Reproduces the recovery experiment quoted in the MBE paper 1 Results
("Validation on simulated data"): simulate ~12,000 MixFrag cherries at four
tau-bin centres sharing one true theta, build the multi-bin count tensors with
the real per-cherry builder, fit by `em_fit` (supplement B.6, closed-form
M-steps, flat priors), and report the recovered parameters.

The protocol is the one in tests/test_mixfrag_cherry_em.py::
test_em_recovers_simulated_parameters, with the per-iteration history,
wall-clock and the simulated corpus summary saved to JSON so the numbers in
the paper have a log behind them.  Several seeds are run so the paper can quote
a mean and spread rather than one draw.

Usage (from ~/tkf-mixdom/python):
  .venv/bin/python -u experiments/mixfrag_recovery/run_mixfrag_recovery.py \
      --seeds 0 1 2 --out experiments/mixfrag_recovery/recovery.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(os.path.dirname(HERE))      # ~/tkf-mixdom/python
sys.path.insert(0, PY)

from build_tkf92_cherry_counts import AA, TAU_MIN, TAU_MAX          # noqa: E402
from build_mixfrag_cherry_counts import (                             # noqa: E402
    count_cherry_mixfrag, geom_bin_edges, discretize_tau,
)
from tkfmixdom.jax.train.mixfrag_cherry_em import em_fit, prep_bins   # noqa: E402
from tkfmixdom.jax.simulate.mixfrag_sim import simulate_mixfrag_cherries  # noqa: E402


def build_counts(cherries, n_bins):
    edges, centers = geom_bin_edges(n_bins, TAU_MIN, TAU_MAX)
    mc = np.zeros((n_bins, AA, AA), np.int64)
    sc = np.zeros((n_bins, AA), np.int64)
    scal = np.zeros((n_bins, 3), np.int64)
    mr, gm, ge = {}, {}, {}
    for sa, sb, t in cherries:
        tb = int(discretize_tau(t, edges))
        count_cherry_mixfrag(sa, sb, tb, mc, sc, scal, mr, gm, ge)
    return dict(n_tau_bins=n_bins, tau_centers=centers, scalars=scal,
                match_counts=mc, singlet_counts=sc,
                match_run=mr, gap_to_match=gm, gap_to_end=ge,
                n_pairs=len(cherries)), centers


def jc_subst(A=AA):
    # (Q, pi) is an additive constant for the indel EM; any proper model will do.
    pi = np.ones(A) / A
    Q = np.ones((A, A)) - A * np.eye(A)
    return Q, pi


def run_one(seed, args, log):
    rng = np.random.default_rng(seed)
    Q, pi = jc_subst()
    true = dict(lam=args.true_lam, mu=args.true_mu,
                exts=list(args.true_exts), weights=list(args.true_weights))
    edges, centers = geom_bin_edges(args.n_bins, TAU_MIN, TAU_MAX)
    t_sim = time.time()
    cherries = []
    per_bin = {}
    for bi in args.bin_idxs:
        t = float(centers[bi])
        sims = simulate_mixfrag_cherries(
            rng, args.n_per_bin, true["lam"], true["mu"],
            np.array(true["exts"]), np.array(true["weights"]), t, Q, pi)
        rows = [(s["row_a"], s["row_b"], s["t"]) for s in sims]
        cherries += rows
        la = np.mean([sum(1 for c in r[0] if c != "-") for r in rows])
        lb = np.mean([sum(1 for c in r[1] if c != "-") for r in rows])
        per_bin[int(bi)] = dict(t=t, n=len(rows), mean_len_a=float(la),
                                mean_len_b=float(lb))
    log(f"seed {seed}: simulated {len(cherries)} cherries in "
        f"{time.time()-t_sim:.1f}s; per bin: {per_bin}")
    agg, centers = build_counts(cherries, args.n_bins)
    n_nonempty = sum(1 for _ in prep_bins(agg, log_fn=lambda *a, **k: None))
    log(f"seed {seed}: {n_nonempty} non-empty tau bins")

    t_fit = time.time()
    hist_times = []

    def log_fit(msg):
        hist_times.append(time.time() - t_fit)
        log(f"  [seed {seed}] {msg}")

    out = em_fit(agg, n_fragtypes=2, Q=Q, pi=pi, n_iter=args.n_iter,
                 tol=args.tol,
                 init_lam=args.init_lam, init_mu=args.init_mu,
                 init_exts=list(args.init_exts), init_weights=list(args.init_weights),
                 prior_alpha_lam=1.0, prior_alpha_mu=1.0, prior_beta=0.0,
                 ext_prior_alpha=1.0, ext_prior_beta=1.0, weight_prior_alpha=1.0,
                 log_fn=log_fit, log_every=args.log_every)
    wall = time.time() - t_fit
    order = np.argsort(out["exts"])
    fit = dict(lam=float(out["lam"]), mu=float(out["mu"]),
               exts=[float(v) for v in np.asarray(out["exts"])[order]],
               weights=[float(v) for v in np.asarray(out["weights"])[order]])
    n_it = len(out["history"])
    log(f"seed {seed}: EM {n_it} iters in {wall:.1f}s -> "
        f"lam={fit['lam']:.4f} mu={fit['mu']:.4f} "
        f"exts={np.round(fit['exts'],4).tolist()} "
        f"weights={np.round(fit['weights'],4).tolist()}  "
        f"(truth lam={true['lam']} mu={true['mu']} exts={true['exts']} "
        f"weights={true['weights']})")
    history = []
    for h in out["history"]:
        hh = {}
        for k, v in h.items():
            if isinstance(v, np.ndarray):
                hh[k] = v.tolist()
            else:
                try:
                    hh[k] = float(v)
                except (TypeError, ValueError):
                    hh[k] = v
        history.append(hh)
    return dict(seed=seed, truth=true, init=dict(
                    lam=args.init_lam, mu=args.init_mu,
                    exts=list(args.init_exts), weights=list(args.init_weights)),
                n_cherries=len(cherries), per_bin=per_bin,
                bin_centers=[float(centers[b]) for b in args.bin_idxs],
                n_iter=n_it, wall_sec=wall, fit=fit, history=history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-bins", type=int, default=32)
    p.add_argument("--bin-idxs", type=int, nargs="+", default=[20, 21, 22, 23])
    p.add_argument("--n-per-bin", type=int, default=3000)
    p.add_argument("--true-lam", type=float, default=0.08)
    p.add_argument("--true-mu", type=float, default=0.11)
    p.add_argument("--true-exts", type=float, nargs=2, default=[0.25, 0.75])
    p.add_argument("--true-weights", type=float, nargs=2, default=[0.65, 0.35])
    p.add_argument("--init-lam", type=float, default=0.04)
    p.add_argument("--init-mu", type=float, default=0.06)
    p.add_argument("--init-exts", type=float, nargs=2, default=[0.4, 0.6])
    p.add_argument("--init-weights", type=float, nargs=2, default=[0.55, 0.45])
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    def log(msg):
        print(msg, flush=True)

    log(f"args: {vars(args)}")
    runs = []
    for s in args.seeds:
        runs.append(run_one(s, args, log))
        json.dump(dict(args=vars(args), runs=runs), open(args.out, "w"),
                  indent=1, default=float)

    # Summary across seeds.
    def col(k, i=None):
        return np.array([r["fit"][k] if i is None else r["fit"][k][i] for r in runs])
    summ = {}
    for k, i, tv in [("lam", None, args.true_lam), ("mu", None, args.true_mu),
                     ("exts", 0, args.true_exts[0]), ("exts", 1, args.true_exts[1]),
                     ("weights", 0, args.true_weights[0]),
                     ("weights", 1, args.true_weights[1])]:
        v = col(k, i)
        name = k if i is None else f"{k}[{i}]"
        summ[name] = dict(truth=tv, mean=float(v.mean()),
                          sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                          values=v.tolist())
    summ["n_iter"] = [r["n_iter"] for r in runs]
    summ["wall_sec"] = [r["wall_sec"] for r in runs]
    json.dump(dict(args=vars(args), runs=runs, summary=summ),
              open(args.out, "w"), indent=1, default=float)
    log("SUMMARY (truth / mean / sd over seeds):")
    for k, v in summ.items():
        if isinstance(v, dict):
            log(f"  {k:12s} truth={v['truth']:.4f} mean={v['mean']:.4f} sd={v['sd']:.4f}")
    log(f"  iters={summ['n_iter']} wall={np.round(summ['wall_sec'],1).tolist()}")
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
