#!/usr/bin/env python3
"""Unaligned-Pfam 2D-FB training and comparison: TKF92 vs GGI-steered TKF92.

Three modes (one per --mode value):

  svi_bw       — Pure-TKF92 SVI-BW with analytic gradients via
                  tkf92_svi_bw.svi_bw_tkf92.  Baseline.
  adam_tkf92   — Adam on the 2D-FB log_prob directly, with constant
                  (lam, mu, ext) shared across all pairs.  Same model
                  family as svi_bw, different optimizer.
  adam_ggi     — Adam on the 2D-FB log_prob with GGI-steered TKF92:
                  (lam0, mu0, x, y) parameters → per-pair
                  (lam(t), mu(t), r(t)) via closed-form GGI flow.  Tests
                  whether the GGI-steered family beats constant TKF92 on
                  unaligned Pfam, without the gap-counts alignment-bias
                  confound from earlier experiments.

Shared infrastructure:
* Breadth-first sampling (one cherry per family until N, then top up).
* Bin-bucketed minibatching for JIT cache reuse.
* Held-out val LL every N_val iters with early-stop patience K.
* jax.checkpoint on forward_backward_2d (lands automatically because the
  log_prob entry points in vjp.py and the new tkf92_adam_fb.py are
  decorated).
* All four OOM mitigations from run_tkf4_svi_bw_pfam_pure_tkf92.py:
  persistent disk cache, --no-command-buffers, --bin-bucketed, --pre-warm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")


# ----- OOM env knobs (must be set BEFORE jax import) -----
def _has_flag(*names):
    return any(n in sys.argv for n in names)


def _arg_value(flag, default=None):
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return default


_jax_cache = _arg_value("--jax-cache-dir",
                         os.path.expanduser("~/.cache/tkfmixdom-jax"))
if _jax_cache and not _has_flag("--no-jax-cache"):
    os.makedirs(_jax_cache, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _jax_cache)

if _has_flag("--no-command-buffers"):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + " --xla_gpu_enable_command_buffer=").strip()

# Coarser geometric padding (round to multiples of 32, capped at this value)
# — with default geomspace bins (~40/axis) we end up with 100+ unique
# (Lx_pad, Ly_pad) shapes and the cumulative compiled-CUBIN memory (~10 GB
# at our scale) OOMs the 11 GB GPU.  TKFMIXDOM_MAX_PAD switches _pad_to_bin
# to 32-multiple rounding for at most 8 bins/axis → ≤64 2D shapes.
_max_pad = _arg_value("--max-pad-cap", None)
if _max_pad:
    os.environ["TKFMIXDOM_MAX_PAD"] = str(int(_max_pad))


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tkfmixdom.jax.core.ctmc import rate_matrix_jc69
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.train.tkf92_svi_bw import svi_bw_tkf92
from tkfmixdom.jax.train.tkf92_adam_fb import adam_fb_train


CANONICAL_VAL_SEED = 0xCAFE  # fixed for canonical val sampling; not user-tunable


def load_breadth_first_pairs(precompiled_dir, max_alignment_len, n_train,
                              n_val, seed=0):
    """Breadth-first sampling: 1 cherry per family until n_train + n_val
    total, then top up further from families with > 1 cherry.

    Returns (train_pairs, val_pairs), each a list of (x_int, y_int, t).
    Val pairs are drawn from a held-out subset of families (disjoint
    family sets between train and val).

    The val set is CANONICAL — independent of the user-provided seed:
      * families are sorted alphabetically before the train/val split,
      * the first round(max(5%·N, 1.5·n_val)) sorted families are val,
      * within-val shuffling uses a fixed seed (CANONICAL_VAL_SEED),
    so every run that requests the same (precompiled_dir, max_alignment_len,
    n_val) gets THE SAME val_pairs in THE SAME ORDER.

    The user --seed controls train_pairs only: it shuffles the train
    families (so different seeds get a different train subsample) and
    drives within-family ordering on the train side.
    """
    from train_pfam import PrecompiledPairSource
    src = PrecompiledPairSource(precompiled_dir,
                                  max_alignment_len=max_alignment_len)
    print(f"PrecompiledPairSource: {src.n_pairs:,} pairs, "
          f"{src.n_families:,} families", flush=True)
    all_decoded = src._decode_all()

    # Group by family
    from collections import defaultdict
    by_family = defaultdict(list)
    for item in all_decoded:
        x_int, y_int, _states, _ac, _dc, t_est, fam = item
        by_family[fam].append((x_int, y_int, float(t_est)))

    # CANONICAL: alphabetically sorted family list — independent of insertion
    # order in by_family (which depends on _decode_all() ordering), and
    # independent of process hash.
    families = sorted(by_family.keys())
    print(f"  {len(families):,} families with at least one decoded pair "
          f"(max_aln_len={max_alignment_len})", flush=True)

    # Canonical val/train split: take first 5% of SORTED families for
    # val.  The pool size DOES NOT scale with n_val — that would
    # re-contaminate cross-eval (val families would silently move into
    # what was train).  Increasing n_val just takes more pairs from
    # the SAME 5% of families, so a larger n_val val set is a clean
    # superset of a smaller one.  Pool size is independent of seed too.
    n_val_fam = int(round(len(families) * 0.05))
    val_families = families[:n_val_fam]
    train_families = families[n_val_fam:]

    # Two independent RNGs: canonical (val) + user-seeded (train).
    rng_val = np.random.default_rng(CANONICAL_VAL_SEED)
    rng_train = np.random.default_rng(seed)
    # Shuffle train families so different seeds get different train subsamples.
    rng_train.shuffle(train_families)

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
    train_pairs = breadth_sample(train_families, by_family, n_train,
                                  rng=rng_train)
    print(f"  breadth-sampled: train={len(train_pairs):,} pairs from "
          f"{len(train_families):,} families; val={len(val_pairs):,} pairs "
          f"from {len(val_families):,} families (CANONICAL — seed-independent).",
          flush=True)
    return train_pairs, val_pairs


def load_sim_pairs(train_file, val_file, max_len=256):
    """Load simulated train + val pairs from pickles, bypass the Pfam
    family-based canonical val sampling.  Filters to max(Lx, Ly) ≤ max_len
    so the bin-bucketing fits inside max-pad-cap (default 256, matching
    Pfam's max_aln_len)."""
    import pickle

    def _filter(pairs, label):
        before = len(pairs)
        pairs = [p for p in pairs
                  if p[0].shape[0] <= max_len and p[1].shape[0] <= max_len]
        dropped = before - len(pairs)
        if dropped:
            print(f"  filtered {label}: dropped {dropped}/{before} pairs "
                  f"with max(Lx,Ly) > {max_len}", flush=True)
        return pairs

    train_pairs = _filter(pickle.load(open(train_file, "rb")), "train")
    val_pairs = _filter(pickle.load(open(val_file, "rb")), "val")
    print(f"  loaded {len(train_pairs):,} train pairs from {train_file}",
          flush=True)
    print(f"  loaded {len(val_pairs):,} val pairs from {val_file}",
          flush=True)
    return train_pairs, val_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["svi_bw", "adam_tkf92", "adam_ggi"],
                   required=True)
    p.add_argument("--precompiled-dir", default="pfam/precompiled")
    p.add_argument("--sim-train-file", default=None,
                    help="Optional: path to a pickle of [(x_int, y_int, t), ...] "
                         "to use as train pairs instead of the Pfam corpus. "
                         "Pair with --sim-val-file.")
    p.add_argument("--sim-val-file", default=None,
                    help="Pickle of val pairs to pair with --sim-train-file.")
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--max-aln-len", type=int, default=256)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=20)
    # SVI-BW knobs
    p.add_argument("--svi-tau", type=float, default=1.0)
    p.add_argument("--svi-kappa", type=float, default=0.7)
    p.add_argument("--init-lam", type=float, default=0.04)
    p.add_argument("--init-mu", type=float, default=0.05)
    p.add_argument("--init-ext", type=float, default=0.5)
    # Adam knobs
    p.add_argument("--adam-lr", type=float, default=1e-2)
    # Adam-GGI inits in the constrained (mu0, rho, x) param.
    # y is derived from x via the symmetric root of reversibility.
    p.add_argument("--init-mu0", type=float, default=0.05)
    p.add_argument("--init-rho", type=float, default=0.9)
    p.add_argument("--init-x",   type=float, default=0.3)
    p.add_argument("--ggi-segment", choices=["lower", "upper"],
                    default="lower",
                    help="GGI x-feasibility segment: lower (x<x_min, default) "
                         "or upper (x>1-x_min, for ext>~0.4).")
    p.add_argument("--ggi-no-prior-swap", action="store_true",
                    help="Train Adam-GGI on TKF92 joint LL (GGI flow's "
                         "transition dynamics but TKF92's stationary ancestor "
                         "prior), instead of the default GGI-native joint LL. "
                         "Isolates whether GGI dynamics or its geometric prior "
                         "is the binding constraint on Pfam.")
    # Substitution model
    p.add_argument("--Q", choices=["lg", "jc"], default="lg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    # OOM knobs (also read pre-import, see top of file)
    p.add_argument("--jax-cache-dir", default=None)
    p.add_argument("--no-jax-cache", action="store_true")
    p.add_argument("--no-command-buffers", action="store_true")
    p.add_argument("--max-pad-cap", type=int, default=None,
                    help="Sets TKFMIXDOM_MAX_PAD env var to coarsen padding "
                         "to 32-multiples capped at this value. ≤8 bins/axis.")
    p.add_argument("--bin-bucketed", action="store_true")
    p.add_argument("--pre-warm", action="store_true")
    args = p.parse_args()

    if args.Q == "lg":
        Q, pi = rate_matrix_lg()
    else:
        Q, pi = rate_matrix_jc69(20)
    Q_np, pi_np = np.asarray(Q), np.asarray(pi)
    print(f"Q model: {args.Q}", flush=True)

    t0 = time.time()
    if args.sim_train_file and args.sim_val_file:
        print(f"Loading simulated pairs from {args.sim_train_file} + "
              f"{args.sim_val_file} ...", flush=True)
        train_pairs, val_pairs = load_sim_pairs(args.sim_train_file,
                                                  args.sim_val_file,
                                                  max_len=args.max_aln_len)
    else:
        print(f"Loading breadth-sampled pairs from {args.precompiled_dir} ...",
              flush=True)
        train_pairs, val_pairs = load_breadth_first_pairs(
            args.precompiled_dir, args.max_aln_len, args.n_train, args.n_val,
            seed=args.seed)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    if args.mode == "svi_bw":
        print(f"\nLaunching SVI-BW: init=({args.init_lam}, {args.init_mu}, "
              f"{args.init_ext}), n_iter={args.n_iter}, "
              f"batch={args.batch_size}", flush=True)
        out = svi_bw_tkf92(
            lambda: iter(train_pairs), n_total_pairs=len(train_pairs),
            init_lam=args.init_lam, init_mu=args.init_mu,
            init_ext=args.init_ext,
            Q=Q_np, pi=pi_np,
            n_iter=args.n_iter, batch_size=args.batch_size,
            svi_tau=args.svi_tau, svi_kappa=args.svi_kappa,
            bin_bucketed=args.bin_bucketed,
            pre_warm=args.pre_warm,
            val_pairs=val_pairs,
            val_every=args.val_every, patience=args.patience,
            seed=args.seed, log_fn=print)
        json.dump({"mode": "svi_bw", "best_lam": out.get("best_lam", out["lam"]),
                   "best_mu": out.get("best_mu", out["mu"]),
                   "best_ext": out.get("best_ext", out["ext"]),
                   "best_val_ll_per_pair": out.get("best_val_ll_per_pair"),
                   "n_train": len(train_pairs), "n_val": len(val_pairs),
                   "args": vars(args),
                   "history": out["history"]},
                  open(args.out, "w"), indent=2, default=float)
    else:
        mode = "tkf92" if args.mode == "adam_tkf92" else "ggi"
        init_params = None  # use defaults from tkf92_adam_fb
        if mode == "tkf92":
            from tkfmixdom.jax.train.tkf92_adam_fb import init_tkf92
            init_params = init_tkf92(args.init_lam, args.init_mu, args.init_ext)
        else:
            from tkfmixdom.jax.train.tkf92_adam_fb import init_ggi
            init_params = init_ggi(args.init_mu0, args.init_rho, args.init_x,
                                    segment=args.ggi_segment)
        print(f"\nLaunching Adam-on-FB ({mode}): "
              f"n_iter={args.n_iter}, batch={args.batch_size}, lr={args.adam_lr}"
              + (f", ggi_segment={args.ggi_segment}" if mode == "ggi" else ""),
              flush=True)
        out = adam_fb_train(
            train_pairs, val_pairs,
            Q=Q_np, pi=pi_np,
            mode=mode,
            init_params=init_params,
            n_iter=args.n_iter, batch_size=args.batch_size,
            val_every=args.val_every, patience=args.patience,
            lr=args.adam_lr,
            seed=args.seed, log_fn=print,
            ggi_segment=args.ggi_segment,
            ggi_prior_swap=not args.ggi_no_prior_swap)
        json.dump({"mode": args.mode,
                   "best_params": [float(p) for p in out["best_params"]],
                   "best_val_ll_per_pair": out["best_val_ll_per_pair"],
                   "n_train": len(train_pairs), "n_val": len(val_pairs),
                   "args": vars(args),
                   "history": out["history"]},
                  open(args.out, "w"), indent=2, default=float)
    print(f"\nSaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
