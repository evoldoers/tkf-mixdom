#!/usr/bin/env python3
"""Train MixFrag by summarised-count EM on Pfam cherry count tensors.

The *fit* half of the two-step MixFrag training pipeline:

  1. ``build_mixfrag_cherry_counts.py`` turns Pfam cherry alignments into
     per-family alignment-summary count tensors (one ``<family>.npz`` each).
  2. THIS script aggregates the chosen families' tensors and fits
     ``(lambda, mu, {r_f}, {w_f})`` by the exact EM of supplement B.6
     (``tkfmixdom.jax.train.mixfrag_cherry_em.em_fit``) -- one forward-backward
     per discretised time bin, in time independent of the number of cherries.

The substitution model ``(Q, pi)`` is held FIXED (LG08 by default; the EM fits
only the indel + fragtype parameters, exactly as ``svi_bw_mixfrag`` does).  The
count tensors' residue axis is the alphabetical amino-acid order
(``ACDEFGHIKLMNPQRSTVWY``), which is the order ``rate_matrix_lg``/``_wag`` return,
so the two index-align directly.

Output: one ``.npz`` with the fitted parameters + EM history + metadata, and a
sibling ``.json`` summary.  Examples::

    # after building counts into pfam/cherries_mixfrag/ :
    python train_mixfrag_cherry_em.py --counts-dir pfam/cherries_mixfrag \\
        --split-file pfam/splits.json --split train --val-split val \\
        --n-fragtypes 3 --out results/mixfrag_F3.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Family / path resolution (mirrors build_mixfrag_cherry_counts).
# ---------------------------------------------------------------------------


def _resolve_families(args) -> list[str]:
    if args.families:
        return [f.strip() for f in args.families.split(",") if f.strip()]
    if args.split_file is not None and args.split is not None:
        with open(os.path.expanduser(args.split_file)) as f:
            data = json.load(f)
        return list(data[args.split])
    counts_dir = Path(os.path.expanduser(args.counts_dir))
    return sorted(p.stem for p in counts_dir.glob("*.npz"))


def _resolve_split_families(split_file, split) -> list[str]:
    with open(os.path.expanduser(split_file)) as f:
        data = json.load(f)
    return list(data[split])


def _existing_paths(counts_dir, families) -> list[str]:
    counts_dir = Path(os.path.expanduser(counts_dir))
    paths, missing = [], 0
    for fam in families:
        p = counts_dir / f"{fam}.npz"
        if p.exists():
            paths.append(str(p))
        else:
            missing += 1
    return paths, missing


# ---------------------------------------------------------------------------
# Core (importable / testable): aggregate -> fit -> (optional) validate.
# ---------------------------------------------------------------------------


def _substitution_model(name):
    from tkfmixdom.jax.core.protein import rate_matrix_lg, rate_matrix_wag
    Q, pi = rate_matrix_lg() if name == "lg" else rate_matrix_wag()
    return np.asarray(Q, np.float64), np.asarray(pi, np.float64)


def train_mixfrag(counts_paths, *, n_fragtypes, substitution="lg",
                  val_paths=None, init_lam=0.02, init_mu=0.04,
                  init_exts=None, init_weights=None, n_iter=500, tol=1e-7,
                  prior_alpha_lam=2.0, prior_alpha_mu=2.0, prior_beta=10.0,
                  ext_prior_alpha=2.0, ext_prior_beta=3.0,
                  weight_prior_alpha=1.5, max_gap=None,
                  checkpoint_path=None, checkpoint_every=0, resume=True,
                  log_fn=print):
    """Aggregate the per-family count tensors at ``counts_paths`` and fit MixFrag
    by summarised-count EM.  Returns the ``em_fit`` result dict, augmented with
    'n_families', 'n_cherries', 'substitution', and (if ``val_paths``) a held-out
    'val_indel_ll' / 'val_sub_ll' / 'val_ll' under the fitted parameters."""
    from build_mixfrag_cherry_counts import aggregate_counts
    from tkfmixdom.jax.train.mixfrag_cherry_em import (
        em_fit, prep_bins, corpus_indel_loglik, corpus_substitution_loglik,
    )

    if not counts_paths:
        raise ValueError("No count-tensor .npz files to train on.")
    Q, pi = _substitution_model(substitution)

    agg = aggregate_counts(counts_paths)
    out = em_fit(agg, n_fragtypes=n_fragtypes, Q=Q, pi=pi,
                 init_lam=init_lam, init_mu=init_mu,
                 init_exts=init_exts, init_weights=init_weights,
                 n_iter=n_iter, tol=tol,
                 prior_alpha_lam=prior_alpha_lam, prior_alpha_mu=prior_alpha_mu,
                 prior_beta=prior_beta, ext_prior_alpha=ext_prior_alpha,
                 ext_prior_beta=ext_prior_beta, weight_prior_alpha=weight_prior_alpha,
                 max_gap=max_gap, checkpoint_path=checkpoint_path,
                 checkpoint_every=checkpoint_every, resume=resume,
                 log_fn=log_fn)
    out["n_families"] = len(counts_paths)
    out["n_cherries"] = int(agg["n_pairs"])
    out["substitution"] = substitution

    if val_paths:
        vagg = aggregate_counts(val_paths)
        vbins = prep_bins(vagg, max_gap=max_gap)
        v_indel = corpus_indel_loglik(vbins, out["lam"], out["mu"],
                                      out["exts"], out["weights"])
        v_sub = corpus_substitution_loglik(vbins, Q, pi)
        out["val_indel_ll"] = float(v_indel)
        out["val_sub_ll"] = float(v_sub)
        out["val_ll"] = float(v_indel + v_sub)
        out["val_n_families"] = len(val_paths)
        out["val_n_cherries"] = int(vagg["n_pairs"])
    return out


def _save(out_path, out):
    """Write the fitted parameters + history + metadata to <out>.npz and a
    sibling <out>.json summary."""
    out_path = Path(os.path.expanduser(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist = out["history"]
    save = dict(
        lam=np.float64(out["lam"]), mu=np.float64(out["mu"]),
        exts=np.asarray(out["exts"], np.float64),
        weights=np.asarray(out["weights"], np.float64),
        n_fragtypes=np.int64(len(out["exts"])),
        final_ll=np.float64(out["final_ll"]), sub_ll=np.float64(out["sub_ll"]),
        n_families=np.int64(out["n_families"]),
        n_cherries=np.int64(out["n_cherries"]),
        substitution=np.array(out["substitution"]),
        hist_iter=np.array([h["iter"] for h in hist], np.int64),
        hist_total_ll=np.array([h["total_ll"] for h in hist], np.float64),
        hist_lam=np.array([h["lam"] for h in hist], np.float64),
        hist_mu=np.array([h["mu"] for h in hist], np.float64),
        hist_exts=np.array([h["exts"] for h in hist], np.float64),
        hist_weights=np.array([h["weights"] for h in hist], np.float64),
    )
    for k in ("val_indel_ll", "val_sub_ll", "val_ll", "val_n_families",
              "val_n_cherries"):
        if k in out:
            save[k] = np.float64(out[k])
    np.savez(out_path, **save)

    summary = {k: (out[k] if not isinstance(out[k], np.ndarray)
                   else out[k].tolist())
               for k in ("lam", "mu", "final_ll", "sub_ll", "n_families",
                         "n_cherries", "substitution", "val_ll")
               if k in out}
    summary["exts"] = np.asarray(out["exts"]).tolist()
    summary["weights"] = np.asarray(out["weights"]).tolist()
    summary["n_iter_run"] = len(hist)
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts-dir",
                   default="~/tkf-mixdom/python/pfam/cherries_mixfrag",
                   help="Directory of per-family .npz tensors from "
                        "build_mixfrag_cherry_counts.")
    p.add_argument("--families", default=None,
                   help="Comma-separated family accessions (overrides --split).")
    p.add_argument("--split", default=None, help="Train split key in --split-file.")
    p.add_argument("--split-file", default=None, help="Path to a splits JSON.")
    p.add_argument("--val-split", default=None,
                   help="Optional held-out split key in --split-file; its "
                        "log-likelihood under the fit is reported.")
    p.add_argument("--n-fragtypes", type=int, default=2)
    p.add_argument("--substitution", choices=["lg", "wag"], default="lg")
    p.add_argument("--n-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--init-lam", type=float, default=0.02)
    p.add_argument("--init-mu", type=float, default=0.04)
    p.add_argument("--init-exts", default=None,
                   help="Comma-separated initial r_f (default: spread 0.3..0.7).")
    p.add_argument("--init-weights", default=None,
                   help="Comma-separated initial w_f (default: uniform).")
    p.add_argument("--prior-beta", type=float, default=10.0)
    p.add_argument("--ext-prior-alpha", type=float, default=2.0)
    p.add_argument("--ext-prior-beta", type=float, default=3.0)
    p.add_argument("--weight-prior-alpha", type=float, default=1.5)
    p.add_argument("--max-gap", type=int, default=100,
                   help="Drop gaps with i+j>max_gap (bounds gap-factor cost; "
                        "~0.02%% of Pfam-train gaps at 100). 0 disables.")
    p.add_argument("--checkpoint", default=None,
                   help="Resume-checkpoint .npz path (written every "
                        "--checkpoint-every iters; resumed on restart).")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Checkpoint cadence in iters (with --checkpoint).")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore an existing checkpoint and restart from init.")
    p.add_argument("--out", required=True, help="Output .npz path.")
    args = p.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)

    families = _resolve_families(args)
    counts_paths, missing = _existing_paths(args.counts_dir, families)
    print(f"[train_mixfrag] {len(counts_paths)} family count files "
          f"({missing} requested families missing); counts_dir={args.counts_dir}",
          flush=True)
    val_paths = None
    if args.val_split is not None and args.split_file is not None:
        val_fams = _resolve_split_families(args.split_file, args.val_split)
        val_paths, vmiss = _existing_paths(args.counts_dir, val_fams)
        print(f"[train_mixfrag] {len(val_paths)} val family count files "
              f"({vmiss} missing).", flush=True)

    init_exts = ([float(x) for x in args.init_exts.split(",")]
                 if args.init_exts else None)
    init_weights = ([float(x) for x in args.init_weights.split(",")]
                    if args.init_weights else None)

    t0 = time.time()
    out = train_mixfrag(
        counts_paths, n_fragtypes=args.n_fragtypes, substitution=args.substitution,
        val_paths=val_paths, init_lam=args.init_lam, init_mu=args.init_mu,
        init_exts=init_exts, init_weights=init_weights, n_iter=args.n_iter,
        tol=args.tol, prior_beta=args.prior_beta,
        ext_prior_alpha=args.ext_prior_alpha, ext_prior_beta=args.ext_prior_beta,
        weight_prior_alpha=args.weight_prior_alpha,
        max_gap=(args.max_gap if args.max_gap and args.max_gap > 0 else None),
        checkpoint_path=args.checkpoint, checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume)

    _save(args.out, out)
    print(f"\n[train_mixfrag] done in {time.time()-t0:.1f}s. "
          f"lam={out['lam']:.5f} mu={out['mu']:.5f} "
          f"exts={np.array2string(np.asarray(out['exts']), precision=4)} "
          f"weights={np.array2string(np.asarray(out['weights']), precision=4)} "
          f"final_ll={out['final_ll']:.4f}"
          + (f" val_ll={out['val_ll']:.4f}" if 'val_ll' in out else ""),
          flush=True)
    print(f"[train_mixfrag] saved -> {args.out} (+ .json)", flush=True)


if __name__ == "__main__":
    main()
