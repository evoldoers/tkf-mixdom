#!/usr/bin/env python3
"""Build MixFrag cherry-count tensors from the SAME precompiled pair shards that
svi-BW trains on, so MixFrag (Maraschino) and MixDom-d3f1 (svi-BW) see EXACTLY
the same data.

Each shard record is {x: seqA, a: CIGAR, y: seqB, t: branch-length, fam, id}.
We expand the CIGAR to gapped columns and feed count_cherry_mixfrag -- the same
counting the tree-cherry pipeline uses -- with tau discretised on the identical
geometric grid. Then em_fit MixFrag F=2 and score held-out val.

Run with PYTHONPATH=<tkf-mixdom/python> (does NOT modify the repo).
"""
import os, sys, glob, json, re, time, argparse
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # em_fit override via --gpu
# TKM_PATH env lets us point at a worktree that has em_fit_logspace (uncapped).
TKM = os.environ.get("TKM_PATH", os.path.expanduser("~/tkf-mixdom/python"))
sys.path.insert(0, TKM)

import zstandard
from build_tkf92_cherry_counts import (
    geom_bin_edges, discretize_tau, AA_TO_IDX, AA, TAU_MIN, TAU_MAX,
)
from build_mixfrag_cherry_counts import (
    count_cherry_mixfrag, SC_START, SC_END, SC_EMPTY, N_SCALARS,
)

CIGAR_RE = re.compile(r"([MID])(\d+)")


def expand_cigar(x, a, y):
    """(x, CIGAR a, y) -> aligned (seq_a, seq_b) with '-' gaps. M consumes both,
    D consumes x (gap in b), I consumes y (gap in a)."""
    sa = []; sb = []; xi = 0; yi = 0
    for op, n in CIGAR_RE.findall(a):
        n = int(n)
        if op == "M":
            sa.append(x[xi:xi+n]); sb.append(y[yi:yi+n]); xi += n; yi += n
        elif op == "D":
            sa.append(x[xi:xi+n]); sb.append("-"*n); xi += n
        else:  # I
            sa.append("-"*n); sb.append(y[yi:yi+n]); yi += n
    return "".join(sa), "".join(sb)


def _process_lines(args):
    """Worker: accumulate a chunk of JSONL records into partial count tensors."""
    lines, n_tau_bins, tau_floor = args
    edges, _ = geom_bin_edges(n_tau_bins, TAU_MIN, TAU_MAX)
    match_counts = np.zeros((n_tau_bins, AA, AA), dtype=np.int64)
    singlet_counts = np.zeros((n_tau_bins, AA), dtype=np.int64)
    scalars = np.zeros((n_tau_bins, N_SCALARS), dtype=np.int64)
    match_run, gap_to_match, gap_to_end = {}, {}, {}
    n_used = n_skip = 0
    for ln in lines:
        if not ln:
            continue
        try:
            r = json.loads(ln)
            x, a, y, t = r["x"], r["a"], r["y"], float(r["t"])
        except Exception:
            n_skip += 1; continue
        if not (t >= tau_floor and np.isfinite(t)):
            n_skip += 1; continue
        sa, sb = expand_cigar(x, a, y)
        if len(sa) != len(sb):
            n_skip += 1; continue
        tb = int(discretize_tau(t, edges))
        count_cherry_mixfrag(sa, sb, tb, match_counts, singlet_counts, scalars,
                             match_run, gap_to_match, gap_to_end)
        n_used += 1
    return (match_counts, singlet_counts, scalars,
            match_run, gap_to_match, gap_to_end, n_used, n_skip)


def build_agg_from_shards(shard_dir, n_tau_bins=32, workers=16, tau_floor=0.0,
                          max_run=None, log=print):
    """Build the corpus agg dict (same structure as aggregate_counts) from all
    shards in shard_dir, using count_cherry_mixfrag on the expanded CIGARs."""
    import multiprocessing as mp
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.jsonl.zst")))
    if not shards:
        raise SystemExit(f"no shards in {shard_dir}")
    edges, centers = geom_bin_edges(n_tau_bins, TAU_MIN, TAU_MAX)
    match_counts = np.zeros((n_tau_bins, AA, AA), dtype=np.int64)
    singlet_counts = np.zeros((n_tau_bins, AA), dtype=np.int64)
    scalars = np.zeros((n_tau_bins, N_SCALARS), dtype=np.int64)
    match_run, gap_to_match, gap_to_end = {}, {}, {}
    n_pairs = n_skip = 0
    t0 = time.time()
    # Read+decompress all records, then chunk by RECORD (shards may be 1 huge
    # file, so per-shard parallelism would serialise) into workers*4 chunks.
    dctx = zstandard.ZstdDecompressor()
    all_lines = []
    for s in shards:
        with open(s, "rb") as fh:
            all_lines.extend(dctx.stream_reader(fh).read()
                             .decode("utf-8", "ignore").splitlines())
    log(f"  [{shard_dir.split('/')[-1]}] {len(all_lines):,} records read "
        f"({time.time()-t0:.0f}s); counting on {workers} workers")
    nchunks = max(workers * 4, 1)
    csz = (len(all_lines) + nchunks - 1) // nchunks
    chunks = [(all_lines[k:k+csz], n_tau_bins, tau_floor)
              for k in range(0, len(all_lines), csz)]
    with mp.Pool(workers) as pool:
        for i, out in enumerate(pool.imap_unordered(_process_lines, chunks), 1):
            mc, sc, scal, mr, gm, ge, nu, ns = out
            match_counts += mc; singlet_counts += sc; scalars += scal
            for dst, src in ((match_run, mr), (gap_to_match, gm), (gap_to_end, ge)):
                for k, v in src.items():
                    dst[k] = dst.get(k, 0) + v
            n_pairs += nu; n_skip += ns
            if i % 8 == 0 or i == len(chunks):
                log(f"  [{shard_dir.split('/')[-1]}] chunk {i}/{len(chunks)}  "
                    f"pairs={n_pairs:,} skip={n_skip} ({time.time()-t0:.0f}s)")
    if max_run is not None:
        drop = {k: v for k, v in match_run.items() if k[1] > max_run}
        if drop:
            nr = sum(drop.values()); nc = sum(k[1]*v for k, v in drop.items())
            log(f"  [max_run={max_run}] dropped {nr} match-runs / {nc} columns "
                f"(run-factor only; substitution counts kept)")
            match_run = {k: v for k, v in match_run.items() if k[1] <= max_run}
    return {
        "match_counts": match_counts, "singlet_counts": singlet_counts,
        "scalars": scalars, "tau_centers": centers.astype(np.float32),
        "tau_edges": edges.astype(np.float32), "n_tau_bins": int(n_tau_bins),
        "n_pairs": int(n_pairs), "match_run": match_run,
        "gap_to_match": gap_to_match, "gap_to_end": gap_to_end,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", default=os.path.join(TKM, "pfam/precompiled_v3_train"))
    ap.add_argument("--val-shards", default=os.path.join(TKM, "pfam/precompiled_v3_val"))
    ap.add_argument("--n-frag", type=int, default=2)
    ap.add_argument("--n-tau-bins", type=int, default=32)
    ap.add_argument("--n-iter", type=int, default=500)
    ap.add_argument("--max-gap", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tau-floor", type=float, default=0.01,
                    help="drop pairs with t < floor (degenerate near-zero "
                         "branch lengths give NaN BDI bridge stats)")
    ap.add_argument("--max-run", type=int, default=700,
                    help="cap match-run length for the indel run-factor "
                         "(phiM underflows ~830; validated <0.03 nat/pair effect)")
    ap.add_argument("--subst", default="lg")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mixfrag_F2_v3shared.npz"))
    ap.add_argument("--gpu", default="")   # CUDA_VISIBLE_DEVICES for em_fit
    ap.add_argument("--logspace", action="store_true",
                    help="use em_fit_logspace (no run cap; needs TKM_PATH at a "
                         "worktree that has the log-space run-factor)")
    args = ap.parse_args()
    if args.logspace:
        args.max_run = None   # uncapped: log-space run-factor handles long runs
    # Build aggs FIRST (multiprocessing fork) -- before importing JAX, so we
    # never fork a JAX-initialised (multithreaded) process.
    print(f"[1/3] building TRAIN agg from {args.train_shards} (tau_floor={args.tau_floor} max_run={args.max_run})")
    agg = build_agg_from_shards(args.train_shards, args.n_tau_bins, args.workers, args.tau_floor, args.max_run)
    print(f"      train pairs={agg['n_pairs']:,}")
    print(f"[2/3] building VAL agg from {args.val_shards} (tau_floor={args.tau_floor} max_run={args.max_run})")
    vagg = build_agg_from_shards(args.val_shards, args.n_tau_bins, args.workers, args.tau_floor, args.max_run)
    print(f"      val pairs={vagg['n_pairs']:,}")

    if args.gpu:
        os.environ["JAX_PLATFORMS"] = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from tkfmixdom.jax.train.mixfrag_cherry_em import (
        em_fit, prep_bins, corpus_indel_loglik, corpus_substitution_loglik)
    from train_mixfrag_cherry_em import _substitution_model
    Q, pi = _substitution_model(args.subst)
    if args.logspace:
        from tkfmixdom.jax.train.mixfrag_cherry_em import em_fit_logspace

    print(f"[3/3] em_fit MixFrag F={args.n_frag} ({args.n_iter} iters) "
          f"logspace={args.logspace}")
    if args.logspace:
        out = em_fit_logspace(agg, n_fragtypes=args.n_frag, Q=Q, pi=pi,
                              n_iter=args.n_iter, max_gap=args.max_gap, log_every=25)
    else:
        out = em_fit(agg, n_fragtypes=args.n_frag, Q=Q, pi=pi, n_iter=args.n_iter,
                     max_gap=args.max_gap, log_every=25)
    vbins = prep_bins(vagg, max_gap=args.max_gap)
    v_ind = float(corpus_indel_loglik(vbins, out["lam"], out["mu"], out["exts"],
                                      out["weights"], use_logspace=args.logspace))
    v_sub = float(corpus_substitution_loglik(vbins, Q, pi))
    ncv = vagg["n_pairs"]
    res = dict(lam=float(out["lam"]), mu=float(out["mu"]),
               exts=np.asarray(out["exts"]).tolist(),
               weights=np.asarray(out["weights"]).tolist(),
               val_indel_ll=v_ind, val_sub_ll=v_sub, val_ll=v_ind+v_sub,
               val_n_pairs=ncv, train_n_pairs=agg["n_pairs"],
               val_ll_per_pair=(v_ind+v_sub)/ncv,
               val_indel_per_pair=v_ind/ncv, val_sub_per_pair=v_sub/ncv)
    np.savez(args.out, **{k: np.asarray(v) for k, v in res.items()})
    with open(args.out.replace(".npz", ".json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\n==== MixFrag F=%d on v3-shared data ====" % args.n_frag)
    for k in ("lam", "mu", "exts", "weights", "train_n_pairs", "val_n_pairs",
              "val_ll_per_pair", "val_indel_per_pair", "val_sub_per_pair"):
        print(f"  {k} = {res[k]}")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
