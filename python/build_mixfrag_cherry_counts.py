#!/usr/bin/env python3
"""Build MixFrag alignment-summary count tensors per Pfam family.

This is the MixFrag analogue of ``build_tkf92_cherry_counts.py``. Same
cherry-picking / tau-binning / multiprocessing pipeline, but instead of the
TKF92 5x5 transition-count tensor it accumulates the alignment-summary counts of
the MixFrag summarised-training section (supplement "MixFrag Training from
Alignment-Summary Counts", sec:mixfrag-cherrytrain):

For each cherry (anc=seq_a, desc=seq_b) at branch length tau, walk the two
aligned rows, classify each non-empty column as Match / Insert (anc gap,
descendant insertion) / Delete (desc gap, ancestral deletion), and accumulate
PER TAU BIN:
  * match_counts[ai, bi]   -- residue pairs at Match columns (substitution; the
                              N_sub of the section; fragtype-independent).
  * singlet_counts[a]      -- residue composition at Insert/Delete columns.
  * N_match(k)             -- number of maximal Match-runs of length k.
  * N_gap_to_match(i,j)    -- number of gaps (i deletions, j insertions),
                              (i,j)!=(0,0), that EXIT INTO A MATCH (internal and
                              leading gaps; factor G_M(i,j)).
  * N_gap_to_end(i,j)      -- number of gaps that EXIT AT THE ALIGNMENT END
                              (trailing gaps and the match-free whole alignment;
                              factor G_E(i,j)).
  * N_start, N_end         -- per-bin indicators: alignment begins / ends with a
                              match (scalars tau_{S,M}=tau_{M,M} / tau_{M,E}).
  * N_empty                -- empty cherries (no columns; the bare S->E path).

The variable-size tensors N_match, N_gap_* are stored SPARSELY (coordinate
arrays) so there is no length cap and aggregation across families is exact.

Saves one .npz per family. CPU-bound; parallelised across families. Reuses the
(JAX-free) parsing / cherry / tau-bin helpers from build_tkf92_cherry_counts so
the two pipelines stay byte-for-byte consistent on cherry selection and binning.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np

# JAX-free helpers shared with the TKF92 builder (importing this module does NOT
# import jax, so it is safe for spawn/fork multiprocessing workers).
from build_tkf92_cherry_counts import (
    AA, AA_TO_IDX, GAP_CHARS, TAU_MIN, TAU_MAX,
    geom_bin_edges, discretize_tau, parse_stockholm, parse_newick,
    extract_cherries, _family_paths,
)

# scalars[:, 0]=start-with-match, [:, 1]=end-with-match, [:, 2]=empty cherry.
SC_START, SC_END, SC_EMPTY = 0, 1, 2
N_SCALARS = 3


# ---------------------------------------------------------------------------
# Per-cherry counting.
# ---------------------------------------------------------------------------


def count_cherry_mixfrag(
    seq_a: str,
    seq_b: str,
    tau_bin: int,
    match_counts: np.ndarray,
    singlet_counts: np.ndarray,
    scalars: np.ndarray,
    match_run: dict,
    gap_to_match: dict,
    gap_to_end: dict,
) -> tuple[int, int, int]:
    """Update tensors in-place from one cherry. Dense arrays are
    (match_counts, singlet_counts, scalars); the run/gap dicts are keyed by
    (tau_bin, k) resp. (tau_bin, i, j) with i=#deletions, j=#insertions.

    Returns (n_match, n_insert, n_delete) for sanity checking.
    """
    n_m = n_i = n_d = 0
    cur_kind = None            # None / 'M' / 'G'
    cur_k = 0                  # current Match-run length
    cur_del = cur_ins = 0      # current gap deletions (i) / insertions (j)
    first_type = None          # 'M' / 'I' / 'D'
    last_type = None

    def close(followed_by_match: bool):
        nonlocal cur_kind, cur_k, cur_del, cur_ins
        if cur_kind == 'M':
            key = (tau_bin, cur_k)
            match_run[key] = match_run.get(key, 0) + 1
        elif cur_kind == 'G':
            key = (tau_bin, cur_del, cur_ins)          # (bin, i=del, j=ins)
            tgt = gap_to_match if followed_by_match else gap_to_end
            tgt[key] = tgt.get(key, 0) + 1
        cur_kind = None
        cur_k = cur_del = cur_ins = 0

    for ca, cb in zip(seq_a, seq_b):
        ca_gap = ca in GAP_CHARS
        cb_gap = cb in GAP_CHARS
        if ca_gap and cb_gap:
            continue
        if not ca_gap and not cb_gap:
            ai = AA_TO_IDX.get(ca.upper())
            bi = AA_TO_IDX.get(cb.upper())
            if ai is None or bi is None:
                continue
            t = 'M'
        elif ca_gap and not cb_gap:                    # insertion (descendant)
            bi = AA_TO_IDX.get(cb.upper())
            if bi is None:
                continue
            t = 'I'
        else:                                          # deletion (ancestral)
            ai = AA_TO_IDX.get(ca.upper())
            if ai is None:
                continue
            t = 'D'

        if first_type is None:
            first_type = t
        last_type = t

        if t == 'M':
            match_counts[tau_bin, ai, bi] += 1
            n_m += 1
            if cur_kind == 'M':
                cur_k += 1
            else:
                if cur_kind == 'G':
                    close(followed_by_match=True)      # gap is followed by a match
                cur_kind, cur_k = 'M', 1
        else:                                          # gap column (I or D)
            if cur_kind != 'G':
                if cur_kind == 'M':
                    close(followed_by_match=False)     # match-run ends (gap follows)
                cur_kind, cur_del, cur_ins = 'G', 0, 0
            if t == 'I':
                singlet_counts[tau_bin, bi] += 1
                cur_ins += 1
                n_i += 1
            else:
                singlet_counts[tau_bin, ai] += 1
                cur_del += 1
                n_d += 1

    if cur_kind is not None:                           # final segment ends the alignment
        close(followed_by_match=False)

    if first_type is None:
        scalars[tau_bin, SC_EMPTY] += 1
    else:
        if first_type == 'M':
            scalars[tau_bin, SC_START] += 1
        if last_type == 'M':
            scalars[tau_bin, SC_END] += 1

    return n_m, n_i, n_d


# ---------------------------------------------------------------------------
# Sparse-dict <-> coordinate-array helpers (used for save + load + aggregate).
# ---------------------------------------------------------------------------


def _dict_to_coords(d: dict, ncols: int):
    """{(idx tuple): count} -> (idx_array (N, ncols) int32, val_array (N,) int64)."""
    if not d:
        return (np.zeros((0, ncols), np.int32), np.zeros((0,), np.int64))
    idx = np.array(list(d.keys()), dtype=np.int32).reshape(-1, ncols)
    val = np.array(list(d.values()), dtype=np.int64)
    return idx, val


# ---------------------------------------------------------------------------
# Per-family worker.
# ---------------------------------------------------------------------------


def process_family(args) -> dict:
    (family, msa_dir, tree_dir, out_dir, n_tau_bins, max_pairs, resume) = args
    msa_dir, tree_dir, out_dir = Path(msa_dir), Path(tree_dir), Path(out_dir)
    msa_path, tree_path, out_path = _family_paths(family, msa_dir, tree_dir, out_dir)

    if resume and out_path.exists():
        return {"family": family, "status": "skipped_existing"}
    if not msa_path.exists():
        return {"family": family, "status": "missing_msa"}
    if not tree_path.exists():
        return {"family": family, "status": "missing_tree"}

    edges, centers = geom_bin_edges(n_tau_bins, TAU_MIN, TAU_MAX)

    try:
        seqs = parse_stockholm(str(msa_path))
    except Exception as e:
        return {"family": family, "status": "parse_msa_error", "error": str(e)}
    if len(seqs) < 2:
        return {"family": family, "status": "msa_too_small"}

    try:
        with open(tree_path) as f:
            root = parse_newick(f.read())
        cherries = extract_cherries(root)
    except Exception as e:
        return {"family": family, "status": "parse_tree_error", "error": str(e)}
    if not cherries:
        return {"family": family, "status": "no_cherries"}
    if max_pairs is not None:
        cherries = cherries[:max_pairs]              # smallest-tau first

    match_counts = np.zeros((n_tau_bins, AA, AA), dtype=np.int64)
    singlet_counts = np.zeros((n_tau_bins, AA), dtype=np.int64)
    scalars = np.zeros((n_tau_bins, N_SCALARS), dtype=np.int64)
    match_run: dict = {}
    gap_to_match: dict = {}
    gap_to_end: dict = {}

    n_used = n_skipped = 0
    tot_m = tot_i = tot_d = 0
    for name_a, name_b, tau in cherries:
        sa, sb = seqs.get(name_a), seqs.get(name_b)
        if sa is None or sb is None or len(sa) != len(sb):
            n_skipped += 1
            continue
        if tau <= 0.0 or not np.isfinite(tau):
            n_skipped += 1
            continue
        tb = int(discretize_tau(tau, edges))
        nm, ni, nd = count_cherry_mixfrag(
            sa, sb, tb, match_counts, singlet_counts, scalars,
            match_run, gap_to_match, gap_to_end)
        tot_m += nm
        tot_i += ni
        tot_d += nd
        n_used += 1

    if n_used == 0:
        return {"family": family, "status": "no_usable_cherries"}

    mr_idx, mr_val = _dict_to_coords(match_run, 2)
    gm_idx, gm_val = _dict_to_coords(gap_to_match, 3)
    ge_idx, ge_val = _dict_to_coords(gap_to_end, 3)
    np.savez(
        out_path,
        match_counts=match_counts.astype(np.int32),
        singlet_counts=singlet_counts.astype(np.int32),
        scalars=scalars.astype(np.int64),
        match_run_idx=mr_idx, match_run_val=mr_val,
        gap_to_match_idx=gm_idx, gap_to_match_val=gm_val,
        gap_to_end_idx=ge_idx, gap_to_end_val=ge_val,
        tau_centers=centers.astype(np.float32),
        tau_edges=edges.astype(np.float32),
        n_tau_bins=np.int64(n_tau_bins),
        n_pairs=np.int64(n_used),
        family=np.array(family),
    )
    return {"family": family, "status": "ok", "n_pairs": n_used,
            "n_skipped": n_skipped, "n_match": tot_m,
            "n_insert": tot_i, "n_delete": tot_d}


# ---------------------------------------------------------------------------
# Load + aggregate helpers (for the training driver / tests; JAX-free).
# ---------------------------------------------------------------------------


def load_family_counts(path) -> dict:
    """Load one per-family .npz into a dict with dense arrays and the run/gap
    counts re-expanded into {(bin, k): v} / {(bin, i, j): v} dicts."""
    z = np.load(path, allow_pickle=True)
    out = {
        "match_counts": z["match_counts"],
        "singlet_counts": z["singlet_counts"],
        "scalars": z["scalars"],
        "tau_centers": z["tau_centers"],
        "tau_edges": z["tau_edges"],
        "n_tau_bins": int(z["n_tau_bins"]),
        "n_pairs": int(z["n_pairs"]),
    }
    out["match_run"] = {tuple(int(x) for x in k): int(v)
                        for k, v in zip(z["match_run_idx"], z["match_run_val"])}
    out["gap_to_match"] = {tuple(int(x) for x in k): int(v)
                           for k, v in zip(z["gap_to_match_idx"], z["gap_to_match_val"])}
    out["gap_to_end"] = {tuple(int(x) for x in k): int(v)
                         for k, v in zip(z["gap_to_end_idx"], z["gap_to_end_val"])}
    return out


def aggregate_counts(paths) -> dict:
    """Sum per-family count tensors into one corpus-level set (dense arrays
    summed, sparse dicts merged). All families must share n_tau_bins / edges."""
    agg = None
    for p in paths:
        f = load_family_counts(p)
        if agg is None:
            agg = {
                "match_counts": f["match_counts"].astype(np.int64).copy(),
                "singlet_counts": f["singlet_counts"].astype(np.int64).copy(),
                "scalars": f["scalars"].astype(np.int64).copy(),
                "tau_centers": f["tau_centers"], "tau_edges": f["tau_edges"],
                "n_tau_bins": f["n_tau_bins"], "n_pairs": 0,
                "match_run": {}, "gap_to_match": {}, "gap_to_end": {},
            }
        agg["match_counts"] += f["match_counts"]
        agg["singlet_counts"] += f["singlet_counts"]
        agg["scalars"] += f["scalars"]
        agg["n_pairs"] += f["n_pairs"]
        for name in ("match_run", "gap_to_match", "gap_to_end"):
            for k, v in f[name].items():
                agg[name][k] = agg[name].get(k, 0) + v
    return agg


# ---------------------------------------------------------------------------
# CLI driver (mirrors build_tkf92_cherry_counts).
# ---------------------------------------------------------------------------


def _resolve_families(args) -> list[str]:
    if args.families:
        return [f.strip() for f in args.families.split(",") if f.strip()]
    if args.split_file is not None and args.split is not None:
        with open(args.split_file) as f:
            data = json.load(f)
        return list(data[args.split])
    msa_dir = Path(args.msa_dir).expanduser()
    fams = []
    for p in sorted(msa_dir.iterdir()):
        if p.suffix == ".sto" or p.name.endswith(".sto.gz"):
            fams.append(p.name.split(".")[0])
    return fams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--msa-dir", default="~/bio-datasets/data/pfam/seed")
    p.add_argument("--tree-dir", default="~/bio-datasets/data/pfam/trees")
    p.add_argument("--out-dir",
                   default="~/tkf-mixdom/python/pfam/cherries_mixfrag")
    p.add_argument("--n-tau-bins", type=int, default=32)
    p.add_argument("--max-pairs-per-fam", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--families", default=None,
                   help="Comma-separated family accessions (overrides --split).")
    p.add_argument("--split", default=None,
                   help="Split key inside --split-file (e.g. 'train').")
    p.add_argument("--split-file", default=None,
                   help="Path to a splits JSON.")
    p.add_argument("--no-resume", action="store_true",
                   help="Reprocess families even if output .npz already exists.")
    p.add_argument("--progress-every", type=int, default=200)
    args = p.parse_args()

    msa_dir = Path(os.path.expanduser(args.msa_dir))
    tree_dir = Path(os.path.expanduser(args.tree_dir))
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    families = _resolve_families(args)
    print(f"[build_mixfrag_cherry_counts] families: {len(families)}; "
          f"msa_dir={msa_dir} tree_dir={tree_dir} out_dir={out_dir}", flush=True)
    print(f"[build_mixfrag_cherry_counts] workers={args.workers} "
          f"n_tau_bins={args.n_tau_bins} "
          f"max_pairs_per_fam={args.max_pairs_per_fam} "
          f"resume={not args.no_resume}", flush=True)

    work = [(fam, str(msa_dir), str(tree_dir), str(out_dir),
             args.n_tau_bins, args.max_pairs_per_fam, not args.no_resume)
            for fam in families]

    t0 = time.time()
    status_counts: dict = {}
    total_pairs = 0

    def _accumulate(res):
        nonlocal total_pairs
        status_counts[res["status"]] = status_counts.get(res["status"], 0) + 1
        if res["status"] == "ok":
            total_pairs += res.get("n_pairs", 0)

    def _progress(i):
        el = time.time() - t0
        rate = i / el if el > 0 else 0.0
        eta = (len(work) - i) / rate if rate > 0 else float("inf")
        print(f"[progress] {i}/{len(work)} fams ({rate:.1f}/s, "
              f"ETA {eta/60:.1f} min) pairs={total_pairs} status={status_counts}",
              flush=True)

    if args.workers <= 1:
        for i, w in enumerate(work, 1):
            _accumulate(process_family(w))
            if i % args.progress_every == 0 or i == len(work):
                _progress(i)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for i, res in enumerate(
                    pool.imap_unordered(process_family, work, chunksize=8), 1):
                _accumulate(res)
                if i % args.progress_every == 0 or i == len(work):
                    _progress(i)

    elapsed = time.time() - t0
    print(f"\n[done] wallclock={elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
    print(f"[done] total cherries: {total_pairs}; status={status_counts}",
          flush=True)


if __name__ == "__main__":
    main()
