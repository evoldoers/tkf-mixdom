#!/usr/bin/env python3
"""Build TKF92 Pair-HMM cherry sufficient-statistic tensors per Pfam family.

For each Pfam family with both an MSA (.sto) and a tree (.nwk):
  1. Iteratively pick cherries from the tree, smallest combined-branch-length
     first; remove and repeat (CherryML / Prillo 2022 nearest-neighbor scheme).
  2. tau_pair = sum of the two leaf-to-parent branch lengths.
  3. Walk the two aligned rows; classify each non-empty column as Match,
     Insert (anc gap), or Delete (desc gap); accumulate match/singlet counts
     and Pair-HMM transition counts (Start/M/I/D/End indexing) per tau bin.
  4. Save .npz per family with match_counts, singlet_counts,
     transition_counts, tau_centers, tau_edges, n_pairs, family.

CPU-bound; parallelised across families via multiprocessing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np

# Reuse the small set of constants/utilities we need from maraschino, without
# pulling JAX into worker processes (importing maraschino imports jax, which
# is incompatible with fork-based multiprocessing). Define local copies.

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA = len(AMINO_ACIDS)
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}
TAU_MIN = 0.001
TAU_MAX = 10.0


def geom_bin_edges(n_bins: int, tau_min: float = TAU_MIN, tau_max: float = TAU_MAX):
    """Geometric bin edges and (geometric-mean) centres. Matches maraschino.py."""
    edges = np.geomspace(tau_min, tau_max, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def discretize_tau(tau: float, edges: np.ndarray) -> int:
    """Map continuous tau to nearest bin index. Matches maraschino.py."""
    idx = int(np.searchsorted(edges, tau)) - 1
    return max(0, min(idx, len(edges) - 2))


def parse_stockholm(filepath: str) -> dict:
    """Parse a Stockholm alignment (plain or .gz). Matches maraschino.py."""
    import gzip
    seqs: dict = {}
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                name, seq = parts[0], parts[1]
                if name in seqs:
                    seqs[name] += seq
                else:
                    seqs[name] = seq
    return seqs

# TKF92 Pair-HMM state indexing (per spec).
S_START = 0
S_MATCH = 1
S_INSERT = 2
S_DELETE = 3
S_END = 4
N_STATES = 5

GAP_CHARS = {"-", "."}


# ---------------------------------------------------------------------------
# Newick parsing (just enough for Pfam tree files).
# ---------------------------------------------------------------------------


class _Node:
    __slots__ = ("name", "branch_length", "children", "parent")

    def __init__(self, name="", branch_length=0.0):
        self.name = name
        self.branch_length = branch_length
        self.children: list[_Node] = []
        self.parent: _Node | None = None

    def is_leaf(self):
        return not self.children


def parse_newick(s: str) -> _Node:
    """Parse a Newick string to a tree of _Node. Supports leaf names containing
    letters, digits, underscores, slashes, hyphens, dots, asterisks; branch
    lengths after ':'; no quoting/comments. Internal node labels permitted but
    unused.
    """
    s = s.strip()
    if s.endswith(";"):
        s = s[:-1]
    pos = [0]

    def parse_subtree() -> _Node:
        node = _Node()
        if pos[0] < len(s) and s[pos[0]] == "(":
            pos[0] += 1  # consume '('
            while True:
                child = parse_subtree()
                child.parent = node
                node.children.append(child)
                if pos[0] >= len(s):
                    break
                if s[pos[0]] == ",":
                    pos[0] += 1
                    continue
                if s[pos[0]] == ")":
                    pos[0] += 1
                    break
        # Optional name.
        name_start = pos[0]
        while pos[0] < len(s) and s[pos[0]] not in ":,()":
            pos[0] += 1
        node.name = s[name_start:pos[0]]
        # Optional branch length.
        if pos[0] < len(s) and s[pos[0]] == ":":
            pos[0] += 1
            bl_start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in ",()":
                pos[0] += 1
            try:
                node.branch_length = float(s[bl_start:pos[0]])
            except ValueError:
                node.branch_length = 0.0
        return node

    return parse_subtree()


# ---------------------------------------------------------------------------
# Cherry picking via iterative pruning.
# ---------------------------------------------------------------------------


def extract_cherries(root: _Node) -> list[tuple[str, str, float]]:
    """Iteratively pick cherries (two-leaf siblings) by smallest combined
    branch length, prune both leaves (parent becomes a leaf), and repeat.

    Returns list of (leaf_name_a, leaf_name_b, tau) where
    tau = bl_a_to_parent + bl_b_to_parent.
    """
    cherries: list[tuple[str, str, float]] = []

    while True:
        # Find all cherries: parents whose children are all leaves AND there
        # are exactly two such children. (Multifurcations don't form cherries.)
        # Also handle a parent with >2 leaf children by treating the two
        # closest leaves as a cherry — but for safety we restrict to the
        # standard case (exactly two leaf children).
        candidates: list[tuple[float, _Node]] = []  # (combined_bl, parent)

        # Walk all nodes once; iterative DFS to avoid recursion depth issues.
        stack = [root]
        while stack:
            node = stack.pop()
            if node.children:
                leaf_kids = [c for c in node.children if c.is_leaf()]
                if len(node.children) == 2 and len(leaf_kids) == 2:
                    bl = leaf_kids[0].branch_length + leaf_kids[1].branch_length
                    candidates.append((bl, node))
                stack.extend(node.children)

        if not candidates:
            # Multifurcating root with all-leaf children: pick the closest
            # pair as a cherry, remove them.
            if root.children and all(c.is_leaf() for c in root.children) and len(root.children) >= 2:
                pairs = []
                ch = root.children
                for i in range(len(ch)):
                    for j in range(i + 1, len(ch)):
                        pairs.append(
                            (ch[i].branch_length + ch[j].branch_length, i, j)
                        )
                if pairs:
                    pairs.sort(key=lambda x: x[0])
                    bl, i, j = pairs[0]
                    cherries.append((ch[i].name, ch[j].name, bl))
                    # Remove these two children.
                    new_children = [c for k, c in enumerate(ch) if k != i and k != j]
                    root.children = new_children
                    if len(root.children) <= 1:
                        break
                    continue
            break

        # Pick smallest combined-branch-length cherry.
        candidates.sort(key=lambda x: x[0])
        bl, parent = candidates[0]
        a, b = parent.children[0], parent.children[1]
        cherries.append((a.name, b.name, bl))
        # Convert parent into a leaf: drop its children, keep its branch_length.
        parent.children = []
        parent.name = parent.name or f"_internal_{id(parent)}"
        # Special case: if parent IS root and now has no children, we're done.
        if parent is root:
            break

    return cherries


# ---------------------------------------------------------------------------
# Per-cherry counting.
# ---------------------------------------------------------------------------


def count_cherry(
    seq_a: str,
    seq_b: str,
    tau_bin: int,
    match_counts: np.ndarray,
    singlet_counts: np.ndarray,
    transition_counts: np.ndarray,
) -> tuple[int, int, int]:
    """Update count tensors in-place from a single cherry (anc=seq_a, desc=seq_b).
    Returns (n_match, n_insert, n_delete) for sanity checking.
    """
    n_m = n_i = n_d = 0
    prev = S_START

    # zip handles equal-length aligned rows; if lengths differ (shouldn't for
    # a proper Stockholm MSA), zip stops at the shorter one.
    for ca, cb in zip(seq_a, seq_b):
        ca_gap = ca in GAP_CHARS
        cb_gap = cb in GAP_CHARS
        if ca_gap and cb_gap:
            continue
        if not ca_gap and not cb_gap:
            ai = AA_TO_IDX.get(ca.upper())
            bi = AA_TO_IDX.get(cb.upper())
            if ai is None or bi is None:
                # Non-canonical AA in a match column — skip.
                continue
            match_counts[tau_bin, ai, bi] += 1
            transition_counts[tau_bin, prev, S_MATCH] += 1
            prev = S_MATCH
            n_m += 1
        elif ca_gap and not cb_gap:
            bi = AA_TO_IDX.get(cb.upper())
            if bi is None:
                continue
            singlet_counts[tau_bin, bi] += 1
            transition_counts[tau_bin, prev, S_INSERT] += 1
            prev = S_INSERT
            n_i += 1
        else:  # cb_gap and not ca_gap
            ai = AA_TO_IDX.get(ca.upper())
            if ai is None:
                continue
            singlet_counts[tau_bin, ai] += 1
            transition_counts[tau_bin, prev, S_DELETE] += 1
            prev = S_DELETE
            n_d += 1

    # Close with End.
    transition_counts[tau_bin, prev, S_END] += 1
    return n_m, n_i, n_d


# ---------------------------------------------------------------------------
# Per-family worker.
# ---------------------------------------------------------------------------


def _family_paths(family: str, msa_dir: Path, tree_dir: Path, out_dir: Path):
    msa = msa_dir / f"{family}.sto"
    if not msa.exists():
        msa_gz = msa_dir / f"{family}.sto.gz"
        if msa_gz.exists():
            msa = msa_gz
    tree = tree_dir / f"{family}.nwk"
    out = out_dir / f"{family}.npz"
    return msa, tree, out


def process_family(args) -> dict:
    """Worker entry point. args is a tuple to keep multiprocessing happy."""
    (family, msa_dir, tree_dir, out_dir, n_tau_bins,
     max_pairs, resume) = args
    msa_dir = Path(msa_dir)
    tree_dir = Path(tree_dir)
    out_dir = Path(out_dir)

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
            tree_str = f.read()
        root = parse_newick(tree_str)
        cherries = extract_cherries(root)
    except Exception as e:
        return {"family": family, "status": "parse_tree_error", "error": str(e)}

    if not cherries:
        return {"family": family, "status": "no_cherries"}

    if max_pairs is not None:
        # Smallest-tau cherries first (already produced in that order).
        cherries = cherries[:max_pairs]

    match_counts = np.zeros((n_tau_bins, AA, AA), dtype=np.int64)
    singlet_counts = np.zeros((n_tau_bins, AA), dtype=np.int64)
    transition_counts = np.zeros((n_tau_bins, N_STATES, N_STATES), dtype=np.int64)

    n_used = 0
    n_skipped_unknown = 0
    tot_m = tot_i = tot_d = 0

    for name_a, name_b, tau in cherries:
        sa = seqs.get(name_a)
        sb = seqs.get(name_b)
        if sa is None or sb is None:
            n_skipped_unknown += 1
            continue
        if len(sa) != len(sb):
            n_skipped_unknown += 1
            continue
        if tau <= 0.0 or not np.isfinite(tau):
            n_skipped_unknown += 1
            continue
        tb = int(discretize_tau(tau, edges))
        nm, ni, nd = count_cherry(
            sa, sb, tb, match_counts, singlet_counts, transition_counts
        )
        tot_m += nm
        tot_i += ni
        tot_d += nd
        n_used += 1

    if n_used == 0:
        return {"family": family, "status": "no_usable_cherries"}

    # Cast to int32 for output (per spec).
    np.savez(
        out_path,
        match_counts=match_counts.astype(np.int32),
        singlet_counts=singlet_counts.astype(np.int32),
        transition_counts=transition_counts.astype(np.int32),
        tau_centers=centers.astype(np.float32),
        tau_edges=edges.astype(np.float32),
        n_pairs=np.int64(n_used),
        family=np.array(family),
    )
    return {
        "family": family,
        "status": "ok",
        "n_pairs": n_used,
        "n_skipped": n_skipped_unknown,
        "n_match": tot_m,
        "n_insert": tot_i,
        "n_delete": tot_d,
    }


# ---------------------------------------------------------------------------
# CLI driver.
# ---------------------------------------------------------------------------


def _resolve_families(args) -> list[str]:
    if args.families:
        return [f.strip() for f in args.families.split(",") if f.strip()]
    if args.split_file is not None and args.split is not None:
        with open(args.split_file) as f:
            data = json.load(f)
        return list(data[args.split])
    # Default: every family for which an MSA exists in msa_dir.
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
    p.add_argument("--out-dir", default="~/tkf-mixdom/python/pfam/cherries_tkf92")
    p.add_argument("--n-tau-bins", type=int, default=32)
    p.add_argument("--max-pairs-per-fam", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--families", default=None,
                   help="Comma-separated list of family accessions (overrides --split).")
    p.add_argument("--split", default=None,
                   help="Split key inside --split-file (e.g. 'train', 'val', 'test').")
    p.add_argument("--split-file", default=None,
                   help="Path to a splits JSON (e.g. .../seed/splits/v1.json).")
    p.add_argument("--no-resume", action="store_true",
                   help="Reprocess families even if output .npz already exists.")
    p.add_argument("--progress-every", type=int, default=200,
                   help="Log a progress line every N families.")
    args = p.parse_args()

    msa_dir = Path(os.path.expanduser(args.msa_dir))
    tree_dir = Path(os.path.expanduser(args.tree_dir))
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    families = _resolve_families(args)
    print(f"[build_tkf92_cherry_counts] families to process: {len(families)}",
          flush=True)
    print(f"[build_tkf92_cherry_counts] msa_dir={msa_dir}", flush=True)
    print(f"[build_tkf92_cherry_counts] tree_dir={tree_dir}", flush=True)
    print(f"[build_tkf92_cherry_counts] out_dir={out_dir}", flush=True)
    print(f"[build_tkf92_cherry_counts] workers={args.workers}, "
          f"n_tau_bins={args.n_tau_bins}, "
          f"max_pairs_per_fam={args.max_pairs_per_fam}, "
          f"resume={not args.no_resume}", flush=True)

    work = [
        (fam, str(msa_dir), str(tree_dir), str(out_dir),
         args.n_tau_bins, args.max_pairs_per_fam, not args.no_resume)
        for fam in families
    ]

    t0 = time.time()
    status_counts: dict[str, int] = {}
    total_pairs = 0

    def _accumulate(res):
        nonlocal total_pairs
        status_counts[res["status"]] = status_counts.get(res["status"], 0) + 1
        if res["status"] == "ok":
            total_pairs += res.get("n_pairs", 0)

    if args.workers <= 1:
        for i, w in enumerate(work, 1):
            res = process_family(w)
            _accumulate(res)
            if i % args.progress_every == 0 or i == len(work):
                el = time.time() - t0
                rate = i / el if el > 0 else 0.0
                eta = (len(work) - i) / rate if rate > 0 else float("inf")
                print(f"[progress] {i}/{len(work)} fams "
                      f"({rate:.1f}/s, ETA {eta/60:.1f} min) "
                      f"pairs={total_pairs} status={status_counts}",
                      flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for i, res in enumerate(
                pool.imap_unordered(process_family, work, chunksize=8), 1
            ):
                _accumulate(res)
                if i % args.progress_every == 0 or i == len(work):
                    el = time.time() - t0
                    rate = i / el if el > 0 else 0.0
                    eta = (len(work) - i) / rate if rate > 0 else float("inf")
                    print(f"[progress] {i}/{len(work)} fams "
                          f"({rate:.1f}/s, ETA {eta/60:.1f} min) "
                          f"pairs={total_pairs} status={status_counts}",
                          flush=True)

    elapsed = time.time() - t0
    print(f"\n[done] wallclock={elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
    print(f"[done] total cherries (pairs): {total_pairs}", flush=True)
    print(f"[done] status: {status_counts}", flush=True)


if __name__ == "__main__":
    main()
