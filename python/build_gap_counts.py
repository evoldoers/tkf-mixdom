#!/usr/bin/env python3
"""Build a Pfam corpus-wide GAP-COUNTS tensor for the conditional-LL fit.

Output: a single .npz with
  gap_counts : shape (n_tau, 4, Lmax+1, Lmax+1), gap-type axis = (SM, MM, ME, SE)
  trans_counts : shape (n_tau, 5, 5) -- the same per-tau Pair HMM transition
                 counts as build_tkf92_cherry_counts produces, used for the
                 ancestor singlet log-marginal
  tau_edges, tau_centers : geomspace bin edges / centres
  n_cherries_per_bin : (n_tau,) -- diagnostic
  meta : a dict with {n_families, n_skipped, Lmax, ...}

Mirrors build_tkf92_cherry_counts.py for the cherry / tau-bin pipeline; the
only addition is the gap-tally on the (M, I, D) column sequence.

Designed to run on a machine that has the Pfam corpus loaded
(~/bio-datasets/data/pfam/seed/{PF*.sto.gz} + ~/bio-datasets/data/pfam-seed/
trees/{PF*.nwk}).  CPU-bound; parallelised across families.
"""
from __future__ import annotations

import argparse
import os
import gzip
import time
import multiprocessing as mp
import json
from pathlib import Path

import numpy as np


# Constants - keep in sync with build_tkf92_cherry_counts.py
GAP_CHARS = set("-.~")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA = len(AMINO_ACIDS)
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}

S_START, S_MATCH, S_INSERT, S_DELETE, S_END = 0, 1, 2, 3, 4
M_, I_, D_ = 1, 2, 3  # alias for column types in gap tally

# Gap types
GAP_SM, GAP_MM, GAP_ME, GAP_SE = 0, 1, 2, 3

TAU_MIN = 0.001
TAU_MAX = 10.0
DEFAULT_N_BINS = 32
DEFAULT_LMAX = 50


# ---------------------------------------------------------------------------
# Geometric bin helpers (copied from build_tkf92_cherry_counts.py)
# ---------------------------------------------------------------------------

def geom_bin_edges(n_bins: int, tau_min: float = TAU_MIN, tau_max: float = TAU_MAX):
    edges = np.geomspace(tau_min, tau_max, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def discretize_tau(tau: float, edges: np.ndarray) -> int:
    idx = int(np.searchsorted(edges, tau)) - 1
    return max(0, min(idx, len(edges) - 2))


# ---------------------------------------------------------------------------
# Stockholm parsing (small subset)
# ---------------------------------------------------------------------------

def parse_stockholm(filepath: str) -> dict:
    if filepath.endswith(".gz"):
        fh = gzip.open(filepath, "rt")
    else:
        fh = open(filepath, "rt")
    seqs = {}
    try:
        for line in fh:
            s = line.rstrip("\n")
            if not s or s.startswith("#") or s == "//":
                continue
            parts = s.split(None, 1)
            if len(parts) != 2:
                continue
            name, seq = parts
            seqs[name] = seqs.get(name, "") + seq
    finally:
        fh.close()
    return seqs


# ---------------------------------------------------------------------------
# Newick parsing + cherry extraction (CherryML / Prillo 2022)
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("name", "branch_length", "children", "parent")

    def __init__(self, name="", branch_length=0.0):
        self.name = name
        self.branch_length = branch_length
        self.children: list[_Node] = []
        self.parent: _Node | None = None

    def is_leaf(self) -> bool:
        return not self.children


def parse_newick(s: str) -> _Node:
    pos = [0]

    def get_char():
        return s[pos[0]] if pos[0] < len(s) else None

    def consume():
        c = get_char()
        pos[0] += 1
        return c

    def parse_subtree() -> _Node:
        node = _Node()
        if get_char() == "(":
            consume()
            node.children.append(parse_subtree())
            node.children[-1].parent = node
            while get_char() == ",":
                consume()
                child = parse_subtree()
                child.parent = node
                node.children.append(child)
            if get_char() == ")":
                consume()
        # label
        label_chars = []
        while get_char() not in (None, ",", ")", ":", ";"):
            label_chars.append(consume())
        node.name = "".join(label_chars)
        if get_char() == ":":
            consume()
            bl_chars = []
            while get_char() not in (None, ",", ")", ";"):
                bl_chars.append(consume())
            try:
                node.branch_length = float("".join(bl_chars))
            except ValueError:
                node.branch_length = 0.0
        return node

    root = parse_subtree()
    return root


def extract_cherries(root: _Node) -> list[tuple[str, str, float]]:
    """Iteratively pick cherries: pairs of sibling leaves whose combined
    branch length is smallest.  Remove and repeat.  Returns list of
    (leaf_a_name, leaf_b_name, tau) where tau = bl(a) + bl(b).
    """
    cherries = []
    # find all sibling-leaf pairs
    while True:
        # collect candidate cherries (pairs of leaf siblings)
        candidates = []
        stack = [root]
        while stack:
            node = stack.pop()
            for c in node.children:
                stack.append(c)
            leaf_children = [c for c in node.children if c.is_leaf()]
            for i in range(len(leaf_children)):
                for j in range(i + 1, len(leaf_children)):
                    a, b = leaf_children[i], leaf_children[j]
                    tau = (a.branch_length or 0.0) + (b.branch_length or 0.0)
                    candidates.append((tau, a, b, node))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        tau, a, b, parent = candidates[0]
        cherries.append((a.name, b.name, tau))
        # remove both leaves from parent; if parent now has 1 child collapse it
        parent.children = [c for c in parent.children if c is not a and c is not b]
        # leaf collapse to keep tree well-formed
        if len(parent.children) == 1 and parent.parent is not None:
            only = parent.children[0]
            # attach `only` directly to parent.parent
            only.branch_length = (only.branch_length or 0.0) + (parent.branch_length or 0.0)
            only.parent = parent.parent
            parent.parent.children = [
                only if c is parent else c for c in parent.parent.children]
        elif not parent.children and parent.parent is not None:
            parent.parent.children = [c for c in parent.parent.children if c is not parent]
    return cherries


# ---------------------------------------------------------------------------
# Per-cherry gap-counts tally
# ---------------------------------------------------------------------------

def cherry_col_seq(seq_a: str, seq_b: str) -> list[int]:
    """Return the column type sequence ({M=1, I=2, D=3}) for one (anc=seq_a,
    desc=seq_b) cherry.  Skip aa-pair-gap columns (both gapped)."""
    out = []
    for ca, cb in zip(seq_a, seq_b):
        ca_gap = ca in GAP_CHARS
        cb_gap = cb in GAP_CHARS
        if ca_gap and cb_gap:
            continue
        if (not ca_gap) and (not cb_gap):
            ai = AA_TO_IDX.get(ca.upper())
            bi = AA_TO_IDX.get(cb.upper())
            if ai is None or bi is None:
                continue
            out.append(M_)
        elif ca_gap and (not cb_gap):
            bi = AA_TO_IDX.get(cb.upper())
            if bi is None:
                continue
            out.append(I_)
        else:
            ai = AA_TO_IDX.get(ca.upper())
            if ai is None:
                continue
            out.append(D_)
    return out


def tally_gaps(col_seq: list[int], tau_bin: int,
               gap_counts: np.ndarray, Lmax: int) -> None:
    """Update gap_counts[tau_bin, gap_type, i, j] in place from one cherry."""
    match_positions = [k for k, c in enumerate(col_seq) if c == M_]
    if not match_positions:
        # SE gap covering everything
        i_del = sum(1 for c in col_seq if c == D_)
        j_ins = sum(1 for c in col_seq if c == I_)
        gap_counts[tau_bin, GAP_SE, min(i_del, Lmax), min(j_ins, Lmax)] += 1
        return
    first_m = match_positions[0]
    last_m = match_positions[-1]
    # SM
    i_del = sum(1 for c in col_seq[:first_m] if c == D_)
    j_ins = sum(1 for c in col_seq[:first_m] if c == I_)
    gap_counts[tau_bin, GAP_SM, min(i_del, Lmax), min(j_ins, Lmax)] += 1
    # MM (between consecutive matches)
    for k in range(len(match_positions) - 1):
        left = match_positions[k]
        right = match_positions[k + 1]
        i_del = sum(1 for c in col_seq[left + 1:right] if c == D_)
        j_ins = sum(1 for c in col_seq[left + 1:right] if c == I_)
        gap_counts[tau_bin, GAP_MM, min(i_del, Lmax), min(j_ins, Lmax)] += 1
    # ME
    i_del = sum(1 for c in col_seq[last_m + 1:] if c == D_)
    j_ins = sum(1 for c in col_seq[last_m + 1:] if c == I_)
    gap_counts[tau_bin, GAP_ME, min(i_del, Lmax), min(j_ins, Lmax)] += 1


def tally_transitions(col_seq: list[int], tau_bin: int,
                       trans_counts: np.ndarray) -> None:
    """Standard 5x5 Pair-HMM transition tally for sanity / singlet computation."""
    prev = S_START
    for c in col_seq:
        if c == M_:
            trans_counts[tau_bin, prev, S_MATCH] += 1
            prev = S_MATCH
        elif c == I_:
            trans_counts[tau_bin, prev, S_INSERT] += 1
            prev = S_INSERT
        elif c == D_:
            trans_counts[tau_bin, prev, S_DELETE] += 1
            prev = S_DELETE
    trans_counts[tau_bin, prev, S_END] += 1


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def process_family(args):
    (family, msa_path, tree_path, n_bins, Lmax, tau_min, tau_max) = args
    try:
        if not os.path.exists(msa_path) or not os.path.exists(tree_path):
            return dict(family=family, ok=False, reason="missing")
        edges, _ = geom_bin_edges(n_bins, tau_min, tau_max)
        seqs = parse_stockholm(msa_path)
        if not seqs:
            return dict(family=family, ok=False, reason="empty_msa")
        with open(tree_path) as fh:
            root = parse_newick(fh.read().strip())
        cherries = extract_cherries(root)
        gap_counts = np.zeros((n_bins, 4, Lmax + 1, Lmax + 1), dtype=np.int64)
        trans_counts = np.zeros((n_bins, 5, 5), dtype=np.int64)
        n_cherries_per_bin = np.zeros(n_bins, dtype=np.int64)
        used = 0
        skipped = 0
        for name_a, name_b, tau in cherries:
            sa = seqs.get(name_a)
            sb = seqs.get(name_b)
            if sa is None or sb is None or tau <= 0:
                skipped += 1
                continue
            ti = discretize_tau(tau, edges)
            col_seq = cherry_col_seq(sa, sb)
            if not col_seq:
                skipped += 1
                continue
            tally_gaps(col_seq, ti, gap_counts, Lmax)
            tally_transitions(col_seq, ti, trans_counts)
            n_cherries_per_bin[ti] += 1
            used += 1
        return dict(family=family, ok=True, used=used, skipped=skipped,
                    gap_counts=gap_counts, trans_counts=trans_counts,
                    n_cherries_per_bin=n_cherries_per_bin)
    except Exception as ex:
        return dict(family=family, ok=False, reason=str(ex)[:200])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--families-json', required=True,
                        help='Path to a JSON file with a list of family names '
                             '(or a list of {"family": ...} entries).')
    parser.add_argument('--msa-dir', required=True,
                        help='Directory of PFxxxxx.sto / .sto.gz files')
    parser.add_argument('--tree-dir', required=True,
                        help='Directory of PFxxxxx.nwk files')
    parser.add_argument('--out', required=True,
                        help='Output .npz path')
    parser.add_argument('--n-bins', type=int, default=DEFAULT_N_BINS)
    parser.add_argument('--Lmax', type=int, default=DEFAULT_LMAX)
    parser.add_argument('--tau-min', type=float, default=TAU_MIN)
    parser.add_argument('--tau-max', type=float, default=TAU_MAX)
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most this many families (0 = all)')
    args = parser.parse_args()

    fams_data = json.load(open(args.families_json))
    if isinstance(fams_data, dict) and 'families' in fams_data:
        fams = [f['family'] if isinstance(f, dict) else f
                for f in fams_data['families']]
    elif isinstance(fams_data, list):
        fams = [f['family'] if isinstance(f, dict) else f for f in fams_data]
    else:
        raise ValueError(f"Unexpected families JSON structure")
    if args.limit > 0:
        fams = fams[:args.limit]

    msa_dir = Path(os.path.expanduser(args.msa_dir))
    tree_dir = Path(os.path.expanduser(args.tree_dir))

    work_items = []
    for fam in fams:
        msa = msa_dir / f"{fam}.sto"
        if not msa.exists():
            msa = msa_dir / f"{fam}.sto.gz"
        tree = tree_dir / f"{fam}.nwk"
        work_items.append((fam, str(msa), str(tree),
                           args.n_bins, args.Lmax, args.tau_min, args.tau_max))

    print(f"Built {len(work_items)} work items.  Workers = {args.workers}.")
    t0 = time.monotonic()
    agg_gap = np.zeros((args.n_bins, 4, args.Lmax + 1, args.Lmax + 1),
                        dtype=np.int64)
    agg_trans = np.zeros((args.n_bins, 5, 5), dtype=np.int64)
    agg_n_cherries = np.zeros(args.n_bins, dtype=np.int64)
    n_ok = 0
    n_skipped = 0
    n_failed = 0
    failed_reasons = {}

    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_family, work_items, chunksize=4)):
            if res['ok']:
                agg_gap += res['gap_counts']
                agg_trans += res['trans_counts']
                agg_n_cherries += res['n_cherries_per_bin']
                n_ok += 1
                n_skipped += res['skipped']
            else:
                n_failed += 1
                failed_reasons[res.get('reason', 'unknown')] = (
                    failed_reasons.get(res.get('reason', 'unknown'), 0) + 1)
            if (i + 1) % 100 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed
                print(f"  processed {i+1}/{len(work_items)} families  "
                      f"({rate:.1f}/s, ok={n_ok}, failed={n_failed})")

    elapsed = time.monotonic() - t0
    print(f"\nTotal processing: {elapsed:.1f}s")
    print(f"  ok: {n_ok}, failed: {n_failed}, skipped cherries: {n_skipped}")
    print(f"  failure reasons (top 5):")
    for reason, count in sorted(failed_reasons.items(), key=lambda x: -x[1])[:5]:
        print(f"    {count}x : {reason}")
    print(f"  total cherries used: {int(agg_n_cherries.sum())}")
    print(f"  total transitions: {int(agg_trans.sum())}")
    print(f"  gap totals per type: SM={int(agg_gap[:, GAP_SM].sum())}, "
          f"MM={int(agg_gap[:, GAP_MM].sum())}, "
          f"ME={int(agg_gap[:, GAP_ME].sum())}, "
          f"SE={int(agg_gap[:, GAP_SE].sum())}")

    edges, centers = geom_bin_edges(args.n_bins, args.tau_min, args.tau_max)
    np.savez(args.out,
             gap_counts=agg_gap,
             trans_counts=agg_trans,
             n_cherries_per_bin=agg_n_cherries,
             tau_edges=edges,
             tau_centers=centers,
             Lmax=args.Lmax,
             n_families_ok=n_ok,
             n_families_failed=n_failed,
             n_cherries_skipped=n_skipped)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    mp.set_start_method('fork', force=True)
    main()
