#!/usr/bin/env python3
"""Benchmark 3 gap patterns for ancestral reconstruction with Felsenstein.

Compares three gap patterns, all using column-by-column Felsenstein:
  1. Union-of-leaves: ancestor present if ANY descendant has char (most generous)
  2. Fitch parsimony: ancestor present if BOTH child subtrees have char,
     plus preorder propagation (parsimonious)
  3. Triad HMM: 1D Viterbi after pairwise alignment — ML parent presence
     from birth-death model (most principled)

For each holdout, reports: ancestor length, identity, tau_leaf, and
logs aligned true/reconstructed sequences with match/mismatch annotation.

Usage:
  cd python && JAX_PLATFORMS=cpu uv run python experiments/ancrec_gap_patterns.py
"""

import os
import sys
import time
import json
import traceback
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp

from tkfmixdom.jax.util.io import (
    parse_newick, TreeNode, AA_TO_INT, INT_TO_AA, seq_to_int, int_to_seq,
)
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.core.params import S, M, I, D, E
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_column
from tkfmixdom.jax.tree.triad_gap_inference import reconstruct_with_triad

# Paths
TREEFAM_DIR = "/home/yam/bio-datasets/data/treefam/treefam_family_data"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "ancrec_gap_patterns.json")

TREEFAM_FAMILIES = [
    "TF312992", "TF314771", "TF315300", "TF315335", "TF315406",
    "TF315675", "TF315741", "TF315828", "TF315840", "TF316543",
]

# TKF92 parameters
INS_RATE = 0.02
DEL_RATE = 0.05
EXT = 0.5

TAU_BINS = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, float('inf'))]
TAU_BIN_LABELS = ["[0,0.05)", "[0.05,0.1)", "[0.1,0.2)", "[0.2,0.5)", "[0.5+)"]


# ============================================================
# Parsing helpers
# ============================================================

def parse_emf_alignment(path):
    """Parse TreeFam EMF alignment file."""
    names = []
    columns = []
    in_data = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SEQ"):
                parts = line.split()
                names.append(parts[2])
            elif line == "DATA":
                in_data = True
            elif line == "//" or line.startswith("#"):
                continue
            elif in_data and line:
                columns.append(line)
    alignment = {}
    for i, name in enumerate(names):
        alignment[name] = ''.join(col[i] if i < len(col) else '-'
                                   for col in columns)
    return names, alignment


def parse_emf_tree(path):
    """Parse TreeFam EMF tree file."""
    tree_str = None
    seq_names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SEQ"):
                parts = line.split()
                seq_names.append(parts[2])
            elif line and not line.startswith("#") and line != "//":
                if '(' in line:
                    tree_str = line
    if tree_str is None:
        return None, seq_names
    tree = parse_newick(tree_str)
    return tree, seq_names


def parse_fasta(path):
    """Parse FASTA, return {name: seq_str}."""
    seqs = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                name = line[1:].split()[0]
                seqs[name] = ''
            elif name:
                seqs[name] += line
    return seqs


# ============================================================
# Tree manipulation
# ============================================================

def _deep_copy_tree(node, parent=None):
    new_node = TreeNode(node.name, node.branch_length)
    new_node.parent = parent
    for c in node.children:
        new_child = _deep_copy_tree(c, new_node)
        new_node.children.append(new_child)
    return new_node


def remove_leaf(tree, leaf_name):
    """Remove a leaf from the tree, return (pruned_tree, parent_name, branch_length)."""
    new_tree = _deep_copy_tree(tree)
    target = None
    for node in new_tree.preorder():
        if node.is_leaf and node.name == leaf_name:
            target = node
            break
    if target is None:
        return None, None, None

    parent = target.parent
    bl = target.branch_length
    if parent is None:
        return None, None, None

    parent.children = [c for c in parent.children if c.name != leaf_name]

    if len(parent.children) == 1 and parent.parent is not None:
        remaining_child = parent.children[0]
        grandparent = parent.parent
        remaining_child.branch_length += parent.branch_length
        remaining_child.parent = grandparent
        grandparent.children = [
            remaining_child if c is parent else c
            for c in grandparent.children
        ]

    if parent.parent is None and len(parent.children) == 1:
        new_root = parent.children[0]
        new_root.parent = None
        new_root.name = parent.name if parent.name else new_root.name
        return new_root, parent.name, bl

    root = new_tree
    while root.parent is not None:
        root = root.parent
    return root, parent.name if parent else None, bl


def find_parent_of_leaf(tree, leaf_name):
    """Find parent and branch length for a leaf."""
    for node in tree.preorder():
        if node.is_leaf and node.name == leaf_name:
            return node.parent.name if node.parent else None, node.branch_length
    return None, None


def name_internal_nodes(tree):
    """Assign names to internal nodes that don't have them."""
    counter = [0]
    for node in tree.preorder():
        if not node.is_leaf and (node.name is None or node.name == ""):
            node.name = f"_internal_{counter[0]}"
            counter[0] += 1


# ============================================================
# Gap inference methods
# ============================================================

def infer_union_presence(tree, leaf_presence):
    """Union-of-leaves: ancestor present if ANY descendant has char."""
    L = len(next(iter(leaf_presence.values())))
    presence = {}
    for node in tree.postorder():
        if node.is_leaf:
            if node.name in leaf_presence:
                presence[node.name] = np.array(leaf_presence[node.name], dtype=bool)
            else:
                presence[node.name] = np.zeros(L, dtype=bool)
        else:
            p = np.zeros(L, dtype=bool)
            for child in node.children:
                if child.name in presence:
                    p |= presence[child.name]
            presence[node.name] = p
    return presence


def infer_fitch_presence(tree, leaf_presence):
    """Fitch parsimony: postorder intersection, preorder propagation.

    Postorder: internal node present if BOTH children present (intersection).
    Preorder: if parent present, child present (propagate down).
    """
    L = len(next(iter(leaf_presence.values())))
    presence = {}

    # Postorder pass: intersection (present if BOTH children present)
    for node in tree.postorder():
        if node.is_leaf:
            if node.name in leaf_presence:
                presence[node.name] = np.array(leaf_presence[node.name], dtype=bool)
            else:
                presence[node.name] = np.zeros(L, dtype=bool)
        else:
            children_with_pres = [c for c in node.children if c.name in presence]
            if len(children_with_pres) == 0:
                presence[node.name] = np.zeros(L, dtype=bool)
            elif len(children_with_pres) == 1:
                presence[node.name] = presence[children_with_pres[0].name].copy()
            else:
                # Intersection: present only if ALL children are present
                p = np.ones(L, dtype=bool)
                for child in children_with_pres:
                    p &= presence[child.name]
                presence[node.name] = p

    # Preorder pass: if parent present, child present
    for node in tree.preorder():
        if node.parent is not None and node.parent.name in presence:
            parent_pres = presence[node.parent.name]
            # If parent says present, child must be present too
            presence[node.name] = presence[node.name] | parent_pres

    return presence


# ============================================================
# NW alignment with traceback for display
# ============================================================

def needleman_wunsch_with_alignment(seq1, seq2, match_score=1, mismatch=-1,
                                     gap_penalty=-2):
    """NW alignment returning identity and aligned sequence strings.

    Args:
        seq1, seq2: integer arrays (amino acid indices)

    Returns:
        identity: fraction of aligned positions that match
        n_aligned: number of aligned (non-gap) positions
        n_matches: number of matching positions
        aln_str1: aligned sequence 1 as string (with gaps '-')
        aln_str2: aligned sequence 2 as string (with gaps '-')
        match_str: match annotation ('|' for match, ' ' for mismatch, '-' for gap)
    """
    n, m = len(seq1), len(seq2)
    if n == 0 and m == 0:
        return 0.0, 0, 0, "", "", ""
    if n == 0:
        s2 = "".join(INT_TO_AA.get(int(c), "X") for c in seq2)
        return 0.0, 0, 0, "-" * m, s2, " " * m
    if m == 0:
        s1 = "".join(INT_TO_AA.get(int(c), "X") for c in seq1)
        return 0.0, 0, 0, s1, "-" * n, " " * n

    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_penalty
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap_penalty
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match_score if seq1[i-1] == seq2[j-1] else mismatch
            dp[i, j] = max(dp[i-1, j-1] + s,
                          dp[i-1, j] + gap_penalty,
                          dp[i, j-1] + gap_penalty)

    # Traceback
    aln1, aln2, match = [], [], []
    i, j = n, m
    matches = aligned = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = match_score if seq1[i-1] == seq2[j-1] else mismatch
            if dp[i, j] == dp[i-1, j-1] + s:
                c1 = INT_TO_AA.get(int(seq1[i-1]), "X")
                c2 = INT_TO_AA.get(int(seq2[j-1]), "X")
                aln1.append(c1)
                aln2.append(c2)
                aligned += 1
                if seq1[i-1] == seq2[j-1]:
                    match.append("|")
                    matches += 1
                else:
                    match.append(" ")
                i -= 1; j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i-1, j] + gap_penalty:
            aln1.append(INT_TO_AA.get(int(seq1[i-1]), "X"))
            aln2.append("-")
            match.append(" ")
            i -= 1
        elif j > 0:
            aln1.append("-")
            aln2.append(INT_TO_AA.get(int(seq2[j-1]), "X"))
            match.append(" ")
            j -= 1
        else:
            break

    aln1.reverse(); aln2.reverse(); match.reverse()
    identity = matches / max(aligned, 1)
    return identity, aligned, matches, "".join(aln1), "".join(aln2), "".join(match)


# ============================================================
# Felsenstein reconstruction with a given gap pattern
# ============================================================

def felsenstein_with_presence(pruned_tree, pruned_msa_int, presence, Q, pi):
    """Column-by-column Felsenstein using a given presence pattern.

    Only reconstructs at columns where root presence is True.

    Args:
        pruned_tree: TreeNode (pruned, with internal nodes named)
        pruned_msa_int: dict {name: (L_msa,) int array, -1=gap}
        presence: dict {name: (L_msa,) bool array}
        Q: rate matrix
        pi: equilibrium frequencies

    Returns:
        ancestor_seq: int array of MAP characters at present positions
    """
    leaf_names = list(pruned_msa_int.keys())
    msa_len = len(next(iter(pruned_msa_int.values())))
    root_name = pruned_tree.name
    root_pres = presence.get(root_name, np.zeros(msa_len, dtype=bool))

    ancestor_chars = []
    for col in range(msa_len):
        if not root_pres[col]:
            continue
        col_chars = {}
        for name in leaf_names:
            c = int(pruned_msa_int[name][col])
            col_chars[name] = c
        # Even if no leaf is present (c<0), marginal_ancestor_column handles it
        post = marginal_ancestor_column(pruned_tree, col_chars, Q, pi)
        ancestor_chars.append(int(np.argmax(np.asarray(post))))

    return np.array(ancestor_chars, dtype=np.int32)


# ============================================================
# Triad Felsenstein
# ============================================================

def method_triad_felsenstein(pruned_tree, leaf_seqs_ungapped, Q, pi):
    """Progressive Felsenstein with triad gap inference.

    Returns the triad's own MAP root sequence directly.
    The triad MSA's gap pattern gives ancestor presence — do NOT re-infer.
    """
    _, _, root_seq = reconstruct_with_triad(
        pruned_tree, leaf_seqs_ungapped,
        ins_rate=INS_RATE, del_rate=DEL_RATE, t_scale=1.0,
        Q=Q, pi=pi, use_tkf92=True, ext=EXT,
        triad_method='viterbi')
    return root_seq


# ============================================================
# Hold-out experiment
# ============================================================

def run_holdout(family_id, held_out, seq_names, aln_int, ungapped, tree,
                Q, pi, verbose):
    """Run all 3 gap-pattern methods on one held-out leaf."""
    _, branch_len = find_parent_of_leaf(tree, held_out)
    if branch_len is None:
        return None

    pruned_tree, parent, bl = remove_leaf(tree, held_out)
    if pruned_tree is None:
        return None

    pruned_leaves = set(n.name for n in pruned_tree.leaves())
    remaining = [n for n in seq_names if n != held_out and n in pruned_leaves]
    if len(remaining) < 2:
        return None

    name_internal_nodes(pruned_tree)

    # MSA for union/fitch methods
    pruned_msa = {n: aln_int[n] for n in remaining}
    # Ungapped seqs for triad method
    pruned_ungapped = {n: ungapped[n] for n in remaining}
    true_seq = ungapped[held_out]

    result = {
        "family": family_id,
        "held_out": held_out,
        "tau_leaf": float(branch_len),
        "true_seq_len": len(true_seq),
        "n_seqs": len(seq_names),
    }

    # Leaf presence from MSA
    leaf_pres = {n: np.array(pruned_msa[n] >= 0, dtype=bool) for n in remaining}

    # ---- Method 1: Union-of-leaves + Felsenstein ----
    t0 = time.time()
    try:
        union_pres = infer_union_presence(pruned_tree, leaf_pres)
        anc_union = felsenstein_with_presence(
            pruned_tree, pruned_msa, union_pres, Q, pi)
        t_union = time.time() - t0
        result["anc_len_union"] = len(anc_union)
        id_u, al_u, ma_u, aln1_u, aln2_u, match_u = needleman_wunsch_with_alignment(
            anc_union, true_seq)
        result["identity_union"] = float(id_u)
        result["matches_union"] = int(ma_u)
        result["aligned_union"] = int(al_u)
        result["time_union"] = float(t_union)
        result["status_union"] = "ok"
        result["aln_recon_union"] = aln1_u
        result["aln_true_union"] = aln2_u
        result["aln_match_union"] = match_u
    except Exception as e:
        result["status_union"] = f"error: {str(e)[:200]}"
        result["identity_union"] = None
        result["time_union"] = time.time() - t0
        if verbose:
            print(f"    {held_out}/union: ERROR {str(e)[:120]}")
            traceback.print_exc()

    # ---- Method 2: Fitch parsimony + Felsenstein ----
    t0 = time.time()
    try:
        fitch_pres = infer_fitch_presence(pruned_tree, leaf_pres)
        anc_fitch = felsenstein_with_presence(
            pruned_tree, pruned_msa, fitch_pres, Q, pi)
        t_fitch = time.time() - t0
        result["anc_len_fitch"] = len(anc_fitch)
        id_f, al_f, ma_f, aln1_f, aln2_f, match_f = needleman_wunsch_with_alignment(
            anc_fitch, true_seq)
        result["identity_fitch"] = float(id_f)
        result["matches_fitch"] = int(ma_f)
        result["aligned_fitch"] = int(al_f)
        result["time_fitch"] = float(t_fitch)
        result["status_fitch"] = "ok"
        result["aln_recon_fitch"] = aln1_f
        result["aln_true_fitch"] = aln2_f
        result["aln_match_fitch"] = match_f
    except Exception as e:
        result["status_fitch"] = f"error: {str(e)[:200]}"
        result["identity_fitch"] = None
        result["time_fitch"] = time.time() - t0
        if verbose:
            print(f"    {held_out}/fitch: ERROR {str(e)[:120]}")
            traceback.print_exc()

    # ---- Method 3: Triad HMM + Felsenstein ----
    t0 = time.time()
    try:
        pt_triad = _deep_copy_tree(pruned_tree)
        # remove_leaf already pruned; need a fresh copy for triad
        pt_triad2, _, _ = remove_leaf(tree, held_out)
        name_internal_nodes(pt_triad2)
        anc_triad = method_triad_felsenstein(pt_triad2, pruned_ungapped, Q, pi)
        t_triad = time.time() - t0
        result["anc_len_triad"] = len(anc_triad)
        id_t, al_t, ma_t, aln1_t, aln2_t, match_t = needleman_wunsch_with_alignment(
            anc_triad, true_seq)
        result["identity_triad"] = float(id_t)
        result["matches_triad"] = int(ma_t)
        result["aligned_triad"] = int(al_t)
        result["time_triad"] = float(t_triad)
        result["status_triad"] = "ok"
        result["aln_recon_triad"] = aln1_t
        result["aln_true_triad"] = aln2_t
        result["aln_match_triad"] = match_t
    except Exception as e:
        result["status_triad"] = f"error: {str(e)[:200]}"
        result["identity_triad"] = None
        result["time_triad"] = time.time() - t0
        if verbose:
            print(f"    {held_out}/triad: ERROR {str(e)[:120]}")
            traceback.print_exc()

    # Print summary line
    if verbose:
        parts = []
        for m in ["union", "fitch", "triad"]:
            v = result.get(f"identity_{m}")
            al = result.get(f"anc_len_{m}", "?")
            if v is not None:
                parts.append(f"{m}={v:.3f}(L={al})")
            else:
                parts.append(f"{m}=ERR")
        print(f"    {held_out} (tau={branch_len:.3f}, true_L={len(true_seq)}): "
              f"{', '.join(parts)}")

        # Print aligned sequences for each method
        for m in ["union", "fitch", "triad"]:
            if result.get(f"status_{m}") != "ok":
                continue
            al = result.get(f"anc_len_{m}", "?")
            ident = result.get(f"identity_{m}", 0)
            recon_str = result.get(f"aln_recon_{m}", "")
            true_str = result.get(f"aln_true_{m}", "")
            match_str = result.get(f"aln_match_{m}", "")
            # Truncate for display
            max_show = 70
            if len(recon_str) > max_show:
                recon_show = recon_str[:max_show] + "..."
                true_show = true_str[:max_show] + "..."
                match_show = match_str[:max_show] + "..."
            else:
                recon_show = recon_str
                true_show = true_str
                match_show = match_str
            print(f"      {m.capitalize()}-Fels (id={ident:.3f}, L={al}):")
            print(f"        recon: {recon_show}")
            print(f"        true:  {true_show}")
            print(f"        match: {match_show}")

    return result


# ============================================================
# TreeFam family processing
# ============================================================

def process_treefam_family(family_id, Q, pi, verbose=True):
    """Process one TreeFam family."""
    aln_path = os.path.join(TREEFAM_DIR, f"{family_id}.aln.emf")
    tree_path = os.path.join(TREEFAM_DIR, f"{family_id}.nh.emf")
    fasta_path = os.path.join(TREEFAM_DIR, f"{family_id}.aa.fasta")

    if not os.path.exists(aln_path):
        print(f"  SKIP {family_id}: alignment not found")
        return []

    seq_names, alignment = parse_emf_alignment(aln_path)
    tree, tree_seq_names = parse_emf_tree(tree_path)
    raw_seqs = parse_fasta(fasta_path)

    tree_leaves = [n.name for n in tree.leaves()]
    common_names = [n for n in seq_names if n in tree_leaves]
    if len(common_names) < 3:
        print(f"  SKIP {family_id}: too few sequences ({len(common_names)})")
        return []
    seq_names = common_names

    name_internal_nodes(tree)

    aln_int = {}
    ungapped = {}
    for name in seq_names:
        aln_str = alignment[name]
        arr = np.array([AA_TO_INT.get(c, -1) for c in aln_str], dtype=np.int32)
        aln_int[name] = arr
        if name in raw_seqs:
            ungapped[name] = seq_to_int(raw_seqs[name], "protein")
        else:
            ungapped[name] = np.array([c for c in arr if c >= 0], dtype=np.int32)

    if verbose:
        print(f"\n  Family: {family_id}  ({len(seq_names)} seqs, "
              f"aln_len={len(next(iter(alignment.values())))})")

    results = []
    for held_out in seq_names:
        if len(ungapped[held_out]) == 0:
            continue
        r = run_holdout(family_id, held_out, seq_names, aln_int, ungapped,
                        tree, Q, pi, verbose)
        if r is not None:
            results.append(r)
    return results


# ============================================================
# Summary and reporting
# ============================================================

def print_summary(holdouts):
    """Print summary statistics."""
    method_names = ["union", "fitch", "triad"]
    print(f"\n{'='*80}")
    print(f"Summary ({len(holdouts)} hold-outs)")
    print(f"{'='*80}")

    # Overall averages
    print(f"\nOverall averages:")
    for m in method_names:
        ids = [r[f"identity_{m}"] for r in holdouts
               if r.get(f"status_{m}") == "ok" and r.get(f"identity_{m}") is not None]
        lens = [r[f"anc_len_{m}"] for r in holdouts
                if r.get(f"status_{m}") == "ok" and r.get(f"anc_len_{m}") is not None]
        true_lens = [r["true_seq_len"] for r in holdouts
                     if r.get(f"status_{m}") == "ok" and r.get(f"anc_len_{m}") is not None]
        n_ok = len(ids)
        if ids:
            # Length ratio: anc_len / true_len
            ratios = [al / tl for al, tl in zip(lens, true_lens) if tl > 0]
            print(f"  {m:<8}: mean_id={np.mean(ids):.4f} "
                  f"(std={np.std(ids):.4f}), "
                  f"mean_anc_len={np.mean(lens):.1f}, "
                  f"mean_len_ratio={np.mean(ratios):.3f}, "
                  f"n={n_ok}")
        else:
            print(f"  {m:<8}: no successful runs")

    # Pairwise length comparisons
    print(f"\n  Ancestor length comparisons:")
    for m1, m2 in [("union", "fitch"), ("union", "triad"), ("fitch", "triad")]:
        pairs = []
        for r in holdouts:
            l1 = r.get(f"anc_len_{m1}")
            l2 = r.get(f"anc_len_{m2}")
            if l1 is not None and l2 is not None and l1 > 0:
                pairs.append((l1, l2))
        if pairs:
            ratios = [l2/l1 for l1, l2 in pairs]
            print(f"    {m2}/{m1}: mean ratio={np.mean(ratios):.3f} "
                  f"(std={np.std(ratios):.3f}), n={len(pairs)}")

    # By tau bins
    print(f"\nBreakdown by tau_leaf:")
    header = f"  {'tau_bin':<12} {'n':>4}"
    for m in method_names:
        header += f"  {'id_'+m:>10} {'L_'+m:>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for (lo, hi), label in zip(TAU_BINS, TAU_BIN_LABELS):
        bin_holdouts = [r for r in holdouts
                        if r.get("tau_leaf") is not None
                        and lo <= r["tau_leaf"] < hi]
        if not bin_holdouts:
            continue
        row = f"  {label:<12} {len(bin_holdouts):>4}"
        for m in method_names:
            ids = [r[f"identity_{m}"] for r in bin_holdouts
                   if r.get(f"status_{m}") == "ok"
                   and r.get(f"identity_{m}") is not None]
            lens = [r[f"anc_len_{m}"] for r in bin_holdouts
                    if r.get(f"status_{m}") == "ok"
                    and r.get(f"anc_len_{m}") is not None]
            if ids:
                row += f"  {np.mean(ids):>10.4f} {np.mean(lens):>8.1f}"
            else:
                row += f"  {'N/A':>10} {'N/A':>8}"
        print(row)


# ============================================================
# Main
# ============================================================

def main():
    print("Loading LG08 rate matrix...")
    Q, pi = rate_matrix_lg()
    Q_np = np.array(Q)
    pi_np = np.array(pi)

    output = {
        "description": "3 gap patterns x Felsenstein ancestral reconstruction",
        "methods": ["union", "fitch", "triad"],
        "method_descriptions": {
            "union": "Union-of-leaves: ancestor present if ANY descendant has char",
            "fitch": "Fitch parsimony: intersection postorder + propagation preorder",
            "triad": "Triad HMM: 1D Viterbi parent presence from birth-death model",
        },
        "params": {
            "ins_rate": INS_RATE,
            "del_rate": DEL_RATE,
            "ext": EXT,
            "substitution": "LG08",
        },
        "families": TREEFAM_FAMILIES,
        "holdouts": [],
    }

    # Start with first family to validate
    print(f"\n{'='*80}")
    print(f"Running on first family to validate...")
    print(f"{'='*80}")

    fam = TREEFAM_FAMILIES[0]
    try:
        results = process_treefam_family(fam, Q_np, pi_np, verbose=True)
        output["holdouts"].extend(results)

        # Quick validation: check lengths
        for r in results[:3]:
            ul = r.get("anc_len_union", "?")
            fl = r.get("anc_len_fitch", "?")
            tl = r.get("anc_len_triad", "?")
            tru = r.get("true_seq_len", "?")
            print(f"    Validation: true={tru}, union={ul}, fitch={fl}, triad={tl}")
    except Exception as e:
        print(f"  ERROR on {fam}: {e}")
        traceback.print_exc()

    # Save intermediate
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nIntermediate saved to {OUTPUT_PATH}")

    # Remaining families
    print(f"\n{'='*80}")
    print(f"Running remaining {len(TREEFAM_FAMILIES)-1} families...")
    print(f"{'='*80}")

    for fam in TREEFAM_FAMILIES[1:]:
        try:
            results = process_treefam_family(fam, Q_np, pi_np, verbose=True)
            output["holdouts"].extend(results)
        except Exception as e:
            print(f"  ERROR on {fam}: {e}")
            traceback.print_exc()

        # Save after each family
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(output, f, indent=2)

    print_summary(output["holdouts"])

    # Final save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nFinal results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
