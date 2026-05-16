#!/usr/bin/env python3
"""Fels40 benchmark: 40-state model with hidden gapped states.

Evaluates the fels40 model on Pfam leaf prediction (same task as carabs and
felsenstein_c10_c20_benchmark.py): predict a withheld leaf given the rest
of the MSA + tree.

The fels40 model uses 40 hidden states (20 present + 20 gapped) with a
40x40 transition matrix. At leaves, amino acids select a single present
state while gaps select all 20 gapped states. The model naturally handles
gap prediction without needing separate Fitch parsimony.

Usage:
    cd python && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python experiments/fit_fels40.py [--n-samples 200]
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.protein_gap40 import (
    build_Q40, build_emission_matrix, felsenstein_40,
    reconstruct_ancestors_40, reconstruct_leaf_40,
)
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.tree.guide_tree import neighbor_joining
from tkfmixdom.jax.util.io import AA_TO_INT, AMINO_ACIDS

# Paths
PFAM_DIR = "/home/yam/bio-datasets/data/pfam-seed"
SPLITS_PATH = os.path.join(PFAM_DIR, "splits", "v1.json")

A = 20  # amino acid alphabet size
N_STATES = 40

# ── Stockholm parser ────────────────────────────────────────────────────

def parse_sto(path):
    """Parse Stockholm format MSA. Returns dict {name: aligned_seq_string}."""
    seqs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = line.split()
            if len(parts) == 2:
                name, seq = parts
                seqs.setdefault(name, '')
                seqs[name] += seq
    return seqs


# ── Tree utilities ──────────────────────────────────────────────────────

from tkfmixdom.jax.util.io import parse_newick, TreeNode


def msa_pairwise_distances(aligned_seqs, Q, pi):
    """Compute pairwise LG08 distance matrix from aligned sequences."""
    names = list(aligned_seqs.keys())
    n = len(names)
    D = np.zeros((n, n))

    t_values = np.concatenate([
        np.linspace(0.001, 0.1, 50),
        np.linspace(0.1, 1.0, 50),
        np.linspace(1.0, 5.0, 50),
        np.linspace(5.0, 20.0, 20),
    ])
    expected_ids = np.zeros(len(t_values))
    for ti, t in enumerate(t_values):
        P_t = np.array(transition_matrix(Q, t))
        expected_ids[ti] = np.sum(np.array(pi) * np.diag(P_t))

    for i in range(n):
        for j in range(i + 1, n):
            si = aligned_seqs[names[i]]
            sj = aligned_seqs[names[j]]
            matches = aligned = 0
            for ci, cj in zip(si, sj):
                if ci not in '.-' and cj not in '.-':
                    aligned += 1
                    if ci == cj:
                        matches += 1
            if aligned == 0:
                D[i, j] = D[j, i] = 5.0
                continue
            obs_id = matches / aligned
            idx = np.argmin(np.abs(expected_ids - obs_id))
            D[i, j] = D[j, i] = t_values[idx]
    return names, D


def remove_leaf(tree, leaf_name):
    """Remove a leaf from the tree, return (pruned_tree, branch_length)."""
    def _deep_copy(node, parent=None):
        new_node = TreeNode(node.name, node.branch_length)
        new_node.parent = parent
        for c in node.children:
            new_child = _deep_copy(c, new_node)
            new_node.children.append(new_child)
        return new_node

    new_tree = _deep_copy(tree)
    target = None
    for node in new_tree.preorder():
        if node.is_leaf and node.name == leaf_name:
            target = node
            break
    if target is None:
        return None, None

    parent = target.parent
    bl = target.branch_length
    if parent is None:
        return None, None

    parent.children = [c for c in parent.children if c.name != leaf_name]

    # If parent now has 1 child and is not root, merge with grandparent
    if len(parent.children) == 1 and parent.parent is not None:
        remaining_child = parent.children[0]
        grandparent = parent.parent
        remaining_child.branch_length += parent.branch_length
        remaining_child.parent = grandparent
        grandparent.children = [
            remaining_child if c is parent else c
            for c in grandparent.children
        ]

    # If root has 1 child, promote child to root
    if parent.parent is None and len(parent.children) == 1:
        new_root = parent.children[0]
        new_root.parent = None
        new_root.name = parent.name if parent.name else new_root.name
        return new_root, bl

    root = new_tree
    while root.parent is not None:
        root = root.parent
    return root, bl


def distance_to_root(node):
    """Sum branch lengths from node to root."""
    d = 0.0
    current = node
    while current is not None:
        d += current.branch_length
        current = current.parent
    return d


def find_leaf_in_tree(tree, name):
    """Find leaf node by name."""
    for node in tree.preorder():
        if node.is_leaf and node.name == name:
            return node
    return None


# ── Fitch parsimony for gaps (for LG08 baseline comparison) ────────────

def fitch_gap_predict(tree, target_name, column_gap_data):
    """Predict gap/no-gap at target leaf using Fitch parsimony."""
    target_leaf = find_leaf_in_tree(tree, target_name)
    if target_leaf is None:
        return False

    parent = target_leaf.parent
    if parent is None:
        return False

    parent.children = [c for c in parent.children if c is not target_leaf]

    def _fitch(node):
        if node.is_leaf:
            name = node.name
            if name in column_gap_data:
                return {column_gap_data[name]}
            return {True, False}
        child_sets = [_fitch(c) for c in node.children]
        intersection = child_sets[0]
        for s in child_sets[1:]:
            intersection = intersection & s
        if intersection:
            return intersection
        union = child_sets[0]
        for s in child_sets[1:]:
            union = union | s
        return union

    states = _fitch(tree)
    parent.children.append(target_leaf)

    if False in states:
        return False
    return True


# ── Fels40 leaf prediction ─────────────────────────────────────────────

def fels40_predict_leaf(pruned_tree, pruned_msa_int, target_bl,
                        Q40, pi40, emission):
    """Predict leaf sequence using fels40 model.

    For each column:
    1. Run Felsenstein pruning on pruned tree (40 states)
    2. Get root posterior over 40 hidden states
    3. Propagate to target via P(t_target)
    4. Map to observed space via emission matrix
    5. Predict MAP observed character (AA or gap)

    Args:
        pruned_tree: tree with target removed
        pruned_msa_int: dict {name: int_array}, -1 for gaps
        target_bl: branch length to target
        Q40: (40, 40) rate matrix
        pi40: (40,) equilibrium
        emission: (40, 21) emission matrix

    Returns:
        predictions: (L,) int array (0-19 for AA, 20 for gap)
        posteriors_obs: (L, 21) posterior over observed characters
    """
    emission_np = np.asarray(emission)
    leaf_names = list(pruned_msa_int.keys())
    L = len(next(iter(pruned_msa_int.values())))

    # Get root posterior from pruned tree
    _, root_posteriors = reconstruct_ancestors_40(
        pruned_tree, pruned_msa_int, Q40, pi40, emission)

    # Propagate root -> target
    t = max(target_bl, 1e-6)
    P_t = np.asarray(transition_matrix(Q40, t))

    predictions = []
    posteriors_obs = []

    for col in range(L):
        post_root = root_posteriors[col]
        # Propagate: P(target_state) = P(t).T @ post_root
        post_target = P_t.T @ post_root
        post_target = post_target / max(np.sum(post_target), 1e-300)

        # Map to observed: P(obs=o) = sum_s E[s,o] * post[s]
        post_obs = emission_np.T @ post_target  # (21,)
        post_obs = post_obs / max(np.sum(post_obs), 1e-300)

        map_obs = int(np.argmax(post_obs))
        predictions.append(map_obs)
        posteriors_obs.append(post_obs)

    return np.array(predictions), np.array(posteriors_obs)


# ── LG08 + Fitch baseline (for comparison) ─────────────────────────────

def lg08_predict_leaf(pruned_tree, pruned_msa_int, target_bl, Q_lg, pi_lg):
    """Predict leaf AA using LG08 Felsenstein (no gap handling)."""
    from tkfmixdom.jax.core.protein_gap import reconstruct_root_gap

    leaf_names = list(pruned_msa_int.keys())
    L = len(next(iter(pruned_msa_int.values())))

    Q21, pi21 = _build_lg21(Q_lg, pi_lg)

    root_seq, root_posts = reconstruct_root_gap(
        pruned_tree, pruned_msa_int, Q21, pi21)

    P_t = np.asarray(transition_matrix(Q21, max(target_bl, 1e-6)))

    predictions = []
    for col in range(L):
        post_root = root_posts[col]
        post_target = P_t.T @ post_root
        post_target = post_target / max(np.sum(post_target), 1e-300)
        predictions.append(int(np.argmax(post_target[:20])))

    return np.array(predictions)


def _build_lg21(Q_lg, pi_lg):
    """Build LG21 rate matrix for baseline."""
    from tkfmixdom.jax.core.protein import rate_matrix_lg21
    return rate_matrix_lg21(ins_rate=0.03, del_rate=0.03)


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate_family(fam_id, aligned_seqs, tree, target_name,
                    remaining_names, Q_lg, pi_lg,
                    Q40, pi40, emission, max_len=256, rng=None):
    """Evaluate fels40 vs LG08+Fitch on one family + target.

    Returns dict with per-model accuracy, or None if skipped.
    """
    L = len(next(iter(aligned_seqs.values())))
    L_crop = min(L, max_len)
    start = 0
    if L > max_len and rng is not None:
        start = rng.randint(0, L - max_len + 1)

    # Build pruned tree
    pruned_tree, target_bl = remove_leaf(tree, target_name)
    if pruned_tree is None:
        return None

    # Distance from target to root (for LG08 baseline)
    target_leaf = find_leaf_in_tree(tree, target_name)
    if target_leaf is None:
        return None
    total_dist = distance_to_root(target_leaf)

    # Build pruned MSA
    pruned_leaves = set(n.name for n in pruned_tree.leaves())
    remaining = [n for n in remaining_names if n in pruned_leaves]
    if len(remaining) < 2:
        return None

    pruned_msa = {}
    for n in remaining:
        seq = np.full(L_crop, -1, dtype=np.int32)
        for j in range(L_crop):
            col = start + j
            if col < len(aligned_seqs[n]):
                ch = aligned_seqs[n][col].upper()
                if ch in AA_TO_INT and AA_TO_INT[ch] < 20:
                    seq[j] = AA_TO_INT[ch]
        pruned_msa[n] = seq

    # Target sequence
    target_aligned = aligned_seqs[target_name]

    # ── Fels40 prediction ──
    preds_40, posts_40 = fels40_predict_leaf(
        pruned_tree, pruned_msa, target_bl, Q40, pi40, emission)

    # ── Evaluate ──
    total_count = 0
    correct_40 = 0
    correct_40_aa_only = 0
    correct_40_gap_only = 0
    total_aa = 0
    total_gap = 0

    for col_idx in range(L_crop):
        col = start + col_idx
        if col >= len(target_aligned):
            break

        target_char = target_aligned[col].upper()

        if target_char in '.-':
            # Target is gap
            total_count += 1
            total_gap += 1
            if preds_40[col_idx] == 20:  # fels40 predicted gap
                correct_40 += 1
                correct_40_gap_only += 1
        elif target_char in AA_TO_INT and AA_TO_INT[target_char] < 20:
            # Target is amino acid
            total_count += 1
            total_aa += 1
            true_aa = AA_TO_INT[target_char]

            pred = preds_40[col_idx]
            if pred == true_aa:
                correct_40 += 1
                correct_40_aa_only += 1
            elif pred == 20:
                pass  # fels40 predicted gap, wrong (AA was true)

    if total_count == 0:
        return None

    result = {
        'family': fam_id,
        'target': target_name,
        'total_dist': float(total_dist),
        'branch_length': float(target_bl) if target_bl else 0.0,
        'n_columns': total_count,
        'n_aa_columns': total_aa,
        'n_gap_columns': total_gap,
        'acc_fels40': correct_40 / total_count,
        'acc_fels40_aa': correct_40_aa_only / total_aa if total_aa > 0 else None,
        'acc_fels40_gap': correct_40_gap_only / total_gap if total_gap > 0 else None,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fels40 (40-state hidden gap) benchmark")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of families to evaluate")
    parser.add_argument("--max-rows", type=int, default=32,
                        help="Max sequences per family")
    parser.add_argument("--max-len", type=int, default=256,
                        help="Max alignment length (crop if longer)")
    parser.add_argument("--r-del", type=float, default=0.03,
                        help="Deletion rate (present -> gapped)")
    parser.add_argument("--r-ins", type=float, default=0.03,
                        help="Insertion rate (gapped -> present)")
    parser.add_argument("--gap-subst-scale", type=float, default=0.0,
                        help="Scaling of hidden substitution in gapped states")
    parser.add_argument("--seed", type=int, default=99999,
                        help="Random seed")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    # Build fels40 model
    Q40, pi40 = build_Q40(r_del=args.r_del, r_ins=args.r_ins,
                           gap_subst_scale=args.gap_subst_scale)
    emission = build_emission_matrix()
    print(f"Built fels40 model: r_del={args.r_del}, r_ins={args.r_ins}, "
          f"gap_subst_scale={args.gap_subst_scale}")
    print(f"Q40 shape: {Q40.shape}, pi40 shape: {pi40.shape}")
    print(f"Row sum max deviation: {float(jnp.abs(Q40.sum(axis=1)).max()):.2e}")
    print(f"pi40 sum: {float(pi40.sum()):.6f}")
    print(f"Present states equilibrium: {float(pi40[:20].sum()):.4f}")
    print(f"Gapped states equilibrium: {float(pi40[20:].sum()):.4f}")

    # Verify detailed balance
    pi_np = np.asarray(pi40)
    Q_np = np.asarray(Q40)
    DB = pi_np[:, None] * Q_np - pi_np[None, :] * Q_np.T
    print(f"Detailed balance max violation: {np.abs(DB).max():.2e}")

    # Load split
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    families = splits[args.split]

    # Filter families
    valid_families = []
    for fam in families:
        sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")
        if os.path.exists(sto_path):
            valid_families.append(fam)
    print(f"\nFound {len(valid_families)}/{len(families)} {args.split} families")

    # LG08 for pairwise distances
    Q_lg, pi_lg = rate_matrix_lg()

    rng = np.random.RandomState(args.seed)
    all_results = []

    # Accumulators
    acc = {'correct': 0, 'count': 0,
           'correct_aa': 0, 'count_aa': 0,
           'correct_gap': 0, 'count_gap': 0}

    t0 = time.time()
    n_processed = 0
    n_skipped = 0

    max_attempts = args.n_samples * 5
    for i in range(max_attempts):
        if n_processed >= args.n_samples:
            break

        fam = valid_families[rng.randint(len(valid_families))]
        sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")

        try:
            aligned_seqs = parse_sto(sto_path)
        except Exception:
            n_skipped += 1
            continue

        names = list(aligned_seqs.keys())
        if len(names) < 4:
            n_skipped += 1
            continue

        # Build NJ tree
        try:
            nj_names, D = msa_pairwise_distances(aligned_seqs, Q_lg, pi_lg)
            tree = neighbor_joining(D, nj_names)
        except Exception:
            n_skipped += 1
            continue

        # Pick target
        leaves = [n.name for n in tree.leaves() if n.name in set(names)]
        if len(leaves) < 4:
            n_skipped += 1
            continue

        target_name = leaves[rng.randint(len(leaves))]
        remaining_names = [n for n in names if n != target_name]

        if len(remaining_names) > args.max_rows:
            tree_leaves = set(n.name for n in tree.leaves())
            in_tree = [n for n in remaining_names if n in tree_leaves]
            if len(in_tree) > args.max_rows:
                chosen = list(rng.choice(in_tree, args.max_rows, replace=False))
                remaining_names = chosen
            else:
                remaining_names = in_tree

        try:
            result = evaluate_family(
                fam, aligned_seqs, tree, target_name,
                remaining_names, Q_lg, pi_lg,
                Q40, pi40, emission,
                max_len=args.max_len, rng=rng)
        except Exception as e:
            traceback.print_exc()
            n_skipped += 1
            continue

        if result is None:
            n_skipped += 1
            continue

        all_results.append(result)
        n_processed += 1

        # Accumulate
        acc['correct'] += int(result['acc_fels40'] * result['n_columns'])
        acc['count'] += result['n_columns']
        if result['n_aa_columns'] > 0 and result['acc_fels40_aa'] is not None:
            acc['correct_aa'] += int(result['acc_fels40_aa'] * result['n_aa_columns'])
            acc['count_aa'] += result['n_aa_columns']
        if result['n_gap_columns'] > 0 and result['acc_fels40_gap'] is not None:
            acc['correct_gap'] += int(result['acc_fels40_gap'] * result['n_gap_columns'])
            acc['count_gap'] += result['n_gap_columns']

        if n_processed % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / n_processed * (args.n_samples - n_processed)
            overall = acc['correct'] / acc['count'] if acc['count'] > 0 else 0
            aa_acc = acc['correct_aa'] / acc['count_aa'] if acc['count_aa'] > 0 else 0
            gap_acc = acc['correct_gap'] / acc['count_gap'] if acc['count_gap'] > 0 else 0
            print(f"\n--- Sample {n_processed}/{args.n_samples} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s ETA, {n_skipped} skipped) ---")
            print(f"  fels40 overall: {overall:.4f} ({acc['correct']}/{acc['count']})")
            print(f"  fels40 AA:      {aa_acc:.4f} ({acc['correct_aa']}/{acc['count_aa']})")
            print(f"  fels40 gap:     {gap_acc:.4f} ({acc['correct_gap']}/{acc['count_gap']})")

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"Fels40 benchmark ({args.split})")
    print(f"  r_del={args.r_del}, r_ins={args.r_ins}, "
          f"gap_subst_scale={args.gap_subst_scale}")
    print(f"  Samples: {n_processed}, Skipped: {n_skipped}, Time: {elapsed:.0f}s")
    if acc['count'] > 0:
        print(f"  Overall accuracy: {acc['correct']/acc['count']:.4f} "
              f"({acc['correct']}/{acc['count']})")
    if acc['count_aa'] > 0:
        print(f"  AA accuracy:      {acc['correct_aa']/acc['count_aa']:.4f} "
              f"({acc['correct_aa']}/{acc['count_aa']})")
    if acc['count_gap'] > 0:
        print(f"  Gap accuracy:     {acc['correct_gap']/acc['count_gap']:.4f} "
              f"({acc['correct_gap']}/{acc['count_gap']})")
    print(f"{'='*70}")

    # Save
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"fels40_results_{args.split}.json")

    summary = {
        'args': vars(args),
        'n_processed': n_processed,
        'n_skipped': n_skipped,
        'elapsed_seconds': elapsed,
        'accuracies': {
            'overall': {
                'accuracy': acc['correct'] / acc['count'] if acc['count'] > 0 else None,
                'correct': acc['correct'],
                'count': acc['count'],
            },
            'aa_only': {
                'accuracy': acc['correct_aa'] / acc['count_aa'] if acc['count_aa'] > 0 else None,
                'correct': acc['correct_aa'],
                'count': acc['count_aa'],
            },
            'gap_only': {
                'accuracy': acc['correct_gap'] / acc['count_gap'] if acc['count_gap'] > 0 else None,
                'correct': acc['correct_gap'],
                'count': acc['count_gap'],
            },
        },
        'per_family': all_results,
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
