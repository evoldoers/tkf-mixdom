#!/usr/bin/env python3
"""Felsenstein ancestral reconstruction with Le-Gascuel C10/C20 mixture + gamma.

This is Felsenstein's best effort: the richest point substitution model available,
using Le-Gascuel C10 or C20 site class profiles (each with its own equilibrium
distribution, shared LG08 exchangeability) combined with gamma rate heterogeneity.

Task (matching carabs): predict a withheld leaf given the rest of the MSA + tree.
Metric: per-column accuracy including gap positions (Fitch parsimony for gaps).

For each MSA column where Fitch predicts "not gap":
  P(target=a | column, tree) = sum_c w_c * sum_g (1/G) * [P_g,c(t_total).T @ post_root_gc]_a
where:
  post_root_gc = pi_c * CL_root_gc / Z_gc   (Felsenstein posterior at root for class c, rate g)
  Q_c = S_LG * diag(pi_c), normalized
  P_g,c(t) = exp(Q_c * rho_g * t)

Usage:
    cd python && uv run python experiments/felsenstein_c10_c20_benchmark.py [--n-samples 200]
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

from tkfmixdom.jax.core.protein import rate_matrix_lg, lg_exchangeability
from tkfmixdom.jax.core.site_class_profiles import get_profiles
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.tree.ancestor import (
    _flatten_tree_for_felsenstein,
    _build_leaf_obs_array,
)
from tkfmixdom.jax.tree.guide_tree import neighbor_joining
from tkfmixdom.jax.util.io import AA_TO_INT, AMINO_ACIDS

# Paths
PFAM_DIR = "/home/yam/bio-datasets/data/pfam-seed"
SPLITS_PATH = os.path.join(PFAM_DIR, "splits", "v1.json")

# AA alphabet: alphabetical order (same as io.AMINO_ACIDS)
AA_ALPHA = AMINO_ACIDS  # "ACDEFGHIKLMNPQRSTVWY"
A = 20

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


# ── Fitch parsimony for gaps ────────────────────────────────────────────

def fitch_gap_predict(tree, target_name, column_gap_data):
    """Predict gap/no-gap at target leaf using Fitch parsimony on pruned tree.

    Args:
        tree: original tree (will be modified temporarily)
        target_name: leaf to predict
        column_gap_data: dict {leaf_name: True/False} for gap/not-gap

    Returns:
        True if gap predicted, False if not-gap
    """
    target_leaf = find_leaf_in_tree(tree, target_name)
    if target_leaf is None:
        return False

    parent = target_leaf.parent
    if parent is None:
        return False

    # Temporarily remove target
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

    # Restore
    parent.children.append(target_leaf)

    # If ambiguous, predict no-gap
    if False in states:
        return False
    return True


# ── Build per-class rate matrices ───────────────────────────────────────

def build_class_rate_matrices(profile_name):
    """Build rate matrices for each class in a Le-Gascuel mixture.

    Each class c uses: Q_c = S_LG * diag(pi_c), normalized to mean rate 1.

    Returns:
        Q_classes: (n_classes, 20, 20) rate matrices
        pi_classes: (n_classes, 20) equilibrium distributions
        weights: (n_classes,) class weights
    """
    profiles, weights, names = get_profiles(profile_name)
    S_lg, _ = lg_exchangeability()

    n_classes = profiles.shape[0]
    Q_classes = np.zeros((n_classes, A, A))
    pi_classes = np.array(profiles)

    for c in range(n_classes):
        pi_c = pi_classes[c]
        Q_c = S_lg * pi_c[None, :]  # Q_ij = S_ij * pi_j for j != i
        np.fill_diagonal(Q_c, 0.0)
        np.fill_diagonal(Q_c, -Q_c.sum(axis=1))
        # Normalize to mean rate 1
        mean_rate = -np.sum(pi_c * np.diag(Q_c))
        if mean_rate > 0:
            Q_c /= mean_rate
        Q_classes[c] = Q_c

    return Q_classes, pi_classes, np.array(weights)


def gamma_rates(alpha, n_cat):
    """Discretized gamma rate multipliers (mean 1) using quantile midpoints.

    Uses the Wilson-Hilferty approximation (same as PAML/IQ-TREE).
    """
    from scipy.stats import gamma as gamma_dist
    # Quantile midpoints
    mid_probs = (np.arange(n_cat) + 0.5) / n_cat
    rates = gamma_dist.ppf(mid_probs, a=alpha, scale=1.0/alpha)
    rates = rates / rates.mean()  # normalize to mean 1
    return rates


# ── Vectorized Felsenstein for mixture models ───────────────────────────

def _matrix_exp_numpy(Q, pi, t):
    """Compute P(t) = exp(Q*t) via eigendecomposition in numpy."""
    n = Q.shape[0]
    if t <= 0:
        return np.eye(n)
    sqrt_pi = np.sqrt(np.maximum(pi, 1e-300))
    S = Q * (sqrt_pi[:, None] / sqrt_pi[None, :])
    S = (S + S.T) / 2
    eigvals, eigvecs = np.linalg.eigh(S)
    M = eigvecs * np.exp(eigvals * t)[None, :]
    M = M @ eigvecs.T
    M = M * (sqrt_pi[None, :] / sqrt_pi[:, None])
    return np.maximum(M, 0.0)


@jax.jit
def _felsenstein_batch(leaf_obs, all_trans_matrices, all_pi, all_P_target,
                       all_weights, postorder_ids, children_ids,
                       n_children, is_leaf):
    """Batch Felsenstein over (columns x mixture components).

    Args:
        leaf_obs: (L, N, A) leaf conditional likelihoods
        all_trans_matrices: (M, N, A, A) transition matrices for M mixture components
        all_pi: (M, A) equilibrium for each component
        all_P_target: (M, A, A) propagation matrices to target
        all_weights: (M,) mixture weights
        postorder_ids: (N,) int array
        children_ids: (N, max_ch) int array
        n_children: (N,) int array
        is_leaf: (N,) bool array

    Returns:
        posteriors: (L, A) weighted mixture posterior at target
    """
    N = is_leaf.shape[0]
    AA = all_pi.shape[1]
    max_ch = children_ids.shape[1]
    root_id = postorder_ids[-1]

    def _peel_one_component_column(trans_matrices, pi, P_target, leaf_ob):
        """Felsenstein for one column, one mixture component."""
        def _peel_step(CL, nid):
            leaf_cl = leaf_ob[nid]
            def _child_msg(ci):
                child_id = children_ids[nid, ci]
                safe_child_id = jnp.maximum(child_id, 0)
                valid = (ci < n_children[nid]) & (child_id >= 0)
                child_cl = CL[safe_child_id]
                msg = trans_matrices[safe_child_id] @ child_cl
                msg = jnp.where(valid, msg, jnp.ones(AA))
                return msg
            child_msgs = jax.vmap(lambda ci: _child_msg(ci))(jnp.arange(max_ch))
            int_cl = jnp.prod(child_msgs, axis=0)
            int_max = jnp.maximum(jnp.max(int_cl), 1e-300)
            int_cl = int_cl / int_max
            cl = jnp.where(is_leaf[nid], leaf_cl, int_cl)
            CL = CL.at[nid].set(cl)
            return CL, None

        CL_init = jnp.zeros((N, AA))
        CL_final, _ = jax.lax.scan(_peel_step, CL_init, postorder_ids)
        root_cl = CL_final[root_id]

        # Root posterior
        joint = pi * root_cl
        Z = jnp.sum(joint)
        root_post = joint / jnp.maximum(Z, 1e-300)

        # Propagate to target
        target_post = P_target.T @ root_post
        return target_post

    # vmap over columns for one component
    def _peel_one_component(trans_matrices, pi, P_target):
        return jax.vmap(
            lambda lo: _peel_one_component_column(trans_matrices, pi, P_target, lo)
        )(leaf_obs)  # (L, A)

    # vmap over mixture components
    all_target_posts = jax.vmap(_peel_one_component)(
        all_trans_matrices, all_pi, all_P_target)  # (M, L, A)

    # Weighted sum: (M, L, A) with weights (M,)
    weighted = all_target_posts * all_weights[:, None, None]
    total = jnp.sum(weighted, axis=0)  # (L, A)

    # Normalize
    Z = jnp.sum(total, axis=1, keepdims=True)
    total = total / jnp.maximum(Z, 1e-300)

    return total


def felsenstein_mixture_predict_leaf(
    pruned_tree, pruned_msa_int, target_dist_to_root,
    Q_classes, pi_classes, class_weights, gamma_rates_arr,
):
    """Predict leaf amino acid posteriors using Felsenstein + mixture model.

    Vectorized over columns AND mixture components (class x gamma) in a
    single JIT call for efficiency.

    Args:
        pruned_tree: tree with target leaf removed
        pruned_msa_int: dict {name: int_array}, -1 for gaps
        target_dist_to_root: distance from target leaf to root
        Q_classes: (C, 20, 20) rate matrices per class
        pi_classes: (C, 20) equilibrium per class
        class_weights: (C,) mixture weights
        gamma_rates_arr: (G,) gamma rate multipliers

    Returns:
        predictions: (L,) int array, predicted AA per column
        posteriors: (L, 20) posterior probabilities
    """
    n_classes = Q_classes.shape[0]
    n_gamma = len(gamma_rates_arr)
    M = n_classes * n_gamma  # total mixture components

    names = list(pruned_msa_int.keys())
    L = len(next(iter(pruned_msa_int.values())))

    # Flatten tree
    (node_list, node_to_id, parent_ids, children_ids,
     n_children, is_leaf, branch_lengths, postorder_ids) = \
        _flatten_tree_for_felsenstein(pruned_tree)
    N = len(node_list)

    # Build name -> node_id mapping
    name_to_nid = {}
    for node in node_list:
        if node.name is not None:
            name_to_nid[node.name] = node_to_id[id(node)]

    # Build leaf observation array
    leaf_obs, is_all_gap = _build_leaf_obs_array(pruned_msa_int, name_to_nid, N, L, A)

    # Precompute ALL transition matrices in numpy (no JAX overhead per matrix)
    all_trans = np.zeros((M, N, A, A), dtype=np.float64)
    all_pi = np.zeros((M, A), dtype=np.float64)
    all_P_target = np.zeros((M, A, A), dtype=np.float64)
    all_weights = np.zeros(M, dtype=np.float64)

    m = 0
    for c in range(n_classes):
        Q_c = Q_classes[c]
        pi_c = pi_classes[c]
        for g_idx in range(n_gamma):
            rho = gamma_rates_arr[g_idx]
            Q_scaled = Q_c * rho

            for nid in range(N):
                if parent_ids[nid] >= 0:
                    t = branch_lengths[nid]
                    all_trans[m, nid] = _matrix_exp_numpy(Q_scaled, pi_c, t)

            all_P_target[m] = _matrix_exp_numpy(Q_scaled, pi_c, target_dist_to_root)
            all_pi[m] = pi_c
            all_weights[m] = class_weights[c] / n_gamma
            m += 1

    # Run batch Felsenstein (single JIT call)
    posteriors = np.array(_felsenstein_batch(
        jnp.asarray(leaf_obs),
        jnp.asarray(all_trans),
        jnp.asarray(all_pi),
        jnp.asarray(all_P_target),
        jnp.asarray(all_weights),
        jnp.asarray(postorder_ids),
        jnp.asarray(children_ids),
        jnp.asarray(n_children),
        jnp.asarray(is_leaf),
    ))

    predictions = np.argmax(posteriors, axis=1).astype(np.int32)
    predictions = np.where(is_all_gap, -1, predictions)

    return predictions, posteriors


# ── Main evaluation ─────────────────────────────────────────────────────

def evaluate_family(fam_id, aligned_seqs, tree, target_name,
                    remaining_names, Q_lg, pi_lg,
                    models_config, max_len=256, rng=None):
    """Evaluate all models on one family + target.

    Args:
        models_config: dict of {model_name: (Q_classes, pi_classes, weights, gamma_rates)}

    Returns:
        dict with per-model accuracy, or None if skipped
    """
    L = len(next(iter(aligned_seqs.values())))

    # Crop if needed
    L_crop = min(L, max_len)
    start = 0
    if L > max_len and rng is not None:
        start = rng.randint(0, L - max_len + 1)

    # Build pruned tree
    pruned_tree, target_bl = remove_leaf(tree, target_name)
    if pruned_tree is None:
        return None

    # Compute distance from target to root
    # target_bl is the branch from target to its parent
    # We need total distance to root of pruned tree
    target_leaf = find_leaf_in_tree(tree, target_name)
    if target_leaf is None:
        return None
    total_dist = distance_to_root(target_leaf)

    # Build pruned MSA (integer-encoded)
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

    # Target sequence for comparison
    target_aligned = aligned_seqs[target_name]

    results = {'family': fam_id, 'target': target_name, 'total_dist': total_dist}
    total_count = 0
    total_correct = {}

    for model_name, (Q_classes, pi_classes, weights, gamma_r) in models_config.items():
        total_correct[model_name] = 0

    # Also run plain LG08 for comparison
    total_correct['lg08'] = 0

    for col_idx in range(L_crop):
        col = start + col_idx
        if col >= len(target_aligned):
            break

        target_char = target_aligned[col].upper()

        # Build gap data for Fitch
        column_gap = {}
        for n in remaining:
            if n in pruned_leaves:
                c_idx = col_idx
                ch = aligned_seqs[n][col].upper() if col < len(aligned_seqs[n]) else '-'
                column_gap[n] = (ch in '.-')

        # Fitch gap prediction (on original tree)
        is_gap_pred = fitch_gap_predict(tree, target_name, column_gap)

        if target_char in '.-':
            # Target is gap
            total_count += 1
            if is_gap_pred:
                for model_name in total_correct:
                    total_correct[model_name] += 1
        elif target_char in AA_TO_INT and AA_TO_INT[target_char] < 20:
            # Target is amino acid
            total_count += 1
            if is_gap_pred:
                # Predicted gap, wrong for all models
                pass
            else:
                true_aa = AA_TO_INT[target_char]

                # Evaluate each mixture model
                # (predictions already computed for the full column range)
                pass  # handled below

    # Instead of per-column loop for mixture models, batch-compute predictions
    # Then evaluate column by column

    # Compute predictions for all models at once
    model_preds = {}
    for model_name, (Q_classes, pi_classes, weights, gamma_r) in models_config.items():
        preds, posts = felsenstein_mixture_predict_leaf(
            pruned_tree, pruned_msa, total_dist,
            Q_classes, pi_classes, weights, gamma_r)
        model_preds[model_name] = preds

    # LG08 baseline (single class, no gamma)
    Q_lg_arr = np.array(Q_lg).reshape(1, A, A)
    pi_lg_arr = np.array(pi_lg).reshape(1, A)
    lg_weights = np.array([1.0])
    lg_gamma = np.array([1.0])
    preds_lg, _ = felsenstein_mixture_predict_leaf(
        pruned_tree, pruned_msa, total_dist,
        Q_lg_arr, pi_lg_arr, lg_weights, lg_gamma)
    model_preds['lg08'] = preds_lg

    # Evaluate
    total_count = 0
    for model_name in list(models_config.keys()) + ['lg08']:
        total_correct[model_name] = 0

    for col_idx in range(L_crop):
        col = start + col_idx
        if col >= len(target_aligned):
            break

        target_char = target_aligned[col].upper()

        # Fitch gap prediction
        column_gap = {}
        for n in remaining:
            if n in pruned_leaves:
                ch = aligned_seqs[n][col].upper() if col < len(aligned_seqs[n]) else '-'
                column_gap[n] = (ch in '.-')
        is_gap_pred = fitch_gap_predict(tree, target_name, column_gap)

        if target_char in '.-':
            total_count += 1
            if is_gap_pred:
                for mn in total_correct:
                    total_correct[mn] += 1
        elif target_char in AA_TO_INT and AA_TO_INT[target_char] < 20:
            total_count += 1
            if is_gap_pred:
                pass  # predicted gap, wrong
            else:
                true_aa = AA_TO_INT[target_char]
                for mn in total_correct:
                    if model_preds[mn][col_idx] == true_aa:
                        total_correct[mn] += 1

    if total_count == 0:
        return None

    for mn in total_correct:
        results[f'acc_{mn}'] = total_correct[mn] / total_count
    results['n_columns'] = total_count
    results['branch_length'] = float(target_bl) if target_bl else 0.0

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Felsenstein C10/C20+gamma benchmark")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of families to evaluate")
    parser.add_argument("--max-rows", type=int, default=32,
                        help="Max sequences per family")
    parser.add_argument("--max-len", type=int, default=256,
                        help="Max alignment length (crop if longer)")
    parser.add_argument("--gamma-cats", type=int, default=4,
                        help="Number of gamma rate categories")
    parser.add_argument("--gamma-alpha", type=float, default=1.0,
                        help="Gamma shape parameter")
    parser.add_argument("--seed", type=int, default=99999,
                        help="Random seed (matches carabs val seed)")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    # Load split
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    families = splits[args.split]

    # Filter to families with .sto files and >= 4 sequences
    valid_families = []
    for fam in families:
        sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")
        if os.path.exists(sto_path):
            valid_families.append(fam)
    print(f"Found {len(valid_families)}/{len(families)} {args.split} families with .sto files")

    # Build models
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg_np = np.array(Q_lg)
    pi_lg_np = np.array(pi_lg)

    gamma_r = gamma_rates(args.gamma_alpha, args.gamma_cats)
    print(f"Gamma rates (alpha={args.gamma_alpha}, G={args.gamma_cats}): {gamma_r}")

    # C10
    Q_c10, pi_c10, w_c10 = build_class_rate_matrices('C10')
    # C20
    Q_c20, pi_c20, w_c20 = build_class_rate_matrices('C20')

    # LG08 + gamma (no mixture)
    Q_lg_1 = Q_lg_np.reshape(1, A, A)
    pi_lg_1 = pi_lg_np.reshape(1, A)
    w_lg_1 = np.array([1.0])

    models_config = {
        'lg08_gamma': (Q_lg_1, pi_lg_1, w_lg_1, gamma_r),
        'c10': (Q_c10, pi_c10, w_c10, np.array([1.0])),
        'c10_gamma': (Q_c10, pi_c10, w_c10, gamma_r),
        'c20': (Q_c20, pi_c20, w_c20, np.array([1.0])),
        'c20_gamma': (Q_c20, pi_c20, w_c20, gamma_r),
    }

    rng = np.random.RandomState(args.seed)
    all_results = []

    # Accumulators
    acc_totals = {mn: {'correct': 0, 'count': 0}
                  for mn in list(models_config.keys()) + ['lg08']}

    t0 = time.time()
    n_processed = 0
    n_skipped = 0

    max_attempts = args.n_samples * 5  # allow skips without stopping early
    for i in range(max_attempts):
        if n_processed >= args.n_samples:
            break

        # Sample a family
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

        # Build NJ tree from LG08 pairwise distances
        try:
            nj_names, D = msa_pairwise_distances(aligned_seqs, Q_lg, pi_lg)
            tree = neighbor_joining(D, nj_names)
        except Exception:
            n_skipped += 1
            continue

        # Pick target leaf
        leaves = [n.name for n in tree.leaves() if n.name in set(names)]
        if len(leaves) < 4:
            n_skipped += 1
            continue

        target_name = leaves[rng.randint(len(leaves))]
        remaining_names = [n for n in names if n != target_name]

        # Subsample if too many
        if len(remaining_names) > args.max_rows:
            # Keep leaves that are in the tree
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
                models_config, max_len=args.max_len, rng=rng)
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
        for mn in acc_totals:
            key = f'acc_{mn}'
            if key in result and result[key] is not None:
                acc_totals[mn]['correct'] += int(result[key] * result['n_columns'])
                acc_totals[mn]['count'] += result['n_columns']

        if n_processed % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / n_processed * (args.n_samples - n_processed) if n_processed > 0 else 0
            print(f"\n--- Sample {n_processed}/{args.n_samples} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                  f"{n_skipped} skipped) ---")
            for mn in sorted(acc_totals.keys()):
                if acc_totals[mn]['count'] > 0:
                    acc = acc_totals[mn]['correct'] / acc_totals[mn]['count']
                    print(f"  {mn:15s}: {acc:.4f} "
                          f"({acc_totals[mn]['correct']}/{acc_totals[mn]['count']})")

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"Felsenstein C10/C20+gamma benchmark ({args.split})")
    print(f"  Samples: {n_processed}, Skipped: {n_skipped}, Time: {elapsed:.0f}s")
    print(f"  Gamma: alpha={args.gamma_alpha}, G={args.gamma_cats}")
    print(f"{'='*70}")
    for mn in sorted(acc_totals.keys()):
        if acc_totals[mn]['count'] > 0:
            acc = acc_totals[mn]['correct'] / acc_totals[mn]['count']
            print(f"  {mn:15s}: {acc:.4f} "
                  f"({acc_totals[mn]['correct']}/{acc_totals[mn]['count']})")
    print(f"{'='*70}")

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"felsenstein_c10_c20_results_{args.split}.json")

    summary = {
        'args': vars(args),
        'n_processed': n_processed,
        'n_skipped': n_skipped,
        'elapsed_seconds': elapsed,
        'gamma_rates': gamma_r.tolist(),
        'accuracies': {},
        'per_family': all_results,
    }
    for mn in sorted(acc_totals.keys()):
        if acc_totals[mn]['count'] > 0:
            summary['accuracies'][mn] = {
                'accuracy': acc_totals[mn]['correct'] / acc_totals[mn]['count'],
                'correct': acc_totals[mn]['correct'],
                'count': acc_totals[mn]['count'],
            }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
