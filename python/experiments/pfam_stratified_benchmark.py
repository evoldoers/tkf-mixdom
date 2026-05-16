#!/usr/bin/env python3
"""Stratified Pfam held-out benchmark: Felsenstein vs TreeVarAnc.

Selects test families stratified by divergence regime and tree size,
runs both Felsenstein and TreeVarAnc reconstruction on each held-out
leaf, reports accuracy per stratum.

Usage:
    cd python && uv run python experiments/pfam_stratified_benchmark.py \
        --n-per-stratum 15 --output experiments/pfam_stratified.json
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tkfmixdom.jax.util.io import parse_newick, TreeNode, AA_TO_INT, AMINO_ACIDS
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.core.params import S, M, I, D, E
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
from tkfmixdom.jax.tree.tree_varanc import (
    tree_varanc,
    build_tkf91_branch_wfst,
    build_tkf91_root_wfst,
    infer_internal_presence,
    name_internal_nodes,
    NEG_INF, BOS_I, BOS_D,
)

# Reuse infrastructure from ancrec_benchmark
from experiments.ancrec_benchmark import (
    parse_sto,
    msa_pairwise_distances,
    remove_leaf,
    find_parent_of_leaf,
    needleman_wunsch_identity,
    method_felsenstein,
    build_mixdom_wfst_log,
    TKF92_INS_RATE, TKF92_DEL_RATE, TKF92_EXT,
    PARAMS_PATH as MIXDOM_PARAMS_PATH,
    PFAM_DIR, SPLITS_PATH,
)
from tkfmixdom.jax.tree.guide_tree import neighbor_joining

# ======================================================================
# Configuration
# ======================================================================

# Ordered: moderate first (most informative), then distant, then close
DIVERGENCE_BINS = [
    ('moderate', 0.30, 0.50),
    ('distant',  0.00, 0.30),
    ('close',    0.50, 1.01),
]

SIZE_BINS = [
    ('medium', 12, 25),
    ('small',  5,  12),
    ('large',  25, 100),
]


# ======================================================================
# Family selection and characterization
# ======================================================================

def characterize_family(fam_id, Q, pi):
    """Compute divergence and size stats for a Pfam family.

    Returns dict with n_seqs, mean_identity, median_length, or None if invalid.
    """
    sto_path = os.path.join(PFAM_DIR, f"{fam_id}.sto")
    if not os.path.exists(sto_path):
        return None

    try:
        aligned_seqs = parse_sto(sto_path)
    except Exception:
        return None

    n_seqs = len(aligned_seqs)
    if n_seqs < 5:
        return None

    names = list(aligned_seqs.keys())
    aln_strs = list(aligned_seqs.values())

    # Pairwise identity (sample)
    rng = np.random.RandomState(hash(fam_id) % 2**31)
    n_pairs = min(30, n_seqs * (n_seqs - 1) // 2)
    ids = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_seqs, 2, replace=False)
        s1, s2 = aln_strs[i], aln_strs[j]
        L = min(len(s1), len(s2))
        matches = sum(1 for k in range(L) if s1[k] == s2[k] and s1[k] not in '.-')
        aligned = sum(1 for k in range(L) if s1[k] not in '.-' and s2[k] not in '.-')
        if aligned > 0:
            ids.append(matches / aligned)

    if not ids:
        return None

    lengths = [len(seq.replace('-', '').replace('.', '')) for seq in aln_strs]

    return {
        'fam_id': fam_id,
        'n_seqs': n_seqs,
        'mean_identity': float(np.mean(ids)),
        'median_length': float(np.median(lengths)),
        'aln_width': len(aln_strs[0]),
    }


def select_stratified_families(n_per_stratum=15, seed=42):
    """Select families stratified by divergence × size."""
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    test_families = splits['test']

    Q, pi = rate_matrix_lg()
    Q, pi = np.asarray(Q), np.asarray(pi)

    print(f"Characterizing {len(test_families)} test families...")
    all_stats = []
    for i, fam_id in enumerate(test_families):
        if i % 500 == 0:
            print(f"  {i}/{len(test_families)}...")
        stats = characterize_family(fam_id, Q, pi)
        if stats is not None:
            all_stats.append(stats)

    print(f"Valid families: {len(all_stats)}")

    # Stratify
    rng = np.random.RandomState(seed)
    selected = {}
    for div_name, div_lo, div_hi in DIVERGENCE_BINS:
        for size_name, size_lo, size_hi in SIZE_BINS:
            key = (div_name, size_name)
            candidates = [
                s for s in all_stats
                if div_lo <= s['mean_identity'] < div_hi
                and size_lo <= s['n_seqs'] < size_hi
                and s['median_length'] >= 30
                and s['median_length'] <= 800
            ]
            rng.shuffle(candidates)
            chosen = candidates[:n_per_stratum]
            selected[key] = chosen
            print(f"  {div_name}/{size_name}: {len(candidates)} candidates, selected {len(chosen)}")

    return selected


# ======================================================================
# Subsample leaves for tree size control
# ======================================================================

def subsample_leaves(aligned_seqs, tree, target_n, rng):
    """Subsample leaves to target size, keeping a connected subtree.

    Randomly selects target_n leaves and prunes the tree.
    """
    leaves = [n.name for n in tree.leaves()]
    if len(leaves) <= target_n:
        return aligned_seqs, tree, leaves

    chosen = list(rng.choice(leaves, target_n, replace=False))

    # Prune tree to chosen leaves
    pruned_tree = _prune_tree(tree, set(chosen))
    pruned_seqs = {n: aligned_seqs[n] for n in chosen if n in aligned_seqs}

    return pruned_seqs, pruned_tree, chosen


def _prune_tree(tree, keep_leaves):
    """Prune a tree to keep only specified leaves."""
    import copy

    def _prune(node):
        if node.is_leaf:
            if node.name in keep_leaves:
                new = TreeNode(name=node.name)
                new.branch_length = node.branch_length
                return new
            return None

        children_pruned = []
        for child in node.children:
            p = _prune(child)
            if p is not None:
                children_pruned.append(p)

        if len(children_pruned) == 0:
            return None
        elif len(children_pruned) == 1:
            # Collapse: add this node's branch length to child's
            child = children_pruned[0]
            if child.branch_length is not None and node.branch_length is not None:
                child.branch_length += node.branch_length
            return child
        else:
            new = TreeNode(name=node.name)
            new.branch_length = node.branch_length
            for child in children_pruned:
                new.children.append(child)
                child.parent = new
            return new

    result = _prune(tree)
    if result is not None:
        result.parent = None
        result.branch_length = None
    return result


# ======================================================================
# TreeVarAnc method
# ======================================================================

def _add_bos_keys_to_wfst(wfst_log):
    """Add BOS_I/BOS_D keys to a MixDom WFST log dict."""
    d = dict(wfst_log)
    d['log_p_bos_i_m'] = np.asarray(d['log_p_im'])[0].copy()
    d['log_p_bos_i_i'] = np.asarray(d['log_p_ii'])[0].copy()
    d['log_p_bos_i_d'] = np.asarray(d['log_p_id'])[0].copy()
    d['log_p_bos_i_e'] = np.asarray(d['log_p_ie'])[0].copy()
    d['log_p_bos_d_m'] = np.asarray(d['log_p_dm'])[:, 0].copy()
    d['log_p_bos_d_i'] = np.asarray(d['log_p_di'])[:, 0].copy()
    d['log_p_bos_d_d'] = np.asarray(d['log_p_dd'])[:, 0].copy()
    d['log_p_bos_d_e'] = np.asarray(d['log_p_de'])[:, 0].copy()
    return d


def method_tree_varanc(pruned_tree, pruned_msa_int, Q, pi,
                        wfst_per_edge, singlet_wfst):
    """TreeVarAnc reconstruction: returns root sequence at Fitch-present positions."""
    name_internal_nodes(pruned_tree)

    # Build MSA presence from aligned sequences
    leaf_presence = {n: np.array(pruned_msa_int[n] >= 0, dtype=bool)
                     for n in pruned_msa_int}
    msa_presence = infer_internal_presence(pruned_tree, leaf_presence)

    # Run tree_varanc
    node_post, edge_post, elbo, elbo_trace, diag = tree_varanc(
        pruned_tree, msa_presence, pruned_msa_int,
        wfst_per_edge, singlet_wfst, pi,
        n_iter=20, tol=1e-6, verbose=False)

    # Extract root MAP sequence at present positions
    root_name = pruned_tree.name
    root_pres = msa_presence[root_name]
    root_post_arr = node_post.get(root_name)

    if root_post_arr is None or len(root_post_arr) == 0:
        return np.array([], dtype=np.int32), diag

    root_seq = np.argmax(root_post_arr, axis=1).astype(np.int32)
    return root_seq, diag


# ======================================================================
# Single holdout evaluation
# ======================================================================

def run_holdout_both(family_id, held_out, seq_names, aln_int, ungapped, tree,
                     Q, pi, wfst_per_edge, singlet_wfst, verbose=True):
    """Run both Felsenstein and TreeVarAnc on one held-out leaf."""
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

    pruned_msa = {n: aln_int[n] for n in remaining}
    true_seq = ungapped[held_out]

    if len(true_seq) == 0:
        return None

    result = {
        "family": family_id,
        "held_out": held_out,
        "tau_leaf": float(branch_len),
        "true_seq_len": len(true_seq),
        "n_remaining": len(remaining),
    }

    # Felsenstein
    t0 = time.time()
    try:
        fels_seq = method_felsenstein(pruned_tree, pruned_msa, Q, pi)
        t_fels = time.time() - t0
        if len(fels_seq) > 0:
            identity, aligned, matches = needleman_wunsch_identity(fels_seq, true_seq)
        else:
            identity, aligned, matches = 0.0, 0, 0
        result['identity_felsenstein'] = float(identity)
        result['time_felsenstein'] = t_fels
        result['status_felsenstein'] = 'ok'
        result['anc_len_felsenstein'] = len(fels_seq)
    except Exception as e:
        result['identity_felsenstein'] = None
        result['time_felsenstein'] = time.time() - t0
        result['status_felsenstein'] = f'error: {str(e)[:200]}'

    # TreeVarAnc
    t0 = time.time()
    try:
        # Build per-edge WFSTs for the pruned tree
        pruned_wfst = {}
        for node in pruned_tree.preorder():
            if node.is_root:
                continue
            key = (node.parent.name, node.name)
            # Find matching WFST by branch length (closest match)
            pruned_wfst[key] = _get_wfst_for_branch(
                wfst_per_edge, node.branch_length)

        tva_seq, tva_diag = method_tree_varanc(
            pruned_tree, pruned_msa, Q, pi, pruned_wfst, singlet_wfst)
        t_tva = time.time() - t0
        if len(tva_seq) > 0:
            identity, aligned, matches = needleman_wunsch_identity(tva_seq, true_seq)
        else:
            identity, aligned, matches = 0.0, 0, 0
        result['identity_tree_varanc'] = float(identity)
        result['time_tree_varanc'] = t_tva
        result['status_tree_varanc'] = 'ok'
        result['anc_len_tree_varanc'] = len(tva_seq)
        result['n_sweeps'] = tva_diag.get('n_sweeps', 0)
        result['converged'] = tva_diag.get('converged', False)
    except Exception as e:
        result['identity_tree_varanc'] = None
        result['time_tree_varanc'] = time.time() - t0
        result['status_tree_varanc'] = f'error: {str(e)[:200]}'
        if verbose:
            traceback.print_exc()

    if verbose:
        f_id = result.get('identity_felsenstein', None)
        t_id = result.get('identity_tree_varanc', None)
        f_str = f"{f_id:.3f}" if f_id is not None else "ERR"
        t_str = f"{t_id:.3f}" if t_id is not None else "ERR"
        print(f"    {held_out} (tau={branch_len:.3f}): fels={f_str} tva={t_str}")

    return result


# WFST cache for branch lengths
_wfst_branch_cache = {}


def _get_wfst_for_branch(wfst_per_edge, branch_length):
    """Get a WFST for a given branch length, using cache."""
    t = max(branch_length, 1e-6)
    t_key = round(t, 4)

    if t_key in _wfst_branch_cache:
        return _wfst_branch_cache[t_key]

    # Try to find an existing WFST with similar branch length
    for key, wfst in wfst_per_edge.items():
        existing_t = round(max(1e-6, _get_bl_from_key(key)), 4)
        if existing_t == t_key:
            _wfst_branch_cache[t_key] = wfst
            return wfst

    # Build new one using MixDom distillation
    wfst = _build_mixdom_wfst_for_t(t)
    _wfst_branch_cache[t_key] = wfst
    return wfst


def _get_bl_from_key(key):
    """Dummy — we don't store branch length in the key."""
    return 0.0


_mixdom_state = {}


def _init_mixdom():
    """Initialize MixDom params (once)."""
    if 'params' in _mixdom_state:
        return
    from tkfmixdom.jax.distill.maraschino import (
        load_params, precompute_mixdom,
    )
    params, n_domains, n_classes = load_params(MIXDOM_PARAMS_PATH)
    precomp = precompute_mixdom(params, n_classes)
    _mixdom_state['params'] = params
    _mixdom_state['n_classes'] = n_classes
    _mixdom_state['precomp'] = precomp


def _build_mixdom_wfst_for_t(t):
    """Build MixDom WFST for a specific branch length."""
    _init_mixdom()
    from tkfmixdom.jax.distill.maraschino import (
        distill_mixdom, normalize_freqs_wfst,
    )
    dist = distill_mixdom(
        _mixdom_state['params'], t,
        _mixdom_state['n_classes'],
        _mixdom_state['precomp'])
    wfst = normalize_freqs_wfst(dist)
    log_wfst = build_mixdom_wfst_log(wfst)
    return _add_bos_keys_to_wfst(log_wfst)


def _build_mixdom_singlet():
    """Build MixDom root singlet WFST."""
    _init_mixdom()
    from tkfmixdom.jax.distill.maraschino import (
        distill_mixdom, normalize_freqs_wfst,
    )
    params = _mixdom_state['params']
    n_classes = _mixdom_state['n_classes']
    precomp = _mixdom_state['precomp']

    # Distill at arbitrary t for singlet
    dist = distill_mixdom(params, 0.1, n_classes, precomp)

    f_singlet = np.asarray(dist['f_singlet'])
    f_start = np.asarray(dist['f_singlet_start'])
    f_end = np.asarray(dist['f_singlet_end'])

    AA = f_singlet.shape[0]
    sl = lambda x: np.log(np.maximum(x, 1e-300))

    total_start = np.sum(f_start)
    p_start_emit = f_start / max(total_start + (1.0 - total_start), 1e-30)
    p_start_end = 1.0 - np.sum(p_start_emit)

    log_p_si = np.broadcast_to(sl(p_start_emit)[None, :], (AA, AA)).copy()
    log_p_se = float(sl(max(p_start_end, 1e-300)))

    Z = np.sum(f_singlet, axis=1) + f_end
    Z = np.maximum(Z, 1e-300)
    p_ii = f_singlet / Z[:, None]
    p_ie = f_end / Z

    log_p_ii = np.broadcast_to(sl(p_ii)[None, :, :], (AA, AA, AA)).copy()
    log_p_ie = np.broadcast_to(sl(p_ie)[None, :], (AA, AA)).copy()

    impossible_4d = np.full((AA, AA, AA, AA), NEG_INF)
    impossible_3d = np.full((AA, AA, AA), NEG_INF)
    impossible_2d = np.full((AA, AA), NEG_INF)

    return {
        'log_p_mm': impossible_4d, 'log_p_mi': impossible_3d,
        'log_p_md': impossible_3d, 'log_p_me': impossible_2d,
        'log_p_im': impossible_4d, 'log_p_ii': log_p_ii,
        'log_p_id': impossible_3d, 'log_p_ie': log_p_ie,
        'log_p_dm': impossible_4d, 'log_p_dd': impossible_3d,
        'log_p_di': impossible_3d, 'log_p_de': impossible_2d,
        'log_p_sm': impossible_2d, 'log_p_si': log_p_si,
        'log_p_sd': impossible_2d, 'log_p_se': log_p_se,
        'log_p_bos_i_m': np.full((AA, AA, AA), NEG_INF),
        'log_p_bos_i_i': log_p_ii[0].copy(),
        'log_p_bos_i_d': np.full((AA, AA), NEG_INF),
        'log_p_bos_i_e': log_p_ie[0].copy(),
    }


# ======================================================================
# Process one family
# ======================================================================

def process_family(fam_stats, Q, pi, wfst_per_edge_cache, singlet_wfst,
                   max_holdouts=5, verbose=True):
    """Process one Pfam family: build tree, run held-out evaluation."""
    fam_id = fam_stats['fam_id']
    sto_path = os.path.join(PFAM_DIR, f"{fam_id}.sto")

    aligned_seqs = parse_sto(sto_path)
    if len(aligned_seqs) < 3:
        return []

    names_list = list(aligned_seqs.keys())

    # Build NJ tree
    names, D = msa_pairwise_distances(aligned_seqs, Q, pi)
    tree = neighbor_joining(D, names)
    name_internal_nodes(tree)

    aln_int = {}
    ungapped = {}
    for name in names:
        seq_str = aligned_seqs[name]
        arr = np.array([AA_TO_INT.get(c.upper(), -1) if c not in '.-' else -1
                        for c in seq_str], dtype=np.int32)
        aln_int[name] = arr
        ungapped[name] = np.array([c for c in arr if c >= 0], dtype=np.int32)

    # Build per-edge MixDom WFSTs
    wfst_per_edge = {}
    for node in tree.preorder():
        if node.is_root:
            continue
        t = max(node.branch_length, 1e-6)
        t_key = round(t, 4)
        if t_key not in wfst_per_edge_cache:
            wfst_per_edge_cache[t_key] = _build_mixdom_wfst_for_t(t)
        wfst_per_edge[(node.parent.name, node.name)] = wfst_per_edge_cache[t_key]

    if verbose:
        print(f"\n  {fam_id}: {len(names)} seqs, aln={len(next(iter(aligned_seqs.values())))} "
              f"id={fam_stats['mean_identity']:.2f}")

    # Select holdouts (random subset for large families)
    rng = np.random.RandomState(hash(fam_id) % 2**31)
    holdout_names = list(names)
    rng.shuffle(holdout_names)
    holdout_names = holdout_names[:max_holdouts]

    results = []
    for held_out in holdout_names:
        if len(ungapped[held_out]) == 0:
            continue
        r = run_holdout_both(
            fam_id, held_out, names, aln_int, ungapped, tree,
            Q, pi, wfst_per_edge, singlet_wfst, verbose=verbose)
        if r is not None:
            r['mean_identity'] = fam_stats['mean_identity']
            r['n_family_seqs'] = fam_stats['n_seqs']
            results.append(r)

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Stratified Pfam benchmark')
    parser.add_argument('--n-per-stratum', type=int, default=15)
    parser.add_argument('--max-holdouts', type=int, default=5)
    parser.add_argument('--output', default='experiments/pfam_stratified.json')
    parser.add_argument('--budget', type=int, default=3600, help='Time budget in seconds')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    verbose = not args.quiet

    Q, pi = rate_matrix_lg()
    Q, pi = np.asarray(Q), np.asarray(pi)

    # Initialize MixDom
    print("Initializing MixDom...")
    _init_mixdom()
    singlet_wfst = _build_mixdom_singlet()
    print("MixDom initialized.")

    # Select families
    selected = select_stratified_families(n_per_stratum=args.n_per_stratum)

    # Run benchmark
    all_results = []
    wfst_cache = {}
    t_start = time.time()

    # Iterate in DIVERGENCE_BINS × SIZE_BINS order (not alphabetical)
    strata_order = [(d, s) for d, _, _ in DIVERGENCE_BINS for s, _, _ in SIZE_BINS]
    for (div_name, size_name) in strata_order:
        families = selected.get((div_name, size_name), [])
        elapsed = time.time() - t_start
        if elapsed > args.budget:
            print(f"Budget exhausted at stratum {div_name}/{size_name}")
            break

        print(f"\n{'='*60}")
        print(f"Stratum: {div_name} divergence, {size_name} tree ({len(families)} families)")
        print(f"{'='*60}")

        for fam_stats in families:
            elapsed = time.time() - t_start
            if elapsed > args.budget:
                break

            try:
                results = process_family(
                    fam_stats, Q, pi, wfst_cache, singlet_wfst,
                    max_holdouts=args.max_holdouts, verbose=verbose)
                for r in results:
                    r['div_regime'] = div_name
                    r['size_regime'] = size_name
                all_results.extend(results)
            except Exception as e:
                print(f"  {fam_stats['fam_id']}: FAILED - {e}")
                if verbose:
                    traceback.print_exc()

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Total holdouts: {len(all_results)}")

    # Summary
    print(f"\n{'='*80}")
    print("RESULTS BY STRATUM")
    print(f"{'='*80}")
    print(f"{'Div':>10} {'Size':>8} {'N':>4} | {'Fels':>8} {'TVA':>8} {'Diff':>7} | "
          f"{'Fels_t':>7} {'TVA_t':>7}")
    print('-' * 75)

    for div_name, _, _ in DIVERGENCE_BINS:
        for size_name, _, _ in SIZE_BINS:
            subset = [r for r in all_results
                      if r.get('div_regime') == div_name
                      and r.get('size_regime') == size_name]
            if not subset:
                continue

            f_ids = [r['identity_felsenstein'] for r in subset
                     if r.get('status_felsenstein') == 'ok' and r['identity_felsenstein'] is not None]
            t_ids = [r['identity_tree_varanc'] for r in subset
                     if r.get('status_tree_varanc') == 'ok' and r['identity_tree_varanc'] is not None]

            f_mean = np.mean(f_ids) if f_ids else float('nan')
            t_mean = np.mean(t_ids) if t_ids else float('nan')
            diff = t_mean - f_mean

            f_times = [r.get('time_felsenstein', 0) for r in subset
                       if r.get('status_felsenstein') == 'ok']
            t_times = [r.get('time_tree_varanc', 0) for r in subset
                       if r.get('status_tree_varanc') == 'ok']

            print(f"{div_name:>10} {size_name:>8} {len(subset):>4} | "
                  f"{f_mean:>8.3f} {t_mean:>8.3f} {diff:>+7.3f} | "
                  f"{np.mean(f_times) if f_times else 0:>7.2f} "
                  f"{np.mean(t_times) if t_times else 0:>7.2f}")

    # Save
    output = {
        'total_time': total_time,
        'n_holdouts': len(all_results),
        'strata': {f"{d}/{s}": len([r for r in all_results
                                     if r.get('div_regime') == d and r.get('size_regime') == s])
                   for d, _, _ in DIVERGENCE_BINS for s, _, _ in SIZE_BINS},
        'results': all_results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
