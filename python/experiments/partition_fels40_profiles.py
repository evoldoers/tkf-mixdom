#!/usr/bin/env python3
"""Partition reconstruction using fels40-derived presence/absence profiles.

Standard partition reconstruction uses Fitch parsimony to decide which
columns are "present" (have a residue) vs "absent" (gap) at each internal
node.  This script replaces Fitch with the fels40 probabilistic gap model:
run 40-state Felsenstein at the root of the rerooted tree to get gap
posterior probabilities, then threshold to get presence/absence.

Methods produced:
  - partition_d3f1_fels40gap  (3-domain, 1-fragment MixDom model)
  - partition_d5f1_fels40gap  (5-domain, 1-fragment MixDom model)

Datasets:
  --dataset unified_short | unified_long | treefam | balibase

Usage:
    cd python && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python -u \\
        experiments/partition_fels40_profiles.py --dataset unified_short
"""

import os
import sys
import json
import copy
import time
import argparse
import traceback

os.environ.setdefault('JAX_ENABLE_X64', '1')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)

# subby used for vectorized 40-state Felsenstein (via RootProb)

from tkfmixdom.jax.tree.tree_varanc import (
    name_internal_nodes, infer_internal_presence,
)
from tkfmixdom.jax.tree.partition_recon import build_inputs
from tkfmixdom.jax.tree.partition_recon_jax import (
    partition_recon_forward_backward_jax,
)
from tkfmixdom.jax.util.io import AA_TO_INT, parse_newick
from tkfmixdom.jax.distill.maraschino import load_params

from experiments.partition_recon_adapter import (
    mixdom_model_from_params, PartitionReconConfig,
)
from experiments.unified_reconstruction_benchmark import (
    prune_leaf_keep_parent, _nw_metrics,
)
from experiments.ancrec_benchmark import parse_sto, PFAM_DIR
from experiments.fels40_reconstruction_benchmark import (
    parse_fasta, parse_treefam_tree, build_msa_int,
    encode_aligned_seq, prune_tree_to_msa, run_fasttree,
    load_pfam_family, load_treefam_family, load_balibase_family,
    DATASET_CONFIGS, TREE_DIR, SAVE_EVERY,
)


t0 = time.time()
def log(msg): print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)


# ---------------------------------------------------------------------------
# Fels40 gap-posterior computation at the root
# ---------------------------------------------------------------------------

def _treenode_to_subby(tree_root):
    """Convert TreeNode to subby's Tree format + leaf ordering.

    Returns (subby_tree, leaf_names) where leaf_names[i] is the name of
    the i-th leaf in subby's preorder.
    """
    import sys
    sys.path.insert(0, os.path.expanduser('~/subby'))
    from subby.formats import Tree as SubbyTree

    # Preorder traversal
    nodes = []
    parent_map = {}

    def _traverse(node, parent_idx):
        idx = len(nodes)
        nodes.append(node)
        parent_map[idx] = parent_idx
        for child in node.children:
            _traverse(child, idx)

    _traverse(tree_root, -1)

    parent_index = np.array([parent_map[i] for i in range(len(nodes))], dtype=np.int32)
    dist_to_parent = np.array([
        max(n.branch_length if n.branch_length else 0.0, 1e-8)
        for n in nodes
    ], dtype=np.float64)
    dist_to_parent[0] = 0.0  # root

    # Build name-to-preorder-index mapping for leaves
    leaf_name_to_idx = {}
    leaf_names = []
    for i, n in enumerate(nodes):
        if n.is_leaf and n.name:
            leaf_name_to_idx[n.name] = i
            leaf_names.append(n.name)

    return SubbyTree(parentIndex=parent_index, distanceToParent=dist_to_parent), leaf_names, leaf_name_to_idx


def _fels40_root_gap_posteriors(tree_root, msa, Q40, pi40):
    """Run 40-state Felsenstein at every column using subby (vectorized).

    Args:
        tree_root: TreeNode (already rerooted, target leaf removed).
        msa: dict {leaf_name: (L,) int array, 0-19=AA, 20=gap}.
        Q40: (40,40) rate matrix.
        pi40: (40,) stationary distribution.

    Returns:
        root_p_gap: (L,) float array — P(gap) at root for each column.
    """
    import sys
    sys.path.insert(0, os.path.expanduser('~/subby'))
    from subby.jax import RootProb
    from subby.jax.types import RateModel

    leaf_names_ordered = list(msa.keys())
    msa_len = len(next(iter(msa.values())))
    R = len(leaf_names_ordered)
    A = 40

    # Convert tree
    subby_tree, subby_leaf_names, leaf_name_to_idx = _treenode_to_subby(tree_root)

    # subby expects (R, C) where R = total nodes in tree (leaves + internals)
    R_total = len(subby_tree.parentIndex)
    alignment = np.full((R_total, msa_len), -1, dtype=np.int32)
    # subby_leaf_names is already a name→preorder_index mapping
    subby_name_to_idx = leaf_name_to_idx

    # Build custom leaf likelihoods (R_total, C, 40)
    # Internal nodes get all-ones (unobserved)
    leaf_liks = np.ones((R_total, msa_len, A), dtype=np.float64)
    for name in leaf_names_ordered:
        if name in subby_name_to_idx:
            leaf_idx = subby_name_to_idx[name]
            alignment[leaf_idx] = msa[name]
            # Override with emission-weighted likelihoods for leaves
            seq = msa[name]
            for col in range(msa_len):
                c = seq[col]
                if 0 <= c < 20:
                    leaf_liks[leaf_idx, col, :] = 0.0
                    leaf_liks[leaf_idx, col, c] = 1.0
                else:
                    leaf_liks[leaf_idx, col, :20] = 0.0
                    leaf_liks[leaf_idx, col, 20:] = 1.0  # gap → gapped states

    # Build subby model
    model = RateModel(subRate=jnp.array(Q40), rootProb=jnp.array(pi40))

    # RootProb returns (A, C) posterior
    root_post = np.asarray(RootProb(
        alignment=jnp.array(alignment),
        tree=subby_tree,
        model=model,
        leaf_likelihoods=jnp.array(leaf_liks),
    ))  # (A, C) or (*H, A, C)

    # Sum gapped states (20-39) for P(gap)
    if root_post.ndim == 2:
        root_p_gap = root_post[20:, :].sum(axis=0)  # (C,)
    else:
        root_p_gap = root_post[..., 20:, :].sum(axis=-2)  # (*H, C)
        root_p_gap = root_p_gap.reshape(-1)[:msa_len]

    return root_p_gap


def _build_fels40_presence(tree_root, msa_remaining, Q40, pi40):
    """Build a presence dict using fels40 gap posteriors at the root
    and Fitch parsimony for remaining internal nodes.

    Args:
        tree_root: rerooted tree (root = target's former parent, target removed).
        msa_remaining: dict {leaf_name: (C,) int array, -1=gap, 0-19=AA}.
        Q40: (40,40) rate matrix.
        pi40: (40,) stationary distribution.

    Returns:
        presence: dict {node_name: (C,) bool array} for all nodes.
    """
    # Step 1: Leaf presence from observed data
    leaf_presence = {name: (seq >= 0)
                     for name, seq in msa_remaining.items()}

    # Step 2: Build 40-state MSA for fels40 (gaps -> code 20)
    msa40 = {}
    for name, seq in msa_remaining.items():
        s = seq.copy()
        s[s < 0] = 20
        s[s >= 20] = 20
        msa40[name] = s

    # Step 3: Compute root gap posteriors via fels40
    root_p_gap = _fels40_root_gap_posteriors(tree_root, msa40, Q40, pi40)

    # Step 4: Root presence = present if P(gap) < 0.5
    root_name = tree_root.name
    root_present = (root_p_gap < 0.5)

    # Step 5: Run Fitch parsimony with the fels40-derived root presence
    # We use infer_internal_presence which does Fitch from leaf data.
    # Then override the root with fels40-derived presence.
    presence = infer_internal_presence(tree_root, leaf_presence)
    presence[root_name] = root_present

    return presence


# ---------------------------------------------------------------------------
# Per-family partition reconstruction with fels40 presence
# ---------------------------------------------------------------------------

def run_partition_fels40(tree, held_out, remaining, msa, C,
                         Q40, pi40, model, config):
    """Run partition reconstruction with fels40-derived presence profiles.

    Returns (pred_seq, elapsed).
    """
    tf = time.time()

    # Re-root at target's parent, remove target leaf
    pruned_tree, _ = prune_leaf_keep_parent(tree, held_out)
    if pruned_tree is None:
        return np.array([], dtype=np.int32), 0.0
    name_internal_nodes(pruned_tree)

    # Build MSA for remaining leaves only
    pruned_leaf_names = {l.name for l in pruned_tree.leaves()}
    pruned_msa = {k: v for k, v in msa.items()
                  if k in pruned_leaf_names and k in remaining}

    if not pruned_msa:
        return np.array([], dtype=np.int32), 0.0

    # Build fels40-derived presence dict
    presence = _build_fels40_presence(pruned_tree, pruned_msa, Q40, pi40)

    # Build partition inputs with custom presence
    inputs = build_inputs(pruned_tree, pruned_msa, presence=presence)

    # Run partition reconstruction
    result = partition_recon_forward_backward_jax(inputs, model)

    # Extract predicted sequence (gap-stripped)
    L = result.root_residue_map.shape[0]
    pred_seq = np.array(
        [int(result.root_residue_map[c])
         for c in range(L) if result.root_residue_map[c] >= 0],
        dtype=np.int32,
    )
    elapsed = time.time() - tf
    return pred_seq, elapsed


def run_partition_fels40_balibase(tree, remaining, msa, C,
                                  Q40, pi40, model, config):
    """BAliBASE variant: tree already has only remaining leaves,
    reconstruct at root directly.
    """
    tf = time.time()
    name_internal_nodes(tree)

    # Build fels40-derived presence dict
    presence = _build_fels40_presence(tree, msa, Q40, pi40)

    # Build partition inputs with custom presence
    inputs = build_inputs(tree, msa, presence=presence)

    # Run partition reconstruction
    result = partition_recon_forward_backward_jax(inputs, model)

    L = result.root_residue_map.shape[0]
    pred_seq = np.array(
        [int(result.root_residue_map[c])
         for c in range(L) if result.root_residue_map[c] >= 0],
        dtype=np.int32,
    )
    elapsed = time.time() - tf
    return pred_seq, elapsed


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def _save(results, nw_accs_d3, nw_accs_d5, results_path, dataset):
    """Save results to JSON."""
    summary = {
        'dataset': dataset,
        'methods': ['partition_d3f1_fels40gap', 'partition_d5f1_fels40gap'],
        'n_families': len(results),
        'mean_nw_accuracy_d3f1': float(np.mean(nw_accs_d3)) if nw_accs_d3 else 0.0,
        'mean_nw_accuracy_d5f1': float(np.mean(nw_accs_d5)) if nw_accs_d5 else 0.0,
        'results': results,
    }
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f'Saved {len(results)} results to {results_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Partition reconstruction with fels40-derived presence profiles')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to run on')
    args = parser.parse_args()

    dataset = args.dataset
    config = DATASET_CONFIGS[dataset]

    log(f'Dataset: {dataset}')

    # Load fels40 model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'pfam', 'fels40_em.npz')
    data = np.load(model_path)
    Q40 = data['Q40']
    pi40 = data['pi40']
    log(f'Loaded fels40 model: Q40 {Q40.shape}, pi40 {pi40.shape}')

    # Load partition models
    log('Loading d3f1 params...')
    params_d3, _, _ = load_params('pfam/svi_bw_d3f1_full_best_val.npz')
    model_d3 = mixdom_model_from_params(params_d3)
    log('Loading d5f1 params...')
    params_d5, _, _ = load_params('pfam/svi_bw_d5f1_full_best_val.npz')
    model_d5 = mixdom_model_from_params(params_d5)
    partition_config = PartitionReconConfig(use_jax=True)

    # Load spec
    spec_path = os.path.join(os.path.dirname(__file__), config['spec_file'])
    with open(spec_path) as f:
        spec = json.load(f)
    families = spec['families']
    log(f'Loaded spec: {len(families)} families')

    # Dataset-specific directories
    if config['type'] == 'pfam':
        pfam_dir = os.path.expanduser(spec.get('pfam_dir', PFAM_DIR))
        tree_dir = os.path.expanduser(spec.get('tree_dir', TREE_DIR))
    elif config['type'] == 'treefam':
        treefam_dir = os.path.expanduser(spec.get('treefam_dir',
            '~/bio-datasets/data/treefam/treefam_family_data'))
    elif config['type'] == 'balibase':
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        balibase_ref_dir = os.path.join(proj_root, spec.get('balibase_ref_dir',
            os.path.expanduser('~/bio-datasets/data/balibase/bali3pdbm/ref')))

    # Output path
    results_path = os.path.join(os.path.dirname(__file__),
                                f'partition_fels40_reconstruction_{dataset}.json')

    # Resume support
    results = []
    done_fams = set()
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                rd = json.load(f)
            if isinstance(rd, dict) and 'results' in rd:
                results = list(rd['results'])
                done_fams = {r['family'] for r in results
                             if 'family' in r
                             and 'partition_d3f1_fels40gap' in r
                             and 'partition_d5f1_fels40gap' in r}
                log(f'Resume: {len(results)} prior results, '
                    f'{len(done_fams)} complete families')
        except Exception as e:
            log(f'Resume: failed to load: {e}')

    # Map existing results by family
    results_by_fam = {}
    for ri, r in enumerate(results):
        if 'family' in r:
            results_by_fam[r['family']] = ri

    n_done = 0
    n_errors = 0
    nw_accs_d3 = []
    nw_accs_d5 = []

    # Collect existing accuracies for running mean
    for r in results:
        if 'partition_d3f1_fels40gap' in r and 'nw_accuracy' in r['partition_d3f1_fels40gap']:
            nw_accs_d3.append(r['partition_d3f1_fels40gap']['nw_accuracy'])
        if 'partition_d5f1_fels40gap' in r and 'nw_accuracy' in r['partition_d5f1_fels40gap']:
            nw_accs_d5.append(r['partition_d5f1_fels40gap']['nw_accuracy'])

    for fi, fspec in enumerate(families):
        fam = fspec['family']
        held_out = fspec['held_out']
        remaining = fspec['remaining']
        true_seq = np.array(fspec['true_seq'], dtype=np.int32)

        if fam in done_fams:
            continue

        jax.clear_caches()  # OOM prevention (forces recompile; prefer geometric padding)

        try:
            # Load MSA and tree
            if config['type'] == 'pfam':
                msa, tree, C = load_pfam_family(fspec, pfam_dir, tree_dir)
            elif config['type'] == 'treefam':
                msa, tree, C = load_treefam_family(fspec, treefam_dir)
            elif config['type'] == 'balibase':
                msa, tree, C = load_balibase_family(fspec, balibase_ref_dir)

            d3_result = {}
            d5_result = {}

            # === partition_d3f1_fels40gap ===
            try:
                if config['type'] == 'balibase':
                    # BAliBASE: tree already has only remaining leaves
                    msa_rem = {n: msa[n] for n in remaining if n in msa}
                    pred_d3, t_d3 = run_partition_fels40_balibase(
                        tree, remaining, msa_rem, C, Q40, pi40,
                        model_d3, partition_config)
                else:
                    pred_d3, t_d3 = run_partition_fels40(
                        tree, held_out, remaining, msa, C, Q40, pi40,
                        model_d3, partition_config)

                nw_d3 = _nw_metrics(pred_d3, true_seq)
                d3_result = {
                    'pred_seq': pred_d3.tolist(),
                    'pred_len': len(pred_d3),
                    'true_len': len(true_seq),
                    'time': round(t_d3, 3),
                    **nw_d3,
                }
                nw_accs_d3.append(nw_d3['nw_accuracy'])
            except Exception as e:
                log(f'  {fam}: partition_d3f1_fels40gap error: {e}')
                traceback.print_exc()
                d3_result = {'nw_accuracy': -1.0, 'time': 0.0}

            # === partition_d5f1_fels40gap ===
            try:
                if config['type'] == 'balibase':
                    msa_rem = {n: msa[n] for n in remaining if n in msa}
                    pred_d5, t_d5 = run_partition_fels40_balibase(
                        tree, remaining, msa_rem, C, Q40, pi40,
                        model_d5, partition_config)
                else:
                    pred_d5, t_d5 = run_partition_fels40(
                        tree, held_out, remaining, msa, C, Q40, pi40,
                        model_d5, partition_config)

                nw_d5 = _nw_metrics(pred_d5, true_seq)
                d5_result = {
                    'pred_seq': pred_d5.tolist(),
                    'pred_len': len(pred_d5),
                    'true_len': len(true_seq),
                    'time': round(t_d5, 3),
                    **nw_d5,
                }
                nw_accs_d5.append(nw_d5['nw_accuracy'])
            except Exception as e:
                log(f'  {fam}: partition_d5f1_fels40gap error: {e}')
                traceback.print_exc()
                d5_result = {'nw_accuracy': -1.0, 'time': 0.0}

            # Store result
            if fam in results_by_fam:
                results[results_by_fam[fam]]['partition_d3f1_fels40gap'] = d3_result
                results[results_by_fam[fam]]['partition_d5f1_fels40gap'] = d5_result
            else:
                entry = {
                    'family': fam,
                    'held_out': held_out,
                    'true_len': len(true_seq),
                    'n_cols': C,
                    'K': len(remaining),
                    'partition_d3f1_fels40gap': d3_result,
                    'partition_d5f1_fels40gap': d5_result,
                }
                results.append(entry)
                results_by_fam[fam] = len(results) - 1

            acc_d3 = d3_result.get('nw_accuracy', -1)
            acc_d5 = d5_result.get('nw_accuracy', -1)
            mean_d3 = np.mean(nw_accs_d3) if nw_accs_d3 else 0
            mean_d5 = np.mean(nw_accs_d5) if nw_accs_d5 else 0
            log(f'[{fi+1}/{len(families)}] {fam}: '
                f'd3={acc_d3:.3f} d5={acc_d5:.3f} '
                f'(mean d3={mean_d3:.3f} d5={mean_d5:.3f})')

            n_done += 1

        except Exception as e:
            log(f'[{fi+1}/{len(families)}] {fam}: ERROR: {e}')
            traceback.print_exc()
            n_errors += 1
            continue

        # Save periodically
        if n_done % SAVE_EVERY == 0:
            _save(results, nw_accs_d3, nw_accs_d5, results_path, dataset)

    # Final save
    _save(results, nw_accs_d3, nw_accs_d5, results_path, dataset)

    log(f'Done: {n_done} families processed, {n_errors} errors')
    if nw_accs_d3:
        log(f'Mean NW accuracy d3f1: {np.mean(nw_accs_d3):.4f} (n={len(nw_accs_d3)})')
    if nw_accs_d5:
        log(f'Mean NW accuracy d5f1: {np.mean(nw_accs_d5):.4f} (n={len(nw_accs_d5)})')


if __name__ == '__main__':
    main()
