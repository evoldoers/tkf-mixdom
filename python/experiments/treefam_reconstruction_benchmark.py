#!/usr/bin/env python3
"""TreeFam held-out leaf prediction benchmark.

Predicts held-out leaf sequences in TreeFam families using:
  1. Felsenstein LG08 — marginal ancestral reconstruction
  2. Partition-recon d3f1 — MixDom PhyloHMM with 3-domain model
  3. Partition-recon d5f1 — MixDom PhyloHMM with 5-domain model
  4. Beam d3f1 — composite beam reconstruction with 3-domain model
  5. Beam d5f1 — composite beam reconstruction with 5-domain model

For each family in the spec file: load MSA + tree from TreeFam,
remove the held-out leaf, predict held-out sequence, score.

Usage:
    cd python && JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES="" uv run python \
        -u experiments/treefam_reconstruction_benchmark.py
"""

import os
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

import sys
import json
import time
import copy
import traceback
import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.dp.hmm import safe_log
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
from tkfmixdom.jax.tree.tree_varanc import infer_internal_presence, name_internal_nodes
from tkfmixdom.jax.util.io import AA_TO_INT, TreeNode, parse_newick
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm
from tkfmixdom.jax.dp.hmm import (
    forward_backward_2d, pair_hmm_emissions, M as M_ST)
from experiments.ancrec_benchmark import needleman_wunsch_identity as _nw_identity


def _nw_metrics(pred_seq, true_seq):
    """Compute NW-based accuracy/precision/recall for comparability with CARABS."""
    nw_id, nw_aligned, nw_matches = _nw_identity(pred_seq, true_seq)
    pred_len = len(pred_seq)
    true_len = len(true_seq)
    prec = nw_matches / max(pred_len, 1)
    rec = nw_matches / max(true_len, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    return {
        'nw_accuracy': float(nw_id),
        'nw_precision': float(prec),
        'nw_recall': float(rec),
        'nw_f1': float(f1),
        'nw_matches': int(nw_matches),
        'nw_aligned': int(nw_aligned),
    }

# --- Constants ---
TREEFAM_DIR = os.path.expanduser("~/bio-datasets/data/treefam/treefam_family_data")
SPEC_PATH = os.path.join(os.path.dirname(__file__), "treefam_reconstruction_spec.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "treefam_reconstruction_benchmark.json")

# Methods controlled by env var
BENCH_METHODS = set(os.environ.get('BENCH_METHODS', 'felsenstein,partition_d3f1,partition_d5f1').split(','))

# Beam search constants
BEAM_WIDTH = 30
MAX_COL = 800

# TKF92 parameters for scoring (same as unified benchmark)
TKF92_INS = 0.046
TKF92_DEL = 0.054
TKF92_EXT = 0.68

t0 = time.time()
def log(msg): print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)


# --- TreeFam data loading ---

def parse_treefam_fasta(fasta_path):
    """Parse aligned FASTA from TreeFam.

    Returns dict {name: aligned_sequence_string}.
    Handles trailing '//' lines.
    """
    seqs = {}
    name = None
    seq_parts = []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('//'):
                continue
            if line.startswith('>'):
                if name is not None:
                    seqs[name] = ''.join(seq_parts)
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        seqs[name] = ''.join(seq_parts)
    return seqs


def parse_treefam_tree(emf_path):
    """Parse newick tree from TreeFam EMF file.

    Lines starting with 'SEQ' are metadata; the newick tree
    is on the line starting with '('.
    """
    with open(emf_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('('):
                return parse_newick(line)
    raise ValueError(f"No newick tree found in {emf_path}")


def build_msa_int(seqs, n_cols=None):
    """Convert aligned string sequences to integer-encoded MSA.

    Returns dict {name: (C,) int32 array} with -1 for gaps.
    Non-standard amino acids (X, B, Z, etc) are mapped to -1.
    """
    msa = {}
    for name, seq_str in seqs.items():
        C = len(seq_str) if n_cols is None else n_cols
        arr = np.full(C, -1, dtype=np.int32)
        for j, ch in enumerate(seq_str[:C]):
            ch_upper = ch.upper()
            if ch_upper in AA_TO_INT:
                idx = AA_TO_INT[ch_upper]
                if idx >= 20:
                    # Non-standard (wildcard) -> gap
                    arr[j] = -1
                else:
                    arr[j] = idx
            # else: stays -1 (gap)
        msa[name] = arr
    return msa


# --- Tree operations ---

def prune_leaf(tree, leaf_name):
    """Remove a leaf from the tree and return a copy."""
    new_tree = copy.deepcopy(tree)
    for node in new_tree.preorder():
        for i, child in enumerate(node.children):
            if child.name == leaf_name and child.is_leaf:
                if len(node.children) == 2:
                    sibling = node.children[1 - i]
                    sibling.branch_length += node.branch_length
                    if node.parent is not None:
                        idx = node.parent.children.index(node)
                        node.parent.children[idx] = sibling
                        sibling.parent = node.parent
                    else:
                        sibling.parent = None
                        return sibling
                else:
                    node.children.pop(i)
                return new_tree
    return new_tree


def prune_tree_to_msa(tree, msa_names):
    """Prune tree to only keep leaves present in msa_names.

    Iteratively removes leaves not in msa_names until all leaves are in set.
    Returns pruned tree (deep copy).
    """
    new_tree = copy.deepcopy(tree)
    changed = True
    while changed:
        changed = False
        leaves = list(new_tree.leaves())
        for leaf in leaves:
            if leaf.name not in msa_names:
                new_tree = prune_leaf(new_tree, leaf.name)
                changed = True
                break  # restart after structural change
    return new_tree


def tree_pairwise_distances(tree):
    """Compute pairwise distances between all leaves via the tree.

    Returns:
        leaf_names: list of leaf names
        dist_mat: (n, n) numpy array of pairwise tree distances
    """
    leaves = list(tree.leaves())
    leaf_names = [l.name for l in leaves]
    n = len(leaf_names)

    def path_to_root(node):
        path = []
        cur = node
        dist = 0.0
        while cur is not None:
            path.append((id(cur), dist))
            dist += cur.branch_length
            cur = cur.parent
        return path

    paths = {}
    for leaf in leaves:
        paths[leaf.name] = dict(path_to_root(leaf))

    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pi = paths[leaf_names[i]]
            pj = paths[leaf_names[j]]
            common = set(pi.keys()) & set(pj.keys())
            min_dist = min(pi[nid] + pj[nid] for nid in common)
            dist_mat[i, j] = min_dist
            dist_mat[j, i] = min_dist
    return leaf_names, dist_mat


def build_mixdom_beam_data(params, n_dom, n_frag, distances_k):
    """Build per-descendant data for MixDom beam reconstruction."""
    from tkfmixdom.jax.models.mixdom import build_nested_trans, state_types as mixdom_state_types
    from tkfmixdom.jax.distill.maraschino import build_rate_matrix
    from tkfmixdom.jax.core.ctmc import transition_matrix as tm_with_pi

    S_exch = np.asarray(params['S_exch'])
    pis = np.asarray(params['pi'])

    log_chi_list, st_list, sub_mats_list, pis_list_k = [], [], [], []
    for t in distances_k:
        chi, _ = build_nested_trans(
            jnp.float32(params['lam0']), jnp.float32(params['mu0']),
            jnp.float32(t),
            jnp.array(params['lam']), jnp.array(params['mu']),
            jnp.array(params['v']),
            jnp.array(params['frag_weights']),
            jnp.array(params['r_frags']))
        st = np.asarray(mixdom_state_types(n_dom, n_frag))
        sub_mats = np.stack([np.asarray(tm_with_pi(
            jnp.array(build_rate_matrix(jnp.array(S_exch[d]),
                                         jnp.array(pis[d]))),
            jnp.array(pis[d]), t)) for d in range(n_dom)])
        log_chi_list.append(np.asarray(safe_log(chi)))
        st_list.append(st)
        sub_mats_list.append(sub_mats)
        pis_list_k.append(pis)
    return log_chi_list, st_list, sub_mats_list, pis_list_k


# --- Scoring ---

def score_prediction(pred_seq, true_seq, log_chi, state_types, sub_matrix, pi):
    """Score a predicted sequence against the true held-out sequence.

    Uses forward-backward to align pred to true, then computes:
    - matches, inserts, deletes
    - precision, recall, accuracy
    """
    pred = jnp.array(pred_seq, dtype=jnp.int32)
    true = jnp.array(true_seq, dtype=jnp.int32)
    Lp, Lt = len(pred), len(true)

    if Lp == 0 or Lt == 0:
        return {
            'matches': 0, 'inserts': int(Lp), 'deletes': int(Lt),
            'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0,
            'log_prob': -1e30, 'pred_len': int(Lp), 'true_len': int(Lt),
        }

    log_prob, posteriors, _ = forward_backward_2d(
        jnp.array(log_chi), state_types, pred, true,
        jnp.array(sub_matrix), jnp.array(pi))

    st_np = np.asarray(state_types)
    post_np = np.asarray(posteriors)

    is_M = (st_np == M_ST)
    match_post = post_np[1:Lp+1, 1:Lt+1, :][:, :, is_M].sum(axis=-1)

    pred_matched = match_post.sum(axis=1)
    true_matched = match_post.sum(axis=0)

    E_matches = float(match_post.sum())
    E_inserts = float((1.0 - pred_matched).clip(0).sum())
    E_deletes = float((1.0 - true_matched).clip(0).sum())

    precision = E_matches / max(E_matches + E_inserts, 1e-10)
    recall = E_matches / max(E_matches + E_deletes, 1e-10)

    correct = 0.0
    for i in range(Lp):
        for j in range(Lt):
            if match_post[i, j] > 1e-6:
                if pred_seq[i] == true_seq[j]:
                    correct += match_post[i, j]
    accuracy = correct / max(E_matches, 1e-10)

    return {
        'matches': float(E_matches),
        'inserts': float(E_inserts),
        'deletes': float(E_deletes),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'log_prob': float(log_prob),
        'pred_len': int(Lp),
        'true_len': int(Lt),
    }


def score_felsenstein_columns(anc_posteriors, true_msa_col, C):
    """Score Felsenstein column predictions against true sequence."""
    post = np.asarray(anc_posteriors)
    A = post.shape[1]
    correct = 0
    log_prob = 0.0
    n_total = 0

    for j in range(C):
        true_char = int(true_msa_col[j])
        if true_char < 0:
            true_char = min(20, A - 1)

        pred = int(np.argmax(post[j]))
        if pred == true_char:
            correct += 1

        if true_char < A:
            p = float(post[j, true_char])
            log_prob += np.log(max(p, 1e-300))
        n_total += 1

    accuracy = correct / max(n_total, 1)
    return {
        'col_accuracy': float(accuracy),
        'col_log_prob': float(log_prob),
        'n_cols': n_total,
        'n_correct': correct,
    }


# --- Reconstruction methods ---

def run_felsenstein(tree, held_out, remaining, msa, C, Q_lg_np, pi_lg_np):
    """Run Felsenstein on pruned tree, return (seq, elapsed)."""
    tf = time.time()
    pruned_tree = prune_leaf(tree, held_out)
    name_internal_nodes(pruned_tree)
    pruned_msa = {}
    for l in remaining:
        seq = msa[l].copy()
        seq[seq >= 20] = -1
        pruned_msa[l] = seq
    ancestor, _ = marginal_ancestor_all_columns_jax(
        pruned_tree, pruned_msa, Q_lg_np, pi_lg_np)
    leaf_pres = {l: np.array(pruned_msa[l] >= 0, dtype=bool)
                 for l in pruned_msa}
    root_pres = infer_internal_presence(pruned_tree, leaf_pres)
    rp = root_pres.get(pruned_tree.name, np.ones(C, dtype=bool))
    fels_seq = np.array([int(ancestor[c]) for c in range(len(ancestor))
                         if c < len(rp) and rp[c] and ancestor[c] >= 0],
                        dtype=np.int32)
    elapsed = time.time() - tf
    return fels_seq, elapsed


def run_partition_recon(tree, held_out, remaining, msa, C, model, config):
    """Run partition-conditioned reconstruction, return (seq, elapsed)."""
    from experiments.partition_recon_adapter import run_partition_reconstruction_method
    return run_partition_reconstruction_method(
        tree, held_out, remaining, msa, C,
        model=model, config=config)


# --- Main ---

def main():
    log('Loading spec...')
    with open(SPEC_PATH) as f:
        spec = json.load(f)

    families = spec['families']
    treefam_dir = os.path.expanduser(spec.get('treefam_dir', TREEFAM_DIR))
    log(f'{len(families)} families in spec, methods: {sorted(BENCH_METHODS)}')

    # Load models
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg_np, pi_lg_np = np.asarray(Q_lg), np.asarray(pi_lg)

    # TKF92 pair HMM for scoring
    t_score_default = 0.5
    log_chi_score, st_score, sub_score, pi_score_out = make_tkf92_pair_hmm(
        TKF92_INS, TKF92_DEL, t_score_default, TKF92_EXT,
        jnp.array(Q_lg_np), jnp.array(pi_lg_np))
    log_chi_s = np.asarray(log_chi_score)
    st_s = np.asarray(st_score)
    sub_s = np.asarray(sub_score)
    pi_s = np.asarray(pi_lg_np)

    # Partition-recon models (lazy load)
    partition_model_d3 = None
    partition_model_d5 = None
    partition_config = None

    if 'partition_d3f1' in BENCH_METHODS or 'partition_d5f1' in BENCH_METHODS:
        from experiments.partition_recon_adapter import (
            mixdom_model_from_params, PartitionReconConfig,
        )
        from tkfmixdom.jax.distill.maraschino import load_params
        partition_config = PartitionReconConfig(use_jax=True)

        if 'partition_d3f1' in BENCH_METHODS:
            log('  Loading d3f1 params...')
            params_d3, n_dom_d3, n_cls_d3 = load_params('pfam/svi_bw_d3f1_full_best_val.npz')
            partition_model_d3 = mixdom_model_from_params(params_d3)
            log(f'  d3f1: n_dom={n_dom_d3}')

        if 'partition_d5f1' in BENCH_METHODS:
            log('  Loading d5f1 params...')
            params_d5, n_dom_d5, n_cls_d5 = load_params('pfam/svi_bw_d5f1_full_best_val.npz')
            partition_model_d5 = mixdom_model_from_params(params_d5)
            log(f'  d5f1: n_dom={n_dom_d5}')

    # Beam models (lazy load)
    beam_params_d3 = None
    beam_params_d5 = None
    beam_n_dom_d3 = beam_n_frag_d3 = 0
    beam_n_dom_d5 = beam_n_frag_d5 = 0
    beam_s_trans_d3 = beam_s_pi_d3 = beam_s_end_d3 = None
    beam_s_trans_d5 = beam_s_pi_d5 = beam_s_end_d5 = None

    if 'beam_d3f1' in BENCH_METHODS or 'beam_d5f1' in BENCH_METHODS:
        from tkfmixdom.jax.tree.composite_beam_jax import composite_beam_reconstruct_jax
        from tkfmixdom.jax.tree.composite_beam import compute_unique_weights
        from tkfmixdom.jax.distill.maraschino import (
            load_params as load_params_beam, precompute_mixdom,
            distill_mixdom, normalize_freqs_wfst)

        if 'beam_d3f1' in BENCH_METHODS:
            log('  Loading beam d3f1 params...')
            bp_d3, beam_n_dom_d3, n_cls_d3 = load_params_beam('pfam/svi_bw_d3f1_full_best_val.npz')
            beam_params_d3 = bp_d3
            beam_n_frag_d3 = 1
            precomp_d3 = precompute_mixdom(bp_d3, max(n_cls_d3, 1))
            dist_d3 = distill_mixdom(bp_d3, 0.1, max(n_cls_d3, 1), precomp_d3)
            wfst_d3 = normalize_freqs_wfst(dist_d3)
            beam_s_trans_d3 = np.log(np.maximum(np.array(wfst_d3['singlet_trans']), 1e-300))
            s_start_d3 = np.array(wfst_d3['singlet_start'])
            s_start_d3 = s_start_d3 / s_start_d3.sum()
            beam_s_pi_d3 = np.log(np.maximum(s_start_d3, 1e-300))
            beam_s_end_d3 = np.log(np.maximum(np.array(wfst_d3['singlet_end']), 1e-300))
            log(f'  beam d3f1: n_dom={beam_n_dom_d3}, n_frag={beam_n_frag_d3}')

        if 'beam_d5f1' in BENCH_METHODS:
            log('  Loading beam d5f1 params...')
            bp_d5, beam_n_dom_d5, n_cls_d5 = load_params_beam('pfam/svi_bw_d5f1_full_best_val.npz')
            beam_params_d5 = bp_d5
            beam_n_frag_d5 = 1
            precomp_d5 = precompute_mixdom(bp_d5, max(n_cls_d5, 1))
            dist_d5 = distill_mixdom(bp_d5, 0.1, max(n_cls_d5, 1), precomp_d5)
            wfst_d5 = normalize_freqs_wfst(dist_d5)
            beam_s_trans_d5 = np.log(np.maximum(np.array(wfst_d5['singlet_trans']), 1e-300))
            s_start_d5 = np.array(wfst_d5['singlet_start'])
            s_start_d5 = s_start_d5 / s_start_d5.sum()
            beam_s_pi_d5 = np.log(np.maximum(s_start_d5, 1e-300))
            beam_s_end_d5 = np.log(np.maximum(np.array(wfst_d5['singlet_end']), 1e-300))
            log(f'  beam d5f1: n_dom={beam_n_dom_d5}, n_frag={beam_n_frag_d5}')

    # Resume support
    # Map method env names to result keys
    _method_to_key = {
        'felsenstein': 'fels', 'partition_d3f1': 'partition_d3f1',
        'partition_d5f1': 'partition_d5f1',
        'beam_d3f1': 'beam_d3f1', 'beam_d5f1': 'beam_d5f1',
    }
    results = []
    results_by_fam = {}  # family -> index in results
    done_fams = set()
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as rf:
                rd = json.load(rf)
            if isinstance(rd, dict) and 'results' in rd:
                results = list(rd['results'])
                for ri, r in enumerate(results):
                    if 'family' in r:
                        results_by_fam[r['family']] = ri
                # A family is fully done only if all requested methods have results
                for r in results:
                    if 'family' not in r:
                        continue
                    all_done = True
                    for m in BENCH_METHODS:
                        rk = _method_to_key.get(m, m)
                        if rk not in r:
                            all_done = False
                            break
                    if all_done:
                        done_fams.add(r['family'])
                log(f'Resume: {len(results)} prior results, {len(done_fams)} fully done')
        except Exception as e:
            log(f'Resume: failed to load: {e}')

    n_done = len(results)
    n_total = len(families)

    for idx, fam_spec in enumerate(families):
        fam = fam_spec['family']
        held_out = fam_spec['held_out']

        # Clear JIT cache to prevent GPU memory accumulation
        jax.clear_caches()  # OOM prevention (forces recompile; prefer geometric padding)

        if fam in done_fams:
            continue

        # Load TreeFam data
        fasta_path = os.path.join(treefam_dir, f'{fam}.aa.fasta')
        tree_path = os.path.join(treefam_dir, f'{fam}.nh.emf')

        if not os.path.exists(fasta_path):
            log(f'[{idx+1}/{n_total}] {fam}: FASTA not found, skipping')
            continue
        if not os.path.exists(tree_path):
            log(f'[{idx+1}/{n_total}] {fam}: tree not found, skipping')
            continue

        try:
            # Parse aligned FASTA
            raw_seqs = parse_treefam_fasta(fasta_path)
            if held_out not in raw_seqs:
                log(f'[{idx+1}/{n_total}] {fam}: held_out {held_out} not in FASTA, skipping')
                continue

            C = len(next(iter(raw_seqs.values())))
            msa = build_msa_int(raw_seqs, n_cols=C)

            # Parse tree
            tree = parse_treefam_tree(tree_path)
            name_internal_nodes(tree)

            # Get tree leaves and MSA names
            tree_leaf_names = {l.name for l in tree.leaves()}
            msa_names = set(raw_seqs.keys())

            # Find common leaves (in both tree and MSA)
            common = tree_leaf_names & msa_names
            if held_out not in common:
                log(f'[{idx+1}/{n_total}] {fam}: held_out {held_out} not in tree, skipping')
                continue

            remaining = sorted(common - {held_out})
            if len(remaining) < 2:
                log(f'[{idx+1}/{n_total}] {fam}: too few remaining leaves ({len(remaining)}), skipping')
                continue

            # Prune tree to only have leaves in MSA
            pruned_full_tree = prune_tree_to_msa(tree, msa_names)
            name_internal_nodes(pruned_full_tree)

            # True sequence from spec
            true_seq = np.array(fam_spec['true_seq'], dtype=np.int32)
            # Filter out gaps (-1) for ungapped true sequence
            true_seq_ungapped = true_seq[true_seq >= 0]

            # Reuse existing result if present (to add new methods)
            if fam in results_by_fam:
                result = results[results_by_fam[fam]]
            else:
                result = {
                    'family': fam,
                    'held_out': held_out,
                    'true_len': int(len(true_seq_ungapped)),
                    'n_cols': C,
                    'K': len(remaining),
                    'mean_dist': fam_spec.get('mean_dist', 0.0),
                }

            # Build scoring pair HMM at representative distance
            t_score = fam_spec.get('mean_dist', 0.5)
            log_chi_score_fam, st_score_fam, sub_score_fam, _ = make_tkf92_pair_hmm(
                TKF92_INS, TKF92_DEL, t_score, TKF92_EXT,
                jnp.array(Q_lg_np), jnp.array(pi_lg_np))
            log_chi_sf = np.asarray(log_chi_score_fam)
            st_sf = np.asarray(st_score_fam)
            sub_sf = np.asarray(sub_score_fam)

            # === 1. Felsenstein ===
            if 'felsenstein' in BENCH_METHODS:
                try:
                    fels_seq, fels_time = run_felsenstein(
                        pruned_full_tree, held_out, remaining, msa, C,
                        Q_lg_np, pi_lg_np)
                    fels_score = score_prediction(
                        fels_seq, true_seq_ungapped,
                        log_chi_sf, st_sf, sub_sf, pi_s)
                    # Column-level scoring
                    pruned_tree_f = prune_leaf(pruned_full_tree, held_out)
                    name_internal_nodes(pruned_tree_f)
                    pruned_msa_f = {n: msa[n] for n in remaining}
                    _, fels_post = marginal_ancestor_all_columns_jax(
                        pruned_tree_f, pruned_msa_f, Q_lg_np, pi_lg_np)
                    true_msa_col = msa[held_out]
                    fels_col = score_felsenstein_columns(fels_post, true_msa_col, C)
                    nw = _nw_metrics(fels_seq, true_seq_ungapped)
                    result['fels'] = {
                        **fels_score, **nw, 'time': float(fels_time),
                        'pred_seq': [int(x) for x in fels_seq],
                        **{f'fels_{k}': v for k, v in fels_col.items()},
                    }
                    log(f'[{idx+1}/{n_total}] {fam}: fels acc={fels_score["accuracy"]*100:.1f}% '
                        f't={fels_time:.1f}s')
                except Exception as e:
                    log(f'[{idx+1}/{n_total}] {fam}: fels error: {e}')
                    traceback.print_exc()
                    result['fels'] = {'accuracy': -1.0, 'time': 0.0}

            # === 2. Partition-recon d3f1 ===
            if 'partition_d3f1' in BENCH_METHODS and partition_model_d3 is not None:
                try:
                    part_d3_pred, part_d3_time = run_partition_recon(
                        pruned_full_tree, held_out, remaining, msa, C,
                        model=partition_model_d3, config=partition_config)
                    part_d3_score = score_prediction(
                        part_d3_pred, true_seq_ungapped,
                        log_chi_sf, st_sf, sub_sf, pi_s)
                    nw = _nw_metrics(part_d3_pred, true_seq_ungapped)
                    result['partition_d3f1'] = {
                        **part_d3_score, **nw, 'time': float(part_d3_time),
                        'pred_seq': [int(x) for x in part_d3_pred],
                    }
                    log(f'[{idx+1}/{n_total}] {fam}: partition_d3f1 '
                        f'acc={part_d3_score["accuracy"]*100:.1f}% t={part_d3_time:.1f}s')
                except Exception as e:
                    log(f'[{idx+1}/{n_total}] {fam}: partition_d3f1 error: {e}')
                    traceback.print_exc()
                    result['partition_d3f1'] = {'accuracy': -1.0, 'time': 0.0}

            # === 3. Partition-recon d5f1 ===
            if 'partition_d5f1' in BENCH_METHODS and partition_model_d5 is not None:
                try:
                    part_d5_pred, part_d5_time = run_partition_recon(
                        pruned_full_tree, held_out, remaining, msa, C,
                        model=partition_model_d5, config=partition_config)
                    part_d5_score = score_prediction(
                        part_d5_pred, true_seq_ungapped,
                        log_chi_sf, st_sf, sub_sf, pi_s)
                    nw = _nw_metrics(part_d5_pred, true_seq_ungapped)
                    result['partition_d5f1'] = {
                        **part_d5_score, **nw, 'time': float(part_d5_time),
                        'pred_seq': [int(x) for x in part_d5_pred],
                    }
                    log(f'[{idx+1}/{n_total}] {fam}: partition_d5f1 '
                        f'acc={part_d5_score["accuracy"]*100:.1f}% t={part_d5_time:.1f}s')
                except Exception as e:
                    log(f'[{idx+1}/{n_total}] {fam}: partition_d5f1 error: {e}')
                    traceback.print_exc()
                    result['partition_d5f1'] = {'accuracy': -1.0, 'time': 0.0}

            # === 4. Beam d3f1 ===
            if 'beam_d3f1' in BENCH_METHODS and beam_params_d3 is not None:
                if C > MAX_COL:
                    result['beam_d3f1'] = {'accuracy': None, 'time': 0.0,
                                           'skipped': f'n_cols={C} > MAX_COL={MAX_COL}'}
                    log(f'[{idx+1}/{n_total}] {fam}: beam_d3f1 skipped (n_cols={C})')
                else:
                    try:
                        # Compute distances and weights
                        rerooted_beam, _ = prune_leaf_keep_parent(pruned_full_tree, held_out)
                        def _dist_to_root_beam(node):
                            d = 0.0
                            while node.parent is not None:
                                d += node.branch_length if node.branch_length else 0.0
                                node = node.parent
                            return d
                        leaf_dist_beam = {}
                        for bnode in rerooted_beam.preorder():
                            if bnode.is_leaf and bnode.name:
                                leaf_dist_beam[bnode.name] = _dist_to_root_beam(bnode)
                        distances_k = [max(leaf_dist_beam.get(l, 1.0), 0.01) for l in remaining]
                        desc_seqs_k = [np.array([c for c in msa[l] if c >= 0], dtype=np.int32)
                                       for l in remaining]
                        weights = compute_unique_weights(pruned_full_tree, pruned_full_tree.name, remaining)

                        lc, st, sm, pl = build_mixdom_beam_data(
                            beam_params_d3, beam_n_dom_d3, beam_n_frag_d3, distances_k)
                        tb = time.time()
                        recon_d3, score_d3 = composite_beam_reconstruct_jax(
                            desc_seqs_k, distances_k, lc, st, sm, pl,
                            beam_n_dom_d3, beam_n_frag_d3,
                            beam_s_trans_d3, beam_s_pi_d3, beam_s_end_d3,
                            beam_width=BEAM_WIDTH,
                            max_len=int(len(true_seq_ungapped) * 1.5),
                            desc_weights=weights)
                        d3_time = time.time() - tb

                        nw = _nw_metrics(recon_d3, true_seq_ungapped)
                        beam_d3_fb = score_prediction(
                            recon_d3, true_seq_ungapped,
                            log_chi_sf, st_sf, sub_sf, pi_s)
                        result['beam_d3f1'] = {
                            **beam_d3_fb, **nw,
                            'nw_identity': float(nw['nw_accuracy']),
                            'time': float(d3_time),
                            'pred_len': int(len(recon_d3)),
                            'true_len': int(len(true_seq_ungapped)),
                            'beam_score': float(score_d3),
                            'pred_seq': [int(x) for x in recon_d3],
                        }
                        log(f'[{idx+1}/{n_total}] {fam}: beam_d3f1 '
                            f'acc={beam_d3_fb["accuracy"]*100:.1f}% '
                            f'nw={nw_id*100:.1f}% t={d3_time:.1f}s')
                    except Exception as e:
                        log(f'[{idx+1}/{n_total}] {fam}: beam_d3f1 error: {e}')
                        traceback.print_exc()
                        result['beam_d3f1'] = {'accuracy': -1.0, 'time': 0.0}

            # === 5. Beam d5f1 ===
            if 'beam_d5f1' in BENCH_METHODS and beam_params_d5 is not None:
                if C > MAX_COL:
                    result['beam_d5f1'] = {'accuracy': None, 'time': 0.0,
                                           'skipped': f'n_cols={C} > MAX_COL={MAX_COL}'}
                    log(f'[{idx+1}/{n_total}] {fam}: beam_d5f1 skipped (n_cols={C})')
                else:
                    try:
                        # Compute distances and weights (reuse if already computed)
                        if 'beam_d3f1' not in BENCH_METHODS or C > MAX_COL:
                            rerooted_beam, _ = prune_leaf_keep_parent(pruned_full_tree, held_out)
                            def _dist_to_root_beam(node):
                                d = 0.0
                                while node.parent is not None:
                                    d += node.branch_length if node.branch_length else 0.0
                                    node = node.parent
                                return d
                            leaf_dist_beam = {}
                            for bnode in rerooted_beam.preorder():
                                if bnode.is_leaf and bnode.name:
                                    leaf_dist_beam[bnode.name] = _dist_to_root_beam(bnode)
                            distances_k = [max(leaf_dist_beam.get(l, 1.0), 0.01) for l in remaining]
                            desc_seqs_k = [np.array([c for c in msa[l] if c >= 0], dtype=np.int32)
                                           for l in remaining]
                            weights = compute_unique_weights(pruned_full_tree, pruned_full_tree.name, remaining)

                        lc, st, sm, pl = build_mixdom_beam_data(
                            beam_params_d5, beam_n_dom_d5, beam_n_frag_d5, distances_k)
                        tb = time.time()
                        recon_d5, score_d5 = composite_beam_reconstruct_jax(
                            desc_seqs_k, distances_k, lc, st, sm, pl,
                            beam_n_dom_d5, beam_n_frag_d5,
                            beam_s_trans_d5, beam_s_pi_d5, beam_s_end_d5,
                            beam_width=BEAM_WIDTH,
                            max_len=int(len(true_seq_ungapped) * 1.5),
                            desc_weights=weights)
                        d5_time = time.time() - tb

                        nw = _nw_metrics(recon_d5, true_seq_ungapped)
                        beam_d5_fb = score_prediction(
                            recon_d5, true_seq_ungapped,
                            log_chi_sf, st_sf, sub_sf, pi_s)
                        result['beam_d5f1'] = {
                            **beam_d5_fb, **nw,
                            'nw_identity': float(nw['nw_accuracy']),
                            'time': float(d5_time),
                            'pred_len': int(len(recon_d5)),
                            'true_len': int(len(true_seq_ungapped)),
                            'beam_score': float(score_d5),
                            'pred_seq': [int(x) for x in recon_d5],
                        }
                        log(f'[{idx+1}/{n_total}] {fam}: beam_d5f1 '
                            f'acc={beam_d5_fb["accuracy"]*100:.1f}% '
                            f'nw={nw_id*100:.1f}% t={d5_time:.1f}s')
                    except Exception as e:
                        log(f'[{idx+1}/{n_total}] {fam}: beam_d5f1 error: {e}')
                        traceback.print_exc()
                        result['beam_d5f1'] = {'accuracy': -1.0, 'time': 0.0}

            if fam not in results_by_fam:
                results.append(result)
                results_by_fam[fam] = len(results) - 1
            n_done = len(results)

            # Save periodically
            if n_done % 5 == 0:
                _save_results(results, n_done)

        except Exception as e:
            log(f'[{idx+1}/{n_total}] {fam}: ERROR: {e}')
            traceback.print_exc()
            continue

    # Final save
    _save_results(results, n_done)

    # Summary
    log(f'\n{"="*60}')
    log(f'Processed {n_done} families')

    method_keys = ['fels', 'partition_d3f1', 'partition_d5f1', 'beam_d3f1', 'beam_d5f1']
    method_labels = ['Felsenstein', 'Partition d3f1', 'Partition d5f1', 'Beam d3f1', 'Beam d5f1']

    log(f'\n{"Method":<16} {"Accuracy":>8} {"Precision":>9} {"Recall":>8} {"N":>5}')
    log('-' * 50)
    for label, key in zip(method_labels, method_keys):
        accs = [r[key]['accuracy'] for r in results
                if isinstance(r.get(key), dict) and r[key].get('accuracy', -1) >= 0]
        precs = [r[key]['precision'] for r in results
                 if isinstance(r.get(key), dict) and r[key].get('precision', -1) >= 0]
        recs = [r[key]['recall'] for r in results
                if isinstance(r.get(key), dict) and r[key].get('recall', -1) >= 0]
        if accs:
            log(f'{label:<16} {np.mean(accs):>7.1%} {np.mean(precs):>8.1%} '
                f'{np.mean(recs):>7.1%} {len(accs):>5}')

    # Felsenstein column-level
    fels_col_accs = [r['fels'].get('fels_col_accuracy', -1) for r in results
                     if isinstance(r.get('fels'), dict) and 'fels_col_accuracy' in r['fels']]
    fels_col_accs = [v for v in fels_col_accs if v >= 0]
    if fels_col_accs:
        log(f'\nFelsenstein column-level: {np.mean(fels_col_accs):.1%}')


def _save_results(results, n_done):
    """Save results to JSON."""
    with open(RESULTS_PATH, 'w') as f:
        json.dump({'results': results, 'n_families': n_done}, f, indent=2)
    log(f'  Saved {n_done} results to {RESULTS_PATH}')


if __name__ == '__main__':
    main()
