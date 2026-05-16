#!/usr/bin/env python3
"""Unified held-out leaf prediction benchmark on FastTree ML trees.

Compares two MixDom models + Felsenstein + TKF92 baselines on the SAME
FastTree tree per family. The two MixDom slots (slot A and slot B) are
configured via ``MIXDOM_<slot>=path:label`` env vars (matching the
``unified_benchmark_long.py`` convention):

  MIXDOM_0=path/to/model.npz:label_0   # slot 0
  MIXDOM_1=path/to/other.npz:label_1   # slot 1
  MIXDOM_2=path/to/third.npz:label_2   # slot 2 (any number of slots OK)

Result keys and log lines use the ``label_<i>`` strings directly — there
is no privileged "d3f1" or "d5f1" anywhere; whatever label you supply
is what appears in the output JSON. To run a single MixDom model, omit
the second env var (or set the path to a non-existent file).

Backward compatibility: ``MIXDOM_D3=path`` and ``MIXDOM_D5=path`` (no
``:label`` suffix) still load into the two slots with default labels
"d3f1" and "d5f1" — these defaults are NOT a privilege; they are just
the historical fallback labels and are easy to override.

Felsenstein and TKF92 method labels are fixed (no model attached).

For each Pfam val family: remove one leaf, predict its residues.
Per-column accuracy including gaps (Fitch parsimony for gap prediction).

Usage:
    cd python && JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 \\
        MIXDOM_0=pfam/annabel_gtr3_imported.npz:annabel_gtr3 \\
        uv run python experiments/unified_reconstruction_benchmark.py
"""

import os
import sys
import json
import time
import copy
import traceback
import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
# Don't force CPU — use whatever JAX finds (GPU if available)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from tkfmixdom.jax.tree.composite_beam_jax import composite_beam_reconstruct_jax
from tkfmixdom.jax.tree.composite_beam import compute_unique_weights
from tkfmixdom.jax.models.mixdom import build_nested_trans, state_types as mixdom_state_types
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm
from tkfmixdom.jax.distill.maraschino import (
    load_params, build_rate_matrix, precompute_mixdom, distill_mixdom, normalize_freqs_wfst)
from tkfmixdom.jax.core.ctmc import transition_matrix, build_Q_from_S_pi
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.params import S, M, I, D, E
from tkfmixdom.jax.dp.hmm import safe_log
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
from tkfmixdom.jax.tree.tree_varanc import infer_internal_presence, name_internal_nodes
from tkfmixdom.jax.util.io import AA_TO_INT, TreeNode, parse_newick
from tkfmixdom.jax.dp.hmm import (
    forward_backward_2d, forward_2d_banded, pair_hmm_emissions,
    pair_hmm_emissions_per_domain, M as M_ST)
from experiments.ancrec_benchmark import (
    parse_sto, needleman_wunsch_identity, PFAM_DIR)
from experiments.reconstruct_util import reroot_at_node


def _nw_metrics(pred_seq, true_seq):
    """Compute NW-based accuracy/precision/recall for comparability with CARABS.

    Returns dict with nw_accuracy, nw_precision, nw_recall, nw_matches, nw_aligned.
    """
    nw_id, nw_aligned, nw_matches = needleman_wunsch_identity(pred_seq, true_seq)
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
TREE_DIR = os.path.expanduser("~/bio-datasets/data/pfam-seed/trees")
MAX_FAMILIES = 200
BEAM_WIDTH = 30
MAX_COL = 100
MIN_SEQS = 5
MAX_SEQS = 50  # limit for beam search tractability
TIMEOUT_PER_FAMILY = 120  # seconds
# Method selection via env var: RECON_METHODS=fels,partition_<label>,...
# Default: all methods. Methods are dynamic based on MIXDOM_<N> env vars:
#   fels, partition_<label_i>, <label_i>, tkf92, <label_i>_fixed, tkf92_fixed
_ENABLED_METHODS = set(os.environ.get('RECON_METHODS', '').split(',')) if os.environ.get('RECON_METHODS') else None


def _parse_mixdom_slots():
    """Parse MIXDOM_<N>=path[:label] env vars into a list of (path, label).

    Convention matches unified_benchmark_long.py. Backward-compat:
    MIXDOM_D3 / MIXDOM_D5 also accepted, mapped to slots 0 and 1
    with fallback labels 'd3f1' / 'd5f1'. Returns ALL configured
    MixDom slots — N-slot support, no truncation.
    """
    slots = []  # list of (path, label) in registration order
    # New-style: MIXDOM_<digit>=path[:label]
    for k, v in sorted(os.environ.items()):
        if k.startswith('MIXDOM_') and k[7:].isdigit():
            parts = v.split(':', 1)
            path = parts[0]
            label = parts[1] if len(parts) == 2 else os.path.splitext(os.path.basename(path))[0]
            slots.append((path, label))
    # Backward-compat: MIXDOM_D3 / MIXDOM_D5 (no ':label' suffix supported here)
    if not slots:
        d3 = os.environ.get('MIXDOM_D3', 'pfam/svi_bw_d3f1_full_best_val.npz')
        d5 = os.environ.get('MIXDOM_D5', 'pfam/svi_bw_d5f1_full_best_val.npz')
        slots = [(d3, 'd3f1'), (d5, 'd5f1')]
    return slots

def _method_enabled(method_name):
    """Check if a method should run (True if no filter or method in filter)."""
    return _ENABLED_METHODS is None or method_name in _ENABLED_METHODS

class _SkipMethod(Exception):
    """Raised to skip a method that's disabled or already done."""
    pass

# TKF92 parameters (CherryML-fitted)
TKF92_INS = 0.046
TKF92_DEL = 0.054
TKF92_EXT = 0.68

def score_prediction(pred_seq, true_seq, log_chi, state_types, sub_matrix, pi,
                      n_dom=1, n_frag=1, sub_matrices=None, pis=None):
    """Score a predicted sequence against the true held-out sequence.

    Uses forward-backward to align pred to true, then computes:
    - matches, inserts (in pred not true), deletes (in true not pred)
    - precision = matches / (matches + inserts)
    - recall = matches / (matches + deletes)
    - per-column accuracy weighted by match posterior
    - log P(true | model) = forward probability with true as descendant

    For Felsenstein: pred_seq is the per-column argmax, true_seq is the
    true held-out residues, and we run FB between them to get alignment
    metrics. The log_chi/state_types should be a TKF92 pair HMM.

    Args:
        pred_seq: (Lp,) int array — predicted residues (no gaps)
        true_seq: (Lt,) int array — true residues (no gaps)
        log_chi: (ns, ns) log transition matrix for pair HMM
        state_types: (ns,) state type codes
        sub_matrix: (A, A) substitution matrix (for single-domain models)
        pi: (A,) equilibrium (for single-domain models)
        n_dom, n_frag: for per-domain models
        sub_matrices: (n_dom, A, A) for per-domain models (overrides sub_matrix)
        pis: (n_dom, A) for per-domain models (overrides pi)

    Returns:
        dict with: matches, inserts, deletes, precision, recall,
                   accuracy, log_prob, pred_len, true_len
    """
    # Filter out non-standard residues (X=20, etc.) that would be
    # out of bounds for the 20×20 substitution matrix
    pred = jnp.array([x for x in pred_seq if 0 <= x < 20], dtype=jnp.int32)
    true = jnp.array([x for x in true_seq if 0 <= x < 20], dtype=jnp.int32)
    Lp, Lt = len(pred), len(true)

    if Lp == 0 or Lt == 0:
        return {
            'matches': 0, 'inserts': int(Lp), 'deletes': int(Lt),
            'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0,
            'log_prob': -1e30, 'pred_len': int(Lp), 'true_len': int(Lt),
        }

    # Build emission table and run FB
    # pred = "ancestor" (x), true = "descendant" (y) in pair HMM convention
    if sub_matrices is not None and pis is not None:
        emit = pair_hmm_emissions_per_domain(
            state_types, pred, true,
            jnp.array(sub_matrices), jnp.array(pis), n_dom, n_frag)
        log_prob, posteriors, _ = forward_backward_2d(
            jnp.array(log_chi), state_types, pred, true,
            None, None, log_emit_table=emit)
    else:
        log_prob, posteriors, _ = forward_backward_2d(
            jnp.array(log_chi), state_types, pred, true,
            jnp.array(sub_matrix), jnp.array(pi))

    st_np = np.asarray(state_types)
    post_np = np.asarray(posteriors)

    # Match posteriors: P(pred_i aligned to true_j)
    is_M = (st_np == M_ST)
    match_post = post_np[1:Lp+1, 1:Lt+1, :][:, :, is_M].sum(axis=-1)  # (Lp, Lt)

    # Insert posteriors: P(pred_i is inserted, not aligned to any true_j)
    is_I = (st_np == 2)  # I-type: consumes descendant (true) only
    is_D = (st_np == 3)  # D-type: consumes ancestor (pred) only

    # For each pred position i: total match posterior (aligned to some true j)
    pred_matched = match_post.sum(axis=1)  # (Lp,)
    # For each true position j: total match posterior (aligned to some pred i)
    true_matched = match_post.sum(axis=0)  # (Lt,)

    # Expected matches = sum of match posteriors
    E_matches = float(match_post.sum())
    # Expected inserts (pred positions not matched to any true)
    E_inserts = float((1.0 - pred_matched).clip(0).sum())
    # Expected deletes (true positions not matched to any pred)
    E_deletes = float((1.0 - true_matched).clip(0).sum())

    # Precision and recall
    precision = E_matches / max(E_matches + E_inserts, 1e-10)
    recall = E_matches / max(E_matches + E_deletes, 1e-10)

    # Per-position accuracy: for each matched (i,j) pair, is pred[i] == true[j]?
    # Weight by match posterior
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
    """Score Felsenstein column predictions against true sequence.

    This is the column-level scoring (no FB alignment needed).
    Serves as a control against the FB-based scoring.

    Args:
        anc_posteriors: (C, A) or (C, 21) posterior at each column
        true_msa_col: (C,) int array — true residue at each column (20=gap, -1=gap)
        C: number of columns

    Returns:
        dict with: accuracy (per-column), log_prob (product of posteriors),
                   n_residue_correct, n_gap_correct, n_total
    """
    post = np.asarray(anc_posteriors)
    A = post.shape[1]
    correct = 0
    log_prob = 0.0
    n_total = 0

    for j in range(C):
        true_char = int(true_msa_col[j])
        if true_char < 0:
            true_char = min(20, A - 1)  # treat -1 as gap

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


t0 = time.time()
def log(msg): print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)


def prune_leaf_keep_parent(tree, leaf_name):
    """Remove a leaf but keep its parent node intact (don't collapse branches).

    Returns (pruned_tree_rerooted_at_parent, parent_node) where the tree
    has been re-rooted at the target's parent and the target leaf removed.
    The parent node is now the root, so standard root reconstruction works.

    Returns (None, None) if the leaf is not found.
    """
    new_tree = copy.deepcopy(tree)
    name_internal_nodes(new_tree)
    # Find the target leaf and its parent
    for node in new_tree.preorder():
        if node.is_leaf and node.name == leaf_name:
            parent = node.parent
            if parent is None:
                raise ValueError(f"Target '{leaf_name}' is the root")
            # Re-root at parent (this deep-copies again internally)
            rerooted = reroot_at_node(new_tree, parent)
            # Remove target leaf (now a direct child of the new root)
            rerooted.children = [c for c in rerooted.children
                                 if not (c.is_leaf and c.name == leaf_name)]
            return rerooted, rerooted
    return None, None


def tree_pairwise_distances(tree):
    """Compute pairwise distances between all leaves via the tree.

    Returns:
        leaf_names: list of leaf names
        dist_mat: (n, n) numpy array of pairwise tree distances
    """
    leaves = list(tree.leaves())
    leaf_names = [l.name for l in leaves]
    n = len(leaf_names)

    # Build path-to-root for each leaf
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
            # Find LCA (lowest common ancestor)
            common = set(pi.keys()) & set(pj.keys())
            min_dist = min(pi[nid] + pj[nid] for nid in common)
            dist_mat[i, j] = min_dist
            dist_mat[j, i] = min_dist
    return leaf_names, dist_mat


def build_mixdom_beam_data(params, n_dom, n_frag, distances_k):
    """Build per-descendant data for MixDom beam reconstruction.

    Returns (log_chi_list, st_list, sub_mats_list, pis_list_k) for the
    class-marginal path. If ``params`` carries MixDom2 per-class data
    (``class_pi``, ``class_S_exch``, ``class_dist``), additionally
    returns ``class_sub_mats_list``, ``class_pis_list``, and
    ``class_dist`` (last element of the tuple). Callers that want the
    class-aware path should request the extended tuple via
    ``build_mixdom_beam_data_class()``.
    """
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
        # Use build_Q_from_S_pi (no rate-normalisation): trained per-domain
        # (S_exch, pi) carry the model's own evolutionary tempo; the unit-
        # normalised helper would strip that scale and silently degrade the
        # d3f1 (and similar trained MixDom) beam-reconstruction results,
        # exactly the rigor-audit failure mode flagged in 2026-05-01.
        sub_mats = np.stack([np.asarray(transition_matrix(
            build_Q_from_S_pi(jnp.array(S_exch[d]), jnp.array(pis[d])),
            t)) for d in range(n_dom)])
        log_chi_list.append(np.asarray(safe_log(chi)))
        st_list.append(st)
        sub_mats_list.append(sub_mats)
        pis_list_k.append(pis)
    return log_chi_list, st_list, sub_mats_list, pis_list_k


def build_mixdom_beam_data_class(params, n_dom, n_frag, distances_k):
    """Build per-descendant data including per-class P(t) for MixDom2.

    Always returns the same first four elements as
    ``build_mixdom_beam_data``, plus three extra arrays for the
    class-aware composite-beam path:

        class_sub_mats_list: list of K (C, A, A) per-class P(t).
        class_pis_list: list of K (C, A) per-class equilibrium.
        class_dist: (D, F, C) per-(domain, fragment) class distribution.

    Raises if ``params`` is missing any of ``class_pi`` / ``class_S_exch``
    / ``class_dist`` — callers should branch on the presence of these
    keys before invoking this function.
    """
    if not all(k in params for k in ('class_pi', 'class_S_exch', 'class_dist')):
        raise ValueError(
            "build_mixdom_beam_data_class: params is missing class_pi, "
            "class_S_exch, or class_dist; this checkpoint does not have "
            "MixDom2 per-class structure.")

    log_chi_list, st_list, sub_mats_list, pis_list_k = build_mixdom_beam_data(
        params, n_dom, n_frag, distances_k)

    class_pi = np.asarray(params['class_pi'])     # (C, A)
    class_S = np.asarray(params['class_S_exch'])  # (C, A, A)
    class_dist = np.asarray(params['class_dist'])  # (D, F, C)
    C = class_pi.shape[0]

    # Build per-class rate matrices (independent of t).
    # See note above on build_Q_from_S_pi vs unit-normalised:
    # trained per-class (S, pi) carry per-class rate scale; unit norm
    # would strip it.
    class_Q_arr = np.stack([
        np.asarray(build_Q_from_S_pi(jnp.array(class_S[c]),
                                     jnp.array(class_pi[c])))
        for c in range(C)])

    class_sub_mats_list = []
    class_pis_list = []
    for t in distances_k:
        class_sub_mats = np.stack([
            np.asarray(transition_matrix(jnp.array(class_Q_arr[c]), t))
            for c in range(C)])
        class_sub_mats_list.append(class_sub_mats)
        class_pis_list.append(class_pi)

    return (log_chi_list, st_list, sub_mats_list, pis_list_k,
            class_sub_mats_list, class_pis_list, class_dist)


def build_tkf92_beam_data(Q_lg, pi_lg, distances_k):
    """Build per-descendant data for TKF92 beam reconstruction.

    Wraps the 5-state TKF92 pair HMM into the composite_beam format:
    n_dom=1, n_frag=1, sub_matrix shape (1, A, A), pis shape (1, A).
    """
    log_chi_list, st_list, sub_mats_list, pis_list_k = [], [], [], []
    pi_lg_np = np.asarray(pi_lg)

    for t in distances_k:
        log_trans, state_types, sub_matrix, pi_out = make_tkf92_pair_hmm(
            TKF92_INS, TKF92_DEL, t, TKF92_EXT,
            jnp.array(Q_lg), jnp.array(pi_lg))
        # composite_beam expects sub_matrices shape (n_dom, A, A) and pis shape (n_dom, A)
        log_chi_list.append(np.asarray(log_trans))
        st_list.append(np.asarray(state_types))
        sub_mats_list.append(np.asarray(sub_matrix)[None, :, :])  # (1, A, A)
        pis_list_k.append(pi_lg_np[None, :])  # (1, A)
    return log_chi_list, st_list, sub_mats_list, pis_list_k


def run_beam(desc_seqs_k, distances_k, log_chi_list, st_list, sub_mats_list,
             pis_list_k, n_dom, n_frag, s_trans, s_pi, s_end,
             true_len, weights):
    """Run beam reconstruction and return (recon_seq, identity, time)."""
    tb = time.time()
    recon, score = composite_beam_reconstruct_jax(
        desc_seqs_k, distances_k,
        log_chi_list, st_list, sub_mats_list, pis_list_k,
        n_dom, n_frag, s_trans, s_pi, s_end,
        beam_width=BEAM_WIDTH,
        max_len=int(true_len * 1.5),
        desc_weights=weights)
    elapsed = time.time() - tb
    identity, _, _ = needleman_wunsch_identity(recon, np.array([], dtype=np.int32))
    return recon, elapsed


def run_felsenstein(tree, held_out, remaining, msa, C, Q_lg_np, pi_lg_np):
    """Run Felsenstein on tree re-rooted at target's parent, return (seq, time).

    Reconstructs at the target's parent node (not the root of the original tree).
    """
    tf = time.time()
    pruned_tree, _ = prune_leaf_keep_parent(tree, held_out)
    name_internal_nodes(pruned_tree)
    # Replace X residues (index 20) with gaps (-1) so Felsenstein
    # marginalizes over them (uniform likelihood, same as gaps).
    pruned_msa = {}
    for l in remaining:
        if l in msa:
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


def main():
    log('Loading models...')

    # --- Partition-recon adapter (PhyloHMM-based ASR) ---
    from experiments.partition_recon_adapter import (
        mixdom_model_from_params, run_partition_reconstruction_method,
        PartitionReconConfig,
    )
    partition_config = PartitionReconConfig(use_jax=True)

    # Resolve MixDom slot configuration via env vars (no privileged
    # default labels; whatever the user passes is what shows up in the
    # output JSON). See _parse_mixdom_slots() for the convention.
    raw_slots = _parse_mixdom_slots()
    log(f'  configured {len(raw_slots)} MixDom slot(s):')
    for i, (p, lbl) in enumerate(raw_slots):
        log(f'    slot {i} ({lbl}): {p}')

    def _infer_n_frag(p):
        if 'frag_weights' in p and np.asarray(p['frag_weights']).ndim == 2:
            return int(np.asarray(p['frag_weights']).shape[1])
        if 'r_frags' in p and np.asarray(p['r_frags']).ndim == 2:
            return int(np.asarray(p['r_frags']).shape[1])
        return 1

    def _slot_path_exists(path):
        if path is None:
            return False
        if os.path.isabs(path):
            return os.path.exists(path)
        return os.path.exists(os.path.join(os.path.dirname(__file__), '..', path))

    # --- Load all MixDom slots (skip any that can't be found) ---
    mixdom_slots = []  # list of dicts: each has path/label/params/n_dom/n_frag/n_cls/s_trans/s_pi/s_end/partition_model
    for path, label in raw_slots:
        if not _slot_path_exists(path):
            log(f'  slot ({label}): not loaded (path "{path}" unset or missing)')
            continue
        params_m, n_dom_m, n_cls_m = load_params(path)
        n_frag_m = _infer_n_frag(params_m)
        precomp_m = precompute_mixdom(params_m, max(n_cls_m, 1))
        dist_m = distill_mixdom(params_m, 0.1, max(n_cls_m, 1), precomp_m)
        wfst_m = normalize_freqs_wfst(dist_m)
        s_trans_m = np.log(np.maximum(np.array(wfst_m['singlet_trans']), 1e-300))
        s_start_m = np.array(wfst_m['singlet_start'])
        s_start_m = s_start_m / s_start_m.sum()
        s_pi_m = np.log(np.maximum(s_start_m, 1e-300))
        s_end_m = np.log(np.maximum(np.array(wfst_m['singlet_end']), 1e-300))
        mixdom_slots.append({
            'path': path, 'label': label,
            'params': params_m, 'n_dom': n_dom_m, 'n_frag': n_frag_m,
            'n_cls': n_cls_m,
            's_trans': s_trans_m, 's_pi': s_pi_m, 's_end': s_end_m,
            'partition_model': mixdom_model_from_params(params_m),
        })
        log(f'  slot ({label}): n_dom={n_dom_m}, n_frag={n_frag_m}, n_cls={n_cls_m}')

    # --- LG08 for Felsenstein and TKF92 ---
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg_np, pi_lg_np = np.asarray(Q_lg), np.asarray(pi_lg)

    # TKF92 singlet model: geometric(kappa) + LG08 pi
    # kappa = ins_rate / del_rate
    kappa = TKF92_INS / TKF92_DEL
    # Singlet: geometric length model with LG08 emissions
    # singlet_trans(a, b) = pi_lg[b] for all a (iid emissions, length via geometric)
    # singlet_start = pi_lg
    # singlet_end = 1 - kappa (probability of ending)
    s_trans_tkf = np.log(np.maximum(pi_lg_np[None, :] * np.ones((20, 1)), 1e-300))
    # Add geometric continuation: log(kappa * pi[b])
    s_trans_tkf = np.log(np.maximum(kappa * pi_lg_np[None, :] * np.ones((20, 1)), 1e-300))
    s_pi_tkf = np.log(np.maximum(pi_lg_np, 1e-300))
    s_end_tkf = np.log(np.maximum(np.full(20, 1.0 - kappa), 1e-300))
    log(f'  TKF92: ins={TKF92_INS}, del={TKF92_DEL}, ext={TKF92_EXT}, kappa={kappa:.4f}')

    # --- Load spec file (exact family list) ---
    # Override with env var RECON_SPEC=relative-to-experiments-dir or
    # absolute path. Default: the original (val-contaminated) spec.
    spec_name = os.environ.get('RECON_SPEC', 'unified_benchmark_spec.json')
    if os.path.isabs(spec_name):
        spec_path = spec_name
    else:
        spec_path = os.path.join(os.path.dirname(__file__), spec_name)
    spec = json.load(open(spec_path))
    log(f'  spec: {spec_path} (override with RECON_SPEC env var)')
    pfam_dir = os.path.expanduser(spec.get('pfam_dir', PFAM_DIR))
    tree_dir = os.path.expanduser(spec.get('tree_dir', TREE_DIR))
    families = spec['families']
    log(f'Loaded spec: {len(families)} families')

    results = []
    done_fams = set()
    # Family ID → index in `results`, so we can MERGE new method outputs
    # into an existing entry rather than appending a duplicate row.
    fam_to_idx: dict[str, int] = {}
    out_name = os.environ.get(
        'RECON_OUT', 'unified_reconstruction_benchmark.json')
    if os.path.isabs(out_name):
        resume_path = out_name
    else:
        resume_path = os.path.join(os.path.dirname(__file__), out_name)
    if os.path.exists(resume_path):
        try:
            with open(resume_path) as _rf:
                _rd = json.load(_rf)
            if isinstance(_rd, dict) and 'results' in _rd:
                results = list(_rd['results'])
                done_fams = {r['family'] for r in results if 'family' in r}
                fam_to_idx = {r['family']: i for i, r in enumerate(results)
                              if 'family' in r}
                log(f'Resume: loaded {len(results)} prior results, '
                    f'{len(done_fams)} families to skip')
        except Exception as _e:
            log(f'Resume: failed to load {resume_path}: {_e}')
    n_fams = len(results)

    for fi, fspec in enumerate(families):
        fam = fspec['family']
        held_out = fspec['held_out']
        remaining = fspec['remaining']
        true_seq = np.array(fspec['true_seq'], dtype=np.int32)
        C = fspec['n_cols']

        # Skip only if ALL enabled methods already have results
        if fam in done_fams:
            existing = next((r for r in results if r['family'] == fam), {})
            all_methods = ['fels']
            for s in mixdom_slots:
                all_methods.append(f'partition_{s["label"]}')
                all_methods.append(s['label'])
            all_methods.append('tkf92')
            enabled = [m for m in all_methods if _method_enabled(m)]
            if all(m in existing for m in enabled):
                continue

        # Check tree file
        tree_path = os.path.join(tree_dir, f'{fam}.nwk')
        if not os.path.exists(tree_path):
            log(f'  {fam}: tree file not found, skipping')
            continue

        # Check MSA file
        sto_path = os.path.join(pfam_dir, f'{fam}.sto')
        if not os.path.exists(sto_path):
            log(f'  {fam}: MSA file not found, skipping')
            continue

        # Parse MSA
        seqs = parse_sto(sto_path)

        # Parse FastTree tree
        try:
            with open(tree_path) as f:
                tree_text = f.read().strip()
            tree = parse_newick(tree_text)
            name_internal_nodes(tree)
        except Exception as e:
            log(f'  {fam}: tree parse error: {e}')
            continue

        # Build MSA as int arrays
        msa = {}
        for name in seqs:
            seq = np.full(C, -1, dtype=np.int32)
            for j, ch in enumerate(seqs[name]):
                if ch in AA_TO_INT:
                    seq[j] = AA_TO_INT[ch]
            msa[name] = seq

        desc_seqs_k = [np.array([c for c in msa[l] if c >= 0], dtype=np.int32)
                       for l in remaining]

        # Compute distances from each remaining leaf to the target's parent node.
        # After prune_leaf_keep_parent, the rerooted tree has the target's parent
        # as root. Distance from a leaf to root = sum of branch lengths on the path.
        rerooted_tree, _ = prune_leaf_keep_parent(tree, held_out)
        def _dist_to_root(node):
            d = 0.0
            while node.parent is not None:
                d += node.branch_length if node.branch_length else 0.0
                node = node.parent
            return d
        leaf_dist = {}
        for node in rerooted_tree.preorder():
            if node.is_leaf and node.name:
                leaf_dist[node.name] = _dist_to_root(node)
        distances_k = [max(leaf_dist.get(l, 1.0), 0.01) for l in remaining]

        # Compute phylo-aware weights from the FULL tree (before pruning)
        weights = compute_unique_weights(tree, tree.name, remaining)

        # Resume-merge: if this family already has an entry, mutate it
        # in place so existing methods are preserved and only missing
        # methods are filled in. Avoids the duplicate-row bug where a
        # second `result` dict gets appended for the same family.
        if fam in fam_to_idx:
            result = results[fam_to_idx[fam]]
        else:
            result = {
                'family': fam,
                'held_out': held_out,
                'true_len': int(len(true_seq)),
                'true_seq': [int(x) for x in true_seq],
                'n_cols': C,
                'K': len(remaining),
                'mean_dist': float(np.mean(distances_k)),
            }

        # Common TKF92 pair HMM for scoring pred vs true (used by all methods)
        t_score = float(np.mean(distances_k))  # representative distance
        log_chi_score, st_score, sub_score, pi_score = make_tkf92_pair_hmm(
            TKF92_INS, TKF92_DEL, t_score, TKF92_EXT,
            jnp.array(Q_lg_np), jnp.array(pi_lg_np))
        log_chi_s = np.asarray(log_chi_score)
        st_s = np.asarray(st_score)
        sub_s = np.asarray(sub_score)
        pi_s = np.asarray(pi_lg_np)

        # Also compute log P(true | remaining seqs, TKF92) via composite forward
        # This is the proper generative score — run forward with true_seq as ancestor
        def composite_log_prob_true(log_chi_list, st_list, sub_mats_list, pis_list_k,
                                     n_dom, n_frag, s_trans, s_pi, s_end, w):
            """Log P(true_seq | descendants) under composite model."""
            from tkfmixdom.jax.tree.composite_beam import _advance_ancestor_row, _terminal_score, _precompute_all_emit_rows
            from scipy.special import logsumexp as _logsumexp
            K = len(desc_seqs_k)
            A = 20
            singlet_weight = 1.0 - sum(w)
            # For each descendant, run full forward with true_seq as ancestor
            total_pair_ll = 0.0
            for k in range(K):
                emit = pair_hmm_emissions_per_domain(
                    jnp.array(st_list[k]), jnp.array(true_seq, dtype=jnp.int32),
                    jnp.array(desc_seqs_k[k], dtype=jnp.int32),
                    jnp.array(sub_mats_list[k]), jnp.array(pis_list_k[k]),
                    n_dom, n_frag)
                ll, _, _ = forward_backward_2d(
                    jnp.array(log_chi_list[k]), jnp.array(st_list[k]),
                    jnp.array(true_seq, dtype=jnp.int32),
                    jnp.array(desc_seqs_k[k], dtype=jnp.int32),
                    None, None, log_emit_table=emit)
                total_pair_ll += w[k] * float(ll)
            # Singlet
            singlet_ll = 0.0
            s_pi_np = np.asarray(s_pi)
            s_trans_np = np.asarray(s_trans)
            s_end_np = np.asarray(s_end)
            for pos, a in enumerate(true_seq):
                if pos == 0:
                    singlet_ll += s_pi_np[a]
                else:
                    singlet_ll += s_trans_np[true_seq[pos-1], a]
            if len(true_seq) > 0:
                singlet_ll += s_end_np[true_seq[-1]]
            return singlet_weight * singlet_ll + total_pair_ll

        # === 1. Felsenstein LG08 ===
        try:
            if not _method_enabled('fels') or 'fels' in result:
                raise _SkipMethod()
            fels_seq, fels_time = run_felsenstein(
                tree, held_out, remaining, msa, C, Q_lg_np, pi_lg_np)
            # Score pred vs true using FB alignment
            fels_score = score_prediction(
                fels_seq, true_seq, log_chi_s, st_s, sub_s, pi_s)
            # Also column-level scoring (Felsenstein control)
            pruned_tree_f, _ = prune_leaf_keep_parent(tree, held_out)
            name_internal_nodes(pruned_tree_f)
            pruned_msa_f = {}
            for n in remaining:
                if n in msa:
                    s = msa[n].copy()
                    s[s >= 20] = -1
                    pruned_msa_f[n] = s
            _, fels_post = marginal_ancestor_all_columns_jax(
                pruned_tree_f, pruned_msa_f, Q_lg_np, pi_lg_np)
            true_msa_col = msa[held_out].copy()
            true_msa_col[true_msa_col >= 20] = -1
            fels_col = score_felsenstein_columns(fels_post, true_msa_col, C)
            nw = _nw_metrics(fels_seq, true_seq)
            result['fels'] = {**fels_score, **nw, 'time': float(fels_time),
                              'pred_seq': [int(x) for x in fels_seq],
                              **{f'fels_{k}': v for k, v in fels_col.items()}}
        except _SkipMethod:
            pass
        except Exception as e:
            log(f'  {fam}: fels error: {e}')
            traceback.print_exc()
            result['fels'] = {'accuracy': -1.0, 'time': 0.0}

        # === 1b–1z. Partition-conditioned reconstruction (PhyloHMM) — per slot ===
        for slot in mixdom_slots:
            mlabel = slot['label']
            try:
                if not _method_enabled(f'partition_{mlabel}') or f'partition_{mlabel}' in result:
                    raise _SkipMethod()
                part_pred, part_time = run_partition_reconstruction_method(
                    tree, held_out, remaining, msa, C,
                    model=slot['partition_model'], config=partition_config)
                part_score = score_prediction(
                    part_pred, true_seq, log_chi_s, st_s, sub_s, pi_s)
                nw = _nw_metrics(part_pred, true_seq)
                result[f'partition_{mlabel}'] = {
                    **part_score, **nw, 'time': float(part_time),
                    'pred_seq': [int(x) for x in part_pred],
                }
            except _SkipMethod:
                pass
            except Exception as e:
                log(f'  {fam}: partition_{mlabel} error: {e}')
                traceback.print_exc()
                result[f'partition_{mlabel}'] = {'accuracy': -1.0, 'time': 0.0}

        # === 2–N. MixDom composite beam (weighted) — per slot ===
        for slot in mixdom_slots:
            mlabel = slot['label']
            params_m = slot['params']
            n_dom_m = slot['n_dom']
            n_frag_m = slot['n_frag']
            s_trans_m = slot['s_trans']
            s_pi_m = slot['s_pi']
            s_end_m = slot['s_end']
            try:
                if not _method_enabled(mlabel) or mlabel in result:
                    raise _SkipMethod()
                m_has_class = all(k in params_m for k in
                                  ('class_pi', 'class_S_exch', 'class_dist'))
                if m_has_class:
                    (lc, st, sm, pl,
                     csm, cpl, cdist) = build_mixdom_beam_data_class(
                        params_m, n_dom_m, n_frag_m, distances_k)
                else:
                    lc, st, sm, pl = build_mixdom_beam_data(
                        params_m, n_dom_m, n_frag_m, distances_k)
                    csm = cpl = cdist = None
                tb = time.time()
                recon_m, score_m = composite_beam_reconstruct_jax(
                    desc_seqs_k, distances_k, lc, st, sm, pl,
                    n_dom_m, n_frag_m, s_trans_m, s_pi_m, s_end_m,
                    beam_width=BEAM_WIDTH,
                    max_len=int(len(true_seq) * 1.5),
                    desc_weights=weights,
                    class_sub_matrices_list=csm,
                    class_pis_list=cpl,
                    class_dist=cdist)
                m_time = time.time() - tb
                m_score = score_prediction(
                    recon_m, true_seq, log_chi_s, st_s, sub_s, pi_s)
                nw = _nw_metrics(recon_m, true_seq)
                m_logp_true = composite_log_prob_true(
                    lc, st, sm, pl, n_dom_m, n_frag_m,
                    s_trans_m, s_pi_m, s_end_m, weights)
                result[mlabel] = {**m_score, **nw, 'time': float(m_time),
                                  'pred_seq': [int(x) for x in recon_m],
                                  'beam_score': float(score_m),
                                  'logp_true': float(m_logp_true)}
            except _SkipMethod:
                pass
            except Exception as e:
                log(f'  {fam}: {mlabel} error: {e}')
                traceback.print_exc()
                result[mlabel] = {'accuracy': -1.0, 'time': 0.0}

        # === 4. TKF92 beam (weighted) ===
        try:
            if not _method_enabled('tkf92') or 'tkf92' in result:
                raise _SkipMethod()
            lc, st, sm, pl = build_tkf92_beam_data(Q_lg_np, pi_lg_np, distances_k)
            tb = time.time()
            recon_tkf, score_tkf = composite_beam_reconstruct_jax(
                desc_seqs_k, distances_k, lc, st, sm, pl,
                1, 1, s_trans_tkf, s_pi_tkf, s_end_tkf,
                beam_width=BEAM_WIDTH,
                max_len=int(len(true_seq) * 1.5),
                desc_weights=weights)
            tkf_time = time.time() - tb
            tkf_score = score_prediction(
                recon_tkf, true_seq, log_chi_s, st_s, sub_s, pi_s)
            nw = _nw_metrics(recon_tkf, true_seq)
            tkf_logp_true = composite_log_prob_true(
                lc, st, sm, pl, 1, 1,
                s_trans_tkf, s_pi_tkf, s_end_tkf, weights)
            result['tkf92'] = {**tkf_score, **nw, 'time': float(tkf_time),
                               'pred_seq': [int(x) for x in recon_tkf],
                               'beam_score': float(score_tkf),
                               'logp_true': float(tkf_logp_true)}
        except _SkipMethod:
            pass
        except Exception as e:
            log(f'  {fam}: tkf92 error: {e}')
            traceback.print_exc()
            result['tkf92'] = {'accuracy': -1.0, 'time': 0.0}

        # === 5-N. Fixed-length beam runs (length = Fitch prediction length) ===
        # Skipped by default — only run when at least one *_fixed method is
        # explicitly listed in RECON_METHODS env var (these doubled the beam
        # cost per family for a comparison that's rarely needed).
        fitch_len = result.get('fels', {}).get('pred_len', len(true_seq))
        _fixed_names = [f'{s["label"]}_fixed' for s in mixdom_slots] + ['tkf92_fixed']
        any_fixed_explicit = (_ENABLED_METHODS is not None
                              and any(m in _ENABLED_METHODS for m in _fixed_names))
        if fitch_len > 0 and any_fixed_explicit:
            fixed_configs = []
            for slot in mixdom_slots:
                fixed_configs.append((
                    f'{slot["label"]}_fixed', slot['params'],
                    slot['n_dom'], slot['n_frag'],
                    slot['s_trans'], slot['s_pi'], slot['s_end'], 'mixdom'))
            fixed_configs.append((
                'tkf92_fixed', None, 1, 1,
                s_trans_tkf, s_pi_tkf, s_end_tkf, 'tkf92'))
            for tag, params_m, nd, nf, s_t, s_p, s_e, model_type in fixed_configs:
                try:
                    if not _method_enabled(tag) or tag in result:
                        raise _SkipMethod()
                    csm = cpl = cdist = None
                    if model_type == 'mixdom':
                        has_class = all(k in params_m for k in
                                        ('class_pi', 'class_S_exch', 'class_dist'))
                        if has_class:
                            (lc, st, sm, pl,
                             csm, cpl, cdist) = build_mixdom_beam_data_class(
                                params_m, nd, nf, distances_k)
                        else:
                            lc, st, sm, pl = build_mixdom_beam_data(
                                params_m, nd, nf, distances_k)
                    else:
                        lc, st, sm, pl = build_tkf92_beam_data(
                            Q_lg_np, pi_lg_np, distances_k)
                    tb = time.time()
                    recon_fix, score_fix = composite_beam_reconstruct_jax(
                        desc_seqs_k, distances_k, lc, st, sm, pl,
                        nd, nf, s_t, s_p, s_e,
                        beam_width=BEAM_WIDTH,
                        max_len=fitch_len + 10,
                        desc_weights=weights,
                        fixed_len=fitch_len,
                        class_sub_matrices_list=csm,
                        class_pis_list=cpl,
                        class_dist=cdist)
                    fix_time = time.time() - tb
                    fix_score = score_prediction(
                        recon_fix, true_seq, log_chi_s, st_s, sub_s, pi_s)
                    nw = _nw_metrics(recon_fix, true_seq)
                    result[tag] = {**fix_score, **nw, 'time': float(fix_time),
                                   'pred_seq': [int(x) for x in recon_fix],
                                   'beam_score': float(score_fix),
                                   'fitch_len': int(fitch_len)}
                except _SkipMethod:
                    pass
                except Exception as e:
                    log(f'  {fam}: {tag} error: {e}')
                    result[tag] = {'accuracy': -1.0, 'time': 0.0}

        # Append only if this is a new family; resume-merge already
        # mutated the existing entry in place.
        if fam not in fam_to_idx:
            fam_to_idx[fam] = len(results)
            results.append(result)
        n_fams += 1

        # Print per-family summary
        def _fmt(d, key='accuracy'):
            v = d.get(key, -1) if isinstance(d, dict) else -1
            return f'{v*100:.1f}%' if isinstance(v, float) and v >= 0 else 'ERR'
        def _fmt_lp(d):
            v = d.get('log_prob', None) or d.get('logp_true', None)
            return f'{v:.1f}' if isinstance(v, float) and v > -1e20 else 'N/A'
        # Note: `accuracy` from score_prediction is already post-FB-alignment
        # accuracy (not raw MSA-column accuracy).
        def _row(label, d):
            return (f'    {label:18s} acc={_fmt(d)} prec={_fmt(d,"precision")} '
                    f'rec={_fmt(d,"recall")} logP={_fmt_lp(d)}')
        log(f'{fam}:')
        log(_row('fels', result.get('fels', {})))
        for slot in mixdom_slots:
            log(_row(f'partition_{slot["label"]}',
                     result.get(f'partition_{slot["label"]}', {})))
        for slot in mixdom_slots:
            log(_row(slot['label'], result.get(slot['label'], {})))
        log(_row('tkf92', result.get('tkf92', {})))
        if any_fixed_explicit:
            for slot in mixdom_slots:
                log(_row(f'{slot["label"]}_fixed',
                         result.get(f'{slot["label"]}_fixed', {})))
            log(_row('tkf92_fixed', result.get('tkf92_fixed', {})))

        # Save after every family
        if n_fams % 5 == 0:
            out_path = os.path.join(os.path.dirname(__file__),
                                    'unified_reconstruction_benchmark.json')
            with open(out_path, 'w') as f:
                json.dump({'results': results, 'n_families': n_fams}, f, indent=2)

    # --- Final summary ---
    log(f'\n{"="*60}')
    log(f'Processed {n_fams} families (spec: {len(families)})')

    method_keys = ['fels']
    method_labels = ['Felsenstein']
    for slot in mixdom_slots:
        method_keys.append(f'partition_{slot["label"]}')
        method_labels.append(f'Partition {slot["label"]}')
    for slot in mixdom_slots:
        method_keys.append(slot['label'])
        method_labels.append(f'MixDom {slot["label"]}')
    method_keys.append('tkf92')
    method_labels.append('TKF92')
    for slot in mixdom_slots:
        method_keys.append(f'{slot["label"]}_fixed')
        method_labels.append(f'{slot["label"]} (fixed)')
    method_keys.append('tkf92_fixed')
    method_labels.append('TKF92 (fixed)')

    # Accuracy / Precision / Recall
    log(f'\n{"Method":<16} {"Accuracy":>8} {"Precision":>9} {"Recall":>8} {"LogP/pos":>9} {"N":>5}')
    log('-' * 60)
    for label, key in zip(method_labels, method_keys):
        accs = [r[key]['accuracy'] for r in results
                if isinstance(r.get(key), dict) and r[key].get('accuracy', -1) >= 0]
        precs = [r[key]['precision'] for r in results
                 if isinstance(r.get(key), dict) and r[key].get('precision', -1) >= 0]
        recs = [r[key]['recall'] for r in results
                if isinstance(r.get(key), dict) and r[key].get('recall', -1) >= 0]
        lps = [r[key].get('log_prob', r[key].get('logp_true', None))
               for r in results if isinstance(r.get(key), dict)]
        lps = [v for v in lps if v is not None and v > -1e20]
        true_lens = [r['true_len'] for r in results
                     if isinstance(r.get(key), dict) and r[key].get('accuracy', -1) >= 0]
        lp_per_pos = [lp/tl for lp, tl in zip(lps, true_lens)] if lps and true_lens else []
        if accs:
            log(f'{label:<16} {np.mean(accs):>7.1%} {np.mean(precs):>8.1%} {np.mean(recs):>7.1%} '
                f'{np.mean(lp_per_pos):>8.2f} {len(accs):>5}')

    # NW-based metrics (CARABS-compatible)
    log(f'\nNW-based metrics (CARABS-compatible):')
    log(f'{"Method":<16} {"NW Acc":>8} {"NW Prec":>9} {"NW Rec":>8} {"N":>5}')
    log('-' * 50)
    for label, key in zip(method_labels, method_keys):
        nw_accs = [r[key]['nw_accuracy'] for r in results
                   if isinstance(r.get(key), dict) and 'nw_accuracy' in r[key]]
        nw_precs = [r[key]['nw_precision'] for r in results
                    if isinstance(r.get(key), dict) and 'nw_precision' in r[key]]
        nw_recs = [r[key]['nw_recall'] for r in results
                   if isinstance(r.get(key), dict) and 'nw_recall' in r[key]]
        if nw_accs:
            log(f'{label:<16} {np.mean(nw_accs):>7.1%} {np.mean(nw_precs):>8.1%} '
                f'{np.mean(nw_recs):>7.1%} {len(nw_accs):>5}')

    # Felsenstein column-level (control)
    fels_col_accs = [r['fels'].get('fels_col_accuracy', -1) for r in results
                     if isinstance(r.get('fels'), dict) and 'fels_col_accuracy' in r['fels']]
    fels_col_lps = [r['fels'].get('fels_col_log_prob', None) for r in results
                    if isinstance(r.get('fels'), dict)]
    fels_col_lps = [v for v in fels_col_lps if v is not None]
    if fels_col_accs:
        fels_col_accs = [v for v in fels_col_accs if v >= 0]
        n_cols = [r['n_cols'] for r in results if isinstance(r.get('fels'), dict)]
        lp_per_col = [lp/nc for lp, nc in zip(fels_col_lps, n_cols)] if fels_col_lps else []
        log(f'\nFelsenstein column-level (control):')
        log(f'  Col accuracy: {np.mean(fels_col_accs):.1%} (vs FB-aligned: above)')
        if lp_per_col:
            log(f'  Col logP/col: {np.mean(lp_per_col):.2f}')

    # Pairwise wins on accuracy
    log(f'\nPairwise comparisons (accuracy, threshold 0.1%):')
    for i, (li, ki) in enumerate(zip(method_labels, method_keys)):
        for j, (lj, kj) in enumerate(zip(method_labels, method_keys)):
            if i >= j:
                continue
            paired = [(r[ki].get('accuracy', -1), r[kj].get('accuracy', -1))
                      for r in results
                      if isinstance(r.get(ki), dict) and isinstance(r.get(kj), dict)
                      and r[ki].get('accuracy', -1) >= 0 and r[kj].get('accuracy', -1) >= 0]
            if not paired:
                continue
            i_wins = sum(1 for a, b in paired if a > b + 0.001)
            j_wins = sum(1 for a, b in paired if b > a + 0.001)
            ties = len(paired) - i_wins - j_wins
            log(f'  {li} vs {lj}: {i_wins}/{j_wins}/{ties} (win/lose/tie)')

    # Timing
    log(f'\nMean times:')
    for label, key in zip(method_labels, method_keys):
        times = [r[key].get('time', 0) for r in results
                 if isinstance(r.get(key), dict) and r[key].get('time', 0) > 0]
        if times:
            log(f'  {label}: {np.mean(times):.2f}s')

    # Save final
    out_path = os.path.join(os.path.dirname(__file__),
                            'unified_reconstruction_benchmark.json')
    output = {
        'benchmark': 'unified_reconstruction',
        'tree_source': 'FastTree ML',
        'spec': 'unified_benchmark_spec.json',
        'n_families': n_fams,
        'beam_width': BEAM_WIDTH,
        'tkf92_params': {'ins': TKF92_INS, 'del': TKF92_DEL, 'ext': TKF92_EXT},
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    log(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
