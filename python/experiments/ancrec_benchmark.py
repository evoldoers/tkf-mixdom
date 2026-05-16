#!/usr/bin/env python3
"""Ancestral reconstruction benchmark on TreeFam + Pfam.

Methods (character reconstruction):
  felsenstein — vectorized marginal posterior via marginal_ancestor_all_columns_jax
  viterbi     — MSA-constrained associative-scan Viterbi

Gap pattern:
  fitch    — Fitch parsimony: ancestor present if BOTH child subtrees have a leaf
  triad    — TKF92 progressive reconstruction (2D Viterbi pairwise alignment)
  triad_1d — TKF92 progressive reconstruction constrained to guide MSA columns

To add a new character reconstruction method, define a function with signature:
    method_name(pruned_tree, pruned_msa_int, Q, pi, **kwargs) -> np.array(int32)
and register it in run_holdout().

Protocol:
  For each family, hold out each leaf one at a time:
  - Remove leaf from tree and MSA
  - Compute gap pattern from remaining MSA
  - Run selected character reconstruction methods
  - Align reconstructed ancestor to true held-out sequence via NW
  - Record: identity, ancestor_length, tau_leaf, timing

Usage:
  uv run python experiments/ancrec_benchmark.py \\
    --treefam-n 100 --methods felsenstein,viterbi \\
    --gap-pattern fitch --output experiments/ancrec_fitch_n100.json
"""

import argparse
import os
import sys
import time
import json
import traceback
import numpy as np

# os.environ.setdefault("JAX_ENABLE_X64", "1")  # Disabled: causes cublas errors on some GPUs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp

from tkfmixdom.jax.util.io import (
    parse_newick, TreeNode, AA_TO_INT, seq_to_int,
)
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.core.params import S, M, I, D, E
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
from tkfmixdom.jax.tree.msa_constrained_viterbi import reconstruct_msa_ancestors
from tkfmixdom.jax.tree.guide_tree import neighbor_joining
from tkfmixdom.jax.distill.maraschino import (
    load_params, precompute_mixdom, distill_mixdom,
    normalize_freqs_wfst, normalize_freqs_hmm,
)
from tkfmixdom.jax.tree.triad_gap_inference import (
    reconstruct_with_triad, extract_triad_msa_presence,
)
from tkfmixdom.jax.tree.triad_1d import reconstruct_1d_triad

# Default paths
TREEFAM_DIR = "/home/yam/bio-datasets/data/treefam/treefam_family_data"
TREEFAM_CLEAN_PATH = os.path.join(os.path.dirname(__file__),
                                   "treefam_clean_families.json")
PFAM_DIR = "/home/yam/bio-datasets/data/pfam-seed"
SPLITS_PATH = os.path.join(PFAM_DIR, "splits", "v1.json")
PARAMS_PATH = "/home/yam/tkf-mixdom/python/pfam/maraschino_d3_trainsplit_entreg.npz"

TIMEOUT = 120.0
SAVE_EVERY = 10
TRIAD_CACHE_DIR = os.path.join(os.path.dirname(__file__), "triad_cache")
TAU_BINS = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, float('inf'))]
TAU_BIN_LABELS = ["[0,0.05)", "[0.05,0.1)", "[0.1,0.2)", "[0.2,0.5)", "[0.5+)"]

# CherryML-fitted TKF92 parameters (from seed_counts.npz, 27K Pfam families)
TKF92_INS_RATE = 0.0458
TKF92_DEL_RATE = 0.0468
TKF92_EXT = 0.683

ALL_METHODS = ["felsenstein", "viterbi"]


# ============================================================
# Triad cache: save/load expensive pairwise Viterbi results
# ============================================================

def _triad_cache_path(family_id, held_out):
    return os.path.join(TRIAD_CACHE_DIR, f"{family_id}__{held_out}.npz")


def _save_triad_cache(family_id, held_out, root_seq, triad_presence,
                      triad_leaf_seqs, triad_msa_len):
    """Save Triad inference results to disk cache."""
    os.makedirs(TRIAD_CACHE_DIR, exist_ok=True)
    path = _triad_cache_path(family_id, held_out)
    data = {'root_seq': root_seq, 'triad_msa_len': triad_msa_len}
    # Save presence and leaf_seqs as separate keys
    for name, pres in triad_presence.items():
        data[f'pres__{name}'] = np.asarray(pres)
    for name, seq in triad_leaf_seqs.items():
        data[f'lseq__{name}'] = np.asarray(seq)
    np.savez_compressed(path, **data)


def _load_triad_cache(family_id, held_out):
    """Load Triad inference results from disk cache. Returns None if miss."""
    path = _triad_cache_path(family_id, held_out)
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=False)
        root_seq = d['root_seq']
        triad_msa_len = int(d['triad_msa_len'])
        triad_presence = {}
        triad_leaf_seqs = {}
        for k in d.files:
            if k.startswith('pres__'):
                triad_presence[k[6:]] = d[k]
            elif k.startswith('lseq__'):
                triad_leaf_seqs[k[6:]] = d[k]
        return root_seq, triad_presence, triad_leaf_seqs, triad_msa_len
    except Exception:
        return None


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


def parse_sto(path):
    """Parse Stockholm format MSA."""
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


# ============================================================
# Tree manipulation
# ============================================================

def remove_leaf(tree, leaf_name):
    """Remove a leaf from the tree, return (pruned_tree, parent_name, branch_length)."""
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


def infer_internal_presence(tree, leaf_presence):
    """Fitch parsimony gap inference for internal nodes.

    Ancestor present at a column if BOTH child subtrees have at least one
    ungapped leaf (intersection rule). Preorder pass: if parent present,
    child present.
    """
    L = len(next(iter(leaf_presence.values())))
    presence = {}
    # Postorder: intersection (present if BOTH children present)
    for node in tree.postorder():
        if node.is_leaf:
            if node.name in leaf_presence:
                presence[node.name] = np.array(leaf_presence[node.name], dtype=bool)
            else:
                presence[node.name] = np.zeros(L, dtype=bool)
        else:
            children_pres = [presence[c.name] for c in node.children if c.name in presence]
            if len(children_pres) >= 2:
                p = children_pres[0] & children_pres[1]
            elif len(children_pres) == 1:
                p = children_pres[0].copy()
            else:
                p = np.zeros(L, dtype=bool)
            presence[node.name] = p
    # Preorder: propagate -- if parent present, child present
    for node in tree.preorder():
        if node.is_root:
            continue
        parent_pres = presence.get(node.parent.name)
        if parent_pres is not None:
            presence[node.name] = presence[node.name] | parent_pres
    return presence


def needleman_wunsch_identity(seq1, seq2, match_score=1, mismatch=-1, gap=-2):
    """NW alignment, return (identity, n_aligned, n_matches)."""
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0, 0, 0
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch)
            delete = dp[i-1, j] + gap
            insert = dp[i, j-1] + gap
            dp[i, j] = max(match, delete, insert)
    i, j = n, m
    matches = aligned = 0
    while i > 0 and j > 0:
        if dp[i, j] == dp[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch):
            aligned += 1
            if seq1[i-1] == seq2[j-1]:
                matches += 1
            i -= 1; j -= 1
        elif dp[i, j] == dp[i-1, j] + gap:
            i -= 1
        else:
            j -= 1
    return matches / max(aligned, 1), aligned, matches


# ============================================================
# MixDom WFST construction
# ============================================================

def build_mixdom_wfst_log(dist_wfst):
    """Convert normalize_freqs_wfst output to log-tensor dict for VarAnc."""
    result = {}
    for key in ['p_mm', 'p_mi', 'p_md', 'p_me',
                'p_im', 'p_ii', 'p_id', 'p_ie',
                'p_dm', 'p_dd', 'p_di', 'p_de',
                'p_sm', 'p_si', 'p_sd', 'p_se']:
        val = dist_wfst[key]
        if isinstance(val, (int, float)):
            result[f'log_{key}'] = float(np.log(max(val, 1e-300)))
        else:
            result[f'log_{key}'] = np.array(
                np.log(np.maximum(np.asarray(val), 1e-300)))
    return result


def build_mixdom_wfst_per_edge(tree, params, n_classes, precomp):
    """Build per-edge MixDom WFST log-tensors, caching by branch length."""
    cache = {}
    wfst_per_edge = {}
    for node in tree.preorder():
        if node.is_root:
            continue
        parent = node.parent
        t = max(node.branch_length, 1e-6)
        t_key = round(t, 6)
        if t_key not in cache:
            dist = distill_mixdom(params, t, n_classes, precomp)
            wfst = normalize_freqs_wfst(dist)
            cache[t_key] = build_mixdom_wfst_log(wfst)
        wfst_per_edge[(parent.name, node.name)] = cache[t_key]
    return wfst_per_edge


# ============================================================
# Method implementations
# ============================================================

def method_felsenstein(pruned_tree, pruned_msa_int, Q, pi):
    """Vectorized Felsenstein marginal posterior at root (Fitch gap pattern)."""
    ancestor, posteriors = marginal_ancestor_all_columns_jax(
        pruned_tree, pruned_msa_int, Q, pi)
    # Apply same Fitch gap pattern as BurlVarAnc
    leaf_names = list(pruned_msa_int.keys())
    leaf_presence = {name: np.array(pruned_msa_int[name] >= 0, dtype=bool)
                     for name in leaf_names}
    root_presence = infer_internal_presence(pruned_tree, leaf_presence)
    root_pres = root_presence[pruned_tree.name]
    return np.array([int(ancestor[c]) for c in range(len(ancestor))
                     if root_pres[c] and ancestor[c] >= 0], dtype=np.int32)


def method_viterbi(pruned_tree, pruned_msa_int, Q, pi):
    """MSA-constrained Viterbi via associative scan (Fitch gap pattern)."""
    ancestor, log_prob = reconstruct_msa_ancestors(
        pruned_tree, pruned_msa_int, Q, pi)
    # Apply same Fitch gap pattern as other methods
    leaf_names = list(pruned_msa_int.keys())
    leaf_presence = {name: np.array(pruned_msa_int[name] >= 0, dtype=bool)
                     for name in leaf_names}
    root_presence = infer_internal_presence(pruned_tree, leaf_presence)
    root_pres = root_presence[pruned_tree.name]
    return np.array([int(ancestor[c]) for c in range(len(ancestor))
                     if root_pres[c] and ancestor[c] >= 0], dtype=np.int32)



# ============================================================
# Hold-out experiment
# ============================================================

def run_holdout(family_id, held_out, seq_names, aln_int, ungapped, tree,
                Q, pi, pi_mix, params, n_classes, precomp, singlet_wfst,
                methods_to_run, verbose, gap_pattern='fitch'):
    """Run selected methods on one held-out leaf."""
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

    result = {
        "family": family_id,
        "held_out": held_out,
        "tau_leaf": float(branch_len),
        "true_seq_len": len(true_seq),
        "n_seqs": len(seq_names),
    }

    method_fns = {}
    if gap_pattern == 'triad_1d_fitch':
        # 1D triad with Fitch floor: triad decides presence, Felsenstein decides characters
        name_internal_nodes(pruned_tree)
        leaf_pres = {n: np.array(pruned_msa[n] >= 0, dtype=bool) for n in pruned_msa}
        A = 20  # amino acids
        leaf_profs = {}
        for n in pruned_msa:
            seq = np.array([c for c in pruned_msa[n] if c >= 0], dtype=np.int32)
            leaf_profs[n] = np.eye(A)[seq]
        node_pres, node_profs, root_seq_1d = reconstruct_1d_triad(
            pruned_tree, leaf_pres, leaf_profs,
            ins_rate=TKF92_INS_RATE, del_rate=TKF92_DEL_RATE,
            Q=Q, pi=pi, use_tkf92=True, ext=TKF92_EXT,
            triad_method='viterbi', fitch_floor=True)
        # Use triad for gap inference, but Felsenstein for character assignment
        root_name = pruned_tree.name
        root_pres = node_pres.get(root_name, np.zeros(len(next(iter(pruned_msa.values()))), dtype=bool))
        def _triad_fels():
            ancestor, _ = marginal_ancestor_all_columns_jax(
                pruned_tree, pruned_msa, Q, pi)
            return np.array([int(ancestor[c]) for c in range(len(ancestor))
                             if c < len(root_pres) and root_pres[c] and ancestor[c] >= 0],
                            dtype=np.int32)
        method_fns["felsenstein"] = _triad_fels

    elif gap_pattern in ('triad', 'triad_1d'):
        # 2D or 1D triad: progressive reconstruction with triad_gap_inference
        name_internal_nodes(pruned_tree)
        use_guide = (gap_pattern == 'triad_1d')
        cache_suffix = '_1d' if use_guide else ''
        cached = _load_triad_cache(family_id, held_out + cache_suffix)
        if cached is not None:
            root_seq, triad_presence, triad_leaf_seqs, triad_msa_len = cached
        else:
            leaf_seqs_ungapped = {n: np.array([c for c in pruned_msa[n] if c >= 0],
                                              dtype=np.int32) for n in pruned_msa}
            # For triad_1d*: pass leaf presence from seed MSA as guide
            # so pairwise alignments follow MSA columns (1D) instead of 2D Viterbi
            if use_guide:
                guide = {n: np.array(pruned_msa[n] >= 0, dtype=bool) for n in pruned_msa}
            else:
                guide = None
            node_profiles, node_alns, root_seq = reconstruct_with_triad(
                pruned_tree, leaf_seqs_ungapped,
                ins_rate=TKF92_INS_RATE, del_rate=TKF92_DEL_RATE, t_scale=1.0,
                Q=Q, pi=pi, use_tkf92=True, ext=TKF92_EXT,
                triad_method='viterbi', guide_msa=guide,
                fitch_floor=use_fitch_floor)
            triad_msa, triad_presence, triad_msa_len = \
                extract_triad_msa_presence(
                    pruned_tree, leaf_seqs_ungapped,
                    node_profiles, node_alns)
            leaf_name_set = set(leaf_seqs_ungapped.keys())
            triad_leaf_seqs = {n: np.array([c for c in triad_msa[n] if c >= 0],
                                           dtype=np.int32)
                               for n in triad_msa if n in leaf_name_set}
            _save_triad_cache(family_id, held_out + cache_suffix, root_seq,
                              triad_presence, triad_leaf_seqs, triad_msa_len)

        # Triad-Felsenstein: direct from triad's root sequence
        method_fns["felsenstein"] = lambda: root_seq

    else:
        # Fitch: standard column-independent methods
        if "felsenstein" in methods_to_run:
            method_fns["felsenstein"] = lambda: method_felsenstein(
                pruned_tree, pruned_msa, Q, pi)
        if "viterbi" in methods_to_run:
            method_fns["viterbi"] = lambda: method_viterbi(
                pruned_tree, pruned_msa, Q, pi)

    for mname, mfn in method_fns.items():
        t0 = time.time()
        try:
            anc_seq = mfn()
            elapsed = time.time() - t0

            if elapsed > TIMEOUT:
                result[f"identity_{mname}"] = None
                result[f"time_{mname}"] = elapsed
                result[f"status_{mname}"] = "timeout"
                continue

            if len(anc_seq) > 0 and len(true_seq) > 0:
                identity, aligned, matches = needleman_wunsch_identity(
                    anc_seq, true_seq)
            else:
                identity, aligned, matches = 0.0, 0, 0

            result[f"identity_{mname}"] = float(identity)
            result[f"matches_{mname}"] = int(matches)
            result[f"aligned_{mname}"] = int(aligned)
            result[f"anc_len_{mname}"] = len(anc_seq)
            result[f"time_{mname}"] = float(elapsed)
            result[f"status_{mname}"] = "ok"

        except Exception as e:
            elapsed = time.time() - t0
            result[f"identity_{mname}"] = None
            result[f"time_{mname}"] = float(elapsed)
            result[f"status_{mname}"] = f"error: {str(e)[:200]}"
            if verbose:
                print(f"    {held_out}/{mname}: ERROR {str(e)[:120]}")
                traceback.print_exc()

    if verbose:
        ids = []
        for m in methods_to_run:
            v = result.get(f"identity_{m}")
            t = result.get(f"time_{m}", 0)
            if v is not None:
                ids.append(f"{m}={v:.3f}({t:.1f}s)")
            else:
                ids.append(f"{m}=ERR")
        print(f"    {held_out} (tau={branch_len:.3f}): {', '.join(ids)}")

    return result


# ============================================================
# TreeFam family selection and processing
# ============================================================

def select_treefam_families(n_families=1000):
    """Select N families from clean families list (5-50 leaves)."""
    with open(TREEFAM_CLEAN_PATH) as f:
        all_families = json.load(f)

    selected = []
    for fam in all_families:
        if len(selected) >= n_families:
            break
        tree_path = os.path.join(TREEFAM_DIR, f"{fam}.nh.emf")
        aln_path = os.path.join(TREEFAM_DIR, f"{fam}.aln.emf")
        if not (os.path.exists(tree_path) and os.path.exists(aln_path)):
            continue
        try:
            with open(tree_path) as tf:
                tree_str = None
                for line in tf:
                    line = line.strip()
                    if '(' in line and not line.startswith('#') and line != '//':
                        tree_str = line
                if tree_str is None:
                    continue
            tree = parse_newick(tree_str)
            n_leaves = len([n for n in tree.preorder() if n.is_leaf])
            if 5 <= n_leaves <= 50:
                selected.append(fam)
        except Exception:
            continue

    return selected[:n_families]


def process_treefam_family(family_id, Q, pi, pi_mix, params, n_classes,
                            precomp, singlet_wfst, methods_to_run,
                            verbose=True, gap_pattern='fitch'):
    """Process one TreeFam family."""
    aln_path = os.path.join(TREEFAM_DIR, f"{family_id}.aln.emf")
    tree_path = os.path.join(TREEFAM_DIR, f"{family_id}.nh.emf")
    fasta_path = os.path.join(TREEFAM_DIR, f"{family_id}.aa.fasta")

    if not os.path.exists(aln_path):
        if verbose:
            print(f"  SKIP {family_id}: alignment not found")
        return []

    seq_names, alignment = parse_emf_alignment(aln_path)
    tree, tree_seq_names = parse_emf_tree(tree_path)
    raw_seqs = parse_fasta(fasta_path)

    tree_leaves = [n.name for n in tree.leaves()]
    common_names = [n for n in seq_names if n in tree_leaves]
    if len(common_names) < 3:
        if verbose:
            print(f"  SKIP {family_id}: too few sequences ({len(common_names)})")
        return []

    # Subsample to 15 leaves for large families
    if len(common_names) > 15:
        rng = np.random.RandomState(hash(family_id) % (2**31))
        indices = rng.choice(len(common_names), 15, replace=False)
        common_names = [common_names[i] for i in sorted(indices)]

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
                        tree, Q, pi, pi_mix, params, n_classes, precomp,
                        singlet_wfst, methods_to_run, verbose,
                        gap_pattern=gap_pattern)
        if r is not None:
            results.append(r)
    return results


# ============================================================
# Pfam family selection and processing
# ============================================================

def select_pfam_families(n_families=100, seed=42):
    """Select families from test split with 5-15 sequences, lengths 50-500."""
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    test_families = splits['test']
    rng = np.random.RandomState(seed)
    rng.shuffle(test_families)

    selected = []
    for fam in test_families:
        if len(selected) >= n_families:
            break
        sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")
        if not os.path.exists(sto_path):
            continue
        try:
            aligned_seqs = parse_sto(sto_path)
        except Exception:
            continue
        n_seqs = len(aligned_seqs)
        if n_seqs < 5 or n_seqs > 15:
            continue
        lengths = [len(seq.replace('-', '').replace('.', ''))
                   for seq in aligned_seqs.values()]
        median_len = np.median(lengths)
        if median_len < 50 or median_len > 500:
            continue
        selected.append(fam)
    return selected


def process_pfam_family(family_id, Q, pi, pi_mix, params, n_classes,
                         precomp, singlet_wfst, methods_to_run,
                         verbose=True, gap_pattern='fitch'):
    """Process one Pfam family (build NJ tree from LG08 pairwise distances)."""
    sto_path = os.path.join(PFAM_DIR, f"{family_id}.sto")
    if not os.path.exists(sto_path):
        return []

    aligned_seqs = parse_sto(sto_path)
    if len(aligned_seqs) < 3:
        return []

    names_list = list(aligned_seqs.keys())
    msa_len = len(next(iter(aligned_seqs.values())))

    # Build NJ tree from LG08 pairwise distances
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

    if verbose:
        print(f"\n  Family: {family_id}  ({len(names)} seqs, aln_len={msa_len})")

    results = []
    for held_out in names:
        if len(ungapped[held_out]) == 0:
            continue
        remaining = [n for n in names if n != held_out]
        if len(remaining) < 2:
            continue
        r = run_holdout(family_id, held_out, names, aln_int, ungapped,
                        tree, Q, pi, pi_mix, params, n_classes, precomp,
                        singlet_wfst, methods_to_run, verbose,
                        gap_pattern=gap_pattern)
        if r is not None:
            results.append(r)
    return results


# ============================================================
# Summary and reporting
# ============================================================

def tau_bin_label(tau):
    """Get tau bin label for a tau value."""
    for (lo, hi), label in zip(TAU_BINS, TAU_BIN_LABELS):
        if lo <= tau < hi:
            return label
    return TAU_BIN_LABELS[-1]


def compute_summary(holdouts, method_names):
    """Compute summary statistics for a set of holdouts."""
    summary = {}
    for m in method_names:
        ids = [r[f"identity_{m}"] for r in holdouts
               if r.get(f"status_{m}") == "ok" and r.get(f"identity_{m}") is not None]
        times = [r[f"time_{m}"] for r in holdouts
                 if r.get(f"status_{m}") == "ok"]

        m_summary = {
            "n": len(ids),
            "mean": float(np.mean(ids)) if ids else None,
            "std": float(np.std(ids)) if ids else None,
            "median": float(np.median(ids)) if ids else None,
            "mean_time": float(np.mean(times)) if times else None,
        }

        by_tau = {}
        for (lo, hi), label in zip(TAU_BINS, TAU_BIN_LABELS):
            bin_holdouts = [r for r in holdouts
                            if r.get("tau_leaf") is not None
                            and lo <= r["tau_leaf"] < hi]
            bin_ids = [r[f"identity_{m}"] for r in bin_holdouts
                       if r.get(f"status_{m}") == "ok"
                       and r.get(f"identity_{m}") is not None]
            by_tau[label] = {
                "n": len(bin_ids),
                "mean": float(np.mean(bin_ids)) if bin_ids else None,
                "std": float(np.std(bin_ids)) if bin_ids else None,
            }
        m_summary["by_tau"] = by_tau
        summary[m] = m_summary
    return summary


def print_summary(holdouts, dataset_name, method_names):
    """Print summary statistics broken down by tau bins."""
    print(f"\n{'='*80}")
    print(f"Summary: {dataset_name} ({len(holdouts)} hold-outs)")
    print(f"{'='*80}")

    print(f"\nOverall averages:")
    for m in method_names:
        ids = [r[f"identity_{m}"] for r in holdouts
               if r.get(f"status_{m}") == "ok" and r.get(f"identity_{m}") is not None]
        times = [r[f"time_{m}"] for r in holdouts
                 if r.get(f"status_{m}") == "ok"]
        n_ok = len(ids)
        if ids:
            print(f"  {m:<20}: mean_id={np.mean(ids):.4f} "
                  f"(std={np.std(ids):.4f}), mean_time={np.mean(times):.2f}s, "
                  f"n={n_ok}")
        else:
            print(f"  {m:<20}: no successful runs")

    print(f"\nBreakdown by tau_leaf:")
    header = f"  {'tau_bin':<12} {'n':>4}"
    for m in method_names:
        header += f" {m:>14}"
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
            if ids:
                row += f" {np.mean(ids):>14.4f}"
            else:
                row += f" {'N/A':>14}"
        print(row)


def print_progress(dataset_name, n_done, n_total, holdouts, method_names,
                   t_start):
    """Print progress summary with running mean identity and ETA."""
    elapsed = time.time() - t_start
    rate = n_done / max(elapsed, 1)
    remaining = (n_total - n_done) / max(rate, 1e-6)
    eta_min = remaining / 60.0

    print(f"\n  --- Progress: {dataset_name} {n_done}/{n_total} families, "
          f"{len(holdouts)} hold-outs, ETA {eta_min:.1f} min ---")
    for m in method_names:
        ids = [r[f"identity_{m}"] for r in holdouts
               if r.get(f"status_{m}") == "ok" and r.get(f"identity_{m}") is not None]
        if ids:
            print(f"    {m:<20}: mean_id={np.mean(ids):.4f} (n={len(ids)})")


# ============================================================
# Resume support
# ============================================================

def load_existing_results(output_path):
    """Load existing results for resume support."""
    if not os.path.exists(output_path):
        return None
    try:
        with open(output_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_completed_families(output, dataset_key):
    """Get set of families already completed in output."""
    if output is None or dataset_key not in output:
        return set()
    holdouts = output.get(dataset_key, {}).get("holdouts", [])
    return set(r["family"] for r in holdouts)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production ancestral reconstruction benchmark")
    parser.add_argument("--treefam-n", type=int, default=1000,
                        help="Number of TreeFam families (default: 1000)")
    parser.add_argument("--pfam-n", type=int, default=100,
                        help="Number of Pfam families (default: 100)")
    parser.add_argument("--methods", type=str, default="felsenstein,viterbi",
                        help="Comma-separated list of methods (default: felsenstein,viterbi)")
    parser.add_argument("--gap-pattern", type=str, default="fitch",
                        choices=["fitch", "triad", "triad_1d", "triad_1d_fitch"],
                        help="Gap pattern method (default: fitch)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "ancrec_production.json"),
                        help="Output JSON path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output (default: True)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-holdout output")
    parser.add_argument("--treefam-only", action="store_true",
                        help="Run only TreeFam families")
    parser.add_argument("--pfam-only", action="store_true",
                        help="Run only Pfam families")
    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    methods_to_run = [m.strip() for m in args.methods.split(",")]
    for m in methods_to_run:
        if m not in ALL_METHODS:
            print(f"ERROR: unknown method '{m}'. Valid: {ALL_METHODS}")
            sys.exit(1)

    output_path = args.output
    t_start = time.time()

    print(f"Ancestral reconstruction production benchmark")
    print(f"  TreeFam families: {args.treefam_n}")
    print(f"  Pfam families:    {args.pfam_n}")
    print(f"  Methods:          {', '.join(methods_to_run)}")
    print(f"  Gap pattern:      {args.gap_pattern}")
    print(f"  Output:           {output_path}")
    print(f"  Resume:           {args.resume}")

    # Load existing results for resume
    existing = None
    if args.resume:
        existing = load_existing_results(output_path)
        if existing is not None:
            n_tf = len(existing.get("treefam", {}).get("holdouts", []))
            n_pf = len(existing.get("pfam", {}).get("holdouts", []))
            print(f"  Resuming: {n_tf} TreeFam + {n_pf} Pfam holdouts found")

    # Initialize output structure
    if existing is not None:
        output = existing
    else:
        output = {
            "treefam": {"families": [], "holdouts": []},
            "pfam": {"families": [], "holdouts": []},
            "summary": {},
            "config": {
                "methods": methods_to_run,
                "gap_pattern": args.gap_pattern,
                "treefam_n": args.treefam_n,
                "pfam_n": args.pfam_n,
            },
        }

    # Load models
    print("\nLoading LG08 rate matrix...")
    Q, pi = rate_matrix_lg()
    Q_np = np.array(Q)
    pi_np = np.array(pi)

    # MixDom parameters (available for future methods; not required for felsenstein/viterbi)
    params = n_classes = precomp = pi_mix = singlet_wfst = None

    # ---- TreeFam ----
    if not args.pfam_only:
        print(f"\n{'='*80}")
        print(f"Selecting {args.treefam_n} TreeFam families...")
        print(f"{'='*80}")

        treefam_families = select_treefam_families(args.treefam_n)
        print(f"Selected {len(treefam_families)} TreeFam families")
        output["treefam"]["families"] = treefam_families

        completed_tf = get_completed_families(output, "treefam") if args.resume else set()
        if completed_tf:
            print(f"  Skipping {len(completed_tf)} already-completed families")

        t_tf_start = time.time()
        n_done = 0
        for fam in treefam_families:
            if fam in completed_tf:
                n_done += 1
                continue

            try:
                results = process_treefam_family(
                    fam, Q_np, pi_np, pi_mix, params, n_classes, precomp,
                    singlet_wfst, methods_to_run, args.verbose,
                    gap_pattern=args.gap_pattern)
                output["treefam"]["holdouts"].extend(results)
            except Exception as e:
                print(f"  ERROR on {fam}: {e}")
                traceback.print_exc()

            n_done += 1
            if n_done % SAVE_EVERY == 0:
                print_progress("TreeFam", n_done, len(treefam_families),
                              output["treefam"]["holdouts"], methods_to_run,
                              t_tf_start)
                with open(output_path, 'w') as f:
                    json.dump(output, f, indent=2)
                print(f"  Incremental save to {output_path}")

        print_summary(output["treefam"]["holdouts"], "TreeFam", methods_to_run)
        output["summary"]["treefam"] = compute_summary(
            output["treefam"]["holdouts"], methods_to_run)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nTreeFam results saved to {output_path}")

    # ---- Pfam ----
    if not args.treefam_only:
        print(f"\n{'='*80}")
        print(f"Selecting {args.pfam_n} Pfam test-split families...")
        print(f"{'='*80}")

        pfam_families = select_pfam_families(n_families=args.pfam_n)
        print(f"Selected {len(pfam_families)} Pfam families")
        output["pfam"]["families"] = pfam_families

        completed_pf = get_completed_families(output, "pfam") if args.resume else set()
        if completed_pf:
            print(f"  Skipping {len(completed_pf)} already-completed families")

        t_pf_start = time.time()
        n_done = 0
        for fam in pfam_families:
            if fam in completed_pf:
                n_done += 1
                continue

            try:
                results = process_pfam_family(
                    fam, Q_np, pi_np, pi_mix, params, n_classes, precomp,
                    singlet_wfst, methods_to_run, args.verbose,
                    gap_pattern=args.gap_pattern)
                output["pfam"]["holdouts"].extend(results)
            except Exception as e:
                print(f"  ERROR on {fam}: {e}")
                traceback.print_exc()

            n_done += 1
            if n_done % SAVE_EVERY == 0:
                print_progress("Pfam", n_done, len(pfam_families),
                              output["pfam"]["holdouts"], methods_to_run,
                              t_pf_start)
                with open(output_path, 'w') as f:
                    json.dump(output, f, indent=2)
                print(f"  Incremental save to {output_path}")

        print_summary(output["pfam"]["holdouts"], "Pfam", methods_to_run)
        output["summary"]["pfam"] = compute_summary(
            output["pfam"]["holdouts"], methods_to_run)

    # Final save
    elapsed_total = time.time() - t_start
    output["total_time_seconds"] = float(elapsed_total)
    output["n_treefam_families"] = len(output["treefam"].get("families", []))
    output["n_pfam_families"] = len(output["pfam"].get("families", []))
    output["n_treefam_holdouts"] = len(output["treefam"].get("holdouts", []))
    output["n_pfam_holdouts"] = len(output["pfam"].get("holdouts", []))

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"DONE. Total time: {elapsed_total/60:.1f} minutes")
    print(f"  TreeFam: {output['n_treefam_families']} families, "
          f"{output['n_treefam_holdouts']} hold-outs")
    print(f"  Pfam:    {output['n_pfam_families']} families, "
          f"{output['n_pfam_holdouts']} hold-outs")
    print(f"  Results: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
