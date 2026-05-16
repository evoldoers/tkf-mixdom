#!/usr/bin/env python3
"""Composite beam WEIGHTED vs UNWEIGHTED vs Felsenstein benchmark.

Compares three ancestral reconstruction strategies on Pfam test families:
  1. Beam (uniform weights, desc_weights=None)
  2. Beam (phylogeny-aware weights via compute_unique_weights)
  3. Felsenstein marginal reconstruction (LG08)

Uses BW d3f2 model, JAX CPU, float64.
"""
import numpy as np
import json
import time
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from tkfmixdom.jax.tree.composite_beam_jax import composite_beam_reconstruct_jax
from tkfmixdom.jax.tree.composite_beam import compute_unique_weights
from tkfmixdom.jax.models.mixdom import build_nested_trans, state_types as mixdom_state_types
from tkfmixdom.jax.distill.maraschino import (
    load_params, build_rate_matrix, precompute_mixdom, distill_mixdom, normalize_freqs_wfst)
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.dp.hmm import safe_log
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
from tkfmixdom.jax.tree.tree_varanc import infer_internal_presence, name_internal_nodes
from tkfmixdom.jax.tree.guide_tree import neighbor_joining
from tkfmixdom.jax.util.io import AA_TO_INT, TreeNode
from experiments.ancrec_benchmark import (
    parse_sto, msa_pairwise_distances, PFAM_DIR, SPLITS_PATH,
    needleman_wunsch_identity)

t0 = time.time()
def log(msg): print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)

# Load BW d3f2 model
params, n_dom, n_cls = load_params('params/best/bw_d3f2_fullseed_15iter.npz')
n_frag = 2
S_exch = np.asarray(params['S_exch'])
pis = np.asarray(params['pi'])
v = np.asarray(params['v'])
pi_avg = np.sum(v[:, None] * pis, axis=0)
pi_avg = pi_avg / pi_avg.sum()

precomp = precompute_mixdom(params, max(n_cls, 1))
dist_ref = distill_mixdom(params, 0.1, max(n_cls, 1), precomp)
wfst_ref = normalize_freqs_wfst(dist_ref)
s_trans = np.log(np.maximum(np.array(wfst_ref['singlet_trans']), 1e-300))
s_start = np.array(wfst_ref['singlet_start'])
s_start = s_start / s_start.sum()
s_pi = np.log(np.maximum(s_start, 1e-300))
s_end = np.log(np.maximum(np.array(wfst_ref['singlet_end']), 1e-300))

Q_lg, pi_lg = rate_matrix_lg()
Q_lg_np, pi_lg_np = np.asarray(Q_lg), np.asarray(pi_lg)

with open(SPLITS_PATH) as f:
    test_fams = json.load(f)['test']

log(f'Model loaded: n_dom={n_dom}, n_frag={n_frag}')
log(f'{len(test_fams)} test families')


def prune_leaf(tree, leaf_name):
    """Remove a leaf from the tree and return a copy."""
    import copy
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


results = []
n_fams = 0

for fam in test_fams:
    sto = os.path.join(PFAM_DIR, f'{fam}.sto')
    if not os.path.exists(sto):
        continue
    seqs = parse_sto(sto)
    n = len(seqs)
    C = len(next(iter(seqs.values())))
    if not (5 <= n <= 12 and 40 <= C <= 80):
        continue

    try:
        nj_names, dist_mat = msa_pairwise_distances(seqs, Q_lg_np, pi_lg_np)
        tree = neighbor_joining(dist_mat, nj_names)
        name_internal_nodes(tree)
    except Exception:
        continue

    # Build MSA as int arrays
    msa = {}
    for name in seqs:
        seq = np.full(C, -1, dtype=np.int32)
        for j, ch in enumerate(seqs[name]):
            if ch in AA_TO_INT:
                seq[j] = AA_TO_INT[ch]
        msa[name] = seq

    leaves = [l.name for l in tree.leaves()]
    holdouts = leaves[:3]

    for held_out in holdouts:
        true_seq = np.array([c for c in msa[held_out] if c >= 0], dtype=np.int32)
        if len(true_seq) < 10:
            continue

        remaining = [l for l in leaves if l != held_out]
        desc_seqs_k = [np.array([c for c in msa[l] if c >= 0], dtype=np.int32)
                       for l in remaining]

        # Distances: pairwise distance / 2 (approximate parent-to-leaf)
        idx_ho = nj_names.index(held_out)
        distances_k = [max(dist_mat[idx_ho, nj_names.index(l)] / 2, 0.01)
                       for l in remaining]

        # === Build per-descendant beam data ===
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
            sub_mats = np.stack([np.asarray(transition_matrix(jnp.array(build_rate_matrix(jnp.array(S_exch[d]),
                                             jnp.array(pis[d]))), t)) for d in range(n_dom)])
            log_chi_list.append(np.asarray(safe_log(chi)))
            st_list.append(st)
            sub_mats_list.append(sub_mats)
            pis_list_k.append(pis)

        # === Compute phylogeny-aware weights ===
        # We need the tree with the held-out node as the "target" (root proxy).
        # Since held_out is a leaf, use the tree root as the target node.
        # The weights reflect unique path fractions from each remaining leaf to root.
        weights = compute_unique_weights(tree, tree.name, remaining)

        # === Beam reconstruction (UNIFORM) ===
        tb = time.time()
        recon_unif, score_unif = composite_beam_reconstruct_jax(
            desc_seqs_k, distances_k,
            log_chi_list, st_list, sub_mats_list, pis_list_k,
            n_dom, n_frag, s_trans, s_pi, s_end,
            beam_width=30, max_len=int(len(true_seq) * 1.5),
            desc_weights=None)
        unif_time = time.time() - tb
        unif_id, _, _ = needleman_wunsch_identity(recon_unif, true_seq)

        # === Beam reconstruction (WEIGHTED) ===
        tw = time.time()
        recon_wt, score_wt = composite_beam_reconstruct_jax(
            desc_seqs_k, distances_k,
            log_chi_list, st_list, sub_mats_list, pis_list_k,
            n_dom, n_frag, s_trans, s_pi, s_end,
            beam_width=30, max_len=int(len(true_seq) * 1.5),
            desc_weights=weights)
        wt_time = time.time() - tw
        wt_id, _, _ = needleman_wunsch_identity(recon_wt, true_seq)

        # === Felsenstein on PRUNED tree ===
        tf = time.time()
        try:
            pruned_tree = prune_leaf(tree, held_out)
            name_internal_nodes(pruned_tree)
            pruned_msa = {l: msa[l] for l in remaining}
            ancestor, _ = marginal_ancestor_all_columns_jax(
                pruned_tree, pruned_msa, Q_lg_np, pi_lg_np)
            leaf_pres = {l: np.array(pruned_msa[l] >= 0, dtype=bool)
                         for l in pruned_msa}
            root_pres = infer_internal_presence(pruned_tree, leaf_pres)
            rp = root_pres.get(pruned_tree.name,
                               np.ones(C, dtype=bool))
            fels_seq = np.array([int(ancestor[c]) for c in range(len(ancestor))
                                 if c < len(rp) and rp[c] and ancestor[c] >= 0],
                                dtype=np.int32)
            fels_id, _, _ = needleman_wunsch_identity(fels_seq, true_seq)
        except Exception as e:
            fels_id = -1.0
            fels_seq = np.array([], dtype=np.int32)
        fels_time = time.time() - tf

        results.append({
            'family': fam, 'held_out': held_out,
            'uniform_id': float(unif_id),
            'weighted_id': float(wt_id),
            'fels_id': float(fels_id),
            'weights': [float(w) for w in weights],
            'uniform_len': int(len(recon_unif)),
            'weighted_len': int(len(recon_wt)),
            'true_len': int(len(true_seq)),
            'fels_len': int(len(fels_seq)),
            'uniform_time': float(unif_time),
            'weighted_time': float(wt_time),
            'fels_time': float(fels_time),
            'K': len(remaining),
            'mean_dist': float(np.mean(distances_k)),
        })
        log(f'{fam}/{held_out[:20]}: uniform={unif_id:.0%} weighted={wt_id:.0%} '
            f'fels={fels_id:.0%} K={len(remaining)}')

    n_fams += 1
    if n_fams >= 20:
        break

# Summary
if results:
    unif_ids = [r['uniform_id'] for r in results]
    wt_ids = [r['weighted_id'] for r in results]
    fels_ids = [r['fels_id'] for r in results if r['fels_id'] >= 0]

    log(f'\n=== {len(results)} holdouts from {n_fams} families ===')
    log(f'Uniform:  mean={np.mean(unif_ids):.1%}, median={np.median(unif_ids):.1%}')
    log(f'Weighted: mean={np.mean(wt_ids):.1%}, median={np.median(wt_ids):.1%}')
    if fels_ids:
        log(f'Fels:     mean={np.mean(fels_ids):.1%}, median={np.median(fels_ids):.1%}')

    # Pairwise comparisons
    paired_uw = [(r['uniform_id'], r['weighted_id']) for r in results]
    u_wins = sum(1 for u, w in paired_uw if u > w + 0.001)
    w_wins = sum(1 for u, w in paired_uw if w > u + 0.001)
    ties_uw = len(paired_uw) - u_wins - w_wins
    log(f'Uniform vs Weighted: U wins={u_wins}, W wins={w_wins}, ties={ties_uw}')

    paired_wf = [(r['weighted_id'], r['fels_id']) for r in results if r['fels_id'] >= 0]
    if paired_wf:
        w_wins_f = sum(1 for w, f in paired_wf if w > f + 0.001)
        f_wins = sum(1 for w, f in paired_wf if f > w + 0.001)
        ties_wf = len(paired_wf) - w_wins_f - f_wins
        log(f'Weighted vs Fels:    W wins={w_wins_f}, F wins={f_wins}, ties={ties_wf}')

    paired_uf = [(r['uniform_id'], r['fels_id']) for r in results if r['fels_id'] >= 0]
    if paired_uf:
        u_wins_f = sum(1 for u, f in paired_uf if u > f + 0.001)
        f_wins_u = sum(1 for u, f in paired_uf if f > u + 0.001)
        ties_uf = len(paired_uf) - u_wins_f - f_wins_u
        log(f'Uniform vs Fels:     U wins={u_wins_f}, F wins={f_wins_u}, ties={ties_uf}')

    log(f'Mean times: uniform={np.mean([r["uniform_time"] for r in results]):.1f}s, '
        f'weighted={np.mean([r["weighted_time"] for r in results]):.1f}s, '
        f'fels={np.mean([r["fels_time"] for r in results]):.1f}s')

out_path = 'experiments/beam_weighted_benchmark.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
log(f'Saved to {out_path}')
