#!/usr/bin/env python3
"""Composite beam (JAX) vs Felsenstein ancestral reconstruction benchmark.

Uses BW d3f2 model for beam search. Felsenstein uses LG08.
Hold-out evaluation on Pfam test families.
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
    # Find and remove the leaf
    for node in new_tree.preorder():
        for i, child in enumerate(node.children):
            if child.name == leaf_name and child.is_leaf:
                # If parent has 2 children, merge sibling up
                if len(node.children) == 2:
                    sibling = node.children[1 - i]
                    sibling.branch_length += node.branch_length
                    if node.parent is not None:
                        idx = node.parent.children.index(node)
                        node.parent.children[idx] = sibling
                        sibling.parent = node.parent
                    else:
                        # node is root, sibling becomes root
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

        # === Beam reconstruction ===
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

        tb = time.time()
        recon, score = composite_beam_reconstruct_jax(
            desc_seqs_k, distances_k,
            log_chi_list, st_list, sub_mats_list, pis_list_k,
            n_dom, n_frag, s_trans, s_pi, s_end,
            beam_width=30, max_len=int(len(true_seq) * 1.5))
        beam_time = time.time() - tb
        beam_id, _, _ = needleman_wunsch_identity(recon, true_seq)

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
            'beam_id': float(beam_id), 'fels_id': float(fels_id),
            'beam_len': int(len(recon)), 'true_len': int(len(true_seq)),
            'fels_len': int(len(fels_seq)),
            'beam_time': float(beam_time), 'fels_time': float(fels_time),
            'K': len(remaining),
            'mean_dist': float(np.mean(distances_k)),
        })
        log(f'{fam}/{held_out[:20]}: beam={beam_id:.1%}(L={len(recon)}) '
            f'fels={fels_id:.1%}(L={len(fels_seq)}) '
            f'true_L={len(true_seq)} K={len(remaining)} t={beam_time:.1f}s')

    n_fams += 1
    if n_fams >= 20:
        break

# Summary
if results:
    beam_ids = [r['beam_id'] for r in results]
    fels_ids = [r['fels_id'] for r in results if r['fels_id'] >= 0]
    log(f'\n=== {len(results)} holdouts from {n_fams} families ===')
    log(f'Beam:  mean={np.mean(beam_ids):.1%}, median={np.median(beam_ids):.1%}')
    if fels_ids:
        log(f'Fels:  mean={np.mean(fels_ids):.1%}, median={np.median(fels_ids):.1%}')
    paired = [(r['beam_id'], r['fels_id']) for r in results if r['fels_id'] >= 0]
    if paired:
        beam_wins = sum(1 for b, f in paired if b > f)
        fels_wins = sum(1 for b, f in paired if f > b)
        ties = sum(1 for b, f in paired if abs(b - f) < 0.001)
        log(f'Beam wins: {beam_wins}, Fels wins: {fels_wins}, Ties: {ties}')
    log(f'Mean beam time: {np.mean([r["beam_time"] for r in results]):.1f}s')

with open('experiments/beam_vs_fels_results.json', 'w') as f:
    json.dump(results, f, indent=2)
log('Saved to experiments/beam_vs_fels_results.json')


if __name__ == '__main__':
    pass
