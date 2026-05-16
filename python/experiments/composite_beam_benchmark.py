#!/usr/bin/env python3
"""Benchmark composite beam reconstruction on Pfam test families.

Compares composite_beam_reconstruct (MixDom pair HMM beam search)
against Felsenstein marginal (column-by-column MAP) for ancestral
sequence reconstruction accuracy.

Protocol:
  For each family, hold out up to 3 leaves. The remaining leaves
  become "descendants" on a star phylogeny. Distances come from
  NJ tree path lengths. Compare reconstructed ancestor to held-out
  ground truth via NW alignment identity.

Usage:
  JAX_PLATFORMS=cpu uv run python experiments/composite_beam_benchmark.py
"""

import os
import sys
import time
import json
import traceback
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp

from tkfmixdom.jax.util.io import parse_newick, TreeNode, AA_TO_INT, seq_to_int
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.core.params import S, M, I, D, E
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
from tkfmixdom.jax.tree.guide_tree import neighbor_joining
from tkfmixdom.jax.models.mixdom import (
    build_nested_trans, state_types as mixdom_state_types,
)
from tkfmixdom.jax.distill.maraschino import (
    load_params, precompute_mixdom, distill_mixdom, normalize_freqs_wfst,
    build_rate_matrix, eigen_decompose, transition_probs_from_eigen,
)
from tkfmixdom.jax.tree.composite_beam import composite_beam_reconstruct

# Import helpers from ancrec_benchmark
from ancrec_benchmark import (
    parse_sto, msa_pairwise_distances, needleman_wunsch_identity,
    remove_leaf, find_parent_of_leaf, name_internal_nodes,
    infer_internal_presence, method_felsenstein,
    PFAM_DIR, SPLITS_PATH,
)


# ============================================================
# Paths
# ============================================================
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..",
                           "params", "best", "bw_d3f2_fullseed_15iter.npz")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__),
                           "composite_beam_mixdom_results.json")

MAX_FAMILIES = 10
MAX_HOLDOUTS = 1
BEAM_WIDTH = 3
MIN_SEQS = 5
MAX_SEQS = 7
MIN_COLS = 20
MAX_COLS = 120
MAX_UNGAPPED_LEN = 50  # max ungapped desc length for beam tractability
MAX_DESC_FOR_BEAM = 3  # limit descendants to keep beam tractable
SINGLET_T = 0.1  # distance for singlet model distillation


def select_pfam_families_filtered(n_families, seed=42):
    """Select Pfam test families: 5-15 seqs, 50-100 MSA columns."""
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
        if n_seqs < MIN_SEQS or n_seqs > MAX_SEQS:
            continue
        aln_len = len(next(iter(aligned_seqs.values())))
        if aln_len < MIN_COLS or aln_len > MAX_COLS:
            continue
        # Check ungapped lengths are reasonable and short enough for beam
        lengths = [len(seq.replace('-', '').replace('.', ''))
                   for seq in aligned_seqs.values()]
        if min(lengths) < 10:
            continue
        if np.median(lengths) > MAX_UNGAPPED_LEN:
            continue
        selected.append(fam)
    return selected


def tree_path_distance(tree, node_a_name, node_b_name):
    """Compute tree path distance between two named nodes."""
    name_to_node = {}
    for n in tree.preorder():
        if n.name:
            name_to_node[n.name] = n

    if node_a_name not in name_to_node or node_b_name not in name_to_node:
        return None

    # Build adjacency with distances
    adj = {}
    for n in tree.preorder():
        nid = id(n)
        if nid not in adj:
            adj[nid] = []
        for c in n.children:
            adj[nid].append((id(c), c.branch_length))
            cid = id(c)
            if cid not in adj:
                adj[cid] = []
            adj[cid].append((nid, c.branch_length))

    # BFS
    start = id(name_to_node[node_a_name])
    target = id(name_to_node[node_b_name])
    visited = {start: 0.0}
    queue = [start]
    while queue:
        curr = queue.pop(0)
        if curr == target:
            return visited[curr]
        for nxt, bl in adj.get(curr, []):
            if nxt not in visited:
                visited[nxt] = visited[curr] + bl
                queue.append(nxt)
    return None


def build_per_desc_pair_hmm(params, n_dom, n_frag, t):
    """Build MixDom pair HMM transition matrix and per-domain P(t) for distance t.

    Returns:
        log_chi: (ns, ns) log transition matrix
        st: (ns,) state types
        sub_matrices: (n_dom, A, A) per-domain substitution matrices
        pis: (n_dom, A) per-domain equilibrium
    """
    frag_weights = params.get('frag_weights')
    r_frags = params.get('r_frags')
    if frag_weights is None:
        frag_weights = jnp.ones((n_dom, n_frag)) / n_frag
    if r_frags is None:
        r_frags = params['r'][:, None] * jnp.ones((n_dom, n_frag))

    chi, state_map = build_nested_trans(
        params['lam0'], params['mu0'], t,
        params['lam'], params['mu'], params['v'],
        frag_weights, r_frags,
    )
    chi_np = np.array(chi)

    st = np.array(mixdom_state_types(n_dom, n_frag))

    # Per-domain substitution matrices
    pis = np.array(params['pi'])  # (n_dom, A)
    S_exch = params['S_exch']

    sub_matrices = np.zeros((n_dom, 20, 20))
    for d in range(n_dom):
        if S_exch.ndim == 3:
            Q_d = build_rate_matrix(S_exch[d], params['pi'][d])
        else:
            Q_d = build_rate_matrix(S_exch, params['pi'][d])
        P_t = np.array(transition_matrix(Q_d, t))
        sub_matrices[d] = P_t

    log_chi = np.log(np.maximum(chi_np, 1e-300))

    return log_chi, st, sub_matrices, pis


def run_composite_beam_holdout(family_id, held_out, seq_names, aln_int,
                                ungapped, tree, params, n_dom, n_frag,
                                singlet_log_trans, singlet_log_pi,
                                singlet_log_end, Q, pi, verbose=True):
    """Run composite beam + Felsenstein on one held-out leaf."""
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
        "n_descendants": len(remaining),
    }

    # --- Felsenstein baseline ---
    t0 = time.time()
    try:
        fels_seq = method_felsenstein(pruned_tree, pruned_msa, Q, pi)
        fels_time = time.time() - t0
        if len(fels_seq) > 0 and len(true_seq) > 0:
            fels_id, fels_aligned, fels_matches = needleman_wunsch_identity(
                fels_seq, true_seq)
        else:
            fels_id, fels_aligned, fels_matches = 0.0, 0, 0
        result["identity_felsenstein"] = float(fels_id)
        result["time_felsenstein"] = float(fels_time)
        result["len_felsenstein"] = len(fels_seq)
    except Exception as e:
        result["identity_felsenstein"] = None
        result["time_felsenstein"] = None
        result["status_felsenstein"] = f"error: {str(e)[:200]}"
        if verbose:
            print(f"      Felsenstein error: {e}")

    # --- Composite beam ---
    t0 = time.time()
    try:
        desc_seqs = []
        distances = []
        log_chi_list = []
        state_types_list = []
        sub_matrices_list = []
        pis_list = []

        for desc_name in remaining:
            desc_seq = np.array([c for c in pruned_msa[desc_name] if c >= 0],
                                dtype=np.int32)
            if len(desc_seq) == 0:
                continue

            # Distance: tree path from held_out to this descendant
            dist = tree_path_distance(tree, held_out, desc_name)
            if dist is None or dist < 1e-6:
                dist = 0.1  # fallback

            desc_seqs.append(desc_seq)
            distances.append(dist)

            # Build pair HMM for this distance
            log_chi, st, sub_mats, pis_k = build_per_desc_pair_hmm(
                params, n_dom, n_frag, dist)
            log_chi_list.append(log_chi)
            state_types_list.append(st)
            sub_matrices_list.append(sub_mats)
            pis_list.append(pis_k)

        if len(desc_seqs) < 2:
            result["identity_beam"] = None
            result["time_beam"] = None
            result["status_beam"] = "too_few_descendants"
            return result

        # Limit to closest descendants for tractability
        if len(desc_seqs) > MAX_DESC_FOR_BEAM:
            order = np.argsort(distances)[:MAX_DESC_FOR_BEAM]
            desc_seqs = [desc_seqs[i] for i in order]
            distances = [distances[i] for i in order]
            log_chi_list = [log_chi_list[i] for i in order]
            state_types_list = [state_types_list[i] for i in order]
            sub_matrices_list = [sub_matrices_list[i] for i in order]
            pis_list = [pis_list[i] for i in order]

        result["n_desc_used"] = len(desc_seqs)

        # Run composite beam
        anc_seq, log_score = composite_beam_reconstruct(
            desc_seqs=desc_seqs,
            distances=distances,
            log_chi_list=log_chi_list,
            state_types_list=state_types_list,
            sub_matrices_list=sub_matrices_list,
            pis_list=pis_list,
            n_dom=n_dom,
            n_frag=n_frag,
            singlet_log_trans=singlet_log_trans,
            singlet_log_pi=singlet_log_pi,
            singlet_log_end=singlet_log_end,
            beam_width=BEAM_WIDTH,
            max_len=min(int(max(len(s) for s in desc_seqs) * 1.3), 200),
        )
        beam_time = time.time() - t0

        if len(anc_seq) > 0 and len(true_seq) > 0:
            beam_id, beam_aligned, beam_matches = needleman_wunsch_identity(
                anc_seq, true_seq)
        else:
            beam_id, beam_aligned, beam_matches = 0.0, 0, 0

        result["identity_beam"] = float(beam_id)
        result["time_beam"] = float(beam_time)
        result["len_beam"] = len(anc_seq)
        result["log_score_beam"] = float(log_score)
        result["status_beam"] = "ok"

    except Exception as e:
        beam_time = time.time() - t0
        result["identity_beam"] = None
        result["time_beam"] = float(beam_time)
        result["status_beam"] = f"error: {str(e)[:200]}"
        if verbose:
            print(f"      Beam error: {e}")
            traceback.print_exc()

    # Print summary
    if verbose:
        beam_id_str = (f"{result.get('identity_beam', 0):.3f}"
                       if result.get('identity_beam') is not None else "ERR")
        fels_id_str = (f"{result.get('identity_felsenstein', 0):.3f}"
                       if result.get('identity_felsenstein') is not None else "ERR")
        beam_t = result.get('time_beam', 0) or 0
        fels_t = result.get('time_felsenstein', 0) or 0
        print(f"    {held_out} (tau={branch_len:.3f}): "
              f"beam={beam_id_str}({beam_t:.1f}s) "
              f"fels={fels_id_str}({fels_t:.1f}s)")

    return result


def main():
    print("=" * 70)
    print("Composite Beam MixDom Reconstruction Benchmark")
    print("=" * 70)

    # Load substitution model
    Q, pi = rate_matrix_lg()
    Q = jnp.array(Q)
    pi = jnp.array(pi)

    # Load MixDom model
    print(f"\nLoading model from {PARAMS_PATH}")
    params, n_dom, n_classes = load_params(PARAMS_PATH)
    n_frag = 2  # d3f2 model
    print(f"  n_dom={n_dom}, n_frag={n_frag}, n_classes={n_classes}")

    # Distill singlet model at reference distance
    print(f"  Distilling singlet at t={SINGLET_T}...")
    precomp = precompute_mixdom(params, n_classes)
    dist = distill_mixdom(params, SINGLET_T, n_classes, precomp)
    wfst = normalize_freqs_wfst(dist)

    singlet_log_trans = np.array(jnp.log(jnp.maximum(wfst['singlet_trans'], 1e-300)))
    singlet_log_pi = np.array(jnp.log(jnp.maximum(wfst['singlet_start'], 1e-300)))
    singlet_log_end = np.array(jnp.log(jnp.maximum(wfst['singlet_end'], 1e-300)))

    print(f"  Singlet model shapes: trans={singlet_log_trans.shape}, "
          f"pi={singlet_log_pi.shape}, end={singlet_log_end.shape}")

    # Select families
    print(f"\nSelecting Pfam test families ({MIN_SEQS}-{MAX_SEQS} seqs, "
          f"{MIN_COLS}-{MAX_COLS} cols)...")
    families = select_pfam_families_filtered(MAX_FAMILIES)
    print(f"  Selected {len(families)} families")

    all_results = []
    total_families = 0

    for fi, fam in enumerate(families):
        sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")
        aligned_seqs = parse_sto(sto_path)
        seq_names = list(aligned_seqs.keys())
        aln_len = len(next(iter(aligned_seqs.values())))

        print(f"\n[{fi+1}/{len(families)}] Family {fam}: "
              f"{len(seq_names)} seqs, {aln_len} cols")

        # Build integer MSA
        aln_int = {}
        ungapped = {}
        for name in seq_names:
            arr = np.array([AA_TO_INT.get(c, -1) for c in aligned_seqs[name]],
                           dtype=np.int32)
            aln_int[name] = arr
            ungapped[name] = np.array([c for c in arr if c >= 0], dtype=np.int32)

        # Build NJ tree from pairwise distances
        names_d, D_mat = msa_pairwise_distances(aligned_seqs, Q, pi)
        tree = neighbor_joining(D_mat, names_d)
        name_internal_nodes(tree)

        # Run holdouts
        holdouts_done = 0
        for held_out in seq_names:
            if holdouts_done >= MAX_HOLDOUTS:
                break
            if len(ungapped[held_out]) == 0:
                continue

            r = run_composite_beam_holdout(
                fam, held_out, seq_names, aln_int, ungapped, tree,
                params, n_dom, n_frag,
                singlet_log_trans, singlet_log_pi, singlet_log_end,
                Q, pi, verbose=True)
            if r is not None:
                all_results.append(r)
                holdouts_done += 1

        total_families += 1

        # Intermediate save
        if total_families % 5 == 0 or fi == len(families) - 1:
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  [Saved {len(all_results)} results to {OUTPUT_PATH}]")

    # Final save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    beam_ids = [r['identity_beam'] for r in all_results
                if r.get('identity_beam') is not None]
    fels_ids = [r['identity_felsenstein'] for r in all_results
                if r.get('identity_felsenstein') is not None]
    beam_times = [r['time_beam'] for r in all_results
                  if r.get('time_beam') is not None
                  and r.get('identity_beam') is not None]
    fels_times = [r['time_felsenstein'] for r in all_results
                  if r.get('time_felsenstein') is not None]

    # Paired comparison (only where both succeeded)
    paired_beam = []
    paired_fels = []
    for r in all_results:
        if (r.get('identity_beam') is not None and
            r.get('identity_felsenstein') is not None):
            paired_beam.append(r['identity_beam'])
            paired_fels.append(r['identity_felsenstein'])

    print(f"\nTotal holdouts: {len(all_results)}")
    print(f"Beam successes: {len(beam_ids)}")
    print(f"Felsenstein successes: {len(fels_ids)}")

    if beam_ids:
        print(f"\nBeam identity:       mean={np.mean(beam_ids):.4f} "
              f"median={np.median(beam_ids):.4f} "
              f"std={np.std(beam_ids):.4f}")
    if fels_ids:
        print(f"Felsenstein identity: mean={np.mean(fels_ids):.4f} "
              f"median={np.median(fels_ids):.4f} "
              f"std={np.std(fels_ids):.4f}")

    if paired_beam:
        wins = sum(1 for b, f in zip(paired_beam, paired_fels) if b > f)
        ties = sum(1 for b, f in zip(paired_beam, paired_fels)
                   if abs(b - f) < 0.001)
        losses = sum(1 for b, f in zip(paired_beam, paired_fels) if b < f)
        diff = np.mean(np.array(paired_beam) - np.array(paired_fels))
        print(f"\nPaired comparison (n={len(paired_beam)}):")
        print(f"  Beam wins: {wins}  Ties: {ties}  Fels wins: {losses}")
        print(f"  Mean difference (beam - fels): {diff:+.4f}")

    if beam_times:
        print(f"\nBeam time:  mean={np.mean(beam_times):.1f}s "
              f"median={np.median(beam_times):.1f}s")
    if fels_times:
        print(f"Fels time:  mean={np.mean(fels_times):.1f}s "
              f"median={np.median(fels_times):.1f}s")

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
