#!/usr/bin/env python3
"""Gamma-annotated reconstruction on real Pfam protein data.

Tests whether per-column gamma rate classes improve ancestral reconstruction
on real protein MSAs (A=20, LG08 model). Uses same evaluation framework
as pfam_stratified_benchmark.py.

Usage:
    cd python && JAX_PLATFORMS=cpu uv run python experiments/pfam_gamma_recon_test.py
"""

import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, '/home/yam/subby')

import jax
jax.config.update('jax_enable_x64', True)

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import ensure_rate_matrix
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
from tkfmixdom.jax.tree.tree_varanc import (
    tree_varanc, build_tkf91_branch_wfst, build_tkf91_root_wfst,
    tree_varanc_block_diagonal, infer_internal_presence, name_internal_nodes,
)
from tkfmixdom.jax.tree.guide_tree import neighbor_joining
from tkfmixdom.jax.util.io import AA_TO_INT

from experiments.ancrec_benchmark import (
    parse_sto, msa_pairwise_distances, remove_leaf,
    find_parent_of_leaf, needleman_wunsch_identity,
    method_felsenstein, TKF92_INS_RATE, TKF92_DEL_RATE,
    PFAM_DIR, SPLITS_PATH,
)

GAMMA_DIR = "gamma_labels"
A = 20
G = 4


def load_gamma_labels(family_id):
    path = os.path.join(GAMMA_DIR, f"{family_id}.G{G}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_holdout(family_id, held_out, aligned_seqs, ungapped, tree, Q, pi,
                gamma_labels, gamma_rates, verbose=True):
    """Run Felsenstein, TVA, TVA+gamma on one held-out leaf."""
    ins_rate, del_rate = TKF92_INS_RATE, TKF92_DEL_RATE

    # Get branch length of held-out
    _, branch_len = find_parent_of_leaf(tree, held_out)
    if branch_len is None:
        return None

    # Prune tree
    pruned_tree, _, _ = remove_leaf(tree, held_out)
    if pruned_tree is None:
        return None

    pruned_leaves = set(n.name for n in pruned_tree.leaves())
    remaining = [n for n in aligned_seqs if n != held_out and n in pruned_leaves]
    if len(remaining) < 3:
        return None

    # Build pruned MSA (integer, -1 for gaps)
    C = len(next(iter(aligned_seqs.values())))
    pruned_msa = {}
    for n in remaining:
        seq = np.full(C, -1, dtype=np.int32)
        for j, ch in enumerate(aligned_seqs[n]):
            if ch in AA_TO_INT:
                seq[j] = AA_TO_INT[ch]
        pruned_msa[n] = seq

    true_seq = ungapped[held_out]

    result = {
        'family': family_id, 'held_out': held_out,
        'tau': float(branch_len), 'true_len': len(true_seq),
    }

    # --- Compute all predictions, then evaluate at matched columns ---

    # Felsenstein posteriors (all columns)
    fels_pred = None
    try:
        _, fels_post = marginal_ancestor_all_columns_jax(pruned_tree, pruned_msa, Q, pi)
        fels_pred = np.argmax(np.asarray(fels_post), axis=1)  # (C,)
    except Exception as e:
        if verbose:
            print(f"    FELS ERROR: {e}")

    # Felsenstein + gamma posteriors (grouped by gamma class)
    fels_g_pred = None
    try:
        fels_g_pred = np.full(C, -1, dtype=np.int32)
        col_groups = {}
        for c in range(C):
            has_data = any(pruned_msa[n][c] >= 0 for n in pruned_msa)
            if not has_data:
                continue
            g = gamma_labels[c] if c < len(gamma_labels) and gamma_labels[c] >= 0 else -1
            col_groups.setdefault(g, []).append(c)

        for g, cols in col_groups.items():
            if g >= 0:
                Q_g = gamma_rates[g] * np.array(Q)
                Q_g = np.array(ensure_rate_matrix(Q_g))
            else:
                Q_g = Q
            msa_g = {n: np.array([pruned_msa[n][c] for c in cols], dtype=np.int32)
                     for n in pruned_msa}
            _, post_g = marginal_ancestor_all_columns_jax(pruned_tree, msa_g, Q_g, pi)
            post_g = np.asarray(post_g)
            for i, c in enumerate(cols):
                fels_g_pred[c] = int(np.argmax(post_g[i]))
    except Exception as e:
        fels_g_pred = None
        if verbose:
            print(f"    FELS+γ ERROR: {e}")

    # TVA posteriors
    tva_pred_at_col = None  # (C,) with -1 for non-present columns
    name_internal_nodes(pruned_tree)
    leaf_presence = {n: np.array(pruned_msa[n] >= 0, dtype=bool) for n in pruned_msa}
    msa_presence = infer_internal_presence(pruned_tree, leaf_presence)
    root_name = pruned_tree.name
    root_pres = msa_presence.get(root_name, np.zeros(C, dtype=bool))

    try:
        wfst_plain = {}
        for node in pruned_tree.preorder():
            if node.is_root: continue
            wfst_plain[(node.parent.name, node.name)] = build_tkf91_branch_wfst(
                ins_rate, del_rate, Q, pi, node.branch_length)
        singlet_plain = build_tkf91_root_wfst(ins_rate, del_rate, pi)

        node_post, _, _, _, _ = tree_varanc(
            pruned_tree, msa_presence, pruned_msa,
            wfst_plain, singlet_plain, pi, n_iter=1, verbose=False)

        root_post = node_post.get(root_name)
        if root_post is not None and len(root_post) > 0:
            tva_argmax = np.argmax(root_post, axis=1).astype(np.int32)
            # Map ungapped positions back to MSA columns
            tva_pred_at_col = np.full(C, -1, dtype=np.int32)
            ungap_idx = 0
            for c in range(C):
                if root_pres[c]:
                    if ungap_idx < len(tva_argmax):
                        tva_pred_at_col[c] = tva_argmax[ungap_idx]
                    ungap_idx += 1
    except Exception as e:
        if verbose:
            print(f"    TVA ERROR: {e}")

    # TVA + gamma posteriors
    tva_g_pred_at_col = None
    try:
        wfst_per_g = []
        sing_per_g = []
        pi_per_g = []
        for g in range(G):
            Q_g = gamma_rates[g] * np.array(Q)
            Q_g = np.array(ensure_rate_matrix(Q_g))
            wg = {}
            for node in pruned_tree.preorder():
                if node.is_root: continue
                wg[(node.parent.name, node.name)] = build_tkf91_branch_wfst(
                    ins_rate, del_rate, Q_g, pi, node.branch_length)
            wfst_per_g.append(wg)
            sing_per_g.append(build_tkf91_root_wfst(ins_rate, del_rate, pi))
            pi_per_g.append(pi)

        gamma_mask = np.zeros((C, G), dtype=np.float64)
        for c in range(C):
            if c < len(gamma_labels) and gamma_labels[c] >= 0:
                gamma_mask[c, gamma_labels[c]] = 1.0
            else:
                gamma_mask[c, :] = 1.0 / G

        tva_g_post, _, _, _, _ = tree_varanc_block_diagonal(
            pruned_tree, msa_presence, pruned_msa,
            wfst_per_g, sing_per_g, pi_per_g,
            D=1, A=A, rate_multiplier_mask=gamma_mask,
            n_iter=1, verbose=False)

        root_post_g = tva_g_post.get(root_name)
        if root_post_g is not None and len(root_post_g) > 0:
            tva_g_argmax = np.argmax(root_post_g, axis=1).astype(np.int32)
            tva_g_pred_at_col = np.full(C, -1, dtype=np.int32)
            ungap_idx = 0
            for c in range(C):
                if root_pres[c]:
                    if ungap_idx < len(tva_g_argmax):
                        tva_g_pred_at_col[c] = tva_g_argmax[ungap_idx]
                    ungap_idx += 1
    except Exception as e:
        if verbose:
            print(f"    TVA+γ ERROR: {e}")

    # --- Evaluate all methods at matched columns ---
    # Eval columns: where the held-out has a residue AND the TVA root is present.
    # This ensures ALL methods are evaluated at the SAME columns — no inflation
    # from TVA skipping uninformative columns.
    held_presence = np.array([aligned_seqs[held_out][c] in 'ACDEFGHIKLMNPQRSTVWY'
                              for c in range(C)], dtype=bool)
    eval_cols = np.where(held_presence & root_pres)[0]

    # Map held-out's ungapped indices
    held_ungap = np.cumsum(held_presence) - 1

    if len(eval_cols) < 5:
        return None

    def _accuracy(pred_at_col):
        """Compute accuracy at eval_cols. pred_at_col is (C,) or None."""
        if pred_at_col is None:
            return None
        n_correct = 0
        n_total = 0
        for c in eval_cols:
            h_idx = int(held_ungap[c])
            if h_idx < len(true_seq) and pred_at_col[c] >= 0:
                n_total += 1
                if pred_at_col[c] == true_seq[h_idx]:
                    n_correct += 1
        return n_correct / max(n_total, 1) if n_total >= 5 else None

    result['fels'] = _accuracy(fels_pred)
    result['fels_gamma'] = _accuracy(fels_g_pred)
    result['tva'] = _accuracy(tva_pred_at_col)
    result['tva_gamma'] = _accuracy(tva_g_pred_at_col)
    result['n_eval'] = int(len(eval_cols))

    if verbose:
        def _fmt(key, label):
            v = result.get(key)
            return f"{label}={v:.3f}" if v is not None else f"{label}=ERR"
        parts = [_fmt('fels','F'), _fmt('fels_gamma','F+γ'),
                 _fmt('tva','T'), _fmt('tva_gamma','T+γ')]
        print(f"    {held_out} (tau={branch_len:.3f}): {' '.join(parts)}")

    return result


def main():
    Q_lg, pi_lg = rate_matrix_lg()
    Q = np.asarray(Q_lg)
    pi = np.asarray(pi_lg)

    # Load test families
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    test_fams = splits['test']

    # Filter: has gamma labels, moderate size (5-50 seqs, 50-500 cols)
    candidates = []
    for fam in test_fams:
        gamma_data = load_gamma_labels(fam)
        if gamma_data is None:
            continue
        sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")
        if not os.path.exists(sto_path):
            continue
        aligned_seqs = parse_sto(sto_path)
        n_seqs = len(aligned_seqs)
        n_cols = gamma_data['n_cols']
        if 5 <= n_seqs <= 50 and 50 <= n_cols <= 300:
            candidates.append((fam, n_seqs, n_cols))

    print(f"Candidate families: {len(candidates)}")

    # Sample 30 families
    N = 30
    rng = np.random.RandomState(42)
    rng.shuffle(candidates)
    families = candidates[:N]

    all_results = []
    t0 = time.time()

    for i, (fam, n_seqs, n_cols) in enumerate(families):
        print(f"\n[{i+1}/{N}] {fam}: {n_seqs} seqs, {n_cols} cols", flush=True)

        # Load alignment
        sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")
        aligned_seqs = parse_sto(sto_path)
        names = list(aligned_seqs.keys())
        C = len(next(iter(aligned_seqs.values())))

        # Ungapped sequences
        ungapped = {}
        for name in names:
            seq = [AA_TO_INT[ch] for ch in aligned_seqs[name] if ch in AA_TO_INT]
            ungapped[name] = np.array(seq, dtype=np.int32)

        # Build NJ tree
        try:
            nj_names, dist = msa_pairwise_distances(aligned_seqs, Q, pi)
            tree = neighbor_joining(dist, nj_names)
        except Exception as e:
            print(f"  SKIP: tree error: {e}")
            continue

        # Load gamma labels
        gamma_data = load_gamma_labels(fam)
        gamma_labels = gamma_data['labels']
        gamma_rates = np.array(gamma_data['rates'])

        # Pick holdouts
        holdout_names = list(names)
        rng.shuffle(holdout_names)
        holdout_names = holdout_names[:3]

        for held_out in holdout_names:
            if len(ungapped.get(held_out, [])) < 10:
                continue
            r = run_holdout(fam, held_out, aligned_seqs, ungapped, tree,
                          Q, pi, gamma_labels, gamma_rates, verbose=True)
            if r is not None:
                all_results.append(r)

        # Interim report every 10 families
        if (i + 1) % 10 == 0 and all_results:
            ok = [r for r in all_results if r.get('fels') is not None and r.get('tva_gamma') is not None]
            if ok:
                elapsed = time.time() - t0
                fels_mean = np.mean([r['fels'] for r in ok])
                fg_vals = [r['fels_gamma'] for r in ok if r.get('fels_gamma') is not None]
                tva_mean = np.mean([r['tva'] for r in ok if r.get('tva') is not None])
                g_mean = np.mean([r['tva_gamma'] for r in ok])
                print(f"\n--- Interim ({len(ok)} holdouts, {elapsed:.0f}s) ---")
                print(f"  Fels:    {fels_mean:.3f}")
                if fg_vals:
                    print(f"  Fels+γ:  {np.mean(fg_vals):.3f}")
                print(f"  TVA:     {tva_mean:.3f}")
                print(f"  TVA+γ:   {g_mean:.3f}")

    # Final
    if all_results:
        ok = [r for r in all_results
              if r.get('fels') is not None and r.get('tva') is not None
              and r.get('tva_gamma') is not None]
        if ok:
            fels = [r['fels'] for r in ok]
            fg = [r['fels_gamma'] for r in ok if r.get('fels_gamma') is not None]
            tva = [r['tva'] for r in ok]
            gamma = [r['tva_gamma'] for r in ok]
            n = len(ok)
            print(f"\n{'='*50}")
            print(f"FINAL ({n} holdouts, {len(families)} families)")
            print(f"{'='*50}")
            print(f"  Fels:    {np.mean(fels):.3f} ± {np.std(fels)/n**0.5:.3f}")
            if fg:
                print(f"  Fels+γ:  {np.mean(fg):.3f} ± {np.std(fg)/len(fg)**0.5:.3f}  Δ={np.mean(fg)-np.mean(fels):+.4f}")
            print(f"  TVA:     {np.mean(tva):.3f} ± {np.std(tva)/n**0.5:.3f}  Δ={np.mean(tva)-np.mean(fels):+.4f}")
            print(f"  TVA+γ:   {np.mean(gamma):.3f} ± {np.std(gamma)/n**0.5:.3f}  Δ={np.mean(gamma)-np.mean(fels):+.4f}")

            # Sign tests
            gw = sum(1 for f, g in zip(fels, gamma) if g > f)
            gl = sum(1 for f, g in zip(fels, gamma) if g < f)
            gt = n - gw - gl
            print(f"  TVA+γ vs Fels: wins/ties/loses = {gw}/{gt}/{gl}")
            if fg:
                fels_m = [r['fels'] for r in ok if r.get('fels_gamma') is not None]
                fw = sum(1 for f, g in zip(fels_m, fg) if g > f)
                fl = sum(1 for f, g in zip(fels_m, fg) if g < f)
                ft = len(fg) - fw - fl
                print(f"  Fels+γ vs Fels: wins/ties/loses = {fw}/{ft}/{fl}")

    with open('experiments/pfam_gamma_recon_test.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to experiments/pfam_gamma_recon_test.json")


if __name__ == '__main__':
    main()
