#!/usr/bin/env python3
"""Held-out leaf prediction benchmark for SVI-BW MixDom models.

For each Pfam val family:
1. Remove one leaf from the MSA
2. Build NJ tree from remaining sequences
3. Felsenstein pruning with per-domain P(t) from MixDom model
4. Fitch parsimony for gap prediction
5. Report per-column accuracy including gaps

Also runs plain LG08 Felsenstein baseline for comparison.

Usage:
    cd python && JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 uv run python \
        experiments/svi_bw_leaf_prediction.py \
        --model-path pfam/svi_bw_d3f1_best_val.npz \
        --out experiments/d3f1_leaf_prediction.json \
        --n-families 200
"""

import os
import sys
import json
import time
import argparse
import traceback
import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from tkfmixdom.jax.distill.maraschino import load_params, build_rate_matrix
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.tree.guide_tree import neighbor_joining
from tkfmixdom.jax.tree.tree_varanc import infer_internal_presence, name_internal_nodes
from tkfmixdom.jax.util.io import AA_TO_INT, TreeNode

from experiments.ancrec_benchmark import (
    parse_sto, msa_pairwise_distances, PFAM_DIR, SPLITS_PATH,
    remove_leaf,
)

t0_global = time.time()
def log(msg): print(f'[{time.time()-t0_global:.0f}s] {msg}', flush=True)

AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"


def felsenstein_column(tree, col_chars, sub_matrices_by_id, pi):
    """Felsenstein pruning for a single MSA column.

    Returns MAP character index, or -1 if all gaps.
    """
    A = len(pi)

    def _prune(node):
        if node.is_leaf:
            c = col_chars.get(node.name, -1)
            if c is None or c < 0:
                return np.ones(A)
            obs = np.zeros(A)
            if 0 <= c < A:
                obs[c] = 1.0
            elif c == 20:  # wildcard X
                obs[:] = 1.0
            else:
                obs[:] = 1.0
            return obs

        partial = np.ones(A)
        for child in node.children:
            child_cond = _prune(child)
            P = sub_matrices_by_id.get(id(child))
            if P is None:
                return np.ones(A)  # fallback
            partial *= (P @ child_cond)

        mx = np.max(partial)
        if mx > 0:
            partial /= mx
        return partial

    root_cond = _prune(tree)
    joint = pi * root_cond
    s = np.sum(joint)
    if s <= 0:
        return -1
    return int(np.argmax(joint))


def mixdom_felsenstein_column(tree, col_chars, domain_sub_matrices_by_id,
                               dom_weights, pis_per_domain):
    """MixDom Felsenstein: mixture over domains for a single column.

    For each domain d, runs Felsenstein pruning with domain-specific P_d(t).
    Marginalizes over domains using dom_weights.

    Returns MAP character index, or -1.
    """
    n_dom = len(dom_weights)
    A = pis_per_domain.shape[1]

    posterior = np.zeros(A)
    for d in range(n_dom):
        pi_d = pis_per_domain[d]

        def _prune_d(node):
            if node.is_leaf:
                c = col_chars.get(node.name, -1)
                if c is None or c < 0:
                    return np.ones(A)
                obs = np.zeros(A)
                if 0 <= c < A:
                    obs[c] = 1.0
                elif c == 20:  # wildcard X
                    obs[:] = 1.0
                else:
                    obs[:] = 1.0
                return obs

            partial = np.ones(A)
            for child in node.children:
                child_cond = _prune_d(child)
                P = domain_sub_matrices_by_id[d].get(id(child))
                if P is None:
                    return np.ones(A)
                partial *= (P @ child_cond)

            mx = np.max(partial)
            if mx > 0:
                partial /= mx
            return partial

        root_cond = _prune_d(tree)
        joint_d = pi_d * root_cond
        posterior += dom_weights[d] * joint_d

    s = np.sum(posterior)
    if s <= 0:
        return -1
    return int(np.argmax(posterior))


def fitch_presence(tree, leaf_presence):
    """Fitch parsimony for ancestral presence/absence at each column.

    Returns dict of {node_name: bool_array} for internal nodes.
    """
    return infer_internal_presence(tree, leaf_presence)


def predict_leaf(tree, msa_int, held_out, true_seq_aligned,
                 Q_lg, pi_lg, domain_info=None):
    """Predict a held-out leaf's sequence using Felsenstein + Fitch.

    Args:
        tree: TreeNode (with held-out leaf removed)
        msa_int: dict {name: int_array} for remaining leaves (with gaps=-1)
        held_out: name of held-out leaf
        true_seq_aligned: int_array of true held-out sequence (aligned, with gaps=-1)
        Q_lg, pi_lg: LG08 rate matrix and equilibrium frequencies
        domain_info: if not None, dict with keys:
            'dom_weights', 'pis', 'S_exch', 'n_dom'
            for MixDom per-domain Felsenstein

    Returns:
        dict with LG08 and optionally MixDom per-column accuracy
    """
    leaves = [n.name for n in tree.leaves()]
    C = len(true_seq_aligned)

    # Leaf presence for Fitch
    leaf_pres = {}
    for name in leaves:
        leaf_pres[name] = np.array(msa_int[name] >= 0, dtype=bool)

    # Fitch presence at root
    root_pres = fitch_presence(tree, leaf_pres)
    root_name = tree.name
    rp = root_pres.get(root_name, np.ones(C, dtype=bool))

    # -- LG08 Felsenstein baseline --
    # Precompute sub matrices
    sub_lg = {}
    for node in tree.preorder():
        if node.parent is not None:
            t = max(node.branch_length, 1e-6)
            sub_lg[id(node)] = np.asarray(transition_matrix(Q_lg, t))

    pi_lg_np = np.asarray(pi_lg)

    correct_lg = 0
    total_lg = 0

    for col in range(C):
        true_c = true_seq_aligned[col]
        is_present = rp[col] if col < len(rp) else False

        if true_c >= 0 and is_present:
            # Predict character
            col_chars = {name: int(msa_int[name][col]) for name in leaves}
            pred = felsenstein_column(tree, col_chars, sub_lg, pi_lg_np)
            if pred >= 0 and pred == true_c:
                correct_lg += 1
            total_lg += 1
        elif true_c < 0 and not is_present:
            correct_lg += 1
            total_lg += 1
        else:
            # Mismatch: predicted gap but true has char, or vice versa
            total_lg += 1

    acc_lg = correct_lg / max(total_lg, 1)

    result = {
        'lg08_correct': correct_lg,
        'lg08_total': total_lg,
        'lg08_accuracy': float(acc_lg),
    }

    # -- MixDom Felsenstein --
    if domain_info is not None:
        n_dom = domain_info['n_dom']
        dom_weights = np.asarray(domain_info['dom_weights'])
        pis = np.asarray(domain_info['pis'])  # (N, A)
        S_exch = np.asarray(domain_info['S_exch'])  # (N, A, A)

        # Precompute per-domain sub matrices
        domain_sub = [{} for _ in range(n_dom)]
        for node in tree.preorder():
            if node.parent is not None:
                t = max(node.branch_length, 1e-6)
                for d in range(n_dom):
                    Q_d = np.asarray(build_rate_matrix(jnp.array(S_exch[d]),
                                                        jnp.array(pis[d])))
                    P_d = np.asarray(transition_matrix(jnp.array(Q_d), t))
                    domain_sub[d][id(node)] = P_d

        correct_mix = 0
        total_mix = 0

        for col in range(C):
            true_c = true_seq_aligned[col]
            is_present = rp[col] if col < len(rp) else False

            if true_c >= 0 and is_present:
                col_chars = {name: int(msa_int[name][col]) for name in leaves}
                pred = mixdom_felsenstein_column(
                    tree, col_chars, domain_sub, dom_weights, pis)
                if pred >= 0 and pred == true_c:
                    correct_mix += 1
                total_mix += 1
            elif true_c < 0 and not is_present:
                correct_mix += 1
                total_mix += 1
            else:
                total_mix += 1

        acc_mix = correct_mix / max(total_mix, 1)
        result['mixdom_correct'] = correct_mix
        result['mixdom_total'] = total_mix
        result['mixdom_accuracy'] = float(acc_mix)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Held-out leaf prediction benchmark')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to SVI-BW model .npz')
    parser.add_argument('--out', type=str, required=True,
                        help='Output JSON path')
    parser.add_argument('--n-families', type=int, default=200,
                        help='Number of val families to process')
    parser.add_argument('--n-holdouts', type=int, default=1,
                        help='Number of holdout leaves per family')
    parser.add_argument('--min-seqs', type=int, default=5)
    parser.add_argument('--max-seqs', type=int, default=30)
    parser.add_argument('--min-cols', type=int, default=30)
    parser.add_argument('--max-cols', type=int, default=300)
    args = parser.parse_args()

    # Load model
    log(f'Loading model from {args.model_path}')
    params, n_dom, n_cls = load_params(args.model_path)
    S_exch = np.asarray(params['S_exch'])
    pis = np.asarray(params['pi'])
    v = np.asarray(params['v'])
    log(f'  n_dom={n_dom}, dom_weights={v}')

    domain_info = {
        'n_dom': n_dom,
        'dom_weights': v,
        'pis': pis,
        'S_exch': S_exch,
    }

    # LG08 baseline
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg_np, pi_lg_np = np.asarray(Q_lg), np.asarray(pi_lg)

    # Load val split
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    val_fams = splits['val']
    log(f'{len(val_fams)} val families, processing up to {args.n_families}')

    results = []
    n_processed = 0

    for fam_idx, fam in enumerate(val_fams):
        if n_processed >= args.n_families:
            break

        sto_path = os.path.join(PFAM_DIR, f'{fam}.sto')
        if not os.path.exists(sto_path):
            continue

        try:
            seqs = parse_sto(sto_path)
        except Exception:
            continue

        n = len(seqs)
        if not (args.min_seqs <= n <= args.max_seqs):
            continue

        C = len(next(iter(seqs.values())))
        if not (args.min_cols <= C <= args.max_cols):
            continue

        # Build NJ tree from MSA
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
                elif ch.upper() == 'X':
                    seq[j] = 20
            msa[name] = seq

        leaves = [l.name for l in tree.leaves()]
        holdouts = leaves[:args.n_holdouts]

        for held_out in holdouts:
            true_aligned = msa[held_out]

            # Remove leaf from tree
            pruned_tree, parent_name, bl = remove_leaf(tree, held_out)
            if pruned_tree is None:
                continue
            name_internal_nodes(pruned_tree)

            # Remaining MSA
            remaining_msa = {name: msa[name] for name in leaves if name != held_out}

            try:
                pred = predict_leaf(
                    pruned_tree, remaining_msa, held_out, true_aligned,
                    Q_lg_np, pi_lg_np, domain_info=domain_info)

                pred['family'] = fam
                pred['held_out'] = held_out
                pred['n_seqs'] = n
                pred['n_cols'] = C
                pred['branch_length'] = float(bl) if bl else 0.0
                results.append(pred)

                log(f'[{n_processed+1}/{args.n_families}] {fam}/{held_out[:15]}: '
                    f'LG08={pred["lg08_accuracy"]:.1%} '
                    f'MixDom={pred.get("mixdom_accuracy", 0):.1%} '
                    f'(cols={C}, seqs={n})')
            except Exception as e:
                log(f'[{n_processed+1}] {fam} ERROR: {e}')
                traceback.print_exc()
                continue

        n_processed += 1

        # Save periodically
        if n_processed % 50 == 0:
            _save_results(args.out, results, args, n_dom)

    _save_results(args.out, results, args, n_dom)
    _print_summary(results)


def _save_results(out_path, results, args, n_dom):
    if not results:
        return
    output = {
        'benchmark': 'leaf_prediction',
        'model_path': args.model_path,
        'n_dom': n_dom,
        'n_families_processed': len(set(r['family'] for r in results)),
        'n_holdouts': len(results),
        'results': results,
    }
    # Compute summaries
    lg_accs = [r['lg08_accuracy'] for r in results]
    output['lg08_summary'] = {
        'mean_accuracy': float(np.mean(lg_accs)),
        'median_accuracy': float(np.median(lg_accs)),
    }
    mix_accs = [r['mixdom_accuracy'] for r in results if 'mixdom_accuracy' in r]
    if mix_accs:
        output['mixdom_summary'] = {
            'mean_accuracy': float(np.mean(mix_accs)),
            'median_accuracy': float(np.median(mix_accs)),
        }

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    log(f'Saved {len(results)} results to {out_path}')


def _print_summary(results):
    if not results:
        log('No results.')
        return

    lg_accs = [r['lg08_accuracy'] for r in results]
    mix_accs = [r['mixdom_accuracy'] for r in results if 'mixdom_accuracy' in r]

    log(f'\n=== SUMMARY ({len(results)} holdouts) ===')
    log(f'LG08 Felsenstein: mean={np.mean(lg_accs):.1%}, median={np.median(lg_accs):.1%}')
    if mix_accs:
        log(f'MixDom Felsenstein: mean={np.mean(mix_accs):.1%}, median={np.median(mix_accs):.1%}')

    # Head-to-head
    paired = [(r['lg08_accuracy'], r['mixdom_accuracy'])
              for r in results if 'mixdom_accuracy' in r]
    if paired:
        mix_wins = sum(1 for l, m in paired if m > l + 0.001)
        lg_wins = sum(1 for l, m in paired if l > m + 0.001)
        ties = len(paired) - mix_wins - lg_wins
        log(f'MixDom wins: {mix_wins}, LG08 wins: {lg_wins}, ties: {ties}')


if __name__ == '__main__':
    main()
