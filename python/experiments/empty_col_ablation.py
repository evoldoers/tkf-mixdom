#!/usr/bin/env python3
"""Empty-column ablation experiment on a single xhard family.

Question: do the all-gap columns introduced by xhard's leaf-subsampling
(observed in 113/200 xhard families, max 38.5% per family) materially
affect the per-method F1, or do all the methods correctly handle them
as trivial absence-predictions that are excluded from F1 anyway?

Approach: pick PF12728 (a small xhard family with 8/81 all-gap columns
in the kept submatrix). Run every method twice — once on the unmodified
spec, once with empty columns stripped from MSA + tree-supporting
inputs. Compare per-method F1.

Methods: fitch, varanc-TKF92, fels21, fels40, d3f1 (MixDom1), d3f1-VBEM
(run6 iter4).

Writes JSON output to `experiments/empty_col_ablation_PF12728.json`.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.setrecursionlimit(50000)
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tkfmixdom.jax.tree.varanc_presence import parse_binary_tree
from tkfmixdom.jax.tree.varanc_presence_mixdom import parse_mixdom_params_npz
from tkfmixdom.jax.train.tree_vbem import predict_holdout_mixdom
from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi
from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT
from experiments.ancrec_benchmark import parse_sto
from experiments.varanc_presence_benchmark import (
    build_binary_tree_from_node, fitch_seeded_init,
    _predict_one_holdout_varanc, predict_holdout_fitch,
    f1_pr,
)
from experiments.fels21_reconstruction_benchmark import (
    run_fels21, prune_tree_to_msa, name_internal_nodes,
    _nw_metrics,
)
from experiments.fels40_reconstruction_benchmark import run_fels40
from experiments.varanc_presence_mixdom_benchmark import (
    compute_sub_LL_per_class_per_column,
)


def find_empty_cols(seqs, leaf_names):
    """Indices of columns with no amino acid in any kept leaf."""
    rows = [seqs[n] for n in leaf_names if n in seqs]
    if not rows:
        return []
    L = max(len(r) for r in rows)
    rows = [r.ljust(L, '-') for r in rows]
    empty = []
    for j in range(L):
        col = [r[j].upper() for r in rows]
        if not any(ch in AA_TO_INT and AA_TO_INT[ch] < 20 for ch in col):
            empty.append(j)
    return empty


def strip_columns(seqs, kept_cols):
    """Return new seqs dict with only kept_cols (sorted ascending)."""
    return {n: ''.join(s[j] if j < len(s) else '-' for j in kept_cols)
            for n, s in seqs.items()}


def gt_present_array(seq, n_cols):
    """Convert a Stockholm sequence to a (n_cols,) {0,1} presence array."""
    out = np.zeros(n_cols, dtype=np.int32)
    s_pad = seq.ljust(n_cols, '-')
    for j in range(n_cols):
        ch = s_pad[j].upper()
        if ch in AA_TO_INT and AA_TO_INT[ch] < 20:
            out[j] = 1
    return out


def msa_int_from_seqs(seqs):
    """Convert seqs dict to msa_int dict (numpy int32, gap = -1)."""
    out = {}
    for name, s in seqs.items():
        L = len(s)
        seq = np.full(L, -1, dtype=np.int32)
        for j, ch in enumerate(s):
            if ch in AA_TO_INT:
                idx = AA_TO_INT[ch]
                if idx < 20:
                    seq[j] = idx
        out[name] = seq
    return out


def run_all_methods(seqs, tree_node, held_out, remaining,
                     mixdom_params_d3f1, mixdom_params_vbem,
                     class_Qs, class_pis,
                     tkf_ins, tkf_del, tkf_ext,
                     fels21_Q, fels21_pi, fels40_Q, fels40_pi):
    """Run every method on the given seqs/tree pair. Returns dict of
    per-method F1 + p_present + per-method timing.

    Each method gets a fresh deepcopy of the tree because
    build_binary_tree_from_node mutates in-place via _binarise, which
    breaks fels21/40's downstream prune+name_internal_nodes chain.
    """
    import copy
    leaf_names = sorted(seqs.keys())
    n_cols = max(len(s) for s in seqs.values())
    held_seq = seqs[held_out]
    gt_present = gt_present_array(held_seq, n_cols)

    # Build presence matrix in `leaf_names` order, plus leaf-row map.
    present_arr = np.zeros((len(leaf_names), n_cols), dtype=np.int32)
    for i, n in enumerate(leaf_names):
        present_arr[i] = gt_present_array(seqs[n], n_cols)

    binary_tree = build_binary_tree_from_node(copy.deepcopy(tree_node))
    bt_leaf_to_msa = {i: leaf_names.index(name)
                       for i, name in enumerate(binary_tree.leaf_names)
                       if name in leaf_names}
    bt_present = np.zeros((binary_tree.num_leaves, n_cols), dtype=np.int32)
    for i in range(binary_tree.num_leaves):
        if i in bt_leaf_to_msa:
            bt_present[i] = present_arr[bt_leaf_to_msa[i]]
    holdout_idx_bt = binary_tree.leaf_names.index(held_out)

    msa_int = msa_int_from_seqs(seqs)

    out = {'gt_present': gt_present.tolist(), 'n_cols': int(n_cols)}

    # 1) fitch
    t0 = time.time()
    p_fitch = predict_holdout_fitch(binary_tree, bt_present, holdout_idx_bt)
    out['fitch'] = {
        'p_present': np.asarray(p_fitch).tolist(),
        'time': time.time() - t0,
    }
    f1, prec, rec, *_ = f1_pr(p_fitch, gt_present)
    out['fitch'].update(f1=float(f1), precision=float(prec), recall=float(rec))

    # 2) varanc-TKF92
    t0 = time.time()
    p_var, _ = _predict_one_holdout_varanc(
        binary_tree, bt_present, holdout_idx_bt,
        held_out, leaf_names,
        tkf_ins, tkf_del, tkf_ext,
        n_iter=150, lr=0.05, seed=0)
    out['varanc'] = {
        'p_present': np.asarray(p_var).tolist(),
        'time': time.time() - t0,
    }
    f1, prec, rec, *_ = f1_pr(p_var, gt_present)
    out['varanc'].update(f1=float(f1), precision=float(prec), recall=float(rec))

    # Build true_seq (held-out residues only) for NW scoring.
    true_seq = np.array(
        [AA_TO_INT[c] for c in held_seq.upper()
         if c in AA_TO_INT and AA_TO_INT[c] < 20],
        dtype=np.int32)

    # 3) fels21
    t0 = time.time()
    pred_seq, _, logp_target, logp_true = run_fels21(
        copy.deepcopy(tree_node), held_out, remaining, msa_int,
        fels21_Q, fels21_pi, held_out_seq=msa_int.get(held_out))
    nw = _nw_metrics(pred_seq, true_seq)
    out['fels21'] = {
        'pred_seq': [int(x) for x in pred_seq],
        'time': time.time() - t0,
        'nw_f1': float(nw['nw_f1']),
        'nw_accuracy': float(nw['nw_accuracy']),
        'logp_target': float(logp_target),
        'logp_true': float(logp_true),
    }

    # 4) fels40
    t0 = time.time()
    pred_seq, _, logp_target, logp_true = run_fels40(
        copy.deepcopy(tree_node), held_out, remaining, msa_int,
        fels40_Q, fels40_pi, held_out_seq=msa_int.get(held_out))
    nw = _nw_metrics(pred_seq, true_seq)
    out['fels40'] = {
        'pred_seq': [int(x) for x in pred_seq],
        'time': time.time() - t0,
        'nw_f1': float(nw['nw_f1']),
        'nw_accuracy': float(nw['nw_accuracy']),
        'logp_target': float(logp_target),
        'logp_true': float(logp_true),
    }

    # 5) d3f1 (MixDom1) via predict_holdout_mixdom
    L_sub_d3f1, _ = compute_sub_LL_per_class_per_column(
        copy.deepcopy(tree_node), msa_int, leaf_names, held_out,
        class_Qs['d3f1'], class_pis['d3f1'])
    seed_e, seed_r = fitch_seeded_init(binary_tree, bt_present,
                                          holdout_idx_bt)
    t0 = time.time()
    p_d3f1 = predict_holdout_mixdom(
        binary_tree, bt_present, holdout_idx_bt, mixdom_params_d3f1,
        L_sub_d3f1, n_iter=150, lr=0.05, seed=0,
        init_edge_logits=np.asarray(seed_e),
        init_root_logit=float(seed_r))
    out['d3f1'] = {
        'p_present': np.asarray(p_d3f1).tolist(),
        'time': time.time() - t0,
    }
    f1, prec, rec, *_ = f1_pr(p_d3f1, gt_present)
    out['d3f1'].update(f1=float(f1), precision=float(prec), recall=float(rec))

    # 6) d3f1-VBEM
    L_sub_vbem, _ = compute_sub_LL_per_class_per_column(
        copy.deepcopy(tree_node), msa_int, leaf_names, held_out,
        class_Qs['vbem'], class_pis['vbem'])
    t0 = time.time()
    p_vbem = predict_holdout_mixdom(
        binary_tree, bt_present, holdout_idx_bt, mixdom_params_vbem,
        L_sub_vbem, n_iter=150, lr=0.05, seed=0,
        init_edge_logits=np.asarray(seed_e),
        init_root_logit=float(seed_r))
    out['d3f1_vbem'] = {
        'p_present': np.asarray(p_vbem).tolist(),
        'time': time.time() - t0,
    }
    f1, prec, rec, *_ = f1_pr(p_vbem, gt_present)
    out['d3f1_vbem'].update(f1=float(f1), precision=float(prec),
                              recall=float(rec))

    return out


def build_class_Q_pi(params):
    n_classes = params['class_pis'].shape[0]
    Qs = np.zeros((n_classes, 20, 20))
    for c in range(n_classes):
        Qs[c] = np.asarray(build_Q_from_S_pi(
            jnp.asarray(params['class_S_exch'][c]),
            jnp.asarray(params['class_pis'][c])))
    return Qs, np.asarray(params['class_pis'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=str, default='PF12728')
    parser.add_argument('--out', type=str,
                        default='experiments/empty_col_ablation_PF12728.json')
    args = parser.parse_args()

    repo_python = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_path = os.path.join(repo_python,
                              'experiments/unified_benchmark_xhard_test_spec.json')
    with open(spec_path) as f:
        spec = json.load(f)
    pfam_dir = os.path.expanduser(spec['pfam_dir'])
    tree_dir = os.path.expanduser(spec['tree_dir'])
    entry = next(e for e in spec['families'] if e['family'] == args.family)
    print(f'family={entry["family"]} held_out={entry["held_out"]} '
          f'K={entry["K"]} n_cols={entry["n_cols"]}')

    # Load tree.
    with open(os.path.join(tree_dir, args.family + '.nwk')) as f:
        tree = parse_newick(f.read().strip())
    spec_leaves = set(entry['remaining']) | {entry['held_out']}
    tree = prune_tree_to_msa(tree, spec_leaves)
    # Skip name_internal_nodes here — fels21/40 deepcopy + re-prune the
    # tree internally, and naming nodes here triggers a cycle through
    # deepcopy that drives preorder to recurse 50k levels deep on the
    # second prune. Each downstream method calls name_internal_nodes
    # on its own deepcopy as needed.

    # Load MSA, restrict to spec leaves.
    seqs_full = parse_sto(os.path.join(pfam_dir, args.family + '.sto'))
    seqs_full = {n: seqs_full[n] for n in spec_leaves if n in seqs_full}

    # Identify empty cols.
    leaf_names = sorted(seqs_full.keys())
    empty_cols = find_empty_cols(seqs_full, leaf_names)
    n_cols_full = max(len(s) for s in seqs_full.values())
    kept_cols = [j for j in range(n_cols_full) if j not in set(empty_cols)]
    print(f'  full n_cols={n_cols_full}, empty_cols={len(empty_cols)}, '
          f'kept_cols={len(kept_cols)}')

    seqs_strip = strip_columns(seqs_full, kept_cols)
    print(f'  stripped n_cols={max(len(s) for s in seqs_strip.values())}')

    # Load TKF92 + per-class params + fels21/40 models.
    with open(os.path.join(repo_python,
                             'experiments/tkf92_fitted_params.json')) as f:
        tkf = json.load(f)
    print(f'  TKF92: ins={tkf["ins_rate"]}, del={tkf["del_rate"]}, '
          f'ext={tkf["ext_rate"]}')

    fels21_data = np.load(os.path.join(repo_python,
                            'pfam/fels21_cherryml.npz'))
    fels21_Q, fels21_pi = fels21_data['Q21'], fels21_data['pi21']

    fels40_data = np.load(os.path.join(repo_python, 'pfam/fels40_em.npz'))
    fels40_Q, fels40_pi = fels40_data['Q40'], fels40_data['pi40']

    mixdom_params_d3f1 = parse_mixdom_params_npz(
        os.path.join(repo_python, 'pfam/svi_bw_d3f1_postfix_best_val.npz'))
    mixdom_params_vbem = parse_mixdom_params_npz(os.path.join(
        repo_python,
        'experiments/tree_svi_vbem_pfam_train_run6/iter0004.npz'))

    class_Qs = {
        'd3f1': build_class_Q_pi(mixdom_params_d3f1)[0],
        'vbem': build_class_Q_pi(mixdom_params_vbem)[0],
    }
    class_pis = {
        'd3f1': build_class_Q_pi(mixdom_params_d3f1)[1],
        'vbem': build_class_Q_pi(mixdom_params_vbem)[1],
    }

    print('\n--- FULL MSA (with empty cols) ---')
    import traceback as tb
    try:
        full = run_all_methods(seqs_full, tree, entry['held_out'],
                                  entry['remaining'],
                                  mixdom_params_d3f1, mixdom_params_vbem,
                                  class_Qs, class_pis,
                                  tkf['ins_rate'], tkf['del_rate'],
                                  tkf['ext_rate'],
                                  fels21_Q, fels21_pi, fels40_Q, fels40_pi)
    except Exception as exc:
        print(f'FAIL on full: {exc}')
        tb.print_exc()
        raise
    print('\n--- STRIPPED MSA (empty cols removed) ---')
    strip = run_all_methods(seqs_strip, tree, entry['held_out'],
                               entry['remaining'],
                               mixdom_params_d3f1, mixdom_params_vbem,
                               class_Qs, class_pis,
                               tkf['ins_rate'], tkf['del_rate'],
                               tkf['ext_rate'],
                               fels21_Q, fels21_pi, fels40_Q, fels40_pi)

    rows = []
    for method in ('fitch', 'varanc', 'fels21', 'fels40',
                    'd3f1', 'd3f1_vbem'):
        f1_full = full[method].get('f1') or full[method].get('nw_f1')
        f1_strip = strip[method].get('f1') or strip[method].get('nw_f1')
        rows.append((method, f1_full, f1_strip,
                      f1_strip - f1_full if f1_full is not None else None))

    print('\n--- F1 comparison ---')
    print(f'{"method":<12} {"full":>10} {"strip":>10} {"delta":>10}')
    for m, ff, fs, delta in rows:
        ff_str = f'{ff:.4f}' if ff is not None else '—'
        fs_str = f'{fs:.4f}' if fs is not None else '—'
        d_str = f'{delta:+.4f}' if delta is not None else '—'
        print(f'{m:<12} {ff_str:>10} {fs_str:>10} {d_str:>10}')

    out_blob = {
        'family': args.family,
        'held_out': entry['held_out'],
        'K': entry['K'],
        'n_cols_full': n_cols_full,
        'n_cols_strip': len(kept_cols),
        'empty_cols': empty_cols,
        'full': full,
        'strip': strip,
        'comparison': [{'method': m, 'f1_full': ff, 'f1_strip': fs,
                          'delta': d}
                         for (m, ff, fs, d) in rows],
    }
    with open(os.path.join(repo_python, args.out), 'w') as f:
        json.dump(out_blob, f, indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
