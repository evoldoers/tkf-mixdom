#!/usr/bin/env python3
"""Leaf-holdout presence/absence reconstruction benchmark — MixDom variant.

Adds a ``varanc_mixdom`` method to the simple varanc-presence pipeline
(``experiments/varanc_presence_benchmark.py``):
- Loads a trained MixDom checkpoint (default:
  ``pfam/svi_bw_d3f2c3_diag_postfix_best_val.npz``).
- For each entry, runs TreeVarAnc-MixDom with class-marginalised
  substitution (per varanc-presence-mixdom.tex appendix M).
- Per-column substitution likelihood is computed by Felsenstein up-pass
  on the full tree (held-out leaf treated as missing) under each
  per-class amino-acid rate matrix; the per-fragchar marginalisation
  uses the model's ``classdist`` (averaged across domains for the
  fragchar marginal).

Usage:
  cd python && JAX_ENABLE_X64=1 uv run python -u \\
      experiments/varanc_presence_mixdom_benchmark.py \\
      --dataset unified_short \\
      --mixdom-params pfam/svi_bw_d3f2c3_diag_postfix_best_val.npz \\
      --out experiments/varanc_presence_mixdom_unified_short.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tkfmixdom.jax.tree.varanc_presence import (
    parse_binary_tree, edge_lookup, BinaryTree,
    make_q_conditionals, make_root_dist, leaf_clamp_to_beta,
    bp_pair_marginals, NYI, PRESENT, DELETED, N_Z,
)
from tkfmixdom.jax.tree.varanc_presence_mixdom import (
    parse_mixdom_params_npz, mixdom_reduced_T_pair,
    make_tuple_dist, fragchar_marginal_from_tuple,
    expected_branch_LL_mixdom, class_marginalised_sub_LL_per_column,
    singlet_root_log_prior_mixdom, elbo_mixdom,
)
from tkfmixdom.jax.tree.varanc_presence import (
    entropy_per_column,
)
from tkfmixdom.jax.tree.felsenstein import felsenstein_pruning
from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi
from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT

# Pull in helpers from the simple benchmark.
from experiments.varanc_presence_benchmark import (
    build_binary_tree_from_node, fitch_seeded_init, fitch_labels,
    f1_pr, logp_binary_pred, logp_binary_true, predict_holdout_fitch,
    load_pfam_family, load_balibase_family, load_treefam_family,
)
# Shape-keyed JIT replacement for _predict_one_holdout_mixdom (Layer 1
# of the mixdom benchmark refactor — JIT cache reuses across families
# of the same padded shape, eliminating the per-family recompile that
# made the legacy path the dominant cost on big benchmarks).
from tkfmixdom.jax.train.tree_vbem import predict_holdout_mixdom


t0 = time.time()
def log(msg): print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Per-column per-class Felsenstein up-pass on the full tree.
# ---------------------------------------------------------------------------


def compute_sub_LL_per_class_per_column(tree_node, msa_int, leaf_names_in_msa,
                                         held_out, class_Qs, class_pis):
    """For each (column, class), compute the Felsenstein up-pass likelihood.

    Treats absent leaves and the held-out leaf as missing data (uniform
    over residues).

    Args:
        tree_node: TreeNode root of the family tree.
        msa_int: dict {leaf_name: int array of residues, -1 for gap}.
        leaf_names_in_msa: list of leaf names in MSA order.
        held_out: name of held-out leaf (treated as missing).
        class_Qs: (C, 20, 20) per-class rate matrix.
        class_pis: (C, 20) per-class stationary.

    Returns:
        L_per_class: (n_cols, C) per-column per-class likelihood.
    """
    L = len(next(iter(msa_int.values())))
    n_classes = class_Qs.shape[0]
    out = np.zeros((L, n_classes), dtype=np.float64)

    for col in range(L):
        leaf_chars = {}
        for name in leaf_names_in_msa:
            if name == held_out:
                leaf_chars[name] = -1  # missing
            else:
                ch = msa_int[name][col] if col < len(msa_int[name]) else -1
                leaf_chars[name] = int(ch)
        for c in range(n_classes):
            Q = class_Qs[c]
            pi = class_pis[c]
            log_p = felsenstein_pruning(tree_node, leaf_chars, Q, pi)
            out[col, c] = float(log_p)

    # Convert log-likelihoods to likelihoods. Apply per-column shift to
    # avoid overflow then keep the shifted likelihood (the shift cancels
    # in class marginalisation but inflates absolute LL — we'll add the
    # shift back into the ELBO offset during reporting if needed).
    out_shifted = out - out.max(axis=-1, keepdims=True)
    return np.exp(out_shifted), out.max(axis=-1)


# ---------------------------------------------------------------------------
# Predict held-out leaf via TreeVarAnc-MixDom.
# ---------------------------------------------------------------------------


def _predict_one_holdout_mixdom(binary_tree, leaf_present_remaining,
                                  holdout_leaf_idx, mixdom_params,
                                  L_sub_per_class,
                                  n_iter=150, lr=0.05, seed=0):
    """Run TreeVarAnc-MixDom with class-marginalised substitution.

    Args:
        binary_tree: BinaryTree (full tree including held-out leaf).
        leaf_present_remaining: (num_leaves, L) {0, 1} per column;
            held-out leaf row will be treated as uniform.
        holdout_leaf_idx: index of held-out leaf in binary_tree.leaf_names.
        mixdom_params: dict from parse_mixdom_params_npz.
        L_sub_per_class: (L, C) per-class Felsenstein likelihood per column.

    Returns:
        p_present: (L,) predicted P(held-out leaf is present) per column.
        internal_MAP_binary: (n_internal, L) per-internal-node MAP binary
            presence indicator.
    """
    L = leaf_present_remaining.shape[1]
    le, re = edge_lookup(binary_tree)

    # Clip degenerate branch lengths (same as simple TreeVarAnc).
    edge_lengths = np.maximum(np.asarray(binary_tree.edge_length), 1e-3)
    binary_tree = binary_tree._replace(edge_length=edge_lengths)

    # Leaf clamp: uniform on held-out leaf.
    leaf_clamp = np.array(leaf_clamp_to_beta(leaf_present_remaining))
    leaf_clamp[holdout_leaf_idx, :, :] = 1.0
    leaf_clamp_jnp = jnp.asarray(leaf_clamp)

    # Inner 3-state init via Fitch seeding (reuse helper). Use TIED
    # per-edge logits (broadcast across columns), matching the simple
    # TreeVarAnc parameterisation — per-(edge, column) logits give too
    # large a parameter space for Adam to optimise reliably.
    edge_logits, root_logit = fitch_seeded_init(
        binary_tree, leaf_present_remaining, holdout_leaf_idx)
    rng = np.random.default_rng(seed)
    edge_logits = edge_logits + jnp.asarray(
        rng.standard_normal(edge_logits.shape) * 0.05, dtype=jnp.float64)

    # The free variational parameters are tied across columns for the
    # inner 3-state q (only `edge_logits` of shape (E, 2) and a scalar
    # `root_logit`); the per-column tuple categorical `tuple_logits` of
    # shape (L, T) remains free per column.

    # Tuple init: bias toward fragchar with highest substitution likelihood
    # for each column. q^(f)_n proportional to L_sub averaged over classes.
    n_dom = mixdom_params['dom_ins'].shape[0]
    n_frag = mixdom_params['frag_weights'].shape[1]
    T = n_dom * n_frag
    # Rough init: per-column favorable fragchar via classdist*L_sub.
    # classdist shape: (D, F, C); use d-mean for fragchar prior.
    classdist_f = jnp.asarray(mixdom_params['classdist']).mean(axis=0)  # (F, C)
    L_sub_jnp = jnp.asarray(L_sub_per_class)  # (L, C)
    log_L_per_frag_init = jax.scipy.special.logsumexp(
        jnp.log(jnp.maximum(classdist_f, 1e-30))[None, :, :]
        + jnp.log(jnp.maximum(L_sub_jnp, 1e-30))[:, None, :],
        axis=-1)  # (L, F)
    # Tile fragchar logits across domains uniformly.
    tuple_logits_init = jnp.tile(log_L_per_frag_init / 2.0, (1, n_dom))  # (L, T = D*F)
    tuple_logits_init = tuple_logits_init.reshape(L, n_dom, n_frag)
    # Add small jitter.
    tuple_logits_init = tuple_logits_init + jnp.asarray(
        rng.standard_normal(tuple_logits_init.shape) * 0.05)
    tuple_logits_init = tuple_logits_init.reshape(L, T)

    # Build closure ELBO function with leaf clamp passed in.
    # Inner logits and root logit are TIED across columns (broadcast).
    @jax.jit
    def neg_elbo(edge_logits, root_logit, tuple_logits):
        # Broadcast tied per-edge logits across columns.
        inner_logits = jnp.broadcast_to(
            edge_logits[:, None, :], (binary_tree.num_edges, L, 2))
        root_logits = jnp.broadcast_to(root_logit, (L,))
        q_cond = make_q_conditionals(inner_logits)
        root_dist = make_root_dist(root_logits)
        pair_marg, log_Z = bp_pair_marginals(
            q_cond, root_dist, leaf_clamp_jnp, binary_tree, le, re)
        q_tau = make_tuple_dist(tuple_logits)
        edge_lens = jnp.asarray(binary_tree.edge_length)

        # Branch LL.
        def log_T_for(t):
            T_pair = mixdom_reduced_T_pair(mixdom_params, t)
            return jnp.log(jnp.maximum(T_pair, 1e-300))
        log_T_per_edge = jax.vmap(log_T_for)(edge_lens)
        branch_LLs = jax.vmap(
            lambda pm, lt: expected_branch_LL_mixdom(pm, q_tau, lt))(
                pair_marg, log_T_per_edge)
        sum_branch_LL = jnp.sum(branch_LLs)

        # Sub LL.
        log_L_per_frag = class_marginalised_sub_LL_per_column(
            L_sub_jnp, mixdom_params['classdist'])
        q_f = fragchar_marginal_from_tuple(q_tau, n_dom, n_frag)
        sum_sub_LL = jnp.sum(q_f * log_L_per_frag)

        # Entropies.
        H_inner = jnp.sum(entropy_per_column(
            pair_marg, root_dist, beta_root=None,
            node_marg_internal=None, q_cond=q_cond, tree=binary_tree))
        log_q_tau = jnp.log(jnp.maximum(q_tau, 1e-30))
        H_tau = -jnp.sum(q_tau * log_q_tau)

        # Root prior.
        log_prior = singlet_root_log_prior_mixdom(
            root_dist, q_tau, mixdom_params, n_dom, n_frag)

        return -(sum_branch_LL + sum_sub_LL + log_prior + H_inner + H_tau
                 + jnp.sum(log_Z))

    optimizer = optax.adam(lr)
    state = optimizer.init((edge_logits, root_logit, tuple_logits_init))

    grad_fn = jax.jit(jax.grad(neg_elbo, argnums=(0, 1, 2)))
    params = (edge_logits, root_logit, tuple_logits_init)

    for step in range(n_iter):
        grads = grad_fn(*params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)

    # Final BP for held-out leaf marginal.
    edge_logits_f, root_logit_f, tuple_logits = params
    inner_logits_f = jnp.broadcast_to(
        edge_logits_f[:, None, :], (binary_tree.num_edges, L, 2))
    root_logits_f = jnp.broadcast_to(root_logit_f, (L,))
    q_cond = make_q_conditionals(inner_logits_f)
    root_dist = make_root_dist(root_logits_f)
    pair_marg, _ = bp_pair_marginals(
        q_cond, root_dist, leaf_clamp_jnp, binary_tree, le, re)

    holdout_node = binary_tree.num_internal + holdout_leaf_idx
    edge_to_holdout = None
    for e in range(binary_tree.num_edges):
        if int(binary_tree.edge_child[e]) == holdout_node:
            edge_to_holdout = e
            break
    p_present = pair_marg[edge_to_holdout, :, :, PRESENT].sum(axis=-1)

    # Per-internal-node MAP for the model-probability selector.
    n_internal = binary_tree.num_internal
    node_marg = np.zeros((n_internal, L, N_Z))
    edge_parent_np = np.asarray(binary_tree.edge_parent)
    edge_child_np = np.asarray(binary_tree.edge_child)
    for v in range(n_internal):
        as_child = np.where(edge_child_np == v)[0]
        as_parent = np.where(edge_parent_np == v)[0]
        if len(as_child):
            e = int(as_child[0])
            node_marg[v] = np.asarray(pair_marg[e]).sum(axis=-2)
        elif len(as_parent):
            e = int(as_parent[0])
            node_marg[v] = np.asarray(pair_marg[e]).sum(axis=-1)
    map_state = node_marg.argmax(axis=-1)
    internal_MAP_binary = (map_state == PRESENT).astype(np.int32)

    return np.asarray(p_present), internal_MAP_binary


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',
                        choices=['unified_short_test', 'unified_long_test',
                                 'unified_hard_test', 'unified_xhard_test',
                                 'unified_short', 'unified_long',
                                 'balibase', 'treefam'],
                        default='unified_short_test')
    parser.add_argument('--mixdom-params', type=str,
                        default='pfam/svi_bw_d3f1_postfix_best_val.npz',
                        help='MixDom params .npz. d3f1 is the default first '
                        'choice; pass d3f2c3 / d3f3c27 / etc to compare.')
    parser.add_argument('--n-families', type=int, default=0)
    parser.add_argument('--methods', type=str,
                        default='mixdom,fitch',
                        help='Comma-separated subset of {mixdom,fitch}.')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--n-iter', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--skip-existing-fams', action='store_true',
                        help='Load existing --out JSON and skip families '
                        'already present in its results list. Used for '
                        'partial re-runs after --out family entries have '
                        'been deleted (see empty_col_cleanup.py).')
    args = parser.parse_args()

    methods = set(args.methods.split(','))

    # Load MixDom params + build per-class Q matrices.
    mixdom_params = parse_mixdom_params_npz(args.mixdom_params)
    log(f"Loaded MixDom params from {args.mixdom_params}")
    log(f"  D={mixdom_params['dom_ins'].shape[0]}, "
        f"F={mixdom_params['frag_weights'].shape[1]}, "
        f"C={mixdom_params['class_pis'].shape[0]}")

    # Build per-class Q from S_exch + pi.
    class_pis = np.asarray(mixdom_params['class_pis'])
    class_S_exch = np.asarray(mixdom_params['class_S_exch'])
    n_classes = class_pis.shape[0]
    class_Qs = np.zeros((n_classes, 20, 20))
    for c in range(n_classes):
        Q = np.asarray(build_Q_from_S_pi(
            jnp.asarray(class_S_exch[c]),
            jnp.asarray(class_pis[c])))
        class_Qs[c] = Q

    if args.dataset == 'unified_short_test':
        spec_path = Path(__file__).parent / 'unified_benchmark_test_spec.json'
    elif args.dataset == 'unified_long_test':
        spec_path = Path(__file__).parent / 'unified_benchmark_long_test_spec.json'
    elif args.dataset == 'unified_hard_test':
        spec_path = Path(__file__).parent / 'unified_benchmark_hard_test_spec.json'
    elif args.dataset == 'unified_xhard_test':
        spec_path = Path(__file__).parent / 'unified_benchmark_xhard_test_spec.json'
    elif args.dataset == 'unified_short':
        spec_path = Path(__file__).parent / 'contaminated_val_split_triage_for_deletion' / 'unified_benchmark_spec.json'
    elif args.dataset == 'unified_long':
        spec_path = Path(__file__).parent / 'contaminated_val_split_triage_for_deletion' / 'unified_benchmark_long_spec.json'
    elif args.dataset == 'balibase':
        spec_path = Path(__file__).parent / 'balibase_reconstruction_spec.json'
    else:
        spec_path = Path(__file__).parent / 'treefam_reconstruction_spec.json'

    with open(spec_path) as f:
        spec = json.load(f)
    families = spec['families']
    pfam_dir = tree_dir = balibase_ref_dir = treefam_dir = None
    if args.dataset.startswith('unified_'):
        pfam_dir = os.path.expanduser(spec['pfam_dir'])
        tree_dir = os.path.expanduser(spec['tree_dir'])
    elif args.dataset == 'balibase':
        balibase_ref_dir = os.path.expanduser(
            "~/bio-datasets/data/balibase/bali3pdbm/ref")
    else:
        treefam_dir = os.path.expanduser(spec.get('treefam_dir',
            "~/bio-datasets/data/treefam/treefam_family_data"))
    if args.n_families > 0:
        families = families[:args.n_families]

    out_path = args.out or f'experiments/varanc_presence_mixdom_{args.dataset}.json'

    # Resume / partial-rerun support: skip families already in --out JSON.
    existing_results = []
    skip_fams = set()
    if args.skip_existing_fams and os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing_results = json.load(f).get('results', [])
            skip_fams = {r['family'] for r in existing_results
                          if r.get('family')}
            log(f"--skip-existing-fams: loaded {len(existing_results)} "
                f"existing entries from {out_path}; will skip those families")
        except Exception as e:
            log(f"--skip-existing-fams: failed to load {out_path}: {e}; "
                f"running all families")

    log(f"Running on {len(families)} families ({args.dataset}); methods={methods}")
    print(f"{'family':<12} {'L':>3} {'V':>3} | "
          + " | ".join(f"{m:>10} F1" for m in sorted(methods))
          + " | time")
    print("-" * (40 + 14 * len(methods)))

    results = list(existing_results)
    save_every = 5
    for fi, fspec in enumerate(families):
        if fspec['family'] in skip_fams:
            continue
        try:
            if args.dataset.startswith('unified_'):
                msa_int, tree_node, C = load_pfam_family(fspec, pfam_dir, tree_dir)
            elif args.dataset == 'balibase':
                msa_int, tree_node, C = load_balibase_family(fspec, balibase_ref_dir)
            else:
                msa_int, tree_node, C = load_treefam_family(fspec, treefam_dir)
        except Exception as e:
            log(f"[{fi+1}/{len(families)}] {fspec['family']}: load failed: {e}")
            continue

        held_out = fspec['held_out']
        if held_out not in msa_int:
            log(f"[{fi+1}/{len(families)}] {fspec['family']}: held_out absent from MSA")
            continue

        leaf_names = sorted(msa_int.keys())
        present_arr = np.zeros((len(leaf_names), C), dtype=np.int32)
        for i, name in enumerate(leaf_names):
            present_arr[i] = (msa_int[name] >= 0).astype(np.int32)
        gt_present = present_arr[leaf_names.index(held_out)]

        try:
            binary_tree = build_binary_tree_from_node(tree_node)
        except Exception as e:
            log(f"[{fi+1}/{len(families)}] {fspec['family']}: tree binarise failed: {e}")
            continue

        bt_leaf_to_present_row = {}
        for li, lname in enumerate(binary_tree.leaf_names):
            if lname in leaf_names:
                bt_leaf_to_present_row[li] = leaf_names.index(lname)
        bt_present = np.zeros((binary_tree.num_leaves, C), dtype=np.int32)
        for li in range(binary_tree.num_leaves):
            row = bt_leaf_to_present_row.get(li)
            if row is not None:
                bt_present[li] = present_arr[row]

        if held_out not in binary_tree.leaf_names:
            log(f"[{fi+1}/{len(families)}] {fspec['family']}: held_out missing in binary tree")
            continue
        holdout_idx_bt = binary_tree.leaf_names.index(held_out)

        entry = {
            'family': fspec['family'],
            'held_out': held_out,
            'n_cols': int(C),
            'n_leaves': int(binary_tree.num_leaves),
            'n_internal': int(binary_tree.num_internal),
            'gt_present': gt_present.tolist(),
            'methods': {},
        }

        line = f"{fspec['family']:<12} {C:>3} {binary_tree.num_internal:>3} |"

        # Method: MixDom-VarAnc.
        if 'mixdom' in methods:
            try:
                tv0 = time.time()
                # Compute per-column per-class Felsenstein likelihoods.
                L_sub_per_class, _ = compute_sub_LL_per_class_per_column(
                    tree_node, msa_int, leaf_names, held_out,
                    class_Qs, class_pis)
                # Pre-compute the Fitch-seeded init the same way the
                # legacy path does internally; pass it to the new
                # shape-keyed predict_holdout_mixdom so the trajectory
                # is comparable across the two paths.
                seed_edge_logits, seed_root_logit = fitch_seeded_init(
                    binary_tree, bt_present, holdout_idx_bt)
                p_pred = predict_holdout_mixdom(
                    binary_tree, bt_present, holdout_idx_bt, mixdom_params,
                    L_sub_per_class, n_iter=args.n_iter, lr=args.lr, seed=fi,
                    init_edge_logits=np.asarray(seed_edge_logits),
                    init_root_logit=float(seed_root_logit))
                tv = time.time() - tv0
                f1, prec, rec, *_ = f1_pr(p_pred, gt_present)
                # logp_target: joint log posterior of the binary
                # presence-prediction sequence — see notes in
                # varanc_presence_benchmark.py near logp_binary_pred.
                # logp_true: same per-column posterior, scored against
                # the ground-truth gt_present vector — see logp_binary_true.
                entry['methods']['mixdom'] = {
                    'p_present': p_pred.tolist(),
                    'f1': f1, 'precision': prec, 'recall': rec, 'time': tv,
                    'logp_target': logp_binary_pred(p_pred),
                    'logp_true': logp_binary_true(p_pred, gt_present),
                }
                line += f"  mixdom {f1:.3f} |"
            except Exception as e:
                import traceback
                line += f"  mixdom ERR |"
                entry['methods']['mixdom'] = {
                    'error': str(e), 'traceback': traceback.format_exc()}

        # Method: Fitch.
        if 'fitch' in methods:
            try:
                tv0 = time.time()
                p_pred = predict_holdout_fitch(
                    binary_tree, bt_present, holdout_idx_bt)
                tv = time.time() - tv0
                f1, prec, rec, *_ = f1_pr(p_pred, gt_present)
                # Hard-label predictor: no probabilistic posterior, so
                # logp_target / logp_true are category errors here. Omit.
                entry['methods']['fitch'] = {
                    'p_present': p_pred.tolist(),
                    'f1': f1, 'precision': prec, 'recall': rec, 'time': tv,
                }
                line += f"   fitch {f1:.3f} |"
            except Exception as e:
                line += f"   fitch ERR |"
                entry['methods']['fitch'] = {'error': str(e)}

        line += f" {time.time()-t0:.0f}s"
        print(line)
        results.append(entry)

        if (fi + 1) % save_every == 0:
            with open(out_path, 'w') as f:
                json.dump({
                    'spec': args.dataset,
                    'mixdom_params': args.mixdom_params,
                    'results': results,
                }, f)

    print("\n" + "=" * 60)
    print(f"Final: n={len(results)} entries")
    for m in sorted(methods):
        f1s = [r['methods'][m]['f1'] for r in results
               if m in r['methods'] and 'f1' in r['methods'][m]]
        if f1s:
            print(f"  {m:>8}: F1 mean={np.mean(f1s):.4f}, median={np.median(f1s):.4f}, n={len(f1s)}")

    with open(out_path, 'w') as f:
        json.dump({
            'spec': args.dataset,
            'mixdom_params': args.mixdom_params,
            'results': results,
        }, f)
    log(f"Saved {out_path}")


if __name__ == '__main__':
    main()
