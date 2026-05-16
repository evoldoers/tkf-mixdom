#!/usr/bin/env python3
"""Tree-VBEM smoke training run.

Loads d3f1 warm-start, runs a small VBEM training (handful of families,
few outer iterations) to verify the pipeline. Does NOT scale to full
Pfam — that's a separate launcher.

Usage:
  cd python && JAX_ENABLE_X64=1 uv run python -u \\
      experiments/tree_vbem_smoke.py \\
      --warm-start pfam/svi_bw_d3f1_postfix_best_val.npz \\
      --n-families 10 --n-outer 3 --n-inner 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from tkfmixdom.jax.tree.varanc_presence_mixdom import (
    parse_mixdom_params_npz,
)
from tkfmixdom.jax.tree.felsenstein import felsenstein_pruning
from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi
from tkfmixdom.jax.train.tree_vbem import vbem_train
from experiments.varanc_presence_benchmark import (
    build_binary_tree_from_node, load_pfam_family,
)
from experiments.varanc_presence_mixdom_benchmark import (
    compute_sub_LL_per_class_per_column,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm-start', type=str,
                        default='pfam/svi_bw_d3f1_postfix_best_val.npz')
    parser.add_argument('--n-families', type=int, default=10)
    parser.add_argument('--n-outer', type=int, default=3)
    parser.add_argument('--n-inner', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--out', type=str,
                        default='experiments/tree_vbem_smoke_history.json')
    parser.add_argument('--checkpoint', action='store_true',
                        help='Write per-iter .npz + history.json to a sibling dir')
    parser.add_argument('--family-offset', type=int, default=0,
                        help='Skip the first N families from unified_short')
    args = parser.parse_args()

    t0 = time.time()
    print(f"[{time.time()-t0:.0f}s] Loading warm-start: {args.warm_start}")
    params0 = parse_mixdom_params_npz(args.warm_start)
    print(f"  D={params0['dom_ins'].shape[0]}, "
          f"F={params0['frag_weights'].shape[1]}, "
          f"C={params0['classdist'].shape[-1]}")
    print(f"  main_ins={float(params0['main_ins']):.5f}, "
          f"main_del={float(params0['main_del']):.5f}")
    print(f"  dom_ins={[float(x) for x in params0['dom_ins']]}")
    print(f"  dom_del={[float(x) for x in params0['dom_del']]}")
    print(f"  dom_w={[float(x) for x in params0['dom_weights']]}")

    # Build per-class Q matrices.
    class_pis = np.asarray(params0['class_pis'])
    class_S_exch = np.asarray(params0['class_S_exch'])
    n_classes = class_pis.shape[0]
    class_Qs = np.zeros((n_classes, 20, 20))
    for c in range(n_classes):
        class_Qs[c] = np.asarray(build_Q_from_S_pi(
            jnp.asarray(class_S_exch[c]),
            jnp.asarray(class_pis[c])))

    # Load training families (use unified_short spec for now).
    spec_path = Path(__file__).parent / 'unified_benchmark_spec.json'
    with open(spec_path) as f:
        spec = json.load(f)
    fam_start = args.family_offset
    fam_end = fam_start + args.n_families
    families_spec = spec['families'][fam_start:fam_end]
    pfam_dir = os.path.expanduser(spec['pfam_dir'])
    tree_dir = os.path.expanduser(spec['tree_dir'])

    # Load each family + compute sub_LL_per_class.
    print(f"[{time.time()-t0:.0f}s] Loading {len(families_spec)} families "
          f"and computing per-class Felsenstein...")
    families = []
    for fi, fspec in enumerate(families_spec):
        try:
            msa_int, tree_node, C = load_pfam_family(fspec, pfam_dir, tree_dir)
        except Exception as e:
            print(f"  skip {fspec['family']}: {e}")
            continue
        leaf_names = sorted(msa_int.keys())
        # No held-out leaf in training mode — use ALL leaves.
        present_arr = np.zeros((len(leaf_names), C), dtype=np.int32)
        for i, name in enumerate(leaf_names):
            present_arr[i] = (msa_int[name] >= 0).astype(np.int32)
        try:
            binary_tree = build_binary_tree_from_node(tree_node)
        except Exception as e:
            print(f"  skip {fspec['family']}: tree build failed: {e}")
            continue
        bt_present = np.zeros((binary_tree.num_leaves, C), dtype=np.int32)
        for li, lname in enumerate(binary_tree.leaf_names):
            if lname in leaf_names:
                bt_present[li] = present_arr[leaf_names.index(lname)]

        # Compute Felsenstein per-class likelihoods (no holdout).
        sub_LL_per_class, _ = compute_sub_LL_per_class_per_column(
            tree_node, msa_int, leaf_names, held_out=None,
            class_Qs=class_Qs, class_pis=class_pis)
        families.append((binary_tree, bt_present, sub_LL_per_class))
        print(f"  loaded {fspec['family']} (L={C}, V={binary_tree.num_internal}, "
              f"leaves={binary_tree.num_leaves})")

    print(f"[{time.time()-t0:.0f}s] {len(families)} families ready. "
          f"Starting VBEM ({args.n_outer} outer × {args.n_inner} inner)...")

    def _save_params_npz(params, path):
        # Save full params as .npz so the run can be resumed / re-loaded.
        # Defensive: auto-create the parent directory in case the
        # caller forgot to (mkdir at startup is the right place but
        # this guards against edits between launch and first iter).
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            main_ins=float(params['main_ins']),
            main_del=float(params['main_del']),
            dom_ins=np.asarray(params['dom_ins']),
            dom_del=np.asarray(params['dom_del']),
            dom_weights=np.asarray(params['dom_weights']),
            frag_weights=np.asarray(params['frag_weights']),
            ext_rates=np.asarray(params['ext_rates']),
            classdist=np.asarray(params['classdist']),
            class_pis=np.asarray(params['class_pis']),
            class_S_exch=np.asarray(params['class_S_exch']),
        )

    ckpt_dir = None
    if args.checkpoint:
        ckpt_dir = Path(args.out).with_suffix('')
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _iter_callback(outer_idx, params_after_M, history):
        if ckpt_dir is None:
            return
        ckpt_path = ckpt_dir / f"iter{outer_idx:03d}.npz"
        _save_params_npz(params_after_M, str(ckpt_path))
        # Also dump the running history JSON.
        with open(ckpt_dir / 'history.json', 'w') as fh:
            json.dump(history, fh, indent=2)
        print(f"  checkpoint: {ckpt_path.name}")

    final_params, history = vbem_train(
        families, params0, n_outer=args.n_outer, n_inner=args.n_inner,
        lr=args.lr, verbose=True, iter_callback=_iter_callback)

    # Save history + final params.
    with open(args.out, 'w') as f:
        json.dump({
            'init': {
                'main_ins': float(params0['main_ins']),
                'main_del': float(params0['main_del']),
                'dom_ins': [float(x) for x in params0['dom_ins']],
                'dom_del': [float(x) for x in params0['dom_del']],
                'dom_weights': [float(x) for x in params0['dom_weights']],
            },
            'final': {
                'main_ins': float(final_params['main_ins']),
                'main_del': float(final_params['main_del']),
                'dom_ins': [float(x) for x in final_params['dom_ins']],
                'dom_del': [float(x) for x in final_params['dom_del']],
                'dom_weights': [float(x) for x in final_params['dom_weights']],
            },
            'history': history,
            'config': {
                'warm_start': args.warm_start,
                'n_families': len(families),
                'n_outer': args.n_outer,
                'n_inner': args.n_inner,
                'lr': args.lr,
            },
        }, f, indent=2)

    # Also save final params as .npz (full-precision, reloadable).
    final_npz = Path(args.out).with_suffix('.npz')
    _save_params_npz(final_params, str(final_npz))

    print(f"[{time.time()-t0:.0f}s] Done. History saved to {args.out}")
    print(f"  Final params saved to {final_npz}")
    print()
    print(f"ELBO trajectory:")
    for h in history:
        print(f"  iter {h['iter']}: mean ELBO/family = {h['mean_elbo']:.2f}")


if __name__ == '__main__':
    main()
