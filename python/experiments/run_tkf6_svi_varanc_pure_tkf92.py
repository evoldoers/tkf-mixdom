#!/usr/bin/env python3
"""TKF §6 production: pure-TKF92 svi-VarAnc on Pfam.

Section sec:results-svi-varanc.  Cherry-trained TKF92 (§3) is refined
on a Pfam holdout set via stochastic VBEM with breadth-first family
minibatch sampling, EMA on aggregate suff stats, and the paper-aligned
TKF92 BW M-step (T = t per branch; L on resolved n̂; notext includes
body→E).

Loads Pfam families on demand (lazy per-minibatch) so memory stays
small.  Runs entirely on CPU (the per-family ancestral-presence ELBO
optimiser has lower JIT overhead than the 2D pair-FB path that hit
GPU OOM for §4).

Output: pfam/tkf92_svi_varanc_pure_train.npz with
  (ins_rate, del_rate, ext, history, val_history).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
_dev_args, _ = _p.parse_known_args()
os.environ['JAX_PLATFORMS'] = _dev_args.device
os.environ.setdefault('JAX_ENABLE_X64', '1')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.train.tkf92_vbem import svi_vbem_train_tkf92
from tkfmixdom.jax.tree.tree_varanc import name_internal_nodes
from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT


def parse_sto(path):
    """Minimal Stockholm parser; same as ancrec_benchmark.parse_sto."""
    seqs = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#') or line.startswith('//') or not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, seq = parts
            seqs[name] = seqs.get(name, '') + seq
    return seqs


def load_family(fam, pfam_dir, tree_dir):
    """Returns (binary_tree, leaf_present) for the family.

    leaf_present is a (n_leaves, n_cols) int array in
    binary_tree.leaf_names order.  1 = residue present at that column,
    0 = gap.  Sequences not in the tree are dropped; tree leaves not
    in the MSA cause an error.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from varanc_presence_benchmark import build_binary_tree_from_node

    tree_path = os.path.join(tree_dir, f'{fam}.nwk')
    sto_path = os.path.join(pfam_dir, f'{fam}.sto')
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f'Tree not found for {fam}')
    if not os.path.exists(sto_path):
        raise FileNotFoundError(f'MSA not found for {fam}')
    seqs = parse_sto(sto_path)
    if not seqs:
        raise ValueError(f'Empty MSA for {fam}')

    with open(tree_path) as f:
        tree_text = f.read().strip()
    tree_node = parse_newick(tree_text)
    name_internal_nodes(tree_node)
    binary_tree = build_binary_tree_from_node(tree_node)

    # Build (n_leaves, n_cols) presence array in binary_tree.leaf_names order.
    n_cols = max(len(s) for s in seqs.values())
    leaf_names = list(binary_tree.leaf_names)
    arr = np.zeros((len(leaf_names), n_cols), dtype=np.int8)
    for i, name in enumerate(leaf_names):
        seq = seqs.get(name)
        if seq is None:
            raise ValueError(f'Leaf {name} not in MSA for {fam}')
        for j, ch in enumerate(seq[:n_cols]):
            if ch != '-' and ch != '.':
                arr[i, j] = 1
    return binary_tree, arr


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spec', type=str,
                    default='experiments/pfam_train_with_trees_spec.json',
                    help='JSON spec listing families.  Default is the '
                         'Pfam TRAIN split (19,850 families with FastTree '
                         'trees).  Per feedback_unified_val_contamination, '
                         'do NOT train on unified_*_test_spec.json — those '
                         'are evaluation splits.')
    p.add_argument('--warm-start', type=str,
                    default='pfam/tkf92_K1_train.npz',
                    help='§3 Maraschino fit; used to init (λ, μ, r=ext).')
    p.add_argument('--pfam-dir', default=os.path.expanduser(
        '~/bio-datasets/data/pfam-seed'))
    p.add_argument('--tree-dir', default=os.path.expanduser(
        '~/bio-datasets/data/pfam-seed/trees'))
    p.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                    help='Selected by env trick at top of file; included '
                         'here so argparse does not error on the flag.')
    p.add_argument('--use-padding', action='store_true',
                    help='Enable JIT-cacheable padded ELBO E-step '
                         '(geometric leaf+col bins, ghost-edge identity '
                         'transitions). Required for GPU runs at '
                         'iter > ~20 to avoid JIT cache OOM.')
    p.add_argument('--n-iter', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--n-inner', type=int, default=30)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--svi-tau', type=float, default=10.0)
    p.add_argument('--svi-kappa', type=float, default=0.7)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out',
                    default='pfam/tkf92_svi_varanc_pure_train.npz')
    args = p.parse_args()

    # Load warm-start params.
    if os.path.exists(args.warm_start):
        ws = np.load(args.warm_start, allow_pickle=True)
        # Maraschino MixDom1-formatted: dom_ins, dom_del scalars; ext_rates 3-d.
        init_ins = float(np.atleast_1d(ws['dom_ins'])[0])
        init_del = float(np.atleast_1d(ws['dom_del'])[0])
        init_ext = float(np.array(ws['ext_rates']).flatten()[0])
        print(f'Warm-start from {args.warm_start}: '
              f'(λ, μ, r)=({init_ins:.5f}, {init_del:.5f}, {init_ext:.4f})',
              flush=True)
    else:
        init_ins, init_del, init_ext = 0.04, 0.05, 0.5
        print(f'Warm-start file not found; using default init '
              f'(λ, μ, r)=({init_ins}, {init_del}, {init_ext})', flush=True)

    with open(args.spec) as f:
        spec = json.load(f)
    families = spec['families']
    print(f'Spec {args.spec}: {len(families)} families.', flush=True)

    # Build family_provider that loads on demand.
    cache = {}

    def family_provider(idx):
        if idx not in cache:
            fam = families[idx]['family']
            cache[idx] = load_family(fam, args.pfam_dir, args.tree_dir)
        return cache[idx]

    print(f'\nLaunching SVI-VBEM: '
          f'init=({init_ins:.5f}, {init_del:.5f}, {init_ext:.4f}), '
          f'n_iter={args.n_iter}, batch={args.batch_size}, '
          f'n_inner={args.n_inner}, '
          f'tau={args.svi_tau}, kappa={args.svi_kappa}', flush=True)

    # Per-iter checkpoint + NaN guard so a crash mid-run preserves
    # the trajectory so far.
    ckpt_path = args.out + '.iter_ckpt.npz'

    def iter_cb(k, ins_k, del_k, ext_k, hist):
        if not (np.isfinite(ins_k) and np.isfinite(del_k)
                  and np.isfinite(ext_k)):
            raise RuntimeError(
                f'NaN/inf parameters at iter {k}: '
                f'(ins, del_, ext)=({ins_k}, {del_k}, {ext_k})')
        np.savez(ckpt_path,
                  ins_rate=ins_k, del_rate=del_k, ext=ext_k,
                  history=np.array(hist, dtype=object),
                  iter=k, n_iter=args.n_iter,
                  batch_size=args.batch_size, n_families=len(families))

    out = svi_vbem_train_tkf92(
        family_provider, init_ins, init_del, init_ext,
        n_total_families=len(families),
        family_indices=list(range(len(families))),
        n_iter=args.n_iter, batch_size=args.batch_size,
        n_inner=args.n_inner, lr=args.lr,
        svi_tau=args.svi_tau, svi_kappa=args.svi_kappa,
        seed=args.seed, verbose=True,
        use_padding=args.use_padding,
        iter_callback=iter_cb)

    print(f'\nFinal: λ={out["ins_rate"]:.5f} μ={out["del_rate"]:.5f} '
          f'ext={out["ext"]:.4f}', flush=True)
    np.savez(args.out,
              ins_rate=out['ins_rate'], del_rate=out['del_rate'],
              ext=out['ext'],
              history=np.array(out['history'], dtype=object),
              val_history=np.array(out['val_history'], dtype=object),
              n_families=len(families),
              n_iter=args.n_iter, batch_size=args.batch_size)
    print(f'Saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
