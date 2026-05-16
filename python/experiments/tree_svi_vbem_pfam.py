#!/usr/bin/env python3
"""SVI-VBEM training run on Pfam.

Per varanc-vbem.tex sec:vbem-svi: stochastic VBEM with EMA on suff
stats and breadth-first minibatch sampling. The minibatch is small
(~10 families) and the sampler guarantees full corpus coverage every
~N/B iterations.

Usage:
  cd python && JAX_ENABLE_X64=1 uv run python -u \\
      experiments/tree_svi_vbem_pfam.py \\
      --warm-start pfam/svi_bw_d3f1_postfix_best_val.npz \\
      --n-families 200 --batch-size 10 --n-iter 200 --n-inner 30 \\
      --tau 10 --kappa 0.7 \\
      --checkpoint --out experiments/tree_svi_vbem_run1.json
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
from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi
from tkfmixdom.jax.train.tree_vbem import svi_vbem_train
from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT
from tkfmixdom.jax.tree.tree_varanc import name_internal_nodes
from experiments.ancrec_benchmark import parse_sto
from experiments.varanc_presence_benchmark import build_binary_tree_from_node
from experiments.varanc_presence_mixdom_benchmark import (
    compute_sub_LL_per_class_per_column,
)


def load_train_pfam_family(fspec, pfam_dir, tree_dir):
    """Training-mode loader: needs only fspec['family'] (no held_out).

    Derives n_cols from the longest sequence in the MSA. Returns
    (msa_int, tree_node, C) — same shape as load_pfam_family.
    """
    fam = fspec['family']
    tree_path = os.path.join(tree_dir, f'{fam}.nwk')
    if not os.path.exists(tree_path):
        tree_path = os.path.join(tree_dir, f'{fam}.tree')
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f'Tree not found for {fam}')
    sto_path = os.path.join(pfam_dir, f'{fam}.sto')
    if not os.path.exists(sto_path):
        raise FileNotFoundError(f'MSA not found for {fam}')

    seqs = parse_sto(sto_path)
    if not seqs:
        raise ValueError(f'Empty MSA for {fam}')
    C = max(len(s) for s in seqs.values())
    msa = {}
    for name, seq in seqs.items():
        arr = np.full(C, -1, dtype=np.int32)
        for j, ch in enumerate(seq):
            if ch in AA_TO_INT:
                idx = AA_TO_INT[ch]
                if idx < 20:
                    arr[j] = idx
        msa[name] = arr

    with open(tree_path) as f:
        tree_text = f.read().strip()
    tree = parse_newick(tree_text)
    name_internal_nodes(tree)
    return msa, tree, C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm-start', type=str,
                        default='pfam/svi_bw_d3f1_postfix_best_val.npz')
    parser.add_argument('--spec', type=str,
                        default='experiments/unified_benchmark_spec.json',
                        help='JSON spec listing families (use the unified_short '
                        'or other spec; first n_families entries are used).')
    parser.add_argument('--n-families', type=int, default=0,
                        help='Number of families from spec (0 = all).')
    parser.add_argument('--family-offset', type=int, default=0)
    parser.add_argument('--shuffle-seed', type=int, default=-1,
                        help='If >=0, shuffle the spec families with this seed '
                        'before applying --family-offset/--n-families. Use to '
                        'pick a random subsample of a large training pool.')
    parser.add_argument('--max-leaves', type=int, default=0,
                        help='Filter spec to families with at most this many '
                        'tree leaves. 0 = no filter. Reduces JIT-cache '
                        'thrashing when the pool has wide tree-size variation.')
    parser.add_argument('--max-cols', type=int, default=0,
                        help='Filter spec to families with at most this many '
                        'alignment columns. 0 = no filter. Same purpose as '
                        '--max-leaves.')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--n-iter', type=int, default=100)
    parser.add_argument('--n-inner', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--tau', type=float, default=10.0,
                        help='SVI step-size schedule offset.')
    parser.add_argument('--kappa', type=float, default=0.7,
                        help='SVI step-size schedule decay (∈(0.5, 1]).')
    parser.add_argument('--sampler-seed', type=int, default=0)
    parser.add_argument('--out', type=str,
                        default='experiments/tree_svi_vbem_history.json')
    parser.add_argument('--checkpoint', action='store_true',
                        help='Write per-iter .npz + history.json to a sibling dir')
    parser.add_argument('--val-n', type=int, default=100,
                        help='Number of held-out training families to evaluate '
                        'mean ELBO on every --val-every-k iterations. The val '
                        'pool is the last N families of the (filtered + shuffled) '
                        'training pool — disjoint from the breadth-first sampler.')
    parser.add_argument('--val-every-k', type=int, default=5)
    parser.add_argument('--val-n-inner', type=int, default=0,
                        help='Adam steps per val-family E-step (0 = same as '
                        'training --n-inner).')
    args = parser.parse_args()

    t0 = time.time()
    print(f"[{time.time()-t0:.0f}s] Loading warm-start: {args.warm_start}")
    params0 = parse_mixdom_params_npz(args.warm_start)
    # Detect full-state checkpoint (has ema_*/sampler_* keys) and pull
    # them out so we can resume the EMA accumulator + sampler visit
    # history rather than discard them.
    from tkfmixdom.jax.train.tree_vbem import (
        deserialize_ema_stats, deserialize_sampler_state,
    )
    _ws_raw = dict(np.load(args.warm_start, allow_pickle=True))
    init_ema = deserialize_ema_stats(_ws_raw, n_dom=params0['dom_ins'].shape[0])
    init_sampler_state = deserialize_sampler_state(_ws_raw)
    init_iter = (int(_ws_raw['svi_iter']) + 1
                  if 'svi_iter' in _ws_raw else 0)
    init_history = []
    if 'svi_history_json' in _ws_raw:
        try:
            init_history = json.loads(str(_ws_raw['svi_history_json']))
        except Exception:
            init_history = []
    print(f"  D={params0['dom_ins'].shape[0]}, "
          f"F={params0['frag_weights'].shape[1]}, "
          f"C={params0['classdist'].shape[-1]}")
    print(f"  main_ins={float(params0['main_ins']):.5f}, "
          f"main_del={float(params0['main_del']):.5f}")
    print(f"  dom_ins={[float(x) for x in params0['dom_ins']]}")
    if init_ema is not None:
        print(f"  RESUMING from full state: iter {init_iter}, "
              f"ema accumulator + sampler visit history loaded "
              f"(history len={len(init_history)})")
    else:
        print(f"  warm-start params only (no EMA/sampler state in "
              f"checkpoint — fresh start of accumulator)")

    # Build per-class Q matrices (stationary; reused for all families).
    class_pis = np.asarray(params0['class_pis'])
    class_S_exch = np.asarray(params0['class_S_exch'])
    n_classes = class_pis.shape[0]
    class_Qs = np.zeros((n_classes, 20, 20))
    for c in range(n_classes):
        class_Qs[c] = np.asarray(build_Q_from_S_pi(
            jnp.asarray(class_S_exch[c]),
            jnp.asarray(class_pis[c])))

    # Load training families from spec.
    with open(args.spec) as f:
        spec = json.load(f)
    all_families = list(spec['families'])
    pfam_dir_eager = os.path.expanduser(spec['pfam_dir'])
    tree_dir_eager = os.path.expanduser(spec['tree_dir'])

    # Optional shape filter — count leaves per .nwk via comma-counting and
    # MSA columns from the .sto's first non-comment line. Both are cheap
    # (single-pass file reads); avoids paying per-iter family loads on
    # families that would explode JIT cache or per-batch wall time.
    if args.max_leaves > 0 or args.max_cols > 0:
        print(f"[{time.time()-t0:.0f}s] Pre-scanning {len(all_families)} families "
              f"for size filter (max_leaves={args.max_leaves}, "
              f"max_cols={args.max_cols})...")
        kept = []
        for fs in all_families:
            fam = fs['family']
            nwk_path = os.path.join(tree_dir_eager, f'{fam}.nwk')
            sto_path = os.path.join(pfam_dir_eager, f'{fam}.sto')
            if not (os.path.exists(nwk_path) and os.path.exists(sto_path)):
                continue
            try:
                if args.max_leaves > 0:
                    with open(nwk_path) as fh:
                        nwk_text = fh.read()
                    n_leaves = nwk_text.count(',') + 1
                    if n_leaves > args.max_leaves:
                        continue
                if args.max_cols > 0:
                    # n_cols ≈ length of first non-blank, non-comment seq line.
                    with open(sto_path) as fh:
                        n_cols = 0
                        for line in fh:
                            if line.startswith('#') or not line.strip() or line.startswith('//'):
                                continue
                            parts = line.split()
                            if len(parts) >= 2:
                                n_cols = len(parts[1])
                                break
                    if n_cols == 0 or n_cols > args.max_cols:
                        continue
            except Exception:
                continue
            kept.append(fs)
        print(f"[{time.time()-t0:.0f}s] Filter kept {len(kept)}/{len(all_families)} families.")
        all_families = kept
    if args.shuffle_seed >= 0:
        rng = np.random.default_rng(args.shuffle_seed)
        order = rng.permutation(len(all_families))
        all_families = [all_families[i] for i in order]
        print(f"  shuffled spec ({len(all_families)} families) with seed "
              f"{args.shuffle_seed}; first 5 after shuffle: "
              f"{[f['family'] for f in all_families[:5]]}")
    fam_start = args.family_offset
    fam_end = fam_start + args.n_families if args.n_families > 0 else None
    selected = all_families[fam_start:fam_end]
    # Carve out the last --val-n entries as the held-out validation pool.
    # Disjoint from the breadth-first training sampler.
    val_n = max(0, int(args.val_n))
    if val_n > 0 and val_n < len(selected):
        families_spec = selected[:-val_n]
        val_families_spec = selected[-val_n:]
    else:
        families_spec = selected
        val_families_spec = []
    pfam_dir = os.path.expanduser(spec['pfam_dir'])
    tree_dir = os.path.expanduser(spec['tree_dir'])

    n_total = len(families_spec)
    if n_total == 0:
        sys.exit("No families in spec.")
    if args.batch_size > n_total:
        print(f"WARN: batch_size {args.batch_size} > n_total {n_total}; "
              f"clamping to {n_total}.")
        args.batch_size = n_total
    fam_names = [fs['family'] for fs in families_spec]
    val_fam_names = [fs['family'] for fs in val_families_spec]
    print(f"  train pool: {n_total} families; val pool: {len(val_families_spec)} "
          f"(disjoint, evaluated every --val-every-k={args.val_every_k} iters)")

    print(f"[{time.time()-t0:.0f}s] {n_total} train families in pool. "
          f"Starting SVI-VBEM with LAZY per-minibatch family loading "
          f"(batch={args.batch_size}, n_iter={args.n_iter}, "
          f"n_inner={args.n_inner}, tau={args.tau}, kappa={args.kappa})...")

    # Lazy provider — load (tree + Felsenstein) only when the sampler picks
    # the family. Breadth-first means each family is loaded at most once
    # per epoch (~n_total/batch_size iterations), so caching across the run
    # is generally unnecessary. We do skip-and-resample if a family fails
    # to load (missing tree, malformed MSA, etc.) to avoid corrupting the
    # batch with stale stats.
    skipped_families = set()
    def _provider_from_spec_list(spec_list):
        def provider(idx):
            fspec = spec_list[idx]
            fam = fspec['family']
            try:
                msa_int, tree_node, C = load_train_pfam_family(
                    fspec, pfam_dir, tree_dir)
            except Exception as e:
                skipped_families.add(fam)
                raise RuntimeError(f"load failed for {fam}: {e}")
            leaf_names = sorted(msa_int.keys())
            present_arr = np.zeros((len(leaf_names), C), dtype=np.int32)
            for i, name in enumerate(leaf_names):
                present_arr[i] = (msa_int[name] >= 0).astype(np.int32)
            binary_tree = build_binary_tree_from_node(tree_node)
            bt_present = np.zeros((binary_tree.num_leaves, C), dtype=np.int32)
            for li, lname in enumerate(binary_tree.leaf_names):
                if lname in leaf_names:
                    bt_present[li] = present_arr[leaf_names.index(lname)]
            sub_LL_per_class, _ = compute_sub_LL_per_class_per_column(
                tree_node, msa_int, leaf_names, held_out=None,
                class_Qs=class_Qs, class_pis=class_pis)
            return (binary_tree, bt_present, sub_LL_per_class)
        return provider

    family_provider = _provider_from_spec_list(families_spec)
    val_provider = (_provider_from_spec_list(val_families_spec)
                    if val_families_spec else None)

    from tkfmixdom.jax.train.tree_vbem import (
        serialize_ema_stats, serialize_sampler_state,
    )
    n_dom = params0['dom_ins'].shape[0]

    def _save_params_npz(params, path):
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

    def _save_full_state_npz(params, path, ema_stats, sampler, it, history):
        """Like _save_params_npz but also saves the EMA accumulator,
        sampler state, current iter and history JSON so a paused run
        can resume without discarding all batches up to `it`.

        Mirrors SVI-BW's _save_checkpoint pattern in train_pfam.py."""
        save = {
            'main_ins': float(params['main_ins']),
            'main_del': float(params['main_del']),
            'dom_ins': np.asarray(params['dom_ins']),
            'dom_del': np.asarray(params['dom_del']),
            'dom_weights': np.asarray(params['dom_weights']),
            'frag_weights': np.asarray(params['frag_weights']),
            'ext_rates': np.asarray(params['ext_rates']),
            'classdist': np.asarray(params['classdist']),
            'class_pis': np.asarray(params['class_pis']),
            'class_S_exch': np.asarray(params['class_S_exch']),
            'svi_iter': np.int32(it),
            'svi_history_json': np.array(json.dumps(history)),
        }
        save.update(serialize_ema_stats(ema_stats, n_dom))
        save.update(serialize_sampler_state(sampler))
        np.savez_compressed(path, **save)

    ckpt_dir = None
    if args.checkpoint:
        ckpt_dir = Path(args.out).with_suffix('')
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Persist family-name table once for downstream ID resolution.
        with open(ckpt_dir / 'fam_names.json', 'w') as fh:
            json.dump(fam_names, fh)

    def _iter_callback(it, params_after_M, history,
                        ema_stats=None, sampler=None):
        if ckpt_dir is None:
            return
        ckpt_path = ckpt_dir / f"iter{it:04d}.npz"
        # Save full state when available (new path), fall back to params-only
        # for back-compat with the legacy 3-arg callback signature.
        if ema_stats is not None and sampler is not None:
            _save_full_state_npz(params_after_M, str(ckpt_path),
                                  ema_stats, sampler, it, history)
        else:
            _save_params_npz(params_after_M, str(ckpt_path))
        with open(ckpt_dir / 'history.json', 'w') as fh:
            json.dump(history, fh, indent=2)

    final_params, history = svi_vbem_train(
        family_provider, params0, n_total_families=n_total,
        batch_size=args.batch_size, n_iter=args.n_iter, n_inner=args.n_inner,
        lr=args.lr, tau=args.tau, kappa=args.kappa,
        sampler_seed=args.sampler_seed,
        verbose=True, iter_callback=_iter_callback,
        val_provider=val_provider,
        val_n=len(val_families_spec),
        val_every_k=args.val_every_k,
        val_n_inner=(args.val_n_inner or args.n_inner),
        init_ema_stats=init_ema,
        init_sampler_state=init_sampler_state,
        start_iter=init_iter,
        init_history=init_history)

    final_npz = Path(args.out).with_suffix('.npz')
    _save_params_npz(final_params, str(final_npz))

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
                'spec': args.spec,
                'n_families': n_total,
                'batch_size': args.batch_size,
                'n_iter': args.n_iter,
                'n_inner': args.n_inner,
                'lr': args.lr,
                'tau': args.tau,
                'kappa': args.kappa,
                'sampler_seed': args.sampler_seed,
            },
        }, f, indent=2)

    print(f"[{time.time()-t0:.0f}s] Done. History saved to {args.out}")
    print(f"  Final params saved to {final_npz}")


if __name__ == '__main__':
    main()
