#!/usr/bin/env python3
"""Build the test-split equivalents of unified_benchmark_spec.json (short)
and unified_benchmark_long_spec.json (long).

Background. Both `unified_benchmark_spec.json` and
`unified_benchmark_long_spec.json` were drawn from the **val** split of
~/bio-datasets/data/pfam/seed/splits/v1.json. The SVI-BW training
checkpoint name `svi_bw_d3f1_postfix_best_val.npz` confirms the same val
set was used for both early-stopping and benchmark evaluation — i.e.
training-time model selection and final reporting were on the same set.
This is data-leakage contamination and *every* reconstruction benchmark
JSON named `*_unified_short*.json` or `*_unified_long*.json` is
contaminated.

Mitigation: this script builds fresh `_test_spec.json` files drawn
strictly from the test split (3,384 fams; 3,191 with trees). Same
construction rules as the original short/long specs:

  short — K (remaining leaves) ∈ [4, 48], n_cols ∈ [12, 100], 200 fams
  long  — K ∈ [4, 48], n_cols ∈ [80, 200], 200 fams

For each family the held-out leaf is the one with highest mean
phylogenetic distance to the other leaves (the "hardest" reconstruction
target — same convention as the original).

Usage:
    cd python && uv run python experiments/build_unified_test_specs.py
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT
from experiments.ancrec_benchmark import parse_sto


SPLITS_PATH = os.path.expanduser('~/bio-datasets/data/pfam/seed/splits/v1.json')
PFAM_DIR = os.path.expanduser('~/bio-datasets/data/pfam/seed')
TREE_DIR = os.path.expanduser('~/bio-datasets/data/pfam-seed/trees')


def collect_leaves(node, out=None):
    if out is None:
        out = []
    if not node.children:
        out.append(node)
    else:
        for c in node.children:
            collect_leaves(c, out)
    return out


def pairwise_distances(node):
    """Compute leaf-to-leaf path distances in a tree.

    Returns dict {(leaf_a_name, leaf_b_name): distance}. Walk the tree to
    build dist-from-root for each leaf, then use LCA distance via path-up
    intersections. For our use we only need mean(dist(leaf, others)) per
    leaf, so we compute a NxN distance matrix.
    """
    # Annotate each node with depth from the root.
    leaves = collect_leaves(node)
    n = len(leaves)
    leaf_idx = {id(L): i for i, L in enumerate(leaves)}

    # For each leaf, walk parents back to root; record (node_id, accumulated
    # dist from leaf). Then for any pair, LCA is the deepest shared id.
    paths = []
    for L in leaves:
        path = []
        cur = L
        d = 0.0
        while cur is not None:
            path.append((id(cur), d))
            if cur.parent is None:
                break
            d += float(cur.branch_length or 0.0)
            cur = cur.parent
        paths.append(path)

    dist = np.zeros((n, n))
    for i in range(n):
        path_i = dict(paths[i])
        for j in range(i + 1, n):
            for nid, dj in paths[j]:
                if nid in path_i:
                    di = path_i[nid]
                    dist[i, j] = di + dj
                    dist[j, i] = dist[i, j]
                    break
    return dist, [L.name for L in leaves]


def family_entry_xhard(fam, n_cols_range,
                        median_div_min=2.5,
                        held_min_nn_min=2.0, held_gap_frac_min=0.35,
                        drop_nearest_k=2, leaf_cap=49,
                        subsample_seed=None):
    """Extra-hard variant: stricter thresholds AND drops the K nearest
    leaves to the held-out target (further isolating it).

    Default thresholds correspond to ~p75 of the test pool on each axis,
    making the resulting set substantially harder than `hard_test`. The
    drop_nearest_k=2 row-removal step takes the picked held-out leaf,
    finds its 2 nearest leaves by tree distance, and removes them from
    `remaining` (so when methods condition on `remaining`, the held-out
    leaf is reconstructed without its closest evolutionary neighbours).

    leaf_cap is the maximum total number of leaves in the entry
    (held_out + dropped + remaining). If the family has more leaves
    than leaf_cap, the *remaining* set is randomly subsampled (using
    subsample_seed) so that total leaves = leaf_cap. The held-out
    target and the K dropped nearest neighbours are always retained.

    Returns None if no eligible leaf exists in the family."""
    sto_path = os.path.join(PFAM_DIR, f'{fam}.sto')
    nwk_path = os.path.join(TREE_DIR, f'{fam}.nwk')
    if not (os.path.exists(nwk_path) and os.path.exists(sto_path)):
        return None
    try:
        seqs = parse_sto(sto_path)
    except Exception:
        return None
    if not seqs:
        return None
    n_cols = max(len(s) for s in seqs.values())
    if not (n_cols_range[0] <= n_cols <= n_cols_range[1]):
        return None
    try:
        with open(nwk_path) as f:
            tree = parse_newick(f.read().strip())
    except Exception:
        return None
    leaves = collect_leaves(tree)
    leaves = [L for L in leaves if L.name in seqs]
    # Need at least held + K dropped + 4 remaining = 5 + drop_nearest_k.
    # No upper bound; large families are subsampled below.
    if len(leaves) < 5 + drop_nearest_k:
        return None
    dist, leaf_names = pairwise_distances(tree)
    name_to_idx = {n: i for i, n in enumerate(leaf_names)}
    keep_idx = [name_to_idx[L.name] for L in leaves if L.name in name_to_idx]
    if len(keep_idx) != len(leaves):
        return None
    sub_dist = dist[np.ix_(keep_idx, keep_idx)]
    sub_names = [leaf_names[i] for i in keep_idx]
    n = len(sub_names)
    if n < 5 + drop_nearest_k:
        return None
    median_div = float(np.median(sub_dist[sub_dist > 0]))
    if median_div < median_div_min:
        return None
    nn_dist = np.zeros(n)
    gap_fracs = np.zeros(n)
    for i in range(n):
        others = np.delete(sub_dist[i], i)
        nn_dist[i] = others.min()
        s = seqs[sub_names[i]]
        n_res = sum(1 for ch in s if ch in AA_TO_INT and AA_TO_INT[ch] < 20)
        gap_fracs[i] = 1.0 - n_res / max(n_cols, 1)
    eligible = (nn_dist >= held_min_nn_min) & (gap_fracs >= held_gap_frac_min)
    if not eligible.any():
        return None
    eligible_idx = np.where(eligible)[0]
    held_local = int(eligible_idx[np.argmax(nn_dist[eligible_idx])])
    held_out = sub_names[held_local]
    held_seq = seqs[held_out]
    true_seq_int = []
    for ch in held_seq:
        if ch in AA_TO_INT and AA_TO_INT[ch] < 20:
            true_seq_int.append(int(AA_TO_INT[ch]))
    if len(true_seq_int) < 4:
        return None
    # Drop the K nearest leaves to held-out from `remaining` to further
    # isolate the prediction target.
    other_idx = [i for i in range(n) if i != held_local]
    other_idx_sorted = sorted(other_idx, key=lambda i: sub_dist[held_local, i])
    drop_set = set(other_idx_sorted[:drop_nearest_k])
    remaining_idx = [i for i in range(n)
                     if i != held_local and i not in drop_set]
    dropped = [sub_names[i] for i in drop_set]
    # Subsample `remaining_idx` if total leaf count exceeds leaf_cap.
    # max_remaining = leaf_cap - 1 (held) - drop_nearest_k (dropped).
    max_remaining = leaf_cap - 1 - drop_nearest_k
    n_subsampled_from = len(remaining_idx)
    if max_remaining >= 4 and len(remaining_idx) > max_remaining:
        if subsample_seed is None:
            rng = np.random.default_rng()
        else:
            # Family-specific seed so each family gets a deterministic
            # but distinct subsample.
            rng = np.random.default_rng(
                hash((subsample_seed, fam)) & 0xFFFFFFFF)
        chosen = sorted(rng.choice(remaining_idx,
                                    size=max_remaining,
                                    replace=False).tolist())
        remaining_idx = chosen
    remaining = [sub_names[i] for i in remaining_idx]
    if len(remaining) < 4:
        return None
    # Re-compute min-NN distance to *remaining-only* set (post row-drop
    # AND post-subsample) for entry metadata.
    post_drop_min_nn = float(min(sub_dist[held_local, j]
                                  for j in remaining_idx))
    return {
        'family': fam,
        'held_out': held_out,
        'remaining': remaining,
        'dropped_neighbours': dropped,
        'true_seq': true_seq_int,
        'true_len': len(true_seq_int),
        'n_cols': int(n_cols),
        'K': int(len(remaining)),
        'mean_dist': float(np.mean(sub_dist[held_local])),
        'median_div': float(median_div),
        'min_nn_dist_pre_drop': float(nn_dist[held_local]),
        'min_nn_dist_post_drop': post_drop_min_nn,
        'gap_frac': float(gap_fracs[held_local]),
        'n_leaves_total': int(n),
        'n_subsampled_from': int(n_subsampled_from),
        'leaf_cap': int(leaf_cap),
    }


def family_entry_hard(fam, n_cols_range,
                       median_div_min=2.086,
                       held_min_nn_min=1.5, held_gap_frac_min=0.25):
    """Hard variant: divergent family + gappy + isolated held-out leaf.

    Family filter: median pairwise tree distance ≥ median_div_min
        (default 2.086, the test-pool p50).
    Held-out-leaf filter: pick the leaf maximizing min-NN tree distance
        among leaves whose gap fraction is ≥ held_gap_frac_min (default
        0.30 — covers ~30% of the test pool's gappiest leaves) AND whose
        min-NN distance is ≥ held_min_nn_min (default 1.5 — well above
        the test-pool median of 1.81).

    Returns None if no eligible leaf exists in the family."""
    sto_path = os.path.join(PFAM_DIR, f'{fam}.sto')
    nwk_path = os.path.join(TREE_DIR, f'{fam}.nwk')
    if not (os.path.exists(nwk_path) and os.path.exists(sto_path)):
        return None
    try:
        seqs = parse_sto(sto_path)
    except Exception:
        return None
    if not seqs:
        return None
    n_cols = max(len(s) for s in seqs.values())
    if not (n_cols_range[0] <= n_cols <= n_cols_range[1]):
        return None
    try:
        with open(nwk_path) as f:
            tree = parse_newick(f.read().strip())
    except Exception:
        return None
    leaves = collect_leaves(tree)
    leaves = [L for L in leaves if L.name in seqs]
    if not (5 <= len(leaves) <= 49):
        return None
    dist, leaf_names = pairwise_distances(tree)
    name_to_idx = {n: i for i, n in enumerate(leaf_names)}
    keep_idx = [name_to_idx[L.name] for L in leaves if L.name in name_to_idx]
    if len(keep_idx) != len(leaves):
        return None
    sub_dist = dist[np.ix_(keep_idx, keep_idx)]
    sub_names = [leaf_names[i] for i in keep_idx]
    n = len(sub_names)
    if n < 5:
        return None
    median_div = float(np.median(sub_dist[sub_dist > 0]))
    if median_div < median_div_min:
        return None
    # Per-leaf min-NN distance and gap fraction.
    nn_dist = np.zeros(n)
    gap_fracs = np.zeros(n)
    for i in range(n):
        others = np.delete(sub_dist[i], i)
        nn_dist[i] = others.min()
        s = seqs[sub_names[i]]
        n_res = sum(1 for ch in s if ch in AA_TO_INT and AA_TO_INT[ch] < 20)
        gap_fracs[i] = 1.0 - n_res / max(n_cols, 1)
    eligible = (nn_dist >= held_min_nn_min) & (gap_fracs >= held_gap_frac_min)
    if not eligible.any():
        return None
    # Pick max-min-NN distance among eligible (gappy + isolated leaves).
    eligible_idx = np.where(eligible)[0]
    held_local = int(eligible_idx[np.argmax(nn_dist[eligible_idx])])
    held_out = sub_names[held_local]
    held_seq = seqs[held_out]
    true_seq_int = []
    for ch in held_seq:
        if ch in AA_TO_INT and AA_TO_INT[ch] < 20:
            true_seq_int.append(int(AA_TO_INT[ch]))
    if len(true_seq_int) < 4:
        return None
    remaining = [s for s in sub_names if s != held_out]
    return {
        'family': fam,
        'held_out': held_out,
        'remaining': remaining,
        'true_seq': true_seq_int,
        'true_len': len(true_seq_int),
        'n_cols': int(n_cols),
        'K': int(len(remaining)),
        'mean_dist': float(np.mean(sub_dist[held_local])),
        'median_div': float(median_div),
        'min_nn_dist': float(nn_dist[held_local]),
        'gap_frac': float(gap_fracs[held_local]),
    }


def family_entry(fam, n_cols_range):
    """Build one spec entry for a family, or return None if it doesn't
    fit the K/n_cols filter."""
    sto_path = os.path.join(PFAM_DIR, f'{fam}.sto')
    nwk_path = os.path.join(TREE_DIR, f'{fam}.nwk')
    if not (os.path.exists(sto_path) and os.path.exists(nwk_path)):
        return None
    try:
        seqs = parse_sto(sto_path)
    except Exception:
        return None
    if not seqs:
        return None
    n_cols = max(len(s) for s in seqs.values())
    if not (n_cols_range[0] <= n_cols <= n_cols_range[1]):
        return None
    try:
        with open(nwk_path) as f:
            tree = parse_newick(f.read().strip())
    except Exception:
        return None
    # Filter to leaves present in MSA.
    leaves = collect_leaves(tree)
    leaves = [L for L in leaves if L.name in seqs]
    if not (5 <= len(leaves) <= 49):  # K = len(leaves) - 1, want 4..48
        return None

    # Re-prune the tree to MSA-present leaves so the dist matrix matches.
    # For simplicity, just compute distances on the full tree and restrict
    # to MSA-present subset.
    dist, leaf_names = pairwise_distances(tree)
    name_to_idx = {n: i for i, n in enumerate(leaf_names)}
    keep_idx = [name_to_idx[L.name] for L in leaves if L.name in name_to_idx]
    if len(keep_idx) != len(leaves):
        return None
    sub_dist = dist[np.ix_(keep_idx, keep_idx)]
    sub_names = [leaf_names[i] for i in keep_idx]
    n = len(sub_names)
    if n < 5:
        return None
    mean_to_others = sub_dist.sum(axis=1) / max(n - 1, 1)
    held_out_local = int(np.argmax(mean_to_others))
    held_out = sub_names[held_out_local]
    held_seq = seqs[held_out]
    true_seq_int = []
    for ch in held_seq:
        if ch in AA_TO_INT and AA_TO_INT[ch] < 20:
            true_seq_int.append(int(AA_TO_INT[ch]))
    if len(true_seq_int) < 4:
        return None
    remaining = [s for s in sub_names if s != held_out]
    return {
        'family': fam,
        'held_out': held_out,
        'remaining': remaining,
        'true_seq': true_seq_int,
        'true_len': len(true_seq_int),
        'n_cols': int(n_cols),
        'K': int(len(remaining)),
        'mean_dist': float(mean_to_others[held_out_local]),
    }


def build_spec(test_fams, n_cols_range, target_n, seed, hard=False,
                xhard=False, xhard_kwargs=None):
    rng = random.Random(seed)
    rng.shuffle(test_fams)
    entries = []
    if xhard:
        kw = xhard_kwargs or {}
        def builder(fam, n_cols_range):
            return family_entry_xhard(fam, n_cols_range, **kw)
    elif hard:
        builder = family_entry_hard
    else:
        builder = family_entry
    for fam in test_fams:
        if len(entries) >= target_n:
            break
        e = builder(fam, n_cols_range)
        if e is not None:
            entries.append(e)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-short', type=str,
                        default='experiments/unified_benchmark_test_spec.json')
    parser.add_argument('--out-long', type=str,
                        default='experiments/unified_benchmark_long_test_spec.json')
    parser.add_argument('--out-hard', type=str,
                        default='experiments/unified_benchmark_hard_test_spec.json')
    parser.add_argument('--out-xhard', type=str,
                        default='experiments/unified_benchmark_xhard_test_spec.json')
    parser.add_argument('--n-fams', type=int, default=200)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--build-only', type=str, default='all',
                        choices=['all', 'short', 'long', 'hard', 'xhard'])
    parser.add_argument('--xhard-leaf-cap', type=int, default=49,
                        help='Max total leaves per xhard entry (subsamples '
                        'larger families).')
    parser.add_argument('--xhard-n-cols-max', type=int, default=200,
                        help='Max alignment columns for xhard entries.')
    parser.add_argument('--xhard-median-div-min', type=float, default=2.5)
    parser.add_argument('--xhard-min-nn-min', type=float, default=2.0)
    parser.add_argument('--xhard-gap-frac-min', type=float, default=0.35)
    parser.add_argument('--xhard-drop-nearest-k', type=int, default=2)
    args = parser.parse_args()

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    test = sorted(set(splits['test']))
    trees_avail = {p.stem for p in Path(TREE_DIR).glob('*.nwk')}
    test_with_trees = sorted(set(test) & trees_avail)
    print(f"Test split: {len(test)} fams, {len(test_with_trees)} with trees.")

    base = {
        'pfam_dir': '~/bio-datasets/data/pfam/seed',
        'tree_dir': '~/bio-datasets/data/pfam-seed/trees',
        'split': 'test (from ~/bio-datasets/data/pfam/seed/splits/v1.json)',
    }

    if args.build_only in ('all', 'short'):
        print(f"\nBuilding SHORT spec ({args.n_fams} fams; n_cols ∈ [12, 100])...")
        short_entries = build_spec(list(test_with_trees), (12, 100),
                                    args.n_fams, args.seed)
        print(f"  Got {len(short_entries)} entries.")
        short_spec = {
            **base,
            'description': 'Test-split unified-short reconstruction benchmark. '
            'Replaces the contaminated val-split unified_benchmark_spec.json. '
            'Held-out leaf = one with highest mean phylogenetic distance.',
            'n_families': len(short_entries),
            'families': short_entries,
        }
        with open(args.out_short, 'w') as f:
            json.dump(short_spec, f, indent=2)
        print(f"  Wrote {args.out_short}")

    if args.build_only in ('all', 'long'):
        print(f"\nBuilding LONG spec ({args.n_fams} fams; n_cols ∈ [80, 200])...")
        long_entries = build_spec(list(test_with_trees), (80, 200),
                                   args.n_fams, args.seed + 1)
        print(f"  Got {len(long_entries)} entries.")
        long_spec = {
            **base,
            'description': 'Test-split unified-long reconstruction benchmark '
            '(longer alignments, n_cols ∈ [80, 200]).',
            'n_families': len(long_entries),
            'families': long_entries,
        }
        with open(args.out_long, 'w') as f:
            json.dump(long_spec, f, indent=2)
        print(f"  Wrote {args.out_long}")

    if args.build_only in ('all', 'xhard'):
        print(f"\nBuilding XHARD spec (target {args.n_fams} fams; "
              f"n_cols ∈ [12, {args.xhard_n_cols_max}], median_div ≥ "
              f"{args.xhard_median_div_min}, held-out leaf min_NN ≥ "
              f"{args.xhard_min_nn_min} AND gap_frac ≥ "
              f"{args.xhard_gap_frac_min}, drop {args.xhard_drop_nearest_k} "
              f"nearest, leaf_cap {args.xhard_leaf_cap} (subsample if larger))...")
        xhard_kwargs = {
            'median_div_min': args.xhard_median_div_min,
            'held_min_nn_min': args.xhard_min_nn_min,
            'held_gap_frac_min': args.xhard_gap_frac_min,
            'drop_nearest_k': args.xhard_drop_nearest_k,
            'leaf_cap': args.xhard_leaf_cap,
            'subsample_seed': args.seed + 3,
        }
        xhard_entries = build_spec(list(test_with_trees),
                                    (12, args.xhard_n_cols_max),
                                    args.n_fams, args.seed + 3, xhard=True,
                                    xhard_kwargs=xhard_kwargs)
        print(f"  Got {len(xhard_entries)} entries.")
        n_subsampled = sum(1 for e in xhard_entries
                           if e.get('n_leaves_total', 0) > e.get('leaf_cap', 0))
        print(f"  ({n_subsampled} entries had subsampled `remaining` from "
              f"larger families.)")
        xhard_spec = {
            **base,
            'description': 'Test-split unified-XHARD reconstruction benchmark. '
            f'Family filter: median pairwise tree distance ≥ '
            f'{args.xhard_median_div_min}. '
            f'Held-out-leaf filter: pick the leaf maximizing min-NN tree '
            f'distance among leaves with gap fraction ≥ '
            f'{args.xhard_gap_frac_min} AND min-NN distance ≥ '
            f'{args.xhard_min_nn_min}. '
            f'Then DROP the {args.xhard_drop_nearest_k} nearest leaves to the '
            f'held-out target from `remaining` (further isolating the target). '
            f'For families with > {args.xhard_leaf_cap} leaves the `remaining` '
            f'set is randomly subsampled (per-family deterministic seed) so '
            f'total leaves do not exceed {args.xhard_leaf_cap}. '
            f'n_cols ∈ [12, {args.xhard_n_cols_max}]. '
            'Designed to stress-test long-branch + sparse-evidence + high-gap '
            'reconstruction together. Strictly harder than the hard_test '
            'spec on every axis.',
            'config': {
                'median_div_min': args.xhard_median_div_min,
                'held_min_nn_min': args.xhard_min_nn_min,
                'held_gap_frac_min': args.xhard_gap_frac_min,
                'drop_nearest_k': args.xhard_drop_nearest_k,
                'leaf_cap': args.xhard_leaf_cap,
                'n_cols_max': args.xhard_n_cols_max,
            },
            'n_families': len(xhard_entries),
            'families': xhard_entries,
        }
        with open(args.out_xhard, 'w') as f:
            json.dump(xhard_spec, f, indent=2)
        print(f"  Wrote {args.out_xhard}")

    if args.build_only in ('all', 'hard'):
        print(f"\nBuilding HARD spec ({args.n_fams} fams; n_cols ∈ [12, 200], "
              f"median_div ≥ 2.086, held-out leaf min_NN ≥ 1.5 AND gap_frac ≥ 0.25)...")
        hard_entries = build_spec(list(test_with_trees), (12, 200),
                                   args.n_fams, args.seed + 2, hard=True)
        print(f"  Got {len(hard_entries)} entries.")
        hard_spec = {
            **base,
            'description': 'Test-split unified-HARD reconstruction benchmark. '
            'Family filter: median pairwise tree distance ≥ test-pool p50 '
            '(2.086). Held-out-leaf filter: pick the leaf maximizing min-NN '
            'tree distance among leaves with gap fraction ≥ 0.25 AND min-NN '
            'distance ≥ 1.5. Designed to stress long-branch reconstruction '
            'AND high-gap presence/absence prediction simultaneously.',
            'n_families': len(hard_entries),
            'families': hard_entries,
        }
        with open(args.out_hard, 'w') as f:
            json.dump(hard_spec, f, indent=2)
        print(f"  Wrote {args.out_hard}")


if __name__ == '__main__':
    main()
