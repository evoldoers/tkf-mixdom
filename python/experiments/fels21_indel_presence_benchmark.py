#!/usr/bin/env python3
"""§5 Re-evaluation of fels21 (gap-augmented LG08) on indel-presence F1,
the same metric used by VarAnc and Fitch parsimony.

Approach:
  - Load the pre-trained 21x21 GTR (CherryML on 100 Pfam families,
    saved in pfam/fels21_fitted.npz; Q21, pi21).
  - For each family + held-out leaf in a unified-test spec, run
    Felsenstein peeling on the leaf-removed tree using the 21-state
    model. At the parent-of-held-out node, extract the 21-state
    posterior per column.
  - Compute p_present = 1 - posterior[col, gap=20] per column.
  - Compute F1 against the ground-truth presence pattern of the
    held-out leaf (1 = residue, 0 = gap).

This makes fels21 directly comparable to VarAnc/Fitch on
indel-presence F1.

Note: the standard fels21 reconstructs at the ROOT after re-rooting
at the held-out's parent. We follow the same convention here.

Outputs JSON in the same format as varanc_presence_*_test.json so
it can be aggregated alongside.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
import json
import time
import numpy as np
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.core.protein_gap import (
    load_fels21_model,
    reconstruct_root_gap,
)
from tkfmixdom.jax.util.io import parse_newick


_AA_TO_INT = {c: i for i, c in enumerate('ACDEFGHIKLMNPQRSTVWY')}


def _reroot_at_node(tree, target):
    """Re-root the tree at the parent of target leaf (so target's parent
    becomes the new root). Returns the new root TreeNode.

    Adapted from protein_gap.reconstruct_held_out_fels21.
    """
    # Walk up from target, collecting the path and reversing parent pointers.
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = cur.parent
    # path = [target, parent, grandparent, ..., root]
    new_root = path[-1]  # original root
    for i in range(len(path) - 1, 0, -1):
        ch = path[i - 1]
        par = path[i]
        # Reverse: par becomes child of ch.
        if par in ch.children:
            continue
        # Detach ch from par (if it was a child) and attach par to ch.
        try:
            par.children.remove(ch)
        except ValueError:
            pass
        ch.children.append(par)
        par.parent = ch
        # Edge length: just keep ch's branch_length.
    new_root = path[0].parent  # parent of held-out target
    return new_root


def f1_from_p_present(p_present, gt_present, threshold=0.5):
    """Compute F1 of binary 'present' calls against ground truth."""
    p_pred = (np.asarray(p_present) > threshold).astype(np.int32)
    gt = np.asarray(gt_present, dtype=np.int32)
    tp = int(((p_pred == 1) & (gt == 1)).sum())
    fp = int(((p_pred == 1) & (gt == 0)).sum())
    fn = int(((p_pred == 0) & (gt == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0, 0.0, 0.0
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


def msa_to_int21_dict(msa_strings):
    """Convert MSA dict {name: aligned_string} to int21 dict (gap=20)."""
    L = len(next(iter(msa_strings.values())))
    out = {}
    for name, seq in msa_strings.items():
        arr = np.full(L, 20, dtype=np.int32)
        for col, ch in enumerate(seq):
            if ch in '-.~':
                arr[col] = 20
            else:
                idx = _AA_TO_INT.get(ch.upper(), -1)
                arr[col] = idx if 0 <= idx < 20 else 20
        out[name] = arr
    return out


def load_msa_from_pfam(pfam_dir, family):
    """Load a Pfam family's stockholm MSA as {name: aligned_string}."""
    path = os.path.join(pfam_dir, family + '.sto')
    if not os.path.isfile(path):
        path = os.path.join(pfam_dir, family + '.fasta')
    msa = {}
    if path.endswith('.sto'):
        with open(path) as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    name, seq = parts
                    msa[name] = msa.get(name, '') + seq
    else:
        with open(path) as f:
            name = None; chunks = []
            for line in f:
                line = line.rstrip()
                if line.startswith('>'):
                    if name is not None:
                        msa[name] = ''.join(chunks)
                    name = line[1:]; chunks = []
                else:
                    chunks.append(line)
            if name is not None:
                msa[name] = ''.join(chunks)
    return msa


def process_family(fspec, Q21, pi21, pfam_dir, tree_dir):
    """Run fels21 indel-presence reconstruction for one family."""
    fam = fspec['family']
    held_out = fspec['held_out']

    msa_strings = load_msa_from_pfam(pfam_dir, fam)
    if held_out not in msa_strings:
        return None
    L = len(next(iter(msa_strings.values())))

    # Parse tree.
    tree_path = os.path.join(tree_dir, fam + '.nwk')
    if not os.path.isfile(tree_path):
        return None
    with open(tree_path) as f:
        nwk = f.read().strip()
    tree = parse_newick(nwk)

    # Find the target leaf and re-root at its parent (drop the target).
    target_leaf = None
    for node in tree.preorder():
        if node.is_leaf and node.name == held_out:
            target_leaf = node; break
    if target_leaf is None or target_leaf.parent is None:
        return None
    rerooted = _reroot_at_node(tree, target_leaf)
    rerooted.children = [c for c in rerooted.children
                         if not (c.is_leaf and c.name == held_out)]

    # Build leaf int21 dict for non-held-out leaves.
    leaf_int21 = msa_to_int21_dict({k: v for k, v in msa_strings.items()
                                       if k != held_out})

    # Ground-truth presence from the held-out leaf.
    gt_seq = msa_strings[held_out]
    gt_present = np.array([1 if ch not in '-.~' else 0 for ch in gt_seq],
                           dtype=np.int32)

    t0 = time.time()
    try:
        _, posteriors = reconstruct_root_gap(
            rerooted, leaf_int21, jnp.asarray(Q21), jnp.asarray(pi21))
        posteriors = np.asarray(posteriors)  # (L, 21)
    except Exception as e:
        return {'family': fam, 'held_out': held_out, 'n_cols': L,
                'error': str(e)[:200]}
    t_elapsed = time.time() - t0

    # p_present = 1 - p[gap]
    p_present = 1.0 - posteriors[:, 20]
    f1, prec, rec = f1_from_p_present(p_present, gt_present)
    return {
        'family': fam, 'held_out': held_out, 'n_cols': int(L),
        'gt_present': gt_present.tolist(),
        'p_present': p_present.tolist(),
        'f1': float(f1), 'precision': float(prec), 'recall': float(rec),
        'time': float(t_elapsed),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True,
                    help='unified_*_test_spec.json')
    p.add_argument('--out', required=True)
    p.add_argument('--n-families', type=int, default=0,
                    help='0=all')
    p.add_argument('--model-path', type=str,
                    default=None,
                    help='fels21 fitted model npz; default pfam/fels21_fitted.npz')
    args = p.parse_args()

    Q21, pi21 = load_fels21_model(args.model_path)
    print(f'Loaded fels21 model: Q21 shape {Q21.shape}, pi21 shape {pi21.shape}')

    with open(args.spec) as f:
        spec = json.load(f)
    families = spec['families']
    pfam_dir = os.path.expanduser(spec.get('pfam_dir', '~/bio-datasets/data/pfam-seed'))
    tree_dir = os.path.expanduser(spec.get('tree_dir',
                                            '~/bio-datasets/data/pfam-seed/trees'))
    if args.n_families > 0:
        families = families[:args.n_families]
    print(f'Spec {args.spec}: {len(families)} families')

    print(f'{"family":<12} {"L":>4} | {"F1":>6} {"prec":>6} {"rec":>6} | {"time":>5}')
    print('-' * 50)

    results = []
    for fi, fspec in enumerate(families):
        try:
            res = process_family(fspec, Q21, pi21, pfam_dir, tree_dir)
        except Exception as e:
            print(f'[{fi+1}/{len(families)}] {fspec["family"]:<12} ERR: {type(e).__name__}: {e}',
                  flush=True)
            continue
        if res is None:
            print(f'[{fi+1}/{len(families)}] {fspec["family"]:<12} SKIP', flush=True)
            continue
        if 'error' in res:
            print(f'[{fi+1}/{len(families)}] {fspec["family"]:<12} '
                  f'ERR: {res["error"]}', flush=True)
            continue
        results.append(res)
        print(f'[{fi+1:>3}/{len(families)}] {res["family"]:<12} '
              f'{res["n_cols"]:>4} | {res["f1"]:>6.3f} {res["precision"]:>6.3f} '
              f'{res["recall"]:>6.3f} | {res["time"]:>5.1f}s', flush=True)
        # Save partial after every 10 fams.
        if (fi + 1) % 10 == 0:
            with open(args.out, 'w') as f:
                json.dump({'spec': args.spec, 'n_total': len(families),
                            'n_complete': len(results), 'partial': True,
                            'results': results}, f, indent=2)

    # Final save.
    out = {'spec': args.spec, 'n_total': len(families), 'partial': False,
           'results': results}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    if results:
        f1s = [r['f1'] for r in results]
        print(f'\nfels21 indel-presence F1: n={len(results)}, '
              f'mean={np.mean(f1s):.4f}, median={np.median(f1s):.4f}')
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
