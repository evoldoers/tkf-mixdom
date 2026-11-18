#!/usr/bin/env python3
"""Combine per-component TKF92 scores of the K=20 mixture into mixture log-likelihoods.

For each (split, convention) with all K per-pair files present (experiments/test_ll/k20/<split>_pc<0|1>_comp<k>.npz,
written by score_matchaligned_val.py with PERPAIR_OUT), computes
  family-level mixture (the fitted model: one component per family):
      sum_f log sum_k w_k prod_{pairs in f} P_k(pair)
  pair-level mixture (a looser bound, each pair free to pick its component):
      sum_pairs log sum_k w_k P_k(pair)
and writes LL/pair lines where revision_tables.py / paper_tables.py look for them:
  val:  experiments/ordsum_val/pathcond_tkf92_K20.log ("LL/pair=..."), experiments/ordsum_val/score_tkf92_K20.log ("LL/pair  = ...")
  test: experiments/test_ll/pathcond_tkf92_K20.log,     experiments/test_ll/ordsum_tkf92_K20.log  ("LL/pair  = ...")
The pair-level numbers go to the same files with suffix _perpair.
"""
import glob, json, os, sys, numpy as np
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.dirname(os.path.dirname(HERE)); os.chdir(PY)
SUB = sys.argv[1] if len(sys.argv) > 1 else 'k20'          # component directory under experiments/test_ll
TAG = sys.argv[2] if len(sys.argv) > 2 else 'tkf92_K20'     # output log stem
K20 = os.path.join(HERE, SUB); W = np.log(np.asarray(json.load(open(f'{K20}/weights.json'))['weights']))
K = len(W)
for split in ('val', 'test'):
    for pc in (1, 0):
        files = [f'{K20}/{split}_pc{pc}_comp{k}.npz' for k in range(K)]
        if not all(os.path.exists(f) for f in files):
            if not (split == 'test' and pc == 1):
                print(f'{split} pc={pc}: {sum(os.path.exists(f) for f in files)}/{K} components, skipping')
            continue
        zs = [np.load(f, allow_pickle=True) for f in files]
        ll = np.stack([z['ll'] for z in zs])            # (K, n_pairs), nan where a component failed on a pair
        fam = zs[0]['fam']; assert all((z['fam'] == fam).all() for z in zs)
        ok = np.isfinite(ll).all(0)
        ll = ll[:, ok]; fam = fam[ok]; n = int(ok.sum())
        pair_mix = logsumexp(ll + W[:, None], axis=0).sum()
        fam_ids, inv = np.unique(fam, return_inverse=True)
        per_fam = np.zeros((K, len(fam_ids))); np.add.at(per_fam.T, inv, ll.T)   # sum of pair LLs per family, per component
        fam_mix = logsumexp(per_fam + W[:, None], axis=0).sum()
        best_comp = logsumexp(np.zeros(1)) * 0 + ll.max(0).sum()               # oracle: best component per pair (diagnostic)
        kind = 'pathcond' if pc else 'ordsum'
        if split == 'val':
            paths = {('pathcond', ''): f'experiments/ordsum_val/pathcond_{TAG}{{s}}.log', ('ordsum', ''): f'experiments/ordsum_val/score_{TAG}{{s}}.log'}
        else:
            paths = {('pathcond', ''): f'experiments/test_ll/pathcond_{TAG}{{s}}.log', ('ordsum', ''): f'experiments/test_ll/ordsum_{TAG}{{s}}.log'}
        for suffix, val in (('', fam_mix), ('_perpair', pair_mix)):
            p = paths[(kind, '')].format(s=suffix)
            with open(p, 'w') as fh:
                fh.write(f'[{TAG}{suffix}] {split} {kind}: K={K} components, {n} pairs, {len(fam_ids)} families\n')
                fh.write(f'  total_ll = {val:.4f}\n  LL/pair  = {val / n:.6f}\nLL/pair={val / n:.6f}\n')
        print(f'{split} {kind}: n={n} fams={len(fam_ids)} | family-mixture LL/pair {fam_mix / n:.4f} | pair-mixture {pair_mix / n:.4f} | best-component oracle {best_comp / n:.4f}')
