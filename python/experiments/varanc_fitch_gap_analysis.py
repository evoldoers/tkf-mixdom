"""Post-hoc analysis of varanc vs Fitch on the unified_short/long benchmarks.

For each entry: compute features (max branch length, mean coverage,
tree depth, n_leaves, n_cols), then bucket the entries and report
mean F1 per method per bucket.

The goal: identify regimes where Fitch saturates (Pfam-easy) and where
the methods diverge.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yam/tkf-mixdom/python')
from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT


TREE_DIR = os.path.expanduser("~/bio-datasets/data/pfam-seed/trees")
PFAM_DIR = os.path.expanduser("~/bio-datasets/data/pfam/seed")


def parse_sto(path):
    seqs = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('#') or line.startswith('//') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                name, seq = parts[0], parts[-1]
                seqs.setdefault(name, []).append(seq)
    return {n: ''.join(s) for n, s in seqs.items()}


def tree_features(tree):
    """Return (max_branch_len, mean_branch_len, depth, n_leaves)."""
    branch_lens = []
    leaves = []
    max_depth = [0]
    def visit(n, depth):
        if not n.children:
            leaves.append(n)
            max_depth[0] = max(max_depth[0], depth)
        else:
            for c in n.children:
                branch_lens.append(c.branch_length)
                visit(c, depth + 1)
    visit(tree, 0)
    return (max(branch_lens) if branch_lens else 0.0,
            float(np.mean(branch_lens)) if branch_lens else 0.0,
            max_depth[0],
            len(leaves))


def coverage_stats(msa, n_cols, leaf_names):
    """Per-column coverage = fraction of leaves Present at that column."""
    cov = np.zeros(n_cols)
    n = 0
    for name in leaf_names:
        if name not in msa:
            continue
        seq = msa[name]
        for j, ch in enumerate(seq[:n_cols]):
            if ch.upper() in AA_TO_INT and AA_TO_INT[ch.upper()] < 20:
                cov[j] += 1
        n += 1
    if n == 0:
        return 0.0, 0.0
    cov = cov / n
    # Mid-coverage = fraction of cols with 0.3 <= cov <= 0.7
    mid_frac = float(np.mean((cov >= 0.3) & (cov <= 0.7)))
    return float(cov.mean()), mid_frac


def analyze(json_path):
    print(f"\n=== {os.path.basename(json_path)} ===")
    with open(json_path) as f:
        d = json.load(f)
    results = d['results']
    print(f"  n entries: {len(results)}")

    rows = []
    for r in results:
        fam = r['family']
        held_out = r['held_out']
        n_cols = r['n_cols']

        # Load tree
        tp = os.path.join(TREE_DIR, f'{fam}.nwk')
        if not os.path.exists(tp):
            tp = os.path.join(TREE_DIR, f'{fam}.tree')
        if not os.path.exists(tp):
            continue
        with open(tp) as f:
            tree = parse_newick(f.read().strip())
        max_bl, mean_bl, depth, n_leaves = tree_features(tree)

        # Load MSA
        sto = os.path.join(PFAM_DIR, f'{fam}.sto')
        if not os.path.exists(sto):
            continue
        msa = parse_sto(sto)
        leaf_names = [n for n in msa.keys() if n != held_out]
        mean_cov, mid_frac = coverage_stats(msa, n_cols, leaf_names)

        # F1s
        m = r['methods']
        f1_var = m.get('varanc', {}).get('f1', np.nan)
        f1_fit = m.get('fitch', {}).get('f1', np.nan)
        f1_f21 = m.get('fels21', {}).get('f1', np.nan)

        rows.append({
            'fam': fam, 'n_cols': n_cols, 'n_leaves': n_leaves,
            'max_bl': max_bl, 'mean_bl': mean_bl, 'depth': depth,
            'mean_cov': mean_cov, 'mid_frac': mid_frac,
            'varanc': f1_var, 'fitch': f1_fit, 'fels21': f1_f21,
            'gap_vf': f1_var - f1_fit,
        })

    if not rows:
        print("  no rows!")
        return

    arr = {k: np.array([r[k] for r in rows]) for k in rows[0].keys() if k != 'fam'}

    print(f"\n  Overall (n={len(rows)}):")
    print(f"    varanc F1 mean = {arr['varanc'].mean():.4f}, fitch = {arr['fitch'].mean():.4f}, fels21 = {arr['fels21'].mean():.4f}")
    print(f"    varanc-fitch gap: mean = {arr['gap_vf'].mean():+.4f}, "
          f"std = {arr['gap_vf'].std():.4f}, "
          f"varanc>fitch: {int((arr['gap_vf']>0).sum())}, "
          f"=: {int((arr['gap_vf']==0).sum())}, "
          f"<: {int((arr['gap_vf']<0).sum())}")

    # Bucket by max branch length
    def bucket(arr_x, arr_y_dict, bins, label):
        print(f"\n  Bucketed by {label}:")
        print(f"    {'bucket':>20} {'n':>4} {'varanc':>7} {'fitch':>7} {'fels21':>7} {'gap_vf':>9}")
        for lo, hi in bins:
            mask = (arr_x >= lo) & (arr_x < hi)
            n = int(mask.sum())
            if n == 0: continue
            v = arr_y_dict['varanc'][mask].mean()
            f = arr_y_dict['fitch'][mask].mean()
            f21 = arr_y_dict['fels21'][mask].mean()
            gap = arr_y_dict['gap_vf'][mask].mean()
            print(f"    [{lo:.2f}, {hi:.2f})   {n:>4} {v:>7.4f} {f:>7.4f} {f21:>7.4f} {gap:>+9.4f}")

    bucket(arr['max_bl'], arr,
           [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 100.0)],
           'max_branch_len')
    bucket(arr['mean_cov'], arr,
           [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)],
           'mean_column_coverage')
    bucket(arr['mid_frac'], arr,
           [(0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.5), (0.5, 1.01)],
           'mid_coverage_column_fraction (0.3<=cov<=0.7)')
    bucket(arr['depth'].astype(float), arr,
           [(0, 5), (5, 10), (10, 20), (20, 50), (50, 1000)],
           'tree_depth')
    bucket(arr['n_leaves'].astype(float), arr,
           [(0, 10), (10, 30), (30, 100), (100, 500), (500, 1e9)],
           'n_leaves')

    # Top entries where varanc beats fitch the most
    rows_sorted = sorted(rows, key=lambda r: r['gap_vf'], reverse=True)
    print(f"\n  Top 5 entries where varanc > fitch (gap_vf):")
    print(f"    {'family':<10} {'n_cols':>6} {'n_lv':>4} {'max_bl':>6} {'mean_cov':>8} {'mid_frac':>8} {'varanc':>7} {'fitch':>7} {'gap':>7}")
    for r in rows_sorted[:5]:
        print(f"    {r['fam']:<10} {r['n_cols']:>6} {r['n_leaves']:>4} {r['max_bl']:>6.2f} {r['mean_cov']:>8.3f} {r['mid_frac']:>8.3f} {r['varanc']:>7.4f} {r['fitch']:>7.4f} {r['gap_vf']:>+7.4f}")

    print(f"\n  Top 5 entries where fitch > varanc (gap_vf):")
    for r in rows_sorted[-5:]:
        print(f"    {r['fam']:<10} {r['n_cols']:>6} {r['n_leaves']:>4} {r['max_bl']:>6.2f} {r['mean_cov']:>8.3f} {r['mid_frac']:>8.3f} {r['varanc']:>7.4f} {r['fitch']:>7.4f} {r['gap_vf']:>+7.4f}")


if __name__ == '__main__':
    for p in [
        '/home/yam/tkf-mixdom/python/experiments/varanc_presence_unified_short_full.json',
        '/home/yam/tkf-mixdom/python/experiments/varanc_presence_unified_long_full.json',
    ]:
        analyze(p)
