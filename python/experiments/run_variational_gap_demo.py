#!/usr/bin/env python3
"""Variational-gap demo for the TKF paper §6 svi-VarAnc section.

For each ext value in {0.0, 0.3, 0.6}:
  * Simulate n_fams families on a 6-leaf caterpillar tree using the
    fully-labeled TKF92 simulator.
  * Compute Oracle WFST n_trans (5x5) from the simulator's true chain
    events (via oracle_n_trans_for_branch).
  * Run BP at TRUTH params, extract BP n_trans via the cumulant trick
    from the variational q's pair_marg (via _compute_n_trans_per_branch).
  * Sum across all (family, branch) pairs -> two 5x5 matrices to compare.

Saves results JSON + a log-log scatter PDF showing the variational gap
(BP overcounts rare entries up to 50x or more relative to oracle truth).
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.simplefilter("ignore")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.random as jr

from tkfmixdom.jax.simulate.tree_mixdom import simulate_tkf92_tree
from tkfmixdom.jax.train.tkf92_vbem import (
    fit_family_estep_tkf92_padded,
    _compute_n_trans_per_branch,
)
sys.path.insert(0, os.path.dirname(__file__))  # tests live alongside
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests',
                                  'level3_thorough'))
from test_tkf92_vbem import _caterpillar_tree, _to_binary_tree
from test_tkf92_vbem_oracle import oracle_n_trans_for_branch


LABELS = ['S', 'M', 'I', 'D', 'E']


def run(n_fams: int = 50, root_length_mean: int = 60,
        ext_grid=(0.0, 0.3, 0.6),
        true_ins: float = 0.04, true_del: float = 0.05,
        out_path: str = "experiments/figures/variational_gap_demo.json"):
    A = 20
    pi = np.full(A, 1.0 / A)
    Q_aa = np.full((A, A), 1.0 / (A - 1)); np.fill_diagonal(Q_aa, -1.0)
    tree = _caterpillar_tree(6, 0.3)
    bt = _to_binary_tree(tree)

    out = {
        'n_fams': n_fams, 'root_length_mean': root_length_mean,
        'tree': '6-leaf caterpillar at branch_length=0.3',
        'true_ins': true_ins, 'true_del': true_del,
        'ext_grid': list(ext_grid),
        'labels': LABELS,
        'oracle': {}, 'bp': {},
    }
    for ext_true in ext_grid:
        oracle_total = np.zeros((5, 5))
        bp_total = np.zeros((5, 5))
        print(f'=== ext = {ext_true} ===', flush=True)
        for trial in range(n_fams):
            res = simulate_tkf92_tree(
                jr.PRNGKey(50000 + int(ext_true * 1000) + trial), tree,
                ins_rate=true_ins, del_rate=true_del, ext=float(ext_true),
                Q=Q_aa, pi=pi, root_length_mean=root_length_mean)
            leaf_seqs, leaf_lineages, msa, n_cols, branches = res
            for b in branches:
                oracle_total += oracle_n_trans_for_branch(
                    b['parent_lineage'], b['child_lineage'], b['child_after'])
            leaf_names = bt.leaf_names
            nc = max(len(s) for s in msa.values())
            lp = np.zeros((len(leaf_names), nc), dtype=np.int32)
            for i, name in enumerate(leaf_names):
                row = msa.get(name)
                if row is not None:
                    n = min(len(row), nc)
                    lp[i, :n] = (row[:n] >= 0).astype(np.int32)
            stats = fit_family_estep_tkf92_padded(
                bt, lp, true_ins, true_del, float(ext_true),
                n_iter=30, lr=0.05)
            for e in range(stats.n_edges):
                bp_total += _compute_n_trans_per_branch(stats.pair_marg[e])
            if (trial + 1) % 10 == 0:
                print(f'  [{trial+1}/{n_fams}]', flush=True)
        out['oracle'][f'{ext_true:.2f}'] = oracle_total.tolist()
        out['bp'][f'{ext_true:.2f}'] = bp_total.tolist()
        # Print summary
        print(f'  Oracle.sum = {oracle_total.sum():.0f}, '
              f'BP.sum = {bp_total.sum():.0f}', flush=True)
        for i in range(5):
            for j in range(5):
                if oracle_total[i, j] > 1.0 or bp_total[i, j] > 1.0:
                    ratio = (bp_total[i, j] / oracle_total[i, j]
                             if oracle_total[i, j] > 0.5 else float('inf'))
                    print(f'    {LABELS[i]}->{LABELS[j]}: '
                          f'oracle={oracle_total[i,j]:.1f}  '
                          f'BP={bp_total[i,j]:.1f}  '
                          f'ratio={ratio:.2f}', flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}', flush=True)
    return out


def make_figure(json_path: str = "experiments/figures/variational_gap_demo.json",
                pdf_path: str = "experiments/figures/variational_gap_demo.pdf"):
    """Render the log-log scatter: Oracle vs BP per (i, j), color-coded."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(json_path) as f:
        d = json.load(f)

    fig, axes = plt.subplots(1, len(d['ext_grid']), figsize=(5*len(d['ext_grid']), 5))
    if len(d['ext_grid']) == 1:
        axes = [axes]

    L = d['labels']
    color_map = {
        ('M', 'M'): 'gray',  # diagonal common
        ('S', 'M'): 'gray',
        ('M', 'E'): 'gray',
    }
    for ax, ext_str in zip(axes, [f'{e:.2f}' for e in d['ext_grid']]):
        oracle = np.array(d['oracle'][ext_str])
        bp = np.array(d['bp'][ext_str])
        all_pts_o, all_pts_b, all_lab, all_col = [], [], [], []
        for i in range(5):
            for j in range(5):
                if oracle[i, j] < 0.1 and bp[i, j] < 0.1:
                    continue
                lab = f'{L[i]}→{L[j]}'
                # Color by category
                if (L[i], L[j]) in color_map:
                    col = color_map[(L[i], L[j])]
                elif L[j] == 'E':
                    col = 'tab:orange'  # terminations
                elif L[i] == 'I' or L[j] == 'I':
                    col = 'tab:red'  # insertion-related
                elif L[i] == 'D' or L[j] == 'D':
                    col = 'tab:blue'  # deletion-related
                else:
                    col = 'gray'
                all_pts_o.append(max(oracle[i, j], 0.5))
                all_pts_b.append(max(bp[i, j], 0.5))
                all_lab.append(lab)
                all_col.append(col)
        for o, b, lab, col in zip(all_pts_o, all_pts_b, all_lab, all_col):
            ax.scatter(o, b, c=col, s=80, edgecolor='black', linewidth=0.5,
                       alpha=0.8)
            # Label points where ratio > 2.5x or < 1/2.5x
            ratio = b / o if o > 0.5 else float('inf')
            if ratio > 2.5 or ratio < 0.4:
                ax.annotate(lab, (o, b), xytext=(5, 5),
                            textcoords='offset points', fontsize=8)
        # y=x diagonal
        lim = max(max(all_pts_o), max(all_pts_b)) * 1.5
        ax.plot([0.5, lim], [0.5, lim], 'k--', alpha=0.3, label='y=x')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.5, lim); ax.set_ylim(0.5, lim)
        ax.set_xlabel('Oracle n_trans (truth)')
        ax.set_ylabel('BP n_trans (variational)')
        ax.set_title(f'ext = {ext_str}')
        ax.grid(alpha=0.3)
    fig.suptitle('Variational gap: BP cumulant counts vs oracle truth\n'
                 '(6-leaf caterpillar, n_fams=' + str(d['n_fams']) +
                 ', root_len_mean=' + str(d['root_length_mean']) + ')',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight')
    print(f'Saved {pdf_path}', flush=True)


if __name__ == '__main__':
    out = run()
    try:
        make_figure()
    except Exception as e:
        print(f'Figure generation failed: {e}', flush=True)
