#!/usr/bin/env python3
"""Diagnostic: does the bias shrink as Adam n_iter -> infinity?

If YES: bias is primarily Adam under-convergence; theory predicts
        beta -> 1 with sufficient inner Adam iters.
If NO:  bias has a structural component independent of Adam — must
        derive analytically.

Sweep n_iter ∈ {30, 100, 300, 1000, 3000} and report total |bias|_1
and per-entry I→I, D→D bias.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests',
                                  'level3_thorough'))

from test_tkf92_vbem import _to_binary_tree, _caterpillar_tree
from test_tkf92_vbem_oracle import oracle_n_trans_for_branch
from tkfmixdom.jax.simulate.tree_mixdom import simulate_tkf92_tree
from tkfmixdom.jax.train.tkf92_vbem import (
    _compute_n_trans_per_branch, fit_family_estep_tkf92_padded,
)


def main():
    A = 20
    pi = np.full(A, 1.0 / A)
    Q = np.full((A, A), 1.0 / (A - 1))
    np.fill_diagonal(Q, -1.0)
    tree_node = _caterpillar_tree(6, 0.3)
    ins, del_, ext = 0.04, 0.05, 0.0
    L = 240
    n_fams = 10  # smaller to keep this quick

    # Pre-simulate all families.
    fams = []
    for trial in range(n_fams):
        res = simulate_tkf92_tree(
            jr.PRNGKey(40000 + L * 100 + trial), tree_node,
            ins_rate=ins, del_rate=del_, ext=ext,
            Q=Q, pi=pi, root_length_mean=L)
        _, _, msa, n_cols, branches = res
        bt = _to_binary_tree(tree_node)
        leaf_names = bt.leaf_names
        nc = max(len(s) for s in msa.values())
        leaf_present = np.zeros((len(leaf_names), nc), dtype=np.int32)
        for i, name in enumerate(leaf_names):
            row = msa.get(name)
            if row is not None:
                leaf_present[i, :len(row)] = (row[:nc] >= 0).astype(np.int32)
        fams.append({'bt': bt, 'leaf_present': leaf_present, 'branches': branches})

    # Compute oracle once.
    oracle_total = np.zeros((5, 5))
    for fam in fams:
        for b in fam['branches']:
            oracle_total += oracle_n_trans_for_branch(
                b['parent_lineage'], b['child_lineage'], b['child_after'])
    print(f'Oracle total events: {oracle_total.sum():.1f} '
          f'across {n_fams} fams\n')

    # Sweep Adam n_iter.
    LABELS = ['S', 'M', 'I', 'D', 'E']
    rows = []
    for n_iter in [30, 100, 300, 1000]:
        bp_total = np.zeros((5, 5))
        for fam in fams:
            stats = fit_family_estep_tkf92_padded(
                fam['bt'], fam['leaf_present'], ins, del_, ext,
                n_iter=n_iter, lr=0.05)
            for e in range(stats.n_edges):
                bp_total += _compute_n_trans_per_branch(stats.pair_marg[e])
        bias = bp_total - oracle_total
        bias_l1 = float(np.sum(np.abs(bias)))
        rel_l1 = bias_l1 / oracle_total.sum()
        ii = bp_total[2, 2]
        dd = bp_total[3, 3]
        oracle_ii = oracle_total[2, 2]
        oracle_dd = oracle_total[3, 3]
        de = bp_total[3, 4]
        oracle_de = oracle_total[3, 4]
        print(f'n_iter={n_iter:>5}: |bias|_1={bias_l1:7.1f} '
              f'({rel_l1*100:5.2f}%) | '
              f'I->I bp={ii:6.1f} (or {oracle_ii:.1f}) | '
              f'D->D bp={dd:6.1f} (or {oracle_dd:.1f}) | '
              f'D->E bp={de:6.1f} (or {oracle_de:.1f})',
              flush=True)
        rows.append({
            'n_iter': n_iter, 'bias_l1': bias_l1, 'rel_l1': rel_l1,
            'I_to_I_bp': float(ii), 'I_to_I_oracle': float(oracle_ii),
            'D_to_D_bp': float(dd), 'D_to_D_oracle': float(oracle_dd),
            'D_to_E_bp': float(de), 'D_to_E_oracle': float(oracle_de),
        })

    import json
    out = {'n_fams': n_fams, 'oracle_total_count': float(oracle_total.sum()),
           'rows': rows}
    out_path = 'experiments/figures/adam_convergence_vs_bias.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
