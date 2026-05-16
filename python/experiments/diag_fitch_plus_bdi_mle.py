#!/usr/bin/env python3
"""Diagnostic: Fitch parsimony ancestral assignment + BDI MLE on
fully-observed transition counts. Compare to truth and to the
variational SVI-VarAnc fixed point (lambda~0.114, mu~0.116 vs truth
0.04, 0.05).

Pipeline:
  1. Simulate n_fams TKF91 families (truth λ=0.04, μ=0.05, ext=0).
  2. Per family, per column, run Fitch parsimony to assign internal
     nodes presence/absence.
  3. Per branch: WFST state per col from (parent, child) presence:
       (1,1)=M  (0,1)=I  (1,0)=D  (0,0)=Ig (skip).
     Aggregate into per-branch n_trans (5x5).
  4. Convert to BDI suff stats (T = t per branch).
  5. Run m_step_tkf92 to recover (ins, del, ext).
  6. Compare to truth and to oracle (which uses simulator's labels).

Hypothesis: Fitch+BDI gives near-truth recovery, demonstrating that
the SVI-VarAnc bias is specifically from the column-factorised q
cumulant approximation, NOT from the BDI MLE itself.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
import numpy as np
import jax.random as jr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests',
                                  'level3_thorough'))

from test_tkf92_vbem import _to_binary_tree, _caterpillar_tree
from test_tkf92_vbem_oracle import (
    oracle_n_trans_for_branch, oracle_suff_stats_from_branches,
)
from tkfmixdom.jax.simulate.tree_mixdom import simulate_tkf92_tree
from tkfmixdom.jax.train.tkf92_vbem import m_step_tkf92
from tkfmixdom.jax.core.bdi import (
    tkf92_stats_from_counts, transition_count_groups,
)
from tkfmixdom.jax.core.params import S, M, I, D, E


def fitch_presence_per_node(bt, leaf_presence_dict, L_msa, mode='standard'):
    """Fitch presence/absence assignment, vectorised over columns.

    Args:
        bt: BinaryTree.
        leaf_presence_dict: name -> (L_msa,) bool array.
        L_msa: int.
        mode: one of {'standard', 'dollo'}.
            - 'standard' = Fitch parsimony for binary states:
              postorder: set_v = (set_l ∩ set_r) if non-empty, else
                         (set_l ∪ set_r);
              preorder traceback: state_v = parent_state if in set_v,
                         else 1-of set_v deterministically (we pick the
                         set's smaller member if the parent's state is
                         not in set_v, which biases ambiguous internals
                         toward 0/absent — symmetric tie-break).
            - 'dollo' = Dollo's law (single-insertion / no homoplasy):
              postorder: parent = AND(children); preorder: parent's
              presence floods down to children.

    Returns: dict node_index -> (L_msa,) bool array.
    """
    num_internal = bt.num_internal
    pres = {}
    for li, lname in enumerate(bt.leaf_names):
        pres[num_internal + li] = np.asarray(
            leaf_presence_dict.get(lname, np.zeros(L_msa, dtype=bool)),
            dtype=bool)

    if mode == 'dollo':
        for v in bt.postorder_internal:
            l = bt.left_child[v]; r = bt.right_child[v]
            pres[int(v)] = pres[int(l)] & pres[int(r)]
        for v in reversed(list(bt.postorder_internal)):
            v = int(v)
            if bt.parent[v] >= 0:
                p = int(bt.parent[v])
                pres[v] = pres[v] | pres[p]
        return pres

    # Standard Fitch on binary states.
    # First pass (postorder): compute the consistent-state set per node
    # per column. Encode as two boolean arrays: has_0[v, k], has_1[v, k].
    has0 = {}
    has1 = {}
    for li, lname in enumerate(bt.leaf_names):
        leaf_pres = pres[num_internal + li]
        has0[num_internal + li] = ~leaf_pres
        has1[num_internal + li] = leaf_pres
    for v in bt.postorder_internal:
        l = int(bt.left_child[v]); r = int(bt.right_child[v])
        # Intersection per state per column.
        i0 = has0[l] & has0[r]
        i1 = has1[l] & has1[r]
        # If intersection is empty (i.e., neither i0 nor i1 set), use union.
        empty = ~(i0 | i1)
        u0 = has0[l] | has0[r]
        u1 = has1[l] | has1[r]
        has0[int(v)] = np.where(empty, u0, i0)
        has1[int(v)] = np.where(empty, u1, i1)

    # Second pass (preorder): traceback. At each node, set state = parent's
    # state if compatible, else pick the smaller compatible state (i.e.,
    # 0 if has0, else 1). Root: pick 1 if root has1 only, 0 if has0 only,
    # else 0 (deterministic tie-break).
    # Find the root (last in postorder).
    root_v = int(bt.postorder_internal[-1])
    # Root state: pick majority across leaves as a sane default for
    # ambiguous root.  In a balanced binary case the root is rarely
    # ambiguous in a single column; we default to 1 if has1 only, 0 if
    # has0 only, else 0.
    pres[root_v] = has1[root_v] & ~has0[root_v]  # forced-1 cols
    # Cols where root is ambiguous (both has0 and has1 true): default 0.
    # Cols where root has0 only: 0 (already set as ~has1 & has0 = False
    # above, so leave as 0).
    # That covers all cases.

    for v in reversed(list(bt.postorder_internal)):
        v = int(v)
        if bt.parent[v] < 0:
            continue
        p = int(bt.parent[v])
        parent_state = pres[p]
        # If parent_state is in this node's set, use it. Else use the
        # smaller compatible state (0 if has0, else 1).
        compat_with_parent = np.where(parent_state, has1[v], has0[v])
        # Default for non-compat: pick 0 if has0, else 1.
        default = np.where(has0[v], False, True)
        pres[v] = np.where(compat_with_parent, parent_state, default)

    return pres


def n_trans_from_fitch_branch(parent_pres, child_pres):
    """Build the (5, 5) WFST n_trans matrix from per-column
    (parent_present, child_present) on one branch.

    State encoding per col:
      (1, 1) -> M
      (0, 1) -> I  (insertion on edge)
      (1, 0) -> D  (deletion on edge)
      (0, 0) -> Ig (skip — branch is non-emitting at this col)

    The WFST chain transitions are between consecutive non-Ig cols,
    plus S -> first non-Ig and last non-Ig -> E.
    """
    # Per-col WFST state (or 'Ig' to skip).
    L = parent_pres.shape[0]
    states = []
    for k in range(L):
        if parent_pres[k] and child_pres[k]:
            states.append(M)
        elif (not parent_pres[k]) and child_pres[k]:
            states.append(I)
        elif parent_pres[k] and (not child_pres[k]):
            states.append(D)
        # else: Ig, skip
    n_trans = np.zeros((5, 5), dtype=np.float64)
    prev = S
    for s in states:
        n_trans[prev, s] += 1.0
        prev = s
    n_trans[prev, E] += 1.0
    return n_trans


def fitch_suff_stats(branches_with_pres, ins_rate, del_rate, ext):
    """Aggregate Fitch-derived BDI suff stats across all branches.

    branches_with_pres: list of dicts {parent_pres, child_pres, t}.
    """
    suff = {'B': 0.0, 'D': 0.0, 'S': 0.0, 'L': 0.0, 'M': 0.0, 'T': 0.0,
            'ext_count': 0.0, 'notext_count': 0.0}
    for b in branches_with_pres:
        n_trans = n_trans_from_fitch_branch(b['parent_pres'], b['child_pres'])
        if n_trans.sum() < 1e-9:
            continue
        t = float(b['t'])
        r = tkf92_stats_from_counts(
            n_trans, ins_rate, del_rate, t, ext, T=t)
        groups = transition_count_groups(r['n_trans_resolved'])
        suff['B'] += r['E_B']
        suff['D'] += r['E_D']
        suff['S'] += r['E_S']
        suff['L'] += float(groups['log_kappa'])
        suff['M'] += float(groups['log_1mkappa'])
        suff['T'] += t
        suff['ext_count'] += r['ext_count']
        suff['notext_count'] += r['notext_count']
    return suff


def main():
    A = 20
    pi = np.full(A, 1.0 / A)
    Q = np.full((A, A), 1.0 / (A - 1))
    np.fill_diagonal(Q, -1.0)
    tree_node = _caterpillar_tree(6, 0.3)
    ins, del_, ext = 0.04, 0.05, 0.0
    L = 240
    n_fams = 30

    fitch_branches_all = []
    dollo_branches_all = []
    oracle_branches_all = []
    actual_n_cols = []
    for trial in range(n_fams):
        res = simulate_tkf92_tree(
            jr.PRNGKey(40000 + L * 100 + trial), tree_node,
            ins_rate=ins, del_rate=del_, ext=ext,
            Q=Q, pi=pi, root_length_mean=L)
        _, _, msa, n_cols, branches = res
        actual_n_cols.append(n_cols)
        bt = _to_binary_tree(tree_node)
        leaf_names = bt.leaf_names
        nc = max(len(s) for s in msa.values())
        # Build leaf_presence dict: name -> (nc,) bool.
        leaf_pres = {}
        for name in leaf_names:
            row = msa.get(name)
            if row is not None:
                p = np.zeros(nc, dtype=bool)
                p[:len(row)] = (row[:nc] >= 0)
                leaf_pres[name] = p
            else:
                leaf_pres[name] = np.zeros(nc, dtype=bool)

        # Per-node presence via BOTH Fitch variants for comparison.
        for mode, target in (('standard', fitch_branches_all),
                              ('dollo', dollo_branches_all)):
            pres_per_node = fitch_presence_per_node(bt, leaf_pres, nc,
                                                     mode=mode)
            for e in range(bt.num_edges):
                p_node = int(bt.edge_parent[e])
                c_node = int(bt.edge_child[e])
                t_e = float(bt.edge_length[e])
                target.append({
                    'parent_pres': pres_per_node[p_node],
                    'child_pres': pres_per_node[c_node],
                    't': t_e,
                })
        # Oracle (true) branch alignments for comparison.
        oracle_branches_all.extend(branches)

    # Aggregate suff stats. Use truth init for tkf92_stats_from_counts
    # (which needs (ins, del, ext) for the resolved-n̂ decomposition).
    fitch_suff = fitch_suff_stats(fitch_branches_all, ins, del_, ext)
    dollo_suff = fitch_suff_stats(dollo_branches_all, ins, del_, ext)
    oracle_suff = oracle_suff_stats_from_branches(oracle_branches_all, ins, del_, ext)

    # Compute MLE from each.
    print('=' * 72)
    print(f'TKF91 simulation: λ_*={ins}, μ_*={del_}, ext_*={ext}, '
          f'L={L}, n_fams={n_fams}, mean_actual_n_cols={np.mean(actual_n_cols):.1f}')
    print(f'Tree: 6-leaf caterpillar, branches=0.3')
    print()

    # Fitch (standard).
    f_ins, f_del, f_ext = m_step_tkf92(fitch_suff)
    print(f'Standard Fitch + BDI MLE:')
    print(f'  ins={f_ins:.5f}  del={f_del:.5f}  ext={f_ext:.5f}')
    print(f'  ratio to truth: ins={f_ins/ins:.3f}x, del={f_del/del_:.3f}x')

    # Dollo (Fitch-floor) for contrast.
    d_ins, d_del, d_ext = m_step_tkf92(dollo_suff)
    print(f'Dollo (intersection-postorder + flood-down preorder) + BDI MLE:')
    print(f'  ins={d_ins:.5f}  del={d_del:.5f}  ext={d_ext:.5f}')
    print(f'  ratio to truth: ins={d_ins/ins:.3f}x, del={d_del/del_:.3f}x')

    # Oracle path.
    o_ins, o_del, o_ext = m_step_tkf92(oracle_suff)
    print(f'Oracle + BDI MLE (uses simulator true labels):')
    print(f'  ins={o_ins:.5f}  del={o_del:.5f}  ext={o_ext:.5f}')
    print(f'  ratio to truth: ins={o_ins/ins:.3f}x, del={o_del/del_:.3f}x')

    # Reference: variational SVI-VarAnc fixed point from prior diag.
    print(f'\nFor reference, variational SVI-VarAnc EM fixed point '
          f'(from quantify_moment_mismatch.py):')
    print(f'  ins~0.114 (2.85x truth), del~0.116 (2.32x truth), '
          f'ext~0.054 (truth 0.0)')

    # Per-entry n_trans comparison.
    print(f'\nPer-entry n_trans comparison:')
    fitch_total = np.zeros((5, 5))
    dollo_total = np.zeros((5, 5))
    oracle_total = np.zeros((5, 5))
    for b in fitch_branches_all:
        fitch_total += n_trans_from_fitch_branch(b['parent_pres'], b['child_pres'])
    for b in dollo_branches_all:
        dollo_total += n_trans_from_fitch_branch(b['parent_pres'], b['child_pres'])
    for b in oracle_branches_all:
        oracle_total += oracle_n_trans_for_branch(
            b['parent_lineage'], b['child_lineage'], b['child_after'])
    LABELS = ['S', 'M', 'I', 'D', 'E']
    print(f'  Entry  | Oracle | Fitch  | Dollo  |  Δ_F   |  Δ_D')
    for i in range(5):
        for j in range(5):
            o = oracle_total[i, j]
            f = fitch_total[i, j]
            d = dollo_total[i, j]
            if o > 0.5 or abs(f - o) > 1 or abs(d - o) > 1:
                print(f'  {LABELS[i]}->{LABELS[j]}  | {o:6.0f} | {f:6.0f} | '
                      f'{d:6.0f} | {f - o:+6.0f} | {d - o:+6.0f}')

    # Save.
    import json
    out = {
        'simulation': {
            'ins_truth': ins, 'del_truth': del_, 'ext_truth': ext,
            'L': L, 'n_fams': n_fams,
            'mean_n_cols': float(np.mean(actual_n_cols)),
            'tree': '6-leaf caterpillar, branches 0.3',
        },
        'fitch_plus_bdi': {
            'ins': float(f_ins), 'del': float(f_del), 'ext': float(f_ext),
            'ratio_to_truth_ins': float(f_ins / ins),
            'ratio_to_truth_del': float(f_del / del_),
        },
        'dollo_plus_bdi': {
            'ins': float(d_ins), 'del': float(d_del), 'ext': float(d_ext),
            'ratio_to_truth_ins': float(d_ins / ins),
            'ratio_to_truth_del': float(d_del / del_),
        },
        'oracle_plus_bdi': {
            'ins': float(o_ins), 'del': float(o_del), 'ext': float(o_ext),
            'ratio_to_truth_ins': float(o_ins / ins),
            'ratio_to_truth_del': float(o_del / del_),
        },
        'variational_em_fixed_point_for_reference': {
            'ins': 0.114, 'del': 0.116, 'ext': 0.054,
            'ratio_to_truth_ins': 2.85, 'ratio_to_truth_del': 2.32,
        },
        'fitch_total_n_trans': fitch_total.tolist(),
        'dollo_total_n_trans': dollo_total.tolist(),
        'oracle_total_n_trans': oracle_total.tolist(),
        'labels': LABELS,
    }
    out_path = 'experiments/figures/fitch_plus_bdi_mle.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
