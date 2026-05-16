#!/usr/bin/env python3
"""Diagnostic: at q = Felsenstein-exact-truth posterior (NOT Adam-fit),
does the BP cumulant W match the oracle simulation count?

If YES: the +110% bias is purely from Adam under-convergence; theory
        predicts beta -> 1 as Adam iters -> infinity.
If NO:  the cumulant formula has its own structural gap even at the
        exact truth posterior; theory must derive beta analytically
        from the cumulant formula's own approximation error.

Strategy:
  1. Simulate one TKF91 family with full edge labels.
  2. Compute oracle n_trans per branch from the labelled simulation.
  3. Set q_cond = TKF91-truth-conditional-matrix per (edge, col) — this
     should make BP-with-q match exact Felsenstein-truth.
  4. Compute pair_marg from BP-with-q.
  5. Compute W from pair_marg via _compute_n_trans_per_branch.
  6. Compare oracle vs W.
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
from tkfmixdom.jax.train.tkf92_vbem import _compute_n_trans_per_branch
from tkfmixdom.jax.tree.varanc_presence import (
    bp_pair_marginals, make_root_dist, make_q_conditionals,
    leaf_clamp_to_beta, edge_lookup,
)


def tkf91_truth_q_cond(t_edge, ins, del_):
    """Compute TKF91 truth transition matrix t_e[parent_state, child_state]
    over {NYI, P, D} for a single edge of length t.

    Per Bishop & Thompson 1991 / Holmes 2003:
      t[NYI, NYI] = 1 - lambda * t * f(...)  (no insertion)
      t[NYI, P]   = inserted probability
      t[P, P]     = exp(-mu * t)             (parent P survives unchanged)
      t[P, D]     = 1 - exp(-mu * t)         (parent P got deleted)
      t[D, D]     = 1                        (D is irreversible)

    For the simplest BDI-on-position interpretation:
      t[NYI, NYI] = exp(-lambda * t)         (no insertion in [0, t])
      t[NYI, P]   = 1 - exp(-lambda * t)     (insertion happened)
    """
    nyi_nyi = float(np.exp(-ins * t_edge))
    nyi_p = 1.0 - nyi_nyi
    p_p = float(np.exp(-del_ * t_edge))
    p_d = 1.0 - p_p
    return jnp.array([
        [nyi_nyi, nyi_p, 0.0],
        [0.0, p_p, p_d],
        [0.0, 0.0, 1.0],
    ], dtype=jnp.float64)


def main():
    # Simulate one TKF91 family.
    A = 20
    pi = np.full(A, 1.0 / A)
    Q = np.full((A, A), 1.0 / (A - 1))
    np.fill_diagonal(Q, -1.0)
    tree_node = _caterpillar_tree(6, 0.3)
    ins, del_, ext = 0.04, 0.05, 0.0  # TKF91
    L = 240

    res = simulate_tkf92_tree(
        jr.PRNGKey(42), tree_node,
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
    print(f'Simulated 1 family: n_cols={nc}, n_edges={bt.num_edges}, '
          f'n_leaves={bt.num_leaves}')

    # Build q_cond from TKF91 truth conditionals per edge.
    edge_lengths = np.maximum(np.asarray(bt.edge_length), 1e-3)
    bt_clipped = bt._replace(edge_length=edge_lengths)
    q_cond_per_edge = np.stack([
        np.asarray(tkf91_truth_q_cond(t, ins, del_))
        for t in edge_lengths
    ])  # (E, 3, 3)
    q_cond = jnp.broadcast_to(
        q_cond_per_edge[:, None, :, :], (bt_clipped.num_edges, nc, 3, 3))
    print(f'q_cond shape: {q_cond.shape}')
    print(f'Edge 0 truth conditional:\n{q_cond_per_edge[0]}')

    # Per-column root distribution: at TKF91 stationary distribution.
    # TKF91 stationary: pi_NYI / pi_P = mu / lambda (from balance).
    # Strictly, pi_P = lambda / (lambda + mu), pi_NYI = mu / (lambda + mu).
    pi_P = ins / (ins + del_)
    pi_NYI = del_ / (ins + del_)
    root_logit_val = float(np.log(pi_NYI / pi_P))
    root_logit = jnp.full((nc,), root_logit_val, dtype=jnp.float64)
    root_dist = make_root_dist(root_logit)

    leaf_clamp = leaf_clamp_to_beta(jnp.asarray(leaf_present))
    le, re = edge_lookup(bt_clipped)

    pair_marg, log_Z = bp_pair_marginals(
        q_cond, root_dist, leaf_clamp, bt_clipped,
        jnp.asarray(le), jnp.asarray(re))
    print(f'pair_marg shape: {pair_marg.shape}')
    print(f'log_Z (per col, first 5): {np.asarray(log_Z)[:5]}')
    print(f'sum log_Z: {float(jnp.sum(log_Z)):.4f}')

    # Now compute W from this pair_marg at q=truth.
    bp_total_truth_q = np.zeros((5, 5))
    for e in range(bt_clipped.num_edges):
        bp_total_truth_q += _compute_n_trans_per_branch(pair_marg[e])

    # And the oracle:
    oracle_total = np.zeros((5, 5))
    for b in branches:
        oracle_total += oracle_n_trans_for_branch(
            b['parent_lineage'], b['child_lineage'], b['child_after'])

    # Compare:
    LABELS = ['S', 'M', 'I', 'D', 'E']
    print(f'\n  Total transition events: oracle={oracle_total.sum():.1f} '
          f'truth-q-BP={bp_total_truth_q.sum():.1f}')
    print(f'  |bias|_1 = {np.sum(np.abs(bp_total_truth_q - oracle_total)):.1f}')
    print('\nPer-entry comparison:')
    print(f'   Entry  |  Oracle  |  Truth-q-BP  |  Bias')
    print('   -------+----------+--------------+--------')
    for i in range(5):
        for j in range(5):
            o, b = oracle_total[i, j], bp_total_truth_q[i, j]
            if o > 0.5 or abs(b - o) > 5:
                print(f'   {LABELS[i]}->{LABELS[j]} | {o:8.1f} | {b:12.1f} | '
                      f'{b - o:+7.1f}')


if __name__ == '__main__':
    main()
