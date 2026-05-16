#!/usr/bin/env python3
"""Diagnostic: at q_cond = TKF91 prior (data-INDEPENDENT) — i.e. q's
per-col pair_marg = exact Felsenstein truth posterior — does the BP
cumulant W match the oracle simulation count, or is there structural
bias even at the exact truth posterior?

If bias L1 / oracle is SAME as Adam-fit q (~4%) and per-entry
inflations match (I->I 47x), then the cumulant trick itself has
structural bias unrelated to q's quality.

If bias is MUCH SMALLER (e.g. <1% / per-entry close to oracle), then
the bias is genuinely from Adam not converging to truth (the ELBO
entropy reward pulling q diffuser, etc.).

Sweeps n_fams ∈ {1, 5, 10, 30} and reports per-entry detail.
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
    bp_pair_marginals, make_root_dist, leaf_clamp_to_beta, edge_lookup,
    tkf92_wfst_T,  # for the canonical TKF91/92 transition matrices
)


def tkf91_prior_q_cond(t_edge, ins, del_):
    """Use the canonical TKF92 (with ext=0) WFST 5x5 row-collapsed to a
    3x3 conditional over {NYI, P, D}."""
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
    A = 20
    pi = np.full(A, 1.0 / A)
    Q = np.full((A, A), 1.0 / (A - 1))
    np.fill_diagonal(Q, -1.0)
    tree_node = _caterpillar_tree(6, 0.3)
    ins, del_, ext = 0.04, 0.05, 0.0  # TKF91
    L = 240

    LABELS = ['S', 'M', 'I', 'D', 'E']

    # Test up to n_fams=30 (matches moment-mismatch table).
    rows = []
    for n_fams in [1, 5, 10, 30]:
        oracle_total = np.zeros((5, 5))
        bp_total = np.zeros((5, 5))
        for trial in range(n_fams):
            res = simulate_tkf92_tree(
                jr.PRNGKey(40000 + L * 100 + trial), tree_node,
                ins_rate=ins, del_rate=del_, ext=ext,
                Q=Q, pi=pi, root_length_mean=L)
            _, _, msa, n_cols, branches = res
            for b in branches:
                oracle_total += oracle_n_trans_for_branch(
                    b['parent_lineage'], b['child_lineage'], b['child_after'])

            bt = _to_binary_tree(tree_node)
            leaf_names = bt.leaf_names
            nc = max(len(s) for s in msa.values())
            leaf_present = np.zeros((len(leaf_names), nc), dtype=np.int32)
            for i, name in enumerate(leaf_names):
                row = msa.get(name)
                if row is not None:
                    leaf_present[i, :len(row)] = (row[:nc] >= 0).astype(np.int32)

            edge_lengths = np.maximum(np.asarray(bt.edge_length), 1e-3)
            bt_clipped = bt._replace(edge_length=edge_lengths)
            q_cond_per_edge = np.stack([
                np.asarray(tkf91_prior_q_cond(t, ins, del_))
                for t in edge_lengths
            ])
            q_cond = jnp.broadcast_to(
                q_cond_per_edge[:, None, :, :],
                (bt_clipped.num_edges, nc, 3, 3))

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
            for e in range(bt_clipped.num_edges):
                bp_total += _compute_n_trans_per_branch(pair_marg[e])

        bias = bp_total - oracle_total
        bias_l1 = float(np.sum(np.abs(bias)))
        rel_l1 = bias_l1 / max(oracle_total.sum(), 1e-9)
        ii_oracle = oracle_total[2, 2]
        ii_bp = bp_total[2, 2]
        dd_oracle = oracle_total[3, 3]
        dd_bp = bp_total[3, 3]
        de_oracle = oracle_total[3, 4]
        de_bp = bp_total[3, 4]
        print(f'n_fams={n_fams:>3}: oracle_total={oracle_total.sum():>7.0f} | '
              f'|bias|_1={bias_l1:>7.1f} ({rel_l1*100:>5.2f}%) | '
              f'I->I bp={ii_bp:>5.1f} (or {ii_oracle:>4.0f}) | '
              f'D->D bp={dd_bp:>5.1f} (or {dd_oracle:>4.0f}) | '
              f'D->E bp={de_bp:>5.1f} (or {de_oracle:>4.0f})',
              flush=True)
        rows.append({
            'n_fams': n_fams,
            'oracle_total': float(oracle_total.sum()),
            'bp_total': float(bp_total.sum()),
            'bias_l1': bias_l1,
            'rel_l1': rel_l1,
            'I_to_I_bp': float(ii_bp),
            'I_to_I_oracle': float(ii_oracle),
            'D_to_D_bp': float(dd_bp),
            'D_to_D_oracle': float(dd_oracle),
            'D_to_E_bp': float(de_bp),
            'D_to_E_oracle': float(de_oracle),
        })

    print('\n--- INTERPRETATION ---')
    if rows:
        first, last = rows[0], rows[-1]
        bias_scale = last['bias_l1'] / first['bias_l1']
        n_scale = last['n_fams'] / first['n_fams']
        sqrt_n = float(np.sqrt(n_scale))
        print(f'  bias_l1 scaled by {bias_scale:.2f}x; n_fams scaled by '
              f'{n_scale:.0f}x (sqrt={sqrt_n:.2f}x).')
        if bias_scale > 0.7 * n_scale:
            print(f'  -> bias scales ~LINEARLY with n_fams (structural).')
        elif bias_scale < 1.5 * sqrt_n:
            print(f'  -> bias scales ~sqrt(n_fams) (MC noise).')
        else:
            print(f'  -> intermediate scaling.')

    import json
    out = {
        'note': 'q_cond = TKF91 prior (data-independent) — pair_marg is '
                'exact Felsenstein truth posterior.',
        'rows': rows,
    }
    out_path = 'experiments/figures/cumulant_structural_bias.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
