#!/usr/bin/env python3
"""§6 deep dive: quantify moment-mismatch between factorized q and truth.

Two experiments:
  (A) For increasing chain lengths L (root_length_mean ∈ {20, 60, 120, 240}),
      compare per-WFST-entry E_q[count] (BP cumulant) vs E_p[count] (oracle
      from simulator).  Plot bias per entry vs L.  Tests whether bias
      scales with L (multiplicative) or saturates (additive).

  (B) Run tree-VBEM EM for many iters on a single fixed simulation, tracking
      rate trajectory.  Fit a saturation model
          rate(iter) ≈ r∞ - (r∞ - r_0) exp(-k·iter)
      to estimate the biased fixed point r∞ and convergence rate k.

Output:
  experiments/figures/moment_mismatch_L_scaling.pdf
  experiments/figures/em_convergence_trajectory.pdf
  experiments/figures/quantify_moment_mismatch.json
"""
from __future__ import annotations
import json, os, sys, warnings
warnings.simplefilter("ignore")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests',
                                  'level3_thorough'))

import numpy as np
import jax, jax.numpy as jnp, jax.random as jr, optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tkfmixdom.jax.simulate.tree_mixdom import simulate_tkf92_tree
from tkfmixdom.jax.train.tkf92_vbem import (
    fit_family_estep_tkf92_padded, _compute_n_trans_per_branch,
    extract_tkf92_suff_stats, m_step_tkf92, vbem_train_tkf92,
)
from test_tkf92_vbem import _caterpillar_tree, _to_binary_tree, _build_family_provider
from test_tkf92_vbem_oracle import oracle_n_trans_for_branch


LABELS = ['S', 'M', 'I', 'D', 'E']


def experiment_A_L_scaling():
    """For increasing root chain length L, compare E_q[counts] (BP) to
    E_p[counts] (oracle).  Sum across all branches and families."""
    A = 20; pi = np.full(A, 1.0/A); Q = np.full((A,A), 1.0/(A-1)); np.fill_diagonal(Q, -1.0)
    tree_node = _caterpillar_tree(6, 0.3)
    n_fams = 30
    true_ins, true_del, true_ext = 0.04, 0.05, 0.0  # TKF91

    L_grid = [20, 60, 120, 240]
    results = []
    for L in L_grid:
        oracle_total = np.zeros((5, 5))
        bp_total = np.zeros((5, 5))
        actual_n_cols = []
        for trial in range(n_fams):
            res = simulate_tkf92_tree(
                jr.PRNGKey(40000 + L * 100 + trial), tree_node,
                ins_rate=true_ins, del_rate=true_del, ext=true_ext,
                Q=Q, pi=pi, root_length_mean=L)
            _, _, msa, n_cols, branches = res
            actual_n_cols.append(n_cols)
            for b in branches:
                oracle_total += oracle_n_trans_for_branch(
                    b['parent_lineage'], b['child_lineage'], b['child_after'])
            bt = _to_binary_tree(tree_node)
            leaf_names = bt.leaf_names
            nc = max(len(s) for s in msa.values())
            lp = np.zeros((len(leaf_names), nc), dtype=np.int32)
            for i, name in enumerate(leaf_names):
                row = msa.get(name)
                if row is not None:
                    lp[i, :len(row)] = (row[:nc] >= 0).astype(np.int32)
            stats = fit_family_estep_tkf92_padded(
                bt, lp, true_ins, true_del, true_ext,
                n_iter=30, lr=0.05)
            for e in range(stats.n_edges):
                bp_total += _compute_n_trans_per_branch(stats.pair_marg[e])
        # Compute bias per entry.
        bias = bp_total - oracle_total
        ratio = bp_total / np.where(oracle_total > 0.5, oracle_total, np.nan)
        results.append({
            'L': L,
            'mean_n_cols': float(np.mean(actual_n_cols)),
            'oracle': oracle_total.tolist(),
            'bp': bp_total.tolist(),
            'bias_total_abs': float(np.sum(np.abs(bias))),
            'bias_signed': bias.tolist(),
            'oracle_total_count': float(oracle_total.sum()),
        })
        print(f'L={L:>3} (mean_n_cols={results[-1]["mean_n_cols"]:.1f}): '
              f'|bias|_1 = {results[-1]["bias_total_abs"]:.1f} '
              f'(out of {results[-1]["oracle_total_count"]:.0f} truth events)',
              flush=True)
        # Per-entry highlights
        for i in range(5):
            for j in range(5):
                if oracle_total[i, j] > 0 or abs(bias[i, j]) > 5:
                    print(f'    {LABELS[i]}->{LABELS[j]}: '
                          f'oracle={oracle_total[i,j]:6.1f}  '
                          f'BP={bp_total[i,j]:6.1f}  '
                          f'bias={bias[i,j]:+6.1f}',
                          flush=True)

    return results


def experiment_B_em_convergence():
    """Run tree-VBEM EM for many iters and track rate trajectory.  Fit
    saturation model to estimate biased fixed point r∞."""
    A = 20; pi = np.full(A, 1.0/A); Q = np.full((A,A), 1.0/(A-1)); np.fill_diagonal(Q, -1.0)
    tree_node = _caterpillar_tree(6, 0.3)
    n_fams = 30
    true_ins, true_del, true_ext = 0.04, 0.05, 0.0  # TKF91

    fams = []
    trees = []
    for trial in range(n_fams):
        res = simulate_tkf92_tree(
            jr.PRNGKey(50000 + trial), tree_node,
            ins_rate=true_ins, del_rate=true_del, ext=true_ext,
            Q=Q, pi=pi, root_length_mean=80)
        fams.append(res[2])
        trees.append(tree_node)
    provider = _build_family_provider(fams, trees)

    # Custom EM loop with checkpointing every iter.
    ins, del_, ext = true_ins, true_del, true_ext  # init at TRUTH to check fixed-point dynamics
    history = [{'iter': 0, 'ins': ins, 'del': del_, 'ext': ext}]
    n_iter = 30
    for k in range(n_iter):
        try:
            from tkfmixdom.jax.train.tkf92_vbem import (
                fit_family_estep_tkf92_padded as _fit_padded
            )
            stats_list = []
            for fi in range(n_fams):
                bt, lp = provider(fi)
                stats = _fit_padded(bt, lp, ins, del_, ext, n_iter=20, lr=0.05)
                stats_list.append(stats)
            suff = extract_tkf92_suff_stats(stats_list, ins, del_, ext)
            ins_new, del_new, ext_new = m_step_tkf92(suff)
            history.append({'iter': k+1, 'ins': float(ins_new),
                              'del': float(del_new), 'ext': float(ext_new)})
            ins, del_, ext = ins_new, del_new, ext_new
            print(f'  iter {k+1:>3}: ins={ins:.5f} del={del_:.5f} ext={ext:.5f}',
                  flush=True)
        except Exception as e:
            print(f'  iter {k+1}: ERROR {e}', flush=True)
            break

    # Fit saturation: r(t) = r_inf - (r_inf - r_0) * exp(-k * t)
    iters = np.array([h['iter'] for h in history])
    ins_traj = np.array([h['ins'] for h in history])
    del_traj = np.array([h['del'] for h in history])

    def fit_saturation(traj, iters):
        from scipy.optimize import curve_fit
        def model(t, r_inf, r0, k):
            return r_inf - (r_inf - r0) * np.exp(-k * t)
        try:
            popt, _ = curve_fit(model, iters, traj,
                                p0=[traj[-1] * 1.5, traj[0], 0.1],
                                maxfev=5000)
            return popt  # (r_inf, r0, k)
        except Exception as e:
            return None

    fit_ins = fit_saturation(ins_traj, iters)
    fit_del = fit_saturation(del_traj, iters)

    return {
        'truth_ins': true_ins,
        'truth_del': true_del,
        'history': history,
        'fit_ins': list(fit_ins) if fit_ins is not None else None,
        'fit_del': list(fit_del) if fit_del is not None else None,
    }


def make_figures(A_results, B_result):
    fig_dir = 'experiments/figures'
    os.makedirs(fig_dir, exist_ok=True)

    # Figure A: L scaling
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    Ls = [r['L'] for r in A_results]
    bias_total_abs = [r['bias_total_abs'] for r in A_results]
    oracle_total = [r['oracle_total_count'] for r in A_results]
    ax = axes[0]
    ax.plot(Ls, bias_total_abs, 'o-', label='|BP − Oracle|_1 (total |bias|)')
    ax.plot(Ls, oracle_total, 's--', label='Oracle total events', alpha=0.5)
    ax.set_xlabel('Root length L (Poisson mean)')
    ax.set_ylabel('count')
    ax.set_title('(A) Total |bias| vs root chain length')
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    rel_bias = [a / o if o > 0 else 0 for a, o in zip(bias_total_abs, oracle_total)]
    ax.plot(Ls, rel_bias, 'o-')
    ax.set_xlabel('Root length L')
    ax.set_ylabel('|bias|_1 / oracle total')
    ax.set_title('(A) Relative bias |bias| / oracle')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/moment_mismatch_L_scaling.pdf', dpi=150,
                bbox_inches='tight')
    print(f'Saved {fig_dir}/moment_mismatch_L_scaling.pdf')

    # Figure B: EM trajectory + saturation fit
    fig, ax = plt.subplots(figsize=(8, 5))
    iters = [h['iter'] for h in B_result['history']]
    ins_traj = [h['ins'] for h in B_result['history']]
    del_traj = [h['del'] for h in B_result['history']]
    ax.plot(iters, ins_traj, 'o-', label='λ (insertion rate)')
    ax.plot(iters, del_traj, 's-', label='μ (deletion rate)')
    ax.axhline(B_result['truth_ins'], color='blue', linestyle=':',
               alpha=0.5, label=f'truth λ = {B_result["truth_ins"]}')
    ax.axhline(B_result['truth_del'], color='orange', linestyle=':',
               alpha=0.5, label=f'truth μ = {B_result["truth_del"]}')
    if B_result['fit_ins'] is not None:
        r_inf, _, _ = B_result['fit_ins']
        ax.axhline(r_inf, color='blue', linestyle='-.', alpha=0.5,
                   label=f'biased FP λ ≈ {r_inf:.3f}')
    if B_result['fit_del'] is not None:
        r_inf, _, _ = B_result['fit_del']
        ax.axhline(r_inf, color='orange', linestyle='-.', alpha=0.5,
                   label=f'biased FP μ ≈ {r_inf:.3f}')
    ax.set_xlabel('EM iteration')
    ax.set_ylabel('rate')
    ax.set_title('(B) EM trajectory init at TRUTH (TKF91 sim, 30 fams)')
    ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/em_convergence_trajectory.pdf', dpi=150,
                bbox_inches='tight')
    print(f'Saved {fig_dir}/em_convergence_trajectory.pdf')


def main():
    print('=' * 60)
    print('Experiment A: count bias vs root chain length L')
    print('=' * 60)
    A_results = experiment_A_L_scaling()
    print('\n' + '=' * 60)
    print('Experiment B: EM convergence trajectory from truth init')
    print('=' * 60)
    B_result = experiment_B_em_convergence()
    print('\nSaturation fits:')
    if B_result['fit_ins']:
        r_inf, r0, k = B_result['fit_ins']
        print(f'  λ: biased FP ≈ {r_inf:.4f} (truth {B_result["truth_ins"]}, '
              f'bias factor {r_inf/B_result["truth_ins"]:.2f}); time constant 1/k = {1/k:.1f}')
    if B_result['fit_del']:
        r_inf, r0, k = B_result['fit_del']
        print(f'  μ: biased FP ≈ {r_inf:.4f} (truth {B_result["truth_del"]}, '
              f'bias factor {r_inf/B_result["truth_del"]:.2f}); time constant 1/k = {1/k:.1f}')

    out = {'experiment_A': A_results, 'experiment_B': B_result}
    out_path = 'experiments/figures/quantify_moment_mismatch.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')

    make_figures(A_results, B_result)


if __name__ == '__main__':
    main()
