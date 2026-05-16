#!/usr/bin/env python3
"""Sim-recon with gamma annotations: Felsenstein vs TVA vs TVA+gamma.

Simulates from class-exposed WFSTs (gamma rate variation baked in),
strips gamma labels, reconstructs with three methods:
1. Felsenstein (column-independent, no gamma)
2. TreeVarAnc (order-1, no gamma)
3. TreeVarAnc + gamma (block-diagonal BP with one-hot gamma mask)

Usage:
    cd python && uv run python experiments/sim_gamma_recon.py
"""

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jr
import time, sys, json

from tkfmixdom.jax.core.ctmc import rate_matrix_jc69, transition_matrix
from tkfmixdom.jax.tree.tree_varanc import (
    tree_varanc, build_tkf91_branch_wfst, build_tkf91_root_wfst,
    name_internal_nodes, tree_varanc_block_diagonal,
)
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax

sys.path.insert(0, '.')
from experiments.sim_recon_benchmark import simulate_tree_msa
from experiments.sim_gamma_recovery import make_balanced_tree


def run():
    A = 4
    Q, pi = rate_matrix_jc69(A)
    Q, pi = np.asarray(Q), np.asarray(pi)
    ins_rate, del_rate = 0.0458, 0.0468
    G = 4

    sys.path.insert(0, '/home/yam/subby')
    from subby.jax.models import gamma_rate_categories

    results = []

    # Sweep alpha (rate variation strength) and tree depth
    configs = [
        # (alpha, min_bl, max_bl, label)
        (1.0,  0.1, 0.3, 'moderate'),
        (0.5,  0.1, 0.3, 'strong_rate_var'),
        (0.3,  0.1, 0.3, 'very_strong_rate_var'),
        (0.5,  0.3, 0.8, 'deep_tree'),
        (1.0,  0.3, 0.8, 'deep_moderate'),
    ]

    for alpha, min_bl, max_bl, label in configs:
        rates, _ = gamma_rate_categories(alpha, G)
        rates = np.array(rates)
        print(f'\n=== {label}: alpha={alpha}, bl=[{min_bl},{max_bl}] ===')
        print(f'Rates: {[f"{r:.3f}" for r in rates]}', flush=True)

        for n_leaves in [8, 16, 32]:
            for seed in range(10):
                rng_np = np.random.RandomState(seed * 100 + n_leaves)
                rng_key = jr.PRNGKey(seed * 100 + n_leaves)

                tree = make_balanced_tree(n_leaves, min_bl, max_bl, rng_np)

                # --- Simulate from class-exposed WFSTs ---
                GA = G * A
                Q_comp = np.zeros((GA, GA))
                for g in range(G):
                    Q_comp[g*A:(g+1)*A, g*A:(g+1)*A] = rates[g] * Q
                Q_comp -= np.diag(np.diag(Q_comp))
                Q_comp -= np.diag(Q_comp.sum(axis=1))
                pi_comp = np.tile(pi / G, G)

                wfst_exp = {}
                for node in tree.preorder():
                    if node.is_root: continue
                    wfst_exp[(node.parent.name, node.name)] = build_tkf91_branch_wfst(
                        ins_rate, del_rate, Q_comp, pi_comp, node.branch_length)
                singlet_exp = build_tkf91_root_wfst(ins_rate, del_rate, pi_comp)

                msa_presence, leaf_seqs, true_root, node_seqs, _ = \
                    simulate_tree_msa(tree, wfst_exp, singlet_exp, pi_comp, rng_key)

                L = next(iter(msa_presence.values())).shape[0]
                if L < 2: continue

                # Extract true gamma labels + residue-only sequences
                true_gamma = np.zeros(L, dtype=np.int32)
                for c in range(L):
                    for name in node_seqs:
                        if msa_presence[name][c]:
                            k = int(msa_presence[name][:c].sum())
                            if k < len(node_seqs[name]):
                                true_gamma[c] = int(node_seqs[name][k]) // A
                            break

                leaf_res = {}
                for name, seq in leaf_seqs.items():
                    r = np.full(L, -1, dtype=np.int32)
                    for c in range(L):
                        if seq[c] >= 0: r[c] = seq[c] % A
                    leaf_res[name] = r

                true_root_res = np.array([int(x) % A for x in true_root])
                root_cols = np.where(msa_presence[tree.name])[0]
                n_eval = min(len(root_cols), len(true_root_res))

                # --- Method 1: Felsenstein ---
                _, fels_post = marginal_ancestor_all_columns_jax(tree, leaf_res, Q, pi)
                fels_pred = np.argmax(np.asarray(fels_post), axis=1)
                fels_acc = float(np.mean(fels_pred[root_cols[:n_eval]] == true_root_res[:n_eval]))

                # --- Method 2: TVA (no gamma) ---
                wfst_plain = {}
                for node in tree.preorder():
                    if node.is_root: continue
                    wfst_plain[(node.parent.name, node.name)] = build_tkf91_branch_wfst(
                        ins_rate, del_rate, Q, pi, node.branch_length)
                singlet_plain = build_tkf91_root_wfst(ins_rate, del_rate, pi)

                tva_post, _, _, _, _ = tree_varanc(
                    tree, msa_presence, leaf_res, wfst_plain, singlet_plain, pi,
                    n_iter=1, verbose=False)
                tva_root = tva_post[tree.name]
                tva_pred = np.argmax(tva_root, axis=1)
                tva_acc = float(np.mean(tva_pred[:n_eval] == true_root_res[:n_eval]))

                # --- Method 3: TVA + gamma (block-diagonal, one-hot mask) ---
                wfst_per_g = []
                sing_per_g = []
                pi_per_g = []
                for g in range(G):
                    Q_g = rates[g] * Q
                    Q_g -= np.diag(np.diag(Q_g))
                    Q_g -= np.diag(Q_g.sum(axis=1))
                    wg = {}
                    for node in tree.preorder():
                        if node.is_root: continue
                        wg[(node.parent.name, node.name)] = build_tkf91_branch_wfst(
                            ins_rate, del_rate, Q_g, pi, node.branch_length)
                    wfst_per_g.append(wg)
                    sing_per_g.append(build_tkf91_root_wfst(ins_rate, del_rate, pi))
                    pi_per_g.append(pi)

                rate_mask = np.zeros((L, G))
                for c in range(L): rate_mask[c, true_gamma[c]] = 1.0

                tva_g_post, _, _, _, _ = tree_varanc_block_diagonal(
                    tree, msa_presence, leaf_res,
                    wfst_per_g, sing_per_g, pi_per_g,
                    D=1, A=A, rate_multiplier_mask=rate_mask,
                    n_iter=1, verbose=False)
                tva_g_root = tva_g_post[tree.name]
                tva_g_pred = np.argmax(tva_g_root, axis=1)
                tva_g_acc = float(np.mean(tva_g_pred[:n_eval] == true_root_res[:n_eval]))

                results.append({
                    'config': label, 'alpha': alpha,
                    'min_bl': min_bl, 'max_bl': max_bl,
                    'n_leaves': n_leaves, 'seed': seed, 'L': L,
                    'fels': fels_acc, 'tva': tva_acc, 'tva_gamma': tva_g_acc,
                })
                print(f'  N={n_leaves} s={seed} L={L}: '
                      f'fels={fels_acc:.3f} tva={tva_acc:.3f} tva+γ={tva_g_acc:.3f}', flush=True)

    # Summary by config
    for cfg_label in dict.fromkeys(r['config'] for r in results):
        cfg_rs = [r for r in results if r['config'] == cfg_label]
        print(f'\n=== {cfg_label} ===')
        print(f'{"N":>3} {"Fels":>8} {"TVA":>8} {"TVA+γ":>8} {"Δ(γ-F)":>8}')
        print('-' * 42)
        for n in [8, 16, 32]:
            rs = [r for r in cfg_rs if r['n_leaves'] == n]
            if rs:
                f_mean = np.mean([r['fels'] for r in rs])
                t_mean = np.mean([r['tva'] for r in rs])
                g_mean = np.mean([r['tva_gamma'] for r in rs])
                print(f'{n:>3} {f_mean:>8.3f} {t_mean:>8.3f} {g_mean:>8.3f} {g_mean-f_mean:>+8.3f}')

    with open('experiments/sim_gamma_recon.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to experiments/sim_gamma_recon.json')


if __name__ == '__main__':
    run()
