#!/usr/bin/env python3
"""§6 experiment (3): exact log p(MSA | params, tree) via Machine Boss.

Uses Machine Boss's phylo-tree intersection in pair-token mode to compute
the EXACT marginal log-likelihood for a small phylo-HMM.  Compares to
our variational ELBO (lower bound) at the same setup.  The variational
gap = exact − ELBO localises whether the +110% rate-recovery bias comes
from (a) factorised q being a poor approximation [large gap], or (b)
BP/cumulant mechanism producing spurious counts independent of q quality
[small gap, but biased rates].

Pipeline:
  1. Pick a tiny 3-leaf tree, TKF91 truth (ext=0).
  2. Generate a small MSA (5-10 columns) with various presence/absence
     patterns.
  3. Encode each column as a pair-token [X,Y],Z with X,Y,Z in {'A',''}.
  4. Call boss with phylo-tree composition + recognize-json + -L.
  5. Run varanc_presence.elbo at truth params, Adam-converged q.
  6. Report exact_log_p, ELBO, gap = exact - ELBO.

Requires: bin/boss built with patches (Makefile linker order + parsers.cpp
peg::parser disambig), libboost-all-dev installed.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                  '..', 'tests', 'level3_thorough'))

from tkfmixdom.jax.tree.varanc_presence import (
    edge_lookup, parse_binary_tree,
    elbo as varanc_elbo, leaf_clamp_to_beta,
)
from test_tkf92_vbem import _to_binary_tree
from tkfmixdom.jax.util.io import parse_newick


BOSS = "/home/yam/machineboss/bin/boss"


def encode_column_pair_token(presence_pattern, leaf_letter='A'):
    """Encode a column's presence/absence pattern as a pair-token string
    matching the ((A,B)Y,C)R binary-3-leaf tree's output alphabet.

    presence_pattern: tuple of bools (or 0/1) for leaves in tree-order
        (A, B, C).

    Encoding for ((A,B)Y,C)R:
        intersect produces nested [outer, inner] structure.
        ((A,B), C) -> [<A,B sub-token>], C_sym.
        A,B sub-token -> a_sym, b_sym (separator-only).
        Final: [a_sym,b_sym],c_sym
    """
    n = len(presence_pattern)
    if n != 3:
        raise NotImplementedError("only 3-leaf encoding here")
    a, b, c = [(leaf_letter if p else '') for p in presence_pattern]
    return f'[{a},{b}],{c}'


def boss_exact_log_p(tree_string, params, columns_pair_tokens):
    """Call boss to compute exact log p(MSA | params, tree) for the
    given phylo-HMM composition.

    tree_string: Newick, e.g., '(A:0.1,B:0.1,C:0.1)R;'.
    params: dict like {'delRate': 0.05, 'insRate': 0.04,
                       'time[A]': 0.1, 'time[B]': 0.1, 'time[C]': 0.1}.
    columns_pair_tokens: list of strings (one token per column).

    Returns: float log p.
    """
    cols_path = Path('/tmp/_mb_cols.json')
    params_path = Path('/tmp/_mb_params.json')
    cols_path.write_text(json.dumps({'sequence': columns_pair_tokens}))
    params_path.write_text(json.dumps(params))
    cmd = [
        BOSS, '--preset', 'tkf91-root-dna-jc', '-m',
        '--begin', '--preset', 'tkf91-branch-dna-jc',
        '--phylo-tree-string', tree_string,
        '--phylo-time-param', 'time', '--end',
        '--recognize-json', str(cols_path),
        '-P', str(params_path), '-L',
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"boss failed: {res.stderr}")
    out = json.loads(res.stdout.strip())
    if isinstance(out, list) and len(out) > 0:
        log_p = float(out[0][-1])
        return log_p
    raise RuntimeError(f"unexpected boss output: {res.stdout}")


def variational_elbo_at_truth(tree_node, leaf_present, ins, del_, ext,
                                 n_iter=200, lr=0.05, seed=0):
    """Run varanc_presence Adam from Fitch-low init, return final ELBO."""
    bt = _to_binary_tree(tree_node)
    L = leaf_present.shape[1]
    le, re = edge_lookup(bt)
    edge_lengths = np.maximum(np.asarray(bt.edge_length), 1e-3)
    bt_clipped = bt._replace(edge_length=edge_lengths)

    rng = np.random.default_rng(seed)
    # Random small init.
    edge_logits = jnp.asarray(
        rng.standard_normal((bt_clipped.num_edges, L, 2)) * 0.1,
        dtype=jnp.float64)
    root_logit = jnp.asarray(
        rng.standard_normal(()) * 0.1, dtype=jnp.float64)

    def neg_elbo(edge_logits, root_logit):
        root_logits = jnp.broadcast_to(root_logit, (L,))
        e, _ = varanc_elbo(edge_logits, root_logits, leaf_present, bt_clipped,
                           ins, del_, ext, le, re)
        return -e

    grad_fn = jax.jit(jax.grad(neg_elbo, argnums=(0, 1)))
    elbo_fn = jax.jit(neg_elbo)

    opt = optax.adam(lr)
    state = opt.init((edge_logits, root_logit))
    params = (edge_logits, root_logit)
    for _ in range(n_iter):
        grads = grad_fn(*params)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
    final_neg_elbo = float(elbo_fn(*params))
    return -final_neg_elbo  # ELBO


def main():
    # Setup: binary 3-leaf tree, TKF91 truth.
    tree_string = '((A:0.1,B:0.1)Y:0.1,C:0.1)R;'
    tree_node = parse_newick(tree_string)
    ins = 0.04
    del_ = 0.05
    ext = 0.0  # TKF91

    # Construct a small MSA.
    # Each column: 0/1 presence pattern for (A, B, C).
    # Mix of all-present, single-leaf-absent, all-but-one-absent:
    column_patterns = [
        (1, 1, 1),  # all present
        (1, 1, 1),
        (1, 0, 1),  # B absent
        (1, 1, 0),  # C absent
        (1, 1, 1),
        (0, 1, 1),  # A absent
        (1, 1, 1),
    ]
    leaf_present = np.array(column_patterns, dtype=np.int32).T  # (3, n_cols)
    print(f'Tree: {tree_string}')
    print(f'leaf_present shape: {leaf_present.shape}')
    print(f'columns:')
    for n, p in enumerate(column_patterns):
        print(f'  col {n}: {p}')

    # Boss exact log p.
    pair_tokens = [encode_column_pair_token(p) for p in column_patterns]
    print(f'\nPair tokens:')
    for n, t in enumerate(pair_tokens):
        print(f'  col {n}: {t!r}')

    boss_params = {
        'delRate': del_, 'insRate': ins,
        'time[A]': 0.1, 'time[B]': 0.1, 'time[Y]': 0.1, 'time[C]': 0.1,
    }
    log_p_exact = boss_exact_log_p(tree_string, boss_params, pair_tokens)
    print(f'\n  Exact log p (Machine Boss): {log_p_exact:.4f}')

    # Variational ELBO at truth params.
    elbo = variational_elbo_at_truth(tree_node, leaf_present, ins, del_, ext)
    print(f'  Variational ELBO:           {elbo:.4f}')
    print(f'  Variational gap (exact - ELBO): {log_p_exact - elbo:.4f}')

    # Save.
    out = {
        'tree': tree_string,
        'truth_params': {'ins': ins, 'del': del_, 'ext': ext},
        'columns': [list(p) for p in column_patterns],
        'pair_tokens': pair_tokens,
        'log_p_exact': log_p_exact,
        'elbo_at_truth_adam_converged': elbo,
        'variational_gap': log_p_exact - elbo,
    }
    out_path = 'experiments/figures/exact_elbo_machineboss.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
