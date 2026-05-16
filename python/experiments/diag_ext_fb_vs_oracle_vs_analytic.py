#!/usr/bin/env python3
"""§2 deeper localisation: now that the M-step formula is proven
unbiased (diag_ext_mstep_localisation.py), where does the FB+M-step
bias actually come from?

Three candidate sources:
  (a) FB-derived n_trans biased relative to the simulator's true n_trans.
  (b) Simulator's true n_trans biased relative to TKF92 analytic E[n_trans].
  (c) Both, somehow.

Test: simulate N_PAIRS TKF92 pairs at truth. For each pair, compute
  - oracle_n_trans (from the simulator's labelled alignment)
  - fb_n_trans    (from forward-backward on (x, y) given truth params)
Sum across pairs. Compare to analytic E[n_trans] * N_PAIRS.

For each of the three n_trans matrices (analytic, oracle, fb), run
through tkf92_stats_from_counts → ext_hat to localise the bias.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np
import jax.numpy as jnp
import jax.random as jr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'level3_thorough'))

from tkfmixdom.jax.core.params import S, M, I, D, E
from tkfmixdom.jax.core.bdi import tkf92_stats_from_counts
from tkfmixdom.jax.core.ctmc import rate_matrix_jc69, transition_matrix
from tkfmixdom.jax.dp.hmm import forward_backward_2d
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm
from tkfmixdom.jax.simulate.simulate import simulate_pair_tkf92

# Re-use the analytic computation from the prior diag.
from experiments.diag_ext_mstep_localisation import expected_n_trans_chi


def alignment_to_n_trans_oracle(alignment):
    """Build (5,5) n_trans from a Gillespie pair alignment list of
    (anc_idx, desc_idx) tuples. Same convention as
    fig_tkf_fb_validation.alignment_to_n_trans (S→first, last→E,
    consecutive ops contribute to n[prev, next])."""
    n = np.zeros((5, 5), dtype=np.float64)
    prev = S
    for a_idx, d_idx in alignment:
        if a_idx is not None and d_idx is not None:
            nxt = M
        elif a_idx is None and d_idx is not None:
            nxt = I
        elif a_idx is not None and d_idx is None:
            nxt = D
        else:
            continue
        n[prev, nxt] += 1.0
        prev = nxt
    n[prev, E] += 1.0
    return n


def main():
    ins, mu, t, ext = 0.08, 0.12, 1.0, 0.5
    N_PAIRS = 1000
    SEED = 2026

    Q_jax, pi_jax = rate_matrix_jc69(20)
    pi_np = np.asarray(pi_jax)
    sub_matrix = np.asarray(transition_matrix(Q_jax, t))

    print(f'Truth: lam={ins}, mu={mu}, t={t}, ext={ext}, N_PAIRS={N_PAIRS}')
    print()

    # 1. Analytic
    n_trans_analytic = expected_n_trans_chi(ins, mu, t, ext) * N_PAIRS

    # 2. Oracle + FB
    log_trans, st, _, _ = make_tkf92_pair_hmm(ins, mu, t, ext, Q_jax, pi_np)
    n_trans_oracle = np.zeros((5, 5))
    n_trans_fb = np.zeros((5, 5))
    n_skipped = 0
    for k in range(N_PAIRS):
        rng = jr.PRNGKey(SEED * 1000 + k)
        anc, desc, aln = simulate_pair_tkf92(rng, ins, mu, t, ext,
                                                sub_matrix, pi_np, max_len=2000)
        if len(anc) == 0 and len(desc) == 0:
            n_skipped += 1
            continue
        x = jnp.asarray(anc); y = jnp.asarray(desc)
        n_trans_oracle += alignment_to_n_trans_oracle(aln)
        _, _, n_chi = forward_backward_2d(log_trans, st, x, y, sub_matrix, pi_np)
        n_trans_fb += np.asarray(n_chi)

    print(f'(skipped {n_skipped} empty pairs)\n')

    LABELS = ['S', 'M', 'I', 'D', 'E']

    def show(name, n):
        print(f'{name}:')
        print(f'{"":>3}', ''.join(f'{l:>10}' for l in LABELS))
        for i in range(5):
            print(f'{LABELS[i]:>3}',
                  ''.join(f'{n[i,j]:>10.2f}' for j in range(5)))
        print()

    show('Analytic E[n_trans] * N_PAIRS', n_trans_analytic)
    show('Oracle n_trans (sum over N_PAIRS)', n_trans_oracle)
    show('FB n_trans (sum over N_PAIRS)', n_trans_fb)

    # 3. Per-entry differences
    print('Per-entry differences (oracle - analytic, fb - analytic):')
    print(f'  Entry  | analytic |   oracle |       fb |  Δ_orac |   Δ_fb')
    for i in range(5):
        for j in range(5):
            a = n_trans_analytic[i, j]
            o = n_trans_oracle[i, j]
            f = n_trans_fb[i, j]
            if a > 1.0 or o > 1.0 or f > 1.0:
                print(f'  {LABELS[i]}->{LABELS[j]}   | {a:>8.2f} | {o:>8.2f} | '
                      f'{f:>8.2f} | {o-a:>+7.2f} | {f-a:>+7.2f}')
    print()

    # 4. Run M-step on each
    T_total = N_PAIRS * t
    for tag, n in [('analytic', n_trans_analytic),
                    ('oracle  ', n_trans_oracle),
                    ('FB      ', n_trans_fb)]:
        r = tkf92_stats_from_counts(n, ins, mu, t, ext, T=T_total)
        ext_h = r['ext_count'] / (r['ext_count'] + r['notext_count'])
        print(f'  {tag} → ext_hat = {ext_h:.6f}, '
              f'rel_bias = {(ext_h - ext)/ext:+.4f}')

    # 5. Verdict
    print()
    r_o = tkf92_stats_from_counts(n_trans_oracle, ins, mu, t, ext, T=T_total)
    r_f = tkf92_stats_from_counts(n_trans_fb, ins, mu, t, ext, T=T_total)
    ext_o = r_o['ext_count'] / (r_o['ext_count'] + r_o['notext_count'])
    ext_f = r_f['ext_count'] / (r_f['ext_count'] + r_f['notext_count'])
    if abs(ext_o - ext) < 0.02:
        if abs(ext_f - ext) < 0.02:
            print('VERDICT: oracle and FB both ≈ truth.  '
                  'Bias is NOT in simulator or FB; mystery elsewhere.')
        else:
            print(f'VERDICT: oracle ≈ truth ({ext_o:.4f}), but FB biased '
                  f'({ext_f:.4f}). The bias is in the FB E-step.')
    else:
        if abs(ext_f - ext_o) < 0.02:
            print(f'VERDICT: oracle and FB BOTH biased ({ext_o:.4f}, {ext_f:.4f}). '
                  f'The bias is in the simulator (off-distribution from analytic).')
        else:
            print(f'VERDICT: oracle ({ext_o:.4f}) ≠ FB ({ext_f:.4f}) ≠ truth.  '
                  f'Both simulator and FB contribute.')


if __name__ == '__main__':
    main()
