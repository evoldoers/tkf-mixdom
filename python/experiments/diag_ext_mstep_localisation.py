#!/usr/bin/env python3
"""§2 localisation: where does the ext-recovery bias come from?

The empirical FB+M-step gives ext_hat ≈ 0.41 vs truth 0.5, persistently
across all N. Possible sources:

  (1) Bias in the FB E-step (FB-derived n_trans deviates from the true
      marginal E[n_trans] under TKF92).
  (2) Bias in tkf92_stats_from_counts (the responsibility-split + M-step
      formula has a defect).
  (3) Simulator generates a distribution slightly off TKF92 truth (so
      both FB and analytic disagree with sample).

Test: compute E[n_trans] ANALYTICALLY under TKF92 truth (no simulation,
no FB), feed to tkf92_stats_from_counts. If ext_hat ≈ 0.5 → bias is in
FB E-step. If ext_hat ≈ 0.41 → bias is in the M-step formula.

Method: TKF92 chi is a 5-state absorbing Markov chain (S start, MID
body, E absorb). Use the fundamental matrix N = (I - Q)^{-1} on body
states to get expected visit counts. E[n_trans[s, s']] = visits(s) *
chi[s, s'].
"""
from __future__ import annotations
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.core.params import S, M, I, D, E, tkf92_trans
from tkfmixdom.jax.core.bdi import tkf92_stats_from_counts


def expected_n_trans_chi(ins, mu, t, ext):
    """Compute the analytic per-pair E[n_trans[s, s']] under TKF92 chi.

    The chi chain is an absorbing Markov chain on {S, M, I, D, E}.
    State S is the start (deterministic; visited exactly once).
    State E is absorbing.
    Body states {M, I, D} are transient.

    Use the fundamental matrix N = (I - Q)^{-1} on transient states
    to get expected # visits to each body state starting from S.

    E[n_trans[s, s']] = (expected # visits to s) * chi[s, s']
    For s = S: visits = 1 (always start there).
    For s in body: visits = N[S, s] (from fundamental matrix).
    """
    chi = np.asarray(tkf92_trans(float(ins), float(mu), float(t), float(ext)))
    # Transient states are S, M, I, D (E is absorbing). Index 0=S, 1=M, 2=I, 3=D.
    transient = [S, M, I, D]
    Q = chi[np.ix_(transient, transient)]
    # Fundamental matrix N = (I - Q)^{-1}
    N = np.linalg.inv(np.eye(4) - Q)
    # Visits to each transient state starting from S (index 0 in transient list)
    s_idx_in_transient = transient.index(S)
    visits = N[s_idx_in_transient, :]  # (4,)
    # Map back to chi state indices
    visits_full = np.zeros(5)
    for i, s_orig in enumerate(transient):
        visits_full[s_orig] = visits[i]

    # E[n_trans[s, s']] = visits[s] * chi[s, s'] for s != E
    n_trans_exp = visits_full[:, None] * chi
    return n_trans_exp


def main():
    ins, mu, t, ext = 0.08, 0.12, 1.0, 0.5
    print(f'Truth: lam={ins}, mu={mu}, t={t}, ext={ext}\n')

    # 1. Compute analytic E[n_trans]
    n_trans_exp = expected_n_trans_chi(ins, mu, t, ext)
    LABELS = ['S', 'M', 'I', 'D', 'E']
    print('Analytic E[n_trans] under TKF92 truth (per pair):')
    print(f'{"":>3}', ''.join(f'{l:>10}' for l in LABELS))
    for i in range(5):
        print(f'{LABELS[i]:>3}',
              ''.join(f'{n_trans_exp[i,j]:>10.4f}' for j in range(5)))
    print()

    # 2. Feed to tkf92_stats_from_counts (using truth ext for the split)
    # The function expects integer-ish counts; scale by N_PAIRS for clarity.
    N_PAIRS = 1000
    n_trans_scaled = n_trans_exp * N_PAIRS
    T_total = N_PAIRS * t
    r = tkf92_stats_from_counts(n_trans_scaled, ins, mu, t, ext, T=T_total)
    ext_count = r['ext_count']
    notext_count = r['notext_count']
    ext_hat = ext_count / (ext_count + notext_count) if (ext_count + notext_count) > 0 else 0.0

    print(f'Per-pair counts scaled to N_PAIRS={N_PAIRS}:')
    print(f'  ext_count    = {ext_count:.4f}')
    print(f'  notext_count = {notext_count:.4f}')
    print(f'  ext_hat = ext_count / (ext_count + notext_count) = {ext_hat:.6f}')
    print(f'  truth ext = {ext}')
    print(f'  rel bias = ({ext_hat} - {ext}) / {ext} = {(ext_hat - ext)/ext:+.4f}')
    print()
    print(f'E_B (births) = {r["E_B"]:.4f}')
    print(f'E_D (deaths) = {r["E_D"]:.4f}')
    print(f'E_S (survival time) = {r["E_S"]:.4f}')
    print()

    # Verdict
    if abs(ext_hat - ext) < 0.01:
        print(f'VERDICT: ext_hat ≈ truth → M-step formula is unbiased.')
        print(f'         Bias must come from FB E-step (or simulator).')
    elif abs(ext_hat - 0.41) < 0.05:
        print(f'VERDICT: ext_hat ≈ 0.41 (matches empirical FB+M-step bias)')
        print(f'         → bias is INSIDE tkf92_stats_from_counts.')
        print(f'         FB E-step + simulator are fine.')
    else:
        print(f'VERDICT: ext_hat = {ext_hat:.4f}, neither truth nor empirical FB.')
        print(f'         Mixed mechanism; localisation incomplete.')

    # Also try at multiple ext_truth values to see the M-step's input-output map
    print()
    print('Input ext vs output ext_hat (M-step idempotency check):')
    print(f'{"ext_in":>8} | {"ext_hat":>8} | {"diff":>8}')
    for ext_in in [0.1, 0.3, 0.5, 0.7, 0.9]:
        n_exp_x = expected_n_trans_chi(ins, mu, t, ext_in) * N_PAIRS
        r_x = tkf92_stats_from_counts(n_exp_x, ins, mu, t, ext_in, T=T_total)
        ext_h = r_x['ext_count'] / (r_x['ext_count'] + r_x['notext_count'])
        print(f'{ext_in:>8.2f} | {ext_h:>8.4f} | {ext_h-ext_in:>+8.4f}')


if __name__ == '__main__':
    main()
