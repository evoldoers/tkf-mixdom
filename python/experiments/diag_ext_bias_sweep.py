"""§2 ext-recovery bias sweep: localise + characterise.

Build a body of evidence on the FB ext-recovery bias remaining after
the simulator stationary-fragment fix.

For each (lam, mu, t, ext_truth) regime, compute:
  * analytic n_trans (TKF92 chi-chain absorbing-Markov fundamental matrix)
  * oracle n_trans (from simulator's labelled alignments)
  * FB n_trans (from forward-backward on (x, y))
Then feed each to the closed-form M-step and report ext_hat.

Sweeps:
  A) ext sweep at fixed (lam, mu, t) = (0.04, 0.05, 0.3)
       ext ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}
  B) t sweep at fixed (lam, mu, ext) = (0.04, 0.05, 0.3)
       t ∈ {0.1, 0.3, 1.0, 3.0}
  C) High-N asymptote at the biased point (0.04, 0.05, 0.3, 0.3)
       N ∈ {500, 2000, 10000, 50000}

For each cell, we report:
  ext_hat from analytic (M-step formula sanity)
  ext_hat from oracle (simulator-vs-analytic discrepancy)
  ext_hat from FB (FB E-step bias)
  These three diagnostics together localise the bias.

Saves results JSON for later analysis + paper figure.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np
import jax.numpy as jnp
import jax.random as jr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.core.params import S, M, I, D, E, tkf92_trans
from tkfmixdom.jax.core.bdi import tkf92_stats_from_counts
from tkfmixdom.jax.core.ctmc import rate_matrix_jc69, transition_matrix
from tkfmixdom.jax.dp.hmm import forward_backward_2d
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm
from tkfmixdom.jax.simulate.simulate import simulate_pair_tkf92
from experiments.diag_ext_mstep_localisation import expected_n_trans_chi


def alignment_to_n_trans_oracle(alignment):
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


def run_cell(ins, mu, t, ext, n_pairs, seed_base=2026):
    """Run one (ins, mu, t, ext, n_pairs) cell. Return diag dict."""
    Q_jax, _ = rate_matrix_jc69(20)
    pi_np = np.full(20, 1.0/20)
    sub_matrix = np.asarray(transition_matrix(Q_jax, t))
    log_trans, st, _, _ = make_tkf92_pair_hmm(ins, mu, t, ext, Q_jax, pi_np)

    n_trans_analytic = expected_n_trans_chi(ins, mu, t, ext) * n_pairs
    n_trans_oracle = np.zeros((5, 5))
    n_trans_fb = np.zeros((5, 5))
    n_skipped = 0
    n_match_oracle = 0
    n_match_fb = 0.0
    for k in range(n_pairs):
        rng = jr.PRNGKey(seed_base * 1000 + k)
        anc, desc, aln = simulate_pair_tkf92(rng, ins, mu, t, ext,
                                              sub_matrix, pi_np, max_len=2000)
        if len(anc) == 0 and len(desc) == 0:
            n_skipped += 1
            continue
        x = jnp.asarray(anc); y = jnp.asarray(desc)
        oracle = alignment_to_n_trans_oracle(aln)
        n_trans_oracle += oracle
        n_match_oracle += int(oracle[M, M])
        try:
            _, _, n_chi = forward_backward_2d(log_trans, st, x, y, sub_matrix, pi_np)
            n_trans_fb += np.asarray(n_chi)
            n_match_fb += float(np.asarray(n_chi)[M, M])
        except Exception:
            pass

    T_total = n_pairs * t
    extracts = {}
    for tag, n in [('analytic', n_trans_analytic),
                    ('oracle', n_trans_oracle),
                    ('fb', n_trans_fb)]:
        r = tkf92_stats_from_counts(n, ins, mu, t, ext, T=T_total)
        denom = r['ext_count'] + r['notext_count']
        eh = float(r['ext_count'] / denom) if denom > 0 else 0.0
        extracts[tag] = {
            'ext_hat': eh,
            'rel_bias': float((eh - ext) / max(ext, 1e-9)) if ext > 0 else float(eh),
            'n_trans': n.tolist(),
            'ext_count': float(r['ext_count']),
            'notext_count': float(r['notext_count']),
            'E_B': float(r['E_B']),
            'E_D': float(r['E_D']),
            'E_S': float(r['E_S']),
        }

    return {
        'ins': ins, 'mu': mu, 't': t, 'ext_truth': ext, 'n_pairs': n_pairs,
        'seed_base': seed_base, 'n_skipped': n_skipped,
        'n_match_oracle_per_pair': n_match_oracle / max(n_pairs - n_skipped, 1),
        'n_match_fb_per_pair': n_match_fb / max(n_pairs - n_skipped, 1),
        **{k: v for k, v in extracts.items()},
    }


def main():
    out_path = 'experiments/figures/diag_ext_bias_sweep.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Sweep A: ext sweep at fixed (lam, mu, t).
    print("=== SWEEP A: ext sweep at (lam, mu, t) = (0.04, 0.05, 0.3) ===")
    sweep_A = []
    for ext in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        t0 = time.time()
        cell = run_cell(0.04, 0.05, 0.3, ext, n_pairs=2000)
        dt = time.time() - t0
        a = cell['analytic']['ext_hat']
        o = cell['oracle']['ext_hat']
        f = cell['fb']['ext_hat']
        print(f"  ext={ext}: analytic={a:.4f} ({(a-ext):+.4f}), "
              f"oracle={o:.4f} ({(o-ext):+.4f}), "
              f"FB={f:.4f} ({(f-ext):+.4f})  ({dt:.0f}s)",
              flush=True)
        sweep_A.append(cell)

    # Sweep B: t sweep at fixed (lam, mu, ext).
    print()
    print("=== SWEEP B: t sweep at (lam, mu, ext) = (0.04, 0.05, 0.3) ===")
    sweep_B = []
    for t in [0.1, 0.3, 1.0, 3.0]:
        t0 = time.time()
        cell = run_cell(0.04, 0.05, t, 0.3, n_pairs=2000)
        dt = time.time() - t0
        a = cell['analytic']['ext_hat']
        o = cell['oracle']['ext_hat']
        f = cell['fb']['ext_hat']
        print(f"  t={t}: analytic={a:.4f} ({(a-0.3):+.4f}), "
              f"oracle={o:.4f} ({(o-0.3):+.4f}), "
              f"FB={f:.4f} ({(f-0.3):+.4f})  ({dt:.0f}s)",
              flush=True)
        sweep_B.append(cell)

    # Sweep C: high-N asymptote at the original biased regime
    print()
    print("=== SWEEP C: high-N asymptote at (0.04, 0.05, 0.3, 0.3) ===")
    sweep_C = []
    for n_pairs in [500, 2000, 10000, 50000]:
        t0 = time.time()
        cell = run_cell(0.04, 0.05, 0.3, 0.3, n_pairs=n_pairs)
        dt = time.time() - t0
        a = cell['analytic']['ext_hat']
        o = cell['oracle']['ext_hat']
        f = cell['fb']['ext_hat']
        print(f"  N={n_pairs}: analytic={a:.4f}, oracle={o:.4f} ({(o-0.3):+.4f}), "
              f"FB={f:.4f} ({(f-0.3):+.4f})  ({dt:.0f}s)",
              flush=True)
        sweep_C.append(cell)

    out = {'sweep_A': sweep_A, 'sweep_B': sweep_B, 'sweep_C': sweep_C}
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
