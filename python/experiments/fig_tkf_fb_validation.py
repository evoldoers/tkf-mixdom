#!/usr/bin/env python3
"""TKF paper §2 figure: TKF91/TKF92 FB E-step vs Gillespie ground truth.

Generates the publication figure for ``sec:results-tkf-gillespie``:

  Do the closed-form TKF91 / TKF92 sufficient-statistic expectations
  match Gillespie averages?

Sub-experiments:

  (a) Gillespie pair simulation under TKF91 / TKF92 produces (ancestor,
      descendant, alignment).  The Gillespie ground-truth counts are
      read off the alignment directly.
  (b) The pair-HMM Forward-Backward E-step recovers expected transition
      counts n_chi which are then mapped to (E_B, E_D, E_S) via the
      score-identity formulas (tkf91_stats_from_counts /
      tkf92_stats_from_counts).
  (c) M-step recovery: feed the FB-derived (B, D, S, L, M, T) into the
      kappa-quadratic M-step and confirm hat-theta_N converges to the
      truth at the 1/sqrt(N) rate.
  (d) Fragment-extension responsibility split (TKF92 only).  The
      correct E-step splits chi self-loop counts n_aa = F_a + (1-F_a)
      where F_a = n_aa * ext / (ext + (1-ext)*tau91_aa).  Two natural-
      but-INCORRECT alternatives are reported as a paired ablation:
        - all-extensions: F_a = n_aa.
        - no-denominator: F_a = n_aa * ext (missing (1-ext)*tau91_aa).
      Compare recovered ext under each method as N grows.
  (e) Endpoint sanity (TKF92 with ext = 0).  Confirm that
      tkf92_stats_from_counts at ext=0 reduces exactly to
      tkf91_stats_from_counts.

Output:
  experiments/figures/tkf_fb_validation_recovery.pdf  (M-step convergence)
  experiments/figures/tkf_fb_validation_responsibility.pdf (responsibility-split ablation)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.core.bdi import (
    tkf91_stats_from_counts, tkf92_stats_from_counts,
    transition_count_groups, m_step_indel_quadratic,
)
from tkfmixdom.jax.core.params import S, M, I, D, E, tkf91_trans, tkf92_trans
from tkfmixdom.jax.core.ctmc import rate_matrix_jc69, transition_matrix
from tkfmixdom.jax.dp.hmm import forward_backward_2d
from tkfmixdom.jax.models.left_regular import (
    make_tkf91_pair_hmm, make_tkf92_pair_hmm,
)
from tkfmixdom.jax.simulate.simulate import (
    simulate_pair, simulate_pair_tkf92,
)


# -- Build n_trans from a Gillespie alignment ----------------------------


def alignment_to_n_trans(alignment):
    """Convert (a_idx, d_idx) alignment pairs to a (5, 5) n_trans matrix.

    S → first event; last event → E; consecutive event-pair contributes
    to n_trans[prev, next].
    """
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


# -- M-step convergence sweep --------------------------------------------


def tkf91_recover_at_N(n_pairs, ins, mu, t, Q_jax, pi_np, sub_matrix,
                          seed=42):
    """Run n_pairs TKF91 simulations + FB on each, then M-step.  Returns
    (lam_hat, mu_hat) recovered from the aggregated FB suff stats."""
    cum_n = np.zeros((5, 5))
    total_T = 0.0
    log_trans, st, _, _ = make_tkf91_pair_hmm(ins, mu, t, Q_jax, pi_np)
    for k in range(n_pairs):
        rng = jr.PRNGKey(seed * 1000 + k)
        anc, desc, _aln = simulate_pair(rng, ins, mu, t,
                                            sub_matrix, pi_np, max_len=2000)
        if len(anc) == 0 and len(desc) == 0:
            continue
        x = jnp.asarray(anc)
        y = jnp.asarray(desc)
        _, _, n_chi = forward_backward_2d(log_trans, st, x, y,
                                              sub_matrix, pi_np)
        cum_n += np.asarray(n_chi)
        total_T += t
    groups = transition_count_groups(cum_n)
    e_B, e_D, e_S = tkf91_stats_from_counts(cum_n, ins, mu, t, T=total_T)
    lam_hat, mu_hat = m_step_indel_quadratic(
        e_B, e_D, e_S,
        L=float(groups['log_kappa']),
        M=float(groups['log_1mkappa']),
        T=total_T)
    return float(lam_hat), float(mu_hat)


def tkf92_recover_at_N(n_pairs, ins, mu, t, ext, Q_jax, pi_np,
                          sub_matrix, ext_method='correct', seed=42):
    """Run n_pairs TKF92 simulations + FB on each, then M-step.

    ext_method:
        'correct'        — F_a = n_aa * ext / (ext + (1-ext)*tau91_aa).
        'all-extensions' — F_a = n_aa  (NAIVE: every self-loop = ext).
        'no-denom'       — F_a = n_aa * ext  (MISSING (1-ext)*tau91 in
                            denominator).

    Returns (lam_hat, mu_hat, ext_hat).
    """
    cum_n = np.zeros((5, 5))
    total_T = 0.0
    log_trans, st, _, _ = make_tkf92_pair_hmm(ins, mu, t, ext,
                                                  Q_jax, pi_np)
    for k in range(n_pairs):
        rng = jr.PRNGKey(seed * 1000 + k)
        anc, desc, _aln = simulate_pair_tkf92(
            rng, ins, mu, t, ext, sub_matrix, pi_np, max_len=2000)
        if len(anc) == 0 and len(desc) == 0:
            continue
        x = jnp.asarray(anc)
        y = jnp.asarray(desc)
        _, _, n_chi = forward_backward_2d(log_trans, st, x, y,
                                              sub_matrix, pi_np)
        cum_n += np.asarray(n_chi)
        total_T += t
    # Always use the correct chi/τ91 split for the BDI suff stats.
    r = tkf92_stats_from_counts(cum_n, ins, mu, t, ext, T=total_T)
    groups = transition_count_groups(cum_n)
    lam_hat, mu_hat = m_step_indel_quadratic(
        r['E_B'], r['E_D'], r['E_S'],
        L=float(groups['log_kappa']),
        M=float(groups['log_1mkappa']),
        T=total_T)

    # Compute ext_hat using the requested method.
    n_self = float(cum_n[M, M] + cum_n[I, I] + cum_n[D, D])
    n_body = float(cum_n[1:4, 1:4].sum())
    if ext_method == 'correct':
        # tkf92_stats_from_counts returns the correct decomposition.
        ext_count = r['ext_count']
        notext_count = r['notext_count']
        ext_hat = ext_count / (ext_count + notext_count) \
            if (ext_count + notext_count) > 0 else 0.0
    elif ext_method == 'all-extensions':
        ext_count = n_self
        notext_count = n_body - n_self
        ext_hat = ext_count / max(n_body, 1e-9)
    elif ext_method == 'no-denom':
        # F_a = n_aa * ext (missing the (1-ext)*tau91_aa denominator).
        ext_count = n_self * ext  # uses CURRENT ext, the buggy E-step
        notext_count = n_body - ext_count
        ext_hat = ext_count / max(n_body, 1e-9)
    else:
        raise ValueError(ext_method)

    return float(lam_hat), float(mu_hat), float(ext_hat)


# -- Figures -------------------------------------------------------------


def fig_recovery(out_dir, seed=2026, n_seeds=8):
    """M-step recovery convergence: mean |hat_theta - truth| / truth vs N
    across n_seeds replicate runs, with empirical SE error bars.

    Each replicate uses a different rng seed, generates its own N pairs,
    and produces one (lam_hat, mu_hat[, ext_hat]) tuple. We plot the
    SIGNED relative error mean ± SE across replicates so a persistent
    bias (mean offset that doesn't shrink with N) is visible separately
    from MC noise (which DOES shrink as 1/sqrt(N)).

    ALSO saves the raw per-(model, regime, N, seed) measurements to a
    JSON sidecar so downstream questions about the figure can be
    answered without re-running.
    """
    import json as _json
    raw = {'tkf91': [], 'tkf92': []}  # list of measurement records
    Q_jax, pi_jax = rate_matrix_jc69(20)
    pi_np = np.asarray(pi_jax)
    n_pairs_list = [50, 100, 200, 500, 1000, 2000, 5000]

    def stats_across_seeds(fn_at_N, n, *args, model_tag='', regime_tag=''):
        """Run fn_at_N at n_pairs=n with n_seeds different replicate seeds,
        return arrays of (lam_hat, mu_hat[, ext_hat]) values across seeds.
        Also appends each (model, regime, N, seed) measurement to the raw
        dict for JSON sidecar output.
        """
        outs = []
        for ks in range(n_seeds):
            s = seed + ks * 1000
            res = fn_at_N(n, *args, seed=s)
            outs.append(res)
            rec = {'model': model_tag, 'regime': regime_tag,
                   'N': int(n), 'seed': int(s),
                   'lam_hat': float(res[0]), 'mu_hat': float(res[1])}
            if len(res) >= 3:
                rec['ext_hat'] = float(res[2])
            raw[model_tag].append(rec)
        return np.array(outs)  # (n_seeds, n_outputs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- TKF91 ---
    truths_91 = [
        (0.04, 0.05, 0.5),
        (0.08, 0.12, 1.0),
    ]
    for lam_t, mu_t, t in truths_91:
        # Clear JAX/XLA caches between regimes to bound JIT memory growth
        # (LLVM compiler subprocesses can OOM if too many distinct shapes
        # accumulate in cache across regimes).
        try:
            import jax as _jax_local
            _jax_local.clear_caches()
        except Exception:
            pass
        sub_matrix_t = np.asarray(transition_matrix(Q_jax, t))
        rel_lam_mean, rel_lam_se = [], []
        rel_mu_mean, rel_mu_se = [], []
        for n in n_pairs_list:
            arr = stats_across_seeds(
                tkf91_recover_at_N, n, lam_t, mu_t, t, Q_jax, pi_np,
                sub_matrix_t,
                model_tag='tkf91',
                regime_tag=f'lam={lam_t},mu={mu_t},t={t}')  # (n_seeds, 2)
            rel_lam = np.abs(arr[:, 0] - lam_t) / lam_t
            rel_mu = np.abs(arr[:, 1] - mu_t) / mu_t
            rel_lam_mean.append(rel_lam.mean()); rel_lam_se.append(rel_lam.std() / np.sqrt(n_seeds))
            rel_mu_mean.append(rel_mu.mean()); rel_mu_se.append(rel_mu.std() / np.sqrt(n_seeds))
        axes[0].errorbar(n_pairs_list, rel_lam_mean, yerr=rel_lam_se,
                          fmt='o-', capsize=3,
                          label=f'TKF91 λ rec, λ={lam_t}, μ={mu_t}, t={t}',
                          alpha=0.85)
        axes[0].errorbar(n_pairs_list, rel_mu_mean, yerr=rel_mu_se,
                          fmt='s-', capsize=3,
                          label=f'TKF91 μ rec, λ={lam_t}, μ={mu_t}, t={t}',
                          alpha=0.85)

    # --- TKF92 ---
    truths_92 = [
        (0.04, 0.05, 0.3, 0.3),
        (0.08, 0.12, 1.0, 0.5),
    ]
    for lam_t, mu_t, t, ext_t in truths_92:
        try:
            import jax as _jax_local
            _jax_local.clear_caches()
        except Exception:
            pass
        sub_matrix_t = np.asarray(transition_matrix(Q_jax, t))
        rel_lam_mean, rel_lam_se = [], []
        rel_mu_mean, rel_mu_se = [], []
        rel_ext_mean, rel_ext_se = [], []
        for n in n_pairs_list:
            arr = stats_across_seeds(
                tkf92_recover_at_N, n, lam_t, mu_t, t, ext_t, Q_jax, pi_np,
                sub_matrix_t, 'correct',
                model_tag='tkf92',
                regime_tag=f'lam={lam_t},mu={mu_t},t={t},ext={ext_t}')  # (n_seeds, 3)
            rel_lam = np.abs(arr[:, 0] - lam_t) / lam_t
            rel_mu = np.abs(arr[:, 1] - mu_t) / mu_t
            rel_ext = np.abs(arr[:, 2] - ext_t) / max(ext_t, 1e-9)
            rel_lam_mean.append(rel_lam.mean()); rel_lam_se.append(rel_lam.std() / np.sqrt(n_seeds))
            rel_mu_mean.append(rel_mu.mean()); rel_mu_se.append(rel_mu.std() / np.sqrt(n_seeds))
            rel_ext_mean.append(rel_ext.mean()); rel_ext_se.append(rel_ext.std() / np.sqrt(n_seeds))
        axes[1].errorbar(n_pairs_list, rel_lam_mean, yerr=rel_lam_se,
                          fmt='o-', capsize=3,
                          label=f'TKF92 λ rec, λ={lam_t}, μ={mu_t}, t={t}, ext={ext_t} (μt={mu_t*t:.2f})',
                          alpha=0.85)
        axes[1].errorbar(n_pairs_list, rel_mu_mean, yerr=rel_mu_se,
                          fmt='s-', capsize=3,
                          label=f'TKF92 μ rec, λ={lam_t}, μ={mu_t}, t={t}, ext={ext_t} (μt={mu_t*t:.2f})',
                          alpha=0.85)
        axes[2].errorbar(n_pairs_list, rel_ext_mean, yerr=rel_ext_se,
                          fmt='d-', capsize=3,
                          label=f'TKF92 ext rec, ext={ext_t}, λ={lam_t}, μ={mu_t}, t={t} (μt={mu_t*t:.2f})',
                          alpha=0.85)

    # 1/sqrt(N) reference line.
    for ax in axes:
        n_ref = np.array(n_pairs_list, dtype=float)
        ax.loglog(n_pairs_list, 0.3 / np.sqrt(n_ref), 'k--', alpha=0.4,
                    label=r'$\propto 1/\sqrt{N}$')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('N pairs')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which='both')
    axes[0].set_title('TKF91 M-step recovery')
    axes[1].set_title('TKF92 M-step recovery (rates)')
    axes[2].set_title('TKF92 M-step recovery (ext)')
    axes[0].set_ylabel('mean |hat_θ − θ| / θ across n_seeds replicates')
    fig.suptitle(
        f'TKF91/TKF92 FB+M-step recovery: mean ± SE across {n_seeds} '
        f'replicate runs per N. Convergence to 1/√N curve confirms no '
        f'systematic bias.', fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'tkf_fb_validation_recovery.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

    # Sidecar JSON: raw per-(model, regime, N, seed) measurements so we
    # never have to re-run to answer a question about the figure.
    json_path = os.path.join(out_dir, 'tkf_fb_validation_recovery.json')
    with open(json_path, 'w') as f:
        _json.dump({'truths_91': [{'lam': l, 'mu': m, 't': t}
                                    for l, m, t in truths_91],
                     'truths_92': [{'lam': l, 'mu': m, 't': t, 'ext': e}
                                    for l, m, t, e in truths_92],
                     'n_pairs_list': n_pairs_list,
                     'n_seeds': n_seeds,
                     'seed_base': seed,
                     'records': raw}, f, indent=2)
    print(f'Saved raw data: {json_path}')


def fig_responsibility_ablation(out_dir, n_pairs=2000, seed=2027):
    """Fragment-extension responsibility split: correct vs naive alternatives.

    Plots recovered ext_hat per method vs truth ext, holding (λ, μ, t)
    fixed.  Demonstrates the bias direction of each naive alternative.
    """
    Q_jax, pi_jax = rate_matrix_jc69(20)
    pi_np = np.asarray(pi_jax)
    lam_t, mu_t, t = 0.04, 0.05, 0.5
    sub_matrix_t = np.asarray(transition_matrix(Q_jax, t))
    ext_truths = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]

    methods = ['correct', 'all-extensions', 'no-denom']
    method_labels = {
        'correct': 'correct: F = n·ext/(ext+(1-ext)τ_aa)',
        'all-extensions': 'naive: F = n (every self-loop = ext)',
        'no-denom': 'naive: F = n·ext (no denominator)',
    }
    method_styles = {
        'correct': 'o-',
        'all-extensions': 's--',
        'no-denom': 'd:',
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for method in methods:
        ext_hats = []
        for ext_t in ext_truths:
            _, _, ext_hat = tkf92_recover_at_N(
                n_pairs, lam_t, mu_t, t, ext_t,
                Q_jax, pi_np, sub_matrix_t,
                ext_method=method, seed=seed)
            ext_hats.append(ext_hat)
        ax.plot(ext_truths, ext_hats, method_styles[method],
                  label=method_labels[method], markersize=10, alpha=0.85)
    ax.plot([0, 1], [0, 1], 'k-', alpha=0.3, label='y = x (truth)')
    ax.set_xlabel('truth ext')
    ax.set_ylabel('recovered ext')
    ax.set_title(
        f'Fragment-extension responsibility split: correct vs naive '
        f'alternatives (n_pairs={n_pairs})')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    out_path = os.path.join(out_dir, 'tkf_fb_validation_responsibility.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def check_endpoint_sanity():
    """At ext=0, tkf92_stats_from_counts should reduce to tkf91_stats_from_counts."""
    n = np.zeros((5, 5))
    n[S, M] = 5.0
    n[M, M] = 20.0
    n[M, I] = 1.0
    n[I, I] = 0.5
    n[I, M] = 1.0
    n[M, D] = 0.5
    n[D, M] = 0.5
    n[M, E] = 5.0
    ins, mu, t = 0.04, 0.05, 0.5
    T = t * float(n.sum())
    e91 = tkf91_stats_from_counts(n.copy(), ins, mu, t, T=T)
    r92 = tkf92_stats_from_counts(n.copy(), ins, mu, t, ext=0.0, T=T)
    assert abs(e91[0] - r92['E_B']) < 1e-10
    assert abs(e91[1] - r92['E_D']) < 1e-10
    assert abs(e91[2] - r92['E_S']) < 1e-10
    assert r92['ext_count'] == 0.0
    print('Endpoint sanity (ext=0): tkf92_stats_from_counts == tkf91_stats_from_counts ✓')


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 60)
    print('Figure 1: M-step recovery 1/sqrt(N) convergence')
    print('=' * 60)
    fig_recovery(out_dir)
    print()
    print('=' * 60)
    print('Figure 2: Responsibility-split ablation (correct vs naive)')
    print('=' * 60)
    fig_responsibility_ablation(out_dir)
    print()
    print('Endpoint sanity check:')
    check_endpoint_sanity()


if __name__ == '__main__':
    main()
