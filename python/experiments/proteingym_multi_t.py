#!/usr/bin/env python3
"""Multi-t and adaptive-t scoring for ProteinGym DMS indel benchmark.

Instead of scoring at a single fixed t, this script explores:
  1. Multi-t averaging: score at multiple t values and average log P(mut|wt,t)
  2. Max-t selection: pick the t that maximizes log P(wt) under the singlet model
  3. Geometric mean: average log-probs across a log-spaced t grid

The hypothesis is that different proteins evolve at different rates, and
a fixed t=0.1 is suboptimal for many proteins. HMMER implicitly adapts
to the protein's evolutionary context through its profile construction.

Usage:
    # Multi-t averaging (default: 8 log-spaced t values)
    python experiments/proteingym_multi_t.py \
        --params params/best/bw_d3f2_fullseed_15iter.npz

    # Test on a few assays first
    python experiments/proteingym_multi_t.py \
        --params params/best/bw_d3f2_fullseed_15iter.npz \
        --assays TCRG1_MOUSE_Tsuboyama_2023_1E0L_indels,SDA_BACSU_Tsuboyama_2023_1PV0_indels

    # Adaptive t: maximize singlet log P(wt) to pick optimal t per protein
    python experiments/proteingym_multi_t.py \
        --params params/best/bw_d3f2_fullseed_15iter.npz --adaptive
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
from scipy import stats as sp_stats

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}


def seq_to_int(seq):
    """Convert amino acid string to integer array. Unknown = -1."""
    return np.array([AA_TO_IDX.get(c, -1) for c in seq], dtype=int)


def build_pair_hmm(params, tau):
    """Build MixDom pair HMM (same as score_proteingym_indel.py)."""
    import jax.numpy as jnp
    from tkfmixdom.jax.models.mixdom import build_nested_trans, state_types as mixdom_state_types
    from tkfmixdom.jax.distill.maraschino import get_lg08, build_rate_matrix
    from tkfmixdom.jax.core.ctmc import transition_matrix
    from tkfmixdom.jax.dp.hmm import safe_log

    n_dom = len(params['dom_ins'])
    n_frag = params['frag_weights'].shape[1] if params['frag_weights'].ndim > 1 else 1

    chi, _ = build_nested_trans(
        jnp.array(float(params['main_ins'])),
        jnp.array(float(params['main_del'])),
        jnp.array(float(tau)),
        jnp.array(params['dom_ins']),
        jnp.array(params['dom_del']),
        jnp.array(params['dom_weights']),
        jnp.array(params['frag_weights']),
        jnp.array(params['ext_rates']),
    )
    log_chi = safe_log(chi)
    st = mixdom_state_types(n_dom, n_frag)

    S_lg, pi_lg = get_lg08()
    Q = build_rate_matrix(S_lg, pi_lg)
    sub_matrix = transition_matrix(Q, tau)

    if 'dom_pis' in params:
        pi = jnp.zeros(20)
        for d in range(n_dom):
            pi = pi + params['dom_weights'][d] * jnp.array(params['dom_pis'][d])
        pi = pi / pi.sum()
    else:
        pi = pi_lg

    return log_chi, st, sub_matrix, pi


def score_variant_multi_t(wt_jnp, mut_jnp, params, tau_values, mode='mean'):
    """Score a single variant at multiple t values.

    Args:
        mode: 'mean' (average log P), 'max' (best t), 'logsumexp' (marginal)
    """
    import jax.numpy as jnp
    from tkfmixdom.jax.dp.hmm import forward_2d

    log_probs = []
    for tau in tau_values:
        log_chi, st, sub_matrix, pi = build_pair_hmm(params, tau)
        log_prob, _ = forward_2d(log_chi, st, wt_jnp, mut_jnp, sub_matrix, pi)
        log_probs.append(float(log_prob))

    if mode == 'mean':
        return np.mean(log_probs)
    elif mode == 'max':
        return np.max(log_probs)
    elif mode == 'logsumexp':
        # log (1/K) sum_k exp(log P_k)  = logsumexp - log K
        from scipy.special import logsumexp
        return logsumexp(log_probs) - np.log(len(log_probs))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def score_wt_singlet(wt_int, params, tau):
    """Compute singlet log P(wt) at a given tau (for adaptive t selection)."""
    from tkfmixdom.jax.dp.singlet_forward import singlet_log_prob
    from score_proteingym_indel import build_singlet_hmm

    sing_start, sing_trans, sing_end, pis = build_singlet_hmm(params, tau)
    return singlet_log_prob(wt_int, sing_start, sing_trans, sing_end, pis)


def find_optimal_t(wt_int, params, t_grid):
    """Find t that maximizes singlet log P(wt)."""
    best_t = t_grid[0]
    best_ll = -np.inf
    for t in t_grid:
        ll = score_wt_singlet(wt_int, params, t)
        if ll > best_ll:
            best_ll = ll
            best_t = t
    return best_t, best_ll


def precompute_pair_hmms(params, tau_values):
    """Precompute pair HMMs for all tau values (avoids rebuilding per variant)."""
    hmms = {}
    for tau in tau_values:
        log_chi, st, sub_matrix, pi = build_pair_hmm(params, tau)
        hmms[tau] = (log_chi, st, sub_matrix, pi)
    return hmms


def score_assay_multi_t(assay_csv, params, tau_values, hmms, wt_seq=None,
                        mode='mean', adaptive_t=None):
    """Score all variants in an assay using multi-t or adaptive-t scoring.

    Args:
        hmms: dict of tau -> (log_chi, st, sub_matrix, pi), precomputed
        mode: scoring mode for multi-t
        adaptive_t: if set, use this single t value (overrides multi-t)

    Returns:
        (predictions, dms_scores, n_scored)
    """
    import jax.numpy as jnp
    from tkfmixdom.jax.dp.hmm import forward_2d

    with open(assay_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], [], 0

    if wt_seq is None:
        ref_csv = os.path.join(os.path.dirname(assay_csv), '..', 'DMS_indels.csv')
        assay_name = os.path.basename(assay_csv).replace('.csv', '')
        if os.path.exists(ref_csv):
            with open(ref_csv) as f:
                for row in csv.DictReader(f):
                    if row['DMS_id'] == assay_name:
                        wt_seq = row['target_seq']
                        break
    if wt_seq is None:
        return [], [], 0

    wt_int = seq_to_int(wt_seq)
    wt_int = wt_int[wt_int >= 0]
    wt_jnp = jnp.array(wt_int)

    if adaptive_t is not None:
        # Use single adaptive t
        use_taus = [adaptive_t]
        use_hmms = {adaptive_t: hmms.get(adaptive_t, build_pair_hmm(params, adaptive_t))}
    else:
        use_taus = tau_values
        use_hmms = hmms

    predictions = []
    dms_scores = []
    n_scored = 0

    for i, row in enumerate(rows):
        mut_seq = row['mutated_sequence']
        dms_score = float(row['DMS_score'])

        mut_int = seq_to_int(mut_seq)
        mut_int = mut_int[mut_int >= 0]
        if len(mut_int) == 0:
            continue

        mut_jnp = jnp.array(mut_int)

        log_probs = []
        for tau in use_taus:
            log_chi, st, sub_matrix, pi = use_hmms[tau]
            lp, _ = forward_2d(log_chi, st, wt_jnp, mut_jnp, sub_matrix, pi)
            log_probs.append(float(lp))

        if mode == 'mean' or len(log_probs) == 1:
            score = np.mean(log_probs)
        elif mode == 'max':
            score = np.max(log_probs)
        elif mode == 'logsumexp':
            from scipy.special import logsumexp
            score = logsumexp(log_probs) - np.log(len(log_probs))
        else:
            score = np.mean(log_probs)

        predictions.append(score)
        dms_scores.append(dms_score)
        n_scored += 1

        if n_scored % 100 == 0:
            print(f"    scored {n_scored}/{len(rows)} variants...", flush=True)

    return predictions, dms_scores, n_scored


def main():
    parser = argparse.ArgumentParser(
        description='Multi-t scoring for ProteinGym indel variants')
    parser.add_argument('--params', required=True, help='MixDom params .npz')
    parser.add_argument('--assays', type=str, default=None,
                        help='Comma-separated assay names (default: all)')
    parser.add_argument('--proteingym-dir', type=str,
                        default=os.path.expanduser(
                            '~/bio-datasets/data/proteingym/DMS_ProteinGym_indels/'))
    parser.add_argument('--ref-csv', type=str,
                        default=os.path.expanduser(
                            '~/bio-datasets/data/proteingym/DMS_indels.csv'))
    parser.add_argument('--out', type=str, default=None, help='Output CSV')
    parser.add_argument('--mode', type=str, default='mean',
                        choices=['mean', 'max', 'logsumexp'],
                        help='How to combine scores across t values')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive t: pick t that maximizes singlet log P(wt)')
    parser.add_argument('--t-grid', type=str, default=None,
                        help='Comma-separated t values (default: log-spaced 0.01-2.0)')
    parser.add_argument('--n-t', type=int, default=8,
                        help='Number of log-spaced t values (default: 8)')
    parser.add_argument('--short-only', action='store_true',
                        help='Only score assays with seq_len <= 150')
    args = parser.parse_args()

    # Load params
    print(f"Loading params: {args.params}")
    d = np.load(args.params, allow_pickle=True)
    params = {
        'main_ins': float(d['main_ins']), 'main_del': float(d['main_del']),
        'dom_ins': d['dom_ins'], 'dom_del': d['dom_del'],
        'dom_weights': d['dom_weights'], 'frag_weights': d['frag_weights'],
        'ext_rates': d['ext_rates'],
    }
    if 'dom_pis' in d:
        params['dom_pis'] = d['dom_pis']

    # Build t grid
    if args.t_grid:
        tau_values = [float(x) for x in args.t_grid.split(',')]
    else:
        tau_values = list(np.geomspace(0.01, 2.0, args.n_t))
    print(f"t values: {[f'{t:.4f}' for t in tau_values]}")

    # Get assay list
    if args.assays:
        assay_names = [a.strip() for a in args.assays.split(',')]
    else:
        assay_names = [f.replace('.csv', '') for f in
                       sorted(os.listdir(args.proteingym_dir)) if f.endswith('.csv')]

    # Load reference for wildtype sequences and seq lengths
    ref = {}
    ref_len = {}
    if os.path.exists(args.ref_csv):
        with open(args.ref_csv) as f:
            for row in csv.DictReader(f):
                ref[row['DMS_id']] = row['target_seq']
                ref_len[row['DMS_id']] = int(row.get('seq_len', 0))

    if args.short_only:
        assay_names = [a for a in assay_names if ref_len.get(a, 9999) <= 150]
        print(f"Short proteins only (<=150 aa): {len(assay_names)} assays")

    # Precompute pair HMMs for all t values
    print(f"Precomputing {len(tau_values)} pair HMMs...", flush=True)
    hmms = precompute_pair_hmms(params, tau_values)
    print(f"  done.", flush=True)

    # For adaptive mode, also prepare a fine t grid for optimization
    if args.adaptive:
        t_opt_grid = list(np.geomspace(0.005, 3.0, 30))
        print(f"Adaptive mode: optimizing t per protein over {len(t_opt_grid)} values")

    # Score each assay
    all_results = {}  # mode -> list of result dicts

    # We'll compare: single-t sweep, multi-t mean, and optionally adaptive
    modes_to_try = ['single_best', args.mode]
    if args.adaptive:
        modes_to_try.append('adaptive')

    # First pass: score each assay at each individual t to find single-best
    print(f"\n=== Scoring {len(assay_names)} assays ===")
    for assay_name in assay_names:
        csv_path = os.path.join(args.proteingym_dir, assay_name + '.csv')
        if not os.path.exists(csv_path):
            print(f"  {assay_name}: not found, skipping")
            continue

        wt_seq = ref.get(assay_name)
        if not wt_seq:
            print(f"  {assay_name}: no wildtype, skipping")
            continue

        wt_int = seq_to_int(wt_seq)
        wt_int = wt_int[wt_int >= 0]
        seq_len = len(wt_int)

        print(f"\n  {assay_name} (L={seq_len}):", flush=True)
        t0 = time.time()

        # --- Single-t sweep (baseline) ---
        single_t_results = {}
        for tau in tau_values:
            preds, scores, n = score_assay_multi_t(
                csv_path, params, [tau], {tau: hmms[tau]}, wt_seq, mode='mean')
            if len(preds) >= 10:
                rho, _ = sp_stats.spearmanr(preds, scores)
                single_t_results[tau] = rho
        if single_t_results:
            best_single_t = max(single_t_results, key=single_t_results.get)
            best_single_rho = single_t_results[best_single_t]
            print(f"    Single-t sweep: best t={best_single_t:.4f}, rho={best_single_rho:.4f}")
            for tau in sorted(single_t_results):
                print(f"      t={tau:.4f}: rho={single_t_results[tau]:.4f}")

        # --- Multi-t combined ---
        preds, scores, n = score_assay_multi_t(
            csv_path, params, tau_values, hmms, wt_seq, mode=args.mode)
        if len(preds) >= 10:
            multi_rho, multi_pval = sp_stats.spearmanr(preds, scores)
            print(f"    Multi-t {args.mode}: rho={multi_rho:.4f}")
        else:
            multi_rho = None
            print(f"    Multi-t: too few variants ({n})")

        # --- Adaptive t ---
        adaptive_rho = None
        adaptive_t_val = None
        if args.adaptive:
            adaptive_t_val, _ = find_optimal_t(wt_int, params, t_opt_grid)
            print(f"    Adaptive t: optimal t={adaptive_t_val:.4f}")
            # Build HMM at optimal t if not in cache
            if adaptive_t_val not in hmms:
                hmms[adaptive_t_val] = build_pair_hmm(params, adaptive_t_val)
            preds_a, scores_a, n_a = score_assay_multi_t(
                csv_path, params, [adaptive_t_val],
                {adaptive_t_val: hmms[adaptive_t_val]}, wt_seq, mode='mean')
            if len(preds_a) >= 10:
                adaptive_rho, _ = sp_stats.spearmanr(preds_a, scores_a)
                print(f"    Adaptive: rho={adaptive_rho:.4f}")

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.1f}s")

        # Collect results
        result = {
            'assay': assay_name,
            'seq_len': seq_len,
            'n_variants': n,
            'time': elapsed,
            'best_single_t': best_single_t if single_t_results else None,
            'best_single_rho': best_single_rho if single_t_results else None,
            'multi_t_mode': args.mode,
            'multi_t_rho': multi_rho,
        }
        if args.adaptive:
            result['adaptive_t'] = adaptive_t_val
            result['adaptive_rho'] = adaptive_rho

        # Add per-t rhos
        for tau in tau_values:
            result[f'rho_t{tau:.4f}'] = single_t_results.get(tau)

        all_results.setdefault('combined', []).append(result)

    # Summary
    results = all_results.get('combined', [])
    if results:
        print(f"\n{'='*70}")
        print(f"SUMMARY ({len(results)} assays)")
        print(f"{'='*70}")

        single_rhos = [r['best_single_rho'] for r in results
                       if r['best_single_rho'] is not None]
        multi_rhos = [r['multi_t_rho'] for r in results
                      if r['multi_t_rho'] is not None]

        if single_rhos:
            print(f"  Best single-t:  mean rho = {np.mean(single_rhos):.4f} "
                  f"+/- {np.std(single_rhos):.4f}")
        if multi_rhos:
            print(f"  Multi-t {args.mode:8s}: mean rho = {np.mean(multi_rhos):.4f} "
                  f"+/- {np.std(multi_rhos):.4f}")

        if args.adaptive:
            adap_rhos = [r['adaptive_rho'] for r in results
                         if r.get('adaptive_rho') is not None]
            if adap_rhos:
                print(f"  Adaptive t:     mean rho = {np.mean(adap_rhos):.4f} "
                      f"+/- {np.std(adap_rhos):.4f}")

        # Per-assay comparison table
        print(f"\n{'Assay':55s} {'Single':>8s} {'Multi':>8s}", end='')
        if args.adaptive:
            print(f" {'Adaptive':>8s}", end='')
        print()
        print('-' * 80)
        for r in sorted(results, key=lambda x: x.get('multi_t_rho') or 0,
                         reverse=True):
            s = f"{r['best_single_rho']:+.4f}" if r['best_single_rho'] is not None else '    N/A'
            m = f"{r['multi_t_rho']:+.4f}" if r['multi_t_rho'] is not None else '    N/A'
            line = f"{r['assay']:55s} {s:>8s} {m:>8s}"
            if args.adaptive:
                a = f"{r['adaptive_rho']:+.4f}" if r.get('adaptive_rho') is not None else '    N/A'
                line += f" {a:>8s}"
            print(line)

    # Save results
    if args.out and results:
        fieldnames = ['assay', 'seq_len', 'n_variants', 'time',
                      'best_single_t', 'best_single_rho',
                      'multi_t_mode', 'multi_t_rho']
        if args.adaptive:
            fieldnames.extend(['adaptive_t', 'adaptive_rho'])
        for tau in tau_values:
            fieldnames.append(f'rho_t{tau:.4f}')

        with open(args.out, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
