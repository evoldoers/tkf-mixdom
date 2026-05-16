#!/usr/bin/env python3
"""Score ProteinGym DMS indel variants using MixDom singlet or pair HMM.

Singlet mode (default):
  score = log P(mut_seq) - log P(wt_seq)

Pair mode (--pair):
  score = log P(wt_seq, mut_seq | t)
  Uses MixDom pair HMM to ask "how likely is this mutant as an
  evolutionary descendant of the wildtype at time t?"
  Since P(wt) is constant for all variants of the same protein,
  the joint score ranks identically to the conditional P(mut|wt,t).

Usage:
    # Singlet mode
    python score_proteingym_indel.py --params params/best/bw_d3f2_fullseed_15iter.npz \
        --assays A4_HUMAN_Seuma_2022_indels

    # Pair mode (sweep tau values)
    python score_proteingym_indel.py --params params/best/bw_d3f2_fullseed_15iter.npz \
        --pair --assays A4_HUMAN_Seuma_2022_indels

    # Pair mode (fixed tau)
    python score_proteingym_indel.py --params params/best/bw_d3f2_fullseed_15iter.npz \
        --pair --tau 0.1 --assays A4_HUMAN_Seuma_2022_indels
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
from scipy import stats

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}


def seq_to_int(seq):
    """Convert amino acid string to integer array. Unknown = -1."""
    return np.array([AA_TO_IDX.get(c, -1) for c in seq], dtype=int)


def build_singlet_hmm(params, tau=1.0):
    """Build singlet HMM from MixDom params.

    Returns (sing_start, sing_trans, sing_end, pis).
    """
    import jax.numpy as jnp
    from tkfmixdom.jax.distill.maraschino import (
        distill_mixdom, bdi_params, get_lg08, build_rate_matrix,
        eigen_decompose, gamma_rates)

    n_dom = len(params['dom_ins'])
    n_frag = params['frag_weights'].shape[1] if params['frag_weights'].ndim > 1 else 1

    # Build maraschino-compatible params
    S_lg, pi_lg = get_lg08()

    # Expand domains × fragments
    N = n_dom * n_frag
    lam = np.zeros(N)
    mu = np.zeros(N)
    r = np.zeros(N)
    v = np.zeros(N)
    pis_out = np.zeros((N, 20))

    for d in range(n_dom):
        for f in range(n_frag):
            idx = d * n_frag + f
            lam[idx] = params['dom_ins'][d]
            mu[idx] = params['dom_del'][d]
            r[idx] = params['ext_rates'][d, f] if params['ext_rates'].ndim > 1 else params['ext_rates'][d]
            v[idx] = params['dom_weights'][d] * (
                params['frag_weights'][d, f] if params['frag_weights'].ndim > 1 else 1.0)
            if 'dom_pis' in params:
                pis_out[idx] = params['dom_pis'][d]
            else:
                pis_out[idx] = np.array(pi_lg)

    v = v / v.sum()

    mara_params = {
        'lam0': jnp.array(float(params['main_ins'])),
        'mu0': jnp.array(float(params['main_del'])),
        'lam': jnp.array(lam),
        'mu': jnp.array(mu),
        'r': jnp.array(r),
        'v': jnp.array(v),
        'pi': jnp.array(pis_out),
        'S_exch': S_lg,
        'alpha_gamma': jnp.array(1.0),
    }

    dist = distill_mixdom(mara_params, tau, 1)

    return (np.array(dist['f_singlet_start']),
            np.array(dist['f_singlet']),
            np.array(dist['f_singlet_end']),
            pis_out)


def build_singlet_hmm_direct(params):
    """Build singlet HMM directly from MixDom params (no distillation).

    Simpler approach: use domain weights as start probs, domain
    self-loops as transitions, equilibrium freqs as emissions.
    """
    from tkfmixdom.jax.core.params import tkf_kappa, tkf_beta

    n_dom = len(params['dom_ins'])
    n_frag = params['frag_weights'].shape[1] if params['frag_weights'].ndim > 1 else 1

    N = n_dom * n_frag
    kappa0 = params['main_ins'] / max(params['main_del'], 1e-10)

    # Per expanded-domain params
    lam = np.zeros(N)
    mu = np.zeros(N)
    r = np.zeros(N)
    v = np.zeros(N)
    pis = np.zeros((N, 20))

    for d in range(n_dom):
        for f in range(n_frag):
            idx = d * n_frag + f
            lam[idx] = params['dom_ins'][d]
            mu[idx] = params['dom_del'][d]
            ext = params['ext_rates'][d, f] if params['ext_rates'].ndim > 1 else params['ext_rates'][d]
            r[idx] = ext
            v[idx] = params['dom_weights'][d] * (
                params['frag_weights'][d, f] if params['frag_weights'].ndim > 1 else 1.0)
            if 'dom_pis' in params:
                pis[idx] = params['dom_pis'][d]
            else:
                from tkfmixdom.jax.core.protein import rate_matrix_lg
                _, pi_lg = rate_matrix_lg()
                pis[idx] = np.array(pi_lg)

    v = v / v.sum()
    kappas = lam / np.maximum(mu, 1e-10)

    # Singlet HMM construction (matching maraschino.py singlet section)
    z0 = np.sum(v * (1 - kappas))
    null_closure = 1.0 / max(1 - kappa0 * z0, 1e-30)
    kappa0_eff = kappa0 * null_closure
    v_nonempty = v * kappas  # NOT divided by (1-z0)
    end_factor = (1 - kappa0) * null_closure

    p_sing = r + (1 - r) * kappas
    T_sing = np.diag(p_sing) + np.outer(1 - p_sing, kappa0_eff * v_nonempty)
    sing_start = kappa0_eff * v_nonempty
    sing_end = (1 - p_sing) * end_factor

    return sing_start, T_sing, sing_end, pis


def build_pair_hmm(params, tau):
    """Build MixDom pair HMM transition matrix and substitution model.

    Returns (log_chi, st, sub_matrix, pi).
    """
    import jax.numpy as jnp
    from tkfmixdom.jax.models.mixdom import build_nested_trans, state_types as mixdom_state_types
    from tkfmixdom.jax.distill.maraschino import get_lg08, build_rate_matrix
    from tkfmixdom.jax.core.ctmc import transition_matrix
    from tkfmixdom.jax.dp.hmm import safe_log

    n_dom = len(params['dom_ins'])
    n_frag = params['frag_weights'].shape[1] if params['frag_weights'].ndim > 1 else 1

    # Build transition matrix chi (linear space)
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

    # State types
    st = mixdom_state_types(n_dom, n_frag)

    # Substitution model: LG08 P(t)
    S_lg, pi_lg = get_lg08()
    Q = build_rate_matrix(S_lg, pi_lg)
    sub_matrix = transition_matrix(Q, tau)

    # Use per-domain pi if available, otherwise LG08
    if 'dom_pis' in params:
        # Average pi across domains weighted by domain weights
        pi = jnp.zeros(20)
        for d in range(n_dom):
            pi = pi + params['dom_weights'][d] * jnp.array(params['dom_pis'][d])
        pi = pi / pi.sum()
    else:
        pi = pi_lg

    return log_chi, st, sub_matrix, pi


def score_assay_pair(assay_csv, params, tau, wt_seq=None):
    """Score all variants using pair HMM forward algorithm.

    Returns (predictions, dms_scores, n_scored).
    """
    import jax.numpy as jnp
    from tkfmixdom.jax.dp.hmm import forward_2d

    log_chi, st, sub_matrix, pi = build_pair_hmm(params, tau)

    # Read assay
    with open(assay_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

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
        print(f"  WARNING: no wildtype sequence found, skipping")
        return [], [], 0

    wt_int = seq_to_int(wt_seq)
    wt_int = wt_int[wt_int >= 0]

    # Use banded 2D Forward for efficiency. DMS indels are typically ≤10
    # residue changes, so band_width=20 is exact for all practical cases.
    # This reduces O(L²) to O(L×B) and avoids JIT OOM on large models.
    from tkfmixdom.jax.dp.hmm import forward_2d_banded

    mut_data = []  # (mut_int, dms_score)
    for row in rows:
        mut_int = seq_to_int(row['mutated_sequence'])
        mut_int = mut_int[mut_int >= 0]
        if len(mut_int) > 0:
            mut_data.append((mut_int, float(row['DMS_score'])))

    if not mut_data:
        return [], [], 0

    Lx = len(wt_int)
    wt_jnp = jnp.array(wt_int)
    band_width = 20  # half-width; covers indels up to ±20 residues

    predictions = []
    dms_scores = []
    n_scored = 0
    for mut_int, dms_score in mut_data:
        Ly = len(mut_int)
        mut_jnp = jnp.array(mut_int)
        # Band center: diagonal (j = i * Ly/Lx), clamped to valid range
        band_center = jnp.round(jnp.arange(Lx + 1) * Ly / max(Lx, 1)).astype(jnp.int32)
        band_center = jnp.clip(band_center, 0, Ly)
        log_prob, _ = forward_2d_banded(
            log_chi, st, wt_jnp, mut_jnp, sub_matrix, pi,
            band_center, band_width)

        predictions.append(float(log_prob))
        dms_scores.append(dms_score)
        n_scored += 1

        if (n_scored % 50) == 0:
            print(f"    scored {n_scored}/{len(mut_data)} variants...", flush=True)

    return predictions, dms_scores, n_scored


def score_assay(assay_csv, params, wt_seq=None):
    """Score all variants in an assay CSV.

    Returns (predictions, dms_scores, mutant_seqs).
    """
    from tkfmixdom.jax.dp.singlet_forward import singlet_log_prob

    sing_start, sing_trans, sing_end, pis = build_singlet_hmm_direct(params)

    # Read assay
    with open(assay_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return [], [], []

    # Get wildtype from reference file or first row without mutations
    if wt_seq is None:
        # Infer from the reference CSV
        ref_csv = os.path.join(os.path.dirname(assay_csv), '..', 'DMS_indels.csv')
        assay_name = os.path.basename(assay_csv).replace('.csv', '')
        if os.path.exists(ref_csv):
            with open(ref_csv) as f:
                for row in csv.DictReader(f):
                    if row['DMS_id'] == assay_name:
                        wt_seq = row['target_seq']
                        break

    if wt_seq is None:
        print(f"  WARNING: no wildtype sequence found, skipping")
        return [], [], []

    wt_int = seq_to_int(wt_seq)
    # Filter out unknown amino acids
    valid = wt_int >= 0
    if not np.all(valid):
        wt_int = wt_int[valid]

    wt_ll = singlet_log_prob(wt_int, sing_start, sing_trans, sing_end, pis)

    predictions = []
    dms_scores = []
    n_scored = 0
    for row in rows:
        mut_seq = row['mutated_sequence']
        dms_score = float(row['DMS_score'])

        mut_int = seq_to_int(mut_seq)
        mut_int = mut_int[mut_int >= 0]

        if len(mut_int) == 0:
            continue

        mut_ll = singlet_log_prob(mut_int, sing_start, sing_trans, sing_end, pis)
        score = mut_ll - wt_ll
        predictions.append(score)
        dms_scores.append(dms_score)
        n_scored += 1

    return predictions, dms_scores, n_scored


def main():
    parser = argparse.ArgumentParser(description='Score ProteinGym indel variants')
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
    parser.add_argument('--pair', action='store_true',
                        help='Use pair HMM scoring: log P(wt, mut | t)')
    parser.add_argument('--tau', type=float, default=None,
                        help='Evolutionary time for pair mode (default: sweep 0.01,0.1,0.5,1.0)')
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

    # Get assay list
    if args.assays:
        assay_names = [a.strip() for a in args.assays.split(',')]
    else:
        assay_names = [f.replace('.csv', '') for f in
                       sorted(os.listdir(args.proteingym_dir)) if f.endswith('.csv')]

    # Load reference for wildtype sequences
    ref = {}
    if os.path.exists(args.ref_csv):
        with open(args.ref_csv) as f:
            for row in csv.DictReader(f):
                ref[row['DMS_id']] = row['target_seq']

    if args.pair:
        # Pair HMM mode
        tau_values = [args.tau] if args.tau else [0.01, 0.1, 0.5, 1.0]
        all_tau_results = {}

        for tau in tau_values:
            print(f"\n=== Pair HMM scoring, tau={tau} ===")
            results = []
            for assay_name in assay_names:
                csv_path = os.path.join(args.proteingym_dir, assay_name + '.csv')
                if not os.path.exists(csv_path):
                    print(f"  {assay_name}: not found, skipping")
                    continue

                wt_seq = ref.get(assay_name)
                print(f"  {assay_name}: scoring...", flush=True)
                t0 = time.time()
                preds, scores, n = score_assay_pair(csv_path, params, tau, wt_seq)
                elapsed = time.time() - t0

                if len(preds) >= 10:
                    rho, pval = stats.spearmanr(preds, scores)
                    print(f"  {assay_name}: rho={rho:.4f} (p={pval:.2e}), "
                          f"{n} variants, {elapsed:.1f}s")
                    results.append({
                        'assay': assay_name, 'rho': rho, 'pval': pval,
                        'n_variants': n, 'time': elapsed, 'tau': tau})
                else:
                    print(f"  {assay_name}: too few variants ({n}), skipping")

            if results:
                rhos = [r['rho'] for r in results]
                mean_rho = np.mean(rhos)
                print(f"\ntau={tau}: Mean Spearman rho: {mean_rho:.4f} "
                      f"(+/-{np.std(rhos):.4f}, {len(results)} assays)")
                all_tau_results[tau] = (mean_rho, results)

        # Report best tau
        if all_tau_results:
            best_tau = max(all_tau_results, key=lambda t: all_tau_results[t][0])
            best_mean, best_results = all_tau_results[best_tau]
            print(f"\n=== Best tau={best_tau}: Mean Spearman rho={best_mean:.4f} ===")
            for r in best_results:
                print(f"  {r['assay']}: rho={r['rho']:.4f}")

            if args.out:
                # Save results for all tau values
                with open(args.out, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'assay', 'tau', 'rho', 'pval', 'n_variants', 'time'])
                    writer.writeheader()
                    for tau in sorted(all_tau_results):
                        _, res = all_tau_results[tau]
                        writer.writerows(res)
                print(f"Saved: {args.out}")
    else:
        # Singlet mode (original)
        results = []
        for assay_name in assay_names:
            csv_path = os.path.join(args.proteingym_dir, assay_name + '.csv')
            if not os.path.exists(csv_path):
                print(f"  {assay_name}: not found, skipping")
                continue

            wt_seq = ref.get(assay_name)
            t0 = time.time()
            preds, scores, n = score_assay(csv_path, params, wt_seq)
            elapsed = time.time() - t0

            if len(preds) >= 10:
                rho, pval = stats.spearmanr(preds, scores)
                print(f"  {assay_name}: rho={rho:.4f} (p={pval:.2e}), "
                      f"{n} variants, {elapsed:.1f}s")
                results.append({
                    'assay': assay_name, 'rho': rho, 'pval': pval,
                    'n_variants': n, 'time': elapsed})
            else:
                print(f"  {assay_name}: too few variants ({n}), skipping")

        if results:
            rhos = [r['rho'] for r in results]
            print(f"\nMean Spearman rho: {np.mean(rhos):.4f} "
                  f"(+/-{np.std(rhos):.4f}, {len(results)} assays)")

        if args.out and results:
            with open(args.out, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=['assay', 'rho', 'pval', 'n_variants', 'time'])
                writer.writeheader()
                writer.writerows(results)
            print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
