#!/usr/bin/env python3
"""MixDom profile scoring for ProteinGym DMS indel benchmark.

Fine-tunes MixDom parameters on a per-protein MSA, analogous to HMMER's
profile construction, then scores variants using the fine-tuned model.

Pipeline for each protein:
  1. Load the protein's MSA (a2m format from ProteinGym)
  2. Select cherry pairs from the MSA
  3. Run constrained E-step (1D FB) on cherry pairs with current MixDom params
  4. M-step to update domain weights, indel rates, extension rates
  5. Optionally iterate (2-3 EM iterations)
  6. Score each variant using the fine-tuned pair HMM

Key design choices:
  - Freeze exchangeability matrices (S_exch), update only rates + weights
  - Equilibrium frequencies (pi) can optionally be estimated from MSA columns
  - The evolutionary time t is estimated from the MSA pairwise distances
  - Domain weights and extension rates adapt to the protein's domain structure
  - A few EM iterations suffice (the generic params are a good starting point)

Usage:
    # Download MSAs first (one-time):
    # curl -o DMS_msa_files.zip https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_msa_files.zip
    # unzip DMS_msa_files.zip -d ~/bio-datasets/data/proteingym/

    # Profile scoring (all assays):
    python experiments/proteingym_profile.py \
        --params params/best/bw_d3f2_fullseed_15iter.npz \
        --msa-dir ~/bio-datasets/data/proteingym/DMS_msa_files/ \
        --out experiments/proteingym_profile_results.csv

    # Test on a few assays:
    python experiments/proteingym_profile.py \
        --params params/best/bw_d3f2_fullseed_15iter.npz \
        --msa-dir ~/bio-datasets/data/proteingym/DMS_msa_files/ \
        --assays TCRG1_MOUSE_Tsuboyama_2023_1E0L_indels,SDA_BACSU_Tsuboyama_2023_1PV0_indels

    # Skip profile (just multi-t with MSA-estimated t):
    python experiments/proteingym_profile.py \
        --params params/best/bw_d3f2_fullseed_15iter.npz \
        --msa-dir ~/bio-datasets/data/proteingym/DMS_msa_files/ \
        --no-finetune --out experiments/proteingym_msa_t_results.csv

    # Estimate per-domain pi from MSA columns:
    python experiments/proteingym_profile.py \
        --params params/best/bw_d3f2_fullseed_15iter.npz \
        --msa-dir ~/bio-datasets/data/proteingym/DMS_msa_files/ \
        --estimate-pi --out experiments/proteingym_profile_pi_results.csv
"""

import argparse
import csv
import os
import sys
import time
import copy

# Enable float64 in JAX BEFORE any jax import / op (audit ledger #8).
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy import stats as sp_stats

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA = len(AA_ORDER)
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}


def seq_to_int(seq):
    """Convert amino acid string to integer array. Unknown = -1."""
    return np.array([AA_TO_IDX.get(c, -1) for c in seq], dtype=int)


# ============================================================
# A2M parsing
# ============================================================

def parse_a2m(filepath):
    """Parse a2m (aligned FASTA) format.

    A2M format:
    - Uppercase: match columns (aligned)
    - Lowercase: insert columns (unaligned, ignored for profile)
    - '-': deletion in match column
    - '.': gap in insert column

    Returns:
        names: list of sequence names
        aligned_seqs: list of strings (match columns only, with gaps as '-')
    """
    import gzip
    opener = gzip.open if filepath.endswith('.gz') else open
    names = []
    seqs = []
    current_name = None
    current_seq = []

    with opener(filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_name is not None:
                    seqs.append(''.join(current_seq))
                current_name = line[1:].split()[0]
                names.append(current_name)
                current_seq = []
            else:
                # Keep only match columns (uppercase + '-')
                match_chars = [c if c.isupper() or c == '-' else ''
                               for c in line]
                current_seq.append(''.join(match_chars))

    if current_name is not None:
        seqs.append(''.join(current_seq))

    return names, seqs


def a2m_to_int_arrays(aligned_seq):
    """Convert a2m aligned sequence to integer array (-1 for gaps)."""
    return np.array([AA_TO_IDX.get(c, -1) for c in aligned_seq], dtype=int)


def extract_pairwise_alignment(seq1_aligned, seq2_aligned):
    """Extract pairwise alignment from two MSA-aligned sequences.

    Removes columns where both sequences are gaps.

    Returns:
        aligned_i: int array with -1 for gaps
        aligned_j: int array with -1 for gaps
    """
    arr1 = a2m_to_int_arrays(seq1_aligned)
    arr2 = a2m_to_int_arrays(seq2_aligned)

    # Keep columns where at least one sequence has a character
    mask = (arr1 >= 0) | (arr2 >= 0)
    return arr1[mask], arr2[mask]


# ============================================================
# Cherry selection from MSA
# ============================================================

def p_distance(seq1, seq2):
    """Compute p-distance between two aligned sequences."""
    matches = mismatches = 0
    for a, b in zip(seq1, seq2):
        if a in AA_TO_IDX and b in AA_TO_IDX:
            if a == b:
                matches += 1
            else:
                mismatches += 1
    total = matches + mismatches
    return mismatches / total if total > 0 else 1.0


def pdist_to_evo_time(pdist):
    """Convert p-distance to evolutionary time (Jukes-Cantor-like)."""
    if pdist >= 0.95:
        return 5.0
    corrected = 1.0 - pdist * (AA / (AA - 1.0))
    if corrected <= 0.01:
        return 5.0
    return -np.log(corrected)


def select_cherry_pairs(names, aligned_seqs, max_pairs=50, min_pdist=0.05,
                         max_pdist=0.8):
    """Select diverse cherry pairs from MSA for profile training.

    Strategy: greedy nearest-neighbor pairing with diversity filter.
    Avoids near-identical pairs (pdist < min_pdist) and very divergent
    pairs (pdist > max_pdist).

    Returns:
        pairs: list of (idx1, idx2, t_est) tuples
    """
    n = len(names)
    if n < 2:
        return []

    # Subsample if too many sequences
    if n > 500:
        rng = np.random.RandomState(42)
        indices = rng.choice(n, 500, replace=False)
        indices = sorted(indices)
    else:
        indices = list(range(n))

    # Compute pairwise distances for subsampled sequences
    n_sub = len(indices)
    dists = np.ones((n_sub, n_sub))
    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            pd = p_distance(aligned_seqs[indices[i]], aligned_seqs[indices[j]])
            dists[i, j] = pd
            dists[j, i] = pd

    # Greedy nearest-neighbor cherry pairing
    used = set()
    pairs = []
    # Sort all pairs by distance
    pair_dists = []
    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            d = dists[i, j]
            if min_pdist <= d <= max_pdist:
                pair_dists.append((d, i, j))
    pair_dists.sort()

    for d, i, j in pair_dists:
        if i in used or j in used:
            continue
        t_est = pdist_to_evo_time(d)
        pairs.append((indices[i], indices[j], t_est))
        used.add(i)
        used.add(j)
        if len(pairs) >= max_pairs:
            break

    return pairs


# ============================================================
# MSA-based equilibrium frequency estimation
# ============================================================

def estimate_column_frequencies(aligned_seqs, pseudocount=1.0):
    """Estimate per-column amino acid frequencies from MSA.

    Returns:
        freqs: (L, 20) array of per-column frequencies
        global_freqs: (20,) global frequencies across all columns
    """
    if not aligned_seqs:
        return None, None

    L = len(aligned_seqs[0])
    counts = np.zeros((L, AA))

    for seq in aligned_seqs:
        for j, c in enumerate(seq):
            if j >= L:
                break
            idx = AA_TO_IDX.get(c, -1)
            if idx >= 0:
                counts[j, idx] += 1.0

    # Add pseudocounts
    counts += pseudocount

    # Normalize per column
    freqs = counts / counts.sum(axis=1, keepdims=True)

    # Global frequencies
    global_counts = counts.sum(axis=0)
    global_freqs = global_counts / global_counts.sum()

    return freqs, global_freqs


# ============================================================
# Profile fine-tuning (EM on MSA cherry pairs)
# ============================================================

def finetune_mixdom_on_msa(params, cherry_pairs, aligned_seqs, t_rep,
                           n_iter=3, n_dom=None, n_frag=None,
                           freeze_rates=False, verbose=True):
    """Fine-tune MixDom parameters on cherry pairs from a protein's MSA.

    This is a lightweight version of train_pfam's EM loop, optimized for
    per-protein profile construction.

    Args:
        params: initial MixDom params dict (from generic training)
        cherry_pairs: list of (idx1, idx2, t_est) from select_cherry_pairs
        aligned_seqs: list of aligned sequences from a2m
        t_rep: representative evolutionary time
        n_iter: number of EM iterations (1-3 typical)
        n_dom: number of domains (inferred from params if None)
        n_frag: number of fragments per domain
        freeze_rates: if True, only update weights/extension, not rates
        verbose: print progress

    Returns:
        updated params dict
    """
    import jax.numpy as jnp
    from tkfmixdom.jax.models.mixdom import (
        build_nested_trans, n_states as mixdom_n_states,
        state_types as mixdom_state_types)
    from tkfmixdom.jax.train.constrained import (
        mixdom_constrained_e_step, prepare_aligned_pairs)
    from tkfmixdom.jax.models.exact_suffstats import exact_suffstats
    from tkfmixdom.jax.core.bdi import bdi_stats_from_counts, m_step_indel_quadratic
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    from tkfmixdom.jax.core.ctmc import transition_matrix
    from tkfmixdom.jax.simulate.msa import alignment_to_states
    from tkfmixdom.jax.dp.hmm import safe_log

    if n_dom is None:
        n_dom = len(params['dom_ins'])
    if n_frag is None:
        n_frag = (params['frag_weights'].shape[1]
                  if params['frag_weights'].ndim > 1 else 1)

    N = mixdom_n_states(n_dom, n_frag)
    st = mixdom_state_types(n_dom, n_frag)
    Q_lg, pi_lg = rate_matrix_lg()

    # Build aligned pairs from cherry selection
    aligned_pairs = []
    for idx1, idx2, t_est in cherry_pairs:
        aln_i, aln_j = extract_pairwise_alignment(
            aligned_seqs[idx1], aligned_seqs[idx2])
        if len(aln_i) > 0:
            aligned_pairs.append((str(idx1), str(idx2), aln_i, aln_j))

    if not aligned_pairs:
        if verbose:
            print("    No valid aligned pairs for profile training")
        return params

    params = copy.deepcopy(params)
    t = t_rep

    # Pseudocount parameters (light regularization toward prior)
    ins_prior_alpha = 1.5
    del_prior_alpha = 1.5
    prior_beta = 0.5
    dom_dirichlet = 2.0
    frag_dirichlet = 2.0
    ext_alpha = 2.0
    ext_beta = 2.0

    for it in range(n_iter):
        # Build transition matrix
        chi, _ = build_nested_trans(
            params['main_ins'], params['main_del'], t,
            jnp.array(params['dom_ins']), jnp.array(params['dom_del']),
            jnp.array(params['dom_weights']),
            jnp.array(params['frag_weights']),
            jnp.array(params['ext_rates']))
        log_chi = safe_log(chi)

        # Build substitution model
        sub_matrix = transition_matrix(Q_lg, t)

        # E-step: constrained 1D FB
        total_ll, agg_n_chi, all_match_info = mixdom_constrained_e_step(
            aligned_pairs, log_chi, st, sub_matrix, pi_lg, N)

        if verbose:
            print(f"    EM iter {it+1}/{n_iter}: LL={total_ll:.1f} "
                  f"({len(aligned_pairs)} pairs)")

        if float(np.sum(np.asarray(agg_n_chi))) < 1.0:
            if verbose:
                print("    Insufficient counts, stopping EM")
            break

        # Extract exact sufficient statistics
        ss = exact_suffstats(
            agg_n_chi,
            params['main_ins'], params['main_del'], t,
            params['dom_ins'], params['dom_del'],
            params['dom_weights'], params['frag_weights'],
            params['ext_rates'])

        # M-step: update parameters
        if not freeze_rates:
            # Top-level indel rates
            top_n = jnp.array(ss['top_5x5'])
            S_idx, M_idx, I_idx, D_idx, E_idx = 0, 1, 2, 3, 4
            top_L = float(jnp.sum(top_n[:, M_idx]) + jnp.sum(top_n[:, D_idx]))
            top_M = float(jnp.sum(top_n[:, E_idx]))
            top_T = t * top_M

            E_B, E_D, E_S = bdi_stats_from_counts(
                top_n, params['main_ins'], params['main_del'], t, T=top_T)

            main_ins_new, main_del_new = m_step_indel_quadratic(
                float(E_B), float(E_D), float(E_S),
                L=top_L, M=top_M, T=top_T,
                prior_alpha_lam=ins_prior_alpha,
                prior_alpha_mu=del_prior_alpha,
                prior_beta=prior_beta)
            params['main_ins'] = float(main_ins_new) \
                if np.isfinite(main_ins_new) else float(params['main_ins'])
            params['main_del'] = float(main_del_new) \
                if np.isfinite(main_del_new) else float(params['main_del'])

            # Per-domain indel rates
            for d in range(n_dom):
                dom_n = jnp.array(ss['dom_M_5x5'][d])
                nk_ID = ss['dom_kappa'][d]
                n1k_ID = ss['dom_1mkappa'][d]
                if float(dom_n.sum()) < 0.01 and nk_ID < 0.01:
                    continue
                n_entries_M = float(jnp.sum(dom_n[0, :]))  # S row
                T_d = t * (n_entries_M + n1k_ID)
                eb, ed, es = bdi_stats_from_counts(
                    dom_n, params['dom_ins'][d], params['dom_del'][d], t, T=T_d)
                nk_M = float(jnp.sum(dom_n[:, M_idx]) + jnp.sum(dom_n[:, D_idx]))
                n1k_M = float(jnp.sum(dom_n[:, E_idx]))
                ni, nd = m_step_indel_quadratic(
                    float(eb), float(ed), float(es),
                    L=nk_M + nk_ID, M=n1k_M + n1k_ID, T=T_d,
                    prior_alpha_lam=ins_prior_alpha,
                    prior_alpha_mu=del_prior_alpha,
                    prior_beta=prior_beta)
                ni = float(ni) if np.isfinite(ni) else float(params['dom_ins'][d])
                nd = float(nd) if np.isfinite(nd) else float(params['dom_del'][d])
                params['dom_ins'][d] = ni
                params['dom_del'][d] = nd

        # Domain weights
        dom_w_counts = np.array(ss['dom_w'])
        dom_w_post = np.maximum(dom_w_counts + dom_dirichlet - 1, 0)
        dom_w_total = dom_w_post.sum()
        if dom_w_total > 1e-10:
            params['dom_weights'] = dom_w_post / dom_w_total
        else:
            params['dom_weights'] = np.ones(n_dom) / n_dom

        # Fragment weights
        for d in range(n_dom):
            fw_post = np.maximum(np.array(ss['frag_w'][d]) + frag_dirichlet - 1, 0)
            fw_total = fw_post.sum()
            if fw_total > 1e-10:
                params['frag_weights'][d] = fw_post / fw_total
            else:
                params['frag_weights'][d] = np.ones(n_frag) / n_frag

        # Fragment extension rates
        for d in range(n_dom):
            for f in range(n_frag):
                a = ss['ext'][d, f] + ext_alpha - 1
                b = ss['term'][d, f] + ext_beta - 1
                total = a + b
                if total > 1e-10:
                    a_pos = max(a, 0.0)
                    b_pos = max(b, 0.0)
                    new_ext = a_pos / (a_pos + b_pos)
                    # Termination prob must remain > 0 (else degenerate
                    # geometric extension distribution).
                    assert b_pos > 1e-12 * total, (
                        f"ext_rates M-step at (d={d}, f={f}): termination "
                        f"posterior {b_pos:.3e} ≈ 0 (ext_beta={ext_beta}); "
                        f"set ext_beta > 1 or check input data.")
                    params['ext_rates'][d, f] = new_ext

    return params


# ============================================================
# Scoring
# ============================================================

def build_pair_hmm(params, tau):
    """Build MixDom pair HMM."""
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


def build_pair_hmm_custom_pi(params, tau, pi_custom):
    """Build MixDom pair HMM with custom equilibrium frequencies.

    Uses the custom pi for emissions but keeps the transition matrix
    from the standard params.
    """
    import jax.numpy as jnp
    from tkfmixdom.jax.distill.maraschino import get_lg08, build_rate_matrix
    from tkfmixdom.jax.core.ctmc import transition_matrix

    log_chi, st, sub_matrix_std, pi_std = build_pair_hmm(params, tau)

    # Build substitution matrix with custom pi
    S_lg, _ = get_lg08()
    Q = build_rate_matrix(S_lg, jnp.array(pi_custom))
    sub_matrix = transition_matrix(Q, tau)

    return log_chi, st, sub_matrix, jnp.array(pi_custom)


def score_assay(assay_csv, log_chi, st, sub_matrix, pi, wt_seq=None):
    """Score all variants using pair HMM forward algorithm.

    Returns (predictions, dms_scores, n_scored).
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

        mut_jnp = jnp.array(mut_int)
        log_prob, _ = forward_2d(log_chi, st, wt_jnp, mut_jnp, sub_matrix, pi)
        predictions.append(float(log_prob))
        dms_scores.append(dms_score)
        n_scored += 1

        if n_scored % 100 == 0:
            print(f"      scored {n_scored}/{len(rows)} variants...", flush=True)

    return predictions, dms_scores, n_scored


def main():
    parser = argparse.ArgumentParser(
        description='MixDom profile scoring for ProteinGym indel variants')
    parser.add_argument('--params', required=True, help='MixDom params .npz')
    parser.add_argument('--msa-dir', type=str, required=True,
                        help='Directory containing ProteinGym MSA files (a2m)')
    parser.add_argument('--assays', type=str, default=None,
                        help='Comma-separated assay names (default: all)')
    parser.add_argument('--proteingym-dir', type=str,
                        default=os.path.expanduser(
                            '~/bio-datasets/data/proteingym/DMS_ProteinGym_indels/'))
    parser.add_argument('--ref-csv', type=str,
                        default=os.path.expanduser(
                            '~/bio-datasets/data/proteingym/DMS_indels.csv'))
    parser.add_argument('--out', type=str, default=None, help='Output CSV')
    parser.add_argument('--n-iter', type=int, default=3,
                        help='EM iterations for profile fine-tuning (default: 3)')
    parser.add_argument('--max-pairs', type=int, default=50,
                        help='Max cherry pairs for profile training (default: 50)')
    parser.add_argument('--tau', type=float, default=None,
                        help='Fixed t for scoring (default: use MSA-estimated t)')
    parser.add_argument('--no-finetune', action='store_true',
                        help='Skip profile fine-tuning (just use MSA-estimated t)')
    parser.add_argument('--estimate-pi', action='store_true',
                        help='Estimate equilibrium frequencies from MSA columns')
    parser.add_argument('--freeze-rates', action='store_true',
                        help='Freeze indel rates, only update weights/extension')
    parser.add_argument('--max-msa-seqs', type=int, default=1000,
                        help='Max MSA sequences to load (default: 1000)')
    parser.add_argument('--short-only', action='store_true',
                        help='Only score assays with seq_len <= 150')
    parser.add_argument('--compare-baseline', action='store_true',
                        help='Also score with generic params for comparison')
    args = parser.parse_args()

    # Load params
    print(f"Loading params: {args.params}")
    d = np.load(args.params, allow_pickle=True)
    generic_params = {
        'main_ins': float(d['main_ins']), 'main_del': float(d['main_del']),
        'dom_ins': np.array(d['dom_ins'], dtype=float),
        'dom_del': np.array(d['dom_del'], dtype=float),
        'dom_weights': np.array(d['dom_weights'], dtype=float),
        'frag_weights': np.array(d['frag_weights'], dtype=float),
        'ext_rates': np.array(d['ext_rates'], dtype=float),
    }
    if 'dom_pis' in d:
        generic_params['dom_pis'] = np.array(d['dom_pis'], dtype=float)

    n_dom = len(generic_params['dom_ins'])
    n_frag = (generic_params['frag_weights'].shape[1]
              if generic_params['frag_weights'].ndim > 1 else 1)
    print(f"  {n_dom} domains, {n_frag} fragments")

    # Get assay list
    if args.assays:
        assay_names = [a.strip() for a in args.assays.split(',')]
    else:
        assay_names = [f.replace('.csv', '') for f in
                       sorted(os.listdir(args.proteingym_dir)) if f.endswith('.csv')]

    # Load reference
    ref = {}
    ref_info = {}
    if os.path.exists(args.ref_csv):
        with open(args.ref_csv) as f:
            for row in csv.DictReader(f):
                ref[row['DMS_id']] = row['target_seq']
                ref_info[row['DMS_id']] = {
                    'msa_file': row.get('MSA_filename', ''),
                    'seq_len': int(row.get('seq_len', 0)),
                    'msa_num_seqs': int(row.get('MSA_num_seqs', 0)),
                }

    if args.short_only:
        assay_names = [a for a in assay_names
                       if ref_info.get(a, {}).get('seq_len', 9999) <= 150]
        print(f"Short proteins only (<=150 aa): {len(assay_names)} assays")

    # Score each assay
    results = []
    for assay_idx, assay_name in enumerate(assay_names):
        csv_path = os.path.join(args.proteingym_dir, assay_name + '.csv')
        if not os.path.exists(csv_path):
            print(f"\n  {assay_name}: DMS file not found, skipping")
            continue

        wt_seq = ref.get(assay_name)
        if not wt_seq:
            print(f"\n  {assay_name}: no wildtype sequence, skipping")
            continue

        info = ref_info.get(assay_name, {})
        seq_len = info.get('seq_len', len(wt_seq))
        msa_filename = info.get('msa_file', '')

        print(f"\n  [{assay_idx+1}/{len(assay_names)}] {assay_name} "
              f"(L={seq_len}, MSA={msa_filename})", flush=True)

        t0 = time.time()

        # Try to find MSA file
        msa_path = None
        if msa_filename:
            candidate = os.path.join(args.msa_dir, msa_filename)
            if os.path.exists(candidate):
                msa_path = candidate
            # Also try with .gz
            elif os.path.exists(candidate + '.gz'):
                msa_path = candidate + '.gz'

        # Determine scoring parameters
        profile_params = None
        t_est = args.tau or 0.1  # default
        pi_custom = None
        n_msa_seqs_used = 0
        n_cherry_pairs = 0

        if msa_path:
            print(f"    Loading MSA: {msa_path}", flush=True)
            names, aligned_seqs = parse_a2m(msa_path)

            # Subsample if too many sequences
            if len(names) > args.max_msa_seqs:
                rng = np.random.RandomState(42)
                keep = rng.choice(len(names), args.max_msa_seqs, replace=False)
                keep = sorted(keep)
                names = [names[i] for i in keep]
                aligned_seqs = [aligned_seqs[i] for i in keep]
            n_msa_seqs_used = len(names)
            print(f"    MSA: {n_msa_seqs_used} sequences, "
                  f"L_aln={len(aligned_seqs[0]) if aligned_seqs else 0}")

            # Select cherry pairs
            cherry_pairs = select_cherry_pairs(
                names, aligned_seqs, max_pairs=args.max_pairs)
            n_cherry_pairs = len(cherry_pairs)
            print(f"    Cherry pairs: {n_cherry_pairs}")

            # Estimate representative t from cherry pair distances
            if cherry_pairs:
                t_values = [t for _, _, t in cherry_pairs]
                t_est = float(np.median(t_values))
                print(f"    Estimated t: {t_est:.4f} (median of cherry pairs)")

            # Estimate equilibrium frequencies from MSA columns
            if args.estimate_pi and aligned_seqs:
                _, global_freqs = estimate_column_frequencies(aligned_seqs)
                if global_freqs is not None:
                    pi_custom = global_freqs
                    print(f"    Estimated pi from MSA columns")

            # Fine-tune MixDom on cherry pairs
            if not args.no_finetune and cherry_pairs:
                print(f"    Fine-tuning MixDom ({args.n_iter} EM iterations)...",
                      flush=True)
                profile_params = finetune_mixdom_on_msa(
                    generic_params, cherry_pairs, aligned_seqs, t_est,
                    n_iter=args.n_iter, n_dom=n_dom, n_frag=n_frag,
                    freeze_rates=args.freeze_rates, verbose=True)
                print(f"    Fine-tuning complete")
        else:
            print(f"    No MSA found, using generic params")

        # Use override tau if specified
        if args.tau:
            t_est = args.tau

        # Score with profile params (or generic if no MSA)
        scoring_params = profile_params if profile_params is not None else generic_params

        # Build pair HMM
        if pi_custom is not None:
            log_chi, st, sub_matrix, pi = build_pair_hmm_custom_pi(
                scoring_params, t_est, pi_custom)
        else:
            log_chi, st, sub_matrix, pi = build_pair_hmm(scoring_params, t_est)

        # Score variants
        print(f"    Scoring variants (t={t_est:.4f})...", flush=True)
        preds, scores, n_scored = score_assay(
            csv_path, log_chi, st, sub_matrix, pi, wt_seq)

        elapsed = time.time() - t0

        result = {
            'assay': assay_name,
            'seq_len': seq_len,
            'n_variants': n_scored,
            'n_msa_seqs': n_msa_seqs_used,
            'n_cherry_pairs': n_cherry_pairs,
            't_est': t_est,
            'has_profile': profile_params is not None,
            'has_custom_pi': pi_custom is not None,
            'time': elapsed,
        }

        if len(preds) >= 10:
            rho, pval = sp_stats.spearmanr(preds, scores)
            result['profile_rho'] = rho
            result['profile_pval'] = pval
            print(f"    Profile rho: {rho:.4f} (p={pval:.2e})")
        else:
            result['profile_rho'] = None
            result['profile_pval'] = None
            print(f"    Too few variants ({n_scored})")

        # Optionally compare with baseline (generic params, fixed t)
        if args.compare_baseline:
            for baseline_t in [0.1]:
                log_chi_b, st_b, sub_b, pi_b = build_pair_hmm(
                    generic_params, baseline_t)
                preds_b, scores_b, n_b = score_assay(
                    csv_path, log_chi_b, st_b, sub_b, pi_b, wt_seq)
                if len(preds_b) >= 10:
                    rho_b, _ = sp_stats.spearmanr(preds_b, scores_b)
                    result[f'baseline_t{baseline_t}_rho'] = rho_b
                    print(f"    Baseline (t={baseline_t}): rho={rho_b:.4f}")

        results.append(result)

    # Summary
    if results:
        print(f"\n{'='*70}")
        print(f"SUMMARY ({len(results)} assays)")
        print(f"{'='*70}")

        profile_rhos = [r['profile_rho'] for r in results
                        if r['profile_rho'] is not None]
        if profile_rhos:
            print(f"  Profile: mean rho = {np.mean(profile_rhos):.4f} "
                  f"+/- {np.std(profile_rhos):.4f}")

        if args.compare_baseline:
            base_rhos = [r.get('baseline_t0.1_rho') for r in results
                         if r.get('baseline_t0.1_rho') is not None]
            if base_rhos:
                print(f"  Baseline (t=0.1): mean rho = {np.mean(base_rhos):.4f} "
                      f"+/- {np.std(base_rhos):.4f}")

        # Per-assay table
        print(f"\n{'Assay':55s} {'Profile':>8s} {'t_est':>6s} {'MSA':>5s} "
              f"{'Pairs':>5s}")
        print('-' * 85)
        for r in sorted(results, key=lambda x: x.get('profile_rho') or -99,
                         reverse=True):
            rho_s = (f"{r['profile_rho']:+.4f}"
                     if r['profile_rho'] is not None else '   N/A')
            prof = '*' if r['has_profile'] else ' '
            print(f"{r['assay']:55s} {rho_s:>8s} {r['t_est']:6.3f} "
                  f"{r['n_msa_seqs']:5d} {r['n_cherry_pairs']:5d} {prof}")

    # Save results
    if args.out and results:
        fieldnames = ['assay', 'seq_len', 'n_variants', 'n_msa_seqs',
                      'n_cherry_pairs', 't_est', 'has_profile', 'has_custom_pi',
                      'profile_rho', 'profile_pval', 'time']
        if args.compare_baseline:
            fieldnames.append('baseline_t0.1_rho')
        with open(args.out, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
