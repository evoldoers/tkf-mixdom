#!/usr/bin/env python3
"""FSA alignment benchmark on BAliBASE using Annabel's MixDom params.

Loads Annabel's trained mixture-of-domains parameters (with site classes,
fragment classes, rate multipliers) and runs FSA sequence annealing on
BAliBASE, scoring against reference alignments.

Usage:
    cd python
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python experiments/alignment_benchmark_annabel.py \
        --params-dir /home/shared/mixdom_params/GTR_3dom_3frag_3site \
        --out experiments/annabel_balibase_gtr3.json

    # With GPU (faster for large models):
    JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 uv run python experiments/alignment_benchmark_annabel.py \
        --params-dir /home/shared/mixdom_params/GTR_10dom_10frag_10site \
        --out experiments/annabel_balibase_gtr10.json
"""

import os
import sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

# Must set JAX_PLATFORMS before importing JAX
if '--cpu' in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from tkfmixdom.jax.models.annabel_mixdom import (
    load_annabel_params,
    build_annabel_transition_matrix,
    build_pair_hmm_emissions,
    pairwise_posteriors_annabel,
    pairwise_posteriors_annabel_jax,
    pairwise_posteriors_annabel_jax_tauopt,
    make_fsa_params_annabel,
)
from tkfmixdom.jax.tree.fsa_anneal import (
    pairwise_posteriors_mixdom,
    select_pairs_full, select_pairs_erdos_renyi,
    sequence_annealing,
)
from tkfmixdom.util.msa_benchmark import parse_fasta, sp_tc_score


# ── Amino acid encoding ──────────────────────────────────────────────
AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"
AA_MAP = {c: i for i, c in enumerate(AA_CHARS)}


def encode_seq(seq_str):
    """Convert amino acid string to integer array, unknowns mapped to 20 (wildcard)."""
    return np.array([AA_MAP.get(c, 20) for c in seq_str.upper() if c not in '.-~'],
                    dtype=np.int32)


def build_msa_from_posteriors(sequences, pair_posteriors, n_anneal=3):
    """Build MSA from pairwise match posteriors using sequence annealing."""
    names = list(sequences.keys())
    n_seqs = len(names)
    seq_lengths = [len(sequences[n]) for n in names]

    col_assignments, msa_length = sequence_annealing(
        n_seqs, seq_lengths, pair_posteriors,
        n_iterations=n_anneal, verbose=False)

    n_cols = max(max(ca) for ca in col_assignments if len(ca) > 0) + 1
    msa_dict = {}
    for si, name in enumerate(names):
        row = np.full(n_cols, -1, dtype=np.int32)
        seq = sequences[name]
        for k in range(len(seq)):
            row[col_assignments[si][k]] = seq[k]
        msa_dict[name] = row

    return msa_dict, n_cols


def msa_to_aligned_strings(msa_dict):
    """Convert integer MSA dict to aligned string dict."""
    result = {}
    for name, row in msa_dict.items():
        chars = []
        for c in row:
            if 0 <= c < 20:
                chars.append(AA_CHARS[c])
            elif c == 20:
                chars.append('X')
            else:
                chars.append('-')
        result[name] = ''.join(chars)
    return result


def estimate_pairwise_time(x, y):
    """Quick pairwise distance estimate using identity fraction.

    Uses the Kimura-like formula: t = -log(max(frac_id, 0.05))
    where frac_id is the fraction of identical residues at shared positions.

    Returns estimated evolutionary time, clipped to [0.01, 5.0].
    """
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    min_len = min(len(x_np), len(y_np))
    if min_len == 0:
        return 1.0
    matches = np.sum(x_np[:min_len] == y_np[:min_len])
    frac_id = matches / min_len
    t = -np.log(max(frac_id, 0.05))
    return float(np.clip(t, 0.01, 5.0))


def process_family_annabel(family_name, in_dir, ref_dir, annabel_params,
                           fsa_params, n_dom, n_frag, use_site_mixture=True,
                           pair_selection='full', n_newton=5):
    """Process one BAliBASE family with Annabel's params."""
    in_path = os.path.join(in_dir, family_name)
    ref_path = os.path.join(ref_dir, family_name)

    if not os.path.exists(in_path) or not os.path.exists(ref_path):
        return None

    raw_seqs = parse_fasta(in_path)
    if len(raw_seqs) < 2:
        return None

    int_seqs = {}
    for name, seq in raw_seqs.items():
        enc = encode_seq(seq)
        if len(enc) == 0:
            continue
        int_seqs[name] = enc

    if len(int_seqs) < 2:
        return None

    names = list(int_seqs.keys())
    n_seqs = len(names)

    if pair_selection == 'full' or n_seqs <= 20:
        pairs = select_pairs_full(n_seqs)
    else:
        pairs = select_pairs_erdos_renyi(n_seqs)

    t0 = time.time()
    pair_posteriors = {}

    if use_site_mixture:
        # Bucketed vmap with OOM fallback — same infrastructure as d3f1
        from tkfmixdom.jax.models.annabel_mixdom import (
            precompute_annabel_nr_params, pairwise_posteriors_annabel_batched,
        )
        from tkfmixdom.jax.dp.hmm import _pad_to_bin
        # Import benchmark infrastructure
        sys.path.insert(0, str(Path(__file__).parent))
        from alignment_benchmark_full import (
            _bucket_pairs_by_padded_shape, _max_batch_size_for_shape,
            _iter_bucket_chunks, _run_chunk_with_oom_fallback,
        )

        nr_params = precompute_annabel_nr_params(annabel_params)
        n_states = 2 + 5 * n_dom * n_frag
        buckets = _bucket_pairs_by_padded_shape(pairs, int_seqs, names)

        def _call(chunk):
            xs = np.stack([entry[4] for entry in chunk])
            ys = np.stack([entry[5] for entry in chunk])
            real_Lxs = np.array([entry[2] for entry in chunk], dtype=np.int32)
            real_Lys = np.array([entry[3] for entry in chunk], dtype=np.int32)
            return pairwise_posteriors_annabel_batched(
                jnp.asarray(xs), jnp.asarray(ys),
                jnp.asarray(real_Lxs), jnp.asarray(real_Lys),
                annabel_params, nr_params, n_dom, n_frag,
                n_newton=n_newton)

        for (Lx_pad, Ly_pad), bucket in buckets.items():
            chunk_size = _max_batch_size_for_shape(Lx_pad, Ly_pad, n_states)
            for chunk in _iter_bucket_chunks(bucket, chunk_size):
                mps_b, _, _ = _run_chunk_with_oom_fallback(_call, chunk)
                for b, (idx, (i, j), Lx_real, Ly_real, _, _) in enumerate(chunk):
                    pair_posteriors[(i, j)] = mps_b[b, :Lx_real, :Ly_real]
    else:
        # Use standard pipeline with per-domain emissions
        for i_idx, j_idx in pairs:
            x = jnp.asarray(int_seqs[names[i_idx]])
            y = jnp.asarray(int_seqs[names[j_idx]])
            mp, tau, lp = pairwise_posteriors_mixdom(
                x, y, fsa_params, n_dom, n_frag)
            pair_posteriors[(i_idx, j_idx)] = mp

    msa_dict, msa_length = build_msa_from_posteriors(int_seqs, pair_posteriors)
    aligned = msa_to_aligned_strings(msa_dict)
    t_elapsed = time.time() - t0

    ref_aln = parse_fasta(ref_path)
    sp, tc = sp_tc_score(aligned, ref_aln, core_only=True)

    return {
        'family': family_name,
        'n_seqs': int(n_seqs),
        'n_pairs': int(len(pairs)),
        'msa_length': int(msa_length),
        'sp': float(sp),
        'tc': float(tc),
        'time': float(t_elapsed),
    }


def process_family_mafft(family_name, in_dir, ref_dir):
    """Run MAFFT on one BAliBASE family and score."""
    in_path = os.path.join(in_dir, family_name)
    ref_path = os.path.join(ref_dir, family_name)

    if not os.path.exists(in_path) or not os.path.exists(ref_path):
        return None

    with tempfile.NamedTemporaryFile(suffix='.fa', delete=False) as tmp:
        tmp_out = tmp.name

    try:
        t0 = time.time()
        cmd = f"mafft --auto --quiet {in_path} > {tmp_out}"
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
        t_elapsed = time.time() - t0
        if result.returncode != 0:
            return None
        test_aln = parse_fasta(tmp_out)
        ref_aln = parse_fasta(ref_path)
        sp, tc = sp_tc_score(test_aln, ref_aln, core_only=True)
        return {'sp': sp, 'tc': tc, 'time': t_elapsed}
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(
        description='FSA alignment benchmark on BAliBASE with Annabel MixDom params')
    parser.add_argument('--params-dir', type=str, required=True,
                        help='Path to Annabel model parameter directory')
    parser.add_argument('--n-families', type=int, default=0,
                        help='Number of families to process (0=all)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--no-site-mixture', action='store_true',
                        help='Use per-domain avg emissions instead of site mixture')
    parser.add_argument('--no-mafft', action='store_true',
                        help='Skip MAFFT comparison')
    parser.add_argument('--pair-selection', type=str, default='erdos_renyi',
                        choices=['full', 'erdos_renyi'])
    parser.add_argument('--max-seqlen', type=int, default=10000,
                        help='Skip families with sequences longer than this (default: 300)')
    parser.add_argument('--max-seqs', type=int, default=50,
                        help='Skip families with more sequences than this (default: 20)')
    parser.add_argument('--n-newton', type=int, default=5,
                        help='Newton-Raphson steps for tau optimization (0=skip NR, use tau_init)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU execution (for large models that OOM on GPU)')
    args = parser.parse_args()

    model_name = os.path.basename(args.params_dir.rstrip('/'))
    use_site_mixture = not args.no_site_mixture

    if args.out is None:
        args.out = f'experiments/annabel_balibase_{model_name}.json'

    # Load params
    print(f"Loading Annabel params from {args.params_dir}")
    annabel_params = load_annabel_params(args.params_dir)
    n_dom = annabel_params['n_dom']
    n_frag = annabel_params['n_frag']
    n_site = annabel_params['n_site']
    n_subrate = annabel_params['n_subrate']
    print(f"  Model: {model_name}")
    print(f"  n_dom={n_dom}, n_frag={n_frag}, n_site={n_site}, n_subrate={n_subrate}")
    print(f"  Total pair HMM states: {2 + 5*n_dom*n_frag}")
    print(f"  Top-level TKF91: lambda={annabel_params['top_lambda']:.5f}, "
          f"mu={annabel_params['top_mu']:.5f}")
    print(f"  Use site-class mixture: {use_site_mixture}")

    # Also build FSA params for per-domain mode or time estimation
    fsa_params, _, _ = make_fsa_params_annabel(annabel_params)

    # BAliBASE paths
    balibase_dir = Path("~/bio-datasets/data/balibase/bali3pdbm").expanduser()
    in_dir = str(balibase_dir / "in")
    ref_dir = str(balibase_dir / "ref")

    families = sorted(os.listdir(in_dir))
    families = [f for f in families if os.path.exists(os.path.join(ref_dir, f))]

    if args.n_families > 0:
        families = families[:args.n_families]

    print(f"\nProcessing {len(families)} BAliBASE families")
    print(f"  pair_selection={args.pair_selection}")
    print()

    results = []
    mafft_results = []

    run_mafft = not args.no_mafft

    hdr = f"{'Family':<12} {'N':>2} | {'Annabel SP':>10} {'TC':>6} | "
    if run_mafft:
        hdr += f"{'MAFFT SP':>8} {'TC':>6} | "
    print(hdr)
    print("-" * len(hdr))

    for fi, family in enumerate(families):
        raw_seqs = parse_fasta(os.path.join(in_dir, family))
        n_seq = len(raw_seqs)

        # Filter by sequence count and length
        max_len = max(len(s) for s in raw_seqs.values()) if raw_seqs else 0
        if n_seq > args.max_seqs or max_len > args.max_seqlen:
            print(f"[{fi+1:>3}/{len(families)}] {family:<12} {n_seq:>2} | "
                  f"SKIP (n={n_seq}, maxlen={max_len})")
            continue

        line = f"{family:<12} {n_seq:>2} | "

        try:
            res = process_family_annabel(
                family, in_dir, ref_dir, annabel_params,
                fsa_params, n_dom, n_frag,
                use_site_mixture=use_site_mixture,
                pair_selection=args.pair_selection,
                n_newton=args.n_newton)
            if res is not None:
                results.append(res)
                line += f"{res['sp']:>10.3f} {res['tc']:>6.3f} | "
            else:
                line += f"{'SKIP':>10} {'':>6} | "
        except Exception as e:
            line += f"{'ERR':>10} {'':>6} | "
            print(f"\n  ERROR on {family}: {e}")
            import traceback; traceback.print_exc()

        if run_mafft:
            mafft_res = process_family_mafft(family, in_dir, ref_dir)
            if mafft_res is not None:
                mafft_results.append({**mafft_res, 'family': family})
                line += f"{mafft_res['sp']:>8.3f} {mafft_res['tc']:>6.3f} | "
            else:
                line += f"{'ERR':>8} {'':>6} | "

        print(f"[{fi+1:>3}/{len(families)}] {line}")

    # Summary
    print("\n" + "=" * 70)
    for label, rlist in [("Annabel MixDom", results), ("MAFFT", mafft_results)]:
        if rlist:
            sps = [r['sp'] for r in rlist]
            tcs = [r['tc'] for r in rlist]
            print(f"{label} (n={len(rlist)}):")
            print(f"  SP: mean={np.mean(sps):.4f}, median={np.median(sps):.4f}")
            print(f"  TC: mean={np.mean(tcs):.4f}, median={np.median(tcs):.4f}")
            if 'time' in rlist[0]:
                times = [r['time'] for r in rlist]
                print(f"  Time: mean={np.mean(times):.1f}s, total={np.sum(times):.0f}s")

    # Head-to-head
    if results and mafft_results:
        mixdom_by_fam = {r['family']: r for r in results}
        mafft_by_fam = {r['family']: r for r in mafft_results}
        common = sorted(set(mixdom_by_fam) & set(mafft_by_fam))
        if common:
            wins_sp = sum(1 for f in common if mixdom_by_fam[f]['sp'] > mafft_by_fam[f]['sp'])
            wins_tc = sum(1 for f in common if mixdom_by_fam[f]['tc'] > mafft_by_fam[f]['tc'])
            print(f"\nAnnabel vs MAFFT (n={len(common)}): SP wins {wins_sp}, TC wins {wins_tc}")

    # Save
    output = {
        'benchmark': 'BAliBASE 3',
        'model': f'Annabel MixDom ({model_name})',
        'params_dir': args.params_dir,
        'n_dom': n_dom,
        'n_frag': n_frag,
        'n_site': n_site,
        'n_subrate': n_subrate,
        'use_site_mixture': use_site_mixture,
        'pair_selection': args.pair_selection,
        'n_families': len(results),
        'results': results,
        'summary': {
            'sp_mean': float(np.mean([r['sp'] for r in results])) if results else 0,
            'sp_median': float(np.median([r['sp'] for r in results])) if results else 0,
            'tc_mean': float(np.mean([r['tc'] for r in results])) if results else 0,
            'tc_median': float(np.median([r['tc'] for r in results])) if results else 0,
            'time_total': float(np.sum([r['time'] for r in results])) if results else 0,
        },
    }
    if mafft_results:
        output['mafft_results'] = mafft_results
        output['mafft_summary'] = {
            'sp_mean': float(np.mean([r['sp'] for r in mafft_results])),
            'sp_median': float(np.median([r['sp'] for r in mafft_results])),
            'tc_mean': float(np.mean([r['tc'] for r in mafft_results])),
            'tc_median': float(np.median([r['tc'] for r in mafft_results])),
        }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
