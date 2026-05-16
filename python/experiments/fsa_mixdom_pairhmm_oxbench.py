#!/usr/bin/env python3
"""FSA alignment benchmark on OXBench using canonical MixDom pair HMM.

Uses the CANONICAL pairwise_posteriors_mixdom from fsa_anneal.py with
per-domain S_exch emissions from the BW d3f2 checkpoint.

Defaults to GPU. Override JAX_PLATFORMS=cpu / CUDA_VISIBLE_DEVICES=
in the environment if you need to force CPU.

Usage:
    cd python && JAX_ENABLE_X64=1 uv run python experiments/fsa_mixdom_pairhmm_oxbench.py
    JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=1 uv run python experiments/fsa_mixdom_pairhmm_oxbench.py --n-families 50
"""

import os
os.environ.setdefault('JAX_ENABLE_X64', '1')

import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from tkfmixdom.jax.distill.maraschino import load_params
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


# ── Build fsa_params from load_params output ─────────────────────────
def make_fsa_params(params, n_dom):
    """Convert load_params output to the dict expected by pairwise_posteriors_mixdom.

    Per-domain S_exch (N, A, A) and pi (N, A) flow through directly so
    `_optimize_tau_mixdom` and `build_per_domain_sub_matrices` see the
    full per-domain rate scale and equilibrium chemistry.
    """
    fsa_params = {
        'main_ins': float(params['lam0']),
        'main_del': float(params['mu0']),
        'dom_ins': np.asarray(params['lam']),
        'dom_del': np.asarray(params['mu']),
        'dom_weights': np.asarray(params['v']),
        'frag_weights': np.asarray(params['frag_weights']),
        'ext_rates': np.asarray(params['r_frags']),  # (N, F) per-fragment extension
        'S_exch': np.asarray(params['S_exch']),       # (N, A, A) per-domain
        'pi': np.asarray(params['pi']),               # (N, A) per-domain equilibrium
    }

    n_frag = fsa_params['frag_weights'].shape[1] if fsa_params['frag_weights'].ndim > 1 else 1

    # MixDom2: forward per-class data through to the FB emission builder.
    # When ``pairwise_posteriors_mixdom`` sees class_pi / class_S_exch /
    # class_dist in the params dict, it switches to the per-class
    # emission table rather than the per-domain class-marginal one.
    for class_key in ('class_pi', 'class_S_exch', 'class_dist'):
        if class_key in params:
            fsa_params[class_key] = np.asarray(params[class_key])

    return fsa_params, n_frag


# ── Build MSA from pairwise posteriors ────────────────────────────────
def build_msa_from_posteriors(sequences, pair_posteriors, n_anneal=3,
                                seed=42):
    """Build MSA from pairwise match posteriors using sequence annealing.

    Single-seed entry point (backwards compatible). For multi-seed sweeps
    over the cached `pair_posteriors`, call `build_msa_from_posteriors_multi`.
    """
    names = list(sequences.keys())
    n_seqs = len(names)
    seq_lengths = [len(sequences[n]) for n in names]

    col_assignments, msa_length = sequence_annealing(
        n_seqs, seq_lengths, pair_posteriors,
        n_iterations=n_anneal, verbose=False, seed=seed)

    # Convert col_assignments to aligned dict
    n_cols = max(max(ca) for ca in col_assignments if len(ca) > 0) + 1
    msa_dict = {}
    for si, name in enumerate(names):
        row = np.full(n_cols, -1, dtype=np.int32)
        seq = sequences[name]
        for k in range(len(seq)):
            row[col_assignments[si][k]] = seq[k]
        msa_dict[name] = row

    return msa_dict, n_cols


def build_msa_from_posteriors_multi(sequences, pair_posteriors, seeds,
                                      n_anneal=3):
    """Build N MSAs from cached pairwise match posteriors, one per seed.

    The pairwise-posterior step is the expensive part; the sequence-
    annealing refinement is cheap and depends on `seed` through the
    sweep-order permutation. Run N seeds in a Python loop and return all
    MSAs; the caller can score each against a reference.

    Args:
        sequences: dict {name: int array}.
        pair_posteriors: precomputed dict {(i, j): (Li, Lj) array}.
        seeds: iterable of int seeds.
        n_anneal: refinement sweeps per seed.

    Returns:
        list of (seed, msa_dict, msa_length) tuples in seed order.
    """
    names = list(sequences.keys())
    n_seqs = len(names)
    seq_lengths = [len(sequences[n]) for n in names]
    runs = []
    for seed in seeds:
        col_assignments, msa_length = sequence_annealing(
            n_seqs, seq_lengths, pair_posteriors,
            n_iterations=n_anneal, verbose=False, seed=int(seed))
        n_cols = max(max(ca) for ca in col_assignments if len(ca) > 0) + 1
        msa_dict = {}
        for si, name in enumerate(names):
            row = np.full(n_cols, -1, dtype=np.int32)
            seq = sequences[name]
            for k in range(len(seq)):
                row[col_assignments[si][k]] = seq[k]
            msa_dict[name] = row
        runs.append((int(seed), msa_dict, n_cols))
    return runs


def msa_to_aligned_strings(msa_dict):
    """Convert integer MSA dict to aligned string dict."""
    result = {}
    for name, row in msa_dict.items():
        chars = []
        for c in row:
            if 0 <= c < 20:
                chars.append(AA_CHARS[c])
            elif c == 20:
                chars.append('X')  # wildcard residue, not a gap
            else:
                chars.append('-')
        result[name] = ''.join(chars)
    return result


# ── OXBench family processing ────────────────────────────────────────
def process_family(family_name, in_dir, ref_dir, fsa_params, n_dom, n_frag,
                   pair_selection='full'):
    """Process one OXBench family: align and score."""
    in_path = os.path.join(in_dir, family_name)
    ref_path = os.path.join(ref_dir, family_name)

    if not os.path.exists(in_path) or not os.path.exists(ref_path):
        return None

    # Parse sequences
    raw_seqs = parse_fasta(in_path)
    if len(raw_seqs) < 2:
        return None

    # Encode to integers
    int_seqs = {}
    str_seqs = {}
    for name, seq in raw_seqs.items():
        enc = encode_seq(seq)
        if len(enc) == 0:
            continue
        int_seqs[name] = enc
        str_seqs[name] = seq

    if len(int_seqs) < 2:
        return None

    names = list(int_seqs.keys())
    n_seqs = len(names)

    # Select pairs
    if pair_selection == 'full' or n_seqs <= 20:
        pairs = select_pairs_full(n_seqs)
    else:
        pairs = select_pairs_erdos_renyi(n_seqs)

    # Compute pairwise posteriors using canonical function
    t0 = time.time()
    pair_posteriors = {}
    pair_taus = {}
    for idx, (i, j) in enumerate(pairs):
        x = jnp.asarray(int_seqs[names[i]])
        y = jnp.asarray(int_seqs[names[j]])
        mp, tau, lp = pairwise_posteriors_mixdom(
            x, y, fsa_params, n_dom, n_frag)
        pair_posteriors[(i, j)] = mp
        pair_taus[(i, j)] = tau

    # Build MSA
    msa_dict, msa_length = build_msa_from_posteriors(int_seqs, pair_posteriors)
    aligned = msa_to_aligned_strings(msa_dict)
    t_elapsed = time.time() - t0

    # Parse reference and score
    ref_aln = parse_fasta(ref_path)
    sp, tc = sp_tc_score(aligned, ref_aln)

    return {
        'family': family_name,
        'n_seqs': int(n_seqs),
        'n_pairs': int(len(pairs)),
        'msa_length': int(msa_length),
        'sp': float(sp),
        'tc': float(tc),
        'time': float(t_elapsed),
    }


# ── MAFFT for comparison ─────────────────────────────────────────────
def process_family_mafft(family_name, in_dir, ref_dir):
    """Run MAFFT on one OXBench family and score."""
    import subprocess
    import tempfile

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
        sp, tc = sp_tc_score(test_aln, ref_aln)
        return {'sp': sp, 'tc': tc, 'time': t_elapsed}
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='FSA alignment benchmark on OXBench with canonical MixDom pair HMM')
    parser.add_argument('--n-families', type=int, default=50,
                        help='Number of families to process (default: 50)')
    parser.add_argument('--no-mafft', action='store_true',
                        help='Skip MAFFT comparison')
    parser.add_argument('--out', type=str,
                        default='experiments/fsa_oxbench_canonical.json',
                        help='Output JSON path')
    parser.add_argument('--pair-selection', type=str, default='full',
                        choices=['full', 'erdos_renyi'])
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model .npz file (default: BW d3f2)')
    parser.add_argument('--benchmark-dir', type=str, default=None,
                        help='Override benchmark directory (default: OXBench)')
    args = parser.parse_args()

    run_mafft = not args.no_mafft

    # Paths
    if args.benchmark_dir:
        in_dir = os.path.join(args.benchmark_dir, 'in')
        ref_dir = os.path.join(args.benchmark_dir, 'ref')
    else:
        oxbench_dir = Path("~/bio-datasets/data/oxbench/ox").expanduser()
        in_dir = str(oxbench_dir / "in")
        ref_dir = str(oxbench_dir / "ref")

    if args.model_path:
        params_path = args.model_path
    else:
        params_path = str(Path(__file__).parent.parent / "params" / "best" / "bw_d3f2_fullseed_15iter.npz")

    # Load params via canonical load_params
    print(f"Loading params from {params_path}")
    params, n_dom, n_cls = load_params(params_path)
    fsa_params, n_frag = make_fsa_params(params, n_dom)
    print(f"  n_dom={n_dom}, n_frag={n_frag}")
    print(f"  dom_weights={fsa_params['dom_weights']}")
    print(f"  main_ins={fsa_params['main_ins']:.6f}, main_del={fsa_params['main_del']:.6f}")

    # Get family list sorted by number of sequences (ascending)
    families = sorted(os.listdir(in_dir))
    families = [f for f in families if os.path.exists(os.path.join(ref_dir, f))]

    nseqs_info = {}
    info_dir = os.path.join(os.path.dirname(in_dir), "info")
    nseqs_file = os.path.join(info_dir, "nseqs.txt")
    if os.path.exists(nseqs_file):
        with open(nseqs_file) as fh:
            for line in fh:
                parts = line.strip().split(';')
                if len(parts) == 2:
                    nseqs_info[parts[0]] = int(parts[1])

    families.sort(key=lambda f: nseqs_info.get(f, 999))
    families = families[:args.n_families]

    print(f"\nProcessing {len(families)} OXBench families")
    print(f"  pair_selection={args.pair_selection}")
    print()

    results = []
    mafft_results = []

    for fi, family in enumerate(families):
        n_seq = nseqs_info.get(family, '?')
        print(f"[{fi+1}/{len(families)}] {family} (n={n_seq})", end='', flush=True)

        # MixDom FSA
        try:
            result = process_family(
                family, in_dir, ref_dir, fsa_params, n_dom, n_frag,
                pair_selection=args.pair_selection)
            if result is not None:
                results.append(result)
                print(f"  SP={result['sp']:.3f} TC={result['tc']:.3f} "
                      f"t={result['time']:.1f}s", end='')
            else:
                print(f"  SKIPPED", end='')
        except Exception as e:
            print(f"  ERROR: {e}", end='')
            import traceback
            traceback.print_exc()

        # MAFFT
        if run_mafft:
            mafft_res = process_family_mafft(family, in_dir, ref_dir)
            if mafft_res is not None:
                mafft_results.append(mafft_res)
                print(f"  | MAFFT SP={mafft_res['sp']:.3f} TC={mafft_res['tc']:.3f}", end='')

        print()

    # Summary
    print("\n" + "="*60)
    if results:
        sps = [r['sp'] for r in results]
        tcs = [r['tc'] for r in results]
        times = [r['time'] for r in results]
        print(f"MixDom FSA (n={len(results)}):")
        print(f"  SP: mean={np.mean(sps):.3f}, median={np.median(sps):.3f}")
        print(f"  TC: mean={np.mean(tcs):.3f}, median={np.median(tcs):.3f}")
        print(f"  Time: mean={np.mean(times):.1f}s, total={np.sum(times):.0f}s")

    if mafft_results:
        sps_m = [r['sp'] for r in mafft_results]
        tcs_m = [r['tc'] for r in mafft_results]
        print(f"\nMAFFT (n={len(mafft_results)}):")
        print(f"  SP: mean={np.mean(sps_m):.3f}, median={np.median(sps_m):.3f}")
        print(f"  TC: mean={np.mean(tcs_m):.3f}, median={np.median(tcs_m):.3f}")

    if results and mafft_results and len(results) == len(mafft_results):
        # Per-family comparison
        wins_sp = sum(1 for r, m in zip(results, mafft_results) if r['sp'] > m['sp'])
        wins_tc = sum(1 for r, m in zip(results, mafft_results) if r['tc'] > m['tc'])
        print(f"\nMixDom wins: SP {wins_sp}/{len(results)}, TC {wins_tc}/{len(results)}")

    # Save results
    out_path = args.out
    output = {
        'benchmark': 'OXBench',
        'model': 'MixDom pair HMM (canonical pairwise_posteriors_mixdom)',
        'params_file': os.path.basename(params_path),
        'n_dom': n_dom,
        'n_frag': n_frag,
        'pair_selection': args.pair_selection,
        'n_families': len(results),
        'mixdom_results': results,
        'mixdom_summary': {
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

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
