#!/usr/bin/env python3
"""FSA alignment benchmark on BAliBASE using canonical MixDom pair HMM.

Adapted from fsa_mixdom_pairhmm_oxbench.py — same code path, different dataset.

Defaults to GPU. Override JAX_PLATFORMS=cpu / CUDA_VISIBLE_DEVICES=
in the environment if you need to force CPU.

Usage:
    cd python && JAX_ENABLE_X64=1 uv run python experiments/fsa_mixdom_pairhmm_balibase.py
    JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=1 uv run python experiments/fsa_mixdom_pairhmm_balibase.py --n-families 50
"""

import os
os.environ.setdefault('JAX_ENABLE_X64', '1')

import json
import subprocess
import sys
import tempfile
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

# Import make_fsa_params and helpers from the OXBench script
from fsa_mixdom_pairhmm_oxbench import (
    make_fsa_params, encode_seq, build_msa_from_posteriors,
    msa_to_aligned_strings, AA_CHARS, AA_MAP,
)


# ── Family processing (reuses OXBench functions) ────────────────────
def process_family(family_name, in_dir, ref_dir, fsa_params, n_dom, n_frag,
                   pair_selection='full'):
    """Process one BAliBASE family: align and score."""
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


def process_family_muscle(family_name, in_dir, ref_dir):
    """Run MUSCLE5 on one BAliBASE family and score."""
    in_path = os.path.join(in_dir, family_name)
    ref_path = os.path.join(ref_dir, family_name)

    if not os.path.exists(in_path) or not os.path.exists(ref_path):
        return None

    with tempfile.NamedTemporaryFile(suffix='.fa', delete=False) as tmp:
        tmp_out = tmp.name

    try:
        t0 = time.time()
        cmd = [os.path.expanduser("~/bin/muscle"),
               "-align", in_path, "-output", tmp_out]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
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


# ── Main ──────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='FSA alignment benchmark on BAliBASE with MixDom, MAFFT, and MUSCLE')
    parser.add_argument('--n-families', type=int, default=0,
                        help='Number of families to process (0=all, default: all)')
    parser.add_argument('--no-mafft', action='store_true')
    parser.add_argument('--no-muscle', action='store_true')
    parser.add_argument('--no-mixdom', action='store_true',
                        help='Skip the MixDom/FSA pairwise-alignment step. '
                        'Useful when only MAFFT/MUSCLE numbers are needed '
                        'and we already have MixDom results from a prior '
                        'run; saves the slow per-family FSA computation.')
    parser.add_argument('--out', type=str,
                        default='experiments/balibase_mixdom_mafft_muscle.json')
    parser.add_argument('--pair-selection', type=str, default='full',
                        choices=['full', 'erdos_renyi'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override default params path (must be a '
                             'load_params-compatible .npz). Default: '
                             'params/best/bw_d3f2_fullseed_15iter.npz')
    parser.add_argument('--label', type=str, default=None,
                        help='Display label for this run (logged with results)')
    args = parser.parse_args()

    run_mafft = not args.no_mafft
    run_muscle = not args.no_muscle
    run_mixdom = not args.no_mixdom

    # Paths
    balibase_dir = Path("~/bio-datasets/data/balibase/bali3pdbm").expanduser()
    in_dir = str(balibase_dir / "in")
    ref_dir = str(balibase_dir / "ref")
    if args.checkpoint:
        params_path = args.checkpoint
    else:
        params_path = str(Path(__file__).parent.parent / "params" / "best" / "bw_d3f2_fullseed_15iter.npz")

    label = args.label or os.path.splitext(os.path.basename(params_path))[0]
    # Load params (skip when --no-mixdom: MAFFT/MUSCLE don't need them)
    if run_mixdom:
        print(f"Loading params from {params_path} (label={label})")
        params, n_dom, n_cls = load_params(params_path)
        fsa_params, n_frag = make_fsa_params(params, n_dom)
        print(f"  n_dom={n_dom}, n_frag={n_frag}")
        print(f"  dom_weights={fsa_params['dom_weights']}")
    else:
        params = fsa_params = None
        n_dom = n_frag = n_cls = 0
        print(f"Skipping MixDom (--no-mixdom); MAFFT/MUSCLE only")

    # Get family list
    families = sorted(os.listdir(in_dir))
    families = [f for f in families if os.path.exists(os.path.join(ref_dir, f))]

    if args.n_families > 0:
        families = families[:args.n_families]

    print(f"\nProcessing {len(families)} BAliBASE families")
    print(f"  pair_selection={args.pair_selection}")
    print(f"  MixDom={'yes' if run_mixdom else 'no'}, "
          f"MAFFT={'yes' if run_mafft else 'no'}, "
          f"MUSCLE={'yes' if run_muscle else 'no'}")
    print()

    results = []
    mafft_results_list = []
    muscle_results_list = []

    # Header
    hdr = f"{'Family':<12} {'N':>2} | "
    if run_mixdom:
        hdr += f"{'MixDom SP':>9} {'TC':>6} | "
    if run_mafft:
        hdr += f"{'MAFFT SP':>8} {'TC':>6} | "
    if run_muscle:
        hdr += f"{'MUSCLE SP':>9} {'TC':>6} | "
    print(hdr)
    print("-" * len(hdr))

    for fi, family in enumerate(families):
        # Get n_seqs from input file
        raw_seqs = parse_fasta(os.path.join(in_dir, family))
        n_seq = len(raw_seqs)
        line = f"{family:<12} {n_seq:>2} | "

        # MixDom FSA (skip if --no-mixdom)
        if run_mixdom:
            mixdom_res = None
            try:
                mixdom_res = process_family(
                    family, in_dir, ref_dir, fsa_params, n_dom, n_frag,
                    pair_selection=args.pair_selection)
                if mixdom_res is not None:
                    results.append(mixdom_res)
                    line += f"{mixdom_res['sp']:>9.3f} {mixdom_res['tc']:>6.3f} | "
                else:
                    line += f"{'SKIP':>9} {'':>6} | "
            except Exception as e:
                line += f"{'ERR':>9} {'':>6} | "
                print(f"\n  ERROR on {family}: {e}")
                import traceback; traceback.print_exc()

        # MAFFT
        if run_mafft:
            mafft_res = process_family_mafft(family, in_dir, ref_dir)
            if mafft_res is not None:
                mafft_results_list.append({**mafft_res, 'family': family})
                line += f"{mafft_res['sp']:>8.3f} {mafft_res['tc']:>6.3f} | "
            else:
                line += f"{'ERR':>8} {'':>6} | "

        # MUSCLE
        if run_muscle:
            muscle_res = process_family_muscle(family, in_dir, ref_dir)
            if muscle_res is not None:
                muscle_results_list.append({**muscle_res, 'family': family})
                line += f"{muscle_res['sp']:>9.3f} {muscle_res['tc']:>6.3f} | "
            else:
                line += f"{'ERR':>9} {'':>6} | "

        print(f"[{fi+1:>3}/{len(families)}] {line}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    for label, rlist in [("MixDom FSA", results),
                         ("MAFFT", mafft_results_list),
                         ("MUSCLE", muscle_results_list)]:
        if rlist:
            sps = [r['sp'] for r in rlist]
            tcs = [r['tc'] for r in rlist]
            print(f"{label} (n={len(rlist)}):")
            print(f"  SP: mean={np.mean(sps):.4f}, median={np.median(sps):.4f}")
            print(f"  TC: mean={np.mean(tcs):.4f}, median={np.median(tcs):.4f}")
            if 'time' in rlist[0]:
                times = [r['time'] for r in rlist]
                print(f"  Time: mean={np.mean(times):.1f}s, total={np.sum(times):.0f}s")

    # Head-to-head comparisons
    if results and mafft_results_list:
        # Match by family name
        mixdom_by_fam = {r['family']: r for r in results}
        mafft_by_fam = {r['family']: r for r in mafft_results_list}
        muscle_by_fam = {r['family']: r for r in muscle_results_list} if muscle_results_list else {}

        common = sorted(set(mixdom_by_fam) & set(mafft_by_fam))
        if common:
            wins_sp = sum(1 for f in common if mixdom_by_fam[f]['sp'] > mafft_by_fam[f]['sp'])
            wins_tc = sum(1 for f in common if mixdom_by_fam[f]['tc'] > mafft_by_fam[f]['tc'])
            print(f"\nMixDom vs MAFFT (n={len(common)}): SP wins {wins_sp}, TC wins {wins_tc}")

        if muscle_by_fam:
            common_m = sorted(set(mixdom_by_fam) & set(muscle_by_fam))
            if common_m:
                wins_sp = sum(1 for f in common_m if mixdom_by_fam[f]['sp'] > muscle_by_fam[f]['sp'])
                wins_tc = sum(1 for f in common_m if mixdom_by_fam[f]['tc'] > muscle_by_fam[f]['tc'])
                print(f"MixDom vs MUSCLE (n={len(common_m)}): SP wins {wins_sp}, TC wins {wins_tc}")

    # ── Save results ─────────────────────────────────────────────────
    output = {
        'benchmark': 'BAliBASE 3',
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
    if mafft_results_list:
        output['mafft_results'] = mafft_results_list
        output['mafft_summary'] = {
            'sp_mean': float(np.mean([r['sp'] for r in mafft_results_list])),
            'sp_median': float(np.median([r['sp'] for r in mafft_results_list])),
            'tc_mean': float(np.mean([r['tc'] for r in mafft_results_list])),
            'tc_median': float(np.median([r['tc'] for r in mafft_results_list])),
        }
    if muscle_results_list:
        output['muscle_results'] = muscle_results_list
        output['muscle_summary'] = {
            'sp_mean': float(np.mean([r['sp'] for r in muscle_results_list])),
            'sp_median': float(np.median([r['sp'] for r in muscle_results_list])),
            'tc_mean': float(np.mean([r['tc'] for r in muscle_results_list])),
            'tc_median': float(np.median([r['tc'] for r in muscle_results_list])),
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

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
