#!/usr/bin/env python3
"""FSA alignment benchmark on OXBench using TKF92 pair HMM (single component).

Mirrors fsa_tkf92_balibase.py but on OXBench data (using the same
geometric-bin padding, _pairwise_posteriors_tkf92_jax forward-backward,
and MSA-reconstruction primitives from fsa_mixdom_pairhmm_oxbench.py).

Defaults to GPU. Override JAX_PLATFORMS=cpu / CUDA_VISIBLE_DEVICES=
in the environment to force CPU.

Usage:
    cd python && JAX_ENABLE_X64=1 \\
        uv run python experiments/fsa_tkf92_oxbench.py
"""

import os
os.environ.setdefault('JAX_ENABLE_X64', '1')

import json
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from tkfmixdom.jax.tree.fsa_anneal import (
    _pairwise_posteriors_tkf92_jax,
    select_pairs_full, select_pairs_erdos_renyi,
)
from tkfmixdom.jax.dp.hmm import _pad_to_bin, _pad_seq
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.util.msa_benchmark import parse_fasta, sp_tc_score
from tkfmixdom.util.expected_pair_f1 import expected_family_f1

# Reuse OxBench script's encoding + MSA-from-posteriors helpers.
from fsa_mixdom_pairhmm_oxbench import (
    encode_seq, build_msa_from_posteriors,
    build_msa_from_posteriors_multi, msa_to_aligned_strings,
)


def _run_pair_loop(int_seqs, names, pairs, ins, del_, ext, Q, pi):
    """Run the per-pair posterior computation for a fixed pair list."""
    import jax
    pair_posteriors = {}
    last_tau = 0.0
    for i_idx, j_idx in pairs:
        x = jnp.asarray(int_seqs[names[i_idx]], dtype=jnp.int32)
        y = jnp.asarray(int_seqs[names[j_idx]], dtype=jnp.int32)
        Lx_pad = _pad_to_bin(int(x.shape[0]))
        Ly_pad = _pad_to_bin(int(y.shape[0]))
        x_pad = _pad_seq(x, Lx_pad)
        y_pad = _pad_seq(y, Ly_pad)
        real_Lx = int(x.shape[0])
        real_Ly = int(y.shape[0])
        mp_pad, tau, lp = _pairwise_posteriors_tkf92_jax(
            x_pad, y_pad,
            jnp.int32(real_Lx), jnp.int32(real_Ly),
            jnp.float64(ins), jnp.float64(del_), jnp.float64(ext),
            jnp.asarray(Q), jnp.asarray(pi))
        mp = np.asarray(mp_pad)[:real_Lx, :real_Ly]
        pair_posteriors[(i_idx, j_idx)] = mp
        last_tau = float(tau)
    return pair_posteriors, last_tau


def process_family(family_name, in_dir, ref_dir, ins, del_, ext, Q, pi,
                     full_pair_cutoff=30, n_seeds=1, seed_base=42):
    """Process one OXBench family with single-TKF92 pair HMM.

    Pair selection is automatic: ``full`` if ``n_seqs <= full_pair_cutoff``,
    else ``erdos_renyi``.  On OOM during the per-pair loop, the family is
    retried once with ``erdos_renyi`` regardless of size.

    When ``n_seeds > 1``, annealing refinement is re-run for ``n_seeds``
    different seeds against the SAME pair_posteriors and per-seed SP/TC
    are recorded.
    """
    import gc
    import jax
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

    use_full = n_seqs <= full_pair_cutoff
    pairs = select_pairs_full(n_seqs) if use_full else \
        select_pairs_erdos_renyi(n_seqs)
    selection_used = 'full' if use_full else 'erdos_renyi'

    t0 = time.time()
    try:
        pair_posteriors, last_tau = _run_pair_loop(
            int_seqs, names, pairs, ins, del_, ext, Q, pi)
    except jax.errors.JaxRuntimeError as e:
        if 'RESOURCE_EXHAUSTED' not in str(e) or selection_used == 'erdos_renyi':
            raise
        print(f"    [OOM-fallback] {family_name}: full ({len(pairs)} pairs) "
              f"OOM'd; retrying with erdos_renyi", flush=True)
        jax.clear_caches()
        gc.collect()
        pairs = select_pairs_erdos_renyi(n_seqs)
        selection_used = 'erdos_renyi'
        pair_posteriors, last_tau = _run_pair_loop(
            int_seqs, names, pairs, ins, del_, ext, Q, pi)

    ref_aln = parse_fasta(ref_path)

    expected_micro = expected_family_f1(
        pair_posteriors, ref_aln, names, core_only=True)['micro']

    if n_seeds <= 1:
        msa_dict, msa_length = build_msa_from_posteriors(
            int_seqs, pair_posteriors, seed=seed_base)
        aligned = msa_to_aligned_strings(msa_dict)
        t_elapsed = time.time() - t0
        sp, tc = sp_tc_score(aligned, ref_aln)
        return {
            'family': family_name,
            'n_seqs': int(n_seqs),
            'n_pairs': int(len(pairs)),
            'pair_selection': selection_used,
            'msa_length': int(msa_length),
            'sp': float(sp),
            'tc': float(tc),
            'expected_f1_micro': expected_micro,
            'time': float(t_elapsed),
            'tau_last': last_tau,
            'n_seeds': 1,
        }

    seeds = [seed_base + k for k in range(n_seeds)]
    runs = build_msa_from_posteriors_multi(
        int_seqs, pair_posteriors, seeds, n_anneal=3)
    per_seed = []
    for seed, msa_dict, msa_length in runs:
        aligned = msa_to_aligned_strings(msa_dict)
        sp, tc = sp_tc_score(aligned, ref_aln)
        per_seed.append({'seed': seed, 'sp': float(sp), 'tc': float(tc),
                         'msa_length': int(msa_length)})
    t_elapsed = time.time() - t0
    sp_arr = np.array([r['sp'] for r in per_seed])
    tc_arr = np.array([r['tc'] for r in per_seed])
    best_idx = int(np.argmax(sp_arr + tc_arr))
    return {
        'family': family_name,
        'n_seqs': int(n_seqs),
        'n_pairs': int(len(pairs)),
        'pair_selection': selection_used,
        'msa_length': int(per_seed[best_idx]['msa_length']),
        'sp': float(per_seed[best_idx]['sp']),
        'tc': float(per_seed[best_idx]['tc']),
        'sp_mean': float(sp_arr.mean()),
        'sp_std': float(sp_arr.std()),
        'tc_mean': float(tc_arr.mean()),
        'tc_std': float(tc_arr.std()),
        'per_seed': per_seed,
        'best_seed': int(per_seed[best_idx]['seed']),
        'expected_f1_micro': expected_micro,
        'time': float(t_elapsed),
        'tau_last': last_tau,
        'n_seeds': int(n_seeds),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='FSA alignment benchmark on OXBench with single TKF92')
    parser.add_argument('--n-families', type=int, default=0,
                          help='Number of families (0=all, default: all)')
    parser.add_argument('--params-json', type=str,
        default=str(Path(__file__).parent / 'tkf92_fitted_params.json'),
        help='Path to TKF92 fitted params JSON.')
    parser.add_argument('--out', type=str,
        default='experiments/oxbench_tkf92.json')
    parser.add_argument('--full-pair-cutoff', type=int, default=30,
                          help='Use full pair-selection if n_seqs <= this; '
                               'else use erdos_renyi. Also: any family '
                               'OOM\\\'ing on full will fall back to '
                               'erdos_renyi automatically. Default 30.')
    parser.add_argument('--label', type=str, default='tkf92_lg08')
    parser.add_argument('--n-seeds', type=int, default=1,
                        help='Number of FSA annealing seeds per family '
                        '(reuses cached pair_posteriors). Default 1.')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base seed; seeds used are [seed_base, '
                        'seed_base+1, ...].')
    args = parser.parse_args()

    with open(args.params_json) as f:
        tkf92 = json.load(f)
    ins = float(tkf92['ins_rate'])
    del_ = float(tkf92['del_rate'])
    ext = float(tkf92['ext_rate'])
    print(f'Loaded TKF92 params from {args.params_json}:')
    print(f'  ins={ins:.5f}, del={del_:.5f}, ext={ext:.4f}, kappa={ins/del_:.4f}')

    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg, pi_lg = np.asarray(Q_lg), np.asarray(pi_lg)
    print(f'  emissions: LG08 ({pi_lg.shape[0]}x{pi_lg.shape[0]} GTR)')

    oxbench_dir = Path("~/bio-datasets/data/oxbench/ox").expanduser()
    in_dir = str(oxbench_dir / "in")
    ref_dir = str(oxbench_dir / "ref")

    families = sorted(os.listdir(in_dir))
    families = [f for f in families if os.path.exists(os.path.join(ref_dir, f))]
    if args.n_families > 0:
        families = families[:args.n_families]
    print(f'\nProcessing {len(families)} OXBench families '
          f'(full-pair-cutoff={args.full_pair_cutoff})')
    print(f'{"Family":<22} {"N":>3} | {"TKF92 SP":>9} {"TC":>6} | {"time":>6}')
    print('-' * 60)
    results = []
    for fi, family in enumerate(families):
        try:
            n_seqs = len(parse_fasta(os.path.join(in_dir, family)))
        except Exception as e:
            print(f'[{fi+1:>3}/{len(families)}] {family:<22} parse-err: {e}',
                  flush=True)
            continue
        line = f'{family:<22} {n_seqs:>3} | '
        try:
            res = process_family(family, in_dir, ref_dir, ins, del_, ext,
                                  Q_lg, pi_lg,
                                  full_pair_cutoff=args.full_pair_cutoff,
                                  n_seeds=args.n_seeds,
                                  seed_base=args.seed_base)
            if res is not None:
                results.append(res)
                line += f'{res["sp"]:>9.3f} {res["tc"]:>6.3f} | {res["time"]:>5.1f}s'
            else:
                line += f'{"SKIP":>9} {"":>6} |'
        except Exception as e:
            line += f'{"ERR":>9} {"":>6} | {type(e).__name__}: {e}'
        print(f'[{fi+1:>3}/{len(families)}] {line}', flush=True)

    print('\n' + '=' * 60)
    if results:
        sps = [r['sp'] for r in results]
        tcs = [r['tc'] for r in results]
        times = [r['time'] for r in results]
        print(f'Summary: n={len(results)}/{len(families)} families OK')
        print(f'  SP: mean={np.mean(sps):.4f}, median={np.median(sps):.4f}')
        print(f'  TC: mean={np.mean(tcs):.4f}, median={np.median(tcs):.4f}')
        print(f'  Time: mean={np.mean(times):.1f}s, total={sum(times):.0f}s')
    out = {'label': args.label,
            'params': {'ins': ins, 'del': del_, 'ext': ext},
            'results': results}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved to {args.out}')


if __name__ == '__main__':
    main()
