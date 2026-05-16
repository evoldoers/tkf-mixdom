#!/usr/bin/env python3
"""FSA alignment benchmark on BAliBASE using a mixture-of-K-TKF92s.

Family-level mixture inference via
`pairwise_posteriors_tkf92_mixture_family` in fsa_anneal.py.

Three approaches selectable via --n-outer-rounds:
  0 → approach (1): single-TKF92 anchor τ̂_i, then per-component FB at τ̂_i,
      family R, mixture posterior. No τ refit.
  1 → approach (2): + one round of τ refit using responsibility-weighted
      mixture E[ll], then final mixture posterior at refit τ.
  ≥2 → approach (3): iterate (FB → R → τ refit) `n_outer_rounds` times.

Defaults to GPU. JAX_PLATFORMS=cpu / CUDA_VISIBLE_DEVICES= override.

Usage:
    cd python && JAX_ENABLE_X64=1 \\
        uv run python experiments/fsa_tkf92_mixture_balibase.py \\
        --mixture-ckpt pfam/tkf92_mixture_K20_train.npz \\
        --n-outer-rounds 0  # approach 1 (anchor-only τ)
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from tkfmixdom.jax.tree.fsa_anneal import (
    pairwise_posteriors_tkf92_mixture_family,
    pairwise_posteriors_tkf92_mixture_streaming,
    _pairwise_posteriors_tkf92_jax,
    select_pairs_full, select_pairs_erdos_renyi,
    sequence_annealing,
)
from tkfmixdom.jax.dp.hmm import _pad_to_bin, _pad_seq
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.util.msa_benchmark import parse_fasta, sp_tc_score
from tkfmixdom.util.expected_pair_f1 import expected_family_f1

from fsa_mixdom_pairhmm_oxbench import (
    encode_seq, build_msa_from_posteriors,
    build_msa_from_posteriors_multi, msa_to_aligned_strings,
    AA_CHARS, AA_MAP,
)


def load_mixture_ckpt(path):
    """Load a fit_tkf92_mixture .npz ckpt (train_pfam-format) and extract
    per-component (ins, del, ext, Q, pi) plus mix_weights."""
    d = np.load(path, allow_pickle=True)
    K = int(d['dom_weights'].shape[0])
    A = int(d['dom_pis'].shape[1])
    mix_ins = np.asarray(d['dom_ins'], dtype=np.float64)            # (K,)
    mix_del = np.asarray(d['dom_del'], dtype=np.float64)            # (K,)
    ext_rates = np.asarray(d['ext_rates'], dtype=np.float64)        # (K, F, F)
    # F=1 only: single self-extension probability per component. The
    # mixture pair-HMM (`pairwise_posteriors_tkf92_mixture_family`)
    # accepts `mix_ext` as a (K,) scalar per component — the per-component
    # TKF92 model only has one extension probability. Multi-fragment
    # ckpts (F>1) would need a different architecture (and the F dimension
    # threaded through the pair-HMM); silently taking ext_rates[:, 0, 0]
    # would be a Category-3 silent-simplification bug. Fail loudly.
    if ext_rates.ndim == 3:
        K_check, F1, F2 = ext_rates.shape
        if F1 != 1 or F2 != 1:
            raise NotImplementedError(
                f"Mixture BAliBase supports only F=1 (single-fragment "
                f"per-component TKF92) ckpts; got ext_rates.shape="
                f"{ext_rates.shape}. The (K, F, F) extension matrix is not "
                f"yet threaded through `pairwise_posteriors_tkf92_mixture_family`.")
        mix_ext = ext_rates[:, 0, 0]
    elif ext_rates.ndim == 2:
        if ext_rates.shape[1] != 1:
            raise NotImplementedError(
                f"Mixture BAliBase supports only F=1 ckpts; got "
                f"ext_rates.shape={ext_rates.shape}.")
        mix_ext = ext_rates[:, 0]
    elif ext_rates.ndim == 1:
        mix_ext = ext_rates
    else:
        raise ValueError(f"Unexpected ext_rates.ndim={ext_rates.ndim}; "
                         f"shape={ext_rates.shape}")
    mix_Q = np.asarray(d['dom_Qs'], dtype=np.float64)               # (K, A, A)
    mix_pi = np.asarray(d['dom_pis'], dtype=np.float64)             # (K, A)
    mix_weights = np.asarray(d['dom_weights'], dtype=np.float64)
    mix_weights = mix_weights / mix_weights.sum()
    return dict(K=K, A=A, mix_weights=mix_weights,
                mix_ins=mix_ins, mix_del=mix_del, mix_ext=mix_ext,
                mix_Q=mix_Q, mix_pi=mix_pi)


def pad_family(int_seqs, names, pairs):
    """Pad all (x, y) pair sequences to the SAME (Lx_pad, Ly_pad) per pair.

    For simplicity (and to keep the JIT cache small), we pad x and y to
    each pair's own bin — but we batch the K-component FB across all
    pairs, so we additionally need same-shape sequences across the
    batch. We solve that by padding to the MAX (Lx_pad, Ly_pad) across
    all pairs in the family. The trim happens at the caller.
    """
    Lx_max = max(len(int_seqs[names[i]]) for i, _ in pairs)
    Ly_max = max(len(int_seqs[names[j]]) for _, j in pairs)
    Lx_pad = _pad_to_bin(Lx_max)
    Ly_pad = _pad_to_bin(Ly_max)
    n_pairs = len(pairs)
    pairs_x = np.zeros((n_pairs, Lx_pad), dtype=np.int32)
    pairs_y = np.zeros((n_pairs, Ly_pad), dtype=np.int32)
    real_Lxs = np.zeros(n_pairs, dtype=np.int32)
    real_Lys = np.zeros(n_pairs, dtype=np.int32)
    for k, (i, j) in enumerate(pairs):
        x = int_seqs[names[i]]
        y = int_seqs[names[j]]
        pairs_x[k, :len(x)] = x
        pairs_y[k, :len(y)] = y
        real_Lxs[k] = len(x)
        real_Lys[k] = len(y)
    return (jnp.asarray(pairs_x), jnp.asarray(pairs_y),
            jnp.asarray(real_Lxs), jnp.asarray(real_Lys))


def process_family(family_name, in_dir, ref_dir,
                   anchor, mix, n_outer_rounds,
                   pair_selection='full', k_chunk=None, pair_chunk=None,
                   sp_tc_core_only=True,
                   n_seeds=1, seed_base=42):
    """Process one BAliBASE family with the mixture-of-TKF92s pipeline.

    When ``n_seeds > 1``, the annealing refinement is re-run for
    ``n_seeds`` different seeds against the SAME pair_posteriors and
    per-seed SP/TC are recorded.
    """
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
    pairs_x, pairs_y, real_Lxs, real_Lys = pad_family(int_seqs, names, pairs)
    if n_outer_rounds == 0:  # streaming-accumulate (memory-efficient)
        # Streaming: anchor tau via single-TKF92 NR, then scan over K to
        # accumulate R-weighted pair_marg.  Memory: O(1 component + total).
        @jax.jit
        def _anchor_tau_one(x, y, rL, rR):
            _, tau_opt, _ = _pairwise_posteriors_tkf92_jax(
                x, y, rL, rR,
                jnp.float64(anchor['ins']), jnp.float64(anchor['del']),
                jnp.float64(anchor['ext']),
                jnp.asarray(anchor['Q']), jnp.asarray(anchor['pi']),
                n_newton=3, tau_init=1.0)
            return tau_opt
        _anchor_tau_vmap = jax.vmap(_anchor_tau_one)
        n_pairs = pairs_x.shape[0]

        # OOM-based pair_chunk fallback.  Start at the user-supplied
        # pair_chunk (or n_pairs if None) and halve on each
        # RESOURCE_EXHAUSTED until success or pair_chunk=1.
        import gc
        chunk_try = pair_chunk if pair_chunk else n_pairs
        chunk_try = min(chunk_try, n_pairs)
        last_err = None
        while chunk_try >= 1:
            try:
                # Anchor tau (recomputed per chunk size — cheap).
                if chunk_try >= n_pairs:
                    tau_anchor = _anchor_tau_vmap(
                        pairs_x, pairs_y, real_Lxs, real_Lys)
                else:
                    chunks = []
                    for s in range(0, n_pairs, chunk_try):
                        e = min(s + chunk_try, n_pairs)
                        chunks.append(_anchor_tau_vmap(
                            pairs_x[s:e], pairs_y[s:e],
                            real_Lxs[s:e], real_Lys[s:e]))
                    tau_anchor = jnp.concatenate(chunks)
                mp_mix, R_family = pairwise_posteriors_tkf92_mixture_streaming(
                    pairs_x, pairs_y, real_Lxs, real_Lys, tau_anchor,
                    jnp.asarray(mix['mix_ins']), jnp.asarray(mix['mix_del']),
                    jnp.asarray(mix['mix_ext']),
                    jnp.asarray(mix['mix_Q']), jnp.asarray(mix['mix_pi']),
                    jnp.asarray(mix['mix_weights']),
                    pair_chunk=chunk_try)
                if chunk_try != (pair_chunk or n_pairs):
                    print(f"    [OOM-fallback] succeeded at "
                          f"pair_chunk={chunk_try} for {family_name}",
                          flush=True)
                tau_per_pair = tau_anchor
                break
            except jax.errors.JaxRuntimeError as e:
                if 'RESOURCE_EXHAUSTED' not in str(e) and \
                   'OUT_OF_MEMORY' not in str(e):
                    raise
                last_err = e
                if chunk_try == 1:
                    raise  # already at minimum, give up
                jax.clear_caches()
                gc.collect()
                new_chunk = max(1, chunk_try // 2)
                print(f"    [OOM-fallback] {family_name} OOM at "
                      f"pair_chunk={chunk_try}; retrying at {new_chunk}",
                      flush=True)
                chunk_try = new_chunk
    else:
        mp_mix, tau_per_pair, R_family = pairwise_posteriors_tkf92_mixture_family(
            pairs_x, pairs_y, real_Lxs, real_Lys,
            anchor_ins=anchor['ins'], anchor_del=anchor['del'],
            anchor_ext=anchor['ext'], anchor_Q=anchor['Q'], anchor_pi=anchor['pi'],
            mix_weights=mix['mix_weights'],
            mix_ins=mix['mix_ins'], mix_del=mix['mix_del'], mix_ext=mix['mix_ext'],
            mix_Q=mix['mix_Q'], mix_pi=mix['mix_pi'],
            n_outer_rounds=n_outer_rounds,
            k_chunk=k_chunk, pair_chunk=pair_chunk)

    # Build pair_posteriors dict, trimming each posterior to real shape.
    pair_posteriors = {}
    mp_np = np.asarray(mp_mix)
    for idx, (i, j) in enumerate(pairs):
        rL = int(real_Lxs[idx])
        rR = int(real_Lys[idx])
        pair_posteriors[(i, j)] = mp_np[idx, :rL, :rR]

    ref_aln = parse_fasta(ref_path)

    expected_micro = expected_family_f1(
        pair_posteriors, ref_aln, names, core_only=sp_tc_core_only)['micro']

    if n_seeds <= 1:
        msa_dict, msa_length = build_msa_from_posteriors(
            int_seqs, pair_posteriors, seed=seed_base)
        aligned = msa_to_aligned_strings(msa_dict)
        t_elapsed = time.time() - t0
        sp, tc = sp_tc_score(aligned, ref_aln, core_only=sp_tc_core_only)
        return {
            'family': family_name,
            'n_seqs': int(n_seqs),
            'n_pairs': int(len(pairs)),
            'msa_length': int(msa_length),
            'sp': float(sp),
            'tc': float(tc),
            'expected_f1_micro': expected_micro,
            'time': float(t_elapsed),
            'R_family': [float(r) for r in np.asarray(R_family)],
            'tau_pair_mean': float(np.mean(np.asarray(tau_per_pair))),
            'tau_pair_max': float(np.max(np.asarray(tau_per_pair))),
            'n_seeds': 1,
        }

    seeds = [seed_base + k for k in range(n_seeds)]
    runs = build_msa_from_posteriors_multi(
        int_seqs, pair_posteriors, seeds, n_anneal=3)
    per_seed = []
    for seed, msa_dict, msa_length in runs:
        aligned = msa_to_aligned_strings(msa_dict)
        sp, tc = sp_tc_score(aligned, ref_aln, core_only=sp_tc_core_only)
        per_seed.append({'seed': seed, 'sp': float(sp), 'tc': float(tc),
                         'msa_length': int(msa_length)})
    t_elapsed = time.time() - t0
    sp_arr = np.array([r['sp'] for r in per_seed])
    tc_arr = np.array([r['tc'] for r in per_seed])
    score_arr = sp_arr + tc_arr
    best_idx = int(np.argmax(score_arr))
    checkpoints = [n for n in (1, 2, 4, 8, 16, 32, 64, 128)
                   if n <= len(per_seed)]
    if len(per_seed) not in checkpoints:
        checkpoints.append(len(per_seed))
    best_at_n = []
    for n in checkpoints:
        prefix_sp = sp_arr[:n]
        prefix_tc = tc_arr[:n]
        prefix_score = prefix_sp + prefix_tc
        idx = int(np.argmax(prefix_score))
        best_at_n.append({
            'n': int(n),
            'sp': float(prefix_sp[idx]),
            'tc': float(prefix_tc[idx]),
            'best_seed': int(per_seed[idx]['seed']),
        })
    return {
        'family': family_name,
        'n_seqs': int(n_seqs),
        'n_pairs': int(len(pairs)),
        'msa_length': int(per_seed[best_idx]['msa_length']),
        'sp': float(per_seed[best_idx]['sp']),
        'tc': float(per_seed[best_idx]['tc']),
        'sp_mean': float(sp_arr.mean()),
        'sp_std': float(sp_arr.std()),
        'tc_mean': float(tc_arr.mean()),
        'tc_std': float(tc_arr.std()),
        'best_at_n': best_at_n,
        'per_seed': per_seed,
        'best_seed': int(per_seed[best_idx]['seed']),
        'expected_f1_micro': expected_micro,
        'time': float(t_elapsed),
        'R_family': [float(r) for r in np.asarray(R_family)],
        'tau_pair_mean': float(np.mean(np.asarray(tau_per_pair))),
        'tau_pair_max': float(np.max(np.asarray(tau_per_pair))),
        'n_seeds': int(n_seeds),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='FSA alignment benchmark on BAliBASE with mixture-of-TKF92s')
    parser.add_argument('--mixture-ckpt', type=str, required=True,
                        help='K-component TKF92 mixture .npz '
                        '(output of fit_tkf92_mixture.py).')
    parser.add_argument('--anchor-json', type=str,
                        default=str(Path(__file__).parent / 'tkf92_fitted_params.json'),
                        help='Single-TKF92 anchor for initial τ̂_i fit.')
    parser.add_argument('--n-outer-rounds', type=int, default=0,
                        help='0=approach (1) [default], 1=approach (2), '
                        '≥2=approach (3).')
    parser.add_argument('--k-chunk', type=int, default=1,
                        help='Process the K mixture components in chunks of '
                        'this size (parallel within chunk via vmap, '
                        'sequential across chunks via lax.map). Lower = '
                        'less peak GPU memory, slower. Must divide K. '
                        'Default 1 (most conservative; works on 11 GiB GPU '
                        'for the largest BAliBase family BB11018 '
                        'n_seqs=12, max_len=750 with --pair-chunk 8).')
    parser.add_argument('--pair-chunk', type=int, default=8,
                        help='Process pairs in chunks of this size (parallel '
                        'within chunk via vmap, sequential across chunks '
                        'via Python loop). Lower = less peak GPU memory, '
                        'slower. Need not divide n_pairs (last chunk may '
                        'be smaller). Default 8 (safe for 11 GiB GPU on '
                        'all BAliBase families).')
    parser.add_argument('--family-order', type=str, default='hardest_first',
                        choices=['hardest_first', 'alphabetical'],
                        help='Order in which to process families. '
                        '"hardest_first" sorts by n_seqs * max_seq_len^2 '
                        'descending — surfaces OOMs / hangs in the first '
                        'few minutes (per feedback_optimize_for_local_machine). '
                        '"alphabetical" matches the existing benchmark scripts.')
    parser.add_argument('--n-families', type=int, default=0,
                        help='Number of families (0=all).')
    parser.add_argument('--pair-selection', type=str, default='full',
                        choices=['full', 'erdos_renyi'])
    parser.add_argument('--out', type=str, default=None,
                        help='Output JSON (default derived from --mixture-ckpt '
                        'and --n-outer-rounds).')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--skip-families', type=str, default='',
                        help='Comma-separated list of family names to hard-skip '
                        '(useful when known to OOM at any chunk size).')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Existing JSON output to resume from. Families '
                        'already present in its results array are skipped; '
                        'remaining families are appended. The output JSON '
                        '(--out) is overwritten with the merged results.')
    parser.add_argument('--in-dir', type=str, default=None,
                        help='Override input MSA directory (default: BAliBASE 3 '
                             '~/bio-datasets/data/balibase/bali3pdbm/in).')
    parser.add_argument('--ref-dir', type=str, default=None,
                        help='Override reference MSA directory.')
    parser.add_argument('--families', type=str, default=None,
                        help='Comma-separated explicit family list; '
                        'overrides --n-families and --family-order.')
    parser.add_argument('--n-seeds', type=int, default=1,
                        help='FSA annealing seeds per family '
                        '(reuses cached pair_posteriors). Default 1.')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base seed; seeds used are [seed_base, '
                        'seed_base+1, ...].')
    parser.add_argument('--core-only', type=lambda s: s.lower() != 'false',
                        default=True,
                        help='SP/TC scoring uses BAliBASE core-only columns '
                             'when True (default; correct for BAliBASE 3). '
                             'Set --core-only=false for OxBench (no core marks).')
    args = parser.parse_args()

    # Load anchor
    with open(args.anchor_json) as f:
        atkf = json.load(f)
    Q_lg, pi_lg = rate_matrix_lg()
    anchor = dict(
        ins=jnp.float64(atkf['ins_rate']),
        del_=jnp.float64(atkf['del_rate']),
        ext=jnp.float64(atkf['ext_rate']),
        Q=jnp.asarray(Q_lg, dtype=jnp.float64),
        pi=jnp.asarray(pi_lg, dtype=jnp.float64),
    )
    # The function signature uses `anchor_del=` (not `anchor_del_=`),
    # so map del_ → del here.
    anchor['del'] = anchor.pop('del_')
    print(f'Anchor TKF92 (single component, LG08 emissions): '
          f'ins={float(anchor["ins"]):.5f} del={float(anchor["del"]):.5f} '
          f'ext={float(anchor["ext"]):.4f}')

    # Load mixture
    mix = load_mixture_ckpt(args.mixture_ckpt)
    print(f'Mixture: K={mix["K"]} components from {args.mixture_ckpt}')
    print(f'  mix_weights: {np.round(mix["mix_weights"], 4)}')
    print(f'  mix_ins range: [{mix["mix_ins"].min():.5f}, {mix["mix_ins"].max():.5f}]')
    print(f'  mix_del range: [{mix["mix_del"].min():.5f}, {mix["mix_del"].max():.5f}]')
    print(f'  mix_ext range: [{mix["mix_ext"].min():.4f}, {mix["mix_ext"].max():.4f}]')

    # Dataset paths (default BAliBASE; --in-dir / --ref-dir override).
    if args.in_dir is None or args.ref_dir is None:
        balibase_dir = Path("~/bio-datasets/data/balibase/bali3pdbm").expanduser()
        in_dir = args.in_dir or str(balibase_dir / "in")
        ref_dir = args.ref_dir or str(balibase_dir / "ref")
    else:
        in_dir = args.in_dir
        ref_dir = args.ref_dir
    if args.families:
        families = [f.strip() for f in args.families.split(',') if f.strip()]
    else:
        families = sorted(os.listdir(in_dir))
        families = [f for f in families if os.path.exists(os.path.join(ref_dir, f))]
    if not args.families and args.family_order == 'hardest_first':
        # Sort by n_seqs * max_len^2 descending — proxy for FB cost
        # (n_pairs * Lx_pad * Ly_pad). Surfaces OOMs / hangs early.
        sizes = []
        for fam in families:
            try:
                raw = parse_fasta(os.path.join(in_dir, fam))
                if len(raw) < 2:
                    sizes.append((fam, 0))
                    continue
                max_len = max(len(s) for s in raw.values())
                n_seqs = len(raw)
                cost = n_seqs * max_len * max_len
                sizes.append((fam, cost))
            except Exception:
                sizes.append((fam, 0))
        sizes.sort(key=lambda r: -r[1])
        families = [f for f, _ in sizes]
        print(f'Processing families in hardest-first order; top 5:')
        for f, c in sizes[:5]:
            print(f'  {f}: cost-proxy={c}')
    if args.n_families > 0:
        families = families[:args.n_families]

    label = args.label or (f'tkf92_mix{mix["K"]}_a{args.n_outer_rounds + 1}'
                           if args.n_outer_rounds <= 1
                           else f'tkf92_mix{mix["K"]}_a3_r{args.n_outer_rounds}')
    out = args.out or f'experiments/balibase_{label}.json'

    resume_results = []
    skip_set = set()
    if args.resume_from:
        with open(args.resume_from) as f:
            prev = json.load(f)
        resume_results = list(prev.get('results', []))
        skip_set = {r['family'] for r in resume_results}
        print(f'Resuming from {args.resume_from}: '
              f'{len(skip_set)} families already done, will skip them.')
        families = [f for f in families if f not in skip_set]
    if args.skip_families.strip():
        hardskip = {f.strip() for f in args.skip_families.split(',') if f.strip()}
        before = len(families)
        families = [f for f in families if f not in hardskip]
        print(f'Hard-skip ({len(hardskip)} families): {sorted(hardskip)} '
              f'— filtered {before - len(families)} from this run.')

    print(f'\nProcessing {len(families)} families '
          f'(approach selector n_outer_rounds={args.n_outer_rounds}, '
          f'pair_selection={args.pair_selection}, label={label})')
    print(f'{"Family":<12} {"N":>2} | {"MIX SP":>8} {"TC":>6} | {"time":>5}')
    print('-' * 55)

    def _save_partial(rs):
        merged = resume_results + rs
        partial = {
            'benchmark': 'BAliBASE 3 (bali3pdbm)',
            'model': f'TKF92 mixture (K={mix["K"]}) family-level posteriors',
            'mixture_ckpt': args.mixture_ckpt,
            'anchor_json': args.anchor_json,
            'n_outer_rounds': int(args.n_outer_rounds),
            'approach': (1 if args.n_outer_rounds == 0
                         else 2 if args.n_outer_rounds == 1 else 3),
            'pair_selection': args.pair_selection,
            'label': label,
            'n_families': len(merged),
            'partial': len(rs) < len(families),
            'results': merged,
        }
        with open(out, 'w') as f:
            json.dump(partial, f, indent=2)

    results = []
    for fi, family in enumerate(families):
        raw_seqs = parse_fasta(os.path.join(in_dir, family))
        n_seq = len(raw_seqs)
        # Estimate peak GPU memory cost & pre-emptively shrink pair_chunk.
        # Peak ~ pair_chunk * Lx_pad * Ly_pad * (5 WFST states) * 8 bytes
        #       * 4 trellis tensors (alpha, beta, joint, marg). Conservative.
        max_len = max((len(s) for s in raw_seqs.values()), default=1)
        Lx_pad = _pad_to_bin(max_len); Ly_pad = _pad_to_bin(max_len)
        per_pair_bytes = Lx_pad * Ly_pad * 5 * 8 * 4
        # Target: stay under 4 GiB peak mixture-pair memory (roughly the
        # safe headroom on the 11 GiB GPU after the K=20 mixture's static
        # state). Halve pair_chunk until estimate fits.
        eff_pair_chunk = args.pair_chunk
        budget_bytes = 4 * 1024**3
        while (eff_pair_chunk > 1 and
               eff_pair_chunk * per_pair_bytes > budget_bytes):
            eff_pair_chunk //= 2
        chunk_note = (f' [pair_chunk={eff_pair_chunk}]'
                      if eff_pair_chunk != args.pair_chunk
                      else '')
        # Pre-print so a slow / hanging family is visible BEFORE results.
        print(f'[{fi+1:>3}/{len(families)}] {family:<12} '
              f'N={n_seq:>2} Lmax={max_len:>4}'
              f' est_pair_mem={per_pair_bytes/1024**2:.0f}MB'
              f'{chunk_note} ... ', end='', flush=True)
        line = ''
        try:
            res = process_family(family, in_dir, ref_dir,
                                 anchor, mix, args.n_outer_rounds,
                                 pair_selection=args.pair_selection,
                                 k_chunk=args.k_chunk,
                                 pair_chunk=eff_pair_chunk,
                                 sp_tc_core_only=args.core_only,
                                 n_seeds=args.n_seeds,
                                 seed_base=args.seed_base)
            if res is not None:
                results.append(res)
                line = f'SP={res["sp"]:.3f} TC={res["tc"]:.3f} ({res["time"]:.1f}s)'
            else:
                line = 'SKIP'
        except Exception as e:
            line = f'ERR {e}'
            import traceback; traceback.print_exc()
        print(line, flush=True)
        # Save partial after every family so an OOM/timeout doesn't lose progress.
        _save_partial(results)

    # Merge with any resumed results.
    results = resume_results + results

    print('\n' + '=' * 50)
    if results:
        sps = [r['sp'] for r in results]; tcs = [r['tc'] for r in results]
        times = [r['time'] for r in results]
        print(f'TKF92 mixture (K={mix["K"]}, approach selector={args.n_outer_rounds}, '
              f'n={len(results)}):')
        print(f'  SP: mean={np.mean(sps):.4f}, median={np.median(sps):.4f}')
        print(f'  TC: mean={np.mean(tcs):.4f}, median={np.median(tcs):.4f}')
        print(f'  Time: mean={np.mean(times):.1f}s, total={np.sum(times):.0f}s')

    output = {
        'benchmark': 'BAliBASE 3 (bali3pdbm)',
        'model': f'TKF92 mixture (K={mix["K"]}) family-level posteriors',
        'mixture_ckpt': args.mixture_ckpt,
        'anchor_json': args.anchor_json,
        'n_outer_rounds': int(args.n_outer_rounds),
        'approach': (1 if args.n_outer_rounds == 0
                     else 2 if args.n_outer_rounds == 1 else 3),
        'pair_selection': args.pair_selection,
        'label': label,
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
    with open(out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to {out}')


if __name__ == '__main__':
    main()
