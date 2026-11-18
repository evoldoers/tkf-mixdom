#!/usr/bin/env python3
"""MixFrag-S svi-Baum-Welch trainer (per-fragtype substitution matrices).

MixFrag-S = the paper's MixFrag (TKF92 with F fragtypes: shared indel rates
lam, mu; per-fragtype extension r_f and weight w_f) in which fragtype f has
its own reversible substitution model (S_f, pi_f) -> Q_f.  No domain level,
no nesting.  Trainer: ``tkfmixdom.jax.train.mixfrag_svi_bw.svi_bw_mixfrag_s``.

Corpus: a precompiled pair directory (manifest.json + shard_*.jsonl.zst),
read with ``train_pfam.PrecompiledPairSource`` so each pair carries its
own branch length t and its training alignment.

E-step modes (--dp-mode):
  path          (default) 1D alignment-conditioned forward-backward along
                each training alignment's column sequence — the paper's
                svi-BW convention (train_pfam's default).  No (Lx, Ly)
                tables, so no --max-len is needed.
  match_aligned 2D forward-backward with the matches pinned to the
                alignment's (i, j) pairs and the indel orderings summed.
  unaligned     the fully unaligned 2D forward-backward (every alignment
                summed over; ~43 ms/pair, needs --max-len).

Checkpoint (npz, written atomically every --ckpt-every iterations and at
the end): 'ins_rate', 'del_rate' (scalars), 'exts' (F,), 'weights' (F,),
'S_exch' (F,A,A), 'pi' (F,A), 'Q' (F,A,A), 'em_iter' (int),
'n_fragtypes' (int), '_config' (json string); plus 'svi_suff_*' (the
blended sufficient statistics, used by --resume) and 'history' (json).

Typical:
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
  .venv/bin/python train_mixfrag_s.py \
      --precompiled-pairs pfam/precompiled_v3_train_nobali \
      --n-fragtypes 2 --dp-mode path --n-iter 1620 --svi-batch 2000 --seed 42 \
      --checkpoint pfam/svibw_mixfrags_F2_nobali_1620.npz --ckpt-every 10 \
      --jax-cache-dir ~/.cache/jax_mixfrag_s --log-file pfam/svibw_mixfrags_F2_nobali_1620.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

CONTRACT_KEYS = ('ins_rate', 'del_rate', 'exts', 'weights', 'S_exch', 'pi',
                 'Q', 'em_iter', 'n_fragtypes', '_config')
SUFF_KEYS = ('B', 'D', 'S', 'L', 'M', 'T', 'F_f', 'E_f', 'W', 'U', 'V')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--precompiled-pairs', required=True,
                   help='precompiled pair dir (manifest.json + shard_*.jsonl.zst)')
    p.add_argument('--n-fragtypes', type=int, default=2)
    p.add_argument('--dp-mode', choices=('path', 'match_aligned', 'unaligned'),
                   default='path',
                   help='E-step: path (1D along the training alignment; '
                        'default), match_aligned (2D, matches pinned), '
                        'unaligned (2D, all alignments summed)')
    p.add_argument('--fb-scan-mode', choices=('associative', 'sequential'),
                   default='associative',
                   help='1D forward-backward scan for --dp-mode path')
    p.add_argument('--n-iter', type=int, default=1620)
    p.add_argument('--svi-batch', type=int, default=2000)
    p.add_argument('--svi-tau', type=float, default=1.0)
    p.add_argument('--svi-kappa', type=float, default=0.7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--init-subst-npz', default='pfam/gtr_refit_Qpi.npz',
                   help='npz with keys Q, pi, S; both fragtypes start here')
    p.add_argument('--init-mixfrag-json',
                   default='experiments/mixfrag_F/mixfrag_F2_logspace.json',
                   help='json with keys lam, mu, exts, weights')
    # priors (the paper's svi-BW recipe)
    p.add_argument('--prior-alpha-lam', type=float, default=2.0)
    p.add_argument('--prior-alpha-mu', type=float, default=2.0)
    p.add_argument('--prior-beta', type=float, default=10.0)
    p.add_argument('--ext-prior-alpha', type=float, default=2.0)
    p.add_argument('--ext-prior-beta', type=float, default=3.0)
    p.add_argument('--weight-prior-alpha', type=float, default=1.5)
    p.add_argument('--pi-pseudo', type=float, default=10.0)
    p.add_argument('--S-pseudo', type=float, default=5.0)
    # output / resume
    p.add_argument('--checkpoint', required=True, help='output npz path')
    p.add_argument('--ckpt-every', type=int, default=10)
    p.add_argument('--resume', action='store_true',
                   help='continue from --checkpoint if it exists')
    p.add_argument('--jax-cache-dir', default=None)
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--log-file', default=None)
    # corpus subsetting / feasibility
    p.add_argument('--max-len', type=int, default=-1,
                   help='drop pairs with max(Lx, Ly) above this; default -1 = '
                        '1024 for the 2D modes (FB memory), no cap for path; '
                        '0 = no cap')
    p.add_argument('--n-pairs-subset', type=int, default=0,
                   help='train on a random subset of this many pairs (0 = all)')
    p.add_argument('--subset-seed', type=int, default=0)
    p.add_argument('--n-total-pairs', type=int, default=0,
                   help='N_total for the SVI scale (default: #pairs loaded)')
    p.add_argument('--full-batch', action='store_true',
                   help='plain EM: batch = all pairs, eta = 1')
    p.add_argument('--chunk-max', type=int, default=-1,
                   help='max pairs per vmapped E-step call (default -1 = '
                        '1024 for --dp-mode path, 256 for the 2D modes)')
    p.add_argument('--chunk-mem-gb', type=float, default=2.0,
                   help='GPU memory budget per E-step call (sizes chunks per bin)')
    p.add_argument('--hr-chunk', type=int, default=128)
    p.add_argument('--bin-stride', type=int, default=1,
                   help='keep every k-th padding bin (fewer JIT shapes; 1 = FB bins)')
    # validation (optional)
    p.add_argument('--val-precompiled-pairs', default=None)
    p.add_argument('--val-every', type=int, default=0)
    p.add_argument('--val-n-pairs', type=int, default=2000)
    return p.parse_args(argv)


def load_xy_t_pairs(precompiled_dir, max_len=None, log=print):
    """Load (x_int, y_int, t) pairs with the loader train_pfam.py uses."""
    from train_pfam import PrecompiledPairSource
    src = PrecompiledPairSource(precompiled_dir)
    log(f'Loading {src.description} ...')
    t0 = time.time()
    raw = src.load_all_pairs()          # (x, y, states, anc, desc, t)
    pairs = []
    n_drop = 0
    for x, y, _st, _ac, _dc, t in raw:
        if max_len and max(x.shape[0], y.shape[0]) > max_len:
            n_drop += 1
            continue
        pairs.append((np.asarray(x, dtype=np.int32),
                      np.asarray(y, dtype=np.int32), float(t)))
    del raw, src
    log(f'  {len(pairs)} pairs loaded ({n_drop} dropped by --max-len {max_len}) '
        f'in {time.time() - t0:.0f}s')
    return pairs


_AA_LUT = None


def _aa_lut():
    """uint8 -> residue index lookup (20 standard amino acids; anything
    else -> 20, the wildcard slot)."""
    global _AA_LUT
    if _AA_LUT is None:
        from tkfmixdom.jax.util.pair_format import AA_TO_IDX
        lut = np.full(256, 20, dtype=np.int8)
        for c, i in AA_TO_IDX.items():
            lut[ord(c)] = i
        _AA_LUT = lut
    return _AA_LUT


_RLE_RE = None


def decode_record_compact(rec):
    """Vectorised ``decode_pair``: (x_int int8, y_int int8, states int8, t)
    with no per-column Python loop (the full corpus is ~1M pairs)."""
    global _RLE_RE
    import re
    if _RLE_RE is None:
        _RLE_RE = re.compile(r'([MID])(\d+)')
    lut = _aa_lut()
    x = lut[np.frombuffer(rec['x'].encode('ascii'), dtype=np.uint8)]
    y = lut[np.frombuffer(rec['y'].encode('ascii'), dtype=np.uint8)]
    runs = _RLE_RE.findall(rec['a'])
    if runs:
        codes = np.array([{'M': 1, 'I': 2, 'D': 3}[c] for c, _ in runs], dtype=np.int8)
        counts = np.array([int(n) for _, n in runs], dtype=np.int64)
        states = np.repeat(codes, counts)
    else:
        states = np.zeros(0, dtype=np.int8)
    return x, y, states, float(rec['t'])


def load_pairs_for_mode(precompiled_dir, dp_mode, max_len=0, log=print):
    """Load the corpus in ``dp_mode``'s compact pair format (see
    ``mixfrag_svi_bw.as_mode_pairs``): 'path' -> (states, anc_full,
    desc_full, t) int8 column arrays; 'match_aligned' -> (x, y, t, mi, mj);
    'unaligned' -> (x, y, t).  ``max_len`` > 0 drops pairs with
    max(Lx, Ly) above it (0 = keep all)."""
    from train_pfam import PrecompiledPairSource
    from tkfmixdom.jax.train.mixfrag_svi_bw import match_positions_from_states
    src = PrecompiledPairSource(precompiled_dir)
    log(f'Loading {src.description} [dp_mode={dp_mode}] ...')
    t0 = time.time()
    pairs = []
    n_drop = 0
    for shard_name in src.shard_files:
        for rec in src._read_shard(shard_name):
            x, y, states, t = decode_record_compact(rec)
            if max_len and max(x.shape[0], y.shape[0]) > max_len:
                n_drop += 1
                continue
            if dp_mode == 'path':
                L = states.shape[0]
                anc_full = np.zeros(L, dtype=np.int8)
                desc_full = np.zeros(L, dtype=np.int8)
                anc_full[states != 2] = x
                desc_full[states != 3] = y
                pairs.append((states, anc_full, desc_full, t))
            elif dp_mode == 'match_aligned':
                mi, mj = match_positions_from_states(states)
                pairs.append((x.astype(np.int32), y.astype(np.int32), t, mi, mj))
            else:
                pairs.append((x.astype(np.int32), y.astype(np.int32), t))
    del src
    log(f'  {len(pairs)} pairs loaded ({n_drop} dropped by --max-len {max_len}) '
        f'in {time.time() - t0:.0f}s')
    return pairs


def subset_pairs(pairs, n, seed):
    if not n or n >= len(pairs):
        return pairs
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(pairs), n, replace=False)
    return [pairs[i] for i in sorted(ix)]


def load_init(args, F, log=print):
    d = np.load(args.init_subst_npz)
    S0 = np.asarray(d['S'], dtype=np.float64)
    pi0 = np.asarray(d['pi'], dtype=np.float64)
    A = pi0.shape[0]
    init_S = np.tile(S0[None], (F, 1, 1))
    init_pi = np.tile(pi0[None], (F, 1))
    with open(args.init_mixfrag_json) as fh:
        mf = json.load(fh)
    lam, mu = float(mf['lam']), float(mf['mu'])
    exts = np.asarray(mf['exts'], dtype=np.float64)
    weights = np.asarray(mf['weights'], dtype=np.float64)
    if exts.shape[0] != F:
        log(f'  WARNING: {args.init_mixfrag_json} has F={exts.shape[0]} != '
            f'--n-fragtypes {F}; using spread exts / uniform weights.')
        exts = np.linspace(0.3, 0.8, F)
        weights = np.full(F, 1.0 / F)
    log(f'  init: lam={lam:.6f} mu={mu:.6f} exts={exts} weights={weights}; '
        f'S, pi from {args.init_subst_npz} (A={A}) tiled to F={F}')
    return lam, mu, exts, weights, init_S, init_pi


def save_checkpoint_atomic(path, state, config, F):
    """Write the CONTRACT npz atomically (tmp file + os.replace)."""
    out = {
        'ins_rate': np.float64(state['lam']),
        'del_rate': np.float64(state['mu']),
        'exts': np.asarray(state['exts'], dtype=np.float64),
        'weights': np.asarray(state['weights'], dtype=np.float64),
        'S_exch': np.asarray(state['S'], dtype=np.float64),
        'pi': np.asarray(state['pi'], dtype=np.float64),
        'Q': np.asarray(state['Q'], dtype=np.float64),
        'em_iter': np.int64(state['em_iter']),
        'n_fragtypes': np.int64(F),
        '_config': np.array(json.dumps(config)),
    }
    if state.get('suff_blend') is not None:
        for k in SUFF_KEYS:
            out['svi_suff_' + k] = np.asarray(state['suff_blend'][k],
                                              dtype=np.float64)
    hist = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in rec.items()} for rec in state.get('history', [])]
    out['history'] = np.array(json.dumps(hist))
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f'.{os.path.basename(path)}.tmp.{os.getpid()}')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **out)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_checkpoint(path):
    d = np.load(path, allow_pickle=False)
    missing = [k for k in CONTRACT_KEYS if k not in d.files]
    if missing:
        raise ValueError(f'{path}: missing contract keys {missing}')
    ck = {k: d[k] for k in d.files}
    suff = None
    if all(('svi_suff_' + k) in d.files for k in SUFF_KEYS):
        suff = {k: d['svi_suff_' + k] for k in SUFF_KEYS}
        suff = {k: (float(v) if v.ndim == 0 else np.asarray(v)) for k, v in suff.items()}
    return ck, suff


def main(argv=None):
    args = parse_args(argv)

    import jax
    jax.config.update('jax_enable_x64', True)
    if args.jax_cache_dir:
        os.makedirs(os.path.expanduser(args.jax_cache_dir), exist_ok=True)
        jax.config.update('jax_compilation_cache_dir',
                          os.path.expanduser(args.jax_cache_dir))
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.5)

    log_fh = open(args.log_file, 'a') if args.log_file else None
    t_start = time.time()

    def log(msg):
        line = f'[{time.time() - t_start:9.1f}s] {msg}'
        print(line, flush=True)
        if log_fh:
            log_fh.write(line + '\n')
            log_fh.flush()

    from tkfmixdom.jax.train.mixfrag_svi_bw import svi_bw_mixfrag_s, lg_prior_S_pi

    F = int(args.n_fragtypes)
    log(f'MixFrag-S svi-BW: F={F}, dp_mode={args.dp_mode}, '
        f'backend={jax.default_backend()}, devices={jax.devices()}')
    log(f'  args: {vars(args)}')

    max_len = args.max_len
    if max_len < 0:
        max_len = 0 if args.dp_mode == 'path' else 1024
    chunk_max = args.chunk_max
    if chunk_max < 0:
        chunk_max = 1024 if args.dp_mode == 'path' else 256
    pairs = load_pairs_for_mode(args.precompiled_pairs, args.dp_mode, max_len, log)
    pairs = subset_pairs(pairs, args.n_pairs_subset, args.subset_seed)
    if args.n_pairs_subset:
        log(f'  subset: {len(pairs)} pairs (seed {args.subset_seed})')
    n_total = int(args.n_total_pairs) if args.n_total_pairs else len(pairs)

    val_pairs = None
    if args.val_precompiled_pairs and args.val_every:
        val_pairs = load_pairs_for_mode(args.val_precompiled_pairs, args.dp_mode,
                                        max_len, log)
        val_pairs = subset_pairs(val_pairs, args.val_n_pairs, args.subset_seed + 1)
        log(f'  val: {len(val_pairs)} pairs every {args.val_every} iterations')

    lam, mu, exts, weights, init_S, init_pi = load_init(args, F, log)
    start_iter = 0
    suff_init = None
    if args.resume and os.path.exists(args.checkpoint):
        ck, suff_init = load_checkpoint(args.checkpoint)
        if int(ck['n_fragtypes']) != F:
            raise SystemExit(f'checkpoint F={int(ck["n_fragtypes"])} != --n-fragtypes {F}')
        # The blended sufficient statistics are E-step-specific: a checkpoint
        # trained under one --dp-mode must not be continued under another
        # (round-1 checkpoints carry no 'dp_mode' and were unaligned 2D).
        if '_config' in ck:
            ck_mode = json.loads(str(ck['_config'])).get('dp_mode', 'unaligned')
            if ck_mode != args.dp_mode:
                raise SystemExit(
                    f'checkpoint {args.checkpoint} was trained with --dp-mode '
                    f'{ck_mode}; resuming under --dp-mode {args.dp_mode} would '
                    f'blend sufficient statistics from a different E-step '
                    f'(use --dp-mode {ck_mode}, or start fresh without --resume)')
        lam, mu = float(ck['ins_rate']), float(ck['del_rate'])
        exts, weights = np.asarray(ck['exts']), np.asarray(ck['weights'])
        init_S, init_pi = np.asarray(ck['S_exch']), np.asarray(ck['pi'])
        start_iter = int(ck['em_iter'])
        log(f'Resuming from {args.checkpoint} at em_iter={start_iter} '
            f'(blended suff stats {"restored" if suff_init else "absent -> re-initialised"})')
        if start_iter >= args.n_iter:
            log('  nothing to do (em_iter >= --n-iter).')
            return 0
    elif args.resume:
        log(f'--resume: no checkpoint at {args.checkpoint}; starting fresh.')

    S_lg, pi_lg = lg_prior_S_pi()
    config = dict(vars(args))
    config.update({'n_total_pairs': n_total, 'n_pairs_loaded': len(pairs),
                   'max_len_effective': max_len, 'chunk_max_effective': chunk_max,
                   'dp_mode': args.dp_mode,
                   'model': 'MixFrag-S', 'A': int(init_pi.shape[-1])})

    def ckpt_fn(state):
        save_checkpoint_atomic(args.checkpoint, state, config, F)
        log(f'  [checkpoint em_iter={state["em_iter"]} -> {args.checkpoint}]')

    out = svi_bw_mixfrag_s(
        pairs, n_total_pairs=n_total, n_fragtypes=F,
        init_lam=lam, init_mu=mu, init_exts=exts, init_weights=weights,
        init_S=init_S, init_pi=init_pi,
        n_iter=args.n_iter, batch_size=args.svi_batch,
        svi_tau=args.svi_tau, svi_kappa=args.svi_kappa,
        prior_alpha_lam=args.prior_alpha_lam, prior_alpha_mu=args.prior_alpha_mu,
        prior_beta=args.prior_beta,
        ext_prior_alpha=args.ext_prior_alpha, ext_prior_beta=args.ext_prior_beta,
        weight_prior_alpha=args.weight_prior_alpha,
        S_prior=S_lg, pi_prior=pi_lg, pi_pseudo=args.pi_pseudo, S_pseudo=args.S_pseudo,
        full_batch=args.full_batch,
        start_iter=start_iter, suff_blend_init=suff_init,
        chunk_max=chunk_max, mem_bytes=args.chunk_mem_gb * 1e9,
        hr_chunk=args.hr_chunk, bin_stride=args.bin_stride,
        dp_mode=args.dp_mode, fb_scan_mode=args.fb_scan_mode,
        checkpoint_fn=ckpt_fn, ckpt_every=args.ckpt_every,
        val_pairs=val_pairs, val_every=args.val_every,
        log_fn=log, log_every=args.log_every, seed=args.seed)
    log(f'MixFrag-S svi-BW complete: em_iter={out["em_iter"]} lam={out["lam"]:.6f} '
        f'mu={out["mu"]:.6f} exts={out["exts"]} weights={out["weights"]}')
    if log_fh:
        log_fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
