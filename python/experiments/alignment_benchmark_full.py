#!/usr/bin/env python3
"""Comprehensive alignment benchmark on BAliBASE3 (120 families) and OXBench (395 families).

Methods:
  1. d3f1 FSA  — MixDom pair HMM (svi_bw_d3f1_full_best_val.npz) + sequence annealing
  2. TKF92 FSA — TKF92 pair HMM (ins=0.046, del=0.054, ext=0.68) + LG08 + sequence annealing
  3. MAFFT     — mafft --auto
  4. MUSCLE    — ~/bin/muscle -align

Usage:
    cd python && JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=1 uv run python experiments/alignment_benchmark_full.py
"""

import os
os.environ.setdefault('JAX_ENABLE_X64', '1')
# Persistent JAX compilation cache: survives across benchmark restarts so
# we don't pay 5–30s recompile per (sequence-length-bin × model) on cold
# start. Cache lives in pfam/jax_cache/.
_JAX_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'pfam', 'jax_cache')
os.makedirs(_JAX_CACHE, exist_ok=True)
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', _JAX_CACHE)
# Disable XLA command buffers: when many unique (Lx_pad, Ly_pad, chunk_size)
# buckets accumulate in the JIT cache (e.g. 24+ graphs alive), the CUDA
# command buffer instantiation runs out of GPU memory and aborts specific
# families. Command buffers are a perf optimization that we don't need.
os.environ.setdefault('XLA_FLAGS',
                      '--xla_gpu_enable_command_buffer=')

import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = Path(__file__).parent.parent
BENCH_ROOT = Path("~/bio-datasets/data").expanduser()
BALIBASE_DIR = BENCH_ROOT / "balibase" / "bali3pdbm"
OXBENCH_DIR = Path("~/bio-datasets/data/oxbench/ox").expanduser()
ALIGN_OUT = Path(__file__).parent / "alignments"
RESULTS_PATH = Path(os.environ.get(
    'BENCH_RESULTS_FILE',
    str(Path(__file__).parent / "alignment_benchmark_full.json")))
# MixDom checkpoints: specify one or more via env vars. Each MIXDOM_<N>
# entry defines a checkpoint path and result key. The benchmark loads all
# of them at startup and runs FSA alignment for each.
#
# Usage:
#   # Single model (replaces old D3F1 behavior):
#   MIXDOM_1=pfam/svi_bw_d3f1_full_best_val.npz:d3f1
#
#   # Multiple models:
#   MIXDOM_1=pfam/svi_bw_d3f1_full_best_val.npz:d3f1
#   MIXDOM_2=pfam/svi_bw_d5f1_full_best_val.npz:d5f1
#   MIXDOM_3=pfam/svi_bw_d10f1_full_best_val.npz:d10f1
#
# Format: MIXDOM_<N>=<checkpoint_path>:<result_key>
#
# Defaults (when no MIXDOM_* env vars are set):
_MIXDOM_DEFAULTS = [
    ('pfam/svi_bw_d3f1_full_best_val.npz', 'd3f1'),
    ('pfam/svi_bw_d5f1_full_best_val.npz', 'd5f1'),
]

def _parse_mixdom_env():
    """Parse MIXDOM_1, MIXDOM_2, ... env vars. Returns [(path, key), ...]."""
    entries = []
    for k, v in sorted(os.environ.items()):
        if k.startswith('MIXDOM_') and k[7:].isdigit():
            parts = v.split(':', 1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
            else:
                entries.append((parts[0], Path(parts[0]).stem))
    return entries if entries else _MIXDOM_DEFAULTS

MIXDOM_MODELS = _parse_mixdom_env()  # [(checkpoint_path, result_key), ...]
MIXDOM_KEYS = {key for _, key in MIXDOM_MODELS}  # set of all MixDom result keys

sys.path.insert(0, str(PYTHON_ROOT))
sys.path.insert(0, str(BENCH_ROOT))

from tkfmixdom.util.msa_benchmark import parse_fasta, sp_tc_score

# ── Amino acid encoding ──────────────────────────────────────────────
AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"
AA_MAP = {c: i for i, c in enumerate(AA_CHARS)}


def encode_seq(seq_str):
    """Convert amino acid string to integer array. Unknowns -> 20."""
    return np.array([AA_MAP.get(c, 20) for c in seq_str.upper() if c not in '.-~'],
                    dtype=np.int32)


def msa_ints_to_strings(msa_dict):
    """Convert integer MSA dict {name: int_array} to {name: string}."""
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


def write_fasta(seqs, filepath):
    """Write sequences dict to FASTA."""
    with open(filepath, 'w') as f:
        for name, seq in seqs.items():
            f.write(f'>{name}\n{seq}\n')


# ── Lazy-loaded JAX models ───────────────────────────────────────────
_jax_loaded = False
_mixdom_loaded = {}  # key → (fsa_params, n_dom, n_frag)
_Q_lg = None
_pi_lg = None


def _load_mixdom_params(npz_path, label):
    """Load a MixDom param npz and return (params_dict, n_dom, n_frag)."""
    import jax.numpy as jnp
    from tkfmixdom.jax.distill.maraschino import load_params, build_rate_matrix

    print(f"Loading {label} params from {npz_path}")
    params, n_dom, n_cls = load_params(str(npz_path))
    v = np.asarray(params['v'])
    pis = np.asarray(params['pi'])
    S_exch = np.asarray(params['S_exch'])
    avg_pi = np.einsum('n,na->a', v, pis)
    avg_pi = avg_pi / avg_pi.sum()
    avg_S = np.einsum('n,nab->ab', v, S_exch)
    fsa = {
        'main_ins': float(params['lam0']),
        'main_del': float(params['mu0']),
        'dom_ins': np.asarray(params['lam']),
        'dom_del': np.asarray(params['mu']),
        'dom_weights': np.asarray(params['v']),
        'frag_weights': np.asarray(params['frag_weights']),
        'ext_rates': np.asarray(params['r_frags']),
        'S_exch': np.asarray(params['S_exch']),
        'pi': avg_pi,
        'Q': np.asarray(build_rate_matrix(jnp.array(avg_S), jnp.array(avg_pi))),
    }
    fw = fsa['frag_weights']
    n_frag = fw.shape[1] if fw.ndim > 1 else 1
    print(f"  {label}: n_dom={n_dom}, n_frag={n_frag}")
    return fsa, n_dom, n_frag


def _ensure_jax():
    global _jax_loaded, _Q_lg, _pi_lg
    if _jax_loaded:
        return
    from tkfmixdom.jax.core.protein import rate_matrix_lg

    for ckpt_path, key in MIXDOM_MODELS:
        full_path = PYTHON_ROOT / ckpt_path
        if os.path.exists(str(full_path)):
            params, n_dom, n_frag = _load_mixdom_params(full_path, key)
            _mixdom_loaded[key] = (params, n_dom, n_frag)

    # LG08 for TKF92
    Q, pi = rate_matrix_lg()
    _Q_lg = np.asarray(Q)
    _pi_lg = np.asarray(pi)
    print("LG08 loaded")
    _jax_loaded = True


# Per-call timing accumulators (last-call snapshot, read by process_family)
_LAST_TIMINGS = {'t_pairs': 0.0, 't_anneal': 0.0, 'n_pairs': 0}


# ── FSA alignment (d3f1 or TKF92) ────────────────────────────────────
def _bucket_pairs_by_padded_shape(pairs, int_seqs, names):
    """Group pairs by (Lx_pad, Ly_pad). Returns a dict
    {(Lx_pad, Ly_pad): [(pair_idx_in_original_list, (i, j), x_pad, y_pad), ...]}.
    x_pad and y_pad are numpy int32 arrays of length Lx_pad/Ly_pad,
    zero-padded from the original sequence.
    """
    from tkfmixdom.jax.dp.hmm import _pad_to_bin
    buckets = {}
    for idx, (i, j) in enumerate(pairs):
        xi = int_seqs[names[i]]
        yj = int_seqs[names[j]]
        Lx_pad = _pad_to_bin(len(xi))
        Ly_pad = _pad_to_bin(len(yj))
        x_pad = np.zeros(Lx_pad, dtype=np.int32)
        x_pad[:len(xi)] = xi
        y_pad = np.zeros(Ly_pad, dtype=np.int32)
        y_pad[:len(yj)] = yj
        buckets.setdefault((Lx_pad, Ly_pad), []).append(
            (idx, (i, j), len(xi), len(yj), x_pad, y_pad))
    return buckets


# Cap per-dispatch batch footprint. The FB posterior tensor is the
# dominant allocation at shape (B, Lx+1, Ly+1, n_states); F, B, emit
# are comparable. We also run into cuBLAS autotuner failures when the
# reduced sufficient-statistic einsums produce huge intermediates
# under vmap. Target ~400 MB per dispatch (cells of f64).
_BATCH_CELL_BUDGET = int(400e6 // 8)  # ~50M cells of f64


def _max_batch_size_for_shape(Lx_pad, Ly_pad, n_states):
    per_pair_cells = (Lx_pad + 1) * (Ly_pad + 1) * n_states * 4  # F + B + emit + posteriors
    return max(1, _BATCH_CELL_BUDGET // max(1, per_pair_cells))


def _iter_bucket_chunks(bucket, chunk_size):
    for start in range(0, len(bucket), chunk_size):
        yield bucket[start:start + chunk_size]


def _is_oom_error(exc):
    """True if exception is GPU OOM / cuBLAS resource-exhaustion / kernel launch failure.

    Includes transient CUDA_ERROR_INVALID_VALUE from GPU state corruption
    after processing many families — these succeed on retry with fresh dispatch.
    """
    msg = str(exc)
    for needle in ('RESOURCE_EXHAUSTED', 'OUT_OF_MEMORY', 'CUBIN',
                   'Autotuner could not find', 'instantiate command buffer',
                   'CUDA error', 'CUDA_ERROR'):
        if needle in msg:
            return True
    return False


def _run_chunk_with_oom_fallback(call_fn, chunk):
    """Try call_fn(chunk). If it raises an OOM, retry one pair at a time.

    JAX is lazy: a returned device array materializes only on .block or
    np.asarray. To make this wrapper actually catch OOMs that happen at
    materialization time, we materialize to numpy INSIDE the try block.
    """
    def _materialize(call_fn, chunk):
        mps_b, taus_b, lps_b = call_fn(chunk)
        return (np.asarray(mps_b), np.asarray(taus_b), np.asarray(lps_b))
    try:
        return _materialize(call_fn, chunk)
    except Exception as e:
        if not _is_oom_error(e):
            raise
        print(f"  [OOM fallback] chunk of {len(chunk)} pairs OOMed; "
              f"retrying one pair at a time...", flush=True)
        all_mps, all_taus, all_lps = [], [], []
        for entry in chunk:
            try:
                sub_mps, sub_taus, sub_lps = _materialize(call_fn, [entry])
            except Exception as e2:
                if not _is_oom_error(e2):
                    raise
                # GPU singleton OOM — retry on CPU
                print(f"  [OOM fallback] singleton pair (Lx={entry[2]}, "
                      f"Ly={entry[3]}) OOMed on GPU; retrying on CPU...",
                      flush=True)
                try:
                    import jax
                    cpu = jax.devices('cpu')[0]
                    cpu_entry = list(entry)
                    cpu_entry[4] = jax.device_put(
                        np.asarray(entry[4]), cpu)
                    cpu_entry[5] = jax.device_put(
                        np.asarray(entry[5]), cpu)
                    with jax.default_device(cpu):
                        sub_mps, sub_taus, sub_lps = _materialize(
                            call_fn, [tuple(cpu_entry)])
                except Exception as e3:
                    print(f"  [OOM fallback] CPU fallback also failed: "
                          f"{e3}; skipping with NaN", flush=True)
                    Lx_pad = entry[4].shape[0]
                    Ly_pad = entry[5].shape[0]
                    sub_mps = np.zeros((1, Lx_pad, Ly_pad), dtype=np.float32)
                    sub_taus = np.array([1.0], dtype=np.float64)
                    sub_lps = np.array([0.0], dtype=np.float64)
            all_mps.append(sub_mps[0:1])
            all_taus.append(sub_taus[0:1])
            all_lps.append(sub_lps[0:1])
        return (np.concatenate(all_mps, axis=0),
                np.concatenate(all_taus, axis=0),
                np.concatenate(all_lps, axis=0))


def _align_fsa_mixdom(int_seqs, names, fsa_params, n_dom, n_frag, gap_factor=1.0,
                       all_pairs=False):
    """Shared MixDom FSA alignment core, parameterized by model params.

    Uses vmap-over-pairs: bucket pairs by padded (Lx, Ly) shape, then
    batch each bucket through pairwise_posteriors_mixdom_batched in a
    single vmap'd dispatch. Cuts per-pair Python+sync overhead.

    gap_factor: TGF gap factor for sequence annealing (1.0 = standard AMA,
        0.0 = maximally aggressive merging, >1.0 = conservative/more gaps).
    all_pairs: if True, use all N(N-1)/2 pairs instead of Erdos-Renyi.
    """
    import jax.numpy as jnp
    from tkfmixdom.jax.tree.fsa_anneal import (
        pairwise_posteriors_mixdom_batched,
        select_pairs_full, select_pairs_erdos_renyi,
        sequence_annealing,
    )
    n_seqs = len(names)
    if all_pairs:
        pairs = select_pairs_full(n_seqs)
    else:
        pairs = select_pairs_full(n_seqs) if n_seqs <= 20 else select_pairs_erdos_renyi(n_seqs)
    pair_posteriors = {}
    t_pairs0 = time.time()

    buckets = _bucket_pairs_by_padded_shape(pairs, int_seqs, names)
    n_states = 2 + 5 * n_dom * n_frag

    def _call(chunk):
        xs = np.stack([entry[4] for entry in chunk])
        ys = np.stack([entry[5] for entry in chunk])
        real_Lxs = np.array([entry[2] for entry in chunk], dtype=np.int32)
        real_Lys = np.array([entry[3] for entry in chunk], dtype=np.int32)
        return pairwise_posteriors_mixdom_batched(
            jnp.asarray(xs), jnp.asarray(ys),
            jnp.asarray(real_Lxs), jnp.asarray(real_Lys),
            fsa_params, n_dom, n_frag)

    for (Lx_pad, Ly_pad), bucket in buckets.items():
        chunk_size = _max_batch_size_for_shape(Lx_pad, Ly_pad, n_states)
        for chunk in _iter_bucket_chunks(bucket, chunk_size):
            mps_b, _, _ = _run_chunk_with_oom_fallback(_call, chunk)
            for b, (idx, (i, j), Lx_real, Ly_real, _, _) in enumerate(chunk):
                pair_posteriors[(i, j)] = mps_b[b, :Lx_real, :Ly_real]

    _LAST_TIMINGS['t_pairs'] = time.time() - t_pairs0
    _LAST_TIMINGS['n_pairs'] = len(pairs)

    seq_lengths = [len(int_seqs[n]) for n in names]
    t_a0 = time.time()
    col_assignments, msa_length = sequence_annealing(
        n_seqs, seq_lengths, pair_posteriors, n_iterations=3, verbose=False,
        gap_factor=gap_factor)
    _LAST_TIMINGS['t_anneal'] = time.time() - t_a0
    n_cols = max(max(ca) for ca in col_assignments if len(ca) > 0) + 1
    msa_dict = {}
    for si, name in enumerate(names):
        row = np.full(n_cols, -1, dtype=np.int32)
        seq = int_seqs[name]
        for k in range(len(seq)):
            row[col_assignments[si][k]] = seq[k]
        msa_dict[name] = row
    return msa_ints_to_strings(msa_dict)


# Global config: set via env var BENCH_ALL_PAIRS=1 to force all N(N-1)/2 pairs
_ALL_PAIRS = os.environ.get('BENCH_ALL_PAIRS', '') == '1'


def align_fsa_mixdom(int_seqs, names, key, gap_factor=1.0):
    """FSA alignment using a loaded MixDom model identified by result key."""
    params, n_dom, n_frag = _mixdom_loaded[key]
    return _align_fsa_mixdom(int_seqs, names, params, n_dom, n_frag,
                             gap_factor=gap_factor, all_pairs=_ALL_PAIRS)


def align_fsa_tkf92(int_seqs, names):
    """Align using TKF92 pair HMM + LG08 + sequence annealing (vmap'd)."""
    import jax.numpy as jnp
    from tkfmixdom.jax.tree.fsa_anneal import (
        pairwise_posteriors_tkf92_batched,
        select_pairs_full, select_pairs_erdos_renyi,
        sequence_annealing,
    )
    ins_rate, del_rate, ext = 0.046, 0.054, 0.68
    n_seqs = len(names)
    pairs = select_pairs_full(n_seqs) if n_seqs <= 20 else select_pairs_erdos_renyi(n_seqs)
    pair_posteriors = {}
    t_pairs0 = time.time()

    buckets = _bucket_pairs_by_padded_shape(pairs, int_seqs, names)
    n_states_tkf = 5  # TKF92 pair HMM has 5 states

    def _call(chunk):
        xs = np.stack([entry[4] for entry in chunk])
        ys = np.stack([entry[5] for entry in chunk])
        real_Lxs = np.array([entry[2] for entry in chunk], dtype=np.int32)
        real_Lys = np.array([entry[3] for entry in chunk], dtype=np.int32)
        return pairwise_posteriors_tkf92_batched(
            jnp.asarray(xs), jnp.asarray(ys),
            jnp.asarray(real_Lxs), jnp.asarray(real_Lys),
            ins_rate, del_rate, ext, _Q_lg, _pi_lg)

    for (Lx_pad, Ly_pad), bucket in buckets.items():
        chunk_size = _max_batch_size_for_shape(Lx_pad, Ly_pad, n_states_tkf)
        for chunk in _iter_bucket_chunks(bucket, chunk_size):
            mps_b, _, _ = _run_chunk_with_oom_fallback(_call, chunk)
            for b, (idx, (i, j), Lx_real, Ly_real, _, _) in enumerate(chunk):
                pair_posteriors[(i, j)] = mps_b[b, :Lx_real, :Ly_real]

    _LAST_TIMINGS['t_pairs'] = time.time() - t_pairs0
    _LAST_TIMINGS['n_pairs'] = len(pairs)

    seq_lengths = [len(int_seqs[n]) for n in names]
    t_a0 = time.time()
    col_assignments, msa_length = sequence_annealing(
        n_seqs, seq_lengths, pair_posteriors, n_iterations=3, verbose=False)
    _LAST_TIMINGS['t_anneal'] = time.time() - t_a0
    n_cols = max(max(ca) for ca in col_assignments if len(ca) > 0) + 1
    msa_dict = {}
    for si, name in enumerate(names):
        row = np.full(n_cols, -1, dtype=np.int32)
        seq = int_seqs[name]
        for k in range(len(seq)):
            row[col_assignments[si][k]] = seq[k]
        msa_dict[name] = row
    return msa_ints_to_strings(msa_dict)


# ── External tools ────────────────────────────────────────────────────
def align_mafft(in_path):
    """Run MAFFT --auto and return aligned seqs dict."""
    with tempfile.NamedTemporaryFile(suffix='.fa', delete=False) as tmp:
        tmp_out = tmp.name
    try:
        result = subprocess.run(
            f"mafft --auto --quiet {in_path}",
            shell=True, capture_output=True, timeout=300)
        if result.returncode != 0:
            return None
        with open(tmp_out, 'wb') as f:
            f.write(result.stdout)
        return parse_fasta(tmp_out)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)


def align_muscle(in_path):
    """Run MUSCLE5 and return aligned seqs dict."""
    with tempfile.NamedTemporaryFile(suffix='.fa', delete=False) as tmp:
        tmp_out = tmp.name
    try:
        result = subprocess.run(
            [os.path.expanduser('~/bin/muscle'), '-align', in_path, '-output', tmp_out],
            capture_output=True, timeout=300)
        if result.returncode != 0:
            return None
        return parse_fasta(tmp_out)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)


# ── Per-family processing ────────────────────────────────────────────
def process_family(family, in_dir, ref_dir, method, align_out_dir):
    """Process one family with one method. Returns result dict or None."""
    in_path = str(in_dir / family)
    ref_path = str(ref_dir / family)
    if not os.path.exists(in_path) or not os.path.exists(ref_path):
        return {'family': family, 'sp': 0.0, 'tc': 0.0, 'time': 0.0, 'error': 'missing_files'}

    raw_seqs = parse_fasta(in_path)
    if len(raw_seqs) < 2:
        return {'family': family, 'sp': 0.0, 'tc': 0.0, 'time': 0.0, 'error': 'too_few_seqs'}

    ref_aln = parse_fasta(ref_path)

    t0 = time.time()
    aligned = None
    error = None

    try:
        if method == 'mafft':
            aligned = align_mafft(in_path)
        elif method == 'muscle':
            aligned = align_muscle(in_path)
        elif method in MIXDOM_KEYS or method == 'tkf92':
            # Encode sequences
            int_seqs = {}
            for name, seq in raw_seqs.items():
                enc = encode_seq(seq)
                if len(enc) > 0:
                    int_seqs[name] = enc
            if len(int_seqs) < 2:
                return {'family': family, 'sp': 0.0, 'tc': 0.0, 'time': 0.0,
                        'error': 'too_few_valid_seqs'}
            names = list(int_seqs.keys())
            if method in MIXDOM_KEYS:
                aligned = align_fsa_mixdom(int_seqs, names, method)
            else:
                aligned = align_fsa_tkf92(int_seqs, names)
    except Exception as e:
        error = str(e)
        traceback.print_exc()

    elapsed = time.time() - t0

    if aligned is None:
        return {'family': family, 'sp': 0.0, 'tc': 0.0, 'time': elapsed,
                'error': error or 'alignment_failed'}

    # Save alignment
    out_path = align_out_dir / f"{family}.fa"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(aligned, str(out_path))

    # Score
    try:
        sp, tc = sp_tc_score(aligned, ref_aln, core_only=True)
    except Exception as e:
        sp, tc = 0.0, 0.0
        error = f"scoring_error: {e}"

    result = {'family': family, 'sp': float(sp), 'tc': float(tc), 'time': float(elapsed)}
    if error:
        result['error'] = error
    if method in MIXDOM_KEYS or method == 'tkf92':
        result['t_pairs'] = float(_LAST_TIMINGS.get('t_pairs', 0.0))
        result['t_anneal'] = float(_LAST_TIMINGS.get('t_anneal', 0.0))
        result['n_pairs'] = int(_LAST_TIMINGS.get('n_pairs', 0))
        result['t_other'] = float(elapsed - result['t_pairs'] - result['t_anneal'])
    return result


# ── Save/load intermediate results ───────────────────────────────────
def save_results(results, path):
    """Save results dict to JSON."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)


def load_results(path):
    """Load results dict from JSON if it exists."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Summary printing ─────────────────────────────────────────────────
def print_summary(results, benchmark_name):
    """Print summary table for a benchmark."""
    print(f"\n{'='*70}")
    print(f"  {benchmark_name} Summary")
    print(f"{'='*70}")
    methods = sorted(results.keys())
    header = f"{'Method':<12} {'N':>4} {'SP mean':>8} {'TC mean':>8} {'SP med':>8} {'TC med':>8}"
    print(header)
    print('-' * len(header))
    for method in methods:
        fam_results = results[method]
        if not fam_results:
            continue
        sps = [r['sp'] for r in fam_results]
        tcs = [r['tc'] for r in fam_results]
        print(f"{method:<12} {len(fam_results):>4} "
              f"{np.mean(sps):>8.3f} {np.mean(tcs):>8.3f} "
              f"{np.median(sps):>8.3f} {np.median(tcs):>8.3f}")


def print_balibase_subsets(results):
    """Print per-subset summary for BAliBASE."""
    subsets = {'RV11': 'BB11', 'RV12': 'BB12', 'RV20': 'BB20',
               'RV30': 'BB30', 'RV40': 'BB40', 'RV50': 'BB50'}
    methods = sorted(results.keys())

    print(f"\n{'='*70}")
    print(f"  BAliBASE Per-Subset SP Scores")
    print(f"{'='*70}")
    header = f"{'Subset':<8} " + " ".join(f"{m:>10}" for m in methods) + f" {'N':>4}"
    print(header)
    print('-' * len(header))

    for subset_name, prefix in sorted(subsets.items()):
        row = f"{subset_name:<8} "
        n = 0
        for method in methods:
            fam_results = results[method]
            subset_results = [r for r in fam_results if r['family'].startswith(prefix)]
            if subset_results:
                n = len(subset_results)
                sp_mean = np.mean([r['sp'] for r in subset_results])
                row += f"{sp_mean:>10.3f} "
            else:
                row += f"{'---':>10} "
        row += f"{n:>4}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    # Initialize output structure
    all_results = load_results(str(RESULTS_PATH)) or {
        'balibase': {}, 'oxbench': {}
    }

    benchmarks = [
        ('balibase', BALIBASE_DIR, 'BAliBASE3'),
        ('oxbench', OXBENCH_DIR, 'OXBench'),
    ]

    # Methods in order: fast external tools first, then FSA methods
    methods_order = ['mafft', 'muscle', 'tkf92'] + [key for _, key in MIXDOM_MODELS]

    # Optional restriction by env var, e.g. BENCH_DATASETS=balibase
    # and BENCH_METHODS=d5f1
    _ds_filter = os.environ.get('BENCH_DATASETS', '').strip()
    if _ds_filter:
        _keep = {s.strip() for s in _ds_filter.split(',') if s.strip()}
        benchmarks = [b for b in benchmarks if b[0] in _keep]
        print(f"BENCH_DATASETS filter: {_keep}")
    _m_filter = os.environ.get('BENCH_METHODS', '').strip()
    if _m_filter:
        _keep_m = {s.strip() for s in _m_filter.split(',') if s.strip()}
        methods_order = [m for m in methods_order if m in _keep_m]
        print(f"BENCH_METHODS filter: {_keep_m}")

    for bench_key, bench_dir, bench_label in benchmarks:
        in_dir = bench_dir / "in"
        ref_dir = bench_dir / "ref"
        families = sorted([f for f in os.listdir(str(in_dir))
                          if os.path.exists(str(ref_dir / f))])
        print(f"\n{'#'*70}")
        print(f"  {bench_label}: {len(families)} families")
        print(f"{'#'*70}")

        if bench_key not in all_results:
            all_results[bench_key] = {}

        for method in methods_order:
            # Check if already done (resume support)
            existing = all_results[bench_key].get(method, [])
            done_families = {r['family'] for r in existing}
            remaining = [f for f in families if f not in done_families]

            if not remaining:
                print(f"\n  [{method}] All {len(families)} families already done, skipping")
                continue

            # Load JAX if needed
            if (method in MIXDOM_KEYS or method == 'tkf92') and not _jax_loaded:
                _ensure_jax()

            align_out = ALIGN_OUT / bench_key / method
            align_out.mkdir(parents=True, exist_ok=True)

            print(f"\n  [{method}] Processing {len(remaining)} families "
                  f"({len(done_families)} already done)")

            results_list = list(existing)  # start from existing
            batch_start = time.time()

            for fi, family in enumerate(remaining):
                result = process_family(
                    family, in_dir, ref_dir, method, align_out)
                results_list.append(result)

                # Progress
                err_str = f" ERROR={result.get('error','')}" if 'error' in result else ''
                tb = ''
                if 't_pairs' in result:
                    tb = (f" [{result['n_pairs']}p, "
                          f"pairs={result['t_pairs']:.1f}s "
                          f"anneal={result['t_anneal']:.1f}s "
                          f"other={result['t_other']:.1f}s]")
                print(f"  [{method}] [{fi+1}/{len(remaining)}] {family}: "
                      f"SP={result['sp']:.3f} TC={result['tc']:.3f} "
                      f"t={result['time']:.1f}s{tb}{err_str}")

                # Save every 10 families
                if (fi + 1) % 10 == 0:
                    all_results[bench_key][method] = results_list
                    save_results(all_results, str(RESULTS_PATH))

            all_results[bench_key][method] = results_list
            save_results(all_results, str(RESULTS_PATH))

            elapsed = time.time() - batch_start
            sps = [r['sp'] for r in results_list]
            print(f"  [{method}] Done: mean SP={np.mean(sps):.3f}, "
                  f"total time={elapsed:.0f}s")

        # Print summary for this benchmark
        print_summary(all_results[bench_key], bench_label)
        if bench_key == 'balibase':
            print_balibase_subsets(all_results[bench_key])

    # Final save
    save_results(all_results, str(RESULTS_PATH))
    print(f"\nResults saved to {RESULTS_PATH}")

    # Print overall summary
    for bench_key, _, bench_label in benchmarks:
        if bench_key in all_results and all_results[bench_key]:
            print_summary(all_results[bench_key], bench_label)
            if bench_key == 'balibase':
                print_balibase_subsets(all_results[bench_key])


if __name__ == '__main__':
    main()
