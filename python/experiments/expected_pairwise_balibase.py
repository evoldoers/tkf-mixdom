#!/usr/bin/env python3
"""Compute expected pairwise sufficient stats on BAliBASE.

Bypasses FSA reconstruction; computes the soft confusion-matrix
sufficient statistics directly from per-pair match posteriors and
reports ``e_tp / total_mass / gold / n_cells`` per pair, per family,
and pooled across the corpus. Every other quantity (E[FP], E[FN],
E[TN], precision, recall, F1) is derivable via the identities in
``tkfmixdom.util.expected_pair_f1``.

Supported methods (``--method``):

    tkf92             single TKF92, params from JSON
    tkf92_mixture     K-TKF92 mixture, anchor JSON + mixture NPZ
    cherryml_mixture  K-CherryML mixture, anchor JSON + cherryml NPZ.
                      The gap state (index 20) of the 21-state CherryML
                      model is dropped and pi renormalised over the 20
                      protein AAs; the anchor TKF92 indel rates are
                      shared across mixture components.
    mixdom            MixDom, NPZ loaded via ``maraschino.load_params``
                      + ``make_fsa_params``.
    mafft             External MAFFT (default invocation). The hard MSA
                      is projected to pairwise match indicators (0/1) and
                      fed through the same expected-F1 pipeline.
    muscle            External MUSCLE v5/v3. Same projection as MAFFT.

Default: GPU. Override with ``JAX_PLATFORMS=cpu`` or unset CUDA env vars.

Examples:
    cd python && JAX_ENABLE_X64=1 uv run python \\
        experiments/expected_pairwise_balibase.py \\
        --method tkf92 \\
        --method-name tkf92_lg08 \\
        --params experiments/tkf92_fitted_params.json \\
        --out experiments/expected_balibase_tkf92.json

    uv run python experiments/expected_pairwise_balibase.py \\
        --method tkf92_mixture \\
        --method-name tkf92_K20 \\
        --anchor experiments/tkf92_fitted_params.json \\
        --params pfam/tkf92_mixture_K20_train.npz \\
        --out experiments/expected_balibase_tkf92_K20.json

    uv run python experiments/expected_pairwise_balibase.py \\
        --method cherryml_mixture \\
        --method-name cherryml_C20 \\
        --anchor experiments/tkf92_fitted_params.json \\
        --params pfam/cherryml_mixture_C20_n5000.npz \\
        --out experiments/expected_balibase_cherryml_C20.json

    uv run python experiments/expected_pairwise_balibase.py \\
        --method mixdom \\
        --method-name mixdom_d3f1 \\
        --params pfam/svi_bw_d3f1_postfix_best_val.npz \\
        --out experiments/expected_balibase_mixdom_d3f1.json
"""

import os
os.environ.setdefault('JAX_ENABLE_X64', '1')

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from tkfmixdom.jax.tree.fsa_anneal import (
    _pairwise_posteriors_tkf92_jax,
    pairwise_posteriors_mixdom,
    pairwise_posteriors_tkf92_mixture_streaming,
    select_pairs_full,
    sequence_annealing,
)
from tkfmixdom.jax.dp.hmm import _pad_to_bin, _pad_seq
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.util.msa_benchmark import parse_fasta, sp_tc_score
from tkfmixdom.util.expected_pair_f1 import (
    expected_family_f1, aggregate_corpus, ref_to_pair_truth)
from tkfmixdom.util import balibase_pair_cache as ppcache

from fsa_mixdom_pairhmm_oxbench import (
    encode_seq, make_fsa_params, build_msa_from_posteriors,
    msa_to_aligned_strings, AA_CHARS)
from fsa_tkf92_mixture_balibase import load_mixture_ckpt
from tkfmixdom.jax.distill.maraschino import load_params as load_mixdom_params


# ---------------------------------------------------------------------------
# Per-pair posterior backends, one per method.
# ---------------------------------------------------------------------------


def _tkf92_pair_posteriors(int_seqs, names, pairs, ins, del_, ext, Q, pi):
    """Per-pair match posteriors for a single TKF92 pair HMM (LG08).

    Each pair is wrapped in try/except: a single pair that OOMs or
    otherwise crashes is recorded in ``failed_pairs`` and skipped,
    rather than aborting the whole family.

    Returns:
        ``(pair_posteriors, failed_pairs)``.
    """
    import jax
    pp = {}
    failed = []
    for i, j in pairs:
        try:
            x = jnp.asarray(int_seqs[names[i]], dtype=jnp.int32)
            y = jnp.asarray(int_seqs[names[j]], dtype=jnp.int32)
            Lx, Ly = int(x.shape[0]), int(y.shape[0])
            Lx_pad, Ly_pad = _pad_to_bin(Lx), _pad_to_bin(Ly)
            x_pad, y_pad = _pad_seq(x, Lx_pad), _pad_seq(y, Ly_pad)
            mp_pad, _, _ = _pairwise_posteriors_tkf92_jax(
                x_pad, y_pad, jnp.int32(Lx), jnp.int32(Ly),
                jnp.float64(ins), jnp.float64(del_), jnp.float64(ext),
                jnp.asarray(Q), jnp.asarray(pi))
            pp[(i, j)] = np.asarray(mp_pad)[:Lx, :Ly]
        except Exception as e:
            jax.clear_caches()
            failed.append({
                'pair': [int(i), int(j)],
                'name_i': names[i],
                'name_j': names[j],
                'Lx': int(len(int_seqs[names[i]])),
                'Ly': int(len(int_seqs[names[j]])),
                'error_type': type(e).__name__,
                'error_msg': str(e)[:500],
            })
    return pp, failed


def _tkf92_mixture_pair_posteriors(int_seqs, names, pairs, anchor, mix,
                                     pair_chunk=None):
    """Mixture-of-K-TKF92 streaming posteriors with anchor τ.

    Whole-family compute (pair-wise OOM fallback inside
    pairwise_posteriors_tkf92_mixture_streaming via ``pair_chunk``);
    a family-level failure is propagated to the caller.
    Returns ``(pair_posteriors, failed_pairs=[])`` for API symmetry
    with ``_tkf92_pair_posteriors``.
    """
    n_pairs = len(pairs)
    Lx_max = max(len(int_seqs[names[i]]) for i, _ in pairs)
    Ly_max = max(len(int_seqs[names[j]]) for _, j in pairs)
    Lx_pad, Ly_pad = _pad_to_bin(Lx_max), _pad_to_bin(Ly_max)
    pairs_x = np.zeros((n_pairs, Lx_pad), dtype=np.int32)
    pairs_y = np.zeros((n_pairs, Ly_pad), dtype=np.int32)
    real_Lxs = np.zeros(n_pairs, dtype=np.int32)
    real_Lys = np.zeros(n_pairs, dtype=np.int32)
    for k, (i, j) in enumerate(pairs):
        x, y = int_seqs[names[i]], int_seqs[names[j]]
        pairs_x[k, :len(x)] = x
        pairs_y[k, :len(y)] = y
        real_Lxs[k] = len(x)
        real_Lys[k] = len(y)
    pairs_x = jnp.asarray(pairs_x)
    pairs_y = jnp.asarray(pairs_y)
    real_Lxs = jnp.asarray(real_Lxs)
    real_Lys = jnp.asarray(real_Lys)

    @jax.jit
    def _anchor_tau_one(x, y, rL, rR):
        _, tau_opt, _ = _pairwise_posteriors_tkf92_jax(
            x, y, rL, rR,
            jnp.float64(anchor['ins']), jnp.float64(anchor['del']),
            jnp.float64(anchor['ext']),
            jnp.asarray(anchor['Q']), jnp.asarray(anchor['pi']),
            n_newton=3, tau_init=1.0)
        return tau_opt
    # Honour pair_chunk for the anchor-tau vmap too — the NR loop here
    # runs autodiff (grad+hess) through the FB scan, which is memory-
    # hungry at large pads. Without chunking, this OOMs before the
    # streaming Q' even starts.
    tau_chunk = pair_chunk if pair_chunk else n_pairs
    tau_chunk = max(1, min(tau_chunk, n_pairs))
    tau_anchor_chunks = []
    for s in range(0, n_pairs, tau_chunk):
        e = min(s + tau_chunk, n_pairs)
        tau_anchor_chunks.append(jax.vmap(_anchor_tau_one)(
            pairs_x[s:e], pairs_y[s:e], real_Lxs[s:e], real_Lys[s:e]))
    tau_anchor = jnp.concatenate(tau_anchor_chunks, axis=0)

    # OOM-halving fallback: start at the user pair_chunk (or n_pairs),
    # halve on RESOURCE_EXHAUSTED, give up at chunk_try=1.
    import gc
    chunk_try = pair_chunk if pair_chunk else n_pairs
    chunk_try = max(1, min(chunk_try, n_pairs))
    mp_mix = None
    while True:
        try:
            mp_mix, _ = pairwise_posteriors_tkf92_mixture_streaming(
                pairs_x, pairs_y, real_Lxs, real_Lys, tau_anchor,
                jnp.asarray(mix['mix_ins']), jnp.asarray(mix['mix_del']),
                jnp.asarray(mix['mix_ext']),
                jnp.asarray(mix['mix_Q']), jnp.asarray(mix['mix_pi']),
                jnp.asarray(mix['mix_weights']),
                pair_chunk=chunk_try)
            break
        except jax.errors.JaxRuntimeError as e:
            msg = str(e)
            if 'RESOURCE_EXHAUSTED' not in msg and \
               'OUT_OF_MEMORY' not in msg:
                raise
            if chunk_try == 1:
                raise   # already at minimum -- propagate up to caller
            jax.clear_caches()
            gc.collect()
            new_chunk = max(1, chunk_try // 2)
            print(f"    [OOM] mixture pair_chunk={chunk_try} -> "
                  f"{new_chunk}", flush=True)
            chunk_try = new_chunk
    mp_mix_np = np.asarray(mp_mix)
    pp = {}
    for k, (i, j) in enumerate(pairs):
        rL, rR = int(real_Lxs[k]), int(real_Lys[k])
        pp[(i, j)] = mp_mix_np[k, :rL, :rR]
    return pp, []


def _load_cherryml_as_mixture(path, anchor):
    """Convert a CherryML mixture .npz into ``mix`` dict for the TKF92
    mixture pair-HMM path. Drops the gap state (index 20) and shares
    anchor indel rates across components.
    """
    d = np.load(path, allow_pickle=True)
    K = int(d['n_classes'])
    S21 = np.asarray(d['S'], dtype=np.float64)        # (K, 21, 21)
    pi21 = np.asarray(d['pi'], dtype=np.float64)      # (K, 21)
    w = np.asarray(d['weights'], dtype=np.float64)
    w = w / w.sum()
    # Drop gap-as-state and renormalise pi.
    S20 = S21[:, :20, :20]
    pi20 = pi21[:, :20]
    pi20 = pi20 / pi20.sum(axis=1, keepdims=True)
    # Build per-component Q from (S, pi).
    from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi
    mix_Q = np.zeros_like(S20)
    for k in range(K):
        mix_Q[k] = np.asarray(build_Q_from_S_pi(jnp.asarray(S20[k]),
                                                jnp.asarray(pi20[k])))
    # Shared indel rates from anchor.
    mix_ins = np.full(K, float(anchor['ins']), dtype=np.float64)
    mix_del = np.full(K, float(anchor['del']), dtype=np.float64)
    mix_ext = np.full(K, float(anchor['ext']), dtype=np.float64)
    return dict(K=K, A=20, mix_weights=w,
                mix_ins=mix_ins, mix_del=mix_del, mix_ext=mix_ext,
                mix_Q=mix_Q, mix_pi=pi20)


def _mixdom_pair_posteriors(int_seqs, names, pairs, fsa_params, n_dom, n_frag):
    """Per-pair MixDom posteriors via the canonical pairwise_posteriors_mixdom.

    Per-pair try/except: a single OOMing pair is recorded in
    ``failed_pairs`` and skipped.

    Returns:
        ``(pair_posteriors, failed_pairs)``.
    """
    import jax
    pp = {}
    failed = []
    for i, j in pairs:
        try:
            x = jnp.asarray(int_seqs[names[i]], dtype=jnp.int32)
            y = jnp.asarray(int_seqs[names[j]], dtype=jnp.int32)
            mp, _, _ = pairwise_posteriors_mixdom(
                x, y, fsa_params, n_dom, n_frag)
            pp[(i, j)] = np.asarray(mp)
        except Exception as e:
            jax.clear_caches()
            failed.append({
                'pair': [int(i), int(j)],
                'name_i': names[i],
                'name_j': names[j],
                'Lx': int(len(int_seqs[names[i]])),
                'Ly': int(len(int_seqs[names[j]])),
                'error_type': type(e).__name__,
                'error_msg': str(e)[:500],
            })
    return pp, failed


def _optimal_accuracy_indicator(posterior: np.ndarray) -> np.ndarray:
    """Holmes-Durbin 1998 optimal-accuracy alignment from match posteriors.

    Given a soft posterior ``P[ri, rj]`` over residue-pair matches,
    return the ``(Li, Lj)`` 0/1 indicator of the monotone-increasing
    matching that maximises Σ P[ri, rj] over its match cells -- i.e.
    the hard alignment with maximum expected TP under the posterior.

    Recurrence:
        A[i, j] = max{ A[i-1, j-1] + P[i-1, j-1],   (match)
                       A[i-1, j],                    (skip ri)
                       A[i, j-1] }.                  (skip rj)

    Ties are broken in favour of matches.

    Reference: Holmes & Durbin, "Dynamic Programming Alignment
    Accuracy", J. Comp. Biol. 5(3):493-504 (1998).
    """
    P = np.asarray(posterior, dtype=np.float64)
    Li, Lj = P.shape
    if Li == 0 or Lj == 0:
        return np.zeros((Li, Lj), dtype=np.float64)
    A = np.zeros((Li + 1, Lj + 1), dtype=np.float64)
    # bt: 0 = match, 1 = skip ri (gap in seq j), 2 = skip rj (gap in seq i)
    bt = np.zeros((Li + 1, Lj + 1), dtype=np.int8)
    for i in range(1, Li + 1):
        row_prev = A[i - 1]
        row_cur = A[i]
        bt_row = bt[i]
        Pi = P[i - 1]
        for j in range(1, Lj + 1):
            m = row_prev[j - 1] + Pi[j - 1]   # match
            u = row_prev[j]                    # skip ri
            v = row_cur[j - 1]                 # skip rj
            if m >= u and m >= v:
                row_cur[j] = m
                bt_row[j] = 0
            elif u >= v:
                row_cur[j] = u
                bt_row[j] = 1
            else:
                row_cur[j] = v
                bt_row[j] = 2
    ind = np.zeros((Li, Lj), dtype=np.float64)
    i, j = Li, Lj
    while i > 0 and j > 0:
        b = bt[i, j]
        if b == 0:
            ind[i - 1, j - 1] = 1.0
            i -= 1
            j -= 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    return ind


def _optimal_accuracy_pair_posteriors(pair_posteriors):
    """Apply optimal-accuracy DP to every soft posterior in a family.

    Returns a new dict of 0/1 indicator matrices with the same keys.
    """
    return {k: _optimal_accuracy_indicator(v)
             for k, v in pair_posteriors.items()
             if isinstance(v, np.ndarray)}


def _msa_to_pair_indicators(int_seqs, names, pairs, msa_strings):
    """Convert a hard MSA into 0/1 indicator "posteriors" per pair.

    For each pair (i, j), cells in the (Li, Lj) grid that the MSA
    places in the same column become 1.0; everything else is 0.
    """
    import warnings
    pp = {}
    for i, j in pairs:
        name_i = names[i]
        name_j = names[j]
        Li = len(int_seqs[name_i])
        Lj = len(int_seqs[name_j])
        if name_i not in msa_strings or name_j not in msa_strings:
            warnings.warn(f'{name_i} or {name_j} missing from MSA output')
            pp[(i, j)] = np.zeros((Li, Lj), dtype=np.float64)
            continue
        # core_only=False: aligners don't annotate core columns; treat
        # every aligned residue pair as a predicted match.
        matches = ref_to_pair_truth(
            {name_i: msa_strings[name_i], name_j: msa_strings[name_j]},
            name_i, name_j, core_only=False)
        post = np.zeros((Li, Lj), dtype=np.float64)
        for (ri, rj) in matches:
            if 0 <= ri < Li and 0 <= rj < Lj:
                post[ri, rj] = 1.0
        pp[(i, j)] = post
    return pp


def _run_external_aligner(raw_seqs, aligner):
    """Run MAFFT or MUSCLE on raw sequences (gap-stripped) and parse output.

    Args:
        raw_seqs: dict {name: ungapped-string}.
        aligner: 'mafft' or 'muscle'.

    Returns:
        dict {name: aligned-string-with-gaps}. Preserves input names.
    """
    import shutil
    import subprocess
    import tempfile

    stripped = {n: s.replace('-', '').replace('.', '').upper()
                 for n, s in raw_seqs.items()
                 if len(s.replace('-', '').replace('.', '')) > 0}
    if len(stripped) < 2:
        return {}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa',
                                       delete=False) as f:
        in_path = f.name
        for name, seq in stripped.items():
            f.write(f'>{name}\n{seq}\n')
    out_path = in_path + '.aln'
    try:
        if aligner == 'mafft':
            mafft_bin = shutil.which('mafft') or '/usr/bin/mafft'
            if not os.path.exists(mafft_bin):
                raise FileNotFoundError('mafft binary not found')
            with open(out_path, 'w') as fo:
                r = subprocess.run(
                    [mafft_bin, '--auto', '--anysymbol', in_path],
                    stdout=fo, stderr=subprocess.PIPE, text=True,
                    timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f'mafft failed: {r.stderr[-500:]}')
        elif aligner == 'muscle':
            muscle_bin = (shutil.which('muscle')
                           or os.path.expanduser('~/bin/muscle'))
            if not os.path.exists(muscle_bin):
                raise FileNotFoundError('muscle binary not found')
            # Try v5 syntax first; fall back to v3.
            r = subprocess.run(
                [muscle_bin, '-align', in_path, '-output', out_path],
                capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                r = subprocess.run(
                    [muscle_bin, '-in', in_path, '-out', out_path],
                    capture_output=True, text=True, timeout=600)
                if r.returncode != 0:
                    raise RuntimeError(f'muscle failed: {r.stderr[-500:]}')
        else:
            raise ValueError(f'unknown aligner {aligner!r}')
        return parse_fasta(out_path)
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def _aligner_pair_posteriors(int_seqs, names, pairs, raw_seqs, aligner):
    """Run an external aligner on the family, project to pair indicators.

    Returns ``(pair_posteriors, failed_pairs=[], extras)`` where
    ``extras['msa_strings']`` is the aligner's reconstructed MSA
    (for downstream SP/TC scoring). Subprocess errors propagate as
    family-level failures.
    """
    msa_strings = _run_external_aligner(raw_seqs, aligner)
    pp = _msa_to_pair_indicators(int_seqs, names, pairs, msa_strings)
    return pp, [], {'msa_strings': msa_strings}


# ---------------------------------------------------------------------------
# Family processing.
# ---------------------------------------------------------------------------


def _build_msa_with_gap_factor(int_seqs, anneal_pp, gap_factor,
                                 n_anneal, seed):
    """Run sequence_annealing with an explicit gap_factor; return
    ``(msa_dict, msa_strings)``."""
    fam_names = list(int_seqs.keys())
    seq_lens = [len(int_seqs[n]) for n in fam_names]
    col_assignments, _ = sequence_annealing(
        len(fam_names), seq_lens, anneal_pp,
        n_iterations=n_anneal,
        gap_factor=gap_factor, edge_weight_threshold=0.0,
        verbose=False, seed=seed)
    n_cols = max(max(ca) for ca in col_assignments if len(ca) > 0) + 1
    msa_dict = {}
    for si, name in enumerate(fam_names):
        row = np.full(n_cols, -1, dtype=np.int32)
        seq = int_seqs[name]
        for k in range(len(seq)):
            row[col_assignments[si][k]] = seq[k]
        msa_dict[name] = row
    msa_strings = msa_to_aligned_strings(msa_dict)
    return msa_dict, msa_strings


def process_family(family_name, in_dir, ref_dir, method_callable,
                    full_pair_cutoff=80, fsa_mode='auto',
                    fsa_anneal_iters=3, fsa_seed=42, fsa_sps=False,
                    cache_method_name=None, cache_params_key=None,
                    cache_disabled=False):
    """Encode a family, build pairs, score with ``method_callable``.

    ``method_callable`` signature:
        ``(int_seqs, names, pairs, raw_seqs) -> (pair_posteriors, kind)``
    where ``kind`` is either ``'soft'`` (posterior probabilities) or
    ``'hard'`` (0/1 indicators from an MSA). The returned dict carries:

        ``micro_post``  / ``per_pair_post`` -- when ``kind == 'soft'``
        ``micro_hard``  / ``per_pair_hard`` -- when ``kind == 'hard'``,
            or when ``kind == 'soft'`` and ``fsa_mode in ('on', 'auto')``
            (in which case sequence_annealing builds an MSA from the
            posteriors and the resulting hard indicators are scored).

    ``fsa_mode``:
        * ``'on'``  -- always run annealing on soft posteriors.
        * ``'off'`` -- never.
        * ``'auto'`` (default) -- run annealing only when the method
                                  returned soft posteriors.
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
        if len(enc) > 0:
            int_seqs[name] = enc
    if len(int_seqs) < 2:
        return None
    names = list(int_seqs.keys())
    n_seqs = len(names)
    if n_seqs > full_pair_cutoff:
        print(f"  [warn] {family_name}: n_seqs={n_seqs} > "
              f"full_pair_cutoff={full_pair_cutoff}; running all pairs")
    pairs = select_pairs_full(n_seqs)

    t0 = time.time()
    # Try cache first.
    cache_hit = False
    if (cache_method_name is not None and cache_params_key is not None
            and not cache_disabled):
        cached = ppcache.load(cache_method_name, family_name,
                                cache_params_key)
        if cached is not None:
            pair_posteriors, kind, failed_pairs = cached
            cache_hit = True
    extras: dict = {}
    if cache_hit:
        # No extras recoverable from cache (we didn't cache MSA strings);
        # downstream MSA-level SP/TC will fall back to per_pair indicators.
        pass
    if not cache_hit:
        try:
            pair_posteriors, kind, failed_pairs, extras = method_callable(
                int_seqs, names, pairs, raw_seqs)
        except Exception as e:
            # Whole-family failure -- nothing was computed. Record it
            # explicitly so the family is visible in the output and
            # downstream code can re-run just these families.
            import jax
            jax.clear_caches()
            return {
                'family': family_name,
                'n_seqs': int(n_seqs),
                'n_pairs': int(len(pairs)),
                'time_posteriors': float(time.time() - t0),
                'micro_post': None,
                'micro_hard': None,
                'per_pair_post': None,
                'per_pair_hard': None,
                '_failed_family': {
                    'error_type': type(e).__name__,
                    'error_msg': str(e)[:500],
                },
            }
        # Save the freshly computed posteriors into the central cache.
        if (cache_method_name is not None
                and cache_params_key is not None
                and not cache_disabled):
            try:
                ppcache.save(cache_method_name, family_name, names,
                              pair_posteriors, kind, cache_params_key,
                              failed_pairs=failed_pairs)
            except Exception as e:
                print(f"   warn: cache.save failed for {family_name}: {e}",
                      flush=True)
    t_post = time.time() - t0
    ref_aln = parse_fasta(ref_path)

    result = {
        'family': family_name,
        'n_seqs': int(n_seqs),
        'n_pairs': int(len(pairs)),
        'time_posteriors': float(t_post),
        'micro_post': None,
        'micro_hard': None,
        'micro_opt': None,
        'micro_fsa_sps': None,
        'per_pair_post': None,
        'per_pair_hard': None,
        'per_pair_opt': None,
        'per_pair_fsa_sps': None,
    }
    if failed_pairs:
        result['_failed_pairs'] = failed_pairs
    if kind == 'soft':
        fam = expected_family_f1(
            pair_posteriors, ref_aln, names, core_only=True)
        result['micro_post'] = fam['micro']
        result['per_pair_post'] = fam['per_pair']
        # Holmes-Durbin optimal-accuracy DP -> hard prediction from same
        # posterior. Directly comparable to FSA / MAFFT / MUSCLE hard.
        try:
            t_opt = time.time()
            opt_pp = _optimal_accuracy_pair_posteriors(pair_posteriors)
            fam_opt = expected_family_f1(
                opt_pp, ref_aln, names, core_only=True)
            result['micro_opt'] = fam_opt['micro']
            result['per_pair_opt'] = fam_opt['per_pair']
            result['time_opt'] = float(time.time() - t_opt)
        except Exception as e:
            result['_opt_failed'] = {
                'error_type': type(e).__name__,
                'error_msg': str(e)[:500],
            }
        anneal_pp = {k: v for k, v in pair_posteriors.items()
                      if isinstance(v, np.ndarray)}
        if fsa_mode in ('on', 'auto'):
            t1 = time.time()
            try:
                _, msa_strings = _build_msa_with_gap_factor(
                    int_seqs, anneal_pp, gap_factor=1.0,
                    n_anneal=fsa_anneal_iters, seed=fsa_seed)
                hard_pp = _msa_to_pair_indicators(
                    int_seqs, names, pairs, msa_strings)
                fam_h = expected_family_f1(
                    hard_pp, ref_aln, names, core_only=True)
                result['micro_hard'] = fam_h['micro']
                result['per_pair_hard'] = fam_h['per_pair']
                # MSA-level pairwise SP / TC via the standard
                # BAliBASE scoring (core columns only).
                sp_msa, tc_msa = sp_tc_score(
                    msa_strings, ref_aln, core_only=True)
                result['msa_sp_g1'] = float(sp_msa)
                result['msa_tc_g1'] = float(tc_msa)
                result['time_fsa'] = float(time.time() - t1)
            except Exception as e:
                result['_fsa_failed'] = {
                    'error_type': type(e).__name__,
                    'error_msg': str(e)[:500],
                }
        if fsa_sps:
            # Second FSA pass with gap_factor=0 -- accept any positive-
            # weight edge merge. Maximises SPS at the cost of precision.
            t1 = time.time()
            try:
                _, msa_strings_sps = _build_msa_with_gap_factor(
                    int_seqs, anneal_pp, gap_factor=0.0,
                    n_anneal=fsa_anneal_iters, seed=fsa_seed)
                hard_pp_sps = _msa_to_pair_indicators(
                    int_seqs, names, pairs, msa_strings_sps)
                fam_sps = expected_family_f1(
                    hard_pp_sps, ref_aln, names, core_only=True)
                result['micro_fsa_sps'] = fam_sps['micro']
                result['per_pair_fsa_sps'] = fam_sps['per_pair']
                sp_msa, tc_msa = sp_tc_score(
                    msa_strings_sps, ref_aln, core_only=True)
                result['msa_sp_g0'] = float(sp_msa)
                result['msa_tc_g0'] = float(tc_msa)
                result['time_fsa_sps'] = float(time.time() - t1)
            except Exception as e:
                result['_fsa_sps_failed'] = {
                    'error_type': type(e).__name__,
                    'error_msg': str(e)[:500],
                }
    elif kind == 'hard':
        fam = expected_family_f1(
            pair_posteriors, ref_aln, names, core_only=True)
        result['micro_hard'] = fam['micro']
        result['per_pair_hard'] = fam['per_pair']
        # External aligners produce their own MSA -- score it via the
        # standard BAliBASE SP/TC.
        msa_strings = extras.get('msa_strings') if extras else None
        if msa_strings is not None:
            try:
                sp_msa, tc_msa = sp_tc_score(
                    msa_strings, ref_aln, core_only=True)
                result['msa_sp_g1'] = float(sp_msa)
                result['msa_tc_g1'] = float(tc_msa)
            except Exception as e:
                result['_msa_score_failed'] = {
                    'error_type': type(e).__name__,
                    'error_msg': str(e)[:500],
                }
    else:
        raise ValueError(f'method_callable returned unknown kind {kind!r}')
    return result


# ---------------------------------------------------------------------------
# Method dispatch.
# ---------------------------------------------------------------------------


def build_method_callable(args):
    """Return a closure ``method(int_seqs, names, pairs) -> pair_posteriors``
    and a dict describing the method config for the output JSON."""
    if args.method == 'tkf92':
        with open(args.params) as f:
            tk = json.load(f)
        ins, del_, ext = float(tk['ins_rate']), float(tk['del_rate']), \
            float(tk['ext_rate'])
        Q_lg, pi_lg = rate_matrix_lg()
        cfg = {'method': 'tkf92', 'params_file': args.params,
               'ins': ins, 'del': del_, 'ext': ext, 'emissions': 'LG08'}

        def _go(int_seqs, names, pairs, _raw_seqs):
            pp, failed = _tkf92_pair_posteriors(
                int_seqs, names, pairs, ins, del_, ext,
                np.asarray(Q_lg), np.asarray(pi_lg))
            return pp, 'soft', failed, {}
        return _go, cfg

    if args.method in ('tkf92_mixture', 'cherryml_mixture'):
        if args.anchor is None:
            raise SystemExit(
                f'--anchor JSON is required for method={args.method}')
        with open(args.anchor) as f:
            tk = json.load(f)
        anchor = {
            'ins': float(tk['ins_rate']), 'del': float(tk['del_rate']),
            'ext': float(tk['ext_rate']),
        }
        Q_lg, pi_lg = rate_matrix_lg()
        anchor['Q'] = np.asarray(Q_lg)
        anchor['pi'] = np.asarray(pi_lg)
        if args.method == 'tkf92_mixture':
            mix = load_mixture_ckpt(args.params)
        else:
            mix = _load_cherryml_as_mixture(args.params, anchor)
        cfg = {'method': args.method, 'params_file': args.params,
               'anchor_file': args.anchor,
               'K': int(mix['K']), 'anchor': anchor.copy()}
        # Strip arrays from cfg JSON.
        cfg['anchor'].pop('Q', None)
        cfg['anchor'].pop('pi', None)

        def _go(int_seqs, names, pairs, _raw_seqs):
            pp, failed = _tkf92_mixture_pair_posteriors(
                int_seqs, names, pairs, anchor, mix,
                pair_chunk=args.pair_chunk)
            return pp, 'soft', failed, {}
        return _go, cfg

    if args.method == 'mixdom':
        params, n_dom, _ = load_mixdom_params(args.params)
        fsa_params, n_frag = make_fsa_params(params, n_dom)
        cfg = {'method': 'mixdom', 'params_file': args.params,
               'n_dom': int(n_dom), 'n_frag': int(n_frag)}

        def _go(int_seqs, names, pairs, _raw_seqs):
            pp, failed = _mixdom_pair_posteriors(
                int_seqs, names, pairs, fsa_params, n_dom, n_frag)
            return pp, 'soft', failed, {}
        return _go, cfg

    if args.method in ('mafft', 'muscle'):
        cfg = {'method': args.method, 'params_file': None}

        def _go(int_seqs, names, pairs, raw_seqs):
            pp, failed, extras = _aligner_pair_posteriors(
                int_seqs, names, pairs, raw_seqs, args.method)
            return pp, 'hard', failed, extras
        return _go, cfg

    raise SystemExit(f'unknown method {args.method!r}')


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', required=True,
                   choices=['tkf92', 'tkf92_mixture',
                            'cherryml_mixture', 'mixdom',
                            'mafft', 'muscle'])
    p.add_argument('--method-name', type=str, default=None,
                   help='Free-form label for the method; defaults to --method.')
    p.add_argument('--params', required=False,
                   help='JSON (tkf92) or NPZ (mixture/mixdom); '
                   'unused for mafft/muscle.')
    p.add_argument('--anchor', type=str, default=None,
                   help='Anchor TKF92 JSON (required for mixture methods).')
    p.add_argument('--out', required=True)
    p.add_argument('--balibase-dir', type=str,
                   default=str(Path('~/bio-datasets/data/balibase/bali3pdbm')
                                .expanduser()))
    p.add_argument('--n-families', type=int, default=0,
                   help='0 = all (default).')
    p.add_argument('--families', type=str, default=None,
                   help='Comma-separated subset (e.g. "BB11001,BB11003").')
    p.add_argument('--full-pair-cutoff', type=int, default=80)
    p.add_argument('--pair-chunk', type=int, default=None,
                   help='K-mixture: pair-batch chunk size (None = full).')
    p.add_argument('--fsa', choices=['on', 'off', 'auto'],
                   default='auto',
                   help='Soft-posterior methods: also run sequence_annealing '
                        'on the posteriors and score the resulting hard MSA. '
                        '"auto" (default) = on for soft methods, no-op for '
                        'hard methods (mafft/muscle). "off" = posterior only.')
    p.add_argument('--fsa-anneal-iters', type=int, default=3)
    p.add_argument('--fsa-seed', type=int, default=42)
    p.add_argument('--fsa-sps', action='store_true',
                   help='Soft methods: also run a second FSA pass with '
                        'gap_factor=0 (accept all positive-weight edge '
                        'merges) -- maximises SPS rather than F1. Adds '
                        'a corpus_fsa_sps branch to the output.')
    p.add_argument('--cache-disabled', action='store_true',
                   help='Bypass the per-family pair-posterior cache '
                        '(both read and write). Default: cache is on.')
    args = p.parse_args()

    method_callable, cfg = build_method_callable(args)
    method_name = args.method_name or args.method
    cfg['method_name'] = method_name
    # Pair-posterior cache key: hash all params files that the method
    # depends on. The cache stores soft posteriors so the FSA gap_factor
    # variants can re-anneal without recomputing the 2D FB.
    cache_param_files = []
    if args.params:
        cache_param_files.append(args.params)
    if args.anchor:
        cache_param_files.append(args.anchor)
    cache_params_key = (ppcache.file_params_key(*cache_param_files,
                                                  extra=method_name)
                         if cache_param_files else method_name)
    cfg['cache_params_key'] = cache_params_key

    in_dir = os.path.join(args.balibase_dir, 'in')
    ref_dir = os.path.join(args.balibase_dir, 'ref')
    families = sorted(os.listdir(in_dir))
    families = [f for f in families
                if os.path.exists(os.path.join(ref_dir, f))]
    if args.families:
        wanted = {s.strip() for s in args.families.split(',')}
        families = [f for f in families if f in wanted]
    if args.n_families > 0:
        families = families[:args.n_families]

    print(f"Running {method_name} ({args.method}) on {len(families)} "
          f"BAliBASE families from {args.balibase_dir} (fsa={args.fsa})",
          flush=True)
    print(f"{'Family':<22} {'N':>3} {'pr':>4} | "
          f"{'f1_post':>7} {'f1_fsa':>6} {'f1_opt':>6} | "
          f"{'gold':>5} | {'time':>6}",
          flush=True)
    print('-' * 80, flush=True)

    results = []
    for fi, fam in enumerate(families):
        try:
            res = process_family(
                fam, in_dir, ref_dir, method_callable,
                full_pair_cutoff=args.full_pair_cutoff,
                fsa_mode=args.fsa,
                fsa_anneal_iters=args.fsa_anneal_iters,
                fsa_seed=args.fsa_seed,
                fsa_sps=args.fsa_sps,
                cache_method_name=method_name,
                cache_params_key=cache_params_key,
                cache_disabled=args.cache_disabled)
        except Exception as e:
            # process_family already handles method failures; this catches
            # very-outer issues (e.g. parse errors). Emit a stub so the
            # family is recorded as having been tried.
            print(f'[{fi+1:>3}/{len(families)}] {fam:<22} OUTER-ERR: '
                  f'{type(e).__name__}: {e}', flush=True)
            results.append({
                'family': fam,
                '_failed_family': {
                    'error_type': type(e).__name__,
                    'error_msg': str(e)[:500],
                    'level': 'outer',
                },
                'micro_post': None, 'micro_hard': None,
                'per_pair_post': None, 'per_pair_hard': None,
            })
            continue
        if res is None:
            continue
        results.append(res)
        if '_failed_family' in res:
            ff = res['_failed_family']
            print(f'[{fi+1:>3}/{len(families)}] {fam:<22} '
                  f'{res["n_seqs"]:>3} {res["n_pairs"]:>4} | '
                  f'FAILED  ({ff["error_type"]}: '
                  f'{ff["error_msg"][:60]})', flush=True)
            continue
        mp = res['micro_post']
        mh = res['micro_hard']
        mo = res.get('micro_opt')
        def _f1(m):
            if m is None or (m['total_mass'] + m['gold']) == 0:
                return None
            return 2 * m['e_tp'] / (m['total_mass'] + m['gold'])
        f1_p = _f1(mp)
        f1_h = _f1(mh)
        f1_o = _f1(mo)
        gold_v = (mp or mh or mo)['gold']
        def _fmt(f):
            return f'{f:.3f}' if f is not None else '   -  '
        fp_tag = (f' [_failed_pairs={len(res["_failed_pairs"])}]'
                   if res.get('_failed_pairs') else '')
        print(f'[{fi+1:>3}/{len(families)}] {fam:<22} '
              f'{res["n_seqs"]:>3} {res["n_pairs"]:>4} | '
              f'{_fmt(f1_p):>7} {_fmt(f1_h):>6} {_fmt(f1_o):>6} | '
              f'{gold_v:>5} | '
              f'{res["time_posteriors"]:>5.1f}s{fp_tag}', flush=True)

    # Pool across families. Build three parallel corpus aggregates by
    # remapping per_pair_{post,hard,opt} to the canonical 'per_pair'
    # field expected by aggregate_corpus.
    corpus_post = aggregate_corpus(
        [{'per_pair': r['per_pair_post']}
         for r in results if r.get('per_pair_post') is not None])
    corpus_hard = aggregate_corpus(
        [{'per_pair': r['per_pair_hard']}
         for r in results if r.get('per_pair_hard') is not None])
    corpus_opt = aggregate_corpus(
        [{'per_pair': r['per_pair_opt']}
         for r in results if r.get('per_pair_opt') is not None])
    corpus_fsa_sps = aggregate_corpus(
        [{'per_pair': r['per_pair_fsa_sps']}
         for r in results if r.get('per_pair_fsa_sps') is not None])
    out_payload = {
        'method_name': method_name,
        'method_type': args.method,
        'config': cfg,
        'balibase_dir': args.balibase_dir,
        'fsa_mode': args.fsa,
        'fsa_sps': bool(args.fsa_sps),
        'n_families': len(results),
        'per_family': results,
        'corpus_post': corpus_post if corpus_post['n_pairs_total'] else None,
        'corpus_hard': corpus_hard if corpus_hard['n_pairs_total'] else None,
        'corpus_opt': corpus_opt if corpus_opt['n_pairs_total'] else None,
        'corpus_fsa_sps': (corpus_fsa_sps
                            if corpus_fsa_sps['n_pairs_total'] else None),
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out_payload, f, indent=2)
    print('-' * 80)
    print(f'wrote {args.out}', flush=True)
    for label, corpus in (('posterior', corpus_post),
                           ('FSA-hard', corpus_hard),
                           ('opt-acc', corpus_opt),
                           ('FSA-sps', corpus_fsa_sps)):
        if not corpus['n_pairs_total']:
            continue
        cm = corpus['micro']
        if cm['total_mass'] + cm['gold'] > 0:
            f1 = 2 * cm['e_tp'] / (cm['total_mass'] + cm['gold'])
            print(f'corpus {label} F1 = {f1:.4f} '
                  f'(e_tp={cm["e_tp"]:.1f}, mass={cm["total_mass"]:.1f}, '
                  f'gold={cm["gold"]}, n_pairs={corpus["n_pairs_total"]})',
                  flush=True)


if __name__ == '__main__':
    main()
