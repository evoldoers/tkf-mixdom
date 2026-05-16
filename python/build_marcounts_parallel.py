#!/usr/bin/env python3
"""Parallel per-family Maraschino cherry-count builder.

Wraps `_count_one_msa` from maraschino.py with multiprocessing so that
N workers count disjoint chunks of the train split simultaneously,
without re-hashing the same files (the bottleneck of running multiple
`maraschino count` processes against the same dir).

Usage:
    uv run python build_marcounts_parallel.py \\
        --msa-dir /home/yam/bio-datasets/data/pfam-seed/ \\
        --out-suffix .marcounts.npz \\
        --n-tau-bins 8 \\
        --split-file /home/yam/bio-datasets/data/pfam-seed/splits/v1.json \\
        --split train \\
        --workers 8
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

# Force CPU before importing maraschino (avoids OOM on shared GPU).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from maraschino import (
    _count_one_msa, _make_metadata, _save_counts, geom_bin_edges, _file_hash,
    TAU_MIN, TAU_MAX,
)


def _msa_out_path(msa_file, suffix):
    base = msa_file
    for ext in ('.sto.gz', '.sto'):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    return base + suffix


def _process_one(args_tuple):
    msa_file, n_tau, edges, centers, out_suffix, msa_dir = args_tuple
    out_path = _msa_out_path(msa_file, out_suffix)
    if os.path.exists(out_path):
        return ('skipped', msa_file)
    try:
        C, B, n_pairs = _count_one_msa(msa_file, n_tau, edges,
                                        gamma_labels=None, n_gamma=1)
    except Exception as e:
        return ('error', msa_file, str(e))
    if n_pairs == 0:
        # Don't write empty count files (would be misleading); record skip.
        return ('empty', msa_file)
    meta = _make_metadata(n_tau, edges, centers, n_pairs,
                           msa_dir=msa_dir, n_gamma=1,
                           extra={'source_file': os.path.basename(msa_file)})
    seen = {_file_hash(msa_file): os.path.basename(msa_file)}
    _save_counts(out_path, C, B, edges, centers, meta, seen)
    return ('written', msa_file, n_pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--msa-dir', required=True)
    p.add_argument('--out-suffix', default='.marcounts.npz')
    p.add_argument('--n-tau-bins', type=int, default=8)
    p.add_argument('--split-file', default=None)
    p.add_argument('--split', choices=['train', 'val', 'test', 'all'],
                   default='train')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--log-every', type=int, default=200)
    args = p.parse_args()

    edges, centers = geom_bin_edges(args.n_tau_bins, TAU_MIN, TAU_MAX)

    # Resolve family list
    fam_set = None
    if args.split_file:
        with open(os.path.expanduser(args.split_file)) as f:
            sd = json.load(f)
        if args.split == 'all':
            fam_set = set().union(*[set(sd.get(s, []))
                                    for s in ('train', 'val', 'test')])
        else:
            fam_set = set(sd.get(args.split, []))
        print(f"[load] split={args.split} → {len(fam_set)} families")

    msa_files = []
    msa_dir = args.msa_dir
    for name in sorted(os.listdir(msa_dir)):
        if not (name.endswith('.sto') or name.endswith('.sto.gz')):
            continue
        if fam_set is not None:
            fam_id = name.split('.')[0]
            if fam_id not in fam_set:
                continue
        msa_files.append(os.path.join(msa_dir, name))
    print(f"[load] {len(msa_files)} MSA files in {msa_dir}")

    # Pre-filter already-counted files (saves the worker overhead)
    todo = []
    n_skip_existing = 0
    for mf in msa_files:
        if os.path.exists(_msa_out_path(mf, args.out_suffix)):
            n_skip_existing += 1
            continue
        todo.append(mf)
    print(f"[load] {n_skip_existing} already counted, {len(todo)} TODO")

    if not todo:
        print("[done] nothing to do")
        return

    # Build worker-input tuples (in shuffled order so chunks are uniform).
    rng = np.random.RandomState(0)
    order = rng.permutation(len(todo))
    inputs = [(todo[int(i)], args.n_tau_bins, edges, centers,
               args.out_suffix, msa_dir) for i in order]

    t0 = time.monotonic()
    n_done = 0
    n_written = 0
    n_empty = 0
    n_error = 0
    last_log = time.monotonic()

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_process_one, inputs, chunksize=4):
            n_done += 1
            kind = result[0]
            if kind == 'written':
                n_written += 1
            elif kind == 'empty':
                n_empty += 1
            elif kind == 'error':
                n_error += 1
                print(f"[error] {result[1]}: {result[2]}", file=sys.stderr)
            now = time.monotonic()
            if (n_done % args.log_every == 0
                    or now - last_log > 60
                    or n_done == len(inputs)):
                rate = n_done / max(now - t0, 1e-3)
                eta_s = (len(inputs) - n_done) / max(rate, 1e-3)
                print(f"[progress] {n_done}/{len(inputs)} "
                      f"(written={n_written}, empty={n_empty}, error={n_error}) "
                      f"rate={rate:.2f} fam/s ETA={eta_s/60:.1f}min")
                last_log = now

    dt = time.monotonic() - t0
    print(f"[done] {n_done} processed in {dt/60:.1f} min "
          f"({n_written} written, {n_empty} empty, {n_error} error)")


if __name__ == '__main__':
    main()
