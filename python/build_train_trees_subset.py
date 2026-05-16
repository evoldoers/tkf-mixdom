#!/usr/bin/env python3
"""Build FastTree trees for a random subset of Pfam train families.

Reads .sto files from --msa-dir, runs FastTree (LG matrix) per family,
saves .nwk to --tree-dir. Parallelised via multiprocessing.

Usage:
    uv run python build_train_trees_subset.py \\
        --split-file ~/bio-datasets/data/pfam/seed/splits/v1.json \\
        --split train --max-families 3000 --workers 16 \\
        --msa-dir ~/bio-datasets/data/pfam/seed/ \\
        --tree-dir ~/bio-datasets/data/pfam/seed/trees/
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def parse_stockholm_to_fasta(sto_path: str, fa_path: str) -> int:
    """Parse a Stockholm MSA and write a FASTA. Returns n_seqs."""
    seqs: dict[str, str] = {}
    opener = gzip.open if sto_path.endswith('.gz') else open
    with opener(sto_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                name, seq = parts[0], parts[1]
                seqs[name] = seqs.get(name, '') + seq
    if len(seqs) < 2:
        return len(seqs)
    with open(fa_path, 'w') as f:
        for name, seq in seqs.items():
            f.write(f'>{name}\n{seq}\n')
    return len(seqs)


def build_tree_one(args_tuple):
    family, msa_dir, tree_dir, fasttree_bin = args_tuple
    sto = os.path.join(msa_dir, f'{family}.sto')
    if not os.path.exists(sto):
        return (family, 'no_sto', 0.0)
    nwk = os.path.join(tree_dir, f'{family}.nwk')
    if os.path.exists(nwk) and os.path.getsize(nwk) > 0:
        return (family, 'cached', 0.0)

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix='.fa', delete=False) as tmp:
        tmp_fa = tmp.name
    try:
        n = parse_stockholm_to_fasta(sto, tmp_fa)
        if n < 3:
            # FastTree needs ≥3 leaves to make a non-trivial tree. For
            # n=2 we write a minimal Newick by hand.
            if n == 2:
                with open(sto) as f:
                    names = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('//'):
                            parts = line.split()
                            if len(parts) >= 2 and parts[0] not in names:
                                names.append(parts[0])
                                if len(names) == 2:
                                    break
                with open(nwk, 'w') as f:
                    f.write(f'({names[0]}:0.5,{names[1]}:0.5);\n')
                return (family, 'pair', time.time() - t0)
            return (family, 'too_few', time.time() - t0)
        # FastTree: -lg model (LG protein), -nosupport (skip support values),
        # -quiet (no progress to stderr).
        proc = subprocess.run(
            [fasttree_bin, '-lg', '-nosupport', '-quiet', tmp_fa],
            capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return (family, f'ft_err:{proc.returncode}', time.time() - t0)
        with open(nwk, 'w') as f:
            f.write(proc.stdout)
        return (family, 'ok', time.time() - t0)
    except subprocess.TimeoutExpired:
        return (family, 'timeout', time.time() - t0)
    except Exception as e:
        return (family, f'err:{type(e).__name__}', time.time() - t0)
    finally:
        if os.path.exists(tmp_fa):
            os.unlink(tmp_fa)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-file', type=str,
                   default=os.path.expanduser(
                       '~/bio-datasets/data/pfam/seed/splits/v1.json'))
    p.add_argument('--split', type=str, default='train')
    p.add_argument('--msa-dir', type=str,
                   default=os.path.expanduser('~/bio-datasets/data/pfam/seed/'))
    p.add_argument('--tree-dir', type=str,
                   default=os.path.expanduser('~/bio-datasets/data/pfam/seed/trees/'))
    p.add_argument('--max-families', type=int, default=3000)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fasttree-bin', type=str,
                   default=os.path.expanduser('~/bin/FastTree'))
    args = p.parse_args()

    msa_dir = os.path.expanduser(args.msa_dir)
    tree_dir = os.path.expanduser(args.tree_dir)
    os.makedirs(tree_dir, exist_ok=True)

    with open(args.split_file) as f:
        splits = json.load(f)
    fams = splits[args.split]
    rng = np.random.RandomState(args.seed)
    if args.max_families and args.max_families < len(fams):
        fams = sorted(rng.choice(fams, args.max_families, replace=False))
    print(f'[plan] {args.split}: {len(fams)} families '
          f'(workers={args.workers}, fasttree={args.fasttree_bin})')

    tasks = [(f, msa_dir, tree_dir, args.fasttree_bin) for f in fams]
    t0 = time.time()
    n_ok = n_cached = n_err = 0
    err_kinds: dict[str, int] = {}
    with mp.Pool(args.workers) as pool:
        for i, (fam, status, dt) in enumerate(pool.imap_unordered(build_tree_one, tasks, chunksize=4)):
            if status == 'ok':
                n_ok += 1
            elif status == 'cached':
                n_cached += 1
            else:
                n_err += 1
                err_kinds[status] = err_kinds.get(status, 0) + 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1.0)
                eta = (len(fams) - (i + 1)) / max(rate, 1e-6)
                print(f'  [{i+1}/{len(fams)}] ok={n_ok} cached={n_cached} '
                      f'err={n_err} | {elapsed:.0f}s | ETA {eta:.0f}s '
                      f'| recent={status} ({dt:.1f}s)', flush=True)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min):')
    print(f'  ok={n_ok}, cached={n_cached}, err={n_err}')
    if err_kinds:
        print('  error breakdown:')
        for k, v in sorted(err_kinds.items(), key=lambda kv: -kv[1]):
            print(f'    {k}: {v}')


if __name__ == '__main__':
    main()
