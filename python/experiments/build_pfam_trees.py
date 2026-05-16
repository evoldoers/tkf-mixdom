#!/usr/bin/env python3
"""Build FastTree trees for Pfam-seed families that don't have one yet.

Reads splits/v1.json, finds families without a corresponding .nwk in
~/bio-datasets/data/pfam-seed/trees, parses each .sto MSA, runs FastTree
on the FASTA-converted MSA, and writes the resulting Newick to the trees
directory.

Parallelised across CPU cores via multiprocessing.

Usage:
    uv run python experiments/build_pfam_trees.py --split train  # 18.7k fams
    uv run python experiments/build_pfam_trees.py --split all    # train+val+test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.ancrec_benchmark import parse_sto


PFAM_DIR = os.path.expanduser('~/bio-datasets/data/pfam/seed')
TREE_DIR = os.path.expanduser('~/bio-datasets/data/pfam-seed/trees')
SPLITS_PATH = os.path.expanduser(
    '~/bio-datasets/data/pfam/seed/splits/v1.json')
FASTTREE = '/home/yam/bin/FastTree'


def build_tree(fam):
    """Read .sto for one family, run FastTree, write .nwk. Returns (fam, ok, msg)."""
    sto_path = os.path.join(PFAM_DIR, f'{fam}.sto')
    nwk_path = os.path.join(TREE_DIR, f'{fam}.nwk')
    if os.path.exists(nwk_path):
        return (fam, True, 'already exists')
    if not os.path.exists(sto_path):
        return (fam, False, f'no .sto at {sto_path}')
    try:
        seqs = parse_sto(sto_path)
    except Exception as e:
        return (fam, False, f'parse_sto failed: {e}')
    if len(seqs) < 3:
        return (fam, False, f'only {len(seqs)} seqs (FastTree needs >=3)')
    fasta = ''.join(f'>{n}\n{s}\n' for n, s in seqs.items())
    try:
        out = subprocess.run(
            [FASTTREE, '-nopr', '-quiet'],
            input=fasta, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return (fam, False, 'FastTree timeout (>600s)')
    if out.returncode != 0 or not out.stdout.strip():
        return (fam, False, f'FastTree failed: {out.stderr[:200]}')
    tmp_path = nwk_path + '.tmp'
    with open(tmp_path, 'w') as f:
        f.write(out.stdout)
    os.replace(tmp_path, nwk_path)
    return (fam, True, f'{len(seqs)} seqs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'],
                        default='train')
    parser.add_argument('--workers', type=int, default=18,
                        help='Parallel processes (default 18 of 20 cores).')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit to first N families (for testing; 0=all).')
    args = parser.parse_args()

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    if args.split == 'all':
        target = sorted(set(splits['train'] + splits['val'] + splits['test']))
    else:
        target = sorted(splits[args.split])

    existing = {p.stem for p in Path(TREE_DIR).glob('*.nwk')}
    missing = [f for f in target if f not in existing]
    if args.limit > 0:
        missing = missing[:args.limit]

    print(f'Split {args.split}: {len(target)} total, '
          f'{len(target) - len(missing)} have trees, {len(missing)} to build.')
    if not missing:
        print('Nothing to do.')
        return

    Path(TREE_DIR).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_done = 0
    n_ok = 0
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_tree, f): f for f in missing}
        for fut in as_completed(futures):
            fam, ok, msg = fut.result()
            n_done += 1
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                print(f'  FAIL {fam}: {msg}')
            if n_done % 100 == 0 or n_done == len(missing):
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (len(missing) - n_done) / rate if rate > 0 else 0
                print(f'[{elapsed:.0f}s] {n_done}/{len(missing)} done '
                      f'(ok={n_ok}, fail={n_fail}, '
                      f'rate={rate:.1f}/s, ETA={eta:.0f}s)', flush=True)
    print(f'Total: {n_done} attempted, {n_ok} ok, {n_fail} failed in '
          f'{time.time()-t0:.0f}s.')


if __name__ == '__main__':
    main()
