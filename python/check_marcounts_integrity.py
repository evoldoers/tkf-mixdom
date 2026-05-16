#!/usr/bin/env python3
"""Spot-check Maraschino .marcounts.npz files for zip corruption.

Walks a directory of per-family .marcounts.npz files and tries to open
each. Files that fail (most commonly a truncated write from a killed
multiprocessing worker) are reported and optionally deleted.

Usage:
    uv run python check_marcounts_integrity.py \\
        --dir /home/yam/bio-datasets/data/pfam-seed/ \\
        [--delete-bad]
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', required=True)
    p.add_argument('--suffix', default='.marcounts.npz')
    p.add_argument('--delete-bad', action='store_true',
                   help='Remove corrupt files (otherwise just print).')
    p.add_argument('--required-keys', nargs='*',
                   default=['B', 'C_MM', 'C_SE', 'tau_centers'])
    args = p.parse_args()

    files = sorted(Path(args.dir).glob(f"*{args.suffix}"))
    print(f"[check] {len(files)} files in {args.dir}")
    bad = []
    for i, p in enumerate(files):
        try:
            d = np.load(p, allow_pickle=True)
            for k in args.required_keys:
                if k not in d.files:
                    raise KeyError(f"missing key {k}")
            d.close()
        except Exception as e:
            bad.append((p, str(e)[:80]))
        if (i + 1) % 1000 == 0:
            print(f"[check] {i+1}/{len(files)} ({len(bad)} bad)")

    print(f"[check] DONE: {len(files) - len(bad)} OK, {len(bad)} bad")
    for path, err in bad[:30]:
        print(f"  {path.name}: {err}")
    if args.delete_bad and bad:
        for path, _ in bad:
            try:
                path.unlink()
            except OSError as e:
                print(f"  could not delete {path}: {e}", file=sys.stderr)
        print(f"[check] deleted {len(bad)} corrupt files")


if __name__ == '__main__':
    main()
