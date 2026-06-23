#!/usr/bin/env python3
"""Convert the medium random-pair-per-family dataset to (x_int, y_int, t)
pickle format consumable by run_tkf92_2dfb_pfam.py --sim-train-file.

Same MSA traversal + JC69 t estimation as medium_random_pair.py.  For
each pair, gather residue indices (gap-free) for each side; use that
plus the t_est as the (x, y, t) tuple.

90/10 random split into train/val, written to
experiments/exp2_gap_dist/medium_fb/{train,val}.pkl.

Optional length cap (default 600) to keep the F-B's 2D DP tractable
on the 11 GB GPU — pairs with anc or des sequence > cap are dropped.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from train_pfam import parse_stockholm
from medium_random_pair import (
    alignment_to_state_seq_from_msa, count_match_identity,
    jc69_t_from_identity, T_MAX_JC,
)

# Same 20-letter amino acid alphabet as `core.protein.AA_TO_INT`.
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {c: i for i, c in enumerate(AA_ORDER)}


def seq_to_int(s, gap_chars="-."):
    """Map an MSA row to an int array, skipping gap chars.  Unknown
    residues (X, B, Z, U, ...) are mapped to a random residue weighted
    by uniform pi for now — quick hack; the gap-LL eval and the F-B
    don't depend strongly on individual residue choices."""
    out = []
    for c in s:
        if c in gap_chars:
            continue
        u = c.upper()
        if u in AA_TO_INT:
            out.append(AA_TO_INT[u])
        else:
            out.append(0)  # default to 'A' for unknowns; rare
    return np.array(out, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msa-dir", default="/home/yam/bio-datasets/data/pfam-seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-families", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=600,
                     help="Drop pairs with either sequence > this length.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out-dir",
                     default="experiments/exp2_gap_dist/medium_fb")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.msa_dir, "PF*.sto")))
    if args.max_families:
        files = files[:args.max_families]
    print(f"  found {len(files):,} Pfam-seed MSA files", flush=True)

    rng = np.random.default_rng(args.seed)
    pairs = []
    n_dropped_short_aln = 0
    n_dropped_too_long = 0
    t0 = time.time()

    for i, f in enumerate(files):
        try:
            names, seqs = parse_stockholm(f)
        except Exception:
            continue
        if len(seqs) < 2:
            continue
        idx_a, idx_d = rng.choice(len(seqs), size=2, replace=False)
        s_anc = seqs[idx_a]
        s_des = seqs[idx_d]
        # T estimate
        n_m, n_id = count_match_identity(s_anc, s_des)
        if n_m < 5:
            n_dropped_short_aln += 1
            continue
        t_est = jc69_t_from_identity(n_id / n_m)
        # Residue sequences
        x_int = seq_to_int(s_anc)
        y_int = seq_to_int(s_des)
        if (len(x_int) > args.max_seq_len or len(y_int) > args.max_seq_len
                or len(x_int) == 0 or len(y_int) == 0):
            n_dropped_too_long += 1
            continue
        pairs.append((x_int, y_int, float(t_est)))
        if (i + 1) % 1000 == 0:
            print(f"  parsed {i+1}/{len(files)}; kept {len(pairs)}; "
                  f"dropped {n_dropped_short_aln} short-aln + "
                  f"{n_dropped_too_long} too-long ({time.time()-t0:.1f}s)",
                  flush=True)
    print(f"  total kept: {len(pairs):,}; dropped "
          f"{n_dropped_short_aln} short-aln + {n_dropped_too_long} too-long "
          f"in {time.time()-t0:.1f}s", flush=True)

    # 90/10 train/val split with shuffling
    rng_split = random.Random(args.seed)
    indices = list(range(len(pairs)))
    rng_split.shuffle(indices)
    n_val = int(round(len(pairs) * args.val_frac))
    val_pairs = [pairs[i] for i in indices[:n_val]]
    train_pairs = [pairs[i] for i in indices[n_val:]]

    pickle.dump(train_pairs,
                open(os.path.join(args.out_dir, "train.pkl"), "wb"))
    pickle.dump(val_pairs,
                open(os.path.join(args.out_dir, "val.pkl"), "wb"))
    # Light meta
    import json
    meta = {
        "msa_dir": args.msa_dir, "seed": args.seed,
        "max_seq_len": args.max_seq_len, "val_frac": args.val_frac,
        "n_train": len(train_pairs), "n_val": len(val_pairs),
        "n_dropped_short_aln": n_dropped_short_aln,
        "n_dropped_too_long": n_dropped_too_long,
        "t_train_stats": {
            "min": float(np.min([p[2] for p in train_pairs])),
            "median": float(np.median([p[2] for p in train_pairs])),
            "mean": float(np.mean([p[2] for p in train_pairs])),
            "max": float(np.max([p[2] for p in train_pairs])),
        },
        "anc_len_train_stats": {
            "median": int(np.median([len(p[0]) for p in train_pairs])),
            "max": int(np.max([len(p[0]) for p in train_pairs])),
        },
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"),
              indent=2)
    print(f"\nSaved {len(train_pairs)} train + {len(val_pairs)} val pairs "
          f"to {args.out_dir}/", flush=True)
    print(f"  t stats: {meta['t_train_stats']}")
    print(f"  anc len stats: {meta['anc_len_train_stats']}")


if __name__ == "__main__":
    main()
