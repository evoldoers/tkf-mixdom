#!/usr/bin/env python3
"""Write a single-shard precompiled-pairs directory holding exactly the
breadth-first Pfam sample used by compare_optimizers*.py (same shard order,
length filter, family split and RNG streams as
run_tkf92_2dfb_pfam.load_breadth_first_pairs), so train_pfam.py's --svi-bw and
--adam modes can be run full-batch on the same fixed corpus.

Usage: make_precompiled_subset.py --n-pairs 1000 --seed 0 --out pfam/precompiled_sub1000_seed0
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
import zstandard as zstd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from train_pfam import PrecompiledPairSource
from tkfmixdom.jax.util.pair_format import decode_pair

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='pfam/precompiled')
ap.add_argument('--n-pairs', type=int, default=1000)
ap.add_argument('--max-len', type=int, default=256)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()

src = PrecompiledPairSource(a.src, max_alignment_len=a.max_len)
by_family = defaultdict(list)
for shard in src.shard_files:
    for rec in src._read_shard(shard):
        x_int, y_int, states, *_ = decode_pair(rec)
        if a.max_len and len(states) > a.max_len:
            continue
        by_family[rec.get('fam', 'unknown')].append(rec)
families = sorted(by_family.keys())
n_val_fam = int(round(len(families) * 0.05))
train_families = families[n_val_fam:]
rng = np.random.default_rng(a.seed)
rng.shuffle(train_families)
q = {f: list(by_family[f]) for f in train_families}
for f in q:
    rng.shuffle(q[f])
out = []
any_left = True
while any_left and len(out) < a.n_pairs:
    any_left = False
    for f in train_families:
        if not q[f]:
            continue
        out.append(q[f].pop()); any_left = True
        if len(out) >= a.n_pairs:
            break
os.makedirs(a.out, exist_ok=True)
data = '\n'.join(json.dumps(r) for r in out) + '\n'
with open(os.path.join(a.out, 'shard_0000.jsonl.zst'), 'wb') as fh:
    fh.write(zstd.ZstdCompressor().compress(data.encode()))
fams = sorted({r.get('fam', 'unknown') for r in out})
json.dump(dict(n_pairs=len(out), n_families=len(fams), n_shards=1,
               shard_files=['shard_0000.jsonl.zst'], pairs_per_shard=len(out),
               complete=True, source=dict(parent=a.src, n_pairs=a.n_pairs, seed=a.seed,
                                          max_len=a.max_len, sampling='breadth-first train split')),
          open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)
print(f"wrote {len(out)} pairs from {len(fams)} families to {a.out}; first families {fams[:3]}")
