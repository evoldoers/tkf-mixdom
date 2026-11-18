#!/usr/bin/env python3
"""A fixed random subset of a precompiled pair corpus (one shard, same JSON lines), for scoring every fit on the
same training pairs. Usage: make_train_subset.py <src_dir> <dst_dir> <n_pairs> <seed>"""
import io, json, os, sys
import numpy as np, zstandard as zstd

src, dst, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
man = json.load(open(os.path.join(src, 'manifest.json')))
lines = []
for sh in man['shard_files']:
    with open(os.path.join(src, sh), 'rb') as fh:
        with zstd.ZstdDecompressor().stream_reader(fh) as rd:
            for line in io.TextIOWrapper(rd, encoding='utf-8'):
                if line.strip(): lines.append(line)
rng = np.random.default_rng(seed); idx = np.sort(rng.choice(len(lines), size=n, replace=False))
os.makedirs(dst, exist_ok=True)
with open(os.path.join(dst, 'shard_0000.jsonl.zst'), 'wb') as fh:
    with zstd.ZstdCompressor().stream_writer(fh) as wr:
        for i in idx: wr.write(lines[i].encode('utf-8'))
fams = set()
for i in idx:
    try:
        d = json.loads(lines[i]); fams.add(d.get('family') or d.get('fam') or d.get('family_id') or i)
    except Exception: fams.add(i)
json.dump({'n_pairs': int(n), 'n_families': len(fams), 'n_shards': 1, 'shard_files': ['shard_0000.jsonl.zst'], 'pairs_per_shard': int(n), 'complete': True,
           'source': {'parent': src, 'n_pairs': int(n), 'seed': seed, 'sampling': 'uniform random subset of the parent corpus'}},
          open(os.path.join(dst, 'manifest.json'), 'w'), indent=1)
print(f'{n} of {len(lines)} pairs -> {dst}')
