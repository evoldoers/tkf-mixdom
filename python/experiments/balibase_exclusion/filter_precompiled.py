#!/usr/bin/env python3
"""Write a copy of a precompiled pair corpus without the excluded families.
Usage: filter_precompiled.py <src_dir> <dst_dir> <exclude_families.json>"""
import os, sys, json
import zstandard as zstd
src, dst, excl = sys.argv[1:4]
exclude = set(json.load(open(excl))['exclude'])
man = json.load(open(os.path.join(src, 'manifest.json')))
os.makedirs(dst, exist_ok=True)
kept, dropped, fams = 0, 0, set(); dropped_fams = set()
for shard in man['shard_files']:
    with open(os.path.join(src, shard), 'rb') as f: data = zstd.ZstdDecompressor().decompress(f.read(), max_output_size=1 << 34)
    out = []
    for line in data.decode().splitlines():
        if not line.strip(): continue
        rec = json.loads(line); fam = rec.get('fam')
        if fam in exclude: dropped += 1; dropped_fams.add(fam); continue
        kept += 1; fams.add(fam); out.append(line)
    with open(os.path.join(dst, shard), 'wb') as f: f.write(zstd.ZstdCompressor(level=9).compress(('\n'.join(out) + '\n').encode()))
man.update(n_pairs=kept, n_families=len(fams), source=dict(man.get('source', {}), filtered_from=src, excluded_families_file=excl, n_excluded_families_present=len(dropped_fams), n_pairs_dropped=dropped))
json.dump(man, open(os.path.join(dst, 'manifest.json'), 'w'), indent=1)
print(f"kept {kept} pairs / {len(fams)} families; dropped {dropped} pairs from {len(dropped_fams)} excluded families present")
