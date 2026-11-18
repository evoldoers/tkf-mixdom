#!/usr/bin/env python3
"""Export the unique sequences of the Pfam training pairs (precompiled_v3_train)
as FASTA, header = <family>|<index>, for homology search against BAliBASE."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_a = sys.argv; sys.argv = ['x']
import train_pfam
sys.argv = _a
src = train_pfam.PrecompiledPairSource('pfam/precompiled_v3_train')
seen = set(); n = 0
with open('experiments/balibase_exclusion/train_seqs.fa', 'w') as f:
    for shard in src.shard_files:
        for rec in src._read_shard(shard):
            for s in (rec['x'], rec['y']):
                key = (rec.get('fam'), s)
                if key in seen: continue
                seen.add(key); n += 1
                f.write(f">{rec.get('fam')}|{n}\n{s}\n")
print(f"wrote {n} unique (family, sequence) records", flush=True)
