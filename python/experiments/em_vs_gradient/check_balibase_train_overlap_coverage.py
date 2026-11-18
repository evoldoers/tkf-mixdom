#!/usr/bin/env python3
"""For each BAliBASE sequence flagged by check_balibase_train_overlap.py, the
fraction of its residues covered by 25-mers that occur in training sequences of
the flagged Pfam families (a proxy for identity to its closest training
relative)."""
import os, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_a = sys.argv; sys.argv = ['x']
import train_pfam
from tkfmixdom.util.msa_benchmark import parse_fasta
sys.argv = _a
K = 25
d = json.load(open('experiments/em_vs_gradient/balibase_train_overlap.json'))
hit_fams = {f for v in d['kmer_hits'].values() for f in v}
src = train_pfam.PrecompiledPairSource('pfam/precompiled_v3_train')
kmers = set()
for shard in src.shard_files:
    for rec in src._read_shard(shard):
        if rec.get('fam') in hit_fams:
            for s in (rec['x'], rec['y']):
                for i in range(0, max(len(s) - K + 1, 0)): kmers.add(s[i:i+K])
bdir = os.path.expanduser('~/bio-datasets/data/balibase/bali3pdbm/in')
out = []
for key in d['kmer_hits']:
    fam, name = key.split('|')
    seq = ''.join(c for c in parse_fasta(os.path.join(bdir, fam))[name].upper() if c.isalpha())
    cov = [0] * len(seq)
    for i in range(0, max(len(seq) - K + 1, 0)):
        if seq[i:i+K] in kmers:
            for j in range(i, i + K): cov[j] = 1
    out.append((fam, name, len(seq), sum(cov) / max(len(seq), 1)))
out.sort(key=lambda r: -r[3])
import numpy as np
c = np.array([r[3] for r in out])
print(f"{len(out)} flagged BAliBASE sequences: coverage by training 25-mers  median {np.median(c):.2f}, >=0.5: {(c>=0.5).sum()}, >=0.8: {(c>=0.8).sum()}, >=0.95: {(c>=0.95).sum()}")
fam_max = defaultdict(float)
for fam, name, L, cv in out: fam_max[fam] = max(fam_max[fam], cv)
fm = np.array(list(fam_max.values()))
print(f"per BAliBASE family (max over its sequences): >=0.5: {(fm>=0.5).sum()}, >=0.8: {(fm>=0.8).sum()}, >=0.95: {(fm>=0.95).sum()} of {len(fm)} flagged families (120 total)")
for r in out[:15]: print(f"   {r[0]} {r[1]:14s} L={r[2]:4d} covered {r[3]:.2f}")
json.dump([dict(family=r[0], seq=r[1], length=r[2], coverage=r[3]) for r in out], open('experiments/em_vs_gradient/balibase_train_overlap_coverage.json', 'w'), indent=1)
