#!/usr/bin/env python3
"""Sequence-level check that BAliBASE (bali3pdbm) inputs do not occur in the
Pfam training pairs: exact ungapped-sequence matches and shared 25-mers
(any BAliBASE sequence sharing a 25-mer with a training sequence is flagged
as a possible near-identical relative). Training sequences are scanned at
stride 12, so any shared stretch >= 37 residues is detected."""
import os, sys, glob, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_a = sys.argv; sys.argv = ['x']
import train_pfam
from tkfmixdom.util.msa_benchmark import parse_fasta
sys.argv = _a
K, STRIDE = 25, 12
bdir = os.path.expanduser('~/bio-datasets/data/balibase/bali3pdbm/in')
bali = {}
for p in sorted(glob.glob(os.path.join(bdir, 'BB*'))):
    for name, seq in parse_fasta(p).items():
        bali[(os.path.basename(p), name)] = ''.join(c for c in seq.upper() if c.isalpha())
kmer_owner = defaultdict(set)
for key, s in bali.items():
    for i in range(0, max(len(s) - K + 1, 0)):
        kmer_owner[s[i:i+K]].add(key)
exact = {s: key for key, s in bali.items()}
print(f"{len(bali)} BAliBASE sequences from {len(set(k[0] for k in bali))} families; {len(kmer_owner)} distinct {K}-mers", flush=True)
src = train_pfam.PrecompiledPairSource('pfam/precompiled_v3_train')
hits = defaultdict(set); exact_hits = set(); n_seq = 0
for shard in src.shard_files:
    for rec in src._read_shard(shard):
        for s in (rec['x'], rec['y']):
            n_seq += 1
            if s in exact: exact_hits.add((exact[s], rec.get('fam')))
            for i in range(0, max(len(s) - K + 1, 0), STRIDE):
                owners = kmer_owner.get(s[i:i+K])
                if owners:
                    for o in owners: hits[o].add(rec.get('fam'))
print(f"scanned {n_seq} training sequences", flush=True)
fams_hit = sorted(set(k[0] for k in hits))
print(f"exact-sequence matches: {len(exact_hits)}")
print(f"BAliBASE sequences sharing a {K}-mer with a training sequence: {len(hits)} / {len(bali)}, in {len(fams_hit)} / {len(set(k[0] for k in bali))} BAliBASE families")
for key in sorted(hits)[:25]:
    print(f"   {key[0]} {key[1]:20s} -> Pfam {sorted(hits[key])[:6]}")
json.dump(dict(n_bali=len(bali), exact=[list(map(str, e)) for e in exact_hits],
               kmer_hits={f"{k[0]}|{k[1]}": sorted(v) for k, v in hits.items()}),
          open('experiments/em_vs_gradient/balibase_train_overlap.json', 'w'), indent=1)
