#!/usr/bin/env python3
"""Build the list of Pfam families to exclude from training so that BAliBASE
(bali3pdbm) inputs have no detectable homolog in the training corpus.

Sources of evidence, all at a lenient reporting threshold (E <= 0.01):
  1. hmmscan of every BAliBASE sequence against Pfam-A.hmm (families whose
     profile matches the sequence, including remote matches);
  2. phmmer of every BAliBASE sequence against the unique training sequences;
  3. jackhmmer (3 iterations) of the same, for remote homologs;
  4. the shared-25-mer families from check_balibase_train_overlap.py.
The union is expanded to whole Pfam clans (consistent with the clan-aware
split) and written to exclude_families.json with per-source provenance.
"""
import os, sys, glob, json, gzip, time
from collections import defaultdict
import pyhmmer
def _s(x): return x.decode() if isinstance(x, bytes) else x
SMOKE = os.environ.get('SMOKE') == '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tkfmixdom.util.msa_benchmark import parse_fasta
E_THRESH = 0.01; CPUS = 16
OUT = 'experiments/balibase_exclusion'
HMM = os.path.expanduser('~/bio-datasets/data/pfam/hmm/Pfam-A.hmm.gz')
CLANS = os.path.expanduser('~/bio-datasets/data/pfam/seed/Pfam-A.clans.tsv.gz')
bdir = os.path.expanduser('~/bio-datasets/data/balibase/bali3pdbm/in')
alphabet = pyhmmer.easel.Alphabet.amino()
queries = []
for p in sorted(glob.glob(os.path.join(bdir, 'BB*'))):
    for name, seq in parse_fasta(p).items():
        s = ''.join(c for c in seq.upper() if c.isalpha())
        queries.append(pyhmmer.easel.TextSequence(name=f"{os.path.basename(p)}|{name}", sequence=s).digitize(alphabet))
if SMOKE: queries = queries[:3]
print(f"{len(queries)} BAliBASE query sequences", flush=True)
evidence = defaultdict(lambda: defaultdict(set))   # family -> source -> queries
t0 = time.time()
# 1. hmmscan vs Pfam-A
with pyhmmer.plan7.HMMFile(HMM) as hf:
    hmms = list(hf)
print(f"loaded {len(hmms)} Pfam HMMs ({time.time()-t0:.0f}s)", flush=True)
for hits in pyhmmer.hmmer.hmmscan(queries, hmms, cpus=CPUS, E=E_THRESH, domE=E_THRESH):
    for h in hits:
        if h.evalue <= E_THRESH:
            evidence[_s(h.accession).split('.')[0]]['hmmscan'].add(_s(hits.query.name))
print(f"hmmscan: {len(evidence)} families ({time.time()-t0:.0f}s)", flush=True)
# 2./3. phmmer and jackhmmer vs training sequences
with pyhmmer.easel.SequenceFile(f'{OUT}/train_seqs.fa', digital=True, alphabet=alphabet) as sf:
    targets = sf.read_block(sequences=50000) if SMOKE else sf.read_block()
print(f"loaded {len(targets)} training sequences ({time.time()-t0:.0f}s)", flush=True)
def fam_of(name): return _s(name).split('|')[0]
for hits in pyhmmer.hmmer.phmmer(queries, targets, cpus=CPUS, E=E_THRESH):
    for h in hits:
        if h.evalue <= E_THRESH: evidence[fam_of(h.name)]['phmmer'].add(_s(hits.query.name))
print(f"after phmmer: {len(evidence)} families ({time.time()-t0:.0f}s)", flush=True)
for result in pyhmmer.hmmer.jackhmmer(queries, targets, cpus=CPUS, max_iterations=3, E=E_THRESH, incE=1e-3):
    last = result[-1] if isinstance(result, list) else result
    for h in last.hits:
        if h.evalue <= E_THRESH: evidence[fam_of(h.name)]['jackhmmer'].add(_s(last.hits.query.name))
print(f"after jackhmmer: {len(evidence)} families ({time.time()-t0:.0f}s)", flush=True)
# 4. shared 25-mers
km = json.load(open('experiments/em_vs_gradient/balibase_train_overlap.json'))
for q, fams in km['kmer_hits'].items():
    for f in fams: evidence[f]['kmer25'].add(q)
# clan expansion
clan_of = {}
with gzip.open(CLANS, 'rt') as f:
    for line in f:
        p = line.rstrip('\n').split('\t'); clan_of[p[0]] = p[1] if len(p) > 1 and p[1] else None
members = defaultdict(set)
for fam, c in clan_of.items():
    if c: members[c].add(fam)
direct = set(evidence); clans = {clan_of.get(f) for f in direct} - {None}
expanded = set(direct)
for c in clans: expanded |= members[c]
v1 = json.load(open(os.path.expanduser('~/bio-datasets/data/pfam/seed/splits/v1.json')))
train = set(v1['train'])
print(f"direct {len(direct)} families, {len(clans)} clans, expanded {len(expanded)}; in train split: direct {len(direct & train)}, expanded {len(expanded & train)}", flush=True)
json.dump(dict(E_thresh=E_THRESH, direct=sorted(direct), clans=sorted(clans), exclude=sorted(expanded), exclude_in_train=sorted(expanded & train),
               evidence={f: {s: sorted(q) for s, q in d.items()} for f, d in evidence.items()}),
          open(f'{OUT}/exclude_families{"_smoke" if SMOKE else ""}.json', 'w'), indent=1)
print('wrote exclude_families.json', flush=True)
