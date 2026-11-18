#!/usr/bin/env python3
"""BAliBASE SP/TC on core blocks AND on all reference columns, for every cached
method, without rerunning any pair HMM.

Posterior-based methods: the gap-factor-0 MSA is rebuilt from the cached pair
posteriors (same builder/seed as expected_pairwise_balibase.py) and scored
twice (sp_tc_score core_only=True / False). MAFFT / MUSCLE: the aligner is
rerun (deterministic, seconds per family) and its MSA scored the same two ways.
Usage: balibase_core_vs_full.py --methods tkf92 mixfrag_F2 ... --out <json>
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from tkfmixdom.util.msa_benchmark import parse_fasta, sp_tc_score
from tkfmixdom.util import balibase_pair_cache as ppcache
from fsa_mixdom_pairhmm_oxbench import encode_seq
from expected_pairwise_balibase import _build_msa_with_gap_factor, _run_external_aligner

p = argparse.ArgumentParser()
p.add_argument('--methods', nargs='+', required=True)
p.add_argument('--balibase-dir', default=str(Path('~/bio-datasets/data/balibase/bali3pdbm').expanduser()))
p.add_argument('--gap-factor', type=float, default=0.0)
p.add_argument('--out', required=True)
args = p.parse_args()
base = ppcache.DEFAULT_CACHE_DIR; in_dir = os.path.join(args.balibase_dir, 'in'); ref_dir = os.path.join(args.balibase_dir, 'ref')
result = {}
for method in args.methods:
    mdir = base / method; fams = sorted(q.stem for q in mdir.glob('*.npz')); rows = []; t_all = time.time()
    print(f"== {method}: {len(fams)} families", flush=True)
    for fam in fams:
        meta = json.loads((mdir / f'{fam}.json').read_text())
        raw = parse_fasta(os.path.join(in_dir, fam)); ref_aln = parse_fasta(os.path.join(ref_dir, fam))
        if method in ('mafft', 'muscle'):
            msa = _run_external_aligner(raw, method)
        else:
            int_seqs = {n: encode_seq(s) for n, s in raw.items()}; int_seqs = {n: s for n, s in int_seqs.items() if len(s) > 0}
            names = list(int_seqs.keys()); d = np.load(mdir / f'{fam}.npz', allow_pickle=True)
            cached_names = [str(n) for n in (d['names'] if 'names' in d.files else meta['names'])]
            pairs = np.asarray(d['pairs'] if 'pairs' in d.files else meta['pairs']); idx = {n: names.index(n) for n in cached_names}
            pp = {}
            for k, (i, j) in enumerate(pairs.tolist()):
                key = f'post_{k}' if f'post_{k}' in d.files else f'p_{i}_{j}'
                pp[(idx[cached_names[i]], idx[cached_names[j]])] = np.asarray(d[key])
            _, msa = _build_msa_with_gap_factor(int_seqs, pp, gap_factor=args.gap_factor, n_anneal=3, seed=42)
        sp_c, tc_c = sp_tc_score(msa, ref_aln, core_only=True); sp_f, tc_f = sp_tc_score(msa, ref_aln, core_only=False)
        rows.append(dict(family=fam, sp_core=float(sp_c), tc_core=float(tc_c), sp_full=float(sp_f), tc_full=float(tc_f)))
    summ = {k: float(np.mean([r[k] for r in rows])) for k in ('sp_core', 'tc_core', 'sp_full', 'tc_full')}
    print(f"   {method}: core SP {summ['sp_core']:.3f} TC {summ['tc_core']:.3f} | full SP {summ['sp_full']:.3f} TC {summ['tc_full']:.3f}  ({time.time()-t_all:.0f}s)", flush=True)
    result[method] = dict(summary=summ, rows=rows)
    json.dump(result, open(args.out, 'w'), indent=1)
print('wrote', args.out)
