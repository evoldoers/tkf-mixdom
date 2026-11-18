#!/usr/bin/env python3
"""Re-score cached BAliBASE pair posteriors with FSA at several gap factors.

Reads the raw per-family pair posteriors saved by the BAliBASE harness in
``~/.cache/tkf-mixdom-balibase/<method>/<family>.npz`` (no 2D DP is rerun),
builds the MSA by sequence annealing at each requested gap factor with the
same builder, seed and anneal iterations as ``expected_pairwise_balibase.py``
(``_build_msa_with_gap_factor``), and scores SP / TC against the BAliBASE
reference on core columns.  Ignores the cache's ``params_key`` (some older
caches did not record one); pass ``--params-key`` to enforce it.

Usage (from ~/tkf-mixdom/python):
  .venv/bin/python experiments/fsa_gap_factor_from_cache.py \
      --method cherryml_C20_persite --gap-factors 1 0 \
      --out experiments/expected_balibase/gapfactor_cherryml_C20_persite.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from tkfmixdom.util.msa_benchmark import parse_fasta, sp_tc_score        # noqa: E402
from tkfmixdom.util import balibase_pair_cache as ppcache                # noqa: E402
from fsa_mixdom_pairhmm_oxbench import encode_seq                        # noqa: E402
from expected_pairwise_balibase import _build_msa_with_gap_factor, _msa_to_pair_indicators  # noqa: E402
from tkfmixdom.util.expected_pair_f1 import expected_family_f1          # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--params-key', default=None)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--balibase-dir',
                   default=str(Path('~/bio-datasets/data/balibase/bali3pdbm').expanduser()))
    p.add_argument('--gap-factors', type=float, nargs='+', default=[1.0, 0.0])
    p.add_argument('--fsa-anneal-iters', type=int, default=3)
    p.add_argument('--fsa-seed', type=int, default=42)
    p.add_argument('--algo', choices=['anneal', 'progressive'], default='anneal', help='MSA builder: AMAP sequence annealing (default) or MEA progressive alignment')
    p.add_argument('--refine-iters', type=int, default=0, help='progressive only: rounds of random-bipartition iterative refinement')
    p.add_argument('--consistency-iters', type=int, default=0, help='ProbCons consistency transformation of the cached posteriors, applied this many times before annealing (0 = off)')
    p.add_argument('--posterior-floor', type=float, default=0.0, help='progressive only: zero posteriors below this before decoding (MUSCLE: 0.01)')
    p.add_argument('--tree-first', action='store_true', help='progressive only: build the guide tree before the consistency transformation (MUSCLE order)')
    p.add_argument('--linkage', choices=['upgma', 'single', 'nj'], default='upgma', help='progressive only: guide tree by UPGMA, single linkage (MUSCLE) or midpoint-rooted neighbour joining')
    p.add_argument('--tree-dist-json', default=None, help='progressive only: JSON {"family|name1|name2": distance} of pairwise branch lengths to build the guide tree from (pairs without an entry get the family maximum; a family with none falls back to expected-accuracy distances)')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    base = Path(args.cache_dir) if args.cache_dir else ppcache.DEFAULT_CACHE_DIR
    mdir = base / args.method
    fams = sorted(q.stem for q in mdir.glob('*.npz'))
    in_dir = os.path.join(args.balibase_dir, 'in')
    ref_dir = os.path.join(args.balibase_dir, 'ref')
    tdist = json.load(open(args.tree_dist_json)) if args.tree_dist_json else None
    print(f"{args.method}: {len(fams)} cached families in {mdir}", flush=True)
    rows = []
    t_all = time.time()
    for fam in fams:
        meta = json.loads((mdir / f'{fam}.json').read_text())
        if args.params_key is not None and meta.get('params_key') != args.params_key:
            print(f"  {fam}: params_key mismatch, skipped", flush=True)
            continue
        raw = parse_fasta(os.path.join(in_dir, fam))
        ref_aln = parse_fasta(os.path.join(ref_dir, fam))
        int_seqs = {n: encode_seq(s) for n, s in raw.items()}
        int_seqs = {n: s for n, s in int_seqs.items() if len(s) > 0}
        names = list(int_seqs.keys())
        d = np.load(mdir / f'{fam}.npz', allow_pickle=True)
        # Two cache schemas: the harness one (names/pairs/post_k in the npz)
        # and the per-site C20 one (names/pairs in the json, p_i_j in the npz).
        cached_names = [str(n) for n in (d['names'] if 'names' in d.files else meta['names'])]
        pairs = np.asarray(d['pairs'] if 'pairs' in d.files else meta['pairs'])
        idx = {n: names.index(n) for n in cached_names}   # cache index -> fasta index
        pp = {}
        for k, (i, j) in enumerate(pairs.tolist()):
            key = f'post_{k}' if f'post_{k}' in d.files else f'p_{i}_{j}'
            pp[(idx[cached_names[i]], idx[cached_names[j]])] = np.asarray(d[key])
        guide = None
        if tdist is not None:
            n = len(names); guide = np.full((n, n), np.nan); have = 0
            for a in range(n):
                for b in range(a + 1, n):
                    v = tdist.get(f'{fam}|{names[a]}|{names[b]}', tdist.get(f'{fam}|{names[b]}|{names[a]}'))
                    if v is not None: guide[a, b] = guide[b, a] = float(v); have += 1
            if have == 0: guide = None
            else:
                mx = np.nanmax(guide); guide = np.where(np.isnan(guide), mx, guide); np.fill_diagonal(guide, 0.0)
        row = dict(family=fam, n_seqs=len(names), n_pairs=len(pp), kind=str(meta.get('kind', 'soft')))
        for g in args.gap_factors:
            t0 = time.time()
            _, msa_strings = _build_msa_with_gap_factor(
                int_seqs, pp, gap_factor=float(g),
                n_anneal=args.fsa_anneal_iters, seed=args.fsa_seed, consistency_iters=args.consistency_iters, algo=args.algo, refine_iters=args.refine_iters,
                posterior_floor=args.posterior_floor, tree_before_consistency=args.tree_first, linkage=args.linkage, guide_dist=guide)
            sp, tc = sp_tc_score(msa_strings, ref_aln, core_only=True)
            tag = f"g{g:g}"
            row[f'msa_sp_{tag}'] = float(sp)
            row[f'msa_tc_{tag}'] = float(tc)
            # pooled F1 of the alignment's residue pairs (core blocks as positives),
            # the quantity in the paper's F1 column
            hard = _msa_to_pair_indicators(int_seqs, names, [tuple(k) for k in pp.keys()], msa_strings)
            row[f'micro_hard_{tag}'] = expected_family_f1(hard, ref_aln, names, core_only=True)['micro']
            row[f'time_{tag}'] = time.time() - t0
        rows.append(row)
        print(f"  {fam} n={len(names)} " + " ".join(
            f"g{g:g}: SP={row[f'msa_sp_g{g:g}']:.3f} TC={row[f'msa_tc_g{g:g}']:.3f}" for g in args.gap_factors),
            flush=True)
        json.dump(dict(method=args.method, gap_factors=args.gap_factors, rows=rows),
                  open(args.out, 'w'), indent=1)
    summ = {}
    for g in args.gap_factors:
        tag = f"g{g:g}"
        e = sum(r[f'micro_hard_{tag}']['e_tp'] for r in rows); tm = sum(r[f'micro_hard_{tag}']['total_mass'] for r in rows)
        gold = sum(r[f'micro_hard_{tag}']['gold'] for r in rows); pr, rc = e / tm, e / gold
        summ[tag] = dict(sp=float(np.mean([r[f'msa_sp_{tag}'] for r in rows])),
                         tc=float(np.mean([r[f'msa_tc_{tag}'] for r in rows])),
                         f1_pairs=float(2 * pr * rc / (pr + rc)))
    json.dump(dict(method=args.method, gap_factors=args.gap_factors, n_families=len(rows),
                   summary=summ, rows=rows), open(args.out, 'w'), indent=1)
    print(f"summary over {len(rows)} families: {summ}  ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
