#!/usr/bin/env python3
"""Fit TKF92 (F=1) / MixFrag (F=2) indel parameters from the BAliBASE REFERENCE alignments by the
summarised-count EM of the paper (sec:cherryem): every within-family pair of reference rows is
reduced to match-run and ordering-summed gap-block counts per branch-length bin, and EM runs in time
independent of the number of pairs. Branch lengths per pair come from a JSON cache of maximum-
likelihood times (written by refit_indel_on_balibase.py); pairs without a cached time get the median.
Substitution model fixed at LG08. This is a disclosed leak of the benchmark's reference alignments.

Usage: fit_counts_on_balibase_refs.py --F 2 --t-cache experiments/balibase_indel_refit/tkf92_bali3pdbm_tcache.json \
         --out experiments/balibase_indel_refit/mixfrag_F2_bali3pdbm_refs.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PY)
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.util.msa_benchmark import parse_fasta
from build_tkf92_cherry_counts import geom_bin_edges, discretize_tau, AA
from build_mixfrag_cherry_counts import count_cherry_mixfrag, N_SCALARS
from tkfmixdom.jax.train.mixfrag_cherry_em import em_fit

ap = argparse.ArgumentParser()
ap.add_argument('--F', type=int, required=True)
ap.add_argument('--balibase-dir', default=os.path.expanduser('~/bio-datasets/data/balibase/bali3pdbm'))
ap.add_argument('--t-cache', required=True)
ap.add_argument('--n-tau-bins', type=int, default=32)
ap.add_argument('--n-iter', type=int, default=400)
ap.add_argument('--max-gap', type=int, default=120, help='drop gap blocks with more than this many residues (bounds the gap grid; the dropped mass is logged)')
ap.add_argument('--log-every', type=int, default=20)
ap.add_argument('--out', required=True)
args = ap.parse_args()
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

tc = json.load(open(args.t_cache)); t_default = float(np.median(list(tc.values())))
edges, centers = geom_bin_edges(args.n_tau_bins)
nb = args.n_tau_bins
match_counts = np.zeros((nb, AA, AA), np.int64); singlet_counts = np.zeros((nb, AA), np.int64); scalars = np.zeros((nb, N_SCALARS), np.int64)
match_run, gap_to_match, gap_to_end = {}, {}, {}
ref_dir = os.path.join(args.balibase_dir, 'ref'); n_pairs = n_missing_t = 0
for fam in sorted(os.listdir(ref_dir)):
    aln = parse_fasta(os.path.join(ref_dir, fam)); names = list(aln)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            t = tc.get(f'{fam}|{names[a]}|{names[b]}', tc.get(f'{fam}|{names[b]}|{names[a]}'))
            if t is None: t = t_default; n_missing_t += 1
            tb = int(discretize_tau(float(t), edges))
            count_cherry_mixfrag(aln[names[a]], aln[names[b]], tb, match_counts, singlet_counts, scalars, match_run, gap_to_match, gap_to_end)
            n_pairs += 1
log(f"{n_pairs} reference pairs from {len(os.listdir(ref_dir))} families ({n_missing_t} without a cached time -> median {t_default:.3f}); "
    f"{len(match_run)} match-run keys, {len(gap_to_match)} gap->match keys, {len(gap_to_end)} gap->end keys")
agg = dict(match_counts=match_counts, singlet_counts=singlet_counts, scalars=scalars, tau_centers=centers, tau_edges=edges,
           n_tau_bins=nb, n_pairs=n_pairs, match_run=match_run, gap_to_match=gap_to_match, gap_to_end=gap_to_end)
Q, pi = rate_matrix_lg(); Q = np.asarray(Q); pi = np.asarray(pi)
init = (dict(lam=0.04428, mu=0.04521, exts=[0.677], weights=[1.0]) if args.F == 1
        else dict(lam=0.03392, mu=0.03469, exts=[0.41396, 0.86031], weights=[0.70086, 0.29914]))
res = em_fit(agg, n_fragtypes=args.F, Q=Q, pi=pi, init_lam=init['lam'], init_mu=init['mu'], init_exts=init['exts'], init_weights=init['weights'],
             n_iter=args.n_iter, tol=-1e12, max_gap=args.max_gap, use_logspace=True, log_fn=log, log_every=args.log_every)
out = dict(model='mixfrag', F=args.F, lam=float(res['lam']), mu=float(res['mu']), exts=[float(e) for e in res['exts']], weights=[float(w) for w in res['weights']],
           n_pairs=n_pairs, balibase_dir=args.balibase_dir, source='reference alignments, summarised-count EM', max_gap=args.max_gap, n_iter=args.n_iter, t_cache=args.t_cache,
           final_ll=float(res['history'][-1]['ll']) if res.get('history') and isinstance(res['history'][-1], dict) and 'll' in res['history'][-1] else None)
json.dump(out, open(args.out, 'w'), indent=1, default=float); log(f"wrote {args.out}: lam {out['lam']:.5f} mu {out['mu']:.5f} exts {np.round(out['exts'],4).tolist()} w {np.round(out['weights'],3).tolist()}")
