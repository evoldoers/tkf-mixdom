#!/usr/bin/env python3
"""Path-conditioned held-out log-likelihood per pair of a MixDom train_pfam
checkpoint on a precompiled pair set (default precompiled_v3_val): the 1D
forward over each pair's given alignment, via the same batched E-step
(`_process_pairs_batched`) that svi-BW uses, summed over all pairs.

Usage: score_pathcond_val.py --checkpoint pfam/svibw_d3f1_perdomain_v4_fix.npz [--precompiled pfam/precompiled_v3_val]
"""
import argparse, json, os, sys, time, types
import numpy as np
os.environ.setdefault("JAX_ENABLE_X64", "1")
PY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, PY)
_argv = sys.argv; sys.argv = ['x']
import train_pfam
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.models.mixdom import state_types as mixdom_state_types
from tkfmixdom.jax.simulate.msa import alignment_to_states
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--precompiled', default='pfam/precompiled_v3_val')
ap.add_argument('--batch', type=int, default=2000)
ap.add_argument('--out', default=None)
ap.add_argument('--stride', type=int, default=1, help='score every k-th pair of the set (deterministic subsample; 1 = all)')
args = ap.parse_args()
(params, *_rest) = train_pfam._load_checkpoint(args.checkpoint)
cfg = json.loads(str(np.load(args.checkpoint, allow_pickle=True)['_config']))
n_dom, n_frag = int(cfg['n_dom']), int(cfg['n_frag'])
Q_lg, pi_lg = rate_matrix_lg(); st = mixdom_state_types(n_dom, n_frag); N = int(np.asarray(st).shape[0])
cd = train_pfam._get_classdist_params(params) if int(params.get('n_classes', 1)) > 1 else None
src = train_pfam.PrecompiledPairSource(args.precompiled)
pairs = [p[:6] for p in src._decode_all()][::max(1, args.stride)]
ns = types.SimpleNamespace(fb_scan_mode='auto')
print(f"{args.checkpoint}: n_dom={n_dom} n_frag={n_frag} n_classes={params.get('n_classes', 1)}; {len(pairs)} pairs from {args.precompiled}", flush=True)
tot, n, t0 = 0.0, 0, time.time()
for i in range(0, len(pairs), args.batch):
    b = pairs[i:i + args.batch]
    d = train_pfam._process_pairs_batched(b, params, st, Q_lg, pi_lg, N, alignment_to_states,
                                          dom_Qs=params.get('dom_Qs'), dom_pis=params.get('dom_pis'),
                                          classdist_params=cd, n_dom=n_dom, n_frag=n_frag,
                                          fb_scan_mode=train_pfam._resolve_fb_scan_mode(ns, N))
    tot += float(d['total_ll']); n += int(d['n_pairs'])
    if (i // args.batch) % 10 == 0:
        print(f"  {n}/{len(pairs)} pairs, LL/pair so far {tot/n:.4f} ({time.time()-t0:.0f}s)", flush=True)
res = dict(checkpoint=args.checkpoint, precompiled=args.precompiled, stride=args.stride, n_pairs=n, total_ll=tot, ll_per_pair=tot / n)
print(f"PATH-CONDITIONED: n_pairs={n} total_ll={tot:.4f} LL/pair={tot/n:.6f}", flush=True)
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)
