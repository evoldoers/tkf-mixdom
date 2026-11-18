#!/usr/bin/env python3
"""Path-conditioned held-out log-likelihood per pair for TKF92 / MixFrag with FIXED
parameters, on a precompiled shard corpus (the Table 2 convention for those models:
the summarised-count likelihood of the given alignment, fragtypes marginalised).
Reuses mixfrag_from_shards.build_agg_from_shards for the count tensors.
Usage: score_pathcond_shards_mixfrag.py --shards pfam/precompiled_v3_test --lam .. --mu .. --exts a,b --weights w1,w2 --out x.json
"""
import os, sys, json, argparse
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from mixfrag_from_shards import build_agg_from_shards
from tkfmixdom.jax.train.mixfrag_cherry_em import prep_bins, corpus_indel_loglik, corpus_substitution_loglik
from train_mixfrag_cherry_em import _substitution_model
ap = argparse.ArgumentParser()
ap.add_argument('--shards', required=True); ap.add_argument('--lam', type=float, required=True); ap.add_argument('--mu', type=float, required=True)
ap.add_argument('--exts', required=True); ap.add_argument('--weights', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n-tau-bins', type=int, default=32); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--tau-floor', type=float, default=0.01)
ap.add_argument('--max-run', type=int, default=700); ap.add_argument('--max-gap', type=int, default=50); ap.add_argument('--subst', default='lg')
a = ap.parse_args()
exts = [float(x) for x in a.exts.split(',')]; weights = [float(x) for x in a.weights.split(',')]
agg = build_agg_from_shards(a.shards, a.n_tau_bins, a.workers, a.tau_floor, a.max_run)
Q, pi = _substitution_model(a.subst); bins = prep_bins(agg, max_gap=a.max_gap)
ind = float(corpus_indel_loglik(bins, a.lam, a.mu, np.asarray(exts), np.asarray(weights))); sub = float(corpus_substitution_loglik(bins, Q, pi)); n = agg['n_pairs']
res = dict(shards=a.shards, lam=a.lam, mu=a.mu, exts=exts, weights=weights, n_pairs=n, indel_ll=ind, sub_ll=sub, ll_per_pair=(ind + sub) / n)
json.dump(res, open(a.out, 'w'), indent=1); print(f"PATH-CONDITIONED (summarised counts): n_pairs={n} LL/pair={(ind+sub)/n:.6f} (indel {ind/n:.4f}, subst {sub/n:.4f})")
