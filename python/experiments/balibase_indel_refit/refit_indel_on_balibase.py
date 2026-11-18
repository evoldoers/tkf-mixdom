#!/usr/bin/env python3
"""Refit the TKF92 / MixFrag indel parameters on BAliBASE's UNALIGNED sequences, the way
ProbCons fitted its transition probabilities (unsupervised EM on unaligned BAliBASE sequences;
MUSCLE 5 inherits them). Substitution model fixed at LG08, as everywhere in the paper.

Data: every within-family pair of the benchmark's input sequences (no reference alignments).
Model: the 2D pair HMM, so the E-step sums over ALL alignments of each pair (the
``compare_optimizers.py`` machinery). Branch lengths: one t per pair, re-estimated every
``--t-every`` rounds by maximum likelihood on a log grid; the global indel parameters by the
closed-form M-step (flat priors -> MLE). Output JSON has lam / mu / exts / weights in the format
``fsa_c20_balibase.py --indel-params`` reads, plus the trajectory and the per-pair times.

This is a disclosed test-set leak of 3 (TKF92) or 5 (MixFrag F=2) indel parameters.

Usage (from ~/tkf-mixdom/python):
  .venv/bin/python experiments/balibase_indel_refit/refit_indel_on_balibase.py --model mixfrag \
      --out experiments/balibase_indel_refit/mixfrag_F2_bali3pdbm.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("TKFMIXDOM_MAX_PAD", "1024")
HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PY); sys.path.insert(0, os.path.join(PY, 'experiments')); sys.path.insert(0, os.path.join(PY, 'experiments', 'em_vs_gradient'))
import jax, jax.numpy as jnp
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.util.msa_benchmark import parse_fasta
from fsa_mixdom_pairhmm_oxbench import encode_seq
from compare_optimizers import Model, make_chunks, em_pass, em_mstep
from tkfmixdom.jax.train.tkf92_svi_bw import _estep_pair_tkf92_forward_only_core
from tkfmixdom.jax.train.mixfrag_svi_bw import _estep_pair_mixfrag_forward_only_core

ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=['tkf92', 'mixfrag'], required=True)
ap.add_argument('--balibase-dir', default=os.path.expanduser('~/bio-datasets/data/balibase/bali3pdbm'))
ap.add_argument('--init-json', default=None, help='initial lam/mu/exts/weights (default: Table 2 values)')
ap.add_argument('--rounds', type=int, default=40)
ap.add_argument('--t-every', type=int, default=2)
ap.add_argument('--t-grid', type=int, default=20)
ap.add_argument('--chunk', type=int, default=8)
ap.add_argument('--max-len', type=int, default=1000)
ap.add_argument('--families', type=int, default=0, help='debug: only the first N families')
ap.add_argument('--t-cache', default=None, help='JSON file of per-pair times; loaded if present (skips the initial grid), written after every re-estimation')
ap.add_argument('--out', required=True)
args = ap.parse_args()
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

Q, pi = rate_matrix_lg(); Q = np.asarray(Q); pi = np.asarray(pi)
model = Model(args.model)
if args.init_json:
    p = json.load(open(args.init_json)); p = p.get('indel_params', p)
else:
    p = (dict(lam=0.04428, mu=0.04521, exts=[0.677], weights=[1.0]) if args.model == 'tkf92'
         else dict(lam=0.03392, mu=0.03469, exts=[0.41396, 0.86031], weights=[0.70086, 0.29914]))
p = dict(lam=float(p['lam']), mu=float(p['mu']), exts=[float(e) for e in p['exts']], weights=[float(w) for w in p['weights']])

# ---- data: all within-family pairs of the benchmark inputs ----
in_dir = os.path.join(args.balibase_dir, 'in')
pairs, meta, n_wild = [], [], 0
for fam in sorted(os.listdir(in_dir))[: (args.families or None)]:
    seqs = parse_fasta(os.path.join(in_dir, fam)); names = list(seqs)
    enc = {}
    for n in names:
        s = encode_seq(seqs[n]); n_wild += int((s == 20).sum()); s = np.where(s == 20, 0, s).astype(np.int32)   # X -> A
        enc[n] = s
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = enc[names[a]], enc[names[b]]
            if 0 < len(x) <= args.max_len and 0 < len(y) <= args.max_len:
                pairs.append((x, y, 1.0)); meta.append((fam, names[a], names[b]))
log(f"{len(pairs)} pairs from {len(os.listdir(in_dir))} families ({n_wild} wildcard residues mapped to A); model {args.model}; init {p}")

# ---- per-pair forward-only log-likelihood at a given t, for the t grid ----
Qj, pij = jnp.asarray(Q), jnp.asarray(pi)
def _lp_core(x, y, t, Lx, Ly, lam, mu, exts, weights):
    if args.model == 'mixfrag':
        return _estep_pair_mixfrag_forward_only_core(x, y, t, Lx, Ly, lam, mu, exts, weights, Qj, pij)
    return _estep_pair_tkf92_forward_only_core(x, y, t, Lx, Ly, lam, mu, exts[0], Qj, pij)
_lp_chunk = jax.jit(jax.vmap(_lp_core, in_axes=(0, 0, 0, 0, 0, None, None, None, None)))
t_grid = np.exp(np.linspace(np.log(0.03), np.log(6.0), args.t_grid))

def reestimate_t(pairs, p):
    """ML t per pair on the log grid. Pairs are bucketed by padded shape exactly as make_chunks does."""
    from tkfmixdom.jax.train.tkf92_svi_bw import _bin_bucket_pairs, _stack_bucket
    from tkfmixdom.jax.dp.hmm import _pad_to_bin
    order = {}
    for i, (x, y, _) in enumerate(pairs):
        order.setdefault((_pad_to_bin(len(x)), _pad_to_bin(len(y))), []).append(i)
    best_t = np.array([t for _, _, t in pairs], dtype=np.float64)
    lam, mu, exts, weights = p['lam'], p['mu'], jnp.asarray(p['exts']), jnp.asarray(p['weights'])
    for key, idxs in order.items():
        for c0 in range(0, len(idxs), args.chunk):
            ids = idxs[c0:c0 + args.chunk]
            lst = list(_bin_bucket_pairs([pairs[i] for i in ids]).values())[0]
            n_real = len(lst)
            if n_real < args.chunk: lst = lst + [lst[-1]] * (args.chunk - n_real)
            xs, ys, _, Lx, Ly = _stack_bucket(lst)
            lps = np.stack([np.asarray(_lp_chunk(jnp.asarray(xs), jnp.asarray(ys), jnp.full(args.chunk, tg), jnp.asarray(Lx), jnp.asarray(Ly), lam, mu, exts, weights)) for tg in t_grid])  # (G, chunk)
            for k, i in enumerate(ids): best_t[i] = t_grid[int(np.argmax(lps[:, k]))]
    return [(x, y, float(best_t[i])) for i, (x, y, _) in enumerate(pairs)]

traj = []
t0 = time.time()
def save_t(pairs):
    if args.t_cache: json.dump({f'{f}|{a}|{b}': t for (f, a, b), (_, _, t) in zip(meta, pairs)}, open(args.t_cache, 'w'))
if args.t_cache and os.path.exists(args.t_cache):
    tc = json.load(open(args.t_cache)); pairs = [(x, y, float(tc.get(f'{f}|{a}|{b}', t))) for (f, a, b), (x, y, t) in zip(meta, pairs)]
    log(f"loaded per-pair t from {args.t_cache}: median {np.median([t for _,_,t in pairs]):.3f}")
else:
    pairs = reestimate_t(pairs, p); save_t(pairs); log(f"initial t: median {np.median([t for _,_,t in pairs]):.3f} ({time.time()-t0:.0f}s)")
for r in range(args.rounds):
    chunks, nb = make_chunks(pairs, args.chunk)
    ll, suff = em_pass(model, chunks, p, Q, pi); ll /= len(pairs)
    traj.append(dict(round=r, ll=ll, params=dict(p), t_median=float(np.median([t for _, _, t in pairs]))))
    log(f"round {r:3d} LL/pair {ll:.4f} lam {p['lam']:.5f} mu {p['mu']:.5f} exts {np.round(p['exts'],4).tolist()} w {np.round(p['weights'],3).tolist()} ({time.time()-t0:.0f}s)")
    p = em_mstep(model, suff, p)
    if (r + 1) % args.t_every == 0: pairs = reestimate_t(pairs, p); save_t(pairs)
out = dict(model=args.model, lam=p['lam'], mu=p['mu'], exts=p['exts'], weights=p['weights'], n_pairs=len(pairs),
           balibase_dir=args.balibase_dir, rounds=args.rounds, trajectory=traj,
           pair_times=[dict(family=f, a=a, b=b, t=t) for (f, a, b), (_, _, t) in zip(meta, pairs)])
json.dump(out, open(args.out, 'w'), indent=1); log(f"wrote {args.out}: {dict(lam=p['lam'], mu=p['mu'], exts=p['exts'], weights=p['weights'])}")
