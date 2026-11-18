#!/usr/bin/env python3
"""Late-stage L-BFGS refinement of a MixFrag-S fit on a fixed training subset
(the MixFrag-S analogue of mixdom_lbfgs_refine.py).

L-BFGS needs a deterministic objective, so instead of the minibatch stream it
maximises the recipe's MAP objective (observed 1D path-conditioned LL +
log-prior; see mixfrag_s_tail_common) on a fixed, seeded, breadth-sampled
subset of the training pairs (one pair per family first).  Each function
evaluation is one 1D E-step over the subset; the gradient is the gradient of
the Q-function at the current parameters (Fisher's identity: the closed-form
custom-VJP route) in the unconstrained parameterisation of the Adam tail.

Objective per evaluation: (LL_subset + log_prior) / n_sub (to maximise).
Every evaluation logs the objective, lam, mu, exts, w and each fragtype's pi.
Output: the contract npz (``--out``; scipy's final iterate) plus
``<out>_history.json``.  (No --resume: scipy's L-BFGS-B state is not cheaply
restartable; 80 evaluations on 20,000 pairs is a few hours at most.)

Recipe (launch_retrain_nobali.sh mirror): --n-sub 20000 --maxfun 80 --seed 42.

Usage:
  mixfrag_s_lbfgs_refine.py --init pfam/svibw_mixfrags_F2_nobali_1620.npz \
      --precompiled pfam/precompiled_v3_train_nobali --n-sub 20000 --maxfun 80 \
      --out pfam/lbfgs_mixfrags_F2_nobali_from1620.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mixfrag_s_tail_common as C                                   # noqa: E402
import jax                                                          # noqa: E402
import jax.numpy as jnp                                             # noqa: E402
import scipy.optimize                                               # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--init', required=True, help='contract npz to start from')
ap.add_argument('--out', required=True, help='output contract npz')
ap.add_argument('--precompiled', default='pfam/precompiled_v3_train_nobali')
ap.add_argument('--n-sub', type=int, default=20000)
ap.add_argument('--maxfun', type=int, default=80)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--batch', type=int, default=2000,
                help='(kept for CLI symmetry with mixdom_lbfgs_refine.py; the 1D '
                     'E-step is chunked per padding bin by --chunk-max / --chunk-mem-gb)')
ap.add_argument('--log-file', default=None)
C.add_prior_args(ap)
C.add_estep_args(ap)
args = ap.parse_args()

C.setup_jax(args)
t_start = time.time()
log = C.make_logger(args.log_file, t_start)
log(f'MixFrag-S L-BFGS refine: backend={jax.default_backend()} args={vars(args)}')

p0, F, A, ck0 = C.load_contract(args.init)
em_iter0 = int(ck0['em_iter'])
parent_config = json.loads(str(ck0['_config'])) if '_config' in ck0 else {}
log(f'  init {args.init}: F={F} A={A} em_iter={em_iter0} lam={p0["lam"]:.6f} '
    f'mu={p0["mu"]:.6f} exts={C.fmt_vec(p0["exts"])} w={C.fmt_vec(p0["weights"])}')
packer = C.Packer(F, A)
priors = C.priors_from_args(args)
vg, lp_fn = C.make_value_grad(packer, priors)
ek = C.estep_kwargs(args)

sub, n_fam = C.breadth_first_subset(
    C.load_path_pairs(args.precompiled, log, with_fam=True), args.n_sub, args.seed)
n_sub = len(sub)
log(f'subset: {n_sub} pairs from {n_fam} families (seed {args.seed})')

hist = []
t0 = time.time()
x0 = packer.pack(p0)


def fun(x):
    v = jnp.asarray(x, jnp.float64)
    p = packer.natural(v)
    t_e = time.time()
    ll, suff, n = C.estep_pairs(sub, p, **ek)
    t_e = time.time() - t_e
    sj = C.suff_to_jnp(suff)
    _q, g = vg(v, sj, 1.0)                       # d/dv [Q + log_prior] == d/dv [LL + log_prior]
    lp = float(lp_fn(v))
    obj = (ll + lp) / n                          # MAP objective per pair (to maximise)
    gvec = np.asarray(g, np.float64) / n
    hist.append(dict(eval=len(hist) + 1, wall=time.time() - t0, map_per_pair=obj,
                     obs_ll=ll, obs_ll_per_pair=ll / n, log_prior=lp, n_pairs=n,
                     grad_norm=float(np.linalg.norm(gvec)), estep_sec=t_e,
                     lam=p['lam'], mu=p['mu'], exts=p['exts'], weights=p['weights'], pi=p['pi']))
    log(f'  eval {len(hist):3d}: MAP/pair={obj:.4f} obs_LL={ll:.1f} obs_LL/pair={ll / n:.4f} '
        f'logprior={lp:.2f} |g|/pair={hist[-1]["grad_norm"]:.3e} lam={p["lam"]:.6f} '
        f'mu={p["mu"]:.6f} exts={C.fmt_vec(p["exts"])} w={C.fmt_vec(p["weights"])} '
        f'(estep {t_e:.1f}s; {time.time() - t0:.0f}s)')
    for f in range(F):
        log(f'      pi[{f}]={C.fmt_vec(p["pi"][f])}')
    return -obj, -gvec


res = scipy.optimize.minimize(fun, x0, jac=True, method='L-BFGS-B',
                              options=dict(maxfun=args.maxfun, maxiter=args.maxfun,
                                           ftol=0.0, gtol=1e-8, maxcor=20))
log(f'L-BFGS: {res.message} nit={res.nit} nfev={res.nfev} final MAP/pair={-res.fun:.4f}')
best = max(hist, key=lambda h: h['map_per_pair'])
log(f'  best evaluated MAP/pair={best["map_per_pair"]:.4f} at eval {best["eval"]}; '
    f'start {hist[0]["map_per_pair"]:.4f}')
p_end = packer.natural(jnp.asarray(res.x, jnp.float64))
log(f'  final lam={p_end["lam"]:.6f} mu={p_end["mu"]:.6f} exts={C.fmt_vec(p_end["exts"])} '
    f'w={C.fmt_vec(p_end["weights"])}')
for f in range(F):
    log(f'  final pi[{f}]={C.fmt_vec(p_end["pi"][f])}')
config = dict(tail='lbfgs', model='MixFrag-S', dp_mode='path', from_checkpoint=args.init,
              from_em_iter=em_iter0, n_sub=n_sub, n_families=n_fam, maxfun=args.maxfun,
              seed=args.seed, precompiled=args.precompiled, priors=priors,
              message=str(res.message), nit=int(res.nit), nfev=int(res.nfev),
              final_map_per_pair=float(-res.fun), parent_config=parent_config)
C.save_contract(args.out, p_end, F, em_iter0, config, hist)
C.dump_history(args.out.replace('.npz', '_history.json'), dict(args=vars(args), config=config), hist)
log(f'saved {args.out}')
