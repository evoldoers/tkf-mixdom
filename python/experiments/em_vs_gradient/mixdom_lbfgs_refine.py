#!/usr/bin/env python3
"""Late-stage L-BFGS refinement of a MixDom fit on a fixed training subset.

L-BFGS needs a deterministic objective, so instead of the minibatch stream
it maximises the MAP objective (observed LL + log-prior, the quantity svi-BW
logs) on a fixed breadth-sampled subset of the training pairs.  Each
function evaluation is one E-step over the subset (`_process_pairs_batched`,
the same code svi-BW and Adam use); the gradient is the gradient of the
Q-function at the current parameters (Fisher's identity), i.e. the custom-VJP
route, taken in the unconstrained parameterisation of adam_train.

Usage:
  mixdom_lbfgs_refine.py --init <maraschino-format init.npz> --n-sub 20000 --maxfun 80 \
      --out pfam/lbfgs_d3f1_from1500.npz --template pfam/svibw_d3f1_perdomain_v4_fix_1500.npz
"""
import argparse, json, os, sys, time, types
import numpy as np
os.environ.setdefault("JAX_ENABLE_X64", "1")
PY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, PY)
_argv = sys.argv; sys.argv = ['x']
import jax, jax.numpy as jnp, scipy.optimize
import train_pfam
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.models.mixdom import state_types as mixdom_state_types
from tkfmixdom.jax.simulate.msa import alignment_to_states
from tkfmixdom.jax.train.adam_train import to_unconstrained, to_constrained, adam_loss_from_delta
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True); ap.add_argument('--template', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--n-sub', type=int, default=20000)
ap.add_argument('--maxfun', type=int, default=80); ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--precompiled', default='pfam/precompiled_v3_train'); ap.add_argument('--batch', type=int, default=2000)
args = ap.parse_args()
PRIORS = dict(ins_prior=(2.0, 10.0), del_prior=(2.0, 10.0), dom_dirichlet=1.5, frag_dirichlet=1.5, ext_alpha=2.0, ext_beta=3.0)
tpl = np.load(args.template, allow_pickle=True); cfg = json.loads(str(tpl['_config']))
N_DOM, N_FRAG = int(cfg['n_dom']), int(cfg['n_frag'])
params0 = train_pfam._load_maraschino_as_train_params(args.init, N_DOM, N_FRAG)
Q_lg, pi_lg = rate_matrix_lg(); st = mixdom_state_types(N_DOM, N_FRAG); N = int(np.asarray(st).shape[0])
args_ns = types.SimpleNamespace(**{k: list(v) if isinstance(v, tuple) else v for k, v in PRIORS.items()}, classdist_dirichlet=1.0, fb_scan_mode='auto')

# fixed breadth-first subset (one pair per family first), seeded
src = train_pfam.PrecompiledPairSource(args.precompiled)
from collections import defaultdict
byfam = defaultdict(list)
for it in src._decode_all():
    byfam[it[6]].append(it[:6])
fams = sorted(byfam); rng = np.random.default_rng(args.seed); rng.shuffle(fams)
for f in fams: rng.shuffle(byfam[f])
sub = []
while len(sub) < args.n_sub and any(byfam[f] for f in fams):
    for f in fams:
        if byfam[f]: sub.append(byfam[f].pop())
        if len(sub) >= args.n_sub: break
print(f"subset: {len(sub)} pairs from {len(set(fams))} families", flush=True)

def estep(params):
    p = {k: (np.asarray(v) if hasattr(v, 'shape') else v) for k, v in params.items()}
    tot_ll = 0.0; delta = None
    for i in range(0, len(sub), args.batch):
        d = train_pfam._process_pairs_batched(sub[i:i+args.batch], p, st, Q_lg, pi_lg, N, alignment_to_states,
                                              dom_Qs=p['dom_Qs'], dom_pis=p['dom_pis'], n_dom=N_DOM, n_frag=N_FRAG,
                                              fb_scan_mode=train_pfam._resolve_fb_scan_mode(args_ns, N))
        tot_ll += float(d['total_ll'])
        if delta is None:
            delta = {k: (np.array(v, dtype=float) if isinstance(v, (np.ndarray, list, float, int)) else v) for k, v in d.items() if k != 'exact_ss'}
            delta['exact_ss'] = {k: np.array(v, dtype=float) for k, v in d['exact_ss'].items()}
        else:
            for k, v in d.items():
                if k == 'exact_ss':
                    for kk, vv in v.items(): delta['exact_ss'][kk] = delta['exact_ss'][kk] + np.asarray(vv, dtype=float)
                elif isinstance(v, (np.ndarray, list, float, int)) and k != 'n_files':
                    delta[k] = delta[k] + np.asarray(v, dtype=float)
    delta['total_ll'] = tot_ll
    return delta

uc0 = to_unconstrained(params0, N_DOM, N_FRAG)
keys = list(uc0.keys()); shapes = {k: np.asarray(uc0[k]).shape for k in keys}
def pack(uc): return np.concatenate([np.asarray(uc[k]).ravel() for k in keys])
def unpack(x):
    out, i = {}, 0
    for k in keys:
        n = int(np.prod(shapes[k])) if shapes[k] else 1; out[k] = jnp.asarray(x[i:i+n].reshape(shapes[k])); i += n
    return out
loss_grad = jax.jit(jax.value_and_grad(lambda uc, dj: adam_loss_from_delta(uc, dj, N_DOM, N_FRAG, **PRIORS)))
hist = []; t0 = time.time()
def fun(x):
    uc = unpack(x); p = to_constrained(uc, N_DOM, N_FRAG)
    p = {k: (np.asarray(v) if hasattr(v, 'shape') else v) for k, v in p.items()}
    delta = estep(p); n = float(delta['n_pairs'])
    lp = float(train_pfam._log_prior(p, args_ns, Q_lg=Q_lg, pi_lg=pi_lg))
    obj = (float(delta['total_ll']) + lp) / n                       # MAP objective per pair (to maximise)
    dj = jax.tree.map(jnp.asarray, delta)
    _, g = loss_grad(uc, dj)                                        # gradient of -(Q+prior)/n_pairs == -grad(obj)
    gvec = pack({k: np.asarray(g[k]) for k in keys})
    hist.append(dict(eval=len(hist) + 1, wall=time.time() - t0, map_per_pair=obj, obs_ll=float(delta['total_ll'])))
    print(f"  eval {len(hist):3d}: MAP/pair={obj:.4f} obs_LL={float(delta['total_ll']):.1f} ({time.time()-t0:.0f}s)", flush=True)
    return -obj, gvec
res = scipy.optimize.minimize(fun, pack(uc0), jac=True, method='L-BFGS-B',
                              options=dict(maxfun=args.maxfun, maxiter=args.maxfun, ftol=0.0, gtol=1e-8, maxcor=20))
print('L-BFGS:', res.message, 'nit', res.nit, 'nfev', res.nfev, flush=True)
best = max(hist, key=lambda h: h['map_per_pair'])
p = to_constrained(unpack(res.x), N_DOM, N_FRAG)
p = {k: (np.asarray(v) if hasattr(v, 'shape') else v) for k, v in p.items()}
d = {k: p[k] for k in ('main_ins', 'main_del', 'dom_ins', 'dom_del', 'dom_weights', 'frag_weights', 'ext_rates', 'dom_Qs', 'dom_pis', 'dom_S_exch')}
AA = np.asarray(p['dom_pis']).shape[1]
d['class_S_exch'] = np.asarray(p['dom_S_exch']).copy(); d['class_pis'] = np.asarray(p['dom_pis']).copy()
d['classdist'] = np.broadcast_to(np.eye(N_DOM)[:, None, :], (N_DOM, N_FRAG, N_DOM)).copy(); d['n_classes_frag'] = np.int64(N_DOM)
for k in ('_config', 'em_iter', 't_rep', 'log_probs', 'step_file_idx'):
    if k in tpl.files: d[k] = tpl[k]
np.savez_compressed(args.out, **d)
json.dump(dict(args=vars(args), history=hist, message=str(res.message), nit=int(res.nit), nfev=int(res.nfev)), open(args.out.replace('.npz', '_history.json'), 'w'), indent=1)
print('saved', args.out, flush=True)
