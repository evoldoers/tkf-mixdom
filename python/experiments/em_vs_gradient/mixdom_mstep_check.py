#!/usr/bin/env python3
"""Is train_pfam's closed-form MixDom M-step an exact maximiser of the
Q-function it is given?  From EM's final parameters theta0:
  1. one E-step gives the sufficient statistics delta (as train_pfam computes them);
  2. theta_em = EM's own next iterate (train_pfam resumed for one iteration);
  3. theta_q  = numerical maximiser of Q(theta; delta) + log-prior (L-BFGS on
     adam_loss_from_delta, the same function Adam differentiates);
  4. compare Q(theta_em) vs Q(theta_q), block by block, and the true MAP
     objective at theta0, theta_em, theta_q (one E-step each).
If Q(theta_em) < Q(theta_q) by more than numerical noise, the closed-form
M-step is not maximising the Q-function; the blocks where theta_em differs
from theta_q say which update is wrong.
"""
import json, os, sys, types
import numpy as np
os.environ.setdefault("JAX_ENABLE_X64", "1")
PY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, PY)
_argv = sys.argv; sys.argv = ['x']
import jax, jax.numpy as jnp, scipy.optimize
import train_pfam
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.models.mixdom import state_types as mixdom_state_types
from tkfmixdom.jax.simulate.msa import alignment_to_states
from tkfmixdom.jax.util.pair_loader import PairStream
from tkfmixdom.jax.train.adam_train import to_unconstrained, to_constrained, adam_loss_from_delta
sys.argv = _argv
MD = 'experiments/em_vs_gradient/results_mixdom'
N_DOM, N_FRAG = 3, 1
PRIORS = dict(ins_prior=(2.0, 10.0), del_prior=(2.0, 10.0), dom_dirichlet=1.5, frag_dirichlet=1.5, ext_alpha=2.0, ext_beta=3.0)
Q_lg, pi_lg = rate_matrix_lg(); st = mixdom_state_types(N_DOM, N_FRAG); N = int(np.asarray(st).shape[0])
pairs = PairStream('pfam/precompiled_sub1000_seed0', seed=0).sample_batch(10**9)
args_ns = types.SimpleNamespace(**{k: list(v) if isinstance(v, tuple) else v for k, v in PRIORS.items()}, classdist_dirichlet=1.0, fb_scan_mode='auto')

def load_ckpt(path):
    d = np.load(path, allow_pickle=True)
    return {k: (float(d[k]) if d[k].shape == () else d[k].copy()) for k in
            ('main_ins','main_del','dom_ins','dom_del','dom_weights','frag_weights','ext_rates','dom_Qs','dom_pis','dom_S_exch')}
def npify(p): return {k: (np.asarray(v) if hasattr(v, 'shape') else v) for k, v in p.items()}
def estep(p):
    p = npify(p)
    return train_pfam._process_pairs_batched(pairs, p, st, Q_lg, pi_lg, N, alignment_to_states, dom_Qs=p['dom_Qs'], dom_pis=p['dom_pis'],
                                             n_dom=N_DOM, n_frag=N_FRAG, fb_scan_mode=train_pfam._resolve_fb_scan_mode(args_ns, N))
def map_obj(p):
    d = estep(p); return float(d['total_ll']) + float(train_pfam._log_prior(npify(p), args_ns, Q_lg=Q_lg, pi_lg=pi_lg)), d

TAG = os.environ.get('MSTEP_TAG', 'em_d3f1')
theta0 = load_ckpt(f'{MD}/{TAG}.npz'); theta_em = load_ckpt(f'{MD}/{TAG}_plus1.npz')
map0, delta = map_obj(theta0); n_pairs = float(delta['n_pairs']); delta_jax = jax.tree.map(jnp.asarray, delta)
Qfun = lambda uc: -float(adam_loss_from_delta(uc, delta_jax, N_DOM, N_FRAG, **PRIORS)) * n_pairs   # Q + log prior (nats)
uc0 = to_unconstrained(theta0, N_DOM, N_FRAG); uc_em = to_unconstrained(theta_em, N_DOM, N_FRAG)
# numerical maximiser of Q from theta0
keys = list(uc0.keys()); shapes = {k: np.asarray(uc0[k]).shape for k in keys}
def pack(uc): return np.concatenate([np.asarray(uc[k]).ravel() for k in keys])
def unpack(x):
    out, i = {}, 0
    for k in keys:
        n = int(np.prod(shapes[k])) if shapes[k] else 1; out[k] = jnp.asarray(x[i:i+n].reshape(shapes[k])); i += n
    return out
vg = jax.jit(jax.value_and_grad(lambda uc: adam_loss_from_delta(uc, delta_jax, N_DOM, N_FRAG, **PRIORS)))
def f(x):
    v, g = vg(unpack(x)); return float(v), pack({k: np.asarray(g[k]) for k in keys})
res = scipy.optimize.minimize(f, pack(uc0), jac=True, method='L-BFGS-B', options=dict(maxiter=5000, maxfun=10000, ftol=0, gtol=1e-9))
uc_q = unpack(res.x); theta_q = npify(to_constrained(uc_q, N_DOM, N_FRAG))
print(f"Q+prior at theta0 (EM final):      {Qfun(uc0):.3f}")
print(f"Q+prior at EM's next iterate:      {Qfun(uc_em):.3f}")
print(f"Q+prior at numerical maximiser:    {Qfun(uc_q):.3f}   (L-BFGS: {res.message}, nit={res.nit})")
# block-wise: start from theta_em and replace one block with theta_q's
for k in keys:
    uc_mix = dict(uc_em); uc_mix[k] = uc_q[k]
    print(f"   replace block {k:22s} in EM iterate by Q-max value: Q+prior = {Qfun(uc_mix):.3f}  (max|diff| = {float(np.max(np.abs(np.asarray(uc_em[k]) - np.asarray(uc_q[k])))):.3g})")
map_em, _ = map_obj(theta_em); map_q, _ = map_obj(theta_q)
print(f"true MAP objective: theta0 {map0:.1f} | EM next iterate {map_em:.1f} | Q-maximiser {map_q:.1f}")
json.dump(dict(Q_theta0=Qfun(uc0), Q_em=Qfun(uc_em), Q_qmax=Qfun(uc_q), map_theta0=map0, map_em=map_em, map_qmax=map_q,
               indel_em=dict(main_ins=theta_em['main_ins'], main_del=theta_em['main_del'], dom_ins=np.asarray(theta_em['dom_ins']).tolist(), dom_del=np.asarray(theta_em['dom_del']).tolist(), dom_weights=np.asarray(theta_em['dom_weights']).tolist(), ext=np.asarray(theta_em['ext_rates']).ravel().tolist()),
               indel_qmax=dict(main_ins=float(theta_q['main_ins']), main_del=float(theta_q['main_del']), dom_ins=np.asarray(theta_q['dom_ins']).tolist(), dom_del=np.asarray(theta_q['dom_del']).tolist(), dom_weights=np.asarray(theta_q['dom_weights']).tolist(), ext=np.asarray(theta_q['ext_rates']).ravel().tolist())),
          open(f'{MD}/mstep_check_{TAG}.json', 'w'), indent=1, default=float)
