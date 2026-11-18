"""Gap-opening probability against branch length (the numbers quoted in paper 1's Discussion). Usage: .venv/bin/python experiments/figs/gap_opening_curve.py
P(open a gap | leaving a match) = visit-weighted mass from the match states to insert or delete states of the pair HMM,
as a function of branch length t, for TKF92 and MixFrag F=2 at their fitted Pfam parameters (same weighting as the
gap-length sketches of Figure 6: match states weighted by their expected visits on a path from the start state)."""
import os, sys, json; os.environ['JAX_PLATFORMS']='cpu'; os.environ['JAX_ENABLE_X64']='1'
sys.path.insert(0,'/home/yam/tkf-mixdom/python')
import numpy as np, jax.numpy as jnp
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm, make_mixfrag_pair_hmm
from tkfmixdom.jax.core.params import S as _S, M as _M, I as _I, D as _D, E as _E
z=np.load('/home/yam/tkf-mixdom/python/pfam/gtr_refit_Qpi.npz'); Q=jnp.asarray(z['Q']); pi=jnp.asarray(z['pi'])
tk=json.load(open('/home/yam/tkf-mixdom/python/experiments/test_ll/tkf92_pfam_params.json'))
mf=json.load(open('/home/yam/tkf-mixdom/python/experiments/mixfrag_F/mixfrag_F2_logspace.json'))
def opening(lt, st):
    A=np.exp(np.asarray(lt,float)); st=np.asarray(st); n=len(st)
    keep=np.where(st!=_E)[0]; An=A[np.ix_(keep,keep)]
    e=np.zeros(len(keep)); e[list(keep).index(int(np.where(st==_S)[0][0]))]=1.0
    v=np.linalg.solve((np.eye(len(keep))-An).T, e)
    Mi=[i for i in range(n) if st[i]==_M]; Gi=[i for i in range(n) if st[i] in (_I,_D)]
    piM=np.array([v[list(keep).index(i)] for i in Mi]); piM/=piM.sum()
    return float(piM @ A[np.ix_(Mi,Gi)].sum(1)), [float(x) for x in A[np.ix_(Mi,Gi)].sum(1)]
ts=np.round(np.arange(0.2,4.001,0.05),3)
for name, mk in (('TKF92', lambda t: make_tkf92_pair_hmm(tk['ins_rate'],tk['del_rate'],t,tk['ext'],Q,pi)),
                 ('MixFrag F=2', lambda t: make_mixfrag_pair_hmm(mf['lam'],mf['mu'],t,jnp.asarray(mf['exts']),jnp.asarray(mf['weights']),Q,pi))):
    o=np.array([opening(*mk(t)[:2])[0] for t in ts])
    f=lambda thr: (ts[np.argmax(o>=thr)] if (o>=thr).any() else None)
    pm=opening(*mk(1.0)[:2])[1]
    print(f"{name}: open(0.76)={o[ts==0.75][0]:.4f} open(1)={o[ts==1.0][0]:.4f} open(1.3)={o[ts==1.3][0]:.4f} open(2)={o[ts==2.0][0]:.4f} open(3)={o[ts==3.0][0]:.4f}; crosses 0.040 at t={f(0.040)}, 0.08 at t={f(0.08)}; per-match-state at t=1: {np.round(pm,4)}")
