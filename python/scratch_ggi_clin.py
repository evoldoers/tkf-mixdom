"""Instantaneous r-relaxation rate from the EXACT generator flow, at the finite-t
matched surrogate theta*(t).  c(t) = -d(logit r)/dt / (mu-lambda).
Tests: (a) does c(t) tend to a constant as t grows?  (b) is it rate-independent
(compare same shape at mu0=1 vs mu0=2 at equal mixing-time (mu-lam)t)?"""
import numpy as np, scratch_ggi_flow as G, scratch_ggi_flow_traj as TR
rng=G.rng

def c_at(x,y,mu0,t,nfit=40000,nflow=30000):
    lam0=mu0*x*(1-y)/(y*(1-x))
    lam,mu,r=TR.direct_fit_ggi(lam0,mu0,x,y,t,nsamp=nfit)   # matched theta*(t)
    rhs,lmr0=TR.flow_rhs(lam,mu,r,t,lam0,mu0,x,y,nsamp=nflow)
    L,Mu,R=lmr0; drdt=rhs[2]
    mlam=Mu-L
    dlogit=drdt/(R*(1-R))
    return R,Mu,L,mlam,-dlogit,(-dlogit)/mlam,Mu*t,mlam*t

print(f"{'shape':<11}{'mu0':>4}{'t':>5}{'r*':>7}{'mu':>7}{'mu-lam':>8}{'k_inst':>8}{'c':>7}{'(mu-lam)t':>10}")
cfg=[(0.4,0.55,1.0,1.0),(0.4,0.55,1.0,2.0),(0.4,0.55,1.0,3.0),(0.4,0.55,1.0,4.0),
     (0.4,0.55,2.0,0.5),(0.4,0.55,2.0,1.0),(0.4,0.55,2.0,1.5),(0.4,0.55,2.0,2.0),
     (0.3,0.50,1.0,2.0),(0.3,0.50,1.0,4.0),(0.5,0.62,1.0,2.0),(0.5,0.62,1.0,4.0)]
for x,y,mu0,t in cfg:
    r,mu,lam,mlam,kinst,c,mut,mlt=c_at(x,y,mu0,t)
    print(f"({x},{y}) {mu0:>4.1f}{t:>5.1f}{r:>7.3f}{mu:>7.3f}{mlam:>8.3f}{kinst:>8.3f}{c:>7.3f}{mlt:>10.2f}")
