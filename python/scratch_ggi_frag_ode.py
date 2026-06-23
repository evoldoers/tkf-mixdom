"""Validate the mean-field fragment-coherence ODE
   dr/dt = lam[x(1+r)-2r] - mu r(1-r)(1-y)/(1-yr)
against direct event simulation of insertions/deletions on a Bernoulli(1-r)
fragment-boundary chain.  lam,mu are per-link rates; lam_I=lam(1-x), mu_D=mu(1-y)."""
import numpy as np
rng=np.random.default_rng(0)

def drdt_formula(r,lam,mu,x,y):
    return lam*(x*(1+r)-2*r) - mu*r*(1-r)*(1-y)/(1-y*r)

def drdt_sim(r,lam,mu,x,y,L0=6000,T=0.025,nrep=600):
    lamI=lam*(1-x); muD=mu*(1-y); q=1-r
    sl=[]
    for _ in range(nrep):
        s=(rng.random(L0)<q); s[0]=True; s=list(map(int,s))
        t=0.0; r0=1-sum(s)/len(s)
        while True:
            L=len(s); R=lamI*(L+1)+muD*L
            t+=rng.exponential(1.0/R)
            if t>T: break
            if rng.random()<lamI*(L+1)/R:                 # insertion
                g=rng.integers(0,L+1); m=int(rng.geometric(1-x))
                run=[1]+[0]*(m-1)
                if g<L: s[g]=1
                s[g:g]=run
            else:                                          # deletion
                p=rng.integers(0,L); j=int(rng.geometric(1-y)); j=min(j,L-p)
                if p+j<L:
                    nf=1 if any(s[p:p+j+1]) else 0
                    s[p+j]=nf
                del s[p:p+j]
                if s: s[0]=1
        L=len(s); rT=1-sum(s)/L
        sl.append((rT-r0)/t)
    return np.mean(sl), np.std(sl)/np.sqrt(nrep)

print(f"{'r':>5}{'(x,y)':>11}{'formula':>10}{'sim':>10}{'sim_se':>9}")
for r,x,y in [(0.10,0.4,0.55),(0.18,0.4,0.55),(0.30,0.4,0.55),(0.30,0.3,0.5),(0.30,0.5,0.62)]:
    lam,mu=0.7,1.0
    f=drdt_formula(r,lam,mu,x,y); s,se=drdt_sim(r,lam,mu,x,y)
    print(f"{r:>5.2f}({x},{y}){f:>10.4f}{s:>10.4f}{se:>9.4f}")
