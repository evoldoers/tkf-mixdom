"""Phase-transition test: is r_inf=0 a PEGGED boundary solution over a range of
shapes (=> transition, r_inf the order parameter), or smoothly ->0 (no transition)?
Decisive sign test: dr/dt of the matched flow at SMALL r, vs kappa.
  dr/dt(small r) > 0  -> r pushed UP -> interior r_inf>0
  dr/dt(small r) < 0  -> r pushed DOWN to 0 -> r_inf=0 (pegged)
Sign change at kappa_c = transition."""
import numpy as np, scratch_ggi_flow as G
def drdt_small_r(x,y,r=0.03,t=4.0,mu0=1.0,nsamp=30000):
    lam0=mu0*x*(1-y)/(y*(1-x)); mu=mu0/(1-y); ellG=x/(y-x)
    kap=ellG*(1-r)/(1+ellG*(1-r)); lam=kap*mu
    lmr0,dlmr,_,_=G.flow_generator(lam,mu,r,t,lam0,mu0,x,y,nsamp=nsamp)
    return kap, lmr0[2], dlmr[2]
print(f"{'(x,y)':>12}{'kappa':>7}{'r_in':>6}{'dr/dt':>9}{'sign':>6}")
for x,y in [(0.15,0.40),(0.2,0.40),(0.25,0.45),(0.3,0.50),(0.4,0.55),(0.5,0.62),(0.6,0.72)]:
    kap,rin,d=drdt_small_r(x,y)
    print(f"({x},{y}){kap:>7.3f}{rin:>6.2f}{d:>9.4f}{'  +' if d>0 else '  -'}",flush=True)
