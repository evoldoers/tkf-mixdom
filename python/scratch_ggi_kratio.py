"""Verify k = 2(mu-lambda) across GGI shapes (varied kappa).
Fit matched-flow r(t)=rinf+(r0-rinf)exp(-kt); report k/(mu-lambda).
If ~2 across kappa -> intensive r couples to n=2 Meixner mode."""
import numpy as np, scratch_ggi_flow as G, scratch_ggi_flow_traj as TR
from scipy.optimize import curve_fit
shapes=[(0.2,0.40),(0.3,0.50),(0.4,0.55),(0.5,0.62),(0.6,0.72)]
ts=[0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0]
print(f"{'(x,y)':>11}{'kappa':>7}{'mu-lam':>8}{'k_fit':>8}{'r_inf':>7}{'k/(mu-lam)':>11}",flush=True)
rows=[]
for x,y in shapes:
    mu0=1.0; lam0=mu0*x*(1-y)/(y*(1-x))
    tr=[TR.direct_fit_ggi(lam0,mu0,x,y,t,nsamp=45000) for t in ts]
    tarr=np.array(ts); rarr=np.array([p[2] for p in tr])
    lam_a=np.mean([p[0] for p in tr]); mu_a=np.mean([p[1] for p in tr]); kap=lam_a/mu_a; ml=mu_a-lam_a
    def m(t,rinf,r0,k): return rinf+(r0-rinf)*np.exp(-k*t)
    try:
        popt,_=curve_fit(m,tarr,rarr,p0=[0.13,0.30,0.6],bounds=([0,0.05,0.05],[0.5,0.6,6]),maxfev=40000)
        rinf,r0,k=popt
    except Exception as ex:
        rinf,r0,k=(np.nan,np.nan,np.nan)
    print(f"({x},{y}){kap:>7.3f}{ml:>8.4f}{k:>8.4f}{rinf:>7.3f}{k/ml:>11.3f}",flush=True)
    rows.append(k/ml)
v=[r for r in rows if r==r]
print(f"\nk/(mu-lambda): mean={np.mean(v):.3f} sd={np.std(v):.3f}  (==2 => intensive n=2; ==1 => n=1)")
