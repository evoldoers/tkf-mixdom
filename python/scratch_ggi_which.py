"""Definitive: measure the matched-flow r(t), fit exp-relaxation-to-floor, and
overlay the snowflake ODE integrated from r(t0).  Decide which model fits."""
import numpy as np, scratch_ggi_flow as G, scratch_ggi_flow_traj as TR
from scipy.optimize import curve_fit
x,y=0.4,0.55; mu0=1.0; lam0=mu0*x*(1-y)/(y*(1-x))
ts=[0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0]
traj=[]
for t in ts:
    lam,mu,r=TR.direct_fit_ggi(lam0,mu0,x,y,t,nsamp=35000)
    traj.append((t,lam,mu,r)); print(f"  t={t:4.2f} lam={lam:.4f} mu={mu:.4f} r={r:.4f} mu-lam={mu-lam:.4f}",flush=True)
tarr=np.array([p[0] for p in traj]); rarr=np.array([p[3] for p in traj])
lam_a=np.mean([p[1] for p in traj]); mu_a=np.mean([p[2] for p in traj])
def model(t,rinf,r0,k): return rinf+(r0-rinf)*np.exp(-k*t)
popt,_=curve_fit(model,tarr,rarr,p0=[0.15,0.3,1.0],maxfev=20000)
rinf_fit,r0_fit,k_fit=popt
# snowflake ODE: integrate dr/dt = lam[x(1+r)-2r] - mu r(1-r)(1-y)/(1-yr) from r(t0)
def drdt(r): return lam_a*(x*(1+r)-2*r)-mu_a*r*(1-r)*(1-y)/(1-y*r)
r=rarr[0]; tt=ts[0]; ode={}
import bisect
grid=sorted(set(ts))
for tgt in grid:
    while tt<tgt-1e-9:
        h=min(0.005,tgt-tt); r=r+h*drdt(r); tt+=h
    ode[tgt]=r
A=mu_a*(1-y)+lam_a*(2-x)*y; Bc=-lam_a*(2-x+x*y)-mu_a*(1-y); C=lam_a*x
rinfB=(-Bc-np.sqrt(Bc**2-4*A*C))/(2*A); kB=lam_a*(2-x)+mu_a*(1-y)*((1-2*rinfB)*(1-y*rinfB)+y*(rinfB-rinfB**2))/(1-y*rinfB)**2
print("\n  t     r_matched   exp-fit   snowflake-ODE")
for (t,_,_,rm) in traj:
    print(f"  {t:4.2f}   {rm:.4f}     {model(t,*popt):.4f}    {ode[t]:.4f}")
print(f"\nexp-fit:        r_inf={rinf_fit:.4f}  r0={r0_fit:.4f}  k={k_fit:.4f}")
print(f"snowflake ODE:  r_inf={rinfB:.4f}  k_local={kB:.4f}")
print(f"mu-lambda avg = {mu_a-lam_a:.4f}    lam+mu = {lam_a+mu_a:.4f}")
