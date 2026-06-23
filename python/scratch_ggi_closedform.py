import numpy as np, scratch_ggi_flow as G, scratch_ggi_flow_traj as TR
M=1; S,I,D=0,2,3
def Nbar(x,y,mu0,t,nrep=60000):
    lam0=mu0*x*(1-y)/(y*(1-x)); Ns=np.zeros((5,5))
    for _ in range(nrep): Ns+=G.count_transitions(G.gillespie_leg2([M]*TR.ggi_equilibrium_len(x,y),lam0,mu0,x,y,t))
    return Ns/nrep
def CE(N,lam,mu,r,t):
    T=G.tau5(lam/mu,np.exp(-mu*t),r); lt=np.where(T>0,np.log(np.clip(T,1e-300,None)),0); return -np.sum(N*lt)
def sig(z): return 1/(1+np.exp(-z))
def logit(p): return np.log(p/(1-p))

band=[0.1,0.2,0.35,0.55,0.8,1.1,1.5,2.0]
for x,y in [(0.4,0.55),(0.3,0.5)]:
    mu0=1.0; lam0=mu0*x*(1-y)/(y*(1-x)); ellG=x/(y-x)
    lam_mm,mu_mm,r_mm=TR.moment_match(lam0,mu0,x,y)
    rows=[]
    for t in band:
        N=Nbar(x,y,mu0,t)
        lk,mk,rk=G.kar_to_lmr(*G.kl_fit(N),t)
        rows.append((t,N,lk,mk,rk))
    # fit closed-form constants in band: logit(r_KL)=a-k t ; mu-bar=mean mu_KL
    ts=np.array([r[0] for r in rows]); rKL=np.array([r[4] for r in rows]); muKL=np.array([r[3] for r in rows]); kapKL=np.array([r[2]/r[3] for r in rows])
    k,a=np.polyfit(ts,logit(rKL),1); k=-k; r0=sig(a)
    mubar=muKL.mean(); kapbar=kapKL.mean()
    print(f"\n=== GGI ({x},{y}), ell_GGI={ellG:.3f} ; mm=(lam{lam_mm:.2f},mu{mu_mm:.2f},r{r_mm:.2f}) ===")
    print(f"  fitted constants: r0={r0:.3f}  k={k:.3f}  mu-bar={mubar:.3f}   |  predicted k=mu(1-kappa)={mubar*(1-kapbar):.3f}")
    print(f"  {'t':>5}{'r_KL':>7}{'r_cf':>7}{'mu_KL':>7}{'kap_KL':>7}{'kap_cf':>7} | {'CE_mm':>8}{'CE_cf':>8}{'CE_KL':>8}{'gap closed':>11}")
    for (t,N,lk,mk,rk) in rows:
        rcf=sig(a-k*t); kcf=ellG*(1-rcf)/(1+ellG*(1-rcf)); lcf=kcf*mubar
        ce_mm=CE(N,lam_mm,mu_mm,r_mm,t); ce_cf=CE(N,lcf,mubar,rcf,t); ce_kl=CE(N,lk,mk,rk,t)
        gc=(ce_mm-ce_cf)/(ce_mm-ce_kl) if ce_mm>ce_kl else float('nan')
        print(f"  {t:>5.2f}{rk:>7.3f}{rcf:>7.3f}{mk:>7.3f}{lk/mk:>7.3f}{kcf:>7.3f} | {ce_mm:>8.3f}{ce_cf:>8.3f}{ce_kl:>8.3f}{gc:>11.1%}")
