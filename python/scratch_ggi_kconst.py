import numpy as np, scratch_ggi_flow as G, scratch_ggi_flow_traj as TR
M=1
def Nbar(x,y,mu0,t,nrep=45000):
    lam0=mu0*x*(1-y)/(y*(1-x)); Ns=np.zeros((5,5))
    for _ in range(nrep): Ns+=G.count_transitions(G.gillespie_leg2([M]*TR.ggi_equilibrium_len(x,y),lam0,mu0,x,y,t))
    return Ns/nrep
def logit(p): return np.log(p/(1-p))
band=[0.1,0.2,0.4,0.7,1.1,1.6]
print("c = k_fit / (mu - lambda):  is it universal?")
print(f"{'(x,y)':<12}{'mu0':>5}{'k_fit':>8}{'mu-lam':>8}{'c':>7}")
# vary shape (x,y) AND rate (mu0) to test rate-independence of c
for x,y,mu0 in [(0.4,0.55,1.0),(0.3,0.5,1.0),(0.5,0.62,1.0),(0.3,0.45,1.0),(0.45,0.6,1.0),(0.4,0.55,2.0),(0.4,0.55,0.5)]:
    rows=[(t,*G.kar_to_lmr(*G.kl_fit(Nbar(x,y,mu0,t)),t)) for t in band]
    ts=np.array([r[0] for r in rows]); rK=np.array([r[3] for r in rows])
    mu=np.mean([r[2] for r in rows]); kap=np.mean([r[1]/r[2] for r in rows])
    k=-np.polyfit(ts,logit(rK),1)[0]
    mlam=mu*(1-kap)
    print(f"({x},{y})    {mu0:>5.1f}{k:>8.3f}{mlam:>8.3f}{k/mlam:>7.3f}")
