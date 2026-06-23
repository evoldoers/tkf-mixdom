"""
Exact KL m-projection fit of the TKF92 self-composition.

Composite = branch1 pair-HMM (X,Y) FST-composed with branch2 conditional
WFST (Y->Z), synchronized on the intermediate tape Y, marginalising Y.
We compute the composite's expected X->Z transition counts via the
fundamental matrix and feed them to the validated KL M-step (argmax sum
Nbar log tau').  Two validation gates must pass first:
  A) r=0   -> composite == TKF91(t+u)
  B) u->0  -> composite == TKF92(t)
plus a total-probability check.

6-state index: S=0, M=1, I0=2, I1=3, D=4, E=5.
5-state (collapsed I0/I1) index for counts: S=0, M=1, I=2, D=3, E=4.
"""
import numpy as np
from scipy.optimize import minimize

def beta_gamma(kap, alpha):
    eps = 1e-12
    alpha = min(alpha, 1 - 1e-13)
    a1 = alpha ** (1 - kap)
    beta = kap * (1 - a1) / (1 - kap * a1)
    gamma = 1 - beta / (kap * (1 - alpha))
    return beta, gamma

# ---------- 5-state TKF92 pair HMM + single-branch expected counts (Stage 1) -
def tau5(kap, alpha, r):
    b, g = beta_gamma(kap, alpha); ob, og, ok = 1-b, 1-g, 1-kap
    T = np.zeros((5, 5))                       # S,M,I,D,E
    T[0] = [0, ob*kap*alpha, b, ob*kap*(1-alpha), ob*ok]
    for s, sc in [(1, 1), (2, 2)]:
        row = (1-r)*np.array([0, ob*kap*alpha, b, ob*kap*(1-alpha), ob*ok]); row[sc] += r
        T[s] = row
    rowd = (1-r)*np.array([0, og*kap*alpha, g, og*kap*(1-alpha), og*ok]); rowd[3] += r
    T[3] = rowd
    return T

def expected_counts5(kap, alpha, r):
    T = tau5(kap, alpha, r)
    Q = T[np.ix_([1,2,3],[1,2,3])]; p0 = T[0,[1,2,3]]
    v = p0 @ np.linalg.inv(np.eye(3)-Q); vis = {0:1.0,1:v[0],2:v[1],3:v[2]}
    N = np.zeros((5,5))
    for s in [0,1,2,3]: N[s] = vis[s]*T[s]
    return N

# ---------- 6-state pair HMM (branch1) and WFST (branch2) -------------------
def pairhmm6(kap, alpha, r):
    b, g = beta_gamma(kap, alpha); ob, og, ok = 1-b, 1-g, 1-kap
    T = np.zeros((6, 6))                       # rows/cols S,M,I0,I1,D,E
    T[0] = [0, ob*kap*alpha, b, 0, ob*kap*(1-alpha), ob*ok]
    T[1] = [0, r+(1-r)*ob*kap*alpha, 0, (1-r)*b, (1-r)*ob*kap*(1-alpha), (1-r)*ob*ok]
    T[2] = [0, (1-r)*ob*kap*alpha, r+(1-r)*b, 0, (1-r)*ob*kap*(1-alpha), (1-r)*ob*ok]
    T[3] = [0, (1-r)*ob*kap*alpha, 0, r+(1-r)*b, (1-r)*ob*kap*(1-alpha), (1-r)*ob*ok]
    T[4] = [0, (1-r)*og*kap*alpha, 0, (1-r)*g, r+(1-r)*og*kap*(1-alpha), (1-r)*og*ok]
    return T

def wfst6(kap, alpha, r):
    b, g = beta_gamma(kap, alpha); ob, og = 1-b, 1-g; p = r+(1-r)*kap
    T = np.zeros((6, 6))
    T[0] = [0, ob*alpha, b, 0, ob*(1-alpha), ob]
    T[1] = [0, (r+(1-r)*ob*kap*alpha)/p, 0, (1-r)*b, ((1-r)*ob*kap*(1-alpha))/p, ob]
    T[2] = [0, (1-r)*ob*alpha, r+(1-r)*b, 0, (1-r)*ob*(1-alpha), (1-r)*ob]
    T[3] = [0, ((1-r)*ob*kap*alpha)/p, 0, r+(1-r)*b, ((1-r)*ob*kap*(1-alpha))/p, ob]
    T[4] = [0, ((1-r)*og*kap*alpha)/p, 0, (1-r)*g, (r+(1-r)*og*kap*(1-alpha))/p, og]
    return T

# ---------- composite expected counts via FST composition -------------------
def composite_counts(kap, alpha_t, alpha_u, r):
    T1 = pairhmm6(kap, alpha_t, r)             # branch1 (X,Y)
    T2 = wfst6(kap, alpha_u, r)                # branch2 (Y->Z)
    QB = [0,1,2,3,4]                            # branch states S,M,I0,I1,D
    # augmented state (q1,q2,f,last): f = Mohri eps-filter {0,1,2}
    def idx(q1,q2,f,last): return ((q1*5+q2)*3+f)*4+last
    Nst = 5*5*3*4
    W = np.zeros((Nst,Nst)); endv = np.zeros(Nst); flows=[]
    Yprod=[1,2,3]; Ycons=[1,4]                  # branch1 Y-writers; branch2 Y-readers
    for q1 in QB:
        for q2 in QB:
            for f in [0,1,2]:
                for last in [0,1,2,3]:
                    src=idx(q1,q2,f,last)
                    # MB/TKF canonical order: branch2 inserts (to the right of
                    # its parent) BEFORE branch1 deletes; no insert after a delete.
                    # filter f: 0=synced, 1=inserting, 2=deleting.
                    # branch2-I (T2 input-eps): allowed if f in {0,1}, new f=1
                    if f in (0,1):
                        for i2 in [2,3]:
                            w=T2[q2,i2]
                            if w>0: W[src,idx(q1,i2,1,2)]+=w; flows.append((src,2,w))
                    # branch1-D (T1 output-eps): allowed from any f, new f=2
                    w=T1[q1,4]
                    if w>0: W[src,idx(4,q2,2,3)]+=w; flows.append((src,3,w))
                    # Y-residue (match): any f, new f=0
                    for a in Yprod:
                        wa=T1[q1,a]
                        if wa==0: continue
                        for c in Ycons:
                            wc=T2[q2,c]
                            if wc==0: continue
                            w=wa*wc
                            if a==1 and c==1: emit=1          # MATCH
                            elif a==1 and c==4: emit=3        # DELETE
                            elif a in (2,3) and c==1: emit=2  # INSERT
                            else: emit=0                      # GHOST
                            nl=emit if emit else last
                            W[src,idx(a,c,0,nl)]+=w
                            if emit: flows.append((src,emit,w))
                    # End: any f
                    we=T1[q1,5]*T2[q2,5]
                    if we>0: endv[src]+=we
    e=np.zeros(Nst); e[idx(0,0,0,0)]=1.0
    v=np.linalg.solve((np.eye(Nst)-W).T, e)
    N=np.zeros((5,5))
    for (src,emit,w) in flows: N[src%4, emit]+=v[src]*w     # last->row, emit->col
    for src in range(Nst):
        if endv[src]>0: N[src%4,4]+=v[src]*endv[src]
    total=sum(v[src]*endv[src] for src in range(Nst))
    return N, total

def kl_fit(Nbar, x0=(0.4,0.5,0.2)):
    def nll(x):
        T=tau5(*x)
        with np.errstate(divide='ignore',invalid='ignore'):
            lt=np.where(T>0, np.log(np.clip(T,1e-300,None)), 0.0)
        return -np.sum(Nbar*lt)
    r=minimize(nll, x0, bounds=[(1e-4,1-1e-4)]*3, method='L-BFGS-B',
               options=dict(ftol=1e-15,gtol=1e-13,maxiter=8000))
    return r.x

# ============================ validation gates =============================
print("=== Gate A: r=0  -> composite == TKF91(t+u) ===")
for (kap,at,au) in [(0.5,0.7,0.6),(0.3,0.5,0.9),(0.7,0.8,0.4)]:
    Nc,tot=composite_counts(kap,at,au,0.0)
    N91=expected_counts5(kap,at*au,0.0)
    print(f" kap={kap} a_t={at} a_u={au}: maxerr={np.max(np.abs(Nc-N91)):.2e}  total={tot:.6f}")

print("\n=== Gate B: u->0 -> composite == TKF92(t) ===")
for (kap,at,r) in [(0.5,0.6,0.3),(0.3,0.8,0.5),(0.7,0.4,0.2)]:
    au=1-1e-9
    Nc,tot=composite_counts(kap,at,au,r)
    Nt=expected_counts5(kap,at*au,r)
    print(f" kap={kap} a_t={at} r={r}: maxerr={np.max(np.abs(Nc-Nt)):.2e}  total={tot:.6f}")

print("\n=== off-manifold check: composite != TKF92(t+u) for r>0 ===")
kap,at,au,r=0.5,0.7,0.6,0.4
Nc,tot=composite_counts(kap,at,au,r)
N0=expected_counts5(kap,at*au,r)
print(f" composite vs TKF92(t+u) maxerr={np.max(np.abs(Nc-N0)):.3e}  total={tot:.6f}")

print("\n=== diagnose r=0: which counts differ (composite vs TKF91(t+u)) ===")
kap,at,au=0.5,0.7,0.6
Nc,_=composite_counts(kap,at,au,0.0); N91=expected_counts5(kap,at*au,0.0)
labels=['S','M','I','D','E']
print("rows=from {S,M,I,D}, cols=to {M,I,D,E}; (composite - TKF91):")
diff=Nc-N91
for s in range(4):
    print(f"  {labels[s]}: "+"  ".join(f"{labels[c]}:{diff[s,c]:+.4f}" for c in range(1,5)))
print("\n=== does the fit recover params despite the convention diff? ===")
for tag,(kap,at,au,r) in [("r=0",(0.5,0.7,0.6,0.0)),("general",(0.5,0.7,0.6,0.3)),
                          ("general2",(0.4,0.6,0.5,0.5))]:
    Nc,_=composite_counts(kap,at,au,r)
    fit=kl_fit(Nc,x0=(kap,at*au,r))
    bare=(kap,at*au,r)
    print(f" {tag:9s} bare(kap,alpha,r)={tuple(round(x,4) for x in bare)} "
          f"-> fit={tuple(round(x,4) for x in fit)}")
