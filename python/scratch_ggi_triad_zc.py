"""Zero-corrected TKF92 surrogate in the Triad: replace tau5 with tau5_zc that
applies R = r/w resampling on empty paths, giving plain geometric stationary
(parameter w = r + (1-r)*kap).  When kap is set by length conservation so that
w = x/y, the stationary distribution matches GGI exactly."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scratch_ggi_flow import tau5

S, M, I, D, E = 0, 1, 2, 3, 4
SS,sI,MM,mI,MD,IM,iI,ID,Ds,Dm,Di,Dd,EE = range(13)
C_map = [S,S,M,M,M,I,I,I,D,D,D,D,E]

def tau5_zc(kap, alpha, r):
    """Zero-corrected TKF92.  R = r/w with w = r + (1-r)*kap.
    Modifies only the S row: y_SY *= w/kap (Y in body), y_SE *= (1-R)*w/kap."""
    T = tau5(kap, alpha, r).copy()
    w = r + (1-r)*kap
    if w <= 1e-12: return T
    R = r/w
    factor = w/kap  # 1/Z
    T[S, M] *= factor
    T[S, I] *= factor
    T[S, D] *= factor
    T[S, E] *= factor * (1-R)
    return T

def z_value(src, dst, lam_g, mu_g, x, y_g):
    if src in (S, M):
        if dst == M: return (1.0, -(lam_g+mu_g))
        if dst == I: return (0.0, lam_g)
        if dst == D: return (0.0, mu_g)
        if dst == E: return (1.0, -lam_g)
    if src == I:
        if dst == I: return (x, 0.0)
        if dst == M or dst == E: return (1-x, 0.0)
        return (0.0, 0.0)
    if src == D:
        if dst == D: return (y_g, 0.0)
        if dst == M: return (1-y_g, 0.0)
        if dst == E: return (1.0, 0.0)
        return (0.0, 0.0)
    return (0.0, 0.0)

def build_AB_zc(lam_s, mu_s, r_s, lam_g, mu_g, x, y_g, t, use_zc=True):
    """Build A and B matrices, using zero-corrected (or standard) TKF92."""
    if use_zc:
        Y = tau5_zc(lam_s/mu_s, np.exp(-mu_s*t), r_s)
    else:
        Y = tau5(lam_s/mu_s, np.exp(-mu_s*t), r_s)
    def y(a,b): return Y[a,b]
    def Z(src,dst): return z_value(src, dst, lam_g, mu_g, x, y_g)
    A = np.zeros((13,13)); B = np.zeros((13,13))
    def setyz(row, col, fac_y, src_z, dst_z):
        z0, z1 = Z(src_z, dst_z)
        A[row,col] += fac_y * z0
        B[row,col] += fac_y * z1
    def sety(row, col, fac_y):
        A[row,col] += fac_y
    # ... [Full Triad table as before; abbreviated]
    setyz(SS,sI,1,S,I); setyz(SS,MM,y(S,M),S,M); setyz(SS,MD,y(S,M),S,D)
    setyz(SS,IM,y(S,I),S,M); setyz(SS,ID,y(S,I),S,D); sety(SS,Ds,y(S,D)); setyz(SS,EE,y(S,E),S,E)
    setyz(sI,sI,1,I,I); setyz(sI,MM,y(S,M),I,M); setyz(sI,MD,y(S,M),I,D)
    setyz(sI,IM,y(S,I),I,M); setyz(sI,ID,y(S,I),I,D); sety(sI,Di,y(S,D)); setyz(sI,EE,y(S,E),I,E)
    setyz(MM,MM,y(M,M),M,M); setyz(MM,mI,1,M,I); setyz(MM,MD,y(M,M),M,D)
    setyz(MM,IM,y(M,I),M,M); setyz(MM,ID,y(M,I),M,D); sety(MM,Dm,y(M,D)); setyz(MM,EE,y(M,E),M,E)
    setyz(mI,MM,y(M,M),I,M); setyz(mI,mI,1,I,I); setyz(mI,MD,y(M,M),I,D)
    setyz(mI,IM,y(M,I),I,M); setyz(mI,ID,y(M,I),I,D); sety(mI,Di,y(M,D)); setyz(mI,EE,y(M,E),I,E)
    setyz(MD,MM,y(M,M),D,M); setyz(MD,mI,1,D,I); setyz(MD,MD,y(M,M),D,D)
    setyz(MD,IM,y(M,I),D,M); setyz(MD,ID,y(M,I),D,D); sety(MD,Dd,y(M,D)); setyz(MD,EE,y(M,E),D,E)
    setyz(IM,MM,y(I,M),M,M); setyz(IM,MD,y(I,M),M,D); setyz(IM,IM,y(I,I),M,M)
    setyz(IM,iI,1,M,I); setyz(IM,ID,y(I,I),M,D); sety(IM,Dm,y(I,D)); setyz(IM,EE,y(I,E),M,E)
    setyz(iI,MM,y(I,M),I,M); setyz(iI,MD,y(I,M),I,D); setyz(iI,IM,y(I,I),I,M)
    setyz(iI,iI,1,I,I); setyz(iI,ID,y(I,I),I,D); sety(iI,Di,y(I,D)); setyz(iI,EE,y(I,E),I,E)
    setyz(ID,MM,y(I,M),D,M); setyz(ID,MD,y(I,M),D,D); setyz(ID,IM,y(I,I),D,M)
    setyz(ID,iI,1,D,I); setyz(ID,ID,y(I,I),D,D); sety(ID,Dd,y(I,D)); setyz(ID,EE,y(I,E),D,E)
    setyz(Ds,MM,y(D,M),S,M); setyz(Ds,MD,y(D,M),S,D); setyz(Ds,IM,y(D,I),S,M)
    setyz(Ds,ID,y(D,I),S,D); sety(Ds,Ds,y(D,D)); setyz(Ds,EE,y(D,E),S,E)
    setyz(Dm,MM,y(D,M),M,M); setyz(Dm,MD,y(D,M),M,D); setyz(Dm,IM,y(D,I),M,M)
    setyz(Dm,ID,y(D,I),M,D); sety(Dm,Dm,y(D,D)); setyz(Dm,EE,y(D,E),M,E)
    setyz(Di,MM,y(D,M),I,M); setyz(Di,MD,y(D,M),I,D); setyz(Di,IM,y(D,I),I,M)
    setyz(Di,ID,y(D,I),I,D); sety(Di,Di,y(D,D)); setyz(Di,EE,y(D,E),I,E)
    setyz(Dd,MM,y(D,M),D,M); setyz(Dd,MD,y(D,M),D,D); setyz(Dd,IM,y(D,I),D,M)
    setyz(Dd,ID,y(D,I),D,D); sety(Dd,Dd,y(D,D)); setyz(Dd,EE,y(D,E),D,E)
    return A, B, Y

def m_from_triad(T):
    R = np.linalg.inv(np.eye(13) - T)
    n = np.outer(R[SS,:], R[:,EE]) * T
    m = np.zeros((5,5))
    for a in range(13):
        for b in range(13):
            m[C_map[a], C_map[b]] += n[a,b]
    return m

def matched_r_zc(lam_s, mu_s, r_in, lam_g, mu_g, x, y_g, t, dt, use_zc=True):
    """Compute matched r* via 1-param fit to the Triad's m matrix.
    Uses zero-corrected TKF92 as the model (with kap(r) tied via length conservation
    if use_zc=True, else free 3-param fit)."""
    A,B,Y = build_AB_zc(lam_s, mu_s, r_in, lam_g, mu_g, x, y_g, t, use_zc=use_zc)
    T = A + B*dt
    m = m_from_triad(T)
    a = x/y_g  # target stationary parameter
    def neg_logL(r):
        if r <= 0 or r >= min(a, 1-1e-7): return 1e10
        kap = (a - r)/(1 - r) if use_zc else lam_s/mu_s
        if kap <= 0 or kap >= 1: return 1e10
        Tnew = tau5_zc(kap, np.exp(-mu_s*t), r) if use_zc else tau5(kap, np.exp(-mu_s*t), r)
        with np.errstate(divide='ignore', invalid='ignore'):
            logT = np.where(Tnew > 1e-300, np.log(np.clip(Tnew, 1e-300, None)), 0)
        return -np.sum(m * logT)
    res = minimize_scalar(neg_logL, bounds=(1e-5, min(a-1e-5, 1-1e-5)), method='bounded')
    return res.x

# ==== Run sign-flip test ====
x, y_g = 0.4, 0.55
mu_0 = 1.0
lam_0 = mu_0*x*(1-y_g)/(y_g*(1-x))
mu_native = mu_0/(1-y_g)  # corrected BO

empirical = [(0.1, 0.7067, 1.0714, 0.2841),
             (0.5, 0.728, 1.086, 0.2564),
             (1.0, 0.7279, 1.0699, 0.2210),
             (2.0, 0.7056, 1.0285, 0.1908),
             (4.0, 0.7229, 1.0478, 0.1668)]

print("Zero-corrected TKF92 surrogate:")
print(f"target stationary parameter w = x/y = {x/y_g:.4f}")
print()
print(f"{'t':>5}{'r in':>8}{'kap(r)':>9}{'r* (ZC, BO)':>14}{'diff':>12}{'sign':>6}")
dt = 1e-5
for (t, lam, mu, r) in empirical:
    kap_bo = (x/y_g - r)/(1-r)
    lam_bo = kap_bo * mu_native
    rs = matched_r_zc(lam_bo, mu_native, r, lam_0, mu_0, x, y_g, t, dt, use_zc=True)
    diff = rs - r
    sign = '+' if diff > 0 else '-' if diff < 0 else '0'
    print(f"{t:>5.2f}{r:>8.4f}{kap_bo:>9.4f}{rs:>14.6f}{diff:>+12.2e}{sign:>6}")

print()
print("For comparison, STANDARD TKF92 surrogate (no zero correction), same BO:")
print(f"{'t':>5}{'r in':>8}{'r* (std, BO)':>14}{'diff':>12}{'sign':>6}")
for (t, lam, mu, r) in empirical:
    kap_bo = (x/y_g - r)/(1-r)
    lam_bo = kap_bo * mu_native
    rs = matched_r_zc(lam_bo, mu_native, r, lam_0, mu_0, x, y_g, t, dt, use_zc=False)
    diff = rs - r
    sign = '+' if diff > 0 else '-' if diff < 0 else '0'
    print(f"{t:>5.2f}{r:>8.4f}{rs:>14.6f}{diff:>+12.2e}{sign:>6}")
