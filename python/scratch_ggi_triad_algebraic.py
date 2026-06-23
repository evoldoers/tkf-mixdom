"""Algebraic A + B*dt expansion of the Triad, giving dr/dt in closed form.

T(dt) = A + B*dt, with A = T|_{dt=0} and B = dT/d(dt)|_0.
R_0 = (I-A)^{-1}.  Then to first order:
  n^(0)_{ab} = R_0[S,a] * A[a,b] * R_0[b,E]      (TKF92's own counts)
  n^(1)_{ab} = R_0[S,a] * B[a,b] * R_0[b,E]
             + (R_0 B R_0)[S,a] * A[a,b] * R_0[b,E]
             + R_0[S,a] * A[a,b] * (R_0 B R_0)[b,E]
Coarse-grain to m^(0), m^(1).
dr/dt = (sum_X m^(1)_XX/u_XX - N^(1)_tot) / (sum_X m^(0)_XX (1-v_XX)/u_XX^2)
"""
import numpy as np
from scratch_ggi_flow import tau5

S, M, I, D, E = 0, 1, 2, 3, 4
SS,sI,MM,mI,MD,IM,iI,ID,Ds,Dm,Di,Dd,EE = range(13)
C_map = [S,S,M,M,M,I,I,I,D,D,D,D,E]  # coarse-graining

def z_value(src, dst, lam_g, mu_g, x, y_g):
    """Return (z at dt=0, dz/d(dt) at dt=0)."""
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

def build_AB(lam_s, mu_s, r_s, lam_g, mu_g, x, y_g, t):
    """Construct A and B per the 13x13 Triad table."""
    Y = tau5(lam_s/mu_s, np.exp(-mu_s*t), r_s)
    def y(a,b): return Y[a,b]
    def Z(src,dst): return z_value(src, dst, lam_g, mu_g, x, y_g)
    A = np.zeros((13,13)); B = np.zeros((13,13))

    # Helper: set T[a,b] = factor_y * z(src,dst).  Splits into A,B parts.
    def setyz(row, col, fac_y, src_z, dst_z):
        z0, z1 = Z(src_z, dst_z)
        A[row,col] += fac_y * z0
        B[row,col] += fac_y * z1
    # Helper: set T[a,b] = fac_y (no z factor)
    def sety(row, col, fac_y):
        A[row,col] += fac_y

    # Row SS
    setyz(SS, sI, 1.0, S, I)         # only z factor
    setyz(SS, MM, y(S,M), S, M)
    setyz(SS, MD, y(S,M), S, D)
    setyz(SS, IM, y(S,I), S, M)
    setyz(SS, ID, y(S,I), S, D)
    sety (SS, Ds, y(S,D))
    setyz(SS, EE, y(S,E), S, E)
    # Row sI
    setyz(sI, sI, 1.0, I, I)
    setyz(sI, MM, y(S,M), I, M)
    setyz(sI, MD, y(S,M), I, D)
    setyz(sI, IM, y(S,I), I, M)
    setyz(sI, ID, y(S,I), I, D)
    sety (sI, Di, y(S,D))
    setyz(sI, EE, y(S,E), I, E)
    # Row MM
    setyz(MM, MM, y(M,M), M, M)
    setyz(MM, mI, 1.0, M, I)
    setyz(MM, MD, y(M,M), M, D)
    setyz(MM, IM, y(M,I), M, M)
    setyz(MM, ID, y(M,I), M, D)
    sety (MM, Dm, y(M,D))
    setyz(MM, EE, y(M,E), M, E)
    # Row mI
    setyz(mI, MM, y(M,M), I, M)
    setyz(mI, mI, 1.0, I, I)
    setyz(mI, MD, y(M,M), I, D)
    setyz(mI, IM, y(M,I), I, M)
    setyz(mI, ID, y(M,I), I, D)
    sety (mI, Di, y(M,D))
    setyz(mI, EE, y(M,E), I, E)
    # Row MD
    setyz(MD, MM, y(M,M), D, M)
    setyz(MD, mI, 1.0, D, I)
    setyz(MD, MD, y(M,M), D, D)
    setyz(MD, IM, y(M,I), D, M)
    setyz(MD, ID, y(M,I), D, D)
    sety (MD, Dd, y(M,D))
    setyz(MD, EE, y(M,E), D, E)
    # Row IM
    setyz(IM, MM, y(I,M), M, M)
    setyz(IM, MD, y(I,M), M, D)
    setyz(IM, IM, y(I,I), M, M)
    setyz(IM, iI, 1.0, M, I)
    setyz(IM, ID, y(I,I), M, D)
    sety (IM, Dm, y(I,D))
    setyz(IM, EE, y(I,E), M, E)
    # Row iI
    setyz(iI, MM, y(I,M), I, M)
    setyz(iI, MD, y(I,M), I, D)
    setyz(iI, IM, y(I,I), I, M)
    setyz(iI, iI, 1.0, I, I)
    setyz(iI, ID, y(I,I), I, D)
    sety (iI, Di, y(I,D))
    setyz(iI, EE, y(I,E), I, E)
    # Row ID
    setyz(ID, MM, y(I,M), D, M)
    setyz(ID, MD, y(I,M), D, D)
    setyz(ID, IM, y(I,I), D, M)
    setyz(ID, iI, 1.0, D, I)
    setyz(ID, ID, y(I,I), D, D)
    sety (ID, Dd, y(I,D))
    setyz(ID, EE, y(I,E), D, E)
    # Row Ds
    setyz(Ds, MM, y(D,M), S, M)
    setyz(Ds, MD, y(D,M), S, D)
    setyz(Ds, IM, y(D,I), S, M)
    setyz(Ds, ID, y(D,I), S, D)
    sety (Ds, Ds, y(D,D))
    setyz(Ds, EE, y(D,E), S, E)
    # Row Dm
    setyz(Dm, MM, y(D,M), M, M)
    setyz(Dm, MD, y(D,M), M, D)
    setyz(Dm, IM, y(D,I), M, M)
    setyz(Dm, ID, y(D,I), M, D)
    sety (Dm, Dm, y(D,D))
    setyz(Dm, EE, y(D,E), M, E)
    # Row Di
    setyz(Di, MM, y(D,M), I, M)
    setyz(Di, MD, y(D,M), I, D)
    setyz(Di, IM, y(D,I), I, M)
    setyz(Di, ID, y(D,I), I, D)
    sety (Di, Di, y(D,D))
    setyz(Di, EE, y(D,E), I, E)
    # Row Dd
    setyz(Dd, MM, y(D,M), D, M)
    setyz(Dd, MD, y(D,M), D, D)
    setyz(Dd, IM, y(D,I), D, M)
    setyz(Dd, ID, y(D,I), D, D)
    sety (Dd, Dd, y(D,D))
    setyz(Dd, EE, y(D,E), D, E)
    return A, B, Y

def coarse_grain(n):
    m = np.zeros((5,5))
    for a in range(13):
        for b in range(13):
            m[C_map[a], C_map[b]] += n[a,b]
    return m

def dr_dt(lam_s, mu_s, r_s, lam_g, mu_g, x, y_g, t):
    A, B, Y = build_AB(lam_s, mu_s, r_s, lam_g, mu_g, x, y_g, t)
    R0 = np.linalg.inv(np.eye(13) - A)
    RBR = R0 @ B @ R0
    # n^(0) and n^(1)
    n0 = np.outer(R0[SS,:], R0[:,EE]) * A
    n1 = (np.outer(R0[SS,:], R0[:,EE]) * B
        + np.outer(RBR[SS,:], R0[:,EE]) * A
        + np.outer(R0[SS,:], RBR[:,EE]) * A)
    m0 = coarse_grain(n0); m1 = coarse_grain(n1)
    # TKF91 self-loops
    Y91 = tau5(lam_s/mu_s, np.exp(-mu_s*t), 0.0)
    v = [Y91[M,M], Y91[I,I], Y91[D,D]]
    u = [r_s + (1-r_s)*vk for vk in v]
    N1 = m1.sum()
    # M-step linearisation
    numer = m1[M,M]/u[0] + m1[I,I]/u[1] + m1[D,D]/u[2] - N1
    denom = m0[M,M]*(1-v[0])/u[0]**2 + m0[I,I]*(1-v[1])/u[1]**2 + m0[D,D]*(1-v[2])/u[2]**2
    # Sanity: zeroth-order self-consistency
    N0 = m0.sum()
    selfcheck = m0[M,M]/u[0] + m0[I,I]/u[1] + m0[D,D]/u[2] - N0
    return numer/denom, m0, m1, v, u, selfcheck

# ==== Run ====
x, y_g = 0.4, 0.55
mu_0 = 1.0
lam_0 = mu_0*x*(1-y_g)/(y_g*(1-x))
r_native = x
mu_native = mu_0*(1-r_native)/(1-y_g)
ell = x/(y_g-x)
kap = ell*(1-r_native)/(1+ell*(1-r_native))
lam_native = kap*mu_native
print(f"GGI: lam_0={lam_0:.4f} mu_0={mu_0} x={x} y={y_g}")
print(f"Native TKF92: lam={lam_native:.4f} mu={mu_native:.4f} r={r_native:.4f}")
print(f"Logistic guess: dr/dt = -mu*r*(1-r)/2 = {-mu_native*r_native*(1-r_native)/2:.5f}")
print(f"BDI guess: dr/dt = -(mu-lam)*r = {-(mu_native-lam_native)*r_native:.5f}")
print()

print(f"{'t':>6}{'dr/dt':>12}{'selfcheck':>13}{'m0[I,I]':>10}{'m0[D,D]':>10}")
for t in [0.01, 0.05, 0.1, 0.3, 1.0, 2.0]:
    val, m0, m1, v, u, sc = dr_dt(lam_native, mu_native, r_native, lam_0, mu_0, x, y_g, t)
    print(f"{t:>6.3f}{val:>12.5f}{sc:>13.2e}{m0[I,I]:>10.4f}{m0[D,D]:>10.4f}")

# Sanity: TKF92-shape GGI (x=y) should give dr/dt = 0 at native
print("\nSanity: TKF92-shaped GGI (x=y=0.4) -- dr/dt should be 0")
y_g2 = 0.4
mu_native2 = 1.0*(1-x)/(1-y_g2)  # = 1.0
print(f"  native: lam={lam_native:.4f} mu={mu_native2:.4f} r={r_native:.4f}")
for t in [0.01, 0.1, 1.0]:
    val, m0, m1, v, u, sc = dr_dt(lam_native, mu_native2, r_native, lam_0, 1.0, x, y_g2, t)
    print(f"  t={t}: dr/dt={val:.6f}  selfcheck={sc:.2e}")
