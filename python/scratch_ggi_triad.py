"""Direct Triad HMM r-flow: build the 14-state matrix from the appendix spec,
extract m_XY via (I-t)^{-1}, solve the M-step implicit equation for r*,
compare dr/dt with logistic guess."""
import numpy as np
from scipy.optimize import brentq
from scratch_ggi_flow import tau5  # working TKF92 5x5 (kap, alpha, r)

S, M, I, D, E = 0, 1, 2, 3, 4

def tkf92_y(lam, mu, r, t):
    kap = lam/mu; alpha = np.exp(-mu*t)
    return tau5(kap, alpha, r)

def ggi_z(lam, mu, x, y_geom, dt):
    """6x6 GGI z matrix; indices (S,M,I,D,E,J)."""
    Z = np.zeros((6,6))
    # S row
    Z[0,1] = 1-(lam+mu)*dt; Z[0,3] = mu*dt; Z[0,4] = 1-lam*dt; Z[0,5] = lam*dt
    # M row
    Z[1,1] = 1-(lam+mu)*dt; Z[1,2] = lam*dt; Z[1,3] = mu*dt; Z[1,4] = 1-lam*dt
    # I row
    Z[2,1] = 1-x; Z[2,2] = x; Z[2,4] = 1-x
    # D row
    Z[3,1] = 1-y_geom; Z[3,3] = y_geom; Z[3,4] = 1
    # J row (same as I-row of z)
    Z[5,1] = 1-x; Z[5,4] = 1-x; Z[5,5] = x
    return Z

# Triad state indices
SS,sJ,MM,mI,MD,IM,iI,ID,Ds,Dj,Dm,Di,Dd,EE = range(14)
C_map = {SS:S, sJ:S, MM:M, mI:M, MD:M, IM:I, iI:I, ID:I, Ds:D, Dj:D, Dm:D, Di:D, Dd:D, EE:E}

def triad_t(lam, mu, r, x, y_geom, t, dt):
    Y = tkf92_y(lam, mu, r, t); Z = ggi_z(lam, mu, x, y_geom, dt)
    def y(a,b): return Y[a,b]
    def z(a,b): return Z[a,b]
    J = 5
    T = np.zeros((14,14))
    # Row SS
    T[SS,sJ]=z(S,J); T[SS,MM]=y(S,M)*z(S,M); T[SS,MD]=y(S,M)*z(S,D)
    T[SS,IM]=y(S,I)*z(S,M); T[SS,ID]=y(S,I)*z(S,D); T[SS,Ds]=y(S,D); T[SS,EE]=y(S,E)*z(S,E)
    # Row sJ
    T[sJ,sJ]=z(J,J); T[sJ,MM]=y(S,M)*z(J,M); T[sJ,MD]=y(S,M)*z(J,D)
    T[sJ,IM]=y(S,I)*z(J,M); T[sJ,ID]=y(S,I)*z(J,D); T[sJ,Dj]=y(S,D); T[sJ,EE]=y(S,E)*z(J,E)
    # Row MM
    T[MM,MM]=y(M,M)*z(M,M); T[MM,mI]=z(M,I); T[MM,MD]=y(M,M)*z(M,D)
    T[MM,IM]=y(M,I)*z(M,M); T[MM,ID]=y(M,I)*z(M,D); T[MM,Dm]=y(M,D); T[MM,EE]=y(M,E)*z(M,E)
    # Row mI
    T[mI,MM]=y(M,M)*z(I,M); T[mI,mI]=z(I,I); T[mI,MD]=y(M,M)*z(I,D)
    T[mI,IM]=y(M,I)*z(I,M); T[mI,ID]=y(M,I)*z(I,D); T[mI,Di]=y(M,D); T[mI,EE]=y(M,E)*z(I,E)
    # Row MD
    T[MD,MM]=y(M,M)*z(D,M); T[MD,mI]=z(D,I); T[MD,MD]=y(M,M)*z(D,D)
    T[MD,IM]=y(M,I)*z(D,M); T[MD,ID]=y(M,I)*z(D,D); T[MD,Dd]=y(M,D); T[MD,EE]=y(M,E)*z(D,E)
    # Row IM
    T[IM,MM]=y(I,M)*z(M,M); T[IM,MD]=y(I,M)*z(M,D); T[IM,IM]=y(I,I)*z(M,M)
    T[IM,iI]=z(M,I); T[IM,ID]=y(I,I)*z(M,D); T[IM,Dm]=y(I,D); T[IM,EE]=y(I,E)*z(M,E)
    # Row iI
    T[iI,MM]=y(I,M)*z(I,M); T[iI,MD]=y(I,M)*z(I,D); T[iI,IM]=y(I,I)*z(I,M)
    T[iI,iI]=z(I,I); T[iI,ID]=y(I,I)*z(I,D); T[iI,Di]=y(I,D); T[iI,EE]=y(I,E)*z(I,E)
    # Row ID
    T[ID,MM]=y(I,M)*z(D,M); T[ID,MD]=y(I,M)*z(D,D); T[ID,IM]=y(I,I)*z(D,M)
    T[ID,iI]=z(D,I); T[ID,ID]=y(I,I)*z(D,D); T[ID,Dd]=y(I,D); T[ID,EE]=y(I,E)*z(D,E)
    # Row Ds
    T[Ds,MM]=y(D,M)*z(S,M); T[Ds,MD]=y(D,M)*z(S,D); T[Ds,IM]=y(D,I)*z(S,M)
    T[Ds,ID]=y(D,I)*z(S,D); T[Ds,Ds]=y(D,D); T[Ds,EE]=y(D,E)*z(S,E)
    # Row Dj
    T[Dj,MM]=y(D,M)*z(J,M); T[Dj,MD]=y(D,M)*z(J,D); T[Dj,IM]=y(D,I)*z(J,M)
    T[Dj,ID]=y(D,I)*z(J,D); T[Dj,Dj]=y(D,D); T[Dj,EE]=y(D,E)*z(J,E)
    # Row Dm
    T[Dm,MM]=y(D,M)*z(M,M); T[Dm,MD]=y(D,M)*z(M,D); T[Dm,IM]=y(D,I)*z(M,M)
    T[Dm,ID]=y(D,I)*z(M,D); T[Dm,Dm]=y(D,D); T[Dm,EE]=y(D,E)*z(M,E)
    # Row Di
    T[Di,MM]=y(D,M)*z(I,M); T[Di,MD]=y(D,M)*z(I,D); T[Di,IM]=y(D,I)*z(I,M)
    T[Di,ID]=y(D,I)*z(I,D); T[Di,Di]=y(D,D); T[Di,EE]=y(D,E)*z(I,E)
    # Row Dd
    T[Dd,MM]=y(D,M)*z(D,M); T[Dd,MD]=y(D,M)*z(D,D); T[Dd,IM]=y(D,I)*z(D,M)
    T[Dd,ID]=y(D,I)*z(D,D); T[Dd,Dd]=y(D,D); T[Dd,EE]=y(D,E)*z(D,E)
    return T

def m_from_triad(T):
    R = np.linalg.inv(np.eye(14) - T)
    n = np.outer(R[SS,:], R[:,EE]) * T
    m = np.zeros((5,5))
    for A in range(14):
        for B in range(14):
            m[C_map[A], C_map[B]] += n[A,B]
    return m

def r_star(m, v_MM, v_II, v_DD):
    N_tot = m.sum()
    def f(r):
        v = [v_MM, v_II, v_DD]
        return sum(m[i+1,i+1]/(r+(1-r)*vi) for i,vi in enumerate(v)) - N_tot
    try:
        return brentq(f, 1e-7, 1-1e-7)
    except ValueError:
        # If implicit eq has no root in (0,1), return boundary
        if f(0.5) > 0: return None
        return None

# ==== Run ====
x, y_geom = 0.4, 0.55
mu_0 = 1.0
lam_0 = mu_0*x*(1-y_geom)/(y_geom*(1-x))
r_native = x
mu_native = mu_0*(1-r_native)/(1-y_geom)
ell_GGI = x/(y_geom-x)
kap_native = ell_GGI*(1-r_native)/(1+ell_GGI*(1-r_native))
lam_native = kap_native*mu_native
print(f"Native: lam={lam_native:.4f} mu={mu_native:.4f} r={r_native:.4f} kap={kap_native:.4f}")
print(f"Logistic guess: dr/dt(t=0) = -mu*r*(1-r)/2 = {-mu_native*r_native*(1-r_native)/2:.4f}")
print()

# sanity check: tau5 at t=0 should be near-identity-ish
Y0 = tkf92_y(lam_native, mu_native, r_native, 0)
print("tau5(t=0):"); print(np.round(Y0,4)); print()

# Verify m-projection at native: at dt=0, t=0+, the matched r should equal r_native
# (since the triad with surrogate=identity and GGI generator step should give the native match)
dt = 1e-5
print(f"{'t':>6}{'r*':>10}{'dr/dt num':>11}{'logistic':>11}")
for t in [1e-3, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0]:
    T = triad_t(lam_native, mu_native, r_native, x, y_geom, t, dt)
    m = m_from_triad(T)
    # v_TKF91: TKF92 with r=0
    Y_91 = tkf92_y(lam_native, mu_native, 0.0, t)
    v_MM, v_II, v_DD = Y_91[M,M], Y_91[I,I], Y_91[D,D]
    rs = r_star(m, v_MM, v_II, v_DD)
    if rs is None:
        print(f"{t:>6.3f}{'FAIL':>10}")
        continue
    drdt = (rs - r_native)/dt
    drdt_log = -mu_native*r_native*(1-r_native)/2
    print(f"{t:>6.3f}{rs:>10.4f}{drdt:>11.4f}{drdt_log:>11.4f}")
