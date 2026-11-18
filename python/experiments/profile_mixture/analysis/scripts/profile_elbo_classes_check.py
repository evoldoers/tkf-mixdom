# Numerical companion to Section 6 of analysis/profile_elbo/profile_elbo.tex:
# a K-class profile mixture (fixed pi^k, rate r_k) scored at O(K + A) per DP cell.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from profile_elbo_check import A, S, sigma, leaf, merge

rng = np.random.default_rng(7)
K = 6
P = rng.dirichlet(np.full(A, 0.5), K)          # class profiles pi^k
R = np.array([0.5, 1.0, 2.0, 0.5, 1.0, 2.0])   # class rates r_k
w = np.full(K, 1 / K)
lP, lR = np.log(P), np.log(R)
G = P @ S                                      # G[k, x] = (S pi^k)_x, exit rate of x in class k

def Phi(s):
    """Per-class complete-data bound, linear in the class-free summary s."""
    U = s['Vp'].sum() - 1.0
    return s['C'] + lP @ s['Vp'] + U * lR - R * (G @ s['T'])

def Phi_augmented(s):
    """Same quantity written with secret destinations (rejected-token term kept)."""
    E = s['T'] @ sigma
    wbar = E - s['T'] @ S
    U = s['Vp'].sum() - 1.0
    return s['C'] + lP @ s['Vp'] + U * lR - R * E + R * (P @ wbar)

def expm(Q, t):
    lam, V = np.linalg.eig(Q)
    return (V @ np.diag(np.exp(lam * t)) @ np.linalg.inv(V)).real

def exact(xs, t1, t2, t3):
    tot = 0.0
    for k in range(K):
        Q = R[k] * S * P[k][None]; np.fill_diagonal(Q, -Q.sum(1))
        tot += w[k] * (P[k] * expm(Q, t1)[:, xs[0]] * expm(Q, t2)[:, xs[1]]
                       * expm(Q, t3)[:, xs[2]]).sum()
    return np.log(tot)

if __name__ == '__main__':
    logS = np.log(np.where(S > 0, S, 1.0))
    t1, t2, tP, t3 = 0.1, 0.15, 0.05, 0.2
    t = tP + t3
    for xs in [(0, 0, 0), (0, 0, 1), (0, 1, 2), (1, 3, 1), (2, 2, 3)]:
        SA, SB = merge(leaf(xs[0]), leaf(xs[1]), t1, t2), leaf(xs[2])
        full = Phi(merge(SA, SB, tP, t3))
        assert np.allclose(full, Phi_augmented(merge(SA, SB, tP, t3)))
        qA, qB = SA['q'], SB['q']
        # per-column vectors (precomputed once per column, O(KA))
        a = np.log(w) + Phi(SA) - R * (t / 2) * (G @ qA)
        b = Phi(SB) - R * (t / 2) * (G @ qB) + lR
        # class-free bridge term: one A-dot with a per-column row vector
        off = logS + np.log(t); np.fill_diagonal(off, 0)
        ell = qA @ off
        bridge = ell @ qB
        # cross term: >= 0 part (log pi) and signed rate part
        crossP = -(lP @ (qA * qB))
        crossR = -(qA @ qB) * lR
        assert np.allclose(a - np.log(w) + b + bridge + crossP + crossR, full)
        lse = lambda F: np.log(np.exp(F) @ np.ones(K))
        ex = exact(xs, t1, t2, t)
        bound = lse(a + b + crossP + crossR) + bridge
        mask = qA * qB > 1e-3                      # sparse overlap
        crossPs = -(lP[:, mask] @ (qA * qB)[mask])
        sparse = np.log(np.exp(a + crossR + crossPs) @ np.exp(b)) + bridge
        nocross = np.log(np.exp(a + crossR) @ np.exp(b)) + bridge
        print(xs, 'exact %.4f  full %.4f  sparse %.4f (|overlap|=%d)  no log-pi cross %.4f  valid %s'
              % (ex, bound, sparse, mask.sum(), nocross, ex >= bound >= sparse - 1e-12 >= nocross - 1e-12))
