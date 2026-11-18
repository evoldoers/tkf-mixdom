# Numerical companion to analysis/profile_elbo/profile_elbo.tex.
# Check: collapsed-Dirichlet ELBO for per-column pi ~ Dir(alpha), GTR with shared S,
# using secret-destination augmentation with rejected tokens optimised analytically.
import numpy as np
from scipy.special import gammaln, digamma
rng = np.random.default_rng(1)
A = 4
S = rng.gamma(2., 1., (A, A)); S = (S + S.T) / 2; np.fill_diagonal(S, 0)
sigma = S.sum(1)

def leaf(x):
    e = np.eye(A)[x]
    return dict(Vp=e.copy(), T=np.zeros(A), C=0.0, q=e.copy())

def merge(SA, SB, tA, tB):
    # bridge q: a~qA, b~qB independent; if a!=b one jump at U(0,t), else no jump
    t = tA + tB; qA, qB = SA['q'], SB['q']
    offdiag = np.outer(qA, qB); np.fill_diagonal(offdiag, 0)
    logS = np.log(np.where(S > 0, S, 1.0))
    Vp = SA['Vp'] + SB['Vp'] - qB + qB * (1 - qA)
    T = SA['T'] + SB['T'] + 0.5 * t * (qA + qB)
    C = SA['C'] + SB['C'] + (offdiag * (logS + np.log(t))).sum()
    q = qA * qB + (tB / t) * qA * (1 - qB) + (tA / t) * qB * (1 - qA)
    return dict(Vp=Vp, T=T, C=C, q=q)

def lB(a): return gammaln(a).sum() - gammaln(a.sum())

def F(s, alpha, beta):
    psib = digamma(beta) - digamma(beta.sum()); pit = np.exp(psib)
    Ts = s['T'] @ sigma; wbar = Ts - s['T'] @ S
    KL = lB(alpha) - lB(beta) + ((beta - alpha) * psib).sum()
    return s['C'] - Ts + s['Vp'] @ psib + pit @ wbar - KL

def fixed_point(s, alpha, iters=50):
    beta = alpha + s['Vp']; out = []
    for _ in range(iters):
        out.append(F(s, alpha, beta))
        psib = digamma(beta) - digamma(beta.sum()); pit = np.exp(psib)
        Vpp = pit * (s['T'] @ sigma - s['T'] @ S)
        beta = alpha + s['Vp'] + Vpp
    return np.array(out)

def exact(xs, t1, t2, t3, alpha, N=400000):
    pis = rng.dirichlet(alpha, N)
    sq = np.sqrt(pis)
    Qt = sq[:, :, None] * S[None] * sq[:, None, :]
    rate = (S[None] * pis[:, None, :]).sum(2)
    Qt[:, np.arange(A), np.arange(A)] = -rate
    lam, U = np.linalg.eigh(Qt)
    def M(t):
        Mt = np.einsum('nik,nk,njk->nij', U, np.exp(lam * t), U)
        return Mt * (1 / sq)[:, :, None] * sq[:, None, :]
    x1, x2, x3 = xs
    L = (pis * M(t1)[:, :, x1] * M(t2)[:, :, x2] * M(t3)[:, :, x3]).sum(1)
    return np.log(L.mean()), np.log(L).std() / np.sqrt(N)


def Fr(s, alpha, beta, a0, b0, a1, b1):
    psib = digamma(beta) - digamma(beta.sum()); pit = np.exp(psib)
    lr = digamma(a1) - np.log(b1); rt = np.exp(lr); rbar = a1 / b1
    E = s['T'] @ sigma; wbar = E - s['T'] @ S
    Uacc = s['Vp'].sum() - 1.0            # accepted jumps = tokens minus the root token
    KLD = lB(alpha) - lB(beta) + ((beta - alpha) * psib).sum()
    KLG = (a1 - a0) * digamma(a1) - gammaln(a1) + gammaln(a0) + a0 * (np.log(b1) - np.log(b0)) + a1 * (b0 - b1) / b1
    return s['C'] + s['Vp'] @ psib + Uacc * lr - rbar * E + rt * (pit @ wbar) - KLD - KLG

def fp(s, alpha, a0, b0, iters=60):
    E = s['T'] @ sigma; wbar = E - s['T'] @ S; Uacc = s['Vp'].sum() - 1.0
    beta, a1, b1 = alpha + s['Vp'], a0 + Uacc, b0 + E
    out = []
    for _ in range(iters):
        out.append(Fr(s, alpha, beta, a0, b0, a1, b1))
        psib = digamma(beta) - digamma(beta.sum()); rt = np.exp(digamma(a1) - np.log(b1))
        Vpp = rt * np.exp(psib) * wbar
        beta, a1, b1 = alpha + s['Vp'] + Vpp, a0 + Uacc + Vpp.sum(), b0 + E
    # closed form at the fixed point
    psib = digamma(beta) - digamma(beta.sum()); lr = digamma(a1) - np.log(b1)
    Vpp = np.exp(lr + psib) * wbar
    closed = (s['C'] + lB(beta) - lB(alpha) + gammaln(a1) - gammaln(a0) + a0*np.log(b0) - a1*np.log(b1)
              + (Vpp * (1 - psib - lr)).sum())
    return np.array(out), closed

def exact_r(xs, t1, t2, t3, alpha, a0, b0, N=400000):
    pis = rng.dirichlet(alpha, N); r = rng.gamma(a0, 1 / b0, N)
    sq = np.sqrt(pis)
    Qt = r[:, None, None] * sq[:, :, None] * S[None] * sq[:, None, :]
    Qt[:, np.arange(A), np.arange(A)] = -r[:, None] * (S[None] * pis[:, None, :]).sum(2)
    lam, U = np.linalg.eigh(Qt)
    M = lambda t: np.einsum('nik,nk,njk->nij', U, np.exp(lam * t), U) * (1 / sq)[:, :, None] * sq[:, None, :]
    x1, x2, x3 = xs
    return np.log((pis * M(t1)[:, :, x1] * M(t2)[:, :, x2] * M(t3)[:, :, x3]).sum(1).mean())

if __name__ == '__main__':
    print('== fixed rate ==')
    alpha = np.full(A, 0.5)
    t1, t2, tP, t3 = 0.1, 0.15, 0.05, 0.2
    for xs in [(0, 0, 0), (0, 0, 1), (0, 1, 2), (2, 2, 2), (1, 3, 1)]:
        s = merge(merge(leaf(xs[0]), leaf(xs[1]), t1, t2), leaf(xs[2]), tP, t3)
        tr = fixed_point(s, alpha)
        ex, _ = exact(xs, t1, t2, tP + t3, alpha)
        rb = [F(s, alpha, rng.gamma(1, 3, A) + 0.1) for _ in range(2000)]
        print(xs, 'exact %.4f  ELBO it1 %.4f it2 %.4f conv %.4f  monotone %s  max(random beta) %.4f  <=exact %s'
              % (ex, tr[0], tr[1], tr[-1], np.all(np.diff(tr) > -1e-10), max(rb), tr[-1] <= ex and max(rb) <= ex))
    print('== r ~ Gamma(2, 2) ==')
    a0 = b0 = 2.0
    t1, t2, tP, t3 = 0.1, 0.15, 0.05, 0.2
    for xs in [(0, 0, 0), (0, 0, 1), (0, 1, 2), (1, 3, 1)]:
        s = merge(merge(leaf(xs[0]), leaf(xs[1]), t1, t2), leaf(xs[2]), tP, t3)
        tr, closed = fp(s, alpha, a0, b0)
        ex = exact_r(xs, t1, t2, tP + t3, alpha, a0, b0)
        rb = max(Fr(s, alpha, rng.gamma(1, 3, A) + .1, a0, b0, rng.gamma(2, 2) + .1, rng.gamma(2, 2) + .1) for _ in range(2000))
        print(xs, 'exact %.4f  it1 %.4f it2 %.4f conv %.4f closed %.4f  monotone %s  bound %s'
              % (ex, tr[0], tr[1], tr[-1], closed, np.all(np.diff(tr) > -1e-10), tr[-1] <= ex and rb <= ex))
