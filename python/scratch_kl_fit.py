"""
Stage 1 of the KL m-projection fit for TKF92 self-composition.

Builds the TKF92 pair-HMM tau'(kappa, alpha, r) [t enters via alpha=e^{-mu t},
beta,gamma functions of (kappa,alpha)], computes a single TKF92's expected
transition counts exactly via the fundamental matrix (I-Q)^{-1}, and verifies
that the KL m-projection (maximize sum Nbar log tau') recovers the generating
parameters (fit-to-self). This validates the M-step before we feed it the
composite's counts.

Indices: S=0, M=1, I=2, D=3, E=4.
"""
import numpy as np
from scipy.optimize import minimize

S, M, I, D, E = 0, 1, 2, 3, 4

def beta_gamma(kappa, alpha):
    a1 = alpha ** (1 - kappa)
    beta = kappa * (1 - a1) / (1 - kappa * a1)
    gamma = 1 - beta / (kappa * (1 - alpha))
    return beta, gamma

def tau_tkf92(kappa, alpha, r):
    """5x5 TKF92 pair-HMM transition matrix (S,M,I,D,E)."""
    b, g = beta_gamma(kappa, alpha)
    ob, og, ok = 1 - b, 1 - g, 1 - kappa
    T = np.zeros((5, 5))
    # S row (no extension)
    T[S] = [0, ob * kappa * alpha, b, ob * kappa * (1 - alpha), ob * ok]
    # M, I rows (beta context) with extension r on the self-loop
    for src, self_col in [(M, M), (I, I)]:
        row = np.array([0, ob * kappa * alpha, b, ob * kappa * (1 - alpha), ob * ok])
        row = (1 - r) * row
        row[self_col] += r
        T[src] = row
    # D row (gamma context) with extension r on D self-loop
    rowd = np.array([0, og * kappa * alpha, g, og * kappa * (1 - alpha), og * ok])
    rowd = (1 - r) * rowd
    rowd[D] += r
    T[D] = rowd
    return T

def expected_counts(kappa, alpha, r):
    """Expected transition counts Nbar[s,s'] for one S->E alignment."""
    T = tau_tkf92(kappa, alpha, r)
    Q = T[np.ix_([M, I, D], [M, I, D])]      # transient block
    p0 = T[S, [M, I, D]]                       # entry distribution from S
    visits = p0 @ np.linalg.inv(np.eye(3) - Q)  # expected visits to M,I,D
    v = {S: 1.0, M: visits[0], I: visits[1], D: visits[2]}
    N = np.zeros((5, 5))
    for s in [S, M, I, D]:
        N[s] = v[s] * T[s]
    return N

def kl_fit(Nbar, theta0=(0.5, 0.5, 0.3)):
    """Maximize sum Nbar log tau' over (kappa, alpha, r): the KL m-projection."""
    def negloglik(x):
        k, a, r = x
        T = tau_tkf92(k, a, r)
        with np.errstate(divide='ignore', invalid='ignore'):
            logT = np.where(T > 0, np.log(np.clip(T, 1e-300, None)), 0.0)
        return -np.sum(Nbar * logT)
    bnds = [(1e-4, 1 - 1e-4)] * 3
    res = minimize(negloglik, theta0, bounds=bnds, method='L-BFGS-B',
                   options=dict(ftol=1e-14, gtol=1e-12, maxiter=5000))
    return res.x

print("=== fit-to-self validation (should recover the generating params) ===")
print(f"{'kappa,alpha,r (true)':>28} -> {'fitted':>30}   max|err|")
ok_all = True
for theta0 in [(0.5, 0.6, 0.3), (0.3, 0.8, 0.5), (0.7, 0.4, 0.1), (0.2, 0.9, 0.6),
               (0.5, 0.5, 0.0), (0.6, 0.3, 0.05)]:
    N = expected_counts(*theta0)
    fit = kl_fit(N, theta0=(0.4, 0.5, 0.2))
    err = np.max(np.abs(np.array(theta0) - fit))
    ok = err < 1e-4
    ok_all = ok_all and ok
    print(f"{str(tuple(round(x,3) for x in theta0)):>28} -> "
          f"{str(tuple(round(x,5) for x in fit)):>30}   {err:.2e}  {'OK' if ok else 'FAIL'}")
print("\nALL PASS" if ok_all else "\nSOME FAILED")
