"""numeric_check.py -- numerical CHECKS (not proof) for Q3, against two ground truths.

(1) Printed (17) vs corrected (17'): evaluate the printed right-hand side by mpmath quadrature,
    using the (proved) closed forms for Z_*, Z_o* and Holmes's kernel c(t|T) as printed;
    compare the excess with int_0^T beta / beta(T).
(2) Exact E-step (the paper's closed form, tkfmixdom bdi.tkf91_stats_from_counts, float64):
    for random pair-HMM paths, compare E[B], E[D], E[S] with Holmes's zone sums using
    (a) corrected (17') and (b) printed (17), with and without the '-T' immortal-link term.
(3) A port of Holmes's own implementation (DART src/tkf/tkfcoeff.cc, TKF_coeff::initialise,
    first-order scheme) at several step sizes: its Z11 converges to Z_** (no 't +').
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
import numpy as np
import mpmath as mp

mp.mp.dps = 60


def factors(lam, mu, s):
    lam, mu, s = mp.mpf(lam), mp.mpf(mu), mp.mpf(s)
    eta, al = mp.e ** (-lam * s), mp.e ** (-mu * s)
    beta = lam * (eta - al) / (mu * eta - lam * al)
    gamma = 1 - mu * beta / (lam * (1 - al)) if s > 0 else mp.mpf(0)
    return al, beta, gamma, eta


def closed(lam, mu, T):
    lam, mu, T = mp.mpf(lam), mp.mpf(mu), mp.mpf(T)
    al, beta, gamma, eta = factors(lam, mu, T)
    k = lam - mu
    d = mu * eta - lam * al
    eps = lam * (1 - al) - mu * (1 - eta)
    Io = lam * mu * (eta**2 - al**2 + 2 * k * T * eta * al) / (k * (al - eta) * d)
    Zo = (mu * eta**2 - lam * al**2 + k * eta * al + k * (lam + mu) * T * eta * al) / (k * (al - eta) * d)
    Is = lam * mu * ((lam + mu) * (al - eta) - k * T * (lam * al + mu * eta)) / (k**2 * d)
    Zs = (2 * lam * mu * (al - eta) - k * T * (lam**2 * al + mu**2 * eta)) / (k**2 * d)
    Ios = k / eps * Is
    Zos = lam * mu * (2 * (al - eta) - k * T * (al + eta)) / (k * d * eps)
    return dict(o=(Io, Zo), s=(Is, Zs), os=(Ios, Zos), ss=(1 + Io, Zo))


def rates(lam, mu, s):
    al, beta, gamma, _ = factors(lam, mu, s)
    return lam * al, lam * (1 - al) * gamma


def printed17(lam, mu, T):
    _, bT, _, _ = factors(lam, mu, T)
    def integrand(t):
        s = T - t
        if s <= mp.mpf('1e-18'):
            return mp.mpf(0)   # integrand is O(t * s) there; dropped mass < 1e-17
        _, bs, _, _ = factors(lam, mu, s)
        c = (1 - bs) / bT
        lv, lw = rates(lam, mu, s)
        C = closed(lam, mu, s)
        Zs, Zos = C['s'][1], C['os'][1]
        return c * (lv * (t + Zs) + lw * (t + Zs + Zos))
    return mp.quad(integrand, [0, T / 2, T])


print("=" * 100)
print("(1) printed (17) minus Z_** (closed form), against int_0^T beta / beta(T)")
print("=" * 100)
print(f"{'lam':>5} {'mu':>5} {'T':>5} {'Z_** (corr.)':>14} {'printed (17)':>14} {'excess':>12} {'intbeta/beta':>12} {'diff':>9}")
GRID = [(1.0, 0.5, 0.1), (1.0, 0.5, 1.0), (1.0, 0.5, 3.0), (0.5, 1.0, 1.0), (2.0, 0.3, 1.0),
        (0.3, 2.0, 1.0), (1.0, 1.3, 2.5), (0.05, 0.08, 1.0)]
for lam, mu, T in GRID:
    Zss = closed(lam, mu, T)['ss'][1]
    P17 = printed17(lam, mu, T)
    _, bT, _, _ = factors(lam, mu, T)
    pred = (lam * T + mp.log(1 - bT)) / (mu * bT)
    print(f"{lam:5.2f} {mu:5.2f} {T:5.2f} {float(Zss):14.8f} {float(P17):14.8f} {float(P17-Zss):12.8f} {float(pred):12.8f} {float(P17-Zss-pred):9.1e}")

print()
print("=" * 100)
print("(2) zone sums vs the exact E-step  bdi.tkf91_stats_from_counts  (float64)")
print("=" * 100)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from tkfmixdom.jax.core import bdi
from tkfmixdom.jax.core.params import S, M, I, D, E

ZONE = {}
for a in 'SMI':
    for b in 'MDE':
        ZONE[(a, b)] = 's'
    ZONE[(a, 'I')] = 'ss'
for b in 'MDE':
    ZONE[('D', b)] = 'o'
ZONE[('D', 'I')] = 'os'
IDX = dict(S=S, M=M, I=I, D=D, E=E)
rng = np.random.default_rng(1)
for lam, mu, T in [(1.0, 0.5, 1.0), (0.5, 1.0, 1.0), (2.0, 0.3, 0.7), (0.04, 0.05, 2.0)]:
    C = {k: (float(v[0]), float(v[1])) for k, v in closed(lam, mu, T).items()}
    _, bT, _, _ = factors(lam, mu, T)
    excess = float((lam * T + mp.log(1 - bT)) / (mu * bT))
    err = dict(B=0., D=0., S=0., offT=0., p17=0.)
    for rep in range(200):
        L = rng.integers(0, 25)
        states = rng.choice(list('MID'), size=L)
        path = ['S'] + list(states) + ['E']
        n = np.zeros((5, 5))
        N = dict(o=0, s=0, os=0, ss=0)
        for a, b in zip(path[:-1], path[1:]):
            n[IDX[a], IDX[b]] += 1
            N[ZONE[(a, b)]] += 1
        EB, ED, ES = [float(x) for x in bdi.tkf91_stats_from_counts(n, lam, mu, T)]
        Ih = sum(N[k] * C[k][0] for k in N)
        Dh = Ih + N['o'] - N['ss']
        Zh = sum(N[k] * C[k][1] for k in N)            # sec 5.4 sum, corrected (17')
        Zp = Zh + N['ss'] * excess                     # sec 5.4 sum, printed (17)
        err['B'] = max(err['B'], abs(Ih - EB))
        err['D'] = max(err['D'], abs(Dh - ED))
        err['S'] = max(err['S'], abs(Zh - T - ES))               # Z - T == E[S] ?
        err['offT'] = max(err['offT'], abs((Zh - ES) - T))       # sec 5.4 as printed is off by exactly T
        err['p17'] = max(err['p17'], abs((Zp - T - ES) - N['ss'] * excess))   # printed (17): off by N_** x excess
    print(f"lam={lam} mu={mu} T={T} (200 random paths, length<=24):")
    print(f"   max|I - E[B]| = {err['B']:.1e}   max|D - E[D]| = {err['D']:.1e}   "
          f"max|(Z - T) - E[S]| = {err['S']:.1e}   [corrected (17'), with -T]")
    print(f"   max|(Z - E[S]) - T| = {err['offT']:.1e}   [sec 5.4 sum as printed overshoots E[S] by exactly T]")
    print(f"   max|(Z_printed17 - T - E[S]) - N_** x {excess:.6f}| = {err['p17']:.1e}   "
          f"[printed (17) overshoots by N_** x int beta/beta]")

print()
print("=" * 100)
print("(3) port of DART tkfcoeff.cc (Holmes's implementation, first-order scheme)")
print("=" * 100)


def dart_coeffs(lam, mu, T, Nsteps):
    dt = T / Nsteps
    Nn = Nsteps + 1
    tt = dt * np.arange(Nn)
    x = np.exp((lam - mu) * tt); y = np.exp(-mu * tt)
    zz = (1 - x) / (mu - lam * x)
    alpha = y; beta = lam * zz
    gamma = np.zeros(Nn); gamma[1:] = 1 - mu * zz[1:] / (1 - y[1:])
    lD = lam * dt * alpha; lI = lam * dt * (1 - alpha) * gamma; lE = lam * dt * (1 - alpha) * (1 - gamma)
    I0 = np.zeros(Nn); I1 = np.zeros(Nn); I01 = np.zeros(Nn); I11 = np.zeros(Nn)
    Z0 = np.zeros(Nn); Z1 = np.zeros(Nn); Z01 = np.zeros(Nn); Z11 = np.zeros(Nn)
    for n in range(1, Nn):
        m = np.arange(n)
        tD, tI, tE = lD[n - m], lI[n - m], lE[n - m]
        v = np.cumprod(1 - (mu * dt + tD + tI))
        if n < 2:
            a = np.ones(n); b = np.zeros(n)
        else:
            a = v * (1 - alpha[n - m - 1]) * (1 - gamma[n - m - 1]) / ((1 - alpha[n]) * (1 - gamma[n]))
            b = alpha[m] * (1 - alpha[n - m - 1]) * (1 - gamma[n - m - 1]) / ((1 - alpha[n]) * gamma[n])
        c = (1 - beta[n - m - 1]) / beta[n]
        # the recursion is explicit in n (only indices n-m-1 <= n-1 are used)
        i0, i1, i01 = I0[n - m - 1], I1[n - m - 1], I01[n - m - 1]
        z0, z1, z01 = Z0[n - m - 1], Z1[n - m - 1], Z01[n - m - 1]
        I0[n] = np.sum(a * tE * (1 + i0)); I1[n] = np.sum(tE * (1 + i0))
        I01[n] = np.sum(b * (tD * (1 + i0) + tI * (1 + i0 + i01)))
        I11[n] = np.sum(c * (tD * (1 + i1) + tI * (1 + i1 + i01)))
        Z0[n] = np.sum(a * (dt + tE * z0)); Z1[n] = np.sum(dt + tE * z0)
        Z01[n] = np.sum(b * (tD * z0 + tI * (z0 + z01)))
        Z11[n] = np.sum(c * (tD * z1 + tI * (z1 + z01)))
    return dict(o=(I0[-1], Z0[-1]), s=(I1[-1], Z1[-1]), os=(I01[-1], Z01[-1]), ss=(I11[-1], Z11[-1]))


for lam, mu, T in [(1.0, 0.5, 1.0), (2.0, 0.3, 1.0)]:
    C = {k: (float(v[0]), float(v[1])) for k, v in closed(lam, mu, T).items()}
    _, bT, _, _ = factors(lam, mu, T)
    excess = float((lam * T + mp.log(1 - bT)) / (mu * bT))
    print(f"lam={lam} mu={mu} T={T}:  closed Z_** = {C['ss'][1]:.6f}, printed-(17) value = {C['ss'][1]+excess:.6f}")
    prev = None
    for Ns in (250, 500, 1000, 2000):
        dc = dart_coeffs(lam, mu, T, Ns)
        errs = {k: max(abs(dc[k][0] - C[k][0]), abs(dc[k][1] - C[k][1])) for k in C}
        line = "  ".join(f"{k}:{errs[k]:.2e}" for k in C)
        print(f"   N={Ns:5d}  max|DART - closed| per zone  {line}   Z11={dc['ss'][1]:.6f}")
