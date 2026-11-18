"""Check the MixFrag bigram distance (analysis/profile_elbo/profile_classes.tex).

Simulates sequence pairs by walking the MixFrag Pair HMM (X = ancestor, Y =
descendant after time t), then
  1. compares the mean of D2 = sum_ab cX(ab) cY(ab) with the closed form
       E[D2] ~ rho^2 N + alpha (L_X - 1) (s - rho) [2 rho + (s - rho) phi],
       phi = 1 - (1 - (1 - beta) alpha) / lbar,
     and the unigram analogue E[D1] = rho L_X L_Y + alpha L_X (s - rho);
  2. inverts each for t per pair (bisection) and reports bias and spread.

Substitution: LG exchangeabilities with either LG frequencies (K=1) or the C20
profile mixture (class drawn per column), each class normalised to rate 1.

Usage: python analysis/scripts/bigram_distance_check.py [--c20] [--reps 400]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_classes_pfam import load_model  # LG S and C20 profiles, numpy only

A = 20


def tkf91(lam, mu, t):
    alpha = np.exp(-mu * t)
    beta = (lam * np.exp(-lam * t) - lam * alpha) / (mu * np.exp(-lam * t) - lam * alpha)
    gamma = 1 - mu * beta / (lam * (1 - alpha))
    kappa = lam / mu
    row = lambda b: np.array([(1 - b) * alpha * kappa, b, (1 - b) * (1 - alpha) * kappa, (1 - b) * (1 - kappa)])
    # columns M, I, D, E; rows S, M, I, D
    return alpha, beta, np.vstack([row(beta), row(beta), row(beta), row(gamma)])


class Subst:
    def __init__(self, c20):
        S, P, w, R = load_model()
        if not c20:
            # LG frequencies as a single class
            import importlib.util, types
            sys.modules["jax"] = types.ModuleType("jax"); sys.modules["jax.numpy"] = np
            spec = importlib.util.spec_from_file_location(
                "lg08", os.path.join(os.path.dirname(__file__), "../../src/lg08.py"))
            lg = importlib.util.module_from_spec(spec); spec.loader.exec_module(lg)
            del sys.modules["jax"], sys.modules["jax.numpy"]
            P = np.asarray(lg.get_lg08()[1], float)[None]
            P /= P.sum()
            w = np.ones(1)
            R = 1.0 / np.einsum("kx,xy,ky->k", P, S, P)
        self.P, self.w, self.R, self.S = P, w, R, S
        self.K = len(w)
        self.pibar = w @ P
        self.rho = float(self.pibar @ self.pibar)
        sq = np.sqrt(P)
        G = P @ S
        B = R[:, None, None] * sq[:, :, None] * S[None] * sq[:, None, :]
        B[:, np.arange(A), np.arange(A)] = -R[:, None] * G
        self.lam, self.U = np.linalg.eigh(B)
        self.sq = sq
        # s(t) = sum_k w_k sum_a pi_a M_aa(t) = sum_k w_k sum_i c_ki e^{lam_ki t}
        self.c = np.einsum("k,ka,kai->ki", w, P, self.U ** 2)

    def s(self, t):
        return float((self.c * np.exp(self.lam * t)).sum())

    def M(self, k, t):
        U, l, sq = self.U[k], self.lam[k], self.sq[k]
        return (U * np.exp(l * t)) @ U.T * (sq[None, :] / sq[:, None])


def simulate_pair(rng, lam, mu, t, r, wf, sub, Mt):
    alpha, beta, tau = tkf91(lam, mu, t)
    F = len(r)
    x, y = [], []
    state, frag = 0, None   # 0=S, 1=M, 2=I, 3=D
    while True:
        if state != 0 and rng.random() < r[frag]:
            nxt = state
        else:
            p = tau[state]
            nxt = rng.choice(4, p=p / p.sum()) + 1   # 1..4 = M, I, D, E
            if nxt == 4:
                break
            frag = rng.choice(F, p=wf)
        state = nxt
        k = rng.choice(sub.K, p=sub.w)
        if state == 1:
            a = rng.choice(A, p=sub.P[k]); b = rng.choice(A, p=Mt[k][a] / Mt[k][a].sum())
            x.append(a); y.append(b)
        elif state == 2:
            y.append(rng.choice(A, p=sub.P[k]))
        else:
            x.append(rng.choice(A, p=sub.P[k]))
    return np.array(x, int), np.array(y, int)


def counts(z, k=2):
    """Unigram and k-gram counts (k = 2 or 3)."""
    c1 = np.bincount(z, minlength=A).astype(float)
    if len(z) < k:
        return c1, np.zeros(A ** k)
    idx = np.zeros(len(z) - k + 1, int)
    for o in range(k):
        idx = idx * A + z[o:len(z) - k + 1 + o]
    return c1, np.bincount(idx, minlength=A ** k).astype(float)


def adjacency_run(t, lam, mu, frag, k):
    """P(k-1 consecutive ancestral adjacencies all preserved | first residue survives).
    Transfer over the fragtype of the current residue: stay in the fragment (r_f),
    or cross a boundary (1-r_f), preserved with prob (1-beta) alpha, new fragtype ~ w."""
    r, wf = frag
    alpha, beta, _ = tkf91(lam, mu, t)
    c = (1 - beta) * alpha
    omega = wf / (1 - r); omega = omega / omega.sum()
    T = np.diag(r) + np.outer((1 - r) * c, wf)
    return float(omega @ np.linalg.matrix_power(T, k - 1) @ np.ones(len(r)))


def expected(t, LX, LY, lam, mu, frag, sub, which):
    """E[D_which] for which = 1 (unigrams), 2 (bigrams) or 3 (trigrams)."""
    alpha, _, _ = tkf91(lam, mu, t)
    s, rho = sub.s(t), sub.rho
    if which == 1:
        return rho * LX * LY + alpha * LX * (s - rho)
    if which == 2:
        phi = adjacency_run(t, lam, mu, frag, 2)
        return rho ** 2 * (LX - 1) * (LY - 1) + alpha * (LX - 1) * (s - rho) * (2 * rho + (s - rho) * phi)
    # trigrams: homology patterns (h1,h2,h3) along a diagonal; (1,0,1) neglected
    n = alpha * (LX - 2)
    p2, p3 = adjacency_run(t, lam, mu, frag, 2), adjacency_run(t, lam, mu, frag, 3)
    a1, b2, c3 = n, n * p2, n * p3
    ex = (c3 * (s ** 3 - rho ** 3) + 2 * (b2 - c3) * (s * s * rho - rho ** 3)
          + (2 * (a1 - b2) + (a1 - 2 * b2 + c3)) * (s * rho * rho - rho ** 3))
    return rho ** 3 * (LX - 2) * (LY - 2) + ex


def invert(obs, LX, LY, lam, mu, frag, sub, which, tmax=20.0):
    f = lambda t: expected(t, LX, LY, lam, mu, frag, sub, which) - obs
    lo, hi = 1e-4, tmax
    if f(lo) <= 0:
        return lo
    if f(hi) >= 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c20", action="store_true")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    sub = Subst(args.c20)
    mu = 0.05
    r = np.array([0.25, 0.75]); wf = np.array([0.65, 0.35])
    lbar = float(wf @ (1 / (1 - r)))   # mean fragment length (reported only)
    links = 130.0
    lam = mu * links / (links + 1)          # mean ~130 fragments, ~300 residues
    print(f"model: {'C20' if args.c20 else 'LG'}, mu={mu}, lam={lam:.5f}, "
          f"r={r.tolist()}, w={wf.tolist()}, mean fragment length {lbar:.3f}, rho={sub.rho:.4f}")
    frag = (r, wf)
    print(f"{'t':>5} {'<L_X>':>6} {'D2 obs/pred':>11} {'D3 obs/pred':>11} | "
          f"{'bigram t med [IQR]':>24} {'trigram t med [IQR]':>24} {'unigram t med [IQR]':>24}")
    for t in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        Mt = [sub.M(k, t) for k in range(sub.K)]
        o2, p2, o3, p3, e1, e2, e3, lx = [], [], [], [], [], [], [], []
        for _ in range(args.reps):
            x, y = simulate_pair(rng, lam, mu, t, r, wf, sub, Mt)
            if len(x) < 3 or len(y) < 3:
                continue
            cx1, cx2 = counts(x, 2); cy1, cy2 = counts(y, 2)
            _, cx3 = counts(x, 3); _, cy3 = counts(y, 3)
            d1, d2, d3 = cx1 @ cy1, cx2 @ cy2, cx3 @ cy3
            LX, LY = len(x), len(y)
            o2.append(d2); p2.append(expected(t, LX, LY, lam, mu, frag, sub, 2))
            o3.append(d3); p3.append(expected(t, LX, LY, lam, mu, frag, sub, 3))
            e1.append(invert(d1, LX, LY, lam, mu, frag, sub, 1))
            e2.append(invert(d2, LX, LY, lam, mu, frag, sub, 2))
            e3.append(invert(d3, LX, LY, lam, mu, frag, sub, 3))
            lx.append(LX)
        fmt = lambda v: "{1:>6.2f} [{0:.2f},{2:.2f}]".format(*np.percentile(v, [25, 50, 75])).rjust(24)
        print(f"{t:>5.2f} {np.mean(lx):>6.0f} {np.mean(o2) / np.mean(p2):>11.3f} "
              f"{np.mean(o3) / np.mean(p3):>11.3f} | {fmt(e2)} {fmt(e3)} {fmt(e1)}")


if __name__ == "__main__":
    main()
