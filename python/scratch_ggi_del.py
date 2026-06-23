"""Exact (resolvent) deletion increment to Nbar, validated vs route A.

A GGI deletion run covers k contiguous Y residues (rate mu0 y^{k-1}(1-y)); each
covered M-column -> D, each covered I-column -> removed; interspersed leg-1 D
columns stay D.  We accumulate E_surr[ sum_{existing runs} (Nbar_after - Nbar_before) ]
exactly by a Markov-reward scan over the surrogate alignment chain: each frontier
state carries the run's expected PARTIAL transition-count change, committed only
when the run TERMINATES (so over-length runs that run past the sequence end -
which do not exist - contribute nothing).  Interspersed D-runs and the geometric
run length are marginalised; truncations are exact up to exponentially small tails.
"""
import numpy as np
import scratch_ggi_flow as G
S, M, I, D, E = 0, 1, 2, 3, 4


def del_increment_exact(kap, alpha, r, mu0, y, KMAX=4000, DMAX=600):
    T = G.tau5(kap, alpha, r)
    Q = T[np.ix_([M, I, D], [M, I, D])]
    v = T[S, [M, I, D]] @ np.linalg.inv(np.eye(3) - Q)
    visit = {S: 1.0, M: v[0], I: v[1], D: v[2]}
    dll = T[D, D]

    out = np.zeros((5, 5))   # committed E[after - before]

    # Frontier: key (anchorL, last_before_col, last_after_isD) ->
    #           [w, B(5x5), A(5x5)]   (weight-summed partial before/after tallies,
    #           NOT yet including the run's right-junction -> R).
    def newval():
        return [0.0, np.zeros((5, 5)), np.zeros((5, 5))]

    def add_state(dst, key, w, B, A):
        s = dst.get(key)
        if s is None:
            dst[key] = [w, B.copy(), A.copy()]
        else:
            s[0] += w; s[1] += B; s[2] += A

    def commit_terminate(frontier):
        """Run stops after current Y residue (w.p. 1-y): add right-junction
        last_before->R (before) and last_after->R (after), commit (after-before)."""
        for (aL, lb, aisD), (w, B, A) in frontier.items():
            for R in (M, I, D, E):
                pr = T[lb, R]
                if pr <= 0:
                    continue
                wR = (1 - y) * pr
                # before gets lb->R ; after gets (D or aL)->R
                Bc = B * wR + _e(lb, R) * (w * wR)
                if aisD:
                    Ac = A * wR + _e(D, R) * (w * wR)
                else:
                    Ac = A * wR + _e(aL, R) * (w * wR)
                out_add(Ac - Bc)

    # closures referencing `out`
    def out_add(mat):
        nonlocal out
        out = out + mat

    # --- start: process Y_1 (type p, left context L) ---
    frontier = {}
    for L in (S, M, I, D):
        for p in (M, I):
            w = visit[L] * T[L, p] * mu0
            if w <= 0:
                continue
            B = _e(L, p) * w                          # before: L->Y_1
            A = (_e(L, D) * w) if p == M else np.zeros((5, 5))  # after: L->D if M
            aisD = (p == M)
            add_state(frontier, (L, p, aisD), w, B, A)

    commit_terminate(frontier)

    for k in range(2, KMAX + 1):
        newf = {}
        for (aL, lb, aisD), (w, B, A) in frontier.items():
            cur = lb
            # d = 0: cur -> p2 directly
            for p2 in (M, I):
                pr = T[cur, p2]
                if pr <= 0:
                    continue
                f = y * pr
                w2 = w * f
                B2 = B * f + _e(cur, p2) * w2
                if p2 == M:
                    A2 = A * f + (_e(D if aisD else aL, D)) * w2
                    naisD = True
                else:
                    A2 = A * f
                    naisD = aisD
                add_state(newf, (aL, p2, naisD), w2, B2, A2)
            # d >= 1: cur -> D^d -> p2.  Walk the D-run with scaling.
            # state after entering the D-run: weight factor y*T[cur,D], then dll each.
            f0 = y * T[cur, D]
            if f0 > 0:
                wD = w * f0
                BD = B * f0 + _e(cur, D) * wD
                AD = (A * f0 + _e(D if aisD else aL, D) * wD)  # this D survives -> after D
                aisD_D = True
                lb_D = D
                fcum = f0
                for d in range(1, DMAX + 1):
                    # exit current D-run (which has d D's) to p2
                    for p2 in (M, I):
                        pr = T[D, p2]
                        if pr <= 0:
                            continue
                        f = pr
                        w2 = wD * f
                        B2 = BD * f + _e(D, p2) * w2
                        if p2 == M:
                            A2 = AD * f + _e(D, D) * w2
                            naisD = True
                        else:
                            A2 = AD * f
                            naisD = aisD_D
                        add_state(newf, (aL, p2, naisD), w2, B2, A2)
                    # extend the D-run by one more interspersed D
                    wD2 = wD * dll
                    BD = BD * dll + _e(D, D) * wD2
                    AD = AD * dll + _e(D, D) * wD2
                    wD = wD2
                    if wD < 1e-20:
                        break
        if not newf:
            break
        commit_terminate(newf)
        frontier = newf
        if max(s[0] for s in newf.values()) < 1e-20:
            break

    return out


def _e(i, j):
    m = np.zeros((5, 5)); m[i, j] = 1.0; return m


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    LBL = ['S', 'M', 'I', 'D', 'E']
    kap, alpha, r = 0.5, 0.6, 0.3
    mu0, y = 1.0, 0.4
    dN = del_increment_exact(kap, alpha, r, mu0, y)
    n = 400000
    acc = np.zeros((5, 5))
    for _ in range(n):
        nd, _ = G.ggi_Ndot_for_sample(G.sample_alignment(kap, alpha, r), 0.0, mu0, 0.0, y)
        acc += nd
    tgt = acc / n
    print(f"max|exact - routeA(n={n})| = {np.max(np.abs(dN - tgt)):.4f}")
    print("exact / routeA rows:")
    for s in range(5):
        e = "  ".join(f"{dN[s, c]:+.4f}" for c in range(5))
        a = "  ".join(f"{tgt[s, c]:+.4f}" for c in range(5))
        print(f"  {LBL[s]}: [{e}]  |  [{a}]")
    print(f"\naggregates: d#D={sum(dN[s,D] for s in range(5)):+.4f}  "
          f"d#I={sum(dN[s,I] for s in range(5)):+.4f}  d#M={sum(dN[s,M] for s in range(5)):+.4f}")
