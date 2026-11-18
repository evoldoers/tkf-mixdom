"""Test the O(K+A) profile-class bound (analysis/profile_elbo/profile_classes.tex)
against exact LG-C20 likelihoods on Pfam seed alignments.

For each family: subsample sequences, build a BioNJ tree from Kimura protein
distances, then for random columns
  * restrict the tree to the column's non-gap leaves,
  * split it at a random edge into two clades A, B (both with >= 2 leaves),
  * compute the exact LG-C20 log-likelihood of the column by pruning, and
  * the bound log sum_k w_k exp(Phi_k(s)) from class-free summaries built by
    single-jump merges (the note's construction), plus two variants:
      - "nocross": final-merge log-pi cross term dropped (pure dot-product form)
      - "sharedpost": q(h) = exact posterior of the single best class, shared
        across classes (isolates the cost of sharing q(h) from the cost of the
        constructed q(h)).
Also a ranking test: the same split with B's residues taken from a different
random column (a misaligned cell), comparing exact and bound differences.

Usage:
  python analysis/scripts/profile_classes_pfam.py \
      --data-dir data/pfam_seed_sample --out results/profile_classes_pfam
"""
import argparse
import gzip
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
AA = "ACDEFGHIKLMNPQRSTVWY"
AIDX = {a: i for i, a in enumerate(AA)}
A = 20
MIN_T = 1e-3


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model():
    # lg08.py imports jax.numpy at module level; get_lg08 itself is numpy-only.
    # numpy stands in for jax.numpy (module-level constants only).
    # Remove the stubs afterwards: scipy probes sys.modules["jax"].
    had = {m: sys.modules.get(m) for m in ("jax", "jax.numpy")}
    sys.modules["jax"] = types.ModuleType("jax")
    sys.modules["jax.numpy"] = np
    try:
        lg = _load("lg08", ROOT / "src/lg08.py")
    finally:
        for m, mod in had.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod
    S, _ = lg.get_lg08()
    S = np.array(S, dtype=float)
    np.fill_diagonal(S, 0.0)
    scp = _load("site_class_profiles",
                ROOT.parents[1] / "tkfmixdom/jax/core/site_class_profiles.py")
    P, w, _ = scp.le_gascuel_c20()
    P = np.asarray(P, float); P /= P.sum(1, keepdims=True)
    w = np.asarray(w, float); w /= w.sum()
    # IQ-TREE convention: each class matrix normalised to mean rate 1.
    R = 1.0 / np.einsum("kx,xy,ky->k", P, S, P)
    return S, P, w, R


class Classes:
    """Per-class eigen-decompositions of Q^k = r_k S diag(pi^k)."""

    def __init__(self, S, P, w, R):
        self.S, self.P, self.w, self.R = S, P, w, R
        self.K = len(w)
        self.lP, self.lR, self.lw = np.log(P), np.log(R), np.log(w)
        self.G = P @ S                          # G[k,x] = (S pi^k)_x
        self.logS = np.log(np.where(S > 0, S, 1.0))
        sq = np.sqrt(P)
        B = R[:, None, None] * sq[:, :, None] * S[None] * sq[:, None, :]
        B[:, np.arange(A), np.arange(A)] = -R[:, None] * self.G
        lam, U = np.linalg.eigh(B)
        self.lam = lam                                        # (K,A)
        self.V = U / sq[:, :, None]                           # M = V e^{lam t} Vi
        self.Vi = np.transpose(U, (0, 2, 1)) * sq[:, None, :]
        self.Q = R[:, None, None] * S[None] * P[:, None, :]
        self.Q[:, np.arange(A), np.arange(A)] = -R[:, None] * self.G

    def M(self, t):
        return np.einsum("kxi,ki,kiy->kxy", self.V, np.exp(self.lam * t), self.Vi)


# ---------------------------------------------------------------- data + tree

def read_stockholm(path):
    seqs = {}
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) == 2:
                seqs[parts[0]] = seqs.get(parts[0], "") + parts[1]
    names = list(seqs)
    L = len(seqs[names[0]])
    X = np.full((len(names), L), -1, dtype=int)
    for i, n in enumerate(names):
        for j, c in enumerate(seqs[n].upper()):
            X[i, j] = AIDX.get(c, -1)
    return names, X


def kimura_distances(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    ok = X >= 0
    for i in range(n):
        both = ok[i] & ok
        same = (X[i] == X) & both
        nb = both.sum(1)
        p = np.where(nb > 0, 1 - same.sum(1) / np.maximum(nb, 1), 1.0)
        arg = 1 - p - 0.2 * p * p
        d = np.where(arg > 0.05, -np.log(np.maximum(arg, 0.05)), 3.0)
        d[nb < 10] = 3.0
        D[i] = d
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    return D


def bionj(D):
    """BioNJ (Gascuel 1997). Returns adjacency {node: {nbr: length}}; leaves 0..n-1."""
    n = D.shape[0]
    D = D.copy(); V = D.copy()
    active = list(range(n))
    adj = {i: {} for i in range(n)}
    nxt = n
    while len(active) > 3:
        m = len(active)
        Dm = D[np.ix_(active, active)]
        r = Dm.sum(1)
        Qm = (m - 2) * Dm - r[:, None] - r[None, :]
        np.fill_diagonal(Qm, np.inf)
        a, b = np.unravel_index(np.argmin(Qm), Qm.shape)
        i, j = active[a], active[b]
        li = 0.5 * D[i, j] + (r[a] - r[b]) / (2 * (m - 2))
        lj = D[i, j] - li
        vij = V[i, j]
        lam = 0.5 if vij <= 0 else 0.5 + sum(V[j, k] - V[i, k] for k in active if k not in (i, j)) / (2 * (m - 2) * vij)
        lam = min(max(lam, 0.0), 1.0)
        u = nxt; nxt += 1
        size = u + 1
        if size > D.shape[0]:
            D = np.pad(D, ((0, n), (0, n))); V = np.pad(V, ((0, n), (0, n)))
        for k in active:
            if k in (i, j):
                continue
            D[u, k] = D[k, u] = lam * (D[i, k] - li) + (1 - lam) * (D[j, k] - lj)
            V[u, k] = V[k, u] = lam * V[i, k] + (1 - lam) * V[j, k] - lam * (1 - lam) * vij
        adj[u] = {}
        for c, l in ((i, li), (j, lj)):
            l = max(l, MIN_T)
            adj[u][c] = l; adj[c][u] = l
        active = [k for k in active if k not in (i, j)] + [u]
    x, y, z = active
    c = nxt
    adj[c] = {}
    for k, l in ((x, 0.5 * (D[x, y] + D[x, z] - D[y, z])),
                 (y, 0.5 * (D[x, y] + D[y, z] - D[x, z])),
                 (z, 0.5 * (D[x, z] + D[y, z] - D[x, y]))):
        l = max(l, MIN_T)
        adj[c][k] = l; adj[k][c] = l
    return adj


def induced_tree(adj, present):
    """Restrict an unrooted tree to the leaves in `present`, suppressing degree-2 nodes."""
    g = {u: dict(v) for u, v in adj.items()}
    changed = True
    while changed:
        changed = False
        for u in list(g):
            if u in g and len(g[u]) <= 1 and u not in present:
                for v in g[u]:
                    del g[v][u]
                del g[u]
                changed = True
    for u in list(g):
        if u in g and len(g[u]) == 2 and u not in present:
            (a, la), (b, lb) = g[u].items()
            del g[a][u]; del g[b][u]
            g[a][b] = la + lb; g[b][a] = la + lb
            del g[u]
    return g


def side_leaves(g, u, parent, present):
    out, stack = [], [(u, parent)]
    while stack:
        x, p = stack.pop()
        if x in present:
            out.append(x)
        stack.extend((y, x) for y in g[x] if y != p)
    return out


# ----------------------------------------------------------------- exact

def partials(g, u, parent, col, C):
    """Per-class log-scaled partial likelihoods at u (subtree away from parent)."""
    L = np.ones((C.K, A)); sc = 0.0
    if u in col:
        L[:] = 0.0; L[:, col[u]] = 1.0
    for v, t in g[u].items():
        if v == parent:
            continue
        Lv, sv = partials(g, v, u, col, C)
        L = L * np.einsum("kxy,ky->kx", C.M(t), Lv)
        m = L.max()
        L /= m; sc += sv + np.log(m)
    return L, sc


def exact_loglik(g, u, v, t, col, C):
    Lu, su = partials(g, u, v, col, C)
    Lv, sv = partials(g, v, u, col, C)
    per = np.log(np.einsum("kx,kxy,ky->k", C.P * Lu, C.M(t), Lv)) + su + sv
    return logsumexp(C.lw + per), per


# ----------------------------------------------------------------- bound

def leaf_summary(x):
    e = np.zeros(A); e[x] = 1.0
    return dict(Vp=e, U=0.0, T=np.zeros(A), C=0.0, q=e.copy())


def merge(SA, SB, tA, tB, C):
    t = max(tA + tB, MIN_T)
    qA, qB = SA["q"], SB["q"]
    off = np.outer(qA, qB); np.fill_diagonal(off, 0)
    return dict(
        Vp=SA["Vp"] + SB["Vp"] - qA * qB,
        U=SA["U"] + SB["U"] + 1.0 - qA @ qB,
        T=SA["T"] + SB["T"] + 0.5 * t * (qA + qB),
        C=SA["C"] + SB["C"] + (off * (C.logS + np.log(t))).sum(),
        q=qA * qB + (tB / t) * qA * (1 - qB) + (tA / t) * qB * (1 - qA),
    )


def summary(g, u, parent, col, C):
    """Summary of the subtree at u (away from parent), with top node u."""
    kids = [(v, t) for v, t in g[u].items() if v != parent]
    if u in col:
        acc = leaf_summary(col[u])
        for v, t in kids:
            acc = merge(acc, summary(g, v, u, col, C), 0.0, t, C)
        return acc
    (v1, t1), (v2, t2) = kids[0], kids[1]
    acc = merge(summary(g, v1, u, col, C), summary(g, v2, u, col, C), t1, t2, C)
    for v, t in kids[2:]:
        acc = merge(acc, summary(g, v, u, col, C), 0.0, t, C)
    return acc


def Phi(s, C):
    return s["C"] + C.lP @ s["Vp"] + s["U"] * C.lR - C.R * (C.G @ s["T"])


def bound(g, u, v, t, col, C):
    SA = summary(g, u, v, col, C)
    SB = summary(g, v, u, col, C)
    s = merge(SA, SB, t, 0.0, C)
    F = Phi(s, C)
    crossP = -(C.lP @ (SA["q"] * SB["q"]))
    return logsumexp(C.lw + F), logsumexp(C.lw + F - crossP)


# ------------------------------------------- shared exact-posterior q(h)

def divdiff(la, lb, t):
    d = la - lb
    same = np.abs(d) < 1e-9
    safe = np.where(same, 1.0, d)
    return np.where(same, t * np.exp(la * t), (np.exp(la * t) - np.exp(lb * t)) / safe)


def posterior_summary(g, u, excl, col, C, kstar):
    """Summary (V', U, T, C, q_top) of q(h) = exact posterior of class kstar on the
    subtree at u away from `excl` (None = whole tree), rooted at u. C is set so that
    Phi_kstar(summary) = log p_kstar(D_subtree) exactly."""
    V, Vi, lam = C.V[kstar], C.Vi[kstar], C.lam[kstar]
    Qk = C.Q[kstar]
    Mk = lambda s: (V * np.exp(lam * s)) @ Vi

    # inside partials (log-scaled) for rooted tree at u, with v as a child via t
    g2 = {x: dict(y) for x, y in g.items()}
    ins = {}

    def up(x, p):
        L = np.ones(A); sc = 0.0
        if x in col:
            L = np.zeros(A); L[col[x]] = 1.0
        for y, s in g2[x].items():
            if y == p:
                continue
            Ly, sy = up(y, x)
            m = Mk(s) @ Ly
            ins[(x, y)] = m
            L = L * m; sc += sy
            mx = L.max(); L = L / mx; sc += np.log(mx)
        ins[x] = (L, sc)
        return L, sc

    Lroot, sroot = up(u, excl)
    pk = C.P[kstar]
    logp = np.log(pk @ Lroot) + sroot
    Vp = np.zeros(A); T = np.zeros(A); U = 0.0
    root_post = pk * Lroot; root_post /= root_post.sum()
    Vp += root_post

    def down(x, p, out_x):
        # out_x: outside message at x (excluding subtree below x), unnormalised
        Lx = ins[x][0]
        for y, s in g2[x].items():
            if y == p:
                continue
            m_y = ins[(x, y)]
            with np.errstate(divide="ignore", invalid="ignore"):
                others = np.where(m_y > 0, out_x * Lx / m_y, 0.0)
            Ly = ins[y][0]
            Ms = Mk(s)
            joint = others[:, None] * Ms * Ly[None, :]
            joint /= joint.sum()
            Wab = np.where(Ms > 0, joint / Ms, 0.0)
            Gm = V.T @ Wab @ Vi.T
            Dm = divdiff(lam[:, None], lam[None, :], s)
            J = Vi.T @ (Gm * Dm) @ V.T          # J[x,y] = sum_ab W_ab int M_ax(u) M_yb(s-u) du
            T[:] += np.diag(J)
            Uxy = Qk * J
            np.fill_diagonal(Uxy, 0.0)
            Vp[:] += Uxy.sum(0)
            nonlocal_U[0] += Uxy.sum()
            out_y = others @ Ms
            out_y = out_y / out_y.max()
            down(y, x, out_y)

    nonlocal_U = [0.0]
    down(u, excl, pk.copy())
    U = nonlocal_U[0]
    Cst = logp - (C.lP[kstar] @ Vp + U * C.lR[kstar] - C.R[kstar] * (C.G[kstar] @ T))
    return dict(Vp=Vp, U=U, T=T, C=Cst, q=root_post)


def shared_posterior_bound(g, u, col, C, kstar):
    """Bound with q(h) = exact posterior of class kstar over the whole tree, shared by all classes."""
    return logsumexp(C.lw + Phi(posterior_summary(g, u, None, col, C, kstar), C))


def hybrid_bound(g, u, v, t, col, C):
    """Exact reference-class posteriors inside each clade (per-column work), cheap
    single-jump merge only at the final (per-DP-cell) step."""
    out = []
    for x, y in ((u, v), (v, u)):
        L, sc = partials(g, x, y, col, C)
        kx = int(np.argmax(C.lw + np.log((C.P * L).sum(1)) + sc))
        out.append(posterior_summary(g, x, y, col, C, kx))
    s = merge(out[0], out[1], t, 0.0, C)
    return logsumexp(C.lw + Phi(s, C))


def exact_bridge_stats(qA, qB, t, C, k):
    """Expected (dests, U, T) and E[log g] + H of the exact class-k bridge a->b of length t,
    with a ~ qA, b ~ qB independent (Holmes-Rubin / Hobolth-Jensen statistics)."""
    V, Vi, lam = C.V[k], C.Vi[k], C.lam[k]
    M = (V * np.exp(lam * t)) @ Vi
    W = np.outer(qA, qB) / M
    J = Vi.T @ ((V.T @ W @ Vi.T) * divdiff(lam[:, None], lam[None, :], t)) @ V.T
    Uxy = C.Q[k] * J
    np.fill_diagonal(Uxy, 0.0)
    dests, T, U = Uxy.sum(0), np.diag(J).copy(), Uxy.sum()
    ElogM = qA @ np.log(M) @ qB
    CH = ElogM - (C.lP[k] @ dests + U * C.lR[k] - C.R[k] * (C.G[k] @ T))
    return dests, U, T, CH


def hybrid_exact_bridge(g, u, v, t, col, C):
    """As hybrid_bound, but the final merge uses the exact bridge of clade A's class."""
    out, ks = [], []
    for x, y in ((u, v), (v, u)):
        L, sc = partials(g, x, y, col, C)
        kx = int(np.argmax(C.lw + np.log((C.P * L).sum(1)) + sc))
        out.append(posterior_summary(g, x, y, col, C, kx)); ks.append(kx)
    SA, SB = out
    dests, U, T, CH = exact_bridge_stats(SA["q"], SB["q"], max(t, MIN_T), C, ks[0])
    s = dict(Vp=SA["Vp"] + SB["Vp"] - SB["q"] + dests, U=SA["U"] + SB["U"] + U,
             T=SA["T"] + SB["T"] + T, C=SA["C"] + SB["C"] + CH)
    return logsumexp(C.lw + Phi(s, C))


def side_scores(g, u, v, col, C):
    """Exact and summary-bound log-likelihoods of the clade at u (away from v) alone."""
    L, sc = partials(g, u, v, col, C)
    ex = logsumexp(C.lw + np.log((C.P * L).sum(1)) + sc)
    bd = logsumexp(C.lw + Phi(summary(g, u, v, col, C), C))
    return ex, bd


# ----------------------------------------------------------------- driver

def run(args):
    rng = np.random.default_rng(args.seed)
    S, P, w, R = load_model()
    C = Classes(S, P, w, R)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recs = []
    files = sorted(Path(args.data_dir).glob("*.sto.gz"))
    for f in files:
        fam = f.name.split(".")[0]
        names, X = read_stockholm(f)
        if X.shape[0] < args.min_seqs:
            continue
        if X.shape[0] > args.max_seqs:
            X = X[rng.choice(X.shape[0], args.max_seqs, replace=False)]
        adj = bionj(kimura_distances(X))
        cols = [j for j in range(X.shape[1]) if (X[:, j] >= 0).sum() >= 4]
        rng.shuffle(cols)
        done = 0
        for j in cols:
            if done >= args.cols_per_family:
                break
            present = {i for i in range(X.shape[0]) if X[i, j] >= 0}
            g = induced_tree(adj, present)
            edges = [(a, b, tt) for a in g for b, tt in g[a].items() if a < b]
            rng.shuffle(edges)
            split = None
            for a, b, tt in edges:
                na = len(side_leaves(g, a, b, present)); nb = len(present) - na
                if na >= 2 and nb >= 2:
                    split = (a, b, tt, na, nb); break
            if split is None:
                continue
            a, b, tt, na, nb = split
            col = {i: X[i, j] for i in present}
            ex, per = exact_loglik(g, a, b, tt, col, C)
            bd, nocross = bound(g, a, b, tt, col, C)
            kstar = int(np.argmax(C.lw + per))
            sp = shared_posterior_bound(g, a, col, C, kstar)
            hy = hybrid_bound(g, a, b, tt, col, C)
            hx = hybrid_exact_bridge(g, a, b, tt, col, C)
            # misaligned cell: B side takes residues from another column
            j2 = int(rng.choice(cols))
            Bside = set(side_leaves(g, b, a, present))
            col2 = {i: X[i, j] for i in present if i not in Bside}
            col2.update({i: X[i, j2] for i in Bside if X[i, j2] >= 0})
            rec = dict(family=fam, col=j, n=len(present), nA=na, nB=nb, t=tt,
                       exact=ex, bound=bd, nocross=nocross, sharedpost=sp, hybrid=hy, hybrid_xb=hx)
            pres2 = set(col2)
            sa = [i for i in side_leaves(g, a, b, present) if i in pres2]
            sb = [i for i in Bside if i in pres2]
            if len(sa) >= 1 and len(sb) >= 1 and len(pres2) >= 3:
                g2 = induced_tree(adj, pres2)
                # locate split edge in g2: endpoint nearest a/b sides
                e2 = None
                for x in g2:
                    for y, t2 in g2[x].items():
                        L1 = set(side_leaves(g2, x, y, pres2))
                        if L1 == set(sa):
                            e2 = (x, y, t2)
                            break
                    if e2:
                        break
                if e2:
                    x2, y2, t2 = e2
                    exA, bdA = side_scores(g, a, b, col, C)
                    exB, bdB = side_scores(g, b, a, col, C)
                    exA2, bdA2 = side_scores(g2, x2, y2, col2, C)
                    exB2, bdB2 = side_scores(g2, y2, x2, col2, C)
                    em, _ = exact_loglik(g2, x2, y2, t2, col2, C)
                    bm, _ = bound(g2, x2, y2, t2, col2, C)
                    hm = hybrid_bound(g2, x2, y2, t2, col2, C)
                    # log-odds match scores: log p(AB) - log p(A) - log p(B)
                    rec["lo_exact"] = ex - exA - exB
                    rec["lo_exact_mis"] = em - exA2 - exB2
                    rec["lo_bound"] = bd - bdA - bdB
                    rec["lo_bound_mis"] = bm - bdA2 - bdB2
                    rec["lo_hybrid"] = hy - exA - exB
                    rec["lo_hybrid_mis"] = hm - exA2 - exB2
            recs.append(rec)
            done += 1
        print(f"{fam}: {done} columns, {X.shape[0]} seqs", flush=True)
    (out / "records.json").write_text(json.dumps(recs, default=float, indent=0))
    report(recs, out)


def report(recs, out):
    r = {k: np.array([x[k] for x in recs], float) for k in ("n", "exact", "bound", "nocross", "sharedpost", "hybrid", "hybrid_xb")}
    assert np.all(r["bound"] <= r["exact"] + 1e-6), "bound violated"
    assert np.all(r["sharedpost"] <= r["exact"] + 1e-6), "shared-posterior bound violated"
    assert np.all(r["hybrid"] <= r["exact"] + 1e-6), "hybrid bound violated"
    assert np.all(r["hybrid_xb"] <= r["exact"] + 1e-6), "exact-bridge hybrid bound violated"
    ghy = r["exact"] - r["hybrid"]; ghx = r["exact"] - r["hybrid_xb"]
    gap = r["exact"] - r["bound"]; gsp = r["exact"] - r["sharedpost"]; gnc = r["exact"] - r["nocross"]
    lines = [f"columns: {len(recs)} from {len({x['family'] for x in recs})} families",
             f"bound valid on all columns: {bool(np.all(gap >= -1e-6))}",
             "gap = exact - bound (nats), by number of non-gap leaves:",
             f"{'leaves':>10} {'cols':>5} {'bound med':>10} {'p90':>7} {'per-leaf':>9} "
             f"{'shared-post med':>16} {'no-cross med':>13} {'hybrid med':>11} {'p90':>6} {'hyb+xbridge':>12}"]
    for lo, hi in ((4, 8), (8, 16), (16, 32), (32, 65)):
        m = (r["n"] >= lo) & (r["n"] < hi)
        if m.sum():
            lines.append(f"{lo:>4}-{hi - 1:<5} {m.sum():>5} {np.median(gap[m]):>10.2f} "
                         f"{np.percentile(gap[m], 90):>7.2f} {np.median(gap[m] / r['n'][m]):>9.3f} "
                         f"{np.median(gsp[m]):>16.2f} {np.median(gnc[m]):>13.2f} {np.median(ghy[m]):>11.2f} {np.percentile(ghy[m], 90):>6.2f} {np.median(ghx[m]):>12.2f}")
    lines.append(f"{'all':>10} {len(gap):>5} {np.median(gap):>10.2f} {np.percentile(gap, 90):>7.2f} "
                 f"{np.median(gap / r['n']):>9.3f} {np.median(gsp):>16.2f} {np.median(gnc):>13.2f} {np.median(ghy):>11.2f} {np.percentile(ghy, 90):>6.2f} {np.median(ghx):>12.2f}")
    lines.append(f"corr(exact, bound) = {np.corrcoef(r['exact'], r['bound'])[0, 1]:.4f}")
    mis = [x for x in recs if "lo_exact" in x]
    if mis:
        from scipy.stats import spearmanr
        de = np.array([x["lo_exact"] - x["lo_exact_mis"] for x in mis])
        lines.append(f"ranking, aligned vs misaligned cell (log-odds match scores, {len(mis)} pairs):")
        lines.append(f"  exact prefers the aligned cell in {np.mean(de > 0):.3f}")
        for name in ("bound", "hybrid"):
            db = np.array([x[f"lo_{name}"] - x[f"lo_{name}_mis"] for x in mis])
            lines.append(f"  {name:>7}: prefers aligned {np.mean(db > 0):.3f}, sign agreement with exact "
                         f"{np.mean(np.sign(de) == np.sign(db)):.3f}, Spearman {spearmanr(de, db)[0]:.3f}")
    txt = "\n".join(lines)
    print(txt)
    (out / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/pfam_seed_sample")
    ap.add_argument("--out", default="results/profile_classes_pfam")
    ap.add_argument("--min-seqs", type=int, default=6)
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--cols-per-family", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())
