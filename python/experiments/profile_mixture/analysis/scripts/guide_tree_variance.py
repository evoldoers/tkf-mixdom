"""Variance-weighted guide trees from trigram distances, tested on Pfam.

Per family (at most --max-seqs sequences), for every pair of sequences:
  * t_ML: maximum-likelihood t under TKF92 + LG-C20 given the pair's alignment
    in the Pfam seed MSA (see bigram_distance_pfam.py);
  * t3: the composition-corrected trigram distance (profile_classes.tex,
    Section "Guide-tree distances from bigram and trigram counts");
  * Var(t3) by the delta method, Var(D3) / (dE[D3]/dt)^2, with Var(D3) from a
    block bootstrap over trigram positions of each sequence; estimates at the
    search bound are censored and given effectively infinite variance.
Trees:
  reference  BioNJ on t_ML
  nj         NJ on t3
  bionj      BioNJ on t3 with its default variances (V = D)
  bionj_var  BioNJ on t3 with the delta-method variances
  kimura_msa control: BioNJ on Kimura distances from the seed MSA itself
Each is compared with the reference by normalized Robinson-Foulds distance.

Usage: python analysis/scripts/guide_tree_variance.py \
         --data-dir data/pfam_seed_sample --out results/guide_tree_variance
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bigram_distance_check import A, Subst, counts, expected, invert  # noqa: E402
from bigram_distance_pfam import PARAMS, Composition, mle_t, pair_columns  # noqa: E402
from profile_classes_pfam import kimura_distances, read_stockholm  # noqa: E402

TMAX = 20.0
BIG_VAR = 1e8
MIN_T = 1e-3


def nj_tree(D, V=None, bionj=True):
    """NJ, or BioNJ with initial variance matrix V (default V = D).
    Returns adjacency {node: {nbr: length}}; leaves are 0..n-1."""
    n = D.shape[0]
    D = D.astype(float).copy()
    V = (D if V is None else V).astype(float).copy()
    size = 2 * n
    D = np.pad(D, ((0, size - n), (0, size - n)))
    V = np.pad(V, ((0, size - n), (0, size - n)))
    active = list(range(n))
    adj = {i: {} for i in range(n)}
    nxt = n
    while len(active) > 3:
        m = len(active)
        Dm = D[np.ix_(active, active)]
        r = Dm.sum(1)
        Q = (m - 2) * Dm - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        a, b = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = active[a], active[b]
        li = 0.5 * D[i, j] + (r[a] - r[b]) / (2 * (m - 2))
        lj = D[i, j] - li
        others = [k for k in active if k not in (i, j)]
        if bionj and V[i, j] > 0:
            lam = 0.5 + sum(V[j, k] - V[i, k] for k in others) / (2 * (m - 2) * V[i, j])
            lam = min(max(lam, 0.0), 1.0)
        else:
            lam = 0.5
        u = nxt; nxt += 1
        for k in others:
            D[u, k] = D[k, u] = lam * (D[i, k] - li) + (1 - lam) * (D[j, k] - lj)
            V[u, k] = V[k, u] = lam * V[i, k] + (1 - lam) * V[j, k] - lam * (1 - lam) * V[i, j]
        adj[u] = {}
        for c, l in ((i, li), (j, lj)):
            l = max(l, MIN_T)
            adj[u][c] = l; adj[c][u] = l
        active = others + [u]
    x, y, z = active
    c = nxt
    adj[c] = {}
    for k, l in ((x, 0.5 * (D[x, y] + D[x, z] - D[y, z])),
                 (y, 0.5 * (D[x, y] + D[y, z] - D[x, z])),
                 (z, 0.5 * (D[x, z] + D[y, z] - D[x, y]))):
        l = max(l, MIN_T)
        adj[c][k] = l; adj[k][c] = l
    return adj


def splits(adj, n):
    """Nontrivial bipartitions of the leaves 0..n-1, each as a frozenset not containing leaf 0."""
    out = set()
    for u in adj:
        for v in adj[u]:
            side, stack, seen = [], [v], {u, v}
            while stack:
                x = stack.pop()
                if x < n:
                    side.append(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y); stack.append(y)
            s = frozenset(side)
            if 1 < len(s) < n - 1:
                out.add(s if 0 not in s else frozenset(range(n)) - s)
    return out


def rf(adj1, adj2, n):
    s1, s2 = splits(adj1, n), splits(adj2, n)
    return len(s1 ^ s2) / (2 * (n - 3))


def trigram_codes(z):
    return z[:-2] * A * A + z[1:-1] * A + z[2:]


def block_bootstrap_counts(codes, B, block, rng):
    """B bootstrap trigram count vectors from a moving-block bootstrap over positions."""
    n = len(codes)
    out = np.zeros((B, A ** 3), np.float32)
    nb = int(np.ceil(n / block))
    for b in range(B):
        starts = rng.integers(0, max(n - block, 0) + 1, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        idx = np.minimum(idx, n - 1)
        out[b] = np.bincount(codes[idx], minlength=A ** 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/pfam_seed_sample")
    ap.add_argument("--out", default="results/guide_tree_variance")
    ap.add_argument("--max-seqs", type=int, default=40)
    ap.add_argument("--min-seqs", type=int, default=8)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--boot", type=int, default=50)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    prm = json.loads(PARAMS.read_text())
    lam, mu, r = prm["lam"], prm["mu"], prm["exts"][0]
    frag = (np.array([r]), np.array([1.0]))
    sub = Subst(c20=True)
    recs = []
    for f in sorted(Path(args.data_dir).glob("*.sto.gz")):
        fam = f.name.split(".")[0]
        _, X = read_stockholm(f)
        ok = np.where((X >= 0).sum(1) >= args.min_len)[0]
        if len(ok) < args.min_seqs:
            continue
        if len(ok) > args.max_seqs:
            ok = rng.choice(ok, args.max_seqs, replace=False)
        X = X[ok]
        n = X.shape[0]
        seqs = [row[row >= 0] for row in X]
        codes = [trigram_codes(s) for s in seqs]
        c3 = [np.bincount(c, minlength=A ** 3).astype(float) for c in codes]
        c1 = [np.bincount(s, minlength=A).astype(float) for s in seqs]
        boot = [block_bootstrap_counts(c, args.boot, args.block, rng) for c in codes]
        Tml = np.zeros((n, n)); T3 = np.zeros((n, n)); V3 = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                types, x, y = pair_columns(X[i], X[j])
                Tml[i, j] = Tml[j, i] = mle_t(types, x, y, lam, mu, r, sub)
                LX, LY = len(seqs[i]), len(seqs[j])
                Lm = 0.5 * (LX + LY)
                scale = (Lm - 2) ** 2 / ((LX - 2) * (LY - 2))
                comp = Composition(sub, float((c1[i] / LX) @ (c1[j] / LY)))
                t3 = invert((c3[i] @ c3[j]) * scale, Lm, Lm, lam, mu, frag, comp, 3, tmax=TMAX)
                varD = float(np.var((boot[i] * boot[j]).sum(1).astype(float) * scale, ddof=1))
                h = 1e-3 * max(t3, 0.01)
                dE = (expected(t3 + h, Lm, Lm, lam, mu, frag, comp, 3)
                      - expected(max(t3 - h, 1e-6), Lm, Lm, lam, mu, frag, comp, 3)) / (2 * h)
                v = BIG_VAR if (t3 >= TMAX * 0.995 or dE == 0) else varD / dE ** 2
                T3[i, j] = T3[j, i] = t3
                V3[i, j] = V3[j, i] = min(v, BIG_VAR)
        ref = nj_tree(Tml)
        trees = dict(nj=nj_tree(T3, bionj=False), bionj=nj_tree(T3),
                     bionj_var=nj_tree(T3, V=V3),
                     kimura_msa=nj_tree(kimura_distances(X)))   # control: alignment-based
        rec = dict(family=fam, n=n, capped=float(np.mean(T3[np.triu_indices(n, 1)] >= TMAX * 0.995)),
                   median_t_ml=float(np.median(Tml[np.triu_indices(n, 1)])),
                   **{k: rf(ref, t, n) for k, t in trees.items()})
        recs.append(rec)
        print(f"{fam}: n={n} median t_ML={rec['median_t_ml']:.2f} capped={rec['capped']:.2f} "
              f"RF nj={rec['nj']:.3f} bionj={rec['bionj']:.3f} bionj_var={rec['bionj_var']:.3f} "
              f"kimura_msa={rec['kimura_msa']:.3f}",
              flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(recs, indent=0))
    report(recs, out)


def report(recs, out):
    get = lambda k: np.array([x[k] for x in recs])
    lines = [f"families: {len(recs)}; normalized Robinson-Foulds distance to the reference tree "
             "(BioNJ on TKF92+LG-C20 ML distances); lower is better",
             f"{'tree':>10} {'mean RF':>8} {'median RF':>10}"]
    for k in ("nj", "bionj", "bionj_var", "kimura_msa"):
        lines.append(f"{k:>10} {get(k).mean():>8.3f} {np.median(get(k)):>10.3f}")
    for a, b in (("bionj_var", "bionj"), ("bionj_var", "nj"), ("bionj", "nj")):
        d = get(a) - get(b)
        w, l = int((d < 0).sum()), int((d > 0).sum())
        p = binomtest(w, w + l).pvalue if w + l else float("nan")
        lines.append(f"{a} vs {b}: better in {w}, worse in {l}, tied in {len(d) - w - l} "
                     f"(sign test p = {p:.3g}); mean RF difference {d.mean():+.3f}")
    txt = "\n".join(lines)
    print(txt)
    (out / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
