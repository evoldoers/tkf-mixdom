#!/usr/bin/env python3
"""Compare tkf-mixdom's LG exchangeabilities with the published LG table.

The published table is src/progalign/lg_paml.py (copied from PAML's
dat/lg.dat, which agrees entry for entry with IQ-TREE's ``model LG``).  Reports
how many of the 190 entries differ, the size of the differences, and tests
whether the local table could be the published one in another order (any
relabeling of amino acids, or a row/column-major mix-up, only permutes the
entries, so the sets of values would be identical).
"""
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parents[1]))   # tkf-mixdom/python
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tkfmixdom.jax.core import protein as P
from progalign import lg_paml
import lg08

PAML = "ARNDCQEGHILKMFPSTWYV"


def full(v):
    S = np.zeros((20, 20))
    k = 0
    for i in range(1, 20):
        S[i, :i] = v[k:k + i]
        k += i
    return S + S.T


def main():
    pub = np.asarray(lg_paml.S_LOWER, float)
    loc = np.asarray(P._LG_S_LOWER, float)
    print("lg08 copy (tkf-dp src/tkfdp/lg08.py) identical to tkf-mixdom's table:",
          bool(np.array_equal(np.asarray(lg08._LG_S_LOWER, float), loc)))
    print("frequencies: max |local - published| =",
          float(np.abs(np.asarray(P._LG_PI, float) - lg_paml.PI).max()))
    diff = np.abs(loc - pub) > 1e-5
    pct = 100 * (loc - pub) / pub
    a = np.abs(pct[diff])
    print(f"entries differing: {diff.sum()} of 190; first differing entry index {np.argmax(diff)} "
          f"(row {PAML[next(i for i in range(1, 20) if i * (i + 1) // 2 > np.argmax(diff))]})")
    print(f"signed difference: {pct[diff].min():.1f}% to {pct[diff].max():+.1f}%; "
          f"|%| quartiles {np.percentile(a, 25):.0f}/{np.percentile(a, 50):.0f}/{np.percentile(a, 75):.0f}, "
          f"90th {np.percentile(a, 90):.0f}")
    print(f"counts: <10%: {(a < 10).sum()}, 10-50%: {((a >= 10) & (a < 50)).sum()}, "
          f"50-100%: {((a >= 50) & (a < 100)).sum()}, >=100%: {(a >= 100).sum()}; "
          f"higher/lower: {(pct[diff] > 0).sum()}/{(pct[diff] < 0).sum()}")
    pi = lg_paml.PI / lg_paml.PI.sum()
    I = [i for i in range(1, 20) for j in range(i)]
    J = [j for i in range(1, 20) for j in range(i)]
    flux = pub * pi[I] * pi[J]
    order = np.argsort(np.abs(pct))
    w = np.cumsum((flux / flux.sum())[order])
    print(f"flux-weighted median |%|: {np.abs(pct)[order][np.searchsorted(w, 0.5)]:.0f}%; "
          f"total flux local/published {(loc * pi[I] * pi[J]).sum() / flux.sum():.3f}")
    pairs = [(PAML[i], PAML[j]) for i in range(1, 20) for j in range(i)]
    o = np.argsort(pct)
    print("largest decreases:", ", ".join(f"{pairs[k][0]}-{pairs[k][1]} {pub[k]:.4f}->{loc[k]:.4f}" for k in o[:3]))
    print("largest increases:", ", ".join(f"{pairs[k][0]}-{pairs[k][1]} {pub[k]:.4f}->{loc[k]:.4f}" for k in o[::-1][:3]))
    # permutation tests
    rp = set(np.round(pub, 6))
    print("\nIs it the published table in another order?")
    print("  same multiset of values:", bool(np.allclose(np.sort(loc), np.sort(pub))))
    print("  local values occurring anywhere in the published table:",
          sum(v in rp for v in np.round(loc, 6)), "of 190")
    c = collections.Counter(np.round(loc, 6))
    print(f"  distinct values: local {len(c)}, published {len(rp)}; "
          f"most repeated local value occurs {max(c.values())} times")
    Sl, Sp = full(loc), full(pub)
    rows_p = {tuple(np.round(np.sort(Sp[i]), 6)) for i in range(20)}
    print("  local rows whose value set matches some published row:",
          sum(tuple(np.round(np.sort(Sl[i]), 6)) in rows_p for i in range(20)), "of 20")
    k = 0
    U = np.zeros((20, 20))
    for j in range(1, 20):
        for i in range(j):
            U[i, j] = pub[k]
            k += 1
    print("  equals published list read column-major:", bool(np.allclose(Sl, U + U.T)))


if __name__ == "__main__":
    main()
