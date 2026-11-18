#!/usr/bin/env python3
"""How well do substitution models explain aligned residue pairs in Pfam seed alignments?

For up to 30 random sequence pairs per Pfam seed alignment (pairs with at least
30 aligned residue pairs), count aligned residue pairs N_ab and maximize
sum_ab N_ab log J(t)_ab over t on a log grid (0.01..1000), where J(t) is the
model's class-marginal joint (the match emission used by
experiments/fsa_c20_balibase.py).  Reports, per aligned residue pair:

    LL      the maximized log-likelihood
    assoc   LL minus the composition term sum N_ab log(pibar_a pibar_b),
            i.e. the pair association the model captures (what drives
            match-versus-gap decisions)

and the median fitted t, the fraction of pairs with t >= 10 (paper 1's clip),
and P(identical) at t = 1000.  No BAliBASE data is used.
"""
import argparse
import gzip
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

import fsa_c20_balibase as F

AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AA)}
MODELS = [  # (subst kind, gamma shape or None)
    ("LG", None), ("LG", 1.0), ("LG", 1.5),
    ("LGp", None), ("LGp", 1.0), ("LGp", 1.5),
    ("C20", None), ("C20", 0.35), ("C20", 0.5),
    ("C20p", None), ("C20p", 0.35),
    ("C20g", None),
    ("CML20", None), ("CML20r", None),
]


def read_sto(path):
    seqs = {}
    for line in gzip.open(path, "rt", errors="ignore"):
        if line.startswith(("#", "//")) or not line.strip():
            continue
        name, s = line.split()[:2]
        seqs[name] = seqs.get(name, "") + s
    return [s.upper() for s in seqs.values()]


def pair_counts(data_dir, pairs_per_family=30, min_pairs=30, seed=0):
    rng = random.Random(seed)
    counts = []
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".sto.gz"):
            continue
        S = read_sto(os.path.join(data_dir, f))
        prs = [(i, j) for i in range(len(S)) for j in range(i + 1, len(S))]
        rng.shuffle(prs)
        for i, j in prs[:pairs_per_family]:
            N = np.zeros((20, 20))
            for a, b in zip(S[i], S[j]):
                if a in IDX and b in IDX:
                    N[IDX[a], IDX[b]] += 1
            if N.sum() >= min_pairs:
                counts.append(N)
    return np.array(counts)


def joint_grid(arrs, ts):
    return np.stack([np.asarray(F._joint(arrs, t)) for t in ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "pfam_seed_sample"))
    ap.add_argument("--out", default=str(ROOT / "results" / "pfam_pair_fit" / "summary.tsv"))
    args = ap.parse_args()
    counts = pair_counts(args.data_dir)
    ident = np.trace(counts, axis1=1, axis2=2) / counts.sum((1, 2))
    n_res = counts.sum()
    print(f"{len(counts)} pairs, {int(n_res)} aligned residue pairs, median identity {np.median(ident):.3f}")
    ts = np.exp(np.linspace(np.log(0.01), np.log(1000), 240))
    rows = []
    hdr = f"{'model':<14}{'LL':>9}{'assoc':>8}{'med t':>8}{'t>=10':>7}{'P(id) inf':>10}"
    print(hdr)
    for kind, g in MODELS:
        arrs = F._sub_arrays(F.load_subst(kind, g))
        J = joint_grid(arrs, ts)
        LL = np.einsum("pab,tab->pt", counts, np.log(np.maximum(J, 1e-300)))
        best, tb = LL.max(1), ts[LL.argmax(1)]
        pib = J[0].sum(1)
        indep = np.einsum("pab,ab->p", counts, np.log(np.outer(pib, pib)))
        name = kind + (f"+G{g:g}" if g else "")
        r = (name, best.sum() / n_res, (best - indep).sum() / n_res, np.median(tb),
             (tb >= 10).mean(), np.trace(J[-1]))
        rows.append(r)
        print(f"{r[0]:<14}{r[1]:>9.4f}{r[2]:>8.4f}{r[3]:>8.2f}{r[4]:>7.2f}{r[5]:>10.3f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("model\tLL_per_pair\tassoc_per_pair\tmedian_t\tfrac_t_ge_10\tP_ident_inf\n")
        for r in rows:
            f.write("\t".join([r[0]] + [f"{x:.5f}" for x in r[1:]]) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
