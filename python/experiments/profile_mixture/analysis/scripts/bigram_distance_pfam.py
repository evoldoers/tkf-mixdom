"""Compare the alignment-free bigram and trigram distances with maximum-likelihood distances on Pfam.

For random pairs of sequences in each Pfam seed alignment:
  * MLE of t under TKF92 + LG-C20, given the pair's alignment induced by the
    Pfam MSA (for F=1 the Pair HMM state path is the column-type sequence);
  * the bigram estimator (moment inversion of E[D2], see bigram_distance_check.py),
    with the null rho either from the model (C20 mean profile) or from the pair's
    own composition, rho_obs = sum_a f_X(a) f_Y(a);
  * the unigram estimator with the model null, for comparison. (With the
    pair's own composition as null the unigram excess is identically zero,
    since D1 = rho_obs L_X L_Y exactly, so that variant is not computed.)
TKF92 parameters are the Pfam-trained fit in
~/tkf-mixdom/python/params/best/cem_tkf92_pfamTrain.json.

Usage: python analysis/scripts/bigram_distance_pfam.py \
         --data-dir data/pfam_seed_sample --out results/bigram_distance_pfam
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bigram_distance_check import A, Subst, tkf91, counts, invert  # noqa: E402
from profile_classes_pfam import read_stockholm  # noqa: E402

PARAMS = Path(__file__).resolve().parents[4] / "params/best/cem_tkf92_pfamTrain.json"   # tkf-mixdom/python/params


def pair_columns(x, y):
    """Column types (1=M, 2=I, 3=D) and residues of the pairwise alignment of rows x, y."""
    keep = (x >= 0) | (y >= 0)
    x, y = x[keep], y[keep]
    types = np.where((x >= 0) & (y >= 0), 1, np.where(y >= 0, 2, 3))
    return types, x, y


def tkf92_loglik(t, types, x, y, lam, mu, r, sub):
    """log P(alignment, sequences | t) under TKF92 (F=1) with the C20 mixture per column."""
    _, _, tau = tkf91(lam, mu, t)            # rows S,M,I,D; cols M,I,D,E
    ll = 0.0
    prev = 0
    for c in types:
        p = tau[prev, c - 1]
        if prev != 0:
            p = (1 - r) * p + (r if prev == c else 0.0)
        ll += np.log(p)
        prev = c
    ll += np.log((1 - r) * tau[prev, 3] if prev != 0 else tau[0, 3])
    m = types == 1
    Mt = np.stack([sub.M(k, t) for k in range(sub.K)])          # (K,A,A)
    joint = np.einsum("k,ka,kab->ab", sub.w, sub.P, Mt)          # sum_k w_k pi^k_a M^k_ab
    ll += np.log(joint[x[m], y[m]]).sum()
    ins, dele = types == 2, types == 3
    ll += np.log(sub.pibar[y[ins]]).sum() + np.log(sub.pibar[x[dele]]).sum()
    return ll


def mle_t(types, x, y, lam, mu, r, sub):
    f = lambda lt: -tkf92_loglik(np.exp(lt), types, x, y, lam, mu, r, sub)
    res = minimize_scalar(f, bounds=(np.log(1e-3), np.log(10.0)), method="bounded",
                          options={"xatol": 1e-4})
    return float(np.exp(res.x))


class Composition:
    """Wraps Subst so that rho can be replaced by a pair-specific value."""

    def __init__(self, sub, rho):
        self.__dict__.update(sub.__dict__)
        self.rho = rho
        self.s = sub.s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/pfam_seed_sample")
    ap.add_argument("--out", default="results/bigram_distance_pfam")
    ap.add_argument("--pairs-per-family", type=int, default=20)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    prm = json.loads(PARAMS.read_text())
    lam, mu, r = prm["lam"], prm["mu"], prm["exts"][0]
    lbar = 1.0 / (1.0 - r)
    frag = (np.array([r]), np.array([1.0]))   # TKF92 = MixFrag with one fragtype
    sub = Subst(c20=True)
    print(f"TKF92 lam={lam:.5f} mu={mu:.5f} r={r:.4f} (mean fragment length {lbar:.3f}); "
          f"C20 rho={sub.rho:.4f}")
    recs = []
    for f in sorted(Path(args.data_dir).glob("*.sto.gz")):
        names, X = read_stockholm(f)
        lens = (X >= 0).sum(1)
        ok = np.where(lens >= args.min_len)[0]
        if len(ok) < 2:
            continue
        for _ in range(args.pairs_per_family):
            i, j = rng.choice(ok, 2, replace=False)
            types, x, y = pair_columns(X[i], X[j])
            xs, ys = X[i][X[i] >= 0], X[j][X[j] >= 0]
            t_ml = mle_t(types, x, y, lam, mu, r, sub)
            cx1, cx2 = counts(xs, 2); cy1, cy2 = counts(ys, 2)
            _, cx3 = counts(xs, 3); _, cy3 = counts(ys, 3)
            LX, LY = len(xs), len(ys)
            Lm = 0.5 * (LX + LY)
            d1, d2, d3 = cx1 @ cy1, cx2 @ cy2, cx3 @ cy3
            # scale the observed statistics to the symmetric length Lm
            d2s = d2 * (Lm - 1) ** 2 / ((LX - 1) * (LY - 1))
            d3s = d3 * (Lm - 2) ** 2 / ((LX - 2) * (LY - 2))
            d1s = d1 * Lm ** 2 / (LX * LY)
            rho_obs = float((cx1 / LX) @ (cy1 / LY))
            comp = Composition(sub, rho_obs)
            recs.append(dict(
                family=f.name.split(".")[0], LX=LX, LY=LY,
                pid=float(np.mean(x[types == 1] == y[types == 1])) if (types == 1).any() else 0.0,
                frac_match=float(np.mean(types == 1)),
                t_ml=t_ml,
                t_bigram=invert(d2s, Lm, Lm, lam, mu, frag, sub, 2),
                t_bigram_comp=invert(d2s, Lm, Lm, lam, mu, frag, comp, 2),
                t_trigram=invert(d3s, Lm, Lm, lam, mu, frag, sub, 3),
                t_trigram_comp=invert(d3s, Lm, Lm, lam, mu, frag, comp, 3),
                t_unigram=invert(d1s, Lm, Lm, lam, mu, frag, sub, 1)))
        print(f"{f.name.split('.')[0]}: {len(recs)} pairs so far", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(recs, indent=0))
    report(recs, out)


def report(recs, out):
    t = np.array([x["t_ml"] for x in recs])
    pid = np.array([x["pid"] for x in recs])
    lines = [f"pairs: {len(recs)} from {len({x['family'] for x in recs})} families; "
             f"MLE t median {np.median(t):.2f}, IQR [{np.percentile(t, 25):.2f}, {np.percentile(t, 75):.2f}]; "
             f"identity at matched columns median {np.median(pid):.2f}",
             "ratio = estimate / MLE, median [IQR], by MLE t bin; 'capped' = estimate at the search bound (20)"]
    bins = [(0, 1), (1, 2), (2, 4), (4, 10.01)]
    lines.append(f"{'estimator':>14} {'Spearman':>9} {'capped':>7} " +
                 " ".join(f"{f'{a}-{min(b, 10):g} (n={((t >= a) & (t < b)).sum()})':>20}" for a, b in bins))
    for k in ("t_trigram_comp", "t_bigram_comp", "t_trigram", "t_bigram", "t_unigram"):
        e = np.array([x[k] for x in recs])
        cells = []
        for a, b in bins:
            m = (t >= a) & (t < b)
            q = np.percentile(e[m] / t[m], [25, 50, 75])
            cells.append(f"{q[1]:>5.2f} [{q[0]:.2f},{q[2]:.2f}]".rjust(20))
        lines.append(f"{k:>14} {spearmanr(t, e)[0]:>9.3f} {np.mean(e > 19.9):>7.2f} " + " ".join(cells))
    txt = "\n".join(lines)
    print(txt)
    (out / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
