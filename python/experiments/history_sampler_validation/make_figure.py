#!/usr/bin/env python3
"""Figure + Markdown summary for the Algorithm 1 validation (H1/H2/H3).

Panels:
  (a) exact ghost-count pmf from the bivariate PGF vs Algorithm 1 histograms
      (H3), a few blocks;
  (b) all two-sample / goodness-of-fit p-values from H1, H2, H3 pooled
      (uniform on [0,1] if the sampler is exact);
  (c) paired z-scores of the 12 history statistics in H1 across regimes
      (standard normal if exact).

Usage: make_figure.py --dir experiments/history_sampler_validation
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm

STAT_LABEL = {"B": "births $B$", "D": "deaths $D$", "ghosts": "ghosts", "S": "exposure $S$",
              "n_mid": "links alive at $t/2$", "depth": "max depth", "imm_children": "immortal's children",
              "ghost_life": "ghost lifetime", "first_birth": "first birth time", "mean_birth": "mean birth time",
              "mean_death": "mean death time", "last_death": "last death time"}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    D = args.dir
    h1 = load(os.path.join(D, "results_h1.json"))["h1"]
    h2 = load(os.path.join(D, "results_h2.json"))["h2"]
    h2m = load(os.path.join(D, "results_h2_mid.json"))
    h2m = h2m["h2"] if h2m else {}
    h3 = load(os.path.join(D, "results_h3.json"))["h3"]
    extra = [load(p)["h1"] for p in sorted(glob.glob(os.path.join(D, "results_h1_long_n60k*.json")))]

    # ---- pooled p-values and paired z ----
    p1 = [v["p"] for r in h1.values() for v in r["features"].values() if v["test"] != "degenerate"]
    z1 = [v["paired_z"] for r in h1.values() for v in r["features"].values()]
    p2 = [v["p"] for src in (h2, h2m) for r in src.values() for tg in r["targets"].values()
          for v in tg["features"].values() if v["test"] != "degenerate"]
    p3 = [b["p"] for r in h3.values() for b in r["blocks"].values()]
    allp = np.array(p1 + p2 + p3)
    n2 = sum(len(r["targets"]) for src in (h2, h2m) for r in src.values())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    # (a) exact pmf vs sampled, H3
    ax = axes[0]
    # regime labels in the paper's notation (validate_history_sampler.REGIMES gives the rates)
    LABEL = {"kappa<1": r"$\lambda<\mu$", "kappa>1": r"$\lambda>\mu$", "equal": r"$\lambda=\mu$",
             "fast": r"$\lambda=1.5$, $\mu=2$"}
    picks = [("kappa<1", "immortal/alive root, n=2"), ("kappa>1", "deleted root, n=1"),
             ("equal", "deleted root, n=3"), ("fast", "immortal/alive root, n=0")]
    for i, (reg, key) in enumerate(picks):
        b = h3[reg]["blocks"][key]
        pe, ps = np.array(b["pmf_exact"][:8]), np.array(b["pmf_sampled"][:8])
        x = np.arange(len(pe)) + (i - 1.5) * 0.18
        ax.bar(x, pe, width=0.17, color=f"C{i}", alpha=0.45,
               label=f"{LABEL[reg]}, {key.replace('immortal/alive root', 'alive').replace('deleted root', 'deleted')}")
        ax.plot(x, ps, "o", color=f"C{i}", ms=3)
    ax.set_xlabel("number of ghosts in the block")
    ax.set_ylabel("probability")
    ax.set_title("(a) exact pmf (bars) vs Algorithm 1 (dots)", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False)
    # (b) pooled p-values
    ax = axes[1]
    ax.hist(allp, bins=10, range=(0, 1), color="0.6", edgecolor="k")
    ax.axhline(len(allp) / 10, color="C3", ls="--", lw=1)
    ksp = kstest(allp, "uniform").pvalue
    ax.set_xlabel("p-value")
    ax.set_ylabel("number of tests")
    ax.set_title(f"(b) {len(allp)} tests; KS vs uniform p={ksp:.2f}", fontsize=10)
    # (c) paired z
    ax = axes[2]
    z = np.array(z1)
    ax.hist(z, bins=np.linspace(-4, 4, 9), density=True, color="0.6", edgecolor="k")
    xx = np.linspace(-4, 4, 200)
    ax.plot(xx, norm.pdf(xx), color="C3", lw=1)
    ax.set_xlabel("paired z-score (forward $-$ Algorithm 1)")
    ax.set_ylabel("density")
    ax.set_title(f"(c) {len(z)} paired statistics", fontsize=10)
    fig.tight_layout()
    out = os.path.join(D, "history_sampler_validation.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)

    # ---- markdown summary ----
    L = ["# Algorithm 1 validation summary", ""]
    L.append(f"Pooled: {len(allp)} tests, {int((allp < 0.01).sum())} below p=0.01, "
             f"{int((allp < 0.05).sum())} below 0.05, min p={allp.min():.4f}; KS vs uniform p={ksp:.3f}.")
    L.append(f"H1 paired z-scores: {len(z)} statistics, max |z|={np.abs(z).max():.2f}, "
             f"mean={z.mean():+.2f}, sd={z.std():.2f}.")
    L.append("")
    L.append("## H1 successive-conditional test (whole branches, 4 ancestral links)")
    L.append("| regime | λ | μ | t | n | distinct alignments | min p (12 tests) | max paired z |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, r in h1.items():
        L.append(f"| {name} | {r['lam']} | {r['mu']} | {r['t']} | {r['n']} | {r['n_distinct_alignments']} | "
                 f"{r['summary']['min_p']:.3f} | {r['max_abs_paired_z']:.2f} |")
    for e in extra:
        for name, r in e.items():
            L.append(f"| {name} (rerun, grid {load(os.path.join(D, 'results_h1_long_n60k.json'))['args']['grid'] if r['n']==60000 else '?'}) "
                     f"| {r['lam']} | {r['mu']} | {r['t']} | {r['n']} | {r['n_distinct_alignments']} | "
                     f"{r['summary']['min_p']:.3f} | {r['max_abs_paired_z']:.2f} |")
    L.append("")
    L.append("## H2 exact conditional by rejection (whole alignments, 3 ancestral links)")
    L.append("| regime | alignment | accepted | Algorithm 1 draws | min p (12 tests) | tests below 0.01 |")
    L.append("|---|---|---|---|---|---|")
    for src in (h2, h2m):
        for name, r in src.items():
            for p, tg in r["targets"].items():
                L.append(f"| {name} | {p} | {tg['n_ref']} | {tg['n_alg']} | {tg['summary']['min_p']:.3f} | "
                         f"{tg['summary']['n_p_below_0_01']} |")
    L.append("")
    L.append("## H3 exact ghost-count pmf per block (bivariate PGF)")
    L.append("| regime | block | chi² p | bins | E[ghosts] exact | sampled |")
    L.append("|---|---|---|---|---|---|")
    for name, r in h3.items():
        for key, b in r["blocks"].items():
            L.append(f"| {name} | {key} | {b['p']:.3f} | {b['n_bins']} | {b['mean_exact']:.4f} | "
                     f"{b['mean_sampled']:.4f} ± {b['se_sampled']:.4f} |")
    open(os.path.join(D, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:4]))
    print(f"H2 targets: {n2}; figure: {out}")


if __name__ == "__main__":
    main()
