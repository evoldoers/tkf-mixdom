#!/usr/bin/env python3
"""Summarize the per-site profile-mixture FSA runs on BAliBASE (fsa_c20_balibase.py).

Per method: mean SP / TC over families (sequence annealing at gap_factor 1, as
paper 1's table, and gap_factor 0), paired differences against a baseline on
the families both scored, a sign test and a paired bootstrap 95% interval.
Also reports the median per-pair substitution/indel time ratio, and with
``--diagnostics`` the BAliBASE reference pair identities, the fraction of
pairs whose substitution time sits at the clip, and the CherryML class rates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        rows[r["family"]] = r
    return rows


def paired(a, b, key, rng, n_boot=10000):
    fams = sorted(f for f in a if f in b and a[f].get(key) is not None
                  and b[f].get(key) is not None)
    d = np.array([a[f][key] - b[f][key] for f in fams])
    if len(d) == 0:
        return None
    boot = rng.choice(d, (n_boot, len(d))).mean(1)
    nz = d[d != 0]
    p = binomtest(int((nz > 0).sum()), len(nz)).pvalue if len(nz) else 1.0
    return dict(n=len(d), mean=d.mean(), lo=np.quantile(boot, 0.025),
                hi=np.quantile(boot, 0.975), wins=int((d > 0).sum()),
                losses=int((d < 0).sum()), p_sign=p)


def diagnostics(run_dir, balibase_dir):
    import os
    import sys
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "experiments"))
    import fsa_c20_balibase as F
    from tkfmixdom.util.msa_benchmark import parse_fasta
    ids = []
    ref = Path(balibase_dir).expanduser() / "ref"
    for f in sorted(os.listdir(ref)):
        s = [v.upper() for v in parse_fasta(str(ref / f)).values()]
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                m = [(x, y) for x, y in zip(s[i], s[j]) if x.isalpha() and y.isalpha()]
                if m:
                    ids.append(np.mean([x == y for x, y in m]))
    ids = np.array(ids)
    print(f"BAliBASE reference pairs: n={len(ids)}, identity over aligned residue pairs "
          f"median {np.median(ids):.3f}, quartiles {np.quantile(ids, .25):.3f}/{np.quantile(ids, .75):.3f}, "
          f"fraction < 0.25: {(ids < 0.25).mean():.2f}")
    for p in sorted(Path(run_dir).glob("*_two*.jsonl")):
        ts = np.array([pt[3] for line in p.read_text().splitlines()
                       for pt in json.loads(line)["pair_times"]])
        tmax = 1000.0 if "_T1000" in p.stem else 10.0
        print(f"  {p.stem:<28} pairs {len(ts):>5}  median t_sub {np.median(ts):7.2f}  "
              f"at clip ({tmax:g}): {(ts >= 0.999 * tmax).mean():.2f}")
    d = np.load(F.CHERRYML_NPZ, allow_pickle=True)
    pi21 = np.asarray(d["pi"], float)
    pi21 = pi21 / pi21.sum(1, keepdims=True)
    S = np.asarray(d["S"], float)[:, :20, :20].copy()
    S[:, np.arange(20), np.arange(20)] = 0
    pi = pi21[:, :20] / pi21[:, :20].sum(1, keepdims=True)
    rate = np.einsum("ka,kab,kb->k", pi, S, pi)
    w = np.asarray(d["weights"], float)
    k = int(np.argmax(w))
    print(f"CherryML C20 (20-state, gap dropped): class mean rates {rate.min():.2f}-{rate.max():.2f}, "
          f"weighted mean {w @ rate:.2f}; heaviest class w={w[k]:.3f}, gap mass {pi21[k, 20]:.2f}, "
          f"rate {rate[k]:.1f}; total weight on classes with gap mass > 0.5: {w[pi21[:, 20] > 0.5].sum():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/fsa_c20_balibase")
    ap.add_argument("--common", action="store_true",
                    help="restrict every mean to families finished by all methods")
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--balibase-dir", default="~/bio-datasets/data/balibase/bali3pdbm")
    args = ap.parse_args()
    if args.diagnostics:
        diagnostics(args.dir, args.balibase_dir)
        return
    rng = np.random.default_rng(0)
    runs = {p.stem: load(p) for p in sorted(Path(args.dir).glob("*.jsonl"))}
    common = set.intersection(*(set(r) for r in runs.values())) if runs else set()
    print(f"families finished by every method: {len(common)}")
    hdr = f"{'method':<22}{'n':>4}{'SP':>8}{'TC':>8}{'SP g0':>8}{'TC g0':>8}{'t_sub/t_ind':>13}"
    print(hdr)
    for name, rows in runs.items():
        fams = [f for f in rows if (f in common or not args.common)]
        m = lambda k: np.mean([rows[f][k] for f in fams if rows[f].get(k) is not None])
        ratios = [ts / ti for f in fams for (_, _, ti, ts, _) in rows[f].get("pair_times", [])
                  if ti > 0]
        print(f"{name:<22}{len(fams):>4}{m('msa_sp_g1'):>8.4f}{m('msa_tc_g1'):>8.4f}"
              f"{m('msa_sp_g0'):>8.4f}{m('msa_tc_g0'):>8.4f}{np.median(ratios):>13.3f}")
    for base_name in ("mixfrag_LG_one", "tkf92_LG_one"):
        if base_name not in runs:
            continue
        indel = base_name.split("_")[0]
        print(f"\npaired differences vs {base_name}  (mean [95% bootstrap], wins/losses, sign-test p)")
        for name, rows in runs.items():
            if name == base_name or not name.startswith(indel):
                continue
            for key in ("msa_sp_g1", "msa_tc_g1", "msa_sp_g0"):
                r = paired(rows, runs[base_name], key, rng)
                if r:
                    print(f"  {name:<22}{key:<10} n={r['n']:>3} {r['mean']:+.4f} "
                          f"[{r['lo']:+.4f},{r['hi']:+.4f}] {r['wins']:>3}/{r['losses']:<3} p={r['p_sign']:.3g}")


if __name__ == "__main__":
    main()
