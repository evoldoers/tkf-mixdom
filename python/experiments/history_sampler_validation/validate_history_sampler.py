#!/usr/bin/env python3
"""Does Algorithm 1 (alignment-conditioned indel-history sampler, MBE paper 1,
tkfmixdom/util/history_traceback.py) sample the exact posterior over
histories?  Three independent simulation checks, each stronger than the
per-block / first-moment checks already in tests/test_history_traceback.py:

H1  Successive-conditional (Geweke-style) test on whole branches.
    Forward-simulate (history, alignment) pairs from the TKF91 branching
    process; for each alignment draw a fresh history with Algorithm 1.  If
    the sampler is exact, (history', alignment) has the same joint law as
    (history, alignment), so every statistic of the history has the same
    marginal distribution in the two sets, and the paired differences have
    mean zero.  Covers all regimes and arbitrary alignments at no rejection
    cost; tests event counts, times, and tree shape jointly.

H2  Exact conditional via rejection on whole alignments.
    For a few short alignments, keep forward-simulated branches whose
    alignment equals the target: their histories are exact draws from the
    posterior.  Compare the distributions of counts, exposure, time-slice
    occupancy, tree depth and event times with Algorithm 1's draws
    (chi-square / two-sample KS).

H3  Exact ghost-count distribution per alignment block.
    The bivariate PGF (paper Eqs. 23-24) gives the joint law of (alive
    descendants, gravestones) in closed form.  Extracting its coefficients
    numerically (2-D Cauchy integral) gives the EXACT conditional pmf of the
    number of ghosts given the block's alive count n and the root's fate,
    with no simulation noise.  Compare Algorithm 1's ghost-count histograms
    to it with a chi-square goodness-of-fit test.

Usage (from ~/tkf-mixdom/python):
  .venv/bin/python -u experiments/history_sampler_validation/validate_history_sampler.py \
      --exp h1 h2 h3 --out experiments/history_sampler_validation/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import chi2, chi2_contingency, chisquare, ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PY)

from tkfmixdom.util.history_traceback import (          # noqa: E402
    HistorySampler, Link, blocks_from_path, expected_stats_from_path,
    summarize, window_coeffs,
)

REGIMES = {                       # name: (lam, mu, t)
    "kappa<1":   (0.7, 1.1, 1.0),
    "kappa>1":   (1.2, 0.6, 1.5),
    "equal":     (0.9, 0.9, 1.0),
    "long":      (0.3, 0.4, 3.0),
    "fast":      (1.5, 2.0, 0.5),
    # added for the scaled-up run (v2): more rate ratios, branch lengths and
    # ancestral lengths (N_ANC overrides the --n-anc default per regime)
    "short":     (0.7, 1.1, 0.3),
    "equal-long": (0.5, 0.5, 2.0),
    "kappa=2":   (2.0, 1.0, 0.8),
    "very-long": (0.2, 0.3, 5.0),
    "many-anc":  (1.0, 1.5, 1.0),
}
N_ANC = {"short": 8, "many-anc": 8, "very-long": 2, "kappa=2": 3}


# ---------------------------------------------------------------------------
# Forward simulation of one TKF91 branch (independent of the sampler code)
# ---------------------------------------------------------------------------

def sim_link(rng, lam, mu, s, tau, ancestral=False, immortal=False):
    ell = np.inf if immortal else rng.exponential(1.0 / mu)
    life = min(ell, s)
    us = np.sort(rng.uniform(0.0, life, rng.poisson(lam * life)))[::-1]
    node = Link(birth=tau, death=(tau + ell if ell < s else None),
                ancestral=ancestral, immortal=immortal)
    node.children = [sim_link(rng, lam, mu, s - u, tau + u) for u in us]
    return node


def sim_branch(rng, lam, mu, t, n_anc):
    hist = [sim_link(rng, lam, mu, t, 0.0, ancestral=True, immortal=True)]
    hist += [sim_link(rng, lam, mu, t, 0.0, ancestral=True) for _ in range(n_anc)]
    return hist


def path_of(history):
    """Pair-HMM path over {M,I,D} implied by a history."""
    out = []
    for root in history:
        if not root.immortal:
            out.append('M' if root.alive else 'D')
        for x in root.descendants_in_order():
            if x is root:
                continue
            if x.alive:
                out.append('I')
    return ''.join(out)


# ---------------------------------------------------------------------------
# History statistics
# ---------------------------------------------------------------------------

def features(history, t):
    s = summarize(history, t)
    births, deaths, depth_max, n_mid = [], [], 0, 0
    ghost_life = 0.0

    def walk(node, d):
        nonlocal depth_max, n_mid, ghost_life
        depth_max = max(depth_max, d)
        for c in node.children:
            births.append(c.birth)
            if c.death is not None:
                deaths.append(c.death)
                ghost_life += c.death - c.birth
            end = t if c.alive else c.death
            if c.birth <= 0.5 * t < end:
                n_mid += 1
            walk(c, d + 1)

    for root in history:
        if (not root.immortal) and (not root.alive):
            deaths.append(root.death)
        if not root.immortal and (root.alive or root.death > 0.5 * t):
            n_mid += 1
        walk(root, 0)
    return dict(
        B=s.births, D=s.deaths, ghosts=s.ghosts, S=s.exposure,
        n_mid=n_mid, depth=depth_max, imm_children=len(history[0].children),
        ghost_life=ghost_life,
        first_birth=min(births) if births else -1.0,
        mean_birth=float(np.mean(births)) if births else -1.0,
        mean_death=float(np.mean(deaths)) if deaths else -1.0,
        last_death=max(deaths) if deaths else -1.0,
    )


DISCRETE = ("B", "D", "ghosts", "n_mid", "depth", "imm_children")
CONTINUOUS = ("S", "ghost_life", "first_birth", "mean_birth", "mean_death", "last_death")


def compare(ref, alg, paired=False):
    """ref, alg: lists of feature dicts.  Returns per-feature test results."""
    out = {}
    for k in DISCRETE + CONTINUOUS:
        a = np.array([r[k] for r in ref], float)
        b = np.array([r[k] for r in alg], float)
        res = dict(mean_ref=float(a.mean()), mean_alg=float(b.mean()),
                   sd_ref=float(a.std()), sd_alg=float(b.std()))
        if k in DISCRETE:
            vals = sorted(set(a) | set(b))
            tab = np.array([[np.sum(a == v) for v in vals], [np.sum(b == v) for v in vals]])
            # merge sparse tail columns so every expected count is >= 5
            while tab.shape[1] > 1 and tab[:, -1].sum() < 10:
                tab[:, -2] += tab[:, -1]
                tab = tab[:, :-1]
            if tab.shape[1] > 1:
                res["test"] = "chi2"
                res["p"] = float(chi2_contingency(tab)[1])
                res["n_categories"] = int(tab.shape[1])
            else:
                res["test"] = "degenerate"
                res["p"] = 1.0
        else:
            res["test"] = "ks"
            res["p"] = float(ks_2samp(a, b).pvalue)
        if paired:
            d = a - b
            se = d.std(ddof=1) / np.sqrt(len(d))
            res["paired_z"] = float(d.mean() / se) if se > 0 else 0.0
        else:
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            res["z_means"] = float((a.mean() - b.mean()) / se) if se > 0 else 0.0
        out[k] = res
    return out


def pvals_summary(cmp):
    ps = [v["p"] for v in cmp.values() if v["test"] != "degenerate"]
    return dict(n_tests=len(ps), min_p=float(min(ps)), n_p_below_0_01=int(sum(p < 0.01 for p in ps)),
                # Fisher's combination is conservative under positive dependence; report anyway
                fisher_p=float(chi2.sf(-2 * np.sum(np.log(np.maximum(ps, 1e-300))), 2 * len(ps))))


# ---------------------------------------------------------------------------
# H0: sampler means vs the closed-form E-step (score-function identity)
# ---------------------------------------------------------------------------

H0_CASES = [  # (name, lam, mu, t, alignment path)
    ("kappa<1", 0.7, 1.1, 1.0, "IMMIDDMIIMDM"),
    ("kappa>1", 1.2, 0.6, 1.5, "IIMDMIMIIDM"),
    ("long",    0.4, 0.5, 2.0, "MIDMMIIDMDIM"),   # (the closed form of expected_stats_from_path divides by lam-mu, so lam=mu is excluded here; H1/H3 cover it)
]


def run_h0(args, log):
    results = {}
    for name, lam, mu, t, path in H0_CASES:
        EB, ED, ES = expected_stats_from_path(path, lam, mu, t)
        hs = HistorySampler(lam, mu, np.random.default_rng(6000 + hash(name) % 1000), grid=args.grid)
        blocks = blocks_from_path(path)
        t0 = time.time()
        xs = np.array([[s.births, s.deaths, s.exposure] for s in
                       (summarize(hs.sample_branch(blocks, t), t) for _ in range(args.n_h0))])
        mean, se = xs.mean(0), xs.std(0, ddof=1) / np.sqrt(len(xs))
        z = (mean - np.array([EB, ED, ES])) / se
        results[name] = dict(lam=lam, mu=mu, t=t, path=path, n=args.n_h0,
                             closed_form=dict(B=EB, D=ED, S=ES),
                             sampled_mean=dict(B=mean[0], D=mean[1], S=mean[2]),
                             se=dict(B=se[0], D=se[1], S=se[2]),
                             z=dict(B=z[0], D=z[1], S=z[2]), wall=time.time() - t0)
        log(f"H0 {name:8s} lam={lam} mu={mu} t={t} path={path}: n={args.n_h0}; "
            f"E[B] {EB:.4f} vs {mean[0]:.4f}±{se[0]:.4f} (z={z[0]:+.2f}); "
            f"E[D] {ED:.4f} vs {mean[1]:.4f}±{se[1]:.4f} (z={z[1]:+.2f}); "
            f"E[S] {ES:.4f} vs {mean[2]:.4f}±{se[2]:.4f} (z={z[2]:+.2f})")
    return results


# ---------------------------------------------------------------------------
# H1: successive-conditional test
# ---------------------------------------------------------------------------

def run_h1(args, log):
    results = {}
    for name, (lam, mu, t) in REGIMES.items():
        if args.regimes and name not in args.regimes:
            continue
        rng_f = np.random.default_rng(1000 + hash(name) % 1000 + args.seed_offset)
        hs = HistorySampler(lam, mu, np.random.default_rng(2000 + hash(name) % 1000 + args.seed_offset), grid=args.grid)
        t0 = time.time()
        ref, alg, paths = [], [], Counter()
        n_anc = N_ANC.get(name, args.n_anc)
        for _ in range(args.n_h1):
            h = sim_branch(rng_f, lam, mu, t, n_anc)
            p = path_of(h)
            paths[p] += 1
            ref.append(features(h, t))
            h2 = hs.sample_branch(blocks_from_path(p), t)
            assert path_of(h2) == p
            alg.append(features(h2, t))
        cmp = compare(ref, alg, paired=True)
        summ = pvals_summary(cmp)
        maxz = max(abs(v["paired_z"]) for v in cmp.values())
        results[name] = dict(lam=lam, mu=mu, t=t, n=args.n_h1, n_anc=n_anc,
                             n_distinct_alignments=len(paths),
                             mean_path_len=float(np.mean([len(p) for p in paths.elements()])),
                             features=cmp, summary=summ, max_abs_paired_z=float(maxz),
                             wall=time.time() - t0)
        log(f"H1 {name:8s} lam={lam} mu={mu} t={t}: n={args.n_h1}, "
            f"{len(paths)} distinct alignments; min p={summ['min_p']:.3f} "
            f"({summ['n_p_below_0_01']}/{summ['n_tests']} below 0.01); "
            f"max |paired z|={maxz:.2f}; {time.time()-t0:.0f}s")
        for k, v in cmp.items():
            log(f"    {k:12s} {v['test']:5s} p={v['p']:.3f}  mean fwd={v['mean_ref']:.4f} "
                f"alg={v['mean_alg']:.4f}  paired z={v['paired_z']:+.2f}")
    return results


# ---------------------------------------------------------------------------
# H2: exact conditional by rejection on whole alignments
# ---------------------------------------------------------------------------

def run_h2(args, log):
    results = {}
    for name in (args.regimes or ("kappa<1", "kappa>1", "equal")):
        lam, mu, t = REGIMES[name]
        n_anc_h2 = N_ANC.get(name, args.n_anc_h2) if args.h2_regime_n_anc else args.n_anc_h2
        rng_f = np.random.default_rng(3000 + hash(name) % 1000 + args.seed_offset)
        t0 = time.time()
        # Pilot: find frequent alignments with both insertions and deletions.
        pilot = Counter()
        store = defaultdict(list)
        for _ in range(args.n_h2_pilot):
            h = sim_branch(rng_f, lam, mu, t, n_anc_h2)
            p = path_of(h)
            pilot[p] += 1
            if 'I' in p and 'D' in p:      # superset of ok(); filtered below
                store[p].append(features(h, t))
        def ok(p):
            if args.h2_require_all:
                return all(c in p for c in 'MID') and len(p) >= args.h2_min_len
            return 'I' in p and 'D' in p
        cands = [p for p, c in pilot.most_common() if ok(p)]
        targets = cands[:args.n_targets]
        log(f"H2 {name}: pilot {args.n_h2_pilot} branches; targets {targets} "
            f"(pilot counts {[pilot[p] for p in targets]})")
        n_sim = args.n_h2_pilot
        while min(len(store[p]) for p in targets) < args.n_h2_ref and n_sim < args.n_h2_max:
            h = sim_branch(rng_f, lam, mu, t, n_anc_h2)
            n_sim += 1
            p = path_of(h)
            if p in targets and len(store[p]) < args.n_h2_ref:
                store[p].append(features(h, t))
        hs = HistorySampler(lam, mu, np.random.default_rng(4000 + hash(name) % 1000 + args.seed_offset), grid=args.grid)
        per_target = {}
        for p in targets:
            ref = store[p]
            blocks = blocks_from_path(p)
            alg = [features(hs.sample_branch(blocks, t), t) for _ in range(args.n_h2_alg)]
            cmp = compare(ref, alg, paired=False)
            summ = pvals_summary(cmp)
            per_target[p] = dict(n_ref=len(ref), n_alg=len(alg),
                                 acceptance=len(ref) / n_sim if len(ref) < args.n_h2_ref else pilot[p] / args.n_h2_pilot,
                                 features=cmp, summary=summ)
            log(f"  target {p}: n_ref={len(ref)} n_alg={len(alg)} min p={summ['min_p']:.3f} "
                f"({summ['n_p_below_0_01']}/{summ['n_tests']} below 0.01)")
            for k, v in cmp.items():
                log(f"    {k:12s} {v['test']:5s} p={v['p']:.3f}  mean rej={v['mean_ref']:.4f} "
                    f"alg={v['mean_alg']:.4f}  z={v['z_means']:+.2f}")
        results[name] = dict(lam=lam, mu=mu, t=t, n_anc=n_anc_h2, n_forward=n_sim,
                             targets=per_target, wall=time.time() - t0)
        log(f"H2 {name}: {n_sim} forward branches, {time.time()-t0:.0f}s")
    return results


# ---------------------------------------------------------------------------
# H3: exact ghost-count pmf per block from the bivariate PGF
# ---------------------------------------------------------------------------

def roots_delta(lam, mu, w):
    """z_+(w), z_-(w), Delta(w): roots of lam z^2 - (lam+mu) z + mu w = 0."""
    disc = np.sqrt((lam + mu) ** 2 - 4 * lam * mu * w + 0j)
    zp = ((lam + mu) + disc) / (2 * lam)
    zm = ((lam + mu) - disc) / (2 * lam)
    return zp, zm, disc


def I_D(lam, mu, t, T, z, w):
    """Paper Eq. 24: I_D(t,T; z,w) = e^{lam (z_+ - 1) T} (1 - rho_{t-T}) / (1 - rho_t)."""
    zp, zm, delta = roots_delta(lam, mu, w)
    rho = lambda s: (z - zp) / (z - zm) * np.exp(delta * s)
    return np.exp(lam * (zp - 1.0) * T) * (1.0 - rho(t - T)) / (1.0 - rho(t))


def block_pgf_on_torus(lam, mu, t, kind, K, rz, rw):
    """Values of the children-subtree PGF on the (z, w) torus, for a block
    whose root is alive ('A': window D(t,t)) or dies at a marginalised time
    ('G': integral over the death time of mu e^{-mu l} D(t,l))."""
    th = 2 * np.pi * np.arange(K) / K
    z = rz * np.exp(1j * th)[:, None]
    w = rw * np.exp(1j * th)[None, :]
    if kind == 'A':
        return I_D(lam, mu, t, t, z, w)
    # trapezoid rule over the root's death time, accumulated to bound memory
    ells = np.linspace(0.0, t, 1601)
    dl = ells[1] - ells[0]
    acc = np.zeros((K, K), complex)
    for i, l in enumerate(ells):
        wgt = dl * (0.5 if i in (0, len(ells) - 1) else 1.0)
        acc += wgt * mu * np.exp(-mu * l) * I_D(lam, mu, t, l, z, w)
    return acc


def coeffs_2d(F, nmax, gmax, rz, rw):
    """[z^n w^g] F for n<=nmax, g<=gmax from values on the K x K torus."""
    K = F.shape[0]
    C = np.fft.fft2(F) / (K * K)          # C[n, g] = mean F e^{-i(n th + g ph)}
    out = np.zeros((nmax + 1, gmax + 1))
    for n in range(nmax + 1):
        for g in range(gmax + 1):
            out[n, g] = (C[n, g] / (rz ** n * rw ** g)).real
    return out


def run_h3(args, log):
    results = {}
    # Contour radii near 1: dividing by r^g amplifies round-off by r^-g, so a
    # small radius is useless beyond g ~ 15; aliasing is bounded by r^K.
    K, rz, rw = 512, 0.8, 0.85
    gmax = 40
    for name in (args.regimes or ("kappa<1", "kappa>1", "equal", "fast")):
        lam, mu, t = REGIMES[name]
        hs = HistorySampler(lam, mu, np.random.default_rng(5000 + hash(name) % 1000 + args.seed_offset), grid=args.grid)
        per_block = {}
        for kind in ('A', 'G'):
            F = block_pgf_on_torus(lam, mu, t, kind, K, rz, rw)
            C = coeffs_2d(F, args.h3_nmax, gmax, rz, rw)
            # cross-check against the sampler's own alive-count coefficients at w=1
            if kind == 'A':
                ref = window_coeffs(lam, mu, t, t, args.h3_nmax)
                err = float(np.max(np.abs(C.sum(1) - ref)))
                log(f"H3 {name} window coefficient check (sum_g of 2-D coeffs vs "
                    f"window_coeffs): max abs err {err:.2e}")
            for n in range(0, args.h3_nmax + 1):
                pmf = C[n] / C[n].sum()
                if pmf.min() < -1e-8:
                    log(f"   warning: negative pmf entry {pmf.min():.2e}")
                pmf = np.maximum(pmf, 0.0); pmf /= pmf.sum()
                blocks = [('I', n)] if kind == 'A' else [('I', 0), ('D', n)]
                idx = 0 if kind == 'A' else 1
                g = np.array([summarize([hs.sample_branch(blocks, t)[idx]], t).ghosts
                              for _ in range(args.n_h3)])
                obs = np.bincount(g, minlength=gmax + 1)[:gmax + 1].astype(float)
                exp = pmf * args.n_h3
                # merge the tail so every expected count >= 5
                cut = gmax
                while cut > 0 and exp[cut:].sum() < 5:
                    cut -= 1
                o = np.concatenate([obs[:cut], [obs[cut:].sum()]])
                e = np.concatenate([exp[:cut], [exp[cut:].sum()]])
                p = float(chisquare(o, e).pvalue) if len(o) > 1 else 1.0
                mean_exact = float((np.arange(gmax + 1) * pmf).sum())
                key = f"{'immortal/alive' if kind=='A' else 'deleted'} root, n={n}"
                per_block[key] = dict(p=p, n_bins=int(len(o)), mean_exact=mean_exact,
                                      mean_sampled=float(g.mean()),
                                      se_sampled=float(g.std() / np.sqrt(len(g))),
                                      pmf_exact=pmf[:12].tolist(),
                                      pmf_sampled=(obs / obs.sum())[:12].tolist())
                log(f"H3 {name:8s} {key:28s} chi2 p={p:.3f} ({len(o)} bins); "
                    f"E[ghosts] exact={mean_exact:.4f} sampled={g.mean():.4f}"
                    f"±{g.std()/np.sqrt(len(g)):.4f}")
        ps = [v["p"] for v in per_block.values()]
        results[name] = dict(lam=lam, mu=mu, t=t, n=args.n_h3, blocks=per_block,
                             min_p=float(min(ps)), n_p_below_0_01=int(sum(p < 0.01 for p in ps)),
                             n_tests=len(ps))
    return results


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["h1", "h2", "h3"])
    ap.add_argument("--grid", type=int, default=2049)
    ap.add_argument("--n-h0", type=int, default=4000)
    ap.add_argument("--n-h1", type=int, default=20000)
    ap.add_argument("--n-anc", type=int, default=4)
    ap.add_argument("--n-anc-h2", type=int, default=3)
    ap.add_argument("--n-h2-pilot", type=int, default=200000)
    ap.add_argument("--n-h2-ref", type=int, default=5000)
    ap.add_argument("--n-h2-max", type=int, default=4000000)
    ap.add_argument("--n-h2-alg", type=int, default=20000)
    ap.add_argument("--n-targets", type=int, default=3)
    ap.add_argument("--h2-require-all", action="store_true",
                    help="H2 targets must contain M, I and D and have length >= --h2-min-len")
    ap.add_argument("--h2-min-len", type=int, default=5)
    ap.add_argument("--n-h3", type=int, default=20000)
    ap.add_argument("--regimes", nargs="*", default=None, help="restrict H1/H2/H3 to these regimes")
    ap.add_argument("--h2-regime-n-anc", action="store_true", help="H2: use the per-regime N_ANC table")
    ap.add_argument("--h3-nmax", type=int, default=3, help="H3: alive counts n=0..nmax per block")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def log(m):
        print(m, flush=True)

    log(f"args: {vars(args)}")
    res = dict(args=vars(args))
    for e in args.exp:
        res[e] = {"h0": run_h0, "h1": run_h1, "h2": run_h2, "h3": run_h3}[e](args, log)
        json.dump(res, open(args.out, "w"), indent=1, default=float)
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
