#!/usr/bin/env python3
"""BDI / TKF sufficient-statistic validation figure for the MBE paper
("Closed-form Baum-Welch for TKF-based evolutionary models", Figure 2).

Three rows, each written as its own PDF (larger fonts than the original
panels from fig_bdi_consistency.py / fig_tkf_bdi_validation.py):

  bdi_consistency_parsimony.pdf
      Endpoint-conditioned E[B|i,j], E[D|i,j], E[S|i,j] (closed form, via
      fig_bdi_consistency.analytic_stats) against Gillespie means started
      from i = CONSISTENCY_I0 links, grouped by the final count j, for the eight regimes of
      math-paper/figures/fig_bdi_consistency_highmu.py.  Overlaid: the
      parsimonious estimates B = max(j-i,0), D = max(i-j,0) and the naive
      exposure t(i+j)/2 (sequence length changes at the branch midpoint).
  tkf_bdi_validation_means_large.pdf
      Unconditional moments (fig_tkf_bdi_validation.analytic_moments)
      against Gillespie means for that script's eight regimes.
  tkf_bdi_validation_shrinkage_large.pdf
      Re-plot of figures/tkf_bdi_validation_shrinkage.json (800 replicate
      seeds per n): mean relative error vs n, which should fall as 1/sqrt(n).

Simulation uses a vectorized lockstep Gillespie (many trajectories advanced
together in numpy), statistically identical to the per-trajectory
simulators of the original scripts but fast enough to run the published
sizes in a couple of minutes.  Results are cached as JSON next to the PDFs;
--replot redraws from the caches without simulating.

Usage (from python/):
    python3.12 experiments/fig_bdi_validation_mbe.py            # simulate + plot
    python3.12 experiments/fig_bdi_validation_mbe.py --replot   # plot only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

import fig_bdi_consistency as fbc            # noqa: E402  analytic_stats(lam, mu, t, j, i=1)
import fig_tkf_bdi_validation as ftv         # noqa: E402  analytic_moments, REGIMES

OUT_DIR = os.path.join(HERE, 'figures')
FONT = 15

# Regimes of math-paper/figures/fig_bdi_consistency_highmu.py (tkf-dp).
CONSISTENCY_REGIMES = [
    (0.01, 0.03, 1.0, 'kappa_lt_1'),
    (0.05, 0.10, 2.0, 'kappa_lt_1'),
    (0.02, 0.02, 1.0, 'kappa_eq_1'),
    (0.05, 0.05, 2.0, 'kappa_eq_1'),
    (0.03, 0.01, 1.0, 'kappa_gt_1'),
    (0.50, 0.40, 3.0, 'kappa_gt_1'),
    (0.40, 0.40, 3.0, 'kappa_eq_1'),
    (0.30, 0.60, 2.5, 'kappa_lt_1'),
]
CONSISTENCY_N = {7: 50_000_000}   # regime 7 needs more for its j >= 10 tail
CONSISTENCY_N_DEFAULT = 10_000_000
MIN_COUNT, MAX_J = 100, 20
CONSISTENCY_I0 = 1   # starting links. With i=1 parsimony can count at most one death,
                     # whereas E[D|i,j] also counts transiently alive lineages.
MEANS_N = 20_000_000
CHUNK = 2_000_000

REGIME_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '>']
# One colour per regime (Okabe-Ito palette plus grey), so each regime's
# points can be identified; lambda < mu, = mu, > mu is given in the legend.
REGIME_COLORS = ['#0072B2', '#56B4E9', '#E69F00', '#999933', '#009E73',
                 '#D55E00', '#CC79A7', '#555555']
CAT_COLORS = {'kappa_lt_1': '#2176AE', 'kappa_eq_1': '#D4A029', 'kappa_gt_1': '#D32F2F'}


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate_bdi(lam, mu, t, n, rng, i0=1):
    """n trajectories of the TKF linear BDI process (per-link birth rate lam,
    death rate mu, plus immigration at rate lam from the immortal link),
    started from i0 links and run for time t.  Returns (X(t), B, D, S),
    with B counting all births including immigrant ones and S = int X."""
    x = np.full(n, i0, np.int64)
    tt = np.zeros(n)
    B = np.zeros(n, np.int64)
    D = np.zeros(n, np.int64)
    S = np.zeros(n)
    active = np.ones(n, bool)
    while active.any():
        idx = np.nonzero(active)[0]
        xi = x[idx]
        rb = lam * (xi + 1)
        rd = mu * xi
        r = rb + rd
        dt = rng.exponential(1.0 / r)
        t_new = tt[idx] + dt
        over = t_new >= t
        S[idx] += xi * np.where(over, t - tt[idx], dt)
        go = idx[~over]
        birth = rng.uniform(size=go.size) < rb[~over] / r[~over]
        x[go] += np.where(birth, 1, -1)
        B[go] += birth
        D[go] += ~birth
        tt[go] = t_new[~over]
        active[idx[over]] = False
    return x, B, D, S


def simulate_chunked(lam, mu, t, n, rng, i0=1):
    parts = []
    done = 0
    while done < n:
        m = min(CHUNK, n - done)
        parts.append(simulate_bdi(lam, mu, t, m, rng, i0))
        done += m
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))


def run_consistency():
    rows = []
    for r, (lam, mu, t, cat) in enumerate(CONSISTENCY_REGIMES):
        n = CONSISTENCY_N.get(r, CONSISTENCY_N_DEFAULT)
        t0 = time.time()
        x, B, D, S = simulate_chunked(lam, mu, t, n, np.random.default_rng(100 + r),
                                      i0=CONSISTENCY_I0)
        for j in range(MAX_J + 1):
            m = x == j
            nj = int(m.sum())
            if nj < MIN_COUNT:
                continue
            res = fbc.analytic_stats(lam, mu, t, j, i=CONSISTENCY_I0)
            if res is None:
                continue
            sims = [B[m], D[m], S[m]]
            rows.append(dict(
                regime=r, lam=lam, mu=mu, t=t, cat=cat, i=CONSISTENCY_I0, j=j, n=nj,
                analytic=[float(v) for v in res],
                sim_mean=[float(s.mean()) for s in sims],
                sim_se=[float(s.std() / np.sqrt(nj)) for s in sims],
                parsimony=[max(j - CONSISTENCY_I0, 0), max(CONSISTENCY_I0 - j, 0),
                           t * (CONSISTENCY_I0 + j) / 2.0]))
        print(f'consistency regime {r} (n={n:,}) done in {time.time() - t0:.1f}s', flush=True)
    return rows


def run_means():
    rows = []
    for r, (i, lam, mu, t, label) in enumerate(ftv.REGIMES):
        t0 = time.time()
        x, B, D, S = simulate_chunked(lam, mu, t, MEANS_N, np.random.default_rng(200 + r), i0=i)
        e_S, e_B, e_D = ftv.analytic_moments(i, lam, mu, t)
        sims = [B, D, S]
        rows.append(dict(
            regime=r, label=label, i=i, lam=lam, mu=mu, t=t, n=MEANS_N,
            analytic=[e_B, e_D, e_S],
            sim_mean=[float(s.mean()) for s in sims],
            sim_se=[float(s.std() / np.sqrt(MEANS_N)) for s in sims]))
        print(f'means regime {r} done in {time.time() - t0:.1f}s', flush=True)
    return rows


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _square_axes(ax, vals, pad_frac=0.05):
    lo, hi = min(vals), max(vals)
    pad = pad_frac * (hi - lo) + 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1, alpha=0.4, zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect('equal')


def plot_consistency(rows, out):
    names = [('births', 'B'), ('deaths', 'D'), ('exposure', 'S')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for k, ax in enumerate(axes):
        vals = []
        for rw in rows:
            c = REGIME_COLORS[rw['regime']]
            a, s, p = rw['analytic'][k], rw['sim_mean'][k], rw['parsimony'][k]
            ax.plot([a, a], [p, s], color=c, lw=0.6, alpha=0.35, zorder=1)
            mk = REGIME_MARKERS[rw['regime']]
            ax.errorbar(a, s, yerr=rw['sim_se'][k], fmt=mk, color=c, ms=5,
                        mec='white', mew=0.4, zorder=3)
            ax.plot(a, p, mk, mfc='none', mec=c, ms=6, mew=1.1, zorder=2)
            vals += [a, s, p]
        _square_axes(ax, vals)
        word, sym = names[k]
        ax.set_title(f'({"abc"[k]}) {word} ${sym}$')
        ax.set_xlabel(rf'closed-form $\mathbb{{E}}[{sym}\mid i,j]$')
        if k == 0:
            ax.set_ylabel('simulation mean or parsimony')
    handles = [Line2D([], [], marker=REGIME_MARKERS[r], ls='', color=REGIME_COLORS[r], ms=7,
                      label=rf'$\lambda{{=}}{lam:g},\ \mu{{=}}{mu:g},\ t{{=}}{t:g}$')
               for r, (lam, mu, t, cat) in enumerate(CONSISTENCY_REGIMES)]
    handles += [Line2D([], [], marker='o', ls='', color='0.4', label='filled: simulation mean'),
                Line2D([], [], marker='o', ls='', mfc='none', mec='0.4',
                       label='open: parsimony ($S$: midpoint)'),
                Line2D([], [], ls='--', color='k', alpha=0.4, label='$y=x$')]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               fontsize=FONT - 3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def plot_means(rows, out):
    syms = ['B', 'D', 'S']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    cmap = plt.get_cmap('tab10')
    for k, ax in enumerate(axes):
        vals = []
        for rw in rows:
            lhop = abs(rw['lam'] - rw['mu']) < 0.005
            ax.errorbar(rw['analytic'][k], rw['sim_mean'][k], yerr=rw['sim_se'][k],
                        fmt='D' if lhop else 'o', color=cmap(rw['regime']), ms=8,
                        mec='k', mew=0.6,
                        label=(rf"$i{{=}}{rw['i']},\ \lambda{{=}}{rw['lam']:g},"
                               rf"\ \mu{{=}}{rw['mu']:g},\ t{{=}}{rw['t']:g}$"))
            vals += [rw['analytic'][k], rw['sim_mean'][k]]
        lo, hi = min(vals), max(vals)
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1, alpha=0.4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f'({"def"[k]}) ' + rf'$\mathbb{{E}}[{syms[k]}]$')
        ax.set_xlabel(rf'closed-form $\mathbb{{E}}[{syms[k]}]$')
        if k == 0:
            ax.set_ylabel('simulation mean')
        ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=FONT - 3,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def plot_shrinkage(data, out):
    stats = [('B', 'sim_B', 'ana_B'), ('D', 'sim_D', 'ana_D'), ('S', 'sim_S', 'ana_S')]
    regs = [r['reg'] for r in data['test_regimes']]
    reg_label = {r['reg']: (rf"$i{{=}}{r['i']},\ \lambda{{=}}{r['lam']:g},"
                            rf"\ \mu{{=}}{r['mu']:g},\ t{{=}}{r['t']:g}$")
                 for r in data['test_regimes']}
    n_values = data['n_values']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for k, (sym, sk, ak) in enumerate(stats):
        ax = axes[k]
        firsts = []
        for reg in regs:
            means, ses = [], []
            for n in n_values:
                rel = np.array([abs(r[sk] - r[ak]) / abs(r[ak]) for r in data['records']
                                if r['regime'] == reg and r['n'] == n])
                means.append(rel.mean())
                ses.append(rel.std() / np.sqrt(rel.size))
            ax.errorbar(n_values, means, yerr=ses, fmt='o-', capsize=3, ms=5,
                        label=reg_label[reg])
            firsts.append(means[0])
        ref = np.array(n_values, float)
        ax.plot(ref, np.median(firsts) * np.sqrt(ref[0] / ref), 'k--', alpha=0.5,
                label=r'$\propto 1/\sqrt{n}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'({"ghi"[k]}) ' + rf'$\mathbb{{E}}[{sym}]$')
        ax.set_xlabel('number of simulations $n$')
        if k == 0:
            ax.set_ylabel('mean relative error')
        ax.grid(alpha=0.3, which='both')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--replot', action='store_true',
                    help='redraw from cached JSON without simulating')
    args = ap.parse_args()
    plt.rcParams.update({'font.size': FONT})
    os.makedirs(OUT_DIR, exist_ok=True)
    cons_json = os.path.join(OUT_DIR, 'bdi_consistency_parsimony.json')
    means_json = os.path.join(OUT_DIR, 'tkf_bdi_validation_means.json')
    if args.replot:
        cons = json.load(open(cons_json))
        means = json.load(open(means_json))
    else:
        cons = run_consistency()
        json.dump(cons, open(cons_json, 'w'), indent=1)
        means = run_means()
        json.dump(means, open(means_json, 'w'), indent=1)
    z = [(rw['sim_mean'][k] - rw['analytic'][k]) / rw['sim_se'][k]
         for rw in cons + means for k in range(3) if rw['sim_se'][k] > 0]
    print(f'simulation vs closed form: {len(z)} comparisons, max |z| = {max(map(abs, z)):.2f}')
    plot_consistency(cons, os.path.join(OUT_DIR, 'bdi_consistency_parsimony.pdf'))
    plot_means(means, os.path.join(OUT_DIR, 'tkf_bdi_validation_means_large.pdf'))
    plot_shrinkage(json.load(open(os.path.join(OUT_DIR, 'tkf_bdi_validation_shrinkage.json'))),
                   os.path.join(OUT_DIR, 'tkf_bdi_validation_shrinkage_large.pdf'))
    print('wrote PDFs to', OUT_DIR)


if __name__ == '__main__':
    main()
