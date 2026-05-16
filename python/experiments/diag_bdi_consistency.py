#!/usr/bin/env python3
"""Diagnostic: is per-(regime, j) deviation in bdi_consistency_D.pdf MC noise or bias?

Computes for each bin:
  - analytic E[D | i=1, j, lam, mu, t]      (current fig-script: FD-of-conservation)
  - alternative analytic E[D] via off-critical extrapolation (kappa = 1-1e-3)
  - sim mean E[D]                            (Gillespie average over bin)
  - per-bin sample std + standard error
  - z-score against (a) script analytic and (b) off-crit analytic
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from collections import defaultdict
from tkfmixdom.jax.simulate.simulate import simulate_bdi_gillespie
from fig_bdi_consistency import bdi_logprob, analytic_stats, REGIMES

N_SIMS = 1_000_000  # 1M per regime; enough for clear z-scores
MIN_COUNT = 100
MAX_J = 10


def ed_via_offcrit(lam, mu, t, j, i=1, eps_kappa=1e-3, eps_fd=1e-6):
    """Compute E[D] via off-critical extrapolation (kappa = 1 - eps_kappa).
    Works for both lam=mu and lam!=mu (the eps_kappa offset is tiny if lam!=mu)."""
    if abs(lam - mu) > 1e-6 * max(lam, mu):
        # General case — use the same formula as analytic_stats
        return analytic_stats(lam, mu, t, j, i)[1]
    lam_off = mu * (1 - eps_kappa)
    d_lam = (bdi_logprob(i, j, lam_off + eps_fd, mu, t) - bdi_logprob(i, j, lam_off - eps_fd, mu, t)) / (2 * eps_fd)
    d_mu = (bdi_logprob(i, j, lam_off, mu + eps_fd, t) - bdi_logprob(i, j, lam_off, mu - eps_fd, t)) / (2 * eps_fd)
    es = (j - i + mu * d_mu - lam_off * d_lam - lam_off * t) / (lam_off - mu)
    return mu * d_mu + mu * es


def run_regime(lam, mu, t, n_sims, seed):
    rng = np.random.RandomState(seed)
    by_j = defaultdict(lambda: {'D': []})
    for _ in range(n_sims):
        j, nb, nd, ni, soj = simulate_bdi_gillespie(rng, 1, lam, mu, t)
        by_j[j]['D'].append(nd)
    return by_j


def main():
    out = {'regimes': [], 'n_sims_per_regime': N_SIMS}
    for r_idx, (lam, mu, t, label, cat) in enumerate(REGIMES):
        t0 = time.time()
        seed = r_idx * 1000
        print(f"[regime {r_idx}] lam={lam},mu={mu},t={t}  N={N_SIMS}  seed={seed}")
        by_j = run_regime(lam, mu, t, N_SIMS, seed)
        rows = []
        for j in sorted(by_j.keys()):
            n_bin = len(by_j[j]['D'])
            if j > MAX_J or n_bin < MIN_COUNT:
                continue
            D = np.array(by_j[j]['D'])
            sim_mean = float(D.mean())
            sim_std = float(D.std(ddof=1))
            se = sim_std / np.sqrt(n_bin)

            ed_script = analytic_stats(lam, mu, t, j, i=1)[1]
            ed_off = ed_via_offcrit(lam, mu, t, j)

            z_script = (sim_mean - ed_script) / se if se > 0 else 0.0
            z_off = (sim_mean - ed_off) / se if se > 0 else 0.0

            row = {
                'j': int(j), 'n_bin': int(n_bin),
                'sim_mean_D': sim_mean, 'sim_std_D': sim_std, 'se_D': float(se),
                'ed_script': float(ed_script), 'ed_offcrit': float(ed_off),
                'z_script': float(z_script), 'z_offcrit': float(z_off),
                'residual_script': float(sim_mean - ed_script),
                'residual_offcrit': float(sim_mean - ed_off),
            }
            rows.append(row)
        elapsed = time.time() - t0
        max_z_s = max((abs(r['z_script']) for r in rows), default=0.0)
        max_z_o = max((abs(r['z_offcrit']) for r in rows), default=0.0)
        print(f"   {len(rows)} bins; max|z_script|={max_z_s:.2f}  max|z_offcrit|={max_z_o:.2f}  elapsed {elapsed:.1f}s")
        for r in rows:
            f1 = '!' if abs(r['z_script']) >= 3 else ' '
            f2 = '!' if abs(r['z_offcrit']) >= 3 else ' '
            print(f"     j={r['j']:2d}  n={r['n_bin']:7d}  sim={r['sim_mean_D']:+.5f}  "
                  f"script={r['ed_script']:+.5f} (z={r['z_script']:+6.2f}){f1}  "
                  f"off={r['ed_offcrit']:+.5f} (z={r['z_offcrit']:+6.2f}){f2}")
        out['regimes'].append({
            'lam': lam, 'mu': mu, 't': t, 'label': label,
            'category': cat, 'seed': seed, 'bins': rows,
        })

    outpath = os.path.join(os.path.dirname(__file__), 'figures', 'bdi_consistency_diag.json')
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {outpath}")


if __name__ == '__main__':
    main()
