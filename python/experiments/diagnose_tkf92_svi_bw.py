#!/usr/bin/env python3
"""Diagnostic: pure-TKF92 SVI-BW E-step + EM convergence on Gillespie data.

Two checks:
  1. FB E-step counts (B, D, S, ext_count, notext_count) at TRUTH params
     should match the Gillespie labeled ground-truth aggregates.
  2. Full-batch EM from a far init should converge to truth.

Uses ``gillespie_pair_tkf92`` (in mixdom_gillespie.py) for ground truth;
empty pairs are KEPT — they encode the (1-β)(1-κ) chi-end mass.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.simulate.mixdom_gillespie import gillespie_pair_tkf92
from tkfmixdom.jax.core.ctmc import rate_matrix_jc69
from tkfmixdom.jax.train.tkf92_svi_bw import (
    estep_pair_tkf92, m_step_lam_mu, m_step_ext,
)


def _empty_suff():
    return {'B': 0.0, 'D': 0.0, 'S': 0.0, 'L': 0.0, 'M': 0.0, 'T': 0.0,
            'ext_count': 0.0, 'notext_count': 0.0}


def main():
    Q, pi = rate_matrix_jc69(20)
    pi_np = np.asarray(pi)
    true_lam, true_mu, true_ext, t = 0.05, 0.07, 0.3, 0.5
    n_pairs_target = 300

    print(f'Truth: lam={true_lam}, mu={true_mu}, ext={true_ext}, t={t}')
    print(f'Generating {n_pairs_target} Gillespie pairs (empties kept)...',
          flush=True)
    rng = np.random.RandomState(42)
    pairs = []
    labeled_total = {'B': 0, 'D': 0, 'S': 0.0, 'ext': 0, 'notext': 0,
                       'subs': 0}
    n_empty = 0
    t0 = time.time()
    for k in range(n_pairs_target):
        out = gillespie_pair_tkf92(rng, true_lam, true_mu, true_ext, t,
                                      Q, pi_np, root_at_stationary=True)
        anc = out['anc_residues']
        desc = out['desc_residues']
        pairs.append((anc, desc, t))
        if len(anc) == 0 and len(desc) == 0:
            n_empty += 1
        L = out['labeled']
        labeled_total['B'] += L['n_fragment_births_per_lineage'] + L['n_fragment_imm']
        labeled_total['D'] += L['n_fragment_deaths']
        labeled_total['S'] += L['fragment_sojourn']
        labeled_total['ext'] += L['n_extensions']
        labeled_total['notext'] += L['n_new_fragment_decisions'] + L['n_fragment_imm']
        labeled_total['subs'] += L['n_substitutions']
    print(f'  generated in {time.time()-t0:.1f}s, '
          f'{len(pairs)} pairs ({n_empty} empty)', flush=True)
    mean_anc = float(np.mean([len(p[0]) for p in pairs]))
    mean_desc = float(np.mean([len(p[1]) for p in pairs]))
    print(f'  mean ancestor length = {mean_anc:.2f}, '
          f'mean descendant length = {mean_desc:.2f}', flush=True)
    print()
    print('Labeled (Gillespie ground truth, fragment-level):')
    print(f'  total fragment births (per-lineage + imm) = {labeled_total["B"]}')
    print(f'  total fragment deaths = {labeled_total["D"]}')
    print(f'  fragment sojourn      = {labeled_total["S"]:.2f}')
    print(f'  ext events            = {labeled_total["ext"]}')
    print(f'  notext events         = {labeled_total["notext"]}')
    print(f'  total subst events    = {labeled_total["subs"]}')

    # FB E-step at truth.
    print()
    print('FB E-step at truth params (should match labeled in expectation):',
          flush=True)
    fb_total = _empty_suff()
    t0 = time.time()
    for k, (x, y, t_pair) in enumerate(pairs):
        r = estep_pair_tkf92(x, y, t_pair, true_lam, true_mu, true_ext,
                                Q, pi_np)
        for kk in fb_total:
            fb_total[kk] += r[kk]
        if (k + 1) % 50 == 0:
            print(f'  pair {k+1}/{len(pairs)}, elapsed {time.time()-t0:.1f}s',
                  flush=True)
    print(f'  E[B] (FB) = {fb_total["B"]:7.2f}    vs labeled = '
          f'{labeled_total["B"]:>4} (note: labeled at FRAGMENT level; '
          f'FB at SITE level)')
    print(f'  E[D] (FB) = {fb_total["D"]:7.2f}    vs labeled = '
          f'{labeled_total["D"]:>4}')
    print(f'  E[S] (FB) = {fb_total["S"]:7.2f}    vs labeled = '
          f'{labeled_total["S"]:.2f}')
    print(f'  ext_count (FB) = {fb_total["ext_count"]:7.2f}    vs labeled = '
          f'{labeled_total["ext"]:>4}')
    print(f'  notext_count (FB) = {fb_total["notext_count"]:7.2f}    vs '
          f'labeled = {labeled_total["notext"]:>4}')

    # M-step at truth-derived suff stats: should give back truth.
    print()
    print('M-step on FB-derived suff stats at truth:')
    lam_at_truth, mu_at_truth = m_step_lam_mu(fb_total)
    ext_at_truth = m_step_ext(fb_total)
    print(f'  recovered lam = {lam_at_truth:.5f} (truth {true_lam}, '
          f'{(lam_at_truth-true_lam)/true_lam:+.2%})')
    print(f'  recovered mu  = {mu_at_truth:.5f} (truth {true_mu}, '
          f'{(mu_at_truth-true_mu)/true_mu:+.2%})')
    print(f'  recovered ext = {ext_at_truth:.4f} (truth {true_ext}, '
          f'diff {ext_at_truth-true_ext:+.4f})')

    # Full-batch EM from far init.
    print()
    print('Full-batch EM (init lam=0.08, mu=0.11, ext=0.5):')
    lam, mu, ext = 0.08, 0.11, 0.5
    for it in range(20):
        suff = _empty_suff()
        for x, y, t_pair in pairs:
            r = estep_pair_tkf92(x, y, t_pair, lam, mu, ext, Q, pi_np)
            for kk in suff:
                suff[kk] += r[kk]
        lam_new, mu_new = m_step_lam_mu(suff)
        ext_new = m_step_ext(suff)
        print(f'  iter {it+1:>2}: lam={lam_new:.5f} mu={mu_new:.5f} '
              f'ext={ext_new:.4f}', flush=True)
        lam, mu, ext = lam_new, mu_new, ext_new

    print()
    print(f'Final: lam={lam:.5f} (truth {true_lam}, '
          f'rel.err {(lam-true_lam)/true_lam:+.2%})')
    print(f'       mu={mu:.5f} (truth {true_mu}, '
          f'rel.err {(mu-true_mu)/true_mu:+.2%})')
    print(f'       ext={ext:.4f} (truth {true_ext}, '
          f'diff {ext-true_ext:+.4f})')


if __name__ == '__main__':
    main()
