#!/usr/bin/env python3
"""§10 figure: SP and TC vs K (mixture components) for the FSA pipeline,
on BAliBASE 3 and OXBench, with MAFFT + MUSCLE baselines for BAliBASE.

Inputs (already collected):
  experiments/balibase_tkf92.json (K=1)
  experiments/balibase_tkf92_mix20_a{1,2,3}.json (K=20 replicates)
  experiments/balibase_mafft_muscle_120.json (MAFFT + MUSCLE)
  experiments/oxbench_tkf92.json (K=1)
  experiments/oxbench_tkf92_mix20.json (K=20; if present)

Saves to experiments/figures/sec10_sp_tc_vs_K.pdf.
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_summary(path, key=None):
    if not os.path.isfile(path):
        return None
    d = json.load(open(path))
    if key:
        rs = d.get(key, [])
    else:
        rs = d.get('results', [])
    sps = [r.get('sp', np.nan) for r in rs if 'sp' in r]
    tcs = [r.get('tc', np.nan) for r in rs if 'tc' in r]
    sps = np.array([s for s in sps if not np.isnan(s)])
    tcs = np.array([t for t in tcs if not np.isnan(t)])
    if len(sps) == 0:
        return None
    return {
        'n': len(sps),
        'sp_mean': float(sps.mean()), 'sp_med': float(np.median(sps)),
        'sp_std': float(sps.std()),
        'tc_mean': float(tcs.mean()), 'tc_med': float(np.median(tcs)),
        'tc_std': float(tcs.std()),
    }


def main():
    base = '/home/yam/tkf-mixdom/python/experiments'

    # BAliBASE TKF92 K=1
    bali_K1 = load_summary(f'{base}/balibase_tkf92.json')

    # BAliBASE K=20 replicates
    bali_K20 = []
    for rep in ('a1', 'a2', 'a3'):
        s = load_summary(f'{base}/balibase_tkf92_mix20_{rep}.json')
        if s is not None:
            bali_K20.append(s)

    # MAFFT + MUSCLE baselines (live in a different JSON)
    bali_mm = json.load(open(f'{base}/balibase_mafft_muscle_120.json'))
    bali_mafft = load_summary(f'{base}/balibase_mafft_muscle_120.json',
                                key='mafft_results')
    bali_muscle = load_summary(f'{base}/balibase_mafft_muscle_120.json',
                                 key='muscle_results')

    # OxBench
    ox_K1 = load_summary(f'{base}/oxbench_tkf92.json')
    ox_K20 = load_summary(f'{base}/oxbench_tkf92_mix20.json')

    print(f'BAliBASE: K=1 SP={bali_K1["sp_mean"]:.3f}  '
          f'K=20 reps SP={[round(r["sp_mean"], 3) for r in bali_K20]}  '
          f'MAFFT={bali_mafft["sp_mean"] if bali_mafft else None}  '
          f'MUSCLE={bali_muscle["sp_mean"] if bali_muscle else None}')
    print(f'OxBench:  K=1 SP={ox_K1["sp_mean"] if ox_K1 else None}  '
          f'K=20 SP={ox_K20["sp_mean"] if ox_K20 else None}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Use Ks 1 and 20 as integers; jitter K=20 reps slightly for readability.
    for ax, metric, ylabel in [
        (axes[0], 'sp', 'Sum-of-Pairs (SP)'),
        (axes[1], 'tc', 'Total-Column (TC)'),
    ]:
        # BAliBASE K=1
        ax.errorbar([1], [bali_K1[f'{metric}_mean']], yerr=[bali_K1[f'{metric}_std']],
                     fmt='o', c='tab:blue', ms=8, label='BAliBASE TKF92 (K=1)',
                     capsize=4)
        # BAliBASE K=20 reps
        for i, r in enumerate(bali_K20):
            ax.errorbar([20 + (i - 1) * 0.6], [r[f'{metric}_mean']],
                          yerr=[r[f'{metric}_std']],
                          fmt='s', c='tab:blue', ms=6, alpha=0.6,
                          capsize=3, mfc='none')
        ax.errorbar([20], [np.mean([r[f'{metric}_mean'] for r in bali_K20])],
                     fmt='D', c='tab:blue', ms=10,
                     label='BAliBASE TKF92 (K=20, mean of 3)', capsize=0)
        # MAFFT / MUSCLE baselines as horizontal lines
        if bali_mafft:
            ax.axhline(bali_mafft[f'{metric}_mean'], ls='--', c='tab:gray',
                        label=f'BAliBASE MAFFT L-INS-i (n={bali_mafft["n"]})')
        if bali_muscle:
            ax.axhline(bali_muscle[f'{metric}_mean'], ls=':', c='black',
                        label=f'BAliBASE MUSCLE 5 (n={bali_muscle["n"]})')
        # OxBench
        if ox_K1:
            ax.errorbar([1], [ox_K1[f'{metric}_mean']],
                          yerr=[ox_K1[f'{metric}_std']],
                          fmt='o', c='tab:orange', ms=8,
                          label=f'OxBench TKF92 (K=1, n={ox_K1["n"]})',
                          capsize=4)
        if ox_K20:
            ax.errorbar([20], [ox_K20[f'{metric}_mean']],
                          yerr=[ox_K20[f'{metric}_std']],
                          fmt='D', c='tab:orange', ms=8,
                          label=f'OxBench TKF92 (K=20, n={ox_K20["n"]})',
                          capsize=4)

        ax.set_xlabel('K (mixture components)')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 8, 20])
        ax.set_xticklabels(['1', '2', '4', '8', '20'])
        ax.set_title(ylabel + ' vs K')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower left' if metric == 'sp' else 'upper left',
                   fontsize=7)
    plt.tight_layout()
    out = 'experiments/figures/sec10_sp_tc_vs_K.pdf'
    plt.savefig(out)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
